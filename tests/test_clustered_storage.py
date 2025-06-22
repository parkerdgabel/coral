"""Test clustered weight storage functionality in HDF5Store"""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.storage.hdf5_store import HDF5Store
from coral.delta.delta_encoder import Delta, DeltaType


class TestClusteredStorage:
    """Test suite for clustered weight storage in HDF5Store"""
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary HDF5 store for testing"""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name
        
        store = HDF5Store(filepath, mode="w")
        yield store
        
        # Cleanup
        store.close()
        if os.path.exists(filepath):
            os.unlink(filepath)
    
    @pytest.fixture
    def sample_weights(self):
        """Create sample weights for testing"""
        # Original weight
        weight1_data = np.random.randn(100, 50).astype(np.float32)
        weight1 = WeightTensor(
            data=weight1_data,
            metadata=WeightMetadata(
                name="layer1.weight",
                shape=(100, 50),
                dtype=np.float32,
                layer_type="Linear",
                model_name="test_model"
            )
        )
        
        # Similar weight (99% similar)
        weight2_data = weight1_data + 0.01 * np.random.randn(100, 50).astype(np.float32)
        weight2 = WeightTensor(
            data=weight2_data,
            metadata=WeightMetadata(
                name="layer1.weight.v2",
                shape=(100, 50),
                dtype=np.float32,
                layer_type="Linear",
                model_name="test_model_v2"
            )
        )
        
        # Centroid (mean of weight1 and weight2)
        centroid_data = (weight1_data + weight2_data) / 2
        centroid = WeightTensor(
            data=centroid_data,
            metadata=WeightMetadata(
                name="cluster_0_centroid",
                shape=(100, 50),
                dtype=np.float32,
                layer_type="Linear",
                model_name="centroid"
            )
        )
        
        return weight1, weight2, centroid
    
    def test_store_clustered_weight(self, temp_store, sample_weights):
        """Test storing a weight as clustered"""
        weight1, weight2, centroid = sample_weights
        
        # Store centroid
        centroid_hash = temp_store.store_centroid(centroid)
        assert centroid_hash is not None
        
        # Create a delta
        delta_data = weight2.data - centroid.data
        delta = Delta(
            delta_type=DeltaType.FLOAT32_RAW,
            data=delta_data,
            metadata={},
            original_shape=weight2.data.shape,
            original_dtype=weight2.data.dtype,
            reference_hash=centroid_hash,
            compression_ratio=2.0
        )
        
        # Store delta
        delta_hash = "delta_test_hash"
        temp_store.store_delta(delta, delta_hash)
        
        # Store weight as clustered
        weight_hash = weight2.compute_hash()
        storage_key = temp_store.store_clustered_weight(
            weight_hash=weight_hash,
            cluster_id="cluster_0",
            delta_hash=delta_hash,
            centroid_hash=centroid_hash,
            metadata=weight2.metadata
        )
        
        assert storage_key == weight_hash
        assert temp_store.is_weight_clustered(weight_hash)
        assert temp_store.exists(weight_hash)
    
    def test_load_clustered_weight(self, temp_store, sample_weights):
        """Test loading a clustered weight"""
        weight1, weight2, centroid = sample_weights
        
        # Store centroid
        centroid_hash = temp_store.store_centroid(centroid)
        
        # Create and store delta
        delta_data = weight2.data - centroid.data
        delta = Delta(
            delta_type=DeltaType.FLOAT32_RAW,
            data=delta_data,
            metadata={},
            original_shape=weight2.data.shape,
            original_dtype=weight2.data.dtype,
            reference_hash=centroid_hash,
            compression_ratio=2.0
        )
        delta_hash = "delta_test_hash"
        temp_store.store_delta(delta, delta_hash)
        
        # Store weight as clustered
        weight_hash = weight2.compute_hash()
        temp_store.store_clustered_weight(
            weight_hash=weight_hash,
            cluster_id="cluster_0",
            delta_hash=delta_hash,
            centroid_hash=centroid_hash,
            metadata=weight2.metadata
        )
        
        # Load the clustered weight
        loaded_weight = temp_store.load(weight_hash)
        
        assert loaded_weight is not None
        assert loaded_weight.metadata.name == weight2.metadata.name
        assert loaded_weight.metadata.shape == weight2.metadata.shape
        np.testing.assert_allclose(loaded_weight.data, weight2.data, rtol=1e-5)
    
    def test_get_clustered_weight_info(self, temp_store, sample_weights):
        """Test getting clustering information for a weight"""
        weight1, weight2, centroid = sample_weights
        
        # Store clustered weight
        centroid_hash = temp_store.store_centroid(centroid)
        delta_hash = "delta_test_hash"
        weight_hash = weight2.compute_hash()
        
        temp_store.store_clustered_weight(
            weight_hash=weight_hash,
            cluster_id="cluster_0",
            delta_hash=delta_hash,
            centroid_hash=centroid_hash,
            metadata=weight2.metadata
        )
        
        # Get clustering info
        info = temp_store.get_clustered_weight_info(weight_hash)
        
        assert info is not None
        assert info["cluster_id"] == "cluster_0"
        assert info["delta_hash"] == delta_hash
        assert info["centroid_hash"] == centroid_hash
        assert info["is_clustered"] is True
    
    def test_migrate_weight_to_clustered(self, temp_store, sample_weights):
        """Test migrating an existing weight to clustered storage"""
        weight1, weight2, centroid = sample_weights
        
        # First store weight normally
        weight_hash = temp_store.store(weight2)
        assert temp_store.exists(weight_hash)
        assert not temp_store.is_weight_clustered(weight_hash)
        
        # Store centroid and delta
        centroid_hash = temp_store.store_centroid(centroid)
        delta_data = weight2.data - centroid.data
        delta = Delta(
            delta_type=DeltaType.FLOAT32_RAW,
            data=delta_data,
            metadata={},
            original_shape=weight2.data.shape,
            original_dtype=weight2.data.dtype,
            reference_hash=centroid_hash,
            compression_ratio=2.0
        )
        delta_hash = "delta_test_hash"
        temp_store.store_delta(delta, delta_hash)
        
        # Migrate to clustered storage
        success = temp_store.migrate_weight_to_clustered(
            weight_hash=weight_hash,
            cluster_id="cluster_0",
            delta_hash=delta_hash,
            centroid_hash=centroid_hash,
            remove_original=True
        )
        
        assert success is True
        assert temp_store.is_weight_clustered(weight_hash)
        assert temp_store.exists(weight_hash)
        
        # Verify original data was removed
        with temp_store._file_operation() as f:
            assert weight_hash not in f["weights"]
            assert weight_hash in f["clustered_weights"]
        
        # Verify we can still load the weight
        loaded_weight = temp_store.load(weight_hash)
        assert loaded_weight is not None
        np.testing.assert_allclose(loaded_weight.data, weight2.data, rtol=1e-5)
    
    def test_list_clustered_weights(self, temp_store, sample_weights):
        """Test listing clustered weights"""
        weight1, weight2, centroid = sample_weights
        
        # Store multiple clustered weights
        centroid_hash = temp_store.store_centroid(centroid)
        
        weights_to_cluster = [weight1, weight2]
        clustered_hashes = []
        
        for i, weight in enumerate(weights_to_cluster):
            delta_data = weight.data - centroid.data
            delta = Delta(
                delta_type=DeltaType.FLOAT32_RAW,
                data=delta_data,
                metadata={},
                original_shape=weight.data.shape,
                original_dtype=weight.data.dtype,
                reference_hash=centroid_hash,
                compression_ratio=2.0
            )
            delta_hash = f"delta_{i}"
            temp_store.store_delta(delta, delta_hash)
            
            weight_hash = weight.compute_hash()
            temp_store.store_clustered_weight(
                weight_hash=weight_hash,
                cluster_id="cluster_0",
                delta_hash=delta_hash,
                centroid_hash=centroid_hash,
                metadata=weight.metadata
            )
            clustered_hashes.append(weight_hash)
        
        # List clustered weights
        clustered_list = temp_store.list_clustered_weights()
        
        assert len(clustered_list) == 2
        for hash_key in clustered_hashes:
            assert hash_key in clustered_list
    
    def test_get_clustered_storage_info(self, temp_store, sample_weights):
        """Test getting storage information for clustered weights"""
        weight1, weight2, centroid = sample_weights
        
        # Store clustered weight
        centroid_hash = temp_store.store_centroid(centroid)
        
        delta_data = weight2.data - centroid.data
        delta = Delta(
            delta_type=DeltaType.FLOAT32_RAW,
            data=delta_data,
            metadata={},
            original_shape=weight2.data.shape,
            original_dtype=weight2.data.dtype,
            reference_hash=centroid_hash,
            compression_ratio=2.0
        )
        delta_hash = "delta_test_hash"
        temp_store.store_delta(delta, delta_hash)
        
        weight_hash = weight2.compute_hash()
        temp_store.store_clustered_weight(
            weight_hash=weight_hash,
            cluster_id="cluster_0",
            delta_hash=delta_hash,
            centroid_hash=centroid_hash,
            metadata=weight2.metadata
        )
        
        # Get storage info
        info = temp_store.get_clustered_storage_info()
        
        assert info["total_clustered_weights"] == 1
        assert info["total_centroids"] == 1
        assert info["estimated_original_size"] > 0
        assert info["actual_clustered_size"] > 0
        # With only one weight, compression ratio might be < 1 due to overhead
        # Real compression benefits come when multiple weights share centroids
        assert info["compression_ratio"] > 0
        # Space savings might be negative with single weight due to overhead
        assert "space_savings_percent" in info
    
    def test_mixed_storage_list_weights(self, temp_store, sample_weights):
        """Test listing weights includes both regular and clustered weights"""
        weight1, weight2, centroid = sample_weights
        
        # Store one weight normally
        weight1_hash = temp_store.store(weight1)
        
        # Store another as clustered
        centroid_hash = temp_store.store_centroid(centroid)
        delta_data = weight2.data - centroid.data
        delta = Delta(
            delta_type=DeltaType.FLOAT32_RAW,
            data=delta_data,
            metadata={},
            original_shape=weight2.data.shape,
            original_dtype=weight2.data.dtype,
            reference_hash=centroid_hash,
            compression_ratio=2.0
        )
        delta_hash = "delta_test_hash"
        temp_store.store_delta(delta, delta_hash)
        
        weight2_hash = weight2.compute_hash()
        temp_store.store_clustered_weight(
            weight_hash=weight2_hash,
            cluster_id="cluster_0",
            delta_hash=delta_hash,
            centroid_hash=centroid_hash,
            metadata=weight2.metadata
        )
        
        # List all weights
        all_weights = temp_store.list_weights()
        
        assert len(all_weights) == 2
        assert weight1_hash in all_weights
        assert weight2_hash in all_weights
    
    def test_get_metadata_for_clustered_weight(self, temp_store, sample_weights):
        """Test getting metadata for a clustered weight"""
        weight1, weight2, centroid = sample_weights
        
        # Store clustered weight
        centroid_hash = temp_store.store_centroid(centroid)
        delta_hash = "delta_test_hash"
        weight_hash = weight2.compute_hash()
        
        temp_store.store_clustered_weight(
            weight_hash=weight_hash,
            cluster_id="cluster_0",
            delta_hash=delta_hash,
            centroid_hash=centroid_hash,
            metadata=weight2.metadata
        )
        
        # Get metadata
        metadata = temp_store.get_metadata(weight_hash)
        
        assert metadata is not None
        assert metadata.name == weight2.metadata.name
        assert metadata.shape == weight2.metadata.shape
        assert metadata.dtype == weight2.metadata.dtype
        assert metadata.layer_type == weight2.metadata.layer_type
        assert metadata.model_name == weight2.metadata.model_name
    
    def test_delete_clustered_weight(self, temp_store, sample_weights):
        """Test deleting a clustered weight"""
        weight1, weight2, centroid = sample_weights
        
        # Store centroid
        centroid_hash = temp_store.store_centroid(centroid)
        
        # Create and store delta
        delta_data = weight2.data - centroid.data
        delta = Delta(
            delta_type=DeltaType.FLOAT32_RAW,
            data=delta_data,
            metadata={},
            original_shape=weight2.data.shape,
            original_dtype=weight2.data.dtype,
            reference_hash=centroid_hash,
            compression_ratio=2.0
        )
        delta_hash = "delta_test_hash"
        temp_store.store_delta(delta, delta_hash)
        
        weight_hash = weight2.compute_hash()
        temp_store.store_clustered_weight(
            weight_hash=weight_hash,
            cluster_id="cluster_0",
            delta_hash=delta_hash,
            centroid_hash=centroid_hash,
            metadata=weight2.metadata
        )
        
        # Delete the weight
        deleted = temp_store.delete(weight_hash)
        
        assert deleted is True
        assert not temp_store.exists(weight_hash)
        assert not temp_store.is_weight_clustered(weight_hash)
        
        # Verify centroid and delta still exist (they might be shared)
        loaded_centroid = temp_store.load_centroid(centroid_hash)
        assert loaded_centroid is not None
        
        loaded_delta = temp_store.load_delta(delta_hash)
        assert loaded_delta is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])