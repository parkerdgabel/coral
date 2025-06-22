"""Test Product Quantization storage in HDF5 backend."""

import numpy as np
import pytest
import tempfile
import os

from coral.storage.hdf5_store import HDF5Store
from coral.delta.product_quantization import PQCodebook
from coral.delta.delta_encoder import Delta, DeltaType
from coral.core.weight_tensor import WeightMetadata, WeightTensor


class TestHDF5PQStorage:
    """Test PQ codebook storage functionality."""
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary HDF5 store."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name
        
        store = HDF5Store(temp_path)
        yield store
        store.close()
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_store_and_load_pq_codebook(self, temp_store):
        """Test storing and loading PQ codebooks."""
        # Create a sample PQ codebook
        subvector_size = 4
        num_subvectors = 8
        num_codewords = 256
        
        codebooks = [
            np.random.randn(num_codewords, subvector_size).astype(np.float32)
            for _ in range(num_subvectors)
        ]
        
        codebook = PQCodebook(
            subvector_size=subvector_size,
            codebooks=codebooks,
            version="1.0",
            training_stats={"iterations": 10, "inertia": 0.5}
        )
        
        codebook_id = "test_codebook_001"
        
        # Store the codebook
        temp_store.store_pq_codebook(codebook_id, codebook)
        
        # Check it exists
        assert temp_store.has_pq_codebook(codebook_id)
        
        # Load it back
        loaded_codebook = temp_store.load_pq_codebook(codebook_id)
        assert loaded_codebook is not None
        assert loaded_codebook.subvector_size == subvector_size
        assert len(loaded_codebook.codebooks) == num_subvectors
        assert loaded_codebook.version == "1.0"
        assert loaded_codebook.training_stats["iterations"] == 10
        
        # Check codebook data
        for i in range(num_subvectors):
            np.testing.assert_array_almost_equal(
                loaded_codebook.codebooks[i],
                codebooks[i]
            )
    
    def test_list_pq_codebooks(self, temp_store):
        """Test listing PQ codebooks."""
        # Initially empty
        assert temp_store.list_pq_codebooks() == []
        
        # Add some codebooks
        codebook_ids = ["cb_001", "cb_002", "cb_003"]
        for cb_id in codebook_ids:
            codebook = PQCodebook(
                subvector_size=4,
                codebooks=[np.random.randn(16, 4).astype(np.float32)],
                version="1.0"
            )
            temp_store.store_pq_codebook(cb_id, codebook)
        
        # List them
        listed = temp_store.list_pq_codebooks()
        assert len(listed) == 3
        assert set(listed) == set(codebook_ids)
    
    def test_delete_pq_codebook(self, temp_store):
        """Test deleting PQ codebooks."""
        codebook_id = "test_delete"
        codebook = PQCodebook(
            subvector_size=4,
            codebooks=[np.random.randn(16, 4).astype(np.float32)],
            version="1.0"
        )
        
        # Store and verify
        temp_store.store_pq_codebook(codebook_id, codebook)
        assert temp_store.has_pq_codebook(codebook_id)
        
        # Delete
        temp_store.delete_pq_codebook(codebook_id)
        assert not temp_store.has_pq_codebook(codebook_id)
        
        # Try deleting non-existent (should not raise)
        temp_store.delete_pq_codebook("non_existent")
    
    def test_pq_delta_storage(self, temp_store):
        """Test storing PQ deltas with indices and residuals."""
        # Create a PQ delta with indices
        indices = np.array([1, 5, 10, 15, 20], dtype=np.uint8)
        residual = np.array([0.01, -0.02, 0.03, -0.04, 0.05], dtype=np.float32)
        
        delta = Delta(
            delta_type=DeltaType.PQ_LOSSLESS,
            data=indices,
            metadata={
                "codebook_id": "test_cb_001",
                "residual": residual,
                "num_subvectors": 5
            },
            original_shape=(5,),
            original_dtype=np.float32,
            reference_hash="ref_001",
            compression_ratio=0.8
        )
        
        delta_hash = "pq_delta_001"
        
        # Store the delta
        temp_store.store_delta(delta, delta_hash)
        
        # Load it back
        loaded_delta = temp_store.load_delta(delta_hash)
        assert loaded_delta is not None
        assert loaded_delta.delta_type == DeltaType.PQ_LOSSLESS
        np.testing.assert_array_equal(loaded_delta.data, indices)
        assert loaded_delta.metadata["codebook_id"] == "test_cb_001"
        assert "residual" in loaded_delta.metadata
        np.testing.assert_array_almost_equal(
            loaded_delta.metadata["residual"],
            residual
        )
    
    def test_pq_delta_without_residual(self, temp_store):
        """Test storing PQ deltas without residuals (lossy mode)."""
        indices = np.array([1, 5, 10, 15, 20], dtype=np.uint16)
        
        delta = Delta(
            delta_type=DeltaType.PQ_ENCODED,
            data=indices,
            metadata={
                "codebook_id": "test_cb_002",
                "num_subvectors": 5
            },
            original_shape=(5,),
            original_dtype=np.float32,
            reference_hash="ref_002",
            compression_ratio=0.9
        )
        
        delta_hash = "pq_delta_002"
        
        # Store the delta
        temp_store.store_delta(delta, delta_hash)
        
        # Load it back
        loaded_delta = temp_store.load_delta(delta_hash)
        assert loaded_delta is not None
        assert loaded_delta.delta_type == DeltaType.PQ_ENCODED
        np.testing.assert_array_equal(loaded_delta.data, indices)
        assert "residual" not in loaded_delta.metadata
    
    def test_gc_orphaned_codebooks(self, temp_store):
        """Test garbage collection of orphaned PQ codebooks."""
        # Create codebooks
        used_codebook_id = "used_cb"
        orphaned_codebook_id = "orphaned_cb"
        
        for cb_id in [used_codebook_id, orphaned_codebook_id]:
            codebook = PQCodebook(
                subvector_size=4,
                codebooks=[np.random.randn(16, 4).astype(np.float32)],
                version="1.0"
            )
            temp_store.store_pq_codebook(cb_id, codebook)
        
        # Create a delta that references the used codebook
        delta = Delta(
            delta_type=DeltaType.PQ_ENCODED,
            data=np.array([1, 2, 3], dtype=np.uint8),
            metadata={"codebook_id": used_codebook_id},
            original_shape=(3,),
            original_dtype=np.float32,
            reference_hash="ref_003",
            compression_ratio=0.7
        )
        temp_store.store_delta(delta, "delta_001")
        
        # Run garbage collection
        deleted = temp_store.gc()
        
        # Check results
        assert deleted["pq_codebooks"] == 1
        assert temp_store.has_pq_codebook(used_codebook_id)
        assert not temp_store.has_pq_codebook(orphaned_codebook_id)
    
    def test_pq_storage_info(self, temp_store):
        """Test PQ storage information retrieval."""
        # Add some codebooks
        for i in range(3):
            codebook = PQCodebook(
                subvector_size=4,
                codebooks=[
                    np.random.randn(256, 4).astype(np.float32)
                    for _ in range(8)
                ],
                version="1.0",
                training_stats={"id": i}
            )
            temp_store.store_pq_codebook(f"cb_{i:03d}", codebook)
        
        # Get PQ storage info
        info = temp_store.get_pq_storage_info()
        
        assert info["total_pq_codebooks"] == 3
        assert info["total_pq_bytes"] > 0
        assert len(info["codebook_stats"]) == 3
        
        # Check individual codebook stats
        for stat in info["codebook_stats"]:
            assert stat["num_subvectors"] == 8
            assert stat["num_codewords"] == 256
            assert stat["subvector_size"] == 4
            assert stat["bytes"] > 0
            assert stat["version"] == "1.0"
    
    def test_migration_support(self):
        """Test migration of old HDF5 files without PQ support."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name
        
        try:
            # Create an old-style store without PQ support
            import h5py
            with h5py.File(temp_path, "w") as f:
                f.create_group("weights")
                f.create_group("metadata")
                f.create_group("deltas")
                f.attrs["version"] = "1.0"
            
            # Open with our store (should trigger migration)
            store = HDF5Store(temp_path, mode="r+")
            
            # Check that PQ group was created
            assert store.has_pq_codebook("dummy") == False  # Group exists but empty
            
            # Check version was updated
            with store._file_operation() as f:
                assert f.attrs["version"] == "2.0"
                assert "migrated_at" in f.attrs
            
            store.close()
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)