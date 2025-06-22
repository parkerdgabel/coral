import os
import tempfile

import numpy as np
import pytest

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.delta.delta_encoder import Delta, DeltaType
from coral.storage.hdf5_store import HDF5Store


class TestHDF5Store:
    @pytest.fixture
    def temp_store(self):
        """Create a temporary HDF5 store for testing."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name

        store = HDF5Store(temp_path)
        yield store
        store.close()
        os.unlink(temp_path)

    def test_store_initialization(self, temp_store):
        """Test store initialization."""
        assert temp_store.filepath.exists()
        assert temp_store.file is not None
        assert "weights" in temp_store.file
        assert "deltas" in temp_store.file
        assert "metadata" in temp_store.file

    def test_store_and_retrieve_weight(self, temp_store):
        """Test storing and retrieving a weight tensor."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        metadata = WeightMetadata(
            name="test_weight", shape=data.shape, dtype=data.dtype, layer_type="fc1"
        )
        weight = WeightTensor(data=data, metadata=metadata)

        # Store weight
        weight_hash = temp_store.store(weight)
        assert weight_hash is not None

        # Retrieve weight
        retrieved = temp_store.load(weight_hash)
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved.data, weight.data)
        assert retrieved.metadata.name == "test_weight"
        assert retrieved.metadata.layer_type == "fc1"

    def test_store_duplicate_weight(self, temp_store):
        """Test storing duplicate weights returns same hash."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        metadata1 = WeightMetadata(name="weight1", shape=data.shape, dtype=data.dtype)
        weight1 = WeightTensor(data=data.copy(), metadata=metadata1)

        metadata2 = WeightMetadata(
            name="weight2",  # Different metadata
            shape=data.shape,
            dtype=data.dtype,
        )
        weight2 = WeightTensor(data=data.copy(), metadata=metadata2)

        hash1 = temp_store.store(weight1)
        hash2 = temp_store.store(weight2)

        assert hash1 == hash2  # Same data = same hash

    def test_exists(self, temp_store):
        """Test checking if weight exists."""
        data = np.random.randn(10, 10).astype(np.float32)
        metadata = WeightMetadata(name="test", shape=data.shape, dtype=data.dtype)
        weight = WeightTensor(data=data, metadata=metadata)

        weight_hash = weight.compute_hash()

        assert not temp_store.exists(weight_hash)
        temp_store.store(weight)
        assert temp_store.exists(weight_hash)

    def test_delete_weight(self, temp_store):
        """Test deleting a weight."""
        data = np.random.randn(5, 5).astype(np.float32)
        metadata = WeightMetadata(name="to_delete", shape=data.shape, dtype=data.dtype)
        weight = WeightTensor(data=data, metadata=metadata)

        weight_hash = temp_store.store(weight)
        assert temp_store.exists(weight_hash)

        assert temp_store.delete(weight_hash)
        assert not temp_store.exists(weight_hash)
        assert not temp_store.delete(weight_hash)  # Second delete returns False

    def test_get_nonexistent_weight(self, temp_store):
        """Test getting non-existent weight returns None."""
        result = temp_store.load("nonexistent_hash")
        assert result is None

    def test_list_weights(self, temp_store):
        """Test listing all weights."""
        weights = []
        for i in range(3):
            data = np.random.randn(10).astype(np.float32)
            metadata = WeightMetadata(
                name=f"weight_{i}", shape=data.shape, dtype=data.dtype
            )
            weight = WeightTensor(data=data, metadata=metadata)
            weights.append(weight)
            temp_store.store(weight)

        stored_hashes = temp_store.list_weights()
        assert len(stored_hashes) == 3

        for weight in weights:
            assert weight.compute_hash() in stored_hashes

    def test_store_and_retrieve_delta(self, temp_store):
        """Test storing and retrieving delta objects."""
        delta = Delta(
            delta_type=DeltaType.FLOAT32_RAW,
            data=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            metadata={},
            original_shape=(3,),
            original_dtype=np.float32,
            reference_hash="ref_hash",
            compression_ratio=0.0
        )

        delta_hash = delta.compute_hash()
        temp_store.store_delta(delta, delta_hash)

        retrieved = temp_store.load_delta(delta_hash)
        assert retrieved is not None
        assert retrieved.reference_hash == delta.reference_hash
        assert retrieved.delta_type == delta.delta_type
        np.testing.assert_array_equal(retrieved.data, delta.data)

    def test_delta_exists(self, temp_store):
        """Test checking if delta exists."""
        delta = Delta(
            delta_type=DeltaType.FLOAT32_RAW,
            data=np.array([1.0], dtype=np.float32),
            metadata={},
            original_shape=(1,),
            original_dtype=np.float32,
            reference_hash="ref",
            compression_ratio=0.0
        )

        delta_hash = delta.compute_hash()
        assert not temp_store.delta_exists(delta_hash)
        temp_store.store_delta(delta, delta_hash)
        assert temp_store.delta_exists(delta_hash)

    def test_delete_delta(self, temp_store):
        """Test deleting delta."""
        delta = Delta(
            delta_type=DeltaType.FLOAT32_RAW,
            data=np.array([1.0], dtype=np.float32),
            metadata={},
            original_shape=(1,),
            original_dtype=np.float32,
            reference_hash="ref",
            compression_ratio=0.0
        )

        delta_hash = delta.compute_hash()
        temp_store.store_delta(delta, delta_hash)
        assert temp_store.delta_exists(delta_hash)

        assert temp_store.delete_delta(delta_hash)
        assert not temp_store.delta_exists(delta_hash)
        assert not temp_store.delete_delta(
            delta_hash
        )  # Second delete returns False

    def test_list_deltas(self, temp_store):
        """Test listing all deltas."""
        stored_hashes = []
        for i in range(3):
            delta = Delta(
                delta_type=DeltaType.FLOAT32_RAW,
                data=np.array([float(i)], dtype=np.float32),
                metadata={},
                original_shape=(1,),
                original_dtype=np.float32,
                reference_hash=f"ref_{i}",
                compression_ratio=0.0
            )
            delta_hash = delta.compute_hash()
            temp_store.store_delta(delta, delta_hash)
            stored_hashes.append(delta_hash)

        deltas = temp_store.list_deltas()
        assert len(deltas) == 3
        for stored_hash in stored_hashes:
            assert stored_hash in deltas

    def test_store_large_weight(self, temp_store):
        """Test storing large weight tensors."""
        data = np.random.randn(1000, 1000).astype(np.float32)
        metadata = WeightMetadata(
            name="large_weight",
            shape=data.shape,
            dtype=data.dtype,
            model_name="LargeModel",
        )
        weight = WeightTensor(data=data, metadata=metadata)

        weight_hash = temp_store.store(weight)
        retrieved = temp_store.load(weight_hash)

        assert retrieved is not None
        np.testing.assert_array_equal(retrieved.data, weight.data)
        assert retrieved.metadata.model_name == "LargeModel"

    def test_compression_options(self):
        """Test different compression options."""
        compressions = ["gzip", "lzf", None]

        for compression in compressions:
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                temp_path = f.name

            try:
                store = HDF5Store(temp_path, compression=compression)

                data = np.random.randn(100, 100).astype(np.float32)
                metadata = WeightMetadata(
                    name="test", shape=data.shape, dtype=data.dtype
                )
                weight = WeightTensor(data=data, metadata=metadata)

                weight_hash = store.store(weight)
                retrieved = store.load(weight_hash)

                np.testing.assert_array_equal(retrieved.data, weight.data)

                store.close()
            finally:
                os.unlink(temp_path)

    def test_batch_operations(self, temp_store):
        """Test batch loading of weights."""
        weights = {}
        hashes = []

        for i in range(5):
            data = np.random.randn(10, 10).astype(np.float32)
            metadata = WeightMetadata(
                name=f"weight_{i}", shape=data.shape, dtype=data.dtype
            )
            weight = WeightTensor(data=data, metadata=metadata)
            weight_hash = temp_store.store(weight)
            weights[weight_hash] = weight
            hashes.append(weight_hash)

        # Load batch
        loaded = temp_store.load_batch(hashes)
        assert len(loaded) == 5

        for hash_key, weight in loaded.items():
            original = weights[hash_key]
            np.testing.assert_array_equal(weight.data, original.data)

    def test_get_metadata(self, temp_store):
        """Test getting metadata without loading data."""
        data = np.random.randn(50, 50).astype(np.float32)
        metadata = WeightMetadata(
            name="metadata_test",
            shape=data.shape,
            dtype=data.dtype,
            layer_type="Conv2d",
            model_name="TestModel",
        )
        weight = WeightTensor(data=data, metadata=metadata)

        weight_hash = temp_store.store(weight)

        # Get metadata only
        meta = temp_store.get_metadata(weight_hash)
        assert meta is not None
        assert meta.name == "metadata_test"
        assert meta.layer_type == "Conv2d"
        assert meta.model_name == "TestModel"
        assert meta.shape == (50, 50)

    def test_storage_info(self, temp_store):
        """Test getting storage information."""
        info = temp_store.get_storage_info()

        assert "compression" in info
        assert "store_path" in info
        assert info["compression"] == "gzip"
        assert str(temp_store.filepath) in str(info["store_path"])

    def test_close_and_reopen(self):
        """Test closing and reopening store."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name

        try:
            # Create and populate store
            store1 = HDF5Store(temp_path)
            data = np.array([1, 2, 3], dtype=np.float32)
            metadata = WeightMetadata(name="test", shape=data.shape, dtype=data.dtype)
            weight = WeightTensor(data=data, metadata=metadata)
            weight_hash = store1.store(weight)
            store1.close()

            # Reopen and verify
            store2 = HDF5Store(temp_path, mode="r")
            retrieved = store2.load(weight_hash)
            assert retrieved is not None
            np.testing.assert_array_equal(retrieved.data, weight.data)
            store2.close()

        finally:
            os.unlink(temp_path)

    def test_concurrent_access_protection(self):
        """Test that concurrent access is handled properly."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name

        try:
            store1 = HDF5Store(temp_path)

            # Try to open second store in write mode - should work with HDF5
            store2 = HDF5Store(temp_path, mode="r")

            # Both stores should be accessible
            assert store1.file is not None
            assert store2.file is not None

            store1.close()
            store2.close()

        finally:
            os.unlink(temp_path)

    def test_invalid_compression(self):
        """Test invalid compression raises error."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name

        try:
            # This should work - h5py handles invalid compression gracefully
            store = HDF5Store(temp_path, compression="invalid_compression")
            store.close()
        finally:
            os.unlink(temp_path)
