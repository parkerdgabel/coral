
import numpy as np
import pytest

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.storage.weight_store import WeightStore


class MockWeightStore(WeightStore):
    """Mock implementation of WeightStore for testing."""

    def __init__(self):
        self.storage = {}
        self.metadata = {}
        self.closed = False

    def store(self, weight: WeightTensor, hash_key: str = None) -> str:
        if hash_key is None:
            hash_key = weight.compute_hash()
        self.storage[hash_key] = weight
        self.metadata[hash_key] = weight.metadata
        return hash_key

    def load(self, hash_key: str) -> WeightTensor:
        return self.storage.get(hash_key)

    def exists(self, hash_key: str) -> bool:
        return hash_key in self.storage

    def delete(self, hash_key: str) -> bool:
        if hash_key in self.storage:
            del self.storage[hash_key]
            del self.metadata[hash_key]
            return True
        return False

    def list_weights(self) -> list:
        return list(self.storage.keys())

    def get_metadata(self, hash_key: str) -> WeightMetadata:
        return self.metadata.get(hash_key)

    def store_batch(self, weights: dict) -> dict:
        result = {}
        for name, weight in weights.items():
            hash_key = self.store(weight)
            result[name] = hash_key
        return result

    def load_batch(self, hash_keys: list) -> dict:
        result = {}
        for hash_key in hash_keys:
            if hash_key in self.storage:
                result[hash_key] = self.storage[hash_key]
        return result

    def get_storage_info(self) -> dict:
        return {
            "num_weights": len(self.storage),
            "total_bytes": sum(w.nbytes for w in self.storage.values()),
        }

    def close(self):
        self.closed = True


class TestWeightStore:
    def test_abstract_methods(self):
        """Test that WeightStore is abstract and can't be instantiated."""
        with pytest.raises(TypeError):
            WeightStore()

    def test_store_and_load(self):
        """Test basic store and load operations."""
        store = MockWeightStore()

        # Create test weight
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        metadata = WeightMetadata(
            name="test_weight", shape=data.shape, dtype=data.dtype
        )
        weight = WeightTensor(data=data, metadata=metadata)

        # Store weight
        hash_key = store.store(weight)
        assert hash_key is not None
        assert store.exists(hash_key)

        # Load weight
        loaded = store.load(hash_key)
        assert loaded is not None
        np.testing.assert_array_equal(loaded.data, weight.data)
        assert loaded.metadata.name == weight.metadata.name

    def test_store_with_custom_hash(self):
        """Test storing with custom hash key."""
        store = MockWeightStore()

        data = np.ones(5, dtype=np.float32)
        metadata = WeightMetadata(name="test", shape=data.shape, dtype=data.dtype)
        weight = WeightTensor(data=data, metadata=metadata)

        custom_hash = "custom_hash_123"
        returned_hash = store.store(weight, hash_key=custom_hash)

        assert returned_hash == custom_hash
        assert store.exists(custom_hash)

    def test_load_nonexistent(self):
        """Test loading non-existent weight."""
        store = MockWeightStore()
        loaded = store.load("nonexistent_hash")
        assert loaded is None

    def test_delete(self):
        """Test deleting weights."""
        store = MockWeightStore()

        # Store a weight
        data = np.ones(3, dtype=np.float32)
        metadata = WeightMetadata(name="test", shape=data.shape, dtype=data.dtype)
        weight = WeightTensor(data=data, metadata=metadata)
        hash_key = store.store(weight)

        # Delete it
        assert store.delete(hash_key) is True
        assert not store.exists(hash_key)
        assert store.load(hash_key) is None

        # Try deleting again
        assert store.delete(hash_key) is False

    def test_list_weights(self):
        """Test listing all weights."""
        store = MockWeightStore()

        # Store multiple weights
        hashes = []
        for i in range(3):
            data = np.ones(5) * i
            metadata = WeightMetadata(
                name=f"weight_{i}", shape=data.shape, dtype=data.dtype
            )
            weight = WeightTensor(data=data, metadata=metadata)
            hash_key = store.store(weight)
            hashes.append(hash_key)

        # List weights
        listed = store.list_weights()
        assert len(listed) == 3
        for h in hashes:
            assert h in listed

    def test_get_metadata(self):
        """Test getting metadata without loading data."""
        store = MockWeightStore()

        data = np.random.randn(10, 10).astype(np.float32)
        metadata = WeightMetadata(
            name="conv.weight", shape=data.shape, dtype=data.dtype, layer_type="Conv2d"
        )
        weight = WeightTensor(data=data, metadata=metadata)
        hash_key = store.store(weight)

        # Get metadata
        loaded_metadata = store.get_metadata(hash_key)
        assert loaded_metadata is not None
        assert loaded_metadata.name == "conv.weight"
        assert loaded_metadata.layer_type == "Conv2d"

        # Try non-existent
        assert store.get_metadata("nonexistent") is None

    def test_batch_operations(self):
        """Test batch store and load."""
        store = MockWeightStore()

        # Create multiple weights
        weights = {}
        for i in range(5):
            data = np.random.randn(3, 3).astype(np.float32)
            metadata = WeightMetadata(
                name=f"layer_{i}", shape=data.shape, dtype=data.dtype
            )
            weights[f"layer_{i}"] = WeightTensor(data=data, metadata=metadata)

        # Batch store
        hash_map = store.store_batch(weights)
        assert len(hash_map) == 5

        # Verify all stored
        for _name, hash_key in hash_map.items():
            assert store.exists(hash_key)

        # Batch load
        hash_keys = list(hash_map.values())
        loaded = store.load_batch(hash_keys)
        assert len(loaded) == 5

        # Verify content
        for name, original_weight in weights.items():
            hash_key = hash_map[name]
            loaded_weight = loaded[hash_key]
            np.testing.assert_array_equal(loaded_weight.data, original_weight.data)

    def test_load_batch_partial(self):
        """Test batch load with some missing weights."""
        store = MockWeightStore()

        # Store some weights
        data1 = np.ones(3, dtype=np.float32)
        metadata1 = WeightMetadata(name="weight1", shape=data1.shape, dtype=data1.dtype)
        weight1 = WeightTensor(data=data1, metadata=metadata1)
        hash1 = store.store(weight1)

        # Try to load existing and non-existing
        loaded = store.load_batch([hash1, "nonexistent_hash"])
        assert len(loaded) == 1
        assert hash1 in loaded
        assert "nonexistent_hash" not in loaded

    def test_storage_info(self):
        """Test getting storage information."""
        store = MockWeightStore()

        # Empty store
        info = store.get_storage_info()
        assert info["num_weights"] == 0
        assert info["total_bytes"] == 0

        # Add weights
        for i in range(3):
            data = np.ones(10, dtype=np.float32) * (
                i + 1
            )  # Different data for each weight
            metadata = WeightMetadata(
                name=f"weight_{i}", shape=data.shape, dtype=data.dtype
            )
            weight = WeightTensor(data=data, metadata=metadata)
            store.store(weight)

        info = store.get_storage_info()
        assert info["num_weights"] == 3
        assert info["total_bytes"] == 120  # 3 * 40

    def test_context_manager(self):
        """Test using store as context manager."""
        store = MockWeightStore()

        with store as s:
            assert s is store
            assert not store.closed

            # Store a weight
            data = np.ones(5, dtype=np.float32)
            metadata = WeightMetadata(name="test", shape=data.shape, dtype=data.dtype)
            weight = WeightTensor(data=data, metadata=metadata)
            s.store(weight)

        # Should be closed after context
        assert store.closed

    def test_close(self):
        """Test explicit close."""
        store = MockWeightStore()
        assert not store.closed

        store.close()
        assert store.closed
