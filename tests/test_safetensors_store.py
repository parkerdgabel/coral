"""Tests for SafetensorsStore"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.storage.safetensors_store import SafetensorsStore


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_weight():
    """Create a sample weight tensor for testing."""
    data = np.random.randn(10, 20).astype(np.float32)
    metadata = WeightMetadata(
        name="test_weight",
        shape=(10, 20),
        dtype=np.float32,
        layer_type="dense",
        model_name="test_model",
        compression_info={"method": "none"},
    )
    return WeightTensor(data=data, metadata=metadata)


@pytest.fixture
def multiple_weights():
    """Create multiple weight tensors for batch testing."""
    weights = {}
    for i in range(5):
        data = np.random.randn(10, 10).astype(np.float32)
        metadata = WeightMetadata(
            name=f"weight_{i}",
            shape=(10, 10),
            dtype=np.float32,
            layer_type="dense",
            model_name="test_model",
        )
        weights[f"weight_{i}"] = WeightTensor(data=data, metadata=metadata)
    return weights


class TestSafetensorsStore:
    """Test suite for SafetensorsStore."""

    def test_initialization(self, temp_storage_dir):
        """Test store initialization."""
        store = SafetensorsStore(temp_storage_dir)
        assert store.storage_path == Path(temp_storage_dir)
        assert not store.use_compression
        assert store.storage_path.exists()

    def test_store_and_load(self, temp_storage_dir, sample_weight):
        """Test storing and loading a weight."""
        store = SafetensorsStore(temp_storage_dir)

        # Store weight
        hash_key = store.store(sample_weight)
        assert hash_key == sample_weight.compute_hash()

        # Load weight
        loaded_weight = store.load(hash_key)
        assert loaded_weight is not None
        assert np.array_equal(loaded_weight.data, sample_weight.data)
        assert loaded_weight.metadata.name == sample_weight.metadata.name
        assert loaded_weight.metadata.shape == sample_weight.metadata.shape
        assert loaded_weight.metadata.dtype == sample_weight.metadata.dtype

    def test_store_with_compression(self, temp_storage_dir, sample_weight):
        """Test storing with compression enabled."""
        store = SafetensorsStore(temp_storage_dir, use_compression=True)

        # Store weight
        hash_key = store.store(sample_weight)

        # Check that compressed file exists
        file_path = store._get_file_path(hash_key)
        assert file_path.suffix == ".gz"
        assert file_path.exists()

        # Load weight
        loaded_weight = store.load(hash_key)
        assert loaded_weight is not None
        assert np.array_equal(loaded_weight.data, sample_weight.data)

    def test_exists(self, temp_storage_dir, sample_weight):
        """Test checking if weight exists."""
        store = SafetensorsStore(temp_storage_dir)

        hash_key = sample_weight.compute_hash()
        assert not store.exists(hash_key)

        store.store(sample_weight)
        assert store.exists(hash_key)

    def test_delete(self, temp_storage_dir, sample_weight):
        """Test deleting a weight."""
        store = SafetensorsStore(temp_storage_dir)

        # Store weight
        hash_key = store.store(sample_weight)
        assert store.exists(hash_key)

        # Delete weight
        assert store.delete(hash_key)
        assert not store.exists(hash_key)

        # Try to delete again
        assert not store.delete(hash_key)

    def test_list_weights(self, temp_storage_dir, multiple_weights):
        """Test listing all weights."""
        store = SafetensorsStore(temp_storage_dir)

        # Store multiple weights
        stored_hashes = set()
        for weight in multiple_weights.values():
            hash_key = store.store(weight)
            stored_hashes.add(hash_key)

        # List weights
        listed_hashes = set(store.list_weights())
        assert listed_hashes == stored_hashes

    def test_get_metadata(self, temp_storage_dir, sample_weight):
        """Test getting metadata without loading data."""
        store = SafetensorsStore(temp_storage_dir)

        # Store weight
        hash_key = store.store(sample_weight)

        # Get metadata
        metadata = store.get_metadata(hash_key)
        assert metadata is not None
        assert metadata.name == sample_weight.metadata.name
        assert metadata.shape == sample_weight.metadata.shape
        assert metadata.dtype == sample_weight.metadata.dtype
        assert metadata.layer_type == sample_weight.metadata.layer_type
        assert metadata.model_name == sample_weight.metadata.model_name

    def test_batch_operations(self, temp_storage_dir, multiple_weights):
        """Test batch store and load operations."""
        store = SafetensorsStore(temp_storage_dir)

        # Batch store
        hash_map = store.store_batch(multiple_weights)
        assert len(hash_map) == len(multiple_weights)

        # Batch load
        hash_keys = list(hash_map.values())
        loaded_weights = store.load_batch(hash_keys)
        assert len(loaded_weights) == len(hash_keys)

        # Verify loaded data
        for name, original_weight in multiple_weights.items():
            hash_key = hash_map[name]
            loaded_weight = loaded_weights[hash_key]
            assert np.array_equal(loaded_weight.data, original_weight.data)

    def test_storage_info(self, temp_storage_dir, multiple_weights):
        """Test getting storage information."""
        store = SafetensorsStore(temp_storage_dir)

        # Get info on empty store
        info = store.get_storage_info()
        assert info["total_files"] == 0
        assert info["total_size_bytes"] == 0

        # Store weights and check again
        store.store_batch(multiple_weights)
        info = store.get_storage_info()
        assert info["total_files"] == len(multiple_weights)
        assert info["total_size_bytes"] > 0
        assert info["storage_path"] == str(temp_storage_dir)

    def test_duplicate_store(self, temp_storage_dir, sample_weight):
        """Test storing the same weight multiple times."""
        store = SafetensorsStore(temp_storage_dir)

        # Store weight twice
        hash_key1 = store.store(sample_weight)
        hash_key2 = store.store(sample_weight)

        # Should return same hash and not create duplicate files
        assert hash_key1 == hash_key2
        assert len(store.list_weights()) == 1

    def test_load_nonexistent(self, temp_storage_dir):
        """Test loading a non-existent weight."""
        store = SafetensorsStore(temp_storage_dir)

        result = store.load("nonexistent_hash")
        assert result is None

    def test_context_manager(self, temp_storage_dir, sample_weight):
        """Test using store as context manager."""
        with SafetensorsStore(temp_storage_dir) as store:
            hash_key = store.store(sample_weight)
            loaded = store.load(hash_key)
            assert loaded is not None

    def test_metadata_preservation(self, temp_storage_dir):
        """Test that all metadata fields are preserved."""
        store = SafetensorsStore(temp_storage_dir)

        # Create weight with full metadata
        data = np.random.randn(5, 5).astype(np.float16)
        metadata = WeightMetadata(
            name="complex_weight",
            shape=(5, 5),
            dtype=np.float16,
            layer_type="conv2d",
            model_name="resnet50",
            compression_info={"method": "quantized", "bits": 8},
            hash="precomputed_hash",
        )
        weight = WeightTensor(data=data, metadata=metadata)

        # Store and load
        hash_key = store.store(weight)
        loaded = store.load(hash_key)

        # Verify all metadata fields
        assert loaded.metadata.name == metadata.name
        assert loaded.metadata.shape == metadata.shape
        assert loaded.metadata.dtype == metadata.dtype
        assert loaded.metadata.layer_type == metadata.layer_type
        assert loaded.metadata.model_name == metadata.model_name
        assert loaded.metadata.compression_info == metadata.compression_info
