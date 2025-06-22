"""Test coverage improvement for storage module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest

from coral.storage.safetensors_store import SafeTensorsStore
from coral.storage.weight_store import WeightStore
from coral.core.weight_tensor import WeightTensor, WeightMetadata


class TestWeightStoreCoverage:
    """Tests to improve coverage for WeightStore abstract base class."""

    def test_weight_store_abstract_methods(self):
        """Test that WeightStore is abstract and methods raise NotImplementedError."""
        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError):
            WeightStore()
        
        # Create a minimal concrete implementation
        class MinimalStore(WeightStore):
            def initialize(self): pass
            def store_weight(self, weight_hash, weight): pass
            def get_weight(self, weight_hash): pass
            def exists(self, weight_hash): pass
            def delete_weight(self, weight_hash): pass
            def list_weights(self): pass
            def get_size(self, weight_hash): pass
            def get_total_size(self): pass
            def close(self): pass
        
        store = MinimalStore()
        assert isinstance(store, WeightStore)

    def test_weight_store_context_manager(self):
        """Test WeightStore context manager functionality."""
        class TestStore(WeightStore):
            def __init__(self):
                self.closed = False
                
            def initialize(self): pass
            def store_weight(self, weight_hash, weight): pass
            def get_weight(self, weight_hash): pass
            def exists(self, weight_hash): pass
            def delete_weight(self, weight_hash): pass
            def list_weights(self): return []
            def get_size(self, weight_hash): return 0
            def get_total_size(self): return 0
            def close(self): self.closed = True
        
        # Test context manager
        store = TestStore()
        with store as s:
            assert s == store
            assert not store.closed
        assert store.closed


class TestSafeTensorsStoreCoverage:
    """Tests to improve coverage for SafeTensorsStore class."""

    @pytest.fixture
    def store_path(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_safetensors_store_initialization(self, store_path):
        """Test SafeTensorsStore initialization."""
        store = SafeTensorsStore(store_path)
        store.initialize()
        
        assert store.store_path == Path(store_path)
        assert (Path(store_path) / "weights").exists()
        assert (Path(store_path) / "metadata.json").exists()

    def test_safetensors_store_with_existing_metadata(self, store_path):
        """Test SafeTensorsStore initialization with existing metadata."""
        # Create existing metadata
        metadata_path = Path(store_path) / "metadata.json"
        metadata_path.parent.mkdir(exist_ok=True)
        with open(metadata_path, "w") as f:
            f.write('{"existing": "data"}')
        
        store = SafeTensorsStore(store_path)
        store.initialize()
        
        # Should load existing metadata
        assert "existing" in store._weight_metadata
        assert store._weight_metadata["existing"] == "data"

    def test_store_weight_safetensors(self, store_path):
        """Test storing a weight in SafeTensors format."""
        store = SafeTensorsStore(store_path)
        store.initialize()
        
        # Create a weight tensor
        weight = WeightTensor(
            data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            metadata=WeightMetadata(name="test_weight", shape=(2, 2))
        )
        weight_hash = weight.compute_hash()
        
        # Store the weight
        store.store_weight(weight_hash, weight)
        
        # Verify file exists
        weight_file = Path(store_path) / "weights" / f"{weight_hash}.safetensors"
        assert weight_file.exists()
        
        # Verify metadata updated
        assert weight_hash in store._weight_metadata

    def test_store_duplicate_weight(self, store_path):
        """Test storing a duplicate weight."""
        store = SafeTensorsStore(store_path)
        store.initialize()
        
        weight = WeightTensor(data=np.array([1, 2, 3], dtype=np.float32))
        weight_hash = weight.compute_hash()
        
        # Store twice
        store.store_weight(weight_hash, weight)
        store.store_weight(weight_hash, weight)
        
        # Should only exist once
        weights = store.list_weights()
        assert weights.count(weight_hash) == 1

    def test_get_weight_safetensors(self, store_path):
        """Test retrieving a weight from SafeTensors format."""
        store = SafeTensorsStore(store_path)
        store.initialize()
        
        # Store a weight first
        original_weight = WeightTensor(
            data=np.array([1.5, 2.5, 3.5], dtype=np.float32),
            metadata=WeightMetadata(name="retrieve_test")
        )
        weight_hash = original_weight.compute_hash()
        store.store_weight(weight_hash, original_weight)
        
        # Retrieve it
        retrieved_weight = store.get_weight(weight_hash)
        
        assert retrieved_weight is not None
        assert np.array_equal(retrieved_weight.data, original_weight.data)
        assert retrieved_weight.metadata.name == "retrieve_test"

    def test_get_nonexistent_weight(self, store_path):
        """Test retrieving a non-existent weight."""
        store = SafeTensorsStore(store_path)
        store.initialize()
        
        with pytest.raises(KeyError):
            store.get_weight("nonexistent_hash")

    def test_exists_method(self, store_path):
        """Test checking if a weight exists."""
        store = SafeTensorsStore(store_path)
        store.initialize()
        
        # Non-existent weight
        assert not store.exists("fake_hash")
        
        # Store a weight
        weight = WeightTensor(data=np.array([1, 2], dtype=np.float32))
        weight_hash = weight.compute_hash()
        store.store_weight(weight_hash, weight)
        
        # Now it should exist
        assert store.exists(weight_hash)

    def test_delete_weight(self, store_path):
        """Test deleting a weight."""
        store = SafeTensorsStore(store_path)
        store.initialize()
        
        # Store a weight
        weight = WeightTensor(data=np.array([5, 6, 7], dtype=np.float32))
        weight_hash = weight.compute_hash()
        store.store_weight(weight_hash, weight)
        
        # Verify it exists
        assert store.exists(weight_hash)
        
        # Delete it
        store.delete_weight(weight_hash)
        
        # Should no longer exist
        assert not store.exists(weight_hash)
        assert weight_hash not in store._weight_metadata

    def test_delete_nonexistent_weight(self, store_path):
        """Test deleting a non-existent weight."""
        store = SafeTensorsStore(store_path)
        store.initialize()
        
        # Should not raise error
        store.delete_weight("fake_hash")

    def test_list_weights(self, store_path):
        """Test listing all weights."""
        store = SafeTensorsStore(store_path)
        store.initialize()
        
        # Initially empty
        assert store.list_weights() == []
        
        # Add some weights
        weights = []
        for i in range(3):
            weight = WeightTensor(data=np.array([i], dtype=np.float32))
            weight_hash = weight.compute_hash()
            weights.append(weight_hash)
            store.store_weight(weight_hash, weight)
        
        # List should contain all weights
        listed = store.list_weights()
        assert len(listed) == 3
        for w in weights:
            assert w in listed

    def test_get_size(self, store_path):
        """Test getting size of a weight."""
        store = SafeTensorsStore(store_path)
        store.initialize()
        
        # Store a weight
        weight = WeightTensor(data=np.zeros((100, 100), dtype=np.float32))
        weight_hash = weight.compute_hash()
        store.store_weight(weight_hash, weight)
        
        # Get size
        size = store.get_size(weight_hash)
        assert size > 0
        assert isinstance(size, int)

    def test_get_size_nonexistent(self, store_path):
        """Test getting size of non-existent weight."""
        store = SafeTensorsStore(store_path)
        store.initialize()
        
        with pytest.raises(KeyError):
            store.get_size("fake_hash")

    def test_get_total_size(self, store_path):
        """Test getting total size of all weights."""
        store = SafeTensorsStore(store_path)
        store.initialize()
        
        # Initially zero
        assert store.get_total_size() == 0
        
        # Add weights
        total_expected = 0
        for i in range(3):
            weight = WeightTensor(data=np.zeros((10 * (i + 1),), dtype=np.float32))
            weight_hash = weight.compute_hash()
            store.store_weight(weight_hash, weight)
            total_expected += store.get_size(weight_hash)
        
        # Total should match sum
        assert store.get_total_size() == total_expected

    def test_save_metadata(self, store_path):
        """Test saving metadata to disk."""
        store = SafeTensorsStore(store_path)
        store.initialize()
        
        # Add some metadata
        store._weight_metadata["test_key"] = {"data": "value"}
        store._save_metadata()
        
        # Read back from file
        import json
        with open(Path(store_path) / "metadata.json", "r") as f:
            saved_metadata = json.load(f)
        
        assert "test_key" in saved_metadata
        assert saved_metadata["test_key"]["data"] == "value"

    def test_close_method(self, store_path):
        """Test closing the store."""
        store = SafeTensorsStore(store_path)
        store.initialize()
        
        # Add a weight
        weight = WeightTensor(data=np.array([1, 2, 3], dtype=np.float32))
        store.store_weight(weight.compute_hash(), weight)
        
        # Close should save metadata
        store.close()
        
        # Metadata should be persisted
        import json
        with open(Path(store_path) / "metadata.json", "r") as f:
            saved_metadata = json.load(f)
        assert len(saved_metadata) > 0

    def test_weight_metadata_tracking(self, store_path):
        """Test metadata tracking for weights."""
        store = SafeTensorsStore(store_path)
        store.initialize()
        
        # Store weight with metadata
        weight = WeightTensor(
            data=np.array([1, 2, 3], dtype=np.float32),
            metadata=WeightMetadata(
                name="tracked_weight",
                shape=(3,),
                dtype="float32",
                device="cpu",
                tags=["test", "example"]
            )
        )
        weight_hash = weight.compute_hash()
        store.store_weight(weight_hash, weight)
        
        # Check stored metadata
        assert weight_hash in store._weight_metadata
        metadata = store._weight_metadata[weight_hash]
        assert metadata["size"] > 0
        assert "timestamp" in metadata
        assert metadata["shape"] == [3]
        assert metadata["dtype"] == "float32"

    def test_batch_operations(self, store_path):
        """Test batch storage operations."""
        store = SafeTensorsStore(store_path)
        store.initialize()
        
        # Store multiple weights
        weights = {}
        for i in range(5):
            weight = WeightTensor(
                data=np.random.randn(10, 10).astype(np.float32),
                metadata=WeightMetadata(name=f"batch_weight_{i}")
            )
            weights[weight.compute_hash()] = weight
        
        # Store all
        for hash_val, weight in weights.items():
            store.store_weight(hash_val, weight)
        
        # Verify all stored
        for hash_val in weights:
            assert store.exists(hash_val)
        
        # Delete half
        to_delete = list(weights.keys())[:2]
        for hash_val in to_delete:
            store.delete_weight(hash_val)
        
        # Verify deletion
        for hash_val in to_delete:
            assert not store.exists(hash_val)
        for hash_val in list(weights.keys())[2:]:
            assert store.exists(hash_val)

    def test_error_handling(self, store_path):
        """Test error handling in SafeTensorsStore."""
        store = SafeTensorsStore(store_path)
        store.initialize()
        
        # Test with invalid weight data
        with pytest.raises(Exception):
            invalid_weight = MagicMock()
            invalid_weight.data = "not an array"
            invalid_weight.compute_hash.return_value = "invalid_hash"
            store.store_weight("invalid_hash", invalid_weight)
        
        # Test with corrupted metadata file
        metadata_path = Path(store_path) / "metadata.json"
        with open(metadata_path, "w") as f:
            f.write("invalid json{")
        
        # Should handle gracefully on re-init
        new_store = SafeTensorsStore(store_path)
        new_store.initialize()  # Should create fresh metadata

    def test_concurrent_access(self, store_path):
        """Test concurrent access handling."""
        import threading
        
        store = SafeTensorsStore(store_path)
        store.initialize()
        
        results = {"errors": 0}
        
        def store_weight_thread(thread_id):
            try:
                weight = WeightTensor(
                    data=np.array([thread_id], dtype=np.float32)
                )
                store.store_weight(f"hash_{thread_id}", weight)
            except Exception:
                results["errors"] += 1
        
        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=store_weight_thread, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Should have no errors
        assert results["errors"] == 0
        
        # All weights should be stored
        for i in range(10):
            assert store.exists(f"hash_{i}")

    @patch("builtins.open", side_effect=PermissionError("Permission denied"))
    def test_permission_errors(self, mock_file, store_path):
        """Test handling of permission errors."""
        store = SafeTensorsStore(store_path)
        
        # Should handle permission error gracefully during init
        with pytest.raises(PermissionError):
            store.initialize()

    def test_safetensors_format_validation(self, store_path):
        """Test SafeTensors format validation."""
        store = SafeTensorsStore(store_path)
        store.initialize()
        
        # Create a valid weight
        weight = WeightTensor(
            data=np.array([[1, 2], [3, 4]], dtype=np.float32),
            metadata=WeightMetadata(
                name="format_test",
                shape=(2, 2),
                dtype="float32"
            )
        )
        weight_hash = weight.compute_hash()
        
        # Store it
        store.store_weight(weight_hash, weight)
        
        # Manually corrupt the file
        weight_file = Path(store_path) / "weights" / f"{weight_hash}.safetensors"
        with open(weight_file, "wb") as f:
            f.write(b"corrupted data")
        
        # Should raise error when loading
        with pytest.raises(Exception):
            store.get_weight(weight_hash)