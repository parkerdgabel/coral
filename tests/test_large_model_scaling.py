"""Tests for large model scaling features."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.storage.hdf5_store import HDF5Store


class TestHDF5StoreScaling:
    """Test HDF5Store scaling features."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary HDF5 store for testing."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name
        store = HDF5Store(filepath)
        yield store
        store.close()
        Path(filepath).unlink(missing_ok=True)

    @pytest.fixture
    def large_weight(self):
        """Create a large weight tensor for testing."""
        data = np.random.randn(1000, 1000).astype(np.float32)
        metadata = WeightMetadata(
            name="large_weight",
            shape=data.shape,
            dtype=data.dtype,
        )
        return WeightTensor(data=data, metadata=metadata)

    def test_chunking_enabled_by_default(self, temp_store):
        """Test that chunking is enabled by default."""
        assert temp_store.enable_chunking is True

    def test_store_with_chunking(self, temp_store, large_weight):
        """Test storing a weight with chunking."""
        hash_key = temp_store.store(large_weight)
        assert temp_store.exists(hash_key)

        # Verify chunking was applied (check HDF5 dataset has chunks)
        dataset = temp_store.file["weights"][hash_key]
        # Large weights should have chunks
        assert dataset.chunks is not None or large_weight.size <= temp_store.chunk_size

    def test_load_slice(self, temp_store, large_weight):
        """Test loading a partial slice of a weight."""
        hash_key = temp_store.store(large_weight)

        # Load first 100 rows
        sliced_data = temp_store.load_slice(hash_key, (slice(0, 100), slice(None)))

        assert sliced_data is not None
        assert sliced_data.shape == (100, 1000)
        np.testing.assert_array_equal(sliced_data, large_weight.data[:100, :])

    def test_load_slice_1d(self, temp_store):
        """Test loading a slice of a 1D weight."""
        data = np.random.randn(10000).astype(np.float32)
        weight = WeightTensor(
            data=data,
            metadata=WeightMetadata(
                name="1d_weight", shape=data.shape, dtype=data.dtype
            ),
        )
        hash_key = temp_store.store(weight)

        # Load first 1000 elements
        sliced_data = temp_store.load_slice(hash_key, slice(0, 1000))

        assert sliced_data is not None
        assert sliced_data.shape == (1000,)
        np.testing.assert_array_equal(sliced_data, data[:1000])

    def test_load_slice_nonexistent(self, temp_store):
        """Test loading a slice of a nonexistent weight."""
        result = temp_store.load_slice("nonexistent_hash", slice(0, 10))
        assert result is None

    def test_iter_weights(self, temp_store):
        """Test iterating over weights in batches."""
        # Store 15 weights
        hashes = []
        for i in range(15):
            data = np.random.randn(100, 100).astype(np.float32)
            weight = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"weight_{i}", shape=data.shape, dtype=data.dtype
                ),
            )
            hashes.append(temp_store.store(weight))

        # Iterate with batch size of 5
        batches = list(temp_store.iter_weights(batch_size=5))

        assert len(batches) == 3  # 15 weights / 5 per batch = 3 batches
        assert len(batches[0]) == 5
        assert len(batches[1]) == 5
        assert len(batches[2]) == 5

        # Verify all weights are returned
        all_hashes = set()
        for batch in batches:
            all_hashes.update(batch.keys())
        assert all_hashes == set(hashes)

    def test_iter_metadata(self, temp_store):
        """Test iterating over metadata without loading weight data."""
        # Store weights
        for i in range(10):
            data = np.random.randn(50, 50).astype(np.float32)
            weight = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"meta_weight_{i}", shape=data.shape, dtype=data.dtype
                ),
            )
            temp_store.store(weight)

        # Iterate metadata with batch size 3
        batches = list(temp_store.iter_metadata(batch_size=3))

        assert (
            len(batches) == 4
        )  # 10 metadata / 3 per batch = 4 batches (with remainder)

        # Verify metadata content
        for batch in batches:
            for _hash_key, meta in batch.items():
                assert isinstance(meta, WeightMetadata)
                assert meta.name.startswith("meta_weight_")

    def test_get_weight_size(self, temp_store, large_weight):
        """Test getting weight size without loading data."""
        hash_key = temp_store.store(large_weight)

        size = temp_store.get_weight_size(hash_key)

        assert size is not None
        assert size == large_weight.nbytes

    def test_get_weight_size_nonexistent(self, temp_store):
        """Test getting size of nonexistent weight."""
        size = temp_store.get_weight_size("nonexistent_hash")
        assert size is None

    def test_estimate_memory_usage(self, temp_store):
        """Test estimating memory usage for weights."""
        # Store weights of varying sizes
        for i in range(5):
            size = (i + 1) * 100
            data = np.random.randn(size, size).astype(np.float32)
            weight = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"sized_weight_{i}", shape=data.shape, dtype=data.dtype
                ),
            )
            temp_store.store(weight)

        estimate = temp_store.estimate_memory_usage()

        assert estimate["num_weights"] == 5
        assert estimate["total_bytes"] > 0
        assert estimate["total_mb"] > 0
        assert estimate["total_gb"] >= 0
        assert len(estimate["largest_weights"]) == 5
        assert estimate["average_bytes"] > 0

    def test_load_by_pattern(self, temp_store):
        """Test loading weights matching a name pattern."""
        # Store weights with different naming patterns
        for i in range(3):
            data = np.random.randn(50, 50).astype(np.float32)
            weight = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"encoder.layer.{i}.weight", shape=data.shape, dtype=data.dtype
                ),
            )
            temp_store.store(weight)

        for i in range(3):
            data = np.random.randn(50, 50).astype(np.float32)
            weight = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"decoder.layer.{i}.weight", shape=data.shape, dtype=data.dtype
                ),
            )
            temp_store.store(weight)

        # Load only encoder weights
        encoder_weights = temp_store.load_by_pattern(r"encoder\..*")

        assert len(encoder_weights) == 3
        for weight in encoder_weights.values():
            assert "encoder" in weight.metadata.name

    def test_load_by_pattern_max_count(self, temp_store):
        """Test load_by_pattern with max_count limit."""
        for i in range(10):
            data = np.random.randn(50, 50).astype(np.float32)
            weight = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"layer.{i}.weight", shape=data.shape, dtype=data.dtype
                ),
            )
            temp_store.store(weight)

        weights = temp_store.load_by_pattern(r"layer\..*", max_count=3)

        assert len(weights) == 3


class TestWeightTensorLazyLoading:
    """Test WeightTensor lazy loading features."""

    def test_is_loaded_property(self):
        """Test is_loaded property for loaded weight."""
        data = np.random.randn(100, 100).astype(np.float32)
        weight = WeightTensor(data=data)
        assert weight.is_loaded is True

    def test_is_loaded_property_not_loaded(self):
        """Test is_loaded property for unloaded weight."""
        metadata = WeightMetadata(name="test", shape=(100, 100), dtype=np.float32)
        weight = WeightTensor(data=None, metadata=metadata, store_ref="test_ref")
        assert weight.is_loaded is False

    def test_can_lazy_load_property(self):
        """Test can_lazy_load property."""
        metadata = WeightMetadata(name="test", shape=(100, 100), dtype=np.float32)

        # Without lazy loader
        weight1 = WeightTensor(data=None, metadata=metadata, store_ref="test_ref")
        assert weight1.can_lazy_load is False

        # With lazy loader
        def dummy_loader(ref: str) -> np.ndarray:
            return np.zeros((100, 100), dtype=np.float32)

        weight2 = WeightTensor(
            data=None, metadata=metadata, store_ref="test_ref", lazy_loader=dummy_loader
        )
        assert weight2.can_lazy_load is True

    def test_lazy_loading(self):
        """Test lazy loading of weight data."""
        expected_data = np.random.randn(100, 100).astype(np.float32)

        def loader(ref: str) -> np.ndarray:
            return expected_data.copy()

        metadata = WeightMetadata(
            name="lazy_test", shape=expected_data.shape, dtype=expected_data.dtype
        )
        weight = WeightTensor(
            data=None, metadata=metadata, store_ref="test_ref", lazy_loader=loader
        )

        assert not weight.is_loaded

        # Access data triggers loading
        data = weight.data

        assert weight.is_loaded
        np.testing.assert_array_equal(data, expected_data)

    def test_unload(self):
        """Test unloading weight data from memory."""
        expected_data = np.random.randn(100, 100).astype(np.float32)

        def loader(ref: str) -> np.ndarray:
            return expected_data.copy()

        metadata = WeightMetadata(
            name="unload_test", shape=expected_data.shape, dtype=expected_data.dtype
        )
        weight = WeightTensor(
            data=expected_data.copy(),
            metadata=metadata,
            store_ref="test_ref",
            lazy_loader=loader,
        )

        assert weight.is_loaded

        # Unload
        result = weight.unload()

        assert result is True
        assert not weight.is_loaded

        # Data can be loaded again
        data = weight.data
        assert weight.is_loaded
        np.testing.assert_array_equal(data, expected_data)

    def test_unload_without_lazy_loader(self):
        """Test that unload fails without lazy loader."""
        data = np.random.randn(100, 100).astype(np.float32)
        weight = WeightTensor(data=data)

        result = weight.unload()

        assert result is False
        assert weight.is_loaded

    def test_set_lazy_loader(self):
        """Test setting lazy loader after creation."""
        metadata = WeightMetadata(name="test", shape=(100, 100), dtype=np.float32)
        weight = WeightTensor(data=None, metadata=metadata, store_ref="test_ref")

        assert not weight.can_lazy_load

        def loader(ref: str) -> np.ndarray:
            return np.zeros((100, 100), dtype=np.float32)

        weight.set_lazy_loader(loader)

        assert weight.can_lazy_load

    def test_create_lazy_classmethod(self):
        """Test creating a lazy-loading weight tensor."""
        expected_data = np.random.randn(100, 100).astype(np.float32)

        def loader(ref: str) -> np.ndarray:
            return expected_data.copy()

        metadata = WeightMetadata(
            name="lazy_create_test",
            shape=expected_data.shape,
            dtype=expected_data.dtype,
        )

        weight = WeightTensor.create_lazy(metadata, "test_ref", loader)

        assert not weight.is_loaded
        assert weight.can_lazy_load
        assert weight.store_ref == "test_ref"

        # Access data triggers loading
        data = weight.data
        assert weight.is_loaded
        np.testing.assert_array_equal(data, expected_data)

    def test_ensure_loaded(self):
        """Test ensure_loaded method."""
        expected_data = np.random.randn(100, 100).astype(np.float32)

        def loader(ref: str) -> np.ndarray:
            return expected_data.copy()

        metadata = WeightMetadata(
            name="ensure_test", shape=expected_data.shape, dtype=expected_data.dtype
        )
        weight = WeightTensor(
            data=None, metadata=metadata, store_ref="test_ref", lazy_loader=loader
        )

        assert not weight.is_loaded

        weight.ensure_loaded()

        assert weight.is_loaded

    def test_repr_shows_loaded_status(self):
        """Test that repr shows whether weight is loaded."""
        data = np.random.randn(10, 10).astype(np.float32)

        loaded_weight = WeightTensor(data=data)
        assert "loaded" in repr(loaded_weight)
        assert "not loaded" not in repr(loaded_weight)

        metadata = WeightMetadata(name="test", shape=(10, 10), dtype=np.float32)
        unloaded_weight = WeightTensor(data=None, metadata=metadata, store_ref="ref")
        assert "not loaded" in repr(unloaded_weight)


class TestDeduplicatorWithLSH:
    """Test that LSH is enabled by default."""

    def test_lsh_enabled_by_default(self):
        """Test that LSH is enabled by default for O(1) similarity search."""
        from coral.core.deduplicator import Deduplicator

        dedup = Deduplicator()
        assert dedup.enable_lsh is True
        assert dedup.lsh_index is not None


class TestChunkingLogic:
    """Test the chunking computation logic."""

    def test_small_weight_no_chunking(self):
        """Test that small weights don't get chunked."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        store = HDF5Store(filepath, chunk_size=1_000_000)

        # Small weight (100 elements << chunk_size)
        chunks = store._compute_chunks((100,))
        assert chunks is None

        store.close()
        Path(filepath).unlink(missing_ok=True)

    def test_large_1d_weight_chunking(self):
        """Test chunking for large 1D weight."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        store = HDF5Store(filepath, chunk_size=1000)

        # Large 1D weight
        chunks = store._compute_chunks((10000,))
        assert chunks is not None
        assert chunks == (1000,)

        store.close()
        Path(filepath).unlink(missing_ok=True)

    def test_large_2d_weight_chunking(self):
        """Test chunking for large 2D weight."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = f.name

        store = HDF5Store(filepath, chunk_size=1000)

        # Large 2D weight (1000x100 = 100k elements)
        # chunks along first dimension
        chunks = store._compute_chunks((1000, 100))
        assert chunks is not None
        # chunk_size / elements_per_slice = 1000 / 100 = 10
        assert chunks[0] == 10
        assert chunks[1] == 100

        store.close()
        Path(filepath).unlink(missing_ok=True)
