"""Tests for lazy weight loading functionality."""

import tempfile
from pathlib import Path

import numpy as np

from coral.core.lazy_weight import (
    LazyLoadConfig,
    LazyWeightCollection,
    StreamingWeightIterator,
    WeightProxy,
)
from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.storage.hdf5_store import HDF5Store


class TestWeightProxy:
    """Tests for WeightProxy lazy loading."""

    def test_proxy_creation_from_weight(self):
        """Test creating a proxy from an existing weight."""
        data = np.random.randn(100, 50).astype(np.float32)
        metadata = WeightMetadata(name="test_weight", shape=(100, 50), dtype=np.float32)
        weight = WeightTensor(data=data, metadata=metadata)

        proxy = WeightProxy.from_weight(weight)

        assert proxy.name == "test_weight"
        assert proxy.shape == (100, 50)
        assert proxy.dtype == np.float32
        assert proxy.is_loaded  # Data was already loaded
        np.testing.assert_array_equal(proxy.data, data)

    def test_proxy_lazy_loading_from_store(self):
        """Test lazy loading from HDF5 store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "weights.h5"
            store = HDF5Store(str(store_path))

            # Store a weight
            data = np.random.randn(64, 32).astype(np.float32)
            metadata = WeightMetadata(
                name="lazy_weight", shape=(64, 32), dtype=np.float32
            )
            weight = WeightTensor(data=data, metadata=metadata)
            hash_key = store.store(weight)

            # Create proxy without loading data
            proxy = WeightProxy.from_store(store, hash_key)

            # Verify metadata is available without loading
            assert proxy.name == "lazy_weight"
            assert proxy.shape == (64, 32)
            assert not proxy.is_loaded

            # Access data triggers loading
            loaded_data = proxy.data
            assert proxy.is_loaded
            np.testing.assert_array_equal(loaded_data, data)

            store.close()

    def test_proxy_unload(self):
        """Test unloading data from proxy."""
        data = np.random.randn(50, 25).astype(np.float32)
        metadata = WeightMetadata(name="unload_test", shape=(50, 25), dtype=np.float32)

        # Create proxy with a load function
        proxy = WeightProxy(
            metadata=metadata,
            hash_key="test_hash",
            load_fn=lambda: data.copy(),
        )

        # Load data
        _ = proxy.data
        assert proxy.is_loaded

        # Unload
        proxy.unload()
        assert not proxy.is_loaded

        # Reload on next access
        reloaded = proxy.data
        assert proxy.is_loaded
        np.testing.assert_array_equal(reloaded, data)

    def test_proxy_materialize(self):
        """Test materializing proxy to full WeightTensor."""
        data = np.random.randn(32, 16).astype(np.float32)
        metadata = WeightMetadata(
            name="materialize_test", shape=(32, 16), dtype=np.float32
        )

        proxy = WeightProxy(
            metadata=metadata,
            hash_key="mat_hash",
            load_fn=lambda: data.copy(),
        )

        weight = proxy.materialize()

        assert isinstance(weight, WeightTensor)
        assert weight.metadata.name == "materialize_test"
        np.testing.assert_array_equal(weight.data, data)

    def test_proxy_nbytes_without_loading(self):
        """Test that nbytes can be calculated without loading data."""
        metadata = WeightMetadata(
            name="nbytes_test", shape=(1000, 768), dtype=np.float32
        )

        proxy = WeightProxy(
            metadata=metadata,
            hash_key="nbytes_hash",
            load_fn=lambda: np.zeros((1000, 768), dtype=np.float32),
        )

        # Check nbytes without loading
        assert not proxy.is_loaded
        expected_bytes = 1000 * 768 * 4  # float32 = 4 bytes
        assert proxy.nbytes == expected_bytes
        assert not proxy.is_loaded  # Still not loaded


class TestLazyWeightCollection:
    """Tests for LazyWeightCollection."""

    def test_collection_add_and_get(self):
        """Test adding and retrieving weights from collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "weights.h5"
            store = HDF5Store(str(store_path))

            # Store some weights
            weights = {}
            for i in range(5):
                data = np.random.randn(32, 16).astype(np.float32)
                metadata = WeightMetadata(
                    name=f"weight_{i}", shape=(32, 16), dtype=np.float32
                )
                weight = WeightTensor(data=data, metadata=metadata)
                hash_key = store.store(weight)
                weights[f"weight_{i}"] = (hash_key, data)

            # Create collection
            collection = LazyWeightCollection(store=store)

            # Add weights to collection
            for name, (hash_key, _) in weights.items():
                collection.add(name, hash_key)

            assert len(collection) == 5

            # Get a proxy
            proxy = collection.get("weight_2")
            assert proxy is not None
            assert proxy.name == "weight_2"
            assert not proxy.is_loaded

            store.close()

    def test_collection_memory_limits(self):
        """Test LRU eviction when memory limits are exceeded."""
        config = LazyLoadConfig(
            max_cache_bytes=1024 * 10,  # 10KB limit
            max_cached_weights=3,
        )

        collection = LazyWeightCollection(config=config)

        # Add weights that exceed the limit
        for i in range(5):
            data = np.random.randn(32, 32).astype(np.float32)  # ~4KB each
            metadata = WeightMetadata(
                name=f"mem_weight_{i}", shape=(32, 32), dtype=np.float32
            )
            weight = WeightTensor(data=data, metadata=metadata)
            collection.add_weight(f"weight_{i}", weight)

        stats = collection.get_memory_stats()
        assert stats["total_weights"] == 5

    def test_collection_unload_all(self):
        """Test unloading all cached data."""
        collection = LazyWeightCollection()

        # Add some weights with data
        for i in range(3):
            data = np.random.randn(16, 16).astype(np.float32)
            metadata = WeightMetadata(
                name=f"unload_all_{i}", shape=(16, 16), dtype=np.float32
            )
            weight = WeightTensor(data=data, metadata=metadata)
            collection.add_weight(f"w_{i}", weight)

        # Verify data is loaded
        assert all(p.is_loaded for p in collection.values())

        # Unload all
        collection.unload_all()

        # Memory stats should show 0 cached
        stats = collection.get_memory_stats()
        assert stats["loaded_weights"] == 0
        assert stats["cached_bytes"] == 0

    def test_collection_iteration(self):
        """Test iterating over collection."""
        collection = LazyWeightCollection()

        names = ["layer1", "layer2", "layer3"]
        for name in names:
            data = np.random.randn(8, 8).astype(np.float32)
            metadata = WeightMetadata(name=name, shape=(8, 8), dtype=np.float32)
            weight = WeightTensor(data=data, metadata=metadata)
            collection.add_weight(name, weight)

        # Test keys
        assert list(collection.keys()) == names

        # Test items
        for name, proxy in collection.items():
            assert name in names
            assert isinstance(proxy, WeightProxy)

        # Test __contains__
        assert "layer1" in collection
        assert "nonexistent" not in collection


class TestStreamingWeightIterator:
    """Tests for StreamingWeightIterator."""

    def test_streaming_iteration(self):
        """Test streaming iteration over weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "weights.h5"
            store = HDF5Store(str(store_path))

            # Store multiple weights
            stored_hashes = []
            for i in range(10):
                data = np.random.randn(16, 8).astype(np.float32)
                metadata = WeightMetadata(
                    name=f"stream_{i}", shape=(16, 8), dtype=np.float32
                )
                weight = WeightTensor(data=data, metadata=metadata)
                hash_key = store.store(weight)
                stored_hashes.append(hash_key)

            # Create iterator
            iterator = StreamingWeightIterator(store, batch_size=3)

            # Iterate and count
            count = 0
            for weight in iterator:
                assert isinstance(weight, WeightTensor)
                count += 1

            assert count == 10
            assert len(iterator) == 10

            store.close()

    def test_streaming_with_filter(self):
        """Test filtering during streaming."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "weights.h5"
            store = HDF5Store(str(store_path))

            # Store weights with different shapes
            for i in range(5):
                shape = (32, 16) if i % 2 == 0 else (64, 32)
                data = np.random.randn(*shape).astype(np.float32)
                metadata = WeightMetadata(
                    name=f"filter_{i}", shape=shape, dtype=np.float32
                )
                weight = WeightTensor(data=data, metadata=metadata)
                store.store(weight)

            # Create iterator with metadata preloading
            iterator = StreamingWeightIterator(store, preload_metadata=True)

            # Filter for only 32x16 weights
            filtered = iterator.filter(lambda m: m.shape == (32, 16))

            count = sum(1 for _ in filtered)
            assert count == 3  # i = 0, 2, 4

            store.close()


class TestWeightTensorLazyLoading:
    """Tests for lazy loading in WeightTensor itself."""

    def test_weight_tensor_with_load_fn(self):
        """Test WeightTensor with custom load function."""
        data = np.random.randn(50, 25).astype(np.float32)
        load_count = [0]

        def load_fn():
            load_count[0] += 1
            return data.copy()

        metadata = WeightMetadata(name="lazy_tensor", shape=(50, 25), dtype=np.float32)

        weight = WeightTensor(metadata=metadata, store_ref="test_ref", load_fn=load_fn)

        # Data not loaded yet
        assert not weight.is_loaded
        assert load_count[0] == 0

        # Access data
        _ = weight.data
        assert weight.is_loaded
        assert load_count[0] == 1

        # Second access shouldn't reload
        _ = weight.data
        assert load_count[0] == 1

    def test_weight_tensor_unload_reload(self):
        """Test unloading and reloading data."""
        data = np.random.randn(20, 10).astype(np.float32)

        metadata = WeightMetadata(
            name="unload_reload", shape=(20, 10), dtype=np.float32
        )

        weight = WeightTensor(
            metadata=metadata,
            store_ref="ref",
            load_fn=lambda: data.copy(),
        )

        # Load data
        first_load = weight.data.copy()
        assert weight.is_loaded

        # Unload
        weight.unload()
        assert not weight.is_loaded

        # Reload
        second_load = weight.data
        np.testing.assert_array_equal(first_load, second_load)

    def test_weight_tensor_with_store(self):
        """Test WeightTensor lazy loading from store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "weights.h5"
            store = HDF5Store(str(store_path))

            # Create and store a weight
            data = np.random.randn(40, 20).astype(np.float32)
            metadata = WeightMetadata(
                name="store_lazy", shape=(40, 20), dtype=np.float32
            )
            original = WeightTensor(data=data, metadata=metadata)
            hash_key = store.store(original)

            # Create new weight tensor with store reference
            lazy_weight = WeightTensor(metadata=metadata, store_ref=hash_key)
            lazy_weight.set_store(store)

            # Verify lazy loading works
            assert not lazy_weight.is_loaded
            loaded_data = lazy_weight.data
            assert lazy_weight.is_loaded
            np.testing.assert_array_equal(loaded_data, data)

            store.close()
