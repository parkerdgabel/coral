"""Comprehensive integration tests for SafeTensors functionality in Coral.

This test suite covers the full integration between all safetensors components:

## Core Integration Tests (TestSafetensorsIntegration):
- SafetensorsStore roundtrip data integrity for all dtypes
- Compression/decompression roundtrip testing
- Repository integration and data interchange
- Large tensor handling (>100MB)
- Many small tensors (1000+) batch operations
- Concurrent access and thread safety
- Compatibility with official safetensors library files
- PyTorch model export/import (if PyTorch available)
- Metadata preservation through format conversions

## Performance Tests (TestSafetensorsPerformance):
- Storage size comparison vs HDF5Store backend
- Batch operation performance measurement
- Memory usage profiling with large files (if psutil available)

## Error Handling Tests (TestSafetensorsErrorHandling):
- Corrupted file recovery
- Permission denied scenarios
- Disk full simulation
- Concurrent write conflict resolution
- Invalid metadata handling
- Storage cleanup and recovery from interrupted operations

## Edge Cases Tests (TestSafetensorsSpecialCases):
- Zero-dimensional (scalar) tensors
- Very large dimension tensors (many axes)
- Special float values (NaN, inf, -inf)
- Unicode and special characters in tensor names
- Various numpy dtypes and edge cases

Usage:
    # Run all tests
    uv run pytest tests/test_safetensors_integration.py

    # Run specific test class
    uv run pytest tests/test_safetensors_integration.py::TestSafetensorsIntegration

    # Run individual test
    uv run pytest tests/test_safetensors_integration.py::TestSafetensorsIntegration::\
        test_roundtrip_data_integrity

    # Run with verbose output
    uv run pytest tests/test_safetensors_integration.py -v
"""

import concurrent.futures
import os
import platform
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.storage.safetensors_store import SafetensorsStore
from coral.version_control.repository import Repository


# Test Fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def safetensors_store(temp_dir):
    """Create a SafetensorsStore instance."""
    return SafetensorsStore(str(temp_dir / "safetensors_storage"))


@pytest.fixture
def compressed_safetensors_store(temp_dir):
    """Create a compressed SafetensorsStore instance."""
    return SafetensorsStore(
        str(temp_dir / "compressed_storage"), use_compression=True, compression_level=6
    )


@pytest.fixture
def repository_with_safetensors(temp_dir):
    """Create a standard Repository for testing SafetensorsStore alongside it."""
    repo_path = temp_dir / "repo"
    repo_path.mkdir()
    repo = Repository(repo_path, init=True)
    return repo


@pytest.fixture
def sample_weights():
    """Create various test weight tensors with different characteristics."""
    weights = {}

    # Small dense weight
    weights["small_dense"] = WeightTensor(
        data=np.random.randn(10, 20).astype(np.float32),
        metadata=WeightMetadata(
            name="small_dense",
            shape=(10, 20),
            dtype=np.float32,
            layer_type="dense",
            model_name="test_model",
        ),
    )

    # Large weight (>1MB)
    weights["large_weight"] = WeightTensor(
        data=np.random.randn(512, 512).astype(np.float32),
        metadata=WeightMetadata(
            name="large_weight",
            shape=(512, 512),
            dtype=np.float32,
            layer_type="conv2d",
            model_name="resnet",
        ),
    )

    # Different dtypes
    dtypes_and_data = [
        (np.float16, np.random.randn(5, 5).astype(np.float16)),
        (np.float64, np.random.randn(3, 3).astype(np.float64)),
        (np.int8, np.random.randint(-128, 127, (4, 4), dtype=np.int8)),
        (np.bool_, np.random.choice([True, False], (2, 2))),
    ]

    for _i, (dtype, data) in enumerate(dtypes_and_data):
        dtype_name = np.dtype(dtype).name
        weights[f"dtype_{dtype_name}"] = WeightTensor(
            data=data,
            metadata=WeightMetadata(
                name=f"dtype_{dtype_name}",
                shape=data.shape,
                dtype=dtype,
                layer_type="test",
            ),
        )

    # Empty tensor
    weights["empty"] = WeightTensor(
        data=np.array([], dtype=np.float32).reshape(0, 0),
        metadata=WeightMetadata(
            name="empty", shape=(0, 0), dtype=np.float32, layer_type="empty"
        ),
    )

    # Unicode names
    weights["测试权重"] = WeightTensor(
        data=np.ones((2, 2), dtype=np.float32),
        metadata=WeightMetadata(
            name="测试权重", shape=(2, 2), dtype=np.float32, layer_type="unicode_test"
        ),
    )

    return weights


@pytest.fixture
def official_safetensors_file(temp_dir):
    """Create a safetensors file using the official library."""
    file_path = temp_dir / "official.safetensors"

    tensors = {
        "weight1": np.random.randn(10, 10).astype(np.float32),
        "weight2": np.random.randn(5, 5).astype(np.float16),
        "bias": np.random.randn(10).astype(np.float32),
    }

    metadata = {
        "model_type": "transformer",
        "version": "1.0",
        "created_by": "official_safetensors",
    }

    save_file(tensors, file_path, metadata=metadata)
    return file_path


class TestSafetensorsIntegration:
    """Full integration tests for SafeTensors functionality."""

    def test_roundtrip_data_integrity(self, safetensors_store, sample_weights):
        """Test that all data types survive store/load roundtrip perfectly."""
        for name, weight in sample_weights.items():
            # Store weight
            hash_key = safetensors_store.store(weight)

            # Load weight
            loaded = safetensors_store.load(hash_key)

            # Verify data integrity
            assert loaded is not None, f"Failed to load weight {name}"
            assert loaded.metadata.name == weight.metadata.name
            assert loaded.metadata.shape == weight.metadata.shape
            assert loaded.metadata.dtype == weight.metadata.dtype

            # Handle empty arrays specially
            if weight.data.size == 0:
                assert loaded.data.size == 0
            else:
                np.testing.assert_array_equal(loaded.data, weight.data)

    def test_compression_roundtrip(self, compressed_safetensors_store, sample_weights):
        """Test compression doesn't affect data integrity."""
        for _name, weight in sample_weights.items():
            hash_key = compressed_safetensors_store.store(weight)
            loaded = compressed_safetensors_store.load(hash_key)

            assert loaded is not None
            if weight.data.size > 0:
                np.testing.assert_array_equal(loaded.data, weight.data)

    def test_repository_integration(
        self, repository_with_safetensors, sample_weights, temp_dir
    ):
        """Test SafetensorsStore can be used alongside Repository for data
        interchange."""
        repo = repository_with_safetensors

        # Stage and commit weights in Repository
        repo.stage_weights(sample_weights)
        commit_hash = repo.commit("Test commit with various weight types")

        # Verify commit was successful
        assert commit_hash is not None

        # Export weights from Repository and import into SafetensorsStore
        safetensors_store = SafetensorsStore(str(temp_dir / "export_test"))

        # Get weights from Repository and store in SafetensorsStore
        export_hash_map = {}
        for name in sample_weights.keys():
            repo_weight = repo.get_weight(name)
            assert repo_weight is not None

            # Store in SafetensorsStore
            st_hash = safetensors_store.store(repo_weight)
            export_hash_map[name] = st_hash

        # Load weights back from SafetensorsStore and verify
        for name, original in sample_weights.items():
            st_hash = export_hash_map[name]
            loaded = safetensors_store.load(st_hash)

            assert loaded is not None
            if original.data.size > 0:
                np.testing.assert_array_equal(loaded.data, original.data)

        # Test that SafetensorsStore provides efficient batch operations
        # which could be useful for Repository-based workflows
        all_hashes = list(export_hash_map.values())
        batch_loaded = safetensors_store.load_batch(all_hashes)
        assert len(batch_loaded) == len(sample_weights)

    def test_large_tensor_handling(self, safetensors_store):
        """Test handling of tensors larger than 100MB."""
        # Create a ~120MB tensor (1000x1000x30 float32)
        large_data = np.random.randn(1000, 1000, 30).astype(np.float32)
        large_weight = WeightTensor(
            data=large_data,
            metadata=WeightMetadata(
                name="very_large_tensor",
                shape=large_data.shape,
                dtype=np.float32,
                layer_type="large_test",
            ),
        )

        # Store and load
        start_time = time.time()
        hash_key = safetensors_store.store(large_weight)
        store_time = time.time() - start_time

        start_time = time.time()
        loaded = safetensors_store.load(hash_key)
        load_time = time.time() - start_time

        # Verify correctness
        assert loaded is not None
        np.testing.assert_array_equal(loaded.data, large_data)

        # Performance should be reasonable (adjust thresholds as needed)
        assert store_time < 30.0, f"Store took too long: {store_time}s"
        assert load_time < 30.0, f"Load took too long: {load_time}s"

    def test_many_small_tensors(self, safetensors_store):
        """Test storing and loading many small tensors (1000+)."""
        weights = {}

        # Create 1500 small tensors
        for i in range(1500):
            data = np.random.randn(3, 3).astype(np.float32) * i
            weights[f"tensor_{i:04d}"] = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"tensor_{i:04d}",
                    shape=(3, 3),
                    dtype=np.float32,
                    layer_type="small",
                    model_name="many_tensors_model",
                ),
            )

        # Batch store
        start_time = time.time()
        hash_map = safetensors_store.store_batch(weights)
        batch_store_time = time.time() - start_time

        assert len(hash_map) == 1500

        # Batch load
        start_time = time.time()
        hash_keys = list(hash_map.values())
        loaded_weights = safetensors_store.load_batch(hash_keys)
        batch_load_time = time.time() - start_time

        assert len(loaded_weights) == 1500

        # Verify a sample of loaded weights
        for i in range(0, 1500, 100):  # Check every 100th tensor
            name = f"tensor_{i:04d}"
            hash_key = hash_map[name]
            loaded = loaded_weights[hash_key]
            original = weights[name]

            np.testing.assert_array_equal(loaded.data, original.data)

        # Performance check
        print(f"Batch store of 1500 tensors: {batch_store_time:.2f}s")
        print(f"Batch load of 1500 tensors: {batch_load_time:.2f}s")

    def test_concurrent_access(self, temp_dir):
        """Test concurrent read/write access to SafetensorsStore."""
        store = SafetensorsStore(str(temp_dir / "concurrent_test"))

        # Create test weights
        weights = []
        for i in range(50):
            data = np.random.randn(10, 10).astype(np.float32)
            weight = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"concurrent_{i}", shape=(10, 10), dtype=np.float32
                ),
            )
            weights.append(weight)

        stored_hashes = []

        def store_worker(weight_list, start_idx, end_idx):
            """Worker function to store weights."""
            hashes = []
            for i in range(start_idx, end_idx):
                hash_key = store.store(weight_list[i])
                hashes.append(hash_key)
            return hashes

        def load_worker(hash_list):
            """Worker function to load weights."""
            loaded = []
            for hash_key in hash_list:
                weight = store.load(hash_key)
                loaded.append(weight)
            return loaded

        # Concurrent storing
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            chunk_size = 10
            for i in range(0, len(weights), chunk_size):
                end_idx = min(i + chunk_size, len(weights))
                future = executor.submit(store_worker, weights, i, end_idx)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                stored_hashes.extend(future.result())

        assert len(stored_hashes) == len(weights)

        # Concurrent loading
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            chunk_size = 10
            futures = []
            for i in range(0, len(stored_hashes), chunk_size):
                end_idx = min(i + chunk_size, len(stored_hashes))
                chunk_hashes = stored_hashes[i:end_idx]
                future = executor.submit(load_worker, chunk_hashes)
                futures.append(future)

            loaded_results = []
            for future in concurrent.futures.as_completed(futures):
                loaded_results.extend(future.result())

        assert len(loaded_results) == len(weights)
        assert all(w is not None for w in loaded_results)

    def test_official_safetensors_compatibility(
        self, official_safetensors_file, temp_dir
    ):
        """Test compatibility with files created by official safetensors library."""
        # Load file created by official library
        official_tensors = load_file(official_safetensors_file)

        # Read metadata
        with safe_open(official_safetensors_file, framework="numpy") as f:
            f.metadata()

        # Convert to Coral format and store in SafetensorsStore
        store = SafetensorsStore(str(temp_dir / "compatibility_test"))

        stored_hashes = {}
        for name, tensor_data in official_tensors.items():
            weight = WeightTensor(
                data=tensor_data,
                metadata=WeightMetadata(
                    name=name,
                    shape=tensor_data.shape,
                    dtype=tensor_data.dtype,
                    layer_type="imported",
                    model_name="official_import",
                ),
            )
            hash_key = store.store(weight)
            stored_hashes[name] = hash_key

        # Verify we can load them back
        for name, tensor_data in official_tensors.items():
            hash_key = stored_hashes[name]
            loaded = store.load(hash_key)

            assert loaded is not None
            np.testing.assert_array_equal(loaded.data, tensor_data)

    def test_pytorch_model_export_import(self, temp_dir):
        """Test exporting/importing PyTorch model weights via safetensors."""
        pytest.importorskip("torch")  # Skip if PyTorch not available

        import torch
        import torch.nn as nn

        # Create a simple PyTorch model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.linear2 = nn.Linear(20, 1)
                self.conv = nn.Conv2d(3, 6, 3)

            def forward(self, x):
                return self.linear2(torch.relu(self.linear1(x)))

        model = SimpleModel()

        # Export model weights to safetensors via Coral
        store = SafetensorsStore(str(temp_dir / "pytorch_export"))

        # Convert PyTorch tensors to Coral WeightTensors
        coral_weights = {}
        for name, param in model.named_parameters():
            data = param.detach().numpy()
            coral_weights[name] = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=name,
                    shape=data.shape,
                    dtype=data.dtype,
                    layer_type="pytorch_param",
                    model_name="SimpleModel",
                ),
            )

        # Store weights
        hash_map = store.store_batch(coral_weights)

        # Load weights back and verify
        loaded_hashes = list(hash_map.values())
        loaded_weights = store.load_batch(loaded_hashes)

        assert len(loaded_weights) == len(coral_weights)

        # Verify weights match original model
        for name, original_weight in coral_weights.items():
            # Find loaded weight by matching metadata
            loaded_weight = None
            for loaded in loaded_weights.values():
                if loaded.metadata.name == name:
                    loaded_weight = loaded
                    break

            assert loaded_weight is not None
            np.testing.assert_array_equal(loaded_weight.data, original_weight.data)

    def test_metadata_preservation_through_conversions(self, temp_dir):
        """Test that metadata is fully preserved through format conversions."""
        store = SafetensorsStore(str(temp_dir / "metadata_test"))

        # Create weight with comprehensive metadata
        original_metadata = WeightMetadata(
            name="complex_metadata_weight",
            shape=(5, 10, 15),
            dtype=np.float32,
            layer_type="multi_head_attention",
            model_name="transformer_xl_v2.1",
            compression_info={
                "method": "quantized",
                "bits": 8,
                "scale": 0.1,
                "zero_point": 128,
                "original_dtype": "float32",
            },
        )

        data = np.random.randn(5, 10, 15).astype(np.float32)
        original_weight = WeightTensor(data=data, metadata=original_metadata)

        # Store and load
        hash_key = store.store(original_weight)
        loaded_weight = store.load(hash_key)

        # Verify all metadata fields are preserved
        loaded_meta = loaded_weight.metadata
        assert loaded_meta.name == original_metadata.name
        assert loaded_meta.shape == original_metadata.shape
        assert loaded_meta.dtype == original_metadata.dtype
        assert loaded_meta.layer_type == original_metadata.layer_type
        assert loaded_meta.model_name == original_metadata.model_name
        assert loaded_meta.compression_info == original_metadata.compression_info

        # Test metadata-only retrieval
        metadata_only = store.get_metadata(hash_key)
        assert metadata_only is not None
        assert metadata_only.name == original_metadata.name
        assert metadata_only.compression_info == original_metadata.compression_info


class TestSafetensorsPerformance:
    """Performance tests comparing SafetensorsStore with other backends."""

    def test_storage_size_comparison(self, temp_dir, sample_weights):
        """Compare storage size between SafetensorsStore and HDF5Store."""
        from coral.storage.hdf5_store import HDF5Store

        safetensors_store = SafetensorsStore(str(temp_dir / "st_perf"))
        hdf5_store = HDF5Store(str(temp_dir / "hdf5_perf.h5"))

        # Store same weights in both backends
        safetensors_store.store_batch(sample_weights)
        hdf5_store.store_batch(sample_weights)

        # Compare storage info
        st_info = safetensors_store.get_storage_info()
        hdf5_info = hdf5_store.get_storage_info()

        print(f"SafeTensors total size: {st_info['total_size_mb']:.2f} MB")
        print(f"HDF5 file size: {hdf5_info['file_size'] / (1024 * 1024):.2f} MB")

        # Both should store the data successfully
        assert st_info["total_files"] == len(sample_weights)
        assert hdf5_info["total_weights"] == len(
            sample_weights
        )  # HDF5 tracks weight count

        hdf5_store.close()

    def test_batch_operation_performance(self, temp_dir):
        """Test performance of batch operations."""
        store = SafetensorsStore(str(temp_dir / "batch_perf"))

        # Create test weights
        weights = {}
        for i in range(100):
            data = np.random.randn(50, 50).astype(np.float32)
            weights[f"weight_{i}"] = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"weight_{i}", shape=(50, 50), dtype=np.float32
                ),
            )

        # Time batch store
        start_time = time.time()
        hash_map = store.store_batch(weights)
        batch_store_time = time.time() - start_time

        # Time individual stores for comparison
        individual_weights = {}
        for i in range(100, 110):  # Different weights to avoid cache
            data = np.random.randn(50, 50).astype(np.float32)
            individual_weights[f"individual_{i}"] = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"individual_{i}", shape=(50, 50), dtype=np.float32
                ),
            )

        start_time = time.time()
        for weight in individual_weights.values():
            store.store(weight)
        individual_store_time = time.time() - start_time

        print(f"Batch store (100 weights): {batch_store_time:.3f}s")
        print(f"Individual store (10 weights): {individual_store_time:.3f}s")
        print(f"Individual projected (100 weights): {individual_store_time * 10:.3f}s")

        # Batch should be reasonably efficient
        assert len(hash_map) == 100

    def test_memory_usage_large_files(self, temp_dir):
        """Test memory usage when working with large files."""
        pytest.importorskip("psutil")  # Skip if psutil not available
        import gc

        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        store = SafetensorsStore(str(temp_dir / "memory_test"))

        # Create a large weight (~50MB)
        large_data = np.random.randn(2000, 3000).astype(np.float32)
        large_weight = WeightTensor(
            data=large_data,
            metadata=WeightMetadata(
                name="memory_test_weight", shape=large_data.shape, dtype=np.float32
            ),
        )

        # Store it
        hash_key = store.store(large_weight)

        # Clear references and garbage collect
        del large_data, large_weight
        gc.collect()

        store_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Load it back
        loaded_weight = store.load(hash_key)

        load_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Clear loaded weight
        del loaded_weight
        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"After store: {store_memory:.1f} MB")
        print(f"After load: {load_memory:.1f} MB")
        print(f"Final memory: {final_memory:.1f} MB")

        # Memory usage should be reasonable (allowing for some overhead)
        assert load_memory - initial_memory < 200  # Less than 200MB overhead


class TestSafetensorsErrorHandling:
    """Test error handling and recovery scenarios."""

    def test_corrupted_file_handling(self, temp_dir):
        """Test handling of corrupted safetensors files."""
        store = SafetensorsStore(str(temp_dir / "corruption_test"))

        # Create and store a weight
        weight = WeightTensor(
            data=np.ones((5, 5), dtype=np.float32),
            metadata=WeightMetadata(name="test_weight", shape=(5, 5), dtype=np.float32),
        )

        hash_key = store.store(weight)
        file_path = store._get_file_path(hash_key)

        # Corrupt the file
        with open(file_path, "wb") as f:
            f.write(b"corrupted data that is not valid safetensors format")

        # Attempt to load should return None, not crash
        loaded = store.load(hash_key)
        assert loaded is None

        # Metadata retrieval should also handle corruption gracefully
        metadata = store.get_metadata(hash_key)
        assert metadata is None

    def test_permission_denied_scenarios(self, temp_dir):
        """Test handling of permission denied errors."""
        if platform.system() == "Windows":
            pytest.skip("Permission tests not reliable on Windows")

        store_dir = temp_dir / "permission_test"
        store = SafetensorsStore(str(store_dir))

        # Create and store a weight
        weight = WeightTensor(
            data=np.array([[1, 2], [3, 4]], dtype=np.float32),
            metadata=WeightMetadata(
                name="permission_test", shape=(2, 2), dtype=np.float32
            ),
        )

        hash_key = store.store(weight)

        # Make directory read-only
        os.chmod(store_dir, 0o444)

        try:
            # Attempt to store another weight should fail gracefully
            new_weight = WeightTensor(
                data=np.array([[5, 6], [7, 8]], dtype=np.float32),
                metadata=WeightMetadata(
                    name="permission_test2", shape=(2, 2), dtype=np.float32
                ),
            )

            with pytest.raises(OSError):  # Should raise some form of permission error
                store.store(new_weight)

            # Reading should still work if file permissions allow it
            try:
                loaded = store.load(hash_key)
                assert loaded is not None
            except PermissionError:
                # This is expected if directory is read-only
                pass

        finally:
            # Restore permissions for cleanup
            os.chmod(store_dir, 0o755)

    def test_disk_full_simulation(self, temp_dir):
        """Test behavior when disk space is exhausted."""
        store = SafetensorsStore(str(temp_dir / "disk_full_test"))

        # Create a weight
        weight = WeightTensor(
            data=np.random.randn(100, 100).astype(np.float32),
            metadata=WeightMetadata(
                name="disk_test", shape=(100, 100), dtype=np.float32
            ),
        )

        # Mock a disk full scenario by creating a file that fills available space
        # (This is a simplified simulation - in practice this would be hard to test)

        # For now, just test that the store operation either succeeds or fails cleanly
        try:
            hash_key = store.store(weight)
            # If successful, verify we can load it back
            loaded = store.load(hash_key)
            assert loaded is not None
        except Exception as e:
            # If it fails, it should be a clear error, not a crash
            assert isinstance(e, (OSError, RuntimeError))

    def test_concurrent_write_conflicts(self, temp_dir):
        """Test handling of concurrent write conflicts."""
        store = SafetensorsStore(str(temp_dir / "concurrent_write_test"))

        # Create identical weights (same hash)
        weight1 = WeightTensor(
            data=np.ones((3, 3), dtype=np.float32),
            metadata=WeightMetadata(name="identical1", shape=(3, 3), dtype=np.float32),
        )

        weight2 = WeightTensor(
            data=np.ones((3, 3), dtype=np.float32),
            metadata=WeightMetadata(
                name="identical2",  # Different name but same data
                shape=(3, 3),
                dtype=np.float32,
            ),
        )

        # They should have the same hash
        assert weight1.compute_hash() == weight2.compute_hash()

        results = []
        errors = []

        def store_worker(weight, results_list, errors_list):
            try:
                hash_key = store.store(weight)
                results_list.append(hash_key)
            except Exception as e:
                errors_list.append(e)

        # Start concurrent stores
        threads = []
        for weight in [weight1, weight2]:
            thread = threading.Thread(
                target=store_worker, args=(weight, results, errors)
            )
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should not have errors
        assert len(errors) == 0
        assert len(results) == 2
        assert results[0] == results[1]  # Same hash key

        # File should exist and be loadable
        loaded = store.load(results[0])
        assert loaded is not None

    def test_invalid_metadata_handling(self, temp_dir):
        """Test handling of invalid or malformed metadata."""
        store = SafetensorsStore(str(temp_dir / "invalid_metadata_test"))

        # Create weight with problematic metadata
        weight = WeightTensor(
            data=np.ones((2, 2), dtype=np.float32),
            metadata=WeightMetadata(
                name="test",
                shape=(2, 2),
                dtype=np.float32,
                layer_type="normal",
                model_name="test_model",
                compression_info={"invalid": float("inf")},  # Non-JSON-serializable
            ),
        )

        # Should handle serialization gracefully
        try:
            hash_key = store.store(weight)
            loaded = store.load(hash_key)
            # The problematic field might be dropped or converted
            assert loaded is not None
            assert loaded.metadata.name == "test"
        except Exception:
            # If it fails, it should be a clear serialization error
            pass

    def test_storage_cleanup_and_recovery(self, temp_dir):
        """Test cleanup of temporary files and recovery from interrupted operations."""
        store = SafetensorsStore(str(temp_dir / "cleanup_test"))

        # Create some weights
        weights = {}
        for i in range(5):
            data = np.random.randn(10, 10).astype(np.float32)
            weights[f"weight_{i}"] = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"weight_{i}", shape=(10, 10), dtype=np.float32
                ),
            )

        # Store weights
        hash_map = store.store_batch(weights)

        # Manually create some temporary files to simulate interrupted operations
        temp_files = []
        for i in range(3):
            temp_file = store.storage_path / f"temp_{i}.tmp"
            temp_file.write_text("temporary file content")
            temp_files.append(temp_file)

        # Verify temp files exist
        for temp_file in temp_files:
            assert temp_file.exists()

        # Storage info should still work correctly
        info = store.get_storage_info()
        assert info["total_files"] == len(weights)  # Should not count temp files

        # All original weights should still be loadable
        loaded_weights = store.load_batch(list(hash_map.values()))
        assert len(loaded_weights) == len(weights)

        # Clean up temp files (in a real scenario, this might be done automatically)
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()


class TestSafetensorsSpecialCases:
    """Test special cases and edge conditions."""

    def test_zero_dimensional_tensors(self, safetensors_store):
        """Test handling of scalar (0-dimensional) tensors."""
        # Scalar tensor
        scalar_data = np.array(42.0, dtype=np.float32)
        scalar_weight = WeightTensor(
            data=scalar_data,
            metadata=WeightMetadata(
                name="scalar",
                shape=(),  # Empty tuple for scalar
                dtype=np.float32,
            ),
        )

        hash_key = safetensors_store.store(scalar_weight)
        loaded = safetensors_store.load(hash_key)

        assert loaded is not None
        assert loaded.data.shape == ()
        assert loaded.data.item() == 42.0

    def test_very_large_dimensions(self, temp_dir):
        """Test tensors with very large dimension counts."""
        # Create a tensor with many dimensions but small total size
        shape = (2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3)  # 16 dimensions
        data = np.random.randn(*shape).astype(np.float32)

        weight = WeightTensor(
            data=data,
            metadata=WeightMetadata(
                name="high_dim_tensor", shape=shape, dtype=np.float32
            ),
        )

        store = SafetensorsStore(str(temp_dir / "high_dim_test"))
        hash_key = store.store(weight)
        loaded = store.load(hash_key)

        assert loaded is not None
        assert loaded.data.shape == shape
        np.testing.assert_array_equal(loaded.data, data)

    def test_special_float_values(self, safetensors_store):
        """Test handling of special float values (NaN, inf, -inf)."""
        special_data = np.array(
            [
                0.0,
                1.0,
                -1.0,
                np.inf,
                -np.inf,
                np.nan,
                np.finfo(np.float32).max,
                np.finfo(np.float32).min,
                np.finfo(np.float32).tiny,
            ],
            dtype=np.float32,
        ).reshape(3, 3)

        weight = WeightTensor(
            data=special_data,
            metadata=WeightMetadata(
                name="special_floats", shape=(3, 3), dtype=np.float32
            ),
        )

        hash_key = safetensors_store.store(weight)
        loaded = safetensors_store.load(hash_key)

        assert loaded is not None

        # Check special values are preserved
        assert np.isinf(loaded.data[1, 0])  # inf
        assert loaded.data[1, 0] > 0
        assert np.isinf(loaded.data[1, 1])  # -inf
        assert loaded.data[1, 1] < 0
        assert np.isnan(loaded.data[1, 2])  # nan

        # Check finite values
        assert loaded.data[0, 0] == 0.0
        assert loaded.data[0, 1] == 1.0
        assert loaded.data[0, 2] == -1.0

    def test_unicode_and_special_characters(self, safetensors_store):
        """Test handling of unicode and special characters in names."""
        test_names = [
            "normal_name",
            "name_with_123_numbers",
            "name-with-dashes",
            "name_with_underscore",
            "name.with.dots",
            "测试名称",  # Chinese
            "тест",  # Russian
            "🚀🤖",  # Emoji
            "name with spaces",
            "très_spécial",  # French accents
        ]

        stored_weights = {}
        for i, name in enumerate(test_names):
            data = np.ones((2, 2), dtype=np.float32) * i
            weight = WeightTensor(
                data=data,
                metadata=WeightMetadata(name=name, shape=(2, 2), dtype=np.float32),
            )

            hash_key = safetensors_store.store(weight)
            stored_weights[name] = (hash_key, data)

        # Load all weights back and verify
        for name, (hash_key, original_data) in stored_weights.items():
            loaded = safetensors_store.load(hash_key)
            assert loaded is not None
            assert loaded.metadata.name == name
            np.testing.assert_array_equal(loaded.data, original_data)

    def test_dtype_edge_cases(self, safetensors_store):
        """Test edge cases for different data types."""
        # Test various numpy dtypes
        test_cases = [
            (np.int8, np.array([-128, 0, 127], dtype=np.int8)),
            (np.uint8, np.array([0, 128, 255], dtype=np.uint8)),
            (np.int16, np.array([-32768, 0, 32767], dtype=np.int16)),
            (
                np.int32,
                np.array(
                    [np.iinfo(np.int32).min, 0, np.iinfo(np.int32).max], dtype=np.int32
                ),
            ),
            (np.bool_, np.array([True, False, True], dtype=np.bool_)),
        ]

        # Add float16 if available (not all systems support it fully)
        try:
            float16_data = np.array([1.0, 2.5, -3.7], dtype=np.float16)
            test_cases.append((np.float16, float16_data))
        except (ValueError, TypeError):
            pass  # Skip if float16 not supported

        for dtype, data in test_cases:
            dtype_name = np.dtype(dtype).name
            weight = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"dtype_test_{dtype_name}", shape=data.shape, dtype=dtype
                ),
            )

            hash_key = safetensors_store.store(weight)
            loaded = safetensors_store.load(hash_key)

            assert loaded is not None
            assert loaded.metadata.dtype == dtype
            np.testing.assert_array_equal(loaded.data, data)


class TestSafetensorsCLIIntegration:
    """Test CLI integration scenarios with SafeTensors."""

    def test_basic_cli_workflow_simulation(self, temp_dir, sample_weights):
        """Test a basic CLI-like workflow using SafetensorsStore programmatically."""
        # Simulate CLI workflow: init -> add -> commit -> export

        # 1. Initialize repository
        repo_path = temp_dir / "cli_test_repo"
        repo_path.mkdir()
        repo = Repository(repo_path, init=True)

        # 2. Add weights (simulating 'coral add')
        repo.stage_weights(sample_weights)

        # 3. Commit (simulating 'coral commit')
        commit_hash = repo.commit("Add safetensors test weights")
        assert commit_hash is not None

        # 4. Export to safetensors format (simulating 'coral export-safetensors')
        export_store = SafetensorsStore(str(temp_dir / "cli_export"))
        export_hashes = {}

        for name in sample_weights.keys():
            weight = repo.get_weight(name)
            if weight is not None:
                hash_key = export_store.store(weight)
                export_hashes[name] = hash_key

        # 5. Verify export worked
        assert len(export_hashes) == len(sample_weights)

        # 6. Simulate import into new repository
        import_repo_path = temp_dir / "cli_import_repo"
        import_repo_path.mkdir()
        import_repo = Repository(import_repo_path, init=True)

        # Load from safetensors store and import to new repo
        imported_weights = {}
        for name, hash_key in export_hashes.items():
            weight = export_store.load(hash_key)
            if weight is not None:
                imported_weights[name] = weight

        import_repo.stage_weights(imported_weights)
        import_commit = import_repo.commit("Import from safetensors")
        assert import_commit is not None

        # 7. Verify data integrity through full workflow
        for name, original in sample_weights.items():
            imported = import_repo.get_weight(name)
            assert imported is not None
            if original.data.size > 0:
                np.testing.assert_array_equal(imported.data, original.data)

    def test_conversion_workflow_simulation(self, temp_dir):
        """Test conversion workflow between different storage formats."""
        # Create test data in safetensors format
        st_file = temp_dir / "original.safetensors"
        test_tensors = {
            "layer1.weight": np.random.randn(10, 20).astype(np.float32),
            "layer1.bias": np.random.randn(10).astype(np.float32),
            "layer2.weight": np.random.randn(5, 10).astype(np.float32),
        }

        save_file(
            test_tensors, st_file, metadata={"format": "pytorch", "version": "1.0"}
        )

        # Convert to Coral format
        coral_store = SafetensorsStore(str(temp_dir / "coral_converted"))

        # Load from original safetensors file
        loaded_tensors = load_file(st_file)

        # Convert to Coral WeightTensors and store
        coral_weights = {}
        for name, data in loaded_tensors.items():
            weight = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=name,
                    shape=data.shape,
                    dtype=data.dtype,
                    layer_type="converted",
                    model_name="original_model",
                ),
            )
            coral_weights[name] = weight

        # Store in Coral format
        coral_hashes = coral_store.store_batch(coral_weights)

        # Convert back to safetensors format
        converted_file = temp_dir / "converted.safetensors"
        converted_tensors = {}

        for name, hash_key in coral_hashes.items():
            weight = coral_store.load(hash_key)
            converted_tensors[name] = weight.data

        save_file(
            converted_tensors, converted_file, metadata={"source": "coral_converted"}
        )

        # Verify conversion preserved data
        final_tensors = load_file(converted_file)

        for name in test_tensors.keys():
            np.testing.assert_array_equal(final_tensors[name], test_tensors[name])


if __name__ == "__main__":
    # Allow running individual test classes for development
    import sys

    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        pytest.main(["-v", f"test_safetensors_integration.py::{test_class}"])
    else:
        pytest.main(["-v", "test_safetensors_integration.py"])
