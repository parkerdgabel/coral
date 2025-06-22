"""Test coverage improvement for utils and other modules."""

import threading
import time
from unittest.mock import MagicMock, patch
import warnings

import numpy as np
import pytest

from coral.utils.thread_safety import (
    ThreadSafeDict,
    ThreadSafeList,
    ThreadSafeSet,
    ReadWriteLock,
    thread_safe_cache,
    synchronized,
)
from coral.utils.visualization import Visualizer
from coral.core.weight_tensor import WeightTensor
from coral.delta.compression import DeltaCompressor


class TestThreadSafetyCoverage:
    """Tests to improve coverage for thread safety utilities."""

    def test_thread_safe_dict_operations(self):
        """Test ThreadSafeDict operations."""
        tsd = ThreadSafeDict()
        
        # Basic operations
        tsd["key1"] = "value1"
        assert tsd["key1"] == "value1"
        assert "key1" in tsd
        assert len(tsd) == 1
        
        # Update multiple
        tsd.update({"key2": "value2", "key3": "value3"})
        assert len(tsd) == 3
        
        # Get with default
        assert tsd.get("key4", "default") == "default"
        
        # Pop
        val = tsd.pop("key1", None)
        assert val == "value1"
        assert "key1" not in tsd
        
        # Items, keys, values
        assert list(tsd.keys()) == ["key2", "key3"]
        assert list(tsd.values()) == ["value2", "value3"]
        assert list(tsd.items()) == [("key2", "value2"), ("key3", "value3")]
        
        # Clear
        tsd.clear()
        assert len(tsd) == 0

    def test_thread_safe_dict_concurrent(self):
        """Test ThreadSafeDict with concurrent access."""
        tsd = ThreadSafeDict()
        errors = []
        
        def writer(start, end):
            try:
                for i in range(start, end):
                    tsd[f"key_{i}"] = i
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def reader():
            try:
                for _ in range(50):
                    _ = list(tsd.items())
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        # Create threads
        threads = []
        threads.append(threading.Thread(target=writer, args=(0, 25)))
        threads.append(threading.Thread(target=writer, args=(25, 50)))
        threads.append(threading.Thread(target=reader))
        
        # Run concurrently
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Check results
        assert len(errors) == 0
        assert len(tsd) == 50

    def test_thread_safe_list_operations(self):
        """Test ThreadSafeList operations."""
        tsl = ThreadSafeList()
        
        # Append and extend
        tsl.append(1)
        tsl.extend([2, 3, 4])
        assert len(tsl) == 4
        assert tsl[0] == 1
        assert tsl[-1] == 4
        
        # Insert and remove
        tsl.insert(1, 1.5)
        assert tsl[1] == 1.5
        tsl.remove(1.5)
        assert 1.5 not in tsl
        
        # Pop
        val = tsl.pop()
        assert val == 4
        assert len(tsl) == 3
        
        # Index and count
        assert tsl.index(2) == 1
        assert tsl.count(2) == 1
        
        # Iteration
        items = []
        for item in tsl:
            items.append(item)
        assert items == [1, 2, 3]
        
        # Clear
        tsl.clear()
        assert len(tsl) == 0

    def test_thread_safe_set_operations(self):
        """Test ThreadSafeSet operations."""
        tss = ThreadSafeSet()
        
        # Add and update
        tss.add(1)
        tss.update([2, 3, 4])
        assert len(tss) == 4
        assert 1 in tss
        
        # Remove and discard
        tss.remove(1)
        assert 1 not in tss
        tss.discard(10)  # Should not raise error
        
        # Pop
        val = tss.pop()
        assert val in [2, 3, 4]
        assert len(tss) == 2
        
        # Set operations
        tss2 = ThreadSafeSet([3, 4, 5])
        union = tss.union(tss2)
        assert len(union) >= 3
        
        intersection = tss.intersection(tss2)
        assert all(x in tss for x in intersection)
        
        # Clear
        tss.clear()
        assert len(tss) == 0

    def test_read_write_lock(self):
        """Test ReadWriteLock functionality."""
        lock = ReadWriteLock()
        shared_data = {"value": 0}
        
        def reader(reader_id):
            with lock.read():
                # Multiple readers can access simultaneously
                value = shared_data["value"]
                time.sleep(0.01)  # Simulate read operation
                return value
        
        def writer(new_value):
            with lock.write():
                # Only one writer at a time
                shared_data["value"] = new_value
                time.sleep(0.01)  # Simulate write operation
        
        # Test multiple readers
        reader_threads = []
        for i in range(3):
            t = threading.Thread(target=reader, args=(i,))
            reader_threads.append(t)
            t.start()
        
        # All readers should complete quickly (parallel reads)
        for t in reader_threads:
            t.join(timeout=0.1)
            assert not t.is_alive()
        
        # Test writer blocks readers
        writer_thread = threading.Thread(target=writer, args=(42,))
        writer_thread.start()
        
        # Try to read while writing
        time.sleep(0.005)  # Let writer acquire lock
        start_time = time.time()
        reader(99)
        elapsed = time.time() - start_time
        assert elapsed >= 0.005  # Reader was blocked
        
        writer_thread.join()
        assert shared_data["value"] == 42

    def test_thread_safe_cache_decorator(self):
        """Test thread_safe_cache decorator."""
        call_count = {"count": 0}
        
        @thread_safe_cache(maxsize=3)
        def expensive_function(x):
            call_count["count"] += 1
            time.sleep(0.01)
            return x * x
        
        # First calls should execute
        assert expensive_function(2) == 4
        assert expensive_function(3) == 9
        assert call_count["count"] == 2
        
        # Cached calls should not execute
        assert expensive_function(2) == 4
        assert expensive_function(3) == 9
        assert call_count["count"] == 2
        
        # Test cache eviction
        expensive_function(4)  # 16
        expensive_function(5)  # 25, this evicts 2
        assert call_count["count"] == 4
        
        # Accessing evicted item
        expensive_function(2)  # Should recompute
        assert call_count["count"] == 5
        
        # Test concurrent access
        results = []
        
        def worker(val):
            results.append(expensive_function(val))
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i % 3,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All results should be correct
        for i, result in enumerate(results):
            assert result == (i % 3) ** 2

    def test_synchronized_decorator(self):
        """Test synchronized method decorator."""
        class Counter:
            def __init__(self):
                self.value = 0
            
            @synchronized
            def increment(self):
                # Without synchronization, this would have race conditions
                current = self.value
                time.sleep(0.001)  # Simulate some work
                self.value = current + 1
            
            @synchronized
            def get_value(self):
                return self.value
        
        counter = Counter()
        
        def worker():
            for _ in range(10):
                counter.increment()
        
        # Run multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Should have correct count (5 threads * 10 increments)
        assert counter.get_value() == 50

    def test_thread_safe_edge_cases(self):
        """Test edge cases for thread-safe collections."""
        # ThreadSafeDict with None values
        tsd = ThreadSafeDict()
        tsd[None] = None
        assert None in tsd
        assert tsd[None] is None
        
        # ThreadSafeList with mixed types
        tsl = ThreadSafeList()
        tsl.extend([1, "two", 3.0, None, True])
        assert len(tsl) == 5
        assert tsl[1] == "two"
        
        # ThreadSafeSet with unhashable types (should fail)
        tss = ThreadSafeSet()
        with pytest.raises(TypeError):
            tss.add([1, 2, 3])  # Lists are unhashable


class TestVisualizerCoverage:
    """Tests to improve coverage for Visualizer class."""

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_weight_distribution(self, mock_savefig, mock_show):
        """Test plotting weight distribution."""
        # Create sample weight
        weight = WeightTensor(
            data=np.random.normal(0, 1, size=(100, 100)).astype(np.float32)
        )
        
        # Plot distribution
        Visualizer.plot_weight_distribution(weight, bins=50, title="Test Distribution")
        
        # Should show the plot
        mock_show.assert_called_once()
        
        # Plot and save
        Visualizer.plot_weight_distribution(
            weight, 
            save_path="test_dist.png",
            figsize=(8, 6)
        )
        mock_savefig.assert_called_once_with("test_dist.png", dpi=150, bbox_inches='tight')

    @patch('matplotlib.pyplot.show')
    def test_plot_weight_distribution_edge_cases(self, mock_show):
        """Test plotting with edge cases."""
        # Single value weight
        weight_single = WeightTensor(data=np.array([1.0], dtype=np.float32))
        Visualizer.plot_weight_distribution(weight_single)
        
        # All zeros
        weight_zeros = WeightTensor(data=np.zeros((10, 10), dtype=np.float32))
        Visualizer.plot_weight_distribution(weight_zeros)
        
        # Very large weight
        weight_large = WeightTensor(data=np.random.randn(1000000).astype(np.float32))
        Visualizer.plot_weight_distribution(weight_large, bins=100)
        
        assert mock_show.call_count == 3

    @patch('matplotlib.pyplot.show')
    def test_plot_compression_comparison(self, mock_show):
        """Test plotting compression comparison."""
        sizes = {
            "Original": 1000,
            "Quantized": 250,
            "Pruned": 400,
            "Combined": 150
        }
        
        Visualizer.plot_compression_comparison(
            sizes,
            title="Compression Methods",
            ylabel="Size (KB)"
        )
        
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    def test_plot_compression_comparison_save(self, mock_savefig):
        """Test saving compression comparison plot."""
        sizes = {"Method1": 100, "Method2": 50}
        
        Visualizer.plot_compression_comparison(
            sizes,
            save_path="compression.png",
            figsize=(10, 6),
            color='green'
        )
        
        mock_savefig.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_similarity_heatmap(self, mock_show):
        """Test plotting similarity heatmap."""
        # Create similarity matrix
        n_weights = 5
        similarity_matrix = np.random.rand(n_weights, n_weights)
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2  # Symmetric
        np.fill_diagonal(similarity_matrix, 1.0)  # Self-similarity = 1
        
        weight_names = [f"weight_{i}" for i in range(n_weights)]
        
        Visualizer.plot_similarity_heatmap(
            similarity_matrix,
            weight_names,
            title="Weight Similarities"
        )
        
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_deduplication_summary(self, mock_show):
        """Test plotting deduplication summary."""
        summary = {
            "original_weights": 100,
            "unique_weights": 60,
            "deduplicated": 40,
            "space_saved": 0.35,
            "compression_ratio": 1.67
        }
        
        Visualizer.plot_deduplication_summary(
            summary,
            title="Deduplication Results"
        )
        
        mock_show.assert_called_once()


class TestDeltaCompressionCoverage:
    """Tests to improve coverage for delta compression utilities."""

    def test_compress_sparse_deltas(self):
        """Test sparse delta compression."""
        # Create sparse data
        indices = np.array([0, 5, 10, 15, 20])
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        compressed, metadata = DeltaCompressor.compress_sparse_deltas(indices, values)
        
        # Verify compression
        assert compressed.shape[0] == len(indices)
        assert metadata["compression_type"] == "sparse_rle"
        assert metadata["original_length"] == 5
        
        # Test decompression
        decompressed_indices, decompressed_values = DeltaCompressor.decompress_sparse_deltas(
            compressed, metadata
        )
        
        assert np.array_equal(indices, decompressed_indices)
        assert np.array_equal(values, decompressed_values)

    def test_compress_sparse_deltas_empty(self):
        """Test sparse delta compression with empty data."""
        indices = np.array([])
        values = np.array([])
        
        compressed, metadata = DeltaCompressor.compress_sparse_deltas(indices, values)
        
        assert len(compressed) == 0
        assert metadata["original_length"] == 0
        
        # Decompress
        dec_indices, dec_values = DeltaCompressor.decompress_sparse_deltas(
            compressed, metadata
        )
        assert len(dec_indices) == 0
        assert len(dec_values) == 0

    def test_adaptive_quantization(self):
        """Test adaptive quantization."""
        # Normal distribution data
        data = np.random.normal(0, 1, 1000).astype(np.float32)
        
        # 8-bit quantization
        quant8, meta8 = DeltaCompressor.adaptive_quantization(data, target_bits=8)
        assert quant8.dtype == np.int8
        assert "scale" in meta8
        assert "outlier_ratio" in meta8
        
        # Dequantize
        dequant8 = DeltaCompressor.dequantize_adaptive(quant8, meta8)
        assert dequant8.shape == data.shape
        
        # 16-bit quantization
        quant16, meta16 = DeltaCompressor.adaptive_quantization(data, target_bits=16)
        assert quant16.dtype == np.int16
        
        # Invalid bits
        with pytest.raises(ValueError):
            DeltaCompressor.adaptive_quantization(data, target_bits=32)

    def test_adaptive_quantization_edge_cases(self):
        """Test adaptive quantization edge cases."""
        # Uniform data (no variation)
        uniform_data = np.ones(100, dtype=np.float32) * 5.0
        quant, meta = DeltaCompressor.adaptive_quantization(uniform_data)
        assert meta["scale"] == 1.0  # No scaling needed
        
        # Dequantize should return exact values
        dequant = DeltaCompressor.dequantize_adaptive(quant, meta)
        assert np.allclose(dequant, uniform_data)

    def test_compress_with_dictionary(self):
        """Test dictionary-based compression."""
        # Data with repeated values
        data = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5], dtype=np.float32)
        data = data.reshape((11,))
        
        compressed, metadata = DeltaCompressor.compress_with_dictionary(
            data, dictionary_size=4
        )
        
        assert "dictionary" in metadata
        assert len(metadata["dictionary"]) <= 4
        assert metadata["dict_coverage"] > 0
        
        # Decompress
        decompressed = DeltaCompressor.decompress_with_dictionary(
            compressed, metadata
        )
        
        # Should match original shape
        assert decompressed.shape == data.shape

    def test_delta_statistics(self):
        """Test delta statistics calculation."""
        # Create delta data with known properties
        delta_data = np.concatenate([
            np.zeros(50),  # 50% zeros
            np.random.normal(0, 0.1, 30),  # Small variations
            np.random.uniform(-1, 1, 20)  # Larger variations
        ])
        
        stats = DeltaCompressor.delta_statistics(delta_data)
        
        assert "mean" in stats
        assert "std" in stats
        assert "sparsity" in stats
        assert "entropy" in stats
        assert stats["zero_ratio"] == 0.5
        assert stats["sparsity"] > 0  # Should detect near-zero values

    def test_recommend_compression(self):
        """Test compression recommendation."""
        # Sparse data
        sparse_data = np.zeros(100)
        sparse_data[[10, 20, 30]] = [1.0, 2.0, 3.0]
        assert DeltaCompressor.recommend_compression(sparse_data) == "sparse"
        
        # Small range data
        small_range = np.random.uniform(-0.1, 0.1, 100)
        rec = DeltaCompressor.recommend_compression(small_range)
        assert rec in ["int8_quantized", "int16_quantized"]
        
        # Large range data
        large_range = np.random.uniform(-1000, 1000, 100)
        assert DeltaCompressor.recommend_compression(large_range) == "float32_raw"

    def test_estimate_entropy(self):
        """Test entropy estimation."""
        # Uniform distribution (high entropy)
        uniform = np.random.uniform(0, 1, 1000)
        entropy_uniform = DeltaCompressor._estimate_entropy(uniform)
        
        # Concentrated distribution (low entropy)
        concentrated = np.random.normal(0, 0.01, 1000)
        entropy_concentrated = DeltaCompressor._estimate_entropy(concentrated)
        
        # Uniform should have higher entropy
        assert entropy_uniform > entropy_concentrated
        
        # Single value (zero entropy)
        single = np.ones(100)
        entropy_single = DeltaCompressor._estimate_entropy(single, bins=10)
        assert entropy_single == 0.0