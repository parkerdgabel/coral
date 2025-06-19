import numpy as np
import pytest

from coral.delta.compression import DeltaCompressor


class TestDeltaCompressor:
    def test_compress_sparse_deltas_empty(self):
        """Test compression of empty sparse deltas."""
        indices = np.array([], dtype=np.int64)
        values = np.array([], dtype=np.float32)

        compressed, metadata = DeltaCompressor.compress_sparse_deltas(indices, values)

        assert len(compressed) == 0
        assert metadata["compression_type"] == "sparse_rle"
        assert metadata["original_length"] == 0

    def test_compress_sparse_deltas_single(self):
        """Test compression of single sparse delta."""
        indices = np.array([5], dtype=np.int64)
        values = np.array([2.5], dtype=np.float32)

        compressed, metadata = DeltaCompressor.compress_sparse_deltas(indices, values)

        assert compressed.shape == (1, 2)
        assert metadata["first_index"] == 5
        assert metadata["original_length"] == 1

    def test_compress_decompress_sparse_deltas(self):
        """Test round-trip compression/decompression of sparse deltas."""
        indices = np.array([1, 5, 10, 11, 20], dtype=np.int64)
        values = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float32)

        compressed, metadata = DeltaCompressor.compress_sparse_deltas(indices, values)
        decompressed_indices, decompressed_values = (
            DeltaCompressor.decompress_sparse_deltas(compressed, metadata)
        )

        np.testing.assert_array_equal(decompressed_indices, indices)
        np.testing.assert_array_almost_equal(decompressed_values, values)

    def test_compress_sparse_deltas_unsorted(self):
        """Test compression of unsorted sparse deltas."""
        indices = np.array([10, 2, 5, 1], dtype=np.int64)
        values = np.array([4.0, 2.0, 3.0, 1.0], dtype=np.float32)

        compressed, metadata = DeltaCompressor.compress_sparse_deltas(indices, values)
        decompressed_indices, decompressed_values = (
            DeltaCompressor.decompress_sparse_deltas(compressed, metadata)
        )

        # Should be sorted after compression
        expected_indices = np.array([1, 2, 5, 10])
        expected_values = np.array([1.0, 2.0, 3.0, 4.0])

        np.testing.assert_array_equal(decompressed_indices, expected_indices)
        np.testing.assert_array_almost_equal(decompressed_values, expected_values)

    def test_adaptive_quantization_int8(self):
        """Test adaptive quantization to 8 bits."""
        data = np.random.randn(1000).astype(np.float32)

        quantized, metadata = DeltaCompressor.adaptive_quantization(data, target_bits=8)

        assert quantized.dtype == np.int8
        assert "scale" in metadata
        assert "min_val" in metadata
        assert "max_val" in metadata
        assert metadata["target_bits"] == 8

    def test_adaptive_quantization_int16(self):
        """Test adaptive quantization to 16 bits."""
        data = np.random.randn(1000).astype(np.float32) * 10

        quantized, metadata = DeltaCompressor.adaptive_quantization(
            data, target_bits=16
        )

        assert quantized.dtype == np.int16
        assert metadata["target_bits"] == 16

    def test_adaptive_quantization_invalid_bits(self):
        """Test adaptive quantization with invalid bit size."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        with pytest.raises(ValueError, match="target_bits must be 8 or 16"):
            DeltaCompressor.adaptive_quantization(data, target_bits=4)

    def test_adaptive_quantization_uniform_data(self):
        """Test adaptive quantization with uniform data."""
        data = np.ones(100, dtype=np.float32) * 5.0

        quantized, metadata = DeltaCompressor.adaptive_quantization(data, target_bits=8)

        assert metadata["std"] == 0.0
        assert len(np.unique(quantized)) == 1

    def test_adaptive_quantization_outliers(self):
        """Test adaptive quantization handles outliers."""
        # Data with outliers
        data = np.concatenate(
            [
                np.random.randn(950).astype(np.float32),  # Normal data
                np.array([100, -100, 200, -200, 500], dtype=np.float32),  # Outliers
            ]
        )

        quantized, metadata = DeltaCompressor.adaptive_quantization(data, target_bits=8)

        assert metadata["outlier_ratio"] > 0
        assert metadata["outlier_ratio"] < 0.1  # Less than 10% outliers

    def test_adaptive_quantization_round_trip(self):
        """Test round-trip adaptive quantization."""
        data = np.random.randn(1000).astype(np.float32) * 2.5

        quantized, metadata = DeltaCompressor.adaptive_quantization(data, target_bits=8)
        dequantized = DeltaCompressor.dequantize_adaptive(quantized, metadata)

        # Check reconstruction error is reasonable
        error = np.abs(data - dequantized)
        assert np.mean(error) < 0.1  # Average error less than 0.1
        assert np.max(error) < 1.0  # Max error less than 1.0

    def test_compress_with_dictionary(self):
        """Test dictionary-based compression."""
        # Create data with repeated values
        data = np.array([1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 4.0, 2.0], dtype=np.float32)

        compressed, metadata = DeltaCompressor.compress_with_dictionary(
            data, dictionary_size=4
        )

        assert "dictionary" in metadata
        assert len(metadata["dictionary"]) <= 4
        assert metadata["dict_coverage"] > 0

    def test_compress_decompress_with_dictionary(self):
        """Test round-trip dictionary compression."""
        # Create 2D data with repeated patterns
        data = np.array(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [3.0, 1.0, 2.0]], dtype=np.float32
        )

        compressed, metadata = DeltaCompressor.compress_with_dictionary(
            data, dictionary_size=3
        )
        decompressed = DeltaCompressor.decompress_with_dictionary(compressed, metadata)

        np.testing.assert_array_almost_equal(decompressed, data)
        assert metadata["original_shape"] == data.shape

    def test_delta_statistics(self):
        """Test delta statistics calculation."""
        delta_data = np.random.randn(1000).astype(np.float32)

        stats = DeltaCompressor.delta_statistics(delta_data)

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "sparsity" in stats
        assert "dynamic_range" in stats
        assert "entropy" in stats
        assert "zero_ratio" in stats

        assert isinstance(stats["mean"], float)
        assert isinstance(stats["entropy"], float)
        assert 0 <= stats["sparsity"] <= 1
        assert 0 <= stats["zero_ratio"] <= 1

    def test_delta_statistics_sparse(self):
        """Test delta statistics for sparse data."""
        # Create sparse delta data
        delta_data = np.zeros(1000, dtype=np.float32)
        delta_data[::10] = np.random.randn(100)  # 10% non-zero

        stats = DeltaCompressor.delta_statistics(delta_data)

        assert stats["sparsity"] > 0.85  # Should detect high sparsity
        assert stats["zero_ratio"] >= 0.9

    def test_estimate_entropy(self):
        """Test entropy estimation."""
        # Uniform distribution (high entropy)
        uniform_data = np.random.uniform(-1, 1, 1000).astype(np.float32)
        uniform_entropy = DeltaCompressor._estimate_entropy(uniform_data)

        # Concentrated distribution (low entropy)
        concentrated_data = np.random.normal(0, 0.1, 1000).astype(np.float32)
        concentrated_entropy = DeltaCompressor._estimate_entropy(concentrated_data)

        assert uniform_entropy > concentrated_entropy
        assert 0 <= uniform_entropy <= 8  # Max entropy for 256 bins

    def test_recommend_compression_sparse(self):
        """Test compression recommendation for sparse data."""
        # Create highly sparse data
        delta_data = np.zeros(1000, dtype=np.float32)
        delta_data[::20] = np.random.randn(50) * 0.1

        recommendation = DeltaCompressor.recommend_compression(delta_data)
        assert recommendation == "sparse"

    def test_recommend_compression_low_range(self):
        """Test compression recommendation for low dynamic range."""
        # Small values with low variance
        delta_data = np.random.normal(0, 0.05, 1000).astype(np.float32)

        recommendation = DeltaCompressor.recommend_compression(delta_data)
        assert recommendation == "int8_quantized"

    def test_recommend_compression_medium_range(self):
        """Test compression recommendation for medium dynamic range."""
        # Medium range values
        delta_data = np.random.uniform(-10, 10, 1000).astype(np.float32)

        recommendation = DeltaCompressor.recommend_compression(delta_data)
        assert recommendation in ["int16_quantized", "compressed"]

    def test_recommend_compression_high_entropy(self):
        """Test compression recommendation for high entropy data."""
        # High entropy, large dynamic range
        delta_data = np.random.uniform(-1000, 1000, 10000).astype(np.float32)

        recommendation = DeltaCompressor.recommend_compression(delta_data)
        assert recommendation == "float32_raw"

    def test_compress_with_dictionary_large(self):
        """Test dictionary compression with large dictionary."""
        # Create data with many unique values
        data = np.random.randint(0, 500, size=(100, 100)).astype(np.float32)

        compressed, metadata = DeltaCompressor.compress_with_dictionary(
            data, dictionary_size=256
        )

        assert len(metadata["dictionary"]) == 256
        assert compressed.shape == data.shape

    def test_adaptive_quantization_empty(self):
        """Test adaptive quantization with empty array."""
        data = np.array([], dtype=np.float32)

        quantized, metadata = DeltaCompressor.adaptive_quantization(data, target_bits=8)

        assert len(quantized) == 0
        assert metadata["outlier_ratio"] == 0
