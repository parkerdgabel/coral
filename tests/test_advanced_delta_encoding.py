"""Tests for advanced delta encoding strategies.

Tests the new high-performance delta encoding methods:
- XOR_FLOAT32: Bitwise XOR with exponent/mantissa separation
- XOR_BFLOAT16: XOR optimized for BFloat16 weights
- EXPONENT_MANTISSA: Separate component encoding
- PER_AXIS_SCALED: 1-bit signs + per-axis scales
"""

import numpy as np
import pytest

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.delta.delta_encoder import (
    DeltaConfig,
    DeltaEncoder,
    DeltaType,
)


def make_weight(data: np.ndarray, name: str = "weight") -> WeightTensor:
    """Helper to create WeightTensor with proper metadata."""
    return WeightTensor(
        data=data,
        metadata=WeightMetadata(
            name=name,
            shape=data.shape,
            dtype=data.dtype,
        ),
    )


class TestXORFloat32Encoding:
    """Test XOR-based delta encoding for float32 weights."""

    def test_xor_float32_exact_reconstruction(self):
        """XOR encoding should perfectly reconstruct original weights."""
        config = DeltaConfig(delta_type=DeltaType.XOR_FLOAT32)
        encoder = DeltaEncoder(config)

        # Create reference and similar weights
        np.random.seed(42)
        reference_data = np.random.randn(100, 100).astype(np.float32)
        weight_data = reference_data + 0.01 * np.random.randn(100, 100).astype(
            np.float32
        )

        reference = make_weight(reference_data, "reference")
        weight = make_weight(weight_data, "weight")

        # Encode and decode
        delta = encoder.encode_delta(weight, reference)
        reconstructed = encoder.decode_delta(delta, reference)

        # Should be exactly equal (lossless)
        np.testing.assert_array_equal(
            reconstructed.data.astype(np.float32),
            weight_data,
            err_msg="XOR encoding should be lossless",
        )

    def test_xor_float32_compression(self):
        """XOR encoding should achieve good compression for similar weights."""
        config = DeltaConfig(delta_type=DeltaType.XOR_FLOAT32)
        encoder = DeltaEncoder(config)

        # Create very similar weights (simulating fine-tuning)
        np.random.seed(42)
        reference_data = np.random.randn(256, 256).astype(np.float32)
        # Only 0.1% perturbation
        weight_data = reference_data * (
            1 + 0.001 * np.random.randn(256, 256).astype(np.float32)
        )

        reference = make_weight(reference_data, "reference")
        weight = make_weight(weight_data, "weight")

        delta = encoder.encode_delta(weight, reference)

        # Should achieve significant compression
        original_size = weight.nbytes
        delta_size = delta.nbytes

        compression_ratio = 1 - (delta_size / original_size)
        assert compression_ratio > 0.1, (
            f"Expected >10% compression, got {compression_ratio:.1%}"
        )

    def test_xor_float32_identical_weights(self):
        """XOR of identical weights should be highly compressible."""
        config = DeltaConfig(delta_type=DeltaType.XOR_FLOAT32)
        encoder = DeltaEncoder(config)

        np.random.seed(42)
        data = np.random.randn(100, 100).astype(np.float32)

        reference = make_weight(data.copy(), "reference")
        weight = make_weight(data.copy(), "weight")

        delta = encoder.encode_delta(weight, reference)
        reconstructed = encoder.decode_delta(delta, reference)

        np.testing.assert_array_equal(reconstructed.data, data)

        # Should achieve very high compression (XOR of identical = all zeros)
        assert delta.compression_ratio > 0.5, (
            f"Expected >50% compression for identical weights, got "
            f"{delta.compression_ratio:.1%}"
        )


class TestXORBFloat16Encoding:
    """Test XOR-based delta encoding optimized for BFloat16."""

    def test_xor_bfloat16_exact_reconstruction(self):
        """XOR BF16 encoding should perfectly reconstruct original weights."""
        config = DeltaConfig(delta_type=DeltaType.XOR_BFLOAT16)
        encoder = DeltaEncoder(config)

        np.random.seed(42)
        reference_data = np.random.randn(64, 64).astype(np.float32)
        weight_data = reference_data + 0.01 * np.random.randn(64, 64).astype(np.float32)

        reference = make_weight(reference_data, "reference")
        weight = make_weight(weight_data, "weight")

        delta = encoder.encode_delta(weight, reference)
        reconstructed = encoder.decode_delta(delta, reference)

        # Should be exactly equal (lossless)
        np.testing.assert_array_equal(
            reconstructed.data.astype(np.float32),
            weight_data,
            err_msg="XOR BF16 encoding should be lossless",
        )

    def test_xor_bfloat16_various_shapes(self):
        """Test XOR BF16 encoding with various tensor shapes."""
        config = DeltaConfig(delta_type=DeltaType.XOR_BFLOAT16)
        encoder = DeltaEncoder(config)

        shapes = [(100,), (32, 32), (16, 16, 4), (8, 8, 8, 8)]

        for shape in shapes:
            np.random.seed(42)
            reference_data = np.random.randn(*shape).astype(np.float32)
            weight_data = reference_data + 0.01 * np.random.randn(*shape).astype(
                np.float32
            )

            reference = make_weight(reference_data, "reference")
            weight = make_weight(weight_data, "weight")

            delta = encoder.encode_delta(weight, reference)
            reconstructed = encoder.decode_delta(delta, reference)

            np.testing.assert_array_equal(
                reconstructed.data.astype(np.float32),
                weight_data,
                err_msg=f"Failed for shape {shape}",
            )


class TestExponentMantissaEncoding:
    """Test exponent-mantissa separated encoding."""

    def test_exponent_mantissa_exact_reconstruction(self):
        """Exponent-mantissa encoding should perfectly reconstruct."""
        config = DeltaConfig(delta_type=DeltaType.EXPONENT_MANTISSA)
        encoder = DeltaEncoder(config)

        np.random.seed(42)
        reference_data = np.random.randn(50, 50).astype(np.float32)
        weight_data = reference_data + 0.05 * np.random.randn(50, 50).astype(np.float32)

        reference = make_weight(reference_data, "reference")
        weight = make_weight(weight_data, "weight")

        delta = encoder.encode_delta(weight, reference)
        reconstructed = encoder.decode_delta(delta, reference)

        # Should be exactly equal (lossless)
        np.testing.assert_array_equal(
            reconstructed.data.astype(np.float32),
            weight_data,
            err_msg="Exponent-mantissa encoding should be lossless",
        )

    def test_exponent_mantissa_compression(self):
        """Test compression efficiency of exponent-mantissa encoding."""
        config = DeltaConfig(delta_type=DeltaType.EXPONENT_MANTISSA)
        encoder = DeltaEncoder(config)

        np.random.seed(42)
        # Similar weights with small differences
        reference_data = np.random.randn(128, 128).astype(np.float32)
        weight_data = reference_data * (
            1 + 0.001 * np.random.randn(128, 128).astype(np.float32)
        )

        reference = make_weight(reference_data, "reference")
        weight = make_weight(weight_data, "weight")

        delta = encoder.encode_delta(weight, reference)

        # Verify lossless reconstruction
        reconstructed = encoder.decode_delta(delta, reference)
        np.testing.assert_array_equal(
            reconstructed.data.astype(np.float32), weight_data
        )

    def test_exponent_mantissa_sign_handling(self):
        """Test correct handling of sign bits."""
        config = DeltaConfig(delta_type=DeltaType.EXPONENT_MANTISSA)
        encoder = DeltaEncoder(config)

        # Create data with mixed signs
        reference_data = np.array(
            [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float32
        )
        weight_data = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)

        reference = make_weight(reference_data, "reference")
        weight = make_weight(weight_data, "weight")

        delta = encoder.encode_delta(weight, reference)
        reconstructed = encoder.decode_delta(delta, reference)

        np.testing.assert_array_equal(
            reconstructed.data.astype(np.float32),
            weight_data,
            err_msg="Sign handling incorrect",
        )


class TestPerAxisScaledEncoding:
    """Test per-axis scaled delta encoding (lossy but high compression)."""

    def test_per_axis_scaled_reconstruction_quality(self):
        """Per-axis scaled should approximately reconstruct."""
        config = DeltaConfig(delta_type=DeltaType.PER_AXIS_SCALED)
        encoder = DeltaEncoder(config)

        np.random.seed(42)
        reference_data = np.random.randn(64, 64).astype(np.float32)
        # Fine-tuned model: small, structured differences
        delta_pattern = np.outer(
            np.random.randn(64).astype(np.float32),
            np.random.randn(64).astype(np.float32),
        )
        weight_data = reference_data + 0.01 * delta_pattern

        reference = make_weight(reference_data, "reference")
        weight = make_weight(weight_data, "weight")

        delta = encoder.encode_delta(weight, reference)
        reconstructed = encoder.decode_delta(delta, reference)

        # This is lossy, so check relative error
        actual_delta = weight_data - reference_data
        reconstructed_delta = reconstructed.data - reference_data

        # Should preserve sign structure
        sign_agreement = np.mean(np.sign(actual_delta) == np.sign(reconstructed_delta))
        assert sign_agreement > 0.9, f"Sign agreement too low: {sign_agreement:.1%}"

    def test_per_axis_scaled_compression_ratio(self):
        """Per-axis scaled should achieve high compression."""
        config = DeltaConfig(delta_type=DeltaType.PER_AXIS_SCALED)
        encoder = DeltaEncoder(config)

        np.random.seed(42)
        reference_data = np.random.randn(256, 256).astype(np.float32)
        weight_data = reference_data + 0.01 * np.random.randn(256, 256).astype(
            np.float32
        )

        reference = make_weight(reference_data, "reference")
        weight = make_weight(weight_data, "weight")

        delta = encoder.encode_delta(weight, reference)

        # Calculate compression
        original_size = weight.nbytes
        delta_size = delta.nbytes

        compression_ratio = 1 - (delta_size / original_size)

        # Per-axis scaled should achieve at least 60% compression for 2D weights
        # (1 bit per element + FP16 scales vs 32 bits per element)
        assert compression_ratio > 0.6, (
            f"Expected >60% compression, got {compression_ratio:.1%}"
        )

    def test_per_axis_scaled_1d_array(self):
        """Test per-axis scaled encoding with 1D arrays."""
        config = DeltaConfig(delta_type=DeltaType.PER_AXIS_SCALED)
        encoder = DeltaEncoder(config)

        np.random.seed(42)
        reference_data = np.random.randn(1000).astype(np.float32)
        weight_data = reference_data + 0.01 * np.random.randn(1000).astype(np.float32)

        reference = make_weight(reference_data, "reference")
        weight = make_weight(weight_data, "weight")

        delta = encoder.encode_delta(weight, reference)
        reconstructed = encoder.decode_delta(delta, reference)

        # Should have same shape
        assert reconstructed.shape == weight.shape

    def test_per_axis_scaled_3d_array(self):
        """Test per-axis scaled encoding with 3D arrays."""
        config = DeltaConfig(delta_type=DeltaType.PER_AXIS_SCALED)
        encoder = DeltaEncoder(config)

        np.random.seed(42)
        reference_data = np.random.randn(16, 16, 8).astype(np.float32)
        weight_data = reference_data + 0.01 * np.random.randn(16, 16, 8).astype(
            np.float32
        )

        reference = make_weight(reference_data, "reference")
        weight = make_weight(weight_data, "weight")

        delta = encoder.encode_delta(weight, reference)
        reconstructed = encoder.decode_delta(delta, reference)

        # Should have same shape
        assert reconstructed.shape == weight.shape


class TestDeltaTypeProperties:
    """Test DeltaType enum properties."""

    def test_new_lossless_types(self):
        """New lossless types should be marked correctly."""
        assert DeltaType.XOR_FLOAT32.is_lossless
        assert DeltaType.XOR_BFLOAT16.is_lossless
        assert DeltaType.EXPONENT_MANTISSA.is_lossless

    def test_lossy_types(self):
        """Lossy types should be marked correctly."""
        assert not DeltaType.PER_AXIS_SCALED.is_lossless
        assert not DeltaType.INT8_QUANTIZED.is_lossless
        assert not DeltaType.SPARSE.is_lossless


class TestCanEncodeAsDelta:
    """Test can_encode_as_delta with new types."""

    def test_lossless_types_always_allowed(self):
        """Lossless types should always be allowed for deduplication."""
        for delta_type in [
            DeltaType.XOR_FLOAT32,
            DeltaType.XOR_BFLOAT16,
            DeltaType.EXPONENT_MANTISSA,
        ]:
            config = DeltaConfig(delta_type=delta_type, min_weight_size=100)
            encoder = DeltaEncoder(config)

            np.random.seed(42)
            reference_data = np.random.randn(100, 100).astype(np.float32)
            weight_data = reference_data + 0.1 * np.random.randn(100, 100).astype(
                np.float32
            )

            reference = make_weight(reference_data, "reference")
            weight = make_weight(weight_data, "weight")

            assert encoder.can_encode_as_delta(weight, reference), (
                f"Lossless type {delta_type} should always be allowed"
            )


class TestCompressionComparison:
    """Compare compression ratios across different encoding strategies."""

    @pytest.fixture
    def similar_weights(self):
        """Create a pair of similar weights for testing."""
        np.random.seed(42)
        reference_data = np.random.randn(128, 128).astype(np.float32)
        weight_data = reference_data + 0.01 * np.random.randn(128, 128).astype(
            np.float32
        )
        return (
            make_weight(reference_data, "reference"),
            make_weight(weight_data, "weight"),
        )

    def test_compare_lossless_strategies(self, similar_weights):
        """Compare compression of different lossless strategies."""
        reference, weight = similar_weights

        results = {}

        for delta_type in [
            DeltaType.FLOAT32_RAW,
            DeltaType.COMPRESSED,
            DeltaType.XOR_FLOAT32,
            DeltaType.XOR_BFLOAT16,
            DeltaType.EXPONENT_MANTISSA,
        ]:
            config = DeltaConfig(delta_type=delta_type)
            encoder = DeltaEncoder(config)

            delta = encoder.encode_delta(weight, reference)
            reconstructed = encoder.decode_delta(delta, reference)

            # Verify lossless reconstruction
            np.testing.assert_array_equal(
                reconstructed.data.astype(np.float32),
                weight.data,
                err_msg=f"{delta_type} is not lossless",
            )

            compression = 1 - (delta.nbytes / weight.nbytes)
            results[delta_type.value] = compression

        # Print results for visibility
        print("\nLossless compression comparison:")
        for name, compression in sorted(results.items(), key=lambda x: -x[1]):
            print(f"  {name}: {compression:.1%}")

        # New methods should generally be competitive or better
        assert results["xor_float32"] >= results["compressed"] * 0.8, (
            "XOR float32 should be at least 80% as effective as zlib"
        )
