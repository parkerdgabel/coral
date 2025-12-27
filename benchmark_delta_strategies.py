#!/usr/bin/env python3
"""Benchmark delta encoding strategies for weight compression.

Compares compression ratios and speed of different delta encoding methods:
- FLOAT32_RAW: Baseline (no compression)
- COMPRESSED: zlib compression
- XOR_FLOAT32: Bitwise XOR with exponent/mantissa separation
- XOR_BFLOAT16: XOR optimized for BFloat16
- EXPONENT_MANTISSA: Separate component encoding
- INT8_QUANTIZED: 8-bit quantization (lossy)
- PER_AXIS_SCALED: 1-bit signs + axis scales (lossy)
"""

import time
from typing import Dict, List, Tuple

import numpy as np

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.delta.delta_encoder import DeltaConfig, DeltaEncoder, DeltaType


def make_weight(data: np.ndarray, name: str = "weight") -> WeightTensor:
    """Create WeightTensor with proper metadata."""
    return WeightTensor(
        data=data,
        metadata=WeightMetadata(
            name=name,
            shape=data.shape,
            dtype=data.dtype,
        ),
    )


def create_similar_weights(
    shape: Tuple[int, ...],
    similarity_level: str = "high",
    seed: int = 42,
) -> Tuple[WeightTensor, WeightTensor]:
    """Create a pair of similar weights for testing.

    Args:
        shape: Shape of weights
        similarity_level: "high" (99.9%), "medium" (99%), or "low" (95%)
        seed: Random seed

    Returns:
        Tuple of (reference_weight, similar_weight)
    """
    np.random.seed(seed)

    # Create base weights
    reference_data = np.random.randn(*shape).astype(np.float32)

    # Add noise based on similarity level
    noise_scale = {
        "high": 0.001,    # 99.9% similar (fine-tuning)
        "medium": 0.01,   # 99% similar (continued training)
        "low": 0.05,      # 95% similar (transfer learning)
    }[similarity_level]

    noise = np.random.randn(*shape).astype(np.float32) * noise_scale
    similar_data = reference_data + noise

    return (
        make_weight(reference_data, "reference"),
        make_weight(similar_data, "similar"),
    )


def benchmark_encoding_strategy(
    delta_type: DeltaType,
    reference: WeightTensor,
    weight: WeightTensor,
    num_iterations: int = 10,
) -> Dict:
    """Benchmark a single encoding strategy.

    Returns dict with compression ratio, encode time, decode time, and lossless flag.
    """
    config = DeltaConfig(delta_type=delta_type)
    encoder = DeltaEncoder(config)

    # Warmup
    delta = encoder.encode_delta(weight, reference)
    _ = encoder.decode_delta(delta, reference)

    # Benchmark encoding
    encode_times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        delta = encoder.encode_delta(weight, reference)
        encode_times.append(time.perf_counter() - start)

    # Benchmark decoding
    decode_times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        reconstructed = encoder.decode_delta(delta, reference)
        decode_times.append(time.perf_counter() - start)

    # Check reconstruction quality
    original = weight.data.astype(np.float32)
    reconstructed_data = reconstructed.data.astype(np.float32)

    is_lossless = np.allclose(original, reconstructed_data, rtol=0, atol=0)
    if not is_lossless:
        max_error = np.max(np.abs(original - reconstructed_data))
        mean_error = np.mean(np.abs(original - reconstructed_data))
    else:
        max_error = 0.0
        mean_error = 0.0

    # Calculate compression ratio
    original_size = weight.nbytes
    delta_size = delta.nbytes
    compression_ratio = 1 - (delta_size / original_size)

    return {
        "delta_type": delta_type.value,
        "original_size": original_size,
        "delta_size": delta_size,
        "compression_ratio": compression_ratio,
        "encode_time_ms": np.mean(encode_times) * 1000,
        "decode_time_ms": np.mean(decode_times) * 1000,
        "is_lossless": is_lossless,
        "max_error": max_error,
        "mean_error": mean_error,
    }


def run_benchmark(shape: Tuple[int, ...], similarity_level: str = "high"):
    """Run benchmark for all strategies on given weight shape."""
    reference, weight = create_similar_weights(shape, similarity_level)

    strategies = [
        DeltaType.FLOAT32_RAW,
        DeltaType.COMPRESSED,
        DeltaType.XOR_FLOAT32,
        DeltaType.XOR_BFLOAT16,
        DeltaType.EXPONENT_MANTISSA,
        DeltaType.INT8_QUANTIZED,
        DeltaType.INT16_QUANTIZED,
        DeltaType.PER_AXIS_SCALED,
        DeltaType.SPARSE,
    ]

    results = []
    for strategy in strategies:
        try:
            result = benchmark_encoding_strategy(strategy, reference, weight)
            results.append(result)
        except Exception as e:
            print(f"  ⚠️  {strategy.value}: {e}")

    return results


def print_results(results: List[Dict], title: str):
    """Pretty print benchmark results."""
    print(f"\n{'='*70}")
    print(f"📊 {title}")
    print(f"{'='*70}")

    # Header
    print(f"{'Strategy':<20} {'Compression':>12} {'Size':>12} {'Encode':>10} {'Decode':>10} {'Loss':>8}")
    print("-" * 70)

    # Sort by compression ratio
    sorted_results = sorted(results, key=lambda x: x["compression_ratio"], reverse=True)

    for r in sorted_results:
        loss_str = "✓ None" if r["is_lossless"] else f"✗ {r['max_error']:.2e}"
        print(
            f"{r['delta_type']:<20} "
            f"{r['compression_ratio']:>11.1%} "
            f"{r['delta_size']:>10,} B "
            f"{r['encode_time_ms']:>8.2f} ms "
            f"{r['decode_time_ms']:>8.2f} ms "
            f"{loss_str:>8}"
        )


def main():
    print("\n" + "=" * 70)
    print("🚀 Delta Encoding Strategy Benchmark")
    print("=" * 70)

    # Test configurations
    test_cases = [
        ((256, 256), "high", "Large matrix, high similarity (fine-tuning)"),
        ((256, 256), "medium", "Large matrix, medium similarity (training)"),
        ((1024, 1024), "high", "Very large matrix, high similarity"),
        ((64, 64, 3, 3), "high", "Conv layer shape, high similarity"),
        ((10000,), "high", "1D bias vector, high similarity"),
    ]

    for shape, similarity, description in test_cases:
        print(f"\n\n📐 {description}")
        print(f"   Shape: {shape}, Similarity: {similarity}")

        results = run_benchmark(shape, similarity)
        print_results(results, f"Results for {shape}")

    # Summary
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    print("""
Key Findings:
- XOR_FLOAT32: Best lossless compression for similar weights
- EXPONENT_MANTISSA: Good compression with separate component handling
- PER_AXIS_SCALED: Excellent compression for 2D weights (lossy)
- SPARSE: Best for weights with many small differences
- COMPRESSED: Good general-purpose lossless option

Recommendations:
- For training checkpoints: XOR_FLOAT32 or COMPRESSED
- For model versioning: EXPONENT_MANTISSA
- For bandwidth-limited: PER_AXIS_SCALED (if ~5% error acceptable)
    """)


if __name__ == "__main__":
    main()
