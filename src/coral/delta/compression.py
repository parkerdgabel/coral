"""Additional compression utilities for delta encoding."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DeltaCompressor:
    """Additional compression utilities for delta data."""

    @staticmethod
    def compress_sparse_deltas(
        indices: np.ndarray, values: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Compress sparse delta representation using run-length encoding."""
        if len(indices) == 0:
            return np.array([], dtype=np.int32), {
                "compression_type": "sparse_rle",
                "original_length": 0,
            }

        # Sort by indices for better compression
        sorted_order = np.argsort(indices)
        sorted_indices = indices[sorted_order]
        sorted_values = values[sorted_order]

        # Run-length encode index differences
        # First element is the actual index, rest are differences
        index_diffs = np.diff(sorted_indices, prepend=0)
        index_diffs[0] = sorted_indices[0]

        # Simple compression: store (diff, value) pairs
        compressed_data = np.column_stack([index_diffs, sorted_values])

        metadata = {
            "compression_type": "sparse_rle",
            "original_length": len(indices),
            "first_index": int(sorted_indices[0]) if len(sorted_indices) > 0 else 0,
        }

        return compressed_data.astype(np.float32), metadata

    @staticmethod
    def decompress_sparse_deltas(
        compressed_data: np.ndarray, metadata: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decompress sparse delta representation."""
        if metadata["original_length"] == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        # Extract differences and values
        index_diffs = compressed_data[:, 0].astype(np.int64)
        values = compressed_data[:, 1]

        # Reconstruct original indices
        # First element is the actual index, rest are differences
        indices = np.cumsum(index_diffs)

        return indices, values

    @staticmethod
    def adaptive_quantization(
        data: np.ndarray, target_bits: int = 8
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply adaptive quantization based on data distribution."""
        if target_bits not in [8, 16]:
            raise ValueError("target_bits must be 8 or 16")

        # Handle empty arrays
        if data.size == 0:
            dtype = np.int8 if target_bits == 8 else np.int16
            metadata = {
                "scale": 1.0,
                "min_val": 0.0,
                "max_val": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "target_bits": target_bits,
                "outlier_ratio": 0.0,
            }
            return np.array([], dtype=dtype), metadata

        # Analyze data distribution
        std_dev = np.std(data)
        mean_val = np.mean(data)

        # Use 3-sigma range for quantization to capture 99.7% of data
        sigma_range = 3 * std_dev
        min_val = mean_val - sigma_range
        max_val = mean_val + sigma_range

        # Clip outliers
        clipped_data = np.clip(data, min_val, max_val)

        # Quantize
        if target_bits == 8:
            quant_min, quant_max = -128, 127
            dtype = np.int8
        else:
            quant_min, quant_max = -32768, 32767
            dtype = np.int16

        if min_val == max_val:
            scale = 1.0
            quantized = np.zeros_like(clipped_data, dtype=dtype)
        else:
            scale = (max_val - min_val) / (quant_max - quant_min)
            normalized = (clipped_data - min_val) / scale + quant_min
            quantized = np.round(normalized).astype(dtype)

        metadata = {
            "scale": float(scale),
            "min_val": float(min_val),
            "max_val": float(max_val),
            "mean": float(mean_val),
            "std": float(std_dev),
            "target_bits": target_bits,
            "outlier_ratio": float(
                np.sum((data < min_val) | (data > max_val)) / data.size
            ),
        }

        return quantized, metadata

    @staticmethod
    def dequantize_adaptive(
        quantized_data: np.ndarray, metadata: dict[str, Any]
    ) -> np.ndarray:
        """Dequantize adaptively quantized data."""
        scale = metadata["scale"]
        min_val = metadata["min_val"]
        target_bits = metadata["target_bits"]

        if target_bits == 8:
            quant_min = -128
        else:
            quant_min = -32768

        # Dequantize
        dequantized = (quantized_data.astype(np.float32) - quant_min) * scale + min_val

        return dequantized

    @staticmethod
    def compress_with_dictionary(
        data: np.ndarray, dictionary_size: int = 256
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Compress using a learned dictionary of common values."""
        # Flatten data for analysis
        flat_data = data.flatten()

        # Find most common values
        unique_vals, counts = np.unique(flat_data, return_counts=True)

        # Select top values for dictionary
        top_indices = np.argsort(counts)[-dictionary_size:]
        dictionary = unique_vals[top_indices]

        # Create compressed representation
        compressed = np.zeros_like(flat_data, dtype=np.int16)
        is_dict_value = np.zeros_like(flat_data, dtype=bool)

        for i, dict_val in enumerate(dictionary):
            mask = flat_data == dict_val
            compressed[mask] = i
            is_dict_value[mask] = True

        # Handle non-dictionary values (store as-is)
        non_dict_data = flat_data[~is_dict_value]

        metadata = {
            "dictionary": dictionary.tolist(),
            "original_shape": data.shape,
            "dict_coverage": float(np.sum(is_dict_value) / len(flat_data)),
            "non_dict_count": len(non_dict_data),
        }

        return compressed.reshape(data.shape), metadata

    @staticmethod
    def decompress_with_dictionary(
        compressed_data: np.ndarray, metadata: dict[str, Any]
    ) -> np.ndarray:
        """Decompress dictionary-compressed data."""
        dictionary = np.array(metadata["dictionary"])
        original_shape = tuple(metadata["original_shape"])

        # Reconstruct using dictionary lookup
        flat_compressed = compressed_data.flatten()
        flat_reconstructed = dictionary[flat_compressed]

        return flat_reconstructed.reshape(original_shape)

    @staticmethod
    def delta_statistics(delta_data: np.ndarray) -> dict[str, float]:
        """Analyze delta statistics to choose optimal compression."""
        stats = {
            "mean": float(np.mean(delta_data)),
            "std": float(np.std(delta_data)),
            "min": float(np.min(delta_data)),
            "max": float(np.max(delta_data)),
            "sparsity": float(np.sum(np.abs(delta_data) < 1e-6) / delta_data.size),
            "dynamic_range": float(np.max(delta_data) - np.min(delta_data)),
            "entropy": DeltaCompressor._estimate_entropy(delta_data),
            "zero_ratio": float(np.sum(delta_data == 0) / delta_data.size),
        }

        return stats

    @staticmethod
    def _estimate_entropy(data: np.ndarray, bins: int = 256) -> float:
        """Estimate entropy of data distribution."""
        hist, _ = np.histogram(data, bins=bins)
        hist = hist[hist > 0]  # Remove zero bins
        probabilities = hist / np.sum(hist)

        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy)

    @staticmethod
    def recommend_compression(delta_data: np.ndarray) -> str:
        """Recommend optimal compression strategy based on delta characteristics."""
        stats = DeltaCompressor.delta_statistics(delta_data)

        # Decision rules based on characteristics
        if stats["sparsity"] > 0.7:
            return "sparse"
        elif stats["dynamic_range"] < 2.0 and stats["std"] < 0.1:
            return "int8_quantized"
        elif stats["dynamic_range"] < 100.0:
            return "int16_quantized"
        elif stats["entropy"] < 5.0:
            return "compressed"
        else:
            return "float32_raw"
