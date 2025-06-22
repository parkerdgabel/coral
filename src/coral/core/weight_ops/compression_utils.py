"""Utilities for compression operations."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse


def select_rank_by_energy(singular_values: np.ndarray, energy_threshold: float = 0.99) -> int:
    """Select rank for SVD based on energy preservation.
    
    Args:
        singular_values: Array of singular values
        energy_threshold: Fraction of total energy to preserve (0-1)
        
    Returns:
        Optimal rank that preserves the specified energy
    """
    if energy_threshold <= 0 or energy_threshold > 1:
        raise ValueError(f"Energy threshold must be in (0, 1], got {energy_threshold}")
    
    # Calculate cumulative energy
    total_energy = np.sum(singular_values ** 2)
    if total_energy == 0:
        return 1
    
    cumulative_energy = np.cumsum(singular_values ** 2)
    energy_fraction = cumulative_energy / total_energy
    
    # Find minimum rank that preserves desired energy
    rank = np.argmax(energy_fraction >= energy_threshold) + 1
    return min(rank, len(singular_values))


def analyze_sparsity(array: np.ndarray, threshold: float = 1e-6) -> Dict[str, float]:
    """Analyze sparsity patterns in an array.
    
    Args:
        array: Input array to analyze
        threshold: Values below this are considered zero
        
    Returns:
        Dictionary with sparsity statistics
    """
    total_elements = array.size
    zero_elements = np.sum(np.abs(array) < threshold)
    
    return {
        "sparsity": zero_elements / total_elements,
        "nnz": total_elements - zero_elements,
        "density": 1 - (zero_elements / total_elements),
        "zero_blocks": _count_zero_blocks(array, threshold),
        "structured_sparsity": _measure_structured_sparsity(array, threshold)
    }


def _count_zero_blocks(array: np.ndarray, threshold: float) -> int:
    """Count number of contiguous zero blocks."""
    flat = array.flatten()
    is_zero = np.abs(flat) < threshold
    
    # Count transitions from non-zero to zero
    transitions = np.diff(np.concatenate([[False], is_zero, [False]]))
    return np.sum(transitions == 1)


def _measure_structured_sparsity(array: np.ndarray, threshold: float) -> float:
    """Measure how structured the sparsity pattern is."""
    if array.ndim < 2:
        return 0.0
    
    # Check row and column sparsity
    is_zero = np.abs(array) < threshold
    
    if array.ndim == 2:
        # Fraction of fully zero rows/columns
        zero_rows = np.all(is_zero, axis=1).sum()
        zero_cols = np.all(is_zero, axis=0).sum()
        return (zero_rows + zero_cols) / (array.shape[0] + array.shape[1])
    else:
        # For higher dimensions, check along different axes
        structured_score = 0
        for axis in range(array.ndim):
            zero_slices = np.all(is_zero, axis=axis).sum()
            structured_score += zero_slices / array.shape[axis]
        return structured_score / array.ndim


def select_sparse_format(array: np.ndarray, threshold: float = 1e-6) -> str:
    """Select optimal sparse format based on sparsity pattern.
    
    Args:
        array: Input array
        threshold: Values below this are considered zero
        
    Returns:
        Recommended sparse format ('csr', 'csc', 'coo')
    """
    if array.ndim != 2:
        return "coo"  # COO works for any dimensionality
    
    stats = analyze_sparsity(array, threshold)
    
    # Very sparse -> COO is most efficient
    if stats["sparsity"] > 0.9:
        return "coo"
    
    # Check row vs column structure
    is_zero = np.abs(array) < threshold
    row_density = np.mean(np.sum(~is_zero, axis=1))
    col_density = np.mean(np.sum(~is_zero, axis=0))
    
    # Use CSR if rows are denser, CSC if columns are denser
    if row_density > col_density * 1.2:
        return "csr"
    elif col_density > row_density * 1.2:
        return "csc"
    else:
        return "csr"  # Default to CSR


def calculate_quantization_params(
    array: np.ndarray,
    bits: int,
    symmetric: bool = True
) -> Tuple[float, float]:
    """Calculate scale and zero point for quantization.
    
    Args:
        array: Array to quantize
        bits: Number of bits for quantization
        symmetric: Whether to use symmetric quantization
        
    Returns:
        Tuple of (scale, zero_point)
    """
    if symmetric:
        # Symmetric quantization
        max_val = np.max(np.abs(array))
        if max_val == 0:
            return 1.0, 0
        
        qmax = 2 ** (bits - 1) - 1
        scale = max_val / qmax
        zero_point = 0
    else:
        # Asymmetric quantization
        min_val = np.min(array)
        max_val = np.max(array)
        
        if min_val == max_val:
            return 1.0, 0
        
        qmin = 0
        qmax = 2 ** bits - 1
        
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
    
    return scale, zero_point


def quantize_array(
    array: np.ndarray,
    scale: float,
    zero_point: float,
    bits: int
) -> np.ndarray:
    """Quantize array using provided parameters.
    
    Args:
        array: Array to quantize
        scale: Quantization scale
        zero_point: Quantization zero point
        bits: Number of bits
        
    Returns:
        Quantized array
    """
    if bits == 8:
        dtype = np.int8 if zero_point == 0 else np.uint8
        qmin = -128 if zero_point == 0 else 0
        qmax = 127 if zero_point == 0 else 255
    elif bits == 16:
        dtype = np.int16 if zero_point == 0 else np.uint16
        qmin = -32768 if zero_point == 0 else 0
        qmax = 32767 if zero_point == 0 else 65535
    else:
        raise ValueError(f"Unsupported bit width: {bits}")
    
    # Quantize
    quantized = np.round(array / scale + zero_point)
    quantized = np.clip(quantized, qmin, qmax).astype(dtype)
    
    return quantized


def dequantize_array(
    quantized: np.ndarray,
    scale: float,
    zero_point: float
) -> np.ndarray:
    """Dequantize array using provided parameters.
    
    Args:
        quantized: Quantized array
        scale: Quantization scale
        zero_point: Quantization zero point
        
    Returns:
        Dequantized array
    """
    return (quantized.astype(np.float32) - zero_point) * scale


def estimate_compression_ratio(
    original_bytes: int,
    compressed_bytes: int
) -> float:
    """Calculate compression ratio.
    
    Args:
        original_bytes: Original size in bytes
        compressed_bytes: Compressed size in bytes
        
    Returns:
        Compression ratio (>1 means compression, <1 means expansion)
    """
    if compressed_bytes == 0:
        return float('inf')
    return original_bytes / compressed_bytes


def recommend_compression_method(
    array: np.ndarray,
    threshold: float = 1e-6
) -> Dict[str, any]:
    """Recommend best compression method for given array.
    
    Args:
        array: Input array
        threshold: Threshold for considering values as zero
        
    Returns:
        Dictionary with recommendation and reasoning
    """
    sparsity_stats = analyze_sparsity(array, threshold)
    
    # Very sparse -> use sparse format
    if sparsity_stats["sparsity"] > 0.8:
        return {
            "method": "sparse",
            "format": select_sparse_format(array, threshold),
            "reason": f"High sparsity ({sparsity_stats['sparsity']:.2%})",
            "estimated_compression": 1 / (1 - sparsity_stats["sparsity"])
        }
    
    # Check if low-rank approximation would work well
    if array.ndim >= 2:
        # Reshape to 2D for SVD analysis
        original_shape = array.shape
        if array.ndim > 2:
            array_2d = array.reshape(original_shape[0], -1)
        else:
            array_2d = array
        
        # Do truncated SVD to estimate rank
        try:
            from scipy.linalg import svd
            _, s, _ = svd(array_2d, full_matrices=False)
            rank_90 = select_rank_by_energy(s, 0.90)
            rank_99 = select_rank_by_energy(s, 0.99)
            
            # If rank is much smaller than dimensions, SVD is good
            max_dim = max(array_2d.shape)
            if rank_99 < max_dim * 0.5:
                compression_ratio = (array_2d.shape[0] * array_2d.shape[1]) / (
                    (array_2d.shape[0] + array_2d.shape[1]) * rank_99
                )
                return {
                    "method": "svd",
                    "rank": rank_99,
                    "reason": f"Low rank structure (rank {rank_99} captures 99% energy)",
                    "estimated_compression": compression_ratio
                }
        except:
            pass
    
    # Check value distribution for quantization
    value_range = np.max(array) - np.min(array)
    value_std = np.std(array)
    
    if value_range > 0 and value_std / value_range < 0.5:
        # Values are clustered, quantization might work well
        return {
            "method": "quantize",
            "bits": 8,
            "reason": "Values are clustered, suitable for quantization",
            "estimated_compression": array.dtype.itemsize / 1  # Assuming 8-bit quantization
        }
    
    # Default to product quantization for large arrays
    if array.size > 10000:
        return {
            "method": "pq",
            "reason": "Large array suitable for product quantization",
            "estimated_compression": 4.0  # Typical PQ compression
        }
    
    # No compression recommended
    return {
        "method": "none",
        "reason": "Array too small or no clear compression benefit",
        "estimated_compression": 1.0
    }