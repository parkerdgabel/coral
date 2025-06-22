#!/usr/bin/env python3
"""Demonstration of advanced compression techniques using computation graphs."""

import numpy as np
from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.core.weight_ops import (
    ComputationGraph, IdentityOp, SVDOp, SparseOp, QuantizeOp,
    select_rank_by_energy, analyze_sparsity, calculate_quantization_params
)
import scipy.sparse as sp


def demo_svd_compression():
    """Demonstrate SVD-based low-rank compression."""
    print("\n=== SVD Compression Demo ===")
    
    # Create a low-rank matrix (rank 5 embedded in 100x50)
    true_rank = 5
    U_true = np.random.randn(100, true_rank)
    V_true = np.random.randn(true_rank, 50)
    low_rank_matrix = U_true @ V_true
    
    # Add small noise
    noisy_matrix = low_rank_matrix + 0.01 * np.random.randn(100, 50)
    
    # Create weight tensor the old way
    weight_old = WeightTensor(
        data=noisy_matrix,
        metadata=WeightMetadata(name="dense_weight", shape=(100, 50), dtype=np.float32)
    )
    print(f"Original weight size: {weight_old.nbytes:,} bytes")
    
    # Perform SVD
    U, S, Vt = np.linalg.svd(noisy_matrix, full_matrices=False)
    
    # Select rank to preserve 99% energy
    optimal_rank = select_rank_by_energy(S, energy_threshold=0.99)
    print(f"Optimal rank for 99% energy: {optimal_rank} (true rank: {true_rank})")
    
    # Create SVD-compressed weight
    svd_op = SVDOp(U[:, :optimal_rank], S[:optimal_rank], Vt[:optimal_rank, :])
    weight_compressed = WeightTensor(
        computation_graph=ComputationGraph(svd_op),
        metadata=WeightMetadata(name="svd_weight", shape=(100, 50), dtype=np.float32)
    )
    
    # Compare sizes
    svd_size = svd_op.get_memory_usage()
    print(f"SVD compressed size: {svd_size:,} bytes")
    print(f"Compression ratio: {weight_old.nbytes / svd_size:.2f}x")
    
    # Verify reconstruction
    reconstructed = weight_compressed.data
    error = np.linalg.norm(noisy_matrix - reconstructed) / np.linalg.norm(noisy_matrix)
    print(f"Reconstruction error: {error:.6f}")


def demo_sparse_compression():
    """Demonstrate sparse matrix compression."""
    print("\n=== Sparse Compression Demo ===")
    
    # Create a sparse matrix
    size = 1000
    density = 0.01  # 1% non-zero
    sparse_data = sp.random(size, size, density=density, format='csr')
    dense_data = sparse_data.toarray()
    
    # Analyze sparsity
    stats = analyze_sparsity(dense_data)
    print(f"Sparsity: {stats['sparsity']:.2%}")
    print(f"Non-zero elements: {stats['nnz']:,}")
    
    # Create dense weight
    weight_dense = WeightTensor(
        data=dense_data,
        metadata=WeightMetadata(name="dense_sparse", shape=(size, size), dtype=np.float32)
    )
    print(f"Dense storage: {weight_dense.nbytes:,} bytes")
    
    # Create sparse weight
    sparse_op = SparseOp(sparse_data, format='csr')
    weight_sparse = WeightTensor(
        computation_graph=ComputationGraph(sparse_op),
        metadata=WeightMetadata(name="sparse_weight", shape=(size, size), dtype=np.float32)
    )
    
    sparse_size = sparse_op.get_memory_usage()
    print(f"Sparse storage: {sparse_size:,} bytes")
    print(f"Compression ratio: {weight_dense.nbytes / sparse_size:.2f}x")


def demo_quantization():
    """Demonstrate quantization compression."""
    print("\n=== Quantization Demo ===")
    
    # Create weight with limited value range
    weight_data = np.random.normal(0, 1, size=(512, 512)).astype(np.float32)
    weight_data = np.clip(weight_data, -3, 3)  # Clip to [-3, 3]
    
    # Original weight
    weight_original = WeightTensor(
        data=weight_data,
        metadata=WeightMetadata(name="original", shape=weight_data.shape, dtype=np.float32)
    )
    print(f"Original size (float32): {weight_original.nbytes:,} bytes")
    
    # Quantize to 8-bit
    scale, zero_point = calculate_quantization_params(weight_data, bits=8, symmetric=True)
    quantized_data = np.round(weight_data / scale).astype(np.int8)
    
    quant_op = QuantizeOp(quantized_data, scale, zero_point, bits=8, symmetric=True)
    weight_quantized = WeightTensor(
        computation_graph=ComputationGraph(quant_op),
        metadata=WeightMetadata(name="quantized", shape=weight_data.shape, dtype=np.float32)
    )
    
    quant_size = quant_op.get_memory_usage()
    print(f"Quantized size (int8): {quant_size:,} bytes")
    print(f"Compression ratio: {weight_original.nbytes / quant_size:.2f}x")
    
    # Check quantization error
    reconstructed = weight_quantized.data
    mse = np.mean((weight_data - reconstructed) ** 2)
    print(f"Quantization MSE: {mse:.6f}")


def demo_combined_compression():
    """Demonstrate combining multiple compression techniques."""
    print("\n=== Combined Compression Demo ===")
    
    # Create a structured weight matrix (e.g., from a neural network layer)
    # Low-rank structure with sparsity
    rank = 10
    size = 500
    
    # Generate low-rank component
    U = np.random.randn(size, rank)
    V = np.random.randn(rank, size)
    low_rank = U @ V
    
    # Add structured sparsity (zero out blocks)
    mask = np.random.random((size, size)) > 0.7  # 70% zeros
    sparse_low_rank = low_rank * mask
    
    # Original weight
    weight_original = WeightTensor(
        data=sparse_low_rank.astype(np.float32),
        metadata=WeightMetadata(name="original", shape=(size, size), dtype=np.float32)
    )
    print(f"Original size: {weight_original.nbytes:,} bytes")
    
    # Method 1: Direct sparse encoding
    sparse_matrix = sp.csr_matrix(sparse_low_rank)
    sparse_op = SparseOp(sparse_matrix)
    weight_sparse = WeightTensor(
        computation_graph=ComputationGraph(sparse_op),
        metadata=WeightMetadata(name="sparse", shape=(size, size), dtype=np.float32)
    )
    print(f"Sparse only: {sparse_op.get_memory_usage():,} bytes")
    
    # Method 2: SVD then sparsify (for comparison)
    U_svd, S_svd, Vt_svd = np.linalg.svd(sparse_low_rank, full_matrices=False)
    optimal_rank = select_rank_by_energy(S_svd, 0.99)
    svd_op = SVDOp(U_svd[:, :optimal_rank], S_svd[:optimal_rank], Vt_svd[:optimal_rank, :])
    weight_svd = WeightTensor(
        computation_graph=ComputationGraph(svd_op),
        metadata=WeightMetadata(name="svd", shape=(size, size), dtype=np.float32)
    )
    print(f"SVD only (rank {optimal_rank}): {svd_op.get_memory_usage():,} bytes")
    
    # Compare compression ratios
    print(f"\nCompression ratios:")
    print(f"  Sparse: {weight_original.nbytes / sparse_op.get_memory_usage():.2f}x")
    print(f"  SVD: {weight_original.nbytes / svd_op.get_memory_usage():.2f}x")


def demo_weight_analysis():
    """Demonstrate automatic compression recommendation."""
    print("\n=== Automatic Compression Analysis ===")
    
    from coral.core.weight_ops.compression_utils import recommend_compression_method
    
    # Test different weight patterns
    test_weights = {
        "sparse_weight": sp.random(1000, 1000, density=0.05).toarray(),
        "low_rank_weight": np.random.randn(200, 10) @ np.random.randn(10, 200),
        "quantizable_weight": np.random.randint(-10, 10, size=(500, 500)).astype(np.float32) / 10,
        "dense_random": np.random.randn(100, 100)
    }
    
    for name, data in test_weights.items():
        print(f"\n{name}:")
        recommendation = recommend_compression_method(data)
        print(f"  Recommended: {recommendation['method']}")
        print(f"  Reason: {recommendation['reason']}")
        print(f"  Estimated compression: {recommendation['estimated_compression']:.2f}x")


if __name__ == "__main__":
    print("=== Coral Advanced Compression Demo ===")
    print("Demonstrating computation graph-based weight compression")
    
    demo_svd_compression()
    demo_sparse_compression()
    demo_quantization()
    demo_combined_compression()
    demo_weight_analysis()
    
    print("\n✨ These compression techniques can be combined and applied")
    print("   automatically across entire model repositories for")
    print("   10-100x storage savings with minimal accuracy loss!")