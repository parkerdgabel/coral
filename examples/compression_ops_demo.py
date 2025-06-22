"""Demonstration of compression operations for weight computation graphs."""

import numpy as np
from scipy import sparse
from scipy.linalg import svd

from coral.core.weight_ops.compression_ops import (
    SVDOp,
    SparseOp,
    QuantizeOp,
    PQOp,
    QuantizationParams,
)
from coral.core.weight_ops.compression_utils import (
    select_rank_by_energy,
    analyze_sparsity,
    calculate_quantization_params,
    quantize_array,
    recommend_compression_method,
)


def demo_svd_compression():
    """Demonstrate SVD compression for low-rank matrices."""
    print("=== SVD Compression Demo ===")
    
    # Create a low-rank matrix (rank 5)
    np.random.seed(42)
    true_rank = 5
    m, n = 100, 80
    low_rank_matrix = np.random.randn(m, true_rank) @ np.random.randn(true_rank, n)
    
    # Add some noise
    low_rank_matrix += 0.01 * np.random.randn(m, n)
    low_rank_matrix = low_rank_matrix.astype(np.float32)
    
    # Compute SVD
    u, s, vt = svd(low_rank_matrix, full_matrices=False)
    v = vt.T
    
    # Select rank based on energy preservation
    selected_rank = select_rank_by_energy(s, energy_threshold=0.99)
    print(f"Matrix shape: {low_rank_matrix.shape}")
    print(f"True rank: {true_rank}")
    print(f"Selected rank (99% energy): {selected_rank}")
    
    # Create SVD operation
    svd_op = SVDOp(u, s, v, rank=selected_rank)
    
    # Reconstruct
    reconstructed = svd_op.forward()
    
    # Calculate compression and error
    original_size = low_rank_matrix.nbytes
    compressed_size = svd_op.get_memory_usage()
    compression_ratio = original_size / compressed_size
    
    error = np.linalg.norm(reconstructed - low_rank_matrix) / np.linalg.norm(low_rank_matrix)
    
    print(f"Original size: {original_size:,} bytes")
    print(f"Compressed size: {compressed_size:,} bytes")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Reconstruction error: {error:.6f}")
    print()


def demo_sparse_compression():
    """Demonstrate sparse matrix compression."""
    print("=== Sparse Compression Demo ===")
    
    # Create a sparse matrix
    np.random.seed(42)
    size = 100
    dense = np.zeros((size, size))
    
    # Add sparse pattern
    num_nonzeros = int(0.05 * size * size)  # 5% density
    indices = np.random.choice(size * size, num_nonzeros, replace=False)
    dense.flat[indices] = np.random.randn(num_nonzeros)
    
    # Analyze sparsity
    stats = analyze_sparsity(dense)
    print(f"Matrix shape: {dense.shape}")
    print(f"Sparsity: {stats['sparsity']:.1%}")
    print(f"Non-zeros: {stats['num_nonzeros']}")
    
    # Create sparse operation
    sparse_matrix = sparse.csr_matrix(dense)
    sparse_op = SparseOp(sparse_matrix, format="csr")
    
    # Calculate compression
    original_size = dense.nbytes
    compressed_size = sparse_op.get_memory_usage()
    compression_ratio = original_size / compressed_size
    
    print(f"Original size: {original_size:,} bytes")
    print(f"Compressed size: {compressed_size:,} bytes")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print()


def demo_quantization():
    """Demonstrate quantization compression."""
    print("=== Quantization Demo ===")
    
    # Create weight matrix
    np.random.seed(42)
    weights = np.random.randn(64, 128).astype(np.float32)
    
    # Try different bit widths
    for bits in [8, 4, 2]:
        print(f"\n{bits}-bit Quantization:")
        
        # Calculate quantization parameters
        scale, zero_point, dtype = calculate_quantization_params(
            weights, bits=bits, symmetric=True
        )
        
        # Quantize
        quantized = quantize_array(weights, scale, zero_point, dtype)
        
        # Create quantization operation
        params = QuantizationParams(
            scale=scale,
            zero_point=zero_point,
            bits=bits,
            symmetric=True,
            dtype=dtype
        )
        quant_op = QuantizeOp(quantized, params)
        
        # Reconstruct
        reconstructed = quant_op.forward()
        
        # Calculate metrics
        original_size = weights.nbytes
        compressed_size = quant_op.get_memory_usage()
        compression_ratio = original_size / compressed_size
        
        error = np.linalg.norm(reconstructed - weights) / np.linalg.norm(weights)
        
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Reconstruction error: {error:.6f}")


def demo_pq_compression():
    """Demonstrate Product Quantization compression."""
    print("\n=== Product Quantization Demo ===")
    
    # Create weight matrix
    np.random.seed(42)
    num_weights = 1000
    weight_dim = 128
    weights = np.random.randn(num_weights, weight_dim).astype(np.float32)
    
    # PQ parameters
    num_subspaces = 8
    codebook_size = 256
    subvector_dim = weight_dim // num_subspaces
    
    print(f"Weight shape: {weights.shape}")
    print(f"PQ parameters: {num_subspaces} subspaces, {codebook_size} codes each")
    
    # Simulate PQ encoding (simplified - normally would use k-means)
    # Create random codebooks
    codebooks = np.random.randn(
        num_subspaces, codebook_size, subvector_dim
    ).astype(np.float32)
    
    # Assign random indices
    indices = np.random.randint(
        0, codebook_size, size=(num_weights, num_subspaces)
    ).astype(np.uint8)
    
    # Create PQ operation
    pq_op = PQOp(indices, codebooks)
    
    # Calculate compression
    original_size = weights.nbytes
    compressed_size = pq_op.get_memory_usage()
    compression_ratio = original_size / compressed_size
    
    print(f"Original size: {original_size:,} bytes")
    print(f"Compressed size: {compressed_size:,} bytes")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print()


def demo_compression_recommendation():
    """Demonstrate automatic compression method selection."""
    print("=== Compression Recommendation Demo ===")
    
    # Test different weight patterns
    test_cases = [
        ("Sparse weights", np.random.randn(100, 100) * (np.random.rand(100, 100) < 0.1)),
        ("Low-rank weights", np.random.randn(100, 10) @ np.random.randn(10, 100)),
        ("Dense weights", np.random.randn(50, 50)),
    ]
    
    for name, weights in test_cases:
        print(f"\n{name}:")
        weights = weights.astype(np.float32)
        
        # Get recommendation
        rec = recommend_compression_method(weights, target_ratio=4.0)
        
        print(f"  Recommended method: {rec.method}")
        print(f"  Estimated compression: {rec.estimated_ratio:.2f}x")
        print(f"  Reason: {rec.reason}")
        if rec.parameters:
            print(f"  Parameters: {rec.parameters}")


if __name__ == "__main__":
    demo_svd_compression()
    demo_sparse_compression()
    demo_quantization()
    demo_pq_compression()
    demo_compression_recommendation()