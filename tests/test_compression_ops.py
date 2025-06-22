"""Tests for compression operations in weight computation graphs."""

import pytest
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
    compute_svd_compression_ratio,
    analyze_sparsity,
    estimate_best_sparse_format,
    calculate_quantization_params,
    quantize_array,
    dequantize_array,
    estimate_pq_compression_ratio,
    recommend_compression_method,
)


class TestSVDOp:
    """Test SVD compression operation."""
    
    def test_svd_reconstruction_lossless(self):
        """Test perfect reconstruction with full rank SVD."""
        # Create test matrix
        np.random.seed(42)
        matrix = np.random.randn(10, 8).astype(np.float32)
        
        # Compute SVD
        u, s, vt = svd(matrix, full_matrices=False)
        v = vt.T
        
        # Create SVD op with full rank
        op = SVDOp(u, s, v, rank=len(s), lossless=True)
        
        # Reconstruct
        reconstructed = op.forward()
        
        # Should be perfect reconstruction
        np.testing.assert_allclose(reconstructed, matrix, rtol=1e-5, atol=1e-6)
        
    def test_svd_reconstruction_lossy(self):
        """Test lossy reconstruction with truncated SVD."""
        # Create low-rank matrix
        np.random.seed(42)
        rank = 3
        matrix = np.random.randn(10, rank) @ np.random.randn(rank, 8)
        matrix = matrix.astype(np.float32)
        
        # Compute SVD
        u, s, vt = svd(matrix, full_matrices=False)
        v = vt.T
        
        # Create SVD op with reduced rank
        op = SVDOp(u, s, v, rank=rank)
        
        # Reconstruct
        reconstructed = op.forward()
        
        # Should be close but not perfect
        error = np.linalg.norm(reconstructed - matrix) / np.linalg.norm(matrix)
        assert error < 1e-5  # Very small error for true low-rank matrix
        
    def test_svd_memory_usage(self):
        """Test memory usage calculation."""
        # Create test matrix
        m, n, k = 100, 80, 10
        u = np.random.randn(m, k).astype(np.float32)
        s = np.random.rand(k).astype(np.float32)
        v = np.random.randn(n, k).astype(np.float32)
        
        op = SVDOp(u, s, v, rank=k)
        
        # Calculate expected memory
        expected = u.nbytes + s.nbytes + v.nbytes
        assert op.get_memory_usage() == expected
        
    def test_svd_serialization(self):
        """Test SVD serialization and deserialization."""
        # Create test SVD
        np.random.seed(42)
        u = np.random.randn(10, 5).astype(np.float32)
        s = np.random.rand(5).astype(np.float32)
        v = np.random.randn(8, 5).astype(np.float32)
        
        op = SVDOp(u, s, v, rank=3, original_shape=(10, 8))
        
        # Serialize
        data = op.serialize()
        
        # Deserialize
        op2 = SVDOp.deserialize(data)
        
        # Check reconstruction matches
        np.testing.assert_allclose(op.forward(), op2.forward(), rtol=1e-6)
        
    def test_svd_with_reshape(self):
        """Test SVD with shape restoration."""
        # Create 3D tensor
        np.random.seed(42)
        tensor = np.random.randn(4, 5, 6).astype(np.float32)
        matrix = tensor.reshape(4, -1)
        
        # Compute SVD
        u, s, vt = svd(matrix, full_matrices=False)
        v = vt.T
        
        # Create SVD op with original shape
        op = SVDOp(u, s, v, rank=len(s), original_shape=(4, 5, 6))
        
        # Reconstruct
        reconstructed = op.forward()
        
        # Check shape and values
        assert reconstructed.shape == (4, 5, 6)
        np.testing.assert_allclose(reconstructed, tensor, rtol=1e-4, atol=1e-6)


class TestSparseOp:
    """Test sparse matrix compression operation."""
    
    def test_sparse_csr_format(self):
        """Test CSR sparse format."""
        # Create sparse matrix
        np.random.seed(42)
        dense = np.random.randn(10, 8)
        dense[np.abs(dense) < 1.5] = 0  # Make it sparse
        
        sparse_matrix = sparse.csr_matrix(dense)
        op = SparseOp(sparse_matrix, format="csr")
        
        # Reconstruct
        reconstructed = op.forward()
        
        # Should match original
        np.testing.assert_allclose(reconstructed, dense)
        
    def test_sparse_coo_format(self):
        """Test COO sparse format."""
        # Create sparse matrix
        np.random.seed(42)
        dense = np.zeros((8, 10))
        # Add some non-zero elements
        indices = [(1, 2), (3, 4), (5, 6), (7, 8)]
        for i, j in indices:
            dense[i, j] = np.random.randn()
            
        sparse_matrix = sparse.coo_matrix(dense)
        op = SparseOp(sparse_matrix, format="coo")
        
        # Reconstruct
        reconstructed = op.forward()
        
        # Should match original
        np.testing.assert_allclose(reconstructed, dense)
        
    def test_sparse_format_conversion(self):
        """Test automatic format conversion."""
        # Create CSR matrix
        dense = np.random.randn(10, 10)
        dense[np.abs(dense) < 1.0] = 0
        
        csr_matrix = sparse.csr_matrix(dense)
        
        # Create ops with different formats
        ops = {
            "csr": SparseOp(csr_matrix, format="csr"),
            "csc": SparseOp(csr_matrix, format="csc"),
            "coo": SparseOp(csr_matrix, format="coo"),
        }
        
        # All should reconstruct the same
        for format, op in ops.items():
            reconstructed = op.forward()
            np.testing.assert_allclose(reconstructed, dense, err_msg=f"Failed for {format}")
            
    def test_sparse_memory_usage(self):
        """Test memory usage calculation for different formats."""
        # Create very sparse matrix
        dense = np.zeros((100, 100))
        dense[0, 0] = 1.0
        dense[50, 50] = 2.0
        dense[99, 99] = 3.0
        
        # Test different formats
        formats = ["csr", "csc", "coo"]
        for fmt in formats:
            sparse_matrix = sparse.csr_matrix(dense)
            op = SparseOp(sparse_matrix, format=fmt)
            
            memory = op.get_memory_usage()
            assert memory > 0
            # Should be much less than dense storage
            assert memory < dense.nbytes
            
    def test_sparse_serialization(self):
        """Test sparse op serialization."""
        # Create sparse matrix
        dense = np.random.randn(8, 6)
        dense[np.abs(dense) < 1.0] = 0
        
        for fmt in ["csr", "csc", "coo"]:
            sparse_matrix = sparse.csr_matrix(dense)
            op = SparseOp(sparse_matrix, format=fmt)
            
            # Serialize
            data = op.serialize()
            
            # Deserialize
            op2 = SparseOp.deserialize(data)
            
            # Check reconstruction
            np.testing.assert_allclose(op.forward(), op2.forward())


class TestQuantizeOp:
    """Test quantization compression operation."""
    
    def test_quantize_8bit_symmetric(self):
        """Test 8-bit symmetric quantization."""
        # Create test data
        np.random.seed(42)
        data = np.random.randn(10, 8).astype(np.float32)
        
        # Calculate quantization params
        scale, zero_point, dtype = calculate_quantization_params(
            data, bits=8, symmetric=True
        )
        
        # Quantize
        quantized = quantize_array(data, scale, zero_point, dtype)
        
        # Create op
        params = QuantizationParams(
            scale=scale,
            zero_point=zero_point,
            bits=8,
            symmetric=True,
            dtype=dtype
        )
        op = QuantizeOp(quantized, params)
        
        # Dequantize
        reconstructed = op.forward()
        
        # Check error is reasonable
        error = np.mean(np.abs(reconstructed - data))
        assert error < 0.1  # Reasonable for 8-bit
        
    def test_quantize_4bit_asymmetric(self):
        """Test 4-bit asymmetric quantization."""
        # Create test data with positive bias
        np.random.seed(42)
        data = np.random.randn(10, 8).astype(np.float32) + 2.0
        
        # Calculate quantization params
        scale, zero_point, dtype = calculate_quantization_params(
            data, bits=4, symmetric=False
        )
        
        # Quantize
        quantized = quantize_array(data, scale, zero_point, dtype)
        
        # Create op
        params = QuantizationParams(
            scale=scale,
            zero_point=zero_point,
            bits=4,
            symmetric=False,
            dtype=dtype
        )
        op = QuantizeOp(quantized, params)
        
        # Dequantize
        reconstructed = op.forward()
        
        # Check relative error
        rel_error = np.linalg.norm(reconstructed - data) / np.linalg.norm(data)
        assert rel_error < 0.2  # Higher error expected for 4-bit
        
    def test_quantize_memory_usage(self):
        """Test memory usage for different bit widths."""
        data = np.random.randn(100, 100).astype(np.float32)
        
        for bits in [2, 4, 8, 16]:
            scale, zero_point, dtype = calculate_quantization_params(
                data, bits=bits, symmetric=True
            )
            quantized = quantize_array(data, scale, zero_point, dtype)
            
            params = QuantizationParams(
                scale=scale,
                zero_point=zero_point,
                bits=bits,
                symmetric=True,
                dtype=dtype
            )
            op = QuantizeOp(quantized, params)
            
            memory = op.get_memory_usage()
            
            # For sub-byte, we pack into bytes
            if bits < 8:
                expected = (data.size * bits + 7) // 8
            else:
                expected = quantized.nbytes
                
            assert memory == expected
            
    def test_quantize_serialization(self):
        """Test quantization serialization."""
        # Create test data
        data = np.random.randn(5, 6).astype(np.float32)
        
        # Quantize
        scale, zero_point, dtype = calculate_quantization_params(
            data, bits=8, symmetric=True
        )
        quantized = quantize_array(data, scale, zero_point, dtype)
        
        params = QuantizationParams(
            scale=scale,
            zero_point=zero_point,
            bits=8,
            symmetric=True,
            dtype=dtype
        )
        op = QuantizeOp(quantized, params, original_shape=(5, 6))
        
        # Serialize
        serialized = op.serialize()
        
        # Deserialize
        op2 = QuantizeOp.deserialize(serialized)
        
        # Check reconstruction
        np.testing.assert_allclose(op.forward(), op2.forward())


class TestPQOp:
    """Test Product Quantization operation."""
    
    def test_pq_basic_reconstruction(self):
        """Test basic PQ reconstruction."""
        # Setup PQ parameters
        num_vectors = 10
        num_subspaces = 4
        subvector_dim = 8
        codebook_size = 16
        
        # Create codebooks
        np.random.seed(42)
        codebooks = np.random.randn(
            num_subspaces, codebook_size, subvector_dim
        ).astype(np.float32)
        
        # Create random indices
        indices = np.random.randint(
            0, codebook_size, size=(num_vectors, num_subspaces)
        ).astype(np.uint8)
        
        # Create op
        op = PQOp(indices, codebooks)
        
        # Reconstruct
        reconstructed = op.forward()
        
        # Check shape
        expected_shape = (num_vectors, num_subspaces * subvector_dim)
        assert reconstructed.shape == expected_shape
        
    def test_pq_with_residual(self):
        """Test PQ with residual."""
        # Setup
        num_vectors = 5
        num_subspaces = 2
        subvector_dim = 4
        codebook_size = 8
        
        np.random.seed(42)
        codebooks = np.random.randn(
            num_subspaces, codebook_size, subvector_dim
        ).astype(np.float32)
        
        indices = np.random.randint(
            0, codebook_size, size=(num_vectors, num_subspaces)
        ).astype(np.uint8)
        
        # Create residual
        residual = np.random.randn(
            num_vectors * num_subspaces * subvector_dim
        ).astype(np.float32) * 0.1
        
        # Create op
        op = PQOp(indices, codebooks, residual=residual)
        
        # Reconstruct
        reconstructed = op.forward()
        
        # Without residual
        op_no_residual = PQOp(indices, codebooks)
        reconstructed_no_residual = op_no_residual.forward()
        
        # Difference should be the residual
        diff = reconstructed.ravel() - reconstructed_no_residual.ravel()
        np.testing.assert_allclose(diff, residual, rtol=1e-6)
        
    def test_pq_serialization(self):
        """Test PQ serialization."""
        # Setup
        np.random.seed(42)
        codebooks = np.random.randn(2, 4, 3).astype(np.float32)
        indices = np.array([[0, 1], [2, 3], [1, 0]], dtype=np.uint8)
        residual = np.random.randn(18).astype(np.float32) * 0.1
        
        op = PQOp(indices, codebooks, residual=residual, shape=(3, 6))
        
        # Serialize
        data = op.serialize()
        
        # Deserialize
        op2 = PQOp.deserialize(data)
        
        # Check reconstruction
        np.testing.assert_allclose(op.forward(), op2.forward())


class TestCompressionUtils:
    """Test compression utility functions."""
    
    def test_select_rank_by_energy(self):
        """Test rank selection based on energy preservation."""
        # Create singular values with exponential decay
        s = np.exp(-np.arange(20) * 0.5)
        
        # Test different thresholds
        rank_90 = select_rank_by_energy(s, 0.9)
        rank_95 = select_rank_by_energy(s, 0.95)
        rank_99 = select_rank_by_energy(s, 0.99)
        
        # Higher threshold should give higher rank
        assert rank_90 <= rank_95 <= rank_99
        
        # Verify energy preservation
        total_energy = np.sum(s ** 2)
        for threshold, rank in [(0.9, rank_90), (0.95, rank_95), (0.99, rank_99)]:
            preserved_energy = np.sum(s[:rank] ** 2)
            assert preserved_energy / total_energy >= threshold
            
    def test_analyze_sparsity(self):
        """Test sparsity analysis."""
        # Create sparse array
        array = np.zeros((10, 10))
        array[0, 0] = 1.0
        array[5, 5] = -2.0
        array[9, 9] = 0.5
        
        stats = analyze_sparsity(array)
        
        assert stats["sparsity"] == 0.97  # 97% sparse
        assert stats["num_zeros"] == 97
        assert stats["num_nonzeros"] == 3
        assert stats["total_elements"] == 100
        
    def test_estimate_best_sparse_format(self):
        """Test sparse format selection."""
        # Create different sparsity patterns
        
        # Row-oriented sparsity (better for CSR)
        row_sparse = np.zeros((100, 100))
        row_sparse[0, :] = np.random.randn(100)
        
        # Column-oriented sparsity (better for CSC)
        col_sparse = np.zeros((100, 100))
        col_sparse[:, 0] = np.random.randn(100)
        
        # Random sparsity (might be better for COO)
        random_sparse = np.zeros((100, 100))
        indices = np.random.choice(10000, 50, replace=False)
        random_sparse.flat[indices] = np.random.randn(50)
        
        # Test format selection
        for array in [row_sparse, col_sparse, random_sparse]:
            sparse_matrix = sparse.csr_matrix(array)
            best_format = estimate_best_sparse_format(sparse_matrix)
            assert best_format in ["csr", "csc", "coo"]
            
    def test_compression_recommendation(self):
        """Test compression method recommendation."""
        # Test sparse array
        sparse_array = np.zeros((100, 100))
        mask = np.random.rand(100, 100) < 0.1
        num_nonzero = np.sum(mask)
        sparse_array[mask] = np.random.randn(num_nonzero)
        
        rec = recommend_compression_method(sparse_array, target_ratio=5.0)
        assert rec.method == "sparse"
        
        # Test low-rank matrix
        low_rank = np.random.randn(100, 5) @ np.random.randn(5, 50)
        rec = recommend_compression_method(low_rank, target_ratio=3.0)
        assert rec.method in ["svd", "quantize"]
        
        # Test general dense matrix
        dense = np.random.randn(50, 50)
        rec = recommend_compression_method(dense, target_ratio=2.0)
        assert rec.method in ["svd", "quantize"]  # Either could be recommended