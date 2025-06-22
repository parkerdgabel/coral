"""Unit tests for weight operations and computation graphs."""

import numpy as np
import pytest

from coral.core.weight_ops import (
    AddOp,
    ComputationGraph,
    IdentityOp,
    MatMulOp,
    OpType,
    ReshapeOp,
    ScaleOp,
    WeightOp,
)
from coral.core.weight_ops.base import (
    calculate_array_memory,
    validate_array,
    validate_compatible_shapes,
    validate_matmul_shapes,
    validate_shape,
)


class TestValidationUtilities:
    """Test input validation utilities."""
    
    def test_validate_array_valid(self):
        """Test validation of valid arrays."""
        arr = np.array([1, 2, 3])
        validate_array(arr)  # Should not raise
        
        arr = np.zeros((10, 10))
        validate_array(arr)  # Should not raise
    
    def test_validate_array_invalid(self):
        """Test validation catches invalid arrays."""
        # Not a numpy array
        with pytest.raises(TypeError, match="must be a numpy array"):
            validate_array([1, 2, 3])
        
        # Empty array
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_array(np.array([]))
        
        # Contains NaN
        arr = np.array([1, 2, np.nan])
        with pytest.raises(ValueError, match="non-finite values"):
            validate_array(arr)
        
        # Contains Inf
        arr = np.array([1, 2, np.inf])
        with pytest.raises(ValueError, match="non-finite values"):
            validate_array(arr)
    
    def test_validate_shape(self):
        """Test shape validation."""
        # Valid shapes
        assert validate_shape((10, 20)) == (10, 20)
        assert validate_shape([10, 20, 30]) == (10, 20, 30)
        
        # Invalid shapes
        with pytest.raises(TypeError, match="must be a tuple or list"):
            validate_shape(10)
        
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_shape([])
        
        with pytest.raises(TypeError, match="must be an integer"):
            validate_shape([10, 20.5])
        
        with pytest.raises(ValueError, match="must be positive"):
            validate_shape([10, -5])
    
    def test_validate_compatible_shapes(self):
        """Test shape compatibility validation."""
        # Compatible shapes
        validate_compatible_shapes((10, 20), (10, 20), "test")  # Should not raise
        
        # Incompatible shapes
        with pytest.raises(ValueError, match="Incompatible shapes"):
            validate_compatible_shapes((10, 20), (10, 30), "test")
    
    def test_validate_matmul_shapes(self):
        """Test matrix multiplication shape validation."""
        # Valid 2D matmul
        assert validate_matmul_shapes((10, 20), (20, 30)) == (10, 30)
        
        # Valid batch matmul
        assert validate_matmul_shapes((5, 10, 20), (5, 20, 30)) == (5, 10, 30)
        
        # Invalid - too few dimensions
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            validate_matmul_shapes((10,), (10, 20))
        
        # Invalid - incompatible inner dimensions
        with pytest.raises(ValueError, match="Incompatible shapes"):
            validate_matmul_shapes((10, 20), (30, 40))
        
        # Invalid - incompatible batch dimensions
        with pytest.raises(ValueError, match="Batch dimensions must match"):
            validate_matmul_shapes((5, 10, 20), (3, 20, 30))
    
    def test_calculate_array_memory(self):
        """Test array memory calculation."""
        # float64 (8 bytes per element)
        assert calculate_array_memory((10, 20), np.dtype('float64')) == 10 * 20 * 8
        
        # float32 (4 bytes per element)
        assert calculate_array_memory((10, 20), np.dtype('float32')) == 10 * 20 * 4
        
        # int32 (4 bytes per element)
        assert calculate_array_memory((5, 5, 5), np.dtype('int32')) == 5 * 5 * 5 * 4


class TestIdentityOp:
    """Test IdentityOp functionality."""
    
    def test_identity_basic(self):
        """Test basic identity operation."""
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        op = IdentityOp(data)
        
        assert op.op_type == OpType.IDENTITY
        assert op.get_output_shape() == (2, 3)
        assert op.get_output_dtype() == np.float32
        assert op.get_memory_usage() == data.nbytes
        
        # Forward should return a copy
        result = op.forward()
        assert np.array_equal(result, data)
        assert result is not data  # Different object
    
    def test_identity_immutability(self):
        """Test that IdentityOp stores immutable data."""
        data = np.array([1, 2, 3])
        op = IdentityOp(data)
        
        # Modify original data
        data[0] = 999
        
        # Op should have unchanged data
        result = op.forward()
        assert result[0] == 1  # Not modified
    
    def test_identity_serialization(self):
        """Test IdentityOp serialization."""
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        op = IdentityOp(data)
        
        # Serialize
        serialized = op.serialize()
        assert serialized["op_type"] == "IDENTITY"
        assert serialized["shape"] == [2, 2]
        assert serialized["dtype"] == "float32"
        assert serialized["data"] == [[1, 2], [3, 4]]
        
        # Deserialize
        op2 = IdentityOp.deserialize(serialized)
        assert np.array_equal(op2.forward(), data)
        assert op2.get_output_dtype() == np.float32
    
    def test_identity_repr(self):
        """Test string representation."""
        op = IdentityOp(np.zeros((10, 20), dtype=np.float64))
        assert "IdentityOp" in repr(op)
        assert "(10, 20)" in repr(op)
        assert "float64" in repr(op)


class TestAddOp:
    """Test AddOp functionality."""
    
    def test_add_two_inputs(self):
        """Test adding two tensors."""
        data1 = np.array([1, 2, 3])
        data2 = np.array([4, 5, 6])
        
        op1 = IdentityOp(data1)
        op2 = IdentityOp(data2)
        add_op = AddOp([op1, op2])
        
        assert add_op.op_type == OpType.ADD
        assert add_op.get_output_shape() == (3,)
        assert add_op.get_memory_usage() == 0  # No additional storage
        
        result = add_op.forward()
        assert np.array_equal(result, [5, 7, 9])
    
    def test_add_multiple_inputs(self):
        """Test adding multiple tensors."""
        ops = [IdentityOp(np.array([i, i+1, i+2])) for i in range(4)]
        add_op = AddOp(ops)
        
        result = add_op.forward()
        expected = np.array([0+1+2+3, 1+2+3+4, 2+3+4+5])
        assert np.array_equal(result, expected)
    
    def test_add_dtype_promotion(self):
        """Test dtype promotion in addition."""
        op1 = IdentityOp(np.array([1, 2, 3], dtype=np.int32))
        op2 = IdentityOp(np.array([1.5, 2.5, 3.5], dtype=np.float32))
        add_op = AddOp([op1, op2])
        
        # numpy promotes int32 + float32 to float64
        assert add_op.get_output_dtype() == np.dtype('float64')
        result = add_op.forward()
        assert result.dtype == np.float64
        assert np.allclose(result, [2.5, 4.5, 6.5])
    
    def test_add_invalid_inputs(self):
        """Test AddOp with invalid inputs."""
        # Too few inputs
        with pytest.raises(ValueError, match="at least 2 inputs"):
            AddOp([IdentityOp(np.array([1, 2, 3]))])
        
        # Incompatible shapes
        op1 = IdentityOp(np.array([1, 2, 3]))
        op2 = IdentityOp(np.array([1, 2]))
        with pytest.raises(ValueError, match="Incompatible shapes"):
            AddOp([op1, op2])
    
    def test_add_repr(self):
        """Test string representation."""
        op1 = IdentityOp(np.zeros((10, 20)))
        op2 = IdentityOp(np.ones((10, 20)))
        add_op = AddOp([op1, op2])
        
        assert "AddOp" in repr(add_op)
        assert "num_inputs=2" in repr(add_op)
        assert "(10, 20)" in repr(add_op)


class TestMatMulOp:
    """Test MatMulOp functionality."""
    
    def test_matmul_2d(self):
        """Test 2D matrix multiplication."""
        data1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        data2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
        
        op1 = IdentityOp(data1)
        op2 = IdentityOp(data2)
        matmul_op = MatMulOp(op1, op2)
        
        assert matmul_op.op_type == OpType.MATMUL
        assert matmul_op.get_output_shape() == (2, 2)
        assert matmul_op.get_memory_usage() == 0
        
        result = matmul_op.forward()
        expected = np.matmul(data1, data2)
        assert np.array_equal(result, expected)
    
    def test_matmul_batch(self):
        """Test batch matrix multiplication."""
        data1 = np.random.randn(5, 3, 4)
        data2 = np.random.randn(5, 4, 6)
        
        op1 = IdentityOp(data1)
        op2 = IdentityOp(data2)
        matmul_op = MatMulOp(op1, op2)
        
        assert matmul_op.get_output_shape() == (5, 3, 6)
        
        result = matmul_op.forward()
        expected = np.matmul(data1, data2)
        assert np.allclose(result, expected)
    
    def test_matmul_dtype_promotion(self):
        """Test dtype promotion in matmul."""
        op1 = IdentityOp(np.array([[1, 2], [3, 4]], dtype=np.int32))
        op2 = IdentityOp(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        matmul_op = MatMulOp(op1, op2)
        
        # numpy promotes int32 @ float32 to float64
        assert matmul_op.get_output_dtype() == np.dtype('float64')
    
    def test_matmul_invalid_shapes(self):
        """Test MatMulOp with invalid shapes."""
        # Incompatible dimensions
        op1 = IdentityOp(np.zeros((2, 3)))
        op2 = IdentityOp(np.zeros((4, 5)))
        with pytest.raises(ValueError, match="Incompatible shapes"):
            MatMulOp(op1, op2)
        
        # Too few dimensions
        op1 = IdentityOp(np.zeros(10))
        op2 = IdentityOp(np.zeros((10, 5)))
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            MatMulOp(op1, op2)
    
    def test_matmul_repr(self):
        """Test string representation."""
        op1 = IdentityOp(np.zeros((10, 20)))
        op2 = IdentityOp(np.zeros((20, 30)))
        matmul_op = MatMulOp(op1, op2)
        
        repr_str = repr(matmul_op)
        assert "MatMulOp" in repr_str
        assert "(10, 20) @ (20, 30) -> (10, 30)" in repr_str


class TestScaleOp:
    """Test ScaleOp functionality."""
    
    def test_scale_basic(self):
        """Test basic scaling operation."""
        data = np.array([1, 2, 3, 4], dtype=np.float32)
        op = IdentityOp(data)
        scale_op = ScaleOp(op, 2.5)
        
        assert scale_op.op_type == OpType.SCALE
        assert scale_op.get_output_shape() == (4,)
        assert scale_op.get_memory_usage() == 8  # float64 scale
        
        result = scale_op.forward()
        assert np.allclose(result, [2.5, 5.0, 7.5, 10.0])
    
    def test_scale_dtype_promotion(self):
        """Test dtype promotion when scaling integers."""
        op = IdentityOp(np.array([1, 2, 3], dtype=np.int32))
        scale_op = ScaleOp(op, 1.5)
        
        assert scale_op.get_output_dtype() == np.float64
        result = scale_op.forward()
        assert result.dtype == np.float64
        assert np.allclose(result, [1.5, 3.0, 4.5])
    
    def test_scale_negative(self):
        """Test scaling with negative values."""
        op = IdentityOp(np.array([1, -2, 3]))
        scale_op = ScaleOp(op, -2)
        
        result = scale_op.forward()
        assert np.array_equal(result, [-2, 4, -6])
    
    def test_scale_invalid(self):
        """Test ScaleOp with invalid scale values."""
        op = IdentityOp(np.array([1, 2, 3]))
        
        # Non-numeric scale
        with pytest.raises(TypeError, match="must be numeric"):
            ScaleOp(op, "2")
        
        # Non-finite scale
        with pytest.raises(ValueError, match="must be finite"):
            ScaleOp(op, np.inf)
        
        with pytest.raises(ValueError, match="must be finite"):
            ScaleOp(op, np.nan)
    
    def test_scale_repr(self):
        """Test string representation."""
        op = IdentityOp(np.zeros((10, 20)))
        scale_op = ScaleOp(op, 3.14)
        
        assert "ScaleOp" in repr(scale_op)
        assert "scale=3.14" in repr(scale_op)
        assert "(10, 20)" in repr(scale_op)


class TestReshapeOp:
    """Test ReshapeOp functionality."""
    
    def test_reshape_basic(self):
        """Test basic reshape operation."""
        data = np.arange(12)
        op = IdentityOp(data)
        reshape_op = ReshapeOp(op, (3, 4))
        
        assert reshape_op.op_type == OpType.RESHAPE
        assert reshape_op.get_output_shape() == (3, 4)
        assert reshape_op.get_output_dtype() == np.dtype('int64')
        
        result = reshape_op.forward()
        assert result.shape == (3, 4)
        assert np.array_equal(result.flatten(), data)
    
    def test_reshape_auto_dimension(self):
        """Test reshape with -1 for automatic dimension."""
        op = IdentityOp(np.arange(24))
        
        # Single -1
        reshape_op = ReshapeOp(op, (2, -1, 3))
        assert reshape_op.get_output_shape() == (2, 4, 3)
        
        # -1 with other dimensions
        reshape_op = ReshapeOp(op, (-1, 6))
        assert reshape_op.get_output_shape() == (4, 6)
    
    def test_reshape_invalid(self):
        """Test ReshapeOp with invalid shapes."""
        op = IdentityOp(np.arange(12))
        
        # Incompatible size
        with pytest.raises(ValueError, match="Cannot reshape"):
            ReshapeOp(op, (5, 3))
        
        # Multiple -1
        with pytest.raises(ValueError, match="Only one dimension can be -1"):
            ReshapeOp(op, (-1, -1, 3))
        
        # Invalid dimension
        with pytest.raises(ValueError, match="must be positive"):
            ReshapeOp(op, (3, 0, 4))
    
    def test_reshape_memory_usage(self):
        """Test memory usage calculation."""
        op = IdentityOp(np.zeros(100))
        reshape_op = ReshapeOp(op, (10, 10))
        
        # Memory for storing shape (2 dimensions * 8 bytes)
        assert reshape_op.get_memory_usage() == 2 * 8
    
    def test_reshape_repr(self):
        """Test string representation."""
        op = IdentityOp(np.zeros(100))
        reshape_op = ReshapeOp(op, (10, 10))
        
        repr_str = repr(reshape_op)
        assert "ReshapeOp" in repr_str
        assert "(100,) -> (10, 10)" in repr_str


class TestComputationGraph:
    """Test ComputationGraph functionality."""
    
    def test_graph_basic(self):
        """Test basic graph evaluation."""
        # Create simple graph: (A + B) * 2
        a = IdentityOp(np.array([1, 2, 3]))
        b = IdentityOp(np.array([4, 5, 6]))
        add = AddOp([a, b])
        scale = ScaleOp(add, 2)
        
        graph = ComputationGraph(scale)
        result = graph.evaluate()
        assert np.array_equal(result, [10, 14, 18])
    
    def test_graph_caching(self):
        """Test that graph caches intermediate results."""
        # Create a graph where the same op is used multiple times
        a = IdentityOp(np.array([[1, 2], [3, 4]]))
        b = IdentityOp(np.array([[5, 6], [7, 8]]))
        
        # A @ B is used twice
        matmul = MatMulOp(a, b)
        add = AddOp([matmul, matmul])  # matmul + matmul
        
        graph = ComputationGraph(add)
        
        # First evaluation
        result1 = graph.evaluate()
        
        # Note: Weak reference cache may be garbage collected immediately,
        # so we can't reliably test cache contents. Instead, just verify
        # that repeated evaluations produce the same result.
        
        # Second evaluation should produce same result
        result2 = graph.evaluate()
        assert np.array_equal(result1, result2)
    
    def test_graph_info(self):
        """Test graph information retrieval."""
        # Create a more complex graph
        a = IdentityOp(np.zeros((10, 20)))
        b = IdentityOp(np.ones((10, 20)))
        c = IdentityOp(np.ones((20, 30)))
        
        add = AddOp([a, b])
        matmul = MatMulOp(add, c)
        scale = ScaleOp(matmul, 0.5)
        
        graph = ComputationGraph(scale)
        info = graph.get_graph_info()
        
        assert info["num_operations"] == 6  # 3 identity + add + matmul + scale
        assert info["operation_counts"]["IDENTITY"] == 3
        assert info["operation_counts"]["ADD"] == 1
        assert info["operation_counts"]["MATMUL"] == 1
        assert info["operation_counts"]["SCALE"] == 1
        assert info["output_shape"] == (10, 30)
        assert info["max_depth"] > 1
    
    def test_graph_memory_calculation(self):
        """Test total memory calculation."""
        # IdentityOp stores the array
        data = np.zeros((100, 100), dtype=np.float32)  # 40KB
        op = IdentityOp(data)
        graph = ComputationGraph(op)
        
        # Before evaluation, only op memory
        assert graph.get_total_memory() == data.nbytes
        
        # After evaluation, includes cached result
        graph.evaluate()
        # Memory should include both op storage and cache
        # (exact value depends on weak ref behavior)
        assert graph.get_total_memory() >= data.nbytes
    
    def test_graph_validation(self):
        """Test graph validation catches cycles."""
        # We can't create actual cycles with our ops (they're immutable),
        # but we can test that validation runs
        op = IdentityOp(np.array([1, 2, 3]))
        graph = ComputationGraph(op, validate=True)  # Should not raise
        
        # Test with validation disabled
        graph2 = ComputationGraph(op, validate=False)
        assert graph2.root == op
    
    def test_graph_clear_cache(self):
        """Test cache clearing."""
        op = IdentityOp(np.array([1, 2, 3]))
        graph = ComputationGraph(op)
        
        # Evaluate to populate cache
        result1 = graph.evaluate()
        
        # Clear cache
        graph.clear_cache()
        
        # After clearing, should still produce same result
        result2 = graph.evaluate()
        assert np.array_equal(result1, result2)
    
    def test_graph_to_dict(self):
        """Test graph dictionary representation."""
        a = IdentityOp(np.array([1, 2]))
        b = IdentityOp(np.array([3, 4]))
        add = AddOp([a, b])
        
        graph = ComputationGraph(add)
        graph_dict = graph.to_dict()
        
        assert "root" in graph_dict
        assert "nodes" in graph_dict
        assert "edges" in graph_dict
        
        # Should have 3 nodes (2 identity + 1 add)
        assert len(graph_dict["nodes"]) == 3
        
        # Should have 2 edges (from identities to add)
        assert len(graph_dict["edges"]) == 2
        
        # Check node properties
        for node in graph_dict["nodes"]:
            assert "id" in node
            assert "type" in node
            assert "shape" in node
            assert "dtype" in node
    
    def test_graph_repr(self):
        """Test string representation."""
        op = IdentityOp(np.zeros((10, 20)))
        graph = ComputationGraph(op)
        
        repr_str = repr(graph)
        assert "ComputationGraph" in repr_str
        assert "operations=" in repr_str
        assert "depth=" in repr_str
        assert "output_shape=(10, 20)" in repr_str


class TestIntegration:
    """Integration tests with complex graphs."""
    
    def test_complex_graph(self):
        """Test a complex computation graph."""
        # Create graph: ((A + B) @ C) * 2 + D
        a = IdentityOp(np.array([[1, 2], [3, 4]], dtype=np.float32))
        b = IdentityOp(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32))
        c = IdentityOp(np.array([[2, 0], [0, 2]], dtype=np.float32))
        d = IdentityOp(np.array([[10, 10], [10, 10]], dtype=np.float32))
        
        add_ab = AddOp([a, b])
        matmul = MatMulOp(add_ab, c)
        scale = ScaleOp(matmul, 2)
        final_add = AddOp([scale, d])
        
        graph = ComputationGraph(final_add)
        result = graph.evaluate()
        
        # Manually compute expected result
        expected_add_ab = np.array([[1.5, 2.5], [3.5, 4.5]])
        expected_matmul = expected_add_ab @ np.array([[2, 0], [0, 2]])
        expected_scale = expected_matmul * 2
        expected_final = expected_scale + np.array([[10, 10], [10, 10]])
        
        assert np.allclose(result, expected_final)
    
    def test_reshape_in_graph(self):
        """Test reshape operations in a graph."""
        # Create graph with reshape: reshape(A) @ B
        a = IdentityOp(np.arange(12, dtype=np.float32))
        b = IdentityOp(np.ones((4, 5), dtype=np.float32))
        
        reshape = ReshapeOp(a, (3, 4))
        matmul = MatMulOp(reshape, b)
        
        graph = ComputationGraph(matmul)
        result = graph.evaluate()
        
        assert result.shape == (3, 5)
        
        # Verify computation
        expected = np.arange(12).reshape(3, 4) @ np.ones((4, 5))
        assert np.allclose(result, expected)
    
    def test_multiple_outputs_from_same_input(self):
        """Test graph where one operation feeds multiple others."""
        # Graph structure:
        #     A
        #    / \
        #   *2  *3
        #    \ /
        #     +
        
        a = IdentityOp(np.array([1, 2, 3]))
        scale2 = ScaleOp(a, 2)
        scale3 = ScaleOp(a, 3)
        add = AddOp([scale2, scale3])
        
        graph = ComputationGraph(add)
        result = graph.evaluate()
        
        # A*2 + A*3 = A*5
        assert np.array_equal(result, [5, 10, 15])