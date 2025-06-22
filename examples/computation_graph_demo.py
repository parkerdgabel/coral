"""Demonstration of WeightTensor with computation graph support.

This example shows how to use the new computation graph functionality
in WeightTensor for lazy evaluation and memory-efficient representation.
"""

import numpy as np
from coral.core.weight_tensor import WeightTensor
from coral.core.weight_ops import (
    ComputationGraph,
    IdentityOp,
    AddOp,
    ScaleOp,
    MatMulOp,
    ReshapeOp,
)


def basic_graph_example():
    """Basic example of creating a weight tensor with computation graph."""
    print("=== Basic Computation Graph Example ===")
    
    # Create a simple computation graph: scale a weight by 2
    data = np.ones((5, 5), dtype=np.float32)
    scale_op = ScaleOp(IdentityOp(data), 2.0)
    graph = ComputationGraph(scale_op)
    
    # Create weight tensor with graph
    weight = WeightTensor(computation_graph=graph)
    
    print(f"Weight tensor created: {weight}")
    print(f"Operation type: {weight.get_operation_type()}")
    print(f"Shape (without evaluation): {weight.shape}")
    print(f"Memory usage: {weight.nbytes} bytes")
    
    # Lazy evaluation - data is computed only when accessed
    print("\nAccessing data (triggers evaluation)...")
    result = weight.data
    print(f"Result shape: {result.shape}")
    print(f"Result mean: {result.mean()}")
    print(f"Materialized: {weight._materialized}")


def complex_graph_example():
    """Example with more complex computation graph."""
    print("\n=== Complex Computation Graph Example ===")
    
    # Create a complex graph: (A + B) * C
    a = np.ones((4, 4), dtype=np.float32)
    b = np.ones((4, 4), dtype=np.float32) * 2
    c = np.array([[2.0]], dtype=np.float32)  # Scalar as 1x1 matrix
    
    # Build the graph
    add_op = AddOp([IdentityOp(a), IdentityOp(b)])
    scale_op = ScaleOp(add_op, 3.0)
    graph = ComputationGraph(scale_op)
    
    # Create weight tensor
    weight = WeightTensor(computation_graph=graph)
    
    print(f"Complex graph: {weight}")
    print(f"Expected result: (1 + 2) * 3 = 9")
    print(f"Actual result mean: {weight.data.mean()}")


def compression_example():
    """Example of applying compression operations."""
    print("\n=== Compression Operation Example ===")
    
    # Start with a regular weight tensor
    data = np.random.randn(10, 10).astype(np.float32)
    weight = WeightTensor(data=data)
    
    print(f"Original weight: {weight}")
    print(f"Original mean: {data.mean():.4f}")
    
    # Apply a compression operation (e.g., quantization simulation via scaling)
    # In real implementation, this would be a QuantizeOp
    compress_op = ScaleOp(IdentityOp(data), 0.5)  # Simulate compression
    compressed = weight.compress_with(compress_op)
    
    print(f"\nCompressed weight: {compressed}")
    print(f"Compressed mean: {compressed.data.mean():.4f}")
    print(f"Operation type: {compressed.get_operation_type()}")


def matrix_operations_example():
    """Example with matrix multiplication."""
    print("\n=== Matrix Operations Example ===")
    
    # Create weight matrices
    w1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    w2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
    
    # Create matrix multiplication graph
    matmul_op = MatMulOp(IdentityOp(w1), IdentityOp(w2))
    graph = ComputationGraph(matmul_op)
    
    weight = WeightTensor(computation_graph=graph)
    
    print(f"MatMul weight tensor: {weight}")
    print(f"W1:\n{w1}")
    print(f"W2:\n{w2}")
    print(f"Result:\n{weight.data}")


def reshape_example():
    """Example with reshape operations."""
    print("\n=== Reshape Operation Example ===")
    
    # Create a 1D weight vector
    data = np.arange(12, dtype=np.float32)
    
    # Reshape to 3x4 matrix
    reshape_op = ReshapeOp(IdentityOp(data), (3, 4))
    graph = ComputationGraph(reshape_op)
    
    weight = WeightTensor(computation_graph=graph)
    
    print(f"Reshaped weight: {weight}")
    print(f"Original shape: {data.shape}")
    print(f"New shape: {weight.shape}")
    print(f"Data:\n{weight.data}")


def backward_compatibility_example():
    """Example showing backward compatibility."""
    print("\n=== Backward Compatibility Example ===")
    
    # Old-style creation still works
    data = np.random.randn(5, 5).astype(np.float32)
    weight = WeightTensor(data=data)
    
    print(f"Old-style weight: {weight}")
    print(f"Has graph: {weight._graph is not None}")
    print(f"Operation type: {weight.get_operation_type()}")
    
    # All existing functionality works
    hash_val = weight.compute_hash()
    print(f"Hash: {hash_val[:16]}...")
    
    # Serialization works
    weight_dict = weight.to_dict()
    print(f"Serialized keys: {list(weight_dict.keys())}")


def main():
    """Run all examples."""
    basic_graph_example()
    complex_graph_example()
    compression_example()
    matrix_operations_example()
    reshape_example()
    backward_compatibility_example()
    
    print("\n=== Summary ===")
    print("WeightTensor now supports:")
    print("- Lazy evaluation via computation graphs")
    print("- Memory-efficient weight representation")
    print("- Complex operation composition")
    print("- Full backward compatibility")
    print("- Graph-based hashing and serialization")


if __name__ == "__main__":
    main()