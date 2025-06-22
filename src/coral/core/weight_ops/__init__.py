"""Weight operations for computation graph-based weight representation.

This module provides the core infrastructure for representing neural network
weights as computation graphs rather than static arrays. This enables:

- Lazy evaluation of weight tensors
- Advanced compression techniques
- Memory-efficient storage
- Composable weight transformations

The module is organized as:
- base.py: Abstract base classes and utilities
- basic_ops.py: Fundamental operations (identity, add, matmul, etc.)
- compression_ops.py: Compression operations (SVD, sparse, quantize, PQ)
- graph.py: Computation graph implementation with caching
"""

from coral.core.weight_ops.base import (
    WeightOp, OpType, register_operation, deserialize_op, OPERATION_REGISTRY
)

# Import operations from the basic_ops module
from coral.core.weight_ops.basic_ops import (
    IdentityOp,
    AddOp,
    MatMulOp,
    ScaleOp,
    ReshapeOp,
)

# Import compression operations
from coral.core.weight_ops.compression_ops import (
    SVDOp,
    SparseOp,
    QuantizeOp,
    PQOp,
)

# Import graph
from coral.core.weight_ops.graph import ComputationGraph

# Import utilities
from coral.core.weight_ops.compression_utils import (
    select_rank_by_energy,
    analyze_sparsity,
    select_sparse_format,
    calculate_quantization_params,
    quantize_array,
    dequantize_array,
    recommend_compression_method,
)

__all__ = [
    # Base classes
    "WeightOp",
    "OpType",
    # Registry functions
    "register_operation",
    "deserialize_op",
    "OPERATION_REGISTRY",
    # Basic operations
    "IdentityOp",
    "AddOp",
    "MatMulOp",
    "ScaleOp",
    "ReshapeOp",
    # Compression operations
    "SVDOp",
    "SparseOp",
    "QuantizeOp",
    "PQOp",
    # Graph
    "ComputationGraph",
    # Utilities
    "select_rank_by_energy",
    "analyze_sparsity",
    "select_sparse_format",
    "calculate_quantization_params",
    "quantize_array",
    "dequantize_array",
    "recommend_compression_method",
]

# Register all operations with the registry
register_operation("IDENTITY", IdentityOp)
register_operation("ADD", AddOp)
register_operation("MATMUL", MatMulOp)
register_operation("SCALE", ScaleOp)
register_operation("RESHAPE", ReshapeOp)
register_operation("SVD", SVDOp)
register_operation("SPARSE", SparseOp)
register_operation("QUANTIZE", QuantizeOp)
register_operation("PQ", PQOp)