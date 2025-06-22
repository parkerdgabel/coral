"""Base class for representing neural network weights"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING
import warnings

import numpy as np
import xxhash

# Import computation graph components when available
# Using TYPE_CHECKING to avoid circular imports and handle when components aren't ready
if TYPE_CHECKING:
    from coral.core.weight_ops import ComputationGraph, WeightOp


@dataclass
class WeightMetadata:
    """Metadata associated with a weight tensor"""

    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    layer_type: Optional[str] = None
    model_name: Optional[str] = None
    compression_info: Dict[str, Any] = field(default_factory=dict)
    hash: Optional[str] = None


class WeightTensor:
    """
    Base class for representing neural network weights with deduplication support.

    This class provides a unified interface for weight tensors with support for:
    - Content-based hashing for deduplication
    - Metadata tracking
    - Lazy loading from storage
    - Compression support
    - Computation graph-based representation (NEW)
    """

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        metadata: Optional[WeightMetadata] = None,
        store_ref: Optional[str] = None,
        computation_graph: Optional["ComputationGraph"] = None,
    ):
        """
        Initialize a WeightTensor.

        Args:
            data: The actual weight data as numpy array
            metadata: Metadata about the weight tensor
            store_ref: Reference to data in storage (for lazy loading)
            computation_graph: Computation graph for lazy evaluation (NEW)
        """
        self._data = data
        self._metadata = metadata
        self._store_ref = store_ref
        self._hash: Optional[str] = None
        self._graph = computation_graph
        self._materialized = False  # Track if graph has been evaluated

        # Handle computation graph initialization
        if computation_graph is not None:
            # Graph mode - data will be lazily evaluated
            self._data = None
            self._materialized = False
            
            # Try to infer metadata from graph if not provided
            if metadata is None:
                try:
                    # Get shape and dtype from graph without full evaluation
                    shape = computation_graph.get_output_shape()
                    dtype = computation_graph.get_output_dtype()
                    self._metadata = WeightMetadata(
                        name="graph_based", shape=shape, dtype=dtype
                    )
                except (AttributeError, NotImplementedError):
                    # Graph doesn't support metadata inference yet
                    pass
        else:
            # Legacy mode - wrap raw data in graph for uniformity
            if data is not None:
                try:
                    # Try to create IdentityOp if available
                    from coral.core.weight_ops import IdentityOp, ComputationGraph
                    self._graph = ComputationGraph(IdentityOp(data))
                except ImportError:
                    # Weight ops not available yet, continue in legacy mode
                    self._graph = None
                
                if metadata is None:
                    # Auto-create metadata from data
                    self._metadata = WeightMetadata(
                        name="unnamed", shape=data.shape, dtype=data.dtype
                    )

    @property
    def data(self) -> np.ndarray:
        """Get the weight data, evaluating computation graph if necessary"""
        if self._data is None:
            # Try to evaluate from computation graph
            if self._graph is not None:
                try:
                    self._data = self._graph.evaluate()
                    self._materialized = True
                except Exception as e:
                    raise ValueError(f"Failed to evaluate computation graph: {e}")
            else:
                raise ValueError("Weight data not loaded and no computation graph available")
        return self._data

    @data.setter
    def data(self, value: np.ndarray) -> None:
        """Set the weight data and invalidate hash"""
        self._data = value
        self._hash = None  # Invalidate cached hash when data changes
        self._materialized = True
        
        # Clear graph since we're setting raw data
        self._graph = None
        
        # Update metadata shape if it exists
        if self._metadata is not None:
            self._metadata.shape = value.shape
            self._metadata.dtype = value.dtype

    @property
    def metadata(self) -> WeightMetadata:
        """Get the weight metadata"""
        if self._metadata is None:
            raise ValueError("No metadata available for this weight tensor")
        return self._metadata

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the weight tensor"""
        return self.metadata.shape

    @property
    def dtype(self) -> np.dtype:
        """Get the data type of the weight tensor"""
        return self.metadata.dtype

    @property
    def size(self) -> int:
        """Get the total number of elements"""
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        """Get the number of bytes used by the tensor"""
        # If we have a computation graph, use its memory calculation
        if self._graph is not None and not self._materialized:
            try:
                return self._graph.get_total_memory()
            except (AttributeError, NotImplementedError):
                # Fall back to regular calculation
                pass
        
        # Use the data's nbytes directly if available
        if self._data is not None:
            return self._data.nbytes
        else:
            # Calculate from metadata
            return self.size * np.dtype(self.dtype).itemsize

    def compute_hash(self, force: bool = False) -> str:
        """
        Compute content-based hash of the weight tensor.

        For computation graphs, hashes the graph structure rather than 
        evaluating the full data (unless already materialized).

        Args:
            force: If True, recompute hash even if cached

        Returns:
            Hexadecimal hash string
        """
        if self._hash is not None and not force:
            return self._hash

        # Use xxhash for fast hashing
        hasher = xxhash.xxh3_64()

        # Include shape and dtype in hash to distinguish identical data
        # with different interpretations
        # Normalize shape to ensure consistent hashing regardless of int types
        normalized_shape = tuple(int(dim) for dim in self.shape)
        # Normalize dtype to ensure consistent hashing regardless of representation
        normalized_dtype = np.dtype(self.dtype).name
        hasher.update(str(normalized_shape).encode())
        hasher.update(normalized_dtype.encode())

        # Hash based on graph structure if available and not materialized
        if self._graph is not None and not self._materialized:
            try:
                # Hash the graph structure instead of evaluating
                graph_hash = self._graph.compute_hash()
                hasher.update(graph_hash.encode())
            except (AttributeError, NotImplementedError):
                # Fall back to data hashing
                hasher.update(self.data.tobytes())
        else:
            # Hash the actual data
            hasher.update(self.data.tobytes())

        self._hash = hasher.hexdigest()
        if self._metadata:
            self._metadata.hash = self._hash

        return self._hash

    def is_similar_to(self, other: "WeightTensor", threshold: float = 0.99) -> bool:
        """
        Check if this weight tensor is similar to another.

        Uses cosine similarity for comparison with numerical stability safeguards.

        Args:
            other: Another WeightTensor to compare with
            threshold: Similarity threshold (0-1)

        Returns:
            True if similarity exceeds threshold
        """
        if self.shape != other.shape or self.dtype != other.dtype:
            return False

        # Flatten arrays for comparison
        a = self.data.flatten()
        b = other.data.flatten()

        # Check for special values (NaN, Inf)
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            # NaN values - consider not similar
            return False
        if np.any(np.isinf(a)) or np.any(np.isinf(b)):
            # Inf values - compare element-wise for exact match
            return bool(np.array_equal(a, b))

        # Check for extreme values that might cause overflow
        max_val = max(np.max(np.abs(a)), np.max(np.abs(b)))
        if max_val > 1e20:  # Use float64 for large values
            a = a.astype(np.float64)
            b = b.astype(np.float64)

        # Compute norms safely
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        # Handle zero vectors
        if norm_a == 0 or norm_b == 0:
            # Both zero vectors are considered similar
            return bool(norm_a == norm_b)

        # Add epsilon to prevent division issues
        epsilon = np.finfo(a.dtype).eps * max(1.0, norm_a, norm_b)

        # Compute cosine similarity with numerical stability
        try:
            # Use einsum for more stable dot product computation
            dot_product = np.sum(a * b)
            
            # Normalize and clip to valid range
            similarity = dot_product / (norm_a * norm_b + epsilon)
            similarity = np.clip(similarity, -1.0, 1.0)
            
            return bool(similarity >= threshold)
        except (RuntimeWarning, FloatingPointError):
            # Fall back to element-wise comparison for edge cases
            return bool(np.allclose(a, b, rtol=1-threshold, atol=1e-8))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            "metadata": {
                "name": self.metadata.name,
                "shape": list(self.metadata.shape),
                "dtype": np.dtype(
                    self.metadata.dtype
                ).name,  # Use .name for proper serialization
                "layer_type": self.metadata.layer_type,
                "model_name": self.metadata.model_name,
                "compression_info": self.metadata.compression_info,
                "hash": self.compute_hash(),
            },
            "store_ref": self._store_ref,
            "has_data": self._data is not None,
        }
        
        # Add graph serialization if available
        if self._graph is not None:
            try:
                result["computation_graph"] = self._graph.serialize()
                result["materialized"] = self._materialized
            except (AttributeError, NotImplementedError):
                # Graph serialization not implemented yet
                pass
                
        return result

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], weight_data: Optional[np.ndarray] = None
    ) -> "WeightTensor":
        """Create WeightTensor from dictionary"""
        metadata = WeightMetadata(
            name=data["metadata"]["name"],
            shape=tuple(data["metadata"]["shape"]),
            dtype=np.dtype(data["metadata"]["dtype"]),
            layer_type=data["metadata"].get("layer_type"),
            model_name=data["metadata"].get("model_name"),
            compression_info=data["metadata"].get("compression_info", {}),
            hash=data["metadata"].get("hash"),
        )

        # Check if computation graph data is available
        computation_graph = None
        if "computation_graph" in data:
            try:
                from coral.core.weight_ops import ComputationGraph
                computation_graph = ComputationGraph.deserialize(data["computation_graph"])
            except (ImportError, AttributeError, NotImplementedError):
                # Graph deserialization not available yet
                pass

        tensor = cls(
            data=weight_data, 
            metadata=metadata, 
            store_ref=data.get("store_ref"),
            computation_graph=computation_graph
        )
        
        # Restore materialization state
        if computation_graph is not None and data.get("materialized", False):
            tensor._materialized = True
            
        return tensor

    def get_computation_graph(self) -> Optional["ComputationGraph"]:
        """Return the underlying computation graph if available"""
        return self._graph

    def materialize(self) -> np.ndarray:
        """Force evaluation of computation graph and cache result"""
        if not self._materialized and self._graph is not None:
            self._data = self._graph.evaluate()
            self._materialized = True
        return self.data

    def get_operation_type(self) -> Optional[str]:
        """Return the type of the root operation in the computation graph"""
        if self._graph is not None:
            try:
                # Try to get op_type from the root operation
                if hasattr(self._graph.root, 'op_type'):
                    op_type = self._graph.root.op_type
                    # Handle enum
                    if hasattr(op_type, 'name'):
                        return op_type.name
                    return str(op_type)
                # Fall back to class name
                return self._graph.root.__class__.__name__
            except (AttributeError, NotImplementedError):
                return "unknown"
        return None

    def compress_with(self, operation: "WeightOp") -> "WeightTensor":
        """
        Create a new WeightTensor with a compression operation applied.
        
        Args:
            operation: A WeightOp that compresses this tensor
            
        Returns:
            New WeightTensor with the compression operation
        """
        try:
            from coral.core.weight_ops import ComputationGraph
            
            # Create new graph with compression operation
            new_graph = ComputationGraph(operation)
            
            # Create new tensor with compressed representation
            return WeightTensor(
                computation_graph=new_graph,
                metadata=self._metadata  # Preserve metadata
            )
        except ImportError:
            warnings.warn(
                "Computation graph support not available. "
                "Cannot apply compression operation.",
                RuntimeWarning
            )
            return self

    def __repr__(self) -> str:
        base = (
            f"WeightTensor(name='{self.metadata.name}', "
            f"shape={self.shape}, dtype={self.dtype}, "
            f"size={self.size}, nbytes={self.nbytes}"
        )
        if self._graph is not None:
            op_type = self.get_operation_type()
            if op_type:
                base += f", op_type='{op_type}'"
        base += ")"
        return base
