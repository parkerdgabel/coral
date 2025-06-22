"""Computation graph implementation for weight operations."""

import weakref
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .base import WeightOp


class ComputationGraph:
    """Represents a directed acyclic graph (DAG) of weight operations.
    
    The graph enables lazy evaluation of weight tensors, where operations
    are only computed when needed. Results are cached using weak references
    to allow garbage collection when memory is needed.
    
    Attributes:
        root: The root operation of the graph
        _cache: Weak reference cache for computed results
        _evaluated: Whether the graph has been evaluated
    """
    
    def __init__(self, root_op: WeightOp):
        """Initialize computation graph with root operation.
        
        Args:
            root_op: The root operation that produces the final output
            
        Raises:
            TypeError: If root_op is not a WeightOp instance
        """
        if not isinstance(root_op, WeightOp):
            raise TypeError(f"root_op must be a WeightOp, got {type(root_op)}")
        
        self.root = root_op
        self._cache = weakref.WeakValueDictionary()
        self._evaluated = False
        
        # Validate graph structure (check for cycles)
        self._validate_graph()
    
    def evaluate(self) -> np.ndarray:
        """Lazily evaluate the graph and return the weight tensor.
        
        Results are cached to avoid redundant computation.
        
        Returns:
            Computed weight tensor as numpy array
        """
        result = self._evaluate_op(self.root)
        self._evaluated = True
        return result
    
    def _evaluate_op(self, op: WeightOp) -> np.ndarray:
        """Recursively evaluate an operation with caching.
        
        Args:
            op: Operation to evaluate
            
        Returns:
            Result of the operation
        """
        # Check cache first
        op_id = id(op)
        if op_id in self._cache:
            return self._cache[op_id].copy()
        
        # Compute result
        result = op.forward()
        
        # Cache result using weak reference
        # This allows garbage collection if memory is needed
        try:
            self._cache[op_id] = result
        except TypeError:
            # Some arrays might not be weakly referenceable
            pass
        
        return result.copy()
    
    def get_total_memory(self) -> int:
        """Calculate total memory usage of the graph.
        
        This includes memory for all operations in the graph.
        
        Returns:
            Total memory usage in bytes
        """
        visited = set()
        return self._get_memory_recursive(self.root, visited)
    
    def _get_memory_recursive(self, op: WeightOp, visited: Set[int]) -> int:
        """Recursively calculate memory usage avoiding double counting.
        
        Args:
            op: Current operation
            visited: Set of already visited operation IDs
            
        Returns:
            Memory usage in bytes
        """
        op_id = id(op)
        if op_id in visited:
            return 0
        
        visited.add(op_id)
        
        # Get memory for this operation
        memory = op.get_memory_usage()
        
        # Add memory from child operations if any
        # This would require operations to expose their inputs
        # For now, we just return the operation's own memory
        
        return memory
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Get output shape without evaluation."""
        return self.root.get_output_shape()
    
    def get_output_dtype(self) -> np.dtype:
        """Get output dtype without evaluation."""
        return self.root.get_output_dtype()
    
    def optimize(self) -> "ComputationGraph":
        """Apply graph optimization passes.
        
        Future optimizations could include:
        - Constant folding
        - Common subexpression elimination
        - Operation fusion
        
        Returns:
            Optimized computation graph
        """
        # For now, return self (no optimization)
        # This is a placeholder for future optimization passes
        return self
    
    def clear_cache(self):
        """Clear the evaluation cache.
        
        This forces re-computation on next evaluation.
        """
        self._cache.clear()
        self._evaluated = False
    
    def is_evaluated(self) -> bool:
        """Check if the graph has been evaluated."""
        return self._evaluated
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the graph.
        
        Returns:
            Dictionary with graph statistics
        """
        return {
            "root_type": type(self.root).__name__,
            "output_shape": self.get_output_shape(),
            "output_dtype": str(self.get_output_dtype()),
            "memory_usage": self.get_total_memory(),
            "is_evaluated": self._evaluated,
            "cache_size": len(self._cache)
        }
    
    def _validate_graph(self):
        """Validate graph structure (check for cycles).
        
        Raises:
            ValueError: If graph contains cycles
        """
        # For now, we assume operations form a valid DAG
        # In a full implementation, we would traverse the graph
        # and check for cycles using DFS
        pass
    
    def __repr__(self) -> str:
        """String representation of the graph."""
        info = self.get_info()
        return (
            f"ComputationGraph("
            f"root={info['root_type']}, "
            f"shape={info['output_shape']}, "
            f"dtype={info['output_dtype']}, "
            f"evaluated={info['is_evaluated']})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation.
        
        Returns:
            Dictionary suitable for serialization
        """
        return {
            "root": self.root.serialize(),
            "info": self.get_info()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComputationGraph":
        """Create graph from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            ComputationGraph instance
        """
        # This would need an operation registry to deserialize
        raise NotImplementedError("Deserialization requires operation registry")