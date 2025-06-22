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
    
    def __init__(self, root_op: WeightOp, validate: bool = True):
        """Initialize computation graph with root operation.
        
        Args:
            root_op: The root operation that produces the final output
            validate: Whether to validate graph structure (check for cycles)
            
        Raises:
            TypeError: If root_op is not a WeightOp instance
        """
        if not isinstance(root_op, WeightOp):
            raise TypeError(f"root_op must be a WeightOp, got {type(root_op)}")
        
        self.root = root_op
        self._cache = weakref.WeakValueDictionary()
        self._evaluated = False
        
        # Validate graph structure (check for cycles)
        if validate:
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
        """Get basic information about the graph.
        
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
    
    def get_graph_info(self) -> Dict[str, Any]:
        """Get detailed information about the graph structure.
        
        Returns:
            Dictionary with graph statistics including operation counts
        """
        from collections import defaultdict
        
        # Count operations and calculate depth
        visited = set()
        operation_counts = defaultdict(int)
        max_depth = 0
        
        def traverse(op: WeightOp, depth: int = 0) -> None:
            nonlocal max_depth
            if id(op) in visited:
                return
            visited.add(id(op))
            
            # Count this operation
            operation_counts[op.op_type.name] += 1
            max_depth = max(max_depth, depth)
            
            # Traverse inputs
            if hasattr(op, '_inputs') and op._inputs:
                for input_op in op._inputs:
                    traverse(input_op, depth + 1)
            elif hasattr(op, '_input') and op._input:
                traverse(op._input, depth + 1)
            elif hasattr(op, '_left') and op._left:
                traverse(op._left, depth + 1)
                if hasattr(op, '_right') and op._right:
                    traverse(op._right, depth + 1)
        
        traverse(self.root)
        
        return {
            "num_operations": len(visited),
            "operation_counts": dict(operation_counts),
            "output_shape": self.get_output_shape(),
            "output_dtype": str(self.get_output_dtype()),
            "max_depth": max_depth
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
        graph_info = self.get_graph_info()
        return (
            f"ComputationGraph("
            f"operations={graph_info['num_operations']}, "
            f"depth={graph_info['max_depth']}, "
            f"output_shape={graph_info['output_shape']})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation with nodes and edges.
        
        Returns:
            Dictionary containing graph structure
        """
        nodes = []
        edges = []
        visited = set()
        
        def traverse(op: WeightOp) -> str:
            """Traverse graph and collect nodes/edges."""
            op_id = str(id(op))
            if op_id in visited:
                return op_id
            visited.add(op_id)
            
            # Add node
            node_info = {
                "id": op_id,
                "type": op.op_type.name,
                "shape": op.get_output_shape(),
                "dtype": str(op.get_output_dtype())
            }
            nodes.append(node_info)
            
            # Process inputs and add edges
            if hasattr(op, '_inputs') and op._inputs:
                for input_op in op._inputs:
                    input_id = traverse(input_op)
                    edges.append({"from": input_id, "to": op_id})
            elif hasattr(op, '_input') and op._input:
                input_id = traverse(op._input)
                edges.append({"from": input_id, "to": op_id})
            elif hasattr(op, '_left') and op._left:
                left_id = traverse(op._left)
                edges.append({"from": left_id, "to": op_id})
                if hasattr(op, '_right') and op._right:
                    right_id = traverse(op._right)
                    edges.append({"from": right_id, "to": op_id})
            
            return op_id
        
        root_id = traverse(self.root)
        
        return {
            "root": root_id,
            "nodes": nodes,
            "edges": edges
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