"""Computation graph implementation for weight operations.

This module provides the ComputationGraph class that manages a directed
acyclic graph (DAG) of weight operations with lazy evaluation and caching.
"""

import weakref
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from coral.core.weight_ops.base import WeightOp, calculate_array_memory


class ComputationGraph:
    """Represents a DAG of weight operations with lazy evaluation.
    
    The computation graph manages a directed acyclic graph of weight
    operations, providing:
    
    - Lazy evaluation with caching/memoization
    - Memory usage tracking
    - Graph validation
    - Optimization passes (future)
    
    The graph uses weak references for caching to allow garbage collection
    of computed results when memory is needed.
    
    Attributes:
        root: The root operation of the graph
        _cache: Weak reference cache for memoization
        _validation_enabled: Whether to validate graph structure
    """
    
    def __init__(self, root_op: WeightOp, validate: bool = True):
        """Initialize computation graph.
        
        Args:
            root_op: The root operation of the graph
            validate: Whether to validate the graph structure
            
        Raises:
            TypeError: If root_op is not a WeightOp
            ValueError: If graph contains cycles (when validate=True)
        """
        if not isinstance(root_op, WeightOp):
            raise TypeError(f"Root must be a WeightOp, got {type(root_op)}")
        
        self.root = root_op
        self._cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._validation_enabled = validate
        
        if validate:
            self._validate_graph()
    
    def evaluate(self) -> np.ndarray:
        """Lazily evaluate the graph and return weight tensor.
        
        Uses memoization to avoid recomputing intermediate results.
        
        Returns:
            The computed weight tensor
        """
        return self._evaluate_op(self.root)
    
    def _evaluate_op(self, op: WeightOp) -> np.ndarray:
        """Recursively evaluate an operation with caching.
        
        Args:
            op: The operation to evaluate
            
        Returns:
            The computed tensor for this operation
        """
        # Check cache first
        op_id = id(op)
        if op_id in self._cache:
            # Return cached result
            return self._cache[op_id].copy()
        
        # Compute the result
        result = op.forward()
        
        # Cache the result (using weak reference via the dict)
        # We store a wrapper object that holds the array
        self._cache[op_id] = _CachedArray(result)
        
        return result.copy()
    
    def get_total_memory(self) -> int:
        """Calculate total memory usage of graph.
        
        This includes:
        - Memory for operation parameters
        - Memory for cached results (if any)
        - Does not include the final output tensor
        
        Returns:
            Total memory usage in bytes
        """
        # Collect all operations in the graph
        all_ops = self._collect_all_operations()
        
        total_memory = 0
        
        # Add memory for operation parameters
        for op in all_ops:
            total_memory += op.get_memory_usage()
        
        # Add memory for cached results
        for op_id, cached_array in self._cache.items():
            if cached_array is not None:
                total_memory += cached_array.nbytes
        
        return total_memory
    
    def get_graph_info(self) -> Dict[str, Any]:
        """Get information about the graph structure.
        
        Returns:
            Dictionary containing:
            - num_operations: Total number of operations
            - operation_counts: Count of each operation type
            - max_depth: Maximum depth of the graph
            - total_parameters: Total memory for operation parameters
            - output_shape: Shape of the final output
            - output_dtype: Data type of the final output
        """
        all_ops = self._collect_all_operations()
        
        # Count operations by type
        op_counts = {}
        total_params = 0
        
        for op in all_ops:
            op_type_name = op.op_type.name
            op_counts[op_type_name] = op_counts.get(op_type_name, 0) + 1
            total_params += op.get_memory_usage()
        
        return {
            "num_operations": len(all_ops),
            "operation_counts": op_counts,
            "max_depth": self._compute_max_depth(),
            "total_parameters": total_params,
            "output_shape": self.root.get_output_shape(),
            "output_dtype": str(self.root.get_output_dtype()),
        }
    
    def clear_cache(self) -> None:
        """Clear the memoization cache.
        
        This forces recomputation on the next evaluation.
        """
        self._cache.clear()
    
    def _validate_graph(self) -> None:
        """Validate that the graph is acyclic.
        
        Raises:
            ValueError: If the graph contains cycles
        """
        # Use DFS with a visiting set to detect cycles
        visited = set()
        visiting = set()
        
        def has_cycle(op: WeightOp) -> bool:
            if id(op) in visiting:
                return True
            if id(op) in visited:
                return False
            
            visiting.add(id(op))
            
            for input_op in op.inputs:
                if has_cycle(input_op):
                    return True
            
            visiting.remove(id(op))
            visited.add(id(op))
            return False
        
        if has_cycle(self.root):
            raise ValueError("Computation graph contains cycles")
    
    def _collect_all_operations(self) -> Set[WeightOp]:
        """Collect all operations in the graph.
        
        Returns:
            Set of all operations reachable from root
        """
        all_ops = set()
        to_visit = deque([self.root])
        
        while to_visit:
            op = to_visit.popleft()
            if op not in all_ops:
                all_ops.add(op)
                to_visit.extend(op.inputs)
        
        return all_ops
    
    def _compute_max_depth(self) -> int:
        """Compute the maximum depth of the graph.
        
        Returns:
            Maximum depth from root to any leaf
        """
        depth_cache = {}
        
        def compute_depth(op: WeightOp) -> int:
            op_id = id(op)
            if op_id in depth_cache:
                return depth_cache[op_id]
            
            if not op.inputs:
                depth = 1
            else:
                depth = 1 + max(compute_depth(inp) for inp in op.inputs)
            
            depth_cache[op_id] = depth
            return depth
        
        return compute_depth(self.root)
    
    def optimize(self) -> 'ComputationGraph':
        """Apply graph optimization passes.
        
        Currently returns self (no optimizations implemented yet).
        Future optimizations might include:
        - Constant folding
        - Operation fusion
        - Common subexpression elimination
        
        Returns:
            Optimized computation graph
        """
        # TODO: Implement optimization passes
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation.
        
        This creates a serializable representation of the graph
        structure that can be used for storage or visualization.
        
        Returns:
            Dictionary representation of the graph
        """
        # Assign unique IDs to each operation
        op_to_id = {}
        id_counter = 0
        
        def assign_ids(op: WeightOp) -> None:
            nonlocal id_counter
            if op not in op_to_id:
                op_to_id[op] = f"op_{id_counter}"
                id_counter += 1
                for inp in op.inputs:
                    assign_ids(inp)
        
        assign_ids(self.root)
        
        # Build graph representation
        nodes = []
        edges = []
        
        for op, op_id in op_to_id.items():
            # Add node
            nodes.append({
                "id": op_id,
                "type": op.op_type.name,
                "shape": list(op.get_output_shape()),
                "dtype": str(op.get_output_dtype()),
            })
            
            # Add edges
            for inp in op.inputs:
                edges.append({
                    "from": op_to_id[inp],
                    "to": op_id,
                })
        
        return {
            "root": op_to_id[self.root],
            "nodes": nodes,
            "edges": edges,
        }
    
    def __repr__(self) -> str:
        """Return string representation."""
        info = self.get_graph_info()
        return (
            f"ComputationGraph("
            f"operations={info['num_operations']}, "
            f"depth={info['max_depth']}, "
            f"output_shape={info['output_shape']})"
        )


class _CachedArray:
    """Wrapper for cached numpy arrays.
    
    This allows us to store arrays in the weak reference dictionary
    and track their memory usage.
    """
    
    def __init__(self, array: np.ndarray):
        """Initialize cached array wrapper.
        
        Args:
            array: The numpy array to cache
        """
        self.array = array
        self.nbytes = array.nbytes
    
    def copy(self) -> np.ndarray:
        """Return a copy of the cached array.
        
        Returns:
            Copy of the array
        """
        return self.array.copy()