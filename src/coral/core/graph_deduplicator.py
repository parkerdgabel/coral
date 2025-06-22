"""Graph-based deduplication engine for weight computation graphs."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import json
import xxhash

import numpy as np

from coral.core.deduplicator import DeduplicationStats, Deduplicator
from coral.core.weight_tensor import WeightTensor

logger = logging.getLogger(__name__)


@dataclass
class GraphOpStats:
    """Statistics for operation-level deduplication."""
    
    total_ops: int = 0
    unique_ops: int = 0
    shared_ops: int = 0
    shared_subgraphs: int = 0
    ops_by_type: Dict[str, int] = field(default_factory=dict)
    memory_saved: int = 0
    computation_saved: int = 0  # Estimated FLOPs saved


@dataclass 
class SharedOperation:
    """Represents a shared operation or subgraph."""
    
    op_hash: str
    op_type: str
    reference_count: int = 1
    size_bytes: int = 0
    computation_cost: int = 0  # Estimated FLOPs
    parents: Set[str] = field(default_factory=set)  # Graphs using this op


class GraphDeduplicator(Deduplicator):
    """
    Deduplication engine for weight computation graphs.
    
    Extends base deduplicator to work at the operation level,
    detecting and sharing common operations and subgraphs.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.99,
        enable_delta_encoding: bool = True,
        graph_mode: bool = True,
        structural_similarity_threshold: float = 0.95,
    ):
        """
        Initialize graph deduplicator.
        
        Args:
            similarity_threshold: Threshold for weight similarity
            enable_delta_encoding: Enable delta encoding for similar weights
            graph_mode: Enable graph-based deduplication
            structural_similarity_threshold: Threshold for graph structure similarity
        """
        super().__init__(
            similarity_threshold=similarity_threshold,
            enable_delta_encoding=enable_delta_encoding
        )
        
        self.graph_mode = graph_mode
        self.structural_similarity_threshold = structural_similarity_threshold
        
        # Graph-specific indexes
        self.op_index: Dict[str, Any] = {}  # op_hash -> operation
        self.shared_ops: Dict[str, SharedOperation] = {}  # op_hash -> shared op info
        self.graph_index: Dict[str, Any] = {}  # graph_hash -> computation graph
        self.graph_to_ops: Dict[str, Set[str]] = {}  # graph_hash -> set of op hashes
        self.similar_graphs: Dict[str, List[Tuple[str, float]]] = {}  # graph -> similar graphs
        
        self.graph_stats = GraphOpStats()
    
    def add_computation_graph(
        self, 
        graph: Any,  # ComputationGraph type
        name: str,
        weight_tensor: Optional[WeightTensor] = None
    ) -> str:
        """
        Add a computation graph for deduplication.
        
        Args:
            graph: ComputationGraph to analyze
            name: Name for the graph
            weight_tensor: Optional WeightTensor if available
            
        Returns:
            Hash of the graph (or reference graph if similar)
        """
        if not self.graph_mode:
            # Fall back to regular weight deduplication
            if weight_tensor:
                return self.add_weight(weight_tensor, name)
            else:
                # Evaluate graph to get weight tensor
                data = graph.evaluate()
                weight = WeightTensor(data=data, metadata={"name": name})
                return self.add_weight(weight, name)
        
        # Compute graph hash
        graph_hash = self._compute_graph_hash(graph)
        
        # Check for exact duplicate graph
        if graph_hash in self.graph_index:
            logger.debug(f"Found exact duplicate graph: {name} -> {graph_hash}")
            self._increment_graph_reference(graph_hash)
            return graph_hash
        
        # Analyze graph operations
        ops_in_graph = self._extract_operations(graph)
        op_hashes = set()
        
        # Process each operation
        for op in ops_in_graph:
            op_hash = self._compute_op_hash(op)
            op_hashes.add(op_hash)
            
            if op_hash in self.shared_ops:
                # Increment reference count for shared op
                self.shared_ops[op_hash].reference_count += 1
                self.shared_ops[op_hash].parents.add(graph_hash)
                self.graph_stats.shared_ops += 1
            else:
                # New unique operation
                self.op_index[op_hash] = op
                self.shared_ops[op_hash] = SharedOperation(
                    op_hash=op_hash,
                    op_type=self._get_op_type(op),
                    size_bytes=self._estimate_op_memory(op),
                    computation_cost=self._estimate_op_flops(op),
                    parents={graph_hash}
                )
                self.graph_stats.unique_ops += 1
                
                # Track operations by type
                op_type = self._get_op_type(op)
                self.graph_stats.ops_by_type[op_type] = \
                    self.graph_stats.ops_by_type.get(op_type, 0) + 1
        
        # Check for similar graphs
        similar_graph = self._find_similar_graph(graph, op_hashes)
        if similar_graph:
            # Store as similar to existing graph
            self._add_similar_graph(graph_hash, similar_graph, graph)
        else:
            # Store as new unique graph
            self.graph_index[graph_hash] = graph
            self.graph_to_ops[graph_hash] = op_hashes
        
        self.graph_stats.total_ops += len(ops_in_graph)
        
        # If weight tensor provided, also do weight-level deduplication
        if weight_tensor:
            self.add_weight(weight_tensor, name)
        
        return graph_hash
    
    def _compute_graph_hash(self, graph: Any) -> str:
        """Compute hash for entire computation graph."""
        hasher = xxhash.xxh3_64()
        
        # Hash graph structure and operations
        graph_data = self._serialize_graph_structure(graph)
        hasher.update(json.dumps(graph_data, sort_keys=True).encode())
        
        return hasher.hexdigest()
    
    def _compute_op_hash(self, op: Any) -> str:
        """Compute hash for a single operation."""
        hasher = xxhash.xxh3_64()
        
        # Hash operation type and parameters
        op_data = {
            "type": self._get_op_type(op),
            "params": self._get_op_params(op)
        }
        hasher.update(json.dumps(op_data, sort_keys=True).encode())
        
        return hasher.hexdigest()
    
    def _extract_operations(self, graph: Any) -> List[Any]:
        """Extract all operations from a computation graph."""
        operations = []
        visited = set()
        
        def traverse(op):
            if op is None or id(op) in visited:
                return
            visited.add(id(op))
            operations.append(op)
            
            # Traverse inputs (assuming ops have inputs attribute)
            if hasattr(op, 'inputs'):
                for input_op in op.inputs:
                    if input_op is not None:
                        traverse(input_op)
        
        # Start from root operation
        if hasattr(graph, 'root'):
            traverse(graph.root)
        
        return operations
    
    def _find_similar_graph(
        self, 
        graph: Any, 
        op_hashes: Set[str]
    ) -> Optional[str]:
        """Find similar existing graph based on structure and operations."""
        best_similarity = self.structural_similarity_threshold
        best_match = None
        
        for existing_hash, existing_ops in self.graph_to_ops.items():
            # Compute Jaccard similarity of operations
            intersection = len(op_hashes & existing_ops)
            union = len(op_hashes | existing_ops)
            
            if union > 0:
                similarity = intersection / union
                
                if similarity > best_similarity:
                    # Also check structural similarity
                    existing_graph = self.graph_index[existing_hash]
                    struct_sim = self._compute_structural_similarity(
                        graph, existing_graph
                    )
                    
                    combined_similarity = (similarity + struct_sim) / 2
                    
                    if combined_similarity > best_similarity:
                        best_similarity = combined_similarity
                        best_match = existing_hash
        
        return best_match
    
    def _compute_structural_similarity(self, graph1: Any, graph2: Any) -> float:
        """Compute structural similarity between two graphs."""
        # Extract graph topology
        topo1 = self._get_graph_topology(graph1)
        topo2 = self._get_graph_topology(graph2)
        
        # Compare node counts
        if len(topo1["nodes"]) != len(topo2["nodes"]):
            node_sim = min(len(topo1["nodes"]), len(topo2["nodes"])) / \
                      max(len(topo1["nodes"]), len(topo2["nodes"]))
        else:
            node_sim = 1.0
        
        # Compare edge counts
        if len(topo1["edges"]) != len(topo2["edges"]):
            edge_sim = min(len(topo1["edges"]), len(topo2["edges"])) / \
                      max(len(topo1["edges"]), len(topo2["edges"]))
        else:
            edge_sim = 1.0
        
        # Compare operation type distribution
        type_sim = self._compare_op_type_distribution(
            topo1["op_types"], topo2["op_types"]
        )
        
        # Weighted average
        return 0.3 * node_sim + 0.3 * edge_sim + 0.4 * type_sim
    
    def _get_graph_topology(self, graph: Any) -> Dict[str, Any]:
        """Extract topology information from graph."""
        nodes = []
        edges = []
        op_types = {}
        
        visited = set()
        
        def traverse(op, node_id):
            if id(op) in visited:
                return
            visited.add(id(op))
            
            op_type = self._get_op_type(op)
            nodes.append({"id": node_id, "type": op_type})
            op_types[op_type] = op_types.get(op_type, 0) + 1
            
            if hasattr(op, 'inputs'):
                for i, input_op in enumerate(op.inputs):
                    if input_op is not None:
                        child_id = f"{node_id}_input_{i}"
                        edges.append({"from": child_id, "to": node_id})
                        traverse(input_op, child_id)
        
        if hasattr(graph, 'root'):
            traverse(graph.root, "root")
        
        return {
            "nodes": nodes,
            "edges": edges,
            "op_types": op_types
        }
    
    def _compare_op_type_distribution(
        self, 
        dist1: Dict[str, int], 
        dist2: Dict[str, int]
    ) -> float:
        """Compare operation type distributions using cosine similarity."""
        all_types = set(dist1.keys()) | set(dist2.keys())
        
        if not all_types:
            return 1.0
        
        # Create vectors
        vec1 = np.array([dist1.get(t, 0) for t in all_types])
        vec2 = np.array([dist2.get(t, 0) for t in all_types])
        
        # Compute cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0 if norm1 == norm2 else 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def _add_similar_graph(
        self, 
        new_hash: str, 
        similar_hash: str, 
        graph: Any
    ):
        """Add a graph as similar to existing graph."""
        if similar_hash not in self.similar_graphs:
            self.similar_graphs[similar_hash] = []
        
        # Compute detailed similarity
        similarity = self._compute_detailed_similarity(
            graph, self.graph_index[similar_hash]
        )
        
        self.similar_graphs[similar_hash].append((new_hash, similarity))
        
        # Still store the graph for potential delta encoding
        self.graph_index[new_hash] = graph
        self.graph_to_ops[new_hash] = self._extract_op_hashes(graph)
        
        logger.debug(
            f"Found similar graph: {new_hash} -> {similar_hash} "
            f"(similarity: {similarity:.4f})"
        )
    
    def _compute_detailed_similarity(self, graph1: Any, graph2: Any) -> float:
        """Compute detailed similarity score between graphs."""
        # Structural similarity
        struct_sim = self._compute_structural_similarity(graph1, graph2)
        
        # Operation similarity
        ops1 = set(self._extract_op_hashes(graph1))
        ops2 = set(self._extract_op_hashes(graph2))
        op_sim = len(ops1 & ops2) / len(ops1 | ops2) if ops1 | ops2 else 1.0
        
        # Parameter similarity (for same-type operations)
        param_sim = self._compute_parameter_similarity(graph1, graph2)
        
        # Weighted combination
        return 0.4 * struct_sim + 0.4 * op_sim + 0.2 * param_sim
    
    def _compute_parameter_similarity(self, graph1: Any, graph2: Any) -> float:
        """Compare parameters of operations in two graphs."""
        ops1 = self._extract_operations(graph1)
        ops2 = self._extract_operations(graph2)
        
        # Group by operation type
        ops1_by_type = {}
        for op in ops1:
            op_type = self._get_op_type(op)
            if op_type not in ops1_by_type:
                ops1_by_type[op_type] = []
            ops1_by_type[op_type].append(op)
        
        ops2_by_type = {}
        for op in ops2:
            op_type = self._get_op_type(op)
            if op_type not in ops2_by_type:
                ops2_by_type[op_type] = []
            ops2_by_type[op_type].append(op)
        
        # Compare parameters for matching operation types
        similarities = []
        for op_type in set(ops1_by_type.keys()) & set(ops2_by_type.keys()):
            # For each pair of same-type operations
            for op1 in ops1_by_type[op_type]:
                for op2 in ops2_by_type[op_type]:
                    param_sim = self._compare_op_parameters(op1, op2)
                    similarities.append(param_sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _compare_op_parameters(self, op1: Any, op2: Any) -> float:
        """Compare parameters of two operations of the same type."""
        params1 = self._get_op_params(op1)
        params2 = self._get_op_params(op2)
        
        if not params1 and not params2:
            return 1.0
        
        # Compare parameter dictionaries
        all_keys = set(params1.keys()) | set(params2.keys())
        if not all_keys:
            return 1.0
        
        matches = 0
        for key in all_keys:
            if key in params1 and key in params2:
                # Compare parameter values
                if self._params_equal(params1[key], params2[key]):
                    matches += 1
                elif self._params_similar(params1[key], params2[key]):
                    matches += 0.5
        
        return matches / len(all_keys)
    
    def _params_equal(self, p1: Any, p2: Any) -> bool:
        """Check if two parameters are exactly equal."""
        if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
            return np.array_equal(p1, p2)
        return p1 == p2
    
    def _params_similar(self, p1: Any, p2: Any, threshold: float = 0.95) -> bool:
        """Check if two parameters are similar."""
        if isinstance(p1, (int, float)) and isinstance(p2, (int, float)):
            if p1 == 0 or p2 == 0:
                return p1 == p2
            ratio = min(p1, p2) / max(p1, p2)
            return ratio > threshold
        return False
    
    def get_graph_deduplication_stats(self) -> Dict[str, Any]:
        """Get detailed graph deduplication statistics."""
        base_stats = self.compute_stats()
        
        # Calculate memory saved by sharing operations
        memory_saved = 0
        computation_saved = 0
        
        for op_hash, shared_op in self.shared_ops.items():
            if shared_op.reference_count > 1:
                # Save memory for each additional reference
                memory_saved += shared_op.size_bytes * (shared_op.reference_count - 1)
                computation_saved += shared_op.computation_cost * (shared_op.reference_count - 1)
        
        self.graph_stats.memory_saved = memory_saved
        self.graph_stats.computation_saved = computation_saved
        
        # Find most shared operations
        most_shared = sorted(
            self.shared_ops.values(),
            key=lambda x: x.reference_count,
            reverse=True
        )[:10]
        
        # Find most common subgraphs
        subgraph_patterns = self._identify_common_subgraphs()
        
        return {
            "weight_stats": base_stats.__dict__,
            "graph_stats": {
                "total_ops": self.graph_stats.total_ops,
                "unique_ops": self.graph_stats.unique_ops,
                "shared_ops": self.graph_stats.shared_ops,
                "shared_subgraphs": len(subgraph_patterns),
                "ops_by_type": self.graph_stats.ops_by_type,
                "memory_saved": self.graph_stats.memory_saved,
                "computation_saved": self.graph_stats.computation_saved,
                "most_shared_ops": [
                    {
                        "op_type": op.op_type,
                        "reference_count": op.reference_count,
                        "memory_saved": op.size_bytes * (op.reference_count - 1),
                    }
                    for op in most_shared
                ],
                "common_subgraphs": subgraph_patterns[:5],
            }
        }
    
    def _identify_common_subgraphs(self) -> List[Dict[str, Any]]:
        """Identify common subgraph patterns across graphs."""
        # This is a simplified version - full implementation would use
        # graph mining algorithms like gSpan or FSG
        
        subgraph_patterns = []
        
        # Look for chains of operations that appear multiple times
        op_chains = {}
        for graph_hash, op_hashes in self.graph_to_ops.items():
            graph = self.graph_index[graph_hash]
            chains = self._extract_op_chains(graph)
            
            for chain in chains:
                chain_key = tuple(chain)
                if chain_key not in op_chains:
                    op_chains[chain_key] = []
                op_chains[chain_key].append(graph_hash)
        
        # Filter chains that appear in multiple graphs
        for chain, graphs in op_chains.items():
            if len(graphs) > 1 and len(chain) > 1:
                subgraph_patterns.append({
                    "pattern": [self._get_op_type_from_hash(h) for h in chain],
                    "occurrences": len(graphs),
                    "graphs": graphs[:5],  # First 5 graphs
                })
        
        return sorted(subgraph_patterns, key=lambda x: x["occurrences"], reverse=True)
    
    def _extract_op_chains(self, graph: Any, max_length: int = 5) -> List[List[str]]:
        """Extract linear chains of operations from graph."""
        chains = []
        visited = set()
        
        def extract_chain(op, current_chain):
            if op is None or id(op) in visited or len(current_chain) >= max_length:
                if len(current_chain) > 1:
                    chains.append(current_chain)
                return
            
            visited.add(id(op))
            op_hash = self._compute_op_hash(op)
            current_chain.append(op_hash)
            
            # Follow single input paths
            if hasattr(op, 'inputs'):
                if len(op.inputs) == 1 and op.inputs[0] is not None:
                    extract_chain(op.inputs[0], current_chain.copy())
                elif len(op.inputs) > 1:
                    # For multiple inputs, we end the chain but record it
                    if len(current_chain) > 1:
                        chains.append(current_chain)
                    # Start new chains from each input
                    for input_op in op.inputs:
                        if input_op is not None:
                            extract_chain(input_op, [])
        
        if hasattr(graph, 'root'):
            extract_chain(graph.root, [])
        
        return chains
    
    def _get_op_type_from_hash(self, op_hash: str) -> str:
        """Get operation type from hash."""
        if op_hash in self.shared_ops:
            return self.shared_ops[op_hash].op_type
        elif op_hash in self.op_index:
            return self._get_op_type(self.op_index[op_hash])
        return "unknown"
    
    def _increment_graph_reference(self, graph_hash: str):
        """Increment reference count for a graph and its operations."""
        if graph_hash in self.graph_to_ops:
            for op_hash in self.graph_to_ops[graph_hash]:
                if op_hash in self.shared_ops:
                    self.shared_ops[op_hash].reference_count += 1
    
    def _extract_op_hashes(self, graph: Any) -> Set[str]:
        """Extract operation hashes from a graph."""
        ops = self._extract_operations(graph)
        return {self._compute_op_hash(op) for op in ops}
    
    def _serialize_graph_structure(self, graph: Any) -> Dict[str, Any]:
        """Serialize graph structure for hashing."""
        # Create a normalized representation that doesn't depend on object IDs
        ops = self._extract_operations(graph)
        
        # Build normalized structure
        structure = {
            "ops": [],
            "connections": []
        }
        
        # Map operations to indices
        op_to_idx = {}
        for i, op in enumerate(ops):
            op_to_idx[id(op)] = i
            structure["ops"].append({
                "type": self._get_op_type(op),
                "params": self._get_op_params(op)
            })
        
        # Build connections
        for i, op in enumerate(ops):
            if hasattr(op, 'inputs'):
                for j, input_op in enumerate(op.inputs):
                    if input_op is not None and id(input_op) in op_to_idx:
                        structure["connections"].append({
                            "from": op_to_idx[id(input_op)],
                            "to": i,
                            "input_index": j
                        })
        
        return structure
    
    def _get_op_type(self, op: Any) -> str:
        """Get operation type name."""
        return op.__class__.__name__
    
    def _get_op_params(self, op: Any) -> Dict[str, Any]:
        """Extract parameters from an operation."""
        params = {}
        
        # Check if op has params attribute (for mocks)
        if hasattr(op, 'params'):
            return op.params
        
        # Common attributes to check
        param_attrs = [
            'scale', 'bias', 'rank', 'bits', 'shape',
            'axis', 'keepdims', 'dtype', 'compression_level'
        ]
        
        for attr in param_attrs:
            if hasattr(op, attr):
                value = getattr(op, attr)
                if value is not None:
                    params[attr] = value
        
        return params
    
    def _estimate_op_memory(self, op: Any) -> int:
        """Estimate memory usage of an operation."""
        if hasattr(op, 'get_memory_usage'):
            return op.get_memory_usage()
        
        # Default estimates by operation type
        op_type = self._get_op_type(op)
        base_size = 100  # Base overhead
        
        if 'Identity' in op_type:
            return base_size + 8  # Just reference
        elif 'Add' in op_type or 'MatMul' in op_type:
            return base_size + 16  # Two references
        elif 'Scale' in op_type:
            return base_size + 12  # Reference + scalar
        else:
            return base_size + 50  # Conservative estimate
    
    def _estimate_op_flops(self, op: Any) -> int:
        """Estimate computational cost of an operation in FLOPs."""
        if hasattr(op, 'get_computation_cost'):
            return op.get_computation_cost()
        
        # Default estimates
        op_type = self._get_op_type(op)
        
        if 'Identity' in op_type:
            return 0
        elif 'Add' in op_type:
            # Assume some default size
            return 1000
        elif 'MatMul' in op_type:
            # Matrix multiplication is expensive
            return 100000
        elif 'Scale' in op_type:
            return 1000
        else:
            return 5000  # Conservative estimate
    
    def clear(self):
        """Clear all stored data."""
        super().clear()
        self.op_index.clear()
        self.shared_ops.clear()
        self.graph_index.clear()
        self.graph_to_ops.clear()
        self.similar_graphs.clear()
        self.graph_stats = GraphOpStats()