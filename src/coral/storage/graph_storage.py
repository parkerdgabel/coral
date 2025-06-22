"""Graph storage serialization for computation graphs in weight operations."""

import json
import logging
from typing import Any, Dict, Optional, List, Tuple
import numpy as np

try:
    from coral.core.weight_ops import WeightOp, ComputationGraph
    from coral.core.weight_ops.basic_ops import (
        IdentityOp, AddOp, MatMulOp, ScaleOp, ReshapeOp
    )
    # These operations are not implemented yet
    SliceOp = None
    ConcatOp = None
    SVDOp = None
    TuckerOp = None
    SparseOp = None
    QuantizeOp = None
    PQOp = None
    DeltaOp = None
    DictionaryOp = None
    WaveletOp = None
    NeuralOp = None
    GeneratorOp = None
except ImportError:
    # Weight ops not available yet
    WeightOp = None
    ComputationGraph = None
    IdentityOp = None
    AddOp = None
    MatMulOp = None
    ScaleOp = None
    ReshapeOp = None
    SliceOp = None
    ConcatOp = None
    SVDOp = None
    TuckerOp = None
    SparseOp = None
    QuantizeOp = None
    PQOp = None
    DeltaOp = None
    DictionaryOp = None
    WaveletOp = None
    NeuralOp = None
    GeneratorOp = None

logger = logging.getLogger(__name__)


class GraphSerializer:
    """Serializer for converting computation graphs to/from storage format.
    
    This class handles the conversion of ComputationGraph objects and their
    WeightOp nodes to a format suitable for HDF5 storage. It maintains
    the DAG structure and all operation parameters.
    """
    
    # Mapping from operation class to type identifier
    OP_TYPE_MAP = {
        IdentityOp: "identity",
        AddOp: "add",
        MatMulOp: "matmul",
        ScaleOp: "scale",
        ReshapeOp: "reshape",
        SliceOp: "slice",
        ConcatOp: "concat",
        SVDOp: "svd",
        TuckerOp: "tucker",
        SparseOp: "sparse",
        QuantizeOp: "quantize",
        PQOp: "pq",
        DeltaOp: "delta",
        DictionaryOp: "dictionary",
        WaveletOp: "wavelet",
        NeuralOp: "neural",
        GeneratorOp: "generator"
    }
    
    # Reverse mapping for deserialization
    TYPE_OP_MAP = {v: k for k, v in OP_TYPE_MAP.items()}
    
    def __init__(self):
        """Initialize the graph serializer."""
        self._node_counter = 0
        self._node_to_id = {}
        self._id_to_node = {}
    
    def serialize_graph(self, graph: ComputationGraph) -> Dict[str, Any]:
        """Serialize a computation graph to storage format.
        
        Args:
            graph: ComputationGraph to serialize
            
        Returns:
            Dictionary containing:
                - nodes: List of serialized nodes
                - edges: List of edge connections (parent_id, child_id)
                - root_id: ID of the root node
                - metadata: Graph-level metadata
        """
        # Reset state for new serialization
        self._node_counter = 0
        self._node_to_id.clear()
        self._id_to_node.clear()
        
        # Traverse graph and assign IDs
        nodes = []
        edges = []
        
        # Depth-first traversal to serialize all nodes
        visited = set()
        stack = [graph.root]
        
        while stack:
            node = stack.pop()
            if id(node) in visited:
                continue
                
            visited.add(id(node))
            node_id = self._get_node_id(node)
            
            # Serialize the node
            serialized_node = self._serialize_node(node)
            nodes.append(serialized_node)
            
            # Process dependencies (child nodes)
            deps = self._get_node_dependencies(node)
            for dep in deps:
                if id(dep) not in visited:
                    stack.append(dep)
                # Record edge
                dep_id = self._get_node_id(dep)
                edges.append((dep_id, node_id))  # dep -> node
        
        # Get root ID
        root_id = self._node_to_id[id(graph.root)]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "root_id": root_id,
            "metadata": {
                "version": "1.0",
                "num_nodes": len(nodes),
                "num_edges": len(edges)
            }
        }
    
    def deserialize_graph(self, data: Dict[str, Any]) -> ComputationGraph:
        """Deserialize a computation graph from storage format.
        
        Args:
            data: Serialized graph data
            
        Returns:
            Reconstructed ComputationGraph
        """
        nodes = data["nodes"]
        edges = data["edges"]
        root_id = data["root_id"]
        
        # First pass: create all nodes
        id_to_op = {}
        for node_data in nodes:
            op = self._deserialize_node(node_data, id_to_op)
            id_to_op[node_data["id"]] = op
        
        # Second pass: connect dependencies
        # Build adjacency list
        children = {}  # parent_id -> [child_ids]
        for parent_id, child_id in edges:
            if parent_id not in children:
                children[parent_id] = []
            children[parent_id].append(child_id)
        
        # Reconstruct operations with their dependencies
        # This requires operations to support dependency injection
        # We'll need to update the ops after creation
        for parent_id, child_ids in children.items():
            parent_op = id_to_op[parent_id]
            child_ops = [id_to_op[child_id] for child_id in child_ids]
            self._set_op_dependencies(parent_op, child_ops)
        
        # Create and return the graph
        root_op = id_to_op[root_id]
        return ComputationGraph(root_op)
    
    def _get_node_id(self, node: WeightOp) -> int:
        """Get or assign ID for a node."""
        node_key = id(node)
        if node_key not in self._node_to_id:
            self._node_to_id[node_key] = self._node_counter
            self._id_to_node[self._node_counter] = node
            self._node_counter += 1
        return self._node_to_id[node_key]
    
    def _serialize_node(self, node: WeightOp) -> Dict[str, Any]:
        """Serialize a single operation node."""
        node_id = self._get_node_id(node)
        op_type = self._get_op_type(node)
        
        # Get serialized form from the operation
        op_data = node.serialize()
        
        return {
            "id": node_id,
            "type": op_type,
            "data": op_data
        }
    
    def _deserialize_node(self, node_data: Dict[str, Any], 
                         id_to_op: Dict[int, WeightOp]) -> WeightOp:
        """Deserialize a single operation node."""
        op_type = node_data["type"]
        op_class = self.TYPE_OP_MAP.get(op_type)
        
        if op_class is None:
            raise ValueError(f"Unknown operation type: {op_type}")
        
        # Use the operation's deserialize method
        op = op_class.deserialize(node_data["data"])
        
        return op
    
    def _get_op_type(self, node: WeightOp) -> str:
        """Get the type identifier for an operation."""
        for op_class, op_type in self.OP_TYPE_MAP.items():
            if isinstance(node, op_class):
                return op_type
        raise ValueError(f"Unknown operation class: {type(node)}")
    
    def _get_node_dependencies(self, node: WeightOp) -> List[WeightOp]:
        """Get dependencies (child nodes) of an operation.
        
        This method needs to be implemented based on how operations
        store their dependencies. Different operations may have different
        ways of storing child operations.
        """
        deps = []
        
        # Handle different operation types
        if hasattr(node, 'inputs') and isinstance(node.inputs, list):
            # Operations like AddOp, ConcatOp that have multiple inputs
            deps.extend(node.inputs)
        elif hasattr(node, 'input') and isinstance(node.input, WeightOp):
            # Operations with single input
            deps.append(node.input)
        elif hasattr(node, 'left') and hasattr(node, 'right'):
            # Binary operations like MatMulOp
            if isinstance(node.left, WeightOp):
                deps.append(node.left)
            if isinstance(node.right, WeightOp):
                deps.append(node.right)
        elif hasattr(node, 'reference_op'):
            # DeltaOp has a reference operation
            if isinstance(node.reference_op, WeightOp):
                deps.append(node.reference_op)
        
        return deps
    
    def _set_op_dependencies(self, parent_op: WeightOp, 
                            child_ops: List[WeightOp]) -> None:
        """Set dependencies for an operation after deserialization.
        
        This updates the parent operation to reference its child operations.
        The exact method depends on the operation type.
        """
        # Handle different operation types
        if hasattr(parent_op, 'inputs') and isinstance(parent_op.inputs, list):
            # Operations like AddOp, ConcatOp
            parent_op.inputs = child_ops
        elif hasattr(parent_op, 'input'):
            # Single input operations
            if len(child_ops) == 1:
                parent_op.input = child_ops[0]
        elif hasattr(parent_op, 'left') and hasattr(parent_op, 'right'):
            # Binary operations - need to determine which is left/right
            # This would require additional metadata in serialization
            if len(child_ops) >= 1:
                parent_op.left = child_ops[0]
            if len(child_ops) >= 2:
                parent_op.right = child_ops[1]
        elif hasattr(parent_op, 'reference_op'):
            # DeltaOp
            if len(child_ops) == 1:
                parent_op.reference_op = child_ops[0]


class GraphStorageFormat:
    """Defines the HDF5 storage format for computation graphs.
    
    Structure:
    /computation_graphs/
        <graph_hash>/
            attrs:
                version: format version
                created_at: timestamp
                root_id: ID of root node
                num_nodes: total nodes
                num_edges: total edges
            nodes/
                <node_id>/
                    attrs:
                        type: operation type
                        metadata: JSON metadata
                    data: operation-specific data (if any)
            edges: dataset of edge pairs
            metadata: graph-level metadata
    """
    
    GROUP_NAME = "computation_graphs"
    VERSION = "1.0"
    
    @staticmethod
    def validate_version(version: str) -> bool:
        """Check if a version is compatible."""
        # For now, only support exact version match
        # In future, could support backward compatibility
        return version == GraphStorageFormat.VERSION