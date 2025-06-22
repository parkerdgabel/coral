"""Graph delta encoding for computation graph differences."""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import xxhash

import numpy as np

logger = logging.getLogger(__name__)


class GraphDeltaType(Enum):
    """Types of graph delta encoding."""
    
    STRUCTURAL = "structural"  # Different graph structure
    PARAMETER = "parameter"  # Same structure, different parameters
    OPERATION = "operation"  # Different operations at specific nodes
    SUBGRAPH = "subgraph"  # Subgraph replacement
    COMBINED = "combined"  # Multiple delta types


@dataclass
class OperationDelta:
    """Delta for a single operation."""
    
    op_id: str  # Identifier within graph
    delta_type: str  # Type of change
    old_value: Any  # Original value/operation
    new_value: Any  # New value/operation
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphDeltaOp:
    """
    Represents differences between two computation graphs.
    
    Enables efficient storage of similar graphs by storing only differences
    from a reference graph.
    """
    
    delta_type: GraphDeltaType
    reference_graph_hash: str
    operations: List[OperationDelta] = field(default_factory=list)
    structural_changes: Dict[str, Any] = field(default_factory=dict)
    parameter_deltas: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    compression_ratio: float = 0.0
    
    def __post_init__(self):
        """Calculate compression ratio after initialization."""
        if not self.compression_ratio:
            self._calculate_compression_ratio()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "delta_type": self.delta_type.value,
            "reference_graph_hash": self.reference_graph_hash,
            "operations": [
                {
                    "op_id": op.op_id,
                    "delta_type": op.delta_type,
                    "old_value": self._serialize_value(op.old_value),
                    "new_value": self._serialize_value(op.new_value),
                    "metadata": op.metadata
                }
                for op in self.operations
            ],
            "structural_changes": self.structural_changes,
            "parameter_deltas": self._serialize_param_deltas(self.parameter_deltas),
            "metadata": self.metadata,
            "compression_ratio": self.compression_ratio
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphDeltaOp":
        """Deserialize from dictionary."""
        operations = [
            OperationDelta(
                op_id=op["op_id"],
                delta_type=op["delta_type"],
                old_value=cls._deserialize_value(op["old_value"]),
                new_value=cls._deserialize_value(op["new_value"]),
                metadata=op.get("metadata", {})
            )
            for op in data.get("operations", [])
        ]
        
        return cls(
            delta_type=GraphDeltaType(data["delta_type"]),
            reference_graph_hash=data["reference_graph_hash"],
            operations=operations,
            structural_changes=data.get("structural_changes", {}),
            parameter_deltas=cls._deserialize_param_deltas(
                data.get("parameter_deltas", {})
            ),
            metadata=data.get("metadata", {}),
            compression_ratio=data.get("compression_ratio", 0.0)
        )
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for storage."""
        if isinstance(value, np.ndarray):
            return {
                "_type": "ndarray",
                "data": value.tolist(),
                "dtype": str(value.dtype),
                "shape": value.shape
            }
        elif hasattr(value, "__dict__"):
            return {
                "_type": "object",
                "class": value.__class__.__name__,
                "data": {k: self._serialize_value(v) 
                        for k, v in value.__dict__.items()}
            }
        else:
            return value
    
    @staticmethod
    def _deserialize_value(value: Any) -> Any:
        """Deserialize a value from storage."""
        if isinstance(value, dict) and "_type" in value:
            if value["_type"] == "ndarray":
                return np.array(
                    value["data"],
                    dtype=np.dtype(value["dtype"])
                ).reshape(value["shape"])
            elif value["_type"] == "object":
                # Return as dict for now - actual object reconstruction
                # would require access to class definitions
                return value["data"]
        return value
    
    def _serialize_param_deltas(self, deltas: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize parameter deltas."""
        serialized = {}
        for key, value in deltas.items():
            if isinstance(value, np.ndarray):
                serialized[key] = self._serialize_value(value)
            else:
                serialized[key] = value
        return serialized
    
    @staticmethod
    def _deserialize_param_deltas(deltas: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize parameter deltas."""
        deserialized = {}
        for key, value in deltas.items():
            deserialized[key] = GraphDeltaOp._deserialize_value(value)
        return deserialized
    
    def _calculate_compression_ratio(self):
        """Estimate compression ratio of delta encoding."""
        # Estimate size of full graph vs delta
        # This is a simplified calculation
        
        delta_size = 0
        
        # Size of operation deltas
        for op in self.operations:
            delta_size += 100  # Base overhead
            delta_size += self._estimate_value_size(op.old_value)
            delta_size += self._estimate_value_size(op.new_value)
        
        # Size of structural changes
        delta_size += len(json.dumps(self.structural_changes))
        
        # Size of parameter deltas
        for key, value in self.parameter_deltas.items():
            delta_size += len(key) + self._estimate_value_size(value)
        
        # Estimate original graph size (assuming delta is ~10-30% of original)
        estimated_original = delta_size * 5
        
        self.compression_ratio = 1.0 - (delta_size / estimated_original)
    
    def _estimate_value_size(self, value: Any) -> int:
        """Estimate size of a value in bytes."""
        if isinstance(value, np.ndarray):
            return value.nbytes
        elif isinstance(value, (int, float)):
            return 8
        elif isinstance(value, str):
            return len(value)
        elif isinstance(value, dict):
            return sum(self._estimate_value_size(v) for v in value.values())
        elif isinstance(value, list):
            return sum(self._estimate_value_size(v) for v in value)
        else:
            return 100  # Default estimate
    
    @property
    def nbytes(self) -> int:
        """Estimate total size in bytes."""
        total = 0
        
        # Operation deltas
        for op in self.operations:
            total += 50  # Overhead
            total += self._estimate_value_size(op.old_value)
            total += self._estimate_value_size(op.new_value)
        
        # Structural changes
        total += len(json.dumps(self.structural_changes))
        
        # Parameter deltas
        for value in self.parameter_deltas.values():
            total += self._estimate_value_size(value)
        
        # Metadata
        total += len(json.dumps(self.metadata))
        
        return total
    
    def compute_hash(self) -> str:
        """Compute hash of this delta."""
        hasher = xxhash.xxh3_64()
        
        # Hash all components
        hasher.update(self.delta_type.value.encode())
        hasher.update(self.reference_graph_hash.encode())
        
        # Hash operations in consistent order
        for op in sorted(self.operations, key=lambda x: x.op_id):
            hasher.update(op.op_id.encode())
            hasher.update(op.delta_type.encode())
            hasher.update(str(op.old_value).encode())
            hasher.update(str(op.new_value).encode())
        
        # Hash other components
        hasher.update(json.dumps(self.structural_changes, sort_keys=True).encode())
        hasher.update(json.dumps(self.parameter_deltas, sort_keys=True).encode())
        
        return hasher.hexdigest()


class GraphDeltaEncoder:
    """Encoder for creating and applying graph deltas."""
    
    def __init__(self):
        """Initialize the encoder."""
        self.cache = {}  # Cache for computed deltas
    
    def encode_delta(
        self,
        target_graph: Any,
        reference_graph: Any,
        reference_hash: str
    ) -> GraphDeltaOp:
        """
        Create a delta encoding from reference to target graph.
        
        Args:
            target_graph: The graph to encode
            reference_graph: The reference graph
            reference_hash: Hash of the reference graph
            
        Returns:
            GraphDeltaOp representing the differences
        """
        # Check cache
        cache_key = (id(target_graph), reference_hash)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Analyze differences
        delta_type, differences = self._analyze_differences(
            target_graph, reference_graph
        )
        
        # Create appropriate delta
        if delta_type == GraphDeltaType.PARAMETER:
            delta = self._create_parameter_delta(
                target_graph, reference_graph, reference_hash, differences
            )
        elif delta_type == GraphDeltaType.STRUCTURAL:
            delta = self._create_structural_delta(
                target_graph, reference_graph, reference_hash, differences
            )
        elif delta_type == GraphDeltaType.OPERATION:
            delta = self._create_operation_delta(
                target_graph, reference_graph, reference_hash, differences
            )
        else:
            # Combined delta
            delta = self._create_combined_delta(
                target_graph, reference_graph, reference_hash, differences
            )
        
        # Cache result
        self.cache[cache_key] = delta
        
        return delta
    
    def decode_delta(
        self,
        delta: GraphDeltaOp,
        reference_graph: Any
    ) -> Any:
        """
        Reconstruct target graph from delta and reference.
        
        Args:
            delta: The graph delta
            reference_graph: The reference graph
            
        Returns:
            Reconstructed target graph
        """
        if delta.delta_type == GraphDeltaType.PARAMETER:
            return self._apply_parameter_delta(delta, reference_graph)
        elif delta.delta_type == GraphDeltaType.STRUCTURAL:
            return self._apply_structural_delta(delta, reference_graph)
        elif delta.delta_type == GraphDeltaType.OPERATION:
            return self._apply_operation_delta(delta, reference_graph)
        else:
            return self._apply_combined_delta(delta, reference_graph)
    
    def _analyze_differences(
        self,
        target: Any,
        reference: Any
    ) -> Tuple[GraphDeltaType, Dict[str, Any]]:
        """Analyze differences between two graphs."""
        differences = {
            "structural": [],
            "parameters": {},
            "operations": [],
            "metadata": {}
        }
        
        # Compare structure
        target_ops = self._extract_operations_map(target)
        ref_ops = self._extract_operations_map(reference)
        
        # Check for structural differences
        target_ids = set(target_ops.keys())
        ref_ids = set(ref_ops.keys())
        
        added_ops = target_ids - ref_ids
        removed_ops = ref_ids - target_ids
        common_ops = target_ids & ref_ids
        
        if added_ops or removed_ops:
            differences["structural"].extend([
                {"type": "added", "ops": list(added_ops)},
                {"type": "removed", "ops": list(removed_ops)}
            ])
        
        # Check operations and parameters in common nodes
        for op_id in common_ops:
            target_op = target_ops[op_id]
            ref_op = ref_ops[op_id]
            
            # Check if operation type changed
            if self._get_op_type(target_op) != self._get_op_type(ref_op):
                differences["operations"].append({
                    "op_id": op_id,
                    "old_type": self._get_op_type(ref_op),
                    "new_type": self._get_op_type(target_op)
                })
            else:
                # Check parameters
                param_diff = self._compare_op_parameters(target_op, ref_op)
                if param_diff:
                    differences["parameters"][op_id] = param_diff
        
        # Determine primary delta type
        if differences["structural"]:
            delta_type = GraphDeltaType.STRUCTURAL
        elif differences["operations"]:
            delta_type = GraphDeltaType.OPERATION
        elif differences["parameters"]:
            delta_type = GraphDeltaType.PARAMETER
        else:
            # If multiple types of changes, use combined
            delta_type = GraphDeltaType.COMBINED
        
        return delta_type, differences
    
    def _create_parameter_delta(
        self,
        target: Any,
        reference: Any,
        ref_hash: str,
        differences: Dict[str, Any]
    ) -> GraphDeltaOp:
        """Create delta for parameter-only changes."""
        delta = GraphDeltaOp(
            delta_type=GraphDeltaType.PARAMETER,
            reference_graph_hash=ref_hash,
            parameter_deltas=differences["parameters"]
        )
        
        # Add metadata about the changes
        delta.metadata["num_param_changes"] = len(differences["parameters"])
        delta.metadata["changed_ops"] = list(differences["parameters"].keys())
        
        return delta
    
    def _create_structural_delta(
        self,
        target: Any,
        reference: Any,
        ref_hash: str,
        differences: Dict[str, Any]
    ) -> GraphDeltaOp:
        """Create delta for structural changes."""
        delta = GraphDeltaOp(
            delta_type=GraphDeltaType.STRUCTURAL,
            reference_graph_hash=ref_hash,
            structural_changes=differences["structural"]
        )
        
        # Add operation deltas for added/modified operations
        for change in differences["structural"]:
            if change["type"] == "added":
                for op_id in change["ops"]:
                    target_ops = self._extract_operations_map(target)
                    if op_id in target_ops:
                        delta.operations.append(
                            OperationDelta(
                                op_id=op_id,
                                delta_type="add",
                                old_value=None,
                                new_value=target_ops[op_id]
                            )
                        )
        
        return delta
    
    def _create_operation_delta(
        self,
        target: Any,
        reference: Any,
        ref_hash: str,
        differences: Dict[str, Any]
    ) -> GraphDeltaOp:
        """Create delta for operation changes."""
        delta = GraphDeltaOp(
            delta_type=GraphDeltaType.OPERATION,
            reference_graph_hash=ref_hash
        )
        
        target_ops = self._extract_operations_map(target)
        ref_ops = self._extract_operations_map(reference)
        
        for op_change in differences["operations"]:
            op_id = op_change["op_id"]
            delta.operations.append(
                OperationDelta(
                    op_id=op_id,
                    delta_type="replace",
                    old_value=ref_ops.get(op_id),
                    new_value=target_ops.get(op_id)
                )
            )
        
        return delta
    
    def _create_combined_delta(
        self,
        target: Any,
        reference: Any,
        ref_hash: str,
        differences: Dict[str, Any]
    ) -> GraphDeltaOp:
        """Create delta combining multiple change types."""
        delta = GraphDeltaOp(
            delta_type=GraphDeltaType.COMBINED,
            reference_graph_hash=ref_hash,
            structural_changes=differences.get("structural", []),
            parameter_deltas=differences.get("parameters", {})
        )
        
        # Add all operation changes
        target_ops = self._extract_operations_map(target)
        ref_ops = self._extract_operations_map(reference)
        
        # Process operation changes
        for op_change in differences.get("operations", []):
            op_id = op_change["op_id"]
            delta.operations.append(
                OperationDelta(
                    op_id=op_id,
                    delta_type="replace",
                    old_value=ref_ops.get(op_id),
                    new_value=target_ops.get(op_id)
                )
            )
        
        return delta
    
    def _apply_parameter_delta(
        self,
        delta: GraphDeltaOp,
        reference: Any
    ) -> Any:
        """Apply parameter-only delta to reference graph."""
        # This would create a copy of the reference graph
        # and update only the parameters specified in the delta
        
        # For now, return a mock implementation
        # Real implementation would depend on the actual graph structure
        logger.info(
            f"Applying parameter delta with {len(delta.parameter_deltas)} changes"
        )
        return reference  # Placeholder
    
    def _apply_structural_delta(
        self,
        delta: GraphDeltaOp,
        reference: Any
    ) -> Any:
        """Apply structural delta to reference graph."""
        logger.info("Applying structural delta")
        return reference  # Placeholder
    
    def _apply_operation_delta(
        self,
        delta: GraphDeltaOp,
        reference: Any
    ) -> Any:
        """Apply operation delta to reference graph."""
        logger.info(
            f"Applying operation delta with {len(delta.operations)} changes"
        )
        return reference  # Placeholder
    
    def _apply_combined_delta(
        self,
        delta: GraphDeltaOp,
        reference: Any
    ) -> Any:
        """Apply combined delta to reference graph."""
        logger.info("Applying combined delta")
        return reference  # Placeholder
    
    def _extract_operations_map(self, graph: Any) -> Dict[str, Any]:
        """Extract operations as a map from ID to operation."""
        ops_map = {}
        visited = set()
        
        def traverse(op, path=""):
            if op is None or id(op) in visited:
                return
            visited.add(id(op))
            
            # Create unique ID based on path
            op_type = self._get_op_type(op)
            if path:
                op_id = f"{path}/{op_type}_{id(op)}"
            else:
                op_id = f"{op_type}_{id(op)}"
            ops_map[op_id] = op
            
            if hasattr(op, 'inputs'):
                for i, input_op in enumerate(op.inputs):
                    if input_op is not None:
                        new_path = f"{path}/input_{i}" if path else f"input_{i}"
                        traverse(input_op, new_path)
        
        if hasattr(graph, 'root'):
            traverse(graph.root, "root")
        
        return ops_map
    
    def _get_op_type(self, op: Any) -> str:
        """Get operation type name."""
        return op.__class__.__name__
    
    def _compare_op_parameters(self, op1: Any, op2: Any) -> Optional[Dict[str, Any]]:
        """Compare parameters between two operations of the same type."""
        param_attrs = [
            'scale', 'bias', 'rank', 'bits', 'shape',
            'axis', 'keepdims', 'dtype', 'compression_level'
        ]
        
        differences = {}
        
        for attr in param_attrs:
            if hasattr(op1, attr) and hasattr(op2, attr):
                val1 = getattr(op1, attr)
                val2 = getattr(op2, attr)
                
                if not self._values_equal(val1, val2):
                    differences[attr] = {
                        "old": val2,
                        "new": val1
                    }
        
        return differences if differences else None
    
    def _values_equal(self, v1: Any, v2: Any) -> bool:
        """Check if two values are equal."""
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            return np.array_equal(v1, v2)
        return v1 == v2
    
    def can_encode_as_delta(
        self,
        target_graph: Any,
        reference_graph: Any,
        max_delta_ratio: float = 0.5
    ) -> bool:
        """
        Check if target can be efficiently encoded as delta from reference.
        
        Args:
            target_graph: Graph to encode
            reference_graph: Reference graph
            max_delta_ratio: Maximum ratio of delta size to original size
            
        Returns:
            True if delta encoding would be efficient
        """
        # Quick structural check
        target_ops = self._extract_operations_map(target_graph)
        ref_ops = self._extract_operations_map(reference_graph)
        
        # If graphs have same number of operations, check similarity
        if len(target_ops) == len(ref_ops):
            # For single-op graphs with same type, delta is efficient
            if len(target_ops) == 1:
                target_op = list(target_ops.values())[0]
                ref_op = list(ref_ops.values())[0]
                if self._get_op_type(target_op) == self._get_op_type(ref_op):
                    return True
        
        # If very different structures, delta may not be efficient
        if len(target_ops) == 0 or len(ref_ops) == 0:
            return False
            
        # Calculate overlap of operation types
        target_types = {self._get_op_type(op) for op in target_ops.values()}
        ref_types = {self._get_op_type(op) for op in ref_ops.values()}
        type_overlap = len(target_types & ref_types) / len(target_types | ref_types)
        
        if type_overlap < 0.5:  # Less than 50% type overlap
            return False
        
        # Estimate delta size
        _, differences = self._analyze_differences(target_graph, reference_graph)
        
        # Count significant changes
        num_structural = len(differences.get("structural", []))
        num_operations = len(differences.get("operations", []))
        num_parameters = len(differences.get("parameters", {}))
        
        total_changes = num_structural + num_operations + num_parameters
        total_ops = len(target_ops)
        
        # If too many changes, delta encoding may not be efficient
        change_ratio = total_changes / total_ops if total_ops > 0 else 1.0
        
        return change_ratio <= max_delta_ratio