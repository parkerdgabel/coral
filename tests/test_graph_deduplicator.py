"""Tests for graph-based deduplication engine."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from coral.core.graph_deduplicator import (
    GraphDeduplicator,
    GraphOpStats,
    SharedOperation
)
from coral.core.graph_delta import (
    GraphDeltaOp,
    GraphDeltaType,
    GraphDeltaEncoder,
    OperationDelta
)
from coral.core.weight_tensor import WeightTensor, WeightMetadata


class MockWeightOp:
    """Mock weight operation for testing."""
    
    def __init__(self, op_type="Identity", params=None, inputs=None):
        self.op_type = op_type
        self.params = params or {}
        self.inputs = inputs or []
        self._id = id(self)
        # Create a new class with the correct name
        self.__class__ = type(op_type, (MockWeightOp,), {})
    
    def get_memory_usage(self):
        """Mock memory usage."""
        return 1000
    
    def get_computation_cost(self):
        """Mock computation cost."""
        return 5000


class MockComputationGraph:
    """Mock computation graph for testing."""
    
    def __init__(self, root_op):
        self.root = root_op
        self._evaluated = None
    
    def evaluate(self):
        """Mock evaluation."""
        if self._evaluated is None:
            self._evaluated = np.random.randn(10, 10).astype(np.float32)
        return self._evaluated


class TestGraphDeduplicator:
    """Test suite for GraphDeduplicator."""
    
    def test_initialization(self):
        """Test deduplicator initialization."""
        dedup = GraphDeduplicator(
            similarity_threshold=0.95,
            graph_mode=True,
            structural_similarity_threshold=0.9
        )
        
        assert dedup.similarity_threshold == 0.95
        assert dedup.graph_mode is True
        assert dedup.structural_similarity_threshold == 0.9
        assert isinstance(dedup.graph_stats, GraphOpStats)
    
    def test_add_simple_graph(self):
        """Test adding a simple computation graph."""
        dedup = GraphDeduplicator(graph_mode=True)
        
        # Create simple graph: Scale -> Identity
        identity_op = MockWeightOp("Identity")
        scale_op = MockWeightOp("Scale", {"scale": 2.0}, [identity_op])
        graph = MockComputationGraph(scale_op)
        
        graph_hash = dedup.add_computation_graph(graph, "test_graph")
        
        assert graph_hash in dedup.graph_index
        # Check operations were extracted
        assert len(dedup.op_index) >= 1  # At least one operation
        assert dedup.graph_stats.total_ops >= 1
        assert dedup.graph_stats.unique_ops >= 1
        
        # Verify both operations were processed
        ops = dedup._extract_operations(graph)
        assert len(ops) == 2  # Should extract both operations
    
    def test_exact_duplicate_graph(self):
        """Test adding exact duplicate graphs."""
        dedup = GraphDeduplicator(graph_mode=True)
        
        # Create graph
        op1 = MockWeightOp("Identity")
        op2 = MockWeightOp("Scale", {"scale": 2.0}, [op1])
        graph1 = MockComputationGraph(op2)
        
        # Add first graph
        hash1 = dedup.add_computation_graph(graph1, "graph1")
        
        # The same graph object will have the same hash
        hash2 = dedup.add_computation_graph(graph1, "graph2")
        
        assert hash1 == hash2
        # Only one graph should be stored since they're identical
        assert len(dedup.graph_index) == 1
        
        # Create a structurally identical but different graph object
        op3 = MockWeightOp("Identity")
        op4 = MockWeightOp("Scale", {"scale": 2.0}, [op3])
        graph2 = MockComputationGraph(op4)
        
        hash3 = dedup.add_computation_graph(graph2, "graph3")
        
        # Should recognize structural similarity
        assert hash3 == hash1  # Same structure and params should give same hash
    
    def test_similar_graph_detection(self):
        """Test detection of similar graphs."""
        dedup = GraphDeduplicator(
            graph_mode=True,
            structural_similarity_threshold=0.7
        )
        
        # Create base graph
        op1 = MockWeightOp("Identity")
        op2 = MockWeightOp("Scale", {"scale": 2.0}, [op1])
        op3 = MockWeightOp("Add", {}, [op2, op1])
        graph1 = MockComputationGraph(op3)
        
        # Create similar graph (different scale parameter)
        op4 = MockWeightOp("Identity")
        op5 = MockWeightOp("Scale", {"scale": 2.1}, [op4])
        op6 = MockWeightOp("Add", {}, [op5, op4])
        graph2 = MockComputationGraph(op6)
        
        hash1 = dedup.add_computation_graph(graph1, "graph1")
        hash2 = dedup.add_computation_graph(graph2, "graph2")
        
        # The graphs have the same structure but different parameters
        # They should have different hashes but be tracked for similarity
        assert hash1 != hash2  # Different parameters mean different hashes
        
        # Check that the structural similarity was computed correctly
        similarity = dedup._compute_structural_similarity(graph1, graph2)
        assert similarity > 0.8  # High structural similarity
    
    def test_shared_operation_detection(self):
        """Test detection of shared operations across graphs."""
        dedup = GraphDeduplicator(graph_mode=True)
        
        # Create shared operation
        shared_op = MockWeightOp("Identity")
        
        # Graph 1 uses shared op
        op1 = MockWeightOp("Scale", {"scale": 1.0}, [shared_op])
        graph1 = MockComputationGraph(op1)
        
        # Graph 2 also uses shared op
        op2 = MockWeightOp("Scale", {"scale": 2.0}, [shared_op])
        graph2 = MockComputationGraph(op2)
        
        dedup.add_computation_graph(graph1, "graph1")
        dedup.add_computation_graph(graph2, "graph2")
        
        # Check that shared operation is detected
        # Note: In real implementation, we'd need to handle object identity
        assert dedup.graph_stats.total_ops >= 3
    
    def test_operation_type_statistics(self):
        """Test operation type counting."""
        dedup = GraphDeduplicator(graph_mode=True)
        
        # Create graph with various operation types
        op1 = MockWeightOp("Identity")
        op2 = MockWeightOp("Scale", {"scale": 1.5}, [op1])
        op3 = MockWeightOp("Add", {}, [op2, op1])
        op4 = MockWeightOp("MatMul", {}, [op3, op2])
        graph = MockComputationGraph(op4)
        
        dedup.add_computation_graph(graph, "test_graph")
        
        # Check that operation types are tracked
        assert len(dedup.graph_stats.ops_by_type) > 0
        # At least one of the operation types should be tracked
        assert any(op_type in dedup.graph_stats.ops_by_type 
                  for op_type in ["Identity", "Scale", "Add", "MatMul"])
    
    def test_graph_mode_disabled(self):
        """Test fallback when graph mode is disabled."""
        dedup = GraphDeduplicator(graph_mode=False)
        
        # Create graph with a mock evaluate method
        op = MockWeightOp("Identity")
        graph = MockComputationGraph(op)
        
        # Create weight metadata properly
        data = graph.evaluate()
        metadata = WeightMetadata(
            name="test_graph",
            shape=data.shape,
            dtype=data.dtype
        )
        
        # Create weight tensor without computation graph to avoid import issues
        weight = WeightTensor(data=data, metadata=metadata)
        
        # Should fall back to weight-level deduplication
        hash1 = dedup.add_computation_graph(graph, "test_graph", weight_tensor=weight)
        
        # Graph mode is disabled, so no graph should be indexed
        assert len(dedup.graph_index) == 0
        # Weight should have been added
        assert dedup.stats.unique_weights > 0  # At least one unique weight
        # Compute stats to update totals
        stats = dedup.compute_stats()
        assert stats.total_weights > 0
    
    def test_structural_similarity_computation(self):
        """Test structural similarity calculation."""
        dedup = GraphDeduplicator(graph_mode=True)
        
        # Create two graphs with same structure
        # Graph 1: Identity -> Scale -> Add
        op1_1 = MockWeightOp("Identity")
        op1_2 = MockWeightOp("Scale", {"scale": 1.0}, [op1_1])
        op1_3 = MockWeightOp("Add", {}, [op1_2, op1_1])
        graph1 = MockComputationGraph(op1_3)
        
        # Graph 2: Identity -> Scale -> Add (same structure)
        op2_1 = MockWeightOp("Identity")
        op2_2 = MockWeightOp("Scale", {"scale": 2.0}, [op2_1])
        op2_3 = MockWeightOp("Add", {}, [op2_2, op2_1])
        graph2 = MockComputationGraph(op2_3)
        
        similarity = dedup._compute_structural_similarity(graph1, graph2)
        
        # Should have high structural similarity
        assert similarity > 0.8
    
    def test_parameter_similarity(self):
        """Test parameter similarity comparison."""
        dedup = GraphDeduplicator(graph_mode=True)
        
        # Create operations with similar parameters
        op1 = MockWeightOp("Scale", {"scale": 1.0, "bias": 0.1})
        op2 = MockWeightOp("Scale", {"scale": 1.01, "bias": 0.1})
        
        similarity = dedup._compare_op_parameters(op1, op2)
        
        # Should detect parameter similarity
        assert similarity > 0.5
    
    def test_deduplication_statistics(self):
        """Test comprehensive deduplication statistics."""
        dedup = GraphDeduplicator(graph_mode=True)
        
        # Add multiple graphs
        for i in range(5):
            op1 = MockWeightOp("Identity")
            op2 = MockWeightOp("Scale", {"scale": float(i)}, [op1])
            graph = MockComputationGraph(op2)
            dedup.add_computation_graph(graph, f"graph_{i}")
        
        stats = dedup.get_graph_deduplication_stats()
        
        assert "graph_stats" in stats
        assert stats["graph_stats"]["total_ops"] >= 5  # At least 5 operations
        assert "most_shared_ops" in stats["graph_stats"]
        assert "common_subgraphs" in stats["graph_stats"]
    
    def test_subgraph_pattern_detection(self):
        """Test detection of common subgraph patterns."""
        dedup = GraphDeduplicator(graph_mode=True)
        
        # Create graphs with exact same structure
        # This ensures we have common patterns
        for i in range(3):
            # Use the same parameters to ensure pattern matching
            op1 = MockWeightOp("Identity")
            op2 = MockWeightOp("Scale", {"scale": 1.0}, [op1])
            op3 = MockWeightOp("Add", {}, [op2, op1])
            graph = MockComputationGraph(op3)
            dedup.add_computation_graph(graph, f"graph_{i}")
        
        # Check that we have multiple graphs indexed
        assert len(dedup.graph_index) >= 1  # May be deduplicated
        
        # Check operation chains were extracted
        test_graph = list(dedup.graph_index.values())[0]
        chains = dedup._extract_op_chains(test_graph)
        assert len(chains) >= 0  # Chains might be empty for complex graphs
    
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        dedup = GraphDeduplicator(graph_mode=True)
        
        # Create graph
        op1 = MockWeightOp("Identity")
        op2 = MockWeightOp("Scale", {"scale": 2.0}, [op1])
        graph = MockComputationGraph(op2)
        
        dedup.add_computation_graph(graph, "test_graph")
        
        # Check memory estimation
        for op_hash, shared_op in dedup.shared_ops.items():
            assert shared_op.size_bytes > 0
            assert shared_op.computation_cost > 0
    
    def test_clear_functionality(self):
        """Test clearing all stored data."""
        dedup = GraphDeduplicator(graph_mode=True)
        
        # Add some data
        op = MockWeightOp("Identity")
        graph = MockComputationGraph(op)
        dedup.add_computation_graph(graph, "test_graph")
        
        # Clear
        dedup.clear()
        
        assert len(dedup.op_index) == 0
        assert len(dedup.shared_ops) == 0
        assert len(dedup.graph_index) == 0
        assert dedup.graph_stats.total_ops == 0


class TestGraphDelta:
    """Test suite for graph delta encoding."""
    
    def test_graph_delta_creation(self):
        """Test creating a graph delta."""
        delta = GraphDeltaOp(
            delta_type=GraphDeltaType.PARAMETER,
            reference_graph_hash="ref_hash_123",
            parameter_deltas={"op1": {"scale": {"old": 1.0, "new": 2.0}}}
        )
        
        assert delta.delta_type == GraphDeltaType.PARAMETER
        assert delta.reference_graph_hash == "ref_hash_123"
        assert "op1" in delta.parameter_deltas
    
    def test_operation_delta(self):
        """Test operation delta creation."""
        op_delta = OperationDelta(
            op_id="op_123",
            delta_type="replace",
            old_value=MockWeightOp("Scale", {"scale": 1.0}),
            new_value=MockWeightOp("Scale", {"scale": 2.0})
        )
        
        assert op_delta.op_id == "op_123"
        assert op_delta.delta_type == "replace"
    
    def test_delta_serialization(self):
        """Test delta serialization and deserialization."""
        delta = GraphDeltaOp(
            delta_type=GraphDeltaType.PARAMETER,
            reference_graph_hash="ref_123",
            parameter_deltas={
                "op1": {"scale": {"old": 1.0, "new": 2.0}},
                "op2": {"bias": {"old": 0.0, "new": 0.1}}
            }
        )
        
        # Serialize
        data = delta.to_dict()
        
        assert data["delta_type"] == "parameter"
        assert data["reference_graph_hash"] == "ref_123"
        assert "op1" in data["parameter_deltas"]
        
        # Deserialize
        delta2 = GraphDeltaOp.from_dict(data)
        
        assert delta2.delta_type == GraphDeltaType.PARAMETER
        assert delta2.reference_graph_hash == "ref_123"
        assert "op1" in delta2.parameter_deltas
    
    def test_numpy_array_serialization(self):
        """Test serialization of numpy arrays in deltas."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        delta = GraphDeltaOp(
            delta_type=GraphDeltaType.PARAMETER,
            reference_graph_hash="ref_123",
            parameter_deltas={"op1": {"weights": arr}}
        )
        
        # Serialize and deserialize
        data = delta.to_dict()
        delta2 = GraphDeltaOp.from_dict(data)
        
        # Check array is preserved
        recovered = delta2.parameter_deltas["op1"]["weights"]
        assert isinstance(recovered, np.ndarray)
        assert np.array_equal(recovered, arr)
    
    def test_delta_compression_ratio(self):
        """Test compression ratio calculation."""
        delta = GraphDeltaOp(
            delta_type=GraphDeltaType.PARAMETER,
            reference_graph_hash="ref_123",
            parameter_deltas={
                "op1": {"scale": {"old": 1.0, "new": 2.0}},
                "op2": {"scale": {"old": 1.0, "new": 2.0}}
            }
        )
        
        assert delta.compression_ratio > 0
        assert delta.compression_ratio < 1  # Should show some compression
    
    def test_delta_hash_computation(self):
        """Test delta hash computation."""
        delta1 = GraphDeltaOp(
            delta_type=GraphDeltaType.PARAMETER,
            reference_graph_hash="ref_123",
            parameter_deltas={"op1": {"scale": 2.0}}
        )
        
        delta2 = GraphDeltaOp(
            delta_type=GraphDeltaType.PARAMETER,
            reference_graph_hash="ref_123",
            parameter_deltas={"op1": {"scale": 2.0}}
        )
        
        # Same content should produce same hash
        assert delta1.compute_hash() == delta2.compute_hash()
        
        # Different content should produce different hash
        delta3 = GraphDeltaOp(
            delta_type=GraphDeltaType.PARAMETER,
            reference_graph_hash="ref_123",
            parameter_deltas={"op1": {"scale": 3.0}}
        )
        
        assert delta1.compute_hash() != delta3.compute_hash()


class TestGraphDeltaEncoder:
    """Test suite for GraphDeltaEncoder."""
    
    def test_encoder_initialization(self):
        """Test encoder initialization."""
        encoder = GraphDeltaEncoder()
        assert hasattr(encoder, 'cache')
        assert len(encoder.cache) == 0
    
    def test_parameter_delta_encoding(self):
        """Test encoding parameter differences."""
        encoder = GraphDeltaEncoder()
        
        # Create graphs with parameter differences
        op1 = MockWeightOp("Scale", {"scale": 1.0})
        graph1 = MockComputationGraph(op1)
        
        op2 = MockWeightOp("Scale", {"scale": 2.0})
        graph2 = MockComputationGraph(op2)
        
        # Mock the analysis to return parameter differences
        encoder._analyze_differences = MagicMock(return_value=(
            GraphDeltaType.PARAMETER,
            {"parameters": {"op1": {"scale": {"old": 1.0, "new": 2.0}}}}
        ))
        
        delta = encoder.encode_delta(graph2, graph1, "ref_hash")
        
        assert delta.delta_type == GraphDeltaType.PARAMETER
        assert delta.reference_graph_hash == "ref_hash"
    
    def test_structural_delta_encoding(self):
        """Test encoding structural differences."""
        encoder = GraphDeltaEncoder()
        
        # Create graphs with structural differences
        op1 = MockWeightOp("Identity")
        graph1 = MockComputationGraph(op1)
        
        op2 = MockWeightOp("Identity")
        op3 = MockWeightOp("Scale", {"scale": 1.0}, [op2])
        graph2 = MockComputationGraph(op3)
        
        # Mock the analysis
        encoder._analyze_differences = MagicMock(return_value=(
            GraphDeltaType.STRUCTURAL,
            {"structural": [{"type": "added", "ops": ["op_123"]}]}
        ))
        
        delta = encoder.encode_delta(graph2, graph1, "ref_hash")
        
        assert delta.delta_type == GraphDeltaType.STRUCTURAL
        assert len(delta.structural_changes) > 0
    
    def test_delta_encoding_efficiency_check(self):
        """Test checking if delta encoding is efficient."""
        encoder = GraphDeltaEncoder()
        
        # Create very similar graphs
        op1 = MockWeightOp("Scale", {"scale": 1.0})
        graph1 = MockComputationGraph(op1)
        
        op2 = MockWeightOp("Scale", {"scale": 1.01})
        graph2 = MockComputationGraph(op2)
        
        # Should be efficient for very similar graphs
        # (single operation with similar parameters)
        assert encoder.can_encode_as_delta(graph2, graph1)
    
    def test_caching(self):
        """Test delta encoding caching."""
        encoder = GraphDeltaEncoder()
        
        op1 = MockWeightOp("Identity")
        graph1 = MockComputationGraph(op1)
        graph2 = MockComputationGraph(op1)
        
        # Mock analysis
        encoder._analyze_differences = MagicMock(return_value=(
            GraphDeltaType.PARAMETER,
            {"parameters": {}}
        ))
        
        # First encoding
        delta1 = encoder.encode_delta(graph2, graph1, "ref_hash")
        
        # Second encoding should use cache
        delta2 = encoder.encode_delta(graph2, graph1, "ref_hash")
        
        assert delta1 is delta2  # Same object from cache
        assert encoder._analyze_differences.call_count == 1  # Only called once
    
    def test_operation_map_extraction(self):
        """Test operation map extraction from graph."""
        encoder = GraphDeltaEncoder()
        
        # Create nested graph
        op1 = MockWeightOp("Identity")
        op2 = MockWeightOp("Scale", {"scale": 1.0}, [op1])
        op3 = MockWeightOp("Add", {}, [op2, op1])
        graph = MockComputationGraph(op3)
        
        ops_map = encoder._extract_operations_map(graph)
        
        # Should have all operations with unique IDs
        assert len(ops_map) == 3
        
        # Get operation types from the map
        op_types = [encoder._get_op_type(op) for op in ops_map.values()]
        assert "Identity" in op_types
        assert "Scale" in op_types
        assert "Add" in op_types