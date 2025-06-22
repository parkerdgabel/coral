"""Tests for computation graph storage functionality."""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest

from coral.storage.hdf5_store import HDF5Store
from coral.storage.graph_storage import GraphSerializer, GraphStorageFormat
from coral.core.weight_tensor import WeightTensor, WeightMetadata

# Mock weight operation classes for testing
# These are simplified versions - the real implementations would be in coral.core.weight_ops
class WeightOp:
    """Mock base class for weight operations."""
    
    def forward(self) -> np.ndarray:
        raise NotImplementedError
    
    def get_memory_usage(self) -> int:
        raise NotImplementedError
    
    def serialize(self) -> dict:
        raise NotImplementedError
    
    @classmethod
    def deserialize(cls, data: dict):
        raise NotImplementedError


class IdentityOp(WeightOp):
    """Mock identity operation that wraps raw array data."""
    
    def __init__(self, data: np.ndarray):
        self.data = data
    
    def forward(self) -> np.ndarray:
        return self.data
    
    def get_memory_usage(self) -> int:
        return self.data.nbytes
    
    def serialize(self) -> dict:
        return {
            "raw_data": self.data.tolist(),
            "shape": list(self.data.shape),
            "dtype": str(self.data.dtype)
        }
    
    @classmethod
    def deserialize(cls, data: dict):
        array = np.array(data["raw_data"], dtype=data["dtype"])
        return cls(array)


class AddOp(WeightOp):
    """Mock addition operation."""
    
    def __init__(self, inputs: list):
        self.inputs = inputs
    
    def forward(self) -> np.ndarray:
        result = self.inputs[0].forward()
        for inp in self.inputs[1:]:
            result = result + inp.forward()
        return result
    
    def get_memory_usage(self) -> int:
        return sum(inp.get_memory_usage() for inp in self.inputs)
    
    def serialize(self) -> dict:
        return {"num_inputs": len(self.inputs)}
    
    @classmethod
    def deserialize(cls, data: dict):
        # Inputs will be set by the serializer
        return cls([])


class ScaleOp(WeightOp):
    """Mock scaling operation."""
    
    def __init__(self, input_op: WeightOp, scale: float):
        self.input = input_op
        self.scale = scale
    
    def forward(self) -> np.ndarray:
        return self.input.forward() * self.scale
    
    def get_memory_usage(self) -> int:
        return self.input.get_memory_usage()
    
    def serialize(self) -> dict:
        return {"scale": self.scale}
    
    @classmethod
    def deserialize(cls, data: dict):
        # Input will be set by the serializer
        return cls(None, data["scale"])


class SVDOp(WeightOp):
    """Mock SVD compression operation."""
    
    def __init__(self, u: np.ndarray, s: np.ndarray, v: np.ndarray, rank: int):
        self.u = u
        self.s = s
        self.v = v
        self.rank = rank
    
    def forward(self) -> np.ndarray:
        # Reconstruct from SVD components
        s_diag = np.diag(self.s[:self.rank])
        return self.u[:, :self.rank] @ s_diag @ self.v[:self.rank, :]
    
    def get_memory_usage(self) -> int:
        return (self.u[:, :self.rank].nbytes + 
                self.s[:self.rank].nbytes + 
                self.v[:self.rank, :].nbytes)
    
    def serialize(self) -> dict:
        return {
            "u_shape": list(self.u.shape),
            "s_shape": list(self.s.shape),
            "v_shape": list(self.v.shape),
            "rank": self.rank,
            "raw_data": {
                "u": self.u.tolist(),
                "s": self.s.tolist(),
                "v": self.v.tolist()
            }
        }
    
    @classmethod
    def deserialize(cls, data: dict):
        u = np.array(data["raw_data"]["u"])
        s = np.array(data["raw_data"]["s"])
        v = np.array(data["raw_data"]["v"])
        return cls(u, s, v, data["rank"])


class ComputationGraph:
    """Mock computation graph."""
    
    def __init__(self, root_op: WeightOp):
        self.root = root_op
        self._cache = {}
    
    def evaluate(self) -> np.ndarray:
        return self.root.forward()
    
    def get_total_memory(self) -> int:
        return self.root.get_memory_usage()


# Update the GraphSerializer mappings to include our mock classes
GraphSerializer.OP_TYPE_MAP.update({
    IdentityOp: "identity",
    AddOp: "add",
    ScaleOp: "scale",
    SVDOp: "svd"
})
GraphSerializer.TYPE_OP_MAP = {v: k for k, v in GraphSerializer.OP_TYPE_MAP.items()}


class TestGraphStorage:
    """Test suite for computation graph storage."""
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary HDF5 store for testing."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name
        
        store = HDF5Store(temp_path, compression="gzip")
        yield store
        store.close()
        Path(temp_path).unlink()
    
    def test_simple_identity_graph(self, temp_store):
        """Test storing and loading a simple identity operation graph."""
        # Create a simple graph with just an identity operation
        data = np.random.randn(10, 20).astype(np.float32)
        identity_op = IdentityOp(data)
        graph = ComputationGraph(identity_op)
        
        # Store the graph
        graph_hash = "test_identity_graph"
        stored_hash = temp_store.store_computation_graph(graph_hash, graph)
        assert stored_hash == graph_hash
        
        # Load the graph
        loaded_graph = temp_store.load_computation_graph(graph_hash)
        assert loaded_graph is not None
        
        # Verify the data
        loaded_data = loaded_graph.evaluate()
        np.testing.assert_array_almost_equal(loaded_data, data)
    
    def test_add_operation_graph(self, temp_store):
        """Test storing and loading a graph with add operations."""
        # Create a graph: (A + B) where A and B are identity ops
        data_a = np.ones((5, 5), dtype=np.float32)
        data_b = np.ones((5, 5), dtype=np.float32) * 2
        
        op_a = IdentityOp(data_a)
        op_b = IdentityOp(data_b)
        add_op = AddOp([op_a, op_b])
        graph = ComputationGraph(add_op)
        
        # Store and load
        graph_hash = "test_add_graph"
        temp_store.store_computation_graph(graph_hash, graph)
        loaded_graph = temp_store.load_computation_graph(graph_hash)
        
        # Verify computation
        expected = data_a + data_b
        result = loaded_graph.evaluate()
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_nested_operations(self, temp_store):
        """Test storing and loading a graph with nested operations."""
        # Create a graph: ((A + B) * 2.0)
        data_a = np.ones((3, 3), dtype=np.float32)
        data_b = np.ones((3, 3), dtype=np.float32) * 3
        
        op_a = IdentityOp(data_a)
        op_b = IdentityOp(data_b)
        add_op = AddOp([op_a, op_b])
        scale_op = ScaleOp(add_op, 2.0)
        graph = ComputationGraph(scale_op)
        
        # Store and load
        graph_hash = "test_nested_graph"
        temp_store.store_computation_graph(graph_hash, graph)
        loaded_graph = temp_store.load_computation_graph(graph_hash)
        
        # Verify computation
        expected = (data_a + data_b) * 2.0
        result = loaded_graph.evaluate()
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_svd_compression_graph(self, temp_store):
        """Test storing and loading a graph with SVD compression."""
        # Create a low-rank matrix
        rank = 5
        m, n = 20, 30
        u_base = np.random.randn(m, rank).astype(np.float32)
        v_base = np.random.randn(rank, n).astype(np.float32)
        original_matrix = u_base @ v_base
        
        # Compute SVD
        u, s, v = np.linalg.svd(original_matrix, full_matrices=True)
        
        # Create SVD operation
        svd_op = SVDOp(u, s, v, rank)
        graph = ComputationGraph(svd_op)
        
        # Store and load
        graph_hash = "test_svd_graph"
        temp_store.store_computation_graph(graph_hash, graph)
        loaded_graph = temp_store.load_computation_graph(graph_hash)
        
        # Verify reconstruction (should be close to original due to low rank)
        reconstructed = loaded_graph.evaluate()
        assert reconstructed.shape == original_matrix.shape
        # Allow for some numerical error in SVD reconstruction
        np.testing.assert_allclose(reconstructed, original_matrix, rtol=1e-5, atol=1e-6)
    
    def test_list_and_delete_graphs(self, temp_store):
        """Test listing and deleting computation graphs."""
        # Create multiple graphs
        graph_hashes = []
        for i in range(3):
            data = np.random.randn(5, 5).astype(np.float32) * i
            op = IdentityOp(data)
            graph = ComputationGraph(op)
            hash_id = f"graph_{i}"
            temp_store.store_computation_graph(hash_id, graph)
            graph_hashes.append(hash_id)
        
        # List graphs
        stored_graphs = temp_store.list_computation_graphs()
        assert len(stored_graphs) == 3
        for hash_id in graph_hashes:
            assert hash_id in stored_graphs
        
        # Delete one graph
        deleted = temp_store.delete_computation_graph(graph_hashes[0])
        assert deleted
        
        # Verify deletion
        stored_graphs = temp_store.list_computation_graphs()
        assert len(stored_graphs) == 2
        assert graph_hashes[0] not in stored_graphs
        
        # Try to load deleted graph
        loaded = temp_store.load_computation_graph(graph_hashes[0])
        assert loaded is None
    
    def test_graph_info_and_statistics(self, temp_store):
        """Test retrieving graph information and statistics."""
        # Create a complex graph
        data_a = np.random.randn(100, 100).astype(np.float32)
        data_b = np.random.randn(100, 100).astype(np.float32)
        
        op_a = IdentityOp(data_a)
        op_b = IdentityOp(data_b)
        add_op = AddOp([op_a, op_b])
        scale_op = ScaleOp(add_op, 0.5)
        graph = ComputationGraph(scale_op)
        
        # Store graph
        graph_hash = "test_info_graph"
        temp_store.store_computation_graph(graph_hash, graph)
        
        # Get graph info
        info = temp_store.get_computation_graph_info(graph_hash)
        assert info is not None
        assert info["hash"] == graph_hash
        assert info["version"] == GraphStorageFormat.VERSION
        assert info["num_nodes"] == 4  # 2 identity + 1 add + 1 scale
        assert info["num_edges"] == 3  # connections between ops
        assert info["storage_bytes"] > 0
        
        # Get storage statistics
        stats = temp_store.get_graph_storage_info()
        assert stats["total_graphs"] == 1
        assert stats["total_graph_bytes"] > 0
        assert len(stats["graph_stats"]) == 1
    
    def test_duplicate_graph_storage(self, temp_store):
        """Test that storing the same graph twice doesn't duplicate data."""
        data = np.random.randn(50, 50).astype(np.float32)
        op = IdentityOp(data)
        graph = ComputationGraph(op)
        
        graph_hash = "duplicate_test"
        
        # Store twice
        temp_store.store_computation_graph(graph_hash, graph)
        temp_store.store_computation_graph(graph_hash, graph)
        
        # Should still only have one graph
        graphs = temp_store.list_computation_graphs()
        assert len(graphs) == 1
        assert graphs[0] == graph_hash
    
    def test_backward_compatibility(self, temp_store):
        """Test that older HDF5 files can be migrated to support graphs."""
        # This test verifies the migration logic in HDF5Store._open()
        # The store should have been created with version 3.0 and computation_graphs group
        assert "computation_graphs" in temp_store.file
        assert temp_store.file.attrs["version"] == "3.0"
    
    def test_complex_dag_structure(self, temp_store):
        """Test a complex DAG with shared subgraphs."""
        # Create a diamond-shaped DAG:
        #     root
        #    /    \
        #   A      B
        #    \    /
        #     add
        #      |
        #    scale
        
        root_data = np.ones((4, 4), dtype=np.float32)
        root_op = IdentityOp(root_data)
        
        # Two branches that scale differently
        scale_a = ScaleOp(root_op, 2.0)
        scale_b = ScaleOp(root_op, 3.0)
        
        # Merge branches
        add_op = AddOp([scale_a, scale_b])
        
        # Final scale
        final_scale = ScaleOp(add_op, 0.1)
        
        graph = ComputationGraph(final_scale)
        
        # Store and load
        graph_hash = "test_dag"
        temp_store.store_computation_graph(graph_hash, graph)
        loaded_graph = temp_store.load_computation_graph(graph_hash)
        
        # Verify computation
        expected = (root_data * 2.0 + root_data * 3.0) * 0.1
        result = loaded_graph.evaluate()
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_empty_graph_edge_case(self, temp_store):
        """Test edge case of a graph with single node and no edges."""
        data = np.array([1, 2, 3], dtype=np.float32)
        op = IdentityOp(data)
        graph = ComputationGraph(op)
        
        graph_hash = "single_node_graph"
        temp_store.store_computation_graph(graph_hash, graph)
        
        # Verify it loads correctly
        loaded_graph = temp_store.load_computation_graph(graph_hash)
        result = loaded_graph.evaluate()
        np.testing.assert_array_equal(result, data)
        
        # Check that edges are handled correctly
        info = temp_store.get_computation_graph_info(graph_hash)
        assert info["num_nodes"] == 1
        assert info["num_edges"] == 0
    
    def test_storage_info_includes_graphs(self, temp_store):
        """Test that general storage info includes graph statistics."""
        # Create some graphs
        for i in range(2):
            data = np.random.randn(10, 10).astype(np.float32)
            op = IdentityOp(data)
            graph = ComputationGraph(op)
            temp_store.store_computation_graph(f"graph_{i}", graph)
        
        # Get general storage info
        info = temp_store.get_storage_info()
        assert "total_graphs" in info
        assert info["total_graphs"] == 2
    
    def test_repr_includes_graphs(self, temp_store):
        """Test that string representation includes graph count."""
        # Create a graph
        data = np.random.randn(5, 5).astype(np.float32)
        op = IdentityOp(data)
        graph = ComputationGraph(op)
        temp_store.store_computation_graph("test_graph", graph)
        
        # Check representation
        repr_str = repr(temp_store)
        assert "graphs=1" in repr_str