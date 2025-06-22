"""Tests for WeightTensor class"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from coral.core.weight_tensor import WeightMetadata, WeightTensor

# Try to import weight ops for graph tests
try:
    from coral.core.weight_ops import (
        ComputationGraph,
        IdentityOp,
        AddOp,
        ScaleOp,
        MatMulOp,
        ReshapeOp,
    )
    WEIGHT_OPS_AVAILABLE = True
except ImportError:
    WEIGHT_OPS_AVAILABLE = False


class TestWeightTensor:
    def test_creation(self):
        """Test weight tensor creation"""
        data = np.random.randn(10, 10).astype(np.float32)
        metadata = WeightMetadata(name="test_weight", shape=(10, 10), dtype=np.float32)

        weight = WeightTensor(data=data, metadata=metadata)

        assert weight.shape == (10, 10)
        assert weight.dtype == np.float32
        assert weight.size == 100
        assert weight.nbytes == 400  # 100 * 4 bytes

    def test_auto_metadata(self):
        """Test automatic metadata creation"""
        data = np.random.randn(5, 5).astype(np.float64)
        weight = WeightTensor(data=data)

        assert weight.metadata.shape == (5, 5)
        assert weight.metadata.dtype == np.float64
        assert weight.metadata.name == "unnamed"

    def test_hash_computation(self):
        """Test content-based hashing"""
        data = np.ones((10, 10), dtype=np.float32)
        weight1 = WeightTensor(data=data)
        weight2 = WeightTensor(data=data.copy())

        # Same data should produce same hash
        assert weight1.compute_hash() == weight2.compute_hash()

        # Different data should produce different hash
        weight3 = WeightTensor(data=data * 2)
        assert weight1.compute_hash() != weight3.compute_hash()

    def test_similarity(self):
        """Test weight similarity detection"""
        # Create deterministic weights for reliable testing
        data1 = np.ones((4, 4), dtype=np.float32)  # All 1s
        data2 = np.zeros((4, 4), dtype=np.float32)
        data2[0, 0] = 1.0  # Mostly zeros with one 1

        weight1 = WeightTensor(data=data1)
        weight2 = WeightTensor(data=data2)

        # These should have moderate similarity (cosine similarity = 0.25)
        assert weight1.is_similar_to(weight2, threshold=0.2)

        # Should not be similar with high threshold
        assert not weight1.is_similar_to(weight2, threshold=0.5)

        # Different shapes should never be similar
        weight3 = WeightTensor(data=np.random.randn(5, 5).astype(np.float32))
        assert not weight1.is_similar_to(weight3)

    def test_serialization(self):
        """Test conversion to/from dict"""
        data = np.random.randn(10, 10).astype(np.float32)
        metadata = WeightMetadata(
            name="test_weight",
            shape=(10, 10),
            dtype=np.float32,
            layer_type="Linear",
            model_name="test_model",
        )

        weight = WeightTensor(data=data, metadata=metadata)

        # Convert to dict
        weight_dict = weight.to_dict()

        assert weight_dict["metadata"]["name"] == "test_weight"
        assert weight_dict["metadata"]["layer_type"] == "Linear"
        assert weight_dict["has_data"]

        # Convert back
        restored = WeightTensor.from_dict(weight_dict, weight_data=data)

        assert restored.metadata.name == weight.metadata.name
        assert restored.shape == weight.shape
        assert np.array_equal(restored.data, weight.data)


@pytest.mark.skipif(not WEIGHT_OPS_AVAILABLE, reason="Weight ops not available")
class TestWeightTensorComputationGraph:
    """Test WeightTensor with computation graph support"""
    
    def test_graph_creation(self):
        """Test creating weight tensor with computation graph"""
        # Create a simple computation graph
        data = np.random.randn(10, 10).astype(np.float32)
        identity_op = IdentityOp(data)
        graph = ComputationGraph(identity_op)
        
        # Create weight tensor with graph
        weight = WeightTensor(computation_graph=graph)
        
        # Check that data is lazily evaluated
        assert weight._data is None
        assert not weight._materialized
        
        # Access data triggers evaluation
        result = weight.data
        assert np.array_equal(result, data)
        assert weight._materialized
        
    def test_graph_operations(self):
        """Test various graph operations"""
        # Create a more complex graph: (A + B) * 2
        a = np.ones((5, 5), dtype=np.float32)
        b = np.ones((5, 5), dtype=np.float32) * 2
        
        add_op = AddOp([IdentityOp(a), IdentityOp(b)])
        scale_op = ScaleOp(add_op, 2.0)
        graph = ComputationGraph(scale_op)
        
        weight = WeightTensor(computation_graph=graph)
        
        # Verify computation
        expected = (a + b) * 2
        assert np.array_equal(weight.data, expected)
        
    def test_graph_memory_calculation(self):
        """Test memory calculation for graphs"""
        data = np.ones((10, 10), dtype=np.float32)
        graph = ComputationGraph(IdentityOp(data))
        
        weight = WeightTensor(computation_graph=graph)
        
        # Before materialization, should use graph memory calculation
        assert weight.nbytes == 400  # 100 * 4 bytes
        
    def test_graph_hash(self):
        """Test hashing of computation graphs"""
        data = np.ones((5, 5), dtype=np.float32)
        
        # Two graphs with same structure should have same hash
        graph1 = ComputationGraph(ScaleOp(IdentityOp(data), 2.0))
        graph2 = ComputationGraph(ScaleOp(IdentityOp(data), 2.0))
        
        weight1 = WeightTensor(computation_graph=graph1)
        weight2 = WeightTensor(computation_graph=graph2)
        
        # Graph structure hashing (not data hashing)
        hash1 = weight1.compute_hash()
        hash2 = weight2.compute_hash()
        
        # After materialization, hashes should be based on actual data
        weight1.materialize()
        weight2.materialize()
        
        assert weight1.compute_hash() == weight2.compute_hash()
        
    def test_backward_compatibility(self):
        """Test that raw data mode still works"""
        data = np.random.randn(10, 10).astype(np.float32)
        weight = WeightTensor(data=data)
        
        # Should wrap in IdentityOp internally
        assert weight._graph is not None
        assert isinstance(weight._graph.root, IdentityOp)
        
        # Should work as before
        assert np.array_equal(weight.data, data)
        
    def test_new_methods(self):
        """Test new methods added for graph support"""
        data = np.ones((5, 5), dtype=np.float32)
        scale_op = ScaleOp(IdentityOp(data), 3.0)
        graph = ComputationGraph(scale_op)
        
        weight = WeightTensor(computation_graph=graph)
        
        # Test get_computation_graph
        assert weight.get_computation_graph() is graph
        
        # Test get_operation_type
        assert weight.get_operation_type() in ["ScaleOp", "scale", "SCALE"]
        
        # Test materialize
        result = weight.materialize()
        assert np.array_equal(result, data * 3.0)
        assert weight._materialized
        
    def test_compress_with(self):
        """Test applying compression operations"""
        data = np.ones((10, 10), dtype=np.float32)
        weight = WeightTensor(data=data)
        
        # Apply a scale compression
        scale_op = ScaleOp(IdentityOp(data), 0.5)
        compressed = weight.compress_with(scale_op)
        
        # Should create new tensor with compression
        assert compressed is not weight
        assert compressed.get_operation_type() in ["ScaleOp", "scale", "SCALE"]
        assert np.array_equal(compressed.data, data * 0.5)
        
    def test_graph_serialization(self):
        """Test serialization with computation graphs"""
        data = np.ones((5, 5), dtype=np.float32)
        graph = ComputationGraph(ScaleOp(IdentityOp(data), 2.0))
        
        weight = WeightTensor(computation_graph=graph)
        weight.materialize()  # Materialize before serialization
        
        # Convert to dict
        weight_dict = weight.to_dict()
        
        # Should include graph data
        assert "computation_graph" in weight_dict
        assert weight_dict["materialized"] is True
        
    def test_matmul_graph(self):
        """Test matrix multiplication in graph"""
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        
        matmul_op = MatMulOp(IdentityOp(a), IdentityOp(b))
        graph = ComputationGraph(matmul_op)
        
        weight = WeightTensor(computation_graph=graph)
        
        expected = np.matmul(a, b)
        assert np.array_equal(weight.data, expected)
        
    def test_reshape_graph(self):
        """Test reshape operation in graph"""
        data = np.arange(12, dtype=np.float32)
        
        reshape_op = ReshapeOp(IdentityOp(data), (3, 4))
        graph = ComputationGraph(reshape_op)
        
        weight = WeightTensor(computation_graph=graph)
        
        assert weight.shape == (3, 4)
        assert np.array_equal(weight.data, data.reshape(3, 4))


class TestWeightTensorBackwardCompatibility:
    """Test that existing code continues to work without changes"""
    
    def test_existing_api_unchanged(self):
        """Test that the existing API works exactly as before"""
        # Test 1: Basic creation
        data = np.random.randn(10, 10).astype(np.float32)
        metadata = WeightMetadata(
            name="layer1.weight",
            shape=(10, 10),
            dtype=np.float32,
            layer_type="Conv2d",
            model_name="resnet50"
        )
        
        weight = WeightTensor(data=data, metadata=metadata)
        
        # All existing properties should work
        assert weight.data is not None
        assert weight.shape == (10, 10)
        assert weight.dtype == np.float32
        assert weight.size == 100
        assert weight.nbytes == 400
        assert weight.metadata.name == "layer1.weight"
        assert weight.metadata.layer_type == "Conv2d"
        assert weight.metadata.model_name == "resnet50"
        
        # Hashing should work
        hash1 = weight.compute_hash()
        assert isinstance(hash1, str)
        
        # Data setter should work
        new_data = np.random.randn(10, 10).astype(np.float32)
        weight.data = new_data
        assert np.array_equal(weight.data, new_data)
        
        # Hash should change after data update
        hash2 = weight.compute_hash()
        assert hash1 != hash2
        
    def test_legacy_store_ref(self):
        """Test store_ref functionality still works"""
        weight = WeightTensor(
            store_ref="weights/layer1/abc123",
            metadata=WeightMetadata(
                name="layer1.weight",
                shape=(100, 200),
                dtype=np.float32
            )
        )
        
        assert weight._store_ref == "weights/layer1/abc123"
        assert weight.shape == (100, 200)
        assert weight.nbytes == 80000  # 100 * 200 * 4
        
    def test_legacy_serialization(self):
        """Test that old serialization format still works"""
        data = np.ones((5, 5), dtype=np.float32)
        weight = WeightTensor(data=data)
        
        # Old style serialization
        weight_dict = weight.to_dict()
        
        # Should have all the old fields
        assert "metadata" in weight_dict
        assert "store_ref" in weight_dict
        assert "has_data" in weight_dict
        
        # Old style deserialization
        restored = WeightTensor.from_dict(weight_dict, weight_data=data)
        assert np.array_equal(restored.data, data)
        
    def test_no_graph_imports_needed(self):
        """Test that WeightTensor works without graph imports"""
        # This simulates environments where weight_ops isn't available
        import sys
        
        # Temporarily remove weight_ops if it exists
        weight_ops_module = sys.modules.get('coral.core.weight_ops')
        if weight_ops_module:
            del sys.modules['coral.core.weight_ops']
        
        try:
            # Should still be able to create and use WeightTensor
            data = np.ones((5, 5), dtype=np.float32)
            weight = WeightTensor(data=data)
            
            assert weight.shape == (5, 5)
            assert np.array_equal(weight.data, data)
            assert weight.compute_hash() is not None
        finally:
            # Restore module if it was there
            if weight_ops_module:
                sys.modules['coral.core.weight_ops'] = weight_ops_module


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
