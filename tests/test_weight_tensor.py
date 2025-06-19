"""Tests for WeightTensor class"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from coral.core.weight_tensor import WeightMetadata, WeightTensor


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
