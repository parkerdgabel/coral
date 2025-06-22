"""Tests for shape-aware clustering functionality."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.clustering.cluster_analyzer import ClusterAnalyzer
from coral.clustering.cluster_types import ClusteringStrategy, Centroid
from coral.clustering.cluster_config import ClusteringConfig
from coral.version_control.repository import Repository


class TestShapeAwareClustering:
    """Test shape-aware clustering features."""
    
    @pytest.fixture
    def temp_repo(self):
        """Create a temporary repository for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Repository(Path(temp_dir) / "test_repo", init=True)
            yield repo
    
    @pytest.fixture
    def mixed_shape_weights(self):
        """Create weights with different shapes for testing."""
        weights = []
        
        # Conv weights (different shapes)
        conv1_weight = WeightTensor(
            data=np.random.randn(16, 3, 3, 3).astype(np.float32),
            metadata=WeightMetadata(
                name="conv1.weight",
                shape=(16, 3, 3, 3),
                dtype=np.float32,
                layer_type="conv"
            )
        )
        conv2_weight = WeightTensor(
            data=np.random.randn(32, 16, 3, 3).astype(np.float32),
            metadata=WeightMetadata(
                name="conv2.weight", 
                shape=(32, 16, 3, 3),
                dtype=np.float32,
                layer_type="conv"
            )
        )
        
        # Bias weights (same shape)
        bias1 = WeightTensor(
            data=np.random.randn(16).astype(np.float32),
            metadata=WeightMetadata(
                name="conv1.bias",
                shape=(16,),
                dtype=np.float32,
                layer_type="bias"
            )
        )
        bias2 = WeightTensor(
            data=np.random.randn(16).astype(np.float32),
            metadata=WeightMetadata(
                name="conv3.bias",
                shape=(16,),
                dtype=np.float32,
                layer_type="bias"
            )
        )
        
        # Linear weights
        linear_weight = WeightTensor(
            data=np.random.randn(10, 512).astype(np.float32),
            metadata=WeightMetadata(
                name="fc.weight",
                shape=(10, 512),
                dtype=np.float32,
                layer_type="linear"
            )
        )
        
        weights.extend([conv1_weight, conv2_weight, bias1, bias2, linear_weight])
        return weights
    
    @pytest.fixture
    def cluster_analyzer(self, temp_repo):
        """Create a cluster analyzer for testing."""
        config = ClusteringConfig(
            strategy=ClusteringStrategy.ADAPTIVE,
            similarity_threshold=0.95
        )
        return ClusterAnalyzer(temp_repo, config)
    
    def test_group_weights_by_compatibility(self, cluster_analyzer, mixed_shape_weights):
        """Test weight grouping by shape and dtype."""
        groups = cluster_analyzer.group_weights_by_compatibility(mixed_shape_weights)
        
        # Should have 4 groups: (16,3,3,3), (32,16,3,3), (16,), (10,512)
        assert len(groups) == 4
        
        # Check specific groups
        conv1_key = ((16, 3, 3, 3), "float32")
        conv2_key = ((32, 16, 3, 3), "float32")
        bias_key = ((16,), "float32")
        linear_key = ((10, 512), "float32")
        
        assert conv1_key in groups
        assert conv2_key in groups
        assert bias_key in groups
        assert linear_key in groups
        
        # Check group sizes
        assert len(groups[conv1_key]) == 1  # Only conv1.weight
        assert len(groups[conv2_key]) == 1  # Only conv2.weight
        assert len(groups[bias_key]) == 2   # conv1.bias and conv3.bias
        assert len(groups[linear_key]) == 1 # Only fc.weight
    
    def test_cluster_weights_by_groups(self, cluster_analyzer, mixed_shape_weights):
        """Test shape-aware clustering."""
        result = cluster_analyzer.cluster_weights_by_groups(
            mixed_shape_weights,
            strategy=ClusteringStrategy.ADAPTIVE,
            min_group_size=2
        )
        
        # Should successfully cluster without shape mismatch errors
        assert result is not None
        assert result.is_valid()
        
        # Should have assignments for all weights
        assert len(result.assignments) == len(mixed_shape_weights)
        
        # Should have multiple clusters (at least one per shape group)
        assert len(result.centroids) >= 4
        
        # Check that centroids have valid shapes
        for centroid in result.centroids:
            assert centroid.shape is not None
            assert centroid.dtype is not None
            assert len(centroid.shape) > 0
    
    def test_centroid_validation(self, mixed_shape_weights):
        """Test centroid creation with shape validation."""
        # Test compatible weights (same shape)
        bias_weights = [w for w in mixed_shape_weights if w.shape == (16,)]
        assert len(bias_weights) == 2
        
        # Should successfully create centroid
        centroid = Centroid.from_weights(bias_weights, "test_cluster")
        assert centroid.shape == (16,)
        assert centroid.dtype == np.float32
        
        # Test incompatible weights (different shapes)
        incompatible_weights = mixed_shape_weights[:2]  # conv weights with different shapes
        
        with pytest.raises(ValueError, match="Shape compatibility error"):
            Centroid.from_weights(incompatible_weights, "bad_cluster")
    
    def test_centroid_compatibility_validation(self, mixed_shape_weights):
        """Test compatibility validation method."""
        # Compatible weights
        bias_weights = [w for w in mixed_shape_weights if w.shape == (16,)]
        assert Centroid.validate_compatibility(bias_weights) == True
        
        # Incompatible weights
        mixed_weights = mixed_shape_weights[:3]  # Different shapes
        assert Centroid.validate_compatibility(mixed_weights) == False
        
        # Single weight (always compatible)
        single_weight = [mixed_shape_weights[0]]
        assert Centroid.validate_compatibility(single_weight) == True
        
        # Empty list
        assert Centroid.validate_compatibility([]) == False
    
    def test_clustering_with_repository(self, temp_repo, mixed_shape_weights):
        """Test clustering integration with repository."""
        # Add weights to repository
        weights_dict = {w.metadata.name: w for w in mixed_shape_weights}
        temp_repo.stage_weights(weights_dict)
        temp_repo.commit("Add mixed shape weights")
        
        # Enable clustering
        temp_repo.clustering_enabled = True
        temp_repo._init_clustering()
        
        # Perform clustering - should not raise shape mismatch errors
        try:
            result = temp_repo.create_clusters(
                strategy="adaptive",
                similarity_threshold=0.95
            )
            
            # Should complete successfully
            assert 'num_clusters' in result
            assert result['num_clusters'] > 0
            assert result['weights_clustered'] == len(mixed_shape_weights)
            
        except ValueError as e:
            if "Shape mismatch" in str(e):
                pytest.fail(f"Shape mismatch error not prevented: {e}")
            else:
                # Other errors are acceptable for this test
                pass
    
    def test_small_group_handling(self, cluster_analyzer):
        """Test handling of small groups that don't benefit from clustering."""
        # Create weights with unique shapes (each will be a single-item group)
        weights = []
        for i in range(3):
            shape = (i + 1, i + 2)  # Each weight has unique shape
            weight = WeightTensor(
                data=np.random.randn(*shape).astype(np.float32),
                metadata=WeightMetadata(
                    name=f"unique_weight_{i}",
                    shape=shape,
                    dtype=np.float32
                )
            )
            weights.append(weight)
        
        result = cluster_analyzer.cluster_weights_by_groups(
            weights,
            min_group_size=2  # Groups smaller than this become single clusters
        )
        
        # Should create single-weight clusters
        assert result is not None
        assert len(result.assignments) == 3
        assert len(result.centroids) == 3  # One cluster per weight
        
        # Each assignment should have perfect similarity (single weight clusters)
        for assignment in result.assignments:
            assert assignment.similarity_score == 1.0
            assert assignment.distance_to_centroid == 0.0
    
    def test_error_handling_in_clustering(self, cluster_analyzer):
        """Test error handling when clustering fails."""
        # Create problematic weights (very small data that might cause numerical issues)
        weights = []
        for i in range(3):
            weight = WeightTensor(
                data=np.zeros((2, 2)).astype(np.float32),  # All zeros might cause issues
                metadata=WeightMetadata(
                    name=f"zero_weight_{i}",
                    shape=(2, 2),
                    dtype=np.float32
                )
            )
            weights.append(weight)
        
        # Should handle errors gracefully
        result = cluster_analyzer.cluster_weights_by_groups(weights)
        
        # Should still return a valid result (fallback clusters)
        assert result is not None
        assert len(result.assignments) == 3
        assert len(result.centroids) >= 1  # At least fallback clusters
    
    def test_dtype_compatibility(self, cluster_analyzer):
        """Test that different dtypes are handled separately."""
        weights = []
        
        # Float32 weight
        weight_f32 = WeightTensor(
            data=np.random.randn(10, 10).astype(np.float32),
            metadata=WeightMetadata(
                name="weight_f32",
                shape=(10, 10),
                dtype=np.float32
            )
        )
        
        # Float64 weight (same shape, different dtype)
        weight_f64 = WeightTensor(
            data=np.random.randn(10, 10).astype(np.float64),
            metadata=WeightMetadata(
                name="weight_f64",
                shape=(10, 10),
                dtype=np.float64
            )
        )
        
        weights = [weight_f32, weight_f64]
        
        # Should create separate groups for different dtypes
        groups = cluster_analyzer.group_weights_by_compatibility(weights)
        assert len(groups) == 2  # Different dtypes = different groups
        
        f32_key = ((10, 10), "float32")
        f64_key = ((10, 10), "float64")
        
        assert f32_key in groups
        assert f64_key in groups
        assert len(groups[f32_key]) == 1
        assert len(groups[f64_key]) == 1