"""
Test numerical stability fixes for clustering system.

This test file verifies that the numerical stability improvements handle:
1. Overflow in dot product calculations
2. Division by zero in similarity computations
3. NaN/Inf value propagation
4. Extreme value handling
"""

import numpy as np
import pytest
import warnings

from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.clustering import ClusterAnalyzer, ClusteringConfig, ClusteringStrategy
from coral.clustering.cluster_analyzer import safe_cosine_similarity, safe_normalize
from coral.version_control.repository import Repository


class TestNumericalStability:
    """Test numerical stability improvements in clustering system."""
    
    def create_weight(self, data: np.ndarray, name: str = "test") -> WeightTensor:
        """Helper to create weight tensor."""
        metadata = WeightMetadata(
            name=name,
            shape=data.shape,
            dtype=data.dtype
        )
        return WeightTensor(data=data, metadata=metadata)
    
    def test_weight_tensor_similarity_overflow(self):
        """Test that weight tensor similarity handles overflow correctly."""
        # Create weights with very large values
        large_val = np.finfo(np.float32).max * 0.9
        weight1 = self.create_weight(np.full((10,), large_val, dtype=np.float32))
        weight2 = self.create_weight(np.full((10,), large_val * 0.99, dtype=np.float32))
        
        # This should not overflow
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Convert warnings to errors
            similarity = weight1.is_similar_to(weight2, threshold=0.95)
            assert isinstance(similarity, bool)
    
    def test_weight_tensor_similarity_zero_vectors(self):
        """Test that zero vector similarity is handled correctly."""
        # Create zero vectors
        weight1 = self.create_weight(np.zeros((10,), dtype=np.float32))
        weight2 = self.create_weight(np.zeros((10,), dtype=np.float32))
        
        # Zero vectors should be similar
        assert weight1.is_similar_to(weight2)
        
        # Zero vs non-zero should not be similar
        weight3 = self.create_weight(np.ones((10,), dtype=np.float32))
        assert not weight1.is_similar_to(weight3)
    
    def test_weight_tensor_similarity_nan_values(self):
        """Test handling of NaN values in similarity computation."""
        # Create weights with NaN
        weight1 = self.create_weight(np.array([1.0, np.nan, 3.0], dtype=np.float32))
        weight2 = self.create_weight(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        
        # NaN weights should not be similar to anything
        assert not weight1.is_similar_to(weight2)
        assert not weight1.is_similar_to(weight1)  # Not even to itself
    
    def test_weight_tensor_similarity_inf_values(self):
        """Test handling of Inf values in similarity computation."""
        # Create weights with Inf
        weight1 = self.create_weight(np.array([1.0, np.inf, 3.0], dtype=np.float32))
        weight2 = self.create_weight(np.array([1.0, np.inf, 3.0], dtype=np.float32))
        weight3 = self.create_weight(np.array([1.0, -np.inf, 3.0], dtype=np.float32))
        
        # Identical Inf weights should be similar
        assert weight1.is_similar_to(weight2)
        
        # Different Inf values should not be similar
        assert not weight1.is_similar_to(weight3)
    
    def test_safe_cosine_similarity_extreme_values(self):
        """Test safe cosine similarity function with extreme values."""
        # Very large values
        large_val = 1e30
        a = np.array([large_val, large_val * 2])
        b = np.array([large_val * 0.9, large_val * 2.1])
        
        similarity = safe_cosine_similarity(a, b)
        assert -1 <= similarity <= 1
        assert not np.isnan(similarity)
        assert not np.isinf(similarity)
        
        # Very small values
        tiny_val = 1e-30
        a = np.array([tiny_val, tiny_val * 2])
        b = np.array([tiny_val * 0.9, tiny_val * 2.1])
        
        similarity = safe_cosine_similarity(a, b)
        assert -1 <= similarity <= 1
        assert not np.isnan(similarity)
    
    def test_safe_cosine_similarity_special_cases(self):
        """Test safe cosine similarity with special cases."""
        # NaN values
        a = np.array([1.0, np.nan, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert safe_cosine_similarity(a, b) == 0.0
        
        # Inf values
        a = np.array([1.0, np.inf, 3.0])
        b = np.array([1.0, np.inf, 3.0])
        assert safe_cosine_similarity(a, b) == 1.0
        
        # Zero vectors
        a = np.zeros(10)
        b = np.zeros(10)
        assert safe_cosine_similarity(a, b) == 1.0
        
        # Zero vs non-zero
        a = np.zeros(10)
        b = np.ones(10)
        assert safe_cosine_similarity(a, b) == 0.0
    
    def test_safe_normalize_function(self):
        """Test safe normalization function."""
        # Normal case
        data = np.array([3.0, 4.0])
        normalized = safe_normalize(data)
        assert np.allclose(np.linalg.norm(normalized), 1.0)
        
        # Zero vector
        data = np.zeros(5)
        normalized = safe_normalize(data)
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))
        
        # NaN values
        data = np.array([1.0, np.nan, 3.0])
        normalized = safe_normalize(data)
        assert not np.any(np.isnan(normalized))
        
        # Inf values
        data = np.array([1.0, np.inf, 3.0])
        normalized = safe_normalize(data)
        assert not np.any(np.isinf(normalized))
    
    def test_clustering_with_extreme_weights(self, tmp_path):
        """Test clustering with extreme weight values."""
        repo = Repository(tmp_path, init=True)
        
        # Add weights with extreme values
        weights = {
            "large": self.create_weight(np.full((32, 32), 1e20, dtype=np.float32)),
            "small": self.create_weight(np.full((32, 32), 1e-20, dtype=np.float32)),
            "normal": self.create_weight(np.random.randn(32, 32).astype(np.float32)),
        }
        
        repo.stage_weights(weights)
        repo.commit("Add extreme weights")
        
        # Create analyzer
        config = ClusteringConfig(
            strategy=ClusteringStrategy.KMEANS,
            similarity_threshold=0.95
        )
        analyzer = ClusterAnalyzer(repo, config)
        
        # This should not raise warnings or errors
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                analysis = analyzer.analyze_repository()
                assert analysis.total_weights == 3
                assert analysis.unique_weights <= 3
            except RuntimeWarning as e:
                pytest.fail(f"Numerical warning raised: {e}")
    
    def test_clustering_with_constant_weights(self, tmp_path):
        """Test clustering with constant value weights."""
        repo = Repository(tmp_path, init=True)
        
        # Add constant weights
        weights = {
            "zeros": self.create_weight(np.zeros((16, 16), dtype=np.float32)),
            "ones": self.create_weight(np.ones((16, 16), dtype=np.float32)),
            "constant": self.create_weight(np.full((16, 16), 42.0, dtype=np.float32)),
        }
        
        repo.stage_weights(weights)
        repo.commit("Add constant weights")
        
        # Create analyzer
        config = ClusteringConfig(
            strategy=ClusteringStrategy.ADAPTIVE,
            similarity_threshold=0.99
        )
        analyzer = ClusterAnalyzer(repo, config)
        
        # Should handle constant weights without errors
        analysis = analyzer.analyze_repository()
        assert analysis.total_weights == 3
        
        # Extract features should work
        weights_list = list(weights.values())
        features = analyzer.extract_features(weights_list, method="statistical")
        assert features.shape[0] == 3
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
    
    def test_similarity_matrix_numerical_stability(self, tmp_path):
        """Test similarity matrix computation with edge cases."""
        repo = Repository(tmp_path, init=True)
        
        # Create weights that might cause numerical issues
        weights_list = [
            self.create_weight(np.zeros((10,), dtype=np.float32), "zeros"),
            self.create_weight(np.ones((10,), dtype=np.float32), "ones"),
            self.create_weight(np.full((10,), 1e10, dtype=np.float32), "large"),
            self.create_weight(np.full((10,), 1e-10, dtype=np.float32), "tiny"),
        ]
        
        config = ClusteringConfig()
        analyzer = ClusterAnalyzer(repo, config)
        
        # Compute similarity matrix
        similarity_matrix = analyzer.compute_similarity_matrix(weights_list, metric="cosine")
        
        # Check properties
        assert similarity_matrix.shape == (4, 4)
        assert np.all(np.diag(similarity_matrix) == 1.0)  # Self-similarity
        assert np.all(similarity_matrix >= -1.0)
        assert np.all(similarity_matrix <= 1.0)
        assert not np.any(np.isnan(similarity_matrix))
        assert not np.any(np.isinf(similarity_matrix))
        
        # Test with euclidean metric
        similarity_matrix = analyzer.compute_similarity_matrix(weights_list, metric="euclidean")
        assert np.all(similarity_matrix >= 0.0)
        assert np.all(similarity_matrix <= 1.0)
        assert not np.any(np.isnan(similarity_matrix))
        
        # Test with correlation metric
        similarity_matrix = analyzer.compute_similarity_matrix(weights_list, metric="correlation")
        assert not np.any(np.isnan(similarity_matrix))
        assert not np.any(np.isinf(similarity_matrix))
    
    def test_clustering_algorithms_with_edge_cases(self, tmp_path):
        """Test all clustering algorithms with numerical edge cases."""
        repo = Repository(tmp_path, init=True)
        
        # Add a variety of edge case weights
        weights = {}
        for i in range(5):
            if i == 0:
                data = np.zeros((8, 8), dtype=np.float32)
            elif i == 1:
                data = np.ones((8, 8), dtype=np.float32) * 1e10
            elif i == 2:
                data = np.full((8, 8), 1e-10, dtype=np.float32)
            else:
                data = np.random.randn(8, 8).astype(np.float32)
            
            weights[f"weight_{i}"] = self.create_weight(data, f"weight_{i}")
        
        repo.stage_weights(weights)
        repo.commit("Add edge case weights")
        
        weights_list = list(weights.values())
        
        # Test each clustering strategy
        for strategy in [ClusteringStrategy.KMEANS, ClusteringStrategy.HIERARCHICAL, 
                        ClusteringStrategy.DBSCAN]:
            config = ClusteringConfig(strategy=strategy)
            analyzer = ClusterAnalyzer(repo, config)
            
            try:
                if strategy == ClusteringStrategy.KMEANS:
                    result = analyzer.cluster_kmeans(weights_list, k=2)
                elif strategy == ClusteringStrategy.HIERARCHICAL:
                    result = analyzer.cluster_hierarchical(weights_list)
                elif strategy == ClusteringStrategy.DBSCAN:
                    result = analyzer.cluster_dbscan(weights_list)
                
                # Verify result validity
                assert result is not None
                assert len(result.assignments) == len(weights_list)
                assert result.metrics is not None
                
                # Check for numerical issues in metrics
                if result.metrics.silhouette_score is not None:
                    assert not np.isnan(result.metrics.silhouette_score)
                    assert -1 <= result.metrics.silhouette_score <= 1
                
            except Exception as e:
                pytest.fail(f"Strategy {strategy} failed with edge cases: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])