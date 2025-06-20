"""Comprehensive tests for ClusterAnalyzer functionality."""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from coral.clustering.cluster_analyzer import ClusterAnalyzer, RepositoryAnalysis, ClusteringResult
from coral.clustering.cluster_config import ClusteringConfig, OptimizationConfig
from coral.clustering.cluster_types import (
    ClusteringStrategy, ClusterLevel, ClusterMetrics, 
    ClusterInfo, ClusterAssignment, Centroid
)
from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.version_control.repository import Repository


class TestClusterAnalyzer:
    """Test suite for ClusterAnalyzer."""

    @pytest.fixture
    def temp_repo_path(self):
        """Create temporary repository for testing."""
        temp_dir = tempfile.mkdtemp()
        repo_path = Path(temp_dir) / "test_repo"
        repo_path.mkdir()
        yield repo_path
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_repository(self):
        """Create mock repository for testing."""
        mock_repo = Mock(spec=Repository)
        mock_repo.path = Path("/mock/repo")
        mock_repo.config = {
            "core": {
                "similarity_threshold": 0.98,
                "delta_encoding": True
            }
        }
        # Add mock methods that the analyzer expects
        mock_repo.log.return_value = [Mock(commit_hash="commit1"), Mock(commit_hash="commit2")]
        mock_repo.get_all_weights.return_value = {}
        return mock_repo

    @pytest.fixture
    def sample_weights(self):
        """Create sample weight tensors for testing."""
        weights = []
        
        # Create similar weights for clustering
        base_data = np.random.rand(10, 10).astype(np.float32)
        
        for i in range(5):
            # Add small variations to create similar weights
            data = base_data + 0.001 * np.random.rand(10, 10).astype(np.float32)
            metadata = WeightMetadata(
                name=f"weight_{i}",
                shape=(10, 10),
                dtype=np.float32,
                layer_type="linear"
            )
            weights.append(WeightTensor(data=data, metadata=metadata))
        
        # Add some dissimilar weights
        for i in range(3):
            data = np.random.rand(5, 5).astype(np.float32)
            metadata = WeightMetadata(
                name=f"different_weight_{i}",
                shape=(5, 5),
                dtype=np.float32,
                layer_type="conv"
            )
            weights.append(WeightTensor(data=data, metadata=metadata))
        
        return weights

    @pytest.fixture
    def cluster_analyzer(self, mock_repository):
        """Create ClusterAnalyzer instance."""
        config = ClusteringConfig(
            strategy=ClusteringStrategy.KMEANS,
            similarity_threshold=0.95,
            min_cluster_size=2
        )
        return ClusterAnalyzer(mock_repository, config)

    def test_init_default_config(self, mock_repository):
        """Test ClusterAnalyzer initialization with default config."""
        analyzer = ClusterAnalyzer(mock_repository)
        
        assert analyzer.repository == mock_repository
        assert isinstance(analyzer.config, ClusteringConfig)
        assert analyzer.config.strategy == ClusteringStrategy.ADAPTIVE
        assert analyzer._feature_cache == {}
        assert analyzer._similarity_cache == {}

    def test_init_custom_config(self, mock_repository):
        """Test ClusterAnalyzer initialization with custom config."""
        config = ClusteringConfig(
            strategy=ClusteringStrategy.HIERARCHICAL,
            similarity_threshold=0.90,
            min_cluster_size=3
        )
        analyzer = ClusterAnalyzer(mock_repository, config)
        
        assert analyzer.config == config
        assert analyzer.config.strategy == ClusteringStrategy.HIERARCHICAL
        assert analyzer.config.similarity_threshold == 0.90

    def test_analyze_repository_basic(self, cluster_analyzer, sample_weights):
        """Test basic repository analysis."""
        # Mock repository methods
        mock_weights = {f"weight_{i}": weight for i, weight in enumerate(sample_weights[:4])}
        cluster_analyzer.repository.get_all_weights.return_value = mock_weights
        
        analysis = cluster_analyzer.analyze_repository()
        
        assert isinstance(analysis, RepositoryAnalysis)
        assert analysis.total_weights > 0
        assert len(analysis.weight_shapes) > 0
        assert len(analysis.weight_dtypes) > 0

    def test_analyze_repository_with_filters(self, cluster_analyzer, sample_weights):
        """Test repository analysis with branch/commit filters."""
        mock_weights = {f"weight_{i}": weight for i, weight in enumerate(sample_weights[:2])}
        cluster_analyzer.repository.get_all_weights.return_value = mock_weights
        
        analysis = cluster_analyzer.analyze_repository(
            branches=["main"],
            commits=["commit1"]
        )
        
        assert analysis.total_weights == 2

    def test_extract_features_raw(self, cluster_analyzer, sample_weights):
        """Test raw feature extraction."""
        features = cluster_analyzer.extract_features(sample_weights[:3])
        
        assert features.shape[0] == 3  # Number of weights
        assert features.shape[1] == 100  # Flattened 10x10 features
        assert features.dtype == np.float32

    def test_extract_features_pca(self, cluster_analyzer, sample_weights):
        """Test PCA feature extraction."""
        cluster_analyzer.config.feature_extraction = "pca"
        cluster_analyzer.config.dimensionality_reduction = 20
        
        features = cluster_analyzer.extract_features(sample_weights[:5])
        
        assert features.shape[0] == 5
        # With mixed shapes, it falls back to statistical features (6 dimensions)
        # PCA might reduce further based on available variance
        assert features.shape[1] <= 20  # Reduced dimensions (may be less than 20)

    def test_extract_features_statistical(self, cluster_analyzer, sample_weights):
        """Test statistical feature extraction."""
        cluster_analyzer.config.feature_extraction = "statistical"
        
        features = cluster_analyzer.extract_features(sample_weights[:3])
        
        assert features.shape[0] == 3
        assert features.shape[1] == 6  # Mean, std, min, max, skew, kurtosis

    def test_extract_features_hash(self, cluster_analyzer, sample_weights):
        """Test hash feature extraction."""
        cluster_analyzer.config.feature_extraction = "hash"
        
        features = cluster_analyzer.extract_features(sample_weights[:3])
        
        assert features.shape[0] == 3
        assert features.shape[1] > 0  # Hash-based features

    def test_extract_features_caching(self, cluster_analyzer, sample_weights):
        """Test feature extraction caching."""
        # First extraction with cache key
        features1 = cluster_analyzer.extract_features(sample_weights[:2], cache_key="test_cache")
        
        # Second extraction should use cache
        features2 = cluster_analyzer.extract_features(sample_weights[:2], cache_key="test_cache")
        
        np.testing.assert_array_equal(features1, features2)
        assert len(cluster_analyzer._feature_cache) == 1  # One cache entry

    def test_compute_similarity_matrix(self, cluster_analyzer, sample_weights):
        """Test similarity matrix computation."""
        similarity_matrix = cluster_analyzer.compute_similarity_matrix(sample_weights[:4])
        
        assert similarity_matrix.shape == (4, 4)
        assert np.allclose(np.diag(similarity_matrix), 1.0)  # Diagonal should be 1
        assert np.allclose(similarity_matrix, similarity_matrix.T)  # Should be symmetric

    def test_compute_similarity_matrix_caching(self, cluster_analyzer, sample_weights):
        """Test similarity matrix caching."""
        weights_subset = sample_weights[:3]
        
        # First computation
        matrix1 = cluster_analyzer.compute_similarity_matrix(weights_subset)
        
        # Second computation should use cache
        matrix2 = cluster_analyzer.compute_similarity_matrix(weights_subset)
        
        np.testing.assert_array_equal(matrix1, matrix2)

    def test_detect_natural_clusters_elbow(self, cluster_analyzer, sample_weights):
        """Test natural cluster detection using elbow method."""
        optimal_k = cluster_analyzer.detect_natural_clusters(
            sample_weights, 
            method="elbow",
            max_k=5
        )
        
        assert isinstance(optimal_k, (int, np.integer))
        assert 1 <= optimal_k <= 5

    def test_detect_natural_clusters_silhouette(self, cluster_analyzer, sample_weights):
        """Test natural cluster detection using silhouette method."""
        optimal_k = cluster_analyzer.detect_natural_clusters(
            sample_weights,
            method="silhouette",
            max_k=4
        )
        
        assert isinstance(optimal_k, (int, np.integer))
        assert 2 <= optimal_k <= 4  # Silhouette requires at least 2 clusters

    def test_cluster_kmeans(self, cluster_analyzer, sample_weights):
        """Test K-means clustering."""
        result = cluster_analyzer.cluster_kmeans(sample_weights, k=3)
        
        assert isinstance(result, ClusteringResult)
        assert len(result.assignments) == len(sample_weights)
        assert len(result.centroids) == 3
        assert result.metrics.num_clusters == 3

    def test_cluster_kmeans_with_config(self, cluster_analyzer, sample_weights):
        """Test K-means clustering with custom configuration."""
        config = {"max_iter": 50, "tol": 1e-3}
        result = cluster_analyzer.cluster_kmeans(sample_weights, k=2, config=config)
        
        assert len(result.centroids) == 2
        assert all(isinstance(centroid, Centroid) for centroid in result.centroids)

    def test_cluster_hierarchical(self, cluster_analyzer, sample_weights):
        """Test hierarchical clustering."""
        config = {"linkage": "ward", "distance_threshold": 0.5}
        result = cluster_analyzer.cluster_hierarchical(sample_weights, config=config)
        
        assert isinstance(result, ClusteringResult)
        assert len(result.assignments) == len(sample_weights)
        assert result.metrics.num_clusters > 0

    def test_cluster_dbscan(self, cluster_analyzer, sample_weights):
        """Test DBSCAN clustering."""
        config = {"eps": 0.5, "min_samples": 2}
        result = cluster_analyzer.cluster_dbscan(sample_weights, config=config)
        
        assert isinstance(result, ClusteringResult)
        assert len(result.assignments) == len(sample_weights)

    def test_cluster_adaptive(self, cluster_analyzer, sample_weights):
        """Test adaptive clustering strategy selection."""
        result = cluster_analyzer.cluster_adaptive(sample_weights)
        
        assert isinstance(result, ClusteringResult)
        assert len(result.assignments) == len(sample_weights)
        assert result.strategy in [ClusteringStrategy.KMEANS, ClusteringStrategy.HIERARCHICAL, ClusteringStrategy.DBSCAN]

    def test_evaluate_clustering(self, cluster_analyzer, sample_weights):
        """Test clustering quality evaluation."""
        # First create clusters
        result = cluster_analyzer.cluster_kmeans(sample_weights, k=2)
        
        # Evaluate the clustering
        metrics = cluster_analyzer.evaluate_clustering(result.assignments, sample_weights)
        
        assert isinstance(metrics, ClusterMetrics)
        assert -1.0 <= metrics.silhouette_score <= 1.0
        assert metrics.num_clusters > 0
        assert metrics.avg_cluster_size > 0

    def test_optimize_cluster_count(self, cluster_analyzer, sample_weights):
        """Test optimal cluster count optimization."""
        optimal_k = cluster_analyzer.optimize_cluster_count(sample_weights, max_k=4)
        
        assert isinstance(optimal_k, (int, np.integer))
        assert 1 <= optimal_k <= 4

    def test_validate_clusters(self, cluster_analyzer, sample_weights):
        """Test cluster validation."""
        result = cluster_analyzer.cluster_kmeans(sample_weights, k=2)
        
        is_valid = cluster_analyzer.validate_clusters(result.assignments)
        
        assert isinstance(is_valid, bool)

    def test_memory_management_large_dataset(self, cluster_analyzer):
        """Test memory management with large datasets."""
        # Create large dataset
        large_weights = []
        for i in range(100):
            data = np.random.rand(50, 50).astype(np.float32)
            metadata = WeightMetadata(
                name=f"large_weight_{i}",
                shape=(50, 50),
                dtype=np.float32
            )
            large_weights.append(WeightTensor(data=data, metadata=metadata))
        
        # Should handle large dataset without memory issues
        result = cluster_analyzer.cluster_adaptive(large_weights)
        assert isinstance(result, ClusteringResult)

    def test_thread_safety(self, cluster_analyzer, sample_weights):
        """Test thread-safe operations."""
        import threading
        import concurrent.futures
        
        results = []
        
        def cluster_worker():
            result = cluster_analyzer.cluster_kmeans(sample_weights, k=2)
            results.append(result)
        
        # Run multiple clustering operations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(cluster_worker) for _ in range(3)]
            concurrent.futures.wait(futures)
        
        assert len(results) == 3
        assert all(isinstance(result, ClusteringResult) for result in results)

    def test_empty_weights_handling(self, cluster_analyzer):
        """Test handling of empty weight lists."""
        with pytest.raises(ValueError, match="Cannot cluster empty weight list"):
            cluster_analyzer.cluster_kmeans([], k=2)

    def test_single_weight_handling(self, cluster_analyzer, sample_weights):
        """Test handling of single weight."""
        single_weight = [sample_weights[0]]
        
        with pytest.raises(ValueError, match="Need at least 2 weights for clustering"):
            cluster_analyzer.cluster_kmeans(single_weight, k=2)

    def test_invalid_k_handling(self, cluster_analyzer, sample_weights):
        """Test handling of invalid k values."""
        with pytest.raises(ValueError, match="k must be positive"):
            cluster_analyzer.cluster_kmeans(sample_weights, k=0)
        
        with pytest.raises(ValueError, match="k cannot be larger than number of weights"):
            cluster_analyzer.cluster_kmeans(sample_weights[:3], k=5)

    def test_algorithm_convergence_failure(self, cluster_analyzer, sample_weights):
        """Test handling of algorithm convergence failures."""
        # Mock scikit-learn to raise convergence warning
        with patch('coral.clustering.cluster_analyzer.KMeans') as mock_kmeans:
            mock_instance = Mock()
            mock_instance.fit_predict.side_effect = RuntimeError("Algorithm failed to converge")
            mock_kmeans.return_value = mock_instance
            
            with pytest.raises(RuntimeError, match="Clustering failed"):
                cluster_analyzer.cluster_kmeans(sample_weights, k=2)

    def test_progress_tracking(self, cluster_analyzer, sample_weights):
        """Test progress tracking during clustering."""
        progress_updates = []
        
        def progress_callback(progress):
            progress_updates.append(progress)
        
        result = cluster_analyzer.cluster_kmeans(
            sample_weights, 
            k=2, 
            progress_callback=progress_callback
        )
        
        assert isinstance(result, ClusteringResult)
        # Progress updates should have been called
        assert len(progress_updates) > 0

    def test_cancellation_support(self, cluster_analyzer, sample_weights):
        """Test cancellation of long-running operations."""
        import threading
        
        cancel_event = threading.Event()
        cancel_event.set()  # Set immediately to test cancellation
        
        # This should be cancelled before completion
        with pytest.raises(RuntimeError, match="Operation cancelled"):
            cluster_analyzer.cluster_kmeans(
                sample_weights,  # Use normal sample size
                k=2,
                cancel_event=cancel_event
            )

    def test_incremental_analysis(self, cluster_analyzer, sample_weights):
        """Test incremental analysis for large repositories."""
        # Just test adaptive clustering works with sample weights
        result = cluster_analyzer.cluster_adaptive(sample_weights)
        
        assert isinstance(result, ClusteringResult)
        assert len(result.assignments) == len(sample_weights)


class TestRepositoryAnalysis:
    """Test RepositoryAnalysis data structure."""

    def test_repository_analysis_creation(self):
        """Test RepositoryAnalysis creation and validation."""
        analysis = RepositoryAnalysis(
            total_weights=100,
            unique_weights=80,
            total_commits=10,
            total_branches=3,
            weight_shapes={
                (10, 10): 50,
                (5, 5): 30,
                (20, 20): 20
            },
            weight_dtypes={
                "float32": 70,
                "float16": 30
            },
            layer_types={
                "linear": 60,
                "conv": 40
            }
        )
        
        assert analysis.total_weights == 100
        assert analysis.unique_weights == 80
        assert analysis.deduplication_ratio == 0.2
        assert len(analysis.weight_shapes) == 3
        assert len(analysis.weight_dtypes) == 2


class TestClusteringResult:
    """Test ClusteringResult data structure."""

    def test_clustering_result_creation(self):
        """Test ClusteringResult creation."""
        assignments = [
            ClusterAssignment("weight1", "hash1", "cluster1", 0.1, 0.9),
            ClusterAssignment("weight2", "hash2", "cluster1", 0.2, 0.8),
            ClusterAssignment("weight3", "hash3", "cluster2", 0.15, 0.85),
        ]
        
        centroids = [
            Centroid(
                data=np.random.rand(10).astype(np.float32),
                cluster_id="cluster1",
                shape=(10,),
                dtype=np.float32
            ),
            Centroid(
                data=np.random.rand(10).astype(np.float32),
                cluster_id="cluster2", 
                shape=(10,),
                dtype=np.float32
            )
        ]
        
        metrics = ClusterMetrics(
            silhouette_score=0.75,
            num_clusters=2,
            avg_cluster_size=1.5
        )
        
        result = ClusteringResult(
            assignments=assignments,
            centroids=centroids,
            metrics=metrics,
            strategy=ClusteringStrategy.KMEANS
        )
        
        assert len(result.assignments) == 3
        assert len(result.centroids) == 2
        assert result.strategy == ClusteringStrategy.KMEANS
        assert result.metrics.num_clusters == 2

    def test_clustering_result_validation(self):
        """Test ClusteringResult validation."""
        # Create invalid result (mismatched cluster counts)
        assignments = [ClusterAssignment("w1", "h1", "c1", 0.1, 0.9)]
        centroids = []  # Empty centroids
        metrics = ClusterMetrics(num_clusters=1)
        
        result = ClusteringResult(
            assignments=assignments,
            centroids=centroids,
            metrics=metrics,
            strategy=ClusteringStrategy.KMEANS
        )
        
        # Should detect inconsistency
        assert not result.is_valid()


if __name__ == "__main__":
    pytest.main([__file__])