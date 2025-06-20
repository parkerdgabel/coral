"""Tests for clustering types and configuration."""

import numpy as np
import pytest

from coral.clustering.cluster_types import (
    Centroid,
    ClusterAssignment,
    ClusterInfo,
    ClusterLevel,
    ClusterMetrics,
    ClusteringStrategy,
)
from coral.clustering.cluster_config import (
    ClusteringConfig,
    HierarchyConfig,
    OptimizationConfig,
)
from coral.core.weight_tensor import WeightMetadata, WeightTensor


class TestClusteringStrategy:
    """Test ClusteringStrategy enum."""

    def test_clustering_strategy_values(self):
        """Test that all clustering strategies have correct values."""
        assert ClusteringStrategy.KMEANS.value == "kmeans"
        assert ClusteringStrategy.HIERARCHICAL.value == "hierarchical"
        assert ClusteringStrategy.DBSCAN.value == "dbscan"
        assert ClusteringStrategy.ADAPTIVE.value == "adaptive"

    def test_clustering_strategy_from_string(self):
        """Test creating ClusteringStrategy from string values."""
        assert ClusteringStrategy("kmeans") == ClusteringStrategy.KMEANS
        assert ClusteringStrategy("hierarchical") == ClusteringStrategy.HIERARCHICAL
        assert ClusteringStrategy("dbscan") == ClusteringStrategy.DBSCAN
        assert ClusteringStrategy("adaptive") == ClusteringStrategy.ADAPTIVE

    def test_invalid_clustering_strategy(self):
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError):
            ClusteringStrategy("invalid_strategy")


class TestClusterLevel:
    """Test ClusterLevel enum."""

    def test_cluster_level_values(self):
        """Test that all cluster levels have correct values."""
        assert ClusterLevel.TENSOR.value == "tensor"
        assert ClusterLevel.BLOCK.value == "block"
        assert ClusterLevel.LAYER.value == "layer"
        assert ClusterLevel.MODEL.value == "model"

    def test_cluster_level_ordering(self):
        """Test that cluster levels have correct ordering."""
        # The enum should be ordered from most granular to least granular
        levels = list(ClusterLevel)
        assert levels[0] == ClusterLevel.TENSOR
        assert levels[1] == ClusterLevel.BLOCK
        assert levels[2] == ClusterLevel.LAYER
        assert levels[3] == ClusterLevel.MODEL


class TestClusterMetrics:
    """Test ClusterMetrics dataclass."""

    def test_cluster_metrics_creation(self):
        """Test creating ClusterMetrics with valid values."""
        metrics = ClusterMetrics(
            silhouette_score=0.8,
            inertia=100.5,
            calinski_harabasz_score=50.2,
            davies_bouldin_score=0.3,
            num_clusters=5,
            avg_cluster_size=20.4,
            compression_ratio=0.85,
        )

        assert metrics.silhouette_score == 0.8
        assert metrics.inertia == 100.5
        assert metrics.calinski_harabasz_score == 50.2
        assert metrics.davies_bouldin_score == 0.3
        assert metrics.num_clusters == 5
        assert metrics.avg_cluster_size == 20.4
        assert metrics.compression_ratio == 0.85

    def test_cluster_metrics_defaults(self):
        """Test ClusterMetrics with default values."""
        metrics = ClusterMetrics()

        assert metrics.silhouette_score == 0.0
        assert metrics.inertia == 0.0
        assert metrics.calinski_harabasz_score == 0.0
        assert metrics.davies_bouldin_score == 0.0
        assert metrics.num_clusters == 0
        assert metrics.avg_cluster_size == 0.0
        assert metrics.compression_ratio == 0.0

    def test_cluster_metrics_validation(self):
        """Test ClusterMetrics validation."""
        # Valid ranges
        metrics = ClusterMetrics(
            silhouette_score=0.5,  # [-1, 1]
            compression_ratio=0.8,  # [0, 1]
            num_clusters=3,  # >= 0
        )
        assert metrics.is_valid()

        # Invalid silhouette score
        invalid_metrics = ClusterMetrics(silhouette_score=2.0)
        assert not invalid_metrics.is_valid()

        # Invalid compression ratio
        invalid_metrics = ClusterMetrics(compression_ratio=1.5)
        assert not invalid_metrics.is_valid()

        # Invalid number of clusters
        invalid_metrics = ClusterMetrics(num_clusters=-1)
        assert not invalid_metrics.is_valid()

    def test_cluster_metrics_serialization(self):
        """Test ClusterMetrics serialization."""
        metrics = ClusterMetrics(
            silhouette_score=0.7,
            inertia=50.0,
            num_clusters=3,
            compression_ratio=0.9,
        )

        # Test to_dict
        metrics_dict = metrics.to_dict()
        expected_keys = {
            "silhouette_score",
            "inertia",
            "calinski_harabasz_score",
            "davies_bouldin_score",
            "num_clusters",
            "avg_cluster_size",
            "compression_ratio",
        }
        assert set(metrics_dict.keys()) == expected_keys
        assert metrics_dict["silhouette_score"] == 0.7
        assert metrics_dict["num_clusters"] == 3

        # Test from_dict
        restored_metrics = ClusterMetrics.from_dict(metrics_dict)
        assert restored_metrics.silhouette_score == 0.7
        assert restored_metrics.inertia == 50.0
        assert restored_metrics.num_clusters == 3
        assert restored_metrics.compression_ratio == 0.9


class TestClusterInfo:
    """Test ClusterInfo dataclass."""

    def test_cluster_info_creation(self):
        """Test creating ClusterInfo."""
        info = ClusterInfo(
            cluster_id="cluster_001",
            strategy=ClusteringStrategy.KMEANS,
            level=ClusterLevel.TENSOR,
            member_count=15,
            centroid_hash="abc123",
            created_at="2024-01-01T00:00:00Z",
        )

        assert info.cluster_id == "cluster_001"
        assert info.strategy == ClusteringStrategy.KMEANS
        assert info.level == ClusterLevel.TENSOR
        assert info.member_count == 15
        assert info.centroid_hash == "abc123"
        assert info.created_at == "2024-01-01T00:00:00Z"

    def test_cluster_info_defaults(self):
        """Test ClusterInfo with minimal required fields."""
        info = ClusterInfo(
            cluster_id="test_cluster", strategy=ClusteringStrategy.KMEANS
        )

        assert info.cluster_id == "test_cluster"
        assert info.strategy == ClusteringStrategy.KMEANS
        assert info.level == ClusterLevel.TENSOR  # default
        assert info.member_count == 0  # default
        assert info.centroid_hash is None  # default
        assert info.created_at is None  # default

    def test_cluster_info_serialization(self):
        """Test ClusterInfo serialization."""
        info = ClusterInfo(
            cluster_id="test_cluster",
            strategy=ClusteringStrategy.HIERARCHICAL,
            level=ClusterLevel.BLOCK,
            member_count=8,
            centroid_hash="xyz789",
        )

        # Test to_dict
        info_dict = info.to_dict()
        assert info_dict["cluster_id"] == "test_cluster"
        assert info_dict["strategy"] == "hierarchical"
        assert info_dict["level"] == "block"
        assert info_dict["member_count"] == 8

        # Test from_dict
        restored_info = ClusterInfo.from_dict(info_dict)
        assert restored_info.cluster_id == "test_cluster"
        assert restored_info.strategy == ClusteringStrategy.HIERARCHICAL
        assert restored_info.level == ClusterLevel.BLOCK
        assert restored_info.member_count == 8


class TestCentroid:
    """Test Centroid dataclass."""

    def test_centroid_creation(self):
        """Test creating Centroid."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        centroid = Centroid(
            data=data,
            cluster_id="centroid_001",
            shape=(3,),
            dtype=np.float32,
        )

        assert np.array_equal(centroid.data, data)
        assert centroid.cluster_id == "centroid_001"
        assert centroid.shape == (3,)
        assert centroid.dtype == np.float32

    def test_centroid_hash_computation(self):
        """Test centroid hash computation."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        centroid = Centroid(
            data=data,
            cluster_id="test",
            shape=(3,),
            dtype=np.float32,
        )

        hash1 = centroid.compute_hash()
        hash2 = centroid.compute_hash()

        # Hash should be consistent
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0

        # Different data should produce different hash
        different_data = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        different_centroid = Centroid(
            data=different_data,
            cluster_id="test2",
            shape=(3,),
            dtype=np.float32,
        )

        assert centroid.compute_hash() != different_centroid.compute_hash()

    def test_centroid_from_weights(self):
        """Test creating centroid from weight list."""
        # Create test weights
        weights = []
        for i in range(3):
            data = np.ones((2, 2), dtype=np.float32) * (i + 1)
            weight = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"weight_{i}",
                    shape=(2, 2),
                    dtype=np.float32,
                ),
            )
            weights.append(weight)

        centroid = Centroid.from_weights(weights, "test_cluster")

        # Should compute mean of all weights
        expected_data = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
        np.testing.assert_array_equal(centroid.data, expected_data)
        assert centroid.cluster_id == "test_cluster"
        assert centroid.shape == (2, 2)
        assert centroid.dtype == np.float32

    def test_centroid_serialization(self):
        """Test Centroid serialization."""
        data = np.array([1.5, 2.5], dtype=np.float32)
        centroid = Centroid(
            data=data,
            cluster_id="test_centroid",
            shape=(2,),
            dtype=np.float32,
        )

        # Test to_dict
        centroid_dict = centroid.to_dict()
        assert centroid_dict["cluster_id"] == "test_centroid"
        assert centroid_dict["shape"] == [2]
        assert centroid_dict["dtype"] == "float32"
        assert "data_bytes" in centroid_dict
        assert "hash" in centroid_dict

        # Test from_dict
        restored_centroid = Centroid.from_dict(centroid_dict)
        assert restored_centroid.cluster_id == "test_centroid"
        assert restored_centroid.shape == (2,)
        assert restored_centroid.dtype == np.float32
        np.testing.assert_array_equal(restored_centroid.data, data)


class TestClusterAssignment:
    """Test ClusterAssignment dataclass."""

    def test_cluster_assignment_creation(self):
        """Test creating ClusterAssignment."""
        assignment = ClusterAssignment(
            weight_name="test_weight",
            weight_hash="hash123",
            cluster_id="cluster_001",
            distance_to_centroid=0.25,
            similarity_score=0.95,
        )

        assert assignment.weight_name == "test_weight"
        assert assignment.weight_hash == "hash123"
        assert assignment.cluster_id == "cluster_001"
        assert assignment.distance_to_centroid == 0.25
        assert assignment.similarity_score == 0.95

    def test_cluster_assignment_defaults(self):
        """Test ClusterAssignment with default values."""
        assignment = ClusterAssignment(
            weight_name="test", weight_hash="hash", cluster_id="cluster"
        )

        assert assignment.distance_to_centroid == 0.0
        assert assignment.similarity_score == 0.0
        assert assignment.is_representative is False

    def test_cluster_assignment_serialization(self):
        """Test ClusterAssignment serialization."""
        assignment = ClusterAssignment(
            weight_name="weight_1",
            weight_hash="abc123",
            cluster_id="cluster_5",
            distance_to_centroid=0.1,
            similarity_score=0.98,
            is_representative=True,
        )

        # Test to_dict
        assignment_dict = assignment.to_dict()
        assert assignment_dict["weight_name"] == "weight_1"
        assert assignment_dict["cluster_id"] == "cluster_5"
        assert assignment_dict["is_representative"] is True

        # Test from_dict
        restored_assignment = ClusterAssignment.from_dict(assignment_dict)
        assert restored_assignment.weight_name == "weight_1"
        assert restored_assignment.weight_hash == "abc123"
        assert restored_assignment.cluster_id == "cluster_5"
        assert restored_assignment.similarity_score == 0.98
        assert restored_assignment.is_representative is True


class TestClusteringConfig:
    """Test ClusteringConfig class."""

    def test_clustering_config_defaults(self):
        """Test ClusteringConfig with default values."""
        config = ClusteringConfig()

        assert config.strategy == ClusteringStrategy.ADAPTIVE
        assert config.level == ClusterLevel.TENSOR
        assert config.similarity_threshold == 0.95
        assert config.min_cluster_size == 2
        assert config.max_clusters is None
        assert config.feature_extraction == "raw"

    def test_clustering_config_validation(self):
        """Test ClusteringConfig validation."""
        # Valid configuration
        config = ClusteringConfig(
            similarity_threshold=0.8,
            min_cluster_size=3,
            max_clusters=10,
        )
        assert config.validate()

        # Invalid similarity threshold
        invalid_config = ClusteringConfig(similarity_threshold=1.5)
        assert not invalid_config.validate()

        # Invalid min_cluster_size
        invalid_config = ClusteringConfig(min_cluster_size=0)
        assert not invalid_config.validate()

        # Invalid max_clusters
        invalid_config = ClusteringConfig(max_clusters=-1)
        assert not invalid_config.validate()

    def test_clustering_config_serialization(self):
        """Test ClusteringConfig serialization."""
        config = ClusteringConfig(
            strategy=ClusteringStrategy.KMEANS,
            level=ClusterLevel.LAYER,
            similarity_threshold=0.9,
            min_cluster_size=5,
            max_clusters=20,
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["strategy"] == "kmeans"
        assert config_dict["level"] == "layer"
        assert config_dict["similarity_threshold"] == 0.9
        assert config_dict["min_cluster_size"] == 5

        # Test from_dict
        restored_config = ClusteringConfig.from_dict(config_dict)
        assert restored_config.strategy == ClusteringStrategy.KMEANS
        assert restored_config.level == ClusterLevel.LAYER
        assert restored_config.similarity_threshold == 0.9
        assert restored_config.max_clusters == 20


class TestHierarchyConfig:
    """Test HierarchyConfig class."""

    def test_hierarchy_config_defaults(self):
        """Test HierarchyConfig with default values."""
        config = HierarchyConfig()

        assert config.levels == [ClusterLevel.TENSOR, ClusterLevel.LAYER]
        assert config.propagate_up is True
        assert config.merge_threshold == 0.8
        assert config.split_threshold == 0.3

    def test_hierarchy_config_validation(self):
        """Test HierarchyConfig validation."""
        # Valid configuration
        config = HierarchyConfig(
            levels=[ClusterLevel.TENSOR, ClusterLevel.BLOCK, ClusterLevel.LAYER],
            merge_threshold=0.7,
            split_threshold=0.4,
        )
        assert config.validate()

        # Invalid: merge_threshold <= split_threshold
        invalid_config = HierarchyConfig(merge_threshold=0.5, split_threshold=0.6)
        assert not invalid_config.validate()

        # Invalid: empty levels
        invalid_config = HierarchyConfig(levels=[])
        assert not invalid_config.validate()

    def test_hierarchy_config_serialization(self):
        """Test HierarchyConfig serialization."""
        config = HierarchyConfig(
            levels=[ClusterLevel.TENSOR, ClusterLevel.LAYER, ClusterLevel.MODEL],
            propagate_up=False,
            merge_threshold=0.85,
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["levels"] == ["tensor", "layer", "model"]
        assert config_dict["propagate_up"] is False
        assert config_dict["merge_threshold"] == 0.85

        # Test from_dict
        restored_config = HierarchyConfig.from_dict(config_dict)
        assert restored_config.levels == [
            ClusterLevel.TENSOR,
            ClusterLevel.LAYER,
            ClusterLevel.MODEL,
        ]
        assert restored_config.propagate_up is False
        assert restored_config.merge_threshold == 0.85


class TestOptimizationConfig:
    """Test OptimizationConfig class."""

    def test_optimization_config_defaults(self):
        """Test OptimizationConfig with default values."""
        config = OptimizationConfig()

        assert config.enable_auto_reclustering is True
        assert config.reclustering_interval == 100
        assert config.quality_threshold == 0.7
        assert config.performance_weight == 0.5
        assert config.compression_weight == 0.5

    def test_optimization_config_validation(self):
        """Test OptimizationConfig validation."""
        # Valid configuration
        config = OptimizationConfig(
            quality_threshold=0.8,
            performance_weight=0.3,
            compression_weight=0.7,
        )
        assert config.validate()

        # Invalid: weights don't sum to 1.0
        invalid_config = OptimizationConfig(
            performance_weight=0.3, compression_weight=0.4
        )
        assert not invalid_config.validate()

        # Invalid: negative reclustering interval
        invalid_config = OptimizationConfig(reclustering_interval=-1)
        assert not invalid_config.validate()

    def test_optimization_config_serialization(self):
        """Test OptimizationConfig serialization."""
        config = OptimizationConfig(
            enable_auto_reclustering=False,
            reclustering_interval=50,
            quality_threshold=0.85,
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["enable_auto_reclustering"] is False
        assert config_dict["reclustering_interval"] == 50
        assert config_dict["quality_threshold"] == 0.85

        # Test from_dict
        restored_config = OptimizationConfig.from_dict(config_dict)
        assert restored_config.enable_auto_reclustering is False
        assert restored_config.reclustering_interval == 50
        assert restored_config.quality_threshold == 0.85