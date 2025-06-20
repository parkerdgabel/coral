"""Configuration classes for clustering-based deduplication."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .cluster_types import ClusterLevel, ClusteringStrategy


@dataclass
class ClusteringConfig:
    """Configuration for clustering-based weight deduplication."""

    # Core clustering parameters
    strategy: ClusteringStrategy = ClusteringStrategy.ADAPTIVE
    level: ClusterLevel = ClusterLevel.TENSOR
    similarity_threshold: float = 0.95  # Minimum similarity for clustering
    min_cluster_size: int = 2  # Minimum weights per cluster
    max_clusters: Optional[int] = None  # Maximum number of clusters (None = unlimited)

    # Feature extraction and preprocessing
    feature_extraction: str = "raw"  # "raw", "pca", "hash", "statistical"
    normalize_features: bool = True  # Normalize features before clustering
    dimensionality_reduction: Optional[int] = None  # Target dimensions for reduction

    # Algorithm-specific parameters
    kmeans_params: Dict[str, Any] = field(default_factory=dict)  # K-means parameters
    hierarchical_params: Dict[str, Any] = field(default_factory=dict)  # Hierarchical parameters
    dbscan_params: Dict[str, Any] = field(default_factory=dict)  # DBSCAN parameters

    # Performance and quality tuning
    random_seed: Optional[int] = 42  # Reproducible clustering results
    max_iterations: int = 100  # Maximum clustering iterations
    convergence_tolerance: float = 1e-4  # Convergence threshold
    parallel_workers: int = -1  # Number of parallel workers (-1 = auto)

    def validate(self) -> bool:
        """Validate configuration parameters."""
        # Check similarity threshold range
        if not (0.0 <= self.similarity_threshold <= 1.0):
            return False

        # Check minimum cluster size
        if self.min_cluster_size < 1:
            return False

        # Check maximum clusters
        if self.max_clusters is not None and self.max_clusters < 1:
            return False

        # Check max_clusters >= min_cluster_size constraint
        if (
            self.max_clusters is not None
            and self.max_clusters < self.min_cluster_size
        ):
            return False

        # Check convergence tolerance
        if self.convergence_tolerance <= 0:
            return False

        # Check max iterations
        if self.max_iterations < 1:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy": self.strategy.value,
            "level": self.level.value,
            "similarity_threshold": self.similarity_threshold,
            "min_cluster_size": self.min_cluster_size,
            "max_clusters": self.max_clusters,
            "feature_extraction": self.feature_extraction,
            "normalize_features": self.normalize_features,
            "dimensionality_reduction": self.dimensionality_reduction,
            "kmeans_params": self.kmeans_params,
            "hierarchical_params": self.hierarchical_params,
            "dbscan_params": self.dbscan_params,
            "random_seed": self.random_seed,
            "max_iterations": self.max_iterations,
            "convergence_tolerance": self.convergence_tolerance,
            "parallel_workers": self.parallel_workers,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusteringConfig":
        """Create from dictionary."""
        return cls(
            strategy=ClusteringStrategy(data.get("strategy", "adaptive")),
            level=ClusterLevel(data.get("level", "tensor")),
            similarity_threshold=data.get("similarity_threshold", 0.95),
            min_cluster_size=data.get("min_cluster_size", 2),
            max_clusters=data.get("max_clusters"),
            feature_extraction=data.get("feature_extraction", "raw"),
            normalize_features=data.get("normalize_features", True),
            dimensionality_reduction=data.get("dimensionality_reduction"),
            kmeans_params=data.get("kmeans_params", {}),
            hierarchical_params=data.get("hierarchical_params", {}),
            dbscan_params=data.get("dbscan_params", {}),
            random_seed=data.get("random_seed", 42),
            max_iterations=data.get("max_iterations", 100),
            convergence_tolerance=data.get("convergence_tolerance", 1e-4),
            parallel_workers=data.get("parallel_workers", -1),
        )

    def get_algorithm_params(self) -> Dict[str, Any]:
        """Get parameters specific to the selected clustering algorithm."""
        if self.strategy == ClusteringStrategy.KMEANS:
            return self.kmeans_params
        elif self.strategy == ClusteringStrategy.HIERARCHICAL:
            return self.hierarchical_params
        elif self.strategy == ClusteringStrategy.DBSCAN:
            return self.dbscan_params
        else:
            return {}

    def set_algorithm_params(self, params: Dict[str, Any]) -> None:
        """Set parameters for the current clustering algorithm."""
        if self.strategy == ClusteringStrategy.KMEANS:
            self.kmeans_params.update(params)
        elif self.strategy == ClusteringStrategy.HIERARCHICAL:
            self.hierarchical_params.update(params)
        elif self.strategy == ClusteringStrategy.DBSCAN:
            self.dbscan_params.update(params)


@dataclass
class HierarchyConfig:
    """Configuration for hierarchical clustering across multiple levels."""

    # Hierarchy structure
    levels: List[ClusterLevel] = field(
        default_factory=lambda: [ClusterLevel.TENSOR, ClusterLevel.LAYER]
    )
    propagate_up: bool = True  # Propagate clusters up the hierarchy
    propagate_down: bool = False  # Propagate clusters down the hierarchy

    # Level transition thresholds
    merge_threshold: float = 0.8  # Threshold for merging clusters at higher levels
    split_threshold: float = 0.3  # Threshold for splitting clusters at lower levels
    min_level_size: Dict[ClusterLevel, int] = field(default_factory=dict)  # Min size per level

    # Cross-level relationships
    enforce_consistency: bool = True  # Ensure parent-child cluster consistency
    max_hierarchy_depth: int = 4  # Maximum depth of hierarchy

    def validate(self) -> bool:
        """Validate hierarchy configuration."""
        # Check that levels list is not empty
        if not self.levels:
            return False

        # Check threshold relationship
        if self.merge_threshold <= self.split_threshold:
            return False

        # Check threshold ranges
        if not (0.0 <= self.merge_threshold <= 1.0):
            return False
        if not (0.0 <= self.split_threshold <= 1.0):
            return False

        # Check hierarchy depth
        if self.max_hierarchy_depth < 1:
            return False

        # Validate min_level_size values
        for size in self.min_level_size.values():
            if size < 1:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "levels": [level.value for level in self.levels],
            "propagate_up": self.propagate_up,
            "propagate_down": self.propagate_down,
            "merge_threshold": self.merge_threshold,
            "split_threshold": self.split_threshold,
            "min_level_size": {
                level.value: size for level, size in self.min_level_size.items()
            },
            "enforce_consistency": self.enforce_consistency,
            "max_hierarchy_depth": self.max_hierarchy_depth,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HierarchyConfig":
        """Create from dictionary."""
        # Convert level strings back to enums
        levels = [ClusterLevel(level) for level in data.get("levels", ["tensor", "layer"])]
        
        # Convert min_level_size back to enum keys
        min_level_size = {}
        for level_str, size in data.get("min_level_size", {}).items():
            min_level_size[ClusterLevel(level_str)] = size

        return cls(
            levels=levels,
            propagate_up=data.get("propagate_up", True),
            propagate_down=data.get("propagate_down", False),
            merge_threshold=data.get("merge_threshold", 0.8),
            split_threshold=data.get("split_threshold", 0.3),
            min_level_size=min_level_size,
            enforce_consistency=data.get("enforce_consistency", True),
            max_hierarchy_depth=data.get("max_hierarchy_depth", 4),
        )

    def get_level_config(self, level: ClusterLevel) -> Dict[str, Any]:
        """Get configuration specific to a clustering level."""
        config = {
            "level": level,
            "min_size": self.min_level_size.get(level, 2),
            "can_merge_up": self.propagate_up and level != self.levels[-1],
            "can_split_down": self.propagate_down and level != self.levels[0],
        }
        return config


@dataclass
class OptimizationConfig:
    """Configuration for clustering optimization and re-clustering policies."""

    # Auto re-clustering policies
    enable_auto_reclustering: bool = True  # Enable automatic re-clustering
    reclustering_interval: int = 100  # Re-cluster after N new weights added
    quality_threshold: float = 0.7  # Re-cluster if quality drops below threshold
    size_threshold_multiplier: float = 2.0  # Re-cluster if size grows by this factor

    # Optimization objectives
    performance_weight: float = 0.5  # Weight for clustering performance in optimization
    compression_weight: float = 0.5  # Weight for compression ratio in optimization
    quality_weight: float = 0.0  # Weight for clustering quality metrics

    # Resource limits
    max_clustering_time: Optional[float] = None  # Maximum time for clustering (seconds)
    max_memory_usage: Optional[int] = None  # Maximum memory usage (bytes)
    enable_incremental: bool = True  # Use incremental clustering when possible

    # Caching and persistence
    cache_centroids: bool = True  # Cache computed centroids
    persist_assignments: bool = True  # Persist cluster assignments
    cleanup_interval: int = 1000  # Clean up unused clusters every N operations

    def validate(self) -> bool:
        """Validate optimization configuration."""
        # Check weight values and their sum
        total_weight = self.performance_weight + self.compression_weight + self.quality_weight
        if abs(total_weight - 1.0) > 1e-6:  # Allow small floating point errors
            # If total is not 1.0, check if at least the main weights sum correctly
            main_total = self.performance_weight + self.compression_weight
            if abs(main_total - 1.0) > 1e-6 and self.quality_weight == 0.0:
                return False

        # Check individual weight ranges
        if not (0.0 <= self.performance_weight <= 1.0):
            return False
        if not (0.0 <= self.compression_weight <= 1.0):
            return False
        if not (0.0 <= self.quality_weight <= 1.0):
            return False

        # Check thresholds
        if not (0.0 <= self.quality_threshold <= 1.0):
            return False
        if self.size_threshold_multiplier <= 1.0:
            return False

        # Check intervals
        if self.reclustering_interval < 1:
            return False
        if self.cleanup_interval < 1:
            return False

        # Check resource limits
        if self.max_clustering_time is not None and self.max_clustering_time <= 0:
            return False
        if self.max_memory_usage is not None and self.max_memory_usage <= 0:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enable_auto_reclustering": self.enable_auto_reclustering,
            "reclustering_interval": self.reclustering_interval,
            "quality_threshold": self.quality_threshold,
            "size_threshold_multiplier": self.size_threshold_multiplier,
            "performance_weight": self.performance_weight,
            "compression_weight": self.compression_weight,
            "quality_weight": self.quality_weight,
            "max_clustering_time": self.max_clustering_time,
            "max_memory_usage": self.max_memory_usage,
            "enable_incremental": self.enable_incremental,
            "cache_centroids": self.cache_centroids,
            "persist_assignments": self.persist_assignments,
            "cleanup_interval": self.cleanup_interval,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationConfig":
        """Create from dictionary."""
        return cls(
            enable_auto_reclustering=data.get("enable_auto_reclustering", True),
            reclustering_interval=data.get("reclustering_interval", 100),
            quality_threshold=data.get("quality_threshold", 0.7),
            size_threshold_multiplier=data.get("size_threshold_multiplier", 2.0),
            performance_weight=data.get("performance_weight", 0.5),
            compression_weight=data.get("compression_weight", 0.5),
            quality_weight=data.get("quality_weight", 0.0),
            max_clustering_time=data.get("max_clustering_time"),
            max_memory_usage=data.get("max_memory_usage"),
            enable_incremental=data.get("enable_incremental", True),
            cache_centroids=data.get("cache_centroids", True),
            persist_assignments=data.get("persist_assignments", True),
            cleanup_interval=data.get("cleanup_interval", 1000),
        )

    def compute_objective_score(
        self, performance_score: float, compression_score: float, quality_score: float = 0.0
    ) -> float:
        """
        Compute weighted objective score for clustering optimization.

        Args:
            performance_score: Performance metric score [0, 1]
            compression_score: Compression ratio score [0, 1]
            quality_score: Clustering quality score [0, 1]

        Returns:
            Weighted objective score [0, 1]
        """
        return (
            self.performance_weight * performance_score
            + self.compression_weight * compression_score
            + self.quality_weight * quality_score
        )