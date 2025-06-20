"""
Coral Clustering Module

This module provides clustering-based deduplication for neural network weights.
It implements multiple clustering strategies and hierarchical organization
to optimize storage and retrieval of similar weight tensors.

Key Features:
- Multiple clustering algorithms (K-means, Hierarchical, DBSCAN, Adaptive)
- Hierarchical clustering across tensor/block/layer/model levels
- Configurable similarity thresholds and optimization policies
- Comprehensive metrics and quality assessment
- Integration with Coral's delta encoding system

Example Usage:
    from coral.clustering import ClusteringConfig, ClusteringStrategy, ClusterLevel
    
    # Configure clustering for tensor-level K-means
    config = ClusteringConfig(
        strategy=ClusteringStrategy.KMEANS,
        level=ClusterLevel.TENSOR,
        similarity_threshold=0.95,
        min_cluster_size=3
    )
    
    # Create hierarchical configuration
    from coral.clustering import HierarchyConfig
    hierarchy = HierarchyConfig(
        levels=[ClusterLevel.TENSOR, ClusterLevel.LAYER, ClusterLevel.MODEL],
        merge_threshold=0.8
    )
"""

from .cluster_analyzer import ClusterAnalyzer, RepositoryAnalysis, ClusteringResult
from .cluster_assigner import ClusterAssigner, AssignmentHistory, AssignmentMetrics
from .cluster_config import ClusteringConfig, HierarchyConfig, OptimizationConfig
from .cluster_hierarchy import ClusterHierarchy, HierarchyNode, HierarchyMetrics
from .cluster_index import ClusterIndex, IndexStats
from .cluster_storage import ClusterStorage
from .cluster_types import (
    Centroid,
    ClusterAssignment,
    ClusterInfo,
    ClusterLevel,
    ClusterMetrics,
    ClusteringStrategy,
)
from .centroid_encoder import CentroidEncoder

# Version information
__version__ = "1.0.0"
__author__ = "Coral Development Team"

# Public API - all classes and enums that should be available to users
__all__ = [
    # Main analyzer class
    "ClusterAnalyzer",
    "RepositoryAnalysis", 
    "ClusteringResult",
    # Assignment management
    "ClusterAssigner",
    "AssignmentHistory",
    "AssignmentMetrics",
    # Hierarchy management
    "ClusterHierarchy",
    "HierarchyNode",
    "HierarchyMetrics",
    # Index management
    "ClusterIndex",
    "IndexStats",
    # Storage management
    "ClusterStorage",
    # Core types and enums
    "ClusteringStrategy",
    "ClusterLevel",
    "ClusterMetrics",
    "ClusterInfo",
    "Centroid",
    "ClusterAssignment",
    # Configuration classes
    "ClusteringConfig",
    "HierarchyConfig",
    "OptimizationConfig",
    # Centroid-based encoding
    "CentroidEncoder",
    # Version info
    "__version__",
]

# Module-level documentation
def get_supported_strategies():
    """Get list of supported clustering strategies."""
    return list(ClusteringStrategy)


def get_supported_levels():
    """Get list of supported clustering levels."""
    return list(ClusterLevel)


def create_default_config(
    strategy: ClusteringStrategy = ClusteringStrategy.ADAPTIVE,
    level: ClusterLevel = ClusterLevel.TENSOR,
) -> ClusteringConfig:
    """
    Create a default clustering configuration.
    
    Args:
        strategy: Clustering algorithm to use
        level: Hierarchical level for clustering
    
    Returns:
        Default clustering configuration
    """
    return ClusteringConfig(strategy=strategy, level=level)


def create_hierarchical_config(
    levels: list = None,
    merge_threshold: float = 0.8,
) -> HierarchyConfig:
    """
    Create a default hierarchical clustering configuration.
    
    Args:
        levels: List of clustering levels (defaults to TENSOR and LAYER)
        merge_threshold: Threshold for merging clusters at higher levels
    
    Returns:
        Default hierarchy configuration
    """
    if levels is None:
        levels = [ClusterLevel.TENSOR, ClusterLevel.LAYER]
    
    return HierarchyConfig(levels=levels, merge_threshold=merge_threshold)


# Integration with Coral's delta system
def is_compatible_with_delta_encoding() -> bool:
    """Check if clustering is compatible with delta encoding."""
    return True  # Clustering works with delta encoding


def get_recommended_config_for_delta() -> ClusteringConfig:
    """Get recommended clustering configuration for use with delta encoding."""
    return ClusteringConfig(
        strategy=ClusteringStrategy.ADAPTIVE,
        level=ClusterLevel.TENSOR,
        similarity_threshold=0.98,  # Higher threshold for delta encoding
        min_cluster_size=2,
        feature_extraction="raw",  # Raw features work best with delta encoding
        normalize_features=False,  # Delta encoding handles normalization
    )