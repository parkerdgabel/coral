"""Clustering strategies for weight deduplication."""

from .base import ClusteringStrategy, ClusteringResult
from .kmeans import KMeansStrategy
from .hierarchical import HierarchicalStrategy
from .dbscan import DBSCANStrategy
from .adaptive import AdaptiveStrategy

__all__ = [
    'ClusteringStrategy',
    'ClusteringResult',
    'KMeansStrategy',
    'HierarchicalStrategy',
    'DBSCANStrategy',
    'AdaptiveStrategy'
]

# Strategy registry for easy lookup
STRATEGIES = {
    'kmeans': KMeansStrategy,
    'hierarchical': HierarchicalStrategy,
    'dbscan': DBSCANStrategy,
    'adaptive': AdaptiveStrategy
}


def get_strategy(name: str, **params) -> ClusteringStrategy:
    """
    Get a clustering strategy by name.
    
    Args:
        name: Strategy name ('kmeans', 'hierarchical', 'dbscan', 'adaptive')
        **params: Parameters to pass to the strategy
        
    Returns:
        Initialized clustering strategy
        
    Raises:
        ValueError: If strategy name is not recognized
    """
    if name not in STRATEGIES:
        raise ValueError(f"Unknown clustering strategy: {name}. "
                        f"Available strategies: {list(STRATEGIES.keys())}")
    
    strategy_class = STRATEGIES[name]
    return strategy_class(**params)