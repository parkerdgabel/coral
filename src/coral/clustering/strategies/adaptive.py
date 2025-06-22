"""Adaptive clustering strategy that selects the best approach based on data characteristics."""

import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from sklearn.metrics import silhouette_score
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from .base import ClusteringStrategy, ClusteringResult
from .kmeans import KMeansStrategy
from .hierarchical import HierarchicalStrategy
from .dbscan import DBSCANStrategy

logger = logging.getLogger(__name__)


class AdaptiveStrategy(ClusteringStrategy):
    """
    Adaptive clustering strategy that analyzes data and selects the best approach.
    
    This strategy examines data characteristics (size, dimensionality, distribution)
    and tries multiple clustering algorithms to find the best fit.
    
    Parameters:
        strategies: List of strategies to try (default: ['kmeans', 'hierarchical', 'dbscan'])
        selection_metric: Metric to use for selection ('silhouette', 'davies_bouldin', 'combined')
        max_samples_full_search: Maximum samples for trying all strategies (default: 1000)
        n_clusters_range: Range of clusters to try for k-means (default: (2, 20))
        parallel: Run strategies in parallel (default: True)
        scale_features: Whether to scale features (default: True)
        strategy_params: Dict of params for each strategy
    
    Example:
        >>> # Let the adaptive strategy choose the best approach
        >>> strategy = AdaptiveStrategy()
        >>> result = strategy.cluster(weights)
        >>> print(f"Selected {result.metadata['selected_strategy']} strategy")
        
        >>> # Customize which strategies to consider
        >>> strategy = AdaptiveStrategy(
        ...     strategies=['kmeans', 'dbscan'],
        ...     strategy_params={
        ...         'kmeans': {'n_clusters': 10},
        ...         'dbscan': {'min_samples': 3}
        ...     }
        ... )
        >>> result = strategy.cluster(weights)
    """
    
    @property
    def name(self) -> str:
        """Strategy name."""
        return "adaptive"
    
    def validate_params(self) -> None:
        """Validate strategy parameters."""
        # Set defaults
        self.params.setdefault('strategies', ['kmeans', 'hierarchical', 'dbscan'])
        self.params.setdefault('selection_metric', 'combined')
        self.params.setdefault('max_samples_full_search', 1000)
        self.params.setdefault('n_clusters_range', (2, 20))
        self.params.setdefault('parallel', True)
        self.params.setdefault('scale_features', True)
        self.params.setdefault('strategy_params', {})
        
        # Validate
        valid_strategies = ['kmeans', 'hierarchical', 'dbscan']
        for strategy in self.params['strategies']:
            if strategy not in valid_strategies:
                raise ValueError(f"Unknown strategy: {strategy}. Must be one of {valid_strategies}")
        
        valid_metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'combined']
        if self.params['selection_metric'] not in valid_metrics:
            raise ValueError(f"selection_metric must be one of {valid_metrics}")
        
        if not isinstance(self.params['n_clusters_range'], (tuple, list)) or len(self.params['n_clusters_range']) != 2:
            raise ValueError("n_clusters_range must be a tuple/list of (min, max)")
    
    def cluster(
        self,
        weights: Dict[str, np.ndarray],
        existing_clusters: Optional[ClusteringResult] = None
    ) -> ClusteringResult:
        """
        Perform adaptive clustering by trying multiple strategies.
        
        Args:
            weights: Dict mapping weight hash to weight array
            existing_clusters: Optional existing clusters for incremental clustering
            
        Returns:
            ClusteringResult from the best-performing strategy
        """
        if len(weights) == 0:
            return ClusteringResult({}, {}, {}, {'error': 'No weights to cluster'})
        
        n_samples = len(weights)
        
        # Analyze data characteristics
        data_stats = self._analyze_data(weights)
        logger.info(f"Data analysis: {n_samples} samples, "
                   f"avg_dim={data_stats['avg_dimensionality']:.0f}, "
                   f"n_shapes={data_stats['n_unique_shapes']}")
        
        # Determine which strategies to try
        strategies_to_try = self._select_strategies(data_stats)
        logger.info(f"Will try strategies: {strategies_to_try}")
        
        # Try each strategy
        results = {}
        
        if self.params['parallel'] and len(strategies_to_try) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=len(strategies_to_try)) as executor:
                future_to_strategy = {
                    executor.submit(
                        self._try_strategy,
                        strategy_name,
                        weights,
                        existing_clusters,
                        data_stats
                    ): strategy_name
                    for strategy_name in strategies_to_try
                }
                
                for future in as_completed(future_to_strategy):
                    strategy_name = future_to_strategy[future]
                    try:
                        result = future.result()
                        if result:
                            results[strategy_name] = result
                    except Exception as e:
                        logger.error(f"Strategy {strategy_name} failed: {e}")
        else:
            # Sequential execution
            for strategy_name in strategies_to_try:
                try:
                    result = self._try_strategy(
                        strategy_name,
                        weights,
                        existing_clusters,
                        data_stats
                    )
                    if result:
                        results[strategy_name] = result
                except Exception as e:
                    logger.error(f"Strategy {strategy_name} failed: {e}")
        
        # Select best result
        if not results:
            logger.error("All strategies failed")
            return ClusteringResult({}, {}, {}, {'error': 'All clustering strategies failed'})
        
        best_strategy, best_result = self._select_best_result(results)
        logger.info(f"Selected {best_strategy} as best strategy")
        
        # Add adaptive metadata
        best_result.metadata['selected_strategy'] = best_strategy
        best_result.metadata['strategies_tried'] = list(results.keys())
        best_result.metadata['data_stats'] = data_stats
        best_result.metadata['adaptive'] = True
        
        # Add comparison metrics
        comparison = {}
        for strategy_name, result in results.items():
            comparison[strategy_name] = {
                'n_clusters': result.metrics.get('n_clusters', 0),
                'silhouette': result.metrics.get('silhouette_score', -1),
                'davies_bouldin': result.metrics.get('davies_bouldin_score', float('inf')),
                'noise_ratio': result.metrics.get('noise_ratio', 0)
            }
        best_result.metadata['strategy_comparison'] = comparison
        
        return best_result
    
    def _analyze_data(self, weights: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze data characteristics to inform strategy selection."""
        shapes = [w.shape for w in weights.values()]
        dimensionalities = [np.prod(shape) for shape in shapes]
        
        # Basic statistics
        stats = {
            'n_samples': len(weights),
            'n_unique_shapes': len(set(shapes)),
            'avg_dimensionality': float(np.mean(dimensionalities)),
            'std_dimensionality': float(np.std(dimensionalities)),
            'min_dimensionality': int(np.min(dimensionalities)),
            'max_dimensionality': int(np.max(dimensionalities))
        }
        
        # Sample some weights for distribution analysis
        sample_size = min(100, len(weights))
        sample_weights = list(weights.values())[:sample_size]
        
        # Analyze value distributions
        all_values = np.concatenate([w.flatten() for w in sample_weights])
        stats['value_mean'] = float(np.mean(all_values))
        stats['value_std'] = float(np.std(all_values))
        stats['value_skew'] = float(self._compute_skewness(all_values))
        
        # Estimate sparsity
        stats['sparsity'] = float(np.mean(all_values == 0))
        
        # Estimate data spread (for DBSCAN eps tuning)
        if len(sample_weights) > 1:
            # Compute pairwise distances on sample
            from sklearn.metrics.pairwise import pairwise_distances
            sample_matrix = np.vstack([w.flatten()[:1000] for w in sample_weights])  # Limit size
            distances = pairwise_distances(sample_matrix, metric='euclidean')
            np.fill_diagonal(distances, np.inf)
            stats['avg_nearest_distance'] = float(np.mean(np.min(distances, axis=1)))
        
        return stats
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _select_strategies(self, data_stats: Dict[str, float]) -> List[str]:
        """Select which strategies to try based on data characteristics."""
        strategies = []
        n_samples = data_stats['n_samples']
        
        # Always include user-specified strategies
        strategies.extend(self.params['strategies'])
        
        # For very large datasets, prefer scalable methods
        if n_samples > self.params['max_samples_full_search']:
            if 'kmeans' not in strategies:
                strategies.append('kmeans')
            # Maybe skip hierarchical for very large data
            if n_samples > 10000 and 'hierarchical' in strategies:
                strategies.remove('hierarchical')
        
        # For sparse data, DBSCAN might work well
        if data_stats['sparsity'] > 0.5 and 'dbscan' not in strategies:
            strategies.append('dbscan')
        
        # For data with multiple shapes, hierarchical might be good
        if data_stats['n_unique_shapes'] > 5 and 'hierarchical' not in strategies:
            strategies.append('hierarchical')
        
        return list(set(strategies))  # Remove duplicates
    
    def _try_strategy(
        self,
        strategy_name: str,
        weights: Dict[str, np.ndarray],
        existing_clusters: Optional[ClusteringResult],
        data_stats: Dict[str, float]
    ) -> Optional[ClusteringResult]:
        """Try a specific clustering strategy."""
        logger.info(f"Trying {strategy_name} strategy...")
        
        # Get strategy-specific params
        strategy_params = self.params['strategy_params'].get(strategy_name, {})
        
        # Add adaptive tuning based on data stats
        if strategy_name == 'kmeans':
            if 'n_clusters' not in strategy_params:
                # Estimate good number of clusters
                n_samples = data_stats['n_samples']
                min_k, max_k = self.params['n_clusters_range']
                # Rule of thumb: sqrt(n/2)
                k_estimate = int(np.sqrt(n_samples / 2))
                k_estimate = max(min_k, min(k_estimate, max_k))
                strategy_params['n_clusters'] = k_estimate
            
            # Use mini-batch for large datasets
            if data_stats['n_samples'] > 5000:
                strategy_params['use_minibatch'] = True
        
        elif strategy_name == 'hierarchical':
            if 'n_clusters' not in strategy_params:
                # Similar estimation as k-means
                n_samples = data_stats['n_samples']
                min_k, max_k = self.params['n_clusters_range']
                k_estimate = int(np.sqrt(n_samples / 2))
                k_estimate = max(min_k, min(k_estimate, max_k))
                strategy_params['n_clusters'] = k_estimate
        
        elif strategy_name == 'dbscan':
            # Auto eps tuning is usually good
            strategy_params['auto_eps'] = strategy_params.get('auto_eps', True)
            
            # Adjust min_samples based on data size
            if 'min_samples' not in strategy_params:
                # Rule of thumb: log(n)
                min_samples = max(2, int(np.log(data_stats['n_samples'])))
                strategy_params['min_samples'] = min_samples
        
        # Always use same feature scaling setting
        strategy_params['scale_features'] = self.params['scale_features']
        
        # Create and run strategy
        try:
            if strategy_name == 'kmeans':
                strategy = KMeansStrategy(**strategy_params)
            elif strategy_name == 'hierarchical':
                strategy = HierarchicalStrategy(**strategy_params)
            elif strategy_name == 'dbscan':
                strategy = DBSCANStrategy(**strategy_params)
            else:
                logger.error(f"Unknown strategy: {strategy_name}")
                return None
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = strategy.cluster(weights, existing_clusters)
            
            # Validate result
            if result.n_clusters == 0 and strategy_name != 'dbscan':
                logger.warning(f"{strategy_name} produced no clusters")
                return None
            
            return result
            
        except Exception as e:
            logger.error(f"Strategy {strategy_name} failed: {e}")
            return None
    
    def _select_best_result(
        self,
        results: Dict[str, ClusteringResult]
    ) -> Tuple[str, ClusteringResult]:
        """Select the best result based on metrics."""
        metric_name = self.params['selection_metric']
        
        scores = {}
        for strategy_name, result in results.items():
            if metric_name == 'silhouette':
                score = result.metrics.get('silhouette_score', -1)
            elif metric_name == 'davies_bouldin':
                # Lower is better for Davies-Bouldin
                db_score = result.metrics.get('davies_bouldin_score', float('inf'))
                score = -db_score if db_score != float('inf') else -1000
            elif metric_name == 'calinski_harabasz':
                score = result.metrics.get('calinski_harabasz_score', 0)
            elif metric_name == 'combined':
                # Combine multiple metrics
                silhouette = result.metrics.get('silhouette_score', -1)
                db_score = result.metrics.get('davies_bouldin_score', float('inf'))
                ch_score = result.metrics.get('calinski_harabasz_score', 0)
                
                # Normalize and combine
                score = 0
                if silhouette > -1:
                    score += silhouette * 0.4  # Weight silhouette heavily
                if db_score != float('inf'):
                    # Normalize DB score (lower is better)
                    normalized_db = 1 / (1 + db_score)
                    score += normalized_db * 0.3
                if ch_score > 0:
                    # Normalize CH score
                    normalized_ch = min(1, ch_score / 1000)
                    score += normalized_ch * 0.3
                
                # Penalty for too many noise points (DBSCAN)
                noise_ratio = result.metrics.get('noise_ratio', 0)
                if noise_ratio > 0.5:
                    score *= (1 - noise_ratio)
            else:
                score = 0
            
            scores[strategy_name] = score
            logger.info(f"{strategy_name} score: {score:.4f}")
        
        # Select best
        best_strategy = max(scores, key=scores.get)
        return best_strategy, results[best_strategy]