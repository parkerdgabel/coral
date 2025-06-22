"""Hierarchical clustering strategy implementation."""

import numpy as np
from typing import Dict, Optional, Union, List
import logging
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

from .base import ClusteringStrategy, ClusteringResult

logger = logging.getLogger(__name__)


class HierarchicalStrategy(ClusteringStrategy):
    """
    Hierarchical clustering strategy.
    
    Supports different linkage methods and allows cutting the dendrogram at various levels.
    Particularly useful for understanding nested structure in model weights.
    
    Parameters:
        linkage_method: Linkage criterion ('ward', 'complete', 'average', 'single')
        n_clusters: Number of clusters to form (default: None, use distance threshold)
        distance_threshold: Distance threshold for forming clusters (default: None)
        metric: Distance metric ('euclidean', 'cosine', 'manhattan')
        scale_features: Whether to scale features (default: True)
        criterion: Criterion for forming clusters ('maxclust', 'distance', 'inconsistent')
        depth: Depth for inconsistency calculation (default: 2)
    
    Example:
        >>> # Cut tree to get exactly 5 clusters
        >>> strategy = HierarchicalStrategy(linkage_method='ward', n_clusters=5)
        >>> result = strategy.cluster(weights)
        
        >>> # Cut tree at specific distance threshold
        >>> strategy = HierarchicalStrategy(linkage_method='complete', distance_threshold=0.5)
        >>> result = strategy.cluster(weights)
    """
    
    @property
    def name(self) -> str:
        """Strategy name."""
        return "hierarchical"
    
    def validate_params(self) -> None:
        """Validate strategy parameters."""
        # Set defaults
        self.params.setdefault('linkage_method', 'ward')
        self.params.setdefault('n_clusters', None)
        self.params.setdefault('distance_threshold', None)
        self.params.setdefault('metric', 'euclidean')
        self.params.setdefault('scale_features', True)
        self.params.setdefault('criterion', 'maxclust')
        self.params.setdefault('depth', 2)
        
        # Validate
        valid_linkages = ['ward', 'complete', 'average', 'single', 'weighted', 'centroid', 'median']
        if self.params['linkage_method'] not in valid_linkages:
            raise ValueError(f"linkage_method must be one of {valid_linkages}")
        
        if self.params['linkage_method'] == 'ward' and self.params['metric'] != 'euclidean':
            logger.warning("Ward linkage requires euclidean metric. Overriding metric.")
            self.params['metric'] = 'euclidean'
        
        if self.params['n_clusters'] is None and self.params['distance_threshold'] is None:
            raise ValueError("Either n_clusters or distance_threshold must be specified")
        
        if self.params['n_clusters'] is not None and self.params['distance_threshold'] is not None:
            raise ValueError("Cannot specify both n_clusters and distance_threshold")
        
        if self.params['n_clusters'] is not None:
            if self.params['n_clusters'] < 1:
                raise ValueError("n_clusters must be >= 1")
            self.params['criterion'] = 'maxclust'
        else:
            self.params['criterion'] = 'distance'
        
        valid_metrics = ['euclidean', 'cosine', 'manhattan', 'chebyshev']
        if self.params['metric'] not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}")
    
    def cluster(
        self,
        weights: Dict[str, np.ndarray],
        existing_clusters: Optional[ClusteringResult] = None
    ) -> ClusteringResult:
        """
        Perform hierarchical clustering on weights.
        
        Args:
            weights: Dict mapping weight hash to weight array
            existing_clusters: Optional existing clusters for incremental clustering
            
        Returns:
            ClusteringResult with clusters, centroids, and metrics
        """
        if len(weights) == 0:
            return ClusteringResult({}, {}, {}, {'error': 'No weights to cluster'})
        
        n_samples = len(weights)
        
        # Single sample case
        if n_samples == 1:
            weight_hash = list(weights.keys())[0]
            return ClusteringResult(
                clusters={0: [weight_hash]},
                centroids={0: weights[weight_hash]},
                metrics={'n_clusters': 1, 'n_samples': 1},
                metadata={'single_sample': True}
            )
        
        # Prepare data
        data_matrix, hash_list, shape_groups = self._prepare_data(weights)
        
        # Scale features if requested
        if self.params['scale_features']:
            scaler = StandardScaler()
            data_matrix = scaler.fit_transform(data_matrix)
        
        # Compute pairwise distances
        logger.info(f"Computing pairwise distances with metric: {self.params['metric']}")
        try:
            if self.params['metric'] == 'cosine':
                # Normalize for cosine distance
                from sklearn.preprocessing import normalize
                data_matrix = normalize(data_matrix, norm='l2')
                distances = pdist(data_matrix, metric='euclidean')
            else:
                distances = pdist(data_matrix, metric=self.params['metric'])
        except Exception as e:
            logger.error(f"Distance computation failed: {e}")
            return ClusteringResult({}, {}, {}, {'error': str(e)})
        
        # Perform hierarchical clustering
        logger.info(f"Performing hierarchical clustering with {self.params['linkage_method']} linkage")
        try:
            Z = linkage(distances, method=self.params['linkage_method'])
        except Exception as e:
            logger.error(f"Linkage computation failed: {e}")
            return ClusteringResult({}, {}, {}, {'error': str(e)})
        
        # Cut the dendrogram to form clusters
        if self.params['n_clusters'] is not None:
            n_clusters = min(self.params['n_clusters'], n_samples)
            labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # Convert to 0-based
        else:
            labels = fcluster(
                Z,
                self.params['distance_threshold'],
                criterion=self.params['criterion'],
                depth=self.params['depth']
            ) - 1  # Convert to 0-based
        
        # Convert labels to clusters
        clusters, noise_points = self._labels_to_clusters(labels, hash_list)
        
        # Compute centroids
        centroids = self._compute_centroids(weights, clusters)
        
        # Compute metrics
        metrics = self._compute_metrics(data_matrix, labels, centroids)
        
        # Add hierarchical-specific metrics
        metrics['cophenetic_correlation'] = self._compute_cophenetic_correlation(Z, distances)
        metrics['max_cluster_distance'] = float(Z[-1, 2]) if len(Z) > 0 else 0.0
        
        # Store dendrogram info in metadata
        metadata = {
            'linkage_method': self.params['linkage_method'],
            'metric': self.params['metric'],
            'shape_groups': shape_groups,
            'scaled': self.params['scale_features'],
            'dendrogram_computed': True,
            'linkage_matrix_shape': Z.shape,
            'params': self.params.copy()
        }
        
        # Add level information - useful for understanding hierarchy
        if self.params['n_clusters'] is None:
            metadata['distance_threshold'] = self.params['distance_threshold']
        else:
            # Find the distance at which we get n_clusters
            if len(Z) >= self.params['n_clusters'] - 1:
                cut_height = Z[-(self.params['n_clusters']-1), 2]
                metadata['cut_height'] = float(cut_height)
        
        result = ClusteringResult(
            clusters=clusters,
            centroids=centroids,
            metrics=metrics,
            metadata=metadata,
            noise_points=noise_points
        )
        
        # Handle incremental clustering if needed
        if existing_clusters:
            result = self._merge_with_existing(result, existing_clusters, weights)
        
        return result
    
    def _compute_cophenetic_correlation(
        self,
        Z: np.ndarray,
        distances: np.ndarray
    ) -> float:
        """Compute cophenetic correlation coefficient."""
        from scipy.cluster.hierarchy import cophenet
        try:
            coph_dists = cophenet(Z)
            correlation = np.corrcoef(distances, coph_dists)[0, 1]
            return float(correlation)
        except Exception as e:
            logger.warning(f"Failed to compute cophenetic correlation: {e}")
            return 0.0
    
    def _merge_with_existing(
        self,
        new_result: ClusteringResult,
        existing_clusters: ClusteringResult,
        weights: Dict[str, np.ndarray]
    ) -> ClusteringResult:
        """
        Merge with existing clusters using hierarchical approach.
        
        This treats existing cluster centroids as additional points and
        re-clusters everything together.
        """
        # Collect all points: new weights + existing centroids
        all_points = {}
        all_points.update(weights)
        
        # Add existing centroids as pseudo-points
        centroid_mapping = {}  # Maps pseudo-hash to original cluster
        for cluster_id, centroid in existing_clusters.centroids.items():
            pseudo_hash = f"__centroid_{cluster_id}__"
            all_points[pseudo_hash] = centroid
            centroid_mapping[pseudo_hash] = cluster_id
        
        # Re-cluster everything
        combined_result = self.cluster(all_points, None)
        
        # Post-process to handle centroid pseudo-points
        final_clusters = {}
        final_centroids = {}
        
        for new_cluster_id, members in combined_result.clusters.items():
            final_members = []
            absorbed_clusters = []
            
            for member in members:
                if member.startswith("__centroid_"):
                    # This is an existing cluster centroid
                    old_cluster_id = centroid_mapping[member]
                    absorbed_clusters.append(old_cluster_id)
                    # Add all members of the old cluster
                    if old_cluster_id in existing_clusters.clusters:
                        final_members.extend(existing_clusters.clusters[old_cluster_id])
                else:
                    # Regular weight
                    final_members.append(member)
            
            if final_members:  # Only keep non-empty clusters
                final_clusters[new_cluster_id] = final_members
                
                # Recompute centroid from actual weights
                cluster_weights = []
                for h in final_members:
                    if h in weights:
                        cluster_weights.append(weights[h])
                
                if cluster_weights:
                    final_centroids[new_cluster_id] = np.mean(cluster_weights, axis=0)
        
        # Update metrics
        combined_result.clusters = final_clusters
        combined_result.centroids = final_centroids
        combined_result.metadata['incremental'] = True
        combined_result.metadata['absorbed_clusters'] = len(centroid_mapping)
        
        return combined_result
    
    def get_dendrogram_data(
        self,
        weights: Dict[str, np.ndarray],
        **kwargs
    ) -> Dict:
        """
        Get dendrogram data for visualization.
        
        Args:
            weights: Dict mapping weight hash to weight array
            **kwargs: Additional arguments for scipy.cluster.hierarchy.dendrogram
            
        Returns:
            Dict with dendrogram data suitable for plotting
        """
        # Prepare data
        data_matrix, hash_list, _ = self._prepare_data(weights)
        
        if self.params['scale_features']:
            scaler = StandardScaler()
            data_matrix = scaler.fit_transform(data_matrix)
        
        # Compute distances and linkage
        distances = pdist(data_matrix, metric=self.params['metric'])
        Z = linkage(distances, method=self.params['linkage_method'])
        
        # Generate dendrogram data
        dend_data = dendrogram(Z, no_plot=True, labels=hash_list, **kwargs)
        
        return dend_data