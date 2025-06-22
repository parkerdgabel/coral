"""K-Means clustering strategy implementation."""

import numpy as np
from typing import Dict, Optional
import logging
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from .base import ClusteringStrategy, ClusteringResult

logger = logging.getLogger(__name__)


class KMeansStrategy(ClusteringStrategy):
    """
    K-Means clustering strategy.
    
    Supports both standard K-Means and Mini-Batch K-Means for large datasets.
    
    Parameters:
        n_clusters: Number of clusters (default: 8)
        max_iter: Maximum iterations (default: 300)
        tol: Convergence tolerance (default: 1e-4)
        n_init: Number of initializations (default: 10)
        use_minibatch: Use mini-batch variant for large datasets (default: auto)
        batch_size: Batch size for mini-batch (default: 1024)
        random_state: Random seed (default: 42)
        scale_features: Whether to scale features (default: True)
    
    Example:
        >>> strategy = KMeansStrategy(n_clusters=10)
        >>> result = strategy.cluster(weights)
        >>> print(f"Found {result.n_clusters} clusters")
    """
    
    @property
    def name(self) -> str:
        """Strategy name."""
        return "kmeans"
    
    def validate_params(self) -> None:
        """Validate strategy parameters."""
        # Set defaults
        self.params.setdefault('n_clusters', 8)
        self.params.setdefault('max_iter', 300)
        self.params.setdefault('tol', 1e-4)
        self.params.setdefault('n_init', 10)
        self.params.setdefault('use_minibatch', 'auto')
        self.params.setdefault('batch_size', 1024)
        self.params.setdefault('random_state', 42)
        self.params.setdefault('scale_features', True)
        
        # Validate
        if self.params['n_clusters'] < 1:
            raise ValueError("n_clusters must be >= 1")
        
        if self.params['max_iter'] < 1:
            raise ValueError("max_iter must be >= 1")
        
        if self.params['tol'] <= 0:
            raise ValueError("tol must be > 0")
        
        if self.params['use_minibatch'] not in ['auto', True, False]:
            raise ValueError("use_minibatch must be 'auto', True, or False")
    
    def cluster(
        self,
        weights: Dict[str, np.ndarray],
        existing_clusters: Optional[ClusteringResult] = None
    ) -> ClusteringResult:
        """
        Perform K-Means clustering on weights.
        
        Args:
            weights: Dict mapping weight hash to weight array
            existing_clusters: Optional existing clusters for incremental clustering
            
        Returns:
            ClusteringResult with clusters, centroids, and metrics
        """
        if len(weights) == 0:
            return ClusteringResult({}, {}, {}, {'error': 'No weights to cluster'})
        
        # Prepare data
        data_matrix, hash_list, shape_groups = self._prepare_data(weights)
        
        # Determine number of clusters
        n_samples = len(weights)
        n_clusters = min(self.params['n_clusters'], n_samples)
        
        if n_clusters < 2:
            # Single cluster case
            return ClusteringResult(
                clusters={0: list(weights.keys())},
                centroids={0: np.mean(list(weights.values()), axis=0)},
                metrics={'n_clusters': 1, 'n_samples': n_samples},
                metadata={'shape_groups': shape_groups}
            )
        
        # Scale features if requested
        if self.params['scale_features']:
            scaler = StandardScaler()
            data_matrix = scaler.fit_transform(data_matrix)
        
        # Choose K-Means variant
        use_minibatch = self.params['use_minibatch']
        if use_minibatch == 'auto':
            use_minibatch = n_samples > 10000
        
        if use_minibatch:
            logger.info(f"Using MiniBatchKMeans with {n_clusters} clusters")
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                max_iter=self.params['max_iter'],
                tol=self.params['tol'],
                batch_size=self.params['batch_size'],
                random_state=self.params['random_state'],
                n_init=min(self.params['n_init'], 3)  # Fewer inits for mini-batch
            )
        else:
            logger.info(f"Using standard KMeans with {n_clusters} clusters")
            kmeans = KMeans(
                n_clusters=n_clusters,
                max_iter=self.params['max_iter'],
                tol=self.params['tol'],
                n_init=self.params['n_init'],
                random_state=self.params['random_state']
            )
        
        # Fit the model
        try:
            labels = kmeans.fit_predict(data_matrix)
        except Exception as e:
            logger.error(f"K-Means clustering failed: {e}")
            return ClusteringResult({}, {}, {}, {'error': str(e)})
        
        # Handle empty clusters
        unique_labels = np.unique(labels)
        if len(unique_labels) < n_clusters:
            logger.warning(f"Only {len(unique_labels)} non-empty clusters out of {n_clusters}")
        
        # Convert labels to clusters
        clusters, noise_points = self._labels_to_clusters(labels, hash_list)
        
        # Compute centroids in original weight space
        centroids = self._compute_centroids(weights, clusters)
        
        # Compute metrics
        metrics = self._compute_metrics(data_matrix, labels, centroids)
        metrics['convergence_iterations'] = kmeans.n_iter_
        metrics['inertia'] = float(kmeans.inertia_)
        
        # Add metadata
        metadata = {
            'algorithm': 'minibatch_kmeans' if use_minibatch else 'kmeans',
            'shape_groups': shape_groups,
            'scaled': self.params['scale_features'],
            'params': self.params.copy()
        }
        
        result = ClusteringResult(
            clusters=clusters,
            centroids=centroids,
            metrics=metrics,
            metadata=metadata,
            noise_points=noise_points
        )
        
        # Handle incremental clustering if existing clusters provided
        if existing_clusters:
            result = self._merge_with_existing(result, existing_clusters, weights)
        
        return result
    
    def _merge_with_existing(
        self,
        new_result: ClusteringResult,
        existing_clusters: ClusteringResult,
        weights: Dict[str, np.ndarray]
    ) -> ClusteringResult:
        """
        Merge with existing clusters by reassigning points to nearest centroids.
        
        This implements a simple incremental K-Means approach.
        """
        # Combine centroids
        all_centroids = {}
        centroid_id = 0
        
        # Add existing centroids
        for old_id, centroid in existing_clusters.centroids.items():
            all_centroids[centroid_id] = centroid
            centroid_id += 1
        
        # Add new centroids
        for new_id, centroid in new_result.centroids.items():
            all_centroids[centroid_id] = centroid
            centroid_id += 1
        
        # Limit total clusters
        if len(all_centroids) > self.params['n_clusters']:
            # Keep only the largest clusters
            cluster_sizes = {}
            for cid, centroid in all_centroids.items():
                if cid < len(existing_clusters.centroids):
                    # Existing cluster
                    old_id = list(existing_clusters.centroids.keys())[cid]
                    size = len(existing_clusters.clusters.get(old_id, []))
                else:
                    # New cluster
                    new_id = list(new_result.centroids.keys())[cid - len(existing_clusters.centroids)]
                    size = len(new_result.clusters.get(new_id, []))
                cluster_sizes[cid] = size
            
            # Keep top n_clusters
            top_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
            top_cluster_ids = [cid for cid, _ in top_clusters[:self.params['n_clusters']]]
            all_centroids = {i: all_centroids[cid] for i, cid in enumerate(top_cluster_ids)}
        
        # Reassign all points to nearest centroid
        all_weights = {}
        all_weights.update(weights)
        
        # Add weights from existing clusters if available
        for cluster_members in existing_clusters.clusters.values():
            for weight_hash in cluster_members:
                if weight_hash not in all_weights:
                    # We would need access to the weight store here
                    # For now, skip weights we don't have
                    logger.warning(f"Cannot access weight {weight_hash} for re-clustering")
        
        # Create new clustering with combined centroids
        result = self._cluster_with_fixed_centroids(all_weights, all_centroids)
        
        return result
    
    def _cluster_with_fixed_centroids(
        self,
        weights: Dict[str, np.ndarray],
        centroids: Dict[int, np.ndarray]
    ) -> ClusteringResult:
        """Assign weights to nearest centroids."""
        clusters = {cid: [] for cid in centroids}
        
        for weight_hash, weight_array in weights.items():
            # Find nearest centroid
            min_dist = float('inf')
            best_cluster = 0
            
            for cid, centroid in centroids.items():
                if weight_array.shape != centroid.shape:
                    continue
                
                dist = np.linalg.norm(weight_array - centroid)
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = cid
            
            clusters[best_cluster].append(weight_hash)
        
        # Remove empty clusters
        clusters = {cid: members for cid, members in clusters.items() if members}
        centroids = {cid: centroid for cid, centroid in centroids.items() if cid in clusters}
        
        # Compute metrics
        data_matrix, hash_list, _ = self._prepare_data(weights)
        labels = np.array([
            next(cid for cid, members in clusters.items() if h in members)
            for h in hash_list
        ])
        
        metrics = self._compute_metrics(data_matrix, labels, centroids)
        
        return ClusteringResult(
            clusters=clusters,
            centroids=centroids,
            metrics=metrics,
            metadata={'incremental': True}
        )