"""DBSCAN clustering strategy implementation."""

import numpy as np
from typing import Dict, Optional, Tuple
import logging
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from .base import ClusteringStrategy, ClusteringResult

logger = logging.getLogger(__name__)


class DBSCANStrategy(ClusteringStrategy):
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) strategy.
    
    Excellent for finding clusters of arbitrary shape and identifying outliers.
    Automatically determines the number of clusters based on data density.
    
    Parameters:
        eps: Maximum distance between points in same neighborhood (default: auto)
        min_samples: Minimum points required to form a dense region (default: 5)
        metric: Distance metric ('euclidean', 'cosine', 'manhattan')
        scale_features: Whether to scale features (default: True)
        auto_eps: Automatically tune eps parameter (default: True)
        eps_quantile: Quantile for auto eps selection (default: 0.3)
        algorithm: Algorithm for nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute')
    
    Example:
        >>> # Automatic eps tuning
        >>> strategy = DBSCANStrategy(min_samples=3, auto_eps=True)
        >>> result = strategy.cluster(weights)
        >>> print(f"Found {result.n_clusters} clusters and {len(result.noise_points)} outliers")
        
        >>> # Manual eps setting
        >>> strategy = DBSCANStrategy(eps=0.5, min_samples=5)
        >>> result = strategy.cluster(weights)
    """
    
    @property
    def name(self) -> str:
        """Strategy name."""
        return "dbscan"
    
    def validate_params(self) -> None:
        """Validate strategy parameters."""
        # Set defaults
        self.params.setdefault('eps', None)
        self.params.setdefault('min_samples', 5)
        self.params.setdefault('metric', 'euclidean')
        self.params.setdefault('scale_features', True)
        self.params.setdefault('auto_eps', True)
        self.params.setdefault('eps_quantile', 0.3)
        self.params.setdefault('algorithm', 'auto')
        
        # Validate
        if self.params['eps'] is not None and self.params['eps'] <= 0:
            raise ValueError("eps must be > 0")
        
        if self.params['min_samples'] < 1:
            raise ValueError("min_samples must be >= 1")
        
        valid_metrics = ['euclidean', 'cosine', 'manhattan', 'chebyshev', 'minkowski']
        if self.params['metric'] not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}")
        
        if self.params['eps_quantile'] <= 0 or self.params['eps_quantile'] >= 1:
            raise ValueError("eps_quantile must be in (0, 1)")
        
        valid_algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
        if self.params['algorithm'] not in valid_algorithms:
            raise ValueError(f"algorithm must be one of {valid_algorithms}")
    
    def cluster(
        self,
        weights: Dict[str, np.ndarray],
        existing_clusters: Optional[ClusteringResult] = None
    ) -> ClusteringResult:
        """
        Perform DBSCAN clustering on weights.
        
        Args:
            weights: Dict mapping weight hash to weight array
            existing_clusters: Optional existing clusters for incremental clustering
            
        Returns:
            ClusteringResult with clusters, centroids, and metrics
        """
        if len(weights) == 0:
            return ClusteringResult({}, {}, {}, {'error': 'No weights to cluster'})
        
        n_samples = len(weights)
        
        # Adjust min_samples if necessary
        min_samples = min(self.params['min_samples'], n_samples)
        
        # Single sample case
        if n_samples == 1:
            weight_hash = list(weights.keys())[0]
            return ClusteringResult(
                clusters={},
                centroids={},
                metrics={'n_clusters': 0, 'n_samples': 1, 'n_noise_points': 1},
                metadata={'single_sample': True},
                noise_points={weight_hash}
            )
        
        # Prepare data
        data_matrix, hash_list, shape_groups = self._prepare_data(weights)
        
        # Scale features if requested
        if self.params['scale_features']:
            scaler = StandardScaler()
            data_matrix = scaler.fit_transform(data_matrix)
        
        # Determine eps if auto-tuning is enabled
        if self.params['auto_eps'] and self.params['eps'] is None:
            eps = self._auto_tune_eps(data_matrix, min_samples)
            logger.info(f"Auto-tuned eps: {eps:.4f}")
        else:
            eps = self.params['eps'] if self.params['eps'] is not None else 0.5
        
        # Perform DBSCAN clustering
        logger.info(f"Performing DBSCAN with eps={eps:.4f}, min_samples={min_samples}")
        
        try:
            dbscan = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric=self.params['metric'],
                algorithm=self.params['algorithm'],
                n_jobs=-1  # Use all available cores
            )
            labels = dbscan.fit_predict(data_matrix)
        except Exception as e:
            logger.error(f"DBSCAN clustering failed: {e}")
            return ClusteringResult({}, {}, {}, {'error': str(e)})
        
        # Convert labels to clusters
        clusters, noise_points = self._labels_to_clusters(labels, hash_list)
        
        # Log clustering results
        n_clusters = len(clusters)
        n_noise = len(noise_points)
        logger.info(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
        
        # Compute centroids only for actual clusters
        centroids = self._compute_centroids(weights, clusters)
        
        # Compute metrics
        metrics = self._compute_metrics(data_matrix, labels, centroids)
        metrics['eps'] = float(eps)
        metrics['min_samples'] = int(min_samples)
        
        # Add density statistics
        if n_clusters > 0:
            cluster_sizes = [len(members) for members in clusters.values()]
            metrics['avg_cluster_size'] = float(np.mean(cluster_sizes))
            metrics['std_cluster_size'] = float(np.std(cluster_sizes))
            metrics['min_cluster_size'] = int(np.min(cluster_sizes))
            metrics['max_cluster_size'] = int(np.max(cluster_sizes))
        
        # Compute noise ratio
        metrics['noise_ratio'] = n_noise / n_samples if n_samples > 0 else 0.0
        
        # Store metadata
        metadata = {
            'algorithm': 'dbscan',
            'metric': self.params['metric'],
            'shape_groups': shape_groups,
            'scaled': self.params['scale_features'],
            'auto_eps': self.params['auto_eps'] and self.params['eps'] is None,
            'params': self.params.copy()
        }
        
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
    
    def _auto_tune_eps(
        self,
        data_matrix: np.ndarray,
        min_samples: int
    ) -> float:
        """
        Automatically tune eps parameter using k-distance graph method.
        
        Args:
            data_matrix: Feature matrix
            min_samples: Minimum samples parameter
            
        Returns:
            Suggested eps value
        """
        # Compute k-nearest neighbors
        k = min_samples
        nbrs = NearestNeighbors(n_neighbors=k, metric=self.params['metric'])
        nbrs.fit(data_matrix)
        
        # Get k-distances
        distances, _ = nbrs.kneighbors(data_matrix)
        k_distances = distances[:, -1]  # Distance to k-th nearest neighbor
        
        # Sort distances
        k_distances = np.sort(k_distances)
        
        # Use quantile to determine eps
        eps = np.quantile(k_distances, self.params['eps_quantile'])
        
        # Ensure eps is reasonable
        if eps <= 0:
            eps = np.mean(k_distances)
        
        # Add some buffer
        eps *= 1.1
        
        return float(eps)
    
    def _merge_with_existing(
        self,
        new_result: ClusteringResult,
        existing_clusters: ClusteringResult,
        weights: Dict[str, np.ndarray]
    ) -> ClusteringResult:
        """
        Merge with existing clusters using density-based approach.
        
        For DBSCAN, we identify which new points should join existing clusters
        based on density criteria.
        """
        # If no existing clusters, just return new result
        if not existing_clusters.clusters:
            return new_result
        
        # Build a unified view of all points
        all_clusters = {}
        all_noise = set()
        
        # Start with existing clusters
        cluster_id_offset = len(existing_clusters.clusters)
        for old_id, members in existing_clusters.clusters.items():
            all_clusters[old_id] = members.copy()
        
        # Add existing noise points
        all_noise.update(existing_clusters.noise_points)
        
        # Process new clusters
        for new_id, members in new_result.clusters.items():
            # Check if any member is close to existing clusters
            merged = False
            
            for member in members:
                if member not in weights:
                    continue
                
                member_weight = weights[member]
                
                # Check distance to existing cluster centroids
                for old_id, centroid in existing_clusters.centroids.items():
                    if member_weight.shape != centroid.shape:
                        continue
                    
                    # Compute distance
                    if self.params['metric'] == 'euclidean':
                        dist = np.linalg.norm(member_weight - centroid)
                    elif self.params['metric'] == 'cosine':
                        from sklearn.metrics.pairwise import cosine_similarity
                        dist = 1 - cosine_similarity(
                            member_weight.reshape(1, -1),
                            centroid.reshape(1, -1)
                        )[0, 0]
                    else:
                        # Fallback to euclidean
                        dist = np.linalg.norm(member_weight - centroid)
                    
                    # If within eps distance, merge with existing cluster
                    eps = self.params.get('eps', new_result.metrics.get('eps', 0.5))
                    if dist <= eps:
                        all_clusters[old_id].extend(members)
                        merged = True
                        break
                
                if merged:
                    break
            
            # If not merged, create new cluster
            if not merged:
                all_clusters[cluster_id_offset + new_id] = members
        
        # Add new noise points
        all_noise.update(new_result.noise_points)
        
        # Some noise points might now be close enough to clusters
        noise_to_remove = set()
        for noise_hash in all_noise:
            if noise_hash not in weights:
                continue
            
            noise_weight = weights[noise_hash]
            
            # Check if close to any cluster centroid
            for cluster_id, centroid in existing_clusters.centroids.items():
                if noise_weight.shape != centroid.shape:
                    continue
                
                # Compute distance
                if self.params['metric'] == 'euclidean':
                    dist = np.linalg.norm(noise_weight - centroid)
                else:
                    dist = np.linalg.norm(noise_weight - centroid)  # Fallback
                
                eps = self.params.get('eps', new_result.metrics.get('eps', 0.5))
                if dist <= eps * 1.5:  # Slightly larger threshold for noise recovery
                    all_clusters[cluster_id].append(noise_hash)
                    noise_to_remove.add(noise_hash)
                    break
        
        all_noise -= noise_to_remove
        
        # Recompute centroids for modified clusters
        all_centroids = {}
        for cluster_id, members in all_clusters.items():
            if cluster_id in existing_clusters.centroids and cluster_id not in new_result.clusters:
                # Unchanged existing cluster
                all_centroids[cluster_id] = existing_clusters.centroids[cluster_id]
            else:
                # Modified or new cluster - recompute centroid
                cluster_weights = [weights[h] for h in members if h in weights]
                if cluster_weights:
                    all_centroids[cluster_id] = np.mean(cluster_weights, axis=0)
        
        # Update metrics
        n_total = sum(len(members) for members in all_clusters.values()) + len(all_noise)
        metrics = new_result.metrics.copy()
        metrics['n_clusters'] = len(all_clusters)
        metrics['n_noise_points'] = len(all_noise)
        metrics['noise_ratio'] = len(all_noise) / n_total if n_total > 0 else 0.0
        
        return ClusteringResult(
            clusters=all_clusters,
            centroids=all_centroids,
            metrics=metrics,
            metadata={**new_result.metadata, 'incremental': True},
            noise_points=all_noise
        )