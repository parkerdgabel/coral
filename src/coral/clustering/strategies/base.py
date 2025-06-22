"""Base clustering strategy interface and utilities."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Result of a clustering operation."""
    
    clusters: Dict[int, List[str]]  # cluster_id -> list of weight hashes
    centroids: Dict[int, np.ndarray]  # cluster_id -> centroid array
    metrics: Dict[str, float]  # clustering quality metrics
    metadata: Dict[str, Any] = field(default_factory=dict)  # strategy-specific metadata
    noise_points: Set[str] = field(default_factory=set)  # points not assigned to any cluster
    
    @property
    def n_clusters(self) -> int:
        """Number of clusters formed."""
        return len(self.clusters)
    
    @property
    def total_points(self) -> int:
        """Total number of points clustered."""
        return sum(len(members) for members in self.clusters.values()) + len(self.noise_points)
    
    def get_cluster_sizes(self) -> Dict[int, int]:
        """Get size of each cluster."""
        return {cluster_id: len(members) for cluster_id, members in self.clusters.items()}


class ClusteringStrategy(ABC):
    """Abstract base class for clustering strategies."""
    
    def __init__(self, **params):
        """Initialize strategy with parameters."""
        self.params = params
        self.validate_params()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass
    
    @abstractmethod
    def validate_params(self) -> None:
        """Validate strategy parameters."""
        pass
    
    @abstractmethod
    def cluster(
        self,
        weights: Dict[str, np.ndarray],
        existing_clusters: Optional[ClusteringResult] = None
    ) -> ClusteringResult:
        """
        Perform clustering on weights.
        
        Args:
            weights: Dict mapping weight hash to weight array
            existing_clusters: Optional existing clusters for incremental clustering
            
        Returns:
            ClusteringResult with clusters, centroids, and metrics
        """
        pass
    
    def _prepare_data(
        self,
        weights: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, List[str], Dict[Tuple[int, ...], List[int]]]:
        """
        Prepare weight data for clustering.
        
        Groups weights by shape and flattens them for clustering.
        
        Returns:
            data_matrix: Flattened weight data (n_samples, n_features)
            hash_list: List of hashes corresponding to rows in data_matrix
            shape_groups: Dict mapping shapes to indices in data_matrix
        """
        # Group weights by shape
        shape_groups = defaultdict(list)
        hash_list = []
        data_rows = []
        
        for idx, (weight_hash, weight_array) in enumerate(weights.items()):
            shape = weight_array.shape
            flat_weight = weight_array.flatten()
            
            shape_groups[shape].append(idx)
            hash_list.append(weight_hash)
            data_rows.append(flat_weight)
        
        # Pad arrays to same length if needed
        if len(set(arr.shape[0] for arr in data_rows)) > 1:
            max_len = max(arr.shape[0] for arr in data_rows)
            data_rows = [
                np.pad(arr, (0, max_len - arr.shape[0]), mode='constant')
                for arr in data_rows
            ]
        
        data_matrix = np.vstack(data_rows)
        return data_matrix, hash_list, dict(shape_groups)
    
    def _compute_centroids(
        self,
        weights: Dict[str, np.ndarray],
        clusters: Dict[int, List[str]]
    ) -> Dict[int, np.ndarray]:
        """Compute centroid for each cluster."""
        centroids = {}
        
        for cluster_id, members in clusters.items():
            if not members:
                continue
                
            # Get weights for cluster members
            cluster_weights = [weights[h] for h in members if h in weights]
            if not cluster_weights:
                continue
            
            # All weights in a cluster should have the same shape
            shape = cluster_weights[0].shape
            if not all(w.shape == shape for w in cluster_weights):
                # Handle mixed shapes by averaging each shape group separately
                shape_groups = defaultdict(list)
                for w in cluster_weights:
                    shape_groups[w.shape].append(w)
                
                # Use the most common shape's centroid
                largest_group = max(shape_groups.values(), key=len)
                centroid = np.mean(largest_group, axis=0)
            else:
                # Simple case: all same shape
                centroid = np.mean(cluster_weights, axis=0)
            
            centroids[cluster_id] = centroid
        
        return centroids
    
    def _compute_metrics(
        self,
        data_matrix: np.ndarray,
        labels: np.ndarray,
        centroids: Optional[Dict[int, np.ndarray]] = None
    ) -> Dict[str, float]:
        """Compute clustering quality metrics."""
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        
        metrics = {}
        
        # Filter out noise points (-1 label)
        valid_mask = labels >= 0
        if np.sum(valid_mask) < 2:
            return metrics
        
        valid_data = data_matrix[valid_mask]
        valid_labels = labels[valid_mask]
        
        n_clusters = len(np.unique(valid_labels))
        
        if n_clusters > 1 and len(valid_labels) > n_clusters:
            try:
                metrics['silhouette_score'] = float(silhouette_score(valid_data, valid_labels))
                metrics['davies_bouldin_score'] = float(davies_bouldin_score(valid_data, valid_labels))
                metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(valid_data, valid_labels))
            except Exception as e:
                logger.warning(f"Failed to compute clustering metrics: {e}")
        
        # Compute inertia if centroids provided
        if centroids:
            inertia = 0.0
            for idx, label in enumerate(valid_labels):
                if label in centroids:
                    centroid_flat = centroids[label].flatten()
                    point = valid_data[idx]
                    # Ensure same length
                    min_len = min(len(centroid_flat), len(point))
                    inertia += np.sum((point[:min_len] - centroid_flat[:min_len]) ** 2)
            metrics['inertia'] = float(inertia)
        
        metrics['n_clusters'] = n_clusters
        metrics['n_noise_points'] = int(np.sum(labels == -1))
        
        return metrics
    
    def _labels_to_clusters(
        self,
        labels: np.ndarray,
        hash_list: List[str]
    ) -> Tuple[Dict[int, List[str]], Set[str]]:
        """Convert label array to cluster dictionary."""
        clusters = defaultdict(list)
        noise_points = set()
        
        for idx, label in enumerate(labels):
            weight_hash = hash_list[idx]
            if label == -1:  # Noise point
                noise_points.add(weight_hash)
            else:
                clusters[int(label)].append(weight_hash)
        
        return dict(clusters), noise_points
    
    def _merge_with_existing(
        self,
        new_result: ClusteringResult,
        existing_clusters: ClusteringResult,
        weights: Dict[str, np.ndarray]
    ) -> ClusteringResult:
        """
        Merge new clustering results with existing clusters.
        
        This is a simple strategy that can be overridden by specific implementations.
        """
        # For now, just return the new result
        # Subclasses can implement more sophisticated merging
        return new_result