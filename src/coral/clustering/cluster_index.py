"""
ClusterIndex for maintaining cluster membership and spatial indexing.

This module provides efficient cluster membership tracking and spatial indexing
for the Coral ML clustering system with:
- O(1) weight to cluster lookup
- O(log n) nearest cluster search using spatial indexing
- Thread-safe concurrent operations
- Memory-efficient storage
- Support for incremental updates and rebalancing
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set, Any
import warnings

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean, cosine

logger = logging.getLogger(__name__)


@dataclass
class WeightClusterInfo:
    """Information about a weight's cluster membership."""
    weight_hash: str
    cluster_id: str
    weight_vector: np.ndarray
    distance_to_centroid: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "weight_hash": self.weight_hash,
            "cluster_id": self.cluster_id,
            "weight_vector": self.weight_vector.tolist(),
            "distance_to_centroid": self.distance_to_centroid,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeightClusterInfo":
        """Create from dictionary."""
        return cls(
            weight_hash=data["weight_hash"],
            cluster_id=data["cluster_id"],
            weight_vector=np.array(data["weight_vector"]),
            distance_to_centroid=data.get("distance_to_centroid", 0.0),
        )


@dataclass 
class ClusterCentroid:
    """Represents the centroid of a cluster."""
    cluster_id: str
    centroid_vector: np.ndarray
    member_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cluster_id": self.cluster_id,
            "centroid_vector": self.centroid_vector.tolist(),
            "member_count": self.member_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterCentroid":
        """Create from dictionary."""
        return cls(
            cluster_id=data["cluster_id"],
            centroid_vector=np.array(data["centroid_vector"]),
            member_count=data.get("member_count", 0),
        )


class SimpleLSH:
    """Simple Locality Sensitive Hashing for high-dimensional data."""
    
    def __init__(self, n_projections: int = 10, seed: int = 42):
        """
        Initialize LSH with random projections.
        
        Args:
            n_projections: Number of random projections
            seed: Random seed for reproducibility
        """
        self.n_projections = n_projections
        self.seed = seed
        self.projection_matrix = None
        self.buckets = defaultdict(list)
        self.data_points = []
        self.point_ids = []
        
    def fit(self, data: np.ndarray, ids: List[str]) -> None:
        """
        Fit LSH on data points.
        
        Args:
            data: Array of shape (n_samples, n_features)
            ids: List of identifiers for each data point
        """
        n_samples, n_features = data.shape
        
        # Generate random projection matrix
        np.random.seed(self.seed)
        self.projection_matrix = np.random.randn(self.n_projections, n_features)
        
        # Project data and hash
        projections = data @ self.projection_matrix.T
        
        # Clear existing buckets
        self.buckets.clear()
        self.data_points = data
        self.point_ids = ids
        
        # Hash each point
        for i, (proj, point_id) in enumerate(zip(projections, ids)):
            # Simple binary hash: positive projection = 1, negative = 0
            hash_code = tuple((proj > 0).astype(int))
            self.buckets[hash_code].append(i)
    
    def query(self, query_point: np.ndarray, k: int = 1) -> Tuple[List[int], List[float]]:
        """
        Find k nearest neighbors using LSH.
        
        Args:
            query_point: Query vector
            k: Number of neighbors to find
            
        Returns:
            Tuple of (indices, distances)
        """
        if self.projection_matrix is None:
            return [], []
        
        # Project query point
        projection = query_point @ self.projection_matrix.T
        hash_code = tuple((projection > 0).astype(int))
        
        # Get candidates from same bucket
        candidate_indices = self.buckets.get(hash_code, [])
        
        # If not enough candidates, check nearby buckets
        if len(candidate_indices) < k * 2:
            # Flip each bit and check those buckets too
            for i in range(self.n_projections):
                nearby_hash = list(hash_code)
                nearby_hash[i] = 1 - nearby_hash[i]
                nearby_hash = tuple(nearby_hash)
                candidate_indices.extend(self.buckets.get(nearby_hash, []))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for idx in candidate_indices:
            if idx not in seen:
                seen.add(idx)
                unique_candidates.append(idx)
        candidate_indices = unique_candidates
        
        if not candidate_indices:
            # Fall back to all points if no candidates
            candidate_indices = list(range(len(self.data_points)))
        
        # Compute exact distances to candidates
        candidates = self.data_points[candidate_indices]
        distances = np.linalg.norm(candidates - query_point, axis=1)
        
        # Get k nearest
        k = min(k, len(distances))
        if k == 0:
            return [], []
        
        if k == 1:
            nearest_idx = np.argmin(distances)
            return [candidate_indices[nearest_idx]], [distances[nearest_idx]]
        
        nearest_idx = np.argpartition(distances, k-1)[:k]
        nearest_idx = nearest_idx[np.argsort(distances[nearest_idx])]
        
        # Convert back to original indices
        original_indices = [candidate_indices[i] for i in nearest_idx]
        nearest_distances = distances[nearest_idx]
        
        return original_indices, nearest_distances.tolist()


class ClusterIndex:
    """
    Maintains cluster membership and provides spatial indexing for clusters.
    
    This class tracks which weights belong to which clusters and provides
    efficient nearest cluster search using spatial indexing techniques.
    """
    
    def __init__(self, dimension_threshold: int = 1000):
        """
        Initialize the cluster index.
        
        Args:
            dimension_threshold: Use KD-tree below this dimension, LSH above
        """
        # Core storage
        self._weight_to_cluster: Dict[str, WeightClusterInfo] = {}
        self._cluster_to_weights: Dict[str, Set[str]] = defaultdict(set)
        self._cluster_centroids: Dict[str, ClusterCentroid] = {}
        
        # Spatial indexing
        self._dimension_threshold = dimension_threshold
        self._kdtree: Optional[KDTree] = None
        self._lsh: Optional[SimpleLSH] = None
        self._centroid_array: Optional[np.ndarray] = None
        self._centroid_ids: List[str] = []
        self._index_needs_rebuild = False
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Initialized ClusterIndex with dimension_threshold={dimension_threshold}")
    
    def add_centroid(self, cluster_id: str, centroid_vector: np.ndarray) -> None:
        """
        Add or update a cluster centroid.
        
        Args:
            cluster_id: Unique identifier for the cluster
            centroid_vector: The centroid vector
        """
        with self._lock:
            self._cluster_centroids[cluster_id] = ClusterCentroid(
                cluster_id=cluster_id,
                centroid_vector=centroid_vector.copy(),
                member_count=len(self._cluster_to_weights.get(cluster_id, set()))
            )
            self._index_dirty = True
    
    def add_weight_to_cluster(self, weight_hash: str, cluster_id: str, 
                             weight_vector: np.ndarray) -> None:
        """
        Add a weight to a cluster.
        
        Args:
            weight_hash: Unique identifier for the weight
            cluster_id: Cluster to assign the weight to
            weight_vector: The weight data as a numpy array
            
        Raises:
            ValueError: If weight already assigned to a different cluster
        """
        with self._lock:
            # Check if weight already assigned
            if weight_hash in self._weight_to_cluster:
                existing_cluster = self._weight_to_cluster[weight_hash].cluster_id
                if existing_cluster != cluster_id:
                    raise ValueError(
                        f"Weight {weight_hash} already assigned to cluster {existing_cluster}"
                    )
                return
            
            # Create weight info
            weight_info = WeightClusterInfo(
                weight_hash=weight_hash,
                cluster_id=cluster_id,
                weight_vector=weight_vector.copy()
            )
            
            # Update mappings
            self._weight_to_cluster[weight_hash] = weight_info
            self._cluster_to_weights[cluster_id].add(weight_hash)
            
            # Update or create centroid
            self._update_cluster_centroid(cluster_id)
            
            # Mark index for rebuild
            self._index_needs_rebuild = True
            
            logger.debug(f"Added weight {weight_hash} to cluster {cluster_id}")
    
    def remove_weight_from_cluster(self, weight_hash: str) -> bool:
        """
        Remove a weight from its cluster.
        
        Args:
            weight_hash: Weight to remove
            
        Returns:
            True if removed, False if weight not found
        """
        with self._lock:
            if weight_hash not in self._weight_to_cluster:
                return False
            
            # Get weight info
            weight_info = self._weight_to_cluster[weight_hash]
            cluster_id = weight_info.cluster_id
            
            # Remove from mappings
            del self._weight_to_cluster[weight_hash]
            self._cluster_to_weights[cluster_id].discard(weight_hash)
            
            # Remove empty clusters
            if not self._cluster_to_weights[cluster_id]:
                del self._cluster_to_weights[cluster_id]
                if cluster_id in self._cluster_centroids:
                    del self._cluster_centroids[cluster_id]
            else:
                # Update centroid
                self._update_cluster_centroid(cluster_id)
            
            # Mark index for rebuild
            self._index_needs_rebuild = True
            
            logger.debug(f"Removed weight {weight_hash} from cluster {cluster_id}")
            return True
    
    def find_nearest_cluster(self, weight_vector: np.ndarray, 
                           k: int = 1) -> List[Tuple[str, float]]:
        """
        Find k nearest clusters to a weight vector.
        
        Args:
            weight_vector: Weight data to find nearest clusters for
            k: Number of nearest clusters to return
            
        Returns:
            List of (cluster_id, distance) tuples sorted by distance
        """
        with self._lock:
            if not self._cluster_centroids:
                return []
            
            # Ensure spatial index is built
            self._ensure_spatial_index()
            
            # Flatten weight vector
            weight_flat = weight_vector.flatten()
            
            # Use appropriate spatial index
            if len(weight_flat) < self._dimension_threshold and self._kdtree is not None:
                # Use KD-tree
                k_query = min(k, len(self._centroid_ids))
                distances, indices = self._kdtree.query(weight_flat, k=k_query)
                
                # Handle single result
                if k_query == 1:
                    distances = [distances]
                    indices = [indices]
                
                results = []
                for dist, idx in zip(distances, indices):
                    cluster_id = self._centroid_ids[idx]
                    results.append((cluster_id, float(dist)))
                
            elif self._lsh is not None:
                # Use LSH for high dimensions
                indices, distances = self._lsh.query(weight_flat, k=min(k, len(self._centroid_ids)))
                
                results = []
                for idx, dist in zip(indices, distances):
                    cluster_id = self._centroid_ids[idx]
                    results.append((cluster_id, float(dist)))
                
            else:
                # Fallback to brute force
                results = self._brute_force_nearest(weight_flat, k)
            
            return results
    
    def get_cluster_members(self, cluster_id: str) -> List[str]:
        """
        Get all weights in a cluster.
        
        Args:
            cluster_id: Cluster identifier
            
        Returns:
            List of weight hashes in the cluster
        """
        with self._lock:
            return list(self._cluster_to_weights.get(cluster_id, []))
    
    def rebalance_clusters(self, strategy: str = "size") -> Dict[str, List[str]]:
        """
        Rebalance clusters for efficiency.
        
        Args:
            strategy: Rebalancing strategy ("size", "distance", "hybrid")
            
        Returns:
            Dict mapping cluster_id to list of reassigned weight hashes
        """
        with self._lock:
            reassignments = {}
            
            if strategy == "size":
                reassignments = self._rebalance_by_size()
            elif strategy == "distance":
                reassignments = self._rebalance_by_distance()
            elif strategy == "hybrid":
                # Combine size and distance strategies
                size_reassign = self._rebalance_by_size()
                dist_reassign = self._rebalance_by_distance()
                
                # Merge reassignments
                for cluster_id in set(size_reassign.keys()) | set(dist_reassign.keys()):
                    weights = set(size_reassign.get(cluster_id, []))
                    weights.update(dist_reassign.get(cluster_id, []))
                    if weights:
                        reassignments[cluster_id] = list(weights)
            else:
                logger.warning(f"Unknown rebalancing strategy: {strategy}")
            
            # Apply reassignments
            for cluster_id, weight_hashes in reassignments.items():
                for weight_hash in weight_hashes:
                    if weight_hash in self._weight_to_cluster:
                        weight_info = self._weight_to_cluster[weight_hash]
                        old_cluster = weight_info.cluster_id
                        
                        # Update assignment
                        weight_info.cluster_id = cluster_id
                        self._cluster_to_weights[old_cluster].discard(weight_hash)
                        self._cluster_to_weights[cluster_id].add(weight_hash)
            
            # Update all affected centroids
            affected_clusters = set(reassignments.keys())
            for weights in reassignments.values():
                for weight_hash in weights:
                    if weight_hash in self._weight_to_cluster:
                        old_cluster = self._weight_to_cluster[weight_hash].cluster_id
                        affected_clusters.add(old_cluster)
            
            for cluster_id in affected_clusters:
                if cluster_id in self._cluster_to_weights:
                    self._update_cluster_centroid(cluster_id)
            
            # Mark index for rebuild
            if reassignments:
                self._index_needs_rebuild = True
            
            return reassignments
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert index to dictionary for serialization."""
        with self._lock:
            return {
                "weight_to_cluster": {
                    wh: info.to_dict() 
                    for wh, info in self._weight_to_cluster.items()
                },
                "cluster_centroids": {
                    cid: centroid.to_dict() 
                    for cid, centroid in self._cluster_centroids.items()
                },
                "dimension_threshold": self._dimension_threshold,
            }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterIndex":
        """Create index from dictionary."""
        index = cls(dimension_threshold=data.get("dimension_threshold", 1000))
        
        # Restore weight assignments
        for weight_hash, info_dict in data.get("weight_to_cluster", {}).items():
            weight_info = WeightClusterInfo.from_dict(info_dict)
            index._weight_to_cluster[weight_hash] = weight_info
            index._cluster_to_weights[weight_info.cluster_id].add(weight_hash)
        
        # Restore centroids
        for cluster_id, centroid_dict in data.get("cluster_centroids", {}).items():
            index._cluster_centroids[cluster_id] = ClusterCentroid.from_dict(centroid_dict)
        
        # Mark for rebuild
        index._index_needs_rebuild = True
        
        return index
    
    # Private helper methods
    
    def _update_cluster_centroid(self, cluster_id: str) -> None:
        """Update the centroid for a cluster based on its members."""
        members = self._cluster_to_weights.get(cluster_id, set())
        if not members:
            return
        
        # Compute mean of all member vectors
        vectors = []
        for weight_hash in members:
            if weight_hash in self._weight_to_cluster:
                vectors.append(self._weight_to_cluster[weight_hash].weight_vector)
        
        if vectors:
            centroid_vector = np.mean(vectors, axis=0)
            self._cluster_centroids[cluster_id] = ClusterCentroid(
                cluster_id=cluster_id,
                centroid_vector=centroid_vector,
                member_count=len(vectors)
            )
    
    def _ensure_spatial_index(self) -> None:
        """Ensure spatial index is built and up-to-date."""
        if not self._index_needs_rebuild:
            return
        
        if not self._cluster_centroids:
            return
        
        # Build centroid array
        self._centroid_ids = list(self._cluster_centroids.keys())
        centroid_vectors = [
            self._cluster_centroids[cid].centroid_vector 
            for cid in self._centroid_ids
        ]
        self._centroid_array = np.array(centroid_vectors)
        
        # Choose indexing method based on dimensionality
        n_features = self._centroid_array.shape[1]
        
        if n_features < self._dimension_threshold:
            # Use KD-tree for low dimensions
            try:
                self._kdtree = KDTree(self._centroid_array)
                self._lsh = None
                logger.debug(f"Built KD-tree index with {len(self._centroid_ids)} centroids")
            except Exception as e:
                logger.warning(f"Failed to build KD-tree: {e}, falling back to brute force")
                self._kdtree = None
                self._lsh = None
        else:
            # Use LSH for high dimensions
            try:
                self._lsh = SimpleLSH(n_projections=min(20, n_features // 50))
                self._lsh.fit(self._centroid_array, self._centroid_ids)
                self._kdtree = None
                logger.debug(f"Built LSH index with {len(self._centroid_ids)} centroids")
            except Exception as e:
                logger.warning(f"Failed to build LSH: {e}, falling back to brute force")
                self._kdtree = None
                self._lsh = None
        
        self._index_needs_rebuild = False
    
    def _brute_force_nearest(self, weight_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Brute force search for nearest clusters."""
        distances = []
        
        for cluster_id, centroid in self._cluster_centroids.items():
            dist = euclidean(weight_vector, centroid.centroid_vector.flatten())
            distances.append((cluster_id, dist))
        
        # Sort by distance and return top k
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def _rebalance_by_size(self) -> Dict[str, List[str]]:
        """Rebalance clusters to have more uniform sizes."""
        if not self._cluster_centroids:
            return {}
        
        # Calculate target size
        total_weights = len(self._weight_to_cluster)
        n_clusters = len(self._cluster_centroids)
        if n_clusters == 0:
            return {}
        target_size = max(1, total_weights // n_clusters)
        
        reassignments = defaultdict(list)
        
        # Find oversized clusters
        oversized = []
        undersized = []
        
        for cluster_id, centroid in self._cluster_centroids.items():
            size = len(self._cluster_to_weights[cluster_id])
            # Use more flexible thresholds for small clusters
            if size > target_size + 1:  # More than 1 over target
                oversized.append((cluster_id, size))
            elif size < target_size:  # Under target
                undersized.append((cluster_id, size))
        
        # Move weights from oversized to undersized clusters
        for over_cluster, over_size in oversized:
            if not undersized:
                break
                
            # Get weights sorted by distance to centroid
            weights = list(self._cluster_to_weights[over_cluster])
            centroid = self._cluster_centroids[over_cluster].centroid_vector
            
            weight_distances = []
            for weight_hash in weights:
                weight_vec = self._weight_to_cluster[weight_hash].weight_vector
                dist = euclidean(weight_vec.flatten(), centroid.flatten())
                weight_distances.append((weight_hash, dist))
            
            # Sort by distance (furthest first for reassignment)
            weight_distances.sort(key=lambda x: x[1], reverse=True)
            
            # Reassign furthest weights
            n_to_move = max(1, min(len(weight_distances) // 3, over_size - target_size))
            
            for i in range(n_to_move):
                if not undersized:
                    break
                    
                weight_hash, _ = weight_distances[i]
                weight_vec = self._weight_to_cluster[weight_hash].weight_vector
                
                # Find best undersized cluster
                best_cluster = None
                best_dist = float('inf')
                
                for under_cluster, under_size in undersized:
                    under_centroid = self._cluster_centroids[under_cluster].centroid_vector
                    dist = euclidean(weight_vec.flatten(), under_centroid.flatten())
                    if dist < best_dist:
                        best_dist = dist
                        best_cluster = under_cluster
                
                if best_cluster:
                    reassignments[best_cluster].append(weight_hash)
                    
                    # Update undersized list
                    new_size = len(self._cluster_to_weights[best_cluster]) + len(reassignments[best_cluster])
                    if new_size >= target_size:
                        undersized = [(c, s) for c, s in undersized if c != best_cluster]
        
        return dict(reassignments)
    
    def _rebalance_by_distance(self) -> Dict[str, List[str]]:
        """Rebalance by moving weights to their nearest clusters."""
        reassignments = defaultdict(list)
        
        # Check each weight's distance to its assigned cluster vs others
        for weight_hash, weight_info in self._weight_to_cluster.items():
            current_cluster = weight_info.cluster_id
            weight_vec = weight_info.weight_vector
            
            # Find nearest cluster
            nearest_clusters = self.find_nearest_cluster(weight_vec, k=2)
            if not nearest_clusters:
                continue
            
            nearest_cluster, nearest_dist = nearest_clusters[0]
            
            # If nearest cluster is different and significantly closer
            if nearest_cluster != current_cluster:
                current_centroid = self._cluster_centroids[current_cluster].centroid_vector
                current_dist = euclidean(weight_vec.flatten(), current_centroid.flatten())
                
                # Reassign if new cluster is >20% closer
                if nearest_dist < current_dist * 0.8:
                    reassignments[nearest_cluster].append(weight_hash)
        
        return dict(reassignments)
    
    def get_cluster_count(self) -> int:
        """Get the number of clusters."""
        with self._lock:
            return len(self._cluster_centroids)
    
    def get_weight_count(self) -> int:
        """Get the total number of weights."""
        with self._lock:
            return len(self._weight_to_cluster)
    
    def get_cluster_sizes(self) -> Dict[str, int]:
        """Get the size of each cluster."""
        with self._lock:
            return {
                cluster_id: len(members) 
                for cluster_id, members in self._cluster_to_weights.items()
            }
    
    def clear(self) -> None:
        """Clear all data from the index."""
        with self._lock:
            self._weight_to_cluster.clear()
            self._cluster_to_weights.clear()
            self._cluster_centroids.clear()
            self._kdtree = None
            self._lsh = None
            self._centroid_array = None
            self._centroid_ids = []
            self._index_needs_rebuild = False
            logger.info("Cleared cluster index")