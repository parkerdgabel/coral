"""
ClusterIndex for fast centroid storage and lookup operations.

This module provides efficient storage and retrieval of cluster centroids with:
- Fast O(log n) nearest neighbor search using spatial indexing
- Hierarchical cluster navigation
- Thread-safe concurrent operations
- Memory-efficient batch operations
- Performance monitoring and optimization
"""

import logging
import threading
import time
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Any, Set, Union
import warnings

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cosine, euclidean
try:
    from sklearn.neighbors import LSHForest
except ImportError:
    # LSHForest has been deprecated, use NearestNeighbors as fallback
    LSHForest = None
import xxhash

from .cluster_types import Centroid, ClusterInfo, ClusterLevel
from ..core.weight_tensor import WeightTensor

logger = logging.getLogger(__name__)


class IndexStats:
    """Statistics and performance metrics for the cluster index."""
    
    def __init__(self):
        self.total_centroids = 0
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.avg_query_time = 0.0
        self.memory_usage_mb = 0.0
        self.index_build_time = 0.0
        self.last_optimization = None
        
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_centroids": self.total_centroids,
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "avg_query_time": self.avg_query_time,
            "memory_usage_mb": self.memory_usage_mb,
            "index_build_time": self.index_build_time,
            "last_optimization": self.last_optimization,
        }


class ClusterIndex:
    """
    Efficient index for cluster centroids with fast lookup operations.
    
    Provides:
    - O(1) centroid retrieval by cluster ID
    - O(log n) nearest neighbor search via spatial indexing
    - Hierarchical cluster navigation
    - Thread-safe concurrent operations
    - Memory-efficient batch operations
    """
    
    def __init__(self, 
                 spatial_index_type: str = "kdtree",
                 cache_size: int = 1000,
                 enable_lsh: bool = False,
                 lsh_n_estimators: int = 10,
                 lsh_n_candidates: int = 50):
        """
        Initialize the cluster index.
        
        Args:
            spatial_index_type: Type of spatial index ("kdtree", "lsh", "brute")
            cache_size: Size of LRU cache for frequent lookups
            enable_lsh: Enable LSH for approximate nearest neighbor search
            lsh_n_estimators: Number of LSH estimators
            lsh_n_candidates: Number of LSH candidates to consider
        """
        # Core storage
        self._centroids: Dict[str, Centroid] = {}
        self._cluster_info: Dict[str, ClusterInfo] = {}
        
        # Spatial indexing
        self._spatial_index_type = spatial_index_type
        self._kdtree: Optional[KDTree] = None
        self._lsh_forest: Optional[LSHForest] = None
        self._enable_lsh = enable_lsh
        self._lsh_n_estimators = lsh_n_estimators
        self._lsh_n_candidates = lsh_n_candidates
        
        # Index maintenance
        self._centroid_vectors: Optional[np.ndarray] = None
        self._centroid_ids: List[str] = []
        self._index_needs_rebuild = False
        
        # Hierarchical structure
        self._level_centroids: Dict[ClusterLevel, Set[str]] = defaultdict(set)
        self._parent_child_map: Dict[str, Set[str]] = defaultdict(set)
        self._child_parent_map: Dict[str, str] = {}
        
        # Performance tracking
        self._stats = IndexStats()
        self._centroid_usage: Counter = Counter()
        self._query_cache: Dict[str, Tuple[str, float]] = {}  # LRU cache
        self._cache_size = cache_size
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(
            f"Initialized ClusterIndex with spatial_index_type={spatial_index_type}, "
            f"cache_size={cache_size}, enable_lsh={enable_lsh}"
        )
    
    def add_centroid(self, centroid: Centroid, cluster_info: Optional[ClusterInfo] = None) -> None:
        """
        Add a centroid to the index.
        
        Args:
            centroid: The centroid to add
            cluster_info: Optional metadata about the cluster
            
        Raises:
            ValueError: If centroid with same cluster_id already exists
        """
        with self._lock:
            if centroid.cluster_id in self._centroids:
                raise ValueError(f"Centroid with cluster_id '{centroid.cluster_id}' already exists")
            
            # Store centroid
            self._centroids[centroid.cluster_id] = centroid
            
            # Store cluster info if provided
            if cluster_info:
                self._cluster_info[centroid.cluster_id] = cluster_info
                self._level_centroids[cluster_info.level].add(centroid.cluster_id)
            
            # Mark index for rebuild
            self._index_needs_rebuild = True
            self._stats.total_centroids += 1
            
            logger.debug(f"Added centroid for cluster {centroid.cluster_id}")
    
    def get_centroid(self, cluster_id: str) -> Optional[Centroid]:
        """
        Retrieve a centroid by cluster ID.
        
        Args:
            cluster_id: The cluster identifier
            
        Returns:
            The centroid if found, None otherwise
        """
        with self._lock:
            centroid = self._centroids.get(cluster_id)
            if centroid:
                self._centroid_usage[cluster_id] += 1
                self._stats.cache_hits += 1
            else:
                self._stats.cache_misses += 1
            return centroid
    
    def update_centroid(self, cluster_id: str, new_centroid: Centroid) -> bool:
        """
        Update an existing centroid.
        
        Args:
            cluster_id: The cluster identifier
            new_centroid: The new centroid data
            
        Returns:
            True if updated, False if centroid not found
        """
        with self._lock:
            if cluster_id not in self._centroids:
                return False
            
            # Update centroid (ensure cluster_id matches)
            new_centroid.cluster_id = cluster_id
            self._centroids[cluster_id] = new_centroid
            
            # Mark index for rebuild
            self._index_needs_rebuild = True
            
            # Clear cache entries for this centroid
            self._invalidate_cache_for_centroid(cluster_id)
            
            logger.debug(f"Updated centroid for cluster {cluster_id}")
            return True
    
    def remove_centroid(self, cluster_id: str) -> bool:
        """
        Remove a centroid from the index.
        
        Args:
            cluster_id: The cluster identifier
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if cluster_id not in self._centroids:
                return False
            
            # Remove centroid and cluster info
            del self._centroids[cluster_id]
            if cluster_id in self._cluster_info:
                cluster_info = self._cluster_info[cluster_id]
                self._level_centroids[cluster_info.level].discard(cluster_id)
                del self._cluster_info[cluster_id]
            
            # Remove from hierarchical maps
            if cluster_id in self._parent_child_map:
                children = self._parent_child_map[cluster_id]
                for child in children:
                    if child in self._child_parent_map:
                        del self._child_parent_map[child]
                del self._parent_child_map[cluster_id]
            
            if cluster_id in self._child_parent_map:
                parent = self._child_parent_map[cluster_id]
                if parent in self._parent_child_map:
                    self._parent_child_map[parent].discard(cluster_id)
                del self._child_parent_map[cluster_id]
            
            # Mark index for rebuild
            self._index_needs_rebuild = True
            self._stats.total_centroids -= 1
            
            # Clear cache entries for this centroid
            self._invalidate_cache_for_centroid(cluster_id)
            
            logger.debug(f"Removed centroid for cluster {cluster_id}")
            return True
    
    def find_nearest_centroid(self, weight: Union[WeightTensor, np.ndarray]) -> Optional[Tuple[str, float]]:
        """
        Find the nearest centroid to a weight tensor.
        
        Args:
            weight: Weight tensor or numpy array to find nearest centroid for
            
        Returns:
            Tuple of (cluster_id, distance) if found, None if no centroids
        """
        start_time = time.time()
        
        with self._lock:
            if not self._centroids:
                return None
            
            # Extract data from weight
            if isinstance(weight, WeightTensor):
                weight_data = weight.data.flatten()
                weight_hash = weight.compute_hash()
            else:
                weight_data = weight.flatten()
                weight_hash = self._compute_array_hash(weight)
            
            # Check cache first
            if weight_hash in self._query_cache:
                result = self._query_cache[weight_hash]
                self._stats.cache_hits += 1
                self._stats.total_queries += 1
                query_time = time.time() - start_time
                self._update_avg_query_time(query_time)
                return result
            
            # Ensure spatial index is built
            self._ensure_spatial_index()
            
            # Find nearest neighbor
            if self._spatial_index_type == "kdtree" and self._kdtree is not None:
                distances, indices = self._kdtree.query(weight_data.reshape(1, -1), k=1)
                nearest_idx = indices[0]
                distance = distances[0]
                cluster_id = self._centroid_ids[nearest_idx]
            
            elif self._spatial_index_type == "lsh" and self._lsh_forest is not None:
                # LSH approximate nearest neighbor
                distances, indices = self._lsh_forest.kneighbors(
                    weight_data.reshape(1, -1), 
                    n_neighbors=1
                )
                if len(indices[0]) > 0:
                    nearest_idx = indices[0][0]
                    distance = distances[0][0]
                    cluster_id = self._centroid_ids[nearest_idx]
                else:
                    # Fallback to brute force
                    cluster_id, distance = self._brute_force_nearest(weight_data)
            
            else:
                # Brute force search
                cluster_id, distance = self._brute_force_nearest(weight_data)
            
            result = (cluster_id, float(distance))
            
            # Update cache (with LRU eviction)
            self._update_cache(weight_hash, result)
            
            # Update stats
            self._stats.cache_misses += 1
            self._stats.total_queries += 1
            self._centroid_usage[cluster_id] += 1
            
            query_time = time.time() - start_time
            self._update_avg_query_time(query_time)
            
            return result
    
    def find_similar_centroids(self, 
                             weight: Union[WeightTensor, np.ndarray], 
                             threshold: float) -> List[Tuple[str, float]]:
        """
        Find all centroids within similarity threshold of a weight.
        
        Args:
            weight: Weight tensor or numpy array
            threshold: Similarity threshold (distance threshold)
            
        Returns:
            List of (cluster_id, distance) tuples for similar centroids
        """
        start_time = time.time()
        
        with self._lock:
            if not self._centroids:
                return []
            
            # Extract data from weight
            if isinstance(weight, WeightTensor):
                weight_data = weight.data.flatten()
            else:
                weight_data = weight.flatten()
            
            # Ensure spatial index is built
            self._ensure_spatial_index()
            
            results = []
            
            if self._spatial_index_type == "kdtree" and self._kdtree is not None:
                # Query within radius
                indices = self._kdtree.query_ball_point(weight_data, r=threshold)
                for idx in indices:
                    cluster_id = self._centroid_ids[idx]
                    centroid = self._centroids[cluster_id]
                    distance = euclidean(weight_data, centroid.data.flatten())
                    results.append((cluster_id, distance))
            
            else:
                # Brute force search within threshold
                for cluster_id, centroid in self._centroids.items():
                    distance = euclidean(weight_data, centroid.data.flatten())
                    if distance <= threshold:
                        results.append((cluster_id, distance))
            
            # Sort by distance
            results.sort(key=lambda x: x[1])
            
            # Update stats
            self._stats.total_queries += 1
            query_time = time.time() - start_time
            self._update_avg_query_time(query_time)
            
            return results
    
    def batch_lookup(self, weights: List[Union[WeightTensor, np.ndarray]]) -> List[Optional[Tuple[str, float]]]:
        """
        Efficiently lookup nearest centroids for multiple weights.
        
        Args:
            weights: List of weight tensors or numpy arrays
            
        Returns:
            List of (cluster_id, distance) tuples, same order as input
        """
        start_time = time.time()
        
        with self._lock:
            if not self._centroids:
                return [None] * len(weights)
            
            # Ensure spatial index is built
            self._ensure_spatial_index()
            
            # Prepare weight data matrix
            weight_matrix = []
            for weight in weights:
                if isinstance(weight, WeightTensor):
                    weight_matrix.append(weight.data.flatten())
                else:
                    weight_matrix.append(weight.flatten())
            
            weight_matrix = np.array(weight_matrix)
            results = []
            
            if self._spatial_index_type == "kdtree" and self._kdtree is not None:
                # Batch query
                distances, indices = self._kdtree.query(weight_matrix, k=1)
                for i, (distance, idx) in enumerate(zip(distances, indices)):
                    cluster_id = self._centroid_ids[idx]
                    results.append((cluster_id, float(distance)))
                    self._centroid_usage[cluster_id] += 1
            
            else:
                # Individual lookups
                for weight in weights:
                    result = self.find_nearest_centroid(weight)
                    results.append(result)
            
            # Update stats
            self._stats.total_queries += len(weights)
            query_time = time.time() - start_time
            self._update_avg_query_time(query_time)
            
            return results
    
    def get_centroids_by_level(self, cluster_level: ClusterLevel) -> List[Centroid]:
        """
        Retrieve all centroids at a specific hierarchy level.
        
        Args:
            cluster_level: The hierarchical level
            
        Returns:
            List of centroids at the specified level
        """
        with self._lock:
            cluster_ids = self._level_centroids.get(cluster_level, set())
            return [self._centroids[cid] for cid in cluster_ids if cid in self._centroids]
    
    def build_hierarchy(self, cluster_relationships: Dict[str, Dict[str, Any]]) -> None:
        """
        Build hierarchical index structure from cluster relationships.
        
        Args:
            cluster_relationships: Dict mapping cluster_id to relationship info
                Format: {cluster_id: {"parent": parent_id, "children": [child_ids]}}
        """
        with self._lock:
            # Clear existing hierarchy
            self._parent_child_map.clear()
            self._child_parent_map.clear()
            
            # Build hierarchy from relationships
            for cluster_id, relationships in cluster_relationships.items():
                if cluster_id not in self._centroids:
                    continue
                
                # Set parent relationship
                parent_id = relationships.get("parent")
                if parent_id and parent_id in self._centroids:
                    self._child_parent_map[cluster_id] = parent_id
                    self._parent_child_map[parent_id].add(cluster_id)
                
                # Set children relationships
                children = relationships.get("children", [])
                for child_id in children:
                    if child_id in self._centroids:
                        self._parent_child_map[cluster_id].add(child_id)
                        self._child_parent_map[child_id] = cluster_id
            
            logger.info(f"Built hierarchy with {len(self._parent_child_map)} parent nodes")
    
    def get_parent_centroid(self, cluster_id: str) -> Optional[Centroid]:
        """
        Get the parent centroid in the hierarchy.
        
        Args:
            cluster_id: Child cluster identifier
            
        Returns:
            Parent centroid if exists, None otherwise
        """
        with self._lock:
            parent_id = self._child_parent_map.get(cluster_id)
            return self._centroids.get(parent_id) if parent_id else None
    
    def get_child_centroids(self, cluster_id: str) -> List[Centroid]:
        """
        Get all child centroids in the hierarchy.
        
        Args:
            cluster_id: Parent cluster identifier
            
        Returns:
            List of child centroids
        """
        with self._lock:
            child_ids = self._parent_child_map.get(cluster_id, set())
            return [self._centroids[cid] for cid in child_ids if cid in self._centroids]
    
    def find_level_centroids(self, level: ClusterLevel, weight: Union[WeightTensor, np.ndarray]) -> List[Tuple[str, float]]:
        """
        Find centroids at a specific hierarchy level closest to a weight.
        
        Args:
            level: Hierarchical level to search
            weight: Weight tensor to find centroids for
            
        Returns:
            List of (cluster_id, distance) tuples sorted by distance
        """
        with self._lock:
            level_centroids = self.get_centroids_by_level(level)
            if not level_centroids:
                return []
            
            # Extract weight data
            if isinstance(weight, WeightTensor):
                weight_data = weight.data.flatten()
            else:
                weight_data = weight.flatten()
            
            # Calculate distances to all centroids at this level
            results = []
            for centroid in level_centroids:
                distance = euclidean(weight_data, centroid.data.flatten())
                results.append((centroid.cluster_id, distance))
            
            # Sort by distance
            results.sort(key=lambda x: x[1])
            return results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive index statistics.
        
        Returns:
            Dictionary containing performance and usage statistics
        """
        with self._lock:
            # Update memory usage estimate
            self._update_memory_usage()
            
            stats = self._stats.to_dict()
            
            # Add additional statistics
            stats.update({
                "spatial_index_type": self._spatial_index_type,
                "enable_lsh": self._enable_lsh,
                "cache_size": self._cache_size,
                "current_cache_entries": len(self._query_cache),
                "hierarchy_depth": self._calculate_hierarchy_depth(),
                "centroids_by_level": {
                    level.value: len(cluster_ids) 
                    for level, cluster_ids in self._level_centroids.items()
                },
                "index_needs_rebuild": self._index_needs_rebuild,
            })
            
            return stats
    
    def get_centroid_usage(self) -> Dict[str, int]:
        """
        Get usage statistics for each centroid.
        
        Returns:
            Dictionary mapping cluster_id to usage count
        """
        with self._lock:
            return dict(self._centroid_usage)
    
    def optimize_index(self) -> None:
        """
        Rebuild and optimize the spatial index for better performance.
        """
        start_time = time.time()
        
        with self._lock:
            logger.info("Starting index optimization...")
            
            # Force rebuild spatial index
            self._build_spatial_index()
            
            # Optimize cache by removing least used entries
            self._optimize_cache()
            
            # Update stats
            self._stats.index_build_time = time.time() - start_time
            self._stats.last_optimization = time.time()
            
            logger.info(f"Index optimization completed in {self._stats.index_build_time:.3f}s")
    
    def validate_index(self) -> Dict[str, Any]:
        """
        Validate index consistency and return validation results.
        
        Returns:
            Dictionary containing validation results and any issues found
        """
        with self._lock:
            issues = []
            
            # Check centroid-cluster_info consistency
            for cluster_id in self._centroids:
                if cluster_id in self._cluster_info:
                    cluster_info = self._cluster_info[cluster_id]
                    if cluster_id not in self._level_centroids[cluster_info.level]:
                        issues.append(f"Centroid {cluster_id} not in level index")
            
            # Check hierarchy consistency
            for child_id, parent_id in self._child_parent_map.items():
                if parent_id not in self._parent_child_map:
                    issues.append(f"Parent {parent_id} missing from parent-child map")
                elif child_id not in self._parent_child_map[parent_id]:
                    issues.append(f"Child {child_id} not in parent's children list")
            
            # Check spatial index consistency
            if self._centroid_vectors is not None and len(self._centroid_ids) != len(self._centroids):
                issues.append("Spatial index out of sync with centroids")
            
            return {
                "is_valid": len(issues) == 0,
                "issues": issues,
                "total_centroids": len(self._centroids),
                "total_cluster_info": len(self._cluster_info),
                "hierarchy_nodes": len(self._parent_child_map),
                "cache_entries": len(self._query_cache),
            }
    
    # Private helper methods
    
    def _ensure_spatial_index(self) -> None:
        """Ensure spatial index is built and up-to-date."""
        if self._index_needs_rebuild or self._kdtree is None:
            self._build_spatial_index()
    
    def _build_spatial_index(self) -> None:
        """Build or rebuild the spatial index."""
        if not self._centroids:
            return
        
        start_time = time.time()
        
        # Collect centroid vectors and IDs  
        vectors = []
        ids = []
        
        for cluster_id, centroid in self._centroids.items():
            vectors.append(centroid.data.flatten())
            ids.append(cluster_id)
        
        self._centroid_vectors = np.array(vectors)
        self._centroid_ids = ids
        
        # Build spatial index based on type
        if self._spatial_index_type == "kdtree":
            self._kdtree = KDTree(self._centroid_vectors)
        
        elif self._spatial_index_type == "lsh" and self._enable_lsh:
            try:
                # LSHForest is deprecated, using NearestNeighbors as approximate method
                # In practice, you might want to use alternatives like Annoy or Faiss
                from sklearn.neighbors import NearestNeighbors
                self._lsh_forest = NearestNeighbors(
                    n_neighbors=min(self._lsh_n_candidates, len(self._centroids)),
                    algorithm='ball_tree'
                )
                self._lsh_forest.fit(self._centroid_vectors)
            except (ImportError, Exception):
                logger.warning("LSH not available, falling back to KDTree")
                self._kdtree = KDTree(self._centroid_vectors)
                self._spatial_index_type = "kdtree"
        
        self._index_needs_rebuild = False
        build_time = time.time() - start_time
        self._stats.index_build_time = build_time
        
        logger.debug(f"Built {self._spatial_index_type} index with {len(vectors)} centroids in {build_time:.3f}s")
    
    def _brute_force_nearest(self, weight_data: np.ndarray) -> Tuple[str, float]:
        """Brute force nearest neighbor search."""
        min_distance = float('inf')
        nearest_id = None
        
        for cluster_id, centroid in self._centroids.items():
            distance = euclidean(weight_data, centroid.data.flatten())
            if distance < min_distance:
                min_distance = distance
                nearest_id = cluster_id
        
        return nearest_id, min_distance
    
    def _update_cache(self, key: str, value: Tuple[str, float]) -> None:
        """Update LRU cache with new entry."""
        if len(self._query_cache) >= self._cache_size:
            # Remove oldest entry (simple FIFO, could be improved to true LRU)
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        
        self._query_cache[key] = value
    
    def _invalidate_cache_for_centroid(self, cluster_id: str) -> None:
        """Remove cache entries that reference a specific centroid."""
        keys_to_remove = []
        for key, (cached_cluster_id, _) in self._query_cache.items():
            if cached_cluster_id == cluster_id:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._query_cache[key]
    
    def _optimize_cache(self) -> None:
        """Optimize cache by removing least used entries."""
        if len(self._query_cache) <= self._cache_size // 2:
            return
        
        # Keep most frequently used centroids in cache
        # This is a simple heuristic - could be improved
        frequently_used = set(self._centroid_usage.most_common(self._cache_size // 2))
        frequently_used_ids = {cluster_id for cluster_id, _ in frequently_used}
        
        optimized_cache = {}
        for key, (cluster_id, distance) in self._query_cache.items():
            if cluster_id in frequently_used_ids:
                optimized_cache[key] = (cluster_id, distance)
        
        self._query_cache = optimized_cache
    
    def _update_avg_query_time(self, query_time: float) -> None:
        """Update exponential moving average of query time."""
        alpha = 0.1  # Smoothing factor
        if self._stats.avg_query_time == 0:
            self._stats.avg_query_time = query_time
        else:
            self._stats.avg_query_time = (
                alpha * query_time + (1 - alpha) * self._stats.avg_query_time
            )
    
    def _update_memory_usage(self) -> None:
        """Estimate current memory usage."""
        memory_bytes = 0
        
        # Centroid storage
        for centroid in self._centroids.values():
            memory_bytes += centroid.nbytes
        
        # Spatial index
        if self._centroid_vectors is not None:
            memory_bytes += self._centroid_vectors.nbytes
        
        # Cache and maps (rough estimate)
        memory_bytes += len(self._query_cache) * 100  # Rough estimate
        memory_bytes += len(self._parent_child_map) * 50
        memory_bytes += len(self._child_parent_map) * 50
        
        self._stats.memory_usage_mb = memory_bytes / (1024 * 1024)
    
    def _calculate_hierarchy_depth(self) -> int:
        """Calculate maximum depth of hierarchy."""
        if not self._child_parent_map:
            return 0
        
        max_depth = 0
        
        # Find root nodes (nodes with no parents)
        roots = set(self._centroids.keys()) - set(self._child_parent_map.keys())
        
        for root in roots:
            depth = self._calculate_node_depth(root, 0)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _calculate_node_depth(self, node_id: str, current_depth: int) -> int:
        """Recursively calculate depth from a node."""
        children = self._parent_child_map.get(node_id, set())
        if not children:
            return current_depth
        
        max_child_depth = current_depth
        for child_id in children:
            child_depth = self._calculate_node_depth(child_id, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def _compute_array_hash(self, array: np.ndarray) -> str:
        """Compute hash for numpy array."""
        hasher = xxhash.xxh3_64()
        hasher.update(array.tobytes())
        return hasher.hexdigest()
    
    def __len__(self) -> int:
        """Return number of centroids in the index."""
        return len(self._centroids)
    
    def __contains__(self, cluster_id: str) -> bool:
        """Check if cluster_id exists in the index."""
        return cluster_id in self._centroids
    
    def __repr__(self) -> str:
        """String representation of the index."""
        return (
            f"ClusterIndex(centroids={len(self._centroids)}, "
            f"spatial_index={self._spatial_index_type}, "
            f"cache_size={self._cache_size})"
        )