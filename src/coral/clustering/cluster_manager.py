"""
High-level cluster manager that coordinates all clustering components.

This module provides a unified interface for managing the entire clustering
lifecycle, from analysis through optimization. It coordinates:
- ClusterAnalyzer for weight analysis and clustering
- ClusterStorage for persistence
- ClusterIndex for efficient lookups
- CentroidEncoder for compression
- ClusterOptimizer for continuous improvement
"""

import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field

import numpy as np

from .cluster_analyzer import ClusterAnalyzer
from .cluster_index import ClusterIndex
from .cluster_storage import ClusterStorage
from .centroid_encoder import CentroidEncoder
from .cluster_optimizer import ClusterOptimizer
from .cluster_config import ClusteringConfig, OptimizationConfig
from .cluster_types import (
    ClusteringStrategy, ClusterLevel, ClusterMetrics,
    Centroid, ClusterAssignment, ClusteringResult
)
from ..core.weight_tensor import WeightTensor
from ..delta.delta_encoder import DeltaEncoder

logger = logging.getLogger(__name__)


@dataclass
class ClusteringStats:
    """Statistics for clustering operations."""
    total_weights: int = 0
    total_clusters: int = 0
    weights_clustered: int = 0
    compression_ratio: float = 1.0
    space_saved: int = 0
    clustering_time: float = 0.0
    last_optimization: Optional[float] = None
    optimization_count: int = 0


@dataclass 
class ClusterLifecycleEvent:
    """Event in cluster lifecycle."""
    timestamp: float
    event_type: str  # created, updated, merged, split, deleted
    cluster_id: str
    details: Dict[str, Any] = field(default_factory=dict)


class ClusterManager:
    """
    High-level manager for clustering operations.
    
    Coordinates all clustering components to provide:
    - Full lifecycle management (create, update, delete, merge)
    - Performance optimization and caching
    - Thread-safe operations
    - Monitoring and statistics
    """
    
    def __init__(
        self,
        storage_path: str,
        delta_encoder: Optional[DeltaEncoder] = None,
        cache_size: int = 1000,
        enable_monitoring: bool = True
    ):
        """
        Initialize cluster manager.
        
        Args:
            storage_path: Path to cluster storage
            delta_encoder: Optional delta encoder for centroid compression
            cache_size: Size of centroid cache
            enable_monitoring: Whether to track detailed statistics
        """
        self.storage_path = Path(storage_path)
        self.delta_encoder = delta_encoder
        self.cache_size = cache_size
        self.enable_monitoring = enable_monitoring
        
        # Initialize components
        self.analyzer = ClusterAnalyzer()
        self.storage = ClusterStorage(str(storage_path))
        self.index = ClusterIndex(storage=self.storage)
        self.encoder = CentroidEncoder(
            storage=self.storage,
            delta_encoder=delta_encoder
        )
        self.optimizer = ClusterOptimizer(storage=self.storage)
        
        # Thread safety
        self._lock = threading.RLock()
        self._operation_lock = threading.Lock()
        
        # Caching
        self._centroid_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Statistics and monitoring
        self.stats = ClusteringStats()
        self._lifecycle_events = []
        
        # Initialize from existing storage
        self._initialize_from_storage()
        
    def _initialize_from_storage(self) -> None:
        """Initialize manager state from existing storage."""
        with self._lock:
            try:
                with self.storage:
                    # Load existing clusters
                    cluster_ids = self.storage.list_clusters()
                    self.stats.total_clusters = len(cluster_ids)
                    
                    # Count assignments
                    total_assignments = 0
                    for cluster_id in cluster_ids:
                        assignments = self.storage.get_cluster_assignments(cluster_id)
                        total_assignments += len(assignments)
                    
                    self.stats.weights_clustered = total_assignments
                    
                    # Build index if clusters exist
                    if cluster_ids:
                        centroids = []
                        for cluster_id in cluster_ids[:100]:  # Limit for initial load
                            centroid = self.storage.load_centroid(cluster_id)
                            if centroid:
                                centroids.append(centroid)
                        
                        if centroids:
                            self.index.build_index(centroids)
                            
            except Exception as e:
                logger.warning(f"Failed to initialize from storage: {e}")
                
    def cluster_weights(
        self,
        weights: List[WeightTensor],
        config: Optional[ClusteringConfig] = None,
        progress_callback: Optional[callable] = None
    ) -> ClusteringResult:
        """
        Cluster a collection of weights.
        
        Args:
            weights: List of weight tensors to cluster
            config: Clustering configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            ClusteringResult with assignments and metrics
        """
        if not weights:
            return ClusteringResult(
                assignments=[],
                centroids=[],
                metrics=ClusterMetrics(num_clusters=0),
                strategy=config.strategy if config else ClusteringStrategy.ADAPTIVE
            )
            
        start_time = time.time()
        
        with self._operation_lock:
            # Use analyzer to perform clustering
            result = self.analyzer.cluster_weights(weights, config)
            
            if result and result.is_valid():
                # Store results
                with self.storage:
                    self.storage.store_clustering_result(result)
                    
                # Update index
                self.index.add_centroids(result.centroids)
                
                # Update statistics
                self.stats.total_weights += len(weights)
                self.stats.weights_clustered += len(result.assignments)
                self.stats.total_clusters = len(self.storage.list_clusters())
                
                # Calculate compression
                original_size = sum(w.data.nbytes for w in weights)
                centroid_size = sum(c.data.nbytes for c in result.centroids)
                if original_size > 0:
                    self.stats.compression_ratio = original_size / centroid_size
                    self.stats.space_saved = original_size - centroid_size
                    
                self.stats.clustering_time += time.time() - start_time
                
                # Log lifecycle events
                if self.enable_monitoring:
                    for centroid in result.centroids:
                        self._log_event(
                            "created",
                            centroid.cluster_id,
                            {"size": len([a for a in result.assignments 
                                         if a.cluster_id == centroid.cluster_id])}
                        )
                        
            return result
            
    def optimize_clusters(
        self,
        config: Optional[OptimizationConfig] = None,
        target_clusters: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Optimize existing clusters.
        
        Args:
            config: Optimization configuration  
            target_clusters: Specific clusters to optimize (all if None)
            
        Returns:
            Optimization results
        """
        start_time = time.time()
        
        with self._operation_lock:
            # Get clusters to optimize
            if target_clusters:
                cluster_ids = target_clusters
            else:
                cluster_ids = self.storage.list_clusters()
                
            # Load centroids
            centroids = []
            with self.storage:
                for cluster_id in cluster_ids:
                    centroid = self._get_centroid_cached(cluster_id)
                    if centroid:
                        centroids.append(centroid)
                        
            if not centroids:
                return {
                    "clusters_optimized": 0,
                    "time_elapsed": 0.0
                }
                
            # Optimize
            result = self.optimizer.optimize(centroids, config)
            
            if result:
                # Update statistics
                self.stats.optimization_count += 1
                self.stats.last_optimization = time.time()
                
                # Log events
                if self.enable_monitoring:
                    self._log_event(
                        "optimization",
                        "batch",
                        {
                            "clusters_merged": result.clusters_merged,
                            "clusters_split": result.clusters_split,
                            "compression_ratio": result.compression_ratio
                        }
                    )
                    
                # Clear cache for updated clusters
                self._invalidate_cache()
                
                return {
                    "clusters_optimized": len(result.optimized_centroids),
                    "clusters_merged": result.clusters_merged,
                    "clusters_split": result.clusters_split,
                    "compression_ratio": result.compression_ratio,
                    "time_elapsed": time.time() - start_time
                }
            else:
                return {
                    "clusters_optimized": 0,
                    "time_elapsed": time.time() - start_time
                }
                
    def assign_weight(
        self,
        weight: WeightTensor,
        similarity_threshold: float = 0.95
    ) -> Optional[ClusterAssignment]:
        """
        Assign a weight to the nearest cluster.
        
        Args:
            weight: Weight tensor to assign
            similarity_threshold: Minimum similarity for assignment
            
        Returns:
            ClusterAssignment if suitable cluster found
        """
        with self._lock:
            # Find nearest centroid
            nearest = self.index.find_nearest(
                weight.data,
                k=1,
                similarity_threshold=similarity_threshold
            )
            
            if nearest:
                cluster_id, similarity = nearest[0]
                
                # Create assignment
                assignment = ClusterAssignment(
                    weight_name=weight.metadata.name if weight.metadata else "",
                    weight_hash=weight.compute_hash(),
                    cluster_id=cluster_id,
                    distance_to_centroid=1.0 - similarity,
                    similarity_score=similarity
                )
                
                # Store assignment
                with self.storage:
                    self.storage.store_assignment(assignment)
                    
                # Update stats
                self.stats.weights_clustered += 1
                
                return assignment
                
        return None
        
    def create_cluster(
        self,
        weights: List[WeightTensor],
        cluster_id: Optional[str] = None
    ) -> Centroid:
        """
        Create a new cluster from weights.
        
        Args:
            weights: Weights to form cluster
            cluster_id: Optional cluster ID
            
        Returns:
            Created centroid
        """
        if not weights:
            raise ValueError("Cannot create cluster from empty weights")
            
        with self._operation_lock:
            # Calculate centroid
            centroid_data = np.mean([w.data for w in weights], axis=0)
            
            # Create centroid
            if cluster_id is None:
                cluster_id = f"cluster_{int(time.time() * 1000)}"
                
            centroid = Centroid(
                cluster_id=cluster_id,
                data=centroid_data,
                shape=centroid_data.shape,
                dtype=centroid_data.dtype
            )
            
            # Store centroid
            with self.storage:
                self.storage.store_centroid(centroid)
                
                # Create assignments
                for weight in weights:
                    assignment = ClusterAssignment(
                        weight_name=weight.metadata.name if weight.metadata else "",
                        weight_hash=weight.compute_hash(),
                        cluster_id=cluster_id,
                        distance_to_centroid=0.0,
                        similarity_score=1.0
                    )
                    self.storage.store_assignment(assignment)
                    
            # Update index
            self.index.add_centroids([centroid])
            
            # Update stats
            self.stats.total_clusters += 1
            self.stats.weights_clustered += len(weights)
            
            # Log event
            if self.enable_monitoring:
                self._log_event(
                    "created",
                    cluster_id,
                    {"size": len(weights)}
                )
                
            return centroid
            
    def merge_clusters(
        self,
        cluster_ids: List[str],
        new_cluster_id: Optional[str] = None
    ) -> Optional[Centroid]:
        """
        Merge multiple clusters into one.
        
        Args:
            cluster_ids: IDs of clusters to merge
            new_cluster_id: Optional ID for merged cluster
            
        Returns:
            Merged centroid if successful
        """
        if len(cluster_ids) < 2:
            return None
            
        with self._operation_lock:
            # Load centroids and assignments
            centroids = []
            all_assignments = []
            
            with self.storage:
                for cluster_id in cluster_ids:
                    centroid = self._get_centroid_cached(cluster_id)
                    if centroid:
                        centroids.append(centroid)
                        assignments = self.storage.get_cluster_assignments(cluster_id)
                        all_assignments.extend(assignments)
                        
            if len(centroids) < 2:
                return None
                
            # Calculate merged centroid
            total_weight = 0
            weighted_sum = None
            
            for i, centroid in enumerate(centroids):
                cluster_size = len([a for a in all_assignments 
                                  if a.cluster_id == centroid.cluster_id])
                if weighted_sum is None:
                    weighted_sum = centroid.data * cluster_size
                else:
                    weighted_sum += centroid.data * cluster_size
                total_weight += cluster_size
                
            merged_data = weighted_sum / total_weight
            
            # Create new centroid
            if new_cluster_id is None:
                new_cluster_id = f"merged_{int(time.time() * 1000)}"
                
            merged_centroid = Centroid(
                cluster_id=new_cluster_id,
                data=merged_data,
                shape=merged_data.shape,
                dtype=merged_data.dtype
            )
            
            with self.storage:
                # Store new centroid
                self.storage.store_centroid(merged_centroid)
                
                # Update assignments
                for assignment in all_assignments:
                    assignment.cluster_id = new_cluster_id
                    self.storage.store_assignment(assignment)
                    
                # Delete old clusters
                for cluster_id in cluster_ids:
                    self.storage.delete_cluster(cluster_id)
                    
            # Update index
            self.index.remove_centroids(cluster_ids)
            self.index.add_centroids([merged_centroid])
            
            # Update stats
            self.stats.total_clusters -= len(cluster_ids) - 1
            
            # Log event
            if self.enable_monitoring:
                self._log_event(
                    "merged",
                    new_cluster_id,
                    {"source_clusters": cluster_ids, "size": len(all_assignments)}
                )
                
            # Invalidate cache
            for cluster_id in cluster_ids:
                self._invalidate_cache(cluster_id)
                
            return merged_centroid
            
    def split_cluster(
        self,
        cluster_id: str,
        n_splits: int = 2,
        config: Optional[ClusteringConfig] = None
    ) -> List[Centroid]:
        """
        Split a cluster into multiple smaller clusters.
        
        Args:
            cluster_id: ID of cluster to split
            n_splits: Number of splits
            config: Clustering configuration for splitting
            
        Returns:
            List of new centroids
        """
        with self._operation_lock:
            # Get cluster assignments
            with self.storage:
                assignments = self.storage.get_cluster_assignments(cluster_id)
                if len(assignments) < n_splits:
                    return []
                    
                # Load weights for assignments
                weights = []
                for assignment in assignments:
                    # This would need access to weight storage
                    # For now, we'll return empty
                    pass
                    
            # Would implement actual splitting logic here
            # For now, return empty list
            return []
            
    def delete_cluster(self, cluster_id: str) -> bool:
        """
        Delete a cluster and its assignments.
        
        Args:
            cluster_id: ID of cluster to delete
            
        Returns:
            True if successful
        """
        with self._operation_lock:
            with self.storage:
                # Get assignments before deletion
                assignments = self.storage.get_cluster_assignments(cluster_id)
                
                # Delete cluster
                success = self.storage.delete_cluster(cluster_id)
                
                if success:
                    # Update index
                    self.index.remove_centroids([cluster_id])
                    
                    # Update stats
                    self.stats.total_clusters -= 1
                    self.stats.weights_clustered -= len(assignments)
                    
                    # Log event
                    if self.enable_monitoring:
                        self._log_event(
                            "deleted",
                            cluster_id,
                            {"size": len(assignments)}
                        )
                        
                    # Invalidate cache
                    self._invalidate_cache(cluster_id)
                    
                return success
                
    def get_cluster_info(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a cluster.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            Cluster information dictionary
        """
        with self._lock:
            centroid = self._get_centroid_cached(cluster_id)
            if not centroid:
                return None
                
            with self.storage:
                assignments = self.storage.get_cluster_assignments(cluster_id)
                
            return {
                "cluster_id": cluster_id,
                "size": len(assignments),
                "centroid_shape": centroid.shape,
                "centroid_dtype": str(centroid.dtype),
                "quality_score": getattr(centroid, 'quality_score', 0.0),
                "created": getattr(centroid, 'created_at', None),
                "assignments": len(assignments),
                "compression_ratio": getattr(centroid, 'compression_ratio', 1.0)
            }
            
    def get_statistics(self) -> ClusteringStats:
        """Get current clustering statistics."""
        return self.stats
        
    def get_lifecycle_events(
        self,
        event_type: Optional[str] = None,
        cluster_id: Optional[str] = None,
        limit: int = 100
    ) -> List[ClusterLifecycleEvent]:
        """
        Get lifecycle events.
        
        Args:
            event_type: Filter by event type
            cluster_id: Filter by cluster ID
            limit: Maximum events to return
            
        Returns:
            List of lifecycle events
        """
        events = self._lifecycle_events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
            
        if cluster_id:
            events = [e for e in events if e.cluster_id == cluster_id]
            
        return events[-limit:]
        
    def _get_centroid_cached(self, cluster_id: str) -> Optional[Centroid]:
        """Get centroid with caching."""
        # Check cache
        if cluster_id in self._centroid_cache:
            self._cache_hits += 1
            return self._centroid_cache[cluster_id]
            
        # Load from storage
        self._cache_misses += 1
        with self.storage:
            centroid = self.storage.load_centroid(cluster_id)
            
        if centroid and len(self._centroid_cache) < self.cache_size:
            self._centroid_cache[cluster_id] = centroid
            
        return centroid
        
    def _invalidate_cache(self, cluster_id: Optional[str] = None) -> None:
        """Invalidate cache entries."""
        if cluster_id:
            self._centroid_cache.pop(cluster_id, None)
        else:
            self._centroid_cache.clear()
            
    def _log_event(
        self,
        event_type: str,
        cluster_id: str,
        details: Dict[str, Any]
    ) -> None:
        """Log a lifecycle event."""
        event = ClusterLifecycleEvent(
            timestamp=time.time(),
            event_type=event_type,
            cluster_id=cluster_id,
            details=details
        )
        
        self._lifecycle_events.append(event)
        
        # Limit event history
        if len(self._lifecycle_events) > 10000:
            self._lifecycle_events = self._lifecycle_events[-5000:]
            
    def close(self) -> None:
        """Close manager and release resources."""
        with self._lock:
            if hasattr(self.storage, 'close'):
                self.storage.close()
            self._centroid_cache.clear()
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()