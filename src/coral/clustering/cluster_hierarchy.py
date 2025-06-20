"""
ClusterHierarchy component for multi-level clustering structure management.

This module provides comprehensive hierarchical clustering management with:
- Multi-level cluster organization (TENSOR, BLOCK, LAYER, MODEL)
- Efficient tree-like hierarchy operations with O(log n) traversal
- Hierarchy construction, navigation, and modification
- Optimization and restructuring capabilities
- Thread-safe operations for concurrent access
- Serialization support for persistence
"""

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import re

import numpy as np
from scipy.spatial.distance import euclidean, cosine
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from .cluster_types import ClusterInfo, ClusterLevel, ClusteringStrategy, Centroid
from .cluster_config import HierarchyConfig
from ..core.weight_tensor import WeightTensor

logger = logging.getLogger(__name__)


@dataclass
class HierarchyNode:
    """Internal representation of a node in the cluster hierarchy."""
    
    cluster_info: ClusterInfo
    parent: Optional['HierarchyNode'] = None
    children: List['HierarchyNode'] = field(default_factory=list)
    depth: int = 0
    
    def add_child(self, child: 'HierarchyNode') -> None:
        """Add a child node and update relationships."""
        if child not in self.children:
            self.children.append(child)
            child.parent = self
            child._update_depth()
    
    def remove_child(self, child: 'HierarchyNode') -> bool:
        """Remove a child node and update relationships."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            child._update_depth()
            return True
        return False
    
    def _update_depth(self) -> None:
        """Update depth based on parent relationship."""
        if self.parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1
        
        # Recursively update children depths
        for child in self.children:
            child._update_depth()
    
    def get_siblings(self) -> List['HierarchyNode']:
        """Get all sibling nodes."""
        if self.parent is None:
            return []
        return [child for child in self.parent.children if child != self]
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """Check if this is a root node."""
        return self.parent is None
    
    def get_descendants(self) -> List['HierarchyNode']:
        """Get all descendant nodes recursively."""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants
    
    def get_ancestors(self) -> List['HierarchyNode']:
        """Get all ancestor nodes up to root."""
        ancestors = []
        current = self.parent
        while current is not None:
            ancestors.append(current)
            current = current.parent
        return ancestors


@dataclass
class HierarchyMetrics:
    """Metrics for evaluating hierarchy quality and structure."""
    
    total_clusters: int = 0
    depth: int = 0
    balance_score: float = 0.0  # How balanced the tree is [0, 1]
    utilization_score: float = 0.0  # How well clusters are utilized [0, 1]
    avg_branching_factor: float = 0.0  # Average number of children per node
    leaf_ratio: float = 0.0  # Ratio of leaf nodes to total nodes [0, 1]
    level_distribution: Dict[ClusterLevel, int] = field(default_factory=dict)
    connectivity_score: float = 0.0  # How well connected the hierarchy is [0, 1]
    redundancy_score: float = 0.0  # Amount of redundancy in the hierarchy [0, 1]
    
    def is_valid(self) -> bool:
        """Validate metric values are within expected ranges."""
        if self.total_clusters < 0 or self.depth < 0:
            return False
        if not (0.0 <= self.balance_score <= 1.0):
            return False
        if not (0.0 <= self.utilization_score <= 1.0):
            return False
        if not (0.0 <= self.leaf_ratio <= 1.0):
            return False
        if not (0.0 <= self.connectivity_score <= 1.0):
            return False
        if not (0.0 <= self.redundancy_score <= 1.0):
            return False
        if self.avg_branching_factor < 0:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_clusters": self.total_clusters,
            "depth": self.depth,
            "balance_score": self.balance_score,
            "utilization_score": self.utilization_score,
            "avg_branching_factor": self.avg_branching_factor,
            "leaf_ratio": self.leaf_ratio,
            "level_distribution": {level.value: count for level, count in self.level_distribution.items()},
            "connectivity_score": self.connectivity_score,
            "redundancy_score": self.redundancy_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HierarchyMetrics":
        """Create from dictionary."""
        level_distribution = {}
        for level_str, count in data.get("level_distribution", {}).items():
            level_distribution[ClusterLevel(level_str)] = count
        
        return cls(
            total_clusters=data.get("total_clusters", 0),
            depth=data.get("depth", 0),
            balance_score=data.get("balance_score", 0.0),
            utilization_score=data.get("utilization_score", 0.0),
            avg_branching_factor=data.get("avg_branching_factor", 0.0),
            leaf_ratio=data.get("leaf_ratio", 0.0),
            level_distribution=level_distribution,
            connectivity_score=data.get("connectivity_score", 0.0),
            redundancy_score=data.get("redundancy_score", 0.0),
        )
    
    @classmethod
    def from_hierarchy(cls, hierarchy: 'ClusterHierarchy') -> "HierarchyMetrics":
        """Compute metrics from a hierarchy instance."""
        if not hierarchy._nodes:
            return cls()
        
        # Basic counts
        total_clusters = len(hierarchy._nodes)
        depth = hierarchy._compute_max_depth()
        
        # Level distribution
        level_distribution = defaultdict(int)
        leaf_count = 0
        total_children = 0
        internal_nodes = 0
        
        for node in hierarchy._nodes.values():
            level_distribution[node.cluster_info.level] += 1
            if node.is_leaf():
                leaf_count += 1
            else:
                internal_nodes += 1
                total_children += len(node.children)
        
        # Calculate metrics
        leaf_ratio = leaf_count / total_clusters if total_clusters > 0 else 0.0
        avg_branching_factor = total_children / internal_nodes if internal_nodes > 0 else 0.0
        
        # Balance score (how evenly distributed nodes are across levels)
        balance_score = hierarchy._compute_balance_score()
        
        # Utilization score (based on cluster member counts)
        utilization_score = hierarchy._compute_utilization_score()
        
        # Connectivity score (how well connected the hierarchy is)
        connectivity_score = hierarchy._compute_connectivity_score()
        
        # Redundancy score (amount of overlap/redundancy)
        redundancy_score = hierarchy._compute_redundancy_score()
        
        return cls(
            total_clusters=total_clusters,
            depth=depth,
            balance_score=balance_score,
            utilization_score=utilization_score,
            avg_branching_factor=avg_branching_factor,
            leaf_ratio=leaf_ratio,
            level_distribution=dict(level_distribution),
            connectivity_score=connectivity_score,
            redundancy_score=redundancy_score,
        )


class ClusterHierarchy:
    """
    Multi-level clustering hierarchy management system.
    
    Provides comprehensive hierarchy operations including:
    - Construction and navigation
    - Multi-level cluster operations
    - Optimization and restructuring
    - Thread-safe concurrent access
    - Serialization and persistence
    """
    
    def __init__(self, config: Optional[HierarchyConfig] = None):
        """
        Initialize cluster hierarchy.
        
        Args:
            config: Hierarchy configuration, uses default if None
        """
        self.config = config or HierarchyConfig()
        
        # Core hierarchy storage
        self._nodes: Dict[str, HierarchyNode] = {}  # cluster_id -> node
        self._level_index: Dict[ClusterLevel, Set[str]] = defaultdict(set)  # level -> cluster_ids
        self._root_nodes: Set[str] = set()  # Root node cluster_ids
        
        # Performance optimization
        self._path_cache: Dict[str, List[HierarchyNode]] = {}  # Cache for path computations
        self._ancestor_cache: Dict[Tuple[str, str], Optional[str]] = {}  # Cache for common ancestors
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._operation_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"Initialized ClusterHierarchy with {len(self.config.levels)} levels")
    
    def build_hierarchy(self, clusters: List[ClusterInfo], config: HierarchyConfig) -> None:
        """
        Build multi-level cluster hierarchy from cluster list.
        
        Args:
            clusters: List of cluster information objects
            config: Hierarchy configuration
        """
        with self._lock:
            logger.info(f"Building hierarchy with {len(clusters)} clusters")
            start_time = time.time()
            
            # Clear existing hierarchy
            self._clear_hierarchy()
            
            # Update config
            self.config = config
            
            # Phase 1: Create nodes for all clusters
            self._create_nodes(clusters)
            
            # Phase 2: Establish parent-child relationships
            self._establish_relationships(clusters)
            
            # Phase 3: Determine hierarchy levels automatically if not specified
            self._auto_determine_levels()
            
            # Phase 4: Validate hierarchy consistency
            if self.config.enforce_consistency:
                validation_result = self.validate_consistency()
                if not validation_result["is_valid"]:
                    logger.warning(f"Hierarchy consistency issues: {validation_result['issues']}")
            
            # Clear caches after rebuild
            self._clear_caches()
            
            build_time = time.time() - start_time
            logger.info(f"Built hierarchy in {build_time:.3f}s with {len(self._nodes)} nodes")
    
    def add_level(self, level: ClusterLevel, clusters: List[ClusterInfo]) -> None:
        """
        Add clusters at a specific hierarchy level.
        
        Args:
            level: Target hierarchy level
            clusters: List of clusters to add
            
        Raises:
            ValueError: If cluster with same ID already exists
        """
        with self._lock:
            logger.debug(f"Adding {len(clusters)} clusters at level {level.value}")
            
            for cluster in clusters:
                if cluster.cluster_id in self._nodes:
                    raise ValueError(f"Cluster '{cluster.cluster_id}' already exists in hierarchy")
                
                # Create node
                cluster.level = level  # Ensure level is set correctly
                node = HierarchyNode(cluster)
                
                # Add to hierarchy
                self._nodes[cluster.cluster_id] = node
                self._level_index[level].add(cluster.cluster_id)
                
                # If no parent specified, treat as root
                if not cluster.parent_cluster_id:
                    self._root_nodes.add(cluster.cluster_id)
            
            # Clear relevant caches
            self._clear_caches()
            
            logger.debug(f"Added {len(clusters)} clusters to level {level.value}")
    
    def get_level_clusters(self, level: ClusterLevel) -> List[ClusterInfo]:
        """
        Get all clusters at a specific hierarchy level.
        
        Args:
            level: Target hierarchy level
            
        Returns:
            List of cluster info objects at the specified level
        """
        with self._lock:
            cluster_ids = self._level_index.get(level, set())
            return [self._nodes[cid].cluster_info for cid in cluster_ids if cid in self._nodes]
    
    def navigate_up(self, cluster_id: str) -> Optional[ClusterInfo]:
        """
        Navigate up the hierarchy to get parent cluster.
        
        Args:
            cluster_id: Child cluster identifier
            
        Returns:
            Parent cluster info if exists, None otherwise
        """
        with self._lock:
            self._operation_count += 1
            
            node = self._nodes.get(cluster_id)
            if not node or not node.parent:
                return None
            
            return node.parent.cluster_info
    
    def navigate_down(self, cluster_id: str) -> List[ClusterInfo]:
        """
        Navigate down the hierarchy to get child clusters.
        
        Args:
            cluster_id: Parent cluster identifier
            
        Returns:
            List of child cluster info objects
        """
        with self._lock:
            self._operation_count += 1
            
            node = self._nodes.get(cluster_id)
            if not node:
                return []
            
            return [child.cluster_info for child in node.children]
    
    def find_path_to_root(self, cluster_id: str) -> List[ClusterInfo]:
        """
        Find path from cluster to root of hierarchy.
        
        Args:
            cluster_id: Starting cluster identifier
            
        Returns:
            List of cluster info objects from cluster to root
        """
        with self._lock:
            self._operation_count += 1
            
            # Check cache first
            if cluster_id in self._path_cache:
                self._cache_hits += 1
                return [node.cluster_info for node in self._path_cache[cluster_id]]
            
            self._cache_misses += 1
            
            node = self._nodes.get(cluster_id)
            if not node:
                return []
            
            # Build path to root
            path = []
            current = node
            while current is not None:
                path.append(current)
                current = current.parent
            
            # Cache the path
            self._path_cache[cluster_id] = path.copy()
            
            return [node.cluster_info for node in path]
    
    def find_common_ancestor(self, cluster_id1: str, cluster_id2: str) -> Optional[ClusterInfo]:
        """
        Find common ancestor of two clusters.
        
        Args:
            cluster_id1: First cluster identifier
            cluster_id2: Second cluster identifier
            
        Returns:
            Common ancestor cluster info if found, None otherwise
        """
        with self._lock:
            self._operation_count += 1
            
            # Check cache first
            cache_key = tuple(sorted([cluster_id1, cluster_id2]))
            if cache_key in self._ancestor_cache:
                self._cache_hits += 1
                ancestor_id = self._ancestor_cache[cache_key]
                if ancestor_id:
                    return self._nodes[ancestor_id].cluster_info
                return None
            
            self._cache_misses += 1
            
            # Handle same cluster case
            if cluster_id1 == cluster_id2:
                node = self._nodes.get(cluster_id1)
                if node:
                    self._ancestor_cache[cache_key] = cluster_id1
                    return node.cluster_info
                return None
            
            # Get both nodes
            node1 = self._nodes.get(cluster_id1)
            node2 = self._nodes.get(cluster_id2)
            
            if not node1 or not node2:
                self._ancestor_cache[cache_key] = None
                return None
            
            # Get ancestors for both nodes
            ancestors1 = set(ancestor.cluster_info.cluster_id for ancestor in node1.get_ancestors())
            ancestors1.add(cluster_id1)  # Include the node itself
            
            # Find first common ancestor
            current = node2
            while current is not None:
                if current.cluster_info.cluster_id in ancestors1:
                    self._ancestor_cache[cache_key] = current.cluster_info.cluster_id
                    return current.cluster_info
                current = current.parent
            
            self._ancestor_cache[cache_key] = None
            return None
    
    def get_all_descendants(self, cluster_id: str) -> List[ClusterInfo]:
        """
        Get all descendant clusters recursively.
        
        Args:
            cluster_id: Parent cluster identifier
            
        Returns:
            List of all descendant cluster info objects
        """
        with self._lock:
            node = self._nodes.get(cluster_id)
            if not node:
                return []
            
            descendants = node.get_descendants()
            return [desc.cluster_info for desc in descendants]
    
    def search_by_criteria(self, criteria: Dict[str, Any]) -> List[ClusterInfo]:
        """
        Search clusters by various criteria.
        
        Args:
            criteria: Dictionary of search criteria
                - level: ClusterLevel
                - strategy: ClusteringStrategy
                - min_member_count: int
                - max_member_count: int
                - name_pattern: regex pattern string
                - custom_filter: callable(ClusterInfo) -> bool
                - has_parent: bool
                - has_children: bool
                - depth_range: (min_depth, max_depth)
                
        Returns:
            List of matching cluster info objects
        """
        with self._lock:
            results = []
            
            for node in self._nodes.values():
                cluster = node.cluster_info
                
                # Check level
                if "level" in criteria and cluster.level != criteria["level"]:
                    continue
                
                # Check strategy
                if "strategy" in criteria and cluster.strategy != criteria["strategy"]:
                    continue
                
                # Check member count range
                if "min_member_count" in criteria and cluster.member_count < criteria["min_member_count"]:
                    continue
                if "max_member_count" in criteria and cluster.member_count > criteria["max_member_count"]:
                    continue
                
                # Check name pattern
                if "name_pattern" in criteria:
                    pattern = criteria["name_pattern"]
                    if not re.search(pattern, cluster.cluster_id):
                        continue
                
                # Check parent/children requirements
                if "has_parent" in criteria:
                    has_parent = node.parent is not None
                    if has_parent != criteria["has_parent"]:
                        continue
                
                if "has_children" in criteria:
                    has_children = len(node.children) > 0
                    if has_children != criteria["has_children"]:
                        continue
                
                # Check depth range
                if "depth_range" in criteria:
                    min_depth, max_depth = criteria["depth_range"]
                    if not (min_depth <= node.depth <= max_depth):
                        continue
                
                # Apply custom filter
                if "custom_filter" in criteria:
                    custom_filter = criteria["custom_filter"]
                    if not custom_filter(cluster):
                        continue
                
                results.append(cluster)
            
            return results
    
    def merge_clusters(self, cluster_ids: List[str], target_level: ClusterLevel, 
                      strategy: str = "centroid") -> Optional[ClusterInfo]:
        """
        Merge clusters into a higher hierarchy level.
        
        Args:
            cluster_ids: List of cluster IDs to merge
            target_level: Target hierarchy level for merged cluster
            strategy: Merge strategy ("centroid", "largest", "weighted_average")
            
        Returns:
            New merged cluster info if successful, None otherwise
            
        Raises:
            ValueError: If clusters cannot be merged or invalid parameters
        """
        with self._lock:
            logger.debug(f"Merging {len(cluster_ids)} clusters to level {target_level.value}")
            
            if len(cluster_ids) < 2:
                raise ValueError("Need at least 2 clusters to merge")
            
            # Validate all clusters exist
            source_nodes = []
            for cluster_id in cluster_ids:
                node = self._nodes.get(cluster_id)
                if not node:
                    raise ValueError(f"Cluster '{cluster_id}' not found")
                source_nodes.append(node)
            
            # Determine merge strategy
            if strategy == "largest":
                # Use the largest cluster as base
                base_node = max(source_nodes, key=lambda n: n.cluster_info.member_count)
                merged_cluster = base_node.cluster_info
            elif strategy == "centroid":
                # Create new cluster with merged properties
                total_members = sum(node.cluster_info.member_count for node in source_nodes)
                merged_cluster = ClusterInfo(
                    cluster_id=f"merged_{uuid.uuid4().hex[:8]}",
                    strategy=ClusteringStrategy.HIERARCHICAL,
                    level=target_level,
                    member_count=total_members,
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ")
                )
            elif strategy == "weighted_average":
                # Weighted average based on member counts
                total_members = sum(node.cluster_info.member_count for node in source_nodes)
                merged_cluster = ClusterInfo(
                    cluster_id=f"merged_weighted_{uuid.uuid4().hex[:8]}",
                    strategy=ClusteringStrategy.HIERARCHICAL,
                    level=target_level,
                    member_count=total_members,
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ")
                )
            else:
                raise ValueError(f"Unknown merge strategy: {strategy}")
            
            # Create new node for merged cluster
            merged_node = HierarchyNode(merged_cluster)
            
            # Add merged cluster to hierarchy
            self._nodes[merged_cluster.cluster_id] = merged_node
            self._level_index[target_level].add(merged_cluster.cluster_id)
            
            # Set up parent-child relationships
            for source_node in source_nodes:
                # Remove from original level and make child of merged cluster
                old_level = source_node.cluster_info.level
                self._level_index[old_level].discard(source_node.cluster_info.cluster_id)
                self._root_nodes.discard(source_node.cluster_info.cluster_id)
                
                # Add as child of merged cluster
                merged_node.add_child(source_node)
            
            # If no existing parent, make it root
            if not merged_cluster.parent_cluster_id:
                self._root_nodes.add(merged_cluster.cluster_id)
            
            # Clear caches
            self._clear_caches()
            
            logger.debug(f"Merged clusters into '{merged_cluster.cluster_id}' at level {target_level.value}")
            return merged_cluster
    
    def split_cluster(self, cluster_id: str, strategy: str = "kmeans", 
                     target_level: Optional[ClusterLevel] = None, n_splits: int = 2) -> List[ClusterInfo]:
        """
        Split cluster into sub-clusters at lower hierarchy level.
        
        Args:
            cluster_id: Cluster to split
            strategy: Split strategy ("kmeans", "hierarchical", "size_based")
            target_level: Target level for split clusters (auto-determined if None)
            n_splits: Number of clusters to split into
            
        Returns:
            List of new split cluster info objects
            
        Raises:
            ValueError: If cluster cannot be split or invalid parameters
        """
        with self._lock:
            logger.debug(f"Splitting cluster '{cluster_id}' into {n_splits} parts")
            
            if n_splits < 2:
                raise ValueError("Need at least 2 splits")
            
            # Get source cluster
            source_node = self._nodes.get(cluster_id)
            if not source_node:
                raise ValueError(f"Cluster '{cluster_id}' not found")
            
            source_cluster = source_node.cluster_info
            
            # Determine target level
            if target_level is None:
                # Auto-determine lower level
                current_level_idx = self.config.levels.index(source_cluster.level)
                if current_level_idx == 0:
                    raise ValueError("Cannot split cluster at lowest level")
                target_level = self.config.levels[current_level_idx - 1]
            
            if target_level == source_cluster.level:
                raise ValueError("Cannot split cluster to same level")
            
            # Create split clusters based on strategy
            split_clusters = []
            
            if strategy == "size_based":
                # Split based on member count
                members_per_split = max(1, source_cluster.member_count // n_splits)
                
                for i in range(n_splits):
                    remaining_members = source_cluster.member_count - i * members_per_split
                    if i == n_splits - 1:
                        # Last split gets remaining members
                        split_size = remaining_members
                    else:
                        split_size = min(members_per_split, remaining_members)
                    
                    split_cluster = ClusterInfo(
                        cluster_id=f"split_{cluster_id}_{i}",
                        strategy=ClusteringStrategy.HIERARCHICAL,
                        level=target_level,
                        member_count=split_size,
                        parent_cluster_id=cluster_id,
                        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ")
                    )
                    split_clusters.append(split_cluster)
            
            elif strategy in ["kmeans", "hierarchical"]:
                # Create equal-sized splits (simplified approach)
                avg_size = source_cluster.member_count // n_splits
                
                for i in range(n_splits):
                    if i == n_splits - 1:
                        # Last cluster gets remainder
                        split_size = source_cluster.member_count - (avg_size * (n_splits - 1))
                    else:
                        split_size = avg_size
                    
                    split_cluster = ClusterInfo(
                        cluster_id=f"split_{strategy}_{cluster_id}_{i}",
                        strategy=ClusteringStrategy.KMEANS if strategy == "kmeans" else ClusteringStrategy.HIERARCHICAL,
                        level=target_level,
                        member_count=split_size,
                        parent_cluster_id=cluster_id,
                        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ")
                    )
                    split_clusters.append(split_cluster)
            
            else:
                raise ValueError(f"Unknown split strategy: {strategy}")
            
            # Add split clusters to hierarchy
            for split_cluster in split_clusters:
                split_node = HierarchyNode(split_cluster)
                self._nodes[split_cluster.cluster_id] = split_node
                self._level_index[target_level].add(split_cluster.cluster_id)
                
                # Set parent relationship
                source_node.add_child(split_node)
            
            # Update source cluster's child list
            child_ids = [cluster.cluster_id for cluster in split_clusters]
            source_cluster.child_cluster_ids = child_ids
            
            # Clear caches
            self._clear_caches()
            
            logger.debug(f"Split cluster '{cluster_id}' into {len(split_clusters)} parts")
            return split_clusters
    
    def promote_cluster(self, cluster_id: str, target_level: ClusterLevel) -> Optional[ClusterInfo]:
        """
        Promote cluster to higher hierarchy level.
        
        Args:
            cluster_id: Cluster to promote
            target_level: Target hierarchy level
            
        Returns:
            Promoted cluster info if successful, None otherwise
            
        Raises:
            ValueError: If cluster cannot be promoted or invalid level
        """
        with self._lock:
            logger.debug(f"Promoting cluster '{cluster_id}' to level {target_level.value}")
            
            # Get source cluster
            source_node = self._nodes.get(cluster_id)
            if not source_node:
                raise ValueError(f"Cluster '{cluster_id}' not found")
            
            source_cluster = source_node.cluster_info
            
            if source_cluster.level == target_level:
                raise ValueError("Cannot promote cluster to same level")
            
            # Remove from old level
            old_level = source_cluster.level
            self._level_index[old_level].discard(cluster_id)
            
            # Update cluster level
            source_cluster.level = target_level
            
            # Add to new level
            self._level_index[target_level].add(cluster_id)
            
            # Clear caches
            self._clear_caches()
            
            logger.debug(f"Promoted cluster '{cluster_id}' to level {target_level.value}")
            return source_cluster
    
    def rebalance_level(self, level: ClusterLevel, strategy: str = "auto",
                       **kwargs) -> Dict[str, Any]:
        """
        Rebalance clusters at a specific hierarchy level.
        
        Args:
            level: Hierarchy level to rebalance
            strategy: Rebalancing strategy ("auto", "size", "quality", "merge_small")
            **kwargs: Strategy-specific parameters
            
        Returns:
            Dictionary with rebalancing results and metrics
        """
        with self._lock:
            logger.debug(f"Rebalancing level {level.value} with strategy '{strategy}'")
            
            level_clusters = self.get_level_clusters(level)
            if len(level_clusters) < 2:
                return {"rebalanced_clusters": level_clusters, "metrics": {}, "changes": 0}
            
            original_count = len(level_clusters)
            rebalanced_clusters = level_clusters.copy()
            changes = 0
            
            if strategy == "size":
                # Rebalance based on cluster sizes
                target_size = kwargs.get("target_size", 10)
                
                # Merge very small clusters
                small_clusters = [c for c in level_clusters if c.member_count < target_size // 2]
                if len(small_clusters) >= 2:
                    cluster_ids = [c.cluster_id for c in small_clusters[:2]]
                    merged = self.merge_clusters(cluster_ids, level)
                    if merged:
                        changes += 1
                
                # Split very large clusters (only if not at lowest level)
                current_level_idx = self.config.levels.index(level)
                if current_level_idx > 0:  # Not at lowest level
                    large_clusters = [c for c in level_clusters if c.member_count > target_size * 2]
                    for large_cluster in large_clusters[:1]:  # Limit to avoid too many changes
                        n_splits = min(4, large_cluster.member_count // target_size)
                        if n_splits >= 2:
                            try:
                                splits = self.split_cluster(large_cluster.cluster_id, n_splits=n_splits)
                                if splits:
                                    changes += len(splits)
                            except ValueError:
                                # Can't split, continue
                                continue
            
            elif strategy == "quality":
                # Rebalance based on quality metrics
                quality_threshold = kwargs.get("quality_threshold", 0.8)
                
                # This would require actual cluster quality computation
                # For now, implement a simple heuristic
                avg_size = sum(c.member_count for c in level_clusters) // len(level_clusters)
                
                # Merge clusters that are too small
                small_clusters = [c for c in level_clusters if c.member_count < avg_size * 0.3]
                if len(small_clusters) >= 2:
                    cluster_ids = [c.cluster_id for c in small_clusters[:2]]
                    merged = self.merge_clusters(cluster_ids, level)
                    if merged:
                        changes += 1
            
            elif strategy == "merge_small":
                # Merge clusters below minimum size
                min_size = kwargs.get("min_size", 5)
                small_clusters = [c for c in level_clusters if c.member_count < min_size]
                
                # Merge pairs of small clusters
                for i in range(0, len(small_clusters) - 1, 2):
                    cluster_ids = [small_clusters[i].cluster_id, small_clusters[i + 1].cluster_id]
                    merged = self.merge_clusters(cluster_ids, level)
                    if merged:
                        changes += 1
            
            # Get updated clusters after rebalancing
            final_clusters = self.get_level_clusters(level)
            
            # Compute metrics
            metrics = {
                "original_count": original_count,
                "final_count": len(final_clusters),
                "changes_made": changes,
                "avg_cluster_size": sum(c.member_count for c in final_clusters) / len(final_clusters) if final_clusters else 0,
                "size_std": np.std([c.member_count for c in final_clusters]) if final_clusters else 0,
            }
            
            logger.debug(f"Rebalanced level {level.value}: {changes} changes, {original_count} -> {len(final_clusters)} clusters")
            
            return {
                "rebalanced_clusters": final_clusters,
                "metrics": metrics,
                "changes": changes
            }
    
    def optimize_hierarchy(self, objectives: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Optimize hierarchy structure for better performance and organization.
        
        Args:
            objectives: List of optimization objectives
                - "balance": Balance tree structure
                - "minimize_depth": Reduce hierarchy depth
                - "maximize_utilization": Improve cluster utilization
                - "reduce_redundancy": Minimize redundant clusters
                
        Returns:
            Dictionary with optimization results
        """
        with self._lock:
            logger.info("Optimizing hierarchy structure")
            start_time = time.time()
            
            if objectives is None:
                objectives = ["balance", "minimize_depth", "maximize_utilization"]
            
            # Get initial metrics
            initial_metrics = self.compute_hierarchy_metrics()
            optimizations_made = []
            
            # Balance optimization
            if "balance" in objectives:
                balance_changes = self._optimize_balance()
                optimizations_made.extend(balance_changes)
            
            # Depth optimization
            if "minimize_depth" in objectives:
                depth_changes = self._optimize_depth()
                optimizations_made.extend(depth_changes)
            
            # Utilization optimization
            if "maximize_utilization" in objectives:
                utilization_changes = self._optimize_utilization()
                optimizations_made.extend(utilization_changes)
            
            # Redundancy reduction
            if "reduce_redundancy" in objectives:
                redundancy_changes = self._reduce_redundancy()
                optimizations_made.extend(redundancy_changes)
            
            # Get final metrics
            final_metrics = self.compute_hierarchy_metrics()
            
            # Clear caches after optimization
            self._clear_caches()
            
            optimization_time = time.time() - start_time
            
            result = {
                "optimized": True,
                "objectives": objectives,
                "optimizations_made": optimizations_made,
                "initial_metrics": initial_metrics.to_dict(),
                "final_metrics": final_metrics.to_dict(),
                "optimization_time": optimization_time,
            }
            
            logger.info(f"Hierarchy optimization completed in {optimization_time:.3f}s with {len(optimizations_made)} changes")
            return result
    
    def validate_consistency(self) -> Dict[str, Any]:
        """
        Validate hierarchy consistency and return validation results.
        
        Returns:
            Dictionary containing validation results and any issues found
        """
        with self._lock:
            issues = []
            
            # Check parent-child relationship consistency
            for cluster_id, node in self._nodes.items():
                cluster = node.cluster_info
                
                # Check parent relationship
                if cluster.parent_cluster_id:
                    parent_node = self._nodes.get(cluster.parent_cluster_id)
                    if not parent_node:
                        issues.append(f"Cluster '{cluster_id}' has orphaned parent '{cluster.parent_cluster_id}'")
                    elif node not in parent_node.children:
                        issues.append(f"Parent-child relationship inconsistent for '{cluster_id}'")
                
                # Check child relationships
                if cluster.child_cluster_ids:
                    for child_id in cluster.child_cluster_ids:
                        child_node = self._nodes.get(child_id)
                        if not child_node:
                            issues.append(f"Cluster '{cluster_id}' has orphaned child '{child_id}'")
                        elif child_node.parent != node:
                            issues.append(f"Child-parent relationship inconsistent for '{child_id}'")
            
            # Check level consistency
            for level, cluster_ids in self._level_index.items():
                for cluster_id in cluster_ids:
                    node = self._nodes.get(cluster_id)
                    if not node:
                        issues.append(f"Level index references non-existent cluster '{cluster_id}'")
                    elif node.cluster_info.level != level:
                        issues.append(f"Cluster '{cluster_id}' level mismatch in index")
            
            # Check root nodes consistency
            for root_id in self._root_nodes:
                node = self._nodes.get(root_id)
                if not node:
                    issues.append(f"Root index references non-existent cluster '{root_id}'")
                elif node.parent is not None:
                    issues.append(f"Root cluster '{root_id}' has parent")
            
            # Check for cycles
            cycle_issues = self._detect_cycles()
            issues.extend(cycle_issues)
            
            # Check hierarchy depth limits
            max_depth = self._compute_max_depth()
            if max_depth > self.config.max_hierarchy_depth:
                issues.append(f"Hierarchy depth {max_depth} exceeds limit {self.config.max_hierarchy_depth}")
            
            return {
                "is_valid": len(issues) == 0,
                "issues": issues,
                "total_clusters": len(self._nodes),
                "total_levels": len(self._level_index),
                "root_count": len(self._root_nodes),
                "max_depth": max_depth,
            }
    
    def compute_hierarchy_metrics(self, include_level_stats: bool = False,
                                include_connectivity: bool = False,
                                include_quality_metrics: bool = False) -> HierarchyMetrics:
        """
        Compute comprehensive hierarchy metrics.
        
        Args:
            include_level_stats: Include detailed level statistics
            include_connectivity: Include connectivity analysis
            include_quality_metrics: Include quality metrics
            
        Returns:
            HierarchyMetrics object with computed metrics
        """
        with self._lock:
            metrics = HierarchyMetrics.from_hierarchy(self)
            
            # If additional metrics requested, store them as a special attribute
            if include_level_stats or include_connectivity or include_quality_metrics:
                metrics_dict = metrics.to_dict()
                
                if include_level_stats:
                    level_stats = {}
                    for level in self.config.levels:
                        clusters = self.get_level_clusters(level)
                        if clusters:
                            level_stats[level.value] = {
                                "count": len(clusters),
                                "avg_size": sum(c.member_count for c in clusters) / len(clusters),
                                "min_size": min(c.member_count for c in clusters),
                                "max_size": max(c.member_count for c in clusters),
                                "total_members": sum(c.member_count for c in clusters),
                            }
                        else:
                            level_stats[level.value] = {
                                "count": 0, "avg_size": 0, "min_size": 0, "max_size": 0, "total_members": 0
                            }
                    # Store additional metrics as attributes
                    setattr(metrics, '_level_stats', level_stats)
                
                if include_connectivity:
                    connectivity_metrics = self._compute_connectivity_metrics()
                    setattr(metrics, '_connectivity_metrics', connectivity_metrics)
                
                if include_quality_metrics:
                    quality_metrics = self._compute_quality_metrics()
                    setattr(metrics, '_quality_metrics', quality_metrics)
                
                # Override to_dict to include additional metrics
                original_to_dict = metrics.to_dict
                def enhanced_to_dict():
                    result = original_to_dict()
                    if hasattr(metrics, '_level_stats'):
                        result["level_stats"] = metrics._level_stats
                    if hasattr(metrics, '_connectivity_metrics'):
                        result["connectivity_metrics"] = metrics._connectivity_metrics
                    if hasattr(metrics, '_quality_metrics'):
                        result["quality_metrics"] = metrics._quality_metrics
                    return result
                metrics.to_dict = enhanced_to_dict
            
            return metrics
    
    def suggest_restructuring(self) -> Dict[str, Any]:
        """
        Suggest hierarchy restructuring improvements.
        
        Returns:
            Dictionary with restructuring suggestions and confidence scores
        """
        with self._lock:
            suggestions = []
            
            # Analyze current metrics
            metrics = self.compute_hierarchy_metrics()
            
            # Suggest based on balance score
            if metrics.balance_score < 0.6:
                suggestions.append({
                    "type": "rebalance_tree",
                    "description": "Tree is unbalanced, consider rebalancing levels",
                    "priority": "high",
                    "confidence": 0.8,
                    "action": "Call rebalance_level() for unbalanced levels"
                })
            
            # Suggest based on depth
            if metrics.depth > self.config.max_hierarchy_depth * 0.8:
                suggestions.append({
                    "type": "reduce_depth",
                    "description": "Hierarchy is too deep, consider merging levels",
                    "priority": "medium",
                    "confidence": 0.7,
                    "action": "Merge clusters at intermediate levels"
                })
            
            # Suggest based on utilization
            if metrics.utilization_score < 0.5:
                suggestions.append({
                    "type": "improve_utilization",
                    "description": "Poor cluster utilization, consider restructuring",
                    "priority": "medium",
                    "confidence": 0.6,
                    "action": "Merge small clusters or split large ones"
                })
            
            # Suggest based on leaf ratio
            if metrics.leaf_ratio > 0.8:
                suggestions.append({
                    "type": "add_intermediate_levels",
                    "description": "Too many leaf nodes, consider adding intermediate levels",
                    "priority": "low",
                    "confidence": 0.5,
                    "action": "Group related leaf clusters into intermediate nodes"
                })
            
            # Check for specific issues in level distribution
            total_clusters = sum(metrics.level_distribution.values())
            for level, count in metrics.level_distribution.items():
                ratio = count / total_clusters if total_clusters > 0 else 0
                
                if ratio < 0.1 and count > 1:  # Very few clusters at this level
                    suggestions.append({
                        "type": "consolidate_level",
                        "description": f"Very few clusters at {level.value} level, consider consolidation",
                        "priority": "low",
                        "confidence": 0.4,
                        "action": f"Merge or redistribute clusters at {level.value} level"
                    })
                elif ratio > 0.7:  # Too many clusters at this level
                    suggestions.append({
                        "type": "split_level",
                        "description": f"Too many clusters at {level.value} level, consider splitting",
                        "priority": "medium",
                        "confidence": 0.6,
                        "action": f"Split large clusters at {level.value} level or add sub-levels"
                    })
            
            # Estimate overall improvement potential
            confidence_scores = [s["confidence"] for s in suggestions]
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            estimated_improvement = min(0.3, overall_confidence * 0.5)  # Conservative estimate
            
            return {
                "suggestions": suggestions,
                "confidence": overall_confidence,
                "estimated_improvement": estimated_improvement,
                "current_metrics": metrics.to_dict(),
            }
    
    def get_all_clusters(self) -> List[ClusterInfo]:
        """Get all clusters in the hierarchy."""
        with self._lock:
            return [node.cluster_info for node in self._nodes.values()]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize hierarchy to dictionary.
        
        Returns:
            Dictionary representation of the hierarchy
        """
        with self._lock:
            # Serialize all clusters
            clusters_data = []
            for node in self._nodes.values():
                cluster_data = node.cluster_info.to_dict()
                cluster_data["depth"] = node.depth
                clusters_data.append(cluster_data)
            
            # Serialize hierarchy structure
            hierarchy_map = {}
            for cluster_id, node in self._nodes.items():
                hierarchy_map[cluster_id] = {
                    "parent": node.parent.cluster_info.cluster_id if node.parent else None,
                    "children": [child.cluster_info.cluster_id for child in node.children],
                    "depth": node.depth,
                }
            
            return {
                "config": self.config.to_dict(),
                "clusters": clusters_data,
                "hierarchy_map": hierarchy_map,
                "level_index": {level.value: list(cluster_ids) for level, cluster_ids in self._level_index.items()},
                "root_nodes": list(self._root_nodes),
                "metadata": {
                    "operation_count": self._operation_count,
                    "cache_hits": self._cache_hits,
                    "cache_misses": self._cache_misses,
                    "serialization_time": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterHierarchy":
        """
        Deserialize hierarchy from dictionary.
        
        Args:
            data: Dictionary representation of hierarchy
            
        Returns:
            ClusterHierarchy instance
        """
        # Create hierarchy with config
        config = HierarchyConfig.from_dict(data["config"])
        hierarchy = cls(config)
        
        # Restore clusters
        clusters = []
        for cluster_data in data["clusters"]:
            cluster = ClusterInfo.from_dict(cluster_data)
            clusters.append(cluster)
        
        # Build hierarchy structure
        hierarchy.build_hierarchy(clusters, config)
        
        # Restore metadata
        metadata = data.get("metadata", {})
        hierarchy._operation_count = metadata.get("operation_count", 0)
        hierarchy._cache_hits = metadata.get("cache_hits", 0)
        hierarchy._cache_misses = metadata.get("cache_misses", 0)
        
        return hierarchy
    
    # Private helper methods
    
    def _clear_hierarchy(self) -> None:
        """Clear all hierarchy data."""
        self._nodes.clear()
        self._level_index.clear()
        self._root_nodes.clear()
        self._clear_caches()
    
    def _clear_caches(self) -> None:
        """Clear all caches."""
        self._path_cache.clear()
        self._ancestor_cache.clear()
    
    def _create_nodes(self, clusters: List[ClusterInfo]) -> None:
        """Create hierarchy nodes for all clusters."""
        for cluster in clusters:
            node = HierarchyNode(cluster)
            self._nodes[cluster.cluster_id] = node
            # Don't add to level index yet - will be done in _auto_determine_levels
            
            # Track root nodes (no parent specified)
            if not cluster.parent_cluster_id:
                self._root_nodes.add(cluster.cluster_id)
    
    def _establish_relationships(self, clusters: List[ClusterInfo]) -> None:
        """Establish parent-child relationships from cluster data."""
        for cluster in clusters:
            node = self._nodes[cluster.cluster_id]
            
            # Set up parent relationship
            if cluster.parent_cluster_id and cluster.parent_cluster_id in self._nodes:
                parent_node = self._nodes[cluster.parent_cluster_id]
                parent_node.add_child(node)
                self._root_nodes.discard(cluster.cluster_id)  # Not a root if has parent
            
            # Set up child relationships
            if cluster.child_cluster_ids:
                for child_id in cluster.child_cluster_ids:
                    if child_id in self._nodes:
                        child_node = self._nodes[child_id]
                        node.add_child(child_node)
                        self._root_nodes.discard(child_id)  # Not a root if has parent
    
    def _auto_determine_levels(self) -> None:
        """Automatically determine hierarchy levels based on structure."""
        # Clear and rebuild level index to ensure consistency
        self._level_index.clear()
        
        for node in self._nodes.values():
            cluster = node.cluster_info
            
            # Preserve existing level if it's valid, otherwise auto-assign
            if cluster.level in self.config.levels:
                # Keep existing level
                pass
            else:
                # Auto-assign level based on characteristics
                if node.is_leaf() and cluster.member_count < 10:
                    cluster.level = ClusterLevel.TENSOR
                elif node.depth == 0:  # Root nodes
                    cluster.level = ClusterLevel.MODEL
                elif node.depth == 1:
                    cluster.level = ClusterLevel.LAYER
                else:
                    cluster.level = ClusterLevel.BLOCK
                    
                # Ensure the assigned level is in configured levels
                if cluster.level not in self.config.levels:
                    cluster.level = self.config.levels[0]  # Default to first level
            
            # Add to level index
            self._level_index[cluster.level].add(cluster.cluster_id)
    
    def _compute_max_depth(self) -> int:
        """Compute maximum depth of hierarchy."""
        if not self._nodes:
            return 0
        return max(node.depth for node in self._nodes.values())
    
    def _compute_balance_score(self) -> float:
        """Compute hierarchy balance score."""
        if not self._nodes:
            return 1.0
        
        # Simple balance metric based on depth variance
        depths = [node.depth for node in self._nodes.values()]
        if len(set(depths)) == 1:  # All same depth
            return 1.0
        
        depth_variance = np.var(depths)
        max_possible_variance = (max(depths) - min(depths)) ** 2 / 4
        
        if max_possible_variance == 0:
            return 1.0
        
        balance_score = 1.0 - (depth_variance / max_possible_variance)
        return max(0.0, min(1.0, balance_score))
    
    def _compute_utilization_score(self) -> float:
        """Compute cluster utilization score."""
        if not self._nodes:
            return 0.0
        
        member_counts = [node.cluster_info.member_count for node in self._nodes.values()]
        if not member_counts:
            return 0.0
        
        # Score based on how evenly distributed cluster sizes are
        mean_size = np.mean(member_counts)
        if mean_size == 0:
            return 0.0
        
        coefficient_of_variation = np.std(member_counts) / mean_size
        utilization_score = 1.0 / (1.0 + coefficient_of_variation)
        
        return max(0.0, min(1.0, utilization_score))
    
    def _compute_connectivity_score(self) -> float:
        """Compute hierarchy connectivity score."""
        if not self._nodes:
            return 0.0
        
        total_nodes = len(self._nodes)
        connected_nodes = sum(1 for node in self._nodes.values() if node.parent or node.children)
        
        connectivity_score = connected_nodes / total_nodes if total_nodes > 0 else 0.0
        return max(0.0, min(1.0, connectivity_score))
    
    def _compute_redundancy_score(self) -> float:
        """Compute hierarchy redundancy score."""
        if not self._nodes:
            return 0.0
        
        # Simple heuristic: clusters with very similar member counts might be redundant
        member_counts = [node.cluster_info.member_count for node in self._nodes.values()]
        unique_counts = len(set(member_counts))
        total_counts = len(member_counts)
        
        redundancy_score = 1.0 - (unique_counts / total_counts) if total_counts > 0 else 0.0
        return max(0.0, min(1.0, redundancy_score))
    
    def _detect_cycles(self) -> List[str]:
        """Detect cycles in hierarchy."""
        issues = []
        visited = set()
        rec_stack = set()
        
        def dfs(node_id: str, path: List[str]) -> bool:
            if node_id in rec_stack:
                cycle_path = path[path.index(node_id):] + [node_id]
                issues.append(f"Cycle detected: {' -> '.join(cycle_path)}")
                return True
            
            if node_id in visited:
                return False
            
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)
            
            node = self._nodes.get(node_id)
            if node:
                for child in node.children:
                    if dfs(child.cluster_info.cluster_id, path):
                        return True
            
            rec_stack.remove(node_id)
            path.pop()
            return False
        
        # Check all nodes
        for node_id in self._nodes:
            if node_id not in visited:
                dfs(node_id, [])
        
        return issues
    
    def _optimize_balance(self) -> List[str]:
        """Optimize hierarchy balance."""
        changes = []
        
        # Find levels with very uneven distribution
        for level in self.config.levels:
            clusters = self.get_level_clusters(level)
            if len(clusters) > 3:
                # Check if some clusters are much larger than others
                sizes = [c.member_count for c in clusters]
                mean_size = np.mean(sizes)
                large_clusters = [c for c in clusters if c.member_count > mean_size * 2]
                
                # Split large clusters
                for large_cluster in large_clusters[:2]:  # Limit changes
                    try:
                        splits = self.split_cluster(large_cluster.cluster_id, n_splits=2)
                        if splits:
                            changes.append(f"Split large cluster {large_cluster.cluster_id}")
                    except ValueError:
                        continue
        
        return changes
    
    def _optimize_depth(self) -> List[str]:
        """Optimize hierarchy depth."""
        changes = []
        
        # Find very deep paths and try to reduce them
        max_depth = self._compute_max_depth()
        if max_depth > self.config.max_hierarchy_depth:
            # Find deepest nodes
            deep_nodes = [node for node in self._nodes.values() if node.depth >= max_depth - 1]
            
            for node in deep_nodes[:3]:  # Limit changes
                if node.parent and len(node.parent.children) == 1:
                    # Try to merge with parent if it's the only child
                    try:
                        parent_id = node.parent.cluster_info.cluster_id
                        merged = self.merge_clusters([node.cluster_info.cluster_id, parent_id], node.cluster_info.level)
                        if merged:
                            changes.append(f"Merged single-child node {node.cluster_info.cluster_id} with parent")
                    except ValueError:
                        continue
        
        return changes
    
    def _optimize_utilization(self) -> List[str]:
        """Optimize cluster utilization."""
        changes = []
        
        # Find very small clusters and merge them
        for level in self.config.levels:
            clusters = self.get_level_clusters(level)
            if len(clusters) > 1:
                sizes = [c.member_count for c in clusters]
                mean_size = np.mean(sizes)
                small_clusters = [c for c in clusters if c.member_count < mean_size * 0.3]
                
                # Merge pairs of small clusters
                for i in range(0, len(small_clusters) - 1, 2):
                    try:
                        cluster_ids = [small_clusters[i].cluster_id, small_clusters[i + 1].cluster_id]
                        merged = self.merge_clusters(cluster_ids, level)
                        if merged:
                            changes.append(f"Merged small clusters at level {level.value}")
                    except ValueError:
                        continue
        
        return changes
    
    def _reduce_redundancy(self) -> List[str]:
        """Reduce hierarchy redundancy."""
        changes = []
        
        # Find clusters with very similar member counts that might be redundant
        for level in self.config.levels:
            clusters = self.get_level_clusters(level)
            if len(clusters) > 2:
                # Group by similar size
                size_groups = defaultdict(list)
                for cluster in clusters:
                    size_key = cluster.member_count // 5 * 5  # Group by 5s
                    size_groups[size_key].append(cluster)
                
                # Merge groups with multiple similar-sized clusters
                for size_key, group in size_groups.items():
                    if len(group) >= 3:  # 3 or more similar clusters
                        try:
                            cluster_ids = [c.cluster_id for c in group[:2]]  # Merge first two
                            merged = self.merge_clusters(cluster_ids, level)
                            if merged:
                                changes.append(f"Merged redundant clusters at level {level.value}")
                        except ValueError:
                            continue
        
        return changes
    
    def _compute_connectivity_metrics(self) -> Dict[str, Any]:
        """Compute detailed connectivity metrics."""
        if not self._nodes:
            return {}
        
        total_nodes = len(self._nodes)
        root_count = len(self._root_nodes)
        leaf_count = sum(1 for node in self._nodes.values() if node.is_leaf())
        internal_count = total_nodes - leaf_count
        
        # Average connectivity
        total_connections = sum(len(node.children) for node in self._nodes.values())
        avg_connectivity = total_connections / total_nodes if total_nodes > 0 else 0.0
        
        return {
            "total_nodes": total_nodes,
            "root_count": root_count,
            "leaf_count": leaf_count,
            "internal_count": internal_count,
            "total_connections": total_connections,
            "avg_connectivity": avg_connectivity,
            "connectivity_ratio": total_connections / (total_nodes - 1) if total_nodes > 1 else 0.0,
        }
    
    def _compute_quality_metrics(self) -> Dict[str, Any]:
        """Compute hierarchy quality metrics."""
        if not self._nodes:
            return {}
        
        # Structural quality
        balance_score = self._compute_balance_score()
        utilization_score = self._compute_utilization_score()
        
        # Efficiency metrics
        avg_depth = np.mean([node.depth for node in self._nodes.values()])
        depth_variance = np.var([node.depth for node in self._nodes.values()])
        
        # Size distribution quality
        sizes = [node.cluster_info.member_count for node in self._nodes.values()]
        size_cv = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0.0
        
        return {
            "balance_score": balance_score,
            "utilization_score": utilization_score,
            "avg_depth": avg_depth,
            "depth_variance": depth_variance,
            "size_coefficient_of_variation": size_cv,
            "structure_quality": (balance_score + utilization_score) / 2,
        }
    
    def _remove_cluster(self, cluster_id: str) -> bool:
        """Remove a cluster from hierarchy (for testing purposes)."""
        with self._lock:
            node = self._nodes.get(cluster_id)
            if not node:
                return False
            
            # Remove from parent
            if node.parent:
                node.parent.remove_child(node)
            
            # Remove from children
            for child in node.children[:]:  # Copy list to avoid modification during iteration
                node.remove_child(child)
            
            # Remove from indexes
            cluster_level = node.cluster_info.level
            self._level_index[cluster_level].discard(cluster_id)
            self._root_nodes.discard(cluster_id)
            
            # Remove node
            del self._nodes[cluster_id]
            
            # Clear caches
            self._clear_caches()
            
            return True
    
    def __len__(self) -> int:
        """Return number of clusters in hierarchy."""
        return len(self._nodes)
    
    def __contains__(self, cluster_id: str) -> bool:
        """Check if cluster exists in hierarchy."""
        return cluster_id in self._nodes
    
    def __repr__(self) -> str:
        """String representation of hierarchy."""
        return (
            f"ClusterHierarchy(clusters={len(self._nodes)}, "
            f"levels={len(self.config.levels)}, "
            f"depth={self._compute_max_depth()})"
        )