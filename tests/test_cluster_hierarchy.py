"""
Tests for ClusterHierarchy multi-level clustering structure management.

This module provides comprehensive tests for the hierarchical clustering system,
including hierarchy construction, navigation, operations, and optimization.
"""

import pytest
import numpy as np
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
from unittest.mock import Mock, patch

from coral.clustering.cluster_hierarchy import ClusterHierarchy, HierarchyNode, HierarchyMetrics
from coral.clustering.cluster_types import ClusterInfo, ClusterLevel, ClusteringStrategy, Centroid
from coral.clustering.cluster_config import HierarchyConfig
from coral.core.weight_tensor import WeightTensor


class TestClusterHierarchy:
    """Test suite for ClusterHierarchy class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = HierarchyConfig(
            levels=[ClusterLevel.TENSOR, ClusterLevel.BLOCK, ClusterLevel.LAYER, ClusterLevel.MODEL],
            merge_threshold=0.8,
            split_threshold=0.3,
            enforce_consistency=True,
            max_hierarchy_depth=4
        )
        self.hierarchy = ClusterHierarchy(self.config)
        
        # Create sample clusters for testing
        self.sample_clusters = self._create_sample_clusters()
    
    def _create_sample_clusters(self) -> List[ClusterInfo]:
        """Create sample clusters for testing."""
        clusters = []
        
        # Tensor level clusters
        for i in range(10):
            cluster = ClusterInfo(
                cluster_id=f"tensor_{i}",
                strategy=ClusteringStrategy.KMEANS,
                level=ClusterLevel.TENSOR,
                member_count=5 + i,
                centroid_hash=f"tensor_hash_{i}",
                created_at="2024-01-01T00:00:00Z"
            )
            clusters.append(cluster)
        
        # Block level clusters
        for i in range(5):
            cluster = ClusterInfo(
                cluster_id=f"block_{i}",
                strategy=ClusteringStrategy.HIERARCHICAL,
                level=ClusterLevel.BLOCK,
                member_count=20 + i * 5,
                centroid_hash=f"block_hash_{i}",
                created_at="2024-01-01T00:00:00Z"
            )
            clusters.append(cluster)
        
        # Layer level clusters
        for i in range(3):
            cluster = ClusterInfo(
                cluster_id=f"layer_{i}",
                strategy=ClusteringStrategy.HIERARCHICAL,
                level=ClusterLevel.LAYER,
                member_count=50 + i * 10,
                centroid_hash=f"layer_hash_{i}",
                created_at="2024-01-01T00:00:00Z"
            )
            clusters.append(cluster)
        
        # Model level cluster
        cluster = ClusterInfo(
            cluster_id="model_0",
            strategy=ClusteringStrategy.HIERARCHICAL,
            level=ClusterLevel.MODEL,
            member_count=100,
            centroid_hash="model_hash_0",
            created_at="2024-01-01T00:00:00Z"
        )
        clusters.append(cluster)
        
        return clusters
    
    def _create_sample_weights(self, n_weights: int = 10) -> List[WeightTensor]:
        """Create sample weight tensors for testing."""
        weights = []
        for i in range(n_weights):
            data = np.random.randn(32, 32).astype(np.float32)
            weight = WeightTensor(
                name=f"weight_{i}",
                data=data,
                shape=(32, 32),
                dtype=np.float32
            )
            weights.append(weight)
        return weights
    
    def test_hierarchy_initialization(self):
        """Test hierarchy initialization with different configurations."""
        # Test default initialization
        hierarchy = ClusterHierarchy()
        assert hierarchy.config is not None
        assert len(hierarchy.config.levels) >= 2
        
        # Test custom configuration
        custom_config = HierarchyConfig(
            levels=[ClusterLevel.TENSOR, ClusterLevel.LAYER],
            merge_threshold=0.9,
            split_threshold=0.2
        )
        hierarchy = ClusterHierarchy(custom_config)
        assert hierarchy.config.merge_threshold == 0.9
        assert hierarchy.config.split_threshold == 0.2
        assert len(hierarchy.config.levels) == 2
    
    def test_build_hierarchy_basic(self):
        """Test basic hierarchy construction."""
        # Build hierarchy with sample clusters
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        # Verify hierarchy structure
        assert len(self.hierarchy.get_all_clusters()) > 0
        
        # Check that each level has clusters
        for level in self.config.levels:
            level_clusters = self.hierarchy.get_level_clusters(level)
            if level != ClusterLevel.MODEL:  # Model level might have fewer clusters
                assert len(level_clusters) > 0
    
    def test_build_hierarchy_with_relationships(self):
        """Test hierarchy construction with parent-child relationships."""
        # Create clusters with explicit relationships
        clusters = self.sample_clusters.copy()
        
        # Set up parent-child relationships (block_0 is parent of tensor_0 and tensor_1)
        clusters[10].child_cluster_ids = [clusters[0].cluster_id, clusters[1].cluster_id]
        clusters[0].parent_cluster_id = clusters[10].cluster_id
        clusters[1].parent_cluster_id = clusters[10].cluster_id
        
        self.hierarchy.build_hierarchy(clusters, self.config)
        
        # Verify relationships
        parent = self.hierarchy.navigate_up(clusters[0].cluster_id)
        assert parent is not None
        assert parent.cluster_id == clusters[10].cluster_id
        
        children = self.hierarchy.navigate_down(clusters[10].cluster_id)
        assert len(children) == 2
        child_ids = [child.cluster_id for child in children]
        assert clusters[0].cluster_id in child_ids
        assert clusters[1].cluster_id in child_ids
    
    def test_add_level_basic(self):
        """Test adding clusters at specific hierarchy levels."""
        # Add clusters at tensor level
        tensor_clusters = [c for c in self.sample_clusters if c.level == ClusterLevel.TENSOR]
        self.hierarchy.add_level(ClusterLevel.TENSOR, tensor_clusters)
        
        # Verify clusters were added
        level_clusters = self.hierarchy.get_level_clusters(ClusterLevel.TENSOR)
        assert len(level_clusters) == len(tensor_clusters)
        
        # Add clusters at block level
        block_clusters = [c for c in self.sample_clusters if c.level == ClusterLevel.BLOCK]
        self.hierarchy.add_level(ClusterLevel.BLOCK, block_clusters)
        
        # Verify both levels have clusters
        tensor_level = self.hierarchy.get_level_clusters(ClusterLevel.TENSOR)
        block_level = self.hierarchy.get_level_clusters(ClusterLevel.BLOCK)
        assert len(tensor_level) == len(tensor_clusters)
        assert len(block_level) == len(block_clusters)
    
    def test_add_level_with_validation(self):
        """Test level addition with validation."""
        # Try to add cluster at non-configured level
        invalid_cluster = ClusterInfo(
            cluster_id="invalid",
            strategy=ClusteringStrategy.KMEANS,
            level=ClusterLevel.TENSOR,  # This is valid
            member_count=1
        )
        
        # Should work for valid level
        self.hierarchy.add_level(ClusterLevel.TENSOR, [invalid_cluster])
        assert len(self.hierarchy.get_level_clusters(ClusterLevel.TENSOR)) == 1
        
        # Test duplicate cluster ID
        duplicate_cluster = ClusterInfo(
            cluster_id="invalid",  # Same ID
            strategy=ClusteringStrategy.KMEANS,
            level=ClusterLevel.TENSOR,
            member_count=2
        )
        
        with pytest.raises(ValueError, match="already exists"):
            self.hierarchy.add_level(ClusterLevel.TENSOR, [duplicate_cluster])
    
    def test_get_level_clusters(self):
        """Test retrieving clusters at specific levels."""
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        # Test each level
        for level in self.config.levels:
            level_clusters = self.hierarchy.get_level_clusters(level)
            assert isinstance(level_clusters, list)
            
            # Verify all clusters at this level
            for cluster in level_clusters:
                assert cluster.level == level
        
        # Test non-existent level
        empty_clusters = self.hierarchy.get_level_clusters(ClusterLevel.TENSOR)
        # Should return empty list for levels not in config if no clusters added
        assert isinstance(empty_clusters, list)
    
    def test_navigate_up_basic(self):
        """Test navigating up the hierarchy."""
        # Build hierarchy with relationships
        clusters = self.sample_clusters.copy()
        clusters[0].parent_cluster_id = clusters[10].cluster_id  # tensor_0 -> block_0
        clusters[10].parent_cluster_id = clusters[15].cluster_id  # block_0 -> layer_0
        
        self.hierarchy.build_hierarchy(clusters, self.config)
        
        # Navigate up from tensor to block
        parent = self.hierarchy.navigate_up(clusters[0].cluster_id)
        assert parent is not None
        assert parent.cluster_id == clusters[10].cluster_id
        assert parent.level == ClusterLevel.BLOCK
        
        # Navigate up from block to layer
        grandparent = self.hierarchy.navigate_up(clusters[10].cluster_id)
        assert grandparent is not None
        assert grandparent.cluster_id == clusters[15].cluster_id
        assert grandparent.level == ClusterLevel.LAYER
        
        # Navigate up from root (should return None)
        root_parent = self.hierarchy.navigate_up(clusters[-1].cluster_id)  # model level
        assert root_parent is None
    
    def test_navigate_up_path_to_root(self):
        """Test navigating up to root of hierarchy."""
        # Create deep hierarchy
        clusters = self.sample_clusters.copy()
        clusters[0].parent_cluster_id = clusters[10].cluster_id  # tensor -> block
        clusters[10].parent_cluster_id = clusters[15].cluster_id  # block -> layer
        clusters[15].parent_cluster_id = clusters[18].cluster_id  # layer -> model
        
        self.hierarchy.build_hierarchy(clusters, self.config)
        
        # Get path to root
        path = self.hierarchy.find_path_to_root(clusters[0].cluster_id)
        assert len(path) == 4  # tensor -> block -> layer -> model
        
        # Verify path order
        assert path[0].cluster_id == clusters[0].cluster_id
        assert path[1].cluster_id == clusters[10].cluster_id
        assert path[2].cluster_id == clusters[15].cluster_id
        assert path[3].cluster_id == clusters[18].cluster_id
    
    def test_navigate_down_basic(self):
        """Test navigating down the hierarchy."""
        # Build hierarchy with relationships
        clusters = self.sample_clusters.copy()
        clusters[10].child_cluster_ids = [clusters[0].cluster_id, clusters[1].cluster_id]
        clusters[0].parent_cluster_id = clusters[10].cluster_id
        clusters[1].parent_cluster_id = clusters[10].cluster_id
        
        self.hierarchy.build_hierarchy(clusters, self.config)
        
        # Navigate down from block to tensors
        children = self.hierarchy.navigate_down(clusters[10].cluster_id)
        assert len(children) == 2
        
        child_ids = [child.cluster_id for child in children]
        assert clusters[0].cluster_id in child_ids
        assert clusters[1].cluster_id in child_ids
        
        # Navigate down from leaf (should return empty list)
        leaf_children = self.hierarchy.navigate_down(clusters[0].cluster_id)
        assert len(leaf_children) == 0
    
    def test_navigate_down_all_descendants(self):
        """Test getting all descendants recursively."""
        # Create hierarchy with multiple levels
        clusters = self.sample_clusters.copy()
        
        # Set up multi-level relationships
        # model -> layer -> block -> tensor
        clusters[18].child_cluster_ids = [clusters[15].cluster_id]  # model -> layer
        clusters[15].parent_cluster_id = clusters[18].cluster_id
        clusters[15].child_cluster_ids = [clusters[10].cluster_id, clusters[11].cluster_id]  # layer -> blocks
        clusters[10].parent_cluster_id = clusters[15].cluster_id
        clusters[11].parent_cluster_id = clusters[15].cluster_id
        clusters[10].child_cluster_ids = [clusters[0].cluster_id, clusters[1].cluster_id]  # block -> tensors
        clusters[0].parent_cluster_id = clusters[10].cluster_id
        clusters[1].parent_cluster_id = clusters[10].cluster_id
        
        self.hierarchy.build_hierarchy(clusters, self.config)
        
        # Get all descendants from model level
        descendants = self.hierarchy.get_all_descendants(clusters[18].cluster_id)
        assert len(descendants) >= 4  # At least layer + 2 blocks + 2 tensors
        
        # Verify descendants include all levels
        descendant_levels = set(desc.level for desc in descendants)
        assert ClusterLevel.LAYER in descendant_levels
        assert ClusterLevel.BLOCK in descendant_levels
        assert ClusterLevel.TENSOR in descendant_levels
    
    def test_merge_clusters_basic(self):
        """Test merging clusters into higher level."""
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        # Get some tensor clusters to merge
        tensor_clusters = self.hierarchy.get_level_clusters(ClusterLevel.TENSOR)[:3]
        cluster_ids = [cluster.cluster_id for cluster in tensor_clusters]
        
        # Merge into block level
        merged_cluster = self.hierarchy.merge_clusters(cluster_ids, ClusterLevel.BLOCK)
        
        assert merged_cluster is not None
        assert merged_cluster.level == ClusterLevel.BLOCK
        assert merged_cluster.member_count == sum(c.member_count for c in tensor_clusters)
        
        # Verify merged cluster is in hierarchy
        block_clusters = self.hierarchy.get_level_clusters(ClusterLevel.BLOCK)
        assert any(c.cluster_id == merged_cluster.cluster_id for c in block_clusters)
    
    def test_merge_clusters_with_strategy(self):
        """Test merging clusters with different strategies."""
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        tensor_clusters = self.hierarchy.get_level_clusters(ClusterLevel.TENSOR)[:2]
        cluster_ids = [cluster.cluster_id for cluster in tensor_clusters]
        
        # Test merge with centroid strategy
        merged_centroid = self.hierarchy.merge_clusters(
            cluster_ids, ClusterLevel.BLOCK, strategy="centroid"
        )
        assert merged_centroid is not None
        
        # Test merge with largest strategy
        merged_largest = self.hierarchy.merge_clusters(
            cluster_ids, ClusterLevel.BLOCK, strategy="largest"
        )
        assert merged_largest is not None
        
        # Verify different strategies produce different results
        assert merged_centroid.cluster_id != merged_largest.cluster_id
    
    def test_split_cluster_basic(self):
        """Test splitting cluster into sub-clusters."""
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        # Get a block cluster to split
        block_clusters = self.hierarchy.get_level_clusters(ClusterLevel.BLOCK)
        if not block_clusters:
            pytest.skip("No block clusters available for splitting test")
        
        target_cluster = block_clusters[0]
        
        # Split into tensor level
        split_clusters = self.hierarchy.split_cluster(
            target_cluster.cluster_id, strategy="kmeans", target_level=ClusterLevel.TENSOR, n_splits=3
        )
        
        assert len(split_clusters) == 3
        assert all(cluster.level == ClusterLevel.TENSOR for cluster in split_clusters)
        
        # Verify split clusters are in hierarchy
        tensor_clusters = self.hierarchy.get_level_clusters(ClusterLevel.TENSOR)
        split_ids = [cluster.cluster_id for cluster in split_clusters]
        for split_id in split_ids:
            assert any(c.cluster_id == split_id for c in tensor_clusters)
    
    def test_split_cluster_with_validation(self):
        """Test cluster splitting with validation."""
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        # Try to split non-existent cluster
        with pytest.raises(ValueError, match="not found"):
            self.hierarchy.split_cluster("non_existent", strategy="kmeans")
        
        # Try to split to same level
        tensor_clusters = self.hierarchy.get_level_clusters(ClusterLevel.TENSOR)
        if tensor_clusters:
            with pytest.raises(ValueError, match="Cannot split.*same level"):
                self.hierarchy.split_cluster(
                    tensor_clusters[0].cluster_id, 
                    strategy="kmeans", 
                    target_level=ClusterLevel.TENSOR
                )
    
    def test_promote_cluster_basic(self):
        """Test promoting cluster to higher level."""
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        # Get tensor cluster to promote
        tensor_clusters = self.hierarchy.get_level_clusters(ClusterLevel.TENSOR)
        if not tensor_clusters:
            pytest.skip("No tensor clusters available for promotion test")
        
        target_cluster = tensor_clusters[0]
        original_id = target_cluster.cluster_id
        
        # Promote to block level
        promoted = self.hierarchy.promote_cluster(original_id, ClusterLevel.BLOCK)
        
        assert promoted is not None
        assert promoted.level == ClusterLevel.BLOCK
        assert promoted.member_count == target_cluster.member_count
        
        # Verify original cluster is removed from tensor level
        updated_tensors = self.hierarchy.get_level_clusters(ClusterLevel.TENSOR)
        assert not any(c.cluster_id == original_id for c in updated_tensors)
        
        # Verify promoted cluster is in block level
        block_clusters = self.hierarchy.get_level_clusters(ClusterLevel.BLOCK)
        assert any(c.cluster_id == promoted.cluster_id for c in block_clusters)
    
    def test_promote_cluster_with_validation(self):
        """Test cluster promotion with validation."""
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        # Try to promote non-existent cluster
        with pytest.raises(ValueError, match="not found"):
            self.hierarchy.promote_cluster("non_existent", ClusterLevel.BLOCK)
        
        # Try to promote to same level
        tensor_clusters = self.hierarchy.get_level_clusters(ClusterLevel.TENSOR)
        if tensor_clusters:
            with pytest.raises(ValueError, match="Cannot promote.*same level"):
                self.hierarchy.promote_cluster(
                    tensor_clusters[0].cluster_id, 
                    ClusterLevel.TENSOR
                )
    
    def test_rebalance_level_basic(self):
        """Test rebalancing clusters at a specific level."""
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        # Get current state
        original_clusters = self.hierarchy.get_level_clusters(ClusterLevel.TENSOR)
        original_count = len(original_clusters)
        
        if original_count == 0:
            pytest.skip("No tensor clusters available for rebalancing test")
        
        # Rebalance tensor level
        rebalanced = self.hierarchy.rebalance_level(ClusterLevel.TENSOR)
        
        assert rebalanced is not None
        assert "rebalanced_clusters" in rebalanced
        assert "metrics" in rebalanced
        
        # Verify clusters still exist (though may be reorganized)
        current_clusters = self.hierarchy.get_level_clusters(ClusterLevel.TENSOR)
        assert len(current_clusters) > 0
    
    def test_rebalance_level_with_strategy(self):
        """Test rebalancing with different strategies."""
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        # Test size-based rebalancing
        size_result = self.hierarchy.rebalance_level(
            ClusterLevel.TENSOR, strategy="size", target_size=5
        )
        assert size_result is not None
        
        # Test quality-based rebalancing
        quality_result = self.hierarchy.rebalance_level(
            ClusterLevel.TENSOR, strategy="quality", quality_threshold=0.8
        )
        assert quality_result is not None
    
    def test_find_path_to_root(self):
        """Test finding path from cluster to root."""
        # Create hierarchy with clear path
        clusters = self.sample_clusters.copy()
        clusters[0].parent_cluster_id = clusters[10].cluster_id  # tensor -> block
        clusters[10].parent_cluster_id = clusters[15].cluster_id  # block -> layer
        clusters[15].parent_cluster_id = clusters[18].cluster_id  # layer -> model
        
        self.hierarchy.build_hierarchy(clusters, self.config)
        
        # Find path from tensor to root
        path = self.hierarchy.find_path_to_root(clusters[0].cluster_id)
        
        assert len(path) == 4
        assert path[0].cluster_id == clusters[0].cluster_id
        assert path[1].cluster_id == clusters[10].cluster_id
        assert path[2].cluster_id == clusters[15].cluster_id
        assert path[3].cluster_id == clusters[18].cluster_id
        
        # Test path from root (should return just the root)
        root_path = self.hierarchy.find_path_to_root(clusters[18].cluster_id)
        assert len(root_path) == 1
        assert root_path[0].cluster_id == clusters[18].cluster_id
    
    def test_find_common_ancestor(self):
        """Test finding common ancestor of two clusters."""
        # Create hierarchy with common ancestor
        clusters = self.sample_clusters.copy()
        
        # Create tree structure:
        #     layer_0
        #    /       \
        # block_0   block_1
        #   |         |
        # tensor_0  tensor_1
        
        clusters[15].child_cluster_ids = [clusters[10].cluster_id, clusters[11].cluster_id]
        clusters[10].parent_cluster_id = clusters[15].cluster_id
        clusters[11].parent_cluster_id = clusters[15].cluster_id
        clusters[10].child_cluster_ids = [clusters[0].cluster_id]
        clusters[11].child_cluster_ids = [clusters[1].cluster_id]
        clusters[0].parent_cluster_id = clusters[10].cluster_id
        clusters[1].parent_cluster_id = clusters[11].cluster_id
        
        self.hierarchy.build_hierarchy(clusters, self.config)
        
        # Find common ancestor of tensor_0 and tensor_1
        ancestor = self.hierarchy.find_common_ancestor(
            clusters[0].cluster_id, clusters[1].cluster_id
        )
        
        assert ancestor is not None
        assert ancestor.cluster_id == clusters[15].cluster_id  # layer_0
        assert ancestor.level == ClusterLevel.LAYER
        
        # Test ancestor of cluster with itself
        self_ancestor = self.hierarchy.find_common_ancestor(
            clusters[0].cluster_id, clusters[0].cluster_id
        )
        assert self_ancestor.cluster_id == clusters[0].cluster_id
    
    def test_search_by_criteria(self):
        """Test searching clusters by various criteria."""
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        # Search by level
        tensor_results = self.hierarchy.search_by_criteria({"level": ClusterLevel.TENSOR})
        assert all(cluster.level == ClusterLevel.TENSOR for cluster in tensor_results)
        
        # Search by strategy
        kmeans_results = self.hierarchy.search_by_criteria({"strategy": ClusteringStrategy.KMEANS})
        assert all(cluster.strategy == ClusteringStrategy.KMEANS for cluster in kmeans_results)
        
        # Search by member count range
        large_results = self.hierarchy.search_by_criteria({"min_member_count": 50})
        assert all(cluster.member_count >= 50 for cluster in large_results)
        
        # Search by multiple criteria
        complex_results = self.hierarchy.search_by_criteria({
            "level": ClusterLevel.BLOCK,
            "strategy": ClusteringStrategy.HIERARCHICAL,
            "min_member_count": 20
        })
        assert all(
            cluster.level == ClusterLevel.BLOCK and 
            cluster.strategy == ClusteringStrategy.HIERARCHICAL and
            cluster.member_count >= 20
            for cluster in complex_results
        )
    
    def test_search_by_criteria_advanced(self):
        """Test advanced search functionality."""
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        # Search with custom filter function
        def custom_filter(cluster):
            return cluster.cluster_id.startswith("tensor_") and cluster.member_count > 5
        
        custom_results = self.hierarchy.search_by_criteria({"custom_filter": custom_filter})
        assert all(
            cluster.cluster_id.startswith("tensor_") and cluster.member_count > 5
            for cluster in custom_results
        )
        
        # Search with regex pattern
        pattern_results = self.hierarchy.search_by_criteria({"name_pattern": r"block_\d+"})
        assert all(cluster.cluster_id.startswith("block_") for cluster in pattern_results)
    
    def test_optimize_hierarchy_basic(self):
        """Test basic hierarchy optimization."""
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        # Get initial metrics
        initial_metrics = self.hierarchy.compute_hierarchy_metrics()
        
        # Optimize hierarchy
        optimization_result = self.hierarchy.optimize_hierarchy()
        
        assert "optimized" in optimization_result
        assert "initial_metrics" in optimization_result
        assert "final_metrics" in optimization_result
        
        # Verify metrics are computed
        final_metrics = optimization_result["final_metrics"]
        assert "total_clusters" in final_metrics
        assert "depth" in final_metrics
        assert "balance_score" in final_metrics
    
    def test_optimize_hierarchy_with_objectives(self):
        """Test hierarchy optimization with specific objectives."""
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        # Optimize for balance
        balance_result = self.hierarchy.optimize_hierarchy(objectives=["balance"])
        assert balance_result is not None
        
        # Optimize for depth
        depth_result = self.hierarchy.optimize_hierarchy(objectives=["minimize_depth"])
        assert depth_result is not None
        
        # Optimize for multiple objectives
        multi_result = self.hierarchy.optimize_hierarchy(
            objectives=["balance", "minimize_depth", "maximize_utilization"]
        )
        assert multi_result is not None
    
    def test_validate_consistency_basic(self):
        """Test hierarchy consistency validation."""
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        # Validate consistent hierarchy
        validation_result = self.hierarchy.validate_consistency()
        
        assert "is_valid" in validation_result
        assert "issues" in validation_result
        assert isinstance(validation_result["issues"], list)
        
        # Should be valid for properly constructed hierarchy
        assert validation_result["is_valid"] is True
        assert len(validation_result["issues"]) == 0
    
    def test_validate_consistency_with_issues(self):
        """Test consistency validation with hierarchy issues."""
        # Build hierarchy with intentional inconsistencies
        clusters = self.sample_clusters.copy()
        
        # Create orphaned child (parent doesn't exist)
        clusters[0].parent_cluster_id = "non_existent_parent"
        
        self.hierarchy.build_hierarchy(clusters, self.config)
        
        # Validate should find issues
        validation_result = self.hierarchy.validate_consistency()
        
        assert validation_result["is_valid"] is False
        assert len(validation_result["issues"]) > 0
        
        # Check for specific issue types
        issues = validation_result["issues"]
        assert any("orphaned" in issue.lower() for issue in issues)
    
    def test_compute_hierarchy_metrics(self):
        """Test computation of hierarchy metrics."""
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        metrics = self.hierarchy.compute_hierarchy_metrics()
        
        # Check required metrics (metrics is HierarchyMetrics object)
        assert hasattr(metrics, "total_clusters")
        assert hasattr(metrics, "depth")
        assert hasattr(metrics, "balance_score")
        assert hasattr(metrics, "utilization_score")
        assert hasattr(metrics, "level_distribution")
        assert hasattr(metrics, "avg_branching_factor")
        assert hasattr(metrics, "leaf_ratio")
        
        # Validate metric ranges
        assert metrics.depth >= 0
        assert 0.0 <= metrics.balance_score <= 1.0
        assert 0.0 <= metrics.utilization_score <= 1.0
        assert 0.0 <= metrics.leaf_ratio <= 1.0
        assert metrics.avg_branching_factor >= 0
    
    def test_compute_hierarchy_metrics_advanced(self):
        """Test advanced hierarchy metrics computation."""
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        # Compute metrics with additional parameters
        detailed_metrics = self.hierarchy.compute_hierarchy_metrics(
            include_level_stats=True,
            include_connectivity=True,
            include_quality_metrics=True
        )
        
        # Convert to dict to access additional metrics
        metrics_dict = detailed_metrics.to_dict()
        
        # Check additional metrics
        assert "level_stats" in metrics_dict
        assert "connectivity_metrics" in metrics_dict
        assert "quality_metrics" in metrics_dict
        
        # Verify level stats
        level_stats = metrics_dict["level_stats"]
        for level in self.config.levels:
            if level.value in level_stats:
                assert "count" in level_stats[level.value]
                assert "avg_size" in level_stats[level.value]
    
    def test_suggest_restructuring_basic(self):
        """Test hierarchy restructuring suggestions."""
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        suggestions = self.hierarchy.suggest_restructuring()
        
        assert "suggestions" in suggestions
        assert "confidence" in suggestions
        assert "estimated_improvement" in suggestions
        
        # Should be list of suggestions
        assert isinstance(suggestions["suggestions"], list)
        
        # Each suggestion should have required fields
        for suggestion in suggestions["suggestions"]:
            assert "type" in suggestion
            assert "description" in suggestion
            assert "priority" in suggestion
    
    def test_suggest_restructuring_with_issues(self):
        """Test restructuring suggestions for problematic hierarchy."""
        # Create unbalanced hierarchy
        unbalanced_clusters = []
        
        # Create many clusters at tensor level, few at others
        for i in range(20):
            cluster = ClusterInfo(
                cluster_id=f"tensor_{i}",
                strategy=ClusteringStrategy.KMEANS,
                level=ClusterLevel.TENSOR,
                member_count=1,  # Very small clusters
            )
            unbalanced_clusters.append(cluster)
        
        # Only one cluster at higher levels
        block_cluster = ClusterInfo(
            cluster_id="block_0",
            strategy=ClusteringStrategy.HIERARCHICAL,
            level=ClusterLevel.BLOCK,
            member_count=100,  # Very large cluster
        )
        unbalanced_clusters.append(block_cluster)
        
        self.hierarchy.build_hierarchy(unbalanced_clusters, self.config)
        
        suggestions = self.hierarchy.suggest_restructuring()
        
        # Should suggest improvements for unbalanced hierarchy
        assert len(suggestions["suggestions"]) > 0
        
        # Should suggest rebalancing or some kind of optimization
        suggestion_types = [s["type"] for s in suggestions["suggestions"]]
        
        # Look for any optimization suggestions (rebalance, improve_utilization, etc.)
        optimization_suggestions = ["rebalance", "improve_utilization", "consolidate_level", "split_level"]
        assert any(any(opt in stype.lower() for opt in optimization_suggestions) for stype in suggestion_types)
    
    def test_thread_safety_basic(self):
        """Test basic thread safety of hierarchy operations."""
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        results = []
        errors = []
        
        def worker_read():
            try:
                for _ in range(10):
                    clusters = self.hierarchy.get_level_clusters(ClusterLevel.TENSOR)
                    results.append(len(clusters))
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)
        
        def worker_write():
            try:
                for i in range(5):
                    new_cluster = ClusterInfo(
                        cluster_id=f"thread_tensor_{i}",
                        strategy=ClusteringStrategy.KMEANS,
                        level=ClusterLevel.TENSOR,
                        member_count=1
                    )
                    self.hierarchy.add_level(ClusterLevel.TENSOR, [new_cluster])
                    time.sleep(0.002)  # Small delay
            except Exception as e:
                errors.append(e)
        
        # Run concurrent operations
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=worker_read))
        threads.append(threading.Thread(target=worker_write))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Check for errors
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) > 0, "No results from read operations"
    
    def test_thread_safety_hierarchy_modification(self):
        """Test thread safety during hierarchy modifications."""
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        errors = []
        
        def worker_navigate():
            try:
                for _ in range(20):
                    # Try to navigate hierarchy
                    clusters = self.hierarchy.get_all_clusters()
                    if clusters:
                        cluster = clusters[0]
                        self.hierarchy.navigate_up(cluster.cluster_id)
                        self.hierarchy.navigate_down(cluster.cluster_id)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def worker_modify():
            try:
                for i in range(10):
                    # Add and remove clusters
                    new_cluster = ClusterInfo(
                        cluster_id=f"temp_{i}",
                        strategy=ClusteringStrategy.KMEANS,
                        level=ClusterLevel.TENSOR,
                        member_count=1
                    )
                    self.hierarchy.add_level(ClusterLevel.TENSOR, [new_cluster])
                    time.sleep(0.001)
                    
                    # Remove some clusters
                    current_clusters = self.hierarchy.get_level_clusters(ClusterLevel.TENSOR)
                    if len(current_clusters) > 5:
                        self.hierarchy._remove_cluster(current_clusters[-1].cluster_id)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent operations
        threads = []
        for _ in range(2):
            threads.append(threading.Thread(target=worker_navigate))
        threads.append(threading.Thread(target=worker_modify))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Check for errors
        assert len(errors) == 0, f"Thread safety errors: {errors}"
    
    def test_serialization_basic(self):
        """Test hierarchy serialization and deserialization."""
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        # Serialize hierarchy
        serialized = self.hierarchy.to_dict()
        
        assert "config" in serialized
        assert "clusters" in serialized
        assert "hierarchy_map" in serialized
        assert "metadata" in serialized
        
        # Deserialize hierarchy
        new_hierarchy = ClusterHierarchy.from_dict(serialized)
        
        # Verify deserialized hierarchy
        assert len(new_hierarchy.get_all_clusters()) == len(self.hierarchy.get_all_clusters())
        assert new_hierarchy.config.merge_threshold == self.hierarchy.config.merge_threshold
        
        # Verify specific clusters
        for level in self.config.levels:
            original_clusters = self.hierarchy.get_level_clusters(level)
            deserialized_clusters = new_hierarchy.get_level_clusters(level)
            assert len(original_clusters) == len(deserialized_clusters)
    
    def test_serialization_with_relationships(self):
        """Test serialization with parent-child relationships."""
        # Create hierarchy with relationships
        clusters = self.sample_clusters.copy()
        clusters[0].parent_cluster_id = clusters[10].cluster_id
        clusters[10].child_cluster_ids = [clusters[0].cluster_id]
        
        self.hierarchy.build_hierarchy(clusters, self.config)
        
        # Serialize and deserialize
        serialized = self.hierarchy.to_dict()
        new_hierarchy = ClusterHierarchy.from_dict(serialized)
        
        # Verify relationships are preserved
        parent = new_hierarchy.navigate_up(clusters[0].cluster_id)
        assert parent is not None
        assert parent.cluster_id == clusters[10].cluster_id
        
        children = new_hierarchy.navigate_down(clusters[10].cluster_id)
        assert len(children) == 1
        assert children[0].cluster_id == clusters[0].cluster_id
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large hierarchies."""
        # Create large hierarchy
        large_clusters = []
        
        # Create many clusters at each level
        for level in [ClusterLevel.TENSOR, ClusterLevel.BLOCK, ClusterLevel.LAYER]:
            for i in range(100):
                cluster = ClusterInfo(
                    cluster_id=f"{level.value}_{i}",
                    strategy=ClusteringStrategy.KMEANS,
                    level=level,
                    member_count=10
                )
                large_clusters.append(cluster)
        
        # Build hierarchy
        start_time = time.time()
        self.hierarchy.build_hierarchy(large_clusters, self.config)
        build_time = time.time() - start_time
        
        # Verify reasonable build time (should be < 1 second for 300 clusters)
        assert build_time < 1.0, f"Hierarchy build took too long: {build_time:.3f}s"
        
        # Test navigation performance
        start_time = time.time()
        for _ in range(100):
            clusters = self.hierarchy.get_level_clusters(ClusterLevel.TENSOR)
            if clusters:
                self.hierarchy.navigate_up(clusters[0].cluster_id)
        navigation_time = time.time() - start_time
        
        # Navigation should be efficient
        assert navigation_time < 0.1, f"Navigation took too long: {navigation_time:.3f}s"
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with empty hierarchy
        empty_hierarchy = ClusterHierarchy(self.config)
        assert len(empty_hierarchy.get_all_clusters()) == 0
        assert empty_hierarchy.navigate_up("non_existent") is None
        assert len(empty_hierarchy.navigate_down("non_existent")) == 0
        
        # Test with single cluster
        single_cluster = [ClusterInfo(
            cluster_id="single",
            strategy=ClusteringStrategy.KMEANS,
            level=ClusterLevel.TENSOR,
            member_count=1
        )]
        
        single_hierarchy = ClusterHierarchy(self.config)
        single_hierarchy.build_hierarchy(single_cluster, self.config)
        
        assert len(single_hierarchy.get_all_clusters()) == 1
        assert single_hierarchy.navigate_up("single") is None
        assert len(single_hierarchy.navigate_down("single")) == 0
        
        # Test with invalid cluster IDs
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        assert self.hierarchy.navigate_up("invalid_id") is None
        assert len(self.hierarchy.navigate_down("invalid_id")) == 0
        assert self.hierarchy.find_common_ancestor("invalid1", "invalid2") is None
    
    def test_integration_with_cluster_index(self):
        """Test integration with ClusterIndex."""
        # This test would be more complete with actual ClusterIndex integration
        # For now, test the interface compatibility
        
        self.hierarchy.build_hierarchy(self.sample_clusters, self.config)
        
        # Simulate cluster index integration
        cluster_index_data = {}
        
        for cluster in self.hierarchy.get_all_clusters():
            if cluster.centroid_hash:
                cluster_index_data[cluster.cluster_id] = {
                    "centroid_hash": cluster.centroid_hash,
                    "level": cluster.level,
                    "parent": cluster.parent_cluster_id,
                    "children": cluster.child_cluster_ids or []
                }
        
        # Verify data structure is compatible
        assert len(cluster_index_data) > 0
        
        for cluster_id, data in cluster_index_data.items():
            assert "centroid_hash" in data
            assert "level" in data
            assert "parent" in data
            assert "children" in data


class TestHierarchyNode:
    """Test suite for HierarchyNode internal class."""
    
    def test_node_creation(self):
        """Test hierarchy node creation."""
        cluster_info = ClusterInfo(
            cluster_id="test_cluster",
            strategy=ClusteringStrategy.KMEANS,
            level=ClusterLevel.TENSOR,
            member_count=10
        )
        
        node = HierarchyNode(cluster_info)
        
        assert node.cluster_info == cluster_info
        assert node.parent is None
        assert len(node.children) == 0
        assert node.depth == 0
    
    def test_node_relationships(self):
        """Test node parent-child relationships."""
        parent_info = ClusterInfo(
            cluster_id="parent",
            strategy=ClusteringStrategy.HIERARCHICAL,
            level=ClusterLevel.BLOCK,
            member_count=20
        )
        
        child_info = ClusterInfo(
            cluster_id="child",
            strategy=ClusteringStrategy.KMEANS,
            level=ClusterLevel.TENSOR,
            member_count=10
        )
        
        parent_node = HierarchyNode(parent_info)
        child_node = HierarchyNode(child_info)
        
        # Add child to parent
        parent_node.add_child(child_node)
        
        assert len(parent_node.children) == 1
        assert child_node in parent_node.children
        assert child_node.parent == parent_node
        assert child_node.depth == 1
    
    def test_node_depth_calculation(self):
        """Test depth calculation in node hierarchy."""
        # Create multi-level hierarchy
        root = HierarchyNode(ClusterInfo(
            cluster_id="root",
            strategy=ClusteringStrategy.HIERARCHICAL,
            level=ClusterLevel.MODEL,
            member_count=100
        ))
        
        level1 = HierarchyNode(ClusterInfo(
            cluster_id="level1",
            strategy=ClusteringStrategy.HIERARCHICAL,
            level=ClusterLevel.LAYER,
            member_count=50
        ))
        
        level2 = HierarchyNode(ClusterInfo(
            cluster_id="level2",
            strategy=ClusteringStrategy.KMEANS,
            level=ClusterLevel.TENSOR,
            member_count=25
        ))
        
        # Build hierarchy
        root.add_child(level1)
        level1.add_child(level2)
        
        # Verify depths
        assert root.depth == 0
        assert level1.depth == 1
        assert level2.depth == 2
    
    def test_node_removal(self):
        """Test node removal from hierarchy."""
        parent = HierarchyNode(ClusterInfo(
            cluster_id="parent",
            strategy=ClusteringStrategy.HIERARCHICAL,
            level=ClusterLevel.BLOCK,
            member_count=20
        ))
        
        child1 = HierarchyNode(ClusterInfo(
            cluster_id="child1",
            strategy=ClusteringStrategy.KMEANS,
            level=ClusterLevel.TENSOR,
            member_count=10
        ))
        
        child2 = HierarchyNode(ClusterInfo(
            cluster_id="child2",
            strategy=ClusteringStrategy.KMEANS,
            level=ClusterLevel.TENSOR,
            member_count=10
        ))
        
        # Add children
        parent.add_child(child1)
        parent.add_child(child2)
        
        assert len(parent.children) == 2
        
        # Remove child1
        parent.remove_child(child1)
        
        assert len(parent.children) == 1
        assert child1 not in parent.children
        assert child2 in parent.children
        assert child1.parent is None


class TestHierarchyMetrics:
    """Test suite for HierarchyMetrics class."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = HierarchyMetrics()
        
        # Check default values
        assert metrics.total_clusters == 0
        assert metrics.depth == 0
        assert metrics.balance_score == 0.0
        assert metrics.utilization_score == 0.0
        assert metrics.avg_branching_factor == 0.0
        assert metrics.leaf_ratio == 0.0
        assert isinstance(metrics.level_distribution, dict)
    
    def test_metrics_computation(self):
        """Test metrics computation from hierarchy."""
        # Create sample hierarchy for metrics
        config = HierarchyConfig(
            levels=[ClusterLevel.TENSOR, ClusterLevel.BLOCK, ClusterLevel.LAYER, ClusterLevel.MODEL]
        )
        hierarchy = ClusterHierarchy()
        
        # Add clusters at different levels
        clusters = []
        
        # Tensor level
        for i in range(10):
            cluster = ClusterInfo(
                cluster_id=f"tensor_{i}",
                strategy=ClusteringStrategy.KMEANS,
                level=ClusterLevel.TENSOR,
                member_count=5
            )
            clusters.append(cluster)
        
        # Block level
        for i in range(5):
            cluster = ClusterInfo(
                cluster_id=f"block_{i}",
                strategy=ClusteringStrategy.HIERARCHICAL,
                level=ClusterLevel.BLOCK,
                member_count=10
            )
            clusters.append(cluster)
        
        hierarchy.build_hierarchy(clusters, config)
        
        # Compute metrics
        metrics = HierarchyMetrics.from_hierarchy(hierarchy)
        
        assert metrics.total_clusters == 15
        assert metrics.depth >= 0
        assert ClusterLevel.TENSOR in metrics.level_distribution
        assert ClusterLevel.BLOCK in metrics.level_distribution
        assert metrics.level_distribution[ClusterLevel.TENSOR] == 10
        assert metrics.level_distribution[ClusterLevel.BLOCK] == 5
    
    def test_metrics_validation(self):
        """Test metrics validation."""
        # Valid metrics
        valid_metrics = HierarchyMetrics(
            total_clusters=10,
            depth=3,
            balance_score=0.8,
            utilization_score=0.9,
            avg_branching_factor=2.5,
            leaf_ratio=0.6
        )
        
        assert valid_metrics.is_valid()
        
        # Invalid metrics (negative values)
        invalid_metrics = HierarchyMetrics(
            total_clusters=-1,
            depth=-1,
            balance_score=-0.5,
            utilization_score=1.5,
            avg_branching_factor=-1.0,
            leaf_ratio=2.0
        )
        
        assert not invalid_metrics.is_valid()
    
    def test_metrics_serialization(self):
        """Test metrics serialization."""
        metrics = HierarchyMetrics(
            total_clusters=10,
            depth=3,
            balance_score=0.8,
            utilization_score=0.9,
            avg_branching_factor=2.5,
            leaf_ratio=0.6,
            level_distribution={ClusterLevel.TENSOR: 8, ClusterLevel.BLOCK: 2}
        )
        
        # Serialize
        serialized = metrics.to_dict()
        
        assert "total_clusters" in serialized
        assert "depth" in serialized
        assert "balance_score" in serialized
        assert "level_distribution" in serialized
        
        # Deserialize
        deserialized = HierarchyMetrics.from_dict(serialized)
        
        assert deserialized.total_clusters == metrics.total_clusters
        assert deserialized.depth == metrics.depth
        assert deserialized.balance_score == metrics.balance_score
        assert deserialized.level_distribution == metrics.level_distribution