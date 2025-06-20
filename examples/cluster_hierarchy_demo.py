#!/usr/bin/env python3
"""
Demonstration of ClusterHierarchy multi-level clustering structure management.

This example shows how to use the ClusterHierarchy component for:
- Building multi-level cluster hierarchies
- Navigating the hierarchy structure
- Performing cluster operations (merge, split, promote)
- Optimizing hierarchy structure
- Validating hierarchy consistency
"""

import numpy as np
from coral.clustering import (
    ClusterHierarchy, 
    ClusterInfo, 
    ClusterLevel, 
    ClusteringStrategy, 
    HierarchyConfig
)
from coral.core import WeightTensor


def create_sample_clusters():
    """Create sample clusters for demonstration."""
    clusters = []
    
    # Tensor level clusters
    for i in range(8):
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
    for i in range(4):
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
    for i in range(2):
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


def create_hierarchy_with_relationships(clusters):
    """Set up parent-child relationships for demonstration."""
    # Create relationships: model -> layers -> blocks -> tensors
    
    # Model has layer children
    clusters[-1].child_cluster_ids = ["layer_0", "layer_1"]
    
    # Layers have block children
    clusters[10].parent_cluster_id = "model_0"  # layer_0
    clusters[10].child_cluster_ids = ["block_0", "block_1"]
    
    clusters[11].parent_cluster_id = "model_0"  # layer_1
    clusters[11].child_cluster_ids = ["block_2", "block_3"]
    
    # Blocks have tensor children
    for i, block_idx in enumerate([8, 9, 10, 11]):  # block_0 to block_3
        block_cluster = clusters[block_idx]
        layer_parent = "layer_0" if i < 2 else "layer_1"
        block_cluster.parent_cluster_id = layer_parent
        
        # Each block has 2 tensor children
        tensor_children = [f"tensor_{i*2}", f"tensor_{i*2+1}"]
        block_cluster.child_cluster_ids = tensor_children
        
        # Set tensor parents
        for tensor_id in tensor_children:
            for cluster in clusters[:8]:  # tensor clusters
                if cluster.cluster_id == tensor_id:
                    cluster.parent_cluster_id = block_cluster.cluster_id
    
    return clusters


def demonstrate_hierarchy_basics():
    """Demonstrate basic hierarchy operations."""
    print("=== ClusterHierarchy Basic Operations ===")
    
    # Create configuration
    config = HierarchyConfig(
        levels=[ClusterLevel.TENSOR, ClusterLevel.BLOCK, ClusterLevel.LAYER, ClusterLevel.MODEL],
        merge_threshold=0.8,
        split_threshold=0.3,
        enforce_consistency=True
    )
    
    # Create hierarchy
    hierarchy = ClusterHierarchy(config)
    
    # Create sample clusters with relationships
    clusters = create_sample_clusters()
    clusters = create_hierarchy_with_relationships(clusters)
    
    # Build hierarchy
    print(f"Building hierarchy with {len(clusters)} clusters...")
    hierarchy.build_hierarchy(clusters, config)
    
    # Show hierarchy structure
    print(f"Hierarchy contains {len(hierarchy)} clusters")
    print(f"Maximum depth: {hierarchy._compute_max_depth()}")
    
    # Show clusters at each level
    for level in config.levels:
        level_clusters = hierarchy.get_level_clusters(level)
        print(f"{level.value} level: {len(level_clusters)} clusters")
        for cluster in level_clusters[:3]:  # Show first 3
            print(f"  - {cluster.cluster_id} (members: {cluster.member_count})")
    
    return hierarchy


def demonstrate_navigation():
    """Demonstrate hierarchy navigation."""
    print("\n=== Hierarchy Navigation ===")
    
    hierarchy = demonstrate_hierarchy_basics()
    
    # Navigate up from tensor to root
    print("Navigating up from tensor_0:")
    path_to_root = hierarchy.find_path_to_root("tensor_0")
    for i, cluster in enumerate(path_to_root):
        indent = "  " * i
        print(f"{indent}{cluster.cluster_id} ({cluster.level.value})")
    
    # Navigate down from model
    print("\nNavigating down from model_0:")
    children = hierarchy.navigate_down("model_0")
    for child in children:
        print(f"  Child: {child.cluster_id} ({child.level.value})")
        
        # Show grandchildren
        grandchildren = hierarchy.navigate_down(child.cluster_id)
        for grandchild in grandchildren:
            print(f"    Grandchild: {grandchild.cluster_id} ({grandchild.level.value})")
    
    # Find common ancestor
    print("\nFinding common ancestor of tensor_0 and tensor_2:")
    ancestor = hierarchy.find_common_ancestor("tensor_0", "tensor_2")
    if ancestor:
        print(f"Common ancestor: {ancestor.cluster_id} ({ancestor.level.value})")
    
    # Get all descendants
    print("\nAll descendants of layer_0:")
    descendants = hierarchy.get_all_descendants("layer_0")
    for desc in descendants:
        print(f"  - {desc.cluster_id} ({desc.level.value})")


def demonstrate_cluster_operations():
    """Demonstrate cluster merge, split, and promote operations."""
    print("\n=== Cluster Operations ===")
    
    hierarchy = demonstrate_hierarchy_basics()
    
    # Merge operation
    print("Merging tensor_0 and tensor_1 into block level:")
    merged_cluster = hierarchy.merge_clusters(
        ["tensor_0", "tensor_1"], 
        ClusterLevel.BLOCK, 
        strategy="centroid"
    )
    if merged_cluster:
        print(f"Created merged cluster: {merged_cluster.cluster_id}")
        print(f"  Members: {merged_cluster.member_count}")
        print(f"  Level: {merged_cluster.level.value}")
    
    # Split operation
    print("\nSplitting block_0 into tensor level:")
    try:
        split_clusters = hierarchy.split_cluster(
            "block_0", 
            strategy="size_based", 
            target_level=ClusterLevel.TENSOR,
            n_splits=3
        )
        print(f"Split into {len(split_clusters)} clusters:")
        for split_cluster in split_clusters:
            print(f"  - {split_cluster.cluster_id} (members: {split_cluster.member_count})")
    except ValueError as e:
        print(f"Split operation failed: {e}")
    
    # Promote operation
    print("\nPromoting tensor_2 to block level:")
    promoted = hierarchy.promote_cluster("tensor_2", ClusterLevel.BLOCK)
    if promoted:
        print(f"Promoted {promoted.cluster_id} to {promoted.level.value}")


def demonstrate_search_and_metrics():
    """Demonstrate search functionality and metrics computation."""
    print("\n=== Search and Metrics ===")
    
    hierarchy = demonstrate_hierarchy_basics()
    
    # Search by criteria
    print("Searching for clusters with > 20 members:")
    large_clusters = hierarchy.search_by_criteria({"min_member_count": 20})
    for cluster in large_clusters:
        print(f"  - {cluster.cluster_id}: {cluster.member_count} members")
    
    # Search by level and strategy
    print("\nSearching for hierarchical clusters at block level:")
    hierarchical_blocks = hierarchy.search_by_criteria({
        "level": ClusterLevel.BLOCK,
        "strategy": ClusteringStrategy.HIERARCHICAL
    })
    for cluster in hierarchical_blocks:
        print(f"  - {cluster.cluster_id}")
    
    # Compute metrics
    print("\nHierarchy metrics:")
    metrics = hierarchy.compute_hierarchy_metrics(
        include_level_stats=True,
        include_connectivity=True
    )
    
    print(f"  Total clusters: {metrics.total_clusters}")
    print(f"  Depth: {metrics.depth}")
    print(f"  Balance score: {metrics.balance_score:.3f}")
    print(f"  Utilization score: {metrics.utilization_score:.3f}")
    print(f"  Leaf ratio: {metrics.leaf_ratio:.3f}")
    print(f"  Avg branching factor: {metrics.avg_branching_factor:.3f}")
    
    # Level distribution
    print("\nLevel distribution:")
    for level, count in metrics.level_distribution.items():
        print(f"  {level.value}: {count} clusters")


def demonstrate_optimization():
    """Demonstrate hierarchy optimization."""
    print("\n=== Hierarchy Optimization ===")
    
    hierarchy = demonstrate_hierarchy_basics()
    
    # Get initial metrics
    initial_metrics = hierarchy.compute_hierarchy_metrics()
    print(f"Initial balance score: {initial_metrics.balance_score:.3f}")
    print(f"Initial utilization score: {initial_metrics.utilization_score:.3f}")
    
    # Optimize hierarchy
    print("\nOptimizing hierarchy...")
    optimization_result = hierarchy.optimize_hierarchy(
        objectives=["balance", "maximize_utilization"]
    )
    
    print(f"Optimization completed in {optimization_result['optimization_time']:.3f}s")
    print(f"Optimizations made: {len(optimization_result['optimizations_made'])}")
    
    # Show improvements
    final_metrics = optimization_result['final_metrics']
    print(f"Final balance score: {final_metrics['balance_score']:.3f}")
    print(f"Final utilization score: {final_metrics['utilization_score']:.3f}")
    
    # Restructuring suggestions
    print("\nRestructuring suggestions:")
    suggestions = hierarchy.suggest_restructuring()
    for suggestion in suggestions['suggestions'][:3]:  # Show first 3
        print(f"  - {suggestion['type']}: {suggestion['description']}")
        print(f"    Priority: {suggestion['priority']}, Confidence: {suggestion['confidence']:.2f}")


def demonstrate_rebalancing():
    """Demonstrate level rebalancing."""
    print("\n=== Level Rebalancing ===")
    
    hierarchy = demonstrate_hierarchy_basics()
    
    # Show initial state
    tensor_clusters = hierarchy.get_level_clusters(ClusterLevel.TENSOR)
    print(f"Initial tensor clusters: {len(tensor_clusters)}")
    sizes = [c.member_count for c in tensor_clusters]
    print(f"Size range: {min(sizes)} - {max(sizes)}")
    
    # Rebalance tensor level
    print("\nRebalancing tensor level...")
    rebalance_result = hierarchy.rebalance_level(
        ClusterLevel.TENSOR,
        strategy="size",
        target_size=8
    )
    
    print(f"Changes made: {rebalance_result['changes']}")
    print(f"Original count: {rebalance_result['metrics']['original_count']}")
    print(f"Final count: {rebalance_result['metrics']['final_count']}")
    print(f"Average size: {rebalance_result['metrics']['avg_cluster_size']:.1f}")


def demonstrate_validation():
    """Demonstrate hierarchy validation."""
    print("\n=== Hierarchy Validation ===")
    
    hierarchy = demonstrate_hierarchy_basics()
    
    # Validate consistency
    validation_result = hierarchy.validate_consistency()
    
    print(f"Hierarchy is valid: {validation_result['is_valid']}")
    print(f"Total clusters: {validation_result['total_clusters']}")
    print(f"Root count: {validation_result['root_count']}")
    print(f"Max depth: {validation_result['max_depth']}")
    
    if validation_result['issues']:
        print("Issues found:")
        for issue in validation_result['issues']:
            print(f"  - {issue}")
    else:
        print("No issues found!")


def demonstrate_serialization():
    """Demonstrate hierarchy serialization."""
    print("\n=== Serialization ===")
    
    hierarchy = demonstrate_hierarchy_basics()
    
    # Serialize hierarchy
    print("Serializing hierarchy...")
    serialized = hierarchy.to_dict()
    
    print(f"Serialized data contains:")
    print(f"  - {len(serialized['clusters'])} clusters")
    print(f"  - {len(serialized['hierarchy_map'])} hierarchy relationships")
    print(f"  - Configuration with {len(serialized['config']['levels'])} levels")
    
    # Deserialize hierarchy
    print("\nDeserializing hierarchy...")
    restored_hierarchy = ClusterHierarchy.from_dict(serialized)
    
    # Verify restoration
    original_count = len(hierarchy)
    restored_count = len(restored_hierarchy)
    print(f"Original clusters: {original_count}")
    print(f"Restored clusters: {restored_count}")
    print(f"Serialization successful: {original_count == restored_count}")


def main():
    """Run all demonstrations."""
    print("ClusterHierarchy Multi-level Clustering Structure Management Demo")
    print("================================================================")
    
    try:
        demonstrate_hierarchy_basics()
        demonstrate_navigation()
        demonstrate_cluster_operations()
        demonstrate_search_and_metrics()
        demonstrate_optimization()
        demonstrate_rebalancing()
        demonstrate_validation()
        demonstrate_serialization()
        
        print("\n=== Demo Complete ===")
        print("All ClusterHierarchy features demonstrated successfully!")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()