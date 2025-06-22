"""
Demo of the ClusterIndex component for Coral ML clustering system.

This example shows how to:
1. Create a cluster index
2. Add weights to clusters
3. Find nearest clusters
4. Rebalance clusters for efficiency
5. Serialize and deserialize the index
"""

import numpy as np
from coral.clustering.cluster_index import ClusterIndex


def main():
    print("=== Coral ClusterIndex Demo ===\n")
    
    # Create a cluster index
    index = ClusterIndex(dimension_threshold=100)
    print(f"Created ClusterIndex with dimension threshold: 100")
    
    # Create some sample weight vectors
    print("\n1. Adding weights to clusters...")
    
    # Cluster 1: Weights around [1, 0, 0]
    index.add_weight_to_cluster("weight_1", "cluster_red", np.array([1.0, 0.1, 0.0]))
    index.add_weight_to_cluster("weight_2", "cluster_red", np.array([0.9, 0.0, 0.1]))
    index.add_weight_to_cluster("weight_3", "cluster_red", np.array([1.1, 0.0, 0.0]))
    
    # Cluster 2: Weights around [0, 1, 0]
    index.add_weight_to_cluster("weight_4", "cluster_green", np.array([0.0, 1.0, 0.1]))
    index.add_weight_to_cluster("weight_5", "cluster_green", np.array([0.1, 0.9, 0.0]))
    
    # Cluster 3: Weights around [0, 0, 1]
    index.add_weight_to_cluster("weight_6", "cluster_blue", np.array([0.0, 0.1, 1.0]))
    index.add_weight_to_cluster("weight_7", "cluster_blue", np.array([0.1, 0.0, 0.9]))
    
    print(f"Total weights: {index.get_weight_count()}")
    print(f"Total clusters: {index.get_cluster_count()}")
    print(f"Cluster sizes: {index.get_cluster_sizes()}")
    
    # Find nearest cluster for a new weight
    print("\n2. Finding nearest clusters...")
    
    query_weight = np.array([0.8, 0.2, 0.1])  # Should be closest to red cluster
    nearest = index.find_nearest_cluster(query_weight, k=2)
    
    print(f"Query weight: {query_weight}")
    print("Nearest clusters:")
    for cluster_id, distance in nearest:
        print(f"  - {cluster_id}: distance = {distance:.4f}")
    
    # Get members of a cluster
    print("\n3. Getting cluster members...")
    red_members = index.get_cluster_members("cluster_red")
    print(f"Members of cluster_red: {red_members}")
    
    # Add more weights to create imbalance
    print("\n4. Creating imbalanced clusters...")
    for i in range(5):
        index.add_weight_to_cluster(f"extra_weight_{i}", "cluster_red", 
                                  np.array([1.0 + i*0.01, 0.0, 0.0]))
    
    print(f"Updated cluster sizes: {index.get_cluster_sizes()}")
    
    # Rebalance clusters
    print("\n5. Rebalancing clusters...")
    
    # Try size-based rebalancing
    size_reassignments = index.rebalance_clusters(strategy="size")
    if size_reassignments:
        print("Size-based reassignments:")
        for cluster_id, weights in size_reassignments.items():
            print(f"  - {cluster_id}: {weights}")
    else:
        print("No size-based reassignments needed")
    
    print(f"Cluster sizes after rebalancing: {index.get_cluster_sizes()}")
    
    # Test with high-dimensional data
    print("\n6. Testing with high-dimensional data...")
    
    # Create a new index for high-dimensional data
    hd_index = ClusterIndex(dimension_threshold=100)
    
    # Add some 200-dimensional weights (will use LSH)
    dim = 200
    hd_index.add_weight_to_cluster("hd_weight_1", "hd_cluster_1", np.random.randn(dim))
    hd_index.add_weight_to_cluster("hd_weight_2", "hd_cluster_1", np.random.randn(dim))
    hd_index.add_weight_to_cluster("hd_weight_3", "hd_cluster_2", np.random.randn(dim))
    
    # Query with high-dimensional weight
    hd_query = np.random.randn(dim)
    hd_nearest = hd_index.find_nearest_cluster(hd_query, k=1)
    
    print(f"High-dimensional query result: {hd_nearest[0][0]} (distance: {hd_nearest[0][1]:.4f})")
    
    # Serialization
    print("\n7. Serialization and deserialization...")
    
    # Serialize the index
    serialized = index.to_dict()
    print(f"Serialized index has keys: {list(serialized.keys())}")
    
    # Create new index from serialized data
    restored_index = ClusterIndex.from_dict(serialized)
    print(f"Restored index has {restored_index.get_weight_count()} weights")
    print(f"Restored cluster sizes: {restored_index.get_cluster_sizes()}")
    
    # Verify restored index works
    restored_nearest = restored_index.find_nearest_cluster(query_weight, k=1)
    print(f"Query on restored index: {restored_nearest[0][0]} (distance: {restored_nearest[0][1]:.4f})")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()