#!/usr/bin/env python3
"""
ClusterIndex Demonstration Script

This script demonstrates the key features of the ClusterIndex component:
- Fast centroid storage and retrieval
- Spatial indexing for efficient nearest neighbor queries
- Hierarchical cluster navigation
- Performance monitoring and statistics
- Thread-safe concurrent operations
"""

import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from coral.clustering.cluster_index import ClusterIndex
from coral.clustering.cluster_types import Centroid, ClusterInfo, ClusterLevel, ClusteringStrategy
from coral.core.weight_tensor import WeightTensor, WeightMetadata


def create_sample_centroids(num_centroids: int = 100, tensor_size: int = 50) -> list:
    """Create sample centroids for demonstration."""
    centroids = []
    cluster_infos = []
    
    levels = list(ClusterLevel)
    
    for i in range(num_centroids):
        # Create centroid with some structure (not completely random)
        base_pattern = np.sin(np.linspace(0, 2 * np.pi, tensor_size)) * (i + 1) * 0.1
        noise = np.random.randn(tensor_size) * 0.05
        data = base_pattern + noise
        
        centroid = Centroid(
            data=data,
            cluster_id=f"demo_cluster_{i}",
            shape=(tensor_size,),
            dtype=data.dtype
        )
        
        cluster_info = ClusterInfo(
            cluster_id=centroid.cluster_id,
            strategy=ClusteringStrategy.KMEANS,
            level=levels[i % len(levels)],
            member_count=np.random.randint(2, 20),
            centroid_hash=centroid.compute_hash(),
            created_at="2024-01-01T00:00:00Z",
        )
        
        centroids.append(centroid)
        cluster_infos.append(cluster_info)
    
    return centroids, cluster_infos


def demonstrate_basic_operations():
    """Demonstrate basic ClusterIndex operations."""
    print("=" * 60)
    print("BASIC OPERATIONS DEMONSTRATION")
    print("=" * 60)
    
    # Create index
    index = ClusterIndex(spatial_index_type="kdtree", cache_size=50)
    print(f"Created ClusterIndex: {index}")
    
    # Create sample data
    centroids, cluster_infos = create_sample_centroids(20, 30)
    
    # Add centroids
    print(f"\nAdding {len(centroids)} centroids...")
    start_time = time.time()
    
    for centroid, cluster_info in zip(centroids, cluster_infos):
        index.add_centroid(centroid, cluster_info)
    
    add_time = time.time() - start_time
    print(f"Added centroids in {add_time:.3f}s")
    print(f"Index now contains {len(index)} centroids")
    
    # Test retrieval
    print("\nTesting centroid retrieval...")
    test_id = centroids[0].cluster_id
    retrieved = index.get_centroid(test_id)
    
    if retrieved:
        print(f"✓ Successfully retrieved centroid '{test_id}'")
        print(f"  Shape: {retrieved.shape}, Hash: {retrieved.compute_hash()[:16]}...")
    else:
        print("✗ Failed to retrieve centroid")
    
    # Test nearest neighbor search
    print("\nTesting nearest neighbor search...")
    query_data = centroids[5].data + np.random.randn(*centroids[5].shape) * 0.01
    
    start_time = time.time()
    result = index.find_nearest_centroid(query_data)
    query_time = time.time() - start_time
    
    if result:
        cluster_id, distance = result
        print(f"✓ Found nearest centroid: '{cluster_id}' (distance: {distance:.4f})")
        print(f"  Query completed in {query_time * 1000:.3f}ms")
    else:
        print("✗ No nearest centroid found")


def demonstrate_similarity_search():
    """Demonstrate similarity-based search capabilities."""
    print("\n" + "=" * 60)
    print("SIMILARITY SEARCH DEMONSTRATION")
    print("=" * 60)
    
    index = ClusterIndex()
    centroids, cluster_infos = create_sample_centroids(50, 40)
    
    # Add centroids
    for centroid, cluster_info in zip(centroids, cluster_infos):
        index.add_centroid(centroid, cluster_info)
    
    # Create query similar to some centroids
    base_centroid = centroids[10]
    query_data = base_centroid.data + np.random.randn(*base_centroid.shape) * 0.02
    
    print(f"Searching for centroids similar to '{base_centroid.cluster_id}'...")
    
    # Test different thresholds
    thresholds = [0.5, 1.0, 2.0, 5.0]
    
    for threshold in thresholds:
        start_time = time.time()
        similar = index.find_similar_centroids(query_data, threshold)
        search_time = time.time() - start_time
        
        print(f"  Threshold {threshold:4.1f}: {len(similar):2d} similar centroids "
              f"(search time: {search_time * 1000:.3f}ms)")
        
        if similar:
            closest_id, closest_distance = similar[0]
            print(f"    Closest: '{closest_id}' (distance: {closest_distance:.4f})")


def demonstrate_batch_operations():
    """Demonstrate batch lookup operations."""
    print("\n" + "=" * 60)
    print("BATCH OPERATIONS DEMONSTRATION")
    print("=" * 60)
    
    index = ClusterIndex()
    centroids, cluster_infos = create_sample_centroids(100, 50)
    
    # Add centroids
    for centroid, cluster_info in zip(centroids, cluster_infos):
        index.add_centroid(centroid, cluster_info)
    
    # Create batch of query weights
    batch_sizes = [10, 50, 100]
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch lookup with {batch_size} queries...")
        
        # Create batch queries
        query_batch = []
        for i in range(batch_size):
            # Create queries similar to existing centroids
            base_idx = i % len(centroids)
            query_data = centroids[base_idx].data + np.random.randn(50) * 0.05
            query_batch.append(query_data)
        
        # Individual queries (for comparison)
        start_time = time.time()
        individual_results = []
        for query in query_batch:
            result = index.find_nearest_centroid(query)
            individual_results.append(result)
        individual_time = time.time() - start_time
        
        # Batch query
        start_time = time.time()
        batch_results = index.batch_lookup(query_batch)
        batch_time = time.time() - start_time
        
        # Compare results
        matches = sum(1 for i, b in zip(individual_results, batch_results) if i == b)
        
        print(f"  Individual queries: {individual_time * 1000:.1f}ms")
        print(f"  Batch query:        {batch_time * 1000:.1f}ms")
        print(f"  Speedup:            {individual_time / batch_time:.1f}x")
        print(f"  Result consistency: {matches}/{batch_size} matches")


def demonstrate_hierarchical_features():
    """Demonstrate hierarchical cluster navigation."""
    print("\n" + "=" * 60)
    print("HIERARCHICAL FEATURES DEMONSTRATION")
    print("=" * 60)
    
    index = ClusterIndex()
    centroids, cluster_infos = create_sample_centroids(16, 30)
    
    # Add centroids with cluster info
    for centroid, cluster_info in zip(centroids, cluster_infos):
        index.add_centroid(centroid, cluster_info)
    
    # Define hierarchy (tree structure)
    hierarchy = {
        "demo_cluster_0": {"children": ["demo_cluster_1", "demo_cluster_2"]},
        "demo_cluster_1": {"parent": "demo_cluster_0", "children": ["demo_cluster_3", "demo_cluster_4"]},
        "demo_cluster_2": {"parent": "demo_cluster_0", "children": ["demo_cluster_5", "demo_cluster_6"]},
        "demo_cluster_3": {"parent": "demo_cluster_1"},
        "demo_cluster_4": {"parent": "demo_cluster_1"},
        "demo_cluster_5": {"parent": "demo_cluster_2"},
        "demo_cluster_6": {"parent": "demo_cluster_2"},
    }
    
    print("Building hierarchical structure...")
    index.build_hierarchy(hierarchy)
    
    # Test hierarchy navigation
    print("\nTesting hierarchy navigation:")
    
    # Test parent-child relationships
    test_nodes = ["demo_cluster_0", "demo_cluster_1", "demo_cluster_3"]
    
    for node_id in test_nodes:
        parent = index.get_parent_centroid(node_id)
        children = index.get_child_centroids(node_id)
        
        parent_id = parent.cluster_id if parent else "None"
        child_ids = [c.cluster_id for c in children]
        
        print(f"  {node_id}:")
        print(f"    Parent: {parent_id}")
        print(f"    Children: {child_ids}")
    
    # Test level-based search
    print("\nTesting level-based search:")
    query_data = np.random.randn(30)
    
    for level in ClusterLevel:
        level_results = index.find_level_centroids(level, query_data)
        print(f"  {level.value:8s}: {len(level_results)} centroids")
        
        if level_results:
            closest_id, distance = level_results[0]
            print(f"            Closest: '{closest_id}' (distance: {distance:.4f})")


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring and statistics."""
    print("\n" + "=" * 60)
    print("PERFORMANCE MONITORING DEMONSTRATION")
    print("=" * 60)
    
    index = ClusterIndex(cache_size=20)
    centroids, cluster_infos = create_sample_centroids(200, 60)
    
    # Add centroids
    print("Adding centroids and performing queries...")
    for centroid, cluster_info in zip(centroids, cluster_infos):
        index.add_centroid(centroid, cluster_info)
    
    # Perform various queries to generate statistics
    for _ in range(100):
        query_data = np.random.randn(60)
        index.find_nearest_centroid(query_data)
    
    # Get statistics
    stats = index.get_index_stats()
    usage = index.get_centroid_usage()
    
    print("\nIndex Statistics:")
    print(f"  Total centroids:     {stats['total_centroids']}")
    print(f"  Total queries:       {stats['total_queries']}")
    print(f"  Cache hit rate:      {stats['cache_hit_rate']:.1%}")
    print(f"  Average query time:  {stats['avg_query_time'] * 1000:.3f}ms")
    print(f"  Memory usage:        {stats['memory_usage_mb']:.2f}MB")
    print(f"  Index build time:    {stats['index_build_time']:.3f}s")
    
    print("\nCentroid Usage (top 5):")
    top_usage = sorted(usage.items(), key=lambda x: x[1], reverse=True)[:5]
    for cluster_id, count in top_usage:
        print(f"  {cluster_id}: {count} queries")
    
    # Test index optimization
    print("\nTesting index optimization...")
    start_time = time.time()
    index.optimize_index()
    optimize_time = time.time() - start_time
    
    print(f"Index optimization completed in {optimize_time:.3f}s")
    
    # Validate index
    validation = index.validate_index()
    print(f"\nIndex validation: {'✓ VALID' if validation['is_valid'] else '✗ INVALID'}")
    
    if validation['issues']:
        print("Issues found:")
        for issue in validation['issues']:
            print(f"  - {issue}")


def demonstrate_concurrent_operations():
    """Demonstrate thread-safe concurrent operations."""
    print("\n" + "=" * 60)
    print("CONCURRENT OPERATIONS DEMONSTRATION")
    print("=" * 60)
    
    index = ClusterIndex()
    centroids, cluster_infos = create_sample_centroids(100, 40)
    
    # Add initial centroids
    for centroid, cluster_info in zip(centroids, cluster_infos):
        index.add_centroid(centroid, cluster_info)
    
    print(f"Starting with {len(index)} centroids")
    
    # Define concurrent operations
    def read_worker(worker_id: int):
        """Worker that performs read operations."""
        results = []
        for i in range(20):
            query_data = np.random.randn(40)
            result = index.find_nearest_centroid(query_data)
            if result:
                results.append(result[1])  # Store distance
        return worker_id, len(results), np.mean(results) if results else 0
    
    def stats_worker():
        """Worker that collects statistics."""
        stats_history = []
        for _ in range(10):
            stats = index.get_index_stats()
            stats_history.append(stats['total_queries'])
            time.sleep(0.01)
        return stats_history
    
    print("\nRunning concurrent read operations...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        # Start read workers
        read_futures = [executor.submit(read_worker, i) for i in range(5)]
        
        # Start stats worker
        stats_future = executor.submit(stats_worker)
        
        # Collect results
        read_results = []
        for future in read_futures:
            worker_id, num_results, avg_distance = future.result()
            read_results.append((worker_id, num_results, avg_distance))
        
        stats_history = stats_future.result()
    
    total_time = time.time() - start_time
    
    print(f"Concurrent operations completed in {total_time:.3f}s")
    
    print("\nWorker Results:")
    total_queries = 0
    for worker_id, num_results, avg_distance in read_results:
        print(f"  Worker {worker_id}: {num_results} queries, avg distance: {avg_distance:.4f}")
        total_queries += num_results
    
    print(f"\nTotal queries processed: {total_queries}")
    print(f"Query throughput: {total_queries / total_time:.1f} queries/sec")
    print(f"Stats progression: {stats_history[0]} → {stats_history[-1]} total queries")


def main():
    """Run all demonstrations."""
    print("ClusterIndex Component Demonstration")
    print("=====================================")
    
    # Run demonstrations
    demonstrate_basic_operations()
    demonstrate_similarity_search()
    demonstrate_batch_operations()
    demonstrate_hierarchical_features()
    demonstrate_performance_monitoring()
    demonstrate_concurrent_operations()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey ClusterIndex features demonstrated:")
    print("✓ Fast centroid storage and retrieval (O(1) by ID)")
    print("✓ Efficient nearest neighbor search (O(log n) with spatial indexing)")
    print("✓ Similarity threshold queries with configurable distance")
    print("✓ Batch operations for improved performance")
    print("✓ Hierarchical cluster navigation and level-based search")
    print("✓ Comprehensive performance monitoring and statistics")
    print("✓ Thread-safe concurrent operations")
    print("✓ Memory-efficient indexing with caching")
    print("✓ Index optimization and validation capabilities")


if __name__ == "__main__":
    main()