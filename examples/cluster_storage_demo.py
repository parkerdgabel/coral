#!/usr/bin/env python3
"""
ClusterStorage Demo - Comprehensive example of cluster persistence and management.

This example demonstrates the key features of the ClusterStorage component:
- Storing and retrieving clusters, centroids, and hierarchies
- Efficient query operations and filtering
- Storage optimization and backup/restore
- Integration with existing HDF5Store infrastructure
"""

import numpy as np
import tempfile
from pathlib import Path

# Import Coral clustering components
from coral.clustering import (
    ClusterStorage, ClusterInfo, ClusteringStrategy, ClusterLevel,
    Centroid, ClusterAssignment, ClusterHierarchy, HierarchyConfig
)
from coral.core.weight_tensor import WeightTensor, WeightMetadata


def create_sample_data():
    """Create sample clustering data for demonstration."""
    print("Creating sample clustering data...")
    
    # Create sample clusters
    clusters = []
    for i in range(10):
        cluster = ClusterInfo(
            cluster_id=f"cluster_{i}",
            strategy=ClusteringStrategy.KMEANS if i % 2 == 0 else ClusteringStrategy.HIERARCHICAL,
            level=ClusterLevel.TENSOR if i < 5 else ClusterLevel.LAYER,
            member_count=20 + i * 5,
            centroid_hash=f"centroid_hash_{i}",
            created_at="2025-06-20T12:00:00Z",
            parent_cluster_id=f"parent_{i//3}" if i > 2 else None
        )
        clusters.append(cluster)
    
    # Create sample centroids
    centroids = []
    for i in range(10):
        # Different shapes to test variety
        if i % 3 == 0:
            shape = (64, 32)
        elif i % 3 == 1:
            shape = (128, 64)
        else:
            shape = (32, 16)
        
        data = np.random.randn(*shape).astype(np.float32)
        centroid = Centroid(
            data=data,
            cluster_id=f"cluster_{i}",
            shape=shape,
            dtype=np.float32
        )
        centroids.append(centroid)
    
    # Create sample assignments
    assignments = []
    for i in range(25):
        assignment = ClusterAssignment(
            weight_name=f"weight_{i}",
            weight_hash=f"hash_{i}",
            cluster_id=f"cluster_{i % 10}",
            distance_to_centroid=float(i * 0.1),
            similarity_score=0.95 - (i * 0.02),
            is_representative=(i % 5 == 0)
        )
        assignments.append(assignment)
    
    # Create sample hierarchy
    config = HierarchyConfig(
        levels=[ClusterLevel.TENSOR, ClusterLevel.LAYER, ClusterLevel.MODEL],
        merge_threshold=0.8,
        split_threshold=0.3
    )
    hierarchy = ClusterHierarchy(config)
    hierarchy.build_hierarchy(clusters[:5], config)  # Use subset for demo
    
    return clusters, centroids, assignments, hierarchy


def demonstrate_basic_operations(storage, clusters, centroids, assignments, hierarchy):
    """Demonstrate basic storage and retrieval operations."""
    print("\n=== Basic Storage Operations ===")
    
    # Store clusters
    print("Storing clusters...")
    cluster_result = storage.store_clusters(clusters)
    print(f"Stored {len(cluster_result)} clusters")
    
    # Store centroids
    print("Storing centroids...")
    centroid_result = storage.store_centroids(centroids)
    print(f"Stored {len(centroid_result)} centroids")
    
    # Store assignments
    print("Storing assignments...")
    assignment_result = storage.store_assignments(assignments)
    print(f"Stored {len(assignment_result)} assignments")
    
    # Store hierarchy
    print("Storing hierarchy...")
    hierarchy_key = storage.store_hierarchy(hierarchy)
    print(f"Stored hierarchy with key: {hierarchy_key}")
    
    # Load and verify data
    print("\nLoading and verifying data...")
    loaded_clusters = storage.load_clusters()
    loaded_centroids = storage.load_centroids()
    loaded_assignments = storage.load_assignments()
    loaded_hierarchy = storage.load_hierarchy()
    
    print(f"Loaded {len(loaded_clusters)} clusters")
    print(f"Loaded {len(loaded_centroids)} centroids")
    print(f"Loaded {len(loaded_assignments)} assignments")
    print(f"Loaded hierarchy with {len(loaded_hierarchy)} clusters")


def demonstrate_query_operations(storage):
    """Demonstrate efficient query and filtering operations."""
    print("\n=== Query Operations ===")
    
    # Query by level
    tensor_clusters = storage.get_clusters_by_level(ClusterLevel.TENSOR)
    layer_clusters = storage.get_clusters_by_level(ClusterLevel.LAYER)
    print(f"Found {len(tensor_clusters)} tensor-level clusters")
    print(f"Found {len(layer_clusters)} layer-level clusters")
    
    # Query by strategy
    kmeans_clusters = storage.get_clusters_by_strategy(ClusteringStrategy.KMEANS)
    hierarchical_clusters = storage.get_clusters_by_strategy(ClusteringStrategy.HIERARCHICAL)
    print(f"Found {len(kmeans_clusters)} K-means clusters")
    print(f"Found {len(hierarchical_clusters)} hierarchical clusters")
    
    # Get cluster metadata without loading full data
    print("\nCluster metadata examples:")
    for i in range(3):
        metadata = storage.get_cluster_metadata(f"cluster_{i}")
        if metadata:
            print(f"  Cluster {i}: {metadata['member_count']} members, strategy: {metadata['strategy']}")
    
    # Batch load specific clusters
    cluster_ids = ["cluster_0", "cluster_2", "cluster_4"]
    batch_clusters = storage.batch_load_clusters(cluster_ids)
    print(f"\nBatch loaded {len(batch_clusters)} specific clusters")
    
    # Load centroids for specific clusters
    specific_centroids = storage.load_centroids(cluster_ids=cluster_ids[:2])
    print(f"Loaded {len(specific_centroids)} centroids for specific clusters")
    
    # Filter assignments
    filtered_assignments = storage.load_assignments(
        filter_criteria={"min_similarity": 0.9, "cluster_ids": ["cluster_0", "cluster_1"]}
    )
    print(f"Found {len(filtered_assignments)} high-similarity assignments")


def demonstrate_storage_optimization(storage):
    """Demonstrate storage optimization and management."""
    print("\n=== Storage Optimization ===")
    
    # Get initial storage info
    initial_info = storage.get_storage_info()
    print(f"Initial storage size: {initial_info['file_size']} bytes")
    print(f"Group counts: {initial_info['group_counts']}")
    
    # Estimate storage requirements
    size_estimates = storage.estimate_storage_size()
    print(f"\nStorage size breakdown:")
    for component, size in size_estimates.items():
        print(f"  {component}: {size} bytes")
    
    # Run garbage collection
    print("\nRunning garbage collection...")
    gc_result = storage.garbage_collect()
    print(f"GC completed in {gc_result['gc_time']:.3f}s")
    print(f"Removed {gc_result['orphaned_centroids_removed']} orphaned centroids")
    print(f"Removed {gc_result['orphaned_assignments_removed']} orphaned assignments")
    
    # Compress storage
    print("\nCompressing storage...")
    compression_result = storage.compress_storage()
    print(f"Compression completed in {compression_result['compression_time']:.3f}s")
    print(f"Space saved: {compression_result['space_saved']} bytes")
    
    # Optimize layout
    print("\nOptimizing storage layout...")
    layout_result = storage.optimize_layout()
    print(f"Layout optimization completed in {layout_result['optimization_time']:.3f}s")


def demonstrate_backup_restore(storage, temp_dir):
    """Demonstrate backup and restore capabilities."""
    print("\n=== Backup and Restore ===")
    
    # Create backup
    print("Creating backup...")
    backup_path = storage.create_backup(str(temp_dir / "backups"))
    print(f"Backup created at: {backup_path}")
    
    # Add some new data to show the difference
    new_cluster = ClusterInfo(
        cluster_id="backup_test_cluster",
        strategy=ClusteringStrategy.DBSCAN,
        level=ClusterLevel.MODEL,
        member_count=100
    )
    storage.store_clusters([new_cluster])
    print("Added new cluster after backup")
    
    # Verify new data exists
    all_clusters = storage.load_clusters()
    cluster_ids = [c.cluster_id for c in all_clusters]
    print(f"Total clusters now: {len(all_clusters)}")
    print(f"New cluster exists: {'backup_test_cluster' in cluster_ids}")
    
    # Restore from backup
    print("\nRestoring from backup...")
    restore_success = storage.restore_from_backup(backup_path)
    print(f"Restore successful: {restore_success}")
    
    # Verify restoration
    restored_clusters = storage.load_clusters()
    restored_ids = [c.cluster_id for c in restored_clusters]
    print(f"Clusters after restore: {len(restored_clusters)}")
    print(f"New cluster removed: {'backup_test_cluster' not in restored_ids}")


def demonstrate_export_import(storage, temp_dir):
    """Demonstrate data export and import."""
    print("\n=== Export and Import ===")
    
    # Export all data
    export_file = temp_dir / "cluster_export.json"
    print(f"Exporting data to {export_file}...")
    export_result = storage.export_clusters(str(export_file))
    
    print(f"Export completed in {export_result['export_time']:.3f}s")
    print(f"Exported {export_result['clusters_exported']} clusters")
    print(f"Exported {export_result['centroids_exported']} centroids")
    print(f"Exported {export_result['assignments_exported']} assignments")
    print(f"Export file size: {export_result['file_size']} bytes")
    
    # Create new storage and import
    import_storage_path = temp_dir / "imported_storage.h5"
    import_storage = ClusterStorage(str(import_storage_path))
    
    try:
        print(f"\nImporting data to new storage...")
        import_result = import_storage.import_clusters(str(export_file))
        
        print(f"Import completed in {import_result['import_time']:.3f}s")
        print(f"Imported {import_result['clusters_imported']} clusters")
        print(f"Imported {import_result['centroids_imported']} centroids")
        print(f"Imported {import_result['assignments_imported']} assignments")
        
        # Verify imported data
        imported_clusters = import_storage.load_clusters()
        print(f"Verified: {len(imported_clusters)} clusters in new storage")
        
    finally:
        import_storage.close()


def demonstrate_integration_with_hdf5store(temp_dir):
    """Demonstrate integration with existing HDF5Store."""
    print("\n=== HDF5Store Integration ===")
    
    from coral.storage.hdf5_store import HDF5Store
    
    # Create HDF5Store and add some weight data
    storage_path = temp_dir / "integrated_storage.h5"
    hdf5_store = HDF5Store(str(storage_path))
    
    # Store a weight tensor
    weight_data = np.random.randn(64, 128).astype(np.float32)
    weight_metadata = WeightMetadata(
        name="demo_weight",
        shape=weight_data.shape,
        dtype=weight_data.dtype,
        layer_type="linear",
        model_name="demo_model"
    )
    weight = WeightTensor(data=weight_data, metadata=weight_metadata)
    weight_hash = hdf5_store.store(weight)
    print(f"Stored weight tensor with hash: {weight_hash}")
    
    # Create cluster storage using the same HDF5 file
    cluster_storage = ClusterStorage(str(storage_path), base_store=hdf5_store)
    
    # Store cluster data
    cluster = ClusterInfo(
        cluster_id="integrated_cluster",
        strategy=ClusteringStrategy.ADAPTIVE,
        level=ClusterLevel.TENSOR,
        member_count=1,
        centroid_hash=weight_hash  # Reference the stored weight
    )
    cluster_storage.store_clusters([cluster])
    print("Stored cluster data in same file")
    
    # Verify both types of data exist
    loaded_weight = hdf5_store.load(weight_hash)
    loaded_clusters = cluster_storage.load_clusters()
    
    print(f"Weight tensor loaded: {loaded_weight is not None}")
    print(f"Clusters loaded: {len(loaded_clusters)}")
    print(f"Cluster references weight: {loaded_clusters[0].centroid_hash == weight_hash}")
    
    # Get comprehensive storage info
    storage_info = cluster_storage.get_storage_info()
    print(f"Total file size: {storage_info['file_size']} bytes")
    print(f"Contains both weights and clusters: {storage_info['group_counts']}")
    
    cluster_storage.close()
    hdf5_store.close()


def main():
    """Main demonstration function."""
    print("Coral ClusterStorage Demo")
    print("=" * 50)
    
    # Create temporary directory for demo files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        storage_file = temp_path / "cluster_storage_demo.h5"
        
        # Create sample data
        clusters, centroids, assignments, hierarchy = create_sample_data()
        
        # Create storage instance
        with ClusterStorage(str(storage_file)) as storage:
            # Demonstrate all major features
            demonstrate_basic_operations(storage, clusters, centroids, assignments, hierarchy)
            demonstrate_query_operations(storage)
            demonstrate_storage_optimization(storage)
            demonstrate_backup_restore(storage, temp_path)
            demonstrate_export_import(storage, temp_path)
        
        # Demonstrate integration
        demonstrate_integration_with_hdf5store(temp_path)
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("ClusterStorage provides comprehensive cluster persistence with:")
    print("  ✓ Efficient HDF5-based storage with compression")
    print("  ✓ Fast query operations and filtering")
    print("  ✓ Storage optimization and garbage collection")
    print("  ✓ Backup and restore capabilities")
    print("  ✓ Data export/import for migration")
    print("  ✓ Seamless integration with HDF5Store")
    print("  ✓ Thread-safe operations for concurrent access")


if __name__ == "__main__":
    main()