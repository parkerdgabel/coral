#!/usr/bin/env python3
"""
Garbage Collection Demo for Coral

This example demonstrates how Coral's enhanced garbage collection system
handles unreferenced weights, deltas, clusters, and centroids.
"""

import numpy as np
import tempfile
import shutil
from pathlib import Path

from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.version_control.repository import Repository
from coral.clustering import ClusteringConfig, ClusteringStrategy


def create_sample_weights(prefix: str, count: int, base_data: np.ndarray = None, noise: float = 0.1):
    """Create sample weights with optional base data and noise."""
    weights = {}
    
    if base_data is None:
        base_data = np.random.randn(10, 10)
    
    for i in range(count):
        # Add noise to create similar but not identical weights
        data = base_data + np.random.randn(*base_data.shape) * noise
        
        weight = WeightTensor(
            data=data,
            metadata=WeightMetadata(
                name=f"{prefix}_weight_{i}",
                shape=data.shape,
                dtype=np.float32,
                model_name=f"{prefix}_model",
                layer_type="dense"
            )
        )
        weights[weight.metadata.name] = weight
    
    return weights


def print_storage_stats(repo: Repository, title: str):
    """Print storage statistics."""
    print(f"\n{title}")
    print("-" * 50)
    
    # Get storage info
    storage_info = repo.get_storage_info()
    
    # Get counts from HDF5 store
    from coral.storage.hdf5_store import HDF5Store
    with HDF5Store(repo.weights_store_path) as store:
        weight_count = len(store.list_weights())
        delta_count = len(store.list_deltas())
    
    # Get cluster counts if available
    cluster_count = 0
    centroid_count = 0
    if repo.cluster_storage:
        with repo.cluster_storage:
            clusters = repo.cluster_storage.load_clusters()
            cluster_count = len(clusters)
            centroids = repo.cluster_storage.load_centroids()
            centroid_count = len(centroids)
    
    print(f"Storage size: {storage_info['file_size'] / 1024:.2f} KB")
    print(f"Weights: {weight_count}")
    print(f"Deltas: {delta_count}")
    print(f"Clusters: {cluster_count}")
    print(f"Centroids: {centroid_count}")


def main():
    """Run the garbage collection demo."""
    print("=== Coral Garbage Collection Demo ===\n")
    
    # Create temporary repository
    temp_dir = tempfile.mkdtemp()
    repo_path = Path(temp_dir) / "gc_demo_repo"
    
    try:
        # Initialize repository with clustering enabled
        print("1. Creating repository with clustering enabled...")
        repo = Repository(repo_path, init=True)
        
        # Enable clustering
        repo.enable_clustering()
        
        print_storage_stats(repo, "Initial State")
        
        # Create and commit first set of weights
        print("\n2. Creating first set of similar weights...")
        weights_v1 = create_sample_weights("v1", count=6, noise=0.05)
        
        repo.stage_weights(weights_v1)
        commit1 = repo.commit("Initial model weights")
        print(f"Committed {len(weights_v1)} weights")
        
        print_storage_stats(repo, "After First Commit")
        
        # Perform clustering
        print("\n3. Clustering the weights...")
        cluster_result = repo.cluster_repository()
        print(f"Created {cluster_result.num_clusters} clusters")
        print(f"Clustered {cluster_result.total_weights_clustered} weights")
        print(f"Compression ratio: {cluster_result.compression_ratio:.2f}x")
        
        print_storage_stats(repo, "After Clustering")
        
        # Create updated weights (simulating model fine-tuning)
        print("\n4. Creating updated weights (simulating fine-tuning)...")
        # Keep some weights, update others
        weights_v2 = {}
        
        # Keep first 3 weights
        for i in range(3):
            name = f"v1_weight_{i}"
            weights_v2[name] = weights_v1[name]
        
        # Add new weights (slightly different)
        new_weights = create_sample_weights("v2", count=4, base_data=weights_v1["v1_weight_0"].data, noise=0.1)
        weights_v2.update(new_weights)
        
        repo.stage_weights(weights_v2)
        commit2 = repo.commit("Fine-tuned model weights")
        print(f"Committed {len(weights_v2)} weights (3 kept, 4 new)")
        
        print_storage_stats(repo, "After Second Commit")
        
        # Create a branch with different weights
        print("\n5. Creating branch with different weights...")
        repo.create_branch("experiment")
        repo.checkout("experiment")
        
        # Completely different weights
        weights_exp = create_sample_weights("exp", count=5, noise=0.2)
        repo.stage_weights(weights_exp)
        commit3 = repo.commit("Experimental weights")
        
        print_storage_stats(repo, "After Branch Commit")
        
        # Switch back to main and run garbage collection
        print("\n6. Running garbage collection...")
        repo.checkout("main")
        
        print("\nBefore GC:")
        print_storage_stats(repo, "Storage Before GC")
        
        # Run GC
        gc_result = repo.gc(include_clusters=True)
        
        print("\nGC Results:")
        print(f"Cleaned weights: {gc_result['cleaned_weights']}")
        print(f"Cleaned deltas: {gc_result['cleaned_deltas']}")
        print(f"Cleaned clusters: {gc_result['cleaned_clusters']}")
        print(f"Cleaned centroids: {gc_result['cleaned_centroids']}")
        print(f"Protected centroids: {gc_result['protected_centroids']}")
        print(f"GC time: {gc_result['gc_time']:.3f} seconds")
        
        print_storage_stats(repo, "After GC")
        
        # Verify weights can still be loaded
        print("\n7. Verifying weight integrity...")
        
        # Check main branch weights
        main_weights = repo.get_all_weights()
        print(f"Main branch weights: {len(main_weights)}")
        
        # Check experimental branch weights
        repo.checkout("experiment")
        exp_weights = repo.get_all_weights()
        print(f"Experiment branch weights: {len(exp_weights)}")
        
        # Demonstrate protected centroids
        print("\n8. Demonstrating centroid protection...")
        repo.checkout("main")
        
        # Create weights that will share centroids with existing clusters
        similar_weights = create_sample_weights("similar", count=3, 
                                              base_data=weights_v1["v1_weight_0"].data, 
                                              noise=0.02)
        repo.stage_weights(similar_weights)
        commit4 = repo.commit("Similar weights using existing centroids")
        
        # Delete the similar weights but keep the originals
        repo.stage_weights(weights_v2)  # Revert to v2 weights
        commit5 = repo.commit("Revert to v2 weights")
        
        # Run GC again
        print("\nRunning GC after creating weights that share centroids...")
        gc_result2 = repo.gc(include_clusters=True)
        
        print(f"Protected centroids: {gc_result2['protected_centroids']}")
        print("(Centroids were protected because they're still referenced by delta encodings)")
        
        # Final statistics
        print("\n9. Final Repository Statistics:")
        stats = repo.get_cluster_statistics()
        print(f"Total clusters: {stats.total_clusters}")
        print(f"Clustered weights: {stats.total_weights}")
        print(f"Average cluster size: {stats.avg_cluster_size:.2f}")
        print(f"Clustering coverage: {stats.clustering_coverage:.1%}")
        
        # Show space savings
        print("\n10. Space Savings Summary:")
        initial_size = len(weights_v1) * weights_v1["v1_weight_0"].data.nbytes
        
        storage_info = repo.get_storage_info()
        final_size = storage_info['file_size']
        
        print(f"Naive storage (no dedup): {initial_size / 1024:.2f} KB")
        print(f"Actual storage (with GC): {final_size / 1024:.2f} KB")
        print(f"Space saved: {(1 - final_size/initial_size) * 100:.1f}%")
        
    finally:
        # Cleanup
        print("\n11. Cleaning up temporary repository...")
        shutil.rmtree(temp_dir)
        print("Demo completed!")


if __name__ == "__main__":
    main()