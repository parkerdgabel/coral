#!/usr/bin/env python3
"""
Demonstration of Coral's clustering migration feature.

This example shows how to migrate an existing repository to use clustering,
which can significantly reduce storage requirements for similar weights.
"""

import numpy as np
from pathlib import Path
import tempfile
import time

from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.version_control.repository import Repository


def create_sample_repository(repo_path: Path) -> Repository:
    """Create a sample repository with multiple model checkpoints."""
    print("Creating sample repository with model checkpoints...")
    
    repo = Repository(repo_path, init=True)
    
    # Simulate a model with multiple layers
    layer_shapes = [
        (784, 256),   # Input layer
        (256, 128),   # Hidden layer 1
        (128, 64),    # Hidden layer 2
        (64, 10),     # Output layer
    ]
    
    # Create initial model
    initial_weights = {}
    for i, shape in enumerate(layer_shapes):
        data = np.random.randn(*shape).astype(np.float32)
        weight = WeightTensor(
            data=data,
            metadata=WeightMetadata(
                name=f"layer_{i}_weight",
                shape=shape,
                dtype=np.float32,
                layer_type="dense",
                model_name="demo_model"
            )
        )
        initial_weights[f"layer_{i}_weight"] = weight
    
    repo.stage_weights(initial_weights)
    repo.commit("Initial model", author="Demo User", email="demo@example.com")
    print(f"  ✓ Created initial model with {len(initial_weights)} weights")
    
    # Simulate training checkpoints (similar weights with small changes)
    num_checkpoints = 5
    for checkpoint in range(num_checkpoints):
        checkpoint_weights = {}
        
        for name, original_weight in initial_weights.items():
            # Simulate small updates during training
            noise = np.random.randn(*original_weight.shape).astype(np.float32) * 0.01
            updated_data = original_weight.data + noise
            
            updated_weight = WeightTensor(
                data=updated_data,
                metadata=WeightMetadata(
                    name=name,
                    shape=original_weight.shape,
                    dtype=original_weight.dtype,
                    layer_type=original_weight.metadata.layer_type,
                    model_name=original_weight.metadata.model_name
                )
            )
            checkpoint_weights[name] = updated_weight
        
        repo.stage_weights(checkpoint_weights)
        repo.commit(
            f"Training checkpoint {checkpoint + 1}",
            author="Demo User",
            email="demo@example.com"
        )
        print(f"  ✓ Created checkpoint {checkpoint + 1}")
    
    # Create a fine-tuned variant
    finetuned_weights = {}
    for name, original_weight in initial_weights.items():
        # Larger changes for fine-tuning
        noise = np.random.randn(*original_weight.shape).astype(np.float32) * 0.05
        finetuned_data = original_weight.data + noise
        
        finetuned_weight = WeightTensor(
            data=finetuned_data,
            metadata=WeightMetadata(
                name=name,
                shape=original_weight.shape,
                dtype=original_weight.dtype,
                layer_type=original_weight.metadata.layer_type,
                model_name="demo_model_finetuned"
            )
        )
        finetuned_weights[name] = finetuned_weight
    
    repo.stage_weights(finetuned_weights)
    repo.commit("Fine-tuned model variant", author="Demo User", email="demo@example.com")
    print("  ✓ Created fine-tuned variant")
    
    return repo


def analyze_repository_size(repo: Repository) -> dict:
    """Analyze repository storage usage."""
    weights_path = repo.weights_store_path
    clusters_path = repo.clusters_store_path
    
    weights_size = weights_path.stat().st_size if weights_path.exists() else 0
    clusters_size = clusters_path.stat().st_size if clusters_path.exists() else 0
    
    # Count total weights
    total_weights = 0
    for commit in repo.version_graph.commits.values():
        total_weights += len(commit.weight_hashes)
    
    return {
        'weights_size_mb': weights_size / (1024 * 1024),
        'clusters_size_mb': clusters_size / (1024 * 1024),
        'total_size_mb': (weights_size + clusters_size) / (1024 * 1024),
        'total_weights': total_weights
    }


def main():
    """Demonstrate clustering migration."""
    print("Coral Clustering Migration Demo")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "demo_repo"
        
        # Create sample repository
        repo = create_sample_repository(repo_path)
        
        # Analyze before migration
        print("\nRepository analysis before migration:")
        before_stats = analyze_repository_size(repo)
        print(f"  Storage size: {before_stats['total_size_mb']:.2f} MB")
        print(f"  Total weights: {before_stats['total_weights']}")
        
        # Analyze clustering potential
        print("\nAnalyzing clustering opportunities...")
        analysis = repo.analyze_clustering(similarity_threshold=0.95)
        print(f"  Potential clusters: {analysis['potential_clusters']}")
        print(f"  Estimated reduction: {analysis['estimated_reduction']:.1%}")
        
        # Perform migration
        print("\nMigrating repository to use clustering...")
        print("  Strategy: adaptive")
        print("  Threshold: 0.95")
        
        start_time = time.time()
        
        # Define progress callback
        def show_progress(current, total):
            percent = (current / total) * 100 if total > 0 else 0
            print(f"\r  Progress: {current}/{total} ({percent:.1f}%)", end="", flush=True)
        
        result = repo.migrate_to_clustering(
            strategy="adaptive",
            threshold=0.95,
            batch_size=10,
            progress_callback=show_progress
        )
        print()  # New line after progress
        
        elapsed = time.time() - start_time
        
        print(f"\nMigration completed in {elapsed:.2f} seconds:")
        print(f"  Weights processed: {result['weights_processed']}")
        print(f"  Clusters created: {result['clusters_created']}")
        print(f"  Space saved: {result['space_saved'] / (1024 * 1024):.2f} MB")
        print(f"  Reduction: {result['reduction_percentage']:.1%}")
        
        if result.get('warnings'):
            print("\nWarnings:")
            for warning in result['warnings']:
                print(f"  - {warning}")
        
        # Analyze after migration
        print("\nRepository analysis after migration:")
        after_stats = analyze_repository_size(repo)
        print(f"  Storage size: {after_stats['total_size_mb']:.2f} MB")
        print(f"  Weights storage: {after_stats['weights_size_mb']:.2f} MB")
        print(f"  Clusters storage: {after_stats['clusters_size_mb']:.2f} MB")
        
        # Verify weights can still be loaded
        print("\nVerifying weight reconstruction...")
        sample_weights = repo.get_all_weights()
        print(f"  ✓ Successfully loaded {len(sample_weights)} weights")
        
        # Test mixed mode - add new weights after migration
        print("\nTesting mixed mode operation...")
        new_weight = WeightTensor(
            data=np.random.randn(32, 32).astype(np.float32),
            metadata=WeightMetadata(
                name="new_layer_weight",
                shape=(32, 32),
                dtype=np.float32,
                layer_type="dense",
                model_name="new_model"
            )
        )
        
        repo.stage_weights({"new_layer_weight": new_weight})
        repo.commit("Added new weight after migration")
        print("  ✓ Successfully added new weight in mixed mode")
        
        # Show clustering status
        print("\nClustering status:")
        status = repo.get_clustering_status()
        print(f"  Enabled: {status['enabled']}")
        print(f"  Strategy: {status.get('strategy', 'unknown')}")
        print(f"  Clusters: {status.get('num_clusters', 0)}")
        print(f"  Clustered weights: {status.get('clustered_weights', 0)}")
        
        # Show overall improvement
        print("\nOverall Results:")
        print(f"  Original size: {before_stats['total_size_mb']:.2f} MB")
        print(f"  Final size: {after_stats['total_size_mb']:.2f} MB")
        
        actual_reduction = 1.0 - (after_stats['total_size_mb'] / before_stats['total_size_mb'])
        print(f"  Actual reduction: {actual_reduction:.1%}")
        print(f"  Compression ratio: {before_stats['total_size_mb'] / after_stats['total_size_mb']:.2f}x")


if __name__ == "__main__":
    main()