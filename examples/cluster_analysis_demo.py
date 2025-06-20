#!/usr/bin/env python3
"""
Demonstration of the ClusterAnalyzer for repository-wide weight analysis.

This example shows how to use the ClusterAnalyzer to analyze weight distributions
and perform clustering on neural network weights in a Coral repository.
"""

import numpy as np
import tempfile
import shutil
from pathlib import Path

from coral.clustering import (
    ClusterAnalyzer, ClusteringConfig, ClusteringStrategy, ClusterLevel
)
from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.version_control.repository import Repository


def create_sample_weights():
    """Create sample weight tensors for clustering demonstration."""
    weights = []
    
    # Create some similar weights (e.g., from fine-tuning)
    base_weight = np.random.rand(20, 20).astype(np.float32)
    
    for i in range(5):
        # Add small variations to simulate fine-tuning
        data = base_weight + 0.01 * np.random.rand(20, 20).astype(np.float32)
        metadata = WeightMetadata(
            name=f"layer_{i}_weight",
            shape=(20, 20),
            dtype=np.float32,
            layer_type="linear",
            model_name="transformer"
        )
        weights.append(WeightTensor(data=data, metadata=metadata))
    
    # Create some different weights (e.g., different layers)
    for i in range(3):
        data = np.random.rand(10, 10).astype(np.float32)
        metadata = WeightMetadata(
            name=f"attention_{i}_weight",
            shape=(10, 10),
            dtype=np.float32,
            layer_type="attention",
            model_name="transformer"
        )
        weights.append(WeightTensor(data=data, metadata=metadata))
    
    # Create some convolutional weights
    for i in range(2):
        data = np.random.rand(32, 3, 3, 3).astype(np.float32)
        metadata = WeightMetadata(
            name=f"conv_{i}_weight",
            shape=(32, 3, 3, 3),
            dtype=np.float32,
            layer_type="conv2d",
            model_name="cnn"
        )
        weights.append(WeightTensor(data=data, metadata=metadata))
    
    return weights


def demonstrate_cluster_analysis():
    """Demonstrate cluster analysis functionality."""
    print("🔬 Coral ClusterAnalyzer Demo")
    print("=" * 40)
    
    # Create temporary repository
    temp_dir = tempfile.mkdtemp()
    repo_path = Path(temp_dir) / "cluster_demo_repo"
    
    try:
        # Initialize repository
        print("📁 Initializing repository...")
        repo = Repository(repo_path, init=True)
        
        # Create sample weights
        print("🎯 Creating sample weights...")
        weights = create_sample_weights()
        print(f"   Created {len(weights)} weight tensors")
        
        # Stage and commit weights
        print("💾 Staging and committing weights...")
        weight_dict = {w.metadata.name: w for w in weights}
        repo.stage_weights(weight_dict)
        commit = repo.commit("Initial weights for clustering demo")
        print(f"   Committed: {commit.commit_hash[:8]}")
        
        # Create ClusterAnalyzer
        print("\n🤖 Setting up ClusterAnalyzer...")
        
        # Configure for adaptive clustering
        config = ClusteringConfig(
            strategy=ClusteringStrategy.ADAPTIVE,
            level=ClusterLevel.TENSOR,
            similarity_threshold=0.95,
            min_cluster_size=2,
            feature_extraction="statistical"  # Works well with mixed shapes
        )
        
        analyzer = ClusterAnalyzer(repo, config)
        print(f"   Strategy: {config.strategy.value}")
        print(f"   Similarity threshold: {config.similarity_threshold}")
        
        # Analyze repository
        print("\n📊 Analyzing repository...")
        analysis = analyzer.analyze_repository()
        
        print(f"   Total weights: {analysis.total_weights}")
        print(f"   Unique weights: {analysis.unique_weights}")
        print(f"   Deduplication ratio: {analysis.deduplication_ratio:.1%}")
        print(f"   Weight shapes: {list(analysis.weight_shapes.keys())}")
        print(f"   Layer types: {list(analysis.layer_types.keys())}")
        
        # Perform clustering
        print("\n🎯 Performing clustering analysis...")
        
        # Test different clustering strategies
        strategies_to_test = [
            (ClusteringStrategy.KMEANS, {"k": 3}),
            (ClusteringStrategy.HIERARCHICAL, {}),
            (ClusteringStrategy.DBSCAN, {}),
            (ClusteringStrategy.ADAPTIVE, {})
        ]
        
        results = {}
        
        for strategy, params in strategies_to_test:
            print(f"\n   Testing {strategy.value} clustering...")
            
            try:
                if strategy == ClusteringStrategy.KMEANS:
                    result = analyzer.cluster_kmeans(weights, **params)
                elif strategy == ClusteringStrategy.HIERARCHICAL:
                    result = analyzer.cluster_hierarchical(weights, params)
                elif strategy == ClusteringStrategy.DBSCAN:
                    result = analyzer.cluster_dbscan(weights, params)
                elif strategy == ClusteringStrategy.ADAPTIVE:
                    result = analyzer.cluster_adaptive(weights, params)
                
                results[strategy] = result
                
                print(f"     ✅ Success!")
                print(f"     Clusters: {result.metrics.num_clusters}")
                print(f"     Silhouette score: {result.metrics.silhouette_score:.3f}")
                print(f"     Compression ratio: {result.metrics.compression_ratio:.1%}")
                print(f"     Execution time: {result.execution_time:.2f}s")
                
            except Exception as e:
                print(f"     ❌ Failed: {e}")
        
        # Show best result
        print("\n🏆 Best clustering result:")
        if results:
            best_strategy = max(results.keys(), 
                              key=lambda s: analyzer._evaluate_clustering_quality(results[s]))
            best_result = results[best_strategy]
            
            print(f"   Strategy: {best_strategy.value}")
            print(f"   Quality score: {analyzer._evaluate_clustering_quality(best_result):.3f}")
            print(f"   Silhouette score: {best_result.metrics.silhouette_score:.3f}")
            
            # Show cluster assignments
            print(f"\n📋 Cluster assignments:")
            cluster_groups = {}
            for assignment in best_result.assignments:
                cluster_id = assignment.cluster_id
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(assignment.weight_name)
            
            for cluster_id, weight_names in cluster_groups.items():
                print(f"   {cluster_id}: {', '.join(weight_names)}")
        
        # Demonstrate feature extraction
        print("\n🔍 Feature extraction methods:")
        feature_methods = ["raw", "statistical", "hash", "pca"]
        
        for method in feature_methods:
            try:
                features = analyzer.extract_features(weights[:5], method=method)
                print(f"   {method:12}: {features.shape} features")
            except Exception as e:
                print(f"   {method:12}: Failed - {e}")
        
        # Demonstrate similarity analysis
        print("\n📏 Similarity analysis:")
        if len(weights) >= 4:
            similarity_matrix = analyzer.compute_similarity_matrix(weights[:4])
            print(f"   Similarity matrix shape: {similarity_matrix.shape}")
            print(f"   Average similarity: {similarity_matrix.mean():.3f}")
            print(f"   Max similarity (non-diagonal): {np.max(similarity_matrix - np.eye(4)):.3f}")
        
        # Natural cluster detection
        print("\n🔍 Natural cluster detection:")
        try:
            optimal_k = analyzer.detect_natural_clusters(weights, method="silhouette", max_k=5)
            print(f"   Optimal clusters (silhouette): {optimal_k}")
        except Exception as e:
            print(f"   Failed: {e}")
        
        print("\n✅ Clustering analysis complete!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\n🧹 Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    demonstrate_cluster_analysis()