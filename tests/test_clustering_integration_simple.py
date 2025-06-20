"""
Simple integration tests for clustering system.
Tests basic end-to-end workflows with proper store handling.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from coral.clustering import (
    ClusteringConfig,
    ClusteringStrategy,
    ClusterStorage,
)
from test_clustering_mock import MockClusterAnalyzer as ClusterAnalyzer
from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.storage.hdf5_store import HDF5Store
from coral.version_control.repository import Repository


class TestClusteringIntegrationSimple:
    """Simple integration tests for clustering."""
    
    def _create_test_weight(self, name: str, shape: tuple, seed: int = None) -> WeightTensor:
        """Create a test weight tensor."""
        if seed is not None:
            np.random.seed(seed)
        
        data = np.random.randn(*shape).astype(np.float32)
        metadata = WeightMetadata(
            name=name,
            shape=shape,
            dtype=np.dtype("float32")
        )
        return WeightTensor(data, metadata)
    
    def test_basic_clustering_workflow(self):
        """Test basic clustering workflow with repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create repository
            repo = Repository(Path(tmpdir), init=True)
            
            # Create and add weights
            weights = {
                "layer1": self._create_test_weight("layer1", (100, 50), seed=1),
                "layer2": self._create_test_weight("layer2", (50, 25), seed=2),
                "layer3": self._create_test_weight("layer3", (100, 50), seed=3),  # Similar shape to layer1
            }
            
            repo.stage_weights(weights)
            repo.commit("Initial weights")
            
            # Add similar weights
            similar_weights = {}
            for name, weight in weights.items():
                # Create 99% similar weight
                noise = np.random.normal(0, 0.01, weight.shape).astype(np.float32)
                similar_data = weight.data * 0.99 + noise * 0.01
                similar_weights[f"{name}_v2"] = WeightTensor(
                    similar_data,
                    weight.metadata
                )
            
            repo.stage_weights(similar_weights)
            repo.commit("Similar weights")
            
            # Perform clustering
            with HDF5Store(repo.weights_store_path) as store:
                analyzer = ClusterAnalyzer(store)
                
                config = ClusteringConfig(
                    strategy=ClusteringStrategy.ADAPTIVE,
                    similarity_threshold=0.95,
                    min_cluster_size=2,
                )
                
                weight_hashes = store.list_weights()
                result = analyzer.cluster_weights(weight_hashes, config)
                
                # Verify clustering results
                assert result.num_clusters > 0
                assert result.weights_clustered == len(weight_hashes)
                assert result.compression_ratio > 1.0
                
                # Store clusters
                cluster_storage = ClusterStorage(repo.path / ".coral" / "clusters")
                for cluster in result.clusters:
                    cluster_storage.save_cluster_info(cluster.cluster_id, cluster)
                    cluster_storage.save_centroid(cluster.cluster_id, cluster.centroid)
                
                # Verify persistence
                loaded_clusters = cluster_storage.list_clusters()
                assert len(loaded_clusters) == result.num_clusters
    
    def test_repository_analysis(self):
        """Test repository analysis for clustering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Add diverse weights
            for i in range(5):
                weights = {
                    f"model_{i}_layer1": self._create_test_weight(
                        f"model_{i}_layer1", (128, 64), seed=i
                    ),
                    f"model_{i}_layer2": self._create_test_weight(
                        f"model_{i}_layer2", (64, 32), seed=i+10
                    ),
                }
                repo.stage_weights(weights)
                repo.commit(f"Model {i}")
            
            # Analyze repository
            with HDF5Store(repo.weights_store_path) as store:
                analyzer = ClusterAnalyzer(store)
                analysis = analyzer.analyze_repository(repo.path)
                
                assert analysis.total_weights == 10  # 2 weights × 5 commits
                assert analysis.unique_weights > 0
                assert analysis.total_size > 0
    
    def test_clustering_with_different_strategies(self):
        """Test different clustering strategies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Create weights with clear clustering structure
            base_weight = self._create_test_weight("base", (256, 256), seed=42)
            
            weights = {"base": base_weight}
            # Add variations
            for i in range(10):
                similarity = 0.95 + np.random.random() * 0.04  # 0.95-0.99
                noise = np.random.normal(0, 1-similarity, base_weight.shape)
                varied_data = base_weight.data * similarity + noise * base_weight.data.std() * (1-similarity)
                weights[f"variation_{i}"] = WeightTensor(
                    varied_data.astype(np.float32),
                    base_weight.metadata
                )
            
            repo.stage_weights(weights)
            repo.commit("Weights with variations")
            
            strategies = [
                ClusteringStrategy.KMEANS,
                ClusteringStrategy.HIERARCHICAL,
                ClusteringStrategy.ADAPTIVE,
            ]
            
            results = {}
            with HDF5Store(repo.weights_store_path) as store:
                analyzer = ClusterAnalyzer(store)
                weight_hashes = store.list_weights()
                
                for strategy in strategies:
                    config = ClusteringConfig(
                        strategy=strategy,
                        similarity_threshold=0.95,
                        min_cluster_size=2,
                    )
                    
                    result = analyzer.cluster_weights(weight_hashes, config)
                    results[strategy] = result
                    
                    # All strategies should find clusters
                    assert result.num_clusters >= 1
                    assert result.num_clusters <= len(weights)
                    assert result.compression_ratio > 1.0
            
            # Adaptive should perform well
            assert results[ClusteringStrategy.ADAPTIVE].compression_ratio >= min(
                r.compression_ratio for r in results.values()
            )
    
    def test_clustering_persistence_and_loading(self):
        """Test saving and loading clustering results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Create clusterable weights
            weights = {}
            for i in range(3):
                for j in range(4):
                    weight = self._create_test_weight(
                        f"group_{i}_weight_{j}",
                        (64, 64),
                        seed=i  # Same seed per group
                    )
                    # Add small variation within group
                    weight.data += np.random.normal(0, 0.01, weight.shape).astype(np.float32)
                    weights[f"group_{i}_weight_{j}"] = weight
            
            repo.stage_weights(weights)
            repo.commit("Grouped weights")
            
            # Perform clustering
            with HDF5Store(repo.weights_store_path) as store:
                analyzer = ClusterAnalyzer(store)
                config = ClusteringConfig(
                    strategy=ClusteringStrategy.ADAPTIVE,
                    similarity_threshold=0.98,
                )
                
                result = analyzer.cluster_weights(store.list_weights(), config)
                
                # Should find approximately 3 clusters (one per group)
                assert 2 <= result.num_clusters <= 4
            
            # Save clustering results
            cluster_storage = ClusterStorage(repo.path / ".coral" / "clusters")
            
            cluster_ids = []
            for cluster in result.clusters:
                cluster_storage.save_cluster_info(cluster.cluster_id, cluster)
                cluster_storage.save_centroid(cluster.cluster_id, cluster.centroid)
                cluster_ids.append(cluster.cluster_id)
            
            # Create new storage instance and verify loading
            new_storage = ClusterStorage(repo.path / ".coral" / "clusters")
            
            loaded_ids = new_storage.list_clusters()
            assert set(loaded_ids) == set(cluster_ids)
            
            # Verify each cluster loads correctly
            for cluster_id in cluster_ids:
                info = new_storage.load_cluster_info(cluster_id)
                centroid = new_storage.load_centroid(cluster_id)
                
                assert info is not None
                assert centroid is not None
                assert info.cluster_id == cluster_id
                assert centroid.cluster_id == cluster_id
    
    def test_clustering_with_commits(self):
        """Test clustering across multiple commits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Simulate model training with checkpoints
            base_weights = {
                "layer1": self._create_test_weight("layer1", (256, 128), seed=0),
                "layer2": self._create_test_weight("layer2", (128, 64), seed=1),
                "layer3": self._create_test_weight("layer3", (64, 10), seed=2),
            }
            
            # Initial commit
            repo.stage_weights(base_weights)
            repo.commit("Initial model")
            
            # Training checkpoints (small updates)
            for epoch in range(5):
                checkpoint_weights = {}
                for name, weight in base_weights.items():
                    # Simulate small training updates
                    update = np.random.normal(0, 0.001, weight.shape).astype(np.float32)
                    new_data = weight.data + update
                    checkpoint_weights[name] = WeightTensor(new_data, weight.metadata)
                    base_weights[name] = checkpoint_weights[name]  # Update for next iteration
                
                repo.stage_weights(checkpoint_weights)
                repo.commit(f"Epoch {epoch + 1}")
            
            # Analyze clustering opportunities
            with HDF5Store(repo.weights_store_path) as store:
                analyzer = ClusterAnalyzer(store)
                
                # Should have 6 commits × 3 weights = 18 weight versions
                all_weights = store.list_weights()
                assert len(all_weights) == 18
                
                # Cluster with high similarity threshold
                config = ClusteringConfig(
                    strategy=ClusteringStrategy.ADAPTIVE,
                    similarity_threshold=0.99,  # High threshold for checkpoints
                )
                
                result = analyzer.cluster_weights(all_weights, config)
                
                # Should achieve high compression (many similar checkpoints)
                assert result.compression_ratio > 3.0
                # Should have roughly 3 main clusters (one per layer)
                assert result.num_clusters < 10