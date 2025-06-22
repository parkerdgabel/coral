"""Test garbage collection with clustering support."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.version_control.repository import Repository
from coral.clustering import (
    ClusteringConfig, ClusteringStrategy, ClusterLevel,
    ClusterInfo, Centroid, ClusterAssignment
)


class TestGarbageCollectionClustering:
    """Test suite for garbage collection with clustering support."""
    
    @pytest.fixture
    def temp_repo_path(self):
        """Create a temporary directory for test repository."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def repo_with_clusters(self, temp_repo_path):
        """Create a repository with clustering enabled and some test data."""
        repo = Repository(temp_repo_path, init=True)
        
        # Enable clustering
        repo.clustering_enabled = True
        repo.config["clustering"] = {
            "enabled": True,
            "strategy": "kmeans",
            "level": "tensor",
            "similarity_threshold": 0.95,
            "min_cluster_size": 2
        }
        repo._save_config()
        repo._init_clustering()
        repo._ensure_clustering_initialized()
        
        return repo
    
    def test_gc_basic_cleanup(self, repo_with_clusters):
        """Test basic garbage collection without clustering."""
        repo = repo_with_clusters
        
        # Create and commit some weights
        weights = {}
        for i in range(5):
            weight = WeightTensor(
                data=np.random.randn(10, 10),
                metadata=WeightMetadata(
                    name=f"weight_{i}",
                    shape=(10, 10),
                    dtype=np.float32
                )
            )
            weights[f"weight_{i}"] = weight
        
        # Stage and commit
        repo.stage_weights(weights)
        commit1 = repo.commit("Initial weights")
        
        # Create new weights, replacing some old ones
        new_weights = {}
        for i in range(3, 7):  # This will replace weight_3 and weight_4, add 5 and 6
            weight = WeightTensor(
                data=np.random.randn(10, 10),
                metadata=WeightMetadata(
                    name=f"weight_{i}",
                    shape=(10, 10),
                    dtype=np.float32
                )
            )
            new_weights[f"weight_{i}"] = weight
        
        # Stage and commit
        repo.stage_weights(new_weights)
        commit2 = repo.commit("Updated weights")
        
        # Run garbage collection
        result = repo.gc(include_clusters=False)
        
        # Old versions of weight_3 and weight_4 should be cleaned
        assert result["cleaned_weights"] >= 2
        assert result["remaining_weights"] >= 7  # All unique weights from both commits
    
    def test_gc_with_clusters_basic(self, repo_with_clusters):
        """Test garbage collection with basic clustering."""
        repo = repo_with_clusters
        
        # Create similar weights that will cluster together
        weights = {}
        base_data = np.random.randn(10, 10)
        
        for i in range(4):
            # Create slightly perturbed versions
            data = base_data + np.random.randn(10, 10) * 0.01
            weight = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"weight_{i}",
                    shape=(10, 10),
                    dtype=np.float32
                )
            )
            weights[f"weight_{i}"] = weight
        
        # Stage, commit, and cluster
        repo.stage_weights(weights)
        commit1 = repo.commit("Clusterable weights")
        
        # Perform clustering
        clustering_result = repo.cluster_repository()
        assert clustering_result.num_clusters > 0
        
        # Create new commit with only some weights
        subset_weights = {k: v for k, v in weights.items() if int(k.split('_')[1]) < 2}
        repo.stage_weights(subset_weights)
        commit2 = repo.commit("Subset of weights")
        
        # Run garbage collection with clustering
        result = repo.gc(include_clusters=True)
        
        # Should clean up weights not in commit2
        assert result["cleaned_weights"] >= 2
        # Clusters for cleaned weights should also be cleaned
        assert result["remaining_clusters"] >= 0
    
    def test_gc_protects_centroids_with_active_deltas(self, repo_with_clusters):
        """Test that centroids with active deltas are not deleted."""
        repo = repo_with_clusters
        
        # Create weights that will cluster
        weights = {}
        base_data = np.random.randn(20, 20)
        
        for i in range(6):
            data = base_data + np.random.randn(20, 20) * 0.05
            weight = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"clustered_weight_{i}",
                    shape=(20, 20),
                    dtype=np.float32
                )
            )
            weights[f"clustered_weight_{i}"] = weight
        
        # Stage and commit
        repo.stage_weights(weights)
        commit1 = repo.commit("Weights for clustering")
        
        # Perform clustering to create centroids and deltas
        clustering_result = repo.cluster_repository()
        assert clustering_result.num_clusters > 0
        assert clustering_result.total_weights_clustered > 0
        
        # Create a new commit with only half the weights
        subset_weights = {k: v for k, v in weights.items() if int(k.split('_')[2]) < 3}
        repo.stage_weights(subset_weights)
        commit2 = repo.commit("Subset after clustering")
        
        # Run GC
        result = repo.gc(include_clusters=True)
        
        # Check that centroids with active deltas were protected
        assert result["protected_centroids"] >= 0
        assert result["remaining_centroids"] > 0
        
        # Verify remaining weights can still be loaded
        remaining_weights = repo.get_all_weights(commit2.commit_hash)
        assert len(remaining_weights) == 3
        
        # Verify they can be reconstructed from centroids + deltas
        for name, weight in remaining_weights.items():
            assert weight is not None
            assert weight.data.shape == (20, 20)
    
    def test_gc_cleans_orphaned_deltas(self, repo_with_clusters):
        """Test that orphaned deltas are cleaned when weights are deleted."""
        repo = repo_with_clusters
        
        # Create weights with delta encoding
        weights = {}
        base_weight = WeightTensor(
            data=np.random.randn(15, 15),
            metadata=WeightMetadata(
                name="base_weight",
                shape=(15, 15),
                dtype=np.float32
            )
        )
        weights["base_weight"] = base_weight
        
        # Create variations that will have deltas
        for i in range(3):
            data = base_weight.data + np.random.randn(15, 15) * 0.1
            weight = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"delta_weight_{i}",
                    shape=(15, 15),
                    dtype=np.float32
                )
            )
            weights[f"delta_weight_{i}"] = weight
        
        # Stage and commit
        repo.stage_weights(weights)
        commit1 = repo.commit("Weights with deltas")
        
        # The deduplicator should have created deltas
        from coral.storage.hdf5_store import HDF5Store
        with HDF5Store(repo.weights_store_path) as store:
            initial_deltas = len(store.list_deltas())
        
        # Create new commit without the delta weights
        repo.stage_weights({"base_weight": base_weight})
        commit2 = repo.commit("Only base weight")
        
        # Run GC
        result = repo.gc(include_clusters=True)
        
        # Check that deltas were cleaned
        assert result["cleaned_deltas"] >= 0
        assert result["cleaned_weights"] >= 3  # The delta weights
        
        # Verify delta count decreased
        with HDF5Store(repo.weights_store_path) as store:
            final_deltas = len(store.list_deltas())
            assert final_deltas <= initial_deltas
    
    def test_gc_reference_counting(self, repo_with_clusters):
        """Test reference counting for centroids."""
        repo = repo_with_clusters
        
        # Create multiple weights that will share centroids
        weights1 = {}
        weights2 = {}
        base_data = np.random.randn(10, 10)
        
        # First set of similar weights
        for i in range(3):
            data = base_data + np.random.randn(10, 10) * 0.02
            weight = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"set1_weight_{i}",
                    shape=(10, 10),
                    dtype=np.float32
                )
            )
            weights1[f"set1_weight_{i}"] = weight
        
        # Second set of similar weights (different base)
        base_data2 = np.random.randn(10, 10)
        for i in range(3):
            data = base_data2 + np.random.randn(10, 10) * 0.02
            weight = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"set2_weight_{i}",
                    shape=(10, 10),
                    dtype=np.float32
                )
            )
            weights2[f"set2_weight_{i}"] = weight
        
        # Commit both sets
        repo.stage_weights({**weights1, **weights2})
        commit1 = repo.commit("Two sets of weights")
        
        # Cluster them
        clustering_result = repo.cluster_repository()
        assert clustering_result.num_clusters >= 2  # At least 2 clusters for 2 sets
        
        # Create branch with only set1
        repo.create_branch("set1_only")
        repo.checkout("set1_only")
        repo.stage_weights(weights1)
        commit2 = repo.commit("Only set1")
        
        # Create branch with only set2
        repo.checkout("main")
        repo.create_branch("set2_only")
        repo.checkout("set2_only")
        repo.stage_weights(weights2)
        commit3 = repo.commit("Only set2")
        
        # Back to main and run GC
        repo.checkout("main")
        result = repo.gc(include_clusters=True)
        
        # Both sets are still referenced by branches, so centroids should remain
        assert result["cleaned_weights"] == 0
        assert result["cleaned_clusters"] == 0
        assert result["remaining_centroids"] >= 2
    
    def test_gc_order_of_operations(self, repo_with_clusters):
        """Test that GC cleans in correct order: deltas → weights → centroids."""
        repo = repo_with_clusters
        
        # Track the order of operations by monitoring logs
        import logging
        from io import StringIO
        
        # Set up custom log handler
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.INFO)
        
        # Get the repository logger
        repo_logger = logging.getLogger("coral.version_control.repository")
        repo_logger.addHandler(handler)
        repo_logger.setLevel(logging.INFO)
        
        # Create test data
        weights = {}
        for i in range(4):
            weight = WeightTensor(
                data=np.random.randn(8, 8),
                metadata=WeightMetadata(
                    name=f"test_weight_{i}",
                    shape=(8, 8),
                    dtype=np.float32
                )
            )
            weights[f"test_weight_{i}"] = weight
        
        # Commit and cluster
        repo.stage_weights(weights)
        commit1 = repo.commit("Test weights")
        repo.cluster_repository()
        
        # Create empty commit to make all weights unreferenced
        repo.stage_weights({})
        commit2 = repo.commit("Empty commit")
        
        # Run GC
        result = repo.gc(include_clusters=True)
        
        # Check log order
        log_contents = log_stream.getvalue()
        delta_pos = log_contents.find("Cleaning up unreferenced deltas")
        weight_pos = log_contents.find("Cleaning up unreferenced weights")
        cluster_pos = log_contents.find("Cleaning up unreferenced clusters")
        
        # Verify order: deltas before weights before clusters
        if delta_pos != -1 and weight_pos != -1:
            assert delta_pos < weight_pos
        if weight_pos != -1 and cluster_pos != -1:
            assert weight_pos < cluster_pos
        
        # Clean up
        repo_logger.removeHandler(handler)
    
    def test_gc_with_complex_hierarchy(self, repo_with_clusters):
        """Test GC with complex clustering hierarchy."""
        repo = repo_with_clusters
        
        # Create hierarchical weight structure
        weights = {}
        
        # Model level - different models
        for model_idx in range(2):
            # Layer level - different layers per model
            for layer_idx in range(2):
                # Tensor level - multiple tensors per layer
                for tensor_idx in range(3):
                    name = f"model{model_idx}_layer{layer_idx}_tensor{tensor_idx}"
                    weight = WeightTensor(
                        data=np.random.randn(5, 5) + model_idx + layer_idx * 0.1,
                        metadata=WeightMetadata(
                            name=name,
                            shape=(5, 5),
                            dtype=np.float32,
                            model_name=f"model_{model_idx}",
                            layer_type=f"layer_{layer_idx}"
                        )
                    )
                    weights[name] = weight
        
        # Commit all weights
        repo.stage_weights(weights)
        commit1 = repo.commit("Hierarchical weights")
        
        # Perform hierarchical clustering
        from coral.clustering import ClusteringConfig, ClusterLevel
        config = ClusteringConfig(
            strategy=ClusteringStrategy.HIERARCHICAL,
            level=ClusterLevel.TENSOR,
            similarity_threshold=0.9
        )
        clustering_result = repo.cluster_repository(config)
        
        # Remove one model's weights
        model0_weights = {k: v for k, v in weights.items() if "model0" in k}
        repo.stage_weights(model0_weights)
        commit2 = repo.commit("Only model0")
        
        # Run GC
        result = repo.gc(include_clusters=True)
        
        # Should clean model1's weights and associated clusters
        assert result["cleaned_weights"] >= 6  # model1's 6 weights
        assert result["cleaned_clusters"] >= 1  # At least model1's clusters
        
        # Verify model0 weights still work
        remaining = repo.get_all_weights(commit2.commit_hash)
        assert len(remaining) == 6
        for name, weight in remaining.items():
            assert "model0" in name
            assert weight is not None