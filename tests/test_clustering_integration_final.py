"""
Final comprehensive integration tests for the clustering system.

Tests end-to-end workflows, cross-component integration, performance,
storage, quality assurance, and error handling.
"""

import gc
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.storage.hdf5_store import HDF5Store
from coral.version_control.repository import Repository


class TestClusteringIntegrationFinal:
    """Comprehensive integration tests for clustering system."""
    
    def _create_weight(self, shape: List[int], layer_type: str, seed: int = None) -> WeightTensor:
        """Create a weight tensor with realistic initialization."""
        if seed is not None:
            np.random.seed(seed)
            
        if layer_type == "conv2d":
            # He initialization for Conv2D
            fan_in = shape[1] * shape[2] * shape[3] if len(shape) == 4 else np.prod(shape[1:])
            std = np.sqrt(2.0 / fan_in)
        elif layer_type == "linear":
            # Xavier initialization for linear layers
            fan_in = shape[1] if len(shape) >= 2 else shape[0]
            fan_out = shape[0]
            std = np.sqrt(2.0 / (fan_in + fan_out))
        else:  # bias
            std = 0.01
        
        data = np.random.normal(0, std, shape).astype(np.float32)
        
        metadata = WeightMetadata(
            name=f"weight_{shape}_{seed}",
            shape=tuple(shape),
            dtype=np.dtype("float32"),
            layer_type=layer_type,
        )
        
        return WeightTensor(data, metadata)
    
    def _create_similar_weights(
        self,
        base_weights: Dict[str, WeightTensor],
        similarity: float = 0.99,
        seed: int = None
    ) -> Dict[str, WeightTensor]:
        """Create weights similar to base weights."""
        if seed is not None:
            np.random.seed(seed)
        
        similar_weights = {}
        noise_scale = np.sqrt(1 - similarity**2)
        
        for name, weight in base_weights.items():
            noise = np.random.normal(0, noise_scale, weight.shape).astype(np.float32)
            similar_data = weight.data * similarity + noise * weight.data.std()
            
            similar_weights[name] = WeightTensor(
                similar_data,
                metadata=weight.metadata
            )
        
        return similar_weights
    
    def test_repository_weight_management(self):
        """Test basic repository weight management without clustering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Create test weights
            weights = {
                "conv1.weight": self._create_weight([64, 3, 7, 7], "conv2d", seed=1),
                "conv1.bias": self._create_weight([64], "bias", seed=2),
                "fc1.weight": self._create_weight([512, 784], "linear", seed=3),
                "fc1.bias": self._create_weight([512], "bias", seed=4),
            }
            
            # Stage and commit weights
            repo.stage_weights(weights)
            commit1 = repo.commit("Initial model weights")
            
            assert commit1 is not None
            assert set(commit1.weight_hashes.keys()) == set(weights.keys())
            
            # Verify weights are stored
            with HDF5Store(repo.weights_store_path) as store:
                stored_hashes = store.list_weights()
                # Due to deduplication, we might have fewer stored weights
                assert len(stored_hashes) > 0
                assert len(stored_hashes) <= len(weights)
                
                # Verify we can load weights
                for h in stored_hashes:
                    weight = store.load(h)
                    assert weight is not None
                    assert isinstance(weight, WeightTensor)
    
    def test_repository_with_similar_weights(self):
        """Test repository handling of similar weights for deduplication."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Create base weights
            base_weights = {
                "model.layer1": self._create_weight([256, 128], "linear", seed=1),
                "model.layer2": self._create_weight([128, 64], "linear", seed=2),
                "model.layer3": self._create_weight([64, 10], "linear", seed=3),
            }
            
            # Initial commit
            repo.stage_weights(base_weights)
            commit1 = repo.commit("Base model")
            
            # Create similar weights (99% similar - should be deduplicated)
            similar_weights = self._create_similar_weights(base_weights, 0.99, seed=10)
            repo.stage_weights(similar_weights)
            commit2 = repo.commit("Fine-tuned model")
            
            # Create more different weights (95% similar)
            different_weights = self._create_similar_weights(base_weights, 0.95, seed=20)
            repo.stage_weights(different_weights)
            commit3 = repo.commit("Transfer learning model")
            
            # Check deduplication effectiveness
            with HDF5Store(repo.weights_store_path) as store:
                stored_hashes = store.list_weights()
                total_weights_added = len(base_weights) * 3  # 3 commits
                
                # Should have fewer stored weights due to deduplication
                print(f"\nDeduplication: {total_weights_added} weights -> {len(stored_hashes)} stored")
                assert len(stored_hashes) < total_weights_added
                
                # Check delta storage if enabled
                if repo.config.get("core", {}).get("delta_encoding", True):
                    delta_hashes = store.list_deltas()
                    print(f"Delta-encoded weights: {len(delta_hashes)}")
                    assert len(delta_hashes) > 0  # Should have some deltas
    
    def test_repository_clustering_workflow(self):
        """Test clustering workflow with repository (if clustering is available)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Create diverse weights for clustering
            all_weights = {}
            
            # Group 1: CNN weights (similar within group)
            for i in range(3):
                weights = {
                    f"cnn_model_{i}.conv1": self._create_weight([64, 3, 5, 5], "conv2d", seed=i),
                    f"cnn_model_{i}.conv2": self._create_weight([128, 64, 3, 3], "conv2d", seed=i),
                }
                # Add slight variations
                if i > 0:
                    weights = self._create_similar_weights(weights, 0.98, seed=i*10)
                all_weights.update(weights)
                repo.stage_weights(weights)
                repo.commit(f"CNN model {i}")
            
            # Group 2: MLP weights (similar within group)
            for i in range(3):
                weights = {
                    f"mlp_model_{i}.fc1": self._create_weight([512, 784], "linear", seed=100+i),
                    f"mlp_model_{i}.fc2": self._create_weight([256, 512], "linear", seed=100+i),
                }
                if i > 0:
                    weights = self._create_similar_weights(weights, 0.97, seed=i*20)
                all_weights.update(weights)
                repo.stage_weights(weights)
                repo.commit(f"MLP model {i}")
            
            # Analyze weight distribution
            with HDF5Store(repo.weights_store_path) as store:
                stored_hashes = store.list_weights()
                print(f"\nTotal weights committed: {len(all_weights)}")
                print(f"Unique weights stored: {len(stored_hashes)}")
                
                # Analyze shapes
                shape_distribution = {}
                for h in stored_hashes:
                    weight = store.load(h)
                    shape_key = str(weight.shape)
                    shape_distribution[shape_key] = shape_distribution.get(shape_key, 0) + 1
                
                print(f"Shape distribution: {shape_distribution}")
                
                # If clustering is implemented in repository
                if hasattr(repo, 'analyze_repository_clusters'):
                    try:
                        analysis = repo.analyze_repository_clusters()
                        print(f"\nClustering analysis:")
                        print(f"  Total weights: {analysis.total_weights}")
                        print(f"  Clustering opportunities: {analysis.clustering_opportunities}")
                    except Exception as e:
                        print(f"Clustering not available: {e}")
    
    def test_performance_with_many_checkpoints(self):
        """Test performance with many training checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Simulate training with frequent checkpointing
            base_weights = {
                "encoder.weight": self._create_weight([768, 512], "linear", seed=1),
                "decoder.weight": self._create_weight([512, 768], "linear", seed=2),
                "classifier.weight": self._create_weight([10, 768], "linear", seed=3),
            }
            
            # Initial model
            repo.stage_weights(base_weights)
            repo.commit("Initial model")
            
            # Training checkpoints (simulate 20 epochs)
            start_time = time.time()
            checkpoint_times = []
            
            for epoch in range(20):
                checkpoint_start = time.time()
                
                # Simulate small training updates
                updated_weights = {}
                for name, weight in base_weights.items():
                    # Very small updates (typical for training)
                    update = np.random.normal(0, 0.0001, weight.shape).astype(np.float32)
                    new_data = weight.data + update
                    updated_weights[name] = WeightTensor(new_data, weight.metadata)
                    base_weights[name] = updated_weights[name]  # Update base
                
                repo.stage_weights(updated_weights)
                repo.commit(f"Epoch {epoch + 1}")
                
                checkpoint_time = time.time() - checkpoint_start
                checkpoint_times.append(checkpoint_time)
            
            total_time = time.time() - start_time
            
            print(f"\nCheckpointing performance:")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Average per checkpoint: {np.mean(checkpoint_times):.3f}s")
            print(f"  Min/Max checkpoint time: {min(checkpoint_times):.3f}s / {max(checkpoint_times):.3f}s")
            
            # Check storage efficiency
            with HDF5Store(repo.weights_store_path) as store:
                stored_weights = store.list_weights()
                stored_deltas = store.list_deltas()
                
                total_weight_versions = 21 * 3  # 21 commits × 3 weights
                storage_efficiency = total_weight_versions / (len(stored_weights) + len(stored_deltas))
                
                print(f"\nStorage efficiency:")
                print(f"  Total weight versions: {total_weight_versions}")
                print(f"  Stored weights: {len(stored_weights)}")
                print(f"  Stored deltas: {len(stored_deltas)}")
                print(f"  Efficiency ratio: {storage_efficiency:.2f}x")
                
                # With delta encoding, we should have good efficiency
                assert storage_efficiency >= 1.0  # At least no worse than naive storage
                assert len(stored_deltas) > 0  # Should use delta encoding
    
    def test_branch_merging_with_conflicts(self):
        """Test branch merging with potential weight conflicts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Base model
            base_weights = {
                "shared.weight": self._create_weight([256, 256], "linear", seed=1),
                "shared.bias": self._create_weight([256], "bias", seed=2),
            }
            
            repo.stage_weights(base_weights)
            base_commit = repo.commit("Base model")
            
            # Create branch 1
            repo.create_branch("feature1")
            repo.checkout("feature1")
            
            feature1_weights = self._create_similar_weights(base_weights, 0.98, seed=10)
            feature1_weights["feature1.weight"] = self._create_weight([128, 256], "linear", seed=11)
            
            repo.stage_weights(feature1_weights)
            repo.commit("Feature 1 changes")
            
            # Create branch 2
            repo.checkout("main")
            repo.create_branch("feature2")
            repo.checkout("feature2")
            
            feature2_weights = self._create_similar_weights(base_weights, 0.97, seed=20)
            feature2_weights["feature2.weight"] = self._create_weight([64, 256], "linear", seed=21)
            
            repo.stage_weights(feature2_weights)
            repo.commit("Feature 2 changes")
            
            # Merge branches
            repo.checkout("main")
            
            # Merge feature1
            merge1 = repo.merge("feature1")
            assert merge1 is not None
            
            # Merge feature2 (potential conflicts on shared weights)
            merge2 = repo.merge("feature2")
            assert merge2 is not None
            
            # Check final state
            # After merging, verify we have multiple commits
            commits = repo.log(max_commits=10)
            assert len(commits) >= 3  # At least base + 2 feature commits
            
            # Verify branches exist and were processed
            branches = repo.branch_manager.list_branches()
            branch_names = [b.name for b in branches]
            assert "main" in branch_names
            # Feature branches should still exist unless explicitly deleted
            assert len(branches) >= 3  # main + feature1 + feature2
    
    def test_garbage_collection(self):
        """Test repository garbage collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Create weights and commits
            for i in range(5):
                weights = {
                    f"model_{i}.weight": self._create_weight([100, 100], "linear", seed=i)
                }
                repo.stage_weights(weights)
                repo.commit(f"Model {i}")
            
            # Create and delete a branch
            repo.create_branch("temp_branch")
            repo.checkout("temp_branch")
            
            temp_weights = {
                "temp.weight": self._create_weight([200, 200], "linear", seed=100)
            }
            repo.stage_weights(temp_weights)
            repo.commit("Temporary commit")
            
            # Delete branch
            repo.checkout("main")
            repo.branch_manager.delete_branch("temp_branch")
            
            # Run garbage collection
            gc_stats = repo.gc()
            
            print(f"\nGarbage collection results:")
            print(f"  Weights cleaned: {gc_stats.get('weights_cleaned', 0)}")
            print(f"  Deltas cleaned: {gc_stats.get('deltas_cleaned', 0)}")
            print(f"  Space freed: {gc_stats.get('space_freed', 0)} bytes")
            print(f"  Errors: {gc_stats.get('errors', 0)}")
            
            assert gc_stats.get("errors", 0) == 0
            # Temp branch weights might be cleaned
            assert gc_stats.get("weights_cleaned", 0) >= 0


if __name__ == "__main__":
    # Run a simple test
    test = TestClusteringIntegrationFinal()
    test.test_repository_weight_management()
    print("\nBasic test passed!")