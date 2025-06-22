"""
Comprehensive integration tests for clustering with Repository.

Tests the full integration of clustering components with the Repository class,
including real weight data, lossless reconstruction, compression ratios,
and edge cases.
"""

import tempfile
import shutil
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

from coral.version_control.repository import Repository
from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.clustering import (
    ClusteringConfig, ClusteringStrategy, ClusterLevel,
    OptimizationConfig, ClusterStorage
)
from coral.storage.hdf5_store import HDF5Store


class TestClusteringRepositoryIntegration:
    """Test clustering integration with Repository class."""
    
    @pytest.fixture
    def temp_repo(self):
        """Create a temporary repository."""
        temp_dir = tempfile.mkdtemp()
        repo = Repository(Path(temp_dir), init=True)
        yield repo
        shutil.rmtree(temp_dir)
        
    @pytest.fixture
    def sample_weights(self) -> Dict[str, WeightTensor]:
        """Create sample weights for testing."""
        np.random.seed(42)
        
        weights = {}
        
        # Create base weights
        base_conv = np.random.randn(64, 3, 7, 7).astype(np.float32)
        base_linear = np.random.randn(768, 768).astype(np.float32)
        base_bias = np.random.randn(64).astype(np.float32)
        
        # Original weights
        weights["conv1.weight"] = WeightTensor(
            data=base_conv,
            metadata=WeightMetadata(name="conv1.weight", shape=(64, 3, 7, 7), dtype=np.float32)
        )
        weights["linear1.weight"] = WeightTensor(
            data=base_linear,
            metadata=WeightMetadata(name="linear1.weight", shape=(768, 768), dtype=np.float32)
        )
        weights["conv1.bias"] = WeightTensor(
            data=base_bias,
            metadata=WeightMetadata(name="conv1.bias", shape=(64,), dtype=np.float32)
        )
        
        # Create similar weights (99% similar)
        noise_factor = 0.01
        weights["conv2.weight"] = WeightTensor(
            data=base_conv + noise_factor * np.random.randn(*base_conv.shape).astype(np.float32),
            metadata=WeightMetadata(name="conv2.weight", shape=(64, 3, 7, 7), dtype=np.float32)
        )
        weights["linear2.weight"] = WeightTensor(
            data=base_linear + noise_factor * np.random.randn(*base_linear.shape).astype(np.float32),
            metadata=WeightMetadata(name="linear2.weight", shape=(768, 768), dtype=np.float32)
        )
        
        # Create very similar weights (99.9% similar)
        tiny_noise = 0.001
        weights["conv3.weight"] = WeightTensor(
            data=base_conv + tiny_noise * np.random.randn(*base_conv.shape).astype(np.float32),
            metadata=WeightMetadata(name="conv3.weight", shape=(64, 3, 7, 7), dtype=np.float32)
        )
        
        # Create duplicates
        weights["conv1_copy.weight"] = WeightTensor(
            data=base_conv.copy(),
            metadata=WeightMetadata(name="conv1_copy.weight", shape=(64, 3, 7, 7), dtype=np.float32)
        )
        
        return weights
        
    def test_repository_clustering_lifecycle(self, temp_repo, sample_weights):
        """Test complete clustering lifecycle in repository."""
        # Stage and commit initial weights
        temp_repo.stage_weights(sample_weights)
        commit1 = temp_repo.commit("Initial weights")
        
        # Analyze repository for clustering opportunities
        analysis = temp_repo.analyze_repository_clusters()
        assert hasattr(analysis, 'clustering_opportunities')
        opportunities = analysis.clustering_opportunities
        assert opportunities.total_clusters_possible > 0
        assert opportunities.estimated_space_savings > 0
        
        # Enable and perform clustering
        config = ClusteringConfig(
            strategy=ClusteringStrategy.ADAPTIVE,
            similarity_threshold=0.95,
            min_cluster_size=2
        )
        
        result = temp_repo.cluster_repository(config)
        assert result.total_weights_clustered > 0
        assert result.num_clusters > 0
        assert result.compression_ratio > 1.0
        assert result.space_savings > 0
        
        # Verify clustering was enabled
        assert temp_repo.clustering_enabled
        
        # Get cluster statistics
        stats = temp_repo.get_cluster_statistics()
        assert stats.total_clusters > 0
        assert stats.total_weights > 0
        assert stats.clustering_coverage > 0
        
    def test_lossless_weight_reconstruction(self, temp_repo, sample_weights):
        """Test that weights can be reconstructed perfectly after clustering."""
        # Stage and commit weights
        temp_repo.stage_weights(sample_weights)
        commit1 = temp_repo.commit("Test weights")
        
        # Enable clustering
        temp_repo.enable_clustering()
        
        # Perform clustering
        result = temp_repo.cluster_repository()
        
        # Retrieve and verify each weight
        for name, original_weight in sample_weights.items():
            reconstructed = temp_repo.get_weight(name, commit1.commit_hash)
            assert reconstructed is not None
            
            # Verify exact reconstruction
            np.testing.assert_array_equal(
                original_weight.data,
                reconstructed.data,
                err_msg=f"Weight {name} was not reconstructed exactly"
            )
            
            # Verify metadata
            assert reconstructed.metadata.name == original_weight.metadata.name
            assert reconstructed.metadata.shape == original_weight.metadata.shape
            
    def test_clustering_with_delta_encoding(self, temp_repo, sample_weights):
        """Test clustering works correctly with delta encoding."""
        # Ensure delta encoding is enabled
        assert temp_repo.deduplicator.enable_delta_encoding
        
        # Stage and commit weights
        temp_repo.stage_weights(sample_weights)
        commit1 = temp_repo.commit("Initial commit")
        
        # Enable clustering
        result = temp_repo.enable_clustering()
        assert result.weights_analyzed > 0
        
        # Verify delta-compatible clustering
        cluster_result = temp_repo.cluster_repository()
        assert cluster_result.delta_compatible
        
        # Create modified weights and commit
        modified_weights = {}
        for name, weight in sample_weights.items():
            # Small modification
            modified_data = weight.data + 0.001 * np.random.randn(*weight.shape).astype(weight.dtype)
            modified_weights[name] = WeightTensor(
                data=modified_data,
                metadata=weight.metadata
            )
            
        temp_repo.stage_weights(modified_weights)
        commit2 = temp_repo.commit("Modified weights")
        
        # Verify weights from both commits
        for name in sample_weights:
            # Original weight
            w1 = temp_repo.get_weight(name, commit1.commit_hash)
            # Modified weight
            w2 = temp_repo.get_weight(name, commit2.commit_hash)
            
            assert w1 is not None and w2 is not None
            assert not np.array_equal(w1.data, w2.data)
            
    def test_compression_ratios(self, temp_repo):
        """Test actual compression ratios achieved by clustering."""
        np.random.seed(42)
        
        # Create weights with varying similarity
        base_weight = np.random.randn(512, 512).astype(np.float32)
        weights = {}
        
        # Create 10 very similar weights (should cluster well)
        for i in range(10):
            noise = 0.001 * np.random.randn(512, 512).astype(np.float32)
            weights[f"similar_{i}"] = WeightTensor(
                data=base_weight + noise,
                metadata=WeightMetadata(name=f"similar_{i}", shape=(512, 512), dtype=np.float32)
            )
            
        # Create 5 moderately similar weights
        for i in range(5):
            noise = 0.05 * np.random.randn(512, 512).astype(np.float32)
            weights[f"moderate_{i}"] = WeightTensor(
                data=base_weight + noise,
                metadata=WeightMetadata(name=f"moderate_{i}", shape=(512, 512), dtype=np.float32)
            )
            
        # Stage and commit
        temp_repo.stage_weights(weights)
        temp_repo.commit("Test compression")
        
        # Measure size before clustering
        storage_size_before = temp_repo.weights_store_path.stat().st_size
        
        # Enable clustering with high similarity threshold
        config = ClusteringConfig(
            strategy=ClusteringStrategy.KMEANS,
            similarity_threshold=0.98,
            min_cluster_size=2
        )
        
        result = temp_repo.cluster_repository(config)
        
        # Verify compression
        assert result.compression_ratio > 2.0  # Should achieve at least 2x compression
        assert result.space_savings > 0
        
        # The very similar weights should form fewer clusters
        assert result.num_clusters < 15  # Should have fewer clusters than weights
        
    def test_clustering_strategies(self, temp_repo, sample_weights):
        """Test different clustering strategies."""
        # Stage and commit weights
        temp_repo.stage_weights(sample_weights)
        temp_repo.commit("Test weights")
        
        strategies = [
            ClusteringStrategy.KMEANS,
            ClusteringStrategy.HIERARCHICAL,
            ClusteringStrategy.DBSCAN,
            ClusteringStrategy.ADAPTIVE
        ]
        
        results = {}
        
        for strategy in strategies:
            # Reset clustering
            if temp_repo.cluster_storage:
                # Clear existing clusters
                with temp_repo.cluster_storage:
                    for cluster_id in temp_repo.cluster_storage.list_clusters():
                        temp_repo.cluster_storage.delete_cluster(cluster_id)
            
            # Cluster with specific strategy
            config = ClusteringConfig(
                strategy=strategy,
                similarity_threshold=0.95,
                min_cluster_size=2
            )
            
            result = temp_repo.cluster_repository(config)
            results[strategy] = result
            
            # Verify each strategy produces valid results
            assert result.total_weights_clustered > 0
            assert result.num_clusters > 0
            assert result.compression_ratio >= 1.0
            
        # Adaptive should generally perform well
        assert results[ClusteringStrategy.ADAPTIVE].compression_ratio >= 1.0
        
    def test_cluster_optimization(self, temp_repo, sample_weights):
        """Test cluster optimization functionality."""
        # Stage and commit weights
        temp_repo.stage_weights(sample_weights)
        temp_repo.commit("Initial weights")
        
        # Initial clustering
        initial_result = temp_repo.cluster_repository()
        initial_clusters = initial_result.num_clusters
        
        # Optimize clusters
        opt_config = OptimizationConfig(
            merge_threshold=0.9,
            split_threshold=0.5,
            rebalance=True
        )
        
        opt_result = temp_repo.optimize_repository_clusters(opt_config)
        
        # Verify optimization occurred
        assert opt_result.clusters_optimized >= 0
        assert opt_result.new_compression_ratio > 0
        
        # Get new statistics
        stats = temp_repo.get_cluster_statistics()
        assert stats.total_clusters > 0
        
    def test_commit_level_clustering(self, temp_repo, sample_weights):
        """Test clustering at commit level."""
        # First commit without clustering
        temp_repo.stage_weights(sample_weights)
        commit1 = temp_repo.commit("Without clustering")
        
        # Enable clustering
        temp_repo.enable_clustering()
        
        # Second commit with clustering
        modified_weights = {}
        for name, weight in sample_weights.items():
            # Slight modification
            modified_weights[name] = WeightTensor(
                data=weight.data * 1.01,
                metadata=weight.metadata
            )
            
        temp_repo.stage_weights(modified_weights)
        commit2 = temp_repo.commit_with_clustering("With clustering")
        
        # Get cluster info for commits
        info1 = temp_repo.get_commit_cluster_info(commit1.commit_hash)
        info2 = temp_repo.get_commit_cluster_info(commit2.commit_hash)
        
        assert info1 is not None
        assert info2 is not None
        assert info2.num_clusters > 0
        assert info2.weights_clustered > 0
        
        # Compare clustering between commits
        comparison = temp_repo.compare_clustering_efficiency(
            commit1.commit_hash,
            commit2.commit_hash
        )
        
        assert comparison is not None
        assert comparison.efficiency_delta != 0
        
    def test_branch_clustering(self, temp_repo, sample_weights):
        """Test branch-level clustering operations."""
        # Create main branch with weights
        temp_repo.stage_weights(sample_weights)
        temp_repo.commit("Main branch commit")
        
        # Create feature branch
        temp_repo.create_branch("feature", "main")
        temp_repo.checkout("feature")
        
        # Add more weights
        extra_weights = {
            "extra1.weight": WeightTensor(
                data=np.random.randn(256, 256).astype(np.float32),
                metadata=WeightMetadata(name="extra1.weight", shape=(256, 256), dtype=np.float32)
            )
        }
        
        temp_repo.stage_weights(extra_weights)
        temp_repo.commit("Feature branch commit")
        
        # Enable clustering
        temp_repo.enable_clustering()
        
        # Cluster branch
        result = temp_repo.cluster_branch_weights("feature")
        assert result.weights_clustered > 0
        assert result.clusters_created > 0
        assert result.commits_affected > 0
        
        # Get branch summary
        summary = temp_repo.get_branch_cluster_summary("feature")
        assert summary.total_weights > 0
        assert summary.total_clusters > 0
        
    def test_clustering_persistence(self, temp_repo, sample_weights):
        """Test that clustering survives repository reload."""
        repo_path = temp_repo.path
        
        # Stage, commit and cluster
        temp_repo.stage_weights(sample_weights)
        commit1 = temp_repo.commit("Test commit")
        
        result = temp_repo.enable_clustering()
        cluster_result = temp_repo.cluster_repository()
        
        original_clusters = cluster_result.num_clusters
        original_compression = cluster_result.compression_ratio
        
        # Close repository (simulate process end)
        del temp_repo
        
        # Reload repository
        new_repo = Repository(repo_path)
        
        # Verify clustering is still enabled and data persists
        assert new_repo.clustering_enabled
        
        stats = new_repo.get_cluster_statistics()
        assert stats.total_clusters == original_clusters
        
        # Verify weights can still be retrieved
        for name in sample_weights:
            weight = new_repo.get_weight(name, commit1.commit_hash)
            assert weight is not None
            np.testing.assert_array_equal(
                weight.data,
                sample_weights[name].data
            )
            
    def test_clustering_garbage_collection(self, temp_repo, sample_weights):
        """Test garbage collection with clustering."""
        # Create multiple commits
        temp_repo.stage_weights(sample_weights)
        commit1 = temp_repo.commit("Commit 1")
        
        # Enable clustering
        temp_repo.enable_clustering()
        temp_repo.cluster_repository()
        
        # Create branch and more commits
        temp_repo.create_branch("temp_branch")
        temp_repo.checkout("temp_branch")
        
        modified = {k: WeightTensor(v.data * 1.1, v.metadata) 
                   for k, v in sample_weights.items()}
        temp_repo.stage_weights(modified)
        temp_repo.commit("Temp commit")
        
        # Delete branch
        temp_repo.checkout("main")
        temp_repo.branch_manager.delete_branch("temp_branch")
        
        # Run garbage collection
        gc_result = temp_repo.gc(include_clusters=True)
        
        assert "cleaned_weights" in gc_result
        assert "cleaned_clusters" in gc_result
        
    def test_clustering_error_handling(self, temp_repo):
        """Test error handling in clustering operations."""
        # Try to cluster without weights
        result = temp_repo.cluster_repository()
        assert result.total_weights_clustered == 0
        assert result.num_clusters == 0
        
        # Try to optimize without clusters
        with pytest.raises(ValueError):
            temp_repo.optimize_repository_clusters()
            
        # Try invalid configuration
        invalid_config = ClusteringConfig(
            strategy=ClusteringStrategy.KMEANS,
            similarity_threshold=2.0,  # Invalid: > 1.0
            min_cluster_size=-1  # Invalid: negative
        )
        
        # Should handle gracefully
        temp_repo.stage_weights({
            "test": WeightTensor(
                data=np.array([1, 2, 3]),
                metadata=WeightMetadata(name="test", shape=(3,), dtype=np.int64)
            )
        })
        temp_repo.commit("Test")
        
        # This should not crash but may produce warnings
        result = temp_repo.cluster_repository(invalid_config)
        
    def test_clustering_performance(self, temp_repo):
        """Test clustering performance with larger datasets."""
        # Create a larger set of weights
        num_weights = 100
        weight_size = (256, 256)
        
        weights = {}
        base_weights = []
        
        # Create 10 base patterns
        for i in range(10):
            base = np.random.randn(*weight_size).astype(np.float32)
            base_weights.append(base)
            
        # Create variations of base patterns
        for i in range(num_weights):
            base_idx = i % 10
            base = base_weights[base_idx]
            noise_level = 0.01 * (i // 10 + 1)  # Increasing noise
            
            weights[f"weight_{i}"] = WeightTensor(
                data=base + noise_level * np.random.randn(*weight_size).astype(np.float32),
                metadata=WeightMetadata(name=f"weight_{i}", shape=weight_size, dtype=np.float32)
            )
            
        # Measure clustering time
        temp_repo.stage_weights(weights)
        temp_repo.commit("Large weight set")
        
        start_time = time.time()
        result = temp_repo.cluster_repository()
        clustering_time = time.time() - start_time
        
        # Performance assertions
        assert clustering_time < 10.0  # Should complete in reasonable time
        assert result.num_clusters < num_weights  # Should create fewer clusters
        assert result.compression_ratio > 1.5  # Should achieve decent compression
        
        # Verify all weights can still be retrieved
        sample_names = list(weights.keys())[:10]
        for name in sample_names:
            weight = temp_repo.get_weight(name)
            assert weight is not None
            
    def test_clustering_cli_methods(self, temp_repo, sample_weights):
        """Test CLI-facing clustering methods."""
        # Stage and commit weights
        temp_repo.stage_weights(sample_weights)
        temp_repo.commit("Test weights")
        
        # Test analyze_clustering
        analysis = temp_repo.analyze_clustering()
        assert "total_weights" in analysis
        assert "potential_clusters" in analysis
        assert "recommendations" in analysis
        
        # Test create_clusters
        result = temp_repo.create_clusters(
            strategy="adaptive",
            similarity_threshold=0.95
        )
        assert "num_clusters" in result
        assert "weights_clustered" in result
        assert "space_saved" in result
        
        # Test get_clustering_status
        status = temp_repo.get_clustering_status()
        assert status["enabled"]
        assert "num_clusters" in status
        assert "clustered_weights" in status
        
        # Test generate_clustering_report
        report = temp_repo.generate_clustering_report(verbose=True)
        assert "overview" in report
        assert "top_clusters" in report
        assert "timestamp" in report
        
        # Test list_clusters
        clusters = temp_repo.list_clusters(sort_by="size", limit=10)
        assert isinstance(clusters, list)
        if clusters:
            assert "id" in clusters[0]
            assert "size" in clusters[0]
            
        # Test get_cluster_info
        if clusters:
            info = temp_repo.get_cluster_info(clusters[0]["id"])
            assert info is not None
            assert "statistics" in info
            assert "weights" in info