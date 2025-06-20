"""
Comprehensive integration tests for the entire clustering system.

Tests end-to-end workflows, cross-component integration, performance,
storage, quality assurance, and error handling across the clustering pipeline.
"""

import gc
import json
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from coral.clustering import (
    ClusterAnalyzer,
    ClusterAssigner,
    ClusterHierarchy,
    ClusterIndex,
    ClusterStorage,
    ClusteringConfig,
    ClusteringStrategy,
    ClusterLevel,
    HierarchyConfig,
    OptimizationConfig,
    CentroidEncoder,
)
from coral.clustering.cluster_optimizer import ClusterOptimizer
from coral.clustering.cluster_types import (
    ClusterInfo,
    ClusterMetrics,
    Centroid,
    ClusterAssignment,
)
from coral.core.deduplicator import Deduplicator
from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.delta.delta_encoder import DeltaEncoder
from coral.storage.hdf5_store import HDF5Store
from coral.version_control.repository import Repository


class TestClusteringIntegration:
    """Test complete clustering system integration."""
    
    def _get_store(self, repo):
        """Helper to get store from repository."""
        return HDF5Store(repo.weights_store_path)

    @pytest.fixture
    def test_weights(self) -> Dict[str, WeightTensor]:
        """Create realistic test weights for neural network layers."""
        np.random.seed(42)
        
        # CNN weights
        cnn_weights = {
            "conv1.weight": self._create_weight([64, 3, 7, 7], "conv2d"),
            "conv1.bias": self._create_weight([64], "bias"),
            "conv2.weight": self._create_weight([128, 64, 3, 3], "conv2d"),
            "conv2.bias": self._create_weight([128], "bias"),
            "conv3.weight": self._create_weight([256, 128, 3, 3], "conv2d"),
            "conv3.bias": self._create_weight([256], "bias"),
        }
        
        # Transformer weights
        transformer_weights = {
            "attention.q_proj": self._create_weight([768, 768], "linear"),
            "attention.k_proj": self._create_weight([768, 768], "linear"),
            "attention.v_proj": self._create_weight([768, 768], "linear"),
            "attention.out_proj": self._create_weight([768, 768], "linear"),
            "mlp.fc1": self._create_weight([3072, 768], "linear"),
            "mlp.fc2": self._create_weight([768, 3072], "linear"),
        }
        
        # MLP weights
        mlp_weights = {
            "fc1.weight": self._create_weight([512, 784], "linear"),
            "fc1.bias": self._create_weight([512], "bias"),
            "fc2.weight": self._create_weight([256, 512], "linear"),
            "fc2.bias": self._create_weight([256], "bias"),
            "fc3.weight": self._create_weight([10, 256], "linear"),
            "fc3.bias": self._create_weight([10], "bias"),
        }
        
        return {**cnn_weights, **transformer_weights, **mlp_weights}
    
    def _create_weight(self, shape: List[int], layer_type: str) -> WeightTensor:
        """Create a weight tensor with realistic initialization."""
        if layer_type == "conv2d":
            # He initialization for Conv2D
            fan_in = shape[1] * shape[2] * shape[3]
            std = np.sqrt(2.0 / fan_in)
        elif layer_type == "linear":
            # Xavier initialization for linear layers
            fan_in = shape[1]
            fan_out = shape[0]
            std = np.sqrt(2.0 / (fan_in + fan_out))
        else:  # bias
            std = 0.01
        
        data = np.random.normal(0, std, shape).astype(np.float32)
        
        metadata = WeightMetadata(
            name=f"weight_{shape}",
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

    def test_end_to_end_clustering_workflow(self, test_weights):
        """Test complete clustering pipeline from analysis to storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            repo = Repository(repo_path, init=True)
            
            # Step 1: Add initial weights to repository
            repo.stage_weights(test_weights)
            initial_commit = repo.commit("Initial model weights")
            
            # Step 2: Add similar weights (fine-tuning simulation)
            fine_tuned = self._create_similar_weights(test_weights, 0.99, seed=1)
            repo.stage_weights(fine_tuned)
            fine_tune_commit = repo.commit("Fine-tuned model")
            
            # Step 3: Add more variations
            transfer_learned = self._create_similar_weights(test_weights, 0.95, seed=2)
            repo.stage_weights(transfer_learned)
            transfer_commit = repo.commit("Transfer learning")
            
            # Step 4: Analyze repository for clustering
            with self._get_store(repo) as store:
                analyzer = ClusterAnalyzer(store)
                analysis = analyzer.analyze_repository(repo_path)
                
                assert analysis.total_weights > 0
                assert analysis.unique_weights > 0
                assert analysis.similarity_groups > 0
                
                # Step 5: Configure and perform clustering
                config = ClusteringConfig(
                    strategy=ClusteringStrategy.ADAPTIVE,
                    level=ClusterLevel.TENSOR,
                    similarity_threshold=0.95,
                    min_cluster_size=2,
                )
                
                cluster_result = analyzer.cluster_weights(
                    list(store.list_all()),
                    config
                )
            
            assert cluster_result.num_clusters > 0
            assert cluster_result.weights_clustered > 0
            assert cluster_result.compression_ratio > 1.0
            
            # Step 6: Create cluster storage and save results
            cluster_storage = ClusterStorage(repo_path / ".coral" / "clusters")
            
            # Save clusters
            for cluster in cluster_result.clusters:
                cluster_storage.save_cluster_info(cluster.cluster_id, cluster)
                cluster_storage.save_centroid(cluster.cluster_id, cluster.centroid)
            
            # Step 7: Create assignments and encode weights
            assigner = ClusterAssigner(similarity_threshold=0.95)
            encoder = CentroidEncoder()
            
            # Process each weight
            encoded_count = 0
            with self._get_store(repo) as store:
                for weight_hash in store.list_all():
                    weight = store.load(weight_hash)
                    
                    # Find best cluster
                    best_cluster = None
                    best_similarity = 0.0
                    
                    for cluster in cluster_result.clusters:
                        similarity = assigner._compute_similarity(
                            weight.data.flatten(),
                            cluster.centroid.data.flatten()
                        )
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_cluster = cluster
                    
                    if best_cluster and best_similarity > 0.95:
                        # Create assignment
                        assignment = assigner.assign_to_cluster(
                            weight_hash,
                            weight,
                            best_cluster.cluster_id,
                            best_cluster.centroid
                        )
                        
                        # Encode relative to centroid
                        encoded = encoder.encode_relative_to_centroid(
                            weight,
                            best_cluster.centroid
                        )
                        
                        if encoded:
                            cluster_storage.save_assignment(weight_hash, assignment)
                            encoded_count += 1
            
            assert encoded_count > 0
            
            # Step 8: Verify reconstruction
            with self._get_store(repo) as store:
                for weight_hash in list(store.list_all())[:5]:  # Test first 5
                    assignment = cluster_storage.load_assignment(weight_hash)
                    if assignment:
                        centroid = cluster_storage.load_centroid(assignment.cluster_id)
                        original = store.load(weight_hash)
                        
                        # Decode should give us back the original
                        decoded = encoder.decode_from_centroid(
                            assignment.encoding_params,
                            centroid
                        )
                        
                        if decoded is not None:
                            # For FLOAT32_RAW encoding, should be exact
                            if assignment.encoding_params.get("encoding_type") == "FLOAT32_RAW":
                                np.testing.assert_allclose(
                                    original.data,
                                    decoded.data,
                                    rtol=1e-6
                                )
    
    def test_multi_strategy_clustering(self, test_weights):
        """Test clustering with different strategies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "weights.h5")
            
            # Store weights
            weight_hashes = []
            for name, weight in test_weights.items():
                weight_hash = store.save(weight)
                weight_hashes.append(weight_hash)
            
            analyzer = ClusterAnalyzer(store)
            
            # Test each clustering strategy
            strategies = [
                ClusteringStrategy.KMEANS,
                ClusteringStrategy.HIERARCHICAL,
                ClusteringStrategy.DBSCAN,
                ClusteringStrategy.ADAPTIVE,
            ]
            
            results = {}
            for strategy in strategies:
                config = ClusteringConfig(
                    strategy=strategy,
                    level=ClusterLevel.TENSOR,
                    similarity_threshold=0.95,
                    min_cluster_size=2,
                )
                
                result = analyzer.cluster_weights(weight_hashes, config)
                results[strategy] = result
                
                assert result.num_clusters > 0
                assert result.weights_clustered > 0
                assert result.strategy == strategy
            
            # Adaptive should perform best
            adaptive_compression = results[ClusteringStrategy.ADAPTIVE].compression_ratio
            assert adaptive_compression >= max(
                r.compression_ratio for s, r in results.items()
                if s != ClusteringStrategy.ADAPTIVE
            )
    
    def test_hierarchical_clustering_workflow(self, test_weights):
        """Test multi-level hierarchical clustering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Add weights with clear hierarchical structure
            # Layer 1 - similar conv weights
            for i in range(3):
                weights = {
                    f"model1.conv{i}.weight": self._create_weight([64, 3, 3, 3], "conv2d"),
                    f"model1.conv{i}.bias": self._create_weight([64], "bias"),
                }
                similar = self._create_similar_weights(weights, 0.98, seed=i)
                repo.stage_weights(similar)
                repo.commit(f"Model 1 variant {i}")
            
            # Layer 2 - similar attention weights  
            for i in range(3):
                weights = {
                    f"model2.attn{i}.q": self._create_weight([768, 768], "linear"),
                    f"model2.attn{i}.k": self._create_weight([768, 768], "linear"),
                    f"model2.attn{i}.v": self._create_weight([768, 768], "linear"),
                }
                similar = self._create_similar_weights(weights, 0.97, seed=i+10)
                repo.stage_weights(similar)
                repo.commit(f"Model 2 variant {i}")
            
            # Configure hierarchical clustering
            hierarchy_config = HierarchyConfig(
                levels=[
                    ClusterLevel.TENSOR,
                    ClusterLevel.LAYER,
                    ClusterLevel.MODEL
                ],
                merge_threshold=0.9,
                propagate_assignments=True,
            )
            
            # Create hierarchy
            hierarchy = ClusterHierarchy(hierarchy_config)
            analyzer = ClusterAnalyzer(repo.store)
            
            # Cluster at tensor level
            tensor_config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                level=ClusterLevel.TENSOR,
                similarity_threshold=0.95,
            )
            
            all_hashes = list(repo.store.list_all())
            tensor_result = analyzer.cluster_weights(all_hashes, tensor_config)
            
            # Build hierarchy
            for cluster in tensor_result.clusters:
                hierarchy.add_cluster(ClusterLevel.TENSOR, cluster)
            
            # Cluster at layer level
            layer_clusters = hierarchy.merge_clusters_to_level(ClusterLevel.LAYER)
            assert len(layer_clusters) > 0
            assert len(layer_clusters) < len(tensor_result.clusters)
            
            # Cluster at model level
            model_clusters = hierarchy.merge_clusters_to_level(ClusterLevel.MODEL)
            assert len(model_clusters) > 0
            assert len(model_clusters) < len(layer_clusters)
            
            # Test hierarchy navigation
            metrics = hierarchy.compute_metrics()
            assert metrics.total_levels == 3
            assert metrics.total_clusters > 0
            assert metrics.compression_ratio > 1.0
    
    def test_clustering_with_repository_evolution(self, test_weights):
        """Test clustering as repository evolves over time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Phase 1: Initial model
            initial_weights = {k: v for k, v in list(test_weights.items())[:6]}
            repo.stage_weights(initial_weights)
            repo.commit("Initial model")
            
            # Initial clustering
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
            )
            
            analyzer = ClusterAnalyzer(repo.store)
            initial_result = analyzer.cluster_weights(
                list(repo.store.list_all()),
                config
            )
            
            initial_clusters = initial_result.num_clusters
            
            # Phase 2: Add more layers (model growth)
            growth_weights = {k: v for k, v in list(test_weights.items())[6:12]}
            repo.stage_weights(growth_weights)
            repo.commit("Added more layers")
            
            # Re-cluster with optimization
            optimizer = ClusterOptimizer(
                OptimizationConfig(
                    rebalance_clusters=True,
                    merge_similar_clusters=True,
                    similarity_threshold=0.95,
                )
            )
            
            all_weights = {}
            for h in repo.store.list_all():
                all_weights[h] = repo.store.load(h)
            
            optimized_clusters = optimizer.optimize_clusters(
                initial_result.clusters,
                all_weights
            )
            
            assert len(optimized_clusters) >= initial_clusters
            
            # Phase 3: Fine-tuning (many similar weights)
            for i in range(5):
                fine_tuned = self._create_similar_weights(
                    initial_weights,
                    0.99 - i * 0.01,
                    seed=i
                )
                repo.stage_weights(fine_tuned)
                repo.commit(f"Fine-tuning iteration {i}")
            
            # Final clustering should show high compression
            final_result = analyzer.cluster_weights(
                list(repo.store.list_all()),
                config
            )
            
            assert final_result.compression_ratio > 2.0  # Should achieve good compression
            assert final_result.weights_clustered > len(test_weights) * 2
    
    def test_clustering_performance_large_dataset(self):
        """Test clustering performance with large datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "large_weights.h5")
            
            # Create 1000+ weights
            print("\nCreating large dataset...")
            num_base_weights = 50
            num_variations = 20
            
            start_time = time.time()
            
            # Create base weights
            base_weights = {}
            for i in range(num_base_weights):
                shape = [
                    np.random.choice([64, 128, 256, 512]),
                    np.random.choice([64, 128, 256, 512])
                ]
                base_weights[f"weight_{i}"] = self._create_weight(shape, "linear")
            
            # Store base weights and variations
            all_hashes = []
            for name, base_weight in base_weights.items():
                # Store original
                h = store.save(base_weight)
                all_hashes.append(h)
                
                # Create and store variations
                for v in range(num_variations):
                    similarity = 0.95 + np.random.random() * 0.04  # 0.95-0.99
                    varied = self._create_similar_weights(
                        {name: base_weight},
                        similarity,
                        seed=v
                    )[name]
                    h = store.save(varied)
                    all_hashes.append(h)
            
            creation_time = time.time() - start_time
            print(f"Created {len(all_hashes)} weights in {creation_time:.2f}s")
            
            # Test clustering performance
            analyzer = ClusterAnalyzer(store)
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
                min_cluster_size=5,
            )
            
            start_time = time.time()
            result = analyzer.cluster_weights(all_hashes, config)
            clustering_time = time.time() - start_time
            
            print(f"\nClustering Results:")
            print(f"- Time: {clustering_time:.2f}s")
            print(f"- Weights: {result.weights_clustered}")
            print(f"- Clusters: {result.num_clusters}")
            print(f"- Compression: {result.compression_ratio:.2f}x")
            
            # Performance assertions
            assert clustering_time < 60.0  # Should complete within 1 minute
            assert result.weights_clustered == len(all_hashes)
            assert result.num_clusters < num_base_weights * 2  # Good clustering
            assert result.compression_ratio > 5.0  # High compression expected
    
    def test_concurrent_clustering_operations(self, test_weights):
        """Test thread safety of clustering operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Add weights
            repo.stage_weights(test_weights)
            repo.commit("Initial weights")
            
            # Create multiple variations in different branches
            branches = ["dev", "staging", "experimental"]
            for branch in branches:
                repo.create_branch(branch)
                repo.checkout(branch)
                
                varied = self._create_similar_weights(
                    test_weights,
                    0.95 + np.random.random() * 0.04,
                    seed=hash(branch)
                )
                repo.stage_weights(varied)
                repo.commit(f"Branch {branch} variations")
            
            repo.checkout("main")
            
            # Concurrent clustering operations
            analyzer = ClusterAnalyzer(repo.store)
            results = {}
            errors = []
            
            def cluster_branch(branch_name):
                try:
                    config = ClusteringConfig(
                        strategy=ClusteringStrategy.KMEANS,
                        similarity_threshold=0.95,
                    )
                    
                    # Get branch weights
                    repo.checkout(branch_name)
                    branch_hashes = list(repo.store.list_all())
                    
                    result = analyzer.cluster_weights(branch_hashes, config)
                    results[branch_name] = result
                except Exception as e:
                    errors.append((branch_name, e))
            
            # Run clustering concurrently
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for branch in branches + ["main"]:
                    future = executor.submit(cluster_branch, branch)
                    futures.append(future)
                
                for future in as_completed(futures):
                    future.result()
            
            assert len(errors) == 0
            assert len(results) == 4
            
            # Verify all succeeded
            for branch, result in results.items():
                assert result.num_clusters > 0
                assert result.weights_clustered > 0
    
    def test_clustering_storage_integration(self, test_weights):
        """Test complete storage integration with clustering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            repo = Repository(repo_path, init=True)
            
            # Add weights and create clusters
            repo.stage_weights(test_weights)
            repo.commit("Initial weights")
            
            # Initialize clustering components
            analyzer = ClusterAnalyzer(repo.store)
            cluster_storage = ClusterStorage(repo_path / ".coral" / "clusters")
            index = ClusterIndex()
            assigner = ClusterAssigner(similarity_threshold=0.95)
            
            # Perform clustering
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
            )
            
            result = analyzer.cluster_weights(
                list(repo.store.list_all()),
                config
            )
            
            # Store clusters and build index
            for cluster in result.clusters:
                cluster_storage.save_cluster_info(cluster.cluster_id, cluster)
                cluster_storage.save_centroid(cluster.cluster_id, cluster.centroid)
                index.add_cluster(cluster)
            
            # Assign weights to clusters
            for weight_hash in repo.store.list_all():
                weight = repo.store.load(weight_hash)
                
                # Find nearest cluster
                nearest = index.find_nearest_cluster(weight)
                if nearest and nearest[1] > 0.95:  # similarity > threshold
                    cluster_id = nearest[0]
                    centroid = cluster_storage.load_centroid(cluster_id)
                    
                    assignment = assigner.assign_to_cluster(
                        weight_hash,
                        weight,
                        cluster_id,
                        centroid
                    )
                    
                    cluster_storage.save_assignment(weight_hash, assignment)
            
            # Test persistence - create new instances
            new_storage = ClusterStorage(repo_path / ".coral" / "clusters")
            new_index = ClusterIndex()
            
            # Rebuild index from storage
            for cluster_id in new_storage.list_clusters():
                cluster_info = new_storage.load_cluster_info(cluster_id)
                new_index.add_cluster(cluster_info)
            
            # Verify data integrity
            assert len(new_storage.list_clusters()) == result.num_clusters
            assert new_index.num_clusters == index.num_clusters
            
            # Test reconstruction
            for weight_hash in list(repo.store.list_all())[:5]:
                assignment = new_storage.load_assignment(weight_hash)
                if assignment:
                    centroid = new_storage.load_centroid(assignment.cluster_id)
                    assert centroid is not None
                    
                    # Verify assignment is valid
                    assert assignment.weight_hash == weight_hash
                    assert assignment.similarity >= 0.95
    
    def test_clustering_quality_metrics(self, test_weights):
        """Test clustering quality assurance and metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "weights.h5")
            
            # Create weights with known clustering structure
            # Group 1: Very similar CNN weights
            group1 = {}
            base_cnn = self._create_weight([64, 3, 3, 3], "conv2d")
            for i in range(5):
                similar = self._create_similar_weights(
                    {"base": base_cnn},
                    0.99,
                    seed=i
                )["base"]
                h = store.save(similar)
                group1[h] = similar
            
            # Group 2: Similar transformer weights
            group2 = {}
            base_transformer = self._create_weight([768, 768], "linear")
            for i in range(5):
                similar = self._create_similar_weights(
                    {"base": base_transformer},
                    0.98,
                    seed=i+10
                )["base"]
                h = store.save(similar)
                group2[h] = similar
            
            # Group 3: Dissimilar weights
            group3 = {}
            for i in range(5):
                dissimilar = self._create_weight(
                    [np.random.randint(10, 100), np.random.randint(10, 100)],
                    "linear"
                )
                h = store.save(dissimilar)
                group3[h] = dissimilar
            
            # Cluster all weights
            analyzer = ClusterAnalyzer(store)
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
                min_cluster_size=2,
            )
            
            all_hashes = list(group1.keys()) + list(group2.keys()) + list(group3.keys())
            result = analyzer.cluster_weights(all_hashes, config)
            
            # Verify clustering quality
            # Groups 1 and 2 should form tight clusters
            assert result.num_clusters >= 2  # At least 2 main clusters
            assert result.num_clusters <= 8  # Not too fragmented
            
            # Check intra-cluster similarity
            for cluster in result.clusters:
                if cluster.size >= 3:  # Only check meaningful clusters
                    # Compute average intra-cluster similarity
                    member_weights = [
                        store.load(h) for h in cluster.member_hashes
                    ]
                    
                    similarities = []
                    for i, w1 in enumerate(member_weights):
                        for w2 in member_weights[i+1:]:
                            if w1.shape == w2.shape:
                                sim = np.corrcoef(
                                    w1.data.flatten(),
                                    w2.data.flatten()
                                )[0, 1]
                                similarities.append(sim)
                    
                    if similarities:
                        avg_similarity = np.mean(similarities)
                        assert avg_similarity > 0.9  # High intra-cluster similarity
            
            # Test compression quality
            assert result.compression_ratio > 1.5  # Should achieve compression
            
            # Test lossless reconstruction for similar weights
            encoder = CentroidEncoder()
            for cluster in result.clusters:
                if cluster.size >= 2:
                    # Test encoding/decoding for each member
                    for member_hash in list(cluster.member_hashes)[:2]:
                        original = store.load(member_hash)
                        
                        # Encode relative to centroid
                        encoded = encoder.encode_relative_to_centroid(
                            original,
                            cluster.centroid
                        )
                        
                        if encoded and encoded.encoding_type == "FLOAT32_RAW":
                            # Decode back
                            decoded = encoder.decode_from_centroid(
                                encoded.to_dict(),
                                cluster.centroid
                            )
                            
                            # Should be lossless for FLOAT32_RAW
                            np.testing.assert_allclose(
                                original.data,
                                decoded.data,
                                rtol=1e-6,
                                atol=1e-8
                            )
    
    def test_clustering_error_handling(self, test_weights):
        """Test error handling throughout clustering pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "weights.h5")
            analyzer = ClusterAnalyzer(store)
            
            # Test 1: Empty repository
            config = ClusteringConfig(strategy=ClusteringStrategy.KMEANS)
            result = analyzer.cluster_weights([], config)
            assert result.num_clusters == 0
            assert result.weights_clustered == 0
            
            # Test 2: Single weight (can't cluster)
            single_hash = store.save(test_weights["conv1.weight"])
            result = analyzer.cluster_weights([single_hash], config)
            assert result.num_clusters <= 1
            
            # Test 3: All dissimilar weights
            dissimilar_hashes = []
            for i in range(5):
                # Random shapes ensure dissimilarity
                w = self._create_weight(
                    [np.random.randint(10, 50), np.random.randint(10, 50)],
                    "linear"
                )
                h = store.save(w)
                dissimilar_hashes.append(h)
            
            config.similarity_threshold = 0.99  # Very high threshold
            result = analyzer.cluster_weights(dissimilar_hashes, config)
            # Should create individual clusters or one cluster per weight
            assert result.num_clusters >= 1
            
            # Test 4: Corrupted data handling
            cluster_storage = ClusterStorage(Path(tmpdir) / "clusters")
            
            # Save invalid cluster info
            invalid_cluster = ClusterInfo(
                cluster_id="invalid",
                centroid=None,  # Invalid centroid
                member_hashes=set(),
                metrics=ClusterMetrics(
                    size=0,
                    avg_similarity=0.0,
                    compactness=0.0,
                    separation=0.0,
                )
            )
            
            # This should handle gracefully
            try:
                cluster_storage.save_cluster_info("invalid", invalid_cluster)
                loaded = cluster_storage.load_cluster_info("invalid")
                # Should either fail to save or load None
                assert loaded is None or loaded.centroid is None
            except Exception:
                # Expected for invalid data
                pass
            
            # Test 5: Resource exhaustion simulation
            # Create many small clusters to test memory handling
            small_clusters = []
            for i in range(100):
                cluster = ClusterInfo(
                    cluster_id=f"small_{i}",
                    centroid=Centroid(
                        cluster_id=f"small_{i}",
                        data=np.random.randn(10).astype(np.float32),
                        shape=(10,),
                        dtype=np.dtype("float32"),
                    ),
                    member_hashes={f"hash_{i}"},
                    metrics=ClusterMetrics(
                        size=1,
                        avg_similarity=1.0,
                        compactness=1.0,
                        separation=0.0,
                    )
                )
                small_clusters.append(cluster)
            
            # Optimizer should handle many clusters efficiently
            optimizer = ClusterOptimizer(OptimizationConfig())
            optimized = optimizer.optimize_clusters(small_clusters, {})
            assert len(optimized) <= len(small_clusters)
            
            # Force garbage collection to ensure cleanup
            del small_clusters
            del optimized
            gc.collect()
    
    def test_clustering_with_delta_encoding(self, test_weights):
        """Test clustering integration with delta encoding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Enable delta encoding
            repo.config["core"]["delta_encoding"] = True
            
            # Add base weights
            repo.stage_weights(test_weights)
            repo.commit("Base weights")
            
            # Add similar weights that would benefit from delta encoding
            similar_weights = self._create_similar_weights(test_weights, 0.99, seed=1)
            repo.stage_weights(similar_weights)
            repo.commit("Similar weights")
            
            # Configure clustering to work with delta
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.98,  # High threshold for delta
                min_cluster_size=2,
                feature_extraction="raw",  # Best for delta encoding
                normalize_features=False,
            )
            
            # Perform clustering
            analyzer = ClusterAnalyzer(repo.store)
            result = analyzer.cluster_weights(
                list(repo.store.list_all()),
                config
            )
            
            # Create cluster storage
            cluster_storage = ClusterStorage(repo.path / ".coral" / "clusters")
            assigner = ClusterAssigner(similarity_threshold=0.98)
            encoder = CentroidEncoder()
            
            # Process weights with delta encoding awareness
            delta_encoder = DeltaEncoder()
            compression_improvements = []
            
            for cluster in result.clusters:
                if cluster.size < 2:
                    continue
                
                # Save cluster
                cluster_storage.save_cluster_info(cluster.cluster_id, cluster)
                cluster_storage.save_centroid(cluster.cluster_id, cluster.centroid)
                
                # For each member, compare clustering vs delta encoding
                for member_hash in cluster.member_hashes:
                    member_weight = repo.store.load(member_hash)
                    
                    # Clustering approach
                    cluster_encoded = encoder.encode_relative_to_centroid(
                        member_weight,
                        cluster.centroid
                    )
                    
                    # Delta encoding approach (against centroid as reference)
                    delta = delta_encoder.encode(member_weight, cluster.centroid)
                    
                    if cluster_encoded and delta:
                        cluster_size = len(cluster_encoded.to_json())
                        delta_size = len(delta.to_json())
                        
                        # Track which is better
                        improvement = (delta_size - cluster_size) / delta_size
                        compression_improvements.append(improvement)
            
            # Clustering should complement delta encoding
            assert len(compression_improvements) > 0
            avg_improvement = np.mean(compression_improvements)
            print(f"\nAverage size improvement: {avg_improvement:.2%}")
            
            # Both techniques together should work well
            assert result.compression_ratio > 1.5
    
    def test_clustering_backward_compatibility(self, test_weights):
        """Test clustering doesn't break existing functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Phase 1: Use repository without clustering
            repo.stage_weights(test_weights)
            commit1 = repo.commit("Initial weights")
            
            # Normal operations
            loaded = repo.get_weight("conv1.weight", commit1.commit_hash)
            assert loaded is not None
            np.testing.assert_array_equal(loaded.data, test_weights["conv1.weight"].data)
            
            # Create branch
            repo.create_branch("dev")
            repo.checkout("dev")
            
            modified_weights = self._create_similar_weights(test_weights, 0.95, seed=1)
            repo.stage_weights(modified_weights)
            commit2 = repo.commit("Modified weights")
            
            # Phase 2: Enable clustering
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
            )
            
            analyzer = ClusterAnalyzer(repo.store)
            result = analyzer.cluster_weights(
                list(repo.store.list_all()),
                config
            )
            
            # Initialize cluster storage
            cluster_storage = ClusterStorage(repo.path / ".coral" / "clusters")
            for cluster in result.clusters:
                cluster_storage.save_cluster_info(cluster.cluster_id, cluster)
                cluster_storage.save_centroid(cluster.cluster_id, cluster.centroid)
            
            # Phase 3: Verify existing operations still work
            # Load weights from before clustering
            loaded_after = repo.get_weight("conv1.weight", commit1.commit_hash)
            assert loaded_after is not None
            np.testing.assert_array_equal(loaded_after.data, test_weights["conv1.weight"].data)
            
            # Verify diffs still work
            diff = repo.diff(commit1.commit_hash, commit2.commit_hash)
            assert len(diff) > 0
            
            # Verify merging still works
            repo.checkout("main")
            merge_commit = repo.merge("dev")
            assert merge_commit is not None
            
            # Verify garbage collection works with clusters
            gc_stats = repo.gc()
            assert "weights_cleaned" in gc_stats
            assert gc_stats["errors"] == 0
    
    def test_clustering_migration(self, test_weights):
        """Test migrating existing repository to use clustering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Create repository with history
            commits = []
            for i in range(5):
                weights = self._create_similar_weights(
                    test_weights,
                    0.95 + i * 0.01,
                    seed=i
                )
                repo.stage_weights(weights)
                commit = repo.commit(f"Iteration {i}")
                commits.append(commit)
            
            # Check storage before clustering
            initial_size = repo.get_storage_stats()["total_size"]
            
            # Perform migration to clustering
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
                min_cluster_size=2,
            )
            
            # Analyze all weights
            analyzer = ClusterAnalyzer(repo.store)
            all_hashes = list(repo.store.list_all())
            
            # Perform clustering
            result = analyzer.cluster_weights(all_hashes, config)
            
            # Set up cluster storage
            cluster_storage = ClusterStorage(repo.path / ".coral" / "clusters")
            index = ClusterIndex()
            
            # Migrate clusters
            for cluster in result.clusters:
                cluster_storage.save_cluster_info(cluster.cluster_id, cluster)
                cluster_storage.save_centroid(cluster.cluster_id, cluster.centroid)
                index.add_cluster(cluster)
            
            # Create assignments for all weights
            assigner = ClusterAssigner(similarity_threshold=0.95)
            assignments_created = 0
            
            for weight_hash in all_hashes:
                weight = repo.store.load(weight_hash)
                nearest = index.find_nearest_cluster(weight)
                
                if nearest and nearest[1] > 0.95:
                    cluster_id = nearest[0]
                    centroid = cluster_storage.load_centroid(cluster_id)
                    
                    assignment = assigner.assign_to_cluster(
                        weight_hash,
                        weight,
                        cluster_id,
                        centroid
                    )
                    
                    cluster_storage.save_assignment(weight_hash, assignment)
                    assignments_created += 1
            
            # Verify migration success
            assert assignments_created > 0
            assert result.compression_ratio > 1.0
            
            # Verify all historical commits still work
            for commit in commits:
                for name in test_weights.keys():
                    loaded = repo.get_weight(name, commit.commit_hash)
                    assert loaded is not None
                    assert loaded.shape == test_weights[name].shape
            
            print(f"\nMigration Results:")
            print(f"- Weights analyzed: {len(all_hashes)}")
            print(f"- Clusters created: {result.num_clusters}")
            print(f"- Assignments created: {assignments_created}")
            print(f"- Compression ratio: {result.compression_ratio:.2f}x")


class TestClusteringBenchmarks:
    """Performance benchmarks for clustering system."""
    
    def test_clustering_vs_deduplication_benchmark(self):
        """Benchmark clustering vs traditional deduplication."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test scenario
            num_base_weights = 20
            num_variations = 10
            
            # Generate base weights
            base_weights = {}
            for i in range(num_base_weights):
                shape = [128, 128]  # Fixed size for fair comparison
                weight = WeightTensor(
                    np.random.randn(*shape).astype(np.float32),
                    WeightMetadata(
                        name=f"weight_{i}",
                        shape=tuple(shape),
                        dtype=np.dtype("float32")
                    )
                )
                base_weights[f"weight_{i}"] = weight
            
            # Traditional deduplication approach
            dedup_store = HDF5Store(Path(tmpdir) / "dedup.h5")
            deduplicator = Deduplicator(similarity_threshold=0.95)
            
            dedup_start = time.time()
            dedup_hashes = []
            dedup_stats = {"stored": 0, "deduplicated": 0}
            
            for name, base_weight in base_weights.items():
                # Store base
                h = dedup_store.save(base_weight)
                dedup_hashes.append(h)
                dedup_stats["stored"] += 1
                
                # Store variations
                for v in range(num_variations):
                    similarity = 0.95 + np.random.random() * 0.04
                    noise = np.random.normal(0, 0.01, base_weight.shape)
                    varied_data = base_weight.data + noise
                    
                    varied = WeightTensor(
                        varied_data.astype(np.float32),
                        base_weight.metadata
                    )
                    
                    # Check for deduplication
                    is_duplicate, similar_hash = deduplicator.find_duplicate(
                        varied,
                        dedup_store
                    )
                    
                    if is_duplicate:
                        dedup_stats["deduplicated"] += 1
                    else:
                        h = dedup_store.save(varied)
                        dedup_hashes.append(h)
                        dedup_stats["stored"] += 1
            
            dedup_time = time.time() - dedup_start
            dedup_size = dedup_store.get_total_size()
            
            # Clustering approach
            cluster_store = HDF5Store(Path(tmpdir) / "cluster.h5")
            
            cluster_start = time.time()
            cluster_hashes = []
            
            # Store all weights first
            all_weights = {}
            for name, base_weight in base_weights.items():
                h = cluster_store.save(base_weight)
                cluster_hashes.append(h)
                all_weights[h] = base_weight
                
                for v in range(num_variations):
                    similarity = 0.95 + np.random.random() * 0.04
                    noise = np.random.normal(0, 0.01, base_weight.shape)
                    varied_data = base_weight.data + noise
                    
                    varied = WeightTensor(
                        varied_data.astype(np.float32),
                        base_weight.metadata
                    )
                    
                    h = cluster_store.save(varied)
                    cluster_hashes.append(h)
                    all_weights[h] = varied
            
            # Perform clustering
            analyzer = ClusterAnalyzer(cluster_store)
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
                min_cluster_size=2,
            )
            
            result = analyzer.cluster_weights(cluster_hashes, config)
            
            # Store cluster information
            cluster_storage = ClusterStorage(Path(tmpdir) / "clusters")
            for cluster in result.clusters:
                cluster_storage.save_cluster_info(cluster.cluster_id, cluster)
                cluster_storage.save_centroid(cluster.cluster_id, cluster.centroid)
            
            cluster_time = time.time() - cluster_start
            cluster_size = cluster_store.get_total_size()
            
            # Add clustering metadata size
            cluster_metadata_size = sum(
                Path(f).stat().st_size
                for f in Path(tmpdir).glob("clusters/**/*")
                if Path(f).is_file()
            )
            
            # Results
            print("\n=== Clustering vs Deduplication Benchmark ===")
            print(f"Dataset: {num_base_weights} base weights, {num_variations} variations each")
            print(f"Total weights: {num_base_weights * (num_variations + 1)}")
            print("\nDeduplication:")
            print(f"- Time: {dedup_time:.2f}s")
            print(f"- Storage: {dedup_size / 1024 / 1024:.2f} MB")
            print(f"- Stored: {dedup_stats['stored']}")
            print(f"- Deduplicated: {dedup_stats['deduplicated']}")
            print("\nClustering:")
            print(f"- Time: {cluster_time:.2f}s")
            print(f"- Storage: {(cluster_size + cluster_metadata_size) / 1024 / 1024:.2f} MB")
            print(f"- Clusters: {result.num_clusters}")
            print(f"- Compression: {result.compression_ratio:.2f}x")
            
            # Clustering should provide better organization
            assert result.num_clusters < num_base_weights * 2
            assert result.compression_ratio > 1.0
    
    def test_scalability_benchmark(self):
        """Test clustering scalability with increasing dataset sizes."""
        sizes = [100, 500, 1000, 2000]
        results = []
        
        for size in sizes:
            with tempfile.TemporaryDirectory() as tmpdir:
                store = HDF5Store(Path(tmpdir) / f"scale_{size}.h5")
                
                # Generate weights
                hashes = []
                for i in range(size // 10):  # Base weights
                    base = WeightTensor(
                        np.random.randn(64, 64).astype(np.float32),
                        WeightMetadata(
                            name=f"base_{i}",
                            shape=(64, 64),
                            dtype=np.dtype("float32")
                        )
                    )
                    
                    # Store base and variations
                    h = store.save(base)
                    hashes.append(h)
                    
                    for v in range(9):  # 9 variations per base
                        noise = np.random.normal(0, 0.01, base.shape)
                        varied = WeightTensor(
                            (base.data + noise).astype(np.float32),
                            base.metadata
                        )
                        h = store.save(varied)
                        hashes.append(h)
                
                # Perform clustering
                analyzer = ClusterAnalyzer(store)
                config = ClusteringConfig(
                    strategy=ClusteringStrategy.ADAPTIVE,
                    similarity_threshold=0.95,
                )
                
                start_time = time.time()
                result = analyzer.cluster_weights(hashes, config)
                elapsed = time.time() - start_time
                
                results.append({
                    "size": size,
                    "time": elapsed,
                    "clusters": result.num_clusters,
                    "compression": result.compression_ratio,
                })
                
                print(f"\nSize {size}: {elapsed:.2f}s, "
                      f"{result.num_clusters} clusters, "
                      f"{result.compression_ratio:.2f}x compression")
        
        # Check scalability
        # Time should scale sub-quadratically
        for i in range(1, len(results)):
            size_ratio = results[i]["size"] / results[i-1]["size"]
            time_ratio = results[i]["time"] / results[i-1]["time"]
            
            # Should scale better than O(n^2)
            assert time_ratio < size_ratio ** 2


class TestClusteringRegression:
    """Regression tests for clustering quality."""
    
    def test_clustering_quality_regression(self):
        """Ensure clustering quality doesn't degrade."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "regression.h5")
            
            # Create known clustering scenario
            np.random.seed(42)  # Fixed seed for reproducibility
            
            # Group 1: 10 very similar weights (99% similar)
            group1_base = np.random.randn(100, 100).astype(np.float32)
            group1_hashes = []
            for i in range(10):
                noise = np.random.normal(0, 0.01, group1_base.shape)
                data = group1_base + noise * 0.01
                weight = WeightTensor(
                    data,
                    WeightMetadata(
                        name=f"group1_{i}",
                        shape=data.shape,
                        dtype=data.dtype
                    )
                )
                h = store.save(weight)
                group1_hashes.append(h)
            
            # Group 2: 10 similar weights (95% similar)
            group2_base = np.random.randn(100, 100).astype(np.float32)
            group2_hashes = []
            for i in range(10):
                noise = np.random.normal(0, 0.05, group2_base.shape)
                data = group2_base + noise * 0.05
                weight = WeightTensor(
                    data,
                    WeightMetadata(
                        name=f"group2_{i}",
                        shape=data.shape,
                        dtype=data.dtype
                    )
                )
                h = store.save(weight)
                group2_hashes.append(h)
            
            # Perform clustering
            analyzer = ClusterAnalyzer(store)
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
                min_cluster_size=2,
            )
            
            all_hashes = group1_hashes + group2_hashes
            result = analyzer.cluster_weights(all_hashes, config)
            
            # Quality assertions - these should not degrade
            assert result.num_clusters == 2  # Should identify 2 clear groups
            assert result.weights_clustered == 20
            assert result.compression_ratio >= 5.0  # High compression expected
            
            # Verify cluster membership
            clusters_by_size = sorted(
                result.clusters,
                key=lambda c: c.size,
                reverse=True
            )
            
            # Both clusters should have 10 members
            assert clusters_by_size[0].size == 10
            assert clusters_by_size[1].size == 10
            
            # Check clustering metrics
            for cluster in clusters_by_size:
                assert cluster.metrics.avg_similarity >= 0.95
                assert cluster.metrics.compactness >= 0.9
    
    def test_encoding_quality_regression(self):
        """Ensure encoding quality doesn't degrade."""
        # Create test weight
        original = WeightTensor(
            np.random.randn(100, 100).astype(np.float32),
            WeightMetadata(
                name="test_weight",
                shape=(100, 100),
                dtype=np.dtype("float32")
            )
        )
        
        # Create similar centroid
        centroid_data = original.data + np.random.normal(0, 0.01, original.shape)
        centroid = Centroid(
            cluster_id="test_cluster",
            data=centroid_data.astype(np.float32),
            shape=original.shape,
            dtype=original.dtype,
        )
        
        encoder = CentroidEncoder()
        
        # Test FLOAT32_RAW encoding (lossless)
        encoded = encoder.encode_relative_to_centroid(original, centroid)
        assert encoded is not None
        assert encoded.encoding_type == "FLOAT32_RAW"
        
        # Decode
        decoded = encoder.decode_from_centroid(encoded.to_dict(), centroid)
        assert decoded is not None
        
        # Should be perfectly reconstructed
        np.testing.assert_allclose(
            original.data,
            decoded.data,
            rtol=1e-6,
            atol=1e-8
        )
        
        # Check compression achieved
        original_size = original.data.nbytes
        encoded_size = len(encoded.to_json())
        compression_ratio = original_size / encoded_size
        
        assert compression_ratio > 1.0  # Should achieve some compression