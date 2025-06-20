"""
Performance benchmarks for the clustering system.

Comprehensive benchmarks measuring clustering performance, scalability,
memory efficiency, and comparison with traditional approaches.
"""

import gc as garbage_collector
import json
import os
import platform
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
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
    CentroidEncoder,
)
from coral.clustering.cluster_optimizer import ClusterOptimizer, OptimizationConfig
from coral.core.deduplicator import Deduplicator
from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.delta.delta_encoder import DeltaEncoder
from coral.storage.hdf5_store import HDF5Store
from coral.version_control.repository import Repository


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    dataset_size: int
    execution_time: float
    memory_peak: float
    memory_delta: float
    compression_ratio: float
    weights_processed: int
    clusters_created: int
    extra_metrics: Dict[str, float]


class BenchmarkUtils:
    """Utilities for benchmarking."""
    
    @staticmethod
    def measure_memory():
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def create_realistic_weights(
        num_models: int = 5,
        layers_per_model: int = 10,
        variations_per_model: int = 5
    ) -> Dict[str, List[WeightTensor]]:
        """Create realistic neural network weight distributions."""
        weights_by_model = {}
        
        for model_idx in range(num_models):
            model_weights = []
            
            # Create base model architecture
            layer_configs = [
                ("conv1", [64, 3, 7, 7], "conv2d"),
                ("conv2", [128, 64, 3, 3], "conv2d"),
                ("conv3", [256, 128, 3, 3], "conv2d"),
                ("fc1", [512, 256 * 4 * 4], "linear"),
                ("fc2", [256, 512], "linear"),
                ("fc3", [10, 256], "linear"),
                ("attn.q", [768, 768], "linear"),
                ("attn.k", [768, 768], "linear"),
                ("attn.v", [768, 768], "linear"),
                ("mlp.dense", [3072, 768], "linear"),
            ][:layers_per_model]
            
            # Base model
            base_weights = {}
            for layer_name, shape, layer_type in layer_configs:
                if layer_type == "conv2d":
                    fan_in = shape[1] * shape[2] * shape[3]
                    std = np.sqrt(2.0 / fan_in)
                elif layer_type == "linear":
                    fan_in = shape[1]
                    fan_out = shape[0]
                    std = np.sqrt(2.0 / (fan_in + fan_out))
                else:
                    std = 0.01
                
                data = np.random.normal(0, std, shape).astype(np.float32)
                metadata = WeightMetadata(
                    name=f"model{model_idx}.{layer_name}",
                    shape=tuple(shape),
                    dtype=np.dtype("float32"),
                    layer_type=layer_type,
                )
                base_weights[layer_name] = WeightTensor(data, metadata)
            
            model_weights.append(base_weights)
            
            # Create variations (fine-tuning, continued training, etc.)
            for var_idx in range(variations_per_model):
                variation = {}
                # Similarity decreases with variation index
                similarity = 0.99 - (var_idx * 0.02)
                
                for layer_name, base_weight in base_weights.items():
                    noise_scale = np.sqrt(1 - similarity**2)
                    noise = np.random.normal(0, noise_scale, base_weight.shape)
                    varied_data = base_weight.data * similarity + noise * base_weight.data.std()
                    
                    metadata = WeightMetadata(
                        name=f"model{model_idx}_var{var_idx}.{layer_name}",
                        shape=base_weight.shape,
                        dtype=base_weight.dtype,
                        layer_type=base_weight.metadata.layer_type,
                    )
                    variation[layer_name] = WeightTensor(
                        varied_data.astype(np.float32),
                        metadata
                    )
                
                model_weights.append(variation)
            
            weights_by_model[f"model_{model_idx}"] = model_weights
        
        return weights_by_model


class TestClusteringPerformance:
    """Performance benchmarks for clustering operations."""
    
    def test_clustering_strategy_comparison(self):
        """Compare performance of different clustering strategies."""
        utils = BenchmarkUtils()
        results = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset
            weights_data = utils.create_realistic_weights(
                num_models=3,
                layers_per_model=8,
                variations_per_model=5
            )
            
            # Flatten weights
            all_weights = []
            for model_weights_list in weights_data.values():
                for weight_dict in model_weights_list:
                    all_weights.extend(weight_dict.values())
            
            print(f"\nDataset: {len(all_weights)} weights")
            
            # Test each strategy
            strategies = [
                ClusteringStrategy.KMEANS,
                ClusteringStrategy.HIERARCHICAL,
                ClusteringStrategy.DBSCAN,
                ClusteringStrategy.ADAPTIVE,
            ]
            
            for strategy in strategies:
                store = HDF5Store(Path(tmpdir) / f"{strategy.value}.h5")
                
                # Store weights
                weight_hashes = []
                for weight in all_weights:
                    h = store.save(weight)
                    weight_hashes.append(h)
                
                # Configure clustering
                config = ClusteringConfig(
                    strategy=strategy,
                    level=ClusterLevel.TENSOR,
                    similarity_threshold=0.95,
                    min_cluster_size=3,
                )
                
                # Measure performance
                analyzer = ClusterAnalyzer(store)
                
                garbage_collector.collect()
                mem_start = utils.measure_memory()
                start_time = time.time()
                
                result = analyzer.cluster_weights(weight_hashes, config)
                
                elapsed = time.time() - start_time
                mem_peak = utils.measure_memory()
                mem_delta = mem_peak - mem_start
                
                benchmark_result = BenchmarkResult(
                    name=f"Strategy: {strategy.value}",
                    dataset_size=len(all_weights),
                    execution_time=elapsed,
                    memory_peak=mem_peak,
                    memory_delta=mem_delta,
                    compression_ratio=result.compression_ratio,
                    weights_processed=result.weights_clustered,
                    clusters_created=result.num_clusters,
                    extra_metrics={
                        "avg_cluster_size": result.weights_clustered / max(result.num_clusters, 1),
                        "outliers": result.outliers,
                    }
                )
                
                results.append(benchmark_result)
                
                print(f"\n{strategy.value}:")
                print(f"  Time: {elapsed:.2f}s")
                print(f"  Memory: +{mem_delta:.1f} MB")
                print(f"  Clusters: {result.num_clusters}")
                print(f"  Compression: {result.compression_ratio:.2f}x")
        
        # Verify adaptive performs well
        adaptive_result = next(r for r in results if "ADAPTIVE" in r.name)
        assert adaptive_result.compression_ratio >= max(
            r.compression_ratio for r in results
        ) * 0.95  # Within 5% of best
    
    def test_scalability_analysis(self):
        """Analyze clustering scalability with dataset size."""
        utils = BenchmarkUtils()
        dataset_sizes = [100, 500, 1000, 2500, 5000]
        results = []
        
        print("\n=== Scalability Analysis ===")
        
        for size in dataset_sizes:
            with tempfile.TemporaryDirectory() as tmpdir:
                store = HDF5Store(Path(tmpdir) / f"scale_{size}.h5")
                
                # Create weights
                num_base = size // 10
                weight_hashes = []
                
                for i in range(num_base):
                    # Base weight
                    base = WeightTensor(
                        np.random.randn(64, 64).astype(np.float32),
                        WeightMetadata(
                            name=f"base_{i}",
                            shape=(64, 64),
                            dtype=np.dtype("float32")
                        )
                    )
                    h = store.save(base)
                    weight_hashes.append(h)
                    
                    # Variations
                    for v in range(9):
                        similarity = 0.95 + np.random.random() * 0.04
                        noise = np.random.normal(0, 0.01, base.shape)
                        varied_data = base.data * similarity + noise * base.data.std()
                        
                        varied = WeightTensor(
                            varied_data.astype(np.float32),
                            base.metadata
                        )
                        h = store.save(varied)
                        weight_hashes.append(h)
                
                # Perform clustering
                config = ClusteringConfig(
                    strategy=ClusteringStrategy.ADAPTIVE,
                    similarity_threshold=0.95,
                    min_cluster_size=5,
                )
                
                analyzer = ClusterAnalyzer(store)
                
                garbage_collector.collect()
                mem_start = utils.measure_memory()
                start_time = time.time()
                
                result = analyzer.cluster_weights(weight_hashes, config)
                
                elapsed = time.time() - start_time
                mem_peak = utils.measure_memory()
                
                results.append({
                    "size": size,
                    "time": elapsed,
                    "memory": mem_peak - mem_start,
                    "clusters": result.num_clusters,
                    "compression": result.compression_ratio,
                    "time_per_weight": elapsed / size,
                })
                
                print(f"\nSize {size:5d}: {elapsed:6.2f}s "
                      f"({elapsed/size*1000:.2f}ms/weight), "
                      f"Memory: +{mem_peak - mem_start:.1f}MB, "
                      f"Compression: {result.compression_ratio:.2f}x")
        
        # Analyze scaling behavior
        print("\nScaling Analysis:")
        for i in range(1, len(results)):
            size_ratio = results[i]["size"] / results[i-1]["size"]
            time_ratio = results[i]["time"] / results[i-1]["time"]
            
            print(f"  {results[i-1]['size']} → {results[i]['size']}: "
                  f"Size ×{size_ratio:.1f}, Time ×{time_ratio:.2f}")
            
            # Should scale sub-quadratically
            assert time_ratio < size_ratio ** 1.5
    
    def test_memory_efficiency(self):
        """Test memory efficiency of clustering operations."""
        utils = BenchmarkUtils()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create large dataset
            weights_data = utils.create_realistic_weights(
                num_models=5,
                layers_per_model=10,
                variations_per_model=10
            )
            
            store = HDF5Store(Path(tmpdir) / "memory_test.h5")
            
            # Store weights and measure baseline memory
            weight_hashes = []
            weight_sizes = []
            
            garbage_collector.collect()
            baseline_memory = utils.measure_memory()
            
            for model_weights_list in weights_data.values():
                for weight_dict in model_weights_list:
                    for weight in weight_dict.values():
                        h = store.save(weight)
                        weight_hashes.append(h)
                        weight_sizes.append(weight.data.nbytes)
            
            total_weight_size = sum(weight_sizes) / 1024 / 1024  # MB
            
            # Test different batch sizes
            batch_sizes = [50, 100, 200, 500]
            results = []
            
            print(f"\n=== Memory Efficiency Test ===")
            print(f"Dataset: {len(weight_hashes)} weights, {total_weight_size:.1f} MB")
            
            for batch_size in batch_sizes:
                config = ClusteringConfig(
                    strategy=ClusteringStrategy.KMEANS,
                    similarity_threshold=0.95,
                    batch_size=batch_size,
                )
                
                analyzer = ClusterAnalyzer(store)
                
                garbage_collector.collect()
                mem_start = utils.measure_memory()
                
                # Process in batches
                all_clusters = []
                for i in range(0, len(weight_hashes), batch_size):
                    batch = weight_hashes[i:i + batch_size]
                    result = analyzer.cluster_weights(batch, config)
                    all_clusters.extend(result.clusters)
                
                mem_peak = utils.measure_memory()
                mem_used = mem_peak - mem_start
                
                results.append({
                    "batch_size": batch_size,
                    "memory_used": mem_used,
                    "memory_efficiency": total_weight_size / mem_used,
                })
                
                print(f"\nBatch size {batch_size}:")
                print(f"  Memory used: {mem_used:.1f} MB")
                print(f"  Efficiency: {total_weight_size / mem_used:.2f}x")
            
            # Larger batches should be more memory efficient
            assert results[-1]["memory_efficiency"] >= results[0]["memory_efficiency"]
    
    def test_hierarchical_performance(self):
        """Test performance of hierarchical clustering."""
        utils = BenchmarkUtils()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create hierarchical dataset
            weights_data = utils.create_realistic_weights(
                num_models=4,
                layers_per_model=8,
                variations_per_model=6
            )
            
            store = HDF5Store(Path(tmpdir) / "hierarchical.h5")
            
            # Store weights with hierarchical naming
            weight_info = []  # (hash, level_info)
            
            for model_name, model_weights_list in weights_data.items():
                for var_idx, weight_dict in enumerate(model_weights_list):
                    for layer_name, weight in weight_dict.items():
                        h = store.save(weight)
                        weight_info.append((
                            h,
                            {
                                "model": model_name,
                                "variant": var_idx,
                                "layer": layer_name,
                                "shape": weight.shape,
                            }
                        ))
            
            print(f"\n=== Hierarchical Clustering Performance ===")
            print(f"Dataset: {len(weight_info)} weights")
            
            # Test flat vs hierarchical clustering
            results = {}
            
            # Flat clustering
            garbage_collector.collect()
            start_time = time.time()
            
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                level=ClusterLevel.TENSOR,
                similarity_threshold=0.95,
            )
            
            analyzer = ClusterAnalyzer(store)
            flat_result = analyzer.cluster_weights(
                [h for h, _ in weight_info],
                config
            )
            
            flat_time = time.time() - start_time
            results["flat"] = {
                "time": flat_time,
                "clusters": flat_result.num_clusters,
                "compression": flat_result.compression_ratio,
            }
            
            # Hierarchical clustering
            garbage_collector.collect()
            start_time = time.time()
            
            hierarchy_config = HierarchyConfig(
                levels=[
                    ClusterLevel.TENSOR,
                    ClusterLevel.LAYER,
                    ClusterLevel.MODEL,
                ],
                merge_threshold=0.9,
            )
            
            hierarchy = ClusterHierarchy(hierarchy_config)
            
            # Level 1: Tensor clustering
            tensor_result = analyzer.cluster_weights(
                [h for h, _ in weight_info],
                config
            )
            
            for cluster in tensor_result.clusters:
                hierarchy.add_cluster(ClusterLevel.TENSOR, cluster)
            
            # Level 2: Layer clustering
            layer_clusters = hierarchy.merge_clusters_to_level(ClusterLevel.LAYER)
            
            # Level 3: Model clustering
            model_clusters = hierarchy.merge_clusters_to_level(ClusterLevel.MODEL)
            
            hierarchical_time = time.time() - start_time
            
            metrics = hierarchy.compute_metrics()
            results["hierarchical"] = {
                "time": hierarchical_time,
                "levels": metrics.total_levels,
                "clusters_total": metrics.total_clusters,
                "compression": metrics.compression_ratio,
            }
            
            print("\nFlat clustering:")
            print(f"  Time: {flat_time:.2f}s")
            print(f"  Clusters: {flat_result.num_clusters}")
            print(f"  Compression: {flat_result.compression_ratio:.2f}x")
            
            print("\nHierarchical clustering:")
            print(f"  Time: {hierarchical_time:.2f}s")
            print(f"  Levels: {metrics.total_levels}")
            print(f"  Total clusters: {metrics.total_clusters}")
            print(f"  Compression: {metrics.compression_ratio:.2f}x")
            
            # Hierarchical should provide better organization
            assert metrics.compression_ratio >= flat_result.compression_ratio * 0.9
    
    def test_optimization_performance(self):
        """Test performance of cluster optimization."""
        utils = BenchmarkUtils()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset with suboptimal clustering
            store = HDF5Store(Path(tmpdir) / "optimize.h5")
            
            # Create clusters that can be merged
            weight_groups = []
            for i in range(10):
                base = WeightTensor(
                    np.random.randn(128, 128).astype(np.float32),
                    WeightMetadata(
                        name=f"group_{i}",
                        shape=(128, 128),
                        dtype=np.dtype("float32")
                    )
                )
                
                group = []
                for j in range(5):
                    # Create very similar weights
                    noise = np.random.normal(0, 0.001, base.shape)
                    similar = WeightTensor(
                        (base.data + noise).astype(np.float32),
                        base.metadata
                    )
                    h = store.save(similar)
                    group.append(h)
                
                weight_groups.append(group)
            
            # Initial clustering with high granularity
            config = ClusteringConfig(
                strategy=ClusteringStrategy.KMEANS,
                similarity_threshold=0.99,  # Very high threshold
                min_cluster_size=2,
            )
            
            analyzer = ClusterAnalyzer(store)
            initial_result = analyzer.cluster_weights(
                [h for group in weight_groups for h in group],
                config
            )
            
            print(f"\n=== Cluster Optimization Performance ===")
            print(f"Initial clusters: {initial_result.num_clusters}")
            print(f"Initial compression: {initial_result.compression_ratio:.2f}x")
            
            # Test optimization
            optimizer = ClusterOptimizer(
                OptimizationConfig(
                    rebalance_clusters=True,
                    merge_similar_clusters=True,
                    similarity_threshold=0.95,  # Lower threshold for merging
                    max_iterations=5,
                )
            )
            
            # Load all weights for optimization
            all_weights = {}
            for group in weight_groups:
                for h in group:
                    all_weights[h] = store.load(h)
            
            garbage_collector.collect()
            start_time = time.time()
            
            optimized_clusters = optimizer.optimize_clusters(
                initial_result.clusters,
                all_weights
            )
            
            optimization_time = time.time() - start_time
            
            print(f"\nOptimization time: {optimization_time:.2f}s")
            print(f"Optimized clusters: {len(optimized_clusters)}")
            
            # Calculate new compression ratio
            total_weights = sum(len(c.member_hashes) for c in optimized_clusters)
            new_compression = total_weights / len(optimized_clusters)
            
            print(f"Optimized compression: {new_compression:.2f}x")
            
            # Should reduce cluster count significantly
            assert len(optimized_clusters) < initial_result.num_clusters * 0.5
            assert new_compression > initial_result.compression_ratio
    
    def test_concurrent_performance(self):
        """Test performance with concurrent operations."""
        utils = BenchmarkUtils()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create shared dataset
            weights_data = utils.create_realistic_weights(
                num_models=4,
                layers_per_model=6,
                variations_per_model=5
            )
            
            store = HDF5Store(Path(tmpdir) / "concurrent.h5")
            
            # Store weights
            all_weights = []
            for model_weights_list in weights_data.values():
                for weight_dict in model_weights_list:
                    all_weights.extend(weight_dict.values())
            
            weight_hashes = []
            for weight in all_weights:
                h = store.save(weight)
                weight_hashes.append(h)
            
            print(f"\n=== Concurrent Operations Performance ===")
            print(f"Dataset: {len(weight_hashes)} weights")
            
            # Test concurrent clustering with different strategies
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            strategies = [
                ClusteringStrategy.KMEANS,
                ClusteringStrategy.HIERARCHICAL,
                ClusteringStrategy.DBSCAN,
                ClusteringStrategy.ADAPTIVE,
            ]
            
            def cluster_with_strategy(strategy):
                config = ClusteringConfig(
                    strategy=strategy,
                    similarity_threshold=0.95,
                )
                
                analyzer = ClusterAnalyzer(store)
                start = time.time()
                result = analyzer.cluster_weights(weight_hashes, config)
                elapsed = time.time() - start
                
                return {
                    "strategy": strategy,
                    "time": elapsed,
                    "clusters": result.num_clusters,
                    "compression": result.compression_ratio,
                }
            
            # Sequential execution
            sequential_start = time.time()
            sequential_results = []
            for strategy in strategies:
                result = cluster_with_strategy(strategy)
                sequential_results.append(result)
            sequential_time = time.time() - sequential_start
            
            # Concurrent execution
            concurrent_start = time.time()
            concurrent_results = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(cluster_with_strategy, strategy): strategy
                    for strategy in strategies
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    concurrent_results.append(result)
            
            concurrent_time = time.time() - concurrent_start
            
            print(f"\nSequential time: {sequential_time:.2f}s")
            print(f"Concurrent time: {concurrent_time:.2f}s")
            print(f"Speedup: {sequential_time / concurrent_time:.2f}x")
            
            # Concurrent should be faster
            assert concurrent_time < sequential_time
            
            # Results should be consistent
            seq_sorted = sorted(sequential_results, key=lambda x: x["strategy"].value)
            con_sorted = sorted(concurrent_results, key=lambda x: x["strategy"].value)
            
            for seq, con in zip(seq_sorted, con_sorted):
                assert seq["strategy"] == con["strategy"]
                assert abs(seq["clusters"] - con["clusters"]) <= 1  # Allow small variance
    
    def test_real_world_scenario(self):
        """Test performance in realistic ML workflow."""
        utils = BenchmarkUtils()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            print("\n=== Real-World ML Workflow Performance ===")
            
            # Simulate training workflow
            base_model = utils.create_realistic_weights(
                num_models=1,
                layers_per_model=12,
                variations_per_model=0
            )["model_0"][0]
            
            # Phase 1: Initial training (checkpoints every epoch)
            print("\nPhase 1: Initial training")
            phase1_start = time.time()
            
            for epoch in range(10):
                # Simulate gradual training updates
                checkpoint = {}
                for name, weight in base_model.items():
                    # Small updates
                    update = np.random.normal(0, 0.001, weight.shape)
                    new_data = weight.data + update
                    checkpoint[name] = WeightTensor(
                        new_data.astype(np.float32),
                        weight.metadata
                    )
                
                repo.stage_weights(checkpoint)
                repo.commit(f"Epoch {epoch}")
                base_model = checkpoint  # Update base for next epoch
            
            phase1_time = time.time() - phase1_start
            
            # Phase 2: Fine-tuning on multiple tasks
            print("\nPhase 2: Fine-tuning")
            phase2_start = time.time()
            
            for task in range(3):
                repo.create_branch(f"task_{task}")
                repo.checkout(f"task_{task}")
                
                task_base = base_model.copy()
                for step in range(5):
                    checkpoint = {}
                    for name, weight in task_base.items():
                        # Task-specific updates
                        update = np.random.normal(0, 0.002, weight.shape)
                        new_data = weight.data + update
                        checkpoint[name] = WeightTensor(
                            new_data.astype(np.float32),
                            weight.metadata
                        )
                    
                    repo.stage_weights(checkpoint)
                    repo.commit(f"Task {task} step {step}")
                    task_base = checkpoint
            
            phase2_time = time.time() - phase2_start
            
            # Phase 3: Cluster entire repository
            print("\nPhase 3: Repository clustering")
            phase3_start = time.time()
            
            repo.checkout("main")
            
            # Analyze repository
            analyzer = ClusterAnalyzer(repo.store)
            analysis = analyzer.analyze_repository(repo.path)
            
            print(f"\nRepository analysis:")
            print(f"  Total weights: {analysis.total_weights}")
            print(f"  Unique weights: {analysis.unique_weights}")
            print(f"  Similarity groups: {analysis.similarity_groups}")
            
            # Perform clustering
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
                min_cluster_size=3,
            )
            
            all_hashes = list(repo.store.list_all())
            result = analyzer.cluster_weights(all_hashes, config)
            
            phase3_time = time.time() - phase3_start
            
            # Results
            print(f"\nWorkflow timing:")
            print(f"  Initial training: {phase1_time:.2f}s")
            print(f"  Fine-tuning: {phase2_time:.2f}s")
            print(f"  Clustering: {phase3_time:.2f}s")
            print(f"  Total: {phase1_time + phase2_time + phase3_time:.2f}s")
            
            print(f"\nClustering results:")
            print(f"  Clusters: {result.num_clusters}")
            print(f"  Compression: {result.compression_ratio:.2f}x")
            print(f"  Space saved: {(1 - 1/result.compression_ratio) * 100:.1f}%")
            
            # Should achieve good compression in this scenario
            assert result.compression_ratio > 3.0  # Many similar checkpoints
            assert result.num_clusters < analysis.total_weights / 5


class TestClusteringComparison:
    """Compare clustering with other approaches."""
    
    def test_clustering_vs_deduplication(self):
        """Compare clustering with traditional deduplication."""
        utils = BenchmarkUtils()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create realistic dataset
            weights_data = utils.create_realistic_weights(
                num_models=3,
                layers_per_model=8,
                variations_per_model=8
            )
            
            all_weights = []
            for model_weights_list in weights_data.values():
                for weight_dict in model_weights_list:
                    all_weights.extend(weight_dict.values())
            
            print(f"\n=== Clustering vs Deduplication ===")
            print(f"Dataset: {len(all_weights)} weights")
            
            # Traditional deduplication
            dedup_store = HDF5Store(Path(tmpdir) / "dedup.h5")
            deduplicator = Deduplicator(similarity_threshold=0.95)
            
            dedup_start = time.time()
            dedup_stats = {
                "stored": 0,
                "deduplicated": 0,
                "comparisons": 0,
            }
            
            for weight in all_weights:
                is_duplicate, similar_hash = deduplicator.find_duplicate(
                    weight,
                    dedup_store
                )
                dedup_stats["comparisons"] += dedup_stats["stored"]  # Compare with all stored
                
                if is_duplicate:
                    dedup_stats["deduplicated"] += 1
                else:
                    dedup_store.save(weight)
                    dedup_stats["stored"] += 1
            
            dedup_time = time.time() - dedup_start
            dedup_size = dedup_store.get_total_size()
            
            # Clustering approach
            cluster_store = HDF5Store(Path(tmpdir) / "cluster.h5")
            
            cluster_start = time.time()
            
            # Store all weights
            weight_hashes = []
            for weight in all_weights:
                h = cluster_store.save(weight)
                weight_hashes.append(h)
            
            # Perform clustering
            analyzer = ClusterAnalyzer(cluster_store)
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
            )
            
            result = analyzer.cluster_weights(weight_hashes, config)
            
            cluster_time = time.time() - cluster_start
            cluster_size = cluster_store.get_total_size()
            
            print("\nDeduplication:")
            print(f"  Time: {dedup_time:.2f}s")
            print(f"  Comparisons: {dedup_stats['comparisons']}")
            print(f"  Storage: {dedup_size / 1024 / 1024:.2f} MB")
            print(f"  Stored: {dedup_stats['stored']}")
            print(f"  Deduplicated: {dedup_stats['deduplicated']}")
            
            print("\nClustering:")
            print(f"  Time: {cluster_time:.2f}s")
            print(f"  Storage: {cluster_size / 1024 / 1024:.2f} MB")
            print(f"  Clusters: {result.num_clusters}")
            print(f"  Compression: {result.compression_ratio:.2f}x")
            
            # Clustering should be more efficient
            assert cluster_time < dedup_time * 2  # Allow some overhead
            assert result.compression_ratio > 1.0
    
    def test_clustering_with_delta_encoding(self):
        """Test clustering combined with delta encoding."""
        utils = BenchmarkUtils()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Create dataset with high similarity
            base_weights = utils.create_realistic_weights(
                num_models=1,
                layers_per_model=10,
                variations_per_model=0
            )["model_0"][0]
            
            # Add many similar variations
            for i in range(20):
                similarity = 0.98 + np.random.random() * 0.019  # 0.98-0.999
                varied = {}
                
                for name, weight in base_weights.items():
                    noise = np.random.normal(0, 0.001, weight.shape)
                    varied_data = weight.data * similarity + noise * 0.01
                    varied[name] = WeightTensor(
                        varied_data.astype(np.float32),
                        weight.metadata
                    )
                
                repo.stage_weights(varied)
                repo.commit(f"Variation {i}")
            
            print("\n=== Clustering + Delta Encoding ===")
            
            # Test 1: Delta encoding alone
            delta_encoder = DeltaEncoder()
            delta_results = []
            
            all_weights = []
            for h in repo.store.list_all():
                all_weights.append((h, repo.store.load(h)))
            
            # Use first weight as reference
            reference = all_weights[0][1]
            for weight_hash, weight in all_weights[1:]:
                if weight.shape == reference.shape:
                    delta = delta_encoder.encode(weight, reference)
                    if delta:
                        delta_results.append({
                            "hash": weight_hash,
                            "delta_size": len(delta.to_json()),
                            "original_size": weight.data.nbytes,
                        })
            
            avg_delta_compression = np.mean([
                r["original_size"] / r["delta_size"]
                for r in delta_results
            ])
            
            print(f"Delta encoding alone:")
            print(f"  Average compression: {avg_delta_compression:.2f}x")
            
            # Test 2: Clustering + Delta encoding
            analyzer = ClusterAnalyzer(repo.store)
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.98,
            )
            
            result = analyzer.cluster_weights(
                [h for h, _ in all_weights],
                config
            )
            
            # For each cluster, use centroid as delta reference
            cluster_delta_results = []
            
            for cluster in result.clusters:
                if cluster.size > 1:
                    centroid = cluster.centroid
                    
                    for member_hash in cluster.member_hashes:
                        member = repo.store.load(member_hash)
                        if member.shape == centroid.shape:
                            delta = delta_encoder.encode(member, centroid)
                            if delta:
                                cluster_delta_results.append({
                                    "hash": member_hash,
                                    "delta_size": len(delta.to_json()),
                                    "original_size": member.data.nbytes,
                                })
            
            avg_cluster_delta_compression = np.mean([
                r["original_size"] / r["delta_size"]
                for r in cluster_delta_results
            ])
            
            print(f"\nClustering + Delta encoding:")
            print(f"  Clusters: {result.num_clusters}")
            print(f"  Average compression: {avg_cluster_delta_compression:.2f}x")
            
            # Combined approach should be better
            assert avg_cluster_delta_compression >= avg_delta_compression * 0.95
    
    def test_system_resource_usage(self):
        """Monitor system resource usage during clustering."""
        utils = BenchmarkUtils()
        
        # Get system info
        cpu_count = psutil.cpu_count()
        total_memory = psutil.virtual_memory().total / 1024 / 1024 / 1024  # GB
        
        print(f"\n=== System Resource Usage ===")
        print(f"System: {platform.system()} {platform.release()}")
        print(f"CPU: {cpu_count} cores")
        print(f"Memory: {total_memory:.1f} GB")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create substantial dataset
            weights_data = utils.create_realistic_weights(
                num_models=5,
                layers_per_model=10,
                variations_per_model=10
            )
            
            store = HDF5Store(Path(tmpdir) / "resources.h5")
            
            # Store weights
            weight_hashes = []
            for model_weights_list in weights_data.values():
                for weight_dict in model_weights_list:
                    for weight in weight_dict.values():
                        h = store.save(weight)
                        weight_hashes.append(h)
            
            print(f"\nDataset: {len(weight_hashes)} weights")
            
            # Monitor resources during clustering
            analyzer = ClusterAnalyzer(store)
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
            )
            
            # Resource monitoring
            process = psutil.Process()
            
            resource_samples = []
            monitoring = True
            
            def monitor_resources():
                while monitoring:
                    resource_samples.append({
                        "cpu_percent": process.cpu_percent(),
                        "memory_mb": process.memory_info().rss / 1024 / 1024,
                        "threads": process.num_threads(),
                    })
                    time.sleep(0.1)
            
            # Start monitoring in background
            import threading
            monitor_thread = threading.Thread(target=monitor_resources)
            monitor_thread.start()
            
            # Perform clustering
            start_time = time.time()
            result = analyzer.cluster_weights(weight_hashes, config)
            elapsed = time.time() - start_time
            
            # Stop monitoring
            monitoring = False
            monitor_thread.join()
            
            # Analyze resource usage
            if resource_samples:
                avg_cpu = np.mean([s["cpu_percent"] for s in resource_samples])
                max_memory = max(s["memory_mb"] for s in resource_samples)
                avg_threads = np.mean([s["threads"] for s in resource_samples])
                
                print(f"\nResource usage during clustering:")
                print(f"  Duration: {elapsed:.2f}s")
                print(f"  Average CPU: {avg_cpu:.1f}%")
                print(f"  Peak memory: {max_memory:.1f} MB")
                print(f"  Average threads: {avg_threads:.1f}")
                print(f"  Throughput: {len(weight_hashes) / elapsed:.1f} weights/s")
                
                # Resource usage should be reasonable
                assert max_memory < total_memory * 1024 * 0.5  # Less than 50% of total
                assert avg_cpu < 100 * cpu_count  # Not maxing out all cores