"""
Edge case tests for clustering system.

Tests unusual inputs, boundary conditions, error scenarios,
and extreme cases to ensure robustness.
"""

import gc
import tempfile
from pathlib import Path
from typing import List

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
    CentroidEncoder,
)
from coral.clustering.cluster_optimizer import ClusterOptimizer, OptimizationConfig
from coral.clustering.cluster_types import ClusterInfo, ClusterMetrics, Centroid
from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.storage.hdf5_store import HDF5Store


class TestClusteringEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_repository_clustering(self):
        """Test clustering on empty repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "empty.h5")
            analyzer = ClusterAnalyzer(store)
            
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
            )
            
            # Cluster empty list
            result = analyzer.cluster_weights([], config)
            
            assert result.num_clusters == 0
            assert result.weights_clustered == 0
            assert result.compression_ratio == 1.0
            assert result.outliers == 0
            assert len(result.clusters) == 0
    
    def test_single_weight_clustering(self):
        """Test clustering with only one weight."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "single.h5")
            
            weight = WeightTensor(
                np.random.randn(64, 64).astype(np.float32),
                WeightMetadata(
                    name="single_weight",
                    shape=(64, 64),
                    dtype=np.dtype("float32")
                )
            )
            
            h = store.save(weight)
            
            # Test different strategies
            strategies = [
                ClusteringStrategy.KMEANS,
                ClusteringStrategy.HIERARCHICAL,
                ClusteringStrategy.DBSCAN,
                ClusteringStrategy.ADAPTIVE,
            ]
            
            for strategy in strategies:
                config = ClusteringConfig(
                    strategy=strategy,
                    similarity_threshold=0.95,
                    min_cluster_size=1,  # Allow single-weight clusters
                )
                
                analyzer = ClusterAnalyzer(store)
                result = analyzer.cluster_weights([h], config)
                
                # Should handle gracefully
                assert result.weights_clustered == 1
                assert result.num_clusters <= 1
                assert result.compression_ratio == 1.0
    
    def test_all_identical_weights(self):
        """Test clustering when all weights are identical."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "identical.h5")
            
            # Create identical weights
            base_data = np.random.randn(128, 128).astype(np.float32)
            hashes = []
            
            for i in range(20):
                weight = WeightTensor(
                    base_data.copy(),
                    WeightMetadata(
                        name=f"identical_{i}",
                        shape=base_data.shape,
                        dtype=np.dtype("float32")
                    )
                )
                h = store.save(weight)
                hashes.append(h)
            
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
            )
            
            analyzer = ClusterAnalyzer(store)
            result = analyzer.cluster_weights(hashes, config)
            
            # Should create single cluster
            assert result.num_clusters == 1
            assert result.weights_clustered == 20
            assert result.compression_ratio == 20.0
            
            # Verify cluster quality
            cluster = result.clusters[0]
            assert cluster.size == 20
            assert cluster.metrics.avg_similarity == 1.0
            assert cluster.metrics.compactness == 1.0
    
    def test_different_shapes_clustering(self):
        """Test clustering with weights of different shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "shapes.h5")
            
            hashes = []
            shapes = [
                (64, 64),
                (128, 64),
                (64, 128),
                (256, 256),
                (32, 32),
                (100,),  # 1D
                (10, 10, 10),  # 3D
                (5, 5, 5, 5),  # 4D
            ]
            
            for i, shape in enumerate(shapes):
                weight = WeightTensor(
                    np.random.randn(*shape).astype(np.float32),
                    WeightMetadata(
                        name=f"shape_{i}",
                        shape=shape,
                        dtype=np.dtype("float32")
                    )
                )
                h = store.save(weight)
                hashes.append(h)
            
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
            )
            
            analyzer = ClusterAnalyzer(store)
            result = analyzer.cluster_weights(hashes, config)
            
            # Should handle different shapes
            assert result.weights_clustered == len(shapes)
            # Each shape should likely be in its own cluster
            assert result.num_clusters >= len(set(shapes))
    
    def test_extreme_values_clustering(self):
        """Test clustering with extreme weight values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "extreme.h5")
            
            extreme_weights = []
            
            # Very large values
            large = WeightTensor(
                np.full((50, 50), 1e10, dtype=np.float32),
                WeightMetadata(
                    name="very_large",
                    shape=(50, 50),
                    dtype=np.dtype("float32")
                )
            )
            extreme_weights.append(large)
            
            # Very small values
            small = WeightTensor(
                np.full((50, 50), 1e-10, dtype=np.float32),
                WeightMetadata(
                    name="very_small",
                    shape=(50, 50),
                    dtype=np.dtype("float32")
                )
            )
            extreme_weights.append(small)
            
            # All zeros
            zeros = WeightTensor(
                np.zeros((50, 50), dtype=np.float32),
                WeightMetadata(
                    name="all_zeros",
                    shape=(50, 50),
                    dtype=np.dtype("float32")
                )
            )
            extreme_weights.append(zeros)
            
            # All ones
            ones = WeightTensor(
                np.ones((50, 50), dtype=np.float32),
                WeightMetadata(
                    name="all_ones",
                    shape=(50, 50),
                    dtype=np.dtype("float32")
                )
            )
            extreme_weights.append(ones)
            
            # Mixed extreme values
            mixed = np.random.randn(50, 50).astype(np.float32)
            mixed[0:10, :] = 1e8
            mixed[10:20, :] = 1e-8
            mixed[20:30, :] = 0
            mixed_weight = WeightTensor(
                mixed,
                WeightMetadata(
                    name="mixed_extreme",
                    shape=(50, 50),
                    dtype=np.dtype("float32")
                )
            )
            extreme_weights.append(mixed_weight)
            
            # Store all weights
            hashes = []
            for weight in extreme_weights:
                h = store.save(weight)
                hashes.append(h)
            
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
            )
            
            analyzer = ClusterAnalyzer(store)
            result = analyzer.cluster_weights(hashes, config)
            
            # Should handle extreme values without crashing
            assert result.weights_clustered == len(extreme_weights)
            assert result.num_clusters >= 1
            
            # Test encoding/decoding extreme values
            encoder = CentroidEncoder()
            for cluster in result.clusters:
                if cluster.size >= 1:
                    member_hash = list(cluster.member_hashes)[0]
                    member = store.load(member_hash)
                    
                    encoded = encoder.encode_relative_to_centroid(
                        member,
                        cluster.centroid
                    )
                    
                    if encoded:
                        decoded = encoder.decode_from_centroid(
                            encoded.to_dict(),
                            cluster.centroid
                        )
                        
                        if decoded is not None:
                            # Should preserve values (within float32 precision)
                            np.testing.assert_allclose(
                                member.data,
                                decoded.data,
                                rtol=1e-5,
                                atol=1e-7
                            )
    
    def test_sparse_weights_clustering(self):
        """Test clustering with very sparse weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "sparse.h5")
            
            hashes = []
            
            # Create weights with different sparsity levels
            sparsity_levels = [0.99, 0.95, 0.90, 0.50, 0.10]
            
            for i, sparsity in enumerate(sparsity_levels):
                data = np.random.randn(256, 256).astype(np.float32)
                mask = np.random.random((256, 256)) > sparsity
                sparse_data = data * mask
                
                weight = WeightTensor(
                    sparse_data,
                    WeightMetadata(
                        name=f"sparse_{sparsity}",
                        shape=(256, 256),
                        dtype=np.dtype("float32"),
                        sparsity=sparsity
                    )
                )
                h = store.save(weight)
                hashes.append(h)
            
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
            )
            
            analyzer = ClusterAnalyzer(store)
            result = analyzer.cluster_weights(hashes, config)
            
            # Should handle sparse weights
            assert result.weights_clustered == len(sparsity_levels)
            
            # Test sparse encoding
            encoder = CentroidEncoder()
            for cluster in result.clusters:
                for member_hash in list(cluster.member_hashes)[:1]:
                    member = store.load(member_hash)
                    
                    # Count non-zero elements
                    non_zero_ratio = np.count_nonzero(member.data) / member.data.size
                    
                    encoded = encoder.encode_relative_to_centroid(
                        member,
                        cluster.centroid
                    )
                    
                    if encoded and non_zero_ratio < 0.1:  # Very sparse
                        # Sparse encoding should be efficient
                        encoding_size = len(encoded.to_json())
                        original_size = member.data.nbytes
                        compression = original_size / encoding_size
                        
                        # Sparse weights should compress well
                        assert compression > 5.0
    
    def test_tiny_weights_clustering(self):
        """Test clustering with very small weight tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "tiny.h5")
            
            hashes = []
            
            # Create tiny weights
            tiny_shapes = [
                (1,),
                (2,),
                (1, 1),
                (2, 2),
                (3, 3),
                (1, 10),
                (10, 1),
            ]
            
            for shape in tiny_shapes:
                weight = WeightTensor(
                    np.random.randn(*shape).astype(np.float32),
                    WeightMetadata(
                        name=f"tiny_{shape}",
                        shape=shape,
                        dtype=np.dtype("float32")
                    )
                )
                h = store.save(weight)
                hashes.append(h)
            
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
                min_weight_size=0,  # Allow tiny weights
            )
            
            analyzer = ClusterAnalyzer(store)
            result = analyzer.cluster_weights(hashes, config)
            
            # Should handle tiny weights
            assert result.weights_clustered == len(tiny_shapes)
            assert result.num_clusters >= 1
    
    def test_huge_weights_clustering(self):
        """Test clustering with very large weight tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "huge.h5")
            
            # Create a few large weights (but not too many to avoid memory issues)
            hashes = []
            
            for i in range(3):
                # ~40MB per weight
                large_weight = WeightTensor(
                    np.random.randn(2048, 2048).astype(np.float32),
                    WeightMetadata(
                        name=f"huge_{i}",
                        shape=(2048, 2048),
                        dtype=np.dtype("float32")
                    )
                )
                h = store.save(large_weight)
                hashes.append(h)
            
            config = ClusteringConfig(
                strategy=ClusteringStrategy.KMEANS,  # Simpler strategy for large weights
                similarity_threshold=0.95,
                batch_size=2,  # Small batch to manage memory
            )
            
            analyzer = ClusterAnalyzer(store)
            result = analyzer.cluster_weights(hashes, config)
            
            # Should handle large weights
            assert result.weights_clustered == 3
            assert result.num_clusters >= 1
            
            # Clean up to free memory
            del large_weight
            gc.collect()
    
    def test_nan_inf_weights(self):
        """Test clustering with NaN and Inf values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "special.h5")
            
            hashes = []
            
            # Weight with NaN
            nan_data = np.random.randn(64, 64).astype(np.float32)
            nan_data[10:20, 10:20] = np.nan
            nan_weight = WeightTensor(
                nan_data,
                WeightMetadata(
                    name="with_nan",
                    shape=(64, 64),
                    dtype=np.dtype("float32")
                )
            )
            hashes.append(store.save(nan_weight))
            
            # Weight with Inf
            inf_data = np.random.randn(64, 64).astype(np.float32)
            inf_data[30:40, 30:40] = np.inf
            inf_data[40:50, 40:50] = -np.inf
            inf_weight = WeightTensor(
                inf_data,
                WeightMetadata(
                    name="with_inf",
                    shape=(64, 64),
                    dtype=np.dtype("float32")
                )
            )
            hashes.append(store.save(inf_weight))
            
            # Normal weight for comparison
            normal_weight = WeightTensor(
                np.random.randn(64, 64).astype(np.float32),
                WeightMetadata(
                    name="normal",
                    shape=(64, 64),
                    dtype=np.dtype("float32")
                )
            )
            hashes.append(store.save(normal_weight))
            
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
            )
            
            analyzer = ClusterAnalyzer(store)
            
            # Should handle special values gracefully
            result = analyzer.cluster_weights(hashes, config)
            assert result.weights_clustered <= len(hashes)
            assert result.num_clusters >= 1
    
    def test_memory_constrained_clustering(self):
        """Test clustering behavior under memory constraints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "memory.h5")
            
            # Create many medium-sized weights
            hashes = []
            for i in range(100):
                weight = WeightTensor(
                    np.random.randn(256, 256).astype(np.float32),
                    WeightMetadata(
                        name=f"weight_{i}",
                        shape=(256, 256),
                        dtype=np.dtype("float32")
                    )
                )
                h = store.save(weight)
                hashes.append(h)
            
            # Use small batch size to simulate memory constraints
            config = ClusteringConfig(
                strategy=ClusteringStrategy.KMEANS,
                similarity_threshold=0.95,
                batch_size=10,  # Process in small batches
                max_memory_mb=100,  # Simulated memory limit
            )
            
            analyzer = ClusterAnalyzer(store)
            
            # Process in batches
            all_clusters = []
            for i in range(0, len(hashes), config.batch_size):
                batch = hashes[i:i + config.batch_size]
                result = analyzer.cluster_weights(batch, config)
                all_clusters.extend(result.clusters)
                
                # Force garbage collection between batches
                gc.collect()
            
            # Should complete without memory issues
            assert len(all_clusters) > 0
            total_weights = sum(c.size for c in all_clusters)
            assert total_weights == len(hashes)
    
    def test_concurrent_clustering_safety(self):
        """Test thread safety of clustering operations."""
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "concurrent.h5")
            
            # Create test weights
            weights = []
            for i in range(20):
                weight = WeightTensor(
                    np.random.randn(128, 128).astype(np.float32),
                    WeightMetadata(
                        name=f"concurrent_{i}",
                        shape=(128, 128),
                        dtype=np.dtype("float32")
                    )
                )
                weights.append(weight)
            
            # Store weights
            hashes = []
            for weight in weights:
                h = store.save(weight)
                hashes.append(h)
            
            # Test concurrent clustering with different configurations
            configs = [
                ClusteringConfig(
                    strategy=ClusteringStrategy.KMEANS,
                    similarity_threshold=0.95,
                ),
                ClusteringConfig(
                    strategy=ClusteringStrategy.HIERARCHICAL,
                    similarity_threshold=0.90,
                ),
                ClusteringConfig(
                    strategy=ClusteringStrategy.DBSCAN,
                    similarity_threshold=0.93,
                ),
            ]
            
            results = []
            errors = []
            
            def cluster_subset(config, subset_hashes):
                try:
                    analyzer = ClusterAnalyzer(store)
                    result = analyzer.cluster_weights(subset_hashes, config)
                    results.append(result)
                except Exception as e:
                    errors.append(e)
            
            # Run clustering concurrently
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for i, config in enumerate(configs):
                    # Each thread gets a subset of weights
                    start = i * 6
                    end = min(start + 8, len(hashes))
                    subset = hashes[start:end]
                    
                    future = executor.submit(cluster_subset, config, subset)
                    futures.append(future)
                
                # Wait for all to complete
                for future in futures:
                    future.result()
            
            # Should complete without errors
            assert len(errors) == 0
            assert len(results) == len(configs)
    
    def test_hierarchical_edge_cases(self):
        """Test edge cases in hierarchical clustering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "hierarchy.h5")
            
            # Create weights with clear hierarchical structure
            hashes = []
            
            # Single weight at top level
            single = WeightTensor(
                np.random.randn(64, 64).astype(np.float32),
                WeightMetadata(
                    name="single_top",
                    shape=(64, 64),
                    dtype=np.dtype("float32")
                )
            )
            hashes.append(store.save(single))
            
            # Empty hierarchy config
            empty_config = HierarchyConfig(
                levels=[],  # No levels
                merge_threshold=0.9,
            )
            
            hierarchy = ClusterHierarchy(empty_config)
            
            # Should handle empty configuration
            metrics = hierarchy.compute_metrics()
            assert metrics.total_levels == 0
            assert metrics.total_clusters == 0
            
            # Single level hierarchy
            single_level_config = HierarchyConfig(
                levels=[ClusterLevel.TENSOR],
                merge_threshold=0.9,
            )
            
            single_hierarchy = ClusterHierarchy(single_level_config)
            
            # Add single cluster
            cluster = ClusterInfo(
                cluster_id="single",
                centroid=Centroid(
                    cluster_id="single",
                    data=np.random.randn(64, 64).astype(np.float32),
                    shape=(64, 64),
                    dtype=np.dtype("float32"),
                ),
                member_hashes={hashes[0]},
                metrics=ClusterMetrics(
                    size=1,
                    avg_similarity=1.0,
                    compactness=1.0,
                    separation=0.0,
                ),
            )
            
            single_hierarchy.add_cluster(ClusterLevel.TENSOR, cluster)
            
            # Should handle single cluster
            metrics = single_hierarchy.compute_metrics()
            assert metrics.total_levels == 1
            assert metrics.total_clusters == 1
    
    def test_optimization_edge_cases(self):
        """Test edge cases in cluster optimization."""
        # Empty clusters
        optimizer = ClusterOptimizer(OptimizationConfig())
        optimized = optimizer.optimize_clusters([], {})
        assert len(optimized) == 0
        
        # Single cluster
        single_cluster = ClusterInfo(
            cluster_id="single",
            centroid=Centroid(
                cluster_id="single",
                data=np.random.randn(100).astype(np.float32),
                shape=(100,),
                dtype=np.dtype("float32"),
            ),
            member_hashes={"hash1"},
            metrics=ClusterMetrics(
                size=1,
                avg_similarity=1.0,
                compactness=1.0,
                separation=0.0,
            ),
        )
        
        optimized = optimizer.optimize_clusters([single_cluster], {"hash1": None})
        assert len(optimized) == 1
        
        # Clusters that can't be merged (different shapes)
        cluster1 = ClusterInfo(
            cluster_id="shape1",
            centroid=Centroid(
                cluster_id="shape1",
                data=np.random.randn(100).astype(np.float32),
                shape=(100,),
                dtype=np.dtype("float32"),
            ),
            member_hashes={"hash1", "hash2"},
            metrics=ClusterMetrics(
                size=2,
                avg_similarity=0.95,
                compactness=0.9,
                separation=0.8,
            ),
        )
        
        cluster2 = ClusterInfo(
            cluster_id="shape2",
            centroid=Centroid(
                cluster_id="shape2",
                data=np.random.randn(200).astype(np.float32),
                shape=(200,),  # Different shape
                dtype=np.dtype("float32"),
            ),
            member_hashes={"hash3", "hash4"},
            metrics=ClusterMetrics(
                size=2,
                avg_similarity=0.95,
                compactness=0.9,
                separation=0.8,
            ),
        )
        
        optimized = optimizer.optimize_clusters([cluster1, cluster2], {})
        assert len(optimized) == 2  # Can't merge different shapes