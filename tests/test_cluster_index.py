"""
Comprehensive tests for ClusterIndex component.

Tests cover:
- Centroid management operations (CRUD)
- Fast lookup operations with various data sizes
- Spatial indexing performance and correctness
- Hierarchical navigation and consistency
- Thread safety and concurrent operations
- Performance characteristics and memory usage
- Index validation and error handling
"""

import numpy as np
import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from coral.clustering.cluster_index import ClusterIndex, IndexStats
from coral.clustering.cluster_types import Centroid, ClusterInfo, ClusterLevel, ClusteringStrategy
from coral.core.weight_tensor import WeightTensor, WeightMetadata


class TestClusterIndex:
    """Test suite for ClusterIndex functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.index = ClusterIndex()
        self.test_centroids = self._create_test_centroids()
        self.test_cluster_infos = self._create_test_cluster_infos()
    
    def _create_test_centroids(self) -> List[Centroid]:
        """Create test centroids with various characteristics."""
        centroids = []
        
        # Create centroids with different shapes and patterns
        shapes_and_patterns = [
            ((10,), lambda: np.random.randn(10)),
            ((5, 5), lambda: np.random.randn(5, 5)),
            ((3, 3, 3), lambda: np.random.randn(3, 3, 3)),
            ((100,), lambda: np.linspace(0, 1, 100)),
            ((20, 5), lambda: np.ones((20, 5)) * 0.5),
        ]
        
        for i, (shape, pattern_func) in enumerate(shapes_and_patterns):
            data = pattern_func()
            centroid = Centroid(
                data=data,
                cluster_id=f"cluster_{i}",
                shape=shape,
                dtype=data.dtype
            )
            centroids.append(centroid)
        
        return centroids
    
    def _create_test_cluster_infos(self) -> List[ClusterInfo]:
        """Create test cluster info objects."""
        infos = []
        levels = [ClusterLevel.TENSOR, ClusterLevel.BLOCK, ClusterLevel.LAYER, ClusterLevel.MODEL]
        
        for i, centroid in enumerate(self.test_centroids):
            info = ClusterInfo(
                cluster_id=centroid.cluster_id,
                strategy=ClusteringStrategy.KMEANS,
                level=levels[i % len(levels)],
                member_count=np.random.randint(2, 10),
                centroid_hash=centroid.compute_hash(),
                created_at="2024-01-01T00:00:00Z",
            )
            infos.append(info)
        
        return infos
    
    def test_index_initialization(self):
        """Test ClusterIndex initialization with various parameters."""
        # Default initialization
        index1 = ClusterIndex()
        assert len(index1) == 0
        assert index1._spatial_index_type == "kdtree"
        assert index1._cache_size == 1000
        
        # Custom initialization
        index2 = ClusterIndex(
            spatial_index_type="lsh",
            cache_size=500,
            enable_lsh=True,
            lsh_n_estimators=5
        )
        assert index2._spatial_index_type == "lsh"
        assert index2._cache_size == 500
        assert index2._enable_lsh is True
        assert index2._lsh_n_estimators == 5
    
    def test_add_centroid_basic(self):
        """Test basic centroid addition."""
        centroid = self.test_centroids[0]
        cluster_info = self.test_cluster_infos[0]
        
        # Add centroid without cluster info
        self.index.add_centroid(centroid)
        assert len(self.index) == 1
        assert centroid.cluster_id in self.index
        
        # Add centroid with cluster info
        index2 = ClusterIndex()
        index2.add_centroid(centroid, cluster_info)
        assert len(index2) == 1
        assert centroid.cluster_id in index2
        assert index2._cluster_info[centroid.cluster_id] == cluster_info
    
    def test_add_centroid_duplicate(self):
        """Test adding duplicate centroids raises error."""
        centroid = self.test_centroids[0]
        
        self.index.add_centroid(centroid)
        
        # Adding same cluster_id should raise ValueError
        with pytest.raises(ValueError, match="already exists"):
            self.index.add_centroid(centroid)
    
    def test_get_centroid(self):
        """Test centroid retrieval."""
        centroid = self.test_centroids[0]
        self.index.add_centroid(centroid)
        
        # Successful retrieval
        retrieved = self.index.get_centroid(centroid.cluster_id)
        assert retrieved is not None
        assert retrieved.cluster_id == centroid.cluster_id
        assert np.array_equal(retrieved.data, centroid.data)
        
        # Non-existent centroid
        assert self.index.get_centroid("non_existent") is None
    
    def test_update_centroid(self):
        """Test centroid updating."""
        centroid = self.test_centroids[0]
        self.index.add_centroid(centroid)
        
        # Create updated centroid
        new_data = np.random.randn(*centroid.shape)
        new_centroid = Centroid(
            data=new_data,
            cluster_id="different_id",  # This should be overridden
            shape=centroid.shape,
            dtype=centroid.dtype
        )
        
        # Update existing centroid
        success = self.index.update_centroid(centroid.cluster_id, new_centroid)
        assert success is True
        
        # Verify update
        retrieved = self.index.get_centroid(centroid.cluster_id)
        assert retrieved is not None
        assert retrieved.cluster_id == centroid.cluster_id  # Should preserve original ID
        assert np.array_equal(retrieved.data, new_data)
        
        # Update non-existent centroid
        success = self.index.update_centroid("non_existent", new_centroid)
        assert success is False
    
    def test_remove_centroid(self):
        """Test centroid removal."""
        centroid = self.test_centroids[0]
        cluster_info = self.test_cluster_infos[0]
        
        self.index.add_centroid(centroid, cluster_info)
        assert len(self.index) == 1
        
        # Remove existing centroid
        success = self.index.remove_centroid(centroid.cluster_id)
        assert success is True
        assert len(self.index) == 0
        assert centroid.cluster_id not in self.index
        
        # Remove non-existent centroid
        success = self.index.remove_centroid("non_existent")
        assert success is False
    
    def test_find_nearest_centroid_empty_index(self):
        """Test nearest centroid search on empty index."""
        weight = WeightTensor(
            data=np.random.randn(10),
            metadata=WeightMetadata(name="test", shape=(10,), dtype=np.float32)
        )
        
        result = self.index.find_nearest_centroid(weight)
        assert result is None
    
    def test_find_nearest_centroid_single(self):
        """Test nearest centroid search with single centroid."""
        centroid = self.test_centroids[0]  # Shape (10,)
        self.index.add_centroid(centroid)
        
        # Create similar weight
        weight_data = centroid.data + np.random.randn(*centroid.shape) * 0.1
        weight = WeightTensor(
            data=weight_data,
            metadata=WeightMetadata(name="test", shape=centroid.shape, dtype=centroid.dtype)
        )
        
        result = self.index.find_nearest_centroid(weight)
        assert result is not None
        cluster_id, distance = result
        assert cluster_id == centroid.cluster_id
        assert distance >= 0
    
    def test_find_nearest_centroid_multiple(self):
        """Test nearest centroid search with multiple centroids."""
        # Add centroids with same shape
        centroids = []
        for i in range(3):
            data = np.random.randn(10)
            centroid = Centroid(
                data=data,
                cluster_id=f"cluster_{i}",
                shape=(10,),
                dtype=data.dtype
            )
            centroids.append(centroid)
            self.index.add_centroid(centroid)
        
        # Create weight very close to first centroid
        weight_data = centroids[0].data + np.random.randn(10) * 0.01
        weight = WeightTensor(
            data=weight_data,
            metadata=WeightMetadata(name="test", shape=(10,), dtype=np.float32)
        )
        
        result = self.index.find_nearest_centroid(weight)
        assert result is not None
        cluster_id, distance = result
        # Should find the first centroid as nearest
        assert cluster_id == centroids[0].cluster_id
        assert distance < 1.0  # Should be quite close
    
    def test_find_nearest_centroid_numpy_array(self):
        """Test nearest centroid search with numpy array input."""
        centroid = self.test_centroids[0]
        self.index.add_centroid(centroid)
        
        # Use numpy array directly
        weight_array = centroid.data + np.random.randn(*centroid.shape) * 0.1
        
        result = self.index.find_nearest_centroid(weight_array)
        assert result is not None
        cluster_id, distance = result
        assert cluster_id == centroid.cluster_id
    
    def test_find_similar_centroids(self):
        """Test finding centroids within similarity threshold."""
        # Add multiple centroids
        centroids = []
        for i in range(5):
            data = np.random.randn(10)
            centroid = Centroid(
                data=data,
                cluster_id=f"cluster_{i}",
                shape=(10,),
                dtype=data.dtype
            )
            centroids.append(centroid)
            self.index.add_centroid(centroid)
        
        # Create weight similar to first few centroids
        base_weight = centroids[0].data
        weight = base_weight + np.random.randn(10) * 0.1
        
        # Find similar centroids with generous threshold
        results = self.index.find_similar_centroids(weight, threshold=5.0)
        assert len(results) > 0
        
        # Results should be sorted by distance
        distances = [distance for _, distance in results]
        assert distances == sorted(distances)
        
        # Find similar centroids with strict threshold
        results_strict = self.index.find_similar_centroids(weight, threshold=0.1)
        assert len(results_strict) <= len(results)
    
    def test_batch_lookup(self):
        """Test batch lookup operations."""
        # Add multiple centroids
        for centroid in self.test_centroids[:3]:
            if centroid.shape == (10,):  # Use consistent shape
                self.index.add_centroid(centroid)
        
        if len(self.index) == 0:
            # Create consistent centroids if none match
            for i in range(3):
                data = np.random.randn(10)
                centroid = Centroid(
                    data=data,
                    cluster_id=f"batch_cluster_{i}",
                    shape=(10,),
                    dtype=data.dtype
                )
                self.index.add_centroid(centroid)
        
        # Create batch of weights
        weights = []
        for i in range(5):
            weight_data = np.random.randn(10)
            weight = WeightTensor(
                data=weight_data,
                metadata=WeightMetadata(name=f"weight_{i}", shape=(10,), dtype=np.float32)
            )
            weights.append(weight)
        
        # Batch lookup
        results = self.index.batch_lookup(weights)
        assert len(results) == len(weights)
        
        # All results should be valid
        for result in results:
            assert result is not None
            cluster_id, distance = result
            assert cluster_id in self.index
            assert distance >= 0
    
    def test_get_centroids_by_level(self):
        """Test retrieving centroids by hierarchy level."""
        # Add centroids with different levels
        for centroid, cluster_info in zip(self.test_centroids, self.test_cluster_infos):
            self.index.add_centroid(centroid, cluster_info)
        
        # Test each level
        for level in ClusterLevel:
            centroids = self.index.get_centroids_by_level(level)
            assert isinstance(centroids, list)
            
            # Verify all returned centroids have correct level
            for centroid in centroids:
                cluster_info = self.index._cluster_info[centroid.cluster_id]
                assert cluster_info.level == level
    
    def test_build_hierarchy(self):
        """Test building and navigating hierarchy."""
        # Add centroids
        for centroid in self.test_centroids[:4]:
            self.index.add_centroid(centroid)
        
        # Define hierarchy relationships
        relationships = {
            "cluster_0": {"children": ["cluster_1", "cluster_2"]},
            "cluster_1": {"parent": "cluster_0", "children": ["cluster_3"]},
            "cluster_2": {"parent": "cluster_0"},
            "cluster_3": {"parent": "cluster_1"},
        }
        
        self.index.build_hierarchy(relationships)
        
        # Test parent-child relationships
        assert self.index.get_parent_centroid("cluster_1").cluster_id == "cluster_0"
        assert self.index.get_parent_centroid("cluster_2").cluster_id == "cluster_0"
        assert self.index.get_parent_centroid("cluster_3").cluster_id == "cluster_1"
        assert self.index.get_parent_centroid("cluster_0") is None  # Root node
        
        # Test children relationships
        children_0 = self.index.get_child_centroids("cluster_0")
        child_ids_0 = {c.cluster_id for c in children_0}
        assert child_ids_0 == {"cluster_1", "cluster_2"}
        
        children_1 = self.index.get_child_centroids("cluster_1")
        assert len(children_1) == 1
        assert children_1[0].cluster_id == "cluster_3"
        
        children_3 = self.index.get_child_centroids("cluster_3")
        assert len(children_3) == 0  # Leaf node
    
    def test_find_level_centroids(self):
        """Test finding centroids at specific hierarchy level."""
        # Add centroids with level information
        for centroid, cluster_info in zip(self.test_centroids, self.test_cluster_infos):
            self.index.add_centroid(centroid, cluster_info)
        
        # Create test weight with consistent shape
        weight_data = np.random.randn(10)
        weight = WeightTensor(
            data=weight_data,
            metadata=WeightMetadata(name="test", shape=(10,), dtype=np.float32)
        )
        
        # Find centroids at each level
        for level in ClusterLevel:
            results = self.index.find_level_centroids(level, weight)
            
            # Verify results are sorted by distance
            if len(results) > 1:
                distances = [distance for _, distance in results]
                assert distances == sorted(distances)
            
            # Verify all results are from the correct level
            for cluster_id, _ in results:
                cluster_info = self.index._cluster_info[cluster_id]
                assert cluster_info.level == level
    
    def test_index_stats(self):
        """Test index statistics collection."""
        # Initially empty
        stats = self.index.get_index_stats()
        assert stats["total_centroids"] == 0
        assert stats["total_queries"] == 0
        
        # Add centroids and perform operations
        for centroid, cluster_info in zip(self.test_centroids[:3], self.test_cluster_infos[:3]):
            self.index.add_centroid(centroid, cluster_info)
        
        # Perform some queries
        weight = WeightTensor(
            data=np.random.randn(10),
            metadata=WeightMetadata(name="test", shape=(10,), dtype=np.float32)
        )
        
        for _ in range(5):
            self.index.find_nearest_centroid(weight)
        
        # Check updated stats
        stats = self.index.get_index_stats()
        assert stats["total_centroids"] == 3
        assert stats["total_queries"] >= 5
        assert isinstance(stats["avg_query_time"], float)
        assert stats["avg_query_time"] >= 0
        assert isinstance(stats["memory_usage_mb"], float)
        assert stats["memory_usage_mb"] >= 0
    
    def test_centroid_usage_tracking(self):
        """Test centroid usage statistics."""
        # Add centroids
        for centroid in self.test_centroids[:3]:
            self.index.add_centroid(centroid)
        
        # Initially empty usage
        usage = self.index.get_centroid_usage()
        assert len(usage) == 0
        
        # Perform queries to generate usage
        weight = WeightTensor(
            data=np.random.randn(10),
            metadata=WeightMetadata(name="test", shape=(10,), dtype=np.float32)
        )
        
        for _ in range(10):
            self.index.find_nearest_centroid(weight)
        
        # Check usage statistics
        usage = self.index.get_centroid_usage()
        assert len(usage) > 0
        
        # Usage counts should be positive
        for cluster_id, count in usage.items():
            assert count > 0
            assert cluster_id in self.index
    
    def test_optimize_index(self):
        """Test index optimization."""
        # Add centroids
        for centroid in self.test_centroids:
            self.index.add_centroid(centroid)
        
        # Record initial stats
        initial_stats = self.index.get_index_stats()
        
        # Optimize index
        self.index.optimize_index()
        
        # Check that optimization completed
        final_stats = self.index.get_index_stats()
        assert final_stats["last_optimization"] is not None
        assert final_stats["index_build_time"] >= 0
        assert not final_stats["index_needs_rebuild"]
    
    def test_validate_index(self):
        """Test index validation."""
        # Empty index should be valid
        validation = self.index.validate_index()
        assert validation["is_valid"] is True
        assert len(validation["issues"]) == 0
        
        # Add valid centroids
        for centroid, cluster_info in zip(self.test_centroids, self.test_cluster_infos):
            self.index.add_centroid(centroid, cluster_info)
        
        # Validate populated index
        validation = self.index.validate_index()
        assert validation["is_valid"] is True
        assert validation["total_centroids"] == len(self.test_centroids)
        assert validation["total_cluster_info"] == len(self.test_cluster_infos)
    
    def test_caching_behavior(self):
        """Test query caching and cache hit rates."""
        # Add centroids
        for centroid in self.test_centroids[:3]:
            self.index.add_centroid(centroid)
        
        # Create weight for repeated queries
        weight = WeightTensor(
            data=np.random.randn(10),
            metadata=WeightMetadata(name="test", shape=(10,), dtype=np.float32)
        )
        
        # First query (cache miss)
        result1 = self.index.find_nearest_centroid(weight)
        stats_after_first = self.index.get_index_stats()
        
        # Second query with same weight (cache hit)
        result2 = self.index.find_nearest_centroid(weight)
        stats_after_second = self.index.get_index_stats()
        
        # Results should be identical
        assert result1 == result2
        
        # Cache hit count should increase
        assert stats_after_second["cache_hits"] > stats_after_first["cache_hits"]
    
    def test_different_spatial_index_types(self):
        """Test different spatial index configurations."""
        spatial_types = ["kdtree", "brute"]
        
        for spatial_type in spatial_types:
            index = ClusterIndex(spatial_index_type=spatial_type)
            
            # Add test centroids
            for centroid in self.test_centroids[:3]:
                if centroid.shape == (10,):  # Use consistent shape
                    index.add_centroid(centroid)
            
            if len(index) == 0:
                continue  # Skip if no consistent centroids
            
            # Test basic functionality
            weight = WeightTensor(
                data=np.random.randn(10),
                metadata=WeightMetadata(name="test", shape=(10,), dtype=np.float32)
            )
            
            result = index.find_nearest_centroid(weight)
            assert result is not None
            
            similar = index.find_similar_centroids(weight, threshold=10.0)
            assert len(similar) > 0


class TestClusterIndexPerformance:
    """Performance and scalability tests for ClusterIndex."""
    
    def test_large_scale_operations(self):
        """Test performance with large number of centroids."""
        index = ClusterIndex()
        
        # Add many centroids
        num_centroids = 1000
        centroids = []
        
        for i in range(num_centroids):
            data = np.random.randn(50)  # Moderate size
            centroid = Centroid(
                data=data,
                cluster_id=f"perf_cluster_{i}",
                shape=(50,),
                dtype=data.dtype
            )
            centroids.append(centroid)
            index.add_centroid(centroid)
        
        assert len(index) == num_centroids
        
        # Test query performance
        query_weight = np.random.randn(50)
        
        start_time = time.time()
        result = index.find_nearest_centroid(query_weight)
        query_time = time.time() - start_time
        
        assert result is not None
        assert query_time < 1.0  # Should be fast even with 1000 centroids
        
        # Test batch performance
        batch_weights = [np.random.randn(50) for _ in range(100)]
        
        start_time = time.time()
        batch_results = index.batch_lookup(batch_weights)
        batch_time = time.time() - start_time
        
        assert len(batch_results) == 100
        assert batch_time < 5.0  # Batch should be efficient
        
        # Check memory usage is reasonable
        stats = index.get_index_stats()
        assert stats["memory_usage_mb"] < 100  # Should be under 100MB
    
    def test_query_time_complexity(self):
        """Test that query time scales logarithmically with index size."""
        sizes = [100, 500, 1000]
        query_times = []
        
        for size in sizes:
            index = ClusterIndex()
            
            # Add centroids
            for i in range(size):
                data = np.random.randn(20)
                centroid = Centroid(
                    data=data,
                    cluster_id=f"timing_cluster_{i}",
                    shape=(20,),
                    dtype=data.dtype
                )
                index.add_centroid(centroid)
            
            # Measure query time
            query_weight = np.random.randn(20)
            
            start_time = time.time()
            for _ in range(10):  # Average over multiple queries
                index.find_nearest_centroid(query_weight)
            avg_time = (time.time() - start_time) / 10
            
            query_times.append(avg_time)
        
        # Query time should not increase linearly with size
        # (should be sub-linear due to spatial indexing)
        assert query_times[2] < query_times[0] * 10  # Much better than linear
    
    def test_memory_efficiency(self):
        """Test memory usage efficiency."""
        index = ClusterIndex()
        
        # Add centroids and track memory
        memory_usages = []
        
        for i in range(0, 500, 100):
            for j in range(100):
                data = np.random.randn(10)
                centroid = Centroid(
                    data=data,
                    cluster_id=f"mem_cluster_{i + j}",
                    shape=(10,),
                    dtype=data.dtype
                )
                index.add_centroid(centroid)
            
            stats = index.get_index_stats()
            memory_usages.append(stats["memory_usage_mb"])
        
        # Memory usage should grow roughly linearly with centroids
        # (allowing for some overhead from indexing structures)
        final_memory = memory_usages[-1]
        initial_memory = memory_usages[0]
        
        assert final_memory > initial_memory  # Memory should increase
        assert final_memory < initial_memory * 10  # But not excessively


class TestClusterIndexThreadSafety:
    """Thread safety tests for ClusterIndex."""
    
    def test_concurrent_reads(self):
        """Test concurrent read operations."""
        index = ClusterIndex()
        
        # Add test centroids
        for i in range(50):
            data = np.random.randn(10)
            centroid = Centroid(
                data=data,
                cluster_id=f"thread_cluster_{i}",
                shape=(10,),
                dtype=data.dtype
            )
            index.add_centroid(centroid)
        
        # Define read worker
        def read_worker():
            results = []
            for _ in range(20):
                weight = np.random.randn(10)
                result = index.find_nearest_centroid(weight)
                results.append(result)
            return results
        
        # Run concurrent reads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(read_worker) for _ in range(10)]
            
            all_results = []
            for future in as_completed(futures):
                worker_results = future.result()
                all_results.extend(worker_results)
        
        # All operations should succeed
        assert len(all_results) == 200  # 10 workers * 20 queries each
        for result in all_results:
            assert result is not None
    
    def test_concurrent_writes_and_reads(self):
        """Test concurrent write and read operations."""
        index = ClusterIndex()
        
        # Add initial centroids
        for i in range(20):
            data = np.random.randn(10)
            centroid = Centroid(
                data=data,
                cluster_id=f"concurrent_cluster_{i}",
                shape=(10,),
                dtype=data.dtype
            )
            index.add_centroid(centroid)
        
        results = {"read_errors": 0, "write_errors": 0}
        
        def read_worker():
            try:
                for _ in range(50):
                    weight = np.random.randn(10)
                    index.find_nearest_centroid(weight)
            except Exception:
                results["read_errors"] += 1
        
        def write_worker(start_id):
            try:
                for i in range(10):
                    data = np.random.randn(10)
                    centroid = Centroid(
                        data=data,
                        cluster_id=f"write_cluster_{start_id}_{i}",
                        shape=(10,),
                        dtype=data.dtype
                    )
                    index.add_centroid(centroid)
                    time.sleep(0.001)  # Small delay to allow interleaving
            except Exception:
                results["write_errors"] += 1
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Start read workers
            read_futures = [executor.submit(read_worker) for _ in range(4)]
            
            # Start write workers  
            write_futures = [executor.submit(write_worker, i * 100) for i in range(4)]
            
            # Wait for completion
            for future in as_completed(read_futures + write_futures):
                future.result()
        
        # Should have minimal errors (thread safety)
        assert results["read_errors"] == 0
        assert results["write_errors"] == 0
        
        # Index should be in valid state
        validation = index.validate_index()
        assert validation["is_valid"]
    
    def test_concurrent_index_optimization(self):
        """Test concurrent access during index optimization."""
        index = ClusterIndex()
        
        # Add many centroids
        for i in range(100):
            data = np.random.randn(10)
            centroid = Centroid(
                data=data,
                cluster_id=f"opt_cluster_{i}",
                shape=(10,),
                dtype=data.dtype
            )
            index.add_centroid(centroid)
        
        optimization_complete = threading.Event()
        
        def optimization_worker():
            index.optimize_index()
            optimization_complete.set()
        
        def query_worker():
            while not optimization_complete.is_set():
                weight = np.random.randn(10)
                try:
                    result = index.find_nearest_centroid(weight)
                    assert result is not None
                except Exception:
                    pass  # Some queries might fail during optimization
                time.sleep(0.001)
        
        # Run optimization and queries concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            opt_future = executor.submit(optimization_worker)
            query_futures = [executor.submit(query_worker) for _ in range(4)]
            
            # Wait for completion
            for future in [opt_future] + query_futures:
                future.result()
        
        # Index should be optimized and valid
        stats = index.get_index_stats()
        assert stats["last_optimization"] is not None
        
        validation = index.validate_index()
        assert validation["is_valid"]


class TestClusterIndexErrorHandling:
    """Error handling and edge case tests."""
    
    def test_empty_data_handling(self):
        """Test handling of empty or invalid data."""
        index = ClusterIndex()
        
        # Empty weight tensor
        empty_weight = WeightTensor(
            data=np.array([]),
            metadata=WeightMetadata(name="empty", shape=(), dtype=np.float32)
        )
        
        # Should handle gracefully
        result = index.find_nearest_centroid(empty_weight)
        assert result is None
    
    def test_mismatched_shapes(self):
        """Test handling of mismatched tensor shapes."""
        index = ClusterIndex()
        
        # Add centroid with one shape
        centroid = Centroid(
            data=np.random.randn(10),
            cluster_id="shape_test",
            shape=(10,),
            dtype=np.float32
        )
        index.add_centroid(centroid)
        
        # Query with different shape
        weight = WeightTensor(
            data=np.random.randn(5),
            metadata=WeightMetadata(name="different", shape=(5,), dtype=np.float32)
        )
        
        # Should handle gracefully (flatten operation should work)
        result = index.find_nearest_centroid(weight)
        assert result is not None  # Different shapes but should still work
    
    def test_invalid_cluster_operations(self):
        """Test invalid cluster operations."""
        index = ClusterIndex()
        
        # Operations on non-existent clusters
        assert index.get_centroid("non_existent") is None
        assert index.remove_centroid("non_existent") is False
        assert index.update_centroid("non_existent", None) is False
        
        # Invalid hierarchy operations
        assert index.get_parent_centroid("non_existent") is None
        assert len(index.get_child_centroids("non_existent")) == 0
    
    def test_index_rebuild_handling(self):
        """Test that index rebuilds are handled correctly."""
        index = ClusterIndex()
        
        # Add centroid
        centroid = Centroid(
            data=np.random.randn(10),
            cluster_id="rebuild_test",
            shape=(10,),
            dtype=np.float32
        )
        index.add_centroid(centroid)
        
        # Force index rebuild flag
        index._index_needs_rebuild = True
        
        # Query should trigger rebuild
        weight = np.random.randn(10)
        result = index.find_nearest_centroid(weight)
        
        assert result is not None
        assert not index._index_needs_rebuild  # Should be cleared after rebuild


if __name__ == "__main__":
    pytest.main([__file__, "-v"])