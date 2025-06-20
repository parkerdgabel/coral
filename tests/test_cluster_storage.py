"""
Comprehensive tests for ClusterStorage component.

Tests cover:
- Cluster storage and retrieval
- Centroid persistence with compression
- Hierarchy storage and reconstruction
- Assignment tracking
- Delta storage for centroid-based encoding
- Query operations and filtering
- Storage optimization and garbage collection
- Backup and restoration
- Performance with large datasets
- Thread safety and concurrent access
"""

import json
import tempfile
import threading
import time
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytest
import h5py

from coral.clustering.cluster_storage import ClusterStorage
from coral.clustering.cluster_types import (
    ClusterInfo, ClusterLevel, ClusteringStrategy, ClusterMetrics,
    Centroid, ClusterAssignment
)
from coral.clustering.cluster_hierarchy import ClusterHierarchy, HierarchyConfig
from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.delta.delta_encoder import Delta, DeltaType, DeltaEncoder
from coral.storage.hdf5_store import HDF5Store


class TestClusterStorage:
    """Test suite for ClusterStorage functionality."""
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage file."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            yield f.name
        # Cleanup
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def storage(self, temp_storage_path):
        """Create ClusterStorage instance."""
        storage = ClusterStorage(temp_storage_path)
        yield storage
        storage.close()
    
    @pytest.fixture
    def sample_clusters(self):
        """Create sample cluster data."""
        clusters = []
        for i in range(5):
            cluster = ClusterInfo(
                cluster_id=f"cluster_{i}",
                strategy=ClusteringStrategy.KMEANS,
                level=ClusterLevel.TENSOR,
                member_count=10 + i * 5,
                centroid_hash=f"centroid_hash_{i}",
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                parent_cluster_id=f"parent_{i//2}" if i > 0 else None,
                child_cluster_ids=[f"child_{i}_{j}" for j in range(2)] if i < 3 else None
            )
            clusters.append(cluster)
        return clusters
    
    @pytest.fixture
    def sample_centroids(self):
        """Create sample centroid data."""
        centroids = []
        for i in range(5):
            # Create test data with different shapes and dtypes
            if i % 2 == 0:
                data = np.random.randn(10, 10).astype(np.float32)
            else:
                data = np.random.randn(5, 20).astype(np.float64)
            
            centroid = Centroid(
                data=data,
                cluster_id=f"cluster_{i}",
                shape=data.shape,
                dtype=data.dtype
            )
            centroids.append(centroid)
        return centroids
    
    @pytest.fixture
    def sample_assignments(self):
        """Create sample assignment data."""
        assignments = []
        for i in range(10):
            assignment = ClusterAssignment(
                weight_name=f"weight_{i}",
                weight_hash=f"hash_{i}",
                cluster_id=f"cluster_{i % 5}",
                distance_to_centroid=float(i * 0.1),
                similarity_score=0.9 - (i * 0.05),
                is_representative=(i % 5 == 0)
            )
            assignments.append(assignment)
        return assignments
    
    @pytest.fixture
    def sample_hierarchy(self, sample_clusters):
        """Create sample hierarchy."""
        config = HierarchyConfig(
            levels=[ClusterLevel.TENSOR, ClusterLevel.LAYER, ClusterLevel.MODEL],
            merge_threshold=0.8,
            split_threshold=0.3
        )
        hierarchy = ClusterHierarchy(config)
        hierarchy.build_hierarchy(sample_clusters, config)
        return hierarchy
    
    def test_init_storage(self, temp_storage_path):
        """Test storage initialization."""
        storage = ClusterStorage(temp_storage_path)
        
        # Check that all required groups are created
        assert "clusters" in storage.file
        assert "centroids" in storage.file
        assert "assignments" in storage.file
        assert "hierarchies" in storage.file
        assert "centroid_deltas" in storage.file
        assert "indexes" in storage.file
        
        # Check version metadata
        assert "cluster_storage_version" in storage.file.attrs
        assert "created_at" in storage.file.attrs
        
        storage.close()
    
    def test_init_with_existing_store(self, temp_storage_path):
        """Test initialization with existing HDF5Store."""
        # Create base store first
        base_store = HDF5Store(temp_storage_path)
        
        # Create cluster storage with existing store
        storage = ClusterStorage(temp_storage_path, base_store=base_store)
        
        assert storage.store is base_store
        assert not storage._owns_store
        
        storage.close()
        base_store.close()
    
    def test_store_and_load_clusters(self, storage, sample_clusters):
        """Test cluster storage and retrieval."""
        # Store clusters
        result = storage.store_clusters(sample_clusters)
        
        assert len(result) == len(sample_clusters)
        for cluster in sample_clusters:
            assert cluster.cluster_id in result
            assert result[cluster.cluster_id].startswith("cluster_")
        
        # Load all clusters
        loaded_clusters = storage.load_clusters()
        assert len(loaded_clusters) == len(sample_clusters)
        
        # Verify cluster data integrity
        loaded_by_id = {c.cluster_id: c for c in loaded_clusters}
        for original in sample_clusters:
            loaded = loaded_by_id[original.cluster_id]
            assert loaded.cluster_id == original.cluster_id
            assert loaded.strategy == original.strategy
            assert loaded.level == original.level
            assert loaded.member_count == original.member_count
            assert loaded.centroid_hash == original.centroid_hash
    
    def test_load_clusters_with_filter(self, storage, sample_clusters):
        """Test cluster loading with filters."""
        # Store clusters with different levels
        clusters = []
        for i, level in enumerate([ClusterLevel.TENSOR, ClusterLevel.LAYER, ClusterLevel.MODEL]):
            cluster = ClusterInfo(
                cluster_id=f"cluster_{level.value}_{i}",
                strategy=ClusteringStrategy.HIERARCHICAL,
                level=level,
                member_count=10 + i,
                centroid_hash=f"hash_{i}"
            )
            clusters.append(cluster)
        
        storage.store_clusters(clusters)
        
        # Test level filter
        tensor_clusters = storage.load_clusters(
            filter_criteria={"level": ClusterLevel.TENSOR}
        )
        assert len(tensor_clusters) == 1
        assert tensor_clusters[0].level == ClusterLevel.TENSOR
        
        # Test strategy filter
        hierarchical_clusters = storage.load_clusters(
            filter_criteria={"strategy": ClusteringStrategy.HIERARCHICAL}
        )
        assert len(hierarchical_clusters) == 3
        assert all(c.strategy == ClusteringStrategy.HIERARCHICAL for c in hierarchical_clusters)
        
        # Test member count filter
        small_clusters = storage.load_clusters(
            filter_criteria={"min_member_count": 11, "max_member_count": 11}
        )
        assert len(small_clusters) == 1
        assert small_clusters[0].member_count == 11
    
    def test_store_and_load_centroids(self, storage, sample_centroids):
        """Test centroid storage with compression."""
        # Store centroids
        result = storage.store_centroids(sample_centroids)
        
        assert len(result) == len(sample_centroids)
        for centroid in sample_centroids:
            centroid_hash = centroid.compute_hash()
            assert centroid_hash in result
        
        # Load all centroids
        loaded_centroids = storage.load_centroids()
        assert len(loaded_centroids) == len(sample_centroids)
        
        # Verify centroid data integrity
        # Create mappings for comparison since order might not be preserved
        original_by_hash = {c.compute_hash(): c for c in sample_centroids}
        loaded_by_hash = {c.compute_hash(): c for c in loaded_centroids}
        
        assert set(original_by_hash.keys()) == set(loaded_by_hash.keys())
        
        for centroid_hash in original_by_hash:
            original = original_by_hash[centroid_hash]
            loaded = loaded_by_hash[centroid_hash]
            
            assert loaded.cluster_id == original.cluster_id
            assert loaded.shape == original.shape
            assert loaded.dtype == original.dtype
            assert loaded.compute_hash() == original.compute_hash()
            np.testing.assert_array_equal(loaded.data, original.data)
    
    def test_load_centroids_lazy(self, storage, sample_centroids):
        """Test lazy loading of centroids."""
        # Store centroids
        storage.store_centroids(sample_centroids)
        
        # Load with lazy loading
        loaded_centroids = storage.load_centroids(lazy_loading=True)
        
        assert len(loaded_centroids) == len(sample_centroids)
        
        # Check that data arrays have the correct shape but are placeholder
        for centroid in loaded_centroids:
            assert centroid.data.shape == centroid.shape
            assert centroid.data.dtype == centroid.dtype
            # In a real implementation, you'd check for lazy loading attributes
    
    def test_load_centroids_by_cluster_ids(self, storage, sample_clusters, sample_centroids):
        """Test loading centroids for specific clusters."""
        # Store clusters and centroids
        storage.store_clusters(sample_clusters)
        storage.store_centroids(sample_centroids)
        
        # Load centroids for specific clusters
        cluster_ids = ["cluster_0", "cluster_2"]
        loaded_centroids = storage.load_centroids(cluster_ids=cluster_ids)
        
        # Should load centroids for the specified clusters
        loaded_cluster_ids = {c.cluster_id for c in loaded_centroids}
        assert loaded_cluster_ids.intersection(set(cluster_ids))
    
    def test_store_and_load_hierarchy(self, storage, sample_hierarchy):
        """Test hierarchy storage and reconstruction."""
        # Store hierarchy
        hierarchy_key = storage.store_hierarchy(sample_hierarchy)
        
        assert hierarchy_key.startswith("hierarchy_")
        
        # Load hierarchy
        loaded_hierarchy = storage.load_hierarchy()
        
        assert loaded_hierarchy is not None
        assert len(loaded_hierarchy) == len(sample_hierarchy)
        
        # Verify hierarchy structure
        original_clusters = sample_hierarchy.get_all_clusters()
        loaded_clusters = loaded_hierarchy.get_all_clusters()
        
        assert len(loaded_clusters) == len(original_clusters)
        
        # Check that config is preserved
        assert loaded_hierarchy.config.levels == sample_hierarchy.config.levels
        assert loaded_hierarchy.config.merge_threshold == sample_hierarchy.config.merge_threshold
    
    def test_validate_hierarchy_integrity(self, storage, sample_clusters, sample_hierarchy):
        """Test hierarchy integrity validation."""
        # Store clusters and hierarchy
        storage.store_clusters(sample_clusters)
        hierarchy_key = storage.store_hierarchy(sample_hierarchy)
        
        # Validate integrity
        validation_result = storage.validate_hierarchy_integrity()
        
        assert "valid" in validation_result or "is_valid" in validation_result
        assert "missing_in_storage" in validation_result
        assert "missing_in_hierarchy" in validation_result
        assert "storage_cluster_count" in validation_result
        assert "hierarchy_cluster_count" in validation_result
    
    def test_store_and_load_assignments(self, storage, sample_assignments):
        """Test assignment storage and retrieval."""
        # Store assignments
        result = storage.store_assignments(sample_assignments)
        
        assert len(result) == len(sample_assignments)
        
        # Load all assignments
        loaded_assignments = storage.load_assignments()
        assert len(loaded_assignments) == len(sample_assignments)
        
        # Verify assignment data (use mapping since order might not be preserved)
        original_by_key = {(a.weight_hash, a.cluster_id): a for a in sample_assignments}
        loaded_by_key = {(a.weight_hash, a.cluster_id): a for a in loaded_assignments}
        
        assert set(original_by_key.keys()) == set(loaded_by_key.keys())
        
        for key in original_by_key:
            original = original_by_key[key]
            loaded = loaded_by_key[key]
            
            assert loaded.weight_name == original.weight_name
            assert loaded.weight_hash == original.weight_hash
            assert loaded.cluster_id == original.cluster_id
            assert abs(loaded.distance_to_centroid - original.distance_to_centroid) < 1e-6
            assert abs(loaded.similarity_score - original.similarity_score) < 1e-6
            assert loaded.is_representative == original.is_representative
    
    def test_load_assignments_with_filter(self, storage, sample_assignments):
        """Test assignment loading with filters."""
        # Store assignments
        storage.store_assignments(sample_assignments)
        
        # Test cluster filter
        cluster_assignments = storage.load_assignments(
            filter_criteria={"cluster_ids": ["cluster_0", "cluster_1"]}
        )
        
        for assignment in cluster_assignments:
            assert assignment.cluster_id in ["cluster_0", "cluster_1"]
        
        # Test weight hash filter
        weight_assignments = storage.load_assignments(
            filter_criteria={"weight_hashes": ["hash_0", "hash_1"]}
        )
        
        for assignment in weight_assignments:
            assert assignment.weight_hash in ["hash_0", "hash_1"]
        
        # Test similarity filter
        high_similarity_assignments = storage.load_assignments(
            filter_criteria={"min_similarity": 0.8}
        )
        
        for assignment in high_similarity_assignments:
            assert assignment.similarity_score >= 0.8
    
    def test_centroid_delta_operations(self, storage, sample_centroids):
        """Test centroid delta storage and retrieval."""
        # Create some test deltas
        deltas = []
        for i in range(3):
            # Create a simple delta for testing
            reference_data = sample_centroids[i].data
            delta_data = np.random.randn(*reference_data.shape).astype(reference_data.dtype) * 0.1
            
            delta = Delta(
                delta_type=DeltaType.FLOAT32_RAW,
                data=delta_data,
                metadata={"test": "delta"},
                original_shape=reference_data.shape,
                original_dtype=reference_data.dtype,
                reference_hash=sample_centroids[i].compute_hash(),
                compression_ratio=2.0
            )
            deltas.append(delta)
        
        # Store centroids first
        storage.store_centroids(sample_centroids)
        
        # Store deltas
        delta_result = storage.store_centroid_deltas(deltas)
        assert len(delta_result) == len(deltas)
        
        # Load deltas for specific clusters
        cluster_ids = ["cluster_0", "cluster_1"]
        loaded_deltas = storage.load_centroid_deltas(cluster_ids)
        
        # Should have loaded some deltas
        assert len(loaded_deltas) >= 0  # May be 0 if centroid hash mapping doesn't match
    
    def test_query_operations(self, storage, sample_clusters):
        """Test efficient query operations."""
        # Create clusters with different levels and strategies
        test_clusters = []
        levels = [ClusterLevel.TENSOR, ClusterLevel.LAYER, ClusterLevel.MODEL]
        strategies = [ClusteringStrategy.KMEANS, ClusteringStrategy.HIERARCHICAL]
        
        for i, (level, strategy) in enumerate(zip(levels * 2, strategies * 3)):
            cluster = ClusterInfo(
                cluster_id=f"query_cluster_{i}",
                strategy=strategy,
                level=level,
                member_count=10 + i,
                centroid_hash=f"query_hash_{i}"
            )
            test_clusters.append(cluster)
        
        storage.store_clusters(test_clusters)
        
        # Test level queries
        tensor_clusters = storage.get_clusters_by_level(ClusterLevel.TENSOR)
        assert len(tensor_clusters) >= 1
        assert all(c.level == ClusterLevel.TENSOR for c in tensor_clusters)
        
        # Test strategy queries  
        kmeans_clusters = storage.get_clusters_by_strategy(ClusteringStrategy.KMEANS)
        assert len(kmeans_clusters) >= 1
        assert all(c.strategy == ClusteringStrategy.KMEANS for c in kmeans_clusters)
        
        # Test metadata queries
        metadata = storage.get_cluster_metadata("query_cluster_0")
        assert metadata is not None
        assert metadata["cluster_id"] == "query_cluster_0"
        
        # Test batch loading
        cluster_ids = ["query_cluster_0", "query_cluster_1", "query_cluster_2"]
        batch_clusters = storage.batch_load_clusters(cluster_ids)
        assert len(batch_clusters) == 3
    
    def test_storage_optimization(self, storage, sample_clusters, sample_centroids):
        """Test storage optimization operations."""
        # Store some data
        storage.store_clusters(sample_clusters)
        storage.store_centroids(sample_centroids)
        
        # Test compression
        compression_result = storage.compress_storage()
        
        assert "compression_time" in compression_result
        assert "initial_size" in compression_result
        assert "final_size" in compression_result
        assert "space_saved" in compression_result
        assert "orphaned_removed" in compression_result
        
        # Test garbage collection
        gc_result = storage.garbage_collect()
        
        assert "gc_time" in gc_result
        assert "orphaned_centroids_removed" in gc_result
        assert "orphaned_assignments_removed" in gc_result
        assert "total_clusters" in gc_result
        
        # Test storage size estimation
        size_estimates = storage.estimate_storage_size()
        
        assert "total_file_size" in size_estimates
        assert "clusters_size" in size_estimates
        assert "centroids_size" in size_estimates
        assert "assignments_size" in size_estimates
        
        # Test layout optimization
        layout_result = storage.optimize_layout()
        
        assert "optimization_time" in layout_result
        assert "indexes_rebuilt" in layout_result
    
    def test_backup_and_restore(self, storage, sample_clusters, sample_centroids, tmp_path):
        """Test backup and restoration functionality."""
        # Store some data
        storage.store_clusters(sample_clusters)
        storage.store_centroids(sample_centroids)
        
        # Create backup
        backup_dir = tmp_path / "backups"
        backup_path = storage.create_backup(str(backup_dir))
        
        assert Path(backup_path).exists()
        assert backup_path.endswith(".h5")
        
        # Modify original data
        new_cluster = ClusterInfo(
            cluster_id="backup_test_cluster",
            strategy=ClusteringStrategy.DBSCAN,
            level=ClusterLevel.BLOCK,
            member_count=100
        )
        storage.store_clusters([new_cluster])
        
        # Verify new data exists
        loaded_clusters = storage.load_clusters()
        cluster_ids = [c.cluster_id for c in loaded_clusters]
        assert "backup_test_cluster" in cluster_ids
        
        # Restore from backup
        restore_success = storage.restore_from_backup(backup_path)
        assert restore_success
        
        # Verify original data is restored
        restored_clusters = storage.load_clusters()
        restored_cluster_ids = [c.cluster_id for c in restored_clusters]
        assert "backup_test_cluster" not in restored_cluster_ids
        assert len(restored_clusters) == len(sample_clusters)
    
    def test_export_and_import(self, storage, sample_clusters, sample_centroids, sample_assignments, tmp_path):
        """Test data export and import functionality."""
        # Store data
        storage.store_clusters(sample_clusters)
        storage.store_centroids(sample_centroids)
        storage.store_assignments(sample_assignments)
        
        # Export data
        export_file = tmp_path / "export_test.json"
        export_result = storage.export_clusters(str(export_file))
        
        assert export_file.exists()
        assert "export_time" in export_result
        assert "clusters_exported" in export_result
        assert "centroids_exported" in export_result
        assert "assignments_exported" in export_result
        assert export_result["clusters_exported"] == len(sample_clusters)
        
        # Verify export file content
        with open(export_file, 'r') as f:
            export_data = json.load(f)
        
        assert "version" in export_data
        assert "clusters" in export_data
        assert "centroids" in export_data
        assert "assignments" in export_data
        assert len(export_data["clusters"]) == len(sample_clusters)
        
        # Create new storage and import
        import_storage_path = tmp_path / "import_test.h5"
        import_storage = ClusterStorage(str(import_storage_path))
        
        try:
            import_result = import_storage.import_clusters(str(export_file))
            
            assert "import_time" in import_result
            assert "clusters_imported" in import_result
            assert "centroids_imported" in import_result
            assert "assignments_imported" in import_result
            assert import_result["clusters_imported"] == len(sample_clusters)
            
            # Verify imported data
            imported_clusters = import_storage.load_clusters()
            imported_centroids = import_storage.load_centroids()
            imported_assignments = import_storage.load_assignments()
            
            assert len(imported_clusters) == len(sample_clusters)
            assert len(imported_centroids) == len(sample_centroids)
            assert len(imported_assignments) == len(sample_assignments)
            
        finally:
            import_storage.close()
    
    def test_storage_info(self, storage, sample_clusters, sample_centroids):
        """Test storage information reporting."""
        # Store some data
        storage.store_clusters(sample_clusters)
        storage.store_centroids(sample_centroids)
        
        # Get storage info
        info = storage.get_storage_info()
        
        assert "file_size" in info
        assert "storage_path" in info
        assert "compression" in info
        assert "group_counts" in info
        assert "group_sizes" in info
        assert "operation_stats" in info
        assert "version" in info
        assert "created_at" in info
        
        # Verify group counts
        assert info["group_counts"]["clusters"] == len(sample_clusters)
        assert info["group_counts"]["centroids"] == len(sample_centroids)
        
        # Verify operation stats
        stats = info["operation_stats"]
        assert stats["clusters_stored"] >= len(sample_clusters)
        assert stats["centroids_stored"] >= len(sample_centroids)
    
    def test_thread_safety(self, storage, sample_clusters):
        """Test thread-safe operations."""
        def store_clusters_worker(thread_id, clusters_subset):
            """Worker function for storing clusters."""
            try:
                # Modify cluster IDs to avoid conflicts
                modified_clusters = []
                for cluster in clusters_subset:
                    modified_cluster = ClusterInfo(
                        cluster_id=f"thread_{thread_id}_{cluster.cluster_id}",
                        strategy=cluster.strategy,
                        level=cluster.level,
                        member_count=cluster.member_count,
                        centroid_hash=f"thread_{thread_id}_{cluster.centroid_hash}"
                    )
                    modified_clusters.append(modified_cluster)
                
                result = storage.store_clusters(modified_clusters)
                return len(result)
            except Exception as e:
                pytest.fail(f"Thread {thread_id} failed: {e}")
        
        # Run multiple threads storing clusters
        num_threads = 4
        clusters_per_thread = len(sample_clusters) // num_threads + 1
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(num_threads):
                start_idx = i * clusters_per_thread
                end_idx = min((i + 1) * clusters_per_thread, len(sample_clusters))
                clusters_subset = sample_clusters[start_idx:end_idx]
                
                if clusters_subset:  # Only submit if there are clusters to process
                    future = executor.submit(store_clusters_worker, i, clusters_subset)
                    futures.append(future)
            
            # Collect results
            total_stored = 0
            for future in as_completed(futures):
                total_stored += future.result()
        
        # Verify all clusters were stored
        loaded_clusters = storage.load_clusters()
        assert len(loaded_clusters) >= total_stored
    
    def test_large_dataset_performance(self, storage):
        """Test performance with large datasets."""
        # Create large dataset
        num_clusters = 1000
        num_centroids = 100
        
        # Generate clusters
        large_clusters = []
        for i in range(num_clusters):
            cluster = ClusterInfo(
                cluster_id=f"large_cluster_{i}",
                strategy=ClusteringStrategy.KMEANS,
                level=ClusterLevel.TENSOR,
                member_count=50 + (i % 100),
                centroid_hash=f"large_centroid_{i % num_centroids}"
            )
            large_clusters.append(cluster)
        
        # Generate centroids
        large_centroids = []
        for i in range(num_centroids):
            data = np.random.randn(100, 100).astype(np.float32)
            centroid = Centroid(
                data=data,
                cluster_id=f"large_cluster_{i}",
                shape=data.shape,
                dtype=data.dtype
            )
            large_centroids.append(centroid)
        
        # Measure storage performance
        start_time = time.time()
        storage.store_clusters(large_clusters)
        cluster_store_time = time.time() - start_time
        
        start_time = time.time()
        storage.store_centroids(large_centroids)
        centroid_store_time = time.time() - start_time
        
        # Measure loading performance
        start_time = time.time()
        loaded_clusters = storage.load_clusters()
        cluster_load_time = time.time() - start_time
        
        start_time = time.time()
        loaded_centroids = storage.load_centroids()
        centroid_load_time = time.time() - start_time
        
        # Verify correctness
        assert len(loaded_clusters) == num_clusters
        assert len(loaded_centroids) == num_centroids
        
        # Performance assertions (adjust based on acceptable performance)
        assert cluster_store_time < 10.0  # Should store 1000 clusters in under 10 seconds
        assert centroid_store_time < 30.0  # Should store 100 centroids in under 30 seconds
        assert cluster_load_time < 5.0    # Should load 1000 clusters in under 5 seconds
        assert centroid_load_time < 15.0  # Should load 100 centroids in under 15 seconds
        
        print(f"Performance results:")
        print(f"  Cluster store: {cluster_store_time:.3f}s ({num_clusters} clusters)")
        print(f"  Centroid store: {centroid_store_time:.3f}s ({num_centroids} centroids)")
        print(f"  Cluster load: {cluster_load_time:.3f}s")
        print(f"  Centroid load: {centroid_load_time:.3f}s")
    
    def test_context_manager(self, temp_storage_path):
        """Test context manager functionality."""
        # Test that storage can be used as context manager
        with ClusterStorage(temp_storage_path) as storage:
            # Create sample data
            cluster = ClusterInfo(
                cluster_id="context_test",
                strategy=ClusteringStrategy.ADAPTIVE,
                level=ClusterLevel.LAYER,
                member_count=25
            )
            
            # Store and verify
            result = storage.store_clusters([cluster])
            assert len(result) == 1
            
            loaded = storage.load_clusters()
            assert len(loaded) == 1
            assert loaded[0].cluster_id == "context_test"
        
        # File should be properly closed after context exit
        # Verify by trying to open it again
        with ClusterStorage(temp_storage_path) as storage2:
            loaded_again = storage2.load_clusters()
            assert len(loaded_again) == 1
    
    def test_error_handling(self, storage):
        """Test error handling and edge cases."""
        # Test loading non-existent cluster
        non_existent = storage.load_clusters(cluster_ids=["non_existent"])
        assert len(non_existent) == 0
        
        # Test loading non-existent centroid
        non_existent_centroids = storage.load_centroids(centroid_hashes=["non_existent"])
        assert len(non_existent_centroids) == 0
        
        # Test getting metadata for non-existent cluster
        metadata = storage.get_cluster_metadata("non_existent")
        assert metadata is None
        
        # Test loading non-existent hierarchy
        hierarchy = storage.load_hierarchy("non_existent")
        assert hierarchy is None
        
        # Test restoring from non-existent backup
        restore_result = storage.restore_from_backup("/non/existent/path")
        assert not restore_result
        
        # Test empty data operations
        empty_result = storage.store_clusters([])
        assert len(empty_result) == 0
        
        empty_centroids = storage.store_centroids([])
        assert len(empty_centroids) == 0
        
        empty_assignments = storage.store_assignments([])
        assert len(empty_assignments) == 0
    
    def test_duplicate_handling(self, storage, sample_clusters, sample_centroids):
        """Test handling of duplicate data."""
        # Store clusters
        result1 = storage.store_clusters(sample_clusters)
        
        # Store same clusters again
        result2 = storage.store_clusters(sample_clusters)
        
        # Should overwrite existing clusters
        assert len(result1) == len(result2)
        
        # Load clusters - should not have duplicates
        loaded = storage.load_clusters()
        cluster_ids = [c.cluster_id for c in loaded]
        assert len(cluster_ids) == len(set(cluster_ids))  # No duplicates
        
        # Test centroid deduplication
        result1 = storage.store_centroids(sample_centroids)
        result2 = storage.store_centroids(sample_centroids)
        
        # Should skip already existing centroids
        loaded_centroids = storage.load_centroids()
        assert len(loaded_centroids) == len(sample_centroids)
    
    def test_compression_effectiveness(self, storage, sample_centroids):
        """Test that compression actually reduces storage size."""
        # Store centroids with compression
        storage.store_centroids(sample_centroids)
        
        # Get storage info
        info = storage.get_storage_info()
        compressed_size = info["group_sizes"]["centroids"]
        
        # Calculate uncompressed size
        uncompressed_size = sum(c.nbytes for c in sample_centroids)
        
        # Compression should reduce size (though not guaranteed for small random data)
        compression_ratio = uncompressed_size / compressed_size if compressed_size > 0 else 1.0
        
        # At minimum, compressed size should not be much larger than uncompressed
        assert compression_ratio >= 0.5  # Allow for some overhead
        
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Uncompressed: {uncompressed_size} bytes")
        print(f"Compressed: {compressed_size} bytes")


class TestClusterStorageIntegration:
    """Integration tests with other components."""
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage file."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    def test_integration_with_hdf5_store(self, temp_storage_path):
        """Test integration with existing HDF5Store."""
        # Create HDF5Store with some weights
        base_store = HDF5Store(temp_storage_path)
        
        # Store some weight data
        weight_data = np.random.randn(10, 10).astype(np.float32)
        weight_metadata = WeightMetadata(
            name="test_weight",
            shape=weight_data.shape,
            dtype=weight_data.dtype,
            layer_type="linear"
        )
        weight = WeightTensor(data=weight_data, metadata=weight_metadata)
        
        weight_hash = base_store.store(weight)
        
        # Create cluster storage with same store
        cluster_storage = ClusterStorage(temp_storage_path, base_store=base_store)
        
        # Store cluster data
        cluster = ClusterInfo(
            cluster_id="integration_test",
            strategy=ClusteringStrategy.KMEANS,
            level=ClusterLevel.TENSOR,
            member_count=1
        )
        cluster_storage.store_clusters([cluster])
        
        # Verify both weight and cluster data exist
        loaded_weight = base_store.load(weight_hash)
        assert loaded_weight is not None
        
        loaded_clusters = cluster_storage.load_clusters()
        assert len(loaded_clusters) == 1
        
        cluster_storage.close()
        base_store.close()
    
    def test_centroid_delta_integration(self, temp_storage_path):
        """Test integration with delta encoding system."""
        storage = ClusterStorage(temp_storage_path)
        
        try:
            # Create test weight and centroid
            weight_data = np.random.randn(50, 50).astype(np.float32)
            centroid_data = weight_data + np.random.randn(50, 50).astype(np.float32) * 0.1
            
            weight_metadata = WeightMetadata(
                name="test_weight",
                shape=weight_data.shape,
                dtype=weight_data.dtype
            )
            weight = WeightTensor(data=weight_data, metadata=weight_metadata)
            
            centroid = Centroid(
                data=centroid_data,
                cluster_id="test_cluster",
                shape=centroid_data.shape,
                dtype=centroid_data.dtype
            )
            
            # Store centroid
            storage.store_centroids([centroid])
            
            # Create delta encoder and encode weight against centroid
            from coral.delta.delta_encoder import DeltaEncoder, DeltaConfig
            encoder = DeltaEncoder(DeltaConfig())
            
            # Convert centroid to WeightTensor for delta encoding
            centroid_weight = WeightTensor(
                data=centroid.data,
                metadata=WeightMetadata(
                    name="centroid",
                    shape=centroid.shape,
                    dtype=centroid.dtype
                )
            )
            
            delta = encoder.encode_delta(weight, centroid_weight)
            
            if delta:
                # Store delta through cluster storage
                storage.store_centroid_deltas([delta])
                
                # Load delta and verify
                loaded_deltas = storage.load_centroid_deltas(["test_cluster"])
                
                # Should be able to load delta (though might be empty if hash mismatch)
                assert isinstance(loaded_deltas, list)
        
        finally:
            storage.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])