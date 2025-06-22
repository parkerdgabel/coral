"""Tests for the ClusterIndex component."""

import numpy as np
import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from coral.clustering.cluster_index import (
    ClusterIndex, WeightClusterInfo, ClusterCentroid, SimpleLSH
)


class TestWeightClusterInfo:
    """Test WeightClusterInfo data class."""
    
    def test_creation(self):
        """Test creating weight cluster info."""
        weight_vector = np.array([1.0, 2.0, 3.0])
        info = WeightClusterInfo(
            weight_hash="hash123",
            cluster_id="cluster1",
            weight_vector=weight_vector,
            distance_to_centroid=0.5
        )
        
        assert info.weight_hash == "hash123"
        assert info.cluster_id == "cluster1"
        np.testing.assert_array_equal(info.weight_vector, weight_vector)
        assert info.distance_to_centroid == 0.5
    
    def test_serialization(self):
        """Test to_dict and from_dict."""
        weight_vector = np.array([1.0, 2.0, 3.0])
        info = WeightClusterInfo(
            weight_hash="hash123",
            cluster_id="cluster1", 
            weight_vector=weight_vector,
            distance_to_centroid=0.5
        )
        
        # Serialize
        data = info.to_dict()
        assert isinstance(data, dict)
        assert data["weight_hash"] == "hash123"
        assert data["cluster_id"] == "cluster1"
        assert data["weight_vector"] == [1.0, 2.0, 3.0]
        
        # Deserialize
        info2 = WeightClusterInfo.from_dict(data)
        assert info2.weight_hash == info.weight_hash
        assert info2.cluster_id == info.cluster_id
        np.testing.assert_array_equal(info2.weight_vector, info.weight_vector)


class TestClusterCentroid:
    """Test ClusterCentroid data class."""
    
    def test_creation(self):
        """Test creating cluster centroid."""
        centroid_vector = np.array([4.0, 5.0, 6.0])
        centroid = ClusterCentroid(
            cluster_id="cluster1",
            centroid_vector=centroid_vector,
            member_count=10
        )
        
        assert centroid.cluster_id == "cluster1"
        np.testing.assert_array_equal(centroid.centroid_vector, centroid_vector)
        assert centroid.member_count == 10
    
    def test_serialization(self):
        """Test to_dict and from_dict."""
        centroid_vector = np.array([4.0, 5.0, 6.0])
        centroid = ClusterCentroid(
            cluster_id="cluster1",
            centroid_vector=centroid_vector,
            member_count=10
        )
        
        # Serialize
        data = centroid.to_dict()
        assert data["cluster_id"] == "cluster1"
        assert data["centroid_vector"] == [4.0, 5.0, 6.0]
        assert data["member_count"] == 10
        
        # Deserialize
        centroid2 = ClusterCentroid.from_dict(data)
        assert centroid2.cluster_id == centroid.cluster_id
        np.testing.assert_array_equal(centroid2.centroid_vector, centroid.centroid_vector)
        assert centroid2.member_count == centroid.member_count


class TestSimpleLSH:
    """Test SimpleLSH implementation."""
    
    def test_fit_and_query(self):
        """Test fitting LSH and querying neighbors."""
        # Create test data
        np.random.seed(42)
        data = np.random.randn(100, 50)
        ids = [f"point_{i}" for i in range(100)]
        
        # Fit LSH
        lsh = SimpleLSH(n_projections=10, seed=42)
        lsh.fit(data, ids)
        
        # Query with a point from the dataset
        query_point = data[0]
        indices, distances = lsh.query(query_point, k=5)
        
        assert len(indices) <= 5  # May return fewer if not enough candidates
        assert len(distances) == len(indices)
        assert indices[0] == 0  # Should find itself as nearest
        assert distances[0] < 0.001  # Distance to itself should be ~0
    
    def test_query_unseen_point(self):
        """Test querying with a new point."""
        np.random.seed(42)
        data = np.random.randn(50, 20)
        ids = [f"point_{i}" for i in range(50)]
        
        lsh = SimpleLSH(n_projections=5)
        lsh.fit(data, ids)
        
        # Query with new point
        query_point = np.random.randn(20)
        indices, distances = lsh.query(query_point, k=3)
        
        assert len(indices) == 3
        assert len(distances) == 3
        assert all(0 <= idx < 50 for idx in indices)
    
    def test_empty_buckets(self):
        """Test handling of queries that hit empty buckets."""
        # Create sparse data that will likely have empty buckets
        data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        ids = ["a", "b", "c"]
        
        lsh = SimpleLSH(n_projections=5)
        lsh.fit(data, ids)
        
        # Query with a very different point
        query_point = np.array([-1, -1, -1])
        indices, distances = lsh.query(query_point, k=2)
        
        assert len(indices) == 2
        assert len(distances) == 2


class TestClusterIndex:
    """Test the main ClusterIndex class."""
    
    def test_initialization(self):
        """Test index initialization."""
        index = ClusterIndex(dimension_threshold=500)
        assert index.get_cluster_count() == 0
        assert index.get_weight_count() == 0
    
    def test_add_weight_to_cluster(self):
        """Test adding weights to clusters."""
        index = ClusterIndex()
        
        # Add first weight
        weight1 = np.array([1.0, 2.0, 3.0])
        index.add_weight_to_cluster("w1", "c1", weight1)
        
        assert index.get_weight_count() == 1
        assert index.get_cluster_count() == 1
        assert index.get_cluster_members("c1") == ["w1"]
        
        # Add second weight to same cluster
        weight2 = np.array([1.1, 2.1, 3.1])
        index.add_weight_to_cluster("w2", "c1", weight2)
        
        assert index.get_weight_count() == 2
        assert index.get_cluster_count() == 1
        assert set(index.get_cluster_members("c1")) == {"w1", "w2"}
        
        # Add weight to different cluster
        weight3 = np.array([4.0, 5.0, 6.0])
        index.add_weight_to_cluster("w3", "c2", weight3)
        
        assert index.get_weight_count() == 3
        assert index.get_cluster_count() == 2
    
    def test_add_duplicate_weight_error(self):
        """Test error when adding weight to different cluster."""
        index = ClusterIndex()
        
        weight = np.array([1.0, 2.0, 3.0])
        index.add_weight_to_cluster("w1", "c1", weight)
        
        # Try to add same weight to different cluster
        with pytest.raises(ValueError, match="already assigned"):
            index.add_weight_to_cluster("w1", "c2", weight)
    
    def test_remove_weight_from_cluster(self):
        """Test removing weights from clusters."""
        index = ClusterIndex()
        
        # Add weights
        index.add_weight_to_cluster("w1", "c1", np.array([1.0, 2.0]))
        index.add_weight_to_cluster("w2", "c1", np.array([1.1, 2.1]))
        index.add_weight_to_cluster("w3", "c2", np.array([3.0, 4.0]))
        
        # Remove weight
        assert index.remove_weight_from_cluster("w1") is True
        assert index.get_weight_count() == 2
        assert index.get_cluster_members("c1") == ["w2"]
        
        # Remove non-existent weight
        assert index.remove_weight_from_cluster("w_unknown") is False
        
        # Remove last weight from cluster (should remove cluster)
        assert index.remove_weight_from_cluster("w2") is True
        assert index.get_cluster_count() == 1
        assert index.get_cluster_members("c1") == []
    
    def test_find_nearest_cluster_empty(self):
        """Test finding nearest cluster with no clusters."""
        index = ClusterIndex()
        weight = np.array([1.0, 2.0, 3.0])
        
        result = index.find_nearest_cluster(weight, k=1)
        assert result == []
    
    def test_find_nearest_cluster_single(self):
        """Test finding nearest cluster with one cluster."""
        index = ClusterIndex()
        
        # Add weights to create cluster
        index.add_weight_to_cluster("w1", "c1", np.array([1.0, 2.0, 3.0]))
        index.add_weight_to_cluster("w2", "c1", np.array([1.1, 2.1, 3.1]))
        
        # Query
        query = np.array([1.05, 2.05, 3.05])
        result = index.find_nearest_cluster(query, k=1)
        
        assert len(result) == 1
        assert result[0][0] == "c1"
        assert result[0][1] >= 0  # Distance (can be 0 if query is at centroid)
    
    def test_find_nearest_cluster_multiple(self):
        """Test finding multiple nearest clusters."""
        index = ClusterIndex()
        
        # Create multiple clusters
        index.add_weight_to_cluster("w1", "c1", np.array([1.0, 0.0]))
        index.add_weight_to_cluster("w2", "c2", np.array([0.0, 1.0]))
        index.add_weight_to_cluster("w3", "c3", np.array([-1.0, 0.0]))
        
        # Query for 2 nearest
        query = np.array([0.5, 0.5])
        result = index.find_nearest_cluster(query, k=2)
        
        assert len(result) == 2
        assert all(isinstance(r[0], str) for r in result)
        assert all(isinstance(r[1], float) for r in result)
        assert result[0][1] <= result[1][1]  # Sorted by distance
    
    def test_get_cluster_members(self):
        """Test getting cluster members."""
        index = ClusterIndex()
        
        # Add weights
        index.add_weight_to_cluster("w1", "c1", np.array([1.0]))
        index.add_weight_to_cluster("w2", "c1", np.array([2.0]))
        index.add_weight_to_cluster("w3", "c2", np.array([3.0]))
        
        assert set(index.get_cluster_members("c1")) == {"w1", "w2"}
        assert index.get_cluster_members("c2") == ["w3"]
        assert index.get_cluster_members("c_unknown") == []
    
    def test_rebalance_by_size(self):
        """Test rebalancing clusters by size."""
        index = ClusterIndex()
        
        # Create imbalanced clusters
        # Cluster 1: 6 weights
        for i in range(6):
            index.add_weight_to_cluster(f"w1_{i}", "c1", np.random.randn(5))
        
        # Cluster 2: 1 weight
        index.add_weight_to_cluster("w2_0", "c2", np.random.randn(5))
        
        # Cluster 3: 1 weight  
        index.add_weight_to_cluster("w3_0", "c3", np.random.randn(5))
        
        initial_sizes = index.get_cluster_sizes()
        assert initial_sizes["c1"] == 6
        assert initial_sizes["c2"] == 1
        assert initial_sizes["c3"] == 1
        
        # Rebalance
        reassignments = index.rebalance_clusters(strategy="size")
        
        # Check that some weights were reassigned
        assert len(reassignments) > 0
        
        # Check new sizes are more balanced
        new_sizes = index.get_cluster_sizes()
        size_variance = np.var(list(new_sizes.values()))
        initial_variance = np.var(list(initial_sizes.values()))
        assert size_variance < initial_variance
    
    def test_rebalance_by_distance(self):
        """Test rebalancing clusters by distance."""
        index = ClusterIndex()
        
        # Create clusters with misassigned weights
        # Cluster 1 centroid around [1, 0]
        index.add_weight_to_cluster("w1", "c1", np.array([1.0, 0.0]))
        index.add_weight_to_cluster("w2", "c1", np.array([1.1, 0.1]))
        
        # Cluster 2 centroid around [0, 1]
        index.add_weight_to_cluster("w3", "c2", np.array([0.0, 1.0]))
        index.add_weight_to_cluster("w4", "c2", np.array([0.1, 1.1]))
        
        # Add misassigned weight (closer to c2 but in c1)
        index.add_weight_to_cluster("w_mis", "c1", np.array([0.2, 0.9]))
        
        # Rebalance by distance
        reassignments = index.rebalance_clusters(strategy="distance")
        
        # The misassigned weight should be moved
        assert len(reassignments) > 0
    
    def test_high_dimensional_data(self):
        """Test with high-dimensional data (triggers LSH)."""
        index = ClusterIndex(dimension_threshold=100)
        
        # Create high-dimensional weights
        dim = 200
        index.add_weight_to_cluster("w1", "c1", np.random.randn(dim))
        index.add_weight_to_cluster("w2", "c1", np.random.randn(dim))
        index.add_weight_to_cluster("w3", "c2", np.random.randn(dim))
        
        # Query should work with LSH
        query = np.random.randn(dim)
        result = index.find_nearest_cluster(query, k=1)
        
        assert len(result) == 1
        assert result[0][0] in ["c1", "c2"]
    
    def test_serialization(self):
        """Test index serialization and deserialization."""
        index = ClusterIndex()
        
        # Add some data
        index.add_weight_to_cluster("w1", "c1", np.array([1.0, 2.0]))
        index.add_weight_to_cluster("w2", "c1", np.array([1.1, 2.1]))
        index.add_weight_to_cluster("w3", "c2", np.array([3.0, 4.0]))
        
        # Serialize
        data = index.to_dict()
        assert "weight_to_cluster" in data
        assert "cluster_centroids" in data
        assert "dimension_threshold" in data
        
        # Deserialize
        index2 = ClusterIndex.from_dict(data)
        assert index2.get_weight_count() == 3
        assert index2.get_cluster_count() == 2
        assert set(index2.get_cluster_members("c1")) == {"w1", "w2"}
        assert index2.get_cluster_members("c2") == ["w3"]
    
    def test_thread_safety(self):
        """Test concurrent operations."""
        index = ClusterIndex()
        errors = []
        
        def add_weights(start_idx):
            try:
                for i in range(10):
                    weight_id = f"w_{start_idx}_{i}"
                    cluster_id = f"c_{start_idx % 3}"
                    weight = np.random.randn(10)
                    index.add_weight_to_cluster(weight_id, cluster_id, weight)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent additions
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(add_weights, i*10) for i in range(5)]
            for future in futures:
                future.result()
        
        assert len(errors) == 0
        assert index.get_weight_count() == 50
        assert index.get_cluster_count() == 3
    
    def test_edge_cases(self):
        """Test various edge cases."""
        index = ClusterIndex()
        
        # Empty cluster members
        assert index.get_cluster_members("nonexistent") == []
        
        # Remove from empty index
        assert index.remove_weight_from_cluster("nonexistent") is False
        
        # Single weight cluster
        index.add_weight_to_cluster("w1", "c1", np.array([1.0]))
        assert index.get_cluster_count() == 1
        
        # Remove single weight (cluster should disappear)
        index.remove_weight_from_cluster("w1")
        assert index.get_cluster_count() == 0
        
        # Clear index
        index.add_weight_to_cluster("w1", "c1", np.array([1.0]))
        index.add_weight_to_cluster("w2", "c2", np.array([2.0]))
        index.clear()
        assert index.get_weight_count() == 0
        assert index.get_cluster_count() == 0
    
    def test_centroid_updates(self):
        """Test that centroids are properly updated."""
        index = ClusterIndex()
        
        # Add initial weights
        index.add_weight_to_cluster("w1", "c1", np.array([1.0, 0.0]))
        index.add_weight_to_cluster("w2", "c1", np.array([3.0, 0.0]))
        
        # Centroid should be at [2.0, 0.0]
        query = np.array([2.0, 0.0])
        result = index.find_nearest_cluster(query, k=1)
        assert result[0][0] == "c1"
        assert result[0][1] < 0.1  # Very close to centroid
        
        # Add another weight
        index.add_weight_to_cluster("w3", "c1", np.array([2.0, 0.0]))
        
        # Centroid should still be at [2.0, 0.0]
        result = index.find_nearest_cluster(query, k=1)
        assert result[0][1] < 0.1
        
        # Remove a weight
        index.remove_weight_from_cluster("w1")
        
        # Centroid should shift to [2.5, 0.0]
        new_query = np.array([2.5, 0.0])
        result = index.find_nearest_cluster(new_query, k=1)
        assert result[0][1] < 0.1
    
    def test_get_cluster_sizes(self):
        """Test getting cluster size information."""
        index = ClusterIndex()
        
        # Add weights to different clusters
        index.add_weight_to_cluster("w1", "c1", np.array([1.0]))
        index.add_weight_to_cluster("w2", "c1", np.array([2.0]))
        index.add_weight_to_cluster("w3", "c2", np.array([3.0]))
        index.add_weight_to_cluster("w4", "c2", np.array([4.0]))
        index.add_weight_to_cluster("w5", "c2", np.array([5.0]))
        
        sizes = index.get_cluster_sizes()
        assert sizes == {"c1": 2, "c2": 3}
    
    def test_rebalance_hybrid(self):
        """Test hybrid rebalancing strategy."""
        index = ClusterIndex()
        
        # Create imbalanced clusters with some misassigned weights
        # Large cluster
        for i in range(5):
            index.add_weight_to_cluster(f"w1_{i}", "c1", np.array([1.0 + i*0.1, 0.0]))
        
        # Small cluster with misassigned weight
        index.add_weight_to_cluster("w2_0", "c2", np.array([10.0, 0.0]))
        index.add_weight_to_cluster("w_mis", "c2", np.array([1.5, 0.0]))  # Should be in c1
        
        # Apply hybrid rebalancing
        reassignments = index.rebalance_clusters(strategy="hybrid")
        
        # Should have some reassignments
        assert len(reassignments) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])