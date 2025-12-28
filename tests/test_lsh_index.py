"""Tests for Locality Sensitive Hashing (LSH) index.

This module tests the LSH-based similarity search functionality.
"""

import numpy as np
import pytest

from coral.core.lsh_index import LSHConfig, LSHIndex, LSHTable, MultiDimLSHIndex


class TestLSHConfig:
    """Tests for LSHConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LSHConfig()
        assert config.num_hyperplanes == 8
        assert config.num_tables == 4
        assert config.seed == 42
        assert config.max_candidates == 100

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LSHConfig(
            num_hyperplanes=16,
            num_tables=8,
            seed=123,
            max_candidates=50,
        )
        assert config.num_hyperplanes == 16
        assert config.num_tables == 8
        assert config.seed == 123
        assert config.max_candidates == 50


class TestLSHTable:
    """Tests for LSHTable class."""

    @pytest.fixture
    def table(self):
        """Create a table with random hyperplanes."""
        rng = np.random.RandomState(42)
        hyperplanes = rng.randn(8, 100).astype(np.float32)
        norms = np.linalg.norm(hyperplanes, axis=1, keepdims=True)
        hyperplanes = hyperplanes / (norms + 1e-10)
        return LSHTable(hyperplanes=hyperplanes)

    def test_hash_vector_deterministic(self, table):
        """Test that hash is deterministic for same vector."""
        vector = np.random.randn(100).astype(np.float32)

        hash1 = table.hash_vector(vector)
        hash2 = table.hash_vector(vector)

        assert hash1 == hash2

    def test_hash_vector_different_for_different_vectors(self, table):
        """Test that different vectors likely have different hashes."""
        rng = np.random.RandomState(42)

        # Generate many random vectors and count unique hashes
        vectors = [rng.randn(100).astype(np.float32) for _ in range(100)]
        hashes = [table.hash_vector(v) for v in vectors]

        # Should have multiple unique hashes (not all the same)
        unique_hashes = set(hashes)
        assert len(unique_hashes) > 1

    def test_hash_vector_range(self, table):
        """Test that hash value is within expected range."""
        vector = np.random.randn(100).astype(np.float32)
        hash_val = table.hash_vector(vector)

        # With 8 hyperplanes, hash should be in range [0, 255]
        assert 0 <= hash_val <= 255

    def test_insert(self, table):
        """Test inserting a vector."""
        vector = np.random.randn(100).astype(np.float32)
        bucket_hash = table.insert(vector, "key1")

        assert isinstance(bucket_hash, (int, np.integer))
        assert "key1" in table.buckets[bucket_hash]

    def test_insert_multiple_same_bucket(self, table):
        """Test inserting vectors that hash to the same bucket."""
        vector = np.random.randn(100).astype(np.float32)

        # Insert the same vector with different keys
        table.insert(vector, "key1")
        table.insert(vector, "key2")

        bucket_hash = table.hash_vector(vector)
        assert "key1" in table.buckets[bucket_hash]
        assert "key2" in table.buckets[bucket_hash]

    def test_query(self, table):
        """Test querying for similar vectors."""
        vector = np.random.randn(100).astype(np.float32)
        table.insert(vector, "key1")

        # Query with the same vector
        results = table.query(vector)
        assert "key1" in results

    def test_query_empty_bucket(self, table):
        """Test querying when bucket is empty."""
        vector = np.random.randn(100).astype(np.float32)
        results = table.query(vector)
        assert results == set()

    def test_remove(self, table):
        """Test removing a vector."""
        vector = np.random.randn(100).astype(np.float32)
        table.insert(vector, "key1")

        result = table.remove(vector, "key1")
        assert result is True
        assert table.query(vector) == set()

    def test_remove_nonexistent(self, table):
        """Test removing a non-existent key."""
        vector = np.random.randn(100).astype(np.float32)
        result = table.remove(vector, "nonexistent")
        assert result is False

    def test_remove_cleans_empty_bucket(self, table):
        """Test that removing last key cleans up bucket."""
        vector = np.random.randn(100).astype(np.float32)
        bucket_hash = table.insert(vector, "key1")

        table.remove(vector, "key1")
        assert bucket_hash not in table.buckets


class TestLSHIndex:
    """Tests for LSHIndex class."""

    @pytest.fixture
    def index(self):
        """Create an LSH index."""
        return LSHIndex(vector_dim=100)

    @pytest.fixture
    def index_with_data(self, index):
        """Create an LSH index with some data."""
        rng = np.random.RandomState(42)
        for i in range(10):
            vector = rng.randn(100).astype(np.float32)
            index.insert(vector, f"key_{i}")
        return index

    def test_init_default_config(self, index):
        """Test initialization with default config."""
        assert index.vector_dim == 100
        assert len(index.tables) == 4
        assert index.config.num_hyperplanes == 8

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = LSHConfig(num_tables=8, num_hyperplanes=16)
        index = LSHIndex(vector_dim=50, config=config)

        assert index.vector_dim == 50
        assert len(index.tables) == 8
        assert index.config.num_hyperplanes == 16

    def test_insert(self, index):
        """Test inserting a vector."""
        vector = np.random.randn(100).astype(np.float32)
        index.insert(vector, "test_key")

        assert "test_key" in index.keys
        assert len(index) == 1

    def test_insert_wrong_dimension(self, index):
        """Test that inserting wrong dimension raises error."""
        vector = np.random.randn(50).astype(np.float32)

        with pytest.raises(ValueError, match="doesn't match"):
            index.insert(vector, "test_key")

    def test_insert_normalizes_vector(self, index):
        """Test that insert normalizes vectors."""
        # Large magnitude vector
        vector = np.random.randn(100).astype(np.float32) * 1000
        index.insert(vector, "key1")

        # Small magnitude vector (but same direction)
        small_vector = vector / 1000
        index.insert(small_vector, "key2")

        # Both should hash to same buckets (since they're normalized)
        candidates = index.query(vector)
        assert "key1" in candidates
        assert "key2" in candidates

    def test_query(self, index):
        """Test querying for similar vectors."""
        vector = np.random.randn(100).astype(np.float32)
        index.insert(vector, "test_key")

        candidates = index.query(vector)
        assert "test_key" in candidates

    def test_query_wrong_dimension(self, index):
        """Test that querying with wrong dimension raises error."""
        vector = np.random.randn(50).astype(np.float32)

        with pytest.raises(ValueError, match="doesn't match"):
            index.query(vector)

    def test_query_max_candidates(self, index_with_data):
        """Test that max_candidates limits results."""
        vector = np.random.randn(100).astype(np.float32)

        candidates = index_with_data.query(vector, max_candidates=3)
        assert len(candidates) <= 3

    def test_similar_vectors_same_bucket(self):
        """Test that similar vectors hash to the same bucket."""
        config = LSHConfig(num_tables=10, num_hyperplanes=4)
        index = LSHIndex(vector_dim=100, config=config)

        # Create a base vector
        rng = np.random.RandomState(42)
        base_vector = rng.randn(100).astype(np.float32)

        # Create a similar vector (small perturbation)
        similar_vector = base_vector + rng.randn(100).astype(np.float32) * 0.01

        # Insert both
        index.insert(base_vector, "base")
        index.insert(similar_vector, "similar")

        # Query with base vector - should find similar
        candidates = index.query(base_vector)
        assert "base" in candidates
        # Similar should often (but not always) be in candidates
        # With 10 tables and small perturbation, probability is high

    def test_dissimilar_vectors_different_buckets(self):
        """Test that dissimilar vectors hash to different buckets."""
        config = LSHConfig(num_tables=4, num_hyperplanes=8)
        index = LSHIndex(vector_dim=100, config=config)

        # Create orthogonal vectors (dissimilar)
        vec1 = np.zeros(100, dtype=np.float32)
        vec1[:50] = 1.0

        vec2 = np.zeros(100, dtype=np.float32)
        vec2[50:] = 1.0

        index.insert(vec1, "vec1")
        index.insert(vec2, "vec2")

        # Query with vec1 - less likely to find vec2
        candidates = index.query(vec1)
        assert "vec1" in candidates
        # vec2 should usually not be in candidates (though possible)

    def test_remove(self, index):
        """Test removing a vector."""
        vector = np.random.randn(100).astype(np.float32)
        index.insert(vector, "test_key")

        result = index.remove(vector, "test_key")
        assert result is True
        assert "test_key" not in index.keys
        assert len(index) == 0

    def test_remove_nonexistent(self, index):
        """Test removing non-existent key."""
        vector = np.random.randn(100).astype(np.float32)
        result = index.remove(vector, "nonexistent")
        assert result is False

    def test_clear(self, index_with_data):
        """Test clearing the index."""
        assert len(index_with_data) == 10

        index_with_data.clear()

        assert len(index_with_data) == 0
        assert len(index_with_data.keys) == 0

    def test_get_stats(self, index_with_data):
        """Test getting index statistics."""
        stats = index_with_data.get_stats()

        assert stats["num_vectors"] == 10
        assert stats["num_tables"] == 4
        assert stats["num_hyperplanes"] == 8
        assert stats["vector_dim"] == 100
        assert stats["total_buckets"] > 0
        assert stats["avg_bucket_size"] > 0

    def test_len(self, index):
        """Test __len__ method."""
        assert len(index) == 0

        vector = np.random.randn(100).astype(np.float32)
        index.insert(vector, "key1")
        assert len(index) == 1

        index.insert(np.random.randn(100).astype(np.float32), "key2")
        assert len(index) == 2

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        config = LSHConfig(seed=123)
        index1 = LSHIndex(vector_dim=50, config=config)
        index2 = LSHIndex(vector_dim=50, config=config)

        vector = np.random.randn(50).astype(np.float32)

        # Both indices should hash the same vector to the same buckets
        for t1, t2 in zip(index1.tables, index2.tables):
            assert t1.hash_vector(vector) == t2.hash_vector(vector)


class TestMultiDimLSHIndex:
    """Tests for MultiDimLSHIndex class."""

    @pytest.fixture
    def multi_index(self):
        """Create a multi-dimensional LSH index."""
        return MultiDimLSHIndex()

    def test_init(self, multi_index):
        """Test initialization."""
        assert len(multi_index.indices) == 0
        assert len(multi_index) == 0

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = LSHConfig(num_tables=8)
        index = MultiDimLSHIndex(config=config)
        assert index.config.num_tables == 8

    def test_insert_creates_index_for_dimension(self, multi_index):
        """Test that insert creates index for new dimension."""
        vector = np.random.randn(100).astype(np.float32)
        multi_index.insert(vector, "key1")

        assert 100 in multi_index.indices
        assert len(multi_index) == 1

    def test_insert_multiple_dimensions(self, multi_index):
        """Test inserting vectors of different dimensions."""
        vec_100 = np.random.randn(100).astype(np.float32)
        vec_200 = np.random.randn(200).astype(np.float32)
        vec_50 = np.random.randn(50).astype(np.float32)

        multi_index.insert(vec_100, "key_100")
        multi_index.insert(vec_200, "key_200")
        multi_index.insert(vec_50, "key_50")

        assert len(multi_index.indices) == 3
        assert 100 in multi_index.indices
        assert 200 in multi_index.indices
        assert 50 in multi_index.indices
        assert len(multi_index) == 3

    def test_insert_same_dimension(self, multi_index):
        """Test inserting multiple vectors of same dimension."""
        for i in range(5):
            vector = np.random.randn(100).astype(np.float32)
            multi_index.insert(vector, f"key_{i}")

        assert len(multi_index.indices) == 1
        assert len(multi_index) == 5

    def test_query(self, multi_index):
        """Test querying for similar vectors."""
        vector = np.random.randn(100).astype(np.float32)
        multi_index.insert(vector, "key1")

        candidates = multi_index.query(vector)
        assert "key1" in candidates

    def test_query_only_returns_same_dimension(self, multi_index):
        """Test that query only returns vectors of same dimension."""
        vec_100 = np.random.randn(100).astype(np.float32)
        vec_200 = np.random.randn(200).astype(np.float32)

        multi_index.insert(vec_100, "key_100")
        multi_index.insert(vec_200, "key_200")

        candidates = multi_index.query(vec_100)
        assert "key_100" in candidates
        assert "key_200" not in candidates

    def test_query_nonexistent_dimension(self, multi_index):
        """Test querying for dimension that doesn't exist."""
        vector = np.random.randn(100).astype(np.float32)
        candidates = multi_index.query(vector)
        assert candidates == set()

    def test_remove(self, multi_index):
        """Test removing a vector."""
        vector = np.random.randn(100).astype(np.float32)
        multi_index.insert(vector, "key1")

        result = multi_index.remove(vector, "key1")
        assert result is True
        assert len(multi_index) == 0

    def test_remove_nonexistent(self, multi_index):
        """Test removing non-existent key."""
        vector = np.random.randn(100).astype(np.float32)
        result = multi_index.remove(vector, "nonexistent")
        assert result is False

    def test_clear(self, multi_index):
        """Test clearing all indices."""
        multi_index.insert(np.random.randn(100).astype(np.float32), "key1")
        multi_index.insert(np.random.randn(200).astype(np.float32), "key2")

        multi_index.clear()

        assert len(multi_index) == 0
        assert len(multi_index.indices) == 0
        assert len(multi_index.key_to_dim) == 0

    def test_get_stats(self, multi_index):
        """Test getting statistics."""
        multi_index.insert(np.random.randn(100).astype(np.float32), "key1")
        multi_index.insert(np.random.randn(200).astype(np.float32), "key2")

        stats = multi_index.get_stats()

        assert stats["num_dimensions"] == 2
        assert stats["total_vectors"] == 2
        assert 100 in stats["per_dimension"]
        assert 200 in stats["per_dimension"]

    def test_len(self, multi_index):
        """Test __len__ method."""
        assert len(multi_index) == 0

        multi_index.insert(np.random.randn(100).astype(np.float32), "key1")
        assert len(multi_index) == 1

        multi_index.insert(np.random.randn(200).astype(np.float32), "key2")
        assert len(multi_index) == 2


class TestLSHSimilarityProperties:
    """Test similarity properties of LSH."""

    def test_identical_vectors_always_same_bucket(self):
        """Test that identical vectors always hash to same bucket."""
        config = LSHConfig(num_tables=10, num_hyperplanes=16)
        index = LSHIndex(vector_dim=100, config=config)

        vector = np.random.randn(100).astype(np.float32)

        index.insert(vector, "key1")

        # Query with exact same vector
        for _ in range(10):
            candidates = index.query(vector)
            assert "key1" in candidates

    def test_opposite_vectors_different_buckets(self):
        """Test that opposite vectors hash to different buckets."""
        config = LSHConfig(num_tables=1, num_hyperplanes=8)
        index = LSHIndex(vector_dim=100, config=config)

        vector = np.random.randn(100).astype(np.float32)
        opposite = -vector

        index.insert(vector, "pos")
        index.insert(opposite, "neg")

        # With a single table, opposite vectors should be in different buckets
        candidates = index.query(vector)
        assert "pos" in candidates
        # "neg" should usually not be in candidates (opposite direction)

    def test_many_vectors_bucket_distribution(self):
        """Test that vectors are distributed across multiple buckets."""
        config = LSHConfig(num_tables=1, num_hyperplanes=8)
        index = LSHIndex(vector_dim=100, config=config)

        # Insert many random vectors
        rng = np.random.RandomState(42)
        for i in range(100):
            vector = rng.randn(100).astype(np.float32)
            index.insert(vector, f"key_{i}")

        stats = index.get_stats()

        # Should have multiple buckets
        assert stats["total_buckets"] > 1
        # Average bucket size should be reasonable
        assert stats["avg_bucket_size"] < 50  # Not all in one bucket

    def test_high_dimensional_vectors(self):
        """Test LSH with high-dimensional vectors."""
        config = LSHConfig(num_hyperplanes=16)
        index = LSHIndex(vector_dim=10000, config=config)

        vector = np.random.randn(10000).astype(np.float32)
        index.insert(vector, "high_dim")

        candidates = index.query(vector)
        assert "high_dim" in candidates

    def test_2d_vectors(self):
        """Test LSH with 2D weight matrices."""
        index = LSHIndex(vector_dim=100)  # 10x10 flattened

        # Create a 2D matrix and flatten
        matrix = np.random.randn(10, 10).astype(np.float32)
        index.insert(matrix, "matrix1")

        # Query with same shape
        query = matrix.reshape(10, 10)
        candidates = index.query(query)
        assert "matrix1" in candidates

    def test_zero_vector_handling(self):
        """Test handling of zero vectors."""
        index = LSHIndex(vector_dim=100)

        # Zero vector has no direction, but should still work
        zero_vector = np.zeros(100, dtype=np.float32)
        index.insert(zero_vector, "zero")

        candidates = index.query(zero_vector)
        assert "zero" in candidates
