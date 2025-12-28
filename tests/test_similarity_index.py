"""Tests for the unified similarity index."""

import numpy as np
import pytest

from coral.core.similarity_index import (
    Fingerprint,
    MultiDimSimilarityIndex,
    SimilarityIndex,
    SimilarityIndexConfig,
)


class TestFingerprint:
    """Tests for Fingerprint class."""

    def test_fingerprint_creation(self):
        """Test creating a fingerprint from bits."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 8, dtype=np.uint8)
        fp = Fingerprint(bits, num_bits=64)

        assert fp.num_bits == 64

    def test_fingerprint_hamming_distance(self):
        """Test Hamming distance calculation."""
        # Create two fingerprints
        bits1 = np.zeros(64, dtype=np.uint8)
        bits2 = np.zeros(64, dtype=np.uint8)

        # Set some differing bits
        bits1[0] = 1
        bits1[10] = 1
        bits1[20] = 1
        bits2[10] = 1  # Same as bits1

        fp1 = Fingerprint(bits1, 64)
        fp2 = Fingerprint(bits2, 64)

        # Should differ in bits 0 and 20
        distance = fp1.hamming_distance(fp2)
        assert distance == 2

    def test_fingerprint_self_distance(self):
        """Test that distance to self is zero."""
        bits = np.random.randint(0, 2, 64, dtype=np.uint8)
        fp = Fingerprint(bits, 64)

        assert fp.hamming_distance(fp) == 0

    def test_fingerprint_estimated_similarity(self):
        """Test estimated cosine similarity from fingerprints."""
        # Identical fingerprints should have similarity ~1
        bits = np.random.randint(0, 2, 64, dtype=np.uint8)
        fp1 = Fingerprint(bits, 64)
        fp2 = Fingerprint(bits.copy(), 64)

        similarity = fp1.estimated_cosine_similarity(fp2)
        assert similarity == pytest.approx(1.0)

        # Opposite fingerprints should have similarity ~-1
        opposite_bits = 1 - bits
        fp3 = Fingerprint(opposite_bits, 64)
        similarity = fp1.estimated_cosine_similarity(fp3)
        assert similarity == pytest.approx(-1.0)

    def test_fingerprint_serialization(self):
        """Test serializing and deserializing fingerprints."""
        bits = np.random.randint(0, 2, 64, dtype=np.uint8)
        fp1 = Fingerprint(bits, 64)

        # Serialize
        data = fp1.to_bytes()

        # Deserialize
        fp2 = Fingerprint.from_bytes(data, 64)

        # Should be equivalent
        assert fp1.hamming_distance(fp2) == 0

    def test_fingerprint_bucket_hash(self):
        """Test extracting bucket hash from specific bits."""
        bits = np.zeros(64, dtype=np.uint8)
        bits[0] = 1
        bits[2] = 1
        bits[5] = 1

        fp = Fingerprint(bits, 64)

        # Extract hash from bits 0, 2, 5
        indices = np.array([0, 2, 5])
        bucket_hash = fp.to_bucket_hash(indices)

        # Expected: bit 0 at position 0, bit 2 at position 1, bit 5 at position 2
        # = 1 + 2 + 4 = 7
        assert bucket_hash == 7


class TestSimilarityIndex:
    """Tests for SimilarityIndex."""

    def test_index_insert_and_query(self):
        """Test basic insert and query operations."""
        config = SimilarityIndexConfig(num_bits=64, num_tables=4)
        index = SimilarityIndex(vector_dim=100, config=config)

        # Insert some vectors
        v1 = np.random.randn(100).astype(np.float32)
        v2 = v1 + np.random.randn(100).astype(np.float32) * 0.1  # Similar to v1
        v3 = np.random.randn(100).astype(np.float32)  # Different

        index.insert(v1, "vec1")
        index.insert(v2, "vec2")
        index.insert(v3, "vec3")

        assert len(index) == 3
        assert "vec1" in index
        assert "vec2" in index

    def test_index_finds_similar_vectors(self):
        """Test that similar vectors are found."""
        config = SimilarityIndexConfig(
            num_bits=64,
            num_tables=4,
            hamming_threshold=0.2,
        )
        index = SimilarityIndex(vector_dim=256, config=config)

        # Create a base vector
        base = np.random.randn(256).astype(np.float32)

        # Create similar vectors (small perturbations)
        similar_vectors = []
        for i in range(5):
            similar = base + np.random.randn(256).astype(np.float32) * 0.05
            similar_vectors.append(similar)
            index.insert(similar, f"similar_{i}")

        # Create dissimilar vectors
        for i in range(5):
            dissimilar = np.random.randn(256).astype(np.float32)
            index.insert(dissimilar, f"dissimilar_{i}")

        # Query with base vector
        results = index.query(base, max_candidates=10)

        # Similar vectors should be found with high estimated similarity
        found_similar = [r for r in results if r.vector_id.startswith("similar_")]
        assert len(found_similar) >= 3  # Should find most similar vectors

    def test_index_query_by_id(self):
        """Test querying by vector ID."""
        config = SimilarityIndexConfig(num_bits=64, num_tables=4)
        index = SimilarityIndex(vector_dim=50, config=config)

        # Insert vectors
        vectors = {}
        for i in range(10):
            v = np.random.randn(50).astype(np.float32)
            index.insert(v, f"vec_{i}")
            vectors[f"vec_{i}"] = v

        # Query by ID
        results = index.query_by_id("vec_0")

        # Should not include self
        assert all(r.vector_id != "vec_0" for r in results)

    def test_index_remove(self):
        """Test removing vectors from index."""
        index = SimilarityIndex(vector_dim=64)

        v1 = np.random.randn(64).astype(np.float32)
        v2 = np.random.randn(64).astype(np.float32)

        index.insert(v1, "vec1")
        index.insert(v2, "vec2")
        assert len(index) == 2

        # Remove one
        result = index.remove("vec1")
        assert result is True
        assert len(index) == 1
        assert "vec1" not in index
        assert "vec2" in index

        # Remove non-existent
        result = index.remove("nonexistent")
        assert result is False

    def test_index_clear(self):
        """Test clearing the index."""
        index = SimilarityIndex(vector_dim=32)

        for i in range(10):
            index.insert(np.random.randn(32).astype(np.float32), f"v_{i}")

        assert len(index) == 10

        index.clear()
        assert len(index) == 0

    def test_index_with_vector_storage(self):
        """Test index with actual vector storage for exact similarity."""
        config = SimilarityIndexConfig(
            num_bits=64,
            num_tables=4,
            store_vectors=True,
        )
        index = SimilarityIndex(vector_dim=100, config=config)

        v1 = np.random.randn(100).astype(np.float32)
        v2 = v1 + np.random.randn(100).astype(np.float32) * 0.1

        index.insert(v1, "vec1")
        index.insert(v2, "vec2")

        results = index.query(v1)

        # With vector storage, actual_similarity should be computed
        if results:
            assert results[0].actual_similarity is not None

    def test_index_stats(self):
        """Test getting index statistics."""
        config = SimilarityIndexConfig(num_bits=64, num_tables=4)
        index = SimilarityIndex(vector_dim=128, config=config)

        for i in range(20):
            index.insert(np.random.randn(128).astype(np.float32), f"v_{i}")

        stats = index.get_stats()

        assert stats["num_vectors"] == 20
        assert stats["vector_dim"] == 128
        assert stats["num_bits"] == 64
        assert stats["num_tables"] == 4


class TestMultiDimSimilarityIndex:
    """Tests for MultiDimSimilarityIndex."""

    def test_multi_dim_insert_different_dims(self):
        """Test inserting vectors of different dimensions."""
        index = MultiDimSimilarityIndex()

        # Insert vectors of different sizes
        v32 = np.random.randn(32).astype(np.float32)
        v64 = np.random.randn(64).astype(np.float32)
        v128 = np.random.randn(128).astype(np.float32)

        index.insert(v32, "dim32")
        index.insert(v64, "dim64")
        index.insert(v128, "dim128")

        assert len(index) == 3
        assert "dim32" in index
        assert "dim64" in index
        assert "dim128" in index

    def test_multi_dim_query_same_dimension(self):
        """Test that queries only return vectors of same dimension."""
        index = MultiDimSimilarityIndex()

        # Insert vectors of different sizes
        for i in range(5):
            index.insert(np.random.randn(32).astype(np.float32), f"small_{i}")
        for i in range(5):
            index.insert(np.random.randn(64).astype(np.float32), f"large_{i}")

        # Query with 32-dim vector
        query = np.random.randn(32).astype(np.float32)
        results = index.query(query)

        # Should only find 32-dim vectors
        for result in results:
            assert result.vector_id.startswith("small_")

    def test_multi_dim_remove(self):
        """Test removing from multi-dim index."""
        index = MultiDimSimilarityIndex()

        index.insert(np.random.randn(32).astype(np.float32), "v1")
        index.insert(np.random.randn(64).astype(np.float32), "v2")

        assert len(index) == 2

        index.remove("v1")
        assert len(index) == 1
        assert "v1" not in index

    def test_multi_dim_clear(self):
        """Test clearing multi-dim index."""
        index = MultiDimSimilarityIndex()

        for dim in [32, 64, 128]:
            for i in range(3):
                index.insert(np.random.randn(dim).astype(np.float32), f"v_{dim}_{i}")

        assert len(index) == 9

        index.clear()
        assert len(index) == 0

    def test_multi_dim_stats(self):
        """Test statistics for multi-dim index."""
        index = MultiDimSimilarityIndex()

        for i in range(5):
            index.insert(np.random.randn(32).astype(np.float32), f"small_{i}")
        for i in range(3):
            index.insert(np.random.randn(64).astype(np.float32), f"large_{i}")

        stats = index.get_stats()

        assert stats["num_dimensions"] == 2
        assert stats["total_vectors"] == 8
        assert 32 in stats["per_dimension"]
        assert 64 in stats["per_dimension"]

    def test_multi_dim_get_fingerprint(self):
        """Test getting fingerprints from multi-dim index."""
        index = MultiDimSimilarityIndex()

        v = np.random.randn(50).astype(np.float32)
        index.insert(v, "test_vec")

        fp = index.get_fingerprint("test_vec")
        assert fp is not None
        assert isinstance(fp, Fingerprint)

        # Non-existent
        fp2 = index.get_fingerprint("nonexistent")
        assert fp2 is None


class TestSimilarityIndexConfig:
    """Tests for SimilarityIndexConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SimilarityIndexConfig()

        assert config.num_bits == 64
        assert config.num_tables == 4
        assert config.bits_per_table == 16  # 64 // 4
        assert config.seed == 42
        assert config.hamming_threshold == 0.15
        assert config.max_candidates == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = SimilarityIndexConfig(
            num_bits=128,
            num_tables=8,
            bits_per_table=8,
            seed=123,
            hamming_threshold=0.1,
            max_candidates=50,
        )

        assert config.num_bits == 128
        assert config.num_tables == 8
        assert config.bits_per_table == 8
        assert config.seed == 123

    def test_auto_bits_per_table(self):
        """Test automatic bits_per_table calculation."""
        config = SimilarityIndexConfig(num_bits=96, num_tables=6)
        assert config.bits_per_table == 16  # 96 // 6

        config2 = SimilarityIndexConfig(num_bits=64, num_tables=8)
        assert config2.bits_per_table == 8  # 64 // 8


class TestSimilarityAccuracy:
    """Tests for accuracy of similarity detection."""

    def test_similar_vectors_detected(self):
        """Test that similar vectors are correctly identified."""
        config = SimilarityIndexConfig(
            num_bits=128,
            num_tables=8,
            hamming_threshold=0.2,
        )
        index = SimilarityIndex(vector_dim=512, config=config)

        # Create base vector
        np.random.seed(42)
        base = np.random.randn(512).astype(np.float32)
        base = base / np.linalg.norm(base)

        # Create similar vector (cosine similarity > 0.9)
        # Use small noise relative to the signal
        noise = np.random.randn(512).astype(np.float32) * 0.01
        similar = base + noise
        similar = similar / np.linalg.norm(similar)

        # Compute actual similarity
        actual_sim = np.dot(base, similar)
        assert actual_sim > 0.9, f"Vectors should be similar: {actual_sim}"

        # Insert and query
        index.insert(similar, "similar")

        # Insert some dissimilar vectors
        for i in range(20):
            v = np.random.randn(512).astype(np.float32)
            v = v / np.linalg.norm(v)
            index.insert(v, f"random_{i}")

        results = index.query(base)

        # Similar vector should be in top results
        top_ids = [r.vector_id for r in results[:5]]
        assert "similar" in top_ids, f"Similar not in top results: {top_ids}"

    def test_dissimilar_vectors_filtered(self):
        """Test that very dissimilar vectors are not returned."""
        config = SimilarityIndexConfig(
            num_bits=64,
            num_tables=4,
            hamming_threshold=0.1,  # Strict threshold
        )
        index = SimilarityIndex(vector_dim=256, config=config)

        # Insert random vectors
        np.random.seed(42)
        for i in range(50):
            v = np.random.randn(256).astype(np.float32)
            index.insert(v, f"vec_{i}")

        # Query with a new random vector
        query = np.random.randn(256).astype(np.float32)
        results = index.query(query, threshold=0.05)

        # With very strict threshold, most random vectors should be filtered
        # (Note: some might pass by chance due to hash collisions)
        for result in results:
            # Estimated similarity should be reasonably high
            assert result.estimated_similarity > 0.5
