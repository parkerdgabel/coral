"""Tests for SimHash implementation.

Tests the SimHash fingerprinting and indexing for fast similarity detection.
"""

import numpy as np
import pytest

from coral.core.simhash import (
    MultiDimSimHashIndex,
    SimHash,
    SimHashConfig,
    SimHashIndex,
)


class TestSimHashFingerprint:
    """Test SimHash fingerprint computation."""

    def test_fingerprint_deterministic(self):
        """Same vector should produce same fingerprint."""
        hasher = SimHash(SimHashConfig(seed=42))

        vector = np.random.randn(100).astype(np.float32)
        fp1 = hasher.compute_fingerprint(vector)
        fp2 = hasher.compute_fingerprint(vector.copy())

        assert fp1 == fp2, "Same vector should have same fingerprint"

    def test_fingerprint_similar_vectors(self):
        """Similar vectors should have small Hamming distance."""
        hasher = SimHash(SimHashConfig(seed=42, num_bits=64))

        np.random.seed(42)
        vector1 = np.random.randn(100).astype(np.float32)
        # Add small perturbation
        vector2 = vector1 + 0.01 * np.random.randn(100).astype(np.float32)

        fp1 = hasher.compute_fingerprint(vector1)
        fp2 = hasher.compute_fingerprint(vector2)

        distance = hasher.hamming_distance(fp1, fp2)

        # Similar vectors should have small distance
        assert distance < 16, (
            f"Similar vectors should have small Hamming distance, got {distance}"
        )

    def test_fingerprint_dissimilar_vectors(self):
        """Dissimilar vectors should have large Hamming distance."""
        hasher = SimHash(SimHashConfig(seed=42, num_bits=64))

        np.random.seed(42)
        vector1 = np.random.randn(100).astype(np.float32)
        vector2 = np.random.randn(100).astype(np.float32)  # Independent random

        fp1 = hasher.compute_fingerprint(vector1)
        fp2 = hasher.compute_fingerprint(vector2)

        distance = hasher.hamming_distance(fp1, fp2)

        # Dissimilar vectors should have distance around 32 (half of 64 bits)
        assert distance > 16, (
            f"Dissimilar vectors should have larger Hamming distance, got {distance}"
        )

    def test_fingerprint_128bit(self):
        """Test 128-bit fingerprints."""
        hasher = SimHash(SimHashConfig(seed=42, num_bits=128))

        vector = np.random.randn(100).astype(np.float32)
        fp = hasher.compute_fingerprint(vector)

        # Should return tuple of two uint64
        assert isinstance(fp, tuple)
        assert len(fp) == 2

    def test_fingerprint_batch(self):
        """Test batch fingerprint computation."""
        hasher = SimHash(SimHashConfig(seed=42))

        np.random.seed(42)
        vectors = [np.random.randn(100).astype(np.float32) for _ in range(10)]

        # Batch compute
        batch_fps = hasher.compute_fingerprint_batch(vectors)

        # Individual compute
        individual_fps = [hasher.compute_fingerprint(v) for v in vectors]

        # Should match
        for batch_fp, individual_fp in zip(batch_fps, individual_fps):
            assert batch_fp == individual_fp

    def test_fingerprint_scale_invariant(self):
        """SimHash should be roughly scale-invariant (normalized input)."""
        hasher = SimHash(SimHashConfig(seed=42, num_bits=64))

        np.random.seed(42)
        vector = np.random.randn(100).astype(np.float32)
        scaled_vector = vector * 100  # Scale by 100x

        fp1 = hasher.compute_fingerprint(vector)
        fp2 = hasher.compute_fingerprint(scaled_vector)

        # Should be identical (normalization)
        assert fp1 == fp2, "SimHash should be scale-invariant"

    def test_fingerprint_empty_batch(self):
        """Empty batch should return empty list."""
        hasher = SimHash(SimHashConfig(seed=42))
        result = hasher.compute_fingerprint_batch([])
        assert result == []


class TestSimHashSimilarity:
    """Test SimHash similarity detection."""

    def test_are_similar_same_vector(self):
        """Same vector should be detected as similar."""
        hasher = SimHash(SimHashConfig(seed=42, similarity_threshold=0.1))

        vector = np.random.randn(100).astype(np.float32)
        fp = hasher.compute_fingerprint(vector)

        assert hasher.are_similar(fp, fp)

    def test_are_similar_near_vectors(self):
        """Near vectors should be detected as similar."""
        hasher = SimHash(SimHashConfig(seed=42, similarity_threshold=0.2))

        np.random.seed(42)
        vector1 = np.random.randn(100).astype(np.float32)
        vector2 = vector1 + 0.01 * np.random.randn(100).astype(np.float32)

        fp1 = hasher.compute_fingerprint(vector1)
        fp2 = hasher.compute_fingerprint(vector2)

        assert hasher.are_similar(fp1, fp2)

    def test_estimated_similarity(self):
        """Test estimated similarity from Hamming distance."""
        hasher = SimHash(SimHashConfig(seed=42, num_bits=64))

        # Same fingerprint should have similarity 1.0
        fp = np.uint64(0xFFFFFFFFFFFFFFFF)
        assert hasher.estimated_similarity(fp, fp) == pytest.approx(1.0)

        # Maximally different should have similarity -1.0
        fp1 = np.uint64(0)
        fp2 = np.uint64(0xFFFFFFFFFFFFFFFF)
        assert hasher.estimated_similarity(fp1, fp2) == pytest.approx(-1.0)

    def test_hamming_distance_128bit(self):
        """Test Hamming distance for 128-bit fingerprints."""
        hasher = SimHash(SimHashConfig(seed=42, num_bits=128))

        fp1 = (np.uint64(0), np.uint64(0))
        fp2 = (np.uint64(1), np.uint64(1))  # 2 bits different

        distance = hasher.hamming_distance(fp1, fp2)
        assert distance == 2


class TestSimHashIndex:
    """Test SimHash index for similarity search."""

    def test_insert_and_query(self):
        """Test basic insert and query operations."""
        index = SimHashIndex(SimHashConfig(seed=42, similarity_threshold=0.2))

        np.random.seed(42)
        vector1 = np.random.randn(100).astype(np.float32)
        vector2 = vector1 + 0.01 * np.random.randn(100).astype(np.float32)
        vector3 = np.random.randn(100).astype(np.float32)  # Dissimilar

        index.insert(vector1, "v1")
        index.insert(vector3, "v3")

        # Query with similar vector should find v1
        results = index.query(vector2)
        result_ids = [vid for vid, _ in results]

        assert "v1" in result_ids, "Should find similar vector"

    def test_query_empty_index(self):
        """Query on empty index should return empty list."""
        index = SimHashIndex()
        vector = np.random.randn(100).astype(np.float32)

        results = index.query(vector)
        assert results == []

    def test_remove(self):
        """Test removing vectors from index."""
        index = SimHashIndex()

        vector = np.random.randn(100).astype(np.float32)
        index.insert(vector, "v1")

        assert index.size == 1

        result = index.remove("v1")
        assert result is True
        assert index.size == 0

        # Removing non-existent should return False
        result = index.remove("v1")
        assert result is False

    def test_clear(self):
        """Test clearing the index."""
        index = SimHashIndex()

        for i in range(10):
            vector = np.random.randn(100).astype(np.float32)
            index.insert(vector, f"v{i}")

        assert index.size == 10

        index.clear()
        assert index.size == 0

    def test_get_stats(self):
        """Test getting index statistics."""
        index = SimHashIndex(SimHashConfig(num_bits=64))

        for i in range(5):
            vector = np.random.randn(100).astype(np.float32)
            index.insert(vector, f"v{i}")

        stats = index.get_stats()

        assert stats["num_vectors"] == 5
        assert stats["fingerprint_bits"] == 64
        assert "num_unique_fingerprints" in stats

    def test_query_by_fingerprint(self):
        """Test querying using precomputed fingerprint."""
        index = SimHashIndex(SimHashConfig(seed=42, similarity_threshold=0.2))

        np.random.seed(42)
        vector1 = np.random.randn(100).astype(np.float32)
        index.insert(vector1, "v1")

        # Compute fingerprint separately
        fp = index.hasher.compute_fingerprint(vector1)

        results = index.query_by_fingerprint(fp)
        result_ids = [vid for vid, _ in results]

        assert "v1" in result_ids


class TestMultiDimSimHashIndex:
    """Test multi-dimensional SimHash index."""

    def test_different_dimensions(self):
        """Test handling vectors of different dimensions."""
        index = MultiDimSimHashIndex(SimHashConfig(seed=42, similarity_threshold=0.2))

        # Insert vectors of different dimensions
        np.random.seed(42)
        v_100 = np.random.randn(100).astype(np.float32)
        v_200 = np.random.randn(200).astype(np.float32)
        v_100_similar = v_100 + 0.01 * np.random.randn(100).astype(np.float32)

        index.insert(v_100, "v100")
        index.insert(v_200, "v200")

        # Query with similar 100-dim vector
        results = index.query(v_100_similar)

        assert "v100" in results
        assert "v200" not in results  # Different dimension

    def test_size_across_dimensions(self):
        """Test size counting across dimensions."""
        index = MultiDimSimHashIndex()

        for dim in [50, 100, 150]:
            for i in range(3):
                vector = np.random.randn(dim).astype(np.float32)
                index.insert(vector, f"v{dim}_{i}")

        assert index.size == 9

    def test_clear_all_dimensions(self):
        """Test clearing all dimension indices."""
        index = MultiDimSimHashIndex()

        for dim in [50, 100]:
            vector = np.random.randn(dim).astype(np.float32)
            index.insert(vector, f"v{dim}")

        index.clear()

        assert index.size == 0
        assert len(index._indices) == 0

    def test_get_stats(self):
        """Test getting stats across dimensions."""
        index = MultiDimSimHashIndex()

        for dim in [100, 200]:
            for i in range(2):
                vector = np.random.randn(dim).astype(np.float32)
                index.insert(vector, f"v{dim}_{i}")

        stats = index.get_stats()

        assert stats["total_vectors"] == 4
        assert stats["num_dimensions"] == 2
        assert 100 in stats["dimension_stats"]
        assert 200 in stats["dimension_stats"]

    def test_remove_by_dimension(self):
        """Test removing vectors from specific dimension."""
        index = MultiDimSimHashIndex()

        np.random.seed(42)
        v_100 = np.random.randn(100).astype(np.float32)
        index.insert(v_100, "v100")

        result = index.remove(v_100, "v100")
        assert result is True
        assert index.size == 0


class TestSimHashPerformance:
    """Test SimHash performance characteristics."""

    def test_batch_produces_same_results(self):
        """Batch processing should produce same results as individual."""
        hasher = SimHash(SimHashConfig(seed=42))

        np.random.seed(42)
        vectors = [np.random.randn(1000).astype(np.float32) for _ in range(100)]

        # Compute both ways
        batch_fps = hasher.compute_fingerprint_batch(vectors)
        individual_fps = [hasher.compute_fingerprint(v) for v in vectors]

        # Results should match
        for batch_fp, individual_fp in zip(batch_fps, individual_fps):
            assert batch_fp == individual_fp, (
                "Batch and individual fingerprints should match"
            )

    def test_fingerprint_size(self):
        """Fingerprint should be fixed size regardless of input."""
        hasher = SimHash(SimHashConfig(seed=42, num_bits=64))

        for dim in [10, 100, 1000, 10000]:
            vector = np.random.randn(dim).astype(np.float32)
            fp = hasher.compute_fingerprint(vector)

            # 64-bit fingerprint is just a uint64
            assert isinstance(fp, np.uint64)


class TestSimHashIntegration:
    """Integration tests for SimHash with weight-like data."""

    def test_neural_network_weight_similarity(self):
        """Test with realistic neural network weight patterns."""
        index = MultiDimSimHashIndex(
            SimHashConfig(seed=42, num_bits=64, similarity_threshold=0.15)
        )

        np.random.seed(42)

        # Create "base" weights for a layer
        base_weights = np.random.randn(256, 256).astype(np.float32)
        index.insert(base_weights, "base")

        # Fine-tuned version (very similar)
        fine_tuned = base_weights + 0.001 * np.random.randn(256, 256).astype(np.float32)

        # Different checkpoint (moderately similar)
        checkpoint = base_weights + 0.01 * np.random.randn(256, 256).astype(np.float32)

        # Query each
        results_ft = index.query(fine_tuned)
        results_cp = index.query(checkpoint)

        # Fine-tuned and checkpoint should find base
        assert "base" in results_ft, "Fine-tuned weights should match base"
        assert "base" in results_cp, "Checkpoint should match base"
