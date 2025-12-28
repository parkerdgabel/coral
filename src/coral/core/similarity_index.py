"""Unified Similarity Index for neural network weights.

This module provides a unified locality-sensitive hashing system that combines
the best features of SimHash and LSH:

- SimHash-style compact fingerprints for storage and quick comparison
- LSH-style multi-table bucketing for efficient candidate retrieval
- Automatic dimension handling (separate indices per dimension)
- Configurable accuracy/speed tradeoffs

The key insight is that SimHash IS a form of LSH - both use random hyperplane
projections. This unified implementation uses:
1. A single set of random hyperplanes (shared)
2. Compact fingerprints for storage and Hamming distance comparison
3. Multiple hash tables using different subsets of bits for lookup

Reference: Charikar, M. (2002). "Similarity estimation techniques from
rounding algorithms". STOC '02.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SimilarityIndexConfig:
    """Configuration for the unified similarity index.

    The index uses random hyperplane hashing with the following tradeoffs:
    - More bits (num_bits) = more accurate similarity estimation
    - More tables (num_tables) = fewer false negatives but more memory
    - Bits per table (bits_per_table) = fewer false positives but more false negatives

    Default settings are tuned for similarity_threshold >= 0.9.
    """

    # Total number of hyperplanes/bits in the fingerprint
    num_bits: int = 64

    # Number of hash tables for lookup (reduces false negatives)
    num_tables: int = 4

    # Bits per table for bucketing (uses different bit subsets)
    # If None, calculated as num_bits // num_tables
    bits_per_table: Optional[int] = None

    # Random seed for reproducibility
    seed: int = 42

    # Hamming distance threshold as fraction of num_bits
    # E.g., 0.15 means 15% of bits can differ and still be similar
    hamming_threshold: float = 0.15

    # Maximum candidates to return from query
    max_candidates: int = 100

    # Whether to store vectors for exact similarity verification
    # Disable for memory efficiency if you only need candidate retrieval
    store_vectors: bool = False

    def __post_init__(self):
        if self.bits_per_table is None:
            self.bits_per_table = max(4, self.num_bits // self.num_tables)


@dataclass
class SimilarityResult:
    """Result from a similarity query."""

    # Vector ID
    vector_id: str

    # Hamming distance (number of differing bits)
    hamming_distance: int

    # Estimated cosine similarity from fingerprint
    estimated_similarity: float

    # Actual cosine similarity (only if vectors are stored)
    actual_similarity: Optional[float] = None


class Fingerprint:
    """A compact binary fingerprint for similarity comparison.

    Fingerprints enable O(1) similarity estimation via Hamming distance.
    """

    __slots__ = ("_bits", "_num_bits")

    def __init__(self, bits: np.ndarray, num_bits: int):
        """Initialize fingerprint from binary array.

        Args:
            bits: Binary array (0s and 1s)
            num_bits: Number of valid bits
        """
        # Pack bits into uint64 array for efficient storage and comparison
        self._num_bits = num_bits
        num_words = (num_bits + 63) // 64
        self._bits = np.zeros(num_words, dtype=np.uint64)

        for i, bit in enumerate(bits[:num_bits]):
            if bit:
                word_idx = i // 64
                bit_idx = i % 64
                self._bits[word_idx] |= np.uint64(1) << np.uint64(bit_idx)

    @property
    def num_bits(self) -> int:
        """Number of bits in the fingerprint."""
        return self._num_bits

    def hamming_distance(self, other: Fingerprint) -> int:
        """Compute Hamming distance to another fingerprint.

        Args:
            other: Another fingerprint

        Returns:
            Number of differing bits
        """
        if self._num_bits != other._num_bits:
            raise ValueError(
                f"Fingerprint size mismatch: {self._num_bits} vs {other._num_bits}"
            )

        distance = 0
        for w1, w2 in zip(self._bits, other._bits):
            xor = w1 ^ w2
            # Count set bits (popcount)
            distance += bin(int(xor)).count("1")
        return distance

    def to_bucket_hash(self, bit_indices: np.ndarray) -> int:
        """Extract a hash value using specific bit indices.

        Args:
            bit_indices: Which bits to use for the hash

        Returns:
            Integer hash value
        """
        result = 0
        for i, bit_idx in enumerate(bit_indices):
            word_idx = bit_idx // 64
            local_bit = bit_idx % 64
            if self._bits[word_idx] & (np.uint64(1) << np.uint64(local_bit)):
                result |= 1 << i
        return result

    def estimated_cosine_similarity(self, other: Fingerprint) -> float:
        """Estimate cosine similarity from Hamming distance.

        The relationship is: cos(θ) ≈ cos(π * d / num_bits)
        where d is the Hamming distance.

        Args:
            other: Another fingerprint

        Returns:
            Estimated cosine similarity in [-1, 1]
        """
        distance = self.hamming_distance(other)
        theta = np.pi * distance / self._num_bits
        return float(np.cos(theta))

    def to_bytes(self) -> bytes:
        """Serialize fingerprint to bytes."""
        return self._bits.tobytes()

    @classmethod
    def from_bytes(cls, data: bytes, num_bits: int) -> Fingerprint:
        """Deserialize fingerprint from bytes."""
        fp = cls.__new__(cls)
        fp._num_bits = num_bits
        fp._bits = np.frombuffer(data, dtype=np.uint64).copy()
        return fp

    def __repr__(self) -> str:
        return f"Fingerprint(num_bits={self._num_bits})"


class SimilarityIndex:
    """Single-dimension similarity index using unified SimHash/LSH.

    This index provides:
    - O(1) average-case similarity lookup via LSH bucketing
    - Compact fingerprints for storage and Hamming distance comparison
    - Estimated similarity from fingerprint comparison

    Example:
        >>> index = SimilarityIndex(vector_dim=1024)
        >>> index.insert(weight_a.data, "weight_a")
        >>> index.insert(weight_b.data, "weight_b")
        >>> results = index.query(query_weight.data)
        >>> for result in results:
        ...     print(f"{result.vector_id}: ~{result.estimated_similarity:.3f}")
    """

    def __init__(
        self,
        vector_dim: int,
        config: Optional[SimilarityIndexConfig] = None,
    ):
        """Initialize similarity index.

        Args:
            vector_dim: Dimension of vectors to index
            config: Index configuration
        """
        self.vector_dim = vector_dim
        self.config = config or SimilarityIndexConfig()
        self._rng = np.random.RandomState(self.config.seed)

        # Generate random hyperplanes (shared across all tables)
        self._hyperplanes = self._rng.randn(self.config.num_bits, vector_dim).astype(
            np.float32
        )
        # Normalize for numerical stability
        norms = np.linalg.norm(self._hyperplanes, axis=1, keepdims=True)
        self._hyperplanes = self._hyperplanes / (norms + 1e-10)

        # Generate bit indices for each table (non-overlapping subsets)
        self._table_bit_indices: list[np.ndarray] = []
        all_bits = np.arange(self.config.num_bits)
        self._rng.shuffle(all_bits)

        for i in range(self.config.num_tables):
            start = i * self.config.bits_per_table
            end = start + self.config.bits_per_table
            if end <= self.config.num_bits:
                self._table_bit_indices.append(all_bits[start:end].copy())
            else:
                # Wrap around if we run out of bits
                indices = np.concatenate(
                    [all_bits[start:], all_bits[: end - self.config.num_bits]]
                )
                self._table_bit_indices.append(indices)

        # Hash tables: table_idx -> bucket_hash -> set of vector_ids
        self._tables: list[dict[int, set[str]]] = [
            {} for _ in range(self.config.num_tables)
        ]

        # Vector ID -> fingerprint mapping
        self._fingerprints: dict[str, Fingerprint] = {}

        # Optional: store vectors for exact similarity
        self._vectors: dict[str, np.ndarray] = {} if self.config.store_vectors else None

    def _compute_fingerprint(self, vector: np.ndarray) -> Fingerprint:
        """Compute fingerprint for a vector.

        Args:
            vector: Input vector (will be flattened and normalized)

        Returns:
            Binary fingerprint
        """
        flat = vector.flatten().astype(np.float32)
        norm = np.linalg.norm(flat)
        if norm > 0:
            flat = flat / norm

        # Project onto hyperplanes
        projections = np.dot(self._hyperplanes, flat)

        # Create binary signature
        bits = (projections > 0).astype(np.uint8)

        return Fingerprint(bits, self.config.num_bits)

    def insert(self, vector: np.ndarray, vector_id: str) -> Fingerprint:
        """Insert a vector into the index.

        Args:
            vector: Vector to insert
            vector_id: Unique identifier

        Returns:
            The computed fingerprint
        """
        if vector.size != self.vector_dim:
            raise ValueError(
                f"Vector dimension {vector.size} doesn't match "
                f"index dimension {self.vector_dim}"
            )

        fingerprint = self._compute_fingerprint(vector)
        self._fingerprints[vector_id] = fingerprint

        # Insert into each hash table
        for table_idx, bit_indices in enumerate(self._table_bit_indices):
            bucket_hash = fingerprint.to_bucket_hash(bit_indices)
            if bucket_hash not in self._tables[table_idx]:
                self._tables[table_idx][bucket_hash] = set()
            self._tables[table_idx][bucket_hash].add(vector_id)

        # Optionally store vector for exact similarity
        if self._vectors is not None:
            flat = vector.flatten().astype(np.float32)
            norm = np.linalg.norm(flat)
            if norm > 0:
                flat = flat / norm
            self._vectors[vector_id] = flat

        return fingerprint

    def query(
        self,
        vector: np.ndarray,
        max_candidates: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> list[SimilarityResult]:
        """Find similar vectors.

        Args:
            vector: Query vector
            max_candidates: Maximum results to return
            threshold: Hamming threshold as fraction (overrides config)

        Returns:
            List of SimilarityResult sorted by estimated similarity
        """
        if vector.size != self.vector_dim:
            raise ValueError(
                f"Vector dimension {vector.size} doesn't match "
                f"index dimension {self.vector_dim}"
            )

        query_fp = self._compute_fingerprint(vector)
        max_cand = max_candidates or self.config.max_candidates
        thresh = threshold or self.config.hamming_threshold
        max_distance = int(self.config.num_bits * thresh)

        # Collect candidates from all tables
        candidates: set[str] = set()
        for table_idx, bit_indices in enumerate(self._table_bit_indices):
            bucket_hash = query_fp.to_bucket_hash(bit_indices)
            if bucket_hash in self._tables[table_idx]:
                candidates.update(self._tables[table_idx][bucket_hash])

        # Score candidates by Hamming distance
        results: list[SimilarityResult] = []
        for vector_id in candidates:
            fp = self._fingerprints.get(vector_id)
            if fp is None:
                continue

            distance = query_fp.hamming_distance(fp)
            if distance <= max_distance:
                estimated_sim = query_fp.estimated_cosine_similarity(fp)

                # Compute actual similarity if vectors are stored
                actual_sim = None
                if self._vectors is not None and vector_id in self._vectors:
                    query_flat = vector.flatten().astype(np.float32)
                    query_norm = np.linalg.norm(query_flat)
                    if query_norm > 0:
                        query_flat = query_flat / query_norm
                    actual_sim = float(np.dot(query_flat, self._vectors[vector_id]))

                results.append(
                    SimilarityResult(
                        vector_id=vector_id,
                        hamming_distance=distance,
                        estimated_similarity=estimated_sim,
                        actual_similarity=actual_sim,
                    )
                )

        # Sort by estimated similarity (descending)
        results.sort(key=lambda r: r.estimated_similarity, reverse=True)
        return results[:max_cand]

    def query_by_id(
        self,
        vector_id: str,
        max_candidates: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> list[SimilarityResult]:
        """Find vectors similar to an already-indexed vector.

        Args:
            vector_id: ID of the query vector (must be in index)
            max_candidates: Maximum results to return
            threshold: Hamming threshold as fraction

        Returns:
            List of SimilarityResult (excluding the query vector itself)
        """
        query_fp = self._fingerprints.get(vector_id)
        if query_fp is None:
            raise ValueError(f"Vector {vector_id} not in index")

        max_cand = max_candidates or self.config.max_candidates
        thresh = threshold or self.config.hamming_threshold
        max_distance = int(self.config.num_bits * thresh)

        # Collect candidates from all tables
        candidates: set[str] = set()
        for table_idx, bit_indices in enumerate(self._table_bit_indices):
            bucket_hash = query_fp.to_bucket_hash(bit_indices)
            if bucket_hash in self._tables[table_idx]:
                candidates.update(self._tables[table_idx][bucket_hash])

        # Remove self
        candidates.discard(vector_id)

        # Score candidates
        results: list[SimilarityResult] = []
        for cand_id in candidates:
            fp = self._fingerprints.get(cand_id)
            if fp is None:
                continue

            distance = query_fp.hamming_distance(fp)
            if distance <= max_distance:
                estimated_sim = query_fp.estimated_cosine_similarity(fp)

                actual_sim = None
                if (
                    self._vectors is not None
                    and vector_id in self._vectors
                    and cand_id in self._vectors
                ):
                    actual_sim = float(
                        np.dot(self._vectors[vector_id], self._vectors[cand_id])
                    )

                results.append(
                    SimilarityResult(
                        vector_id=cand_id,
                        hamming_distance=distance,
                        estimated_similarity=estimated_sim,
                        actual_similarity=actual_sim,
                    )
                )

        results.sort(key=lambda r: r.estimated_similarity, reverse=True)
        return results[:max_cand]

    def get_fingerprint(self, vector_id: str) -> Optional[Fingerprint]:
        """Get the fingerprint for a vector ID."""
        return self._fingerprints.get(vector_id)

    def remove(self, vector_id: str) -> bool:
        """Remove a vector from the index.

        Args:
            vector_id: ID to remove

        Returns:
            True if found and removed
        """
        fp = self._fingerprints.pop(vector_id, None)
        if fp is None:
            return False

        # Remove from all tables
        for table_idx, bit_indices in enumerate(self._table_bit_indices):
            bucket_hash = fp.to_bucket_hash(bit_indices)
            if bucket_hash in self._tables[table_idx]:
                self._tables[table_idx][bucket_hash].discard(vector_id)
                if not self._tables[table_idx][bucket_hash]:
                    del self._tables[table_idx][bucket_hash]

        # Remove stored vector if applicable
        if self._vectors is not None:
            self._vectors.pop(vector_id, None)

        return True

    def clear(self) -> None:
        """Clear all vectors from the index."""
        self._fingerprints.clear()
        for table in self._tables:
            table.clear()
        if self._vectors is not None:
            self._vectors.clear()

    def __len__(self) -> int:
        return len(self._fingerprints)

    def __contains__(self, vector_id: str) -> bool:
        return vector_id in self._fingerprints

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        bucket_sizes = []
        for table in self._tables:
            for bucket in table.values():
                bucket_sizes.append(len(bucket))

        return {
            "num_vectors": len(self._fingerprints),
            "vector_dim": self.vector_dim,
            "num_bits": self.config.num_bits,
            "num_tables": self.config.num_tables,
            "bits_per_table": self.config.bits_per_table,
            "total_buckets": sum(len(t) for t in self._tables),
            "avg_bucket_size": np.mean(bucket_sizes) if bucket_sizes else 0,
            "max_bucket_size": max(bucket_sizes) if bucket_sizes else 0,
            "stores_vectors": self._vectors is not None,
        }


class MultiDimSimilarityIndex:
    """Similarity index that handles vectors of different dimensions.

    Maintains separate indices per dimension since LSH requires fixed dimensions.
    """

    def __init__(self, config: Optional[SimilarityIndexConfig] = None):
        """Initialize multi-dimensional similarity index.

        Args:
            config: Shared configuration for all dimension-specific indices
        """
        self.config = config or SimilarityIndexConfig()
        self._indices: dict[int, SimilarityIndex] = {}
        self._id_to_dim: dict[str, int] = {}

    def insert(self, vector: np.ndarray, vector_id: str) -> Fingerprint:
        """Insert a vector, automatically handling its dimension.

        Args:
            vector: Vector to insert
            vector_id: Unique identifier

        Returns:
            The computed fingerprint
        """
        dim = vector.size

        if dim not in self._indices:
            self._indices[dim] = SimilarityIndex(dim, self.config)

        self._id_to_dim[vector_id] = dim
        return self._indices[dim].insert(vector, vector_id)

    def query(
        self,
        vector: np.ndarray,
        max_candidates: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> list[SimilarityResult]:
        """Find similar vectors of the same dimension.

        Args:
            vector: Query vector
            max_candidates: Maximum results to return
            threshold: Hamming threshold as fraction

        Returns:
            List of SimilarityResult sorted by similarity
        """
        dim = vector.size
        if dim not in self._indices:
            return []
        return self._indices[dim].query(vector, max_candidates, threshold)

    def query_by_id(
        self,
        vector_id: str,
        max_candidates: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> list[SimilarityResult]:
        """Find vectors similar to an indexed vector.

        Args:
            vector_id: ID of query vector
            max_candidates: Maximum results
            threshold: Hamming threshold

        Returns:
            List of similar vectors
        """
        dim = self._id_to_dim.get(vector_id)
        if dim is None or dim not in self._indices:
            return []
        return self._indices[dim].query_by_id(vector_id, max_candidates, threshold)

    def get_fingerprint(self, vector_id: str) -> Optional[Fingerprint]:
        """Get fingerprint for a vector ID."""
        dim = self._id_to_dim.get(vector_id)
        if dim is None or dim not in self._indices:
            return None
        return self._indices[dim].get_fingerprint(vector_id)

    def remove(self, vector_id: str) -> bool:
        """Remove a vector from the index."""
        dim = self._id_to_dim.pop(vector_id, None)
        if dim is None or dim not in self._indices:
            return False
        return self._indices[dim].remove(vector_id)

    def clear(self) -> None:
        """Clear all indices."""
        for index in self._indices.values():
            index.clear()
        self._indices.clear()
        self._id_to_dim.clear()

    def __len__(self) -> int:
        return len(self._id_to_dim)

    def __contains__(self, vector_id: str) -> bool:
        return vector_id in self._id_to_dim

    def get_stats(self) -> dict[str, Any]:
        """Get statistics across all dimension indices."""
        return {
            "num_dimensions": len(self._indices),
            "total_vectors": len(self._id_to_dim),
            "per_dimension": {
                dim: index.get_stats() for dim, index in self._indices.items()
            },
        }


# Convenience aliases for backward compatibility
UnifiedLSHIndex = MultiDimSimilarityIndex
