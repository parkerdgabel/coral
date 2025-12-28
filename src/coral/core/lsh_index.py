"""Locality Sensitive Hashing (LSH) for efficient similarity search.

This module provides an LSH-based index for finding similar weight tensors
in O(1) average time instead of O(n) linear scan. It uses random hyperplane
hashing for cosine similarity.

The algorithm:
1. Generate k random hyperplanes (normal vectors)
2. For each weight tensor, compute which side of each hyperplane it falls on
3. This gives a k-bit hash where similar vectors tend to have the same hash
4. Store vectors in hash buckets
5. To find similar vectors, only search in the same bucket (and nearby buckets)

Trade-offs:
- More hyperplanes (k) = fewer false positives, but more false negatives
- More hash tables (L) = fewer false negatives, but more memory
- Default (k=8, L=4) works well for similarity threshold >= 0.9
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LSHConfig:
    """Configuration for LSH index."""

    # Number of hyperplanes per hash table (bits per hash)
    num_hyperplanes: int = 8

    # Number of hash tables (for reducing false negatives)
    num_tables: int = 4

    # Random seed for reproducibility
    seed: Optional[int] = 42

    # Maximum number of candidates to return from lookup
    max_candidates: int = 100


@dataclass
class LSHTable:
    """A single LSH hash table."""

    hyperplanes: np.ndarray  # Shape: (num_hyperplanes, vector_dim)
    buckets: dict[int, set[str]] = field(default_factory=dict)

    def hash_vector(self, vector: np.ndarray) -> int:
        """Compute hash for a vector using hyperplane hashing."""
        # Dot product with each hyperplane
        projections = np.dot(self.hyperplanes, vector.flatten())
        # Convert to bits: 1 if positive, 0 if negative
        bits = (projections > 0).astype(np.int32)
        # Convert bit array to integer
        hash_val = 0
        for bit in bits:
            hash_val = (hash_val << 1) | bit
        return hash_val

    def insert(self, vector: np.ndarray, key: str) -> int:
        """Insert a vector with its key, return bucket hash."""
        bucket_hash = self.hash_vector(vector)
        if bucket_hash not in self.buckets:
            self.buckets[bucket_hash] = set()
        self.buckets[bucket_hash].add(key)
        return bucket_hash

    def query(self, vector: np.ndarray) -> set[str]:
        """Find all keys in the same bucket as the query vector."""
        bucket_hash = self.hash_vector(vector)
        return self.buckets.get(bucket_hash, set()).copy()

    def remove(self, vector: np.ndarray, key: str) -> bool:
        """Remove a key from its bucket. Returns True if found."""
        bucket_hash = self.hash_vector(vector)
        if bucket_hash in self.buckets and key in self.buckets[bucket_hash]:
            self.buckets[bucket_hash].discard(key)
            if not self.buckets[bucket_hash]:
                del self.buckets[bucket_hash]
            return True
        return False


class LSHIndex:
    """Locality Sensitive Hashing index for fast similarity search.

    This index enables O(1) average-time similarity lookups by hashing
    similar vectors to the same buckets with high probability.

    Example:
        >>> index = LSHIndex(vector_dim=1000)
        >>> index.insert(weight_a.data, "weight_a")
        >>> index.insert(weight_b.data, "weight_b")
        >>> candidates = index.query(query_weight.data)
        >>> # Only need to compare against candidates, not all weights
    """

    def __init__(
        self,
        vector_dim: int,
        config: Optional[LSHConfig] = None,
    ):
        """Initialize LSH index.

        Args:
            vector_dim: Dimension of vectors to index (flattened weight size)
            config: LSH configuration parameters
        """
        self.vector_dim = vector_dim
        self.config = config or LSHConfig()

        # Initialize random state
        self.rng = np.random.RandomState(self.config.seed)

        # Create hash tables with random hyperplanes
        self.tables: list[LSHTable] = []
        for _ in range(self.config.num_tables):
            # Random hyperplanes from standard normal distribution
            hyperplanes = self.rng.randn(
                self.config.num_hyperplanes, vector_dim
            ).astype(np.float32)
            # Normalize hyperplanes for numerical stability
            norms = np.linalg.norm(hyperplanes, axis=1, keepdims=True)
            hyperplanes = hyperplanes / (norms + 1e-10)
            self.tables.append(LSHTable(hyperplanes=hyperplanes))

        # Track inserted keys for stats
        self.keys: set[str] = set()
        self._key_to_hashes: dict[str, list[int]] = {}

    def insert(self, vector: np.ndarray, key: str) -> None:
        """Insert a vector into the index.

        Args:
            vector: Weight tensor data (will be flattened)
            key: Unique identifier for this vector (e.g., hash)
        """
        flat_vector = vector.flatten().astype(np.float32)

        if flat_vector.shape[0] != self.vector_dim:
            raise ValueError(
                f"Vector dimension {flat_vector.shape[0]} doesn't match "
                f"index dimension {self.vector_dim}"
            )

        # Normalize vector for cosine similarity
        norm = np.linalg.norm(flat_vector)
        if norm > 0:
            flat_vector = flat_vector / norm

        # Insert into all tables
        hashes = []
        for table in self.tables:
            h = table.insert(flat_vector, key)
            hashes.append(h)

        self.keys.add(key)
        self._key_to_hashes[key] = hashes

    def query(
        self,
        vector: np.ndarray,
        max_candidates: Optional[int] = None,
    ) -> set[str]:
        """Find candidate similar vectors.

        Args:
            vector: Query vector (will be flattened)
            max_candidates: Maximum candidates to return (None = use config)

        Returns:
            Set of candidate keys that might be similar
        """
        flat_vector = vector.flatten().astype(np.float32)

        if flat_vector.shape[0] != self.vector_dim:
            raise ValueError(
                f"Vector dimension {flat_vector.shape[0]} doesn't match "
                f"index dimension {self.vector_dim}"
            )

        # Normalize for cosine similarity
        norm = np.linalg.norm(flat_vector)
        if norm > 0:
            flat_vector = flat_vector / norm

        # Union of candidates from all tables
        candidates: set[str] = set()
        for table in self.tables:
            candidates.update(table.query(flat_vector))

        # Limit candidates if requested
        max_cand = max_candidates or self.config.max_candidates
        if len(candidates) > max_cand:
            # Random sample to limit
            candidates = set(list(candidates)[:max_cand])

        return candidates

    def remove(self, vector: np.ndarray, key: str) -> bool:
        """Remove a vector from the index.

        Args:
            vector: The vector data
            key: Key to remove

        Returns:
            True if key was found and removed
        """
        if key not in self.keys:
            return False

        flat_vector = vector.flatten().astype(np.float32)
        norm = np.linalg.norm(flat_vector)
        if norm > 0:
            flat_vector = flat_vector / norm

        for table in self.tables:
            table.remove(flat_vector, key)

        self.keys.discard(key)
        self._key_to_hashes.pop(key, None)
        return True

    def clear(self) -> None:
        """Clear all vectors from the index."""
        for table in self.tables:
            table.buckets.clear()
        self.keys.clear()
        self._key_to_hashes.clear()

    def __len__(self) -> int:
        """Return number of indexed vectors."""
        return len(self.keys)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the index."""
        bucket_sizes = []
        for table in self.tables:
            for bucket in table.buckets.values():
                bucket_sizes.append(len(bucket))

        return {
            "num_vectors": len(self.keys),
            "num_tables": len(self.tables),
            "num_hyperplanes": self.config.num_hyperplanes,
            "total_buckets": sum(len(t.buckets) for t in self.tables),
            "avg_bucket_size": np.mean(bucket_sizes) if bucket_sizes else 0,
            "max_bucket_size": max(bucket_sizes) if bucket_sizes else 0,
            "vector_dim": self.vector_dim,
        }


class MultiDimLSHIndex:
    """LSH index that handles vectors of different dimensions.

    Since LSH requires fixed dimensions, this maintains separate indices
    for each unique dimension encountered.
    """

    def __init__(self, config: Optional[LSHConfig] = None):
        """Initialize multi-dimensional LSH index.

        Args:
            config: LSH configuration (shared across all dimension-specific indices)
        """
        self.config = config or LSHConfig()
        self.indices: dict[int, LSHIndex] = {}
        self.key_to_dim: dict[str, int] = {}

    def insert(self, vector: np.ndarray, key: str) -> None:
        """Insert a vector, automatically handling its dimension."""
        dim = vector.size

        if dim not in self.indices:
            self.indices[dim] = LSHIndex(dim, self.config)

        self.indices[dim].insert(vector, key)
        self.key_to_dim[key] = dim

    def query(
        self,
        vector: np.ndarray,
        max_candidates: Optional[int] = None,
    ) -> set[str]:
        """Query for similar vectors of the same dimension."""
        dim = vector.size

        if dim not in self.indices:
            return set()

        return self.indices[dim].query(vector, max_candidates)

    def remove(self, vector: np.ndarray, key: str) -> bool:
        """Remove a vector from the index."""
        if key not in self.key_to_dim:
            return False

        dim = self.key_to_dim[key]
        if dim in self.indices:
            result = self.indices[dim].remove(vector, key)
            if result:
                del self.key_to_dim[key]
            return result
        return False

    def clear(self) -> None:
        """Clear all indices."""
        for index in self.indices.values():
            index.clear()
        self.indices.clear()
        self.key_to_dim.clear()

    def __len__(self) -> int:
        """Return total number of indexed vectors."""
        return len(self.key_to_dim)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about all indices."""
        return {
            "num_dimensions": len(self.indices),
            "total_vectors": len(self.key_to_dim),
            "per_dimension": {
                dim: index.get_stats() for dim, index in self.indices.items()
            },
        }
