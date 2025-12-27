"""SimHash implementation for fast similarity detection of neural network weights.

SimHash is a locality-sensitive hashing technique that generates compact binary
fingerprints where similar vectors produce similar fingerprints. This enables
O(1) similarity detection via Hamming distance comparison.

Key properties:
- Fixed-size fingerprint (64 or 128 bits)
- Similar vectors have small Hamming distance
- Hamming distance approximates angular distance
- Can be stored and compared with O(1) operations

Reference: Charikar, M. (2002). "Similarity estimation techniques from
rounding algorithms". STOC '02.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SimHashConfig:
    """Configuration for SimHash fingerprinting."""

    # Number of bits in the fingerprint (64 or 128)
    num_bits: int = 64

    # Number of random hyperplanes (same as num_bits for standard SimHash)
    # More hyperplanes = more accurate but slower
    num_hyperplanes: Optional[int] = None

    # Random seed for reproducibility
    seed: int = 42

    # Hamming distance threshold for similarity (as fraction of num_bits)
    # E.g., 0.1 means 10% of bits can differ and still be similar
    similarity_threshold: float = 0.1

    def __post_init__(self):
        if self.num_hyperplanes is None:
            self.num_hyperplanes = self.num_bits


class SimHash:
    """SimHash fingerprinting for fast similarity detection.

    SimHash works by:
    1. Projecting vectors onto random hyperplanes
    2. Creating a binary signature based on which side of each hyperplane
    3. Similar vectors produce similar signatures (small Hamming distance)

    This implementation supports:
    - 64-bit and 128-bit fingerprints
    - Efficient batch processing
    - Hamming distance computation
    - Similarity detection with configurable threshold
    """

    def __init__(self, config: Optional[SimHashConfig] = None):
        """Initialize SimHash with configuration.

        Args:
            config: SimHashConfig with fingerprint parameters
        """
        self.config = config or SimHashConfig()
        self._hyperplanes: Optional[np.ndarray] = None
        self._hyperplane_dim: Optional[int] = None
        self._rng = np.random.RandomState(self.config.seed)

    def _initialize_hyperplanes(self, dim: int) -> None:
        """Initialize random hyperplanes for the given dimension.

        Args:
            dim: Dimension of input vectors
        """
        if self._hyperplanes is not None and self._hyperplane_dim == dim:
            return

        # Generate random unit vectors as hyperplane normals
        self._hyperplanes = self._rng.randn(self.config.num_hyperplanes, dim)

        # Normalize to unit vectors
        norms = np.linalg.norm(self._hyperplanes, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        self._hyperplanes = self._hyperplanes / norms

        self._hyperplane_dim = dim
        logger.debug(
            f"Initialized {self.config.num_hyperplanes} hyperplanes for dim={dim}"
        )

    def compute_fingerprint(
        self, vector: np.ndarray
    ) -> Union[np.uint64, Tuple[np.uint64, np.uint64]]:
        """Compute SimHash fingerprint for a vector.

        Args:
            vector: Input vector (will be flattened)

        Returns:
            64-bit or 128-bit fingerprint as uint64 (or tuple of two uint64)
        """
        # Flatten and normalize
        flat = vector.flatten().astype(np.float32)
        norm = np.linalg.norm(flat)
        if norm > 0:
            flat = flat / norm

        # Initialize hyperplanes if needed
        self._initialize_hyperplanes(len(flat))

        # Project onto hyperplanes
        projections = np.dot(self._hyperplanes, flat)

        # Create binary signature (1 if projection > 0, else 0)
        bits = (projections > 0).astype(np.uint8)

        # Pack into integers
        return self._pack_bits(bits)

    def compute_fingerprint_batch(
        self, vectors: List[np.ndarray]
    ) -> List[Union[np.uint64, Tuple[np.uint64, np.uint64]]]:
        """Compute fingerprints for multiple vectors efficiently.

        Args:
            vectors: List of input vectors

        Returns:
            List of fingerprints
        """
        if not vectors:
            return []

        # Flatten and normalize all vectors
        flat_vectors = []
        for v in vectors:
            flat = v.flatten().astype(np.float32)
            norm = np.linalg.norm(flat)
            if norm > 0:
                flat = flat / norm
            flat_vectors.append(flat)

        # Stack for batch processing
        stacked = np.vstack(flat_vectors)

        # Initialize hyperplanes if needed
        self._initialize_hyperplanes(stacked.shape[1])

        # Batch projection: (num_vectors, dim) @ (dim, num_hyperplanes).T
        projections = np.dot(stacked, self._hyperplanes.T)

        # Create binary signatures
        bits_batch = (projections > 0).astype(np.uint8)

        # Pack each into integers
        return [self._pack_bits(bits) for bits in bits_batch]

    def _pack_bits(
        self, bits: np.ndarray
    ) -> Union[np.uint64, Tuple[np.uint64, np.uint64]]:
        """Pack binary array into uint64 fingerprint(s).

        Args:
            bits: Binary array of length num_bits

        Returns:
            Packed fingerprint
        """
        if self.config.num_bits <= 64:
            # Pack into single uint64
            result = np.uint64(0)
            for i, bit in enumerate(bits[: self.config.num_bits]):
                if bit:
                    result |= np.uint64(1) << np.uint64(i)
            return result
        else:
            # Pack into two uint64s
            low = np.uint64(0)
            high = np.uint64(0)
            for i in range(min(64, len(bits))):
                if bits[i]:
                    low |= np.uint64(1) << np.uint64(i)
            for i in range(64, min(128, len(bits))):
                if bits[i]:
                    high |= np.uint64(1) << np.uint64(i - 64)
            return (low, high)

    @staticmethod
    def hamming_distance(
        fp1: Union[np.uint64, Tuple[np.uint64, np.uint64]],
        fp2: Union[np.uint64, Tuple[np.uint64, np.uint64]],
    ) -> int:
        """Compute Hamming distance between two fingerprints.

        Args:
            fp1: First fingerprint
            fp2: Second fingerprint

        Returns:
            Number of differing bits
        """
        if isinstance(fp1, tuple):
            # 128-bit fingerprints
            xor_low = fp1[0] ^ fp2[0]
            xor_high = fp1[1] ^ fp2[1]
            return bin(int(xor_low)).count("1") + bin(int(xor_high)).count("1")
        else:
            # 64-bit fingerprints
            xor = fp1 ^ fp2
            return bin(int(xor)).count("1")

    def are_similar(
        self,
        fp1: Union[np.uint64, Tuple[np.uint64, np.uint64]],
        fp2: Union[np.uint64, Tuple[np.uint64, np.uint64]],
        threshold: Optional[float] = None,
    ) -> bool:
        """Check if two fingerprints indicate similar vectors.

        Args:
            fp1: First fingerprint
            fp2: Second fingerprint
            threshold: Optional override for similarity threshold

        Returns:
            True if Hamming distance is below threshold
        """
        thresh = threshold or self.config.similarity_threshold
        max_distance = int(self.config.num_bits * thresh)

        distance = self.hamming_distance(fp1, fp2)
        return distance <= max_distance

    def estimated_similarity(
        self,
        fp1: Union[np.uint64, Tuple[np.uint64, np.uint64]],
        fp2: Union[np.uint64, Tuple[np.uint64, np.uint64]],
    ) -> float:
        """Estimate cosine similarity from fingerprint distance.

        The relationship between Hamming distance and cosine similarity is:
        cos(θ) ≈ cos(π * d / num_bits)

        where d is the Hamming distance.

        Args:
            fp1: First fingerprint
            fp2: Second fingerprint

        Returns:
            Estimated cosine similarity in [-1, 1]
        """
        distance = self.hamming_distance(fp1, fp2)
        # Hamming distance / num_bits approximates θ/π
        theta = np.pi * distance / self.config.num_bits
        return float(np.cos(theta))


class SimHashIndex:
    """Index for fast similarity search using SimHash fingerprints.

    This index stores fingerprints and enables O(1) average-case lookup
    for similar vectors using Hamming distance.
    """

    def __init__(self, config: Optional[SimHashConfig] = None):
        """Initialize SimHash index.

        Args:
            config: SimHashConfig with fingerprint parameters
        """
        self.config = config or SimHashConfig()
        self.hasher = SimHash(self.config)

        # Storage: fingerprint -> list of (id, vector_hash)
        self._fingerprints: Dict[
            Union[np.uint64, Tuple[np.uint64, np.uint64]], List[str]
        ] = {}

        # ID to fingerprint mapping for O(1) lookup
        self._id_to_fingerprint: Dict[
            str, Union[np.uint64, Tuple[np.uint64, np.uint64]]
        ] = {}

        # Statistics
        self._num_vectors = 0

    def insert(self, vector: np.ndarray, vector_id: str) -> None:
        """Insert a vector into the index.

        Args:
            vector: Vector to insert
            vector_id: Unique identifier for the vector
        """
        fingerprint = self.hasher.compute_fingerprint(vector)

        if fingerprint not in self._fingerprints:
            self._fingerprints[fingerprint] = []
        self._fingerprints[fingerprint].append(vector_id)

        self._id_to_fingerprint[vector_id] = fingerprint
        self._num_vectors += 1

    def query(
        self,
        vector: np.ndarray,
        max_candidates: int = 100,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, int]]:
        """Find similar vectors in the index.

        Args:
            vector: Query vector
            max_candidates: Maximum number of candidates to return
            threshold: Optional override for similarity threshold

        Returns:
            List of (vector_id, hamming_distance) tuples, sorted by distance
        """
        query_fp = self.hasher.compute_fingerprint(vector)
        thresh = threshold or self.config.similarity_threshold
        max_distance = int(self.config.num_bits * thresh)

        candidates = []

        for fp, ids in self._fingerprints.items():
            distance = self.hasher.hamming_distance(query_fp, fp)
            if distance <= max_distance:
                for vid in ids:
                    candidates.append((vid, distance))

        # Sort by distance and limit
        candidates.sort(key=lambda x: x[1])
        return candidates[:max_candidates]

    def query_by_fingerprint(
        self,
        fingerprint: Union[np.uint64, Tuple[np.uint64, np.uint64]],
        max_candidates: int = 100,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, int]]:
        """Find similar vectors using a precomputed fingerprint.

        Args:
            fingerprint: Precomputed SimHash fingerprint
            max_candidates: Maximum number of candidates to return
            threshold: Optional override for similarity threshold

        Returns:
            List of (vector_id, hamming_distance) tuples, sorted by distance
        """
        thresh = threshold or self.config.similarity_threshold
        max_distance = int(self.config.num_bits * thresh)

        candidates = []

        for fp, ids in self._fingerprints.items():
            distance = self.hasher.hamming_distance(fingerprint, fp)
            if distance <= max_distance:
                for vid in ids:
                    candidates.append((vid, distance))

        candidates.sort(key=lambda x: x[1])
        return candidates[:max_candidates]

    def remove(self, vector_id: str) -> bool:
        """Remove a vector from the index.

        Args:
            vector_id: ID of vector to remove

        Returns:
            True if vector was found and removed
        """
        if vector_id not in self._id_to_fingerprint:
            return False

        fingerprint = self._id_to_fingerprint[vector_id]

        if fingerprint in self._fingerprints:
            self._fingerprints[fingerprint].remove(vector_id)
            if not self._fingerprints[fingerprint]:
                del self._fingerprints[fingerprint]

        del self._id_to_fingerprint[vector_id]
        self._num_vectors -= 1
        return True

    def clear(self) -> None:
        """Clear all vectors from the index."""
        self._fingerprints.clear()
        self._id_to_fingerprint.clear()
        self._num_vectors = 0

    @property
    def size(self) -> int:
        """Return number of vectors in the index."""
        return self._num_vectors

    def get_stats(self) -> Dict[str, int]:
        """Get index statistics.

        Returns:
            Dictionary with index statistics
        """
        return {
            "num_vectors": self._num_vectors,
            "num_unique_fingerprints": len(self._fingerprints),
            "fingerprint_bits": self.config.num_bits,
            "avg_vectors_per_fingerprint": (
                self._num_vectors / len(self._fingerprints) if self._fingerprints else 0
            ),
        }


class MultiDimSimHashIndex:
    """SimHash index that handles vectors of different dimensions.

    Maintains separate SimHash indices per dimension, similar to MultiDimLSHIndex.
    """

    def __init__(self, config: Optional[SimHashConfig] = None):
        """Initialize multi-dimensional SimHash index.

        Args:
            config: SimHashConfig with fingerprint parameters
        """
        self.config = config or SimHashConfig()
        self._indices: Dict[int, SimHashIndex] = {}

    def _get_or_create_index(self, dim: int) -> SimHashIndex:
        """Get or create index for given dimension."""
        if dim not in self._indices:
            self._indices[dim] = SimHashIndex(self.config)
        return self._indices[dim]

    def insert(self, vector: np.ndarray, vector_id: str) -> None:
        """Insert a vector into the appropriate dimension-specific index.

        Args:
            vector: Vector to insert
            vector_id: Unique identifier for the vector
        """
        dim = vector.size
        index = self._get_or_create_index(dim)
        index.insert(vector, vector_id)

    def query(
        self,
        vector: np.ndarray,
        max_candidates: int = 100,
        threshold: Optional[float] = None,
    ) -> List[str]:
        """Find similar vectors with matching dimension.

        Args:
            vector: Query vector
            max_candidates: Maximum number of candidates to return
            threshold: Optional override for similarity threshold

        Returns:
            List of vector_ids sorted by similarity
        """
        dim = vector.size
        if dim not in self._indices:
            return []

        results = self._indices[dim].query(vector, max_candidates, threshold)
        return [vid for vid, _ in results]

    def remove(self, vector: np.ndarray, vector_id: str) -> bool:
        """Remove a vector from the index.

        Args:
            vector: Vector to remove (needed for dimension)
            vector_id: ID of vector to remove

        Returns:
            True if vector was found and removed
        """
        dim = vector.size
        if dim not in self._indices:
            return False
        return self._indices[dim].remove(vector_id)

    def clear(self) -> None:
        """Clear all vectors from all dimension indices."""
        for index in self._indices.values():
            index.clear()
        self._indices.clear()

    @property
    def size(self) -> int:
        """Return total number of vectors across all dimensions."""
        return sum(index.size for index in self._indices.values())

    def get_stats(self) -> Dict[str, any]:
        """Get index statistics across all dimensions.

        Returns:
            Dictionary with aggregated statistics
        """
        return {
            "total_vectors": self.size,
            "num_dimensions": len(self._indices),
            "dimension_stats": {
                dim: index.get_stats() for dim, index in self._indices.items()
            },
        }
