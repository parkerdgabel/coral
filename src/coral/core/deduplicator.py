"""Core deduplication engine for weight tensors"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from coral.core.weight_tensor import WeightTensor
from coral.delta.delta_encoder import Delta, DeltaConfig, DeltaEncoder

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationStats:
    """Statistics about deduplication results"""

    total_weights: int = 0
    unique_weights: int = 0
    duplicate_weights: int = 0
    similar_weights: int = 0
    bytes_saved: int = 0
    compression_ratio: float = 0.0

    def update(self, original_bytes: int, deduplicated_bytes: int):
        """Update compression statistics"""
        self.bytes_saved = original_bytes - deduplicated_bytes
        if original_bytes > 0:
            self.compression_ratio = self.bytes_saved / original_bytes


@dataclass
class WeightGroup:
    """Group of weights that are identical or similar"""

    reference_hash: str
    reference_weight: WeightTensor
    duplicates: List[Tuple[str, WeightTensor]] = field(default_factory=list)
    similar: List[Tuple[str, WeightTensor, float]] = field(
        default_factory=list
    )  # (name, weight, similarity)
    deltas: Dict[str, Delta] = field(
        default_factory=dict
    )  # name -> delta for similar weights

    @property
    def total_count(self) -> int:
        """Total number of weights in this group"""
        return 1 + len(self.duplicates) + len(self.similar)

    @property
    def bytes_saved(self) -> int:
        """Bytes saved by deduplication in this group"""
        ref_bytes = self.reference_weight.nbytes
        # Exact duplicates save full size
        duplicate_savings = ref_bytes * len(self.duplicates)
        # Similar weights use delta encoding - calculate actual savings
        similar_savings = 0
        for name, weight, _ in self.similar:
            original_size = weight.nbytes
            if name in self.deltas:
                delta_size = self.deltas[name].nbytes
                similar_savings += original_size - delta_size
            else:
                # Estimate if delta not computed yet
                similar_savings += int(original_size * 0.5)
        return duplicate_savings + similar_savings


class Deduplicator:
    """
    Core deduplication engine for neural network weights.

    Supports:
    - Exact deduplication through content hashing
    - Similarity-based deduplication with lossless delta encoding
    - Reference counting
    - Deduplication statistics
    """

    def __init__(
        self,
        similarity_threshold: float = 0.99,
        delta_config: Optional[DeltaConfig] = None,
        enable_delta_encoding: bool = True,
    ):
        """
        Initialize the deduplicator.

        Args:
            similarity_threshold: Threshold for considering weights similar (0-1)
            delta_config: Configuration for delta encoding
            enable_delta_encoding: Whether to use delta encoding for similar weights
        """
        self.similarity_threshold = similarity_threshold
        self.enable_delta_encoding = enable_delta_encoding
        self.delta_encoder = (
            DeltaEncoder(delta_config or DeltaConfig())
            if enable_delta_encoding
            else None
        )

        self.weight_index: Dict[str, WeightTensor] = {}  # hash -> weight
        self.weight_groups: Dict[str, WeightGroup] = {}  # reference_hash -> group
        self.name_to_hash: Dict[str, str] = {}  # weight name -> hash
        self.name_to_delta: Dict[
            str, str
        ] = {}  # weight name -> delta hash (for similar weights)
        self.delta_index: Dict[str, Delta] = {}  # delta hash -> delta object
        self.stats = DeduplicationStats()

        # Thread safety lock for concurrent access
        self._lock = threading.RLock()

    def add_weight(self, weight: WeightTensor, name: Optional[str] = None) -> str:
        """
        Add a weight to the deduplicator and check for duplicates.

        This method is thread-safe.

        Args:
            weight: WeightTensor to add
            name: Optional name for the weight

        Returns:
            Hash of the weight (or reference weight if duplicate/similar)
        """
        if name is None:
            name = weight.metadata.name

        # Compute hash outside the lock (CPU-intensive, doesn't need synchronization)
        weight_hash = weight.compute_hash()

        with self._lock:
            # Check for exact duplicate
            if weight_hash in self.weight_index:
                # Exact duplicate found
                self._add_duplicate(weight_hash, name, weight)
                return weight_hash

            # Check for similar weights
            similar_ref = self._find_similar_weight(weight)
            if similar_ref:
                # Similar weight found
                self._add_similar(similar_ref, name, weight)
                return similar_ref

            # New unique weight
            self._add_unique_weight(weight_hash, name, weight)
            return weight_hash

    def _add_duplicate(self, ref_hash: str, name: str, weight: WeightTensor):
        """Add an exact duplicate to existing group"""
        if ref_hash not in self.weight_groups:
            # Create group if it doesn't exist
            self.weight_groups[ref_hash] = WeightGroup(
                reference_hash=ref_hash, reference_weight=self.weight_index[ref_hash]
            )

        self.weight_groups[ref_hash].duplicates.append((name, weight))
        self.name_to_hash[name] = ref_hash
        self.stats.duplicate_weights += 1
        logger.debug(f"Found exact duplicate: {name} -> {ref_hash}")

    def _add_similar(self, ref_hash: str, name: str, weight: WeightTensor):
        """Add a similar weight to existing group"""
        ref_weight = self.weight_index[ref_hash]
        similarity = self._compute_similarity(weight, ref_weight)

        if ref_hash not in self.weight_groups:
            self.weight_groups[ref_hash] = WeightGroup(
                reference_hash=ref_hash, reference_weight=ref_weight
            )

        group = self.weight_groups[ref_hash]
        group.similar.append((name, weight, similarity))

        # Create delta encoding if enabled
        if self.enable_delta_encoding and self.delta_encoder:
            try:
                if self.delta_encoder.can_encode_as_delta(weight, ref_weight):
                    delta = self.delta_encoder.encode_delta(weight, ref_weight)
                    delta_hash = self._compute_delta_hash(delta)

                    # Store delta
                    self.delta_index[delta_hash] = delta
                    group.deltas[name] = delta
                    self.name_to_delta[name] = delta_hash

                    logger.debug(
                        f"Created delta for {name}: "
                        f"{delta.compression_ratio:.2%} compression"
                    )
                else:
                    logger.debug(
                        f"Delta encoding not efficient for {name}, storing reference"
                    )
            except (ValueError, TypeError, np.linalg.LinAlgError) as e:
                # Catch specific expected errors during delta encoding:
                # - ValueError: shape/dtype mismatch, invalid delta parameters
                # - TypeError: incompatible data types
                # - LinAlgError: numerical computation errors
                logger.warning(f"Failed to create delta for {name}: {e}")

        self.name_to_hash[name] = ref_hash
        self.stats.similar_weights += 1
        logger.debug(
            f"Found similar weight: {name} -> {ref_hash} (similarity: {similarity:.4f})"
        )

    def _add_unique_weight(self, weight_hash: str, name: str, weight: WeightTensor):
        """Add a new unique weight"""
        self.weight_index[weight_hash] = weight
        self.name_to_hash[name] = weight_hash
        self.stats.unique_weights += 1
        logger.debug(f"Added unique weight: {name} ({weight_hash})")

    def _find_similar_weight(self, weight: WeightTensor) -> Optional[str]:
        """
        Find a similar weight in the index.

        Returns hash of similar weight or None.
        """
        # Only check weights with same shape and dtype
        candidates = [
            (hash_val, w)
            for hash_val, w in self.weight_index.items()
            if w.shape == weight.shape and w.dtype == weight.dtype
        ]

        # Find most similar weight above threshold
        best_similarity = self.similarity_threshold
        best_hash = None

        for hash_val, candidate in candidates:
            if weight.is_similar_to(candidate, self.similarity_threshold):
                similarity = self._compute_similarity(weight, candidate)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_hash = hash_val

        return best_hash

    def _compute_similarity(
        self, weight1: WeightTensor, weight2: WeightTensor
    ) -> float:
        """Compute cosine similarity between two weights."""
        # Import here to avoid circular dependency with utils
        from coral.utils.similarity import cosine_similarity

        return cosine_similarity(weight1.data, weight2.data)

    def get_weight_by_name(self, name: str) -> Optional[WeightTensor]:
        """Get weight by name, reconstructing from delta if needed.

        This method is thread-safe.
        """
        with self._lock:
            if name not in self.name_to_hash:
                return None

            # Check if this is a delta-encoded similar weight
            if name in self.name_to_delta and self.enable_delta_encoding:
                return self._reconstruct_from_delta(name)

            # Otherwise get the reference weight directly
            hash_val = self.name_to_hash[name]
            return self.weight_index.get(hash_val)

    def get_weight_group(self, name: str) -> Optional[WeightGroup]:
        """Get the weight group containing a named weight"""
        if name not in self.name_to_hash:
            return None

        hash_val = self.name_to_hash[name]

        # Check if it's a reference weight
        if hash_val in self.weight_groups:
            return self.weight_groups[hash_val]

        # Check all groups for this weight
        for group in self.weight_groups.values():
            for dup_name, _ in group.duplicates:
                if dup_name == name:
                    return group
            for sim_name, _, _ in group.similar:
                if sim_name == name:
                    return group

        return None

    def compute_stats(self) -> DeduplicationStats:
        """Compute and return deduplication statistics.

        This method is thread-safe.
        """
        with self._lock:
            self.stats.total_weights = len(self.name_to_hash)

            # Calculate bytes
            original_bytes = 0
            deduplicated_bytes = 0

            for name in self.name_to_hash:
                weight = self._get_weight_by_name_unlocked(name)
                if weight:
                    original_bytes += weight.nbytes

            # Count unique weights and their bytes
            for weight in self.weight_index.values():
                deduplicated_bytes += weight.nbytes

            # Add estimated delta encoding for similar weights
            for group in self.weight_groups.values():
                # Estimate 50% size for delta-encoded similar weights
                for _, weight, _ in group.similar:
                    deduplicated_bytes += weight.nbytes // 2

            self.stats.update(original_bytes, deduplicated_bytes)
            return self.stats

    def _get_weight_by_name_unlocked(self, name: str) -> Optional[WeightTensor]:
        """Get weight by name without locking (for internal use when lock is held)."""
        if name not in self.name_to_hash:
            return None

        # Check if this is a delta-encoded similar weight
        if name in self.name_to_delta and self.enable_delta_encoding:
            return self._reconstruct_from_delta(name)

        # Otherwise get the reference weight directly
        hash_val = self.name_to_hash[name]
        return self.weight_index.get(hash_val)

    def get_deduplication_report(self) -> Dict[str, Any]:
        """Get detailed deduplication report"""
        stats = self.compute_stats()

        # Find largest groups
        largest_groups = sorted(
            self.weight_groups.values(), key=lambda g: g.bytes_saved, reverse=True
        )[:10]

        return {
            "summary": {
                "total_weights": stats.total_weights,
                "unique_weights": stats.unique_weights,
                "duplicate_weights": stats.duplicate_weights,
                "similar_weights": stats.similar_weights,
                "bytes_saved": stats.bytes_saved,
                "compression_ratio": stats.compression_ratio,
            },
            "largest_groups": [
                {
                    "reference_name": group.reference_weight.metadata.name,
                    "total_weights": group.total_count,
                    "duplicates": len(group.duplicates),
                    "similar": len(group.similar),
                    "bytes_saved": group.bytes_saved,
                }
                for group in largest_groups
            ],
        }

    def clear(self):
        """Clear all stored weights and statistics.

        This method is thread-safe.
        """
        with self._lock:
            self.weight_index.clear()
            self.weight_groups.clear()
            self.name_to_hash.clear()
            self.name_to_delta.clear()
            self.delta_index.clear()
            self.stats = DeduplicationStats()

    def _compute_delta_hash(self, delta: Delta) -> str:
        """Compute hash for a delta object."""
        import xxhash

        hasher = xxhash.xxh3_64()
        hasher.update(delta.reference_hash.encode())
        hasher.update(delta.data.tobytes())
        hasher.update(str(delta.delta_type.value).encode())
        return hasher.hexdigest()

    def _reconstruct_from_delta(self, name: str) -> Optional[WeightTensor]:
        """Reconstruct original weight from delta encoding."""
        if name not in self.name_to_delta or not self.delta_encoder:
            return None

        delta_hash = self.name_to_delta[name]
        delta = self.delta_index.get(delta_hash)
        if not delta:
            logger.error(f"Delta not found for weight {name}")
            return None

        # Get reference weight
        ref_weight = self.weight_index.get(delta.reference_hash)
        if not ref_weight:
            logger.error(f"Reference weight not found for delta {delta_hash}")
            return None

        try:
            # Reconstruct original weight
            reconstructed = self.delta_encoder.decode_delta(delta, ref_weight)
            return reconstructed
        except Exception as e:
            logger.error(f"Failed to reconstruct weight {name} from delta: {e}")
            return None

    def get_delta_by_name(self, name: str) -> Optional[Delta]:
        """Get delta object by weight name."""
        if name not in self.name_to_delta:
            return None

        delta_hash = self.name_to_delta[name]
        return self.delta_index.get(delta_hash)

    def is_delta_encoded(self, name: str) -> bool:
        """Check if a weight is delta-encoded."""
        return name in self.name_to_delta

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get detailed compression statistics including delta encoding."""
        base_stats = self.compute_stats()

        # Calculate delta-specific stats
        total_deltas = len(self.delta_index)
        total_delta_size = sum(delta.nbytes for delta in self.delta_index.values())

        # Calculate original size for delta-encoded weights
        original_delta_size = 0
        for name in self.name_to_delta:
            weight = self._get_original_weight_from_group(name)
            if weight:
                original_delta_size += weight.nbytes

        # Compression ratio can be negative if delta is larger than original
        # (due to metadata overhead)
        # In this case, we show the actual expansion ratio
        delta_compression_ratio = (
            1.0 - (total_delta_size / original_delta_size)
            if original_delta_size > 0
            else 0.0
        )

        return {
            **base_stats.__dict__,
            "delta_stats": {
                "total_deltas": total_deltas,
                "total_delta_size": total_delta_size,
                "original_delta_size": original_delta_size,
                "delta_compression_ratio": delta_compression_ratio,
                "average_delta_size": total_delta_size / total_deltas
                if total_deltas > 0
                else 0,
            },
        }

    def _get_original_weight_from_group(self, name: str) -> Optional[WeightTensor]:
        """Get original weight data from weight group (before delta encoding)."""
        if name not in self.name_to_hash:
            return None

        ref_hash = self.name_to_hash[name]
        group = self.weight_groups.get(ref_hash)
        if not group:
            return None

        # Look for the weight in similar weights list
        for weight_name, weight, _ in group.similar:
            if weight_name == name:
                return weight

        return None
