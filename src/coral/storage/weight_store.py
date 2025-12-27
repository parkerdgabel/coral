"""Abstract interface for weight storage backends with large model scaling support"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from coral.core.weight_tensor import WeightMetadata, WeightTensor


class WeightStore(ABC):
    """
    Abstract base class for weight storage backends.

    Implementations should provide:
    - Content-addressable storage using hashes
    - Metadata storage and retrieval
    - Batch operations for efficiency
    - Optional compression support
    - **Partial/slice loading for large weights**
    - **Memory-mapped loading for huge weights**
    - **Streaming iterators for bounded memory usage**
    """

    @abstractmethod
    def store(self, weight: WeightTensor, hash_key: Optional[str] = None) -> str:
        """
        Store a weight tensor.

        Args:
            weight: WeightTensor to store
            hash_key: Optional hash to use as key (will compute if not provided)

        Returns:
            Hash key used for storage
        """
        pass

    @abstractmethod
    def load(self, hash_key: str) -> Optional[WeightTensor]:
        """
        Load a weight tensor by hash.

        Args:
            hash_key: Hash key of the weight

        Returns:
            WeightTensor if found, None otherwise
        """
        pass

    @abstractmethod
    def exists(self, hash_key: str) -> bool:
        """Check if a weight exists in storage"""
        pass

    @abstractmethod
    def delete(self, hash_key: str) -> bool:
        """
        Delete a weight from storage.

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def list_weights(self) -> List[str]:
        """List all weight hashes in storage"""
        pass

    @abstractmethod
    def get_metadata(self, hash_key: str) -> Optional[WeightMetadata]:
        """Get metadata for a weight without loading data"""
        pass

    @abstractmethod
    def store_batch(self, weights: Dict[str, WeightTensor]) -> Dict[str, str]:
        """
        Store multiple weights efficiently.

        Args:
            weights: Dict mapping names to WeightTensors

        Returns:
            Dict mapping names to storage hashes
        """
        pass

    @abstractmethod
    def load_batch(self, hash_keys: List[str]) -> Dict[str, WeightTensor]:
        """
        Load multiple weights efficiently.

        Args:
            hash_keys: List of hash keys to load

        Returns:
            Dict mapping hash keys to WeightTensors
        """
        pass

    @abstractmethod
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about storage usage and statistics"""
        pass

    @abstractmethod
    def close(self):
        """Close the storage backend and cleanup resources"""
        pass

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    # =========================================================================
    # Large Model Scaling Methods (Optional - with default implementations)
    # =========================================================================

    def load_slice(
        self,
        hash_key: str,
        slices: Union[slice, Tuple[slice, ...]],
    ) -> Optional[np.ndarray]:
        """
        Load a partial slice of weight data without loading the entire tensor.

        This is essential for large models where loading full weights would exceed RAM.

        Args:
            hash_key: Hash key of the weight
            slices: Slice object or tuple of slices for multi-dimensional access

        Returns:
            Numpy array containing the sliced data, or None if not found

        Note:
            Default implementation loads full weight and slices in memory.
            Subclasses should override for efficient partial loading.
        """
        weight = self.load(hash_key)
        if weight is None:
            return None
        return weight.data[slices]

    def iter_weights(
        self,
        batch_size: int = 100,
        hash_keys: Optional[List[str]] = None,
    ) -> Generator[Dict[str, WeightTensor], None, None]:
        """
        Iterate over weights in batches for memory-bounded processing.

        Args:
            batch_size: Number of weights to yield per batch
            hash_keys: Optional list of specific hashes to iterate over

        Yields:
            Dict mapping hash keys to WeightTensor objects, batch_size at a time
        """
        keys = hash_keys if hash_keys is not None else self.list_weights()

        batch: Dict[str, WeightTensor] = {}
        for hash_key in keys:
            weight = self.load(hash_key)
            if weight is not None:
                batch[hash_key] = weight

            if len(batch) >= batch_size:
                yield batch
                batch = {}

        if batch:
            yield batch

    def iter_metadata(
        self,
        batch_size: int = 1000,
        hash_keys: Optional[List[str]] = None,
    ) -> Generator[Dict[str, WeightMetadata], None, None]:
        """
        Iterate over weight metadata without loading weight data.

        Args:
            batch_size: Number of metadata entries per batch
            hash_keys: Optional list of specific hashes

        Yields:
            Dict mapping hash keys to WeightMetadata objects
        """
        keys = hash_keys if hash_keys is not None else self.list_weights()

        batch: Dict[str, WeightMetadata] = {}
        for hash_key in keys:
            metadata = self.get_metadata(hash_key)
            if metadata is not None:
                batch[hash_key] = metadata

            if len(batch) >= batch_size:
                yield batch
                batch = {}

        if batch:
            yield batch

    def get_weight_size(self, hash_key: str) -> Optional[int]:
        """
        Get the size of a weight in bytes without loading data.

        Default implementation loads the weight to get size.
        Subclasses should override for efficient size queries.
        """
        weight = self.load(hash_key)
        return weight.nbytes if weight else None

    def estimate_memory_usage(
        self, hash_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Estimate memory required to load weights.

        Args:
            hash_keys: Specific weights to check (default: all weights)

        Returns:
            Dict with memory estimation details
        """
        keys = hash_keys if hash_keys is not None else self.list_weights()

        total_bytes = 0
        weight_sizes: List[Tuple[str, int]] = []

        for hash_key in keys:
            size = self.get_weight_size(hash_key)
            if size is not None:
                total_bytes += size
                weight_sizes.append((hash_key, size))

        weight_sizes.sort(key=lambda x: x[1], reverse=True)

        return {
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "total_gb": total_bytes / (1024 * 1024 * 1024),
            "num_weights": len(weight_sizes),
            "largest_weights": weight_sizes[:10],
            "average_bytes": total_bytes / len(weight_sizes) if weight_sizes else 0,
        }
