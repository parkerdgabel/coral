"""Abstract interface for weight storage backends"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from coral.core.weight_tensor import WeightMetadata, WeightTensor


class WeightStore(ABC):
    """
    Abstract base class for weight storage backends.

    Implementations should provide:
    - Content-addressable storage using hashes
    - Metadata storage and retrieval
    - Batch operations for efficiency
    - Optional compression support

    Note:
    - Implementations should accept both str and Path-like objects for path parameters
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
    def list_weights(self) -> list[str]:
        """List all weight hashes in storage"""
        pass

    @abstractmethod
    def get_metadata(self, hash_key: str) -> Optional[WeightMetadata]:
        """Get metadata for a weight without loading data"""
        pass

    @abstractmethod
    def store_batch(self, weights: dict[str, WeightTensor]) -> dict[str, str]:
        """
        Store multiple weights efficiently.

        Args:
            weights: Dict mapping names to WeightTensors

        Returns:
            Dict mapping names to storage hashes
        """
        pass

    @abstractmethod
    def load_batch(self, hash_keys: list[str]) -> dict[str, WeightTensor]:
        """
        Load multiple weights efficiently.

        Args:
            hash_keys: List of hash keys to load

        Returns:
            Dict mapping hash keys to WeightTensors
        """
        pass

    @abstractmethod
    def get_storage_info(self) -> dict[str, Any]:
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
