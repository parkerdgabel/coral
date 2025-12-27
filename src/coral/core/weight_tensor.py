"""Base class for representing neural network weights"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import xxhash

if TYPE_CHECKING:
    pass  # For type hints that would cause circular imports


@dataclass
class WeightMetadata:
    """Metadata associated with a weight tensor"""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    layer_type: Optional[str] = None
    model_name: Optional[str] = None
    compression_info: dict[str, Any] = field(default_factory=dict)
    hash: Optional[str] = None


class WeightTensor:
    """
    Base class for representing neural network weights with deduplication support.

    This class provides a unified interface for weight tensors with support for:
    - Content-based hashing for deduplication
    - Metadata tracking
    - Lazy loading from storage
    - Compression support
    """

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        metadata: Optional[WeightMetadata] = None,
        store_ref: Optional[str] = None,
    ):
        """
        Initialize a WeightTensor.

        Args:
            data: The actual weight data as numpy array
            metadata: Metadata about the weight tensor
            store_ref: Reference to data in storage (for lazy loading)
        """
        self._data = data
        self._metadata = metadata
        self._store_ref = store_ref
        self._hash: Optional[str] = None

        if data is not None and metadata is None:
            # Auto-create metadata from data
            self._metadata = WeightMetadata(
                name="unnamed", shape=data.shape, dtype=data.dtype
            )

    @property
    def data(self) -> np.ndarray:
        """Get the weight data, loading from storage if necessary"""
        if self._data is None:
            raise ValueError("Weight data not loaded and no store reference available")
        return self._data

    @data.setter
    def data(self, value: np.ndarray) -> None:
        """Set the weight data and invalidate hash"""
        self._data = value
        self._hash = None  # Invalidate cached hash when data changes
        # Update metadata shape if it exists
        if self._metadata is not None:
            self._metadata.shape = value.shape
            self._metadata.dtype = value.dtype

    @property
    def metadata(self) -> WeightMetadata:
        """Get the weight metadata"""
        if self._metadata is None:
            raise ValueError("No metadata available for this weight tensor")
        return self._metadata

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the weight tensor"""
        return self.metadata.shape

    @property
    def dtype(self) -> np.dtype:
        """Get the data type of the weight tensor"""
        return self.metadata.dtype

    @property
    def size(self) -> int:
        """Get the total number of elements"""
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        """Get the number of bytes used by the tensor"""
        # Use the data's nbytes directly if available, otherwise calculate
        if self._data is not None:
            return self._data.nbytes
        else:
            # Calculate from metadata
            return self.size * np.dtype(self.dtype).itemsize

    def compute_hash(self, force: bool = False) -> str:
        """
        Compute content-based hash of the weight tensor.

        Args:
            force: If True, recompute hash even if cached

        Returns:
            Hexadecimal hash string
        """
        if self._hash is not None and not force:
            return self._hash

        # Use xxhash for fast hashing
        hasher = xxhash.xxh3_64()

        # Include shape and dtype in hash to distinguish identical data
        # with different interpretations
        # Normalize shape to ensure consistent hashing regardless of int types
        normalized_shape = tuple(int(dim) for dim in self.shape)
        # Normalize dtype to ensure consistent hashing regardless of representation
        normalized_dtype = np.dtype(self.dtype).name
        hasher.update(str(normalized_shape).encode())
        hasher.update(normalized_dtype.encode())

        # Hash the actual data
        hasher.update(self.data.tobytes())

        self._hash = hasher.hexdigest()
        if self._metadata:
            self._metadata.hash = self._hash

        return self._hash

    def is_similar_to(self, other: "WeightTensor", threshold: float = 0.99) -> bool:
        """
        Check if this weight tensor is similar to another.

        Uses cosine similarity for comparison.

        Args:
            other: Another WeightTensor to compare with
            threshold: Similarity threshold (0-1)

        Returns:
            True if similarity exceeds threshold
        """
        if self.shape != other.shape or self.dtype != other.dtype:
            return False

        # Import here to avoid circular dependency with utils
        from coral.utils.similarity import cosine_similarity

        similarity = cosine_similarity(self.data, other.data)
        return similarity >= threshold

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "metadata": {
                "name": self.metadata.name,
                "shape": list(self.metadata.shape),
                "dtype": np.dtype(
                    self.metadata.dtype
                ).name,  # Use .name for proper serialization
                "layer_type": self.metadata.layer_type,
                "model_name": self.metadata.model_name,
                "compression_info": self.metadata.compression_info,
                "hash": self.compute_hash(),
            },
            "store_ref": self._store_ref,
            "has_data": self._data is not None,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], weight_data: Optional[np.ndarray] = None
    ) -> "WeightTensor":
        """Create WeightTensor from dictionary"""
        metadata = WeightMetadata(
            name=data["metadata"]["name"],
            shape=tuple(data["metadata"]["shape"]),
            dtype=np.dtype(data["metadata"]["dtype"]),
            layer_type=data["metadata"].get("layer_type"),
            model_name=data["metadata"].get("model_name"),
            compression_info=data["metadata"].get("compression_info", {}),
            hash=data["metadata"].get("hash"),
        )

        return cls(data=weight_data, metadata=metadata, store_ref=data.get("store_ref"))

    def __repr__(self) -> str:
        return (
            f"WeightTensor(name='{self.metadata.name}', "
            f"shape={self.shape}, dtype={self.dtype}, "
            f"size={self.size}, nbytes={self.nbytes})"
        )
