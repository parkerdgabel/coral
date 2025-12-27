"""Base class for representing neural network weights with large model scaling."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import numpy as np
import xxhash

if TYPE_CHECKING:
    pass  # For type hints that would cause circular imports


# Type alias for lazy loader functions
LazyLoader = Callable[[str], Optional[np.ndarray]]


@dataclass
class WeightMetadata:
    """Metadata associated with a weight tensor"""

    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    layer_type: Optional[str] = None
    model_name: Optional[str] = None
    compression_info: Dict[str, Any] = field(default_factory=dict)
    hash: Optional[str] = None


class WeightTensor:
    """
    Base class for representing neural network weights with deduplication support.

    This class provides a unified interface for weight tensors with support for:
    - Content-based hashing for deduplication
    - Metadata tracking
    - **Lazy loading from storage (large model scaling)**
    - **Memory-mapped data access for huge weights**
    - Compression support
    """

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        metadata: Optional[WeightMetadata] = None,
        store_ref: Optional[str] = None,
        lazy_loader: Optional[LazyLoader] = None,
    ):
        """
        Initialize a WeightTensor.

        Args:
            data: The actual weight data as numpy array
            metadata: Metadata about the weight tensor
            store_ref: Reference to data in storage (for lazy loading)
            lazy_loader: Optional function to load data on-demand from store_ref.
                        Signature: (store_ref: str) -> np.ndarray
        """
        self._data = data
        self._metadata = metadata
        self._store_ref = store_ref
        self._lazy_loader = lazy_loader
        self._hash: Optional[str] = None
        self._is_loaded = data is not None

        if data is not None and metadata is None:
            # Auto-create metadata from data
            self._metadata = WeightMetadata(
                name="unnamed", shape=data.shape, dtype=data.dtype
            )

    @property
    def data(self) -> np.ndarray:
        """Get the weight data, loading from storage if necessary (lazy loading)"""
        if self._data is None:
            if self._lazy_loader is not None and self._store_ref is not None:
                # Lazy load from storage
                loaded_data = self._lazy_loader(self._store_ref)
                if loaded_data is not None:
                    self._data = loaded_data
                    self._is_loaded = True
                else:
                    raise ValueError(
                        f"Failed to load weight from store ref: {self._store_ref}"
                    )
            else:
                raise ValueError(
                    "Weight data not loaded and no store reference available"
                )
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
    def shape(self) -> Tuple[int, ...]:
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

    def to_dict(self) -> Dict[str, Any]:
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
        cls, data: Dict[str, Any], weight_data: Optional[np.ndarray] = None
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

    # =========================================================================
    # Large Model Scaling Methods
    # =========================================================================

    @property
    def is_loaded(self) -> bool:
        """Check if weight data is currently loaded in memory"""
        return self._is_loaded and self._data is not None

    @property
    def store_ref(self) -> Optional[str]:
        """Get the storage reference for this weight"""
        return self._store_ref

    @property
    def can_lazy_load(self) -> bool:
        """Check if this weight supports lazy loading"""
        return self._lazy_loader is not None and self._store_ref is not None

    def unload(self) -> bool:
        """
        Unload weight data from memory to free RAM.

        Only works if the weight can be lazy-loaded again later.
        Useful for managing memory with very large model collections.

        Returns:
            True if data was unloaded, False if unloading is not safe
        """
        if not self.can_lazy_load:
            return False

        self._data = None
        self._is_loaded = False
        return True

    def set_lazy_loader(self, loader: LazyLoader) -> None:
        """
        Set the lazy loader function for on-demand data loading.

        Args:
            loader: Function that takes a store_ref and returns numpy array
        """
        self._lazy_loader = loader

    def ensure_loaded(self) -> None:
        """
        Ensure weight data is loaded into memory.

        Raises:
            ValueError: If data cannot be loaded
        """
        # Accessing .data triggers lazy loading
        _ = self.data

    @classmethod
    def create_lazy(
        cls,
        metadata: WeightMetadata,
        store_ref: str,
        lazy_loader: LazyLoader,
    ) -> "WeightTensor":
        """
        Create a lazy-loading WeightTensor without loading data.

        This is useful for creating weight references that don't consume
        memory until the data is actually accessed.

        Args:
            metadata: Weight metadata (shape, dtype, name, etc.)
            store_ref: Storage reference for lazy loading
            lazy_loader: Function to load data on demand

        Returns:
            WeightTensor that will load data on first access
        """
        tensor = cls(data=None, metadata=metadata, store_ref=store_ref)
        tensor._lazy_loader = lazy_loader
        return tensor

    def __repr__(self) -> str:
        loaded_str = "loaded" if self.is_loaded else "not loaded"
        return (
            f"WeightTensor(name='{self.metadata.name}', "
            f"shape={self.shape}, dtype={self.dtype}, "
            f"size={self.size}, nbytes={self.nbytes}, {loaded_str})"
        )
