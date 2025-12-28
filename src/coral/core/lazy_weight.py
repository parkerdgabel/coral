"""Lazy loading support for weight tensors.

This module provides lazy loading capabilities for weight tensors, enabling
memory-efficient handling of large models by loading weight data on-demand
from storage.

Key features:
- WeightProxy: A lightweight proxy that holds metadata but loads data lazily
- StreamingWeightIterator: Memory-efficient iteration over large weight collections
- Batch loading with configurable memory limits
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

from coral.core.weight_tensor import WeightMetadata, WeightTensor

if TYPE_CHECKING:
    from coral.storage.weight_store import WeightStore

logger = logging.getLogger(__name__)


@dataclass
class LazyLoadConfig:
    """Configuration for lazy loading behavior."""

    # Maximum memory (bytes) to use for cached weights
    max_cache_bytes: int = 1024 * 1024 * 1024  # 1GB default

    # Number of weights to keep in LRU cache
    max_cached_weights: int = 100

    # Whether to prefetch adjacent weights
    enable_prefetch: bool = False

    # Number of weights to prefetch
    prefetch_count: int = 5


class WeightProxy:
    """A lightweight proxy for lazy-loaded weight tensors.

    WeightProxy holds only metadata and a reference to storage, loading
    the actual weight data on-demand when accessed. This enables working
    with models larger than available RAM.

    Example:
        >>> store = HDF5Store("weights.h5")
        >>> proxy = WeightProxy.from_store(store, "abc123")
        >>> print(proxy.shape)  # No data loaded yet
        (1024, 768)
        >>> data = proxy.data  # Data loaded now
        >>> weight = proxy.materialize()  # Get full WeightTensor
    """

    __slots__ = (
        "_metadata",
        "_hash_key",
        "_store_ref",
        "_cached_data",
        "_load_fn",
        "__weakref__",
    )

    def __init__(
        self,
        metadata: WeightMetadata,
        hash_key: str,
        load_fn: Optional[Callable[[], np.ndarray]] = None,
        store: Optional[WeightStore] = None,
    ):
        """Initialize a weight proxy.

        Args:
            metadata: Weight metadata (shape, dtype, name, etc.)
            hash_key: Content hash for this weight
            load_fn: Optional custom function to load data
            store: Optional store reference for loading
        """
        self._metadata = metadata
        self._hash_key = hash_key
        self._cached_data: Optional[np.ndarray] = None

        if load_fn is not None:
            self._load_fn = load_fn
        elif store is not None:
            # Create a weak reference to avoid circular refs
            store_ref = weakref.ref(store)

            def _load_from_store() -> np.ndarray:
                s = store_ref()
                if s is None:
                    raise RuntimeError("Store has been garbage collected")
                weight = s.load(hash_key)
                if weight is None:
                    raise ValueError(f"Weight {hash_key} not found in store")
                return weight.data

            self._load_fn = _load_from_store
        else:
            self._load_fn = None
        self._store_ref = weakref.ref(store) if store else None

    @classmethod
    def from_store(cls, store: WeightStore, hash_key: str) -> WeightProxy:
        """Create a proxy from a store and hash key.

        Args:
            store: Weight store containing the data
            hash_key: Content hash of the weight

        Returns:
            WeightProxy that will load data on demand
        """
        metadata = store.get_metadata(hash_key)
        if metadata is None:
            raise ValueError(f"Weight {hash_key} not found in store")
        return cls(metadata=metadata, hash_key=hash_key, store=store)

    @classmethod
    def from_weight(
        cls, weight: WeightTensor, store: Optional[WeightStore] = None
    ) -> WeightProxy:
        """Create a proxy from an existing weight tensor.

        If data is already loaded, it will be cached in the proxy.

        Args:
            weight: Existing weight tensor
            store: Optional store for future lazy loading

        Returns:
            WeightProxy wrapping the weight
        """
        hash_key = weight.compute_hash()
        proxy = cls(metadata=weight.metadata, hash_key=hash_key, store=store)
        if weight._data is not None:
            proxy._cached_data = weight._data
        return proxy

    @property
    def metadata(self) -> WeightMetadata:
        """Get weight metadata (no data loading required)."""
        return self._metadata

    @property
    def hash_key(self) -> str:
        """Get the content hash (no data loading required)."""
        return self._hash_key

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape from metadata (no data loading required)."""
        return self._metadata.shape

    @property
    def dtype(self) -> np.dtype:
        """Get dtype from metadata (no data loading required)."""
        return self._metadata.dtype

    @property
    def name(self) -> str:
        """Get name from metadata (no data loading required)."""
        return self._metadata.name

    @property
    def nbytes(self) -> int:
        """Get size in bytes (calculated from metadata, no loading required)."""
        return int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize

    @property
    def is_loaded(self) -> bool:
        """Check if data is currently loaded in memory."""
        return self._cached_data is not None

    @property
    def data(self) -> np.ndarray:
        """Get weight data, loading from store if necessary."""
        if self._cached_data is None:
            self._load_data()
        return self._cached_data

    def _load_data(self) -> None:
        """Load data from storage."""
        if self._load_fn is None:
            raise RuntimeError(
                f"Cannot load weight {self._hash_key}: no load function or store"
            )
        logger.debug(f"Lazy loading weight {self._hash_key}")
        self._cached_data = self._load_fn()

    def unload(self) -> None:
        """Unload cached data to free memory."""
        if self._cached_data is not None:
            logger.debug(f"Unloading weight {self._hash_key}")
            self._cached_data = None

    def materialize(self) -> WeightTensor:
        """Convert proxy to a full WeightTensor with loaded data.

        Returns:
            WeightTensor with data loaded
        """
        return WeightTensor(
            data=self.data, metadata=self._metadata, store_ref=self._hash_key
        )

    def __repr__(self) -> str:
        loaded = "loaded" if self.is_loaded else "not loaded"
        return (
            f"WeightProxy(name='{self.name}', shape={self.shape}, "
            f"dtype={self.dtype}, {loaded})"
        )


class LazyWeightCollection:
    """A memory-efficient collection of lazy-loaded weights.

    This collection keeps only metadata in memory and loads weight data
    on-demand, with LRU caching to manage memory usage.

    Example:
        >>> collection = LazyWeightCollection(store)
        >>> collection.add("layer1.weight", "hash123")
        >>> collection.add("layer2.weight", "hash456")
        >>>
        >>> # Iterate without loading all into memory
        >>> for name, proxy in collection.items():
        ...     if "attention" in name:
        ...         weight = proxy.materialize()
        ...         process(weight)
    """

    def __init__(
        self,
        store: Optional[WeightStore] = None,
        config: Optional[LazyLoadConfig] = None,
    ):
        """Initialize lazy weight collection.

        Args:
            store: Weight store for loading data
            config: Lazy loading configuration
        """
        self._store = store
        self._config = config or LazyLoadConfig()
        self._proxies: dict[str, WeightProxy] = {}
        self._access_order: list[str] = []  # For LRU tracking
        self._total_cached_bytes: int = 0

    def add(
        self,
        name: str,
        hash_key: str,
        metadata: Optional[WeightMetadata] = None,
    ) -> WeightProxy:
        """Add a weight to the collection.

        Args:
            name: Weight name
            hash_key: Content hash
            metadata: Optional pre-loaded metadata

        Returns:
            WeightProxy for the added weight
        """
        if metadata is None and self._store is not None:
            metadata = self._store.get_metadata(hash_key)
        if metadata is None:
            raise ValueError(f"Cannot get metadata for weight {hash_key}")

        proxy = WeightProxy(metadata=metadata, hash_key=hash_key, store=self._store)
        self._proxies[name] = proxy
        return proxy

    def add_weight(self, name: str, weight: WeightTensor) -> WeightProxy:
        """Add an existing weight tensor to the collection.

        Args:
            name: Weight name
            weight: Weight tensor (data may be cached)

        Returns:
            WeightProxy wrapping the weight
        """
        proxy = WeightProxy.from_weight(weight, store=self._store)
        self._proxies[name] = proxy
        if proxy.is_loaded:
            self._total_cached_bytes += proxy.nbytes
        return proxy

    def get(self, name: str) -> Optional[WeightProxy]:
        """Get a weight proxy by name."""
        proxy = self._proxies.get(name)
        if proxy is not None:
            self._update_access_order(name)
        return proxy

    def load(self, name: str) -> Optional[WeightTensor]:
        """Load and return a weight by name.

        Args:
            name: Weight name

        Returns:
            Loaded WeightTensor or None if not found
        """
        proxy = self.get(name)
        if proxy is None:
            return None

        # Check memory limits before loading
        if not proxy.is_loaded:
            self._ensure_memory_available(proxy.nbytes)

        weight = proxy.materialize()
        self._total_cached_bytes += proxy.nbytes
        return weight

    def _update_access_order(self, name: str) -> None:
        """Update LRU access order."""
        if name in self._access_order:
            self._access_order.remove(name)
        self._access_order.append(name)

    def _ensure_memory_available(self, needed_bytes: int) -> None:
        """Ensure enough memory is available by evicting old entries."""
        while (
            self._total_cached_bytes + needed_bytes > self._config.max_cache_bytes
            and self._access_order
        ):
            # Evict least recently used
            oldest_name = self._access_order.pop(0)
            proxy = self._proxies.get(oldest_name)
            if proxy is not None and proxy.is_loaded:
                self._total_cached_bytes -= proxy.nbytes
                proxy.unload()
                logger.debug(f"Evicted weight {oldest_name} from cache")

    def unload_all(self) -> None:
        """Unload all cached weight data."""
        for proxy in self._proxies.values():
            proxy.unload()
        self._total_cached_bytes = 0
        self._access_order.clear()

    def items(self) -> Iterator[tuple[str, WeightProxy]]:
        """Iterate over (name, proxy) pairs."""
        return iter(self._proxies.items())

    def keys(self) -> Iterator[str]:
        """Iterate over weight names."""
        return iter(self._proxies.keys())

    def values(self) -> Iterator[WeightProxy]:
        """Iterate over weight proxies."""
        return iter(self._proxies.values())

    def __len__(self) -> int:
        return len(self._proxies)

    def __contains__(self, name: str) -> bool:
        return name in self._proxies

    def __getitem__(self, name: str) -> WeightProxy:
        proxy = self.get(name)
        if proxy is None:
            raise KeyError(name)
        return proxy

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory usage statistics."""
        loaded_count = sum(1 for p in self._proxies.values() if p.is_loaded)
        total_size = sum(p.nbytes for p in self._proxies.values())

        return {
            "total_weights": len(self._proxies),
            "loaded_weights": loaded_count,
            "cached_bytes": self._total_cached_bytes,
            "total_bytes": total_size,
            "max_cache_bytes": self._config.max_cache_bytes,
            "cache_utilization": self._total_cached_bytes / self._config.max_cache_bytes
            if self._config.max_cache_bytes > 0
            else 0,
        }


class StreamingWeightIterator:
    """Memory-efficient iterator over weights from storage.

    Loads weights one at a time (or in small batches) to minimize memory usage.
    Useful for processing large models that don't fit in memory.

    Example:
        >>> store = HDF5Store("large_model.h5")
        >>> for weight in StreamingWeightIterator(store):
        ...     process(weight)
        ...     # weight is automatically unloaded when next is called
    """

    def __init__(
        self,
        store: WeightStore,
        hash_keys: Optional[list[str]] = None,
        batch_size: int = 1,
        preload_metadata: bool = True,
    ):
        """Initialize streaming iterator.

        Args:
            store: Weight store to iterate over
            hash_keys: Optional list of specific keys to iterate
            batch_size: Number of weights to load at once
            preload_metadata: Whether to preload all metadata upfront
        """
        self._store = store
        self._hash_keys = hash_keys or store.list_weights()
        self._batch_size = batch_size
        self._index = 0
        self._current_batch: list[WeightTensor] = []
        self._batch_index = 0

        # Optionally preload metadata for all weights
        self._metadata_cache: dict[str, WeightMetadata] = {}
        if preload_metadata:
            for key in self._hash_keys:
                meta = store.get_metadata(key)
                if meta:
                    self._metadata_cache[key] = meta

    def __iter__(self) -> StreamingWeightIterator:
        self._index = 0
        self._current_batch = []
        self._batch_index = 0
        return self

    def __next__(self) -> WeightTensor:
        # Load next batch if needed
        if self._batch_index >= len(self._current_batch):
            if self._index >= len(self._hash_keys):
                raise StopIteration

            # Clear previous batch to free memory
            self._current_batch.clear()

            # Load next batch
            end_idx = min(self._index + self._batch_size, len(self._hash_keys))
            for i in range(self._index, end_idx):
                weight = self._store.load(self._hash_keys[i])
                if weight is not None:
                    self._current_batch.append(weight)

            self._index = end_idx
            self._batch_index = 0

            if not self._current_batch:
                raise StopIteration

        weight = self._current_batch[self._batch_index]
        self._batch_index += 1
        return weight

    def __len__(self) -> int:
        return len(self._hash_keys)

    def get_metadata(self, hash_key: str) -> Optional[WeightMetadata]:
        """Get cached metadata for a weight."""
        return self._metadata_cache.get(hash_key)

    def filter(
        self, predicate: Callable[[WeightMetadata], bool]
    ) -> StreamingWeightIterator:
        """Create a filtered iterator based on metadata.

        Args:
            predicate: Function that takes metadata and returns True to include

        Returns:
            New iterator with filtered keys
        """
        filtered_keys = [
            key
            for key in self._hash_keys
            if key in self._metadata_cache and predicate(self._metadata_cache[key])
        ]
        return StreamingWeightIterator(
            store=self._store,
            hash_keys=filtered_keys,
            batch_size=self._batch_size,
            preload_metadata=False,
        )
