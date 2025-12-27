"""HDF5-based storage backend for weight tensors with large model scaling support"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import h5py
import numpy as np

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.delta.delta_encoder import Delta
from coral.storage.weight_store import WeightStore

logger = logging.getLogger(__name__)

# Default chunk size for HDF5 datasets (1MB worth of float32 elements)
DEFAULT_CHUNK_ELEMENTS = 256 * 1024  # 256K elements = 1MB for float32


class HDF5Store(WeightStore):
    """
    HDF5-based storage backend for weight tensors.

    Features:
    - Content-addressable storage using hash-based keys
    - Compression support (gzip, lzf)
    - Efficient batch operations
    - Metadata stored as HDF5 attributes
    - **Chunked storage for partial reads (large model scaling)**
    - **Memory-mapped loading for weights exceeding RAM**
    - **Streaming iterators for bounded memory usage**
    - **Slice/partial loading for selective weight access**
    """

    def __init__(
        self,
        filepath: str,
        compression: Optional[str] = "gzip",
        compression_opts: Optional[int] = 4,
        mode: str = "a",
        enable_chunking: bool = True,
        chunk_size: Optional[int] = None,
    ):
        """
        Initialize HDF5 storage.

        Args:
            filepath: Path to HDF5 file
            compression: Compression algorithm ('gzip', 'lzf', None)
            compression_opts: Compression level (1-9 for gzip)
            mode: File mode ('r', 'r+', 'w', 'a')
            enable_chunking: Enable HDF5 chunking for partial reads (default: True)
            chunk_size: Number of elements per chunk (default: 256K = 1MB for float32)
        """
        self.filepath = Path(filepath)
        self.compression = compression
        self.compression_opts = compression_opts
        self.mode = mode
        self.enable_chunking = enable_chunking
        self.chunk_size = chunk_size or DEFAULT_CHUNK_ELEMENTS

        # Ensure directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Open HDF5 file
        self.file = h5py.File(self.filepath, mode)

        # Create groups if they don't exist
        if mode in ["w", "a"]:
            if "weights" not in self.file:
                self.file.create_group("weights")
            if "metadata" not in self.file:
                self.file.create_group("metadata")
            if "deltas" not in self.file:
                self.file.create_group("deltas")

    def _compute_chunks(self, shape: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
        """
        Compute optimal chunk shape for a given weight shape.

        For 1D arrays: chunks along the single dimension
        For nD arrays: chunks along the largest dimension, keep others whole
        """
        if not self.enable_chunking:
            return None

        total_elements = int(np.prod(shape))
        if total_elements <= self.chunk_size:
            # Small enough to be a single chunk
            return None

        # For 1D arrays
        if len(shape) == 1:
            return (min(self.chunk_size, shape[0]),)

        # For nD arrays, chunk along the first (typically batch/output) dimension
        # Keep other dimensions whole for efficient layer-wise access
        chunk_shape = list(shape)
        elements_per_slice = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        slices_per_chunk = max(1, self.chunk_size // elements_per_slice)
        chunk_shape[0] = min(slices_per_chunk, shape[0])

        return tuple(chunk_shape)

    def store(self, weight: WeightTensor, hash_key: Optional[str] = None) -> str:
        """Store a weight tensor with optional chunking for large model support"""
        if hash_key is None:
            hash_key = weight.compute_hash()

        # Check if already exists
        if self.exists(hash_key):
            logger.debug(f"Weight {hash_key} already exists in storage")
            return hash_key

        # Compute chunks for partial read support
        chunks = self._compute_chunks(weight.data.shape)

        # Store weight data with chunking
        # Note: lzf compression doesn't accept compression_opts
        # Note: If no compression, don't pass compression_opts
        weights_group = self.file["weights"]
        if self.compression is None:
            compression_opts = None
        elif self.compression == "lzf":
            compression_opts = None
        else:
            compression_opts = self.compression_opts

        dataset = weights_group.create_dataset(
            hash_key,
            data=weight.data,
            compression=self.compression,
            compression_opts=compression_opts,
            chunks=chunks,
        )

        # Store metadata as attributes
        metadata = weight.metadata
        dataset.attrs["name"] = metadata.name
        dataset.attrs["shape"] = metadata.shape
        dataset.attrs["dtype"] = np.dtype(metadata.dtype).name
        dataset.attrs["layer_type"] = metadata.layer_type or ""
        dataset.attrs["model_name"] = metadata.model_name or ""
        dataset.attrs["compression_info"] = json.dumps(metadata.compression_info)
        dataset.attrs["hash"] = hash_key

        # Flush to ensure data is written
        self.file.flush()

        logger.debug(f"Stored weight {metadata.name} with hash {hash_key}")
        return hash_key

    def load(self, hash_key: str) -> Optional[WeightTensor]:
        """Load a weight tensor by hash"""
        if not self.exists(hash_key):
            return None

        dataset = self.file["weights"][hash_key]

        # Load data
        data = np.array(dataset)

        # Load metadata from attributes
        # Ensure shape is converted to tuple of Python ints to maintain consistency
        shape_array = dataset.attrs["shape"]
        normalized_shape = tuple(int(dim) for dim in shape_array)

        metadata = WeightMetadata(
            name=dataset.attrs["name"],
            shape=normalized_shape,
            dtype=np.dtype(dataset.attrs["dtype"]),
            layer_type=dataset.attrs.get("layer_type") or None,
            model_name=dataset.attrs.get("model_name") or None,
            compression_info=json.loads(dataset.attrs.get("compression_info", "{}")),
            hash=dataset.attrs.get("hash", hash_key),
        )

        return WeightTensor(data=data, metadata=metadata, store_ref=hash_key)

    def exists(self, hash_key: str) -> bool:
        """Check if a weight exists in storage"""
        return hash_key in self.file["weights"]

    def delete(self, hash_key: str) -> bool:
        """Delete a weight from storage"""
        if not self.exists(hash_key):
            return False

        del self.file["weights"][hash_key]
        self.file.flush()
        return True

    def list_weights(self) -> List[str]:
        """List all weight hashes in storage"""
        return list(self.file["weights"].keys())

    def get_metadata(self, hash_key: str) -> Optional[WeightMetadata]:
        """Get metadata for a weight without loading data"""
        if not self.exists(hash_key):
            return None

        dataset = self.file["weights"][hash_key]

        # Ensure shape is converted to tuple of Python ints to maintain consistency
        shape_array = dataset.attrs["shape"]
        normalized_shape = tuple(int(dim) for dim in shape_array)

        return WeightMetadata(
            name=dataset.attrs["name"],
            shape=normalized_shape,
            dtype=np.dtype(dataset.attrs["dtype"]),
            layer_type=dataset.attrs.get("layer_type") or None,
            model_name=dataset.attrs.get("model_name") or None,
            compression_info=json.loads(dataset.attrs.get("compression_info", "{}")),
            hash=dataset.attrs.get("hash", hash_key),
        )

    def store_batch(self, weights: Dict[str, WeightTensor]) -> Dict[str, str]:
        """Store multiple weights efficiently"""
        result = {}

        for name, weight in weights.items():
            hash_key = self.store(weight)
            result[name] = hash_key

        return result

    def load_batch(self, hash_keys: List[str]) -> Dict[str, WeightTensor]:
        """Load multiple weights efficiently"""
        result = {}

        for hash_key in hash_keys:
            weight = self.load(hash_key)
            if weight is not None:
                result[hash_key] = weight

        return result

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about storage usage and statistics"""
        weights_group = self.file["weights"]

        total_weights = len(weights_group)
        total_bytes = 0
        compressed_bytes = 0

        for key in weights_group:
            dataset = weights_group[key]
            total_bytes += dataset.nbytes
            compressed_bytes += dataset.id.get_storage_size()

        compression_ratio = (
            1.0 - (compressed_bytes / total_bytes) if total_bytes > 0 else 0.0
        )

        return {
            "filepath": str(self.filepath),
            "file_size": os.path.getsize(self.filepath)
            if self.filepath.exists()
            else 0,
            "total_weights": total_weights,
            "total_bytes": total_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": compression_ratio,
            "compression": self.compression,
        }

    def close(self):
        """Close the HDF5 file"""
        if hasattr(self, "file") and self.file:
            self.file.close()

    def __enter__(self) -> "HDF5Store":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context manager and close file."""
        self.close()
        return False

    def store_delta(self, delta: Delta, delta_hash: str) -> str:
        """Store a delta object."""
        if self.delta_exists(delta_hash):
            logger.debug(f"Delta {delta_hash} already exists in storage")
            return delta_hash

        deltas_group = self.file["deltas"]

        # Store delta data
        # Note: lzf compression doesn't accept compression_opts
        compression_opts = None if self.compression == "lzf" else self.compression_opts
        dataset = deltas_group.create_dataset(
            delta_hash,
            data=delta.data,
            compression=self.compression,
            compression_opts=compression_opts,
        )

        # Store delta metadata as attributes
        dataset.attrs["delta_type"] = delta.delta_type.value
        dataset.attrs["original_shape"] = delta.original_shape
        dataset.attrs["original_dtype"] = np.dtype(delta.original_dtype).name
        dataset.attrs["reference_hash"] = delta.reference_hash
        dataset.attrs["compression_ratio"] = delta.compression_ratio
        dataset.attrs["metadata"] = json.dumps(delta.metadata)

        self.file.flush()
        logger.debug(f"Stored delta {delta_hash}")
        return delta_hash

    def load_delta(self, delta_hash: str) -> Optional[Delta]:
        """Load a delta object by hash."""
        if not self.delta_exists(delta_hash):
            return None

        dataset = self.file["deltas"][delta_hash]

        # Reconstruct delta object
        from coral.delta.delta_encoder import DeltaType

        delta = Delta(
            delta_type=DeltaType(dataset.attrs["delta_type"]),
            data=np.array(dataset),
            metadata=json.loads(dataset.attrs.get("metadata", "{}")),
            original_shape=tuple(dataset.attrs["original_shape"]),
            original_dtype=np.dtype(dataset.attrs["original_dtype"]),
            reference_hash=dataset.attrs["reference_hash"],
            compression_ratio=dataset.attrs.get("compression_ratio", 0.0),
        )

        return delta

    def delta_exists(self, delta_hash: str) -> bool:
        """Check if a delta exists in storage."""
        return delta_hash in self.file["deltas"]

    def delete_delta(self, delta_hash: str) -> bool:
        """Delete a delta from storage."""
        if not self.delta_exists(delta_hash):
            return False

        del self.file["deltas"][delta_hash]
        self.file.flush()
        return True

    def list_deltas(self) -> List[str]:
        """List all delta hashes in storage."""
        return list(self.file["deltas"].keys())

    def get_delta_storage_info(self) -> Dict[str, Any]:
        """Get information about delta storage."""
        if "deltas" not in self.file:
            return {"total_deltas": 0, "total_delta_bytes": 0}

        deltas_group = self.file["deltas"]
        total_deltas = len(deltas_group)
        total_delta_bytes = 0

        for delta_hash in deltas_group:
            dataset = deltas_group[delta_hash]
            total_delta_bytes += dataset.nbytes

        return {"total_deltas": total_deltas, "total_delta_bytes": total_delta_bytes}

    # =========================================================================
    # Large Model Scaling Methods
    # =========================================================================

    def load_slice(
        self,
        hash_key: str,
        slices: Union[slice, Tuple[slice, ...]],
    ) -> Optional[np.ndarray]:
        """
        Load a partial slice of weight data without loading the entire tensor.

        This is essential for large models where loading full weights would exceed RAM.
        Requires chunked storage for efficient partial reads.

        Args:
            hash_key: Hash key of the weight
            slices: Slice object or tuple of slices for multi-dimensional access
                Example: slice(0, 1000) for first 1000 elements
                Example: (slice(0, 100), slice(None)) for first 100 rows

        Returns:
            Numpy array containing the sliced data, or None if not found
        """
        if not self.exists(hash_key):
            return None

        dataset = self.file["weights"][hash_key]
        return np.array(dataset[slices])

    def load_mmap(
        self,
        hash_key: str,
        temp_dir: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Load a weight as a memory-mapped array for very large weights.

        Memory-mapped arrays allow accessing weights larger than RAM by
        loading data on-demand from disk. Useful for 100B+ parameter models.

        Args:
            hash_key: Hash key of the weight
            temp_dir: Directory for temporary memory-mapped file (default: system temp)

        Returns:
            Memory-mapped numpy array, or None if not found

        Note:
            The returned array is read-only and should not be modified.
            For best performance, access data sequentially or in chunks.
        """
        import tempfile

        if not self.exists(hash_key):
            return None

        dataset = self.file["weights"][hash_key]
        shape = dataset.shape
        dtype = dataset.dtype

        # Create temporary memory-mapped file
        if temp_dir:
            temp_file = tempfile.NamedTemporaryFile(
                dir=temp_dir, suffix=".mmap", delete=False
            )
        else:
            temp_file = tempfile.NamedTemporaryFile(suffix=".mmap", delete=False)

        temp_path = temp_file.name
        temp_file.close()

        # Create memory-mapped array and copy data in chunks
        mmap_array = np.memmap(temp_path, dtype=dtype, mode="w+", shape=shape)

        # Copy in chunks to avoid loading entire weight into memory
        if len(shape) == 1:
            chunk_size = min(self.chunk_size, shape[0])
            for start in range(0, shape[0], chunk_size):
                end = min(start + chunk_size, shape[0])
                mmap_array[start:end] = dataset[start:end]
        else:
            # For nD arrays, copy along first dimension
            chunk_size = max(1, self.chunk_size // int(np.prod(shape[1:])))
            for start in range(0, shape[0], chunk_size):
                end = min(start + chunk_size, shape[0])
                mmap_array[start:end] = dataset[start:end]

        mmap_array.flush()

        # Reopen as read-only
        return np.memmap(temp_path, dtype=dtype, mode="r", shape=shape)

    def iter_weights(
        self,
        batch_size: int = 100,
        hash_keys: Optional[List[str]] = None,
    ) -> Generator[Dict[str, WeightTensor], None, None]:
        """
        Iterate over weights in batches for memory-bounded processing.

        Instead of loading all weights at once, this generator yields batches
        of weights, allowing processing of repositories with millions of weights
        without running out of memory.

        Args:
            batch_size: Number of weights to yield per batch
            hash_keys: Optional list of specific hashes to iterate over
                      (default: all weights)

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

        # Yield remaining weights
        if batch:
            yield batch

    def iter_metadata(
        self,
        batch_size: int = 1000,
        hash_keys: Optional[List[str]] = None,
    ) -> Generator[Dict[str, WeightMetadata], None, None]:
        """
        Iterate over weight metadata without loading weight data.

        Extremely efficient for querying/filtering large repositories
        since only metadata attributes are read, not the full tensor data.

        Args:
            batch_size: Number of metadata entries per batch
            hash_keys: Optional list of specific hashes (default: all weights)

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

        Args:
            hash_key: Hash key of the weight

        Returns:
            Size in bytes, or None if weight doesn't exist
        """
        if not self.exists(hash_key):
            return None

        dataset = self.file["weights"][hash_key]
        return dataset.nbytes

    def estimate_memory_usage(
        self, hash_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Estimate memory required to load weights.

        Useful for planning batch sizes and checking if weights will fit in RAM.

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

        # Sort by size descending
        weight_sizes.sort(key=lambda x: x[1], reverse=True)

        return {
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "total_gb": total_bytes / (1024 * 1024 * 1024),
            "num_weights": len(weight_sizes),
            "largest_weights": weight_sizes[:10],
            "average_bytes": total_bytes / len(weight_sizes) if weight_sizes else 0,
        }

    def load_by_pattern(
        self,
        name_pattern: str,
        max_count: Optional[int] = None,
    ) -> Dict[str, WeightTensor]:
        """
        Load weights matching a name pattern (useful for sparse checkout).

        Args:
            name_pattern: Regex pattern to match against weight names
                Example: "encoder.*" for all encoder layers
                Example: "layer\\.[0-9]+\\.weight" for layer weights

            max_count: Maximum number of weights to load (for memory safety)

        Returns:
            Dict mapping hash keys to matching WeightTensor objects
        """
        import re

        pattern = re.compile(name_pattern)
        result: Dict[str, WeightTensor] = {}
        count = 0

        for hash_key in self.list_weights():
            if max_count is not None and count >= max_count:
                break

            metadata = self.get_metadata(hash_key)
            if metadata and pattern.match(metadata.name):
                weight = self.load(hash_key)
                if weight is not None:
                    result[hash_key] = weight
                    count += 1

        return result

    def __repr__(self) -> str:
        info = self.get_storage_info()
        delta_info = self.get_delta_storage_info()
        return (
            f"HDF5Store(filepath='{self.filepath}', "
            f"weights={info['total_weights']}, "
            f"deltas={delta_info['total_deltas']}, "
            f"compression={self.compression})"
        )
