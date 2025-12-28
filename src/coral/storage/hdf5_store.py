"""HDF5-based storage backend for weight tensors"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

import h5py
import numpy as np

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.delta.delta_encoder import Delta
from coral.storage.weight_store import WeightStore

logger = logging.getLogger(__name__)


class HDF5Store(WeightStore):
    """
    HDF5-based storage backend for weight tensors.

    Features:
    - Content-addressable storage using hash-based keys
    - Compression support (gzip, lzf)
    - Efficient batch operations
    - Metadata stored as HDF5 attributes
    """

    def __init__(
        self,
        filepath: Union[str, os.PathLike],
        compression: Optional[str] = "gzip",
        compression_opts: Optional[int] = 4,
        mode: str = "a",
    ):
        """
        Initialize HDF5 storage.

        Args:
            filepath: Path to HDF5 file (str or Path-like object)
            compression: Compression algorithm ('gzip', 'lzf', None)
            compression_opts: Compression level (1-9 for gzip)
            mode: File mode ('r', 'r+', 'w', 'a')
        """
        self.filepath = Path(filepath)
        self.compression = compression
        self.compression_opts = compression_opts
        self.mode = mode

        # Validate mode parameter
        valid_modes = {'r', 'r+', 'w', 'w-', 'x', 'a'}
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid HDF5 file mode: '{mode}'. "
                f"Valid modes are: {', '.join(sorted(valid_modes))}"
            )

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

    def store(self, weight: WeightTensor, hash_key: Optional[str] = None) -> str:
        """Store a weight tensor"""
        if hash_key is None:
            hash_key = weight.compute_hash()

        # Check if already exists
        if self.exists(hash_key):
            logger.debug(f"Weight {hash_key} already exists in storage")
            return hash_key

        # Store weight data
        weights_group = self.file["weights"]
        # Only pass compression_opts for gzip (lzf doesn't accept options)
        create_kwargs = {
            "data": weight.data,
            "compression": self.compression,
        }
        if self.compression == "gzip" and self.compression_opts is not None:
            create_kwargs["compression_opts"] = self.compression_opts
        dataset = weights_group.create_dataset(hash_key, **create_kwargs)

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

    def list_weights(self) -> list[str]:
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

    def store_batch(self, weights: dict[str, WeightTensor]) -> dict[str, str]:
        """Store multiple weights efficiently"""
        result = {}

        for name, weight in weights.items():
            hash_key = self.store(weight)
            result[name] = hash_key

        return result

    def load_batch(self, hash_keys: list[str]) -> dict[str, WeightTensor]:
        """Load multiple weights efficiently"""
        result = {}

        for hash_key in hash_keys:
            weight = self.load(hash_key)
            if weight is not None:
                result[hash_key] = weight

        return result

    def get_storage_info(self) -> dict[str, Any]:
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

    def __enter__(self) -> HDF5Store:
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
        # Only pass compression_opts for gzip (lzf doesn't accept options)
        create_kwargs = {
            "data": delta.data,
            "compression": self.compression,
        }
        if self.compression == "gzip" and self.compression_opts is not None:
            create_kwargs["compression_opts"] = self.compression_opts
        dataset = deltas_group.create_dataset(delta_hash, **create_kwargs)

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

    def list_deltas(self) -> list[str]:
        """List all delta hashes in storage."""
        return list(self.file["deltas"].keys())

    def get_delta_storage_info(self) -> dict[str, Any]:
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

    def __repr__(self) -> str:
        info = self.get_storage_info()
        delta_info = self.get_delta_storage_info()
        return (
            f"HDF5Store(filepath='{self.filepath}', "
            f"weights={info['total_weights']}, "
            f"deltas={delta_info['total_deltas']}, "
            f"compression={self.compression})"
        )
