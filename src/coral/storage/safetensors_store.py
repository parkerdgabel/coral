"""SafeTensors-based storage backend for Coral weight management"""

import gzip
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.storage.weight_store import WeightStore

logger = logging.getLogger(__name__)


class SafetensorsStore(WeightStore):
    """
    SafeTensors-based storage backend for weight tensors.

    This implementation stores each weight tensor as a separate .safetensors file
    in a directory structure. Metadata is stored in the safetensors format's
    __metadata__ field for efficient retrieval without loading the full tensor.

    Features:
    - Lazy loading with SafetensorsReader
    - Batch operations for efficiency
    - File locking for concurrent access
    - Optional gzip compression
    - Automatic directory creation
    """

    def __init__(
        self,
        storage_path: str,
        use_compression: bool = False,
        compression_level: int = 6,
    ):
        """
        Initialize SafetensorsStore.

        Args:
            storage_path: Directory path for storing safetensors files
            use_compression: Whether to use gzip compression
            compression_level: Gzip compression level (1-9)
        """
        self.storage_path = Path(storage_path)
        self.use_compression = use_compression
        self.compression_level = compression_level
        self._lock = threading.RLock()

        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized SafetensorsStore at {self.storage_path}")

    def _get_file_path(self, hash_key: str) -> Path:
        """Get the file path for a given hash key."""
        extension = ".safetensors.gz" if self.use_compression else ".safetensors"
        return self.storage_path / f"{hash_key}{extension}"

    def _serialize_metadata(self, metadata: WeightMetadata) -> Dict[str, str]:
        """Convert WeightMetadata to dict for safetensors metadata."""
        return {
            "__coral_name": metadata.name,
            "__coral_shape": json.dumps(list(metadata.shape)),
            "__coral_dtype": np.dtype(
                metadata.dtype
            ).name,  # Store dtype name, not class string
            "__coral_layer_type": metadata.layer_type or "",
            "__coral_model_name": metadata.model_name or "",
            "__coral_compression_info": json.dumps(metadata.compression_info),
            "__coral_hash": metadata.hash or "",
        }

    def _deserialize_metadata(
        self, safetensors_metadata: Dict[str, str]
    ) -> WeightMetadata:
        """Convert safetensors metadata back to WeightMetadata."""
        return WeightMetadata(
            name=safetensors_metadata.get("__coral_name", "unnamed"),
            shape=tuple(json.loads(safetensors_metadata.get("__coral_shape", "[1]"))),
            dtype=np.dtype(safetensors_metadata.get("__coral_dtype", "float32")),
            layer_type=safetensors_metadata.get("__coral_layer_type") or None,
            model_name=safetensors_metadata.get("__coral_model_name") or None,
            compression_info=json.loads(
                safetensors_metadata.get("__coral_compression_info", "{}")
            ),
            hash=safetensors_metadata.get("__coral_hash") or None,
        )

    def store(self, weight: WeightTensor, hash_key: Optional[str] = None) -> str:
        """
        Store a weight tensor.

        Args:
            weight: WeightTensor to store
            hash_key: Optional hash to use as key (will compute if not provided)

        Returns:
            Hash key used for storage
        """
        with self._lock:
            # Compute hash if not provided
            if hash_key is None:
                hash_key = weight.compute_hash()

            file_path = self._get_file_path(hash_key)

            # Skip if already exists
            if file_path.exists():
                logger.debug(f"Weight {hash_key} already exists, skipping store")
                return hash_key

            # Prepare data for safetensors
            tensors = {"weight": weight.data}
            metadata = self._serialize_metadata(weight.metadata)

            # Save to temporary file first for atomicity
            temp_path = file_path.with_suffix(".tmp")

            try:
                if self.use_compression:
                    # Save to temp file first, then compress
                    uncompressed_temp = temp_path.with_suffix(".safetensors")
                    save_file(tensors, uncompressed_temp, metadata=metadata)

                    # Read and compress the file
                    with open(uncompressed_temp, "rb") as f_in:
                        with gzip.open(
                            temp_path, "wb", compresslevel=self.compression_level
                        ) as f_out:
                            f_out.write(f_in.read())

                    # Clean up uncompressed temp file
                    uncompressed_temp.unlink()
                else:
                    save_file(tensors, temp_path, metadata=metadata)

                # Atomic rename
                temp_path.rename(file_path)

                logger.debug(f"Stored weight {hash_key} to {file_path}")
                return hash_key

            except Exception as e:
                # Clean up temporary file on error
                if temp_path.exists():
                    temp_path.unlink()
                raise RuntimeError(f"Failed to store weight {hash_key}: {e}") from e

    def load(self, hash_key: str) -> Optional[WeightTensor]:
        """
        Load a weight tensor by hash.

        Args:
            hash_key: Hash key of the weight

        Returns:
            WeightTensor if found, None otherwise
        """
        with self._lock:
            file_path = self._get_file_path(hash_key)

            if not file_path.exists():
                logger.debug(f"Weight {hash_key} not found")
                return None

            try:
                if self.use_compression:
                    # Decompress to temporary file
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        suffix=".safetensors", delete=False
                    ) as tmp_file:
                        with gzip.open(file_path, "rb") as f_in:
                            tmp_file.write(f_in.read())
                        tmp_path = tmp_file.name

                    try:
                        with safe_open(tmp_path, framework="numpy") as f:
                            # Get metadata
                            metadata_dict = f.metadata()
                            metadata = self._deserialize_metadata(metadata_dict)

                            # Get tensor data
                            data = f.get_tensor("weight")
                    finally:
                        # Clean up temp file
                        Path(tmp_path).unlink()
                else:
                    with safe_open(file_path, framework="numpy") as f:
                        # Get metadata
                        metadata_dict = f.metadata()
                        metadata = self._deserialize_metadata(metadata_dict)

                        # Get tensor data
                        data = f.get_tensor("weight")

                return WeightTensor(data=data, metadata=metadata)

            except Exception as e:
                logger.error(f"Failed to load weight {hash_key}: {e}")
                return None

    def exists(self, hash_key: str) -> bool:
        """Check if a weight exists in storage."""
        return self._get_file_path(hash_key).exists()

    def delete(self, hash_key: str) -> bool:
        """
        Delete a weight from storage.

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            file_path = self._get_file_path(hash_key)

            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.debug(f"Deleted weight {hash_key}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to delete weight {hash_key}: {e}")
                    return False

            return False

    def list_weights(self) -> List[str]:
        """List all weight hashes in storage."""
        weights = []

        # Find all safetensors files
        pattern = "*.safetensors.gz" if self.use_compression else "*.safetensors"

        for file_path in self.storage_path.glob(pattern):
            # Extract hash from filename
            if self.use_compression:
                hash_key = file_path.stem[: -len(".safetensors")]
            else:
                hash_key = file_path.stem

            weights.append(hash_key)

        return weights

    def get_metadata(self, hash_key: str) -> Optional[WeightMetadata]:
        """Get metadata for a weight without loading data."""
        with self._lock:
            file_path = self._get_file_path(hash_key)

            if not file_path.exists():
                return None

            try:
                if self.use_compression:
                    # Decompress to temporary file
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        suffix=".safetensors", delete=False
                    ) as tmp_file:
                        with gzip.open(file_path, "rb") as f_in:
                            tmp_file.write(f_in.read())
                        tmp_path = tmp_file.name

                    try:
                        with safe_open(tmp_path, framework="numpy") as f:
                            metadata_dict = f.metadata()
                    finally:
                        # Clean up temp file
                        Path(tmp_path).unlink()
                else:
                    with safe_open(file_path, framework="numpy") as f:
                        metadata_dict = f.metadata()

                return self._deserialize_metadata(metadata_dict)

            except Exception as e:
                logger.error(f"Failed to get metadata for {hash_key}: {e}")
                return None

    def store_batch(self, weights: Dict[str, WeightTensor]) -> Dict[str, str]:
        """
        Store multiple weights efficiently.

        Args:
            weights: Dict mapping names to WeightTensors

        Returns:
            Dict mapping names to storage hashes
        """
        results = {}

        for name, weight in weights.items():
            try:
                hash_key = self.store(weight)
                results[name] = hash_key
            except Exception as e:
                logger.error(f"Failed to store weight {name}: {e}")
                # Continue with other weights

        return results

    def load_batch(self, hash_keys: List[str]) -> Dict[str, WeightTensor]:
        """
        Load multiple weights efficiently.

        Args:
            hash_keys: List of hash keys to load

        Returns:
            Dict mapping hash keys to WeightTensors
        """
        results = {}

        for hash_key in hash_keys:
            weight = self.load(hash_key)
            if weight is not None:
                results[hash_key] = weight
            else:
                logger.warning(f"Weight {hash_key} not found during batch load")

        return results

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about storage usage and statistics."""
        total_size = 0
        file_count = 0

        pattern = "*.safetensors.gz" if self.use_compression else "*.safetensors"

        for file_path in self.storage_path.glob(pattern):
            total_size += file_path.stat().st_size
            file_count += 1

        return {
            "storage_path": str(self.storage_path),
            "use_compression": self.use_compression,
            "compression_level": self.compression_level,
            "total_files": file_count,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "average_file_size_bytes": total_size / file_count if file_count > 0 else 0,
        }

    def close(self) -> None:
        """Close the storage backend and cleanup resources."""
        # SafeTensors doesn't require explicit cleanup
        logger.info(f"Closed SafetensorsStore at {self.storage_path}")

    def __repr__(self) -> str:
        return (
            f"SafetensorsStore(path={self.storage_path}, "
            f"compression={self.use_compression})"
        )
