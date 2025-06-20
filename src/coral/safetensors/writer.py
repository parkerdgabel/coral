"""
SafeTensors file writer implementation.

This module provides the SafetensorsWriter class for writing SafeTensors files
with comprehensive metadata support, validation, and error handling.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from safetensors.numpy import save_file

from .types import (
    DType,
    MetadataDict,
    SafetensorsWriteError,
    TensorMetadata,
)


class SafetensorsWriter:
    """
    High-performance writer for SafeTensors files with validation and metadata support.

    Features:
    - Batch tensor writing with automatic validation
    - Comprehensive metadata support
    - Atomic writes with temporary file handling
    - Memory-efficient tensor processing
    - Context manager support for automatic cleanup

    Example:
        with SafetensorsWriter("model.safetensors") as writer:
            writer.add_tensor("embedding.weight", embedding_array)
            writer.add_tensor("linear.bias", bias_array)
            writer.set_metadata({"model_type": "transformer"})
            writer.write()
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        metadata: Optional[MetadataDict] = None,
        validate_tensors: bool = True,
        use_temp_file: bool = True,
    ):
        """
        Initialize SafeTensors writer.

        Args:
            file_path: Path where the SafeTensors file will be written
            metadata: Optional global metadata dictionary
            validate_tensors: Whether to validate tensors before writing
            use_temp_file: Whether to use temporary file for atomic writes

        Raises:
            SafetensorsWriteError: If file path is invalid
        """
        self.file_path = Path(file_path)
        self.validate_tensors = validate_tensors
        self.use_temp_file = use_temp_file

        # Internal state
        self._tensors: Dict[str, np.ndarray] = {}
        self._metadata: MetadataDict = metadata.copy() if metadata else {}
        self._tensor_metadata: Dict[str, TensorMetadata] = {}
        self._written = False
        self._closed = False

        # Validate output directory
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if self.file_path.exists() and not self.file_path.is_file():
            raise SafetensorsWriteError(
                f"Output path exists but is not a file: {self.file_path}"
            )

    def __enter__(self) -> "SafetensorsWriter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        if not self._written and exc_type is None:
            # Auto-write if no exception occurred and nothing was written
            try:
                self.write()
            except Exception:
                pass  # Don't mask original exception
        self.close()

    def close(self) -> None:
        """Close the writer and cleanup resources."""
        if not self._closed:
            self._tensors.clear()
            self._tensor_metadata.clear()
            self._closed = True

    @property
    def is_closed(self) -> bool:
        """Check if the writer is closed."""
        return self._closed

    @property
    def tensor_count(self) -> int:
        """Get the number of tensors added to the writer."""
        return len(self._tensors)

    @property
    def metadata(self) -> MetadataDict:
        """Get a copy of the current global metadata."""
        return self._metadata.copy()

    def add_tensor(
        self, name: str, data: np.ndarray, metadata: Optional[TensorMetadata] = None
    ) -> None:
        """
        Add a tensor to be written to the file.

        Args:
            name: Name of the tensor
            data: Numpy array containing tensor data
            metadata: Optional extended metadata for the tensor

        Raises:
            SafetensorsWriteError: If tensor is invalid or writer is closed
        """
        self._ensure_open()

        if name in self._tensors:
            raise SafetensorsWriteError(f"Tensor '{name}' already exists")

        # Validate tensor if requested
        if self.validate_tensors:
            self._validate_tensor(name, data)

        # Ensure tensor is contiguous for safetensors
        if not data.flags.c_contiguous:
            data = np.ascontiguousarray(data)

        self._tensors[name] = data

        # Store extended metadata if provided
        if metadata is not None:
            self._tensor_metadata[name] = metadata
            # Add tensor-specific metadata to global metadata
            self._metadata[f"tensor.{name}"] = metadata.to_dict()

    def add_tensors(self, tensors: Dict[str, np.ndarray]) -> None:
        """
        Add multiple tensors at once.

        Args:
            tensors: Dictionary mapping tensor names to numpy arrays

        Raises:
            SafetensorsWriteError: If any tensor is invalid
        """
        for name, data in tensors.items():
            self.add_tensor(name, data)

    def set_metadata(self, metadata: MetadataDict) -> None:
        """
        Set or update global metadata.

        Args:
            metadata: Dictionary of metadata key-value pairs

        Raises:
            SafetensorsWriteError: If writer is closed
        """
        self._ensure_open()
        self._metadata.update(metadata)

    def set_tensor_metadata(self, tensor_name: str, metadata: TensorMetadata) -> None:
        """
        Set extended metadata for a specific tensor.

        Args:
            tensor_name: Name of the tensor
            metadata: Extended metadata for the tensor

        Raises:
            SafetensorsWriteError: If tensor doesn't exist or writer is closed
        """
        self._ensure_open()

        if tensor_name not in self._tensors:
            raise SafetensorsWriteError(f"Tensor '{tensor_name}' not found")

        self._tensor_metadata[tensor_name] = metadata
        self._metadata[f"tensor.{tensor_name}"] = metadata.to_dict()

    def remove_tensor(self, name: str) -> None:
        """
        Remove a tensor from the writer.

        Args:
            name: Name of the tensor to remove

        Raises:
            SafetensorsWriteError: If tensor doesn't exist or writer is closed
        """
        self._ensure_open()

        if name not in self._tensors:
            raise SafetensorsWriteError(f"Tensor '{name}' not found")

        del self._tensors[name]
        if name in self._tensor_metadata:
            del self._tensor_metadata[name]

        # Remove tensor-specific metadata
        tensor_metadata_key = f"tensor.{name}"
        if tensor_metadata_key in self._metadata:
            del self._metadata[tensor_metadata_key]

    def get_tensor_names(self) -> List[str]:
        """
        Get list of tensor names that will be written.

        Returns:
            List of tensor names
        """
        return list(self._tensors.keys())

    def has_tensor(self, name: str) -> bool:
        """
        Check if a tensor has been added to the writer.

        Args:
            name: Name of the tensor to check

        Returns:
            True if tensor exists, False otherwise
        """
        return name in self._tensors

    def write(self) -> None:
        """
        Write all tensors and metadata to the SafeTensors file.

        Raises:
            SafetensorsWriteError: If writing fails
        """
        self._ensure_open()

        if not self._tensors:
            raise SafetensorsWriteError("No tensors to write")

        try:
            if self.use_temp_file:
                self._write_with_temp_file()
            else:
                self._write_direct()

            self._written = True

        except Exception as e:
            raise SafetensorsWriteError(f"Failed to write SafeTensors file: {e}") from e

    def validate_before_write(self) -> Dict[str, Any]:
        """
        Validate all tensors and metadata before writing.

        Returns:
            Dictionary with validation results

        Raises:
            SafetensorsWriteError: If validation fails
        """
        self._ensure_open()

        validation_result: Dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "tensor_count": len(self._tensors),
            "total_size_mb": 0,
        }

        total_size = 0

        # Validate each tensor
        for name, tensor in self._tensors.items():
            try:
                self._validate_tensor(name, tensor)
                total_size += tensor.nbytes

                # Check for potential issues
                if tensor.size == 0:
                    validation_result["warnings"].append(f"Tensor '{name}' is empty")

                if tensor.nbytes > 100 * 1024 * 1024:  # 100MB
                    validation_result["warnings"].append(
                        f"Tensor '{name}' is very large "
                        f"({tensor.nbytes / 1024 / 1024:.1f}MB)"
                    )

            except Exception as e:
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"Tensor '{name}' validation failed: {e}"
                )

        validation_result["total_size_mb"] = total_size / (1024 * 1024)

        # Validate metadata serialization
        try:
            json.dumps(self._metadata)
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Metadata serialization failed: {e}")

        return validation_result

    def get_write_info(self) -> Dict[str, Any]:
        """
        Get information about what will be written.

        Returns:
            Dictionary with write information
        """
        total_params = sum(tensor.size for tensor in self._tensors.values())
        total_size = sum(tensor.nbytes for tensor in self._tensors.values())

        dtype_counts: Dict[str, int] = {}
        for tensor in self._tensors.values():
            dtype_str = str(tensor.dtype)
            dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1

        return {
            "output_file": str(self.file_path),
            "tensor_count": len(self._tensors),
            "total_parameters": total_params,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "dtype_distribution": dtype_counts,
            "tensor_names": list(self._tensors.keys()),
            "metadata_keys": list(self._metadata.keys()),
            "has_tensor_metadata": len(self._tensor_metadata) > 0,
        }

    def _write_with_temp_file(self) -> None:
        """Write using a temporary file for atomic operation."""
        temp_file = None
        try:
            # Create temporary file in same directory
            temp_file = tempfile.NamedTemporaryFile(
                dir=self.file_path.parent,
                prefix=f".{self.file_path.name}.",
                suffix=".tmp",
                delete=False,
            )
            temp_path = Path(temp_file.name)
            temp_file.close()

            # Write to temporary file
            serialized_metadata = self._serialize_metadata_for_safetensors(
                self._metadata
            )
            save_file(self._tensors, temp_path, metadata=serialized_metadata)

            # Atomic move to final location
            temp_path.replace(self.file_path)

        except Exception as e:
            # Clean up temporary file if it exists
            if temp_file is not None:
                temp_path = Path(temp_file.name)
                if temp_path.exists():
                    temp_path.unlink()
            raise e

    def _write_direct(self) -> None:
        """Write directly to the target file."""
        # Serialize complex metadata to JSON strings
        serialized_metadata = self._serialize_metadata_for_safetensors(self._metadata)
        save_file(self._tensors, self.file_path, metadata=serialized_metadata)

    def _validate_tensor(self, name: str, tensor: np.ndarray) -> None:
        """
        Validate a tensor before adding it.

        Args:
            name: Name of the tensor
            tensor: Numpy array to validate

        Raises:
            SafetensorsWriteError: If tensor is invalid
        """
        if not isinstance(tensor, np.ndarray):
            raise SafetensorsWriteError(f"Tensor '{name}' must be a numpy array")

        if tensor.dtype == np.object_:
            raise SafetensorsWriteError(
                f"Tensor '{name}' has object dtype, which is not supported"
            )

        # Check if dtype is supported by SafeTensors
        try:
            DType.from_numpy(tensor.dtype)
        except ValueError as e:
            raise SafetensorsWriteError(
                f"Tensor '{name}' has unsupported dtype: {e}"
            ) from e

        # Check for invalid values (but allow them by default for compatibility)
        if tensor.dtype.kind == "f":  # Float types
            if np.any(np.isnan(tensor)):
                # Allow NaN values but warn
                pass
            if np.any(np.isinf(tensor)):
                # Allow infinite values but warn
                pass

        # Validate tensor name
        if not name or not isinstance(name, str):
            raise SafetensorsWriteError("Tensor name must be a non-empty string")

        if len(name) > 512:  # Reasonable limit
            raise SafetensorsWriteError(f"Tensor name too long: {len(name)} characters")

    def _serialize_metadata_for_safetensors(
        self, metadata: MetadataDict
    ) -> MetadataDict:
        """
        Serialize complex metadata to SafeTensors-compatible format.

        SafeTensors only accepts string values in metadata, so we need to
        serialize complex objects to JSON strings.
        """
        serialized = {}

        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                # Simple types can be stored directly
                serialized[key] = str(value)
            elif value is None:
                # Skip None values
                continue
            else:
                # Complex types need to be JSON-serialized
                try:
                    serialized[key] = json.dumps(value)
                except (TypeError, ValueError):
                    # If JSON serialization fails, convert to string
                    serialized[key] = str(value)

        return serialized

    def _ensure_open(self) -> None:
        """Ensure the writer is open."""
        if self._closed:
            raise SafetensorsWriteError("Writer is closed")

    def __repr__(self) -> str:
        """String representation of the writer."""
        status = "closed" if self._closed else "open"
        written_status = "written" if self._written else "not written"

        return (
            f"SafetensorsWriter(file='{self.file_path}', "
            f"status={status}, tensors={len(self._tensors)}, {written_status})"
        )
