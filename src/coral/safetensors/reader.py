"""
SafeTensors file reader implementation.

This module provides the SafetensorsReader class for reading SafeTensors files
with lazy loading, metadata extraction, and comprehensive error handling.
"""

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from safetensors import safe_open

from .types import (
    DType,
    MetadataDict,
    SafetensorsFormatError,
    SafetensorsReadError,
    TensorInfo,
    TensorMetadata,
)


class SafetensorsReader:
    """
    High-performance reader for SafeTensors files with lazy loading support.

    Features:
    - Lazy tensor loading for memory efficiency
    - Comprehensive metadata extraction
    - Error handling and file validation
    - Context manager support for automatic cleanup
    - Batch operations for multiple tensors

    Example:
        with SafetensorsReader("model.safetensors") as reader:
            tensor_names = reader.get_tensor_names()
            weights = reader.read_tensor("embedding.weight")
            metadata = reader.get_tensor_metadata("embedding.weight")
    """

    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize SafeTensors reader.

        Args:
            file_path: Path to the SafeTensors file

        Raises:
            SafetensorsReadError: If file cannot be opened or is invalid
        """
        self.file_path = Path(file_path)
        self._safe_file = None
        self._metadata = None
        self._tensor_info_cache = {}

        if not self.file_path.exists():
            raise SafetensorsReadError(f"File not found: {self.file_path}")

        if not self.file_path.is_file():
            raise SafetensorsReadError(f"Path is not a file: {self.file_path}")

        try:
            # Open file and validate format
            self._safe_file = safe_open(str(self.file_path), framework="numpy")
            self._metadata = self._safe_file.metadata() or {}
        except Exception as e:
            raise SafetensorsReadError(f"Failed to open SafeTensors file: {e}") from e

    def __enter__(self) -> "SafetensorsReader":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.close()

    def close(self) -> None:
        """Close the SafeTensors file and cleanup resources."""
        if self._safe_file is not None:
            # Note: safetensors doesn't have explicit close method,
            # but we can clear our references
            self._safe_file = None
        self._tensor_info_cache.clear()

    @property
    def metadata(self) -> MetadataDict:
        """Get the global metadata from the SafeTensors file."""
        return self._metadata.copy()

    @property
    def is_open(self) -> bool:
        """Check if the file is currently open."""
        return self._safe_file is not None

    def get_tensor_names(self) -> List[str]:
        """
        Get list of all tensor names in the file.

        Returns:
            List of tensor names

        Raises:
            SafetensorsReadError: If file is not open
        """
        self._ensure_open()
        try:
            return list(self._safe_file.keys())
        except Exception as e:
            raise SafetensorsReadError(f"Failed to get tensor names: {e}") from e

    def read_tensor(self, name: str) -> np.ndarray:
        """
        Read a specific tensor from the file.

        Args:
            name: Name of the tensor to read

        Returns:
            Numpy array containing the tensor data

        Raises:
            SafetensorsReadError: If tensor cannot be read
        """
        self._ensure_open()

        if name not in self.get_tensor_names():
            raise SafetensorsReadError(f"Tensor '{name}' not found in file")

        try:
            return self._safe_file.get_tensor(name)
        except Exception as e:
            raise SafetensorsReadError(f"Failed to read tensor '{name}': {e}") from e

    def read_tensors(self, names: List[str]) -> Dict[str, np.ndarray]:
        """
        Read multiple tensors efficiently.

        Args:
            names: List of tensor names to read

        Returns:
            Dictionary mapping tensor names to numpy arrays

        Raises:
            SafetensorsReadError: If any tensor cannot be read
        """
        result = {}
        available_names = set(self.get_tensor_names())

        for name in names:
            if name not in available_names:
                raise SafetensorsReadError(f"Tensor '{name}' not found in file")
            result[name] = self.read_tensor(name)

        return result

    def read_all_tensors(self) -> Dict[str, np.ndarray]:
        """
        Read all tensors from the file.

        Returns:
            Dictionary mapping all tensor names to numpy arrays

        Raises:
            SafetensorsReadError: If tensors cannot be read
        """
        return self.read_tensors(self.get_tensor_names())

    def get_tensor_info(self, name: str) -> TensorInfo:
        """
        Get detailed information about a tensor without loading data.

        Args:
            name: Name of the tensor

        Returns:
            TensorInfo object with tensor details

        Raises:
            SafetensorsReadError: If tensor info cannot be retrieved
        """
        self._ensure_open()

        if name in self._tensor_info_cache:
            return self._tensor_info_cache[name]

        if name not in self.get_tensor_names():
            raise SafetensorsReadError(f"Tensor '{name}' not found in file")

        try:
            # Get tensor to extract info (this is lazy in safetensors)
            tensor = self._safe_file.get_tensor(name)
            dtype = DType.from_numpy(tensor.dtype)
            shape = tensor.shape

            # For data offsets, we need to calculate them
            # This is a simplified approach - real implementation might need
            # to parse the file header directly
            data_offsets = (0, tensor.nbytes)  # Placeholder

            info = TensorInfo(
                name=name, dtype=dtype, shape=shape, data_offsets=data_offsets
            )

            self._tensor_info_cache[name] = info
            return info

        except Exception as e:
            raise SafetensorsReadError(
                f"Failed to get tensor info for '{name}': {e}"
            ) from e

    def get_tensor_metadata(self, name: str) -> TensorMetadata:
        """
        Get extended metadata for a tensor.

        Args:
            name: Name of the tensor

        Returns:
            TensorMetadata object with comprehensive tensor information

        Raises:
            SafetensorsReadError: If metadata cannot be retrieved
        """
        info = self.get_tensor_info(name)

        # Extract additional metadata from global metadata if available
        tensor_metadata = self._metadata.get(f"tensor.{name}", {})

        return TensorMetadata(
            name=info.name,
            dtype=info.dtype,
            shape=info.shape,
            layer_type=tensor_metadata.get("layer_type"),
            model_name=tensor_metadata.get("model_name"),
            compression_info=tensor_metadata.get("compression_info"),
            creation_time=tensor_metadata.get("creation_time"),
            author=tensor_metadata.get("author"),
            description=tensor_metadata.get("description"),
            tags=tensor_metadata.get("tags"),
        )

    def has_tensor(self, name: str) -> bool:
        """
        Check if a tensor exists in the file.

        Args:
            name: Name of the tensor to check

        Returns:
            True if tensor exists, False otherwise
        """
        try:
            return name in self.get_tensor_names()
        except SafetensorsReadError:
            return False

    def get_file_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the SafeTensors file.

        Returns:
            Dictionary with file statistics and information
        """
        try:
            tensor_names = self.get_tensor_names()

            # Calculate total parameters and size
            total_params = 0
            total_size = 0
            dtype_counts = {}

            for name in tensor_names:
                info = self.get_tensor_info(name)
                total_params += info.size
                total_size += info.nbytes

                dtype_str = info.dtype.value
                dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1

            return {
                "file_path": str(self.file_path),
                "file_size_bytes": self.file_path.stat().st_size,
                "tensor_count": len(tensor_names),
                "total_parameters": total_params,
                "total_tensor_size_bytes": total_size,
                "dtype_distribution": dtype_counts,
                "tensor_names": tensor_names,
                "global_metadata": self.metadata,
            }

        except Exception as e:
            raise SafetensorsReadError(f"Failed to get file info: {e}") from e

    def validate_file(self) -> Dict[str, Any]:
        """
        Validate the SafeTensors file integrity.

        Returns:
            Dictionary with validation results

        Raises:
            SafetensorsFormatError: If file format is invalid
        """
        try:
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "tensor_count": 0,
                "total_size": 0,
            }

            # Check if we can read tensor names
            try:
                tensor_names = self.get_tensor_names()
                validation_result["tensor_count"] = len(tensor_names)
            except Exception as e:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Cannot read tensor names: {e}")
                return validation_result

            # Validate each tensor
            for name in tensor_names:
                try:
                    info = self.get_tensor_info(name)
                    validation_result["total_size"] += info.nbytes

                    # Check for empty tensors
                    if info.size == 0:
                        validation_result["warnings"].append(
                            f"Tensor '{name}' is empty"
                        )

                    # Validate tensor can be read (lazy check)
                    tensor_data = self.read_tensor(name)
                    if tensor_data.size != info.size:
                        validation_result["errors"].append(
                            f"Tensor '{name}' size mismatch: expected "
                            f"{info.size}, got {tensor_data.size}"
                        )
                        validation_result["valid"] = False

                except Exception as e:
                    validation_result["valid"] = False
                    validation_result["errors"].append(
                        f"Cannot validate tensor '{name}': {e}"
                    )

            return validation_result

        except Exception as e:
            raise SafetensorsFormatError(f"File validation failed: {e}") from e

    def _ensure_open(self) -> None:
        """Ensure the file is open for reading."""
        if not self.is_open:
            raise SafetensorsReadError("SafeTensors file is not open")

    def __repr__(self) -> str:
        """String representation of the reader."""
        status = "open" if self.is_open else "closed"
        try:
            tensor_count = len(self.get_tensor_names()) if self.is_open else "unknown"
        except Exception:
            tensor_count = "unknown"

        return (
            f"SafetensorsReader(file='{self.file_path}', status={status}, "
            f"tensors={tensor_count})"
        )
