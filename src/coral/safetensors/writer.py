"""Safetensors file writer implementation."""

import json
import logging
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from coral.safetensors.types import (
    DType,
    MetadataDict,
    SafetensorsError,
    SafetensorsIOError,
    TensorDict,
    TensorInfo,
)

logger = logging.getLogger(__name__)


class SafetensorsWriter:
    """Writer for Safetensors format files.

    This class provides functionality to write tensors to Safetensors format files,
    which store tensors in a simple, safe binary format with JSON metadata.

    The Safetensors format consists of:
    1. 8 bytes: header size (uint64 little-endian)
    2. N bytes: JSON header with tensor metadata
    3. M bytes: raw tensor data (contiguous, no gaps)

    Attributes:
        file_path: Path where the Safetensors file will be written
        metadata: File-level metadata to include

    Example:
        >>> # Write tensors one by one
        >>> writer = SafetensorsWriter("model.safetensors")
        >>> writer.add_tensor("weight", np.random.randn(10, 10))
        >>> writer.add_tensor("bias", np.zeros(10))
        >>> writer.write()
        
        >>> # Write tensors using context manager
        >>> with SafetensorsWriter("model.safetensors") as writer:
        ...     writer.add_tensor("layer1.weight", layer1_weight)
        ...     writer.add_tensor("layer1.bias", layer1_bias)
        
        >>> # Write tensors in one call
        >>> SafetensorsWriter.save_tensors(
        ...     "model.safetensors",
        ...     {"weight": weight_array, "bias": bias_array},
        ...     metadata={"model": "my_model", "version": "1.0"}
        ... )
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        metadata: Optional[MetadataDict] = None,
    ) -> None:
        """Initialize SafetensorsWriter.

        Args:
            file_path: Path where the Safetensors file will be written
            metadata: Optional file-level metadata
        """
        self.file_path = Path(file_path)
        self.metadata = metadata or {}
        self._tensors: TensorDict = {}
        self._tensor_info: Dict[str, TensorInfo] = {}

    def add_tensor(self, name: str, tensor: np.ndarray) -> None:
        """Add a tensor to be written.

        Args:
            name: Name for the tensor
            tensor: NumPy array to save

        Raises:
            ValueError: If tensor name is invalid or already exists
            SafetensorsError: If tensor cannot be processed
        """
        # Validate tensor name
        if not name:
            raise ValueError("Tensor name cannot be empty")
        
        if name in self._tensors:
            raise ValueError(f"Tensor '{name}' already added")
        
        if name == "__metadata__":
            raise ValueError("Tensor name '__metadata__' is reserved")
        
        # Validate tensor
        if not isinstance(tensor, np.ndarray):
            raise SafetensorsError(f"Tensor must be a numpy array, got {type(tensor)}")
        
        # Check if dtype is supported
        try:
            dtype = DType.from_numpy(tensor.dtype)
            # Special handling for bfloat16 (not natively supported by numpy)
            if dtype == DType.BFLOAT16 and tensor.dtype != np.float32:
                raise SafetensorsError(
                    f"Tensor '{name}' marked as bfloat16 must be stored as float32 in numpy"
                )
        except ValueError as e:
            raise SafetensorsError(f"Unsupported dtype for tensor '{name}': {e}")
        
        # Check tensor size
        if tensor.size == 0:
            logger.warning(f"Tensor '{name}' is empty (size=0)")
        
        # Ensure tensor is contiguous
        if not tensor.flags['C_CONTIGUOUS']:
            logger.debug(f"Making tensor '{name}' C-contiguous")
            tensor = np.ascontiguousarray(tensor)
        
        self._tensors[name] = tensor
        logger.debug(f"Added tensor '{name}' with shape {tensor.shape} and dtype {tensor.dtype}")

    def add_tensors(self, tensors: TensorDict) -> None:
        """Add multiple tensors to be written.

        Args:
            tensors: Dictionary mapping names to numpy arrays

        Raises:
            ValueError: If any tensor name is invalid
            SafetensorsError: If any tensor cannot be processed
        """
        for name, tensor in tensors.items():
            self.add_tensor(name, tensor)

    def write(self) -> None:
        """Write all tensors to the Safetensors file.

        The file format is:
        - 8 bytes: header size (uint64 little-endian)
        - N bytes: JSON header with tensor metadata
        - M bytes: raw tensor data (contiguous, no gaps)

        Raises:
            SafetensorsIOError: If write fails
            SafetensorsError: If no tensors to write
        """
        if not self._tensors:
            raise SafetensorsError("No tensors to write")
        
        logger.info(f"Writing {len(self._tensors)} tensors to {self.file_path}")
        
        # Calculate offsets for each tensor
        current_offset = 0
        tensor_infos = []
        total_data_size = 0
        
        for name, tensor in self._tensors.items():
            # Get dtype
            dtype = DType.from_numpy(tensor.dtype)
            
            # Calculate size
            byte_size = tensor.nbytes
            total_data_size += byte_size
            
            # Create tensor info
            info = TensorInfo(
                name=name,
                shape=tensor.shape,
                dtype=dtype,
                data_offsets=(current_offset, current_offset + byte_size),
            )
            
            tensor_infos.append(info)
            self._tensor_info[name] = info
            current_offset += byte_size
            
            logger.debug(f"Tensor '{name}': offset={info.data_offsets[0]}, size={byte_size}")
        
        # Build header
        header: Dict[str, Any] = {}
        
        # Add metadata if present
        if self.metadata:
            header["__metadata__"] = self.metadata
            logger.debug(f"Including metadata: {list(self.metadata.keys())}")
        
        # Add tensor information
        for info in tensor_infos:
            header[info.name] = info.to_dict()
        
        # Serialize header to JSON (compact format)
        header_json = json.dumps(header, separators=(",", ":"), sort_keys=True)
        header_bytes = header_json.encode("utf-8")
        header_size = len(header_bytes)
        
        logger.debug(f"Header size: {header_size} bytes, data size: {total_data_size} bytes")
        
        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        try:
            with open(self.file_path, "wb") as f:
                # Write header size (8 bytes, little endian)
                f.write(struct.pack("<Q", header_size))
                
                # Write header
                f.write(header_bytes)
                
                # Write tensor data contiguously
                bytes_written = 0
                for name in self._tensors:
                    tensor = self._tensors[name]
                    tensor_bytes = tensor.tobytes()
                    f.write(tensor_bytes)
                    bytes_written += len(tensor_bytes)
                
                # Verify we wrote the expected amount of data
                if bytes_written != total_data_size:
                    raise SafetensorsError(
                        f"Data size mismatch: expected {total_data_size}, wrote {bytes_written}"
                    )
                
            logger.info(f"Successfully wrote {self.file_path} ({8 + header_size + total_data_size} bytes)")
        
        except (OSError, IOError) as e:
            logger.error(f"Failed to write file: {e}")
            raise SafetensorsIOError(f"Failed to write file: {e}")
        except Exception as e:
            logger.error(f"Unexpected error writing file: {e}")
            # Clean up partial file
            if self.file_path.exists():
                try:
                    self.file_path.unlink()
                except Exception:
                    pass
            raise

    def estimate_file_size(self) -> int:
        """Estimate the final file size before writing.

        This provides an accurate estimate by simulating the header creation
        with the actual offsets that would be used.

        Returns:
            Estimated size in bytes
        """
        if not self._tensors:
            return 0
        
        # Calculate actual offsets
        current_offset = 0
        header_dict: Dict[str, Any] = {}
        
        # Add metadata if present
        if self.metadata:
            header_dict["__metadata__"] = self.metadata
        
        # Build header with actual offsets
        for name, tensor in self._tensors.items():
            byte_size = tensor.nbytes
            header_dict[name] = {
                "shape": list(tensor.shape),
                "dtype": DType.from_numpy(tensor.dtype).value,
                "data_offsets": [current_offset, current_offset + byte_size],
            }
            current_offset += byte_size
        
        # Create actual header JSON
        header_json = json.dumps(header_dict, separators=(",", ":"), sort_keys=True)
        header_size = len(header_json.encode("utf-8"))
        
        # Total size = 8 bytes (header size) + header + tensor data
        tensor_size = sum(t.nbytes for t in self._tensors.values())
        total_size = 8 + header_size + tensor_size
        
        logger.debug(f"Estimated file size: {total_size} bytes "
                    f"(header: {header_size}, data: {tensor_size})")
        
        return total_size

    def clear(self) -> None:
        """Clear all tensors and metadata."""
        self._tensors.clear()
        self._tensor_info.clear()
        logger.debug("Cleared all tensors")
    
    def remove_tensor(self, name: str) -> None:
        """Remove a tensor that was previously added.

        Args:
            name: Name of the tensor to remove

        Raises:
            KeyError: If tensor not found
        """
        if name not in self._tensors:
            raise KeyError(f"Tensor '{name}' not found")
        
        del self._tensors[name]
        if name in self._tensor_info:
            del self._tensor_info[name]
        logger.debug(f"Removed tensor '{name}'")
    
    def get_tensor_names(self) -> list[str]:
        """Get list of tensor names that will be written.

        Returns:
            List of tensor names
        """
        return list(self._tensors.keys())
    
    def get_tensor_info(self, name: str) -> Optional[TensorInfo]:
        """Get information about a tensor.

        Args:
            name: Name of the tensor

        Returns:
            TensorInfo if tensor exists and has been processed, None otherwise
        """
        return self._tensor_info.get(name)
    
    def validate(self) -> None:
        """Validate that the writer is ready to write.

        Raises:
            SafetensorsError: If validation fails
        """
        if not self._tensors:
            raise SafetensorsError("No tensors to write")
        
        # Check for duplicate tensor references
        seen_ids = set()
        for name, tensor in self._tensors.items():
            tensor_id = id(tensor)
            if tensor_id in seen_ids:
                logger.warning(f"Tensor '{name}' shares memory with another tensor")
            seen_ids.add(tensor_id)
        
        # Validate all dtypes are supported
        for name, tensor in self._tensors.items():
            try:
                DType.from_numpy(tensor.dtype)
            except ValueError as e:
                raise SafetensorsError(f"Unsupported dtype for tensor '{name}': {e}")
        
        logger.debug("Validation passed")
    
    def __enter__(self) -> "SafetensorsWriter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - write file if no exception."""
        if exc_type is None and self._tensors:
            self.write()
    
    @staticmethod
    def save_tensors(
        file_path: Union[str, Path],
        tensors: TensorDict,
        metadata: Optional[MetadataDict] = None,
    ) -> None:
        """Convenience method to save tensors in one call.

        Args:
            file_path: Path where the Safetensors file will be written
            tensors: Dictionary mapping names to numpy arrays
            metadata: Optional file-level metadata

        Raises:
            SafetensorsIOError: If write fails
            SafetensorsError: If tensors are invalid
        """
        writer = SafetensorsWriter(file_path, metadata)
        writer.add_tensors(tensors)
        writer.write()