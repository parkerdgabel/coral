"""Safetensors file reader implementation."""

import json
import logging
import mmap
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from coral.safetensors.types import (
    DType,
    MetadataDict,
    SafetensorsFormatError,
    SafetensorsIOError,
    TensorInfo,
)

# Configure logger
logger = logging.getLogger(__name__)


class SafetensorsReader:
    """Reader for Safetensors format files.

    This class provides functionality to read tensors from Safetensors format files,
    which store tensors in a simple, safe binary format with JSON metadata.

    The Safetensors format is:
    [8 bytes header size][JSON header][raw tensor data]

    Features:
    - Memory-mapped file access for efficient reading
    - Lazy loading of tensor data
    - Zero-copy operations where possible
    - Comprehensive error handling and validation

    Attributes:
        file_path: Path to the Safetensors file
        header: Parsed header containing tensor information
        metadata: File-level metadata
    """

    def __init__(self, file_path: Union[str, Path], use_mmap: bool = True) -> None:
        """Initialize SafetensorsReader.

        Args:
            file_path: Path to the Safetensors file to read
            use_mmap: Whether to use memory mapping for file access

        Raises:
            SafetensorsIOError: If file cannot be opened
            SafetensorsFormatError: If file format is invalid
        """
        self.file_path = Path(file_path)
        self.use_mmap = use_mmap
        self._file_handle: Optional[Any] = None
        self._mmap: Optional[mmap.mmap] = None
        self._header: Optional[Dict[str, Any]] = None
        self._metadata: Optional[MetadataDict] = None
        self._tensor_info: Dict[str, TensorInfo] = {}
        self._header_size: int = 0
        self._data_offset: int = 0  # Offset where tensor data starts
        
        if not self.file_path.exists():
            raise SafetensorsIOError(f"File not found: {self.file_path}")
        
        logger.debug(f"Opening safetensors file: {self.file_path}")
        self._load_header()
        
        # Open memory map if requested
        if self.use_mmap:
            self._open_mmap()

    def _load_header(self) -> None:
        """Load and parse the Safetensors header.

        Raises:
            SafetensorsFormatError: If header is malformed
            SafetensorsIOError: If file cannot be read
        """
        try:
            with open(self.file_path, "rb") as f:
                # Read header size (8 bytes, little endian uint64)
                header_size_bytes = f.read(8)
                if len(header_size_bytes) != 8:
                    raise SafetensorsFormatError(
                        f"Invalid header size: expected 8 bytes, got {len(header_size_bytes)}"
                    )
                
                self._header_size = struct.unpack("<Q", header_size_bytes)[0]
                logger.debug(f"Header size: {self._header_size} bytes")
                
                # Validate header size
                if self._header_size == 0:
                    raise SafetensorsFormatError("Header size cannot be zero")
                if self._header_size > 100 * 1024 * 1024:  # 100MB limit for header
                    raise SafetensorsFormatError(
                        f"Header size too large: {self._header_size} bytes"
                    )
                
                # Read header JSON
                header_bytes = f.read(self._header_size)
                if len(header_bytes) != self._header_size:
                    raise SafetensorsFormatError(
                        f"Incomplete header: expected {self._header_size} bytes, "
                        f"got {len(header_bytes)}"
                    )
                
                # Parse header
                try:
                    self._header = json.loads(header_bytes.decode("utf-8"))
                except UnicodeDecodeError as e:
                    raise SafetensorsFormatError(f"Header is not valid UTF-8: {e}")
                
                # Calculate data offset
                self._data_offset = 8 + self._header_size
                
                # Extract metadata and tensor info
                self._metadata = self._header.get("__metadata__", {})
                logger.debug(f"Found metadata: {list(self._metadata.keys())}")
                
                # Parse tensor information
                tensor_count = 0
                for name, info in self._header.items():
                    if name == "__metadata__":
                        continue
                    
                    # Validate tensor info structure
                    if not isinstance(info, dict):
                        raise SafetensorsFormatError(
                            f"Invalid tensor info for '{name}': expected dict, got {type(info)}"
                        )
                    
                    required_fields = ["shape", "dtype", "data_offsets"]
                    for field in required_fields:
                        if field not in info:
                            raise SafetensorsFormatError(
                                f"Missing required field '{field}' for tensor '{name}'"
                            )
                    
                    # Validate data offsets
                    offsets = info["data_offsets"]
                    if not isinstance(offsets, list) or len(offsets) != 2:
                        raise SafetensorsFormatError(
                            f"Invalid data_offsets for tensor '{name}': {offsets}"
                        )
                    
                    if offsets[0] < 0 or offsets[1] <= offsets[0]:
                        raise SafetensorsFormatError(
                            f"Invalid offset range for tensor '{name}': {offsets}"
                        )
                    
                    self._tensor_info[name] = TensorInfo(
                        name=name,
                        shape=tuple(info["shape"]),
                        dtype=DType(info["dtype"]),
                        data_offsets=tuple(info["data_offsets"]),
                    )
                    tensor_count += 1
                
                logger.info(f"Loaded header with {tensor_count} tensors")
        
        except (OSError, IOError) as e:
            logger.error(f"Failed to read file: {e}")
            raise SafetensorsIOError(f"Failed to read file: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON header: {e}")
            raise SafetensorsFormatError(f"Invalid JSON header: {e}")
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid header format: {e}")
            raise SafetensorsFormatError(f"Invalid header format: {e}")

    def _open_mmap(self) -> None:
        """Open memory map for the file.
        
        Raises:
            SafetensorsIOError: If memory mapping fails
        """
        try:
            self._file_handle = open(self.file_path, "rb")
            self._mmap = mmap.mmap(self._file_handle.fileno(), 0, access=mmap.ACCESS_READ)
            logger.debug(f"Opened memory map for file: {self.file_path}")
        except (OSError, IOError) as e:
            logger.error(f"Failed to create memory map: {e}")
            if self._file_handle:
                self._file_handle.close()
                self._file_handle = None
            raise SafetensorsIOError(f"Failed to create memory map: {e}")
    
    def _close_mmap(self) -> None:
        """Close memory map and file handle."""
        if self._mmap:
            try:
                self._mmap.close()
            except Exception as e:
                logger.warning(f"Error closing memory map: {e}")
            self._mmap = None
        
        if self._file_handle:
            try:
                self._file_handle.close()
            except Exception as e:
                logger.warning(f"Error closing file handle: {e}")
            self._file_handle = None

    @property
    def header(self) -> Dict[str, Any]:
        """Get the parsed header.

        Returns:
            Header dictionary
        """
        return self._header or {}

    @property
    def metadata(self) -> MetadataDict:
        """Get file-level metadata.

        Returns:
            Metadata dictionary
        """
        return self._metadata or {}

    def get_tensor_names(self) -> List[str]:
        """Get list of all tensor names in the file.

        Returns:
            List of tensor names
        """
        return list(self._tensor_info.keys())

    def get_tensor_info(self, name: str) -> Optional[TensorInfo]:
        """Get information about a specific tensor.

        Args:
            name: Tensor name

        Returns:
            TensorInfo if tensor exists, None otherwise
        """
        return self._tensor_info.get(name)

    def read_tensor(self, name: str, copy: bool = True) -> np.ndarray:
        """Read a single tensor from the file.

        Args:
            name: Name of the tensor to read
            copy: Whether to return a copy of the data (True) or a view (False).
                  When False and using mmap, returns a zero-copy view.

        Returns:
            NumPy array containing the tensor data

        Raises:
            KeyError: If tensor name not found
            SafetensorsIOError: If read fails
        """
        if name not in self._tensor_info:
            raise KeyError(f"Tensor '{name}' not found in file")
        
        info = self._tensor_info[name]
        logger.debug(f"Reading tensor '{name}' with shape {info.shape} and dtype {info.dtype}")
        
        try:
            # Calculate absolute offset in file
            offset = self._data_offset + info.data_offsets[0]
            num_bytes = info.byte_size
            
            # Use memory map if available for zero-copy operation
            if self._mmap and not copy:
                # Create a numpy array view directly from mmap
                dtype = info.dtype.to_numpy()
                
                # Special handling for bfloat16
                if info.dtype == DType.BFLOAT16:
                    logger.warning(
                        f"Tensor '{name}' uses bfloat16 which is not natively supported "
                        "by NumPy. Loading as float32."
                    )
                
                # Create array from memory map (zero-copy)
                tensor = np.frombuffer(
                    self._mmap,
                    dtype=dtype,
                    count=np.prod(info.shape),
                    offset=offset
                )
                
                # Reshape without copying
                tensor = tensor.reshape(info.shape)
                
                # Make read-only to prevent accidental modifications
                tensor.flags.writeable = False
                
                logger.debug(f"Created zero-copy view for tensor '{name}'")
                return tensor
            
            else:
                # Fall back to file reading
                if self._mmap:
                    # Read from mmap and copy
                    data_bytes = self._mmap[offset:offset + num_bytes]
                else:
                    # Read from file
                    with open(self.file_path, "rb") as f:
                        f.seek(offset)
                        data_bytes = f.read(num_bytes)
                        
                        if len(data_bytes) != num_bytes:
                            raise SafetensorsIOError(
                                f"Incomplete tensor data for '{name}': "
                                f"expected {num_bytes} bytes, got {len(data_bytes)}"
                            )
                
                # Convert to numpy array
                dtype = info.dtype.to_numpy()
                
                # Special handling for bfloat16
                if info.dtype == DType.BFLOAT16:
                    logger.warning(
                        f"Tensor '{name}' uses bfloat16 which is not natively supported "
                        "by NumPy. Loading as float32."
                    )
                
                tensor = np.frombuffer(data_bytes, dtype=dtype)
                
                # Reshape to correct shape
                return tensor.reshape(info.shape)
        
        except (OSError, IOError) as e:
            logger.error(f"Failed to read tensor '{name}': {e}")
            raise SafetensorsIOError(f"Failed to read tensor '{name}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error reading tensor '{name}': {e}")
            raise SafetensorsIOError(f"Unexpected error reading tensor '{name}': {e}")

    def read_tensors(
        self,
        names: Optional[List[str]] = None,
        exclude: Optional[Set[str]] = None,
        copy: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Read multiple tensors from the file.

        Args:
            names: List of tensor names to read. If None, read all tensors.
            exclude: Set of tensor names to exclude
            copy: Whether to return copies of the data (True) or views (False)

        Returns:
            Dictionary mapping tensor names to numpy arrays

        Raises:
            SafetensorsIOError: If read fails
        """
        if names is None:
            names = self.get_tensor_names()
        
        if exclude:
            names = [n for n in names if n not in exclude]
        
        logger.info(f"Reading {len(names)} tensors from file")
        
        result = {}
        for name in names:
            try:
                result[name] = self.read_tensor(name, copy=copy)
            except Exception as e:
                logger.error(f"Failed to read tensor '{name}': {e}")
                # Clean up any partially read tensors
                result.clear()
                raise
        
        logger.info(f"Successfully read {len(result)} tensors")
        return result

    def __enter__(self) -> "SafetensorsReader":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self._close_mmap()

    def validate(self) -> bool:
        """Validate the Safetensors file integrity.

        This method performs comprehensive validation including:
        - Header format validation
        - Tensor offset validation
        - File size consistency checks
        - Data accessibility verification

        Returns:
            True if file is valid, False otherwise
        """
        try:
            # Get file size
            file_size = self.file_path.stat().st_size
            logger.debug(f"Validating file with size: {file_size} bytes")
            
            # Check minimum file size (8 bytes header size + header + some data)
            if file_size < 8 + self._header_size:
                logger.error("File too small to contain valid data")
                return False
            
            # Validate each tensor
            max_offset = 0
            for name in self.get_tensor_names():
                info = self._tensor_info[name]
                
                # Verify offsets are valid
                if info.data_offsets[0] >= info.data_offsets[1]:
                    logger.error(f"Invalid offsets for tensor '{name}': {info.data_offsets}")
                    return False
                
                # Check that tensor data fits in file
                tensor_end = self._data_offset + info.data_offsets[1]
                if tensor_end > file_size:
                    logger.error(
                        f"Tensor '{name}' extends beyond file size: "
                        f"{tensor_end} > {file_size}"
                    )
                    return False
                
                # Track maximum offset
                max_offset = max(max_offset, info.data_offsets[1])
                
                # Verify we can calculate the expected size
                expected_size = np.prod(info.shape) * info.dtype.to_numpy().itemsize
                actual_size = info.byte_size
                
                if expected_size != actual_size:
                    logger.error(
                        f"Size mismatch for tensor '{name}': "
                        f"expected {expected_size} bytes, found {actual_size} bytes"
                    )
                    return False
                
                # Try reading a small portion to verify accessibility
                try:
                    offset = self._data_offset + info.data_offsets[0]
                    
                    if self._mmap:
                        # Test mmap access
                        test_data = self._mmap[offset:offset + min(16, info.byte_size)]
                        if len(test_data) == 0:
                            logger.error(f"Cannot read data for tensor '{name}'")
                            return False
                    else:
                        # Test file access
                        with open(self.file_path, "rb") as f:
                            f.seek(offset)
                            test_bytes = f.read(min(16, info.byte_size))
                            if len(test_bytes) == 0:
                                logger.error(f"Cannot read data for tensor '{name}'")
                                return False
                except Exception as e:
                    logger.error(f"Failed to access tensor '{name}': {e}")
                    return False
            
            # Check that all tensor data is accounted for
            expected_data_size = self._data_offset + max_offset
            if expected_data_size > file_size:
                logger.error(
                    f"File size mismatch: expected at least {expected_data_size} bytes, "
                    f"found {file_size} bytes"
                )
                return False
            
            logger.info("File validation successful")
            return True
        
        except Exception as e:
            logger.error(f"Validation failed with error: {e}")
            return False
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get information about the Safetensors file.
        
        Returns:
            Dictionary containing file information including:
            - file_size: Total file size in bytes
            - header_size: Size of the header in bytes
            - num_tensors: Number of tensors in the file
            - total_parameters: Total number of parameters across all tensors
            - memory_usage: Estimated memory usage when loaded
            - tensors: List of tensor information
        """
        file_size = self.file_path.stat().st_size
        
        tensor_info = []
        total_params = 0
        memory_usage = 0
        
        for name, info in self._tensor_info.items():
            num_params = np.prod(info.shape)
            total_params += num_params
            memory_usage += info.byte_size
            
            tensor_info.append({
                "name": name,
                "shape": info.shape,
                "dtype": info.dtype.value,
                "parameters": num_params,
                "bytes": info.byte_size,
            })
        
        return {
            "file_path": str(self.file_path),
            "file_size": file_size,
            "header_size": self._header_size,
            "num_tensors": len(self._tensor_info),
            "total_parameters": total_params,
            "memory_usage": memory_usage,
            "metadata": self.metadata,
            "tensors": tensor_info,
        }
    
    def close(self) -> None:
        """Close the reader and release resources."""
        self._close_mmap()
        logger.debug(f"Closed SafetensorsReader for {self.file_path}")
    
    def __contains__(self, name: str) -> bool:
        """Check if a tensor exists in the file.
        
        Args:
            name: Tensor name to check
            
        Returns:
            True if tensor exists, False otherwise
        """
        return name in self._tensor_info
    
    def __len__(self) -> int:
        """Get the number of tensors in the file.
        
        Returns:
            Number of tensors
        """
        return len(self._tensor_info)
    
    def __getitem__(self, name: str) -> np.ndarray:
        """Get a tensor by name using subscript notation.
        
        Args:
            name: Tensor name
            
        Returns:
            NumPy array containing the tensor data
            
        Raises:
            KeyError: If tensor not found
        """
        return self.read_tensor(name)
    
    def keys(self) -> List[str]:
        """Get tensor names (dict-like interface).
        
        Returns:
            List of tensor names
        """
        return self.get_tensor_names()
    
    def values(self) -> List[np.ndarray]:
        """Get all tensors (dict-like interface).
        
        Returns:
            List of numpy arrays
        """
        return [self.read_tensor(name) for name in self.get_tensor_names()]
    
    def items(self) -> List[Tuple[str, np.ndarray]]:
        """Get name-tensor pairs (dict-like interface).
        
        Returns:
            List of (name, tensor) tuples
        """
        return [(name, self.read_tensor(name)) for name in self.get_tensor_names()]