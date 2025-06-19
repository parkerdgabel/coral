"""
Type definitions and enums for SafeTensors integration.

This module provides comprehensive type definitions, enums, and exceptions
used throughout the Coral SafeTensors system.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class DType(Enum):
    """Supported data types for SafeTensors format."""

    # Float types
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"

    # Integer types
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"

    # Unsigned integer types
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"

    # Boolean type
    BOOL = "bool"

    @classmethod
    def from_numpy(cls, np_dtype: np.dtype) -> "DType":
        """Convert numpy dtype to SafeTensors DType enum."""
        dtype_map = {
            np.float16: cls.FLOAT16,
            np.float32: cls.FLOAT32,
            np.float64: cls.FLOAT64,
            np.int8: cls.INT8,
            np.int16: cls.INT16,
            np.int32: cls.INT32,
            np.int64: cls.INT64,
            np.uint8: cls.UINT8,
            np.uint16: cls.UINT16,
            np.uint32: cls.UINT32,
            np.uint64: cls.UINT64,
            np.bool_: cls.BOOL,
        }

        # Handle numpy dtype object directly
        if isinstance(np_dtype, type):
            np_dtype = np.dtype(np_dtype)

        # Try direct lookup first
        if np_dtype.type in dtype_map:
            return dtype_map[np_dtype.type]

        # Fallback to string matching
        dtype_str = str(np_dtype)
        for np_type, safetensors_type in dtype_map.items():
            if dtype_str == str(np.dtype(np_type)):
                return safetensors_type

        raise ValueError(f"Unsupported numpy dtype: {np_dtype}")

    def to_numpy(self) -> np.dtype:
        """Convert SafeTensors DType to numpy dtype."""
        dtype_map = {
            self.FLOAT16: np.float16,
            self.FLOAT32: np.float32,
            self.FLOAT64: np.float64,
            self.INT8: np.int8,
            self.INT16: np.int16,
            self.INT32: np.int32,
            self.INT64: np.int64,
            self.UINT8: np.uint8,
            self.UINT16: np.uint16,
            self.UINT32: np.uint32,
            self.UINT64: np.uint64,
            self.BOOL: np.bool_,
        }
        return np.dtype(dtype_map[self])


@dataclass
class TensorInfo:
    """Information about a tensor in a SafeTensors file."""

    name: str
    dtype: DType
    shape: Tuple[int, ...]
    data_offsets: Tuple[int, int]  # (start, end) byte offsets in file

    @property
    def size(self) -> int:
        """Total number of elements in the tensor."""
        if not self.shape:
            return 1  # Scalar tensor
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        """Total number of bytes required for tensor data."""
        return self.size * self.dtype.to_numpy().itemsize


@dataclass
class TensorMetadata:
    """Extended metadata for a tensor beyond basic SafeTensors info."""

    name: str
    dtype: DType
    shape: Tuple[int, ...]

    # Coral-specific metadata
    layer_type: Optional[str] = None
    model_name: Optional[str] = None
    compression_info: Optional[Dict[str, Any]] = None
    creation_time: Optional[str] = None
    author: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        result = {
            "name": self.name,
            "dtype": self.dtype.value if self.dtype else None,
            "shape": list(self.shape),
        }

        # Add optional fields if present
        optional_fields = [
            "layer_type",
            "model_name",
            "compression_info",
            "creation_time",
            "author",
            "description",
            "tags",
        ]

        for field in optional_fields:
            value = getattr(self, field, None)
            if value is not None:
                result[field] = value

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TensorMetadata":
        """Create TensorMetadata from dictionary."""
        # Required fields
        name = data["name"]
        dtype = DType(data["dtype"])
        shape = tuple(data["shape"])

        # Optional fields
        kwargs = {}
        optional_fields = [
            "layer_type",
            "model_name",
            "compression_info",
            "creation_time",
            "author",
            "description",
            "tags",
        ]

        for field in optional_fields:
            if field in data:
                kwargs[field] = data[field]

        return cls(name=name, dtype=dtype, shape=shape, **kwargs)


class SafetensorsError(Exception):
    """Base exception for SafeTensors-related errors."""

    pass


class SafetensorsReadError(SafetensorsError):
    """Error occurred while reading SafeTensors file."""

    pass


class SafetensorsWriteError(SafetensorsError):
    """Error occurred while writing SafeTensors file."""

    pass


class SafetensorsFormatError(SafetensorsError):
    """SafeTensors file format is invalid or corrupted."""

    pass


class SafetensorsMetadataError(SafetensorsError):
    """Error in SafeTensors metadata processing."""

    pass


class SafetensorsConversionError(SafetensorsError):
    """Error during format conversion."""

    pass


# Type aliases for convenience
TensorDict = Dict[str, np.ndarray]
MetadataDict = Dict[str, Any]
OffsetInfo = Tuple[int, int]  # (start, end) byte offsets
ShapeType = Tuple[int, ...]
