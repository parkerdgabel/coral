"""Type definitions for Safetensors integration."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Type aliases
TensorDict = Dict[str, np.ndarray]
SafetensorDict = Dict[str, Union[np.ndarray, "torch.Tensor"]]  # type: ignore
MetadataDict = Dict[str, Any]
ShapeType = Tuple[int, ...]


class DType(Enum):
    """Supported data types in Safetensors format."""

    FLOAT32 = "F32"
    FLOAT16 = "F16"
    BFLOAT16 = "BF16"
    INT32 = "I32"
    INT16 = "I16"
    INT8 = "I8"
    UINT8 = "U8"
    BOOL = "BOOL"
    FLOAT64 = "F64"
    INT64 = "I64"

    @classmethod
    def from_numpy(cls, dtype: np.dtype) -> "DType":
        """Convert numpy dtype to Safetensors DType.

        Args:
            dtype: NumPy data type

        Returns:
            Corresponding Safetensors DType

        Raises:
            ValueError: If dtype is not supported
        """
        mapping = {
            np.float32: cls.FLOAT32,
            np.float16: cls.FLOAT16,
            np.float64: cls.FLOAT64,
            np.int32: cls.INT32,
            np.int16: cls.INT16,
            np.int8: cls.INT8,
            np.uint8: cls.UINT8,
            np.int64: cls.INT64,
            np.bool_: cls.BOOL,
        }
        
        if dtype.type not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        return mapping[dtype.type]

    def to_numpy(self) -> np.dtype:
        """Convert Safetensors DType to numpy dtype.

        Returns:
            Corresponding NumPy data type
        """
        mapping = {
            self.FLOAT32: np.float32,
            self.FLOAT16: np.float16,
            self.FLOAT64: np.float64,
            self.INT32: np.int32,
            self.INT16: np.int16,
            self.INT8: np.int8,
            self.UINT8: np.uint8,
            self.INT64: np.int64,
            self.BOOL: np.bool_,
        }
        
        if self == self.BFLOAT16:
            # bfloat16 is not natively supported by numpy
            # Use float32 as fallback
            return np.float32
        
        return mapping[self]


@dataclass
class TensorInfo:
    """Information about a tensor in Safetensors format."""

    name: str
    shape: ShapeType
    dtype: DType
    data_offsets: Tuple[int, int]  # (start, end) byte offsets in file

    @property
    def byte_size(self) -> int:
        """Calculate the byte size of the tensor.

        Returns:
            Size in bytes
        """
        return self.data_offsets[1] - self.data_offsets[0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for Safetensors header.

        Returns:
            Dictionary representation
        """
        return {
            "shape": list(self.shape),
            "dtype": self.dtype.value,
            "data_offsets": list(self.data_offsets),
        }


@dataclass
class TensorMetadata:
    """Metadata associated with a tensor."""

    shape: ShapeType
    dtype: str
    requires_grad: Optional[bool] = None
    device: Optional[str] = None
    is_parameter: Optional[bool] = None
    additional_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dictionary representation
        """
        result = {
            "shape": list(self.shape),
            "dtype": self.dtype,
        }
        
        if self.requires_grad is not None:
            result["requires_grad"] = self.requires_grad
        if self.device is not None:
            result["device"] = self.device
        if self.is_parameter is not None:
            result["is_parameter"] = self.is_parameter
        if self.additional_metadata:
            result.update(self.additional_metadata)
        
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TensorMetadata":
        """Create from dictionary.

        Args:
            data: Dictionary with metadata

        Returns:
            TensorMetadata instance
        """
        # Extract known fields
        shape = tuple(data["shape"])
        dtype = data["dtype"]
        requires_grad = data.get("requires_grad")
        device = data.get("device")
        is_parameter = data.get("is_parameter")
        
        # Collect additional metadata
        known_fields = {"shape", "dtype", "requires_grad", "device", "is_parameter"}
        additional_metadata = {k: v for k, v in data.items() if k not in known_fields}
        
        return cls(
            shape=shape,
            dtype=dtype,
            requires_grad=requires_grad,
            device=device,
            is_parameter=is_parameter,
            additional_metadata=additional_metadata if additional_metadata else None,
        )


class SafetensorsError(Exception):
    """Base exception for Safetensors operations."""

    pass


class SafetensorsFormatError(SafetensorsError):
    """Raised when Safetensors file format is invalid."""

    pass


class SafetensorsIOError(SafetensorsError):
    """Raised when I/O operations fail."""

    pass