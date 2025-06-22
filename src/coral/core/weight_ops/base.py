"""Base classes and utilities for weight operations.

This module defines the abstract base class for all weight operations
and common utilities used across the weight_ops module.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class OpType(Enum):
    """Enumeration of operation types for type checking and serialization."""
    
    # Basic operations
    IDENTITY = auto()
    ADD = auto()
    MATMUL = auto()
    SCALE = auto()
    RESHAPE = auto()
    SLICE = auto()
    CONCAT = auto()
    
    # Compression operations (for future use)
    SVD = auto()
    TUCKER = auto()
    SPARSE = auto()
    QUANTIZE = auto()
    PQ = auto()
    DELTA = auto()
    
    # Advanced operations (for future use)
    DICTIONARY = auto()
    WAVELET = auto()
    NEURAL = auto()
    GENERATOR = auto()


class WeightOp(ABC):
    """Abstract base class for weight operations in computation graph.
    
    All weight operations must inherit from this class and implement
    the required abstract methods. Operations should be immutable once
    created to ensure graph consistency.
    
    Attributes:
        op_type: The type of this operation (must be set by subclasses)
    """
    
    @abstractmethod
    def forward(self) -> np.ndarray:
        """Compute and return the weight tensor.
        
        This method should perform the actual computation defined by this
        operation. It may recursively call forward() on input operations.
        
        Returns:
            The computed weight tensor as a numpy array
        """
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> int:
        """Return memory usage in bytes for this operation.
        
        This should include the memory needed to store any parameters
        of this operation, but not the memory of computed outputs
        (which may be cached elsewhere).
        
        Returns:
            Memory usage in bytes
        """
        pass
    
    @abstractmethod
    def get_output_shape(self) -> Tuple[int, ...]:
        """Return the shape of the output tensor without computing it.
        
        This allows for shape inference without materializing tensors.
        
        Returns:
            Tuple representing the output shape
        """
        pass
    
    @abstractmethod
    def get_output_dtype(self) -> np.dtype:
        """Return the dtype of the output tensor without computing it.
        
        Returns:
            The numpy dtype of the output
        """
        pass
    
    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """Serialize operation for storage.
        
        The serialized representation should contain all information
        needed to reconstruct this operation, including any parameters
        and references to input operations.
        
        Returns:
            Dictionary containing serialized operation data
        """
        pass
    
    @classmethod
    @abstractmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'WeightOp':
        """Reconstruct operation from serialized data.
        
        Args:
            data: Dictionary containing serialized operation data
            
        Returns:
            Reconstructed WeightOp instance
        """
        pass
    
    def __repr__(self) -> str:
        """Return string representation of operation."""
        return f"{self.__class__.__name__}(op_type={self.op_type.name})"


# Input validation utilities

def validate_array(array: np.ndarray, name: str = "array") -> None:
    """Validate that input is a proper numpy array.
    
    Args:
        array: The array to validate
        name: Name of the array for error messages
        
    Raises:
        TypeError: If array is not a numpy array
        ValueError: If array has invalid properties
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array)}")
    
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values (NaN or Inf)")


def validate_shape(shape: Union[Tuple[int, ...], List[int]], name: str = "shape", 
                  allow_minus_one: bool = False) -> Tuple[int, ...]:
    """Validate and normalize a shape specification.
    
    Args:
        shape: The shape to validate (tuple or list of ints)
        name: Name for error messages
        allow_minus_one: Whether to allow -1 for automatic dimension inference
        
    Returns:
        Normalized shape as a tuple
        
    Raises:
        TypeError: If shape is not a tuple or list
        ValueError: If shape contains invalid values
    """
    if not isinstance(shape, (tuple, list)):
        raise TypeError(f"{name} must be a tuple or list, got {type(shape)}")
    
    shape = tuple(shape)
    
    if not shape:
        raise ValueError(f"{name} cannot be empty")
    
    for i, dim in enumerate(shape):
        if not isinstance(dim, int):
            raise TypeError(f"{name}[{i}] must be an integer, got {type(dim)}")
        if allow_minus_one and dim == -1:
            continue  # -1 is allowed for reshape
        if dim <= 0:
            raise ValueError(f"{name}[{i}] must be positive, got {dim}")
    
    return shape


def validate_compatible_shapes(shape1: Tuple[int, ...], shape2: Tuple[int, ...], 
                             op_name: str) -> None:
    """Validate that two shapes are compatible for an operation.
    
    Args:
        shape1: First shape
        shape2: Second shape
        op_name: Name of the operation for error messages
        
    Raises:
        ValueError: If shapes are not compatible
    """
    if shape1 != shape2:
        raise ValueError(
            f"Incompatible shapes for {op_name}: {shape1} vs {shape2}"
        )


def validate_matmul_shapes(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> Tuple[int, ...]:
    """Validate shapes for matrix multiplication and return output shape.
    
    Args:
        shape1: Shape of first matrix
        shape2: Shape of second matrix
        
    Returns:
        Output shape after matrix multiplication
        
    Raises:
        ValueError: If shapes are not compatible for matmul
    """
    if len(shape1) < 2 or len(shape2) < 2:
        raise ValueError(
            f"Matrices must have at least 2 dimensions for matmul, "
            f"got shapes {shape1} and {shape2}"
        )
    
    # Check inner dimensions match
    if shape1[-1] != shape2[-2]:
        raise ValueError(
            f"Incompatible shapes for matmul: {shape1} @ {shape2}"
        )
    
    # Calculate output shape
    if len(shape1) == 2 and len(shape2) == 2:
        # Simple 2D case
        return (shape1[0], shape2[1])
    else:
        # Broadcasting for batch dimensions
        batch1 = shape1[:-2]
        batch2 = shape2[:-2]
        
        # Numpy-style broadcasting
        if batch1 != batch2:
            # For simplicity, we require exact batch dimension match
            # Full numpy broadcasting could be added later
            raise ValueError(
                f"Batch dimensions must match for matmul: {batch1} vs {batch2}"
            )
        
        return batch1 + (shape1[-2], shape2[-1])


def calculate_array_memory(shape: Tuple[int, ...], dtype: np.dtype) -> int:
    """Calculate memory usage for an array of given shape and dtype.
    
    Args:
        shape: Shape of the array
        dtype: Data type of the array
        
    Returns:
        Memory usage in bytes
    """
    num_elements = int(np.prod(shape))
    return num_elements * dtype.itemsize


# Operation registry for deserialization
OPERATION_REGISTRY: Dict[str, type] = {}


def register_operation(op_type: str, op_class: type) -> None:
    """Register an operation class for deserialization.
    
    Args:
        op_type: Operation type name (e.g., "IDENTITY", "ADD")
        op_class: The operation class
    """
    OPERATION_REGISTRY[op_type] = op_class


def deserialize_op(data: Dict[str, Any]) -> 'WeightOp':
    """Deserialize any weight operation from stored data.
    
    Args:
        data: Dictionary containing serialized operation data
        
    Returns:
        Reconstructed WeightOp instance
        
    Raises:
        ValueError: If operation type is unknown
    """
    op_type = data.get("op_type")
    if not op_type:
        raise ValueError("Missing 'op_type' in serialized data")
    
    if op_type not in OPERATION_REGISTRY:
        raise ValueError(f"Unknown operation type: {op_type}")
    
    op_class = OPERATION_REGISTRY[op_type]
    return op_class.deserialize(data)