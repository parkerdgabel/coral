"""Basic operations for weight computation graphs."""

import copy
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from .base import OpType, WeightOp, validate_array, validate_compatible_shapes, validate_matmul_shapes, validate_shape


class IdentityOp(WeightOp):
    """Identity operation that returns stored array unchanged.
    
    This is the fundamental operation for backward compatibility,
    wrapping raw numpy arrays in the computation graph framework.
    """
    
    def __init__(self, data: np.ndarray):
        """Initialize with array data.
        
        Args:
            data: Numpy array to store
            
        Raises:
            TypeError: If data is not a numpy array
            ValueError: If data is empty or contains non-finite values
        """
        validate_array(data)
        # Store a copy to ensure immutability
        self._data = data.copy()
        self._shape = data.shape
        self._dtype = data.dtype
        self.op_type = OpType.IDENTITY
    
    def forward(self) -> np.ndarray:
        """Return the stored array."""
        # Return a copy to prevent external modification
        return self._data.copy()
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Return output shape without computing."""
        return self._shape
    
    def get_output_dtype(self) -> np.dtype:
        """Return output dtype without computing."""
        return self._dtype
    
    def get_memory_usage(self) -> int:
        """Return memory usage in bytes."""
        return self._data.nbytes
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize operation for storage."""
        return {
            "op_type": "IDENTITY",
            "data": self._data.tolist(),
            "shape": list(self._shape),
            "dtype": str(self._dtype)
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "IdentityOp":
        """Deserialize from stored representation."""
        arr = np.array(data["data"], dtype=data["dtype"])
        return cls(arr)
    
    def __repr__(self) -> str:
        """String representation of operation."""
        return f"IdentityOp(shape={self._shape}, dtype={self._dtype})"


class AddOp(WeightOp):
    """Element-wise addition of multiple tensors."""
    
    def __init__(self, inputs: List[WeightOp]):
        """Initialize with input operations.
        
        Args:
            inputs: List of WeightOp instances to add
            
        Raises:
            ValueError: If inputs have incompatible shapes or less than 2 inputs
        """
        if len(inputs) < 2:
            raise ValueError("AddOp requires at least 2 inputs")
        
        # Get shapes without evaluation
        shapes = [op.get_output_shape() for op in inputs]
        base_shape = shapes[0]
        
        # Validate all shapes are compatible (broadcastable)
        for i, shape in enumerate(shapes[1:], 1):
            try:
                np.broadcast_shapes(base_shape, shape)
            except ValueError:
                raise ValueError(
                    f"Incompatible shapes: Input {i} with shape {shape} is not broadcastable with shape {base_shape}"
                )
        
        self._inputs = inputs
        # Output shape is the broadcast shape
        self._output_shape = np.broadcast_shapes(*shapes)
        # Output dtype is the common dtype
        self._output_dtype = np.result_type(*[op.get_output_dtype() for op in inputs])
        self.op_type = OpType.ADD
    
    def forward(self) -> np.ndarray:
        """Compute element-wise sum of inputs."""
        arrays = [op.forward() for op in self._inputs]
        result = arrays[0].copy()
        for arr in arrays[1:]:
            result = result + arr  # Uses numpy broadcasting
        return result
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Return output shape without computing."""
        return self._output_shape
    
    def get_output_dtype(self) -> np.dtype:
        """Return output dtype without computing."""
        return self._output_dtype
    
    def get_memory_usage(self) -> int:
        """Return memory usage of this operation (not inputs)."""
        # AddOp itself stores no data, only references
        return 0
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize operation for storage."""
        return {
            "type": OpType.ADD.value,
            "inputs": [op.serialize() for op in self._inputs]
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "AddOp":
        """Deserialize from stored representation."""
        # This would need a registry of operation types to deserialize inputs
        # For now, we'll raise NotImplementedError
        raise NotImplementedError("Deserialization requires operation registry")
    
    def __repr__(self) -> str:
        """String representation of operation."""
        return f"AddOp(num_inputs={len(self._inputs)}, shape={self._output_shape})"


class MatMulOp(WeightOp):
    """Matrix multiplication operation."""
    
    def __init__(self, left: WeightOp, right: WeightOp):
        """Initialize with two input operations.
        
        Args:
            left: Left operand
            right: Right operand
            
        Raises:
            ValueError: If shapes are incompatible for matrix multiplication
        """
        left_shape = left.get_output_shape()
        right_shape = right.get_output_shape()
        
        # Validate shapes are compatible
        validate_matmul_shapes(left_shape, right_shape)
        
        self._left = left
        self._right = right
        
        # Compute output shape
        if len(left_shape) == 1 and len(right_shape) == 1:
            # Dot product
            self._output_shape = ()
        elif len(left_shape) == 1:
            # Vector-matrix
            self._output_shape = right_shape[1:]
        elif len(right_shape) == 1:
            # Matrix-vector
            self._output_shape = left_shape[:-1]
        else:
            # General matrix multiplication
            # For shapes (..., i, k) @ (..., k, j) -> (..., i, j)
            self._output_shape = left_shape[:-2] + (left_shape[-2],) + (right_shape[-1],)
        
        self._output_dtype = np.result_type(left.get_output_dtype(), right.get_output_dtype())
        self.op_type = OpType.MATMUL
    
    def forward(self) -> np.ndarray:
        """Compute matrix multiplication."""
        left_data = self._left.forward()
        right_data = self._right.forward()
        return np.matmul(left_data, right_data)
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Return output shape without computing."""
        return self._output_shape
    
    def get_output_dtype(self) -> np.dtype:
        """Return output dtype without computing."""
        return self._output_dtype
    
    def get_memory_usage(self) -> int:
        """Return memory usage of this operation (not inputs)."""
        # MatMulOp itself stores no data, only references
        return 0
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize operation for storage."""
        return {
            "type": OpType.MATMUL.value,
            "left": self._left.serialize(),
            "right": self._right.serialize()
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "MatMulOp":
        """Deserialize from stored representation."""
        raise NotImplementedError("Deserialization requires operation registry")
    
    def __repr__(self) -> str:
        """String representation of operation."""
        left_shape = self._left.get_output_shape()
        right_shape = self._right.get_output_shape()
        return f"MatMulOp({left_shape} @ {right_shape} -> {self._output_shape})"


class ScaleOp(WeightOp):
    """Element-wise scaling by a scalar value."""
    
    def __init__(self, input_op: WeightOp, scale: Union[float, int]):
        """Initialize with input operation and scale factor.
        
        Args:
            input_op: Input operation to scale
            scale: Scalar scaling factor
            
        Raises:
            TypeError: If scale is not numeric
            ValueError: If scale is not finite
        """
        if not isinstance(scale, (int, float)):
            raise TypeError(f"Scale must be numeric, got {type(scale)}")
        if not np.isfinite(scale):
            raise ValueError(f"Scale must be finite, got {scale}")
        
        self._input = input_op
        self._scale = float(scale)
        self._output_shape = input_op.get_output_shape()
        self._output_dtype = input_op.get_output_dtype()
        self.op_type = OpType.SCALE
    
    def forward(self) -> np.ndarray:
        """Compute scaled array."""
        return self._input.forward() * self._scale
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Return output shape without computing."""
        return self._output_shape
    
    def get_output_dtype(self) -> np.dtype:
        """Return output dtype without computing."""
        return self._output_dtype
    
    def get_memory_usage(self) -> int:
        """Return memory usage of this operation (not inputs)."""
        # ScaleOp stores only a scalar value
        return 8  # Size of a float64
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize operation for storage."""
        return {
            "type": OpType.SCALE.value,
            "input": self._input.serialize(),
            "scale": self._scale
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "ScaleOp":
        """Deserialize from stored representation."""
        raise NotImplementedError("Deserialization requires operation registry")


class ReshapeOp(WeightOp):
    """Reshape tensor to new shape."""
    
    def __init__(self, input_op: WeightOp, shape: Union[Tuple[int, ...], List[int]]):
        """Initialize with input operation and target shape.
        
        Args:
            input_op: Input operation to reshape
            shape: Target shape (can include -1 for auto-inference)
            
        Raises:
            ValueError: If shape is invalid or incompatible with input size
        """
        self._input = input_op
        input_shape = input_op.get_output_shape()
        
        # Validate and normalize shape
        shape = validate_shape(shape)
        
        # Calculate total size
        input_size = np.prod(input_shape)
        
        # Handle -1 in shape (auto-infer dimension)
        if -1 in shape:
            if shape.count(-1) > 1:
                raise ValueError("Can only have one -1 dimension in reshape")
            
            # Calculate the inferred dimension
            known_size = 1
            for dim in shape:
                if dim != -1:
                    known_size *= dim
            
            if input_size % known_size != 0:
                raise ValueError(
                    f"Cannot reshape array of size {input_size} into shape {shape}"
                )
            
            inferred_dim = input_size // known_size
            shape = tuple(inferred_dim if dim == -1 else dim for dim in shape)
        
        # Validate final shape
        output_size = np.prod(shape)
        if output_size != input_size:
            raise ValueError(
                f"Cannot reshape array of size {input_size} into shape {shape} "
                f"(size {output_size})"
            )
        
        self._output_shape = shape
        self._output_dtype = input_op.get_output_dtype()
        self.op_type = OpType.RESHAPE
    
    def forward(self) -> np.ndarray:
        """Compute reshaped array."""
        return self._input.forward().reshape(self._output_shape)
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Return output shape without computing."""
        return self._output_shape
    
    def get_output_dtype(self) -> np.dtype:
        """Return output dtype without computing."""
        return self._output_dtype
    
    def get_memory_usage(self) -> int:
        """Return memory usage of this operation (not inputs)."""
        # ReshapeOp stores only shape information
        return len(self._output_shape) * 8  # Size of shape tuple
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize operation for storage."""
        return {
            "type": OpType.RESHAPE.value,
            "input": self._input.serialize(),
            "shape": self._output_shape
        }
    
    @classmethod 
    def deserialize(cls, data: Dict[str, Any]) -> "ReshapeOp":
        """Deserialize from stored representation."""
        raise NotImplementedError("Deserialization requires operation registry")