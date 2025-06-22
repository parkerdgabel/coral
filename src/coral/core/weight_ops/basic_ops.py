"""Basic weight operations for computation graphs.

This module implements fundamental operations that serve as building blocks
for more complex weight transformations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from coral.core.weight_ops.base import (
    OpType,
    WeightOp,
    calculate_array_memory,
    validate_array,
    validate_compatible_shapes,
    validate_matmul_shapes,
    validate_shape,
)


class IdentityOp(WeightOp):
    """Identity operation that wraps raw numpy arrays.
    
    This operation provides backward compatibility by wrapping existing
    numpy arrays in the computation graph framework. It stores the array
    directly and returns it unchanged when evaluated.
    
    Attributes:
        data: The wrapped numpy array
    """
    
    def __init__(self, data: np.ndarray):
        """Initialize identity operation.
        
        Args:
            data: The numpy array to wrap
            
        Raises:
            TypeError: If data is not a numpy array
            ValueError: If data contains invalid values
        """
        super().__init__(OpType.IDENTITY, inputs=[])
        validate_array(data, "data")
        self.data = data.copy()  # Store a copy to ensure immutability
        self._shape = data.shape
        self._dtype = data.dtype
    
    def forward(self) -> np.ndarray:
        """Return the wrapped array.
        
        Returns:
            Copy of the wrapped array
        """
        return self.data.copy()
    
    def get_memory_usage(self) -> int:
        """Return memory usage of the stored array.
        
        Returns:
            Memory usage in bytes
        """
        return self.data.nbytes
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Return shape of the wrapped array.
        
        Returns:
            Array shape
        """
        return self._shape
    
    def get_output_dtype(self) -> np.dtype:
        """Return dtype of the wrapped array.
        
        Returns:
            Array dtype
        """
        return self._dtype
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize the identity operation.
        
        Returns:
            Serialized representation
        """
        return {
            "op_type": self.op_type.name,
            "data": self.data.tolist(),
            "shape": list(self._shape),
            "dtype": str(self._dtype),
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'IdentityOp':
        """Deserialize identity operation.
        
        Args:
            data: Serialized data
            
        Returns:
            Reconstructed IdentityOp
        """
        array_data = np.array(data["data"], dtype=data["dtype"])
        array_data = array_data.reshape(data["shape"])
        return cls(array_data)
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"IdentityOp(shape={self._shape}, dtype={self._dtype})"


class AddOp(WeightOp):
    """Addition operation that adds multiple tensors element-wise.
    
    All input tensors must have the same shape. The operation computes
    the element-wise sum of all inputs.
    """
    
    def __init__(self, inputs: List[WeightOp]):
        """Initialize addition operation.
        
        Args:
            inputs: List of operations to add
            
        Raises:
            ValueError: If fewer than 2 inputs or incompatible shapes
        """
        super().__init__(OpType.ADD, inputs)
        
        if len(inputs) < 2:
            raise ValueError("AddOp requires at least 2 inputs")
        
        # Validate all inputs have the same shape
        base_shape = inputs[0].get_output_shape()
        base_dtype = inputs[0].get_output_dtype()
        
        for i, inp in enumerate(inputs[1:], 1):
            validate_compatible_shapes(
                base_shape, 
                inp.get_output_shape(),
                f"AddOp input {i}"
            )
    
    def forward(self) -> np.ndarray:
        """Compute element-wise sum of inputs.
        
        Returns:
            Sum of all input tensors
        """
        result = self.inputs[0].forward()
        for inp in self.inputs[1:]:
            result = result + inp.forward()
        return result
    
    def get_memory_usage(self) -> int:
        """Return memory usage (minimal for add operation).
        
        Returns:
            Memory usage in bytes (0 as no parameters stored)
        """
        return 0  # No additional storage needed
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Return output shape (same as inputs).
        
        Returns:
            Output shape
        """
        return self.inputs[0].get_output_shape()
    
    def get_output_dtype(self) -> np.dtype:
        """Return output dtype.
        
        Returns:
            Output dtype (promoted if necessary)
        """
        # Use numpy's type promotion rules
        dtypes = [inp.get_output_dtype() for inp in self.inputs]
        return np.result_type(*dtypes)
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize the add operation.
        
        Returns:
            Serialized representation
        """
        return {
            "op_type": self.op_type.name,
            "num_inputs": len(self.inputs),
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'AddOp':
        """Deserialize add operation.
        
        Note: This requires the inputs to be reconstructed separately
        and passed to the constructor.
        
        Args:
            data: Serialized data
            
        Returns:
            Reconstructed AddOp (placeholder, needs input reconstruction)
        """
        # In practice, graph deserialization would handle input reconstruction
        raise NotImplementedError(
            "AddOp deserialization requires graph-level reconstruction"
        )
    
    def __repr__(self) -> str:
        """Return string representation."""
        shape = self.get_output_shape()
        return f"AddOp(num_inputs={len(self.inputs)}, shape={shape})"


class MatMulOp(WeightOp):
    """Matrix multiplication operation.
    
    Computes matrix multiplication between two tensors. Supports 2D
    matrices and higher-dimensional tensors with batch dimensions.
    """
    
    def __init__(self, left: WeightOp, right: WeightOp):
        """Initialize matrix multiplication.
        
        Args:
            left: Left operand
            right: Right operand
            
        Raises:
            ValueError: If shapes are incompatible for matmul
        """
        super().__init__(OpType.MATMUL, inputs=[left, right])
        
        # Validate shapes are compatible
        self._output_shape = validate_matmul_shapes(
            left.get_output_shape(),
            right.get_output_shape()
        )
    
    def forward(self) -> np.ndarray:
        """Compute matrix multiplication.
        
        Returns:
            Result of matrix multiplication
        """
        left = self.inputs[0].forward()
        right = self.inputs[1].forward()
        return np.matmul(left, right)
    
    def get_memory_usage(self) -> int:
        """Return memory usage (minimal for matmul).
        
        Returns:
            Memory usage in bytes
        """
        return 0  # No additional storage needed
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Return output shape.
        
        Returns:
            Output shape after matmul
        """
        return self._output_shape
    
    def get_output_dtype(self) -> np.dtype:
        """Return output dtype.
        
        Returns:
            Output dtype (promoted if necessary)
        """
        return np.result_type(
            self.inputs[0].get_output_dtype(),
            self.inputs[1].get_output_dtype()
        )
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize the matmul operation.
        
        Returns:
            Serialized representation
        """
        return {
            "op_type": self.op_type.name,
            "output_shape": list(self._output_shape),
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'MatMulOp':
        """Deserialize matmul operation.
        
        Args:
            data: Serialized data
            
        Returns:
            Reconstructed MatMulOp (placeholder)
        """
        raise NotImplementedError(
            "MatMulOp deserialization requires graph-level reconstruction"
        )
    
    def __repr__(self) -> str:
        """Return string representation."""
        left_shape = self.inputs[0].get_output_shape()
        right_shape = self.inputs[1].get_output_shape()
        return f"MatMulOp({left_shape} @ {right_shape} -> {self._output_shape})"


class ScaleOp(WeightOp):
    """Element-wise scaling operation.
    
    Multiplies a tensor by a scalar value element-wise.
    """
    
    def __init__(self, input_op: WeightOp, scale: float):
        """Initialize scaling operation.
        
        Args:
            input_op: Operation to scale
            scale: Scalar multiplier
            
        Raises:
            TypeError: If scale is not numeric
            ValueError: If scale is not finite
        """
        super().__init__(OpType.SCALE, inputs=[input_op])
        
        if not isinstance(scale, (int, float)):
            raise TypeError(f"Scale must be numeric, got {type(scale)}")
        
        if not np.isfinite(scale):
            raise ValueError(f"Scale must be finite, got {scale}")
        
        self.scale = float(scale)
    
    def forward(self) -> np.ndarray:
        """Compute scaled tensor.
        
        Returns:
            Input tensor multiplied by scale
        """
        return self.inputs[0].forward() * self.scale
    
    def get_memory_usage(self) -> int:
        """Return memory usage for storing scale parameter.
        
        Returns:
            Memory usage in bytes
        """
        return 8  # 8 bytes for float64 scale
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Return output shape (same as input).
        
        Returns:
            Output shape
        """
        return self.inputs[0].get_output_shape()
    
    def get_output_dtype(self) -> np.dtype:
        """Return output dtype.
        
        Returns:
            Output dtype (may be promoted to float)
        """
        input_dtype = self.inputs[0].get_output_dtype()
        # Scaling by float promotes integer types to float
        if np.issubdtype(input_dtype, np.integer):
            return np.dtype('float64')
        return input_dtype
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize the scale operation.
        
        Returns:
            Serialized representation
        """
        return {
            "op_type": self.op_type.name,
            "scale": self.scale,
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'ScaleOp':
        """Deserialize scale operation.
        
        Args:
            data: Serialized data
            
        Returns:
            Reconstructed ScaleOp (placeholder)
        """
        raise NotImplementedError(
            "ScaleOp deserialization requires graph-level reconstruction"
        )
    
    def __repr__(self) -> str:
        """Return string representation."""
        shape = self.get_output_shape()
        return f"ScaleOp(scale={self.scale}, shape={shape})"


class ReshapeOp(WeightOp):
    """Reshape operation that changes tensor dimensions.
    
    Changes the shape of a tensor while preserving the total number
    of elements. The operation is a view when possible (no data copy).
    """
    
    def __init__(self, input_op: WeightOp, new_shape: Union[Tuple[int, ...], List[int]]):
        """Initialize reshape operation.
        
        Args:
            input_op: Operation to reshape
            new_shape: Target shape (may contain -1 for auto-inference)
            
        Raises:
            ValueError: If new shape is incompatible with input size
        """
        super().__init__(OpType.RESHAPE, inputs=[input_op])
        
        # Validate and normalize the new shape
        new_shape = list(new_shape)
        input_shape = input_op.get_output_shape()
        input_size = int(np.prod(input_shape))
        
        # Handle -1 in shape (infer dimension)
        neg_one_count = new_shape.count(-1)
        if neg_one_count > 1:
            raise ValueError("Only one dimension can be -1")
        
        if neg_one_count == 1:
            # Calculate the inferred dimension
            neg_one_idx = new_shape.index(-1)
            known_size = 1
            for i, dim in enumerate(new_shape):
                if i != neg_one_idx:
                    if dim <= 0:
                        raise ValueError(f"Invalid dimension {dim} in shape")
                    known_size *= dim
            
            if input_size % known_size != 0:
                raise ValueError(
                    f"Cannot reshape array of size {input_size} into shape {new_shape}"
                )
            
            new_shape[neg_one_idx] = input_size // known_size
        
        # Validate final shape
        new_shape = validate_shape(new_shape, "new_shape")
        new_size = int(np.prod(new_shape))
        
        if new_size != input_size:
            raise ValueError(
                f"Cannot reshape array of size {input_size} into shape {new_shape} "
                f"(size {new_size})"
            )
        
        self.new_shape = new_shape
    
    def forward(self) -> np.ndarray:
        """Compute reshaped tensor.
        
        Returns:
            Reshaped tensor
        """
        return self.inputs[0].forward().reshape(self.new_shape)
    
    def get_memory_usage(self) -> int:
        """Return memory usage for storing shape.
        
        Returns:
            Memory usage in bytes
        """
        return len(self.new_shape) * 8  # 8 bytes per dimension
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Return output shape.
        
        Returns:
            New shape after reshape
        """
        return self.new_shape
    
    def get_output_dtype(self) -> np.dtype:
        """Return output dtype (same as input).
        
        Returns:
            Output dtype
        """
        return self.inputs[0].get_output_dtype()
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize the reshape operation.
        
        Returns:
            Serialized representation
        """
        return {
            "op_type": self.op_type.name,
            "new_shape": list(self.new_shape),
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'ReshapeOp':
        """Deserialize reshape operation.
        
        Args:
            data: Serialized data
            
        Returns:
            Reconstructed ReshapeOp (placeholder)
        """
        raise NotImplementedError(
            "ReshapeOp deserialization requires graph-level reconstruction"
        )
    
    def __repr__(self) -> str:
        """Return string representation."""
        old_shape = self.inputs[0].get_output_shape()
        return f"ReshapeOp({old_shape} -> {self.new_shape})"