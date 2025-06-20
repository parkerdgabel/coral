"""Weight quantization for compression"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from coral.core.weight_tensor import WeightMetadata, WeightTensor

logger = logging.getLogger(__name__)


class Quantizer:
    """
    Quantization methods for weight compression.

    Supports:
    - Uniform quantization (8-bit, 4-bit, 2-bit)
    - Dynamic quantization with scale and zero-point
    - Symmetric and asymmetric quantization
    """

    @staticmethod
    def quantize_uniform(
        weight: WeightTensor, bits: int = 8, symmetric: bool = True
    ) -> Tuple[WeightTensor, Dict[str, Any]]:
        """
        Perform uniform quantization on weights.

        Args:
            weight: Weight tensor to quantize
            bits: Number of bits for quantization (2, 4, or 8)
            symmetric: Use symmetric quantization

        Returns:
            Tuple of (quantized_weight, quantization_params)
        """
        if bits not in [2, 4, 8]:
            raise ValueError(f"Unsupported bit width: {bits}")

        data = weight.data

        # Handle empty arrays
        if data.size == 0:
            quantized: np.ndarray = data.astype(np.int8 if bits == 8 else np.int16)
            return WeightTensor(
                data=quantized,
                metadata=WeightMetadata(
                    name=weight.metadata.name + "_quantized",
                    shape=weight.shape,
                    dtype=quantized.dtype,
                    compression_info={"scale": 1.0, "zero_point": 0},
                ),
            ), {"scale": 1.0, "zero_point": 0, "bits": bits, "symmetric": symmetric}

        if symmetric:
            # Symmetric quantization
            max_val: float = float(np.max(np.abs(data)))
            scale = max_val / (2 ** (bits - 1) - 1) if max_val > 0 else 1.0
            zero_point = 0

            # Quantize
            quantized = np.round(data / scale).astype(
                np.int8 if bits == 8 else np.int16
            )
            quantized = np.clip(quantized, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        else:
            # Asymmetric quantization
            min_val: float = float(np.min(data))
            max_val_asym: float = float(np.max(data))

            # Calculate scale and zero point
            qmin = 0
            qmax = 2**bits - 1
            scale = (max_val_asym - min_val) / (qmax - qmin)
            zero_point = qmin - min_val / scale

            # Quantize
            quantized = np.round((data - min_val) / scale + qmin)
            quantized = np.clip(quantized, qmin, qmax).astype(
                np.uint8 if bits == 8 else np.uint16
            )

        # Create quantized weight tensor
        quant_metadata = WeightMetadata(
            name=weight.metadata.name + "_quantized",
            shape=weight.shape,
            dtype=quantized.dtype,
            layer_type=weight.metadata.layer_type,
            model_name=weight.metadata.model_name,
            compression_info={
                "method": "uniform_quantization",
                "bits": bits,
                "symmetric": symmetric,
                "scale": float(scale),
                "zero_point": float(zero_point),
                "original_dtype": str(weight.dtype),
            },
        )

        quantized_weight = WeightTensor(data=quantized, metadata=quant_metadata)

        quantization_params = {
            "scale": scale,
            "zero_point": zero_point,
            "bits": bits,
            "symmetric": symmetric,
        }

        logger.debug(
            f"Quantized {weight.metadata.name} to {bits} bits "
            f"(compression ratio: {weight.nbytes / quantized_weight.nbytes:.2f}x)"
        )

        return quantized_weight, quantization_params

    @staticmethod
    def dequantize(
        quantized_weight: WeightTensor,
        quantization_params: Optional[Dict[str, Any]] = None,
    ) -> WeightTensor:
        """
        Dequantize a quantized weight tensor.

        Args:
            quantized_weight: Quantized weight tensor
            quantization_params: Quantization parameters (if not in metadata)

        Returns:
            Dequantized weight tensor
        """
        # Get quantization info
        if quantization_params is None:
            compression_info = quantized_weight.metadata.compression_info
            if "scale" not in compression_info:
                raise ValueError("No quantization parameters found")
            quantization_params = compression_info

        scale = quantization_params["scale"]
        zero_point = quantization_params["zero_point"]
        symmetric = quantization_params.get("symmetric", True)

        # Dequantize
        if symmetric:
            dequantized = quantized_weight.data.astype(np.float32) * scale
        else:
            dequantized = (
                quantized_weight.data.astype(np.float32) - zero_point
            ) * scale

        # Create dequantized weight tensor
        dequant_metadata = WeightMetadata(
            name=quantized_weight.metadata.name.replace("_quantized", "_dequantized"),
            shape=quantized_weight.shape,
            dtype=np.float32,
            layer_type=quantized_weight.metadata.layer_type,
            model_name=quantized_weight.metadata.model_name,
            compression_info={},
        )

        return WeightTensor(data=dequantized, metadata=dequant_metadata)

    @staticmethod
    def quantize_per_channel(
        weight: WeightTensor, bits: int = 8, axis: int = 0
    ) -> Tuple[WeightTensor, Dict[str, Any]]:
        """
        Perform per-channel quantization.

        Args:
            weight: Weight tensor to quantize
            bits: Number of bits for quantization
            axis: Axis along which to compute per-channel scales

        Returns:
            Tuple of (quantized_weight, quantization_params)
        """
        data = weight.data

        # Compute per-channel min/max
        axes_to_reduce = tuple(i for i in range(data.ndim) if i != axis)
        channel_mins = np.min(data, axis=axes_to_reduce, keepdims=True)
        channel_maxs = np.max(data, axis=axes_to_reduce, keepdims=True)

        # Compute per-channel scale and zero point
        qmin = 0
        qmax = 2**bits - 1
        scales = (channel_maxs - channel_mins) / (qmax - qmin)
        zero_points = qmin - channel_mins / scales

        # Quantize
        quantized = np.round((data - channel_mins) / scales + qmin)
        quantized = np.clip(quantized, qmin, qmax).astype(
            np.uint8 if bits == 8 else np.uint16
        )

        # Create quantized weight tensor
        quant_metadata = WeightMetadata(
            name=weight.metadata.name + "_quantized_per_channel",
            shape=weight.shape,
            dtype=quantized.dtype,
            layer_type=weight.metadata.layer_type,
            model_name=weight.metadata.model_name,
            compression_info={
                "method": "per_channel_quantization",
                "bits": bits,
                "axis": axis,
                "original_dtype": str(weight.dtype),
            },
        )

        quantized_weight = WeightTensor(data=quantized, metadata=quant_metadata)

        quantization_params = {
            "scales": scales.squeeze(),
            "zero_points": zero_points.squeeze(),
            "bits": bits,
            "axis": axis,
        }

        return quantized_weight, quantization_params

    @staticmethod
    def estimate_quantization_error(
        weight: WeightTensor, bits: int = 8, symmetric: bool = True
    ) -> float:
        """
        Estimate quantization error without actually quantizing.

        Returns:
            Mean squared error of quantization
        """
        # Quantize and dequantize
        quantized, params = Quantizer.quantize_uniform(weight, bits, symmetric)
        dequantized = Quantizer.dequantize(quantized, params)

        # Compute MSE
        mse = np.mean((weight.data - dequantized.data) ** 2)
        return float(mse)
