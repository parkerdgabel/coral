"""Delta encoding for similar weights to enable efficient deduplication.

This module provides multiple delta encoding strategies for storing differences
between similar neural network weights. Strategies vary in compression ratio
and reconstruction fidelity:

LOSSLESS strategies (perfect reconstruction):
- FLOAT32_RAW: No compression, stores raw float32 differences
- COMPRESSED: zlib compression of float32 differences

LOSSY strategies (approximate reconstruction):
- INT8_QUANTIZED: ~75% compression, introduces quantization error
- INT16_QUANTIZED: ~50% compression, smaller quantization error
- SPARSE: Discards differences below threshold (default 1e-6)

For archival or when exact reconstruction is required, use FLOAT32_RAW or
COMPRESSED. For training checkpoints where small errors are acceptable,
quantized strategies provide better compression.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core.weight_tensor import WeightTensor

logger = logging.getLogger(__name__)

# ============================================================================
# Constants for delta encoding
# ============================================================================

# Approximate overhead in bytes for delta object metadata (delta type, shape,
# dtype, reference hash, etc.)
DELTA_OBJECT_OVERHEAD_BYTES = 200

# Approximate overhead in bytes for serialized metadata (JSON encoding overhead)
DELTA_METADATA_OVERHEAD_BYTES = 200

# Default sparse encoding threshold - values with absolute difference below
# this are considered zero. NOTE: This makes SPARSE encoding technically lossy.
DEFAULT_SPARSE_THRESHOLD = 1e-6

# Default compression level for zlib (1-9, higher = better compression but slower)
DEFAULT_COMPRESSION_LEVEL = 6

# Default minimum weight size in bytes to consider delta encoding
# (smaller weights have too much overhead from metadata)
DEFAULT_MIN_WEIGHT_SIZE_BYTES = 512

# Default similarity threshold for deduplication
DEFAULT_SIMILARITY_THRESHOLD = 0.98

# Set of lossless delta types for easy checking
LOSSLESS_DELTA_TYPES = frozenset(["float32_raw", "compressed"])


class DeltaType(Enum):
    """Types of delta encoding strategies.

    Lossless (perfect reconstruction):
        FLOAT32_RAW: Raw float32 differences, no compression
        COMPRESSED: zlib-compressed float32 differences

    Lossy (approximate reconstruction):
        INT8_QUANTIZED: 8-bit quantization with scale/offset (~75% smaller)
        INT16_QUANTIZED: 16-bit quantization with scale/offset (~50% smaller)
        SPARSE: Only stores non-zero differences (discards values < threshold)
    """

    # Lossless strategies
    FLOAT32_RAW = "float32_raw"
    COMPRESSED = "compressed"

    # Lossy strategies (clearly marked)
    INT8_QUANTIZED = "int8_quantized"  # LOSSY: quantization error
    INT16_QUANTIZED = "int16_quantized"  # LOSSY: quantization error
    SPARSE = "sparse"  # LOSSY: discards small differences

    @property
    def is_lossless(self) -> bool:
        """Return True if this encoding strategy is lossless."""
        return self.value in LOSSLESS_DELTA_TYPES


class DeltaReconstructionError(Exception):
    """Raised when delta reconstruction fails due to hash mismatch in strict mode."""

    pass


@dataclass
class DeltaConfig:
    """Configuration for delta encoding."""

    delta_type: DeltaType = DeltaType.FLOAT32_RAW
    sparse_threshold: float = DEFAULT_SPARSE_THRESHOLD
    quantization_bits: int = 8  # Bits for quantized encoding (8 or 16)
    compression_level: int = DEFAULT_COMPRESSION_LEVEL
    max_delta_ratio: float = 1.0  # Allow delta up to 100% of original size
    min_compression_ratio: float = 0.0  # Minimum compression ratio (0% = always store)
    min_weight_size: int = DEFAULT_MIN_WEIGHT_SIZE_BYTES
    strict_reconstruction: bool = False  # If True, raise error on hash mismatch

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "delta_type": self.delta_type.value,
            "sparse_threshold": self.sparse_threshold,
            "quantization_bits": self.quantization_bits,
            "compression_level": self.compression_level,
            "max_delta_ratio": self.max_delta_ratio,
            "min_compression_ratio": self.min_compression_ratio,
            "min_weight_size": self.min_weight_size,
            "strict_reconstruction": self.strict_reconstruction,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeltaConfig":
        """Create from dictionary."""
        data = data.copy()
        data["delta_type"] = DeltaType(data["delta_type"])
        return cls(**data)


@dataclass
class Delta:
    """Represents a delta encoding of weight differences."""

    delta_type: DeltaType
    data: np.ndarray
    metadata: Dict[str, Any]
    original_shape: Tuple[int, ...]
    original_dtype: np.dtype
    reference_hash: str
    compression_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "delta_type": self.delta_type.value,
            "data_bytes": self.data.tobytes(),
            "data_dtype": str(self.data.dtype),
            "data_shape": self.data.shape,
            "metadata": self.metadata,
            "original_shape": self.original_shape,
            "original_dtype": str(self.original_dtype),
            "reference_hash": self.reference_hash,
            "compression_ratio": self.compression_ratio,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Delta":
        """Create from dictionary."""
        # Reconstruct data array
        data_bytes = data["data_bytes"]
        data_dtype = np.dtype(data["data_dtype"])
        data_shape = tuple(data["data_shape"])

        delta_data = np.frombuffer(data_bytes, dtype=data_dtype).reshape(data_shape)

        return cls(
            delta_type=DeltaType(data["delta_type"]),
            data=delta_data,
            metadata=data["metadata"],
            original_shape=tuple(data["original_shape"]),
            original_dtype=np.dtype(data["original_dtype"]),
            reference_hash=data["reference_hash"],
            compression_ratio=data.get("compression_ratio", 0.0),
        )

    @property
    def nbytes(self) -> int:
        """Get size in bytes of the delta including metadata overhead."""
        try:
            # Ensure we have a proper numpy array
            if not hasattr(self.data, "nbytes"):
                raise TypeError(f"Delta data is not a numpy array: {type(self.data)}")

            # Get data size safely
            data_nbytes = self.data.nbytes
            if not isinstance(data_nbytes, int):
                raise TypeError(
                    f"data.nbytes returned non-int: {type(data_nbytes)} = {data_nbytes}"
                )

            # Calculate metadata size
            metadata_size = len(json.dumps(self.metadata).encode())
            if not isinstance(metadata_size, int):
                raise TypeError(
                    f"metadata size calculation returned non-int: {type(metadata_size)}"
                )

            # Add constant overhead for delta object fields
            return data_nbytes + metadata_size + DELTA_OBJECT_OVERHEAD_BYTES
        except Exception as e:
            logger.error(f"Error calculating Delta.nbytes: {e}")
            logger.error(f"  self.data: {self.data} (type: {type(self.data)})")
            logger.error(
                f"  self.metadata: {self.metadata} (type: {type(self.metadata)})"
            )
            raise


class DeltaEncoder:
    """Encoder for creating and applying delta encodings."""

    def __init__(self, config: Optional[DeltaConfig] = None):
        self.config = config or DeltaConfig()

    def can_encode_as_delta(
        self, weight: WeightTensor, reference: WeightTensor
    ) -> bool:
        """Check if weight can be efficiently encoded as delta from reference."""
        if weight.shape != reference.shape or weight.dtype != reference.dtype:
            return False

        # Skip very small weights - overhead makes delta encoding inefficient
        if weight.nbytes < self.config.min_weight_size:
            logger.debug(
                f"Skipping delta encoding for small weight: {weight.nbytes} bytes < "
                f"{self.config.min_weight_size} bytes"
            )
            return False

        # Calculate theoretical delta size including overhead
        delta_data = weight.data - reference.data

        if self.config.delta_type == DeltaType.SPARSE:
            # For sparse, count non-zero elements
            non_zero = np.abs(delta_data) > self.config.sparse_threshold
            estimated_data_size = np.sum(non_zero) * (
                np.dtype(weight.dtype).itemsize + 4
            )  # value + index
        elif self.config.delta_type in [
            DeltaType.INT8_QUANTIZED,
            DeltaType.INT16_QUANTIZED,
        ]:
            # For quantized, fixed size based on quantization
            bits = 8 if self.config.delta_type == DeltaType.INT8_QUANTIZED else 16
            estimated_data_size = weight.size * bits // 8
        elif self.config.delta_type == DeltaType.COMPRESSED:
            # For compressed, estimate 50% compression ratio
            estimated_data_size = weight.nbytes // 2
        else:
            # For raw, same size as original data
            estimated_data_size = weight.nbytes

        # Add metadata and object overhead
        total_overhead = DELTA_METADATA_OVERHEAD_BYTES + DELTA_OBJECT_OVERHEAD_BYTES
        estimated_total_size = estimated_data_size + total_overhead

        # Check if delta encoding is worthwhile
        # For FLOAT32_RAW and COMPRESSED, we always allow encoding as they enable
        # lossless deduplication
        if self.config.delta_type in [DeltaType.FLOAT32_RAW, DeltaType.COMPRESSED]:
            # These deltas enable lossless deduplication
            return True

        # For other types, check compression efficiency
        compression_ratio = (weight.nbytes - estimated_total_size) / weight.nbytes

        if compression_ratio < self.config.min_compression_ratio:
            logger.debug(
                f"Delta encoding not efficient: {compression_ratio:.2%} < "
                f"{self.config.min_compression_ratio:.2%}"
            )
            return False

        # Also check max delta ratio for backwards compatibility
        ratio = estimated_total_size / weight.nbytes
        return ratio <= self.config.max_delta_ratio

    def encode_delta(self, weight: WeightTensor, reference: WeightTensor) -> Delta:
        """Encode weight as delta from reference."""
        if weight.shape != reference.shape or weight.dtype != reference.dtype:
            raise ValueError("Weight and reference must have same shape and dtype")

        # Ensure reference hash is computed and cached
        reference_hash = reference.compute_hash()

        # Calculate raw delta
        delta_data = weight.data - reference.data

        # Apply encoding strategy
        # Handle both string and enum types for delta_type
        delta_type = self.config.delta_type
        if isinstance(delta_type, str):
            delta_type = DeltaType(delta_type)

        if delta_type == DeltaType.FLOAT32_RAW:
            encoded_data, metadata = self._encode_raw(delta_data)
        elif delta_type == DeltaType.INT8_QUANTIZED:
            encoded_data, metadata = self._encode_quantized(delta_data, 8)
        elif delta_type == DeltaType.INT16_QUANTIZED:
            encoded_data, metadata = self._encode_quantized(delta_data, 16)
        elif delta_type == DeltaType.SPARSE:
            encoded_data, metadata = self._encode_sparse(delta_data)
        elif delta_type == DeltaType.COMPRESSED:
            encoded_data, metadata = self._encode_compressed(delta_data)
        else:
            raise ValueError(f"Unsupported delta type: {delta_type}")

        # Calculate compression ratio with proper handling of metadata overhead
        try:
            original_size = weight.nbytes
            if not isinstance(original_size, int):
                raise TypeError(
                    f"weight.nbytes returned non-int: {type(original_size)} = "
                    f"{original_size}"
                )

            metadata_size = len(json.dumps(metadata).encode())
            if not isinstance(metadata_size, int):
                raise TypeError(
                    f"metadata size calculation returned non-int: {type(metadata_size)}"
                )

            # Ensure encoded_data has proper nbytes
            if not hasattr(encoded_data, "nbytes"):
                raise TypeError(
                    f"encoded_data is not a numpy array: {type(encoded_data)}"
                )

            encoded_nbytes = encoded_data.nbytes
            if not isinstance(encoded_nbytes, int):
                raise TypeError(
                    f"encoded_data.nbytes returned non-int: {type(encoded_nbytes)} = "
                    f"{encoded_nbytes}"
                )

            delta_size = encoded_nbytes + metadata_size + DELTA_OBJECT_OVERHEAD_BYTES

        except Exception as e:
            logger.error(f"Error calculating delta size in encode_delta: {e}")
            logger.error(f"  weight: {weight} (type: {type(weight)})")
            logger.error(f"  encoded_data: {encoded_data} (type: {type(encoded_data)})")
            logger.error(f"  metadata: {metadata} (type: {type(metadata)})")
            raise

        # Use a more meaningful compression ratio calculation
        # Positive values indicate compression, negative values indicate expansion
        if original_size > 0:
            compression_ratio = (original_size - delta_size) / original_size
        else:
            compression_ratio = 0.0

        return Delta(
            delta_type=self.config.delta_type,
            data=encoded_data,
            metadata=metadata,
            original_shape=weight.shape,
            original_dtype=weight.dtype,
            reference_hash=reference_hash,  # Use the pre-computed hash
            compression_ratio=compression_ratio,
        )

    def decode_delta(self, delta: Delta, reference: WeightTensor) -> WeightTensor:
        """Decode delta and reconstruct original weight.

        Args:
            delta: The delta encoding to apply
            reference: The reference weight to apply delta to

        Returns:
            Reconstructed weight tensor

        Raises:
            DeltaReconstructionError: If strict_reconstruction is enabled and
                reference hash doesn't match the expected hash in delta
        """
        # Ensure reference hash is computed and validate against delta
        reference_hash = reference.compute_hash()
        if reference_hash != delta.reference_hash:
            message = (
                f"Reference hash mismatch during delta decoding: "
                f"expected {delta.reference_hash}, got {reference_hash}. "
                f"This may indicate data corruption or storage serialization issues."
            )
            if self.config.strict_reconstruction:
                raise DeltaReconstructionError(message)
            logger.warning(f"{message} Decoding will continue (strict mode disabled).")

        # Decode delta data based on type
        # Handle both string and enum types for backwards compatibility
        delta_type = delta.delta_type
        if isinstance(delta_type, str):
            delta_type = DeltaType(delta_type)

        if delta_type == DeltaType.FLOAT32_RAW:
            delta_data = self._decode_raw(delta.data, delta.metadata)
        elif delta_type == DeltaType.INT8_QUANTIZED:
            delta_data = self._decode_quantized(delta.data, delta.metadata, 8)
        elif delta_type == DeltaType.INT16_QUANTIZED:
            delta_data = self._decode_quantized(delta.data, delta.metadata, 16)
        elif delta_type == DeltaType.SPARSE:
            delta_data = self._decode_sparse(
                delta.data, delta.metadata, delta.original_shape
            )
        elif delta_type == DeltaType.COMPRESSED:
            delta_data = self._decode_compressed(delta.data, delta.metadata)
        else:
            raise ValueError(f"Unsupported delta type: {delta_type}")

        # Reconstruct original weight
        reconstructed_data = reference.data + delta_data

        # Create new weight tensor with reconstructed data
        reconstructed_weight = WeightTensor(
            data=reconstructed_data.astype(delta.original_dtype),
            metadata=reference.metadata,  # Use reference metadata, update if needed
        )

        return reconstructed_weight

    def _encode_raw(self, delta_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Encode as raw float32 differences."""
        return delta_data.astype(np.float32), {}

    def _decode_raw(
        self, encoded_data: np.ndarray, metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Decode raw float32 differences."""
        return encoded_data

    def _encode_quantized(
        self, delta_data: np.ndarray, bits: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Encode as quantized differences."""
        # Calculate scale and offset for quantization
        min_val = np.min(delta_data)
        max_val = np.max(delta_data)

        if min_val == max_val:
            # Constant delta
            scale = 1.0
            offset = min_val
            quantized = np.zeros_like(
                delta_data, dtype=np.int8 if bits == 8 else np.int16
            )
        else:
            # Quantize to target bit depth
            if bits == 8:
                quant_min, quant_max = -128, 127
                dtype = np.int8
            else:  # 16 bits
                quant_min, quant_max = -32768, 32767
                dtype = np.int16

            scale = (max_val - min_val) / (quant_max - quant_min)
            offset = min_val

            # Quantize
            normalized = (delta_data - offset) / scale
            quantized = np.clip(normalized, quant_min, quant_max).astype(dtype)

        metadata = {"scale": float(scale), "offset": float(offset), "bits": bits}

        return quantized, metadata

    def _decode_quantized(
        self, encoded_data: np.ndarray, metadata: Dict[str, Any], bits: int
    ) -> np.ndarray:
        """Decode quantized differences."""
        scale = metadata["scale"]
        offset = metadata["offset"]

        # Dequantize
        decoded = encoded_data.astype(np.float32) * scale + offset
        return decoded

    def _encode_sparse(
        self, delta_data: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Encode as sparse differences (only non-zero values)."""
        # Find non-zero elements
        mask = np.abs(delta_data) > self.config.sparse_threshold
        indices = np.where(mask)
        values = delta_data[mask]

        # Flatten indices for storage
        flat_indices = np.ravel_multi_index(indices, delta_data.shape)

        # Combine indices and values
        sparse_data = np.column_stack([flat_indices, values.astype(np.float32)])

        metadata = {
            "original_shape": delta_data.shape,
            "num_nonzero": len(values),
            "sparse_threshold": self.config.sparse_threshold,
        }

        return sparse_data.astype(np.float32), metadata

    def _decode_sparse(
        self,
        encoded_data: np.ndarray,
        metadata: Dict[str, Any],
        original_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Decode sparse differences."""
        # Reconstruct full delta array
        delta_data = np.zeros(original_shape, dtype=np.float32)

        if encoded_data.size > 0:
            indices = encoded_data[:, 0].astype(np.int64)
            values = encoded_data[:, 1]

            # Convert flat indices back to multi-dimensional
            multi_indices = np.unravel_index(indices, original_shape)
            delta_data[multi_indices] = values

        return delta_data

    def _encode_compressed(
        self, delta_data: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Encode with compression."""
        import zlib

        # Compress raw bytes
        raw_bytes = delta_data.astype(np.float32).tobytes()
        compressed_bytes = zlib.compress(raw_bytes, level=self.config.compression_level)

        # Convert to numpy array for consistent storage
        compressed_array = np.frombuffer(compressed_bytes, dtype=np.uint8)

        metadata = {
            "original_shape": delta_data.shape,
            "compression_level": self.config.compression_level,
            "original_size": len(raw_bytes),
            "compressed_size": len(compressed_bytes),
        }

        return compressed_array, metadata

    def _decode_compressed(
        self, encoded_data: np.ndarray, metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Decode compressed differences."""
        import zlib

        # Decompress
        compressed_bytes = encoded_data.tobytes()
        raw_bytes = zlib.decompress(compressed_bytes)

        # Reconstruct array
        original_shape = tuple(metadata["original_shape"])
        delta_data = np.frombuffer(raw_bytes, dtype=np.float32).reshape(original_shape)

        return delta_data

    def estimate_delta_size(self, weight: WeightTensor, reference: WeightTensor) -> int:
        """Estimate delta size without actually encoding."""
        if weight.shape != reference.shape or weight.dtype != reference.dtype:
            return weight.nbytes  # Fallback to full size

        delta_data = weight.data - reference.data

        if self.config.delta_type == DeltaType.SPARSE:
            non_zero = np.sum(np.abs(delta_data) > self.config.sparse_threshold)
            return non_zero * 8  # 4 bytes index + 4 bytes value
        elif self.config.delta_type == DeltaType.INT8_QUANTIZED:
            return weight.size
        elif self.config.delta_type == DeltaType.INT16_QUANTIZED:
            return weight.size * 2
        else:
            return weight.nbytes  # Raw float32 size
