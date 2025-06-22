"""Delta encoding for similar weights to enable lossless deduplication."""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core.weight_tensor import WeightTensor
from .product_quantization import PQConfig, PQCodebook, train_codebooks, encode_vector, decode_vector

logger = logging.getLogger(__name__)


class DeltaType(Enum):
    """Types of delta encoding strategies."""

    FLOAT32_RAW = "float32_raw"  # Raw float32 differences
    INT8_QUANTIZED = "int8_quantized"  # Quantized to int8 with scale/offset
    INT16_QUANTIZED = "int16_quantized"  # Quantized to int16 with scale/offset
    SPARSE = "sparse"  # Store only non-zero differences
    COMPRESSED = "compressed"  # Compressed raw differences
    PQ_ENCODED = "pq_encoded"  # Lossy PQ encoding
    PQ_LOSSLESS = "pq_lossless"  # PQ with residuals for perfect reconstruction


@dataclass
class DeltaConfig:
    """Configuration for delta encoding."""

    delta_type: DeltaType = DeltaType.FLOAT32_RAW
    sparse_threshold: float = (
        1e-6  # Values below this are considered zero for sparse encoding
    )
    quantization_bits: int = 8  # Bits for quantized encoding (8 or 16)
    compression_level: int = 6  # Compression level for compressed deltas
    max_delta_ratio: float = (
        1.0  # Allow delta up to 100% of original size (for raw encoding)
    )
    min_compression_ratio: float = (
        0.0  # Minimum compression ratio to store delta (0% = always store if efficient)
    )
    min_weight_size: int = (
        512  # Minimum weight size in bytes to consider delta encoding
    )
    # Product Quantization parameters
    pq_num_subvectors: int = 8
    pq_bits_per_subvector: int = 8
    pq_codebook_learning_samples: int = 10000
    pq_use_residual: bool = True

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
            "pq_num_subvectors": self.pq_num_subvectors,
            "pq_bits_per_subvector": self.pq_bits_per_subvector,
            "pq_codebook_learning_samples": self.pq_codebook_learning_samples,
            "pq_use_residual": self.pq_use_residual,
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
            object_overhead = (
                200  # Approximate overhead for delta type, shape, dtype, etc.
            )

            return data_nbytes + metadata_size + object_overhead
        except Exception as e:
            logger.error(f"Error calculating Delta.nbytes: {e}")
            logger.error(f"  self.data: {self.data} (type: {type(self.data)})")
            logger.error(
                f"  self.metadata: {self.metadata} (type: {type(self.metadata)})"
            )
            raise

    def compute_hash(self) -> str:
        """
        Compute hash for this delta object.
        
        Uses the same logic as HDF5Store._compute_delta_hash for consistency.
        
        Returns:
            Hexadecimal hash string
        """
        import xxhash
        
        hasher = xxhash.xxh3_64()
        hasher.update(self.reference_hash.encode())
        hasher.update(self.data.tobytes())
        hasher.update(str(self.delta_type.value).encode())
        return hasher.hexdigest()


class DeltaEncoder:
    """Encoder for creating and applying delta encodings."""

    def __init__(self, config: Optional[DeltaConfig] = None):
        self.config = config or DeltaConfig()
        self._pq_codebooks: Dict[Tuple[Tuple[int, ...], np.dtype], PQCodebook] = {}

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
        
        # For PQ encoding, require minimum size of 1KB
        if self.config.delta_type in [DeltaType.PQ_ENCODED, DeltaType.PQ_LOSSLESS]:
            if weight.nbytes < 1024:  # 1KB minimum for PQ
                logger.debug(
                    f"Skipping PQ encoding for small weight: {weight.nbytes} bytes < 1KB"
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
        elif self.config.delta_type == DeltaType.PQ_ENCODED:
            # PQ lossy: just indices
            indices_size = self.config.pq_num_subvectors * 4
            estimated_data_size = indices_size
        elif self.config.delta_type == DeltaType.PQ_LOSSLESS:
            # PQ lossless: indices + residual
            indices_size = self.config.pq_num_subvectors * 4
            residual_size = weight.nbytes
            estimated_data_size = indices_size + residual_size
        else:
            # For raw, same size as original data
            estimated_data_size = weight.nbytes

        # Add metadata and object overhead
        metadata_overhead = 200  # Approximate metadata size
        object_overhead = 200  # Approximate object overhead
        estimated_total_size = estimated_data_size + metadata_overhead + object_overhead

        # Check if delta encoding is worthwhile
        # For FLOAT32_RAW, COMPRESSED, and PQ_LOSSLESS, we always allow encoding as they enable
        # lossless deduplication
        if self.config.delta_type in [DeltaType.FLOAT32_RAW, DeltaType.COMPRESSED, DeltaType.PQ_LOSSLESS]:
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
        elif delta_type == DeltaType.PQ_ENCODED:
            try:
                encoded_data, metadata = self._encode_pq(delta_data, use_residual=False)
            except Exception as e:
                logger.warning(f"PQ encoding failed: {e}. Falling back to COMPRESSED.")
                encoded_data, metadata = self._encode_compressed(delta_data)
                delta_type = DeltaType.COMPRESSED
        elif delta_type == DeltaType.PQ_LOSSLESS:
            try:
                encoded_data, metadata = self._encode_pq(delta_data, use_residual=True)
            except Exception as e:
                logger.warning(f"PQ lossless encoding failed: {e}. Falling back to COMPRESSED.")
                encoded_data, metadata = self._encode_compressed(delta_data)
                delta_type = DeltaType.COMPRESSED
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

            object_overhead = 200  # Approximate overhead for delta object fields
            delta_size = encoded_nbytes + metadata_size + object_overhead

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
            delta_type=delta_type,  # Use the potentially updated delta_type
            data=encoded_data,
            metadata=metadata,
            original_shape=weight.shape,
            original_dtype=weight.dtype,
            reference_hash=reference_hash,  # Use the pre-computed hash
            compression_ratio=compression_ratio,
        )

    def decode_delta(self, delta: Delta, reference: WeightTensor) -> WeightTensor:
        """Decode delta and reconstruct original weight."""
        # Ensure reference hash is computed and validate against delta
        reference_hash = reference.compute_hash()
        if reference_hash != delta.reference_hash:
            logger.warning(
                f"Reference hash mismatch during delta decoding: "
                f"expected {delta.reference_hash}, got {reference_hash}. "
                f"This may indicate storage serialization issues but decoding "
                f"will continue."
            )

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
        elif delta_type in [DeltaType.PQ_ENCODED, DeltaType.PQ_LOSSLESS]:
            delta_data = self._decode_pq(delta.data, delta.metadata)
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

    def _get_or_create_codebook(self, shape: Tuple[int, ...], dtype: np.dtype) -> PQCodebook:
        """Get existing or create new PQ codebook for given shape/dtype."""
        key = (shape, str(dtype))
        if key not in self._pq_codebooks:
            # Need to train a new codebook - this requires sample data
            # For now, we'll create a placeholder that will be trained on first use
            logger.info(f"Creating new PQ codebook for shape {shape}, dtype {dtype}")
            self._pq_codebooks[key] = None  # Will be trained on first encode
        return self._pq_codebooks[key]

    def _train_pq_codebook(self, delta_data: np.ndarray) -> PQCodebook:
        """Train PQ codebook on delta data."""
        # Flatten the delta data for training
        flat_delta = delta_data.flatten()
        
        # Create PQ config
        pq_config = PQConfig(
            num_subvectors=self.config.pq_num_subvectors,
            bits_per_subvector=self.config.pq_bits_per_subvector,
            use_residual=self.config.pq_use_residual
        )
        
        # Generate training samples by creating multiple vectors from the delta
        # Each vector will be a sliding window of the flattened delta
        vector_dim = flat_delta.size
        
        # For PQ to work properly, we need multiple training vectors
        # We'll create them by using overlapping windows or by adding noise
        num_samples = min(self.config.pq_codebook_learning_samples, 1000)
        
        if flat_delta.size < 100:
            # For very small deltas, create synthetic training data
            training_vectors = np.zeros((num_samples, flat_delta.size))
            for i in range(num_samples):
                # Add varying amounts of noise to create diverse training samples
                noise_scale = 0.1 * np.std(flat_delta) if np.std(flat_delta) > 0 else 0.01
                training_vectors[i] = flat_delta + noise_scale * np.random.randn(flat_delta.size)
        else:
            # For larger deltas, we can create sliding windows or add noise
            training_vectors = np.zeros((num_samples, flat_delta.size))
            for i in range(num_samples):
                # Create variations by adding small noise
                noise_scale = 0.05 * np.std(flat_delta) if np.std(flat_delta) > 0 else 0.01
                training_vectors[i] = flat_delta + noise_scale * np.random.randn(flat_delta.size)
        
        # Train codebook
        codebook = train_codebooks(training_vectors, pq_config)
        
        # Cache the trained codebook
        key = (delta_data.shape, str(delta_data.dtype))
        self._pq_codebooks[key] = codebook
        
        return codebook

    def _encode_pq(self, delta_data: np.ndarray, use_residual: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Encode delta using Product Quantization."""
        # Get or create codebook for this shape/dtype
        key = (delta_data.shape, str(delta_data.dtype))
        codebook = self._pq_codebooks.get(key)
        
        if codebook is None:
            # Need to train codebook first
            codebook = self._train_pq_codebook(delta_data)
        
        # Create PQ config for encoding
        pq_config = PQConfig(
            num_subvectors=self.config.pq_num_subvectors,
            bits_per_subvector=self.config.pq_bits_per_subvector,
            use_residual=use_residual
        )
        
        # Flatten delta for PQ encoding
        flat_delta = delta_data.flatten()
        
        # Encode using PQ
        indices, residual = encode_vector(flat_delta, codebook, pq_config)
        
        # Prepare encoded data
        if use_residual and residual is not None:
            # Store both indices and residual
            # Pack indices as uint32 and residual as float32
            encoded_data = np.concatenate([
                indices.astype(np.uint32).view(np.uint8),
                residual.astype(np.float32).view(np.uint8)
            ])
            metadata = {
                "original_shape": delta_data.shape,
                "num_indices": len(indices),
                "has_residual": True,
                "indices_dtype": "uint32",
                "residual_dtype": "float32",
                "codebook_key": str(key),  # Store reference to cached codebook instead of full codebook
                "pq_config": {
                    "num_subvectors": pq_config.num_subvectors,
                    "bits_per_subvector": pq_config.bits_per_subvector,
                }
            }
        else:
            # Store only indices
            encoded_data = indices.astype(np.uint32).view(np.uint8)
            metadata = {
                "original_shape": delta_data.shape,
                "num_indices": len(indices),
                "has_residual": False,
                "indices_dtype": "uint32",
                "codebook_key": str(key),  # Store reference to cached codebook instead of full codebook
                "pq_config": {
                    "num_subvectors": pq_config.num_subvectors,
                    "bits_per_subvector": pq_config.bits_per_subvector,
                }
            }
        
        return encoded_data, metadata

    def _decode_pq(self, encoded_data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Decode PQ-encoded delta."""
        # Get cached codebook using key from metadata
        codebook_key = metadata.get("codebook_key")
        if codebook_key:
            # Parse the key back to tuple
            import ast
            key = ast.literal_eval(codebook_key)
            codebook = self._pq_codebooks.get(key)
            if codebook is None:
                raise ValueError(f"Codebook not found for key: {key}")
        else:
            # Fallback: reconstruct from full codebook data if available
            if "codebook" in metadata:
                codebook = PQCodebook.from_dict(metadata["codebook"])
            else:
                raise ValueError("No codebook reference or data found in metadata")
        
        # Extract indices
        num_indices = metadata["num_indices"]
        indices_dtype = np.dtype(metadata["indices_dtype"])
        indices_bytes = num_indices * indices_dtype.itemsize
        
        indices = encoded_data[:indices_bytes].view(indices_dtype)
        
        # Extract residual if present
        residual = None
        if metadata.get("has_residual", False):
            residual_dtype = np.dtype(metadata["residual_dtype"])
            residual_data = encoded_data[indices_bytes:].view(residual_dtype)
            residual = residual_data
        
        # Get original shape for reconstruction
        original_shape = tuple(metadata["original_shape"])
        original_size = np.prod(original_shape)
        
        # Decode vector
        reconstructed_flat = decode_vector(indices, codebook, residual, original_size)
        
        # Reshape to original shape
        reconstructed = reconstructed_flat.reshape(original_shape)
        
        return reconstructed

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
        elif self.config.delta_type == DeltaType.PQ_ENCODED:
            # PQ: M indices (4 bytes each) + codebook metadata
            indices_size = self.config.pq_num_subvectors * 4
            codebook_overhead = 1024  # Estimated codebook metadata overhead
            return indices_size + codebook_overhead
        elif self.config.delta_type == DeltaType.PQ_LOSSLESS:
            # PQ + residual: indices + full residual vector
            indices_size = self.config.pq_num_subvectors * 4
            residual_size = weight.nbytes  # Full precision residual
            codebook_overhead = 1024
            return indices_size + residual_size + codebook_overhead
        else:
            return weight.nbytes  # Raw float32 size
