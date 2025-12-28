# Delta Module

This module provides delta encoding strategies for storing differences between similar neural network weights, enabling efficient deduplication with configurable compression/fidelity trade-offs.

## Overview

Delta encoding stores the difference between a weight tensor and a reference tensor. When weights are similar (e.g., fine-tuned models), the delta is small and highly compressible. This module provides:

- **Multiple encoding strategies** (lossless and lossy)
- **Configurable compression** with trade-offs between size and fidelity
- **Automatic compression selection** based on data characteristics

## Key Files

### `delta_encoder.py`

The main encoder implementation with multiple strategies.

**Constants**:
- `DELTA_OBJECT_OVERHEAD_BYTES = 200` - Metadata overhead per delta
- `DELTA_METADATA_OVERHEAD_BYTES = 200` - JSON encoding overhead
- `DEFAULT_SPARSE_THRESHOLD = 1e-6` - Threshold for sparse encoding
- `DEFAULT_COMPRESSION_LEVEL = 6` - zlib compression level
- `DEFAULT_MIN_WEIGHT_SIZE_BYTES = 512` - Minimum size for delta encoding
- `DEFAULT_SIMILARITY_THRESHOLD = 0.98` - Similarity threshold for deduplication

**DeltaType** (Enum):
```python
# Lossless (perfect reconstruction)
FLOAT32_RAW = "float32_raw"      # No compression
COMPRESSED = "compressed"        # zlib-compressed float32
XOR_FLOAT32 = "xor_float32"     # Bitwise XOR, 15-25% better
XOR_BFLOAT16 = "xor_bfloat16"   # Optimized for BF16
EXPONENT_MANTISSA = "exponent_mantissa"  # Separate component encoding

# Lossy (approximate reconstruction)
INT8_QUANTIZED = "int8_quantized"   # ~75% compression
INT16_QUANTIZED = "int16_quantized" # ~50% compression
SPARSE = "sparse"                    # Discards small differences
PER_AXIS_SCALED = "per_axis_scaled" # 1-bit signs + axis scales
```

Use `delta_type.is_lossless` to check if a strategy is lossless.

**DeltaConfig** (dataclass):
- `delta_type`: Encoding strategy (default: FLOAT32_RAW)
- `sparse_threshold`: Threshold for sparse encoding (default: 1e-6)
- `quantization_bits`: 8 or 16 for quantized encoding
- `compression_level`: zlib level 1-9 (default: 6)
- `max_delta_ratio`: Max delta/original size ratio (default: 1.0)
- `min_compression_ratio`: Minimum compression to encode (default: 0.0)
- `min_weight_size`: Minimum weight size in bytes (default: 512)
- `strict_reconstruction`: Raise error on hash mismatch (default: False)

**Delta** (dataclass):
- `delta_type`: Encoding type used
- `data`: Encoded numpy array
- `metadata`: Strategy-specific metadata
- `original_shape`: Original weight shape
- `original_dtype`: Original weight dtype
- `reference_hash`: Hash of reference weight
- `compression_ratio`: Achieved compression (positive = savings)

**DeltaEncoder** (class):

Key methods:
```python
encoder.can_encode_as_delta(weight, reference)  # Check if encoding is worthwhile
encoder.encode_delta(weight, reference)          # Create delta encoding
encoder.decode_delta(delta, reference)           # Reconstruct original
encoder.estimate_delta_size(weight, reference)   # Estimate size without encoding
```

**DeltaReconstructionError**: Raised when strict reconstruction fails due to hash mismatch.

### Encoding Strategies Explained

**FLOAT32_RAW**:
- Stores `weight - reference` as float32
- No compression, fastest encoding/decoding
- Use when speed matters more than size

**COMPRESSED**:
- zlib compression of float32 differences
- Typically ~50% compression
- Good balance of speed and size

**XOR_FLOAT32** (Lossless, Best Compression):
- Reinterprets floats as uint32, computes bitwise XOR
- Separates sign+exponent (9 bits) and mantissa (23 bits)
- Compresses each stream independently
- Exploits identical exponents in similar weights
- 15-25% better than COMPRESSED

**XOR_BFLOAT16** (Lossless):
- Optimized for BFloat16 weights (common in ML training)
- Takes upper 16 bits for BF16 representation
- Stores lower bits separately for lossless reconstruction

**EXPONENT_MANTISSA** (Lossless):
- Separates delta into sign, exponent, and mantissa components
- Packs signs (1 bit each) separately
- Compresses each stream with optimal settings
- 10-20% better than COMPRESSED

**INT8_QUANTIZED** (Lossy):
- Quantizes differences to 8-bit integers
- Stores scale and offset for dequantization
- ~75% compression with quantization error

**INT16_QUANTIZED** (Lossy):
- Quantizes to 16-bit integers
- ~50% compression with smaller error than INT8

**SPARSE** (Lossy):
- Only stores values above threshold
- Discards small differences (< sparse_threshold)
- Best for weights with many near-zero deltas

**PER_AXIS_SCALED** (Lossy):
- Stores 1-bit signs per element
- Per-row and per-column FP16 scale factors
- Reconstructs as `sign * row_scale * col_scale`
- ~5x compression for structured deltas

### `compression.py`

Additional compression utilities.

**DeltaCompressor** (class):
- `compress_sparse_deltas()` - RLE for sparse data
- `decompress_sparse_deltas()` - Decompress RLE
- `adaptive_quantization()` - 3-sigma range quantization
- `dequantize_adaptive()` - Dequantize adaptive data
- `compress_with_dictionary()` - Dictionary-based compression
- `decompress_with_dictionary()` - Dictionary decompression
- `delta_statistics()` - Analyze delta characteristics
- `recommend_compression()` - Suggest optimal strategy

**Automatic Strategy Selection**:
```python
stats = DeltaCompressor.delta_statistics(delta_data)
recommended = DeltaCompressor.recommend_compression(delta_data)
# Returns: "sparse", "int8_quantized", "int16_quantized",
#          "compressed", or "float32_raw"
```

## Usage Examples

### Basic Delta Encoding

```python
from coral.delta import DeltaEncoder, DeltaConfig, DeltaType

# Configure lossless compression
config = DeltaConfig(delta_type=DeltaType.COMPRESSED)
encoder = DeltaEncoder(config)

# Check if delta encoding is worthwhile
if encoder.can_encode_as_delta(weight, reference):
    # Encode
    delta = encoder.encode_delta(weight, reference)
    print(f"Compression: {delta.compression_ratio:.2%}")

    # Decode
    reconstructed = encoder.decode_delta(delta, reference)
    # For lossless types: np.array_equal(reconstructed.data, weight.data)
```

### Advanced XOR Encoding

```python
# Best lossless compression for similar weights
config = DeltaConfig(delta_type=DeltaType.XOR_FLOAT32)
encoder = DeltaEncoder(config)

delta = encoder.encode_delta(weight, reference)
# Typically 15-25% better than COMPRESSED
```

### Lossy Encoding for Training Checkpoints

```python
# 8-bit quantization for checkpoint storage
config = DeltaConfig(
    delta_type=DeltaType.INT8_QUANTIZED,
    min_compression_ratio=0.5  # Only encode if >50% compression
)
encoder = DeltaEncoder(config)

delta = encoder.encode_delta(checkpoint_weight, reference)
# ~75% size reduction with small quantization error
```

### Automatic Strategy Selection

```python
from coral.delta.compression import DeltaCompressor

# Analyze delta characteristics
delta_data = weight.data - reference.data
stats = DeltaCompressor.delta_statistics(delta_data)
print(f"Sparsity: {stats['sparsity']:.2%}")
print(f"Entropy: {stats['entropy']:.2f} bits")

# Get recommendation
strategy = DeltaCompressor.recommend_compression(delta_data)
config = DeltaConfig(delta_type=DeltaType(strategy))
```

## Choosing a Strategy

| Strategy | Lossless | Typical Compression | Best For |
|----------|----------|---------------------|----------|
| FLOAT32_RAW | Yes | 0% | Speed priority |
| COMPRESSED | Yes | 50% | General use |
| XOR_FLOAT32 | Yes | 60-70% | Best lossless compression |
| EXPONENT_MANTISSA | Yes | 55-65% | Fine-tuned weights |
| INT8_QUANTIZED | No | 75% | Training checkpoints |
| INT16_QUANTIZED | No | 50% | Balance of error/compression |
| SPARSE | No | Variable | Sparse deltas (>70% near-zero) |
| PER_AXIS_SCALED | No | 80% | Structured fine-tuning deltas |

## Dependencies

- `numpy` - Array operations
- `zlib` (stdlib) - Compression for COMPRESSED, XOR_* types
- Internal: `coral.core.weight_tensor`

## Testing

Related test files:
- `tests/test_delta_encoding.py` - Basic encoding/decoding
- `tests/test_advanced_delta_encoding.py` - XOR and exponent-mantissa strategies
- `tests/test_delta_compression.py` - Compression utilities
- `tests/test_delta_reconstruction_consistency.py` - Lossless verification
