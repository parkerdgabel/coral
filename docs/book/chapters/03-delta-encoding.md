# Chapter 3: Delta Encoding System

## Introduction

One of Coral's most innovative features is its delta encoding system, which enables efficient storage of similar neural network weights without sacrificing reconstruction fidelity. This chapter explores how delta encoding solves the fundamental challenge of storing model variations—from fine-tuning snapshots to training checkpoints—in a way that is both space-efficient and maintains perfect reconstruction when needed.

## 3.1 The Problem Delta Encoding Solves

### 3.1.1 Why Similar Weights Waste Storage

In typical machine learning workflows, we frequently encounter scenarios where neural network weights are nearly identical:

- **Fine-tuning**: When adapting a pre-trained model to a new task, most weights change by less than 1%
- **Training Checkpoints**: Consecutive checkpoints during training differ only slightly as the model converges
- **Hyperparameter Sweeps**: Models trained with different learning rates often have 95-99% similarity
- **Ensemble Models**: Multiple models trained from similar initializations share substantial weight overlap

Consider a realistic example: a ResNet-50 model fine-tuned for 10 epochs, saving checkpoints every epoch. Each checkpoint contains ~25 million parameters (100MB at float32 precision). Without deduplication, storing 10 checkpoints requires 1GB of storage. Yet consecutive checkpoints often differ by less than 0.5%—meaning we're storing nearly identical data repeatedly.

### 3.1.2 The Challenge of Slight Variations

Naive approaches to this problem fall short:

**Hash-based Deduplication**: Content-addressable storage systems (like git) use cryptographic hashes to identify duplicate data. However, even a single bit difference in a weight tensor produces a completely different hash, making exact deduplication useless for nearly-identical weights.

```python
# Example: Tiny differences break hash-based deduplication
import numpy as np
import xxhash

weight_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
weight_b = np.array([1.0, 2.0, 3.0000001], dtype=np.float32)  # 99.99999% similar

hash_a = xxhash.xxh64(weight_a.tobytes()).hexdigest()
hash_b = xxhash.xxh64(weight_b.tobytes()).hexdigest()

print(f"Hash A: {hash_a}")
print(f"Hash B: {hash_b}")
print(f"Identical: {hash_a == hash_b}")  # False!
```

**Lossy Compression**: Standard compression algorithms (gzip, zlib) can reduce storage but still treat each weight tensor independently. More aggressive lossy compression techniques (quantization, pruning) sacrifice reconstruction fidelity—you cannot recover the original weights exactly.

### 3.1.3 Previous Approaches and Their Limitations

Before Coral's delta encoding system, there were two main approaches:

1. **Store Every Version Independently**
   - Simple but wasteful
   - Redundant storage of similar data
   - Example: 10 checkpoints = 10× storage cost

2. **Use Lossy Deduplication**
   - Replace similar weights with a single representative
   - Reduces storage but loses information
   - Cannot reconstruct original weights exactly
   - Unacceptable for archival or reproducibility

### 3.1.4 The Need for Lossless Reconstruction

Machine learning applications demand perfect reconstruction in many scenarios:

- **Scientific Reproducibility**: Research experiments must be exactly reproducible
- **Model Debugging**: Analyzing training dynamics requires precise weight values
- **Legal/Compliance**: Some domains require bit-exact model archival
- **Ensemble Methods**: Model ensembles rely on exact weight differences

Delta encoding provides a solution: **store the differences between similar weights instead of the weights themselves, enabling perfect reconstruction while achieving significant compression**.

## 3.2 Delta Encoding Concept

### 3.2.1 What is a Delta?

A delta is a compact representation of the **difference** between two weight tensors. Instead of storing:

```
Weight A: [1.00, 2.00, 3.00, 4.00, 5.00]  (20 bytes)
Weight B: [1.01, 2.02, 3.01, 4.03, 5.02]  (20 bytes)
Total: 40 bytes
```

We store:

```
Reference (A): [1.00, 2.00, 3.00, 4.00, 5.00]     (20 bytes)
Delta (B-A):   [0.01, 0.02, 0.01, 0.03, 0.02]     (compressed: ~8 bytes)
Total: 28 bytes (30% savings)
```

### 3.2.2 Reference Weights vs Delta Weights

In Coral's system, weights are classified as either:

- **Reference Weights**: Stored in full, serve as anchors for delta encoding
- **Delta Weights**: Stored as differences from a reference weight

The system automatically identifies similar weights using cosine similarity and stores the most suitable one as the reference:

```python
from coral.core.deduplicator import Deduplicator
from coral.delta import DeltaConfig, DeltaType

# Configure delta encoding
delta_config = DeltaConfig(delta_type=DeltaType.COMPRESSED)
dedup = Deduplicator(
    similarity_threshold=0.98,  # Weights with >98% similarity
    delta_config=delta_config,
    enable_delta_encoding=True
)

# Add weights - first becomes reference, similar ones become deltas
ref_hash = dedup.add_weight(checkpoint_epoch_1["layer.weight"], "epoch_1.weight")
sim_hash = dedup.add_weight(checkpoint_epoch_2["layer.weight"], "epoch_2.weight")

# Check if delta-encoded
print(f"Is delta: {dedup.is_delta_encoded('epoch_2.weight')}")  # True
```

### 3.2.3 Reconstruction Process

Reconstructing a weight from its delta is straightforward:

1. **Retrieve the reference weight** using the stored reference hash
2. **Decode the delta data** according to the delta type
3. **Apply the delta** to the reference: `reconstructed = reference + delta`

```python
# Reconstruction happens transparently when retrieving weights
reconstructed = dedup.get_weight_by_name("epoch_2.weight")
# Returns the exact original weight, reconstructed from reference + delta
```

### 3.2.4 Space Savings Calculation

Compression ratio is calculated as:

```
compression_ratio = (original_size - delta_size) / original_size
```

For example:
- Original weight: 100,000 bytes
- Delta (with metadata): 30,000 bytes
- Compression ratio: (100,000 - 30,000) / 100,000 = **0.70 (70% compression)**

The actual savings depend on:
- How similar the weights are (more similar → better compression)
- Which delta encoding strategy is used
- The compression overhead from metadata

## 3.3 Delta Strategies (DeltaType Enum)

Coral provides multiple delta encoding strategies, each optimized for different trade-offs between compression ratio and reconstruction fidelity.

### 3.3.1 Lossless Strategies

These strategies guarantee **perfect reconstruction** of the original weights.

#### FLOAT32_RAW

The simplest lossless strategy: stores the raw float32 differences with no compression.

```python
from coral.delta import DeltaConfig, DeltaEncoder, DeltaType

config = DeltaConfig(delta_type=DeltaType.FLOAT32_RAW)
encoder = DeltaEncoder(config)

delta = encoder.encode_delta(weight, reference)
reconstructed = encoder.decode_delta(delta, reference)

# Perfect reconstruction guaranteed
assert np.array_equal(reconstructed.data, weight.data)
```

**Characteristics:**
- Compression: ~0-10% (mainly from metadata reduction)
- Speed: Fastest encoding/decoding
- Use case: When you need perfect reconstruction with minimal CPU overhead

#### COMPRESSED

Applies zlib compression to the raw float32 differences.

```python
config = DeltaConfig(
    delta_type=DeltaType.COMPRESSED,
    compression_level=6  # 1-9, higher = better compression but slower
)
encoder = DeltaEncoder(config)

delta = encoder.encode_delta(weight, reference)
# Delta is smaller due to zlib compression
print(f"Compression ratio: {delta.compression_ratio:.1%}")
```

**Characteristics:**
- Compression: ~50-70% (depends on delta characteristics)
- Speed: Moderate (compression overhead)
- Use case: Default choice for most scenarios—good balance of compression and speed

**How it works:**
1. Computes raw delta: `delta = weight - reference`
2. Converts to float32 bytes
3. Applies zlib compression with configurable level
4. Stores compressed bytes + metadata (shape, compression level)

#### XOR_FLOAT32

Advanced bitwise XOR encoding with exponent/mantissa separation.

```python
config = DeltaConfig(delta_type=DeltaType.XOR_FLOAT32)
encoder = DeltaEncoder(config)

delta = encoder.encode_delta(weight, reference)
reconstructed = encoder.decode_delta(delta, reference)
```

**Characteristics:**
- Compression: 15-25% better than COMPRESSED for similar weights
- Speed: Moderate (bitwise operations + compression)
- Use case: When weights are very similar (fine-tuning, consecutive checkpoints)

**How it works:**
1. Reinterprets float32 values as uint32 bit representations
2. Computes bitwise XOR: `xor_result = weight_bits ^ reference_bits`
3. Separates into sign+exponent (9 bits) and mantissa (23 bits)
4. Compresses each stream independently with optimal compression levels
5. Exploits the fact that similar weights often have identical exponents

The key insight: when weights are similar, their exponents are often identical, making the exponent XOR stream highly compressible (often approaching 95% zeros).

#### XOR_BFLOAT16

XOR encoding optimized for BFloat16 weights, commonly used in modern ML training.

```python
config = DeltaConfig(delta_type=DeltaType.XOR_BFLOAT16)
encoder = DeltaEncoder(config)

# Works with float32 data but optimized for BF16 precision
delta = encoder.encode_delta(weight, reference)
```

**Characteristics:**
- Compression: Similar to XOR_FLOAT32
- Speed: Slightly faster (only 16 bits for upper half)
- Use case: Models trained with BFloat16 or when upper 16 bits contain most signal

**How it works:**
1. Extracts upper 16 bits from float32 (BFloat16 representation)
2. XORs the BFloat16 representations
3. Also stores lower 16 bits XOR for lossless reconstruction
4. Compresses both streams

#### EXPONENT_MANTISSA

Separate encoding of float components, inspired by ZipNN research.

```python
config = DeltaConfig(delta_type=DeltaType.EXPONENT_MANTISSA)
encoder = DeltaEncoder(config)

delta = encoder.encode_delta(weight, reference)
```

**Characteristics:**
- Compression: 10-20% better than COMPRESSED
- Speed: Moderate (component separation + compression)
- Use case: When delta values have low entropy (structured patterns)

**How it works:**
1. Computes arithmetic delta: `delta = weight - reference`
2. Separates into sign (1 bit), exponent (8 bits), mantissa (23 bits)
3. Packs signs into bits (8 per byte)
4. Compresses each stream with tailored compression levels:
   - Signs: High compression (level 9)
   - Exponents: High compression (low entropy)
   - Mantissas: Moderate compression (higher entropy)

### 3.3.2 Lossy Strategies

These strategies sacrifice perfect reconstruction for higher compression ratios.

#### INT8_QUANTIZED

8-bit quantization of delta values.

```python
config = DeltaConfig(
    delta_type=DeltaType.INT8_QUANTIZED,
    quantization_bits=8
)
encoder = DeltaEncoder(config)

delta = encoder.encode_delta(weight, reference)
# Reconstruction has quantization error
reconstructed = encoder.decode_delta(delta, reference)
```

**Characteristics:**
- Compression: ~75% (4 bytes → 1 byte per value)
- Error: Quantization introduces small errors
- Use case: Training checkpoints where exact values aren't critical

**How it works:**
1. Computes delta: `delta = weight - reference`
2. Finds min/max values in delta
3. Quantizes to [-128, 127] range: `quantized = (delta - offset) / scale`
4. Stores int8 values + scale/offset metadata
5. Dequantization: `reconstructed = quantized * scale + offset`

#### INT16_QUANTIZED

16-bit quantization for better fidelity.

```python
config = DeltaConfig(delta_type=DeltaType.INT16_QUANTIZED)
encoder = DeltaEncoder(config)
```

**Characteristics:**
- Compression: ~50% (4 bytes → 2 bytes per value)
- Error: Smaller quantization error than INT8
- Use case: When you need higher precision than INT8 but not full float32

#### SPARSE

Stores only non-zero differences, discarding values below a threshold.

```python
config = DeltaConfig(
    delta_type=DeltaType.SPARSE,
    sparse_threshold=1e-6  # Discard differences < 1e-6
)
encoder = DeltaEncoder(config)

delta = encoder.encode_delta(weight, reference)
```

**Characteristics:**
- Compression: >95% when most differences are near-zero
- Error: Discards small differences (below threshold)
- Use case: Weights with localized changes (e.g., transfer learning on final layers)

**How it works:**
1. Computes delta: `delta = weight - reference`
2. Identifies non-zero elements: `mask = abs(delta) > threshold`
3. Stores only (index, value) pairs for non-zero elements
4. Reconstruction fills zeros for missing elements

#### PER_AXIS_SCALED

1-bit signs with per-axis floating-point scales.

```python
config = DeltaConfig(delta_type=DeltaType.PER_AXIS_SCALED)
encoder = DeltaEncoder(config)

# Best for 2D weight matrices with structured deltas
delta = encoder.encode_delta(weight_2d, reference_2d)
```

**Characteristics:**
- Compression: 20-30% better than quantization for structured deltas
- Error: Loses magnitude information, preserves sign patterns
- Use case: Fine-tuned models where deltas have structured row/column patterns

**How it works:**
1. Computes delta (reshaped to 2D if needed)
2. Extracts per-row and per-column scale factors
3. Stores only sign bits (1 bit per element)
4. Stores row_scales (float16) and col_scales (float16)
5. Reconstruction: `delta ≈ sign * row_scale * col_scale`

### 3.3.3 Strategy Comparison Table

| Strategy | Lossless | Compression | Speed | Best Use Case |
|----------|----------|-------------|-------|---------------|
| FLOAT32_RAW | ✓ | 0-10% | Fastest | Minimal CPU overhead needed |
| COMPRESSED | ✓ | 50-70% | Fast | General purpose, default choice |
| XOR_FLOAT32 | ✓ | 65-85% | Moderate | Very similar weights (fine-tuning) |
| XOR_BFLOAT16 | ✓ | 65-85% | Moderate | BFloat16 training |
| EXPONENT_MANTISSA | ✓ | 60-80% | Moderate | Structured delta patterns |
| INT8_QUANTIZED | ✗ | ~75% | Fast | Training checkpoints |
| INT16_QUANTIZED | ✗ | ~50% | Fast | Higher precision than INT8 |
| SPARSE | ✗ | >95%* | Fast | Localized changes |
| PER_AXIS_SCALED | ✗ | 70-90% | Moderate | Structured fine-tuning |

*For sparse data only

## 3.4 DeltaConfig - Configuration Options

The `DeltaConfig` class controls delta encoding behavior:

```python
from coral.delta import DeltaConfig, DeltaType

config = DeltaConfig(
    # Strategy selection
    delta_type=DeltaType.COMPRESSED,

    # SPARSE strategy threshold
    sparse_threshold=1e-6,  # Discard differences < 1e-6

    # Quantization precision
    quantization_bits=8,  # 8 or 16

    # Compression level for zlib
    compression_level=6,  # 1-9 (higher = better compression, slower)

    # Delta size constraints
    max_delta_ratio=1.0,  # Don't create deltas larger than original
    min_compression_ratio=0.0,  # Minimum compression required
    min_weight_size=512,  # Don't delta-encode small weights (<512 bytes)

    # Reconstruction verification
    strict_reconstruction=False  # Raise error if reference hash mismatches
)
```

### Configuration Parameters Explained

**delta_type**: Selects the encoding strategy (see section 3.3)

**sparse_threshold**: For SPARSE encoding, values with `|delta| < sparse_threshold` are discarded (default: 1e-6)

**quantization_bits**: Bit precision for quantized encoding (8 or 16 bits)

**compression_level**: zlib compression level (1-9):
- 1: Fastest, least compression
- 6: Default, good balance
- 9: Slowest, best compression

**max_delta_ratio**: Maximum allowed delta size as fraction of original size (default: 1.0). If delta would be larger than original, store the weight in full instead.

**min_compression_ratio**: Minimum compression required to use delta encoding (default: 0.0). Set to 0.3 to require at least 30% compression.

**min_weight_size**: Skip delta encoding for weights smaller than this many bytes (default: 512). Overhead makes delta encoding inefficient for small weights.

**strict_reconstruction**: If True, raise `DeltaReconstructionError` if reference hash doesn't match during decoding. Useful for detecting data corruption.

### Serialization

Configurations can be serialized for storage:

```python
# Save configuration
config_dict = config.to_dict()
# Store in repository metadata, config files, etc.

# Restore configuration
restored = DeltaConfig.from_dict(config_dict)
```

## 3.5 Delta Class - The Delta Object

The `Delta` class represents an encoded delta:

```python
from coral.delta import Delta

# Delta objects are created by DeltaEncoder
delta = encoder.encode_delta(weight, reference)

# Attributes
print(f"Delta type: {delta.delta_type}")
print(f"Reference hash: {delta.reference_hash}")
print(f"Compression ratio: {delta.compression_ratio:.1%}")
print(f"Original shape: {delta.original_shape}")
print(f"Original dtype: {delta.original_dtype}")
print(f"Data shape: {delta.data.shape}")
print(f"Metadata: {delta.metadata}")
```

### Delta Attributes

**delta_type**: The encoding strategy used (DeltaType enum)

**data**: Encoded delta data (numpy array, varies by strategy)

**metadata**: Strategy-specific metadata (dict):
- For COMPRESSED: `compression_level`, `original_size`, `compressed_size`
- For INT8_QUANTIZED: `scale`, `offset`, `bits`
- For SPARSE: `num_nonzero`, `sparse_threshold`
- For XOR strategies: component sizes and compression statistics

**original_shape**: Shape of the original weight tensor (for reconstruction)

**original_dtype**: Dtype of the original weight (typically float32)

**reference_hash**: Hash of the reference weight (for retrieval)

**compression_ratio**: Calculated compression ratio (0.0-1.0)

### Size Calculation

The `nbytes` property includes data and metadata overhead:

```python
delta_size = delta.nbytes  # Includes:
# - delta.data.nbytes (encoded data)
# - len(json.dumps(metadata)) (metadata JSON)
# - DELTA_OBJECT_OVERHEAD_BYTES (constant overhead ~200 bytes)
```

### Serialization

Deltas can be serialized for storage:

```python
# Serialize to dictionary
delta_dict = delta.to_dict()

# Deserialize from dictionary
restored_delta = Delta.from_dict(delta_dict)
```

## 3.6 DeltaEncoder - The Encoding Engine

The `DeltaEncoder` class performs encoding and decoding operations.

### Initialization

```python
from coral.delta import DeltaEncoder, DeltaConfig, DeltaType

config = DeltaConfig(delta_type=DeltaType.COMPRESSED)
encoder = DeltaEncoder(config)
```

### Encoding Deltas

```python
from coral.core.weight_tensor import WeightTensor

# Encode weight as delta from reference
delta = encoder.encode_delta(weight, reference)

# Delta object contains all information needed for reconstruction
print(f"Delta size: {delta.nbytes} bytes")
print(f"Original size: {weight.nbytes} bytes")
print(f"Savings: {delta.compression_ratio:.1%}")
```

**Requirements:**
- `weight` and `reference` must have identical shapes and dtypes
- Reference hash is automatically computed and stored

### Decoding Deltas

```python
# Reconstruct original weight from delta
reconstructed = encoder.decode_delta(delta, reference)

# Verify reconstruction (for lossless strategies)
if config.delta_type.is_lossless:
    np.testing.assert_array_equal(reconstructed.data, weight.data)
```

**Process:**
1. Validates reference hash (optional, if strict_reconstruction=True)
2. Decodes delta data according to delta_type
3. Applies delta to reference: `reconstructed = reference + delta`
4. Returns new WeightTensor with reconstructed data

### Feasibility Check

Before encoding, check if delta encoding is worthwhile:

```python
if encoder.can_encode_as_delta(weight, reference):
    delta = encoder.encode_delta(weight, reference)
else:
    # Store weight in full
    pass
```

**Checks performed:**
- Shape and dtype compatibility
- Minimum weight size threshold
- Estimated compression ratio vs. minimum required
- For lossless types: always allowed (enables deduplication)

### Example: Complete Encoding Workflow

```python
import numpy as np
from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.delta import DeltaConfig, DeltaEncoder, DeltaType

# Create weights
reference_data = np.random.randn(1000, 500).astype(np.float32)
weight_data = reference_data + 0.01 * np.random.randn(1000, 500).astype(np.float32)

reference = WeightTensor(
    data=reference_data,
    metadata=WeightMetadata(name="reference", shape=(1000, 500), dtype=np.float32)
)
weight = WeightTensor(
    data=weight_data,
    metadata=WeightMetadata(name="weight", shape=(1000, 500), dtype=np.float32)
)

# Configure and encode
config = DeltaConfig(delta_type=DeltaType.COMPRESSED, compression_level=9)
encoder = DeltaEncoder(config)

# Check feasibility
if encoder.can_encode_as_delta(weight, reference):
    # Encode
    delta = encoder.encode_delta(weight, reference)

    print(f"Original: {weight.nbytes / 1024:.1f} KB")
    print(f"Delta: {delta.nbytes / 1024:.1f} KB")
    print(f"Compression: {delta.compression_ratio:.1%}")

    # Decode
    reconstructed = encoder.decode_delta(delta, reference)

    # Verify
    np.testing.assert_array_almost_equal(
        reconstructed.data, weight.data, decimal=6
    )
    print("Reconstruction verified!")
```

## 3.7 DeltaCompressor - Compression Utilities

The `DeltaCompressor` class provides additional compression utilities and analysis tools.

### Delta Statistics

Analyze delta characteristics to choose optimal compression:

```python
from coral.delta import DeltaCompressor

delta_data = weight.data - reference.data

stats = DeltaCompressor.delta_statistics(delta_data)
print(f"Mean: {stats['mean']:.6f}")
print(f"Std: {stats['std']:.6f}")
print(f"Sparsity: {stats['sparsity']:.1%}")
print(f"Dynamic range: {stats['dynamic_range']:.6f}")
print(f"Entropy: {stats['entropy']:.2f} bits")
```

### Automatic Strategy Recommendation

Get a recommended strategy based on delta characteristics:

```python
recommended = DeltaCompressor.recommend_compression(delta_data)
print(f"Recommended strategy: {recommended}")

# Decision logic:
# - sparsity > 0.7 → "sparse"
# - dynamic_range < 2.0 and std < 0.1 → "int8_quantized"
# - dynamic_range < 100.0 → "int16_quantized"
# - entropy < 5.0 → "compressed"
# - else → "float32_raw"
```

### Adaptive Quantization

Apply distribution-aware quantization:

```python
quantized, metadata = DeltaCompressor.adaptive_quantization(
    delta_data,
    target_bits=8  # 8 or 16
)

# Uses 3-sigma range for quantization (captures 99.7% of data)
# Clips outliers to prevent wasting quantization range

print(f"Outlier ratio: {metadata['outlier_ratio']:.1%}")

# Dequantize
dequantized = DeltaCompressor.dequantize_adaptive(quantized, metadata)
```

## 3.8 Practical Examples

### Example 1: Fine-tuning Scenario (99.9% Similarity)

Fine-tuning a pre-trained model typically changes less than 1% of weight values.

```python
import numpy as np
from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.delta import DeltaConfig, DeltaEncoder, DeltaType

# Simulate fine-tuning: 99.9% of weights identical
pretrained_weights = np.random.randn(2048, 2048).astype(np.float32)

# Fine-tuned: tiny random perturbations
finetuned_weights = pretrained_weights + 0.001 * np.random.randn(2048, 2048).astype(np.float32)

pretrained = WeightTensor(
    data=pretrained_weights,
    metadata=WeightMetadata(name="pretrained", shape=(2048, 2048), dtype=np.float32)
)
finetuned = WeightTensor(
    data=finetuned_weights,
    metadata=WeightMetadata(name="finetuned", shape=(2048, 2048), dtype=np.float32)
)

# Use XOR_FLOAT32 for maximum compression of very similar weights
config = DeltaConfig(delta_type=DeltaType.XOR_FLOAT32)
encoder = DeltaEncoder(config)

delta = encoder.encode_delta(finetuned, pretrained)

print(f"Original model: {pretrained.nbytes / (1024**2):.2f} MB")
print(f"Delta size: {delta.nbytes / (1024**2):.2f} MB")
print(f"Compression: {delta.compression_ratio:.1%}")
print(f"Storage savings: {(1 - delta.nbytes / pretrained.nbytes):.1%}")

# Perfect reconstruction
reconstructed = encoder.decode_delta(delta, pretrained)
np.testing.assert_array_equal(reconstructed.data, finetuned_weights)
print("✓ Lossless reconstruction verified")
```

### Example 2: Training Checkpoints (99% Similarity)

Consecutive training checkpoints are highly similar as gradients make small updates.

```python
# Simulate consecutive training checkpoints
checkpoint_1 = np.random.randn(1024, 1024).astype(np.float32)
checkpoint_2 = checkpoint_1 + 0.01 * np.random.randn(1024, 1024).astype(np.float32)

w1 = WeightTensor(
    data=checkpoint_1,
    metadata=WeightMetadata(name="epoch_10", shape=(1024, 1024), dtype=np.float32)
)
w2 = WeightTensor(
    data=checkpoint_2,
    metadata=WeightMetadata(name="epoch_11", shape=(1024, 1024), dtype=np.float32)
)

# Use COMPRESSED for good balance
config = DeltaConfig(delta_type=DeltaType.COMPRESSED, compression_level=6)
encoder = DeltaEncoder(config)

delta = encoder.encode_delta(w2, w1)

print(f"\nTraining Checkpoints:")
print(f"Epoch 10: {w1.nbytes / (1024**2):.2f} MB")
print(f"Epoch 11 delta: {delta.nbytes / (1024**2):.2f} MB")
print(f"Compression: {delta.compression_ratio:.1%}")

# For multiple checkpoints
num_checkpoints = 10
total_full_size = num_checkpoints * w1.nbytes
total_delta_size = w1.nbytes + (num_checkpoints - 1) * delta.nbytes
total_savings = 1 - (total_delta_size / total_full_size)

print(f"\nFor {num_checkpoints} checkpoints:")
print(f"Without delta: {total_full_size / (1024**2):.2f} MB")
print(f"With delta: {total_delta_size / (1024**2):.2f} MB")
print(f"Total savings: {total_savings:.1%}")
```

### Example 3: Transfer Learning (95% Similarity)

Transfer learning often freezes early layers and fine-tunes later layers, creating sparse differences.

```python
# Simulate transfer learning: most weights identical, some changed significantly
base_model = np.random.randn(512, 512).astype(np.float32)
transfer_model = base_model.copy()

# Change only 5% of weights (final layers)
num_changed = int(0.05 * transfer_model.size)
indices = np.random.choice(transfer_model.size, num_changed, replace=False)
transfer_model.flat[indices] += np.random.randn(num_changed).astype(np.float32)

base = WeightTensor(
    data=base_model,
    metadata=WeightMetadata(name="base", shape=(512, 512), dtype=np.float32)
)
transfer = WeightTensor(
    data=transfer_model,
    metadata=WeightMetadata(name="transfer", shape=(512, 512), dtype=np.float32)
)

# Use SPARSE for localized changes
config = DeltaConfig(delta_type=DeltaType.SPARSE, sparse_threshold=1e-6)
encoder = DeltaEncoder(config)

delta = encoder.encode_delta(transfer, base)

print(f"\nTransfer Learning:")
print(f"Base model: {base.nbytes / (1024**2):.2f} MB")
print(f"Delta size: {delta.nbytes / (1024**2):.2f} MB")
print(f"Compression: {delta.compression_ratio:.1%}")
print(f"Non-zero elements: {delta.metadata['num_nonzero']}")
print(f"Sparsity: {1 - (delta.metadata['num_nonzero'] / base.size):.1%}")

# Approximate reconstruction (lossy due to threshold)
reconstructed = encoder.decode_delta(delta, base)
mse = np.mean((reconstructed.data - transfer_model) ** 2)
print(f"Reconstruction MSE: {mse:.2e}")
```

### Example 4: Repository Integration

Complete workflow using Coral's repository system:

```python
from coral.version_control.repository import Repository
from pathlib import Path
import tempfile

# Create repository with delta encoding enabled
repo_path = Path(tempfile.mkdtemp())
repo = Repository(repo_path, init=True)

# Configure for compressed delta encoding
repo.config["core"]["delta_type"] = "compressed"
repo.config["core"]["enable_delta_encoding"] = True

# Create model snapshots
base_weights = {
    f"layer_{i}.weight": WeightTensor(
        data=np.random.randn(256, 256).astype(np.float32),
        metadata=WeightMetadata(name=f"layer_{i}.weight", shape=(256, 256), dtype=np.float32)
    )
    for i in range(5)
}

# Commit base model
repo.stage_weights(base_weights)
repo.commit("Initial model")

# Create fine-tuned version (similar weights)
finetuned_weights = {
    f"layer_{i}.weight": WeightTensor(
        data=base_weights[f"layer_{i}.weight"].data + 0.01 * np.random.randn(256, 256).astype(np.float32),
        metadata=WeightMetadata(name=f"layer_{i}.weight", shape=(256, 256), dtype=np.float32)
    )
    for i in range(5)
}

# Commit fine-tuned model (will use delta encoding)
repo.stage_weights(finetuned_weights)
repo.commit("Fine-tuned model")

# Retrieve weights (transparent delta reconstruction)
retrieved = repo.get_weight("layer_0.weight")
print(f"Retrieved weight shape: {retrieved.shape}")

# Check deduplication statistics
stats = repo.deduplicator.get_compression_stats()
print(f"\nRepository Statistics:")
print(f"Total weights: {stats['total_weights']}")
print(f"Unique weights: {stats['unique_weights']}")
print(f"Deduplicated: {stats['num_deduplicated']}")
print(f"Delta-encoded: {stats['delta_stats']['total_deltas']}")
print(f"Overall compression: {stats['compression_ratio']:.1%}")
```

## 3.9 Best Practices

### Choosing a Delta Strategy

Use this decision tree:

1. **Do you need perfect reconstruction?**
   - Yes → Use lossless strategy (go to 2)
   - No → Use lossy strategy (go to 5)

2. **Are weights very similar (>99% similarity)?**
   - Yes → Use `XOR_FLOAT32` or `XOR_BFLOAT16`
   - No → Go to 3

3. **Is CPU time critical?**
   - Yes → Use `COMPRESSED` with low compression level (3-4)
   - No → Go to 4

4. **Default choice:**
   - Use `COMPRESSED` with level 6

5. **Are changes sparse (localized to few elements)?**
   - Yes → Use `SPARSE`
   - No → Go to 6

6. **Are deltas structured (per-row/column patterns)?**
   - Yes → Use `PER_AXIS_SCALED`
   - No → Use `INT8_QUANTIZED` or `INT16_QUANTIZED`

### Configuration Tips

- Set `min_weight_size=1024` to avoid delta overhead on small weights
- Use `strict_reconstruction=True` in production to detect corruption
- Adjust `compression_level` based on CPU vs storage trade-off
- Monitor `compression_ratio` and adjust strategy if needed

### Performance Optimization

- Batch similar operations together
- Reuse encoder instances (avoid re-creating)
- Use lossless strategies to enable similarity-based deduplication
- Profile your specific workload to find optimal settings

## 3.10 Summary

Delta encoding is a cornerstone of Coral's efficient storage system, providing:

- **Lossless storage** of similar weights with 50-85% compression
- **Flexible strategies** balancing compression, speed, and fidelity
- **Transparent integration** with repository and deduplication systems
- **Practical compression** for real-world ML workflows (fine-tuning, checkpoints, ensembles)

By storing differences instead of duplicating similar data, delta encoding enables git-like version control for neural networks without the storage explosion that would make it impractical. Combined with content-addressable storage and deduplication, Coral achieves 50-90% space savings on typical ML workflows while maintaining perfect reconstruction fidelity when needed.
