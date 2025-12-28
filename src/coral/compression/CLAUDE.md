# Compression Module

> **EXPERIMENTAL**: This module is experimental and not yet fully integrated into the core Coral workflow. The API may change in future versions.

This module provides weight compression techniques including quantization and pruning for reducing model size.

## Overview

The compression module provides:
- **Quantization**: Reduce precision of weights (8-bit, 4-bit, 2-bit)
- **Pruning**: Remove unimportant weights (magnitude-based, structured)

## Key Files

### `quantization.py`

Weight quantization methods for compression.

**Quantizer** (class):

Static methods for quantization:
```python
# Uniform quantization (most common)
quantized, params = Quantizer.quantize_uniform(
    weight,
    bits=8,          # 2, 4, or 8
    symmetric=True   # Symmetric or asymmetric
)

# Dequantize to float32
dequantized = Quantizer.dequantize(quantized, params)

# Per-channel quantization (better accuracy)
quantized, params = Quantizer.quantize_per_channel(
    weight,
    bits=8,
    axis=0    # Channel axis
)

# Estimate quantization error
mse = Quantizer.estimate_quantization_error(weight, bits=8)
```

**Quantization Parameters**:
- `scale`: Float scaling factor
- `zero_point`: Zero point offset
- `bits`: Bit width used
- `symmetric`: Whether symmetric quantization was used

**Symmetric vs Asymmetric**:
- **Symmetric**: Range centered at 0, zero_point=0, for signed weights
- **Asymmetric**: Range from min to max, uses full bit range, for positive activations

### `pruning.py`

Weight pruning methods for sparsity.

**Pruner** (class):

Static methods for pruning:
```python
# Magnitude-based pruning
pruned, mask = Pruner.prune_magnitude(
    weight,
    sparsity=0.5,    # Remove 50% of weights
    threshold=None   # Or use absolute threshold
)

# Structured pruning (entire channels)
pruned, mask = Pruner.prune_structured(
    weight,
    sparsity=0.3,
    axis=0           # Prune along this axis
)

# Get sparsity statistics
stats = Pruner.get_sparsity_stats(weight)
# Returns: {'sparsity': 0.5, 'zero_count': 1000, 'total': 2000}
```

## Usage Examples

### 8-bit Quantization

```python
from coral.compression.quantization import Quantizer

# Quantize weights
quantized, params = Quantizer.quantize_uniform(weight, bits=8)

# Check compression ratio
ratio = weight.nbytes / quantized.nbytes  # 4x for float32 -> int8
print(f"Compression: {ratio:.1f}x")

# Dequantize when needed
recovered = Quantizer.dequantize(quantized, params)
```

### Low-bit Quantization

```python
# 4-bit quantization (16x compression vs float32)
quantized_4bit, params = Quantizer.quantize_uniform(weight, bits=4)

# 2-bit quantization (aggressive, for experiments)
quantized_2bit, params = Quantizer.quantize_uniform(weight, bits=2)

# Check error
mse_4bit = Quantizer.estimate_quantization_error(weight, bits=4)
mse_2bit = Quantizer.estimate_quantization_error(weight, bits=2)
```

### Per-Channel Quantization

```python
# Better accuracy for conv/linear layers
quantized, params = Quantizer.quantize_per_channel(
    weight,
    bits=8,
    axis=0  # Output channels
)

# Scales and zero points per channel
print(f"Channel scales: {params['scales'].shape}")
```

### Magnitude Pruning

```python
from coral.compression.pruning import Pruner

# Prune 50% of smallest weights
pruned, mask = Pruner.prune_magnitude(weight, sparsity=0.5)

# Check actual sparsity
stats = Pruner.get_sparsity_stats(pruned)
print(f"Sparsity: {stats['sparsity']:.1%}")
```

## Compression Ratios

| Method | Compression | Notes |
|--------|-------------|-------|
| INT8 Uniform | 4x | Minimal accuracy loss |
| INT4 Uniform | 8x | Moderate accuracy loss |
| INT2 Uniform | 16x | Significant accuracy loss |
| 50% Pruning | ~2x (with sparse storage) | Requires sparse format |
| 90% Pruning | ~10x (with sparse storage) | May need fine-tuning |

## Integration with Delta Encoding

Quantization works well with delta encoding:

```python
from coral.delta import DeltaEncoder, DeltaConfig, DeltaType

# INT8 quantized delta encoding
config = DeltaConfig(delta_type=DeltaType.INT8_QUANTIZED)
encoder = DeltaEncoder(config)

# Encode difference between quantized weights
delta = encoder.encode_delta(current_weight, reference_weight)
# Achieves both quantization compression and delta efficiency
```

## Limitations

1. **Lossy Compression**: Both quantization and pruning reduce accuracy
2. **Not Production-Ready**: API may change
3. **No Automatic Integration**: Must be applied manually
4. **Quantization Error**: Accumulates across layers

## Future Plans

- Integration with training workflow
- Quantization-aware training support
- Mixed-precision quantization
- Neural architecture search for optimal compression

## Dependencies

- `numpy` - Array operations
- Internal: `coral.core.weight_tensor`

## Testing

Related test files:
- `tests/test_compression.py` - Quantization and pruning tests
