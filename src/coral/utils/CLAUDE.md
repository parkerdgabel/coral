# Utils Module

This module provides utility functions for JSON serialization, similarity computation, and visualization.

## Overview

The utils module provides:
- **JSON Utilities**: NumPy-aware JSON encoding
- **Similarity Functions**: Various similarity metrics for weight comparison
- **Visualization**: Data preparation for weight analysis and model comparison

## Key Files

### `json_utils.py`

JSON serialization utilities that handle NumPy types.

**NumpyJSONEncoder** (class):
```python
from coral.utils import NumpyJSONEncoder
import json

# Use as custom encoder
data = {"weights": np.array([1, 2, 3]), "dtype": np.float32}
json_str = json.dumps(data, cls=NumpyJSONEncoder)
```

**Convenience Functions**:
```python
from coral.utils import dumps_numpy, dump_numpy

# Serialize to string
json_str = dumps_numpy({"array": np.array([1, 2, 3])})

# Serialize to file
with open("data.json", "w") as f:
    dump_numpy({"array": np.array([1, 2, 3])}, f)
```

**Type Conversions**:
| NumPy Type | Python Type |
|------------|-------------|
| `np.integer` | `int` |
| `np.floating` | `float` |
| `np.ndarray` | `list` |
| `np.dtype` | `str` |
| `np.bool_` | `bool` |
| `np.void`, `np.complexfloating` | `str` |

### `similarity.py`

Similarity computation utilities for weight tensors.

**cosine_similarity(a, b)**:
```python
from coral.utils import cosine_similarity

# Compute cosine similarity (-1 to 1)
sim = cosine_similarity(weight_a, weight_b)

# Note: Scale-invariant! [1,2,3] and [100,200,300] have sim=1.0
# For weights where magnitude matters, use weight_similarity() instead
```

**magnitude_similarity(a, b)**:
```python
from coral.utils import magnitude_similarity

# Ratio of smaller to larger norm (0 to 1)
sim = magnitude_similarity(weight_a, weight_b)
# Returns 1.0 for equal magnitudes, approaches 0 as magnitudes diverge
```

**weight_similarity(a, b, direction_weight=0.7, magnitude_weight=0.3)**:
```python
from coral.utils import weight_similarity

# Hybrid metric considering both direction and magnitude
sim = weight_similarity(weight_a, weight_b)

# Same direction but 100x scale difference -> ~0.71 similarity
# Small perturbation -> ~0.999 similarity

# Customize weights
sim = weight_similarity(a, b, direction_weight=0.5, magnitude_weight=0.5)
```

**relative_difference(a, b)**:
```python
from coral.utils import relative_difference

# Get element-wise difference statistics
mean_rel, max_rel, rmse = relative_difference(weight_a, weight_b)
# Relative diff computed as |a-b| / (|a| + |b| + eps)
```

**are_similar(a, b, threshold=0.99, check_magnitude=True, magnitude_tolerance=0.1)**:
```python
from coral.utils import are_similar

# Check if weights are similar enough for deduplication
if are_similar(weight_a, weight_b, threshold=0.99):
    print("Weights can be deduplicated")

# Disable magnitude check for direction-only comparison
if are_similar(weight_a, weight_b, check_magnitude=False):
    print("Same direction")
```

### `visualization.py`

Visualization data preparation utilities.

**plot_weight_distribution(weights, bins=50)**:
```python
from coral.utils import plot_weight_distribution

# Get histogram data for weight distributions
distributions = plot_weight_distribution(weight_list)

for name, data in distributions.items():
    print(f"{name}:")
    print(f"  Mean: {data['mean']:.4f}")
    print(f"  Std: {data['std']:.4f}")
    print(f"  Min: {data['min']:.4f}, Max: {data['max']:.4f}")
    print(f"  Sparsity: {data['sparsity']:.2%}")
    # data['histogram'] and data['bin_edges'] for plotting
```

**plot_deduplication_stats(stats)**:
```python
from coral.utils import plot_deduplication_stats

# Prepare deduplication stats for visualization
viz_data = plot_deduplication_stats(dedup_stats)

print(f"Unique: {viz_data['weight_counts']['unique']}")
print(f"Compression ratio: {viz_data['compression']['compression_ratio']:.2f}x")
# viz_data['pie_data'] for pie charts
```

**compare_models(weights_a, weights_b, model_a_name, model_b_name)**:
```python
from coral.utils import compare_models

# Compare two model weight sets
diff = compare_models(
    base_weights,
    finetuned_weights,
    "Base Model",
    "Fine-tuned"
)

# Summary
print(f"Overall similarity: {diff['summary']['overall_similarity']:.1%}")
print(f"Changed params: {diff['summary']['changed_params']:,}")

# Categories
print(f"Identical layers: {diff['category_counts']['identical']}")
print(f"Major changes: {diff['category_counts']['major_changes']}")

# Most changed layers
for layer in diff['most_changed_layers'][:5]:
    print(f"  {layer['name']}: {layer['combined_similarity']:.2%}")
```

**format_model_diff(diff_data, verbose=False)**:
```python
from coral.utils import format_model_diff

# Format comparison as human-readable string
diff = compare_models(base_weights, finetuned_weights)
report = format_model_diff(diff, verbose=True)
print(report)
```

## Usage Examples

### JSON Serialization with NumPy

```python
from coral.utils import dumps_numpy
import numpy as np

# Serialize complex data with NumPy types
data = {
    "weights": np.random.randn(10, 10).astype(np.float32),
    "shape": (10, 10),
    "dtype": np.float32,
    "metrics": {
        "accuracy": np.float64(0.95),
        "steps": np.int64(1000),
    }
}

json_str = dumps_numpy(data, indent=2)
```

### Weight Comparison Pipeline

```python
from coral.utils import (
    cosine_similarity,
    weight_similarity,
    are_similar,
    relative_difference
)

# Quick similarity check
if are_similar(old_weight, new_weight, threshold=0.99):
    print("Weights are similar - can use delta encoding")
else:
    print("Weights differ significantly")

# Detailed comparison
cos_sim = cosine_similarity(old_weight, new_weight)
combined_sim = weight_similarity(old_weight, new_weight)
mean_rel, max_rel, rmse = relative_difference(old_weight, new_weight)

print(f"Direction similarity: {cos_sim:.4f}")
print(f"Combined similarity: {combined_sim:.4f}")
print(f"RMSE: {rmse:.6f}")
```

### Model Analysis

```python
from coral.utils import compare_models, format_model_diff

# Load models (as dict[str, WeightTensor])
base_model = load_weights("base_model.pt")
finetuned_model = load_weights("finetuned_model.pt")

# Compare
diff = compare_models(
    base_model,
    finetuned_model,
    "BERT Base",
    "BERT Fine-tuned"
)

# Print formatted report
print(format_model_diff(diff))

# Programmatic analysis
if diff['summary']['overall_similarity'] > 0.95:
    print("Models are very similar - good candidate for delta storage")

# Find layers with major changes
major_changes = diff['categories']['major_changes']
print(f"Layers with major changes: {major_changes}")
```

## Dependencies

- `numpy` - Array operations
- Internal: `coral.core.deduplicator`, `coral.core.weight_tensor`

## Testing

Related test files:
- `tests/test_visualization.py` - Visualization utility tests
