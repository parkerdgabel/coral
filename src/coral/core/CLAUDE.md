# Core Module

This module contains the fundamental data structures and algorithms for neural network weight management and deduplication.

## Overview

The core module provides:
- **WeightTensor**: The fundamental data structure for storing neural network weights with metadata
- **Deduplicator**: Engine for identifying and eliminating duplicate/similar weights
- **SimHash**: Locality-sensitive hashing for O(1) similarity detection
- **LSH Index**: Multi-table locality-sensitive hashing for efficient similarity search

## Key Files

### `weight_tensor.py`

Defines `WeightTensor` and `WeightMetadata` classes.

**WeightMetadata** (dataclass):
- `name`: Weight name (e.g., "layer1.weight")
- `shape`: Tensor shape as tuple
- `dtype`: NumPy dtype
- `layer_type`: Optional layer type (e.g., "Linear", "Conv2d")
- `model_name`: Optional model identifier
- `compression_info`: Dictionary with compression metadata
- `hash`: Content hash (computed lazily)

**WeightTensor** (class):
- Wraps numpy array with metadata
- Supports lazy loading via `store_ref`
- Content-based hashing using xxHash (xxh3_64)
- Similarity comparison via cosine similarity
- Serialization to/from dictionary

**Key Methods**:
```python
weight.compute_hash()       # Compute xxHash of content
weight.is_similar_to(other, threshold=0.99)  # Cosine similarity check
weight.to_dict()            # Serialize for storage
WeightTensor.from_dict(d)   # Deserialize
```

### `deduplicator.py`

Core deduplication engine with support for exact and similarity-based deduplication.

**DeduplicationStats** (dataclass):
- Tracks total/unique/duplicate/similar weights
- Calculates bytes saved and compression ratio

**WeightGroup** (dataclass):
- Groups identical or similar weights
- Stores reference weight, duplicates list, and delta encodings
- Calculates bytes saved per group

**Deduplicator** (class):
- Thread-safe (uses `threading.RLock`)
- Configurable similarity threshold (default: 0.99)
- Optional LSH for O(1) lookups (enable for >10k weights)
- Delta encoding for similar weights

**Key Methods**:
```python
dedup.add_weight(weight, name)      # Returns hash (reference if duplicate)
dedup.get_weight_by_name(name)      # Reconstructs from delta if needed
dedup.compute_stats()               # Returns DeduplicationStats
dedup.get_deduplication_report()    # Detailed report with largest groups
dedup.is_delta_encoded(name)        # Check if weight uses delta encoding
```

**Constructor Parameters**:
- `similarity_threshold`: Float 0-1 (default: 0.99)
- `delta_config`: Optional DeltaConfig for encoding settings
- `enable_delta_encoding`: Whether to use delta encoding (default: True)
- `enable_lsh`: Enable LSH index for large-scale deployments (default: False)
- `lsh_config`: Optional LSHConfig
- `magnitude_tolerance`: Maximum relative magnitude difference (default: 0.1)

### `simhash.py`

SimHash fingerprinting for O(1) similarity detection using Hamming distance.

**SimHashConfig** (dataclass):
- `num_bits`: Fingerprint size, 64 or 128 (default: 64)
- `num_hyperplanes`: Number of random hyperplanes (default: same as num_bits)
- `seed`: Random seed for reproducibility (default: 42)
- `similarity_threshold`: Hamming distance threshold as fraction (default: 0.1)

**SimHash** (class):
- Projects vectors onto random hyperplanes
- Creates binary signatures based on projection signs
- Supports batch processing

**Key Methods**:
```python
simhash.compute_fingerprint(vector)     # Returns uint64 fingerprint
simhash.hamming_distance(fp1, fp2)      # Static method
simhash.are_similar(fp1, fp2)           # Check if within threshold
simhash.estimated_similarity(fp1, fp2)  # Estimate cosine similarity
```

**SimHashIndex** (class):
- Stores fingerprints for fast lookup
- O(1) average-case similarity search

**MultiDimSimHashIndex** (class):
- Handles vectors of different dimensions
- Maintains separate SimHash indices per dimension

### `lsh_index.py`

Locality Sensitive Hashing for efficient similarity search.

**Algorithm**:
1. Generate k random hyperplanes (normal vectors)
2. Compute which side of each hyperplane each vector falls on
3. Create k-bit hash where similar vectors tend to share the same hash
4. Store vectors in hash buckets
5. Query returns only vectors in matching bucket(s)

**Trade-offs**:
- More hyperplanes (k) = fewer false positives, more false negatives
- More hash tables (L) = fewer false negatives, more memory
- Default (k=8, L=4) works well for similarity threshold >= 0.9

**LSHConfig** (dataclass):
- `num_hyperplanes`: Bits per hash (default: 8)
- `num_tables`: Number of hash tables (default: 4)
- `seed`: Random seed (default: 42)
- `max_candidates`: Max candidates to return (default: 100)

**LSHIndex** (class):
- Fixed-dimension index
- Cosine similarity via hyperplane hashing

**MultiDimLSHIndex** (class):
- Handles vectors of different dimensions
- Maintains separate LSH indices per dimension
- Used by Deduplicator when `enable_lsh=True`

## Design Patterns

1. **Content-Addressable Storage**: Weights identified by xxHash content hash
2. **Lazy Loading**: WeightTensor supports `store_ref` for deferred data loading
3. **Thread Safety**: Deduplicator uses RLock for concurrent access
4. **Pluggable Similarity**: LSH and SimHash are optional optimizations

## Dependencies

- `numpy` - Array operations
- `xxhash` - Fast content hashing
- Internal: `coral.delta.delta_encoder`, `coral.utils.similarity`

## Usage Example

```python
from coral.core import WeightTensor, Deduplicator
from coral.core.weight_tensor import WeightMetadata
import numpy as np

# Create weight tensor
weight = WeightTensor(
    data=np.random.randn(256, 128).astype(np.float32),
    metadata=WeightMetadata(name="layer1.weight", shape=(256, 128), dtype=np.float32)
)

# Use deduplicator
dedup = Deduplicator(similarity_threshold=0.98, enable_lsh=True)
hash1 = dedup.add_weight(weight, "layer1.weight")

# Similar weight will be delta-encoded
similar_weight = WeightTensor(
    data=weight.data + np.random.randn(256, 128).astype(np.float32) * 0.01,
    metadata=WeightMetadata(name="layer1.weight.v2", shape=(256, 128), dtype=np.float32)
)
hash2 = dedup.add_weight(similar_weight, "layer1.weight.v2")

# Get stats
stats = dedup.compute_stats()
print(f"Compression ratio: {stats.compression_ratio:.2%}")
```

## Testing

Related test files:
- `tests/test_weight_tensor.py` - WeightTensor functionality
- `tests/test_deduplicator.py` - Deduplication logic
- `tests/test_simhash.py` - SimHash fingerprinting
