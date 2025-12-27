# Chapter 2: Core Architecture

## Overview

The Coral system is built on a foundation of carefully designed core components that work together to provide efficient, lossless neural network weight versioning. At the heart of the system are two fundamental classes: **WeightTensor**, which represents individual weight tensors with metadata and content-based addressing, and **Deduplicator**, which intelligently eliminates redundant storage through exact and similarity-based deduplication with lossless delta encoding.

This chapter explores these core components in depth, examining their design philosophy, implementation details, and the sophisticated algorithms that enable Coral to achieve 40-50% space savings while maintaining perfect fidelity.

```
┌─────────────────────────────────────────────────────────────┐
│                    Coral Core Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌─────────────────┐               │
│  │ WeightTensor │────────▶│  Deduplicator   │               │
│  └──────────────┘         └─────────────────┘               │
│       │                           │                          │
│       │ Content Hash              │ Delta Encoding           │
│       │ (xxHash64)                │ (Lossless)               │
│       │                           │                          │
│       ▼                           ▼                          │
│  ┌──────────────┐         ┌─────────────────┐               │
│  │   Metadata   │         │  Similarity     │               │
│  │   (Shape,    │         │  Detection      │               │
│  │   dtype,     │         │  - Cosine       │               │
│  │   hash)      │         │  - Magnitude    │               │
│  └──────────────┘         │  - SimHash      │               │
│                           │  - LSH Index    │               │
│                           └─────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

## 1. WeightTensor: The Fundamental Data Structure

### 1.1 Purpose and Design Philosophy

`WeightTensor` is the atomic unit of the Coral system. It encapsulates a neural network weight tensor along with rich metadata, providing a unified interface for weight management regardless of the underlying ML framework (PyTorch, TensorFlow, JAX, etc.).

The design philosophy centers on three key principles:

1. **Content-Addressable**: Weights are identified by their content hash, enabling automatic deduplication
2. **Metadata-Rich**: Every weight carries information about its origin, shape, type, and compression
3. **Lazy Loading**: Weight data can be loaded on-demand from storage, reducing memory footprint

### 1.2 Core Structure

The `WeightTensor` class is defined in `/home/user/coral/src/coral/core/weight_tensor.py`:

```python
class WeightTensor:
    """
    Base class for representing neural network weights with deduplication support.

    Supports:
    - Content-based hashing for deduplication
    - Metadata tracking
    - Lazy loading from storage
    - Compression support
    """

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        metadata: Optional[WeightMetadata] = None,
        store_ref: Optional[str] = None,
    ):
        self._data = data
        self._metadata = metadata
        self._store_ref = store_ref  # Reference for lazy loading
        self._hash: Optional[str] = None
```

### 1.3 WeightMetadata: Structured Metadata

Metadata is separated into a dedicated dataclass for clarity and efficiency:

```python
@dataclass
class WeightMetadata:
    """Metadata associated with a weight tensor"""

    name: str                                    # Layer/parameter name
    shape: Tuple[int, ...]                       # Tensor dimensions
    dtype: np.dtype                              # Data type (float32, etc.)
    layer_type: Optional[str] = None             # Conv2D, Linear, etc.
    model_name: Optional[str] = None             # Parent model identifier
    compression_info: Dict[str, Any] = field(default_factory=dict)
    hash: Optional[str] = None                   # Content hash (cached)
```

This separation allows metadata to be stored and transmitted independently from the potentially large weight data, enabling efficient indexing and queries.

### 1.4 Content-Based Hashing with xxHash64

Coral uses **xxHash64**, a non-cryptographic hash function optimized for speed, to compute content-based identifiers for weights. This is critical for deduplication performance.

```python
def compute_hash(self, force: bool = False) -> str:
    """
    Compute content-based hash of the weight tensor.

    Returns:
        Hexadecimal hash string
    """
    if self._hash is not None and not force:
        return self._hash  # Return cached hash

    # Use xxhash for fast hashing
    hasher = xxhash.xxh3_64()

    # Include shape and dtype in hash to distinguish identical data
    # with different interpretations
    normalized_shape = tuple(int(dim) for dim in self.shape)
    normalized_dtype = np.dtype(self.dtype).name
    hasher.update(str(normalized_shape).encode())
    hasher.update(normalized_dtype.encode())

    # Hash the actual data
    hasher.update(self.data.tobytes())

    self._hash = hasher.hexdigest()
    if self._metadata:
        self._metadata.hash = self._hash

    return self._hash
```

**Why Include Shape and Dtype?**

Two tensors with identical binary data but different shapes/dtypes represent different weights. For example:
- `[1, 2, 3, 4]` as float32 with shape `(4,)`
- `[1, 2, 3, 4]` as float32 with shape `(2, 2)`

These are semantically different and must have different hashes.

**Performance Characteristics:**
- xxHash64 is **10x faster** than MD5 for large tensors
- Hashing is cached and only recomputed when data changes
- Typical hash time: ~1-2ms for a 100MB weight tensor

### 1.5 Lazy Loading

WeightTensor supports lazy loading to minimize memory usage:

```python
@property
def data(self) -> np.ndarray:
    """Get the weight data, loading from storage if necessary"""
    if self._data is None:
        raise ValueError(
            "Weight data not loaded and no store reference available"
        )
    return self._data
```

When a `WeightTensor` is loaded from storage, it can initially contain only metadata and a `store_ref`. The actual data is loaded on first access. This is crucial for:

- **Memory Efficiency**: Load only the weights you need
- **Fast Metadata Queries**: Scan commit history without loading all weights
- **Streaming Operations**: Process large model collections without exhausting RAM

### 1.6 Similarity Detection

`WeightTensor` provides a high-level similarity check that integrates with the similarity utilities:

```python
def is_similar_to(self, other: "WeightTensor", threshold: float = 0.99) -> bool:
    """
    Check if this weight tensor is similar to another.

    Uses cosine similarity for comparison.
    """
    if self.shape != other.shape or self.dtype != other.dtype:
        return False

    from coral.utils.similarity import cosine_similarity

    similarity = cosine_similarity(self.data, other.data)
    return similarity >= threshold
```

This method performs early-exit checks (shape, dtype) before expensive similarity computation.

### 1.7 Serialization

Efficient serialization enables storage and transmission:

```python
def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary for serialization"""
    return {
        "metadata": {
            "name": self.metadata.name,
            "shape": list(self.metadata.shape),
            "dtype": np.dtype(self.metadata.dtype).name,
            "layer_type": self.metadata.layer_type,
            "model_name": self.metadata.model_name,
            "compression_info": self.metadata.compression_info,
            "hash": self.compute_hash(),
        },
        "store_ref": self._store_ref,
        "has_data": self._data is not None,
    }

@classmethod
def from_dict(
    cls,
    data: Dict[str, Any],
    weight_data: Optional[np.ndarray] = None
) -> "WeightTensor":
    """Create WeightTensor from dictionary"""
    metadata = WeightMetadata(
        name=data["metadata"]["name"],
        shape=tuple(data["metadata"]["shape"]),
        dtype=np.dtype(data["metadata"]["dtype"]),
        layer_type=data["metadata"].get("layer_type"),
        model_name=data["metadata"].get("model_name"),
        compression_info=data["metadata"].get("compression_info", {}),
        hash=data["metadata"].get("hash"),
    )

    return cls(
        data=weight_data,
        metadata=metadata,
        store_ref=data.get("store_ref")
    )
```

### 1.8 Usage Examples

**Creating a WeightTensor:**

```python
import numpy as np
from coral.core.weight_tensor import WeightTensor, WeightMetadata

# From PyTorch
pytorch_weight = model.conv1.weight.detach().cpu().numpy()
metadata = WeightMetadata(
    name="conv1.weight",
    shape=pytorch_weight.shape,
    dtype=pytorch_weight.dtype,
    layer_type="Conv2d",
    model_name="ResNet50"
)
weight_tensor = WeightTensor(data=pytorch_weight, metadata=metadata)

# Compute content hash
hash_value = weight_tensor.compute_hash()
print(f"Weight hash: {hash_value}")

# Check similarity
other_weight = WeightTensor(...)
if weight_tensor.is_similar_to(other_weight, threshold=0.99):
    print("Weights are 99% similar!")
```

**Lazy Loading Pattern:**

```python
# Create with store reference only (no data)
lazy_weight = WeightTensor(
    data=None,
    metadata=metadata,
    store_ref="hdf5://weights.h5/conv1.weight"
)

# Data loaded on first access
# (In practice, the storage backend handles this)
weight_data = lazy_weight.data  # Triggers load from storage
```

## 2. Deduplicator: The Deduplication Engine

### 2.1 Purpose and Architecture

The `Deduplicator` is Coral's intelligent storage optimization engine. It identifies and eliminates redundant weight storage through two complementary strategies:

1. **Exact Deduplication**: Identical weights (same hash) share a single copy
2. **Similarity-Based Deduplication**: Near-identical weights use **lossless delta encoding**

```
┌───────────────────────────────────────────────────────────┐
│                    Deduplicator Flow                       │
└───────────────────────────────────────────────────────────┘

    Input Weight
         │
         ▼
    Compute Hash
         │
         ├──────────────┐
         │              │
    [Hash Exists?]      │
         │ YES          │ NO
         │              │
         ▼              ▼
    Exact Duplicate    Check Similarity
         │              │
         │              ├─────────────┐
         │              │             │
         │         [Similar?]         │
         │              │ YES         │ NO
         │              │             │
         │              ▼             ▼
         │         Create Delta    New Unique Weight
         │              │             │
         └──────────────┴─────────────┘
                        │
                        ▼
              Update Statistics
              Store Reference
```

### 2.2 Core Data Structures

#### WeightGroup: Organizing Related Weights

```python
@dataclass
class WeightGroup:
    """Group of weights that are identical or similar"""

    reference_hash: str                          # Hash of reference weight
    reference_weight: WeightTensor               # The canonical copy
    duplicates: List[Tuple[str, WeightTensor]]  # Exact duplicates
    similar: List[Tuple[str, WeightTensor, float]]  # (name, weight, similarity)
    deltas: Dict[str, Delta]                     # name -> delta for similar weights

    @property
    def total_count(self) -> int:
        """Total number of weights in this group"""
        return 1 + len(self.duplicates) + len(self.similar)

    @property
    def bytes_saved(self) -> int:
        """Bytes saved by deduplication in this group"""
        ref_bytes = self.reference_weight.nbytes
        # Exact duplicates save full size
        duplicate_savings = ref_bytes * len(self.duplicates)
        # Similar weights save (original_size - delta_size)
        similar_savings = sum(
            weight.nbytes - self.deltas[name].nbytes
            for name, weight, _ in self.similar
            if name in self.deltas
        )
        return duplicate_savings + similar_savings
```

#### DeduplicationStats: Tracking Performance

```python
@dataclass
class DeduplicationStats:
    """Statistics about deduplication results"""

    total_weights: int = 0
    unique_weights: int = 0
    duplicate_weights: int = 0
    similar_weights: int = 0
    bytes_saved: int = 0
    compression_ratio: float = 0.0

    def update(self, original_bytes: int, deduplicated_bytes: int):
        """Update compression statistics"""
        self.bytes_saved = original_bytes - deduplicated_bytes
        if original_bytes > 0:
            self.compression_ratio = self.bytes_saved / original_bytes
```

### 2.3 Deduplicator Initialization

```python
class Deduplicator:
    def __init__(
        self,
        similarity_threshold: float = 0.99,
        delta_config: Optional[DeltaConfig] = None,
        enable_delta_encoding: bool = True,
        enable_lsh: bool = False,
        lsh_config: Optional[LSHConfig] = None,
        magnitude_tolerance: float = 0.1,
    ):
        self.similarity_threshold = similarity_threshold
        self.magnitude_tolerance = magnitude_tolerance
        self.enable_delta_encoding = enable_delta_encoding

        # Delta encoder for lossless compression of similar weights
        self.delta_encoder = (
            DeltaEncoder(delta_config or DeltaConfig())
            if enable_delta_encoding
            else None
        )

        # LSH index for O(1) similarity lookup (optional)
        self.enable_lsh = enable_lsh
        self.lsh_index = None
        if enable_lsh:
            self.lsh_index = MultiDimLSHIndex(lsh_config or LSHConfig())

        # Storage indices
        self.weight_index: Dict[str, WeightTensor] = {}  # hash -> weight
        self.weight_groups: Dict[str, WeightGroup] = {}  # ref_hash -> group
        self.name_to_hash: Dict[str, str] = {}           # name -> hash
        self.delta_index: Dict[str, Delta] = {}          # delta_hash -> delta

        # Thread safety
        self._lock = threading.RLock()
```

**Key Configuration Options:**

- **similarity_threshold**: Cosine similarity threshold for considering weights similar (default: 0.99)
- **enable_delta_encoding**: Enable lossless delta encoding for similar weights (default: True)
- **enable_lsh**: Use LSH for O(1) similarity search instead of O(n) scan (default: False, enable for >10k weights)
- **magnitude_tolerance**: Maximum relative magnitude difference for similarity (default: 0.1 = 10%)

### 2.4 Adding Weights: The Deduplication Pipeline

```python
def add_weight(self, weight: WeightTensor, name: Optional[str] = None) -> str:
    """
    Add a weight to the deduplicator and check for duplicates.

    Thread-safe method that handles exact and similarity-based deduplication.

    Returns:
        Hash of the weight (or reference weight if duplicate/similar)
    """
    if name is None:
        name = weight.metadata.name

    # Compute hash outside lock (CPU-intensive, no synchronization needed)
    weight_hash = weight.compute_hash()

    with self._lock:
        # 1. Check for exact duplicate
        if weight_hash in self.weight_index:
            self._add_duplicate(weight_hash, name, weight)
            return weight_hash

        # 2. Check for similar weights
        similar_ref = self._find_similar_weight(weight)
        if similar_ref:
            self._add_similar(similar_ref, name, weight)
            return similar_ref

        # 3. New unique weight
        self._add_unique_weight(weight_hash, name, weight)
        return weight_hash
```

**Performance Note:** Computing the hash outside the lock allows multiple threads to hash weights in parallel, maximizing CPU utilization.

### 2.5 Exact Deduplication

```python
def _add_duplicate(self, ref_hash: str, name: str, weight: WeightTensor):
    """Add an exact duplicate to existing group"""
    if ref_hash not in self.weight_groups:
        # Create group if it doesn't exist
        self.weight_groups[ref_hash] = WeightGroup(
            reference_hash=ref_hash,
            reference_weight=self.weight_index[ref_hash]
        )

    self.weight_groups[ref_hash].duplicates.append((name, weight))
    self.name_to_hash[name] = ref_hash
    self.stats.duplicate_weights += 1
    logger.debug(f"Found exact duplicate: {name} -> {ref_hash}")
```

Exact duplicates are trivial: just store a reference to the canonical copy. This is common in scenarios like:
- Multiple training checkpoints with identical layers (e.g., frozen layers)
- Model ensembles with shared components
- Checkpoint snapshots taken without weight changes

### 2.6 Similarity-Based Deduplication with Delta Encoding

This is where Coral's innovation shines. Similar weights are stored as **deltas** from a reference weight, enabling perfect reconstruction while saving 50-95% space.

```python
def _add_similar(self, ref_hash: str, name: str, weight: WeightTensor):
    """Add a similar weight to existing group"""
    ref_weight = self.weight_index[ref_hash]
    similarity = self._compute_similarity(weight, ref_weight)

    if ref_hash not in self.weight_groups:
        self.weight_groups[ref_hash] = WeightGroup(
            reference_hash=ref_hash,
            reference_weight=ref_weight
        )

    group = self.weight_groups[ref_hash]
    group.similar.append((name, weight, similarity))

    # Create delta encoding if enabled
    if self.enable_delta_encoding and self.delta_encoder:
        try:
            if self.delta_encoder.can_encode_as_delta(weight, ref_weight):
                # Encode as delta: stores difference, not full weight
                delta = self.delta_encoder.encode_delta(weight, ref_weight)
                delta_hash = self._compute_delta_hash(delta)

                # Store delta
                self.delta_index[delta_hash] = delta
                group.deltas[name] = delta
                self.name_to_delta[name] = delta_hash

                logger.debug(
                    f"Created delta for {name}: "
                    f"{delta.compression_ratio:.2%} compression"
                )
        except (ValueError, TypeError, np.linalg.LinAlgError) as e:
            logger.warning(f"Failed to create delta for {name}: {e}")

    self.name_to_hash[name] = ref_hash
    self.stats.similar_weights += 1
```

**Delta Encoding Example:**

```python
# Reference weight (1000 elements)
ref = np.array([1.0, 2.0, 3.0, ..., 1000.0])  # 4KB at float32

# Similar weight (99.5% similar, fine-tuned)
similar = np.array([1.01, 2.02, 3.01, ..., 1000.05])  # 4KB at float32

# Delta encoding stores only the difference
delta = similar - ref  # Compressed to ~500 bytes
# Perfect reconstruction: similar = ref + delta ✓
```

### 2.7 Finding Similar Weights: Linear Scan vs LSH

The deduplicator supports two strategies for finding similar weights:

#### Linear Scan (Default, O(n))

```python
def _find_similar_weight(self, weight: WeightTensor) -> Optional[str]:
    """Find a similar weight in the index using O(n) scan."""
    from coral.utils.similarity import are_similar

    # Scan all weights with matching shape/dtype
    candidates = [
        (hash_val, w)
        for hash_val, w in self.weight_index.items()
        if w.shape == weight.shape and w.dtype == weight.dtype
    ]

    # Find most similar weight above threshold
    best_similarity = self.similarity_threshold
    best_hash = None

    for hash_val, candidate in candidates:
        if are_similar(
            weight.data,
            candidate.data,
            threshold=self.similarity_threshold,
            check_magnitude=True,
            magnitude_tolerance=self.magnitude_tolerance,
        ):
            similarity = self._compute_similarity(weight, candidate)
            if similarity > best_similarity:
                best_similarity = similarity
                best_hash = hash_val

    return best_hash
```

**When to Use:** Default choice for most workloads (<10,000 unique weights). Simple, reliable, no tuning required.

#### LSH-Accelerated Search (Optional, O(1) average)

```python
def _find_similar_weight(self, weight: WeightTensor) -> Optional[str]:
    """Find similar weight using O(1) LSH lookup."""
    from coral.utils.similarity import are_similar

    if self.lsh_index is not None:
        # O(1) average case: get candidate hashes from LSH
        candidate_hashes = self.lsh_index.query(weight.data)
        candidates = [
            (h, self.weight_index[h])
            for h in candidate_hashes
            if h in self.weight_index
            and self.weight_index[h].shape == weight.shape
            and self.weight_index[h].dtype == weight.dtype
        ]
    else:
        # O(n) fallback
        candidates = [...]

    # Find best match among candidates
    ...
```

**When to Use:** Enable for very large-scale deployments (>10,000 unique weights). Requires tuning `LSHConfig` parameters.

### 2.8 Retrieving Weights with Automatic Delta Reconstruction

```python
def get_weight_by_name(self, name: str) -> Optional[WeightTensor]:
    """Get weight by name, reconstructing from delta if needed.

    Thread-safe with automatic delta reconstruction.
    """
    with self._lock:
        if name not in self.name_to_hash:
            return None

        # Check if this is a delta-encoded similar weight
        if name in self.name_to_delta and self.enable_delta_encoding:
            return self._reconstruct_from_delta(name)

        # Otherwise get the reference weight directly
        hash_val = self.name_to_hash[name]
        return self.weight_index.get(hash_val)

def _reconstruct_from_delta(self, name: str) -> Optional[WeightTensor]:
    """Reconstruct original weight from delta encoding."""
    delta_hash = self.name_to_delta[name]
    delta = self.delta_index.get(delta_hash)

    # Get reference weight
    ref_weight = self.weight_index.get(delta.reference_hash)

    # Reconstruct: original = reference + delta
    reconstructed = self.delta_encoder.decode_delta(delta, ref_weight)
    return reconstructed
```

**Transparency:** Users never need to know whether a weight is stored directly or as a delta. The deduplicator handles reconstruction automatically.

### 2.9 Thread Safety

The deduplicator uses a reentrant lock (`threading.RLock`) to ensure thread safety:

```python
self._lock = threading.RLock()

def add_weight(self, weight, name):
    weight_hash = weight.compute_hash()  # Outside lock

    with self._lock:
        # Critical section: modify shared indices
        ...

def get_weight_by_name(self, name):
    with self._lock:
        # Critical section: read from indices
        ...
```

**Why RLock?** A reentrant lock allows the same thread to acquire the lock multiple times, which is necessary since some methods call other methods that also acquire the lock.

### 2.10 Performance Optimization

The deduplicator employs several optimization techniques:

1. **Hash Computation Outside Lock**: CPU-intensive hashing happens without holding the lock
2. **Early-Exit Shape/Dtype Checks**: Avoid expensive similarity computation for incompatible weights
3. **Magnitude Pre-filtering**: Check magnitude similarity before cosine similarity
4. **Lazy Delta Encoding**: Deltas are created during insertion, not on every access
5. **Optional LSH**: Trade memory for O(1) similarity lookup at scale

### 2.11 Usage Examples

**Basic Deduplication:**

```python
from coral.core.deduplicator import Deduplicator
from coral.core.weight_tensor import WeightTensor

# Initialize deduplicator
dedup = Deduplicator(
    similarity_threshold=0.99,
    enable_delta_encoding=True
)

# Add weights from multiple checkpoints
for i, checkpoint in enumerate(checkpoints):
    for layer_name, weight_data in checkpoint.items():
        weight = WeightTensor(data=weight_data, metadata=...)
        dedup.add_weight(weight, name=f"ckpt{i}.{layer_name}")

# Get statistics
stats = dedup.compute_stats()
print(f"Total weights: {stats.total_weights}")
print(f"Unique weights: {stats.unique_weights}")
print(f"Space saved: {stats.bytes_saved / 1e6:.1f} MB")
print(f"Compression ratio: {stats.compression_ratio:.1%}")

# Retrieve a weight (automatically reconstructs from delta if needed)
weight = dedup.get_weight_by_name("ckpt5.conv1.weight")
```

**Detailed Deduplication Report:**

```python
report = dedup.get_deduplication_report()

print("\nSummary:")
print(f"  Total weights: {report['summary']['total_weights']}")
print(f"  Duplicates: {report['summary']['duplicate_weights']}")
print(f"  Similar: {report['summary']['similar_weights']}")

print("\nTop 10 Largest Groups:")
for group in report['largest_groups']:
    print(f"  {group['reference_name']}:")
    print(f"    Total: {group['total_weights']} weights")
    print(f"    Saved: {group['bytes_saved'] / 1e6:.1f} MB")
```

## 3. Similarity Detection

Similarity detection is the foundation of Coral's space-efficient storage. Multiple algorithms work together to identify near-duplicate weights accurately and efficiently.

### 3.1 Cosine Similarity

Cosine similarity measures the angle between two vectors, ranging from -1 (opposite) to 1 (identical direction):

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two arrays.

    Note: Scale-invariant. [1,2,3] and [100,200,300] have similarity 1.0.
    """
    a_flat = np.asarray(a).flatten().astype(np.float64)
    b_flat = np.asarray(b).flatten().astype(np.float64)

    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)

    # Handle zero vectors
    if norm_a == 0 and norm_b == 0:
        return 1.0
    if norm_a == 0 or norm_b == 0:
        return 0.0

    dot_product = np.dot(a_flat, b_flat)
    return float(dot_product / (norm_a * norm_b))
```

**Limitation:** Cosine similarity is scale-invariant. For neural network weights, `[1,2,3]` and `[100,200,300]` should NOT be considered similar (different magnitudes imply different learned features).

### 3.2 Magnitude Similarity

Magnitude similarity addresses the scale-invariance problem:

```python
def magnitude_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute magnitude similarity between two arrays.

    Returns ratio of smaller to larger norm (0-1).
    """
    norm_a = np.linalg.norm(a.flatten().astype(np.float64))
    norm_b = np.linalg.norm(b.flatten().astype(np.float64))

    if norm_a == 0 and norm_b == 0:
        return 1.0
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(min(norm_a, norm_b) / max(norm_a, norm_b))
```

**Example:**
- `a = [1, 2, 3]`, `b = [1.01, 2.01, 3.01]`: magnitude_sim ≈ 0.997
- `a = [1, 2, 3]`, `b = [100, 200, 300]`: magnitude_sim ≈ 0.01

### 3.3 Combined Weight Similarity

Coral uses a hybrid metric that considers both direction and magnitude:

```python
def weight_similarity(
    a: np.ndarray,
    b: np.ndarray,
    direction_weight: float = 0.7,
    magnitude_weight: float = 0.3,
) -> float:
    """
    Compute similarity for neural network weights.

    Hybrid metric: 70% directional, 30% magnitude (tunable).
    """
    # Normalize weights
    total = direction_weight + magnitude_weight
    d_weight = direction_weight / total
    m_weight = magnitude_weight / total

    # Compute components
    cos_sim = cosine_similarity(a, b)
    mag_sim = magnitude_similarity(a, b)

    # Combine: map cosine [-1,1] to [0,1]
    cos_normalized = (cos_sim + 1.0) / 2.0

    return float(d_weight * cos_normalized + m_weight * mag_sim)
```

**Default Weights (0.7/0.3):**
- Prioritizes directional similarity (most important for functionality)
- Penalizes large magnitude differences
- Tuned for typical neural network training scenarios

### 3.4 The are_similar Function

This is the primary entry point for similarity checks in the deduplicator:

```python
def are_similar(
    a: np.ndarray,
    b: np.ndarray,
    threshold: float = 0.99,
    check_magnitude: bool = True,
    magnitude_tolerance: float = 0.1,
) -> bool:
    """
    Check if two arrays are similar for deduplication.

    Args:
        threshold: Cosine similarity threshold (0-1)
        check_magnitude: Also verify magnitudes are similar
        magnitude_tolerance: Max relative magnitude difference (0-1)
    """
    # Check directional similarity
    cos_sim = cosine_similarity(a, b)
    if cos_sim < threshold:
        return False

    # Optionally check magnitude similarity
    if check_magnitude:
        mag_sim = magnitude_similarity(a, b)
        # tolerance of 0.1 means we need mag_sim >= 0.9
        if mag_sim < (1.0 - magnitude_tolerance):
            return False

    return True
```

**Two-Stage Check:**
1. **Directional**: Must exceed cosine threshold (default 0.99)
2. **Magnitude**: Magnitudes must be within tolerance (default ±10%)

### 3.5 SimHash: Fast Similarity Fingerprinting

For very large-scale deployments, Coral supports **SimHash** - a locality-sensitive hashing technique that produces compact binary fingerprints where similar vectors have similar fingerprints.

```python
class SimHash:
    """SimHash fingerprinting for fast similarity detection.

    Works by:
    1. Projecting vectors onto random hyperplanes
    2. Creating binary signature based on projection signs
    3. Similar vectors produce similar signatures (small Hamming distance)
    """

    def compute_fingerprint(self, vector: np.ndarray) -> np.uint64:
        """Compute 64-bit SimHash fingerprint."""
        # Flatten and normalize
        flat = vector.flatten().astype(np.float32)
        norm = np.linalg.norm(flat)
        if norm > 0:
            flat = flat / norm

        # Initialize random hyperplanes if needed
        self._initialize_hyperplanes(len(flat))

        # Project onto hyperplanes
        projections = np.dot(self._hyperplanes, flat)

        # Create binary signature (1 if projection > 0, else 0)
        bits = (projections > 0).astype(np.uint8)

        # Pack into 64-bit integer
        return self._pack_bits(bits)
```

**Hamming Distance:**

Two fingerprints are compared using Hamming distance (number of differing bits):

```python
@staticmethod
def hamming_distance(fp1: np.uint64, fp2: np.uint64) -> int:
    """Compute Hamming distance between fingerprints."""
    xor = fp1 ^ fp2
    return bin(int(xor)).count("1")
```

**SimHash vs Full Similarity:**
- **Speed**: SimHash is O(1), cosine similarity is O(n) where n = vector dimension
- **Accuracy**: SimHash provides approximate similarity, cosine is exact
- **Use Case**: SimHash for initial filtering, then verify with cosine similarity

### 3.6 LSH Index: O(1) Similarity Lookup

Locality-Sensitive Hashing (LSH) enables sub-linear similarity search by organizing vectors into hash buckets:

```python
class LSHIndex:
    """LSH index for fast similarity search.

    Algorithm:
    1. Generate k random hyperplanes (normal vectors)
    2. For each weight, compute which side of each hyperplane it falls on
    3. This gives a k-bit hash where similar vectors have the same hash
    4. Store vectors in hash buckets
    5. To find similar vectors, only search in the same bucket
    """

    def __init__(self, vector_dim: int, config: Optional[LSHConfig] = None):
        self.vector_dim = vector_dim
        self.config = config or LSHConfig()

        # Create L hash tables with k hyperplanes each
        self.tables: List[LSHTable] = []
        for _ in range(self.config.num_tables):
            hyperplanes = self.rng.randn(
                self.config.num_hyperplanes,
                vector_dim
            ).astype(np.float32)
            # Normalize for stability
            norms = np.linalg.norm(hyperplanes, axis=1, keepdims=True)
            hyperplanes = hyperplanes / (norms + 1e-10)
            self.tables.append(LSHTable(hyperplanes=hyperplanes))
```

**Query Process:**

```python
def query(self, vector: np.ndarray) -> Set[str]:
    """Find candidate similar vectors."""
    flat_vector = vector.flatten().astype(np.float32)

    # Normalize for cosine similarity
    norm = np.linalg.norm(flat_vector)
    if norm > 0:
        flat_vector = flat_vector / norm

    # Union of candidates from all tables
    candidates: Set[str] = set()
    for table in self.tables:
        candidates.update(table.query(flat_vector))

    return candidates
```

**LSH Configuration Trade-offs:**

```python
@dataclass
class LSHConfig:
    num_hyperplanes: int = 8   # More = fewer false positives
    num_tables: int = 4         # More = fewer false negatives, more memory
    seed: Optional[int] = 42
    max_candidates: int = 100
```

**Typical Performance:**
- Default (k=8, L=4): ~95% recall for similarity ≥ 0.9
- Query time: ~1-2ms for 100k indexed weights (vs ~100ms for linear scan)
- Memory overhead: ~10% of original weight storage

## 4. Design Patterns Used

Coral's architecture embodies several key design patterns that contribute to its flexibility, efficiency, and maintainability.

### 4.1 Content-Addressable Storage Pattern

Every weight is identified by its content hash, not by arbitrary names or IDs:

```python
# Traditional storage: name-based
storage["conv1.weight"] = weight_data

# Coral: content-based
hash_value = compute_hash(weight_data)
storage[hash_value] = weight_data
name_to_hash["conv1.weight"] = hash_value
```

**Benefits:**
- **Automatic Deduplication**: Identical content naturally maps to the same key
- **Integrity Verification**: Hash mismatch indicates corruption
- **Location Independence**: Same content has same identifier across systems
- **Efficient Diff/Merge**: Compare hashes instead of full data

### 4.2 Metadata Separation Pattern

Weight data and metadata are stored separately:

```
┌─────────────────────┐     ┌──────────────────────┐
│   Metadata Store    │     │    Weight Store      │
│  (JSON/Protobuf)    │     │    (HDF5/Binary)     │
├─────────────────────┤     ├──────────────────────┤
│ name: conv1.weight  │────▶│ hash: a3f2...        │
│ shape: (64,3,7,7)   │     │ data: [binary blob]  │
│ dtype: float32      │     └──────────────────────┘
│ hash: a3f2...       │
│ layer_type: Conv2d  │
└─────────────────────┘
```

**Benefits:**
- **Fast Metadata Queries**: Scan commit history without loading weight data
- **Efficient Indexing**: Build search indices on metadata only
- **Lazy Loading**: Load weight data on-demand
- **Bandwidth Optimization**: Transfer only metadata for remote queries

### 4.3 Pluggable Backends Interface

Storage backends implement a common interface, enabling multiple implementations:

```python
class WeightStore(ABC):
    """Abstract interface for weight storage backends."""

    @abstractmethod
    def store_weight(self, hash: str, weight: WeightTensor) -> None:
        """Store a weight tensor."""
        pass

    @abstractmethod
    def load_weight(self, hash: str) -> WeightTensor:
        """Load a weight tensor by hash."""
        pass

    @abstractmethod
    def exists(self, hash: str) -> bool:
        """Check if weight exists."""
        pass

# Implementations:
# - HDF5Store: Production storage with compression
# - MemoryStore: In-memory for testing
# - S3Store: Cloud storage (future)
# - RedisStore: Distributed cache (future)
```

**Benefits:**
- **Testability**: Use in-memory backend for fast tests
- **Flexibility**: Choose backend based on deployment needs
- **Future-Proof**: Add new backends without changing core logic

### 4.4 Factory Pattern for Weight Creation

`WeightTensor` provides factory methods for different data sources:

```python
# From dictionary (deserialization)
weight = WeightTensor.from_dict(data_dict, weight_data=array)

# From PyTorch
weight = WeightTensor.from_pytorch(model.conv1.weight)

# From TensorFlow
weight = WeightTensor.from_tensorflow(layer.kernel)

# From JAX
weight = WeightTensor.from_jax(params['layer1']['kernel'])
```

This pattern centralizes validation and normalization logic, ensuring consistent weight creation across different frameworks.

### 4.5 Lazy Evaluation Pattern

WeightTensor uses lazy evaluation for both data loading and hash computation:

```python
@property
def data(self) -> np.ndarray:
    """Lazy load data from storage if not in memory."""
    if self._data is None and self._store_ref is not None:
        # Load from storage on first access
        self._data = self._storage_backend.load(self._store_ref)
    return self._data

def compute_hash(self, force: bool = False) -> str:
    """Lazy compute and cache hash."""
    if self._hash is not None and not force:
        return self._hash  # Return cached value
    # Compute and cache
    self._hash = self._compute_hash_impl()
    return self._hash
```

**Benefits:**
- **Memory Efficiency**: Only load what's needed
- **Performance**: Avoid redundant computation
- **Scalability**: Handle datasets larger than RAM

## Summary

This chapter explored the core architecture of the Coral system, focusing on two fundamental components:

**WeightTensor** provides:
- Content-based addressing with xxHash64 for fast, reliable identification
- Rich metadata tracking with shape, dtype, layer type, and compression info
- Lazy loading capabilities for memory efficiency
- Built-in similarity detection
- Framework-agnostic representation

**Deduplicator** provides:
- Exact deduplication via content hashing (100% space savings for duplicates)
- Similarity-based deduplication with lossless delta encoding (50-95% savings)
- Thread-safe concurrent access with RLock
- Optional LSH acceleration for O(1) similarity search at scale
- Comprehensive statistics and reporting

**Similarity Detection** offers:
- Cosine similarity for directional comparison
- Magnitude similarity for scale awareness
- Hybrid weight_similarity metric tuned for neural networks
- SimHash for fast approximate similarity fingerprinting
- LSH indexing for sub-linear similarity search

**Design Patterns** ensure:
- Content-addressable storage for automatic deduplication
- Metadata separation for efficient queries
- Pluggable backends for flexibility
- Factory methods for consistent creation
- Lazy evaluation for performance

Together, these components form a robust foundation for the Coral system, enabling efficient, lossless versioning of neural network weights with practical space savings of 40-50% in real-world ML workflows.

In the next chapter, we'll explore the **Delta Encoding System**, which enables perfect reconstruction of similar weights from compact delta representations - the key innovation that makes Coral's similarity-based deduplication lossless.
