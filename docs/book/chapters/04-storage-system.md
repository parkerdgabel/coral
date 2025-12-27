# Chapter 4: Storage System

The storage system is the foundation of Coral's version control capabilities, providing efficient, scalable persistence for neural network weights. This chapter explores Coral's content-addressable storage architecture, pluggable backend system, and optimization strategies for handling large-scale machine learning models.

## 4.1 Storage Architecture Overview

### Content-Addressable Storage

Coral employs a content-addressable storage (CAS) model, similar to Git, where data is stored and retrieved based on its content hash rather than arbitrary file paths. This approach provides several key benefits:

**Automatic Deduplication**: Identical weights are stored only once, regardless of how many times they appear across different models or checkpoints. If two models share identical layer weights, Coral stores them once and references them multiple times.

**Data Integrity**: Content hashing ensures that stored weights cannot be corrupted without detection. When loading weights, Coral can verify the hash matches the expected value.

**Deterministic Storage**: The same weight data always produces the same hash, making storage operations idempotent and predictable.

### Hash-Based Addressing

Coral uses xxHash, a fast non-cryptographic hash algorithm, to generate content addresses. Each `WeightTensor` computes its hash from the normalized weight data:

```python
import xxhash
import numpy as np

def compute_hash(data: np.ndarray) -> str:
    """Compute xxHash of weight data."""
    hasher = xxhash.xxh64()
    hasher.update(data.tobytes())
    return hasher.hexdigest()
```

The hash serves as the primary key for storage operations. When you store a weight, the hash determines where it lives in the storage backend:

```python
from coral.core.weight_tensor import WeightTensor, WeightMetadata
import numpy as np

# Create a weight tensor
data = np.random.randn(512, 512).astype(np.float32)
metadata = WeightMetadata(
    name="layer1.weight",
    shape=data.shape,
    dtype=data.dtype,
    layer_type="Linear"
)
weight = WeightTensor(data=data, metadata=metadata)

# Hash is computed automatically
hash_key = weight.compute_hash()
# Example: 'a3f2c1d9e4b5...'
```

### Separation of Data and Metadata

Coral maintains a clean separation between weight data and metadata:

**Weight Data**: The actual numerical arrays (parameters) stored efficiently in binary format with optional compression.

**Metadata**: Descriptive information about weights (name, shape, dtype, layer type, model name) stored as structured attributes alongside the data.

This separation enables:
- Fast metadata queries without loading large weight arrays
- Efficient batch operations that filter by metadata
- Flexible metadata schemas that can evolve independently of data storage

### Why HDF5?

Coral uses HDF5 (Hierarchical Data Format version 5) as its primary storage format for several compelling reasons:

**Hierarchical Organization**: HDF5 supports groups and datasets, allowing Coral to organize weights, deltas, and metadata in a logical tree structure within a single file.

**Built-in Compression**: Native support for multiple compression algorithms (gzip, lzf, szip) with configurable compression levels.

**Efficient I/O**: HDF5 provides chunked storage and partial I/O capabilities, enabling efficient loading of specific weights without reading the entire file.

**Metadata Support**: Attributes can be attached to datasets, providing a natural way to store metadata alongside weight data.

**Cross-Platform**: HDF5 files are portable across different operating systems and architectures.

**Python Integration**: The h5py library provides a Pythonic interface that integrates seamlessly with NumPy arrays.

## 4.2 WeightStore Abstract Interface

Coral's storage system is designed around a pluggable backend architecture. The `WeightStore` abstract base class defines a consistent interface that all storage backends must implement.

### Abstract Base Class Design

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from coral.core.weight_tensor import WeightMetadata, WeightTensor

class WeightStore(ABC):
    """Abstract base class for weight storage backends."""

    @abstractmethod
    def store(self, weight: WeightTensor, hash_key: Optional[str] = None) -> str:
        """Store a weight tensor and return its hash key."""
        pass

    @abstractmethod
    def load(self, hash_key: str) -> Optional[WeightTensor]:
        """Load a weight tensor by hash."""
        pass

    @abstractmethod
    def exists(self, hash_key: str) -> bool:
        """Check if a weight exists in storage."""
        pass

    @abstractmethod
    def delete(self, hash_key: str) -> bool:
        """Delete a weight from storage."""
        pass
```

### Required Methods

The `WeightStore` interface defines nine core methods that every backend must implement:

#### store(weight, hash_key)
Stores a weight tensor and returns its hash key. If `hash_key` is not provided, it's computed from the weight data. The method should be idempotent - storing the same weight multiple times should not create duplicates.

#### load(hash_key)
Retrieves a weight tensor by its hash. Returns `None` if the weight doesn't exist. This operation should be efficient and support lazy loading where possible.

#### exists(hash_key)
Checks whether a weight exists without loading its data. This is much faster than attempting to load and checking for `None`.

#### delete(hash_key)
Removes a weight from storage. Returns `True` if the weight was deleted, `False` if it didn't exist. This method is critical for garbage collection.

#### list_weights()
Returns a list of all weight hashes currently in storage. Used for inventory operations and garbage collection.

```python
@abstractmethod
def list_weights(self) -> List[str]:
    """List all weight hashes in storage."""
    pass
```

#### get_metadata(hash_key)
Retrieves only the metadata for a weight without loading the actual data array. This enables fast queries and filtering:

```python
@abstractmethod
def get_metadata(self, hash_key: str) -> Optional[WeightMetadata]:
    """Get metadata for a weight without loading data."""
    pass
```

#### store_batch(weights) and load_batch(hash_keys)
Batch operations for storing and loading multiple weights efficiently. Backends can optimize these operations with parallelization or pipelining:

```python
@abstractmethod
def store_batch(self, weights: Dict[str, WeightTensor]) -> Dict[str, str]:
    """Store multiple weights efficiently."""
    pass

@abstractmethod
def load_batch(self, hash_keys: List[str]) -> Dict[str, WeightTensor]:
    """Load multiple weights efficiently."""
    pass
```

#### get_storage_info()
Returns statistics and information about storage usage:

```python
@abstractmethod
def get_storage_info(self) -> Dict[str, Any]:
    """Get information about storage usage and statistics."""
    pass
```

### Benefits of the Pluggable Backend Approach

The abstract `WeightStore` interface provides several advantages:

**Flexibility**: Switch between storage backends (local HDF5, cloud S3, databases) without changing application code.

**Testability**: Mock storage backends for testing without requiring actual storage infrastructure.

**Extensibility**: Add new storage backends (PostgreSQL, Redis, custom formats) by implementing the interface.

**Optimization**: Backends can optimize for specific use cases (local development, cloud deployment, distributed training).

**Migration**: Migrate data between backends using the common interface.

## 4.3 HDF5Store - Primary Backend

The `HDF5Store` is Coral's primary storage backend, optimized for local development and single-machine deployments.

### HDF5 File Format Benefits

HDF5 provides an ideal foundation for weight storage:

```python
from coral.storage import HDF5Store

# Initialize with compression
store = HDF5Store(
    filepath="/path/to/repo/.coral/objects/weights.h5",
    compression="gzip",
    compression_opts=4  # Compression level 1-9
)
```

### Three HDF5 Groups: Weights, Metadata, and Deltas

The HDF5Store organizes data into three logical groups within a single HDF5 file:

**weights/**: Stores actual weight tensors as datasets, one per hash key
**metadata/**: Reserved for future metadata-only storage optimizations
**deltas/**: Stores delta encoding objects for similar weights

```
weights.h5
├── weights/
│   ├── a3f2c1d9... (dataset: shape (512, 512), dtype float32)
│   ├── b5e7a8f1... (dataset: shape (256, 256), dtype float32)
│   └── c9d4f2e8... (dataset: shape (128, 128), dtype float32)
├── metadata/
│   └── (reserved for future use)
└── deltas/
    ├── d1a5e9c3... (delta from reference a3f2c1d9...)
    └── e7f3b2d8... (delta from reference b5e7a8f1...)
```

Each dataset in the `weights/` group represents one unique weight tensor, stored with compression and metadata attributes.

### Compression Options

HDF5Store supports multiple compression algorithms:

**gzip**: Universal compression, good balance of ratio and speed (default, level 4)
**lzf**: Faster compression, lower ratio, good for frequently accessed weights
**szip**: Specialized scientific data compression (requires additional library)
**None**: No compression, fastest I/O but largest file size

```python
# High compression for archival
archive_store = HDF5Store(
    "archive.h5",
    compression="gzip",
    compression_opts=9  # Maximum compression
)

# Fast I/O for active development
dev_store = HDF5Store(
    "dev.h5",
    compression="lzf"  # Faster, lower compression
)

# No compression for maximum speed
speed_store = HDF5Store(
    "speed.h5",
    compression=None
)
```

### Metadata as HDF5 Attributes

Weight metadata is stored as HDF5 dataset attributes, enabling fast access without loading the array data:

```python
def store(self, weight: WeightTensor, hash_key: Optional[str] = None) -> str:
    """Store a weight tensor."""
    if hash_key is None:
        hash_key = weight.compute_hash()

    # Store weight data
    weights_group = self.file["weights"]
    dataset = weights_group.create_dataset(
        hash_key,
        data=weight.data,
        compression=self.compression,
        compression_opts=self.compression_opts,
    )

    # Store metadata as attributes
    metadata = weight.metadata
    dataset.attrs["name"] = metadata.name
    dataset.attrs["shape"] = metadata.shape
    dataset.attrs["dtype"] = np.dtype(metadata.dtype).name
    dataset.attrs["layer_type"] = metadata.layer_type or ""
    dataset.attrs["model_name"] = metadata.model_name or ""
    dataset.attrs["hash"] = hash_key

    self.file.flush()
    return hash_key
```

This approach allows querying metadata without loading large arrays:

```python
# Fast metadata query - no array loading
metadata = store.get_metadata(hash_key)
print(f"Layer: {metadata.name}, Shape: {metadata.shape}")

# Full weight loading - loads array data
weight = store.load(hash_key)
print(f"Data: {weight.data}")
```

### Batch Operations for Efficiency

Batch operations reduce overhead by amortizing file I/O and lock acquisition:

```python
# Store multiple weights efficiently
weights = {
    "layer1": weight1,
    "layer2": weight2,
    "layer3": weight3,
}
hash_map = store.store_batch(weights)
# Returns: {"layer1": "a3f2...", "layer2": "b5e7...", "layer3": "c9d4..."}

# Load multiple weights efficiently
hashes = ["a3f2...", "b5e7...", "c9d4..."]
loaded = store.load_batch(hashes)
# Returns: {hash: WeightTensor, ...}
```

While the current implementation calls `store()` and `load()` sequentially, the interface supports future optimizations like parallel I/O or vectorized operations.

### Delta Storage and Retrieval

HDF5Store provides specialized methods for storing delta-encoded weights:

```python
from coral.delta.delta_encoder import Delta, DeltaType

# Store a delta object
delta = Delta(
    delta_type=DeltaType.FLOAT32_RAW,
    data=delta_array,
    original_shape=(512, 512),
    original_dtype=np.float32,
    reference_hash="a3f2c1d9...",
    compression_ratio=0.95
)

delta_hash = store.store_delta(delta, "d1a5e9c3...")

# Load a delta object
loaded_delta = store.load_delta("d1a5e9c3...")

# Check delta existence
if store.delta_exists("d1a5e9c3..."):
    print("Delta exists in storage")
```

Delta objects are stored in the `deltas/` group with their own metadata attributes, enabling efficient similarity-based deduplication while maintaining perfect reconstruction fidelity.

### Garbage Collection Support

The HDF5Store provides methods for garbage collection to reclaim storage from unreferenced weights:

```python
# List all weights in storage
all_weights = store.list_weights()

# List all deltas
all_deltas = store.list_deltas()

# Delete unreferenced weights
for hash_key in unreferenced_weights:
    store.delete(hash_key)

# Delete unreferenced deltas
for delta_hash in unreferenced_deltas:
    store.delete_delta(delta_hash)
```

The Repository uses these methods to implement garbage collection that removes weights and deltas not referenced by any commit.

### Thread Safety Considerations

HDF5 has limitations with concurrent access. The HDF5Store is designed for single-writer, multiple-reader scenarios:

- **Single Writer**: Only one process should open the store in write mode (`mode='a'` or `mode='w'`)
- **Multiple Readers**: Multiple processes can open in read-only mode (`mode='r'`)
- **File Locking**: The operating system and HDF5 library handle file locks

For multi-process or distributed scenarios, consider:
- Using S3Store for cloud-based concurrent access
- Implementing application-level locking
- Using separate HDF5 files per process

### Performance Characteristics

HDF5Store provides excellent performance for typical ML workflows:

**Write Performance**: 100-500 MB/s depending on compression (gzip level 4)
**Read Performance**: 500-2000 MB/s with lazy loading
**Metadata Queries**: <1ms per query (no array loading)
**Compression Ratio**: 1.5-3x for typical neural network weights with gzip

Example benchmark for a ResNet-50 model (25M parameters):

```python
import time
import numpy as np
from coral.storage import HDF5Store
from coral.core.weight_tensor import WeightTensor, WeightMetadata

store = HDF5Store("benchmark.h5", compression="gzip", compression_opts=4)

# Store 25M parameters (100MB)
data = np.random.randn(25_000_000).astype(np.float32)
metadata = WeightMetadata(name="resnet50", shape=data.shape, dtype=data.dtype)
weight = WeightTensor(data=data, metadata=metadata)

start = time.time()
hash_key = store.store(weight)
write_time = time.time() - start
# Typical: 0.2-0.5 seconds

start = time.time()
loaded = store.load(hash_key)
read_time = time.time() - start
# Typical: 0.05-0.15 seconds

print(f"Write: {100/write_time:.1f} MB/s")
print(f"Read: {100/read_time:.1f} MB/s")
```

## 4.4 S3Store - Cloud Storage Backend

The `S3Store` enables cloud storage for Coral repositories, supporting AWS S3, MinIO, DigitalOcean Spaces, and any S3-compatible service.

### AWS S3 and S3-Compatible Storage

S3Store works with any S3-compatible object storage:

**AWS S3**: Amazon's cloud object storage
**MinIO**: Self-hosted S3-compatible storage
**DigitalOcean Spaces**: Managed S3-compatible storage
**Wasabi**: High-performance S3-compatible storage
**Backblaze B2**: Cost-effective S3-compatible storage

```python
from coral.storage.s3_store import S3Store, S3Config

# AWS S3
aws_config = S3Config(
    bucket="my-ml-models",
    region="us-west-2",
    prefix="coral/experiments/"
)
aws_store = S3Store(aws_config)

# MinIO (self-hosted)
minio_config = S3Config(
    bucket="models",
    endpoint_url="http://localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    prefix="coral/"
)
minio_store = S3Store(minio_config)
```

### Object Naming Convention

S3Store organizes objects using a hierarchical key structure:

```
s3://bucket/prefix/
├── weights/
│   ├── a3f2c1d9...npz (compressed weight data)
│   ├── b5e7a8f1...npz
│   └── c9d4f2e8...npz
├── metadata/
│   ├── a3f2c1d9...json (weight metadata)
│   ├── b5e7a8f1...json
│   └── c9d4f2e8...json
└── deltas/
    ├── d1a5e9c3...npz (delta data)
    └── e7f3b2d8...npz
```

Each weight is stored as two objects:
- **weights/{hash}.npz**: Compressed NumPy array
- **metadata/{hash}.json**: JSON metadata

### Authentication and Configuration

S3Store supports multiple authentication methods:

**AWS Credentials Chain**: Uses default AWS credentials (environment variables, ~/.aws/credentials, IAM role)
**Explicit Credentials**: Provide access key and secret key directly
**IAM Roles**: For EC2 instances or Lambda functions

```python
# Default AWS credentials chain
config1 = S3Config(bucket="my-bucket")

# Explicit credentials
config2 = S3Config(
    bucket="my-bucket",
    access_key="AKIAIOSFODNN7EXAMPLE",
    secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
)

# Custom endpoint (MinIO)
config3 = S3Config(
    bucket="my-bucket",
    endpoint_url="https://minio.example.com",
    access_key="my-access-key",
    secret_key="my-secret-key"
)
```

### Remote Storage Workflow

Using S3Store for remote model storage:

```python
from coral.storage import HDF5Store
from coral.storage.s3_store import S3Store, S3Config

# Local development with HDF5
local_store = HDF5Store("local_weights.h5")

# Store weights locally
hash_key = local_store.store(weight)

# Configure S3 for remote backup
s3_config = S3Config(
    bucket="ml-model-backup",
    region="us-east-1",
    prefix="coral/project-x/"
)
s3_store = S3Store(s3_config)

# Sync to S3
stats = s3_store.sync_from_local(local_store)
print(f"Uploaded {stats['uploaded']} weights")
print(f"Skipped {stats['skipped']} (already uploaded)")
print(f"Transferred {stats['bytes_transferred']} bytes")

# Later, sync from S3 to new local machine
new_local_store = HDF5Store("new_machine_weights.h5")
stats = s3_store.sync_to_local(new_local_store)
print(f"Downloaded {stats['downloaded']} weights")
```

### When to Use S3 vs Local HDF5

**Use HDF5Store when:**
- Working on a single machine
- Need maximum performance (local disk I/O)
- Repository size is manageable locally
- No need for collaboration or remote access

**Use S3Store when:**
- Sharing models across team members
- Working from multiple machines
- Need backup and disaster recovery
- Repository size exceeds local storage
- Training in cloud environments (AWS, GCP, Azure)

**Hybrid Approach:**
Use both together - HDF5 for active development, S3 for backup and collaboration:

```python
# Local fast storage
local = HDF5Store(".coral/objects/weights.h5")

# Remote backup
remote = S3Store(S3Config(bucket="team-models"))

# Work locally
local.store(weight)

# Periodic sync to cloud
remote.sync_from_local(local)
```

## 4.5 Storage Configuration

### Compression Level Settings

Compression trades CPU time for storage space. Choose based on your priorities:

```python
# Maximum speed, no compression
store = HDF5Store("fast.h5", compression=None)

# Balanced (default) - gzip level 4
store = HDF5Store("balanced.h5", compression="gzip", compression_opts=4)

# Maximum compression - gzip level 9
store = HDF5Store("archived.h5", compression="gzip", compression_opts=9)

# Fast compression - lzf
store = HDF5Store("dev.h5", compression="lzf")
```

Performance comparison (ResNet-50, 100MB weights):

| Compression | Level | File Size | Write Time | Read Time |
|-------------|-------|-----------|------------|-----------|
| None        | -     | 100 MB    | 0.15s      | 0.05s     |
| lzf         | -     | 65 MB     | 0.25s      | 0.08s     |
| gzip        | 1     | 55 MB     | 0.30s      | 0.10s     |
| gzip        | 4     | 40 MB     | 0.45s      | 0.12s     |
| gzip        | 9     | 35 MB     | 1.20s      | 0.15s     |

### Chunk Size Optimization

HDF5 uses chunking for efficient partial I/O. The default chunk size works well for most cases, but you can optimize for specific access patterns:

```python
# HDF5 automatically determines chunk size, but you can influence it
# by structuring your weights appropriately

# For models with many small weights, batch operations help
weights = {f"layer_{i}": small_weight for i in range(100)}
store.store_batch(weights)  # More efficient than 100 individual stores

# For very large weights, consider splitting into smaller tensors
large_weight = np.random.randn(10000, 10000)
# Consider splitting into chunks if only partial loading is needed
```

### Memory Management

Coral's storage system is designed for efficient memory usage:

**Lazy Loading**: Weight data is only loaded when accessed
**Streaming**: Large models can be loaded piece by piece
**Context Managers**: Automatic resource cleanup

```python
# Context manager ensures file is closed
with HDF5Store("weights.h5") as store:
    weight = store.load(hash_key)
    # Process weight
# File automatically closed

# Manual management
store = HDF5Store("weights.h5")
try:
    weight = store.load(hash_key)
finally:
    store.close()
```

### Caching Strategies

For frequently accessed weights, consider application-level caching:

```python
from functools import lru_cache

class CachedStore:
    def __init__(self, store: HDF5Store):
        self.store = store
        self._cache = {}

    def load(self, hash_key: str):
        if hash_key not in self._cache:
            self._cache[hash_key] = self.store.load(hash_key)
        return self._cache[hash_key]

    def clear_cache(self):
        self._cache.clear()

# Use cached store for repeated access
store = HDF5Store("weights.h5")
cached = CachedStore(store)

# First load: reads from disk
w1 = cached.load(hash_key)

# Second load: returns cached copy
w2 = cached.load(hash_key)

# Clear cache when memory is low
cached.clear_cache()
```

## 4.6 Storage Operations

### Storing Weights

Basic weight storage:

```python
from coral.storage import HDF5Store
from coral.core.weight_tensor import WeightTensor, WeightMetadata
import numpy as np

store = HDF5Store("weights.h5")

# Create weight
data = np.random.randn(512, 256).astype(np.float32)
metadata = WeightMetadata(
    name="encoder.layer1.weight",
    shape=data.shape,
    dtype=data.dtype,
    layer_type="Linear",
    model_name="TransformerEncoder"
)
weight = WeightTensor(data=data, metadata=metadata)

# Store weight
hash_key = store.store(weight)
print(f"Stored weight with hash: {hash_key}")

# Store with explicit hash
custom_hash = "my_custom_hash"
hash_key = store.store(weight, hash_key=custom_hash)
```

### Loading Weights with Lazy Loading

```python
# Load full weight
weight = store.load(hash_key)
print(f"Loaded {weight.metadata.name}: {weight.data.shape}")

# Load only metadata (fast, no array loading)
metadata = store.get_metadata(hash_key)
print(f"Weight info: {metadata.name}, {metadata.shape}")

# Check existence before loading
if store.exists(hash_key):
    weight = store.load(hash_key)
else:
    print("Weight not found")
```

### Batch Operations

Batch operations are more efficient for multiple weights:

```python
# Store multiple weights
weights = {}
for i in range(10):
    data = np.random.randn(100, 100).astype(np.float32)
    metadata = WeightMetadata(name=f"layer{i}", shape=data.shape, dtype=data.dtype)
    weights[f"layer{i}"] = WeightTensor(data=data, metadata=metadata)

# Batch store
hash_map = store.store_batch(weights)
# Returns: {"layer0": "hash0", "layer1": "hash1", ...}

# Batch load
hashes = list(hash_map.values())
loaded_weights = store.load_batch(hashes)
# Returns: {"hash0": WeightTensor, "hash1": WeightTensor, ...}
```

### Garbage Collection

Remove unreferenced weights to reclaim storage:

```python
# Get all stored weights
all_hashes = set(store.list_weights())

# Get referenced weights (from repository commits)
referenced_hashes = set(repo.get_all_referenced_weights())

# Find unreferenced weights
unreferenced = all_hashes - referenced_hashes

# Delete unreferenced weights
for hash_key in unreferenced:
    store.delete(hash_key)

print(f"Removed {len(unreferenced)} unreferenced weights")

# Same for deltas
all_delta_hashes = set(store.list_deltas())
referenced_deltas = set(repo.get_all_referenced_deltas())
unreferenced_deltas = all_delta_hashes - referenced_deltas

for delta_hash in unreferenced_deltas:
    store.delete_delta(delta_hash)
```

### Storage Statistics

Monitor storage usage and performance:

```python
# Get storage information
info = store.get_storage_info()

print(f"Storage backend: {info['filepath']}")
print(f"File size: {info['file_size'] / (1024**2):.2f} MB")
print(f"Total weights: {info['total_weights']}")
print(f"Uncompressed size: {info['total_bytes'] / (1024**2):.2f} MB")
print(f"Compressed size: {info['compressed_bytes'] / (1024**2):.2f} MB")
print(f"Compression ratio: {info['compression_ratio']:.2%}")

# Delta storage info
delta_info = store.get_delta_storage_info()
print(f"Total deltas: {delta_info['total_deltas']}")
print(f"Delta storage: {delta_info['total_delta_bytes'] / (1024**2):.2f} MB")
```

## 4.7 Integration with Other Components

### How Repository Uses Storage

The Repository class uses the storage system as its persistence layer:

```python
class Repository:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.coral_dir = self.path / ".coral"

        # Create storage instance
        self.weights_store_path = self.coral_dir / "objects" / "weights.h5"

        # Storage is accessed through deduplicator and version control
        self.deduplicator = Deduplicator(...)

    def stage_weights(self, weights: Dict[str, WeightTensor]):
        """Stage weights for commit."""
        # Deduplication happens here, using storage
        deduplicated = self.deduplicator.deduplicate(weights)

        # Store deduplicated weights
        with HDF5Store(self.weights_store_path) as store:
            hash_map = store.store_batch(deduplicated)

        return hash_map
```

### Delta Encoding Storage Workflow

When similar weights are detected, the deduplicator creates delta objects:

```python
from coral.core.deduplicator import Deduplicator
from coral.delta.delta_encoder import DeltaEncoder, DeltaConfig

# Initialize deduplicator with delta encoding
config = DeltaConfig()
deduplicator = Deduplicator(
    similarity_threshold=0.98,
    delta_config=config,
    enable_delta_encoding=True
)

# Storage workflow
store = HDF5Store("weights.h5")

# Weights to deduplicate
weight_a = WeightTensor(data_a, metadata_a)  # Reference weight
weight_b = WeightTensor(data_b, metadata_b)  # 99% similar to weight_a

# Deduplication process
result = deduplicator.deduplicate({"a": weight_a, "b": weight_b})

# Store reference weight normally
hash_a = store.store(weight_a)

# Store similar weight as delta
if result.deltas:
    for delta_hash, delta_obj in result.deltas.items():
        store.store_delta(delta_obj, delta_hash)

# Reconstruction during load
def load_with_delta(hash_key):
    # Check if it's a delta reference
    if store.delta_exists(hash_key):
        delta = store.load_delta(hash_key)
        reference = store.load(delta.reference_hash)
        # Reconstruct original weight
        reconstructed = delta_encoder.reconstruct(delta, reference.data)
        return WeightTensor(data=reconstructed, metadata=...)
    else:
        # Normal weight
        return store.load(hash_key)
```

### Deduplicator Integration

The deduplicator uses storage to identify duplicate weights:

```python
class Deduplicator:
    def __init__(self, storage: HDF5Store, ...):
        self.storage = storage
        self.weight_registry = {}  # hash -> metadata

    def deduplicate(self, weights: Dict[str, WeightTensor]):
        """Deduplicate weights using content hashing."""
        deduplicated = {}
        hash_to_name = {}

        for name, weight in weights.items():
            hash_key = weight.compute_hash()

            # Check if weight already exists in storage
            if self.storage.exists(hash_key):
                # Reuse existing weight
                hash_to_name[hash_key] = name
            else:
                # New weight, will be stored
                deduplicated[name] = weight

        return deduplicated
```

## 4.8 Best Practices

### Choosing Compression Settings

**Development**: Use `lzf` compression for fast iteration
```python
dev_store = HDF5Store("dev.h5", compression="lzf")
```

**Production**: Use `gzip` level 4 for balanced performance
```python
prod_store = HDF5Store("prod.h5", compression="gzip", compression_opts=4)
```

**Archive**: Use `gzip` level 9 for long-term storage
```python
archive_store = HDF5Store("archive.h5", compression="gzip", compression_opts=9)
```

### Optimizing for Large Models

For very large models (billions of parameters):

**1. Use Batch Operations**
```python
# Don't do this
for name, weight in model_weights.items():
    store.store(weight)

# Do this
store.store_batch(model_weights)
```

**2. Enable Delta Encoding**
```python
# In repository config
{
    "core": {
        "delta_encoding": true,
        "similarity_threshold": 0.98
    }
}
```

**3. Consider S3 for Very Large Repositories**
```python
# Local development
local_store = HDF5Store("local.h5")

# Remote archive
s3_store = S3Store(S3Config(bucket="large-models"))
s3_store.sync_from_local(local_store)
```

### Backup Strategies

**Strategy 1: Local + Cloud Hybrid**
```python
# Keep working copy local
local = HDF5Store(".coral/objects/weights.h5")

# Periodic backup to S3
s3 = S3Store(S3Config(bucket="coral-backup"))

# Daily backup script
s3.sync_from_local(local)
```

**Strategy 2: File-Level Backup**
```bash
# Simple file copy
cp .coral/objects/weights.h5 /backup/weights-$(date +%Y%m%d).h5

# Or use rsync
rsync -av .coral/ /backup/coral-backup/
```

**Strategy 3: Git-LFS for Coral Metadata**
```bash
# Track HDF5 files with Git LFS
git lfs track "*.h5"

# Commit Coral repository structure (but not weights.h5)
git add .coral/config.json .coral/refs/ .coral/HEAD
git commit -m "Save Coral structure"
```

### Migration Between Backends

Migrate from HDF5 to S3:

```python
from coral.storage import HDF5Store
from coral.storage.s3_store import S3Store, S3Config

# Source: Local HDF5
source = HDF5Store(".coral/objects/weights.h5")

# Destination: S3
dest = S3Store(S3Config(
    bucket="coral-production",
    region="us-west-2"
))

# Migrate all weights
stats = dest.sync_from_local(source)
print(f"Migrated {stats['uploaded']} weights to S3")
```

Migrate from S3 to HDF5:

```python
# Source: S3
source = S3Store(S3Config(bucket="coral-production"))

# Destination: Local HDF5
dest = HDF5Store("new-local.h5")

# Download all weights
stats = source.sync_to_local(dest)
print(f"Downloaded {stats['downloaded']} weights from S3")
```

---

The storage system is the cornerstone of Coral's version control capabilities, providing efficient, flexible, and scalable persistence for neural network weights. By understanding content-addressable storage, leveraging the pluggable backend architecture, and following best practices for compression and optimization, you can build reliable ML workflows that handle models of any size.

In the next chapter, we'll explore how the Version Control system builds on this storage foundation to provide git-like branching, committing, and merging for neural network weights.
