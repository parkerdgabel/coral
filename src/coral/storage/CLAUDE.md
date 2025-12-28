# Storage Module

This module provides storage backends for neural network weights, implementing content-addressable storage with compression support.

## Overview

The storage module provides:
- **Abstract `WeightStore` interface** for pluggable backends
- **HDF5Store** for local content-addressable storage
- **S3Store** for cloud storage (AWS S3, MinIO, DigitalOcean Spaces)

## Key Files

### `weight_store.py`

Abstract base class defining the storage interface.

**WeightStore** (ABC):

Required methods to implement:
```python
store(weight, hash_key=None) -> str        # Store weight, return hash
load(hash_key) -> Optional[WeightTensor]   # Load by hash
exists(hash_key) -> bool                    # Check existence
delete(hash_key) -> bool                    # Delete weight
list_weights() -> list[str]                 # List all hashes
get_metadata(hash_key) -> Optional[WeightMetadata]  # Get metadata only
store_batch(weights) -> dict[str, str]      # Batch store
load_batch(hash_keys) -> dict[str, WeightTensor]   # Batch load
get_storage_info() -> dict[str, Any]        # Storage statistics
close()                                      # Cleanup resources
```

Supports context manager protocol (`with store: ...`).

### `hdf5_store.py`

HDF5-based local storage with compression.

**HDF5Store** (class):

Constructor:
```python
HDF5Store(
    filepath: str,              # Path to HDF5 file
    compression: str = "gzip",  # "gzip", "lzf", or None
    compression_opts: int = 4,  # Compression level 1-9 (gzip only)
    mode: str = "a"             # File mode: "r", "r+", "w", "a"
)
```

**HDF5 Structure**:
```
file.h5
├── weights/           # Weight data group
│   ├── <hash1>       # Dataset with weight array
│   └── <hash2>       # Each has attrs for metadata
├── metadata/          # Metadata group (legacy)
└── deltas/           # Delta encodings group
    └── <delta_hash>  # Delta data with attrs
```

**Weight Dataset Attributes**:
- `name`: Weight name
- `shape`: Tensor shape
- `dtype`: Data type
- `layer_type`: Layer type (optional)
- `model_name`: Model name (optional)
- `compression_info`: JSON string
- `hash`: Content hash

**Additional Methods for Deltas**:
```python
store_delta(delta, delta_hash) -> str   # Store delta object
load_delta(delta_hash) -> Optional[Delta]
delta_exists(delta_hash) -> bool
delete_delta(delta_hash) -> bool
list_deltas() -> list[str]
get_delta_storage_info() -> dict[str, Any]
```

**Usage Example**:
```python
from coral.storage import HDF5Store
from coral.core import WeightTensor

# Create/open store
with HDF5Store("weights.h5", compression="gzip", compression_opts=9) as store:
    # Store weight
    hash_key = store.store(weight)

    # Load weight
    loaded = store.load(hash_key)

    # Get storage stats
    info = store.get_storage_info()
    print(f"Weights: {info['total_weights']}")
    print(f"Compression: {info['compression_ratio']:.2%}")
```

### `s3_store.py`

S3-compatible cloud storage backend.

**S3Config** (dataclass):
```python
S3Config(
    bucket: str,                    # S3 bucket name
    prefix: str = "coral/",         # Key prefix
    region: str = None,             # AWS region
    endpoint_url: str = None,       # For MinIO/custom endpoints
    access_key: str = None,         # AWS access key (optional)
    secret_key: str = None,         # AWS secret key (optional)
    max_concurrency: int = 10,      # Concurrent uploads/downloads
    compression: str = "gzip",      # "gzip", "lz4", or "none"
    chunk_size: int = 8*1024*1024   # 8MB multipart chunks
)
```

**S3Store** (class):

```python
from coral.storage.s3_store import S3Store, S3Config

# AWS S3
config = S3Config(bucket="my-models", region="us-east-1")
store = S3Store(config)

# MinIO
config = S3Config(
    bucket="my-models",
    endpoint_url="http://localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin"
)
store = S3Store(config)
```

**S3 Key Structure**:
```
{prefix}/weights/{hash}.npz    # Weight data
{prefix}/metadata/{hash}.json  # Metadata
{prefix}/deltas/{hash}.npz     # Delta data
```

**Features**:
- Automatic bucket creation if doesn't exist
- Concurrent batch uploads/downloads using ThreadPoolExecutor
- Automatic compression (gzip or lz4)
- Sync methods for local ↔ S3 transfers

**Sync Methods**:
```python
# Upload local weights to S3
stats = s3_store.sync_from_local(local_store)
print(f"Uploaded: {stats['uploaded']}, Skipped: {stats['skipped']}")

# Download S3 weights to local
stats = s3_store.sync_to_local(local_store)
print(f"Downloaded: {stats['downloaded']}")
```

**Dependencies**:
- Requires `boto3` (`pip install coral-ml[s3]`)
- Optional: `lz4` for LZ4 compression

## Compression Support

**HDF5Store Compression**:
| Type | Notes |
|------|-------|
| `gzip` | Best compression, adjustable level 1-9 |
| `lzf` | Faster, moderate compression |
| `None` | No compression |

**S3Store Compression**:
| Type | Notes |
|------|-------|
| `gzip` | Default, good compression |
| `lz4` | Faster, requires `lz4` package |
| `none` | No compression |

## Design Patterns

1. **Content-Addressable**: Weights stored by content hash
2. **Lazy Loading**: Only load data when accessed
3. **Batch Operations**: Efficient bulk store/load
4. **Context Manager**: Automatic resource cleanup
5. **Pluggable Backends**: Implement `WeightStore` interface

## Creating Custom Backends

```python
from coral.storage.weight_store import WeightStore

class MyCustomStore(WeightStore):
    def store(self, weight, hash_key=None):
        if hash_key is None:
            hash_key = weight.compute_hash()
        # Store weight data...
        return hash_key

    def load(self, hash_key):
        # Load and return WeightTensor...
        pass

    # Implement remaining abstract methods...
```

## Dependencies

- `h5py` - HDF5 file operations (HDF5Store)
- `boto3` - AWS SDK (S3Store, optional)
- `lz4` - LZ4 compression (S3Store, optional)
- Internal: `coral.core.weight_tensor`, `coral.delta.delta_encoder`

## Testing

Related test files:
- `tests/test_weight_store.py` - Abstract interface tests
- `tests/test_hdf5_store.py` - HDF5 backend tests
