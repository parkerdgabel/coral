# Remotes Module

This module provides remote repository management for syncing weights across storage backends, similar to git remotes.

## Overview

The remotes module provides:
- **RemoteConfig**: Configuration for remote storage backends
- **Remote**: Manages push/pull operations with a remote
- **RemoteManager**: Manages multiple remotes for a repository
- **SyncResult**: Statistics from sync operations

## Key Files

### `remote.py`

Remote repository management.

**RemoteConfig** (dataclass):
```python
RemoteConfig(
    name: str,                    # e.g., "origin"
    url: str,                     # e.g., "s3://bucket/prefix"
    backend: str = "s3",          # s3, gcs, azure, file

    # Optional credentials (usually from environment)
    access_key: str = None,
    secret_key: str = None,
    region: str = None,
    endpoint_url: str = None,     # For MinIO, etc.

    # Sync settings
    auto_push: bool = False,      # Auto-push on commit
    auto_pull: bool = False,      # Auto-pull before operations
)
```

**URL Parsing**:
```python
# Create from URL string
config = RemoteConfig.from_url("origin", "s3://my-bucket/models")
config = RemoteConfig.from_url("backup", "file:///backup/models")
config = RemoteConfig.from_url("minio", "minio://localhost:9000/bucket")
```

**Remote** (class):
```python
# Create remote from config
remote = Remote.from_config(config)

# Push weights to remote
result = remote.push(
    local_store,
    weight_hashes=None,  # None = push all missing
    force=False          # Overwrite existing
)
print(f"Pushed: {result.pushed_weights}")
print(f"Bytes: {result.bytes_transferred}")

# Pull weights from remote
result = remote.pull(
    local_store,
    weight_hashes=None,  # None = pull all missing
    force=False
)
print(f"Pulled: {result.pulled_weights}")

# Get remote info
info = remote.get_remote_info()
print(f"URL: {info['url']}")
print(f"Storage: {info['storage_info']}")

# List weights on remote
hashes = remote.list_remote_weights()
```

**SyncResult** (dataclass):
```python
SyncResult(
    pushed_weights: int = 0,
    pulled_weights: int = 0,
    pushed_commits: int = 0,
    pulled_commits: int = 0,
    bytes_transferred: int = 0,
    errors: list[str] = []
)

# Check success
if result.success:  # No errors
    print("Sync completed")
```

**RemoteManager** (class):
```python
manager = RemoteManager(config_path)

# Add remote
manager.add(RemoteConfig.from_url("origin", "s3://bucket/path"))

# List remotes
for name in manager.list():
    print(name)

# Get remote instance
remote = manager.get("origin")

# Remove remote
manager.remove("origin")
```

### `sync.py`

Sync operations (used by Repository class).

## Supported Backends

| Backend | URL Format | Notes |
|---------|------------|-------|
| S3 | `s3://bucket/prefix` | AWS S3 or S3-compatible |
| MinIO | `minio://host:port/bucket` | Converts to S3 with custom endpoint |
| File | `file:///path/to/dir` | Local filesystem |
| GCS | `gcs://bucket/prefix` | Future support |
| Azure | `azure://container/path` | Future support |

## Usage Examples

### Basic Push/Pull

```python
from coral.remotes.remote import RemoteConfig, Remote
from coral.storage.hdf5_store import HDF5Store

# Configure remote
config = RemoteConfig.from_url("origin", "s3://my-bucket/coral")

# Create remote
remote = Remote.from_config(config)

# Push local weights
with HDF5Store("./local/weights.h5") as local_store:
    result = remote.push(local_store)
    print(f"Pushed {result.pushed_weights} weights")

# Pull remote weights
with HDF5Store("./local/weights.h5") as local_store:
    result = remote.pull(local_store)
    print(f"Pulled {result.pulled_weights} weights")
```

### Using with Repository

```python
from coral import Repository

repo = Repository("./my-model")

# Add remote
repo.add_remote("origin", "s3://my-bucket/models")

# Check sync status
status = repo.get_sync_status("origin")
print(f"Need to push: {status['needs_push']}")
print(f"Need to pull: {status['needs_pull']}")

# Sync (bidirectional)
result = repo.sync("origin")
print(f"Pushed: {result['total_pushed']}")
print(f"Pulled: {result['total_pulled']}")

# Or push/pull separately
repo.push("origin")
repo.pull("origin")
```

### MinIO Configuration

```python
# MinIO with custom endpoint
config = RemoteConfig(
    name="minio",
    url="s3://my-bucket/coral",
    backend="s3",
    endpoint_url="http://localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
)

remote = Remote.from_config(config)
```

## Configuration Storage

Remotes are stored in `.coral/remotes.json`:
```json
{
  "remotes": {
    "origin": {
      "name": "origin",
      "url": "s3://my-bucket/coral",
      "backend": "s3",
      "endpoint_url": null,
      "auto_push": false,
      "auto_pull": false
    }
  }
}
```

Credentials are NOT stored in config file - use environment variables:
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`
- `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY`

## Dependencies

- Internal: `coral.storage.weight_store`, `coral.storage.s3_store`, `coral.storage.hdf5_store`

## Testing

Related test files:
- `tests/test_remotes.py` - Remote operations tests
