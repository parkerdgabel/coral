# Version Control Module

This module provides git-like version control for neural network weights, including repository management, branching, committing, merging, and remote synchronization.

## Overview

The version control module provides:
- **Repository**: Main class for all version control operations
- **BranchManager**: Branch creation, checkout, deletion
- **Commit**: Immutable commit objects with metadata
- **VersionGraph**: DAG for version history tracking

## Key Files

### `repository.py`

The main Repository class that coordinates all version control operations.

**MergeStrategy** (Enum):
```python
OURS = "ours"       # Prefer weights from current branch
THEIRS = "theirs"   # Prefer weights from source branch
FAIL = "fail"       # Raise MergeConflictError on conflicts
AVERAGE = "average" # Average: (ours + theirs) / 2
WEIGHTED = "weighted" # Weighted: alpha * ours + (1-alpha) * theirs
```

**MergeConflictError**: Exception raised when merge conflicts occur with `FAIL` strategy.

**Repository** (class):

Constructor:
```python
Repository(
    path: Path,     # Repository root path
    init: bool = False  # Initialize new repository
)
```

**Repository Structure**:
```
.coral/
├── config.json       # Repository configuration
├── HEAD              # Current branch reference
├── objects/
│   ├── commits/      # Commit objects (JSON)
│   └── weights.h5    # Weight storage (HDF5)
├── refs/
│   └── heads/        # Branch references
├── staging/
│   └── staged.json   # Staged weights
├── versions/         # Named versions
├── remotes.json      # Remote configurations
└── sync/             # Sync state files
```

**Core Methods**:
```python
# Staging and Commits
repo.stage_weights(weights: dict[str, WeightTensor]) -> dict[str, str]
repo.commit(message, author=None, email=None, tags=None) -> Commit

# Branching
repo.create_branch(name, from_ref=None)
repo.checkout(target)  # Branch name or commit hash
repo.merge(source_branch, message=None, strategy=MergeStrategy.OURS, merge_alpha=0.5) -> Commit

# Weight Access
repo.get_weight(name, commit_ref=None) -> Optional[WeightTensor]
repo.get_all_weights(commit_ref=None) -> dict[str, WeightTensor]

# History
repo.log(max_commits=10, branch=None) -> list[Commit]
repo.diff(from_ref, to_ref=None) -> dict
repo.tag_version(name, description=None, metrics=None, commit_ref=None) -> Version

# Maintenance
repo.gc() -> dict[str, int]  # Garbage collect unreferenced weights
```

**Remote Operations**:
```python
# Remote Management
repo.add_remote(name, url)
repo.remove_remote(name)
repo.list_remotes() -> dict[str, dict]
repo.get_remote(name) -> Optional[dict]

# Sync Operations
repo.push(remote_name, force=False, progress_callback=None) -> dict
repo.pull(remote_name, force=False, progress_callback=None) -> dict
repo.sync(remote_name, progress_callback=None) -> dict

# Incremental Sync (more efficient)
repo.incremental_push(remote_name, progress_callback=None) -> dict
repo.incremental_pull(remote_name, progress_callback=None) -> dict
repo.get_sync_status(remote_name) -> dict
```

**Configuration** (`config.json`):
```json
{
  "user": {"name": "Anonymous", "email": "anonymous@example.com"},
  "core": {
    "compression": "gzip",
    "similarity_threshold": 0.98,
    "delta_encoding": true
  }
}
```

### `branch.py`

Branch management.

**Branch** (dataclass):
- `name`: Branch name
- `commit_hash`: Current commit hash
- `parent_branch`: Optional parent branch name

**BranchManager** (class):
```python
manager.create_branch(name, commit_hash, parent_branch=None) -> Branch
manager.get_branch(name) -> Optional[Branch]
manager.update_branch(name, commit_hash)
manager.delete_branch(name)
manager.list_branches() -> list[Branch]
manager.branch_exists(name) -> bool
manager.get_current_branch() -> str
manager.set_current_branch(name)
manager.get_branch_commit(name) -> Optional[str]
```

### `commit.py`

Immutable commit objects.

**CommitMetadata** (dataclass):
- `author`: Author name
- `email`: Author email
- `message`: Commit message
- `timestamp`: Datetime (auto-generated)
- `tags`: List of tags

**Commit** (class):
```python
Commit(
    commit_hash: str,
    parent_hashes: list[str],
    weight_hashes: dict[str, str],  # name -> hash
    metadata: CommitMetadata,
    delta_weights: dict[str, str] = {}  # name -> delta hash
)
```

**Properties**:
- `is_merge_commit`: True if >1 parent
- `is_root_commit`: True if no parents

**Methods**:
```python
commit.get_changed_weights(parent) -> dict[str, str]
commit.get_added_weights(parent) -> set[str]
commit.get_removed_weights(parent) -> set[str]
commit.save(path)
Commit.load(path) -> Commit
```

### `version.py`

Version graph and named versions.

**Version** (dataclass):
- `version_id`: Unique identifier
- `commit_hash`: Associated commit
- `name`: Human-readable name (e.g., "v1.0")
- `description`: Optional description
- `metrics`: Optional dict of metrics (e.g., `{"accuracy": 0.95}`)

**VersionGraph** (class):

Uses NetworkX DiGraph for efficient graph operations.

```python
graph.add_commit(commit)
graph.add_version(version)
graph.get_commit(commit_hash) -> Optional[Commit]
graph.get_version(version_id) -> Optional[Version]

# Graph Operations
graph.get_commit_ancestors(commit_hash) -> list[str]
graph.get_commit_descendants(commit_hash) -> list[str]
graph.get_common_ancestor(commit1, commit2) -> Optional[str]
graph.get_commit_path(from_hash, to_hash) -> Optional[list[str]]
graph.get_branch_history(tip_hash, max_depth=None) -> list[str]
graph.get_divergence_point(branch1_tip, branch2_tip) -> Optional[str]

# Weight History
graph.get_weight_history(weight_name, from_commit) -> list[tuple[str, str]]
graph.find_commits_with_weight(weight_hash) -> list[str]
```

## Merge Strategies

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| `OURS` | Take current branch weights | Default, safe |
| `THEIRS` | Take source branch weights | Overwrite with source |
| `FAIL` | Raise exception | Manual resolution |
| `AVERAGE` | `(ours + theirs) / 2` | Combine training runs |
| `WEIGHTED` | `α*ours + (1-α)*theirs` | Controlled blending |

**Example**:
```python
# Merge with weight averaging
commit = repo.merge(
    "experiment",
    strategy=MergeStrategy.WEIGHTED,
    merge_alpha=0.7  # 70% current, 30% source
)
```

## Usage Examples

### Basic Workflow

```python
from coral.version_control import Repository
from coral.core import WeightTensor

# Initialize repository
repo = Repository("./my_model", init=True)

# Stage and commit weights
weights = {"layer1.weight": weight1, "layer1.bias": bias1}
repo.stage_weights(weights)
commit = repo.commit("Initial model weights")

# Create branch for experiment
repo.create_branch("experiment")
repo.checkout("experiment")

# ... modify and commit more weights ...

# Merge back to main
repo.checkout("main")
repo.merge("experiment", strategy=MergeStrategy.AVERAGE)
```

### Remote Sync

```python
# Add remote
repo.add_remote("origin", "s3://my-bucket/coral")

# Check sync status
status = repo.get_sync_status("origin")
print(f"Need to push: {status['needs_push']}")
print(f"Need to pull: {status['needs_pull']}")

# Sync (bidirectional)
result = repo.sync("origin")
print(f"Pushed: {result['total_pushed']}, Pulled: {result['total_pulled']}")
```

### Version Tagging

```python
# Tag current commit as a version
version = repo.tag_version(
    "v1.0",
    description="Production model",
    metrics={"accuracy": 0.95, "loss": 0.05}
)

# Access version later
weights = repo.get_all_weights(version.commit_hash)
```

## Delta Encoding Integration

The repository automatically handles delta encoding:

1. When staging weights, the deduplicator checks for similar existing weights
2. If a weight is similar (>98% by default), it stores a delta instead
3. Commits track delta references in `delta_weights` field
4. When loading weights, deltas are automatically reconstructed

## Dependencies

- `networkx` - Version graph operations
- Internal: `coral.core.deduplicator`, `coral.delta.delta_encoder`, `coral.storage.hdf5_store`

## Testing

Related test files:
- `tests/test_version_control.py` - Core version control features
- `tests/test_repository_coverage.py` - Repository class coverage
- `tests/test_repository_extended.py` - Extended repository tests
