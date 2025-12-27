# Chapter 5: Version Control System

## Introduction

At the heart of Coral lies a sophisticated version control system that brings git-like capabilities to neural network weights. Just as git revolutionized source code management by treating code as a directed acyclic graph (DAG) of immutable snapshots, Coral applies the same principles to model weights, enabling systematic tracking, branching, and merging of machine learning models throughout their development lifecycle.

This chapter explores Coral's version control architecture, demonstrating how familiar git concepts map seamlessly to the unique challenges of weight management. You'll learn to leverage commits, branches, and merges to organize your ML experiments with the same rigor software engineers apply to code.

## 1. Git-Like Version Control Concept

### Why Version Control for Weights Matters

Machine learning development is inherently experimental. Data scientists train models with different hyperparameters, architectures, and datasets, creating numerous weight variations that need systematic organization. Without version control, teams face:

- **Lost experiments**: "Which weights produced that 95% accuracy?"
- **Collaboration chaos**: Multiple researchers modifying the same baseline model
- **Irreproducible results**: Inability to recreate past model states
- **Storage explosion**: Redundant copies of similar weights consuming disk space

Coral solves these problems by treating weights as first-class versioned artifacts.

### Mapping Git Concepts to ML Weights

The translation from git to Coral is remarkably direct:

| Git Concept | Coral Equivalent | Purpose |
|-------------|------------------|---------|
| File | WeightTensor | Individual weight arrays with metadata |
| Commit | Commit | Immutable snapshot of all model weights |
| Branch | Branch | Parallel experiment or development line |
| Tag | Version | Named milestone (e.g., "production-v1.2") |
| Merge | Merge | Combine weights from different experiments |
| Repository | Repository | Complete version control database |
| Staging Area | Staging | Weights prepared for next commit |

```
Git Workflow                    Coral Workflow
============                    ==============

1. Edit files                   1. Train model / modify weights
2. git add files                2. repo.stage_weights(weights)
3. git commit -m "msg"          3. repo.commit("msg")
4. git branch feature           4. repo.create_branch("feature")
5. git merge feature            5. repo.merge("feature")
6. git tag v1.0                 6. repo.tag_version("v1.0")
```

### Repository Structure: The .coral Directory

When you initialize a Coral repository, it creates a `.coral` directory analogous to `.git`:

```
project/
├── .coral/                      # Version control metadata
│   ├── config.json             # Repository configuration
│   ├── HEAD                    # Current branch pointer
│   ├── objects/                # Content-addressable storage
│   │   ├── weights.h5          # HDF5 weight storage
│   │   └── commits/            # Commit metadata
│   │       ├── a1b2c3d4.json  # Individual commit files
│   │       └── e5f6g7h8.json
│   ├── refs/                   # Branch references
│   │   └── heads/              # Branch pointers
│   │       ├── main
│   │       └── experiment-lr
│   ├── staging/                # Staged weights
│   │   └── staged.json
│   ├── versions/               # Tagged versions
│   │   └── v1.0.json
│   └── remotes.json           # Remote repository configs
└── train.py                    # Your training code
```

### The Version Graph (DAG)

Coral maintains commits as a directed acyclic graph (DAG), enabling complex branching and merging patterns:

```
        A---B---C (main)
             \
              D---E (experiment-lr)
```

Each node represents an immutable commit containing complete weight snapshots. The graph structure enables:

- **Linear history**: Follow a single development path
- **Branching**: Diverge for parallel experiments
- **Merging**: Combine results from multiple branches
- **Ancestry tracking**: Find common ancestors for conflict resolution

## 2. Repository - The Central API

The `Repository` class is your primary interface to Coral's version control system, providing methods for all weight management operations.

### Initialization and Configuration

**Creating a new repository:**

```python
from pathlib import Path
from coral.version_control import Repository

# Initialize new repository
repo = Repository(Path("./my_project"), init=True)

# Open existing repository
repo = Repository(Path("./my_project"))
```

This creates the `.coral` directory structure and initializes default configuration.

### Configuration Options

Coral repositories store configuration in `.coral/config.json`:

```json
{
  "user": {
    "name": "Jane Doe",
    "email": "jane@example.com"
  },
  "core": {
    "compression": "gzip",
    "similarity_threshold": 0.98,
    "delta_encoding": true,
    "delta_type": "float32_raw"
  }
}
```

**Key configuration parameters:**

- **user.name, user.email**: Default commit author information
- **core.compression**: HDF5 compression algorithm (`gzip`, `lzf`, `szip`)
- **core.similarity_threshold**: Deduplication threshold (0.0-1.0)
- **core.delta_encoding**: Enable lossless delta encoding for similar weights
- **core.delta_type**: Delta encoding strategy (`float32_raw`, `compressed`, `int8_quantized`, `sparse`)

### Directory Structure Deep Dive

**Objects Directory (`objects/`):**
- `weights.h5`: Content-addressable HDF5 store with weight data and deltas
- `commits/`: JSON files for each commit, named by commit hash

**Refs Directory (`refs/`):**
- `heads/`: Branch references (JSON files mapping branch names to commit hashes)

**Staging Directory (`staging/`):**
- `staged.json`: Temporarily stores weights prepared for commit

**HEAD File:**
- Contains current branch reference (e.g., `ref: refs/heads/main`)
- Or commit hash for detached HEAD state

### Key Methods Overview

The Repository class provides these essential operations:

```python
# Staging and committing
repo.stage_weights(weights_dict)
repo.commit(message, author, email, tags)

# Branch operations
repo.create_branch(name, from_ref)
repo.checkout(branch_or_commit)
repo.merge(source_branch, strategy)

# Weight operations
repo.get_weight(name, commit_ref)
repo.get_all_weights(commit_ref)

# History and inspection
repo.log(max_commits, branch)
repo.diff(from_ref, to_ref)

# Versioning
repo.tag_version(name, description, metrics)

# Remote operations
repo.add_remote(name, url)
repo.push(remote_name)
repo.pull(remote_name)
repo.sync(remote_name)
```

## 3. Commits - Immutable Snapshots

### Commit Class Structure

A `Commit` object represents an immutable snapshot of model weights at a specific point in time:

```python
@dataclass
class Commit:
    commit_hash: str                    # Unique identifier (SHA-256)
    parent_hashes: List[str]            # Parent commit(s)
    weight_hashes: Dict[str, str]       # name -> weight hash mapping
    metadata: CommitMetadata            # Author, message, timestamp
    delta_weights: Dict[str, str]       # name -> delta hash (for compression)
```

### Commit Hash Generation

Coral uses SHA-256 hashing to generate unique, collision-resistant commit identifiers:

```python
commit_content = {
    "parent_hashes": parent_hashes,
    "weight_hashes": weight_hashes,
    "metadata": metadata.to_dict(),
    "delta_weights": delta_weights,
}
commit_hash = hashlib.sha256(
    json.dumps(commit_content, sort_keys=True).encode()
).hexdigest()[:32]  # 128-bit hash
```

This ensures commits are content-addressable and tamper-evident.

### Parent Hashes: Single and Merge Commits

Commits track their lineage through parent hashes:

**Root commit** (no parents):
```python
commit = Commit(
    commit_hash="a1b2c3d4",
    parent_hashes=[],  # No parents
    weight_hashes={"layer1.weight": "h1", "layer1.bias": "h2"},
    metadata=CommitMetadata(...)
)
```

**Regular commit** (single parent):
```python
commit = Commit(
    commit_hash="e5f6g7h8",
    parent_hashes=["a1b2c3d4"],  # One parent
    weight_hashes={"layer1.weight": "h3", "layer1.bias": "h4"},
    metadata=CommitMetadata(...)
)
```

**Merge commit** (multiple parents):
```python
merge_commit = Commit(
    commit_hash="i9j0k1l2",
    parent_hashes=["e5f6g7h8", "m3n4o5p6"],  # Two parents
    weight_hashes={"layer1.weight": "h5", "layer1.bias": "h6"},
    metadata=CommitMetadata(..., tags=["merge"])
)

assert merge_commit.is_merge_commit  # True
```

### Weight Hashes Mapping

The `weight_hashes` dictionary maps weight names to their content hashes:

```python
{
    "encoder.layer1.weight": "xxhash_a1b2c3d4",
    "encoder.layer1.bias": "xxhash_e5f6g7h8",
    "decoder.layer1.weight": "xxhash_i9j0k1l2"
}
```

These hashes enable:
- **Content-addressable storage**: Same hash = same content
- **Deduplication**: Identical weights across commits share storage
- **Integrity verification**: Detect corruption by recomputing hashes

### Delta Weights for Lossless Compression

For similar weights, Coral stores deltas instead of full duplicates:

```python
commit.delta_weights = {
    "encoder.layer1.weight": "delta_hash_x1y2z3",  # Stored as delta
}
```

When this weight is loaded:
1. Load reference weight using `weight_hashes["encoder.layer1.weight"]`
2. Load delta using `delta_weights["encoder.layer1.weight"]`
3. Reconstruct original weight perfectly via delta decoding

This achieves 90-98% compression while maintaining perfect fidelity.

### CommitMetadata

Every commit includes rich metadata:

```python
@dataclass
class CommitMetadata:
    author: str                         # "Jane Doe"
    email: str                          # "jane@example.com"
    message: str                        # Commit description
    timestamp: datetime                 # Auto-generated
    tags: List[str]                     # ["baseline", "best-model"]
```

### Commit Operations

**Creating commits** (covered in Section 4)

**Reading commits:**
```python
# Get commit from version graph
commit = repo.version_graph.get_commit("a1b2c3d4")

print(f"Author: {commit.metadata.author}")
print(f"Message: {commit.metadata.message}")
print(f"Weights: {len(commit.weight_hashes)}")
```

**Traversing commit history:**
```python
# Get parent commit
parent_hash = commit.parent_hashes[0]
parent = repo.version_graph.get_commit(parent_hash)

# Get all ancestors
ancestors = repo.version_graph.get_commit_ancestors("a1b2c3d4")

# Get branch history (linear)
history = repo.version_graph.get_branch_history("a1b2c3d4", max_depth=10)
```

## 4. Staging and Committing Workflow

### The Staging Area Concept

Like git's staging area, Coral requires explicitly staging weights before committing:

```
Working State         Staging Area        Repository
=============         ============        ==========
(in memory)           (.coral/staging)    (.coral/objects)

Trained weights  -->  stage_weights() --> commit()
```

This two-step process allows you to:
- Review what will be committed
- Selectively commit subsets of weights
- Run validation before finalizing

### stage_weights() Method

The `stage_weights()` method prepares weights for commit:

```python
from coral.core.weight_tensor import WeightTensor, WeightMetadata
import numpy as np

# Prepare weights dictionary
weights = {
    "encoder.weight": WeightTensor(
        data=np.random.randn(128, 64).astype(np.float32),
        metadata=WeightMetadata(
            name="encoder.weight",
            shape=(128, 64),
            dtype=np.float32,
            layer_type="Linear"
        )
    ),
    "encoder.bias": WeightTensor(
        data=np.random.randn(128).astype(np.float32),
        metadata=WeightMetadata(
            name="encoder.bias",
            shape=(128,),
            dtype=np.float32
        )
    ),
}

# Stage weights
staged_hashes = repo.stage_weights(weights)
# Returns: {"encoder.weight": "hash1", "encoder.bias": "hash2"}
```

**What happens during staging:**

1. **Deduplication check**: Each weight is compared against existing weights using similarity threshold
2. **Delta encoding**: Similar weights stored as deltas from references
3. **Storage**: New unique weights written to HDF5 store
4. **Staging metadata**: Weight hashes and delta information saved to `staged.json`

### commit() Method

After staging, create an immutable commit:

```python
commit = repo.commit(
    message="Add encoder layer with 128 hidden units",
    author="Jane Doe",              # Optional, uses config default
    email="jane@example.com",       # Optional, uses config default
    tags=["baseline", "encoder-v1"] # Optional
)

print(f"Created commit: {commit.commit_hash}")
print(f"Parent: {commit.parent_hashes}")
```

### What Happens During a Commit

```
1. Load staged weights from staging/staged.json
   ↓
2. Get current branch HEAD commit (parent)
   ↓
3. Calculate additional delta encodings relative to parent
   ↓
4. Create CommitMetadata with author, timestamp, message
   ↓
5. Generate commit hash from content
   ↓
6. Create Commit object with all metadata
   ↓
7. Save commit to objects/commits/{hash}.json
   ↓
8. Update version graph
   ↓
9. Update current branch to point to new commit
   ↓
10. Clear staging area
```

### Complete Example: Training and Committing

```python
import torch
import torch.nn as nn
from coral.version_control import Repository
from coral.integrations.pytorch import pytorch_to_coral

# Initialize repository
repo = Repository("./experiments", init=True)

# Train model
model = nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# ... training code ...

# Convert PyTorch weights to Coral
weights = pytorch_to_coral(model.state_dict())

# Stage and commit
repo.stage_weights(weights)
commit = repo.commit(
    message="Initial baseline: 2-layer MLP, lr=0.001",
    tags=["baseline", "mlp"]
)

print(f"Committed {len(weights)} weights")
print(f"Commit hash: {commit.commit_hash}")
```

## 5. Branches - Parallel Development

### Branch Class and BranchManager

Branches are lightweight pointers to commits, stored as JSON files in `.coral/refs/heads/`:

```python
@dataclass
class Branch:
    name: str                   # "main", "experiment-lr"
    commit_hash: str            # Points to tip commit
    parent_branch: Optional[str] # Track branch origin
```

The `BranchManager` class handles all branch operations:

```python
# Access through repository
manager = repo.branch_manager
```

### Creating Branches

**Create branch from current HEAD:**
```python
# On main branch with commit "abc123"
repo.create_branch("experiment-dropout")

# Creates branch pointing to same commit as main
```

**Create branch from specific reference:**
```python
# Create from specific commit
repo.create_branch("hotfix", from_ref="abc123")

# Create from another branch
repo.create_branch("feature-v2", from_ref="experiment-lr")
```

**Branch creation workflow:**
```
main:     A---B---C (HEAD)
                   \
experiment:         C (new branch, same commit)
```

### Switching Branches (Checkout)

The `checkout()` method switches your working context:

```python
# Switch to existing branch
repo.checkout("experiment-dropout")
print(repo.branch_manager.get_current_branch())  # "experiment-dropout"

# Subsequent commits affect this branch
repo.stage_weights(modified_weights)
repo.commit("Increase dropout to 0.5")
```

**Checkout to specific commit (detached HEAD):**
```python
# Checkout historical commit
repo.checkout("abc123")  # Detached HEAD state

# HEAD file contains: "abc123" (not "ref: refs/heads/...")
```

### Listing Branches

```python
# Get all branches
branches = repo.branch_manager.list_branches()

for branch in branches:
    current = "* " if branch.name == repo.branch_manager.get_current_branch() else "  "
    print(f"{current}{branch.name} -> {branch.commit_hash[:8]}")

# Output:
#   experiment-lr -> e5f6g7h8
# * main -> a1b2c3d4
```

### Deleting Branches

```python
# Cannot delete current branch
try:
    repo.branch_manager.delete_branch("main")  # Fails if on main
except ValueError as e:
    print(e)  # "Cannot delete current branch"

# Switch first, then delete
repo.checkout("main")
repo.branch_manager.delete_branch("experiment-lr")
```

### The HEAD Concept

HEAD indicates your current position in the version graph:

**Normal state (attached HEAD):**
```
HEAD -> refs/heads/main -> commit abc123
```

**Detached HEAD state:**
```
HEAD -> commit abc123 (no branch)
```

Check HEAD state:
```python
with open(repo.coral_dir / "HEAD") as f:
    head_content = f.read()

if head_content.startswith("ref:"):
    print(f"On branch: {head_content.split('/')[-1]}")
else:
    print(f"Detached HEAD at: {head_content}")
```

### Parent Branch Tracking

Branches remember their origin:

```python
# Create feature branch from main
repo.checkout("main")
repo.create_branch("feature-attention")

# Branch tracks parent
branch = repo.branch_manager.get_branch("feature-attention")
print(branch.parent_branch)  # None (not currently tracked)

# Useful for visualizing branch relationships
```

## 6. Merging - Combining Work

### MergeStrategy Enum

Coral provides five strategies for resolving merge conflicts:

```python
from coral.version_control import MergeStrategy

class MergeStrategy(Enum):
    OURS = "ours"           # Keep current branch weights
    THEIRS = "theirs"       # Take source branch weights
    FAIL = "fail"           # Raise error on conflicts
    AVERAGE = "average"     # Average: (ours + theirs) / 2
    WEIGHTED = "weighted"   # Weighted: α*ours + (1-α)*theirs
```

### Three-Way Merge Algorithm

Coral uses three-way merging with a common ancestor:

```
Common Ancestor (A):  layer1.weight = [1, 2, 3]
                           /              \
Current Branch (B):   [1, 2, 4]      [1, 5, 3]  Source Branch (C)
                           \              /
Merged Result (D):         ? ? ?
```

**Merge decision tree:**

1. **No conflict**: Both branches have identical weight → Use either
2. **Only current changed**: Current ≠ Ancestor, Source = Ancestor → Use current
3. **Only source changed**: Source ≠ Ancestor, Current = Ancestor → Use source
4. **Both changed (conflict)**: Apply merge strategy

### Merge Commits (Multiple Parents)

Merge commits record multiple parent hashes:

```python
repo.checkout("main")
merge_commit = repo.merge("experiment-lr", message="Merge improved learning rate")

print(merge_commit.parent_hashes)
# ["main_tip_hash", "experiment_tip_hash"]

assert merge_commit.is_merge_commit  # True
```

**Version graph after merge:**
```
main:          A---B---D (merge commit)
                    \  /
experiment-lr:       C
```

### Conflict Detection

Conflicts occur when both branches modify the same weight differently:

```python
# Current: layer1.weight = [1, 2, 4]
# Source:  layer1.weight = [1, 5, 3]
# Ancestor: layer1.weight = [1, 2, 3]

# Both changed layer1.weight → CONFLICT!
```

### Conflict Resolution Examples

**Strategy: OURS** (default for safety)
```python
merge_commit = repo.merge(
    "experiment-dropout",
    strategy=MergeStrategy.OURS
)
# Keeps current branch weights for conflicts
```

**Strategy: THEIRS**
```python
merge_commit = repo.merge(
    "experiment-dropout",
    strategy=MergeStrategy.THEIRS
)
# Takes source branch weights for conflicts
```

**Strategy: FAIL** (manual resolution required)
```python
from coral.version_control import MergeConflictError

try:
    merge_commit = repo.merge(
        "experiment-dropout",
        strategy=MergeStrategy.FAIL
    )
except MergeConflictError as e:
    print(f"Conflicts in: {e.conflicts}")
    # ["layer1.weight", "layer2.bias"]
    # Manual resolution needed
```

**Strategy: AVERAGE** (neural network specific)
```python
merge_commit = repo.merge(
    "experiment-ensemble",
    strategy=MergeStrategy.AVERAGE
)
# Merged weight = (current + source) / 2
# Useful for model ensembling!
```

**Strategy: WEIGHTED** (custom blending)
```python
merge_commit = repo.merge(
    "experiment-fine-tuned",
    strategy=MergeStrategy.WEIGHTED,
    merge_alpha=0.7  # 70% current, 30% source
)
# Merged weight = 0.7 * current + 0.3 * source
# Useful for interpolating between models
```

### Fast-Forward Merges

When source branch is ahead of current branch with no divergence:

```
main:      A---B (current)
                \
feature:         C---D (source, ahead)

# Fast-forward main to D (no merge commit needed)
repo.checkout("main")
repo.merge("feature")  # Fast-forward

main:      A---B---C---D (current, no merge commit)
```

## 7. Tags and Versions

### tag_version() Method

Tags create named, immutable references to commits:

```python
version = repo.tag_version(
    name="v1.0.0",
    description="Production release - CNN classifier",
    metrics={
        "train_accuracy": 0.987,
        "val_accuracy": 0.952,
        "test_accuracy": 0.948,
        "loss": 0.124
    },
    commit_ref=None  # None = current HEAD
)
```

### Version Class with Metrics

```python
@dataclass
class Version:
    version_id: str                     # Unique ID (hash of name:commit)
    commit_hash: str                    # Points to specific commit
    name: str                           # Human-readable name
    description: Optional[str]          # Detailed description
    metrics: Optional[Dict[str, float]] # Performance metrics
```

### Finding Tagged Versions

```python
# Load version from file
version_file = repo.coral_dir / "versions" / f"{version.version_id}.json"
with open(version_file) as f:
    version_data = json.load(f)

restored_version = Version.from_dict(version_data)

# Get weights for this version
weights = repo.get_all_weights(commit_ref=restored_version.commit_hash)
```

### Use Cases for Tagging

**1. Production deployments:**
```python
repo.tag_version(
    name="production-v2.1",
    description="Deployed to EU region on 2024-01-15",
    metrics={"latency_ms": 45, "accuracy": 0.96}
)
```

**2. Experiment milestones:**
```python
repo.tag_version(
    name="exp-42-best",
    description="Best result from hyperparameter sweep",
    metrics={"loss": 0.089, "f1_score": 0.923}
)
```

**3. Paper submissions:**
```python
repo.tag_version(
    name="icml-2024-submission",
    description="Model weights for ICML 2024 paper",
    metrics={"test_acc": 0.954}
)
```

**4. Regulatory compliance:**
```python
repo.tag_version(
    name="audit-2024-q1",
    description="Model audited and approved for medical use",
    metrics={"sensitivity": 0.98, "specificity": 0.95}
)
```

## 8. History and Diffing

### log() - Viewing Commit History

The `log()` method retrieves commit history similar to `git log`:

```python
# Get recent commits on current branch
commits = repo.log(max_commits=5)

for commit in commits:
    print(f"Commit: {commit.commit_hash}")
    print(f"Author: {commit.metadata.author}")
    print(f"Date: {commit.metadata.timestamp}")
    print(f"Message: {commit.metadata.message}")
    print()

# Output:
# Commit: e5f6g7h8
# Author: Jane Doe
# Date: 2024-01-15 14:23:45
# Message: Increase dropout to 0.5
# ...
```

**Get history for specific branch:**
```python
main_history = repo.log(max_commits=10, branch="main")
feature_history = repo.log(max_commits=10, branch="experiment-lr")
```

### diff() - Comparing Commits

The `diff()` method shows changes between two commits:

```python
diff_result = repo.diff(
    from_ref="abc123",  # Older commit
    to_ref="e5f6g7h8"   # Newer commit
)

print(f"Added weights: {diff_result['added']}")
# ["layer3.weight", "layer3.bias"]

print(f"Removed weights: {diff_result['removed']}")
# ["old_layer.weight"]

print(f"Modified weights: {diff_result['modified']}")
# {"layer1.weight": {"from_hash": "h1", "to_hash": "h2"}}

print(f"Summary: {diff_result['summary']}")
# {"total_added": 2, "total_removed": 1, "total_modified": 1}
```

**Compare with current HEAD:**
```python
diff = repo.diff(from_ref="abc123")  # to_ref defaults to current HEAD
```

### get_common_ancestor() - Finding Merge Base

Find the common ancestor of two branches for merge analysis:

```python
ancestor_hash = repo.version_graph.get_common_ancestor(
    commit1_hash="main_tip",
    commit2_hash="feature_tip"
)

ancestor_commit = repo.version_graph.get_commit(ancestor_hash)
print(f"Branches diverged at: {ancestor_commit.metadata.message}")
```

**Visualizing divergence:**
```python
# Get commits unique to each branch
main_commits = repo.log(branch="main")
feature_commits = repo.log(branch="feature")

common = repo.version_graph.get_common_ancestor(
    main_commits[0].commit_hash,
    feature_commits[0].commit_hash
)

print(f"Diverged {len([c for c in main_commits if c.commit_hash != common])} commits on main")
print(f"Diverged {len([c for c in feature_commits if c.commit_hash != common])} commits on feature")
```

### VersionGraph for Navigation

The `VersionGraph` class provides powerful graph traversal:

```python
graph = repo.version_graph

# Get all ancestors (full history)
all_ancestors = graph.get_commit_ancestors("current_hash")

# Get all descendants (future commits)
descendants = graph.get_commit_descendants("old_hash")

# Get path between commits
path = graph.get_commit_path(from_hash="A", to_hash="D")
# Returns: ["A", "B", "C", "D"]

# Track weight evolution
weight_history = graph.get_weight_history(
    weight_name="encoder.weight",
    from_commit="current_hash"
)
# Returns: [("hash1", "weight_hash1"), ("hash2", "weight_hash2"), ...]
```

## 9. Weight Operations

### get_weight() - Retrieving Individual Weights

Load a specific weight from a commit:

```python
# Get from current HEAD
weight = repo.get_weight("encoder.layer1.weight")

# Get from specific commit
weight = repo.get_weight(
    name="encoder.layer1.weight",
    commit_ref="abc123"
)

# Get from tagged version
version = repo.version_graph.versions["version_id"]
weight = repo.get_weight(
    name="encoder.layer1.weight",
    commit_ref=version.commit_hash
)

print(f"Shape: {weight.shape}")
print(f"Dtype: {weight.dtype}")
print(f"Data: {weight.data}")
```

### get_all_weights() - Retrieving Complete Snapshots

Load all weights from a commit:

```python
# Get all weights from current HEAD
weights = repo.get_all_weights()

# Get from specific commit
weights = repo.get_all_weights(commit_ref="abc123")

# Inspect loaded weights
for name, weight in weights.items():
    print(f"{name}: {weight.shape} ({weight.dtype})")

# Output:
# encoder.layer1.weight: (128, 64) (float32)
# encoder.layer1.bias: (128,) (float32)
# decoder.layer1.weight: (64, 128) (float32)
```

### Delta Reconstruction (Automatic)

Coral automatically reconstructs delta-encoded weights transparently:

```python
# Weight stored as delta in commit
commit.delta_weights = {"encoder.weight": "delta_hash_xyz"}

# Loading reconstructs perfectly
weight = repo.get_weight("encoder.weight", commit_ref=commit.commit_hash)

# Original data restored exactly (lossless)
assert weight.data.shape == (128, 64)  # Correct shape
# No information loss!
```

**Behind the scenes:**
1. Check if weight has delta encoding in commit
2. Load delta object from storage
3. Load reference weight from storage
4. Reconstruct original via `delta_encoder.decode_delta()`
5. Return perfect reconstruction

### Loading from Specific Commits/Branches

```python
# Load from branch tip
repo.checkout("experiment-lr")
weights = repo.get_all_weights()  # Uses current branch HEAD

# Load from historical commit
old_weights = repo.get_all_weights(commit_ref="old_commit_hash")

# Compare weight evolution
current_weight = repo.get_weight("layer1.weight")
old_weight = repo.get_weight("layer1.weight", commit_ref="old_commit_hash")

difference = np.abs(current_weight.data - old_weight.data).mean()
print(f"Average weight change: {difference}")
```

## 10. Remote Operations

Coral supports pushing and pulling weights to remote storage backends, enabling team collaboration and backup.

### Configuring Remotes

**Add remote repository:**
```python
# S3 remote
repo.add_remote(
    name="origin",
    url="s3://my-bucket/coral-models/"
)

# MinIO remote
repo.add_remote(
    name="backup",
    url="minio://minio.example.com/models/"
)

# Local file remote (for testing)
repo.add_remote(
    name="local-backup",
    url="file:///backup/coral-repo"
)

# List configured remotes
remotes = repo.list_remotes()
for name, config in remotes.items():
    print(f"{name}: {config['url']} ({config['backend']})")
```

**Remove remote:**
```python
repo.remove_remote("old-backup")
```

### push() - Uploading Weights

Push local weights to remote storage:

```python
def progress_callback(current, total, bytes_transferred, hash_key):
    print(f"Pushing {current}/{total}: {hash_key[:8]} ({bytes_transferred} bytes)")

result = repo.push(
    remote_name="origin",
    force=False,  # Don't overwrite existing weights
    progress_callback=progress_callback
)

print(f"Pushed: {result['weights_pushed']} weights")
print(f"Transferred: {result['bytes_transferred']} bytes")
print(f"Skipped: {result['skipped']} (already on remote)")
```

### pull() - Downloading Weights

Pull weights from remote storage:

```python
result = repo.pull(
    remote_name="origin",
    force=False,  # Don't overwrite local weights
    progress_callback=progress_callback
)

print(f"Pulled: {result['weights_pulled']} weights")
print(f"Transferred: {result['bytes_transferred']} bytes")
```

### sync() - Bidirectional Synchronization

Synchronize local and remote in both directions:

```python
result = repo.sync(
    remote_name="origin",
    progress_callback=progress_callback
)

print("Push results:", result['push'])
print("Pull results:", result['pull'])
print(f"Total pushed: {result['total_pushed']}")
print(f"Total pulled: {result['total_pulled']}")
print(f"Synced: {result['is_synced']}")
```

### get_sync_status() - Checking Synchronization

Check what would be pushed/pulled without transferring:

```python
status = repo.get_sync_status("origin")

print(f"Local-only weights: {status['needs_push']}")
print(f"Remote-only weights: {status['needs_pull']}")
print(f"Synchronized: {status['is_synced']}")

if not status['is_synced']:
    print(f"Ahead by: {status['needs_push']} weights")
    print(f"Behind by: {status['needs_pull']} weights")
```

### Incremental Sync (Efficient)

For large repositories, use incremental operations:

```python
# Push only new weights
push_result = repo.incremental_push("origin")
print(f"Incremental push: {push_result['weights_pushed']} new weights")

# Pull only new weights
pull_result = repo.incremental_pull("origin")
print(f"Incremental pull: {pull_result['weights_pulled']} new weights")
```

### Clone Operations

Clone a remote repository (conceptual):

```python
# Initialize new repository
repo = Repository("./cloned-repo", init=True)

# Add remote
repo.add_remote("origin", "s3://shared-bucket/models/")

# Pull all weights
repo.pull("origin")

# Now have complete copy of remote repository
```

## Conclusion

Coral's version control system brings software engineering best practices to machine learning weight management. By treating weights as first-class versioned artifacts with git-like operations, Coral enables:

- **Reproducibility**: Every model state is immutably recorded with full lineage
- **Collaboration**: Multiple researchers work on parallel branches with structured merging
- **Experimentation**: Branch freely for experiments without fear of losing work
- **Efficiency**: Delta encoding and deduplication minimize storage overhead
- **Production-ready**: Tag stable versions, track deployments, and maintain audit trails

The familiar git workflow—branch, commit, merge, tag—maps seamlessly to neural network weights, making Coral intuitive for anyone with version control experience. Combined with automatic deduplication and lossless delta encoding, Coral provides industrial-strength weight versioning that scales from research prototypes to production ML systems.

In the next chapter, we'll explore Coral's training integration, showing how to automatically checkpoint models during training with configurable policies and seamless version control integration.
