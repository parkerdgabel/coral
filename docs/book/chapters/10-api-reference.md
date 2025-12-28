# Chapter 10: API Reference

This chapter provides comprehensive documentation for all public APIs in the Coral system. Use this as a complete reference guide for developing applications with Coral.

## Package Structure

### Main Package Imports

```python
from coral import (
    WeightTensor,       # Core weight data structure
    Deduplicator,       # Deduplication engine
    WeightStore,        # Abstract storage interface
    HDF5Store,          # HDF5 storage backend
    Repository,         # Version control repository
    DeltaEncoder,       # Delta encoding engine
    DeltaConfig,        # Delta configuration
    DeltaType,          # Delta encoding strategies
    DeltaReconstructionError,  # Delta decoding error
)
```

### Subpackage Organization

- **`coral.core`** - Core abstractions (WeightTensor, Deduplicator, SimHash)
- **`coral.delta`** - Delta encoding for similar weights
- **`coral.storage`** - Storage backends (HDF5, S3)
- **`coral.version_control`** - Git-like version control (Repository, Commit, Branch)
- **`coral.training`** - Training integration (CheckpointManager, TrainingState)
- **`coral.integrations`** - Framework integrations (PyTorch, TensorFlow, HuggingFace)
- **`coral.experiments`** - Experiment tracking
- **`coral.registry`** - Model registry publishing
- **`coral.compression`** - Compression techniques (quantization, pruning)
- **`coral.utils`** - Utility functions (similarity, visualization, JSON)

---

## Core Module API

### WeightTensor Class

The fundamental data structure representing neural network weights with metadata.

#### Constructor

```python
WeightTensor(
    data: Optional[np.ndarray] = None,
    metadata: Optional[WeightMetadata] = None,
    store_ref: Optional[str] = None
)
```

**Parameters:**
- `data`: The actual weight data as numpy array
- `metadata`: Metadata about the weight tensor
- `store_ref`: Reference to data in storage (for lazy loading)

#### Properties

```python
@property
def data(self) -> np.ndarray:
    """Get the weight data, loading from storage if necessary."""

@property
def metadata(self) -> WeightMetadata:
    """Get the weight metadata."""

@property
def shape(self) -> Tuple[int, ...]:
    """Get the shape of the weight tensor."""

@property
def dtype(self) -> np.dtype:
    """Get the data type of the weight tensor."""

@property
def size(self) -> int:
    """Get the total number of elements."""

@property
def nbytes(self) -> int:
    """Get the number of bytes used by the tensor."""
```

#### Methods

```python
def compute_hash(self, force: bool = False) -> str:
    """
    Compute content-based hash of the weight tensor.

    Args:
        force: If True, recompute hash even if cached

    Returns:
        Hexadecimal hash string (xxHash)
    """

def is_similar_to(self, other: WeightTensor, threshold: float = 0.99) -> bool:
    """
    Check if this weight tensor is similar to another using cosine similarity.

    Args:
        other: Another WeightTensor to compare with
        threshold: Similarity threshold (0-1)

    Returns:
        True if similarity exceeds threshold
    """

def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary for serialization."""

@classmethod
def from_dict(cls, data: Dict[str, Any], weight_data: Optional[np.ndarray] = None) -> WeightTensor:
    """Create WeightTensor from dictionary."""
```

#### Example Usage

```python
import numpy as np
from coral import WeightTensor
from coral.core.weight_tensor import WeightMetadata

# Create weight tensor
data = np.random.randn(128, 256).astype(np.float32)
metadata = WeightMetadata(
    name="layer1.weight",
    shape=data.shape,
    dtype=data.dtype,
    layer_type="Linear",
    model_name="MyModel"
)
weight = WeightTensor(data=data, metadata=metadata)

# Compute hash
hash_val = weight.compute_hash()
print(f"Weight hash: {hash_val}")

# Check similarity
other_data = data + np.random.randn(*data.shape) * 0.01
other_weight = WeightTensor(data=other_data, metadata=metadata)
is_similar = weight.is_similar_to(other_weight, threshold=0.98)
print(f"Weights similar: {is_similar}")
```

### WeightMetadata Dataclass

```python
@dataclass
class WeightMetadata:
    """Metadata associated with a weight tensor."""

    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    layer_type: Optional[str] = None
    model_name: Optional[str] = None
    compression_info: Dict[str, Any] = field(default_factory=dict)
    hash: Optional[str] = None
```

### Deduplicator Class

The core deduplication engine that identifies and eliminates duplicate/similar weights with lossless delta encoding.

#### Constructor

```python
Deduplicator(
    similarity_threshold: float = 0.99,
    delta_config: Optional[DeltaConfig] = None,
    enable_delta_encoding: bool = True,
    enable_lsh: bool = False,
    lsh_config: Optional[LSHConfig] = None,
    magnitude_tolerance: float = 0.1
)
```

**Parameters:**
- `similarity_threshold`: Threshold for considering weights similar (0-1)
- `delta_config`: Configuration for delta encoding
- `enable_delta_encoding`: Whether to use delta encoding for similar weights
- `enable_lsh`: Enable LSH for O(1) similarity lookup (for >10k weights)
- `lsh_config`: Configuration for LSH index
- `magnitude_tolerance`: Maximum relative magnitude difference (e.g., 0.1 = 10%)

#### Methods

```python
def add_weight(self, weight: WeightTensor, name: Optional[str] = None) -> str:
    """
    Add a weight to the deduplicator and check for duplicates.
    Thread-safe.

    Args:
        weight: WeightTensor to add
        name: Optional name for the weight

    Returns:
        Hash of the weight (or reference weight if duplicate/similar)
    """

def get_weight_by_name(self, name: str) -> Optional[WeightTensor]:
    """
    Get weight by name, reconstructing from delta if needed.
    Thread-safe.
    """

def get_weight_group(self, name: str) -> Optional[WeightGroup]:
    """Get the weight group containing a named weight."""

def compute_stats(self) -> DeduplicationStats:
    """
    Compute and return deduplication statistics.
    Thread-safe.
    """

def get_deduplication_report(self) -> Dict[str, Any]:
    """Get detailed deduplication report with largest groups."""

def clear(self):
    """
    Clear all stored weights and statistics.
    Thread-safe.
    """

def is_delta_encoded(self, name: str) -> bool:
    """Check if a weight is delta-encoded."""

def get_delta_by_name(self, name: str) -> Optional[Delta]:
    """Get delta object by weight name."""

def get_compression_stats(self) -> Dict[str, Any]:
    """Get detailed compression statistics including delta encoding."""
```

#### Example Usage

```python
from coral import Deduplicator, DeltaConfig, DeltaType

# Create deduplicator with delta encoding
config = DeltaConfig(delta_type=DeltaType.COMPRESSED)
dedup = Deduplicator(
    similarity_threshold=0.98,
    delta_config=config,
    enable_delta_encoding=True
)

# Add weights
for name, weight in weights.items():
    ref_hash = dedup.add_weight(weight, name)
    print(f"{name} -> {ref_hash}")

# Get statistics
stats = dedup.compute_stats()
print(f"Unique weights: {stats.unique_weights}")
print(f"Duplicates: {stats.duplicate_weights}")
print(f"Similar weights: {stats.similar_weights}")
print(f"Space saved: {stats.bytes_saved} bytes ({stats.compression_ratio:.2%})")

# Retrieve weight (automatically reconstructs from delta)
retrieved = dedup.get_weight_by_name("layer1.weight")
```

### DeduplicationStats Dataclass

```python
@dataclass
class DeduplicationStats:
    """Statistics about deduplication results."""

    total_weights: int = 0
    unique_weights: int = 0
    duplicate_weights: int = 0
    similar_weights: int = 0
    bytes_saved: int = 0
    compression_ratio: float = 0.0
```

### WeightGroup Dataclass

```python
@dataclass
class WeightGroup:
    """Group of weights that are identical or similar."""

    reference_hash: str
    reference_weight: WeightTensor
    duplicates: List[Tuple[str, WeightTensor]] = field(default_factory=list)
    similar: List[Tuple[str, WeightTensor, float]] = field(default_factory=list)
    deltas: Dict[str, Delta] = field(default_factory=dict)

    @property
    def total_count(self) -> int:
        """Total number of weights in this group."""

    @property
    def bytes_saved(self) -> int:
        """Bytes saved by deduplication in this group."""
```

---

## Delta Module API

### DeltaType Enum

Defines delta encoding strategies with varying compression ratios and fidelity.

```python
class DeltaType(Enum):
    # Lossless strategies (perfect reconstruction)
    FLOAT32_RAW = "float32_raw"           # ~50% compression
    COMPRESSED = "compressed"              # ~70% compression
    XOR_FLOAT32 = "xor_float32"           # 15-25% better
    XOR_BFLOAT16 = "xor_bfloat16"         # Optimized for BF16
    EXPONENT_MANTISSA = "exponent_mantissa"  # 10-20% better

    # Lossy strategies (approximate reconstruction)
    INT8_QUANTIZED = "int8_quantized"     # ~75% compression
    INT16_QUANTIZED = "int16_quantized"   # ~50% compression
    SPARSE = "sparse"                      # Discards small diffs
    PER_AXIS_SCALED = "per_axis_scaled"   # 1-bit signs + scales

    @property
    def is_lossless(self) -> bool:
        """Return True if this encoding strategy is lossless."""
```

### DeltaConfig Dataclass

```python
@dataclass
class DeltaConfig:
    """Configuration for delta encoding."""

    delta_type: DeltaType = DeltaType.FLOAT32_RAW
    sparse_threshold: float = 1e-6
    quantization_bits: int = 8
    compression_level: int = 6
    max_delta_ratio: float = 1.0
    min_compression_ratio: float = 0.0
    min_weight_size: int = 512
    strict_reconstruction: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeltaConfig:
        """Create from dictionary."""
```

### Delta Dataclass

```python
@dataclass
class Delta:
    """Represents a delta encoding of weight differences."""

    delta_type: DeltaType
    data: np.ndarray
    metadata: Dict[str, Any]
    original_shape: Tuple[int, ...]
    original_dtype: np.dtype
    reference_hash: str
    compression_ratio: float = 0.0

    @property
    def nbytes(self) -> int:
        """Get size in bytes of the delta including metadata overhead."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Delta:
        """Create from dictionary."""
```

### DeltaEncoder Class

Encoder for creating and applying delta encodings between similar weights.

#### Constructor

```python
DeltaEncoder(config: Optional[DeltaConfig] = None)
```

#### Methods

```python
def can_encode_as_delta(
    self,
    weight: WeightTensor,
    reference: WeightTensor
) -> bool:
    """
    Check if weight can be efficiently encoded as delta from reference.

    Returns:
        True if delta encoding is worthwhile
    """

def encode_delta(
    self,
    weight: WeightTensor,
    reference: WeightTensor
) -> Delta:
    """
    Encode weight as delta from reference.

    Args:
        weight: Weight to encode
        reference: Reference weight

    Returns:
        Delta object containing encoded differences

    Raises:
        ValueError: If weights have incompatible shapes/dtypes
    """

def decode_delta(
    self,
    delta: Delta,
    reference: WeightTensor
) -> WeightTensor:
    """
    Decode delta and reconstruct original weight.

    Args:
        delta: The delta encoding to apply
        reference: The reference weight to apply delta to

    Returns:
        Reconstructed weight tensor

    Raises:
        DeltaReconstructionError: If strict mode enabled and reference hash mismatches
    """

def estimate_delta_size(
    self,
    weight: WeightTensor,
    reference: WeightTensor
) -> int:
    """Estimate delta size without actually encoding."""
```

#### Example Usage

```python
from coral import DeltaEncoder, DeltaConfig, DeltaType

# Create encoder with lossless compression
config = DeltaConfig(
    delta_type=DeltaType.COMPRESSED,
    compression_level=9,
    strict_reconstruction=True
)
encoder = DeltaEncoder(config)

# Check if delta encoding is efficient
if encoder.can_encode_as_delta(weight, reference):
    # Encode
    delta = encoder.encode_delta(weight, reference)
    print(f"Delta size: {delta.nbytes} bytes")
    print(f"Compression: {delta.compression_ratio:.2%}")

    # Decode (perfect reconstruction for lossless types)
    reconstructed = encoder.decode_delta(delta, reference)
    assert np.allclose(reconstructed.data, weight.data)
```

---

## Storage Module API

### WeightStore Abstract Base Class

Abstract interface that all storage backends must implement.

```python
class WeightStore(ABC):
    """Abstract base class for weight storage backends."""

    @abstractmethod
    def store(self, weight: WeightTensor, hash_key: Optional[str] = None) -> str:
        """Store a weight tensor."""

    @abstractmethod
    def load(self, hash_key: str) -> Optional[WeightTensor]:
        """Load a weight tensor by hash."""

    @abstractmethod
    def exists(self, hash_key: str) -> bool:
        """Check if a weight exists in storage."""

    @abstractmethod
    def delete(self, hash_key: str) -> bool:
        """Delete a weight from storage."""

    @abstractmethod
    def list_weights(self) -> List[str]:
        """List all weight hashes in storage."""

    @abstractmethod
    def get_metadata(self, hash_key: str) -> Optional[WeightMetadata]:
        """Get metadata for a weight without loading data."""

    @abstractmethod
    def store_batch(self, weights: Dict[str, WeightTensor]) -> Dict[str, str]:
        """Store multiple weights efficiently."""

    @abstractmethod
    def load_batch(self, hash_keys: List[str]) -> Dict[str, WeightTensor]:
        """Load multiple weights efficiently."""

    @abstractmethod
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about storage usage and statistics."""

    @abstractmethod
    def close(self):
        """Close the storage backend and cleanup resources."""
```

### HDF5Store Class

HDF5-based storage backend with compression and delta support.

#### Constructor

```python
HDF5Store(
    filepath: str,
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = 4,
    mode: str = "a"
)
```

**Parameters:**
- `filepath`: Path to HDF5 file
- `compression`: Compression algorithm ('gzip', 'lzf', None)
- `compression_opts`: Compression level (1-9 for gzip)
- `mode`: File mode ('r', 'r+', 'w', 'a')

#### Methods

Implements all `WeightStore` methods plus:

```python
def store_delta(self, delta: Delta, hash_key: str) -> str:
    """Store a delta object."""

def load_delta(self, hash_key: str) -> Optional[Delta]:
    """Load a delta object by hash."""

def list_deltas(self) -> List[str]:
    """List all delta hashes in storage."""
```

#### Example Usage

```python
from coral import HDF5Store, WeightTensor

# Create store with context manager
with HDF5Store("weights.h5", compression="gzip", compression_opts=9) as store:
    # Store weights
    hash_key = store.store(weight)

    # Load weights
    loaded = store.load(hash_key)

    # Batch operations
    hashes = store.store_batch({"w1": weight1, "w2": weight2})
    weights = store.load_batch(list(hashes.values()))

    # Get info
    info = store.get_storage_info()
    print(f"Total weights: {info['total_weights']}")
    print(f"Storage size: {info['total_size_bytes']} bytes")
```

### S3Store Class (Optional)

S3-based storage backend (requires `boto3`).

```python
from coral.storage import get_s3_store, S3Config

config = S3Config(
    bucket_name="my-weights",
    region="us-west-2",
    prefix="models/"
)
store = get_s3_store(config)
```

---

## Version Control Module API

### Repository Class

Main repository class for git-like version control of neural network weights.

#### Constructor

```python
Repository(path: Path, init: bool = False)
```

**Parameters:**
- `path`: Path to repository
- `init`: If True, initialize a new repository

#### Core Methods

##### Staging

```python
def stage_weights(self, weights: Dict[str, WeightTensor]) -> Dict[str, str]:
    """
    Stage weights for commit with delta encoding support.

    Args:
        weights: Dictionary mapping names to WeightTensors

    Returns:
        Dictionary mapping names to reference hashes
    """
```

##### Commits

```python
def commit(
    self,
    message: str,
    author: Optional[str] = None,
    email: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Commit:
    """
    Create a commit with staged weights.

    Args:
        message: Commit message
        author: Author name (defaults to config)
        email: Author email (defaults to config)
        tags: Optional tags for this commit

    Returns:
        Commit object
    """

def log(
    self,
    branch: Optional[str] = None,
    max_count: Optional[int] = None
) -> List[Commit]:
    """
    Get commit history.

    Args:
        branch: Branch name (defaults to current)
        max_count: Maximum number of commits to return

    Returns:
        List of commits in reverse chronological order
    """

def diff(
    self,
    commit_a: str,
    commit_b: str,
    weight_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compare two commits.

    Returns:
        Dictionary with added, removed, and modified weights
    """
```

##### Branches

```python
def create_branch(self, name: str, from_commit: Optional[str] = None) -> Branch:
    """Create a new branch."""

def checkout(self, branch_or_commit: str) -> None:
    """Checkout a branch or commit."""

def merge(
    self,
    source_branch: str,
    strategy: MergeStrategy = MergeStrategy.FAIL,
    alpha: float = 0.5
) -> Commit:
    """
    Merge source branch into current branch.

    Args:
        source_branch: Name of branch to merge
        strategy: How to handle conflicts
        alpha: Weight for WEIGHTED strategy (alpha * ours + (1-alpha) * theirs)

    Returns:
        Merge commit

    Raises:
        MergeConflictError: If conflicts and strategy is FAIL
    """

def list_branches(self) -> List[str]:
    """List all branches."""

def current_branch(self) -> Optional[str]:
    """Get current branch name."""
```

##### Tags

```python
def tag_version(
    self,
    tag_name: str,
    commit_hash: Optional[str] = None,
    message: Optional[str] = None
) -> None:
    """Create a tag at a commit."""

def list_tags(self) -> List[str]:
    """List all tags."""
```

##### Weights

```python
def get_weight(self, name: str, commit: Optional[str] = None) -> Optional[WeightTensor]:
    """
    Get a specific weight by name from a commit.

    Args:
        name: Weight name
        commit: Commit hash (defaults to HEAD)
    """

def get_all_weights(self, commit: Optional[str] = None) -> Dict[str, WeightTensor]:
    """Get all weights from a commit."""
```

##### Remote Operations

```python
def push(self, remote_url: str, branch: Optional[str] = None) -> None:
    """Push commits to remote repository."""

def pull(self, remote_url: str, branch: Optional[str] = None) -> None:
    """Pull commits from remote repository."""

def sync(self, remote_url: str) -> None:
    """Sync with remote repository (pull then push)."""
```

##### Maintenance

```python
def gc(self, aggressive: bool = False) -> Dict[str, int]:
    """
    Run garbage collection to clean unreferenced weights.

    Args:
        aggressive: If True, more aggressive cleanup

    Returns:
        Statistics about cleaned objects
    """
```

#### Example Usage

```python
from coral import Repository
from pathlib import Path

# Initialize repository
repo = Repository(Path("./my_project"), init=True)

# Stage and commit weights
repo.stage_weights(weights)
commit = repo.commit("Initial model checkpoint")
print(f"Committed: {commit.commit_hash}")

# Create branch for experiment
repo.create_branch("experiment-1")
repo.checkout("experiment-1")

# Later, merge back
repo.checkout("main")
repo.merge("experiment-1", strategy=MergeStrategy.OURS)

# View history
for commit in repo.log(max_count=10):
    print(f"{commit.commit_hash[:8]} - {commit.message}")

# Clean up
stats = repo.gc()
print(f"Cleaned {stats['weights_removed']} unreferenced weights")
```

### Commit Class

```python
@dataclass
class Commit:
    commit_hash: str
    message: str
    author: str
    email: str
    timestamp: float
    parent_commits: List[str]
    weight_hashes: Dict[str, str]
    metadata: CommitMetadata
    tags: List[str] = field(default_factory=list)
```

### CommitMetadata Dataclass

```python
@dataclass
class CommitMetadata:
    branch: str
    is_merge: bool = False
    merge_strategy: Optional[str] = None
    training_metrics: Dict[str, float] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)
```

### MergeStrategy Enum

```python
class MergeStrategy(Enum):
    OURS = "ours"          # Prefer current branch weights
    THEIRS = "theirs"      # Prefer source branch weights
    FAIL = "fail"          # Raise error on conflicts
    AVERAGE = "average"    # Average conflicting weights
    WEIGHTED = "weighted"  # Weighted average
```

---

## Training Module API

### CheckpointConfig Dataclass

```python
@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management."""

    # Checkpoint frequency
    save_every_n_steps: Optional[int] = None
    save_every_n_epochs: Optional[int] = None
    save_on_best_metric: Optional[str] = None
    minimize_metric: bool = True

    # Retention policy
    keep_last_n_checkpoints: Optional[int] = None
    keep_best_n_checkpoints: Optional[int] = None
    keep_checkpoint_every_n_epochs: Optional[int] = None

    # Storage options
    save_optimizer_state: bool = True
    save_scheduler_state: bool = True
    save_random_state: bool = True
    use_incremental_saves: bool = True

    # Commit options
    auto_commit: bool = True
    commit_message_template: str = "Checkpoint at epoch {epoch}, step {step}"
    tag_best_checkpoints: bool = True

    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0
```

**Early Stopping Parameters**:
- `early_stopping_patience`: Number of checkpoints without improvement before triggering early stop (None to disable)
- `early_stopping_threshold`: Minimum improvement required to reset patience counter

### CheckpointManager Class

```python
class CheckpointManager:
    """Manages training checkpoints with Coral version control."""

    def __init__(
        self,
        repository: Repository,
        config: CheckpointConfig,
        model_name: str,
        experiment_name: Optional[str] = None
    ):
        """Initialize checkpoint manager."""

    def should_save_checkpoint(self, state: TrainingState) -> bool:
        """Determine if a checkpoint should be saved."""

    def save_checkpoint(
        self,
        weights: Dict[str, WeightTensor],
        state: TrainingState,
        force: bool = False
    ) -> Optional[str]:
        """
        Save a checkpoint.

        Returns:
            Commit hash if saved, None otherwise
        """

    def load_checkpoint(self, commit_hash: str) -> Tuple[Dict[str, WeightTensor], TrainingState]:
        """Load checkpoint from commit."""

    def cleanup_old_checkpoints(self) -> int:
        """Remove old checkpoints according to retention policy."""

    def diff_checkpoints(self, ref_a: str, ref_b: str) -> Dict[str, Any]:
        """
        Compare two checkpoints.

        Args:
            ref_a: Commit hash of first checkpoint
            ref_b: Commit hash of second checkpoint

        Returns:
            Dictionary with:
            - identical: bool - True if all weights match exactly
            - changed: List[str] - Weights with different values
            - added: List[str] - Weights only in second checkpoint
            - removed: List[str] - Weights only in first checkpoint
            - similarity: Dict[str, float] - Cosine similarity for changed weights
        """

    @property
    def should_stop_early(self) -> bool:
        """
        Check if early stopping should be triggered.

        Returns True if no_improvement_count >= early_stopping_patience
        """

    @property
    def no_improvement_count(self) -> int:
        """
        Get the number of checkpoints since last improvement.
        """
```

### TrainingState Class

```python
class TrainingState:
    """Training state for checkpointing."""

    epoch: int
    global_step: int
    loss: float
    metrics: Dict[str, float]
    model_name: str
    experiment_name: str
    optimizer_state: Optional[Dict]
    scheduler_state: Optional[Dict]
    random_state: Optional[Dict]

    def save(self, filepath: str) -> None:
        """Save state to JSON file."""

    @classmethod
    def load(cls, filepath: str) -> TrainingState:
        """Load state from JSON file."""
```

---

## Integrations Module API

### PyTorchIntegration Class

Static methods for PyTorch model conversion.

```python
class PyTorchIntegration:
    """Integration utilities for PyTorch models."""

    @staticmethod
    def model_to_weights(model: nn.Module) -> Dict[str, WeightTensor]:
        """Convert PyTorch model to Coral weights."""

    @staticmethod
    def weights_to_model(weights: Dict[str, WeightTensor], model: nn.Module) -> None:
        """Load Coral weights into PyTorch model."""

    @staticmethod
    def save_optimizer_state(optimizer: Optimizer) -> Dict[str, Any]:
        """Save optimizer state."""

    @staticmethod
    def load_optimizer_state(optimizer: Optimizer, state: Dict[str, Any]) -> None:
        """Load optimizer state."""

    @staticmethod
    def save_scheduler_state(scheduler: _LRScheduler) -> Dict[str, Any]:
        """Save scheduler state."""

    @staticmethod
    def load_scheduler_state(scheduler: _LRScheduler, state: Dict[str, Any]) -> None:
        """Load scheduler state."""

    @staticmethod
    def get_random_state() -> Dict[str, Any]:
        """Get random state for reproducibility."""

    @staticmethod
    def set_random_state(state: Dict[str, Any]) -> None:
        """Set random state for reproducibility."""
```

### CoralTrainer Class

PyTorch trainer with integrated Coral checkpointing.

```python
class CoralTrainer:
    """PyTorch trainer with Coral version control integration."""

    def __init__(
        self,
        model: nn.Module,
        repository: Repository,
        experiment_name: str,
        checkpoint_config: Optional[CheckpointConfig] = None
    ):
        """Initialize trainer."""

    def train_step(
        self,
        batch: Any,
        optimizer: Optimizer,
        loss_fn: Callable
    ) -> float:
        """Execute a training step."""

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        force: bool = False
    ) -> Optional[str]:
        """Save checkpoint if policy conditions met."""

    def restore_checkpoint(self, commit_hash: str) -> None:
        """Restore model from checkpoint."""
```

### Checkpointer Class

Simplified, Pythonic API for PyTorch checkpointing with context manager support.

```python
class Checkpointer:
    """Simplified checkpoint management for PyTorch models."""

    def __init__(
        self,
        model: nn.Module,
        repo: Union[str, Path, Repository],
        experiment: str,
        *,
        every_n_epochs: int = 0,
        every_n_steps: int = 0,
        on_best: Optional[str] = None,
        minimize: bool = True,
        keep_last: int = 5,
        keep_best: int = 3,
        resume: Union[bool, str] = False,
        tracker: Optional[ExperimentBridge] = None,
    ):
        """
        Initialize checkpointer.

        Args:
            model: PyTorch model to checkpoint
            repo: Repository path or object
            experiment: Experiment name
            every_n_epochs: Save every N epochs (0 to disable)
            every_n_steps: Save every N steps (0 to disable)
            on_best: Metric name to monitor for best checkpoint
            minimize: Whether lower metric values are better
            keep_last: Number of recent checkpoints to keep
            keep_best: Number of best checkpoints to keep
            resume: True for latest, "best" for best, or False
            tracker: Optional experiment tracker (MLflow, W&B)
        """

    def __enter__(self) -> Checkpointer:
        """Start experiment tracking run."""

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End experiment tracking run."""

    def log(self, epoch: int, step: int, **metrics) -> Optional[str]:
        """
        Log metrics and save checkpoint if conditions met.

        Returns:
            Commit hash if checkpoint saved, None otherwise
        """

    @property
    def epoch(self) -> int:
        """Current epoch number."""

    @property
    def global_step(self) -> int:
        """Current global step."""

    @property
    def best_commit(self) -> Optional[str]:
        """Commit hash of best checkpoint."""

    @property
    def metrics(self) -> Dict[str, float]:
        """Current metric values."""

    @property
    def repo(self) -> Repository:
        """Underlying repository object."""
```

### Unified load() and save() Functions

```python
def load(
    model: nn.Module,
    repo: Union[str, Path, Repository],
    commit: Optional[str] = None,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Load weights from repository into model.

    Args:
        model: PyTorch model to load into
        repo: Repository path or object
        commit: Commit hash (None for HEAD)
        strict: Raise error on missing/unexpected keys

    Returns:
        Dictionary with:
        - loaded: List of weight names loaded
        - matched: Whether all weights matched
    """

def save(
    model: nn.Module,
    repo: Union[str, Path, Repository],
    message: str = "Checkpoint"
) -> Dict[str, Any]:
    """
    Save model weights to repository.

    Args:
        model: PyTorch model to save
        repo: Repository path or object
        message: Commit message

    Returns:
        Dictionary with:
        - commit_hash: Hash of created commit
        - weights_saved: Number of weights saved
    """
```

### Utility Functions

```python
def load_into_model(model: nn.Module, weights: Dict[str, WeightTensor]) -> None:
    """Load Coral weights into PyTorch model."""

def save_model(model: nn.Module, path: str) -> Dict[str, WeightTensor]:
    """Save PyTorch model as Coral weights."""

def load_model_from_coral(model: nn.Module, repo: Repository, commit: str) -> None:
    """Load model from Coral repository."""

def save_model_to_coral(model: nn.Module, repo: Repository, message: str) -> Commit:
    """Save model to Coral repository."""

def compare_model_weights(model1: nn.Module, model2: nn.Module) -> Dict[str, float]:
    """Compare weights between two models."""
```

### StreamingWeightLoader Class

```python
class StreamingWeightLoader:
    """Stream weights from storage without loading all into memory."""

    def __init__(self, repo: Repository, commit: str):
        """Initialize streaming loader."""

    def __iter__(self) -> Iterator[Tuple[str, WeightTensor]]:
        """Iterate over weights."""

    def load_layer(self, layer_name: str) -> WeightTensor:
        """Load a specific layer's weights."""
```

### Framework Callback Aliases

For consistency and discoverability, Coral provides aliases for framework callbacks:

```python
# PyTorch Lightning
from coral.integrations.lightning import CoralCallback
from coral.integrations.lightning import CoralLightningCallback  # Alias

# HuggingFace Transformers
from coral.integrations.hf_trainer import CoralTrainerCallback
from coral.integrations.hf_trainer import CoralHFCallback  # Alias
```

Both callbacks now accept a `repo` parameter that can be a path string, `Path` object, or `Repository` instance:

```python
from coral import Repository
from coral.integrations.lightning import CoralCallback

# Using Repository object
repo = Repository("./checkpoints", init=True)
callback = CoralCallback(repo=repo)

# Using path string (deprecated, use repo instead)
callback = CoralCallback(repo_path="./checkpoints", init=True)
```

---

## Experiments Module API

### ExperimentStatus Enum

```python
class ExperimentStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
```

### ExperimentMetric Dataclass

```python
@dataclass
class ExperimentMetric:
    name: str
    value: float
    step: int
    timestamp: float
```

### Experiment Dataclass

```python
@dataclass
class Experiment:
    name: str
    status: ExperimentStatus
    start_time: float
    end_time: Optional[float]
    metrics: List[ExperimentMetric]
    config: Dict[str, Any]
    commit_hashes: List[str]
    tags: List[str]
```

### ExperimentTracker Class

```python
class ExperimentTracker:
    """Track ML experiments with Coral."""

    def __init__(self, repository: Repository):
        """Initialize tracker."""

    def start(self, name: str, config: Optional[Dict] = None, tags: Optional[List[str]] = None) -> Experiment:
        """Start a new experiment."""

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""

    def end(self, status: ExperimentStatus = ExperimentStatus.COMPLETED) -> None:
        """End current experiment."""

    def list_experiments(self) -> List[Experiment]:
        """List all experiments."""

    def get_best_experiment(self, metric: str, minimize: bool = True) -> Optional[Experiment]:
        """Get experiment with best metric value."""

    def compare_experiments(self, names: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments."""
```

---

## Registry Module API

### RegistryType Enum

```python
class RegistryType(Enum):
    HUGGINGFACE = "huggingface"
    MLFLOW = "mlflow"
    LOCAL = "local"
```

### PublishResult Dataclass

```python
@dataclass
class PublishResult:
    registry_type: RegistryType
    model_id: str
    url: Optional[str]
    success: bool
    message: str
```

### ModelPublisher Class

```python
class ModelPublisher:
    """Publish models to various registries."""

    def __init__(self, repository: Repository):
        """Initialize publisher."""

    def publish_huggingface(
        self,
        model_id: str,
        commit_ref: str,
        token: Optional[str] = None,
        private: bool = False
    ) -> PublishResult:
        """Publish model to Hugging Face Hub."""

    def publish_mlflow(
        self,
        model_name: str,
        commit_ref: str,
        experiment_id: Optional[str] = None
    ) -> PublishResult:
        """Publish model to MLflow registry."""

    def publish_local(
        self,
        output_path: str,
        commit_ref: str,
        format: str = "safetensors"
    ) -> PublishResult:
        """Export model to local directory."""
```

---

## Compression Module API

### Quantizer Class

```python
class Quantizer:
    """Weight quantization for compression."""

    def quantize_int8(self, weight: WeightTensor) -> WeightTensor:
        """Quantize to 8-bit integers."""

    def quantize_int4(self, weight: WeightTensor) -> WeightTensor:
        """Quantize to 4-bit integers."""

    def dequantize(self, quantized: WeightTensor) -> WeightTensor:
        """Dequantize back to float."""
```

### Pruner Class

```python
class Pruner:
    """Weight pruning for compression."""

    def magnitude_prune(self, weight: WeightTensor, sparsity: float) -> WeightTensor:
        """Prune by magnitude threshold."""

    def structured_prune(self, weight: WeightTensor, ratio: float) -> WeightTensor:
        """Structured pruning (remove entire channels/filters)."""
```

---

## Utility Functions

### Similarity Functions

```python
from coral.utils.similarity import (
    cosine_similarity,
    magnitude_similarity,
    weight_similarity,
    are_similar
)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two arrays.
    Scale-invariant.

    Returns:
        Similarity value between -1 and 1
    """

def magnitude_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute magnitude similarity (ratio of norms).

    Returns:
        Similarity value between 0 and 1
    """

def weight_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Combined similarity metric for neural network weights.
    Considers both direction and magnitude.

    Returns:
        Combined similarity score
    """

def are_similar(
    a: np.ndarray,
    b: np.ndarray,
    threshold: float = 0.99,
    check_magnitude: bool = True,
    magnitude_tolerance: float = 0.1
) -> bool:
    """
    Check if two weights are similar.

    Args:
        threshold: Cosine similarity threshold
        check_magnitude: Also check magnitude difference
        magnitude_tolerance: Max relative magnitude difference
    """
```

### Visualization Functions

```python
from coral.utils.visualization import (
    plot_weight_distribution,
    plot_similarity_matrix,
    plot_deduplication_stats,
    visualize_delta_compression
)
```

### JSON Utilities

```python
from coral.utils.json_utils import dump_numpy, load_numpy

def dump_numpy(obj: Any, fp: IO, **kwargs) -> None:
    """Dump object with numpy arrays to JSON."""

def load_numpy(fp: IO) -> Any:
    """Load object with numpy arrays from JSON."""
```

---

## Summary

This API reference covers all major public interfaces in the Coral system. For additional examples and advanced usage patterns, refer to:

- **Chapter 3**: Core concepts and basic usage
- **Chapter 4**: Delta encoding in depth
- **Chapter 5**: Version control workflows
- **Chapter 6**: Training integration
- **Chapter 7**: Production deployment

For the most up-to-date API documentation and implementation details, see the source code and inline docstrings in the `/home/user/coral/src/coral/` directory.
