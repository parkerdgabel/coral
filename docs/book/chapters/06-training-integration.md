# Chapter 6: Training Integration

Training machine learning models is an iterative, experimental process that generates numerous intermediate checkpoints. As models train, they produce snapshots at different epochs, steps, and performance levels—each potentially valuable for analysis, rollback, or deployment. Traditional checkpointing approaches scatter these artifacts across filesystems or cloud storage, making it difficult to track lineage, compare versions, or recover from failures.

Coral's training integration solves this problem by providing seamless, automatic version control for model weights throughout the training lifecycle. By integrating directly with training loops, Coral captures checkpoints with full provenance, enabling git-like workflows for machine learning experiments.

This chapter covers Coral's comprehensive training integration system, from low-level checkpoint management to high-level framework callbacks.

## 6.1 Training Integration Overview

### Why Training Integration Matters

Training integration addresses several critical challenges in ML workflows:

**Checkpoint Sprawl**: Traditional training produces scattered checkpoint files (e.g., `model_epoch_1.pt`, `model_epoch_2.pt`, `best_model.pt`). Coral organizes these as versioned commits with full history.

**Experiment Tracking**: Coral automatically captures training metrics, hyperparameters, and model states alongside weights, creating a complete record of each training run.

**Storage Efficiency**: Multiple checkpoints of similar models benefit from Coral's deduplication and delta encoding, reducing storage by 50-98% compared to naive checkpointing.

**Reproducibility**: By versioning optimizer states, scheduler states, and random seeds, Coral enables exact training resumption from any checkpoint.

**Collaboration**: Teams can share training runs through Coral's version control system, branching experiments and merging successful approaches just like code.

### Automatic Checkpointing Benefits

Coral's automatic checkpointing provides:

- **Configurable policies**: Save every N epochs, every N steps, or when metrics improve
- **Retention management**: Automatically clean up old checkpoints while preserving important ones
- **Metric-based saving**: Save only when validation loss decreases or accuracy improves
- **State preservation**: Capture complete training state including optimizers and schedulers
- **Callback hooks**: Execute custom logic after checkpoint saves
- **Zero boilerplate**: Training loops require minimal code changes

### Coral's Approach to Training Workflows

Coral's training integration follows a layered architecture:

1. **TrainingState**: Encapsulates complete training state (epoch, metrics, optimizer, etc.)
2. **CheckpointConfig**: Declarative configuration for checkpoint policies
3. **CheckpointManager**: Core checkpoint management logic (save, load, cleanup)
4. **Framework Integration**: High-level wrappers for PyTorch, Lightning, and HuggingFace
5. **Repository**: Underlying version control system for weight storage

This design separates concerns: configuration is declarative, management is framework-agnostic, and integration layers provide framework-specific conveniences.

## 6.2 CheckpointConfig - Configuration Options

The `CheckpointConfig` dataclass defines all checkpoint behavior through declarative settings. It's designed to be framework-agnostic and covers four main areas: frequency, retention, storage, and commits.

### Checkpoint Frequency

Control when checkpoints are created:

```python
from coral.training import CheckpointConfig

config = CheckpointConfig(
    # Save checkpoint every 5 epochs
    save_every_n_epochs=5,

    # Save checkpoint every 1000 training steps
    save_every_n_steps=1000,

    # Save when validation loss improves
    save_on_best_metric="val_loss",
    minimize_metric=True,  # True for loss, False for accuracy
)
```

**Parameters**:
- `save_every_n_epochs`: Save every N epochs (None to disable)
- `save_every_n_steps`: Save every N steps (None to disable)
- `save_on_best_metric`: Metric name to monitor for best model saving
- `minimize_metric`: Whether lower metric values are better (True for loss, False for accuracy)

These conditions are evaluated independently—checkpoints save when *any* condition is met.

### Retention Policy

Manage checkpoint lifecycle to control storage growth:

```python
config = CheckpointConfig(
    # Keep only the last 10 checkpoints
    keep_last_n_checkpoints=10,

    # Keep the best 3 checkpoints by metric
    keep_best_n_checkpoints=3,

    # Keep checkpoints at epoch 10, 20, 30, etc.
    keep_checkpoint_every_n_epochs=10,
)
```

**Parameters**:
- `keep_last_n_checkpoints`: Number of recent checkpoints to preserve
- `keep_best_n_checkpoints`: Number of best checkpoints (by metric) to preserve
- `keep_checkpoint_every_n_epochs`: Keep periodic checkpoints at epoch intervals

Coral's cleanup policy is conservative: it preserves checkpoints that match *any* retention criterion. The best checkpoint is always kept, even if it falls outside retention windows.

### Storage Options

Control what gets saved with each checkpoint:

```python
config = CheckpointConfig(
    # Save optimizer state (for exact training resumption)
    save_optimizer_state=True,

    # Save scheduler state (learning rate schedules)
    save_scheduler_state=True,

    # Save random state (for reproducibility)
    save_random_state=True,

    # Use incremental saves (leverage Coral's deduplication)
    use_incremental_saves=True,
)
```

**Parameters**:
- `save_optimizer_state`: Include optimizer state (Adam momentum, etc.)
- `save_scheduler_state`: Include learning rate scheduler state
- `save_random_state`: Include random number generator states (PyTorch, NumPy, CUDA)
- `use_incremental_saves`: Enable delta encoding for similar checkpoints

Saving optimizer and scheduler states enables exact training continuation. Random state preservation ensures deterministic training resumption, critical for debugging and reproducibility.

### Commit Behavior

Control how checkpoints integrate with version control:

```python
config = CheckpointConfig(
    # Automatically create commits for checkpoints
    auto_commit=True,

    # Template for commit messages (supports {epoch}, {step}, {loss}, and any metric)
    commit_message_template="Checkpoint at epoch {epoch}, step {step}, loss={loss:.4f}",

    # Automatically tag best checkpoints
    tag_best_checkpoints=True,
)
```

**Parameters**:
- `auto_commit`: Create version control commits automatically
- `commit_message_template`: Format string for commit messages (uses TrainingState fields)
- `tag_best_checkpoints`: Create version tags for best checkpoints

When `auto_commit=True`, each checkpoint becomes a version-controlled commit with full history and metadata. Tags make best models easy to find and load later.

### Early Stopping

Configure early stopping to halt training when metrics stop improving:

```python
config = CheckpointConfig(
    save_on_best_metric="val_loss",
    minimize_metric=True,

    # Stop training after 5 epochs without improvement
    early_stopping_patience=5,

    # Require at least 0.001 improvement to count as progress
    early_stopping_threshold=0.001,
)
```

**Parameters**:
- `early_stopping_patience`: Number of checkpoints without improvement before stopping (None to disable)
- `early_stopping_threshold`: Minimum improvement required to reset patience counter

Early stopping works with the `CheckpointManager`:

```python
manager = CheckpointManager(repo, config, model_name="MyModel")

for epoch in range(num_epochs):
    # ... training ...
    manager.save_checkpoint(weights, state)

    # Check if training should stop
    if manager.should_stop_early:
        print(f"Early stopping at epoch {epoch}")
        print(f"No improvement for {manager.no_improvement_count} checkpoints")
        break
```

### Example Configurations

**Aggressive Checkpointing** (research/debugging):
```python
# Save frequently, keep everything
research_config = CheckpointConfig(
    save_every_n_epochs=1,
    save_every_n_steps=500,
    save_on_best_metric="val_loss",
    minimize_metric=True,
    keep_last_n_checkpoints=None,  # Keep all
    save_optimizer_state=True,
    save_scheduler_state=True,
    save_random_state=True,
    auto_commit=True,
    tag_best_checkpoints=True,
)
```

**Production Training** (efficiency-focused):
```python
# Save selectively, aggressive cleanup
production_config = CheckpointConfig(
    save_every_n_epochs=10,
    save_on_best_metric="val_accuracy",
    minimize_metric=False,
    keep_last_n_checkpoints=3,
    keep_best_n_checkpoints=5,
    keep_checkpoint_every_n_epochs=50,
    save_optimizer_state=False,  # Don't need to resume
    save_scheduler_state=False,
    save_random_state=False,
    auto_commit=True,
    tag_best_checkpoints=True,
)
```

**Long Training Run** (storage-efficient):
```python
# Balance checkpoint frequency with storage
long_run_config = CheckpointConfig(
    save_every_n_epochs=5,
    save_on_best_metric="val_loss",
    minimize_metric=True,
    keep_last_n_checkpoints=5,
    keep_best_n_checkpoints=10,
    keep_checkpoint_every_n_epochs=25,
    save_optimizer_state=True,
    use_incremental_saves=True,  # Leverage delta encoding
    auto_commit=True,
)
```

**Metric-Only Checkpointing** (sparse but high-quality):
```python
# Save only when performance improves
metric_only_config = CheckpointConfig(
    save_on_best_metric="val_f1_score",
    minimize_metric=False,
    keep_best_n_checkpoints=10,
    save_optimizer_state=True,
    auto_commit=True,
    tag_best_checkpoints=True,
    commit_message_template="Best F1: {val_f1_score:.4f} at epoch {epoch}",
)
```

## 6.3 CheckpointManager - Checkpoint Management

The `CheckpointManager` class implements Coral's checkpoint logic. It connects a `Repository` with a `CheckpointConfig` and manages the full checkpoint lifecycle: creation, storage, loading, and cleanup.

### Initialization

```python
from coral.version_control.repository import Repository
from coral.training import CheckpointManager, CheckpointConfig

# Create or open repository
repo = Repository("./my_model", init=True)

# Configure checkpointing
config = CheckpointConfig(
    save_every_n_epochs=2,
    save_on_best_metric="accuracy",
    minimize_metric=False,
    keep_last_n_checkpoints=5,
)

# Initialize checkpoint manager
manager = CheckpointManager(
    repository=repo,
    config=config,
    model_name="ResNet50",
    experiment_name="experiment_lr_0.001",
)
```

**Parameters**:
- `repository`: Coral repository instance
- `config`: CheckpointConfig defining behavior
- `model_name`: Human-readable model identifier
- `experiment_name`: Experiment identifier (defaults to timestamp)

### should_save_checkpoint() Logic

The manager evaluates checkpoint conditions through `should_save_checkpoint()`:

```python
from coral.training import TrainingState

state = TrainingState(
    epoch=10,
    global_step=5000,
    learning_rate=0.001,
    loss=0.234,
    metrics={"accuracy": 0.95, "val_loss": 0.189},
)

if manager.should_save_checkpoint(state):
    # Checkpoint will be saved
    pass
```

**Decision Logic**:
1. **Step-based**: Save if `global_step % save_every_n_steps == 0`
2. **Epoch-based**: Save if `epoch % save_every_n_epochs == 0`
3. **Metric-based**: Save if metric improved over previous best

The method returns `True` if *any* condition is satisfied. This allows combining multiple policies (e.g., "save every 10 epochs AND when loss improves").

### save_checkpoint() Process

The core save operation:

```python
from coral.integrations.pytorch import PyTorchIntegration

# Convert model to weights
weights = PyTorchIntegration.model_to_weights(model)

# Create training state
state = TrainingState(
    epoch=5,
    global_step=1000,
    learning_rate=0.001,
    loss=0.345,
    metrics={"accuracy": 0.92, "val_loss": 0.298},
    optimizer_state=PyTorchIntegration.save_optimizer_state(optimizer),
    scheduler_state=PyTorchIntegration.save_scheduler_state(scheduler),
)

# Save checkpoint
commit_hash = manager.save_checkpoint(weights, state)

if commit_hash:
    print(f"Checkpoint saved: {commit_hash[:8]}")
```

**Save Process**:
1. Check `should_save_checkpoint()` (unless `force=True`)
2. Stage weights in repository
3. Save `TrainingState` to JSON file
4. Create commit (if `auto_commit=True`)
5. Tag as best (if applicable)
6. Record in checkpoint history
7. Clean up old checkpoints
8. Call registered callbacks

**Return Value**: Commit hash (str) if saved, None if checkpoint was skipped.

### load_checkpoint() Restoration

Load a checkpoint to restore training:

```python
# Load latest checkpoint
checkpoint = manager.load_checkpoint()

# Load specific checkpoint by hash
checkpoint = manager.load_checkpoint(commit_hash="a7f3c2d1")

# Load best checkpoint
checkpoint = manager.load_checkpoint(load_best=True)

if checkpoint:
    weights = checkpoint["weights"]  # Dict[str, WeightTensor]
    state = checkpoint["state"]      # TrainingState
    commit_hash = checkpoint["commit_hash"]

    # Restore model weights
    PyTorchIntegration.weights_to_model(weights, model)

    # Restore optimizer state
    if state.optimizer_state:
        PyTorchIntegration.load_optimizer_state(optimizer, state.optimizer_state)
```

**Load Options**:
- `commit_hash`: Load specific checkpoint by hash
- `load_best`: Load the best checkpoint (by metric)
- Default: Load the most recent checkpoint

The returned dictionary contains:
- `weights`: Dictionary of `WeightTensor` objects
- `state`: `TrainingState` with training metadata
- `commit_hash`: Commit hash for the checkpoint

### register_checkpoint_callback() for Custom Hooks

Callbacks execute custom logic after checkpoint saves:

```python
def log_checkpoint(state: TrainingState, commit_hash: str):
    """Log checkpoint to external system."""
    print(f"Checkpoint saved at epoch {state.epoch}")
    if commit_hash:
        # Could log to MLflow, W&B, etc.
        print(f"Commit: {commit_hash[:8]}")

def notify_on_milestone(state: TrainingState, commit_hash: str):
    """Send notification when reaching milestones."""
    if state.metrics.get("accuracy", 0) > 0.95:
        print("🎉 Achieved 95% accuracy!")
        # Send email, Slack message, etc.

# Register callbacks
manager.register_checkpoint_callback(log_checkpoint)
manager.register_checkpoint_callback(notify_on_milestone)

# List registered callbacks
print(manager.list_callbacks())  # ['log_checkpoint', 'notify_on_milestone']

# Unregister a callback
manager.unregister_checkpoint_callback(log_checkpoint)

# Clear all callbacks
manager.clear_callbacks()
```

**Callback Signature**: `callback(state: TrainingState, commit_hash: Optional[str]) -> None`

**Error Handling**: Callbacks are executed with automatic error catching. If a callback raises an exception, it's logged but doesn't interrupt checkpoint saving or other callbacks.

### Tracking: checkpoint_history, best_metric_value

The manager maintains checkpoint metadata:

```python
# View checkpoint history
for checkpoint in manager.checkpoint_history:
    print(f"Epoch {checkpoint['epoch']}: {checkpoint['commit_hash'][:8]}")
    print(f"  Metrics: {checkpoint['metrics']}")
    print(f"  Best: {checkpoint['is_best']}")

# Get current best checkpoint
print(f"Best metric value: {manager.best_metric_value}")
print(f"Best checkpoint: {manager.best_checkpoint_hash[:8]}")

# List only best checkpoints
best_checkpoints = manager.list_checkpoints(only_best=True)

# Get info about specific checkpoint
info = manager.get_checkpoint_info(commit_hash)
```

**Checkpoint History Structure**:
```python
{
    "commit_hash": "a7f3c2d1...",
    "epoch": 10,
    "global_step": 5000,
    "timestamp": "2025-01-15T10:30:45",
    "metrics": {"accuracy": 0.95, "val_loss": 0.189},
    "is_best": True,
}
```

History is persisted to `.coral/checkpoints/{experiment_name}.json` and survives process restarts.

### diff_checkpoints() for Comparison

Compare two checkpoints to understand what changed:

```python
# Compare two checkpoints
diff = manager.diff_checkpoints("a7f3c2d1", "b8e4f5a2")

print(f"Checkpoints identical: {diff['identical']}")
print(f"Changed weights: {diff['changed']}")
print(f"Added weights: {diff['added']}")
print(f"Removed weights: {diff['removed']}")

# Similarity scores for changed weights
for name, similarity in diff['similarity'].items():
    print(f"  {name}: {similarity:.4f} cosine similarity")
```

**Return Value**:
```python
{
    "identical": False,           # True if all weights match exactly
    "changed": ["layer1.weight"], # Weights with different values
    "added": ["layer3.weight"],   # Weights only in second checkpoint
    "removed": ["old_layer"],     # Weights only in first checkpoint
    "similarity": {               # Cosine similarity for changed weights
        "layer1.weight": 0.9823
    }
}
```

This is useful for:
- Understanding what changed between training runs
- Debugging unexpected model behavior
- Identifying which layers were most affected by fine-tuning

## 6.4 TrainingState - State Encapsulation

`TrainingState` is a dataclass that captures complete training state at a checkpoint. It's designed to be framework-agnostic and comprehensive.

### Core Training Metrics

Basic fields for training progress:

```python
from coral.training import TrainingState

state = TrainingState(
    epoch=10,                    # Current epoch number
    global_step=5000,            # Total training steps across all epochs
    learning_rate=0.0001,        # Current learning rate
    loss=0.234,                  # Current loss value
    metrics={                    # Additional metrics
        "accuracy": 0.95,
        "val_loss": 0.189,
        "val_accuracy": 0.93,
        "f1_score": 0.94,
    },
)
```

### Optimizer and Scheduler States

Optional fields for exact training resumption:

```python
state = TrainingState(
    epoch=10,
    global_step=5000,
    learning_rate=0.0001,
    loss=0.234,

    # Optimizer state (e.g., Adam momentum buffers)
    optimizer_state=optimizer.state_dict(),

    # Scheduler state (e.g., StepLR counter)
    scheduler_state=scheduler.state_dict(),

    # Random state for reproducibility
    random_state={
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state(),
    },
)
```

### Training Configuration

Optional metadata about training setup:

```python
state = TrainingState(
    epoch=10,
    global_step=5000,
    learning_rate=0.0001,
    loss=0.234,

    # Training configuration
    batch_size=32,
    gradient_accumulation_steps=4,
    max_epochs=100,
    max_steps=50000,
)
```

### Metadata Fields

Additional context for experiments:

```python
state = TrainingState(
    epoch=10,
    global_step=5000,
    learning_rate=0.0001,
    loss=0.234,

    # Metadata
    model_name="ResNet50",
    dataset_name="ImageNet",
    experiment_name="resnet50_lr_sweep",
    notes="Testing learning rate 0.001 with cosine annealing",
)
```

### Serialization

TrainingState supports JSON serialization:

```python
# Save to file
state.save("/path/to/state.json")

# Load from file
loaded_state = TrainingState.load("/path/to/state.json")

# Convert to dictionary
state_dict = state.to_dict()

# Create from dictionary
state = TrainingState.from_dict(state_dict)
```

**PyTorch Tensor Handling**: When serializing optimizer/scheduler states containing PyTorch tensors, they're automatically converted to nested lists and reconstructed on load.

### Utility Methods

```python
# Update metrics dynamically
state.update_metrics(accuracy=0.96, val_accuracy=0.94)

# Format a summary string
summary = state.format_summary()
# "Epoch: 10 | Step: 5000 | Learning Rate: 1.00e-04 | Loss: 0.2340 | accuracy: 0.9600"
```

## 6.5 PyTorch Integration

The `PyTorchIntegration` class provides static utility methods for converting between PyTorch models and Coral's weight representation.

### model_to_weights() Conversion

Convert a PyTorch model to Coral weights:

```python
from coral.integrations.pytorch import PyTorchIntegration
import torch.nn as nn

# Your PyTorch model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

# Convert to Coral weights
weights = PyTorchIntegration.model_to_weights(model)

# weights is a Dict[str, WeightTensor]
for name, weight in weights.items():
    print(f"{name}: {weight.shape}")
# Output:
# 0.weight: (128, 784)
# 0.bias: (128,)
# 2.weight: (10, 128)
# 2.bias: (10,)
```

**Conversion Process**:
1. Iterate over all model parameters via `model.named_parameters()`
2. Convert each parameter to NumPy array (detached, CPU)
3. Create `WeightTensor` with appropriate metadata
4. Infer layer type from parameter's parent module

### weights_to_model() Loading

Load Coral weights back into a PyTorch model:

```python
# Load weights into model
PyTorchIntegration.weights_to_model(weights, model)

# Model parameters are now updated
# Note: Uses strict=False, so missing/unexpected keys are ignored
```

**Loading Behavior**:
- Converts each `WeightTensor` back to PyTorch tensor
- Calls `model.load_state_dict()` with `strict=False`
- Missing keys in weights are skipped
- Extra keys in weights are ignored

### save_optimizer_state() and load_optimizer_state()

Preserve optimizer state for exact training continuation:

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)

# After some training...
optimizer_state = PyTorchIntegration.save_optimizer_state(optimizer)

# Later, restore optimizer state
PyTorchIntegration.load_optimizer_state(optimizer, optimizer_state)
```

This preserves:
- Momentum buffers (for Adam, SGD with momentum)
- Second moment estimates (for Adam)
- Step counts
- Per-parameter state

### save_scheduler_state() and load_scheduler_state()

Preserve learning rate scheduler state:

```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# After some epochs...
scheduler_state = PyTorchIntegration.save_scheduler_state(scheduler)

# Later, restore scheduler state
PyTorchIntegration.load_scheduler_state(scheduler, scheduler_state)
```

This preserves the scheduler's internal state (e.g., epoch counter for StepLR, plateau counter for ReduceLROnPlateau).

### Random State Management

For reproducible training resumption:

```python
# Save random state before checkpoint
random_state = PyTorchIntegration.get_random_state()

# random_state contains:
# {
#     "torch": torch.get_rng_state(),
#     "torch_cuda": [cuda_state_per_device] or None,
# }

# Later, restore random state
PyTorchIntegration.set_random_state(random_state)
```

This ensures that training resumed from a checkpoint produces identical results (assuming deterministic operations).

## 6.6 CoralTrainer - High-Level Training Wrapper

`CoralTrainer` provides a high-level API that wraps PyTorch training loops with automatic Coral checkpointing. It's designed to minimize code changes while maximizing functionality.

### Initialization with Repository

```python
from coral import Repository
from coral.integrations.pytorch import CoralTrainer
from coral.training import CheckpointConfig
import torch.nn as nn

# Create model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10),
)

# Create or open repository
repo = Repository("./mnist_model", init=True)

# Configure checkpointing
config = CheckpointConfig(
    save_every_n_epochs=1,
    save_on_best_metric="val_accuracy",
    minimize_metric=False,
    keep_last_n_checkpoints=10,
    keep_best_n_checkpoints=5,
)

# Initialize trainer
trainer = CoralTrainer(
    model=model,
    repository=repo,
    experiment_name="mnist_baseline",
    checkpoint_config=config,
)
```

### Setting Optimizer and Scheduler

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Create optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=50)

# Register with trainer
trainer.set_optimizer(optimizer)
trainer.set_scheduler(scheduler)
```

The trainer will automatically save and restore optimizer/scheduler states with checkpoints.

### step() for Training Updates

Call `step()` during training to update metrics:

```python
# Inside training loop
for batch_idx, (data, target) in enumerate(dataloader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # Update trainer with loss and metrics
    if batch_idx % 10 == 0:
        accuracy = compute_accuracy(output, target)
        trainer.step(loss=loss.item(), accuracy=accuracy)
```

**What step() does**:
1. Increments `global_step` counter
2. Updates training metrics
3. Calls step-end callbacks
4. Checks if checkpoint should be saved (based on `save_every_n_steps`)
5. Automatically saves checkpoint if conditions are met

### epoch_end() for End-of-Epoch

Call at the end of each epoch:

```python
for epoch in range(num_epochs):
    model.train()
    # ... training loop with trainer.step() ...

    # Validation
    val_loss, val_accuracy = validate(model, val_loader)

    # Update epoch-level metrics
    trainer.update_metrics(val_loss=val_loss, val_accuracy=val_accuracy)

    # End of epoch (triggers epoch-based checkpointing)
    trainer.epoch_end(epoch)
```

**What epoch_end() does**:
1. Updates current epoch
2. Calls epoch-end callbacks
3. Checks if checkpoint should be saved (based on `save_every_n_epochs` or metrics)
4. Steps the scheduler (if set)
5. Automatically saves checkpoint if conditions are met

### Manual Checkpoint Control

Override automatic behavior with manual control:

```python
# Force save a checkpoint (ignore conditions)
commit_hash = trainer.save_checkpoint(force=True)

# Load latest checkpoint
trainer.load_checkpoint()

# Load specific checkpoint
trainer.load_checkpoint(commit_hash="a7f3c2d1")

# Load best checkpoint
trainer.load_checkpoint(load_best=True)

# Load with control over what to restore
trainer.load_checkpoint(
    load_best=True,
    load_optimizer=True,
    load_scheduler=True,
)
```

### Callback Registration

Register callbacks for custom behavior:

```python
def on_epoch_end(trainer):
    """Called at the end of each epoch."""
    print(f"Epoch {trainer.current_epoch} completed")
    print(f"  Loss: {trainer.training_metrics.get('loss', 0):.4f}")
    print(f"  Accuracy: {trainer.training_metrics.get('accuracy', 0):.4f}")

def on_checkpoint_save(trainer, commit_hash):
    """Called after saving a checkpoint."""
    print(f"💾 Checkpoint saved: {commit_hash[:8]}")
    # Could log to external system, send notifications, etc.

# Register callbacks
trainer.add_callback("epoch_end", on_epoch_end)
trainer.add_callback("checkpoint_save", on_checkpoint_save)
trainer.add_callback("step_end", on_step_end)  # Optional
```

**Available Callback Types**:
- `epoch_end`: Called at the end of each epoch
- `step_end`: Called at the end of each training step
- `checkpoint_save`: Called after saving a checkpoint

### Complete Training Loop Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from coral import Repository
from coral.integrations.pytorch import CoralTrainer
from coral.training import CheckpointConfig

# 1. Define model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.layers(x)

# 2. Create data loaders
train_dataset = TensorDataset(torch.randn(10000, 784), torch.randint(0, 10, (10000,)))
val_dataset = TensorDataset(torch.randn(2000, 784), torch.randint(0, 10, (2000,)))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 3. Initialize Coral
model = MLP()
repo = Repository("./mlp_mnist", init=True)
config = CheckpointConfig(
    save_every_n_epochs=2,
    save_on_best_metric="val_accuracy",
    minimize_metric=False,
    keep_last_n_checkpoints=5,
    keep_best_n_checkpoints=3,
)

trainer = CoralTrainer(
    model=model,
    repository=repo,
    experiment_name="mlp_baseline",
    checkpoint_config=config,
)

# 4. Setup training
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()

trainer.set_optimizer(optimizer)
trainer.set_scheduler(scheduler)

# 5. Training loop
num_epochs = 20

for epoch in range(num_epochs):
    # Training
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update trainer
        if batch_idx % 10 == 0:
            pred = output.argmax(dim=1)
            accuracy = pred.eq(target).float().mean().item()
            trainer.step(loss=loss.item(), accuracy=accuracy)

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            val_correct += pred.eq(target).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = val_correct / len(val_dataset)

    # Update metrics and end epoch
    trainer.update_metrics(val_loss=val_loss, val_accuracy=val_accuracy)
    trainer.epoch_end(epoch)

    print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}")

# 6. Training complete
print("\n=== Training Summary ===")
summary = trainer.get_training_summary()
for key, value in summary.items():
    if key != "metrics":
        print(f"{key}: {value}")

# List checkpoints
checkpoints = trainer.list_checkpoints()
print(f"\nSaved {len(checkpoints)} checkpoints")

# Load best model
trainer.load_checkpoint(load_best=True)
print("Loaded best checkpoint for inference")
```

## 6.7 Checkpointer - Simplified PyTorch API

The `Checkpointer` class provides a streamlined, Pythonic API for PyTorch checkpointing with context manager support, automatic resume, and experiment tracking integration.

### Basic Usage

```python
from coral.integrations.pytorch import Checkpointer

# Create checkpointer with simple parameters
ckpt = Checkpointer(
    model,
    "./checkpoints",          # Repository path (or Repository object)
    "my-experiment",          # Experiment name
    every_n_epochs=1,         # Save every epoch
    on_best="val_loss",       # Save when val_loss improves
    minimize=True,            # Lower is better
    keep_last=5,              # Keep 5 most recent
    keep_best=3,              # Keep 3 best
)
```

### Context Manager Pattern

Use as a context manager for clean experiment tracking:

```python
from coral.integrations.pytorch import Checkpointer

with Checkpointer(model, repo, "experiment-1") as ckpt:
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader)
        val_loss = validate(model, val_loader)

        # Log metrics and save if needed
        ckpt.log(epoch, step, loss=train_loss, val_loss=val_loss)
```

The context manager automatically:
- Starts an experiment tracking run (if tracker provided)
- Ends the run with status "completed" or "failed"
- Handles exceptions gracefully

### Automatic Resume

Resume training from the latest or best checkpoint:

```python
# Resume from latest checkpoint if available
ckpt = Checkpointer(
    model,
    repo,
    "experiment-1",
    resume=True,  # Auto-resume from latest
)

# Resume from best checkpoint
ckpt = Checkpointer(
    model,
    repo,
    "experiment-1",
    resume="best",  # Resume from best metric
)

# Check resume status
print(f"Resumed from epoch: {ckpt.epoch}")
print(f"Starting step: {ckpt.global_step}")
```

### Experiment Tracking Integration

Integrate with MLflow or W&B through the experiment bridge:

```python
from coral.integrations.pytorch import Checkpointer
from coral.integrations import MLflowBridge

tracker = MLflowBridge(experiment_name="my-mlflow-experiment")

with Checkpointer(model, repo, "experiment", tracker=tracker) as ckpt:
    for epoch in range(num_epochs):
        # Metrics automatically logged to MLflow
        ckpt.log(epoch, step, loss=loss, accuracy=accuracy)
```

### Properties and State Access

```python
ckpt = Checkpointer(model, repo, "experiment")

# Current training state
print(ckpt.epoch)         # Current epoch
print(ckpt.global_step)   # Current step

# Best checkpoint info
print(ckpt.best_commit)   # Commit hash of best checkpoint
print(ckpt.metrics)       # Current metric values

# Repository access
print(ckpt.repo)          # Underlying Repository object
```

### Complete Example

```python
import torch
import torch.nn as nn
from coral.integrations.pytorch import Checkpointer

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

with Checkpointer(
    model,
    "./mnist_checkpoints",
    "mnist-classifier",
    every_n_epochs=1,
    on_best="val_loss",
    minimize=True,
    keep_last=5,
    keep_best=3,
    resume=True,
) as ckpt:
    start_epoch = ckpt.epoch

    for epoch in range(start_epoch, 20):
        # Training
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Validation
        val_loss = validate(model, val_loader)

        # Log and potentially save
        ckpt.log(epoch, global_step, loss=loss.item(), val_loss=val_loss)

    print(f"Best checkpoint: {ckpt.best_commit}")
```

## 6.8 Framework Integration Examples

Coral provides integration layers for popular ML frameworks, enabling automatic checkpointing with minimal code changes.

### PyTorch Complete Example

See the complete example in Section 6.6. Key points:

```python
from coral.integrations.pytorch import CoralTrainer, PyTorchIntegration

# Direct API (lower-level)
weights = PyTorchIntegration.model_to_weights(model)
repo.stage_weights(weights)
repo.commit("Save checkpoint")

# Trainer API (higher-level)
trainer = CoralTrainer(model, repo, experiment_name="my_experiment")
trainer.set_optimizer(optimizer)
trainer.epoch_end(epoch)  # Automatic checkpointing
```

### Unified load() and save() Functions

For simple one-off operations, use the unified `load()` and `save()` functions:

```python
from coral.integrations.pytorch import load, save

# Save model to repository
result = save(model, repo, "Checkpoint after epoch 10")
print(f"Saved commit: {result['commit_hash']}")
print(f"Weights saved: {result['weights_saved']}")

# Load model from repository
result = load(model, repo, commit="a7f3c2d1", strict=True)
print(f"Loaded {len(result['loaded'])} weights")
print(f"Matched: {result['matched']}")
```

These functions provide a simpler alternative to the full `CoralTrainer` when you just need basic save/load functionality.

### PyTorch Lightning Integration

For PyTorch Lightning users, Coral provides a callback:

```python
import pytorch_lightning as pl
from coral.integrations.lightning import CoralCallback
# Alias also available:
# from coral.integrations.lightning import CoralLightningCallback

# Create Coral callback with repository object
from coral import Repository
repo = Repository("./lightning_model", init=True)

coral_callback = CoralCallback(
    repo=repo,                  # Can also use repo_path="./lightning_model"
    save_every_n_epochs=1,
    save_on_best="val_loss",
    mode="min",
    branch="experiment_1",
    push_to=None,  # Optional: push to remote after each save
)

# Add to Lightning Trainer
trainer = pl.Trainer(
    max_epochs=50,
    callbacks=[coral_callback],
)

# Train normally - Coral handles checkpointing automatically
trainer.fit(model, train_dataloader, val_dataloader)

# Load best model from Coral
coral_callback.load_from_coral(model, commit_ref=None)
```

**CoralCallback Parameters**:
- `repo_path`: Path to Coral repository
- `init`: Initialize repository if it doesn't exist
- `save_every_n_epochs`: Checkpoint frequency (epochs)
- `save_every_n_steps`: Checkpoint frequency (steps)
- `save_on_best`: Metric name to monitor (e.g., "val_loss")
- `mode`: "min" or "max" for metric optimization
- `branch`: Git branch for checkpoints
- `push_to`: Remote name for automatic pushing
- `include_optimizer`: Save optimizer state
- `metadata_keys`: List of trainer attributes to include in metadata

**Automatic Features**:
- Saves at epoch/step intervals
- Saves when validation metrics improve
- Creates commits with metric metadata
- Optionally pushes to remote repositories
- Integrates with Lightning's callback system

### HuggingFace Transformers Integration

For transformer models with the HuggingFace Trainer:

```python
from transformers import Trainer, TrainingArguments
from coral.integrations.hf_trainer import CoralTrainerCallback
# Alias also available:
# from coral.integrations.hf_trainer import CoralHFCallback

# Create Coral callback with repository object
from coral import Repository
repo = Repository("./bert_finetuned", init=True)

coral_callback = CoralTrainerCallback(
    repo=repo,                  # Can also use repo_path="./bert_finetuned"
    save_every_n_steps=1000,
    save_on_best="eval_accuracy",
    mode="max",
    branch="bert_finetuning",
)

# Configure HuggingFace Trainer
training_args = TrainingArguments(
    output_dir="./hf_output",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    eval_steps=500,
)

# Create Trainer with Coral callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[coral_callback],
)

# Train - Coral handles versioning
trainer.train()

# Load best model
coral_callback.load_from_coral(model)
```

**CoralTrainerCallback Features**:
- Integrates with HuggingFace Trainer lifecycle
- Saves on step/epoch boundaries
- Monitors evaluation metrics
- Creates version-controlled checkpoints
- Supports distributed training
- Handles model sharding/parallelism

### Framework-Agnostic Patterns

For other frameworks or custom training loops, use the core components directly:

```python
from coral import Repository
from coral.training import CheckpointManager, CheckpointConfig, TrainingState
from coral.core.weight_tensor import WeightTensor, WeightMetadata
import numpy as np

# 1. Initialize repository and checkpoint manager
repo = Repository("./custom_framework", init=True)
config = CheckpointConfig(
    save_every_n_epochs=5,
    save_on_best_metric="val_loss",
    minimize_metric=True,
)
manager = CheckpointManager(repo, config, model_name="CustomModel")

# 2. Training loop
for epoch in range(num_epochs):
    # ... your training code ...

    # Convert your model weights to Coral format
    weights = {}
    for name, param in your_model.get_parameters():
        weights[name] = WeightTensor(
            data=param.astype(np.float32),
            metadata=WeightMetadata(
                name=name,
                shape=param.shape,
                dtype=param.dtype,
            ),
        )

    # Create training state
    state = TrainingState(
        epoch=epoch,
        global_step=global_step,
        learning_rate=current_lr,
        loss=train_loss,
        metrics={"val_loss": val_loss, "val_accuracy": val_accuracy},
    )

    # Save checkpoint (automatic based on config)
    commit_hash = manager.save_checkpoint(weights, state)

    if commit_hash:
        print(f"Saved checkpoint: {commit_hash[:8]}")

# 3. Load checkpoint for inference
checkpoint = manager.load_checkpoint(load_best=True)
if checkpoint:
    for name, weight_tensor in checkpoint["weights"].items():
        your_model.set_parameter(name, weight_tensor.data)
```

**Key Integration Points**:
1. **Weight Conversion**: Convert framework-specific parameters to NumPy arrays
2. **Metadata Creation**: Populate `WeightMetadata` with parameter information
3. **State Tracking**: Create `TrainingState` with metrics and configuration
4. **Checkpoint Lifecycle**: Use `CheckpointManager` for save/load/cleanup

## 6.9 Best Practices

### Checkpoint Frequency Recommendations

**Research/Debugging**:
- Save every 1-5 epochs
- Save every 100-500 steps
- Save on metric improvements
- Keep all checkpoints for analysis

**Production Training**:
- Save every 10-20 epochs
- Save on metric improvements only
- Aggressive cleanup (keep last 5, best 3)
- Reduce storage overhead

**Long Training Runs** (days/weeks):
- Save every 5-10 epochs
- Save every 1000-5000 steps
- Keep periodic checkpoints (every 25-50 epochs)
- Use delta encoding for storage efficiency

**Fine-tuning**:
- Save on metric improvements only
- Keep best 5-10 checkpoints
- Don't need optimizer state (if not resuming)

### Metric-Based Saving Strategies

**Classification**:
```python
config = CheckpointConfig(
    save_on_best_metric="val_accuracy",
    minimize_metric=False,  # Maximize accuracy
    keep_best_n_checkpoints=10,
)
```

**Regression**:
```python
config = CheckpointConfig(
    save_on_best_metric="val_mae",  # Mean Absolute Error
    minimize_metric=True,
    keep_best_n_checkpoints=5,
)
```

**Multi-Metric**:
```python
# Primary metric for checkpointing
config = CheckpointConfig(
    save_on_best_metric="val_f1",
    minimize_metric=False,
    # But track multiple metrics
)

# In training loop
state.metrics = {
    "val_f1": f1_score,
    "val_precision": precision,
    "val_recall": recall,
    "val_loss": loss,
}
```

**Composite Metrics**:
```python
# Define custom metric
weighted_score = 0.7 * accuracy + 0.3 * (1 - loss)

state.metrics = {
    "val_accuracy": accuracy,
    "val_loss": loss,
    "weighted_score": weighted_score,
}

config = CheckpointConfig(
    save_on_best_metric="weighted_score",
    minimize_metric=False,
)
```

### Storage Optimization Tips

**Enable Delta Encoding**:
```python
config = CheckpointConfig(
    use_incremental_saves=True,  # Leverage deduplication
)
```

**Reduce State Overhead**:
```python
# For production deployment (not resuming training)
config = CheckpointConfig(
    save_optimizer_state=False,  # Don't need optimizer
    save_scheduler_state=False,  # Don't need scheduler
    save_random_state=False,     # Don't need reproducibility
)
```

**Aggressive Cleanup**:
```python
config = CheckpointConfig(
    keep_last_n_checkpoints=3,    # Only recent checkpoints
    keep_best_n_checkpoints=5,    # Only best checkpoints
    keep_checkpoint_every_n_epochs=None,  # No periodic keeping
)
```

**Periodic Garbage Collection**:
```python
# After training or periodically
repo.gc()  # Clean up unreferenced weights and deltas
```

### Recovery from Interruptions

**Automatic Resume**:
```python
# At start of training script
checkpoint = manager.load_checkpoint()  # Load latest

if checkpoint:
    # Resume training
    state = checkpoint["state"]
    start_epoch = state.epoch + 1
    global_step = state.global_step

    # Restore model
    PyTorchIntegration.weights_to_model(checkpoint["weights"], model)

    # Restore optimizer
    if state.optimizer_state:
        PyTorchIntegration.load_optimizer_state(optimizer, state.optimizer_state)

    # Restore scheduler
    if state.scheduler_state:
        PyTorchIntegration.load_scheduler_state(scheduler, state.scheduler_state)

    # Restore random state for determinism
    if state.random_state:
        PyTorchIntegration.set_random_state(state.random_state)

    print(f"Resumed from epoch {state.epoch}, step {state.global_step}")
else:
    # Fresh training
    start_epoch = 0
    global_step = 0
    print("Starting fresh training")

# Continue training from start_epoch
for epoch in range(start_epoch, num_epochs):
    # ... training ...
```

**Checkpoint Validation**:
```python
# Verify checkpoint integrity
checkpoint = manager.load_checkpoint()

if checkpoint:
    weights = checkpoint["weights"]
    state = checkpoint["state"]

    # Check for expected weights
    expected_keys = set(model.state_dict().keys())
    actual_keys = set(weights.keys())

    missing = expected_keys - actual_keys
    unexpected = actual_keys - expected_keys

    if missing:
        print(f"Warning: Missing weights: {missing}")
    if unexpected:
        print(f"Warning: Unexpected weights: {unexpected}")

    # Verify metrics are reasonable
    if state.loss < 0 or state.loss > 1e6:
        print(f"Warning: Unusual loss value: {state.loss}")
```

**Backup Best Checkpoint**:
```python
# Export best checkpoint to external storage
checkpoint = manager.load_checkpoint(load_best=True)

if checkpoint:
    commit_hash = checkpoint["commit_hash"]

    # Create tag for easy reference
    repo.tag_version(
        name="production_v1.0",
        description="Best model for production deployment",
        commit_ref=commit_hash,
    )

    # Export to external storage
    import shutil
    backup_dir = "/backup/models"
    shutil.copytree(repo.repo_path, f"{backup_dir}/model_{commit_hash[:8]}")
```

**Distributed Training Considerations**:
```python
# Save checkpoints only on rank 0
if torch.distributed.get_rank() == 0:
    commit_hash = trainer.save_checkpoint(force=True)

# Broadcast checkpoint hash to all ranks
if torch.distributed.get_world_size() > 1:
    commit_hash_obj = [commit_hash] if rank == 0 else [None]
    torch.distributed.broadcast_object_list(commit_hash_obj, src=0)
    commit_hash = commit_hash_obj[0]

# All ranks can load the checkpoint
trainer.load_checkpoint(commit_hash=commit_hash)
```

## Summary

Coral's training integration provides production-grade checkpoint management with git-like version control. Key takeaways:

1. **Declarative Configuration**: `CheckpointConfig` separates policy from implementation
2. **Comprehensive State**: `TrainingState` captures everything needed for exact resumption
3. **Automatic Management**: `CheckpointManager` handles save/load/cleanup lifecycle
4. **Framework Support**: Integration layers for PyTorch, Lightning, and HuggingFace
5. **Storage Efficiency**: Delta encoding and deduplication minimize storage overhead
6. **Production Ready**: Callback system, metrics tracking, and error handling

Training integration transforms ad-hoc checkpoint saving into a structured, version-controlled workflow that scales from research to production.

**Next Chapter**: Chapter 7 covers the CLI interface, providing git-like commands for exploring, managing, and deploying versioned model weights.
