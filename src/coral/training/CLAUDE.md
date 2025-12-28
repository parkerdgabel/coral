# Training Module

This module provides checkpoint management and training state tracking for ML training workflows with Coral version control.

## Overview

The training module provides:
- **CheckpointConfig**: Configuration for checkpoint policies
- **CheckpointManager**: Manages training checkpoints with retention policies
- **TrainingState**: Captures complete training state for resumption

## Key Files

### `checkpoint_manager.py`

Manages training checkpoints with configurable save and retention policies.

**CheckpointConfig** (dataclass):
```python
CheckpointConfig(
    # When to save
    save_every_n_steps: Optional[int] = None,
    save_every_n_epochs: Optional[int] = None,
    save_on_best_metric: Optional[str] = None,
    minimize_metric: bool = True,  # True for loss, False for accuracy

    # What to keep
    keep_last_n_checkpoints: Optional[int] = None,
    keep_best_n_checkpoints: Optional[int] = None,
    keep_checkpoint_every_n_epochs: Optional[int] = None,

    # What to save
    save_optimizer_state: bool = True,
    save_scheduler_state: bool = True,
    save_random_state: bool = True,
    use_incremental_saves: bool = True,

    # Commit options
    auto_commit: bool = True,
    commit_message_template: str = "Checkpoint at epoch {epoch}, step {step}",
    tag_best_checkpoints: bool = True,
)
```

**CheckpointManager** (class):
```python
manager = CheckpointManager(
    repository=repo,
    config=CheckpointConfig(...),
    model_name="MyModel",
    experiment_name="experiment_001",
)

# Check if checkpoint should be saved
if manager.should_save_checkpoint(state):
    commit_hash = manager.save_checkpoint(weights, state)

# Load checkpoint
checkpoint_data = manager.load_checkpoint(
    commit_hash=None,    # Specific commit or None for latest
    load_best=False,     # Load best checkpoint instead
)
weights = checkpoint_data["weights"]
state = checkpoint_data["state"]

# List checkpoints
checkpoints = manager.list_checkpoints(
    include_metrics=True,
    only_best=False,
)

# Get info about specific checkpoint
info = manager.get_checkpoint_info(commit_hash)

# Register callbacks
def my_callback(state: TrainingState, commit_hash: Optional[str]):
    print(f"Checkpoint saved: {commit_hash}")

manager.register_checkpoint_callback(my_callback)
manager.unregister_checkpoint_callback(my_callback)
manager.clear_callbacks()
print(manager.list_callbacks())
```

**Checkpoint Storage**:
- Checkpoints are stored as commits in the repository
- Training state saved as JSON alongside commit
- Best checkpoints tagged as versions
- History tracked in `.coral/checkpoints/{experiment_name}.json`

### `training_state.py`

Captures complete training state for checkpoint resumption.

**TrainingState** (dataclass):
```python
TrainingState(
    # Core state
    epoch: int,
    global_step: int,
    learning_rate: float,
    loss: float,
    metrics: dict[str, float] = {},

    # Serialized states
    optimizer_state: Optional[dict] = None,
    scheduler_state: Optional[dict] = None,
    random_state: Optional[dict] = None,

    # Timing
    timestamp: datetime = datetime.now(),

    # Training config
    batch_size: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
    max_epochs: Optional[int] = None,
    max_steps: Optional[int] = None,

    # Metadata
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
    notes: Optional[str] = None,
)
```

**Methods**:
```python
# Update metrics
state.update_metrics(accuracy=0.95, f1=0.92)

# Format summary
print(state.format_summary())
# Output: "Epoch: 5 | Step: 1000 | Learning Rate: 1.00e-04 | Loss: 0.1234 | accuracy: 0.9500"

# Save/load
state.save("state.json")
loaded_state = TrainingState.load("state.json")

# Convert to/from dict
data = state.to_dict()
state = TrainingState.from_dict(data)
```

**Tensor Serialization**:
- PyTorch tensors in optimizer/scheduler/random states are automatically serialized
- Tensors converted to numpy arrays for JSON storage
- Deserialized back to tensors when loading

## Usage Examples

### Basic Checkpoint Management

```python
from coral import Repository
from coral.training import CheckpointConfig, CheckpointManager, TrainingState

# Setup
repo = Repository("./checkpoints", init=True)
config = CheckpointConfig(
    save_every_n_epochs=1,
    save_on_best_metric="val_loss",
    minimize_metric=True,
    keep_last_n_checkpoints=5,
    keep_best_n_checkpoints=3,
)
manager = CheckpointManager(repo, config, "MyModel", "experiment_001")

# Training loop
for epoch in range(10):
    for step, batch in enumerate(dataloader):
        loss = train_step(batch)

        # Create training state
        state = TrainingState(
            epoch=epoch,
            global_step=epoch * len(dataloader) + step,
            learning_rate=optimizer.param_groups[0]['lr'],
            loss=loss,
            metrics={"val_loss": val_loss},
        )

        # Check and save
        if manager.should_save_checkpoint(state):
            weights = convert_model_to_weights(model)
            commit_hash = manager.save_checkpoint(weights, state)
            print(f"Saved: {commit_hash}")

# Load best
best = manager.load_checkpoint(load_best=True)
```

### With PyTorch Integration

```python
from coral.integrations.pytorch import CoralTrainer
from coral.training import CheckpointConfig

config = CheckpointConfig(
    save_every_n_epochs=1,
    save_on_best_metric="val_loss",
    keep_best_n_checkpoints=3,
)

trainer = CoralTrainer(
    model=model,
    repository=repo,
    experiment_name="my_experiment",
    checkpoint_config=config,
)

# Trainer handles checkpointing automatically
for epoch in range(10):
    for batch in train_loader:
        loss = train_step(batch)
        trainer.step(loss, val_loss=val_loss)
    trainer.epoch_end(epoch)

# Resume from best
trainer.load_checkpoint(load_best=True)
```

### Checkpoint Callbacks

```python
from coral.training import CheckpointManager, TrainingState

def log_to_wandb(state: TrainingState, commit_hash: Optional[str]):
    """Log checkpoint to Weights & Biases."""
    wandb.log({
        "checkpoint_epoch": state.epoch,
        "checkpoint_step": state.global_step,
        "checkpoint_hash": commit_hash,
    })

def notify_slack(state: TrainingState, commit_hash: Optional[str]):
    """Send Slack notification on checkpoint."""
    if state.metrics.get("val_loss", 1.0) < 0.1:
        send_slack_message(f"Great checkpoint: {commit_hash}")

manager.register_checkpoint_callback(log_to_wandb)
manager.register_checkpoint_callback(notify_slack)

# Later
manager.unregister_checkpoint_callback(log_to_wandb)
```

## Retention Policies

The CheckpointManager automatically manages checkpoint retention:

1. **keep_last_n_checkpoints**: Always keep the N most recent
2. **keep_best_n_checkpoints**: Always keep the N best by metric
3. **keep_checkpoint_every_n_epochs**: Keep periodic checkpoints (e.g., every 10 epochs)
4. **Best checkpoint**: Always preserved regardless of other policies

Example:
```python
config = CheckpointConfig(
    save_every_n_epochs=1,           # Save every epoch
    keep_last_n_checkpoints=5,       # Keep last 5
    keep_best_n_checkpoints=3,       # Keep 3 best
    keep_checkpoint_every_n_epochs=10, # Keep every 10th epoch
)
```

With 50 epochs, this keeps:
- Last 5 (epochs 46-50)
- Best 3 by metric
- Every 10th (epochs 10, 20, 30, 40, 50)
- Total: up to ~13 checkpoints instead of 50

## Dependencies

- Internal: `coral.core.weight_tensor`, `coral.version_control.repository`

## Testing

Related test files:
- `tests/test_training.py` - Checkpoint manager tests
