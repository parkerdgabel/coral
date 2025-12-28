# Experiments Module

This module provides experiment tracking with metrics logging, comparison, and best model finding capabilities.

## Overview

The experiments module provides:
- **ExperimentTracker**: Main class for tracking experiments
- **Experiment**: Represents a single experiment/training run
- **ExperimentMetric**: A single metric measurement
- **ExperimentStatus**: Status enum (pending, running, completed, failed, cancelled)

## Key Files

### `experiment.py`

Experiment tracking implementation.

**ExperimentStatus** (Enum):
```python
ExperimentStatus.PENDING    # Not started
ExperimentStatus.RUNNING    # In progress
ExperimentStatus.COMPLETED  # Finished successfully
ExperimentStatus.FAILED     # Finished with error
ExperimentStatus.CANCELLED  # Stopped by user
```

**ExperimentMetric** (dataclass):
```python
ExperimentMetric(
    name: str,              # e.g., "loss", "accuracy"
    value: float,
    step: int | None = None,
    timestamp: datetime = datetime.now(),
    metadata: dict = {}
)
```

**Experiment** (dataclass):
```python
Experiment(
    experiment_id: str,
    name: str,
    description: str | None = None,
    status: ExperimentStatus = ExperimentStatus.PENDING,
    created_at: datetime,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
    metrics: list[ExperimentMetric] = [],
    params: dict = {},
    tags: list[str] = [],
    commit_hash: str | None = None,
    branch: str | None = None,
    parent_experiment: str | None = None,
    notes: str | None = None
)
```

**Experiment Methods**:
```python
# Get metric history
history = exp.get_metric_history("loss")

# Get latest metric value
latest = exp.get_latest_metric("accuracy")

# Get best metric value
best = exp.get_best_metric("loss", mode="min")

# Duration in seconds
print(f"Duration: {exp.duration:.0f}s")

# List all metric names
print(exp.metric_names)
```

**ExperimentTracker** (class):
```python
from coral import Repository
from coral.experiments import ExperimentTracker

repo = Repository("./my-model")
tracker = ExperimentTracker(repo)
```

## Basic Workflow

### Starting and Ending Experiments

```python
# Start experiment
exp = tracker.start(
    name="bert-finetuning",
    description="Fine-tuning BERT on custom dataset",
    params={"lr": 0.001, "batch_size": 32, "epochs": 10},
    tags=["bert", "classification"]
)

# Training loop
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        loss, acc = train_step(batch)

        # Log metrics
        tracker.log("loss", loss, step=step)
        tracker.log("accuracy", acc, step=step)

    # Log end-of-epoch metrics
    val_loss, val_acc = validate()
    tracker.log_metrics({"val_loss": val_loss, "val_acc": val_acc})

# End experiment with commit
commit = repo.commit("Training complete")
tracker.end(commit_hash=commit.commit_hash)
```

### Logging Methods

```python
# Log single metric
tracker.log("loss", 0.5, step=100)

# Log multiple metrics at once
tracker.log_metrics({"loss": 0.5, "accuracy": 0.85}, step=100)

# Log/update parameters
tracker.log_params({"epochs": 20, "scheduler": "cosine"})

# Add tags
tracker.add_tags(["production", "v2"])

# Set notes
tracker.set_notes("Trying new learning rate schedule")
```

### Handling Failures

```python
try:
    # Training code
    ...
except Exception as e:
    tracker.fail(error_message=str(e))
    raise
```

## Querying Experiments

### List Experiments

```python
# List all experiments
experiments = tracker.list()

# Filter by status
running = tracker.list(status=ExperimentStatus.RUNNING)
completed = tracker.list(status=ExperimentStatus.COMPLETED)

# Filter by tags
bert_exps = tracker.list(tags=["bert"])

# Filter by branch
main_exps = tracker.list(branch="main")

# Limit results
recent = tracker.list(limit=10)
```

### Get Specific Experiment

```python
# By ID
exp = tracker.get("abc123def456")

# By name (returns list - multiple exps can have same name)
exps = tracker.get_by_name("bert-finetuning")
```

### Compare Experiments

```python
comparison = tracker.compare(
    experiment_ids=["exp1", "exp2", "exp3"],
    metrics=["loss", "accuracy"]  # None for all metrics
)

print(comparison["experiments"])  # Basic info
print(comparison["metrics"])      # Metric values
print(comparison["params"])       # Parameter differences
```

### Find Best Experiments

```python
# Find experiments with lowest loss
best_loss = tracker.find_best(
    metric="loss",
    mode="min",
    status=ExperimentStatus.COMPLETED,
    limit=5
)

for result in best_loss:
    print(f"{result['name']}: loss={result['best_value']:.4f}")
    print(f"  Commit: {result['commit_hash']}")
    print(f"  Params: {result['params']}")

# Find experiments with highest accuracy
best_acc = tracker.find_best(metric="accuracy", mode="max")
```

## Resume and Management

### Resume Experiment

```python
# Resume a failed/stopped experiment
new_exp = tracker.resume("original_experiment_id")
# Creates new experiment with same params, linked as parent
```

### Delete Experiment

```python
tracker.delete("experiment_id")
```

### Get Summary

```python
summary = tracker.get_summary()
print(f"Total: {summary['total_experiments']}")
print(f"By status: {summary['by_status']}")
print(f"Unique names: {summary['unique_names']}")
print(f"Branches: {summary['branches']}")
print(f"Metrics logged: {summary['total_metrics_logged']}")
```

## Storage

Experiments are stored in `.coral/experiments/`:
```
.coral/experiments/
├── abc123def456.json
├── xyz789ghi012.json
└── ...
```

Each experiment is a JSON file with full metadata and metric history.

## Usage Examples

### Complete Training Example

```python
from coral import Repository
from coral.experiments import ExperimentTracker

repo = Repository("./checkpoints", init=True)
tracker = ExperimentTracker(repo)

# Start experiment
exp = tracker.start(
    name="resnet-training",
    params={"lr": 0.01, "momentum": 0.9}
)

try:
    best_val_loss = float('inf')

    for epoch in range(100):
        train_loss = train_epoch(model, train_loader)
        val_loss, val_acc = validate(model, val_loader)

        # Log metrics
        tracker.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        }, step=epoch)

        # Save checkpoint on improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            weights = model_to_weights(model)
            repo.stage_weights(weights)
            commit = repo.commit(f"Epoch {epoch}: val_loss={val_loss:.4f}")

    # End successfully
    tracker.end(commit_hash=commit.commit_hash)

except Exception as e:
    tracker.fail(str(e))
    raise
```

### Finding Best Model Across Experiments

```python
# Find top 3 experiments by validation accuracy
results = tracker.find_best(
    metric="val_accuracy",
    mode="max",
    tags=["production-ready"],
    limit=3
)

# Load best model
best = results[0]
weights = repo.get_all_weights(best["commit_hash"])
```

## Dependencies

- Internal: `coral.version_control.repository`

## Testing

Related test files:
- `tests/test_experiments.py` - Experiment tracking tests
