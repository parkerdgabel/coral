# Registry Module

This module provides model publishing capabilities to various model registries including Hugging Face Hub, MLflow, and local export.

## Overview

The registry module provides:
- **ModelPublisher**: Main class for publishing models
- **RegistryType**: Enum of supported registries
- **PublishResult**: Result of a publish operation

## Key Files

### `registry.py`

Model publishing implementation.

**RegistryType** (Enum):
```python
RegistryType.HUGGINGFACE  # Hugging Face Hub
RegistryType.MLFLOW       # MLflow Model Registry
RegistryType.LOCAL        # Local file export
```

**PublishResult** (dataclass):
```python
PublishResult(
    success: bool,
    registry: RegistryType,
    model_name: str,
    version: str | None = None,
    url: str | None = None,
    error: str | None = None,
    metadata: dict = {},
    published_at: datetime = datetime.now()
)
```

**ModelPublisher** (class):
```python
from coral import Repository
from coral.registry import ModelPublisher

repo = Repository("./my-model")
publisher = ModelPublisher(repo)
```

## Publishing Methods

### Hugging Face Hub

```python
result = publisher.publish_huggingface(
    repo_id="username/my-model",
    commit_ref=None,              # None for HEAD
    private=False,
    description="My fine-tuned model",
    base_model="bert-base-uncased",
    metrics={"accuracy": 0.95, "f1": 0.92},
    tags=["bert", "text-classification"],
    token=None                    # Uses HF_TOKEN env or login
)

if result.success:
    print(f"Published: {result.url}")
else:
    print(f"Error: {result.error}")
```

**Output Files**:
- `model.safetensors` - Model weights
- `README.md` - Generated model card
- `coral_metadata.json` - Coral version info

### MLflow Registry

```python
result = publisher.publish_mlflow(
    model_name="my-model",
    commit_ref=None,
    tracking_uri="http://mlflow:5000",
    experiment_name="production-models",
    description="Production model v1.0",
    tags={"team": "ml", "stage": "production"},
    metrics={"accuracy": 0.95}
)

if result.success:
    print(f"Version: {result.version}")
    print(f"URL: {result.url}")
```

**MLflow Artifacts**:
- `weights.npz` - NumPy weights archive
- Metrics logged to run
- Tags and parameters recorded

### Local Export

```python
result = publisher.publish_local(
    output_path="./exported-model",
    commit_ref=None,
    format="safetensors",  # or "npz", "pt"
    include_metadata=True
)

if result.success:
    print(f"Exported to: {result.url}")
```

**Supported Formats**:
| Format | Extension | Notes |
|--------|-----------|-------|
| `safetensors` | `.safetensors` | Recommended, safe and fast |
| `npz` | `.npz` | NumPy compressed archive |
| `pt` | `.pt` | PyTorch state dict |

## History Tracking

```python
# Get publish history
history = publisher.get_history(
    registry=RegistryType.HUGGINGFACE,  # Filter by registry
    success_only=True,
    limit=50
)

for result in history:
    print(f"{result.model_name}: {result.url}")

# Get latest publish for a model
latest = publisher.get_latest("username/my-model")
if latest:
    print(f"Last published: {latest.published_at}")
```

History is stored in `.coral/registry/history.json`.

## Model Card Generation

The publisher automatically generates model cards:

```markdown
---
tags: ["bert", "text-classification"]
base_model: bert-base-uncased
library_name: coral
model-index:
  - name: my-model
    results:
      - task: accuracy
        value: 0.95
---

# my-model

My fine-tuned model

## Model Details

This model was versioned and published using [Coral](https://github.com/parkerdgabel/coral).

**Base Model**: [bert-base-uncased](https://huggingface.co/bert-base-uncased)

## Metrics

| Metric | Value |
|--------|-------|
| accuracy | 0.9500 |
| f1 | 0.9200 |
```

## Usage Examples

### Publish Training Result

```python
from coral import Repository
from coral.registry import ModelPublisher

# After training
repo = Repository("./checkpoints")
publisher = ModelPublisher(repo)

# Find best checkpoint
checkpoints = repo.log(max_commits=10)
best_commit = checkpoints[0].commit_hash  # Or find by metric

# Publish to HF Hub
result = publisher.publish_huggingface(
    repo_id="myorg/my-model",
    commit_ref=best_commit,
    metrics={"val_accuracy": 0.95},
    base_model="bert-base-uncased",
)
print(f"Published: {result.url}")
```

### Multi-Registry Publishing

```python
# Publish to multiple registries
results = []

# HuggingFace
results.append(publisher.publish_huggingface("myorg/model"))

# MLflow
results.append(publisher.publish_mlflow("model-prod"))

# Local backup
results.append(publisher.publish_local("./backup/model"))

# Check all succeeded
if all(r.success for r in results):
    print("All publishes succeeded!")
```

## Dependencies

- `huggingface-hub` - HF Hub API (optional)
- `safetensors` - SafeTensors format (optional)
- `mlflow` - MLflow API (optional)
- Internal: `coral.version_control.repository`

Install optional dependencies:
```bash
pip install coral-ml[huggingface]  # For HF Hub
pip install coral-ml[mlflow]       # For MLflow
```

## Testing

Related test files:
- `tests/test_registry.py` - Publishing tests
