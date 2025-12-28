# Integrations Module

This module provides framework integrations for PyTorch, PyTorch Lightning, and Hugging Face, enabling seamless use of Coral version control in ML training workflows.

## Overview

The integrations module provides:
- **PyTorch Integration**: Model-to-weight conversion, CoralTrainer, streaming loader
- **Lightning Integration**: CoralCallback for automatic checkpointing
- **HuggingFace Integration**: Delta-efficient Hub downloads, HF Trainer callback

## Key Files

### `pytorch.py`

Core PyTorch integration with utilities and trainer class.

**PyTorchIntegration** (class):

Static utility methods:
```python
PyTorchIntegration.model_to_weights(model) -> dict[str, WeightTensor]
PyTorchIntegration.weights_to_model(weights, model)
PyTorchIntegration.save_optimizer_state(optimizer) -> dict
PyTorchIntegration.load_optimizer_state(optimizer, state)
PyTorchIntegration.save_scheduler_state(scheduler) -> dict
PyTorchIntegration.load_scheduler_state(scheduler, state)
PyTorchIntegration.get_random_state() -> dict  # For reproducibility
PyTorchIntegration.set_random_state(state)
```

**CoralTrainer** (class):

Full-featured trainer with automatic checkpointing:
```python
trainer = CoralTrainer(
    model=model,
    repository=repo,
    experiment_name="my-experiment",
    checkpoint_config=CheckpointConfig(
        save_every_n_epochs=1,
        save_on_best_metric="loss",
        minimize_metric=True,
        keep_last_n_checkpoints=5,
        keep_best_n_checkpoints=3,
    )
)

trainer.set_optimizer(optimizer)
trainer.set_scheduler(scheduler)

# Training loop
for epoch in range(epochs):
    for batch in dataloader:
        loss = train_step(batch)
        trainer.step(loss, accuracy=acc)  # Records metrics, may checkpoint
    trainer.epoch_end(epoch)

# Load best checkpoint
trainer.load_checkpoint(load_best=True)
```

**Convenience Functions**:
```python
# Save model
save_model_to_coral(model, repo, "Commit message") -> commit_hash

# Load model
load_model_from_coral(model, repo, commit_ref=None) -> bool

# Flexible loading with key mapping
load_into_model(model, weights, strict=True, key_map=None, device='cuda')

# Load from repo path
load_from_repo(model, "./repo", branch="main", device='cuda')

# Load from remote
load_from_remote(model, "./repo", remote_name="origin", pull_first=True)

# Save with options
save_model(model, "./repo", "Message", branch="exp", push_to="origin")

# Compare weights
compare_model_weights(model, weights) -> dict  # identical, modified, similarity

# Create model from weights
create_model_from_weights(ModelClass, weights, device='cuda', **model_kwargs)
```

**StreamingWeightLoader** (class):

Memory-efficient loading for large models:
```python
loader = StreamingWeightLoader("./large-model-repo", commit_ref=None)

# Get info without loading
print(f"Weights: {loader.num_weights}")
print(f"Memory needed: {loader.estimate_memory()} bytes")

# Stream weights one at a time
for name, tensor in loader.stream_weights(device='cuda', progress=True):
    # Process each weight individually
    pass

# Stream directly into model
stats = loader.stream_into_model(model, device='cuda', strict=False)

# Convenience function
stats = stream_load_model(model, "./repo", device='cuda', progress=True)
```

### `lightning.py`

PyTorch Lightning callback for automatic checkpointing.

**CoralCallback** (class):
```python
from coral.integrations.lightning import CoralCallback

callback = CoralCallback(
    repo_path="./weights",
    init=True,                    # Create repo if needed
    save_every_n_epochs=1,        # Save every N epochs
    save_every_n_steps=0,         # Save every N steps (0 to disable)
    save_on_best="val_loss",      # Metric to monitor
    mode="min",                   # 'min' or 'max'
    branch="experiment",          # Branch to commit to
    push_to="origin",             # Remote to push to
    include_optimizer=False,      # Save optimizer state
    metadata_keys=["current_epoch", "global_step"],
)

trainer = pl.Trainer(callbacks=[callback])
trainer.fit(model)

# Load weights back
callback.load_from_coral(pl_module, commit_ref=None, strict=True)
```

**Callbacks triggered**:
- `on_train_epoch_end`: Saves at epoch intervals and best metrics
- `on_train_batch_end`: Saves at step intervals
- `on_train_end`: Final checkpoint

### `huggingface.py`

Delta-efficient Hugging Face Hub integration.

**CoralHubClient** (class):
```python
from coral.integrations.huggingface import CoralHubClient

client = CoralHubClient(
    cache_dir="~/.coral/hub",
    token="hf_xxx",               # Optional HF token
    delta_config=DeltaConfig(...),
    similarity_threshold=0.95,
)

# Get model info
info = client.get_model_info("meta-llama/Llama-2-7b-hf")
print(f"Files: {info.files}")
print(f"Base model: {info.base_model}")

# Download with delta optimization
weights = client.download_model(
    "username/my-finetuned-llama",
    base_model="meta-llama/Llama-2-7b-hf",  # Auto-detected if possible
)

# Check savings
stats = client.last_download_stats
print(f"Could save {stats.savings_percent:.1f}% with deltas")

# Upload model
url = client.upload_model(
    weights,
    repo_id="username/my-model",
    base_model="meta-llama/Llama-2-7b-hf",
    private=False,
)

# Compare two models
comparison = client.compare_models("model-a", "model-b")
print(f"Identical: {comparison['identical_weights']}")
print(f"Similar: {comparison['similar_weights']}")
```

**Convenience function**:
```python
from coral.integrations.huggingface import load_pretrained_efficient

weights = load_pretrained_efficient(
    "username/finetuned-bert",
    base_model="bert-base-uncased",
)
```

### `hf_trainer.py`

Hugging Face Transformers Trainer callback.

**CoralTrainerCallback** (class):
```python
from coral.integrations.hf_trainer import CoralTrainerCallback
from transformers import Trainer

callback = CoralTrainerCallback(
    repo_path="./weights",
    init=True,
    save_every_n_epochs=1,
    save_every_n_steps=500,
    save_on_best="eval_loss",
    mode="min",
    branch="training",
    push_to="origin",
    save_on_train_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[callback],
)
trainer.train()

# Load weights back
callback.load_from_coral(model, commit_ref=None, strict=True)
```

**Callbacks triggered**:
- `on_step_end`: Saves at step intervals
- `on_epoch_end`: Saves at epoch intervals
- `on_evaluate`: Saves best model based on eval metrics
- `on_train_end`: Final checkpoint

## Usage Examples

### Basic PyTorch Training

```python
from coral import Repository
from coral.integrations.pytorch import CoralTrainer
from coral.training import CheckpointConfig

# Initialize
repo = Repository("./checkpoints", init=True)
config = CheckpointConfig(save_every_n_epochs=1, save_on_best_metric="val_loss")
trainer = CoralTrainer(model, repo, "my-experiment", config)

# Training loop
for epoch in range(10):
    for batch in train_loader:
        loss = train_step(batch)
        trainer.step(loss)
    trainer.epoch_end(epoch)

# Resume from best
trainer.load_checkpoint(load_best=True)
```

### PyTorch Lightning

```python
from coral.integrations.lightning import CoralCallback
import pytorch_lightning as pl

callback = CoralCallback(
    repo_path="./weights",
    save_every_n_epochs=1,
    save_on_best="val_loss",
    push_to="origin",
)

trainer = pl.Trainer(callbacks=[callback])
trainer.fit(model, train_dataloader, val_dataloader)
```

### HuggingFace Trainer

```python
from coral.integrations.hf_trainer import CoralTrainerCallback
from transformers import Trainer, TrainingArguments

callback = CoralTrainerCallback(
    repo_path="./weights",
    save_every_n_steps=500,
    save_on_best="eval_loss",
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(...),
    callbacks=[callback],
)
trainer.train()
```

### Large Model Loading

```python
from coral.integrations.pytorch import stream_load_model

# Memory-efficient loading for LLMs
stats = stream_load_model(
    large_model,
    "./llm-weights",
    device='cuda',
    progress=True,
)
print(f"Loaded {stats['matched']} weights")
```

## Dependencies

- `torch` - PyTorch (optional, for pytorch.py)
- `pytorch-lightning` or `lightning` - Lightning (optional, for lightning.py)
- `transformers` - HuggingFace Transformers (optional, for hf_trainer.py)
- `huggingface-hub` - HF Hub API (optional, for huggingface.py)
- `safetensors` - Efficient tensor format (optional, for huggingface.py)

Install with:
```bash
pip install coral-ml[torch]      # PyTorch
pip install coral-ml[lightning]  # Lightning
pip install coral-ml[huggingface] # HuggingFace
```

## Testing

Related test files:
- `tests/test_pytorch_integration.py` - PyTorch integration tests
- `tests/test_training.py` - Training and checkpoint tests
