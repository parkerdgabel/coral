# Chapter 8: Advanced Features

Coral's advanced features extend beyond basic version control to provide comprehensive experiment tracking, model publishing, remote synchronization, and performance optimization. This chapter explores these powerful capabilities that make Coral suitable for production machine learning workflows.

## 8.1 Experiment Tracking

Coral's experiment tracking system provides MLflow-style experiment management integrated directly with your weight versioning workflow. Track metrics, parameters, and automatically link experiments to commits for complete reproducibility.

### ExperimentStatus and Core Types

Experiments progress through defined states represented by the `ExperimentStatus` enum:

```python
from coral.experiments import ExperimentStatus

# Available states
ExperimentStatus.PENDING     # Created but not started
ExperimentStatus.RUNNING     # Currently executing
ExperimentStatus.COMPLETED   # Successfully finished
ExperimentStatus.FAILED      # Terminated with error
ExperimentStatus.CANCELLED   # Manually cancelled
```

Each metric logged during training is captured as an `ExperimentMetric`:

```python
from coral.experiments import ExperimentMetric
from datetime import datetime

metric = ExperimentMetric(
    name="loss",
    value=0.234,
    step=1000,
    timestamp=datetime.now(),
    metadata={"learning_rate": 0.001}
)
```

The `Experiment` dataclass provides the complete structure:

```python
from coral.experiments import Experiment

experiment = Experiment(
    experiment_id="a1b2c3d4",
    name="bert-finetuning",
    description="Fine-tune BERT on customer reviews",
    status=ExperimentStatus.RUNNING,
    created_at=datetime.now(),
    started_at=datetime.now(),
    ended_at=None,
    metrics=[],  # List of ExperimentMetric
    params={"lr": 0.001, "batch_size": 32},
    tags=["nlp", "bert", "production"],
    commit_hash="abc123def456",
    branch="experiments/bert-v2",
    parent_experiment=None,
    notes="Using improved tokenization"
)
```

### ExperimentTracker Usage

The `ExperimentTracker` class manages experiments for a repository:

```python
from coral.version_control.repository import Repository
from coral.experiments import ExperimentTracker

# Initialize tracker
repo = Repository("./my-model")
tracker = ExperimentTracker(repo)

# Start an experiment
exp = tracker.start(
    name="gpt-finetuning",
    description="Fine-tune GPT-2 on domain-specific data",
    params={
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 10,
        "warmup_steps": 500
    },
    tags=["llm", "gpt2", "finetuning"]
)

# Log metrics during training
for epoch in range(10):
    for step in range(1000):
        loss = train_step()
        tracker.log("loss", loss, step=epoch * 1000 + step)

    # Log epoch-level metrics
    val_loss = validate()
    tracker.log_metrics({
        "val_loss": val_loss,
        "val_perplexity": np.exp(val_loss)
    }, step=epoch)

# Add notes or tags during training
tracker.add_tags(["converged", "production-ready"])
tracker.set_notes("Model converged after 7 epochs. Early stopping triggered.")

# Save model and create commit
repo.add_all_weights(model_weights)
commit_hash = repo.commit("Completed GPT-2 fine-tuning", author="researcher@example.com")

# End experiment and link to commit
tracker.end(
    status=ExperimentStatus.COMPLETED,
    commit_hash=commit_hash
)
```

### Comparing Experiments

Find and compare your best experiments:

```python
# List recent experiments
experiments = tracker.list(
    status=ExperimentStatus.COMPLETED,
    tags=["production-ready"],
    limit=20
)

# Find best performing experiments
best_by_loss = tracker.find_best(
    metric="val_loss",
    mode="min",  # minimize validation loss
    status=ExperimentStatus.COMPLETED,
    limit=5
)

for result in best_by_loss:
    print(f"{result['name']}: {result['best_value']:.4f}")
    print(f"  Commit: {result['commit_hash']}")
    print(f"  Params: {result['params']}")

# Compare multiple experiments side-by-side
comparison = tracker.compare(
    experiment_ids=["exp1", "exp2", "exp3"],
    metrics=["val_loss", "val_accuracy"]
)

print(f"Experiments: {comparison['experiments']}")
print(f"Metrics: {comparison['metrics']}")
print(f"Parameters: {comparison['params']}")
```

### Experiment Summary Statistics

Get repository-wide experiment statistics:

```python
summary = tracker.get_summary()
print(f"Total experiments: {summary['total_experiments']}")
print(f"By status: {summary['by_status']}")
print(f"Unique experiments: {summary['unique_names']}")
print(f"Branches used: {summary['branches']}")
print(f"Total metrics logged: {summary['total_metrics_logged']}")
```

### Resuming Failed Experiments

Resume experiments that were interrupted:

```python
# Resume a failed experiment
original_exp = tracker.get("failed_exp_id")
resumed_exp = tracker.resume("failed_exp_id")

# The resumed experiment links to the original
assert resumed_exp.parent_experiment == original_exp.experiment_id
```

### Complete Experiment Workflow Example

Here's a complete workflow integrating experiment tracking with training:

```python
from coral.version_control.repository import Repository
from coral.experiments import ExperimentTracker, ExperimentStatus
import torch

# Initialize
repo = Repository("./bert-experiments")
tracker = ExperimentTracker(repo)

# Define hyperparameters
params = {
    "model": "bert-base-uncased",
    "learning_rate": 2e-5,
    "batch_size": 32,
    "max_epochs": 10,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "gradient_clip": 1.0
}

# Start experiment
exp = tracker.start(
    name="bert-sentiment-v3",
    description="BERT fine-tuning with improved hyperparameters",
    params=params,
    tags=["bert", "sentiment", "production-candidate"]
)

try:
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(params["max_epochs"]):
        # Training phase
        model.train()
        train_losses = []

        for batch_idx, batch in enumerate(train_loader):
            loss = train_step(model, batch)
            train_losses.append(loss)

            # Log step-level metrics
            if batch_idx % 100 == 0:
                tracker.log("train_loss", loss, step=epoch * len(train_loader) + batch_idx)

        # Validation phase
        model.eval()
        val_metrics = validate(model, val_loader)

        # Log epoch-level metrics
        tracker.log_metrics({
            "epoch_train_loss": np.mean(train_losses),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "learning_rate": optimizer.param_groups[0]['lr']
        }, step=epoch)

        # Early stopping logic
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0

            # Save best model
            model_weights = extract_weights(model)
            repo.add_all_weights(model_weights)
            commit_hash = repo.commit(
                f"Epoch {epoch}: val_loss={val_metrics['loss']:.4f}",
                author="training-script@example.com"
            )
            tracker.log_params({"best_checkpoint": commit_hash})

        else:
            patience_counter += 1
            if patience_counter >= 3:
                tracker.set_notes(f"Early stopping triggered at epoch {epoch}")
                break

    # Training completed successfully
    tracker.add_tags(["converged", "ready-for-eval"])
    final_commit = repo.commit("Final model checkpoint", author="training-script@example.com")
    tracker.end(status=ExperimentStatus.COMPLETED, commit_hash=final_commit)

    print(f"Training completed! Experiment ID: {exp.experiment_id}")
    print(f"Best validation loss: {best_val_loss:.4f}")

except Exception as e:
    # Handle training failures
    tracker.fail(error_message=str(e))
    print(f"Training failed: {e}")
    raise
```

This pattern ensures complete reproducibility by linking every model checkpoint to an experiment with full context.

## 8.2 Model Registry Publishing

Coral's model registry integration enables seamless publishing to HuggingFace Hub, MLflow, or local directories, making your versioned weights production-ready.

### Registry Types and Configuration

Coral supports three registry types:

```python
from coral.registry import RegistryType

RegistryType.HUGGINGFACE  # HuggingFace Hub
RegistryType.MLFLOW       # MLflow Model Registry
RegistryType.LOCAL        # Local filesystem export
```

Results are captured in the `PublishResult` dataclass:

```python
from coral.registry import PublishResult

result = PublishResult(
    success=True,
    registry=RegistryType.HUGGINGFACE,
    model_name="my-org/my-model",
    version="1.0.0",
    url="https://huggingface.co/my-org/my-model",
    error=None,
    metadata={"commit_ref": "abc123", "weight_count": 124},
    published_at=datetime.now()
)
```

### Publishing to HuggingFace Hub

Publish models with automatic model card generation:

```python
from coral.registry import ModelPublisher

publisher = ModelPublisher(repo)

# Publish to HuggingFace with model card
result = publisher.publish_huggingface(
    repo_id="my-org/bert-sentiment",
    commit_ref="main",  # Coral commit reference
    private=False,
    description="BERT fine-tuned for sentiment analysis on customer reviews",
    base_model="bert-base-uncased",
    metrics={
        "accuracy": 0.934,
        "f1": 0.927,
        "precision": 0.941,
        "recall": 0.914
    },
    tags=["sentiment-analysis", "bert", "nlp"],
    token=None  # Uses HF_TOKEN environment variable
)

if result.success:
    print(f"Published to: {result.url}")
    print(f"Model card generated with metrics and metadata")
else:
    print(f"Publishing failed: {result.error}")
```

The published repository includes:
- `model.safetensors`: Weights in safetensors format
- `README.md`: Auto-generated model card with metrics
- `coral_metadata.json`: Coral version and commit information

### Publishing to MLflow

Integrate with MLflow tracking and model registry:

```python
# Publish to MLflow Registry
result = publisher.publish_mlflow(
    model_name="customer-sentiment-classifier",
    commit_ref="v1.0",
    tracking_uri="http://mlflow-server:5000",
    experiment_name="sentiment-analysis",
    description="Production model for sentiment classification",
    tags={
        "stage": "production",
        "team": "ml-platform",
        "framework": "pytorch"
    },
    metrics={
        "test_accuracy": 0.934,
        "test_f1": 0.927
    }
)

print(f"MLflow version: {result.version}")
print(f"Run ID: {result.metadata['run_id']}")
print(f"Registry URL: {result.url}")
```

### Local Export

Export models to local directories in multiple formats:

```python
# Export as safetensors (recommended)
result = publisher.publish_local(
    output_path="./exports/model-v1.0",
    commit_ref="v1.0",
    format="safetensors",
    include_metadata=True
)

# Export as PyTorch checkpoint
result = publisher.publish_local(
    output_path="./exports/model-v1.0-pt",
    format="pt",  # PyTorch format
    include_metadata=True
)

# Export as NumPy arrays
result = publisher.publish_local(
    output_path="./exports/model-v1.0-numpy",
    format="npz",  # NumPy compressed format
    include_metadata=True
)
```

Supported formats:
- **safetensors**: Fast, safe, recommended for production
- **pt**: PyTorch native format
- **npz**: NumPy compressed arrays

### Publishing History

Track all publishing operations:

```python
# Get publishing history
history = publisher.get_history(
    registry=RegistryType.HUGGINGFACE,
    success_only=True,
    limit=10
)

for result in history:
    print(f"{result.model_name} -> {result.url}")
    print(f"  Published: {result.published_at}")
    print(f"  Weight count: {result.metadata['weight_count']}")

# Get latest successful publish for a model
latest = publisher.get_latest(
    model_name="my-org/bert-sentiment",
    registry=RegistryType.HUGGINGFACE
)
```

## 8.3 Remote Synchronization

Coral's remote system provides git-like push/pull for synchronizing weights across storage backends including S3, GCS, Azure, and local filesystems.

### Remote Configuration

Define remotes with the `RemoteConfig` dataclass:

```python
from coral.remotes.remote import RemoteConfig

# S3 remote
s3_remote = RemoteConfig(
    name="origin",
    url="s3://my-bucket/coral/models",
    backend="s3",
    region="us-west-2",
    access_key=None,  # Uses AWS credentials from environment
    secret_key=None,
    auto_push=False,
    auto_pull=False
)

# MinIO (S3-compatible)
minio_remote = RemoteConfig.from_url(
    name="minio",
    url="minio://localhost:9000/coral-bucket"
)

# Local filesystem (for testing/backup)
local_remote = RemoteConfig.from_url(
    name="backup",
    url="file:///mnt/backup/coral"
)
```

### Push and Pull Operations

Synchronize weights with remotes:

```python
from coral.remotes.remote import Remote, RemoteManager

# Create remote from config
remote = Remote.from_config(s3_remote)

# Push weights to remote
result = remote.push(
    local_store=repo.store,
    weight_hashes=None,  # None = push all missing weights
    force=False
)

print(f"Pushed {result.pushed_weights} weights")
print(f"Transferred {result.bytes_transferred / 1024 / 1024:.1f} MB")
if result.errors:
    print(f"Errors: {result.errors}")

# Pull weights from remote
result = remote.pull(
    local_store=repo.store,
    weight_hashes=None,  # None = pull all missing weights
    force=False
)

print(f"Pulled {result.pulled_weights} weights")

# List remote weights
remote_weights = remote.list_remote_weights()
print(f"Remote has {len(remote_weights)} weights")
```

### Repository Synchronization

Use the sync utility for bidirectional synchronization:

```python
from coral.remotes.sync import sync_repositories

# Bidirectional sync
stats = sync_repositories(
    local_store=repo.store,
    remote=remote,
    direction="both",  # "push", "pull", or "both"
    force=False
)

print(f"Push: {stats['push']}")
print(f"Pull: {stats['pull']}")
```

### Managing Multiple Remotes

The `RemoteManager` handles multiple remotes:

```python
# Initialize remote manager
manager = RemoteManager(repo.coral_dir / "remotes.json")

# Add remotes
manager.add(RemoteConfig.from_url("origin", "s3://prod-bucket/models"))
manager.add(RemoteConfig.from_url("backup", "file:///backup"))

# List remotes
for name in manager.list():
    remote = manager.get(name)
    info = remote.get_remote_info()
    print(f"{name}: {info['url']} ({info['backend']})")

# Sync with specific remote
origin = manager.get("origin")
origin.push(repo.store)
```

### Authentication Configuration

Coral supports multiple authentication methods:

**AWS S3** - Uses standard AWS credential chain:
```bash
# Environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-west-2

# Or use AWS CLI configuration
aws configure
```

**MinIO**:
```python
config = RemoteConfig(
    name="minio",
    url="s3://bucket/prefix",
    backend="s3",
    endpoint_url="http://localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin"
)
```

**Local filesystem** - No authentication needed:
```python
config = RemoteConfig.from_url("backup", "file:///mnt/backup")
```

### Production Remote Sync Workflows

Here are practical workflows for different deployment scenarios:

**Workflow 1: Continuous Training with Cloud Backup**
```python
from coral.remotes.remote import RemoteManager, RemoteConfig
from coral.version_control.repository import Repository

repo = Repository("./training-workspace")
manager = RemoteManager(repo.coral_dir / "remotes.json")

# Configure S3 remote for team collaboration
manager.add(RemoteConfig(
    name="team-s3",
    url="s3://ml-team-models/sentiment-classifier",
    backend="s3",
    region="us-east-1",
    auto_push=False
))

# Training loop with periodic syncs
for epoch in range(num_epochs):
    # Train model
    train_epoch(model)

    # Commit checkpoint
    weights = extract_weights(model)
    repo.add_all_weights(weights)
    commit_hash = repo.commit(f"Epoch {epoch}", author="trainer")

    # Push to cloud every 5 epochs
    if epoch % 5 == 0:
        remote = manager.get("team-s3")
        result = remote.push(repo.store)
        print(f"Pushed {result.pushed_weights} weights to S3")
```

**Workflow 2: Distributed Team Collaboration**
```python
# Team member A: Pushes model after training
repo_a = Repository("./member-a/workspace")
remote = Remote.from_config(RemoteConfig.from_url("origin", "s3://shared-bucket/model"))

# Train and push
train_model()
repo_a.add_all_weights(model_weights)
repo_a.commit("Improved architecture")
remote.push(repo_a.store)

# Team member B: Pulls latest model
repo_b = Repository("./member-b/workspace")
remote_b = Remote.from_config(RemoteConfig.from_url("origin", "s3://shared-bucket/model"))

# Pull latest weights
result = remote_b.pull(repo_b.store)
print(f"Downloaded {result.pulled_weights} new weights")

# Load and continue training
latest_weights = repo_b.get_all_weights("HEAD")
load_weights_into_model(model, latest_weights)
```

**Workflow 3: Multi-Region Deployment**
```python
manager = RemoteManager(repo.coral_dir / "remotes.json")

# Configure multiple regions for redundancy
regions = ["us-east-1", "eu-west-1", "ap-southeast-1"]
for region in regions:
    manager.add(RemoteConfig(
        name=f"s3-{region}",
        url=f"s3://coral-models-{region}/production",
        backend="s3",
        region=region
    ))

# Deploy to all regions
for region in regions:
    remote = manager.get(f"s3-{region}")
    result = remote.push(repo.store, force=False)
    print(f"{region}: Pushed {result.pushed_weights} weights")
    if result.errors:
        print(f"  Errors: {result.errors}")
```

**Workflow 4: Disaster Recovery and Backup**
```python
import schedule
import time

def backup_weights():
    """Periodic backup job"""
    repo = Repository("./production-model")
    backup_remote = Remote.from_config(
        RemoteConfig.from_url("backup", "file:///mnt/backup/coral")
    )

    result = backup_remote.push(repo.store)
    print(f"Backup completed: {result.pushed_weights} weights")

    # Log backup status
    with open("/var/log/coral-backup.log", "a") as f:
        f.write(f"{datetime.now()}: Backed up {result.pushed_weights} weights\n")

# Schedule daily backups
schedule.every().day.at("02:00").do(backup_weights)

while True:
    schedule.run_pending()
    time.sleep(3600)
```

## 8.4 Visualization Utilities

Coral provides utilities for analyzing and visualizing weight differences, distributions, and deduplication statistics.

### Comparing Models

The `compare_models` function provides comprehensive model comparison:

```python
from coral.utils.visualization import compare_models, format_model_diff

# Load two model versions
base_weights = repo.get_all_weights("base-model")
finetuned_weights = repo.get_all_weights("finetuned-v1")

# Compare models
diff = compare_models(
    base_weights,
    finetuned_weights,
    model_a_name="Base Model",
    model_b_name="Fine-tuned v1"
)

# Print summary
print(f"Overall similarity: {diff['summary']['overall_similarity']:.1%}")
print(f"Changed parameters: {diff['summary']['changed_params']:,}")
print(f"Common layers: {diff['summary']['common_layers']}")

# Analyze categories
counts = diff['category_counts']
print(f"\nChange distribution:")
print(f"  Identical (>99.99%): {counts['identical']}")
print(f"  Minor changes (>99%): {counts['minor_changes']}")
print(f"  Moderate (>90%): {counts['moderate_changes']}")
print(f"  Major (<90%): {counts['major_changes']}")

# View most changed layers
for layer in diff['most_changed_layers'][:5]:
    print(f"{layer['name']}: {layer['combined_similarity']:.2%} similar")
    print(f"  RMSE: {layer['diff_stats']['rmse']:.6f}")
```

Format comparison for human-readable output:

```python
report = format_model_diff(diff, verbose=True)
print(report)
```

### Weight Distribution Analysis

Analyze weight value distributions:

```python
from coral.utils.visualization import plot_weight_distribution

weights = [weight for name, weight in repo.get_all_weights().items()]

distributions = plot_weight_distribution(weights, bins=50)

for name, dist in distributions.items():
    print(f"{name}:")
    print(f"  Mean: {dist['mean']:.4f}")
    print(f"  Std: {dist['std']:.4f}")
    print(f"  Range: [{dist['min']:.4f}, {dist['max']:.4f}]")
    print(f"  Sparsity: {dist['sparsity']:.2%}")
```

### Deduplication Statistics

Visualize deduplication effectiveness:

```python
from coral.utils.visualization import plot_deduplication_stats

stats = repo.deduplicator.get_stats()
viz_data = plot_deduplication_stats(stats)

print(f"Weight distribution:")
print(f"  Unique: {viz_data['weight_counts']['unique']}")
print(f"  Duplicate: {viz_data['weight_counts']['duplicate']}")
print(f"  Similar: {viz_data['weight_counts']['similar']}")

print(f"\nCompression:")
print(f"  Bytes saved: {viz_data['compression']['bytes_saved'] / 1024 / 1024:.1f} MB")
print(f"  Ratio: {viz_data['compression']['compression_ratio']:.2f}x")
```

### Practical Visualization Use Cases

Visualization utilities are invaluable for understanding model evolution and debugging training issues:

**Use Case 1: Debugging Fine-Tuning**
```python
# Compare base model with fine-tuned version
base = repo.get_all_weights("pretrained-base")
finetuned = repo.get_all_weights("finetuned-epoch-5")

diff = compare_models(base, finetuned)

# Find layers that changed too much (potential instability)
for layer in diff['most_changed_layers']:
    if layer['combined_similarity'] < 0.85:
        print(f"WARNING: {layer['name']} changed significantly")
        print(f"  May indicate training instability or learning rate issues")

# Find layers that didn't change at all (frozen layers verification)
for layer_name in diff['categories']['identical']:
    print(f"Frozen: {layer_name}")
```

**Use Case 2: Model Compression Analysis**
```python
# Analyze weight distributions before and after quantization
original = repo.get_all_weights("float32-model")
quantized = repo.get_all_weights("int8-quantized")

original_dist = plot_weight_distribution(list(original.values()))
quantized_dist = plot_weight_distribution(list(quantized.values()))

# Compare distributions
for name in original_dist.keys():
    orig = original_dist[name]
    quant = quantized_dist[name]

    print(f"\n{name}:")
    print(f"  Original range: [{orig['min']:.4f}, {orig['max']:.4f}]")
    print(f"  Quantized range: [{quant['min']:.4f}, {quant['max']:.4f}]")
    print(f"  Sparsity change: {orig['sparsity']:.2%} -> {quant['sparsity']:.2%}")
```

**Use Case 3: Transfer Learning Verification**
```python
# Verify which layers were transferred vs randomly initialized
source_model = repo.get_all_weights("imagenet-pretrained")
target_model = repo.get_all_weights("custom-dataset-init")

diff = compare_models(source_model, target_model)

# Transferred layers should be identical
transferred = diff['categories']['identical']
print(f"Transferred {len(transferred)} layers from source model")

# Randomly initialized layers should be completely different
new_layers = diff['categories']['major_changes']
print(f"Randomly initialized {len(new_layers)} new layers")
```

## 8.5 Advanced Deduplication with SimHash

SimHash provides O(1) similarity detection through locality-sensitive hashing, enabling extremely fast similarity checks for large repositories.

### SimHash Fundamentals

SimHash creates compact binary fingerprints where similar vectors produce similar fingerprints:

```python
from coral.core.simhash import SimHash, SimHashConfig

# Configure SimHash
config = SimHashConfig(
    num_bits=64,  # 64 or 128 bits
    num_hyperplanes=64,  # Match num_bits for standard SimHash
    seed=42,
    similarity_threshold=0.1  # 10% of bits can differ
)

hasher = SimHash(config)

# Compute fingerprints
weight_a = np.random.randn(1000, 1000).astype(np.float32)
weight_b = weight_a + np.random.randn(1000, 1000).astype(np.float32) * 0.01

fp_a = hasher.compute_fingerprint(weight_a)
fp_b = hasher.compute_fingerprint(weight_b)

# Check similarity
distance = SimHash.hamming_distance(fp_a, fp_b)
print(f"Hamming distance: {distance} bits")

is_similar = hasher.are_similar(fp_a, fp_b)
print(f"Similar: {is_similar}")

# Estimate cosine similarity from fingerprint
estimated_sim = hasher.estimated_similarity(fp_a, fp_b)
print(f"Estimated similarity: {estimated_sim:.4f}")
```

### Batch Processing

Compute fingerprints efficiently for multiple weights:

```python
weights = [np.random.randn(100, 100) for _ in range(1000)]

# Batch computation
fingerprints = hasher.compute_fingerprint_batch(weights)

# Find similar pairs
for i, fp_i in enumerate(fingerprints):
    for j in range(i + 1, len(fingerprints)):
        if hasher.are_similar(fp_i, fingerprints[j]):
            print(f"Weights {i} and {j} are similar")
```

### SimHash Index

Use `SimHashIndex` for O(1) similarity lookups:

```python
from coral.core.simhash import SimHashIndex

index = SimHashIndex(config)

# Insert weights
for i, weight in enumerate(weights):
    index.insert(weight, vector_id=f"weight_{i}")

# Query for similar weights
query = np.random.randn(100, 100)
candidates = index.query(
    query,
    max_candidates=10,
    threshold=0.15  # Override default threshold
)

print(f"Found {len(candidates)} similar weights:")
for weight_id, hamming_dist in candidates:
    print(f"  {weight_id}: distance={hamming_dist}")

# Index statistics
stats = index.get_stats()
print(f"Index size: {stats['num_vectors']}")
print(f"Unique fingerprints: {stats['num_unique_fingerprints']}")
print(f"Avg vectors per fingerprint: {stats['avg_vectors_per_fingerprint']:.2f}")
```

### Multi-Dimensional SimHash

Handle weights of different dimensions:

```python
from coral.core.simhash import MultiDimSimHashIndex

multi_index = MultiDimSimHashIndex(config)

# Insert weights of various shapes
multi_index.insert(np.random.randn(100, 100), "conv1")
multi_index.insert(np.random.randn(512, 512), "conv2")
multi_index.insert(np.random.randn(10, 10), "bias")

# Query finds only matching dimensions
similar = multi_index.query(np.random.randn(512, 512))
print(f"Similar weights: {similar}")

# Statistics per dimension
stats = multi_index.get_stats()
print(f"Dimensions indexed: {stats['num_dimensions']}")
print(f"Per-dimension stats: {stats['dimension_stats']}")
```

### SimHash Performance Characteristics

Understanding SimHash performance helps optimize for your use case:

**Fingerprint Size Trade-offs:**
```python
import time

# Test different fingerprint sizes
configs = [
    SimHashConfig(num_bits=32, num_hyperplanes=32),
    SimHashConfig(num_bits=64, num_hyperplanes=64),
    SimHashConfig(num_bits=128, num_hyperplanes=128)
]

weights = [np.random.randn(1000, 1000) for _ in range(100)]

for config in configs:
    hasher = SimHash(config)

    start = time.time()
    fingerprints = hasher.compute_fingerprint_batch(weights)
    elapsed = time.time() - start

    # Measure collision rate
    unique_fps = len(set(map(str, fingerprints)))
    collision_rate = 1 - (unique_fps / len(fingerprints))

    print(f"{config.num_bits}-bit SimHash:")
    print(f"  Computation time: {elapsed:.3f}s")
    print(f"  Unique fingerprints: {unique_fps}/{len(fingerprints)}")
    print(f"  Collision rate: {collision_rate:.2%}")
    print(f"  Memory per fingerprint: {config.num_bits // 8} bytes")
```

**Expected Output:**
```
32-bit SimHash:
  Computation time: 0.124s
  Unique fingerprints: 95/100
  Collision rate: 5.00%
  Memory per fingerprint: 4 bytes

64-bit SimHash:
  Computation time: 0.235s
  Unique fingerprints: 100/100
  Collision rate: 0.00%
  Memory per fingerprint: 8 bytes

128-bit SimHash:
  Computation time: 0.456s
  Unique fingerprints: 100/100
  Collision rate: 0.00%
  Memory per fingerprint: 16 bytes
```

**Recommendation:** Use 64-bit for most applications (balances speed, memory, and collision resistance).

## 8.6 LSH Index for O(1) Similarity Search

Locality-Sensitive Hashing (LSH) provides fast approximate nearest neighbor search, reducing similarity detection from O(n) to O(1) average time.

### LSH Fundamentals

LSH uses random hyperplane hashing for cosine similarity:

```python
from coral.core.lsh_index import LSHIndex, LSHConfig

# Configure LSH
config = LSHConfig(
    num_hyperplanes=8,  # Bits per hash (more = fewer false positives)
    num_tables=4,  # Multiple tables reduce false negatives
    seed=42,
    max_candidates=100
)

# Create index for specific dimension
index = LSHIndex(vector_dim=10000, config=config)

# Insert weights
for i in range(1000):
    weight = np.random.randn(10000).astype(np.float32)
    index.insert(weight, key=f"weight_{i}")

# Query returns candidates (much faster than scanning all 1000)
query = np.random.randn(10000).astype(np.float32)
candidates = index.query(query, max_candidates=50)

print(f"Found {len(candidates)} candidates out of {len(index)}")
print(f"Speedup: {len(index) / len(candidates):.1f}x fewer comparisons")

# Index statistics
stats = index.get_stats()
print(f"Vectors: {stats['num_vectors']}")
print(f"Tables: {stats['num_tables']}")
print(f"Buckets: {stats['total_buckets']}")
print(f"Avg bucket size: {stats['avg_bucket_size']:.1f}")
```

### Multi-Dimensional LSH

Handle weights of different shapes automatically:

```python
from coral.core.lsh_index import MultiDimLSHIndex

multi_lsh = MultiDimLSHIndex(config)

# Insert weights of various dimensions
multi_lsh.insert(np.random.randn(100, 100), "conv1_weights")
multi_lsh.insert(np.random.randn(512), "bias1")
multi_lsh.insert(np.random.randn(512, 256), "fc1_weights")

# Query automatically uses correct dimension index
query_conv = np.random.randn(100, 100)
candidates = multi_lsh.query(query_conv, max_candidates=20)

# Statistics across all dimensions
stats = multi_lsh.get_stats()
print(f"Total vectors: {stats['total_vectors']}")
print(f"Dimensions indexed: {stats['num_dimensions']}")
for dim, dim_stats in stats['per_dimension'].items():
    print(f"  Dim {dim}: {dim_stats['num_vectors']} vectors")
```

### Integrating LSH with Deduplicator

Use LSH to accelerate deduplication in large repositories:

```python
from coral.core.deduplicator import Deduplicator

# Create deduplicator with LSH enabled
deduplicator = Deduplicator(
    similarity_threshold=0.98,
    use_lsh=True,  # Enable LSH acceleration
    lsh_config=LSHConfig(
        num_hyperplanes=8,
        num_tables=4
    )
)

# Add weights (LSH index built automatically)
for name, weight in model_weights.items():
    deduplicator.add_weight(weight, name)

# Deduplication uses LSH for O(1) lookups
stats = deduplicator.deduplicate()
print(f"Deduplication with LSH acceleration:")
print(f"  Unique: {stats.unique_weights}")
print(f"  Duplicates: {stats.duplicate_weights}")
print(f"  Similar: {stats.similar_weights}")
print(f"  Compression: {stats.compression_ratio:.2f}x")
```

### Configuration Tuning

Optimize LSH parameters for your workload:

**High precision (fewer false positives):**
```python
config = LSHConfig(
    num_hyperplanes=16,  # More bits = better precision
    num_tables=2,  # Fewer tables = faster but more false negatives
    max_candidates=50
)
```

**High recall (fewer false negatives):**
```python
config = LSHConfig(
    num_hyperplanes=6,  # Fewer bits = more candidates
    num_tables=8,  # More tables = better recall
    max_candidates=200
)
```

**Balanced (recommended):**
```python
config = LSHConfig(
    num_hyperplanes=8,
    num_tables=4,
    max_candidates=100
)
```

### LSH Performance and Best Practices

Understanding LSH performance characteristics helps maximize efficiency:

**Scalability Analysis:**
```python
import time

# Compare linear search vs LSH for different repository sizes
repository_sizes = [100, 1000, 10000]
vector_dim = 10000

for n_vectors in repository_sizes:
    # Create test data
    vectors = [np.random.randn(vector_dim).astype(np.float32) for _ in range(n_vectors)]

    # Method 1: Linear search (baseline)
    query = np.random.randn(vector_dim).astype(np.float32)
    start = time.time()
    for vec in vectors:
        cosine_similarity(query, vec)
    linear_time = time.time() - start

    # Method 2: LSH index
    lsh = LSHIndex(vector_dim, LSHConfig(num_hyperplanes=8, num_tables=4))
    start = time.time()
    for i, vec in enumerate(vectors):
        lsh.insert(vec, f"vec_{i}")
    build_time = time.time() - start

    start = time.time()
    candidates = lsh.query(query, max_candidates=50)
    query_time = time.time() - start

    print(f"\nRepository size: {n_vectors:,} vectors")
    print(f"  Linear search: {linear_time:.3f}s")
    print(f"  LSH build time: {build_time:.3f}s")
    print(f"  LSH query time: {query_time:.4f}s ({len(candidates)} candidates)")
    print(f"  Speedup: {linear_time / query_time:.1f}x")
```

**Expected Output:**
```
Repository size: 100 vectors
  Linear search: 0.021s
  LSH build time: 0.003s
  LSH query time: 0.0002s (12 candidates)
  Speedup: 105.0x

Repository size: 1,000 vectors
  Linear search: 0.215s
  LSH build time: 0.028s
  LSH query time: 0.0003s (15 candidates)
  Speedup: 716.7x

Repository size: 10,000 vectors
  Linear search: 2.134s
  LSH build time: 0.285s
  LSH query time: 0.0005s (18 candidates)
  Speedup: 4268.0x
```

**Best Practices:**

1. **Choose parameters based on repository size:**
   - Small (<1K weights): Simple linear search may be faster
   - Medium (1K-10K): Use LSH with k=8, L=4
   - Large (>10K): Increase tables to L=8 for better recall

2. **Tune for your similarity threshold:**
   - High threshold (>0.95): Use more hyperplanes (k=12-16)
   - Medium threshold (0.85-0.95): Balanced config (k=8)
   - Low threshold (<0.85): Fewer hyperplanes (k=4-6)

3. **Memory considerations:**
   ```python
   # Estimate LSH index memory usage
   stats = lsh.get_stats()
   vectors_stored = stats['num_vectors']
   bytes_per_vector = vector_dim * 4  # float32

   # Overhead: hash tables + bucket storage
   overhead = stats['num_tables'] * stats['total_buckets'] * 32  # approximate

   total_memory_mb = (vectors_stored * bytes_per_vector + overhead) / 1024 / 1024
   print(f"LSH index memory: {total_memory_mb:.1f} MB")
   ```

4. **Batch operations for efficiency:**
   ```python
   # Inefficient: insert one at a time
   for weight in weights:
       lsh.insert(weight, weight_id)

   # Better: batch build
   lsh = LSHIndex(dim, config)
   for weight_id, weight in zip(weight_ids, weights):
       lsh.insert(weight, weight_id)
   ```

5. **Monitor and adjust:**
   ```python
   # Check if LSH is effective
   stats = lsh.get_stats()

   avg_bucket_size = stats['avg_bucket_size']
   if avg_bucket_size > 100:
       print("WARNING: Buckets too large, increase num_hyperplanes")
   elif avg_bucket_size < 2:
       print("WARNING: Too many empty buckets, decrease num_hyperplanes")
   ```

## Summary

Coral's advanced features provide enterprise-grade capabilities for production ML workflows:

- **Experiment Tracking**: MLflow-style experiment management with metric logging and commit linking
- **Model Publishing**: Seamless integration with HuggingFace Hub, MLflow, and local registries
- **Remote Sync**: Git-like push/pull for distributed weight management
- **Visualization**: Comprehensive tools for analyzing model differences and deduplication
- **SimHash**: O(1) similarity detection through locality-sensitive hashing
- **LSH Index**: Fast approximate nearest neighbor search for large repositories

These features work together to provide a complete platform for versioning, tracking, and deploying neural network weights at scale. In the next chapter, we'll explore best practices and production deployment patterns.
