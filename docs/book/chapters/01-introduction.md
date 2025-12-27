# Chapter 1: Introduction to the Coral System

## The Problem Space

### The Challenge of Managing Neural Network Weights

In modern machine learning, neural network weights represent the culmination of expensive training processes—often requiring days or weeks of GPU computation and significant energy costs. Yet despite their value, these weights are typically managed using ad-hoc solutions that range from simple file copying to overcomplicated custom infrastructure.

Consider a typical ML development workflow:

- **During training**: You save checkpoints every few epochs, "just in case." After 100 epochs with checkpoints every 5 steps, you have 20 checkpoint files, many nearly identical.
- **During experimentation**: You create variations by fine-tuning hyperparameters, generating dozens of model versions that differ by only a few layers.
- **During deployment**: You need to track which model version is in production, what metrics it achieved, and maintain the ability to rollback.
- **During collaboration**: Team members share models via cloud storage, losing track of lineage, training conditions, and performance metrics.

The result? **Storage explosion, lost experiments, and broken reproducibility.**

### Why Traditional Approaches Fall Short

#### 1. File Copying: Simple but Wasteful

The most common approach is to save each checkpoint as a separate file:

```
checkpoints/
  model_epoch_005.pth    # 528 MB
  model_epoch_010.pth    # 528 MB (99.9% identical to epoch 5)
  model_epoch_015.pth    # 528 MB (99.8% identical to epoch 10)
  ...
  model_epoch_100.pth    # 528 MB
```

For a ResNet-50 model (25M parameters), storing 20 checkpoints requires **10.6 GB** even though consecutive checkpoints are typically 99%+ identical. This approach:

- Wastes massive amounts of storage (typical ML teams report terabytes of redundant weights)
- Provides no mechanism for tracking relationships between models
- Offers no way to compare or diff model versions
- Makes it difficult to identify which checkpoint corresponds to which experiment

#### 2. Git for Binary Files: Poorly Suited

Some teams attempt to use Git for model versioning. While Git excels at text-based source code, it struggles with large binary files:

- **Poor compression**: Git's delta compression is optimized for text, not floating-point arrays
- **Repository bloat**: Each checkpoint inflates the `.git` directory since binary diffs are inefficient
- **Slow operations**: Cloning, pulling, and pushing become prohibitively slow with large weights
- **No semantic understanding**: Git can't compute meaningful diffs of weight tensors or leverage domain-specific optimizations

#### 3. The Scale Problem

Modern models have grown exponentially:

| Model | Parameters | Storage (FP32) | 100 Checkpoints |
|-------|-----------|---------------|----------------|
| ResNet-50 | 25M | 100 MB | 10 GB |
| BERT-base | 110M | 440 MB | 44 GB |
| GPT-3 | 175B | 700 GB | 70 TB |
| GPT-4 (estimated) | 1.7T | 6.8 TB | 680 TB |

Training large transformer models generates hundreds of checkpoints. Without intelligent deduplication, storing these becomes economically infeasible. **A single training run of a large language model can easily generate petabytes of checkpoint data.**

#### 4. The Tracking Problem

Beyond storage, ML practitioners need to:

- Track which checkpoint achieved the best validation accuracy
- Compare weights across different hyperparameter configurations
- Merge improvements from multiple fine-tuning experiments
- Maintain reproducibility (which exact weights were used for a published result?)
- Collaborate with team members without overwriting each other's work

Traditional file systems and version control tools weren't designed for these neural network-specific needs.

## What is Coral?

**Coral is a production-ready neural network weight versioning system that brings git-like version control to ML models.** Think of it as "git for neural networks"—but with deep understanding of weight tensors and their unique properties.

### Core Concept

Just as Git revolutionized source code management with content-addressable storage and efficient delta compression, Coral applies similar principles to neural network weights:

```
Traditional Approach          Coral Approach
================              ==============

checkpoint_1.pth              main branch
checkpoint_2.pth    ────►     ├── commit abc123 (epoch 10)
checkpoint_3.pth              ├── commit def456 (epoch 20) [delta from abc123]
fine_tuned_v1.pth             └── experiment branch
fine_tuned_v2.pth                 └── commit 789ghi (fine-tuned) [delta from def456]
```

Coral recognizes that:
1. Consecutive checkpoints are typically 99%+ similar (only gradients changed)
2. Fine-tuned models share most weights with their base models
3. Multiple experiments often share common layers (e.g., pre-trained backbones)
4. Exact duplicates are common (same initialization across experiments)

### Key Value Propositions

#### 1. Lossless Delta Encoding

**The Innovation**: Unlike traditional compression that loses information, Coral's delta encoding enables **perfect reconstruction** of similar weights while achieving 90-98% compression.

```python
# Before: Information loss
weight_original = [1.234567, 2.345678, 3.456789]
weight_after_compression = [1.23, 2.35, 3.46]  # Precision lost!

# Coral: Perfect reconstruction via delta encoding
repo.stage_weights({"layer1": weight_v1})
repo.commit("Base model")

# Version 2 is 99% similar (only gradient updates)
repo.stage_weights({"layer1": weight_v2})  # Stored as delta
repo.commit("After 1 epoch")

# Later: perfect reconstruction
loaded = repo.get_weight("layer1")  # Exactly equals weight_v2!
assert np.array_equal(loaded, weight_v2)  # ✓ Perfect match
```

The delta encoding system offers multiple strategies optimized for different scenarios:

- **FLOAT32_RAW**: Uncompressed deltas, ~50% space savings, instant reconstruction
- **COMPRESSED**: zlib-compressed deltas, ~70% space savings, fast reconstruction
- **XOR_FLOAT32**: Bitwise XOR encoding, 15-25% better than raw
- **INT16_QUANTIZED**: Slight precision loss, ~90% space savings

#### 2. Proven Space Savings

Benchmarked on real-world ML workflows, Coral achieves:

- **47.6% space savings** compared to naive PyTorch checkpoint storage
- **1.91x compression ratio** on average across diverse model architectures
- **Up to 98% savings** for exact duplicates or nearly-identical checkpoints

These numbers are from actual production use cases, not theoretical maximums.

#### 3. Complete Git-like Workflow

Coral provides familiar version control operations:

```bash
# Initialize repository
coral-ml init my_model_project

# Make changes and commit
coral-ml add trained_model.pth
coral-ml commit -m "Initial model: accuracy=87.3%"

# Branch for experiments
coral-ml branch fine_tune_augmentation
coral-ml checkout fine_tune_augmentation

# After experimenting
coral-ml commit -m "Fine-tuned with augmentation: accuracy=91.2%"

# Merge successful experiments
coral-ml checkout main
coral-ml merge fine_tune_augmentation

# Tag production versions
coral-ml tag v1.0 -d "Production model" --metric accuracy=91.2
```

#### 4. Content-Addressable Storage

Like Git, Coral uses content hashing (xxHash for speed) to identify weights:

- Identical weights are stored only once, regardless of how many times they appear
- Weights are immutable—the hash is the identity
- Deduplication happens automatically across all branches and commits

#### 5. Framework Agnostic

While Coral has deep PyTorch integration, it works with any framework:

```python
# PyTorch
from coral.integrations.pytorch import CoralTrainer
trainer = CoralTrainer(pytorch_model, repo, "experiment_1")

# TensorFlow (coming soon)
from coral.integrations.tensorflow import CoralCallback
model.fit(x, y, callbacks=[CoralCallback(repo)])

# Or framework-agnostic
from coral import WeightTensor
weights = {"layer1": WeightTensor(data=numpy_array, metadata=...)}
repo.stage_weights(weights)
```

## Key Features Overview

### Deduplication: Exact and Similarity-Based

Coral employs a two-tier deduplication strategy:

**Exact Deduplication**: Weights with identical values (same hash) are stored once.
```python
# Common scenario: same initialization across experiments
model_A = initialize_resnet50()  # Hash: abc123
model_B = initialize_resnet50()  # Hash: abc123 (identical!)
# Only stored once, referenced twice
```

**Similarity-Based Deduplication**: Weights that are highly similar (>98% cosine similarity by default) are stored as deltas.
```python
dedup = Deduplicator(
    similarity_threshold=0.98,      # Consider >98% similar
    enable_delta_encoding=True,     # Store as delta
    batch_size=100                  # Process in batches for performance
)

# Fine-tuned model: 99.5% similar to base
fine_tuned_hash, delta_info = dedup.add_weight(fine_tuned_weight, "fc.weight")
print(delta_info)
# {'is_delta': True, 'reference_hash': 'abc123',
#  'compression_ratio': 0.95, 'bytes_saved': 1_234_567}
```

### Delta Encoding: 8+ Encoding Strategies

Coral's delta encoding system offers fine-grained control over the compression vs. quality tradeoff:

| Strategy | Reconstruction | Compression | Use Case |
|----------|---------------|-------------|----------|
| FLOAT32_RAW | Perfect (lossless) | ~50% | Fast access, high fidelity |
| COMPRESSED | Perfect (lossless) | ~70% | Balanced default |
| XOR_FLOAT32 | Perfect (lossless) | ~75% | Better compression |
| XOR_BFLOAT16 | Perfect (lossless) | ~80% | BFloat16 models |
| EXPONENT_MANTISSA | Perfect (lossless) | ~75% | Float component separation |
| INT16_QUANTIZED | Minor loss | ~90% | Acceptable precision loss |
| INT8_QUANTIZED | Some loss | ~95% | Maximum compression |
| SPARSE | Perfect for sparse | >95% | Few changed weights |

Configure per-repository:

```python
from coral.delta import DeltaConfig, DeltaType

config = DeltaConfig(
    delta_type=DeltaType.COMPRESSED,    # Good default
    similarity_threshold=0.99,          # Only store deltas for very similar
    compression_level=6                 # Balance speed vs size
)

repo = Repository("./models", init=True, delta_config=config)
```

### Version Control: Branches, Tags, and Merges

Full git-like version control adapted for neural networks:

**Branching**:
```python
# Create experimental branch
repo.create_branch("experiment_dropout_0.5")
repo.checkout("experiment_dropout_0.5")
repo.stage_weights(experimental_weights)
repo.commit("Added dropout regularization")
```

**Tagging**:
```python
# Tag with metrics
repo.tag_version("v1.0", "Production model", metrics={
    "accuracy": 0.923,
    "f1_score": 0.915,
    "inference_time_ms": 45
})
```

**Merging with conflict resolution**:
```python
# Merge experiment back to main
repo.checkout("main")
merge_commit = repo.merge(
    "experiment_dropout_0.5",
    strategy=MergeStrategy.AVERAGE  # Average conflicting weights
)
```

For neural networks, merging strategies like `AVERAGE` or `WEIGHTED` can intelligently combine weights from different training runs—a capability unique to Coral.

### Training Integration: Seamless Checkpointing

The `CoralTrainer` wrapper makes version control transparent during training:

```python
from coral.integrations.pytorch import CoralTrainer
from coral.training import CheckpointConfig

config = CheckpointConfig(
    save_every_n_epochs=5,              # Regular snapshots
    save_on_best_metric="val_accuracy", # Save when improving
    keep_best_n_checkpoints=3,          # Limit storage
    auto_commit=True,                   # Automatic git-like commits
    tag_best_checkpoints=True           # Tag best models
)

trainer = CoralTrainer(model, repo, "resnet50_imagenet", config)

# Training loop - checkpointing happens automatically!
for epoch in range(100):
    for batch in dataloader:
        loss = train_step(batch)
        trainer.step(loss=loss.item())  # Updates internal state

    trainer.epoch_end(epoch, val_accuracy=validate())
    # Coral automatically:
    # 1. Checks if checkpoint should be saved (per config)
    # 2. Deduplicates weights (saves only deltas)
    # 3. Commits with metadata (epoch, loss, metrics)
    # 4. Tags if best checkpoint
```

### Experiment Tracking and Model Registry

Built-in experiment tracking without external dependencies:

```python
# Each commit automatically records:
commit_metadata = CommitMetadata(
    message="Epoch 50: lr=0.001",
    author="user@example.com",
    metrics={
        "train_loss": 0.234,
        "val_accuracy": 0.923,
        "learning_rate": 0.001
    },
    tags=["experiment", "resnet50"],
    environment={
        "torch_version": "2.0.0",
        "cuda_version": "11.8",
        "gpu": "A100"
    }
)

# Query history
best_commits = repo.find_commits(
    metric_filter={"val_accuracy": (">", 0.92)},
    tag_filter=["production-ready"]
)
```

### CLI Interface: 40+ Commands

Professional command-line interface with git-like ergonomics:

```bash
# Repository management
coral-ml init [path]              # Initialize repository
coral-ml clone <url> [path]       # Clone from remote
coral-ml status                   # Show working tree status
coral-ml config --list            # Show configuration

# Basic workflow
coral-ml add <files>              # Stage weight files
coral-ml commit -m <msg>          # Commit staged weights
coral-ml log [--oneline]          # Show commit history
coral-ml diff <ref1> [ref2]       # Show weight differences

# Branching and merging
coral-ml branch [name]            # List or create branches
coral-ml checkout <branch>        # Switch branches
coral-ml merge <branch>           # Merge branches
coral-ml tag <name> [--metric]    # Create tagged version

# Advanced features
coral-ml gc [--dry-run]           # Garbage collection
coral-ml deduplicate              # Manual deduplication
coral-ml export <ref> <path>      # Export weights
coral-ml import <path>            # Import weights

# Information
coral-ml show <ref>               # Show commit details
coral-ml ls-tree <ref>            # List weights in commit
coral-ml reflog                   # Show reference history
```

## Use Cases

### 1. Training Checkpoints: Save Storage During Long Runs

**Problem**: Training a BERT model for 100 epochs with checkpoints every 5 epochs generates 20 checkpoint files, each ~440 MB, totaling **8.8 GB**.

**Coral Solution**:
```python
trainer = CoralTrainer(bert_model, repo, "bert_pretraining")

# After 100 epochs with 20 checkpoints:
# - Naive storage: 8.8 GB
# - Coral storage: 4.6 GB (47.6% savings)
# - Perfect reconstruction of any checkpoint
```

Consecutive checkpoints differ only by gradient updates, making them ideal for delta encoding. Coral automatically:
- Detects similarity between consecutive checkpoints
- Stores only the delta (changes)
- Enables instant reconstruction when loading

### 2. Model Versioning: Track Experiments and Iterations

**Problem**: You run 15 experiments varying hyperparameters. Each produces a 528 MB model. Total: **7.9 GB**. Which one achieved 91% accuracy?

**Coral Solution**:
```bash
# Experiment 1: baseline
coral-ml commit -m "Baseline: lr=0.001" --metric accuracy=0.873

# Experiment 2: higher learning rate
coral-ml commit -m "Increased lr=0.01" --metric accuracy=0.891

# Experiment 15: best config
coral-ml commit -m "Best: lr=0.005, dropout=0.3" --metric accuracy=0.912
coral-ml tag v1.0 --metric accuracy=0.912

# Find all experiments with >90% accuracy
coral-ml log --filter "accuracy>0.90"

# Storage: 2.1 GB (73% savings via deduplication)
```

Each experiment is a commit with metrics attached. Coral's content-addressable storage means shared weights (like pre-trained backbones) are stored only once.

### 3. Transfer Learning: Efficiently Store Fine-Tuned Models

**Problem**: You fine-tune BERT for 10 different tasks. Each fine-tuned model is 99.8% identical to the base model (only the classification head changes). Naive storage: **4.4 GB**.

**Coral Solution**:
```python
# Store base model
repo.stage_weights(bert_base_weights)
repo.commit("BERT base pretrained")

# Fine-tune for sentiment analysis (only head changed)
repo.stage_weights(bert_sentiment_weights)  # 99.8% similar
repo.commit("Fine-tuned: sentiment")  # Stored as delta!

# Fine-tune for NER (only head changed)
repo.stage_weights(bert_ner_weights)  # 99.8% similar
repo.commit("Fine-tuned: NER")  # Stored as delta!

# 10 fine-tuned models stored as deltas
# Storage: 0.5 GB (88% savings)
```

Delta encoding shines here: the shared BERT backbone is stored once, and only the modified classification heads are stored as deltas.

### 4. Team Collaboration: Shared Model Repositories

**Problem**: Three team members independently train model variants. They share via cloud storage, leading to duplicates, lost lineage, and confusion about which model is best.

**Coral Solution**:
```bash
# Team member 1: baseline
coral-ml commit -m "Baseline ResNet-50" --author alice@team.com
coral-ml push origin main

# Team member 2: data augmentation experiment
coral-ml checkout -b experiment_augmentation
coral-ml commit -m "Added augmentation" --author bob@team.com
coral-ml push origin experiment_augmentation

# Team member 3: architecture modification
coral-ml checkout -b experiment_wider
coral-ml commit -m "Wider network" --author carol@team.com
coral-ml push origin experiment_wider

# Later: review all experiments
coral-ml log --all --graph  # See complete history
coral-ml diff main experiment_augmentation  # Compare approaches

# Merge best approach
coral-ml merge experiment_augmentation
coral-ml tag v2.0 -d "Production: with augmentation"
```

Coral provides the same collaborative benefits as Git for source code: branching, merging, history tracking, and conflict resolution—but for model weights.

### 5. Model Deployment: Track Production Versions

**Problem**: You have multiple model versions in production across different services. When issues arise, you need to quickly identify which exact weights are deployed where.

**Coral Solution**:
```python
# Tag production deployments
repo.tag_version("production_api_v1", "API server v1", metrics={
    "accuracy": 0.923,
    "latency_p99_ms": 45,
    "deployed_date": "2024-01-15",
    "deployed_by": "deploy-bot"
})

repo.tag_version("production_mobile_v1", "Mobile app v1", metrics={
    "accuracy": 0.915,  # Slightly lower accuracy
    "size_mb": 25,      # Optimized for mobile
    "deployed_date": "2024-01-20"
})

# Audit: what's deployed where?
deployments = repo.list_tags(pattern="production_*")
for tag in deployments:
    print(f"{tag.name}: {tag.description}")
    print(f"  Deployed: {tag.metadata['deployed_date']}")
    print(f"  Accuracy: {tag.metadata['accuracy']}")

# Rollback if needed
repo.checkout("production_api_v1")  # Instant rollback to previous version
```

Coral serves as a model registry, providing audit trails for compliance, instant rollback capabilities, and clear deployment tracking.

## Architecture Preview

### High-Level Component Overview

The Coral system is organized into distinct layers, each with clear responsibilities:

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INTERFACES                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  CLI (coral) │  │  Python API  │  │   Training   │      │
│  │   40+ cmds   │  │   Library    │  │ Integration  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                  VERSION CONTROL LAYER                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Repository  │  │   Branches   │  │    Commits   │      │
│  │   Manager    │  │   & Merging  │  │   & Tags     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│              DEDUPLICATION & COMPRESSION LAYER               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Deduplicator │  │    Delta     │  │  Similarity  │      │
│  │   Engine     │  │   Encoder    │  │   Detector   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                     STORAGE LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  HDF5 Store  │  │  Content-    │  │   Garbage    │      │
│  │  (weights)   │  │  Addressable │  │  Collection  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                       DATA LAYER                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              WeightTensor + Metadata                  │   │
│  │  (NumPy arrays with rich metadata and hashing)       │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction

**Data Flow: Storing Weights**

```
1. User commits model weights
   └─> Repository.commit(message)

2. Repository stages weights
   └─> WeightTensor created with metadata
       └─> Content hash computed (xxHash)

3. Deduplication check
   └─> Deduplicator.add_weight(tensor)
       ├─> Exact match found? → Reference existing
       ├─> Similar weight found? → Create delta
       └─> Unique? → Store new weight

4. Delta encoding (if similar)
   └─> DeltaEncoder.encode_delta(current, reference)
       └─> Choose strategy (COMPRESSED, etc.)
       └─> Compute delta
       └─> Store delta object

5. Storage
   └─> HDF5Store.store(weight_or_delta)
       └─> Content-addressable storage
       └─> Optional compression (gzip/lzf)

6. Commit creation
   └─> Create commit object with metadata
       └─> Update branch pointer
       └─> Update version graph
```

**Data Flow: Loading Weights**

```
1. User checks out commit or tag
   └─> Repository.checkout(ref)

2. Resolve reference
   └─> BranchManager.resolve(ref)
       └─> Find commit hash

3. Load commit metadata
   └─> Read weight references from commit

4. Load each weight
   └─> HDF5Store.load(hash)
       ├─> Is it a delta?
       │   └─> Load reference weight
       │   └─> DeltaEncoder.decode_delta(delta, ref)
       │   └─> Reconstruct original
       └─> Regular weight?
           └─> Load directly

5. Return WeightTensors to user
```

### Core Components Deep Dive

**WeightTensor**: The fundamental data structure representing a neural network parameter tensor.
- Stores NumPy array of weights
- Rich metadata (name, shape, dtype, layer type, model name)
- Content hashing for identity
- Lazy loading support

**Deduplicator**: Intelligence engine for finding duplicates and similar weights.
- Maintains hash → weight mapping
- Computes similarity scores (cosine similarity)
- Decides when to create deltas vs store unique weights
- Tracks statistics (compression ratio, bytes saved)

**DeltaEncoder**: Converts similar weights into compact delta representations.
- Multiple encoding strategies (8+ types)
- Lossless reconstruction for most strategies
- Configurable compression levels
- Automatic strategy selection based on characteristics

**HDF5Store**: High-performance storage backend.
- Content-addressable: hash → data mapping
- Compression support (gzip, lzf, szip)
- Separate groups for weights and deltas
- Batch operations for performance
- Lazy loading via HDF5 dataset references

**Repository**: Git-like version control operations.
- Branching, committing, merging, tagging
- Conflict resolution with merge strategies
- History tracking and querying
- Configuration management
- Integration with all other components

**BranchManager**: Manages branches and references.
- HEAD pointer tracking
- Branch creation/deletion
- Reference resolution (tags, branches, commit hashes)
- Detached HEAD support

**CoralTrainer**: PyTorch training integration.
- Wraps model for automatic checkpointing
- Configurable checkpoint policies
- Metric tracking and best model detection
- Training state persistence
- Callback system for custom logic

### Design Principles

**1. Content-Addressable Everything**
Weights are identified by their content hash, not by file paths or arbitrary IDs. This enables:
- Automatic deduplication (same content = same hash)
- Immutable storage (content never changes)
- Efficient comparison and diffing

**2. Lossless Where Possible**
Unlike traditional compression, Coral prioritizes perfect reconstruction:
- Delta encoding reconstructs exact original values
- Only lossy when explicitly configured (INT8_QUANTIZED, etc.)
- Transparency: you know what you're getting

**3. Pluggable Architecture**
Key abstractions allow for extension:
- `WeightStore` interface: swap HDF5 for S3, filesystem, etc.
- `DeltaEncoder` strategies: add new encoding algorithms
- Merge strategies: customize conflict resolution for your use case

**4. Zero-Copy Operations**
Where possible, Coral avoids copying large weight arrays:
- HDF5 memory mapping for direct access
- NumPy views instead of copies
- Lazy loading of deltas (only decode when accessed)

**5. Familiar Git Semantics**
Coral intentionally mirrors Git's mental model:
- Commits are immutable snapshots
- Branches are lightweight pointers
- Merging combines histories
- Tags mark important versions

This makes Coral immediately familiar to developers who already know Git.

## What's Next?

This introduction has covered the fundamental problems Coral solves, its core concepts, key features, use cases, and architectural overview. In the following chapters, we'll dive deep into each component:

- **Chapter 2: WeightTensor and Core Data Structures** - Understanding the foundation
- **Chapter 3: Deduplication and Delta Encoding** - How Coral achieves compression
- **Chapter 4: Version Control Deep Dive** - Branches, commits, merges, and conflict resolution
- **Chapter 5: Storage Systems** - HDF5 backend, optimization, and pluggable stores
- **Chapter 6: Training Integration** - PyTorch and TensorFlow integration patterns
- **Chapter 7: CLI Reference** - Complete command reference and workflows
- **Chapter 8: Advanced Topics** - Custom merge strategies, distributed storage, GPU acceleration
- **Chapter 9: Production Deployment** - Best practices, performance tuning, monitoring
- **Chapter 10: Extending Coral** - Plugin development, custom storage backends, integrations

---

**Ready to get started?** The next chapter introduces WeightTensor, the fundamental building block of the Coral system, and shows how to create, manipulate, and store individual weight tensors.
