# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Coral is a production-ready neural network weight versioning system that provides git-like version control for ML model weights. It features **lossless delta encoding** for similar weights, automatic deduplication, content-addressable storage, and seamless training integration. Think "git for neural networks" with perfect fidelity and maximum storage efficiency.

**Version**: 1.0.0
**Package Name**: `coral-ml`
**Python Support**: 3.9+

## Development Workflow

### Git Worktree Setup

When beginning development, create a git worktree to isolate your changes:

```bash
# Create a new worktree with a descriptive branch name
git worktree add ../coral-feature-<feature-name> -b feature/<feature-name>

# Navigate to the new worktree
cd ../coral-feature-<feature-name>

# Set up the development environment in the new worktree
uv sync --extra dev --extra torch
```

After completing development:

```bash
# From within the worktree, push your changes
git push -u origin feature/<feature-name>

# Return to main repository
cd ../coral

# Merge the feature branch
git merge feature/<feature-name>

# Clean up the worktree
git worktree remove ../coral-feature-<feature-name>

# Delete the feature branch after merging
git branch -d feature/<feature-name>
```

### Development Commands

All Python-related commands should be run through `uv`:

```bash
# Install dependencies
uv sync

# Install with optional dependencies
uv sync --extra dev --extra torch

# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_weight_tensor.py

# Run tests with coverage
uv run pytest --cov=coral --cov-report=html

# Code formatting and linting
uv run ruff format src/ tests/ examples/
uv run ruff check src/ tests/ examples/

# Fix auto-fixable linting issues
uv run ruff check --fix src/ tests/ examples/

# Type checking
uv run mypy src/

# Run examples
uv run python examples/basic_usage.py
uv run python examples/pytorch_training_example.py
uv run python examples/delta_encoding_demo.py

# Run benchmarks to measure space savings
uv run python benchmark.py
uv run python benchmark_delta_strategies.py
```

## Architecture

### Directory Structure

```
src/coral/
├── __init__.py              # Package exports
├── cli/                     # Command-line interface
│   ├── __init__.py
│   └── main.py              # CLI entry point and commands
├── core/                    # Core data structures and algorithms
│   ├── __init__.py
│   ├── weight_tensor.py     # WeightTensor class
│   ├── deduplicator.py      # Deduplication engine
│   ├── simhash.py           # SimHash fingerprinting for O(1) similarity
│   └── lsh_index.py         # LSH index for efficient similarity search
├── delta/                   # Delta encoding system
│   ├── __init__.py
│   ├── delta_encoder.py     # Main encoder with multiple strategies
│   └── compression.py       # Compression utilities
├── storage/                 # Storage backends
│   ├── __init__.py
│   ├── weight_store.py      # Abstract storage interface
│   ├── hdf5_store.py        # HDF5 local storage backend
│   └── s3_store.py          # S3-compatible cloud storage
├── version_control/         # Git-like versioning
│   ├── __init__.py
│   ├── repository.py        # Main Repository class
│   ├── branch.py            # Branch management
│   ├── commit.py            # Commit objects
│   └── version.py           # Version graph
├── integrations/            # Framework integrations
│   ├── __init__.py
│   ├── pytorch.py           # PyTorch integration (CoralTrainer)
│   ├── lightning.py         # PyTorch Lightning callback
│   ├── huggingface.py       # HF Hub delta-efficient downloads
│   └── hf_trainer.py        # HF Trainer callback
├── training/                # Training management
│   ├── __init__.py
│   ├── checkpoint_manager.py # Checkpoint policies
│   └── training_state.py    # Training state tracking
├── compression/             # Weight compression
│   ├── __init__.py
│   ├── quantization.py      # Quantization techniques
│   └── pruning.py           # Weight pruning
├── remotes/                 # Remote repository management
│   ├── __init__.py
│   ├── remote.py            # Remote configuration
│   └── sync.py              # Sync operations
├── registry/                # Model publishing
│   ├── __init__.py
│   └── registry.py          # HF Hub, MLflow publishing
├── experiments/             # Experiment tracking
│   ├── __init__.py
│   └── experiment.py        # Experiment/run tracking
└── utils/                   # Utilities
    ├── __init__.py
    ├── visualization.py     # Visualization utilities
    ├── json_utils.py        # JSON serialization
    └── similarity.py        # Similarity calculations
```

### Core Components

- **WeightTensor** (`core/weight_tensor.py`): Fundamental data structure representing neural network weights with metadata (shape, dtype, name, hash)
- **Deduplicator** (`core/deduplicator.py`): Advanced engine for identifying and eliminating duplicate/similar weights with **lossless delta encoding**
- **SimHash** (`core/simhash.py`): Locality-sensitive hashing for O(1) similarity detection using Hamming distance
- **LSH Index** (`core/lsh_index.py`): Locality-Sensitive Hashing index for O(1) average-time similarity search

### Delta Encoding System

- **DeltaEncoder** (`delta/delta_encoder.py`): Multiple encoding strategies for storing weight differences

**Lossless strategies** (perfect reconstruction):
- `FLOAT32_RAW`: Raw float32 differences, no compression
- `COMPRESSED`: zlib-compressed float32 differences
- `XOR_FLOAT32`: Bitwise XOR with exponent/mantissa separation (15-25% better)
- `XOR_BFLOAT16`: XOR optimized for BFloat16 weights
- `EXPONENT_MANTISSA`: Separate encoding of float components (10-20% better)

**Lossy strategies** (approximate reconstruction):
- `INT8_QUANTIZED`: 8-bit quantization (~75% compression)
- `INT16_QUANTIZED`: 16-bit quantization (~50% compression)
- `SPARSE`: Only stores non-zero differences (discards values < threshold)
- `PER_AXIS_SCALED`: 1-bit signs + per-axis FP16 scales

### Storage Backends

- **HDF5Store** (`storage/hdf5_store.py`): Local content-addressable storage with compression
- **S3Store** (`storage/s3_store.py`): S3-compatible cloud storage (AWS S3, MinIO, DigitalOcean Spaces)
- **WeightStore** (`storage/weight_store.py`): Abstract interface for storage backends

### Version Control

- **Repository** (`version_control/repository.py`): Main class for all version control operations
- **BranchManager** (`version_control/branch.py`): Branch creation, checkout, deletion
- **Commit** (`version_control/commit.py`): Immutable commit objects with metadata
- **VersionGraph** (`version_control/version.py`): DAG for version history

**Merge Strategies**:
- `OURS`: Prefer current branch weights
- `THEIRS`: Prefer source branch weights
- `FAIL`: Raise error on conflicts
- `AVERAGE`: Average conflicting weights
- `WEIGHTED`: Weighted average with configurable alpha

### Framework Integrations

- **CoralTrainer** (`integrations/pytorch.py`): PyTorch training with automatic checkpointing
- **CoralCallback** (`integrations/lightning.py`): PyTorch Lightning callback
- **CoralHubClient** (`integrations/huggingface.py`): Delta-efficient HF Hub downloads
- **HFTrainerCallback** (`integrations/hf_trainer.py`): Hugging Face Trainer integration

### Training & Experiments

- **CheckpointManager** (`training/checkpoint_manager.py`): Configurable checkpoint policies
- **TrainingState** (`training/training_state.py`): Training state tracking
- **ExperimentTracker** (`experiments/experiment.py`): Experiment tracking with metrics

### Model Publishing

- **ModelPublisher** (`registry/registry.py`): Publish to HuggingFace Hub, MLflow, or local registries

### Key Design Patterns

1. **Content-Addressable Storage**: Weights identified by xxHash content hash
2. **Lossless Delta Encoding**: Similar weights stored as deltas for perfect reconstruction
3. **Metadata Separation**: Weight data and metadata handled separately
4. **Pluggable Backends**: Storage backends implement `WeightStore` interface
5. **Git-like Versioning**: Complete branch/commit/merge workflow
6. **Similarity-Based Deduplication**: Configurable thresholds with LSH acceleration

## CLI Commands

The CLI is accessible via `coral-ml` (defined in pyproject.toml):

```bash
# Repository initialization
coral-ml init [path]

# Weight management
coral-ml add <weights...>          # Stage weights
coral-ml commit -m "message"       # Commit staged weights
coral-ml status                    # Show repository status
coral-ml show <weight> [-c commit] # Show weight info

# History and comparison
coral-ml log [-n N] [--oneline]    # Show commit history
coral-ml diff <from> [to]          # Show differences
coral-ml compare <ref1> [ref2]     # Compare commits/branches

# Branching
coral-ml branch [name]             # Create or list branches
coral-ml branch -d <name>          # Delete branch
coral-ml checkout <target>         # Checkout branch or commit
coral-ml merge <branch> [-m msg]   # Merge branch

# Tagging
coral-ml tag <name> [-d desc]      # Create version tag

# Remote operations
coral-ml remote add <name> <url>   # Add remote
coral-ml remote remove <name>      # Remove remote
coral-ml remote list               # List remotes
coral-ml push [remote] [--all]     # Push to remote
coral-ml pull [remote] [--all]     # Pull from remote
coral-ml clone <url> [path]        # Clone remote repository
coral-ml sync [remote]             # Bidirectional sync
coral-ml sync-status [remote]      # Show sync status

# Maintenance
coral-ml gc                        # Garbage collect unreferenced weights
coral-ml stats [--json]            # Show repository statistics

# Experiments
coral-ml experiment                # Manage experiments
```

## Test-Driven Development

Follow TDD practices:

1. Write tests first for new features
2. Run tests to ensure they fail initially
3. Implement feature to make tests pass
4. Refactor while keeping tests green
5. Ensure code coverage remains above 80%

### Code Coverage Requirements

- **Minimum Coverage**: 80% for all new code
- **Target Coverage**: 90%+ for core modules
- **Current Coverage**: 84%

### Running Tests and Coverage

```bash
# Run tests with coverage report
uv run pytest --cov=coral --cov-report=term-missing

# Generate detailed HTML coverage report
uv run pytest --cov=coral --cov-report=html
# View report: open htmlcov/index.html

# Check coverage for specific modules
uv run pytest --cov=coral.core --cov=coral.delta tests/

# Fail if coverage is below threshold
uv run pytest --cov=coral --cov-fail-under=80
```

### Test Structure

- `test_weight_tensor.py`: Core WeightTensor functionality
- `test_deduplicator.py`: Deduplication logic and similarity detection
- `test_delta_encoding.py`: Delta encoding and lossless reconstruction
- `test_advanced_delta_encoding.py`: Advanced delta strategies (XOR, exponent-mantissa)
- `test_simhash.py`: SimHash fingerprinting
- `test_version_control.py`: Git-like version control features
- `test_training.py`: Training integration and checkpoint management
- `test_pytorch_integration.py`: PyTorch-specific integration tests
- `test_hdf5_store.py`: HDF5 storage backend
- `test_cli.py`: CLI command tests
- `test_experiments.py`: Experiment tracking
- `test_registry.py`: Model publishing

## Key Implementation Details

### Weight Hashing
- Uses xxHash for fast content hashing
- Hashes computed on normalized weight data
- Content-addressable storage uses hashes as keys

### Similarity Detection
- **SimHash**: O(1) fingerprint-based similarity via Hamming distance
- **LSH Index**: Multi-table locality-sensitive hashing for candidate retrieval
- Configurable similarity threshold (default 0.98)
- Optimized for weights with same shape/dtype
- Uses cosine similarity to detect near-duplicates

### Version Control Model
- Branches store references to weight collections
- Commits are immutable snapshots with metadata
- Merging uses content-based conflict resolution
- Repository stored in `.coral/` directory

### Storage Optimization
- HDF5 backend with configurable compression (gzip, lzf, szip)
- Delta storage in separate HDF5 group with metadata
- Batch operations for better performance
- Lazy loading of weight data and delta reconstruction
- Automatic garbage collection of unreferenced weights

## Entry Points

- **CLI**: `coral-ml` command (defined in pyproject.toml `[project.scripts]`)
- **Python API**: Import from `coral` package
- **Main classes**: `WeightTensor`, `Deduplicator`, `HDF5Store`, `Repository`, `DeltaEncoder`
- **Training**: `CoralTrainer`, `CheckpointManager` for seamless training integration

## Dependencies

### Core (required)
- `numpy>=1.21.0` - Array operations
- `h5py>=3.0.0` - HDF5 storage
- `xxhash>=3.0.0` - Fast content hashing
- `tqdm>=4.60.0` - Progress bars
- `networkx>=3.1` - Version graph management

### Optional Dependencies

Install with `pip install coral-ml[extra]` or `uv sync --extra <extra>`:

- **dev**: pytest, pytest-cov, ruff, mypy
- **torch**: PyTorch integration
- **tensorflow**: TensorFlow integration (future)
- **s3**: boto3 for S3/MinIO storage
- **huggingface**: huggingface-hub, safetensors for HF Hub integration
- **all**: All optional dependencies

## Usage Examples

### Basic Python API

```python
from coral import Repository, WeightTensor
from coral.core.weight_tensor import WeightMetadata
import numpy as np

# Initialize repository
repo = Repository("./my_model", init=True)

# Create weights
weight = WeightTensor(
    data=np.random.randn(256, 128).astype(np.float32),
    metadata=WeightMetadata(name="layer1.weight", shape=(256, 128), dtype=np.float32)
)

# Stage and commit
repo.stage_weights({"layer1.weight": weight})
commit = repo.commit("Initial weights")

# Branch workflow
repo.create_branch("experiment")
repo.checkout("experiment")
# ... modify weights ...
repo.checkout("main")
repo.merge("experiment")
```

### Delta Encoding for Similar Weights

```python
from coral import DeltaEncoder, DeltaConfig, DeltaType

# Configure lossless encoding
config = DeltaConfig(delta_type=DeltaType.COMPRESSED)
encoder = DeltaEncoder(config)

# Encode delta between similar weights
delta = encoder.encode(original_weight, similar_weight)

# Reconstruct exactly
reconstructed = encoder.decode(delta, original_weight)
assert np.array_equal(reconstructed.data, similar_weight.data)  # Perfect!
```

### PyTorch Training Integration

```python
from coral.integrations.pytorch import CoralTrainer
from coral.training import CheckpointConfig

config = CheckpointConfig(
    save_every_n_epochs=1,
    save_on_best_metric="val_loss",
    keep_best_n_checkpoints=3,
)

trainer = CoralTrainer(model, repo_path="./checkpoints", config=config)
trainer.fit(train_loader, val_loader, epochs=10)
```

## Benchmarking & Performance

### Running Benchmarks

```bash
uv run python benchmark.py
uv run python benchmark_delta_strategies.py
```

### Current Performance (v1.0)

- **Space Savings**: 47.6% reduction vs naive PyTorch storage
- **Compression Ratio**: 1.91x
- **Test Scale**: 18 models, 5.3M parameters, 126 weight tensors
- **Test Coverage**: 84%

### Performance Goals

- Target >50% space savings for typical ML workflows
- Maintain <1 second overhead for small models (<100M params)
- SimHash lookup: O(1) average time
- LSH candidate retrieval: O(1) average vs O(n) linear scan

### Optimization Tips

- Use higher similarity thresholds (0.98+) for training checkpoints
- Enable lossless delta encoding for model variations
- Use `XOR_FLOAT32` or `EXPONENT_MANTISSA` for best lossless compression
- Batch commits when storing multiple related models
- Run `repo.gc()` periodically to clean unreferenced weights

## Documentation

Comprehensive documentation is available in `docs/`:

- `docs/book/` - Complete system book with chapters on:
  - Introduction and getting started
  - Core architecture
  - Delta encoding deep-dive
  - Storage system
  - Version control
  - Training integration
  - CLI interface
  - Advanced features
  - Performance benchmarks
  - API reference

- `docs/research/` - Research notes on weight deduplication methods
