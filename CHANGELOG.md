# Changelog

All notable changes to Coral will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2024-12-27

### Added
- **Core Features**
  - `WeightTensor` class for neural network weight storage with metadata
  - Content-addressable storage using xxHash for fast deduplication
  - `Deduplicator` engine with configurable similarity thresholds
  - SimHash fingerprinting for O(1) similarity detection
  - LSH (Locality-Sensitive Hashing) index for efficient similarity search

- **Delta Encoding System**
  - Multiple lossless encoding strategies:
    - `FLOAT32_RAW`: Raw float32 differences
    - `COMPRESSED`: zlib-compressed float32 differences
    - `XOR_FLOAT32`: Bitwise XOR with exponent/mantissa separation (15-25% better)
    - `XOR_BFLOAT16`: Optimized for BFloat16 weights
    - `EXPONENT_MANTISSA`: Separate encoding of float components (10-20% better)
  - Lossy encoding strategies for higher compression:
    - `INT8_QUANTIZED`: 8-bit quantization (~75% compression)
    - `INT16_QUANTIZED`: 16-bit quantization (~50% compression)
    - `SPARSE`: Stores only non-zero differences
    - `PER_AXIS_SCALED`: 1-bit signs + per-axis FP16 scales

- **Storage Backends**
  - HDF5 local storage with configurable compression (gzip, lzf, szip)
  - S3-compatible cloud storage (AWS S3, MinIO, DigitalOcean Spaces)
  - Abstract `WeightStore` interface for custom backends

- **Version Control**
  - Git-like repository management with `.coral/` directory
  - Branch creation, checkout, and deletion
  - Immutable commit objects with metadata and timestamps
  - Version graph (DAG) for history tracking
  - Merge strategies: OURS, THEIRS, FAIL, AVERAGE, WEIGHTED

- **Framework Integrations**
  - `CoralTrainer` for PyTorch training with automatic checkpointing
  - `CoralCallback` for PyTorch Lightning
  - `CoralHubClient` for delta-efficient HuggingFace Hub downloads
  - `HFTrainerCallback` for Hugging Face Trainer integration

- **Training & Experiments**
  - `CheckpointManager` with configurable checkpoint policies
  - `TrainingState` for training state tracking
  - `ExperimentTracker` for experiment tracking with metrics

- **CLI (`coral-ml`)**
  - `init`, `add`, `commit`, `status`, `show` commands
  - `log`, `diff`, `compare` for history inspection
  - `branch`, `checkout`, `merge` for branching
  - `tag` for version tagging
  - `remote`, `push`, `pull`, `clone`, `sync` for remote operations
  - `gc`, `stats` for maintenance
  - `experiment` for experiment management

- **Model Publishing**
  - `ModelPublisher` for HuggingFace Hub, MLflow, and local registries

### Performance
- 47.6% space savings vs naive PyTorch storage (1.91x compression ratio)
- Tested with 18 models, 5.3M parameters, 126 weight tensors
- 84% test coverage

### Documentation
- Comprehensive documentation in `docs/book/`
- Research notes in `docs/research/`
- CLI help and examples

[Unreleased]: https://github.com/parkerdgabel/coral/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/parkerdgabel/coral/releases/tag/v1.0.0
