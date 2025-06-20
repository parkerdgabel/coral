# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Coral is a comprehensive neural network weight versioning system that provides git-like version control for ML model weights. It features **lossless delta encoding** for similar weights, automatic deduplication, and comprehensive training integration. Think "git for neural networks" with perfect fidelity and maximum storage efficiency.

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
uv run benchmark.py
```

## Architecture

### Core Components

- **WeightTensor** (`core/weight_tensor.py`): Fundamental data structure representing neural network weights with metadata
- **Deduplicator** (`core/deduplicator.py`): Advanced engine for identifying and eliminating duplicate/similar weights with **lossless delta encoding**
- **Delta Encoding** (`delta/`): **NEW** Lossless reconstruction system for similar weights with multiple compression strategies
- **Storage System** (`storage/`): HDF5-based content-addressable storage with compression and delta support
- **Version Control** (`version_control/`): Complete git-like system with branching, committing, merging, and conflict resolution
- **CLI** (`cli/`): Full-featured command-line interface with git-like commands for weight management

### Integration Points

- **PyTorch Integration** (`integrations/pytorch.py`): Seamless model training integration with automatic checkpointing
- **Training Management** (`training/`): Comprehensive checkpoint management with configurable policies
- **Compression** (`compression/`): Quantization and pruning techniques for space optimization

### Key Design Patterns

1. **Content-Addressable Storage**: Weights are identified by content hash (xxHash)
2. **Lossless Delta Encoding**: Similar weights stored as deltas from reference for perfect reconstruction
3. **Metadata Separation**: Weight data and metadata are handled separately for efficiency
4. **Pluggable Backends**: Storage backends implement the `WeightStore` interface
5. **Git-like Versioning**: Complete branch/commit/merge workflow for model development
6. **Similarity-Based Deduplication**: Uses configurable similarity thresholds with lossless reconstruction

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
- **Coverage Reports**: Generate HTML reports with `uv run pytest --cov=coral --cov-report=html`
- **CI/CD Integration**: Coverage checks must pass before merging

### Running Coverage

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

Test structure:
- `test_weight_tensor.py`: Core WeightTensor functionality
- `test_deduplicator.py`: Deduplication logic and similarity detection
- `test_delta_encoding.py`: **NEW** Delta encoding and lossless reconstruction
- `test_version_control.py`: Git-like version control features
- `test_training.py`: Training integration and checkpoint management

## Key Implementation Details

### Weight Hashing
- Uses xxHash for fast content hashing
- Hashes are computed on normalized weight data
- Content-addressable storage uses these hashes as keys

### Delta Encoding System ⭐ NEW FEATURE
- **Lossless Reconstruction**: Perfect reconstruction of similar weights
- **Multiple Strategies**: Float32 raw, quantized (8/16-bit), sparse, compressed
- **Automatic Selection**: Chooses optimal encoding based on data characteristics
- **Space Efficient**: 90-98% compression for similar weights
- **Configurable**: Choose quality vs compression trade-offs per repository

### Similarity Detection
- Configurable similarity threshold (default 0.98)
- Optimized for weights with same shape/dtype
- Uses cosine similarity to detect near-duplicates
- **NO INFORMATION LOSS**: Similar weights reconstructed perfectly via delta encoding

### Version Control Model
- Branches store references to weight collections
- Commits are immutable snapshots with metadata
- Merging uses content-based conflict resolution

### Storage Optimization
- HDF5 backend with configurable compression (gzip, lzf, szip)
- **Delta Storage**: Separate HDF5 group for delta objects with metadata
- Batch operations for better performance
- Lazy loading of weight data and delta reconstruction
- **Automatic Cleanup**: Garbage collection of unreferenced weights and deltas

## Entry Points

- CLI: `coral` command (defined in setup.py console_scripts)
- Python API: Import from `coral` package
- Main classes: `WeightTensor`, `Deduplicator`, `HDF5Store`, `Repository`, `DeltaEncoder`
- Training: `CoralTrainer`, `CheckpointManager` for seamless training integration

## Dependencies

Core dependencies:
- numpy (array operations)
- h5py (HDF5 storage with delta support)
- xxhash (fast hashing)
- protobuf (serialization)
- tqdm (progress bars)
- networkx (version graph management)

Optional:
- torch (PyTorch integration)
- tensorflow (TensorFlow integration)

## 🎯 Key Innovations

### Lossless Delta Encoding
**Problem Solved**: Previous versions lost information when deduplicating similar weights.
**Solution**: Delta encoding system that stores differences from reference weights, enabling perfect reconstruction.

### Usage Example
```python
# Before: Information loss
weight_a = [1.0, 2.0, 3.0]
weight_b = [1.01, 2.01, 3.01]  # 99% similar
# After deduplication: both became [1.0, 2.0, 3.0] ❌

# After: Perfect reconstruction  
loaded_a = repo.get_weight("weight_a")  # [1.0, 2.0, 3.0] ✓
loaded_b = repo.get_weight("weight_b")  # [1.01, 2.01, 3.01] ✓ Exact!
```

### Delta Encoding Strategies
- `FLOAT32_RAW`: Perfect reconstruction, ~50% compression
- `COMPRESSED`: Perfect reconstruction, ~70% compression  
- `INT8_QUANTIZED`: Minor accuracy loss, ~90% compression
- `SPARSE`: Perfect for few changes, >95% compression

## 🚀 Production Features

- **Automatic Checkpointing**: CoralTrainer handles training checkpoints transparently
- **Git-like CLI**: Full command suite (init, add, commit, branch, merge, tag, diff, log)
- **Training Integration**: Configurable checkpoint policies with metric-based saving
- **Scalable Storage**: Content-addressable with garbage collection
- **Framework Agnostic**: Clean abstractions work with any ML framework

## 📊 Benchmarking & Performance

### Running Benchmarks

The `benchmark.py` script measures actual space savings achieved through Coral's deduplication and delta encoding:

```bash
uv run benchmark.py
```

### Benchmark Targets

**Current Performance (v1.0):**
- **Space Savings**: 47.6% reduction vs naive PyTorch storage
- **Compression Ratio**: 1.91x
- **Test Scale**: 18 models, 5.3M parameters, 126 weight tensors

**Development Goals:**
- Always aim to improve these metrics with each release
- Target >50% space savings for typical ML workflows
- Maintain <1 second overhead for small models (<100M params)

### Benchmark Methodology

The benchmark creates realistic ML scenarios:
1. **Base Models**: CNN and MLP architectures
2. **Variations**: 99.9% similar (fine-tuning), 99% similar (continued training), 95% similar (transfer learning)
3. **Checkpoints**: Exact duplicates (training snapshots)

This simulates real ML workflows where models evolve incrementally, creating many similar weight sets that benefit from deduplication.

### Performance Tips

To maximize space savings:
- Use higher similarity thresholds (0.98+) for training checkpoints
- Enable delta encoding for model variations
- Batch commits when storing multiple related models
- Run `repo.gc()` periodically to clean unreferenced weights

## 🔍 Development Insights

### Module Organization

The codebase follows a clean separation of concerns:

1. **Core Module** (`coral/core/`): Contains fundamental data structures
   - `WeightTensor`: Base abstraction for neural network weights
   - `Deduplicator`: Handles weight similarity detection and deduplication
   - Uses content-addressable storage with xxHash for fast lookups

2. **Delta Module** (`coral/delta/`): Implements lossless reconstruction
   - `DeltaEncoder`: Manages multiple encoding strategies
   - `Delta`: Represents encoded differences between similar weights
   - Supports FLOAT32_RAW, INT8/16_QUANTIZED, SPARSE, and COMPRESSED modes

3. **Storage Module** (`coral/storage/`): Handles persistence
   - `HDF5Store`: Primary storage backend with compression support
   - `SafetensorsStore`: NEW - SafeTensors format integration
   - `WeightStore`: Abstract interface for storage backends

4. **Version Control** (`coral/version_control/`): Git-like functionality
   - `Repository`: Main interface for version control operations
   - `BranchManager`: Handles branch operations
   - `Commit`: Immutable snapshots with metadata
   - `VersionGraph`: Manages commit relationships

5. **Training Integration** (`coral/training/`): ML framework integration
   - `CheckpointManager`: Automatic checkpoint saving during training
   - `TrainingState`: Tracks training progress and metrics
   - Policy-based checkpoint management (every N steps, best metric, etc.)

6. **SafeTensors Integration** (`coral/safetensors/`): NEW feature
   - Full read/write support for SafeTensors format
   - Bidirectional conversion between Coral and SafeTensors
   - Metadata preservation and lazy loading
   - CLI commands: `import-safetensors`, `export-safetensors`, `convert`

### Testing Strategy

The test suite shows extensive coverage focus with multiple test files targeting 80%+ coverage:
- Unit tests for each module (test_weight_tensor.py, test_deduplicator.py, etc.)
- Integration tests for PyTorch and SafeTensors
- CLI command coverage tests
- Delta reconstruction consistency tests
- Multiple coverage-focused test files indicate recent push to reach 80% threshold

### Key Technical Patterns

1. **Content-Addressable Storage**: All weights identified by content hash
   - Uses xxHash for performance
   - Enables automatic deduplication

2. **Lazy Loading**: Weight data loaded on-demand
   - Store references kept until data needed
   - Reduces memory footprint

3. **Delta Encoding**: Critical innovation for space savings
   - Stores only differences from reference weights
   - Multiple encoding strategies based on data characteristics
   - Automatic strategy selection for optimal compression

4. **Plugin Architecture**: Storage backends are pluggable
   - Interface defined in `WeightStore` abstract class
   - Easy to add new storage formats

### Development Tips

1. **Always Use UV**: All Python commands should use `uv run`
   ```bash
   uv run pytest tests/test_specific.py
   uv run python examples/demo.py
   ```

2. **Test First**: Follow TDD practices
   - Write tests before implementation
   - Use existing test patterns as templates
   - Aim for >80% coverage on new code

3. **Handle Edge Cases**:
   - Empty weights/tensors
   - Very small weights (< min_weight_size for delta encoding)
   - Duplicate names in staging
   - Concurrent modifications

4. **Performance Considerations**:
   - Batch operations when possible (see deduplicator batch methods)
   - Use lazy loading for large models
   - Configure delta encoding thresholds based on use case

5. **CLI Development**:
   - Follow git-like command structure
   - Use subparsers for command organization
   - Provide helpful error messages with context
   - Support both long and short options where sensible

6. **Integration Points**:
   - PyTorch: See `integrations/pytorch.py` and `CoralTrainer` class
   - SafeTensors: Use converter functions for import/export
   - Custom frameworks: Implement `WeightStore` interface

### Common Development Tasks

1. **Adding a New Storage Backend**:
   - Inherit from `WeightStore` abstract class
   - Implement all required methods
   - Add tests following existing patterns
   - Update CLI to support new format if needed

2. **Adding a New Delta Encoding Strategy**:
   - Add new `DeltaType` enum value
   - Implement encoding/decoding logic in `DeltaEncoder`
   - Add strategy selection logic
   - Write comprehensive tests

3. **Extending CLI Commands**:
   - Add parser in `cli/main.py`
   - Implement command logic in `CoralCLI` class
   - Add corresponding tests
   - Update help text and documentation

4. **Improving Performance**:
   - Profile with `cProfile` or `line_profiler`
   - Focus on batch operations
   - Consider parallel processing for large repositories
   - Optimize hash computations and similarity checks

### Debugging Tips

1. **Enable Logging**: Set log level to DEBUG for detailed traces
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check Repository State**:
   - Use `coral-ml status` to see staging area
   - Check `.coral/` directory structure
   - Verify HDF5 file contents with `h5dump` or h5py

3. **Delta Encoding Issues**:
   - Check similarity threshold settings
   - Verify min_weight_size configuration
   - Use `DeltaEncoder` directly to test encoding

4. **Version Control Problems**:
   - Examine commit graph with `coral-ml log`
   - Check branch references in `.coral/refs/heads/`
   - Verify HEAD pointer in `.coral/HEAD`