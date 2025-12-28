# Chapter 11: Configuration System

Coral provides a comprehensive configuration system that allows you to customize behavior across all components. This chapter covers configuration file formats, environment variables, programmatic configuration, validation, and best practices.

## 11.1 Configuration Overview

### Design Philosophy

Coral's configuration system follows these principles:

1. **Sensible Defaults**: Works out of the box without any configuration
2. **Multiple Sources**: Configuration from files, environment, and code
3. **Hierarchical Priority**: Clear precedence rules for overlapping settings
4. **Type Safety**: Strong typing with dataclass-based schema
5. **Validation**: Comprehensive validation with clear error messages

### Configuration Sources and Priority

Configuration is loaded from multiple sources, with later sources overriding earlier ones:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Configuration Priority                        │
│              (Highest priority at top)                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Programmatic overrides      (CoralConfig passed to code)    │
│  2. Environment variables       (CORAL_* prefix)                │
│  3. Repository config           (.coral/coral.toml)             │
│  4. User config                 (~/.config/coral/config.toml)   │
│  5. Default values              (Built into schema)             │
└─────────────────────────────────────────────────────────────────┘
```

This allows:
- **Global defaults** in user config for all projects
- **Project-specific settings** in repository config
- **Runtime overrides** via environment variables or code

### Configuration File Format

Coral uses **TOML** format for configuration files:

```toml
# Example coral.toml

[user]
name = "Alice Johnson"
email = "alice@example.com"

[core]
compression = "gzip"
compression_level = 6
similarity_threshold = 0.98
delta_encoding = true
delta_type = "compressed"

[storage]
compression = "gzip"
compression_level = 6
max_cache_size = 1073741824

[logging]
level = "INFO"
format = "%(levelname)s - %(name)s - %(message)s"
```

## 11.2 Configuration Sections

### User Configuration

User identity for commits and experiments:

```toml
[user]
name = "Your Name"         # Author name for commits
email = "you@example.com"  # Author email for commits
```

**Python API:**
```python
from coral import CoralConfig

config = CoralConfig()
config.user.name = "Alice Johnson"
config.user.email = "alice@example.com"
```

### Core Configuration

Primary system behavior settings:

```toml
[core]
# Compression settings
compression = "gzip"          # Options: "gzip", "lzf", "none"
compression_level = 6         # Range: 1-9 (higher = better compression, slower)

# Deduplication settings
similarity_threshold = 0.98   # Range: 0.0-1.0 (similarity for dedup)
magnitude_tolerance = 0.1     # Range: 0.0-1.0 (magnitude difference tolerance)

# Delta encoding
delta_encoding = true         # Enable delta storage for similar weights
delta_type = "compressed"     # See Delta Types section below
strict_reconstruction = true  # Verify reconstruction accuracy

# Default branch
default_branch = "main"       # Branch created on init
```

**Delta Types:**

| Type | Description | Compression | Lossy? |
|------|-------------|-------------|--------|
| `compressed` | zlib-compressed float32 differences | Good | No |
| `xor_float32` | Bitwise XOR with exponent/mantissa separation | Better | No |
| `xor_bfloat16` | XOR optimized for BFloat16 | Better | No |
| `exponent_mantissa` | Separate encoding of float components | Best | No |
| `float32_raw` | Uncompressed float32 differences | None | No |
| `int8_quantized` | 8-bit quantized differences | ~75% | Yes |
| `int16_quantized` | 16-bit quantized differences | ~50% | Yes |
| `sparse` | Only stores non-zero differences | Variable | Yes |
| `per_axis_scaled` | Per-axis scaling with 1-bit signs | ~87% | Yes |

**Recommended settings by use case:**

```toml
# Training checkpoints (lossless, good compression)
[core]
delta_type = "xor_float32"
strict_reconstruction = true

# Research experiments (lossless, best compression)
[core]
delta_type = "exponent_mantissa"
strict_reconstruction = true

# Quick prototyping (lossy, maximum compression)
[core]
delta_type = "int8_quantized"
strict_reconstruction = false
```

### Delta Encoding Configuration

Fine-grained control over delta encoding:

```toml
[delta]
sparse_threshold = 0.001      # Threshold for sparse encoding (values below ignored)
quantization_bits = 16        # Bits for quantized encoding (8 or 16)
min_weight_size = 1024        # Minimum size for delta encoding (bytes)
max_delta_ratio = 1.5         # Maximum delta/original size ratio
```

**Python API:**
```python
config.delta.sparse_threshold = 0.001
config.delta.quantization_bits = 8  # For maximum compression
config.delta.min_weight_size = 4096  # Only encode larger weights
```

### Storage Configuration

Storage backend settings:

```toml
[storage]
compression = "gzip"              # Storage-level compression
compression_level = 6             # Compression level (1-9)
max_cache_size = 1073741824       # 1GB cache size
enable_mmap = true                # Memory-mapped file access
```

### S3 Configuration

Settings for S3-compatible storage backends:

```toml
[s3]
endpoint_url = ""                 # Custom endpoint (for MinIO, etc.)
region = "us-east-1"              # AWS region
access_key = ""                   # Access key (prefer env vars)
secret_key = ""                   # Secret key (prefer env vars)
max_concurrency = 10              # Parallel upload/download threads
chunk_size = 8388608              # 8MB multipart chunk size
use_ssl = true                    # Use HTTPS
verify_ssl = true                 # Verify SSL certificates
```

**Security Note:** Never store credentials in config files. Use environment variables:

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="us-west-2"
```

### LSH Configuration

Locality-Sensitive Hashing for similarity search:

```toml
[lsh]
num_hyperplanes = 8               # Bits per hash (precision)
num_tables = 4                    # Number of hash tables (recall)
seed = 42                         # Random seed for reproducibility
max_candidates = 100              # Maximum candidates per query
```

**Tuning guidance:**

```toml
# High precision (fewer false positives)
[lsh]
num_hyperplanes = 16
num_tables = 2

# High recall (fewer false negatives)
[lsh]
num_hyperplanes = 6
num_tables = 8

# Balanced (recommended)
[lsh]
num_hyperplanes = 8
num_tables = 4
```

### SimHash Configuration

Fast fingerprint-based similarity:

```toml
[simhash]
num_bits = 64                     # Fingerprint size (64 or 128)
similarity_threshold = 0.1       # Max Hamming distance ratio
seed = 42                         # Random seed
```

### Checkpoint Configuration

Training checkpoint settings:

```toml
[checkpoint]
save_every_n_epochs = 1           # Save frequency (epochs)
save_every_n_steps = 0            # Save frequency (steps, 0 = disabled)
keep_best_n_checkpoints = 3       # Best checkpoints to retain
keep_last_n_checkpoints = 5       # Recent checkpoints to retain
metric_name = "val_loss"          # Metric for "best" selection
metric_mode = "min"               # "min" or "max"
auto_commit = true                # Auto-commit on save
commit_message_template = "Checkpoint: epoch {epoch}, step {step}"
```

**Example configurations:**

```toml
# Research: Keep everything
[checkpoint]
save_every_n_epochs = 1
keep_best_n_checkpoints = 10
keep_last_n_checkpoints = 100

# Production: Minimal storage
[checkpoint]
save_every_n_epochs = 5
keep_best_n_checkpoints = 1
keep_last_n_checkpoints = 2
```

### Logging Configuration

Logging behavior:

```toml
[logging]
level = "INFO"                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
format = "%(levelname)s - %(name)s - %(message)s"
file = ""                         # Log file path (empty = stderr only)
```

### HuggingFace Integration

Settings for HuggingFace Hub integration:

```toml
[huggingface]
token = ""                        # HF token (prefer HF_TOKEN env var)
default_organization = ""         # Default org for publishing
cache_dir = ""                    # HF cache directory
```

### Remote Configurations

Define named remote storage backends:

```toml
[remotes.origin]
url = "s3://my-bucket/coral-repo"
backend = "s3"
region = "us-east-1"
auto_push = false
auto_pull = false

[remotes.backup]
url = "file:///mnt/backup/coral"
backend = "file"

[remotes.team]
url = "s3://team-bucket/shared-models"
backend = "s3"
region = "eu-west-1"
```

## 11.3 Environment Variables

All configuration options can be set via environment variables with the `CORAL_` prefix:

### Naming Convention

```
CORAL_<SECTION>_<KEY>
```

Examples:
- `CORAL_CORE_COMPRESSION` → `core.compression`
- `CORAL_CORE_SIMILARITY_THRESHOLD` → `core.similarity_threshold`
- `CORAL_DELTA_QUANTIZATION_BITS` → `delta.quantization_bits`
- `CORAL_LOGGING_LEVEL` → `logging.level`

### Type Conversion

Environment variables are automatically converted:

| Value | Converted To |
|-------|--------------|
| `true`, `yes`, `1`, `on` | `True` |
| `false`, `no`, `0`, `off` | `False` |
| `none`, `null`, `` | `None` |
| `123` | `int` |
| `12.34` | `float` |
| Other | `str` |

### Common Environment Variables

```bash
# User identity
export CORAL_USER_NAME="Alice Johnson"
export CORAL_USER_EMAIL="alice@example.com"

# Core settings
export CORAL_CORE_COMPRESSION="gzip"
export CORAL_CORE_COMPRESSION_LEVEL=6
export CORAL_CORE_SIMILARITY_THRESHOLD=0.98
export CORAL_CORE_DELTA_TYPE="xor_float32"

# Logging
export CORAL_LOGGING_LEVEL="DEBUG"

# S3 credentials (standard AWS pattern)
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="us-west-2"

# HuggingFace
export HF_TOKEN="hf_..."
```

### CI/CD Example

```yaml
# GitHub Actions example
jobs:
  train:
    runs-on: ubuntu-latest
    env:
      CORAL_CORE_COMPRESSION: gzip
      CORAL_CORE_DELTA_TYPE: compressed
      CORAL_LOGGING_LEVEL: INFO
      AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
      AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    steps:
      - uses: actions/checkout@v4
      - name: Train model
        run: python train.py
```

## 11.4 Configuration Files

### User Configuration

Global configuration for all projects:

**Location:** `~/.config/coral/config.toml`

```toml
# ~/.config/coral/config.toml

[user]
name = "Alice Johnson"
email = "alice@example.com"

[core]
compression = "gzip"
compression_level = 6
similarity_threshold = 0.98

[logging]
level = "INFO"

[huggingface]
default_organization = "my-org"
```

### Repository Configuration

Project-specific settings:

**Location:** `.coral/coral.toml`

```toml
# .coral/coral.toml

[core]
# Override compression for this project
compression = "lzf"
compression_level = 4

# Higher similarity threshold for fine-tuning
similarity_threshold = 0.995
delta_type = "exponent_mantissa"

[checkpoint]
save_every_n_epochs = 1
keep_best_n_checkpoints = 5
metric_name = "val_accuracy"
metric_mode = "max"

[remotes.origin]
url = "s3://project-bucket/models"
backend = "s3"
region = "us-east-1"
```

### Legacy Configuration

Coral supports the legacy `config.json` format for backwards compatibility:

```json
{
  "user": {
    "name": "Alice Johnson",
    "email": "alice@example.com"
  },
  "core": {
    "compression": "gzip",
    "similarity_threshold": 0.98,
    "delta_encoding": true,
    "delta_type": "compressed"
  }
}
```

**Migration:** Use `coral-ml config migrate` to convert to TOML format.

## 11.5 Python API

### Loading Configuration

```python
from coral import load_config, get_default_config, CoralConfig
from pathlib import Path

# Load from default locations
config = load_config()

# Load with specific repository path
config = load_config(repo_path=Path("./my-project"))

# Load with custom user config
config = load_config(
    repo_path=Path("./my-project"),
    user_config_path=Path("/custom/config.toml")
)

# Get defaults only (no file loading)
config = get_default_config()
```

### Creating Configuration Programmatically

```python
from coral import CoralConfig, CoreConfig, UserConfig, DeltaEncodingConfig

# Create with all defaults
config = CoralConfig()

# Create with custom settings
config = CoralConfig(
    user=UserConfig(
        name="Alice Johnson",
        email="alice@example.com"
    ),
    core=CoreConfig(
        compression="gzip",
        compression_level=9,
        similarity_threshold=0.99,
        delta_type="xor_float32"
    ),
    delta=DeltaEncodingConfig(
        quantization_bits=16,
        sparse_threshold=0.0001
    )
)
```

### Accessing Configuration Values

```python
# Direct attribute access
print(config.core.compression)  # "gzip"
print(config.user.name)  # "Alice Johnson"

# Dot notation access
value = config.get_nested("core.similarity_threshold")  # 0.98
value = config.get_nested("delta.quantization_bits", default=16)

# Setting values
config.set_nested("core.compression_level", 9)
config.core.similarity_threshold = 0.995
```

### Using with Repository

```python
from coral import Repository, CoralConfig

# Create repository with custom config
config = CoralConfig()
config.core.similarity_threshold = 0.995

repo = Repository("./my-model", init=True, config=config)

# Access repository config
print(repo.coral_config.core.delta_type)

# Update configuration
repo.update_config("core.compression_level", 9)

# Save to repository
repo.save_config()
```

### Configuration Validation

```python
from coral import validate_config, CoralConfig

config = CoralConfig()
config.core.compression_level = 15  # Invalid!

result = validate_config(config)

if not result:
    print("Configuration errors:")
    for error in result.errors:
        print(f"  {error}")

    print("Configuration warnings:")
    for warning in result.warnings:
        print(f"  {warning}")
```

**Example output:**
```
Configuration errors:
  core.compression_level: must be between 1 and 9 (got: 15)

Configuration warnings:
  core.similarity_threshold: low threshold may cause false positives in deduplication (got: 0.85)
```

### Serialization

```python
# Convert to dictionary
config_dict = config.to_dict()
print(config_dict)
# {'user': {'name': 'Alice', ...}, 'core': {...}, ...}

# Create from dictionary
new_config = CoralConfig.from_dict(config_dict)
```

## 11.6 CLI Commands

### coral-ml config list

List all configuration values:

```bash
coral-ml config list

# Output:
Configuration:
  user.name = "Alice Johnson"
  user.email = "alice@example.com"
  core.compression = "gzip"
  core.compression_level = 6
  core.similarity_threshold = 0.98
  core.delta_encoding = true
  core.delta_type = "compressed"
  ...
```

### coral-ml config get

Get a specific configuration value:

```bash
coral-ml config get core.similarity_threshold
# 0.98

coral-ml config get user.name
# Alice Johnson

coral-ml config get delta.quantization_bits
# 16
```

### coral-ml config set

Set a configuration value in the repository config:

```bash
# Set compression level
coral-ml config set core.compression_level 9
# Set core.compression_level = 9

# Set similarity threshold
coral-ml config set core.similarity_threshold 0.995
# Set core.similarity_threshold = 0.995

# Set delta type
coral-ml config set core.delta_type xor_float32
# Set core.delta_type = xor_float32
```

### coral-ml config show

Show the current configuration file:

```bash
coral-ml config show

# Output:
[user]
name = "Alice Johnson"
email = "alice@example.com"

[core]
compression = "gzip"
compression_level = 9
similarity_threshold = 0.995
delta_type = "xor_float32"
...
```

### coral-ml config validate

Validate the current configuration:

```bash
coral-ml config validate

# Output (valid):
Configuration is valid.

# Output (with issues):
Configuration validation:

Errors:
  core.compression_level: must be between 1 and 9 (got: 15)

Warnings:
  core.similarity_threshold: low threshold may cause false positives (got: 0.85)
```

### coral-ml config migrate

Migrate legacy config.json to TOML format:

```bash
coral-ml config migrate

# Output:
Migrated legacy config to .coral/coral.toml
Old config backed up to .coral/config.json.bak
```

## 11.7 Practical Examples

### Example 1: High-Compression Training Setup

Optimize for minimal storage during training experiments:

```toml
# .coral/coral.toml - High compression configuration

[core]
compression = "gzip"
compression_level = 9
similarity_threshold = 0.98
delta_type = "exponent_mantissa"  # Best lossless compression
strict_reconstruction = true

[checkpoint]
save_every_n_epochs = 1
keep_best_n_checkpoints = 3
keep_last_n_checkpoints = 5
metric_name = "val_loss"
metric_mode = "min"

[storage]
compression = "gzip"
compression_level = 9
```

### Example 2: Fast Iteration Configuration

Optimize for speed during rapid prototyping:

```toml
# .coral/coral.toml - Speed-optimized configuration

[core]
compression = "lzf"           # Faster than gzip
compression_level = 1
similarity_threshold = 0.95   # More aggressive dedup
delta_type = "compressed"     # Good balance
strict_reconstruction = false

[checkpoint]
save_every_n_epochs = 5       # Less frequent saves
keep_best_n_checkpoints = 1
keep_last_n_checkpoints = 2

[storage]
compression = "lzf"
compression_level = 1
```

### Example 3: Team Collaboration Setup

Configuration for multi-user development:

```toml
# ~/.config/coral/config.toml - User-specific settings

[user]
name = "Alice Johnson"
email = "alice@team.com"

[logging]
level = "INFO"

[huggingface]
default_organization = "my-team"
```

```toml
# .coral/coral.toml - Project-wide settings

[core]
compression = "gzip"
compression_level = 6
similarity_threshold = 0.98
delta_type = "xor_float32"

[remotes.origin]
url = "s3://team-bucket/bert-project"
backend = "s3"
region = "us-east-1"

[remotes.backup]
url = "file:///mnt/shared/backup"
backend = "file"
```

### Example 4: CI/CD Pipeline Configuration

Environment-based configuration for automated pipelines:

```bash
#!/bin/bash
# ci-train.sh

# Set configuration via environment
export CORAL_CORE_COMPRESSION="gzip"
export CORAL_CORE_COMPRESSION_LEVEL=6
export CORAL_CORE_DELTA_TYPE="compressed"
export CORAL_LOGGING_LEVEL="WARNING"
export CORAL_CHECKPOINT_AUTO_COMMIT="true"

# AWS credentials from secrets
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}"
export AWS_REGION="us-east-1"

# Run training
python train.py

# Push results
coral-ml push origin
```

### Example 5: Large Model Configuration

Optimized settings for large language models:

```toml
# .coral/coral.toml - Large model configuration

[core]
compression = "gzip"
compression_level = 6
similarity_threshold = 0.995  # Higher threshold for subtle changes
delta_type = "xor_float32"
strict_reconstruction = true

[delta]
min_weight_size = 1048576     # Only delta-encode weights > 1MB
max_delta_ratio = 1.2         # Fallback to full storage if delta is large

[storage]
max_cache_size = 4294967296   # 4GB cache for large weights
enable_mmap = true            # Memory-mapped access

[s3]
max_concurrency = 20          # More parallel uploads
chunk_size = 16777216         # 16MB chunks for multipart

[lsh]
num_hyperplanes = 12          # Higher precision for large repos
num_tables = 6
max_candidates = 200
```

## 11.8 Configuration Best Practices

### 1. Use Layered Configuration

- **User config** (`~/.config/coral/config.toml`): Personal preferences, identity
- **Repo config** (`.coral/coral.toml`): Project-specific settings
- **Environment variables**: CI/CD, secrets, runtime overrides

### 2. Never Store Secrets in Config Files

```toml
# BAD - Don't do this!
[s3]
access_key = "AKIAIOSFODNN7EXAMPLE"
secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
```

Instead, use environment variables:
```bash
export AWS_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE"
export AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
```

### 3. Validate Before Production

Always validate configuration in CI:

```bash
coral-ml config validate || exit 1
```

### 4. Document Project Settings

Include configuration rationale in your project:

```toml
# .coral/coral.toml
#
# Configuration for BERT fine-tuning project
#
# We use exponent_mantissa delta encoding because fine-tuning
# creates many similar weights with small numerical differences.
#
# Similarity threshold of 0.995 is appropriate for fine-tuning
# where weights change subtly between epochs.

[core]
delta_type = "exponent_mantissa"
similarity_threshold = 0.995
```

### 5. Use Appropriate Delta Types

| Scenario | Recommended Delta Type |
|----------|----------------------|
| Training checkpoints | `xor_float32` or `exponent_mantissa` |
| Fine-tuning | `exponent_mantissa` |
| Quick experiments | `compressed` |
| Maximum compression | `int8_quantized` (lossy) |
| BFloat16 models | `xor_bfloat16` |

### 6. Monitor Storage Efficiency

Regularly check if your configuration is effective:

```bash
coral-ml stats

# Look for:
# - Delta-encoded weights percentage
# - Compression ratio
# - Space savings
```

Adjust settings if savings are lower than expected.

## 11.9 Troubleshooting

### Configuration Not Loading

**Symptom:** Settings in config file aren't being applied.

**Solutions:**
1. Check file location: `~/.config/coral/config.toml` or `.coral/coral.toml`
2. Validate TOML syntax: `coral-ml config validate`
3. Check priority: Environment variables override files
4. Enable debug logging: `export CORAL_LOGGING_LEVEL=DEBUG`

### Environment Variable Not Working

**Symptom:** `CORAL_*` environment variable is ignored.

**Solutions:**
1. Check naming: `CORAL_<SECTION>_<KEY>` (all uppercase, underscores)
2. Verify export: `echo $CORAL_CORE_COMPRESSION`
3. Check type conversion: Booleans need `true`/`false`, not `True`/`False`

### Validation Errors

**Common errors and fixes:**

```bash
# Error: compression_level must be between 1 and 9
coral-ml config set core.compression_level 6

# Error: similarity_threshold must be between 0.0 and 1.0
coral-ml config set core.similarity_threshold 0.98

# Error: quantization_bits must be 8 or 16
coral-ml config set delta.quantization_bits 16
```

### Migration Issues

**Symptom:** Legacy `config.json` not being migrated.

**Solution:**
```bash
# Force migration
coral-ml config migrate

# Manual migration if needed
mv .coral/config.json .coral/config.json.bak
coral-ml config set core.compression gzip
# ... set other values
```

## Summary

Coral's configuration system provides:

- **TOML-based** configuration files for readability
- **Multiple sources** with clear priority ordering
- **Environment variables** for runtime and CI/CD overrides
- **Type-safe schema** with dataclass-based validation
- **CLI commands** for easy configuration management
- **Python API** for programmatic access

Key configuration areas:
- **Core**: Compression, similarity, delta encoding
- **Delta**: Fine-grained delta encoding control
- **Storage**: Backend settings and caching
- **Checkpoint**: Training checkpoint policies
- **Remotes**: Remote storage backends

Use layered configuration with user, repository, and environment sources to create flexible, maintainable ML pipelines.
