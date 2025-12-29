# CLI Module

This module provides the command-line interface for Coral, offering git-like commands for neural network weight version control.

## Overview

The CLI provides:
- Git-like workflow (`init`, `add`, `commit`, `checkout`, `branch`, `merge`)
- Remote operations (`push`, `pull`, `clone`, `sync`)
- Weight inspection (`log`, `diff`, `show`, `stats`)
- Experiment management

## Entry Point

The CLI is accessible via `coral-ml` (defined in pyproject.toml):
```bash
coral-ml <command> [options]
```

## Key Files

### `main.py`

The main CLI implementation using argparse.

**CoralCLI** (class):
- Creates argument parser with subcommands
- Dispatches to appropriate handler methods
- Provides consistent output formatting

## Commands Reference

### Repository Management

```bash
# Initialize new repository
coral-ml init [path]

# Show repository status
coral-ml status

# Show repository statistics
coral-ml stats [--json]

# Garbage collect unreferenced weights
coral-ml gc
```

### Weight Management

```bash
# Stage weights for commit
coral-ml add <weight_files...>

# Commit staged weights
coral-ml commit -m "message" [--author NAME] [--email EMAIL] [-t TAG]

# Show weight information
coral-ml show <weight_name> [-c COMMIT]
```

### History & Comparison

```bash
# Show commit history
coral-ml log [-n NUMBER] [--oneline]

# Show differences between commits
coral-ml diff <from_ref> [to_ref]

# Compare weights between commits/branches
coral-ml compare <ref1> [ref2] [-v]
```

### Branching

```bash
# List branches
coral-ml branch

# Create branch
coral-ml branch <name>

# Delete branch
coral-ml branch -d <name>

# Checkout branch or commit
coral-ml checkout <target>

# Merge branch
coral-ml merge <branch> [-m MESSAGE]
```

### Tagging

```bash
# Create version tag
coral-ml tag <name> [-d DESCRIPTION] [-c COMMIT] [--metric KEY=VALUE]
```

### Remote Operations

```bash
# Manage remotes
coral-ml remote add <name> <url>
coral-ml remote remove <name>
coral-ml remote list
coral-ml remote show <name>

# Push to remote
coral-ml push [remote] [--all] [--force]

# Pull from remote
coral-ml pull [remote] [--all] [--force]

# Clone repository
coral-ml clone <url> [path]

# Bidirectional sync
coral-ml sync [remote]

# Show sync status
coral-ml sync-status [remote] [--json]
```

### Experiments

```bash
# Manage experiments (subcommands)
coral-ml experiment <subcommand>
```

## Remote URL Formats

The CLI supports various remote URL formats:

| Format | Description |
|--------|-------------|
| `s3://bucket/path` | AWS S3 |
| `minio://host:port/bucket` | MinIO (S3-compatible) |
| `file:///path/to/dir` | Local filesystem |

## Example Workflows

### Basic Workflow

```bash
# Initialize repository
coral-ml init ./my-model

# Add weights (supports .npy and .npz files)
coral-ml add weights.npz

# Commit
coral-ml commit -m "Initial model weights"

# View history
coral-ml log
```

### Branching Workflow

```bash
# Create experiment branch
coral-ml branch experiment

# Switch to branch
coral-ml checkout experiment

# Make changes and commit
coral-ml add weights.npz
coral-ml commit -m "Experimental changes"

# Switch back and merge
coral-ml checkout main
coral-ml merge experiment -m "Merge experiment"
```

### Remote Sync

```bash
# Add remote
coral-ml remote add origin s3://my-bucket/models

# Push weights
coral-ml push origin

# Check sync status
coral-ml sync-status origin

# Pull from remote
coral-ml pull origin
```

## Output Formats

Commands support various output formats:

```bash
# JSON output for scripting
coral-ml stats --json
coral-ml sync-status origin --json

# Compact output
coral-ml log --oneline

# Verbose output
coral-ml compare ref1 ref2 -v
```

## Weight File Formats

The `coral-ml add` command currently supports these formats:

| Extension | Format |
|-----------|--------|
| `.npz` | NumPy compressed archive |
| `.npy` | NumPy array file |

For other formats like PyTorch (`.pt`, `.pth`), SafeTensors (`.safetensors`), or HDF5 (`.h5`),
use the Python API with the appropriate integration:

```python
from coral import Repository
from coral.integrations.pytorch import PyTorchIntegration
import torch

# Load PyTorch model
model = torch.load("model.pt")

# Convert to Coral weights
weights = PyTorchIntegration.model_to_weights(model)

# Stage and commit
repo = Repository("./my_model")
repo.stage_weights(weights)
repo.commit("Added PyTorch model weights")
```

## Implementation Notes

### Command Pattern

Each command follows this pattern:
1. Parse arguments with argparse
2. Find repository (searches parent directories for `.coral/`)
3. Execute operation on repository
4. Print formatted output

### Error Handling

- Missing repository: Suggests `coral-ml init`
- Invalid references: Shows available branches/commits
- Network errors: Retries with exponential backoff

### Progress Display

Long operations show progress:
```
Pushing weights: 100%|██████████| 50/50 [00:30<00:00, 1.67weights/s]
```

## Dependencies

- `argparse` (stdlib) - Command-line parsing
- `tqdm` - Progress bars
- Internal: `coral.version_control.repository`

## Testing

Related test files:
- `tests/test_cli.py` - CLI command tests
- `tests/test_cli_commands_coverage.py` - CLI coverage tests
