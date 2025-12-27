# Chapter 7: CLI Interface

The Coral command-line interface provides a git-like experience for managing neural network weights. This chapter covers all CLI commands, their usage patterns, and practical workflows for model versioning.

## CLI Overview

### Design Philosophy

Coral's CLI follows git's design principles:

- **Familiar commands**: `init`, `add`, `commit`, `branch`, `merge`, `push`, `pull`
- **Intuitive workflows**: Stage changes, commit snapshots, manage branches
- **Human-readable output**: Clear status messages and progress indicators
- **Composable operations**: Commands can be chained and scripted

### Installation and Setup

After installing Coral, the `coral-ml` command becomes available:

```bash
# Install Coral
pip install coral-ml

# Verify installation
coral-ml --help

# Initialize a new repository
coral-ml init my-model
cd my-model
```

### Getting Help

Every command supports `--help` for detailed usage:

```bash
# General help
coral-ml --help

# Command-specific help
coral-ml commit --help
coral-ml experiment --help
coral-ml publish --help
```

### Command Structure

Commands follow a consistent pattern:

```bash
coral-ml <command> [subcommand] [options] [arguments]

# Examples:
coral-ml init                          # Simple command
coral-ml commit -m "message"           # Command with options
coral-ml experiment start "exp-1"      # Command with subcommand
coral-ml remote add origin s3://...    # Nested subcommand
```

## Repository Commands

### coral-ml init

Initialize a new Coral repository.

**Syntax:**
```bash
coral-ml init [path]
```

**Options:**
- `path`: Repository path (default: current directory)

**Example:**
```bash
# Initialize in current directory
coral-ml init

# Initialize in specific directory
coral-ml init ./my-model

# Output:
# Initialized empty Coral repository in /path/to/my-model/.coral
```

**What it creates:**
- `.coral/` directory with repository structure
- `weights.h5` storage file
- `staging/` directory for uncommitted changes
- `branches/` directory for branch management
- Default `main` branch

### coral-ml status

Show the current repository status.

**Syntax:**
```bash
coral-ml status
```

**Example output:**
```bash
On branch main

Changes to be committed:
  new weight: model.layer1.weight (a7c405c2)
  new weight: model.layer1.bias (f40de49b)
  new weight: model.layer2.weight (218ad6c8)
```

**Displays:**
- Current branch name
- Staged weights ready for commit
- Weight names and hash prefixes

### coral-ml stats

Show repository statistics and storage savings.

**Syntax:**
```bash
coral-ml stats [--json]
```

**Options:**
- `--json`: Output in JSON format

**Example output:**
```bash
============================================================
Coral Repository Statistics
============================================================

Repository: /home/user/my-model
Branch: main
Commits: 15

Weight Storage:
  Total weights stored: 450
  Unique weights: 125 (27.8%)
  Delta-encoded: 89 (71.2%)
  Duplicates eliminated: 325

Storage Savings:
  Raw size (uncompressed): 2.4 GB
  Actual size on disk: 487.3 MB
  Space saved: 1.9 GB
  Compression ratio: 4.9x
  Savings: 79.7%

Comparison with alternatives:
  If using git-lfs: ~2.4 GB
  If using naive storage: ~2.4 GB
  Coral saves you: 1.9 GB

Delta Encoding Breakdown:
  Average delta size: 12.3% of original
  Best compression: 2.1%
```

**JSON output:**
```bash
coral-ml stats --json

# Output:
{
  "total_commits": 15,
  "total_weights": 450,
  "unique_weights": 125,
  "duplicate_weights": 325,
  "delta_weights": 89,
  "raw_size": 2573741824,
  "actual_size": 511098368,
  "bytes_saved": 2062643456,
  "compression_ratio": 4.9,
  "savings_pct": 79.7
}
```

## Weight Management Commands

### coral-ml add

Stage weight files for commit.

**Syntax:**
```bash
coral-ml add <files...>
```

**Supported formats:**
- `.npy`: Single NumPy array
- `.npz`: Multiple NumPy arrays (archive)

**Examples:**
```bash
# Add single weight file
coral-ml add layer1.npy
# Staged 1 weight(s)

# Add multiple files
coral-ml add weights/*.npy
# Staged 45 weight(s)

# Add NPZ archive (extracts all arrays)
coral-ml add model_checkpoint.npz
# Staged 128 weight(s)
```

**Common workflow:**
```python
# In your training script
import numpy as np

# Extract PyTorch weights
weights = {name: param.cpu().numpy()
          for name, param in model.named_parameters()}

# Save as NPZ
np.savez("checkpoint_epoch_10.npz", **weights)
```

```bash
# Stage the weights
coral-ml add checkpoint_epoch_10.npz
```

### coral-ml show

Display detailed information about a weight.

**Syntax:**
```bash
coral-ml show <weight> [-c commit]
```

**Options:**
- `-c, --commit`: Specific commit to show weight from (default: HEAD)

**Example:**
```bash
coral-ml show model.encoder.layer.0.weight

# Output:
Weight: model.encoder.layer.0.weight
Shape: (768, 768)
Dtype: float32
Size: 2,359,296 bytes
Hash: a7c405c2f40de49b218ad6c8

Layer type: Linear
Model: bert-base-uncased

Statistics:
  Min: -0.087234
  Max: 0.092145
  Mean: 0.000123
  Std: 0.028456
```

**Show from specific commit:**
```bash
coral-ml show model.encoder.layer.0.weight -c abc123
```

## Commit Commands

### coral-ml commit

Commit staged weights with a message.

**Syntax:**
```bash
coral-ml commit -m <message> [--author name] [--email email] [-t tag]
```

**Options:**
- `-m, --message`: Commit message (required)
- `--author`: Author name
- `--email`: Author email
- `-t, --tag`: Add tags (can be repeated)

**Examples:**
```bash
# Basic commit
coral-ml commit -m "Initial model weights"
# [main a7c405c2] Initial model weights
#  128 weight(s) changed

# Commit with author info
coral-ml commit -m "Fine-tuned on custom dataset" \
  --author "Alice Johnson" \
  --email "alice@example.com"

# Commit with tags
coral-ml commit -m "Best model from grid search" \
  -t "production" \
  -t "v1.0"
```

### coral-ml log

Show commit history.

**Syntax:**
```bash
coral-ml log [-n N] [--oneline]
```

**Options:**
- `-n, --number`: Number of commits to show (default: 10)
- `--oneline`: Compact one-line per commit format

**Example (default format):**
```bash
coral-ml log -n 3

# Output:
commit a7c405c2f40de49b218ad6c83a88753c0342a1fe
Author: Alice Johnson <alice@example.com>
Date:   2025-12-27 10:30:45
Tags:   production, v1.0

    Fine-tuned BERT on SQuAD dataset

commit f40de49b218ad6c83a88753c0342a1fea7c405c2
Author: Bob Smith <bob@example.com>
Date:   2025-12-26 15:22:10

    Added dropout regularization

commit 218ad6c83a88753c0342a1fea7c405c2f40de49b
Author: Alice Johnson <alice@example.com>
Date:   2025-12-25 09:15:33

    Initial BERT base model
```

**Example (oneline format):**
```bash
coral-ml log --oneline

# Output:
a7c405c2 Fine-tuned BERT on SQuAD dataset
f40de49b Added dropout regularization
218ad6c8 Initial BERT base model
3a88753c Updated learning rate schedule
c0342a1f Fixed weight initialization
```

### coral-ml diff

Show differences between commits.

**Syntax:**
```bash
coral-ml diff <from_ref> [to_ref]
```

**Arguments:**
- `from_ref`: Source commit/branch/tag
- `to_ref`: Target commit/branch/tag (default: HEAD)

**Example:**
```bash
coral-ml diff main feature-branch

# Output:
Comparing main -> feature-branch
  Added:    12 weight(s)
  Removed:  3 weight(s)
  Modified: 8 weight(s)

Added weights:
  + model.new_layer.weight
  + model.new_layer.bias
  + model.attention.query.weight
  ...

Removed weights:
  - model.old_layer.weight
  - model.old_layer.bias
  - model.deprecated.weight

Modified weights:
  ~ model.encoder.layer.0.weight
    a7c405c2 -> f40de49b
  ~ model.encoder.layer.0.bias
    218ad6c8 -> 3a88753c
```

### coral-ml compare

Compare weights between commits with detailed statistics.

**Syntax:**
```bash
coral-ml compare <ref1> [ref2] [-v]
```

**Options:**
- `ref1`: First reference (commit/branch/tag)
- `ref2`: Second reference (default: HEAD)
- `-v, --verbose`: Show per-layer details

**Example:**
```bash
coral-ml compare v1.0 v2.0

# Output:
Comparing v1.0 vs v2.0

Summary:
  Total parameters: 125M
  Changed parameters: 8.2M (6.6%)
  Average change magnitude: 0.0023
  Max change magnitude: 0.145

Layer-wise comparison:
  encoder.layer.0: 2.1% changed (avg: 0.0019)
  encoder.layer.1: 3.4% changed (avg: 0.0031)
  decoder.layer.0: 12.5% changed (avg: 0.0087)
```

**Verbose output:**
```bash
coral-ml compare v1.0 v2.0 -v

# Shows per-layer statistics, distributions, and change patterns
```

### coral-ml tag

Create a named version tag.

**Syntax:**
```bash
coral-ml tag <name> [-d description] [-c commit] [--metric key=value]
```

**Options:**
- `name`: Tag name (required)
- `-d, --description`: Tag description
- `-c, --commit`: Commit to tag (default: HEAD)
- `--metric`: Add metrics (can be repeated)

**Examples:**
```bash
# Simple tag
coral-ml tag v1.0.0

# Tag with description
coral-ml tag v1.0.0 -d "Production release with SQuAD fine-tuning"

# Tag specific commit
coral-ml tag v0.9.0 -c f40de49b

# Tag with metrics
coral-ml tag best-squad-model \
  --metric accuracy=0.912 \
  --metric f1=0.895 \
  --metric latency_ms=42.3 \
  -d "Best performing model on SQuAD v2.0"

# Output:
Tagged version 'best-squad-model' (218ad6c83a88753c)
```

## Branch Commands

### coral-ml branch

Manage branches.

**Syntax:**
```bash
coral-ml branch [name]           # Create branch
coral-ml branch -l               # List branches
coral-ml branch -d <name>        # Delete branch
```

**Options:**
- `name`: Branch name to create
- `-l, --list`: List all branches
- `-d, --delete`: Delete a branch

**Examples:**
```bash
# List branches
coral-ml branch -l
# Output:
* main
  experiment-1
  feature-dropout

# Create new branch
coral-ml branch experiment-2
# Created branch experiment-2

# Delete branch
coral-ml branch -d old-experiment
# Deleted branch old-experiment
```

### coral-ml checkout

Switch to a branch or commit.

**Syntax:**
```bash
coral-ml checkout <target>
```

**Arguments:**
- `target`: Branch name, tag, or commit hash

**Examples:**
```bash
# Switch to branch
coral-ml checkout experiment-1
# Switched to 'experiment-1'

# Checkout specific commit
coral-ml checkout a7c405c2
# Switched to 'a7c405c2'

# Checkout tag
coral-ml checkout v1.0.0
# Switched to 'v1.0.0'
```

### coral-ml merge

Merge branches.

**Syntax:**
```bash
coral-ml merge <branch> [-m message]
```

**Options:**
- `branch`: Branch to merge (required)
- `-m, --message`: Custom merge message

**Examples:**
```bash
# Merge feature branch into current branch
coral-ml checkout main
coral-ml merge experiment-1

# Output:
Merged experiment-1 into main
[main f40de49b] Merge experiment-1 into main

# Merge with custom message
coral-ml merge feature-dropout -m "Integrate dropout improvements"
```

## Storage Commands

### coral-ml gc

Garbage collect unreferenced weights.

**Syntax:**
```bash
coral-ml gc
```

**Example:**
```bash
coral-ml gc

# Output:
Garbage collection complete:
  Cleaned: 47 weight(s)
  Remaining: 215 weight(s)
```

**When to use:**
- After deleting branches with unique weights
- After force-pushing to remotes
- Periodically to reclaim disk space
- After experimenting with many temporary commits

**Storage optimization tips:**
1. Run `gc` after major branch cleanups
2. Use `coral-ml stats` to verify space savings
3. Garbage collection is safe - only removes unreferenced weights
4. Delta-encoded weights are cleaned if references are removed

## Remote Commands

### coral-ml remote add/list/show/remove

Manage remote storage locations.

**Syntax:**
```bash
coral-ml remote add <name> <url>      # Add remote
coral-ml remote list                  # List remotes
coral-ml remote show <name>           # Show remote details
coral-ml remote remove <name>         # Remove remote
```

**Supported URLs:**
- `s3://bucket/path` - AWS S3
- `minio://host/bucket` - MinIO
- `file:///path` - Local filesystem

**Examples:**
```bash
# Add S3 remote
coral-ml remote add origin s3://my-bucket/models/bert

# Add MinIO remote
coral-ml remote add backup minio://minio.example.com/models

# Add local filesystem remote
coral-ml remote add local file:///mnt/storage/coral-weights

# List all remotes
coral-ml remote list
# Output:
origin  s3://my-bucket/models/bert
backup  minio://minio.example.com/models
local   file:///mnt/storage/coral-weights

# Show remote details
coral-ml remote show origin
# Output:
Remote: origin
  URL: s3://my-bucket/models/bert
  Backend: s3
  Endpoint: https://s3.amazonaws.com

# Remove remote
coral-ml remote remove backup
# Removed remote 'backup'
```

### coral-ml push

Push weights to remote storage.

**Syntax:**
```bash
coral-ml push [remote] [--all] [-f]
```

**Options:**
- `remote`: Remote name (default: origin)
- `--all`: Push all weights (not just current branch)
- `-f, --force`: Force push (overwrite remote)

**Examples:**
```bash
# Push to default remote (origin)
coral-ml push

# Output:
Pushing to origin (s3://my-bucket/models/bert)...
Pushing: 100%|████████████| 45/45 [00:12<00:00, 3.75weights/s]

Push complete:
  Weights pushed: 45
  Bytes transferred: 487.3 MB
  Skipped (already exists): 83

# Push to specific remote
coral-ml push backup

# Force push (overwrite)
coral-ml push origin --force
```

### coral-ml pull

Pull weights from remote storage.

**Syntax:**
```bash
coral-ml pull [remote] [--all] [-f]
```

**Options:**
- `remote`: Remote name (default: origin)
- `--all`: Pull all weights (not just current branch)
- `-f, --force`: Force pull (overwrite local)

**Examples:**
```bash
# Pull from default remote
coral-ml pull

# Output:
Pulling from origin (s3://my-bucket/models/bert)...
Pulling: 100%|████████████| 23/23 [00:08<00:00, 2.88weights/s]

Pull complete:
  Weights pulled: 23
  Bytes transferred: 198.7 MB
  Skipped (already exists): 105

# Pull from specific remote
coral-ml pull backup

# Force pull
coral-ml pull origin --force
```

### coral-ml clone

Clone a remote repository.

**Syntax:**
```bash
coral-ml clone <url> [path]
```

**Arguments:**
- `url`: Remote URL to clone from
- `path`: Local path (default: derived from URL)

**Examples:**
```bash
# Clone from S3
coral-ml clone s3://my-bucket/models/bert

# Output:
Cloning s3://my-bucket/models/bert into bert...
Clone complete:
  Weights: 128
  Bytes: 1,024,567,890

# Clone to specific path
coral-ml clone s3://my-bucket/models/gpt2 ./my-gpt2-model
```

### coral-ml sync

Bidirectional synchronization with remote.

**Syntax:**
```bash
coral-ml sync [remote]
```

**Arguments:**
- `remote`: Remote name (default: origin)

**Example:**
```bash
coral-ml sync origin

# Output:
Syncing with origin (s3://my-bucket/models/bert)...
Syncing: 100%|████████████| 68/68 [00:15<00:00, 4.53weights/s]

Sync complete:
  Weights pushed: 12
  Weights pulled: 8
  Bytes transferred: 287.4 MB
```

### coral-ml sync-status

Show synchronization status with remote.

**Syntax:**
```bash
coral-ml sync-status [remote] [--json]
```

**Options:**
- `remote`: Remote name (default: origin)
- `--json`: Output in JSON format

**Example:**
```bash
coral-ml sync-status origin

# Output:
Sync status with origin:
  Local weights:  145
  Remote weights: 138

  Ahead by: 12 weight(s) (need push)
  Behind by: 5 weight(s) (need pull)

  Run 'coral sync' to synchronize
```

**JSON output:**
```bash
coral-ml sync-status origin --json

# Output:
{
  "total_local": 145,
  "total_remote": 138,
  "needs_push": 12,
  "needs_pull": 5,
  "is_synced": false
}
```

## Experiment Commands

### coral-ml experiment start

Start a new experiment.

**Syntax:**
```bash
coral-ml experiment start <name> [-d desc] [-p key=value] [-t tag]
```

**Options:**
- `name`: Experiment name (required)
- `-d, --description`: Experiment description
- `-p, --param`: Parameters (format: key=value, can be repeated)
- `-t, --tag`: Tags (can be repeated)

**Examples:**
```bash
# Simple experiment
coral-ml experiment start "bert-finetuning-v1"
# Started experiment: bert-finetuning-v1
#   ID: a7c405c2f40de49b

# Experiment with parameters
coral-ml experiment start "bert-squad-lr-sweep" \
  -d "Learning rate sweep on SQuAD v2.0" \
  -p learning_rate=0.0001 \
  -p batch_size=32 \
  -p epochs=10 \
  -t "squad" \
  -t "lr-sweep"
```

### coral-ml experiment log

Log a metric value.

**Syntax:**
```bash
coral-ml experiment log <metric> <value> [-s step]
```

**Options:**
- `metric`: Metric name (required)
- `value`: Metric value (required)
- `-s, --step`: Training step/epoch

**Examples:**
```bash
# Log simple metric
coral-ml experiment log loss 0.342
# Logged loss=0.342

# Log with step
coral-ml experiment log accuracy 0.891 -s 1000
# Logged accuracy=0.891
#   Step: 1000

# Common usage in training loop:
for epoch in range(10):
    train_loss = train_epoch()
    val_acc = validate()

    coral-ml experiment log train_loss $train_loss -s $epoch
    coral-ml experiment log val_accuracy $val_acc -s $epoch
```

### coral-ml experiment end

End the current experiment.

**Syntax:**
```bash
coral-ml experiment end [--status] [-c commit]
```

**Options:**
- `--status`: Final status (completed/failed/cancelled, default: completed)
- `-c, --commit`: Associate with commit hash

**Examples:**
```bash
# End successfully
coral-ml experiment end
# Ended experiment: bert-finetuning-v1
#   Status: completed
#   Duration: 3247.8s

# End with commit association
coral-ml experiment end -c a7c405c2
# Ended experiment: bert-squad-lr-sweep
#   Status: completed
#   Duration: 5432.1s

# Mark as failed
coral-ml experiment end --status failed

# Mark as cancelled
coral-ml experiment end --status cancelled
```

### coral-ml experiment list

List experiments.

**Syntax:**
```bash
coral-ml experiment list [--status] [-n N] [--json]
```

**Options:**
- `--status`: Filter by status (pending/running/completed/failed/cancelled)
- `-n, --number`: Limit (default: 20)
- `--json`: Output in JSON format

**Example:**
```bash
coral-ml experiment list -n 5

# Output:
ID                 Name                      Status       Created
---------------------------------------------------------------------------
a7c405c2f40de49b   bert-squad-lr-sweep       completed    2025-12-27 10:30
f40de49b218ad6c8   gpt2-pretraining          running      2025-12-27 09:15
218ad6c83a88753c   resnet-cifar10            completed    2025-12-26 15:20
3a88753c0342a1fe   bert-finetuning-v1        failed       2025-12-26 08:45
c0342a1fea7c405c   vit-imagenet              cancelled    2025-12-25 14:30

# Filter by status
coral-ml experiment list --status completed
```

### coral-ml experiment show

Show detailed experiment information.

**Syntax:**
```bash
coral-ml experiment show <id> [--json]
```

**Arguments:**
- `id`: Experiment ID (required)

**Options:**
- `--json`: Output in JSON format

**Example:**
```bash
coral-ml experiment show a7c405c2f40de49b

# Output:
Experiment: bert-squad-lr-sweep
  ID: a7c405c2f40de49b
  Status: completed
  Created: 2025-12-27 08:00:00
  Started: 2025-12-27 08:00:05
  Ended: 2025-12-27 09:34:13
  Duration: 5647.8s
  Branch: main
  Commit: f40de49b218ad6c8

Description: Learning rate sweep on SQuAD v2.0

Parameters:
  learning_rate: 0.0001
  batch_size: 32
  epochs: 10
  warmup_steps: 500

Tags: squad, lr-sweep

Metrics:
  train_loss:
    Latest: 0.234
    Best: 0.189
  val_accuracy:
    Latest: 0.912
    Best: 0.915
  val_f1:
    Latest: 0.895
    Best: 0.897
```

### coral-ml experiment compare

Compare multiple experiments.

**Syntax:**
```bash
coral-ml experiment compare <exp_ids...> [-m metric] [--json]
```

**Arguments:**
- `exp_ids`: Experiment IDs to compare (space-separated)

**Options:**
- `-m, --metric`: Specific metrics to compare (can be repeated)
- `--json`: Output in JSON format

**Example:**
```bash
coral-ml experiment compare a7c405c2 f40de49b 218ad6c8

# Output:
Experiment Comparison
============================================================
Metric              a7c405c2f40d    f40de49b218a    218ad6c83a88
------------------------------------------------------------
train_loss          0.1890          0.2340          0.3120
val_accuracy        0.9150          0.8920          0.8650
val_f1              0.8970          0.8750          0.8420
```

### coral-ml experiment best

Find best experiments by metric.

**Syntax:**
```bash
coral-ml experiment best <metric> [--mode] [-n N] [--json]
```

**Arguments:**
- `metric`: Metric name to optimize (required)

**Options:**
- `--mode`: Optimization mode (min/max, default: min)
- `-n, --number`: Number of results (default: 10)
- `--json`: Output in JSON format

**Examples:**
```bash
# Find experiments with lowest loss
coral-ml experiment best train_loss --mode min -n 5

# Output:
Best Experiments by train_loss (min)
============================================================
Rank   ID              Name                  Value
------------------------------------------------------------
1      a7c405c2f4      bert-squad-lr-sweep   0.1890
2      f40de49b21      gpt2-pretraining      0.2340
3      218ad6c83a      bert-finetuning-v1    0.3120
4      3a88753c03      resnet-cifar10        0.4567
5      c0342a1fea      vit-imagenet          0.5234

# Find experiments with highest accuracy
coral-ml experiment best val_accuracy --mode max -n 3
```

### coral-ml experiment delete

Delete an experiment.

**Syntax:**
```bash
coral-ml experiment delete <id>
```

**Arguments:**
- `id`: Experiment ID to delete (required)

**Example:**
```bash
coral-ml experiment delete a7c405c2f40de49b
# Deleted experiment: a7c405c2f40de49b
```

## Publishing Commands

### coral-ml publish huggingface

Publish model to Hugging Face Hub.

**Syntax:**
```bash
coral-ml publish huggingface <repo_id> [options]
```

**Arguments:**
- `repo_id`: Repository ID (org/name or user/name)

**Options:**
- `-c, --commit`: Commit reference (default: HEAD)
- `--private`: Create private repository
- `-d, --description`: Model description
- `--base-model`: Base model name
- `--metric`: Metrics (format: key=value, can be repeated)
- `-t, --tag`: Tags (can be repeated)

**Examples:**
```bash
# Simple publish
coral-ml publish huggingface my-org/bert-squad

# Output:
Publishing to Hugging Face Hub: my-org/bert-squad...
✓ Published successfully!
  URL: https://huggingface.co/my-org/bert-squad

# Publish with full metadata
coral-ml publish huggingface my-org/bert-squad-v2 \
  -c a7c405c2 \
  --private \
  -d "BERT fine-tuned on SQuAD v2.0 with custom preprocessing" \
  --base-model "bert-base-uncased" \
  --metric accuracy=0.912 \
  --metric f1=0.895 \
  -t "question-answering" \
  -t "squad"
```

**Authentication:**
```bash
# Login to Hugging Face first
huggingface-cli login

# Or set token environment variable
export HF_TOKEN="hf_..."
```

### coral-ml publish mlflow

Publish model to MLflow Model Registry.

**Syntax:**
```bash
coral-ml publish mlflow <model_name> [options]
```

**Arguments:**
- `model_name`: Model name in registry

**Options:**
- `-c, --commit`: Commit reference (default: HEAD)
- `--tracking-uri`: MLflow tracking URI
- `--experiment`: MLflow experiment name
- `-d, --description`: Model description
- `--metric`: Metrics (format: key=value, can be repeated)

**Examples:**
```bash
# Publish to local MLflow
coral-ml publish mlflow bert-squad

# Output:
Publishing to MLflow: bert-squad...
✓ Published successfully!
  Version: 1
  URL: http://localhost:5000/#/models/bert-squad

# Publish to remote MLflow server
coral-ml publish mlflow bert-squad-production \
  --tracking-uri "http://mlflow.example.com:5000" \
  --experiment "squad-training" \
  -d "Production BERT model for SQuAD QA" \
  --metric accuracy=0.912 \
  --metric f1=0.895

# Output:
✓ Published successfully!
  Version: 3
  URL: http://mlflow.example.com:5000/#/models/bert-squad-production
```

### coral-ml publish local

Export model to local directory.

**Syntax:**
```bash
coral-ml publish local <output_path> [options]
```

**Arguments:**
- `output_path`: Output directory path

**Options:**
- `-c, --commit`: Commit reference (default: HEAD)
- `--format`: Output format (safetensors/npz/pt, default: safetensors)
- `--no-metadata`: Skip metadata files

**Examples:**
```bash
# Export as safetensors (recommended)
coral-ml publish local ./exports/bert-v1

# Output:
Exporting to: ./exports/bert-v1...
✓ Exported successfully!
  Path: ./exports/bert-v1
  Format: safetensors

# Files created:
# - model.safetensors
# - coral_metadata.json

# Export as PyTorch
coral-ml publish local ./exports/bert-v1-torch --format pt

# Export as NumPy (no metadata)
coral-ml publish local ./exports/bert-v1-numpy \
  --format npz \
  --no-metadata

# Export specific commit
coral-ml publish local ./exports/bert-v0.9 -c f40de49b
```

### coral-ml publish history

Show publishing history.

**Syntax:**
```bash
coral-ml publish history
```

**Example:**
```bash
coral-ml publish history

# Output:
Publish History
================================================================================

huggingface: my-org/bert-squad-v2
  Status: Success
  Date: 2025-12-27 10:45:23
  Version: None
  URL: https://huggingface.co/my-org/bert-squad-v2

mlflow: bert-squad-production
  Status: Success
  Date: 2025-12-27 09:30:15
  Version: 3
  URL: http://mlflow.example.com:5000/#/models/bert-squad-production

local: ./exports/bert-v1
  Status: Success
  Date: 2025-12-27 08:15:47
  URL: file:///home/user/exports/bert-v1

huggingface: my-org/bert-base
  Status: Failed
  Date: 2025-12-26 16:22:10
  Error: Authentication failed
```

## Practical Workflows

### Basic Workflow

Complete workflow for model versioning:

```bash
# 1. Initialize repository
mkdir my-bert-model && cd my-bert-model
coral-ml init

# 2. Train and save weights
python train.py  # Saves weights.npz

# 3. Stage weights
coral-ml add weights.npz

# 4. Check status
coral-ml status
# On branch main
# Changes to be committed:
#   new weight: encoder.layer.0.weight (a7c405c2)
#   ...

# 5. Commit
coral-ml commit -m "Initial BERT base model" \
  --author "Alice" \
  --email "alice@example.com"

# 6. Tag important versions
coral-ml tag v1.0.0 -d "First production release"

# 7. View history
coral-ml log --oneline

# 8. Check storage savings
coral-ml stats
```

### Branching Workflow

Parallel experiment development:

```bash
# Start from main branch
coral-ml checkout main

# Create experiment branch
coral-ml branch experiment-dropout
coral-ml checkout experiment-dropout

# Make changes and commit
python train_with_dropout.py
coral-ml add checkpoint.npz
coral-ml commit -m "Added dropout regularization"

# Create another experiment
coral-ml checkout main
coral-ml branch experiment-lr-schedule
coral-ml checkout experiment-lr-schedule

python train_with_schedule.py
coral-ml add checkpoint.npz
coral-ml commit -m "Implemented cosine LR schedule"

# Compare experiments
coral-ml compare experiment-dropout experiment-lr-schedule -v

# Merge best experiment
coral-ml checkout main
coral-ml merge experiment-lr-schedule -m "Integrate LR schedule improvements"

# Clean up
coral-ml branch -d experiment-dropout
coral-ml gc
```

### Experiment Tracking Workflow

Systematic experiment management:

```bash
# Start experiment
coral-ml experiment start "bert-squad-hyperopt" \
  -d "Hyperparameter optimization for SQuAD" \
  -p learning_rate=0.0001 \
  -p batch_size=32 \
  -p warmup_ratio=0.1 \
  -t "squad" \
  -t "hyperopt"

# Train model (with metric logging)
python train.py  # Script calls coral experiment log internally

# Or manually log metrics
for epoch in {1..10}; do
    # ... training code ...
    coral-ml experiment log train_loss 0.234 -s $epoch
    coral-ml experiment log val_f1 0.895 -s $epoch
done

# Commit weights
coral-ml add model_epoch_10.npz
coral-ml commit -m "Best checkpoint from hyperopt"

# End experiment with commit reference
coral-ml experiment end -c $(git rev-parse HEAD)

# Find best experiments
coral-ml experiment best val_f1 --mode max -n 5

# Compare top experiments
coral-ml experiment compare <id1> <id2> <id3>

# Tag and publish best model
coral-ml tag best-squad-model \
  --metric f1=0.912 \
  --metric accuracy=0.895

coral-ml publish huggingface my-org/bert-squad-best \
  -d "Best BERT model from hyperparameter search" \
  --metric f1=0.912
```

### Team Collaboration Workflow

Multi-user development:

```bash
# Team member 1: Initial setup
coral-ml init
coral-ml remote add origin s3://team-bucket/bert-project

# Train initial model
python train_base.py
coral-ml add base_model.npz
coral-ml commit -m "Initial base model" \
  --author "Alice" --email "alice@team.com"

# Push to team storage
coral-ml push origin

# Team member 2: Clone and continue
coral-ml clone s3://team-bucket/bert-project ./bert-project
cd bert-project

# Check status
coral-ml log
coral-ml stats

# Create feature branch
coral-ml branch feature-attention-improvements
coral-ml checkout feature-attention-improvements

# Make improvements
python train_improved.py
coral-ml add improved_model.npz
coral-ml commit -m "Improved attention mechanism" \
  --author "Bob" --email "bob@team.com"

# Push feature branch
coral-ml push origin

# Team member 1: Pull updates and merge
coral-ml pull origin
coral-ml branch -l
# * main
#   feature-attention-improvements

coral-ml checkout main
coral-ml merge feature-attention-improvements

# Sync everything
coral-ml sync origin

# Check sync status
coral-ml sync-status origin
# Status: ✓ Fully synced

# Team leader: Tag and publish release
coral-ml tag v2.0.0 \
  -d "Production release with attention improvements" \
  --metric accuracy=0.925

coral-ml publish huggingface team-org/bert-production \
  --base-model "bert-base-uncased" \
  --metric accuracy=0.925
```

## Quick Reference

### Most Common Commands

```bash
# Repository setup
coral-ml init                          # Initialize repository
coral-ml remote add origin <url>       # Add remote storage

# Daily workflow
coral-ml add weights.npz               # Stage weights
coral-ml commit -m "message"           # Commit changes
coral-ml push                          # Push to remote
coral-ml pull                          # Pull from remote

# Branching
coral-ml branch feature-x              # Create branch
coral-ml checkout feature-x            # Switch to branch
coral-ml merge feature-x               # Merge branch

# Information
coral-ml status                        # Repository status
coral-ml log --oneline                 # Commit history
coral-ml stats                         # Storage statistics

# Experiments
coral-ml experiment start "name"       # Start experiment
coral-ml experiment log metric value   # Log metrics
coral-ml experiment end                # End experiment

# Publishing
coral-ml tag v1.0.0                    # Tag version
coral-ml publish huggingface org/model # Publish to HF
coral-ml publish local ./export        # Export locally
```

### Command Categories

| Category | Commands |
|----------|----------|
| **Repository** | init, status, stats |
| **Weights** | add, show |
| **Commits** | commit, log, diff, compare, tag |
| **Branches** | branch, checkout, merge |
| **Remote** | remote, push, pull, clone, sync, sync-status |
| **Experiments** | experiment start/log/end/list/show/compare/best/delete |
| **Publishing** | publish huggingface/mlflow/local, publish history |
| **Maintenance** | gc |

### Exit Codes

- `0`: Success
- `1`: Error (see error message)

All commands return non-zero exit codes on failure, making them suitable for scripts and CI/CD pipelines.

---

This chapter covered the complete Coral CLI interface. The next chapter will explore training integration with PyTorch and TensorFlow, showing how to automate weight versioning during model training.
