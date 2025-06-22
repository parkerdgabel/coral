# Clustering Migration Guide

This guide explains how to migrate existing Coral repositories to use the clustering system for improved storage efficiency.

## Overview

The clustering migration feature allows you to convert an existing repository to use clustering-based deduplication. This can significantly reduce storage requirements for repositories with many similar weights (e.g., training checkpoints, fine-tuned models).

## Benefits

- **Space Savings**: Typically 50-70% reduction in storage for similar weights
- **Backwards Compatible**: Supports mixed mode where some weights are clustered and others are not
- **Non-Destructive**: Original weight data is preserved through lossless delta encoding
- **Incremental**: New weights added after migration can also be clustered

## Usage

### Basic Migration

To migrate an existing repository to use clustering:

```bash
coral-ml cluster migrate
```

This will:
1. Analyze all weights in the repository
2. Create clusters using the adaptive strategy
3. Convert weights to use delta encoding from cluster centroids
4. Enable clustering for future commits

### Migration Options

#### Strategy Selection

Choose the clustering algorithm:

```bash
coral-ml cluster migrate --strategy kmeans
```

Available strategies:
- `adaptive` (default): Automatically selects best strategy
- `kmeans`: K-means clustering
- `hierarchical`: Hierarchical clustering
- `dbscan`: Density-based clustering

#### Similarity Threshold

Control how similar weights must be to cluster together:

```bash
coral-ml cluster migrate --threshold 0.95
```

Lower thresholds create more clusters but better preserve weight differences.

#### Backup Creation

Create a backup before migration:

```bash
coral-ml cluster migrate --backup
```

The backup will be stored as `.coral_backup_<timestamp>` in the repository directory.

#### Dry Run

Preview what would happen without making changes:

```bash
coral-ml cluster migrate --dry-run
```

#### Batch Processing

Control memory usage for large repositories:

```bash
coral-ml cluster migrate --batch-size 1000
```

#### Force Re-migration

Re-run migration on an already clustered repository:

```bash
coral-ml cluster migrate --force
```

## Example Workflow

### 1. Analyze Repository

First, check clustering potential:

```bash
coral-ml cluster analyze

# Output:
# Repository Clustering Analysis:
#   Total weights:        1,250
#   Unique weights:       125
#   Potential clusters:   42
#   Estimated reduction:  68.5%
#   Clustering quality:   0.921
```

### 2. Create Backup (Optional)

For safety, create a backup:

```bash
coral-ml cluster migrate --backup --dry-run

# Output:
# Migration Analysis:
#   Total weights:        1,250
#   Potential clusters:   42
#   Estimated reduction:  68.5%
#   
# Backup would require 125.3 MB
# No changes made (dry run)
```

### 3. Perform Migration

Run the actual migration:

```bash
coral-ml cluster migrate --backup --threshold 0.98

# Output:
# Creating backup...
# Backup created at: .coral_backup_20241115_143022
# 
# Migrating repository to use clustering...
# Strategy: adaptive
# Threshold: 0.98
# Batch size: 1000
# Progress: 1250/1250 (100.0%)
# 
# Migration complete:
#   Weights processed:   1,250
#   Clusters created:    42
#   Space saved:         85.7 MB
#   Reduction:           68.5%
#   Time elapsed:        3.2s
# 
# ✓ Migration validation passed
```

### 4. Verify Results

Check the clustering status:

```bash
coral-ml cluster status

# Output:
# Clustering Status:
#   Strategy:          adaptive
#   Clusters:          42
#   Clustered weights: 1,250
#   Space saved:       85.7 MB
#   Reduction:         68.5%
#   Last updated:      2024-11-15 14:30:25
```

## Mixed Mode Operation

After migration, the repository operates in "mixed mode":

- **Existing weights** are accessed through cluster reconstruction
- **New weights** can be added normally and will be evaluated for clustering
- Both clustered and non-clustered weights work transparently

Example:

```python
# Before migration
repo = Repository(".")
weights = repo.get_all_weights()  # Loads from regular storage

# After migration  
repo = Repository(".")
weights = repo.get_all_weights()  # Automatically reconstructs from clusters

# Add new weights - works the same
new_weight = WeightTensor(...)
repo.stage_weights({"new_weight": new_weight})
repo.commit("Added after migration")  # Can be clustered if similar to existing
```

## Performance Considerations

- **Migration Time**: Scales with repository size, typically 1-5 seconds per 1000 weights
- **Memory Usage**: Controlled by `--batch-size` parameter
- **Read Performance**: Clustered weights have slight overhead for reconstruction
- **Write Performance**: New commits may take longer due to clustering analysis

## Troubleshooting

### Migration Fails

If migration fails:
1. Check the error message for specific issues
2. If you created a backup, you can restore it
3. Try with a smaller batch size: `--batch-size 100`
4. Use `--dry-run` to preview without changes

### Validation Errors

If validation fails after migration:
```bash
coral-ml cluster validate --fix
```

### Performance Issues

For large repositories:
1. Use smaller batch sizes
2. Consider migrating during off-peak hours
3. Monitor disk space during migration

## Best Practices

1. **Always backup** important repositories before migration
2. **Test with dry-run** first to understand the impact
3. **Choose appropriate threshold**: 
   - 0.98+ for training checkpoints
   - 0.95+ for fine-tuned models
   - 0.90+ for different model variants
4. **Monitor results** with `coral-ml cluster status`
5. **Optimize periodically** with `coral-ml cluster optimize`