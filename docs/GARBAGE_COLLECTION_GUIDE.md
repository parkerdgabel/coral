# Garbage Collection Guide for Coral

## Overview

Coral's garbage collection (GC) system is designed to clean up unreferenced data while maintaining the integrity of your model weights, clusters, and delta encodings. The enhanced GC system now fully supports clustered weight storage, ensuring efficient cleanup while protecting active data.

## Key Features

### 1. **Intelligent Reference Tracking**
- Tracks references from commits to weights
- Monitors delta encodings and their reference weights/centroids
- Maintains reference counts for centroids with active deltas
- Protects centroids that are still being used by delta encodings

### 2. **Proper Cleanup Order**
The GC follows a strict order to ensure data integrity:
1. **Deltas** - Clean orphaned deltas first
2. **Weights** - Remove unreferenced weights
3. **Clusters & Centroids** - Clean unused clusters and their centroids

### 3. **Clustering-Aware Cleanup**
- Identifies clusters that are no longer referenced by any weights
- Protects centroids that have active delta encodings referencing them
- Cleans up orphaned assignments when weights are deleted
- Maintains cluster integrity during cleanup

## Usage

### Basic Garbage Collection

```python
from coral.version_control.repository import Repository

# Open repository
repo = Repository("path/to/repo")

# Run basic GC (includes clustering by default)
result = repo.gc()

print(f"Cleaned {result['cleaned_weights']} weights")
print(f"Cleaned {result['cleaned_deltas']} deltas")
print(f"Cleaned {result['cleaned_clusters']} clusters")
print(f"Protected {result['protected_centroids']} centroids with active references")
```

### GC Without Clustering Support

```python
# Run GC without clustering cleanup (faster but less thorough)
result = repo.gc(include_clusters=False)
```

### Understanding GC Results

The GC returns a detailed dictionary with statistics:

```python
{
    "cleaned_weights": 10,        # Unreferenced weights removed
    "cleaned_deltas": 5,          # Orphaned deltas removed
    "cleaned_clusters": 3,        # Unused clusters removed
    "cleaned_centroids": 2,       # Orphaned centroids removed
    "remaining_weights": 50,      # Weights still in use
    "remaining_deltas": 20,       # Active deltas
    "remaining_clusters": 10,     # Active clusters
    "remaining_centroids": 8,     # Active centroids
    "protected_centroids": 2,     # Centroids kept due to active deltas
    "gc_time": 1.234             # Time taken in seconds
}
```

## How It Works

### 1. Reference Discovery Phase

The GC first identifies all referenced data:

```python
# Pseudocode of reference discovery
referenced_weights = set()
for commit in all_commits:
    for weight_hash in commit.weight_hashes:
        referenced_weights.add(weight_hash)

referenced_deltas = set()
for commit in all_commits:
    for delta_hash in commit.delta_weights:
        referenced_deltas.add(delta_hash)
```

### 2. Cluster Reference Tracking

For clustered weights, the GC tracks cluster and centroid references:

```python
# Find clusters used by referenced weights
referenced_clusters = set()
for weight_hash in referenced_weights:
    assignment = get_weight_assignment(weight_hash)
    if assignment:
        referenced_clusters.add(assignment.cluster_id)

# Track centroid reference counts
centroid_ref_count = {}
for delta in all_deltas:
    if delta.reference_hash in centroid_hashes:
        centroid_ref_count[delta.reference_hash] += 1
```

### 3. Cleanup Phase

The cleanup follows a strict order:

#### Step 1: Clean Deltas
```python
# Remove unreferenced deltas
for delta_hash in unreferenced_deltas:
    # Decrement centroid reference if needed
    if delta.references_centroid:
        centroid_ref_count[delta.reference_hash] -= 1
    delete_delta(delta_hash)
```

#### Step 2: Clean Weights
```python
# Remove unreferenced weights
for weight_hash in unreferenced_weights:
    # Update cluster assignments if needed
    assignment = get_weight_assignment(weight_hash)
    if assignment:
        decrement_cluster_references(assignment.cluster_id)
    delete_weight(weight_hash)
```

#### Step 3: Clean Clusters
```python
# Remove unreferenced clusters, but protect centroids with active deltas
for cluster_id in unreferenced_clusters:
    centroid_hash = get_cluster_centroid(cluster_id)
    if centroid_ref_count.get(centroid_hash, 0) > 0:
        # Centroid has active deltas, keep the cluster
        protected_centroids += 1
        continue
    delete_cluster(cluster_id)
```

## Best Practices

### 1. Regular Cleanup
Run GC periodically to prevent storage bloat:

```python
# After major operations
repo.gc()

# In automated scripts
if repo.get_storage_info()['file_size'] > 1_000_000_000:  # 1GB
    repo.gc()
```

### 2. Before Backups
Always run GC before creating backups:

```python
# Clean up before backup
gc_result = repo.gc()
print(f"Freed {gc_result['cleaned_weights']} weights before backup")

# Create backup
backup_path = repo.create_backup()
```

### 3. Monitor Protected Centroids
High numbers of protected centroids may indicate inefficient clustering:

```python
result = repo.gc()
if result['protected_centroids'] > result['remaining_centroids'] * 0.5:
    print("Consider re-clustering - many centroids protected only by deltas")
    repo.optimize_repository_clusters()
```

## Troubleshooting

### Issue: GC Takes Too Long

For large repositories, GC can be time-consuming. Solutions:

1. **Run GC more frequently** to prevent large accumulations
2. **Use incremental GC** (future feature)
3. **Disable clustering cleanup** for faster runs:
   ```python
   repo.gc(include_clusters=False)
   ```

### Issue: Storage Not Freed

If GC doesn't free expected space:

1. **Check for active branches**:
   ```python
   # All branches keep their commits' weights
   branches = repo.branch_manager.list_branches()
   print(f"Active branches: {len(branches)}")
   ```

2. **Look for protected centroids**:
   ```python
   result = repo.gc()
   if result['protected_centroids'] > 0:
       print(f"{result['protected_centroids']} centroids protected by active deltas")
   ```

3. **Verify delta encoding**:
   ```python
   # Deltas keep references to their base weights/centroids
   with HDF5Store(repo.weights_store_path) as store:
       print(f"Active deltas: {len(store.list_deltas())}")
   ```

### Issue: Weights Can't Be Loaded After GC

This indicates a bug in reference tracking. Enable debug logging:

```python
import logging
logging.getLogger("coral.version_control.repository").setLevel(logging.DEBUG)

# Run GC with debug output
result = repo.gc()
```

## Advanced Usage

### Custom Reference Tracking

For specialized workflows, you can extend reference tracking:

```python
class CustomRepository(Repository):
    def _get_additional_references(self):
        """Override to add custom reference tracking."""
        refs = set()
        # Add your custom references
        return refs
    
    def gc(self, include_clusters=True):
        # Add custom references before running GC
        custom_refs = self._get_additional_references()
        # ... integrate with standard GC
        return super().gc(include_clusters)
```

### Monitoring GC Performance

Track GC performance over time:

```python
import json
from datetime import datetime

def run_gc_with_monitoring(repo):
    start_storage = repo.get_storage_info()['file_size']
    
    result = repo.gc()
    
    end_storage = repo.get_storage_info()['file_size']
    
    # Log results
    gc_log = {
        'timestamp': datetime.now().isoformat(),
        'storage_before': start_storage,
        'storage_after': end_storage,
        'space_freed': start_storage - end_storage,
        'gc_time': result['gc_time'],
        **result
    }
    
    with open('gc_history.json', 'a') as f:
        json.dump(gc_log, f)
        f.write('\n')
    
    return result
```

## Future Enhancements

Planned improvements to the GC system:

1. **Incremental GC** - Clean up in smaller chunks
2. **Concurrent GC** - Run cleanup in background
3. **Smart Scheduling** - Automatic GC based on activity
4. **Compression After GC** - Repack storage files
5. **GC Policies** - Configurable retention policies

## Summary

The enhanced garbage collection system in Coral provides:

- **Complete cleanup** of unreferenced weights, deltas, clusters, and centroids
- **Intelligent protection** of centroids with active delta references
- **Proper ordering** to maintain data integrity
- **Detailed reporting** for monitoring and debugging
- **Clustering awareness** for optimal storage efficiency

Use GC regularly to maintain optimal repository performance and storage efficiency.