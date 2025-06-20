# Coral Clustering CLI Guide

The Coral CLI provides comprehensive clustering commands for analyzing, creating, and managing weight clusters in your repository.

## Overview

The clustering commands are organized under the `coral-ml cluster` subcommand with the following operations:

### Analysis Commands
- `analyze` - Analyze repository for clustering opportunities
- `status` - Show current clustering status and statistics  
- `report` - Generate detailed clustering report

### Operations Commands
- `create` - Create clusters using specified strategy
- `optimize` - Optimize existing clusters
- `rebalance` - Rebalance cluster assignments

### Management Commands
- `list` - List all clusters in repository
- `info` - Show detailed cluster information
- `export` - Export clustering configuration
- `import` - Import clustering configuration

### Comparison and Validation Commands
- `compare` - Compare clustering between commits
- `validate` - Validate cluster integrity and quality
- `benchmark` - Run clustering performance benchmarks

## Command Reference

### coral-ml cluster analyze

Analyze repository for clustering opportunities.

```bash
coral-ml cluster analyze [options]
```

**Options:**
- `--commit` - Analyze specific commit (default: HEAD)
- `--threshold` - Similarity threshold for analysis (default: 0.98)
- `--format` - Output format: text, json, csv (default: text)

**Example:**
```bash
coral-ml cluster analyze --threshold 0.95 --format json
```

### coral-ml cluster status

Show current clustering status and statistics.

```bash
coral-ml cluster status
```

Displays:
- Whether clustering is enabled
- Current strategy
- Number of clusters
- Space saved
- Cluster health metrics

### coral-ml cluster create

Create clusters using specified strategy.

```bash
coral-ml cluster create [strategy] [options]
```

**Strategies:**
- `kmeans` - K-means clustering
- `hierarchical` - Hierarchical clustering  
- `dbscan` - DBSCAN clustering
- `adaptive` - Adaptive strategy (default)

**Options:**
- `--levels` - Hierarchy levels for multi-level clustering (default: 3)
- `--threshold` - Similarity threshold (default: 0.98)
- `--optimize` - Auto-optimize after clustering
- `--dry-run` - Show what would be done without making changes

**Example:**
```bash
coral-ml cluster create kmeans --levels 2 --threshold 0.95 --optimize
```

### coral-ml cluster optimize

Optimize existing clusters.

```bash
coral-ml cluster optimize [options]
```

**Options:**
- `--aggressive` - Use aggressive optimization strategies
- `--target-reduction` - Target space reduction percentage

**Example:**
```bash
coral-ml cluster optimize --aggressive --target-reduction 0.4
```

### coral-ml cluster list

List all clusters in repository.

```bash
coral-ml cluster list [options]
```

**Options:**
- `--sort-by` - Sort by: id, size, quality, compression (default: id)
- `--limit` - Limit number of clusters shown
- `--format` - Output format: text, json, csv (default: text)

**Example:**
```bash
coral-ml cluster list --sort-by size --limit 10 --format json
```

### coral-ml cluster info

Show detailed information about a specific cluster.

```bash
coral-ml cluster info <cluster_id> [options]
```

**Options:**
- `--show-weights` - Show individual weight information
- `--show-stats` - Show detailed statistics

**Example:**
```bash
coral-ml cluster info cluster_001 --show-stats --show-weights
```

### coral-ml cluster report

Generate detailed clustering report.

```bash
coral-ml cluster report [options]
```

**Options:**
- `--output` - Output file path (default: stdout)
- `--format` - Output format: text, json, csv, html (default: text)
- `--verbose` - Include detailed weight information

**Example:**
```bash
coral-ml cluster report --format html --output report.html --verbose
```

### coral-ml cluster export

Export clustering configuration.

```bash
coral-ml cluster export <output_file> [options]
```

**Options:**
- `--format` - Export format: json, yaml, toml (default: json)
- `--include-weights` - Include weight data in export

**Example:**
```bash
coral-ml cluster export clusters.json --format json --include-weights
```

### coral-ml cluster import

Import clustering configuration.

```bash
coral-ml cluster import <input_file> [options]
```

**Options:**
- `--merge` - Merge with existing clusters instead of replacing
- `--validate` - Validate configuration before importing

**Example:**
```bash
coral-ml cluster import clusters.json --validate --merge
```

### coral-ml cluster compare

Compare clustering between two commits.

```bash
coral-ml cluster compare <commit1> <commit2> [options]
```

**Options:**
- `--show-migrations` - Show weight migrations between clusters
- `--format` - Output format: text, json (default: text)

**Example:**
```bash
coral-ml cluster compare HEAD~1 HEAD --show-migrations
```

### coral-ml cluster validate

Validate cluster integrity and quality.

```bash
coral-ml cluster validate [options]
```

**Options:**
- `--fix` - Attempt to fix validation errors
- `--strict` - Use strict validation rules

**Example:**
```bash
coral-ml cluster validate --fix --strict
```

### coral-ml cluster benchmark

Run clustering performance benchmarks.

```bash
coral-ml cluster benchmark [options]
```

**Options:**
- `--strategies` - Strategies to benchmark (default: all)
- `--iterations` - Number of iterations per strategy (default: 3)
- `--output` - Output results to file

**Example:**
```bash
coral-ml cluster benchmark --strategies kmeans hierarchical --iterations 5 --output benchmark.json
```

### coral-ml cluster rebalance

Rebalance cluster assignments.

```bash
coral-ml cluster rebalance [options]
```

**Options:**
- `--max-cluster-size` - Maximum weights per cluster
- `--min-cluster-size` - Minimum weights per cluster (default: 2)

**Example:**
```bash
coral-ml cluster rebalance --max-cluster-size 10 --min-cluster-size 3
```

## Typical Workflow

1. **Analyze repository for clustering potential:**
   ```bash
   coral-ml cluster analyze
   ```

2. **Create clusters with optimal strategy:**
   ```bash
   coral-ml cluster create adaptive --threshold 0.95
   ```

3. **Check clustering status:**
   ```bash
   coral-ml cluster status
   ```

4. **Generate report:**
   ```bash
   coral-ml cluster report --format html --output clustering_report.html
   ```

5. **Optimize clusters:**
   ```bash
   coral-ml cluster optimize --aggressive
   ```

6. **Validate integrity:**
   ```bash
   coral-ml cluster validate --fix
   ```

## Output Formats

Most commands support multiple output formats:

- **text** - Human-readable format (default)
- **json** - Machine-readable JSON format
- **csv** - Comma-separated values for analysis
- **html** - HTML format for reports (report command only)

## Error Handling

The CLI provides clear error messages and helpful suggestions:

- Repository not initialized
- Clustering not enabled
- Invalid cluster IDs
- Configuration validation errors
- File not found errors

## Performance Considerations

- Use `--dry-run` to preview operations before execution
- Limit operations with `--limit` for large repositories
- Run benchmarks to compare strategy performance
- Use progress indicators for long operations

## Integration with CI/CD

The clustering CLI commands can be integrated into CI/CD pipelines:

```bash
# Analyze and fail if clustering quality is low
quality=$(coral-ml cluster analyze --format json | jq '.clustering_quality')
if (( $(echo "$quality < 0.8" | bc -l) )); then
    echo "Clustering quality too low: $quality"
    exit 1
fi

# Create clusters and optimize
coral-ml cluster create adaptive --optimize

# Generate report
coral-ml cluster report --format json --output cluster_metrics.json
```

## Tips and Best Practices

1. **Start with analysis** - Always analyze before creating clusters
2. **Use appropriate thresholds** - Higher thresholds (0.98+) for similar weights
3. **Monitor cluster health** - Regular validation ensures integrity
4. **Export configurations** - Save successful clustering setups
5. **Benchmark strategies** - Test different approaches for your data
6. **Rebalance periodically** - Maintain optimal cluster sizes
7. **Use verbose output** - For debugging and detailed insights