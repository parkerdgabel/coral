# Coral ML Benchmarking Suite

Comprehensive benchmarking system for evaluating Coral ML's weight versioning performance, compression efficiency, and system scalability.

## Quick Start

```bash
# Run standard benchmarks (recommended)
uv run run_benchmarks.py

# Quick sanity check (~1 minute)
uv run run_benchmarks.py --profile quick

# Comprehensive analysis (~20+ minutes) 
uv run run_benchmarks.py --profile comprehensive

# Generate detailed reports and visualizations
uv run run_benchmarks.py --output my_results --verbose
```

## Overview

The Coral ML benchmarking suite provides comprehensive performance analysis across multiple dimensions:

- **Compression Efficiency**: Space savings through deduplication and delta encoding
- **Performance Metrics**: Operation timing and throughput analysis  
- **Clustering Effectiveness**: Repository-wide optimization analysis
- **Scalability Characteristics**: Performance with increasing data sizes
- **Memory Usage**: RAM consumption patterns and optimization
- **Accuracy Validation**: Ensuring lossless reconstruction quality

## Architecture

### Benchmark Profiles

| Profile | Duration | Description | Use Case |
|---------|----------|-------------|----------|
| **Quick** | ~1 min | Basic sanity checks | CI/CD, rapid iteration |
| **Standard** | ~5 min | Typical benchmarks | Development, PRs |
| **Full** | ~10 min | Comprehensive testing | Release validation |
| **Comprehensive** | ~20+ min | Exhaustive analysis | Research, optimization |

### Benchmark Suites

#### 1. Compression Benchmark (`benchmarks/compression.py`)
Evaluates space savings through various scenarios:
- **Similar Models**: Fine-tuning variations (99.9% similarity)
- **Training Checkpoints**: Exact duplicates and incremental saves
- **Transfer Learning**: Architecture reuse (95% similarity)
- **Mixed Workloads**: Realistic ML development patterns

**Key Metrics:**
- Compression ratio (e.g., 2.5x)
- Space saved percentage
- Delta encoding efficiency
- Deduplication hit rate

#### 2. Performance Benchmark (`benchmarks/performance.py`)
Measures operational performance:
- **Storage Operations**: Save, load, commit times
- **Delta Operations**: Encoding, reconstruction speeds
- **Query Performance**: Weight lookup and filtering
- **Batch Operations**: Bulk processing efficiency

**Key Metrics:**
- Operations per second
- Latency percentiles (p50, p95, p99)
- Throughput (MB/s)
- Memory efficiency

#### 3. Clustering Benchmark (`benchmarks/clustering.py`)
Analyzes clustering system effectiveness:
- **Strategy Comparison**: K-means vs hierarchical vs adaptive
- **Scalability**: Performance with repository size
- **Optimization**: Dynamic rebalancing effectiveness
- **Quality Metrics**: Cluster cohesion and separation

**Key Metrics:**
- Clustering time
- Compression improvement
- Cluster quality scores
- Memory overhead

#### 4. Scalability Benchmark (`benchmarks/scalability.py`)
Tests performance at scale:
- **Model Size**: Small (1M) to extra-large (1B+ parameters)
- **Repository Size**: Few models to hundreds of models
- **Concurrent Access**: Multi-user scenarios
- **Hardware Utilization**: CPU, memory, I/O patterns

**Key Metrics:**
- Scaling coefficients
- Resource utilization
- Bottleneck identification
- Performance degradation points

#### 5. Memory Benchmark (`benchmarks/memory.py`)
Monitors memory usage patterns:
- **Peak Usage**: Maximum memory consumption
- **Memory Leaks**: Long-running operation analysis
- **Garbage Collection**: Cleanup effectiveness
- **Lazy Loading**: Memory efficiency of deferred operations

**Key Metrics:**
- Peak memory usage
- Memory growth rate
- GC pressure
- Memory efficiency ratio

#### 6. Accuracy Benchmark (`benchmarks/accuracy.py`)
Validates lossless reconstruction:
- **Bit-Perfect Accuracy**: Exact weight reconstruction
- **Numerical Stability**: Floating-point precision preservation
- **Edge Cases**: Extreme values, NaN, infinity handling
- **Data Integrity**: Corruption detection and prevention

**Key Metrics:**
- Reconstruction accuracy (must be 100%)
- Numerical error bounds
- Data corruption detection
- Edge case handling

## Getting Started

### System Requirements

**Minimum Requirements:**
- Python 3.8+
- 4GB RAM
- 2GB free disk space
- Modern CPU (2+ cores recommended)

**Recommended Configuration:**
- Python 3.9+
- 8GB+ RAM
- SSD storage
- Multi-core CPU (4+ cores)

### Installation

```bash
# Install Coral ML with development dependencies
uv sync --extra dev --extra torch

# Verify installation
uv run python -c "import coral; print('Coral ML ready')"

# Check system requirements
uv run run_benchmarks.py --help
```

### Basic Usage

```bash
# Standard benchmark run
uv run run_benchmarks.py

# Quick development check
uv run run_benchmarks.py --profile quick --verbose

# Full analysis with custom output
uv run run_benchmarks.py --profile full --output benchmark_results

# Run specific benchmark suite
uv run python -m benchmarks.compression

# Legacy compatibility
uv run run_benchmarks.py --legacy  # Also runs benchmark.py
```

## Configuration

### Profile Configuration

Customize benchmark profiles by editing `benchmarks/main.py`:

```python
# Custom profile example
custom_config = {
    'suites': ['compression', 'performance'],
    'iterations': 5,
    'models': ['small', 'medium', 'large'],
    'timeout': 1200
}

orchestrator.run_all(
    profile=BenchmarkProfile.CUSTOM,
    custom_config=custom_config
)
```

### Environment Variables

```bash
# Control parallel execution
export CORAL_BENCHMARK_WORKERS=4

# Set memory limits
export CORAL_BENCHMARK_MEMORY_LIMIT=8192

# Custom temporary directory
export CORAL_BENCHMARK_TEMP_DIR=/tmp/coral_bench

# Log level
export CORAL_BENCHMARK_LOG_LEVEL=DEBUG
```

### Hardware-Specific Tuning

**For Limited Memory Systems (<4GB):**
```bash
# Use quick profile with reduced concurrency
uv run run_benchmarks.py --profile quick
export CORAL_BENCHMARK_WORKERS=1
```

**For High-Performance Systems (16GB+):**
```bash
# Use comprehensive profile with max parallelism
uv run run_benchmarks.py --profile comprehensive
export CORAL_BENCHMARK_WORKERS=8
```

**For SSD Systems:**
```bash
# Enable high-throughput tests
export CORAL_BENCHMARK_IO_INTENSIVE=true
```

## Understanding Results

### Compression Results

```
Compression Performance:
┌─────────────────┬──────────────┬─────────────┐
│ Scenario        │ Compression  │ Space Saved │
├─────────────────┼──────────────┼─────────────┤
│ Similar Models  │ 15.2x        │ 93.4%       │
│ Checkpoints     │ 25.8x        │ 96.1%       │
│ Transfer Learn  │ 4.3x         │ 76.7%       │
│ Mixed Workload  │ 8.7x         │ 88.5%       │
└─────────────────┴──────────────┴─────────────┘
```

**Interpretation:**
- **>10x compression**: Excellent deduplication opportunity
- **5-10x compression**: Good optimization potential
- **<5x compression**: Limited similarity, focus on other optimizations
- **>90% space saved**: Highly effective for storage costs

### Performance Results

```
Operation Performance:
┌─────────────────┬───────────┬─────────────┐
│ Operation       │ Time (ms) │ Throughput  │
├─────────────────┼───────────┼─────────────┤
│ Save Weight     │ 12.3      │ 81.3 MB/s   │
│ Load Weight     │ 8.7       │ 114.9 MB/s  │
│ Delta Encode    │ 45.2      │ 22.1 MB/s   │
│ Reconstruct     │ 23.1      │ 43.3 MB/s   │
└─────────────────┴───────────┴─────────────┘
```

**Interpretation:**
- **<50ms operations**: Suitable for interactive use
- **<500ms operations**: Acceptable for batch processing
- **>1s operations**: May need optimization for large-scale use
- **>50 MB/s throughput**: Good I/O performance

### Clustering Results

```
Clustering Efficiency:
┌─────────────┬──────────┬─────────────┬──────────┐
│ Strategy    │ Clusters │ Compression │ Time (s) │
├─────────────┼──────────┼─────────────┼──────────┤
│ K-means     │ 42       │ 12.3x       │ 3.2      │
│ Hierarchical│ 38       │ 11.8x       │ 8.7      │
│ Adaptive    │ 45       │ 13.1x       │ 4.1      │
└─────────────┴──────────┴─────────────┴──────────┘
```

**Interpretation:**
- **Adaptive strategy**: Usually best balance of quality and speed
- **More clusters**: Higher granularity, potentially better compression
- **<10s clustering**: Acceptable for most repositories
- **>2x compression improvement**: Clustering is worthwhile

## Performance Optimization

### Based on Benchmark Results

#### High Compression Scenarios
```python
# Optimize for maximum space savings
repo_config = {
    'similarity_threshold': 0.995,  # Very high similarity
    'delta_encoding': 'COMPRESSED',  # Best compression
    'clustering_strategy': 'adaptive',
    'enable_gc': True
}
```

#### High Performance Scenarios  
```python
# Optimize for speed over compression
repo_config = {
    'similarity_threshold': 0.98,   # Lower threshold for speed
    'delta_encoding': 'FLOAT32_RAW', # Fastest encoding
    'clustering_strategy': 'kmeans',  # Fastest clustering
    'batch_size': 1000  # Larger batches
}
```

#### Memory Constrained Scenarios
```python
# Optimize for low memory usage
repo_config = {
    'lazy_loading': True,
    'memory_limit_mb': 2048,
    'cleanup_frequency': 'aggressive',
    'streaming_operations': True
}
```

### System-Level Optimizations

**Storage Optimization:**
```bash
# Use faster storage for temp files
export CORAL_BENCHMARK_TEMP_DIR=/dev/shm  # RAM disk

# Enable storage compression
export CORAL_STORAGE_COMPRESSION=lzf
```

**CPU Optimization:**
```bash
# Use all available cores
export CORAL_BENCHMARK_WORKERS=$(nproc)

# Enable CPU-intensive optimizations
export CORAL_ENABLE_SIMD=true
```

**Memory Optimization:**
```bash
# Increase memory limits for large models
export CORAL_BENCHMARK_MEMORY_LIMIT=16384

# Enable memory mapping
export CORAL_USE_MMAP=true
```

## Troubleshooting

### Common Issues

#### "Out of Memory" Errors

**Symptoms:**
- Process killed during large model benchmarks
- Memory usage grows continuously

**Solutions:**
```bash
# Reduce memory pressure
uv run run_benchmarks.py --profile quick
export CORAL_BENCHMARK_MEMORY_LIMIT=4096

# Use streaming operations
export CORAL_ENABLE_STREAMING=true

# Increase swap space (Linux/macOS)
sudo swapon -s  # Check current swap
```

#### Slow Performance

**Symptoms:**
- Benchmarks taking much longer than estimated
- High CPU usage with little progress

**Solutions:**
```bash
# Reduce parallelism
export CORAL_BENCHMARK_WORKERS=1

# Use faster profile for testing
uv run run_benchmarks.py --profile quick

# Check disk I/O with faster storage
export CORAL_BENCHMARK_TEMP_DIR=/tmp
```

#### Benchmark Failures

**Symptoms:**
- Suites failing with exceptions
- Inconsistent results across runs

**Solutions:**
```bash
# Run with verbose logging
uv run run_benchmarks.py --verbose

# Test individual suites
uv run python -m benchmarks.compression --verbose

# Check system requirements
uv run python -c "
from benchmarks.main import BenchmarkOrchestrator
print(BenchmarkOrchestrator().validate_system_requirements())
"
```

#### Accuracy Issues

**Symptoms:**
- Reconstruction accuracy < 100%
- Numerical precision warnings

**Investigation:**
```bash
# Run accuracy-focused benchmarks
uv run python -m benchmarks.accuracy --strict

# Check floating-point settings
uv run python -c "
import numpy as np
print('Float precision:', np.finfo(np.float32))
print('Default dtype:', np.array([1.0]).dtype)
"
```

### Debug Mode

```bash
# Enable comprehensive debugging
export CORAL_DEBUG=1
export CORAL_BENCHMARK_LOG_LEVEL=DEBUG
uv run run_benchmarks.py --verbose --profile quick 2>&1 | tee debug.log
```

### Performance Profiling

```bash
# Profile specific operations
uv run python -m cProfile -o benchmark.prof run_benchmarks.py --profile quick

# Analyze profile
uv run python -c "
import pstats
stats = pstats.Stats('benchmark.prof')
stats.sort_stats('cumulative').print_stats(20)
"

# Memory profiling (requires memory_profiler)
uv run mprof run run_benchmarks.py --profile quick
uv run mprof plot
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Benchmark Tests
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install uv
        uv sync --extra dev --extra torch
    
    - name: Run quick benchmarks
      run: uv run run_benchmarks.py --profile quick --no-visualize
    
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: benchmark_results_*/
```

### Performance Regression Detection

```bash
# Baseline benchmarks (run on main branch)
uv run run_benchmarks.py --profile standard --output baseline

# Feature benchmarks (run on feature branch)  
uv run run_benchmarks.py --profile standard --output feature

# Compare results
uv run python -c "
from benchmarks.analysis import BenchmarkAnalyzer
analyzer = BenchmarkAnalyzer()
comparison = analyzer.compare_results('baseline', 'feature')
print('Performance regression:', comparison['regression_detected'])
"
```

## Advanced Usage

### Custom Benchmark Development

Create custom benchmarks by extending base classes:

```python
from benchmarks.base import BenchmarkSuite

class CustomBenchmark(BenchmarkSuite):
    def __init__(self):
        super().__init__("custom")
    
    def run_scenario(self, name: str, config: dict) -> dict:
        # Implement custom benchmark logic
        return {"custom_metric": 42.0}
    
    def get_scenarios(self) -> list:
        return ["scenario1", "scenario2"]

# Register and run
orchestrator = BenchmarkOrchestrator()
orchestrator.suites['custom'] = CustomBenchmark()
results = orchestrator.run_single_suite('custom')
```

### Automated Performance Analysis

```python
from benchmarks.analysis import BenchmarkAnalyzer

# Load results
analyzer = BenchmarkAnalyzer()
results = analyzer.load_results("benchmark_results_20240101_120000")

# Automated analysis
insights = analyzer.extract_insights(results)
recommendations = analyzer.generate_recommendations(results)

print("Key insights:", insights)
print("Recommendations:", recommendations)
```

### Integration with Monitoring

```python
# Send metrics to monitoring system
from benchmarks.monitoring import MetricsCollector

collector = MetricsCollector(
    backend="prometheus",  # or "grafana", "datadog"
    endpoint="http://localhost:9090"
)

# Run benchmarks with monitoring
results = orchestrator.run_all(BenchmarkProfile.STANDARD)
collector.send_metrics(results)
```

## Best Practices

### Regular Benchmarking

1. **Commit Hooks**: Run quick benchmarks on every commit
2. **PR Validation**: Standard benchmarks for pull requests  
3. **Release Testing**: Full benchmarks before releases
4. **Performance Monitoring**: Weekly comprehensive benchmarks

### Result Analysis

1. **Trend Analysis**: Track metrics over time
2. **Regression Detection**: Compare against baselines
3. **Optimization Opportunities**: Identify bottlenecks
4. **Hardware Scaling**: Test on different configurations

### Documentation

1. **Benchmark Results**: Document significant changes
2. **Performance Characteristics**: Update system requirements
3. **Optimization Guide**: Share successful optimizations
4. **Known Issues**: Document limitations and workarounds

## Contributing

### Adding New Benchmarks

1. Create benchmark class extending `BenchmarkSuite`
2. Implement required methods (`run_scenario`, `get_scenarios`)
3. Add comprehensive docstrings and type hints
4. Include unit tests for benchmark logic
5. Update orchestrator registration
6. Document new metrics and interpretation

### Improving Existing Benchmarks

1. Maintain backward compatibility in result format
2. Add new scenarios without removing existing ones
3. Improve accuracy and reliability
4. Optimize benchmark performance itself
5. Enhance error handling and reporting

## Support

### Getting Help

- **Documentation**: Check this guide and API docs
- **Issues**: Create GitHub issues for bugs/features
- **Discussions**: Use GitHub discussions for questions
- **Examples**: See `examples/` directory for usage patterns

### Reporting Performance Issues

When reporting performance issues, include:

1. **System Information**: Hardware, OS, Python version
2. **Benchmark Results**: Full output from affected benchmarks  
3. **Reproduction Steps**: Exact commands and configurations
4. **Expected vs Actual**: What performance you expected
5. **Profiling Data**: If available, include profiling results

## References

- **Coral ML Documentation**: Core system documentation
- **HDF5 Performance**: [HDF5 optimization guide](https://docs.h5py.org/en/stable/high/optimization.html)
- **NumPy Performance**: [NumPy performance tips](https://numpy.org/doc/stable/user/c-info.how-to-extend.html)
- **Python Profiling**: [Python performance profiling](https://docs.python.org/3/library/profile.html)