# Coral: Neural Network Weight Storage and Deduplication

Coral is a Python library for efficient storage and deduplication of neural network weights. It provides content-addressable storage, automatic deduplication, and various compression techniques to minimize storage requirements for large models.

## Features

- **Content-Addressable Storage**: Weights are stored and retrieved using content-based hashes
- **Automatic Deduplication**: Detect and eliminate duplicate and similar weights
- **Compression Support**: Multiple compression techniques including quantization and pruning
- **Flexible Storage Backends**: HDF5-based storage with compression support
- **Framework Integration**: Support for PyTorch and TensorFlow models (coming soon)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/coral.git
cd coral

# Install in development mode
pip install -e .

# Install with ML framework support
pip install -e ".[torch]"  # For PyTorch support
pip install -e ".[tensorflow]"  # For TensorFlow support
```

## Quick Start

```python
import numpy as np
from coral import WeightTensor, Deduplicator, HDF5Store
from coral.core.weight_tensor import WeightMetadata

# Create a weight tensor
weight = WeightTensor(
    data=np.random.randn(256, 256).astype(np.float32),
    metadata=WeightMetadata(
        name="layer1.weight",
        shape=(256, 256),
        dtype=np.float32,
        layer_type="Linear"
    )
)

# Initialize deduplicator
dedup = Deduplicator(similarity_threshold=0.99)

# Add weights and check for duplicates
hash_key = dedup.add_weight(weight)

# Store weights persistently
with HDF5Store("weights.h5") as store:
    store.store(weight)
    
    # Load weight back
    loaded = store.load(hash_key)
```

## Core Components

### WeightTensor

The fundamental data structure representing neural network weights with metadata:

```python
from coral import WeightTensor
from coral.core.weight_tensor import WeightMetadata

# Create weight with metadata
metadata = WeightMetadata(
    name="conv1.weight",
    shape=(64, 3, 3, 3),
    dtype=np.float32,
    layer_type="Conv2d",
    model_name="resnet50"
)

weight = WeightTensor(data=weight_array, metadata=metadata)

# Access properties
print(weight.shape)  # (64, 3, 3, 3)
print(weight.nbytes)  # Number of bytes
print(weight.compute_hash())  # Content hash
```

### Deduplicator

Identifies and tracks duplicate and similar weights:

```python
from coral import Deduplicator

# Initialize with similarity threshold
dedup = Deduplicator(similarity_threshold=0.98)

# Add weights
for weight in weights:
    ref_hash = dedup.add_weight(weight)
    
# Get deduplication report
report = dedup.get_deduplication_report()
print(f"Unique weights: {report['summary']['unique_weights']}")
print(f"Bytes saved: {report['summary']['bytes_saved']}")
```

### Storage Backends

#### HDF5Store

Efficient storage with built-in compression:

```python
from coral import HDF5Store

with HDF5Store("model_weights.h5", compression="gzip") as store:
    # Store weights
    hash_key = store.store(weight)
    
    # Batch operations
    hashes = store.store_batch({"layer1": weight1, "layer2": weight2})
    
    # Get storage statistics
    info = store.get_storage_info()
    print(f"Total weights: {info['total_weights']}")
    print(f"Compression ratio: {info['compression_ratio']:.2%}")
```

### Compression Techniques

#### Quantization

Reduce weight precision for smaller storage:

```python
from coral.compression import Quantizer

# 8-bit quantization
quantized, params = Quantizer.quantize_uniform(weight, bits=8)
print(f"Compression: {weight.nbytes / quantized.nbytes:.2f}x")

# Dequantize back
dequantized = Quantizer.dequantize(quantized)

# Per-channel quantization
quantized, params = Quantizer.quantize_per_channel(weight, bits=8, axis=0)
```

#### Pruning

Introduce sparsity by removing small weights:

```python
from coral.compression import Pruner

# Magnitude-based pruning
pruned, info = Pruner.prune_magnitude(weight, sparsity=0.5)
print(f"Pruned elements: {info['pruned_elements']}")

# Structured pruning (prune entire channels)
pruned, info = Pruner.prune_magnitude(
    weight, sparsity=0.3, structured=True, axis=0
)

# Analyze sparsity pattern
pattern = Pruner.get_sparsity_pattern(pruned)
```

## Advanced Usage

### Integrated Workflow

Combine deduplication, compression, and storage:

```python
# Setup
dedup = Deduplicator(similarity_threshold=0.98)
store = HDF5Store("compressed_weights.h5")

# Process weights
for name, weight in model_weights.items():
    # Check for duplicates
    ref_hash = dedup.add_weight(weight, name)
    
    # Only store unique weights
    if ref_hash == weight.compute_hash():
        # Apply compression
        compressed, _ = Quantizer.quantize_uniform(weight, bits=8)
        
        # Store
        store.store(compressed)

# Get statistics
dedup_report = dedup.get_deduplication_report()
storage_info = store.get_storage_info()
```

### Custom Storage Backend

Implement your own storage backend:

```python
from coral.storage import WeightStore

class CustomStore(WeightStore):
    def store(self, weight, hash_key=None):
        # Your implementation
        pass
    
    def load(self, hash_key):
        # Your implementation
        pass
    
    # Implement other required methods...
```

## Architecture

Coral is designed with modularity and extensibility in mind:

```
coral/
├── core/
│   ├── weight_tensor.py      # Weight representation
│   └── deduplicator.py       # Deduplication engine
├── storage/
│   ├── weight_store.py       # Storage interface
│   └── hdf5_store.py         # HDF5 implementation
├── compression/
│   ├── quantization.py       # Quantization methods
│   └── pruning.py            # Pruning methods
└── integrations/
    ├── pytorch.py            # PyTorch integration (coming soon)
    └── tensorflow.py         # TensorFlow integration (coming soon)
```

## Performance Considerations

- **Hash Computation**: Uses xxHash for fast content hashing
- **Similarity Detection**: Optimized for weights with same shape/dtype
- **Batch Operations**: Use batch methods for better performance
- **Compression Trade-offs**: Balance compression ratio vs. accuracy loss

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License

## Roadmap

- [ ] PyTorch model integration
- [ ] TensorFlow model integration
- [ ] Delta encoding for similar weights
- [ ] Distributed storage support
- [ ] Model versioning and tracking
- [ ] Advanced compression algorithms
- [ ] GPU-accelerated operations