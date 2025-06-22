# Weight Computation Graph Architecture Design

## Overview

Transform Coral's weight representation from static numpy arrays to dynamic computation graphs. This enables lazy evaluation, advanced compression techniques, and memory-efficient storage of large models.

## Core Architecture

### 1. WeightOp Base Class
```python
class WeightOp(ABC):
    """Abstract base class for weight operations in computation graph."""
    
    @abstractmethod
    def forward(self) -> np.ndarray:
        """Compute and return the weight tensor."""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> int:
        """Return memory usage in bytes for this operation."""
        pass
    
    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """Serialize operation for storage."""
        pass
    
    @classmethod
    @abstractmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'WeightOp':
        """Reconstruct operation from serialized data."""
        pass
```

### 2. Concrete Operations

#### Basic Operations
- **IdentityOp**: Wraps raw numpy arrays (backward compatibility)
- **AddOp**: Adds multiple weight tensors
- **MatMulOp**: Matrix multiplication
- **ScaleOp**: Element-wise scaling
- **ReshapeOp**: Tensor reshaping
- **SliceOp**: Extract sub-tensors
- **ConcatOp**: Concatenate tensors

#### Compression Operations
- **SVDOp**: Low-rank approximation via SVD
- **TuckerOp**: Tucker decomposition for conv layers
- **SparseOp**: Sparse matrix formats (CSR, COO, CSC)
- **QuantizeOp**: Quantization to lower bit widths
- **PQOp**: Product quantization decoding
- **DeltaOp**: Delta from reference weight

#### Advanced Operations
- **DictionaryOp**: Dictionary learning compression
- **WaveletOp**: Wavelet transform compression
- **NeuralOp**: Neural network-based compression
- **GeneratorOp**: Procedural weight generation

### 3. ComputationGraph Class
```python
class ComputationGraph:
    """Represents a DAG of weight operations."""
    
    def __init__(self, root_op: WeightOp):
        self.root = root_op
        self._cache = {}  # Memoization cache
        
    def evaluate(self) -> np.ndarray:
        """Lazily evaluate the graph and return weight tensor."""
        return self._evaluate_op(self.root)
        
    def get_total_memory(self) -> int:
        """Calculate total memory usage of graph."""
        pass
        
    def optimize(self) -> 'ComputationGraph':
        """Apply graph optimization passes."""
        pass
```

### 4. Integration Points

#### WeightTensor Enhancement
```python
class WeightTensor:
    def __init__(self, data=None, metadata=None, computation_graph=None):
        if computation_graph is not None:
            self._graph = computation_graph
            self._data = None  # Lazy evaluation
        else:
            # Backward compatibility
            self._graph = ComputationGraph(IdentityOp(data))
            self._data = data
            
    @property
    def data(self):
        if self._data is None:
            self._data = self._graph.evaluate()
        return self._data
```

## Implementation Phases

### Phase 1: Core Infrastructure (Agent 1)
1. Create `src/coral/core/weight_ops/` module structure
2. Implement WeightOp base class and type system
3. Implement basic operations (Identity, Add, MatMul, Scale)
4. Create ComputationGraph with lazy evaluation
5. Add comprehensive unit tests

### Phase 2: Compression Operations (Agent 2)
1. Implement SVDOp with configurable rank
2. Implement SparseOp with multiple formats
3. Implement QuantizeOp with bit-width options
4. Create operation benchmarks
5. Add compression-specific tests

### Phase 3: Storage Integration (Agent 3)
1. Extend HDF5Store to save computation graphs
2. Implement graph serialization/deserialization
3. Update storage format versioning
4. Add backward compatibility layer
5. Create migration utilities

### Phase 4: Deduplication Enhancement (Agent 4)
1. Modify Deduplicator to work at operation level
2. Implement operation-level similarity detection
3. Create graph-based delta encoding
4. Update deduplication statistics
5. Add deduplication tests

### Phase 5: WeightTensor Migration (Agent 5)
1. Update WeightTensor to use ComputationGraph
2. Maintain full backward compatibility
3. Add lazy evaluation with caching
4. Update all WeightTensor tests
5. Create migration documentation

## Benefits

1. **Memory Efficiency**: Only materialize weights when needed
2. **Advanced Compression**: Support for any compression technique
3. **Composability**: Combine multiple compression methods
4. **Flexibility**: Easy to add new operations
5. **Debugging**: Visualize weight computation pipeline

## Example Usage

```python
# Low-rank approximation
weight = WeightTensor(computation_graph=ComputationGraph(
    SVDOp(U, S, V, rank=10)
))

# Sparse + Quantized
weight = WeightTensor(computation_graph=ComputationGraph(
    QuantizeOp(
        SparseOp(indices, values, shape),
        bits=4
    )
))

# Dictionary learning
weight = WeightTensor(computation_graph=ComputationGraph(
    MatMulOp(
        DictionaryOp(dictionary_id),
        SparseOp(sparse_codes)
    )
))
```

## Migration Strategy

1. **Phase 1**: Implement alongside existing system
2. **Phase 2**: Add opt-in flag for computation graphs
3. **Phase 3**: Migrate internal usage gradually
4. **Phase 4**: Make computation graphs default
5. **Phase 5**: Deprecate old raw array storage

## Performance Considerations

1. **Caching**: Aggressive caching of evaluated tensors
2. **Parallel Evaluation**: Multi-threaded graph execution
3. **GPU Acceleration**: CUDA kernels for operations
4. **Memory Pooling**: Reuse temporary buffers
5. **Graph Optimization**: Constant folding, operation fusion

## Testing Strategy

1. **Unit Tests**: Each operation in isolation
2. **Integration Tests**: Full graph evaluation
3. **Performance Tests**: Benchmark vs raw arrays
4. **Compatibility Tests**: Ensure backward compatibility
5. **Stress Tests**: Large graphs, deep nesting