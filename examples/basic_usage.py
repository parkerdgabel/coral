"""Basic usage example for Coral weight storage and deduplication"""

import numpy as np
from pathlib import Path
import json

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from coral import WeightTensor, Deduplicator, HDF5Store
from coral.compression import Quantizer, Pruner
from coral.core.weight_tensor import WeightMetadata


def create_sample_weights():
    """Create some sample weight tensors for demonstration"""
    weights = []
    
    # Create a conv2d weight
    conv_weight = WeightTensor(
        data=np.random.randn(64, 3, 3, 3).astype(np.float32),
        metadata=WeightMetadata(
            name="conv1.weight",
            shape=(64, 3, 3, 3),
            dtype=np.float32,
            layer_type="Conv2d",
            model_name="resnet50"
        )
    )
    weights.append(conv_weight)
    
    # Create a duplicate weight (exact copy)
    duplicate_weight = WeightTensor(
        data=conv_weight.data.copy(),
        metadata=WeightMetadata(
            name="conv1_copy.weight",
            shape=(64, 3, 3, 3),
            dtype=np.float32,
            layer_type="Conv2d",
            model_name="resnet50"
        )
    )
    weights.append(duplicate_weight)
    
    # Create a similar weight (slightly modified)
    similar_data = conv_weight.data + np.random.randn(*conv_weight.shape).astype(np.float32) * 0.01
    similar_weight = WeightTensor(
        data=similar_data,
        metadata=WeightMetadata(
            name="conv1_similar.weight",
            shape=(64, 3, 3, 3),
            dtype=np.float32,
            layer_type="Conv2d",
            model_name="resnet50"
        )
    )
    weights.append(similar_weight)
    
    # Create a linear layer weight
    linear_weight = WeightTensor(
        data=np.random.randn(1000, 2048).astype(np.float32),
        metadata=WeightMetadata(
            name="fc.weight",
            shape=(1000, 2048),
            dtype=np.float32,
            layer_type="Linear",
            model_name="resnet50"
        )
    )
    weights.append(linear_weight)
    
    # Create batch norm weights
    bn_weight = WeightTensor(
        data=np.random.randn(64).astype(np.float32),
        metadata=WeightMetadata(
            name="bn1.weight",
            shape=(64,),
            dtype=np.float32,
            layer_type="BatchNorm",
            model_name="resnet50"
        )
    )
    weights.append(bn_weight)
    
    return weights


def demonstrate_deduplication():
    """Demonstrate weight deduplication"""
    print("=== Weight Deduplication Demo ===\n")
    
    # Create sample weights
    weights = create_sample_weights()
    print(f"Created {len(weights)} sample weights")
    
    # Initialize deduplicator
    dedup = Deduplicator(similarity_threshold=0.98)
    
    # Add weights to deduplicator
    for weight in weights:
        hash_key = dedup.add_weight(weight)
        print(f"Added {weight.metadata.name}: hash={hash_key[:8]}...")
    
    # Get deduplication report
    report = dedup.get_deduplication_report()
    print("\nDeduplication Report:")
    print(json.dumps(report, indent=2))
    
    return dedup, weights


def demonstrate_storage():
    """Demonstrate weight storage with HDF5"""
    print("\n=== Weight Storage Demo ===\n")
    
    # Create storage directory
    storage_dir = Path("./data")
    storage_dir.mkdir(exist_ok=True)
    
    # Initialize HDF5 store
    with HDF5Store(str(storage_dir / "weights.h5"), compression='gzip') as store:
        # Create and store weights
        weights = create_sample_weights()
        
        print("Storing weights...")
        for weight in weights:
            hash_key = store.store(weight)
            print(f"Stored {weight.metadata.name}: {hash_key[:8]}...")
        
        # Get storage info
        info = store.get_storage_info()
        print("\nStorage Info:")
        print(json.dumps(info, indent=2))
        
        # Load a weight back
        print("\nLoading weight back...")
        loaded = store.load(store.list_weights()[0])
        if loaded:
            print(f"Loaded: {loaded}")


def demonstrate_compression():
    """Demonstrate weight compression techniques"""
    print("\n=== Weight Compression Demo ===\n")
    
    # Create a sample weight
    weight = WeightTensor(
        data=np.random.randn(256, 256).astype(np.float32),
        metadata=WeightMetadata(
            name="layer.weight",
            shape=(256, 256),
            dtype=np.float32,
            layer_type="Linear"
        )
    )
    
    print(f"Original weight: {weight}")
    print(f"Original size: {weight.nbytes} bytes")
    
    # Quantization
    print("\n1. Quantization (8-bit):")
    quantized, quant_params = Quantizer.quantize_uniform(weight, bits=8)
    print(f"Quantized size: {quantized.nbytes} bytes")
    print(f"Compression ratio: {weight.nbytes / quantized.nbytes:.2f}x")
    
    # Estimate quantization error
    error = Quantizer.estimate_quantization_error(weight, bits=8)
    print(f"Quantization MSE: {error:.6f}")
    
    # Pruning
    print("\n2. Magnitude Pruning (50% sparsity):")
    pruned, prune_info = Pruner.prune_magnitude(weight, sparsity=0.5)
    print(f"Pruned elements: {prune_info['pruned_elements']}")
    print(f"Actual sparsity: {prune_info['sparsity']:.2%}")
    
    # Combined compression
    print("\n3. Combined (Pruning + Quantization):")
    pruned_weight, _ = Pruner.prune_magnitude(weight, sparsity=0.5)
    pruned_quantized, _ = Quantizer.quantize_uniform(pruned_weight, bits=8)
    
    # In practice, sparse formats would save more space
    theoretical_sparse_size = pruned_quantized.nbytes * (1 - prune_info['sparsity'])
    print(f"Theoretical sparse size: {theoretical_sparse_size:.0f} bytes")
    print(f"Combined compression ratio: {weight.nbytes / theoretical_sparse_size:.2f}x")


def demonstrate_integration():
    """Demonstrate integrated workflow"""
    print("\n=== Integrated Workflow Demo ===\n")
    
    # Create weights
    weights = create_sample_weights()
    
    # Setup deduplicator and storage
    dedup = Deduplicator(similarity_threshold=0.98)
    storage_path = Path("./data/integrated.h5")
    storage_path.parent.mkdir(exist_ok=True)
    
    with HDF5Store(str(storage_path)) as store:
        # Process each weight
        for weight in weights:
            # 1. Check for deduplication
            ref_hash = dedup.add_weight(weight)
            
            # 2. Only store if unique
            if dedup.stats.unique_weights > 0 and ref_hash == weight.compute_hash():
                # This is a unique weight, apply compression
                compressed, _ = Quantizer.quantize_uniform(weight, bits=8)
                
                # Store compressed weight
                store.store(compressed)
                print(f"Stored unique weight: {weight.metadata.name}")
            else:
                print(f"Skipped duplicate/similar: {weight.metadata.name}")
        
        # Final statistics
        dedup_stats = dedup.compute_stats()
        storage_info = store.get_storage_info()
        
        print("\nFinal Statistics:")
        print(f"Total weights processed: {dedup_stats.total_weights}")
        print(f"Unique weights stored: {dedup_stats.unique_weights}")
        print(f"Deduplication savings: {dedup_stats.compression_ratio:.2%}")
        print(f"Storage compression ratio: {storage_info['compression_ratio']:.2%}")


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_deduplication()
    demonstrate_storage()
    demonstrate_compression()
    demonstrate_integration()
    
    print("\n=== Demo Complete ===")
    print("Check the ./data directory for stored weight files.")