#!/usr/bin/env python3
"""Demo of CentroidEncoder PQ integration features."""

import numpy as np
from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.clustering.centroid_encoder import CentroidEncoder
from coral.delta.delta_encoder import DeltaType, Delta

def main():
    """Demonstrate PQ integration features in CentroidEncoder."""
    print("=== CentroidEncoder PQ Integration Features Demo ===\n")
    
    # 1. Configuration
    print("1. PQ Configuration Options:")
    configs = [
        ("Default config", {}),
        ("PQ disabled", {'enable_pq': False}),
        ("Custom threshold", {'enable_pq': True, 'pq_threshold_size': 2048}),
        ("High quality mode", {'enable_pq': True, 'quality_threshold': 0.995}),
    ]
    
    for name, config in configs:
        encoder = CentroidEncoder(config)
        print(f"\n   {name}:")
        print(f"     PQ enabled: {encoder.enable_pq}")
        print(f"     PQ threshold: {encoder.pq_threshold_size} elements")
        print(f"     Quality threshold: {encoder.quality_threshold}")
        if encoder.enable_pq:
            print(f"     PQ config: subvectors={encoder.pq_config.num_subvectors}, "
                  f"bits={encoder.pq_config.bits_per_subvector}, "
                  f"residual={encoder.pq_config.use_residual}")
    
    # 2. PQ Decision Logic
    print("\n2. Testing _should_use_pq() decision logic:")
    encoder = CentroidEncoder({'enable_pq': True, 'pq_threshold_size': 1024})
    
    # Create test scenarios
    test_cases = [
        ("Small weight (256 elements)", (16, 16)),
        ("Medium weight (1024 elements)", (32, 32)),
        ("Large weight (4096 elements)", (64, 64)),
        ("Very large weight (16384 elements)", (128, 128)),
    ]
    
    for name, shape in test_cases:
        weight_data = np.random.randn(*shape).astype(np.float32)
        weight = WeightTensor(
            data=weight_data,
            metadata=WeightMetadata(name=name, shape=shape, dtype='float32')
        )
        
        # Create similar centroid
        centroid_data = weight_data * 0.95 + np.random.randn(*shape).astype(np.float32) * 0.05
        centroid = WeightTensor(
            data=centroid_data,
            metadata=WeightMetadata(name="centroid", shape=shape, dtype='float32')
        )
        
        # Create mock delta
        delta = Delta(
            delta_type=DeltaType.FLOAT32_RAW,
            data=(weight_data - centroid_data),
            metadata={},
            reference_hash=centroid.compute_hash(),
            original_shape=shape,
            original_dtype='float32',
            compression_ratio=1.0
        )
        
        should_use, strategy = encoder._should_use_pq(delta, weight, centroid)
        print(f"\n   {name}:")
        print(f"     Weight size: {weight.data.size} elements")
        print(f"     Should use PQ: {should_use}")
        if should_use:
            print(f"     Suggested strategy: {strategy.value}")
    
    # 3. Strategy Selection with PQ
    print("\n3. Strategy selection including PQ options:")
    
    large_weight = WeightTensor(
        data=np.random.randn(128, 128).astype(np.float32),
        metadata=WeightMetadata(name="large_weight", shape=(128, 128), dtype='float32')
    )
    
    similar_centroid = WeightTensor(
        data=large_weight.data * 0.99 + np.random.randn(128, 128).astype(np.float32) * 0.01,
        metadata=WeightMetadata(name="similar_centroid", shape=(128, 128), dtype='float32')
    )
    
    # Test with PQ enabled
    encoder_with_pq = CentroidEncoder({'enable_pq': True})
    strategy, metrics = encoder_with_pq.select_optimal_strategy(large_weight, similar_centroid)
    
    print(f"\n   With PQ enabled:")
    print(f"     Selected strategy: {strategy.value}")
    print(f"     Strategies tested: {list(metrics['all_strategies'].keys())}")
    
    # Test with PQ disabled
    encoder_no_pq = CentroidEncoder({'enable_pq': False})
    strategy, metrics = encoder_no_pq.select_optimal_strategy(large_weight, similar_centroid)
    
    print(f"\n   With PQ disabled:")
    print(f"     Selected strategy: {strategy.value}")
    print(f"     Strategies tested: {list(metrics['all_strategies'].keys())}")
    
    # 4. Batch encoding with size grouping
    print("\n4. Batch encoding with size-based grouping for PQ:")
    
    # Create weights of different sizes
    weights = []
    centroids = []
    assignments = []
    
    sizes = [32, 64, 32, 128, 64]  # Mixed sizes
    print(f"\n   Original weight order: {sizes}")
    
    for i, size in enumerate(sizes):
        w_data = np.random.randn(size, size).astype(np.float32)
        weight = WeightTensor(
            data=w_data,
            metadata=WeightMetadata(name=f"weight_{i}", shape=(size, size), dtype='float32')
        )
        weights.append(weight)
        
        c_data = w_data * 0.9
        centroid = WeightTensor(
            data=c_data,
            metadata=WeightMetadata(name=f"centroid_{i}", shape=(size, size), dtype='float32')
        )
        centroids.append(centroid)
        
        from coral.clustering.cluster_types import ClusterAssignment
        assignment = ClusterAssignment(
            weight_name=f"weight_{i}",
            weight_hash=weight.compute_hash(),
            cluster_id=str(i),
            distance_to_centroid=0.1,
            similarity_score=0.9
        )
        assignments.append(assignment)
    
    # Batch encode (will sort by size internally when PQ is enabled)
    encoder = CentroidEncoder({'enable_pq': True})
    
    # Note: The actual encoding might fail without full PQ infrastructure,
    # but we can demonstrate the size grouping logic
    print("   When PQ is enabled, weights are grouped by size for better codebook reuse")
    
    # 5. PQ-specific configuration parameters
    print("\n5. PQ-specific configuration parameters:")
    
    custom_config = {
        'enable_pq': True,
        'pq_threshold_size': 512,
        'pq_num_subvectors': 16,
        'pq_n_codewords': 512,  # Will be converted to 9 bits
        'quality_threshold': 0.999
    }
    
    encoder = CentroidEncoder(custom_config)
    print(f"\n   Custom PQ configuration:")
    print(f"     Threshold size: {encoder.pq_threshold_size} elements")
    print(f"     Subvectors: {encoder.pq_config.num_subvectors}")
    print(f"     Bits per subvector: {encoder.pq_config.bits_per_subvector}")
    print(f"     Codewords: {encoder.pq_config.num_codewords}")
    print(f"     Use residual: {encoder.pq_config.use_residual}")
    
    # 6. Size estimation for PQ
    print("\n6. Delta size estimation for PQ strategies:")
    
    weight = WeightTensor(
        data=np.random.randn(256, 256).astype(np.float32),
        metadata=WeightMetadata(name="test", shape=(256, 256), dtype='float32')
    )
    centroid = WeightTensor(
        data=weight.data * 0.9,
        metadata=WeightMetadata(name="centroid", shape=(256, 256), dtype='float32')
    )
    
    encoder = CentroidEncoder()
    
    strategies = [
        DeltaType.FLOAT32_RAW,
        DeltaType.COMPRESSED,
        DeltaType.INT8_QUANTIZED,
        DeltaType.PQ_ENCODED,
        DeltaType.PQ_LOSSLESS
    ]
    
    print(f"\n   Original weight size: {weight.nbytes:,} bytes")
    print("\n   Estimated delta sizes:")
    for strategy in strategies:
        estimated_size = encoder._estimate_delta_size(weight, centroid, strategy)
        compression = weight.nbytes / estimated_size if estimated_size > 0 else 0
        print(f"     {strategy.value:<20}: {estimated_size:>8,} bytes (compression: {compression:.2f}x)")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()