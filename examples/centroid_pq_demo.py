#!/usr/bin/env python3
"""Demo of CentroidEncoder with Product Quantization integration."""

import numpy as np
from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.clustering.centroid_encoder import CentroidEncoder
from coral.delta.delta_encoder import DeltaType

def main():
    """Demonstrate PQ integration in CentroidEncoder."""
    print("=== CentroidEncoder with Product Quantization Demo ===\n")
    
    # Create a large weight tensor (above PQ threshold)
    print("1. Creating large weight tensor (128x128 = 16,384 elements)")
    weight_data = np.random.randn(128, 128).astype(np.float32)
    weight = WeightTensor(
        data=weight_data,
        metadata=WeightMetadata(
            name="large_model_weight",
            shape=weight_data.shape,
            dtype=str(weight_data.dtype)
        )
    )
    print(f"   Weight size: {weight.nbytes:,} bytes\n")
    
    # Create a similar centroid (as would come from clustering)
    print("2. Creating cluster centroid (similar to weight)")
    # Simulate centroid as average of similar weights with small variation
    centroid_data = weight_data * 0.95 + np.random.randn(*weight_data.shape).astype(np.float32) * 0.05
    centroid = WeightTensor(
        data=centroid_data,
        metadata=WeightMetadata(
            name="cluster_centroid",
            shape=centroid_data.shape,
            dtype=str(centroid_data.dtype)
        )
    )
    
    # Initialize encoder with PQ enabled
    print("3. Initializing CentroidEncoder with PQ support")
    encoder = CentroidEncoder({
        'enable_pq': True,
        'pq_threshold_size': 1024,  # Enable PQ for weights > 1024 elements
        'quality_threshold': 0.99,   # High quality requirement
    })
    print(f"   PQ enabled: {encoder.enable_pq}")
    print(f"   PQ threshold: {encoder.pq_threshold_size} elements")
    print(f"   Quality threshold: {encoder.quality_threshold}\n")
    
    # Test different encoding strategies
    print("4. Testing different encoding strategies:")
    strategies = ["auto", "FLOAT32_RAW", "COMPRESSED", "INT8_QUANTIZED", "PQ_ENCODED", "PQ_LOSSLESS"]
    
    results = []
    for strategy in strategies:
        try:
            encoded = encoder.encode_weight(weight, centroid, strategy=strategy)
            stats = encoder.get_encoding_stats(encoded)
            
            results.append({
                'strategy': strategy,
                'encoded_size': stats['encoded_size'],
                'compression_ratio': stats['compression_ratio'],
                'is_lossless': stats['is_lossless'],
                'actual_strategy': stats['encoding_strategy']
            })
            
            # Decode to verify
            reconstructed = encoder.decode_weight(encoded, centroid)
            quality = encoder.assess_encoding_quality(weight, reconstructed)
            results[-1]['mse'] = quality['mse']
            results[-1]['cosine_similarity'] = quality['cosine_similarity']
            
        except Exception as e:
            print(f"   Error with strategy {strategy}: {e}")
    
    # Display results
    print("\n5. Encoding Results Summary:")
    print(f"   {'Strategy':<15} {'Size (bytes)':<12} {'Compression':<12} {'Lossless':<10} {'MSE':<12} {'Cosine Sim':<12}")
    print("   " + "-" * 83)
    
    for result in results:
        print(f"   {result['strategy']:<15} "
              f"{result['encoded_size']:<12,} "
              f"{result['compression_ratio']:<12.2f} "
              f"{'Yes' if result['is_lossless'] else 'No':<10} "
              f"{result.get('mse', 0):<12.2e} "
              f"{result.get('cosine_similarity', 0):<12.4f}")
    
    # Test batch encoding with PQ
    print("\n6. Testing batch encoding with size-based grouping:")
    weights = []
    centroids = []
    assignments = []
    
    # Create weights of different sizes
    sizes = [32, 64, 32, 128, 64, 128]  # Will be sorted for PQ efficiency
    for i, size in enumerate(sizes):
        w_data = np.random.randn(size, size).astype(np.float32)
        weight = WeightTensor(
            data=w_data,
            metadata=WeightMetadata(
                name=f"weight_{i}",
                shape=w_data.shape,
                dtype=str(w_data.dtype)
            )
        )
        weights.append(weight)
        
        # Create matching centroid
        c_data = w_data * 0.9 + np.random.randn(size, size).astype(np.float32) * 0.1
        centroid = WeightTensor(
            data=c_data,
            metadata=WeightMetadata(
                name=f"centroid_{i}",
                shape=c_data.shape,
                dtype=str(c_data.dtype)
            )
        )
        centroids.append(centroid)
        
        # Create assignment
        from coral.clustering.cluster_types import ClusterAssignment
        assignment = ClusterAssignment(
            weight_name=f"weight_{i}",
            weight_hash=weight.compute_hash(),
            cluster_id=str(i),
            distance_to_centroid=0.1,
            similarity_score=0.9
        )
        assignments.append(assignment)
    
    # Batch encode
    deltas = encoder.batch_encode(weights, assignments, centroids)
    
    print(f"\n   Batch encoded {len(deltas)} weights")
    print(f"   Weights sorted by size for PQ codebook efficiency: {encoder.enable_pq}")
    
    # Show PQ analysis
    print("\n7. PQ Suitability Analysis:")
    print("   Testing when PQ encoding is beneficial...")
    
    # Create test scenarios
    test_cases = [
        ("Small weight (below threshold)", np.random.randn(16, 16).astype(np.float32)),
        ("Large similar weight", weight_data * 0.99 + np.random.randn(*weight_data.shape).astype(np.float32) * 0.01),
        ("Large different weight", np.random.randn(128, 128).astype(np.float32) * 10),
        ("Structured pattern weight", np.tile(np.arange(128), (128, 1)).astype(np.float32))
    ]
    
    for name, test_data in test_cases:
        test_weight = WeightTensor(
            data=test_data,
            metadata=WeightMetadata(
                name=name,
                shape=test_data.shape,
                dtype=str(test_data.dtype)
            )
        )
        
        # Use auto strategy to see what encoder chooses
        encoded = encoder.encode_weight(test_weight, centroid, strategy="auto")
        stats = encoder.get_encoding_stats(encoded)
        
        print(f"\n   {name}:")
        print(f"     Size: {test_weight.nbytes:,} bytes ({test_weight.data.size} elements)")
        print(f"     Selected strategy: {stats['encoding_strategy']}")
        print(f"     Compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"     Is lossless: {stats['is_lossless']}")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()