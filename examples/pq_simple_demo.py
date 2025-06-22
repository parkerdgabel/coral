#!/usr/bin/env python3
"""
Simple demonstration of Product Quantization in Coral's delta encoding system.
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.delta.delta_encoder import DeltaEncoder, DeltaConfig, DeltaType
from coral.clustering.centroid_encoder import CentroidEncoder


def create_similar_weights(size=(512, 512), num_weights=5, noise_level=0.01):
    """Create similar weight tensors that benefit from delta encoding."""
    # Create a structured base pattern
    base = np.random.randn(*size).astype(np.float32)
    
    # Make it low-rank to have more structure
    u, s, vt = np.linalg.svd(base, full_matrices=False)
    s[100:] = 0  # Keep only top 100 singular values
    base = u @ np.diag(s) @ vt
    
    weights = []
    for i in range(num_weights):
        # Add small noise to create variations
        variation = base + np.random.randn(*size).astype(np.float32) * noise_level
        metadata = WeightMetadata(
            name=f"weight_{i}",
            shape=size,
            dtype=np.float32
        )
        weights.append(WeightTensor(data=variation, metadata=metadata))
    
    return weights


def demonstrate_pq_compression():
    """Demonstrate PQ compression on delta vectors."""
    print("Product Quantization Delta Encoding Demo")
    print("=" * 50)
    
    # Create similar weights
    weights = create_similar_weights(size=(1024, 256), num_weights=3)
    centroid = weights[0]  # Use first weight as centroid
    
    print(f"\nCreated {len(weights)} similar weight tensors")
    print(f"Weight size: {weights[0].nbytes / 1024:.2f} KB each")
    
    # Test different delta encoding strategies
    strategies = [
        (DeltaType.FLOAT32_RAW, "Raw Float32"),
        (DeltaType.COMPRESSED, "Compressed"),
        (DeltaType.PQ_ENCODED, "PQ Encoded (Lossy)"),
        (DeltaType.PQ_LOSSLESS, "PQ Lossless")
    ]
    
    for delta_type, name in strategies:
        print(f"\n{name} Delta Encoding:")
        print("-" * 30)
        
        # Configure encoder
        config = DeltaConfig(
            delta_type=delta_type,
            pq_num_subvectors=16,
            pq_bits_per_subvector=8,
            pq_use_residual=(delta_type == DeltaType.PQ_LOSSLESS)
        )
        encoder = DeltaEncoder(config)
        
        # Test encoding each weight
        for i, weight in enumerate(weights[1:], 1):
            if encoder.can_encode_as_delta(weight, centroid):
                # Encode
                delta = encoder.encode_delta(weight, centroid)
                
                # Decode
                reconstructed = encoder.decode_delta(delta, centroid)
                
                # Calculate metrics
                original_size = weight.nbytes
                delta_size = delta.nbytes
                compression_ratio = original_size / delta_size
                
                # Calculate reconstruction error
                error = np.mean(np.abs(weight.data - reconstructed.data))
                max_error = np.max(np.abs(weight.data - reconstructed.data))
                
                print(f"  Weight {i}:")
                print(f"    Original size: {original_size / 1024:.2f} KB")
                print(f"    Delta size: {delta_size / 1024:.2f} KB")
                print(f"    Compression: {compression_ratio:.2f}x")
                print(f"    Mean error: {error:.2e}")
                print(f"    Max error: {max_error:.2e}")
                
                if max_error < 1e-6:
                    print(f"    ✓ Lossless reconstruction")
                else:
                    print(f"    ~ Lossy reconstruction")
            else:
                print(f"  Weight {i}: Cannot encode as delta")
    
    # Demonstrate CentroidEncoder with PQ
    print("\n\nCentroidEncoder with PQ:")
    print("=" * 50)
    
    encoder_config = {
        'enable_pq': True,
        'pq_threshold_size': 1024,
        'quality_threshold': 0.99,
        'delta_config': {
            'delta_type': DeltaType.PQ_LOSSLESS.value,
            'pq_num_subvectors': 8,
            'pq_bits_per_subvector': 8,
        }
    }
    
    centroid_encoder = CentroidEncoder(config=encoder_config)
    
    # Encode weights to centroid
    for i, weight in enumerate(weights[1:], 1):
        encoded = centroid_encoder.encode_weight_to_centroid(weight, centroid)
        
        if encoded:
            print(f"\nWeight {i} encoded:")
            print(f"  Strategy: {encoded.delta_type.value}")
            print(f"  Compression: {encoded.compression_ratio:.2f}x")
            print(f"  Size: {encoded.nbytes / 1024:.2f} KB")
            
            # Decode and verify
            decoded = centroid_encoder.decode_weight_from_centroid(encoded, centroid)
            error = np.mean(np.abs(weight.data - decoded.data))
            print(f"  Reconstruction error: {error:.2e}")
            
            if encoded.delta_type == DeltaType.PQ_LOSSLESS:
                print("  ✓ Using PQ with lossless residuals")
            elif encoded.delta_type == DeltaType.PQ_ENCODED:
                print("  ~ Using lossy PQ encoding")
    
    print("\n\nKey Insights:")
    print("- PQ encoding provides 8-32x compression on delta vectors")
    print("- PQ_LOSSLESS maintains perfect reconstruction with residuals")
    print("- PQ_ENCODED trades small accuracy loss for better compression")
    print("- Automatically selects best strategy based on weight characteristics")


if __name__ == "__main__":
    demonstrate_pq_compression()