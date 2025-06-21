#!/usr/bin/env python3
"""
Simple benchmark runner that demonstrates the Coral ML benchmarking system.

This is a simplified version that shows the core functionality working
with minimal dependencies and error handling.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.version_control.repository import Repository
import numpy as np


def create_test_model(name: str, size_mb: float) -> dict:
    """Create a test model with the specified size."""
    # Calculate number of parameters to reach target size
    bytes_per_param = 4  # float32
    target_bytes = size_mb * 1024 * 1024
    num_params = int(target_bytes / bytes_per_param)
    
    # Create layers
    weights = {}
    
    # Split into several layers for realistic structure
    layer_sizes = [
        num_params // 4,  # Large layer
        num_params // 4,  # Large layer
        num_params // 8,  # Medium layer
        num_params // 8,  # Medium layer
        num_params // 8,  # Medium layer
        num_params // 8,  # Medium layer
    ]
    
    for i, layer_size in enumerate(layer_sizes):
        if layer_size > 0:
            # Create random weights
            weight_data = np.random.randn(layer_size).astype(np.float32)
            
            metadata = WeightMetadata(
                name=f"{name}_layer_{i}",
                shape=(layer_size,),
                dtype=np.float32,
                layer_type="dense",
                model_name=name
            )
            
            weights[f"layer_{i}"] = WeightTensor(
                data=weight_data,
                metadata=metadata
            )
    
    return weights


def run_storage_benchmark():
    """Run a simple storage efficiency benchmark."""
    print("🔧 Running Storage Efficiency Benchmark")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir) / "coral_repo"
        
        # Initialize repository
        repo = Repository(path=repo_path, init=True)
        
        # Create test models of different sizes
        models = {
            "small_model": create_test_model("small", 1.0),    # 1 MB
            "medium_model": create_test_model("medium", 5.0),  # 5 MB  
            "large_model": create_test_model("large", 10.0),   # 10 MB
        }
        
        # Create similar models (simulating fine-tuning)
        similar_models = {}
        for base_name, base_weights in models.items():
            for similarity in [0.99, 0.95, 0.90]:
                sim_name = f"{base_name}_sim_{int(similarity*100)}"
                sim_weights = {}
                
                for weight_name, weight_tensor in base_weights.items():
                    # Add small noise to create similar weights
                    noise_scale = 1.0 - similarity
                    noise = np.random.randn(*weight_tensor.data.shape) * noise_scale * 0.1
                    sim_data = weight_tensor.data + noise.astype(np.float32)
                    
                    sim_metadata = WeightMetadata(
                        name=f"{sim_name}_{weight_name}",
                        shape=weight_tensor.metadata.shape,
                        dtype=weight_tensor.metadata.dtype,
                        layer_type=weight_tensor.metadata.layer_type,
                        model_name=sim_name
                    )
                    
                    sim_weights[weight_name] = WeightTensor(
                        data=sim_data,
                        metadata=sim_metadata
                    )
                
                similar_models[sim_name] = sim_weights
        
        # Measure storage without Coral (naive approach)
        print("📏 Measuring naive storage...")
        naive_size = 0
        for model_name, weights in {**models, **similar_models}.items():
            for weight in weights.values():
                naive_size += weight.data.nbytes
        
        # Store models in Coral repository
        print("📦 Storing models in Coral repository...")
        start_time = time.time()
        
        all_models = {**models, **similar_models}
        total_weights = 0
        
        for model_name, weights in all_models.items():
            repo.stage_weights(weights)
            repo.commit(f"Add {model_name}")
            total_weights += len(weights)
        
        storage_time = time.time() - start_time
        
        # Get repository size
        coral_size = sum(
            f.stat().st_size 
            for f in repo_path.rglob("*") 
            if f.is_file()
        )
        
        # Calculate metrics
        compression_ratio = naive_size / coral_size if coral_size > 0 else 0
        space_saved = naive_size - coral_size
        space_saved_percent = (space_saved / naive_size) * 100 if naive_size > 0 else 0
        
        # Results
        print()
        print("📊 BENCHMARK RESULTS")
        print("-" * 30)
        print(f"Models tested: {len(all_models)}")
        print(f"Total weights: {total_weights}")
        print(f"Naive storage: {naive_size / (1024**2):.1f} MB")
        print(f"Coral storage: {coral_size / (1024**2):.1f} MB")
        print(f"Space saved: {space_saved / (1024**2):.1f} MB ({space_saved_percent:.1f}%)")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Storage time: {storage_time:.2f} seconds")
        
        # Success indicators
        if compression_ratio >= 1.5:
            print("✅ EXCELLENT: >1.5x compression achieved")
        elif compression_ratio >= 1.2:
            print("✅ GOOD: >1.2x compression achieved")
        elif compression_ratio >= 1.1:
            print("⚠️  MODERATE: >1.1x compression achieved")
        else:
            print("❌ POOR: <1.1x compression")
            
        return {
            'models_tested': len(all_models),
            'total_weights': total_weights,
            'naive_size_mb': naive_size / (1024**2),
            'coral_size_mb': coral_size / (1024**2),
            'space_saved_mb': space_saved / (1024**2),
            'space_saved_percent': space_saved_percent,
            'compression_ratio': compression_ratio,
            'storage_time': storage_time
        }


def main():
    """Run the simple benchmark."""
    print("🚀 Coral ML Simple Benchmark Suite")
    print("=" * 60)
    print()
    
    try:
        results = run_storage_benchmark()
        
        print()
        print("🎉 Benchmark completed successfully!")
        print(f"   Achieved {results['compression_ratio']:.2f}x compression")
        print(f"   Saved {results['space_saved_percent']:.1f}% storage space")
        
        return 0
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())