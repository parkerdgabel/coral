#!/usr/bin/env python3
"""
Focused architectural diversity benchmark for Coral ML.

This is a streamlined version that tests key architectural differences
in space savings without creating overly large models.
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.version_control.repository import Repository
import numpy as np


def create_cnn_weights(name: str, size_mb: float = 1.0) -> Dict[str, WeightTensor]:
    """Create CNN-style weights with convolutional patterns."""
    target_params = int(size_mb * 1024 * 1024 / 4)
    
    # Distribute parameters across typical CNN layers
    base_channels = min(64, int(np.sqrt(target_params / 100)))
    
    weights = {}
    
    # Conv layers with typical CNN initialization patterns
    weights['conv1'] = WeightTensor(
        data=np.random.normal(0, np.sqrt(2.0/9), (base_channels, 3, 3, 3)).astype(np.float32),
        metadata=WeightMetadata(name=f"{name}_conv1", shape=(base_channels, 3, 3, 3), 
                               dtype=np.float32, layer_type="conv", model_name=name)
    )
    
    weights['conv2'] = WeightTensor(
        data=np.random.normal(0, np.sqrt(2.0/(base_channels*9)), (base_channels*2, base_channels, 3, 3)).astype(np.float32),
        metadata=WeightMetadata(name=f"{name}_conv2", shape=(base_channels*2, base_channels, 3, 3), 
                               dtype=np.float32, layer_type="conv", model_name=name)
    )
    
    # Dense layer
    remaining_params = max(1000, target_params - base_channels*3*3*3 - base_channels*2*base_channels*3*3)
    fc_size = int(np.sqrt(remaining_params))
    
    weights['fc'] = WeightTensor(
        data=np.random.normal(0, np.sqrt(2.0/fc_size), (10, fc_size)).astype(np.float32),
        metadata=WeightMetadata(name=f"{name}_fc", shape=(10, fc_size), 
                               dtype=np.float32, layer_type="dense", model_name=name)
    )
    
    return weights


def create_transformer_weights(name: str, size_mb: float = 1.0) -> Dict[str, WeightTensor]:
    """Create Transformer-style weights with attention patterns."""
    target_params = int(size_mb * 1024 * 1024 / 4)
    
    d_model = min(512, int(np.sqrt(target_params / 20)))
    
    weights = {}
    
    # Attention weights (different initialization than CNN)
    weights['q_proj'] = WeightTensor(
        data=np.random.normal(0, 1.0/np.sqrt(d_model), (d_model, d_model)).astype(np.float32),
        metadata=WeightMetadata(name=f"{name}_q_proj", shape=(d_model, d_model), 
                               dtype=np.float32, layer_type="attention", model_name=name)
    )
    
    weights['k_proj'] = WeightTensor(
        data=np.random.normal(0, 1.0/np.sqrt(d_model), (d_model, d_model)).astype(np.float32),
        metadata=WeightMetadata(name=f"{name}_k_proj", shape=(d_model, d_model), 
                               dtype=np.float32, layer_type="attention", model_name=name)
    )
    
    weights['v_proj'] = WeightTensor(
        data=np.random.normal(0, 1.0/np.sqrt(d_model), (d_model, d_model)).astype(np.float32),
        metadata=WeightMetadata(name=f"{name}_v_proj", shape=(d_model, d_model), 
                               dtype=np.float32, layer_type="attention", model_name=name)
    )
    
    # Feed forward
    ff_dim = d_model * 4
    weights['ff1'] = WeightTensor(
        data=np.random.normal(0, np.sqrt(2.0/(d_model + ff_dim)), (ff_dim, d_model)).astype(np.float32),
        metadata=WeightMetadata(name=f"{name}_ff1", shape=(ff_dim, d_model), 
                               dtype=np.float32, layer_type="dense", model_name=name)
    )
    
    weights['ff2'] = WeightTensor(
        data=np.random.normal(0, np.sqrt(2.0/(ff_dim + d_model)), (d_model, ff_dim)).astype(np.float32),
        metadata=WeightMetadata(name=f"{name}_ff2", shape=(d_model, ff_dim), 
                               dtype=np.float32, layer_type="dense", model_name=name)
    )
    
    return weights


def create_rnn_weights(name: str, size_mb: float = 1.0) -> Dict[str, WeightTensor]:
    """Create RNN-style weights with recurrent patterns."""
    target_params = int(size_mb * 1024 * 1024 / 4)
    
    hidden_size = min(256, int(np.sqrt(target_params / 12)))
    
    weights = {}
    
    # LSTM gates (4 gates: input, forget, gate, output)
    weights['ih_weight'] = WeightTensor(
        data=np.random.normal(0, 1.0/np.sqrt(hidden_size), (4 * hidden_size, hidden_size)).astype(np.float32),
        metadata=WeightMetadata(name=f"{name}_ih_weight", shape=(4 * hidden_size, hidden_size), 
                               dtype=np.float32, layer_type="rnn", model_name=name)
    )
    
    weights['hh_weight'] = WeightTensor(
        data=np.random.normal(0, 1.0/np.sqrt(hidden_size), (4 * hidden_size, hidden_size)).astype(np.float32),
        metadata=WeightMetadata(name=f"{name}_hh_weight", shape=(4 * hidden_size, hidden_size), 
                               dtype=np.float32, layer_type="rnn", model_name=name)
    )
    
    # Biases initialized to zero except forget gate (initialized to 1)
    bias_data = np.zeros(4 * hidden_size, dtype=np.float32)
    bias_data[hidden_size:2*hidden_size] = 1.0  # Forget gate bias
    
    weights['bias'] = WeightTensor(
        data=bias_data,
        metadata=WeightMetadata(name=f"{name}_bias", shape=(4 * hidden_size,), 
                               dtype=np.float32, layer_type="rnn", model_name=name)
    )
    
    return weights


def create_variations(base_weights: Dict[str, WeightTensor], similarity: float, suffix: str) -> Dict[str, WeightTensor]:
    """Create variations of base weights with controlled similarity."""
    variations = {}
    noise_scale = (1.0 - similarity) * 0.1
    
    for name, tensor in base_weights.items():
        noise = np.random.randn(*tensor.data.shape) * noise_scale
        new_data = tensor.data + noise.astype(np.float32)
        
        new_metadata = WeightMetadata(
            name=f"{tensor.metadata.name}_{suffix}",
            shape=tensor.metadata.shape,
            dtype=tensor.metadata.dtype,
            layer_type=tensor.metadata.layer_type,
            model_name=f"{tensor.metadata.model_name}_{suffix}"
        )
        
        variations[name] = WeightTensor(data=new_data, metadata=new_metadata)
    
    return variations


def run_focused_benchmark():
    """Run focused architectural diversity benchmark."""
    print("🏗️  Focused Architectural Diversity Benchmark")
    print("=" * 55)
    
    results = {}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir) / "coral_arch_repo"
        repo = Repository(path=repo_path, init=True)
        
        # Create base models for each architecture
        print("📦 Creating base models...")
        base_models = {
            'cnn': create_cnn_weights('cnn_base', 2.0),
            'transformer': create_transformer_weights('transformer_base', 2.0),
            'rnn': create_rnn_weights('rnn_base', 2.0)
        }
        
        # Test 1: Same architecture, high similarity (should compress well)
        print("\n🔄 Test 1: Same Architecture Variations")
        same_arch_models = []
        
        for arch, base_weights in base_models.items():
            same_arch_models.append(base_weights)
            # Add 3 variations with high similarity
            for i in range(3):
                variation = create_variations(base_weights, 0.99, f"var{i+1}")
                same_arch_models.append(variation)
        
        same_arch_result = measure_compression(repo, same_arch_models, "same_architecture")
        results['same_architecture'] = same_arch_result
        
        # Clear repo for next test
        repo = Repository(path=Path(temp_dir) / "coral_arch_repo2", init=True)
        
        # Test 2: Different architectures (should compress less)
        print("\n🌈 Test 2: Different Architectures")
        diff_arch_models = []
        
        for arch, base_weights in base_models.items():
            diff_arch_models.append(base_weights)
            # Add 1 variation of each architecture
            variation = create_variations(base_weights, 0.99, "var1")
            diff_arch_models.append(variation)
        
        diff_arch_result = measure_compression(repo, diff_arch_models, "different_architectures")
        results['different_architectures'] = diff_arch_result
        
        # Clear repo for next test  
        repo = Repository(path=Path(temp_dir) / "coral_arch_repo3", init=True)
        
        # Test 3: Mixed similarity levels
        print("\n📊 Test 3: Mixed Similarity Levels")
        mixed_models = []
        
        base_cnn = base_models['cnn']
        mixed_models.append(base_cnn)
        
        # Add variations with different similarity levels
        for i, sim in enumerate([0.99, 0.95, 0.90, 0.85]):
            variation = create_variations(base_cnn, sim, f"sim{int(sim*100)}")
            mixed_models.append(variation)
        
        mixed_result = measure_compression(repo, mixed_models, "mixed_similarity")
        results['mixed_similarity'] = mixed_result
    
    return results


def measure_compression(repo: Repository, models: List[Dict[str, WeightTensor]], test_name: str) -> Dict[str, Any]:
    """Measure compression for a set of models."""
    print(f"  📏 Measuring {test_name}...")
    
    # Calculate naive storage size
    naive_size = sum(
        sum(tensor.data.nbytes for tensor in model.values())
        for model in models
    )
    
    # Store in Coral repository
    start_time = time.time()
    for i, model in enumerate(models):
        repo.stage_weights(model)
        repo.commit(f"{test_name} model {i}")
    
    storage_time = time.time() - start_time
    
    # Calculate Coral storage size
    coral_size = sum(
        f.stat().st_size 
        for f in repo.coral_dir.rglob("*") 
        if f.is_file()
    )
    
    compression_ratio = naive_size / max(coral_size, 1)
    space_savings = (1 - coral_size / naive_size) * 100 if naive_size > 0 else 0
    
    result = {
        'models_count': len(models),
        'naive_size_mb': naive_size / (1024**2),
        'coral_size_mb': coral_size / (1024**2),
        'compression_ratio': compression_ratio,
        'space_savings_percent': space_savings,
        'storage_time_seconds': storage_time
    }
    
    print(f"    Models: {len(models)}")
    print(f"    Compression: {compression_ratio:.2f}x")
    print(f"    Space saved: {space_savings:.1f}%")
    
    return result


def main():
    """Run the focused architectural benchmark."""
    print("🚀 Coral ML Focused Architecture Benchmark")
    print("=" * 60)
    
    try:
        results = run_focused_benchmark()
        
        print("\n" + "=" * 60)
        print("📈 ARCHITECTURAL BENCHMARK RESULTS")
        print("=" * 60)
        
        # Same architecture results
        same_arch = results.get('same_architecture', {})
        print(f"\n🔄 Same Architecture Variations:")
        print(f"   Models: {same_arch.get('models_count', 0)}")
        print(f"   Compression: {same_arch.get('compression_ratio', 0):.2f}x")
        print(f"   Space saved: {same_arch.get('space_savings_percent', 0):.1f}%")
        
        # Different architectures results
        diff_arch = results.get('different_architectures', {})
        print(f"\n🌈 Different Architectures:")
        print(f"   Models: {diff_arch.get('models_count', 0)}")
        print(f"   Compression: {diff_arch.get('compression_ratio', 0):.2f}x")
        print(f"   Space saved: {diff_arch.get('space_savings_percent', 0):.1f}%")
        
        # Mixed similarity results
        mixed_sim = results.get('mixed_similarity', {})
        print(f"\n📊 Mixed Similarity Levels:")
        print(f"   Models: {mixed_sim.get('models_count', 0)}")
        print(f"   Compression: {mixed_sim.get('compression_ratio', 0):.2f}x")
        print(f"   Space saved: {mixed_sim.get('space_savings_percent', 0):.1f}%")
        
        # Analysis
        print(f"\n🔍 ANALYSIS:")
        same_compression = same_arch.get('compression_ratio', 1.0)
        diff_compression = diff_arch.get('compression_ratio', 1.0)
        
        if same_compression > diff_compression * 1.2:
            print("   ✅ Coral excels with same-architecture models")
        else:
            print("   ⚠️  Similar compression across architectures")
        
        print(f"   Same-arch advantage: {same_compression/diff_compression:.2f}x")
        
        # Overall assessment
        best_compression = max(same_compression, diff_compression)
        if best_compression >= 2.0:
            print("   ✅ EXCELLENT: >2.0x compression achieved")
        elif best_compression >= 1.5:
            print("   ✅ GOOD: >1.5x compression achieved")
        elif best_compression >= 1.2:
            print("   ⚠️  MODERATE: >1.2x compression achieved")
        else:
            print("   ❌ POOR: <1.2x compression")
        
        return 0
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())