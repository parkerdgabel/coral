#!/usr/bin/env python3
"""
Safetensors Conversion Demo

This example demonstrates how to convert between Coral and Safetensors formats:
1. Convert a Coral repository to Safetensors for sharing
2. Import Safetensors files into Coral for version control
3. Batch convert multiple Safetensors files
"""

import shutil
from pathlib import Path

import numpy as np

from coral import Repository
from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.safetensors.converter import (
    batch_convert_safetensors,
    convert_coral_to_safetensors,
    convert_safetensors_to_coral,
)
from coral.safetensors.reader import SafetensorsReader
from coral.safetensors.writer import SafetensorsWriter


def create_sample_model_weights():
    """Create sample model weights for demonstration."""
    weights = {}
    
    # Transformer-like architecture weights
    configs = [
        # Embeddings
        ("embedding.weight", (50000, 768)),
        
        # Attention layers
        ("attention.0.query.weight", (768, 768)),
        ("attention.0.query.bias", (768,)),
        ("attention.0.key.weight", (768, 768)),
        ("attention.0.key.bias", (768,)),
        ("attention.0.value.weight", (768, 768)),
        ("attention.0.value.bias", (768,)),
        ("attention.0.output.weight", (768, 768)),
        ("attention.0.output.bias", (768,)),
        
        # FFN layers
        ("ffn.0.up.weight", (768, 3072)),
        ("ffn.0.up.bias", (3072,)),
        ("ffn.0.down.weight", (3072, 768)),
        ("ffn.0.down.bias", (768,)),
        
        # Layer norms
        ("ln.0.weight", (768,)),
        ("ln.0.bias", (768,)),
        
        # Output layer
        ("lm_head.weight", (768, 50000)),
    ]
    
    for name, shape in configs:
        data = np.random.randn(*shape).astype(np.float32) * 0.02
        metadata = WeightMetadata(
            name=name,
            shape=shape,
            dtype=np.float32,
            layer_type=name.split('.')[-1],
            model_name="demo_transformer",
        )
        weights[name] = WeightTensor(data=data, metadata=metadata)
    
    return weights


def demo_coral_to_safetensors():
    """Demonstrate exporting Coral weights to Safetensors."""
    print("=== Demo 1: Coral → Safetensors Export ===\n")
    
    # Create a Coral repository
    repo_path = Path("demo_coral_repo")
    if repo_path.exists():
        shutil.rmtree(repo_path)
    
    repo = Repository(repo_path, init=True)
    print(f"✓ Created Coral repository at {repo_path}")
    
    # Create and stage model weights
    weights = create_sample_model_weights()
    repo.stage_weights(weights)
    repo.commit("Initial model checkpoint")
    print(f"✓ Added {len(weights)} weights to repository")
    
    # Export to Safetensors
    output_path = Path("exported_model.safetensors")
    convert_coral_to_safetensors(
        source=repo,
        output_path=output_path,
        include_metadata=True,
        custom_metadata={
            "model_type": "transformer",
            "num_parameters": sum(w.size for w in weights.values()),
            "training_stage": "pretrained",
        }
    )
    
    print(f"✓ Exported to {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Verify the export
    reader = SafetensorsReader(output_path)
    print(f"✓ Safetensors file contains {len(reader.get_tensor_names())} tensors")
    print(f"✓ Metadata: {dict(list(reader.metadata.items())[:5])}...")  # Show first 5 items
    
    return repo_path, output_path


def demo_safetensors_to_coral():
    """Demonstrate importing Safetensors to Coral."""
    print("\n=== Demo 2: Safetensors → Coral Import ===\n")
    
    # Create a Safetensors file to import
    safetensors_path = Path("model_to_import.safetensors")
    writer = SafetensorsWriter(
        safetensors_path,
        metadata={
            "model_name": "imported_model",
            "version": "1.0",
            "framework": "pytorch",
        }
    )
    
    # Add some tensors
    for i in range(3):
        data = np.random.randn(10, 10).astype(np.float32)
        writer.add_tensor(f"layer_{i}.weight", data)
        writer.add_tensor(f"layer_{i}.bias", np.zeros(10, dtype=np.float32))
    
    writer.write()
    print(f"✓ Created Safetensors file: {safetensors_path}")
    
    # Import to new Coral repository
    import_repo_path = Path("imported_coral_repo")
    if import_repo_path.exists():
        shutil.rmtree(import_repo_path)
    
    weight_mapping = convert_safetensors_to_coral(
        source_path=safetensors_path,
        target=import_repo_path,
        preserve_metadata=True,
    )
    
    print(f"✓ Imported {len(weight_mapping)} weights to {import_repo_path}")
    
    # Verify the import
    imported_repo = Repository(import_repo_path)
    for name in ["layer_0.weight", "layer_1.bias", "layer_2.weight"]:
        weight = imported_repo.get_weight(name)
        if weight:
            print(f"✓ Successfully imported {name}: shape={weight.shape}, dtype={weight.dtype}")
    
    return import_repo_path


def demo_batch_conversion():
    """Demonstrate batch converting multiple Safetensors files."""
    print("\n=== Demo 3: Batch Conversion ===\n")
    
    # Create directory structure with multiple models
    models_dir = Path("safetensors_models")
    if models_dir.exists():
        shutil.rmtree(models_dir)
    
    # Create subdirectories for different model types
    for model_type in ["bert", "gpt", "vit"]:
        type_dir = models_dir / model_type
        type_dir.mkdir(parents=True)
        
        # Create multiple checkpoints
        for checkpoint in range(2):
            file_path = type_dir / f"checkpoint_{checkpoint}.safetensors"
            writer = SafetensorsWriter(
                file_path,
                metadata={
                    "model_type": model_type,
                    "checkpoint": checkpoint,
                    "epoch": checkpoint * 10,
                }
            )
            
            # Add model-specific tensors
            if model_type == "bert":
                writer.add_tensor("embeddings.word_embeddings.weight", 
                                np.random.randn(30522, 768).astype(np.float32) * 0.02)
                writer.add_tensor("encoder.layer.0.attention.self.query.weight",
                                np.random.randn(768, 768).astype(np.float32) * 0.02)
            elif model_type == "gpt":
                writer.add_tensor("wte.weight",
                                np.random.randn(50257, 768).astype(np.float32) * 0.02)
                writer.add_tensor("h.0.attn.c_attn.weight",
                                np.random.randn(768, 2304).astype(np.float32) * 0.02)
            else:  # vit
                writer.add_tensor("patch_embed.proj.weight",
                                np.random.randn(768, 3, 16, 16).astype(np.float32) * 0.02)
                writer.add_tensor("blocks.0.attn.qkv.weight",
                                np.random.randn(2304, 768).astype(np.float32) * 0.02)
            
            writer.write()
    
    print(f"✓ Created model directory structure:")
    for path in models_dir.rglob("*.safetensors"):
        print(f"  - {path.relative_to(models_dir)}")
    
    # Batch convert to Coral
    batch_repo_path = Path("batch_coral_repo")
    if batch_repo_path.exists():
        shutil.rmtree(batch_repo_path)
    
    results = batch_convert_safetensors(
        source_dir=models_dir,
        target=batch_repo_path,
        recursive=True,
        preserve_structure=True,
    )
    
    print(f"\n✓ Batch converted {len(results)} files")
    
    # Show conversion results
    batch_repo = Repository(batch_repo_path)
    print("\n✓ Repository contains weights from:")
    for file_path, mapping in results.items():
        print(f"  - {Path(file_path).name}: {len(mapping)} weights")
    
    # Check deduplication stats
    stats = batch_repo.deduplicator.compute_stats()
    if stats.total_weights > stats.unique_weights:
        saved = stats.total_weights - stats.unique_weights
        print(f"\n✓ Deduplication saved {saved} duplicate weights "
              f"({saved/stats.total_weights*100:.1f}% reduction)")
    
    return batch_repo_path


def cleanup_demo_files():
    """Clean up demonstration files."""
    print("\n=== Cleanup ===\n")
    
    paths_to_clean = [
        "demo_coral_repo",
        "exported_model.safetensors",
        "model_to_import.safetensors",
        "imported_coral_repo",
        "safetensors_models",
        "batch_coral_repo",
    ]
    
    for path in paths_to_clean:
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_dir():
                shutil.rmtree(path_obj)
            else:
                path_obj.unlink()
            print(f"✓ Cleaned up {path}")


def main():
    """Run all demonstrations."""
    print("🔄 Coral ↔️ Safetensors Conversion Demo")
    print("=" * 50)
    
    try:
        # Run demonstrations
        demo_coral_to_safetensors()
        demo_safetensors_to_coral()
        demo_batch_conversion()
        
        print("\n✨ All demonstrations completed successfully!")
        
        # Auto cleanup in non-interactive mode
        import sys
        if sys.stdin.isatty():
            # Interactive mode
            print("\nWould you like to clean up demo files? (y/n): ", end="")
            response = input().strip().lower()
            if response == 'y':
                cleanup_demo_files()
            else:
                print("\n💡 Demo files preserved for inspection.")
                print("   Run cleanup_demo_files() to remove them later.")
        else:
            # Non-interactive mode - auto cleanup
            print("\n🧹 Auto-cleaning demo files (non-interactive mode)...")
            cleanup_demo_files()
    
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()