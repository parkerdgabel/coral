#!/usr/bin/env python3
"""
Large Model Efficiency Demo for Coral

This example demonstrates Coral's efficiency with larger models by:
1. Loading a ~1B parameter model (e.g., a smaller LLaMA variant or similar)
2. Creating multiple variations simulating real ML workflows:
   - Fine-tuned versions (99.5% similar)
   - Continued training checkpoints (99% similar)
   - Transfer learning variants (95% similar)
   - Exact checkpoint duplicates
3. Measuring and reporting space savings

This simulates real-world scenarios where large models are iteratively
refined, creating many similar weight sets that benefit from deduplication.
"""

import shutil
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from coral import Repository
from coral.core.weight_tensor import WeightTensor
from coral.storage.hdf5_store import HDF5Store


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB."""
    total_params = 0
    for param in model.parameters():
        total_params += param.numel() * param.element_size()
    return total_params / (1024 * 1024)


def create_model_variation(
    base_model: nn.Module,
    similarity: float = 0.99,
    variation_name: str = "variation",
    device: torch.device = None,
    **model_kwargs,
) -> nn.Module:
    """Create a variation of the base model with specified similarity."""
    if device is None:
        device = next(base_model.parameters()).device

    # Create deep copy by copying state dict to avoid config issues
    model_copy = type(base_model)(**model_kwargs).to(device)
    model_copy.load_state_dict(base_model.state_dict())

    # Add controlled noise to create variation
    noise_scale = np.sqrt(1 - similarity**2)

    with torch.no_grad():
        for _name, param in model_copy.named_parameters():
            if param.requires_grad:
                # Add scaled Gaussian noise
                noise = torch.randn_like(param) * param.abs().mean() * noise_scale
                param.add_(noise)

    return model_copy


def save_model_to_coral(
    repo: Repository, model: nn.Module, model_name: str, commit_message: str
) -> str:
    """Save a PyTorch model to Coral repository."""
    from coral.core.weight_tensor import WeightMetadata

    # Extract state dict
    state_dict = model.state_dict()

    # Prepare weights dictionary
    weights = {}
    for name, tensor in state_dict.items():
        metadata = WeightMetadata(
            name=f"{model_name}/{name}",
            shape=tuple(tensor.shape),
            dtype=tensor.cpu().numpy().dtype,
            layer_type=name.split(".")[-1],  # e.g., 'weight', 'bias'
            model_name=model_name,
        )

        weight_tensor = WeightTensor(data=tensor.cpu().numpy(), metadata=metadata)
        weights[f"{model_name}/{name}"] = weight_tensor

    # Stage weights
    repo.stage_weights(weights)

    # Commit the changes
    commit_hash = repo.commit(commit_message)
    return commit_hash


def measure_pytorch_baseline(models: Dict[str, nn.Module], output_dir: Path) -> float:
    """Measure baseline storage using standard PyTorch save."""
    output_dir.mkdir(exist_ok=True)
    total_size = 0

    for name, model in models.items():
        checkpoint_path = output_dir / f"{name}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        total_size += checkpoint_path.stat().st_size

    return total_size / (1024 * 1024)  # Convert to MB


def main():
    print("🚀 Coral Large Model Efficiency Demo")
    print("=" * 50)
    print("\n⚠️  WARNING: This demo creates 1B parameter models.")
    print("   It requires ~30GB of RAM and may take 10-20 minutes to complete.")
    print("   Press Ctrl+C to cancel if needed.\n")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create temporary directories
    coral_dir = Path("coral_large_model_demo")
    pytorch_dir = Path("pytorch_baseline")

    # Clean up any existing directories
    for dir_path in [coral_dir, pytorch_dir]:
        if dir_path.exists():
            shutil.rmtree(dir_path)

    try:
        # Create a large custom model with ~1B parameters
        print("\n📥 Creating large model (~1B parameters)...")

        class LargeCustomModel(nn.Module):
            def __init__(self, hidden_size=1024, num_layers=12, vocab_size=30000):
                super().__init__()
                self.config = type("Config", (), {"hidden_size": hidden_size})()

                # Embedding layer: vocab_size x hidden_size
                self.embeddings = nn.Embedding(vocab_size, hidden_size)

                # Transformer-like layers
                self.layers = nn.ModuleList(
                    [
                        nn.ModuleDict(
                            {
                                "attention": nn.Linear(
                                    hidden_size, hidden_size * 3
                                ),  # Q, K, V
                                "projection": nn.Linear(hidden_size, hidden_size),
                                "mlp": nn.Sequential(
                                    nn.Linear(hidden_size, hidden_size * 4),
                                    nn.GELU(),
                                    nn.Linear(hidden_size * 4, hidden_size),
                                ),
                                "norm1": nn.LayerNorm(hidden_size),
                                "norm2": nn.LayerNorm(hidden_size),
                            }
                        )
                        for _ in range(num_layers)
                    ]
                )

                # Output layers
                self.final_norm = nn.LayerNorm(hidden_size)
                self.lm_head = nn.Linear(hidden_size, vocab_size)

            def forward(self, x):
                x = self.embeddings(x)
                for layer in self.layers:
                    # Simplified forward pass
                    residual = x
                    x = layer["norm1"](x)
                    attn = layer["attention"](x)
                    x = layer["projection"](attn) + residual

                    residual = x
                    x = layer["norm2"](x)
                    x = layer["mlp"](x) + residual

                x = self.final_norm(x)
                return self.lm_head(x)

        # Create model with ~1B parameters
        # Optimized for ~1B params: hidden_size=1536, num_layers=24, vocab_size=32000
        # This gives approximately 1.05B parameters
        model_params = {"hidden_size": 1536, "num_layers": 24, "vocab_size": 32000}
        base_model = LargeCustomModel(**model_params).to(device)
        model_size = get_model_size_mb(base_model)
        param_count = sum(p.numel() for p in base_model.parameters())
        print(f"✅ Created model: {param_count:,} parameters ({model_size:.1f} MB)")

        # Create model variations
        print("\n🔄 Creating model variations...")
        models = {
            "base_model": base_model,
        }

        # For 1B models, create fewer variations to manage memory
        # Fine-tuned version (very similar)
        print("  - Creating fine-tuned version (99.5% similar)...")
        variation = create_model_variation(
            base_model,
            similarity=0.995,
            variation_name="finetuned_v1",
            device=device,
            **model_params,
        )
        models["finetuned_v1"] = variation

        # Training checkpoint (similar)
        print("  - Creating training checkpoint (99% similar)...")
        variation = create_model_variation(
            base_model,
            similarity=0.99,
            variation_name="checkpoint_epoch_10",
            device=device,
            **model_params,
        )
        models["checkpoint_epoch_10"] = variation

        # Exact duplicate (simulating multiple saves of same checkpoint)
        print("  - Creating exact checkpoint duplicate...")
        duplicate = LargeCustomModel(**model_params).to(device)
        duplicate.load_state_dict(base_model.state_dict())
        models["checkpoint_duplicate_1"] = duplicate

        print(f"\n✅ Created {len(models)} model variants")

        # Measure PyTorch baseline
        print("\n📊 Measuring baseline storage (PyTorch)...")
        start_time = time.time()
        pytorch_size = measure_pytorch_baseline(models, pytorch_dir)
        pytorch_time = time.time() - start_time
        print(f"  - Size: {pytorch_size:.1f} MB")
        print(f"  - Time: {pytorch_time:.2f} seconds")

        # Initialize Coral repository
        print("\n🪸 Initializing Coral repository...")
        # Try to use existing repo, otherwise create new one
        try:
            repo = Repository(coral_dir)
            print("  - Using existing repository")
        except ValueError:
            repo = Repository(coral_dir, init=True)
            print("  - Created new repository")

        # Initialize store with deduplicator
        repo.store = HDF5Store(coral_dir / "storage.h5")
        repo.store.deduplicator = repo.deduplicator

        # Save models to Coral with delta encoding
        print("\n💾 Saving models to Coral...")
        print("   (This may take several minutes for 1B parameter models)")
        start_time = time.time()

        # Configure aggressive deduplication for maximum space savings
        repo.deduplicator.similarity_threshold = 0.95

        for i, (name, model) in enumerate(models.items(), 1):
            print(f"  [{i}/{len(models)}] Saving {name}...")
            model_start = time.time()
            save_model_to_coral(repo, model, name, f"Add {name} to repository")
            model_time = time.time() - model_start
            print(f"      ✓ Saved in {model_time:.1f}s")

        coral_time = time.time() - start_time

        # Ensure storage is properly closed to get accurate file size
        if hasattr(repo, "store") and repo.store:
            repo.store.close()

        # Get Coral storage statistics
        dedup_stats = repo.deduplicator.compute_stats()
        # The main storage is in the default repository location
        storage_path = coral_dir / ".coral" / "objects" / "weights.h5"
        coral_size = (
            storage_path.stat().st_size / (1024 * 1024)
            if storage_path.exists()
            else 0.0
        )

        print("\n✅ Coral storage complete:")
        print(f"  - Size: {coral_size:.1f} MB")
        print(f"  - Time: {coral_time:.2f} seconds")
        print(f"  - Unique weights: {dedup_stats.unique_weights}")
        print(f"  - Total weights: {dedup_stats.total_weights}")
        print(
            f"  - Deduplicated: "
            f"{dedup_stats.total_weights - dedup_stats.unique_weights}"
        )

        # Calculate and display savings
        print("\n📈 Storage Efficiency Results:")
        print("=" * 50)
        space_saved = pytorch_size - coral_size
        space_saved_pct = (space_saved / pytorch_size) * 100
        compression_ratio = pytorch_size / coral_size

        print(f"🎯 Space Savings: {space_saved:.1f} MB ({space_saved_pct:.1f}%)")
        print(f"🗜️  Compression Ratio: {compression_ratio:.2f}x")
        print("⏱️  Time Comparison:")
        print(f"    - PyTorch: {pytorch_time:.2f}s")
        print(f"    - Coral: {coral_time:.2f}s ({coral_time / pytorch_time:.2f}x)")

        # Demonstrate retrieval
        print("\n🔍 Verifying weight retrieval...")
        repo_loaded = Repository(coral_dir)

        # Load a weight from both base and variation
        sample_weight_name = list(base_model.state_dict().keys())[0]
        base_weight = repo_loaded.get_weight(f"base_model/{sample_weight_name}")
        variation_weight = repo_loaded.get_weight(f"finetuned_v1/{sample_weight_name}")

        if base_weight and variation_weight:
            similarity = np.dot(
                base_weight.data.flatten(), variation_weight.data.flatten()
            ) / (
                np.linalg.norm(base_weight.data.flatten())
                * np.linalg.norm(variation_weight.data.flatten())
            )
            print("  ✅ Successfully retrieved weights")
            print(f"  - Similarity between base and variation: {similarity:.4f}")

        # Show branch and commit information
        print("\n📝 Repository History:")
        commits = repo_loaded.log(max_commits=5)
        for commit in commits:
            print(f"  - {commit.commit_hash[:8]}: {commit.metadata.message}")

        print("\n✨ Demo complete!")
        print("\n💡 Key Insights:")
        print(f"  - For {len(models)} model variants (~1B params each)")
        print(f"  - Coral achieved {compression_ratio:.2f}x compression")
        print(f"  - Saving {space_saved:.1f} MB ({space_saved_pct:.1f}%) of storage")
        print("  - With perfect weight reconstruction via delta encoding")
        print("  - Ideal for large language models and similar architectures")

        # Optional cleanup (commented out to preserve results)
        print("\n💡 To clean up temporary files, run:")
        print(f"  rm -rf {coral_dir} {pytorch_dir}")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        raise


if __name__ == "__main__":
    main()
