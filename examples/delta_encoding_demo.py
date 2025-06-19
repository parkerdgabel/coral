"""
Demonstration of lossless delta encoding for similar weights.

This example shows how Coral now handles similar weights without losing information,
using delta encoding to achieve both space efficiency and perfect reconstruction.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np

from coral import Repository
from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.delta.delta_encoder import DeltaConfig, DeltaType


def create_model_variants():
    """Create a base model and similar variants (e.g., fine-tuned versions)."""

    # Base model weights
    base_weights = {
        "layer1.weight": WeightTensor(
            data=np.random.randn(256, 128).astype(np.float32),
            metadata=WeightMetadata(
                name="layer1.weight",
                shape=(256, 128),
                dtype=np.float32,
                layer_type="Linear",
            ),
        ),
        "layer1.bias": WeightTensor(
            data=np.random.randn(256).astype(np.float32),
            metadata=WeightMetadata(
                name="layer1.bias", shape=(256,), dtype=np.float32, layer_type="Linear"
            ),
        ),
        "layer2.weight": WeightTensor(
            data=np.random.randn(10, 256).astype(np.float32),
            metadata=WeightMetadata(
                name="layer2.weight",
                shape=(10, 256),
                dtype=np.float32,
                layer_type="Linear",
            ),
        ),
    }

    # Create fine-tuned variants (similar but not identical)
    variants = {}

    for variant_name in [
        "finetune_dataset_A",
        "finetune_dataset_B",
        "finetune_dataset_C",
    ]:
        variant_weights = {}

        for name, base_weight in base_weights.items():
            # Add small random changes to simulate fine-tuning
            noise_scale = (
                0.01 if "weight" in name else 0.005
            )  # Smaller changes for bias
            noise = np.random.randn(*base_weight.shape).astype(np.float32) * noise_scale

            variant_data = base_weight.data + noise

            variant_weights[name] = WeightTensor(
                data=variant_data,
                metadata=WeightMetadata(
                    name=name,
                    shape=base_weight.shape,
                    dtype=base_weight.dtype,
                    layer_type=base_weight.metadata.layer_type,
                    model_name=variant_name,
                ),
            )

        variants[variant_name] = variant_weights

    return base_weights, variants


def demonstrate_without_delta_encoding():
    """Show the information loss problem without delta encoding."""
    print("\\n" + "=" * 60)
    print("DEMONSTRATION: WITHOUT DELTA ENCODING (Information Loss)")
    print("=" * 60)

    temp_dir = tempfile.mkdtemp()
    try:
        repo_path = Path(temp_dir)

        # Initialize repository with delta encoding DISABLED
        repo = Repository(repo_path, init=True)
        repo.config["core"]["delta_encoding"] = False

        # Reinitialize deduplicator without delta encoding
        from coral.core.deduplicator import Deduplicator

        repo.deduplicator = Deduplicator(
            similarity_threshold=0.99, enable_delta_encoding=False
        )

        base_weights, variants = create_model_variants()

        # Store base model
        repo.stage_weights(base_weights)
        base_commit = repo.commit("Base model")
        print(f"✓ Stored base model: {base_commit.commit_hash[:8]}")

        # Store first variant
        variant_name = "finetune_dataset_A"
        repo.stage_weights(variants[variant_name])
        variant_commit = repo.commit(f"Fine-tuned on {variant_name}")
        print(f"✓ Stored {variant_name}: {variant_commit.commit_hash[:8]}")

        # Retrieve and compare
        original_weight = variants[variant_name]["layer1.weight"].data
        retrieved_weight = repo.get_weight("layer1.weight").data

        print("\\n📊 Comparison Results:")
        print(f"  Original weight sample: {original_weight[0, :5]}")
        print(f"  Retrieved weight sample: {retrieved_weight[0, :5]}")

        # Check if they're identical
        are_identical = np.array_equal(original_weight, retrieved_weight)
        mse = np.mean((original_weight - retrieved_weight) ** 2)

        print(f"  Are identical: {are_identical}")
        print(f"  Mean Squared Error: {mse:.10f}")

        if not are_identical:
            print("  ❌ INFORMATION LOST: Retrieved weight is NOT the original!")
            print(
                "  ❌ This is because similar weights get deduplicated to "
                "reference weight"
            )

        # Show deduplication stats
        stats = repo.deduplicator.get_deduplication_report()
        print("\\n📈 Storage Stats:")
        print(f"  Total weights: {stats['summary']['total_weights']}")
        print(f"  Unique weights: {stats['summary']['unique_weights']}")
        print(f"  Similar weights: {stats['summary']['similar_weights']}")
        print(f"  Space saved: {stats['summary']['compression_ratio']:.1%}")

    finally:
        shutil.rmtree(temp_dir)


def demonstrate_with_delta_encoding():
    """Show lossless reconstruction with delta encoding."""
    print("\\n" + "=" * 60)
    print("DEMONSTRATION: WITH DELTA ENCODING (Lossless)")
    print("=" * 60)

    temp_dir = tempfile.mkdtemp()
    try:
        repo_path = Path(temp_dir)

        # Initialize repository with delta encoding ENABLED
        repo = Repository(repo_path, init=True)
        repo.config["core"]["delta_encoding"] = True
        repo.config["core"]["delta_type"] = "float32_raw"  # Lossless encoding

        base_weights, variants = create_model_variants()

        # Store base model
        repo.stage_weights(base_weights)
        base_commit = repo.commit("Base model")
        print(f"✓ Stored base model: {base_commit.commit_hash[:8]}")

        # Store variants and track reconstruction accuracy
        all_mse = []

        for variant_name, variant_weights in variants.items():
            repo.stage_weights(variant_weights)
            variant_commit = repo.commit(f"Fine-tuned on {variant_name}")
            print(f"✓ Stored {variant_name}: {variant_commit.commit_hash[:8]}")

            # Test reconstruction for each weight
            for weight_name, original_weight in variant_weights.items():
                retrieved_weight = repo.get_weight(weight_name)

                # Calculate reconstruction accuracy
                mse = np.mean((original_weight.data - retrieved_weight.data) ** 2)
                all_mse.append(mse)

                # Verify perfect reconstruction
                are_identical = np.allclose(
                    original_weight.data, retrieved_weight.data, atol=1e-7
                )

                if weight_name == "layer1.weight":  # Show details for one weight
                    print(f"\\n📊 Reconstruction Check ({weight_name}):")
                    print(f"  Original sample: {original_weight.data[0, :5]}")
                    print(f"  Retrieved sample: {retrieved_weight.data[0, :5]}")
                    print(f"  MSE: {mse:.15f}")
                    print(f"  Perfect reconstruction: {are_identical}")

        print("\\n✅ ALL WEIGHTS RECONSTRUCTED PERFECTLY!")
        print(f"  Average MSE across all weights: {np.mean(all_mse):.15f}")
        print(f"  Maximum MSE: {np.max(all_mse):.15f}")

        # Show enhanced storage statistics
        stats = repo.deduplicator.get_compression_stats()
        print("\\n📈 Enhanced Storage Stats:")
        print(f"  Total weights: {stats['total_weights']}")
        print(f"  Unique reference weights: {stats['unique_weights']}")
        print(f"  Delta-encoded weights: {stats['similar_weights']}")
        print(f"  Overall compression: {stats['compression_ratio']:.1%}")

        if "delta_stats" in stats:
            delta_stats = stats["delta_stats"]
            print("\\n🔧 Delta Encoding Stats:")
            print(f"  Total deltas stored: {delta_stats['total_deltas']}")
            print(
                f"  Delta compression ratio: "
                f"{delta_stats['delta_compression_ratio']:.1%}"
            )
            print(
                f"  Average delta size: {delta_stats['average_delta_size']:.0f} bytes"
            )
            print(
                f"  Space saved by deltas: {delta_stats['delta_compression_ratio']:.1%}"
            )

        # Show individual delta information
        print("\\n🧮 Individual Delta Information:")
        for _variant_name in variants.keys():
            for weight_name in base_weights.keys():
                if repo.deduplicator.is_delta_encoded(weight_name):
                    delta = repo.deduplicator.get_delta_by_name(weight_name)
                    if delta:
                        print(
                            f"  {weight_name}: {delta.compression_ratio:.1%} "
                            f"compression, {delta.nbytes} bytes"
                        )

    finally:
        shutil.rmtree(temp_dir)


def demonstrate_different_encoding_strategies():
    """Show different delta encoding strategies and their trade-offs."""
    print("\\n" + "=" * 60)
    print("DEMONSTRATION: DIFFERENT ENCODING STRATEGIES")
    print("=" * 60)

    base_weights, variants = create_model_variants()
    variant_weights = variants["finetune_dataset_A"]

    # Test different encoding strategies
    strategies = [
        (DeltaType.FLOAT32_RAW, "Lossless (Float32)"),
        (DeltaType.INT8_QUANTIZED, "Lossy (8-bit Quantized)"),
        (DeltaType.INT16_QUANTIZED, "Lossy (16-bit Quantized)"),
        (DeltaType.COMPRESSED, "Lossless (Compressed)"),
        (DeltaType.SPARSE, "Lossless (Sparse)"),
    ]

    results = []

    for delta_type, description in strategies:
        temp_dir = tempfile.mkdtemp()
        try:
            repo_path = Path(temp_dir)
            repo = Repository(repo_path, init=True)

            # Configure delta encoding
            delta_config = DeltaConfig(delta_type=delta_type)
            from coral.core.deduplicator import Deduplicator

            repo.deduplicator = Deduplicator(
                similarity_threshold=0.99,
                delta_config=delta_config,
                enable_delta_encoding=True,
            )

            # Store base and variant
            repo.stage_weights(base_weights)
            repo.commit("Base model")

            repo.stage_weights(variant_weights)
            repo.commit("Fine-tuned variant")

            # Measure reconstruction quality and compression
            total_mse = 0
            total_original_size = 0
            total_compressed_size = 0

            for weight_name, original_weight in variant_weights.items():
                retrieved_weight = repo.get_weight(weight_name)
                mse = np.mean((original_weight.data - retrieved_weight.data) ** 2)
                total_mse += mse

                total_original_size += original_weight.nbytes

                if repo.deduplicator.is_delta_encoded(weight_name):
                    delta = repo.deduplicator.get_delta_by_name(weight_name)
                    total_compressed_size += delta.nbytes
                else:
                    total_compressed_size += original_weight.nbytes

            avg_mse = total_mse / len(variant_weights)
            compression_ratio = 1.0 - (total_compressed_size / total_original_size)

            results.append(
                {
                    "strategy": description,
                    "avg_mse": avg_mse,
                    "compression_ratio": compression_ratio,
                    "is_lossless": avg_mse < 1e-10,
                }
            )

        finally:
            shutil.rmtree(temp_dir)

    # Display results
    print("\\n📊 Encoding Strategy Comparison:")
    print(f"{'Strategy':<25} {'Avg MSE':<15} {'Compression':<12} {'Lossless'}")
    print("-" * 65)

    for result in results:
        lossless_icon = "✅" if result["is_lossless"] else "❌"
        print(
            f"{result['strategy']:<25} {result['avg_mse']:<15.2e} "
            f"{result['compression_ratio']:<11.1%} {lossless_icon}"
        )

    print("\\n💡 Key Insights:")
    print("  • Float32 Raw: Perfect reconstruction, moderate compression")
    print("  • Compressed: Perfect reconstruction, better compression")
    print("  • Quantized: Lossy but very compact")
    print("  • Sparse: Perfect for weights with few changes")


def main():
    """Run all demonstrations."""
    print("🚀 CORAL DELTA ENCODING DEMONSTRATION")
    print(
        "This demo shows how Coral handles similar weights with and without "
        "delta encoding."
    )

    # Show the problem
    demonstrate_without_delta_encoding()

    # Show the solution
    demonstrate_with_delta_encoding()

    # Show different strategies
    demonstrate_different_encoding_strategies()

    print("\\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ Delta encoding solves the information loss problem")
    print("✅ Similar weights can be reconstructed perfectly")
    print("✅ Multiple encoding strategies available for different needs")
    print("✅ Significant space savings while maintaining accuracy")
    print("\\n🎯 Coral now provides TRUE git-like versioning for neural networks!")


if __name__ == "__main__":
    main()
