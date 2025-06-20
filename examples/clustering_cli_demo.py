#!/usr/bin/env python3
"""
Demo script showing clustering CLI commands.

This script demonstrates how to use the clustering CLI commands
to analyze, create, and manage weight clusters in a Coral repository.
"""

import subprocess
import sys
import numpy as np
from pathlib import Path
import tempfile
import json

from coral.version_control.repository import Repository
from coral.core.weight_tensor import WeightTensor, WeightMetadata


def run_command(cmd: str):
    """Run a CLI command and print output."""
    print(f"\n$ {cmd}")
    print("-" * 60)
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr, file=sys.stderr)
    return result.returncode


def create_demo_repository(path: Path):
    """Create a demo repository with sample weights."""
    print("Creating demo repository with sample weights...")
    
    # Initialize repository
    repo = Repository(path, init=True)
    
    # Create some sample weights with clustering potential
    weights = {}
    
    # Group 1: Similar conv weights
    base_conv = np.random.randn(64, 3, 3, 3) * 0.1
    for i in range(5):
        noise = np.random.randn(*base_conv.shape) * 0.001
        weight = WeightTensor(
            data=base_conv + noise,
            metadata=WeightMetadata(
                name=f"conv_layer_{i}",
                shape=base_conv.shape,
                dtype=base_conv.dtype,
                layer_type="conv2d",
                model_name="resnet18"
            )
        )
        weights[f"conv_layer_{i}"] = weight
    
    # Group 2: Similar fc weights
    base_fc = np.random.randn(1000, 512) * 0.05
    for i in range(3):
        noise = np.random.randn(*base_fc.shape) * 0.0005
        weight = WeightTensor(
            data=base_fc + noise,
            metadata=WeightMetadata(
                name=f"fc_layer_{i}",
                shape=base_fc.shape,
                dtype=base_fc.dtype,
                layer_type="linear",
                model_name="resnet18"
            )
        )
        weights[f"fc_layer_{i}"] = weight
    
    # Group 3: Different sized weights
    for i, size in enumerate([128, 256, 512]):
        weight = WeightTensor(
            data=np.random.randn(size, size) * 0.1,
            metadata=WeightMetadata(
                name=f"varied_layer_{i}",
                shape=(size, size),
                dtype=np.float32,
                layer_type="linear"
            )
        )
        weights[f"varied_layer_{i}"] = weight
    
    # Stage and commit
    repo.stage_weights(weights)
    repo.commit("Initial model weights")
    
    print(f"Created repository with {len(weights)} weights")
    return repo


def main():
    """Run clustering CLI demo."""
    print("Coral Clustering CLI Demo")
    print("=" * 60)
    
    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        
        # Create demo repository
        repo = create_demo_repository(repo_path)
        
        # Change to repo directory for CLI commands
        import os
        os.chdir(repo_path)
        
        # 1. Analyze clustering opportunities
        print("\n1. Analyzing clustering opportunities")
        run_command("coral-ml cluster analyze")
        
        # Show different output formats
        print("\n   JSON format:")
        run_command("coral-ml cluster analyze --format json")
        
        print("\n   CSV format:")
        run_command("coral-ml cluster analyze --format csv")
        
        # 2. Check clustering status (before enabling)
        print("\n2. Checking clustering status")
        run_command("coral-ml cluster status")
        
        # 3. Create clusters
        print("\n3. Creating clusters with different strategies")
        
        # Try dry run first
        print("\n   Dry run:")
        run_command("coral-ml cluster create kmeans --dry-run")
        
        # Actually create clusters
        print("\n   Creating clusters:")
        run_command("coral-ml cluster create adaptive --threshold 0.95")
        
        # 4. Check status after clustering
        print("\n4. Checking status after clustering")
        run_command("coral-ml cluster status")
        
        # 5. List clusters
        print("\n5. Listing clusters")
        run_command("coral-ml cluster list")
        run_command("coral-ml cluster list --format json --limit 5")
        run_command("coral-ml cluster list --sort-by size")
        
        # 6. Get cluster info
        print("\n6. Getting cluster information")
        # Note: We need to get actual cluster IDs from the list
        result = subprocess.run(
            ["coral-ml", "cluster", "list", "--format", "json"],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout:
            clusters = json.loads(result.stdout)
            if clusters:
                cluster_id = clusters[0]['id']
                run_command(f"coral-ml cluster info {cluster_id}")
                run_command(f"coral-ml cluster info {cluster_id} --show-stats --show-weights")
        
        # 7. Generate report
        print("\n7. Generating clustering report")
        run_command("coral-ml cluster report")
        run_command("coral-ml cluster report --format csv")
        run_command("coral-ml cluster report --verbose --output cluster_report.txt")
        
        # 8. Optimize clusters
        print("\n8. Optimizing clusters")
        run_command("coral-ml cluster optimize")
        run_command("coral-ml cluster optimize --aggressive --target-reduction 0.5")
        
        # 9. Validate clusters
        print("\n9. Validating clusters")
        run_command("coral-ml cluster validate")
        run_command("coral-ml cluster validate --strict")
        
        # 10. Export/Import configuration
        print("\n10. Exporting and importing cluster configuration")
        run_command("coral-ml cluster export clusters.json")
        run_command("coral-ml cluster export clusters_full.json --include-weights")
        
        # Show exported config
        if Path("clusters.json").exists():
            print("\n   Exported configuration:")
            with open("clusters.json") as f:
                config = json.load(f)
                print(json.dumps(config, indent=2)[:500] + "...")
        
        # 11. Benchmark clustering strategies
        print("\n11. Benchmarking clustering strategies")
        run_command("coral-ml cluster benchmark --strategies kmeans hierarchical --iterations 2")
        
        # 12. Compare clustering (create another commit first)
        print("\n12. Comparing clustering between commits")
        
        # Add more weights and commit
        new_weights = {}
        for i in range(3):
            weight = WeightTensor(
                data=np.random.randn(100, 100) * 0.1,
                metadata=WeightMetadata(
                    name=f"new_layer_{i}",
                    shape=(100, 100),
                    dtype=np.float32
                )
            )
            new_weights[f"new_layer_{i}"] = weight
        
        repo.stage_weights(new_weights)
        repo.commit("Added new layers")
        
        # Re-cluster with new weights
        run_command("coral-ml cluster create kmeans")
        
        # Compare
        run_command("coral-ml cluster compare HEAD~1 HEAD")
        
        # 13. Rebalance clusters
        print("\n13. Rebalancing clusters")
        run_command("coral-ml cluster rebalance --max-cluster-size 5 --min-cluster-size 2")
        
        # 14. Show help
        print("\n14. Getting help")
        subprocess.run(["coral-ml", "cluster", "--help"])
        
        print("\n" + "=" * 60)
        print("Demo completed! The clustering CLI provides comprehensive")
        print("tools for analyzing and managing weight clusters in your")
        print("Coral repository.")


if __name__ == "__main__":
    main()