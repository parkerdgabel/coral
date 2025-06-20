#!/usr/bin/env python3
"""Demo script to test the new Coral CLI commands for Safetensors support."""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coral.safetensors.writer import SafetensorsWriter


def run_command(cmd: str, cwd: str = None) -> int:
    """Run a command using uv and capture output."""
    full_cmd = f"uv run {cmd}"
    print(f"\n$ {full_cmd}")
    result = subprocess.run(
        full_cmd, shell=True, cwd=cwd, capture_output=True, text=True
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return result.returncode


def create_test_safetensors_file(path: Path) -> None:
    """Create a test Safetensors file with some example weights."""
    # Create some test tensors
    tensors = {
        "model.layer1.weight": np.random.randn(64, 32).astype(np.float32),
        "model.layer1.bias": np.random.randn(64).astype(np.float32),
        "model.layer2.weight": np.random.randn(128, 64).astype(np.float32),
        "model.layer2.bias": np.random.randn(128).astype(np.float32),
        "model.output.weight": np.random.randn(10, 128).astype(np.float32),
        "model.output.bias": np.random.randn(10).astype(np.float32),
    }

    # Add metadata
    metadata = {
        "model_name": "test_model",
        "model_version": "1.0.0",
        "description": "Test model for CLI demo",
        "framework": "pytorch",
    }

    # Write the file
    writer = SafetensorsWriter(path, metadata=metadata)
    for name, data in tensors.items():
        writer.add_tensor(name, data)
    writer.write()

    print(f"Created test Safetensors file: {path}")
    print(f"  - {len(tensors)} tensors")
    print(f"  - Total size: {sum(t.nbytes for t in tensors.values()) / 1024:.1f} KB")


def main():
    """Run the CLI demo."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        print("=== Coral CLI Safetensors Demo ===\n")

        # Create test files
        test_safetensors = temp_path / "test_model.safetensors"
        create_test_safetensors_file(test_safetensors)

        test_npz = temp_path / "test_weights.npz"
        np.savez_compressed(
            test_npz,
            encoder_weight=np.random.randn(256, 128).astype(np.float32),
            encoder_bias=np.random.randn(256).astype(np.float32),
            decoder_weight=np.random.randn(128, 256).astype(np.float32),
            decoder_bias=np.random.randn(128).astype(np.float32),
        )
        print(f"\nCreated test NPZ file: {test_npz}")

        # 1. Initialize a Coral repository
        repo_path = temp_path / "coral_repo"
        repo_path.mkdir()

        print("\n=== 1. Initialize Coral Repository ===")
        run_command("coral-ml init", cwd=str(repo_path))

        # 2. Import Safetensors file
        print("\n=== 2. Import Safetensors File ===")
        run_command(
            f"coral-ml import-safetensors {test_safetensors}", cwd=str(repo_path)
        )

        # Check status
        print("\n=== 3. Check Repository Status ===")
        run_command("coral-ml status", cwd=str(repo_path))
        run_command("coral-ml log --oneline", cwd=str(repo_path))

        # 3. Export specific weights
        print("\n=== 4. Export Specific Weights ===")
        output_file = temp_path / "exported_layers.safetensors"
        run_command(
            f"coral-ml export-safetensors {output_file} "
            "--weights model.layer1.weight model.layer1.bias "
            "--metadata author=coral-demo --metadata exported_from=coral",
            cwd=str(repo_path),
        )

        # 4. Import with exclusions
        print("\n=== 5. Import with Exclusions ===")
        test_safetensors2 = temp_path / "test_model2.safetensors"
        create_test_safetensors_file(test_safetensors2)

        run_command(
            f"coral-ml import-safetensors {test_safetensors2} "
            "--exclude model.output.weight model.output.bias",
            cwd=str(repo_path),
        )

        # 5. Convert NPZ to Safetensors
        print("\n=== 6. Convert NPZ to Safetensors ===")
        converted_st = temp_path / "converted.safetensors"
        run_command(f"coral-ml convert {test_npz} {converted_st}")

        # 6. Convert Safetensors to NPZ
        print("\n=== 7. Convert Safetensors to NPZ ===")
        converted_npz = temp_path / "converted.npz"
        run_command(f"coral-ml convert {test_safetensors} {converted_npz}")

        # 7. Convert Safetensors to HDF5
        print("\n=== 8. Convert Safetensors to HDF5 ===")
        converted_h5 = temp_path / "converted.h5"
        run_command(f"coral-ml convert {test_safetensors} {converted_h5}")

        # 8. Export entire repository
        print("\n=== 9. Export Entire Repository ===")
        full_export = temp_path / "full_export.safetensors"
        run_command(f"coral-ml export-safetensors {full_export}", cwd=str(repo_path))

        # 9. Convert repository to Safetensors
        print("\n=== 10. Convert Repository to Safetensors ===")
        repo_export = temp_path / "repo_export.safetensors"
        run_command(f"coral-ml convert {repo_path} {repo_export}")

        print("\n=== Demo Complete ===")
        print(f"\nAll files created in: {temp_path}")

        # List created files
        print("\nCreated files:")
        for file in sorted(temp_path.rglob("*")):
            if file.is_file():
                size_kb = file.stat().st_size / 1024
                print(f"  - {file.relative_to(temp_path)} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
