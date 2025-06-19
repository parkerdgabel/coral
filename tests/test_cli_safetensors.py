"""Test CLI Safetensors commands."""

import subprocess

import numpy as np

from coral.safetensors.reader import SafetensorsReader
from coral.safetensors.writer import SafetensorsWriter
from coral.version_control.repository import Repository


def run_cli(args: str, cwd: str = None) -> subprocess.CompletedProcess:
    """Run CLI command and return result."""
    cmd = f"python -m coral.cli.main {args}"
    return subprocess.run(cmd.split(), capture_output=True, text=True, cwd=cwd)


class TestCLISafetensors:
    """Test CLI Safetensors commands."""

    def test_import_safetensors(self, tmp_path):
        """Test importing a Safetensors file."""
        # Create test Safetensors file
        st_file = tmp_path / "test.safetensors"
        writer = SafetensorsWriter(st_file, metadata={"test": "metadata"})
        writer.add_tensor("weight1", np.ones((10, 10), dtype=np.float32))
        writer.add_tensor("weight2", np.zeros((5, 5), dtype=np.float32))
        writer.write()

        # Initialize repository
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        Repository(repo_path, init=True)

        # Import file
        result = run_cli(f"import-safetensors {st_file}", cwd=str(repo_path))

        assert result.returncode == 0
        assert "Successfully imported 2 weight(s)" in result.stdout

        # Verify weights were imported
        repo = Repository(repo_path)
        weights = repo.get_all_weights()
        assert len(weights) == 2
        assert "weight1" in weights
        assert "weight2" in weights

    def test_import_safetensors_with_filters(self, tmp_path):
        """Test importing with weight filters."""
        # Create test file
        st_file = tmp_path / "test.safetensors"
        writer = SafetensorsWriter(st_file)
        for i in range(5):
            writer.add_tensor(f"weight{i}", np.ones((3, 3), dtype=np.float32) * i)
        writer.write()

        # Initialize repository
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        Repository(repo_path, init=True)

        # Import specific weights
        result = run_cli(
            f"import-safetensors {st_file} --weights weight1 weight3",
            cwd=str(repo_path),
        )

        assert result.returncode == 0
        assert "Successfully imported 2 weight(s)" in result.stdout

        # Import with exclusions (in a fresh repository)
        repo_path2 = tmp_path / "repo2"
        repo_path2.mkdir()
        Repository(repo_path2, init=True)

        result = run_cli(
            f"import-safetensors {st_file} --exclude weight0 weight4",
            cwd=str(repo_path2),
        )

        assert result.returncode == 0
        assert "Successfully imported 3 weight(s)" in result.stdout

    def test_export_safetensors(self, tmp_path):
        """Test exporting to Safetensors format."""
        # Create repository with weights
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        repo = Repository(repo_path, init=True)

        # Add some weights
        weights = {
            "layer1.weight": np.random.randn(10, 10).astype(np.float32),
            "layer1.bias": np.random.randn(10).astype(np.float32),
            "layer2.weight": np.random.randn(5, 10).astype(np.float32),
        }

        from coral.core.weight_tensor import WeightMetadata, WeightTensor

        weight_tensors = {}
        for name, data in weights.items():
            weight_tensors[name] = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=name,
                    shape=data.shape,
                    dtype=data.dtype,
                ),
            )

        repo.stage_weights(weight_tensors)
        repo.commit("Add test weights")

        # Export all weights
        output_file = tmp_path / "exported.safetensors"
        result = run_cli(f"export-safetensors {output_file}", cwd=str(repo_path))

        assert result.returncode == 0
        assert "Successfully exported 3 weight(s)" in result.stdout
        assert output_file.exists()

        # Verify exported file
        reader = SafetensorsReader(output_file)
        assert len(reader.get_tensor_names()) == 3
        assert set(reader.get_tensor_names()) == set(weights.keys())

    def test_export_safetensors_with_metadata(self, tmp_path):
        """Test exporting with custom metadata."""
        # Create repository
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        repo = Repository(repo_path, init=True)

        # Add a weight
        from coral.core.weight_tensor import WeightMetadata, WeightTensor

        weight = WeightTensor(
            data=np.ones((5, 5), dtype=np.float32),
            metadata=WeightMetadata(
                name="test_weight",
                shape=(5, 5),
                dtype=np.float32,
            ),
        )
        repo.stage_weights({"test_weight": weight})
        repo.commit("Add test weight")

        # Export with metadata
        output_file = tmp_path / "with_metadata.safetensors"
        result = run_cli(
            f"export-safetensors {output_file} "
            "--metadata author=test --metadata version=1.0",
            cwd=str(repo_path),
        )

        assert result.returncode == 0

        # Verify metadata
        reader = SafetensorsReader(output_file)
        assert reader.metadata["author"] == "test"
        assert reader.metadata["version"] == "1.0"
        assert "coral.branch" in reader.metadata

    def test_convert_safetensors_to_npz(self, tmp_path):
        """Test converting Safetensors to NPZ."""
        # Create Safetensors file
        st_file = tmp_path / "input.safetensors"
        writer = SafetensorsWriter(st_file)
        writer.add_tensor("array1", np.ones((10, 10), dtype=np.float32))
        writer.add_tensor("array2", np.zeros((5, 5), dtype=np.float32))
        writer.write()

        # Convert to NPZ
        npz_file = tmp_path / "output.npz"
        result = run_cli(f"convert {st_file} {npz_file}")

        assert result.returncode == 0
        assert "Converted 2 weight(s) to NPZ format" in result.stdout
        assert npz_file.exists()

        # Verify NPZ content
        data = np.load(npz_file)
        assert len(data.files) == 2
        assert "array1" in data.files
        assert "array2" in data.files
        assert data["array1"].shape == (10, 10)
        assert data["array2"].shape == (5, 5)

    def test_convert_npz_to_safetensors(self, tmp_path):
        """Test converting NPZ to Safetensors."""
        # Create NPZ file
        npz_file = tmp_path / "input.npz"
        np.savez_compressed(
            npz_file,
            weight1=np.ones((10, 10), dtype=np.float32),
            weight2=np.zeros((5, 5), dtype=np.float32),
        )

        # Convert to Safetensors
        st_file = tmp_path / "output.safetensors"
        result = run_cli(f"convert {npz_file} {st_file}")

        assert result.returncode == 0
        assert "Converted 2 weight(s) to Safetensors format" in result.stdout
        assert st_file.exists()

        # Verify Safetensors content
        reader = SafetensorsReader(st_file)
        assert len(reader.get_tensor_names()) == 2
        assert "weight1" in reader.get_tensor_names()
        assert "weight2" in reader.get_tensor_names()

    def test_convert_repo_to_safetensors(self, tmp_path):
        """Test converting entire repository to Safetensors."""
        # Create repository with weights
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        repo = Repository(repo_path, init=True)

        # Add weights
        from coral.core.weight_tensor import WeightMetadata, WeightTensor

        weights = {}
        for i in range(3):
            data = np.ones((5, 5), dtype=np.float32) * i
            weights[f"weight{i}"] = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"weight{i}",
                    shape=data.shape,
                    dtype=data.dtype,
                ),
            )

        repo.stage_weights(weights)
        repo.commit("Add weights")

        # Convert repository
        output_file = tmp_path / "repo_export.safetensors"
        result = run_cli(f"convert {repo_path} {output_file}")

        assert result.returncode == 0
        assert "Converted 3 weight(s) to Safetensors format" in result.stdout
        assert output_file.exists()

        # Verify content
        reader = SafetensorsReader(output_file)
        assert len(reader.get_tensor_names()) == 3

    def test_error_handling(self, tmp_path):
        """Test error handling in CLI commands."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        Repository(repo_path, init=True)

        # Test non-existent file
        result = run_cli(
            "import-safetensors nonexistent.safetensors", cwd=str(repo_path)
        )
        assert result.returncode == 1
        assert "File not found" in result.stderr

        # Test invalid metadata format
        result = run_cli(
            "export-safetensors out.st --metadata invalid_format", cwd=str(repo_path)
        )
        assert result.returncode == 1
        assert "Invalid metadata format" in result.stderr

        # Test unsupported conversion
        dummy_file = tmp_path / "dummy.txt"
        dummy_file.write_text("dummy")
        result = run_cli(f"convert {dummy_file} output.safetensors")
        assert result.returncode == 1
        assert "Unsupported conversion" in result.stderr
