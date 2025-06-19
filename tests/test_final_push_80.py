"""Final push to reach 80% coverage - targeting specific gaps."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from coral.cli.main import CoralCLI
from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.integrations.pytorch import PyTorchIntegration
from coral.storage.hdf5_store import HDF5Store
from coral.version_control.commit import Commit, CommitMetadata
from coral.version_control.repository import Repository
from coral.version_control.version import Version


class TestFinalPush80:
    """Final tests targeting specific coverage gaps."""

    def test_cli_main_entry_point(self):
        """Test CLI main entry point and error handling."""
        cli = CoralCLI()

        # Test with no arguments
        with pytest.raises(SystemExit):
            cli.parser.parse_args([])

        # Test help
        with pytest.raises(SystemExit):
            cli.parser.parse_args(["--help"])

        # Test invalid command
        with pytest.raises(SystemExit):
            cli.parser.parse_args(["invalid-command"])

    def test_cli_run_method(self):
        """Test CLI run method with mocked repository."""
        cli = CoralCLI()

        # Mock sys.argv for init command
        with patch("sys.argv", ["coral", "init"]):
            with patch("coral.version_control.repository.Repository") as _mock_repo:
                with pytest.raises(SystemExit) as exc_info:
                    cli.run()
                # Should exit with 0 on success
                assert exc_info.value.code == 0

    def test_hdf5_store_error_handling(self):
        """Test HDF5Store error handling."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            store_path = tmp.name

        try:
            store = HDF5Store(store_path)

            # Test getting non-existent weight
            result = store.get("non-existent-hash")
            assert result is None

            # Test invalid operations
            with pytest.raises((TypeError, ValueError, AttributeError)):
                store.store(None)  # Should fail with None

            store.close()

        finally:
            Path(store_path).unlink(missing_ok=True)

    def test_repository_error_paths(self):
        """Test Repository error handling paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test loading without init
            with pytest.raises(ValueError, match="not a Coral repository"):
                Repository(tmpdir)

            # Test invalid operations
            repo = Repository(tmpdir, init=True)

            # Test getting non-existent branch
            branch = repo._get_current_branch()
            assert branch is not None

            # Test resolving invalid ref
            with pytest.raises(ValueError):
                repo._resolve_ref("invalid-ref-name")

    def test_commit_metadata_minimal(self):
        """Test CommitMetadata with minimal required fields."""
        metadata = CommitMetadata(
            message="Test", author="Author", email="test@example.com"
        )

        # Should have timestamp even if not provided
        assert metadata.timestamp is not None
        assert metadata.tags == []  # Default empty

    def test_version_dict_operations(self):
        """Test Version to_dict and from_dict."""
        version = Version(
            name="v1.0",
            version_id="v123",
            commit_hash="c456",
            description="Test version",
        )

        # Convert to dict
        v_dict = version.to_dict()
        assert v_dict["name"] == "v1.0"
        assert v_dict["version_id"] == "v123"

        # Convert from dict
        version2 = Version.from_dict(v_dict)
        assert version2.name == version.name
        assert version2.commit_hash == version.commit_hash

    def test_pytorch_integration_error_paths(self):
        """Test PyTorchIntegration error handling."""
        integration = PyTorchIntegration()

        # Test without torch available
        with patch("coral.integrations.pytorch.TORCH_AVAILABLE", False):
            with pytest.raises(ImportError, match="PyTorch is not installed"):
                integration.model_to_weights(None)

            with pytest.raises(ImportError, match="PyTorch is not installed"):
                integration.weights_to_model(None, {})

    def test_hdf5_store_file_operations(self):
        """Test HDF5Store file operations."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            store_path = tmp.name

        try:
            # Create and write
            store = HDF5Store(store_path)

            # Store a weight
            weight = WeightTensor(
                data=np.array([1, 2, 3, 4, 5], dtype=np.float32),
                metadata=WeightMetadata(name="test", shape=(5,), dtype=np.float32),
            )

            hash_key = weight.compute_hash()
            store.store(hash_key, weight)

            # Close and reopen
            store.close()

            # Verify file exists and can be reopened
            assert Path(store_path).exists()
            store2 = HDF5Store(store_path)

            # Verify data persisted
            loaded = store2.load(hash_key)
            assert loaded is not None
            np.testing.assert_array_equal(loaded.data, weight.data)

            store2.close()

        finally:
            Path(store_path).unlink(missing_ok=True)

    def test_repository_staging_operations(self):
        """Test Repository staging area operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(tmpdir, init=True)

            # Test staging area file
            staging_file = repo.coral_dir / "staging.json"

            # Stage a weight
            weight = WeightTensor(
                data=np.array([1, 2, 3], dtype=np.float32),
                metadata=WeightMetadata(name="w1", shape=(3,), dtype=np.float32),
            )

            repo.stage_weights({"w1": weight})

            # Verify staging file exists
            assert staging_file.exists()

            # Test unstaging
            repo._staging_area.clear()
            repo._save_staging_area()

            # Verify empty staging
            status = repo.status()
            assert len(status["staged"]) == 0

    def test_commit_parent_handling(self):
        """Test Commit with different parent configurations."""
        # No parents (initial commit)
        commit1 = Commit(
            commit_hash="c1",
            parent_hashes=[],
            weight_hashes={"w1": "h1"},
            metadata=CommitMetadata("Initial", "Author", "email@test.com"),
        )
        assert len(commit1.parent_hashes) == 0

        # Single parent
        commit2 = Commit(
            commit_hash="c2",
            parent_hashes=["c1"],
            weight_hashes={"w1": "h1", "w2": "h2"},
            metadata=CommitMetadata("Second", "Author", "email@test.com"),
        )
        assert len(commit2.parent_hashes) == 1

        # Multiple parents (merge commit)
        commit3 = Commit(
            commit_hash="c3",
            parent_hashes=["c1", "c2"],
            weight_hashes={"w1": "h1", "w2": "h2", "w3": "h3"},
            metadata=CommitMetadata("Merge", "Author", "email@test.com"),
        )
        assert len(commit3.parent_hashes) == 2

    def test_version_metrics_handling(self):
        """Test Version with and without metrics."""
        # Version without metrics
        v1 = Version(
            name="v1.0", version_id="v1", commit_hash="c1", description="Basic version"
        )
        assert v1.metrics == {}

        # Version with metrics
        v2 = Version(
            name="v2.0",
            version_id="v2",
            commit_hash="c2",
            description="Version with metrics",
            metrics={"accuracy": 0.95, "loss": 0.05, "f1": 0.92},
        )
        assert len(v2.metrics) == 3
        assert v2.metrics["accuracy"] == 0.95
