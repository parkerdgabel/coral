"""Final 3% to reach 80% coverage."""

import datetime
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Import all key modules
from coral.cli.main import CoralCLI
from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.integrations.pytorch import TORCH_AVAILABLE, PyTorchIntegration
from coral.storage.hdf5_store import HDF5Store
from coral.version_control.branch import BranchManager
from coral.version_control.commit import Commit, CommitMetadata
from coral.version_control.repository import Repository
from coral.version_control.version import Version


class TestRemaining3Percent:
    """Tests for remaining 3% coverage."""

    def test_repository_comprehensive_workflow(self):
        """Test comprehensive repository workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize
            repo = Repository(tmpdir, init=True)

            # Test all path methods
            assert repo._get_objects_dir() == repo.coral_dir / "objects"
            assert repo._get_refs_dir() == repo.coral_dir / "refs"
            assert repo._get_heads_dir() == repo.coral_dir / "refs" / "heads"
            assert repo._get_tags_dir() == repo.coral_dir / "refs" / "tags"

            # Create multiple commits
            for i in range(3):
                weight = WeightTensor(
                    data=np.random.randn(10).astype(np.float32),
                    metadata=WeightMetadata(
                        name=f"w{i}", shape=(10,), dtype=np.float32
                    ),
                )
                repo.stage_weights({f"w{i}": weight})
                repo.commit(f"Commit {i}")

            # Test list operations
            commits = repo._list_commits()
            assert len(commits) >= 3

            # Test branch operations
            branches = repo.list_branches()
            assert any(b.name == "main" for b in branches)

    def test_commit_comprehensive(self):
        """Test Commit class comprehensively."""
        # Test with all fields
        metadata = CommitMetadata(
            message="Full test",
            author="Test User",
            email="test@example.com",
            timestamp=datetime.datetime.now(),
            tags=["v1.0", "stable", "release"],
        )

        commit = Commit(
            commit_hash="fullhash123",
            parent_hashes=["p1", "p2", "p3"],
            weight_hashes={f"w{i}": f"h{i}" for i in range(10)},
            metadata=metadata,
        )

        # Test dict conversion
        commit_dict = commit.to_dict()
        assert commit_dict["commit_hash"] == "fullhash123"
        assert len(commit_dict["parent_hashes"]) == 3
        assert len(commit_dict["weight_hashes"]) == 10

        # Test from_dict
        restored = Commit.from_dict(commit_dict)
        assert restored.commit_hash == commit.commit_hash
        assert len(restored.weight_hashes) == 10

    def test_version_comprehensive(self):
        """Test Version class comprehensively."""
        # Test with all fields including metrics
        version = Version(
            name="v2.5.0",
            version_id="version_full_123",
            commit_hash="commit_abc",
            description="Comprehensive test version",
            metrics={
                "accuracy": 0.98,
                "precision": 0.97,
                "recall": 0.96,
                "f1_score": 0.965,
                "loss": 0.02,
            },
            timestamp=datetime.datetime.now(),
        )

        # Test properties
        assert version.name == "v2.5.0"
        assert len(version.metrics) == 5
        assert version.metrics["accuracy"] == 0.98

        # Test dict conversion
        v_dict = version.to_dict()
        assert "metrics" in v_dict
        assert v_dict["metrics"]["f1_score"] == 0.965

        # Test from_dict
        restored = Version.from_dict(v_dict)
        assert restored.name == version.name
        assert len(restored.metrics) == 5

    def test_pytorch_integration_comprehensive(self):
        """Test PyTorchIntegration comprehensively."""
        # Test availability check
        if TORCH_AVAILABLE:
            integration = PyTorchIntegration()
            assert integration is not None
        else:
            # Test import error
            integration = PyTorchIntegration()
            with pytest.raises(ImportError):
                integration.model_to_weights(None)

    def test_cli_comprehensive_error_handling(self):
        """Test CLI comprehensive error handling."""
        cli = CoralCLI()

        # Test all command error paths
        commands_with_errors = [
            ("add", FileNotFoundError("Model file not found")),
            ("commit", ValueError("No changes staged")),
            ("checkout", ValueError("Branch not found")),
            ("merge", ValueError("Merge conflict")),
            ("tag", ValueError("Tag already exists")),
            ("branch", ValueError("Branch already exists")),
            ("show", ValueError("Weight not found")),
        ]

        for cmd, error in commands_with_errors:
            with patch("coral.cli.main.Repository") as mock_repo_class:
                mock_repo = Mock()
                mock_repo_class.return_value = mock_repo

                # Set up the error
                if cmd == "add":
                    with patch(
                        "coral.cli.main.load_weights_from_path", side_effect=error
                    ):
                        with patch("sys.exit") as mock_exit:
                            with patch("builtins.print"):
                                import argparse

                                args = argparse.Namespace(command=cmd, path="model.pth")

                                cli._run_add(args)
                                mock_exit.assert_called_with(1)
                else:
                    # Generic error setup for other commands
                    method = getattr(mock_repo, cmd if cmd != "tag" else "tag_version")
                    method.side_effect = error

                    with patch("sys.exit") as mock_exit:
                        with patch("builtins.print"):
                            # Create appropriate args for each command
                            if cmd == "commit":
                                args = argparse.Namespace(
                                    command=cmd, message="test", author=None, email=None
                                )
                            elif cmd == "checkout":
                                args = argparse.Namespace(command=cmd, target="branch")
                            elif cmd == "merge":
                                args = argparse.Namespace(command=cmd, branch="feature")
                            elif cmd == "tag":
                                args = argparse.Namespace(
                                    command=cmd, name="v1.0", description="desc"
                                )
                            elif cmd == "branch":
                                args = argparse.Namespace(
                                    command=cmd, name="new", delete=None, list=False
                                )
                            elif cmd == "show":
                                args = argparse.Namespace(
                                    command=cmd, weight_name="weight"
                                )

                            # Run command
                            method_name = f"_run_{cmd}"
                            if hasattr(cli, method_name):
                                getattr(cli, method_name)(args)
                                mock_exit.assert_called_with(1)

    def test_hdf5_store_edge_cases(self):
        """Test HDF5Store edge cases."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            store_path = tmp.name

        try:
            store = HDF5Store(store_path)

            # Test empty operations
            assert store.list_weights() == []

            # Test metadata for non-existent
            meta = store.get_metadata("non-existent")
            assert meta is None

            # Test exists for non-existent
            assert not store.exists("non-existent")

            # Store and verify
            weight = WeightTensor(
                data=np.array([1, 2, 3], dtype=np.float32),
                metadata=WeightMetadata(name="test", shape=(3,), dtype=np.float32),
            )
            hash_key = weight.compute_hash()
            store.store(hash_key, weight)

            # Now test exists
            assert store.exists(hash_key)

            # Test list
            weights = store.list_weights()
            assert hash_key in weights

            store.close()

        finally:
            Path(store_path).unlink(missing_ok=True)

    def test_branch_manager_comprehensive(self):
        """Test BranchManager comprehensively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            refs_dir = Path(tmpdir) / "refs" / "heads"
            refs_dir.mkdir(parents=True)

            manager = BranchManager(refs_dir)

            # Create multiple branches
            branches_data = [
                ("main", "commit1"),
                ("develop", "commit2"),
                ("feature/test", "commit3"),
                ("hotfix/urgent", "commit4"),
            ]

            for name, commit in branches_data:
                branch = manager.create_branch(name, commit)
                assert branch.name == name
                assert branch.commit_hash == commit

            # List all branches
            branches = manager.list_branches()
            assert len(branches) == 4

            # Update branch
            manager.update_branch("develop", "commit5")
            develop = manager.get_branch("develop")
            assert develop.commit_hash == "commit5"

            # Delete branch
            manager.delete_branch("hotfix/urgent")
            assert manager.get_branch("hotfix/urgent") is None

            # List after delete
            branches = manager.list_branches()
            assert len(branches) == 3
