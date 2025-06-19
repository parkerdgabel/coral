"""Final 4% coverage push - very targeted tests."""

import tempfile
from pathlib import Path

import numpy as np

# Import all modules to ensure coverage
from coral.cli.main import CoralCLI
from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.integrations.pytorch import PyTorchIntegration
from coral.storage.hdf5_store import HDF5Store
from coral.storage.weight_store import WeightStore
from coral.training.checkpoint_manager import CheckpointConfig
from coral.version_control.commit import CommitMetadata
from coral.version_control.repository import Repository
from coral.version_control.version import Version


class TestCoverageFinal4Percent:
    """Very targeted tests to reach 80% coverage."""

    def test_cli_command_execution_paths(self):
        """Test CLI command execution paths."""
        cli = CoralCLI()

        # Test that CLI has expected attributes
        assert hasattr(cli, "parser")
        assert hasattr(cli, "run")

        # Test subparsers exist
        subparsers_actions = [
            action
            for action in cli.parser._actions
            if isinstance(action, type(cli.parser._subparsers_action))
        ]
        assert len(subparsers_actions) > 0

    def test_hdf5_store_metadata_operations(self):
        """Test HDF5Store metadata operations."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            store_path = tmp.name

        try:
            store = HDF5Store(store_path)

            # Test store initialization creates groups
            assert store._file is not None
            assert "weights" in store._file
            assert "deltas" in store._file
            assert "metadata" in store._file

            # Test get_metadata for non-existent
            meta = store.get_metadata("non-existent")
            assert meta is None

            # Test get_storage_info
            info = store.get_storage_info()
            assert isinstance(info, dict)
            assert "compression" in info
            assert info["compression"] == "gzip"

            store.close()

        finally:
            Path(store_path).unlink(missing_ok=True)

    def test_repository_internal_methods(self):
        """Test Repository internal helper methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(tmpdir, init=True)

            # Test HEAD file operations
            head_file = repo.coral_dir / "HEAD"
            assert head_file.exists()

            # Test branch ref path
            branch_ref = repo._get_branch_ref_path("test-branch")
            assert branch_ref == repo.coral_dir / "refs" / "heads" / "test-branch"

            # Test commit file path
            commit_path = repo._get_commit_path("abc123")
            assert commit_path == repo.coral_dir / "objects" / "commits" / "abc123.json"

            # Test version file path
            version_path = repo._get_version_path("v1.0")
            assert version_path == repo.coral_dir / "refs" / "tags" / "v1.0.json"

    def test_commit_metadata_edge_cases(self):
        """Test CommitMetadata edge cases."""
        # Test with None/empty values where allowed
        metadata = CommitMetadata(
            message="",  # Empty message
            author="A",  # Short author
            email="a@b",  # Minimal email
        )

        assert metadata.message == ""
        assert metadata.author == "A"
        assert metadata.email == "a@b"

        # Test dict conversion preserves all fields
        meta_dict = metadata.to_dict()
        assert "timestamp" in meta_dict
        assert "tags" in meta_dict

    def test_version_equality_and_repr(self):
        """Test Version equality and string representation."""
        v1 = Version("v1.0", "id1", "hash1", "First")
        v2 = Version("v1.0", "id1", "hash1", "First")
        v3 = Version("v2.0", "id2", "hash2", "Second")

        # Test equality
        assert v1 == v2  # Same values
        assert v1 != v3  # Different values

        # String representation
        str_repr = str(v1)
        assert "v1.0" in str_repr

        # Test dict roundtrip
        v1_dict = v1.to_dict()
        v1_restored = Version.from_dict(v1_dict)
        assert v1_restored.name == v1.name
        assert v1_restored.version_id == v1.version_id

    def test_pytorch_integration_is_available(self):
        """Test PyTorchIntegration availability check."""
        # Test the TORCH_AVAILABLE flag
        from coral.integrations.pytorch import TORCH_AVAILABLE

        # Flag should be boolean
        assert isinstance(TORCH_AVAILABLE, bool)

        # Test class creation works regardless
        integration = PyTorchIntegration()
        assert integration is not None

    def test_weight_store_interface_completeness(self):
        """Test WeightStore abstract interface is complete."""
        # Check all abstract methods are defined
        abstract_methods = WeightStore.__abstractmethods__
        expected_methods = {
            "store",
            "load",
            "exists",
            "list_weights",
            "get_metadata",
            "get_storage_info",
            "store_batch",
            "load_batch",
        }
        assert abstract_methods == expected_methods

    def test_checkpoint_config_defaults(self):
        """Test CheckpointConfig with defaults."""
        config = CheckpointConfig()

        # Test default values
        assert config.save_every_n_epochs == 1
        assert config.keep_last_n_checkpoints == 5
        assert config.save_on_best_metric is None

        # Test with custom values
        config2 = CheckpointConfig(
            save_every_n_epochs=10,
            keep_last_n_checkpoints=3,
            save_on_best_metric="val_loss",
            minimize_metric=True,
        )
        assert config2.save_every_n_epochs == 10
        assert config2.save_on_best_metric == "val_loss"

    def test_repository_log_method(self):
        """Test Repository log method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(tmpdir, init=True)

            # Add some commits
            weight = WeightTensor(
                data=np.ones(3, dtype=np.float32),
                metadata=WeightMetadata(name="w1", shape=(3,), dtype=np.float32),
            )

            # First commit
            repo.stage_weights({"w1": weight})
            repo.commit("First commit")

            # Second commit
            weight2 = WeightTensor(
                data=np.ones(3, dtype=np.float32) * 2,
                metadata=WeightMetadata(name="w2", shape=(3,), dtype=np.float32),
            )
            repo.stage_weights({"w2": weight2})
            repo.commit("Second commit")

            # Test log
            log = repo.log(max_commits=10)
            assert len(log) == 2
            assert log[0].metadata.message == "Second commit"
            assert log[1].metadata.message == "First commit"

            # Test log with limit
            log_limited = repo.log(max_commits=1)
            assert len(log_limited) == 1

    def test_hdf5_store_exists_method(self):
        """Test HDF5Store exists method."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            store_path = tmp.name

        try:
            store = HDF5Store(store_path)

            # Test non-existent
            assert not store.exists("non-existent-hash")

            # Store a weight
            weight = WeightTensor(
                data=np.array([1, 2, 3], dtype=np.float32),
                metadata=WeightMetadata(name="test", shape=(3,), dtype=np.float32),
            )
            hash_key = weight.compute_hash()
            store.store(hash_key, weight)

            # Test exists
            assert store.exists(hash_key)

            store.close()

        finally:
            Path(store_path).unlink(missing_ok=True)
