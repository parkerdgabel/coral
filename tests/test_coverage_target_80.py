"""Targeted tests to reach 80% coverage."""

import datetime
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from coral.cli.main import CoralCLI
from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.integrations.pytorch import PyTorchIntegration
from coral.storage.hdf5_store import HDF5Store
from coral.storage.weight_store import WeightStore
from coral.version_control.commit import CommitMetadata
from coral.version_control.repository import Repository
from coral.version_control.version import Version


class TestTargetedCoverage80:
    """Targeted tests to reach 80% coverage."""

    def test_cli_parser_commands(self):
        """Test CLI parser command structure."""
        cli = CoralCLI()

        # Test that parser has subcommands
        assert cli.parser is not None
        assert hasattr(cli.parser, "_subparsers")

        # Test parsing init command
        args = cli.parser.parse_args(["init"])
        assert args.command == "init"

        # Test parsing add command
        args = cli.parser.parse_args(["add", "model.pth"])
        assert args.command == "add"
        assert args.path == "model.pth"

        # Test parsing commit command
        args = cli.parser.parse_args(["commit", "-m", "test message"])
        assert args.command == "commit"
        assert args.message == "test message"

        # Test parsing status command
        args = cli.parser.parse_args(["status"])
        assert args.command == "status"

    def test_hdf5_store_init_with_compression(self):
        """Test HDF5Store initialization with different compression options."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            store_path = tmp.name

        try:
            # Test default compression
            store1 = HDF5Store(store_path)
            assert store1.compression == "gzip"
            store1.close()

            # Test custom compression
            Path(store_path).unlink()
            store2 = HDF5Store(store_path, compression="lzf")
            assert store2.compression == "lzf"
            store2.close()

        finally:
            Path(store_path).unlink(missing_ok=True)

    def test_hdf5_store_put_and_get(self):
        """Test HDF5Store put and get operations."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            store_path = tmp.name

        try:
            store = HDF5Store(store_path)

            # Test put weight
            weight = WeightTensor(
                data=np.array([1, 2, 3], dtype=np.float32),
                metadata=WeightMetadata(name="test", shape=(3,), dtype=np.float32),
            )
            hash_key = weight.compute_hash()
            store.put(hash_key, weight)

            # Test contains
            assert store.contains(hash_key)

            # Test get
            retrieved = store.get(hash_key)
            assert retrieved is not None
            np.testing.assert_array_equal(retrieved.data, weight.data)

            # Test list
            keys = store.list()
            assert hash_key in keys

            store.close()

        finally:
            Path(store_path).unlink(missing_ok=True)

    def test_repository_paths_and_dirs(self):
        """Test Repository paths and directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(tmpdir, init=True)

            # Test directory structure
            assert repo.coral_dir.exists()
            assert (repo.coral_dir / "objects").exists()
            assert (repo.coral_dir / "refs" / "heads").exists()
            assert (repo.coral_dir / "logs").exists()

            # Test HEAD file
            head_file = repo.coral_dir / "HEAD"
            assert head_file.exists()

    def test_commit_metadata_full_serialization(self):
        """Test CommitMetadata with all fields and serialization."""
        timestamp = datetime.datetime.now()
        metadata = CommitMetadata(
            message="Test commit",
            author="Test Author",
            email="test@example.com",
            timestamp=timestamp,
            tags=["v1.0", "release"],
            metadata={"key": "value", "build": 123},
        )

        # Test properties
        assert metadata.message == "Test commit"
        assert metadata.author == "Test Author"
        assert metadata.email == "test@example.com"
        assert len(metadata.tags) == 2
        assert metadata.metadata["build"] == 123

        # Test to_dict
        meta_dict = metadata.to_dict()
        assert meta_dict["message"] == "Test commit"
        assert "timestamp" in meta_dict

        # Test from_dict
        metadata2 = CommitMetadata.from_dict(meta_dict)
        assert metadata2.message == metadata.message
        assert metadata2.author == metadata.author

    def test_version_full_serialization(self):
        """Test Version with all fields and serialization."""
        timestamp = datetime.datetime.now()
        version = Version(
            name="v1.0.0",
            version_id="version123",
            commit_hash="commit456",
            description="Major release",
            metrics={"accuracy": 0.95, "loss": 0.05},
            timestamp=timestamp,
            metadata={"framework": "pytorch", "dataset": "imagenet"},
        )

        # Test properties
        assert version.name == "v1.0.0"
        assert version.metrics["accuracy"] == 0.95
        assert version.metadata["framework"] == "pytorch"

        # Test to_dict
        version_dict = version.to_dict()
        assert version_dict["name"] == "v1.0.0"
        assert "timestamp" in version_dict

        # Test from_dict
        version2 = Version.from_dict(version_dict)
        assert version2.name == version.name
        assert version2.description == version.description

    def test_pytorch_integration_model_operations(self):
        """Test PyTorchIntegration model operations."""
        # Mock torch
        with patch("coral.integrations.pytorch.torch") as _mock_torch:
            # Setup mock model
            mock_model = Mock()
            mock_model.state_dict.return_value = {
                "layer1.weight": Mock(numpy=lambda: np.ones((10, 5), dtype=np.float32)),
                "layer1.bias": Mock(numpy=lambda: np.zeros(10, dtype=np.float32)),
            }

            # Test model to weights
            integration = PyTorchIntegration()
            weights = integration.model_to_weights(mock_model, prefix="model")

            assert len(weights) == 2
            assert "model.layer1.weight" in weights
            assert "model.layer1.bias" in weights

    def test_weight_store_abstract_methods(self):
        """Test WeightStore abstract base class."""

        # Create a concrete implementation for testing
        class TestStore(WeightStore):
            def __init__(self):
                self.weights = {}

            def put(self, hash_key, weight):
                self.weights[hash_key] = weight

            def get(self, hash_key):
                return self.weights.get(hash_key)

            def contains(self, hash_key):
                return hash_key in self.weights

            def delete(self, hash_key):
                if hash_key in self.weights:
                    del self.weights[hash_key]

            def list(self):
                return list(self.weights.keys())

            def close(self):
                pass

        # Test implementation
        store = TestStore()
        weight = WeightTensor(
            data=np.array([1, 2, 3], dtype=np.float32),
            metadata=WeightMetadata(name="test", shape=(3,), dtype=np.float32),
        )

        hash_key = "test_hash"
        store.put(hash_key, weight)
        assert store.contains(hash_key)

        retrieved = store.get(hash_key)
        assert retrieved is not None

        keys = store.list()
        assert hash_key in keys

        store.delete(hash_key)
        assert not store.contains(hash_key)

    def test_cli_log_command_parsing(self):
        """Test CLI log command parsing."""
        cli = CoralCLI()

        # Test log with number
        args = cli.parser.parse_args(["log", "-n", "20"])
        assert args.command == "log"
        assert args.number == 20

        # Test log with branch
        args = cli.parser.parse_args(["log", "--branch", "feature"])
        assert args.command == "log"
        assert args.branch == "feature"

    def test_cli_diff_command_parsing(self):
        """Test CLI diff command parsing."""
        cli = CoralCLI()

        # Test diff with refs
        args = cli.parser.parse_args(["diff", "main", "feature"])
        assert args.command == "diff"
        assert args.from_ref == "main"
        assert args.to_ref == "feature"

    def test_cli_tag_command_parsing(self):
        """Test CLI tag command parsing."""
        cli = CoralCLI()

        # Test tag creation
        args = cli.parser.parse_args(["tag", "v1.0.0", "-d", "First release"])
        assert args.command == "tag"
        assert args.name == "v1.0.0"
        assert args.description == "First release"

    def test_repository_get_weight_from_commit(self):
        """Test Repository getting weight from specific commit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(tmpdir, init=True)

            # Add and commit weight
            weight = WeightTensor(
                data=np.array([1, 2, 3], dtype=np.float32),
                metadata=WeightMetadata(name="w1", shape=(3,), dtype=np.float32),
            )
            repo.stage_weights({"w1": weight})
            commit = repo.commit("Initial")

            # Test getting weight with commit ref
            retrieved = repo.get_weight("w1", ref=commit.commit_hash)
            assert retrieved is not None
            np.testing.assert_array_equal(retrieved.data, weight.data)
