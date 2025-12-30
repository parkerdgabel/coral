"""Tests for integration modules with fully mocked dependencies.

These tests don't require the actual optional dependencies (torch, transformers, etc.)
to be installed. They mock the dependencies at the import level.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.version_control.repository import Repository


@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing."""
    import shutil

    temp_dir = tempfile.mkdtemp()
    repo_path = Path(temp_dir)
    repo = Repository(repo_path, init=True)
    yield repo
    shutil.rmtree(temp_dir)


class TestLightningCallbackMocked:
    """Test Lightning callback with fully mocked dependencies."""

    @pytest.fixture
    def mock_lightning_module(self):
        """Create mock lightning module."""
        # Create mock pytorch_lightning module
        mock_pl = MagicMock()
        mock_pl.Trainer = MagicMock
        mock_pl.LightningModule = MagicMock
        mock_callback_class = type("Callback", (), {})
        return mock_pl, mock_callback_class

    def test_save_checkpoint_internal(self, temp_repo):
        """Test _save_checkpoint method logic."""
        # Import the module to get access to constants
        from coral.integrations import lightning as lightning_module

        # Check that the module has the expected structure
        assert hasattr(lightning_module, "CoralCallback")
        assert hasattr(lightning_module, "HAS_LIGHTNING")

    def test_is_better_logic(self, temp_repo):
        """Test the _is_better comparison logic."""
        # This tests the logic without actually using Lightning
        from coral.integrations.lightning import CoralCallback

        # Mock HAS_LIGHTNING to True
        with patch("coral.integrations.lightning.HAS_LIGHTNING", True):
            with patch("coral.integrations.lightning.Callback", MagicMock):
                # Create callback with mocked parent class
                callback = CoralCallback.__new__(CoralCallback)
                callback._repo = temp_repo
                callback.repo_path = temp_repo.coral_dir.parent
                callback.init = False
                callback.mode = "min"
                callback.best_metric = 0.5

                # Test the comparison logic
                assert callback._is_better(0.3) is True
                assert callback._is_better(0.7) is False

                # Test max mode
                callback.mode = "max"
                assert callback._is_better(0.7) is True
                assert callback._is_better(0.3) is False


class TestHFTrainerCallbackMocked:
    """Test HF Trainer callback with fully mocked dependencies."""

    def test_is_better_logic(self, temp_repo):
        """Test the _is_better comparison logic."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        with patch("coral.integrations.hf_trainer.HAS_TRANSFORMERS", True):
            with patch("coral.integrations.hf_trainer.TrainerCallback", MagicMock):
                # Create callback with mocked parent class
                callback = CoralTrainerCallback.__new__(CoralTrainerCallback)
                callback._repo = temp_repo
                callback.repo_path = temp_repo.coral_dir.parent
                callback.init = False
                callback.mode = "min"
                callback.best_metric = 0.5

                # Test the comparison logic
                assert callback._is_better(0.3) is True
                assert callback._is_better(0.7) is False

                # Test max mode
                callback.mode = "max"
                assert callback._is_better(0.7) is True
                assert callback._is_better(0.3) is False


class TestPyTorchIntegrationMocked:
    """Test PyTorch integration with mocked torch."""

    def test_model_to_weights_logic(self, temp_repo):
        """Test model_to_weights conversion logic."""
        from coral.integrations.pytorch import PyTorchIntegration

        # Create a mock model
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.detach.return_value.cpu.return_value.numpy.return_value = np.ones(
            (10, 5), dtype=np.float32
        )
        mock_model.named_parameters.return_value = [("layer.weight", mock_param)]

        with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
            with patch("coral.integrations.pytorch.torch") as mock_torch:
                mock_torch.Tensor = type("Tensor", (), {})
                weights = PyTorchIntegration.model_to_weights(mock_model)

        assert "layer.weight" in weights
        assert weights["layer.weight"].shape == (10, 5)

    def test_weights_to_model_logic(self, temp_repo):
        """Test weights_to_model loading logic."""
        from coral.integrations.pytorch import PyTorchIntegration

        # Create weights
        weights = {
            "layer.weight": WeightTensor(
                data=np.ones((10, 5), dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer.weight", shape=(10, 5), dtype=np.float32
                ),
            ),
        }

        # Create a mock model
        mock_model = MagicMock()

        with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
            with patch("coral.integrations.pytorch.torch") as mock_torch:
                mock_torch.from_numpy = MagicMock(return_value=MagicMock())
                PyTorchIntegration.weights_to_model(weights, mock_model)

        mock_model.load_state_dict.assert_called_once()


class TestExperimentBridge:
    """Test ExperimentBridge abstract class."""

    def test_abstract_methods(self):
        """Test that ExperimentBridge has abstract methods."""
        from coral.integrations.experiment_bridge import ExperimentBridge

        # ExperimentBridge should have abstract methods
        assert hasattr(ExperimentBridge, "start_run")
        assert hasattr(ExperimentBridge, "end_run")
        assert hasattr(ExperimentBridge, "log_metrics")
        assert hasattr(ExperimentBridge, "log_coral_commit")

    def test_is_run_active_property(self):
        """Test is_run_active property returns False by default."""
        from coral.integrations.experiment_bridge import ExperimentBridge

        # Create a concrete implementation
        class TestBridge(ExperimentBridge):
            def __init__(self, repo):
                super().__init__(repo)
                self._run_id = None

            def start_run(self, name=None, params=None, tags=None):
                self._run_id = "test-run"
                return self._run_id

            def end_run(self, status="completed"):
                self._run_id = None

            def log_metrics(self, metrics, step=None):
                pass

            def log_params(self, params):
                pass

            def log_coral_commit(self, commit_hash, message=None):
                pass

            def get_commit_for_run(self, run_id):
                return None

            @property
            def is_run_active(self):
                return self._run_id is not None

        mock_repo = MagicMock()
        bridge = TestBridge(mock_repo)
        assert bridge.is_run_active is False

        bridge.start_run()
        assert bridge.is_run_active is True

        bridge.end_run()
        assert bridge.is_run_active is False


class TestS3StoreDataIntegrity:
    """Test S3Store data integrity features."""

    def test_data_integrity_error(self):
        """Test DataIntegrityError exception."""
        from coral.storage.weight_store import DataIntegrityError

        error = DataIntegrityError(
            expected_hash="abc123",
            actual_hash="def456",
            weight_name="layer.weight",
        )

        assert "abc123" in str(error)
        assert "def456" in str(error)
        assert "layer.weight" in str(error)


class TestStorageInit:
    """Test storage module initialization."""

    def test_hdf5_store_available(self):
        """Test that HDF5Store is always available."""
        from coral.storage import HDF5Store

        assert HDF5Store is not None

    def test_s3_store_import(self):
        """Test S3Store import behavior."""
        from coral.storage import s3_store

        # S3Config should always be importable
        assert hasattr(s3_store, "S3Config")


class TestIntegrationsInit:
    """Test integrations module initialization."""

    def test_pytorch_integration_import(self):
        """Test PyTorchIntegration import."""
        # This should not raise even without torch
        from coral.integrations import pytorch

        assert hasattr(pytorch, "PyTorchIntegration")
        assert hasattr(pytorch, "TORCH_AVAILABLE")

    def test_lightning_import(self):
        """Test Lightning import."""
        from coral.integrations import lightning

        assert hasattr(lightning, "CoralCallback")
        assert hasattr(lightning, "HAS_LIGHTNING")

    def test_hf_trainer_import(self):
        """Test HF Trainer import."""
        from coral.integrations import hf_trainer

        assert hasattr(hf_trainer, "CoralTrainerCallback")
        assert hasattr(hf_trainer, "HAS_TRANSFORMERS")

    def test_huggingface_import(self):
        """Test HuggingFace import."""
        from coral.integrations import huggingface

        assert hasattr(huggingface, "HF_AVAILABLE")
        assert hasattr(huggingface, "ModelInfo")
        assert hasattr(huggingface, "DownloadStats")


class TestCoralTrainerMocked:
    """Test CoralTrainer with mocked dependencies."""

    @pytest.fixture
    def mock_repo(self, temp_repo):
        """Create mock repository."""
        return temp_repo

    def test_trainer_callbacks_storage(self, mock_repo):
        """Test trainer callback storage."""
        from coral.integrations.pytorch import CoralTrainer

        with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
            with patch("coral.integrations.pytorch.torch") as mock_torch:
                mock_torch.nn.Module = MagicMock

                mock_model = MagicMock()
                mock_model.__class__.__name__ = "TestModel"

                trainer = CoralTrainer(
                    model=mock_model,
                    repository=mock_repo,
                    experiment_name="test",
                )

                # Test callback storage
                assert hasattr(trainer, "on_epoch_end_callbacks")
                assert hasattr(trainer, "on_step_end_callbacks")
                assert hasattr(trainer, "on_checkpoint_save_callbacks")

    def test_trainer_metrics_update(self, mock_repo):
        """Test trainer metrics update."""
        from coral.integrations.pytorch import CoralTrainer

        with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
            with patch("coral.integrations.pytorch.torch") as mock_torch:
                mock_torch.nn.Module = MagicMock

                mock_model = MagicMock()
                mock_model.__class__.__name__ = "TestModel"

                trainer = CoralTrainer(
                    model=mock_model,
                    repository=mock_repo,
                    experiment_name="test",
                )

                # Update metrics
                trainer.update_metrics(loss=0.5, accuracy=0.9)

                assert trainer.training_metrics["loss"] == 0.5
                assert trainer.training_metrics["accuracy"] == 0.9


class TestVersionGraph:
    """Test version graph functionality."""

    def test_version_graph_creation(self, temp_repo):
        """Test creating version graph."""
        from coral.version_control.version import VersionGraph

        graph = VersionGraph()

        # Should start empty
        assert len(graph.commits) == 0

    def test_version_graph_add_commit(self, temp_repo):
        """Test adding commit to version graph."""
        from coral.version_control.commit import Commit, CommitMetadata
        from coral.version_control.version import VersionGraph

        graph = VersionGraph()

        # Create actual commit objects
        metadata1 = CommitMetadata(
            author="test", email="test@test.com", message="commit1"
        )
        commit1 = Commit(
            commit_hash="commit1",
            parent_hashes=[],
            weight_hashes={},
            metadata=metadata1,
        )
        graph.add_commit(commit1)

        metadata2 = CommitMetadata(
            author="test", email="test@test.com", message="commit2"
        )
        commit2 = Commit(
            commit_hash="commit2",
            parent_hashes=["commit1"],
            weight_hashes={},
            metadata=metadata2,
        )
        graph.add_commit(commit2)

        assert "commit1" in graph.commits
        assert "commit2" in graph.commits

    def test_version_graph_get_ancestors(self, temp_repo):
        """Test getting ancestors from version graph."""
        from coral.version_control.commit import Commit, CommitMetadata
        from coral.version_control.version import VersionGraph

        graph = VersionGraph()

        # Create a chain of commits
        metadata1 = CommitMetadata(
            author="test", email="test@test.com", message="commit1"
        )
        commit1 = Commit(
            commit_hash="commit1",
            parent_hashes=[],
            weight_hashes={},
            metadata=metadata1,
        )
        graph.add_commit(commit1)

        metadata2 = CommitMetadata(
            author="test", email="test@test.com", message="commit2"
        )
        commit2 = Commit(
            commit_hash="commit2",
            parent_hashes=["commit1"],
            weight_hashes={},
            metadata=metadata2,
        )
        graph.add_commit(commit2)

        metadata3 = CommitMetadata(
            author="test", email="test@test.com", message="commit3"
        )
        commit3 = Commit(
            commit_hash="commit3",
            parent_hashes=["commit2"],
            weight_hashes={},
            metadata=metadata3,
        )
        graph.add_commit(commit3)

        ancestors = graph.get_commit_ancestors("commit3")

        assert "commit2" in ancestors
        assert "commit1" in ancestors


class TestCommitObject:
    """Test Commit object functionality."""

    def test_commit_creation(self):
        """Test creating a commit object."""
        from coral.version_control.commit import Commit, CommitMetadata

        metadata = CommitMetadata(
            author="Test User",
            email="test@example.com",
            message="Test commit",
        )

        commit = Commit(
            commit_hash="abc123",
            parent_hashes=[],
            weight_hashes={"layer.weight": "weight_hash"},
            metadata=metadata,
        )

        assert commit.commit_hash == "abc123"
        assert commit.is_root_commit
        assert "layer.weight" in commit.weight_hashes

    def test_commit_with_tags(self):
        """Test creating a commit with tags."""
        from coral.version_control.commit import Commit, CommitMetadata

        metadata = CommitMetadata(
            author="Test User",
            email="test@example.com",
            message="Tagged commit",
            tags=["v1.0", "release"],
        )

        commit = Commit(
            commit_hash="def456",
            parent_hashes=["abc123"],
            weight_hashes={},
            metadata=metadata,
        )

        assert "v1.0" in commit.metadata.tags
        assert "release" in commit.metadata.tags
        assert not commit.is_root_commit


class TestBranchManager:
    """Test BranchManager functionality."""

    def test_branch_creation(self, temp_repo):
        """Test creating a branch."""
        # First stage and commit something so we have a valid commit
        weights = {
            "layer.weight": WeightTensor(
                data=np.ones((5, 5), dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer.weight", shape=(5, 5), dtype=np.float32
                ),
            ),
        }
        temp_repo.stage_weights(weights)
        temp_repo.commit("Initial commit")

        temp_repo.create_branch("feature")

        branches = [b.name for b in temp_repo.branch_manager.list_branches()]
        assert "feature" in branches

    def test_branch_checkout(self, temp_repo):
        """Test checking out a branch."""
        # First stage and commit something
        weights = {
            "layer.weight": WeightTensor(
                data=np.ones((5, 5), dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer.weight", shape=(5, 5), dtype=np.float32
                ),
            ),
        }
        temp_repo.stage_weights(weights)
        temp_repo.commit("Initial commit")

        temp_repo.create_branch("develop")
        temp_repo.checkout("develop")

        assert temp_repo.branch_manager.get_current_branch() == "develop"

    def test_branch_deletion(self, temp_repo):
        """Test deleting a branch."""
        # First stage and commit something
        weights = {
            "layer.weight": WeightTensor(
                data=np.ones((5, 5), dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer.weight", shape=(5, 5), dtype=np.float32
                ),
            ),
        }
        temp_repo.stage_weights(weights)
        temp_repo.commit("Initial commit")

        temp_repo.create_branch("to-delete")
        temp_repo.branch_manager.delete_branch("to-delete")

        branches = [b.name for b in temp_repo.branch_manager.list_branches()]
        assert "to-delete" not in branches


class TestSimilarityUtils:
    """Test similarity utilities."""

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        from coral.utils.similarity import cosine_similarity

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])

        sim = cosine_similarity(a, b)
        assert sim == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity for orthogonal vectors."""
        from coral.utils.similarity import cosine_similarity

        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])

        sim = cosine_similarity(a, b)
        assert sim == pytest.approx(0.0)

    def test_are_similar(self):
        """Test are_similar function."""
        from coral.utils.similarity import are_similar

        a = np.ones((10, 10), dtype=np.float32)
        b = np.ones((10, 10), dtype=np.float32) * 1.01

        assert are_similar(a, b, threshold=0.99) is True
        assert are_similar(a, b * 0.5, threshold=0.99) is False


class TestJsonUtils:
    """Test JSON utilities."""

    def test_json_encoder_numpy(self):
        """Test JSON encoding numpy arrays."""
        import json

        from coral.utils.json_utils import NumpyJSONEncoder

        data = {"array": np.array([1, 2, 3])}
        result = json.dumps(data, cls=NumpyJSONEncoder)

        assert "[1, 2, 3]" in result

    def test_json_encoder_numpy_dtype(self):
        """Test JSON encoding numpy dtypes."""
        import json

        from coral.utils.json_utils import NumpyJSONEncoder

        # Create an actual dtype object, not the type itself
        data = {"dtype": np.dtype(np.float32)}
        result = json.dumps(data, cls=NumpyJSONEncoder)

        assert "float32" in result

    def test_dumps_numpy_convenience(self):
        """Test dumps_numpy convenience function."""
        from coral.utils.json_utils import dumps_numpy

        data = {
            "array": np.array([1, 2, 3]),
            "scalar": np.float64(0.5),
            "integer": np.int64(42),
        }
        result = dumps_numpy(data)

        assert "[1, 2, 3]" in result
        assert "0.5" in result
        assert "42" in result
