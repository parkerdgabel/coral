"""Tests for HuggingFace Trainer integration.

This module tests the CoralTrainerCallback for HuggingFace Transformers.
Tests are skipped if HuggingFace Transformers is not installed.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.version_control.repository import Repository

# Check if Transformers is available
try:
    from transformers import TrainerCallback

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing."""
    temp_dir = tempfile.mkdtemp()
    repo_path = Path(temp_dir)
    repo = Repository(repo_path, init=True)
    yield repo
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_training_args():
    """Create mock training arguments."""
    args = MagicMock()
    args.learning_rate = 5e-5
    args.num_train_epochs = 3
    args.per_device_train_batch_size = 16
    args.weight_decay = 0.01
    args.warmup_steps = 100
    return args


@pytest.fixture
def mock_trainer_state():
    """Create mock trainer state."""
    state = MagicMock()
    state.global_step = 500
    state.epoch = 1.5
    return state


@pytest.fixture
def mock_trainer_control():
    """Create mock trainer control."""
    return MagicMock()


@pytest.fixture
def mock_hf_model():
    """Create a mock HuggingFace model."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.model_type = "bert"

    # Create mock state dict with tensor-like objects
    mock_weight = MagicMock()
    mock_weight.cpu.return_value.numpy.return_value = np.random.randn(768, 768).astype(
        np.float32
    )
    mock_bias = MagicMock()
    mock_bias.cpu.return_value.numpy.return_value = np.random.randn(768).astype(
        np.float32
    )

    model.state_dict.return_value = {
        "encoder.layer.0.attention.self.query.weight": mock_weight,
        "encoder.layer.0.attention.self.query.bias": mock_bias,
    }

    return model


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="HuggingFace Transformers not installed")
class TestCoralTrainerCallbackInitialization:
    """Test CoralTrainerCallback initialization."""

    def test_init_with_repo_path(self, temp_repo):
        """Test initialization with repo path."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        callback = CoralTrainerCallback(repo=temp_repo.coral_dir.parent)

        assert callback.repo_path is not None
        assert callback.save_every_n_epochs == 0
        assert callback.save_every_n_steps == 0

    def test_init_with_repository_object(self, temp_repo):
        """Test initialization with Repository object."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        callback = CoralTrainerCallback(repo=temp_repo)

        assert callback._repo is temp_repo

    def test_init_with_save_options(self, temp_repo):
        """Test initialization with save options."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        callback = CoralTrainerCallback(
            repo=temp_repo,
            save_every_n_epochs=1,
            save_every_n_steps=500,
            save_on_best="eval_loss",
            mode="min",
        )

        assert callback.save_every_n_epochs == 1
        assert callback.save_every_n_steps == 500
        assert callback.save_on_best == "eval_loss"
        assert callback.mode == "min"

    def test_init_with_max_mode(self, temp_repo):
        """Test initialization with max mode."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        callback = CoralTrainerCallback(
            repo=temp_repo,
            save_on_best="eval_accuracy",
            mode="max",
        )

        assert callback.mode == "max"
        assert callback.best_metric == float("-inf")

    def test_init_with_min_mode(self, temp_repo):
        """Test initialization with min mode."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        callback = CoralTrainerCallback(
            repo=temp_repo,
            save_on_best="eval_loss",
            mode="min",
        )

        assert callback.mode == "min"
        assert callback.best_metric == float("inf")

    def test_init_with_experiment_bridge(self, temp_repo):
        """Test initialization with experiment bridge."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        mock_bridge = MagicMock()
        callback = CoralTrainerCallback(
            repo=temp_repo,
            experiment_bridge=mock_bridge,
        )

        assert callback.experiment_bridge is mock_bridge

    def test_init_with_save_on_train_end(self, temp_repo):
        """Test initialization with save_on_train_end option."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        callback = CoralTrainerCallback(
            repo=temp_repo,
            save_on_train_end=False,
        )

        assert callback.save_on_train_end is False

    def test_init_requires_repo(self):
        """Test that init requires repo or repo_path."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        with pytest.raises(ValueError, match="Either repo or repo_path"):
            CoralTrainerCallback()

    def test_deprecated_repo_path(self, temp_repo):
        """Test that repo_path is deprecated."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        with pytest.warns(DeprecationWarning, match="repo_path is deprecated"):
            callback = CoralTrainerCallback(repo_path=temp_repo.coral_dir.parent)

        assert callback.repo_path is not None


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="HuggingFace Transformers not installed")
class TestCoralTrainerCallbackMethods:
    """Test CoralTrainerCallback methods."""

    def test_is_better_min_mode(self, temp_repo):
        """Test _is_better in min mode."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        callback = CoralTrainerCallback(
            repo=temp_repo, save_on_best="eval_loss", mode="min"
        )
        callback.best_metric = 0.5

        assert callback._is_better(0.4) is True
        assert callback._is_better(0.6) is False
        assert callback._is_better(0.5) is False

    def test_is_better_max_mode(self, temp_repo):
        """Test _is_better in max mode."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        callback = CoralTrainerCallback(
            repo=temp_repo, save_on_best="eval_accuracy", mode="max"
        )
        callback.best_metric = 0.8

        assert callback._is_better(0.9) is True
        assert callback._is_better(0.7) is False
        assert callback._is_better(0.8) is False


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="HuggingFace Transformers not installed")
class TestCoralTrainerCallbackLifecycle:
    """Test CoralTrainerCallback lifecycle methods."""

    def test_on_train_begin(
        self,
        temp_repo,
        mock_training_args,
        mock_trainer_state,
        mock_trainer_control,
        mock_hf_model,
    ):
        """Test on_train_begin callback."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        mock_bridge = MagicMock()
        mock_bridge.is_run_active = False

        callback = CoralTrainerCallback(
            repo=temp_repo,
            experiment_bridge=mock_bridge,
        )

        result = callback.on_train_begin(
            mock_training_args,
            mock_trainer_state,
            mock_trainer_control,
            model=mock_hf_model,
        )

        mock_bridge.start_run.assert_called_once()
        assert result is mock_trainer_control

    def test_on_train_begin_without_bridge(
        self,
        temp_repo,
        mock_training_args,
        mock_trainer_state,
        mock_trainer_control,
        mock_hf_model,
    ):
        """Test on_train_begin without experiment bridge."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        callback = CoralTrainerCallback(repo=temp_repo)

        result = callback.on_train_begin(
            mock_training_args,
            mock_trainer_state,
            mock_trainer_control,
            model=mock_hf_model,
        )

        assert result is mock_trainer_control

    def test_on_step_end_saves_at_interval(
        self,
        temp_repo,
        mock_training_args,
        mock_trainer_state,
        mock_trainer_control,
        mock_hf_model,
    ):
        """Test on_step_end saves at step interval."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        callback = CoralTrainerCallback(
            repo=temp_repo,
            save_every_n_steps=500,
        )

        mock_trainer_state.global_step = 500
        result = callback.on_step_end(
            mock_training_args,
            mock_trainer_state,
            mock_trainer_control,
            model=mock_hf_model,
        )

        commits = callback.repo.log(max_commits=1)
        assert len(commits) == 1
        assert "Step 500" in commits[0].metadata.message
        assert result is mock_trainer_control

    def test_on_step_end_no_duplicate_saves(
        self,
        temp_repo,
        mock_training_args,
        mock_trainer_state,
        mock_trainer_control,
        mock_hf_model,
    ):
        """Test on_step_end doesn't save duplicates."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        callback = CoralTrainerCallback(
            repo=temp_repo,
            save_every_n_steps=500,
        )

        mock_trainer_state.global_step = 500

        # First save
        callback.on_step_end(
            mock_training_args,
            mock_trainer_state,
            mock_trainer_control,
            model=mock_hf_model,
        )

        # Second call at same step should not create new commit
        callback.on_step_end(
            mock_training_args,
            mock_trainer_state,
            mock_trainer_control,
            model=mock_hf_model,
        )

        commits = callback.repo.log(max_commits=10)
        step_500_commits = [c for c in commits if "Step 500" in c.metadata.message]
        assert len(step_500_commits) == 1

    def test_on_epoch_end_saves_at_interval(
        self,
        temp_repo,
        mock_training_args,
        mock_trainer_state,
        mock_trainer_control,
        mock_hf_model,
    ):
        """Test on_epoch_end saves at epoch interval."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        callback = CoralTrainerCallback(
            repo=temp_repo,
            save_every_n_epochs=1,
        )

        mock_trainer_state.epoch = 1.0
        result = callback.on_epoch_end(
            mock_training_args,
            mock_trainer_state,
            mock_trainer_control,
            model=mock_hf_model,
        )

        commits = callback.repo.log(max_commits=1)
        assert len(commits) == 1
        assert "Epoch 1" in commits[0].metadata.message
        assert result is mock_trainer_control

    def test_on_log_logs_to_bridge(
        self,
        temp_repo,
        mock_training_args,
        mock_trainer_state,
        mock_trainer_control,
    ):
        """Test on_log logs metrics to experiment bridge."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        mock_bridge = MagicMock()
        mock_bridge.is_run_active = True

        callback = CoralTrainerCallback(
            repo=temp_repo,
            experiment_bridge=mock_bridge,
        )

        logs = {"loss": 0.5, "learning_rate": 5e-5}
        result = callback.on_log(
            mock_training_args,
            mock_trainer_state,
            mock_trainer_control,
            logs=logs,
        )

        mock_bridge.log_metrics.assert_called_once()
        assert result is mock_trainer_control

    def test_on_evaluate_saves_on_best(
        self,
        temp_repo,
        mock_training_args,
        mock_trainer_state,
        mock_trainer_control,
        mock_hf_model,
    ):
        """Test on_evaluate saves on best metric."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        callback = CoralTrainerCallback(
            repo=temp_repo,
            save_on_best="eval_loss",
            mode="min",
        )

        mock_trainer_state.global_step = 1000
        metrics = {"eval_loss": 0.3}

        result = callback.on_evaluate(
            mock_training_args,
            mock_trainer_state,
            mock_trainer_control,
            model=mock_hf_model,
            metrics=metrics,
        )

        assert callback.best_metric == 0.3
        assert callback.best_step == 1000

        commits = callback.repo.log(max_commits=1)
        assert len(commits) == 1
        assert "Best eval_loss" in commits[0].metadata.message
        assert result is mock_trainer_control

    def test_on_train_end_saves_final(
        self,
        temp_repo,
        mock_training_args,
        mock_trainer_state,
        mock_trainer_control,
        mock_hf_model,
    ):
        """Test on_train_end saves final checkpoint."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        callback = CoralTrainerCallback(
            repo=temp_repo,
            save_on_train_end=True,
        )

        mock_trainer_state.global_step = 1500
        mock_trainer_state.epoch = 3.0

        result = callback.on_train_end(
            mock_training_args,
            mock_trainer_state,
            mock_trainer_control,
            model=mock_hf_model,
        )

        commits = callback.repo.log(max_commits=1)
        assert len(commits) == 1
        assert "Training complete" in commits[0].metadata.message
        assert result is mock_trainer_control

    def test_on_train_end_no_save_when_disabled(
        self,
        temp_repo,
        mock_training_args,
        mock_trainer_state,
        mock_trainer_control,
        mock_hf_model,
    ):
        """Test on_train_end doesn't save when disabled."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        callback = CoralTrainerCallback(
            repo=temp_repo,
            save_on_train_end=False,
        )

        result = callback.on_train_end(
            mock_training_args,
            mock_trainer_state,
            mock_trainer_control,
            model=mock_hf_model,
        )

        commits = callback.repo.log(max_commits=1)
        assert len(commits) == 0
        assert result is mock_trainer_control

    def test_on_train_end_ends_experiment_run(
        self,
        temp_repo,
        mock_training_args,
        mock_trainer_state,
        mock_trainer_control,
        mock_hf_model,
    ):
        """Test on_train_end ends experiment run."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        mock_bridge = MagicMock()
        mock_bridge.is_run_active = True

        callback = CoralTrainerCallback(
            repo=temp_repo,
            experiment_bridge=mock_bridge,
        )

        callback.on_train_end(
            mock_training_args,
            mock_trainer_state,
            mock_trainer_control,
            model=mock_hf_model,
        )

        mock_bridge.end_run.assert_called_once_with("completed")


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="HuggingFace Transformers not installed")
class TestCoralTrainerCallbackSaveCheckpoint:
    """Test _save_checkpoint method."""

    def test_save_checkpoint_creates_commit(self, temp_repo, mock_hf_model):
        """Test that _save_checkpoint creates a commit."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        callback = CoralTrainerCallback(repo=temp_repo)

        callback._save_checkpoint(mock_hf_model, message="Test commit")

        commits = callback.repo.log(max_commits=1)
        assert len(commits) == 1
        assert commits[0].metadata.message == "Test commit"

    def test_save_checkpoint_with_metrics(self, temp_repo, mock_hf_model):
        """Test _save_checkpoint with metrics."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        callback = CoralTrainerCallback(repo=temp_repo, save_on_best="eval_loss")

        callback._save_checkpoint(
            mock_hf_model,
            message="Best model",
            metrics={"eval_loss": 0.25},
            step=1000,
        )

        commits = callback.repo.log(max_commits=1)
        assert len(commits) == 1
        assert "best_eval_loss" in commits[0].metadata.tags
        assert "step_1000" in commits[0].metadata.tags

    def test_save_checkpoint_with_epoch_tag(self, temp_repo, mock_hf_model):
        """Test _save_checkpoint with epoch tag."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        callback = CoralTrainerCallback(repo=temp_repo)

        callback._save_checkpoint(
            mock_hf_model,
            message="Epoch checkpoint",
            epoch=5,
        )

        commits = callback.repo.log(max_commits=1)
        assert len(commits) == 1
        assert "epoch_5" in commits[0].metadata.tags

    def test_save_checkpoint_with_experiment_bridge(self, temp_repo, mock_hf_model):
        """Test _save_checkpoint with experiment bridge."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        mock_bridge = MagicMock()
        mock_bridge.is_run_active = True

        callback = CoralTrainerCallback(
            repo=temp_repo,
            experiment_bridge=mock_bridge,
        )

        callback._save_checkpoint(
            mock_hf_model,
            message="Test commit",
            metrics={"loss": 0.5},
            step=100,
        )

        mock_bridge.log_coral_commit.assert_called_once()
        mock_bridge.log_metrics.assert_called_once_with({"loss": 0.5}, step=100)


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="HuggingFace Transformers not installed")
class TestCoralTrainerCallbackLoadFromCoral:
    """Test load_from_coral method."""

    def test_load_from_coral(self, temp_repo):
        """Test loading weights from Coral."""
        from coral.integrations.hf_trainer import CoralTrainerCallback

        # First, save some weights
        weights = {
            "encoder.layer.0.weight": WeightTensor(
                data=np.random.randn(768, 768).astype(np.float32),
                metadata=WeightMetadata(
                    name="encoder.layer.0.weight",
                    shape=(768, 768),
                    dtype=np.float32,
                ),
            ),
        }
        temp_repo.stage_weights(weights)
        temp_repo.commit("Initial weights")

        callback = CoralTrainerCallback(repo=temp_repo)

        # Create mock model with load_state_dict
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.missing_keys = []
        mock_result.unexpected_keys = []
        mock_model.load_state_dict.return_value = mock_result

        with patch("coral.integrations.hf_trainer.torch") as mock_torch:
            mock_torch.from_numpy = MagicMock(return_value=MagicMock())
            result = callback.load_from_coral(mock_model, strict=False)

        assert "loaded_weights" in result
        assert result["missing_keys"] == []
        assert result["unexpected_keys"] == []


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="HuggingFace Transformers not installed")
class TestCoralHFCallbackAlias:
    """Test that CoralHFCallback is an alias."""

    def test_alias_exists(self):
        """Test that CoralHFCallback is an alias for CoralTrainerCallback."""
        from coral.integrations.hf_trainer import CoralHFCallback, CoralTrainerCallback

        assert CoralHFCallback is CoralTrainerCallback


class TestCoralTrainerCallbackWithoutTransformers:
    """Tests that work without Transformers installed."""

    def test_import_error_without_transformers(self, temp_repo):
        """Test that CoralTrainerCallback raises ImportError without Transformers."""
        # Only run this test if Transformers is NOT installed
        if HAS_TRANSFORMERS:
            pytest.skip("HuggingFace Transformers is installed")

        with patch("coral.integrations.hf_trainer.HAS_TRANSFORMERS", False):
            from coral.integrations.hf_trainer import CoralTrainerCallback

            with pytest.raises(ImportError, match="HuggingFace Transformers is required"):
                CoralTrainerCallback(repo=temp_repo)
