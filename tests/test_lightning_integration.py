"""Tests for PyTorch Lightning integration.

This module tests the CoralCallback for PyTorch Lightning.
Tests are skipped if PyTorch Lightning is not installed.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from coral.version_control.repository import Repository

# Check if PyTorch Lightning is available
try:
    import pytorch_lightning as pl

    HAS_LIGHTNING = True
except ImportError:
    try:
        import lightning.pytorch as pl

        HAS_LIGHTNING = True
    except ImportError:
        HAS_LIGHTNING = False
        pl = None


@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing."""
    temp_dir = tempfile.mkdtemp()
    repo_path = Path(temp_dir)
    repo = Repository(repo_path, init=True)
    yield repo
    shutil.rmtree(temp_dir)


@pytest.mark.skipif(not HAS_LIGHTNING, reason="PyTorch Lightning not installed")
class TestCoralCallbackInitialization:
    """Test CoralCallback initialization."""

    def test_init_with_repo_path(self, temp_repo):
        """Test initialization with repo path."""
        from coral.integrations.lightning import CoralCallback

        callback = CoralCallback(repo=temp_repo.coral_dir.parent)

        assert callback.repo_path is not None
        assert callback.save_every_n_epochs == 0
        assert callback.save_every_n_steps == 0

    def test_init_with_repository_object(self, temp_repo):
        """Test initialization with Repository object."""
        from coral.integrations.lightning import CoralCallback

        callback = CoralCallback(repo=temp_repo)

        assert callback._repo is temp_repo

    def test_init_with_save_options(self, temp_repo):
        """Test initialization with save options."""
        from coral.integrations.lightning import CoralCallback

        callback = CoralCallback(
            repo=temp_repo,
            save_every_n_epochs=5,
            save_every_n_steps=100,
            save_on_best="val_loss",
            mode="min",
        )

        assert callback.save_every_n_epochs == 5
        assert callback.save_every_n_steps == 100
        assert callback.save_on_best == "val_loss"
        assert callback.mode == "min"

    def test_init_with_max_mode(self, temp_repo):
        """Test initialization with max mode."""
        from coral.integrations.lightning import CoralCallback

        callback = CoralCallback(
            repo=temp_repo,
            save_on_best="accuracy",
            mode="max",
        )

        assert callback.mode == "max"
        assert callback.best_metric == float("-inf")

    def test_init_with_min_mode(self, temp_repo):
        """Test initialization with min mode."""
        from coral.integrations.lightning import CoralCallback

        callback = CoralCallback(
            repo=temp_repo,
            save_on_best="loss",
            mode="min",
        )

        assert callback.mode == "min"
        assert callback.best_metric == float("inf")

    def test_init_with_experiment_bridge(self, temp_repo):
        """Test initialization with experiment bridge."""
        from coral.integrations.lightning import CoralCallback

        mock_bridge = MagicMock()
        callback = CoralCallback(
            repo=temp_repo,
            experiment_bridge=mock_bridge,
        )

        assert callback.experiment_bridge is mock_bridge

    def test_init_requires_repo(self):
        """Test that init requires repo or repo_path."""
        from coral.integrations.lightning import CoralCallback

        with pytest.raises(ValueError, match="Either repo or repo_path"):
            CoralCallback()

    def test_deprecated_repo_path(self, temp_repo):
        """Test that repo_path is deprecated."""
        from coral.integrations.lightning import CoralCallback

        with pytest.warns(DeprecationWarning, match="repo_path is deprecated"):
            callback = CoralCallback(repo_path=temp_repo.coral_dir.parent)

        assert callback.repo_path is not None


@pytest.mark.skipif(not HAS_LIGHTNING, reason="PyTorch Lightning not installed")
class TestCoralCallbackMethods:
    """Test CoralCallback methods."""

    def test_is_better_min_mode(self, temp_repo):
        """Test _is_better in min mode."""
        from coral.integrations.lightning import CoralCallback

        callback = CoralCallback(repo=temp_repo, save_on_best="loss", mode="min")
        callback.best_metric = 0.5

        assert callback._is_better(0.4) is True
        assert callback._is_better(0.6) is False
        assert callback._is_better(0.5) is False

    def test_is_better_max_mode(self, temp_repo):
        """Test _is_better in max mode."""
        from coral.integrations.lightning import CoralCallback

        callback = CoralCallback(repo=temp_repo, save_on_best="accuracy", mode="max")
        callback.best_metric = 0.8

        assert callback._is_better(0.9) is True
        assert callback._is_better(0.7) is False
        assert callback._is_better(0.8) is False

    def test_extract_metadata(self, temp_repo):
        """Test _extract_metadata method."""
        from coral.integrations.lightning import CoralCallback

        callback = CoralCallback(
            repo=temp_repo,
            metadata_keys=["current_epoch", "global_step"],
        )

        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 5
        mock_trainer.global_step = 1000

        metadata = callback._extract_metadata(mock_trainer)

        assert metadata["current_epoch"] == 5
        assert metadata["global_step"] == 1000


@pytest.mark.skipif(not HAS_LIGHTNING, reason="PyTorch Lightning not installed")
class TestCoralLightningCallbackAlias:
    """Test that CoralLightningCallback is an alias."""

    def test_alias_exists(self):
        """Test that CoralLightningCallback is an alias for CoralCallback."""
        from coral.integrations.lightning import CoralCallback, CoralLightningCallback

        assert CoralLightningCallback is CoralCallback


class TestCoralCallbackWithoutLightning:
    """Tests that work without PyTorch Lightning installed."""

    def test_import_error_without_lightning(self, temp_repo):
        """Test that CoralCallback raises ImportError without Lightning."""
        # Only run this test if Lightning is NOT installed
        if HAS_LIGHTNING:
            pytest.skip("PyTorch Lightning is installed")

        with patch("coral.integrations.lightning.HAS_LIGHTNING", False):
            # Need to reload the module to get the updated check
            from coral.integrations.lightning import CoralCallback

            with pytest.raises(ImportError, match="PyTorch Lightning is required"):
                CoralCallback(repo=temp_repo)


@pytest.mark.skipif(not HAS_LIGHTNING, reason="PyTorch Lightning not installed")
class TestCoralCallbackLifecycle:
    """Test CoralCallback lifecycle methods."""

    def test_on_train_start_with_bridge(self, temp_repo):
        """Test on_train_start with experiment bridge."""
        from coral.integrations.lightning import CoralCallback

        mock_bridge = MagicMock()
        mock_bridge.is_run_active = False

        callback = CoralCallback(
            repo=temp_repo,
            experiment_bridge=mock_bridge,
        )

        mock_trainer = MagicMock()
        mock_module = MagicMock()
        mock_module.hparams = {"lr": 0.001}
        mock_module.__class__.__name__ = "TestModel"

        callback.on_train_start(mock_trainer, mock_module)

        mock_bridge.start_run.assert_called_once()

    def test_on_train_start_bridge_already_active(self, temp_repo):
        """Test on_train_start doesn't start new run if already active."""
        from coral.integrations.lightning import CoralCallback

        mock_bridge = MagicMock()
        mock_bridge.is_run_active = True

        callback = CoralCallback(
            repo=temp_repo,
            experiment_bridge=mock_bridge,
        )

        mock_trainer = MagicMock()
        mock_module = MagicMock()

        callback.on_train_start(mock_trainer, mock_module)

        mock_bridge.start_run.assert_not_called()

    def test_on_train_epoch_end_logs_to_bridge(self, temp_repo):
        """Test on_train_epoch_end logs metrics to experiment bridge."""
        from coral.integrations.lightning import CoralCallback

        mock_bridge = MagicMock()
        mock_bridge.is_run_active = True

        callback = CoralCallback(
            repo=temp_repo,
            experiment_bridge=mock_bridge,
        )

        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 5
        mock_trainer.global_step = 1000
        mock_trainer.callback_metrics = {"val_loss": 0.3, "train_loss": 0.2}

        mock_module = MagicMock()
        mock_module.state_dict.return_value = {}

        callback.on_train_epoch_end(mock_trainer, mock_module)

        mock_bridge.log_metrics.assert_called()

    def test_on_train_batch_end_saves_at_interval(self, temp_repo):
        """Test on_train_batch_end saves at step interval."""
        from coral.integrations.lightning import CoralCallback

        callback = CoralCallback(
            repo=temp_repo,
            save_every_n_steps=500,
        )

        mock_trainer = MagicMock()
        mock_trainer.global_step = 500
        mock_trainer.current_epoch = 1

        mock_module = MagicMock()
        mock_module.state_dict.return_value = {}

        callback.on_train_batch_end(
            mock_trainer, mock_module, outputs={}, batch={}, batch_idx=499
        )

        # Should have committed
        commits = callback.repo.log(max_commits=10)
        assert any("Step 500" in c.metadata.message for c in commits)

    def test_on_train_end_saves_final_and_logs(self, temp_repo):
        """Test on_train_end saves final checkpoint and logs best metric."""
        from coral.integrations.lightning import CoralCallback

        callback = CoralCallback(
            repo=temp_repo,
            save_on_best="val_loss",
        )
        callback.best_metric = 0.1
        callback.best_epoch = 5

        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 10
        mock_trainer.global_step = 5000

        mock_module = MagicMock()
        mock_module.state_dict.return_value = {}

        callback.on_train_end(mock_trainer, mock_module)

        commits = callback.repo.log(max_commits=10)
        assert any("Training complete" in c.metadata.message for c in commits)

    def test_on_train_end_ends_bridge_run(self, temp_repo):
        """Test on_train_end ends experiment bridge run."""
        from coral.integrations.lightning import CoralCallback

        mock_bridge = MagicMock()
        mock_bridge.is_run_active = True

        callback = CoralCallback(
            repo=temp_repo,
            experiment_bridge=mock_bridge,
        )

        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 10

        mock_module = MagicMock()
        mock_module.state_dict.return_value = {}

        callback.on_train_end(mock_trainer, mock_module)

        mock_bridge.end_run.assert_called_once_with("completed")


@pytest.mark.skipif(not HAS_LIGHTNING, reason="PyTorch Lightning not installed")
class TestCoralCallbackSaveCheckpoint:
    """Test _save_checkpoint method."""

    def test_save_checkpoint_with_optimizer(self, temp_repo):
        """Test saving checkpoint with optimizer state."""
        from coral.integrations.lightning import CoralCallback

        callback = CoralCallback(
            repo=temp_repo,
            include_optimizer=True,
        )

        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 5
        mock_trainer.global_step = 1000
        mock_trainer.optimizers = [MagicMock()]
        mock_trainer.optimizers[0].state_dict.return_value = {"lr": 0.001}

        mock_module = MagicMock()
        import numpy as np

        mock_weight = MagicMock()
        mock_weight.cpu.return_value.numpy.return_value = np.ones((10, 10))
        mock_module.state_dict.return_value = {"weight": mock_weight}

        callback._save_checkpoint(mock_trainer, mock_module, "Test checkpoint")

        commits = callback.repo.log(max_commits=1)
        assert len(commits) == 1
        assert commits[0].metadata.message == "Test checkpoint"

    def test_save_checkpoint_with_metrics_and_tags(self, temp_repo):
        """Test saving checkpoint with metrics adds tags."""
        from coral.integrations.lightning import CoralCallback

        callback = CoralCallback(
            repo=temp_repo,
            save_on_best="val_loss",
        )

        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 5
        mock_trainer.global_step = 1000

        mock_module = MagicMock()
        mock_module.state_dict.return_value = {}

        callback._save_checkpoint(
            mock_trainer,
            mock_module,
            "Best model",
            metrics={"val_loss": 0.1},
        )

        commits = callback.repo.log(max_commits=1)
        assert len(commits) == 1
        assert "best_val_loss" in commits[0].metadata.tags

    def test_save_checkpoint_logs_to_bridge(self, temp_repo):
        """Test saving checkpoint logs to experiment bridge."""
        from coral.integrations.lightning import CoralCallback

        mock_bridge = MagicMock()
        mock_bridge.is_run_active = True

        callback = CoralCallback(
            repo=temp_repo,
            experiment_bridge=mock_bridge,
        )

        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 5
        mock_trainer.global_step = 1000

        mock_module = MagicMock()
        mock_module.state_dict.return_value = {}

        callback._save_checkpoint(
            mock_trainer,
            mock_module,
            "Test checkpoint",
            metrics={"loss": 0.5},
        )

        mock_bridge.log_coral_commit.assert_called_once()
        mock_bridge.log_metrics.assert_called_once()


@pytest.mark.skipif(not HAS_LIGHTNING, reason="PyTorch Lightning not installed")
class TestCoralCallbackLoadFromCoral:
    """Test load_from_coral method."""

    def test_load_from_coral_skips_optimizer(self, temp_repo):
        """Test load_from_coral skips optimizer state."""
        from coral.integrations.lightning import CoralCallback
        import numpy as np
        from coral.core.weight_tensor import WeightTensor, WeightMetadata

        # Save weights including optimizer state
        weights = {
            "encoder.weight": WeightTensor(
                data=np.random.randn(10, 10).astype(np.float32),
                metadata=WeightMetadata(
                    name="encoder.weight", shape=(10, 10), dtype=np.float32
                ),
            ),
            "optimizer_state": WeightTensor(
                data=np.array([1, 2, 3]).astype(np.float32),
                metadata=WeightMetadata(
                    name="optimizer_state", shape=(3,), dtype=np.float32
                ),
            ),
        }
        temp_repo.stage_weights(weights)
        temp_repo.commit("Initial weights")

        callback = CoralCallback(repo=temp_repo)

        mock_module = MagicMock()
        mock_result = MagicMock()
        mock_result.missing_keys = []
        mock_result.unexpected_keys = []
        mock_module.load_state_dict.return_value = mock_result

        with patch("coral.integrations.lightning.torch") as mock_torch:
            mock_torch.from_numpy = MagicMock(return_value=MagicMock())
            result = callback.load_from_coral(mock_module, strict=False)

        # optimizer_state should be skipped, so only 1 weight loaded
        assert result["loaded_weights"] == 1


@pytest.mark.skipif(not HAS_LIGHTNING, reason="PyTorch Lightning not installed")
class TestCoralCallbackBranchHandling:
    """Test branch handling in CoralCallback."""

    def test_repo_property_switches_branch(self, temp_repo):
        """Test that repo property switches to specified branch."""
        from coral.integrations.lightning import CoralCallback

        # Create callback with branch specified
        callback = CoralCallback(
            repo=temp_repo,
            branch="experiment",
        )

        # Access repo property to trigger lazy init
        _ = callback.repo

        # Check that branch was created and checked out
        branches = [b.name for b in temp_repo.branch_manager.list_branches()]
        assert "experiment" in branches


@pytest.mark.skipif(not HAS_LIGHTNING, reason="PyTorch Lightning not installed")
class TestCoralCallbackEpochSaving:
    """Test epoch-based saving."""

    def test_on_train_epoch_end_saves_at_interval(self, temp_repo):
        """Test saving at epoch intervals."""
        from coral.integrations.lightning import CoralCallback

        callback = CoralCallback(
            repo=temp_repo,
            save_every_n_epochs=2,
        )

        mock_trainer = MagicMock()
        mock_trainer.global_step = 100
        mock_trainer.callback_metrics = {}

        mock_module = MagicMock()
        mock_module.state_dict.return_value = {}

        # Epoch 0 - no save (0+1) % 2 == 1
        mock_trainer.current_epoch = 0
        callback.on_train_epoch_end(mock_trainer, mock_module)

        # Epoch 1 - save (1+1) % 2 == 0
        mock_trainer.current_epoch = 1
        callback.on_train_epoch_end(mock_trainer, mock_module)

        commits = callback.repo.log(max_commits=10)
        epoch_commits = [c for c in commits if "Epoch 2" in c.metadata.message]
        assert len(epoch_commits) == 1

    def test_on_train_epoch_end_saves_on_best(self, temp_repo):
        """Test saving on best metric improvement."""
        from coral.integrations.lightning import CoralCallback

        callback = CoralCallback(
            repo=temp_repo,
            save_on_best="val_loss",
            mode="min",
        )

        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 1
        mock_trainer.global_step = 100
        mock_trainer.callback_metrics = {"val_loss": 0.5}

        mock_module = MagicMock()
        mock_module.state_dict.return_value = {}

        # First epoch
        callback.on_train_epoch_end(mock_trainer, mock_module)

        # Check best metric was updated
        assert callback.best_metric == 0.5
        assert callback.best_epoch == 1

        # Second epoch with improvement
        mock_trainer.current_epoch = 2
        mock_trainer.callback_metrics = {"val_loss": 0.3}
        callback.on_train_epoch_end(mock_trainer, mock_module)

        assert callback.best_metric == 0.3
        assert callback.best_epoch == 2
