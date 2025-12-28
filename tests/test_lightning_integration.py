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
