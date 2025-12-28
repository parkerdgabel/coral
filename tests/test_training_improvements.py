"""Tests for training integration improvements.

Tests the new Checkpointer class, early stopping, checkpoint diff,
and unified load/save API.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.training.checkpoint_manager import CheckpointConfig, CheckpointManager
from coral.training.training_state import TrainingState
from coral.version_control.repository import Repository


@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing."""
    temp_dir = tempfile.mkdtemp()
    repo_path = Path(temp_dir)
    repo = Repository(repo_path, init=True)
    yield repo
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_weights():
    """Create sample weights for testing."""
    return {
        "layer1.weight": WeightTensor(
            data=np.random.randn(10, 5).astype(np.float32),
            metadata=WeightMetadata(
                name="layer1.weight",
                shape=(10, 5),
                dtype=np.float32,
                layer_type="Linear",
            ),
        ),
        "layer1.bias": WeightTensor(
            data=np.random.randn(10).astype(np.float32),
            metadata=WeightMetadata(
                name="layer1.bias", shape=(10,), dtype=np.float32, layer_type="Linear"
            ),
        ),
    }


class TestEarlyStopping:
    """Test early stopping functionality."""

    def test_early_stopping_config(self):
        """Test early stopping configuration options."""
        config = CheckpointConfig(
            save_on_best_metric="val_loss",
            minimize_metric=True,
            early_stopping_patience=5,
            early_stopping_threshold=0.001,
        )

        assert config.early_stopping_patience == 5
        assert config.early_stopping_threshold == 0.001

    def test_early_stopping_triggers_after_patience(self, temp_repo, sample_weights):
        """Test that early stopping triggers after patience is exhausted."""
        config = CheckpointConfig(
            save_on_best_metric="val_loss",
            minimize_metric=True,
            early_stopping_patience=3,
            early_stopping_threshold=0.0,
            auto_commit=True,
        )

        manager = CheckpointManager(
            repository=temp_repo, config=config, model_name="TestModel"
        )

        # First checkpoint - establishes baseline
        state1 = TrainingState(
            epoch=1,
            global_step=100,
            learning_rate=0.01,
            loss=1.0,
            metrics={"val_loss": 1.0},
        )
        manager.save_checkpoint(sample_weights, state1, force=True)
        assert not manager.should_stop_early

        # Worse checkpoint - increment no improvement counter
        state2 = TrainingState(
            epoch=2,
            global_step=200,
            learning_rate=0.01,
            loss=1.1,
            metrics={"val_loss": 1.1},
        )
        manager.save_checkpoint(sample_weights, state2, force=True)
        assert manager.no_improvement_count >= 1

        # Still worse - increment again
        state3 = TrainingState(
            epoch=3,
            global_step=300,
            learning_rate=0.01,
            loss=1.2,
            metrics={"val_loss": 1.2},
        )
        manager.save_checkpoint(sample_weights, state3, force=True)
        assert manager.no_improvement_count >= 2

        # Still no improvement - should trigger early stopping
        state4 = TrainingState(
            epoch=4,
            global_step=400,
            learning_rate=0.01,
            loss=1.3,
            metrics={"val_loss": 1.3},
        )
        manager.save_checkpoint(sample_weights, state4, force=True)
        assert manager.no_improvement_count >= 3
        assert manager.should_stop_early

    def test_early_stopping_resets_on_improvement(self, temp_repo, sample_weights):
        """Test that improvement resets the no-improvement counter."""
        config = CheckpointConfig(
            save_on_best_metric="val_loss",
            minimize_metric=True,
            early_stopping_patience=3,
            auto_commit=True,
        )

        manager = CheckpointManager(
            repository=temp_repo, config=config, model_name="TestModel"
        )

        # First checkpoint
        state1 = TrainingState(
            epoch=1,
            global_step=100,
            learning_rate=0.01,
            loss=1.0,
            metrics={"val_loss": 1.0},
        )
        manager.save_checkpoint(sample_weights, state1, force=True)

        # Worse checkpoint
        state2 = TrainingState(
            epoch=2,
            global_step=200,
            learning_rate=0.01,
            loss=1.1,
            metrics={"val_loss": 1.1},
        )
        manager.save_checkpoint(sample_weights, state2, force=True)
        assert manager.no_improvement_count >= 1

        # Better checkpoint - should reset counter
        state3 = TrainingState(
            epoch=3,
            global_step=300,
            learning_rate=0.01,
            loss=0.9,
            metrics={"val_loss": 0.9},
        )
        manager.save_checkpoint(sample_weights, state3, force=True)
        assert manager.no_improvement_count == 0
        assert not manager.should_stop_early

    def test_early_stopping_threshold(self, temp_repo, sample_weights):
        """Test that threshold affects what counts as improvement."""
        config = CheckpointConfig(
            save_on_best_metric="val_loss",
            minimize_metric=True,
            early_stopping_patience=3,
            early_stopping_threshold=0.1,  # Require 0.1 improvement
            auto_commit=True,
        )

        manager = CheckpointManager(
            repository=temp_repo, config=config, model_name="TestModel"
        )

        # First checkpoint at 1.0
        state1 = TrainingState(
            epoch=1,
            global_step=100,
            learning_rate=0.01,
            loss=1.0,
            metrics={"val_loss": 1.0},
        )
        manager.save_checkpoint(sample_weights, state1, force=True)

        # Tiny improvement (0.95) - not enough to meet threshold
        state2 = TrainingState(
            epoch=2,
            global_step=200,
            learning_rate=0.01,
            loss=0.95,
            metrics={"val_loss": 0.95},
        )
        manager.save_checkpoint(sample_weights, state2, force=True)
        # This is still an improvement for _is_best_checkpoint, but not for threshold
        # The no_improvement_count should be 1 because threshold wasn't met
        assert manager.no_improvement_count >= 1


class TestCheckpointDiff:
    """Test checkpoint diff functionality."""

    def test_diff_identical_checkpoints(self, temp_repo, sample_weights):
        """Test diff returns identical for same weights."""
        config = CheckpointConfig(auto_commit=True)
        manager = CheckpointManager(
            repository=temp_repo, config=config, model_name="TestModel"
        )

        # Save two identical checkpoints
        state1 = TrainingState(epoch=1, global_step=100, learning_rate=0.01, loss=0.5)
        commit1 = manager.save_checkpoint(sample_weights, state1, force=True)

        state2 = TrainingState(epoch=2, global_step=200, learning_rate=0.01, loss=0.4)
        commit2 = manager.save_checkpoint(sample_weights, state2, force=True)

        # Diff should show identical
        diff = manager.diff_checkpoints(commit1, commit2)
        assert diff["identical"]
        assert len(diff["changed"]) == 0
        assert len(diff["added"]) == 0
        assert len(diff["removed"]) == 0

    def test_diff_changed_weights(self, temp_repo, sample_weights):
        """Test diff detects changed weights."""
        config = CheckpointConfig(auto_commit=True)
        manager = CheckpointManager(
            repository=temp_repo, config=config, model_name="TestModel"
        )

        # Save first checkpoint
        state1 = TrainingState(epoch=1, global_step=100, learning_rate=0.01, loss=0.5)
        commit1 = manager.save_checkpoint(sample_weights, state1, force=True)

        # Modify weights and save again
        modified_weights = {
            "layer1.weight": WeightTensor(
                data=np.random.randn(10, 5).astype(np.float32),  # New random data
                metadata=WeightMetadata(
                    name="layer1.weight",
                    shape=(10, 5),
                    dtype=np.float32,
                ),
            ),
            "layer1.bias": sample_weights["layer1.bias"],  # Keep bias same
        }

        state2 = TrainingState(epoch=2, global_step=200, learning_rate=0.01, loss=0.4)
        commit2 = manager.save_checkpoint(modified_weights, state2, force=True)

        # Diff should show layer1.weight changed
        diff = manager.diff_checkpoints(commit1, commit2)
        assert not diff["identical"]
        assert "layer1.weight" in diff["changed"]
        assert "layer1.weight" in diff["similarity"]
        # Cosine similarity is between -1 and 1
        assert -1 <= diff["similarity"]["layer1.weight"] <= 1

    def test_diff_added_removed_weights(self, temp_repo, sample_weights):
        """Test diff detects added and removed weights."""
        config = CheckpointConfig(auto_commit=True)
        manager = CheckpointManager(
            repository=temp_repo, config=config, model_name="TestModel"
        )

        # Save first checkpoint
        state1 = TrainingState(epoch=1, global_step=100, learning_rate=0.01, loss=0.5)
        commit1 = manager.save_checkpoint(sample_weights, state1, force=True)

        # Different set of weights
        different_weights = {
            "layer2.weight": WeightTensor(
                data=np.random.randn(20, 10).astype(np.float32),
                metadata=WeightMetadata(
                    name="layer2.weight",
                    shape=(20, 10),
                    dtype=np.float32,
                ),
            ),
        }

        state2 = TrainingState(epoch=2, global_step=200, learning_rate=0.01, loss=0.4)
        commit2 = manager.save_checkpoint(different_weights, state2, force=True)

        # Diff should show additions and removals
        diff = manager.diff_checkpoints(commit1, commit2)
        assert not diff["identical"]
        assert "layer2.weight" in diff["added"]
        assert "layer1.weight" in diff["removed"]
        assert "layer1.bias" in diff["removed"]


class TestCheckpointer:
    """Test the new Checkpointer class."""

    def test_checkpointer_initialization(self, temp_repo):
        """Test Checkpointer initialization."""
        # Mock torch
        torch = Mock()
        torch.nn.Module = Mock

        model = Mock()
        model.__class__.__name__ = "TestModel"
        model.named_parameters = Mock(return_value=[])

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                from coral.integrations.pytorch import Checkpointer

                ckpt = Checkpointer(
                    model,
                    temp_repo,
                    "test-experiment",
                    every_n_epochs=1,
                    on_best="val_loss",
                )

                assert ckpt.experiment == "test-experiment"
                assert ckpt.epoch == 0
                assert ckpt.global_step == 0
                assert ckpt.repo is temp_repo

    def test_checkpointer_with_path(self):
        """Test Checkpointer with path string."""
        temp_dir = tempfile.mkdtemp()
        try:
            torch = Mock()
            torch.nn.Module = Mock

            model = Mock()
            model.__class__.__name__ = "TestModel"
            model.named_parameters = Mock(return_value=[])

            with patch("coral.integrations.pytorch.torch", torch):
                with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                    from coral.integrations.pytorch import Checkpointer

                    ckpt = Checkpointer(
                        model,
                        temp_dir,
                        "test-experiment",
                    )

                    assert ckpt.experiment == "test-experiment"
                    # Repo should be initialized from path
                    assert ckpt.repo is not None
        finally:
            shutil.rmtree(temp_dir)

    def test_checkpointer_context_manager(self, temp_repo):
        """Test Checkpointer context manager."""
        torch = Mock()
        torch.nn.Module = Mock

        model = Mock()
        model.__class__.__name__ = "TestModel"
        model.named_parameters = Mock(return_value=[])

        tracker = Mock()
        tracker.start_run = Mock()
        tracker.end_run = Mock()
        tracker.is_run_active = True

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                from coral.integrations.pytorch import Checkpointer

                ckpt = Checkpointer(
                    model,
                    temp_repo,
                    "test-experiment",
                    tracker=tracker,
                )

                with ckpt:
                    tracker.start_run.assert_called_once()

                tracker.end_run.assert_called_with("completed")

    def test_checkpointer_properties(self, temp_repo):
        """Test Checkpointer properties."""
        torch = Mock()
        torch.nn.Module = Mock

        model = Mock()
        model.__class__.__name__ = "TestModel"
        model.named_parameters = Mock(return_value=[])

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                from coral.integrations.pytorch import Checkpointer

                ckpt = Checkpointer(model, temp_repo, "test-experiment")

                assert ckpt.epoch == 0
                assert ckpt.global_step == 0
                assert ckpt.best_commit is None
                assert isinstance(ckpt.metrics, dict)


class TestUnifiedLoadSave:
    """Test unified load and save functions."""

    def test_save_function(self, temp_repo, sample_weights):
        """Test unified save function."""
        torch = Mock()
        torch.nn.Module = Mock
        torch.from_numpy = Mock(side_effect=lambda x: Mock(data=x))

        model = Mock()
        model.__class__.__name__ = "TestModel"
        model.named_parameters = Mock(
            return_value=[
                (
                    "layer1.weight",
                    Mock(
                        detach=Mock(
                            return_value=Mock(
                                cpu=Mock(
                                    return_value=Mock(
                                        numpy=Mock(
                                            return_value=sample_weights[
                                                "layer1.weight"
                                            ].data
                                        )
                                    )
                                )
                            )
                        )
                    ),
                ),
            ]
        )

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                from coral.integrations.pytorch import save

                result = save(model, temp_repo, "Test save")

                assert "commit_hash" in result
                assert result["weights_saved"] == 1

    def test_load_function(self, temp_repo, sample_weights):
        """Test unified load function."""
        # First save some weights
        temp_repo.stage_weights(sample_weights)
        temp_repo.commit("Initial weights")

        torch = Mock()
        torch.nn.Module = Mock
        torch.from_numpy = Mock(side_effect=lambda x: Mock(data=x))

        model = Mock()
        model.state_dict = Mock(
            return_value={"layer1.weight": Mock(), "layer1.bias": Mock()}
        )
        model.load_state_dict = Mock()

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                from coral.integrations.pytorch import load

                result = load(model, temp_repo, strict=False)

                assert "loaded" in result
                assert "matched" in result


class TestCallbackAliases:
    """Test callback class aliases."""

    def test_lightning_callback_alias(self):
        """Test CoralLightningCallback is alias for CoralCallback."""
        try:
            from coral.integrations.lightning import (
                CoralCallback,
                CoralLightningCallback,
            )

            assert CoralLightningCallback is CoralCallback
        except ImportError:
            pytest.skip("PyTorch Lightning not installed")

    def test_hf_callback_alias(self):
        """Test CoralHFCallback is alias for CoralTrainerCallback."""
        try:
            from coral.integrations.hf_trainer import (
                CoralHFCallback,
                CoralTrainerCallback,
            )

            assert CoralHFCallback is CoralTrainerCallback
        except ImportError:
            pytest.skip("HuggingFace Transformers not installed")

    def test_callback_accepts_repository(self, temp_repo):
        """Test callbacks accept Repository object."""
        try:
            from coral.integrations.lightning import CoralCallback

            callback = CoralCallback(repo=temp_repo)
            assert callback._repo is temp_repo
        except ImportError:
            pytest.skip("PyTorch Lightning not installed")
