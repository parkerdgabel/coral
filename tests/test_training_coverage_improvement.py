"""Test coverage improvement for training module."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coral.training.training_state import TrainingState
from coral.training.checkpoint_manager import (
    CheckpointManager,
    CheckpointPolicy,
    CheckpointMetrics,
)
from coral.core.weight_tensor import WeightTensor


class TestTrainingStateCoverage:
    """Tests to improve coverage for TrainingState class."""

    def test_training_state_initialization(self):
        """Test TrainingState initialization with various parameters."""
        state = TrainingState(
            epoch=5,
            global_step=1000,
            optimizer_state={"lr": 0.001, "momentum": 0.9},
            model_state={"layer1": "weights", "layer2": "weights"},
            metrics={"loss": 0.5, "accuracy": 0.95},
            custom_data={"experiment": "test", "version": 1},
        )
        
        assert state.epoch == 5
        assert state.global_step == 1000
        assert state.optimizer_state["lr"] == 0.001
        assert state.model_state["layer1"] == "weights"
        assert state.metrics["loss"] == 0.5
        assert state.custom_data["experiment"] == "test"

    def test_training_state_to_dict(self):
        """Test TrainingState serialization to dictionary."""
        state = TrainingState(
            epoch=3,
            global_step=500,
            optimizer_state={"lr": 0.01},
            metrics={"loss": 0.3},
        )
        
        state_dict = state.to_dict()
        assert state_dict["epoch"] == 3
        assert state_dict["global_step"] == 500
        assert state_dict["optimizer_state"]["lr"] == 0.01
        assert state_dict["metrics"]["loss"] == 0.3

    def test_training_state_from_dict(self):
        """Test TrainingState deserialization from dictionary."""
        state_dict = {
            "epoch": 10,
            "global_step": 2000,
            "optimizer_state": {"lr": 0.001, "beta1": 0.9},
            "model_state": {"encoder": "weights"},
            "metrics": {"loss": 0.1, "val_loss": 0.15},
            "custom_data": {"seed": 42},
        }
        
        state = TrainingState.from_dict(state_dict)
        assert state.epoch == 10
        assert state.global_step == 2000
        assert state.optimizer_state["beta1"] == 0.9
        assert state.metrics["val_loss"] == 0.15

    def test_training_state_save_load(self, tmp_path):
        """Test saving and loading TrainingState to/from file."""
        state = TrainingState(
            epoch=7,
            global_step=1400,
            metrics={"train_loss": 0.25, "val_loss": 0.3},
        )
        
        # Save to file
        filepath = tmp_path / "training_state.json"
        state.save(str(filepath))
        assert filepath.exists()
        
        # Load from file
        loaded_state = TrainingState.load(str(filepath))
        assert loaded_state.epoch == 7
        assert loaded_state.global_step == 1400
        assert loaded_state.metrics["train_loss"] == 0.25

    def test_training_state_update(self):
        """Test updating TrainingState."""
        state = TrainingState()
        
        # Update with new values
        state.update(
            epoch=2,
            global_step=200,
            metrics={"loss": 0.4},
            custom_data={"note": "checkpoint"},
        )
        
        assert state.epoch == 2
        assert state.global_step == 200
        assert state.metrics["loss"] == 0.4
        assert state.custom_data["note"] == "checkpoint"

    def test_training_state_edge_cases(self):
        """Test TrainingState with edge cases."""
        # Empty state
        state = TrainingState()
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.optimizer_state == {}
        assert state.model_state == {}
        assert state.metrics == {}
        assert state.custom_data == {}
        
        # State with None values
        state_dict = {
            "epoch": None,
            "global_step": None,
            "optimizer_state": None,
            "model_state": None,
            "metrics": None,
            "custom_data": None,
        }
        state = TrainingState.from_dict(state_dict)
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.optimizer_state == {}


class TestCheckpointManagerCoverage:
    """Tests to improve coverage for CheckpointManager class."""

    @pytest.fixture
    def mock_repository(self):
        """Create a mock repository."""
        repo = MagicMock()
        repo.stage_weight = MagicMock()
        repo.commit = MagicMock(return_value="commit_hash_123")
        repo.get_weight = MagicMock()
        repo.checkout = MagicMock()
        repo.list_tags = MagicMock(return_value=[])
        repo.tag = MagicMock()
        return repo

    def test_checkpoint_policy_initialization(self):
        """Test CheckpointPolicy initialization."""
        policy = CheckpointPolicy(
            save_every_n_steps=100,
            save_every_n_epochs=5,
            max_checkpoints=10,
            save_on_improvement=True,
            metric_name="val_loss",
            metric_mode="min",
        )
        
        assert policy.save_every_n_steps == 100
        assert policy.save_every_n_epochs == 5
        assert policy.max_checkpoints == 10
        assert policy.save_on_improvement is True
        assert policy.metric_name == "val_loss"
        assert policy.metric_mode == "min"

    def test_checkpoint_manager_initialization(self, mock_repository):
        """Test CheckpointManager initialization."""
        manager = CheckpointManager(
            repository=mock_repository,
            checkpoint_dir="checkpoints",
            policy=CheckpointPolicy(save_every_n_steps=50),
        )
        
        assert manager.repository == mock_repository
        assert manager.checkpoint_dir == "checkpoints"
        assert manager.policy.save_every_n_steps == 50

    def test_should_save_checkpoint_by_steps(self, mock_repository):
        """Test checkpoint saving decision based on steps."""
        policy = CheckpointPolicy(save_every_n_steps=100)
        manager = CheckpointManager(mock_repository, policy=policy)
        
        # Should not save at step 50
        assert not manager.should_save_checkpoint(global_step=50)
        
        # Should save at step 100
        assert manager.should_save_checkpoint(global_step=100)
        
        # Should save at step 200
        assert manager.should_save_checkpoint(global_step=200)

    def test_should_save_checkpoint_by_epochs(self, mock_repository):
        """Test checkpoint saving decision based on epochs."""
        policy = CheckpointPolicy(save_every_n_epochs=2)
        manager = CheckpointManager(mock_repository, policy=policy)
        
        # Should not save at epoch 1
        assert not manager.should_save_checkpoint(epoch=1)
        
        # Should save at epoch 2
        assert manager.should_save_checkpoint(epoch=2)
        
        # Should save at epoch 4
        assert manager.should_save_checkpoint(epoch=4)

    def test_should_save_checkpoint_on_improvement(self, mock_repository):
        """Test checkpoint saving decision based on metric improvement."""
        policy = CheckpointPolicy(
            save_on_improvement=True,
            metric_name="loss",
            metric_mode="min",
        )
        manager = CheckpointManager(mock_repository, policy=policy)
        
        # First checkpoint should always save
        metrics = CheckpointMetrics(loss=0.5)
        assert manager.should_save_checkpoint(metrics=metrics)
        
        # Update best metric
        manager._update_best_metric(0.5)
        
        # Should not save with worse metric
        metrics = CheckpointMetrics(loss=0.6)
        assert not manager.should_save_checkpoint(metrics=metrics)
        
        # Should save with better metric
        metrics = CheckpointMetrics(loss=0.4)
        assert manager.should_save_checkpoint(metrics=metrics)

    def test_save_checkpoint(self, mock_repository, tmp_path):
        """Test saving a checkpoint."""
        manager = CheckpointManager(
            mock_repository,
            checkpoint_dir=str(tmp_path),
            policy=CheckpointPolicy(),
        )
        
        # Create model weights
        weights = {
            "layer1": WeightTensor(data=np.random.randn(10, 10)),
            "layer2": WeightTensor(data=np.random.randn(5, 5)),
        }
        
        # Create training state
        state = TrainingState(epoch=5, global_step=1000, metrics={"loss": 0.3})
        
        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            weights=weights,
            training_state=state,
            checkpoint_name="checkpoint_1000",
        )
        
        # Verify repository interactions
        assert mock_repository.stage_weight.call_count == 2
        assert mock_repository.commit.called
        
        # Verify checkpoint info saved
        assert checkpoint_path == str(tmp_path / "checkpoint_1000")

    def test_load_checkpoint(self, mock_repository, tmp_path):
        """Test loading a checkpoint."""
        manager = CheckpointManager(
            mock_repository,
            checkpoint_dir=str(tmp_path),
            policy=CheckpointPolicy(),
        )
        
        # Create checkpoint info file
        checkpoint_dir = tmp_path / "checkpoint_1000"
        checkpoint_dir.mkdir()
        checkpoint_info = {
            "commit_hash": "abc123",
            "weight_names": ["layer1", "layer2"],
            "training_state": {
                "epoch": 5,
                "global_step": 1000,
                "metrics": {"loss": 0.3},
            },
        }
        
        with open(checkpoint_dir / "checkpoint_info.json", "w") as f:
            json.dump(checkpoint_info, f)
        
        # Mock weight retrieval
        mock_repository.get_weight.side_effect = [
            WeightTensor(data=np.random.randn(10, 10)),
            WeightTensor(data=np.random.randn(5, 5)),
        ]
        
        # Load checkpoint
        weights, state = manager.load_checkpoint("checkpoint_1000")
        
        # Verify
        assert len(weights) == 2
        assert "layer1" in weights
        assert "layer2" in weights
        assert state.epoch == 5
        assert state.global_step == 1000

    def test_checkpoint_cleanup(self, mock_repository, tmp_path):
        """Test checkpoint cleanup when max_checkpoints is exceeded."""
        policy = CheckpointPolicy(max_checkpoints=2)
        manager = CheckpointManager(
            mock_repository,
            checkpoint_dir=str(tmp_path),
            policy=policy,
        )
        
        # Create multiple checkpoint directories
        for i in range(4):
            checkpoint_dir = tmp_path / f"checkpoint_{i}"
            checkpoint_dir.mkdir()
            info = {"timestamp": i}
            with open(checkpoint_dir / "checkpoint_info.json", "w") as f:
                json.dump(info, f)
        
        # Add checkpoints to manager
        manager._checkpoints = [
            str(tmp_path / f"checkpoint_{i}") for i in range(4)
        ]
        
        # Cleanup
        manager._cleanup_old_checkpoints()
        
        # Should keep only 2 most recent
        remaining = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(remaining) == 2
        assert (tmp_path / "checkpoint_2").exists()
        assert (tmp_path / "checkpoint_3").exists()

    def test_list_checkpoints(self, mock_repository, tmp_path):
        """Test listing available checkpoints."""
        manager = CheckpointManager(
            mock_repository,
            checkpoint_dir=str(tmp_path),
            policy=CheckpointPolicy(),
        )
        
        # Create checkpoint directories
        checkpoints = []
        for i in range(3):
            checkpoint_dir = tmp_path / f"checkpoint_{i*100}"
            checkpoint_dir.mkdir()
            info = {
                "timestamp": f"2024-01-0{i+1}T00:00:00",
                "epoch": i,
                "global_step": i * 100,
                "metrics": {"loss": 1.0 - i * 0.1},
            }
            with open(checkpoint_dir / "checkpoint_info.json", "w") as f:
                json.dump(info, f)
            checkpoints.append(f"checkpoint_{i*100}")
        
        # List checkpoints
        listed = manager.list_checkpoints()
        
        assert len(listed) == 3
        for cp in checkpoints:
            assert cp in listed

    def test_get_best_checkpoint(self, mock_repository, tmp_path):
        """Test getting the best checkpoint based on metric."""
        policy = CheckpointPolicy(
            save_on_improvement=True,
            metric_name="accuracy",
            metric_mode="max",
        )
        manager = CheckpointManager(
            mock_repository,
            checkpoint_dir=str(tmp_path),
            policy=policy,
        )
        
        # Create checkpoints with different metrics
        for i, acc in enumerate([0.8, 0.9, 0.85]):
            checkpoint_dir = tmp_path / f"checkpoint_{i}"
            checkpoint_dir.mkdir()
            info = {
                "metrics": {"accuracy": acc},
                "epoch": i,
            }
            with open(checkpoint_dir / "checkpoint_info.json", "w") as f:
                json.dump(info, f)
        
        # Get best checkpoint
        best = manager.get_best_checkpoint()
        
        assert best == "checkpoint_1"  # Has accuracy 0.9

    def test_checkpoint_callbacks(self, mock_repository):
        """Test checkpoint callbacks."""
        callback_data = {"called": False, "checkpoint_path": None}
        
        def on_checkpoint_saved(checkpoint_path):
            callback_data["called"] = True
            callback_data["checkpoint_path"] = checkpoint_path
        
        manager = CheckpointManager(
            mock_repository,
            policy=CheckpointPolicy(),
            on_checkpoint_saved=on_checkpoint_saved,
        )
        
        # Trigger callback
        manager._trigger_callback("test_checkpoint")
        
        assert callback_data["called"]
        assert callback_data["checkpoint_path"] == "test_checkpoint"

    def test_checkpoint_metrics(self):
        """Test CheckpointMetrics class."""
        # Test with keyword arguments
        metrics = CheckpointMetrics(loss=0.5, accuracy=0.95, val_loss=0.6)
        assert metrics.loss == 0.5
        assert metrics.accuracy == 0.95
        assert metrics.val_loss == 0.6
        
        # Test conversion to dict
        metrics_dict = metrics.to_dict()
        assert metrics_dict["loss"] == 0.5
        assert metrics_dict["accuracy"] == 0.95
        assert metrics_dict["val_loss"] == 0.6
        
        # Test from dict
        new_metrics = CheckpointMetrics.from_dict(metrics_dict)
        assert new_metrics.loss == 0.5
        assert new_metrics.accuracy == 0.95

    def test_resume_from_checkpoint(self, mock_repository, tmp_path):
        """Test resuming training from checkpoint."""
        manager = CheckpointManager(
            mock_repository,
            checkpoint_dir=str(tmp_path),
            policy=CheckpointPolicy(),
        )
        
        # Create a checkpoint
        checkpoint_dir = tmp_path / "checkpoint_latest"
        checkpoint_dir.mkdir()
        checkpoint_info = {
            "commit_hash": "xyz789",
            "weight_names": ["model"],
            "training_state": {
                "epoch": 10,
                "global_step": 2000,
                "optimizer_state": {"lr": 0.001},
                "metrics": {"loss": 0.1},
            },
        }
        
        with open(checkpoint_dir / "checkpoint_info.json", "w") as f:
            json.dump(checkpoint_info, f)
        
        # Mock weight retrieval
        mock_repository.get_weight.return_value = WeightTensor(
            data=np.random.randn(20, 20)
        )
        
        # Resume from checkpoint
        weights, state = manager.resume_from_latest()
        
        assert weights is not None
        assert state is not None
        assert state.epoch == 10
        assert state.global_step == 2000
        assert state.optimizer_state["lr"] == 0.001

    def test_auto_checkpoint_tagging(self, mock_repository):
        """Test automatic tagging of best checkpoints."""
        policy = CheckpointPolicy(
            save_on_improvement=True,
            metric_name="loss",
            metric_mode="min",
            tag_best_checkpoint=True,
        )
        manager = CheckpointManager(
            mock_repository,
            policy=policy,
        )
        
        # Simulate saving a best checkpoint
        manager._tag_best_checkpoint("commit_123", 0.1)
        
        # Verify tag was created
        mock_repository.tag.assert_called_once()
        call_args = mock_repository.tag.call_args
        assert "best" in call_args[0][1]  # Tag name contains "best"
        assert call_args[0][0] == "commit_123"  # Correct commit hash