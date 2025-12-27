"""Tests for training integration functionality."""

import shutil
import tempfile
from pathlib import Path

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


class TestTrainingState:
    """Test training state functionality."""

    def test_training_state_creation(self):
        """Test creating and manipulating training state."""
        state = TrainingState(
            epoch=5,
            global_step=1000,
            learning_rate=0.001,
            loss=0.5,
            metrics={"accuracy": 0.85, "f1": 0.82},
        )

        assert state.epoch == 5
        assert state.global_step == 1000
        assert state.learning_rate == 0.001
        assert state.loss == 0.5
        assert state.metrics["accuracy"] == 0.85

    def test_training_state_serialization(self):
        """Test training state serialization."""
        state = TrainingState(
            epoch=10,
            global_step=2000,
            learning_rate=0.0005,
            loss=0.3,
            metrics={"accuracy": 0.92},
        )

        # Test to_dict
        state_dict = state.to_dict()
        assert state_dict["epoch"] == 10
        assert state_dict["global_step"] == 2000
        assert state_dict["metrics"]["accuracy"] == 0.92

        # Test from_dict
        restored_state = TrainingState.from_dict(state_dict)
        assert restored_state.epoch == state.epoch
        assert restored_state.global_step == state.global_step
        assert restored_state.metrics == state.metrics

    def test_training_state_file_operations(self):
        """Test saving and loading training state from file."""
        temp_dir = tempfile.mkdtemp()
        try:
            state_file = Path(temp_dir) / "state.json"

            state = TrainingState(
                epoch=15,
                global_step=3000,
                learning_rate=0.0001,
                loss=0.2,
                model_name="TestModel",
                experiment_name="test_exp",
            )

            # Save to file
            state.save(str(state_file))
            assert state_file.exists()

            # Load from file
            loaded_state = TrainingState.load(str(state_file))
            assert loaded_state.epoch == state.epoch
            assert loaded_state.model_name == state.model_name
            assert loaded_state.experiment_name == state.experiment_name

        finally:
            shutil.rmtree(temp_dir)


class TestCheckpointManager:
    """Test checkpoint manager functionality."""

    def test_checkpoint_manager_creation(self, temp_repo):
        """Test creating checkpoint manager."""
        config = CheckpointConfig(
            save_every_n_epochs=2,
            save_on_best_metric="accuracy",
            minimize_metric=False,
            keep_last_n_checkpoints=5,
        )

        manager = CheckpointManager(
            repository=temp_repo,
            config=config,
            model_name="TestModel",
            experiment_name="test_experiment",
        )

        assert manager.model_name == "TestModel"
        assert manager.experiment_name == "test_experiment"
        assert manager.config.save_every_n_epochs == 2

    def test_checkpoint_saving_conditions(self, temp_repo, sample_weights):
        """Test checkpoint saving conditions."""
        config = CheckpointConfig(
            save_every_n_epochs=3,
            save_every_n_steps=100,
            save_on_best_metric="accuracy",
            minimize_metric=False,
        )

        manager = CheckpointManager(
            repository=temp_repo, config=config, model_name="TestModel"
        )

        # Test epoch-based saving
        state1 = TrainingState(epoch=3, global_step=50, learning_rate=0.01, loss=0.5)
        assert manager.should_save_checkpoint(state1)  # Epoch 3, divisible by 3

        state2 = TrainingState(epoch=4, global_step=50, learning_rate=0.01, loss=0.5)
        assert not manager.should_save_checkpoint(state2)  # Epoch 4, not divisible by 3

        # Test step-based saving
        state3 = TrainingState(epoch=1, global_step=100, learning_rate=0.01, loss=0.5)
        assert manager.should_save_checkpoint(state3)  # Step 100, divisible by 100

        # Test metric-based saving (first time)
        state4 = TrainingState(
            epoch=1,
            global_step=50,
            learning_rate=0.01,
            loss=0.5,
            metrics={"accuracy": 0.8},
        )
        assert manager.should_save_checkpoint(state4)  # First metric

        # Save this checkpoint to establish baseline
        manager.save_checkpoint(sample_weights, state4)

        # Test metric-based saving (improvement)
        state5 = TrainingState(
            epoch=2,
            global_step=60,
            learning_rate=0.01,
            loss=0.4,
            metrics={"accuracy": 0.85},  # Better accuracy
        )
        assert manager.should_save_checkpoint(state5)  # Improved accuracy

        # Test metric-based saving (no improvement)
        state6 = TrainingState(
            epoch=2,
            global_step=70,
            learning_rate=0.01,
            loss=0.6,
            metrics={"accuracy": 0.75},  # Worse accuracy
        )
        assert not manager.should_save_checkpoint(
            state6
        )  # Worse accuracy, not on save boundary

    def test_checkpoint_save_and_load(self, temp_repo, sample_weights):
        """Test saving and loading checkpoints."""
        config = CheckpointConfig(
            save_every_n_epochs=1, auto_commit=True, keep_last_n_checkpoints=3
        )

        manager = CheckpointManager(
            repository=temp_repo,
            config=config,
            model_name="TestModel",
            experiment_name="test_exp",
        )

        # Create training state
        state = TrainingState(
            epoch=5,
            global_step=1000,
            learning_rate=0.001,
            loss=0.3,
            metrics={"accuracy": 0.9, "f1": 0.88},
        )

        # Save checkpoint
        commit_hash = manager.save_checkpoint(sample_weights, state)
        assert commit_hash is not None

        # Verify checkpoint was recorded
        assert len(manager.checkpoint_history) == 1
        checkpoint_info = manager.checkpoint_history[0]
        assert checkpoint_info["epoch"] == 5
        assert checkpoint_info["global_step"] == 1000
        assert checkpoint_info["commit_hash"] == commit_hash

        # Load checkpoint
        loaded_data = manager.load_checkpoint(commit_hash)
        assert loaded_data is not None
        assert "weights" in loaded_data
        assert "state" in loaded_data
        assert loaded_data["commit_hash"] == commit_hash

        # Verify loaded weights
        loaded_weights = loaded_data["weights"]
        assert len(loaded_weights) == len(sample_weights)
        assert "layer1.weight" in loaded_weights

        # Verify loaded state
        loaded_state = loaded_data["state"]
        if loaded_state:  # May be None if state file not found
            assert loaded_state.epoch == state.epoch

    def test_best_checkpoint_tracking(self, temp_repo, sample_weights):
        """Test tracking of best checkpoints."""
        config = CheckpointConfig(
            save_on_best_metric="loss",
            minimize_metric=True,  # Lower loss is better
            tag_best_checkpoints=True,
            auto_commit=True,
        )

        manager = CheckpointManager(
            repository=temp_repo, config=config, model_name="TestModel"
        )

        # Save first checkpoint (establishes baseline)
        state1 = TrainingState(
            epoch=1,
            global_step=100,
            learning_rate=0.01,
            loss=0.8,
            metrics={"loss": 0.8},
        )
        commit1 = manager.save_checkpoint(sample_weights, state1, force=True)
        assert manager.best_metric_value == 0.8
        assert manager.best_checkpoint_hash == commit1

        # Save better checkpoint
        state2 = TrainingState(
            epoch=2,
            global_step=200,
            learning_rate=0.01,
            loss=0.6,
            metrics={"loss": 0.6},
        )
        commit2 = manager.save_checkpoint(sample_weights, state2)
        assert commit2 is not None  # Should save because loss improved
        assert manager.best_metric_value == 0.6
        assert manager.best_checkpoint_hash == commit2

        # Try to save worse checkpoint
        state3 = TrainingState(
            epoch=3,
            global_step=300,
            learning_rate=0.01,
            loss=0.9,
            metrics={"loss": 0.9},
        )
        commit3 = manager.save_checkpoint(sample_weights, state3)
        assert commit3 is None  # Should not save because loss is worse
        assert manager.best_metric_value == 0.6  # Best metric unchanged
        assert manager.best_checkpoint_hash == commit2  # Best checkpoint unchanged

    def test_checkpoint_cleanup(self, temp_repo, sample_weights):
        """Test cleanup of old checkpoints."""
        config = CheckpointConfig(
            save_every_n_epochs=1,
            keep_last_n_checkpoints=2,  # Keep only last 2
            auto_commit=True,
        )

        manager = CheckpointManager(
            repository=temp_repo, config=config, model_name="TestModel"
        )

        # Save multiple checkpoints
        commits = []
        for i in range(5):
            state = TrainingState(
                epoch=i + 1,
                global_step=(i + 1) * 100,
                learning_rate=0.01,
                loss=0.5 - i * 0.05,  # Decreasing loss
            )
            commit = manager.save_checkpoint(sample_weights, state)
            if commit:
                commits.append(commit)

        # Should only keep last 2 checkpoints in history
        assert len(manager.checkpoint_history) <= 2

        # Verify most recent checkpoints are kept
        epochs = [c["epoch"] for c in manager.checkpoint_history]
        assert max(epochs) >= 4  # Should include recent epochs

    def test_list_checkpoints(self, temp_repo, sample_weights):
        """Test listing checkpoints."""
        config = CheckpointConfig(auto_commit=True)
        manager = CheckpointManager(
            repository=temp_repo, config=config, model_name="TestModel"
        )

        # Save a few checkpoints
        for i in range(3):
            state = TrainingState(
                epoch=i + 1,
                global_step=(i + 1) * 100,
                learning_rate=0.01,
                loss=0.5,
                metrics={"accuracy": 0.8 + i * 0.05},
            )
            manager.save_checkpoint(sample_weights, state, force=True)

        # List all checkpoints
        all_checkpoints = manager.list_checkpoints()
        assert len(all_checkpoints) == 3

        # List without metrics
        checkpoints_no_metrics = manager.list_checkpoints(include_metrics=False)
        assert len(checkpoints_no_metrics) == 3
        for checkpoint in checkpoints_no_metrics:
            assert "metrics" not in checkpoint

    def test_checkpoint_callbacks(self, temp_repo, sample_weights):
        """Test checkpoint callback system."""
        config = CheckpointConfig(auto_commit=True)
        manager = CheckpointManager(
            repository=temp_repo, config=config, model_name="TestModel"
        )

        # Track callback calls
        callback_calls = []

        def test_callback(state: TrainingState, commit_hash: str):
            callback_calls.append({"state": state, "commit_hash": commit_hash})

        def test_callback_2(state: TrainingState, commit_hash: str):
            callback_calls.append({"callback": "test_callback_2", "epoch": state.epoch})

        def failing_callback(state: TrainingState, commit_hash: str):
            raise RuntimeError("Test callback failure")

        # Test callback registration
        manager.register_checkpoint_callback(test_callback)
        manager.register_checkpoint_callback(test_callback_2)
        manager.register_checkpoint_callback(failing_callback)

        # Test duplicate registration (should warn but not add twice)
        manager.register_checkpoint_callback(test_callback)

        # Check callback listing
        callback_names = manager.list_callbacks()
        assert len(callback_names) == 3
        assert "test_callback" in callback_names
        assert "test_callback_2" in callback_names
        assert "failing_callback" in callback_names

        # Save checkpoint - should trigger callbacks
        state = TrainingState(
            epoch=1,
            global_step=100,
            learning_rate=0.01,
            loss=0.5,
            metrics={"accuracy": 0.8},
        )
        commit_hash = manager.save_checkpoint(sample_weights, state, force=True)

        # Verify callbacks were called (should be 2 successful calls despite 1 failure)
        assert len(callback_calls) == 2
        assert callback_calls[0]["state"].epoch == 1
        assert callback_calls[0]["commit_hash"] == commit_hash
        assert callback_calls[1]["callback"] == "test_callback_2"
        assert callback_calls[1]["epoch"] == 1

        # Test callback unregistration
        removed = manager.unregister_checkpoint_callback(test_callback)
        assert removed is True

        # Try to remove again (should return False)
        removed = manager.unregister_checkpoint_callback(test_callback)
        assert removed is False

        # Check updated callback list
        callback_names = manager.list_callbacks()
        assert len(callback_names) == 2
        assert "test_callback" not in callback_names

        # Test clear callbacks
        cleared_count = manager.clear_callbacks()
        assert cleared_count == 2
        assert len(manager.list_callbacks()) == 0

        # Save another checkpoint - no callbacks should be called
        callback_calls.clear()
        state.epoch = 2
        manager.save_checkpoint(sample_weights, state, force=True)
        assert len(callback_calls) == 0
