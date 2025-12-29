from unittest.mock import Mock, patch

import numpy as np

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.integrations.pytorch import CoralTrainer, PyTorchIntegration
from coral.version_control.repository import Repository

# Mock PyTorch since it's optional
torch = Mock()
torch.nn.Module = Mock
torch.Tensor = Mock
torch.save = Mock()
torch.load = Mock()


class TestPyTorchIntegration:
    def test_model_to_weights(self):
        """Test converting PyTorch model to weight tensors."""
        # Mock model
        model = Mock()
        model.named_parameters.return_value = [
            (
                "layer1.weight",
                Mock(
                    detach=Mock(
                        return_value=Mock(
                            cpu=Mock(
                                return_value=Mock(
                                    numpy=Mock(return_value=np.random.randn(10, 20))
                                )
                            )
                        )
                    ),
                    shape=(10, 20),
                    dtype=Mock(name="float32"),
                ),
            ),
            (
                "layer1.bias",
                Mock(
                    detach=Mock(
                        return_value=Mock(
                            cpu=Mock(
                                return_value=Mock(
                                    numpy=Mock(return_value=np.random.randn(10))
                                )
                            )
                        )
                    ),
                    shape=(10,),
                    dtype=Mock(name="float32"),
                ),
            ),
        ]

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                weight_tensors = PyTorchIntegration.model_to_weights(model)

        assert len(weight_tensors) == 2
        assert "layer1.weight" in weight_tensors
        assert "layer1.bias" in weight_tensors

        # Check weight tensor properties
        assert isinstance(weight_tensors["layer1.weight"], WeightTensor)
        assert weight_tensors["layer1.weight"].shape == (10, 20)

    def test_weights_to_model(self):
        """Test loading weight tensors into model."""
        # Mock model
        model = Mock()
        state_dict = {}
        model.load_state_dict = Mock(
            side_effect=lambda sd, strict: state_dict.update(sd)
        )

        # Create weight tensors
        weight_tensors = {
            "layer1.weight": WeightTensor(
                data=np.random.randn(10, 20).astype(np.float32),
                metadata={"name": "layer1.weight"},
            ),
            "layer1.bias": WeightTensor(
                data=np.random.randn(10).astype(np.float32),
                metadata={"name": "layer1.bias"},
            ),
        }

        # Mock torch.from_numpy
        torch.from_numpy = Mock(side_effect=lambda x: Mock(data=x))

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                PyTorchIntegration.weights_to_model(weight_tensors, model)

        # Should have called load_state_dict
        model.load_state_dict.assert_called_once()

    def test_weights_to_model_missing_weights(self):
        """Test loading with missing weights."""
        model = Mock()
        model.load_state_dict = Mock()

        weight_tensors = {
            "layer1.weight": WeightTensor(
                data=np.ones((5, 5), dtype=np.float32),
                metadata={"name": "layer1.weight"},
            )
        }

        torch.from_numpy = Mock(return_value=Mock())

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                # Should handle missing weights gracefully
                PyTorchIntegration.weights_to_model(weight_tensors, model)

        model.load_state_dict.assert_called_once()


class TestCheckpointing:
    def test_save_optimizer_state(self):
        """Test saving optimizer state."""
        optimizer = Mock()
        optimizer.state_dict.return_value = {"lr": 0.001, "momentum": 0.9}

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                state = PyTorchIntegration.save_optimizer_state(optimizer)

        assert state == {"lr": 0.001, "momentum": 0.9}

    def test_load_optimizer_state(self):
        """Test loading optimizer state."""
        optimizer = Mock()
        state = {"lr": 0.001, "momentum": 0.9}

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                PyTorchIntegration.load_optimizer_state(optimizer, state)

        optimizer.load_state_dict.assert_called_once_with(state)


class TestCoralTrainer:
    def setup_method(self):
        """Set up test fixtures."""
        self.model = Mock()
        self.model.__class__.__name__ = "TestModel"

        # Mock repository
        self.repo = Mock(spec=Repository)
        self.repo.current_branch = "main"

        # Mock coral_dir - needed by CheckpointManager
        import tempfile
        from pathlib import Path

        self.temp_dir = tempfile.mkdtemp()
        self.repo.coral_dir = Path(self.temp_dir) / ".coral"
        self.repo.coral_dir.mkdir(parents=True, exist_ok=True)
        (self.repo.coral_dir / "checkpoints").mkdir(exist_ok=True)

        # Create trainer with mocked torch
        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                self.trainer = CoralTrainer(
                    model=self.model,
                    repository=self.repo,
                    experiment_name="test_experiment",
                )

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if hasattr(self, "temp_dir"):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.model == self.model
        assert self.trainer.repository == self.repo
        assert self.trainer.experiment_name == "test_experiment"
        assert self.trainer.current_epoch == 0
        assert self.trainer.global_step == 0

    def test_set_optimizer(self):
        """Test setting optimizer."""
        optimizer = Mock()
        self.trainer.set_optimizer(optimizer)
        assert self.trainer.optimizer == optimizer

    def test_update_metrics(self):
        """Test updating metrics."""
        self.trainer.update_metrics(loss=0.5, accuracy=0.9)
        assert self.trainer.training_metrics["loss"] == 0.5
        assert self.trainer.training_metrics["accuracy"] == 0.9

    def test_step(self):
        """Test training step."""
        # Mock save_checkpoint to avoid model conversion
        self.trainer.save_checkpoint = Mock()

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                self.trainer.step(loss=0.3, accuracy=0.85)

        assert self.trainer.global_step == 1
        assert self.trainer.training_metrics["loss"] == 0.3
        assert self.trainer.training_metrics["accuracy"] == 0.85

    def test_epoch_end(self):
        """Test epoch end."""
        # Set up checkpoint manager mock
        self.trainer.checkpoint_manager.save_checkpoint = Mock(
            return_value="checkpoint_hash"
        )

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                # Mock model weights
                self.trainer.model.named_parameters = Mock(
                    return_value=[
                        (
                            "weight",
                            Mock(
                                detach=Mock(
                                    return_value=Mock(
                                        cpu=Mock(
                                            return_value=Mock(
                                                numpy=Mock(return_value=np.ones((3, 3)))
                                            )
                                        )
                                    )
                                )
                            ),
                        )
                    ]
                )

                self.trainer.epoch_end(epoch=1)

        assert self.trainer.current_epoch == 1

    def test_save_checkpoint(self):
        """Test saving checkpoint."""
        # Mock model weights
        self.trainer.model.named_parameters = Mock(
            return_value=[
                (
                    "weight",
                    Mock(
                        detach=Mock(
                            return_value=Mock(
                                cpu=Mock(
                                    return_value=Mock(
                                        numpy=Mock(return_value=np.ones((3, 3)))
                                    )
                                )
                            )
                        )
                    ),
                )
            ]
        )

        # Mock checkpoint manager
        self.trainer.checkpoint_manager.save_checkpoint = Mock(
            return_value="checkpoint_123"
        )

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                checkpoint_id = self.trainer.save_checkpoint(force=True)

        assert checkpoint_id == "checkpoint_123"
        self.trainer.checkpoint_manager.save_checkpoint.assert_called_once()

    def test_load_checkpoint(self):
        """Test loading checkpoint."""
        from coral.training.training_state import TrainingState

        # Mock checkpoint data using TrainingState
        state = TrainingState(
            epoch=5,
            global_step=100,
            learning_rate=0.001,
            loss=0.2,
            metrics={"loss": 0.2, "accuracy": 0.95},
            optimizer_state={"lr": 0.001},
        )

        self.trainer.checkpoint_manager.load_checkpoint = Mock(
            return_value={
                "weights": {
                    "weight": WeightTensor(
                        data=np.ones((3, 3)),
                        metadata=WeightMetadata(
                            name="weight", shape=(3, 3), dtype=np.float32
                        ),
                    )
                },
                "state": state,
                "commit_hash": "checkpoint_123",
            }
        )

        # Set optimizer
        optimizer = Mock()
        self.trainer.set_optimizer(optimizer)

        # Mock weights_to_model
        PyTorchIntegration.weights_to_model = Mock()
        PyTorchIntegration.load_optimizer_state = Mock()

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                self.trainer.load_checkpoint("checkpoint_123")

        assert self.trainer.current_epoch == 5
        assert self.trainer.global_step == 100
        assert self.trainer.training_metrics["loss"] == 0.2

    def test_add_callback(self):
        """Test adding callbacks."""
        callback = Mock()
        self.trainer.add_callback("epoch_end", callback)
        assert callback in self.trainer.on_epoch_end_callbacks

    def test_get_training_summary(self):
        """Test getting training summary."""
        self.trainer.current_epoch = 10
        self.trainer.global_step = 1000
        self.trainer.training_metrics = {"loss": 0.1, "accuracy": 0.98}

        # Mock model.parameters() to return iterable
        param1 = Mock()
        param1.numel.return_value = 100
        param1.requires_grad = True
        param2 = Mock()
        param2.numel.return_value = 50
        param2.requires_grad = False
        self.model.parameters = Mock(return_value=[param1, param2])

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                summary = self.trainer.get_training_summary()

        assert summary["current_epoch"] == 10
        assert summary["global_step"] == 1000
        assert summary["experiment_name"] == "test_experiment"
        assert "metrics" in summary
        assert summary["num_parameters"] == 150
        assert summary["num_trainable_parameters"] == 100

    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        self.trainer.checkpoint_manager.list_checkpoints = Mock(
            return_value=[
                {"checkpoint_id": "ckpt1", "epoch": 1},
                {"checkpoint_id": "ckpt2", "epoch": 2},
            ]
        )

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                checkpoints = self.trainer.list_checkpoints()

        assert len(checkpoints) == 2
        assert checkpoints[0]["checkpoint_id"] == "ckpt1"


class TestSchedulerState:
    """Test scheduler state save/load."""

    def test_save_scheduler_state(self):
        """Test saving scheduler state."""
        scheduler = Mock()
        scheduler.state_dict.return_value = {"last_epoch": 5, "step_size": 10}

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                state = PyTorchIntegration.save_scheduler_state(scheduler)

        assert state == {"last_epoch": 5, "step_size": 10}

    def test_load_scheduler_state(self):
        """Test loading scheduler state."""
        scheduler = Mock()
        state = {"last_epoch": 5, "step_size": 10}

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                PyTorchIntegration.load_scheduler_state(scheduler, state)

        scheduler.load_state_dict.assert_called_once_with(state)


class TestRandomState:
    """Test random state save/load for reproducibility."""

    def test_get_random_state(self):
        """Test getting random state."""
        mock_torch = Mock()
        mock_torch.get_rng_state.return_value = Mock()
        mock_torch.cuda.is_available.return_value = False

        with patch("coral.integrations.pytorch.torch", mock_torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                state = PyTorchIntegration.get_random_state()

        assert "torch" in state
        mock_torch.get_rng_state.assert_called_once()

    def test_get_random_state_with_cuda(self):
        """Test getting random state with CUDA available."""
        mock_torch = Mock()
        mock_torch.get_rng_state.return_value = Mock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_rng_state_all.return_value = [Mock()]

        with patch("coral.integrations.pytorch.torch", mock_torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                state = PyTorchIntegration.get_random_state()

        assert "torch_cuda" in state
        mock_torch.cuda.get_rng_state_all.assert_called_once()

    def test_set_random_state(self):
        """Test setting random state."""
        mock_torch = Mock()
        mock_tensor = Mock()
        mock_tensor.dtype = Mock()
        mock_tensor.dtype.__ne__ = Mock(return_value=False)
        mock_torch.Tensor = Mock
        mock_torch.uint8 = "uint8"
        mock_torch.cuda.is_available.return_value = False

        state = {"torch": mock_tensor, "torch_cuda": None}

        with patch("coral.integrations.pytorch.torch", mock_torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                PyTorchIntegration.set_random_state(state)

        mock_torch.set_rng_state.assert_called_once()


class TestCoralTrainerCallbacks:
    """Test callback functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        import tempfile
        from pathlib import Path

        self.model = Mock()
        self.model.__class__.__name__ = "TestModel"
        self.repo = Mock(spec=Repository)
        self.temp_dir = tempfile.mkdtemp()
        self.repo.coral_dir = Path(self.temp_dir) / ".coral"
        self.repo.coral_dir.mkdir(parents=True, exist_ok=True)
        (self.repo.coral_dir / "checkpoints").mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if hasattr(self, "temp_dir"):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_step_callback(self):
        """Test adding step end callback."""
        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                trainer = CoralTrainer(
                    model=self.model,
                    repository=self.repo,
                    experiment_name="test",
                )
                callback = Mock()
                trainer.add_callback("step_end", callback)
                assert callback in trainer.on_step_end_callbacks

    def test_add_checkpoint_callback(self):
        """Test adding checkpoint save callback."""
        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                trainer = CoralTrainer(
                    model=self.model,
                    repository=self.repo,
                    experiment_name="test",
                )
                callback = Mock()
                trainer.add_callback("checkpoint_save", callback)
                assert callback in trainer.on_checkpoint_save_callbacks

    def test_add_invalid_callback(self):
        """Test adding invalid callback type."""
        import pytest

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                trainer = CoralTrainer(
                    model=self.model,
                    repository=self.repo,
                    experiment_name="test",
                )
                with pytest.raises(ValueError, match="Unknown event"):
                    trainer.add_callback("invalid_event", Mock())


class TestCoralTrainerExperimentBridge:
    """Test experiment bridge integration."""

    def setup_method(self):
        """Set up test fixtures."""
        import tempfile
        from pathlib import Path

        self.model = Mock()
        self.model.__class__.__name__ = "TestModel"
        self.repo = Mock(spec=Repository)
        self.temp_dir = tempfile.mkdtemp()
        self.repo.coral_dir = Path(self.temp_dir) / ".coral"
        self.repo.coral_dir.mkdir(parents=True, exist_ok=True)
        (self.repo.coral_dir / "checkpoints").mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if hasattr(self, "temp_dir"):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_update_metrics_with_bridge(self):
        """Test updating metrics logs to experiment bridge."""
        mock_bridge = Mock()
        mock_bridge.is_run_active = True

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                trainer = CoralTrainer(
                    model=self.model,
                    repository=self.repo,
                    experiment_name="test",
                    experiment_bridge=mock_bridge,
                )
                trainer.update_metrics(loss=0.5, accuracy=0.9)

        mock_bridge.log_metrics.assert_called_once()

    def test_start_experiment(self):
        """Test starting experiment via bridge."""
        mock_bridge = Mock()
        mock_bridge.start_run.return_value = "run_id_123"

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                trainer = CoralTrainer(
                    model=self.model,
                    repository=self.repo,
                    experiment_name="test",
                    experiment_bridge=mock_bridge,
                )
                run_id = trainer.start_experiment(
                    params={"lr": 0.001}, tags=["test"]
                )

        assert run_id == "run_id_123"
        mock_bridge.start_run.assert_called_once()

    def test_start_experiment_no_bridge(self):
        """Test starting experiment without bridge returns None."""
        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                trainer = CoralTrainer(
                    model=self.model,
                    repository=self.repo,
                    experiment_name="test",
                )
                run_id = trainer.start_experiment()

        assert run_id is None

    def test_end_experiment(self):
        """Test ending experiment via bridge."""
        mock_bridge = Mock()
        mock_bridge.is_run_active = True

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                trainer = CoralTrainer(
                    model=self.model,
                    repository=self.repo,
                    experiment_name="test",
                    experiment_bridge=mock_bridge,
                )
                trainer.end_experiment("completed")

        mock_bridge.end_run.assert_called_once_with(status="completed")

    def test_end_experiment_no_bridge(self):
        """Test ending experiment without bridge is a no-op."""
        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                trainer = CoralTrainer(
                    model=self.model,
                    repository=self.repo,
                    experiment_name="test",
                )
                # Should not raise
                trainer.end_experiment()


class TestCoralTrainerScheduler:
    """Test scheduler integration."""

    def setup_method(self):
        """Set up test fixtures."""
        import tempfile
        from pathlib import Path

        self.model = Mock()
        self.model.__class__.__name__ = "TestModel"
        self.repo = Mock(spec=Repository)
        self.temp_dir = tempfile.mkdtemp()
        self.repo.coral_dir = Path(self.temp_dir) / ".coral"
        self.repo.coral_dir.mkdir(parents=True, exist_ok=True)
        (self.repo.coral_dir / "checkpoints").mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if hasattr(self, "temp_dir"):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_set_scheduler(self):
        """Test setting scheduler."""
        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                trainer = CoralTrainer(
                    model=self.model,
                    repository=self.repo,
                    experiment_name="test",
                )
                scheduler = Mock()
                trainer.set_scheduler(scheduler)
                assert trainer.scheduler == scheduler

    def test_epoch_end_steps_scheduler(self):
        """Test epoch end steps scheduler."""
        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                trainer = CoralTrainer(
                    model=self.model,
                    repository=self.repo,
                    experiment_name="test",
                )
                trainer.checkpoint_manager.save_checkpoint = Mock(return_value=None)
                trainer.checkpoint_manager.should_save_checkpoint = Mock(
                    return_value=False
                )

                scheduler = Mock()
                trainer.set_scheduler(scheduler)
                trainer.epoch_end(epoch=1)

        scheduler.step.assert_called_once()


class TestLoadCheckpointNoState:
    """Test loading checkpoint with no state."""

    def setup_method(self):
        """Set up test fixtures."""
        import tempfile
        from pathlib import Path

        self.model = Mock()
        self.model.__class__.__name__ = "TestModel"
        self.repo = Mock(spec=Repository)
        self.temp_dir = tempfile.mkdtemp()
        self.repo.coral_dir = Path(self.temp_dir) / ".coral"
        self.repo.coral_dir.mkdir(parents=True, exist_ok=True)
        (self.repo.coral_dir / "checkpoints").mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if hasattr(self, "temp_dir"):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_checkpoint_no_checkpoint_found(self):
        """Test loading when no checkpoint exists."""
        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                trainer = CoralTrainer(
                    model=self.model,
                    repository=self.repo,
                    experiment_name="test",
                )
                trainer.checkpoint_manager.load_checkpoint = Mock(return_value=None)

                result = trainer.load_checkpoint("nonexistent")

        assert result is False


class TestGetLearningRate:
    """Test getting learning rate."""

    def setup_method(self):
        """Set up test fixtures."""
        import tempfile
        from pathlib import Path

        self.model = Mock()
        self.model.__class__.__name__ = "TestModel"
        self.repo = Mock(spec=Repository)
        self.temp_dir = tempfile.mkdtemp()
        self.repo.coral_dir = Path(self.temp_dir) / ".coral"
        self.repo.coral_dir.mkdir(parents=True, exist_ok=True)
        (self.repo.coral_dir / "checkpoints").mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if hasattr(self, "temp_dir"):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_learning_rate_no_optimizer(self):
        """Test getting learning rate with no optimizer set."""
        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                trainer = CoralTrainer(
                    model=self.model,
                    repository=self.repo,
                    experiment_name="test",
                )
                lr = trainer._get_learning_rate()

        assert lr == 0.0

    def test_get_learning_rate_with_optimizer(self):
        """Test getting learning rate with optimizer set."""
        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                trainer = CoralTrainer(
                    model=self.model,
                    repository=self.repo,
                    experiment_name="test",
                )
                optimizer = Mock()
                optimizer.param_groups = [{"lr": 0.001}]
                trainer.set_optimizer(optimizer)
                lr = trainer._get_learning_rate()

        assert lr == 0.001


class TestGetLayerType:
    """Test _get_layer_type utility function."""

    def test_get_layer_type(self):
        """Test getting layer type from parameter name."""
        from coral.integrations.pytorch import _get_layer_type

        model = Mock()
        model.layer1 = Mock()
        model.layer1.__class__.__name__ = "Linear"

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                layer_type = _get_layer_type(model, "layer1.weight")

        assert layer_type == "Linear"

    def test_get_layer_type_nested(self):
        """Test getting layer type from nested parameter name."""
        from coral.integrations.pytorch import _get_layer_type

        model = Mock()
        model.encoder = Mock()
        model.encoder.layer1 = Mock()
        model.encoder.layer1.__class__.__name__ = "Conv2d"

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                layer_type = _get_layer_type(model, "encoder.layer1.weight")

        assert layer_type == "Conv2d"

    def test_get_layer_type_not_found(self):
        """Test getting layer type when not found."""
        from coral.integrations.pytorch import _get_layer_type

        # Create a mock that returns False for hasattr by using spec
        class EmptyModel:
            pass

        model = EmptyModel()

        with patch("coral.integrations.pytorch.torch", torch):
            with patch("coral.integrations.pytorch.TORCH_AVAILABLE", True):
                layer_type = _get_layer_type(model, "nonexistent.weight")

        assert layer_type is None
