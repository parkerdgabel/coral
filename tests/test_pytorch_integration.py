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
