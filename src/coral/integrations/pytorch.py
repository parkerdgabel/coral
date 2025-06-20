"""PyTorch integration for Coral version control."""

import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

# Type checking imports
if TYPE_CHECKING:
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        from torch.optim import Optimizer  # type: ignore
        from torch.optim.lr_scheduler import _LRScheduler  # type: ignore
    except ImportError:
        # Type stub placeholders
        torch = None  # type: ignore
        nn = None  # type: ignore
        Optimizer = None  # type: ignore
        _LRScheduler = None  # type: ignore

# Runtime availability check
try:
    import torch
    import torch.nn as nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.training.checkpoint_manager import CheckpointConfig, CheckpointManager
from coral.training.training_state import TrainingState
from coral.version_control.repository import Repository

logger = logging.getLogger(__name__)


class PyTorchIntegration:
    """Integration utilities for PyTorch models."""

    @staticmethod
    def model_to_weights(model: "nn.Module") -> Dict[str, WeightTensor]:
        """Convert PyTorch model to Coral weights."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed")

        weights = {}

        for name, param in model.named_parameters():
            # Convert to numpy array
            data = param.detach().cpu().numpy()

            # Create metadata
            metadata = WeightMetadata(
                name=name,
                shape=data.shape,
                dtype=data.dtype,
                layer_type=_get_layer_type(model, name),
                model_name=model.__class__.__name__,
            )

            # Create weight tensor
            weight = WeightTensor(data=data, metadata=metadata)
            weights[name] = weight

        return weights

    @staticmethod
    def weights_to_model(weights: Dict[str, WeightTensor], model: "nn.Module") -> None:
        """Load Coral weights into PyTorch model."""
        if not TORCH_AVAILABLE or torch is None:
            raise ImportError("PyTorch is not installed")

        state_dict = {}
        for name, weight in weights.items():
            state_dict[name] = torch.from_numpy(weight.data)

        model.load_state_dict(state_dict, strict=False)

    @staticmethod
    def save_optimizer_state(optimizer: "Optimizer") -> Dict[str, Any]:
        """Save optimizer state."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed")

        state_dict: Dict[str, Any] = optimizer.state_dict()
        return state_dict

    @staticmethod
    def load_optimizer_state(optimizer: "Optimizer", state: Dict[str, Any]) -> None:
        """Load optimizer state."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed")

        optimizer.load_state_dict(state)

    @staticmethod
    def save_scheduler_state(scheduler: "_LRScheduler") -> Dict[str, Any]:
        """Save scheduler state."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed")

        state_dict: Dict[str, Any] = scheduler.state_dict()
        return state_dict

    @staticmethod
    def load_scheduler_state(scheduler: "_LRScheduler", state: Dict[str, Any]) -> None:
        """Load scheduler state."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed")

        scheduler.load_state_dict(state)

    @staticmethod
    def get_random_state() -> Dict[str, Any]:
        """Get random state for reproducibility."""
        if not TORCH_AVAILABLE or torch is None:
            raise ImportError("PyTorch is not installed")

        return {
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        }

    @staticmethod
    def set_random_state(state: Dict[str, Any]) -> None:
        """Set random state for reproducibility."""
        if not TORCH_AVAILABLE or torch is None:
            raise ImportError("PyTorch is not installed")

        if "torch" in state and state["torch"] is not None:
            torch_state = state["torch"]
            # Ensure it's a ByteTensor
            if not isinstance(torch_state, torch.Tensor):
                torch_state = torch.tensor(torch_state, dtype=torch.uint8)
            elif torch_state.dtype != torch.uint8:
                torch_state = torch_state.to(dtype=torch.uint8)
            torch.set_rng_state(torch_state)

        if (
            "torch_cuda" in state
            and state["torch_cuda"] is not None
            and torch.cuda.is_available()
        ):
            cuda_states = state["torch_cuda"]
            if isinstance(cuda_states, list):
                # Convert each state to ByteTensor if needed
                converted_states = []
                for cuda_state in cuda_states:
                    if not isinstance(cuda_state, torch.Tensor):
                        cuda_state = torch.tensor(cuda_state, dtype=torch.uint8)
                    elif cuda_state.dtype != torch.uint8:
                        cuda_state = cuda_state.to(dtype=torch.uint8)
                    converted_states.append(cuda_state)
                torch.cuda.set_rng_state_all(converted_states)


class CoralTrainer:
    """PyTorch trainer with Coral version control integration."""

    def __init__(
        self,
        model: "nn.Module",
        repository: Repository,
        experiment_name: str,
        checkpoint_config: Optional[CheckpointConfig] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed")

        self.model = model
        self.repository = repository
        self.experiment_name = experiment_name

        # Initialize checkpoint manager
        config = checkpoint_config or CheckpointConfig(
            save_every_n_epochs=1,
            save_on_best_metric="loss",
            minimize_metric=True,
            keep_last_n_checkpoints=5,
            keep_best_n_checkpoints=3,
        )

        self.checkpoint_manager = CheckpointManager(
            repository=repository,
            config=config,
            model_name=model.__class__.__name__,
            experiment_name=experiment_name,
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.training_metrics: Dict[str, float] = {}

        # Optional components
        self.optimizer: Optional["Optimizer"] = None
        self.scheduler: Optional["_LRScheduler"] = None

        # Callbacks
        self.on_epoch_end_callbacks: List[Callable[..., Any]] = []
        self.on_step_end_callbacks: List[Callable[..., Any]] = []
        self.on_checkpoint_save_callbacks: List[Callable[..., Any]] = []

    def set_optimizer(self, optimizer: "Optimizer") -> None:
        """Set the optimizer."""
        self.optimizer = optimizer

    def set_scheduler(self, scheduler: "_LRScheduler") -> None:
        """Set the learning rate scheduler."""
        self.scheduler = scheduler

    def add_callback(self, event: str, callback: Callable[..., Any]) -> None:
        """Add a callback for training events."""
        if event == "epoch_end":
            self.on_epoch_end_callbacks.append(callback)
        elif event == "step_end":
            self.on_step_end_callbacks.append(callback)
        elif event == "checkpoint_save":
            self.on_checkpoint_save_callbacks.append(callback)
        else:
            raise ValueError(f"Unknown event: {event}")

    def update_metrics(self, **metrics: float) -> None:
        """Update training metrics."""
        self.training_metrics.update(metrics)

    def step(self, loss: float, **metrics: float) -> None:
        """Record a training step."""
        self.global_step += 1
        self.training_metrics["loss"] = loss
        self.training_metrics.update(metrics)

        # Call step callbacks
        for callback in self.on_step_end_callbacks:
            callback(self)

        # Check if we should save a checkpoint
        if self._should_save_checkpoint():
            self.save_checkpoint()

    def epoch_end(self, epoch: int) -> None:
        """Record end of epoch."""
        self.current_epoch = epoch

        # Call epoch callbacks
        for callback in self.on_epoch_end_callbacks:
            callback(self)

        # Check if we should save a checkpoint
        if self._should_save_checkpoint():
            self.save_checkpoint()

        # Update scheduler
        if self.scheduler:
            self.scheduler.step()

    def save_checkpoint(self, force: bool = False) -> Optional[str]:
        """Save a checkpoint."""
        # Create training state
        state = TrainingState(
            epoch=self.current_epoch,
            global_step=self.global_step,
            learning_rate=self._get_learning_rate(),
            loss=self.training_metrics.get("loss", 0.0),
            metrics=self.training_metrics.copy(),
            optimizer_state=PyTorchIntegration.save_optimizer_state(self.optimizer)
            if self.optimizer
            else None,
            scheduler_state=PyTorchIntegration.save_scheduler_state(self.scheduler)
            if self.scheduler
            else None,
            random_state=PyTorchIntegration.get_random_state(),
            model_name=self.model.__class__.__name__,
            experiment_name=self.experiment_name,
        )

        # Convert model to weights
        weights = PyTorchIntegration.model_to_weights(self.model)

        # Save checkpoint
        commit_hash = self.checkpoint_manager.save_checkpoint(
            weights, state, force=force
        )

        if commit_hash:
            logger.info(f"Saved checkpoint: {commit_hash}")

            # Call checkpoint callbacks
            for callback in self.on_checkpoint_save_callbacks:
                callback(self, commit_hash)

        return commit_hash

    def load_checkpoint(
        self,
        commit_hash: Optional[str] = None,
        load_best: bool = False,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
    ) -> bool:
        """Load a checkpoint."""
        checkpoint_data = self.checkpoint_manager.load_checkpoint(
            commit_hash=commit_hash, load_best=load_best
        )

        if checkpoint_data is None:
            logger.warning("No checkpoint found")
            return False

        weights = checkpoint_data["weights"]
        state = checkpoint_data["state"]

        # Load weights into model
        PyTorchIntegration.weights_to_model(weights, self.model)

        if state:
            # Restore training state
            self.current_epoch = state.epoch
            self.global_step = state.global_step
            self.training_metrics = state.metrics.copy()

            # Load optimizer state
            if load_optimizer and self.optimizer and state.optimizer_state:
                PyTorchIntegration.load_optimizer_state(
                    self.optimizer, state.optimizer_state
                )

            # Load scheduler state
            if load_scheduler and self.scheduler and state.scheduler_state:
                PyTorchIntegration.load_scheduler_state(
                    self.scheduler, state.scheduler_state
                )

            # Restore random state
            if state.random_state:
                PyTorchIntegration.set_random_state(state.random_state)

        logger.info(f"Loaded checkpoint: {checkpoint_data['commit_hash']}")
        return True

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints."""
        return self.checkpoint_manager.list_checkpoints()

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        return {
            "experiment_name": self.experiment_name,
            "model_name": self.model.__class__.__name__,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "metrics": self.training_metrics.copy(),
            "learning_rate": self._get_learning_rate(),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "num_trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
        }

    def _should_save_checkpoint(self) -> bool:
        """Check if a checkpoint should be saved."""
        # Create temporary state for checking
        state = TrainingState(
            epoch=self.current_epoch,
            global_step=self.global_step,
            learning_rate=self._get_learning_rate(),
            loss=self.training_metrics.get("loss", 0.0),
            metrics=self.training_metrics.copy(),
        )

        return self.checkpoint_manager.should_save_checkpoint(state)

    def _get_learning_rate(self) -> float:
        """Get current learning rate."""
        if self.optimizer and hasattr(self.optimizer, "param_groups"):
            return float(self.optimizer.param_groups[0]["lr"])
        return 0.0


def _get_layer_type(model: "nn.Module", param_name: str) -> Optional[str]:
    """Get the layer type for a parameter."""
    # Parse parameter name to find the layer
    parts = param_name.split(".")

    current = model
    for part in parts[:-1]:  # Exclude the parameter name (weight/bias)
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            return None

    return current.__class__.__name__ if current else None


# Utility functions for common PyTorch workflows
def save_model_to_coral(
    model: "nn.Module",
    repository: Repository,
    message: str,
    model_name: Optional[str] = None,
    **metadata: Any,
) -> str:
    """Save a PyTorch model to Coral repository."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    # Convert model to weights
    weights = PyTorchIntegration.model_to_weights(model)

    # Update metadata
    for weight in weights.values():
        if model_name:
            weight.metadata.model_name = model_name
        for key, value in metadata.items():
            setattr(weight.metadata, key, value)

    # Stage and commit
    repository.stage_weights(weights)
    commit = repository.commit(message=message)

    return commit.commit_hash


def load_model_from_coral(
    model: "nn.Module", repository: Repository, commit_ref: Optional[str] = None
) -> bool:
    """Load a PyTorch model from Coral repository."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    # Load weights
    weights = repository.get_all_weights(commit_ref)

    if not weights:
        return False

    # Load into model
    PyTorchIntegration.weights_to_model(weights, model)

    return True
