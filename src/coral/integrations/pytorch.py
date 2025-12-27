"""PyTorch integration for Coral version control."""

import logging
from typing import Any, Callable, Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    # Create dummy classes for type hints
    class nn:
        class Module:
            pass

    class Optimizer:
        pass

    class _LRScheduler:
        pass


from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.training.checkpoint_manager import CheckpointConfig, CheckpointManager
from coral.training.training_state import TrainingState
from coral.version_control.repository import Repository

logger = logging.getLogger(__name__)


class PyTorchIntegration:
    """Integration utilities for PyTorch models."""

    @staticmethod
    def model_to_weights(model: nn.Module) -> Dict[str, WeightTensor]:
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
    def weights_to_model(weights: Dict[str, WeightTensor], model: nn.Module) -> None:
        """Load Coral weights into PyTorch model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed")

        state_dict = {}
        for name, weight in weights.items():
            state_dict[name] = torch.from_numpy(weight.data)

        model.load_state_dict(state_dict, strict=False)

    @staticmethod
    def save_optimizer_state(optimizer: Optimizer) -> Dict[str, Any]:
        """Save optimizer state."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed")

        return optimizer.state_dict()

    @staticmethod
    def load_optimizer_state(optimizer: Optimizer, state: Dict[str, Any]) -> None:
        """Load optimizer state."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed")

        optimizer.load_state_dict(state)

    @staticmethod
    def save_scheduler_state(scheduler: _LRScheduler) -> Dict[str, Any]:
        """Save scheduler state."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed")

        return scheduler.state_dict()

    @staticmethod
    def load_scheduler_state(scheduler: _LRScheduler, state: Dict[str, Any]) -> None:
        """Load scheduler state."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed")

        scheduler.load_state_dict(state)

    @staticmethod
    def get_random_state() -> Dict[str, Any]:
        """Get random state for reproducibility."""
        if not TORCH_AVAILABLE:
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
        if not TORCH_AVAILABLE:
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
        model: nn.Module,
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
        self.training_metrics = {}

        # Optional components
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[_LRScheduler] = None

        # Callbacks
        self.on_epoch_end_callbacks: List[Callable] = []
        self.on_step_end_callbacks: List[Callable] = []
        self.on_checkpoint_save_callbacks: List[Callable] = []

    def set_optimizer(self, optimizer: Optimizer) -> None:
        """Set the optimizer."""
        self.optimizer = optimizer

    def set_scheduler(self, scheduler: _LRScheduler) -> None:
        """Set the learning rate scheduler."""
        self.scheduler = scheduler

    def add_callback(self, event: str, callback: Callable) -> None:
        """Add a callback for training events."""
        if event == "epoch_end":
            self.on_epoch_end_callbacks.append(callback)
        elif event == "step_end":
            self.on_step_end_callbacks.append(callback)
        elif event == "checkpoint_save":
            self.on_checkpoint_save_callbacks.append(callback)
        else:
            raise ValueError(f"Unknown event: {event}")

    def update_metrics(self, **metrics) -> None:
        """Update training metrics."""
        self.training_metrics.update(metrics)

    def step(self, loss: float, **metrics) -> None:
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
        if self.optimizer:
            return self.optimizer.param_groups[0]["lr"]
        return 0.0


def _get_layer_type(model: nn.Module, param_name: str) -> Optional[str]:
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
    model: nn.Module,
    repository: Repository,
    message: str,
    model_name: Optional[str] = None,
    **metadata,
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
    model: nn.Module, repository: Repository, commit_ref: Optional[str] = None
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


# ==================== Direct Model Loading API ====================


def load_into_model(
    model: nn.Module,
    weights: Dict[str, WeightTensor],
    strict: bool = True,
    key_map: Optional[Dict[str, str]] = None,
    prefix: str = "",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load Coral weights directly into a PyTorch model.

    This is the main entry point for loading weights into models. It provides
    flexible options for handling weight name mismatches and device placement.

    Args:
        model: PyTorch model to load weights into
        weights: Dictionary of Coral WeightTensor objects
        strict: If True, raise error for missing/unexpected keys
        key_map: Optional mapping from weight names to model parameter names
        prefix: Prefix to add to weight names when matching model keys
        device: Device to place weights on ('cpu', 'cuda', 'cuda:0', etc.)

    Returns:
        Dictionary with load statistics:
        - loaded: List of loaded weight names
        - missing: List of missing model parameters
        - unexpected: List of weights not in model
        - matched: Number of successfully matched weights

    Raises:
        RuntimeError: If strict=True and there are missing/unexpected keys

    Example:
        >>> from coral.integrations.pytorch import load_into_model
        >>> weights = repo.get_all_weights()
        >>> stats = load_into_model(model, weights, device='cuda')
        >>> print(f"Loaded {stats['matched']} weights")
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    # Get model's expected keys
    model_keys = set(model.state_dict().keys())

    # Build state dict from weights
    state_dict = {}
    loaded = []
    unexpected = []

    for name, weight in weights.items():
        # Apply key mapping if provided
        if key_map and name in key_map:
            target_key = key_map[name]
        else:
            target_key = prefix + name if prefix else name

        # Convert to tensor
        tensor = torch.from_numpy(weight.data.copy())

        # Move to device if specified
        if device:
            tensor = tensor.to(device)

        if target_key in model_keys:
            state_dict[target_key] = tensor
            loaded.append(name)
        else:
            unexpected.append(name)

    # Find missing keys
    missing = list(model_keys - set(state_dict.keys()))

    # Check strict mode
    if strict and (missing or unexpected):
        error_msg = []
        if missing:
            error_msg.append(f"Missing keys: {missing[:5]}...")
        if unexpected:
            error_msg.append(f"Unexpected keys: {unexpected[:5]}...")
        raise RuntimeError(
            f"Weight loading failed in strict mode. {' '.join(error_msg)}"
        )

    # Load into model
    model.load_state_dict(state_dict, strict=False)

    return {
        "loaded": loaded,
        "missing": missing,
        "unexpected": unexpected,
        "matched": len(loaded),
    }


def load_from_repo(
    model: nn.Module,
    repo_path: str,
    commit_ref: Optional[str] = None,
    branch: Optional[str] = None,
    tag: Optional[str] = None,
    strict: bool = True,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load weights from a Coral repository into a PyTorch model.

    Args:
        model: PyTorch model to load weights into
        repo_path: Path to Coral repository
        commit_ref: Specific commit hash to load from
        branch: Branch name to load from (alternative to commit_ref)
        tag: Tag/version name to load from (alternative to commit_ref)
        strict: If True, raise error for missing/unexpected keys
        device: Device to place weights on

    Returns:
        Dictionary with load statistics

    Example:
        >>> from coral.integrations.pytorch import load_from_repo
        >>> stats = load_from_repo(model, "./my-model", branch="main")
    """
    from pathlib import Path

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    repo = Repository(Path(repo_path))

    # Determine commit reference
    if commit_ref is None:
        if branch:
            commit_ref = repo.branch_manager.get_branch_commit(branch)
        elif tag:
            # Find version by name
            for v in repo.version_graph.versions.values():
                if v.name == tag:
                    commit_ref = v.commit_hash
                    break
            if commit_ref is None:
                raise ValueError(f"Tag '{tag}' not found")

    # Load weights
    weights = repo.get_all_weights(commit_ref)

    if not weights:
        raise ValueError("No weights found in repository")

    return load_into_model(model, weights, strict=strict, device=device)


def load_from_remote(
    model: nn.Module,
    repo_path: str,
    remote_name: str = "origin",
    commit_ref: Optional[str] = None,
    strict: bool = True,
    device: Optional[str] = None,
    pull_first: bool = True,
) -> Dict[str, Any]:
    """
    Load weights from a remote Coral repository into a PyTorch model.

    This function first pulls weights from the remote, then loads them
    into the model.

    Args:
        model: PyTorch model to load weights into
        repo_path: Path to local Coral repository
        remote_name: Name of remote to pull from
        commit_ref: Specific commit hash to load from
        strict: If True, raise error for missing/unexpected keys
        device: Device to place weights on
        pull_first: If True, pull from remote before loading

    Returns:
        Dictionary with load statistics and pull info

    Example:
        >>> stats = load_from_remote(model, "./my-model", remote_name="origin")
    """
    from pathlib import Path

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    repo = Repository(Path(repo_path))

    # Pull from remote if requested
    pull_stats = None
    if pull_first:
        pull_stats = repo.pull(remote_name)
        logger.info(
            f"Pulled {pull_stats.get('weights_pulled', 0)} weights from {remote_name}"
        )

    # Load weights
    weights = repo.get_all_weights(commit_ref)

    if not weights:
        raise ValueError("No weights found after pulling from remote")

    load_stats = load_into_model(model, weights, strict=strict, device=device)

    if pull_stats:
        load_stats["pull_stats"] = pull_stats

    return load_stats


def save_model(
    model: nn.Module,
    repo_path: str,
    message: str,
    branch: Optional[str] = None,
    create_branch: bool = False,
    push_to: Optional[str] = None,
    **metadata,
) -> Dict[str, Any]:
    """
    Save a PyTorch model to a Coral repository.

    This is a convenience function for saving models with optional
    branching and remote push.

    Args:
        model: PyTorch model to save
        repo_path: Path to Coral repository
        message: Commit message
        branch: Branch to save to (creates if doesn't exist and create_branch=True)
        create_branch: If True, create branch if it doesn't exist
        push_to: Remote name to push to after commit
        **metadata: Additional metadata to attach to weights

    Returns:
        Dictionary with save info:
        - commit_hash: Hash of the commit
        - weights_saved: Number of weights saved
        - push_stats: Push statistics (if push_to was specified)

    Example:
        >>> save_model(model, "./my-model", "Add fine-tuned weights",
        ...            branch="experiment-1", push_to="origin")
    """
    from pathlib import Path

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    repo = Repository(Path(repo_path))

    # Handle branching
    if branch:
        current_branch = repo.branch_manager.get_current_branch()
        if branch != current_branch:
            branches = [b.name for b in repo.branch_manager.list_branches()]
            if branch not in branches:
                if create_branch:
                    repo.create_branch(branch)
                else:
                    raise ValueError(
                        f"Branch '{branch}' doesn't exist. "
                        "Set create_branch=True to create it."
                    )
            repo.checkout(branch)

    # Convert model to weights
    weights = PyTorchIntegration.model_to_weights(model)

    # Add metadata
    model_name = metadata.pop("model_name", model.__class__.__name__)
    for weight in weights.values():
        weight.metadata.model_name = model_name
        for key, value in metadata.items():
            if hasattr(weight.metadata, key):
                setattr(weight.metadata, key, value)

    # Stage and commit
    repo.stage_weights(weights)
    commit = repo.commit(message=message)

    result = {
        "commit_hash": commit.commit_hash,
        "weights_saved": len(weights),
        "branch": repo.branch_manager.get_current_branch(),
    }

    # Push if requested
    if push_to:
        push_stats = repo.push(push_to)
        result["push_stats"] = push_stats
        logger.info(f"Pushed to {push_to}")

    return result


def compare_model_weights(
    model: nn.Module,
    weights: Dict[str, WeightTensor],
) -> Dict[str, Any]:
    """
    Compare a PyTorch model's current weights with Coral weights.

    Useful for checking if a model has been modified since loading.

    Args:
        model: PyTorch model to compare
        weights: Coral weights to compare against

    Returns:
        Dictionary with comparison results:
        - identical: List of identical weight names
        - modified: List of modified weight names
        - model_only: List of weights only in model
        - coral_only: List of weights only in Coral
        - similarity: Dict of weight name -> similarity score
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    import numpy as np

    from coral.utils.similarity import cosine_similarity

    model_weights = PyTorchIntegration.model_to_weights(model)

    identical = []
    modified = []
    model_only = []
    coral_only = []
    similarity = {}

    # Check all model weights
    for name, model_weight in model_weights.items():
        if name in weights:
            coral_weight = weights[name]
            if model_weight.shape == coral_weight.shape:
                # Compare values
                if np.allclose(model_weight.data, coral_weight.data, atol=1e-6):
                    identical.append(name)
                else:
                    modified.append(name)
                    sim = cosine_similarity(model_weight.data, coral_weight.data)
                    similarity[name] = float(sim)
            else:
                modified.append(name)
                similarity[name] = 0.0
        else:
            model_only.append(name)

    # Check for weights only in Coral
    for name in weights:
        if name not in model_weights:
            coral_only.append(name)

    return {
        "identical": identical,
        "modified": modified,
        "model_only": model_only,
        "coral_only": coral_only,
        "similarity": similarity,
        "is_identical": len(modified) == 0 and len(model_only) == 0,
    }


def create_model_from_weights(
    model_class: type,
    weights: Dict[str, WeightTensor],
    device: Optional[str] = None,
    **model_kwargs,
) -> nn.Module:
    """
    Create a new model instance and load weights into it.

    Args:
        model_class: PyTorch model class to instantiate
        weights: Coral weights to load
        device: Device to place model on
        **model_kwargs: Arguments to pass to model constructor

    Returns:
        Model instance with loaded weights

    Example:
        >>> from my_models import MyModel
        >>> weights = repo.get_all_weights()
        >>> model = create_model_from_weights(MyModel, weights, device='cuda')
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    # Create model instance
    model = model_class(**model_kwargs)

    # Move to device first if specified
    if device:
        model = model.to(device)

    # Load weights
    load_into_model(model, weights, strict=False, device=device)

    return model
