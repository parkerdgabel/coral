"""HuggingFace Transformers Trainer integration for Coral.

This module provides a callback for the HuggingFace Trainer that automatically
saves model checkpoints to a Coral repository.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState
    from transformers.training_args import TrainingArguments

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    TrainerCallback = object  # type: ignore
    TrainerControl = None  # type: ignore
    TrainerState = None  # type: ignore
    TrainingArguments = None  # type: ignore


class CoralTrainerCallback(TrainerCallback):
    """HuggingFace Trainer callback for automatic Coral checkpointing.

    This callback automatically saves model weights to a Coral repository
    at specified intervals or when metrics improve.

    Example:
        >>> from coral.integrations.hf_trainer import CoralTrainerCallback
        >>> from transformers import Trainer, TrainingArguments
        >>>
        >>> callback = CoralTrainerCallback(
        ...     repo_path="./weights",
        ...     save_every_n_steps=500,
        ...     save_on_best="eval_loss",
        ... )
        >>> trainer = Trainer(
        ...     model=model,
        ...     args=training_args,
        ...     callbacks=[callback],
        ... )
        >>> trainer.train()

    Args:
        repo_path: Path to Coral repository (will be created if init=True)
        init: If True, initialize the repository if it doesn't exist
        save_every_n_epochs: Save checkpoint every N epochs (0 to disable)
        save_every_n_steps: Save checkpoint every N steps (0 to disable)
        save_on_best: Metric name to monitor for best model saving (None to disable)
        mode: 'min' or 'max' for metric monitoring
        branch: Branch to commit to (None for current branch)
        push_to: Remote name to push to after each save (None to disable)
        save_on_train_end: Whether to save at end of training
    """

    def __init__(
        self,
        repo_path: Union[str, Path],
        init: bool = True,
        save_every_n_epochs: int = 0,
        save_every_n_steps: int = 0,
        save_on_best: Optional[str] = None,
        mode: str = "min",
        branch: Optional[str] = None,
        push_to: Optional[str] = None,
        save_on_train_end: bool = True,
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "HuggingFace Transformers is required for CoralTrainerCallback. "
                "Install with: pip install transformers"
            )

        super().__init__()

        self.repo_path = Path(repo_path)
        self.init = init
        self.save_every_n_epochs = save_every_n_epochs
        self.save_every_n_steps = save_every_n_steps
        self.save_on_best = save_on_best
        self.mode = mode
        self.branch = branch
        self.push_to = push_to
        self.save_on_train_end = save_on_train_end

        # Best metric tracking
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.best_step = -1

        # Track last saved step to avoid duplicates
        self._last_saved_step = -1
        self._last_saved_epoch = -1

        # Repository will be initialized lazily
        self._repo = None

    @property
    def repo(self):
        """Lazily initialize the repository."""
        if self._repo is None:
            from coral.version_control.repository import Repository

            self._repo = Repository(self.repo_path, init=self.init)

            # Switch to specified branch if provided
            if self.branch:
                try:
                    self._repo.create_branch(self.branch)
                except ValueError:
                    pass  # Branch already exists
                self._repo.checkout(self.branch)

        return self._repo

    def _is_better(self, current: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == "min":
            return current < self.best_metric
        return current > self.best_metric

    def _save_checkpoint(
        self,
        model: Any,
        message: str,
        metrics: Optional[dict[str, float]] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        """Save model weights to Coral repository."""
        from coral.core.weight_tensor import WeightMetadata, WeightTensor

        # Get model state dict
        state_dict = model.state_dict()

        # Convert to WeightTensors
        weights = {}
        for name, param in state_dict.items():
            if hasattr(param, "cpu"):
                data = param.cpu().numpy()
            else:
                data = param
            weights[name] = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=name,
                    shape=data.shape,
                    dtype=data.dtype,
                ),
            )

        # Stage and commit
        self.repo.stage_weights(weights)

        tags = []
        if metrics and self.save_on_best:
            tags.append(f"best_{self.save_on_best}")
        if step is not None:
            tags.append(f"step_{step}")
        if epoch is not None:
            tags.append(f"epoch_{epoch}")

        commit = self.repo.commit(
            message=message,
            tags=tags if tags else None,
        )

        logger.info(f"Coral: Saved checkpoint [{commit.commit_hash[:8]}] {message}")

        # Push if configured
        if self.push_to:
            try:
                result = self.repo.push(self.push_to)
                pushed = result["weights_pushed"]
                logger.info(f"Coral: Pushed {pushed} weights to {self.push_to}")
            except Exception as e:
                logger.warning(f"Coral: Failed to push to {self.push_to}: {e}")

    def on_step_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        model: Any = None,
        **kwargs,
    ):
        """Called at the end of each training step."""
        if self.save_every_n_steps > 0 and model is not None:
            step = state.global_step
            if step > 0 and step % self.save_every_n_steps == 0:
                if step != self._last_saved_step:
                    self._save_checkpoint(
                        model,
                        message=f"Step {step} checkpoint",
                        step=step,
                    )
                    self._last_saved_step = step

        return control

    def on_epoch_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        model: Any = None,
        **kwargs,
    ):
        """Called at the end of each epoch."""
        if self.save_every_n_epochs > 0 and model is not None:
            epoch = int(state.epoch)
            if epoch > 0 and epoch % self.save_every_n_epochs == 0:
                if epoch != self._last_saved_epoch:
                    self._save_checkpoint(
                        model,
                        message=f"Epoch {epoch} checkpoint",
                        epoch=epoch,
                    )
                    self._last_saved_epoch = epoch

        return control

    def on_evaluate(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        model: Any = None,
        metrics: Optional[dict[str, float]] = None,
        **kwargs,
    ):
        """Called after evaluation."""
        if self.save_on_best and metrics and model is not None:
            metric_value = metrics.get(self.save_on_best)
            if metric_value is not None:
                if self._is_better(metric_value):
                    self.best_metric = metric_value
                    self.best_step = state.global_step
                    self._save_checkpoint(
                        model,
                        message=f"Best {self.save_on_best}: {metric_value:.6f}",
                        metrics={self.save_on_best: metric_value},
                        step=state.global_step,
                    )

        return control

    def on_train_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        model: Any = None,
        **kwargs,
    ):
        """Called at the end of training."""
        if self.save_on_train_end and model is not None:
            epoch = int(state.epoch) if state.epoch else 0
            self._save_checkpoint(
                model,
                message=f"Training complete (step {state.global_step})",
                step=state.global_step,
                epoch=epoch,
            )

            if self.save_on_best:
                logger.info(
                    f"Coral: Best {self.save_on_best}: {self.best_metric:.6f} "
                    f"at step {self.best_step}"
                )

        return control

    def load_from_coral(
        self,
        model: Any,
        commit_ref: Optional[str] = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load weights from Coral repository into a HuggingFace model.

        Args:
            model: The HuggingFace model to load weights into
            commit_ref: Specific commit to load from (None for latest)
            strict: If True, require all weights to match

        Returns:
            Dictionary with loading info (matched, missing, unexpected keys)
        """
        import torch

        weights = self.repo.get_all_weights(commit_ref)

        # Convert WeightTensors back to state dict
        state_dict = {}
        for name, weight in weights.items():
            state_dict[name] = torch.from_numpy(weight.data)

        # Load into model
        result = model.load_state_dict(state_dict, strict=strict)

        return {
            "missing_keys": result.missing_keys
            if hasattr(result, "missing_keys")
            else [],
            "unexpected_keys": result.unexpected_keys
            if hasattr(result, "unexpected_keys")
            else [],
            "loaded_weights": len(state_dict),
            "commit_ref": commit_ref,
        }
