"""PyTorch Lightning integration for Coral.

This module provides a callback for PyTorch Lightning that automatically
saves model checkpoints to a Coral repository, with optional integration
to experiment tracking systems (MLflow, Weights & Biases, etc.).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from coral.integrations.experiment_bridge import ExperimentBridge
    from coral.version_control.repository import Repository

logger = logging.getLogger(__name__)

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback

    HAS_LIGHTNING = True
except ImportError:
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import Callback

        HAS_LIGHTNING = True
    except ImportError:
        HAS_LIGHTNING = False
        pl = None  # type: ignore
        Callback = object  # type: ignore


class CoralCallback(Callback):
    """PyTorch Lightning callback for automatic Coral checkpointing.

    This callback automatically saves model weights to a Coral repository
    at specified intervals or when metrics improve. Optionally integrates
    with experiment tracking systems via ExperimentBridge.

    Example:
        >>> from coral.integrations.lightning import CoralCallback
        >>> callback = CoralCallback(
        ...     repo_path="./weights",
        ...     save_every_n_epochs=1,
        ...     save_on_best="val_loss",
        ... )
        >>> trainer = pl.Trainer(callbacks=[callback])
        >>> trainer.fit(model)

    Example with MLflow tracking:
        >>> from coral.integrations.mlflow_bridge import MLflowBridge
        >>> from coral import Repository
        >>> repo = Repository("./weights", init=True)
        >>> bridge = MLflowBridge(repo, experiment_name="training")
        >>> callback = CoralCallback(
        ...     repo_path="./weights",
        ...     experiment_bridge=bridge,
        ... )
        >>> trainer = pl.Trainer(callbacks=[callback])
        >>> trainer.fit(model)

    Args:
        repo: Coral repository (as path or Repository object). Preferred over repo_path.
        repo_path: Path to Coral repository (DEPRECATED: use repo instead)
        init: If True, initialize the repository if it doesn't exist
        save_every_n_epochs: Save checkpoint every N epochs (0 to disable)
        save_every_n_steps: Save checkpoint every N steps (0 to disable)
        save_on_best: Metric name to monitor for best model saving (None to disable)
        mode: 'min' or 'max' for metric monitoring
        branch: Branch to commit to (None for current branch)
        push_to: Remote name to push to after each save (None to disable)
        include_optimizer: Whether to include optimizer state
        metadata_keys: List of trainer attributes to include in commit metadata
        experiment_bridge: Optional bridge to external experiment tracker
    """

    def __init__(
        self,
        repo: Union[str, Path, Repository, None] = None,
        repo_path: Union[str, Path, None] = None,
        init: bool = True,
        save_every_n_epochs: int = 0,
        save_every_n_steps: int = 0,
        save_on_best: Optional[str] = None,
        mode: str = "min",
        branch: Optional[str] = None,
        push_to: Optional[str] = None,
        include_optimizer: bool = False,
        metadata_keys: Optional[list] = None,
        experiment_bridge: Optional[ExperimentBridge] = None,
    ):
        if not HAS_LIGHTNING:
            raise ImportError(
                "PyTorch Lightning is required for CoralCallback. "
                "Install with: pip install pytorch-lightning"
            )

        super().__init__()

        # Handle both repo and repo_path for backwards compatibility
        if repo is not None:
            if isinstance(repo, (str, Path)):
                self.repo_path = Path(repo)
                self._repo = None
            else:
                # It's a Repository object
                self._repo = repo
                self.repo_path = (
                    repo.coral_dir.parent if hasattr(repo, "coral_dir") else None
                )
        elif repo_path is not None:
            warnings.warn(
                "repo_path is deprecated, use repo instead",
                DeprecationWarning,
                stacklevel=2,
            )
            self.repo_path = Path(repo_path)
            self._repo = None
        else:
            raise ValueError("Either repo or repo_path must be provided")

        self.init = init
        self.save_every_n_epochs = save_every_n_epochs
        self.save_every_n_steps = save_every_n_steps
        self.save_on_best = save_on_best
        self.mode = mode
        self.branch = branch
        self.push_to = push_to
        self.include_optimizer = include_optimizer
        self.metadata_keys = metadata_keys or ["current_epoch", "global_step"]

        # Best metric tracking
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = -1

        # Optional experiment tracking
        self.experiment_bridge = experiment_bridge

    @property
    def repo(self):
        """Lazily initialize the repository."""
        if self._repo is None:
            from coral.version_control.repository import Repository

            self._repo = Repository(self.repo_path, init=self.init)

        # Switch to specified branch if provided (both lazy init and passed repo)
        if self.branch and hasattr(self._repo, "current_branch"):
            current_branch = self._repo.current_branch()
            if current_branch != self.branch:
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

    def _extract_metadata(self, trainer: pl.Trainer) -> dict[str, Any]:
        """Extract metadata from trainer."""
        metadata = {}
        for key in self.metadata_keys:
            if hasattr(trainer, key):
                metadata[key] = getattr(trainer, key)
        return metadata

    def _save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        message: str,
        metrics: Optional[dict[str, float]] = None,
    ) -> None:
        """Save model weights to Coral repository."""
        from coral.core.weight_tensor import WeightMetadata, WeightTensor

        # Get model state dict
        state_dict = pl_module.state_dict()

        # Optionally include optimizer state
        if self.include_optimizer and trainer.optimizers:
            opt = trainer.optimizers[0]
            state_dict["optimizer_state"] = opt.state_dict()

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

        # Add metrics to commit
        metadata = self._extract_metadata(trainer)
        if metrics:
            metadata.update(metrics)

        tags = []
        if metrics and self.save_on_best:
            tags.append(f"best_{self.save_on_best}")

        commit = self.repo.commit(
            message=message,
            tags=tags if tags else None,
        )

        logger.info(f"Coral: Saved checkpoint [{commit.commit_hash[:8]}] {message}")

        # Log to experiment tracker if configured
        if self.experiment_bridge and self.experiment_bridge.is_run_active:
            self.experiment_bridge.log_coral_commit(commit.commit_hash, message)
            if metrics:
                self.experiment_bridge.log_metrics(metrics)

        # Push if configured
        if self.push_to:
            try:
                result = self.repo.push(self.push_to)
                pushed = result["weights_pushed"]
                logger.info(f"Coral: Pushed {pushed} weights to {self.push_to}")
            except Exception as e:
                logger.warning(f"Coral: Failed to push to {self.push_to}: {e}")

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called at the start of training."""
        # Start experiment run if bridge is configured
        if self.experiment_bridge and not self.experiment_bridge.is_run_active:
            # Extract hyperparameters from trainer if available
            params = {}
            if hasattr(pl_module, "hparams"):
                params = dict(pl_module.hparams)
            params["model_class"] = pl_module.__class__.__name__

            self.experiment_bridge.start_run(
                name=pl_module.__class__.__name__,
                params=params,
            )

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called at the end of each training epoch."""
        epoch = trainer.current_epoch

        # Log epoch metrics to experiment tracker
        if self.experiment_bridge and self.experiment_bridge.is_run_active:
            epoch_metrics = {}
            for k, v in trainer.callback_metrics.items():
                if isinstance(v, (int, float)):
                    epoch_metrics[k] = float(v)
                elif hasattr(v, "item") and v.numel() == 1:
                    epoch_metrics[k] = v.item()
            if epoch_metrics:
                self.experiment_bridge.log_metrics(
                    epoch_metrics, step=trainer.global_step
                )

        # Save at epoch interval
        if self.save_every_n_epochs > 0:
            if (epoch + 1) % self.save_every_n_epochs == 0:
                self._save_checkpoint(
                    trainer,
                    pl_module,
                    message=f"Epoch {epoch + 1} checkpoint",
                )

        # Check for best metric
        if self.save_on_best and trainer.callback_metrics:
            current = trainer.callback_metrics.get(self.save_on_best)
            if current is not None:
                current_val = float(current)
                if self._is_better(current_val):
                    self.best_metric = current_val
                    self.best_epoch = epoch
                    self._save_checkpoint(
                        trainer,
                        pl_module,
                        message=f"Best {self.save_on_best}: {current_val:.6f}",
                        metrics={self.save_on_best: current_val},
                    )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called at the end of each training batch."""
        if self.save_every_n_steps > 0:
            step = trainer.global_step
            if step > 0 and step % self.save_every_n_steps == 0:
                self._save_checkpoint(
                    trainer,
                    pl_module,
                    message=f"Step {step} checkpoint",
                )

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called at the end of training."""
        self._save_checkpoint(
            trainer,
            pl_module,
            message=f"Training complete (epoch {trainer.current_epoch + 1})",
        )

        if self.save_on_best:
            metric = self.save_on_best
            best = self.best_metric
            epoch = self.best_epoch + 1
            logger.info(f"Coral: Best {metric}: {best:.6f} at epoch {epoch}")

        # End experiment run if bridge is configured
        if self.experiment_bridge and self.experiment_bridge.is_run_active:
            self.experiment_bridge.end_run("completed")

    def load_from_coral(
        self,
        pl_module: pl.LightningModule,
        commit_ref: Optional[str] = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load weights from Coral repository into a Lightning module.

        Args:
            pl_module: The Lightning module to load weights into
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
            if name == "optimizer_state":
                continue  # Skip optimizer state for model loading
            state_dict[name] = torch.from_numpy(weight.data)

        # Load into model
        result = pl_module.load_state_dict(state_dict, strict=strict)

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


# Alias for more descriptive name
CoralLightningCallback = CoralCallback
