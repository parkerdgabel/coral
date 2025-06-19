import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from coral.core.weight_tensor import WeightTensor
from coral.version_control.repository import Repository

from .training_state import TrainingState

logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management."""

    # Checkpoint frequency
    save_every_n_steps: Optional[int] = None
    save_every_n_epochs: Optional[int] = None
    save_on_best_metric: Optional[str] = None
    minimize_metric: bool = True  # True for loss, False for accuracy

    # Retention policy
    keep_last_n_checkpoints: Optional[int] = None
    keep_best_n_checkpoints: Optional[int] = None
    keep_checkpoint_every_n_epochs: Optional[int] = None

    # Storage options
    save_optimizer_state: bool = True
    save_scheduler_state: bool = True
    save_random_state: bool = True
    use_incremental_saves: bool = True

    # Commit options
    auto_commit: bool = True
    commit_message_template: str = "Checkpoint at epoch {epoch}, step {step}"
    tag_best_checkpoints: bool = True


class CheckpointManager:
    """Manages training checkpoints with Coral version control."""

    def __init__(
        self,
        repository: Repository,
        config: CheckpointConfig,
        model_name: str,
        experiment_name: Optional[str] = None,
    ):
        self.repository = repository
        self.config = config
        self.model_name = model_name
        self.experiment_name = (
            experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Tracking
        self.checkpoint_history: List[Dict[str, Any]] = []
        self.best_metric_value: Optional[float] = None
        self.best_checkpoint_hash: Optional[str] = None

        # Callback system
        self._callbacks: List[Callable[[TrainingState, Optional[str]], None]] = []

        # Load existing checkpoint history
        self._load_checkpoint_history()

    def should_save_checkpoint(self, state: TrainingState) -> bool:
        """Determine if a checkpoint should be saved."""
        # Check step-based saving
        if (
            self.config.save_every_n_steps
            and state.global_step % self.config.save_every_n_steps == 0
        ):
            return True

        # Check epoch-based saving
        if (
            self.config.save_every_n_epochs
            and state.epoch % self.config.save_every_n_epochs == 0
        ):
            return True

        # Check metric-based saving
        if (
            self.config.save_on_best_metric
            and self.config.save_on_best_metric in state.metrics
        ):
            metric_value = state.metrics[self.config.save_on_best_metric]

            if self.best_metric_value is None:
                return True

            if self.config.minimize_metric:
                return metric_value < self.best_metric_value
            else:
                return metric_value > self.best_metric_value

        return False

    def save_checkpoint(
        self,
        weights: Dict[str, WeightTensor],
        state: TrainingState,
        force: bool = False,
    ) -> Optional[str]:
        """Save a checkpoint."""
        if not force and not self.should_save_checkpoint(state):
            return None

        logger.info(
            f"Saving checkpoint at epoch {state.epoch}, step {state.global_step}"
        )

        # Update state with model and experiment info
        state.model_name = self.model_name
        state.experiment_name = self.experiment_name

        # Stage weights
        self.repository.stage_weights(weights)

        # Save training state
        state_file = self.repository.staging_dir / "training_state.json"
        state.save(str(state_file))

        # Create commit if auto-commit is enabled
        commit_hash = None
        if self.config.auto_commit:
            # Merge metrics with standard fields, avoiding duplicates
            format_args = {
                "epoch": state.epoch,
                "step": state.global_step,
                "loss": state.loss,
            }
            # Add metrics that don't conflict
            for key, value in state.metrics.items():
                if key not in format_args:
                    format_args[key] = value

            message = self.config.commit_message_template.format(**format_args)

            commit = self.repository.commit(
                message=message, tags=["checkpoint", self.experiment_name]
            )
            commit_hash = commit.commit_hash

            # Copy training state to commits directory
            import shutil

            state_dest = self.repository.commits_dir / f"{commit_hash}_state.json"
            shutil.copy2(state_file, state_dest)

            # Tag as best if applicable
            if self._is_best_checkpoint(state):
                self.best_checkpoint_hash = commit_hash
                self.best_metric_value = state.metrics.get(
                    self.config.save_on_best_metric
                )

                if self.config.tag_best_checkpoints:
                    self.repository.tag_version(
                        name=f"{self.model_name}_best",
                        description=(
                            f"Best checkpoint for {self.config.save_on_best_metric}"
                        ),
                        metrics=state.metrics,
                        commit_ref=commit_hash,
                    )

        # Record checkpoint
        checkpoint_info = {
            "commit_hash": commit_hash,
            "epoch": state.epoch,
            "global_step": state.global_step,
            "timestamp": datetime.now().isoformat(),
            "metrics": state.metrics.copy(),
            "is_best": self._is_best_checkpoint(state),
        }
        self.checkpoint_history.append(checkpoint_info)

        # Save checkpoint history
        self._save_checkpoint_history()

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        # Call registered callbacks
        self._call_callbacks(state, commit_hash)

        return commit_hash

    def load_checkpoint(
        self, commit_hash: Optional[str] = None, load_best: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Load a checkpoint."""
        if load_best and self.best_checkpoint_hash:
            commit_hash = self.best_checkpoint_hash
        elif commit_hash is None:
            # Load latest checkpoint
            if self.checkpoint_history:
                commit_hash = self.checkpoint_history[-1]["commit_hash"]
            else:
                return None

        logger.info(f"Loading checkpoint {commit_hash}")

        # Load weights
        weights = self.repository.get_all_weights(commit_hash)

        # Load training state
        commit = self.repository.version_graph.get_commit(commit_hash)
        if commit:
            # Look for training state in commit directory
            state_file = self.repository.commits_dir / f"{commit_hash}_state.json"
            if state_file.exists():
                state = TrainingState.load(str(state_file))
            else:
                # Create minimal state from commit metadata
                state = TrainingState(
                    epoch=0,
                    global_step=0,
                    learning_rate=0.0,
                    loss=0.0,
                    timestamp=commit.metadata.timestamp,
                )
        else:
            state = None

        return {"weights": weights, "state": state, "commit_hash": commit_hash}

    def get_checkpoint_info(self, commit_hash: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific checkpoint."""
        for checkpoint in self.checkpoint_history:
            if checkpoint["commit_hash"] == commit_hash:
                return checkpoint
        return None

    def list_checkpoints(
        self, include_metrics: bool = True, only_best: bool = False
    ) -> List[Dict[str, Any]]:
        """List available checkpoints."""
        checkpoints = self.checkpoint_history

        if only_best:
            checkpoints = [c for c in checkpoints if c.get("is_best", False)]

        if not include_metrics:
            # Remove metrics from output
            checkpoints = [
                {k: v for k, v in c.items() if k != "metrics"} for c in checkpoints
            ]

        return checkpoints

    def register_checkpoint_callback(
        self, callback: Callable[[TrainingState, Optional[str]], None]
    ) -> None:
        """Register a callback to be called after saving a checkpoint.

        Args:
            callback: Function that takes (TrainingState, Optional[str]) where
                     the second argument is the commit hash (None if auto_commit=False)
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
            logger.debug(f"Registered checkpoint callback: {callback.__name__}")
        else:
            logger.warning(f"Callback {callback.__name__} already registered")

    def unregister_checkpoint_callback(
        self, callback: Callable[[TrainingState, Optional[str]], None]
    ) -> bool:
        """Unregister a previously registered callback.

        Args:
            callback: The callback function to remove

        Returns:
            bool: True if callback was found and removed, False otherwise
        """
        try:
            self._callbacks.remove(callback)
            logger.debug(f"Unregistered checkpoint callback: {callback.__name__}")
            return True
        except ValueError:
            logger.warning(f"Callback {callback.__name__} not found for removal")
            return False

    def clear_callbacks(self) -> int:
        """Clear all registered callbacks.

        Returns:
            int: Number of callbacks that were cleared
        """
        count = len(self._callbacks)
        self._callbacks.clear()
        logger.debug(f"Cleared {count} checkpoint callbacks")
        return count

    def list_callbacks(self) -> List[str]:
        """List names of all registered callbacks.

        Returns:
            List[str]: List of callback function names
        """
        return [callback.__name__ for callback in self._callbacks]

    def _is_best_checkpoint(self, state: TrainingState) -> bool:
        """Check if this is the best checkpoint so far."""
        if not self.config.save_on_best_metric:
            return False

        metric_name = self.config.save_on_best_metric
        if metric_name not in state.metrics:
            return False

        metric_value = state.metrics[metric_name]

        if self.best_metric_value is None:
            return True

        if self.config.minimize_metric:
            return metric_value < self.best_metric_value
        else:
            return metric_value > self.best_metric_value

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints according to retention policy."""
        if not self.config.keep_last_n_checkpoints:
            return

        # Sort checkpoints by timestamp
        sorted_checkpoints = sorted(
            self.checkpoint_history, key=lambda x: x["timestamp"]
        )

        # Identify checkpoints to keep
        keep_hashes = set()

        # Keep last N checkpoints
        if self.config.keep_last_n_checkpoints:
            recent = sorted_checkpoints[-self.config.keep_last_n_checkpoints :]
            keep_hashes.update(c["commit_hash"] for c in recent)

        # Keep best N checkpoints
        if self.config.keep_best_n_checkpoints:
            best = sorted(
                [c for c in sorted_checkpoints if c.get("is_best", False)],
                key=lambda x: x.get("metrics", {}).get(
                    self.config.save_on_best_metric, float("inf")
                ),
                reverse=not self.config.minimize_metric,
            )[: self.config.keep_best_n_checkpoints]
            keep_hashes.update(c["commit_hash"] for c in best)

        # Keep periodic checkpoints
        if self.config.keep_checkpoint_every_n_epochs:
            periodic = [
                c
                for c in sorted_checkpoints
                if c["epoch"] % self.config.keep_checkpoint_every_n_epochs == 0
            ]
            keep_hashes.update(c["commit_hash"] for c in periodic)

        # Always keep the best checkpoint
        if self.best_checkpoint_hash:
            keep_hashes.add(self.best_checkpoint_hash)

        # Remove checkpoints not in keep set
        # Note: We don't actually delete commits from the repository,
        # we just remove them from our tracking
        self.checkpoint_history = [
            c for c in self.checkpoint_history if c["commit_hash"] in keep_hashes
        ]

    def _save_checkpoint_history(self) -> None:
        """Save checkpoint history to file."""
        history_file = (
            self.repository.coral_dir / "checkpoints" / f"{self.experiment_name}.json"
        )
        history_file.parent.mkdir(exist_ok=True)

        with open(history_file, "w") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "experiment_name": self.experiment_name,
                    "best_checkpoint_hash": self.best_checkpoint_hash,
                    "best_metric_value": self.best_metric_value,
                    "checkpoints": self.checkpoint_history,
                },
                f,
                indent=2,
            )

    def _load_checkpoint_history(self) -> None:
        """Load checkpoint history from file."""
        history_file = (
            self.repository.coral_dir / "checkpoints" / f"{self.experiment_name}.json"
        )

        if history_file.exists():
            with open(history_file) as f:
                data = json.load(f)

            self.checkpoint_history = data.get("checkpoints", [])
            self.best_checkpoint_hash = data.get("best_checkpoint_hash")
            self.best_metric_value = data.get("best_metric_value")

    def _call_callbacks(self, state: TrainingState, commit_hash: Optional[str]) -> None:
        """Call all registered callbacks with error handling.

        Args:
            state: The training state when checkpoint was saved
            commit_hash: Hash of the created commit (None if auto_commit=False)
        """
        for callback in self._callbacks:
            try:
                callback(state, commit_hash)
                logger.debug(f"Successfully called callback: {callback.__name__}")
            except Exception as e:
                logger.error(
                    f"Error in checkpoint callback {callback.__name__}: {e}",
                    exc_info=True,
                )
                # Continue with other callbacks even if one fails
