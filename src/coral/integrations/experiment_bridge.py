"""Base class for experiment tracking bridges.

This module provides an abstract interface for integrating Coral's weight
versioning with external experiment tracking systems like MLflow and
Weights & Biases.

Example:
    >>> from coral.integrations.mlflow_bridge import MLflowBridge
    >>> bridge = MLflowBridge(repo, experiment_name="my-experiment")
    >>> bridge.start_run("training-run-1", params={"lr": 0.001})
    >>> # ... training loop ...
    >>> bridge.log_metrics({"loss": 0.5, "accuracy": 0.95}, step=100)
    >>> bridge.log_coral_commit(commit_hash, "Epoch 1 checkpoint")
    >>> bridge.end_run()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from coral.core.weight_tensor import WeightTensor
    from coral.version_control.repository import Repository


class ExperimentBridge(ABC):
    """Abstract base class for bridging Coral with experiment trackers.

    This class defines the interface for connecting Coral's weight versioning
    system with external experiment tracking tools. Subclasses implement the
    specific integration for each tracking system (MLflow, W&B, etc.).

    The key concept is bidirectional linking:
    - Log Coral commit hashes to experiment runs for traceability
    - Query experiment runs to retrieve associated weights from Coral

    Attributes:
        repo: The Coral repository for weight versioning
    """

    def __init__(self, repo: Repository):
        """Initialize the experiment bridge.

        Args:
            repo: Coral repository for weight versioning
        """
        self.repo = repo

    @abstractmethod
    def start_run(
        self,
        name: str,
        params: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> str:
        """Start a new experiment run.

        Args:
            name: Name of the experiment run
            params: Hyperparameters and configuration to log
            tags: Tags to associate with the run

        Returns:
            Run ID from the experiment tracking system
        """
        pass

    @abstractmethod
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to the current run.

        Args:
            metrics: Dictionary of metric names to values
            step: Training step/iteration number
        """
        pass

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log additional parameters to the current run.

        Args:
            params: Dictionary of parameter names to values
        """
        pass

    @abstractmethod
    def log_coral_commit(
        self,
        commit_hash: str,
        message: Optional[str] = None,
    ) -> None:
        """Link a Coral commit to the current experiment run.

        This is the key integration point - it records the Coral commit
        hash in the experiment tracker, allowing users to trace back
        from an experiment run to the exact model weights.

        Args:
            commit_hash: The Coral commit hash to link
            message: Optional commit message for context
        """
        pass

    @abstractmethod
    def end_run(self, status: str = "completed") -> None:
        """End the current experiment run.

        Args:
            status: Final status of the run ("completed", "failed", "cancelled")
        """
        pass

    @abstractmethod
    def get_commit_for_run(self, run_id: str) -> Optional[str]:
        """Retrieve the Coral commit hash associated with an experiment run.

        Args:
            run_id: The experiment tracking system's run ID

        Returns:
            The Coral commit hash, or None if not found
        """
        pass

    def load_weights_for_run(self, run_id: str) -> dict[str, WeightTensor]:
        """Load Coral weights associated with an experiment run.

        This enables loading the exact model weights used in a specific
        experiment run, providing full reproducibility.

        Args:
            run_id: The experiment tracking system's run ID

        Returns:
            Dictionary of weight names to WeightTensor objects

        Raises:
            ValueError: If no Coral commit is linked to the run
        """
        commit_hash = self.get_commit_for_run(run_id)
        if commit_hash is None:
            raise ValueError(f"No Coral commit linked to run {run_id}")
        return self.repo.get_all_weights(commit_hash)

    def set_tags(self, tags: dict[str, str]) -> None:  # noqa: B027
        """Set additional tags on the current run.

        Default implementation does nothing. Override in subclasses
        that support dynamic tag updates.

        Args:
            tags: Dictionary of tag names to values
        """

    @property
    def is_run_active(self) -> bool:
        """Check if there is an active experiment run.

        Default implementation returns False. Override in subclasses.

        Returns:
            True if a run is currently active
        """
        return False
