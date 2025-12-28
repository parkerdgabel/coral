"""MLflow integration bridge for Coral.

This module provides integration between Coral's weight versioning system
and MLflow experiment tracking.

Example:
    >>> from coral import Repository
    >>> from coral.integrations.mlflow_bridge import MLflowBridge
    >>>
    >>> repo = Repository("./my-model")
    >>> bridge = MLflowBridge(
    ...     repo,
    ...     tracking_uri="http://localhost:5000",
    ...     experiment_name="fine-tuning",
    ... )
    >>>
    >>> bridge.start_run("bert-lr-sweep", params={"lr": 0.001, "epochs": 10})
    >>> # ... training loop ...
    >>> bridge.log_metrics({"loss": 0.5, "accuracy": 0.95}, step=100)
    >>>
    >>> # After saving weights to Coral
    >>> commit = repo.commit("Epoch 1 checkpoint")
    >>> bridge.log_coral_commit(commit.commit_hash)
    >>>
    >>> bridge.end_run()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from .experiment_bridge import ExperimentBridge

if TYPE_CHECKING:
    from coral.version_control.repository import Repository

logger = logging.getLogger(__name__)

# Check for MLflow availability
try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:
    mlflow = None  # type: ignore
    HAS_MLFLOW = False


class MLflowBridge(ExperimentBridge):
    """Bridge between MLflow experiment tracking and Coral weight versioning.

    This bridge enables:
    - Logging Coral commit hashes to MLflow runs for traceability
    - Retrieving model weights from Coral based on MLflow run IDs
    - Automatic logging of Coral repository metadata to MLflow

    The bridge logs Coral information under the "coral." namespace in MLflow
    to avoid conflicts with other logged parameters and tags.

    Attributes:
        repo: The Coral repository
        tracking_uri: MLflow tracking server URI
        experiment_name: MLflow experiment name
    """

    def __init__(
        self,
        repo: Repository,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ):
        """Initialize the MLflow bridge.

        Args:
            repo: Coral repository for weight versioning
            tracking_uri: MLflow tracking server URI
                (uses MLFLOW_TRACKING_URI env var if not set)
            experiment_name: MLflow experiment name (creates if doesn't exist)
            artifact_location: Custom artifact storage location

        Raises:
            ImportError: If MLflow is not installed
        """
        if not HAS_MLFLOW:
            raise ImportError(
                "MLflow is required for MLflowBridge. "
                "Install with: pip install mlflow"
            )

        super().__init__(repo)

        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._active_run = None
        self._run_id: Optional[str] = None

        # Configure MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        if experiment_name:
            mlflow.set_experiment(experiment_name)

    def start_run(
        self,
        name: str,
        params: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> str:
        """Start a new MLflow run.

        Args:
            name: Name of the run
            params: Hyperparameters to log
            tags: Tags to associate with the run

        Returns:
            MLflow run ID
        """
        self._active_run = mlflow.start_run(run_name=name)
        self._run_id = self._active_run.info.run_id

        # Log Coral repository info
        mlflow.set_tag("coral.repo_path", str(self.repo.path))
        mlflow.set_tag("coral.branch", self.repo.branch_manager.get_current_branch())

        # Log parameters
        if params:
            mlflow.log_params(params)

        # Log tags
        if tags:
            for tag in tags:
                mlflow.set_tag(tag, "true")

        logger.info(f"Started MLflow run: {name} ({self._run_id})")
        return self._run_id

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metric names to values
            step: Training step number
        """
        if self._active_run is None:
            logger.warning("No active MLflow run. Call start_run() first.")
            return

        mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log additional parameters to MLflow.

        Args:
            params: Dictionary of parameter names to values
        """
        if self._active_run is None:
            logger.warning("No active MLflow run. Call start_run() first.")
            return

        mlflow.log_params(params)

    def log_coral_commit(
        self,
        commit_hash: str,
        message: Optional[str] = None,
    ) -> None:
        """Link a Coral commit to the current MLflow run.

        This logs the commit hash as both a parameter (for querying) and
        a tag (for display in the UI).

        Args:
            commit_hash: The Coral commit hash
            message: Optional commit message
        """
        if self._active_run is None:
            logger.warning("No active MLflow run. Call start_run() first.")
            return

        # Log as parameter for querying
        mlflow.log_param("coral.commit_hash", commit_hash)
        mlflow.log_param("coral.commit_short", commit_hash[:8])

        # Log commit message if provided
        if message:
            # MLflow has a 500 char limit for params
            mlflow.log_param("coral.commit_message", message[:500])

        # Also set as tag for UI visibility
        mlflow.set_tag("coral.commit", commit_hash[:8])

        # Update branch info in case it changed
        mlflow.set_tag("coral.branch", self.repo.branch_manager.get_current_branch())

        logger.debug(f"Logged Coral commit {commit_hash[:8]} to MLflow")

    def end_run(self, status: str = "completed") -> None:
        """End the current MLflow run.

        Args:
            status: Final status ("completed", "failed", "cancelled")
        """
        if self._active_run is None:
            logger.warning("No active MLflow run to end.")
            return

        # Map status to MLflow run status
        status_map = {
            "completed": "FINISHED",
            "failed": "FAILED",
            "cancelled": "KILLED",
        }
        mlflow_status = status_map.get(status, "FINISHED")

        mlflow.end_run(status=mlflow_status)
        logger.info(f"Ended MLflow run: {self._run_id} ({status})")

        self._active_run = None
        self._run_id = None

    def get_commit_for_run(self, run_id: str) -> Optional[str]:
        """Get the Coral commit hash for an MLflow run.

        Args:
            run_id: MLflow run ID

        Returns:
            Coral commit hash, or None if not found
        """
        try:
            run = mlflow.get_run(run_id)
            return run.data.params.get("coral.commit_hash")
        except Exception as e:
            logger.warning(f"Failed to get commit for run {run_id}: {e}")
            return None

    def find_runs_for_commit(self, commit_hash: str) -> list[str]:
        """Find all MLflow runs associated with a Coral commit.

        Args:
            commit_hash: Full or partial Coral commit hash

        Returns:
            List of MLflow run IDs
        """
        try:
            # Search for runs with matching commit hash
            runs = mlflow.search_runs(
                filter_string=f"params.coral.commit_hash = '{commit_hash}'",
                output_format="list",
            )
            return [run.info.run_id for run in runs]
        except Exception as e:
            logger.warning(f"Failed to search runs for commit {commit_hash}: {e}")
            return []

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set additional tags on the current run.

        Args:
            tags: Dictionary of tag names to values
        """
        if self._active_run is None:
            logger.warning("No active MLflow run. Call start_run() first.")
            return

        for key, value in tags.items():
            mlflow.set_tag(key, value)

    @property
    def is_run_active(self) -> bool:
        """Check if there is an active MLflow run.

        Returns:
            True if a run is currently active
        """
        return self._active_run is not None

    @property
    def current_run_id(self) -> Optional[str]:
        """Get the current active run ID.

        Returns:
            Run ID or None if no active run
        """
        return self._run_id

    def log_artifact(
        self, local_path: str, artifact_path: Optional[str] = None
    ) -> None:
        """Log an artifact to MLflow.

        This can be used to log additional files alongside the Coral weights,
        such as configuration files, plots, or evaluation results.

        Args:
            local_path: Path to the local file or directory
            artifact_path: Destination path within the artifact store
        """
        if self._active_run is None:
            logger.warning("No active MLflow run. Call start_run() first.")
            return

        mlflow.log_artifact(local_path, artifact_path)

    def get_run_info(self, run_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Get information about an MLflow run.

        Args:
            run_id: Run ID (uses current run if not specified)

        Returns:
            Dictionary with run information, or None if not found
        """
        target_run_id = run_id or self._run_id
        if target_run_id is None:
            return None

        try:
            run = mlflow.get_run(target_run_id)
            return {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
                "tags": dict(run.data.tags),
                "coral_commit": run.data.params.get("coral.commit_hash"),
            }
        except Exception as e:
            logger.warning(f"Failed to get run info for {target_run_id}: {e}")
            return None
