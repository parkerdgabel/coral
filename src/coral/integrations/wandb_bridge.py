"""Weights & Biases integration bridge for Coral.

This module provides integration between Coral's weight versioning system
and Weights & Biases (W&B) experiment tracking.

Example:
    >>> from coral import Repository
    >>> from coral.integrations.wandb_bridge import WandbBridge
    >>>
    >>> repo = Repository("./my-model")
    >>> bridge = WandbBridge(
    ...     repo,
    ...     project="my-project",
    ...     entity="my-team",
    ... )
    >>>
    >>> bridge.start_run("bert-experiment", params={"lr": 0.001})
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

# Check for W&B availability
try:
    import wandb

    HAS_WANDB = True
except ImportError:
    wandb = None  # type: ignore
    HAS_WANDB = False


class WandbBridge(ExperimentBridge):
    """Bridge between Weights & Biases and Coral weight versioning.

    This bridge enables:
    - Logging Coral commit hashes to W&B runs for traceability
    - Retrieving model weights from Coral based on W&B run IDs
    - Automatic logging of Coral repository metadata to W&B

    The bridge logs Coral information in the run config under the "coral"
    namespace to avoid conflicts with other logged configuration.

    Attributes:
        repo: The Coral repository
        project: W&B project name
        entity: W&B entity (user or team)
    """

    def __init__(
        self,
        repo: Repository,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        mode: Optional[str] = None,
    ):
        """Initialize the W&B bridge.

        Args:
            repo: Coral repository for weight versioning
            project: W&B project name
            entity: W&B entity (user or team name)
            mode: W&B mode ("online", "offline", "disabled")

        Raises:
            ImportError: If wandb is not installed
        """
        if not HAS_WANDB:
            raise ImportError(
                "Weights & Biases is required for WandbBridge. "
                "Install with: pip install wandb"
            )

        super().__init__(repo)

        self.project = project
        self.entity = entity
        self.mode = mode
        self._run = None
        self._run_id: Optional[str] = None

    def start_run(
        self,
        name: str,
        params: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> str:
        """Start a new W&B run.

        Args:
            name: Name of the run
            params: Configuration/hyperparameters to log
            tags: Tags to associate with the run

        Returns:
            W&B run ID
        """
        # Build config with Coral info
        config = params.copy() if params else {}
        config["coral"] = {
            "repo_path": str(self.repo.path),
            "branch": self.repo.branch_manager.get_current_branch(),
        }

        # Initialize W&B run
        init_kwargs = {
            "project": self.project,
            "entity": self.entity,
            "name": name,
            "config": config,
            "tags": tags,
        }
        if self.mode:
            init_kwargs["mode"] = self.mode

        self._run = wandb.init(**init_kwargs)
        self._run_id = self._run.id

        logger.info(f"Started W&B run: {name} ({self._run_id})")
        return self._run_id

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to W&B.

        Args:
            metrics: Dictionary of metric names to values
            step: Training step number
        """
        if self._run is None:
            logger.warning("No active W&B run. Call start_run() first.")
            return

        wandb.log(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log additional parameters to W&B config.

        Args:
            params: Dictionary of parameter names to values
        """
        if self._run is None:
            logger.warning("No active W&B run. Call start_run() first.")
            return

        wandb.config.update(params)

    def log_coral_commit(
        self,
        commit_hash: str,
        message: Optional[str] = None,
    ) -> None:
        """Link a Coral commit to the current W&B run.

        This logs the commit hash to the run config for easy retrieval
        and to the run summary for visibility in the W&B UI.

        Args:
            commit_hash: The Coral commit hash
            message: Optional commit message
        """
        if self._run is None:
            logger.warning("No active W&B run. Call start_run() first.")
            return

        # Update config with commit info
        coral_config = {
            "coral.commit_hash": commit_hash,
            "coral.commit_short": commit_hash[:8],
            "coral.branch": self.repo.branch_manager.get_current_branch(),
        }
        if message:
            coral_config["coral.commit_message"] = message[:500]

        wandb.config.update(coral_config, allow_val_change=True)

        # Also add to summary for UI visibility
        wandb.run.summary["coral_commit"] = commit_hash[:8]
        current_branch = self.repo.branch_manager.get_current_branch()
        wandb.run.summary["coral_branch"] = current_branch

        logger.debug(f"Logged Coral commit {commit_hash[:8]} to W&B")

    def end_run(self, status: str = "completed") -> None:
        """End the current W&B run.

        Args:
            status: Final status ("completed", "failed", "cancelled")
        """
        if self._run is None:
            logger.warning("No active W&B run to end.")
            return

        # Map status to W&B exit codes
        exit_code = 0 if status == "completed" else 1

        wandb.finish(exit_code=exit_code)
        logger.info(f"Ended W&B run: {self._run_id} ({status})")

        self._run = None
        self._run_id = None

    def get_commit_for_run(self, run_path: str) -> Optional[str]:
        """Get the Coral commit hash for a W&B run.

        Args:
            run_path: W&B run path (entity/project/run_id) or just run_id

        Returns:
            Coral commit hash, or None if not found
        """
        try:
            api = wandb.Api()

            # Handle both full path and just run_id
            if "/" not in run_path:
                # Construct full path from project/entity
                if self.entity and self.project:
                    run_path = f"{self.entity}/{self.project}/{run_path}"
                elif self.project:
                    run_path = f"{self.project}/{run_path}"
                else:
                    logger.warning(
                        f"Cannot resolve run_id {run_path} without project/entity"
                    )
                    return None

            run = api.run(run_path)
            return run.config.get("coral.commit_hash")
        except Exception as e:
            logger.warning(f"Failed to get commit for run {run_path}: {e}")
            return None

    def find_runs_for_commit(
        self,
        commit_hash: str,
        project: Optional[str] = None,
        entity: Optional[str] = None,
    ) -> list[str]:
        """Find all W&B runs associated with a Coral commit.

        Args:
            commit_hash: Full or partial Coral commit hash
            project: W&B project (uses instance default if not specified)
            entity: W&B entity (uses instance default if not specified)

        Returns:
            List of W&B run paths
        """
        try:
            api = wandb.Api()
            project = project or self.project
            entity = entity or self.entity

            if not project:
                logger.warning("Project name required to search runs")
                return []

            # Build path
            path = f"{entity}/{project}" if entity else project

            # Search for runs with matching commit
            runs = api.runs(
                path,
                filters={"config.coral.commit_hash": commit_hash},
            )
            return [run.path for run in runs]
        except Exception as e:
            logger.warning(f"Failed to search runs for commit {commit_hash}: {e}")
            return []

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set additional tags on the current run.

        Note: W&B tags are a list of strings, not key-value pairs.
        This method converts the dict values to tags.

        Args:
            tags: Dictionary of tag names to values
        """
        if self._run is None:
            logger.warning("No active W&B run. Call start_run() first.")
            return

        # W&B tags are strings, so combine key-value pairs
        tag_list = [f"{k}:{v}" for k, v in tags.items()]
        wandb.run.tags = list(wandb.run.tags) + tag_list

    @property
    def is_run_active(self) -> bool:
        """Check if there is an active W&B run.

        Returns:
            True if a run is currently active
        """
        return self._run is not None

    @property
    def current_run_id(self) -> Optional[str]:
        """Get the current active run ID.

        Returns:
            Run ID or None if no active run
        """
        return self._run_id

    @property
    def current_run_url(self) -> Optional[str]:
        """Get the URL for the current run in the W&B UI.

        Returns:
            Run URL or None if no active run
        """
        if self._run is None:
            return None
        return self._run.url

    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        local_path: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log an artifact to W&B.

        This can be used to log additional files alongside the Coral weights.

        Args:
            name: Artifact name
            artifact_type: Artifact type (e.g., "model", "dataset", "config")
            local_path: Path to the local file or directory
            metadata: Optional metadata for the artifact
        """
        if self._run is None:
            logger.warning("No active W&B run. Call start_run() first.")
            return

        artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata)
        artifact.add_file(local_path)
        wandb.log_artifact(artifact)

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Update the run summary.

        The summary contains the final values displayed in the W&B UI table.

        Args:
            summary: Dictionary of summary metrics
        """
        if self._run is None:
            logger.warning("No active W&B run. Call start_run() first.")
            return

        for key, value in summary.items():
            wandb.run.summary[key] = value

    def watch_model(
        self, model: Any, log: str = "gradients", log_freq: int = 100
    ) -> None:
        """Watch a PyTorch model for automatic gradient/parameter logging.

        Args:
            model: PyTorch model to watch
            log: What to log ("gradients", "parameters", "all")
            log_freq: Logging frequency in steps
        """
        if self._run is None:
            logger.warning("No active W&B run. Call start_run() first.")
            return

        wandb.watch(model, log=log, log_freq=log_freq)

    def get_run_info(self, run_path: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Get information about a W&B run.

        Args:
            run_path: Run path (uses current run if not specified)

        Returns:
            Dictionary with run information, or None if not found
        """
        if run_path is None and self._run is None:
            return None

        try:
            if run_path:
                api = wandb.Api()
                run = api.run(run_path)
            else:
                run = self._run

            return {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state if hasattr(run, "state") else "running",
                "url": run.url,
                "config": dict(run.config),
                "summary": dict(run.summary) if hasattr(run, "summary") else {},
                "tags": list(run.tags) if hasattr(run, "tags") else [],
                "coral_commit": run.config.get("coral.commit_hash"),
            }
        except Exception as e:
            logger.warning(f"Failed to get run info: {e}")
            return None
