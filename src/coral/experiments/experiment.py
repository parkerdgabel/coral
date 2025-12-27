"""Experiment tracking implementation.

Provides experiment tracking with metrics logging, comparison, and best model
finding capabilities.
"""

from __future__ import annotations

import builtins
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from coral.version_control.repository import Repository

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an experiment."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentMetric:
    """A single metric measurement."""

    name: str
    value: float
    step: int | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "step": self.step,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentMetric:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            value=data["value"],
            step=data.get("step"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Experiment:
    """Represents a single experiment/training run."""

    experiment_id: str
    name: str
    description: str | None = None
    status: ExperimentStatus = ExperimentStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    metrics: list[ExperimentMetric] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    commit_hash: str | None = None
    branch: str | None = None
    parent_experiment: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "metrics": [m.to_dict() for m in self.metrics],
            "params": self.params,
            "tags": self.tags,
            "commit_hash": self.commit_hash,
            "branch": self.branch,
            "parent_experiment": self.parent_experiment,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Experiment:
        """Create from dictionary."""
        return cls(
            experiment_id=data["experiment_id"],
            name=data["name"],
            description=data.get("description"),
            status=ExperimentStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=(
                datetime.fromisoformat(data["started_at"])
                if data.get("started_at")
                else None
            ),
            ended_at=(
                datetime.fromisoformat(data["ended_at"])
                if data.get("ended_at")
                else None
            ),
            metrics=[ExperimentMetric.from_dict(m) for m in data.get("metrics", [])],
            params=data.get("params", {}),
            tags=data.get("tags", []),
            commit_hash=data.get("commit_hash"),
            branch=data.get("branch"),
            parent_experiment=data.get("parent_experiment"),
            notes=data.get("notes"),
        )

    def get_metric_history(self, name: str) -> list[ExperimentMetric]:
        """Get all values of a specific metric."""
        return [m for m in self.metrics if m.name == name]

    def get_latest_metric(self, name: str) -> ExperimentMetric | None:
        """Get the latest value of a specific metric."""
        history = self.get_metric_history(name)
        return history[-1] if history else None

    def get_best_metric(
        self, name: str, mode: str = "min"
    ) -> ExperimentMetric | None:
        """Get the best value of a specific metric.

        Args:
            name: Metric name
            mode: 'min' for minimum, 'max' for maximum
        """
        history = self.get_metric_history(name)
        if not history:
            return None
        if mode == "min":
            return min(history, key=lambda m: m.value)
        return max(history, key=lambda m: m.value)

    @property
    def duration(self) -> float | None:
        """Get experiment duration in seconds."""
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return None

    @property
    def metric_names(self) -> list[str]:
        """Get list of unique metric names."""
        return list({m.name for m in self.metrics})


class ExperimentTracker:
    """Tracks experiments for a Coral repository.

    Provides functionality to:
    - Start and stop experiments
    - Log metrics during training
    - Compare experiments
    - Find best performing experiments

    Example:
        >>> from coral.version_control.repository import Repository
        >>> from coral.experiments import ExperimentTracker
        >>>
        >>> repo = Repository("./my-model")
        >>> tracker = ExperimentTracker(repo)
        >>>
        >>> # Start an experiment
        >>> exp = tracker.start("bert-finetuning", params={"lr": 0.001})
        >>>
        >>> # Log metrics during training
        >>> tracker.log("loss", 0.5, step=100)
        >>> tracker.log("accuracy", 0.85, step=100)
        >>>
        >>> # End experiment and associate with commit
        >>> tracker.end(commit_hash="abc123")
    """

    def __init__(self, repo: Repository):
        """Initialize experiment tracker.

        Args:
            repo: Coral Repository instance
        """
        self.repo = repo
        self.experiments_dir = repo.coral_dir / "experiments"
        self.experiments_dir.mkdir(exist_ok=True)
        self.current_experiment: Experiment | None = None
        self._experiments: dict[str, Experiment] = {}
        self._load_experiments()

    def _load_experiments(self) -> None:
        """Load all experiments from disk."""
        for exp_file in self.experiments_dir.glob("*.json"):
            try:
                with open(exp_file) as f:
                    data = json.load(f)
                exp = Experiment.from_dict(data)
                self._experiments[exp.experiment_id] = exp
            except Exception as e:
                logger.warning(f"Failed to load experiment {exp_file}: {e}")

    def _save_experiment(self, experiment: Experiment) -> None:
        """Save experiment to disk."""
        exp_file = self.experiments_dir / f"{experiment.experiment_id}.json"
        with open(exp_file, "w") as f:
            json.dump(experiment.to_dict(), f, indent=2)

    def _generate_experiment_id(self, name: str) -> str:
        """Generate unique experiment ID."""
        import hashlib

        timestamp = datetime.now().isoformat()
        content = f"{name}:{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def start(
        self,
        name: str,
        description: str | None = None,
        params: dict[str, Any] | None = None,
        tags: builtins.list[str] | None = None,
        parent: str | None = None,
    ) -> Experiment:
        """Start a new experiment.

        Args:
            name: Experiment name
            description: Optional description
            params: Training parameters (lr, batch_size, etc.)
            tags: Tags for categorization
            parent: Parent experiment ID (for resuming/continuing)

        Returns:
            The created Experiment
        """
        if (
            self.current_experiment
            and self.current_experiment.status == ExperimentStatus.RUNNING
        ):
            raise ValueError(
                f"Experiment '{self.current_experiment.name}' is already running. "
                "End it first with tracker.end()"
            )

        experiment_id = self._generate_experiment_id(name)
        current_branch = self.repo.branch_manager.get_current_branch()

        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            status=ExperimentStatus.RUNNING,
            started_at=datetime.now(),
            params=params or {},
            tags=tags or [],
            branch=current_branch,
            parent_experiment=parent,
        )

        self._experiments[experiment_id] = experiment
        self.current_experiment = experiment
        self._save_experiment(experiment)

        logger.info(f"Started experiment: {name} ({experiment_id})")
        return experiment

    def log(
        self,
        name: str,
        value: float,
        step: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a metric value.

        Args:
            name: Metric name (e.g., 'loss', 'accuracy')
            value: Metric value
            step: Training step/epoch
            metadata: Additional metadata
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start() first.")

        metric = ExperimentMetric(
            name=name,
            value=value,
            step=step,
            metadata=metadata or {},
        )
        self.current_experiment.metrics.append(metric)
        self._save_experiment(self.current_experiment)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric name to value
            step: Training step/epoch
        """
        for name, value in metrics.items():
            self.log(name, value, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log/update experiment parameters.

        Args:
            params: Dictionary of parameter name to value
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start() first.")

        self.current_experiment.params.update(params)
        self._save_experiment(self.current_experiment)

    def add_tags(self, tags: builtins.list[str]) -> None:
        """Add tags to current experiment.

        Args:
            tags: List of tags to add
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start() first.")

        self.current_experiment.tags.extend(tags)
        self.current_experiment.tags = list(set(self.current_experiment.tags))
        self._save_experiment(self.current_experiment)

    def set_notes(self, notes: str) -> None:
        """Set notes for current experiment.

        Args:
            notes: Notes text
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start() first.")

        self.current_experiment.notes = notes
        self._save_experiment(self.current_experiment)

    def end(
        self,
        status: ExperimentStatus = ExperimentStatus.COMPLETED,
        commit_hash: str | None = None,
    ) -> Experiment:
        """End the current experiment.

        Args:
            status: Final status (completed, failed, cancelled)
            commit_hash: Associated commit hash

        Returns:
            The ended Experiment
        """
        if not self.current_experiment:
            raise ValueError("No active experiment to end.")

        self.current_experiment.status = status
        self.current_experiment.ended_at = datetime.now()
        self.current_experiment.commit_hash = commit_hash

        self._save_experiment(self.current_experiment)

        experiment = self.current_experiment
        self.current_experiment = None

        logger.info(
            f"Ended experiment: {experiment.name} ({experiment.experiment_id}) "
            f"- {status.value}"
        )
        return experiment

    def fail(self, error_message: str | None = None) -> Experiment:
        """Mark current experiment as failed.

        Args:
            error_message: Optional error message to add to notes
        """
        if error_message and self.current_experiment:
            existing_notes = self.current_experiment.notes or ""
            self.current_experiment.notes = f"{existing_notes}\nError: {error_message}"
        return self.end(status=ExperimentStatus.FAILED)

    def get(self, experiment_id: str) -> Experiment | None:
        """Get an experiment by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment if found, None otherwise
        """
        return self._experiments.get(experiment_id)

    def get_by_name(self, name: str) -> builtins.list[Experiment]:
        """Get all experiments with a given name.

        Args:
            name: Experiment name

        Returns:
            List of matching experiments
        """
        return [e for e in self._experiments.values() if e.name == name]

    def list(
        self,
        status: ExperimentStatus | None = None,
        tags: builtins.list[str] | None = None,
        branch: str | None = None,
        limit: int = 50,
    ) -> builtins.list[Experiment]:
        """List experiments with optional filtering.

        Args:
            status: Filter by status
            tags: Filter by tags (experiments must have all tags)
            branch: Filter by branch
            limit: Maximum number to return

        Returns:
            List of matching experiments, sorted by creation time (newest first)
        """
        experiments = list(self._experiments.values())

        if status:
            experiments = [e for e in experiments if e.status == status]

        if tags:
            experiments = [e for e in experiments if all(t in e.tags for t in tags)]

        if branch:
            experiments = [e for e in experiments if e.branch == branch]

        # Sort by creation time, newest first
        experiments.sort(key=lambda e: e.created_at, reverse=True)

        return experiments[:limit]

    def compare(
        self,
        experiment_ids: builtins.list[str],
        metrics: builtins.list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare multiple experiments.

        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: Specific metrics to compare (None for all)

        Returns:
            Comparison dictionary with experiments and metrics
        """
        experiments = []
        for exp_id in experiment_ids:
            exp = self.get(exp_id)
            if exp:
                experiments.append(exp)

        if not experiments:
            return {"experiments": [], "metrics": {}, "params": {}}

        # Collect all metric names if not specified
        if metrics is None:
            metrics = list({m for e in experiments for m in e.metric_names})

        # Build comparison table
        comparison = {
            "experiments": [
                {
                    "id": e.experiment_id,
                    "name": e.name,
                    "status": e.status.value,
                    "duration": e.duration,
                    "created_at": e.created_at.isoformat(),
                }
                for e in experiments
            ],
            "metrics": {},
            "params": {},
        }

        # Compare metrics
        for metric_name in metrics:
            comparison["metrics"][metric_name] = {}
            for exp in experiments:
                best = exp.get_best_metric(metric_name)
                latest = exp.get_latest_metric(metric_name)
                comparison["metrics"][metric_name][exp.experiment_id] = {
                    "best": best.value if best else None,
                    "latest": latest.value if latest else None,
                    "count": len(exp.get_metric_history(metric_name)),
                }

        # Compare params
        all_params = set()
        for exp in experiments:
            all_params.update(exp.params.keys())

        for param in all_params:
            comparison["params"][param] = {
                exp.experiment_id: exp.params.get(param) for exp in experiments
            }

        return comparison

    def find_best(
        self,
        metric: str,
        mode: str = "min",
        status: ExperimentStatus | None = ExperimentStatus.COMPLETED,
        tags: builtins.list[str] | None = None,
        limit: int = 10,
    ) -> builtins.list[dict[str, Any]]:
        """Find best experiments by a metric.

        Args:
            metric: Metric name to optimize
            mode: 'min' for minimum, 'max' for maximum
            status: Filter by status (default: completed only)
            tags: Filter by tags
            limit: Maximum number to return

        Returns:
            List of experiments with their best metric values
        """
        experiments = self.list(status=status, tags=tags, limit=1000)

        results = []
        for exp in experiments:
            best = exp.get_best_metric(metric, mode=mode)
            if best:
                results.append(
                    {
                        "experiment_id": exp.experiment_id,
                        "name": exp.name,
                        "metric": metric,
                        "best_value": best.value,
                        "best_step": best.step,
                        "commit_hash": exp.commit_hash,
                        "params": exp.params,
                    }
                )

        # Sort by metric value
        reverse = mode == "max"
        results.sort(key=lambda x: x["best_value"], reverse=reverse)

        return results[:limit]

    def delete(self, experiment_id: str) -> bool:
        """Delete an experiment.

        Args:
            experiment_id: Experiment ID to delete

        Returns:
            True if deleted, False if not found
        """
        if experiment_id not in self._experiments:
            return False

        del self._experiments[experiment_id]

        exp_file = self.experiments_dir / f"{experiment_id}.json"
        if exp_file.exists():
            exp_file.unlink()

        return True

    def resume(self, experiment_id: str) -> Experiment:
        """Resume a stopped or failed experiment.

        Creates a new experiment with the same parameters and links
        it to the original as a parent.

        Args:
            experiment_id: Experiment ID to resume

        Returns:
            New experiment
        """
        original = self.get(experiment_id)
        if not original:
            raise ValueError(f"Experiment {experiment_id} not found")

        return self.start(
            name=f"{original.name}-resumed",
            description=original.description,
            params=original.params.copy(),
            tags=original.tags.copy(),
            parent=experiment_id,
        )

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of all experiments.

        Returns:
            Summary dictionary with counts and statistics
        """
        experiments = list(self._experiments.values())

        status_counts = {}
        for status in ExperimentStatus:
            status_counts[status.value] = sum(
                1 for e in experiments if e.status == status
            )

        return {
            "total_experiments": len(experiments),
            "by_status": status_counts,
            "unique_names": len({e.name for e in experiments}),
            "branches": list({e.branch for e in experiments if e.branch}),
            "total_metrics_logged": sum(len(e.metrics) for e in experiments),
        }
