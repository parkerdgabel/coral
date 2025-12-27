"""Experiment tracking for Coral.

This module provides experiment tracking capabilities for ML training runs,
allowing users to track metrics, compare experiments, and find the best
performing models.

Example:
    >>> from coral.experiments import ExperimentTracker
    >>>
    >>> tracker = ExperimentTracker(repo)
    >>> tracker.start("fine-tuning-bert")
    >>> tracker.log_metric("loss", 0.5)
    >>> tracker.log_metric("accuracy", 0.95)
    >>> tracker.end()
"""

from .experiment import (
    Experiment,
    ExperimentMetric,
    ExperimentStatus,
    ExperimentTracker,
)

__all__ = [
    "Experiment",
    "ExperimentMetric",
    "ExperimentStatus",
    "ExperimentTracker",
]
