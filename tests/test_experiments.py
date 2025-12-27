"""Tests for experiment tracking functionality."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.experiments import (
    Experiment,
    ExperimentMetric,
    ExperimentStatus,
    ExperimentTracker,
)
from coral.version_control.repository import Repository


@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing."""
    temp_dir = tempfile.mkdtemp()
    repo_path = Path(temp_dir)
    repo = Repository(repo_path, init=True)
    yield repo
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_weights():
    """Create sample weights for testing."""
    return {
        "layer1.weight": WeightTensor(
            data=np.random.randn(10, 5).astype(np.float32),
            metadata=WeightMetadata(
                name="layer1.weight",
                shape=(10, 5),
                dtype=np.float32,
            ),
        ),
    }


class TestExperimentMetric:
    """Test ExperimentMetric class."""

    def test_metric_creation(self):
        """Test creating a metric."""
        metric = ExperimentMetric(name="loss", value=0.5, step=100)
        assert metric.name == "loss"
        assert metric.value == 0.5
        assert metric.step == 100

    def test_metric_serialization(self):
        """Test metric serialization."""
        metric = ExperimentMetric(
            name="accuracy",
            value=0.95,
            step=500,
            metadata={"phase": "validation"},
        )

        data = metric.to_dict()
        assert data["name"] == "accuracy"
        assert data["value"] == 0.95
        assert data["step"] == 500
        assert data["metadata"]["phase"] == "validation"

        restored = ExperimentMetric.from_dict(data)
        assert restored.name == metric.name
        assert restored.value == metric.value
        assert restored.step == metric.step


class TestExperiment:
    """Test Experiment class."""

    def test_experiment_creation(self):
        """Test creating an experiment."""
        exp = Experiment(
            experiment_id="test123",
            name="bert-finetuning",
            description="Fine-tuning BERT on custom data",
            params={"lr": 0.001, "batch_size": 32},
            tags=["nlp", "transformer"],
        )

        assert exp.experiment_id == "test123"
        assert exp.name == "bert-finetuning"
        assert exp.status == ExperimentStatus.PENDING
        assert exp.params["lr"] == 0.001
        assert "nlp" in exp.tags

    def test_experiment_metrics(self):
        """Test experiment metric methods."""
        exp = Experiment(experiment_id="test", name="test-exp")

        exp.metrics.append(ExperimentMetric(name="loss", value=1.0, step=0))
        exp.metrics.append(ExperimentMetric(name="loss", value=0.5, step=100))
        exp.metrics.append(ExperimentMetric(name="loss", value=0.3, step=200))
        exp.metrics.append(ExperimentMetric(name="accuracy", value=0.8, step=100))
        exp.metrics.append(ExperimentMetric(name="accuracy", value=0.9, step=200))

        # Test get_metric_history
        loss_history = exp.get_metric_history("loss")
        assert len(loss_history) == 3

        # Test get_latest_metric
        latest_loss = exp.get_latest_metric("loss")
        assert latest_loss.value == 0.3

        # Test get_best_metric
        best_loss = exp.get_best_metric("loss", mode="min")
        assert best_loss.value == 0.3

        best_acc = exp.get_best_metric("accuracy", mode="max")
        assert best_acc.value == 0.9

    def test_experiment_serialization(self):
        """Test experiment serialization."""
        exp = Experiment(
            experiment_id="test123",
            name="test-exp",
            params={"lr": 0.001},
            tags=["test"],
        )
        exp.status = ExperimentStatus.RUNNING
        exp.metrics.append(ExperimentMetric(name="loss", value=0.5))

        data = exp.to_dict()
        restored = Experiment.from_dict(data)

        assert restored.experiment_id == exp.experiment_id
        assert restored.name == exp.name
        assert restored.status == exp.status
        assert len(restored.metrics) == 1

    def test_experiment_metric_names(self):
        """Test getting unique metric names."""
        exp = Experiment(experiment_id="test", name="test-exp")
        exp.metrics.append(ExperimentMetric(name="loss", value=0.5))
        exp.metrics.append(ExperimentMetric(name="loss", value=0.4))
        exp.metrics.append(ExperimentMetric(name="accuracy", value=0.9))

        names = exp.metric_names
        assert len(names) == 2
        assert "loss" in names
        assert "accuracy" in names


class TestExperimentTracker:
    """Test ExperimentTracker class."""

    def test_start_experiment(self, temp_repo):
        """Test starting an experiment."""
        tracker = ExperimentTracker(temp_repo)

        exp = tracker.start(
            name="test-experiment",
            description="A test experiment",
            params={"lr": 0.001, "epochs": 10},
            tags=["test"],
        )

        assert exp.name == "test-experiment"
        assert exp.status == ExperimentStatus.RUNNING
        assert exp.params["lr"] == 0.001
        assert tracker.current_experiment == exp

    def test_log_metrics(self, temp_repo):
        """Test logging metrics."""
        tracker = ExperimentTracker(temp_repo)
        tracker.start("test-exp")

        tracker.log("loss", 0.5, step=0)
        tracker.log("loss", 0.4, step=100)
        tracker.log("accuracy", 0.8, step=100)

        exp = tracker.current_experiment
        assert len(exp.metrics) == 3
        assert exp.get_latest_metric("loss").value == 0.4

    def test_log_metrics_batch(self, temp_repo):
        """Test logging multiple metrics at once."""
        tracker = ExperimentTracker(temp_repo)
        tracker.start("test-exp")

        tracker.log_metrics({"loss": 0.5, "accuracy": 0.8}, step=100)

        exp = tracker.current_experiment
        assert len(exp.metrics) == 2

    def test_end_experiment(self, temp_repo):
        """Test ending an experiment."""
        tracker = ExperimentTracker(temp_repo)
        tracker.start("test-exp")
        tracker.log("loss", 0.5)

        exp = tracker.end(status=ExperimentStatus.COMPLETED, commit_hash="abc123")

        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.commit_hash == "abc123"
        assert exp.ended_at is not None
        assert tracker.current_experiment is None

    def test_fail_experiment(self, temp_repo):
        """Test failing an experiment."""
        tracker = ExperimentTracker(temp_repo)
        tracker.start("test-exp")

        exp = tracker.fail(error_message="Out of memory")

        assert exp.status == ExperimentStatus.FAILED
        assert "Out of memory" in exp.notes

    def test_list_experiments(self, temp_repo):
        """Test listing experiments."""
        tracker = ExperimentTracker(temp_repo)

        # Create multiple experiments
        tracker.start("exp1")
        tracker.end()

        tracker.start("exp2")
        tracker.end(status=ExperimentStatus.FAILED)

        tracker.start("exp3")
        tracker.end()

        all_exps = tracker.list()
        assert len(all_exps) == 3

        completed = tracker.list(status=ExperimentStatus.COMPLETED)
        assert len(completed) == 2

        failed = tracker.list(status=ExperimentStatus.FAILED)
        assert len(failed) == 1

    def test_get_experiment(self, temp_repo):
        """Test getting experiment by ID."""
        tracker = ExperimentTracker(temp_repo)

        exp = tracker.start("test-exp")
        tracker.end()

        retrieved = tracker.get(exp.experiment_id)
        assert retrieved is not None
        assert retrieved.name == "test-exp"

        not_found = tracker.get("nonexistent")
        assert not_found is None

    def test_compare_experiments(self, temp_repo):
        """Test comparing experiments."""
        tracker = ExperimentTracker(temp_repo)

        # Create first experiment
        exp1 = tracker.start("exp1", params={"lr": 0.001})
        tracker.log("loss", 0.5)
        tracker.log("accuracy", 0.85)
        tracker.end()

        # Create second experiment
        exp2 = tracker.start("exp2", params={"lr": 0.01})
        tracker.log("loss", 0.4)
        tracker.log("accuracy", 0.90)
        tracker.end()

        comparison = tracker.compare([exp1.experiment_id, exp2.experiment_id])

        assert len(comparison["experiments"]) == 2
        assert "loss" in comparison["metrics"]
        assert "accuracy" in comparison["metrics"]
        assert "lr" in comparison["params"]

    def test_find_best(self, temp_repo):
        """Test finding best experiments."""
        tracker = ExperimentTracker(temp_repo)

        # Create experiments with different metrics
        tracker.start("exp1")
        tracker.log("loss", 0.5)
        tracker.end()

        tracker.start("exp2")
        tracker.log("loss", 0.3)
        tracker.end()

        tracker.start("exp3")
        tracker.log("loss", 0.4)
        tracker.end()

        # Find best by loss (min)
        best = tracker.find_best("loss", mode="min", limit=3)
        assert len(best) == 3
        assert best[0]["best_value"] == 0.3  # exp2 should be first

    def test_delete_experiment(self, temp_repo):
        """Test deleting an experiment."""
        tracker = ExperimentTracker(temp_repo)

        exp = tracker.start("to-delete")
        tracker.end()

        assert tracker.delete(exp.experiment_id)
        assert tracker.get(exp.experiment_id) is None
        assert not tracker.delete("nonexistent")

    def test_resume_experiment(self, temp_repo):
        """Test resuming an experiment."""
        tracker = ExperimentTracker(temp_repo)

        # Create and fail an experiment
        original = tracker.start("original", params={"lr": 0.001})
        tracker.log("loss", 0.5)
        tracker.fail()

        # Resume it
        resumed = tracker.resume(original.experiment_id)

        assert resumed.name == "original-resumed"
        assert resumed.params["lr"] == 0.001
        assert resumed.parent_experiment == original.experiment_id

    def test_get_summary(self, temp_repo):
        """Test getting summary statistics."""
        tracker = ExperimentTracker(temp_repo)

        tracker.start("exp1")
        tracker.log("loss", 0.5)
        tracker.end()

        tracker.start("exp2")
        tracker.fail()

        summary = tracker.get_summary()

        assert summary["total_experiments"] == 2
        assert summary["by_status"]["completed"] == 1
        assert summary["by_status"]["failed"] == 1
        assert summary["total_metrics_logged"] == 1

    def test_add_tags(self, temp_repo):
        """Test adding tags to experiment."""
        tracker = ExperimentTracker(temp_repo)
        tracker.start("test-exp", tags=["initial"])

        tracker.add_tags(["new-tag", "another"])

        exp = tracker.current_experiment
        assert "initial" in exp.tags
        assert "new-tag" in exp.tags
        assert "another" in exp.tags

    def test_set_notes(self, temp_repo):
        """Test setting notes."""
        tracker = ExperimentTracker(temp_repo)
        tracker.start("test-exp")

        tracker.set_notes("This experiment tests a new approach")

        exp = tracker.current_experiment
        assert "new approach" in exp.notes

    def test_persistence(self, temp_repo):
        """Test that experiments persist across tracker instances."""
        tracker1 = ExperimentTracker(temp_repo)
        exp = tracker1.start("persistent-exp")
        tracker1.log("loss", 0.5)
        tracker1.end()

        # Create new tracker instance
        tracker2 = ExperimentTracker(temp_repo)

        # Should be able to retrieve the experiment
        retrieved = tracker2.get(exp.experiment_id)
        assert retrieved is not None
        assert retrieved.name == "persistent-exp"
        assert len(retrieved.metrics) == 1

    def test_cannot_start_while_running(self, temp_repo):
        """Test that starting a new experiment while one is running raises error."""
        tracker = ExperimentTracker(temp_repo)
        tracker.start("first-exp")

        with pytest.raises(ValueError, match="already running"):
            tracker.start("second-exp")

    def test_log_without_active_experiment(self, temp_repo):
        """Test that logging without active experiment raises error."""
        tracker = ExperimentTracker(temp_repo)

        with pytest.raises(ValueError, match="No active experiment"):
            tracker.log("loss", 0.5)
