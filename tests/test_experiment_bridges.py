"""Tests for experiment tracking bridges (WandB and MLflow).

This module tests the WandbBridge and MLflowBridge.
Tests are skipped if the respective libraries are not installed.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from coral.version_control.repository import Repository

# Check if WandB is available
try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

# Check if MLflow is available
try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    mlflow = None


@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing."""
    temp_dir = tempfile.mkdtemp()
    repo_path = Path(temp_dir)
    repo = Repository(repo_path, init=True)
    yield repo
    shutil.rmtree(temp_dir)


# ============================================================================
# WandB Bridge Tests
# ============================================================================


@pytest.mark.skipif(not HAS_WANDB, reason="Weights & Biases not installed")
class TestWandbBridge:
    """Tests for WandbBridge."""

    @pytest.fixture
    def mock_wandb_run(self):
        """Mock wandb run and API."""
        mock_run = MagicMock()
        mock_run.id = "test-run-123"
        mock_run.url = "https://wandb.ai/test/run/test-run-123"
        mock_run.name = "test-run"
        mock_run.tags = []
        mock_run.summary = {}
        mock_run.config = {}
        return mock_run

    def test_init_with_options(self, temp_repo):
        """Test initialization with options."""
        from coral.integrations.wandb_bridge import WandbBridge

        bridge = WandbBridge(
            temp_repo,
            project="test-project",
            entity="test-entity",
            mode="offline",
        )

        assert bridge.project == "test-project"
        assert bridge.entity == "test-entity"
        assert bridge.mode == "offline"
        assert bridge.is_run_active is False

    def test_start_run(self, temp_repo, mock_wandb_run):
        """Test starting a run."""
        from coral.integrations.wandb_bridge import WandbBridge

        with patch.object(wandb, "init", return_value=mock_wandb_run):
            bridge = WandbBridge(temp_repo, project="test-project")
            run_id = bridge.start_run("test-run", params={"lr": 0.001})

        assert run_id == "test-run-123"
        assert bridge.is_run_active is True
        assert bridge.current_run_id == "test-run-123"

    def test_start_run_with_tags(self, temp_repo, mock_wandb_run):
        """Test starting a run with tags."""
        from coral.integrations.wandb_bridge import WandbBridge

        with patch.object(wandb, "init", return_value=mock_wandb_run) as mock_init:
            bridge = WandbBridge(temp_repo, project="test-project")
            bridge.start_run("test-run", tags=["experiment", "baseline"])

            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["tags"] == ["experiment", "baseline"]

    def test_log_metrics(self, temp_repo, mock_wandb_run):
        """Test logging metrics."""
        from coral.integrations.wandb_bridge import WandbBridge

        with patch.object(wandb, "init", return_value=mock_wandb_run):
            with patch.object(wandb, "log") as mock_log:
                bridge = WandbBridge(temp_repo, project="test-project")
                bridge.start_run("test-run")
                bridge.log_metrics({"loss": 0.5, "accuracy": 0.95}, step=100)

                mock_log.assert_called_once_with(
                    {"loss": 0.5, "accuracy": 0.95}, step=100
                )

    def test_log_metrics_without_active_run(self, temp_repo):
        """Test logging metrics without active run."""
        from coral.integrations.wandb_bridge import WandbBridge

        with patch.object(wandb, "log") as mock_log:
            bridge = WandbBridge(temp_repo, project="test-project")

            # Should not raise, just warn
            bridge.log_metrics({"loss": 0.5})
            mock_log.assert_not_called()

    def test_log_params(self, temp_repo, mock_wandb_run):
        """Test logging parameters."""
        from coral.integrations.wandb_bridge import WandbBridge

        mock_config = MagicMock()
        with patch.object(wandb, "init", return_value=mock_wandb_run):
            with patch.object(wandb, "config", mock_config):
                bridge = WandbBridge(temp_repo, project="test-project")
                bridge.start_run("test-run")
                bridge.log_params({"epochs": 10, "batch_size": 32})

                mock_config.update.assert_called_with({"epochs": 10, "batch_size": 32})

    def test_log_coral_commit(self, temp_repo, mock_wandb_run):
        """Test logging a Coral commit."""
        from coral.integrations.wandb_bridge import WandbBridge

        mock_config = MagicMock()
        with patch.object(wandb, "init", return_value=mock_wandb_run):
            with patch.object(wandb, "config", mock_config):
                bridge = WandbBridge(temp_repo, project="test-project")
                bridge.start_run("test-run")
                bridge.log_coral_commit("abc123def456", message="Test commit")

                # Verify config update was called
                assert mock_config.update.called

    def test_end_run(self, temp_repo, mock_wandb_run):
        """Test ending a run."""
        from coral.integrations.wandb_bridge import WandbBridge

        with patch.object(wandb, "init", return_value=mock_wandb_run):
            with patch.object(wandb, "finish") as mock_finish:
                bridge = WandbBridge(temp_repo, project="test-project")
                bridge.start_run("test-run")
                bridge.end_run("completed")

                mock_finish.assert_called_once_with(exit_code=0)
                assert bridge.is_run_active is False

    def test_end_run_failed(self, temp_repo, mock_wandb_run):
        """Test ending a failed run."""
        from coral.integrations.wandb_bridge import WandbBridge

        with patch.object(wandb, "init", return_value=mock_wandb_run):
            with patch.object(wandb, "finish") as mock_finish:
                bridge = WandbBridge(temp_repo, project="test-project")
                bridge.start_run("test-run")
                bridge.end_run("failed")

                mock_finish.assert_called_once_with(exit_code=1)

    def test_current_run_url(self, temp_repo, mock_wandb_run):
        """Test getting current run URL."""
        from coral.integrations.wandb_bridge import WandbBridge

        with patch.object(wandb, "init", return_value=mock_wandb_run):
            with patch.object(wandb, "run", mock_wandb_run):
                bridge = WandbBridge(temp_repo, project="test-project")
                bridge.start_run("test-run")

                assert (
                    bridge.current_run_url == "https://wandb.ai/test/run/test-run-123"
                )

    def test_current_run_url_no_run(self, temp_repo):
        """Test getting current run URL when no run is active."""
        from coral.integrations.wandb_bridge import WandbBridge

        bridge = WandbBridge(temp_repo, project="test-project")

        assert bridge.current_run_url is None

    def test_set_tags(self, temp_repo, mock_wandb_run):
        """Test setting tags."""
        from coral.integrations.wandb_bridge import WandbBridge

        with patch.object(wandb, "init", return_value=mock_wandb_run):
            with patch.object(wandb, "run", mock_wandb_run):
                bridge = WandbBridge(temp_repo, project="test-project")
                bridge.start_run("test-run")
                bridge.set_tags({"model": "bert", "dataset": "squad"})

                # Verify tags were updated
                assert mock_wandb_run.tags is not None

    def test_log_artifact(self, temp_repo, mock_wandb_run):
        """Test logging an artifact."""
        from coral.integrations.wandb_bridge import WandbBridge

        mock_artifact = MagicMock()
        with patch.object(wandb, "init", return_value=mock_wandb_run):
            with patch.object(wandb, "Artifact", return_value=mock_artifact):
                with patch.object(wandb, "log_artifact") as mock_log_artifact:
                    bridge = WandbBridge(temp_repo, project="test-project")
                    bridge.start_run("test-run")

                    # Create a test file
                    test_file = temp_repo.coral_dir.parent / "test_artifact.txt"
                    test_file.write_text("test content")

                    bridge.log_artifact(
                        name="test-artifact",
                        artifact_type="model",
                        local_path=str(test_file),
                    )

                    mock_log_artifact.assert_called_once()

    def test_log_summary(self, temp_repo, mock_wandb_run):
        """Test logging summary."""
        from coral.integrations.wandb_bridge import WandbBridge

        with patch.object(wandb, "init", return_value=mock_wandb_run):
            with patch.object(wandb, "run", mock_wandb_run):
                bridge = WandbBridge(temp_repo, project="test-project")
                bridge.start_run("test-run")
                bridge.log_summary({"final_accuracy": 0.95, "final_loss": 0.1})

                assert mock_wandb_run.summary["final_accuracy"] == 0.95
                assert mock_wandb_run.summary["final_loss"] == 0.1

    def test_watch_model(self, temp_repo, mock_wandb_run):
        """Test watching a model."""
        from coral.integrations.wandb_bridge import WandbBridge

        with patch.object(wandb, "init", return_value=mock_wandb_run):
            with patch.object(wandb, "watch") as mock_watch:
                bridge = WandbBridge(temp_repo, project="test-project")
                bridge.start_run("test-run")

                mock_model = MagicMock()
                bridge.watch_model(mock_model, log="gradients", log_freq=100)

                mock_watch.assert_called_once_with(
                    mock_model, log="gradients", log_freq=100
                )

    def test_get_commit_for_run(self, temp_repo, mock_wandb_run):
        """Test getting commit for a run."""
        from coral.integrations.wandb_bridge import WandbBridge

        # Mock API response
        mock_api = MagicMock()
        mock_api_run = MagicMock()
        mock_api_run.config = {"coral.commit_hash": "abc123def456"}
        mock_api.run.return_value = mock_api_run

        with patch.object(wandb, "Api", return_value=mock_api):
            bridge = WandbBridge(
                temp_repo, project="test-project", entity="test-entity"
            )

            commit = bridge.get_commit_for_run("test-entity/test-project/run123")

            assert commit == "abc123def456"

    def test_find_runs_for_commit(self, temp_repo, mock_wandb_run):
        """Test finding runs for a commit."""
        from coral.integrations.wandb_bridge import WandbBridge

        # Mock API response
        mock_api = MagicMock()
        mock_found_run = MagicMock()
        mock_found_run.path = "test-entity/test-project/run123"
        mock_api.runs.return_value = [mock_found_run]

        with patch.object(wandb, "Api", return_value=mock_api):
            bridge = WandbBridge(
                temp_repo, project="test-project", entity="test-entity"
            )

            runs = bridge.find_runs_for_commit("abc123def456")

            assert runs == ["test-entity/test-project/run123"]

    def test_get_run_info(self, temp_repo, mock_wandb_run):
        """Test getting run info."""
        from coral.integrations.wandb_bridge import WandbBridge

        with patch.object(wandb, "init", return_value=mock_wandb_run):
            with patch.object(wandb, "run", mock_wandb_run):
                bridge = WandbBridge(temp_repo, project="test-project")
                bridge.start_run("test-run")

                info = bridge.get_run_info()

                assert info is not None
                assert "run_id" in info
                assert "url" in info


class TestWandbBridgeWithoutWandB:
    """Tests that work without WandB installed."""

    def test_import_error_without_wandb(self, temp_repo):
        """Test that WandbBridge raises ImportError without WandB."""
        if HAS_WANDB:
            pytest.skip("Weights & Biases is installed")

        with patch("coral.integrations.wandb_bridge.HAS_WANDB", False):
            from coral.integrations.wandb_bridge import WandbBridge

            with pytest.raises(ImportError, match="Weights & Biases is required"):
                WandbBridge(temp_repo)


# ============================================================================
# MLflow Bridge Tests
# ============================================================================


@pytest.mark.skipif(not HAS_MLFLOW, reason="MLflow not installed")
class TestMLflowBridge:
    """Tests for MLflowBridge."""

    @pytest.fixture
    def mock_mlflow_run(self):
        """Mock MLflow run."""
        mock_run_info = MagicMock()
        mock_run_info.run_id = "mlflow-run-123"
        mock_run_info.run_name = "test-run"
        mock_run_info.status = "RUNNING"
        mock_run_info.start_time = 1234567890
        mock_run_info.end_time = None

        mock_run = MagicMock()
        mock_run.info = mock_run_info
        return mock_run

    def test_init_with_tracking_uri(self, temp_repo):
        """Test initialization with tracking URI."""
        from coral.integrations.mlflow_bridge import MLflowBridge

        with patch.object(mlflow, "set_tracking_uri") as mock_set_uri:
            with patch.object(mlflow, "set_experiment") as mock_set_exp:
                bridge = MLflowBridge(
                    temp_repo,
                    tracking_uri="http://localhost:5000",
                    experiment_name="test-experiment",
                )

                mock_set_uri.assert_called_once_with("http://localhost:5000")
                mock_set_exp.assert_called_once_with("test-experiment")
                assert bridge.is_run_active is False

    def test_start_run(self, temp_repo, mock_mlflow_run):
        """Test starting a run."""
        from coral.integrations.mlflow_bridge import MLflowBridge

        with patch.object(mlflow, "set_tracking_uri"):
            with patch.object(mlflow, "set_experiment"):
                with patch.object(mlflow, "start_run", return_value=mock_mlflow_run):
                    with patch.object(mlflow, "log_params"):
                        bridge = MLflowBridge(
                            temp_repo, experiment_name="test-experiment"
                        )
                        run_id = bridge.start_run("test-run", params={"lr": 0.001})

                        assert run_id == "mlflow-run-123"
                        assert bridge.is_run_active is True
                        assert bridge.current_run_id == "mlflow-run-123"

    def test_start_run_with_tags(self, temp_repo, mock_mlflow_run):
        """Test starting a run with tags."""
        from coral.integrations.mlflow_bridge import MLflowBridge

        with patch.object(mlflow, "set_tracking_uri"):
            with patch.object(mlflow, "set_experiment"):
                with patch.object(mlflow, "start_run", return_value=mock_mlflow_run):
                    with patch.object(mlflow, "set_tag") as mock_set_tag:
                        bridge = MLflowBridge(
                            temp_repo, experiment_name="test-experiment"
                        )
                        bridge.start_run("test-run", tags=["experiment", "baseline"])

                        # Verify set_tag was called for each tag
                        assert mock_set_tag.call_count >= 2

    def test_log_metrics(self, temp_repo, mock_mlflow_run):
        """Test logging metrics."""
        from coral.integrations.mlflow_bridge import MLflowBridge

        with patch.object(mlflow, "set_tracking_uri"):
            with patch.object(mlflow, "set_experiment"):
                with patch.object(mlflow, "start_run", return_value=mock_mlflow_run):
                    with patch.object(mlflow, "log_metrics") as mock_log:
                        bridge = MLflowBridge(
                            temp_repo, experiment_name="test-experiment"
                        )
                        bridge.start_run("test-run")
                        bridge.log_metrics({"loss": 0.5, "accuracy": 0.95}, step=100)

                        mock_log.assert_called_once_with(
                            {"loss": 0.5, "accuracy": 0.95}, step=100
                        )

    def test_log_metrics_without_active_run(self, temp_repo):
        """Test logging metrics without active run."""
        from coral.integrations.mlflow_bridge import MLflowBridge

        with patch.object(mlflow, "set_tracking_uri"):
            with patch.object(mlflow, "set_experiment"):
                with patch.object(mlflow, "log_metrics") as mock_log:
                    bridge = MLflowBridge(temp_repo, experiment_name="test-experiment")

                    # Should not raise, just warn
                    bridge.log_metrics({"loss": 0.5})
                    mock_log.assert_not_called()

    def test_log_params(self, temp_repo, mock_mlflow_run):
        """Test logging parameters."""
        from coral.integrations.mlflow_bridge import MLflowBridge

        with patch.object(mlflow, "set_tracking_uri"):
            with patch.object(mlflow, "set_experiment"):
                with patch.object(mlflow, "start_run", return_value=mock_mlflow_run):
                    with patch.object(mlflow, "log_params") as mock_log:
                        bridge = MLflowBridge(
                            temp_repo, experiment_name="test-experiment"
                        )
                        bridge.start_run("test-run")
                        bridge.log_params({"epochs": 10, "batch_size": 32})

                        mock_log.assert_called_with({"epochs": 10, "batch_size": 32})

    def test_log_coral_commit(self, temp_repo, mock_mlflow_run):
        """Test logging a Coral commit."""
        from coral.integrations.mlflow_bridge import MLflowBridge

        with patch.object(mlflow, "set_tracking_uri"):
            with patch.object(mlflow, "set_experiment"):
                with patch.object(mlflow, "start_run", return_value=mock_mlflow_run):
                    with patch.object(mlflow, "log_param") as mock_log:
                        bridge = MLflowBridge(
                            temp_repo, experiment_name="test-experiment"
                        )
                        bridge.start_run("test-run")
                        bridge.log_coral_commit("abc123def456", message="Test commit")

                        # Verify log_param was called for commit hash
                        assert mock_log.called

    def test_end_run(self, temp_repo, mock_mlflow_run):
        """Test ending a run."""
        from coral.integrations.mlflow_bridge import MLflowBridge

        with patch.object(mlflow, "set_tracking_uri"):
            with patch.object(mlflow, "set_experiment"):
                with patch.object(mlflow, "start_run", return_value=mock_mlflow_run):
                    with patch.object(mlflow, "end_run") as mock_end:
                        bridge = MLflowBridge(
                            temp_repo, experiment_name="test-experiment"
                        )
                        bridge.start_run("test-run")
                        bridge.end_run("completed")

                        mock_end.assert_called_once_with(status="FINISHED")
                        assert bridge.is_run_active is False

    def test_end_run_failed(self, temp_repo, mock_mlflow_run):
        """Test ending a failed run."""
        from coral.integrations.mlflow_bridge import MLflowBridge

        with patch.object(mlflow, "set_tracking_uri"):
            with patch.object(mlflow, "set_experiment"):
                with patch.object(mlflow, "start_run", return_value=mock_mlflow_run):
                    with patch.object(mlflow, "end_run") as mock_end:
                        bridge = MLflowBridge(
                            temp_repo, experiment_name="test-experiment"
                        )
                        bridge.start_run("test-run")
                        bridge.end_run("failed")

                        mock_end.assert_called_once_with(status="FAILED")

    def test_end_run_cancelled(self, temp_repo, mock_mlflow_run):
        """Test ending a cancelled run."""
        from coral.integrations.mlflow_bridge import MLflowBridge

        with patch.object(mlflow, "set_tracking_uri"):
            with patch.object(mlflow, "set_experiment"):
                with patch.object(mlflow, "start_run", return_value=mock_mlflow_run):
                    with patch.object(mlflow, "end_run") as mock_end:
                        bridge = MLflowBridge(
                            temp_repo, experiment_name="test-experiment"
                        )
                        bridge.start_run("test-run")
                        bridge.end_run("cancelled")

                        mock_end.assert_called_once_with(status="KILLED")

    def test_set_tags(self, temp_repo, mock_mlflow_run):
        """Test setting tags."""
        from coral.integrations.mlflow_bridge import MLflowBridge

        with patch.object(mlflow, "set_tracking_uri"):
            with patch.object(mlflow, "set_experiment"):
                with patch.object(mlflow, "start_run", return_value=mock_mlflow_run):
                    with patch.object(mlflow, "set_tag") as mock_set_tag:
                        bridge = MLflowBridge(
                            temp_repo, experiment_name="test-experiment"
                        )
                        bridge.start_run("test-run")
                        bridge.set_tags({"model": "bert", "dataset": "squad"})

                        # Verify set_tag was called for each tag
                        assert mock_set_tag.call_count >= 2

    def test_log_artifact(self, temp_repo, mock_mlflow_run):
        """Test logging an artifact."""
        from coral.integrations.mlflow_bridge import MLflowBridge

        with patch.object(mlflow, "set_tracking_uri"):
            with patch.object(mlflow, "set_experiment"):
                with patch.object(mlflow, "start_run", return_value=mock_mlflow_run):
                    with patch.object(mlflow, "log_artifact") as mock_log:
                        bridge = MLflowBridge(
                            temp_repo, experiment_name="test-experiment"
                        )
                        bridge.start_run("test-run")

                        # Create a test file
                        test_file = temp_repo.coral_dir.parent / "test_artifact.txt"
                        test_file.write_text("test content")

                        bridge.log_artifact(str(test_file), artifact_path="artifacts")

                        mock_log.assert_called_once_with(str(test_file), "artifacts")

    def test_get_commit_for_run(self, temp_repo):
        """Test getting commit for a run."""
        from coral.integrations.mlflow_bridge import MLflowBridge

        # Mock get_run response
        mock_run_data = MagicMock()
        mock_run_data.params = {"coral.commit_hash": "abc123def456"}
        mock_run = MagicMock()
        mock_run.data = mock_run_data

        with patch.object(mlflow, "set_tracking_uri"):
            with patch.object(mlflow, "set_experiment"):
                with patch.object(mlflow, "get_run", return_value=mock_run):
                    bridge = MLflowBridge(temp_repo, experiment_name="test-experiment")

                    commit = bridge.get_commit_for_run("mlflow-run-123")

                    assert commit == "abc123def456"

    def test_find_runs_for_commit(self, temp_repo, mock_mlflow_run):
        """Test finding runs for a commit."""
        from coral.integrations.mlflow_bridge import MLflowBridge

        # Mock search_runs response
        mock_run_info = MagicMock()
        mock_run_info.run_id = "mlflow-run-123"
        mock_found_run = MagicMock()
        mock_found_run.info = mock_run_info

        with patch.object(mlflow, "set_tracking_uri"):
            with patch.object(mlflow, "set_experiment"):
                with patch.object(mlflow, "search_runs", return_value=[mock_found_run]):
                    bridge = MLflowBridge(temp_repo, experiment_name="test-experiment")

                    runs = bridge.find_runs_for_commit("abc123def456")

                    assert runs == ["mlflow-run-123"]

    def test_get_run_info(self, temp_repo, mock_mlflow_run):
        """Test getting run info."""
        from coral.integrations.mlflow_bridge import MLflowBridge

        # Mock get_run response
        mock_run_info = MagicMock()
        mock_run_info.run_id = "mlflow-run-123"
        mock_run_info.run_name = "test-run"
        mock_run_info.status = "FINISHED"
        mock_run_info.start_time = 1234567890
        mock_run_info.end_time = 1234567999

        mock_run_data = MagicMock()
        mock_run_data.params = {"lr": "0.001"}
        mock_run_data.metrics = {"loss": 0.5}
        mock_run_data.tags = {"model": "bert"}

        mock_run = MagicMock()
        mock_run.info = mock_run_info
        mock_run.data = mock_run_data

        with patch.object(mlflow, "set_tracking_uri"):
            with patch.object(mlflow, "set_experiment"):
                with patch.object(mlflow, "start_run", return_value=mock_mlflow_run):
                    with patch.object(mlflow, "get_run", return_value=mock_run):
                        bridge = MLflowBridge(
                            temp_repo, experiment_name="test-experiment"
                        )
                        bridge.start_run("test-run")

                        info = bridge.get_run_info()

                        assert info is not None
                        assert info["run_id"] == "mlflow-run-123"
                        assert info["status"] == "FINISHED"

    def test_get_run_info_no_run(self, temp_repo):
        """Test getting run info when no run is active."""
        from coral.integrations.mlflow_bridge import MLflowBridge

        with patch.object(mlflow, "set_tracking_uri"):
            with patch.object(mlflow, "set_experiment"):
                bridge = MLflowBridge(temp_repo, experiment_name="test-experiment")

                info = bridge.get_run_info()

                assert info is None


class TestMLflowBridgeWithoutMLflow:
    """Tests that work without MLflow installed."""

    def test_import_error_without_mlflow(self, temp_repo):
        """Test that MLflowBridge raises ImportError without MLflow."""
        if HAS_MLFLOW:
            pytest.skip("MLflow is installed")

        with patch("coral.integrations.mlflow_bridge.HAS_MLFLOW", False):
            from coral.integrations.mlflow_bridge import MLflowBridge

            with pytest.raises(ImportError, match="MLflow is required"):
                MLflowBridge(temp_repo)


# ============================================================================
# ExperimentBridge Base Tests
# ============================================================================


class TestExperimentBridgeBase:
    """Tests for ExperimentBridge base class."""

    def test_base_class_is_abstract(self):
        """Test ExperimentBridge is an abstract base class."""
        from abc import ABC

        from coral.integrations.experiment_bridge import ExperimentBridge

        # Verify it's an abstract class
        assert issubclass(ExperimentBridge, ABC)

    def test_base_class_cannot_be_instantiated(self, temp_repo):
        """Test that ExperimentBridge cannot be instantiated."""
        from coral.integrations.experiment_bridge import ExperimentBridge

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ExperimentBridge(temp_repo)
