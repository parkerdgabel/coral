"""Tests for model registry functionality."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.registry import ModelPublisher, PublishResult, RegistryType
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
def temp_repo_with_weights(temp_repo):
    """Create a temporary repository with committed weights."""
    weights = {
        "layer1.weight": WeightTensor(
            data=np.random.randn(10, 5).astype(np.float32),
            metadata=WeightMetadata(
                name="layer1.weight",
                shape=(10, 5),
                dtype=np.float32,
            ),
        ),
        "layer1.bias": WeightTensor(
            data=np.random.randn(10).astype(np.float32),
            metadata=WeightMetadata(
                name="layer1.bias",
                shape=(10,),
                dtype=np.float32,
            ),
        ),
    }

    temp_repo.stage_weights(weights)
    temp_repo.commit(message="Initial commit")

    return temp_repo


class TestPublishResult:
    """Test PublishResult class."""

    def test_publish_result_creation(self):
        """Test creating a publish result."""
        result = PublishResult(
            success=True,
            registry=RegistryType.HUGGINGFACE,
            model_name="org/model",
            url="https://huggingface.co/org/model",
        )

        assert result.success
        assert result.registry == RegistryType.HUGGINGFACE
        assert result.model_name == "org/model"
        assert result.url == "https://huggingface.co/org/model"

    def test_publish_result_failure(self):
        """Test failed publish result."""
        result = PublishResult(
            success=False,
            registry=RegistryType.MLFLOW,
            model_name="my-model",
            error="Connection failed",
        )

        assert not result.success
        assert result.error == "Connection failed"

    def test_publish_result_serialization(self):
        """Test publish result serialization."""
        result = PublishResult(
            success=True,
            registry=RegistryType.LOCAL,
            model_name="/path/to/export",
            version="1.0.0",
            metadata={"format": "safetensors"},
        )

        data = result.to_dict()
        assert data["success"]
        assert data["registry"] == "local"
        assert data["model_name"] == "/path/to/export"
        assert data["version"] == "1.0.0"
        assert data["metadata"]["format"] == "safetensors"


class TestModelPublisher:
    """Test ModelPublisher class."""

    def test_publisher_creation(self, temp_repo):
        """Test creating a publisher."""
        publisher = ModelPublisher(temp_repo)
        assert publisher.repo == temp_repo
        assert publisher.registry_dir.exists()

    def test_publish_local_npz(self, temp_repo_with_weights):
        """Test publishing to local directory in npz format."""
        publisher = ModelPublisher(temp_repo_with_weights)

        with tempfile.TemporaryDirectory() as export_dir:
            result = publisher.publish_local(
                output_path=export_dir,
                format="npz",
            )

            assert result.success
            assert result.registry == RegistryType.LOCAL

            # Check files were created
            export_path = Path(export_dir)
            assert (export_path / "model.npz").exists()
            assert (export_path / "coral_metadata.json").exists()

            # Verify weights can be loaded
            loaded = np.load(export_path / "model.npz", allow_pickle=False)
            assert "layer1.weight" in loaded.files
            assert "layer1.bias" in loaded.files

    def test_publish_local_no_metadata(self, temp_repo_with_weights):
        """Test publishing without metadata."""
        publisher = ModelPublisher(temp_repo_with_weights)

        with tempfile.TemporaryDirectory() as export_dir:
            result = publisher.publish_local(
                output_path=export_dir,
                format="npz",
                include_metadata=False,
            )

            assert result.success
            export_path = Path(export_dir)
            assert (export_path / "model.npz").exists()
            assert not (export_path / "coral_metadata.json").exists()

    def test_publish_local_safetensors_missing(self, temp_repo_with_weights):
        """Test that safetensors format fails gracefully if not installed."""
        publisher = ModelPublisher(temp_repo_with_weights)

        with tempfile.TemporaryDirectory() as export_dir:
            result = publisher.publish_local(
                output_path=export_dir,
                format="safetensors",
            )

            # May succeed if safetensors is installed, may fail if not
            # Just verify we get a result either way
            assert isinstance(result, PublishResult)

    def test_publish_local_invalid_format(self, temp_repo_with_weights):
        """Test publishing with invalid format."""
        publisher = ModelPublisher(temp_repo_with_weights)

        with tempfile.TemporaryDirectory() as export_dir:
            result = publisher.publish_local(
                output_path=export_dir,
                format="invalid",
            )

            assert not result.success
            assert "Unsupported format" in result.error

    def test_publish_huggingface_missing_deps(self, temp_repo_with_weights):
        """Test that HuggingFace publish fails gracefully without deps."""
        publisher = ModelPublisher(temp_repo_with_weights)

        result = publisher.publish_huggingface(
            repo_id="test/model",
        )

        # Should fail with import error if deps not installed
        # or succeed if they are (but we don't have HF credentials)
        assert isinstance(result, PublishResult)

    def test_publish_mlflow_missing_deps(self, temp_repo_with_weights):
        """Test that MLflow publish fails gracefully without deps."""
        publisher = ModelPublisher(temp_repo_with_weights)

        result = publisher.publish_mlflow(
            model_name="test-model",
        )

        # Should fail with import error or other error
        assert isinstance(result, PublishResult)

    def test_publish_history(self, temp_repo_with_weights):
        """Test publish history tracking."""
        publisher = ModelPublisher(temp_repo_with_weights)

        with tempfile.TemporaryDirectory() as export_dir1:
            publisher.publish_local(export_dir1, format="npz")

        with tempfile.TemporaryDirectory() as export_dir2:
            publisher.publish_local(export_dir2, format="npz")

        history = publisher.get_history()
        assert len(history) == 2

        # Check history filtering
        local_only = publisher.get_history(registry=RegistryType.LOCAL)
        assert len(local_only) == 2

        success_only = publisher.get_history(success_only=True)
        assert len(success_only) == 2

    def test_get_latest(self, temp_repo_with_weights):
        """Test getting latest publish for a model."""
        publisher = ModelPublisher(temp_repo_with_weights)

        with tempfile.TemporaryDirectory() as export_dir:
            result = publisher.publish_local(export_dir, format="npz")

            latest = publisher.get_latest(result.model_name)
            assert latest is not None
            assert latest.model_name == result.model_name

            # Non-existent model
            not_found = publisher.get_latest("nonexistent")
            assert not_found is None

    def test_publish_no_weights(self, temp_repo):
        """Test publishing when repo has no weights."""
        publisher = ModelPublisher(temp_repo)

        with tempfile.TemporaryDirectory() as export_dir:
            result = publisher.publish_local(export_dir, format="npz")

            assert not result.success
            assert "No weights found" in result.error

    def test_model_card_generation(self, temp_repo_with_weights):
        """Test model card generation."""
        publisher = ModelPublisher(temp_repo_with_weights)

        card = publisher._generate_model_card(
            model_name="my-model",
            description="A fine-tuned model",
            base_model="bert-base-uncased",
            metrics={"accuracy": 0.95, "f1": 0.92},
            tags=["nlp", "transformer"],
        )

        assert "my-model" in card
        assert "A fine-tuned model" in card
        assert "bert-base-uncased" in card
        assert "accuracy" in card
        assert "0.95" in card
        assert "nlp" in card

    def test_history_persistence(self, temp_repo_with_weights):
        """Test that history persists across publisher instances."""
        publisher1 = ModelPublisher(temp_repo_with_weights)

        with tempfile.TemporaryDirectory() as export_dir:
            publisher1.publish_local(export_dir, format="npz")

        # Create new publisher instance
        publisher2 = ModelPublisher(temp_repo_with_weights)
        history = publisher2.get_history()

        assert len(history) == 1


class TestRegistryType:
    """Test RegistryType enum."""

    def test_registry_types(self):
        """Test registry type values."""
        assert RegistryType.HUGGINGFACE.value == "huggingface"
        assert RegistryType.MLFLOW.value == "mlflow"
        assert RegistryType.LOCAL.value == "local"
