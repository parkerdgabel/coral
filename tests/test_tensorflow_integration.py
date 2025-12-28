"""Tests for TensorFlow/Keras integration."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from coral.core.weight_tensor import WeightTensor
from coral.version_control.repository import Repository

# Mock TensorFlow since it's optional
tf = Mock()
keras = Mock()


class TestTensorFlowIntegration:
    """Tests for TensorFlowIntegration class."""

    def test_model_to_weights(self):
        """Test converting Keras model to weight tensors."""
        # Create mock weights
        mock_weight1 = Mock()
        mock_weight1.name = "dense/kernel:0"
        mock_weight1.numpy.return_value = np.random.randn(10, 20).astype(np.float32)

        mock_weight2 = Mock()
        mock_weight2.name = "dense/bias:0"
        mock_weight2.numpy.return_value = np.random.randn(20).astype(np.float32)

        # Create mock model
        model = Mock()
        model.weights = [mock_weight1, mock_weight2]
        model.layers = []
        model.__class__.__name__ = "Sequential"

        with patch("coral.integrations.tensorflow.TF_AVAILABLE", True):
            from coral.integrations.tensorflow import TensorFlowIntegration

            weight_tensors = TensorFlowIntegration.model_to_weights(model)

        assert len(weight_tensors) == 2
        assert "dense/kernel:0" in weight_tensors
        assert "dense/bias:0" in weight_tensors

        # Check weight tensor properties
        assert isinstance(weight_tensors["dense/kernel:0"], WeightTensor)
        assert weight_tensors["dense/kernel:0"].shape == (10, 20)

    def test_weights_to_model(self):
        """Test loading weight tensors into Keras model."""
        # Create mock weights
        mock_weight1 = Mock()
        mock_weight1.name = "dense/kernel:0"
        mock_weight1.numpy.return_value = np.zeros((10, 20), dtype=np.float32)

        mock_weight2 = Mock()
        mock_weight2.name = "dense/bias:0"
        mock_weight2.numpy.return_value = np.zeros(20, dtype=np.float32)

        # Create mock model
        model = Mock()
        model.weights = [mock_weight1, mock_weight2]

        # Create weight tensors
        weight_tensors = {
            "dense/kernel:0": WeightTensor(
                data=np.random.randn(10, 20).astype(np.float32),
                metadata={"name": "dense/kernel:0"},
            ),
            "dense/bias:0": WeightTensor(
                data=np.random.randn(20).astype(np.float32),
                metadata={"name": "dense/bias:0"},
            ),
        }

        with patch("coral.integrations.tensorflow.TF_AVAILABLE", True):
            from coral.integrations.tensorflow import TensorFlowIntegration

            TensorFlowIntegration.weights_to_model(weight_tensors, model)

        # Should have called set_weights
        model.set_weights.assert_called_once()
        call_args = model.set_weights.call_args[0][0]
        assert len(call_args) == 2

    def test_weights_to_model_missing_weights(self, caplog):
        """Test loading with missing weights logs warning."""
        # Create mock weight in model but not in saved weights
        mock_weight = Mock()
        mock_weight.name = "dense/kernel:0"
        mock_weight.numpy.return_value = np.zeros((10, 20), dtype=np.float32)

        model = Mock()
        model.weights = [mock_weight]

        # Empty weight tensors - nothing to load
        weight_tensors = {}

        with patch("coral.integrations.tensorflow.TF_AVAILABLE", True):
            from coral.integrations.tensorflow import TensorFlowIntegration

            TensorFlowIntegration.weights_to_model(weight_tensors, model)

        # Should still call set_weights (with original values)
        model.set_weights.assert_called_once()

    def test_get_trainable_weights(self):
        """Test getting only trainable weights."""
        mock_weight = Mock()
        mock_weight.name = "dense/kernel:0"
        mock_weight.numpy.return_value = np.random.randn(5, 10).astype(np.float32)

        model = Mock()
        model.trainable_weights = [mock_weight]
        model.layers = []
        model.__class__.__name__ = "Sequential"

        with patch("coral.integrations.tensorflow.TF_AVAILABLE", True):
            from coral.integrations.tensorflow import TensorFlowIntegration

            weights = TensorFlowIntegration.get_trainable_weights(model)

        assert len(weights) == 1
        assert "dense/kernel:0" in weights

    def test_get_non_trainable_weights(self):
        """Test getting only non-trainable weights."""
        mock_weight = Mock()
        mock_weight.name = "batch_norm/moving_mean:0"
        mock_weight.numpy.return_value = np.zeros(10, dtype=np.float32)

        model = Mock()
        model.non_trainable_weights = [mock_weight]
        model.layers = []
        model.__class__.__name__ = "Sequential"

        with patch("coral.integrations.tensorflow.TF_AVAILABLE", True):
            from coral.integrations.tensorflow import TensorFlowIntegration

            weights = TensorFlowIntegration.get_non_trainable_weights(model)

        assert len(weights) == 1
        assert "batch_norm/moving_mean:0" in weights

    def test_get_optimizer_weights(self):
        """Test getting optimizer weights."""
        mock_opt_weight = Mock()
        mock_opt_weight.name = "Adam/m/dense/kernel:0"
        mock_opt_weight.numpy.return_value = np.zeros((5, 10), dtype=np.float32)

        optimizer = Mock()
        optimizer.weights = [mock_opt_weight]

        with patch("coral.integrations.tensorflow.TF_AVAILABLE", True):
            from coral.integrations.tensorflow import TensorFlowIntegration

            weights = TensorFlowIntegration.get_optimizer_weights(optimizer)

        assert len(weights) == 1


class TestTensorFlowSaveLoad:
    """Tests for save and load convenience functions."""

    def test_save_model(self, tmp_path):
        """Test saving a model to repository."""
        # Create mock model
        mock_weight = Mock()
        mock_weight.name = "dense/kernel:0"
        mock_weight.numpy.return_value = np.random.randn(5, 10).astype(np.float32)

        model = Mock()
        model.weights = [mock_weight]
        model.layers = []
        model.__class__.__name__ = "Sequential"

        with patch("coral.integrations.tensorflow.TF_AVAILABLE", True):
            from coral.integrations.tensorflow import save

            # Initialize repository
            repo = Repository(tmp_path / "test_repo", init=True)

            result = save(model, repo, "Initial checkpoint")

        assert "commit_hash" in result
        assert result["weights_saved"] == 1

    def test_save_with_branch(self, tmp_path):
        """Test saving to a new branch."""
        mock_weight = Mock()
        mock_weight.name = "dense/kernel:0"
        mock_weight.numpy.return_value = np.random.randn(5, 10).astype(np.float32)

        model = Mock()
        model.weights = [mock_weight]
        model.layers = []
        model.__class__.__name__ = "Sequential"

        with patch("coral.integrations.tensorflow.TF_AVAILABLE", True):
            from coral.integrations.tensorflow import save

            repo = Repository(tmp_path / "test_repo", init=True)

            # First commit on main
            save(model, repo, "Initial")

            # Create branch and save
            result = save(
                model,
                repo,
                "Branch checkpoint",
                branch="experiment",
                create_branch=True,
            )

        assert result["branch"] == "experiment"

    def test_save_with_tag(self, tmp_path):
        """Test saving with a version tag."""
        mock_weight = Mock()
        mock_weight.name = "dense/kernel:0"
        mock_weight.numpy.return_value = np.random.randn(5, 10).astype(np.float32)

        model = Mock()
        model.weights = [mock_weight]
        model.layers = []
        model.__class__.__name__ = "Sequential"

        with patch("coral.integrations.tensorflow.TF_AVAILABLE", True):
            from coral.integrations.tensorflow import save

            repo = Repository(tmp_path / "test_repo", init=True)
            result = save(model, repo, "Release v1.0", tag="v1.0")

        assert result.get("tag") == "v1.0"

    def test_load_model(self, tmp_path):
        """Test loading weights into a model."""
        # Create and save mock model
        mock_weight_save = Mock()
        mock_weight_save.name = "dense/kernel:0"
        saved_data = np.random.randn(5, 10).astype(np.float32)
        mock_weight_save.numpy.return_value = saved_data

        model_save = Mock()
        model_save.weights = [mock_weight_save]
        model_save.layers = []
        model_save.__class__.__name__ = "Sequential"

        with patch("coral.integrations.tensorflow.TF_AVAILABLE", True):
            from coral.integrations.tensorflow import load, save

            repo = Repository(tmp_path / "test_repo", init=True)
            save(model_save, repo, "Initial")

            # Create model to load into
            mock_weight_load = Mock()
            mock_weight_load.name = "dense/kernel:0"
            mock_weight_load.numpy.return_value = np.zeros((5, 10), dtype=np.float32)

            model_load = Mock()
            model_load.weights = [mock_weight_load]

            result = load(model_load, repo)

        assert result["matched"] == 1
        assert "dense/kernel:0" in result["loaded"]
        model_load.set_weights.assert_called_once()

    def test_load_strict_mode(self, tmp_path):
        """Test strict mode raises error for missing weights."""
        mock_weight = Mock()
        mock_weight.name = "dense/kernel:0"
        mock_weight.numpy.return_value = np.random.randn(5, 10).astype(np.float32)

        model_save = Mock()
        model_save.weights = [mock_weight]
        model_save.layers = []
        model_save.__class__.__name__ = "Sequential"

        with patch("coral.integrations.tensorflow.TF_AVAILABLE", True):
            from coral.integrations.tensorflow import load, save

            repo = Repository(tmp_path / "test_repo", init=True)
            save(model_save, repo, "Initial")

            # Model with different weight name
            mock_weight_load = Mock()
            mock_weight_load.name = "different/weight:0"
            mock_weight_load.numpy.return_value = np.zeros((5, 10), dtype=np.float32)

            model_load = Mock()
            model_load.weights = [mock_weight_load]

            with pytest.raises(ValueError, match="Missing weights"):
                load(model_load, repo, strict=True)


class TestCompareModelWeights:
    """Tests for compare_model_weights function."""

    def test_compare_identical_models(self):
        """Test comparing identical models."""
        data = np.random.randn(5, 10).astype(np.float32)

        mock_weight1 = Mock()
        mock_weight1.name = "dense/kernel:0"
        mock_weight1.numpy.return_value = data.copy()

        mock_weight2 = Mock()
        mock_weight2.name = "dense/kernel:0"
        mock_weight2.numpy.return_value = data.copy()

        model1 = Mock()
        model1.weights = [mock_weight1]
        model1.layers = []
        model1.__class__.__name__ = "Sequential"

        model2 = Mock()
        model2.weights = [mock_weight2]
        model2.layers = []
        model2.__class__.__name__ = "Sequential"

        with patch("coral.integrations.tensorflow.TF_AVAILABLE", True):
            from coral.integrations.tensorflow import compare_model_weights

            result = compare_model_weights(model1, model2)

        assert len(result["identical"]) == 1
        assert len(result["different"]) == 0

    def test_compare_different_models(self):
        """Test comparing models with different weights."""
        mock_weight1 = Mock()
        mock_weight1.name = "dense/kernel:0"
        mock_weight1.numpy.return_value = np.zeros((5, 10), dtype=np.float32)

        mock_weight2 = Mock()
        mock_weight2.name = "dense/kernel:0"
        mock_weight2.numpy.return_value = np.ones((5, 10), dtype=np.float32)

        model1 = Mock()
        model1.weights = [mock_weight1]
        model1.layers = []
        model1.__class__.__name__ = "Sequential"

        model2 = Mock()
        model2.weights = [mock_weight2]
        model2.layers = []
        model2.__class__.__name__ = "Sequential"

        with patch("coral.integrations.tensorflow.TF_AVAILABLE", True):
            from coral.integrations.tensorflow import compare_model_weights

            result = compare_model_weights(model1, model2)

        assert len(result["identical"]) == 0
        assert len(result["different"]) == 1
        assert result["different"][0]["reason"] == "value_difference"

    def test_compare_shape_mismatch(self):
        """Test comparing models with shape mismatch."""
        mock_weight1 = Mock()
        mock_weight1.name = "dense/kernel:0"
        mock_weight1.numpy.return_value = np.zeros((5, 10), dtype=np.float32)

        mock_weight2 = Mock()
        mock_weight2.name = "dense/kernel:0"
        mock_weight2.numpy.return_value = np.zeros((10, 20), dtype=np.float32)

        model1 = Mock()
        model1.weights = [mock_weight1]
        model1.layers = []
        model1.__class__.__name__ = "Sequential"

        model2 = Mock()
        model2.weights = [mock_weight2]
        model2.layers = []
        model2.__class__.__name__ = "Sequential"

        with patch("coral.integrations.tensorflow.TF_AVAILABLE", True):
            from coral.integrations.tensorflow import compare_model_weights

            result = compare_model_weights(model1, model2)

        assert len(result["different"]) == 1
        assert result["different"][0]["reason"] == "shape_mismatch"


class TestTensorFlowNotInstalled:
    """Tests for behavior when TensorFlow is not installed."""

    def test_import_error_on_save(self, tmp_path):
        """Test that save raises ImportError when TF not installed."""
        with patch("coral.integrations.tensorflow.TF_AVAILABLE", False):
            from coral.integrations.tensorflow import save

            repo = Repository(tmp_path / "test_repo", init=True)

            with pytest.raises(ImportError, match="TensorFlow is not installed"):
                save(Mock(), repo, "test")

    def test_import_error_on_load(self, tmp_path):
        """Test that load raises ImportError when TF not installed."""
        with patch("coral.integrations.tensorflow.TF_AVAILABLE", False):
            from coral.integrations.tensorflow import load

            repo = Repository(tmp_path / "test_repo", init=True)

            with pytest.raises(ImportError, match="TensorFlow is not installed"):
                load(Mock(), repo)

    def test_import_error_on_model_to_weights(self):
        """Test that model_to_weights raises ImportError when TF not installed."""
        with patch("coral.integrations.tensorflow.TF_AVAILABLE", False):
            from coral.integrations.tensorflow import TensorFlowIntegration

            with pytest.raises(ImportError, match="TensorFlow is not installed"):
                TensorFlowIntegration.model_to_weights(Mock())
