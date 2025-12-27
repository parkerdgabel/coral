"""Final test file to ensure 80% coverage."""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Import all modules to boost coverage
from coral.cli.main import CoralCLI
from coral.core.deduplicator import DeduplicationStats
from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.integrations.pytorch import PyTorchIntegration
from coral.storage.hdf5_store import HDF5Store
from coral.storage.weight_store import WeightStore
from coral.training.checkpoint_manager import CheckpointConfig
from coral.training.training_state import TrainingState
from coral.utils.visualization import plot_deduplication_stats, plot_weight_distribution
from coral.version_control.branch import BranchManager


class TestCoverageFinal80Percent:
    """Final tests to reach 80% coverage."""

    def test_cli_add_torch_support(self):
        """Test CLI add command with unsupported file format."""
        cli = CoralCLI()

        args = argparse.Namespace(weights=["model.pth"])
        repo_path = Path(".")

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            # Create a real temporary file with .pth extension
            with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                # Update args with the real file path
                args = argparse.Namespace(weights=[tmp_path])

                with patch("builtins.print") as mock_print:
                    # Should return error for unsupported format
                    result = cli._cmd_add(args, repo_path)
                    assert result == 1  # Error code for unsupported format

            finally:
                Path(tmp_path).unlink(missing_ok=True)

    def test_weight_store_abstract_methods(self):
        """Test WeightStore abstract methods cannot be instantiated."""

        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            WeightStore()

    def test_deduplication_stats_properties(self):
        """Test DeduplicationStats attributes."""
        stats = DeduplicationStats(
            total_weights=100,
            unique_weights=60,
            duplicate_weights=30,
            similar_weights=10,
            bytes_saved=1024 * 1024 * 50,
        )

        # Test attributes
        assert stats.total_weights == 100
        assert stats.unique_weights == 60
        assert stats.duplicate_weights == 30
        assert stats.similar_weights == 10

        # Test update method
        stats.update(original_bytes=1000, deduplicated_bytes=600)
        assert stats.bytes_saved == 400
        assert stats.compression_ratio == pytest.approx(0.4, rel=1e-3)

        # Test edge case with zero original bytes
        stats_edge = DeduplicationStats()
        stats_edge.update(original_bytes=0, deduplicated_bytes=0)
        assert stats_edge.compression_ratio == 0.0

    def test_branch_manager_edge_cases(self):
        """Test BranchManager edge cases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            coral_dir = repo_path / ".coral"
            refs_dir = coral_dir / "refs" / "heads"
            refs_dir.mkdir(parents=True)

            # Create HEAD file
            (coral_dir / "HEAD").write_text("ref: refs/heads/main")

            manager = BranchManager(repo_path)

            # Test getting non-existent branch
            assert manager.get_branch("non-existent") is None

            # Test creating a branch
            branch = manager.create_branch("new-branch", "commit123")
            assert branch is not None
            assert branch.commit_hash == "commit123"

            # Test getting the created branch
            retrieved = manager.get_branch("new-branch")
            assert retrieved.commit_hash == "commit123"

            # Test listing branches
            branches = manager.list_branches()
            assert len(branches) >= 1
            branch_names = [b.name for b in branches]
            assert "new-branch" in branch_names

    def test_visualization_edge_cases(self):
        """Test visualization functions."""
        # Create test weight
        weight = WeightTensor(
            data=np.random.randn(10, 10).astype(np.float32),
            metadata=WeightMetadata(name="test", shape=(10, 10), dtype=np.float32),
        )

        # Test plot_weight_distribution
        result = plot_weight_distribution([weight])
        assert isinstance(result, dict)
        assert "test" in result
        assert "histogram" in result["test"]
        assert "mean" in result["test"]
        assert "std" in result["test"]

        # Test plot_deduplication_stats
        stats = DeduplicationStats(
            total_weights=100,
            unique_weights=60,
            duplicate_weights=30,
            similar_weights=10,
        )
        stats_viz = plot_deduplication_stats(stats)
        assert isinstance(stats_viz, dict)
        assert "weight_counts" in stats_viz
        assert "compression" in stats_viz

    def test_pytorch_integration_without_torch(self):
        """Test PyTorchIntegration when torch is not available."""
        with patch("coral.integrations.pytorch.TORCH_AVAILABLE", False):
            integration = PyTorchIntegration()

            with pytest.raises(ImportError, match="PyTorch is not installed"):
                integration.model_to_weights(None)

            with pytest.raises(ImportError, match="PyTorch is not installed"):
                integration.weights_to_model(None, {})

    def test_checkpoint_config_validation(self):
        """Test CheckpointConfig with various settings."""
        # Test with all options
        config = CheckpointConfig(
            save_every_n_epochs=5,
            save_every_n_steps=1000,
            save_on_best_metric="accuracy",
            minimize_metric=False,
            keep_last_n_checkpoints=3,
            save_optimizer_state=True,
            save_scheduler_state=False,
        )

        assert config.save_every_n_epochs == 5
        assert config.save_every_n_steps == 1000
        assert config.save_on_best_metric == "accuracy"
        assert config.minimize_metric is False
        assert config.keep_last_n_checkpoints == 3
        assert config.save_optimizer_state is True
        assert config.save_scheduler_state is False

    def test_training_state_comprehensive(self):
        """Test TrainingState comprehensively."""
        state = TrainingState(
            epoch=10,
            global_step=1000,
            learning_rate=0.001,
            loss=0.5,
            metrics={"accuracy": 0.95, "precision": 0.93, "recall": 0.94, "f1": 0.935},
            optimizer_state={"momentum": 0.9, "betas": [0.9, 0.999]},
            batch_size=32,
            experiment_name="baseline",
            model_name="test-model",
        )

        # Test dict conversion
        state_dict = state.to_dict()
        assert state_dict["epoch"] == 10
        assert state_dict["metrics"]["accuracy"] == 0.95
        assert state_dict["optimizer_state"]["momentum"] == 0.9
        assert state_dict["experiment_name"] == "baseline"
        assert state_dict["batch_size"] == 32

        # Test from_dict
        restored = TrainingState.from_dict(state_dict)
        assert restored.epoch == state.epoch
        assert restored.metrics["f1"] == state.metrics["f1"]
        assert restored.batch_size == 32
        assert restored.model_name == "test-model"

    def test_hdf5_store_batch_operations(self):
        """Test HDF5Store batch operations."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            store_path = tmp.name

        try:
            store = HDF5Store(store_path)

            # Create and store multiple weights
            weights = {}
            hashes = []
            for i in range(5):
                data = np.random.randn(10, 10).astype(np.float32)
                weight = WeightTensor(
                    data=data,
                    metadata=WeightMetadata(
                        name=f"weight_{i}", shape=data.shape, dtype=data.dtype
                    ),
                )
                hash_key = weight.compute_hash()
                hashes.append(hash_key)
                weights[hash_key] = weight
                store.store(weight, hash_key)

            # Test load_batch
            loaded = store.load_batch(hashes)
            assert len(loaded) == 5
            for hash_key, weight in loaded.items():
                assert hash_key in weights
                np.testing.assert_array_equal(weight.data, weights[hash_key].data)

            # Test load_batch with missing keys
            mixed_hashes = hashes[:3] + ["missing1", "missing2"]
            loaded_mixed = store.load_batch(mixed_hashes)
            assert len(loaded_mixed) == 3
            assert "missing1" not in loaded_mixed
            assert "missing2" not in loaded_mixed

            store.close()

        finally:
            Path(store_path).unlink(missing_ok=True)
