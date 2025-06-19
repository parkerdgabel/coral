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
        """Test CLI add command with torch file support."""
        cli = CoralCLI()

        args = argparse.Namespace(weights=["model.pth"])
        repo_path = Path(".")

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            # Create mock path
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path.suffix = ".pth"

            with patch("coral.cli.main.Path", return_value=mock_path):
                # Mock torch
                mock_torch = Mock()
                mock_state_dict = {
                    "layer.weight": Mock(
                        shape=(10, 20), dtype=Mock(__str__=lambda self: "torch.float32")
                    ),
                }
                for tensor in mock_state_dict.values():
                    tensor.numpy = Mock(
                        return_value=np.random.randn(*tensor.shape).astype(np.float32)
                    )

                mock_torch.load.return_value = mock_state_dict

                with patch.dict("sys.modules", {"torch": mock_torch}):
                    with patch("builtins.print") as _mock_print:
                        # Should work with torch support
                        result = cli._cmd_add(args, repo_path)
                        assert result == 0
                        mock_repo.stage_weights.assert_called()

    def test_weight_store_abstract_methods(self):
        """Test WeightStore abstract methods raise NotImplementedError."""

        # Create a minimal concrete implementation for testing
        class MinimalStore(WeightStore):
            pass

        store = MinimalStore()

        # Test all abstract methods
        with pytest.raises(NotImplementedError):
            store.store("hash", Mock())

        with pytest.raises(NotImplementedError):
            store.load("hash")

        with pytest.raises(NotImplementedError):
            store.exists("hash")

        with pytest.raises(NotImplementedError):
            store.delete("hash")

        with pytest.raises(NotImplementedError):
            store.list_weights()

        with pytest.raises(NotImplementedError):
            store.get_metadata("hash")

        with pytest.raises(NotImplementedError):
            store.get_storage_info()

    def test_deduplication_stats_properties(self):
        """Test DeduplicationStats properties."""
        stats = DeduplicationStats(
            total_weights=100,
            unique_weights=60,
            duplicate_weights=30,
            similar_weights=10,
            bytes_saved=1024 * 1024 * 50,
            deduplication_time=1.5,
        )

        # Test properties
        assert stats.compression_ratio == pytest.approx(100 / 60, rel=1e-3)
        assert stats.space_savings_percent == pytest.approx(40.0, rel=1e-3)

        # Test edge case with no unique weights
        stats_edge = DeduplicationStats(
            total_weights=0,
            unique_weights=0,
            duplicate_weights=0,
            similar_weights=0,
            bytes_saved=0,
            deduplication_time=0,
        )
        assert stats_edge.compression_ratio == 1.0
        assert stats_edge.space_savings_percent == 0.0

    def test_branch_manager_edge_cases(self):
        """Test BranchManager edge cases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            refs_dir = Path(tmpdir) / "refs" / "heads"
            refs_dir.mkdir(parents=True)

            manager = BranchManager(refs_dir)

            # Test getting non-existent branch
            assert manager.get_branch("non-existent") is None

            # Test updating non-existent branch creates it
            manager.update_branch("new-branch", "commit123")
            branch = manager.get_branch("new-branch")
            assert branch is not None
            assert branch.commit_hash == "commit123"

            # Test deleting non-existent branch (should not raise)
            manager.delete_branch("another-non-existent")

            # Test listing with subdirectories
            subdir = refs_dir / "feature"
            subdir.mkdir()
            (subdir / "test").write_text("subcommit")

            branches = manager.list_branches()
            branch_names = [b.name for b in branches]
            assert "feature/test" in branch_names

    def test_visualization_edge_cases(self):
        """Test visualization functions with edge cases."""
        # Test with matplotlib not available
        with patch("coral.utils.visualization.plt", None):
            # Should not raise, just return early
            plot_weight_distribution(Mock())

            stats = Mock()
            plot_deduplication_stats(stats)

        # Test with matplotlib available but headless
        mock_plt = Mock()
        with patch("coral.utils.visualization.plt", mock_plt):
            # Create mock weight
            weight = Mock()
            weight.metadata = Mock(name="test")
            weight.data = np.array([1, 2, 3])

            plot_weight_distribution(weight, show=False, save_path="test.png")
            mock_plt.savefig.assert_called_once_with("test.png")
            mock_plt.close.assert_called_once()

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
            save_training_state=True,
        )

        assert config.save_every_n_epochs == 5
        assert config.save_every_n_steps == 1000
        assert config.save_on_best_metric == "accuracy"
        assert config.minimize_metric is False
        assert config.keep_last_n_checkpoints == 3
        assert config.save_optimizer_state is True
        assert config.save_training_state is True

    def test_training_state_comprehensive(self):
        """Test TrainingState comprehensively."""
        state = TrainingState(
            epoch=10,
            global_step=1000,
            learning_rate=0.001,
            loss=0.5,
            metrics={"accuracy": 0.95, "precision": 0.93, "recall": 0.94, "f1": 0.935},
            optimizer_state={"momentum": 0.9, "betas": [0.9, 0.999]},
            custom_data={"experiment": "baseline", "config": {"batch_size": 32}},
        )

        # Test dict conversion
        state_dict = state.to_dict()
        assert state_dict["epoch"] == 10
        assert state_dict["metrics"]["accuracy"] == 0.95
        assert state_dict["optimizer_state"]["momentum"] == 0.9
        assert state_dict["custom_data"]["experiment"] == "baseline"

        # Test from_dict
        restored = TrainingState.from_dict(state_dict)
        assert restored.epoch == state.epoch
        assert restored.metrics["f1"] == state.metrics["f1"]
        assert restored.custom_data["config"]["batch_size"] == 32

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
                store.store(hash_key, weight)

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
