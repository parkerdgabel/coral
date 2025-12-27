"""Comprehensive imports and basic tests for 80% coverage."""

import tempfile
from pathlib import Path

import numpy as np

# Import all modules comprehensively
from coral import __version__
from coral.cli.main import CoralCLI
from coral.compression.pruning import Pruner
from coral.compression.quantization import Quantizer
from coral.core.deduplicator import Deduplicator
from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.delta.delta_encoder import DeltaEncoder, DeltaType
from coral.storage.hdf5_store import HDF5Store
from coral.training.checkpoint_manager import CheckpointConfig
from coral.training.training_state import TrainingState


class TestComprehensiveImports:
    """Comprehensive tests through imports and basic usage."""

    def test_version_exists(self):
        """Test version is defined."""
        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_weight_tensor_comprehensive(self):
        """Test WeightTensor comprehensively."""
        # Create with all metadata fields
        data = np.random.randn(10, 20).astype(np.float32)
        metadata = WeightMetadata(
            name="layer.weight",
            shape=data.shape,
            dtype=data.dtype,
            layer_type="Linear",
            model_name="TestModel",
            compression_info={"method": "none"},
        )

        weight = WeightTensor(data=data, metadata=metadata)

        # Test all properties
        assert weight.shape == (10, 20)
        assert weight.dtype == np.float32
        assert weight.size == 200
        assert weight.nbytes == 800

        # Test methods
        hash1 = weight.compute_hash()
        hash2 = weight.compute_hash()
        assert hash1 == hash2
        assert len(hash1) > 0

        # Test string representation
        str_repr = str(weight)
        assert "layer.weight" in str_repr

    def test_deduplicator_comprehensive(self):
        """Test Deduplicator comprehensively."""
        dedup = Deduplicator(similarity_threshold=0.95)

        # Create test weights
        weights = {}
        for i in range(5):
            data = np.random.randn(10, 10).astype(np.float32)
            if i > 2:
                # Make similar to first weight
                data = (
                    weights[list(weights.keys())[0]].data
                    + np.random.randn(10, 10).astype(np.float32) * 0.01
                )

            weight = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"w{i}", shape=data.shape, dtype=data.dtype
                ),
            )
            weights[f"w{i}"] = weight

        # Add weights to deduplicator
        for name, weight in weights.items():
            dedup.add_weight(weight, name)

        # Get stats
        stats = dedup.compute_stats()

        # Stats should be valid
        assert stats.total_weights == 5
        assert stats.unique_weights >= 1
        assert stats.duplicate_weights >= 0
        assert stats.similar_weights >= 0

    def test_delta_encoder_comprehensive(self):
        """Test DeltaEncoder comprehensively."""
        # Create reference and target
        ref_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        target_data = np.array([1.1, 2.0, 3.2, 4.0, 5.3], dtype=np.float32)

        ref = WeightTensor(
            data=ref_data,
            metadata=WeightMetadata(
                name="ref", shape=ref_data.shape, dtype=ref_data.dtype
            ),
        )

        target = WeightTensor(
            data=target_data,
            metadata=WeightMetadata(
                name="target", shape=target_data.shape, dtype=target_data.dtype
            ),
        )

        # Test encoding strategies
        from coral.delta.delta_encoder import DeltaConfig

        for strategy in [DeltaType.SPARSE, DeltaType.FLOAT32_RAW]:
            config = DeltaConfig(delta_type=strategy)
            encoder = DeltaEncoder(config)

            # Test can encode check
            can_encode = encoder.can_encode_as_delta(target, ref)
            assert isinstance(can_encode, bool)

            # Test encoding
            delta = encoder.encode_delta(target, ref)
            assert delta is not None
            assert delta.delta_type == strategy
            assert delta.reference_hash == ref.compute_hash()

            # Test decoding
            decoded = encoder.decode_delta(delta, ref)
            assert decoded is not None
            # For SPARSE, allow some tolerance due to thresholding
            if strategy == DeltaType.SPARSE:
                np.testing.assert_allclose(
                    decoded.data, target_data, rtol=1e-3, atol=1e-6
                )
            else:
                np.testing.assert_allclose(decoded.data, target_data, rtol=1e-5)

    def test_hdf5_store_comprehensive(self):
        """Test HDF5Store comprehensively."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            store_path = tmp.name

        try:
            # Test with different compression options
            for compression in ["gzip", "lzf", None]:
                if Path(store_path).exists():
                    Path(store_path).unlink()

                store = HDF5Store(store_path, compression=compression)

                # Store weights
                weight = WeightTensor(
                    data=np.random.randn(100, 100).astype(np.float32),
                    metadata=WeightMetadata(
                        name="large_weight",
                        shape=(100, 100),
                        dtype=np.float32,
                        model_name="TestModel",
                    ),
                )

                hash_key = weight.compute_hash()
                store.store(weight, hash_key)

                # Test all methods
                assert store.exists(hash_key)
                assert hash_key in store.list_weights()

                loaded = store.load(hash_key)
                assert loaded is not None
                np.testing.assert_array_equal(loaded.data, weight.data)

                meta = store.get_metadata(hash_key)
                assert meta is not None
                assert meta.name == "large_weight"

                info = store.get_storage_info()
                assert "compression" in info
                assert info["compression"] == compression

                store.close()

        finally:
            Path(store_path).unlink(missing_ok=True)

    def test_compression_modules(self):
        """Test compression modules."""
        # Test Quantizer
        quantizer = Quantizer()
        data = np.random.randn(10, 10).astype(np.float32)
        weight = WeightTensor(
            data=data,
            metadata=WeightMetadata(name="test", shape=data.shape, dtype=data.dtype),
        )

        # Quantize
        quantized, params = quantizer.quantize_uniform(weight, bits=8)
        assert quantized is not None
        assert params is not None
        assert "scale" in params
        assert "zero_point" in params

        # Dequantize
        dequantized = quantizer.dequantize(quantized, params)
        assert dequantized is not None

        # Test Pruner
        pruner = Pruner()

        # Magnitude pruning
        pruned, pruning_info = pruner.prune_magnitude(weight, sparsity=0.5)
        assert pruned is not None
        assert pruning_info is not None
        assert "mask" in pruning_info
        mask = pruning_info["mask"]
        assert np.sum(mask) / mask.size <= 0.5

        # Random pruning (set seed for reproducibility)
        np.random.seed(42)
        pruned, pruning_info = pruner.prune_random(weight, sparsity=0.3)
        assert pruned is not None
        assert "mask" in pruning_info
        mask = pruning_info["mask"]
        # Allow some tolerance for randomness (should be ~70% remaining)
        assert np.sum(mask) / mask.size <= 0.75

    def test_training_modules(self):
        """Test training modules."""
        # Test CheckpointConfig
        config = CheckpointConfig(
            save_every_n_epochs=5,
            save_on_best_metric="loss",
            minimize_metric=True,
            keep_last_n_checkpoints=3,
        )

        assert config.save_every_n_epochs == 5
        assert config.save_on_best_metric == "loss"

        # Test TrainingState
        state = TrainingState(
            epoch=10,
            global_step=1000,
            learning_rate=0.001,
            loss=0.5,
            metrics={"accuracy": 0.95},
        )

        assert state.epoch == 10
        assert state.metrics["accuracy"] == 0.95

        # Test dict conversion
        state_dict = state.to_dict()
        restored = TrainingState.from_dict(state_dict)
        assert restored.epoch == state.epoch

    def test_cli_module(self):
        """Test CLI module."""
        cli = CoralCLI()

        # Test parser exists
        assert cli.parser is not None

        # Test can parse commands
        for cmd in ["init", "status", "add", "commit", "log"]:
            try:
                args = cli.parser.parse_args([cmd])
                assert args.command == cmd
            except SystemExit:
                # Some commands require additional args
                pass
