"""Tests for delta encoding functionality."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from coral.core.deduplicator import Deduplicator
from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.delta.delta_encoder import DeltaConfig, DeltaEncoder, DeltaType
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
def similar_weights():
    """Create similar weights for testing delta encoding."""
    base_data = np.random.randn(100, 50).astype(np.float32)

    # Create reference weight
    reference = WeightTensor(
        data=base_data,
        metadata=WeightMetadata(
            name="reference_weight",
            shape=(100, 50),
            dtype=np.float32,
            layer_type="Linear",
        ),
    )

    # Create similar weight (add small noise)
    noise = np.random.randn(*base_data.shape).astype(np.float32) * 0.01
    similar_data = base_data + noise

    similar = WeightTensor(
        data=similar_data,
        metadata=WeightMetadata(
            name="similar_weight",
            shape=(100, 50),
            dtype=np.float32,
            layer_type="Linear",
        ),
    )

    return reference, similar


class TestDeltaEncoder:
    """Test delta encoding functionality."""

    def test_delta_config_serialization(self):
        """Test delta configuration serialization."""
        config = DeltaConfig(
            delta_type=DeltaType.INT8_QUANTIZED,
            sparse_threshold=1e-5,
            quantization_bits=8,
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["delta_type"] == "int8_quantized"
        assert config_dict["sparse_threshold"] == 1e-5

        # Test from_dict
        restored_config = DeltaConfig.from_dict(config_dict)
        assert restored_config.delta_type == DeltaType.INT8_QUANTIZED
        assert restored_config.sparse_threshold == 1e-5

    def test_can_encode_as_delta(self, similar_weights):
        """Test delta encoding feasibility check."""
        reference, similar = similar_weights

        encoder = DeltaEncoder(DeltaConfig(delta_type=DeltaType.FLOAT32_RAW))

        # Should be able to encode similar weights
        assert encoder.can_encode_as_delta(similar, reference)

        # Should not encode different shapes
        different_shape = WeightTensor(
            data=np.random.randn(50, 100).astype(np.float32),
            metadata=WeightMetadata(
                name="different", shape=(50, 100), dtype=np.float32
            ),
        )
        assert not encoder.can_encode_as_delta(different_shape, reference)

    def test_float32_raw_encoding(self, similar_weights):
        """Test raw float32 delta encoding."""
        reference, similar = similar_weights

        config = DeltaConfig(delta_type=DeltaType.FLOAT32_RAW)
        encoder = DeltaEncoder(config)

        # Encode delta
        delta = encoder.encode_delta(similar, reference)
        assert delta.delta_type == DeltaType.FLOAT32_RAW
        assert delta.original_shape == similar.shape
        assert delta.reference_hash == reference.compute_hash()

        # Decode and verify
        reconstructed = encoder.decode_delta(delta, reference)
        np.testing.assert_array_almost_equal(
            reconstructed.data, similar.data, decimal=6
        )
        assert reconstructed.shape == similar.shape
        assert reconstructed.dtype == similar.dtype

    def test_quantized_encoding(self, similar_weights):
        """Test quantized delta encoding."""
        reference, similar = similar_weights

        # Test 8-bit quantization
        config = DeltaConfig(delta_type=DeltaType.INT8_QUANTIZED)
        encoder = DeltaEncoder(config)

        delta = encoder.encode_delta(similar, reference)
        assert delta.delta_type == DeltaType.INT8_QUANTIZED
        assert delta.data.dtype == np.int8

        # Decode and check similarity (not exact due to quantization)
        reconstructed = encoder.decode_delta(delta, reference)
        assert reconstructed.shape == similar.shape

        # Should be reasonably close (within quantization error)
        mse = np.mean((reconstructed.data - similar.data) ** 2)
        assert mse < 0.01  # Adjust threshold as needed

    def test_sparse_encoding(self):
        """Test sparse delta encoding."""
        # Create mostly similar weights with few differences
        base_data = np.ones((50, 50), dtype=np.float32)
        reference = WeightTensor(
            data=base_data,
            metadata=WeightMetadata(name="ref", shape=(50, 50), dtype=np.float32),
        )

        # Create sparse differences
        sparse_data = base_data.copy()
        sparse_data[10, 20] = 2.0
        sparse_data[30, 40] = 0.5

        similar = WeightTensor(
            data=sparse_data,
            metadata=WeightMetadata(name="sparse", shape=(50, 50), dtype=np.float32),
        )

        config = DeltaConfig(delta_type=DeltaType.SPARSE, sparse_threshold=1e-6)
        encoder = DeltaEncoder(config)

        delta = encoder.encode_delta(similar, reference)
        assert delta.delta_type == DeltaType.SPARSE

        # Should have good compression for sparse data
        assert delta.compression_ratio > 0.9

        # Verify reconstruction
        reconstructed = encoder.decode_delta(delta, reference)
        np.testing.assert_array_almost_equal(
            reconstructed.data, similar.data, decimal=6
        )

    def test_compressed_encoding(self, similar_weights):
        """Test compressed delta encoding."""
        reference, similar = similar_weights

        config = DeltaConfig(delta_type=DeltaType.COMPRESSED)
        encoder = DeltaEncoder(config)

        delta = encoder.encode_delta(similar, reference)
        assert delta.delta_type == DeltaType.COMPRESSED

        # Should achieve some compression
        assert delta.compression_ratio > 0

        # Verify exact reconstruction
        reconstructed = encoder.decode_delta(delta, reference)
        np.testing.assert_array_equal(reconstructed.data, similar.data)


class TestDeduplicatorWithDeltas:
    """Test deduplicator with delta encoding enabled."""

    def test_delta_enabled_deduplication(self, similar_weights):
        """Test deduplication with delta encoding enabled."""
        reference, similar = similar_weights

        config = DeltaConfig(delta_type=DeltaType.FLOAT32_RAW)
        dedup = Deduplicator(
            similarity_threshold=0.95, delta_config=config, enable_delta_encoding=True
        )

        # Add reference weight first
        ref_hash = dedup.add_weight(reference, "reference")

        # Add similar weight
        sim_hash = dedup.add_weight(similar, "similar")

        # Should point to same reference
        assert sim_hash == ref_hash

        # Should be delta encoded
        assert dedup.is_delta_encoded("similar")
        assert not dedup.is_delta_encoded("reference")

        # Should be able to get delta
        delta = dedup.get_delta_by_name("similar")
        assert delta is not None
        assert delta.reference_hash == ref_hash

    def test_lossless_reconstruction(self, similar_weights):
        """Test lossless reconstruction of delta-encoded weights."""
        reference, similar = similar_weights

        dedup = Deduplicator(enable_delta_encoding=True)

        # Add weights
        dedup.add_weight(reference, "reference")
        dedup.add_weight(similar, "similar")

        # Retrieve weights
        retrieved_ref = dedup.get_weight_by_name("reference")
        retrieved_sim = dedup.get_weight_by_name("similar")

        # Reference should be exact
        np.testing.assert_array_equal(retrieved_ref.data, reference.data)

        # Similar should be reconstructed exactly (lossless)
        np.testing.assert_array_almost_equal(
            retrieved_sim.data, similar.data, decimal=6
        )

    def test_compression_statistics(self, similar_weights):
        """Test compression statistics with delta encoding."""
        reference, similar = similar_weights

        # Use compressed delta encoding for actual compression
        config = DeltaConfig(delta_type=DeltaType.COMPRESSED)
        dedup = Deduplicator(enable_delta_encoding=True, delta_config=config)
        dedup.add_weight(reference, "reference")
        dedup.add_weight(similar, "similar")

        stats = dedup.get_compression_stats()

        # Should have delta statistics
        assert "delta_stats" in stats
        delta_stats = stats["delta_stats"]

        assert delta_stats["total_deltas"] == 1
        assert delta_stats["delta_compression_ratio"] > 0
        assert delta_stats["total_delta_size"] > 0


class TestRepositoryWithDeltas:
    """Test repository operations with delta encoding."""

    def test_stage_and_commit_with_deltas(self, temp_repo, similar_weights):
        """Test staging and committing with delta encoding."""
        reference, similar = similar_weights

        weights = {"reference": reference, "similar": similar}

        # Stage weights
        staged = temp_repo.stage_weights(weights)
        assert len(staged) == 2

        # Should be using same reference hash for similar weights
        assert staged["reference"] == staged["similar"]

        # Commit
        commit = temp_repo.commit("Test delta encoding")
        assert commit is not None

        # Verify commit contains both weights
        assert len(commit.weight_hashes) == 2

    def test_weight_retrieval_with_deltas(self, temp_repo, similar_weights):
        """Test weight retrieval with delta reconstruction."""
        reference, similar = similar_weights

        weights = {"reference": reference, "similar": similar}

        # Stage and commit
        temp_repo.stage_weights(weights)
        temp_repo.commit("Test retrieval")

        # Retrieve weights
        retrieved_ref = temp_repo.get_weight("reference")
        retrieved_sim = temp_repo.get_weight("similar")

        assert retrieved_ref is not None
        assert retrieved_sim is not None

        # Should reconstruct original data
        np.testing.assert_array_equal(retrieved_ref.data, reference.data)
        np.testing.assert_array_almost_equal(
            retrieved_sim.data, similar.data, decimal=5
        )

    def test_different_delta_types(self, temp_repo):
        """Test different delta encoding types."""
        # Create base weight
        base_data = np.random.randn(20, 30).astype(np.float32)
        reference = WeightTensor(
            data=base_data,
            metadata=WeightMetadata(name="ref", shape=(20, 30), dtype=np.float32),
        )

        # Create similar weights with different characteristics
        similar1_data = (
            base_data + np.random.randn(*base_data.shape).astype(np.float32) * 0.001
        )  # Very similar
        similar1 = WeightTensor(
            data=similar1_data,
            metadata=WeightMetadata(name="sim1", shape=(20, 30), dtype=np.float32),
        )

        weights = {"reference": reference, "similar1": similar1}

        # Test with different delta configurations
        for delta_type in [
            DeltaType.FLOAT32_RAW,
            DeltaType.INT8_QUANTIZED,
            DeltaType.COMPRESSED,
        ]:
            # Update repository configuration
            temp_repo.config["core"]["delta_type"] = delta_type.value

            # Reinitialize deduplicator with new config
            from coral.delta.delta_encoder import DeltaConfig

            delta_config = DeltaConfig(delta_type=delta_type)
            temp_repo.deduplicator = Deduplicator(
                similarity_threshold=0.98,
                delta_config=delta_config,
                enable_delta_encoding=True,
            )

            # Stage and commit
            temp_repo.stage_weights(weights)
            temp_repo.commit(f"Test {delta_type.value}")

            # Verify retrieval works
            retrieved = temp_repo.get_weight("similar1")
            assert retrieved is not None

            # Check that shape and basic properties are preserved
            assert retrieved.shape == similar1.shape
            assert retrieved.dtype == similar1.dtype
