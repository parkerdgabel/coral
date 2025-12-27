"""Final push to reach 80% coverage."""

import tempfile
from pathlib import Path

import numpy as np

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.delta.delta_encoder import Delta, DeltaEncoder, DeltaType
from coral.storage.hdf5_store import HDF5Store
from coral.version_control.branch import Branch
from coral.version_control.commit import Commit, CommitMetadata
from coral.version_control.version import Version


class TestFinalCoverage80:
    """Final tests to reach 80% coverage."""

    def test_hdf5_store_basic_operations(self):
        """Test HDF5Store basic operations to increase coverage."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            store_path = tmp.name

        try:
            # Create store
            store = HDF5Store(store_path, compression="gzip")

            # Store weight
            weight = WeightTensor(
                data=np.array([1, 2, 3, 4, 5], dtype=np.float32),
                metadata=WeightMetadata(name="test", shape=(5,), dtype=np.float32),
            )
            hash_key = store.store(weight)
            assert hash_key is not None

            # Check existence
            assert store.exists(hash_key)

            # Load weight
            retrieved = store.load(hash_key)
            assert retrieved is not None
            np.testing.assert_array_equal(retrieved.data, weight.data)

            # List weights
            weights = store.list_weights()
            assert hash_key in weights

            # Store delta
            delta = Delta(
                delta_type=DeltaType.SPARSE,
                data=np.array([0.1, 0.2], dtype=np.float32),
                metadata={"test": "delta"},
                original_shape=(5,),
                original_dtype=np.dtype(np.float32),
                reference_hash="ref123",
            )
            delta_hash = "delta_" + hash_key
            store.store_delta(delta, delta_hash)

            # Get stats
            stats = store.get_storage_info()
            assert stats["total_weights"] >= 1
            delta_stats = store.get_delta_storage_info()
            assert delta_stats["total_deltas"] >= 1

            # Close store
            store.close()

            # Reopen and verify
            store2 = HDF5Store(store_path)
            assert store2.exists(hash_key)
            store2.close()

        finally:
            Path(store_path).unlink(missing_ok=True)

    def test_delta_encoder_operations(self):
        """Test DeltaEncoder operations."""
        from coral.delta.delta_encoder import DeltaConfig

        # Create test data
        reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        target = np.array([1.1, 2.0, 3.2, 4.0, 5.3], dtype=np.float32)

        ref_weight = WeightTensor(
            data=reference,
            metadata=WeightMetadata(
                name="ref", shape=reference.shape, dtype=reference.dtype
            ),
        )

        target_weight = WeightTensor(
            data=target,
            metadata=WeightMetadata(
                name="target", shape=target.shape, dtype=target.dtype
            ),
        )

        # Create encoder instance
        config = DeltaConfig(delta_type=DeltaType.SPARSE)
        encoder = DeltaEncoder(config)

        # Test encoding
        delta = encoder.encode_delta(target_weight, ref_weight)
        assert delta is not None
        assert delta.delta_type == DeltaType.SPARSE
        assert delta.reference_hash == ref_weight.compute_hash()

        # Test decoding
        decoded = encoder.decode_delta(delta, ref_weight)
        assert decoded is not None
        np.testing.assert_allclose(decoded.data, target, rtol=1e-5)

    def test_commit_operations(self):
        """Test commit operations for coverage."""
        metadata = CommitMetadata(
            author="Test Author",
            email="test@example.com",
            message="Test commit",
            tags=["v1", "test"],
        )

        commit = Commit(
            commit_hash="test_hash_123",
            parent_hashes=["parent1", "parent2"],
            weight_hashes={"w1": "h1", "w2": "h2", "w3": "h3"},
            metadata=metadata,
        )

        # Test dict conversion
        commit_dict = commit.to_dict()
        assert isinstance(commit_dict, dict)
        assert commit_dict["commit_hash"] == "test_hash_123"

        # Test deserialization
        commit2 = Commit.from_dict(commit_dict)
        assert commit2.commit_hash == commit.commit_hash
        assert len(commit2.parent_hashes) == 2
        assert len(commit2.weight_hashes) == 3
        assert commit2.metadata.message == "Test commit"

    def test_branch_operations(self):
        """Test branch operations for coverage."""
        branch = Branch(name="feature-x", commit_hash="commit123")

        # Test dict serialization
        branch_dict = branch.to_dict()
        assert isinstance(branch_dict, dict)
        assert branch_dict["name"] == "feature-x"

        # Test dict deserialization
        branch2 = Branch.from_dict(branch_dict)
        assert branch2.name == "feature-x"
        assert branch2.commit_hash == "commit123"

    def test_version_operations(self):
        """Test version operations for coverage."""
        version = Version(
            version_id="version_456",
            commit_hash="commit789",
            name="v2.0.0",
            description="Major release",
            metrics={"accuracy": 0.95, "loss": 0.05},
        )

        # Test properties
        assert version.name == "v2.0.0"
        assert version.metrics["accuracy"] == 0.95

        # Test dict serialization
        version_dict = version.to_dict()
        assert isinstance(version_dict, dict)
        assert version_dict["name"] == "v2.0.0"

        # Test dict deserialization
        version2 = Version.from_dict(version_dict)
        assert version2.name == version.name
        assert version2.description == version.description
        assert version2.metrics["loss"] == 0.05

    def test_hdf5_store_batch_operations(self):
        """Test HDF5Store batch operations."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            store_path = tmp.name

        try:
            store = HDF5Store(store_path)

            # Create multiple weights
            weights = {}
            for i in range(5):
                data = np.random.randn(10).astype(np.float32) * i
                weight = WeightTensor(
                    data=data,
                    metadata=WeightMetadata(
                        name=f"w{i}", shape=data.shape, dtype=data.dtype
                    ),
                )
                weights[f"w{i}"] = weight

            # Batch store
            hash_map = store.store_batch(weights)
            assert len(hash_map) == 5

            # Batch retrieve
            hashes = list(hash_map.values())
            retrieved = store.load_batch(hashes)
            assert len(retrieved) == 5

            # Verify content
            for name, weight in weights.items():
                hash_key = hash_map[name]
                retrieved_weight = retrieved[hash_key]
                np.testing.assert_array_equal(retrieved_weight.data, weight.data)

            store.close()

        finally:
            Path(store_path).unlink(missing_ok=True)

    def test_metadata_operations(self):
        """Test metadata operations for coverage."""
        # WeightMetadata
        meta1 = WeightMetadata(
            name="conv.weight",
            shape=(64, 3, 3, 3),
            dtype=np.float32,
            layer_type="Conv2d",
            model_name="ResNet",
            compression_info={"method": "quantization", "bits": 8},
        )

        # Test metadata attributes
        assert meta1.name == "conv.weight"
        assert meta1.layer_type == "Conv2d"
        assert meta1.compression_info["bits"] == 8
        assert meta1.model_name == "ResNet"

        # CommitMetadata with all fields
        commit_meta = CommitMetadata(
            author="John Doe",
            email="john@example.com",
            message="Feature complete",
            tags=["release", "v1.0", "stable"],
        )

        # Test tags field
        assert len(commit_meta.tags) == 3
