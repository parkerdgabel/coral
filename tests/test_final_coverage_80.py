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
            hash_key = store.store_weight(weight)
            assert hash_key is not None

            # Check existence
            assert store.has_weight(hash_key)

            # Get weight
            retrieved = store.get_weight(hash_key)
            assert retrieved is not None
            np.testing.assert_array_equal(retrieved.data, weight.data)

            # List weights
            weights = store.list_weights()
            assert hash_key in weights

            # Store delta
            delta = Delta(
                delta_type=DeltaType.SPARSE,
                reference_hash="ref123",
                delta_data={"indices": [0, 1], "values": [0.1, 0.2]},
                metadata={"test": "delta"},
            )
            delta_hash = store.store_delta(delta)
            assert delta_hash is not None

            # Get stats
            stats = store.get_stats()
            assert stats["weight_count"] >= 1
            assert stats["delta_count"] >= 1

            # Close store
            store.close()

            # Reopen and verify
            store2 = HDF5Store(store_path)
            assert store2.has_weight(hash_key)
            store2.close()

        finally:
            Path(store_path).unlink(missing_ok=True)

    def test_delta_encoder_operations(self):
        """Test DeltaEncoder operations."""
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

        # Test encoding
        delta = DeltaEncoder.encode(
            ref_weight, target_weight, strategy=DeltaType.SPARSE
        )
        assert delta is not None
        assert delta.delta_type == DeltaType.SPARSE
        assert delta.reference_hash == ref_weight.compute_hash()

        # Test decoding
        decoded = DeltaEncoder.decode(ref_weight, delta)
        assert decoded is not None
        np.testing.assert_allclose(decoded.data, target, rtol=1e-5)

        # Test similarity
        similarity = DeltaEncoder.compute_similarity(ref_weight, target_weight)
        assert 0 <= similarity <= 1

    def test_commit_operations(self):
        """Test commit operations for coverage."""
        metadata = CommitMetadata(
            message="Test commit",
            author="Test Author",
            email="test@example.com",
            tags=["v1", "test"],
        )

        commit = Commit(
            commit_hash="test_hash_123",
            parent_hashes=["parent1", "parent2"],
            weight_hashes={"w1": "h1", "w2": "h2", "w3": "h3"},
            metadata=metadata,
        )

        # Test serialization
        commit_json = commit.to_json()
        assert isinstance(commit_json, str)

        # Test deserialization
        commit2 = Commit.from_json(commit_json)
        assert commit2.commit_hash == commit.commit_hash
        assert len(commit2.parent_hashes) == 2
        assert len(commit2.weight_hashes) == 3

        # Test dict conversion
        commit_dict = commit.to_dict()
        commit3 = Commit.from_dict(commit_dict)
        assert commit3.metadata.message == "Test commit"

    def test_branch_operations(self):
        """Test branch operations for coverage."""
        branch = Branch(name="feature-x", commit_hash="commit123")

        # Test JSON serialization
        branch_json = branch.to_json()
        assert isinstance(branch_json, str)

        # Test JSON deserialization
        branch2 = Branch.from_json(branch_json)
        assert branch2.name == "feature-x"
        assert branch2.commit_hash == "commit123"

    def test_version_operations(self):
        """Test version operations for coverage."""
        version = Version(
            name="v2.0.0",
            version_id="version_456",
            commit_hash="commit789",
            description="Major release",
            metrics={"accuracy": 0.95, "loss": 0.05},
        )

        # Test properties
        assert version.name == "v2.0.0"
        assert version.metrics["accuracy"] == 0.95

        # Test JSON serialization
        version_json = version.to_json()
        assert isinstance(version_json, str)

        # Test JSON deserialization
        version2 = Version.from_json(version_json)
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
            hash_map = store.store_weights_batch(weights)
            assert len(hash_map) == 5

            # Batch retrieve
            hashes = list(hash_map.values())
            retrieved = store.get_weights_batch(hashes)
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

        # Test dict conversion
        meta_dict = meta1.to_dict()
        meta2 = WeightMetadata.from_dict(meta_dict)
        assert meta2.name == meta1.name
        assert meta2.layer_type == meta1.layer_type
        assert meta2.compression_info["bits"] == 8

        # CommitMetadata with all fields
        commit_meta = CommitMetadata(
            message="Feature complete",
            author="John Doe",
            email="john@example.com",
            tags=["release", "v1.0", "stable"],
            metadata={"branch": "main", "ci_passed": True},
        )

        # Test metadata field
        assert commit_meta.metadata["ci_passed"] is True
        assert len(commit_meta.tags) == 3
