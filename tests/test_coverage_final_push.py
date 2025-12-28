"""Final tests to reach 80% coverage."""

import tempfile
from pathlib import Path

import numpy as np

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.training.training_state import TrainingState
from coral.version_control.branch import Branch, BranchManager
from coral.version_control.commit import Commit, CommitMetadata


class TestCoverageFinalPush:
    """Final tests to reach 80% coverage threshold."""

    def test_weight_tensor_properties(self):
        """Test WeightTensor properties and methods."""
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        metadata = WeightMetadata(name="test", shape=data.shape, dtype=data.dtype)
        weight = WeightTensor(data=data, metadata=metadata)

        # Test properties
        assert weight.shape == (2, 2)
        assert weight.dtype == np.float32
        assert weight.nbytes == 16
        assert weight.size == 4

        # Test hash computation
        hash1 = weight.compute_hash()
        hash2 = weight.compute_hash()
        assert hash1 == hash2

        # Test string representation
        str_repr = str(weight)
        assert "test" in str_repr

    def test_commit_properties(self):
        """Test Commit properties."""
        metadata = CommitMetadata(
            message="Test commit", author="Test Author", email="test@example.com"
        )

        commit = Commit(
            commit_hash="abc123",
            parent_hashes=["parent1", "parent2"],
            weight_hashes={"w1": "h1", "w2": "h2"},
            metadata=metadata,
        )

        # Test properties
        assert len(commit.parent_hashes) == 2
        assert len(commit.weight_hashes) == 2
        assert commit.metadata.message == "Test commit"

        # Test serialization
        commit_dict = commit.to_dict()
        assert commit_dict["commit_hash"] == "abc123"
        assert "parent_hashes" in commit_dict
        assert "weight_hashes" in commit_dict

    def test_branch_manager_basics(self):
        """Test BranchManager basic operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # BranchManager expects repo_path, not refs_dir
            repo_path = Path(tmpdir)
            coral_dir = repo_path / ".coral"
            coral_dir.mkdir(parents=True)
            (coral_dir / "refs" / "heads").mkdir(parents=True)
            (coral_dir / "HEAD").write_text("ref: refs/heads/main")

            manager = BranchManager(repo_path)

            # Create branch
            branch = manager.create_branch("test-branch", "commit123")
            assert branch.name == "test-branch"
            assert branch.commit_hash == "commit123"

            # Get branch
            retrieved = manager.get_branch("test-branch")
            assert retrieved is not None
            assert retrieved.name == "test-branch"

            # Update branch
            manager.update_branch("test-branch", "commit456")
            updated = manager.get_branch("test-branch")
            assert updated.commit_hash == "commit456"

            # Delete branch
            manager.delete_branch("test-branch")
            assert manager.get_branch("test-branch") is None

    def test_training_state_full(self):
        """Test TrainingState with all fields."""
        state = TrainingState(
            epoch=10,
            global_step=1000,
            learning_rate=0.001,
            loss=0.5,
            metrics={"accuracy": 0.95, "val_loss": 0.6},
            optimizer_state={"lr": 0.001, "momentum": 0.9},
            scheduler_state={"last_epoch": 10},
            random_state={"seed": 42},
            model_name="TestModel",
            experiment_name="exp1",
        )

        # Test all fields
        assert state.epoch == 10
        assert state.global_step == 1000
        assert state.metrics["accuracy"] == 0.95
        assert state.optimizer_state["momentum"] == 0.9

        # Test serialization
        state_dict = state.to_dict()
        assert state_dict["epoch"] == 10
        assert "metrics" in state_dict
        assert "timestamp" in state_dict

        # Test from_dict
        state2 = TrainingState.from_dict(state_dict)
        assert state2.epoch == state.epoch
        assert state2.loss == state.loss

    def test_weight_metadata_equality(self):
        """Test WeightMetadata equality and hashing."""
        metadata1 = WeightMetadata(name="weight1", shape=(10, 20), dtype=np.float32)

        metadata2 = WeightMetadata(name="weight1", shape=(10, 20), dtype=np.float32)

        metadata3 = WeightMetadata(name="weight2", shape=(10, 20), dtype=np.float32)

        # Test equality
        assert metadata1 == metadata2
        assert metadata1 != metadata3

        # Test that metadata can be used in WeightTensor
        data = np.random.randn(10, 20).astype(np.float32)
        weight = WeightTensor(data=data, metadata=metadata1)
        weight_dict = weight.to_dict()
        assert weight_dict["metadata"]["name"] == "weight1"
        assert weight_dict["metadata"]["shape"] == [10, 20]

    def test_commit_metadata_dict(self):
        """Test CommitMetadata dict conversion."""

        metadata = CommitMetadata(
            message="Test",
            author="Author",
            email="email@test.com",
            tags=["v1.0", "release"],
        )

        # To dict
        meta_dict = metadata.to_dict()
        assert meta_dict["message"] == "Test"
        assert meta_dict["author"] == "Author"
        assert "timestamp" in meta_dict
        assert meta_dict["tags"] == ["v1.0", "release"]

        # From dict
        metadata2 = CommitMetadata.from_dict(meta_dict)
        assert metadata2.message == metadata.message
        assert metadata2.author == metadata.author

    def test_branch_serialization(self):
        """Test Branch serialization."""
        branch = Branch("feature", "commit123")

        # To dict
        branch_dict = branch.to_dict()
        assert branch_dict["name"] == "feature"
        assert branch_dict["commit_hash"] == "commit123"

        # From dict
        branch2 = Branch.from_dict(branch_dict)
        assert branch2.name == branch.name
        assert branch2.commit_hash == branch.commit_hash

    def test_training_state_minimal(self):
        """Test TrainingState with minimal fields."""
        state = TrainingState(epoch=1, global_step=100, learning_rate=0.01, loss=1.0)

        # Check defaults
        assert state.metrics == {}
        assert state.optimizer_state is None
        assert state.scheduler_state is None
        assert state.model_name is None
