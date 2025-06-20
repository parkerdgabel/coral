"""Tests for version control functionality."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.version_control.branch import BranchManager
from coral.version_control.commit import Commit, CommitMetadata
from coral.version_control.repository import Repository
from coral.version_control.version import VersionGraph


@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing."""
    temp_dir = tempfile.mkdtemp()
    repo_path = Path(temp_dir)
    repo = Repository(repo_path, init=True)
    yield repo
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_weights():
    """Create sample weights for testing."""
    weights = {}

    # Create different types of weights
    weights["layer1.weight"] = WeightTensor(
        data=np.random.randn(64, 32).astype(np.float32),
        metadata=WeightMetadata(
            name="layer1.weight", shape=(64, 32), dtype=np.float32, layer_type="Linear"
        ),
    )

    weights["layer1.bias"] = WeightTensor(
        data=np.random.randn(64).astype(np.float32),
        metadata=WeightMetadata(
            name="layer1.bias", shape=(64,), dtype=np.float32, layer_type="Linear"
        ),
    )

    weights["layer2.weight"] = WeightTensor(
        data=np.random.randn(10, 64).astype(np.float32),
        metadata=WeightMetadata(
            name="layer2.weight", shape=(10, 64), dtype=np.float32, layer_type="Linear"
        ),
    )

    return weights


class TestRepository:
    """Test repository operations."""

    def test_repository_initialization(self, temp_repo):
        """Test repository initialization."""
        assert temp_repo.coral_dir.exists()
        assert temp_repo.objects_dir.exists()
        assert temp_repo.commits_dir.exists()
        assert temp_repo.staging_dir.exists()

        # Check initial branch
        assert temp_repo.branch_manager.get_current_branch() == "main"

    def test_stage_and_commit(self, temp_repo, sample_weights):
        """Test staging and committing weights."""
        # Stage weights
        staged = temp_repo.stage_weights(sample_weights)
        assert len(staged) == len(sample_weights)

        # Verify staging file exists
        staged_file = temp_repo.staging_dir / "staged.json"
        assert staged_file.exists()

        # Commit
        commit = temp_repo.commit("Initial commit")
        assert commit is not None
        assert commit.metadata.message == "Initial commit"
        assert len(commit.weight_hashes) == len(sample_weights)

        # Verify staging is cleared
        assert not staged_file.exists()

    def test_weight_retrieval(self, temp_repo, sample_weights):
        """Test weight retrieval after commit."""
        # Stage and commit
        temp_repo.stage_weights(sample_weights)
        temp_repo.commit("Test commit")

        # Retrieve individual weight
        weight = temp_repo.get_weight("layer1.weight")
        assert weight is not None
        assert weight.metadata.name == "layer1.weight"
        np.testing.assert_array_equal(weight.data, sample_weights["layer1.weight"].data)

        # Retrieve all weights
        all_weights = temp_repo.get_all_weights()
        assert len(all_weights) == len(sample_weights)
        assert "layer1.weight" in all_weights

    def test_branch_operations(self, temp_repo, sample_weights):
        """Test branch creation and switching."""
        # Create initial commit
        temp_repo.stage_weights(sample_weights)
        temp_repo.commit("Initial commit")

        # Create new branch
        temp_repo.create_branch("feature")
        assert temp_repo.branch_manager.branch_exists("feature")

        # Switch to new branch
        temp_repo.checkout("feature")
        assert temp_repo.branch_manager.get_current_branch() == "feature"

        # Modify weights and commit on feature branch
        modified_weights = sample_weights.copy()
        modified_weights["layer3.weight"] = WeightTensor(
            data=np.random.randn(5, 10).astype(np.float32),
            metadata=WeightMetadata(
                name="layer3.weight",
                shape=(5, 10),
                dtype=np.float32,
                layer_type="Linear",
            ),
        )

        temp_repo.stage_weights(modified_weights)
        temp_repo.commit("Add layer3")

        # Switch back to main
        temp_repo.checkout("main")
        assert temp_repo.branch_manager.get_current_branch() == "main"

        # Verify different weights on different branches
        main_weights = temp_repo.get_all_weights()
        assert "layer3.weight" not in main_weights

        temp_repo.checkout("feature")
        feature_weights = temp_repo.get_all_weights()
        assert "layer3.weight" in feature_weights

    def test_diff_functionality(self, temp_repo, sample_weights):
        """Test diff between commits."""
        # Create first commit
        temp_repo.stage_weights(sample_weights)
        commit1 = temp_repo.commit("First commit")

        # Modify weights
        modified_weights = sample_weights.copy()
        # Create a significantly different weight to ensure it's not deduplicated
        modified_weights["layer1.weight"] = WeightTensor(
            data=np.random.randn(64, 32).astype(np.float32)
            + 10.0,  # Significantly different
            metadata=WeightMetadata(
                name="layer1.weight",
                shape=(64, 32),
                dtype=np.float32,
                layer_type="Linear",
            ),
        )
        del modified_weights["layer1.bias"]  # Remove one
        modified_weights["new_layer.weight"] = WeightTensor(  # Add new
            data=np.random.randn(20, 30).astype(np.float32),
            metadata=WeightMetadata(
                name="new_layer.weight", shape=(20, 30), dtype=np.float32
            ),
        )

        # Create second commit
        temp_repo.stage_weights(modified_weights)
        commit2 = temp_repo.commit("Modified weights")

        # Test diff
        diff = temp_repo.diff(commit1.commit_hash, commit2.commit_hash)

        assert "new_layer.weight" in diff["added"]
        assert "layer1.bias" in diff["removed"]
        assert "layer1.weight" in diff["modified"]

        assert diff["summary"]["total_added"] == 1
        assert diff["summary"]["total_removed"] == 1
        assert diff["summary"]["total_modified"] == 1

    def test_version_tagging(self, temp_repo, sample_weights):
        """Test version tagging functionality."""
        # Create commit
        temp_repo.stage_weights(sample_weights)
        commit = temp_repo.commit("Version to tag")

        # Tag version
        version = temp_repo.tag_version(
            name="v1.0",
            description="First stable version",
            metrics={"accuracy": 0.95, "loss": 0.05},
        )

        assert version.name == "v1.0"
        assert version.description == "First stable version"
        assert version.metrics["accuracy"] == 0.95
        assert version.commit_hash == commit.commit_hash

        # Test get_version
        retrieved = temp_repo.get_version("v1.0")
        assert retrieved is not None
        assert retrieved.name == "v1.0"
        assert retrieved.description == "First stable version"

        # Test list_versions
        versions = temp_repo.list_versions()
        assert len(versions) == 1
        assert versions[0].name == "v1.0"

        # Create another version
        temp_repo.stage_weights(sample_weights)
        temp_repo.commit("Version 2")
        temp_repo.tag_version("v2.0", "Second version")

        # Test list_versions with multiple versions
        versions = temp_repo.list_versions()
        assert len(versions) == 2
        version_names = [v.name for v in versions]
        assert "v1.0" in version_names
        assert "v2.0" in version_names

    def test_log_functionality(self, temp_repo, sample_weights):
        """Test commit log."""
        commits = []

        # Create multiple commits
        for i in range(3):
            weights = {}
            # Create significantly different weights for each commit
            for name, weight in sample_weights.items():
                weights[name] = WeightTensor(
                    data=np.random.randn(*weight.shape).astype(np.float32) + i * 5.0,
                    metadata=WeightMetadata(
                        name=weight.metadata.name,
                        shape=weight.metadata.shape,
                        dtype=weight.metadata.dtype,
                        layer_type=weight.metadata.layer_type,
                    ),
                )

            temp_repo.stage_weights(weights)
            commit = temp_repo.commit(f"Commit {i + 1}")
            commits.append(commit)

        # Test log
        log = temp_repo.log(max_commits=2)
        assert len(log) == 2
        assert log[0].commit_hash == commits[-1].commit_hash  # Most recent first
        assert log[1].commit_hash == commits[-2].commit_hash

    def test_repository_status(self, temp_repo, sample_weights):
        """Test repository status functionality."""
        # Initial status - should be empty
        status = temp_repo.status()
        assert "staged" in status
        assert len(status["staged"]) == 0

        # Stage some weights
        temp_repo.stage_weights(sample_weights)

        # Check status shows staged weights
        status = temp_repo.status()
        assert len(status["staged"]) == len(sample_weights)
        assert all(name in status["staged"] for name in sample_weights.keys())

        # Commit and check status is cleared
        temp_repo.commit("Test commit")
        status = temp_repo.status()
        assert len(status["staged"]) == 0

    def test_repository_error_handling(self, sample_weights):
        """Test repository error handling cases."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Test loading non-existent repository without init
            with pytest.raises(ValueError, match="Not a Coral repository"):
                Repository(temp_dir)  # No init=True

            # Initialize properly
            repo = Repository(temp_dir, init=True)

            # Test committing with no staged changes
            with pytest.raises(ValueError, match="Nothing to commit"):
                repo.commit("Empty commit")

            # Test checking out non-existent branch
            with pytest.raises(ValueError, match="Invalid target"):
                repo.checkout("non-existent-branch")

            # Test getting non-existent weight (should return None, not raise)
            assert repo.get_weight("non-existent-weight") is None

            # Add a commit to test get_commit error
            repo.stage_weights(sample_weights)
            repo.commit("Test commit")

            # Test getting non-existent commit
            with pytest.raises(ValueError, match="Commit not found"):
                repo.get_commit("non-existent-hash")

        finally:
            shutil.rmtree(temp_dir)


class TestBranchManager:
    """Test branch management."""

    def test_branch_creation(self):
        """Test branch creation and management."""
        temp_dir = tempfile.mkdtemp()
        try:
            repo_path = Path(temp_dir)
            manager = BranchManager(repo_path)

            # Create branch
            branch = manager.create_branch("test-branch", "dummy-hash")
            assert branch.name == "test-branch"
            assert branch.commit_hash == "dummy-hash"

            # Check branch exists
            assert manager.branch_exists("test-branch")

            # Get branch
            retrieved = manager.get_branch("test-branch")
            assert retrieved.name == "test-branch"
            assert retrieved.commit_hash == "dummy-hash"

            # List branches
            branches = manager.list_branches()
            assert len(branches) >= 1
            assert any(b.name == "test-branch" for b in branches)

        finally:
            shutil.rmtree(temp_dir)


class TestVersionGraph:
    """Test version graph functionality."""

    def test_commit_relationships(self):
        """Test commit parent-child relationships."""
        graph = VersionGraph()

        # Create commits
        commit1 = Commit(
            commit_hash="hash1",
            parent_hashes=[],
            weight_hashes={"w1": "h1"},
            metadata=CommitMetadata(
                author="Test", email="test@test.com", message="First"
            ),
        )

        commit2 = Commit(
            commit_hash="hash2",
            parent_hashes=["hash1"],
            weight_hashes={"w1": "h2"},
            metadata=CommitMetadata(
                author="Test", email="test@test.com", message="Second"
            ),
        )

        graph.add_commit(commit1)
        graph.add_commit(commit2)

        # Test relationships
        ancestors = graph.get_commit_ancestors("hash2")
        assert "hash1" in ancestors

        descendants = graph.get_commit_descendants("hash1")
        assert "hash2" in descendants

        # Test history
        history = graph.get_branch_history("hash2")
        assert history == ["hash2", "hash1"]

    def test_merge_scenarios(self):
        """Test merge commit scenarios."""
        graph = VersionGraph()

        # Create branch scenario
        # main: A -> B
        # branch: A -> C
        # merge: B + C -> D

        commits = []
        commits.append(
            Commit("A", [], {"w1": "h1"}, CommitMetadata("Test", "test@test.com", "A"))
        )
        commits.append(
            Commit(
                "B", ["A"], {"w1": "h2"}, CommitMetadata("Test", "test@test.com", "B")
            )
        )
        commits.append(
            Commit(
                "C", ["A"], {"w1": "h3"}, CommitMetadata("Test", "test@test.com", "C")
            )
        )
        commits.append(
            Commit(
                "D",
                ["B", "C"],
                {"w1": "h4"},
                CommitMetadata("Test", "test@test.com", "Merge"),
            )
        )

        for commit in commits:
            graph.add_commit(commit)

        # Test merge detection
        merge_commit = graph.get_commit("D")
        assert merge_commit.is_merge_commit

        # Test common ancestor
        common = graph.get_common_ancestor("B", "C")
        assert common == "A"
