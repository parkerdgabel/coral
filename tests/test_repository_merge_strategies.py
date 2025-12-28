"""Tests for repository merge strategies and advanced operations."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.version_control.repository import MergeStrategy, Repository


def create_weight_tensor(data, name="test_weight", **kwargs):
    """Helper to create weight tensor with proper metadata."""
    metadata = WeightMetadata(name=name, shape=data.shape, dtype=data.dtype, **kwargs)
    return WeightTensor(data=data, metadata=metadata)


class TestMergeStrategies:
    """Tests for different merge strategies."""

    @pytest.fixture
    def temp_repo(self):
        """Create a temporary repository for testing."""
        temp_dir = tempfile.mkdtemp()
        repo = Repository(temp_dir, init=True)
        yield repo
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def diverged_repo(self, temp_repo):
        """Create a repo with diverged branches for merge testing."""
        # Initial commit
        weight = create_weight_tensor(np.ones(10) * 5, "shared_weight")
        temp_repo.stage_weights({"shared_weight": weight})
        temp_repo.commit("Initial commit")

        # Create feature branch and modify weight
        temp_repo.create_branch("feature")
        temp_repo.checkout("feature")
        feature_weight = create_weight_tensor(np.ones(10) * 10, "shared_weight")
        temp_repo.stage_weights({"shared_weight": feature_weight})
        temp_repo.commit("Feature change")

        # Switch to main and make different change
        temp_repo.checkout("main")
        main_weight = create_weight_tensor(np.ones(10) * 20, "shared_weight")
        temp_repo.stage_weights({"shared_weight": main_weight})
        temp_repo.commit("Main change")

        return temp_repo

    def test_merge_strategy_ours(self, diverged_repo):
        """Test OURS merge strategy keeps current branch weights."""
        merge_commit = diverged_repo.merge("feature", strategy=MergeStrategy.OURS)

        assert merge_commit is not None

        # Get the merged weight - should be 20 (main's value)
        weight = diverged_repo.get_weight("shared_weight")
        np.testing.assert_array_almost_equal(weight.data, np.ones(10) * 20)

    def test_merge_strategy_theirs(self, diverged_repo):
        """Test THEIRS merge strategy takes source branch weights."""
        merge_commit = diverged_repo.merge("feature", strategy=MergeStrategy.THEIRS)

        assert merge_commit is not None

        # Get the merged weight - should be 10 (feature's value)
        weight = diverged_repo.get_weight("shared_weight")
        np.testing.assert_array_almost_equal(weight.data, np.ones(10) * 10)

    def test_merge_strategy_average(self, diverged_repo):
        """Test AVERAGE merge strategy averages weights."""
        merge_commit = diverged_repo.merge("feature", strategy=MergeStrategy.AVERAGE)

        assert merge_commit is not None

        # Get the merged weight - should be 15 (average of 10 and 20)
        weight = diverged_repo.get_weight("shared_weight")
        np.testing.assert_array_almost_equal(weight.data, np.ones(10) * 15)

    def test_merge_strategy_weighted(self, diverged_repo):
        """Test WEIGHTED merge strategy with alpha."""
        # Use alpha=0.3 (30% ours, 70% theirs)
        merge_commit = diverged_repo.merge(
            "feature", strategy=MergeStrategy.WEIGHTED, merge_alpha=0.3
        )

        assert merge_commit is not None

        # Get the merged weight
        # Expected: 0.3 * 20 (main) + 0.7 * 10 (feature) = 6 + 7 = 13
        weight = diverged_repo.get_weight("shared_weight")
        np.testing.assert_array_almost_equal(weight.data, np.ones(10) * 13)

    def test_merge_strategy_weighted_equal(self, diverged_repo):
        """Test WEIGHTED merge with alpha=0.5 equals AVERAGE."""
        merge_commit = diverged_repo.merge(
            "feature", strategy=MergeStrategy.WEIGHTED, merge_alpha=0.5
        )

        assert merge_commit is not None

        # Expected: 0.5 * 20 + 0.5 * 10 = 15
        weight = diverged_repo.get_weight("shared_weight")
        np.testing.assert_array_almost_equal(weight.data, np.ones(10) * 15)

    def test_merge_strategy_fail_raises_conflict(self, diverged_repo):
        """Test FAIL merge strategy raises on conflicts."""
        from coral.version_control.repository import MergeConflictError

        with pytest.raises(MergeConflictError):
            diverged_repo.merge("feature", strategy=MergeStrategy.FAIL)

    def test_merge_with_shape_mismatch_falls_back(self, temp_repo):
        """Test merge with incompatible shapes falls back to OURS."""
        # Initial commit
        weight = create_weight_tensor(np.ones(10), "weight")
        temp_repo.stage_weights({"weight": weight})
        temp_repo.commit("Initial")

        # Create feature with different shape
        temp_repo.create_branch("feature")
        temp_repo.checkout("feature")
        feature_weight = create_weight_tensor(np.ones(20), "weight")  # Different shape
        temp_repo.stage_weights({"weight": feature_weight})
        temp_repo.commit("Feature with different shape")

        # Main with original shape but different values
        temp_repo.checkout("main")
        main_weight = create_weight_tensor(np.ones(10) * 5, "weight")
        temp_repo.stage_weights({"weight": main_weight})
        temp_repo.commit("Main change")

        # Merge with AVERAGE - should fall back due to shape mismatch
        merge_commit = temp_repo.merge("feature", strategy=MergeStrategy.AVERAGE)

        assert merge_commit is not None

        # Should use main's weight due to shape mismatch
        weight = temp_repo.get_weight("weight")
        assert weight.shape == (10,)

    def test_merge_new_weight_only_on_source(self, temp_repo):
        """Test merge when weight only exists on source branch."""
        # Initial commit
        weight1 = create_weight_tensor(np.ones(10), "weight1")
        temp_repo.stage_weights({"weight1": weight1})
        temp_repo.commit("Initial")

        # Create feature with additional weight
        temp_repo.create_branch("feature")
        temp_repo.checkout("feature")
        weight2 = create_weight_tensor(np.ones(5) * 2, "weight2")
        temp_repo.stage_weights({"weight1": weight1, "weight2": weight2})
        temp_repo.commit("Add weight2")

        # Back to main (no weight2)
        temp_repo.checkout("main")

        # Merge should add weight2
        merge_commit = temp_repo.merge("feature", strategy=MergeStrategy.AVERAGE)

        assert merge_commit is not None
        weight2_merged = temp_repo.get_weight("weight2")
        assert weight2_merged is not None
        np.testing.assert_array_almost_equal(weight2_merged.data, np.ones(5) * 2)


class TestRemoteOperations:
    """Tests for remote repository operations."""

    @pytest.fixture
    def temp_repo(self):
        """Create a temporary repository."""
        temp_dir = tempfile.mkdtemp()
        repo = Repository(temp_dir, init=True)
        yield repo
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def remote_dir(self):
        """Create a temporary directory for remote."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_add_remote(self, temp_repo, remote_dir):
        """Test adding a remote."""
        temp_repo.add_remote("origin", f"file://{remote_dir}")

        remotes = temp_repo.list_remotes()
        assert "origin" in remotes
        assert remotes["origin"]["url"] == f"file://{remote_dir}"

    def test_get_remote(self, temp_repo, remote_dir):
        """Test getting a remote."""
        temp_repo.add_remote("origin", f"file://{remote_dir}")

        remote = temp_repo.get_remote("origin")
        assert remote is not None
        assert remote["url"] == f"file://{remote_dir}"

    def test_get_nonexistent_remote(self, temp_repo):
        """Test getting a non-existent remote."""
        remote = temp_repo.get_remote("nonexistent")
        assert remote is None

    def test_remove_remote(self, temp_repo, remote_dir):
        """Test removing a remote."""
        temp_repo.add_remote("origin", f"file://{remote_dir}")
        temp_repo.remove_remote("origin")

        remotes = temp_repo.list_remotes()
        assert "origin" not in remotes

    def test_remove_nonexistent_remote_raises(self, temp_repo):
        """Test removing non-existent remote raises error."""
        with pytest.raises(ValueError, match="not found"):
            temp_repo.remove_remote("nonexistent")

    def test_list_remotes_empty(self, temp_repo):
        """Test listing remotes when none exist."""
        remotes = temp_repo.list_remotes()
        assert remotes == {}

    def test_add_multiple_remotes(self, temp_repo, remote_dir):
        """Test adding multiple remotes."""
        temp_repo.add_remote("origin", f"file://{remote_dir}/origin")
        temp_repo.add_remote("backup", f"file://{remote_dir}/backup")

        remotes = temp_repo.list_remotes()
        assert len(remotes) == 2
        assert "origin" in remotes
        assert "backup" in remotes


class TestRepositoryDiff:
    """Tests for diff operations."""

    @pytest.fixture
    def temp_repo(self):
        """Create a temporary repository."""
        temp_dir = tempfile.mkdtemp()
        repo = Repository(temp_dir, init=True)
        yield repo
        shutil.rmtree(temp_dir)

    def test_diff_between_commits(self, temp_repo):
        """Test diff between two commits."""
        # First commit
        weight1 = create_weight_tensor(np.ones(10), "weight1")
        temp_repo.stage_weights({"weight1": weight1})
        commit1 = temp_repo.commit("First")

        # Second commit with modifications
        weight1_modified = create_weight_tensor(np.ones(10) * 2, "weight1")
        weight2 = create_weight_tensor(np.ones(5), "weight2")
        temp_repo.stage_weights({"weight1": weight1_modified, "weight2": weight2})
        commit2 = temp_repo.commit("Second")

        # Diff between commits using full hashes
        diff = temp_repo.diff(commit1.commit_hash, commit2.commit_hash)

        assert "weight1" in diff["modified"]
        assert "weight2" in diff["added"]

    def test_diff_with_head(self, temp_repo):
        """Test diff from commit to HEAD."""
        # First commit
        weight = create_weight_tensor(np.ones(10), "weight")
        temp_repo.stage_weights({"weight": weight})
        commit1 = temp_repo.commit("First")

        # Second commit
        weight2 = create_weight_tensor(np.ones(5), "weight2")
        temp_repo.stage_weights({"weight": weight, "weight2": weight2})
        temp_repo.commit("Second")

        # Diff from first commit to current HEAD
        diff = temp_repo.diff(commit1.commit_hash)

        assert "weight2" in diff["added"]


class TestRepositoryGetAllWeights:
    """Tests for get_all_weights method."""

    @pytest.fixture
    def temp_repo(self):
        """Create a temporary repository."""
        temp_dir = tempfile.mkdtemp()
        repo = Repository(temp_dir, init=True)
        yield repo
        shutil.rmtree(temp_dir)

    def test_get_all_weights(self, temp_repo):
        """Test getting all weights from current commit."""
        weight1 = create_weight_tensor(np.ones(10), "weight1")
        weight2 = create_weight_tensor(np.ones(5), "weight2")

        temp_repo.stage_weights({"weight1": weight1, "weight2": weight2})
        temp_repo.commit("Add weights")

        weights = temp_repo.get_all_weights()

        assert len(weights) == 2
        assert "weight1" in weights
        assert "weight2" in weights
        np.testing.assert_array_equal(weights["weight1"].data, np.ones(10))

    def test_get_all_weights_from_commit(self, temp_repo):
        """Test getting all weights from specific commit."""
        weight1 = create_weight_tensor(np.ones(10), "weight1")
        temp_repo.stage_weights({"weight1": weight1})
        commit1 = temp_repo.commit("First")

        weight2 = create_weight_tensor(np.ones(5), "weight2")
        temp_repo.stage_weights({"weight1": weight1, "weight2": weight2})
        temp_repo.commit("Second")

        # Get weights from first commit
        weights = temp_repo.get_all_weights(commit_ref=commit1.commit_hash)

        assert len(weights) == 1
        assert "weight1" in weights

    def test_get_all_weights_empty(self, temp_repo):
        """Test getting weights when none committed."""
        weights = temp_repo.get_all_weights()
        assert weights == {}
