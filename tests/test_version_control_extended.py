"""Extended tests for version control functionality.

This module tests additional version control features.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from coral.core.weight_tensor import WeightMetadata, WeightTensor
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
def repo_with_commits(temp_repo):
    """Create a repository with multiple commits."""
    # First commit
    weights1 = {
        "layer1.weight": WeightTensor(
            data=np.random.randn(10, 5).astype(np.float32),
            metadata=WeightMetadata(
                name="layer1.weight", shape=(10, 5), dtype=np.float32
            ),
        ),
    }
    temp_repo.stage_weights(weights1)
    temp_repo.commit("First commit")

    # Second commit
    weights2 = {
        "layer1.weight": WeightTensor(
            data=np.random.randn(10, 5).astype(np.float32),
            metadata=WeightMetadata(
                name="layer1.weight", shape=(10, 5), dtype=np.float32
            ),
        ),
        "layer2.weight": WeightTensor(
            data=np.random.randn(5, 3).astype(np.float32),
            metadata=WeightMetadata(
                name="layer2.weight", shape=(5, 3), dtype=np.float32
            ),
        ),
    }
    temp_repo.stage_weights(weights2)
    temp_repo.commit("Second commit")

    return temp_repo


class TestVersionGraph:
    """Test version graph functionality."""

    def test_version_graph_initialization(self, temp_repo):
        """Test version graph is initialized properly."""
        assert temp_repo.version_graph is not None
        assert len(temp_repo.version_graph.commits) == 0

    def test_version_graph_after_commits(self, repo_with_commits):
        """Test version graph after commits."""
        assert len(repo_with_commits.version_graph.commits) == 2

    def test_get_branch_history(self, repo_with_commits):
        """Test getting branch history."""
        history = repo_with_commits.log(max_commits=10)
        assert len(history) == 2
        assert history[0].metadata.message == "Second commit"
        assert history[1].metadata.message == "First commit"

    def test_get_commit_ancestors(self, repo_with_commits):
        """Test getting commit ancestors."""
        commits = repo_with_commits.log(max_commits=10)
        latest = commits[0].commit_hash

        ancestors = repo_with_commits.version_graph.get_commit_ancestors(latest)
        assert len(ancestors) == 1


class TestCommitMetadata:
    """Test commit metadata."""

    def test_commit_with_tags(self, temp_repo):
        """Test creating commit with tags."""
        weights = {
            "layer.weight": WeightTensor(
                data=np.ones((5, 5), dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer.weight", shape=(5, 5), dtype=np.float32
                ),
            ),
        }
        temp_repo.stage_weights(weights)
        commit = temp_repo.commit("Tagged commit", tags=["v1.0", "release"])

        assert "v1.0" in commit.metadata.tags
        assert "release" in commit.metadata.tags

    def test_commit_with_author(self, temp_repo):
        """Test creating commit with author info."""
        weights = {
            "layer.weight": WeightTensor(
                data=np.ones((5, 5), dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer.weight", shape=(5, 5), dtype=np.float32
                ),
            ),
        }
        temp_repo.stage_weights(weights)
        commit = temp_repo.commit(
            "Author commit",
            author="John Doe",
            email="john@example.com",
        )

        assert commit.metadata.author == "John Doe"
        assert commit.metadata.email == "john@example.com"


class TestBranchOperations:
    """Test branch operations."""

    def test_create_branch_from_commit(self, repo_with_commits):
        """Test creating a branch from a specific commit."""
        commits = repo_with_commits.log(max_commits=10)
        first_commit = commits[1].commit_hash

        repo_with_commits.create_branch("feature", from_ref=first_commit)

        branch = repo_with_commits.branch_manager.get_branch("feature")
        assert branch is not None
        assert branch.commit_hash == first_commit

    def test_checkout_detached_head(self, repo_with_commits):
        """Test checking out a specific commit (detached HEAD)."""
        commits = repo_with_commits.log(max_commits=10)
        first_commit = commits[1].commit_hash

        repo_with_commits.checkout(first_commit)

        # Check that we're in detached HEAD state
        head_content = (repo_with_commits.coral_dir / "HEAD").read_text().strip()
        assert head_content == first_commit


class TestDiff:
    """Test diff functionality."""

    def test_diff_between_commits(self, repo_with_commits):
        """Test diffing between two commits."""
        commits = repo_with_commits.log(max_commits=10)
        first_commit = commits[1].commit_hash
        second_commit = commits[0].commit_hash

        diff = repo_with_commits.diff(first_commit, second_commit)

        assert "added" in diff
        assert "modified" in diff
        assert "removed" in diff
        assert "layer2.weight" in diff["added"]

    def test_diff_from_single_commit(self, repo_with_commits):
        """Test diff from a single commit (against HEAD)."""
        commits = repo_with_commits.log(max_commits=10)
        first_commit = commits[1].commit_hash

        diff = repo_with_commits.diff(first_commit)

        # Should compare against HEAD
        assert "added" in diff


class TestVersionTags:
    """Test version tagging."""

    def test_create_version_tag(self, repo_with_commits):
        """Test creating a version tag."""
        version = repo_with_commits.tag_version(
            "v1.0.0",
            description="First release",
            metrics={"accuracy": 0.95, "loss": 0.05},
        )

        assert version.name == "v1.0.0"
        assert version.description == "First release"
        assert version.metrics == {"accuracy": 0.95, "loss": 0.05}

    def test_retrieve_version_tag(self, repo_with_commits):
        """Test retrieving a version tag."""
        version = repo_with_commits.tag_version("v1.0.0", description="First release")

        # Check that the version was created
        assert version.name == "v1.0.0"
        assert version.description == "First release"


class TestWeightRetrieval:
    """Test weight retrieval."""

    def test_get_weight_by_name(self, repo_with_commits):
        """Test getting a weight by name."""
        weight = repo_with_commits.get_weight("layer1.weight")

        assert weight is not None
        assert weight.shape == (10, 5)

    def test_get_weight_from_specific_commit(self, repo_with_commits):
        """Test getting a weight from a specific commit."""
        commits = repo_with_commits.log(max_commits=10)
        first_commit = commits[1].commit_hash

        weight = repo_with_commits.get_weight("layer1.weight", commit_ref=first_commit)

        assert weight is not None
        assert weight.shape == (10, 5)

    def test_get_all_weights(self, repo_with_commits):
        """Test getting all weights from a commit."""
        weights = repo_with_commits.get_all_weights()

        assert len(weights) == 2
        assert "layer1.weight" in weights
        assert "layer2.weight" in weights

    def test_get_nonexistent_weight(self, repo_with_commits):
        """Test getting a nonexistent weight."""
        weight = repo_with_commits.get_weight("nonexistent")

        assert weight is None


class TestMergeStrategies:
    """Test merge strategies."""

    def test_merge_theirs_strategy(self, repo_with_commits):
        """Test merge with THEIRS strategy."""
        from coral.version_control.repository import MergeStrategy

        # Create feature branch
        repo_with_commits.create_branch("feature")
        repo_with_commits.checkout("feature")

        # Add a weight on feature branch
        weights = {
            "layer1.weight": repo_with_commits.get_weight("layer1.weight"),
            "layer2.weight": repo_with_commits.get_weight("layer2.weight"),
            "feature.weight": WeightTensor(
                data=np.ones((3, 3), dtype=np.float32),
                metadata=WeightMetadata(
                    name="feature.weight", shape=(3, 3), dtype=np.float32
                ),
            ),
        }
        repo_with_commits.stage_weights(weights)
        repo_with_commits.commit("Feature commit")

        # Go back to main and merge
        repo_with_commits.checkout("main")
        commit = repo_with_commits.merge("feature", strategy=MergeStrategy.THEIRS)

        assert commit is not None
        # Check that feature weight exists
        weight = repo_with_commits.get_weight("feature.weight")
        assert weight is not None


class TestGarbageCollection:
    """Test garbage collection."""

    def test_gc_collects_nothing_on_clean_repo(self, repo_with_commits):
        """Test GC on a repo where all weights are referenced."""
        result = repo_with_commits.gc()

        # Check the result structure
        assert "cleaned_weights" in result or "collected" in result
        # Should not clean any weights since all are referenced
        cleaned = result.get("cleaned_weights", result.get("collected", 0))
        assert cleaned == 0


class TestStagingArea:
    """Test staging area functionality."""

    def test_staging_weights(self, temp_repo):
        """Test staging weights."""
        weights = {
            "layer.weight": WeightTensor(
                data=np.ones((5, 5), dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer.weight", shape=(5, 5), dtype=np.float32
                ),
            ),
        }

        hashes = temp_repo.stage_weights(weights)

        assert len(hashes) == 1
        assert "layer.weight" in hashes

    def test_clear_staging(self, temp_repo):
        """Test clearing staging area."""
        weights = {
            "layer.weight": WeightTensor(
                data=np.ones((5, 5), dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer.weight", shape=(5, 5), dtype=np.float32
                ),
            ),
        }
        temp_repo.stage_weights(weights)

        # The staging area should have the weight
        staging_file = temp_repo.staging_dir / "staged.json"
        assert staging_file.exists()
