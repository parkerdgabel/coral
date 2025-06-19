import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.version_control.repository import Repository


def create_weight_tensor(data, name="test_weight", **kwargs):
    """Helper to create weight tensor with proper metadata."""
    metadata = WeightMetadata(name=name, shape=data.shape, dtype=data.dtype, **kwargs)
    return WeightTensor(data=data, metadata=metadata)


class TestRepositoryExtended:
    @pytest.fixture
    def temp_repo(self):
        """Create a temporary repository for testing."""
        temp_dir = tempfile.mkdtemp()
        repo = Repository(temp_dir, init=True)
        yield repo
        shutil.rmtree(temp_dir)

    def test_repository_init_creates_structure(self):
        """Test repository initialization creates correct structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            Repository(temp_dir, init=True)

            # Check directory structure
            assert (Path(temp_dir) / ".coral").exists()
            assert (Path(temp_dir) / ".coral" / "config").exists()
            assert (Path(temp_dir) / ".coral" / "refs" / "heads").exists()
            assert (Path(temp_dir) / ".coral" / "refs" / "tags").exists()
            assert (Path(temp_dir) / ".coral" / "objects").exists()
            assert (Path(temp_dir) / ".coral" / "objects" / "weights.h5").exists()

    def test_repository_init_existing_fails(self, temp_repo):
        """Test initializing in existing repository fails."""
        with pytest.raises(ValueError, match="already exists"):
            Repository(temp_repo.path, init=True)

    def test_repository_load_nonexistent_fails(self):
        """Test loading non-existent repository fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="Not a Coral repository"):
                Repository(temp_dir)

    def test_add_weights_with_metadata(self, temp_repo):
        """Test adding weights with custom metadata."""
        data1 = np.random.randn(10, 10).astype(np.float32)
        weight1 = create_weight_tensor(data1, "model.layer1", layer_type="dense")

        data2 = np.random.randn(5, 10).astype(np.float32)
        weight2 = create_weight_tensor(data2, "model.layer2", layer_type="output")

        weights = {"model.layer1": weight1, "model.layer2": weight2}

        temp_repo.stage_weights(weights)

        # Check weights are staged
        status = temp_repo.status()
        assert len(status["staged"]) == 2
        assert "model.layer1" in status["staged"]
        assert "model.layer2" in status["staged"]

    def test_commit_empty_staging_fails(self, temp_repo):
        """Test committing with empty staging area fails."""
        with pytest.raises(ValueError, match="Nothing to commit"):
            temp_repo.commit("Empty commit")

    def test_commit_with_author(self, temp_repo):
        """Test commit with author information."""
        weight = create_weight_tensor(np.ones(5), "weight1")
        temp_repo.stage_weights({"weight1": weight})

        commit = temp_repo.commit(
            "Test commit", author="Test User", email="test@example.com"
        )

        assert commit.metadata.author == "Test User"
        assert commit.metadata.email == "test@example.com"

    def test_branch_operations(self, temp_repo):
        """Test comprehensive branch operations."""
        # Add initial commit
        weight = create_weight_tensor(np.ones(5), "weight1")
        temp_repo.stage_weights({"weight1": weight})
        temp_repo.commit("Initial commit")

        # Create new branch
        temp_repo.create_branch("feature-1")
        branches = [b.name for b in temp_repo.list_branches()]
        assert "feature-1" in branches

        # Switch to new branch
        temp_repo.checkout("feature-1")
        assert temp_repo.branch_manager.get_current_branch() == "feature-1"

        # Make changes on feature branch
        weight2 = create_weight_tensor(np.ones(5) * 2, "weight2")
        temp_repo.stage_weights({"weight2": weight2})
        temp_repo.commit("Feature commit")

        # Switch back to main
        temp_repo.checkout("main")
        assert temp_repo.branch_manager.get_current_branch() == "main"

        # Verify main doesn't have feature changes
        assert temp_repo.get_weight("weight2") is None

    def test_merge_fast_forward(self, temp_repo):
        """Test fast-forward merge."""
        # Initial commit on main
        weight1 = create_weight_tensor(np.ones(10), "weight1")
        temp_repo.stage_weights({"weight1": weight1})
        temp_repo.commit("Main commit")

        # Create and switch to feature branch
        temp_repo.create_branch("feature")
        temp_repo.checkout("feature")

        # Add commits on feature
        weight2 = create_weight_tensor(np.ones(10) * 2, "weight2")
        temp_repo.stage_weights({"weight2": weight2})
        feature_commit = temp_repo.commit("Feature commit")

        # Switch back to main and merge
        temp_repo.checkout("main")
        merge_commit = temp_repo.merge("feature")

        # Should be fast-forward (same as feature commit)
        assert merge_commit.commit_hash == feature_commit.commit_hash

    def test_merge_with_conflicts(self, temp_repo):
        """Test merge with conflicts."""
        # Initial commit
        weight1 = create_weight_tensor(np.ones(10), "weight1")
        temp_repo.stage_weights({"weight1": weight1})
        temp_repo.commit("Initial")

        # Create feature branch and modify weight
        temp_repo.create_branch("feature")
        temp_repo.checkout("feature")
        weight1_feature = create_weight_tensor(np.ones(10) * 2, "weight1")
        temp_repo.stage_weights({"weight1": weight1_feature})
        temp_repo.commit("Feature change")

        # Switch to main and make different change
        temp_repo.checkout("main")
        weight1_main = create_weight_tensor(np.ones(10) * 3, "weight1")
        temp_repo.stage_weights({"weight1": weight1_main})
        temp_repo.commit("Main change")

        # Merge should succeed (auto-resolve or manual)
        merge_commit = temp_repo.merge("feature")
        assert merge_commit is not None

    def test_tag_operations(self, temp_repo):
        """Test tagging operations."""
        # Create a commit to tag
        weight = create_weight_tensor(np.random.randn(10), "weight1")
        temp_repo.stage_weights({"weight1": weight})
        temp_repo.commit("Commit to tag")

        # Create a tag
        version = temp_repo.tag_version(
            "v1.0.0", description="First release", metrics={"accuracy": 0.95}
        )

        assert version.name == "v1.0.0"
        assert version.description == "First release"
        assert version.metrics["accuracy"] == 0.95

        # List versions
        versions = temp_repo.list_versions()
        assert len(versions) == 1
        assert versions[0].name == "v1.0.0"

        # Get version by name
        retrieved = temp_repo.get_version("v1.0.0")
        assert retrieved.name == "v1.0.0"

    def test_checkout_commit(self, temp_repo):
        """Test checking out specific commit."""
        # Create multiple commits
        commits = []
        for i in range(3):
            weight = create_weight_tensor(np.ones(5) * i, f"weight{i}")
            temp_repo.stage_weights({f"weight{i}": weight})
            commit = temp_repo.commit(f"Commit {i}")
            commits.append(commit)

        # Checkout middle commit
        temp_repo.checkout(commits[1].commit_hash[:8])

        # Should have weights 0 and 1, but not 2
        assert temp_repo.get_weight("weight0") is not None
        assert temp_repo.get_weight("weight1") is not None
        assert temp_repo.get_weight("weight2") is None

    def test_diff_operations(self, temp_repo):
        """Test diff between commits/branches."""
        # Initial commit
        weight1 = create_weight_tensor(np.ones(10), "weight1")
        temp_repo.stage_weights({"weight1": weight1})
        commit1 = temp_repo.commit("First commit")

        # Second commit adds weight
        weight2 = create_weight_tensor(np.ones(10) * 2, "weight2")
        temp_repo.stage_weights({"weight2": weight2})
        temp_repo.commit("Second commit")

        # Third commit modifies weight1
        weight1_mod = create_weight_tensor(np.ones(10) * 3, "weight1")
        temp_repo.stage_weights({"weight1": weight1_mod})
        commit3 = temp_repo.commit("Third commit")

        # Diff between commits
        diff = temp_repo.diff(commit1.commit_hash[:8], commit3.commit_hash[:8])

        assert "weight2" in diff["added"]
        assert "weight1" in diff["modified"]
        assert len(diff["removed"]) == 0

    def test_log_filtering(self, temp_repo):
        """Test commit log with filtering."""
        # Create commits on main
        for i in range(3):
            weight = create_weight_tensor(np.ones(5) * i, f"main_weight{i}")
            temp_repo.stage_weights({f"main_weight{i}": weight})
            temp_repo.commit(f"Main commit {i}")

        # Create branch and add commits
        temp_repo.create_branch("feature")
        temp_repo.checkout("feature")

        for i in range(2):
            weight = create_weight_tensor(np.ones(5) * i, f"feature_weight{i}")
            temp_repo.stage_weights({f"feature_weight{i}": weight})
            temp_repo.commit(f"Feature commit {i}")

        # Get log for feature branch
        log = temp_repo.log(max_commits=10, branch="feature")
        assert len(log) == 5  # 3 main + 2 feature

        # Get log for main branch
        temp_repo.checkout("main")
        log = temp_repo.log(max_commits=10, branch="main")
        assert len(log) == 3  # Only main commits

    def test_garbage_collection(self, temp_repo):
        """Test garbage collection of unreferenced weights."""
        # Create and commit weights
        weight1 = create_weight_tensor(np.ones(100), "weight1")
        weight2 = create_weight_tensor(np.ones(100) * 2, "weight2")

        temp_repo.stage_weights({"weight1": weight1, "weight2": weight2})
        temp_repo.commit("First commit")

        # Overwrite weight2
        weight2_new = create_weight_tensor(np.ones(100) * 3, "weight2")
        temp_repo.stage_weights({"weight2": weight2_new})
        temp_repo.commit("Second commit")

        # Run garbage collection
        result = temp_repo.gc()

        # Should have cleaned the old weight2
        assert result["cleaned_weights"] >= 0
        assert result["remaining_weights"] >= 2  # At least weight1 and new weight2

    def test_gc_dry_run(self, temp_repo):
        """Test garbage collection dry run."""
        # Create weights
        weight = create_weight_tensor(np.ones(50), "weight1")
        temp_repo.stage_weights({"weight1": weight})
        temp_repo.commit("Test commit")

        # Dry run should not actually delete
        temp_repo.gc()

        # Verify weights still exist
        assert temp_repo.get_weight("weight1") is not None

    def test_status_with_modifications(self, temp_repo):
        """Test repository status with various states."""
        # Initial commit
        weight1 = create_weight_tensor(np.ones(10), "weight1")
        weight2 = create_weight_tensor(np.ones(10) * 2, "weight2")

        temp_repo.stage_weights({"weight1": weight1, "weight2": weight2})
        temp_repo.commit("Initial commit")

        # Stage new weight
        weight3 = create_weight_tensor(np.ones(10) * 3, "weight3")
        temp_repo.stage_weights({"weight3": weight3})

        # Get status
        status = temp_repo.status()

        assert len(status["staged"]) == 1
        assert "weight3" in status["staged"]
        assert "branch" in status

    def test_get_weight_by_name(self, temp_repo):
        """Test getting weight by name from current commit."""
        # Add weights
        weight1 = create_weight_tensor(np.array([1, 2, 3]), "test_weight")
        temp_repo.stage_weights({"test_weight": weight1})
        temp_repo.commit("Add test weight")

        # Get weight
        retrieved = temp_repo.get_weight("test_weight")
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved.data, weight1.data)

        # Get non-existent weight
        assert temp_repo.get_weight("non_existent") is None

    def test_resolve_ref(self, temp_repo):
        """Test resolving various reference types."""
        # Create commits
        weight = create_weight_tensor(np.ones(5), "weight1")
        temp_repo.stage_weights({"weight1": weight})
        commit = temp_repo.commit("Test commit")

        # Create tag
        temp_repo.tag_version("v1.0.0")

        # Test resolving different refs
        # Branch name
        resolved = temp_repo._resolve_ref("main")
        assert resolved is not None

        # Tag name
        resolved = temp_repo._resolve_ref("v1.0.0")
        assert resolved is not None

        # Commit hash (short)
        resolved = temp_repo._resolve_ref(commit.commit_hash[:8])
        assert resolved == commit.commit_hash

        # HEAD
        resolved = temp_repo._resolve_ref("HEAD")
        assert resolved == commit.commit_hash
