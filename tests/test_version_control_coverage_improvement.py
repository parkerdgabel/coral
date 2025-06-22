"""Test coverage improvement for version control module."""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coral.version_control.repository import Repository, RepositoryError
from coral.version_control.commit import Commit
from coral.version_control.branch import Branch, BranchManager
from coral.version_control.version import Version, VersionGraph
from coral.storage.hdf5_store import HDF5Store
from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.core.deduplicator import Deduplicator


class TestRepositoryCoverage:
    """Tests to improve coverage for Repository class."""

    @pytest.fixture
    def repo_path(self):
        """Create a temporary repository path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def repository(self, repo_path):
        """Create a repository instance."""
        repo = Repository(repo_path)
        repo.init()
        return repo

    def test_repository_init_with_existing_repo(self, repo_path):
        """Test initializing in an already initialized repository."""
        # First init
        repo1 = Repository(repo_path)
        repo1.init()
        
        # Second init should raise error
        repo2 = Repository(repo_path)
        with pytest.raises(RepositoryError, match="already initialized"):
            repo2.init()

    def test_repository_load(self, repo_path):
        """Test loading an existing repository."""
        # Initialize first
        repo1 = Repository(repo_path)
        repo1.init()
        
        # Stage and commit some weights
        weight = WeightTensor(data=np.array([1, 2, 3], dtype=np.float32))
        repo1.stage_weight("test_weight", weight)
        repo1.commit("Initial commit")
        
        # Load in new instance
        repo2 = Repository(repo_path)
        repo2.load()
        
        # Should have same state
        assert repo2.current_branch == repo1.current_branch
        assert len(repo2.list_commits()) == 1

    def test_repository_load_non_existent(self, repo_path):
        """Test loading from non-existent repository."""
        repo = Repository(repo_path)
        with pytest.raises(RepositoryError, match="No repository found"):
            repo.load()

    def test_stage_weight_with_metadata(self, repository):
        """Test staging weights with custom metadata."""
        weight = WeightTensor(
            data=np.array([[1, 2], [3, 4]], dtype=np.float32),
            metadata=WeightMetadata(
                name="layer1_weight",
                shape=(2, 2),
                dtype="float32",
                tags=["conv", "layer1"]
            )
        )
        
        repository.stage_weight("layer1/weight", weight)
        
        # Check staged
        assert "layer1/weight" in repository._staged_weights
        staged_weight = repository._staged_weights["layer1/weight"]
        assert staged_weight.metadata.name == "layer1_weight"
        assert "conv" in staged_weight.metadata.tags

    def test_unstage_weight(self, repository):
        """Test unstaging weights."""
        # Stage multiple weights
        weight1 = WeightTensor(data=np.array([1, 2], dtype=np.float32))
        weight2 = WeightTensor(data=np.array([3, 4], dtype=np.float32))
        
        repository.stage_weight("weight1", weight1)
        repository.stage_weight("weight2", weight2)
        
        # Unstage one
        repository.unstage_weight("weight1")
        
        assert "weight1" not in repository._staged_weights
        assert "weight2" in repository._staged_weights

    def test_commit_empty_staging(self, repository):
        """Test committing with empty staging area."""
        with pytest.raises(RepositoryError, match="No weights staged"):
            repository.commit("Empty commit")

    def test_commit_with_author_and_timestamp(self, repository):
        """Test commit with custom author and timestamp."""
        weight = WeightTensor(data=np.array([1], dtype=np.float32))
        repository.stage_weight("test", weight)
        
        timestamp = datetime.now().isoformat()
        commit_hash = repository.commit(
            message="Test commit",
            author="Test Author <test@example.com>",
            timestamp=timestamp
        )
        
        # Get commit details
        commit = repository.get_commit(commit_hash)
        assert commit.author == "Test Author <test@example.com>"
        assert commit.timestamp == timestamp

    def test_get_weight_from_commit(self, repository):
        """Test retrieving weight from specific commit."""
        # Create multiple commits
        weight1 = WeightTensor(data=np.array([1, 2], dtype=np.float32))
        repository.stage_weight("weight", weight1)
        commit1 = repository.commit("First version")
        
        weight2 = WeightTensor(data=np.array([3, 4], dtype=np.float32))
        repository.stage_weight("weight", weight2)
        commit2 = repository.commit("Second version")
        
        # Get weights from different commits
        retrieved1 = repository.get_weight("weight", commit=commit1)
        retrieved2 = repository.get_weight("weight", commit=commit2)
        
        assert np.array_equal(retrieved1.data, weight1.data)
        assert np.array_equal(retrieved2.data, weight2.data)

    def test_get_weight_non_existent(self, repository):
        """Test getting non-existent weight."""
        with pytest.raises(RepositoryError, match="Weight .* not found"):
            repository.get_weight("non_existent")

    def test_list_weights(self, repository):
        """Test listing weights in repository."""
        # Add weights
        weights = {
            "model/layer1/weight": np.array([1, 2]),
            "model/layer1/bias": np.array([3]),
            "model/layer2/weight": np.array([4, 5, 6]),
        }
        
        for name, data in weights.items():
            repository.stage_weight(name, WeightTensor(data=data.astype(np.float32)))
        repository.commit("Add model weights")
        
        # List all
        all_weights = repository.list_weights()
        assert len(all_weights) == 3
        for name in weights:
            assert name in all_weights
        
        # List with prefix
        layer1_weights = repository.list_weights(prefix="model/layer1")
        assert len(layer1_weights) == 2
        assert "model/layer1/weight" in layer1_weights
        assert "model/layer1/bias" in layer1_weights

    def test_checkout_commit(self, repository):
        """Test checking out specific commits."""
        # Create commits
        weight1 = WeightTensor(data=np.array([1], dtype=np.float32))
        repository.stage_weight("w", weight1)
        commit1 = repository.commit("v1")
        
        weight2 = WeightTensor(data=np.array([2], dtype=np.float32))
        repository.stage_weight("w", weight2)
        commit2 = repository.commit("v2")
        
        # Checkout first commit
        repository.checkout(commit1)
        w = repository.get_weight("w")
        assert np.array_equal(w.data, weight1.data)
        
        # Checkout second commit
        repository.checkout(commit2)
        w = repository.get_weight("w")
        assert np.array_equal(w.data, weight2.data)

    def test_checkout_detached_head(self, repository):
        """Test checkout resulting in detached HEAD state."""
        # Create branch with commits
        weight = WeightTensor(data=np.array([1], dtype=np.float32))
        repository.stage_weight("w", weight)
        commit = repository.commit("test")
        
        # Checkout commit directly (not branch)
        repository.checkout(commit)
        
        # Should be in detached HEAD state
        assert repository._head == commit
        assert repository.current_branch is None

    def test_branch_operations(self, repository):
        """Test comprehensive branch operations."""
        # Create initial commit
        weight = WeightTensor(data=np.array([1], dtype=np.float32))
        repository.stage_weight("w", weight)
        repository.commit("Initial")
        
        # Create new branch
        repository.create_branch("feature")
        assert "feature" in repository.list_branches()
        
        # Switch to new branch
        repository.checkout("feature")
        assert repository.current_branch == "feature"
        
        # Make changes on feature branch
        weight2 = WeightTensor(data=np.array([2], dtype=np.float32))
        repository.stage_weight("w", weight2)
        repository.commit("Feature change")
        
        # Switch back to main
        repository.checkout("main")
        w = repository.get_weight("w")
        assert np.array_equal(w.data, np.array([1]))
        
        # Delete feature branch
        repository.delete_branch("feature")
        assert "feature" not in repository.list_branches()

    def test_delete_current_branch(self, repository):
        """Test deleting current branch (should fail)."""
        with pytest.raises(RepositoryError, match="Cannot delete current branch"):
            repository.delete_branch("main")

    def test_merge_fast_forward(self, repository):
        """Test fast-forward merge."""
        # Setup: main with one commit
        weight1 = WeightTensor(data=np.array([1], dtype=np.float32))
        repository.stage_weight("w", weight1)
        base_commit = repository.commit("Base")
        
        # Create feature branch and add commit
        repository.create_branch("feature")
        repository.checkout("feature")
        weight2 = WeightTensor(data=np.array([2], dtype=np.float32))
        repository.stage_weight("w", weight2)
        feature_commit = repository.commit("Feature")
        
        # Merge into main (should fast-forward)
        repository.checkout("main")
        merge_commit = repository.merge("feature")
        
        # Should fast-forward (merge_commit equals feature_commit)
        assert merge_commit == feature_commit
        w = repository.get_weight("w")
        assert np.array_equal(w.data, weight2.data)

    def test_merge_with_conflicts(self, repository):
        """Test merge with conflicts."""
        # Base commit
        weight = WeightTensor(data=np.array([1], dtype=np.float32))
        repository.stage_weight("w", weight)
        repository.commit("Base")
        
        # Branch 1 changes
        repository.create_branch("branch1")
        repository.checkout("branch1")
        weight1 = WeightTensor(data=np.array([2], dtype=np.float32))
        repository.stage_weight("w", weight1)
        repository.commit("Branch1 change")
        
        # Branch 2 changes (conflicting)
        repository.checkout("main")
        repository.create_branch("branch2")
        repository.checkout("branch2")
        weight2 = WeightTensor(data=np.array([3], dtype=np.float32))
        repository.stage_weight("w", weight2)
        repository.commit("Branch2 change")
        
        # Try to merge - should detect conflict
        repository.checkout("main")
        repository.merge("branch1")  # First merge succeeds
        
        # Second merge should conflict
        with pytest.raises(RepositoryError, match="conflict"):
            repository.merge("branch2")

    def test_diff_between_commits(self, repository):
        """Test diff functionality between commits."""
        # First commit
        weights1 = {
            "a": WeightTensor(data=np.array([1], dtype=np.float32)),
            "b": WeightTensor(data=np.array([2], dtype=np.float32)),
        }
        for name, w in weights1.items():
            repository.stage_weight(name, w)
        commit1 = repository.commit("First")
        
        # Second commit - modify and add
        weights2 = {
            "a": WeightTensor(data=np.array([10], dtype=np.float32)),  # modified
            "c": WeightTensor(data=np.array([3], dtype=np.float32)),   # added
        }
        for name, w in weights2.items():
            repository.stage_weight(name, w)
        commit2 = repository.commit("Second")
        
        # Get diff
        diff = repository.diff(commit1, commit2)
        
        assert "a" in diff.modified
        assert "b" in diff.removed
        assert "c" in diff.added

    def test_diff_with_staging(self, repository):
        """Test diff including staged changes."""
        # Initial commit
        weight = WeightTensor(data=np.array([1], dtype=np.float32))
        repository.stage_weight("w", weight)
        repository.commit("Initial")
        
        # Stage new changes
        weight2 = WeightTensor(data=np.array([2], dtype=np.float32))
        repository.stage_weight("w", weight2)
        repository.stage_weight("new", weight2)
        
        # Diff against staging
        diff = repository.diff()
        
        assert "w" in diff.modified
        assert "new" in diff.added

    def test_tagging(self, repository):
        """Test tag operations."""
        # Create commits
        weight = WeightTensor(data=np.array([1], dtype=np.float32))
        repository.stage_weight("w", weight)
        commit = repository.commit("Release candidate")
        
        # Create tag
        repository.tag(commit, "v1.0.0", "First release")
        
        # List tags
        tags = repository.list_tags()
        assert "v1.0.0" in tags
        
        # Get tag info
        tag_info = repository.get_tag("v1.0.0")
        assert tag_info["commit"] == commit
        assert tag_info["message"] == "First release"
        
        # Checkout tag
        repository.checkout("v1.0.0")
        assert repository._head == commit

    def test_duplicate_tag(self, repository):
        """Test creating duplicate tag."""
        weight = WeightTensor(data=np.array([1], dtype=np.float32))
        repository.stage_weight("w", weight)
        commit = repository.commit("Test")
        
        repository.tag(commit, "v1", "First")
        
        # Duplicate tag should fail
        with pytest.raises(RepositoryError, match="already exists"):
            repository.tag(commit, "v1", "Duplicate")

    def test_log_operations(self, repository):
        """Test log functionality."""
        # Create multiple commits
        for i in range(5):
            weight = WeightTensor(data=np.array([i], dtype=np.float32))
            repository.stage_weight("w", weight)
            repository.commit(f"Commit {i}")
        
        # Get full log
        log = repository.log()
        assert len(log) == 5
        
        # Get limited log
        log_limited = repository.log(max_entries=3)
        assert len(log_limited) == 3
        
        # Log from specific commit
        commits = repository.list_commits()
        log_from = repository.log(from_commit=commits[2])
        assert len(log_from) == 3  # commits 2, 1, 0

    def test_garbage_collection(self, repository):
        """Test garbage collection of unreferenced weights."""
        # Create commits then reset
        weight1 = WeightTensor(data=np.array([1], dtype=np.float32))
        repository.stage_weight("w", weight1)
        commit1 = repository.commit("To be collected")
        
        weight2 = WeightTensor(data=np.array([2], dtype=np.float32))
        repository.stage_weight("w", weight2)
        commit2 = repository.commit("Keep this")
        
        # Reset to first commit (second becomes unreferenced)
        repository.reset(commit1, hard=True)
        
        # Run garbage collection
        collected = repository.gc()
        
        # Should collect unreferenced weights
        assert collected > 0

    def test_reset_operations(self, repository):
        """Test reset functionality."""
        # Create base state
        weight1 = WeightTensor(data=np.array([1], dtype=np.float32))
        repository.stage_weight("w", weight1)
        commit1 = repository.commit("Base")
        
        # Make changes
        weight2 = WeightTensor(data=np.array([2], dtype=np.float32))
        repository.stage_weight("w", weight2)
        commit2 = repository.commit("Changes")
        
        # Soft reset - moves HEAD but keeps staging
        repository.stage_weight("new", weight2)
        repository.reset(commit1, hard=False)
        assert repository._head == commit1
        assert "new" in repository._staged_weights
        
        # Hard reset - moves HEAD and clears staging
        repository.stage_weight("another", weight2)
        repository.reset(commit1, hard=True)
        assert repository._head == commit1
        assert len(repository._staged_weights) == 0

    def test_status_information(self, repository):
        """Test repository status information."""
        # Clean status
        status = repository.status()
        assert status["branch"] == "main"
        assert status["clean"] is True
        assert len(status["staged"]) == 0
        
        # With staged changes
        weight = WeightTensor(data=np.array([1], dtype=np.float32))
        repository.stage_weight("w", weight)
        
        status = repository.status()
        assert status["clean"] is False
        assert "w" in status["staged"]
        assert status["head"] is None  # No commits yet

    def test_export_import(self, repository, tmp_path):
        """Test export and import functionality."""
        # Create repository with content
        weight = WeightTensor(data=np.array([1, 2, 3], dtype=np.float32))
        repository.stage_weight("test_weight", weight)
        repository.commit("Test commit")
        repository.tag(repository._head, "v1.0")
        
        # Export
        export_file = tmp_path / "export.coral"
        repository.export(str(export_file))
        assert export_file.exists()
        
        # Import into new repository
        new_repo_path = tmp_path / "new_repo"
        new_repo_path.mkdir()
        new_repo = Repository(str(new_repo_path))
        new_repo.init()
        
        new_repo.import_data(str(export_file))
        
        # Verify content
        assert "test_weight" in new_repo.list_weights()
        assert "v1.0" in new_repo.list_tags()

    def test_config_management(self, repository):
        """Test repository configuration."""
        # Set config values
        repository.set_config("user.name", "Test User")
        repository.set_config("user.email", "test@example.com")
        repository.set_config("core.compression", "gzip")
        
        # Get config values
        assert repository.get_config("user.name") == "Test User"
        assert repository.get_config("user.email") == "test@example.com"
        assert repository.get_config("core.compression") == "gzip"
        
        # Get non-existent config
        assert repository.get_config("non.existent") is None
        assert repository.get_config("non.existent", default="default") == "default"

    def test_hooks_system(self, repository):
        """Test repository hooks."""
        hook_data = {"pre_commit": False, "post_commit": False}
        
        def pre_commit_hook(repo, message):
            hook_data["pre_commit"] = True
            # Can modify or validate commit
            if "WIP" in message:
                raise RepositoryError("WIP commits not allowed")
        
        def post_commit_hook(repo, commit_hash):
            hook_data["post_commit"] = True
        
        # Register hooks
        repository.register_hook("pre_commit", pre_commit_hook)
        repository.register_hook("post_commit", post_commit_hook)
        
        # Trigger hooks with commit
        weight = WeightTensor(data=np.array([1], dtype=np.float32))
        repository.stage_weight("w", weight)
        repository.commit("Normal commit")
        
        assert hook_data["pre_commit"] is True
        assert hook_data["post_commit"] is True
        
        # Test hook rejection
        repository.stage_weight("w2", weight)
        with pytest.raises(RepositoryError, match="WIP commits not allowed"):
            repository.commit("WIP: work in progress")

    def test_concurrent_operations(self, repository):
        """Test concurrent repository operations."""
        import threading
        
        results = {"errors": []}
        
        def worker(worker_id):
            try:
                # Each worker creates and commits a weight
                weight = WeightTensor(
                    data=np.array([worker_id], dtype=np.float32)
                )
                repository.stage_weight(f"weight_{worker_id}", weight)
                repository.commit(f"Worker {worker_id} commit")
            except Exception as e:
                results["errors"].append(str(e))
        
        # Run concurrent operations
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Should handle concurrency gracefully
        assert len(results["errors"]) < 5  # Some conflicts expected
        commits = repository.list_commits()
        assert len(commits) >= 1  # At least one should succeed

    def test_remote_operations_mock(self, repository):
        """Test remote repository operations (mocked)."""
        # Mock remote functionality
        repository._remotes = {}
        
        # Add remote
        repository.add_remote("origin", "https://example.com/repo.git")
        assert "origin" in repository.list_remotes()
        
        # Remove remote
        repository.remove_remote("origin")
        assert "origin" not in repository.list_remotes()
        
        # Mock push/pull operations
        with patch.object(repository, "_push_to_remote") as mock_push:
            repository.push("origin", "main")
            mock_push.assert_called_once()
        
        with patch.object(repository, "_pull_from_remote") as mock_pull:
            repository.pull("origin", "main")
            mock_pull.assert_called_once()