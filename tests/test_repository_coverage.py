"""Repository coverage tests."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.version_control.repository import Repository


class TestRepositoryCoverage:
    """Tests to improve repository coverage."""

    def test_repository_init_and_basic_ops(self):
        """Test repository initialization and basic operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize repository
            repo = Repository(tmpdir, init=True)

            # Test basic properties
            assert repo.path == Path(tmpdir)
            assert repo.coral_dir.exists()
            assert repo.branch_manager.get_current_branch() is not None

            # Create and stage weights
            weight1 = WeightTensor(
                data=np.array([1, 2, 3], dtype=np.float32),
                metadata=WeightMetadata(name="w1", shape=(3,), dtype=np.float32),
            )
            weight2 = WeightTensor(
                data=np.array([4, 5, 6], dtype=np.float32),
                metadata=WeightMetadata(name="w2", shape=(3,), dtype=np.float32),
            )

            # Stage weights
            staged = repo.stage_weights({"w1": weight1, "w2": weight2})
            assert len(staged) == 2

            # Commit
            commit = repo.commit("Initial commit")
            assert commit is not None
            assert len(commit.weight_hashes) == 2

            # Get weight
            retrieved = repo.get_weight("w1")
            assert retrieved is not None
            np.testing.assert_array_equal(retrieved.data, weight1.data)

            # Verify nothing staged after commit (staging file should not exist)
            staged_file = repo.staging_dir / "staged.json"
            assert not staged_file.exists()

            # Log
            log_entries = repo.log(max_commits=10)
            assert len(log_entries) == 1
            assert log_entries[0].metadata.message == "Initial commit"

    def test_repository_branching(self):
        """Test repository branching operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(tmpdir, init=True)

            # Add initial commit
            weight = WeightTensor(
                data=np.ones(5, dtype=np.float32),
                metadata=WeightMetadata(name="w1", shape=(5,), dtype=np.float32),
            )
            repo.stage_weights({"w1": weight})
            initial_commit = repo.commit("Initial")

            # Create branch
            repo.create_branch("feature")
            branches = repo.branch_manager.list_branches()
            assert "feature" in [b.name for b in branches]

            # Switch branch
            repo.checkout("feature")
            assert repo.branch_manager.get_current_branch() == "feature"

            # Make change on feature branch
            weight2 = WeightTensor(
                data=np.ones(5, dtype=np.float32) * 2,
                metadata=WeightMetadata(name="w2", shape=(5,), dtype=np.float32),
            )
            repo.stage_weights({"w2": weight2})
            repo.commit("Feature commit")

            # Switch back to main
            repo.checkout("main")
            assert repo.branch_manager.get_current_branch() == "main"

            # Verify isolation
            assert repo.get_weight("w2") is None

            # Get commit
            commit = repo.version_graph.get_commit(initial_commit.commit_hash)
            assert commit is not None
            assert commit.metadata.message == "Initial"

    def test_repository_tagging(self):
        """Test repository tagging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(tmpdir, init=True)

            # Add commit
            weight = WeightTensor(
                data=np.ones(3, dtype=np.float32),
                metadata=WeightMetadata(name="w1", shape=(3,), dtype=np.float32),
            )
            repo.stage_weights({"w1": weight})
            repo.commit("Version 1.0")

            # Tag version
            version = repo.tag_version("v1.0.0", "First release")
            assert version.name == "v1.0.0"
            assert version.description == "First release"

            # List versions
            versions = list(repo.version_graph.versions.values())
            assert len(versions) == 1
            assert versions[0].name == "v1.0.0"

            # Get version by ID
            retrieved = repo.version_graph.get_version(version.version_id)
            assert retrieved is not None
            assert retrieved.name == "v1.0.0"

    def test_repository_diff(self):
        """Test repository diff operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(tmpdir, init=True)

            # First commit
            weight1 = WeightTensor(
                data=np.ones(5, dtype=np.float32),
                metadata=WeightMetadata(name="w1", shape=(5,), dtype=np.float32),
            )
            repo.stage_weights({"w1": weight1})
            commit1 = repo.commit("First")

            # Second commit with modifications
            weight1_mod = WeightTensor(
                data=np.ones(5, dtype=np.float32) * 1.1,
                metadata=WeightMetadata(name="w1", shape=(5,), dtype=np.float32),
            )
            weight2 = WeightTensor(
                data=np.zeros(3, dtype=np.float32),
                metadata=WeightMetadata(name="w2", shape=(3,), dtype=np.float32),
            )
            repo.stage_weights({"w1": weight1_mod, "w2": weight2})
            commit2 = repo.commit("Second")

            # Diff commits
            diff = repo.diff(commit1.commit_hash, commit2.commit_hash)
            assert "added" in diff
            assert "w2" in diff["added"]
            assert "modified" in diff
            assert "w1" in diff["modified"]

    def test_repository_error_cases(self):
        """Test repository error handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test loading non-existent repo
            with pytest.raises(ValueError):
                Repository(tmpdir)  # No init=True

            # Initialize properly
            repo = Repository(tmpdir, init=True)

            # Test committing with no staged changes
            with pytest.raises(ValueError):
                repo.commit("Empty commit")

            # Test checking out non-existent branch
            with pytest.raises(ValueError):
                repo.checkout("non-existent")

            # Test getting non-existent weight
            assert repo.get_weight("non-existent") is None

            # Test getting non-existent commit (returns None, doesn't raise)
            assert repo.version_graph.get_commit("non-existent-hash") is None
