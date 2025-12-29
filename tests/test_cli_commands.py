"""Tests for CLI command execution.

This module tests the actual execution of CLI commands, not just argument parsing.
"""

import json
import os
import shutil
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from coral.cli.main import CoralCLI
from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.version_control.repository import Repository


class TestCLICommandExecution:
    """Test CLI command execution."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        tmpdir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        yield tmpdir
        os.chdir(original_cwd)
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def cli(self):
        """Create a CLI instance."""
        return CoralCLI()

    @pytest.fixture
    def initialized_repo(self, temp_dir):
        """Create an initialized repository."""
        repo = Repository(Path(temp_dir), init=True)
        return repo

    @pytest.fixture
    def repo_with_weights(self, initialized_repo, temp_dir):
        """Create a repository with some committed weights."""
        weights = {
            "layer1.weight": WeightTensor(
                data=np.random.randn(10, 5).astype(np.float32),
                metadata=WeightMetadata(
                    name="layer1.weight",
                    shape=(10, 5),
                    dtype=np.float32,
                ),
            ),
            "layer1.bias": WeightTensor(
                data=np.random.randn(10).astype(np.float32),
                metadata=WeightMetadata(
                    name="layer1.bias",
                    shape=(10,),
                    dtype=np.float32,
                ),
            ),
        }
        initialized_repo.stage_weights(weights)
        initialized_repo.commit(message="Initial commit")
        return initialized_repo

    # === Init Command Tests ===

    def test_cmd_init_creates_repository(self, cli, temp_dir):
        """Test that init command creates a repository."""
        result = cli.run(["init"])
        assert result == 0
        assert (Path(temp_dir) / ".coral").exists()

    def test_cmd_init_with_path(self, cli, temp_dir):
        """Test init command with custom path."""
        custom_path = Path(temp_dir) / "my_repo"
        custom_path.mkdir()
        result = cli.run(["init", str(custom_path)])
        assert result == 0
        assert (custom_path / ".coral").exists()

    def test_cmd_init_already_initialized(self, cli, initialized_repo, temp_dir):
        """Test init on already initialized repo gives error."""
        # Try to reinitialize - should fail
        result = cli.run(["init"])
        assert result == 1  # Should return error

    # === Status Command Tests ===

    def test_cmd_status_empty_repo(self, cli, initialized_repo):
        """Test status command on empty repository."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["status"])
            output = fake_out.getvalue()
        assert result == 0
        assert "On branch main" in output
        assert "Nothing staged" in output

    def test_cmd_status_with_staged_weights(self, cli, initialized_repo, temp_dir):
        """Test status command shows staged weights."""
        # Create and stage a weight file
        weight_data = np.random.randn(10, 5).astype(np.float32)
        np.save(Path(temp_dir) / "test_weight.npy", weight_data)

        # Stage it
        cli.run(["add", "test_weight.npy"])

        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["status"])
            output = fake_out.getvalue()

        assert result == 0
        assert "Changes to be committed" in output
        assert "test_weight" in output

    # === Add Command Tests ===

    def test_cmd_add_npy_file(self, cli, initialized_repo, temp_dir):
        """Test adding a .npy file."""
        weight_data = np.random.randn(10, 5).astype(np.float32)
        np.save(Path(temp_dir) / "weights.npy", weight_data)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["add", "weights.npy"])
            output = fake_out.getvalue()

        assert result == 0
        assert "Staged 1 weight(s)" in output

    def test_cmd_add_npz_file(self, cli, initialized_repo, temp_dir):
        """Test adding a .npz file with multiple weights."""
        weights = {
            "layer1": np.random.randn(10, 5).astype(np.float32),
            "layer2": np.random.randn(5, 3).astype(np.float32),
        }
        np.savez(Path(temp_dir) / "model.npz", **weights)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["add", "model.npz"])
            output = fake_out.getvalue()

        assert result == 0
        assert "Staged 2 weight(s)" in output

    def test_cmd_add_nonexistent_file(self, cli, initialized_repo):
        """Test adding a nonexistent file."""
        result = cli.run(["add", "nonexistent.npy"])
        assert result == 1

    def test_cmd_add_unsupported_format(self, cli, initialized_repo, temp_dir):
        """Test adding unsupported file format."""
        # Create a text file
        (Path(temp_dir) / "weights.txt").write_text("not a weight file")

        result = cli.run(["add", "weights.txt"])
        assert result == 1

    # === Commit Command Tests ===

    def test_cmd_commit_basic(self, cli, initialized_repo, temp_dir):
        """Test basic commit command."""
        # Stage a weight
        weight_data = np.random.randn(10, 5).astype(np.float32)
        np.save(Path(temp_dir) / "weights.npy", weight_data)
        cli.run(["add", "weights.npy"])

        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["commit", "-m", "First commit"])
            output = fake_out.getvalue()

        assert result == 0
        assert "First commit" in output
        assert "1 weight(s) changed" in output

    def test_cmd_commit_with_author(self, cli, initialized_repo, temp_dir):
        """Test commit with author info."""
        weight_data = np.random.randn(10, 5).astype(np.float32)
        np.save(Path(temp_dir) / "weights.npy", weight_data)
        cli.run(["add", "weights.npy"])

        result = cli.run(
            [
                "commit",
                "-m",
                "Test commit",
                "--author",
                "John Doe",
                "--email",
                "john@example.com",
            ]
        )
        assert result == 0

    # === Log Command Tests ===

    def test_cmd_log_empty_repo(self, cli, initialized_repo):
        """Test log command on empty repository."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["log"])
            output = fake_out.getvalue()

        assert result == 0
        assert "No commits yet" in output

    def test_cmd_log_with_commits(self, cli, repo_with_weights):
        """Test log command shows commits."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["log"])
            output = fake_out.getvalue()

        assert result == 0
        assert "Initial commit" in output

    def test_cmd_log_oneline(self, cli, repo_with_weights):
        """Test log command with --oneline flag."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["log", "--oneline"])
            output = fake_out.getvalue()

        assert result == 0
        # Oneline format should be more compact
        lines = [line for line in output.strip().split("\n") if line]
        assert len(lines) >= 1

    def test_cmd_log_limit(self, cli, initialized_repo, temp_dir):
        """Test log command with limit."""
        # Create multiple commits
        for i in range(5):
            weight_data = np.random.randn(10, 5).astype(np.float32)
            np.save(Path(temp_dir) / f"weights_{i}.npy", weight_data)
            cli.run(["add", f"weights_{i}.npy"])
            cli.run(["commit", "-m", f"Commit {i}"])

        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["log", "-n", "2"])
            output = fake_out.getvalue()

        assert result == 0
        # Should show at most 2 commits
        assert output.count("commit ") <= 2

    # === Branch Command Tests ===

    def test_cmd_branch_list(self, cli, repo_with_weights):
        """Test listing branches."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["branch"])
            output = fake_out.getvalue()

        assert result == 0
        assert "main" in output

    def test_cmd_branch_create(self, cli, repo_with_weights):
        """Test creating a branch."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["branch", "feature"])
            output = fake_out.getvalue()

        assert result == 0
        assert "Created branch feature" in output

    def test_cmd_branch_delete(self, cli, repo_with_weights):
        """Test deleting a branch."""
        # First create a branch
        cli.run(["branch", "temp-branch"])

        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["branch", "-d", "temp-branch"])
            output = fake_out.getvalue()

        assert result == 0
        assert "Deleted branch temp-branch" in output

    # === Checkout Command Tests ===

    def test_cmd_checkout_branch(self, cli, repo_with_weights):
        """Test checking out a branch."""
        # Create a branch
        cli.run(["branch", "feature"])

        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["checkout", "feature"])
            output = fake_out.getvalue()

        assert result == 0
        assert "Switched to 'feature'" in output

    # === Merge Command Tests ===

    def test_cmd_merge_branch(self, cli, repo_with_weights, temp_dir):
        """Test merging a branch."""
        # Create feature branch and add a commit
        cli.run(["branch", "feature"])
        cli.run(["checkout", "feature"])

        weight_data = np.random.randn(5, 5).astype(np.float32)
        np.save(Path(temp_dir) / "feature_weight.npy", weight_data)
        cli.run(["add", "feature_weight.npy"])
        cli.run(["commit", "-m", "Feature commit"])

        # Switch back and merge
        cli.run(["checkout", "main"])

        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["merge", "feature"])
            output = fake_out.getvalue()

        assert result == 0
        assert "Merged feature into main" in output

    # === Diff Command Tests ===

    def test_cmd_diff_between_commits(self, cli, repo_with_weights, temp_dir):
        """Test diff between two commits."""
        # Add another commit
        weight_data = np.random.randn(20, 10).astype(np.float32)
        np.save(Path(temp_dir) / "new_weight.npy", weight_data)
        cli.run(["add", "new_weight.npy"])
        cli.run(["commit", "-m", "Second commit"])

        # Get commit hashes - use full hashes for reliable lookup
        repo = Repository(Path(temp_dir))
        commits = repo.log(max_commits=2)
        hash1 = commits[1].commit_hash  # Older (full hash)
        hash2 = commits[0].commit_hash  # Newer (full hash)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["diff", hash1, hash2])
            output = fake_out.getvalue()

        assert result == 0
        assert "Comparing" in output

    # === Tag Command Tests ===

    def test_cmd_tag_create(self, cli, repo_with_weights):
        """Test creating a tag."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["tag", "v1.0.0"])
            output = fake_out.getvalue()

        assert result == 0
        assert "Tagged version 'v1.0.0'" in output

    def test_cmd_tag_with_description(self, cli, repo_with_weights):
        """Test creating a tag with description."""
        with patch("sys.stdout", new=StringIO()):
            result = cli.run(["tag", "v1.0.0", "-d", "First release"])

        assert result == 0

    def test_cmd_tag_with_metrics(self, cli, repo_with_weights):
        """Test creating a tag with metrics."""
        result = cli.run(["tag", "v1.0.0", "--metric", "accuracy=0.95"])
        assert result == 0

    # === Show Command Tests ===

    def test_cmd_show_weight(self, cli, repo_with_weights):
        """Test showing weight information."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["show", "layer1.weight"])
            output = fake_out.getvalue()

        assert result == 0
        assert "Weight: layer1.weight" in output
        assert "Shape:" in output
        assert "Dtype:" in output

    def test_cmd_show_nonexistent_weight(self, cli, repo_with_weights):
        """Test showing nonexistent weight."""
        result = cli.run(["show", "nonexistent"])
        assert result == 1

    # === GC Command Tests ===

    def test_cmd_gc(self, cli, repo_with_weights):
        """Test garbage collection command."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["gc"])
            output = fake_out.getvalue()

        assert result == 0
        assert "Garbage collection complete" in output

    # === Stats Command Tests ===

    def test_cmd_stats(self, cli, repo_with_weights):
        """Test stats command."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["stats"])
            output = fake_out.getvalue()

        assert result == 0
        assert "Coral Repository Statistics" in output
        assert "Weight Storage:" in output
        assert "Storage Savings:" in output

    def test_cmd_stats_json(self, cli, repo_with_weights):
        """Test stats command with JSON output."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["stats", "--json"])
            output = fake_out.getvalue()

        assert result == 0
        # Should be valid JSON
        stats = json.loads(output)
        assert "total_commits" in stats
        assert "total_weights" in stats

    # === Remote Command Tests ===

    def test_cmd_remote_list_empty(self, cli, initialized_repo):
        """Test listing remotes when none exist."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["remote", "list"])
            output = fake_out.getvalue()

        assert result == 0
        assert "No remotes configured" in output

    def test_cmd_remote_add(self, cli, initialized_repo, temp_dir):
        """Test adding a remote."""
        remote_path = Path(temp_dir) / "remote"
        remote_path.mkdir()

        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["remote", "add", "origin", f"file://{remote_path}"])
            output = fake_out.getvalue()

        assert result == 0
        assert "Added remote 'origin'" in output

    def test_cmd_remote_show(self, cli, initialized_repo, temp_dir):
        """Test showing remote details."""
        remote_path = Path(temp_dir) / "remote"
        remote_path.mkdir()
        cli.run(["remote", "add", "origin", f"file://{remote_path}"])

        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["remote", "show", "origin"])
            output = fake_out.getvalue()

        assert result == 0
        assert "Remote: origin" in output
        assert "URL:" in output

    def test_cmd_remote_remove(self, cli, initialized_repo, temp_dir):
        """Test removing a remote."""
        remote_path = Path(temp_dir) / "remote"
        remote_path.mkdir()
        cli.run(["remote", "add", "origin", f"file://{remote_path}"])

        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["remote", "remove", "origin"])
            output = fake_out.getvalue()

        assert result == 0
        assert "Removed remote 'origin'" in output

    # === Config Command Tests ===

    def test_cmd_config_show(self, cli, initialized_repo):
        """Test showing configuration."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["config", "show"])
            output = fake_out.getvalue()

        assert result == 0
        assert "Coral Configuration" in output
        assert "[user]" in output
        assert "[core]" in output

    def test_cmd_config_list(self, cli, initialized_repo):
        """Test listing configuration options."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["config", "list"])
            output = fake_out.getvalue()

        assert result == 0
        assert "Available Configuration Options" in output
        assert "core.similarity_threshold" in output

    def test_cmd_config_get(self, cli, initialized_repo):
        """Test getting a configuration value."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["config", "get", "core.similarity_threshold"])
            output = fake_out.getvalue()

        assert result == 0
        # Should print a numeric value
        assert output.strip()

    def test_cmd_config_get_nonexistent(self, cli, initialized_repo):
        """Test getting a nonexistent config key."""
        result = cli.run(["config", "get", "nonexistent.key"])
        assert result == 1

    def test_cmd_config_set(self, cli, initialized_repo):
        """Test setting a configuration value."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["config", "set", "core.similarity_threshold", "0.95"])
            output = fake_out.getvalue()

        assert result == 0
        assert "Set core.similarity_threshold = 0.95" in output

    def test_cmd_config_validate(self, cli, initialized_repo):
        """Test validating configuration."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["config", "validate"])
            output = fake_out.getvalue()

        assert result == 0
        assert "Configuration is valid" in output


class TestCLIHelpers:
    """Test CLI helper methods."""

    def test_format_bytes(self):
        """Test byte formatting."""
        cli = CoralCLI()

        assert cli._format_bytes(500) == "500.0 B"
        assert cli._format_bytes(1024) == "1.0 KB"
        assert cli._format_bytes(1024 * 1024) == "1.0 MB"
        assert cli._format_bytes(1024 * 1024 * 1024) == "1.0 GB"

    def test_parse_config_value_bool(self):
        """Test parsing boolean config values."""
        cli = CoralCLI()

        assert cli._parse_config_value("true") is True
        assert cli._parse_config_value("True") is True
        assert cli._parse_config_value("yes") is True
        assert cli._parse_config_value("1") is True

        assert cli._parse_config_value("false") is False
        assert cli._parse_config_value("False") is False
        assert cli._parse_config_value("no") is False
        assert cli._parse_config_value("0") is False

    def test_parse_config_value_numeric(self):
        """Test parsing numeric config values."""
        cli = CoralCLI()

        assert cli._parse_config_value("42") == 42
        assert cli._parse_config_value("3.14") == 3.14
        assert cli._parse_config_value("-10") == -10

    def test_parse_config_value_string(self):
        """Test parsing string config values."""
        cli = CoralCLI()

        assert cli._parse_config_value("hello") == "hello"
        assert cli._parse_config_value("some text") == "some text"

    def test_parse_config_value_none(self):
        """Test parsing None config values."""
        cli = CoralCLI()

        assert cli._parse_config_value("none") is None
        assert cli._parse_config_value("null") is None
        assert cli._parse_config_value("") is None


class TestCLIErrorHandling:
    """Test CLI error handling."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        tmpdir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        yield tmpdir
        os.chdir(original_cwd)
        shutil.rmtree(tmpdir)

    def test_command_without_repo(self, temp_dir):
        """Test running command outside of a repository."""
        cli = CoralCLI()
        result = cli.run(["status"])
        assert result == 1  # Should fail

    def test_no_command(self, temp_dir):
        """Test running CLI without command."""
        cli = CoralCLI()
        with patch("sys.stdout", new=StringIO()):
            result = cli.run([])
        assert result == 0  # Should print help and exit normally

    def test_find_repo_root(self, temp_dir):
        """Test finding repository root from subdirectory."""
        # Initialize repo
        Repository(Path(temp_dir), init=True)

        # Create a subdirectory
        subdir = Path(temp_dir) / "subdir" / "nested"
        subdir.mkdir(parents=True)
        os.chdir(subdir)

        cli = CoralCLI()
        repo_root = cli._find_repo_root()

        assert repo_root is not None
        # Use resolve() to handle macOS symlinks (/tmp -> /private/tmp, /var -> /private/var)
        assert repo_root.resolve() == Path(temp_dir).resolve()

    def test_checkout_nonexistent_branch(self, temp_dir):
        """Test checking out a nonexistent branch."""
        repo = Repository(Path(temp_dir), init=True)

        # Create initial commit
        weight = WeightTensor(
            data=np.ones(5).astype(np.float32),
            metadata=WeightMetadata(name="weight", shape=(5,), dtype=np.float32),
        )
        repo.stage_weights({"weight": weight})
        repo.commit("Initial")

        cli = CoralCLI()
        result = cli.run(["checkout", "nonexistent-branch"])
        assert result == 1  # Should fail


class TestCLIPushPull:
    """Test CLI push and pull commands."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        tmpdir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        yield tmpdir
        os.chdir(original_cwd)
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def repo_with_remote(self, temp_dir):
        """Create a repository with a remote."""
        repo = Repository(Path(temp_dir), init=True)

        # Create a weight and commit
        weight = WeightTensor(
            data=np.random.randn(10, 5).astype(np.float32),
            metadata=WeightMetadata(
                name="layer.weight", shape=(10, 5), dtype=np.float32
            ),
        )
        repo.stage_weights({"layer.weight": weight})
        repo.commit("Initial commit")

        # Create remote directory
        remote_path = Path(temp_dir) / "remote_storage"
        remote_path.mkdir()

        # Add remote
        repo.add_remote("origin", f"file://{remote_path}")

        return repo

    def test_push_no_remote(self, temp_dir):
        """Test push without remote configured."""
        repo = Repository(Path(temp_dir), init=True)

        weight = WeightTensor(
            data=np.ones(5).astype(np.float32),
            metadata=WeightMetadata(name="weight", shape=(5,), dtype=np.float32),
        )
        repo.stage_weights({"weight": weight})
        repo.commit("Initial")

        cli = CoralCLI()
        result = cli.run(["push", "origin"])
        assert result == 1  # Should fail - no remote

    def test_pull_no_remote(self, temp_dir):
        """Test pull without remote configured."""
        repo = Repository(Path(temp_dir), init=True)

        weight = WeightTensor(
            data=np.ones(5).astype(np.float32),
            metadata=WeightMetadata(name="weight", shape=(5,), dtype=np.float32),
        )
        repo.stage_weights({"weight": weight})
        repo.commit("Initial")

        cli = CoralCLI()
        result = cli.run(["pull", "origin"])
        assert result == 1  # Should fail - no remote

    def test_push_to_file_remote(self, repo_with_remote):
        """Test pushing to a file:// remote."""
        cli = CoralCLI()

        with patch("sys.stdout", new=StringIO()) as fake_out:
            result = cli.run(["push", "origin"])
            output = fake_out.getvalue()

        assert result == 0
        assert "Push complete" in output

    def test_sync_status_no_remote(self, temp_dir):
        """Test sync status without remote."""
        Repository(Path(temp_dir), init=True)

        cli = CoralCLI()
        result = cli.run(["sync-status", "origin"])
        assert result == 1  # Should fail


class TestCLICompareCommand:
    """Test CLI compare command."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        tmpdir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        yield tmpdir
        os.chdir(original_cwd)
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def repo_with_history(self, temp_dir):
        """Create a repository with multiple commits."""
        repo = Repository(Path(temp_dir), init=True)

        # First commit
        weight1 = WeightTensor(
            data=np.ones(10).astype(np.float32),
            metadata=WeightMetadata(name="weight1", shape=(10,), dtype=np.float32),
        )
        repo.stage_weights({"weight1": weight1})
        repo.commit("First commit")

        # Second commit with additional weight
        weight2 = WeightTensor(
            data=np.ones(5).astype(np.float32),
            metadata=WeightMetadata(name="weight2", shape=(5,), dtype=np.float32),
        )
        repo.stage_weights({"weight1": weight1, "weight2": weight2})
        repo.commit("Second commit")

        return repo

    def test_compare_branches(self, repo_with_history):
        """Test comparing main to a new branch."""
        # Create a feature branch
        repo_with_history.create_branch("feature")
        repo_with_history.checkout("feature")

        # Add a new weight on feature
        weight3 = WeightTensor(
            data=np.ones(3).astype(np.float32),
            metadata=WeightMetadata(name="weight3", shape=(3,), dtype=np.float32),
        )
        repo_with_history.stage_weights(
            {
                "weight1": repo_with_history.get_weight("weight1"),
                "weight2": repo_with_history.get_weight("weight2"),
                "weight3": weight3,
            }
        )
        repo_with_history.commit("Feature commit")

        cli = CoralCLI()

        with patch("sys.stdout", new=StringIO()):
            result = cli.run(["compare", "main", "feature"])

        assert result == 0
