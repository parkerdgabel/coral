"""CLI and imports coverage boost."""

import argparse
from unittest.mock import Mock, patch

from coral.cli.main import CoralCLI


class TestCLICoverage:
    """Test CLI to boost coverage."""

    def test_cli_init_command(self):
        """Test CLI init command handling."""
        cli = CoralCLI()

        # Mock Repository
        with patch("coral.cli.main.Repository") as mock_repo:
            with patch("builtins.print"):
                # Simulate init command
                args = argparse.Namespace(command="init", path=".")
                result = cli._cmd_init(args)

                # Verify Repository was called with init=True
                assert mock_repo.call_count >= 1
                assert result == 0

    def test_cli_status_command(self):
        """Test CLI status command handling."""
        cli = CoralCLI()

        # Mock Repository
        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            mock_repo.branch_manager.get_current_branch.return_value = "main"

            # Mock the staging directory
            mock_staging_dir = Mock()
            mock_staged_file = Mock()
            mock_staged_file.exists.return_value = False
            mock_staging_dir.__truediv__ = Mock(return_value=mock_staged_file)
            mock_repo.staging_dir = mock_staging_dir

            from pathlib import Path

            with patch("builtins.print"):
                args = argparse.Namespace(command="status")
                result = cli._cmd_status(args, Path("."))

                assert result == 0

    def test_cli_add_command(self):
        """Test CLI add command handling."""
        cli = CoralCLI()

        import numpy as np
        from pathlib import Path

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo.stage_weights.return_value = {"model": "hash123"}
            mock_repo_class.return_value = mock_repo

            # Mock Path and numpy
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path.suffix = ".npy"
            mock_path.stem = "model"

            with patch("coral.cli.main.Path", return_value=mock_path):
                with patch("numpy.load", return_value=np.array([1, 2, 3])):
                    with patch("builtins.print"):
                        args = argparse.Namespace(command="add", weights=["model.npy"])
                        result = cli._cmd_add(args, Path("."))

                        mock_repo.stage_weights.assert_called_once()
                        assert result == 0

    def test_cli_commit_command(self):
        """Test CLI commit command handling."""
        cli = CoralCLI()

        from pathlib import Path

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            mock_commit = Mock()
            mock_commit.commit_hash = "abc123"
            mock_commit.metadata = Mock()
            mock_commit.metadata.message = "Test commit"
            mock_commit.weight_hashes = {"w1": "h1"}
            mock_repo.commit.return_value = mock_commit
            mock_repo.branch_manager.get_current_branch.return_value = "main"

            with patch("builtins.print"):
                args = argparse.Namespace(
                    command="commit",
                    message="Test commit",
                    author=None,
                    email=None,
                    tag=None,
                )
                result = cli._cmd_commit(args, Path("."))

                mock_repo.commit.assert_called_once_with(
                    message="Test commit", author=None, email=None, tags=[]
                )
                assert result == 0

    def test_cli_log_command(self):
        """Test CLI log command handling."""
        cli = CoralCLI()

        from pathlib import Path

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            mock_commit = Mock()
            mock_commit.commit_hash = "abc123"
            mock_commit.metadata = Mock()
            mock_commit.metadata.message = "Test message"
            mock_commit.metadata.author = "Test Author"
            mock_commit.metadata.email = "test@example.com"
            mock_commit.metadata.timestamp = "2024-01-01T00:00:00"
            mock_commit.metadata.tags = []
            mock_repo.log.return_value = [mock_commit]

            with patch("builtins.print"):
                args = argparse.Namespace(command="log", number=10, oneline=False)
                result = cli._cmd_log(args, Path("."))

                mock_repo.log.assert_called_once_with(max_commits=10)
                assert result == 0

    def test_cli_branch_command(self):
        """Test CLI branch command handling."""
        cli = CoralCLI()

        from pathlib import Path

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            mock_branch = Mock()
            mock_branch.name = "main"
            mock_repo.branch_manager.list_branches.return_value = [mock_branch]
            mock_repo.branch_manager.get_current_branch.return_value = "main"

            with patch("builtins.print"):
                # List branches
                args = argparse.Namespace(
                    command="branch", name=None, delete=None, list=True
                )
                result = cli._cmd_branch(args, Path("."))

                mock_repo.branch_manager.list_branches.assert_called_once()
                assert result == 0

            # Create branch
            mock_repo.branch_manager.list_branches.reset_mock()
            with patch("builtins.print"):
                args = argparse.Namespace(
                    command="branch", name="feature", delete=None, list=False
                )
                result = cli._cmd_branch(args, Path("."))

                mock_repo.create_branch.assert_called_once_with("feature")
                assert result == 0

    def test_cli_checkout_command(self):
        """Test CLI checkout command handling."""
        cli = CoralCLI()

        from pathlib import Path

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            with patch("builtins.print"):
                args = argparse.Namespace(command="checkout", target="feature-branch")
                result = cli._cmd_checkout(args, Path("."))

                mock_repo.checkout.assert_called_once_with("feature-branch")
                assert result == 0

    def test_cli_tag_command(self):
        """Test CLI tag command handling."""
        cli = CoralCLI()

        from pathlib import Path

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            mock_version = Mock()
            mock_version.name = "v1.0.0"
            mock_version.version_id = "vid123"
            mock_repo.tag_version.return_value = mock_version

            with patch("builtins.print"):
                args = argparse.Namespace(
                    command="tag",
                    name="v1.0.0",
                    description="First release",
                    metric=None,
                    commit=None,
                )
                result = cli._cmd_tag(args, Path("."))

                mock_repo.tag_version.assert_called_once_with(
                    name="v1.0.0",
                    description="First release",
                    metrics=None,
                    commit_ref=None,
                )
                assert result == 0

    def test_cli_error_handling(self):
        """Test CLI error handling."""
        cli = CoralCLI()

        from pathlib import Path

        # Test repository not found error
        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo_class.side_effect = ValueError("Not a Coral repository")

            with patch("builtins.print"):
                args = argparse.Namespace(command="status")
                # The run() method catches exceptions, so test that directly
                with patch.object(cli.parser, "parse_args", return_value=args):
                    with patch.object(
                        cli, "_find_repo_root", return_value="/fake/repo"
                    ):
                        result = cli.run()
                        assert result == 1
