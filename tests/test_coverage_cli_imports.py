"""CLI and imports coverage boost."""

import argparse
from unittest.mock import Mock, patch

from coral.cli.main import CoralCLI


class TestCLICoverage:
    """Test CLI to boost coverage."""

    def test_cli_init_command(self):
        """Test CLI init command handling."""
        cli = CoralCLI()

        # Mock Repository and system exit
        with patch("coral.cli.main.Repository") as mock_repo:
            with patch("sys.exit") as mock_exit:
                # Simulate init command
                args = argparse.Namespace(command="init", path=".")
                cli._run_init(args)

                # Verify Repository was called with init=True
                mock_repo.assert_called_once_with(".", init=True)
                mock_exit.assert_called_once_with(0)

    def test_cli_status_command(self):
        """Test CLI status command handling."""
        cli = CoralCLI()

        # Mock Repository
        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            mock_repo.status.return_value = {
                "staged": {},
                "modified": {},
                "branch": "main",
            }

            with patch("sys.exit") as mock_exit:
                args = argparse.Namespace(command="status")
                cli._run_status(args)

                mock_repo.status.assert_called_once()
                mock_exit.assert_called_once_with(0)

    def test_cli_add_command(self):
        """Test CLI add command handling."""
        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            # Mock loading weights
            with patch("coral.cli.main.load_weights_from_path") as mock_load:
                mock_load.return_value = {"weight1": Mock()}

                with patch("sys.exit") as mock_exit:
                    args = argparse.Namespace(command="add", path="model.pth")
                    cli._run_add(args)

                    mock_load.assert_called_once_with("model.pth")
                    mock_repo.stage_weights.assert_called_once()
                    mock_exit.assert_called_once_with(0)

    def test_cli_commit_command(self):
        """Test CLI commit command handling."""
        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            mock_commit = Mock()
            mock_commit.commit_hash = "abc123"
            mock_repo.commit.return_value = mock_commit

            with patch("sys.exit") as mock_exit:
                args = argparse.Namespace(
                    command="commit", message="Test commit", author=None, email=None
                )
                cli._run_commit(args)

                mock_repo.commit.assert_called_once_with(
                    "Test commit", author=None, email=None
                )
                mock_exit.assert_called_once_with(0)

    def test_cli_log_command(self):
        """Test CLI log command handling."""
        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            mock_commit = Mock()
            mock_commit.commit_hash = "abc123"
            mock_commit.metadata = Mock()
            mock_commit.metadata.message = "Test message"
            mock_commit.metadata.author = "Test Author"
            mock_commit.metadata.timestamp = Mock()
            mock_repo.log.return_value = [mock_commit]

            with patch("sys.exit") as mock_exit:
                args = argparse.Namespace(command="log", number=10, branch=None)
                cli._run_log(args)

                mock_repo.log.assert_called_once_with(max_commits=10)
                mock_exit.assert_called_once_with(0)

    def test_cli_branch_command(self):
        """Test CLI branch command handling."""
        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            mock_repo.current_branch = "main"

            mock_branch = Mock()
            mock_branch.name = "main"
            mock_repo.list_branches.return_value = [mock_branch]

            with patch("sys.exit") as mock_exit:
                # List branches
                args = argparse.Namespace(
                    command="branch", name=None, delete=None, list=True
                )
                cli._run_branch(args)

                mock_repo.list_branches.assert_called_once()
                mock_exit.assert_called_with(0)

                # Create branch
                mock_exit.reset_mock()
                args = argparse.Namespace(
                    command="branch", name="feature", delete=None, list=False
                )
                cli._run_branch(args)

                mock_repo.create_branch.assert_called_once_with("feature")
                mock_exit.assert_called_with(0)

    def test_cli_checkout_command(self):
        """Test CLI checkout command handling."""
        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            with patch("sys.exit") as mock_exit:
                args = argparse.Namespace(command="checkout", target="feature-branch")
                cli._run_checkout(args)

                mock_repo.checkout.assert_called_once_with("feature-branch")
                mock_exit.assert_called_once_with(0)

    def test_cli_tag_command(self):
        """Test CLI tag command handling."""
        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            mock_version = Mock()
            mock_version.name = "v1.0.0"
            mock_repo.tag_version.return_value = mock_version

            with patch("sys.exit") as mock_exit:
                args = argparse.Namespace(
                    command="tag", name="v1.0.0", description="First release"
                )
                cli._run_tag(args)

                mock_repo.tag_version.assert_called_once_with("v1.0.0", "First release")
                mock_exit.assert_called_once_with(0)

    def test_cli_error_handling(self):
        """Test CLI error handling."""
        cli = CoralCLI()

        # Test repository not found error
        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo_class.side_effect = ValueError("Not a Coral repository")

            with patch("sys.exit") as mock_exit:
                args = argparse.Namespace(command="status")
                cli._run_status(args)

                mock_exit.assert_called_once_with(1)
