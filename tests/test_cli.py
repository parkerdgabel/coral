import argparse
import os
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from coral.cli.main import CoralCLI


class TestCLI:
    @pytest.fixture
    def cli(self):
        """Create a CLI instance."""
        return CoralCLI()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            yield tmpdir
            os.chdir(original_cwd)

    def test_cli_creation(self, cli):
        """Test CLI instance creation."""
        assert cli is not None
        assert cli.parser is not None
        assert isinstance(cli.parser, argparse.ArgumentParser)

    def test_parser_commands(self, cli):
        """Test that all commands are registered."""
        # Parse help to check commands exist
        with patch("sys.argv", ["coral", "--help"]):
            with pytest.raises(SystemExit):
                with patch("sys.stdout", new=StringIO()):
                    cli.parser.parse_args()

        # Basic test - ensure parser has subparsers
        assert cli.parser._subparsers is not None

    @patch("coral.cli.main.Repository")
    def test_init_command(self, mock_repo_class, cli, temp_dir):
        """Test repository initialization command."""
        mock_repo_class.init.return_value = Mock()

        # Test init without path
        args = cli.parser.parse_args(["init"])
        assert args.command == "init"
        assert args.path == "."

        # Test init with path
        args = cli.parser.parse_args(["init", "/custom/path"])
        assert args.command == "init"
        assert args.path == "/custom/path"

    def test_add_command_parsing(self, cli):
        """Test add command parsing."""
        args = cli.parser.parse_args(["add", "model1.pth", "model2.pth"])
        assert args.command == "add"
        assert args.weights == ["model1.pth", "model2.pth"]

    def test_commit_command_parsing(self, cli):
        """Test commit command parsing."""
        args = cli.parser.parse_args(["commit", "-m", "Initial commit"])
        assert args.command == "commit"
        assert args.message == "Initial commit"

        # Test with author info
        args = cli.parser.parse_args(
            [
                "commit",
                "-m",
                "Test",
                "--author",
                "John Doe",
                "--email",
                "john@example.com",
            ]
        )
        assert args.author == "John Doe"
        assert args.email == "john@example.com"

    def test_status_command_parsing(self, cli):
        """Test status command parsing."""
        args = cli.parser.parse_args(["status"])
        assert args.command == "status"

    def test_log_command_parsing(self, cli):
        """Test log command parsing."""
        args = cli.parser.parse_args(["log"])
        assert args.command == "log"

        # Test with limit
        args = cli.parser.parse_args(["log", "-n", "10"])
        assert args.number == 10

    def test_branch_command_parsing(self, cli):
        """Test branch command parsing."""
        # List branches
        args = cli.parser.parse_args(["branch"])
        assert args.command == "branch"

        # Create branch
        args = cli.parser.parse_args(["branch", "new-feature"])
        assert args.name == "new-feature"

        # Delete branch
        args = cli.parser.parse_args(["branch", "-d", "old-feature"])
        assert args.delete == "old-feature"

    def test_checkout_command_parsing(self, cli):
        """Test checkout command parsing."""
        args = cli.parser.parse_args(["checkout", "feature-branch"])
        assert args.command == "checkout"
        assert args.target == "feature-branch"

    def test_merge_command_parsing(self, cli):
        """Test merge command parsing."""
        args = cli.parser.parse_args(["merge", "feature-branch"])
        assert args.command == "merge"
        assert args.branch == "feature-branch"

    def test_diff_command_parsing(self, cli):
        """Test diff command parsing."""
        # Diff between refs
        args = cli.parser.parse_args(["diff", "commit1", "commit2"])
        assert args.command == "diff"
        assert args.from_ref == "commit1"
        assert args.to_ref == "commit2"

        # Diff with only from ref
        args = cli.parser.parse_args(["diff", "commit1"])
        assert args.from_ref == "commit1"
        assert args.to_ref is None

    def test_tag_command_parsing(self, cli):
        """Test tag command parsing."""
        # Create tag
        args = cli.parser.parse_args(["tag", "v1.0.0"])
        assert args.command == "tag"
        assert args.name == "v1.0.0"

        # Create tag with description
        args = cli.parser.parse_args(["tag", "v1.0.0", "-d", "Release version"])
        assert args.description == "Release version"

        # Create tag with metrics
        args = cli.parser.parse_args(["tag", "v1.0.0", "--metric", "accuracy=0.95"])
        assert args.metric == ["accuracy=0.95"]

    @patch("coral.cli.main.Repository")
    def test_run_init(self, mock_repo_class, cli, temp_dir):
        """Test running init command."""
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo

        # Run init command
        cli.run(["init"])

        # Verify repository was initialized with init=True
        mock_repo_class.assert_called_once()
        call_args = mock_repo_class.call_args
        assert call_args[1]["init"] is True

    @patch("coral.cli.main.Repository")
    def test_run_status(self, mock_repo_class, cli, temp_dir):
        """Test running status command."""
        mock_repo = Mock()
        mock_branch_manager = Mock()
        mock_branch_manager.get_current_branch.return_value = "main"
        mock_repo.branch_manager = mock_branch_manager
        mock_repo.staging_dir = Path(".coral/staging")
        mock_repo_class.return_value = mock_repo

        # Create a mock repository directory
        os.makedirs(".coral")

        # Run status command
        with patch("sys.stdout", new=StringIO()) as fake_out:
            cli.run(["status"])
            output = fake_out.getvalue()

        # Should show current branch
        assert "On branch main" in output

    def test_invalid_command(self, cli):
        """Test handling of invalid command."""
        with pytest.raises(SystemExit):
            cli.parser.parse_args(["invalid-command"])

    def test_command_help(self, cli):
        """Test help for specific commands."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            with pytest.raises(SystemExit):
                cli.parser.parse_args(["commit", "--help"])
            output = fake_out.getvalue()
            # Help should mention message requirement
            assert "-m" in output or "--message" in output

    def test_cli_run_method_dispatch(self, cli):
        """Test CLI run method dispatches to correct handler."""
        # Mock all run methods
        for cmd in [
            "init",
            "add",
            "commit",
            "status",
            "log",
            "checkout",
            "branch",
            "merge",
            "diff",
            "tag",
            "show",
            "gc",
        ]:
            setattr(cli, f"_cmd_{cmd}", Mock())

        # Test each command
        for cmd in [
            "init",
            "add",
            "commit",
            "status",
            "log",
            "checkout",
            "branch",
            "merge",
            "diff",
            "tag",
            "show",
            "gc",
        ]:
            # Reset all mocks
            for cmd2 in [
                "init",
                "add",
                "commit",
                "status",
                "log",
                "checkout",
                "branch",
                "merge",
                "diff",
                "tag",
                "show",
                "gc",
            ]:
                getattr(cli, f"_cmd_{cmd2}").reset_mock()

            # Create args for command
            args = Mock()
            args.command = cmd

            # Add required attributes based on command
            if cmd == "add":
                args.weights = ["model.pth"]
            elif cmd == "commit":
                args.message = "test"
                args.author = None
                args.email = None
                args.tag = None
            elif cmd == "log":
                args.number = 10
                args.branch = None
            elif cmd == "checkout":
                args.target = "branch"
            elif cmd == "branch":
                args.name = None
                args.delete = None
                args.list = True
            elif cmd == "merge":
                args.branch = "feature"
            elif cmd == "diff":
                args.from_ref = "main"
                args.to_ref = "feature"
            elif cmd == "tag":
                args.name = "v1.0"
                args.description = "desc"
                args.metric = None
                args.commit = None
            elif cmd == "show":
                args.weight = "weight"
                args.commit = None
            elif cmd == "gc":
                pass  # GC doesn't need special args
            elif cmd == "init":
                args.path = "."

            with patch.object(cli.parser, "parse_args", return_value=args):
                cli.run()

            # Verify correct method was called
            getattr(cli, f"_cmd_{cmd}").assert_called_once()

    # Repository detection and error handling tests
    def test_cli_find_repo_root(self, cli):
        """Test _find_repo_root method."""
        # Test when .coral exists in current directory
        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_path = Mock()
            mock_coral = Mock()
            mock_coral.exists.return_value = True
            mock_path.__truediv__ = Mock(return_value=mock_coral)
            mock_cwd.return_value = mock_path

            result = cli._find_repo_root()
            assert result == mock_path

        # Test when .coral doesn't exist anywhere
        with patch("pathlib.Path.cwd") as mock_cwd:
            # Create a chain of paths
            root = Mock()
            root.parent = root  # Root is its own parent

            path = Mock()
            path.parent = root
            mock_coral = Mock()
            mock_coral.exists.return_value = False
            path.__truediv__ = Mock(return_value=mock_coral)
            root.__truediv__ = Mock(return_value=mock_coral)

            mock_cwd.return_value = path

            result = cli._find_repo_root()
            assert result is None

    def test_cli_run_not_in_repo(self, cli):
        """Test CLI run when not in a repository."""
        args = argparse.Namespace(command="status")
        with patch.object(cli.parser, "parse_args", return_value=args):
            with patch.object(cli, "_find_repo_root", return_value=None):
                with patch("builtins.print") as mock_print:
                    result = cli.run()
                    assert result == 1
                    mock_print.assert_called_with(
                        "Error: Not in a Coral repository", file=sys.stderr
                    )

    def test_cli_run_no_command(self, cli):
        """Test CLI run with no command."""
        # Mock parser to return args with no command
        args = argparse.Namespace(command=None)
        with patch.object(cli.parser, "parse_args", return_value=args):
            with patch.object(cli.parser, "print_help") as mock_help:
                result = cli.run()
                assert result == 0
                mock_help.assert_called_once()

    def test_cli_run_unknown_command(self, cli):
        """Test CLI run with unknown command."""
        args = argparse.Namespace(command="unknown")
        with patch.object(cli.parser, "parse_args", return_value=args):
            with patch("builtins.print") as mock_print:
                result = cli.run()
                assert result == 1
                mock_print.assert_called_with(
                    "Error: Unknown command 'unknown'", file=sys.stderr
                )
