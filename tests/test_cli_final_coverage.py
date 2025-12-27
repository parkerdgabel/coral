"""Final CLI coverage tests to reach 80%."""

import argparse
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from coral.cli.main import CoralCLI, main


class TestCLIFinalCoverage:
    """Final CLI tests to boost coverage to 80%."""

    def test_cli_all_cmd_methods_exist(self):
        """Test all CLI _cmd_* methods exist."""
        cli = CoralCLI()

        expected_methods = [
            "_cmd_init",
            "_cmd_add",
            "_cmd_commit",
            "_cmd_status",
            "_cmd_log",
            "_cmd_checkout",
            "_cmd_branch",
            "_cmd_merge",
            "_cmd_diff",
            "_cmd_tag",
            "_cmd_show",
            "_cmd_gc",
        ]

        for method in expected_methods:
            assert hasattr(cli, method)
            assert callable(getattr(cli, method))

    def test_cli_find_repo_root(self):
        """Test _find_repo_root method."""
        cli = CoralCLI()

        # Test when .coral exists in current directory
        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_path = Mock(spec=Path)
            mock_coral = Mock()
            mock_coral.exists.return_value = True
            mock_path.__truediv__ = Mock(return_value=mock_coral)
            mock_cwd.return_value = mock_path

            result = cli._find_repo_root()
            assert result == mock_path

        # Test when .coral doesn't exist anywhere
        with patch("pathlib.Path.cwd") as mock_cwd:
            # Create a chain of paths
            root = Mock(spec=Path)
            root.parent = root  # Root is its own parent

            path = Mock(spec=Path)
            path.parent = root
            mock_coral = Mock()
            mock_coral.exists.return_value = False
            path.__truediv__ = Mock(return_value=mock_coral)

            mock_cwd.return_value = path

            result = cli._find_repo_root()
            assert result is None

    def test_cli_run_no_command(self):
        """Test CLI run with no command."""
        cli = CoralCLI()

        # Mock parser to return args with no command
        args = argparse.Namespace(command=None)
        with patch.object(cli.parser, "parse_args", return_value=args):
            with patch.object(cli.parser, "print_help") as mock_help:
                result = cli.run()
                assert result == 0
                mock_help.assert_called_once()

    def test_cli_run_unknown_command(self):
        """Test CLI run with unknown command."""
        cli = CoralCLI()

        args = argparse.Namespace(command="unknown")
        with patch.object(cli.parser, "parse_args", return_value=args):
            with patch("builtins.print") as mock_print:
                result = cli.run()
                assert result == 1
                mock_print.assert_called_with(
                    "Error: Unknown command 'unknown'", file=sys.stderr
                )

    def test_cli_run_not_in_repo(self):
        """Test CLI run when not in a repository."""
        cli = CoralCLI()

        args = argparse.Namespace(command="status")
        with patch.object(cli.parser, "parse_args", return_value=args):
            with patch.object(cli, "_find_repo_root", return_value=None):
                with patch("builtins.print") as mock_print:
                    result = cli.run()
                    assert result == 1
                    mock_print.assert_called_with(
                        "Error: Not in a Coral repository", file=sys.stderr
                    )

    def test_cli_cmd_init_success(self):
        """Test _cmd_init success."""
        cli = CoralCLI()

        args = argparse.Namespace(path=".")
        with patch("coral.cli.main.Repository") as mock_repo:
            with patch("builtins.print") as mock_print:
                result = cli._cmd_init(args)
                assert result == 0
                mock_repo.assert_called_once()
                mock_print.assert_called_once()

    def test_cli_cmd_init_error(self):
        """Test _cmd_init error."""
        cli = CoralCLI()

        args = argparse.Namespace(path=".")
        with patch(
            "coral.cli.main.Repository", side_effect=ValueError("Already exists")
        ):
            with patch("builtins.print") as mock_print:
                result = cli._cmd_init(args)
                assert result == 1
                assert mock_print.call_count == 1

    def test_cli_cmd_add_npy_file(self):
        """Test _cmd_add with .npy file."""
        cli = CoralCLI()

        args = argparse.Namespace(weights=["model.npy"])
        repo_path = Path(".")

        with patch("coral.cli.main.Repository") as mock_repo_class:
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.suffix", ".npy"):
                    with patch("pathlib.Path.stem", "model"):
                        with patch("numpy.load", return_value=np.array([1, 2, 3])):
                            with patch("builtins.print") as mock_print:
                                mock_repo = Mock()
                                mock_repo.stage_weights.return_value = {"model": "hash123"}
                                mock_repo_class.return_value = mock_repo

                                result = cli._cmd_add(args, repo_path)
                                assert result == 0
                                mock_repo.stage_weights.assert_called_once()
                                mock_print.assert_called()

    def test_cli_cmd_add_npz_file(self):
        """Test _cmd_add with .npz file."""
        cli = CoralCLI()

        args = argparse.Namespace(weights=["model.npz"])
        repo_path = Path(".")

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo.stage_weights.return_value = {"layer1": "hash1", "layer2": "hash2"}
            mock_repo_class.return_value = mock_repo

            # Create a mock Path object with proper attributes
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path.suffix = ".npz"

            with patch("coral.cli.main.Path", return_value=mock_path):
                # Mock numpy archive
                mock_archive = {"layer1": np.array([1, 2]), "layer2": np.array([3, 4])}
                with patch("numpy.load", return_value=mock_archive):
                    with patch("builtins.print") as _mock_print:
                        result = cli._cmd_add(args, repo_path)
                        assert result == 0
                        mock_repo.stage_weights.assert_called()

    def test_cli_cmd_add_unsupported_format(self):
        """Test _cmd_add with unsupported file format."""
        cli = CoralCLI()

        args = argparse.Namespace(weights=["model.txt"])
        repo_path = Path(".")

        with patch("coral.cli.main.Repository"):
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path.suffix = ".txt"

            with patch("coral.cli.main.Path", return_value=mock_path):
                with patch("builtins.print") as mock_print:
                    result = cli._cmd_add(args, repo_path)
                    assert result == 1
                    mock_print.assert_called_with(
                        "Error: Unsupported file format: .txt", file=sys.stderr
                    )

    def test_cli_cmd_add_file_not_found(self):
        """Test _cmd_add with non-existent file."""
        cli = CoralCLI()

        args = argparse.Namespace(weights=["missing.npy"])
        repo_path = Path(".")

        with patch("coral.cli.main.Repository"):
            with patch("pathlib.Path.exists", return_value=False):
                with patch("builtins.print") as mock_print:
                    result = cli._cmd_add(args, repo_path)
                    assert result == 1
                    assert "File not found" in str(mock_print.call_args)

    def test_cli_cmd_commit(self):
        """Test _cmd_commit."""
        cli = CoralCLI()

        args = argparse.Namespace(
            message="Test commit", author=None, email=None, tag=None
        )
        repo_path = Path(".")

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

            with patch("builtins.print") as mock_print:
                result = cli._cmd_commit(args, repo_path)
                assert result == 0
                mock_repo.commit.assert_called_once_with(
                    message="Test commit", author=None, email=None, tags=[]
                )
                mock_print.assert_called()

    def test_cli_cmd_status(self):
        """Test _cmd_status."""
        cli = CoralCLI()

        args = argparse.Namespace()
        repo_path = Path(".")

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            mock_repo.branch_manager.get_current_branch.return_value = "main"

            # Mock the staging directory and file
            mock_staging_dir = Mock()
            mock_staged_file = Mock()
            mock_staged_file.exists.return_value = True
            mock_staging_dir.__truediv__ = Mock(return_value=mock_staged_file)
            mock_repo.staging_dir = mock_staging_dir

            # Mock the file read
            with patch("builtins.open", create=True) as mock_open:
                import io
                mock_open.return_value.__enter__ = Mock(
                    return_value=io.StringIO('{"w1": "hash1", "w2": "hash2"}')
                )
                mock_open.return_value.__exit__ = Mock(return_value=False)

                with patch("json.load", return_value={"w1": "hash1", "w2": "hash2"}):
                    with patch("builtins.print") as mock_print:
                        result = cli._cmd_status(args, repo_path)
                        assert result == 0
                        assert mock_print.call_count >= 2  # Branch + staged weights

    def test_cli_cmd_log_oneline(self):
        """Test _cmd_log with oneline format."""
        cli = CoralCLI()

        args = argparse.Namespace(number=5, oneline=True)
        repo_path = Path(".")

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            # Create mock commits
            commits = []
            for i in range(3):
                mock_commit = Mock()
                mock_commit.commit_hash = f"hash{i}"
                mock_commit.metadata = Mock()
                mock_commit.metadata.message = f"Commit {i}"
                commits.append(mock_commit)

            mock_repo.log.return_value = commits

            with patch("builtins.print") as mock_print:
                result = cli._cmd_log(args, repo_path)
                assert result == 0
                assert mock_print.call_count == 3  # One per commit

    def test_cli_cmd_log_full(self):
        """Test _cmd_log with full format."""
        cli = CoralCLI()

        args = argparse.Namespace(number=10, oneline=False)
        repo_path = Path(".")

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            # Create mock commit with all details
            mock_commit = Mock()
            mock_commit.commit_hash = "fullhash123"
            mock_commit.metadata = Mock()
            mock_commit.metadata.message = "Full commit message"
            mock_commit.metadata.author = "Test Author"
            mock_commit.metadata.email = "test@example.com"
            mock_commit.metadata.timestamp = "2024-01-01T00:00:00"
            mock_commit.metadata.tags = ["tag1", "tag2"]
            mock_commit.weight_hashes = {"w1": "h1", "w2": "h2"}

            mock_repo.log.return_value = [mock_commit]

            with patch("builtins.print") as mock_print:
                result = cli._cmd_log(args, repo_path)
                assert result == 0
                # Should print commit details, author, date, tags, message
                assert mock_print.call_count >= 5

    def test_cli_main_function_integration(self):
        """Test main() function integration."""
        # Test successful init
        with patch("sys.argv", ["coral", "init", "."]):
            with patch("coral.cli.main.Repository") as _mock_repo:
                with patch("builtins.print"):
                    with patch("sys.exit") as mock_exit:
                        main()
                        mock_exit.assert_called_with(0)

        # Test error case
        with patch("sys.argv", ["coral", "commit", "-m", "test"]):
            with patch.object(CoralCLI, "_find_repo_root", return_value=None):
                with patch("builtins.print"):
                    with patch("sys.exit") as mock_exit:
                        main()
                        mock_exit.assert_called_with(1)
