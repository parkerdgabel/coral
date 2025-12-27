"""Comprehensive CLI coverage tests."""

import argparse
from unittest.mock import Mock, patch

from coral.cli.main import CoralCLI, main


class TestCLIComprehensive:
    """Comprehensive CLI tests for coverage."""

    def test_cli_main_function(self):
        """Test the main() entry point."""
        # Test main with init command
        test_args = ["coral", "init"]
        with patch("sys.argv", test_args):
            with patch("coral.cli.main.CoralCLI") as mock_cli_class:
                mock_cli = Mock()
                mock_cli.run.return_value = 0
                mock_cli_class.return_value = mock_cli

                with patch("sys.exit") as mock_exit:
                    main()

                    mock_cli_class.assert_called_once()
                    mock_cli.run.assert_called_once()
                    mock_exit.assert_called_once_with(0)

    def test_cli_run_method_all_commands(self):
        """Test CLI run method routing to all commands."""
        cli = CoralCLI()

        # Create mock methods for all commands
        cli._cmd_init = Mock(return_value=0)
        cli._cmd_add = Mock(return_value=0)
        cli._cmd_commit = Mock(return_value=0)
        cli._cmd_status = Mock(return_value=0)
        cli._cmd_log = Mock(return_value=0)
        cli._cmd_checkout = Mock(return_value=0)
        cli._cmd_branch = Mock(return_value=0)
        cli._cmd_merge = Mock(return_value=0)
        cli._cmd_diff = Mock(return_value=0)
        cli._cmd_tag = Mock(return_value=0)
        cli._cmd_show = Mock(return_value=0)
        cli._cmd_gc = Mock(return_value=0)

        # Test each command with necessary args
        test_cases = [
            ("init", ["init"], cli._cmd_init, False),
            ("add", ["add", "file.npy"], cli._cmd_add, True),
            ("commit", ["commit", "-m", "test"], cli._cmd_commit, True),
            ("status", ["status"], cli._cmd_status, True),
            ("log", ["log"], cli._cmd_log, True),
            ("checkout", ["checkout", "branch"], cli._cmd_checkout, True),
            ("branch", ["branch"], cli._cmd_branch, True),
            ("merge", ["merge", "branch"], cli._cmd_merge, True),
            ("diff", ["diff", "ref1"], cli._cmd_diff, True),
            ("tag", ["tag", "v1.0"], cli._cmd_tag, True),
            ("show", ["show", "weight"], cli._cmd_show, True),
            ("gc", ["gc"], cli._cmd_gc, True),
        ]

        for cmd_name, cmd_args, mock_method, needs_repo in test_cases:
            # Reset mocks
            cli._cmd_init.reset_mock()
            cli._cmd_add.reset_mock()
            cli._cmd_commit.reset_mock()
            cli._cmd_status.reset_mock()
            cli._cmd_log.reset_mock()
            cli._cmd_checkout.reset_mock()
            cli._cmd_branch.reset_mock()
            cli._cmd_merge.reset_mock()
            cli._cmd_diff.reset_mock()
            cli._cmd_tag.reset_mock()
            cli._cmd_show.reset_mock()
            cli._cmd_gc.reset_mock()

            # Parse args
            args = cli.parser.parse_args(cmd_args)

            # For init, no need to mock find_repo_root
            if not needs_repo:
                with patch.object(cli.parser, "parse_args", return_value=args):
                    result = cli.run()
                    assert result == 0
                    mock_method.assert_called_once()
            else:
                # For other commands, need to mock find_repo_root
                with patch.object(cli.parser, "parse_args", return_value=args):
                    with patch.object(
                        cli, "_find_repo_root", return_value="/fake/repo"
                    ):
                        result = cli.run()
                        assert result == 0
                        mock_method.assert_called_once()

    def test_cli_merge_command(self):
        """Test CLI merge command."""
        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            mock_commit = Mock()
            mock_commit.commit_hash = "merged123"
            mock_commit.metadata = Mock()
            mock_commit.metadata.message = "Merge branch feature"
            mock_repo.merge.return_value = mock_commit
            mock_repo.branch_manager.get_current_branch.return_value = "main"

            from pathlib import Path

            with patch("builtins.print"):
                args = argparse.Namespace(
                    command="merge", branch="feature", message=None
                )
                result = cli._cmd_merge(args, Path("."))

                mock_repo.merge.assert_called_once_with("feature", message=None)
                assert result == 0

    def test_cli_diff_command(self):
        """Test CLI diff command."""
        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            mock_repo.diff.return_value = {
                "added": ["weight1"],
                "modified": {"weight2": {"from_hash": "h1", "to_hash": "h2"}},
                "removed": ["weight3"],
                "summary": {
                    "total_added": 1,
                    "total_modified": 1,
                    "total_removed": 1,
                },
            }

            from pathlib import Path

            with patch("builtins.print"):
                args = argparse.Namespace(
                    command="diff", from_ref="main", to_ref="feature"
                )
                result = cli._cmd_diff(args, Path("."))

                mock_repo.diff.assert_called_once_with("main", "feature")
                assert result == 0

    def test_cli_show_command(self):
        """Test CLI show command."""
        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            # Mock weight
            mock_weight = Mock()
            mock_weight.metadata = Mock()
            mock_weight.metadata.name = "test_weight"
            mock_weight.metadata.layer_type = None
            mock_weight.metadata.model_name = None
            mock_weight.shape = (10, 20)
            mock_weight.dtype = "float32"
            mock_weight.nbytes = 800
            mock_weight.compute_hash.return_value = "hash123"
            mock_weight.data = Mock()
            mock_weight.data.min.return_value = 0.0
            mock_weight.data.max.return_value = 1.0
            mock_weight.data.mean.return_value = 0.5
            mock_weight.data.std.return_value = 0.1

            mock_repo.get_weight.return_value = mock_weight

            from pathlib import Path

            with patch("builtins.print"):
                args = argparse.Namespace(
                    command="show", weight="test_weight", commit=None
                )
                result = cli._cmd_show(args, Path("."))

                mock_repo.get_weight.assert_called_once_with(
                    "test_weight", commit_ref=None
                )
                assert result == 0

    def test_cli_gc_command(self):
        """Test CLI garbage collection command."""
        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            mock_repo.gc.return_value = {
                "cleaned_weights": 10,
                "remaining_weights": 5,
            }

            from pathlib import Path

            with patch("builtins.print"):
                args = argparse.Namespace(command="gc")
                result = cli._cmd_gc(args, Path("."))

                mock_repo.gc.assert_called_once()
                assert result == 0

    def test_cli_format_bytes(self):
        """Test _format_bytes helper."""
        cli = CoralCLI()

        # Test various byte sizes
        assert "1.0 B" in cli._format_bytes(1)
        assert "1.0 KB" in cli._format_bytes(1024)
        assert "1.0 MB" in cli._format_bytes(1024 * 1024)
        assert "1.0 GB" in cli._format_bytes(1024 * 1024 * 1024)

    def test_cli_resolve_ref(self):
        """Test _resolve_ref helper."""
        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            # Mock commit in version graph
            mock_commit = Mock()
            mock_commit.commit_hash = "abc123"
            mock_repo.version_graph.get_commit.return_value = mock_commit

            result = cli._resolve_ref(mock_repo, "abc123")
            assert result == mock_commit

    def test_cli_error_handling(self):
        """Test CLI error message handling."""
        cli = CoralCLI()

        # Test exception during command execution
        args = argparse.Namespace(command="status")
        with patch.object(cli.parser, "parse_args", return_value=args):
            with patch.object(cli, "_find_repo_root", return_value="/fake/repo"):
                with patch.object(
                    cli, "_cmd_status", side_effect=Exception("Test error")
                ):
                    with patch("builtins.print"):
                        result = cli.run()
                        assert result == 1
