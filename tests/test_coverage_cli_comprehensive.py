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
                mock_cli_class.return_value = mock_cli

                main()

                mock_cli_class.assert_called_once()
                mock_cli.run.assert_called_once()

    def test_cli_run_method_all_commands(self):
        """Test CLI run method routing to all commands."""
        cli = CoralCLI()

        # Create mock methods for all commands
        cli._run_init = Mock()
        cli._run_add = Mock()
        cli._run_commit = Mock()
        cli._run_status = Mock()
        cli._run_log = Mock()
        cli._run_checkout = Mock()
        cli._run_branch = Mock()
        cli._run_merge = Mock()
        cli._run_diff = Mock()
        cli._run_tag = Mock()
        cli._run_show = Mock()
        cli._run_gc = Mock()

        # Test each command
        commands = [
            ("init", cli._run_init),
            ("add", cli._run_add),
            ("commit", cli._run_commit),
            ("status", cli._run_status),
            ("log", cli._run_log),
            ("checkout", cli._run_checkout),
            ("branch", cli._run_branch),
            ("merge", cli._run_merge),
            ("diff", cli._run_diff),
            ("tag", cli._run_tag),
            ("show", cli._run_show),
            ("gc", cli._run_gc),
        ]

        for cmd_name, mock_method in commands:
            with patch("sys.argv", ["coral", cmd_name]):
                # Reset mocks
                for _, method in commands:
                    method.reset_mock()

                # Parse args
                args = cli.parser.parse_args([cmd_name])
                args.command = cmd_name

                # Call run with mocked args
                with patch.object(cli.parser, "parse_args", return_value=args):
                    cli.run()

                # Verify only the right method was called
                mock_method.assert_called_once()

    def test_cli_merge_command(self):
        """Test CLI merge command."""
        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            mock_commit = Mock()
            mock_commit.commit_hash = "merged123"
            mock_repo.merge.return_value = mock_commit

            with patch("sys.exit") as mock_exit:
                args = argparse.Namespace(command="merge", branch="feature")
                cli._run_merge(args)

                mock_repo.merge.assert_called_once_with("feature")
                mock_exit.assert_called_once_with(0)

    def test_cli_diff_command(self):
        """Test CLI diff command."""
        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            mock_repo.diff.return_value = {
                "added": ["weight1"],
                "modified": ["weight2"],
                "removed": ["weight3"],
            }

            with patch("sys.exit") as mock_exit:
                args = argparse.Namespace(
                    command="diff", from_ref="main", to_ref="feature"
                )
                cli._run_diff(args)

                mock_repo.diff.assert_called_once_with("main", "feature")
                mock_exit.assert_called_once_with(0)

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
            mock_weight.metadata.shape = (10, 20)
            mock_weight.metadata.dtype = "float32"
            mock_weight.data = Mock()
            mock_weight.data.mean = Mock(return_value=0.5)
            mock_weight.data.std = Mock(return_value=0.1)

            mock_repo.get_weight.return_value = mock_weight

            with patch("sys.exit") as mock_exit:
                args = argparse.Namespace(command="show", weight_name="test_weight")
                cli._run_show(args)

                mock_repo.get_weight.assert_called_once_with("test_weight")
                mock_exit.assert_called_once_with(0)

    def test_cli_gc_command(self):
        """Test CLI garbage collection command."""
        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            mock_repo.gc.return_value = {
                "removed_weights": 10,
                "removed_deltas": 5,
                "bytes_freed": 1024 * 1024 * 100,  # 100MB
            }

            with patch("sys.exit") as mock_exit:
                args = argparse.Namespace(command="gc", dry_run=False)
                cli._run_gc(args)

                mock_repo.gc.assert_called_once_with(dry_run=False)
                mock_exit.assert_called_once_with(0)

    def test_cli_load_weights_from_path(self):
        """Test load_weights_from_path function."""
        from coral.cli.main import load_weights_from_path

        # Mock torch
        with patch("coral.cli.main.torch") as mock_torch:
            # Setup mock tensor
            mock_tensor = Mock()
            mock_tensor.numpy.return_value = Mock()
            mock_tensor.shape = (10, 20)
            mock_tensor.dtype = Mock()

            # Setup state dict
            state_dict = {"layer1.weight": mock_tensor, "layer1.bias": mock_tensor}

            mock_torch.load.return_value = state_dict

            # Test loading
            weights = load_weights_from_path("model.pth")

            assert len(weights) == 2
            assert "layer1.weight" in weights
            assert "layer1.bias" in weights

    def test_cli_print_helpers(self):
        """Test CLI print helper functions."""
        cli = CoralCLI()

        # Test _print_status
        with patch("builtins.print") as mock_print:
            status = {
                "branch": "main",
                "staged": {"w1": "hash1"},
                "modified": {"w2": "hash2"},
            }
            cli._print_status(status)

            # Verify print was called
            assert mock_print.call_count > 0

        # Test _print_diff
        with patch("builtins.print") as mock_print:
            diff = {"added": ["w1"], "modified": ["w2"], "removed": ["w3"]}
            cli._print_diff(diff)

            # Verify print was called
            assert mock_print.call_count > 0

    def test_cli_error_messages(self):
        """Test CLI error message handling."""
        cli = CoralCLI()

        # Test various error conditions
        with patch("coral.cli.main.Repository") as mock_repo_class:
            # Not a repository error
            mock_repo_class.side_effect = ValueError("Not a Coral repository")

            with patch("builtins.print") as mock_print:
                with patch("sys.exit") as mock_exit:
                    args = argparse.Namespace(command="status")
                    cli._run_status(args)

                    # Check error was printed
                    mock_print.assert_called()
                    printed = " ".join(
                        str(arg)
                        for call in mock_print.call_args_list
                        for arg in call[0]
                    )
                    assert "Error" in printed or "error" in printed
                    mock_exit.assert_called_with(1)
