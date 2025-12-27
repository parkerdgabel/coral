"""Direct CLI coverage tests."""

from unittest.mock import Mock, patch

from coral.cli.main import CoralCLI


class TestCLIDirectCoverage:
    """Direct CLI method tests for coverage."""

    def test_cli_has_run_methods(self):
        """Test CLI has all expected _cmd_* methods."""
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

    def test_cli_parser_structure(self):
        """Test CLI parser has correct structure."""
        cli = CoralCLI()

        # Parser should exist
        assert cli.parser is not None

        # Should have subparsers
        assert hasattr(cli.parser, "_subparsers")

    def test_cli_simple_command_parse(self):
        """Test simple command parsing."""
        cli = CoralCLI()

        # Test init
        args = cli.parser.parse_args(["init"])
        assert args.command == "init"

        # Test status
        args = cli.parser.parse_args(["status"])
        assert args.command == "status"

        # Test gc
        args = cli.parser.parse_args(["gc"])
        assert args.command == "gc"

    @patch("coral.cli.main.sys.exit")
    @patch("coral.cli.main.print")
    def test_main_function_with_mock(self, mock_print, mock_exit):
        """Test main function with mocked components."""
        # Mock sys.argv
        with patch("sys.argv", ["coral", "init"]):
            with patch("coral.cli.main.Repository") as mock_repo:
                # Call main
                from coral.cli.main import main

                main()

                # Should create repository with init=True
                # The method calls Path(args.path).resolve()
                mock_repo.assert_called_once()
                assert mock_repo.call_args[1] == {"init": True}
                mock_exit.assert_called_with(0)

    def test_cli_error_handling_pattern(self):
        """Test CLI error handling pattern."""
        from pathlib import Path

        cli = CoralCLI()

        # Create a mock args object
        args = Mock()
        args.command = "status"

        # Mock Repository to raise error
        with patch("coral.cli.main.Repository", side_effect=ValueError("Not a repo")):
            with patch("builtins.print"):
                with patch.object(cli, "_find_repo_root", return_value=Path(".")):
                    # Parse args and run
                    with patch.object(cli.parser, "parse_args", return_value=args):
                        result = cli.run()

                    # Should return 1 on error
                    assert result == 1
