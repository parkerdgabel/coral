"""Direct CLI coverage tests."""

from unittest.mock import Mock, patch

from coral.cli.main import CoralCLI


class TestCLIDirectCoverage:
    """Direct CLI method tests for coverage."""

    def test_cli_has_run_methods(self):
        """Test CLI has all expected _run_* methods."""
        cli = CoralCLI()

        expected_methods = [
            "_run_init",
            "_run_add",
            "_run_commit",
            "_run_status",
            "_run_log",
            "_run_checkout",
            "_run_branch",
            "_run_merge",
            "_run_diff",
            "_run_tag",
            "_run_show",
            "_run_gc",
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

        # Test gc with dry-run
        args = cli.parser.parse_args(["gc", "--dry-run"])
        assert args.command == "gc"
        assert args.dry_run is True

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
                mock_repo.assert_called_with(".", init=True)
                mock_exit.assert_called_with(0)

    def test_cli_error_handling_pattern(self):
        """Test CLI error handling pattern."""
        cli = CoralCLI()

        # Create a mock args object
        args = Mock()
        args.command = "status"

        # Mock Repository to raise error
        with patch("coral.cli.main.Repository", side_effect=ValueError("Not a repo")):
            with patch("coral.cli.main.print") as mock_print:
                with patch("coral.cli.main.sys.exit") as mock_exit:
                    # Parse args and run
                    with patch.object(cli.parser, "parse_args", return_value=args):
                        cli.run()

                    # Should print error and exit(1)
                    mock_print.assert_called()
                    mock_exit.assert_called_with(1)
