"""CLI commands coverage tests."""

import argparse
from pathlib import Path
from unittest.mock import Mock, patch

from coral.cli.main import CoralCLI


class TestCLICommandsCoverage:
    """Test remaining CLI commands for coverage."""

    def test_cli_cmd_checkout(self):
        """Test _cmd_checkout."""
        cli = CoralCLI()

        args = argparse.Namespace(target="feature-branch")
        repo_path = Path(".")

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            with patch("builtins.print") as _mock_print:
                result = cli._cmd_checkout(args, repo_path)
                assert result == 0
                mock_repo.checkout.assert_called_once_with("feature-branch")
                _mock_print.assert_called()

    def test_cli_cmd_branch_list(self):
        """Test _cmd_branch list mode."""
        cli = CoralCLI()

        args = argparse.Namespace(name=None, delete=None, list=True)
        repo_path = Path(".")

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            # Mock branch_manager
            mock_branch_manager = Mock()
            mock_repo.branch_manager = mock_branch_manager

            # Mock branches
            branches = []
            for name in ["main", "develop", "feature"]:
                branch = Mock()
                branch.name = name
                branches.append(branch)
            mock_branch_manager.list_branches.return_value = branches
            mock_branch_manager.get_current_branch.return_value = "main"

            with patch("builtins.print") as mock_print:
                result = cli._cmd_branch(args, repo_path)
                assert result == 0
                assert mock_print.call_count == 3  # One per branch
                # Check that current branch is marked
                calls = mock_print.call_args_list
                assert any("*" in str(call) for call in calls)

    def test_cli_cmd_branch_create(self):
        """Test _cmd_branch create mode."""
        cli = CoralCLI()

        args = argparse.Namespace(name="new-feature", delete=None, list=False)
        repo_path = Path(".")

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            with patch("builtins.print") as mock_print:
                result = cli._cmd_branch(args, repo_path)
                assert result == 0
                mock_repo.create_branch.assert_called_once_with("new-feature")
                mock_print.assert_called()

    def test_cli_cmd_branch_delete(self):
        """Test _cmd_branch delete mode."""
        cli = CoralCLI()

        args = argparse.Namespace(name=None, delete="old-feature", list=False)
        repo_path = Path(".")

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            mock_branch_manager = Mock()
            mock_repo.branch_manager = mock_branch_manager

            with patch("builtins.print") as _mock_print:
                result = cli._cmd_branch(args, repo_path)
                assert result == 0
                mock_branch_manager.delete_branch.assert_called_once_with("old-feature")
                _mock_print.assert_called()

    def test_cli_cmd_merge(self):
        """Test _cmd_merge."""
        cli = CoralCLI()

        args = argparse.Namespace(branch="feature", message=None)
        repo_path = Path(".")

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            # Mock branch_manager
            mock_branch_manager = Mock()
            mock_repo.branch_manager = mock_branch_manager
            mock_branch_manager.get_current_branch.return_value = "main"

            # Mock commit
            mock_commit = Mock()
            mock_commit.commit_hash = "merge123abcdef"
            mock_commit.metadata = Mock()
            mock_commit.metadata.message = "Merge feature into main"
            mock_repo.merge.return_value = mock_commit

            with patch("builtins.print") as _mock_print:
                result = cli._cmd_merge(args, repo_path)
                assert result == 0
                mock_repo.merge.assert_called_once_with("feature", message=None)
                assert _mock_print.call_count == 2  # Two print statements

    def test_cli_cmd_diff(self):
        """Test _cmd_diff."""
        cli = CoralCLI()

        args = argparse.Namespace(from_ref="main", to_ref="feature")
        repo_path = Path(".")

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            mock_repo.diff.return_value = {
                "added": ["w1", "w2", "w3"],
                "modified": {
                    "w4": {"from_hash": "h1", "to_hash": "h2"},
                    "w5": {"from_hash": "h3", "to_hash": "h4"},
                },
                "removed": ["w6"],
                "summary": {"total_added": 3, "total_removed": 1, "total_modified": 2},
            }

            with patch("builtins.print") as _mock_print:
                result = cli._cmd_diff(args, repo_path)
                assert result == 0
                mock_repo.diff.assert_called_once_with("main", "feature")
                # Should print header + summary (4 lines)
                assert _mock_print.call_count >= 4

    def test_cli_cmd_diff_default_to_ref(self):
        """Test _cmd_diff with default to_ref."""
        cli = CoralCLI()

        args = argparse.Namespace(from_ref="main", to_ref=None)
        repo_path = Path(".")

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            mock_repo.diff.return_value = {
                "added": [],
                "modified": {},
                "removed": [],
                "summary": {"total_added": 0, "total_removed": 0, "total_modified": 0},
            }

            with patch("builtins.print") as _mock_print:
                result = cli._cmd_diff(args, repo_path)
                assert result == 0
                mock_repo.diff.assert_called_once_with("main", None)
                # Should print at least the summary
                assert _mock_print.call_count >= 4

    def test_cli_cmd_tag(self):
        """Test _cmd_tag."""
        cli = CoralCLI()

        args = argparse.Namespace(
            name="v1.0.0",
            description="First release",
            commit=None,
            metric=["accuracy=0.95", "loss=0.05"],
        )
        repo_path = Path(".")

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            mock_version = Mock()
            mock_version.name = "v1.0.0"
            mock_repo.tag_version.return_value = mock_version

            with patch("builtins.print") as _mock_print:
                result = cli._cmd_tag(args, repo_path)
                assert result == 0
                # Check metrics were parsed
                call_args = mock_repo.tag_version.call_args
                assert call_args[1]["metrics"] == {"accuracy": 0.95, "loss": 0.05}
                _mock_print.assert_called()

    def test_cli_cmd_tag_no_metrics(self):
        """Test _cmd_tag without metrics."""
        cli = CoralCLI()

        args = argparse.Namespace(
            name="v2.0.0", description=None, commit="abc123", metric=None
        )
        repo_path = Path(".")

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            mock_version = Mock()
            mock_version.name = "v2.0.0"
            mock_repo.tag_version.return_value = mock_version

            with patch("builtins.print") as _mock_print:
                result = cli._cmd_tag(args, repo_path)
                assert result == 0
                mock_repo.tag_version.assert_called_once()

    def test_cli_cmd_show(self):
        """Test _cmd_show."""
        cli = CoralCLI()

        args = argparse.Namespace(weight="layer.weight", commit=None)
        repo_path = Path(".")

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            # Mock weight
            mock_weight = Mock()
            mock_weight.metadata = Mock()
            mock_weight.metadata.name = "layer.weight"
            mock_weight.metadata.shape = [10, 20]
            mock_weight.metadata.dtype = Mock()
            mock_weight.metadata.dtype.__str__ = Mock(return_value="float32")
            mock_weight.metadata.layer_type = "Linear"
            mock_weight.metadata.model_name = "TestModel"
            mock_weight.data = Mock()
            mock_weight.data.mean = Mock(return_value=0.5)
            mock_weight.data.std = Mock(return_value=0.1)
            mock_weight.data.min = Mock(return_value=-1.0)
            mock_weight.data.max = Mock(return_value=1.0)
            mock_weight.data.size = 200
            mock_weight.nbytes = 800

            mock_repo.get_weight.return_value = mock_weight

            with patch("builtins.print") as _mock_print:
                result = cli._cmd_show(args, repo_path)
                assert result == 0
                # Should print many details
                assert _mock_print.call_count >= 8

    def test_cli_cmd_show_with_commit(self):
        """Test _cmd_show with specific commit."""
        cli = CoralCLI()

        args = argparse.Namespace(weight="layer.bias", commit="abc123")
        repo_path = Path(".")

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo

            # Mock weight - minimal version
            mock_weight = Mock()
            mock_weight.metadata = Mock()
            mock_weight.metadata.name = "layer.bias"
            mock_weight.shape = [10]
            mock_weight.dtype = Mock()
            mock_weight.dtype.__str__ = Mock(return_value="float32")
            mock_weight.metadata.layer_type = None
            mock_weight.metadata.model_name = None
            mock_weight.data = Mock()
            mock_weight.data.mean = Mock(return_value=0.0)
            mock_weight.data.std = Mock(return_value=0.0)
            mock_weight.data.min = Mock(return_value=0.0)
            mock_weight.data.max = Mock(return_value=0.0)
            mock_weight.data.size = 10
            mock_weight.nbytes = 40
            mock_weight.compute_hash = Mock(return_value="hash123")

            mock_repo.get_weight.return_value = mock_weight

            with patch("builtins.print") as _mock_print:
                result = cli._cmd_show(args, repo_path)
                assert result == 0
                mock_repo.get_weight.assert_called_once_with(
                    "layer.bias", commit_ref="abc123"
                )

    def test_cli_cmd_gc(self):
        """Test _cmd_gc."""
        cli = CoralCLI()

        args = argparse.Namespace()
        repo_path = Path(".")

        with patch("coral.cli.main.Repository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            mock_repo.gc.return_value = {"cleaned_weights": 15, "remaining_weights": 85}

            with patch("builtins.print") as _mock_print:
                result = cli._cmd_gc(args, repo_path)
                assert result == 0
                mock_repo.gc.assert_called_once()
                # Should print summary
                assert _mock_print.call_count >= 3

    def test_cli_error_handling_comprehensive(self):
        """Test comprehensive error handling in CLI."""
        cli = CoralCLI()

        # Test generic exception handling
        args = argparse.Namespace(command="status")
        with patch.object(cli.parser, "parse_args", return_value=args):
            with patch.object(cli, "_find_repo_root", return_value=Path(".")):
                with patch(
                    "coral.cli.main.Repository",
                    side_effect=Exception("Unexpected error"),
                ):
                    with patch("builtins.print") as _mock_print:
                        with patch("sys.stderr"):
                            result = cli.run()
                            assert result == 1
                            # Check error was printed
                            assert any(
                                "Unexpected error" in str(call)
                                for call in _mock_print.call_args_list
                            )

    def test_cli_parser_structure_comprehensive(self):
        """Test CLI parser has all expected commands and arguments."""
        cli = CoralCLI()

        # Get subparsers
        subparsers_actions = [
            action
            for action in cli.parser._actions
            if hasattr(action, "choices") and action.choices
        ]

        assert len(subparsers_actions) == 1
        subparsers = subparsers_actions[0].choices

        # Check all commands exist
        expected_commands = {
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
            "remote",
            "push",
            "pull",
            "clone",
            "stats",
            "compare",
            "sync",
            "sync-status",
        }
        assert set(subparsers.keys()) == expected_commands

        # Check some specific command arguments
        # Log command should have --oneline
        log_parser = subparsers["log"]
        log_actions = {action.dest for action in log_parser._actions}
        assert "oneline" in log_actions
        assert "number" in log_actions

        # Tag command should have metrics
        tag_parser = subparsers["tag"]
        tag_actions = {action.dest for action in tag_parser._actions}
        assert "metric" in tag_actions
        assert "description" in tag_actions
