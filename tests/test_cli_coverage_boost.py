"""CLI coverage boost to reach 80%."""

from unittest.mock import MagicMock, Mock, patch

from coral.cli.main import CoralCLI


class TestCLICoverageBoost:
    """Boost CLI coverage to reach 80% total."""

    def test_cli_all_cmd_methods(self):
        """Test all CLI _cmd_* methods exist and can be called."""
        cli = CoralCLI()

        # Get all _cmd_* methods
        run_methods = [attr for attr in dir(cli) if attr.startswith("_cmd_")]
        assert len(run_methods) >= 12  # Should have at least 12 command handlers

        # Test each method can be instantiated
        for method_name in run_methods:
            method = getattr(cli, method_name)
            assert callable(method)

    def test_cli_init_cmd_method(self):
        """Test _cmd_init method."""
        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo:
            with patch("builtins.print") as mock_print:
                args = Mock(path=".")
                cli._cmd_init(args)

                # The method calls Path(args.path).resolve()
                mock_repo.assert_called_once()
                assert mock_repo.call_args[1] == {"init": True}
                mock_print.assert_called()

    def test_cli_add_cmd_method(self):
        """Test _cmd_add method."""
        from pathlib import Path

        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            with patch("builtins.print") as mock_print:
                # Setup mocks
                mock_repo = Mock()
                mock_repo_class.return_value = mock_repo
                mock_repo.stage_weights.return_value = {"layer.weight": "hash123"}

                # Mock numpy load
                import numpy as np

                with patch("numpy.load") as mock_np_load:
                    mock_np_load.return_value = np.array([[1, 2], [3, 4]])

                    args = Mock(weights=["model.npy"])
                    repo_path = Path(".")
                    cli._cmd_add(args, repo_path)

                    mock_print.assert_called()

    def test_cli_commit_cmd_method(self):
        """Test _cmd_commit method."""
        from pathlib import Path

        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            with patch("builtins.print") as mock_print:
                mock_repo = Mock()
                mock_repo_class.return_value = mock_repo

                mock_commit = Mock()
                mock_commit.commit_hash = "abc123"
                mock_commit.metadata = Mock()
                mock_commit.metadata.message = "Test commit"
                mock_commit.weight_hashes = {"w1": "h1"}
                mock_repo.commit.return_value = mock_commit
                mock_repo.branch_manager.get_current_branch.return_value = "main"

                args = Mock(message="Test commit", author=None, email=None, tag=None)
                repo_path = Path(".")
                cli._cmd_commit(args, repo_path)

                mock_repo.commit.assert_called_once_with(
                    message="Test commit", author=None, email=None, tags=[]
                )
                mock_print.assert_called()

    def test_cli_status_cmd_method(self):
        """Test _cmd_status method."""
        from pathlib import Path

        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            with patch("builtins.print") as mock_print:
                mock_repo = Mock()
                mock_repo_class.return_value = mock_repo
                mock_repo.branch_manager.get_current_branch.return_value = "main"
                mock_repo.staging_dir = Path("/tmp/staging")

                args = Mock()
                repo_path = Path(".")
                cli._cmd_status(args, repo_path)

                # Should print status
                assert mock_print.call_count >= 1

    def test_cli_log_cmd_method(self):
        """Test _cmd_log method."""
        from pathlib import Path

        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            with patch("builtins.print") as mock_print:
                mock_repo = Mock()
                mock_repo_class.return_value = mock_repo

                # Mock commit
                mock_commit = Mock()
                mock_commit.commit_hash = "abc123"
                mock_commit.metadata = Mock()
                mock_commit.metadata.message = "Test message"
                mock_commit.metadata.author = "Test Author"
                mock_commit.metadata.email = "test@example.com"
                mock_commit.metadata.timestamp = "2025-01-01"
                mock_commit.metadata.tags = []
                mock_repo.log.return_value = [mock_commit]

                args = Mock(number=10, oneline=False)
                repo_path = Path(".")
                cli._cmd_log(args, repo_path)

                mock_repo.log.assert_called_once_with(max_commits=10)
                mock_print.assert_called()

    def test_cli_checkout_cmd_method(self):
        """Test _cmd_checkout method."""
        from pathlib import Path

        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            with patch("builtins.print") as mock_print:
                mock_repo = Mock()
                mock_repo_class.return_value = mock_repo

                args = Mock(target="feature-branch")
                repo_path = Path(".")
                cli._cmd_checkout(args, repo_path)

                mock_repo.checkout.assert_called_once_with("feature-branch")
                mock_print.assert_called()

    def test_cli_branch_cmd_method_all_modes(self):
        """Test _cmd_branch method in all modes."""
        from pathlib import Path

        cli = CoralCLI()
        repo_path = Path(".")

        # Test list mode
        with patch("coral.cli.main.Repository") as mock_repo_class:
            with patch("builtins.print"):
                mock_repo = Mock()
                mock_repo_class.return_value = mock_repo
                mock_repo.branch_manager.get_current_branch.return_value = "main"

                mock_branch = Mock()
                mock_branch.name = "main"
                mock_repo.branch_manager.list_branches.return_value = [mock_branch]

                args = Mock(name=None, delete=None, list=True)
                cli._cmd_branch(args, repo_path)
                mock_repo.branch_manager.list_branches.assert_called_once()

        # Test create mode
        with patch("coral.cli.main.Repository") as mock_repo_class:
            with patch("builtins.print"):
                mock_repo = Mock()
                mock_repo_class.return_value = mock_repo

                args = Mock()
                args.name = "new-feature"
                args.delete = None
                args.list = False
                cli._cmd_branch(args, repo_path)
                mock_repo.create_branch.assert_called_once_with("new-feature")

        # Test delete mode
        with patch("coral.cli.main.Repository") as mock_repo_class:
            with patch("builtins.print"):
                mock_repo = Mock()
                mock_repo_class.return_value = mock_repo

                args = Mock(name=None, delete="old-feature", list=False)
                cli._cmd_branch(args, repo_path)
                mock_repo.branch_manager.delete_branch.assert_called_once_with(
                    "old-feature"
                )

    def test_cli_merge_cmd_method(self):
        """Test _cmd_merge method."""
        from pathlib import Path

        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            with patch("builtins.print") as mock_print:
                mock_repo = Mock()
                mock_repo_class.return_value = mock_repo

                mock_commit = Mock()
                mock_commit.commit_hash = "merged123"
                mock_commit.metadata = Mock()
                mock_commit.metadata.message = "Merge feature"
                mock_repo.merge.return_value = mock_commit
                mock_repo.branch_manager.get_current_branch.return_value = "main"

                args = Mock(branch="feature", message=None)
                repo_path = Path(".")
                cli._cmd_merge(args, repo_path)

                mock_repo.merge.assert_called_once_with("feature", message=None)
                mock_print.assert_called()

    def test_cli_diff_cmd_method(self):
        """Test _cmd_diff method."""
        from pathlib import Path

        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            with patch("builtins.print") as mock_print:
                mock_repo = Mock()
                mock_repo_class.return_value = mock_repo
                mock_repo.diff.return_value = {
                    "added": ["w1", "w2"],
                    "modified": {"w3": {"from_hash": "h1", "to_hash": "h2"}},
                    "removed": [],
                    "summary": {
                        "total_added": 2,
                        "total_modified": 1,
                        "total_removed": 0,
                    },
                }

                args = Mock(from_ref="main", to_ref="feature")
                repo_path = Path(".")
                cli._cmd_diff(args, repo_path)

                mock_repo.diff.assert_called_once_with("main", "feature")
                # Should print diff details
                assert mock_print.call_count >= 4

    def test_cli_tag_cmd_method(self):
        """Test _cmd_tag method."""
        from pathlib import Path

        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            with patch("builtins.print") as mock_print:
                mock_repo = Mock()
                mock_repo_class.return_value = mock_repo

                mock_version = Mock()
                mock_version.name = "v1.0.0"
                mock_version.version_id = "vid123"
                mock_repo.tag_version.return_value = mock_version

                args = Mock(
                    name="v1.0.0", description="First release", metric=None, commit=None
                )
                repo_path = Path(".")
                cli._cmd_tag(args, repo_path)

                mock_repo.tag_version.assert_called_once()
                mock_print.assert_called()

    def test_cli_show_cmd_method(self):
        """Test _cmd_show method."""
        from pathlib import Path

        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            with patch("builtins.print") as mock_print:
                mock_repo = Mock()
                mock_repo_class.return_value = mock_repo

                # Mock weight
                mock_weight = Mock()
                mock_weight.metadata = Mock()
                mock_weight.metadata.name = "test_weight"
                mock_weight.metadata.layer_type = None
                mock_weight.metadata.model_name = None
                mock_weight.shape = [10, 20]
                mock_weight.dtype = Mock()
                mock_weight.dtype.__str__ = Mock(return_value="float32")
                mock_weight.nbytes = 800
                mock_weight.compute_hash = Mock(return_value="hash123")
                mock_weight.data = Mock()
                mock_weight.data.mean = Mock(return_value=0.5)
                mock_weight.data.std = Mock(return_value=0.1)
                mock_weight.data.min = Mock(return_value=0.0)
                mock_weight.data.max = Mock(return_value=1.0)

                mock_repo.get_weight.return_value = mock_weight

                args = Mock(weight="test_weight", commit=None)
                repo_path = Path(".")
                cli._cmd_show(args, repo_path)

                mock_repo.get_weight.assert_called_once_with(
                    "test_weight", commit_ref=None
                )
                # Should print weight details
                assert mock_print.call_count >= 5

    def test_cli_gc_cmd_method(self):
        """Test _cmd_gc method."""
        from pathlib import Path

        cli = CoralCLI()

        with patch("coral.cli.main.Repository") as mock_repo_class:
            with patch("builtins.print") as mock_print:
                mock_repo = Mock()
                mock_repo_class.return_value = mock_repo
                mock_repo.gc.return_value = {
                    "cleaned_weights": 10,
                    "remaining_weights": 90,
                }

                args = Mock()
                repo_path = Path(".")
                cli._cmd_gc(args, repo_path)

                mock_repo.gc.assert_called_once()
                mock_print.assert_called()

    def test_cli_cmd_method_dispatch(self):
        """Test CLI run method dispatches to correct handler."""
        cli = CoralCLI()

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
                args.path = "model.pth"
            elif cmd == "commit":
                args.message = "test"
                args.author = None
                args.email = None
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
            elif cmd == "show":
                args.weight_name = "weight"
            elif cmd == "gc":
                args.dry_run = False
            elif cmd == "init":
                args.path = "."

            with patch.object(cli.parser, "parse_args", return_value=args):
                cli.run()

            # Verify correct method was called
            getattr(cli, f"_cmd_{cmd}").assert_called_once()
