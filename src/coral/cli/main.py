#!/usr/bin/env python3
"""
Coral CLI - Git-like version control for neural network weights
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.version_control.repository import Repository


class CoralCLI:
    """Main CLI interface for Coral."""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            prog="coral-ml", description="Version control for neural network weights"
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Init command
        init_parser = subparsers.add_parser("init", help="Initialize a new repository")
        init_parser.add_argument("path", nargs="?", default=".", help="Repository path")

        # Add command
        add_parser = subparsers.add_parser("add", help="Stage weights for commit")
        add_parser.add_argument("weights", nargs="+", help="Weight files to add")

        # Commit command
        commit_parser = subparsers.add_parser("commit", help="Commit staged weights")
        commit_parser.add_argument(
            "-m", "--message", required=True, help="Commit message"
        )
        commit_parser.add_argument("--author", help="Author name")
        commit_parser.add_argument("--email", help="Author email")
        commit_parser.add_argument("-t", "--tag", action="append", help="Add tags")

        # Status command
        subparsers.add_parser("status", help="Show repository status")

        # Log command
        log_parser = subparsers.add_parser("log", help="Show commit history")
        log_parser.add_argument(
            "-n", "--number", type=int, default=10, help="Number of commits"
        )
        log_parser.add_argument(
            "--oneline", action="store_true", help="Show compact output"
        )

        # Checkout command
        checkout_parser = subparsers.add_parser(
            "checkout", help="Checkout branch or commit"
        )
        checkout_parser.add_argument("target", help="Branch name or commit hash")

        # Branch command
        branch_parser = subparsers.add_parser("branch", help="Manage branches")
        branch_parser.add_argument("name", nargs="?", help="Branch name to create")
        branch_parser.add_argument("-d", "--delete", help="Delete branch")
        branch_parser.add_argument(
            "-l", "--list", action="store_true", help="List branches"
        )

        # Merge command
        merge_parser = subparsers.add_parser("merge", help="Merge branches")
        merge_parser.add_argument("branch", help="Branch to merge")
        merge_parser.add_argument("-m", "--message", help="Merge commit message")

        # Diff command
        diff_parser = subparsers.add_parser("diff", help="Show differences")
        diff_parser.add_argument("from_ref", help="From reference")
        diff_parser.add_argument(
            "to_ref", nargs="?", help="To reference (default: HEAD)"
        )

        # Tag command
        tag_parser = subparsers.add_parser("tag", help="Tag a version")
        tag_parser.add_argument("name", help="Version name")
        tag_parser.add_argument("-d", "--description", help="Version description")
        tag_parser.add_argument("-c", "--commit", help="Commit to tag (default: HEAD)")
        tag_parser.add_argument(
            "--metric", action="append", help="Add metric (format: key=value)"
        )

        # Show command
        show_parser = subparsers.add_parser("show", help="Show weight information")
        show_parser.add_argument("weight", help="Weight name")
        show_parser.add_argument("-c", "--commit", help="Commit reference")

        # GC command
        subparsers.add_parser("gc", help="Garbage collect unreferenced weights")

        # Remote command
        remote_parser = subparsers.add_parser("remote", help="Manage remote storage")
        remote_subparsers = remote_parser.add_subparsers(
            dest="remote_command", help="Remote commands"
        )

        # remote add
        remote_add = remote_subparsers.add_parser("add", help="Add a remote")
        remote_add.add_argument("name", help="Remote name (e.g., origin)")
        remote_add.add_argument(
            "url",
            help="Remote URL (s3://bucket/path, minio://host/bucket, file:///path)",
        )

        # remote remove
        remote_rm = remote_subparsers.add_parser("remove", help="Remove a remote")
        remote_rm.add_argument("name", help="Remote name to remove")

        # remote list
        remote_subparsers.add_parser("list", help="List remotes")

        # remote show
        remote_show = remote_subparsers.add_parser("show", help="Show remote details")
        remote_show.add_argument("name", help="Remote name")

        # Push command
        push_parser = subparsers.add_parser("push", help="Push weights to remote")
        push_parser.add_argument(
            "remote", nargs="?", default="origin", help="Remote name (default: origin)"
        )
        push_parser.add_argument("--all", action="store_true", help="Push all weights")
        push_parser.add_argument(
            "--force", "-f", action="store_true", help="Force push (overwrite remote)"
        )

        # Pull command
        pull_parser = subparsers.add_parser("pull", help="Pull weights from remote")
        pull_parser.add_argument(
            "remote", nargs="?", default="origin", help="Remote name (default: origin)"
        )
        pull_parser.add_argument("--all", action="store_true", help="Pull all weights")
        pull_parser.add_argument(
            "--force", "-f", action="store_true", help="Force pull (overwrite local)"
        )

        # Clone command
        clone_parser = subparsers.add_parser("clone", help="Clone a remote repository")
        clone_parser.add_argument("url", help="Remote URL to clone")
        clone_parser.add_argument(
            "path", nargs="?", default=".", help="Local path (default: current dir)"
        )

        # Stats command
        stats_parser = subparsers.add_parser(
            "stats", help="Show repository statistics and storage savings"
        )
        stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

        # Compare command
        compare_parser = subparsers.add_parser(
            "compare", help="Compare weights between commits or branches"
        )
        compare_parser.add_argument(
            "ref1", help="First reference (commit, branch, or tag)"
        )
        compare_parser.add_argument(
            "ref2", nargs="?", help="Second reference (default: HEAD)"
        )
        compare_parser.add_argument(
            "-v", "--verbose", action="store_true", help="Show per-layer details"
        )

        # Sync command
        sync_parser = subparsers.add_parser(
            "sync", help="Bidirectional sync with remote (push and pull)"
        )
        sync_parser.add_argument(
            "remote", nargs="?", default="origin", help="Remote name (default: origin)"
        )

        # Sync-status command
        sync_status_parser = subparsers.add_parser(
            "sync-status", help="Show sync status with remote"
        )
        sync_status_parser.add_argument(
            "remote", nargs="?", default="origin", help="Remote name (default: origin)"
        )
        sync_status_parser.add_argument(
            "--json", action="store_true", help="Output as JSON"
        )

        # Experiment command
        experiment_parser = subparsers.add_parser(
            "experiment", help="Manage experiments"
        )
        experiment_subparsers = experiment_parser.add_subparsers(
            dest="experiment_command", help="Experiment commands"
        )

        # experiment start
        exp_start = experiment_subparsers.add_parser(
            "start", help="Start an experiment"
        )
        exp_start.add_argument("name", help="Experiment name")
        exp_start.add_argument("-d", "--description", help="Experiment description")
        exp_start.add_argument(
            "-p", "--param", action="append", help="Parameter (format: key=value)"
        )
        exp_start.add_argument("-t", "--tag", action="append", help="Add tags")

        # experiment log
        exp_log = experiment_subparsers.add_parser("log", help="Log a metric")
        exp_log.add_argument("metric", help="Metric name")
        exp_log.add_argument("value", type=float, help="Metric value")
        exp_log.add_argument("-s", "--step", type=int, help="Training step")

        # experiment end
        exp_end = experiment_subparsers.add_parser("end", help="End current experiment")
        exp_end.add_argument(
            "--status",
            choices=["completed", "failed", "cancelled"],
            default="completed",
            help="Final status",
        )
        exp_end.add_argument("-c", "--commit", help="Associate with commit hash")

        # experiment list
        exp_list = experiment_subparsers.add_parser("list", help="List experiments")
        exp_list.add_argument(
            "--status",
            choices=["pending", "running", "completed", "failed", "cancelled"],
            help="Filter by status",
        )
        exp_list.add_argument("-n", "--number", type=int, default=20, help="Limit")
        exp_list.add_argument("--json", action="store_true", help="Output as JSON")

        # experiment show
        exp_show = experiment_subparsers.add_parser(
            "show", help="Show experiment details"
        )
        exp_show.add_argument("experiment_id", help="Experiment ID")
        exp_show.add_argument("--json", action="store_true", help="Output as JSON")

        # experiment compare
        exp_compare = experiment_subparsers.add_parser(
            "compare", help="Compare experiments"
        )
        exp_compare.add_argument(
            "experiments", nargs="+", help="Experiment IDs to compare"
        )
        exp_compare.add_argument(
            "-m", "--metric", action="append", help="Metrics to compare"
        )
        exp_compare.add_argument("--json", action="store_true", help="Output as JSON")

        # experiment best
        exp_best = experiment_subparsers.add_parser(
            "best", help="Find best experiments by metric"
        )
        exp_best.add_argument("metric", help="Metric name to optimize")
        exp_best.add_argument(
            "--mode",
            choices=["min", "max"],
            default="min",
            help="Optimization mode",
        )
        exp_best.add_argument("-n", "--number", type=int, default=10, help="Limit")
        exp_best.add_argument("--json", action="store_true", help="Output as JSON")

        # experiment delete
        exp_delete = experiment_subparsers.add_parser(
            "delete", help="Delete an experiment"
        )
        exp_delete.add_argument("experiment_id", help="Experiment ID to delete")

        # Publish command
        publish_parser = subparsers.add_parser(
            "publish", help="Publish model to registry"
        )
        publish_subparsers = publish_parser.add_subparsers(
            dest="publish_command", help="Publish commands"
        )

        # publish huggingface
        pub_hf = publish_subparsers.add_parser(
            "huggingface", help="Publish to Hugging Face Hub"
        )
        pub_hf.add_argument("repo_id", help="Repository ID (org/name)")
        pub_hf.add_argument("-c", "--commit", help="Commit reference (default: HEAD)")
        pub_hf.add_argument(
            "--private", action="store_true", help="Create private repo"
        )
        pub_hf.add_argument("-d", "--description", help="Model description")
        pub_hf.add_argument("--base-model", help="Base model this was fine-tuned from")
        pub_hf.add_argument(
            "--metric", action="append", help="Add metric (format: key=value)"
        )
        pub_hf.add_argument("-t", "--tag", action="append", help="Add tags")

        # publish mlflow
        pub_mlflow = publish_subparsers.add_parser(
            "mlflow", help="Publish to MLflow Model Registry"
        )
        pub_mlflow.add_argument("model_name", help="Model name in registry")
        pub_mlflow.add_argument(
            "-c", "--commit", help="Commit reference (default: HEAD)"
        )
        pub_mlflow.add_argument("--tracking-uri", help="MLflow tracking URI")
        pub_mlflow.add_argument("--experiment", help="MLflow experiment name")
        pub_mlflow.add_argument("-d", "--description", help="Model description")
        pub_mlflow.add_argument(
            "--metric", action="append", help="Add metric (format: key=value)"
        )

        # publish local (export)
        pub_local = publish_subparsers.add_parser(
            "local", help="Export to local directory"
        )
        pub_local.add_argument("output_path", help="Output directory")
        pub_local.add_argument(
            "-c", "--commit", help="Commit reference (default: HEAD)"
        )
        pub_local.add_argument(
            "--format",
            choices=["safetensors", "npz", "pt"],
            default="safetensors",
            help="Output format",
        )
        pub_local.add_argument(
            "--no-metadata", action="store_true", help="Skip metadata files"
        )

        # publish history
        publish_subparsers.add_parser("history", help="Show publish history")

        return parser

    def run(self, args=None) -> int:
        """Run the CLI."""
        args = self.parser.parse_args(args)

        if not args.command:
            self.parser.print_help()
            return 0

        # Find repository root
        if args.command != "init":
            repo_path = self._find_repo_root()
            if repo_path is None:
                print("Error: Not in a Coral repository", file=sys.stderr)
                return 1

        # Execute command
        try:
            if args.command == "init":
                return self._cmd_init(args)
            elif args.command == "add":
                return self._cmd_add(args, repo_path)
            elif args.command == "commit":
                return self._cmd_commit(args, repo_path)
            elif args.command == "status":
                return self._cmd_status(args, repo_path)
            elif args.command == "log":
                return self._cmd_log(args, repo_path)
            elif args.command == "checkout":
                return self._cmd_checkout(args, repo_path)
            elif args.command == "branch":
                return self._cmd_branch(args, repo_path)
            elif args.command == "merge":
                return self._cmd_merge(args, repo_path)
            elif args.command == "diff":
                return self._cmd_diff(args, repo_path)
            elif args.command == "tag":
                return self._cmd_tag(args, repo_path)
            elif args.command == "show":
                return self._cmd_show(args, repo_path)
            elif args.command == "gc":
                return self._cmd_gc(args, repo_path)
            elif args.command == "remote":
                return self._cmd_remote(args, repo_path)
            elif args.command == "push":
                return self._cmd_push(args, repo_path)
            elif args.command == "pull":
                return self._cmd_pull(args, repo_path)
            elif args.command == "clone":
                return self._cmd_clone(args)
            elif args.command == "stats":
                return self._cmd_stats(args, repo_path)
            elif args.command == "compare":
                return self._cmd_compare(args, repo_path)
            elif args.command == "sync":
                return self._cmd_sync(args, repo_path)
            elif args.command == "sync-status":
                return self._cmd_sync_status(args, repo_path)
            elif args.command == "experiment":
                return self._cmd_experiment(args, repo_path)
            elif args.command == "publish":
                return self._cmd_publish(args, repo_path)
            else:
                print(f"Error: Unknown command '{args.command}'", file=sys.stderr)
                return 1
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    def _find_repo_root(self) -> Optional[Path]:
        """Find the repository root by looking for .coral directory."""
        current = Path.cwd()

        while current != current.parent:
            if (current / ".coral").exists():
                return current
            current = current.parent

        return None

    def _cmd_init(self, args) -> int:
        """Initialize a new repository."""
        path = Path(args.path).resolve()

        try:
            Repository(path, init=True)
            print(f"Initialized empty Coral repository in {path / '.coral'}")
            return 0
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    def _cmd_add(self, args, repo_path: Path) -> int:
        """Add weights to staging."""
        repo = Repository(repo_path)

        weights = {}
        for weight_path in args.weights:
            # Load weight file (assuming numpy format for now)
            path = Path(weight_path)
            if not path.exists():
                print(f"Error: File not found: {weight_path}", file=sys.stderr)
                return 1

            # Load based on file extension
            if path.suffix == ".npy":
                data = np.load(path)
                name = path.stem
            elif path.suffix == ".npz":
                # Load compressed numpy archive
                archive = np.load(path)
                for name, data in archive.items():
                    weight = WeightTensor(
                        data=data,
                        metadata=WeightMetadata(
                            name=name, shape=data.shape, dtype=data.dtype
                        ),
                    )
                    weights[name] = weight
                continue
            else:
                print(f"Error: Unsupported file format: {path.suffix}", file=sys.stderr)
                return 1

            # Create weight tensor
            weight = WeightTensor(
                data=data,
                metadata=WeightMetadata(name=name, shape=data.shape, dtype=data.dtype),
            )
            weights[name] = weight

        # Stage weights
        staged = repo.stage_weights(weights)
        print(f"Staged {len(staged)} weight(s)")

        return 0

    def _cmd_commit(self, args, repo_path: Path) -> int:
        """Commit staged weights."""
        repo = Repository(repo_path)

        commit = repo.commit(
            message=args.message,
            author=args.author,
            email=args.email,
            tags=args.tag or [],
        )

        print(
            f"[{repo.branch_manager.get_current_branch()} "
            f"{commit.commit_hash[:8]}] {commit.metadata.message}"
        )
        print(f" {len(commit.weight_hashes)} weight(s) changed")

        return 0

    def _cmd_status(self, args, repo_path: Path) -> int:
        """Show repository status."""
        repo = Repository(repo_path)

        # Current branch
        current_branch = repo.branch_manager.get_current_branch()
        print(f"On branch {current_branch}")

        # Check for staged files
        staged_file = repo.staging_dir / "staged.json"
        if staged_file.exists():
            with open(staged_file) as f:
                staged_data = json.load(f)

            # Handle both old flat format and new nested format
            if isinstance(staged_data, dict) and "weights" in staged_data:
                staged = staged_data["weights"]
            else:
                staged = staged_data

            if staged:
                print("\nChanges to be committed:")
                for name, hash_val in staged.items():
                    print(f"  new weight: {name} ({hash_val[:8]})")
        else:
            print("\nNothing staged for commit")

        return 0

    def _cmd_log(self, args, repo_path: Path) -> int:
        """Show commit history."""
        repo = Repository(repo_path)

        commits = repo.log(max_commits=args.number)

        if not commits:
            print("No commits yet")
            return 0

        for commit in commits:
            if args.oneline:
                print(f"{commit.commit_hash[:8]} {commit.metadata.message}")
            else:
                print(f"commit {commit.commit_hash}")
                print(f"Author: {commit.metadata.author} <{commit.metadata.email}>")
                print(f"Date:   {commit.metadata.timestamp}")
                if commit.metadata.tags:
                    print(f"Tags:   {', '.join(commit.metadata.tags)}")
                print(f"\n    {commit.metadata.message}\n")

        return 0

    def _cmd_checkout(self, args, repo_path: Path) -> int:
        """Checkout branch or commit."""
        repo = Repository(repo_path)

        repo.checkout(args.target)
        print(f"Switched to '{args.target}'")

        return 0

    def _cmd_branch(self, args, repo_path: Path) -> int:
        """Manage branches."""
        repo = Repository(repo_path)

        if args.list or (not args.name and not args.delete):
            # List branches
            branches = repo.branch_manager.list_branches()
            current = repo.branch_manager.get_current_branch()

            for branch in branches:
                if branch.name == current:
                    print(f"* {branch.name}")
                else:
                    print(f"  {branch.name}")
        elif args.delete:
            # Delete branch
            repo.branch_manager.delete_branch(args.delete)
            print(f"Deleted branch {args.delete}")
        elif args.name:
            # Create branch
            repo.create_branch(args.name)
            print(f"Created branch {args.name}")

        return 0

    def _cmd_merge(self, args, repo_path: Path) -> int:
        """Merge branches."""
        repo = Repository(repo_path)

        commit = repo.merge(args.branch, message=args.message)
        current = repo.branch_manager.get_current_branch()

        print(f"Merged {args.branch} into {current}")
        print(f"[{current} {commit.commit_hash[:8]}] {commit.metadata.message}")

        return 0

    def _cmd_diff(self, args, repo_path: Path) -> int:
        """Show differences between commits."""
        repo = Repository(repo_path)

        diff = repo.diff(args.from_ref, args.to_ref)

        # Print summary
        print(f"Comparing {args.from_ref} -> {args.to_ref or 'HEAD'}")
        print(f"  Added:    {diff['summary']['total_added']} weight(s)")
        print(f"  Removed:  {diff['summary']['total_removed']} weight(s)")
        print(f"  Modified: {diff['summary']['total_modified']} weight(s)")

        # Show details
        if diff["added"]:
            print("\nAdded weights:")
            for name in diff["added"]:
                print(f"  + {name}")

        if diff["removed"]:
            print("\nRemoved weights:")
            for name in diff["removed"]:
                print(f"  - {name}")

        if diff["modified"]:
            print("\nModified weights:")
            for name, info in diff["modified"].items():
                print(f"  ~ {name}")
                print(f"    {info['from_hash'][:8]} -> {info['to_hash'][:8]}")

        return 0

    def _cmd_tag(self, args, repo_path: Path) -> int:
        """Tag a version."""
        repo = Repository(repo_path)

        # Parse metrics
        metrics = {}
        if args.metric:
            for metric in args.metric:
                key, value = metric.split("=", 1)
                metrics[key] = float(value)

        version = repo.tag_version(
            name=args.name,
            description=args.description,
            metrics=metrics if metrics else None,
            commit_ref=args.commit,
        )

        print(f"Tagged version '{version.name}' ({version.version_id})")

        return 0

    def _cmd_show(self, args, repo_path: Path) -> int:
        """Show weight information."""
        repo = Repository(repo_path)

        weight = repo.get_weight(args.weight, commit_ref=args.commit)

        if weight is None:
            print(f"Error: Weight '{args.weight}' not found", file=sys.stderr)
            return 1

        print(f"Weight: {weight.metadata.name}")
        print(f"Shape: {weight.shape}")
        print(f"Dtype: {weight.dtype}")
        print(f"Size: {weight.nbytes:,} bytes")
        print(f"Hash: {weight.compute_hash()}")

        if weight.metadata.layer_type:
            print(f"Layer type: {weight.metadata.layer_type}")
        if weight.metadata.model_name:
            print(f"Model: {weight.metadata.model_name}")

        # Show statistics
        print("\nStatistics:")
        print(f"  Min: {weight.data.min():.6f}")
        print(f"  Max: {weight.data.max():.6f}")
        print(f"  Mean: {weight.data.mean():.6f}")
        print(f"  Std: {weight.data.std():.6f}")

        return 0

    def _cmd_gc(self, args, repo_path: Path) -> int:
        """Garbage collect unreferenced weights."""
        repo = Repository(repo_path)

        result = repo.gc()

        print("Garbage collection complete:")
        print(f"  Cleaned: {result['cleaned_weights']} weight(s)")
        print(f"  Remaining: {result['remaining_weights']} weight(s)")

        return 0

    def _cmd_remote(self, args, repo_path: Path) -> int:
        """Manage remotes."""
        repo = Repository(repo_path)

        if not args.remote_command or args.remote_command == "list":
            # List remotes
            remotes = repo.list_remotes()
            if not remotes:
                print("No remotes configured")
            else:
                for name, config in remotes.items():
                    print(f"{name}\t{config.get('url', 'N/A')}")
            return 0

        elif args.remote_command == "add":
            repo.add_remote(args.name, args.url)
            print(f"Added remote '{args.name}' -> {args.url}")
            return 0

        elif args.remote_command == "remove":
            repo.remove_remote(args.name)
            print(f"Removed remote '{args.name}'")
            return 0

        elif args.remote_command == "show":
            remote = repo.get_remote(args.name)
            if not remote:
                print(f"Error: Remote '{args.name}' not found", file=sys.stderr)
                return 1
            print(f"Remote: {args.name}")
            print(f"  URL: {remote.get('url', 'N/A')}")
            print(f"  Backend: {remote.get('backend', 'N/A')}")
            if remote.get("endpoint_url"):
                print(f"  Endpoint: {remote['endpoint_url']}")
            return 0

        return 0

    def _cmd_push(self, args, repo_path: Path) -> int:
        """Push weights to remote."""
        repo = Repository(repo_path)

        remote_name = args.remote
        remote = repo.get_remote(remote_name)

        if not remote:
            print(f"Error: Remote '{remote_name}' not found", file=sys.stderr)
            print("Use 'coral remote add <name> <url>' to add a remote")
            return 1

        print(f"Pushing to {remote_name} ({remote['url']})...")

        pbar = None
        try:
            # Create progress bar
            bytes_so_far = 0

            def progress_callback(current: int, total: int, bytes_: int, hash_key: str):
                nonlocal pbar, bytes_so_far
                if pbar is None:
                    pbar = tqdm(
                        total=total,
                        desc="Pushing",
                        unit="weights",
                        bar_format=(
                            "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                        ),
                    )
                pbar.update(1)
                bytes_so_far += bytes_
                pbar.set_postfix(
                    {"bytes": self._format_bytes(bytes_so_far)}, refresh=True
                )

            result = repo.push(
                remote_name, force=args.force, progress_callback=progress_callback
            )

            if pbar:
                pbar.close()

            print("\nPush complete:")
            print(f"  Weights pushed: {result.get('weights_pushed', 0)}")
            transferred = result.get("bytes_transferred", 0)
            print(f"  Bytes transferred: {self._format_bytes(transferred)}")
            if result.get("skipped", 0) > 0:
                print(f"  Skipped (already exists): {result['skipped']}")

            return 0
        except Exception as e:
            if pbar:
                pbar.close()
            print(f"Error: Push failed: {e}", file=sys.stderr)
            return 1

    def _cmd_pull(self, args, repo_path: Path) -> int:
        """Pull weights from remote."""
        repo = Repository(repo_path)

        remote_name = args.remote
        remote = repo.get_remote(remote_name)

        if not remote:
            print(f"Error: Remote '{remote_name}' not found", file=sys.stderr)
            print("Use 'coral remote add <name> <url>' to add a remote")
            return 1

        print(f"Pulling from {remote_name} ({remote['url']})...")

        pbar = None
        try:
            # Create progress bar
            bytes_so_far = 0

            def progress_callback(current: int, total: int, bytes_: int, hash_key: str):
                nonlocal pbar, bytes_so_far
                if pbar is None:
                    pbar = tqdm(
                        total=total,
                        desc="Pulling",
                        unit="weights",
                        bar_format=(
                            "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                        ),
                    )
                pbar.update(1)
                bytes_so_far += bytes_
                pbar.set_postfix(
                    {"bytes": self._format_bytes(bytes_so_far)}, refresh=True
                )

            result = repo.pull(
                remote_name, force=args.force, progress_callback=progress_callback
            )

            if pbar:
                pbar.close()

            print("\nPull complete:")
            print(f"  Weights pulled: {result.get('weights_pulled', 0)}")
            transferred = result.get("bytes_transferred", 0)
            print(f"  Bytes transferred: {self._format_bytes(transferred)}")
            if result.get("skipped", 0) > 0:
                print(f"  Skipped (already exists): {result['skipped']}")

            return 0
        except Exception as e:
            if pbar:
                pbar.close()
            print(f"Error: Pull failed: {e}", file=sys.stderr)
            return 1

    def _cmd_clone(self, args) -> int:
        """Clone a remote repository."""

        url = args.url
        path = Path(args.path).resolve()

        # Determine repository name from URL if path is current dir
        if args.path == ".":
            # Extract name from URL
            if url.startswith("s3://"):
                name = url.rstrip("/").split("/")[-1]
            elif url.startswith("file://"):
                name = Path(url[7:]).name
            else:
                name = "coral-repo"
            path = Path.cwd() / name

        print(f"Cloning {url} into {path}...")

        try:
            # Initialize new repository
            repo = Repository(path, init=True)

            # Add origin remote
            repo.add_remote("origin", url)

            # Pull all weights
            result = repo.pull("origin")

            print("Clone complete:")
            print(f"  Weights: {result.get('weights_pulled', 0)}")
            print(f"  Bytes: {result.get('bytes_transferred', 0):,}")

            return 0
        except Exception as e:
            print(f"Error: Clone failed: {e}", file=sys.stderr)
            return 1

    def _cmd_stats(self, args, repo_path: Path) -> int:
        """Show repository statistics and storage savings."""
        repo = Repository(repo_path)

        # Collect statistics
        stats = self._calculate_repo_stats(repo)

        if args.json:
            print(json.dumps(stats, indent=2))
            return 0

        # Human-readable output
        print("=" * 60)
        print("Coral Repository Statistics")
        print("=" * 60)
        print()

        # Repository info
        print(f"Repository: {repo_path}")
        print(f"Branch: {repo.branch_manager.get_current_branch()}")
        print(f"Commits: {stats['total_commits']}")
        print()

        # Weight statistics
        print("Weight Storage:")
        print(f"  Total weights stored: {stats['total_weights']}")
        print(
            f"  Unique weights: {stats['unique_weights']} ({stats['unique_pct']:.1f}%)"
        )
        print(f"  Delta-encoded: {stats['delta_weights']} ({stats['delta_pct']:.1f}%)")
        print(f"  Duplicates eliminated: {stats['duplicate_weights']}")
        print()

        # Storage savings
        print("Storage Savings:")
        print(f"  Raw size (uncompressed): {self._format_bytes(stats['raw_size'])}")
        print(f"  Actual size on disk: {self._format_bytes(stats['actual_size'])}")
        print(f"  Space saved: {self._format_bytes(stats['bytes_saved'])}")
        print(f"  Compression ratio: {stats['compression_ratio']:.1f}x")
        print(f"  Savings: {stats['savings_pct']:.1f}%")
        print()

        # Comparison with alternatives
        print("Comparison with alternatives:")
        print(f"  If using git-lfs: ~{self._format_bytes(stats['raw_size'])}")
        print(f"  If using naive storage: ~{self._format_bytes(stats['raw_size'])}")
        print(f"  Coral saves you: {self._format_bytes(stats['bytes_saved'])}")
        print()

        # Delta encoding breakdown
        if stats["delta_weights"] > 0:
            print("Delta Encoding Breakdown:")
            print(f"  Average delta size: {stats['avg_delta_ratio']:.1f}% of original")
            print(f"  Best compression: {stats['best_delta_ratio']:.1f}%")
            print()

        return 0

    def _calculate_repo_stats(self, repo: Repository) -> dict:
        """Calculate repository statistics."""
        from coral.storage.hdf5_store import HDF5Store

        total_commits = len(repo.version_graph.commits)

        # Count weights
        all_weight_hashes = set()
        total_weight_refs = 0

        for commit in repo.version_graph.commits.values():
            total_weight_refs += len(commit.weight_hashes)
            all_weight_hashes.update(commit.weight_hashes.values())

        unique_weights = len(all_weight_hashes)
        duplicate_weights = total_weight_refs - unique_weights

        # Calculate storage sizes
        raw_size = 0
        actual_size = 0
        delta_count = 0
        delta_ratios = []

        with HDF5Store(repo.weights_store_path) as store:
            for hash_key in all_weight_hashes:
                weight = store.load(hash_key)
                if weight:
                    raw_size += weight.nbytes

            # Get actual file size
            if repo.weights_store_path.exists():
                actual_size = repo.weights_store_path.stat().st_size

            # Count deltas
            delta_hashes = set()
            for commit in repo.version_graph.commits.values():
                if hasattr(commit, "delta_weights") and commit.delta_weights:
                    delta_hashes.update(commit.delta_weights.values())

            delta_count = len(delta_hashes)

            # Calculate delta compression ratios
            for delta_hash in delta_hashes:
                delta = store.load_delta(delta_hash)
                if delta and hasattr(delta, "compression_ratio"):
                    delta_ratios.append(delta.compression_ratio * 100)

        bytes_saved = raw_size - actual_size
        compression_ratio = raw_size / actual_size if actual_size > 0 else 1.0
        savings_pct = (bytes_saved / raw_size * 100) if raw_size > 0 else 0

        unique_pct = (
            (unique_weights / total_weight_refs * 100) if total_weight_refs > 0 else 0
        )
        delta_pct = (delta_count / unique_weights * 100) if unique_weights > 0 else 0

        return {
            "total_commits": total_commits,
            "total_weights": total_weight_refs,
            "unique_weights": unique_weights,
            "duplicate_weights": duplicate_weights,
            "delta_weights": delta_count,
            "unique_pct": unique_pct,
            "delta_pct": delta_pct,
            "raw_size": raw_size,
            "actual_size": actual_size,
            "bytes_saved": bytes_saved,
            "compression_ratio": compression_ratio,
            "savings_pct": savings_pct,
            "avg_delta_ratio": sum(delta_ratios) / len(delta_ratios)
            if delta_ratios
            else 0,
            "best_delta_ratio": min(delta_ratios) if delta_ratios else 0,
        }

    def _format_bytes(self, size: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"

    def _cmd_compare(self, args, repo_path: Path) -> int:
        """Compare weights between two commits."""
        from coral.utils.visualization import compare_models, format_model_diff

        repo = Repository(repo_path)

        ref1 = args.ref1
        ref2 = args.ref2

        # Resolve references
        commit1 = self._resolve_ref(repo, ref1)
        if not commit1:
            print(f"Error: Could not resolve reference '{ref1}'", file=sys.stderr)
            return 1

        if ref2:
            commit2 = self._resolve_ref(repo, ref2)
            if not commit2:
                print(f"Error: Could not resolve reference '{ref2}'", file=sys.stderr)
                return 1
        else:
            # Default to HEAD
            current_branch = repo.branch_manager.get_current_branch()
            head_hash = repo.branch_manager.get_branch_commit(current_branch)
            commit2 = repo.version_graph.get_commit(head_hash)
            ref2 = "HEAD"

        # Load weights from both commits
        weights1 = repo.get_all_weights(commit1.commit_hash)
        weights2 = repo.get_all_weights(commit2.commit_hash)

        if not weights1:
            print(f"Error: No weights found in '{ref1}'", file=sys.stderr)
            return 1
        if not weights2:
            print(f"Error: No weights found in '{ref2}'", file=sys.stderr)
            return 1

        # Compare models
        diff = compare_models(weights1, weights2, ref1, ref2)

        # Format and print
        print(format_model_diff(diff, verbose=args.verbose))

        return 0

    def _cmd_sync(self, args, repo_path: Path) -> int:
        """Bidirectional sync with remote."""
        repo = Repository(repo_path)

        remote_name = args.remote
        remote = repo.get_remote(remote_name)

        if not remote:
            print(f"Error: Remote '{remote_name}' not found", file=sys.stderr)
            print("Use 'coral remote add <name> <url>' to add a remote")
            return 1

        print(f"Syncing with {remote_name} ({remote['url']})...")

        pbar = None
        try:
            bytes_so_far = 0

            def progress_callback(current: int, total: int, bytes_: int, hash_key: str):
                nonlocal pbar, bytes_so_far
                if pbar is None:
                    pbar = tqdm(
                        total=total,
                        desc="Syncing",
                        unit="weights",
                        bar_format=(
                            "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                        ),
                    )
                pbar.update(1)
                bytes_so_far += bytes_
                pbar.set_postfix(
                    {"bytes": self._format_bytes(bytes_so_far)}, refresh=True
                )

            result = repo.sync(remote_name, progress_callback=progress_callback)

            if pbar:
                pbar.close()

            print("\nSync complete:")
            print(f"  Weights pushed: {result['total_pushed']}")
            print(f"  Weights pulled: {result['total_pulled']}")
            transferred = result["bytes_transferred"]
            print(f"  Bytes transferred: {self._format_bytes(transferred)}")

            return 0
        except Exception as e:
            if pbar:
                pbar.close()
            print(f"Error: Sync failed: {e}", file=sys.stderr)
            return 1

    def _cmd_sync_status(self, args, repo_path: Path) -> int:
        """Show sync status with remote."""
        repo = Repository(repo_path)

        remote_name = args.remote
        remote = repo.get_remote(remote_name)

        if not remote:
            print(f"Error: Remote '{remote_name}' not found", file=sys.stderr)
            return 1

        try:
            status = repo.get_sync_status(remote_name)

            if args.json:
                print(json.dumps(status, indent=2))
                return 0

            print(f"Sync status with {remote_name}:")
            print(f"  Local weights:  {status['total_local']}")
            print(f"  Remote weights: {status['total_remote']}")
            print()

            if status["is_synced"]:
                print("  Status: ✓ Fully synced")
            else:
                if status["needs_push"] > 0:
                    print(f"  Ahead by: {status['needs_push']} weight(s) (need push)")
                if status["needs_pull"] > 0:
                    print(f"  Behind by: {status['needs_pull']} weight(s) (need pull)")
                print()
                print("  Run 'coral sync' to synchronize")

            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    def _cmd_experiment(self, args, repo_path: Path) -> int:
        """Manage experiments."""
        from coral.experiments import ExperimentStatus, ExperimentTracker

        repo = Repository(repo_path)
        tracker = ExperimentTracker(repo)

        if not args.experiment_command:
            # Show experiment summary
            summary = tracker.get_summary()
            print("Experiment Summary:")
            print(f"  Total experiments: {summary['total_experiments']}")
            for status, count in summary["by_status"].items():
                if count > 0:
                    print(f"  {status}: {count}")
            return 0

        if args.experiment_command == "start":
            # Parse params
            params = {}
            if args.param:
                for p in args.param:
                    key, value = p.split("=", 1)
                    try:
                        params[key] = float(value)
                    except ValueError:
                        params[key] = value

            exp = tracker.start(
                name=args.name,
                description=args.description,
                params=params,
                tags=args.tag or [],
            )
            print(f"Started experiment: {exp.name}")
            print(f"  ID: {exp.experiment_id}")
            return 0

        elif args.experiment_command == "log":
            tracker.log(args.metric, args.value, step=args.step)
            print(f"Logged {args.metric}={args.value}")
            if args.step:
                print(f"  Step: {args.step}")
            return 0

        elif args.experiment_command == "end":
            status_map = {
                "completed": ExperimentStatus.COMPLETED,
                "failed": ExperimentStatus.FAILED,
                "cancelled": ExperimentStatus.CANCELLED,
            }
            exp = tracker.end(
                status=status_map[args.status],
                commit_hash=args.commit,
            )
            print(f"Ended experiment: {exp.name}")
            print(f"  Status: {exp.status.value}")
            if exp.duration:
                print(f"  Duration: {exp.duration:.1f}s")
            return 0

        elif args.experiment_command == "list":
            status = ExperimentStatus(args.status) if args.status else None
            experiments = tracker.list(status=status, limit=args.number)

            if args.json:
                print(json.dumps([e.to_dict() for e in experiments], indent=2))
                return 0

            if not experiments:
                print("No experiments found")
                return 0

            print(f"{'ID':<18} {'Name':<25} {'Status':<12} {'Created'}")
            print("-" * 75)
            for exp in experiments:
                print(
                    f"{exp.experiment_id:<18} "
                    f"{exp.name[:24]:<25} "
                    f"{exp.status.value:<12} "
                    f"{exp.created_at.strftime('%Y-%m-%d %H:%M')}"
                )
            return 0

        elif args.experiment_command == "show":
            exp = tracker.get(args.experiment_id)
            if not exp:
                print(f"Experiment not found: {args.experiment_id}", file=sys.stderr)
                return 1

            if args.json:
                print(json.dumps(exp.to_dict(), indent=2))
                return 0

            print(f"Experiment: {exp.name}")
            print(f"  ID: {exp.experiment_id}")
            print(f"  Status: {exp.status.value}")
            print(f"  Created: {exp.created_at}")
            if exp.started_at:
                print(f"  Started: {exp.started_at}")
            if exp.ended_at:
                print(f"  Ended: {exp.ended_at}")
            if exp.duration:
                print(f"  Duration: {exp.duration:.1f}s")
            if exp.branch:
                print(f"  Branch: {exp.branch}")
            if exp.commit_hash:
                print(f"  Commit: {exp.commit_hash}")
            if exp.description:
                print(f"\nDescription: {exp.description}")
            if exp.params:
                print("\nParameters:")
                for k, v in exp.params.items():
                    print(f"  {k}: {v}")
            if exp.tags:
                print(f"\nTags: {', '.join(exp.tags)}")
            if exp.metrics:
                print("\nMetrics:")
                for metric_name in exp.metric_names:
                    latest = exp.get_latest_metric(metric_name)
                    best = exp.get_best_metric(metric_name)
                    print(f"  {metric_name}:")
                    print(f"    Latest: {latest.value:.4f}")
                    print(f"    Best: {best.value:.4f}")
            return 0

        elif args.experiment_command == "compare":
            comparison = tracker.compare(
                args.experiments,
                metrics=args.metric,
            )

            if args.json:
                print(json.dumps(comparison, indent=2))
                return 0

            if not comparison["experiments"]:
                print("No experiments found")
                return 1

            # Print comparison table
            print("Experiment Comparison")
            print("=" * 60)

            # Header
            exp_ids = [e["id"][:12] for e in comparison["experiments"]]
            print(f"{'Metric':<20}", end="")
            for exp_id in exp_ids:
                print(f"{exp_id:<15}", end="")
            print()
            print("-" * 60)

            # Metrics
            for metric, values in comparison["metrics"].items():
                print(f"{metric:<20}", end="")
                for exp in comparison["experiments"]:
                    val = values.get(exp["id"], {}).get("best")
                    if val is not None:
                        print(f"{val:<15.4f}", end="")
                    else:
                        print(f"{'N/A':<15}", end="")
                print()
            return 0

        elif args.experiment_command == "best":
            results = tracker.find_best(
                metric=args.metric,
                mode=args.mode,
                limit=args.number,
            )

            if args.json:
                print(json.dumps(results, indent=2))
                return 0

            if not results:
                print(f"No experiments found with metric '{args.metric}'")
                return 0

            print(f"Best Experiments by {args.metric} ({args.mode})")
            print("=" * 60)
            print(f"{'Rank':<6} {'ID':<14} {'Name':<20} {'Value'}")
            print("-" * 60)
            for i, r in enumerate(results, 1):
                print(
                    f"{i:<6} "
                    f"{r['experiment_id'][:12]:<14} "
                    f"{r['name'][:18]:<20} "
                    f"{r['best_value']:.4f}"
                )
            return 0

        elif args.experiment_command == "delete":
            if tracker.delete(args.experiment_id):
                print(f"Deleted experiment: {args.experiment_id}")
                return 0
            else:
                print(f"Experiment not found: {args.experiment_id}", file=sys.stderr)
                return 1

        return 0

    def _cmd_publish(self, args, repo_path: Path) -> int:
        """Publish model to registry."""
        from coral.registry import ModelPublisher

        repo = Repository(repo_path)
        publisher = ModelPublisher(repo)

        if not args.publish_command:
            # Show publish history summary
            history = publisher.get_history(limit=10)
            if not history:
                print("No publish history")
                return 0

            print("Recent Publishes:")
            print(f"{'Registry':<15} {'Model':<30} {'Status':<10} {'Date'}")
            print("-" * 70)
            for r in history:
                status = "✓" if r.success else "✗"
                print(
                    f"{r.registry.value:<15} "
                    f"{r.model_name[:28]:<30} "
                    f"{status:<10} "
                    f"{r.published_at.strftime('%Y-%m-%d %H:%M')}"
                )
            return 0

        if args.publish_command == "huggingface":
            # Parse metrics
            metrics = {}
            if args.metric:
                for m in args.metric:
                    key, value = m.split("=", 1)
                    metrics[key] = float(value)

            print(f"Publishing to Hugging Face Hub: {args.repo_id}...")
            result = publisher.publish_huggingface(
                repo_id=args.repo_id,
                commit_ref=args.commit,
                private=args.private,
                description=args.description,
                base_model=args.base_model,
                metrics=metrics if metrics else None,
                tags=args.tag,
            )

            if result.success:
                print("✓ Published successfully!")
                print(f"  URL: {result.url}")
                return 0
            else:
                print(f"✗ Publish failed: {result.error}", file=sys.stderr)
                return 1

        elif args.publish_command == "mlflow":
            # Parse metrics
            metrics = {}
            if args.metric:
                for m in args.metric:
                    key, value = m.split("=", 1)
                    metrics[key] = float(value)

            print(f"Publishing to MLflow: {args.model_name}...")
            result = publisher.publish_mlflow(
                model_name=args.model_name,
                commit_ref=args.commit,
                tracking_uri=args.tracking_uri,
                experiment_name=args.experiment,
                description=args.description,
                metrics=metrics if metrics else None,
            )

            if result.success:
                print("✓ Published successfully!")
                print(f"  Version: {result.version}")
                if result.url:
                    print(f"  URL: {result.url}")
                return 0
            else:
                print(f"✗ Publish failed: {result.error}", file=sys.stderr)
                return 1

        elif args.publish_command == "local":
            print(f"Exporting to: {args.output_path}...")
            result = publisher.publish_local(
                output_path=args.output_path,
                commit_ref=args.commit,
                format=args.format,
                include_metadata=not args.no_metadata,
            )

            if result.success:
                print("✓ Exported successfully!")
                print(f"  Path: {args.output_path}")
                print(f"  Format: {args.format}")
                return 0
            else:
                print(f"✗ Export failed: {result.error}", file=sys.stderr)
                return 1

        elif args.publish_command == "history":
            history = publisher.get_history(limit=50)
            if not history:
                print("No publish history")
                return 0

            print("Publish History")
            print("=" * 80)
            for r in history:
                status = "Success" if r.success else "Failed"
                print(f"\n{r.registry.value}: {r.model_name}")
                print(f"  Status: {status}")
                print(f"  Date: {r.published_at}")
                if r.version:
                    print(f"  Version: {r.version}")
                if r.url:
                    print(f"  URL: {r.url}")
                if r.error:
                    print(f"  Error: {r.error}")
            return 0

        return 0

    def _resolve_ref(self, repo: Repository, ref: str):
        """Resolve a reference to a commit."""
        # Try as commit hash
        commit = repo.version_graph.get_commit(ref)
        if commit:
            return commit

        # Try as branch name
        try:
            branch_hash = repo.branch_manager.get_branch_commit(ref)
            if branch_hash:
                return repo.version_graph.get_commit(branch_hash)
        except Exception:
            pass

        # Try as tag/version name
        for version in repo.version_graph.versions.values():
            if version.name == ref:
                return repo.version_graph.get_commit(version.commit_hash)

        # Try as partial commit hash
        for hash_key in repo.version_graph.commits:
            if hash_key.startswith(ref):
                return repo.version_graph.get_commit(hash_key)

        return None


def main():
    """Main entry point."""
    cli = CoralCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()
