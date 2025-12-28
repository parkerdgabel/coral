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

        # Config command
        config_parser = subparsers.add_parser("config", help="Manage configuration")
        config_subparsers = config_parser.add_subparsers(
            dest="config_command", help="Configuration commands"
        )

        # config list
        config_list = config_subparsers.add_parser(
            "list", help="List all configuration options"
        )
        config_list.add_argument("--json", action="store_true", help="Output as JSON")

        # config get
        config_get = config_subparsers.add_parser(
            "get", help="Get a configuration value"
        )
        config_get.add_argument(
            "key", help="Configuration key (e.g., core.similarity_threshold)"
        )

        # config set
        config_set = config_subparsers.add_parser(
            "set", help="Set a configuration value"
        )
        config_set.add_argument(
            "--global",
            dest="global_config",
            action="store_true",
            help="Set in global user config",
        )
        config_set.add_argument(
            "key", help="Configuration key (e.g., core.similarity_threshold)"
        )
        config_set.add_argument("value", help="Value to set")

        # config show
        config_show = config_subparsers.add_parser(
            "show", help="Show effective configuration (merged from all sources)"
        )
        config_show.add_argument("--json", action="store_true", help="Output as JSON")

        # config validate
        config_subparsers.add_parser("validate", help="Validate configuration")

        # config migrate
        config_subparsers.add_parser(
            "migrate", help="Migrate from legacy config.json to coral.toml"
        )

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
            elif args.command == "config":
                return self._cmd_config(args, repo_path)
            elif args.command == "compare":
                return self._cmd_compare(args, repo_path)
            elif args.command == "sync":
                return self._cmd_sync(args, repo_path)
            elif args.command == "sync-status":
                return self._cmd_sync_status(args, repo_path)
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
                data = np.load(path, allow_pickle=False)
                name = path.stem
            elif path.suffix == ".npz":
                # Load compressed numpy archive
                archive = np.load(path, allow_pickle=False)
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

    def _cmd_config(self, args, repo_path: Path) -> int:
        """Manage configuration."""
        from coral.config import (
            ConfigLoader,
            load_config,
            validate_config,
        )

        if not args.config_command or args.config_command == "show":
            # Show effective configuration
            config = load_config(repo_path=repo_path)

            if args.config_command == "show" and hasattr(args, "json") and args.json:
                print(json.dumps(config.to_dict(), indent=2))
                return 0

            self._print_config(config)
            return 0

        elif args.config_command == "list":
            # List all available configuration options
            config = load_config(repo_path=repo_path)

            if args.json:
                print(json.dumps(config.to_dict(), indent=2))
                return 0

            self._print_config_list(config)
            return 0

        elif args.config_command == "get":
            # Get a specific configuration value
            config = load_config(repo_path=repo_path)
            value = config.get_nested(args.key)

            if value is None:
                print(f"Error: Unknown configuration key: {args.key}", file=sys.stderr)
                return 1

            # Handle object values
            if hasattr(value, "to_dict"):
                print(json.dumps(value.to_dict(), indent=2))
            else:
                print(value)
            return 0

        elif args.config_command == "set":
            # Set a configuration value
            config = load_config(repo_path=repo_path)

            # Parse value (convert to appropriate type)
            value = self._parse_config_value(args.value)

            try:
                config.set_nested(args.key, value)
            except KeyError:
                print(f"Error: Unknown configuration key: {args.key}", file=sys.stderr)
                return 1

            # Save configuration
            loader = ConfigLoader(repo_path=repo_path)
            if args.global_config:
                loader.save_user_config(config)
                print(f"Set {args.key} = {value} (global)")
            else:
                loader.save_repo_config(config)
                print(f"Set {args.key} = {value}")
            return 0

        elif args.config_command == "validate":
            # Validate configuration
            config = load_config(repo_path=repo_path)
            result = validate_config(config)

            if result.valid:
                print("Configuration is valid")
                if result.warnings:
                    print("\nWarnings:")
                    for warning in result.warnings:
                        print(f"  - {warning}")
                return 0
            else:
                print("Configuration has errors:")
                for error in result.errors:
                    print(f"  - {error}")
                if result.warnings:
                    print("\nWarnings:")
                    for warning in result.warnings:
                        print(f"  - {warning}")
                return 1

        elif args.config_command == "migrate":
            # Migrate from legacy config.json to coral.toml
            legacy_path = repo_path / ".coral" / "config.json"
            toml_path = repo_path / ".coral" / "coral.toml"

            if not legacy_path.exists():
                print("No legacy config.json found")
                return 0

            if toml_path.exists():
                print("coral.toml already exists. Delete it first to re-migrate.")
                return 1

            # Load legacy config and save as TOML
            config = load_config(repo_path=repo_path)
            loader = ConfigLoader(repo_path=repo_path)
            loader.save_repo_config(config)

            print(f"Migrated configuration to {toml_path}")
            print("You can now delete config.json if you no longer need it.")
            return 0

        return 0

    def _print_config(self, config) -> None:
        """Print configuration in human-readable format."""
        print("Coral Configuration")
        print("=" * 50)
        print()

        # User
        print("[user]")
        print(f"  name = {config.user.name!r}")
        print(f"  email = {config.user.email!r}")
        print()

        # Core
        print("[core]")
        print(f"  compression = {config.core.compression!r}")
        print(f"  compression_level = {config.core.compression_level}")
        print(f"  similarity_threshold = {config.core.similarity_threshold}")
        print(f"  magnitude_tolerance = {config.core.magnitude_tolerance}")
        print(f"  enable_lsh = {str(config.core.enable_lsh).lower()}")
        print(f"  delta_encoding = {str(config.core.delta_encoding).lower()}")
        print(f"  delta_type = {config.core.delta_type!r}")
        strict_recon = str(config.core.strict_reconstruction).lower()
        print(f"  strict_reconstruction = {strict_recon}")
        print()

        # Delta
        print("[delta]")
        print(f"  sparse_threshold = {config.delta.sparse_threshold}")
        print(f"  quantization_bits = {config.delta.quantization_bits}")
        print(f"  min_weight_size = {config.delta.min_weight_size}")
        print(f"  max_delta_ratio = {config.delta.max_delta_ratio}")
        print()

        # Storage
        print("[storage]")
        print(f"  compression = {config.storage.compression!r}")
        print(f"  compression_level = {config.storage.compression_level}")
        print()

        # Logging
        print("[logging]")
        print(f"  level = {config.logging.level!r}")
        print()

    def _print_config_list(self, config) -> None:
        """Print available configuration options."""
        print("Available Configuration Options")
        print("=" * 60)
        print()
        print("Use 'coral config get <key>' to view a value")
        print("Use 'coral config set <key> <value>' to set a value")
        print()

        options = [
            ("user.name", "string", "User name for commits"),
            ("user.email", "string", "User email for commits"),
            ("core.compression", "string", "Compression type: gzip, lzf, none"),
            ("core.compression_level", "int", "Compression level (1-9)"),
            ("core.similarity_threshold", "float", "Threshold for deduplication (0-1)"),
            ("core.magnitude_tolerance", "float", "Magnitude tolerance (0-1)"),
            ("core.enable_lsh", "bool", "Enable LSH for large repos"),
            ("core.delta_encoding", "bool", "Enable delta encoding"),
            ("core.delta_type", "string", "Delta encoding strategy"),
            ("core.strict_reconstruction", "bool", "Fail on hash mismatch"),
            ("delta.sparse_threshold", "float", "Sparse encoding threshold"),
            ("delta.quantization_bits", "int", "Quantization bits (8 or 16)"),
            ("delta.min_weight_size", "int", "Min weight size for delta (bytes)"),
            ("delta.max_delta_ratio", "float", "Max delta size ratio"),
            ("storage.compression", "string", "HDF5 compression type"),
            ("storage.compression_level", "int", "HDF5 compression level"),
            ("lsh.num_hyperplanes", "int", "LSH hyperplanes per table"),
            ("lsh.num_tables", "int", "Number of LSH tables"),
            ("lsh.max_candidates", "int", "Max LSH candidates"),
            ("simhash.num_bits", "int", "SimHash fingerprint bits (64/128)"),
            ("simhash.similarity_threshold", "float", "SimHash threshold"),
            ("logging.level", "string", "Log level: DEBUG, INFO, etc."),
        ]

        print(f"{'Key':<35} {'Type':<8} {'Description'}")
        print("-" * 60)
        for key, type_str, desc in options:
            print(f"{key:<35} {type_str:<8} {desc}")

    def _parse_config_value(self, value: str):
        """Parse a configuration value from string."""
        # Handle booleans
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        if value.lower() in ("false", "no", "0", "off"):
            return False

        # Handle None
        if value.lower() in ("none", "null", ""):
            return None

        # Try numeric types
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # Return as string
        return value

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
