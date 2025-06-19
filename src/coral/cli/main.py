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

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.version_control.repository import Repository


class CoralCLI:
    """Main CLI interface for Coral."""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            prog="coral", description="Version control for neural network weights"
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
        subparsers.add_parser(
            "gc", help="Garbage collect unreferenced weights"
        )

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


def main():
    """Main entry point."""
    cli = CoralCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()
