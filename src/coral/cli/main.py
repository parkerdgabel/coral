#!/usr/bin/env python3
"""
Coral CLI - Git-like version control for neural network weights
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import numpy as np
from tqdm import tqdm

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.version_control.repository import Repository
from coral.safetensors.converter import (
    convert_coral_to_safetensors,
    convert_safetensors_to_coral,
)


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
        subparsers.add_parser(
            "gc", help="Garbage collect unreferenced weights"
        )

        # Import Safetensors command
        import_st_parser = subparsers.add_parser(
            "import-safetensors", help="Import weights from a Safetensors file"
        )
        import_st_parser.add_argument("file", help="Path to Safetensors file")
        import_st_parser.add_argument(
            "--weights", nargs="+", help="Specific weights to import (default: all)"
        )
        import_st_parser.add_argument(
            "--exclude", nargs="+", help="Weights to exclude from import"
        )
        import_st_parser.add_argument(
            "--no-metadata", action="store_true", help="Don't preserve metadata"
        )

        # Export Safetensors command
        export_st_parser = subparsers.add_parser(
            "export-safetensors", help="Export weights to a Safetensors file"
        )
        export_st_parser.add_argument("output", help="Output Safetensors file path")
        export_st_parser.add_argument(
            "--weights", nargs="+", help="Specific weights to export (default: all)"
        )
        export_st_parser.add_argument(
            "--no-metadata", action="store_true", help="Don't include Coral metadata"
        )
        export_st_parser.add_argument(
            "--metadata", action="append", help="Add custom metadata (format: key=value)"
        )

        # Convert command
        convert_parser = subparsers.add_parser(
            "convert", help="Convert between weight file formats"
        )
        convert_parser.add_argument("input", help="Input file path")
        convert_parser.add_argument("output", help="Output file path")
        convert_parser.add_argument(
            "--weights", nargs="+", help="Specific weights to convert (default: all)"
        )
        convert_parser.add_argument(
            "--no-metadata", action="store_true", help="Don't preserve/include metadata"
        )

        # Cluster command group
        cluster_parser = subparsers.add_parser(
            "cluster", help="Manage weight clustering"
        )
        cluster_subparsers = cluster_parser.add_subparsers(
            dest="cluster_command", help="Clustering commands"
        )

        # Cluster analyze command
        analyze_parser = cluster_subparsers.add_parser(
            "analyze", help="Analyze repository for clustering opportunities"
        )
        analyze_parser.add_argument(
            "--commit", help="Analyze specific commit (default: HEAD)"
        )
        analyze_parser.add_argument(
            "--threshold", type=float, default=0.98,
            help="Similarity threshold for analysis (default: 0.98)"
        )
        analyze_parser.add_argument(
            "--format", choices=["text", "json", "csv"], default="text",
            help="Output format (default: text)"
        )

        # Cluster status command
        cluster_subparsers.add_parser(
            "status", help="Show current clustering status and statistics"
        )

        # Cluster report command
        report_parser = cluster_subparsers.add_parser(
            "report", help="Generate detailed clustering report"
        )
        report_parser.add_argument(
            "--output", help="Output file path (default: stdout)"
        )
        report_parser.add_argument(
            "--format", choices=["text", "json", "csv", "html"], default="text",
            help="Output format (default: text)"
        )
        report_parser.add_argument(
            "--verbose", action="store_true", help="Include detailed weight information"
        )

        # Cluster create command
        create_parser = cluster_subparsers.add_parser(
            "create", help="Create clusters using specified strategy"
        )
        create_parser.add_argument(
            "strategy", nargs="?", default="adaptive",
            choices=["kmeans", "hierarchical", "dbscan", "adaptive"],
            help="Clustering strategy (default: adaptive)"
        )
        create_parser.add_argument(
            "--levels", type=int, default=3,
            help="Number of hierarchy levels for multi-level clustering (default: 3)"
        )
        create_parser.add_argument(
            "--threshold", type=float, default=0.98,
            help="Similarity threshold for clustering (default: 0.98)"
        )
        create_parser.add_argument(
            "--optimize", action="store_true",
            help="Auto-optimize after clustering"
        )
        create_parser.add_argument(
            "--dry-run", action="store_true",
            help="Show what would be done without making changes"
        )

        # Cluster optimize command
        optimize_parser = cluster_subparsers.add_parser(
            "optimize", help="Optimize existing clusters"
        )
        optimize_parser.add_argument(
            "--aggressive", action="store_true",
            help="Use aggressive optimization strategies"
        )
        optimize_parser.add_argument(
            "--target-reduction", type=float,
            help="Target space reduction percentage"
        )

        # Cluster rebalance command
        rebalance_parser = cluster_subparsers.add_parser(
            "rebalance", help="Rebalance cluster assignments"
        )
        rebalance_parser.add_argument(
            "--max-cluster-size", type=int,
            help="Maximum weights per cluster"
        )
        rebalance_parser.add_argument(
            "--min-cluster-size", type=int, default=2,
            help="Minimum weights per cluster (default: 2)"
        )

        # Cluster list command
        list_parser = cluster_subparsers.add_parser(
            "list", help="List all clusters in repository"
        )
        list_parser.add_argument(
            "--sort-by", choices=["id", "size", "quality", "compression"],
            default="id", help="Sort clusters by (default: id)"
        )
        list_parser.add_argument(
            "--limit", type=int, help="Limit number of clusters shown"
        )
        list_parser.add_argument(
            "--format", choices=["text", "json", "csv"], default="text",
            help="Output format (default: text)"
        )

        # Cluster info command
        info_parser = cluster_subparsers.add_parser(
            "info", help="Show detailed cluster information"
        )
        info_parser.add_argument("cluster_id", help="Cluster ID to inspect")
        info_parser.add_argument(
            "--show-weights", action="store_true",
            help="Show individual weight information"
        )
        info_parser.add_argument(
            "--show-stats", action="store_true",
            help="Show detailed statistics"
        )

        # Cluster export command
        export_parser = cluster_subparsers.add_parser(
            "export", help="Export clustering configuration"
        )
        export_parser.add_argument("output_file", help="Output file path")
        export_parser.add_argument(
            "--format", choices=["json", "yaml", "toml"], default="json",
            help="Export format (default: json)"
        )
        export_parser.add_argument(
            "--include-weights", action="store_true",
            help="Include weight data in export"
        )

        # Cluster import command
        import_parser = cluster_subparsers.add_parser(
            "import", help="Import clustering configuration"
        )
        import_parser.add_argument("input_file", help="Input file path")
        import_parser.add_argument(
            "--merge", action="store_true",
            help="Merge with existing clusters instead of replacing"
        )
        import_parser.add_argument(
            "--validate", action="store_true",
            help="Validate configuration before importing"
        )

        # Cluster compare command
        compare_parser = cluster_subparsers.add_parser(
            "compare", help="Compare clustering between commits"
        )
        compare_parser.add_argument("commit1", help="First commit to compare")
        compare_parser.add_argument("commit2", help="Second commit to compare")
        compare_parser.add_argument(
            "--show-migrations", action="store_true",
            help="Show weight migrations between clusters"
        )
        compare_parser.add_argument(
            "--format", choices=["text", "json"], default="text",
            help="Output format (default: text)"
        )

        # Cluster validate command
        validate_parser = cluster_subparsers.add_parser(
            "validate", help="Validate cluster integrity and quality"
        )
        validate_parser.add_argument(
            "--fix", action="store_true",
            help="Attempt to fix validation errors"
        )
        validate_parser.add_argument(
            "--strict", action="store_true",
            help="Use strict validation rules"
        )

        # Cluster benchmark command
        benchmark_parser = cluster_subparsers.add_parser(
            "benchmark", help="Run clustering performance benchmarks"
        )
        benchmark_parser.add_argument(
            "--strategies", nargs="+",
            choices=["kmeans", "hierarchical", "dbscan", "adaptive"],
            help="Strategies to benchmark (default: all)"
        )
        benchmark_parser.add_argument(
            "--iterations", type=int, default=3,
            help="Number of iterations per strategy (default: 3)"
        )
        benchmark_parser.add_argument(
            "--output", help="Output results to file"
        )

        return parser

    def run(self, args=None) -> int:
        """Run the CLI."""
        args = self.parser.parse_args(args)

        if not args.command:
            self.parser.print_help()
            return 0

        # Find repository root
        if args.command not in ["init", "convert"]:
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
            elif args.command == "import-safetensors":
                return self._cmd_import_safetensors(args, repo_path)
            elif args.command == "export-safetensors":
                return self._cmd_export_safetensors(args, repo_path)
            elif args.command == "convert":
                return self._cmd_convert(args)
            elif args.command == "cluster":
                return self._cmd_cluster(args, repo_path)
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

    def _cmd_import_safetensors(self, args, repo_path: Path) -> int:
        """Import weights from a Safetensors file."""
        repo = Repository(repo_path)
        
        # Check if file exists
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            return 1
        
        if not file_path.suffix == ".safetensors":
            print("Warning: File does not have .safetensors extension", file=sys.stderr)
        
        # Prepare exclude set
        exclude_weights = set(args.exclude) if args.exclude else None
        
        try:
            print(f"Importing weights from {file_path.name}...")
            
            # Convert with progress
            weight_mapping = convert_safetensors_to_coral(
                source_path=file_path,
                target=repo,
                preserve_metadata=not args.no_metadata,
                weight_names=args.weights,
                exclude_weights=exclude_weights,
            )
            
            print(f"✓ Successfully imported {len(weight_mapping)} weight(s)")
            
            # Show deduplication stats
            stats = repo.deduplicator.compute_stats()
            if stats.total_weights > stats.unique_weights:
                saved = stats.total_weights - stats.unique_weights
                reduction = (saved / stats.total_weights) * 100
                print(f"✓ Deduplicated {saved} weight(s) ({reduction:.1f}% reduction)")
            
            return 0
            
        except Exception as e:
            print(f"Error importing safetensors: {e}", file=sys.stderr)
            return 1

    def _cmd_export_safetensors(self, args, repo_path: Path) -> int:
        """Export weights to a Safetensors file."""
        repo = Repository(repo_path)
        
        # Parse custom metadata
        custom_metadata = {}
        if args.metadata:
            for item in args.metadata:
                if '=' not in item:
                    print(f"Error: Invalid metadata format: {item} (expected key=value)", file=sys.stderr)
                    return 1
                key, value = item.split('=', 1)
                custom_metadata[key] = value
        
        try:
            output_path = Path(args.output)
            
            # Add .safetensors extension if not present
            if output_path.suffix != ".safetensors":
                output_path = output_path.with_suffix(".safetensors")
            
            # Check if output directory exists
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"Exporting weights to {output_path}...")
            
            # Export with progress
            convert_coral_to_safetensors(
                source=repo,
                output_path=output_path,
                weight_names=args.weights,
                include_metadata=not args.no_metadata,
                custom_metadata=custom_metadata if custom_metadata else None,
            )
            
            # Get file size
            file_size = output_path.stat().st_size
            size_mb = file_size / (1024 * 1024)
            
            # Count weights exported
            if args.weights:
                num_weights = len(args.weights)
            else:
                num_weights = len(repo.get_all_weights())
            
            print(f"✓ Successfully exported {num_weights} weight(s)")
            print(f"✓ Output file: {output_path} ({size_mb:.1f} MB)")
            
            return 0
            
        except Exception as e:
            print(f"Error exporting to safetensors: {e}", file=sys.stderr)
            return 1

    def _cmd_convert(self, args) -> int:
        """Convert between weight file formats."""
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        # Check if input exists
        if not input_path.exists():
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            return 1
        
        # Detect format based on extensions
        input_ext = input_path.suffix.lower()
        output_ext = output_path.suffix.lower()
        
        # Add extensions if missing
        if not output_ext:
            if input_ext == ".safetensors":
                output_ext = ".npz"
            else:
                output_ext = ".safetensors"
            output_path = output_path.with_suffix(output_ext)
        
        try:
            # Safetensors to Coral/NPZ
            if input_ext == ".safetensors" and output_ext in [".npz", ".h5", ".hdf5"]:
                print(f"Converting Safetensors to {output_ext[1:].upper()}...")
                
                # For NPZ output, we need to create a temporary repository
                if output_ext == ".npz":
                    import tempfile
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_repo = Repository(Path(temp_dir), init=True)
                        
                        # Import to temporary repository
                        weight_mapping = convert_safetensors_to_coral(
                            source_path=input_path,
                            target=temp_repo,
                            preserve_metadata=not args.no_metadata,
                            weight_names=args.weights,
                        )
                        
                        # Export as NPZ
                        weights = temp_repo.get_all_weights()
                        weight_dict = {name: tensor.data for name, tensor in weights.items()}
                        np.savez_compressed(output_path, **weight_dict)
                        
                        print(f"✓ Converted {len(weight_dict)} weight(s) to NPZ format")
                else:
                    # Direct HDF5 conversion
                    from coral.storage.hdf5_store import HDF5Store
                    store = HDF5Store(output_path)
                    
                    weight_mapping = convert_safetensors_to_coral(
                        source_path=input_path,
                        target=store,
                        preserve_metadata=not args.no_metadata,
                        weight_names=args.weights,
                    )
                    
                    store.close()
                    print(f"✓ Converted {len(weight_mapping)} weight(s) to HDF5 format")
            
            # NPZ to Safetensors
            elif input_ext == ".npz" and output_ext == ".safetensors":
                print("Converting NPZ to Safetensors...")
                
                # Load NPZ file
                archive = np.load(input_path)
                
                # Create temporary repository for conversion
                import tempfile
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_repo = Repository(Path(temp_dir), init=True)
                    
                    # Stage all weights
                    weights = {}
                    for name in archive.files:
                        data = archive[name]
                        weight = WeightTensor(
                            data=data,
                            metadata=WeightMetadata(
                                name=name,
                                shape=data.shape,
                                dtype=data.dtype,
                            ),
                        )
                        weights[name] = weight
                    
                    # Filter weights if specified
                    if args.weights:
                        weights = {k: v for k, v in weights.items() if k in args.weights}
                    
                    temp_repo.stage_weights(weights)
                    temp_repo.commit("Import from NPZ")
                    
                    # Export to safetensors
                    convert_coral_to_safetensors(
                        source=temp_repo,
                        output_path=output_path,
                        include_metadata=not args.no_metadata,
                    )
                    
                    print(f"✓ Converted {len(weights)} weight(s) to Safetensors format")
            
            # Coral repository to Safetensors
            elif input_path.is_dir() and (input_path / ".coral").exists() and output_ext == ".safetensors":
                print("Converting Coral repository to Safetensors...")
                
                repo = Repository(input_path)
                convert_coral_to_safetensors(
                    source=repo,
                    output_path=output_path,
                    weight_names=args.weights,
                    include_metadata=not args.no_metadata,
                )
                
                num_weights = len(args.weights) if args.weights else len(repo.get_all_weights())
                print(f"✓ Converted {num_weights} weight(s) to Safetensors format")
            
            else:
                print(f"Error: Unsupported conversion from {input_ext} to {output_ext}", file=sys.stderr)
                print("Supported conversions:", file=sys.stderr)
                print("  - .safetensors → .npz, .h5, .hdf5", file=sys.stderr)
                print("  - .npz → .safetensors", file=sys.stderr)
                print("  - Coral repository → .safetensors", file=sys.stderr)
                return 1
            
            # Show output file info
            file_size = output_path.stat().st_size
            size_mb = file_size / (1024 * 1024)
            print(f"✓ Output file: {output_path} ({size_mb:.1f} MB)")
            
            return 0
            
        except Exception as e:
            print(f"Error during conversion: {e}", file=sys.stderr)
            return 1

    def _cmd_cluster(self, args, repo_path: Path) -> int:
        """Handle cluster subcommands."""
        if not args.cluster_command:
            self.parser.parse_args(['cluster', '--help'])
            return 0
        
        # Dispatch to specific cluster command
        if args.cluster_command == "analyze":
            return self._cmd_cluster_analyze(args, repo_path)
        elif args.cluster_command == "status":
            return self._cmd_cluster_status(args, repo_path)
        elif args.cluster_command == "report":
            return self._cmd_cluster_report(args, repo_path)
        elif args.cluster_command == "create":
            return self._cmd_cluster_create(args, repo_path)
        elif args.cluster_command == "optimize":
            return self._cmd_cluster_optimize(args, repo_path)
        elif args.cluster_command == "rebalance":
            return self._cmd_cluster_rebalance(args, repo_path)
        elif args.cluster_command == "list":
            return self._cmd_cluster_list(args, repo_path)
        elif args.cluster_command == "info":
            return self._cmd_cluster_info(args, repo_path)
        elif args.cluster_command == "export":
            return self._cmd_cluster_export(args, repo_path)
        elif args.cluster_command == "import":
            return self._cmd_cluster_import(args, repo_path)
        elif args.cluster_command == "compare":
            return self._cmd_cluster_compare(args, repo_path)
        elif args.cluster_command == "validate":
            return self._cmd_cluster_validate(args, repo_path)
        elif args.cluster_command == "benchmark":
            return self._cmd_cluster_benchmark(args, repo_path)
        else:
            print(f"Error: Unknown cluster command '{args.cluster_command}'", file=sys.stderr)
            return 1

    def _cmd_cluster_analyze(self, args, repo_path: Path) -> int:
        """Analyze repository for clustering opportunities."""
        repo = Repository(repo_path)
        
        if args.format != "json":
            print("Analyzing repository for clustering opportunities...")
        
        # Analyze clustering potential
        analysis = repo.analyze_clustering(
            commit_ref=args.commit,
            similarity_threshold=args.threshold
        )
        
        if args.format == "json":
            print(json.dumps(analysis, indent=2))
        elif args.format == "csv":
            # CSV format
            print("metric,value")
            print(f"total_weights,{analysis['total_weights']}")
            print(f"unique_weights,{analysis['unique_weights']}")
            print(f"potential_clusters,{analysis['potential_clusters']}")
            print(f"estimated_reduction,{analysis['estimated_reduction']:.2%}")
            print(f"clustering_quality,{analysis['clustering_quality']:.3f}")
        else:
            # Text format
            print(f"\nRepository Clustering Analysis:")
            print(f"  Total weights:        {analysis['total_weights']:,}")
            print(f"  Unique weights:       {analysis['unique_weights']:,}")
            print(f"  Potential clusters:   {analysis['potential_clusters']:,}")
            print(f"  Estimated reduction:  {analysis['estimated_reduction']:.1%}")
            print(f"  Clustering quality:   {analysis['clustering_quality']:.3f}")
            
            if analysis.get('recommendations'):
                print("\nRecommendations:")
                for rec in analysis['recommendations']:
                    print(f"  - {rec}")
            
            if analysis.get('weight_distribution'):
                print("\nWeight distribution by shape:")
                for shape, count in analysis['weight_distribution'].items():
                    print(f"  {shape}: {count} weights")
        
        return 0

    def _cmd_cluster_status(self, args, repo_path: Path) -> int:
        """Show current clustering status."""
        repo = Repository(repo_path)
        
        status = repo.get_clustering_status()
        
        if not status['enabled']:
            print("Clustering is not enabled in this repository.")
            print("Run 'coral-ml cluster create' to enable clustering.")
            return 0
        
        print(f"Clustering Status:")
        print(f"  Strategy:          {status['strategy']}")
        print(f"  Clusters:          {status['num_clusters']:,}")
        print(f"  Clustered weights: {status['clustered_weights']:,}")
        print(f"  Space saved:       {status['space_saved_bytes'] / (1024*1024):.1f} MB")
        print(f"  Reduction:         {status['reduction_percentage']:.1%}")
        print(f"  Last updated:      {status['last_updated']}")
        
        if status.get('cluster_health'):
            print(f"\nCluster Health:")
            print(f"  Healthy:   {status['cluster_health']['healthy']}")
            print(f"  Warnings:  {status['cluster_health']['warnings']}")
            print(f"  Errors:    {status['cluster_health']['errors']}")
        
        return 0

    def _cmd_cluster_report(self, args, repo_path: Path) -> int:
        """Generate detailed clustering report."""
        repo = Repository(repo_path)
        
        print("Generating clustering report...")
        report = repo.generate_clustering_report(verbose=args.verbose)
        
        # Format report based on output format
        if args.format == "json":
            output = json.dumps(report, indent=2)
        elif args.format == "csv":
            # Generate CSV report
            lines = ["cluster_id,size,quality,compression_ratio,space_saved"]
            for cluster in report['clusters']:
                lines.append(f"{cluster['id']},{cluster['size']},{cluster['quality']:.3f},"
                           f"{cluster['compression_ratio']:.2f},{cluster['space_saved']}")
            output = "\n".join(lines)
        elif args.format == "html":
            # Generate HTML report
            output = self._generate_html_report(report)
        else:
            # Text format
            output = self._format_text_report(report, args.verbose)
        
        # Output to file or stdout
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(output)
            print(f"Report saved to: {output_path}")
        else:
            print(output)
        
        return 0

    def _cmd_cluster_create(self, args, repo_path: Path) -> int:
        """Create clusters using specified strategy."""
        repo = Repository(repo_path)
        
        if args.dry_run:
            print(f"DRY RUN: Would create clusters using {args.strategy} strategy")
            analysis = repo.analyze_clustering(similarity_threshold=args.threshold)
            print(f"  Estimated clusters: {analysis['potential_clusters']}")
            print(f"  Estimated reduction: {analysis['estimated_reduction']:.1%}")
            return 0
        
        print(f"Creating clusters using {args.strategy} strategy...")
        
        # Create progress bar
        with tqdm(desc="Creating clusters", unit="weights") as pbar:
            def progress_callback(current, total):
                pbar.total = total
                pbar.update(current - pbar.n)
            
            result = repo.create_clusters(
                strategy=args.strategy,
                levels=args.levels,
                similarity_threshold=args.threshold,
                progress_callback=progress_callback
            )
        
        print(f"\nClustering complete:")
        print(f"  Created clusters:    {result['num_clusters']:,}")
        print(f"  Clustered weights:   {result['weights_clustered']:,}")
        print(f"  Space saved:         {result['space_saved'] / (1024*1024):.1f} MB")
        print(f"  Reduction:           {result['reduction_percentage']:.1%}")
        print(f"  Time elapsed:        {result['time_elapsed']:.1f}s")
        
        if args.optimize:
            print("\nOptimizing clusters...")
            opt_result = repo.optimize_clusters()
            print(f"  Additional savings:  {opt_result['additional_savings'] / (1024*1024):.1f} MB")
        
        return 0

    def _cmd_cluster_optimize(self, args, repo_path: Path) -> int:
        """Optimize existing clusters."""
        repo = Repository(repo_path)
        
        print("Optimizing clusters...")
        
        result = repo.optimize_clusters(
            aggressive=args.aggressive,
            target_reduction=args.target_reduction
        )
        
        print(f"\nOptimization complete:")
        print(f"  Clusters optimized:  {result['clusters_optimized']}")
        print(f"  Weights moved:       {result['weights_moved']}")
        print(f"  Space saved:         {result['space_saved'] / (1024*1024):.1f} MB")
        print(f"  Quality improvement: {result['quality_improvement']:.3f}")
        
        if result.get('warnings'):
            print("\nWarnings:")
            for warning in result['warnings']:
                print(f"  - {warning}")
        
        return 0

    def _cmd_cluster_rebalance(self, args, repo_path: Path) -> int:
        """Rebalance cluster assignments."""
        repo = Repository(repo_path)
        
        print("Rebalancing clusters...")
        
        result = repo.rebalance_clusters(
            max_cluster_size=args.max_cluster_size,
            min_cluster_size=args.min_cluster_size
        )
        
        print(f"\nRebalancing complete:")
        print(f"  Clusters merged:     {result['clusters_merged']}")
        print(f"  Clusters split:      {result['clusters_split']}")
        print(f"  Weights reassigned:  {result['weights_reassigned']}")
        print(f"  Balance score:       {result['balance_score']:.3f}")
        
        return 0

    def _cmd_cluster_list(self, args, repo_path: Path) -> int:
        """List all clusters."""
        repo = Repository(repo_path)
        
        clusters = repo.list_clusters(sort_by=args.sort_by, limit=args.limit)
        
        if not clusters:
            print("No clusters found.")
            return 0
        
        if args.format == "json":
            print(json.dumps(clusters, indent=2))
        elif args.format == "csv":
            print("cluster_id,size,quality,compression_ratio,centroid_hash")
            for cluster in clusters:
                print(f"{cluster['id']},{cluster['size']},{cluster['quality']:.3f},"
                     f"{cluster['compression_ratio']:.2f},{cluster['centroid_hash'][:8]}")
        else:
            # Text format
            print(f"{'ID':<12} {'Size':>6} {'Quality':>8} {'Compression':>11} {'Centroid'}")
            print("-" * 60)
            for cluster in clusters:
                print(f"{cluster['id']:<12} {cluster['size']:>6} "
                     f"{cluster['quality']:>8.3f} {cluster['compression_ratio']:>10.1f}x "
                     f"{cluster['centroid_hash'][:8]}")
            
            print(f"\nTotal clusters: {len(clusters)}")
        
        return 0

    def _cmd_cluster_info(self, args, repo_path: Path) -> int:
        """Show detailed cluster information."""
        repo = Repository(repo_path)
        
        info = repo.get_cluster_info(args.cluster_id)
        
        if not info:
            print(f"Error: Cluster '{args.cluster_id}' not found", file=sys.stderr)
            return 1
        
        print(f"Cluster: {info['id']}")
        print(f"Size: {info['size']} weights")
        print(f"Quality: {info['quality']:.3f}")
        print(f"Compression: {info['compression_ratio']:.2f}x")
        print(f"Space saved: {info['space_saved'] / (1024*1024):.1f} MB")
        print(f"Centroid: {info['centroid_hash']}")
        print(f"Created: {info['created']}")
        
        if args.show_stats:
            print(f"\nStatistics:")
            stats = info['statistics']
            print(f"  Mean similarity: {stats['mean_similarity']:.3f}")
            print(f"  Min similarity:  {stats['min_similarity']:.3f}")
            print(f"  Max similarity:  {stats['max_similarity']:.3f}")
            print(f"  Std deviation:   {stats['std_deviation']:.3f}")
        
        if args.show_weights:
            print(f"\nWeights ({len(info['weights'])}):")
            for weight in info['weights']:
                print(f"  - {weight['name']} ({weight['hash'][:8]}) "
                     f"similarity: {weight['similarity']:.3f}")
        
        return 0

    def _cmd_cluster_export(self, args, repo_path: Path) -> int:
        """Export clustering configuration."""
        repo = Repository(repo_path)
        
        print(f"Exporting clustering configuration to {args.output_file}...")
        
        config = repo.export_clustering_config(include_weights=args.include_weights)
        
        output_path = Path(args.output_file)
        
        if args.format == "json":
            output_path.write_text(json.dumps(config, indent=2))
        elif args.format == "yaml":
            try:
                import yaml
                output_path.write_text(yaml.dump(config, default_flow_style=False))
            except ImportError:
                print("Error: PyYAML not installed. Use 'pip install pyyaml'", file=sys.stderr)
                return 1
        elif args.format == "toml":
            try:
                import toml
                output_path.write_text(toml.dumps(config))
            except ImportError:
                print("Error: toml not installed. Use 'pip install toml'", file=sys.stderr)
                return 1
        
        file_size = output_path.stat().st_size / 1024
        print(f"Configuration exported ({file_size:.1f} KB)")
        
        return 0

    def _cmd_cluster_import(self, args, repo_path: Path) -> int:
        """Import clustering configuration."""
        repo = Repository(repo_path)
        
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"Error: File not found: {args.input_file}", file=sys.stderr)
            return 1
        
        # Load configuration
        try:
            if input_path.suffix == ".json":
                config = json.loads(input_path.read_text())
            elif input_path.suffix in [".yaml", ".yml"]:
                import yaml
                config = yaml.safe_load(input_path.read_text())
            elif input_path.suffix == ".toml":
                import toml
                config = toml.loads(input_path.read_text())
            else:
                print(f"Error: Unsupported format: {input_path.suffix}", file=sys.stderr)
                return 1
        except Exception as e:
            print(f"Error loading configuration: {e}", file=sys.stderr)
            return 1
        
        if args.validate:
            print("Validating configuration...")
            validation = repo.validate_clustering_config(config)
            if not validation['valid']:
                print("Configuration validation failed:")
                for error in validation['errors']:
                    print(f"  - {error}")
                return 1
            print("Configuration is valid.")
        
        print(f"Importing clustering configuration...")
        
        result = repo.import_clustering_config(config, merge=args.merge)
        
        print(f"\nImport complete:")
        print(f"  Clusters imported:   {result['clusters_imported']}")
        print(f"  Weights assigned:    {result['weights_assigned']}")
        print(f"  Conflicts resolved:  {result['conflicts_resolved']}")
        
        return 0

    def _cmd_cluster_compare(self, args, repo_path: Path) -> int:
        """Compare clustering between commits."""
        repo = Repository(repo_path)
        
        print(f"Comparing clustering: {args.commit1} vs {args.commit2}")
        
        comparison = repo.compare_clustering(args.commit1, args.commit2)
        
        if args.format == "json":
            print(json.dumps(comparison, indent=2))
        else:
            # Text format
            print(f"\nClustering Comparison:")
            print(f"  Commit 1: {comparison['commit1']['hash'][:8]} ({comparison['commit1']['date']})")
            print(f"    Clusters: {comparison['commit1']['num_clusters']}")
            print(f"    Weights:  {comparison['commit1']['clustered_weights']}")
            
            print(f"\n  Commit 2: {comparison['commit2']['hash'][:8]} ({comparison['commit2']['date']})")
            print(f"    Clusters: {comparison['commit2']['num_clusters']}")
            print(f"    Weights:  {comparison['commit2']['clustered_weights']}")
            
            print(f"\n  Changes:")
            print(f"    Clusters added:    {comparison['changes']['clusters_added']}")
            print(f"    Clusters removed:  {comparison['changes']['clusters_removed']}")
            print(f"    Clusters modified: {comparison['changes']['clusters_modified']}")
            
            if args.show_migrations and comparison.get('weight_migrations'):
                print(f"\n  Weight Migrations ({len(comparison['weight_migrations'])}):")
                for migration in comparison['weight_migrations'][:10]:
                    print(f"    {migration['weight']}: {migration['from']} -> {migration['to']}")
                if len(comparison['weight_migrations']) > 10:
                    print(f"    ... and {len(comparison['weight_migrations']) - 10} more")
        
        return 0

    def _cmd_cluster_validate(self, args, repo_path: Path) -> int:
        """Validate cluster integrity."""
        repo = Repository(repo_path)
        
        print("Validating cluster integrity...")
        
        validation = repo.validate_clusters(strict=args.strict)
        
        print(f"\nValidation Results:")
        print(f"  Valid:     {validation['valid']}")
        print(f"  Errors:    {len(validation['errors'])}")
        print(f"  Warnings:  {len(validation['warnings'])}")
        
        if validation['errors']:
            print("\nErrors:")
            for error in validation['errors']:
                print(f"  ✗ {error}")
        
        if validation['warnings']:
            print("\nWarnings:")
            for warning in validation['warnings']:
                print(f"  ⚠ {warning}")
        
        if args.fix and not validation['valid']:
            print("\nAttempting to fix errors...")
            fix_result = repo.fix_clustering_errors()
            print(f"  Fixed: {fix_result['errors_fixed']} error(s)")
            print(f"  Failed: {fix_result['errors_remaining']} error(s)")
            
            # Return success if all errors were fixed
            return 0 if fix_result['errors_remaining'] == 0 else 1
        
        return 0 if validation['valid'] else 1

    def _cmd_cluster_benchmark(self, args, repo_path: Path) -> int:
        """Run clustering benchmarks."""
        repo = Repository(repo_path)
        
        strategies = args.strategies or ["kmeans", "hierarchical", "dbscan", "adaptive"]
        
        print(f"Running clustering benchmarks...")
        print(f"Strategies: {', '.join(strategies)}")
        print(f"Iterations: {args.iterations}")
        print()
        
        results = []
        
        for strategy in strategies:
            print(f"Benchmarking {strategy}...")
            strategy_results = []
            
            for i in range(args.iterations):
                start_time = time.time()
                
                # Run clustering
                result = repo.create_clusters(
                    strategy=strategy,
                    benchmark_mode=True
                )
                
                elapsed = time.time() - start_time
                
                strategy_results.append({
                    'iteration': i + 1,
                    'time': elapsed,
                    'clusters': result['num_clusters'],
                    'quality': result['quality'],
                    'reduction': result['reduction_percentage']
                })
                
                print(f"  Iteration {i+1}: {elapsed:.2f}s, "
                     f"{result['num_clusters']} clusters, "
                     f"quality: {result['quality']:.3f}")
            
            # Calculate averages
            avg_time = sum(r['time'] for r in strategy_results) / len(strategy_results)
            avg_quality = sum(r['quality'] for r in strategy_results) / len(strategy_results)
            avg_reduction = sum(r['reduction'] for r in strategy_results) / len(strategy_results)
            
            results.append({
                'strategy': strategy,
                'avg_time': avg_time,
                'avg_quality': avg_quality,
                'avg_reduction': avg_reduction,
                'iterations': strategy_results
            })
        
        # Display summary
        print("\nBenchmark Summary:")
        print(f"{'Strategy':<12} {'Avg Time':>10} {'Avg Quality':>12} {'Avg Reduction':>14}")
        print("-" * 50)
        for result in results:
            print(f"{result['strategy']:<12} {result['avg_time']:>9.2f}s "
                 f"{result['avg_quality']:>12.3f} {result['avg_reduction']:>13.1%}")
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(json.dumps({
                'timestamp': datetime.now().isoformat(),
                'iterations': args.iterations,
                'results': results
            }, indent=2))
            print(f"\nResults saved to: {output_path}")
        
        return 0

    def _format_text_report(self, report: Dict[str, Any], verbose: bool) -> str:
        """Format clustering report as text."""
        lines = []
        
        lines.append("Coral Clustering Report")
        lines.append("=" * 50)
        lines.append(f"Generated: {report['timestamp']}")
        lines.append(f"Repository: {report['repository']}")
        lines.append("")
        
        # Overview
        lines.append("Overview:")
        lines.append(f"  Total clusters:      {report['overview']['total_clusters']:,}")
        lines.append(f"  Clustered weights:   {report['overview']['clustered_weights']:,}")
        lines.append(f"  Space saved:         {report['overview']['space_saved'] / (1024*1024):.1f} MB")
        lines.append(f"  Overall reduction:   {report['overview']['reduction_percentage']:.1%}")
        lines.append(f"  Average quality:     {report['overview']['average_quality']:.3f}")
        lines.append("")
        
        # Top clusters
        lines.append("Top Clusters by Size:")
        for cluster in report['top_clusters'][:10]:
            lines.append(f"  {cluster['id']}: {cluster['size']} weights, "
                        f"quality: {cluster['quality']:.3f}, "
                        f"saved: {cluster['space_saved'] / 1024:.1f} KB")
        
        if verbose and report.get('cluster_details'):
            lines.append("")
            lines.append("Detailed Cluster Information:")
            for cluster in report['cluster_details']:
                lines.append(f"\n  Cluster {cluster['id']}:")
                lines.append(f"    Size: {cluster['size']} weights")
                lines.append(f"    Quality: {cluster['quality']:.3f}")
                lines.append(f"    Compression: {cluster['compression_ratio']:.2f}x")
                lines.append(f"    Representative: {cluster['centroid_name']}")
        
        return "\n".join(lines)

    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML clustering report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Coral Clustering Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
        .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Coral Clustering Report</h1>
    <p>Generated: {report['timestamp']}</p>
    
    <div class="summary">
        <h2>Overview</h2>
        <p>Total Clusters: <span class="metric">{report['overview']['total_clusters']:,}</span></p>
        <p>Space Saved: <span class="metric">{report['overview']['space_saved'] / (1024*1024):.1f} MB</span></p>
        <p>Reduction: <span class="metric">{report['overview']['reduction_percentage']:.1%}</span></p>
    </div>
    
    <h2>Cluster Details</h2>
    <table>
        <tr>
            <th>Cluster ID</th>
            <th>Size</th>
            <th>Quality</th>
            <th>Compression</th>
            <th>Space Saved</th>
        </tr>
"""
        
        for cluster in report['clusters']:
            html += f"""
        <tr>
            <td>{cluster['id']}</td>
            <td>{cluster['size']}</td>
            <td>{cluster['quality']:.3f}</td>
            <td>{cluster['compression_ratio']:.2f}x</td>
            <td>{cluster['space_saved'] / 1024:.1f} KB</td>
        </tr>
"""
        
        html += """
    </table>
</body>
</html>
"""
        return html


def main():
    """Main entry point."""
    cli = CoralCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()
