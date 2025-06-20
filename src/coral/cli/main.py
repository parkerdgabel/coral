#!/usr/bin/env python3
"""
Coral CLI - Git-like version control for neural network weights
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.safetensors.converter import (
    convert_coral_to_safetensors,
    convert_safetensors_to_coral,
)
from coral.version_control.repository import Repository


class CoralCLI:
    """Main CLI interface for Coral."""

    def __init__(self) -> None:
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
            "--metadata",
            action="append",
            help="Add custom metadata (format: key=value)",
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

        return parser

    def run(self, args: Optional[List[str]] = None) -> int:
        """Run the CLI."""
        parsed_args = self.parser.parse_args(args)

        if not parsed_args.command:
            self.parser.print_help()
            return 0

        # Find repository root
        repo_path: Optional[Path] = None
        if parsed_args.command not in ["init", "convert"]:
            repo_path = self._find_repo_root()
            if repo_path is None:
                print("Error: Not in a Coral repository", file=sys.stderr)
                return 1

        # Execute command
        try:
            if parsed_args.command == "init":
                return self._cmd_init(parsed_args)
            elif parsed_args.command == "add":
                assert repo_path is not None
                return self._cmd_add(parsed_args, repo_path)
            elif parsed_args.command == "commit":
                assert repo_path is not None
                return self._cmd_commit(parsed_args, repo_path)
            elif parsed_args.command == "status":
                assert repo_path is not None
                return self._cmd_status(parsed_args, repo_path)
            elif parsed_args.command == "log":
                assert repo_path is not None
                return self._cmd_log(parsed_args, repo_path)
            elif parsed_args.command == "checkout":
                assert repo_path is not None
                return self._cmd_checkout(parsed_args, repo_path)
            elif parsed_args.command == "branch":
                assert repo_path is not None
                return self._cmd_branch(parsed_args, repo_path)
            elif parsed_args.command == "merge":
                assert repo_path is not None
                return self._cmd_merge(parsed_args, repo_path)
            elif parsed_args.command == "diff":
                assert repo_path is not None
                return self._cmd_diff(parsed_args, repo_path)
            elif parsed_args.command == "tag":
                assert repo_path is not None
                return self._cmd_tag(parsed_args, repo_path)
            elif parsed_args.command == "show":
                assert repo_path is not None
                return self._cmd_show(parsed_args, repo_path)
            elif parsed_args.command == "gc":
                assert repo_path is not None
                return self._cmd_gc(parsed_args, repo_path)
            elif parsed_args.command == "import-safetensors":
                assert repo_path is not None
                return self._cmd_import_safetensors(parsed_args, repo_path)
            elif parsed_args.command == "export-safetensors":
                assert repo_path is not None
                return self._cmd_export_safetensors(parsed_args, repo_path)
            elif parsed_args.command == "convert":
                return self._cmd_convert(parsed_args)
            else:
                print(
                    f"Error: Unknown command '{parsed_args.command}'", file=sys.stderr
                )
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

    def _cmd_init(self, args: Any) -> int:
        """Initialize a new repository."""
        path = Path(args.path).resolve()

        try:
            Repository(path, init=True)
            print(f"Initialized empty Coral repository in {path / '.coral'}")
            return 0
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    def _cmd_add(self, args: Any, repo_path: Path) -> int:
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

    def _cmd_commit(self, args: Any, repo_path: Path) -> int:
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

    def _cmd_status(self, args: Any, repo_path: Path) -> int:
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

    def _cmd_log(self, args: Any, repo_path: Path) -> int:
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

    def _cmd_checkout(self, args: Any, repo_path: Path) -> int:
        """Checkout branch or commit."""
        repo = Repository(repo_path)

        repo.checkout(args.target)
        print(f"Switched to '{args.target}'")

        return 0

    def _cmd_branch(self, args: Any, repo_path: Path) -> int:
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

    def _cmd_merge(self, args: Any, repo_path: Path) -> int:
        """Merge branches."""
        repo = Repository(repo_path)

        commit = repo.merge(args.branch, message=args.message)
        current = repo.branch_manager.get_current_branch()

        print(f"Merged {args.branch} into {current}")
        print(f"[{current} {commit.commit_hash[:8]}] {commit.metadata.message}")

        return 0

    def _cmd_diff(self, args: Any, repo_path: Path) -> int:
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

    def _cmd_tag(self, args: Any, repo_path: Path) -> int:
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

    def _cmd_show(self, args: Any, repo_path: Path) -> int:
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

    def _cmd_gc(self, args: Any, repo_path: Path) -> int:
        """Garbage collect unreferenced weights."""
        repo = Repository(repo_path)

        result = repo.gc()

        print("Garbage collection complete:")
        print(f"  Cleaned: {result['cleaned_weights']} weight(s)")
        print(f"  Remaining: {result['remaining_weights']} weight(s)")

        return 0

    def _cmd_import_safetensors(self, args: Any, repo_path: Path) -> int:
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

    def _cmd_export_safetensors(self, args: Any, repo_path: Path) -> int:
        """Export weights to a Safetensors file."""
        repo = Repository(repo_path)

        # Parse custom metadata
        custom_metadata = {}
        if args.metadata:
            for item in args.metadata:
                if "=" not in item:
                    print(
                        f"Error: Invalid metadata format: {item} (expected key=value)",
                        file=sys.stderr,
                    )
                    return 1
                key, value = item.split("=", 1)
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

    def _cmd_convert(self, args: Any) -> int:
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
                        weight_dict = {
                            name: tensor.data for name, tensor in weights.items()
                        }
                        np.savez_compressed(str(output_path), **weight_dict)  # type: ignore[arg-type]

                        print(f"✓ Converted {len(weight_dict)} weight(s) to NPZ format")
                else:
                    # Direct HDF5 conversion
                    from coral.storage.hdf5_store import HDF5Store

                    store = HDF5Store(str(output_path))

                    # Create a temporary repository and use it for conversion
                    import tempfile

                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_repo = Repository(Path(temp_dir), init=True)
                        weight_mapping = convert_safetensors_to_coral(
                            source_path=input_path,
                            target=temp_repo,
                            preserve_metadata=not args.no_metadata,
                            weight_names=args.weights,
                        )

                        # Copy weights to HDF5 store
                        weights = temp_repo.get_all_weights()
                        for _name, weight in weights.items():
                            store.store(weight)

                    store.close()  # type: ignore[no-untyped-call]
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
                        weights = {
                            k: v for k, v in weights.items() if k in args.weights
                        }

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
            elif (
                input_path.is_dir()
                and (input_path / ".coral").exists()
                and output_ext == ".safetensors"
            ):
                print("Converting Coral repository to Safetensors...")

                repo = Repository(input_path)
                convert_coral_to_safetensors(
                    source=repo,
                    output_path=output_path,
                    weight_names=args.weights,
                    include_metadata=not args.no_metadata,
                )

                num_weights = (
                    len(args.weights) if args.weights else len(repo.get_all_weights())
                )
                print(f"✓ Converted {num_weights} weight(s) to Safetensors format")

            else:
                print(
                    f"Error: Unsupported conversion from {input_ext} to {output_ext}",
                    file=sys.stderr,
                )
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


def main() -> None:
    """Main entry point."""
    cli = CoralCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()
