import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, cast

from ..core.deduplicator import Deduplicator
from ..core.weight_tensor import WeightTensor
from ..delta.delta_encoder import DeltaConfig
from ..storage.hdf5_store import HDF5Store
from .branch import BranchManager
from .commit import Commit, CommitMetadata
from .version import Version, VersionGraph

logger = logging.getLogger(__name__)


class Repository:
    """Main repository class for version control operations."""

    def __init__(self, path: Path, init: bool = False):
        self.path = Path(path)
        self.coral_dir = self.path / ".coral"

        if init:
            self._initialize_repository()
        elif not self.coral_dir.exists():
            raise ValueError(f"Not a Coral repository: {self.path}")

        # Load configuration first
        self.config = self._load_config()

        # Initialize components
        self.branch_manager = BranchManager(self.path)
        self.version_graph = VersionGraph()
        self.current_branch_file = self.coral_dir / "HEAD"

        # Configure delta encoding based on repository settings
        from ..delta.delta_encoder import DeltaType

        delta_config = DeltaConfig()
        if self.config.get("core", {}).get("delta_encoding", True):
            delta_type_str = self.config.get("core", {}).get(
                "delta_type", "float32_raw"
            )
            delta_config.delta_type = DeltaType(delta_type_str)
            # Allow smaller weights for delta encoding in tests
            delta_config.min_weight_size = self.config.get("core", {}).get(
                "min_weight_size", 16
            )

        self.deduplicator = Deduplicator(
            similarity_threshold=self.config.get("core", {}).get(
                "similarity_threshold", 0.98
            ),
            delta_config=delta_config,
            enable_delta_encoding=self.config.get("core", {}).get(
                "delta_encoding", True
            ),
        )

        # Storage paths
        self.objects_dir = self.coral_dir / "objects"
        self.weights_store_path = self.objects_dir / "weights.h5"
        self.commits_dir = self.objects_dir / "commits"
        self.staging_dir = self.coral_dir / "staging"

        # Ensure directories exist
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.commits_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)

        # Load existing commits
        self._load_commits()

    @property
    def current_branch(self) -> str:
        """Get current branch name."""
        return self.branch_manager.get_current_branch()

    def _initialize_repository(self) -> None:
        """Initialize a new repository."""
        if self.coral_dir.exists():
            raise ValueError(f"Repository already exists at {self.path}")

        # Create directory structure
        self.coral_dir.mkdir(parents=True)
        (self.coral_dir / "objects").mkdir()
        (self.coral_dir / "objects" / "commits").mkdir()
        (self.coral_dir / "objects" / "weights.h5").touch()
        (self.coral_dir / "refs" / "heads").mkdir(parents=True)
        (self.coral_dir / "refs" / "tags").mkdir(parents=True)
        (self.coral_dir / "staging").mkdir()
        (self.coral_dir / "versions").mkdir()

        # Create initial config
        config = {
            "user": {"name": "Anonymous", "email": "anonymous@example.com"},
            "core": {
                "compression": "gzip",
                "similarity_threshold": 0.98,
                "delta_encoding": True,
                "min_weight_size": 16,
            },
        }

        with open(self.coral_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Also create the expected "config" file for backward compatibility
        with open(self.coral_dir / "config", "w") as f:
            json.dump(config, f, indent=2)

        # Create initial branch reference
        with open(self.coral_dir / "HEAD", "w") as f:
            f.write("ref: refs/heads/main")

        # Create the actual main branch - initially with no commits (empty hash)
        # The branch will be properly initialized when the first commit is made
        main_branch_file = self.coral_dir / "refs" / "heads" / "main"
        initial_branch_data = {
            "name": "main",
            "commit_hash": "",  # Empty until first commit
            "parent_branch": None,
        }
        with open(main_branch_file, "w") as f:
            json.dump(initial_branch_data, f, indent=2)

    def _load_config(self) -> Dict[str, Any]:
        """Load repository configuration."""
        config_file = self.coral_dir / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                data: Dict[str, Any] = json.load(f)
                return data
        return {}

    def _load_commits(self) -> None:
        """Load all commits into the version graph."""
        for commit_file in self.commits_dir.glob("*.json"):
            commit = Commit.load(commit_file)
            self.version_graph.add_commit(commit)

    def stage_weights(self, weights: Dict[str, WeightTensor]) -> Dict[str, str]:
        """Stage weights for commit with delta encoding support."""
        staged = {}

        with HDF5Store(
            str(self.weights_store_path),
            compression=self.config.get("core", {}).get("compression", "gzip"),
        ) as store_base:
            store = cast(HDF5Store, store_base)
            for name, weight in weights.items():
                # Add to deduplicator
                ref_hash = self.deduplicator.add_weight(weight, name)

                # Store weight if it's a new unique reference
                if ref_hash == weight.compute_hash():
                    store.store(weight)

                # Store delta if this weight was delta-encoded
                if self.deduplicator.is_delta_encoded(name):
                    delta = self.deduplicator.get_delta_by_name(name)
                    if delta:
                        delta_hash = self.deduplicator.name_to_delta[name]
                        store.store_delta(delta, delta_hash)
                        logger.debug(
                            f"Stored delta for {name}: "
                            f"{delta.compression_ratio:.2%} compression"
                        )

                staged[name] = ref_hash

        # Save staging info with delta information
        staging_info = {
            "weights": staged,
            "deltas": {
                name: self.deduplicator.name_to_delta.get(name)
                for name in staged
                if self.deduplicator.is_delta_encoded(name)
            },
        }

        with open(self.staging_dir / "staged.json", "w") as f:
            json.dump(staging_info, f, indent=2)

        return staged

    def commit(
        self,
        message: str,
        author: Optional[str] = None,
        email: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Commit:
        """Create a new commit from staged weights."""
        # Load staged weights
        staged_file = self.staging_dir / "staged.json"
        if not staged_file.exists():
            raise ValueError("Nothing to commit")

        with open(staged_file) as f:
            staged_data = json.load(f)

        # Handle both old flat format and new nested format
        if isinstance(staged_data, dict) and "weights" in staged_data:
            weight_hashes = staged_data["weights"]
            # Get delta information from staging
            staged_deltas = staged_data.get("deltas", {})
        else:
            weight_hashes = staged_data
            staged_deltas = {}

        if not weight_hashes:
            raise ValueError("Nothing to commit")

        # Get current branch and parent
        current_branch = self.branch_manager.get_current_branch()
        parent_commit_hash = self.branch_manager.get_branch_commit(current_branch)
        parent_hashes = [parent_commit_hash] if parent_commit_hash else []

        # Create commit metadata
        metadata = CommitMetadata(
            author=author or self.config.get("user", {}).get("name", "Anonymous"),
            email=email
            or self.config.get("user", {}).get("email", "anonymous@example.com"),
            message=message,
            tags=tags or [],
        )

        # Calculate deltas if we have a parent commit
        if parent_hashes and parent_hashes[0]:
            parent_commit = self.version_graph.get_commit(parent_hashes[0])
            if parent_commit:
                # Calculate deltas between current weights and parent
                commit_deltas = self._calculate_deltas(weight_hashes, parent_commit)
                # Merge with staged deltas (staged deltas from
                # deduplication have priority)
                delta_weights = {**commit_deltas, **staged_deltas}
            else:
                delta_weights = staged_deltas
        else:
            # Use staged deltas from deduplicator for root commits
            delta_weights = staged_deltas

        # Create commit hash
        commit_content = {
            "parent_hashes": parent_hashes,
            "weight_hashes": weight_hashes,
            "metadata": metadata.to_dict(),
            "delta_weights": delta_weights,
        }
        commit_hash = hashlib.sha256(
            json.dumps(commit_content, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Create and save commit
        commit = Commit(
            commit_hash=commit_hash,
            parent_hashes=parent_hashes,
            weight_hashes=weight_hashes,
            metadata=metadata,
            delta_weights=delta_weights,
        )

        commit.save(self.commits_dir / f"{commit_hash}.json")
        self.version_graph.add_commit(commit)

        # Update branch
        self.branch_manager.update_branch(current_branch, commit_hash)

        # Clear staging
        staged_file.unlink()

        return commit

    def checkout(self, target: str) -> None:
        """Checkout a branch or commit."""
        # Check if target is a branch
        branch = self.branch_manager.get_branch(target)
        if branch:
            self.branch_manager.set_current_branch(target)
            return

        # Resolve target to commit hash
        resolved_hash = self._resolve_ref(target)
        if resolved_hash:
            commit = self.version_graph.get_commit(resolved_hash)
            if commit:
                # Create detached HEAD state
                with open(self.coral_dir / "HEAD", "w") as f:
                    f.write(commit.commit_hash)
                return

        raise ValueError(f"Invalid target: {target}")

    def create_branch(self, name: str, from_ref: Optional[str] = None) -> None:
        """Create a new branch."""
        if from_ref:
            # Check if it's a valid commit
            commit = self.version_graph.get_commit(from_ref)
            if not commit:
                branch = self.branch_manager.get_branch(from_ref)
                if not branch:
                    raise ValueError(f"Invalid reference: {from_ref}")
                commit_hash = branch.commit_hash
            else:
                commit_hash = from_ref
        else:
            # Use current HEAD
            current_branch = self.branch_manager.get_current_branch()
            commit_hash_tmp = self.branch_manager.get_branch_commit(current_branch)
            if not commit_hash_tmp:
                raise ValueError("No commits in repository")
            commit_hash = commit_hash_tmp

        self.branch_manager.create_branch(name, commit_hash)

    def merge(self, source_branch: str, message: Optional[str] = None) -> Commit:
        """Merge another branch into current branch."""
        current_branch = self.branch_manager.get_current_branch()

        # Get branch commits
        current_commit_hash = self.branch_manager.get_branch_commit(current_branch)
        source_commit_hash = self.branch_manager.get_branch_commit(source_branch)

        if not current_commit_hash or not source_commit_hash:
            raise ValueError("Invalid branch state")

        # Check if already up to date
        if current_commit_hash == source_commit_hash:
            raise ValueError("Already up to date")

        # Find common ancestor
        common_ancestor = self.version_graph.get_common_ancestor(
            current_commit_hash, source_commit_hash
        )

        # Fast-forward if possible
        if common_ancestor == current_commit_hash:
            self.branch_manager.update_branch(current_branch, source_commit_hash)
            commit = self.version_graph.get_commit(source_commit_hash)
            if not commit:
                raise ValueError(f"Cannot find commit {source_commit_hash}")
            return commit

        # Perform three-way merge
        current_commit = self.version_graph.get_commit(current_commit_hash)
        source_commit = self.version_graph.get_commit(source_commit_hash)
        ancestor_commit = (
            self.version_graph.get_commit(common_ancestor) if common_ancestor else None
        )

        if not current_commit or not source_commit:
            raise ValueError("Cannot find required commits for merge")

        # Merge weights
        merged_weights = self._merge_weights(
            current_commit, source_commit, ancestor_commit
        )

        # Stage merged weights
        self.stage_weights(merged_weights)

        # Create merge commit
        merge_message = (
            message or f"Merge branch '{source_branch}' into {current_branch}"
        )

        return self.commit(message=merge_message, tags=["merge"])

    def _reconstruct_weight_from_storage(
        self, name: str, weight_hash: str, commit: Commit, store: "HDF5Store"
    ) -> Optional[WeightTensor]:
        """Unified method to reconstruct a weight from storage, handling deltas
        consistently."""
        # Check if this weight has a delta encoding in the commit
        if hasattr(commit, "delta_weights") and name in commit.delta_weights:
            delta_hash = commit.delta_weights[name]
            delta = store.load_delta(delta_hash)
            if delta:
                # Load reference weight
                ref_weight = store.load(delta.reference_hash)
                if ref_weight and self.deduplicator.delta_encoder:
                    # Reconstruct original weight from delta
                    return self.deduplicator.delta_encoder.decode_delta(
                        delta, ref_weight
                    )

        # Otherwise load the weight directly
        weight = store.load(weight_hash)
        return weight

    def get_weight(
        self, name: str, commit_ref: Optional[str] = None
    ) -> Optional[WeightTensor]:
        """Get a specific weight from a commit, reconstructing from delta if needed."""
        if commit_ref is None:
            # Use current HEAD - check if detached or on branch
            if self.current_branch_file.exists():
                with open(self.current_branch_file) as f:
                    head_content = f.read().strip()
                    if head_content.startswith("ref: refs/heads/"):
                        # On a branch
                        branch_name = head_content.replace("ref: refs/heads/", "")
                        commit_ref = self.branch_manager.get_branch_commit(branch_name)
                    else:
                        # Detached HEAD - commit hash directly
                        commit_ref = head_content
            else:
                commit_ref = None

        if not commit_ref:
            return None

        commit = self.version_graph.get_commit(commit_ref)
        if not commit or name not in commit.weight_hashes:
            return None

        weight_hash = commit.weight_hashes[name]

        with HDF5Store(str(self.weights_store_path)) as store_base:
            store = cast(HDF5Store, store_base)
            return self._reconstruct_weight_from_storage(
                name, weight_hash, commit, store
            )

    def get_all_weights(
        self, commit_ref: Optional[str] = None
    ) -> Dict[str, WeightTensor]:
        """Get all weights from a commit, reconstructing from deltas if needed."""
        if commit_ref is None:
            # Use current HEAD - check if detached or on branch
            if self.current_branch_file.exists():
                with open(self.current_branch_file) as f:
                    head_content = f.read().strip()
                    if head_content.startswith("ref: refs/heads/"):
                        # On a branch
                        branch_name = head_content.replace("ref: refs/heads/", "")
                        commit_ref = self.branch_manager.get_branch_commit(branch_name)
                    else:
                        # Detached HEAD - commit hash directly
                        commit_ref = head_content
            else:
                commit_ref = None

        if not commit_ref:
            return {}

        commit = self.version_graph.get_commit(commit_ref)
        if not commit:
            return {}

        weights = {}
        with HDF5Store(str(self.weights_store_path)) as store_base:
            store = cast(HDF5Store, store_base)
            for name, weight_hash in commit.weight_hashes.items():
                weight = self._reconstruct_weight_from_storage(
                    name, weight_hash, commit, store
                )
                if weight:
                    weights[name] = weight

        return weights

    def diff(self, from_ref: str, to_ref: Optional[str] = None) -> Dict[str, Any]:
        """Show differences between commits."""
        if to_ref is None:
            # Compare with current HEAD
            current_branch = self.branch_manager.get_current_branch()
            to_ref = self.branch_manager.get_branch_commit(current_branch)

        if not to_ref:
            raise ValueError("Cannot determine target commit for diff")

        # Resolve references to full commit hashes
        resolved_from = self._resolve_ref(from_ref)
        resolved_to = self._resolve_ref(to_ref)

        if not resolved_from or not resolved_to:
            raise ValueError("Invalid commit references")

        from_commit = self.version_graph.get_commit(resolved_from)
        to_commit = self.version_graph.get_commit(resolved_to)

        if not from_commit or not to_commit:
            raise ValueError("Invalid commit references")

        diff_info: Dict[str, Any] = {
            "added": list(to_commit.get_added_weights(from_commit)),
            "removed": list(to_commit.get_removed_weights(from_commit)),
            "modified": {},
            "summary": {},
        }

        # Check for modified weights
        for name in set(from_commit.weight_hashes.keys()) & set(
            to_commit.weight_hashes.keys()
        ):
            if from_commit.weight_hashes[name] != to_commit.weight_hashes[name]:
                diff_info["modified"][name] = {
                    "from_hash": from_commit.weight_hashes[name],
                    "to_hash": to_commit.weight_hashes[name],
                }
            elif (
                hasattr(to_commit, "delta_weights") and name in to_commit.delta_weights
            ):
                # Weight was delta-encoded, still considered modified
                diff_info["modified"][name] = {
                    "from_hash": from_commit.weight_hashes[name],
                    "to_hash": to_commit.weight_hashes[name],
                    "delta_encoded": True,
                    "delta_hash": to_commit.delta_weights[name],
                }

        # Calculate summary statistics
        diff_info["summary"] = {
            "total_added": len(diff_info["added"]),
            "total_removed": len(diff_info["removed"]),
            "total_modified": len(diff_info["modified"]),
        }

        return diff_info

    def log(self, max_commits: int = 10, branch: Optional[str] = None) -> List[Commit]:
        """Get commit history."""
        if branch is None:
            branch = self.branch_manager.get_current_branch()

        tip_hash = self.branch_manager.get_branch_commit(branch)
        if not tip_hash:
            return []

        history_hashes = self.version_graph.get_branch_history(
            tip_hash, max_depth=max_commits
        )

        commits = []
        for h in history_hashes:
            if h:
                commit = self.version_graph.get_commit(h)
                if commit:
                    commits.append(commit)
        return commits

    def tag_version(
        self,
        name: str,
        description: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        commit_ref: Optional[str] = None,
    ) -> Version:
        """Tag a commit as a named version."""
        if commit_ref is None:
            current_branch = self.branch_manager.get_current_branch()
            commit_ref = self.branch_manager.get_branch_commit(current_branch)

        if not commit_ref or not self.version_graph.get_commit(commit_ref):
            raise ValueError(f"Invalid commit: {commit_ref}")

        version_id = hashlib.sha256(f"{name}:{commit_ref}".encode()).hexdigest()[:8]

        version = Version(
            version_id=version_id,
            commit_hash=commit_ref,
            name=name,
            description=description,
            metrics=metrics,
        )

        self.version_graph.add_version(version)

        # Save version info
        version_file = self.coral_dir / "versions" / f"{version_id}.json"
        version_file.parent.mkdir(exist_ok=True)

        with open(version_file, "w") as f:
            json.dump(version.to_dict(), f, indent=2)

        return version

    def _calculate_deltas(
        self, weight_hashes: Dict[str, str], parent_commit: Commit
    ) -> Dict[str, str]:
        """Calculate delta encodings for changed weights."""
        deltas: Dict[str, str] = {}

        if (
            not self.deduplicator.enable_delta_encoding
            or not self.deduplicator.delta_encoder
        ):
            logger.debug("Delta encoding disabled, skipping delta calculation")
            return deltas

        with HDF5Store(str(self.weights_store_path)) as store_base:
            store = cast(HDF5Store, store_base)
            for name, current_hash in weight_hashes.items():
                # Check if weight changed from parent
                if (
                    name in parent_commit.weight_hashes
                    and parent_commit.weight_hashes[name] != current_hash
                ):
                    parent_hash = parent_commit.weight_hashes[name]

                    try:
                        # Load current and parent weights
                        current_weight = store.load(current_hash)
                        parent_weight = store.load(parent_hash)

                        if current_weight is None or parent_weight is None:
                            logger.warning(
                                f"Could not load weights for delta calculation: {name}"
                            )
                            continue

                        # Check if delta encoding is beneficial
                        if self.deduplicator.delta_encoder.can_encode_as_delta(
                            current_weight, parent_weight
                        ):
                            # Create delta encoding
                            delta = self.deduplicator.delta_encoder.encode_delta(
                                current_weight, parent_weight
                            )

                            # Generate delta hash
                            delta_hash = self.deduplicator._compute_delta_hash(delta)

                            # Store delta in the storage backend
                            store.store_delta(delta, delta_hash)

                            # Track delta for this weight
                            deltas[name] = delta_hash

                            logger.debug(
                                f"Created delta for {name}: "
                                f"{delta.compression_ratio:.2%} compression, "
                                f"hash {delta_hash}"
                            )
                        else:
                            logger.debug(
                                f"Delta encoding not beneficial for {name}, "
                                f"storing full weight"
                            )

                    except Exception as e:
                        logger.error(f"Failed to create delta for {name}: {e}")
                        # Continue without delta encoding for this weight

        return deltas

    def _merge_weights(
        self, current: Commit, source: Commit, ancestor: Optional[Commit]
    ) -> Dict[str, WeightTensor]:
        """Perform three-way merge of weights."""
        merged = {}

        all_names = set(current.weight_hashes.keys()) | set(source.weight_hashes.keys())
        if ancestor:
            all_names |= set(ancestor.weight_hashes.keys())

        with HDF5Store(str(self.weights_store_path)) as store_base:
            store = cast(HDF5Store, store_base)
            for name in all_names:
                # Get weights from each commit
                current_hash = current.weight_hashes.get(name)
                source_hash = source.weight_hashes.get(name)
                ancestor_hash = ancestor.weight_hashes.get(name) if ancestor else None

                # Simple merge strategy
                if current_hash == source_hash:
                    # No conflict
                    if current_hash:
                        weight = store.load(current_hash)
                        if weight:
                            merged[name] = weight
                elif ancestor_hash is None:
                    # Both added the weight - conflict
                    # For now, prefer current
                    if current_hash:
                        weight = store.load(current_hash)
                        if weight:
                            merged[name] = weight
                elif current_hash == ancestor_hash:
                    # Only source changed
                    if source_hash:
                        weight = store.load(source_hash)
                        if weight:
                            merged[name] = weight
                elif source_hash == ancestor_hash:
                    # Only current changed
                    if current_hash:
                        weight = store.load(current_hash)
                        if weight:
                            merged[name] = weight
                else:
                    # Both changed - conflict
                    # For now, prefer current
                    if current_hash:
                        weight = store.load(current_hash)
                        if weight:
                            merged[name] = weight

        return merged

    def gc(self) -> Dict[str, int]:
        """Garbage collect unreferenced weights."""
        # Find all referenced weight hashes
        referenced_hashes: Set[str] = set()

        for commit in self.version_graph.commits.values():
            referenced_hashes.update(commit.weight_hashes.values())

        # Clean up unreferenced weights
        cleaned = 0
        with HDF5Store(str(self.weights_store_path)) as store_base:
            store = cast(HDF5Store, store_base)
            all_hashes = set(store.list_weights())
            unreferenced = all_hashes - referenced_hashes

            for hash_val in unreferenced:
                store.delete(hash_val)
                cleaned += 1

        return {"cleaned_weights": cleaned, "remaining_weights": len(referenced_hashes)}

    def status(self) -> Dict[str, Any]:
        """Get repository status."""
        status = {
            "branch": self.current_branch,
            "staged": {},
            "clean": True,
        }

        # Check for staged weights
        staged_file = self.staging_dir / "staged.json"
        if staged_file.exists():
            with open(staged_file) as f:
                staged_data = json.load(f)

            # Handle both old flat format and new nested format
            if isinstance(staged_data, dict) and "weights" in staged_data:
                staged_weights = staged_data["weights"]
            else:
                staged_weights = staged_data

            status["staged"] = staged_weights
            status["clean"] = len(staged_weights) == 0

        return status

    def list_branches(self) -> List[Any]:
        """List all branches."""
        return self.branch_manager.list_branches()

    def list_versions(self) -> List[Version]:
        """List all versions."""
        return list(self.version_graph.versions.values())

    def get_version(self, name: str) -> Optional[Version]:
        """Get version by name."""
        for version in self.version_graph.versions.values():
            if version.name == name:
                return version
        return None

    def get_commit(self, commit_hash: str) -> Commit:
        """Get commit by hash."""
        commit = self.version_graph.get_commit(commit_hash)
        if commit is None:
            raise ValueError(f"Commit not found: {commit_hash}")
        return commit

    def _resolve_ref(self, ref: str) -> Optional[str]:
        """Resolve reference to commit hash."""
        # Handle HEAD
        if ref == "HEAD":
            return self.branch_manager.get_branch_commit(self.current_branch)

        # Try as branch name
        branch_commit = self.branch_manager.get_branch_commit(ref)
        if branch_commit:
            return branch_commit

        # Try as tag/version name
        version = self.get_version(ref)
        if version:
            return version.commit_hash

        # Try as commit hash (short or full)
        commit = self.version_graph.get_commit(ref)
        if commit:
            return ref

        # Try to find commit by short hash
        for commit_hash in self.version_graph.commits:
            if commit_hash.startswith(ref):
                return commit_hash

        return None
