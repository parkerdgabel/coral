import hashlib
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from ..core.deduplicator import Deduplicator
from ..core.weight_tensor import WeightTensor
from ..delta.delta_encoder import DeltaConfig
from ..storage.hdf5_store import HDF5Store
from ..utils.json_utils import dump_numpy
from .branch import BranchManager
from .commit import Commit, CommitMetadata
from .version import Version, VersionGraph

logger = logging.getLogger(__name__)


class MergeStrategy(Enum):
    """Strategy for resolving merge conflicts.

    For neural network weights, AVERAGE and WEIGHTED strategies can be useful
    for combining changes from different training runs or experiments.
    """

    OURS = "ours"  # Prefer weights from current branch
    THEIRS = "theirs"  # Prefer weights from source branch
    FAIL = "fail"  # Raise MergeConflictError on conflicts
    AVERAGE = "average"  # Average conflicting weights: (ours + theirs) / 2
    WEIGHTED = "weighted"  # Weighted average: alpha * ours + (1-alpha) * theirs


class MergeConflictError(Exception):
    """Raised when merge conflicts occur and strategy is FAIL."""

    def __init__(self, conflicts: List[str], message: str = "Merge conflicts detected"):
        self.conflicts = conflicts
        super().__init__(f"{message}: {', '.join(conflicts)}")


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

        # Configure delta encoding based on repository settings
        from ..delta.delta_encoder import DeltaType

        delta_config = DeltaConfig()
        if self.config.get("core", {}).get("delta_encoding", True):
            delta_type_str = self.config.get("core", {}).get(
                "delta_type", "float32_raw"
            )
            delta_config.delta_type = DeltaType(delta_type_str)

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
        (self.coral_dir / "staging").mkdir()

        # Create initial config
        config = {
            "user": {"name": "Anonymous", "email": "anonymous@example.com"},
            "core": {
                "compression": "gzip",
                "similarity_threshold": 0.98,
                "delta_encoding": True,
            },
        }

        with open(self.coral_dir / "config.json", "w") as f:
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

    def _load_config(self) -> Dict:
        """Load repository configuration."""
        config_file = self.coral_dir / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                return json.load(f)
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
            self.weights_store_path,
            compression=self.config.get("core", {}).get("compression", "gzip"),
        ) as store:
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
            dump_numpy(staging_info, f, indent=2)

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
            raise ValueError("No weights staged for commit")

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
            raise ValueError("No weights to commit")

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
                # Merge with staged deltas (staged deltas from deduplication have priority)
                delta_weights = {**commit_deltas, **staged_deltas}
            else:
                delta_weights = staged_deltas
        else:
            # Use staged deltas from deduplicator for root commits
            delta_weights = staged_deltas

        # Create commit hash (using 32 hex chars = 128 bits for collision resistance)
        commit_content = {
            "parent_hashes": parent_hashes,
            "weight_hashes": weight_hashes,
            "metadata": metadata.to_dict(),
            "delta_weights": delta_weights,
        }
        commit_hash = hashlib.sha256(
            json.dumps(commit_content, sort_keys=True).encode()
        ).hexdigest()[:32]

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

        # Check if target is a commit
        commit = self.version_graph.get_commit(target)
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
            commit_hash = self.branch_manager.get_branch_commit(current_branch)
            if not commit_hash:
                raise ValueError("No commits in repository")

        self.branch_manager.create_branch(name, commit_hash)

    def merge(
        self,
        source_branch: str,
        message: Optional[str] = None,
        strategy: MergeStrategy = MergeStrategy.OURS,
        merge_alpha: float = 0.5,
    ) -> Commit:
        """Merge another branch into current branch.

        Args:
            source_branch: Branch to merge from
            message: Optional merge commit message
            strategy: How to resolve conflicts:
                - OURS: Take weights from current branch
                - THEIRS: Take weights from source branch
                - FAIL: Raise MergeConflictError on conflicts
                - AVERAGE: Average the conflicting weights ((ours + theirs) / 2)
                - WEIGHTED: Weighted average (merge_alpha * ours + (1-merge_alpha) * theirs)
            merge_alpha: Weight for current branch when using WEIGHTED strategy (0-1).
                0.5 = equal weight, 0.7 = prefer current, 0.3 = prefer source

        Returns:
            The merge commit

        Raises:
            MergeConflictError: If strategy is FAIL and conflicts are detected
            ValueError: If branches are in invalid state
        """
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
            return self.version_graph.get_commit(source_commit_hash)

        # Perform three-way merge
        current_commit = self.version_graph.get_commit(current_commit_hash)
        source_commit = self.version_graph.get_commit(source_commit_hash)
        ancestor_commit = (
            self.version_graph.get_commit(common_ancestor) if common_ancestor else None
        )

        # Merge weights
        merged_weights = self._merge_weights(
            current_commit, source_commit, ancestor_commit, strategy, merge_alpha
        )

        # Stage merged weights
        self.stage_weights(merged_weights)

        # Create merge commit
        merge_message = (
            message or f"Merge branch '{source_branch}' into {current_branch}"
        )

        return self.commit(message=merge_message, tags=["merge"])

    def _reconstruct_weight_from_storage(
        self, name: str, weight_hash: str, commit: Commit, store: HDF5Store
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
            # Use current HEAD
            current_branch = self.branch_manager.get_current_branch()
            commit_ref = self.branch_manager.get_branch_commit(current_branch)

        commit = self.version_graph.get_commit(commit_ref)
        if not commit or name not in commit.weight_hashes:
            return None

        weight_hash = commit.weight_hashes[name]

        with HDF5Store(self.weights_store_path) as store:
            return self._reconstruct_weight_from_storage(
                name, weight_hash, commit, store
            )

    def get_all_weights(
        self, commit_ref: Optional[str] = None
    ) -> Dict[str, WeightTensor]:
        """Get all weights from a commit, reconstructing from deltas if needed."""
        if commit_ref is None:
            current_branch = self.branch_manager.get_current_branch()
            commit_ref = self.branch_manager.get_branch_commit(current_branch)

        commit = self.version_graph.get_commit(commit_ref)
        if not commit:
            return {}

        weights = {}
        with HDF5Store(self.weights_store_path) as store:
            for name, weight_hash in commit.weight_hashes.items():
                weight = self._reconstruct_weight_from_storage(
                    name, weight_hash, commit, store
                )
                if weight:
                    weights[name] = weight

        return weights

    def diff(self, from_ref: str, to_ref: Optional[str] = None) -> Dict[str, Dict]:
        """Show differences between commits."""
        if to_ref is None:
            # Compare with current HEAD
            current_branch = self.branch_manager.get_current_branch()
            to_ref = self.branch_manager.get_branch_commit(current_branch)

        from_commit = self.version_graph.get_commit(from_ref)
        to_commit = self.version_graph.get_commit(to_ref)

        if not from_commit or not to_commit:
            raise ValueError("Invalid commit references")

        diff_info = {
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

        return [self.version_graph.get_commit(h) for h in history_hashes if h]

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

        if not self.version_graph.get_commit(commit_ref):
            raise ValueError(f"Invalid commit: {commit_ref}")

        version_id = hashlib.sha256(f"{name}:{commit_ref}".encode()).hexdigest()[:16]

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
        deltas = {}

        if not self.deduplicator.enable_delta_encoding or not self.deduplicator.delta_encoder:
            logger.debug("Delta encoding disabled, skipping delta calculation")
            return deltas

        with HDF5Store(self.weights_store_path) as store:
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
                            logger.warning(f"Could not load weights for delta calculation: {name}")
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
                                f"Created delta for {name}: {delta.compression_ratio:.2%} compression, "
                                f"hash {delta_hash}"
                            )
                        else:
                            logger.debug(
                                f"Delta encoding not beneficial for {name}, storing full weight"
                            )
                    
                    except Exception as e:
                        logger.error(f"Failed to create delta for {name}: {e}")
                        # Continue without delta encoding for this weight

        return deltas

    def _merge_weights(
        self,
        current: Commit,
        source: Commit,
        ancestor: Optional[Commit],
        strategy: MergeStrategy = MergeStrategy.OURS,
        merge_alpha: float = 0.5,
    ) -> Dict[str, WeightTensor]:
        """Perform three-way merge of weights.

        Args:
            current: Current branch commit
            source: Source branch commit
            ancestor: Common ancestor commit (if any)
            strategy: How to resolve conflicts
            merge_alpha: Weight for current branch when using WEIGHTED strategy

        Returns:
            Merged weights dictionary

        Raises:
            MergeConflictError: If strategy is FAIL and conflicts are detected
        """
        import numpy as np

        from ..core.weight_tensor import WeightMetadata

        merged = {}
        conflicts = []

        all_names = set(current.weight_hashes.keys()) | set(source.weight_hashes.keys())
        if ancestor:
            all_names |= set(ancestor.weight_hashes.keys())

        with HDF5Store(self.weights_store_path) as store:
            for name in all_names:
                # Get weights from each commit
                current_hash = current.weight_hashes.get(name)
                source_hash = source.weight_hashes.get(name)
                ancestor_hash = ancestor.weight_hashes.get(name) if ancestor else None

                # Determine merge outcome
                if current_hash == source_hash:
                    # No conflict - both have same version
                    if current_hash:
                        merged[name] = store.load(current_hash)
                elif current_hash == ancestor_hash:
                    # Only source changed - take source
                    if source_hash:
                        merged[name] = store.load(source_hash)
                elif source_hash == ancestor_hash:
                    # Only current changed - take current
                    if current_hash:
                        merged[name] = store.load(current_hash)
                else:
                    # Conflict: both changed differently, or both added new weight
                    conflicts.append(name)

                    if strategy == MergeStrategy.FAIL:
                        # Will raise after collecting all conflicts
                        continue

                    # Load weights for conflict resolution
                    current_weight = store.load(current_hash) if current_hash else None
                    source_weight = store.load(source_hash) if source_hash else None

                    if strategy == MergeStrategy.THEIRS:
                        # Prefer source branch
                        merged[name] = source_weight or current_weight

                    elif strategy == MergeStrategy.AVERAGE:
                        # Average the weights: (ours + theirs) / 2
                        if current_weight and source_weight:
                            # Check compatible shapes
                            if current_weight.shape == source_weight.shape:
                                avg_data = (current_weight.data + source_weight.data) / 2
                                merged[name] = WeightTensor(
                                    data=avg_data.astype(current_weight.dtype),
                                    metadata=WeightMetadata(
                                        name=name,
                                        shape=current_weight.shape,
                                        dtype=current_weight.dtype,
                                    ),
                                )
                            else:
                                # Incompatible shapes, fall back to OURS
                                logger.warning(
                                    f"Cannot average {name}: shape mismatch "
                                    f"{current_weight.shape} vs {source_weight.shape}"
                                )
                                merged[name] = current_weight
                        else:
                            merged[name] = current_weight or source_weight

                    elif strategy == MergeStrategy.WEIGHTED:
                        # Weighted average: alpha * ours + (1-alpha) * theirs
                        if current_weight and source_weight:
                            if current_weight.shape == source_weight.shape:
                                weighted_data = (
                                    merge_alpha * current_weight.data
                                    + (1 - merge_alpha) * source_weight.data
                                )
                                merged[name] = WeightTensor(
                                    data=weighted_data.astype(current_weight.dtype),
                                    metadata=WeightMetadata(
                                        name=name,
                                        shape=current_weight.shape,
                                        dtype=current_weight.dtype,
                                    ),
                                )
                            else:
                                logger.warning(
                                    f"Cannot weighted-merge {name}: shape mismatch "
                                    f"{current_weight.shape} vs {source_weight.shape}"
                                )
                                merged[name] = current_weight
                        else:
                            # One is missing, use weighted fallback
                            if current_weight:
                                merged[name] = current_weight
                            else:
                                merged[name] = source_weight

                    else:  # MergeStrategy.OURS (default)
                        # Prefer current branch
                        merged[name] = current_weight or source_weight

        # Raise if strategy is FAIL and conflicts were detected
        if strategy == MergeStrategy.FAIL and conflicts:
            raise MergeConflictError(conflicts)

        if conflicts:
            logger.warning(
                f"Resolved {len(conflicts)} merge conflict(s) using strategy "
                f"'{strategy.value}': {', '.join(conflicts)}"
            )

        return merged

    def gc(self) -> Dict[str, int]:
        """Garbage collect unreferenced weights."""
        # Find all referenced weight hashes
        referenced_hashes = set()

        for commit in self.version_graph.commits.values():
            referenced_hashes.update(commit.weight_hashes.values())

        # Clean up unreferenced weights
        cleaned = 0
        with HDF5Store(self.weights_store_path) as store:
            all_hashes = set(store.list_weights())
            unreferenced = all_hashes - referenced_hashes

            for hash_val in unreferenced:
                store.delete(hash_val)
                cleaned += 1

        return {"cleaned_weights": cleaned, "remaining_weights": len(referenced_hashes)}
