import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class CommitMetadata:
    """Metadata for a commit."""

    author: str
    email: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "author": self.author,
            "email": self.email,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommitMetadata":
        """Create from dictionary."""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class Commit:
    """Represents a commit in the version control system."""

    def __init__(
        self,
        commit_hash: str,
        parent_hashes: List[str],
        weight_hashes: Dict[str, str],
        metadata: CommitMetadata,
        delta_weights: Optional[Dict[str, str]] = None,
    ):
        self.commit_hash = commit_hash
        self.parent_hashes = parent_hashes
        self.weight_hashes = weight_hashes  # name -> weight hash
        self.metadata = metadata
        self.delta_weights = delta_weights or {}  # name -> delta hash

    @property
    def is_merge_commit(self) -> bool:
        """Check if this is a merge commit."""
        return len(self.parent_hashes) > 1

    @property
    def is_root_commit(self) -> bool:
        """Check if this is a root commit."""
        return len(self.parent_hashes) == 0

    def get_changed_weights(self, parent_commit: Optional["Commit"]) -> Dict[str, str]:
        """Get weights that changed from parent commit."""
        if parent_commit is None:
            return self.weight_hashes

        changed = {}
        for name, hash_val in self.weight_hashes.items():
            if (
                name not in parent_commit.weight_hashes
                or parent_commit.weight_hashes[name] != hash_val
            ):
                changed[name] = hash_val

        return changed

    def get_added_weights(self, parent_commit: Optional["Commit"]) -> Set[str]:
        """Get weights added in this commit."""
        if parent_commit is None:
            return set(self.weight_hashes.keys())

        return set(self.weight_hashes.keys()) - set(parent_commit.weight_hashes.keys())

    def get_removed_weights(self, parent_commit: Optional["Commit"]) -> Set[str]:
        """Get weights removed in this commit."""
        if parent_commit is None:
            return set()

        return set(parent_commit.weight_hashes.keys()) - set(self.weight_hashes.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "commit_hash": self.commit_hash,
            "parent_hashes": self.parent_hashes,
            "weight_hashes": self.weight_hashes,
            "metadata": self.metadata.to_dict(),
            "delta_weights": self.delta_weights,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Commit":
        """Create from dictionary."""
        data = data.copy()
        data["metadata"] = CommitMetadata.from_dict(data["metadata"])
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save commit to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Commit":
        """Load commit from file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
