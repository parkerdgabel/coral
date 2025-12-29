import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class Branch:
    """Represents a branch in the version control system."""

    name: str
    commit_hash: str
    parent_branch: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "commit_hash": self.commit_hash,
            "parent_branch": self.parent_branch,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Branch":
        """Create from dictionary."""
        return cls(**data)


class BranchManager:
    """Manages branches in the repository.

    This class is thread-safe. All public methods use an RLock to ensure
    safe concurrent access. RLock allows nested calls from the same thread.
    """

    def __init__(self, repo_path: Union[str, os.PathLike]):
        self.repo_path = Path(repo_path)
        self.refs_path = self.repo_path / ".coral" / "refs" / "heads"
        self.refs_path.mkdir(parents=True, exist_ok=True)
        self.current_branch_file = self.repo_path / ".coral" / "HEAD"
        self._lock = threading.RLock()

    def create_branch(
        self, name: str, commit_hash: str, parent_branch: Optional[str] = None
    ) -> Branch:
        """Create a new branch."""
        with self._lock:
            if self._branch_exists_unlocked(name):
                raise ValueError(f"Branch '{name}' already exists")

            branch = Branch(
                name=name, commit_hash=commit_hash, parent_branch=parent_branch
            )
            self._save_branch(branch)
            return branch

    def get_branch(self, name: str) -> Optional[Branch]:
        """Get branch by name."""
        with self._lock:
            return self._get_branch_unlocked(name)

    def _get_branch_unlocked(self, name: str) -> Optional[Branch]:
        """Get branch without acquiring lock (internal use only)."""
        branch_file = self.refs_path / name
        if not branch_file.exists():
            return None

        with open(branch_file) as f:
            data = json.load(f)
        return Branch.from_dict(data)

    def update_branch(self, name: str, commit_hash: str) -> None:
        """Update branch to point to new commit."""
        with self._lock:
            branch = self._get_branch_unlocked(name)
            if branch is None:
                raise ValueError(f"Branch '{name}' does not exist")

            branch.commit_hash = commit_hash
            self._save_branch(branch)

    def delete_branch(self, name: str) -> None:
        """Delete a branch."""
        with self._lock:
            if name == self._get_current_branch_unlocked():
                raise ValueError("Cannot delete current branch")

            branch_file = self.refs_path / name
            if branch_file.exists():
                branch_file.unlink()

    def list_branches(self) -> list[Branch]:
        """List all branches."""
        with self._lock:
            branches = []
            for branch_file in self.refs_path.iterdir():
                if branch_file.is_file():
                    branch = self._get_branch_unlocked(branch_file.name)
                    if branch:
                        branches.append(branch)
            return branches

    def branch_exists(self, name: str) -> bool:
        """Check if branch exists."""
        with self._lock:
            return self._branch_exists_unlocked(name)

    def _branch_exists_unlocked(self, name: str) -> bool:
        """Check if branch exists without acquiring lock (internal use only)."""
        return (self.refs_path / name).exists()

    def get_current_branch(self) -> str:
        """Get current branch name."""
        with self._lock:
            return self._get_current_branch_unlocked()

    def _get_current_branch_unlocked(self) -> str:
        """Get current branch without acquiring lock (internal use only)."""
        if not self.current_branch_file.exists():
            return "main"

        with open(self.current_branch_file) as f:
            return f.read().strip().replace("ref: refs/heads/", "")

    def set_current_branch(self, name: str) -> None:
        """Set current branch."""
        with self._lock:
            if not self._branch_exists_unlocked(name):
                raise ValueError(f"Branch '{name}' does not exist")

            with open(self.current_branch_file, "w") as f:
                f.write(f"ref: refs/heads/{name}")

    def get_branch_commit(self, name: str) -> Optional[str]:
        """Get commit hash for branch."""
        with self._lock:
            branch = self._get_branch_unlocked(name)
            if not branch:
                return None
            # Return None for empty commit hash (new repository)
            return branch.commit_hash if branch.commit_hash else None

    def _save_branch(self, branch: Branch) -> None:
        """Save branch to file (must be called with lock held)."""
        branch_file = self.refs_path / branch.name
        with open(branch_file, "w") as f:
            json.dump(branch.to_dict(), f, indent=2)
