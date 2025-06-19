import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Branch:
    """Represents a branch in the version control system."""

    name: str
    commit_hash: str
    parent_branch: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "commit_hash": self.commit_hash,
            "parent_branch": self.parent_branch,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Branch":
        """Create from dictionary."""
        return cls(**data)


class BranchManager:
    """Manages branches in the repository."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.refs_path = repo_path / ".coral" / "refs" / "heads"
        self.refs_path.mkdir(parents=True, exist_ok=True)
        self.current_branch_file = repo_path / ".coral" / "HEAD"

    def create_branch(
        self, name: str, commit_hash: str, parent_branch: Optional[str] = None
    ) -> Branch:
        """Create a new branch."""
        if self.branch_exists(name):
            raise ValueError(f"Branch '{name}' already exists")

        branch = Branch(name=name, commit_hash=commit_hash, parent_branch=parent_branch)
        self._save_branch(branch)
        return branch

    def get_branch(self, name: str) -> Optional[Branch]:
        """Get branch by name."""
        branch_file = self.refs_path / name
        if not branch_file.exists():
            return None

        with open(branch_file) as f:
            data = json.load(f)
        return Branch.from_dict(data)

    def update_branch(self, name: str, commit_hash: str) -> None:
        """Update branch to point to new commit."""
        branch = self.get_branch(name)
        if branch is None:
            raise ValueError(f"Branch '{name}' does not exist")

        branch.commit_hash = commit_hash
        self._save_branch(branch)

    def delete_branch(self, name: str) -> None:
        """Delete a branch."""
        if name == self.get_current_branch():
            raise ValueError("Cannot delete current branch")

        branch_file = self.refs_path / name
        if branch_file.exists():
            branch_file.unlink()

    def list_branches(self) -> List[Branch]:
        """List all branches."""
        branches = []
        for branch_file in self.refs_path.iterdir():
            if branch_file.is_file():
                branch = self.get_branch(branch_file.name)
                if branch:
                    branches.append(branch)
        return branches

    def branch_exists(self, name: str) -> bool:
        """Check if branch exists."""
        return (self.refs_path / name).exists()

    def get_current_branch(self) -> str:
        """Get current branch name."""
        if not self.current_branch_file.exists():
            return "main"

        with open(self.current_branch_file) as f:
            return f.read().strip().replace("ref: refs/heads/", "")

    def set_current_branch(self, name: str) -> None:
        """Set current branch."""
        if not self.branch_exists(name):
            raise ValueError(f"Branch '{name}' does not exist")

        with open(self.current_branch_file, "w") as f:
            f.write(f"ref: refs/heads/{name}")

    def get_branch_commit(self, name: str) -> Optional[str]:
        """Get commit hash for branch."""
        branch = self.get_branch(name)
        if not branch:
            return None
        # Return None for empty commit hash (new repository)
        return branch.commit_hash if branch.commit_hash else None

    def _save_branch(self, branch: Branch) -> None:
        """Save branch to file."""
        branch_file = self.refs_path / branch.name
        with open(branch_file, "w") as f:
            json.dump(branch.to_dict(), f, indent=2)
