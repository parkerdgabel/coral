from dataclasses import dataclass
from typing import Optional

import networkx as nx

from .commit import Commit


@dataclass
class Version:
    """Represents a version of the model."""

    version_id: str
    commit_hash: str
    name: str
    description: Optional[str] = None
    metrics: Optional[dict[str, float]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "version_id": self.version_id,
            "commit_hash": self.commit_hash,
            "name": self.name,
            "description": self.description,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Version":
        """Create from dictionary."""
        return cls(**data)


class VersionGraph:
    """Manages the version graph and commit relationships."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.commits: dict[str, Commit] = {}
        self.versions: dict[str, Version] = {}

    def add_commit(self, commit: Commit) -> None:
        """Add commit to the graph."""
        self.commits[commit.commit_hash] = commit
        self.graph.add_node(commit.commit_hash, commit=commit)

        for parent_hash in commit.parent_hashes:
            if parent_hash in self.graph:
                self.graph.add_edge(parent_hash, commit.commit_hash)

    def add_version(self, version: Version) -> None:
        """Add named version."""
        self.versions[version.version_id] = version

    def get_commit(self, commit_hash: str) -> Optional[Commit]:
        """Get commit by hash."""
        return self.commits.get(commit_hash)

    def get_version(self, version_id: str) -> Optional[Version]:
        """Get version by ID."""
        return self.versions.get(version_id)

    def get_commit_ancestors(self, commit_hash: str) -> list[str]:
        """Get all ancestors of a commit."""
        if commit_hash not in self.graph:
            return []

        return list(nx.ancestors(self.graph, commit_hash))

    def get_commit_descendants(self, commit_hash: str) -> list[str]:
        """Get all descendants of a commit."""
        if commit_hash not in self.graph:
            return []

        return list(nx.descendants(self.graph, commit_hash))

    def get_common_ancestor(
        self, commit1_hash: str, commit2_hash: str
    ) -> Optional[str]:
        """Find common ancestor of two commits."""
        if commit1_hash not in self.graph or commit2_hash not in self.graph:
            return None

        ancestors1 = set(self.get_commit_ancestors(commit1_hash)) | {commit1_hash}
        ancestors2 = set(self.get_commit_ancestors(commit2_hash)) | {commit2_hash}

        common = ancestors1 & ancestors2
        if not common:
            return None

        # Find the most recent common ancestor
        # Get topological order and reverse it manually for newer NetworkX versions
        topo_order = list(nx.topological_sort(self.graph))
        for commit_hash in reversed(topo_order):
            if commit_hash in common:
                return commit_hash

        return None

    def get_commit_path(self, from_hash: str, to_hash: str) -> Optional[list[str]]:
        """Get path between two commits if it exists."""
        if from_hash not in self.graph or to_hash not in self.graph:
            return None

        try:
            return nx.shortest_path(self.graph, from_hash, to_hash)
        except nx.NetworkXNoPath:
            return None

    def get_branch_history(
        self, tip_hash: str, max_depth: Optional[int] = None
    ) -> list[str]:
        """Get linear history from a commit backwards."""
        history = []
        current = tip_hash
        depth = 0

        while current and (max_depth is None or depth < max_depth):
            if current not in self.commits:
                break

            history.append(current)
            commit = self.commits[current]

            # Follow first parent for linear history
            if commit.parent_hashes:
                current = commit.parent_hashes[0]
            else:
                current = None

            depth += 1

        return history

    def get_weight_history(
        self, weight_name: str, from_commit: str
    ) -> list[tuple[str, str]]:
        """Get history of changes for a specific weight."""
        history = []

        for commit_hash in self.get_branch_history(from_commit):
            commit = self.commits[commit_hash]
            if weight_name in commit.weight_hashes:
                history.append((commit_hash, commit.weight_hashes[weight_name]))

        return history

    def find_commits_with_weight(self, weight_hash: str) -> list[str]:
        """Find all commits containing a specific weight hash."""
        commits = []

        for commit_hash, commit in self.commits.items():
            if weight_hash in commit.weight_hashes.values():
                commits.append(commit_hash)

        return commits

    def get_divergence_point(self, branch1_tip: str, branch2_tip: str) -> Optional[str]:
        """Find where two branches diverged."""
        return self.get_common_ancestor(branch1_tip, branch2_tip)
