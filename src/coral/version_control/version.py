from __future__ import annotations

import threading
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
    def from_dict(cls, data: dict) -> Version:
        """Create from dictionary."""
        return cls(**data)


class VersionGraph:
    """Manages the version graph and commit relationships.

    This class is thread-safe. All public methods use an RLock to ensure
    safe concurrent access. RLock allows nested calls from the same thread.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.commits: dict[str, Commit] = {}
        self.versions: dict[str, Version] = {}
        self._lock = threading.RLock()

    def add_commit(self, commit: Commit) -> None:
        """Add commit to the graph."""
        with self._lock:
            self.commits[commit.commit_hash] = commit
            self.graph.add_node(commit.commit_hash, commit=commit)

            for parent_hash in commit.parent_hashes:
                if parent_hash in self.graph:
                    self.graph.add_edge(parent_hash, commit.commit_hash)

    def add_version(self, version: Version) -> None:
        """Add named version."""
        with self._lock:
            self.versions[version.version_id] = version

    def get_commit(self, commit_hash: str) -> Optional[Commit]:
        """Get commit by hash."""
        with self._lock:
            return self.commits.get(commit_hash)

    def get_version(self, version_id: str) -> Optional[Version]:
        """Get version by ID."""
        with self._lock:
            return self.versions.get(version_id)

    def get_commit_ancestors(self, commit_hash: str) -> list[str]:
        """Get all ancestors of a commit."""
        with self._lock:
            if commit_hash not in self.graph:
                return []
            return list(nx.ancestors(self.graph, commit_hash))

    def get_commit_descendants(self, commit_hash: str) -> list[str]:
        """Get all descendants of a commit."""
        with self._lock:
            if commit_hash not in self.graph:
                return []
            return list(nx.descendants(self.graph, commit_hash))

    def get_common_ancestor(
        self, commit1_hash: str, commit2_hash: str
    ) -> Optional[str]:
        """Find common ancestor of two commits."""
        with self._lock:
            if commit1_hash not in self.graph or commit2_hash not in self.graph:
                return None

            ancestors1 = set(self._get_commit_ancestors_unlocked(commit1_hash)) | {
                commit1_hash
            }
            ancestors2 = set(self._get_commit_ancestors_unlocked(commit2_hash)) | {
                commit2_hash
            }

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

    def _get_commit_ancestors_unlocked(self, commit_hash: str) -> list[str]:
        """Get ancestors without acquiring lock (internal use only)."""
        if commit_hash not in self.graph:
            return []
        return list(nx.ancestors(self.graph, commit_hash))

    def get_commit_path(self, from_hash: str, to_hash: str) -> Optional[list[str]]:
        """Get path between two commits if it exists."""
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            history = []

            for commit_hash in self._get_branch_history_unlocked(from_commit):
                commit = self.commits[commit_hash]
                if weight_name in commit.weight_hashes:
                    history.append((commit_hash, commit.weight_hashes[weight_name]))

            return history

    def _get_branch_history_unlocked(
        self, tip_hash: str, max_depth: Optional[int] = None
    ) -> list[str]:
        """Get branch history without acquiring lock (internal use only)."""
        history = []
        current = tip_hash
        depth = 0

        while current and (max_depth is None or depth < max_depth):
            if current not in self.commits:
                break

            history.append(current)
            commit = self.commits[current]

            if commit.parent_hashes:
                current = commit.parent_hashes[0]
            else:
                current = None

            depth += 1

        return history

    def find_commits_with_weight(self, weight_hash: str) -> list[str]:
        """Find all commits containing a specific weight hash."""
        with self._lock:
            commits = []

            for commit_hash, commit in self.commits.items():
                if weight_hash in commit.weight_hashes.values():
                    commits.append(commit_hash)

            return commits

    def get_divergence_point(self, branch1_tip: str, branch2_tip: str) -> Optional[str]:
        """Find where two branches diverged."""
        # get_common_ancestor already acquires the lock
        return self.get_common_ancestor(branch1_tip, branch2_tip)
