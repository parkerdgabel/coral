"""Remote repository management for Coral.

Provides git-like remote operations for syncing weights across storage backends.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from coral.storage.weight_store import WeightStore

logger = logging.getLogger(__name__)


@dataclass
class RemoteConfig:
    """Configuration for a remote repository.

    Supports multiple backend types:
    - s3: AWS S3 or S3-compatible storage
    - gcs: Google Cloud Storage (future)
    - azure: Azure Blob Storage (future)
    - file: Local filesystem (for testing)
    """

    name: str  # e.g., "origin"
    url: str  # e.g., "s3://bucket/prefix" or "file:///path/to/repo"
    backend: str = "s3"  # s3, gcs, azure, file

    # Optional credentials (usually from environment)
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    region: Optional[str] = None
    endpoint_url: Optional[str] = None  # For MinIO, etc.

    # Sync settings
    auto_push: bool = False  # Auto-push on commit
    auto_pull: bool = False  # Auto-pull before operations

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        # Don't serialize credentials
        d.pop("access_key", None)
        d.pop("secret_key", None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RemoteConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_url(cls, name: str, url: str) -> "RemoteConfig":
        """Create from a URL string.

        Examples:
            s3://my-bucket/coral/models
            file:///home/user/backup
            minio://localhost:9000/bucket
        """
        if url.startswith("s3://"):
            # S3 URL: s3://bucket/prefix
            return cls(
                name=name,
                url=url,
                backend="s3",
            )
        elif url.startswith("file://"):
            return cls(name=name, url=url, backend="file")
        elif url.startswith("minio://"):
            # MinIO uses S3 protocol with custom endpoint
            path = url[8:]
            endpoint, rest = path.split("/", 1)
            return cls(
                name=name,
                url=f"s3://{rest}",
                backend="s3",
                endpoint_url=f"http://{endpoint}",
            )
        else:
            raise ValueError(f"Unsupported remote URL: {url}")


@dataclass
class SyncResult:
    """Result of a sync operation."""

    pushed_weights: int = 0
    pulled_weights: int = 0
    pushed_commits: int = 0
    pulled_commits: int = 0
    bytes_transferred: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


class Remote:
    """Manages a remote Coral repository.

    Example:
        >>> remote = Remote.from_config(RemoteConfig.from_url(
        ...     "origin", "s3://my-bucket/models"
        ... ))
        >>> remote.push(local_store, commits=["abc123"])
        >>> remote.pull(local_store)
    """

    def __init__(self, config: RemoteConfig, store: WeightStore):
        """Initialize remote with config and storage backend.

        Args:
            config: Remote configuration
            store: Storage backend for the remote
        """
        self.config = config
        self.store = store
        self._refs: Dict[str, str] = {}  # branch/tag -> commit hash

    @classmethod
    def from_config(cls, config: RemoteConfig) -> "Remote":
        """Create Remote from configuration.

        This will create the appropriate storage backend based on config.
        """
        if config.backend == "s3":
            from coral.storage.s3_store import S3Config, S3Store

            # Parse S3 URL
            url = config.url
            if url.startswith("s3://"):
                path = url[5:]
                parts = path.split("/", 1)
                bucket = parts[0]
                prefix = parts[1] if len(parts) > 1 else "coral/"
            else:
                raise ValueError(f"Invalid S3 URL: {url}")

            s3_config = S3Config(
                bucket=bucket,
                prefix=prefix if prefix.endswith("/") else prefix + "/",
                region=config.region,
                endpoint_url=config.endpoint_url,
                access_key=config.access_key,
                secret_key=config.secret_key,
            )
            store = S3Store(s3_config)
            return cls(config, store)

        elif config.backend == "file":
            from coral.storage.hdf5_store import HDF5Store

            # Parse file URL
            url = config.url
            if url.startswith("file://"):
                path = Path(url[7:])
            else:
                path = Path(url)

            store_path = path / "weights.h5"
            store_path.parent.mkdir(parents=True, exist_ok=True)
            store = HDF5Store(store_path)
            return cls(config, store)

        else:
            raise ValueError(f"Unsupported backend: {config.backend}")

    def push(
        self,
        local_store: WeightStore,
        weight_hashes: Optional[Set[str]] = None,
        force: bool = False,
    ) -> SyncResult:
        """Push weights from local store to remote.

        Args:
            local_store: Local storage backend
            weight_hashes: Specific hashes to push (None = all missing)
            force: Overwrite existing weights on remote

        Returns:
            SyncResult with statistics
        """
        result = SyncResult()

        # Determine what to push
        local_hashes = set(local_store.list_weights())
        remote_hashes = set(self.store.list_weights())

        if weight_hashes:
            to_push = weight_hashes & local_hashes
        else:
            to_push = local_hashes - remote_hashes

        if force:
            to_push = weight_hashes or local_hashes

        logger.info(f"Pushing {len(to_push)} weights to {self.config.name}")

        for hash_key in to_push:
            try:
                weight = local_store.load(hash_key)
                if weight:
                    self.store.store(weight, hash_key)
                    result.pushed_weights += 1
                    result.bytes_transferred += weight.nbytes
            except Exception as e:
                result.errors.append(f"Failed to push {hash_key}: {e}")
                logger.error(f"Push failed for {hash_key}: {e}")

        return result

    def pull(
        self,
        local_store: WeightStore,
        weight_hashes: Optional[Set[str]] = None,
        force: bool = False,
    ) -> SyncResult:
        """Pull weights from remote to local store.

        Args:
            local_store: Local storage backend
            weight_hashes: Specific hashes to pull (None = all missing)
            force: Overwrite existing local weights

        Returns:
            SyncResult with statistics
        """
        result = SyncResult()

        # Determine what to pull
        local_hashes = set(local_store.list_weights())
        remote_hashes = set(self.store.list_weights())

        if weight_hashes:
            to_pull = weight_hashes & remote_hashes
        else:
            to_pull = remote_hashes - local_hashes

        if force:
            to_pull = weight_hashes or remote_hashes

        logger.info(f"Pulling {len(to_pull)} weights from {self.config.name}")

        for hash_key in to_pull:
            try:
                weight = self.store.load(hash_key)
                if weight:
                    local_store.store(weight, hash_key)
                    result.pulled_weights += 1
                    result.bytes_transferred += weight.nbytes
            except Exception as e:
                result.errors.append(f"Failed to pull {hash_key}: {e}")
                logger.error(f"Pull failed for {hash_key}: {e}")

        return result

    def fetch_refs(self) -> Dict[str, str]:
        """Fetch remote references (branches, tags).

        Returns:
            Dict mapping ref names to commit hashes
        """
        # Try to load refs from remote
        try:
            # Check for refs file in remote
            # This would be stored as metadata in the remote
            pass
        except Exception:
            pass

        return self._refs

    def list_remote_weights(self) -> List[str]:
        """List all weight hashes on remote."""
        return self.store.list_weights()

    def get_remote_info(self) -> Dict[str, Any]:
        """Get information about the remote."""
        return {
            "name": self.config.name,
            "url": self.config.url,
            "backend": self.config.backend,
            "storage_info": self.store.get_storage_info(),
        }

    def close(self):
        """Close the remote connection."""
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RemoteManager:
    """Manages multiple remotes for a repository."""

    def __init__(self, config_path: Path):
        """Initialize remote manager.

        Args:
            config_path: Path to remotes config file (usually .coral/remotes.json)
        """
        self.config_path = config_path
        self.remotes: Dict[str, RemoteConfig] = {}
        self._load_config()

    def _load_config(self):
        """Load remotes from config file."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                data = json.load(f)
                for name, remote_data in data.get("remotes", {}).items():
                    self.remotes[name] = RemoteConfig.from_dict(remote_data)

    def _save_config(self):
        """Save remotes to config file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "remotes": {
                name: config.to_dict() for name, config in self.remotes.items()
            }
        }
        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)

    def add(self, config: RemoteConfig) -> None:
        """Add a remote."""
        self.remotes[config.name] = config
        self._save_config()

    def remove(self, name: str) -> bool:
        """Remove a remote."""
        if name in self.remotes:
            del self.remotes[name]
            self._save_config()
            return True
        return False

    def get(self, name: str) -> Optional[Remote]:
        """Get a Remote instance by name."""
        if name not in self.remotes:
            return None
        return Remote.from_config(self.remotes[name])

    def list(self) -> List[str]:
        """List all remote names."""
        return list(self.remotes.keys())
