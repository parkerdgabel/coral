"""Tests for remote repository management."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.remotes.remote import Remote, RemoteConfig, RemoteManager, SyncResult
from coral.remotes.sync import sync_repositories
from coral.storage.hdf5_store import HDF5Store


class TestRemoteConfig:
    """Tests for RemoteConfig dataclass."""

    def test_creation(self):
        """Test basic RemoteConfig creation."""
        config = RemoteConfig(
            name="origin",
            url="s3://bucket/prefix",
            backend="s3",
        )
        assert config.name == "origin"
        assert config.url == "s3://bucket/prefix"
        assert config.backend == "s3"
        assert config.access_key is None
        assert config.auto_push is False

    def test_to_dict_excludes_credentials(self):
        """Test that to_dict excludes sensitive credentials."""
        config = RemoteConfig(
            name="origin",
            url="s3://bucket/prefix",
            backend="s3",
            access_key="secret_key_123",
            secret_key="secret_value_456",
        )
        d = config.to_dict()
        assert "access_key" not in d
        assert "secret_key" not in d
        assert d["name"] == "origin"
        assert d["url"] == "s3://bucket/prefix"

    def test_from_dict(self):
        """Test creating RemoteConfig from dictionary."""
        data = {
            "name": "origin",
            "url": "file:///tmp/backup",
            "backend": "file",
            "auto_push": True,
        }
        config = RemoteConfig.from_dict(data)
        assert config.name == "origin"
        assert config.url == "file:///tmp/backup"
        assert config.backend == "file"
        assert config.auto_push is True

    def test_from_url_s3(self):
        """Test creating RemoteConfig from S3 URL."""
        config = RemoteConfig.from_url("origin", "s3://my-bucket/coral/models")
        assert config.name == "origin"
        assert config.url == "s3://my-bucket/coral/models"
        assert config.backend == "s3"

    def test_from_url_file(self):
        """Test creating RemoteConfig from file URL."""
        config = RemoteConfig.from_url("backup", "file:///home/user/backup")
        assert config.name == "backup"
        assert config.url == "file:///home/user/backup"
        assert config.backend == "file"

    def test_from_url_minio(self):
        """Test creating RemoteConfig from MinIO URL."""
        config = RemoteConfig.from_url("minio", "minio://localhost:9000/bucket/prefix")
        assert config.name == "minio"
        assert config.backend == "s3"
        assert config.endpoint_url == "http://localhost:9000"
        assert "bucket" in config.url

    def test_from_url_invalid(self):
        """Test that invalid URLs raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported remote URL"):
            RemoteConfig.from_url("bad", "http://example.com")


class TestSyncResult:
    """Tests for SyncResult dataclass."""

    def test_success_when_no_errors(self):
        """Test success property when no errors."""
        result = SyncResult(pushed_weights=5, pulled_weights=3)
        assert result.success is True

    def test_failure_when_errors(self):
        """Test success property when errors exist."""
        result = SyncResult(
            pushed_weights=5,
            errors=["Failed to push weight1"],
        )
        assert result.success is False

    def test_default_values(self):
        """Test default values are initialized correctly."""
        result = SyncResult()
        assert result.pushed_weights == 0
        assert result.pulled_weights == 0
        assert result.pushed_commits == 0
        assert result.pulled_commits == 0
        assert result.bytes_transferred == 0
        assert result.errors == []


class TestRemote:
    """Tests for Remote class with file backend."""

    @pytest.fixture
    def local_store(self, tmp_path):
        """Create a local HDF5 store with test weights."""
        store_path = tmp_path / "local" / "weights.h5"
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store = HDF5Store(str(store_path))

        # Add some test weights
        for i in range(3):
            weight = WeightTensor(
                data=np.random.randn(10, 10).astype(np.float32),
                metadata=WeightMetadata(
                    name=f"layer{i}.weight",
                    shape=(10, 10),
                    dtype=np.float32,
                ),
            )
            store.store(weight)

        yield store
        store.close()

    @pytest.fixture
    def remote_store(self, tmp_path):
        """Create an empty remote store."""
        remote_path = tmp_path / "remote"
        remote_path.mkdir(parents=True, exist_ok=True)
        config = RemoteConfig.from_url("origin", f"file://{remote_path}")
        return Remote.from_config(config)

    def test_from_config_file_backend(self, tmp_path):
        """Test creating Remote with file backend."""
        remote_path = tmp_path / "remote"
        config = RemoteConfig.from_url("origin", f"file://{remote_path}")
        remote = Remote.from_config(config)

        assert remote.config.name == "origin"
        assert remote.config.backend == "file"
        assert isinstance(remote.store, HDF5Store)
        remote.close()

    def test_from_config_invalid_backend(self, tmp_path):
        """Test that invalid backend raises ValueError."""
        config = RemoteConfig(
            name="bad",
            url="foo://bar",
            backend="unsupported",
        )
        with pytest.raises(ValueError, match="Unsupported backend"):
            Remote.from_config(config)

    def test_push_weights(self, local_store, remote_store):
        """Test pushing weights to remote."""
        result = remote_store.push(local_store)

        assert result.success is True
        assert result.pushed_weights == 3
        assert result.bytes_transferred > 0

        # Verify weights are on remote
        remote_weights = remote_store.list_remote_weights()
        assert len(remote_weights) == 3

        remote_store.close()

    def test_push_specific_weights(self, local_store, remote_store):
        """Test pushing specific weights only."""
        local_hashes = set(local_store.list_weights())
        first_hash = list(local_hashes)[0]

        result = remote_store.push(local_store, weight_hashes={first_hash})

        assert result.success is True
        assert result.pushed_weights == 1

        remote_store.close()

    def test_pull_weights(self, local_store, remote_store):
        """Test pulling weights from remote."""
        # First push to remote
        remote_store.push(local_store)

        # Create a new empty local store
        with tempfile.TemporaryDirectory() as tmp_dir:
            new_store_path = Path(tmp_dir) / "new_local" / "weights.h5"
            new_store_path.parent.mkdir(parents=True, exist_ok=True)
            new_store = HDF5Store(str(new_store_path))

            # Pull from remote
            result = remote_store.pull(new_store)

            assert result.success is True
            assert result.pulled_weights == 3
            assert result.bytes_transferred > 0

            # Verify weights are in new store
            new_weights = new_store.list_weights()
            assert len(new_weights) == 3

            new_store.close()

        remote_store.close()

    def test_push_only_missing(self, local_store, remote_store):
        """Test that push only syncs missing weights."""
        # Push once
        result1 = remote_store.push(local_store)
        assert result1.pushed_weights == 3

        # Push again - should push nothing
        result2 = remote_store.push(local_store)
        assert result2.pushed_weights == 0

        remote_store.close()

    def test_push_force(self, local_store, remote_store):
        """Test force push overwrites existing weights."""
        # Push once
        remote_store.push(local_store)

        # Force push - should push all again
        result = remote_store.push(local_store, force=True)
        assert result.pushed_weights == 3

        remote_store.close()

    def test_get_remote_info(self, remote_store):
        """Test getting remote info."""
        info = remote_store.get_remote_info()

        assert info["name"] == "origin"
        assert info["backend"] == "file"
        assert "storage_info" in info

        remote_store.close()

    def test_context_manager(self, tmp_path):
        """Test Remote as context manager."""
        remote_path = tmp_path / "remote"
        config = RemoteConfig.from_url("origin", f"file://{remote_path}")

        with Remote.from_config(config) as remote:
            assert remote.config.name == "origin"


class TestRemoteManager:
    """Tests for RemoteManager class."""

    @pytest.fixture
    def config_path(self, tmp_path):
        """Create a temporary config path."""
        return tmp_path / ".coral" / "remotes.json"

    def test_add_remote(self, config_path):
        """Test adding a remote."""
        manager = RemoteManager(config_path)
        config = RemoteConfig.from_url("origin", "file:///tmp/backup")

        manager.add(config)

        assert "origin" in manager.remotes
        assert manager.remotes["origin"].url == "file:///tmp/backup"
        # Verify saved to file
        assert config_path.exists()

    def test_remove_remote(self, config_path):
        """Test removing a remote."""
        manager = RemoteManager(config_path)
        config = RemoteConfig.from_url("origin", "file:///tmp/backup")
        manager.add(config)

        result = manager.remove("origin")

        assert result is True
        assert "origin" not in manager.remotes

    def test_remove_nonexistent(self, config_path):
        """Test removing a non-existent remote returns False."""
        manager = RemoteManager(config_path)

        result = manager.remove("nonexistent")

        assert result is False

    def test_list_remotes(self, config_path):
        """Test listing remotes."""
        manager = RemoteManager(config_path)
        manager.add(RemoteConfig.from_url("origin", "file:///tmp/origin"))
        manager.add(RemoteConfig.from_url("backup", "file:///tmp/backup"))

        names = manager.list()

        assert "origin" in names
        assert "backup" in names
        assert len(names) == 2

    def test_get_remote(self, config_path, tmp_path):
        """Test getting a Remote instance."""
        manager = RemoteManager(config_path)
        remote_path = tmp_path / "remote"
        manager.add(RemoteConfig.from_url("origin", f"file://{remote_path}"))

        remote = manager.get("origin")

        assert remote is not None
        assert remote.config.name == "origin"
        remote.close()

    def test_get_nonexistent(self, config_path):
        """Test getting a non-existent remote returns None."""
        manager = RemoteManager(config_path)

        remote = manager.get("nonexistent")

        assert remote is None

    def test_persistence(self, config_path):
        """Test that remotes are persisted across manager instances."""
        # Add remote with first manager
        manager1 = RemoteManager(config_path)
        manager1.add(RemoteConfig.from_url("origin", "file:///tmp/backup"))

        # Create new manager and verify it loads the remote
        manager2 = RemoteManager(config_path)

        assert "origin" in manager2.remotes
        assert manager2.remotes["origin"].url == "file:///tmp/backup"


class TestSyncRepositories:
    """Tests for sync_repositories function."""

    @pytest.fixture
    def stores(self, tmp_path):
        """Create local and remote stores with test data."""
        local_path = tmp_path / "local" / "weights.h5"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_store = HDF5Store(str(local_path))

        # Add weights to local
        for i in range(2):
            weight = WeightTensor(
                data=np.random.randn(5, 5).astype(np.float32),
                metadata=WeightMetadata(
                    name=f"local_layer{i}.weight",
                    shape=(5, 5),
                    dtype=np.float32,
                ),
            )
            local_store.store(weight)

        # Create remote with different weights
        remote_path = tmp_path / "remote"
        remote_path.mkdir(parents=True, exist_ok=True)
        config = RemoteConfig.from_url("origin", f"file://{remote_path}")
        remote = Remote.from_config(config)

        # Add different weights to remote
        for i in range(2):
            weight = WeightTensor(
                data=np.random.randn(5, 5).astype(np.float32),
                metadata=WeightMetadata(
                    name=f"remote_layer{i}.weight",
                    shape=(5, 5),
                    dtype=np.float32,
                ),
            )
            remote.store.store(weight)

        yield local_store, remote

        local_store.close()
        remote.close()

    def test_sync_push(self, stores):
        """Test push-only sync."""
        local_store, remote = stores

        stats = sync_repositories(local_store, remote, direction="push")

        assert stats["push"] is not None
        assert stats["push"]["weights"] == 2
        assert stats["pull"] is None

    def test_sync_pull(self, stores):
        """Test pull-only sync."""
        local_store, remote = stores

        stats = sync_repositories(local_store, remote, direction="pull")

        assert stats["push"] is None
        assert stats["pull"] is not None
        assert stats["pull"]["weights"] == 2

    def test_sync_both(self, stores):
        """Test bidirectional sync."""
        local_store, remote = stores

        stats = sync_repositories(local_store, remote, direction="both")

        assert stats["push"] is not None
        assert stats["push"]["weights"] == 2
        assert stats["pull"] is not None
        assert stats["pull"]["weights"] == 2

        # Verify both stores now have all 4 weights
        assert len(local_store.list_weights()) == 4
        assert len(remote.store.list_weights()) == 4

    def test_sync_force(self, stores):
        """Test force sync."""
        local_store, remote = stores

        # First sync
        sync_repositories(local_store, remote, direction="both")

        # Force sync again - should overwrite
        stats = sync_repositories(local_store, remote, direction="both", force=True)

        assert stats["push"]["weights"] == 4  # All weights pushed
        assert stats["pull"]["weights"] == 4  # All weights pulled
