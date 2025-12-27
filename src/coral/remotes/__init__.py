"""Remote storage and synchronization for Coral repositories.

This module provides git-like remote operations:
- Push local weights to remote storage (S3, GCS, etc.)
- Pull remote weights to local storage
- Clone remote repositories
- Track remote references
"""

from coral.remotes.remote import Remote, RemoteConfig
from coral.remotes.sync import sync_repositories

__all__ = ["Remote", "RemoteConfig", "sync_repositories"]
