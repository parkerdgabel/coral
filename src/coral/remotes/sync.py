"""Repository synchronization utilities."""

import logging
from typing import Any, Dict

from coral.remotes.remote import Remote
from coral.storage.weight_store import WeightStore

logger = logging.getLogger(__name__)


def sync_repositories(
    local_store: WeightStore,
    remote: Remote,
    direction: str = "both",
    force: bool = False,
) -> Dict[str, Any]:
    """Synchronize local and remote repositories.

    Args:
        local_store: Local storage backend
        remote: Remote to sync with
        direction: "push", "pull", or "both"
        force: Overwrite existing weights

    Returns:
        Dict with sync statistics
    """
    stats = {
        "push": None,
        "pull": None,
    }

    if direction in ("push", "both"):
        result = remote.push(local_store, force=force)
        stats["push"] = {
            "weights": result.pushed_weights,
            "bytes": result.bytes_transferred,
            "errors": result.errors,
        }
        logger.info(
            f"Pushed {result.pushed_weights} weights "
            f"({result.bytes_transferred / 1024 / 1024:.1f} MB)"
        )

    if direction in ("pull", "both"):
        result = remote.pull(local_store, force=force)
        stats["pull"] = {
            "weights": result.pulled_weights,
            "bytes": result.bytes_transferred,
            "errors": result.errors,
        }
        logger.info(
            f"Pulled {result.pulled_weights} weights "
            f"({result.bytes_transferred / 1024 / 1024:.1f} MB)"
        )

    return stats
