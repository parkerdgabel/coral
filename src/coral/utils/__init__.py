"""Utility functions for coral"""

from coral.utils.visualization import plot_deduplication_stats, plot_weight_distribution
from coral.utils.thread_safety import (
    FileLock,
    ThreadSafeDict,
    RepositoryLockManager,
    ThreadSafeCounter,
    atomic_write,
    with_retry,
    get_named_lock
)

__all__ = [
    "plot_weight_distribution", 
    "plot_deduplication_stats",
    "FileLock",
    "ThreadSafeDict", 
    "RepositoryLockManager",
    "ThreadSafeCounter",
    "atomic_write",
    "with_retry",
    "get_named_lock"
]
