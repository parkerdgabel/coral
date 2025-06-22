"""Thread safety utilities for Coral ML.

This module provides thread-safe operations and locking mechanisms
to ensure data integrity during concurrent operations.
"""

import os
import threading
import time
import hashlib
import tempfile
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, Any, Callable
import logging
import sys

# Platform-specific imports
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    # Windows doesn't have fcntl
    HAS_FCNTL = False
    try:
        import msvcrt
    except ImportError:
        # Not on Windows either, use file-based locking only
        msvcrt = None

logger = logging.getLogger(__name__)


class FileLock:
    """File-based locking mechanism for cross-process synchronization."""
    
    def __init__(self, path: Path, timeout: float = 30.0):
        """Initialize file lock.
        
        Args:
            path: Path to the file to lock
            timeout: Maximum time to wait for lock acquisition
        """
        self.path = Path(path)
        self.lockfile = Path(str(path) + ".lock")
        self.timeout = timeout
        self.lock_fd = None
        self.acquired = False
        
    def acquire(self, blocking: bool = True) -> bool:
        """Acquire the file lock.
        
        Args:
            blocking: Whether to block until lock is acquired
            
        Returns:
            True if lock was acquired, False otherwise
        """
        if self.acquired:
            return True
            
        start_time = time.time()
        
        # Ensure lock directory exists
        self.lockfile.parent.mkdir(parents=True, exist_ok=True)
        
        while True:
            try:
                # Try to create lock file exclusively
                self.lock_fd = os.open(
                    str(self.lockfile),
                    os.O_CREAT | os.O_EXCL | os.O_RDWR
                )
                
                # Write PID to lock file for debugging
                os.write(self.lock_fd, str(os.getpid()).encode())
                
                # Try to acquire exclusive lock
                if HAS_FCNTL:
                    fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                elif msvcrt is not None:
                    # Windows file locking
                    hfile = msvcrt.get_osfhandle(self.lock_fd)
                    msvcrt.locking(hfile, msvcrt.LK_NBLCK, 1)
                # If neither is available, rely on exclusive file creation only
                
                self.acquired = True
                return True
                
            except OSError as e:
                if self.lock_fd is not None:
                    os.close(self.lock_fd)
                    self.lock_fd = None
                    
                # Check if lock file exists and is stale
                if self.lockfile.exists():
                    try:
                        # Check if lock is stale (older than timeout)
                        lock_age = time.time() - self.lockfile.stat().st_mtime
                        if lock_age > self.timeout:
                            logger.warning(f"Removing stale lock: {self.lockfile}")
                            self.lockfile.unlink(missing_ok=True)
                            continue
                    except Exception:
                        pass
                
                if not blocking:
                    return False
                    
                # Check timeout
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"Failed to acquire lock on {self.path} after {self.timeout}s")
                    
                # Wait a bit before retrying
                time.sleep(0.1)
    
    def release(self):
        """Release the file lock."""
        if not self.acquired:
            return
            
        try:
            if self.lock_fd is not None:
                if HAS_FCNTL:
                    fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                elif msvcrt is not None:
                    # Windows unlock
                    hfile = msvcrt.get_osfhandle(self.lock_fd)
                    msvcrt.locking(hfile, msvcrt.LK_UNLCK, 1)
                os.close(self.lock_fd)
                self.lock_fd = None
                
            self.lockfile.unlink(missing_ok=True)
            self.acquired = False
            
        except Exception as e:
            logger.error(f"Error releasing lock {self.lockfile}: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


class ThreadSafeDict:
    """Thread-safe dictionary implementation."""
    
    def __init__(self):
        self._dict = {}
        self._lock = threading.RLock()
        
    def get(self, key: Any, default: Any = None) -> Any:
        """Thread-safe get operation."""
        with self._lock:
            return self._dict.get(key, default)
            
    def set(self, key: Any, value: Any):
        """Thread-safe set operation."""
        with self._lock:
            self._dict[key] = value
            
    def pop(self, key: Any, default: Any = None) -> Any:
        """Thread-safe pop operation."""
        with self._lock:
            return self._dict.pop(key, default)
            
    def clear(self):
        """Thread-safe clear operation."""
        with self._lock:
            self._dict.clear()
            
    def items(self):
        """Thread-safe items iteration."""
        with self._lock:
            return list(self._dict.items())
            
    def __contains__(self, key: Any) -> bool:
        """Thread-safe containment check."""
        with self._lock:
            return key in self._dict
            
    def __len__(self) -> int:
        """Thread-safe length."""
        with self._lock:
            return len(self._dict)


class RepositoryLockManager:
    """Manages locks for repository operations."""
    
    def __init__(self, repo_path: Path):
        """Initialize lock manager.
        
        Args:
            repo_path: Path to repository root
        """
        self.repo_path = Path(repo_path)
        self.coral_dir = self.repo_path / ".coral"
        self.locks_dir = self.coral_dir / "locks"
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread-local storage for nested locking
        self._local = threading.local()
        
        # Global repository lock for critical operations
        self._repo_lock = threading.RLock()
        
    @contextmanager
    def staging_lock(self):
        """Lock for staging directory operations."""
        lock_path = self.locks_dir / "staging.lock"
        with FileLock(lock_path):
            yield
            
    @contextmanager
    def hdf5_lock(self, file_path: Path):
        """Lock for HDF5 file operations.
        
        Args:
            file_path: Path to HDF5 file
        """
        # Create a unique lock name based on file path
        lock_name = hashlib.sha256(str(file_path).encode()).hexdigest()[:16]
        lock_path = self.locks_dir / f"hdf5_{lock_name}.lock"
        
        with FileLock(lock_path):
            yield
            
    @contextmanager
    def commit_lock(self):
        """Lock for commit operations."""
        lock_path = self.locks_dir / "commit.lock"
        with FileLock(lock_path):
            yield
            
    @contextmanager
    def branch_lock(self):
        """Lock for branch operations."""
        lock_path = self.locks_dir / "branch.lock"
        with FileLock(lock_path):
            yield
            
    @contextmanager
    def repository_lock(self):
        """Global repository lock for critical operations."""
        with self._repo_lock:
            yield


def atomic_write(path: Path, data: bytes, mode: int = 0o644):
    """Write data to file atomically.
    
    Args:
        path: Target file path
        data: Data to write
        mode: File permissions
    """
    # Write to temporary file in same directory
    temp_fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp"
    )
    
    try:
        # Write data
        os.write(temp_fd, data)
        os.fsync(temp_fd)
        os.close(temp_fd)
        
        # Set permissions
        os.chmod(temp_path, mode)
        
        # Atomic rename
        os.rename(temp_path, path)
        
    except Exception:
        # Clean up on error
        try:
            os.close(temp_fd)
        except Exception:
            pass
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        raise


def with_retry(func: Callable, max_attempts: int = 3, delay: float = 0.1) -> Any:
    """Execute function with retry logic.
    
    Args:
        func: Function to execute
        max_attempts: Maximum number of attempts
        delay: Delay between attempts
        
    Returns:
        Function result
    """
    last_error = None
    
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise last_error


class ThreadSafeCounter:
    """Thread-safe counter implementation."""
    
    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = threading.Lock()
        
    def increment(self, amount: int = 1) -> int:
        """Increment counter and return new value."""
        with self._lock:
            self._value += amount
            return self._value
            
    def decrement(self, amount: int = 1) -> int:
        """Decrement counter and return new value."""
        with self._lock:
            self._value -= amount
            return self._value
            
    def get(self) -> int:
        """Get current value."""
        with self._lock:
            return self._value
            
    def reset(self, value: int = 0):
        """Reset counter to specified value."""
        with self._lock:
            self._value = value


# Global lock registry to prevent deadlocks
_lock_registry = ThreadSafeDict()


def get_named_lock(name: str) -> threading.RLock:
    """Get or create a named lock.
    
    Args:
        name: Lock name
        
    Returns:
        RLock instance
    """
    lock = _lock_registry.get(name)
    if lock is None:
        lock = threading.RLock()
        _lock_registry.set(name, lock)
    return lock