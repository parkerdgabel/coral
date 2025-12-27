"""Storage backends for weight persistence"""

from coral.storage.hdf5_store import HDF5Store
from coral.storage.weight_store import WeightStore

# Lazy import for S3 (requires boto3)
try:
    from coral.storage.s3_store import S3Config, S3Store

    _S3_AVAILABLE = True
except ImportError:
    _S3_AVAILABLE = False
    S3Store = None
    S3Config = None

__all__ = ["WeightStore", "HDF5Store", "S3Store", "S3Config"]


def get_s3_store(*args, **kwargs):
    """Get S3Store, raising helpful error if boto3 not installed."""
    if not _S3_AVAILABLE:
        raise ImportError(
            "S3 storage requires boto3. Install with: pip install coral-ml[s3]"
        )
    from coral.storage.s3_store import S3Store

    return S3Store(*args, **kwargs)
