"""Storage backends for weight persistence"""

from coral.storage.weight_store import WeightStore
from coral.storage.hdf5_store import HDF5Store

__all__ = ["WeightStore", "HDF5Store"]