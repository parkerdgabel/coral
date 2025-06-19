"""
Coral: Neural network weight storage and deduplication system
"""

__version__ = "0.1.0"

from coral.core.weight_tensor import WeightTensor
from coral.core.deduplicator import Deduplicator
from coral.storage.weight_store import WeightStore
from coral.storage.hdf5_store import HDF5Store

__all__ = [
    "WeightTensor",
    "Deduplicator", 
    "WeightStore",
    "HDF5Store",
]