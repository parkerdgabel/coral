"""
Coral: Neural network weight storage and deduplication system
"""

__version__ = "1.0.0"

from coral.core.deduplicator import Deduplicator
from coral.core.weight_tensor import WeightTensor
from coral.delta.delta_encoder import (
    DeltaConfig,
    DeltaEncoder,
    DeltaReconstructionError,
    DeltaType,
)
from coral.storage.hdf5_store import HDF5Store
from coral.storage.weight_store import WeightStore
from coral.version_control.repository import Repository

__all__ = [
    "WeightTensor",
    "Deduplicator",
    "WeightStore",
    "HDF5Store",
    "Repository",
    "DeltaEncoder",
    "DeltaConfig",
    "DeltaReconstructionError",
    "DeltaType",
]
