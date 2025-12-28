"""
Coral: Neural network weight storage and deduplication system
"""

__version__ = "1.0.0"

from coral.config import (
    CoralConfig,
    CoreConfig,
    DeltaEncodingConfig,
    UserConfig,
    get_default_config,
    load_config,
    validate_config,
)
from coral.core.deduplicator import Deduplicator
from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.delta.delta_encoder import (
    DeltaConfig,
    DeltaEncoder,
    DeltaReconstructionError,
    DeltaType,
)
from coral.storage.hdf5_store import HDF5Store
from coral.storage.weight_store import WeightStore
from coral.version_control.repository import (
    MergeConflictError,
    MergeStrategy,
    Repository,
)

__all__ = [
    # Core
    "WeightTensor",
    "WeightMetadata",
    "Deduplicator",
    # Storage
    "WeightStore",
    "HDF5Store",
    # Version Control
    "Repository",
    "MergeStrategy",
    "MergeConflictError",
    # Delta Encoding
    "DeltaEncoder",
    "DeltaConfig",
    "DeltaReconstructionError",
    "DeltaType",
    # Configuration
    "CoralConfig",
    "CoreConfig",
    "DeltaEncodingConfig",
    "UserConfig",
    "load_config",
    "get_default_config",
    "validate_config",
]
