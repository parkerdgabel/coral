"""Core abstractions for weight storage and deduplication"""

from coral.core.deduplicator import Deduplicator
from coral.core.weight_tensor import WeightTensor

__all__ = ["WeightTensor", "Deduplicator"]
