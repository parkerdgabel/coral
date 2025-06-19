"""Core abstractions for weight storage and deduplication"""

from coral.core.weight_tensor import WeightTensor
from coral.core.deduplicator import Deduplicator

__all__ = ["WeightTensor", "Deduplicator"]