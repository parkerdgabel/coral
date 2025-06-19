"""Compression techniques for weight storage"""

from coral.compression.pruning import Pruner
from coral.compression.quantization import Quantizer

__all__ = ["Quantizer", "Pruner"]
