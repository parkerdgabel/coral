"""Compression techniques for weight storage"""

from coral.compression.quantization import Quantizer
from coral.compression.pruning import Pruner

__all__ = ["Quantizer", "Pruner"]