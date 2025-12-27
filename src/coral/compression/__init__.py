"""Compression techniques for weight storage.

.. warning::
    EXPERIMENTAL: This module is experimental and not yet integrated into the
    core Coral workflow. The API may change in future versions.

Provides:
- Quantizer: Weight quantization (8-bit, 4-bit, 2-bit)
- Pruner: Weight pruning for sparsity
"""

import warnings

from coral.compression.pruning import Pruner
from coral.compression.quantization import Quantizer

__all__ = ["Quantizer", "Pruner"]

# Issue deprecation warning on import
warnings.warn(
    "coral.compression is experimental and may change in future versions.",
    FutureWarning,
    stacklevel=2,
)
