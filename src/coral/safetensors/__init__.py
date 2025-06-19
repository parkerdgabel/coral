"""Safetensors integration for Coral.

This module provides functionality to read, write, and convert between
Coral's weight storage format and the Safetensors format.

Safetensors is a simple, safe way to store tensors on disk, designed to be
a drop-in replacement for pickle while being safer and faster.
"""

from coral.safetensors.converter import (
    convert_coral_to_safetensors,
    convert_safetensors_to_coral,
)
from coral.safetensors.metadata import (
    SafetensorsMetadata,
    extract_metadata,
    merge_metadata,
)
from coral.safetensors.reader import SafetensorsReader
from coral.safetensors.types import (
    DType,
    SafetensorDict,
    TensorInfo,
    TensorMetadata,
)
from coral.safetensors.writer import SafetensorsWriter

__all__ = [
    # Reader/Writer
    "SafetensorsReader",
    "SafetensorsWriter",
    # Converter functions
    "convert_coral_to_safetensors",
    "convert_safetensors_to_coral",
    # Metadata handling
    "SafetensorsMetadata",
    "extract_metadata",
    "merge_metadata",
    # Types
    "DType",
    "TensorInfo",
    "TensorMetadata",
    "SafetensorDict",
]