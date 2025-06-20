"""
Coral SafeTensors Integration

This module provides comprehensive support for the SafeTensors format in Coral,
enabling secure, fast, and cross-framework tensor serialization.

Key Components:
- SafetensorsReader: Read safetensors files with lazy loading
- SafetensorsWriter: Write tensors to safetensors format
- SafetensorsStore: Storage backend using safetensors format
- Converter functions: Import/export between Coral and safetensors

Example Usage:
    # Reading safetensors files
    with SafetensorsReader("model.safetensors") as reader:
        weights = reader.read_tensor("embedding.weight")

    # Writing safetensors files
    with SafetensorsWriter("output.safetensors") as writer:
        writer.add_tensor("weights", numpy_array)

    # Converting from Coral to safetensors
    convert_coral_to_safetensors(repository, "model.safetensors")
"""

from .converter import (
    batch_convert_safetensors,
    convert_coral_to_safetensors,
    convert_safetensors_to_coral,
)
from .metadata import SafetensorsMetadata, extract_metadata, merge_metadata
from .reader import SafetensorsReader
from .types import DType, SafetensorsError, TensorInfo, TensorMetadata
from .writer import SafetensorsWriter

__all__ = [
    # Core classes
    "SafetensorsReader",
    "SafetensorsWriter",
    # Conversion functions
    "convert_coral_to_safetensors",
    "convert_safetensors_to_coral",
    "batch_convert_safetensors",
    # Metadata utilities
    "SafetensorsMetadata",
    "extract_metadata",
    "merge_metadata",
    # Types and exceptions
    "DType",
    "TensorInfo",
    "TensorMetadata",
    "SafetensorsError",
]

__version__ = "1.0.0"
