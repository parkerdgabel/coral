#!/usr/bin/env python3
"""Demonstration of SafetensorsReader functionality.

This example shows how to use the SafetensorsReader to:
1. Read safetensors files with memory mapping
2. Access individual tensors with zero-copy operations
3. Validate file integrity
4. Get file information and metadata
"""

import logging
import tempfile
from pathlib import Path

import numpy as np

from coral.safetensors.reader import SafetensorsReader
from coral.safetensors.writer import SafetensorsWriter

# Configure logging to see debug messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def create_sample_file() -> Path:
    """Create a sample safetensors file for demonstration."""
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    
    # Create sample tensors
    tensors = {
        "embeddings": np.random.randn(1000, 768).astype(np.float32),
        "attention.weight": np.random.randn(768, 768).astype(np.float32),
        "attention.bias": np.random.randn(768).astype(np.float32),
        "mlp.fc1.weight": np.random.randn(3072, 768).astype(np.float16),
        "mlp.fc1.bias": np.random.randn(3072).astype(np.float16),
        "mlp.fc2.weight": np.random.randn(768, 3072).astype(np.float16),
        "mlp.fc2.bias": np.random.randn(768).astype(np.float16),
        "layer_norm.weight": np.ones(768, dtype=np.float32),
        "layer_norm.bias": np.zeros(768, dtype=np.float32),
    }
    
    # Add metadata
    metadata = {
        "model_type": "transformer",
        "hidden_size": 768,
        "num_layers": 12,
        "created_by": "safetensors_reader_demo",
    }
    
    # Write file
    with SafetensorsWriter(temp_path) as writer:
        writer.write_metadata(metadata)
        writer.write_tensors(tensors)
    
    print(f"Created sample file: {temp_path}")
    print(f"File size: {temp_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return temp_path


def demonstrate_basic_reading(file_path: Path) -> None:
    """Demonstrate basic reading functionality."""
    print("\n=== Basic Reading ===")
    
    with SafetensorsReader(file_path) as reader:
        # Get basic information
        print(f"Number of tensors: {len(reader)}")
        print(f"Tensor names: {reader.keys()}")
        
        # Access metadata
        print(f"\nMetadata: {reader.metadata}")
        
        # Read individual tensor
        embeddings = reader["embeddings"]
        print(f"\nEmbeddings shape: {embeddings.shape}")
        print(f"Embeddings dtype: {embeddings.dtype}")
        print(f"Embeddings mean: {embeddings.mean():.6f}")
        
        # Check if tensor exists
        print(f"\nContains 'embeddings': {'embeddings' in reader}")
        print(f"Contains 'nonexistent': {'nonexistent' in reader}")


def demonstrate_zero_copy(file_path: Path) -> None:
    """Demonstrate zero-copy reading with memory mapping."""
    print("\n=== Zero-Copy Reading ===")
    
    # With memory mapping (default)
    with SafetensorsReader(file_path, use_mmap=True) as reader:
        # Read without copying (returns read-only view)
        attention_weight = reader.read_tensor("attention.weight", copy=False)
        print(f"Attention weight shape: {attention_weight.shape}")
        print(f"Is writable: {attention_weight.flags.writeable}")
        print(f"Owns data: {attention_weight.flags.owndata}")
        
        # Try to modify (should fail)
        try:
            attention_weight[0, 0] = 999.0
        except ValueError as e:
            print(f"Cannot modify zero-copy view: {e}")
        
        # Read with copy (default behavior)
        attention_weight_copy = reader.read_tensor("attention.weight", copy=True)
        print(f"\nCopied tensor owns data: {attention_weight_copy.flags.owndata}")
        attention_weight_copy[0, 0] = 999.0  # This works
        print(f"Modified first element: {attention_weight_copy[0, 0]}")


def demonstrate_batch_reading(file_path: Path) -> None:
    """Demonstrate reading multiple tensors at once."""
    print("\n=== Batch Reading ===")
    
    with SafetensorsReader(file_path) as reader:
        # Read all MLP weights
        mlp_tensors = reader.read_tensors(
            names=["mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"]
        )
        
        print("MLP tensors:")
        for name, tensor in mlp_tensors.items():
            print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
        
        # Read all tensors except embeddings
        other_tensors = reader.read_tensors(exclude={"embeddings"})
        print(f"\nRead {len(other_tensors)} tensors (excluding embeddings)")


def demonstrate_file_info(file_path: Path) -> None:
    """Demonstrate getting file information."""
    print("\n=== File Information ===")
    
    with SafetensorsReader(file_path) as reader:
        info = reader.get_file_info()
        
        print(f"File size: {info['file_size'] / 1024 / 1024:.2f} MB")
        print(f"Header size: {info['header_size']} bytes")
        print(f"Number of tensors: {info['num_tensors']}")
        print(f"Total parameters: {info['total_parameters']:,}")
        print(f"Memory usage: {info['memory_usage'] / 1024 / 1024:.2f} MB")
        
        print("\nTensor details:")
        for tensor in info['tensors'][:3]:  # Show first 3 tensors
            print(f"  {tensor['name']}: "
                  f"shape={tensor['shape']}, "
                  f"dtype={tensor['dtype']}, "
                  f"params={tensor['parameters']:,}")


def demonstrate_validation(file_path: Path) -> None:
    """Demonstrate file validation."""
    print("\n=== File Validation ===")
    
    with SafetensorsReader(file_path) as reader:
        is_valid = reader.validate()
        print(f"File is valid: {is_valid}")
        
        # Get detailed tensor info
        for name in ["embeddings", "attention.weight"]:
            info = reader.get_tensor_info(name)
            if info:
                print(f"\n{name}:")
                print(f"  Shape: {info.shape}")
                print(f"  Dtype: {info.dtype.value}")
                print(f"  Byte size: {info.byte_size:,}")
                print(f"  Offsets: {info.data_offsets}")


def demonstrate_dict_interface(file_path: Path) -> None:
    """Demonstrate dictionary-like interface."""
    print("\n=== Dictionary Interface ===")
    
    with SafetensorsReader(file_path) as reader:
        # Iterate over tensor names
        print("Tensor names:")
        for i, name in enumerate(reader.keys()):
            if i < 3:  # Show first 3
                print(f"  {name}")
        
        # Iterate over name-tensor pairs
        print("\nIterating over items:")
        for i, (name, tensor) in enumerate(reader.items()):
            if i < 2:  # Show first 2
                print(f"  {name}: mean={tensor.mean():.6f}")


def demonstrate_error_handling() -> None:
    """Demonstrate error handling."""
    print("\n=== Error Handling ===")
    
    # Try to open non-existent file
    try:
        reader = SafetensorsReader("nonexistent.safetensors")
    except Exception as e:
        print(f"Expected error for non-existent file: {type(e).__name__}: {e}")
    
    # Create invalid file
    invalid_file = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
    invalid_path = Path(invalid_file.name)
    invalid_file.write(b"invalid data")
    invalid_file.close()
    
    # Try to open invalid file
    try:
        reader = SafetensorsReader(invalid_path)
    except Exception as e:
        print(f"Expected error for invalid file: {type(e).__name__}: {e}")
    
    invalid_path.unlink()


def main():
    """Run all demonstrations."""
    print("SafetensorsReader Demonstration")
    print("=" * 50)
    
    # Create sample file
    file_path = create_sample_file()
    
    try:
        # Run demonstrations
        demonstrate_basic_reading(file_path)
        demonstrate_zero_copy(file_path)
        demonstrate_batch_reading(file_path)
        demonstrate_file_info(file_path)
        demonstrate_validation(file_path)
        demonstrate_dict_interface(file_path)
        demonstrate_error_handling()
        
        print("\n" + "=" * 50)
        print("Demonstration complete!")
        
    finally:
        # Clean up
        if file_path.exists():
            file_path.unlink()
            print(f"\nCleaned up temporary file: {file_path}")


if __name__ == "__main__":
    main()