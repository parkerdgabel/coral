#!/usr/bin/env python3
"""Demonstration of Coral's Safetensors implementation."""

import numpy as np
from pathlib import Path
import tempfile

from coral.safetensors.writer import SafetensorsWriter
from coral.safetensors.reader import SafetensorsReader


def main():
    """Demonstrate Safetensors writer and reader functionality."""
    # Create a temporary directory for our demo
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "model.safetensors"
        
        print("=== Coral Safetensors Demo ===\n")
        
        # 1. Create some example tensors (simulating a neural network)
        tensors = {
            "encoder.weight": np.random.randn(768, 512).astype(np.float32),
            "encoder.bias": np.zeros(768, dtype=np.float32),
            "decoder.weight": np.random.randn(512, 768).astype(np.float32),
            "decoder.bias": np.zeros(512, dtype=np.float32),
            "embeddings": np.random.randn(10000, 768).astype(np.float16),
        }
        
        metadata = {
            "model": "demo_transformer",
            "version": "1.0",
            "framework": "coral",
            "description": "Example model for Safetensors demo"
        }
        
        # 2. Write tensors using context manager
        print("Writing tensors to Safetensors file...")
        with SafetensorsWriter(file_path, metadata=metadata) as writer:
            for name, tensor in tensors.items():
                writer.add_tensor(name, tensor)
                print(f"  Added {name}: shape={tensor.shape}, dtype={tensor.dtype}")
        
        # 3. Show file size information
        file_size = file_path.stat().st_size
        print(f"\nFile written: {file_path.name}")
        print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        
        # 4. Read back the tensors
        print("\nReading tensors from Safetensors file...")
        reader = SafetensorsReader(file_path)
        
        # Show metadata
        print(f"\nMetadata: {reader.metadata}")
        
        # List all tensors
        print(f"\nAvailable tensors: {reader.get_tensor_names()}")
        
        # Read and verify each tensor
        print("\nVerifying tensors...")
        for name, original in tensors.items():
            loaded = reader.read_tensor(name)
            matches = np.array_equal(loaded, original)
            print(f"  {name}: {'✓' if matches else '✗'} (matches original)")
            
            # Show tensor info
            info = reader.get_tensor_info(name)
            if info:
                print(f"    Shape: {info.shape}, dtype: {info.dtype.value}, "
                      f"offset: {info.data_offsets[0]}, size: {info.data_offsets[1] - info.data_offsets[0]} bytes")
        
        # 5. Demonstrate efficient partial loading
        print("\nDemonstrating efficient tensor loading...")
        # Only load one specific tensor
        encoder_weight = reader.read_tensor("encoder.weight")
        print(f"Loaded encoder.weight: shape={encoder_weight.shape}, mean={encoder_weight.mean():.4f}")
        
        # 6. Show file size estimation accuracy
        print("\nFile size estimation demo:")
        writer2 = SafetensorsWriter(Path(tmpdir) / "test2.safetensors")
        writer2.add_tensors(tensors)
        estimated = writer2.estimate_file_size()
        writer2.write()
        actual = (Path(tmpdir) / "test2.safetensors").stat().st_size
        print(f"  Estimated: {estimated:,} bytes")
        print(f"  Actual: {actual:,} bytes")
        print(f"  Accuracy: {100 * (1 - abs(estimated - actual) / actual):.2f}%")
        
        print("\n✓ Demo completed successfully!")


if __name__ == "__main__":
    main()