#!/usr/bin/env python3
"""Example demonstrating SafetensorsStore usage in Coral."""

import numpy as np
import tempfile
from pathlib import Path

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.storage.safetensors_store import SafetensorsStore


def main():
    """Demonstrate SafetensorsStore functionality."""
    
    # Create a temporary directory for storage
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Created temporary storage at: {tmpdir}")
        
        # Initialize SafetensorsStore
        print("\n1. Initialize SafetensorsStore")
        store = SafetensorsStore(tmpdir)
        print(f"   Storage initialized: {store}")
        
        # Create sample weights
        print("\n2. Create sample weights")
        weights = []
        for i in range(3):
            data = np.random.randn(10, 20).astype(np.float32)
            metadata = WeightMetadata(
                name=f"layer_{i}_weight",
                shape=data.shape,
                dtype=data.dtype,
                layer_type="dense",
                model_name="example_model",
                compression_info={"method": "none"}
            )
            weight = WeightTensor(data=data, metadata=metadata)
            weights.append(weight)
            print(f"   Created: {weight}")
        
        # Store weights
        print("\n3. Store weights")
        hash_keys = []
        for weight in weights:
            hash_key = store.store(weight)
            hash_keys.append(hash_key)
            print(f"   Stored {weight.metadata.name} with hash: {hash_key}")
        
        # List stored weights
        print("\n4. List all stored weights")
        stored_hashes = store.list_weights()
        print(f"   Found {len(stored_hashes)} weights: {stored_hashes}")
        
        # Load weights back
        print("\n5. Load weights back")
        for hash_key in hash_keys[:2]:  # Load first two
            loaded = store.load(hash_key)
            print(f"   Loaded: {loaded}")
        
        # Get metadata without loading data
        print("\n6. Get metadata without loading full data")
        metadata = store.get_metadata(hash_keys[0])
        print(f"   Metadata: name={metadata.name}, shape={metadata.shape}, dtype={metadata.dtype}")
        
        # Check storage info
        print("\n7. Storage information")
        info = store.get_storage_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Test with compression
        print("\n8. Test with compression enabled")
        compressed_store = SafetensorsStore(
            Path(tmpdir) / "compressed",
            use_compression=True,
            compression_level=6
        )
        
        # Store a large weight with compression
        large_data = np.random.randn(1000, 1000).astype(np.float32)
        large_weight = WeightTensor(
            data=large_data,
            metadata=WeightMetadata(
                name="large_weight",
                shape=large_data.shape,
                dtype=large_data.dtype,
                model_name="example_model"
            )
        )
        
        hash_key = compressed_store.store(large_weight)
        print(f"   Stored large weight with compression: {hash_key}")
        
        # Compare file sizes
        uncompressed_path = store._get_file_path(hash_key)
        compressed_path = compressed_store._get_file_path(hash_key)
        
        # Store in uncompressed for comparison
        store.store(large_weight)
        
        if uncompressed_path.exists() and compressed_path.exists():
            uncompressed_size = uncompressed_path.stat().st_size
            compressed_size = compressed_path.stat().st_size
            compression_ratio = uncompressed_size / compressed_size
            print(f"   Uncompressed size: {uncompressed_size:,} bytes")
            print(f"   Compressed size: {compressed_size:,} bytes")
            print(f"   Compression ratio: {compression_ratio:.2f}x")
        
        # Batch operations
        print("\n9. Batch operations")
        batch_weights = {
            f"batch_weight_{i}": WeightTensor(
                data=np.random.randn(5, 5).astype(np.float32),
                metadata=WeightMetadata(
                    name=f"batch_weight_{i}",
                    shape=(5, 5),
                    dtype=np.float32
                )
            )
            for i in range(5)
        }
        
        # Batch store
        hash_map = store.store_batch(batch_weights)
        print(f"   Stored {len(hash_map)} weights in batch")
        
        # Batch load
        loaded_batch = store.load_batch(list(hash_map.values()))
        print(f"   Loaded {len(loaded_batch)} weights in batch")
        
        print("\n10. Cleanup")
        store.close()
        compressed_store.close()
        print("   Stores closed successfully")


if __name__ == "__main__":
    main()