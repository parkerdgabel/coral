#!/usr/bin/env python3
"""Compare storage backends: HDF5Store vs SafetensorsStore."""

import numpy as np
import tempfile
import time
from pathlib import Path

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.storage.hdf5_store import HDF5Store
from coral.storage.safetensors_store import SafetensorsStore


def create_test_weights(num_weights: int = 50, size: tuple = (100, 100)) -> list:
    """Create test weights with various sizes."""
    weights = []
    for i in range(num_weights):
        # Vary sizes to test different scenarios
        if i % 3 == 0:
            shape = (50, 200)  # Different shape
        else:
            shape = size

        data = np.random.randn(*shape).astype(np.float32)
        metadata = WeightMetadata(
            name=f"weight_{i}",
            shape=shape,
            dtype=np.float32,
            layer_type="dense",
            model_name="test_model",
        )
        weights.append(WeightTensor(data=data, metadata=metadata))

    return weights


def benchmark_store(store_class, store_kwargs, weights, name):
    """Benchmark a storage backend."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize store
        if store_class == HDF5Store:
            store_path = Path(tmpdir) / "weights.h5"
        else:
            store_path = tmpdir

        store = store_class(store_path, **store_kwargs)

        # Benchmark storing
        start_time = time.time()
        hash_keys = []
        for weight in weights:
            hash_key = store.store(weight)
            hash_keys.append(hash_key)
        store_time = time.time() - start_time

        # Get storage size
        storage_info = store.get_storage_info()
        if store_class == HDF5Store:
            total_size = Path(store_path).stat().st_size
        else:
            total_size = storage_info["total_size_bytes"]

        # Benchmark loading
        start_time = time.time()
        for hash_key in hash_keys:
            loaded = store.load(hash_key)
            assert loaded is not None
        load_time = time.time() - start_time

        # Benchmark metadata retrieval
        start_time = time.time()
        for hash_key in hash_keys[:10]:  # Just test first 10
            metadata = store.get_metadata(hash_key)
            assert metadata is not None
        metadata_time = time.time() - start_time

        store.close()

        return {
            "name": name,
            "store_time": store_time,
            "load_time": load_time,
            "metadata_time": metadata_time,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
        }


def main():
    """Run storage backend comparison."""
    print("Storage Backend Comparison: HDF5 vs SafeTensors\n")

    # Create test data
    print("Creating test weights...")
    weights = create_test_weights(num_weights=100, size=(100, 100))
    total_data_size = sum(w.nbytes for w in weights)
    print(
        f"Created {len(weights)} weights, total size: {total_data_size / 1024 / 1024:.2f} MB\n"
    )

    # Test different configurations
    configs = [
        (HDF5Store, {}, "HDF5 (no compression)"),
        (HDF5Store, {"compression": "gzip", "compression_opts": 4}, "HDF5 (gzip)"),
        (SafetensorsStore, {}, "SafeTensors (no compression)"),
        (
            SafetensorsStore,
            {"use_compression": True, "compression_level": 6},
            "SafeTensors (gzip)",
        ),
    ]

    results = []
    for store_class, kwargs, name in configs:
        print(f"\nBenchmarking {name}...")
        result = benchmark_store(store_class, kwargs, weights, name)
        results.append(result)

        print(f"  Store time: {result['store_time']:.3f}s")
        print(f"  Load time: {result['load_time']:.3f}s")
        print(f"  Metadata time: {result['metadata_time']:.3f}s")
        print(f"  Storage size: {result['total_size_mb']:.2f} MB")
        print(
            f"  Compression ratio: {total_data_size / result['total_size_bytes']:.2f}x"
        )

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(
        f"{'Backend':<30} {'Store (s)':<12} {'Load (s)':<12} {'Size (MB)':<12} {'Ratio':<8}"
    )
    print("-" * 60)

    for result in results:
        print(
            f"{result['name']:<30} "
            f"{result['store_time']:<12.3f} "
            f"{result['load_time']:<12.3f} "
            f"{result['total_size_mb']:<12.2f} "
            f"{total_data_size / result['total_size_bytes']:<8.2f}x"
        )

    # Performance insights
    print("\n" + "=" * 60)
    print("INSIGHTS")
    print("=" * 60)

    # Find fastest for each operation
    fastest_store = min(results, key=lambda x: x["store_time"])
    fastest_load = min(results, key=lambda x: x["load_time"])
    smallest_size = min(results, key=lambda x: x["total_size_bytes"])

    print(
        f"Fastest store: {fastest_store['name']} ({fastest_store['store_time']:.3f}s)"
    )
    print(f"Fastest load: {fastest_load['name']} ({fastest_load['load_time']:.3f}s)")
    print(
        f"Smallest size: {smallest_size['name']} ({smallest_size['total_size_mb']:.2f} MB)"
    )

    # SafeTensors vs HDF5 comparison
    hdf5_uncompressed = next(r for r in results if r["name"] == "HDF5 (no compression)")
    safetensors_uncompressed = next(
        r for r in results if r["name"] == "SafeTensors (no compression)"
    )

    print(f"\nSafeTensors vs HDF5 (uncompressed):")
    print(
        f"  Store: {safetensors_uncompressed['store_time'] / hdf5_uncompressed['store_time']:.2f}x speed"
    )
    print(
        f"  Load: {safetensors_uncompressed['load_time'] / hdf5_uncompressed['load_time']:.2f}x speed"
    )
    print(
        f"  Size: {safetensors_uncompressed['total_size_bytes'] / hdf5_uncompressed['total_size_bytes']:.2f}x ratio"
    )


if __name__ == "__main__":
    main()
