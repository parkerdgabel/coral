#!/usr/bin/env python3
"""
Performance Demo for Coral ML

Demonstrates the performance testing capabilities with a subset of tests
that can run quickly to show the system's performance characteristics.
"""

import gc
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import psutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.core.deduplicator import Deduplicator
from coral.storage.hdf5_store import HDF5Store
from coral.version_control.repository import Repository


class QuickPerformanceDemo:
    """Quick performance demonstration."""
    
    def __init__(self):
        self.temp_dir = None
        self.results = {}
    
    def setup(self):
        """Setup demo environment."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="coral_perf_demo_"))
        print(f"Demo directory: {self.temp_dir}")
    
    def teardown(self):
        """Cleanup demo environment."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
        gc.collect()
    
    def _create_test_tensor(self, shape, name="test_tensor"):
        """Create a test tensor."""
        data = np.random.randn(*shape).astype(np.float32)
        metadata = WeightMetadata(
            name=name,
            shape=shape,
            dtype=data.dtype,
            layer_type="demo_layer"
        )
        return WeightTensor(data=data, metadata=metadata)
    
    def demo_memory_usage(self):
        """Demo memory usage measurement."""
        print("\n=== Memory Usage Demo ===")
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        results = {'initial_memory_mb': initial_memory, 'tests': []}
        
        # Test different tensor sizes
        sizes = [(100, 100), (500, 500), (1000, 1000)]
        
        for size in sizes:
            print(f"Testing tensor size: {size[0]}x{size[1]}")
            
            # Create tensor
            tensor = self._create_test_tensor(size, f"memory_test_{size[0]}x{size[1]}")
            
            # Measure memory after creation
            after_creation = process.memory_info().rss / 1024 / 1024
            
            # Store in repository
            repo_path = self.temp_dir / f"memory_repo_{size[0]}x{size[1]}"
            repo_path.mkdir()
            repo = Repository(repo_path, init=True)
            
            start_time = time.perf_counter()
            weights_dict = {f"memory_test_{size[0]}x{size[1]}": tensor}
            repo.stage_weights(weights_dict)
            repo.commit(f"Memory test {size}")
            store_time = time.perf_counter() - start_time
            
            # Measure memory after storage
            after_storage = process.memory_info().rss / 1024 / 1024
            
            # Retrieve from repository
            start_time = time.perf_counter()
            retrieved = repo.get_weight(f"memory_test_{size[0]}x{size[1]}")
            retrieve_time = time.perf_counter() - start_time
            
            # Calculate metrics
            tensor_size_mb = tensor.data.nbytes / 1024 / 1024
            memory_efficiency = (after_storage - initial_memory) / tensor_size_mb
            
            test_result = {
                'size': size,
                'tensor_size_mb': tensor_size_mb,
                'memory_after_creation_mb': after_creation,
                'memory_after_storage_mb': after_storage,
                'memory_efficiency_ratio': memory_efficiency,
                'store_time_ms': store_time * 1000,
                'retrieve_time_ms': retrieve_time * 1000,
                'data_integrity': np.allclose(tensor.data, retrieved.data)
            }
            
            results['tests'].append(test_result)
            
            print(f"  Size: {tensor_size_mb:.1f}MB")
            print(f"  Memory efficiency: {memory_efficiency:.2f}x")
            print(f"  Store time: {store_time*1000:.1f}ms")
            print(f"  Retrieve time: {retrieve_time*1000:.1f}ms")
            print(f"  Data integrity: {'✓' if test_result['data_integrity'] else '✗'}")
            
            # Cleanup
            del tensor, retrieved
            if hasattr(repo, 'close'):
                repo.close()
            del repo
            gc.collect()
        
        self.results['memory_usage'] = results
        return results
    
    def demo_deduplication_performance(self):
        """Demo deduplication performance."""
        print("\n=== Deduplication Performance Demo ===")
        
        results = {'tests': []}
        
        # Create base tensor
        base_tensor = self._create_test_tensor((200, 200), "dedup_base")
        
        # Test different numbers of similar tensors
        for num_similar in [10, 50, 100]:
            print(f"Testing deduplication with {num_similar} similar tensors")
            
            # Create similar tensors
            similar_tensors = [base_tensor]
            for i in range(num_similar - 1):
                # Add small noise to create similar but not identical tensors
                noise = np.random.randn(*base_tensor.shape) * 0.001
                similar_data = base_tensor.data + noise.astype(base_tensor.dtype)
                
                metadata = WeightMetadata(
                    name=f"similar_{i}",
                    shape=similar_data.shape,
                    dtype=similar_data.dtype
                )
                
                tensor = WeightTensor(data=similar_data, metadata=metadata)
                similar_tensors.append(tensor)
            
            # Test deduplication
            deduplicator = Deduplicator(similarity_threshold=0.95)
            
            start_time = time.perf_counter()
            for tensor in similar_tensors:
                deduplicator.add_weight(tensor)
            dedup_time = time.perf_counter() - start_time
            
            # Calculate metrics
            unique_weights = len(deduplicator.weight_index)
            deduplication_ratio = num_similar / unique_weights if unique_weights > 0 else 1
            throughput = num_similar / dedup_time if dedup_time > 0 else 0
            
            test_result = {
                'num_similar_tensors': num_similar,
                'unique_weights_found': unique_weights,
                'deduplication_ratio': deduplication_ratio,
                'dedup_time_ms': dedup_time * 1000,
                'throughput_tensors_per_sec': throughput
            }
            
            results['tests'].append(test_result)
            
            print(f"  Unique weights: {unique_weights}/{num_similar}")
            print(f"  Deduplication ratio: {deduplication_ratio:.2f}x")
            print(f"  Processing time: {dedup_time*1000:.1f}ms")
            print(f"  Throughput: {throughput:.1f} tensors/sec")
            
            # Cleanup
            del similar_tensors, deduplicator
            gc.collect()
        
        self.results['deduplication_performance'] = results
        return results
    
    def demo_repository_scaling(self):
        """Demo repository scaling performance."""
        print("\n=== Repository Scaling Demo ===")
        
        results = {'tests': []}
        
        # Create repository
        repo_path = self.temp_dir / "scaling_repo"
        repo_path.mkdir()
        repo = Repository(repo_path, init=True)
        
        base_tensor = self._create_test_tensor((100, 100), "scaling_base")
        
        # Test different repository sizes
        cumulative_weights = 0
        for batch_size in [10, 25, 50]:
            print(f"Adding {batch_size} weights to repository (total: {cumulative_weights + batch_size})")
            
            batch_start = time.perf_counter()
            
            # Add batch of weights
            for i in range(batch_size):
                # Create variation
                noise = np.random.randn(*base_tensor.shape) * 0.01
                data = base_tensor.data + noise.astype(base_tensor.dtype)
                
                metadata = WeightMetadata(
                    name=f"scaling_test_{cumulative_weights + i}",
                    shape=data.shape,
                    dtype=data.dtype
                )
                
                tensor = WeightTensor(data=data, metadata=metadata)
                weights_dict = {f"scaling_test_{cumulative_weights + i}": tensor}
                repo.stage_weights(weights_dict)
                repo.commit(f"Scaling test {cumulative_weights + i}")
            
            batch_time = time.perf_counter() - batch_start
            cumulative_weights += batch_size
            
            # Test retrieval performance
            retrieval_times = []
            for i in range(min(5, cumulative_weights)):  # Test 5 random retrievals
                weight_name = f"scaling_test_{i}"
                start_time = time.perf_counter()
                retrieved = repo.get_weight(weight_name)
                retrieve_time = time.perf_counter() - start_time
                if retrieved is not None:
                    retrieval_times.append(retrieve_time)
                del retrieved
            
            # Measure storage size
            storage_path = repo.weights_store_path
            storage_size_mb = storage_path.stat().st_size / 1024 / 1024 if storage_path.exists() else 0
            
            test_result = {
                'batch_size': batch_size,
                'total_weights': cumulative_weights,
                'batch_time_ms': batch_time * 1000,
                'avg_store_time_per_weight_ms': (batch_time / batch_size) * 1000,
                'avg_retrieve_time_ms': np.mean(retrieval_times) * 1000 if retrieval_times else 0,
                'storage_size_mb': storage_size_mb,
                'storage_per_weight_kb': (storage_size_mb * 1024) / cumulative_weights,
                'throughput_weights_per_sec': batch_size / batch_time
            }
            
            results['tests'].append(test_result)
            
            print(f"  Batch time: {batch_time*1000:.1f}ms")
            print(f"  Avg store time per weight: {test_result['avg_store_time_per_weight_ms']:.1f}ms")
            print(f"  Avg retrieve time: {test_result['avg_retrieve_time_ms']:.1f}ms")
            print(f"  Storage size: {storage_size_mb:.2f}MB")
            print(f"  Storage per weight: {test_result['storage_per_weight_kb']:.1f}KB")
        
        # Cleanup
        if hasattr(repo, 'close'):
            repo.close()
        del repo
        
        self.results['repository_scaling'] = results
        return results
    
    def demo_concurrent_operations(self):
        """Demo concurrent operations (simplified)."""
        print("\n=== Concurrent Operations Demo ===")
        
        from concurrent.futures import ThreadPoolExecutor
        
        results = {'tests': []}
        
        base_tensor = self._create_test_tensor((200, 200), "concurrent_base")
        
        for num_threads in [1, 2, 4]:
            print(f"Testing {num_threads} concurrent threads")
            
            # Create repository for this test
            repo_path = self.temp_dir / f"concurrent_repo_{num_threads}"
            repo_path.mkdir()
            repo = Repository(repo_path, init=True)
            
            def worker_function(worker_id):
                try:
                    # Create unique tensor
                    noise = np.random.randn(*base_tensor.shape) * (worker_id + 1) * 0.01
                    data = base_tensor.data + noise.astype(base_tensor.dtype)
                    
                    metadata = WeightMetadata(
                        name=f"concurrent_worker_{worker_id}",
                        shape=data.shape,
                        dtype=data.dtype
                    )
                    
                    tensor = WeightTensor(data=data, metadata=metadata)
                    
                    # Store tensor
                    start_time = time.perf_counter()
                    weights_dict = {f"concurrent_worker_{worker_id}": tensor}
                    repo.stage_weights(weights_dict)
                    repo.commit(f"Concurrent worker {worker_id}")
                    store_time = time.perf_counter() - start_time
                    
                    return {'worker_id': worker_id, 'success': True, 'store_time': store_time}
                    
                except Exception as e:
                    return {'worker_id': worker_id, 'success': False, 'error': str(e)}
            
            # Run concurrent operations
            start_time = time.perf_counter()
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker_function, i) for i in range(num_threads)]
                worker_results = [future.result() for future in futures]
            total_time = time.perf_counter() - start_time
            
            # Analyze results
            successful = [r for r in worker_results if r['success']]
            failed = [r for r in worker_results if not r['success']]
            
            test_result = {
                'num_threads': num_threads,
                'successful_operations': len(successful),
                'failed_operations': len(failed),
                'success_rate': len(successful) / num_threads,
                'total_time_ms': total_time * 1000,
                'avg_store_time_ms': np.mean([r['store_time'] for r in successful]) * 1000 if successful else 0,
                'throughput_ops_per_sec': num_threads / total_time if total_time > 0 else 0
            }
            
            results['tests'].append(test_result)
            
            print(f"  Success rate: {test_result['success_rate']:.1%}")
            print(f"  Total time: {total_time*1000:.1f}ms")
            print(f"  Avg store time: {test_result['avg_store_time_ms']:.1f}ms")
            print(f"  Throughput: {test_result['throughput_ops_per_sec']:.1f} ops/sec")
            
            if failed:
                print(f"  Errors: {[r['error'] for r in failed]}")
            
            # Cleanup
            if hasattr(repo, 'close'):
                repo.close()
            del repo
            gc.collect()
        
        self.results['concurrent_operations'] = results
        return results
    
    def generate_report(self):
        """Generate performance report."""
        print("\n" + "="*60)
        print("PERFORMANCE DEMO REPORT")
        print("="*60)
        
        # System information
        print(f"\nSystem Information:")
        print(f"  CPU cores: {psutil.cpu_count()}")
        print(f"  Memory: {psutil.virtual_memory().total / 1024**3:.1f}GB")
        print(f"  Python: {sys.version.split()[0]}")
        
        # Memory usage summary
        if 'memory_usage' in self.results:
            print(f"\nMemory Usage:")
            for test in self.results['memory_usage']['tests']:
                size_str = f"{test['size'][0]}x{test['size'][1]}"
                print(f"  {size_str}: {test['memory_efficiency_ratio']:.2f}x efficiency, "
                      f"{test['store_time_ms']:.1f}ms store, {test['retrieve_time_ms']:.1f}ms retrieve")
        
        # Deduplication summary
        if 'deduplication_performance' in self.results:
            print(f"\nDeduplication Performance:")
            for test in self.results['deduplication_performance']['tests']:
                print(f"  {test['num_similar_tensors']} tensors: {test['deduplication_ratio']:.2f}x compression, "
                      f"{test['throughput_tensors_per_sec']:.1f} tensors/sec")
        
        # Repository scaling summary
        if 'repository_scaling' in self.results:
            print(f"\nRepository Scaling:")
            for test in self.results['repository_scaling']['tests']:
                print(f"  {test['total_weights']} weights: {test['storage_per_weight_kb']:.1f}KB/weight, "
                      f"{test['throughput_weights_per_sec']:.1f} weights/sec")
        
        # Concurrent operations summary
        if 'concurrent_operations' in self.results:
            print(f"\nConcurrent Operations:")
            for test in self.results['concurrent_operations']['tests']:
                print(f"  {test['num_threads']} threads: {test['success_rate']:.1%} success, "
                      f"{test['throughput_ops_per_sec']:.1f} ops/sec")
        
        print(f"\n✅ Performance demo completed successfully!")
        return self.results
    
    def run_demo(self):
        """Run the complete performance demo."""
        print("Coral Performance Demo")
        print("="*60)
        
        try:
            self.setup()
            
            # Run demo tests
            self.demo_memory_usage()
            self.demo_deduplication_performance()
            self.demo_repository_scaling()
            self.demo_concurrent_operations()
            
            # Generate report
            return self.generate_report()
            
        finally:
            self.teardown()


def main():
    """Main entry point."""
    demo = QuickPerformanceDemo()
    results = demo.run_demo()
    return results


if __name__ == "__main__":
    main()