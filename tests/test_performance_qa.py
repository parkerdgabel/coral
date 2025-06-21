"""
Performance QA Test Suite for Coral ML

This module provides comprehensive performance testing to identify:
- Memory leaks and usage patterns
- Performance bottlenecks and scalability limits
- Concurrent operation efficiency
- Storage and clustering performance
- System resource utilization

All tests measure baseline and peak metrics with detailed profiling.
"""

import gc
import json
import multiprocessing
import os
import psutil
import shutil
import sys
import tempfile
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

import numpy as np
import pytest
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.core.deduplicator import Deduplicator
from coral.storage.hdf5_store import HDF5Store
from coral.version_control.repository import Repository
from coral.delta.delta_encoder import DeltaEncoder, DeltaStrategy
from coral.training.checkpoint_manager import CheckpointManager


class PerformanceProfiler:
    """Comprehensive performance profiling utility."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset profiler state."""
        self.start_time = None
        self.memory_samples = []
        self.cpu_samples = []
        self.gc_stats = []
        self.monitoring_active = False
        self.process = psutil.Process()
        self._initial_memory = None
        self._peak_memory = 0
        
    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring_active = True
        self.start_time = time.perf_counter()
        self._initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self._peak_memory = self._initial_memory
        
        # Start memory tracing
        tracemalloc.start()
        
    def sample_resources(self):
        """Sample current resource usage."""
        if not self.monitoring_active:
            return
            
        # Memory usage
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        self._peak_memory = max(self._peak_memory, memory_mb)
        self.memory_samples.append(memory_mb)
        
        # CPU usage
        cpu_percent = self.process.cpu_percent()
        self.cpu_samples.append(cpu_percent)
        
        # Garbage collection stats
        gc_stats = gc.get_stats()
        self.gc_stats.append(gc_stats)
        
    def stop_monitoring(self):
        """Stop resource monitoring and return results."""
        if not self.monitoring_active:
            return {}
            
        self.monitoring_active = False
        end_time = time.perf_counter()
        
        # Get tracemalloc snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        # Stop tracing
        tracemalloc.stop()
        
        return {
            'duration_seconds': end_time - self.start_time,
            'initial_memory_mb': self._initial_memory,
            'peak_memory_mb': self._peak_memory,
            'memory_growth_mb': self._peak_memory - self._initial_memory,
            'avg_memory_mb': np.mean(self.memory_samples) if self.memory_samples else 0,
            'memory_samples': self.memory_samples,
            'avg_cpu_percent': np.mean(self.cpu_samples) if self.cpu_samples else 0,
            'peak_cpu_percent': max(self.cpu_samples) if self.cpu_samples else 0,
            'cpu_samples': self.cpu_samples,
            'gc_collections': len(self.gc_stats),
            'top_memory_allocations': [(stat.traceback.format(), stat.size / 1024 / 1024) 
                                     for stat in top_stats[:5]]
        }
    
    def measure_operation_time(self, operation, *args, **kwargs):
        """Measure operation execution time with resource monitoring."""
        self.sample_resources()
        start_time = time.perf_counter()
        
        try:
            result = operation(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.perf_counter()
        self.sample_resources()
        
        return {
            'result': result,
            'success': success,
            'error': error,
            'duration_ms': (end_time - start_time) * 1000,
            'duration_seconds': end_time - start_time
        }


class PerformanceQATestSuite:
    """Comprehensive performance QA test suite."""
    
    def __init__(self):
        self.temp_dir = None
        self.repositories = {}
        self.profiler = PerformanceProfiler()
        self.test_results = {}
        
        # System information
        self.cpu_count = multiprocessing.cpu_count()
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Test configuration
        self.large_tensor_sizes = [
            (1000, 1000),      # 1M parameters
            (2000, 2000),      # 4M parameters
            (5000, 5000),      # 25M parameters
        ]
        
        self.stress_test_sizes = [10, 50, 100, 500, 1000, 5000]
        self.concurrency_levels = [1, 2, 4, 8, 16, 32]
        
    def setup(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="coral_perf_qa_"))
        print(f"Test directory: {self.temp_dir}")
        
        # Create test repositories
        self._create_test_repositories()
        
    def teardown(self):
        """Cleanup test environment."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            
        # Close repositories
        for repo in self.repositories.values():
            if hasattr(repo, 'close'):
                repo.close()
                
        # Force garbage collection
        gc.collect()
        
    def _create_test_repositories(self):
        """Create test repositories with different configurations."""
        configs = {
            'standard': {'similarity_threshold': 0.95},
            'high_similarity': {'similarity_threshold': 0.99},
            'low_similarity': {'similarity_threshold': 0.85},
        }
        
        for name, config in configs.items():
            repo_path = self.temp_dir / f"repo_{name}"
            repo_path.mkdir(parents=True)
            
            store = HDF5Store(repo_path / "weights.h5")
            self.repositories[name] = Repository(repo_path, store=store, **config)
    
    def _create_test_tensor(self, shape: Tuple[int, ...], name: str = "test_tensor", 
                           dtype: np.dtype = np.float32, random_seed: int = 42) -> WeightTensor:
        """Create a test tensor with specified properties."""
        np.random.seed(random_seed)
        
        # Create data based on shape size to test memory efficiency
        if np.prod(shape) > 10_000_000:  # Large tensor
            # Use a pattern to make compression more predictable
            data = np.random.randn(*shape).astype(dtype) * 0.1
        else:
            data = np.random.randn(*shape).astype(dtype)
            
        metadata = WeightMetadata(
            name=name,
            shape=shape,
            dtype=dtype,
            layer_type="test_layer",
            model_name="test_model"
        )
        
        return WeightTensor(data=data, metadata=metadata)
    
    def _create_similar_tensor(self, base_tensor: WeightTensor, similarity: float = 0.95, 
                              name: str = "similar_tensor") -> WeightTensor:
        """Create a tensor similar to base_tensor with specified similarity level."""
        base_data = base_tensor.data
        noise_scale = np.sqrt(2 * (1 - similarity)) * np.std(base_data)
        noise = np.random.randn(*base_data.shape) * noise_scale
        
        similar_data = base_data + noise.astype(base_data.dtype)
        
        metadata = WeightMetadata(
            name=name,
            shape=base_data.shape,
            dtype=base_data.dtype,
            layer_type=base_tensor.metadata.layer_type,
            model_name=base_tensor.metadata.model_name
        )
        
        return WeightTensor(data=similar_data, metadata=metadata)
    
    def test_memory_usage_patterns(self):
        """Test 1: Memory usage patterns during large operations."""
        print("\n=== Test 1: Memory Usage Patterns ===")
        
        results = {}
        
        for size in self.large_tensor_sizes:
            print(f"Testing tensor size: {size}")
            
            self.profiler.reset()
            self.profiler.start_monitoring()
            
            # Create large tensor
            tensor = self._create_test_tensor(size, f"large_tensor_{size[0]}x{size[1]}")
            self.profiler.sample_resources()
            
            # Store in repository
            repo = self.repositories['standard']
            repo.store_weight(tensor, "large_tensor_test")
            self.profiler.sample_resources()
            
            # Retrieve from repository
            weight_id = list(repo.store.list_weights().keys())[0]
            retrieved_tensor = repo.store.get_weight(weight_id)
            self.profiler.sample_resources()
            
            # Clean up
            del tensor, retrieved_tensor
            gc.collect()
            
            perf_stats = self.profiler.stop_monitoring()
            
            results[f"{size[0]}x{size[1]}"] = {
                'tensor_size_mb': np.prod(size) * 4 / 1024 / 1024,  # float32
                'memory_stats': perf_stats,
                'memory_efficiency': perf_stats['memory_growth_mb'] / (np.prod(size) * 4 / 1024 / 1024)
            }
            
            repo.clear()
        
        self.test_results['memory_usage_patterns'] = results
        return results
    
    def test_memory_leak_detection(self):
        """Test 2: Memory leak detection during repeated operations."""
        print("\n=== Test 2: Memory Leak Detection ===")
        
        self.profiler.reset()
        self.profiler.start_monitoring()
        
        repo = self.repositories['standard']
        base_tensor = self._create_test_tensor((1000, 1000), "base_tensor")
        
        memory_measurements = []
        
        # Repeat operations many times to detect leaks
        for i in range(100):
            # Create similar tensor
            similar_tensor = self._create_similar_tensor(base_tensor, name=f"similar_{i}")
            
            # Store and retrieve
            repo.store_weight(similar_tensor, f"leak_test_{i}")
            weight_ids = list(repo.store.list_weights().keys())
            if weight_ids:
                retrieved = repo.store.get_weight(weight_ids[-1])
                del retrieved
            
            # Sample memory every 10 iterations
            if i % 10 == 0:
                self.profiler.sample_resources()
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_measurements.append((i, current_memory))
            
            del similar_tensor
            
            # Force garbage collection periodically
            if i % 20 == 0:
                gc.collect()
        
        perf_stats = self.profiler.stop_monitoring()
        
        # Analyze memory growth trend
        if len(memory_measurements) > 2:
            iterations, memory_values = zip(*memory_measurements)
            memory_growth_trend = np.polyfit(iterations, memory_values, 1)[0]  # Linear trend
        else:
            memory_growth_trend = 0
        
        results = {
            'memory_measurements': memory_measurements,
            'memory_growth_trend_mb_per_iteration': memory_growth_trend,
            'total_iterations': 100,
            'final_memory_mb': memory_measurements[-1][1] if memory_measurements else 0,
            'initial_memory_mb': memory_measurements[0][1] if memory_measurements else 0,
            'performance_stats': perf_stats,
            'potential_memory_leak': memory_growth_trend > 0.1  # > 0.1 MB per iteration
        }
        
        self.test_results['memory_leak_detection'] = results
        repo.clear()
        return results
    
    def test_hdf5_concurrent_stress(self):
        """Test 3: Stress test HDF5 storage with concurrent operations."""
        print("\n=== Test 3: HDF5 Concurrent Stress Test ===")
        
        results = {}
        
        for num_threads in self.concurrency_levels:
            if num_threads > self.cpu_count * 2:
                continue
                
            print(f"Testing with {num_threads} threads")
            
            self.profiler.reset()
            self.profiler.start_monitoring()
            
            repo = self.repositories['standard']
            repo.clear()
            
            # Create base tensor for concurrent operations
            base_tensor = self._create_test_tensor((500, 500), "concurrent_base")
            
            def concurrent_worker(worker_id: int):
                """Worker function for concurrent operations."""
                try:
                    # Create unique tensor for this worker
                    worker_tensor = self._create_similar_tensor(
                        base_tensor, 
                        name=f"worker_{worker_id}_tensor"
                    )
                    
                    # Store tensor
                    start_time = time.perf_counter()
                    repo.store_weight(worker_tensor, f"concurrent_test_{worker_id}")
                    store_time = time.perf_counter() - start_time
                    
                    # Retrieve tensor
                    weight_ids = list(repo.store.list_weights().keys())
                    if weight_ids:
                        start_time = time.perf_counter()
                        retrieved = repo.store.get_weight(weight_ids[-1])
                        retrieve_time = time.perf_counter() - start_time
                        del retrieved
                    else:
                        retrieve_time = 0
                    
                    return {
                        'worker_id': worker_id,
                        'store_time': store_time,
                        'retrieve_time': retrieve_time,
                        'success': True,
                        'error': None
                    }
                except Exception as e:
                    return {
                        'worker_id': worker_id,
                        'store_time': 0,
                        'retrieve_time': 0,
                        'success': False,
                        'error': str(e)
                    }
            
            # Run concurrent operations
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(concurrent_worker, i) for i in range(num_threads)]
                worker_results = [future.result() for future in as_completed(futures)]
            
            perf_stats = self.profiler.stop_monitoring()
            
            # Analyze results
            successful_workers = [r for r in worker_results if r['success']]
            failed_workers = [r for r in worker_results if not r['success']]
            
            if successful_workers:
                avg_store_time = np.mean([r['store_time'] for r in successful_workers])
                avg_retrieve_time = np.mean([r['retrieve_time'] for r in successful_workers])
            else:
                avg_store_time = 0
                avg_retrieve_time = 0
            
            results[f"threads_{num_threads}"] = {
                'num_threads': num_threads,
                'successful_operations': len(successful_workers),
                'failed_operations': len(failed_workers),
                'success_rate': len(successful_workers) / num_threads,
                'avg_store_time': avg_store_time,
                'avg_retrieve_time': avg_retrieve_time,
                'performance_stats': perf_stats,
                'errors': [r['error'] for r in failed_workers]
            }
            
            repo.clear()
        
        self.test_results['hdf5_concurrent_stress'] = results
        return results
    
    def test_clustering_performance_scaling(self):
        """Test 4: Clustering performance with large numbers of tensors."""
        print("\n=== Test 4: Clustering Performance Scaling ===")
        
        results = {}
        
        for num_tensors in [100, 500, 1000, 2000]:
            if num_tensors > 2000:  # Skip very large tests to avoid memory issues
                continue
                
            print(f"Testing clustering with {num_tensors} tensors")
            
            self.profiler.reset()
            self.profiler.start_monitoring()
            
            # Create base tensor and variations
            base_tensor = self._create_test_tensor((100, 100), "clustering_base")
            tensors = [base_tensor]
            
            # Create similar tensors for clustering
            for i in range(num_tensors - 1):
                similar_tensor = self._create_similar_tensor(
                    base_tensor, 
                    similarity=0.95 + np.random.random() * 0.04,  # 95-99% similar
                    name=f"cluster_tensor_{i}"
                )
                tensors.append(similar_tensor)
            
            self.profiler.sample_resources()
            
            # Test deduplication performance
            deduplicator = Deduplicator(similarity_threshold=0.95)
            
            dedup_start = time.perf_counter()
            for tensor in tensors:
                deduplicator.add_weight(tensor)
            dedup_time = time.perf_counter() - dedup_start
            
            self.profiler.sample_resources()
            
            # Test repository clustering (if available)
            repo = self.repositories['standard']
            repo.clear()
            
            # Store all tensors
            store_start = time.perf_counter()
            for i, tensor in enumerate(tensors):
                repo.store_weight(tensor, f"clustering_test_{i}")
            store_time = time.perf_counter() - store_start
            
            self.profiler.sample_resources()
            
            perf_stats = self.profiler.stop_monitoring()
            
            # Analyze clustering results
            unique_weights = len(deduplicator.weight_index)
            deduplication_ratio = num_tensors / unique_weights if unique_weights > 0 else 1
            
            results[f"tensors_{num_tensors}"] = {
                'num_tensors': num_tensors,
                'unique_weights': unique_weights,
                'deduplication_ratio': deduplication_ratio,
                'dedup_time_seconds': dedup_time,
                'store_time_seconds': store_time,
                'tensors_per_second_dedup': num_tensors / dedup_time if dedup_time > 0 else 0,
                'tensors_per_second_store': num_tensors / store_time if store_time > 0 else 0,
                'performance_stats': perf_stats
            }
            
            # Clean up
            del tensors, deduplicator
            repo.clear()
            gc.collect()
        
        self.test_results['clustering_performance_scaling'] = results
        return results
    
    def test_delta_encoding_cpu_profiling(self):
        """Test 5: Profile CPU usage during delta encoding operations."""
        print("\n=== Test 5: Delta Encoding CPU Profiling ===")
        
        results = {}
        
        # Test different delta strategies
        strategies = [
            DeltaStrategy.FLOAT32_RAW,
            DeltaStrategy.COMPRESSED,
            DeltaStrategy.INT8_QUANTIZED
        ]
        
        base_tensor = self._create_test_tensor((2000, 2000), "delta_base")
        
        for strategy in strategies:
            print(f"Testing strategy: {strategy.value}")
            
            self.profiler.reset()
            self.profiler.start_monitoring()
            
            encoder = DeltaEncoder(default_strategy=strategy)
            
            # Create multiple similar tensors
            similar_tensors = []
            for i in range(20):
                similar_tensor = self._create_similar_tensor(
                    base_tensor, 
                    similarity=0.98,
                    name=f"delta_similar_{i}"
                )
                similar_tensors.append(similar_tensor)
            
            self.profiler.sample_resources()
            
            # Encode deltas
            encode_times = []
            decode_times = []
            compression_ratios = []
            
            for tensor in similar_tensors:
                # Encode delta
                encode_start = time.perf_counter()
                delta = encoder.encode_delta(tensor.data, base_tensor.data)
                encode_time = time.perf_counter() - encode_start
                encode_times.append(encode_time)
                
                # Decode delta
                decode_start = time.perf_counter()
                reconstructed = encoder.decode_delta(delta, base_tensor.data)
                decode_time = time.perf_counter() - decode_start
                decode_times.append(decode_time)
                
                # Calculate compression ratio
                original_size = tensor.data.nbytes
                if hasattr(delta, 'data'):
                    compressed_size = len(delta.data) if isinstance(delta.data, bytes) else delta.data.nbytes
                else:
                    compressed_size = original_size  # No compression
                
                compression_ratio = original_size / compressed_size if compressed_size > 0 else 1
                compression_ratios.append(compression_ratio)
                
                # Verify reconstruction quality
                mse = np.mean((tensor.data - reconstructed) ** 2)
                
                self.profiler.sample_resources()
            
            perf_stats = self.profiler.stop_monitoring()
            
            results[strategy.value] = {
                'num_tensors': len(similar_tensors),
                'avg_encode_time': np.mean(encode_times),
                'avg_decode_time': np.mean(decode_times),
                'avg_compression_ratio': np.mean(compression_ratios),
                'encode_throughput_tensors_per_sec': len(similar_tensors) / sum(encode_times),
                'decode_throughput_tensors_per_sec': len(similar_tensors) / sum(decode_times),
                'performance_stats': perf_stats
            }
            
            del similar_tensors, encoder
            gc.collect()
        
        self.test_results['delta_encoding_cpu_profiling'] = results
        return results
    
    def test_deep_commit_history_performance(self):
        """Test 6: Performance with very deep commit histories."""
        print("\n=== Test 6: Deep Commit History Performance ===")
        
        self.profiler.reset()
        self.profiler.start_monitoring()
        
        repo = self.repositories['standard']
        repo.clear()
        
        # Create a series of commits
        num_commits = 1000
        base_tensor = self._create_test_tensor((100, 100), "commit_base")
        
        commit_times = []
        retrieval_times = []
        
        for i in range(num_commits):
            # Create slight variation
            tensor = self._create_similar_tensor(
                base_tensor, 
                similarity=0.99,
                name=f"commit_{i}"
            )
            
            # Commit with timing
            commit_start = time.perf_counter()
            repo.store_weight(tensor, f"Deep commit {i}")
            commit_time = time.perf_counter() - commit_start
            commit_times.append(commit_time)
            
            # Periodically test retrieval performance
            if i % 50 == 0:
                weight_ids = list(repo.store.list_weights().keys())
                if weight_ids:
                    retrieve_start = time.perf_counter()
                    retrieved = repo.store.get_weight(weight_ids[-1])
                    retrieve_time = time.perf_counter() - retrieve_start
                    retrieval_times.append(retrieve_time)
                    del retrieved
                
                self.profiler.sample_resources()
        
        perf_stats = self.profiler.stop_monitoring()
        
        # Analyze performance trends
        commit_trend = np.polyfit(range(len(commit_times)), commit_times, 1)[0]
        
        results = {
            'num_commits': num_commits,
            'avg_commit_time': np.mean(commit_times),
            'commit_time_trend': commit_trend,
            'first_100_avg_commit_time': np.mean(commit_times[:100]),
            'last_100_avg_commit_time': np.mean(commit_times[-100:]),
            'avg_retrieval_time': np.mean(retrieval_times) if retrieval_times else 0,
            'performance_degradation': commit_trend > 0.001,  # More than 1ms increase per commit
            'performance_stats': perf_stats
        }
        
        self.test_results['deep_commit_history_performance'] = results
        repo.clear()
        return results
    
    def test_extreme_tensor_sizes(self):
        """Test 7: Storage limits with extremely large tensors."""
        print("\n=== Test 7: Extreme Tensor Sizes ===")
        
        results = {}
        
        # Test progressively larger tensors
        extreme_sizes = [
            (10000, 10000),   # 100M parameters (~400MB)
            (15000, 15000),   # 225M parameters (~900MB)
            (20000, 20000),   # 400M parameters (~1.6GB)
        ]
        
        for size in extreme_sizes:
            tensor_size_gb = np.prod(size) * 4 / (1024**3)  # float32 in GB
            
            # Skip if tensor would use more than 50% of available memory
            if tensor_size_gb > self.total_memory_gb * 0.5:
                results[f"{size[0]}x{size[1]}"] = {
                    'skipped': True,
                    'reason': f'Tensor size ({tensor_size_gb:.2f}GB) exceeds 50% of available memory ({self.total_memory_gb:.2f}GB)'
                }
                continue
            
            print(f"Testing extreme tensor size: {size} ({tensor_size_gb:.2f}GB)")
            
            self.profiler.reset()
            self.profiler.start_monitoring()
            
            try:
                # Create large tensor
                create_start = time.perf_counter()
                tensor = self._create_test_tensor(size, f"extreme_tensor_{size[0]}x{size[1]}")
                create_time = time.perf_counter() - create_start
                
                self.profiler.sample_resources()
                
                # Store tensor
                repo = self.repositories['standard']
                repo.clear()
                
                store_start = time.perf_counter()
                repo.store_weight(tensor, "extreme_tensor_test")
                store_time = time.perf_counter() - store_start
                
                self.profiler.sample_resources()
                
                # Retrieve tensor
                weight_ids = list(repo.store.list_weights().keys())
                if weight_ids:
                    retrieve_start = time.perf_counter()
                    retrieved_tensor = repo.store.get_weight(weight_ids[0])
                    retrieve_time = time.perf_counter() - retrieve_start
                    
                    # Verify data integrity
                    data_integrity = np.allclose(tensor.data, retrieved_tensor.data)
                    del retrieved_tensor
                else:
                    retrieve_time = 0
                    data_integrity = False
                
                perf_stats = self.profiler.stop_monitoring()
                
                results[f"{size[0]}x{size[1]}"] = {
                    'tensor_size_gb': tensor_size_gb,
                    'create_time': create_time,
                    'store_time': store_time,
                    'retrieve_time': retrieve_time,
                    'data_integrity': data_integrity,
                    'throughput_gb_per_sec_store': tensor_size_gb / store_time if store_time > 0 else 0,
                    'throughput_gb_per_sec_retrieve': tensor_size_gb / retrieve_time if retrieve_time > 0 else 0,
                    'performance_stats': perf_stats,
                    'success': True
                }
                
                del tensor
                repo.clear()
                gc.collect()
                
            except Exception as e:
                perf_stats = self.profiler.stop_monitoring()
                results[f"{size[0]}x{size[1]}"] = {
                    'tensor_size_gb': tensor_size_gb,
                    'success': False,
                    'error': str(e),
                    'performance_stats': perf_stats
                }
        
        self.test_results['extreme_tensor_sizes'] = results
        return results
    
    def test_deduplication_performance_scaling(self):
        """Test 8: Deduplication performance with many similar weights."""
        print("\n=== Test 8: Deduplication Performance Scaling ===")
        
        results = {}
        
        # Test different numbers of similar weights
        similarity_test_sizes = [100, 500, 1000, 2000, 5000]
        
        for num_weights in similarity_test_sizes:
            if num_weights > 5000:  # Skip very large tests
                continue
                
            print(f"Testing deduplication with {num_weights} similar weights")
            
            self.profiler.reset()
            self.profiler.start_monitoring()
            
            # Create base tensor
            base_tensor = self._create_test_tensor((200, 200), "dedup_base")
            
            # Create many similar tensors with different similarity levels
            similar_tensors = []
            for i in range(num_weights):
                # Vary similarity to create realistic clustering patterns
                similarity = 0.90 + (i % 100) / 1000  # 90% to 99% similarity
                tensor = self._create_similar_tensor(
                    base_tensor, 
                    similarity=similarity,
                    name=f"dedup_tensor_{i}"
                )
                similar_tensors.append(tensor)
            
            self.profiler.sample_resources()
            
            # Test deduplication
            deduplicator = Deduplicator(similarity_threshold=0.95)
            
            dedup_start = time.perf_counter()
            for tensor in similar_tensors:
                deduplicator.add_weight(tensor)
            dedup_time = time.perf_counter() - dedup_start
            
            self.profiler.sample_resources()
            
            # Test repository storage with deduplication
            repo = self.repositories['high_similarity']
            repo.clear()
            
            repo_start = time.perf_counter()
            for i, tensor in enumerate(similar_tensors):
                repo.store_weight(tensor, f"dedup_repo_test_{i}")
            repo_time = time.perf_counter() - repo_start
            
            perf_stats = self.profiler.stop_monitoring()
            
            # Analyze deduplication effectiveness
            unique_weights = len(deduplicator.weight_index)
            deduplication_ratio = num_weights / unique_weights if unique_weights > 0 else 1
            
            # Measure storage efficiency
            repo_size_mb = 0
            if repo.store.file_path.exists():
                repo_size_mb = repo.store.file_path.stat().st_size / 1024 / 1024
            
            naive_size_mb = sum(tensor.data.nbytes for tensor in similar_tensors) / 1024 / 1024
            storage_efficiency = naive_size_mb / repo_size_mb if repo_size_mb > 0 else 1
            
            results[f"weights_{num_weights}"] = {
                'num_weights': num_weights,
                'unique_weights': unique_weights,
                'deduplication_ratio': deduplication_ratio,
                'dedup_time_seconds': dedup_time,
                'repo_time_seconds': repo_time,
                'weights_per_second_dedup': num_weights / dedup_time if dedup_time > 0 else 0,
                'weights_per_second_repo': num_weights / repo_time if repo_time > 0 else 0,
                'storage_efficiency_ratio': storage_efficiency,
                'repo_size_mb': repo_size_mb,
                'naive_size_mb': naive_size_mb,
                'performance_stats': perf_stats
            }
            
            del similar_tensors, deduplicator
            repo.clear()
            gc.collect()
        
        self.test_results['deduplication_performance_scaling'] = results
        return results
    
    def test_cli_command_performance(self):
        """Test 9: CLI command performance benchmarks."""
        print("\n=== Test 9: CLI Command Performance ===")
        
        # This test would benchmark CLI commands, but since we're in a test environment,
        # we'll simulate the core operations that CLI commands perform
        
        self.profiler.reset()
        self.profiler.start_monitoring()
        
        repo = self.repositories['standard']
        repo.clear()
        
        # Simulate common CLI operations
        operations = {}
        
        # 1. Repository initialization (already done, but time similar operation)
        init_start = time.perf_counter()
        temp_repo_path = self.temp_dir / "cli_test_repo"
        temp_repo_path.mkdir()
        temp_store = HDF5Store(temp_repo_path / "weights.h5")
        temp_repo = Repository(temp_repo_path, store=temp_store)
        init_time = time.perf_counter() - init_start
        operations['init'] = init_time
        
        # 2. Adding weights (simulating 'coral add')
        tensor = self._create_test_tensor((500, 500), "cli_test_tensor")
        add_start = time.perf_counter()
        temp_repo.store_weight(tensor, "CLI add simulation")
        add_time = time.perf_counter() - add_start
        operations['add'] = add_time
        
        # 3. Listing weights (simulating 'coral list')
        list_start = time.perf_counter()
        weight_list = temp_repo.store.list_weights()
        list_time = time.perf_counter() - list_start
        operations['list'] = list_time
        
        # 4. Status check (simulating 'coral status')
        status_start = time.perf_counter()
        # Simulate status check operations
        weight_count = len(weight_list)
        repo_size = temp_repo.store.file_path.stat().st_size if temp_repo.store.file_path.exists() else 0
        status_time = time.perf_counter() - status_start
        operations['status'] = status_time
        
        perf_stats = self.profiler.stop_monitoring()
        
        results = {
            'operations': operations,
            'total_time': sum(operations.values()),
            'performance_stats': perf_stats
        }
        
        self.test_results['cli_command_performance'] = results
        return results
    
    def test_garbage_collection_efficiency(self):
        """Test 10: Garbage collection and cleanup efficiency."""
        print("\n=== Test 10: Garbage Collection Efficiency ===")
        
        self.profiler.reset()
        self.profiler.start_monitoring()
        
        repo = self.repositories['standard']
        repo.clear()
        
        # Create many objects and then delete them to test GC
        initial_objects = len(gc.get_objects())
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Phase 1: Create many objects
        tensors = []
        for i in range(100):
            tensor = self._create_test_tensor((100, 100), f"gc_test_{i}")
            tensors.append(tensor)
            repo.store_weight(tensor, f"GC test {i}")
        
        after_creation_objects = len(gc.get_objects())
        after_creation_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        self.profiler.sample_resources()
        
        # Phase 2: Delete objects
        del tensors
        
        # Phase 3: Force garbage collection
        gc_start = time.perf_counter()
        collected = gc.collect()
        gc_time = time.perf_counter() - gc_start
        
        after_gc_objects = len(gc.get_objects())
        after_gc_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Phase 4: Clear repository
        repo_clear_start = time.perf_counter()
        repo.clear()
        repo_clear_time = time.perf_counter() - repo_clear_start
        
        final_objects = len(gc.get_objects())
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        perf_stats = self.profiler.stop_monitoring()
        
        results = {
            'object_counts': {
                'initial': initial_objects,
                'after_creation': after_creation_objects,
                'after_gc': after_gc_objects,
                'final': final_objects
            },
            'memory_usage_mb': {
                'initial': initial_memory,
                'after_creation': after_creation_memory,
                'after_gc': after_gc_memory,
                'final': final_memory
            },
            'gc_collected_objects': collected,
            'gc_time_seconds': gc_time,
            'repo_clear_time_seconds': repo_clear_time,
            'memory_recovered_mb': after_creation_memory - final_memory,
            'objects_cleaned_up': after_creation_objects - final_objects,
            'performance_stats': perf_stats
        }
        
        self.test_results['garbage_collection_efficiency'] = results
        return results
    
    def test_training_integration_performance(self):
        """Test 11: Training integration with long-running operations."""
        print("\n=== Test 11: Training Integration Performance ===")
        
        self.profiler.reset()
        self.profiler.start_monitoring()
        
        # Create checkpoint manager
        checkpoint_dir = self.temp_dir / "checkpoints"
        checkpoint_dir.mkdir()
        
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=10,
            save_frequency=5
        )
        
        # Simulate training loop with checkpointing
        training_times = []
        checkpoint_times = []
        
        base_tensor = self._create_test_tensor((1000, 1000), "training_base")
        
        for epoch in range(50):  # Simulate 50 training epochs
            # Simulate training step
            training_start = time.perf_counter()
            
            # Create updated weights (simulating training)
            updated_tensor = self._create_similar_tensor(
                base_tensor, 
                similarity=0.99,
                name=f"epoch_{epoch}_weights"
            )
            
            training_time = time.perf_counter() - training_start
            training_times.append(training_time)
            
            # Checkpoint every 5 epochs
            if epoch % 5 == 0:
                checkpoint_start = time.perf_counter()
                
                # Save checkpoint
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.npz"
                np.savez(checkpoint_path, weights=updated_tensor.data)
                
                checkpoint_time = time.perf_counter() - checkpoint_start
                checkpoint_times.append(checkpoint_time)
                
                self.profiler.sample_resources()
        
        perf_stats = self.profiler.stop_monitoring()
        
        results = {
            'num_epochs': 50,
            'avg_training_time_per_epoch': np.mean(training_times),
            'avg_checkpoint_time': np.mean(checkpoint_times) if checkpoint_times else 0,
            'total_training_time': sum(training_times),
            'total_checkpoint_time': sum(checkpoint_times),
            'checkpoint_overhead_percent': (sum(checkpoint_times) / sum(training_times)) * 100 if training_times else 0,
            'checkpoints_created': len(checkpoint_times),
            'performance_stats': perf_stats
        }
        
        self.test_results['training_integration_performance'] = results
        return results
    
    def test_clustering_memory_profiling(self):
        """Test 12: Memory usage profiling during clustering operations."""
        print("\n=== Test 12: Clustering Memory Profiling ===")
        
        self.profiler.reset()
        self.profiler.start_monitoring()
        
        # Create diverse set of tensors for clustering
        base_tensors = []
        for i in range(10):
            base_tensor = self._create_test_tensor((200, 200), f"cluster_base_{i}")
            base_tensors.append(base_tensor)
        
        # Create clusters of similar tensors
        all_tensors = []
        for base_tensor in base_tensors:
            cluster_tensors = [base_tensor]
            for j in range(20):  # 20 similar tensors per base
                similar_tensor = self._create_similar_tensor(
                    base_tensor, 
                    similarity=0.95 + np.random.random() * 0.04,
                    name=f"cluster_{len(all_tensors)}_{j}"
                )
                cluster_tensors.append(similar_tensor)
            all_tensors.extend(cluster_tensors)
        
        self.profiler.sample_resources()
        
        # Test clustering with memory profiling
        repo = self.repositories['high_similarity']
        repo.clear()
        
        # Store all tensors
        store_start = time.perf_counter()
        for i, tensor in enumerate(all_tensors):
            repo.store_weight(tensor, f"clustering_memory_test_{i}")
            
            # Sample memory every 50 tensors
            if i % 50 == 0:
                self.profiler.sample_resources()
        
        store_time = time.perf_counter() - store_start
        
        # Analyze clustering effectiveness
        weight_ids = list(repo.store.list_weights().keys())
        unique_weights = len(weight_ids)
        total_tensors = len(all_tensors)
        clustering_ratio = total_tensors / unique_weights if unique_weights > 0 else 1
        
        perf_stats = self.profiler.stop_monitoring()
        
        results = {
            'total_tensors': total_tensors,
            'unique_weights_stored': unique_weights,
            'clustering_ratio': clustering_ratio,
            'store_time_seconds': store_time,
            'tensors_per_second': total_tensors / store_time if store_time > 0 else 0,
            'memory_efficiency': clustering_ratio,
            'performance_stats': perf_stats
        }
        
        self.test_results['clustering_memory_profiling'] = results
        repo.clear()
        return results
    
    def generate_performance_report(self):
        """Generate comprehensive performance report."""
        report = {
            'system_info': {
                'cpu_count': self.cpu_count,
                'total_memory_gb': self.total_memory_gb,
                'python_version': sys.version,
                'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'test_results': self.test_results,
            'summary': self._generate_summary(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save report to file
        report_path = self.temp_dir / "performance_qa_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nPerformance report saved to: {report_path}")
        return report
    
    def _generate_summary(self):
        """Generate summary of test results."""
        summary = {
            'total_tests': len(self.test_results),
            'memory_issues': [],
            'performance_issues': [],
            'scalability_issues': [],
            'recommendations': []
        }
        
        # Analyze memory usage patterns
        if 'memory_leak_detection' in self.test_results:
            memory_test = self.test_results['memory_leak_detection']
            if memory_test.get('potential_memory_leak', False):
                summary['memory_issues'].append(
                    f"Potential memory leak detected: {memory_test['memory_growth_trend_mb_per_iteration']:.3f} MB per iteration"
                )
        
        # Analyze performance bottlenecks
        if 'deep_commit_history_performance' in self.test_results:
            commit_test = self.test_results['deep_commit_history_performance']
            if commit_test.get('performance_degradation', False):
                summary['performance_issues'].append(
                    f"Performance degradation with commit history: {commit_test['commit_time_trend']:.6f} seconds increase per commit"
                )
        
        # Analyze scalability
        if 'clustering_performance_scaling' in self.test_results:
            clustering_test = self.test_results['clustering_performance_scaling']
            # Check if performance scales linearly
            tensor_counts = []
            processing_times = []
            for key, value in clustering_test.items():
                if key.startswith('tensors_'):
                    tensor_counts.append(value['num_tensors'])
                    processing_times.append(value['dedup_time_seconds'])
            
            if len(tensor_counts) > 2:
                # Check if performance degrades non-linearly
                efficiency_ratios = [t/n for t, n in zip(processing_times, tensor_counts)]
                if max(efficiency_ratios) / min(efficiency_ratios) > 2:
                    summary['scalability_issues'].append(
                        "Non-linear performance degradation detected in clustering"
                    )
        
        return summary
    
    def _generate_recommendations(self):
        """Generate performance recommendations based on test results."""
        recommendations = []
        
        # Memory recommendations
        if 'memory_usage_patterns' in self.test_results:
            memory_patterns = self.test_results['memory_usage_patterns']
            high_memory_ops = [k for k, v in memory_patterns.items() 
                             if v.get('memory_efficiency', 1) > 5]
            if high_memory_ops:
                recommendations.append(
                    f"Consider memory optimization for operations with large tensors: {high_memory_ops}"
                )
        
        # Concurrency recommendations
        if 'hdf5_concurrent_stress' in self.test_results:
            concurrent_results = self.test_results['hdf5_concurrent_stress']
            failed_ops = [k for k, v in concurrent_results.items() 
                         if v.get('success_rate', 1) < 0.9]
            if failed_ops:
                recommendations.append(
                    f"Improve concurrency handling for thread levels: {failed_ops}"
                )
        
        # Performance recommendations
        if 'delta_encoding_cpu_profiling' in self.test_results:
            delta_results = self.test_results['delta_encoding_cpu_profiling']
            slow_strategies = [k for k, v in delta_results.items() 
                             if v.get('encode_throughput_tensors_per_sec', 0) < 1]
            if slow_strategies:
                recommendations.append(
                    f"Consider optimizing delta encoding strategies: {slow_strategies}"
                )
        
        return recommendations
    
    def run_all_tests(self):
        """Run all performance tests."""
        print("Starting Coral Performance QA Test Suite")
        print(f"System: {self.cpu_count} CPUs, {self.total_memory_gb:.2f}GB RAM")
        print("="*60)
        
        try:
            self.setup()
            
            # Run all tests
            self.test_memory_usage_patterns()
            self.test_memory_leak_detection()
            self.test_hdf5_concurrent_stress()
            self.test_clustering_performance_scaling()
            self.test_delta_encoding_cpu_profiling()
            self.test_deep_commit_history_performance()
            self.test_extreme_tensor_sizes()
            self.test_deduplication_performance_scaling()
            self.test_cli_command_performance()
            self.test_garbage_collection_efficiency()
            self.test_training_integration_performance()
            self.test_clustering_memory_profiling()
            
            # Generate report
            report = self.generate_performance_report()
            
            print("\n" + "="*60)
            print("PERFORMANCE QA TEST SUITE COMPLETE")
            print("="*60)
            
            # Print summary
            summary = report['summary']
            print(f"Total tests: {summary['total_tests']}")
            print(f"Memory issues: {len(summary['memory_issues'])}")
            print(f"Performance issues: {len(summary['performance_issues'])}")
            print(f"Scalability issues: {len(summary['scalability_issues'])}")
            
            if summary['memory_issues']:
                print("\nMemory Issues:")
                for issue in summary['memory_issues']:
                    print(f"  - {issue}")
            
            if summary['performance_issues']:
                print("\nPerformance Issues:")
                for issue in summary['performance_issues']:
                    print(f"  - {issue}")
            
            if summary['scalability_issues']:
                print("\nScalability Issues:")
                for issue in summary['scalability_issues']:
                    print(f"  - {issue}")
            
            return report
            
        finally:
            self.teardown()


def main():
    """Main entry point for performance testing."""
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    suite = PerformanceQATestSuite()
    report = suite.run_all_tests()
    
    return report


if __name__ == "__main__":
    main()