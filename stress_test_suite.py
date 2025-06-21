#!/usr/bin/env python3
"""
Coral Stress Test Suite

Specialized stress testing for finding system breaking points,
memory limits, and performance boundaries under extreme conditions.

This suite pushes the system to its limits to identify:
- Maximum tensor sizes before OOM
- Concurrency limits before failures
- Repository size limits
- Memory pressure handling
- Recovery from extreme conditions
"""

import gc
import json
import multiprocessing
import os
import psutil
import resource
import signal
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.core.deduplicator import Deduplicator
from coral.storage.hdf5_store import HDF5Store
from coral.version_control.repository import Repository


class SystemMonitor:
    """Monitor system resources during stress tests."""
    
    def __init__(self):
        self.monitoring = False
        self.samples = []
        self.start_time = None
        
    def start(self):
        """Start monitoring."""
        self.monitoring = True
        self.start_time = time.time()
        self.samples = []
        
    def sample(self):
        """Take a resource sample."""
        if not self.monitoring:
            return
            
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            sample = {
                'timestamp': time.time() - self.start_time,
                'memory_rss_mb': memory_info.rss / 1024 / 1024,
                'memory_vms_mb': memory_info.vms / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'open_files': len(process.open_files()),
                'threads': process.num_threads(),
                'system_memory_percent': psutil.virtual_memory().percent,
                'system_memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024
            }
            
            self.samples.append(sample)
            
        except Exception as e:
            # Don't let monitoring failures break the stress test
            pass
    
    def stop(self):
        """Stop monitoring and return results."""
        self.monitoring = False
        
        if not self.samples:
            return {}
        
        # Calculate statistics
        memory_values = [s['memory_rss_mb'] for s in self.samples]
        cpu_values = [s['cpu_percent'] for s in self.samples if s['cpu_percent'] is not None]
        
        return {
            'duration_seconds': time.time() - self.start_time,
            'sample_count': len(self.samples),
            'peak_memory_mb': max(memory_values),
            'avg_memory_mb': sum(memory_values) / len(memory_values),
            'peak_cpu_percent': max(cpu_values) if cpu_values else 0,
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            'max_open_files': max(s['open_files'] for s in self.samples),
            'max_threads': max(s['threads'] for s in self.samples),
            'min_system_memory_available_gb': min(s['system_memory_available_gb'] for s in self.samples),
            'samples': self.samples
        }


@contextmanager
def timeout_context(seconds):
    """Context manager for operation timeout."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set up signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class StressTestSuite:
    """Comprehensive stress testing suite."""
    
    def __init__(self):
        self.temp_dir = None
        self.monitor = SystemMonitor()
        self.results = {}
        
        # System limits
        self.max_memory_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
        self.cpu_count = multiprocessing.cpu_count()
        
        # Test parameters
        self.stress_levels = {
            'light': 0.1,    # 10% of system resources
            'moderate': 0.3, # 30% of system resources
            'heavy': 0.6,    # 60% of system resources
            'extreme': 0.9   # 90% of system resources
        }
        
    def setup(self):
        """Setup stress test environment."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="coral_stress_"))
        print(f"Stress test directory: {self.temp_dir}")
        
        # Set memory limits to prevent system crash
        try:
            # Limit to 80% of available memory
            max_memory_bytes = int(self.max_memory_gb * 0.8 * 1024 * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
            print(f"Set memory limit to {max_memory_bytes / 1024 / 1024 / 1024:.1f}GB")
        except Exception as e:
            print(f"Could not set memory limit: {e}")
    
    def teardown(self):
        """Cleanup stress test environment."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
        
        # Force cleanup
        gc.collect()
    
    def stress_test_maximum_tensor_size(self):
        """Find maximum tensor size before OOM."""
        print("\n=== Stress Test: Maximum Tensor Size ===")
        
        results = {
            'max_successful_size': None,
            'max_successful_memory_mb': 0,
            'oom_size': None,
            'test_sizes': []
        }
        
        # Start with reasonable size and increase exponentially
        base_size = 1000
        multiplier = 1.5
        current_size = base_size
        
        while True:
            tensor_size_gb = (current_size * current_size * 4) / (1024**3)  # float32
            
            # Stop if tensor would use more than 70% of available memory
            if tensor_size_gb > self.max_memory_gb * 0.7:
                print(f"Stopping at size {current_size}x{current_size} ({tensor_size_gb:.2f}GB) - would exceed memory limit")
                break
            
            print(f"Testing tensor size: {current_size}x{current_size} ({tensor_size_gb:.2f}GB)")
            
            self.monitor.start()
            
            try:
                with timeout_context(120):  # 2 minute timeout
                    # Create tensor
                    data = np.random.randn(current_size, current_size).astype(np.float32)
                    
                    metadata = WeightMetadata(
                        name=f"stress_tensor_{current_size}",
                        shape=data.shape,
                        dtype=data.dtype
                    )
                    
                    tensor = WeightTensor(data=data, metadata=metadata)
                    
                    # Try to store it
                    repo_path = self.temp_dir / f"stress_repo_{current_size}"
                    repo_path.mkdir()
                    store = HDF5Store(repo_path / "weights.h5")
                    repo = Repository(repo_path, store=store)
                    
                    repo.store_weight(tensor, f"Stress test {current_size}")
                    
                    # Success - record metrics
                    monitor_results = self.monitor.stop()
                    
                    results['max_successful_size'] = (current_size, current_size)
                    results['max_successful_memory_mb'] = monitor_results.get('peak_memory_mb', 0)
                    results['test_sizes'].append({
                        'size': (current_size, current_size),
                        'size_gb': tensor_size_gb,
                        'success': True,
                        'peak_memory_mb': monitor_results.get('peak_memory_mb', 0),
                        'duration_seconds': monitor_results.get('duration_seconds', 0)
                    })
                    
                    # Clean up
                    del tensor, data
                    repo.store.close()
                    del repo, store
                    gc.collect()
                    
                    # Increase size for next iteration
                    current_size = int(current_size * multiplier)
                    
            except (MemoryError, TimeoutError, Exception) as e:
                monitor_results = self.monitor.stop()
                
                print(f"Failed at size {current_size}x{current_size}: {type(e).__name__}: {e}")
                
                results['oom_size'] = (current_size, current_size)
                results['test_sizes'].append({
                    'size': (current_size, current_size),
                    'size_gb': tensor_size_gb,
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
                
                break
        
        self.results['maximum_tensor_size'] = results
        return results
    
    def stress_test_concurrent_limit(self):
        """Find maximum concurrent operations before failures."""
        print("\n=== Stress Test: Concurrent Operation Limit ===")
        
        results = {
            'max_successful_threads': 0,
            'failure_thread_count': None,
            'test_results': []
        }
        
        # Start with reasonable concurrency and increase
        base_tensor = self._create_test_tensor((500, 500))
        
        for num_threads in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            if num_threads > self.cpu_count * 8:  # Stop at 8x CPU count
                break
                
            print(f"Testing {num_threads} concurrent threads")
            
            self.monitor.start()
            
            # Create repository for this test
            repo_path = self.temp_dir / f"concurrent_stress_{num_threads}"
            repo_path.mkdir()
            store = HDF5Store(repo_path / "weights.h5")
            repo = Repository(repo_path, store=store)
            
            success_count = 0
            errors = []
            
            def worker_function(worker_id):
                try:
                    # Create unique tensor
                    data = np.random.randn(500, 500).astype(np.float32) * (worker_id + 1)
                    metadata = WeightMetadata(
                        name=f"concurrent_worker_{worker_id}",
                        shape=data.shape,
                        dtype=data.dtype
                    )
                    tensor = WeightTensor(data=data, metadata=metadata)
                    
                    # Store tensor
                    repo.store_weight(tensor, f"Concurrent worker {worker_id}")
                    
                    return {'worker_id': worker_id, 'success': True, 'error': None}
                    
                except Exception as e:
                    return {'worker_id': worker_id, 'success': False, 'error': str(e)}
            
            try:
                with timeout_context(180):  # 3 minute timeout
                    with ThreadPoolExecutor(max_workers=num_threads) as executor:
                        futures = [executor.submit(worker_function, i) for i in range(num_threads)]
                        worker_results = [future.result(timeout=60) for future in as_completed(futures, timeout=120)]
                
                monitor_results = self.monitor.stop()
                
                # Analyze results
                success_count = sum(1 for r in worker_results if r['success'])
                errors = [r['error'] for r in worker_results if not r['success']]
                
                success_rate = success_count / num_threads
                
                results['test_results'].append({
                    'num_threads': num_threads,
                    'success_count': success_count,
                    'success_rate': success_rate,
                    'errors': errors,
                    'peak_memory_mb': monitor_results.get('peak_memory_mb', 0),
                    'max_open_files': monitor_results.get('max_open_files', 0),
                    'duration_seconds': monitor_results.get('duration_seconds', 0)
                })
                
                if success_rate >= 0.9:  # 90% success rate threshold
                    results['max_successful_threads'] = num_threads
                else:
                    results['failure_thread_count'] = num_threads
                    print(f"Failed at {num_threads} threads with {success_rate:.1%} success rate")
                    break
                
            except Exception as e:
                monitor_results = self.monitor.stop()
                print(f"Catastrophic failure at {num_threads} threads: {e}")
                
                results['test_results'].append({
                    'num_threads': num_threads,
                    'catastrophic_failure': True,
                    'error': str(e)
                })
                
                results['failure_thread_count'] = num_threads
                break
            
            finally:
                # Clean up
                repo.store.close()
                del repo, store
                gc.collect()
        
        self.results['concurrent_limit'] = results
        return results
    
    def stress_test_repository_size_limit(self):
        """Test repository size limits."""
        print("\n=== Stress Test: Repository Size Limit ===")
        
        results = {
            'max_weights_stored': 0,
            'storage_size_mb': 0,
            'test_phases': []
        }
        
        # Create repository
        repo_path = self.temp_dir / "size_stress_repo"
        repo_path.mkdir()
        store = HDF5Store(repo_path / "weights.h5")
        repo = Repository(repo_path, store=store)
        
        base_tensor = self._create_test_tensor((100, 100))
        
        self.monitor.start()
        
        try:
            phase = 1
            weights_per_phase = 1000
            total_weights = 0
            
            while True:
                print(f"Phase {phase}: Adding {weights_per_phase} weights (total: {total_weights + weights_per_phase})")
                
                phase_start = time.time()
                
                # Add weights in batch
                for i in range(weights_per_phase):
                    # Create variation
                    noise = np.random.randn(*base_tensor.shape) * 0.01
                    data = base_tensor.data + noise.astype(base_tensor.dtype)
                    
                    metadata = WeightMetadata(
                        name=f"size_stress_{total_weights + i}",
                        shape=data.shape,
                        dtype=data.dtype
                    )
                    
                    tensor = WeightTensor(data=data, metadata=metadata)
                    repo.store_weight(tensor, f"Size stress {total_weights + i}")
                    
                    # Sample resources every 100 weights
                    if (total_weights + i) % 100 == 0:
                        self.monitor.sample()
                        
                        # Check if we're running out of space or memory
                        if psutil.virtual_memory().percent > 90:
                            print("Stopping due to high memory usage")
                            raise MemoryError("High system memory usage")
                
                total_weights += weights_per_phase
                phase_time = time.time() - phase_start
                
                # Check storage size
                storage_size_mb = store.file_path.stat().st_size / 1024 / 1024
                
                results['test_phases'].append({
                    'phase': phase,
                    'weights_added': weights_per_phase,
                    'total_weights': total_weights,
                    'storage_size_mb': storage_size_mb,
                    'phase_time_seconds': phase_time,
                    'weights_per_second': weights_per_phase / phase_time
                })
                
                results['max_weights_stored'] = total_weights
                results['storage_size_mb'] = storage_size_mb
                
                print(f"Phase {phase} complete: {total_weights} weights, {storage_size_mb:.1f}MB storage")
                
                # Stop if storage gets too large (> 1GB) or performance degrades significantly
                if storage_size_mb > 1024 or phase_time > 300:  # 5 minute timeout per phase
                    print(f"Stopping due to size ({storage_size_mb:.1f}MB) or time ({phase_time:.1f}s) limits")
                    break
                
                phase += 1
                
                # Increase batch size for later phases
                if phase > 3:
                    weights_per_phase = 2000
                
        except Exception as e:
            print(f"Repository size stress test failed: {e}")
            results['error'] = str(e)
        
        finally:
            monitor_results = self.monitor.stop()
            results['monitor_results'] = monitor_results
            
            repo.store.close()
            del repo, store
            gc.collect()
        
        self.results['repository_size_limit'] = results
        return results
    
    def stress_test_memory_pressure_recovery(self):
        """Test recovery from memory pressure situations."""
        print("\n=== Stress Test: Memory Pressure Recovery ===")
        
        results = {
            'recovery_phases': [],
            'max_memory_mb': 0,
            'recovery_successful': False
        }
        
        self.monitor.start()
        
        try:
            # Phase 1: Create memory pressure
            print("Phase 1: Creating memory pressure...")
            
            large_objects = []
            pressure_size_mb = self.max_memory_gb * 1024 * 0.4  # Use 40% of available memory
            
            while len(large_objects) * 100 < pressure_size_mb:  # 100MB per object
                large_array = np.random.randn(25, 1000, 1000).astype(np.float32)  # ~100MB
                large_objects.append(large_array)
                
                self.monitor.sample()
                
                if psutil.virtual_memory().percent > 85:
                    print("Reached 85% memory usage, stopping pressure creation")
                    break
            
            memory_after_pressure = psutil.virtual_memory().percent
            results['max_memory_mb'] = psutil.Process().memory_info().rss / 1024 / 1024
            
            print(f"Created {len(large_objects)} large objects, memory usage: {memory_after_pressure:.1f}%")
            
            # Phase 2: Try to perform normal operations under pressure
            print("Phase 2: Testing operations under memory pressure...")
            
            repo_path = self.temp_dir / "pressure_recovery_repo"
            repo_path.mkdir()
            store = HDF5Store(repo_path / "weights.h5")
            repo = Repository(repo_path, store=store)
            
            operations_successful = 0
            operations_failed = 0
            
            for i in range(10):
                try:
                    # Try to create and store a tensor
                    tensor = self._create_test_tensor((200, 200), f"pressure_test_{i}")
                    repo.store_weight(tensor, f"Pressure test {i}")
                    operations_successful += 1
                    del tensor
                    
                except Exception as e:
                    operations_failed += 1
                    print(f"Operation {i} failed under pressure: {e}")
                
                self.monitor.sample()
            
            results['recovery_phases'].append({
                'phase': 'under_pressure',
                'operations_successful': operations_successful,
                'operations_failed': operations_failed,
                'memory_percent': psutil.virtual_memory().percent
            })
            
            # Phase 3: Release memory pressure and test recovery
            print("Phase 3: Releasing memory pressure and testing recovery...")
            
            del large_objects
            gc.collect()
            
            memory_after_cleanup = psutil.virtual_memory().percent
            print(f"Memory after cleanup: {memory_after_cleanup:.1f}%")
            
            # Test recovery operations
            recovery_successful = 0
            recovery_failed = 0
            
            for i in range(10):
                try:
                    tensor = self._create_test_tensor((200, 200), f"recovery_test_{i}")
                    repo.store_weight(tensor, f"Recovery test {i}")
                    recovery_successful += 1
                    del tensor
                    
                except Exception as e:
                    recovery_failed += 1
                    print(f"Recovery operation {i} failed: {e}")
                
                self.monitor.sample()
            
            results['recovery_phases'].append({
                'phase': 'after_recovery',
                'operations_successful': recovery_successful,
                'operations_failed': recovery_failed,
                'memory_percent': psutil.virtual_memory().percent
            })
            
            # Determine if recovery was successful
            results['recovery_successful'] = (
                recovery_successful >= operations_successful and
                recovery_failed <= operations_failed
            )
            
            repo.store.close()
            del repo, store
            
        except Exception as e:
            print(f"Memory pressure recovery test failed: {e}")
            results['error'] = str(e)
        
        finally:
            monitor_results = self.monitor.stop()
            results['monitor_results'] = monitor_results
            gc.collect()
        
        self.results['memory_pressure_recovery'] = results
        return results
    
    def stress_test_extreme_similarity_clustering(self):
        """Test clustering with extremely similar tensors."""
        print("\n=== Stress Test: Extreme Similarity Clustering ===")
        
        results = {
            'clustering_phases': [],
            'max_tensors_clustered': 0,
            'deduplication_effectiveness': 0
        }
        
        self.monitor.start()
        
        try:
            base_tensor = self._create_test_tensor((200, 200))
            
            # Create extremely similar tensors (99.9% similarity)
            similar_tensors = [base_tensor]
            
            batch_size = 500
            num_batches = 10
            
            for batch in range(num_batches):
                print(f"Creating similarity batch {batch + 1}/{num_batches}")
                
                batch_tensors = []
                for i in range(batch_size):
                    # Create extremely similar tensor
                    noise_scale = np.std(base_tensor.data) * 0.001  # 0.1% noise
                    noise = np.random.randn(*base_tensor.shape) * noise_scale
                    similar_data = base_tensor.data + noise.astype(base_tensor.dtype)
                    
                    metadata = WeightMetadata(
                        name=f"similar_tensor_{batch}_{i}",
                        shape=similar_data.shape,
                        dtype=similar_data.dtype
                    )
                    
                    tensor = WeightTensor(data=similar_data, metadata=metadata)
                    batch_tensors.append(tensor)
                
                similar_tensors.extend(batch_tensors)
                
                # Test deduplication on current set
                deduplicator = Deduplicator(similarity_threshold=0.999)
                
                dedup_start = time.time()
                for tensor in similar_tensors:
                    deduplicator.add_weight(tensor)
                dedup_time = time.time() - dedup_start
                
                unique_weights = len(deduplicator.weight_index)
                total_weights = len(similar_tensors)
                deduplication_ratio = total_weights / unique_weights if unique_weights > 0 else 1
                
                results['clustering_phases'].append({
                    'batch': batch + 1,
                    'total_tensors': total_weights,
                    'unique_weights': unique_weights,
                    'deduplication_ratio': deduplication_ratio,
                    'dedup_time_seconds': dedup_time,
                    'tensors_per_second': total_weights / dedup_time if dedup_time > 0 else 0
                })
                
                results['max_tensors_clustered'] = total_weights
                results['deduplication_effectiveness'] = deduplication_ratio
                
                self.monitor.sample()
                
                # Clean up deduplicator for next iteration
                del deduplicator
                gc.collect()
                
                # Stop if deduplication becomes too slow
                if dedup_time > 60:  # More than 1 minute
                    print(f"Stopping due to slow deduplication: {dedup_time:.1f}s")
                    break
                
                # Stop if memory usage is too high
                if psutil.virtual_memory().percent > 80:
                    print("Stopping due to high memory usage")
                    break
            
        except Exception as e:
            print(f"Extreme similarity clustering test failed: {e}")
            results['error'] = str(e)
        
        finally:
            monitor_results = self.monitor.stop()
            results['monitor_results'] = monitor_results
            gc.collect()
        
        self.results['extreme_similarity_clustering'] = results
        return results
    
    def _create_test_tensor(self, shape: Tuple[int, ...], name: str = "test_tensor") -> WeightTensor:
        """Create a test tensor."""
        data = np.random.randn(*shape).astype(np.float32)
        metadata = WeightMetadata(
            name=name,
            shape=shape,
            dtype=data.dtype
        )
        return WeightTensor(data=data, metadata=metadata)
    
    def run_all_stress_tests(self):
        """Run all stress tests."""
        print("Starting Coral Stress Test Suite")
        print(f"System: {self.cpu_count} CPUs, {self.max_memory_gb:.2f}GB RAM")
        print("="*60)
        
        try:
            self.setup()
            
            # Run stress tests in order of increasing severity
            self.stress_test_maximum_tensor_size()
            self.stress_test_concurrent_limit()
            self.stress_test_repository_size_limit()
            self.stress_test_extreme_similarity_clustering()
            self.stress_test_memory_pressure_recovery()
            
            # Generate report
            self._generate_stress_report()
            
        finally:
            self.teardown()
    
    def _generate_stress_report(self):
        """Generate comprehensive stress test report."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = self.temp_dir / f"stress_test_report_{timestamp}.json"
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': {
                'cpu_count': self.cpu_count,
                'memory_gb': self.max_memory_gb,
                'python_version': sys.version
            },
            'stress_test_results': self.results,
            'summary': self._generate_stress_summary()
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nStress test report saved: {report_path}")
        self._print_stress_summary(report['summary'])
        
        return report
    
    def _generate_stress_summary(self):
        """Generate summary of stress test results."""
        summary = {
            'breaking_points': {},
            'performance_limits': {},
            'recovery_capability': {},
            'overall_assessment': 'unknown'
        }
        
        # Analyze breaking points
        if 'maximum_tensor_size' in self.results:
            max_tensor = self.results['maximum_tensor_size']
            if max_tensor['max_successful_size']:
                size = max_tensor['max_successful_size']
                summary['breaking_points']['max_tensor_size'] = f"{size[0]}x{size[1]}"
                summary['breaking_points']['max_tensor_memory_mb'] = max_tensor['max_successful_memory_mb']
        
        if 'concurrent_limit' in self.results:
            concurrent = self.results['concurrent_limit']
            summary['breaking_points']['max_concurrent_threads'] = concurrent['max_successful_threads']
        
        if 'repository_size_limit' in self.results:
            repo_size = self.results['repository_size_limit']
            summary['breaking_points']['max_weights_in_repo'] = repo_size['max_weights_stored']
            summary['breaking_points']['max_storage_size_mb'] = repo_size['storage_size_mb']
        
        # Analyze performance limits
        if 'extreme_similarity_clustering' in self.results:
            clustering = self.results['extreme_similarity_clustering']
            summary['performance_limits']['max_tensors_clustered'] = clustering['max_tensors_clustered']
            summary['performance_limits']['deduplication_effectiveness'] = clustering['deduplication_effectiveness']
        
        # Analyze recovery capability
        if 'memory_pressure_recovery' in self.results:
            recovery = self.results['memory_pressure_recovery']
            summary['recovery_capability']['memory_pressure_recovery'] = recovery['recovery_successful']
        
        # Overall assessment
        breaking_points_found = len([v for v in summary['breaking_points'].values() if v is not None])
        if breaking_points_found == 0:
            summary['overall_assessment'] = 'excellent'
        elif breaking_points_found <= 2:
            summary['overall_assessment'] = 'good'
        else:
            summary['overall_assessment'] = 'needs_improvement'
        
        return summary
    
    def _print_stress_summary(self, summary):
        """Print stress test summary."""
        print("\n" + "="*60)
        print("STRESS TEST SUMMARY")
        print("="*60)
        
        print("\n🔥 BREAKING POINTS:")
        for key, value in summary['breaking_points'].items():
            if value is not None:
                print(f"  {key}: {value}")
        
        print("\n⚡ PERFORMANCE LIMITS:")
        for key, value in summary['performance_limits'].items():
            if value is not None:
                print(f"  {key}: {value}")
        
        print("\n🔄 RECOVERY CAPABILITY:")
        for key, value in summary['recovery_capability'].items():
            status = "✅ PASS" if value else "❌ FAIL"
            print(f"  {key}: {status}")
        
        print(f"\n📊 OVERALL ASSESSMENT: {summary['overall_assessment'].upper()}")


def main():
    """Main entry point."""
    suite = StressTestSuite()
    suite.run_all_stress_tests()


if __name__ == "__main__":
    main()