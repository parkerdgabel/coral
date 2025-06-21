#!/usr/bin/env python3
"""
Comprehensive demonstration of Coral ML profiling and scalability testing tools.

This example shows how to use the profiling, scalability, and stress testing
frameworks to analyze Coral's performance characteristics.
"""

import tempfile
import time
from pathlib import Path

import numpy as np

from coral.core.weight_tensor import WeightTensor
from coral.storage.hdf5_store import HDF5Store
from coral.version_control.repository import Repository

# Import the new benchmarking tools
from coral.benchmarks.profiling import (
    CPUProfiler,
    MemoryProfiler,
    IOProfiler,
    ComprehensiveProfiler,
    profile_function
)
from coral.benchmarks.scalability import (
    ScalabilityConfig,
    ScalabilityTestSuite
)
from coral.benchmarks.stress_tests import (
    StressTestConfig,
    ComprehensiveStressTester
)


def create_test_model(size_mb: float = 10.0) -> list[WeightTensor]:
    """Create a test model with specified size.
    
    Args:
        size_mb: Target size in MB
        
    Returns:
        List of WeightTensor objects
    """
    target_bytes = int(size_mb * 1024 * 1024)
    weights = []
    current_bytes = 0
    layer_idx = 0
    
    while current_bytes < target_bytes:
        # Create varied layer sizes
        if layer_idx % 3 == 0:
            shape = (256, 256)
        elif layer_idx % 3 == 1:
            shape = (128, 512)
        else:
            shape = (512,)
            
        weight_data = np.random.randn(*shape).astype(np.float32)
        weight_tensor = WeightTensor(
            data=weight_data,
            name=f"layer_{layer_idx}",
            layer_type="linear" if len(shape) == 2 else "bias",
            model_name="demo_model"
        )
        
        weights.append(weight_tensor)
        current_bytes += weight_tensor.data.nbytes
        layer_idx += 1
        
    return weights


@profile_function(profiler_type="cpu")
def cpu_intensive_operation():
    """Example CPU-intensive operation for profiling."""
    print("Performing CPU-intensive operation...")
    result = 0
    for i in range(1000000):
        result += i * i
    return result


@profile_function(profiler_type="memory")
def memory_intensive_operation():
    """Example memory-intensive operation for profiling."""
    print("Performing memory-intensive operation...")
    arrays = []
    for i in range(50):
        arr = np.random.rand(1000, 1000)
        arrays.append(arr)
    return arrays


def demonstrate_profiling():
    """Demonstrate profiling capabilities."""
    print("\n" + "="*60)
    print("PROFILING DEMONSTRATION")
    print("="*60)
    
    # Create temporary repository for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        store = HDF5Store(repo_path / "profile_test.h5")
        repository = Repository(store)
        
        # Generate test model
        test_model = create_test_model(5.0)  # 5MB model
        
        print("\n1. CPU Profiling")
        print("-" * 30)
        
        cpu_profiler = CPUProfiler(Path("profiling_results"))
        
        with cpu_profiler.profile("coral_operations_cpu"):
            # Store model weights
            for weight in test_model[:10]:  # First 10 weights
                repository.store_weight(weight)
            repository.commit("CPU profiling test")
            
            # Perform CPU-intensive operation
            cpu_intensive_operation()
            
        print("✓ CPU profiling completed")
        
        print("\n2. Memory Profiling")
        print("-" * 30)
        
        memory_profiler = MemoryProfiler(Path("profiling_results"))
        
        with memory_profiler.profile("coral_operations_memory"):
            # Perform memory-intensive operations
            memory_intensive_operation()
            
            # Store remaining weights
            for weight in test_model[10:]:
                repository.store_weight(weight)
            repository.commit("Memory profiling test")
            
        print("✓ Memory profiling completed")
        
        print("\n3. I/O Profiling")
        print("-" * 30)
        
        io_profiler = IOProfiler(Path("profiling_results"))
        
        with io_profiler.profile("coral_operations_io"):
            # Create multiple commits (I/O intensive)
            for i in range(5):
                batch_weights = create_test_model(1.0)  # 1MB batches
                for weight in batch_weights:
                    weight.model_name = f"io_test_model_{i}"
                    repository.store_weight(weight)
                repository.commit(f"I/O test batch {i}")
                
        print("✓ I/O profiling completed")
        
        print("\n4. Comprehensive Profiling")
        print("-" * 30)
        
        comprehensive_profiler = ComprehensiveProfiler(Path("profiling_results"))
        
        with comprehensive_profiler.profile_all("coral_comprehensive"):
            # Mixed workload
            cpu_result = cpu_intensive_operation()
            memory_arrays = memory_intensive_operation()
            
            # Repository operations
            final_model = create_test_model(2.0)
            for weight in final_model:
                weight.model_name = "comprehensive_test"
                repository.store_weight(weight)
            repository.commit("Comprehensive profiling test")
            
        print("✓ Comprehensive profiling completed")
        
    print("\n✅ All profiling demonstrations completed!")
    print("📊 Results saved to: ./profiling_results/")


def demonstrate_scalability():
    """Demonstrate scalability testing."""
    print("\n" + "="*60)
    print("SCALABILITY TESTING DEMONSTRATION")
    print("="*60)
    
    # Configure scalability tests
    config = ScalabilityConfig(
        test_name="coral_scalability_demo",
        max_models=500,  # Reduced for demo
        repository_sizes=[10, 50, 100, 250, 500],
        model_size_mb=2.0,  # Smaller models for faster testing
        max_concurrent_users=20,
        test_duration_seconds=30,  # Shorter duration for demo
        output_dir=Path("scalability_results")
    )
    
    print(f"Configuration:")
    print(f"  - Repository sizes: {config.repository_sizes}")
    print(f"  - Model size: {config.model_size_mb}MB")
    print(f"  - Max concurrent users: {config.max_concurrent_users}")
    print(f"  - Test duration: {config.test_duration_seconds}s")
    
    # Run scalability tests
    test_suite = ScalabilityTestSuite(config)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\n🚀 Running scalability tests...")
        start_time = time.time()
        
        results = test_suite.run_comprehensive_scalability_tests()
        
        end_time = time.time()
        print(f"✓ Scalability tests completed in {end_time - start_time:.1f}s")
        
        # Print summary
        if 'summary' in results:
            summary = results['summary']
            print(f"\n📊 Test Summary:")
            print(f"  - Total tests: {summary.get('total_tests_run', 0)}")
            print(f"  - Successful tests: {summary.get('successful_tests', 0)}")
            print(f"  - Bottlenecks found: {summary.get('bottlenecks_found', 0)}")
            
            if summary.get('scaling_limits_identified'):
                print(f"  - Scaling limits: {summary['scaling_limits_identified'][:3]}")
                
        # Print repository scaling results
        if 'repository_scaling' in results:
            print(f"\n📈 Repository Scaling Results:")
            for size, metrics in results['repository_scaling'].items():
                print(f"  - {size} models: "
                      f"{metrics.throughput_ops_per_sec:.1f} ops/sec, "
                      f"{metrics.memory_usage_mb:.1f}MB memory, "
                      f"{metrics.error_rate*100:.1f}% errors")
                      
    print(f"\n✅ Scalability testing completed!")
    print(f"📊 Results saved to: ./scalability_results/")


def demonstrate_stress_testing():
    """Demonstrate stress testing."""
    print("\n" + "="*60)
    print("STRESS TESTING DEMONSTRATION")
    print("="*60)
    
    # Configure stress tests
    config = StressTestConfig(
        test_name="coral_stress_demo",
        duration_seconds=30,  # Short duration for demo
        max_operations_per_second=100,
        max_concurrent_threads=10,
        max_concurrent_processes=3,
        memory_limit_gb=1.0,  # Lower limits for demo
        disk_space_limit_gb=0.5,
        error_injection_rate=0.02,  # 2% error rate
        output_dir=Path("stress_test_results")
    )
    
    print(f"Configuration:")
    print(f"  - Duration: {config.duration_seconds}s")
    print(f"  - Max ops/sec: {config.max_operations_per_second}")
    print(f"  - Concurrent threads: {config.max_concurrent_threads}")
    print(f"  - Concurrent processes: {config.max_concurrent_processes}")
    print(f"  - Error injection rate: {config.error_injection_rate*100:.0f}%")
    
    # Run stress tests
    stress_tester = ComprehensiveStressTester(config)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\n🔥 Running stress tests...")
        start_time = time.time()
        
        results = stress_tester.run_all_stress_tests(Path(temp_dir))
        
        end_time = time.time()
        print(f"✓ Stress tests completed in {end_time - start_time:.1f}s")
        
        # Print results summary
        print(f"\n📊 Stress Test Results:")
        for test_name, metrics in results.items():
            if hasattr(metrics, 'operations_attempted'):
                success_rate = (1 - metrics.error_rate) * 100
                print(f"  - {test_name.replace('_', ' ').title()}:")
                print(f"    * Operations: {metrics.operations_attempted}")
                print(f"    * Success rate: {success_rate:.1f}%")
                print(f"    * Peak memory: {metrics.peak_memory_mb:.1f}MB")
                print(f"    * Peak CPU: {metrics.peak_cpu_percent:.1f}%")
                
                if metrics.memory_leaks_detected:
                    print(f"    * ⚠️ Memory leak detected")
                if metrics.error_rate > 0.1:
                    print(f"    * ⚠️ High error rate: {metrics.error_rate*100:.1f}%")
                    
    print(f"\n✅ Stress testing completed!")
    print(f"📊 Results saved to: ./stress_test_results/")


def demonstrate_decorator_usage():
    """Demonstrate profiling decorator usage."""
    print("\n" + "="*60)
    print("PROFILING DECORATOR DEMONSTRATION")
    print("="*60)
    
    print("Profiling functions using decorators...")
    
    # These functions are already decorated above
    print("\n1. CPU-intensive function (decorated with @profile_function)")
    cpu_result = cpu_intensive_operation()
    
    print("\n2. Memory-intensive function (decorated with @profile_function)")
    memory_result = memory_intensive_operation()
    
    print(f"\n✅ Decorator profiling completed!")
    print(f"📊 Individual function profiles saved to: ./profiling_results/")


def main():
    """Main demonstration function."""
    print("🔬 Coral ML Profiling and Scalability Testing Demonstration")
    print("=" * 70)
    
    print("\nThis demonstration showcases Coral's comprehensive performance")
    print("analysis tools including profiling, scalability, and stress testing.")
    
    try:
        # 1. Profiling demonstration
        demonstrate_profiling()
        
        # 2. Decorator usage demonstration
        demonstrate_decorator_usage()
        
        # 3. Scalability testing demonstration
        demonstrate_scalability()
        
        # 4. Stress testing demonstration
        demonstrate_stress_testing()
        
        print("\n" + "="*70)
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        print("\n📁 Generated Files:")
        print("  - ./profiling_results/ - Detailed profiling reports")
        print("  - ./scalability_results/ - Scalability test results")
        print("  - ./stress_test_results/ - Stress test reports")
        
        print("\n💡 Next Steps:")
        print("  - Review profiling reports to identify bottlenecks")
        print("  - Analyze scalability results for capacity planning")
        print("  - Examine stress test results for stability assessment")
        print("  - Use findings to optimize Coral configurations")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()