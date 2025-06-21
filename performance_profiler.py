#!/usr/bin/env python3
"""
Coral Performance Profiler

Advanced performance profiling tool for identifying bottlenecks,
memory leaks, and scalability issues in the Coral ML system.

Usage:
    python performance_profiler.py --test all
    python performance_profiler.py --test memory
    python performance_profiler.py --test concurrency
    python performance_profiler.py --test scalability
"""

import argparse
import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tests.test_performance_qa import PerformanceQATestSuite


class AdvancedProfiler:
    """Advanced profiling capabilities for Coral performance analysis."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("performance_reports")
        self.output_dir.mkdir(exist_ok=True)
        
    def profile_function(self, func, *args, **kwargs):
        """Profile a specific function with cProfile."""
        profiler = cProfile.Profile()
        
        # Run function with profiling
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # Analyze results
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative').print_stats(20)
        
        profile_output = stats_stream.getvalue()
        
        return result, profile_output
    
    def profile_memory_intensive_operations(self):
        """Profile memory-intensive operations."""
        print("Profiling memory-intensive operations...")
        
        suite = PerformanceQATestSuite()
        suite.setup()
        
        try:
            # Profile memory usage patterns
            result, profile_output = self.profile_function(
                suite.test_memory_usage_patterns
            )
            
            self._save_profile_report(
                "memory_intensive_operations",
                profile_output,
                result
            )
            
            return result
            
        finally:
            suite.teardown()
    
    def profile_concurrent_operations(self):
        """Profile concurrent operations."""
        print("Profiling concurrent operations...")
        
        suite = PerformanceQATestSuite()
        suite.setup()
        
        try:
            # Profile HDF5 concurrent stress
            result, profile_output = self.profile_function(
                suite.test_hdf5_concurrent_stress
            )
            
            self._save_profile_report(
                "concurrent_operations",
                profile_output,
                result
            )
            
            return result
            
        finally:
            suite.teardown()
    
    def profile_scalability_limits(self):
        """Profile scalability limits."""
        print("Profiling scalability limits...")
        
        suite = PerformanceQATestSuite()
        suite.setup()
        
        try:
            # Profile clustering performance scaling
            result1, profile_output1 = self.profile_function(
                suite.test_clustering_performance_scaling
            )
            
            # Profile deduplication performance scaling
            result2, profile_output2 = self.profile_function(
                suite.test_deduplication_performance_scaling
            )
            
            self._save_profile_report(
                "scalability_clustering",
                profile_output1,
                result1
            )
            
            self._save_profile_report(
                "scalability_deduplication",
                profile_output2,
                result2
            )
            
            return {"clustering": result1, "deduplication": result2}
            
        finally:
            suite.teardown()
    
    def profile_delta_encoding_performance(self):
        """Profile delta encoding performance."""
        print("Profiling delta encoding performance...")
        
        suite = PerformanceQATestSuite()
        suite.setup()
        
        try:
            result, profile_output = self.profile_function(
                suite.test_delta_encoding_cpu_profiling
            )
            
            self._save_profile_report(
                "delta_encoding_performance",
                profile_output,
                result
            )
            
            return result
            
        finally:
            suite.teardown()
    
    def profile_storage_performance(self):
        """Profile storage performance."""
        print("Profiling storage performance...")
        
        suite = PerformanceQATestSuite()
        suite.setup()
        
        try:
            # Profile extreme tensor sizes
            result, profile_output = self.profile_function(
                suite.test_extreme_tensor_sizes
            )
            
            self._save_profile_report(
                "storage_performance",
                profile_output,
                result
            )
            
            return result
            
        finally:
            suite.teardown()
    
    def run_comprehensive_analysis(self):
        """Run comprehensive performance analysis."""
        print("Running comprehensive performance analysis...")
        
        results = {}
        
        # Run all profiling tests
        results['memory'] = self.profile_memory_intensive_operations()
        results['concurrency'] = self.profile_concurrent_operations()
        results['scalability'] = self.profile_scalability_limits()
        results['delta_encoding'] = self.profile_delta_encoding_performance()
        results['storage'] = self.profile_storage_performance()
        
        # Generate comprehensive report
        self._generate_comprehensive_report(results)
        
        return results
    
    def _save_profile_report(self, test_name: str, profile_output: str, test_results: Dict[str, Any]):
        """Save profile report to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save profile output
        profile_file = self.output_dir / f"{test_name}_profile_{timestamp}.txt"
        with open(profile_file, 'w') as f:
            f.write(f"Profile Report: {test_name}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.write(profile_output)
        
        # Save test results
        results_file = self.output_dir / f"{test_name}_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"Saved profile report: {profile_file}")
        print(f"Saved test results: {results_file}")
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]):
        """Generate comprehensive performance report."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"comprehensive_performance_report_{timestamp}.json"
        
        # Analyze results for key insights
        insights = self._analyze_results(results)
        
        comprehensive_report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': insights,
            'detailed_results': results,
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        print(f"\nComprehensive report saved: {report_file}")
        
        # Print key insights
        self._print_insights(insights)
        
        return comprehensive_report
    
    def _analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results to extract key insights."""
        insights = {
            'critical_issues': [],
            'performance_bottlenecks': [],
            'memory_concerns': [],
            'scalability_limits': [],
            'recommendations': []
        }
        
        # Analyze memory results
        if 'memory' in results:
            memory_results = results['memory']
            for size_key, size_data in memory_results.items():
                if isinstance(size_data, dict) and 'memory_efficiency' in size_data:
                    if size_data['memory_efficiency'] > 10:
                        insights['memory_concerns'].append(
                            f"High memory overhead for {size_key}: {size_data['memory_efficiency']:.1f}x"
                        )
        
        # Analyze concurrency results
        if 'concurrency' in results:
            concurrency_results = results['concurrency']
            for thread_key, thread_data in concurrency_results.items():
                if isinstance(thread_data, dict) and 'success_rate' in thread_data:
                    if thread_data['success_rate'] < 0.8:
                        insights['critical_issues'].append(
                            f"Low success rate for {thread_key}: {thread_data['success_rate']:.1%}"
                        )
        
        # Analyze scalability results
        if 'scalability' in results:
            scalability_results = results['scalability']
            
            # Check clustering scalability
            if 'clustering' in scalability_results:
                clustering_data = scalability_results['clustering']
                throughput_degradation = self._check_throughput_degradation(clustering_data, 'tensors_per_second_dedup')
                if throughput_degradation:
                    insights['scalability_limits'].append(
                        f"Clustering throughput degradation: {throughput_degradation:.1%}"
                    )
            
            # Check deduplication scalability
            if 'deduplication' in scalability_results:
                dedup_data = scalability_results['deduplication']
                throughput_degradation = self._check_throughput_degradation(dedup_data, 'weights_per_second_dedup')
                if throughput_degradation:
                    insights['scalability_limits'].append(
                        f"Deduplication throughput degradation: {throughput_degradation:.1%}"
                    )
        
        # Generate recommendations
        if insights['memory_concerns']:
            insights['recommendations'].append("Consider implementing memory pooling or lazy loading for large tensors")
        
        if insights['critical_issues']:
            insights['recommendations'].append("Investigate and fix concurrency issues before production deployment")
        
        if insights['scalability_limits']:
            insights['recommendations'].append("Implement performance optimizations for large-scale operations")
        
        return insights
    
    def _check_throughput_degradation(self, results: Dict[str, Any], throughput_key: str) -> float:
        """Check for throughput degradation across different scales."""
        throughputs = []
        scales = []
        
        for key, value in results.items():
            if isinstance(value, dict) and throughput_key in value:
                # Extract scale from key (e.g., "tensors_1000" -> 1000)
                try:
                    scale = int(key.split('_')[-1])
                    throughput = value[throughput_key]
                    if throughput > 0:
                        throughputs.append(throughput)
                        scales.append(scale)
                except (ValueError, IndexError):
                    continue
        
        if len(throughputs) >= 2:
            # Calculate degradation: (max_throughput - min_throughput) / max_throughput
            max_throughput = max(throughputs)
            min_throughput = min(throughputs)
            degradation = (max_throughput - min_throughput) / max_throughput
            return degradation if degradation > 0.2 else 0  # Only report if > 20% degradation
        
        return 0
    
    def _print_insights(self, insights: Dict[str, Any]):
        """Print key insights to console."""
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS INSIGHTS")
        print("="*60)
        
        if insights['critical_issues']:
            print("\n🚨 CRITICAL ISSUES:")
            for issue in insights['critical_issues']:
                print(f"  - {issue}")
        
        if insights['performance_bottlenecks']:
            print("\n⚠️ PERFORMANCE BOTTLENECKS:")
            for bottleneck in insights['performance_bottlenecks']:
                print(f"  - {bottleneck}")
        
        if insights['memory_concerns']:
            print("\n💾 MEMORY CONCERNS:")
            for concern in insights['memory_concerns']:
                print(f"  - {concern}")
        
        if insights['scalability_limits']:
            print("\n📈 SCALABILITY LIMITS:")
            for limit in insights['scalability_limits']:
                print(f"  - {limit}")
        
        if insights['recommendations']:
            print("\n💡 RECOMMENDATIONS:")
            for rec in insights['recommendations']:
                print(f"  - {rec}")
        
        if not any([insights['critical_issues'], insights['performance_bottlenecks'], 
                   insights['memory_concerns'], insights['scalability_limits']]):
            print("\n✅ No significant performance issues detected!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Coral Performance Profiler")
    parser.add_argument(
        '--test', 
        choices=['all', 'memory', 'concurrency', 'scalability', 'delta', 'storage'],
        default='all',
        help='Type of performance test to run'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("performance_reports"),
        help='Output directory for reports'
    )
    
    args = parser.parse_args()
    
    profiler = AdvancedProfiler(output_dir=args.output_dir)
    
    print(f"Coral Performance Profiler")
    print(f"Output directory: {args.output_dir}")
    print(f"Test type: {args.test}")
    print("="*60)
    
    try:
        if args.test == 'all':
            results = profiler.run_comprehensive_analysis()
        elif args.test == 'memory':
            results = profiler.profile_memory_intensive_operations()
        elif args.test == 'concurrency':
            results = profiler.profile_concurrent_operations()
        elif args.test == 'scalability':
            results = profiler.profile_scalability_limits()
        elif args.test == 'delta':
            results = profiler.profile_delta_encoding_performance()
        elif args.test == 'storage':
            results = profiler.profile_storage_performance()
        
        print(f"\nProfiling complete! Results saved to {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\nProfiling interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nProfiling failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()