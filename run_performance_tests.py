#!/usr/bin/env python3
"""
Performance Test Runner for Coral ML

Runs a curated set of performance tests to demonstrate the system's capabilities
and identify any performance issues or bottlenecks.
"""

import argparse
import json
import time
from pathlib import Path

from performance_demo import QuickPerformanceDemo


def run_basic_performance_test():
    """Run basic performance demonstration."""
    print("="*60)
    print("CORAL ML PERFORMANCE TEST RUNNER")
    print("="*60)
    print("Running basic performance tests...")
    print()
    
    demo = QuickPerformanceDemo()
    results = demo.run_demo()
    
    return results


def save_results(results, output_file="performance_test_results.json"):
    """Save results to JSON file."""
    output_path = Path(output_file)
    
    # Add timestamp and system info
    test_report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_type': 'basic_performance_demo',
        'results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(test_report, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    return output_path


def analyze_results(results):
    """Analyze results and provide recommendations."""
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    issues = []
    recommendations = []
    
    # Analyze memory usage
    if 'memory_usage' in results:
        memory_tests = results['memory_usage']['tests']
        high_efficiency = [t for t in memory_tests if t['memory_efficiency_ratio'] > 50]
        if high_efficiency:
            issues.append(f"High memory overhead detected for small tensors")
            recommendations.append("Consider optimizing metadata storage for small tensors")
    
    # Analyze concurrent operations
    if 'concurrent_operations' in results:
        concurrent_tests = results['concurrent_operations']['tests']
        failed_tests = [t for t in concurrent_tests if t['success_rate'] < 1.0]
        if failed_tests:
            issues.append(f"Concurrency issues detected at {len(failed_tests)} thread levels")
            recommendations.append("CRITICAL: Fix thread safety issues before production")
    
    # Analyze deduplication performance
    if 'deduplication_performance' in results:
        dedup_tests = results['deduplication_performance']['tests']
        slow_dedup = [t for t in dedup_tests if t['throughput_tensors_per_sec'] < 1000]
        if slow_dedup:
            issues.append("Slow deduplication performance detected")
            recommendations.append("Investigate deduplication algorithm performance")
        else:
            print("✅ Excellent deduplication performance (>1000 tensors/sec)")
    
    # Print analysis
    if issues:
        print("\n⚠️  ISSUES DETECTED:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    
    if recommendations:
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    if not issues:
        print("\n✅ No significant performance issues detected!")
    
    # Performance summary
    print(f"\n📊 PERFORMANCE SUMMARY:")
    if 'deduplication_performance' in results:
        best_dedup = max(results['deduplication_performance']['tests'], 
                        key=lambda x: x['deduplication_ratio'])
        print(f"  Best deduplication: {best_dedup['deduplication_ratio']:.1f}x compression")
    
    if 'repository_scaling' in results:
        latest_scaling = results['repository_scaling']['tests'][-1]
        print(f"  Storage efficiency: {latest_scaling['storage_per_weight_kb']:.1f}KB per weight")
    
    if 'concurrent_operations' in results:
        single_thread = next((t for t in results['concurrent_operations']['tests'] 
                             if t['num_threads'] == 1), None)
        if single_thread:
            print(f"  Single-thread performance: {single_thread['throughput_ops_per_sec']:.1f} ops/sec")
    
    return issues, recommendations


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Coral ML Performance Test Runner")
    parser.add_argument(
        '--output', '-o',
        default='performance_test_results.json',
        help='Output file for test results'
    )
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze existing results file'
    )
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Load and analyze existing results
        try:
            with open(args.output, 'r') as f:
                data = json.load(f)
                results = data.get('results', {})
        except FileNotFoundError:
            print(f"Results file not found: {args.output}")
            return 1
    else:
        # Run performance tests
        try:
            results = run_basic_performance_test()
            save_results(results, args.output)
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            return 1
        except Exception as e:
            print(f"\nTest failed with error: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Analyze results
    issues, recommendations = analyze_results(results)
    
    # Return exit code based on issues
    if any("CRITICAL" in rec for rec in recommendations):
        print("\n🚨 CRITICAL ISSUES DETECTED - DO NOT DEPLOY TO PRODUCTION")
        return 2
    elif issues:
        print("\n⚠️  PERFORMANCE ISSUES DETECTED - REVIEW RECOMMENDED")
        return 1
    else:
        print("\n✅ ALL PERFORMANCE TESTS PASSED")
        return 0


if __name__ == "__main__":
    exit(main())