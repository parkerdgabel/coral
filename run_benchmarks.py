#!/usr/bin/env python3
"""
Main benchmark runner for Coral ML.

This script provides a simple interface to run comprehensive benchmarks
of the Coral ML weight versioning system. It supports different benchmark
profiles and generates detailed reports.

Usage:
    uv run run_benchmarks.py                    # Run standard benchmarks
    uv run run_benchmarks.py --profile quick    # Run quick benchmarks
    uv run run_benchmarks.py --profile full     # Run comprehensive benchmarks
    uv run run_benchmarks.py --output report    # Generate detailed reports
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from benchmarks.main import BenchmarkOrchestrator, BenchmarkProfile
from benchmarks.visualization import BenchmarkVisualizer
from benchmarks.analysis import BenchmarkAnalyzer


class BenchmarkRunner:
    """Main benchmark runner with profile support and reporting."""
    
    def __init__(self):
        """Initialize benchmark runner."""
        self.orchestrator = BenchmarkOrchestrator()
        self.visualizer = BenchmarkVisualizer()
        self.analyzer = BenchmarkAnalyzer()
        self.start_time: Optional[float] = None
        
    def run(
        self,
        profile: BenchmarkProfile = BenchmarkProfile.STANDARD,
        output_dir: Optional[Path] = None,
        generate_report: bool = True,
        visualize: bool = True,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run benchmarks with specified profile.
        
        Args:
            profile: Benchmark profile to run
            output_dir: Directory for output files
            generate_report: Whether to generate detailed reports
            visualize: Whether to create visualizations
            verbose: Enable verbose output
            
        Returns:
            Dictionary with benchmark results
        """
        self.start_time = time.time()
        
        # Set up output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"benchmark_results_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Coral ML Benchmark Suite")
        print(f"{'='*60}")
        print(f"Profile: {profile.value}")
        print(f"Output: {output_dir}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # Run benchmarks
        results = self.orchestrator.run_all(profile, verbose=verbose)
        
        # Generate analysis
        if generate_report:
            print("\nGenerating analysis...")
            analysis = self.analyzer.analyze_results(results)
            self._save_analysis(analysis, output_dir)
            
        # Create visualizations
        if visualize:
            print("\nCreating visualizations...")
            self._create_visualizations(results, output_dir)
            
        # Generate final report
        if generate_report:
            print("\nGenerating final report...")
            self._generate_report(results, analysis, output_dir)
            
        # Print summary
        self._print_summary(results, analysis)
        
        print(f"\nResults saved to: {output_dir}")
        
        return results
        
    def _save_analysis(self, analysis: Dict[str, Any], output_dir: Path) -> None:
        """Save analysis results to file."""
        import json
        
        analysis_file = output_dir / "analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
            
    def _create_visualizations(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Create and save visualizations."""
        # Compression ratios
        if 'compression' in results:
            fig = self.visualizer.plot_compression_ratios(results['compression'])
            fig.savefig(output_dir / "compression_ratios.png", dpi=150, bbox_inches='tight')
            
        # Performance metrics
        if 'performance' in results:
            fig = self.visualizer.plot_performance_metrics(results['performance'])
            fig.savefig(output_dir / "performance_metrics.png", dpi=150, bbox_inches='tight')
            
        # Clustering efficiency
        if 'clustering' in results:
            fig = self.visualizer.plot_clustering_efficiency(results['clustering'])
            fig.savefig(output_dir / "clustering_efficiency.png", dpi=150, bbox_inches='tight')
            
        # Scalability curves
        if 'scalability' in results:
            fig = self.visualizer.plot_scalability_curves(results['scalability'])
            fig.savefig(output_dir / "scalability_curves.png", dpi=150, bbox_inches='tight')
            
    def _generate_report(
        self,
        results: Dict[str, Any],
        analysis: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """Generate comprehensive markdown report."""
        report_file = output_dir / "benchmark_report.md"
        
        with open(report_file, 'w') as f:
            # Header
            f.write("# Coral ML Benchmark Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Runtime**: {self._format_duration(time.time() - self.start_time)}\n")
            f.write(f"- **Average Compression**: {analysis.get('avg_compression', 0):.2f}x\n")
            f.write(f"- **Best Compression**: {analysis.get('best_compression', 0):.2f}x\n")
            f.write(f"- **Performance Impact**: {analysis.get('performance_impact', 0):.1%}\n\n")
            
            # Detailed Results
            f.write("## Detailed Results\n\n")
            
            # Compression results
            if 'compression' in results:
                f.write("### Compression Performance\n\n")
                f.write("| Scenario | Compression Ratio | Space Saved |\n")
                f.write("|----------|------------------|-------------|\n")
                for scenario, data in results['compression'].items():
                    ratio = data.get('compression_ratio', 1.0)
                    saved = (1 - 1/ratio) * 100
                    f.write(f"| {scenario} | {ratio:.2f}x | {saved:.1f}% |\n")
                f.write("\n")
                
            # Performance results
            if 'performance' in results:
                f.write("### Operation Performance\n\n")
                f.write("| Operation | Time (ms) | Throughput |\n")
                f.write("|-----------|-----------|------------|\n")
                for op, data in results['performance'].items():
                    time_ms = data.get('time_ms', 0)
                    throughput = data.get('throughput', 'N/A')
                    f.write(f"| {op} | {time_ms:.2f} | {throughput} |\n")
                f.write("\n")
                
            # Clustering results
            if 'clustering' in results:
                f.write("### Clustering Efficiency\n\n")
                f.write("| Strategy | Clusters | Compression | Time (s) |\n")
                f.write("|----------|----------|-------------|----------|\n")
                for strategy, data in results['clustering'].items():
                    clusters = data.get('num_clusters', 0)
                    compression = data.get('compression_ratio', 1.0)
                    time_s = data.get('time_seconds', 0)
                    f.write(f"| {strategy} | {clusters} | {compression:.2f}x | {time_s:.2f} |\n")
                f.write("\n")
                
            # Recommendations
            f.write("## Recommendations\n\n")
            for rec in analysis.get('recommendations', []):
                f.write(f"- {rec}\n")
            f.write("\n")
            
            # System Information
            f.write("## System Information\n\n")
            import platform
            f.write(f"- **Platform**: {platform.platform()}\n")
            f.write(f"- **Python**: {platform.python_version()}\n")
            f.write(f"- **CPU**: {platform.processor()}\n")
            
    def _print_summary(self, results: Dict[str, Any], analysis: Dict[str, Any]) -> None:
        """Print summary to console."""
        elapsed = time.time() - self.start_time
        
        print(f"\n{'='*60}")
        print("Benchmark Summary")
        print(f"{'='*60}")
        print(f"Total runtime: {self._format_duration(elapsed)}")
        print(f"Average compression: {analysis.get('avg_compression', 0):.2f}x")
        print(f"Best compression: {analysis.get('best_compression', 0):.2f}x")
        print(f"Performance impact: {analysis.get('performance_impact', 0):.1%}")
        
        # Key insights
        if 'key_insights' in analysis:
            print("\nKey Insights:")
            for insight in analysis['key_insights']:
                print(f"  • {insight}")
                
        print(f"{'='*60}")
        
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run Coral ML benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run run_benchmarks.py                    # Run standard benchmarks
  uv run run_benchmarks.py --profile quick    # Quick benchmarks (~1 min)
  uv run run_benchmarks.py --profile full     # Full benchmarks (~10 min)
  uv run run_benchmarks.py --no-visualize     # Skip visualization
  uv run run_benchmarks.py --output results   # Custom output directory
        """
    )
    
    parser.add_argument(
        '--profile',
        type=str,
        choices=['quick', 'standard', 'full', 'comprehensive'],
        default='standard',
        help='Benchmark profile to run (default: standard)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip report generation'
    )
    
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Integration with existing benchmark.py
    parser.add_argument(
        '--legacy',
        action='store_true',
        help='Run legacy benchmark.py for comparison'
    )
    
    args = parser.parse_args()
    
    # Run legacy benchmark if requested
    if args.legacy:
        print("Running legacy benchmark for comparison...")
        import subprocess
        subprocess.run([sys.executable, "benchmark.py"], check=True)
        print("\n")
    
    # Map profile string to enum
    profile_map = {
        'quick': BenchmarkProfile.QUICK,
        'standard': BenchmarkProfile.STANDARD,
        'full': BenchmarkProfile.FULL,
        'comprehensive': BenchmarkProfile.COMPREHENSIVE
    }
    profile = profile_map[args.profile]
    
    # Run benchmarks
    runner = BenchmarkRunner()
    
    try:
        results = runner.run(
            profile=profile,
            output_dir=Path(args.output) if args.output else None,
            generate_report=not args.no_report,
            visualize=not args.no_visualize,
            verbose=args.verbose
        )
        
        # Success
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\nError running benchmarks: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()