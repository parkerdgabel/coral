#!/usr/bin/env python3
"""
Coral ML Automated Benchmark Runner Demo

This example demonstrates how to use the automated benchmark runner with
comprehensive configuration support, scheduling, and CI/CD integration.

Usage:
    uv run python examples/automated_benchmark_runner_demo.py
"""

import json
import tempfile
from pathlib import Path

# Import the benchmark configuration and runner
from benchmarks.config import BenchmarkConfiguration, BenchmarkProfile, ResourceConstraints
from benchmarks.runner import AutomatedBenchmarkRunner

# Import available benchmark suites
from benchmarks.storage_benchmarks import StorageBenchmarkSuite
from benchmarks.performance_benchmarks import PerformanceBenchmarkSuite


def demo_basic_automated_run():
    """Demonstrate basic automated benchmark run."""
    print("🔄 Demo: Basic Automated Benchmark Run")
    print("=" * 50)
    
    # Create configuration with automation features
    config = BenchmarkConfiguration(
        name="demo_automated_benchmarks",
        profile=BenchmarkProfile.QUICK,  # Use quick profile for demo
        parallel_execution=True,
        max_workers=2,
        measurement_runs=2,
        warmup_runs=1,
        verbose=True,
        
        # Enable automation features
        automation_enabled=True,
        ci_mode=True,
        save_intermediate=True,
        report_formats=["json", "html"],
        
        # Resource constraints
        resource_constraints=ResourceConstraints(
            max_memory_gb=4.0,
            max_cpu_percent=80.0,
            max_execution_time=300.0
        ),
        
        # Output configuration
        output_dir=Path("demo_benchmark_results")
    )
    
    # Create automated benchmark runner
    runner = AutomatedBenchmarkRunner(config)
    
    # Available benchmark suites for demo
    suite_classes = [StorageBenchmarkSuite, PerformanceBenchmarkSuite]
    
    print(f"Running benchmarks with profile: {config.profile.value}")
    print(f"Output directory: {config.output_dir}")
    print(f"Parallel execution: {config.parallel_execution}")
    print(f"Max workers: {config.max_workers}")
    
    try:
        # Run comprehensive benchmarks
        results = runner.run_comprehensive_benchmarks(
            suite_classes=suite_classes,
            suite_filter={"StorageBenchmarkSuite"},  # Only run storage suite for demo
            benchmark_filter=None  # Run all benchmarks in selected suites
        )
        
        # Print summary
        execution_summary = results.get("execution_summary", {})
        print(f"\n📊 Results Summary:")
        print(f"  Total benchmarks: {execution_summary.get('total_benchmarks', 0)}")
        print(f"  Successful: {execution_summary.get('successful', 0)}")
        print(f"  Failed: {execution_summary.get('failed', 0)}")
        print(f"  Success rate: {execution_summary.get('success_rate', 0):.1%}")
        print(f"  Total duration: {execution_summary.get('total_duration', 0):.2f}s")
        
        # Show resource usage
        resource_usage = results.get("resource_usage", {})
        if resource_usage:
            memory = resource_usage.get("memory_usage", {})
            cpu = resource_usage.get("cpu_usage", {})
            print(f"\n💾 Resource Usage:")
            print(f"  Peak memory: {memory.get('peak_gb', 0):.2f} GB")
            print(f"  Peak CPU: {cpu.get('peak_percent', 0):.1f}%")
        
        return results
        
    except Exception as e:
        print(f"❌ Benchmark run failed: {e}")
        return None


def demo_ci_cd_configurations():
    """Demonstrate CI/CD configuration generation."""
    print("\n🔧 Demo: CI/CD Configuration Generation")
    print("=" * 50)
    
    config = BenchmarkConfiguration(
        profile=BenchmarkProfile.CI_CD,
        ci_mode=True,
        fail_on_regression=True,
        regression_threshold=0.1
    )
    
    runner = AutomatedBenchmarkRunner(config)
    
    # Generate CI/CD configurations
    configurations = runner.generate_ci_cd_configurations()
    
    print("Generated CI/CD configurations:")
    
    # Show GitHub Actions configuration
    github_config = configurations.get("github_actions", {})
    print(f"\n📁 GitHub Actions workflow:")
    print(f"  Name: {github_config.get('name', 'N/A')}")
    print(f"  Triggers: {list(github_config.get('on', {}).keys())}")
    
    # Show cron configurations
    print(f"\n⏰ Cron configurations:")
    print(f"  Daily: {configurations.get('cron_daily', 'N/A')}")
    print(f"  Weekly: {configurations.get('cron_weekly', 'N/A')}")
    
    # Save configurations to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save GitHub Actions workflow
        github_workflow_dir = temp_path / ".github" / "workflows"
        github_workflow_dir.mkdir(parents=True, exist_ok=True)
        
        github_workflow_file = github_workflow_dir / "benchmarks.yml"
        with open(github_workflow_file, 'w') as f:
            import yaml
            yaml.dump(github_config, f, default_flow_style=False)
        
        print(f"\n💾 Saved GitHub Actions workflow to: {github_workflow_file}")
        
        # Show file contents (first few lines)
        with open(github_workflow_file, 'r') as f:
            lines = f.readlines()[:10]
            print("Preview:")
            for line in lines:
                print(f"    {line.rstrip()}")
    
    return configurations


def demo_performance_trends():
    """Demonstrate performance trends analysis."""
    print("\n📈 Demo: Performance Trends Analysis")
    print("=" * 50)
    
    config = BenchmarkConfiguration(
        output_dir=Path("demo_benchmark_results")
    )
    
    runner = AutomatedBenchmarkRunner(config)
    
    # Try to analyze performance trends
    try:
        trends = runner.get_performance_trends(days=7)  # Last 7 days
        
        analysis_days = trends.get("analysis_period_days", 0)
        trend_data = trends.get("trends", {})
        
        print(f"Analysis period: {analysis_days} days")
        
        if trend_data:
            print("\nTrend analysis:")
            for metric_name, trend_info in trend_data.items():
                direction = trend_info.get("trend_direction", "unknown")
                data_points = trend_info.get("data_points", 0)
                current_value = trend_info.get("current_value", 0)
                
                direction_emoji = {
                    "improving": "📈",
                    "degrading": "📉",
                    "stable": "➡️"
                }.get(direction, "❓")
                
                print(f"  {direction_emoji} {metric_name}:")
                print(f"    Current: {current_value:.4f}")
                print(f"    Trend: {direction}")
                print(f"    Data points: {data_points}")
        else:
            print("No trend data available (need historical benchmarks)")
    
    except Exception as e:
        print(f"❌ Trends analysis failed: {e}")
        print("This is normal if no historical data exists yet")


def demo_configuration_management():
    """Demonstrate configuration management features."""
    print("\n⚙️ Demo: Configuration Management")
    print("=" * 50)
    
    # Create a comprehensive configuration
    config = BenchmarkConfiguration(
        name="comprehensive_demo_config",
        profile=BenchmarkProfile.COMPREHENSIVE,
        
        # Execution settings
        parallel_execution=True,
        max_workers=4,
        timeout=600.0,
        measurement_runs=10,
        warmup_runs=3,
        
        # Automation settings
        automation_enabled=True,
        continuous_benchmarking=True,
        continuous_interval_hours=12,
        max_continuous_iterations=5,
        
        # CI/CD settings
        ci_mode=False,
        fail_on_regression=True,
        regression_threshold=0.05,
        
        # Monitoring settings
        monitoring_enabled=True,
        alert_on_regression=True,
        alert_threshold=0.10,
        
        # Resource constraints
        resource_constraints=ResourceConstraints(
            max_memory_gb=8.0,
            max_cpu_percent=90.0,
            max_execution_time=1800.0,
            thread_limit=8,
            process_limit=4
        ),
        
        # Output settings
        output_dir=Path("comprehensive_benchmark_results"),
        save_intermediate=True,
        generate_plots=True,
        report_formats=["json", "html", "csv"],
        
        # Metadata
        metadata={
            "project": "coral_ml_demo",
            "version": "1.0.0",
            "environment": "development",
            "description": "Comprehensive demo configuration",
            "tags": ["demo", "comprehensive", "automation"]
        }
    )
    
    print("Created comprehensive configuration:")
    print(f"  Name: {config.name}")
    print(f"  Profile: {config.profile.value}")
    print(f"  Automation enabled: {config.automation_enabled}")
    print(f"  Continuous benchmarking: {config.continuous_benchmarking}")
    print(f"  Monitoring enabled: {config.monitoring_enabled}")
    print(f"  Resource limits: {config.resource_constraints.max_memory_gb}GB memory, {config.resource_constraints.max_cpu_percent}% CPU")
    
    # Save configuration to file
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = Path(temp_dir) / "comprehensive_config.yaml"
        config.save_to_file(config_file)
        
        print(f"\n💾 Saved configuration to: {config_file}")
        
        # Load it back
        loaded_config = BenchmarkConfiguration.load_from_file(config_file)
        print(f"✅ Successfully loaded configuration: {loaded_config.name}")
        
        # Show file preview
        with open(config_file, 'r') as f:
            lines = f.readlines()[:15]
            print("\nConfiguration file preview:")
            for line in lines:
                print(f"    {line.rstrip()}")
    
    return config


def main():
    """Run all demos."""
    print("🚀 Coral ML Automated Benchmark Runner Demo")
    print("=" * 60)
    print("This demo showcases the automated benchmark runner with")
    print("comprehensive configuration support, CI/CD integration,")
    print("and performance monitoring capabilities.")
    print()
    
    try:
        # Demo 1: Basic automated run
        results = demo_basic_automated_run()
        
        # Demo 2: CI/CD configurations
        configurations = demo_ci_cd_configurations()
        
        # Demo 3: Performance trends
        demo_performance_trends()
        
        # Demo 4: Configuration management
        config = demo_configuration_management()
        
        print("\n✅ All demos completed successfully!")
        print("\nNext steps:")
        print("1. Run 'uv run python -m benchmarks.cli run --help' for CLI options")
        print("2. Generate your own configuration with 'uv run python -m benchmarks.cli config generate'")
        print("3. Set up CI/CD with 'uv run python -m benchmarks.cli cicd generate'")
        print("4. Try continuous benchmarking with 'uv run python -m benchmarks.cli continuous'")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()