#!/usr/bin/env python3
"""
Example script demonstrating the Coral ML Automated Benchmark Runner.

This script shows how to:
1. Create custom benchmark configurations
2. Run benchmarks programmatically
3. Analyze and export results
4. Integrate with different profiles and hardware configurations
"""

import sys
import tempfile
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coral.benchmarks import (
    AutomatedBenchmarkRunner,
    BenchmarkConfiguration,
    BenchmarkProfile,
    HardwareType,
    ResourceConstraints,
    ConfigurationManager,
)

# Note: These imports would work when the benchmark suites are properly implemented
# from coral.benchmarks.storage_benchmarks import StorageBenchmarkSuite
# from coral.benchmarks.performance_benchmarks import PerformanceBenchmarkSuite


def example_basic_usage():
    """Basic usage example with default configuration."""
    print("🚀 Basic Benchmark Example")
    print("=" * 50)
    
    # Create default configuration
    config = BenchmarkConfiguration(
        name="basic_example",
        profile=BenchmarkProfile.QUICK,  # Fast for demonstration
        verbose=True,
    )
    
    print(f"Configuration: {config.name}")
    print(f"Profile: {config.profile.value}")
    print(f"Output directory: {config.output_dir}")
    print(f"Parallel execution: {config.parallel_execution}")
    print(f"Max workers: {config.max_workers}")
    
    # Create runner
    runner = AutomatedBenchmarkRunner(config)
    
    # Note: Commented out because benchmark suites need to be properly implemented
    # # Run benchmarks
    # results = runner.run_comprehensive_benchmarks([
    #     StorageBenchmarkSuite,
    #     PerformanceBenchmarkSuite,
    # ])
    # 
    # # Print summary
    # execution_summary = results.get("execution_summary", {})
    # print(f"\n📊 Results Summary:")
    # print(f"Total benchmarks: {execution_summary.get('total_benchmarks', 0)}")
    # print(f"Successful: {execution_summary.get('successful', 0)}")
    # print(f"Failed: {execution_summary.get('failed', 0)}")
    # print(f"Success rate: {execution_summary.get('success_rate', 0):.1%}")
    
    print("✓ Basic example configuration created successfully")


def example_comprehensive_configuration():
    """Example with comprehensive configuration options."""
    print("\n🔧 Comprehensive Configuration Example")
    print("=" * 50)
    
    # Create resource constraints
    constraints = ResourceConstraints(
        max_memory_gb=8.0,
        max_cpu_percent=80.0,
        max_execution_time=600.0,  # 10 minutes
        thread_limit=8,
        process_limit=4,
    )
    
    # Create comprehensive configuration
    config = BenchmarkConfiguration(
        name="comprehensive_benchmarks",
        profile=BenchmarkProfile.COMPREHENSIVE,
        
        # Execution settings
        parallel_execution=True,
        max_workers=4,
        timeout=300.0,
        measurement_runs=10,
        warmup_runs=3,
        
        # Hardware optimization
        hardware_type=HardwareType.CPU,
        auto_detect_hardware=True,
        optimize_for_hardware=True,
        
        # Resource management
        resource_constraints=constraints,
        
        # Reporting
        save_intermediate=True,
        generate_plots=True,
        report_formats=["json", "html", "csv"],
        
        # Debug settings
        verbose=True,
        debug=False,
        
        # Metadata
        metadata={
            "project": "coral_ml_demo",
            "version": "1.0.0",
            "environment": "development",
            "description": "Comprehensive benchmark demonstration",
            "tags": ["demo", "comprehensive"],
        }
    )
    
    print(f"Configuration name: {config.name}")
    print(f"Profile: {config.profile.value}")
    print(f"Hardware type: {config.hardware_type.value}")
    print(f"Resource constraints: Memory={constraints.max_memory_gb}GB, CPU={constraints.max_cpu_percent}%")
    print(f"Measurement runs: {config.measurement_runs}")
    print(f"Report formats: {config.report_formats}")
    
    print("✓ Comprehensive configuration created successfully")


def example_ci_cd_configuration():
    """Example configuration optimized for CI/CD pipelines."""
    print("\n🔄 CI/CD Configuration Example")
    print("=" * 50)
    
    config = BenchmarkConfiguration(
        name="ci_cd_benchmarks",
        profile=BenchmarkProfile.CI_CD,
        
        # CI/CD optimizations
        ci_mode=True,
        fail_on_regression=True,
        regression_threshold=0.1,  # 10% threshold
        
        # Fast execution
        measurement_runs=3,
        warmup_runs=1,
        timeout=120.0,
        
        # Minimal output
        verbose=False,
        save_intermediate=False,
        generate_plots=False,
        report_formats=["json"],  # Machine-readable only
        
        metadata={
            "ci_system": "github_actions",
            "build_id": "example_build_123",
        }
    )
    
    print(f"CI mode: {config.ci_mode}")
    print(f"Fail on regression: {config.fail_on_regression}")
    print(f"Regression threshold: {config.regression_threshold:.1%}")
    print(f"Measurement runs: {config.measurement_runs}")
    print(f"Report formats: {config.report_formats}")
    
    print("✓ CI/CD configuration created successfully")


def example_configuration_management():
    """Example of configuration file management."""
    print("\n📁 Configuration Management Example")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir)
        
        # Create configuration manager
        manager = ConfigurationManager()
        
        # Generate different profile configurations
        profiles = [BenchmarkProfile.QUICK, BenchmarkProfile.STANDARD, BenchmarkProfile.COMPREHENSIVE]
        
        for profile in profiles:
            config_file = config_dir / f"config_{profile.value}.yaml"
            manager.generate_default_config(config_file, profile)
            print(f"Generated: {config_file.name}")
            
            # Load and validate the configuration
            loaded_config = BenchmarkConfiguration.load_from_file(config_file)
            print(f"  - Profile: {loaded_config.profile.value}")
            print(f"  - Measurement runs: {loaded_config.measurement_runs}")
            print(f"  - Timeout: {loaded_config.timeout}s")
        
        print(f"\n✓ Generated {len(profiles)} configuration files")


def example_hardware_detection():
    """Example of hardware detection and optimization."""
    print("\n💻 Hardware Detection Example")
    print("=" * 50)
    
    # Create configuration with auto-detection
    config = BenchmarkConfiguration(
        name="hardware_optimized",
        auto_detect_hardware=True,
        optimize_for_hardware=True,
    )
    
    print(f"Auto-detected hardware type: {config.hardware_type.value}")
    print(f"Optimal workers: {config.max_workers}")
    print(f"Hardware optimization: {config.optimize_for_hardware}")
    
    # Show different hardware optimizations
    hardware_types = [
        HardwareType.CPU,
        HardwareType.GPU,
        HardwareType.HIGH_MEMORY,
        HardwareType.LOW_MEMORY,
        HardwareType.MULTI_CORE,
        HardwareType.SINGLE_CORE,
    ]
    
    print(f"\nHardware optimization examples:")
    for hw_type in hardware_types:
        test_config = BenchmarkConfiguration(
            hardware_type=hw_type,
            auto_detect_hardware=False,
        )
        print(f"  {hw_type.value}: {test_config.max_workers} workers")
    
    print("✓ Hardware detection example completed")


def example_suite_specific_configuration():
    """Example of suite-specific configuration."""
    print("\n⚙️ Suite-Specific Configuration Example")
    print("=" * 50)
    
    from coral.benchmarks.config import SuiteConfig
    
    # Create base configuration
    config = BenchmarkConfiguration(name="suite_specific_example")
    
    # Configure storage benchmarks
    storage_config = SuiteConfig(
        enabled=True,
        iterations=5,
        warmup_iterations=2,
        timeout=180.0,
        parallel=True,
        parameters={
            "compression_types": ["gzip", "lzf", "szip"],
            "test_sizes": [1000, 10000, 100000],
            "chunk_sizes": [1024, 4096, 16384],
        }
    )
    config.set_suite_config("storage", storage_config)
    
    # Configure performance benchmarks
    performance_config = SuiteConfig(
        enabled=True,
        iterations=10,
        warmup_iterations=3,
        timeout=300.0,
        parallel=False,  # Better for timing accuracy
        parameters={
            "model_sizes": ["small", "medium", "large"],
            "batch_sizes": [1, 16, 64],
        }
    )
    config.set_suite_config("performance", performance_config)
    
    # Configure clustering benchmarks
    clustering_config = SuiteConfig(
        enabled=True,
        iterations=3,
        parameters={
            "strategies": ["kmeans", "hierarchical", "adaptive"],
            "similarity_thresholds": [0.95, 0.98, 0.99],
            "cluster_sizes": [10, 50, 100],
        }
    )
    config.set_suite_config("clustering", clustering_config)
    
    # Show configurations
    for suite_name in ["storage", "performance", "clustering"]:
        suite_config = config.get_suite_config(suite_name)
        print(f"{suite_name}:")
        print(f"  Enabled: {suite_config.enabled}")
        print(f"  Iterations: {suite_config.iterations}")
        print(f"  Parameters: {len(suite_config.parameters)} configured")
    
    print("✓ Suite-specific configurations created successfully")


def main():
    """Run all configuration examples."""
    print("🔬 Coral ML Automated Benchmark Runner Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_usage()
        example_comprehensive_configuration()
        example_ci_cd_configuration()
        example_configuration_management()
        example_hardware_detection()
        example_suite_specific_configuration()
        
        print(f"\n🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("1. Try running: coral-bench run --profile quick")
        print("2. Generate a config: coral-bench config generate --profile standard --output my_config.yaml")
        print("3. Run with config: coral-bench run --config my_config.yaml")
        print("4. See help: coral-bench --help")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())