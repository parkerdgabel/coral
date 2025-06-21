#!/usr/bin/env python3
"""
Example script showing how to run Coral Data Integrity QA tests.

This script demonstrates the proper way to execute the comprehensive
data integrity test suite and interpret the results.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_quick_integrity_check():
    """Run a quick data integrity check."""
    print("=" * 60)
    print("CORAL DATA INTEGRITY - QUICK CHECK")
    print("=" * 60)
    
    try:
        # Import the main test runner
        from run_data_integrity_qa import DataIntegrityQARunner
        
        # Create test runner
        runner = DataIntegrityQARunner("qa_reports_quick")
        
        # Run just the comprehensive tests (fastest)
        print("Running comprehensive data integrity tests...")
        comprehensive_results = runner.run_comprehensive_tests()
        
        if comprehensive_results["success"]:
            success_rate = comprehensive_results["success_rate"]
            accuracy_rate = comprehensive_results["perfect_accuracy_rate"]
            corruption_rate = comprehensive_results["corruption_detection_rate"]
            
            print(f"\nRESULTS:")
            print(f"  Test Success Rate: {success_rate:.1%}")
            print(f"  Perfect Accuracy Rate: {accuracy_rate:.1%}")
            print(f"  Corruption Detection Rate: {corruption_rate:.1%}")
            
            if success_rate >= 0.95 and accuracy_rate >= 0.95 and corruption_rate >= 0.90:
                print("\n✓ DATA INTEGRITY VERIFIED")
                print("  Coral is ready for use with data integrity guarantees")
                return True
            else:
                print("\n⚠ DATA INTEGRITY ISSUES DETECTED")
                print("  Review detailed test results before production use")
                return False
        else:
            print(f"\n✗ TESTS FAILED: {comprehensive_results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n✗ TEST EXECUTION FAILED: {e}")
        return False


def run_full_integrity_suite():
    """Run the full data integrity test suite."""
    print("=" * 60)
    print("CORAL DATA INTEGRITY - FULL SUITE")
    print("=" * 60)
    
    try:
        # Import the main test runner
        from run_data_integrity_qa import DataIntegrityQARunner
        
        # Create test runner
        runner = DataIntegrityQARunner("qa_reports_full")
        
        # Run all tests
        print("Running full data integrity test suite...")
        print("This may take several minutes...\n")
        
        results = runner.run_all_tests()
        
        # Print summary
        runner.print_summary()
        
        summary = results["summary"]
        return summary["data_integrity_guaranteed"]
        
    except Exception as e:
        print(f"\n✗ TEST EXECUTION FAILED: {e}")
        return False


def run_specific_test_suite(suite_name: str):
    """Run a specific test suite."""
    print(f"=" * 60)
    print(f"CORAL DATA INTEGRITY - {suite_name.upper()} TESTS")
    print(f"=" * 60)
    
    try:
        # Import the main test runner
        from run_data_integrity_qa import DataIntegrityQARunner
        
        # Create test runner
        runner = DataIntegrityQARunner(f"qa_reports_{suite_name}")
        
        # Run specific suite
        if suite_name == "comprehensive":
            results = runner.run_comprehensive_tests()
        elif suite_name == "safetensors":
            results = runner.run_safetensors_tests()
        elif suite_name == "backup_recovery":
            results = runner.run_backup_recovery_tests()
        else:
            print(f"Unknown test suite: {suite_name}")
            return False
        
        if results["success"]:
            success_rate = results.get("success_rate", 0)
            total_tests = results.get("total_tests", 0)
            passed_tests = results.get("passed_tests", results.get("successful_tests", 0))
            
            print(f"\nRESULTS:")
            print(f"  Tests Passed: {passed_tests}/{total_tests}")
            print(f"  Success Rate: {success_rate:.1%}")
            
            if success_rate >= 0.95:
                print(f"\n✓ {suite_name.upper()} TESTS PASSED")
                return True
            else:
                print(f"\n⚠ {suite_name.upper()} TESTS HAVE ISSUES")
                return False
        else:
            print(f"\n✗ {suite_name.upper()} TESTS FAILED: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n✗ TEST EXECUTION FAILED: {e}")
        return False


def main():
    """Main entry point with command-line options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Coral Data Integrity QA Tests")
    parser.add_argument("--mode", choices=["quick", "full", "suite"], default="quick",
                       help="Test mode (default: quick)")
    parser.add_argument("--suite", choices=["comprehensive", "safetensors", "backup_recovery"],
                       help="Specific test suite to run (only with --mode suite)")
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        success = run_quick_integrity_check()
    elif args.mode == "full":
        success = run_full_integrity_suite()
    elif args.mode == "suite":
        if not args.suite:
            print("Error: --suite is required with --mode suite")
            return 1
        success = run_specific_test_suite(args.suite)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())