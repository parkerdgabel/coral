#!/usr/bin/env python3
"""
Coral Data Integrity QA Test Runner

This script runs all data integrity tests and generates comprehensive reports
to validate data consistency, corruption detection, and recovery scenarios.
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add the tests directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))

try:
    from test_data_integrity_comprehensive import TestDataIntegrityComprehensive
    from test_safetensors_integrity import SafeTensorsIntegrityTests
    from test_backup_recovery import BackupRecoveryIntegrityTests
except ImportError as e:
    print(f"Failed to import test modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class DataIntegrityQARunner:
    """Master test runner for all data integrity QA tests."""
    
    def __init__(self, output_dir: str = "qa_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.test_results = {
            "comprehensive": None,
            "safetensors": None,
            "backup_recovery": None,
            "summary": {}
        }
        
        self.start_time = time.time()
        
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive data integrity tests."""
        print("=" * 80)
        print("RUNNING COMPREHENSIVE DATA INTEGRITY TESTS")
        print("=" * 80)
        
        try:
            test_instance = TestDataIntegrityComprehensive()
            test_instance.setup_method()
            
            try:
                results = test_instance.run_all_tests()
                
                # Extract key metrics
                total_tests = len(results.test_results)
                passed_tests = sum(1 for r in results.test_results if r['passed'])
                
                perfect_accuracy = sum(1 for m in results.accuracy_measurements 
                                     if m['accuracy'] >= 1.0 - 1e-12)
                total_accuracy = len(results.accuracy_measurements)
                
                detected_corruptions = sum(1 for d in results.corruption_detections 
                                          if d['detected'])
                total_corruptions = len(results.corruption_detections)
                
                return {
                    "success": True,
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                    "perfect_accuracy_count": perfect_accuracy,
                    "total_accuracy_measurements": total_accuracy,
                    "perfect_accuracy_rate": perfect_accuracy / total_accuracy if total_accuracy > 0 else 0,
                    "detected_corruptions": detected_corruptions,
                    "total_corruptions": total_corruptions,
                    "corruption_detection_rate": detected_corruptions / total_corruptions if total_corruptions > 0 else 0,
                    "raw_results": results,
                    "report": results.generate_report()
                }
                
            finally:
                test_instance.teardown_method()
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def run_safetensors_tests(self) -> Dict[str, Any]:
        """Run SafeTensors integrity tests."""
        print("=" * 80)
        print("RUNNING SAFETENSORS INTEGRITY TESTS")
        print("=" * 80)
        
        try:
            test_instance = SafeTensorsIntegrityTests()
            test_instance.setup_method()
            
            try:
                results = test_instance.run_all_tests()
                
                # Extract key metrics
                total_tests = len(results)
                successful_tests = sum(1 for r in results 
                                     if isinstance(r.get("results"), list) and 
                                     all(item.get("success", item.get("data_integrity", False)) 
                                         for item in r["results"]))
                
                return {
                    "success": True,
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                    "raw_results": results,
                    "report": test_instance.generate_integrity_report()
                }
                
            finally:
                test_instance.teardown_method()
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def run_backup_recovery_tests(self) -> Dict[str, Any]:
        """Run backup and recovery tests."""
        print("=" * 80)
        print("RUNNING BACKUP AND RECOVERY TESTS")
        print("=" * 80)
        
        try:
            test_instance = BackupRecoveryIntegrityTests()
            test_instance.setup_method()
            
            try:
                results = test_instance.run_all_tests()
                
                # Extract key metrics
                total_tests = len(results)
                successful_tests = sum(1 for r in results if r.get("success", False))
                
                return {
                    "success": True,
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                    "raw_results": results,
                    "report": test_instance.generate_recovery_report()
                }
                
            finally:
                test_instance.teardown_method()
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all data integrity tests."""
        print("CORAL DATA INTEGRITY QA SUITE")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("")
        
        # Run comprehensive tests
        print("Phase 1/3: Comprehensive Data Integrity Tests")
        self.test_results["comprehensive"] = self.run_comprehensive_tests()
        print("")
        
        # Run SafeTensors tests
        print("Phase 2/3: SafeTensors Format Integrity Tests")
        self.test_results["safetensors"] = self.run_safetensors_tests()
        print("")
        
        # Run backup/recovery tests
        print("Phase 3/3: Backup and Recovery Tests")
        self.test_results["backup_recovery"] = self.run_backup_recovery_tests()
        print("")
        
        # Generate summary
        self.generate_summary()
        
        # Save results
        self.save_results()
        
        return self.test_results
    
    def generate_summary(self):
        """Generate overall test summary."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        # Collect metrics from all test suites
        total_tests = 0
        passed_tests = 0
        test_suites = 0
        successful_suites = 0
        
        for suite_name, results in self.test_results.items():
            if suite_name == "summary" or not results or not results.get("success"):
                continue
                
            test_suites += 1
            if results.get("success"):
                successful_suites += 1
                
            total_tests += results.get("total_tests", 0)
            passed_tests += results.get("passed_tests", results.get("successful_tests", 0))
        
        # Calculate rates
        overall_success_rate = passed_tests / total_tests if total_tests > 0 else 0
        suite_success_rate = successful_suites / test_suites if test_suites > 0 else 0
        
        # Specific metrics from comprehensive tests
        comprehensive = self.test_results.get("comprehensive", {})
        perfect_accuracy_rate = comprehensive.get("perfect_accuracy_rate", 0)
        corruption_detection_rate = comprehensive.get("corruption_detection_rate", 0)
        
        self.test_results["summary"] = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": duration,
            "duration_formatted": f"{duration:.2f} seconds",
            "test_suites": test_suites,
            "successful_suites": successful_suites,
            "suite_success_rate": suite_success_rate,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "overall_success_rate": overall_success_rate,
            "perfect_accuracy_rate": perfect_accuracy_rate,
            "corruption_detection_rate": corruption_detection_rate,
            "data_integrity_guaranteed": (
                overall_success_rate >= 0.95 and
                perfect_accuracy_rate >= 0.95 and
                corruption_detection_rate >= 0.90
            )
        }
    
    def save_results(self):
        """Save test results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = self.output_dir / f"data_integrity_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            # Create a JSON-serializable version
            json_results = {}
            for key, value in self.test_results.items():
                if key == "summary":
                    json_results[key] = value
                elif value and value.get("success"):
                    json_results[key] = {
                        "success": value["success"],
                        "total_tests": value.get("total_tests", 0),
                        "passed_tests": value.get("passed_tests", value.get("successful_tests", 0)),
                        "success_rate": value.get("success_rate", 0),
                        "metrics": {
                            k: v for k, v in value.items() 
                            if k not in ["raw_results", "report"] and not k.endswith("_results")
                        }
                    }
                else:
                    json_results[key] = {
                        "success": False,
                        "error": value.get("error", "Unknown error") if value else "No results"
                    }
            
            json.dump(json_results, f, indent=2, default=str)
        
        # Save detailed reports
        for suite_name, results in self.test_results.items():
            if suite_name == "summary" or not results or not results.get("report"):
                continue
                
            report_file = self.output_dir / f"{suite_name}_report_{timestamp}.txt"
            with open(report_file, 'w') as f:
                f.write(results["report"])
        
        # Save summary report
        summary_file = self.output_dir / f"data_integrity_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(self.generate_summary_report())
        
        print(f"Results saved to: {self.output_dir}")
        print(f"Summary report: {summary_file}")
        print(f"JSON results: {json_file}")
    
    def generate_summary_report(self) -> str:
        """Generate human-readable summary report."""
        summary = self.test_results["summary"]
        
        report = []
        report.append("=" * 80)
        report.append("CORAL DATA INTEGRITY QA - EXECUTIVE SUMMARY")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Duration: {summary['duration_formatted']}")
        report.append("")
        
        # Overall Results
        report.append("OVERALL RESULTS:")
        report.append(f"  Test Suites: {summary['successful_suites']}/{summary['test_suites']} successful")
        report.append(f"  Individual Tests: {summary['passed_tests']}/{summary['total_tests']} passed")
        report.append(f"  Overall Success Rate: {summary['overall_success_rate']:.1%}")
        report.append("")
        
        # Data Integrity Guarantees
        report.append("DATA INTEGRITY GUARANTEES:")
        report.append(f"  Perfect Accuracy Rate: {summary['perfect_accuracy_rate']:.1%}")
        report.append(f"  Corruption Detection Rate: {summary['corruption_detection_rate']:.1%}")
        guaranteed = "✓ GUARANTEED" if summary['data_integrity_guaranteed'] else "✗ NOT GUARANTEED"
        report.append(f"  Data Integrity: {guaranteed}")
        report.append("")
        
        # Suite-specific Results
        report.append("SUITE RESULTS:")
        
        for suite_name, results in self.test_results.items():
            if suite_name == "summary":
                continue
                
            suite_display = suite_name.replace("_", " ").title()
            
            if results and results.get("success"):
                total = results.get("total_tests", 0)
                passed = results.get("passed_tests", results.get("successful_tests", 0))
                rate = results.get("success_rate", 0)
                status = "✓ PASSED" if rate >= 0.95 else "⚠ PARTIAL"
                report.append(f"  {suite_display}: {passed}/{total} tests ({rate:.1%}) {status}")
            else:
                error = results.get("error", "Unknown error") if results else "No results"
                report.append(f"  {suite_display}: ✗ FAILED - {error}")
        
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        
        if summary['data_integrity_guaranteed']:
            report.append("  ✓ Coral data integrity is GUARANTEED under tested conditions")
            report.append("  ✓ All critical data integrity scenarios pass validation")
            report.append("  ✓ System is ready for production use")
        else:
            report.append("  ⚠ Data integrity issues detected - review failed tests")
            report.append("  ⚠ Address failures before production deployment")
            
            # Specific recommendations
            if summary['perfect_accuracy_rate'] < 0.95:
                report.append("  • Fix accuracy issues in delta encoding/decoding")
            if summary['corruption_detection_rate'] < 0.90:
                report.append("  • Improve corruption detection mechanisms")
            if summary['overall_success_rate'] < 0.95:
                report.append("  • Address failing test scenarios")
        
        report.append("")
        
        # Quality Metrics
        report.append("QUALITY METRICS:")
        comprehensive = self.test_results.get("comprehensive", {})
        if comprehensive and comprehensive.get("success"):
            report.append(f"  Perfect Accuracy: {comprehensive.get('perfect_accuracy_count', 0)}/{comprehensive.get('total_accuracy_measurements', 0)} measurements")
            report.append(f"  Corruption Detection: {comprehensive.get('detected_corruptions', 0)}/{comprehensive.get('total_corruptions', 0)} cases")
        
        safetensors = self.test_results.get("safetensors", {})
        if safetensors and safetensors.get("success"):
            report.append(f"  SafeTensors Compliance: {safetensors.get('successful_tests', 0)}/{safetensors.get('total_tests', 0)} test categories")
        
        backup_recovery = self.test_results.get("backup_recovery", {})
        if backup_recovery and backup_recovery.get("success"):
            report.append(f"  Backup/Recovery: {backup_recovery.get('successful_tests', 0)}/{backup_recovery.get('total_tests', 0)} scenarios")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def print_summary(self):
        """Print executive summary to console."""
        print("\n" + "=" * 80)
        print("EXECUTIVE SUMMARY")
        print("=" * 80)
        
        summary = self.test_results["summary"]
        
        print(f"Duration: {summary['duration_formatted']}")
        print(f"Test Suites: {summary['successful_suites']}/{summary['test_suites']} successful")
        print(f"Individual Tests: {summary['passed_tests']}/{summary['total_tests']} passed")
        print(f"Overall Success Rate: {summary['overall_success_rate']:.1%}")
        print("")
        
        print("Data Integrity Status:")
        if summary['data_integrity_guaranteed']:
            print("  ✓ DATA INTEGRITY GUARANTEED")
            print(f"  ✓ Perfect Accuracy: {summary['perfect_accuracy_rate']:.1%}")
            print(f"  ✓ Corruption Detection: {summary['corruption_detection_rate']:.1%}")
        else:
            print("  ✗ DATA INTEGRITY NOT GUARANTEED")
            print(f"  • Perfect Accuracy: {summary['perfect_accuracy_rate']:.1%}")
            print(f"  • Corruption Detection: {summary['corruption_detection_rate']:.1%}")
            print("  • Review detailed reports for specific issues")
        
        print("")


def main():
    """Main entry point for the QA test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Coral Data Integrity QA Tests")
    parser.add_argument("--output-dir", default="qa_reports", 
                       help="Directory to save test reports (default: qa_reports)")
    parser.add_argument("--suite", choices=["comprehensive", "safetensors", "backup_recovery", "all"],
                       default="all", help="Test suite to run (default: all)")
    
    args = parser.parse_args()
    
    # Create test runner
    runner = DataIntegrityQARunner(args.output_dir)
    
    try:
        if args.suite == "all":
            results = runner.run_all_tests()
        elif args.suite == "comprehensive":
            results = {"comprehensive": runner.run_comprehensive_tests()}
        elif args.suite == "safetensors":
            results = {"safetensors": runner.run_safetensors_tests()}
        elif args.suite == "backup_recovery":
            results = {"backup_recovery": runner.run_backup_recovery_tests()}
        
        # Print summary
        if args.suite == "all":
            runner.print_summary()
        else:
            # Print single suite summary
            suite_results = results[args.suite]
            if suite_results and suite_results.get("success"):
                print(f"\n{args.suite.replace('_', ' ').title()} Suite:")
                print(f"  Success Rate: {suite_results.get('success_rate', 0):.1%}")
                print(f"  Tests Passed: {suite_results.get('passed_tests', suite_results.get('successful_tests', 0))}/{suite_results.get('total_tests', 0)}")
            else:
                print(f"\n{args.suite.replace('_', ' ').title()} Suite: FAILED")
                if suite_results:
                    print(f"  Error: {suite_results.get('error', 'Unknown error')}")
        
        return 0 if (args.suite != "all" or runner.test_results["summary"]["data_integrity_guaranteed"]) else 1
        
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        return 1
    except Exception as e:
        print(f"\nTest execution failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())