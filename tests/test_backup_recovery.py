"""
Backup and Recovery Integrity Tests

This module tests backup and recovery operations to ensure data integrity
is maintained during backup creation, restoration, and recovery scenarios.
"""

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import zipfile
import tarfile

import h5py
import numpy as np
import pytest

from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.version_control.repository import Repository


class BackupRecoveryIntegrityTests:
    """Test backup and recovery operations integrity."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.backup_dir = tempfile.mkdtemp()
        self.recovery_results = []
        
    def teardown_method(self):
        """Cleanup after each test method."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        shutil.rmtree(self.backup_dir, ignore_errors=True)
    
    def _create_test_weight(self, data: np.ndarray, name: str, **kwargs) -> WeightTensor:
        """Helper to create test weight tensor."""
        metadata = WeightMetadata(name=name, shape=data.shape, dtype=data.dtype, **kwargs)
        return WeightTensor(data=data, metadata=metadata)
    
    def _compute_repository_checksum(self, repo_path: str) -> str:
        """Compute checksum of entire repository state."""
        checksums = []
        
        coral_dir = os.path.join(repo_path, ".coral")
        if not os.path.exists(coral_dir):
            return ""
        
        # Walk through all files in .coral directory
        for root, dirs, files in os.walk(coral_dir):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                    file_checksum = hashlib.sha256(file_data).hexdigest()
                    relative_path = os.path.relpath(file_path, coral_dir)
                    checksums.append(f"{relative_path}:{file_checksum}")
        
        # Combine all file checksums
        combined = "\n".join(sorted(checksums))
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _create_repository_backup(self, repo_path: str, backup_path: str, 
                                  backup_type: str = "full") -> Dict[str, Any]:
        """Create repository backup using different strategies."""
        backup_info = {
            "backup_type": backup_type,
            "timestamp": time.time(),
            "source_path": repo_path,
            "backup_path": backup_path,
            "success": False
        }
        
        try:
            coral_source = os.path.join(repo_path, ".coral")
            if not os.path.exists(coral_source):
                raise ValueError("Source repository not found")
            
            if backup_type == "full":
                # Full directory copy
                coral_backup = os.path.join(backup_path, ".coral")
                shutil.copytree(coral_source, coral_backup)
                
            elif backup_type == "zip":
                # ZIP archive backup
                zip_path = os.path.join(backup_path, "coral_backup.zip")
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for root, dirs, files in os.walk(coral_source):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arc_path = os.path.relpath(file_path, repo_path)
                            zf.write(file_path, arc_path)
                backup_info["archive_path"] = zip_path
                
            elif backup_type == "tar":
                # TAR archive backup
                tar_path = os.path.join(backup_path, "coral_backup.tar.gz")
                with tarfile.open(tar_path, 'w:gz') as tf:
                    tf.add(coral_source, arcname=".coral")
                backup_info["archive_path"] = tar_path
                
            elif backup_type == "incremental":
                # Incremental backup (only changed files)
                coral_backup = os.path.join(backup_path, ".coral")
                os.makedirs(coral_backup, exist_ok=True)
                
                # Copy only files that don't exist or are newer
                for root, dirs, files in os.walk(coral_source):
                    for file in files:
                        source_file = os.path.join(root, file)
                        rel_path = os.path.relpath(source_file, coral_source)
                        backup_file = os.path.join(coral_backup, rel_path)
                        
                        # Create backup directory if needed
                        backup_file_dir = os.path.dirname(backup_file)
                        os.makedirs(backup_file_dir, exist_ok=True)
                        
                        # Check if file needs to be backed up
                        need_backup = True
                        if os.path.exists(backup_file):
                            source_mtime = os.path.getmtime(source_file)
                            backup_mtime = os.path.getmtime(backup_file)
                            need_backup = source_mtime > backup_mtime
                        
                        if need_backup:
                            shutil.copy2(source_file, backup_file)
            
            # Verify backup
            if backup_type in ["full", "incremental"]:
                backup_checksum = self._compute_repository_checksum(backup_path)
                source_checksum = self._compute_repository_checksum(repo_path)
                backup_info["checksum_match"] = backup_checksum == source_checksum
                backup_info["backup_checksum"] = backup_checksum
                backup_info["source_checksum"] = source_checksum
            
            backup_info["success"] = True
            
        except Exception as e:
            backup_info["error"] = str(e)
        
        return backup_info
    
    def _restore_repository_backup(self, backup_path: str, restore_path: str, 
                                   backup_type: str = "full") -> Dict[str, Any]:
        """Restore repository from backup."""
        restore_info = {
            "backup_type": backup_type,
            "backup_path": backup_path,
            "restore_path": restore_path,
            "success": False
        }
        
        try:
            # Clear existing repository
            coral_restore = os.path.join(restore_path, ".coral")
            if os.path.exists(coral_restore):
                shutil.rmtree(coral_restore)
            
            if backup_type == "full" or backup_type == "incremental":
                # Direct directory copy
                coral_backup = os.path.join(backup_path, ".coral")
                shutil.copytree(coral_backup, coral_restore)
                
            elif backup_type == "zip":
                # Extract ZIP archive
                zip_path = os.path.join(backup_path, "coral_backup.zip")
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(restore_path)
                    
            elif backup_type == "tar":
                # Extract TAR archive
                tar_path = os.path.join(backup_path, "coral_backup.tar.gz")
                with tarfile.open(tar_path, 'r:gz') as tf:
                    tf.extractall(restore_path)
            
            # Verify restore
            if os.path.exists(coral_restore):
                restore_info["success"] = True
                restore_info["restored_checksum"] = self._compute_repository_checksum(restore_path)
            
        except Exception as e:
            restore_info["error"] = str(e)
        
        return restore_info
    
    def test_full_backup_restore(self):
        """Test full backup and restore operations."""
        # Create test repository
        repo = Repository(self.test_dir, init=True)
        
        # Add test data
        test_weights = {}
        for i in range(10):
            data = np.random.randn(100, 100).astype(np.float32)
            test_weights[f"weight_{i}"] = self._create_test_weight(data, f"weight_{i}")
        
        repo.stage_weights(test_weights)
        original_commit = repo.commit("Original data")
        
        # Create branches and additional commits
        repo.create_branch("feature")
        repo.checkout("feature")
        feature_weight = self._create_test_weight(np.random.randn(50, 50), "feature_weight")
        repo.stage_weights({"feature_weight": feature_weight})
        feature_commit = repo.commit("Feature commit")
        
        repo.checkout("main")
        
        # Compute original checksum
        original_checksum = self._compute_repository_checksum(self.test_dir)
        
        # Test different backup methods
        backup_methods = ["full", "zip", "tar", "incremental"]
        
        for method in backup_methods:
            method_backup_dir = os.path.join(self.backup_dir, f"backup_{method}")
            os.makedirs(method_backup_dir, exist_ok=True)
            
            # Create backup
            backup_info = self._create_repository_backup(
                self.test_dir, method_backup_dir, method
            )
            
            # Test restore
            restore_dir = os.path.join(self.backup_dir, f"restore_{method}")
            os.makedirs(restore_dir, exist_ok=True)
            
            restore_info = self._restore_repository_backup(
                method_backup_dir, restore_dir, method
            )
            
            # Verify restored repository
            verification_results = self._verify_restored_repository(
                restore_dir, test_weights, original_commit, feature_commit
            )
            
            self.recovery_results.append({
                "test": f"full_backup_restore_{method}",
                "backup_info": backup_info,
                "restore_info": restore_info,
                "verification": verification_results,
                "original_checksum": original_checksum,
                "success": backup_info["success"] and restore_info["success"] and verification_results["success"]
            })
    
    def _verify_restored_repository(self, repo_path: str, expected_weights: Dict[str, WeightTensor],
                                    original_commit, feature_commit) -> Dict[str, Any]:
        """Verify restored repository integrity."""
        verification = {
            "success": False,
            "weights_verified": 0,
            "commits_verified": 0,
            "branches_verified": 0,
            "errors": []
        }
        
        try:
            # Load restored repository
            restored_repo = Repository(repo_path)
            
            # Verify weights
            restored_weights = restored_repo.get_all_weights()
            verification["total_weights_found"] = len(restored_weights)
            verification["expected_weights"] = len(expected_weights)
            
            for weight_name, expected_weight in expected_weights.items():
                try:
                    restored_weight = restored_repo.get_weight(weight_name)
                    if np.array_equal(expected_weight.data, restored_weight.data):
                        verification["weights_verified"] += 1
                    else:
                        verification["errors"].append(f"Weight {weight_name} data mismatch")
                except Exception as e:
                    verification["errors"].append(f"Failed to load weight {weight_name}: {e}")
            
            # Verify commits
            try:
                restored_original = restored_repo.get_commit(original_commit.commit_hash)
                if restored_original:
                    verification["commits_verified"] += 1
                else:
                    verification["errors"].append("Original commit not found")
            except Exception as e:
                verification["errors"].append(f"Failed to verify original commit: {e}")
            
            try:
                restored_feature = restored_repo.get_commit(feature_commit.commit_hash)
                if restored_feature:
                    verification["commits_verified"] += 1
                else:
                    verification["errors"].append("Feature commit not found")
            except Exception as e:
                verification["errors"].append(f"Failed to verify feature commit: {e}")
            
            # Verify branches
            try:
                branches = restored_repo.branch_manager.list_branches()
                expected_branches = {"main", "feature"}
                found_branches = set(branches)
                
                if expected_branches.issubset(found_branches):
                    verification["branches_verified"] = len(expected_branches)
                else:
                    missing = expected_branches - found_branches
                    verification["errors"].append(f"Missing branches: {missing}")
                    
            except Exception as e:
                verification["errors"].append(f"Failed to verify branches: {e}")
            
            # Overall success
            verification["success"] = (
                verification["weights_verified"] == len(expected_weights) and
                verification["commits_verified"] == 2 and
                verification["branches_verified"] == 2 and
                len(verification["errors"]) == 0
            )
            
        except Exception as e:
            verification["errors"].append(f"Repository verification failed: {e}")
        
        return verification
    
    def test_partial_corruption_recovery(self):
        """Test recovery from partial repository corruption."""
        # Create test repository
        repo = Repository(self.test_dir, init=True)
        
        # Add test data
        test_weights = {}
        for i in range(5):
            data = np.random.randn(50, 50).astype(np.float32)
            test_weights[f"weight_{i}"] = self._create_test_weight(data, f"weight_{i}")
        
        repo.stage_weights(test_weights)
        commit = repo.commit("Test data")
        
        # Create backup
        backup_info = self._create_repository_backup(self.test_dir, self.backup_dir, "full")
        
        # Simulate partial corruption
        hdf5_path = os.path.join(self.test_dir, ".coral", "objects", "weights.h5")
        
        # Corrupt middle of file
        with open(hdf5_path, 'r+b') as f:
            file_size = f.seek(0, 2)  # Seek to end to get size
            f.seek(file_size // 2)
            f.write(b"CORRUPTED" * 1000)
        
        # Attempt to use corrupted repository
        corruption_detected = False
        try:
            corrupted_repo = Repository(self.test_dir)
            corrupted_weights = corrupted_repo.get_all_weights()
        except Exception as e:
            corruption_detected = True
            corruption_error = str(e)
        
        # Recover from backup
        restore_info = self._restore_repository_backup(self.backup_dir, self.test_dir, "full")
        
        # Verify recovery
        try:
            recovered_repo = Repository(self.test_dir)
            recovered_weights = recovered_repo.get_all_weights()
            
            recovery_success = len(recovered_weights) == len(test_weights)
            
            # Verify data integrity
            data_integrity = True
            for weight_name, expected_weight in test_weights.items():
                recovered_weight = recovered_repo.get_weight(weight_name)
                if not np.array_equal(expected_weight.data, recovered_weight.data):
                    data_integrity = False
                    break
            
            self.recovery_results.append({
                "test": "partial_corruption_recovery",
                "corruption_detected": corruption_detected,
                "backup_success": backup_info["success"],
                "restore_success": restore_info["success"],
                "recovery_success": recovery_success,
                "data_integrity": data_integrity,
                "weights_recovered": len(recovered_weights),
                "expected_weights": len(test_weights)
            })
            
        except Exception as e:
            self.recovery_results.append({
                "test": "partial_corruption_recovery",
                "corruption_detected": corruption_detected,
                "backup_success": backup_info["success"],
                "restore_success": restore_info["success"],
                "recovery_success": False,
                "error": str(e)
            })
    
    def test_incremental_backup_consistency(self):
        """Test incremental backup consistency."""
        # Create initial repository
        repo = Repository(self.test_dir, init=True)
        
        # Add initial weights
        initial_weights = {}
        for i in range(3):
            data = np.random.randn(30, 30).astype(np.float32)
            initial_weights[f"initial_{i}"] = self._create_test_weight(data, f"initial_{i}")
        
        repo.stage_weights(initial_weights)
        repo.commit("Initial commit")
        
        # Create initial backup
        initial_backup_dir = os.path.join(self.backup_dir, "initial")
        os.makedirs(initial_backup_dir, exist_ok=True)
        initial_backup = self._create_repository_backup(
            self.test_dir, initial_backup_dir, "full"
        )
        
        # Add more weights
        time.sleep(1)  # Ensure different timestamps
        additional_weights = {}
        for i in range(2):
            data = np.random.randn(20, 20).astype(np.float32)
            additional_weights[f"additional_{i}"] = self._create_test_weight(data, f"additional_{i}")
        
        repo.stage_weights(additional_weights)
        repo.commit("Additional commit")
        
        # Create incremental backup
        incremental_backup_dir = os.path.join(self.backup_dir, "incremental")
        os.makedirs(incremental_backup_dir, exist_ok=True)
        
        # First copy initial backup as base
        shutil.copytree(
            os.path.join(initial_backup_dir, ".coral"),
            os.path.join(incremental_backup_dir, ".coral")
        )
        
        # Then apply incremental backup
        incremental_backup = self._create_repository_backup(
            self.test_dir, incremental_backup_dir, "incremental"
        )
        
        # Verify incremental backup
        restore_dir = os.path.join(self.backup_dir, "incremental_restore")
        os.makedirs(restore_dir, exist_ok=True)
        
        restore_info = self._restore_repository_backup(
            incremental_backup_dir, restore_dir, "incremental"
        )
        
        # Verify all weights are present
        try:
            restored_repo = Repository(restore_dir)
            restored_weights = restored_repo.get_all_weights()
            
            all_expected_weights = {**initial_weights, **additional_weights}
            incremental_success = len(restored_weights) == len(all_expected_weights)
            
            # Verify data integrity
            data_integrity = True
            for weight_name, expected_weight in all_expected_weights.items():
                try:
                    restored_weight = restored_repo.get_weight(weight_name)
                    if not np.array_equal(expected_weight.data, restored_weight.data):
                        data_integrity = False
                        break
                except Exception:
                    data_integrity = False
                    break
            
            self.recovery_results.append({
                "test": "incremental_backup_consistency",
                "initial_backup_success": initial_backup["success"],
                "incremental_backup_success": incremental_backup["success"],
                "restore_success": restore_info["success"],
                "incremental_success": incremental_success,
                "data_integrity": data_integrity,
                "initial_weights": len(initial_weights),
                "additional_weights": len(additional_weights),
                "restored_weights": len(restored_weights)
            })
            
        except Exception as e:
            self.recovery_results.append({
                "test": "incremental_backup_consistency",
                "initial_backup_success": initial_backup.get("success", False),
                "incremental_backup_success": incremental_backup.get("success", False),
                "restore_success": restore_info.get("success", False),
                "error": str(e)
            })
    
    def test_backup_compression_integrity(self):
        """Test backup compression and integrity."""
        # Create test repository with large data
        repo = Repository(self.test_dir, init=True)
        
        # Create large weights to test compression
        large_weights = {}
        for i in range(3):
            # Create large but compressible data (repeated patterns)
            base_pattern = np.random.randn(100, 100).astype(np.float32)
            large_data = np.tile(base_pattern, (5, 5))  # 500x500 with repeated pattern
            large_weights[f"large_{i}"] = self._create_test_weight(large_data, f"large_{i}")
        
        repo.stage_weights(large_weights)
        repo.commit("Large weights")
        
        # Test different compression methods
        compression_methods = ["zip", "tar"]
        
        for method in compression_methods:
            method_backup_dir = os.path.join(self.backup_dir, f"compressed_{method}")
            os.makedirs(method_backup_dir, exist_ok=True)
            
            # Create compressed backup
            backup_info = self._create_repository_backup(
                self.test_dir, method_backup_dir, method
            )
            
            # Check compression ratio
            if method == "zip":
                archive_path = os.path.join(method_backup_dir, "coral_backup.zip")
            else:
                archive_path = os.path.join(method_backup_dir, "coral_backup.tar.gz")
            
            if os.path.exists(archive_path):
                original_size = sum(
                    os.path.getsize(os.path.join(root, file))
                    for root, dirs, files in os.walk(os.path.join(self.test_dir, ".coral"))
                    for file in files
                )
                compressed_size = os.path.getsize(archive_path)
                compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
                
                backup_info["original_size"] = original_size
                backup_info["compressed_size"] = compressed_size
                backup_info["compression_ratio"] = compression_ratio
            
            # Test restore
            restore_dir = os.path.join(self.backup_dir, f"restore_compressed_{method}")
            os.makedirs(restore_dir, exist_ok=True)
            
            restore_info = self._restore_repository_backup(
                method_backup_dir, restore_dir, method
            )
            
            # Verify integrity after compression/decompression
            integrity_verified = False
            try:
                restored_repo = Repository(restore_dir)
                restored_weights = restored_repo.get_all_weights()
                
                if len(restored_weights) == len(large_weights):
                    integrity_verified = True
                    for weight_name, expected_weight in large_weights.items():
                        restored_weight = restored_repo.get_weight(weight_name)
                        if not np.array_equal(expected_weight.data, restored_weight.data):
                            integrity_verified = False
                            break
                            
            except Exception as e:
                restore_info["verification_error"] = str(e)
            
            self.recovery_results.append({
                "test": f"backup_compression_integrity_{method}",
                "backup_info": backup_info,
                "restore_info": restore_info,
                "integrity_verified": integrity_verified,
                "compression_ratio": backup_info.get("compression_ratio", 1.0)
            })
    
    def generate_recovery_report(self) -> str:
        """Generate comprehensive backup and recovery report."""
        report = []
        report.append("=" * 80)
        report.append("BACKUP AND RECOVERY INTEGRITY REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        total_tests = len(self.recovery_results)
        successful_tests = sum(1 for r in self.recovery_results if r.get("success", False))
        
        report.append(f"SUMMARY:")
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Successful Tests: {successful_tests}")
        report.append(f"Success Rate: {successful_tests/total_tests*100:.1f}%" if total_tests > 0 else "Success Rate: N/A")
        report.append("")
        
        # Detailed results
        for result in self.recovery_results:
            test_name = result["test"]
            report.append(f"TEST: {test_name.upper()}")
            report.append("-" * 40)
            
            # Test-specific reporting
            if "full_backup_restore" in test_name:
                backup_info = result.get("backup_info", {})
                restore_info = result.get("restore_info", {})
                verification = result.get("verification", {})
                
                report.append(f"Backup Success: {backup_info.get('success', False)}")
                report.append(f"Restore Success: {restore_info.get('success', False)}")
                report.append(f"Weights Verified: {verification.get('weights_verified', 0)}/{verification.get('expected_weights', 0)}")
                report.append(f"Commits Verified: {verification.get('commits_verified', 0)}")
                report.append(f"Branches Verified: {verification.get('branches_verified', 0)}")
                
                if verification.get("errors"):
                    report.append("Errors:")
                    for error in verification["errors"]:
                        report.append(f"  - {error}")
                        
            elif "partial_corruption_recovery" in test_name:
                report.append(f"Corruption Detected: {result.get('corruption_detected', False)}")
                report.append(f"Recovery Success: {result.get('recovery_success', False)}")
                report.append(f"Data Integrity: {result.get('data_integrity', False)}")
                report.append(f"Weights Recovered: {result.get('weights_recovered', 0)}/{result.get('expected_weights', 0)}")
                
            elif "incremental_backup" in test_name:
                report.append(f"Initial Backup: {result.get('initial_backup_success', False)}")
                report.append(f"Incremental Backup: {result.get('incremental_backup_success', False)}")
                report.append(f"Restore Success: {result.get('restore_success', False)}")
                report.append(f"Data Integrity: {result.get('data_integrity', False)}")
                report.append(f"Weights: {result.get('initial_weights', 0)} + {result.get('additional_weights', 0)} = {result.get('restored_weights', 0)}")
                
            elif "compression_integrity" in test_name:
                backup_info = result.get("backup_info", {})
                report.append(f"Backup Success: {backup_info.get('success', False)}")
                report.append(f"Compression Ratio: {result.get('compression_ratio', 1.0):.3f}")
                report.append(f"Integrity Verified: {result.get('integrity_verified', False)}")
                
                if "original_size" in backup_info and "compressed_size" in backup_info:
                    orig_size = backup_info["original_size"]
                    comp_size = backup_info["compressed_size"]
                    report.append(f"Size: {orig_size} → {comp_size} bytes")
            
            # Overall success
            overall_success = result.get("success", False)
            report.append(f"Overall Success: {overall_success}")
            
            if "error" in result:
                report.append(f"Error: {result['error']}")
            
            report.append("")
        
        return "\n".join(report)
    
    def run_all_tests(self):
        """Run all backup and recovery tests."""
        print("Running Backup and Recovery Integrity Tests...")
        print("=" * 50)
        
        test_methods = [
            self.test_full_backup_restore,
            self.test_partial_corruption_recovery,
            self.test_incremental_backup_consistency, 
            self.test_backup_compression_integrity,
        ]
        
        for test_method in test_methods:
            try:
                print(f"Running {test_method.__name__}...")
                test_method()
                print(f"✓ {test_method.__name__} completed")
            except Exception as e:
                print(f"✗ {test_method.__name__} failed: {e}")
                self.recovery_results.append({
                    "test": test_method.__name__,
                    "success": False,
                    "error": str(e)
                })
        
        # Generate report
        report = self.generate_recovery_report()
        print("\n" + report)
        
        return self.recovery_results


def run_backup_recovery_tests():
    """Run backup and recovery integrity tests."""
    test_instance = BackupRecoveryIntegrityTests()
    test_instance.setup_method()
    
    try:
        return test_instance.run_all_tests()
    finally:
        test_instance.teardown_method()


if __name__ == "__main__":
    run_backup_recovery_tests()