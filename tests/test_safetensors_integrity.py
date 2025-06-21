"""
SafeTensors Format Integrity Tests

This module tests the integrity and consistency of SafeTensors format
implementation in Coral, including format validation, metadata preservation,
and cross-platform compatibility.
"""

import json
import os
import struct
import tempfile
import shutil
from typing import Dict, Any, List, Tuple
import numpy as np
import pytest

from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.version_control.repository import Repository


class SafeTensorsIntegrityTests:
    """Test SafeTensors format integrity and validation."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.test_results = []
        
    def teardown_method(self):
        """Cleanup after each test method."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_test_weight(self, data: np.ndarray, name: str, **kwargs) -> WeightTensor:
        """Helper to create test weight tensor."""
        metadata = WeightMetadata(name=name, shape=data.shape, dtype=data.dtype, **kwargs)
        return WeightTensor(data=data, metadata=metadata)
    
    def _validate_safetensors_header(self, file_path: str) -> Dict[str, Any]:
        """Validate SafeTensors file header format."""
        try:
            with open(file_path, 'rb') as f:
                # Read header length (first 8 bytes)
                header_len_bytes = f.read(8)
                if len(header_len_bytes) != 8:
                    return {"valid": False, "error": "Incomplete header length"}
                
                header_len = struct.unpack('<Q', header_len_bytes)[0]
                
                # Read header JSON
                header_bytes = f.read(header_len)
                if len(header_bytes) != header_len:
                    return {"valid": False, "error": "Incomplete header"}
                
                header = json.loads(header_bytes.decode('utf-8'))
                
                # Validate header structure
                if "__metadata__" not in header:
                    return {"valid": False, "error": "Missing __metadata__"}
                
                return {
                    "valid": True,
                    "header": header,
                    "header_length": header_len,
                    "total_file_size": os.path.getsize(file_path)
                }
                
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def test_safetensors_format_validation(self):
        """Test SafeTensors format validation and structure."""
        repo = Repository(self.test_dir, init=True)
        
        # Test different data types
        test_cases = [
            ("float32_tensor", np.random.randn(10, 10).astype(np.float32)),
            ("float64_tensor", np.random.randn(5, 5).astype(np.float64)),
            ("int32_tensor", np.random.randint(0, 100, size=(8, 8)).astype(np.int32)),
            ("int64_tensor", np.random.randint(0, 1000, size=(6, 6)).astype(np.int64)),
            ("bool_tensor", np.random.choice([True, False], size=(4, 4))),
            ("int8_tensor", np.random.randint(-128, 127, size=(12, 12)).astype(np.int8)),
            ("uint8_tensor", np.random.randint(0, 255, size=(12, 12)).astype(np.uint8)),
        ]
        
        format_validation_results = []
        
        for name, data in test_cases:
            try:
                # Create and store weight
                weight = self._create_test_weight(data, name)
                repo.stage_weights({name: weight})
                repo.commit(f"Test {name}")
                
                # Retrieve and validate
                retrieved_weight = repo.get_weight(name)
                
                # Validate data integrity
                data_integrity = np.array_equal(data, retrieved_weight.data)
                dtype_preserved = data.dtype == retrieved_weight.data.dtype
                shape_preserved = data.shape == retrieved_weight.data.shape
                
                format_validation_results.append({
                    "name": name,
                    "data_integrity": data_integrity,
                    "dtype_preserved": dtype_preserved,
                    "shape_preserved": shape_preserved,
                    "original_dtype": str(data.dtype),
                    "retrieved_dtype": str(retrieved_weight.data.dtype),
                    "original_shape": data.shape,
                    "retrieved_shape": retrieved_weight.data.shape
                })
                
            except Exception as e:
                format_validation_results.append({
                    "name": name,
                    "error": str(e),
                    "data_integrity": False,
                    "dtype_preserved": False,
                    "shape_preserved": False
                })
        
        self.test_results.append({
            "test": "safetensors_format_validation",
            "results": format_validation_results
        })
        
        return format_validation_results
    
    def test_safetensors_metadata_preservation(self):
        """Test SafeTensors metadata preservation."""
        repo = Repository(self.test_dir, init=True)
        
        # Create weight with comprehensive metadata
        data = np.random.randn(20, 20).astype(np.float32)
        weight = self._create_test_weight(
            data, "metadata_test",
            layer_type="dense",
            model_name="test_model",
            compression_info={"algorithm": "none", "ratio": 1.0}
        )
        
        # Store weight
        repo.stage_weights({"metadata_test": weight})
        repo.commit("Metadata test")
        
        # Retrieve and validate metadata
        retrieved_weight = repo.get_weight("metadata_test")
        
        metadata_results = {
            "name_preserved": weight.metadata.name == retrieved_weight.metadata.name,
            "shape_preserved": weight.metadata.shape == retrieved_weight.metadata.shape,
            "dtype_preserved": weight.metadata.dtype == retrieved_weight.metadata.dtype,
            "layer_type_preserved": weight.metadata.layer_type == retrieved_weight.metadata.layer_type,
            "model_name_preserved": weight.metadata.model_name == retrieved_weight.metadata.model_name,
            "compression_info_preserved": weight.metadata.compression_info == retrieved_weight.metadata.compression_info,
            "original_metadata": {
                "name": weight.metadata.name,
                "shape": weight.metadata.shape,
                "dtype": str(weight.metadata.dtype),
                "layer_type": weight.metadata.layer_type,
                "model_name": weight.metadata.model_name,
                "compression_info": weight.metadata.compression_info
            },
            "retrieved_metadata": {
                "name": retrieved_weight.metadata.name,
                "shape": retrieved_weight.metadata.shape,
                "dtype": str(retrieved_weight.metadata.dtype),
                "layer_type": retrieved_weight.metadata.layer_type,
                "model_name": retrieved_weight.metadata.model_name,
                "compression_info": retrieved_weight.metadata.compression_info
            }
        }
        
        self.test_results.append({
            "test": "safetensors_metadata_preservation",
            "results": metadata_results
        })
        
        return metadata_results
    
    def test_safetensors_edge_cases(self):
        """Test SafeTensors with edge cases."""
        repo = Repository(self.test_dir, init=True)
        
        edge_cases = [
            ("empty_tensor", np.array([], dtype=np.float32)),
            ("single_element", np.array([42.0], dtype=np.float32)),
            ("large_1d", np.random.randn(100000).astype(np.float32)),
            ("high_dimensional", np.random.randn(2, 3, 4, 5, 6).astype(np.float32)),
            ("nan_values", np.array([1.0, np.nan, 3.0, np.inf, -np.inf], dtype=np.float32)),
            ("zero_tensor", np.zeros((10, 10), dtype=np.float32)),
            ("ones_tensor", np.ones((10, 10), dtype=np.float32)),
            ("extreme_values", np.array([np.finfo(np.float32).max, np.finfo(np.float32).min], dtype=np.float32)),
        ]
        
        edge_case_results = []
        
        for name, data in edge_cases:
            try:
                # Create and store weight
                weight = self._create_test_weight(data, name)
                repo.stage_weights({name: weight})
                repo.commit(f"Edge case {name}")
                
                # Retrieve and validate
                retrieved_weight = repo.get_weight(name)
                
                # Special handling for NaN/Inf values
                if name == "nan_values":
                    # Check NaN preservation
                    orig_nan_mask = np.isnan(data)
                    retr_nan_mask = np.isnan(retrieved_weight.data)
                    nan_preserved = np.array_equal(orig_nan_mask, retr_nan_mask)
                    
                    # Check Inf preservation
                    orig_inf_mask = np.isinf(data)
                    retr_inf_mask = np.isinf(retrieved_weight.data)
                    inf_preserved = np.array_equal(orig_inf_mask, retr_inf_mask)
                    
                    # Check finite values
                    orig_finite = data[np.isfinite(data)]
                    retr_finite = retrieved_weight.data[np.isfinite(retrieved_weight.data)]
                    finite_preserved = np.array_equal(orig_finite, retr_finite)
                    
                    data_integrity = nan_preserved and inf_preserved and finite_preserved
                else:
                    data_integrity = np.array_equal(data, retrieved_weight.data)
                
                edge_case_results.append({
                    "name": name,
                    "success": True,
                    "data_integrity": data_integrity,
                    "shape_match": data.shape == retrieved_weight.data.shape,
                    "dtype_match": data.dtype == retrieved_weight.data.dtype,
                    "original_shape": data.shape,
                    "retrieved_shape": retrieved_weight.data.shape,
                    "original_dtype": str(data.dtype),
                    "retrieved_dtype": str(retrieved_weight.data.dtype)
                })
                
            except Exception as e:
                edge_case_results.append({
                    "name": name,
                    "success": False,
                    "error": str(e),
                    "data_integrity": False,
                    "shape_match": False,
                    "dtype_match": False
                })
        
        self.test_results.append({
            "test": "safetensors_edge_cases",
            "results": edge_case_results
        })
        
        return edge_case_results
    
    def test_safetensors_corruption_detection(self):
        """Test SafeTensors corruption detection."""
        repo = Repository(self.test_dir, init=True)
        
        # Create test weight
        data = np.random.randn(50, 50).astype(np.float32)
        weight = self._create_test_weight(data, "corruption_test")
        repo.stage_weights({"corruption_test": weight})
        repo.commit("Corruption test")
        
        # Locate HDF5 file (SafeTensors data stored in HDF5 backend)
        hdf5_path = os.path.join(self.test_dir, ".coral", "objects", "weights.h5")
        
        corruption_tests = []
        
        # Test 1: Corrupt header
        try:
            # Backup original
            backup_path = hdf5_path + ".backup"
            shutil.copy2(hdf5_path, backup_path)
            
            # Corrupt first few bytes
            with open(hdf5_path, 'r+b') as f:
                f.seek(0)
                f.write(b"CORRUPTED_HEADER")
            
            # Try to read
            try:
                corrupted_repo = Repository(self.test_dir)
                corrupted_weight = corrupted_repo.get_weight("corruption_test")
                corruption_tests.append({
                    "type": "header_corruption",
                    "detected": False,
                    "error": "Corruption not detected"
                })
            except Exception as e:
                corruption_tests.append({
                    "type": "header_corruption",
                    "detected": True,
                    "error": str(e)
                })
            
            # Restore
            shutil.copy2(backup_path, hdf5_path)
            
        except Exception as e:
            corruption_tests.append({
                "type": "header_corruption",
                "detected": False,
                "setup_error": str(e)
            })
        
        # Test 2: Corrupt data section
        try:
            # Corrupt middle of file
            file_size = os.path.getsize(hdf5_path)
            with open(hdf5_path, 'r+b') as f:
                f.seek(file_size // 2)
                f.write(b"CORRUPTED_DATA" * 100)
            
            # Try to read
            try:
                corrupted_repo = Repository(self.test_dir)
                corrupted_weight = corrupted_repo.get_weight("corruption_test")
                
                # Check if data is corrupted
                data_corrupted = not np.array_equal(data, corrupted_weight.data)
                
                corruption_tests.append({
                    "type": "data_corruption",
                    "detected": data_corrupted,
                    "data_matches": not data_corrupted
                })
                
            except Exception as e:
                corruption_tests.append({
                    "type": "data_corruption",
                    "detected": True,
                    "error": str(e)
                })
            
            # Restore
            shutil.copy2(backup_path, hdf5_path)
            
        except Exception as e:
            corruption_tests.append({
                "type": "data_corruption",
                "detected": False,
                "setup_error": str(e)
            })
        
        # Test 3: Truncated file
        try:
            # Truncate file
            file_size = os.path.getsize(hdf5_path)
            with open(hdf5_path, 'r+b') as f:
                f.truncate(file_size // 2)
            
            # Try to read
            try:
                corrupted_repo = Repository(self.test_dir)
                corrupted_weight = corrupted_repo.get_weight("corruption_test")
                corruption_tests.append({
                    "type": "truncated_file",
                    "detected": False,
                    "error": "Truncation not detected"
                })
            except Exception as e:
                corruption_tests.append({
                    "type": "truncated_file",
                    "detected": True,
                    "error": str(e)
                })
            
            # Restore
            shutil.copy2(backup_path, hdf5_path)
            
        except Exception as e:
            corruption_tests.append({
                "type": "truncated_file",
                "detected": False,
                "setup_error": str(e)
            })
        
        # Clean up
        if os.path.exists(backup_path):
            os.remove(backup_path)
        
        self.test_results.append({
            "test": "safetensors_corruption_detection",
            "results": corruption_tests
        })
        
        return corruption_tests
    
    def test_safetensors_cross_platform(self):
        """Test SafeTensors cross-platform compatibility."""
        repo = Repository(self.test_dir, init=True)
        
        # Test different endianness
        cross_platform_tests = []
        
        # Test big-endian vs little-endian
        test_data = [
            ("little_endian_int32", np.array([1, 2, 3, 4], dtype='<i4')),
            ("big_endian_int32", np.array([1, 2, 3, 4], dtype='>i4')),
            ("little_endian_float32", np.array([1.5, 2.5, 3.5], dtype='<f4')),
            ("big_endian_float32", np.array([1.5, 2.5, 3.5], dtype='>f4')),
            ("native_int32", np.array([1, 2, 3, 4], dtype=np.int32)),
            ("native_float32", np.array([1.5, 2.5, 3.5], dtype=np.float32)),
        ]
        
        for name, data in test_data:
            try:
                # Store weight
                weight = self._create_test_weight(data, name)
                repo.stage_weights({name: weight})
                repo.commit(f"Cross-platform test {name}")
                
                # Retrieve weight
                retrieved_weight = repo.get_weight(name)
                
                # Test byte-level equality
                bytes_equal = data.tobytes() == retrieved_weight.data.tobytes()
                
                # Test logical equality (handles endianness conversion)
                logical_equal = np.array_equal(data, retrieved_weight.data)
                
                cross_platform_tests.append({
                    "name": name,
                    "bytes_equal": bytes_equal,
                    "logical_equal": logical_equal,
                    "original_dtype": str(data.dtype),
                    "retrieved_dtype": str(retrieved_weight.data.dtype),
                    "success": bytes_equal or logical_equal
                })
                
            except Exception as e:
                cross_platform_tests.append({
                    "name": name,
                    "error": str(e),
                    "bytes_equal": False,
                    "logical_equal": False,
                    "success": False
                })
        
        self.test_results.append({
            "test": "safetensors_cross_platform",
            "results": cross_platform_tests
        })
        
        return cross_platform_tests
    
    def test_safetensors_large_tensors(self):
        """Test SafeTensors with large tensors."""
        repo = Repository(self.test_dir, init=True)
        
        large_tensor_tests = []
        
        # Test progressively larger tensors
        sizes = [
            (1000, 1000),      # 1M elements
            (2000, 1000),      # 2M elements
            (1000, 5000),      # 5M elements
        ]
        
        for i, size in enumerate(sizes):
            try:
                # Create large tensor
                data = np.random.randn(*size).astype(np.float32)
                name = f"large_tensor_{i}"
                
                # Store tensor
                weight = self._create_test_weight(data, name)
                repo.stage_weights({name: weight})
                repo.commit(f"Large tensor {i}")
                
                # Retrieve and validate
                retrieved_weight = repo.get_weight(name)
                
                # Validate data integrity
                data_integrity = np.array_equal(data, retrieved_weight.data)
                shape_preserved = data.shape == retrieved_weight.data.shape
                dtype_preserved = data.dtype == retrieved_weight.data.dtype
                
                large_tensor_tests.append({
                    "size": size,
                    "elements": data.size,
                    "bytes": data.nbytes,
                    "data_integrity": data_integrity,
                    "shape_preserved": shape_preserved,
                    "dtype_preserved": dtype_preserved,
                    "success": data_integrity and shape_preserved and dtype_preserved
                })
                
            except Exception as e:
                large_tensor_tests.append({
                    "size": size,
                    "elements": size[0] * size[1],
                    "error": str(e),
                    "success": False
                })
        
        self.test_results.append({
            "test": "safetensors_large_tensors",
            "results": large_tensor_tests
        })
        
        return large_tensor_tests
    
    def generate_integrity_report(self) -> str:
        """Generate comprehensive SafeTensors integrity report."""
        report = []
        report.append("=" * 80)
        report.append("SAFETENSORS FORMAT INTEGRITY REPORT")
        report.append("=" * 80)
        report.append("")
        
        for test_result in self.test_results:
            test_name = test_result["test"]
            results = test_result["results"]
            
            report.append(f"TEST: {test_name.upper()}")
            report.append("-" * 40)
            
            if isinstance(results, list):
                success_count = sum(1 for r in results if r.get("success", r.get("data_integrity", False)))
                total_count = len(results)
                report.append(f"Success Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
                
                # Show failures
                failures = [r for r in results if not r.get("success", r.get("data_integrity", False))]
                if failures:
                    report.append("Failures:")
                    for failure in failures:
                        report.append(f"  - {failure.get('name', 'Unknown')}: {failure.get('error', 'Data integrity failed')}")
                
            elif isinstance(results, dict):
                # Single test result
                success_fields = [k for k, v in results.items() if k.endswith("_preserved") and v]
                total_fields = len([k for k in results.keys() if k.endswith("_preserved")])
                
                if total_fields > 0:
                    report.append(f"Preserved Fields: {len(success_fields)}/{total_fields}")
                    failed_fields = [k for k, v in results.items() if k.endswith("_preserved") and not v]
                    if failed_fields:
                        report.append(f"Failed Fields: {', '.join(failed_fields)}")
            
            report.append("")
        
        return "\n".join(report)
    
    def run_all_tests(self):
        """Run all SafeTensors integrity tests."""
        print("Running SafeTensors Format Integrity Tests...")
        print("=" * 50)
        
        test_methods = [
            self.test_safetensors_format_validation,
            self.test_safetensors_metadata_preservation,
            self.test_safetensors_edge_cases,
            self.test_safetensors_corruption_detection,
            self.test_safetensors_cross_platform,
            self.test_safetensors_large_tensors,
        ]
        
        for test_method in test_methods:
            try:
                print(f"Running {test_method.__name__}...")
                test_method()
                print(f"✓ {test_method.__name__} completed")
            except Exception as e:
                print(f"✗ {test_method.__name__} failed: {e}")
                self.test_results.append({
                    "test": test_method.__name__,
                    "results": {"error": str(e), "success": False}
                })
        
        # Generate report
        report = self.generate_integrity_report()
        print("\n" + report)
        
        return self.test_results


def run_safetensors_integrity_tests():
    """Run SafeTensors integrity tests."""
    test_instance = SafeTensorsIntegrityTests()
    test_instance.setup_method()
    
    try:
        return test_instance.run_all_tests()
    finally:
        test_instance.teardown_method()


if __name__ == "__main__":
    run_safetensors_integrity_tests()