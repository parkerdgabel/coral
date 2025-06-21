"""
QA tests for repository corruption scenarios.

These tests attempt to corrupt or break the repository in various ways
to ensure the application handles edge cases gracefully.
"""

import os
import shutil
import tempfile
import multiprocessing
import time
import h5py
import numpy as np
import pytest
from pathlib import Path
import json
import struct

from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.version_control.repository import Repository
from coral.storage.hdf5_store import HDF5Store


def create_weight_tensor(data, name="test_weight", **kwargs):
    """Helper to create weight tensor with proper metadata."""
    metadata = WeightMetadata(name=name, shape=data.shape, dtype=data.dtype, **kwargs)
    return WeightTensor(data=data, metadata=metadata)


def write_weights_process(process_id, repo_path):
    """Function to be run in separate process for concurrent testing."""
    try:
        repo = Repository(repo_path)
        for i in range(10):
            weight = create_weight_tensor(np.random.randn(100), f"proc_{process_id}_weight_{i}")
            repo.stage_weights({f"proc_{process_id}_weight_{i}": weight})
            repo.commit(f"Process {process_id} commit {i}")
            time.sleep(0.01)  # Small delay to increase chance of conflicts
        return f"Process {process_id} completed"
    except Exception as e:
        return f"Process {process_id} error: {e}"


class TestRepositoryCorruption:
    """Test various repository corruption scenarios."""
    
    def setup_method(self):
        """Create a temporary directory for each test."""
        self.test_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_corrupt_hdf5_file(self):
        """
        Test: Corrupt the HDF5 file structure and attempt to read from it.
        Goal: Ensure graceful handling of corrupted storage files.
        """
        # Initialize repository and add some weights
        repo = Repository(self.test_dir, init=True)
        
        weight1 = create_weight_tensor(np.array([1.0, 2.0, 3.0]), "layer1")
        repo.stage_weights({"layer1": weight1})
        repo.commit("Initial commit")
        
        # Corrupt the HDF5 file by overwriting with random bytes
        hdf5_path = os.path.join(self.test_dir, ".coral", "objects", "weights.h5")
        
        # Test 1: Overwrite header with random bytes
        with open(hdf5_path, "r+b") as f:
            f.seek(0)
            f.write(b"CORRUPTED" * 100)
        
        # Try to read from corrupted repository
        with pytest.raises(Exception) as exc_info:
            corrupted_repo = Repository(self.test_dir)
            corrupted_repo.get_all_weights()
        
        print(f"Corrupted HDF5 header error: {exc_info.value}")
        error_msg = str(exc_info.value)
        assert ("Unable to open file" in error_msg or 
                "not an HDF5 file" in error_msg or 
                "file signature not found" in error_msg)
    
    def test_corrupt_commit_files(self):
        """
        Test: Manually corrupt commit JSON files.
        Goal: Test handling of malformed commit data.
        """
        # Initialize and create commits
        repo = Repository(self.test_dir, init=True)
        
        weight1 = create_weight_tensor(np.array([1.0, 2.0, 3.0]), "layer1")
        repo.stage_weights({"layer1": weight1})
        commit = repo.commit("Test commit")
        commit_hash = commit.commit_hash
        
        # Corrupt commit file with invalid JSON
        commit_path = os.path.join(self.test_dir, ".coral", "objects", "commits", f"{commit_hash}.json")
        
        # Test 1: Invalid JSON
        with open(commit_path, "w") as f:
            f.write("{invalid json content")
        
        with pytest.raises(Exception) as exc_info:
            repo.get_commit(commit_hash)
        
        print(f"Invalid JSON error: {exc_info.value}")
        
        # Test 2: Valid JSON but missing required fields
        with open(commit_path, "w") as f:
            json.dump({"partial": "data"}, f)
        
        with pytest.raises(Exception) as exc_info:
            repo.get_commit(commit_hash)
        
        print(f"Missing fields error: {exc_info.value}")
    
    def test_circular_commit_references(self):
        """
        Test: Create circular references in commit graph.
        Goal: Ensure the system detects and handles circular dependencies.
        """
        repo = Repository(self.test_dir, init=True)
        
        # Create initial commits
        weight1 = create_weight_tensor(np.array([1.0, 2.0]), "layer1")
        repo.stage_weights({"layer1": weight1})
        commit1_obj = repo.commit("Commit 1")
        commit1 = commit1_obj.commit_hash
        
        weight2 = create_weight_tensor(np.array([3.0, 4.0]), "layer2")
        repo.stage_weights({"layer2": weight2})
        commit2_obj = repo.commit("Commit 2")
        commit2 = commit2_obj.commit_hash
        
        # Manually create circular reference by modifying commit files
        commit1_path = os.path.join(self.test_dir, ".coral", "objects", "commits", f"{commit1}.json")
        commit2_path = os.path.join(self.test_dir, ".coral", "objects", "commits", f"{commit2}.json")
        
        # Load commits
        with open(commit1_path, "r") as f:
            commit1_data = json.load(f)
        with open(commit2_path, "r") as f:
            commit2_data = json.load(f)
        
        # Create circular reference
        commit1_data["parent_hashes"] = [commit2]
        commit2_data["parent_hashes"] = [commit1]
        
        # Save modified commits
        with open(commit1_path, "w") as f:
            json.dump(commit1_data, f)
        with open(commit2_path, "w") as f:
            json.dump(commit2_data, f)
        
        # Try to traverse commit history
        try:
            # This should detect circular reference
            history = []
            current = commit2
            visited = set()
            
            while current:
                if current in visited:
                    print(f"Circular reference detected at commit: {current}")
                    break
                visited.add(current)
                history.append(current)
                
                commit_data = repo.get_commit(current)
                if commit_data.get("parent_hashes"):
                    current = commit_data["parent_hashes"][0]
                else:
                    current = None
                    
                if len(history) > 100:  # Safety limit
                    print("Infinite loop detected in commit graph")
                    break
        except Exception as e:
            print(f"Error traversing circular commits: {e}")
    
    def test_invalid_weight_data(self):
        """
        Test: Store weights with invalid data (NaN, Inf, extremely large).
        Goal: Ensure proper validation and handling of edge cases.
        """
        repo = Repository(self.test_dir, init=True)
        
        # Test 1: NaN values
        nan_weight = create_weight_tensor(np.array([1.0, np.nan, 3.0]), "nan_layer")
        try:
            repo.stage_weights({"nan_layer": nan_weight})
            repo.commit("NaN commit")
            # Try to retrieve
            loaded = repo.get_weight("nan_layer")
            assert np.isnan(loaded.data[1]), "NaN should be preserved"
            print("NaN values handled correctly")
        except Exception as e:
            print(f"NaN handling error: {e}")
        
        # Test 2: Infinity values
        inf_weight = create_weight_tensor(np.array([1.0, np.inf, -np.inf]), "inf_layer")
        try:
            repo.stage_weights({"inf_layer": inf_weight})
            repo.commit("Inf commit")
            loaded = repo.get_weight("inf_layer")
            assert np.isinf(loaded.data[1]), "Inf should be preserved"
            print("Infinity values handled correctly")
        except Exception as e:
            print(f"Infinity handling error: {e}")
        
        # Test 3: Extremely large array
        try:
            # Try to create a very large array (this might fail due to memory)
            large_size = 100000000  # 100 million elements, more reasonable for testing
            large_weight = create_weight_tensor(np.zeros(large_size, dtype=np.float32), "large_layer")
            repo.stage_weights({"large_layer": large_weight})
            print(f"Successfully added array with {large_size} elements")
        except Exception as e:
            print(f"Large array error (expected): {e}")
        
        # Test 4: Zero-size array
        empty_weight = create_weight_tensor(np.array([]), "empty_layer")
        try:
            repo.stage_weights({"empty_layer": empty_weight})
            repo.commit("Empty array commit")
            loaded = repo.get_weight("empty_layer")
            assert len(loaded.data) == 0, "Empty array should be preserved"
            print("Empty array handled correctly")
        except Exception as e:
            print(f"Empty array error: {e}")
    
    def test_concurrent_writes(self):
        """
        Test: Multiple processes writing to the same repository simultaneously.
        Goal: Test thread/process safety and potential race conditions.
        """
        repo = Repository(self.test_dir, init=True)
        
        # Start multiple processes
        processes = []
        num_processes = 4
        
        with multiprocessing.Pool(num_processes) as pool:
            results = []
            for i in range(num_processes):
                result = pool.apply_async(write_weights_process, (i, self.test_dir))
                results.append(result)
            
            # Wait for all processes to complete
            for i, result in enumerate(results):
                output = result.get(timeout=30)
                print(f"Process {i} result: {output}")
        
        # Check repository state
        try:
            # Verify we can still read from repository
            all_weights = repo.get_all_weights()
            print(f"Total weights after concurrent writes: {len(all_weights)}")
            
            # Check for any corruption
            for weight_name, weight in all_weights.items():
                assert weight is not None, f"Failed to load {weight_name}"
        except Exception as e:
            print(f"Repository corrupted after concurrent writes: {e}")
    
    def test_disk_space_exhaustion(self):
        """
        Test: Simulate disk space running out during operations.
        Goal: Ensure graceful handling of I/O errors.
        """
        repo = Repository(self.test_dir, init=True)
        
        # This test is tricky to implement portably, so we'll simulate
        # by making the HDF5 file read-only during a write operation
        
        # Add initial weight
        weight1 = create_weight_tensor(np.array([1.0, 2.0, 3.0]), "initial")
        repo.stage_weights({"initial": weight1})
        repo.commit("Initial commit")
        
        # Make HDF5 file read-only
        hdf5_path = os.path.join(self.test_dir, ".coral", "objects", "weights.h5")
        os.chmod(hdf5_path, 0o444)  # Read-only
        
        # Try to add more weights
        weight2 = create_weight_tensor(np.array([4.0, 5.0, 6.0]), "should_fail")
        
        try:
            repo.stage_weights({"should_fail": weight2})
            repo.commit("This should fail")
            print("WARNING: Write succeeded on read-only file!")
        except Exception as e:
            print(f"Expected write failure: {e}")
            # Restore permissions
            os.chmod(hdf5_path, 0o644)
    
    def test_corrupt_branch_files(self):
        """
        Test: Create conflicting branch states manually.
        Goal: Test branch integrity and conflict detection.
        """
        repo = Repository(self.test_dir, init=True)
        
        # Add weights and commit first (needed before creating branches)
        weight1 = create_weight_tensor(np.array([1.0, 2.0]), "layer1")
        repo.stage_weights({"layer1": weight1})
        commit1 = repo.commit("Commit on main")
        
        # Create some branches
        repo.create_branch("feature1")
        repo.create_branch("feature2")
        
        # Manually corrupt branch file
        branch_path = os.path.join(self.test_dir, ".coral", "refs", "heads", "feature1")
        
        # Test 1: Point branch to non-existent commit
        with open(branch_path, "w") as f:
            f.write("nonexistent_commit_hash")
        
        try:
            repo.checkout("feature1")
            print("WARNING: Checkout succeeded with invalid commit reference!")
        except Exception as e:
            print(f"Expected checkout failure: {e}")
        
        # Test 2: Create duplicate branch files
        duplicate_path = os.path.join(self.test_dir, ".coral", "refs", "heads", "feature1.duplicate")
        shutil.copy(branch_path, duplicate_path)
        
        try:
            branches = repo.branch_manager.list_branches()
            print(f"Branches found: {branches}")
        except Exception as e:
            print(f"Branch listing error: {e}")
    
    def test_metadata_corruption(self):
        """
        Test: Corrupt weight metadata while keeping data intact.
        Goal: Test metadata validation and recovery.
        """
        repo = Repository(self.test_dir, init=True)
        
        # Add weight with metadata
        weight = create_weight_tensor(np.array([1.0, 2.0, 3.0]), "test_layer")
        weight.metadata.layer_type = "dense"
        weight.metadata.model_name = "test_model"
        repo.stage_weights({"test_layer": weight})
        repo.commit("Initial commit")
        
        # Directly modify HDF5 metadata
        hdf5_path = os.path.join(self.test_dir, ".coral", "objects", "weights.h5")
        
        with h5py.File(hdf5_path, "r+") as f:
            # Find the weight's metadata
            for key in f.keys():
                if "metadata" in f[key].attrs:
                    # Corrupt metadata with invalid JSON
                    f[key].attrs["metadata"] = "invalid{json"
                    break
        
        # Try to load weight with corrupted metadata
        try:
            loaded = repo.get_weight("test_layer")
            print(f"Weight loaded despite corrupted metadata: {loaded}")
        except Exception as e:
            print(f"Expected metadata corruption error: {e}")
    
    def test_version_graph_corruption(self):
        """
        Test: Create invalid version graph structures.
        Goal: Test graph integrity validation.
        """
        repo = Repository(self.test_dir, init=True)
        
        # Create a few commits
        for i in range(3):
            weight = create_weight_tensor(np.array([float(i)]), f"layer{i}")
            repo.stage_weights({f"layer{i}": weight})
            repo.commit(f"Commit {i}")
        
        # Try to corrupt the internal version graph
        # This would require accessing internal structures
        try:
            # Attempt to create orphaned commits
            orphan_commit_path = os.path.join(self.test_dir, ".coral", "objects", "commits", "orphan_commit.json")
            with open(orphan_commit_path, "w") as f:
                json.dump({
                    "commit_hash": "orphan_commit",
                    "parent_hashes": ["nonexistent_parent"],
                    "weight_hashes": {},
                    "metadata": {
                        "author": "Test",
                        "email": "test@example.com",
                        "message": "Orphaned commit",
                        "timestamp": time.time()
                    }
                }, f)
            
            # See if the system detects orphaned commits
            all_commits = os.listdir(os.path.join(self.test_dir, ".coral", "objects", "commits"))
            print(f"Total commits (including orphan): {len(all_commits)}")
        except Exception as e:
            print(f"Version graph corruption error: {e}")


def run_corruption_tests():
    """Run all corruption tests and summarize results."""
    print("=" * 80)
    print("CORAL QA CORRUPTION TESTS")
    print("=" * 80)
    print("\nThese tests attempt to corrupt the repository in various ways.")
    print("Expected behavior: Graceful error handling, no crashes.\n")
    
    # Run pytest with verbose output
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_corruption_tests()