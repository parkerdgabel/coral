"""Integration tests for consistent delta reconstruction behavior."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.version_control.repository import Repository


@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing."""
    temp_dir = tempfile.mkdtemp()
    repo_path = Path(temp_dir)
    repo = Repository(repo_path, init=True)
    yield repo
    shutil.rmtree(temp_dir)


@pytest.fixture
def delta_test_weights():
    """Create weights for delta reconstruction testing."""
    np.random.seed(12345)  # Fixed seed for reproducibility
    base_data = np.random.randn(50, 30).astype(np.float32)

    # Create reference weight
    reference = WeightTensor(
        data=base_data,
        metadata=WeightMetadata(name="ref_weight", shape=(50, 30), dtype=np.float32),
    )

    # Create similar weight (high similarity)
    noise = np.random.randn(*base_data.shape).astype(np.float32) * 0.01
    similar_data = base_data + noise
    similar = WeightTensor(
        data=similar_data,
        metadata=WeightMetadata(name="sim_weight", shape=(50, 30), dtype=np.float32),
    )

    # Create exact duplicate
    duplicate = WeightTensor(
        data=base_data.copy(),
        metadata=WeightMetadata(name="dup_weight", shape=(50, 30), dtype=np.float32),
    )

    # Create unique weight
    unique_data = np.random.randn(50, 30).astype(np.float32) * 2.0
    unique = WeightTensor(
        data=unique_data,
        metadata=WeightMetadata(name="unique_weight", shape=(50, 30), dtype=np.float32),
    )

    return {
        "reference": reference,
        "similar": similar,
        "duplicate": duplicate,
        "unique": unique,
    }


class TestDeltaReconstructionConsistency:
    """Test consistency between different delta reconstruction paths."""

    def test_get_weight_vs_get_all_weights_consistency(
        self, temp_repo, delta_test_weights
    ):
        """Test that get_weight and get_all_weights return identical results."""
        # Stage and commit test weights
        temp_repo.stage_weights(delta_test_weights)
        temp_repo.commit("Test delta consistency")

        # Verify delta encoding occurred
        assert temp_repo.deduplicator.is_delta_encoded("similar")
        assert not temp_repo.deduplicator.is_delta_encoded("reference")
        assert not temp_repo.deduplicator.is_delta_encoded("duplicate")
        assert not temp_repo.deduplicator.is_delta_encoded("unique")

        # Get weights individually
        individual_weights = {}
        for name in delta_test_weights.keys():
            individual_weights[name] = temp_repo.get_weight(name)

        # Get all weights at once
        all_weights = temp_repo.get_all_weights()

        # Verify both methods return the same weights
        assert set(individual_weights.keys()) == set(all_weights.keys())

        for name in delta_test_weights.keys():
            individual_weight = individual_weights[name]
            all_weight = all_weights[name]

            # Both should exist
            assert individual_weight is not None, f"Individual weight {name} is None"
            assert all_weight is not None, f"All weights {name} is None"

            # Data should be identical
            np.testing.assert_array_equal(
                individual_weight.data,
                all_weight.data,
                err_msg=f"Data mismatch for weight {name}",
            )

            # Metadata should match
            assert individual_weight.shape == all_weight.shape
            assert individual_weight.dtype == all_weight.dtype
            assert individual_weight.metadata.name == all_weight.metadata.name

    def test_delta_reconstruction_accuracy(self, temp_repo, delta_test_weights):
        """Test that delta reconstruction preserves original data accurately."""
        original_weights = delta_test_weights.copy()

        # Stage and commit
        temp_repo.stage_weights(delta_test_weights)
        temp_repo.commit("Test reconstruction accuracy")

        # Retrieve and verify accuracy
        for name, original_weight in original_weights.items():
            retrieved_weight = temp_repo.get_weight(name)
            assert retrieved_weight is not None

            if temp_repo.deduplicator.is_delta_encoded(name):
                # For delta-encoded weights, verify high accuracy
                np.testing.assert_array_almost_equal(
                    retrieved_weight.data,
                    original_weight.data,
                    decimal=5,  # Very high precision expected
                    err_msg=f"Delta reconstruction inaccurate for {name}",
                )
            else:
                # For non-delta weights, should be exact
                np.testing.assert_array_equal(
                    retrieved_weight.data,
                    original_weight.data,
                    err_msg=f"Non-delta weight data mismatch for {name}",
                )

    def test_storage_vs_memory_consistency(self, temp_repo, delta_test_weights):
        """Test consistency between storage and in-memory delta reconstruction."""
        # Stage and commit weights
        temp_repo.stage_weights(delta_test_weights)
        temp_repo.commit("Test storage vs memory")

        # Get weights from storage (via repository)
        storage_weights = {}
        for name in delta_test_weights.keys():
            storage_weights[name] = temp_repo.get_weight(name)

        # Get weights from in-memory deduplicator
        memory_weights = {}
        for name in delta_test_weights.keys():
            memory_weights[name] = temp_repo.deduplicator.get_weight_by_name(name)

        # Compare results
        for name in delta_test_weights.keys():
            storage_weight = storage_weights[name]
            memory_weight = memory_weights[name]

            assert storage_weight is not None, f"Storage weight {name} is None"
            assert memory_weight is not None, f"Memory weight {name} is None"

            # Data should be very close (may have small reconstruction differences)
            np.testing.assert_array_almost_equal(
                storage_weight.data,
                memory_weight.data,
                decimal=5,  # Allow for minor reconstruction differences
                err_msg=f"Storage vs memory mismatch for {name}",
            )

    def test_commit_specific_retrieval(self, temp_repo, delta_test_weights):
        """Test delta reconstruction works for specific commit references."""
        # Create first commit
        temp_repo.stage_weights(delta_test_weights)
        commit1 = temp_repo.commit("First commit")

        # Modify weights and create second commit
        modified_weights = {}
        for name, weight in delta_test_weights.items():
            if name == "reference":
                # Create new data array with modification
                new_data = weight.data.copy() + 0.1
                modified_weights[name] = WeightTensor(
                    data=new_data, metadata=weight.metadata
                )
            else:
                modified_weights[name] = weight
        temp_repo.stage_weights(modified_weights)
        commit2 = temp_repo.commit("Second commit")

        # Verify retrieval from specific commits
        for name in delta_test_weights.keys():
            # Get from first commit
            weight1 = temp_repo.get_weight(name, commit1.commit_hash)

            # Get from second commit
            weight2 = temp_repo.get_weight(name, commit2.commit_hash)

            assert weight1 is not None, f"Weight {name} not found in commit1"
            assert weight2 is not None, f"Weight {name} not found in commit2"

            # Reference weight should be different between commits
            if name == "reference":
                assert not np.array_equal(weight1.data, weight2.data), (
                    "Reference weight should differ between commits"
                )
            # Other weights should be the same
            else:
                np.testing.assert_array_almost_equal(
                    weight1.data,
                    weight2.data,
                    decimal=5,
                    err_msg=f"Weight {name} should be same between commits",
                )

    def test_get_all_weights_commit_specific(self, temp_repo, delta_test_weights):
        """Test get_all_weights works correctly for specific commits."""
        # Create commit
        temp_repo.stage_weights(delta_test_weights)
        commit = temp_repo.commit("Test commit-specific get_all_weights")

        # Get all weights from specific commit
        all_weights_by_commit = temp_repo.get_all_weights(commit.commit_hash)

        # Get all weights from current HEAD
        all_weights_current = temp_repo.get_all_weights()

        # Should be identical
        assert set(all_weights_by_commit.keys()) == set(all_weights_current.keys())

        for name in delta_test_weights.keys():
            np.testing.assert_array_equal(
                all_weights_by_commit[name].data,
                all_weights_current[name].data,
                err_msg=f"Commit-specific vs current mismatch for {name}",
            )
