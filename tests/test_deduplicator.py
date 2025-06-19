"""Tests for Deduplicator class"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from coral.core.deduplicator import Deduplicator
from coral.core.weight_tensor import WeightMetadata, WeightTensor


class TestDeduplicator:
    def test_exact_deduplication(self):
        """Test detection of exact duplicates"""
        dedup = Deduplicator(similarity_threshold=0.99)

        # Create identical weights
        data = np.random.randn(10, 10).astype(np.float32)
        weight1 = WeightTensor(
            data=data,
            metadata=WeightMetadata(name="weight1", shape=(10, 10), dtype=np.float32),
        )
        weight2 = WeightTensor(
            data=data.copy(),
            metadata=WeightMetadata(name="weight2", shape=(10, 10), dtype=np.float32),
        )

        # Add weights
        hash1 = dedup.add_weight(weight1)
        hash2 = dedup.add_weight(weight2)

        # Should return same hash for duplicates
        assert hash1 == hash2

        # Check statistics
        stats = dedup.compute_stats()
        assert stats.total_weights == 2
        assert stats.unique_weights == 1
        assert stats.duplicate_weights == 1
        assert stats.similar_weights == 0

    def test_similar_deduplication(self):
        """Test detection of similar weights"""
        dedup = Deduplicator(similarity_threshold=0.98)

        # Create similar weights
        data1 = np.random.randn(10, 10).astype(np.float32)
        data2 = data1 + np.random.randn(10, 10).astype(np.float32) * 0.01

        weight1 = WeightTensor(
            data=data1,
            metadata=WeightMetadata(name="weight1", shape=(10, 10), dtype=np.float32),
        )
        weight2 = WeightTensor(
            data=data2,
            metadata=WeightMetadata(name="weight2", shape=(10, 10), dtype=np.float32),
        )

        # Add weights
        hash1 = dedup.add_weight(weight1)
        hash2 = dedup.add_weight(weight2)

        # Should detect similarity
        assert hash1 == hash2  # Second weight should reference first

        # Check statistics
        stats = dedup.compute_stats()
        assert stats.similar_weights == 1

    def test_unique_weights(self):
        """Test handling of unique weights"""
        dedup = Deduplicator(similarity_threshold=0.99)

        # Create different weights
        weights = []
        for i in range(5):
            data = np.random.randn(10 + i, 10).astype(np.float32)
            weight = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"weight{i}", shape=data.shape, dtype=np.float32
                ),
            )
            weights.append(weight)

        # Add all weights
        hashes = []
        for weight in weights:
            hash_val = dedup.add_weight(weight)
            hashes.append(hash_val)

        # All should be unique
        assert len(set(hashes)) == 5

        stats = dedup.compute_stats()
        assert stats.total_weights == 5
        assert stats.unique_weights == 5
        assert stats.duplicate_weights == 0

    def test_weight_groups(self):
        """Test weight group management"""
        dedup = Deduplicator(similarity_threshold=0.98)

        # Create a reference weight
        ref_data = np.random.randn(10, 10).astype(np.float32)
        ref_weight = WeightTensor(
            data=ref_data,
            metadata=WeightMetadata(name="reference", shape=(10, 10), dtype=np.float32),
        )

        # Create duplicates and similar weights
        dup_weight = WeightTensor(
            data=ref_data.copy(),
            metadata=WeightMetadata(name="duplicate", shape=(10, 10), dtype=np.float32),
        )

        similar_data = ref_data + np.random.randn(10, 10).astype(np.float32) * 0.01
        similar_weight = WeightTensor(
            data=similar_data,
            metadata=WeightMetadata(name="similar", shape=(10, 10), dtype=np.float32),
        )

        # Add weights
        dedup.add_weight(ref_weight)
        dedup.add_weight(dup_weight)
        dedup.add_weight(similar_weight)

        # Get weight group
        group = dedup.get_weight_group("reference")
        assert group is not None
        assert group.total_count == 3  # ref + duplicate + similar
        assert len(group.duplicates) == 1
        assert len(group.similar) == 1

    def test_deduplication_report(self):
        """Test comprehensive deduplication report"""
        dedup = Deduplicator(similarity_threshold=0.98)

        # Add various weights
        base_data = np.random.randn(100, 100).astype(np.float32)

        # Add original
        dedup.add_weight(
            WeightTensor(
                data=base_data,
                metadata=WeightMetadata(
                    name="original", shape=(100, 100), dtype=np.float32
                ),
            )
        )

        # Add duplicates
        for i in range(3):
            dedup.add_weight(
                WeightTensor(
                    data=base_data.copy(),
                    metadata=WeightMetadata(
                        name=f"dup{i}", shape=(100, 100), dtype=np.float32
                    ),
                )
            )

        # Add similar
        for i in range(2):
            similar = base_data + np.random.randn(100, 100).astype(np.float32) * 0.005
            dedup.add_weight(
                WeightTensor(
                    data=similar,
                    metadata=WeightMetadata(
                        name=f"similar{i}", shape=(100, 100), dtype=np.float32
                    ),
                )
            )

        # Get report
        report = dedup.get_deduplication_report()

        assert report["summary"]["total_weights"] == 6
        assert report["summary"]["unique_weights"] == 1
        assert report["summary"]["duplicate_weights"] == 3
        assert report["summary"]["similar_weights"] == 2
        assert report["summary"]["bytes_saved"] > 0
        assert len(report["largest_groups"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
