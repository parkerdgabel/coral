import numpy as np

from coral.core.deduplicator import DeduplicationStats
from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.utils.visualization import plot_deduplication_stats, plot_weight_distribution


class TestVisualization:
    def test_plot_weight_distribution(self):
        """Test weight distribution analysis."""
        # Create test weights
        weights = [
            WeightTensor(
                data=np.random.randn(10, 10).astype(np.float32),
                metadata=WeightMetadata(name="layer1.weight", shape=(10, 10), dtype=np.float32),
            ),
            WeightTensor(
                data=np.random.randn(5, 5).astype(np.float32),
                metadata=WeightMetadata(name="layer2.weight", shape=(5, 5), dtype=np.float32),
            ),
            WeightTensor(
                data=np.zeros((3, 3), dtype=np.float32),  # All zeros for sparsity test
                metadata=WeightMetadata(name="layer3.weight", shape=(3, 3), dtype=np.float32),
            ),
        ]

        # Get distribution data
        distributions = plot_weight_distribution(weights, bins=30)

        # Check structure
        assert len(distributions) == 3
        assert "layer1.weight" in distributions
        assert "layer2.weight" in distributions
        assert "layer3.weight" in distributions

        # Check layer1 stats
        layer1_stats = distributions["layer1.weight"]
        assert "histogram" in layer1_stats
        assert "bin_edges" in layer1_stats
        assert "mean" in layer1_stats
        assert "std" in layer1_stats
        assert "min" in layer1_stats
        assert "max" in layer1_stats
        assert "sparsity" in layer1_stats

        # Check histogram properties
        assert len(layer1_stats["histogram"]) == 30
        assert len(layer1_stats["bin_edges"]) == 31  # bins + 1

        # Check layer3 sparsity (all zeros)
        layer3_stats = distributions["layer3.weight"]
        assert layer3_stats["sparsity"] == 1.0
        assert layer3_stats["mean"] == 0.0
        assert layer3_stats["std"] == 0.0

    def test_plot_weight_distribution_empty(self):
        """Test weight distribution with empty list."""
        distributions = plot_weight_distribution([], bins=10)
        assert distributions == {}

    def test_plot_weight_distribution_single_value(self):
        """Test weight distribution with uniform weights."""
        weights = [
            WeightTensor(
                data=np.ones((5, 5), dtype=np.float32) * 2.5,
                metadata=WeightMetadata(name="uniform.weight", shape=(5, 5), dtype=np.float32),
            )
        ]

        distributions = plot_weight_distribution(weights, bins=10)

        uniform_stats = distributions["uniform.weight"]
        assert uniform_stats["mean"] == 2.5
        assert uniform_stats["std"] == 0.0
        assert uniform_stats["min"] == 2.5
        assert uniform_stats["max"] == 2.5
        assert uniform_stats["sparsity"] == 0.0

    def test_plot_deduplication_stats(self):
        """Test deduplication statistics visualization."""
        # Create mock stats
        stats = DeduplicationStats(
            total_weights=100,
            unique_weights=60,
            duplicate_weights=25,
            similar_weights=15,
            bytes_saved=1024 * 1024 * 10,  # 10MB
            compression_ratio=1.5,
        )

        # Get visualization data
        viz_data = plot_deduplication_stats(stats)

        # Check structure
        assert "weight_counts" in viz_data
        assert "compression" in viz_data
        assert "pie_data" in viz_data

        # Check weight counts
        counts = viz_data["weight_counts"]
        assert counts["unique"] == 60
        assert counts["duplicate"] == 25
        assert counts["similar"] == 15
        assert counts["total"] == 100

        # Check compression stats
        compression = viz_data["compression"]
        assert compression["bytes_saved"] == 1024 * 1024 * 10
        assert compression["compression_ratio"] == 1.5

        # Check pie data
        pie_data = viz_data["pie_data"]
        assert len(pie_data) == 3
        assert pie_data[0]["label"] == "Unique"
        assert pie_data[0]["value"] == 60
        assert pie_data[1]["label"] == "Duplicate"
        assert pie_data[1]["value"] == 25
        assert pie_data[2]["label"] == "Similar"
        assert pie_data[2]["value"] == 15

    def test_plot_deduplication_stats_no_duplicates(self):
        """Test deduplication stats with no duplicates."""
        stats = DeduplicationStats(
            total_weights=50,
            unique_weights=50,
            duplicate_weights=0,
            similar_weights=0,
            bytes_saved=0,
            compression_ratio=1.0,
        )

        viz_data = plot_deduplication_stats(stats)

        assert viz_data["weight_counts"]["unique"] == 50
        assert viz_data["weight_counts"]["duplicate"] == 0
        assert viz_data["weight_counts"]["similar"] == 0
        assert viz_data["compression"]["bytes_saved"] == 0
        assert viz_data["compression"]["compression_ratio"] == 1.0
