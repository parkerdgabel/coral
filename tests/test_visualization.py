import numpy as np

from coral.core.deduplicator import DeduplicationStats
from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.utils.visualization import plot_deduplication_stats, plot_weight_distribution


class TestVisualization:
    def test_plot_weight_distribution(self):
        """Test weight distribution analysis."""
        # Create test weights
        weights = [
            WeightTensor(
                data=np.random.randn(10, 10).astype(np.float32),
                metadata=WeightMetadata(
                    name="layer1.weight", shape=(10, 10), dtype=np.float32
                ),
            ),
            WeightTensor(
                data=np.random.randn(5, 5).astype(np.float32),
                metadata=WeightMetadata(
                    name="layer2.weight", shape=(5, 5), dtype=np.float32
                ),
            ),
            WeightTensor(
                data=np.zeros((3, 3), dtype=np.float32),  # All zeros for sparsity test
                metadata=WeightMetadata(
                    name="layer3.weight", shape=(3, 3), dtype=np.float32
                ),
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
                metadata=WeightMetadata(
                    name="uniform.weight", shape=(5, 5), dtype=np.float32
                ),
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


class TestModelComparison:
    """Tests for model comparison visualization functions."""

    def test_compare_models_identical(self):
        """Test comparing identical models."""
        from coral.utils.visualization import compare_models

        # Create identical weights
        weights_a = {
            "layer1.weight": WeightTensor(
                data=np.ones((10, 5), dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer1.weight", shape=(10, 5), dtype=np.float32
                ),
            ),
            "layer1.bias": WeightTensor(
                data=np.zeros(10, dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer1.bias", shape=(10,), dtype=np.float32
                ),
            ),
        }
        weights_b = {
            "layer1.weight": WeightTensor(
                data=np.ones((10, 5), dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer1.weight", shape=(10, 5), dtype=np.float32
                ),
            ),
            "layer1.bias": WeightTensor(
                data=np.zeros(10, dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer1.bias", shape=(10,), dtype=np.float32
                ),
            ),
        }

        diff = compare_models(weights_a, weights_b, "Model A", "Model B")

        assert "summary" in diff
        assert "layer_comparisons" in diff
        assert diff["summary"]["overall_similarity"] == 1.0
        assert diff["summary"]["common_layers"] == 2
        assert len(diff["summary"]["only_in_a"]) == 0
        assert len(diff["summary"]["only_in_b"]) == 0

    def test_compare_models_different(self):
        """Test comparing different models."""
        from coral.utils.visualization import compare_models

        np.random.seed(42)

        weights_a = {
            "layer1.weight": WeightTensor(
                data=np.random.randn(10, 5).astype(np.float32),
                metadata=WeightMetadata(
                    name="layer1.weight", shape=(10, 5), dtype=np.float32
                ),
            ),
        }
        weights_b = {
            "layer1.weight": WeightTensor(
                data=np.random.randn(10, 5).astype(np.float32),
                metadata=WeightMetadata(
                    name="layer1.weight", shape=(10, 5), dtype=np.float32
                ),
            ),
        }

        diff = compare_models(weights_a, weights_b, "Model A", "Model B")

        assert diff["summary"]["overall_similarity"] < 1.0
        assert diff["summary"]["common_layers"] == 1

        # Check layer comparison has similarity metrics
        layer_comp = diff["layer_comparisons"][0]
        assert "cosine_similarity" in layer_comp
        assert "magnitude_similarity" in layer_comp
        assert "combined_similarity" in layer_comp
        assert "diff_stats" in layer_comp

    def test_compare_models_missing_layers(self):
        """Test comparing models with different layers."""
        from coral.utils.visualization import compare_models

        weights_a = {
            "layer1.weight": WeightTensor(
                data=np.ones((10, 5), dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer1.weight", shape=(10, 5), dtype=np.float32
                ),
            ),
            "layer2.weight": WeightTensor(
                data=np.ones((5, 3), dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer2.weight", shape=(5, 3), dtype=np.float32
                ),
            ),
        }
        weights_b = {
            "layer1.weight": WeightTensor(
                data=np.ones((10, 5), dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer1.weight", shape=(10, 5), dtype=np.float32
                ),
            ),
            "layer3.weight": WeightTensor(
                data=np.ones((3, 3), dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer3.weight", shape=(3, 3), dtype=np.float32
                ),
            ),
        }

        diff = compare_models(weights_a, weights_b, "Model A", "Model B")

        assert diff["summary"]["common_layers"] == 1
        assert "layer2.weight" in diff["summary"]["only_in_a"]
        assert "layer3.weight" in diff["summary"]["only_in_b"]

    def test_compare_models_shape_mismatch(self):
        """Test comparing models with shape mismatches."""
        from coral.utils.visualization import compare_models

        weights_a = {
            "layer1.weight": WeightTensor(
                data=np.ones((10, 5), dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer1.weight", shape=(10, 5), dtype=np.float32
                ),
            ),
        }
        weights_b = {
            "layer1.weight": WeightTensor(
                data=np.ones((5, 10), dtype=np.float32),  # Different shape
                metadata=WeightMetadata(
                    name="layer1.weight", shape=(5, 10), dtype=np.float32
                ),
            ),
        }

        diff = compare_models(weights_a, weights_b, "Model A", "Model B")

        layer_comp = diff["layer_comparisons"][0]
        assert layer_comp["shape_mismatch"] is True
        assert layer_comp["combined_similarity"] == 0.0

    def test_compare_models_empty(self):
        """Test comparing empty models."""
        from coral.utils.visualization import compare_models

        diff = compare_models({}, {}, "Empty A", "Empty B")

        assert diff["summary"]["common_layers"] == 0
        assert diff["summary"]["total_layers_a"] == 0
        assert diff["summary"]["total_layers_b"] == 0
        assert diff["summary"]["overall_similarity"] == 0.0

    def test_compare_models_categories(self):
        """Test that models are categorized correctly."""
        from coral.utils.visualization import compare_models

        # Create weights with known similarity
        base = np.random.randn(10, 10).astype(np.float32)

        weights_a = {
            "identical": WeightTensor(
                data=base.copy(),
                metadata=WeightMetadata(
                    name="identical", shape=(10, 10), dtype=np.float32
                ),
            ),
            "minor": WeightTensor(
                data=base + np.random.randn(10, 10).astype(np.float32) * 0.001,
                metadata=WeightMetadata(name="minor", shape=(10, 10), dtype=np.float32),
            ),
        }
        weights_b = {
            "identical": WeightTensor(
                data=base.copy(),
                metadata=WeightMetadata(
                    name="identical", shape=(10, 10), dtype=np.float32
                ),
            ),
            "minor": WeightTensor(
                data=base,  # Original
                metadata=WeightMetadata(name="minor", shape=(10, 10), dtype=np.float32),
            ),
        }

        diff = compare_models(weights_a, weights_b, "Model A", "Model B")

        assert "categories" in diff
        assert "category_counts" in diff
        # identical layer should be in identical category
        assert "identical" in diff["categories"]["identical"]


class TestFormatModelDiff:
    """Tests for format_model_diff function."""

    def test_format_model_diff_basic(self):
        """Test basic model diff formatting."""
        from coral.utils.visualization import compare_models, format_model_diff

        weights_a = {
            "layer1.weight": WeightTensor(
                data=np.ones((10, 5), dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer1.weight", shape=(10, 5), dtype=np.float32
                ),
            ),
        }
        weights_b = {
            "layer1.weight": WeightTensor(
                data=np.ones((10, 5), dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer1.weight", shape=(10, 5), dtype=np.float32
                ),
            ),
        }

        diff = compare_models(weights_a, weights_b, "Model A", "Model B")
        output = format_model_diff(diff)

        assert "Model Comparison: Model A vs Model B" in output
        assert "Summary:" in output
        assert "Overall Similarity:" in output
        assert "Common Layers:" in output
        assert "Change Categories:" in output

    def test_format_model_diff_verbose(self):
        """Test verbose model diff formatting."""
        from coral.utils.visualization import compare_models, format_model_diff

        weights_a = {
            "layer1.weight": WeightTensor(
                data=np.random.randn(10, 5).astype(np.float32),
                metadata=WeightMetadata(
                    name="layer1.weight", shape=(10, 5), dtype=np.float32
                ),
            ),
        }
        weights_b = {
            "layer1.weight": WeightTensor(
                data=np.random.randn(10, 5).astype(np.float32),
                metadata=WeightMetadata(
                    name="layer1.weight", shape=(10, 5), dtype=np.float32
                ),
            ),
        }

        diff = compare_models(weights_a, weights_b, "Model A", "Model B")
        output = format_model_diff(diff, verbose=True)

        assert "Per-Layer Details:" in output
        assert "layer1.weight:" in output
        assert "Cosine Similarity:" in output
        assert "Magnitude Similarity:" in output

    def test_format_model_diff_with_missing_layers(self):
        """Test formatting when layers are missing."""
        from coral.utils.visualization import compare_models, format_model_diff

        weights_a = {
            "layer1.weight": WeightTensor(
                data=np.ones((10, 5), dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer1.weight", shape=(10, 5), dtype=np.float32
                ),
            ),
            "layer2.weight": WeightTensor(
                data=np.ones((5, 3), dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer2.weight", shape=(5, 3), dtype=np.float32
                ),
            ),
        }
        weights_b = {
            "layer1.weight": WeightTensor(
                data=np.ones((10, 5), dtype=np.float32),
                metadata=WeightMetadata(
                    name="layer1.weight", shape=(10, 5), dtype=np.float32
                ),
            ),
        }

        diff = compare_models(weights_a, weights_b, "Model A", "Model B")
        output = format_model_diff(diff)

        assert "Layers only in Model A:" in output
        assert "layer2.weight" in output

    def test_format_model_diff_most_changed(self):
        """Test that most changed layers are shown."""
        from coral.utils.visualization import compare_models, format_model_diff

        np.random.seed(42)
        weights_a = {}
        weights_b = {}

        for i in range(10):
            weights_a[f"layer{i}.weight"] = WeightTensor(
                data=np.random.randn(10, 10).astype(np.float32),
                metadata=WeightMetadata(
                    name=f"layer{i}.weight", shape=(10, 10), dtype=np.float32
                ),
            )
            weights_b[f"layer{i}.weight"] = WeightTensor(
                data=np.random.randn(10, 10).astype(np.float32),
                metadata=WeightMetadata(
                    name=f"layer{i}.weight", shape=(10, 10), dtype=np.float32
                ),
            )

        diff = compare_models(weights_a, weights_b, "Model A", "Model B")
        output = format_model_diff(diff)

        assert "Most Changed Layers:" in output
