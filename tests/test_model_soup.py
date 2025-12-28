"""Tests for model souping functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from coral.core.model_soup import (
    ModelSoup,
    SoupConfig,
    SoupResult,
    SoupStrategy,
    dare_ties_merge,
    greedy_soup,
    ties_merge,
    uniform_soup,
)
from coral.core.weight_tensor import WeightMetadata, WeightTensor


def create_weight(name: str, shape: tuple, seed: int = 0) -> WeightTensor:
    """Helper to create a weight tensor."""
    rng = np.random.RandomState(seed)
    data = rng.randn(*shape).astype(np.float32)
    return WeightTensor(
        data=data,
        metadata=WeightMetadata(name=name, shape=shape, dtype=np.float32),
    )


def create_model(
    names: list[str], shape: tuple, seed: int = 0
) -> dict[str, WeightTensor]:
    """Helper to create a model with multiple weights."""
    model = {}
    for i, name in enumerate(names):
        model[name] = create_weight(name, shape, seed=seed + i)
    return model


class TestUniformSoup:
    """Tests for uniform model soup (simple averaging)."""

    def test_uniform_soup_two_models(self):
        """Test uniform averaging of two models."""
        model1 = create_model(["layer1", "layer2"], (10, 5), seed=1)
        model2 = create_model(["layer1", "layer2"], (10, 5), seed=2)

        result = uniform_soup([model1, model2])

        assert "layer1" in result
        assert "layer2" in result

        # Check that result is average
        expected_layer1 = (model1["layer1"].data + model2["layer1"].data) / 2
        np.testing.assert_array_almost_equal(result["layer1"].data, expected_layer1)

    def test_uniform_soup_three_models(self):
        """Test uniform averaging of three models."""
        models = [create_model(["w"], (8, 4), seed=i) for i in range(3)]

        result = uniform_soup(models)

        expected = np.mean([m["w"].data for m in models], axis=0)
        np.testing.assert_array_almost_equal(result["w"].data, expected)

    def test_uniform_soup_single_model(self):
        """Test uniform soup with single model returns that model."""
        model = create_model(["layer"], (5, 5), seed=42)

        soup = ModelSoup(SoupConfig(strategy=SoupStrategy.UNIFORM))
        result = soup.merge([model])

        assert result.num_models == 1
        np.testing.assert_array_equal(result.weights["layer"].data, model["layer"].data)

    def test_uniform_soup_weighted(self):
        """Test uniform soup with custom weights."""
        model1 = create_model(["w"], (4, 4), seed=1)
        model2 = create_model(["w"], (4, 4), seed=2)

        config = SoupConfig(
            strategy=SoupStrategy.UNIFORM,
            model_weights=[0.8, 0.2],  # 80% model1, 20% model2
        )
        soup = ModelSoup(config)
        result = soup.merge([model1, model2])

        expected = 0.8 * model1["w"].data + 0.2 * model2["w"].data
        np.testing.assert_array_almost_equal(result.weights["w"].data, expected)

    def test_uniform_soup_missing_weights(self):
        """Test handling of models with different weight sets."""
        model1 = {"shared": create_weight("shared", (5, 5), seed=1)}
        model2 = {
            "shared": create_weight("shared", (5, 5), seed=2),
            "unique": create_weight("unique", (3, 3), seed=3),
        }

        result = uniform_soup([model1, model2])

        # Shared should be averaged
        assert "shared" in result
        # Unique should be included (from single model)
        assert "unique" in result


class TestGreedySoup:
    """Tests for greedy model soup with validation-based selection."""

    def test_greedy_soup_requires_validation_fn(self):
        """Test that greedy soup raises error without validation function."""
        models = [create_model(["w"], (5, 5), seed=i) for i in range(3)]

        with pytest.raises(ValueError, match="validation_fn"):
            config = SoupConfig(strategy=SoupStrategy.GREEDY)
            soup = ModelSoup(config)
            soup.merge(models)

    def test_greedy_soup_selects_best(self):
        """Test greedy soup selects models that improve validation."""
        # Create models where averaging helps
        models = [create_model(["w"], (10, 10), seed=i) for i in range(4)]

        # Validation function: prefer weights with smaller L2 norm
        def validate(weights):
            return -np.sum(weights["w"] ** 2)

        result = greedy_soup(models, validate)

        assert isinstance(result, SoupResult)
        assert result.strategy == SoupStrategy.GREEDY
        assert len(result.included_indices) >= 1
        assert len(result.validation_scores) >= 1

    def test_greedy_soup_tracks_scores(self):
        """Test that greedy soup tracks validation scores."""
        models = [create_model(["w"], (8, 8), seed=i) for i in range(3)]

        def validate(weights):
            return 1.0  # Constant score - all models "improve"

        result = greedy_soup(models, validate)

        assert len(result.validation_scores) > 0
        assert "final_score" in result.stats


class TestTIESMerging:
    """Tests for TIES-Merging algorithm."""

    def test_ties_requires_base_model(self):
        """Test that TIES requires a base model."""
        models = [create_model(["w"], (5, 5), seed=i) for i in range(2)]

        with pytest.raises(ValueError, match="base model"):
            config = SoupConfig(strategy=SoupStrategy.TIES)
            soup = ModelSoup(config)
            soup.merge(models)  # No base_model provided

    def test_ties_basic_merge(self):
        """Test basic TIES merge operation."""
        base = create_model(["w"], (10, 10), seed=0)
        model1 = create_model(["w"], (10, 10), seed=1)
        model2 = create_model(["w"], (10, 10), seed=2)

        result = ties_merge([model1, model2], base, density=0.5)

        assert "w" in result
        assert result["w"].shape == (10, 10)

    def test_ties_density_parameter(self):
        """Test that TIES density parameter affects trimming."""
        base = create_model(["w"], (20, 20), seed=0)
        model1 = create_model(["w"], (20, 20), seed=1)
        model2 = create_model(["w"], (20, 20), seed=2)

        # High density keeps more params
        result_high = ties_merge([model1, model2], base, density=0.9)
        # Low density keeps fewer params
        result_low = ties_merge([model1, model2], base, density=0.1)

        # Results should be different
        assert not np.allclose(result_high["w"].data, result_low["w"].data)

    def test_ties_sign_election(self):
        """Test TIES sign election mechanism."""
        # Create base and models with specific sign patterns
        base = {
            "w": WeightTensor(
                data=np.zeros((4, 4), dtype=np.float32),
                metadata=WeightMetadata(name="w", shape=(4, 4), dtype=np.float32),
            )
        }

        # Model 1: positive deltas
        m1_data = np.ones((4, 4), dtype=np.float32)
        model1 = {
            "w": WeightTensor(
                data=m1_data,
                metadata=WeightMetadata(name="w", shape=(4, 4), dtype=np.float32),
            )
        }

        # Model 2: mostly positive deltas (majority should win)
        m2_data = np.ones((4, 4), dtype=np.float32) * 0.5
        model2 = {
            "w": WeightTensor(
                data=m2_data,
                metadata=WeightMetadata(name="w", shape=(4, 4), dtype=np.float32),
            )
        }

        result = ties_merge([model1, model2], base, density=1.0)

        # Result should be positive since both models have positive deltas
        assert np.all(result["w"].data >= 0)


class TestDAREMerging:
    """Tests for DARE (Drop and Rescale) merging."""

    def test_dare_ties_basic(self):
        """Test basic DARE + TIES merge."""
        base = create_model(["w"], (10, 10), seed=0)
        model1 = create_model(["w"], (10, 10), seed=1)
        model2 = create_model(["w"], (10, 10), seed=2)

        result = dare_ties_merge([model1, model2], base, drop_rate=0.5, density=0.5)

        assert "w" in result
        assert result["w"].shape == (10, 10)

    def test_dare_drop_rate_affects_sparsity(self):
        """Test that drop rate affects sparsification."""
        base = create_model(["w"], (50, 50), seed=0)
        model1 = create_model(["w"], (50, 50), seed=1)
        model2 = create_model(["w"], (50, 50), seed=2)

        # High drop rate = more sparse
        result_high_drop = dare_ties_merge(
            [model1, model2], base, drop_rate=0.99, density=1.0
        )
        # Low drop rate = less sparse
        result_low_drop = dare_ties_merge(
            [model1, model2], base, drop_rate=0.1, density=1.0
        )

        # Results should differ significantly
        diff = np.abs(result_high_drop["w"].data - result_low_drop["w"].data)
        assert np.mean(diff) > 0.01

    def test_dare_linear_merge(self):
        """Test DARE with linear averaging (no TIES)."""
        base = create_model(["w"], (8, 8), seed=0)
        model1 = create_model(["w"], (8, 8), seed=1)
        model2 = create_model(["w"], (8, 8), seed=2)

        config = SoupConfig(
            strategy=SoupStrategy.DARE_LINEAR,
            dare_drop_rate=0.5,
        )
        soup = ModelSoup(config)
        result = soup.merge([model1, model2], base_model=base)

        assert result.strategy == SoupStrategy.DARE_LINEAR
        assert "w" in result.weights


class TestTaskArithmetic:
    """Tests for task arithmetic merging."""

    def test_task_arithmetic_basic(self):
        """Test basic task arithmetic operation."""
        base = create_model(["w"], (10, 10), seed=0)
        model1 = create_model(["w"], (10, 10), seed=1)
        model2 = create_model(["w"], (10, 10), seed=2)

        config = SoupConfig(
            strategy=SoupStrategy.TASK_ARITHMETIC,
            task_scaling=0.5,
        )
        soup = ModelSoup(config)
        result = soup.merge([model1, model2], base_model=base)

        assert result.strategy == SoupStrategy.TASK_ARITHMETIC
        assert "scaling" in result.stats
        assert result.stats["scaling"] == 0.5

    def test_task_arithmetic_scaling(self):
        """Test that task scaling affects result magnitude."""
        base = {
            "w": WeightTensor(
                data=np.zeros((5, 5), dtype=np.float32),
                metadata=WeightMetadata(name="w", shape=(5, 5), dtype=np.float32),
            )
        }
        # Two models with task vectors of 1.0 each
        model1 = {
            "w": WeightTensor(
                data=np.ones((5, 5), dtype=np.float32),
                metadata=WeightMetadata(name="w", shape=(5, 5), dtype=np.float32),
            )
        }
        model2 = {
            "w": WeightTensor(
                data=np.ones((5, 5), dtype=np.float32),
                metadata=WeightMetadata(name="w", shape=(5, 5), dtype=np.float32),
            )
        }

        # Scale = 1.0: result = base + 1.0 * (tv1 + tv2) = 0 + 1.0 * (1 + 1) = 2
        config_1 = SoupConfig(strategy=SoupStrategy.TASK_ARITHMETIC, task_scaling=1.0)
        soup_1 = ModelSoup(config_1)
        result_1 = soup_1.merge([model1, model2], base_model=base)

        # Scale = 0.5: result = base + 0.5 * (tv1 + tv2) = 0 + 0.5 * (1 + 1) = 1
        config_05 = SoupConfig(strategy=SoupStrategy.TASK_ARITHMETIC, task_scaling=0.5)
        soup_05 = ModelSoup(config_05)
        result_05 = soup_05.merge([model1, model2], base_model=base)

        # Result with scale=1.0 should be ~2x result with scale=0.5
        mean_1 = np.mean(result_1.weights["w"].data)
        mean_05 = np.mean(result_05.weights["w"].data)
        ratio = mean_1 / mean_05
        assert abs(ratio - 2.0) < 0.01


class TestSoupResult:
    """Tests for SoupResult dataclass."""

    def test_soup_result_fields(self):
        """Test SoupResult contains expected fields."""
        weights = create_model(["w"], (5, 5), seed=1)
        result = SoupResult(
            weights=weights,
            strategy=SoupStrategy.UNIFORM,
            num_models=3,
            included_indices=[0, 1, 2],
            validation_scores=[0.8, 0.85, 0.9],
            stats={"key": "value"},
        )

        assert result.weights == weights
        assert result.strategy == SoupStrategy.UNIFORM
        assert result.num_models == 3
        assert result.included_indices == [0, 1, 2]
        assert result.validation_scores == [0.8, 0.85, 0.9]
        assert result.stats == {"key": "value"}


class TestModelSoupIntegration:
    """Integration tests for model soup with repository."""

    def test_soup_with_repository_merge(self):
        """Test model souping through repository merge."""
        from coral.version_control.repository import MergeStrategy, Repository

        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)

            # Create initial commit
            base_weights = create_model(["layer1", "layer2"], (32, 16), seed=0)
            repo.stage_weights(base_weights)
            repo.commit("Base model")

            # Create branch and modify
            repo.create_branch("experiment")
            repo.checkout("experiment")

            exp_weights = create_model(["layer1", "layer2"], (32, 16), seed=1)
            repo.stage_weights(exp_weights)
            repo.commit("Experiment changes")

            # Back to main and make different changes
            repo.checkout("main")
            main_weights = create_model(["layer1", "layer2"], (32, 16), seed=2)
            repo.stage_weights(main_weights)
            repo.commit("Main changes")

            # Merge with TIES strategy
            merge_commit = repo.merge(
                "experiment",
                strategy=MergeStrategy.TIES,
                ties_density=0.5,
            )

            assert merge_commit is not None
            assert "merge" in merge_commit.metadata.tags

            # Verify merged weights exist
            merged = repo.get_all_weights()
            assert "layer1" in merged
            assert "layer2" in merged

    def test_greedy_soup_merge_requires_validation(self):
        """Test that GREEDY_SOUP merge raises without validation_fn."""
        from coral.version_control.repository import MergeStrategy, Repository

        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)

            # Setup branches (abbreviated)
            base = create_model(["w"], (8, 8), seed=0)
            repo.stage_weights(base)
            repo.commit("Base")

            repo.create_branch("exp")
            repo.checkout("exp")
            repo.stage_weights(create_model(["w"], (8, 8), seed=1))
            repo.commit("Exp")

            repo.checkout("main")
            repo.stage_weights(create_model(["w"], (8, 8), seed=2))
            repo.commit("Main")

            with pytest.raises(ValueError, match="validation_fn"):
                repo.merge("exp", strategy=MergeStrategy.GREEDY_SOUP)

    def test_ties_merge_uses_ancestor_as_base(self):
        """Test that TIES uses common ancestor when no base_commit_ref."""
        from coral.version_control.repository import MergeStrategy, Repository

        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)

            # Create base
            base = create_model(["w"], (16, 16), seed=0)
            repo.stage_weights(base)
            repo.commit("Base")

            # Branch and modify
            repo.create_branch("feature")
            repo.checkout("feature")
            repo.stage_weights(create_model(["w"], (16, 16), seed=1))
            repo.commit("Feature")

            # Main modifications
            repo.checkout("main")
            repo.stage_weights(create_model(["w"], (16, 16), seed=2))
            repo.commit("Main update")

            # TIES merge should use common ancestor (base_commit) as base
            merge = repo.merge("feature", strategy=MergeStrategy.TIES)

            assert merge is not None


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_uniform_soup_function(self):
        """Test uniform_soup convenience function."""
        models = [create_model(["w"], (5, 5), seed=i) for i in range(2)]
        result = uniform_soup(models)

        assert isinstance(result, dict)
        assert "w" in result

    def test_greedy_soup_function(self):
        """Test greedy_soup convenience function."""
        models = [create_model(["w"], (5, 5), seed=i) for i in range(3)]

        result = greedy_soup(models, lambda w: -np.sum(w["w"] ** 2))

        assert isinstance(result, SoupResult)
        assert result.strategy == SoupStrategy.GREEDY

    def test_ties_merge_function(self):
        """Test ties_merge convenience function."""
        base = create_model(["w"], (5, 5), seed=0)
        models = [create_model(["w"], (5, 5), seed=i) for i in range(1, 3)]

        result = ties_merge(models, base, density=0.5)

        assert isinstance(result, dict)
        assert "w" in result

    def test_dare_ties_merge_function(self):
        """Test dare_ties_merge convenience function."""
        base = create_model(["w"], (5, 5), seed=0)
        models = [create_model(["w"], (5, 5), seed=i) for i in range(1, 3)]

        result = dare_ties_merge(models, base, drop_rate=0.5, density=0.5)

        assert isinstance(result, dict)
        assert "w" in result


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_models_list_raises(self):
        """Test that empty models list raises error."""
        soup = ModelSoup()

        with pytest.raises(ValueError, match="At least one model"):
            soup.merge([])

    def test_shape_mismatch_warning(self):
        """Test warning on shape mismatch during averaging."""
        model1 = {"w": create_weight("w", (5, 5), seed=1)}
        model2 = {"w": create_weight("w", (10, 10), seed=2)}  # Different shape

        # Should not crash, but will log warning
        result = uniform_soup([model1, model2])
        assert "w" in result

    def test_multiple_weight_names(self):
        """Test soup with many different weight names."""
        names = [f"layer{i}" for i in range(10)]
        models = [create_model(names, (8, 4), seed=i) for i in range(3)]

        result = uniform_soup(models)

        for name in names:
            assert name in result
