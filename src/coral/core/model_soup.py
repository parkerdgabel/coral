"""Model Souping: Advanced weight merging algorithms.

This module implements model souping techniques from recent ML research for
combining multiple fine-tuned models into a single, better-performing model.

Algorithms implemented:
- Uniform Soup: Simple weight averaging (baseline)
- Greedy Soup: Iteratively add models only if they improve validation metric
- TIES-Merging: Trim redundant params, Elect sign, merge aligned values
- DARE: Drop and Rescale delta parameters before merging

References:
- Wortsman et al. (2022) "Model soups: averaging weights of multiple fine-tuned
  models improves accuracy without increasing inference time" (ICML 2022)
- Yadav et al. (2023) "TIES-Merging: Resolving Interference When Merging Models"
- Yu et al. (2023) "Language Models are Super Mario: Absorbing Abilities from
  Homologous Models as a Free Lunch" (DARE)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

from coral.core.weight_tensor import WeightMetadata, WeightTensor

logger = logging.getLogger(__name__)


class SoupStrategy(Enum):
    """Available model souping strategies."""

    UNIFORM = "uniform"  # Simple averaging
    GREEDY = "greedy"  # Add models only if they improve validation
    TIES = "ties"  # TIES-Merging: Trim, Elect Sign, Merge
    DARE_TIES = "dare_ties"  # DARE preprocessing + TIES
    DARE_LINEAR = "dare_linear"  # DARE preprocessing + linear interpolation
    TASK_ARITHMETIC = "task_arithmetic"  # Task vectors with scaling


@dataclass
class SoupConfig:
    """Configuration for model souping."""

    # Strategy to use
    strategy: SoupStrategy = SoupStrategy.UNIFORM

    # For greedy soup: validation function that returns a score (higher=better)
    # Signature: (weights: dict[str, np.ndarray]) -> float
    validation_fn: Optional[Callable[[dict[str, np.ndarray]], float]] = None

    # For TIES: Top-k% of parameters to keep (by magnitude)
    ties_density: float = 0.5

    # For TIES: How to resolve sign conflicts ("total" or "frequency")
    ties_sign_method: str = "total"

    # For DARE: Probability of dropping a delta parameter
    dare_drop_rate: float = 0.9

    # For DARE: Random seed for reproducibility
    dare_seed: int = 42

    # For task arithmetic: Scaling factor for task vectors
    task_scaling: float = 1.0

    # Whether to normalize weights before averaging
    normalize_weights: bool = False

    # Weights for each model (for weighted averaging)
    model_weights: Optional[list[float]] = None


@dataclass
class SoupResult:
    """Result of a model souping operation."""

    # The merged weights
    weights: dict[str, WeightTensor]

    # Strategy used
    strategy: SoupStrategy

    # Number of models included in the soup
    num_models: int

    # For greedy soup: which models were included
    included_indices: list[int] = field(default_factory=list)

    # Validation scores during greedy selection (if applicable)
    validation_scores: list[float] = field(default_factory=list)

    # Statistics about the merge
    stats: dict[str, Any] = field(default_factory=dict)


class ModelSoup:
    """Main class for model souping operations.

    Example:
        >>> soup = ModelSoup(SoupConfig(strategy=SoupStrategy.GREEDY))
        >>> result = soup.merge([model1_weights, model2_weights, model3_weights])
        >>> merged_weights = result.weights
    """

    def __init__(self, config: Optional[SoupConfig] = None):
        """Initialize model soup.

        Args:
            config: Configuration for souping algorithms
        """
        self.config = config or SoupConfig()

    def merge(
        self,
        models: list[dict[str, WeightTensor]],
        base_model: Optional[dict[str, WeightTensor]] = None,
    ) -> SoupResult:
        """Merge multiple models using the configured strategy.

        Args:
            models: List of model weight dictionaries to merge
            base_model: Optional base/pretrained model for computing task vectors.
                Required for TIES, DARE, and task arithmetic strategies.

        Returns:
            SoupResult containing merged weights and metadata
        """
        if not models:
            raise ValueError("At least one model is required")

        if len(models) == 1:
            return SoupResult(
                weights=models[0],
                strategy=self.config.strategy,
                num_models=1,
                included_indices=[0],
            )

        strategy = self.config.strategy

        if strategy == SoupStrategy.UNIFORM:
            return self._uniform_soup(models)
        elif strategy == SoupStrategy.GREEDY:
            return self._greedy_soup(models)
        elif strategy == SoupStrategy.TIES:
            if base_model is None:
                raise ValueError("TIES-Merging requires a base model")
            return self._ties_soup(models, base_model)
        elif strategy == SoupStrategy.DARE_TIES:
            if base_model is None:
                raise ValueError("DARE-TIES requires a base model")
            return self._dare_ties_soup(models, base_model)
        elif strategy == SoupStrategy.DARE_LINEAR:
            if base_model is None:
                raise ValueError("DARE-Linear requires a base model")
            return self._dare_linear_soup(models, base_model)
        elif strategy == SoupStrategy.TASK_ARITHMETIC:
            if base_model is None:
                raise ValueError("Task arithmetic requires a base model")
            return self._task_arithmetic_soup(models, base_model)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _uniform_soup(
        self, models: list[dict[str, WeightTensor]]
    ) -> SoupResult:
        """Simple uniform averaging of all model weights.

        This is the baseline souping method - just average all weights.
        """
        merged = {}
        all_names = set()
        for model in models:
            all_names.update(model.keys())

        model_weights = self.config.model_weights
        if model_weights is None:
            model_weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights to sum to 1
            total = sum(model_weights)
            model_weights = [w / total for w in model_weights]

        for name in all_names:
            # Collect all weights for this parameter
            weights_data = []
            weights_for_avg = []
            example_weight = None

            for i, model in enumerate(models):
                if name in model:
                    w = model[name]
                    weights_data.append(w.data)
                    weights_for_avg.append(model_weights[i])
                    if example_weight is None:
                        example_weight = w

            if not weights_data:
                continue

            # Normalize weights for this parameter
            total = sum(weights_for_avg)
            weights_for_avg = [w / total for w in weights_for_avg]

            # Weighted average
            avg_data = np.zeros_like(weights_data[0])
            for data, weight in zip(weights_data, weights_for_avg):
                if data.shape == avg_data.shape:
                    avg_data += weight * data
                else:
                    logger.warning(
                        f"Shape mismatch for {name}: {data.shape} vs {avg_data.shape}"
                    )

            merged[name] = WeightTensor(
                data=avg_data.astype(example_weight.dtype),
                metadata=WeightMetadata(
                    name=name,
                    shape=avg_data.shape,
                    dtype=example_weight.dtype,
                ),
            )

        return SoupResult(
            weights=merged,
            strategy=SoupStrategy.UNIFORM,
            num_models=len(models),
            included_indices=list(range(len(models))),
        )

    def _greedy_soup(
        self, models: list[dict[str, WeightTensor]]
    ) -> SoupResult:
        """Greedy soup: add models only if they improve validation metric.

        Models are sorted by individual performance, then added to the soup
        one by one, keeping each addition only if it improves the soup's
        overall validation score.
        """
        if self.config.validation_fn is None:
            raise ValueError("Greedy soup requires a validation_fn in config")

        validate = self.config.validation_fn

        # First, evaluate each model individually
        individual_scores = []
        for i, model in enumerate(models):
            # Convert to numpy dict for validation
            weights_np = {name: w.data for name, w in model.items()}
            score = validate(weights_np)
            individual_scores.append((i, score))
            logger.debug(f"Model {i} individual score: {score:.4f}")

        # Sort by score (descending - best first)
        individual_scores.sort(key=lambda x: x[1], reverse=True)

        # Start with the best individual model
        best_idx, best_score = individual_scores[0]
        soup_indices = [best_idx]
        soup_weights = [models[best_idx]]
        validation_scores = [best_score]

        logger.info(
            f"Starting greedy soup with model {best_idx} (score: {best_score:.4f})"
        )

        # Try adding each remaining model
        for idx, _ in individual_scores[1:]:
            # Create candidate soup with this model added
            candidate_weights = soup_weights + [models[idx]]

            # Compute uniform average of candidate soup
            candidate_soup = self._compute_uniform_average(candidate_weights)

            # Evaluate candidate
            candidate_score = validate(candidate_soup)

            if candidate_score > best_score:
                # Keep this model in the soup
                soup_indices.append(idx)
                soup_weights.append(models[idx])
                best_score = candidate_score
                validation_scores.append(candidate_score)
                logger.info(
                    f"Added model {idx} to soup (score: {candidate_score:.4f})"
                )
            else:
                logger.debug(
                    f"Rejected model {idx} (would have score: {candidate_score:.4f})"
                )

        # Create final soup
        if len(soup_weights) == 1:
            merged = soup_weights[0]
        else:
            merged = self._uniform_soup(soup_weights).weights

        return SoupResult(
            weights=merged,
            strategy=SoupStrategy.GREEDY,
            num_models=len(soup_indices),
            included_indices=soup_indices,
            validation_scores=validation_scores,
            stats={
                "num_candidates": len(models),
                "num_included": len(soup_indices),
                "final_score": best_score,
            },
        )

    def _compute_uniform_average(
        self, models: list[dict[str, WeightTensor]]
    ) -> dict[str, np.ndarray]:
        """Compute uniform average as numpy dict (for validation)."""
        all_names = set()
        for model in models:
            all_names.update(model.keys())

        result = {}
        for name in all_names:
            weights_data = [
                model[name].data for model in models if name in model
            ]
            if weights_data:
                result[name] = np.mean(weights_data, axis=0)

        return result

    def _compute_task_vectors(
        self,
        models: list[dict[str, WeightTensor]],
        base_model: dict[str, WeightTensor],
    ) -> list[dict[str, np.ndarray]]:
        """Compute task vectors (delta from base model) for each model."""
        task_vectors = []

        for model in models:
            task_vector = {}
            for name in model:
                if name in base_model:
                    delta = model[name].data - base_model[name].data
                    task_vector[name] = delta
                else:
                    # New parameter not in base, use as-is
                    task_vector[name] = model[name].data
            task_vectors.append(task_vector)

        return task_vectors

    def _ties_soup(
        self,
        models: list[dict[str, WeightTensor]],
        base_model: dict[str, WeightTensor],
    ) -> SoupResult:
        """TIES-Merging: Trim, Elect Sign, and Merge.

        Steps:
        1. Compute task vectors (deltas from base)
        2. TRIM: Keep only top-k% parameters by magnitude
        3. ELECT SIGN: For each parameter, choose majority sign
        4. MERGE: Average only values that agree with elected sign
        """
        density = self.config.ties_density
        sign_method = self.config.ties_sign_method

        # Step 1: Compute task vectors
        task_vectors = self._compute_task_vectors(models, base_model)

        all_names = set()
        for tv in task_vectors:
            all_names.update(tv.keys())

        merged = {}
        stats = {"trimmed_params": 0, "sign_conflicts": 0}

        for name in all_names:
            # Collect deltas for this parameter
            deltas = [tv.get(name) for tv in task_vectors]
            deltas = [d for d in deltas if d is not None]

            if not deltas:
                continue

            # Step 2: TRIM - keep only top-k% by magnitude per task vector
            trimmed_deltas = []
            for delta in deltas:
                flat = delta.flatten()
                k = max(1, int(len(flat) * density))
                threshold = np.partition(np.abs(flat), -k)[-k]
                trimmed = np.where(np.abs(delta) >= threshold, delta, 0.0)
                trimmed_deltas.append(trimmed)
                stats["trimmed_params"] += np.sum(np.abs(delta) < threshold)

            # Step 3: ELECT SIGN - determine majority sign for each element
            stacked = np.stack(trimmed_deltas)

            if sign_method == "total":
                # Sum of magnitudes per sign
                positive_mass = np.sum(np.where(stacked > 0, stacked, 0), axis=0)
                negative_mass = np.sum(np.where(stacked < 0, -stacked, 0), axis=0)
                elected_sign = np.where(positive_mass >= negative_mass, 1, -1)
            else:  # "frequency"
                # Count of positive vs negative
                positive_count = np.sum(stacked > 0, axis=0)
                negative_count = np.sum(stacked < 0, axis=0)
                elected_sign = np.where(positive_count >= negative_count, 1, -1)

            # Step 4: MERGE - average only values agreeing with elected sign
            merged_delta = np.zeros_like(deltas[0])
            counts = np.zeros_like(deltas[0])

            for delta in trimmed_deltas:
                # Mask: value agrees with elected sign (or is zero)
                agrees = ((delta > 0) & (elected_sign > 0)) | \
                         ((delta < 0) & (elected_sign < 0))
                merged_delta += np.where(agrees, delta, 0.0)
                counts += agrees.astype(float)

            # Average where we have contributions
            counts = np.maximum(counts, 1)  # Avoid division by zero
            merged_delta = merged_delta / counts

            # Count sign conflicts
            sign_conflicts = np.sum(
                (stacked > 0).any(axis=0) & (stacked < 0).any(axis=0)
            )
            stats["sign_conflicts"] += int(sign_conflicts)

            # Add merged delta to base
            if name in base_model:
                final_data = base_model[name].data + merged_delta
            else:
                final_data = merged_delta

            merged[name] = WeightTensor(
                data=final_data.astype(base_model.get(name, models[0][name]).dtype),
                metadata=WeightMetadata(
                    name=name,
                    shape=final_data.shape,
                    dtype=base_model.get(name, models[0][name]).dtype,
                ),
            )

        return SoupResult(
            weights=merged,
            strategy=SoupStrategy.TIES,
            num_models=len(models),
            included_indices=list(range(len(models))),
            stats=stats,
        )

    def _apply_dare(
        self, task_vectors: list[dict[str, np.ndarray]]
    ) -> list[dict[str, np.ndarray]]:
        """Apply DARE: Drop and Rescale delta parameters.

        Randomly drops a fraction of delta parameters and rescales
        the remaining ones to preserve the expected value.
        """
        drop_rate = self.config.dare_drop_rate
        rng = np.random.RandomState(self.config.dare_seed)

        sparsified = []
        for tv in task_vectors:
            sparse_tv = {}
            for name, delta in tv.items():
                # Create random mask
                mask = rng.random(delta.shape) > drop_rate
                # Rescale remaining values
                if drop_rate < 1.0:
                    scale = 1.0 / (1.0 - drop_rate)
                else:
                    scale = 0.0
                sparse_tv[name] = delta * mask * scale
            sparsified.append(sparse_tv)

        return sparsified

    def _dare_ties_soup(
        self,
        models: list[dict[str, WeightTensor]],
        base_model: dict[str, WeightTensor],
    ) -> SoupResult:
        """DARE preprocessing followed by TIES-Merging."""
        # Compute task vectors
        task_vectors = self._compute_task_vectors(models, base_model)

        # Apply DARE sparsification
        sparse_vectors = self._apply_dare(task_vectors)

        # Create pseudo-models from sparse vectors for TIES
        sparse_models = []
        for sv in sparse_vectors:
            sparse_model = {}
            for name, delta in sv.items():
                if name in base_model:
                    data = base_model[name].data + delta
                else:
                    data = delta
                sparse_model[name] = WeightTensor(
                    data=data,
                    metadata=WeightMetadata(
                        name=name, shape=data.shape, dtype=data.dtype
                    ),
                )
            sparse_models.append(sparse_model)

        # Apply TIES
        result = self._ties_soup(sparse_models, base_model)
        result.strategy = SoupStrategy.DARE_TIES
        return result

    def _dare_linear_soup(
        self,
        models: list[dict[str, WeightTensor]],
        base_model: dict[str, WeightTensor],
    ) -> SoupResult:
        """DARE preprocessing followed by linear averaging."""
        # Compute task vectors
        task_vectors = self._compute_task_vectors(models, base_model)

        # Apply DARE sparsification
        sparse_vectors = self._apply_dare(task_vectors)

        # Average the sparse task vectors
        all_names = set()
        for sv in sparse_vectors:
            all_names.update(sv.keys())

        merged = {}
        for name in all_names:
            deltas = [sv.get(name) for sv in sparse_vectors]
            deltas = [d for d in deltas if d is not None]

            if not deltas:
                continue

            avg_delta = np.mean(deltas, axis=0)

            if name in base_model:
                final_data = base_model[name].data + avg_delta
            else:
                final_data = avg_delta

            merged[name] = WeightTensor(
                data=final_data.astype(base_model.get(name, models[0][name]).dtype),
                metadata=WeightMetadata(
                    name=name,
                    shape=final_data.shape,
                    dtype=base_model.get(name, models[0][name]).dtype,
                ),
            )

        return SoupResult(
            weights=merged,
            strategy=SoupStrategy.DARE_LINEAR,
            num_models=len(models),
            included_indices=list(range(len(models))),
        )

    def _task_arithmetic_soup(
        self,
        models: list[dict[str, WeightTensor]],
        base_model: dict[str, WeightTensor],
    ) -> SoupResult:
        """Task arithmetic: scale and add task vectors.

        Merged = Base + scaling * sum(task_vectors)
        """
        scaling = self.config.task_scaling

        # Compute and sum task vectors
        task_vectors = self._compute_task_vectors(models, base_model)

        all_names = set()
        for tv in task_vectors:
            all_names.update(tv.keys())

        merged = {}
        for name in all_names:
            deltas = [tv.get(name) for tv in task_vectors]
            deltas = [d for d in deltas if d is not None]

            if not deltas:
                continue

            # Sum all task vectors and scale
            total_delta = sum(deltas) * scaling

            if name in base_model:
                final_data = base_model[name].data + total_delta
            else:
                final_data = total_delta

            example = base_model.get(name, models[0][name])
            merged[name] = WeightTensor(
                data=final_data.astype(example.dtype),
                metadata=WeightMetadata(
                    name=name,
                    shape=final_data.shape,
                    dtype=example.dtype,
                ),
            )

        return SoupResult(
            weights=merged,
            strategy=SoupStrategy.TASK_ARITHMETIC,
            num_models=len(models),
            included_indices=list(range(len(models))),
            stats={"scaling": scaling},
        )


def uniform_soup(models: list[dict[str, WeightTensor]]) -> dict[str, WeightTensor]:
    """Convenience function for uniform model soup.

    Args:
        models: List of model weight dictionaries

    Returns:
        Merged weights dictionary
    """
    soup = ModelSoup(SoupConfig(strategy=SoupStrategy.UNIFORM))
    return soup.merge(models).weights


def greedy_soup(
    models: list[dict[str, WeightTensor]],
    validation_fn: Callable[[dict[str, np.ndarray]], float],
) -> SoupResult:
    """Convenience function for greedy model soup.

    Args:
        models: List of model weight dictionaries
        validation_fn: Function that evaluates model weights and returns score

    Returns:
        SoupResult with merged weights and selection info
    """
    config = SoupConfig(strategy=SoupStrategy.GREEDY, validation_fn=validation_fn)
    soup = ModelSoup(config)
    return soup.merge(models)


def ties_merge(
    models: list[dict[str, WeightTensor]],
    base_model: dict[str, WeightTensor],
    density: float = 0.5,
) -> dict[str, WeightTensor]:
    """Convenience function for TIES-Merging.

    Args:
        models: List of fine-tuned model weights
        base_model: Base/pretrained model weights
        density: Top-k% of parameters to keep (default 0.5)

    Returns:
        Merged weights dictionary
    """
    config = SoupConfig(strategy=SoupStrategy.TIES, ties_density=density)
    soup = ModelSoup(config)
    return soup.merge(models, base_model).weights


def dare_ties_merge(
    models: list[dict[str, WeightTensor]],
    base_model: dict[str, WeightTensor],
    drop_rate: float = 0.9,
    density: float = 0.5,
) -> dict[str, WeightTensor]:
    """Convenience function for DARE + TIES-Merging.

    Args:
        models: List of fine-tuned model weights
        base_model: Base/pretrained model weights
        drop_rate: Fraction of delta parameters to drop (default 0.9)
        density: TIES density parameter (default 0.5)

    Returns:
        Merged weights dictionary
    """
    config = SoupConfig(
        strategy=SoupStrategy.DARE_TIES,
        dare_drop_rate=drop_rate,
        ties_density=density,
    )
    soup = ModelSoup(config)
    return soup.merge(models, base_model).weights
