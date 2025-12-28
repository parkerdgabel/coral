"""TensorFlow/Keras integration for Coral version control.

This module provides utilities for saving and loading TensorFlow/Keras model
weights using Coral's version control system.

Example:
    >>> from coral.integrations.tensorflow import save, load
    >>> # Save model weights
    >>> save(model, "./weights", "Initial checkpoint")
    >>> # Load into model
    >>> load(model, "./weights")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except ImportError:
    tf = None  # type: ignore[assignment]
    keras = None  # type: ignore[assignment]
    TF_AVAILABLE = False


from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.version_control.repository import Repository

logger = logging.getLogger(__name__)


def _require_tensorflow() -> None:
    """Raise ImportError if TensorFlow is not available."""
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is not installed. "
            "Install with: pip install tensorflow>=2.8.0"
        )


def _get_layer_type(model: Any, weight_name: str) -> Optional[str]:
    """Get the layer type for a weight.

    Args:
        model: Keras model
        weight_name: Name of the weight

    Returns:
        Layer type string or None
    """
    if not TF_AVAILABLE:
        return None

    # Parse layer name from weight name (format: "layer_name/variable_name:0")
    parts = weight_name.split("/")
    if len(parts) >= 1:
        layer_name = parts[0]
        for layer in model.layers:
            if layer.name == layer_name:
                return layer.__class__.__name__
    return None


class TensorFlowIntegration:
    """Integration utilities for TensorFlow/Keras models.

    Provides static methods for converting between Keras models and
    Coral WeightTensor objects.

    Example:
        >>> weights = TensorFlowIntegration.model_to_weights(model)
        >>> TensorFlowIntegration.weights_to_model(weights, model)
    """

    @staticmethod
    def model_to_weights(model: Any) -> dict[str, WeightTensor]:
        """Convert Keras model weights to Coral weights.

        Args:
            model: Keras model (tf.keras.Model or keras.Model)

        Returns:
            Dictionary mapping weight names to WeightTensor objects

        Example:
            >>> model = keras.Sequential([keras.layers.Dense(10)])
            >>> model.build((None, 5))
            >>> weights = TensorFlowIntegration.model_to_weights(model)
        """
        _require_tensorflow()

        weights = {}

        for weight in model.weights:
            name = weight.name
            # Get numpy data
            data = weight.numpy()

            # Create metadata
            metadata = WeightMetadata(
                name=name,
                shape=data.shape,
                dtype=data.dtype,
                layer_type=_get_layer_type(model, name),
                model_name=model.__class__.__name__,
            )

            # Create weight tensor
            weight_tensor = WeightTensor(data=data, metadata=metadata)
            weights[name] = weight_tensor

        return weights

    @staticmethod
    def weights_to_model(weights: dict[str, WeightTensor], model: Any) -> None:
        """Load Coral weights into a Keras model.

        Args:
            weights: Dictionary of Coral WeightTensor objects
            model: Keras model to load weights into

        Note:
            The model must have the same architecture (layer names and shapes)
            as when the weights were saved.

        Example:
            >>> TensorFlowIntegration.weights_to_model(weights, model)
        """
        _require_tensorflow()

        # Create list of weight values in model weight order
        weight_values = []
        missing = []
        loaded = []

        for model_weight in model.weights:
            name = model_weight.name
            if name in weights:
                weight_values.append(weights[name].data)
                loaded.append(name)
            else:
                # Keep original weight if not in saved weights
                weight_values.append(model_weight.numpy())
                missing.append(name)

        if missing:
            logger.warning(f"Missing weights in saved data: {missing}")

        # Set all weights at once
        model.set_weights(weight_values)
        logger.debug(f"Loaded {len(loaded)} weights into model")

    @staticmethod
    def get_trainable_weights(model: Any) -> dict[str, WeightTensor]:
        """Get only trainable weights from a Keras model.

        Args:
            model: Keras model

        Returns:
            Dictionary of trainable weights only
        """
        _require_tensorflow()

        weights = {}

        for weight in model.trainable_weights:
            name = weight.name
            data = weight.numpy()

            metadata = WeightMetadata(
                name=name,
                shape=data.shape,
                dtype=data.dtype,
                layer_type=_get_layer_type(model, name),
                model_name=model.__class__.__name__,
            )

            weight_tensor = WeightTensor(data=data, metadata=metadata)
            weights[name] = weight_tensor

        return weights

    @staticmethod
    def get_non_trainable_weights(model: Any) -> dict[str, WeightTensor]:
        """Get only non-trainable weights from a Keras model.

        This includes batch normalization statistics and other non-trainable
        variables.

        Args:
            model: Keras model

        Returns:
            Dictionary of non-trainable weights only
        """
        _require_tensorflow()

        weights = {}

        for weight in model.non_trainable_weights:
            name = weight.name
            data = weight.numpy()

            metadata = WeightMetadata(
                name=name,
                shape=data.shape,
                dtype=data.dtype,
                layer_type=_get_layer_type(model, name),
                model_name=model.__class__.__name__,
            )

            weight_tensor = WeightTensor(data=data, metadata=metadata)
            weights[name] = weight_tensor

        return weights

    @staticmethod
    def get_optimizer_weights(optimizer: Any) -> dict[str, WeightTensor]:
        """Get optimizer state weights.

        Args:
            optimizer: Keras optimizer

        Returns:
            Dictionary of optimizer weights
        """
        _require_tensorflow()

        weights = {}

        if hasattr(optimizer, "weights") and optimizer.weights:
            for i, weight in enumerate(optimizer.weights):
                if hasattr(weight, "name"):
                    name = f"optimizer/{weight.name}"
                else:
                    name = f"optimizer/var_{i}"
                data = weight.numpy()

                metadata = WeightMetadata(
                    name=name,
                    shape=data.shape,
                    dtype=data.dtype,
                )

                weight_tensor = WeightTensor(data=data, metadata=metadata)
                weights[name] = weight_tensor

        return weights


def save(
    model: Any,
    repo: Union[str, Path, Repository],
    message: str,
    *,
    branch: Optional[str] = None,
    create_branch: bool = False,
    tag: Optional[str] = None,
    push_to: Optional[str] = None,
    init: bool = False,
    include_optimizer: bool = False,
    trainable_only: bool = False,
    **metadata,
) -> dict[str, Any]:
    """Save a TensorFlow/Keras model to a Coral repository.

    This is the recommended entry point for saving TensorFlow models.

    Args:
        model: Keras model to save
        repo: Path to repository or Repository object
        message: Commit message
        branch: Branch to save to
        create_branch: If True, create branch if it doesn't exist
        tag: Optional tag/version name to create for this commit
        push_to: Remote name to push to after commit
        init: If True, initialize repository if it doesn't exist
        include_optimizer: If True, also save optimizer state
        trainable_only: If True, only save trainable weights
        **metadata: Additional metadata to attach to weights

    Returns:
        Dictionary with save info:
        - commit_hash: Hash of the commit
        - weights_saved: Number of weights saved
        - branch: Current branch name
        - tag: Tag name if created
        - push_stats: Push statistics (if push_to was specified)

    Example:
        >>> from tensorflow import keras
        >>> model = keras.Sequential([keras.layers.Dense(10)])
        >>> model.build((None, 5))
        >>> result = save(model, "./weights", "Initial checkpoint")
        >>> print(result["commit_hash"])
    """
    _require_tensorflow()

    # Handle Repository object or path
    if isinstance(repo, (str, Path)):
        repository = Repository(Path(repo), init=init)
    else:
        repository = repo

    # Handle branching
    if branch:
        current_branch = repository.branch_manager.get_current_branch()
        if branch != current_branch:
            branches = [b.name for b in repository.branch_manager.list_branches()]
            if branch not in branches:
                if create_branch:
                    repository.create_branch(branch)
                else:
                    raise ValueError(
                        f"Branch '{branch}' doesn't exist. "
                        "Set create_branch=True to create it."
                    )
            repository.checkout(branch)

    # Convert model to weights
    if trainable_only:
        weights = TensorFlowIntegration.get_trainable_weights(model)
    else:
        weights = TensorFlowIntegration.model_to_weights(model)

    # Add optimizer weights if requested
    if include_optimizer and hasattr(model, "optimizer") and model.optimizer:
        optimizer_weights = TensorFlowIntegration.get_optimizer_weights(model.optimizer)
        weights.update(optimizer_weights)

    # Add metadata
    model_name = metadata.pop("model_name", model.__class__.__name__)
    for weight in weights.values():
        weight.metadata.model_name = model_name
        for key, value in metadata.items():
            if hasattr(weight.metadata, key):
                setattr(weight.metadata, key, value)

    # Stage and commit
    repository.stage_weights(weights)
    commit = repository.commit(message=message)

    result = {
        "commit_hash": commit.commit_hash,
        "weights_saved": len(weights),
        "branch": repository.branch_manager.get_current_branch(),
    }

    # Create tag if requested
    if tag:
        repository.tag_version(
            name=tag,
            description=message,
            commit_ref=commit.commit_hash,
        )
        result["tag"] = tag

    # Push if requested
    if push_to:
        push_stats = repository.push(push_to)
        result["push_stats"] = push_stats
        logger.info(f"Pushed to {push_to}")

    return result


def load(
    model: Any,
    repo: Union[str, Path, Repository],
    *,
    ref: Optional[str] = None,
    tag: Optional[str] = None,
    branch: Optional[str] = None,
    remote: Optional[str] = None,
    pull: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Load weights from a Coral repository into a TensorFlow/Keras model.

    This is the recommended entry point for loading models.

    Args:
        model: Keras model to load weights into
        repo: Path to repository or Repository object
        ref: Specific commit hash to load from
        tag: Tag/version name to load from
        branch: Branch name to load from
        remote: Remote name to pull from before loading
        pull: If True and remote specified, pull before loading
        strict: If True, raise error for missing weights

    Returns:
        Dictionary with load statistics:
        - loaded: List of loaded weight names
        - missing: List of missing model parameters
        - matched: Number of successfully matched weights
        - pull_stats: Pull statistics (if remote/pull used)

    Example:
        >>> from tensorflow import keras
        >>> model = keras.Sequential([keras.layers.Dense(10)])
        >>> model.build((None, 5))
        >>> stats = load(model, "./weights")
    """
    _require_tensorflow()

    # Handle Repository object or path
    if isinstance(repo, (str, Path)):
        repository = Repository(Path(repo))
    else:
        repository = repo

    # Pull from remote if requested
    pull_stats = None
    if remote and pull:
        pull_stats = repository.pull(remote)
        weights_count = pull_stats.get("weights_pulled", 0)
        logger.info(f"Pulled {weights_count} weights from {remote}")

    # Determine commit reference
    commit_ref = ref
    if commit_ref is None:
        if branch:
            commit_ref = repository.branch_manager.get_branch_commit(branch)
        elif tag:
            for v in repository.version_graph.versions.values():
                if v.name == tag:
                    commit_ref = v.commit_hash
                    break
            if commit_ref is None:
                raise ValueError(f"Tag '{tag}' not found")

    # Get weights from repository
    weights = repository.get_all_weights(commit_ref)
    if not weights:
        raise ValueError("No weights found in repository")

    # Load into model
    result = load_into_model(model, weights, strict=strict)

    if pull_stats:
        result["pull_stats"] = pull_stats

    return result


def load_into_model(
    model: Any,
    weights: dict[str, WeightTensor],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Load Coral weights into a Keras model.

    Args:
        model: Keras model to load weights into
        weights: Dictionary of Coral WeightTensor objects
        strict: If True, raise error for missing weights

    Returns:
        Dictionary with:
        - loaded: List of loaded weight names
        - missing: List of weights in model but not in saved data
        - unexpected: List of saved weights not in model
        - matched: Number of matched weights
    """
    _require_tensorflow()

    model_weight_names = {w.name for w in model.weights}
    saved_weight_names = set(weights.keys())

    loaded = []
    missing = []
    unexpected = list(saved_weight_names - model_weight_names)

    # Load weights that exist in both
    weight_values = []
    for model_weight in model.weights:
        name = model_weight.name
        if name in weights:
            weight_values.append(weights[name].data)
            loaded.append(name)
        else:
            weight_values.append(model_weight.numpy())
            missing.append(name)

    if strict and missing:
        raise ValueError(f"Missing weights in saved data: {missing}")

    if missing:
        logger.warning(f"Missing weights: {missing}")
    if unexpected:
        logger.warning(f"Unexpected weights in saved data: {unexpected}")

    model.set_weights(weight_values)

    return {
        "loaded": loaded,
        "missing": missing,
        "unexpected": unexpected,
        "matched": len(loaded),
    }


def save_model(
    model: Any,
    repository: Repository,
    message: str,
    **metadata,
) -> str:
    """Save Keras model weights to repository.

    Convenience function that wraps TensorFlowIntegration.model_to_weights.

    Args:
        model: Keras model to save
        repository: Coral repository
        message: Commit message
        **metadata: Additional metadata

    Returns:
        Commit hash
    """
    result = save(model, repository, message, **metadata)
    return result["commit_hash"]


def load_model_from_coral(
    model: Any,
    repository: Repository,
    commit_ref: Optional[str] = None,
) -> dict[str, Any]:
    """Load weights from Coral repository into Keras model.

    Args:
        model: Keras model to load into
        repository: Coral repository
        commit_ref: Optional specific commit to load from

    Returns:
        Load statistics dictionary
    """
    return load(model, repository, ref=commit_ref)


def compare_model_weights(
    model1: Any,
    model2: Any,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Compare weights between two Keras models.

    Args:
        model1: First Keras model
        model2: Second Keras model
        tolerance: Numerical tolerance for comparison

    Returns:
        Dictionary with comparison results:
        - identical: List of identical weight names
        - different: List of different weight names with max difference
        - missing_in_first: Weights in model2 but not model1
        - missing_in_second: Weights in model1 but not model2
    """
    _require_tensorflow()

    weights1 = TensorFlowIntegration.model_to_weights(model1)
    weights2 = TensorFlowIntegration.model_to_weights(model2)

    names1 = set(weights1.keys())
    names2 = set(weights2.keys())

    identical = []
    different = []

    for name in names1 & names2:
        w1 = weights1[name].data
        w2 = weights2[name].data

        if w1.shape != w2.shape:
            different.append({
                "name": name,
                "reason": "shape_mismatch",
                "shape1": w1.shape,
                "shape2": w2.shape,
            })
        else:
            max_diff = float(np.max(np.abs(w1 - w2)))
            if max_diff <= tolerance:
                identical.append(name)
            else:
                different.append({
                    "name": name,
                    "reason": "value_difference",
                    "max_diff": max_diff,
                })

    return {
        "identical": identical,
        "different": different,
        "missing_in_first": list(names2 - names1),
        "missing_in_second": list(names1 - names2),
    }
