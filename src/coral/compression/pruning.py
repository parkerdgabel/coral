"""Weight pruning for sparsity-based compression"""

import logging
from typing import Any, Optional

import numpy as np

from coral.core.weight_tensor import WeightMetadata, WeightTensor

logger = logging.getLogger(__name__)


class Pruner:
    """
    Weight pruning methods for introducing sparsity.

    Supports:
    - Magnitude-based pruning
    - Structured pruning (channels, filters)
    - Gradual pruning schedules
    """

    @staticmethod
    def prune_magnitude(
        weight: WeightTensor,
        sparsity: float = 0.5,
        structured: bool = False,
        axis: Optional[int] = None,
    ) -> tuple[WeightTensor, dict[str, Any]]:
        """
        Prune weights based on magnitude.

        Args:
            weight: Weight tensor to prune
            sparsity: Target sparsity level (0-1)
            structured: Whether to use structured pruning
            axis: Axis for structured pruning

        Returns:
            Tuple of (pruned_weight, pruning_mask)
        """
        if not 0 <= sparsity < 1:
            raise ValueError(f"Sparsity must be in [0, 1), got {sparsity}")

        data = weight.data.copy()

        if structured and axis is not None:
            # Structured pruning along specified axis
            mask = Pruner._structured_magnitude_pruning(data, sparsity, axis)
        else:
            # Unstructured pruning
            mask = Pruner._unstructured_magnitude_pruning(data, sparsity)

        # Apply mask
        pruned_data = data * mask

        # Count actual sparsity
        actual_sparsity = 1.0 - (np.count_nonzero(mask) / mask.size)

        # Create pruned weight tensor
        pruned_metadata = WeightMetadata(
            name=weight.metadata.name + "_pruned",
            shape=weight.shape,
            dtype=weight.dtype,
            layer_type=weight.metadata.layer_type,
            model_name=weight.metadata.model_name,
            compression_info={
                "method": "magnitude_pruning",
                "target_sparsity": sparsity,
                "actual_sparsity": actual_sparsity,
                "structured": structured,
                "axis": axis,
            },
        )

        pruned_weight = WeightTensor(data=pruned_data, metadata=pruned_metadata)

        pruning_info = {
            "mask": mask,
            "sparsity": actual_sparsity,
            "pruned_elements": int(mask.size * actual_sparsity),
        }

        logger.debug(f"Pruned {weight.metadata.name} to {actual_sparsity:.2%} sparsity")

        return pruned_weight, pruning_info

    @staticmethod
    def _unstructured_magnitude_pruning(
        data: np.ndarray, sparsity: float
    ) -> np.ndarray:
        """Unstructured magnitude-based pruning"""
        # Flatten and get absolute values
        flat_weights = data.flatten()
        abs_weights = np.abs(flat_weights)

        # Find threshold
        k = int(len(flat_weights) * sparsity)
        if k > 0:
            threshold = np.partition(abs_weights, k)[k]
            mask = abs_weights >= threshold
        else:
            mask = np.ones_like(flat_weights, dtype=bool)

        return mask.reshape(data.shape)

    @staticmethod
    def _structured_magnitude_pruning(
        data: np.ndarray, sparsity: float, axis: int
    ) -> np.ndarray:
        """Structured magnitude-based pruning along an axis"""
        # Compute importance scores for each structure
        axes_to_reduce = tuple(i for i in range(data.ndim) if i != axis)
        importance = np.sum(np.abs(data), axis=axes_to_reduce)

        # Find structures to prune
        n_structures = importance.shape[0]
        n_prune = int(n_structures * sparsity)

        if n_prune > 0:
            threshold_idx = np.partition(importance, n_prune)[n_prune]
            structure_mask = importance >= threshold_idx
        else:
            structure_mask = np.ones_like(importance, dtype=bool)

        # Expand mask to full shape
        shape = [1] * data.ndim
        shape[axis] = n_structures
        structure_mask = structure_mask.reshape(shape)

        # Broadcast to full mask
        mask = np.broadcast_to(structure_mask, data.shape).copy()

        return mask

    @staticmethod
    def prune_random(
        weight: WeightTensor, sparsity: float = 0.5
    ) -> tuple[WeightTensor, dict[str, Any]]:
        """
        Random pruning (mainly for testing/comparison).

        Args:
            weight: Weight tensor to prune
            sparsity: Target sparsity level (0-1)

        Returns:
            Tuple of (pruned_weight, pruning_mask)
        """
        data = weight.data.copy()

        # Create random mask
        mask = np.random.rand(*data.shape) >= sparsity
        pruned_data = data * mask

        actual_sparsity = 1.0 - (np.count_nonzero(mask) / mask.size)

        # Create pruned weight tensor
        pruned_metadata = WeightMetadata(
            name=weight.metadata.name + "_pruned_random",
            shape=weight.shape,
            dtype=weight.dtype,
            layer_type=weight.metadata.layer_type,
            model_name=weight.metadata.model_name,
            compression_info={
                "method": "random_pruning",
                "target_sparsity": sparsity,
                "actual_sparsity": actual_sparsity,
            },
        )

        pruned_weight = WeightTensor(data=pruned_data, metadata=pruned_metadata)

        pruning_info = {
            "mask": mask,
            "sparsity": actual_sparsity,
            "pruned_elements": int(mask.size * actual_sparsity),
        }

        return pruned_weight, pruning_info

    @staticmethod
    def get_sparsity_pattern(weight: WeightTensor) -> dict[str, Any]:
        """
        Analyze sparsity pattern of a weight tensor.

        Returns:
            Dictionary with sparsity statistics
        """
        data = weight.data

        # Overall sparsity
        total_elements = data.size
        zero_elements = np.count_nonzero(data == 0)
        sparsity = zero_elements / total_elements

        # Per-axis sparsity
        axis_sparsity = {}
        for axis in range(data.ndim):
            axes_to_check = tuple(i for i in range(data.ndim) if i != axis)
            zeros_per_slice = np.sum(data == 0, axis=axes_to_check)
            total_per_slice = np.prod([data.shape[i] for i in axes_to_check])
            axis_sparsity[f"axis_{axis}"] = (zeros_per_slice / total_per_slice).tolist()

        return {
            "total_sparsity": sparsity,
            "zero_elements": zero_elements,
            "total_elements": total_elements,
            "axis_sparsity": axis_sparsity,
            "shape": data.shape,
        }

    @staticmethod
    def apply_mask(weight: WeightTensor, mask: np.ndarray) -> WeightTensor:
        """
        Apply a pruning mask to a weight tensor.

        Args:
            weight: Weight tensor
            mask: Binary mask (same shape as weight)

        Returns:
            Masked weight tensor
        """
        if mask.shape != weight.shape:
            raise ValueError(
                f"Mask shape {mask.shape} doesn't match weight shape {weight.shape}"
            )

        masked_data = weight.data * mask

        masked_metadata = WeightMetadata(
            name=weight.metadata.name + "_masked",
            shape=weight.shape,
            dtype=weight.dtype,
            layer_type=weight.metadata.layer_type,
            model_name=weight.metadata.model_name,
            compression_info=weight.metadata.compression_info.copy(),
        )

        return WeightTensor(data=masked_data, metadata=masked_metadata)
