"""Visualization utilities for weight analysis"""

import logging
from typing import Any, Dict, List

import numpy as np

from coral.core.deduplicator import DeduplicationStats
from coral.core.weight_tensor import WeightTensor

logger = logging.getLogger(__name__)


def plot_weight_distribution(
    weights: List[WeightTensor], bins: int = 50
) -> Dict[str, Any]:
    """
    Analyze weight distribution (returns data for plotting).

    Since we can't use matplotlib in this environment, this returns
    the histogram data that can be plotted elsewhere.

    Args:
        weights: List of weight tensors
        bins: Number of histogram bins

    Returns:
        Dictionary with histogram data for each weight
    """
    distributions = {}

    for weight in weights:
        data = weight.data.flatten()
        hist, bin_edges = np.histogram(data, bins=bins)

        distributions[weight.metadata.name] = {
            "histogram": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "sparsity": float(np.count_nonzero(data == 0) / data.size),
        }

    return distributions


def plot_deduplication_stats(stats: DeduplicationStats) -> Dict[str, Any]:
    """
    Prepare deduplication statistics for visualization.

    Args:
        stats: Deduplication statistics

    Returns:
        Dictionary with visualization data
    """
    return {
        "weight_counts": {
            "unique": stats.unique_weights,
            "duplicate": stats.duplicate_weights,
            "similar": stats.similar_weights,
            "total": stats.total_weights,
        },
        "compression": {
            "bytes_saved": stats.bytes_saved,
            "compression_ratio": stats.compression_ratio,
        },
        "pie_data": [
            {"label": "Unique", "value": stats.unique_weights},
            {"label": "Duplicate", "value": stats.duplicate_weights},
            {"label": "Similar", "value": stats.similar_weights},
        ],
    }
