"""Utility functions for coral"""

from coral.utils.json_utils import NumpyJSONEncoder, dump_numpy, dumps_numpy
from coral.utils.similarity import (
    are_similar,
    cosine_similarity,
    magnitude_similarity,
    relative_difference,
    weight_similarity,
)
from coral.utils.visualization import (
    compare_models,
    format_model_diff,
    plot_deduplication_stats,
    plot_weight_distribution,
)

__all__ = [
    "plot_weight_distribution",
    "plot_deduplication_stats",
    "compare_models",
    "format_model_diff",
    "NumpyJSONEncoder",
    "dumps_numpy",
    "dump_numpy",
    "cosine_similarity",
    "magnitude_similarity",
    "weight_similarity",
    "relative_difference",
    "are_similar",
]
