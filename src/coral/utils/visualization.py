"""Visualization utilities for weight analysis"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from coral.core.deduplicator import DeduplicationStats
from coral.core.weight_tensor import WeightTensor

logger = logging.getLogger(__name__)


def plot_weight_distribution(
    weights: list[WeightTensor], bins: int = 50
) -> dict[str, Any]:
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


def plot_deduplication_stats(stats: DeduplicationStats) -> dict[str, Any]:
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


def compare_models(
    weights_a: dict[str, WeightTensor],
    weights_b: dict[str, WeightTensor],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> dict[str, Any]:
    """
    Compare two sets of model weights and generate diff visualization data.

    This function analyzes differences between two models, useful for:
    - Comparing fine-tuned models to their base models
    - Analyzing training checkpoints
    - Understanding model evolution

    Args:
        weights_a: First model's weights (dict of name -> WeightTensor)
        weights_b: Second model's weights (dict of name -> WeightTensor)
        model_a_name: Display name for first model
        model_b_name: Display name for second model

    Returns:
        Dictionary with comprehensive comparison data for visualization

    Example:
        >>> from coral.utils.visualization import compare_models
        >>> diff = compare_models(base_weights, finetuned_weights)
        >>> print(f"Similarity: {diff['summary']['overall_similarity']:.1%}")
    """
    # Import here to avoid circular dependency
    from coral.utils.similarity import cosine_similarity, magnitude_similarity

    all_names = set(weights_a.keys()) | set(weights_b.keys())
    common_names = set(weights_a.keys()) & set(weights_b.keys())

    # Layer-by-layer comparison
    layer_comparisons = []
    total_params_a = 0
    total_params_b = 0
    changed_params = 0

    for name in sorted(all_names):
        in_a = name in weights_a
        in_b = name in weights_b

        comparison = {
            "name": name,
            "in_model_a": in_a,
            "in_model_b": in_b,
        }

        if in_a:
            wa = weights_a[name]
            comparison["shape_a"] = list(wa.shape)
            comparison["dtype_a"] = str(wa.dtype)
            comparison["params_a"] = int(np.prod(wa.shape))
            total_params_a += comparison["params_a"]

        if in_b:
            wb = weights_b[name]
            comparison["shape_b"] = list(wb.shape)
            comparison["dtype_b"] = str(wb.dtype)
            comparison["params_b"] = int(np.prod(wb.shape))
            total_params_b += comparison["params_b"]

        # Compute similarity for common layers
        if in_a and in_b:
            wa, wb = weights_a[name], weights_b[name]

            if wa.shape == wb.shape and wa.dtype == wb.dtype:
                cos_sim = cosine_similarity(wa.data, wb.data)
                mag_sim = magnitude_similarity(wa.data, wb.data)

                comparison["cosine_similarity"] = float(cos_sim)
                comparison["magnitude_similarity"] = float(mag_sim)
                comparison["combined_similarity"] = float(
                    0.7 * ((cos_sim + 1) / 2) + 0.3 * mag_sim
                )

                # Compute element-wise difference statistics
                diff = wa.data.astype(np.float64) - wb.data.astype(np.float64)
                comparison["diff_stats"] = {
                    "mean": float(np.mean(diff)),
                    "std": float(np.std(diff)),
                    "abs_mean": float(np.mean(np.abs(diff))),
                    "abs_max": float(np.max(np.abs(diff))),
                    "rmse": float(np.sqrt(np.mean(diff**2))),
                }

                # Count changed parameters (threshold: 1e-6)
                changed = int(np.sum(np.abs(diff) > 1e-6))
                comparison["changed_params"] = changed
                changed_params += changed
            else:
                comparison["shape_mismatch"] = True
                comparison["combined_similarity"] = 0.0

        layer_comparisons.append(comparison)

    # Summary statistics
    similarities = [
        lc["combined_similarity"]
        for lc in layer_comparisons
        if "combined_similarity" in lc
    ]

    summary = {
        "model_a_name": model_a_name,
        "model_b_name": model_b_name,
        "total_layers_a": len(weights_a),
        "total_layers_b": len(weights_b),
        "common_layers": len(common_names),
        "only_in_a": list(set(weights_a.keys()) - set(weights_b.keys())),
        "only_in_b": list(set(weights_b.keys()) - set(weights_a.keys())),
        "total_params_a": total_params_a,
        "total_params_b": total_params_b,
        "changed_params": changed_params,
        "overall_similarity": float(np.mean(similarities)) if similarities else 0.0,
        "min_similarity": float(np.min(similarities)) if similarities else 0.0,
        "max_similarity": float(np.max(similarities)) if similarities else 0.0,
    }

    # Identify most changed layers
    most_changed = sorted(
        [lc for lc in layer_comparisons if "combined_similarity" in lc],
        key=lambda x: x["combined_similarity"],
    )[:10]

    # Category breakdown
    categories = {
        "identical": [],  # similarity >= 0.9999
        "minor_changes": [],  # similarity >= 0.99
        "moderate_changes": [],  # similarity >= 0.9
        "major_changes": [],  # similarity < 0.9
    }

    for lc in layer_comparisons:
        if "combined_similarity" not in lc:
            continue
        sim = lc["combined_similarity"]
        if sim >= 0.9999:
            categories["identical"].append(lc["name"])
        elif sim >= 0.99:
            categories["minor_changes"].append(lc["name"])
        elif sim >= 0.9:
            categories["moderate_changes"].append(lc["name"])
        else:
            categories["major_changes"].append(lc["name"])

    return {
        "summary": summary,
        "layer_comparisons": layer_comparisons,
        "most_changed_layers": most_changed,
        "categories": categories,
        "category_counts": {k: len(v) for k, v in categories.items()},
    }


def format_model_diff(diff_data: dict[str, Any], verbose: bool = False) -> str:
    """
    Format model comparison data as a human-readable string.

    Args:
        diff_data: Output from compare_models()
        verbose: If True, include per-layer details

    Returns:
        Formatted string representation of the diff
    """
    lines = []
    summary = diff_data["summary"]

    # Header
    lines.append("=" * 60)
    model_a = summary["model_a_name"]
    model_b = summary["model_b_name"]
    lines.append(f"Model Comparison: {model_a} vs {model_b}")
    lines.append("=" * 60)
    lines.append("")

    # Summary stats
    lines.append("Summary:")
    lines.append(f"  Overall Similarity: {summary['overall_similarity']:.2%}")
    min_sim = summary["min_similarity"]
    max_sim = summary["max_similarity"]
    lines.append(f"  Similarity Range: {min_sim:.2%} - {max_sim:.2%}")
    lines.append(f"  Common Layers: {summary['common_layers']}")
    lines.append(f"  Parameters (A): {summary['total_params_a']:,}")
    lines.append(f"  Parameters (B): {summary['total_params_b']:,}")
    lines.append(f"  Changed Parameters: {summary['changed_params']:,}")
    lines.append("")

    # Layers only in one model
    if summary["only_in_a"]:
        lines.append(f"Layers only in {summary['model_a_name']}:")
        for name in summary["only_in_a"][:5]:
            lines.append(f"  - {name}")
        if len(summary["only_in_a"]) > 5:
            lines.append(f"  ... and {len(summary['only_in_a']) - 5} more")
        lines.append("")

    if summary["only_in_b"]:
        lines.append(f"Layers only in {summary['model_b_name']}:")
        for name in summary["only_in_b"][:5]:
            lines.append(f"  - {name}")
        if len(summary["only_in_b"]) > 5:
            lines.append(f"  ... and {len(summary['only_in_b']) - 5} more")
        lines.append("")

    # Category breakdown
    counts = diff_data["category_counts"]
    lines.append("Change Categories:")
    lines.append(f"  Identical (>99.99%):     {counts['identical']}")
    lines.append(f"  Minor Changes (>99%):    {counts['minor_changes']}")
    lines.append(f"  Moderate Changes (>90%): {counts['moderate_changes']}")
    lines.append(f"  Major Changes (<90%):    {counts['major_changes']}")
    lines.append("")

    # Most changed layers
    if diff_data["most_changed_layers"]:
        lines.append("Most Changed Layers:")
        for lc in diff_data["most_changed_layers"][:5]:
            sim = lc.get("combined_similarity", 0)
            lines.append(f"  {lc['name']}: {sim:.2%} similar")
        lines.append("")

    # Verbose: per-layer details
    if verbose:
        lines.append("Per-Layer Details:")
        lines.append("-" * 60)
        for lc in diff_data["layer_comparisons"]:
            if "combined_similarity" in lc:
                lines.append(f"{lc['name']}:")
                lines.append(f"  Shape: {lc.get('shape_a', 'N/A')}")
                cos_sim = lc["cosine_similarity"]
                mag_sim = lc["magnitude_similarity"]
                lines.append(f"  Cosine Similarity: {cos_sim:.4f}")
                lines.append(f"  Magnitude Similarity: {mag_sim:.4f}")
                if "diff_stats" in lc:
                    ds = lc["diff_stats"]
                    lines.append(f"  Diff RMSE: {ds['rmse']:.6f}")
                    lines.append(f"  Diff Max: {ds['abs_max']:.6f}")
                lines.append("")

    return "\n".join(lines)
