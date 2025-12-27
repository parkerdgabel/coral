"""Similarity computation utilities for weight tensors."""

from typing import Union

import numpy as np


def cosine_similarity(
    a: Union[np.ndarray, "np.floating"],
    b: Union[np.ndarray, "np.floating"],
) -> float:
    """Compute cosine similarity between two arrays.

    Note: Cosine similarity is scale-invariant, meaning [1,2,3] and [100,200,300]
    have similarity 1.0. For neural network weights where magnitude matters,
    use `weight_similarity()` instead.

    Args:
        a: First array (will be flattened)
        b: Second array (will be flattened)

    Returns:
        Cosine similarity value between -1 and 1.
        Returns 1.0 if both arrays are zero vectors.
        Returns 0.0 if exactly one array is a zero vector.
    """
    # Flatten arrays
    a_flat = np.asarray(a).flatten().astype(np.float64)
    b_flat = np.asarray(b).flatten().astype(np.float64)

    # Compute norms
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)

    # Handle zero vectors
    if norm_a == 0 and norm_b == 0:
        return 1.0
    if norm_a == 0 or norm_b == 0:
        return 0.0

    # Compute cosine similarity
    dot_product = np.dot(a_flat, b_flat)
    return float(dot_product / (norm_a * norm_b))


def magnitude_similarity(
    a: Union[np.ndarray, "np.floating"],
    b: Union[np.ndarray, "np.floating"],
) -> float:
    """Compute magnitude similarity between two arrays.

    Uses the ratio of smaller to larger norm, giving 1.0 for equal magnitudes
    and approaching 0.0 as magnitudes diverge.

    Args:
        a: First array (will be flattened)
        b: Second array (will be flattened)

    Returns:
        Magnitude similarity value between 0 and 1.
        Returns 1.0 if both have equal magnitude.
        Returns 0.0 if either is zero and the other is not.
    """
    a_flat = np.asarray(a).flatten().astype(np.float64)
    b_flat = np.asarray(b).flatten().astype(np.float64)

    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)

    # Handle zero vectors
    if norm_a == 0 and norm_b == 0:
        return 1.0
    if norm_a == 0 or norm_b == 0:
        return 0.0

    # Ratio of smaller to larger norm
    return float(min(norm_a, norm_b) / max(norm_a, norm_b))


def weight_similarity(
    a: Union[np.ndarray, "np.floating"],
    b: Union[np.ndarray, "np.floating"],
    direction_weight: float = 0.7,
    magnitude_weight: float = 0.3,
) -> float:
    """Compute similarity between neural network weights.

    This is a hybrid metric that considers both direction (cosine similarity)
    and magnitude. Unlike pure cosine similarity, this will correctly identify
    that [1,2,3] and [100,200,300] are NOT similar despite having the same
    direction.

    The default weights (0.7 direction, 0.3 magnitude) are tuned for typical
    neural network training where small magnitude changes are common but
    large scale differences indicate different models.

    Args:
        a: First weight array
        b: Second weight array
        direction_weight: Weight for directional (cosine) similarity [0-1]
        magnitude_weight: Weight for magnitude similarity [0-1]

    Returns:
        Combined similarity value between 0 and 1.

    Example:
        >>> a = np.array([1.0, 2.0, 3.0])
        >>> b = np.array([1.01, 2.01, 3.01])  # Small perturbation
        >>> weight_similarity(a, b)
        0.999...  # Very similar

        >>> c = np.array([100.0, 200.0, 300.0])  # Same direction, 100x scale
        >>> weight_similarity(a, c)
        0.71...  # Low similarity due to magnitude difference
    """
    if direction_weight + magnitude_weight == 0:
        raise ValueError("At least one weight must be non-zero")

    # Normalize weights
    total = direction_weight + magnitude_weight
    d_weight = direction_weight / total
    m_weight = magnitude_weight / total

    # Compute components
    cos_sim = cosine_similarity(a, b)
    mag_sim = magnitude_similarity(a, b)

    # Combine: cosine can be negative, so we map [-1, 1] to [0, 1]
    cos_normalized = (cos_sim + 1.0) / 2.0

    return float(d_weight * cos_normalized + m_weight * mag_sim)


def relative_difference(
    a: Union[np.ndarray, "np.floating"],
    b: Union[np.ndarray, "np.floating"],
) -> tuple[float, float, float]:
    """Compute relative difference statistics between two arrays.

    Useful for understanding how weights differ element-wise.

    Args:
        a: First array
        b: Second array

    Returns:
        Tuple of (mean_relative_diff, max_relative_diff, rmse)
        where relative diff is |a-b| / (|a| + |b| + eps)
    """
    a_flat = np.asarray(a).flatten().astype(np.float64)
    b_flat = np.asarray(b).flatten().astype(np.float64)

    eps = 1e-10
    diff = np.abs(a_flat - b_flat)
    scale = np.abs(a_flat) + np.abs(b_flat) + eps

    relative_diff = diff / scale

    mean_rel = float(np.mean(relative_diff))
    max_rel = float(np.max(relative_diff))
    rmse = float(np.sqrt(np.mean(diff**2)))

    return mean_rel, max_rel, rmse


def are_similar(
    a: np.ndarray,
    b: np.ndarray,
    threshold: float = 0.99,
    check_magnitude: bool = True,
    magnitude_tolerance: float = 0.1,
) -> bool:
    """Check if two arrays are similar for neural network weight deduplication.

    Args:
        a: First array
        b: Second array
        threshold: Cosine similarity threshold (0-1)
        check_magnitude: If True, also verify magnitudes are similar
        magnitude_tolerance: Maximum allowed relative magnitude difference (0-1)
            E.g., 0.1 means magnitudes must be within 10% of each other

    Returns:
        True if arrays are similar enough for deduplication
    """
    # Check directional similarity
    cos_sim = cosine_similarity(a, b)
    if cos_sim < threshold:
        return False

    # Optionally check magnitude similarity
    if check_magnitude:
        mag_sim = magnitude_similarity(a, b)
        # magnitude_tolerance of 0.1 means we need mag_sim >= 0.9
        if mag_sim < (1.0 - magnitude_tolerance):
            return False

    return True
