"""Similarity computation utilities for weight tensors."""

from typing import Union

import numpy as np


def cosine_similarity(
    a: Union[np.ndarray, "np.floating"],
    b: Union[np.ndarray, "np.floating"],
) -> float:
    """Compute cosine similarity between two arrays.

    Args:
        a: First array (will be flattened)
        b: Second array (will be flattened)

    Returns:
        Cosine similarity value between -1 and 1.
        Returns 1.0 if both arrays are zero vectors.
        Returns 0.0 if exactly one array is a zero vector.
    """
    # Flatten arrays
    a_flat = np.asarray(a).flatten()
    b_flat = np.asarray(b).flatten()

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


def are_similar(
    a: np.ndarray,
    b: np.ndarray,
    threshold: float = 0.99,
) -> bool:
    """Check if two arrays are similar based on cosine similarity.

    Args:
        a: First array
        b: Second array
        threshold: Similarity threshold (0-1)

    Returns:
        True if cosine similarity >= threshold
    """
    return cosine_similarity(a, b) >= threshold
