"""Product Quantization implementation for efficient vector compression.

This module implements Product Quantization (PQ), a technique for compressing
high-dimensional vectors by splitting them into subvectors and quantizing each
independently. Supports both lossy and lossless modes with residual encoding.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Literal
import numpy as np
from sklearn.cluster import KMeans
import warnings


@dataclass
class PQConfig:
    """Configuration for Product Quantization.
    
    Attributes:
        num_subvectors: Number of subvectors (M) to split each vector into
        bits_per_subvector: Bits used for each subvector (K = 2^bits codewords)
        codebook_init: Initialization method for codebooks ("kmeans++" or "random")
        max_iterations: Maximum iterations for K-means training
        use_residual: Whether to store residuals for lossless reconstruction
        residual_quantization: Optional bits for quantizing residuals (None = store full precision)
    """
    num_subvectors: int = 8
    bits_per_subvector: int = 8
    codebook_init: Literal["k-means++", "random"] = "k-means++"
    max_iterations: int = 20
    use_residual: bool = True
    residual_quantization: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_subvectors < 1:
            raise ValueError(f"num_subvectors must be >= 1, got {self.num_subvectors}")
        if self.bits_per_subvector < 1 or self.bits_per_subvector > 16:
            raise ValueError(f"bits_per_subvector must be in [1, 16], got {self.bits_per_subvector}")
        if self.residual_quantization is not None and self.residual_quantization < 1:
            raise ValueError(f"residual_quantization must be >= 1, got {self.residual_quantization}")
    
    @property
    def num_codewords(self) -> int:
        """Number of codewords per codebook (K = 2^bits)."""
        return 2 ** self.bits_per_subvector


class PQCodebook:
    """Product Quantization codebook storage and serialization.
    
    Attributes:
        subvector_size: Dimension of each subvector
        codebooks: List of M codebooks, each K x subvector_size
        version: Version string for compatibility
        training_stats: Dictionary of training metrics
    """
    
    def __init__(
        self,
        subvector_size: int,
        codebooks: List[np.ndarray],
        version: str = "1.0",
        training_stats: Optional[Dict] = None
    ):
        """Initialize PQ codebook.
        
        Args:
            subvector_size: Dimension of each subvector
            codebooks: List of M codebooks, each K x subvector_size
            version: Version string for compatibility
            training_stats: Optional training metrics
        """
        self.subvector_size = subvector_size
        self.codebooks = codebooks
        self.version = version
        self.training_stats = training_stats or {}
        
        # Validate codebooks
        for i, cb in enumerate(codebooks):
            if cb.shape[1] != subvector_size:
                raise ValueError(
                    f"Codebook {i} has wrong subvector size: "
                    f"expected {subvector_size}, got {cb.shape[1]}"
                )
    
    def to_dict(self) -> Dict:
        """Serialize codebook to dictionary.
        
        Returns:
            Dictionary representation suitable for storage
        """
        return {
            "subvector_size": self.subvector_size,
            "codebooks": [cb.tolist() for cb in self.codebooks],
            "version": self.version,
            "training_stats": self.training_stats,
            "num_subvectors": len(self.codebooks),
            "num_codewords": self.codebooks[0].shape[0] if self.codebooks else 0
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PQCodebook":
        """Deserialize codebook from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            PQCodebook instance
        """
        codebooks = [np.array(cb, dtype=np.float32) for cb in data["codebooks"]]
        return cls(
            subvector_size=data["subvector_size"],
            codebooks=codebooks,
            version=data.get("version", "1.0"),
            training_stats=data.get("training_stats", {})
        )


def split_vector(vector: np.ndarray, num_subvectors: int) -> List[np.ndarray]:
    """Split vector into subvectors.
    
    Args:
        vector: Input vector to split
        num_subvectors: Number of subvectors to create
        
    Returns:
        List of subvectors
        
    Raises:
        ValueError: If vector cannot be evenly split
    """
    if len(vector) == 0:
        raise ValueError("Cannot split empty vector")
    
    vector_dim = len(vector)
    
    # Handle case where vector dimension is not divisible by M
    if vector_dim % num_subvectors != 0:
        # Pad vector to make it divisible
        pad_size = num_subvectors - (vector_dim % num_subvectors)
        vector = np.pad(vector, (0, pad_size), mode='constant', constant_values=0)
        warnings.warn(
            f"Vector dimension {vector_dim} not divisible by {num_subvectors}. "
            f"Padding with {pad_size} zeros."
        )
    
    subvector_size = len(vector) // num_subvectors
    subvectors = []
    
    for i in range(num_subvectors):
        start = i * subvector_size
        end = start + subvector_size
        subvectors.append(vector[start:end])
    
    return subvectors


def train_codebooks(
    vectors: np.ndarray,
    config: PQConfig
) -> PQCodebook:
    """Train PQ codebooks using K-means clustering.
    
    Args:
        vectors: Training vectors, shape (n_vectors, dim)
        config: PQ configuration
        
    Returns:
        Trained PQCodebook
        
    Raises:
        ValueError: If vectors are invalid or empty
    """
    if len(vectors) == 0:
        raise ValueError("Cannot train codebooks on empty vectors")
    
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    
    n_vectors, vector_dim = vectors.shape
    
    # Determine subvector size
    if vector_dim % config.num_subvectors != 0:
        # Pad vectors if needed
        pad_size = config.num_subvectors - (vector_dim % config.num_subvectors)
        vectors = np.pad(vectors, ((0, 0), (0, pad_size)), mode='constant')
        vector_dim = vectors.shape[1]
    
    subvector_size = vector_dim // config.num_subvectors
    
    # Split all vectors into subvectors
    all_subvectors = []
    for m in range(config.num_subvectors):
        start = m * subvector_size
        end = start + subvector_size
        all_subvectors.append(vectors[:, start:end])
    
    # Train codebook for each subvector position
    codebooks = []
    training_stats = {
        "inertias": [],
        "n_iterations": [],
        "n_vectors": n_vectors,
        "vector_dim": vector_dim
    }
    
    for m, subvecs in enumerate(all_subvectors):
        # Use fewer clusters if we have fewer unique subvectors
        n_unique = len(np.unique(subvecs, axis=0))
        n_clusters = min(config.num_codewords, n_unique, n_vectors)
        
        if n_clusters < config.num_codewords:
            warnings.warn(
                f"Subvector {m}: Using {n_clusters} clusters instead of "
                f"{config.num_codewords} due to limited unique vectors"
            )
        
        # Train K-means
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=config.codebook_init,
            max_iter=config.max_iterations,
            n_init=1,
            random_state=42
        )
        kmeans.fit(subvecs)
        
        # Pad codebook if needed
        if n_clusters < config.num_codewords:
            # Duplicate existing centroids to fill remaining slots
            centroids = kmeans.cluster_centers_
            while len(centroids) < config.num_codewords:
                n_to_add = min(n_clusters, config.num_codewords - len(centroids))
                centroids = np.vstack([centroids, centroids[:n_to_add]])
            codebooks.append(centroids)
        else:
            codebooks.append(kmeans.cluster_centers_)
        
        training_stats["inertias"].append(float(kmeans.inertia_))
        training_stats["n_iterations"].append(kmeans.n_iter_)
    
    return PQCodebook(
        subvector_size=subvector_size,
        codebooks=codebooks,
        training_stats=training_stats
    )


def encode_vector(
    vector: np.ndarray,
    codebook: PQCodebook,
    config: PQConfig
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Encode vector using PQ codebooks.
    
    Args:
        vector: Vector to encode
        codebook: Trained PQ codebook
        config: PQ configuration
        
    Returns:
        Tuple of (indices, residual):
            - indices: Array of subvector indices, shape (M,)
            - residual: Optional residual for lossless reconstruction
    """
    if len(vector) == 0:
        raise ValueError("Cannot encode empty vector")
    
    # Split vector into subvectors
    subvectors = split_vector(vector, config.num_subvectors)
    
    # Find nearest codeword for each subvector
    indices = np.zeros(config.num_subvectors, dtype=np.uint32)
    reconstructed = np.zeros_like(vector)
    
    for m, (subvec, cb) in enumerate(zip(subvectors, codebook.codebooks)):
        # Compute distances to all codewords
        distances = np.sum((cb - subvec) ** 2, axis=1)
        nearest_idx = np.argmin(distances)
        indices[m] = nearest_idx
        
        # Store reconstruction
        start = m * codebook.subvector_size
        end = start + codebook.subvector_size
        reconstructed[start:end] = cb[nearest_idx]
    
    # Compute residual if needed
    residual = None
    if config.use_residual:
        # Trim reconstruction to original vector size if padded
        if len(reconstructed) > len(vector):
            reconstructed = reconstructed[:len(vector)]
        
        residual = vector - reconstructed
        
        # Quantize residual if specified
        if config.residual_quantization is not None:
            residual = quantize_residual(residual, config.residual_quantization)
    
    return indices, residual


def decode_vector(
    indices: np.ndarray,
    codebook: PQCodebook,
    residual: Optional[np.ndarray] = None,
    original_dim: Optional[int] = None
) -> np.ndarray:
    """Reconstruct vector from PQ indices and optional residual.
    
    Args:
        indices: Subvector indices, shape (M,)
        codebook: PQ codebook used for encoding
        residual: Optional residual for lossless reconstruction
        original_dim: Original vector dimension (for handling padding)
        
    Returns:
        Reconstructed vector
    """
    if len(indices) != len(codebook.codebooks):
        raise ValueError(
            f"Number of indices ({len(indices)}) doesn't match "
            f"number of codebooks ({len(codebook.codebooks)})"
        )
    
    # Reconstruct from codebooks
    reconstructed = []
    for m, (idx, cb) in enumerate(zip(indices, codebook.codebooks)):
        if idx >= len(cb):
            raise ValueError(
                f"Index {idx} out of range for codebook {m} "
                f"with {len(cb)} codewords"
            )
        reconstructed.append(cb[idx])
    
    reconstructed = np.concatenate(reconstructed)
    
    # Trim to original dimension if specified
    if original_dim is not None and len(reconstructed) > original_dim:
        reconstructed = reconstructed[:original_dim]
    
    # Add residual if provided
    if residual is not None:
        if len(residual) != len(reconstructed):
            raise ValueError(
                f"Residual dimension ({len(residual)}) doesn't match "
                f"reconstructed dimension ({len(reconstructed)})"
            )
        reconstructed = reconstructed + residual
    
    return reconstructed


def compute_asymmetric_distance(
    vector: np.ndarray,
    indices: np.ndarray,
    codebook: PQCodebook
) -> float:
    """Compute asymmetric distance between vector and PQ-encoded vector.
    
    This is useful for similarity search where the query is not quantized
    but database vectors are PQ-encoded.
    
    Args:
        vector: Query vector (not quantized)
        indices: PQ indices of database vector
        codebook: PQ codebook
        
    Returns:
        Squared Euclidean distance
    """
    if len(vector) == 0:
        raise ValueError("Cannot compute distance for empty vector")
    
    # Split query vector
    subvectors = split_vector(vector, len(codebook.codebooks))
    
    # Compute distance for each subvector
    distance = 0.0
    for m, (subvec, idx, cb) in enumerate(zip(subvectors, indices, codebook.codebooks)):
        if idx >= len(cb):
            raise ValueError(
                f"Index {idx} out of range for codebook {m} "
                f"with {len(cb)} codewords"
            )
        distance += np.sum((subvec - cb[idx]) ** 2)
    
    return distance


def quantize_residual(residual: np.ndarray, bits: int) -> np.ndarray:
    """Quantize residual vector to specified bit precision.
    
    Args:
        residual: Residual vector to quantize
        bits: Number of bits for quantization
        
    Returns:
        Quantized residual
    """
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    
    # Find range for quantization
    min_val = np.min(residual)
    max_val = np.max(residual)
    
    # Handle edge case of uniform residual
    if min_val == max_val:
        return np.zeros_like(residual)
    
    # Quantize to specified bits
    n_levels = 2 ** bits
    scale = (max_val - min_val) / (n_levels - 1)
    
    # Quantize and dequantize
    quantized = np.round((residual - min_val) / scale)
    quantized = np.clip(quantized, 0, n_levels - 1)
    dequantized = quantized * scale + min_val
    
    return dequantized