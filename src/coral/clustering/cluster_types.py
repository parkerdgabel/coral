"""Core data structures and enums for clustering-based deduplication."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xxhash

from ..core.weight_tensor import WeightTensor

logger = logging.getLogger(__name__)


class ClusteringStrategy(Enum):
    """Clustering algorithms supported for weight deduplication."""

    KMEANS = "kmeans"  # K-means clustering
    HIERARCHICAL = "hierarchical"  # Hierarchical/agglomerative clustering
    DBSCAN = "dbscan"  # Density-based clustering
    ADAPTIVE = "adaptive"  # Adaptive strategy selection based on data characteristics


class ClusterLevel(Enum):
    """Hierarchical levels for clustering weights."""

    TENSOR = "tensor"  # Individual weight tensors
    BLOCK = "block"  # Weight blocks (e.g., attention heads)
    LAYER = "layer"  # Full layers
    MODEL = "model"  # Entire models


@dataclass
class ClusterMetrics:
    """Quality and performance metrics for clustering results."""

    silhouette_score: float = 0.0  # Clustering quality [-1, 1]
    inertia: float = 0.0  # Within-cluster sum of squares
    calinski_harabasz_score: float = 0.0  # Variance ratio criterion
    davies_bouldin_score: float = 0.0  # Average similarity between clusters
    num_clusters: int = 0  # Number of clusters formed
    avg_cluster_size: float = 0.0  # Average number of weights per cluster
    compression_ratio: float = 0.0  # Space savings achieved [0, 1]

    def is_valid(self) -> bool:
        """Validate metric values are within expected ranges."""
        if not (-1.0 <= self.silhouette_score <= 1.0):
            return False
        if not (0.0 <= self.compression_ratio <= 1.0):
            return False
        if self.num_clusters < 0:
            return False
        if self.avg_cluster_size < 0:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "silhouette_score": self.silhouette_score,
            "inertia": self.inertia,
            "calinski_harabasz_score": self.calinski_harabasz_score,
            "davies_bouldin_score": self.davies_bouldin_score,
            "num_clusters": self.num_clusters,
            "avg_cluster_size": self.avg_cluster_size,
            "compression_ratio": self.compression_ratio,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterMetrics":
        """Create from dictionary."""
        return cls(
            silhouette_score=data.get("silhouette_score", 0.0),
            inertia=data.get("inertia", 0.0),
            calinski_harabasz_score=data.get("calinski_harabasz_score", 0.0),
            davies_bouldin_score=data.get("davies_bouldin_score", 0.0),
            num_clusters=data.get("num_clusters", 0),
            avg_cluster_size=data.get("avg_cluster_size", 0.0),
            compression_ratio=data.get("compression_ratio", 0.0),
        )


@dataclass
class ClusterInfo:
    """Metadata about a cluster of weights."""

    cluster_id: str  # Unique identifier for the cluster
    strategy: ClusteringStrategy  # Algorithm used to create cluster
    level: ClusterLevel = ClusterLevel.TENSOR  # Hierarchical level
    member_count: int = 0  # Number of weights in cluster
    centroid_hash: Optional[str] = None  # Hash of cluster centroid
    created_at: Optional[str] = None  # Timestamp of cluster creation
    parent_cluster_id: Optional[str] = None  # Parent in hierarchy
    child_cluster_ids: Optional[List[str]] = None  # Children in hierarchy

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cluster_id": self.cluster_id,
            "strategy": self.strategy.value,
            "level": self.level.value,
            "member_count": self.member_count,
            "centroid_hash": self.centroid_hash,
            "created_at": self.created_at,
            "parent_cluster_id": self.parent_cluster_id,
            "child_cluster_ids": self.child_cluster_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterInfo":
        """Create from dictionary."""
        return cls(
            cluster_id=data["cluster_id"],
            strategy=ClusteringStrategy(data["strategy"]),
            level=ClusterLevel(data["level"]),
            member_count=data.get("member_count", 0),
            centroid_hash=data.get("centroid_hash"),
            created_at=data.get("created_at"),
            parent_cluster_id=data.get("parent_cluster_id"),
            child_cluster_ids=data.get("child_cluster_ids"),
        )


@dataclass
class Centroid:
    """Represents the center of a cluster of weights."""

    data: np.ndarray  # Centroid weight data
    cluster_id: str  # Associated cluster identifier
    shape: Tuple[int, ...]  # Shape of the centroid tensor
    dtype: np.dtype  # Data type of the centroid
    hash: Optional[str] = None  # Content hash of centroid data

    def compute_hash(self, force: bool = False) -> str:
        """
        Compute content-based hash of the centroid.

        Args:
            force: If True, recompute hash even if cached

        Returns:
            Hexadecimal hash string
        """
        if self.hash is not None and not force:
            return self.hash

        # Use xxhash for fast hashing, similar to WeightTensor
        hasher = xxhash.xxh3_64()

        # Include shape and dtype in hash
        normalized_shape = tuple(int(dim) for dim in self.shape)
        normalized_dtype = np.dtype(self.dtype).name
        hasher.update(str(normalized_shape).encode())
        hasher.update(normalized_dtype.encode())

        # Hash the centroid data
        hasher.update(self.data.tobytes())

        self.hash = hasher.hexdigest()
        return self.hash

    @classmethod
    def from_weights(cls, weights: List[WeightTensor], cluster_id: str) -> "Centroid":
        """
        Create centroid by computing mean of weight tensors.

        Args:
            weights: List of weight tensors in the cluster
            cluster_id: Identifier for the cluster

        Returns:
            Centroid representing the cluster center
        """
        if not weights:
            raise ValueError("Cannot create centroid from empty weight list")

        # Verify all weights have same shape and dtype
        reference_shape = weights[0].shape
        reference_dtype = weights[0].dtype

        for i, weight in enumerate(weights):
            if weight.shape != reference_shape:
                weight_name = weight.metadata.name if weight.metadata else f"weight_{i}"
                raise ValueError(
                    f"Shape compatibility error for weight '{weight_name}': "
                    f"expected {reference_shape}, got {weight.shape}. "
                    f"All weights in a cluster must have identical shapes."
                )
            if weight.dtype != reference_dtype:
                weight_name = weight.metadata.name if weight.metadata else f"weight_{i}"
                raise ValueError(
                    f"Dtype compatibility error for weight '{weight_name}': "
                    f"expected {reference_dtype}, got {weight.dtype}. "
                    f"All weights in a cluster must have identical dtypes."
                )

        # Compute mean of all weights with numerical stability
        try:
            weight_data = np.stack([w.data for w in weights], axis=0)
            
            # Handle special values
            if np.any(np.isnan(weight_data)):
                logger.warning(f"NaN values detected in weights for cluster {cluster_id}")
                weight_data = np.nan_to_num(weight_data, nan=0.0)
            
            if np.any(np.isinf(weight_data)):
                logger.warning(f"Infinite values detected in weights for cluster {cluster_id}")
                weight_data = np.clip(weight_data, -1e20, 1e20)
            
            centroid_data = np.mean(weight_data, axis=0)
            
            # Ensure centroid has same dtype as input weights
            if centroid_data.dtype != reference_dtype:
                centroid_data = centroid_data.astype(reference_dtype)
                
        except Exception as e:
            logger.error(f"Failed to compute centroid for cluster {cluster_id}: {e}")
            # Fallback: use first weight as centroid
            centroid_data = weights[0].data.copy()

        return cls(
            data=centroid_data,
            cluster_id=cluster_id,
            shape=reference_shape,
            dtype=reference_dtype,
        )
    
    @classmethod
    def validate_compatibility(cls, weights: List[WeightTensor]) -> bool:
        """
        Validate that all weights are compatible for clustering.
        
        Args:
            weights: List of weight tensors to validate
            
        Returns:
            True if all weights are compatible, False otherwise
        """
        if not weights:
            return False
        
        if len(weights) == 1:
            return True
            
        reference_shape = weights[0].shape
        reference_dtype = weights[0].dtype
        
        for weight in weights[1:]:
            if weight.shape != reference_shape or weight.dtype != reference_dtype:
                return False
                
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data_bytes": self.data.tobytes(),
            "data_shape": list(self.shape),
            "data_dtype": np.dtype(self.dtype).name,
            "cluster_id": self.cluster_id,
            "shape": list(self.shape),
            "dtype": np.dtype(self.dtype).name,
            "hash": self.compute_hash(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Centroid":
        """Create from dictionary."""
        # Reconstruct data array
        data_bytes = data["data_bytes"]
        data_dtype = np.dtype(data["data_dtype"])
        data_shape = tuple(data["data_shape"])

        centroid_data = np.frombuffer(data_bytes, dtype=data_dtype).reshape(data_shape)

        return cls(
            data=centroid_data,
            cluster_id=data["cluster_id"],
            shape=tuple(data["shape"]),
            dtype=np.dtype(data["dtype"]),
            hash=data.get("hash"),
        )

    @property
    def nbytes(self) -> int:
        """Get the number of bytes used by the centroid."""
        return self.data.nbytes


@dataclass
class ClusterAssignment:
    """Represents assignment of a weight to a cluster."""

    weight_name: str  # Name/identifier of the weight
    weight_hash: str  # Content hash of the weight
    cluster_id: str  # Assigned cluster identifier
    distance_to_centroid: float = 0.0  # Distance from weight to cluster centroid
    similarity_score: float = 0.0  # Similarity to cluster centroid [0, 1]
    is_representative: bool = False  # Whether this weight represents the cluster
    delta_hash: Optional[str] = None  # Hash of delta encoding for this weight

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "weight_name": self.weight_name,
            "weight_hash": self.weight_hash,
            "cluster_id": self.cluster_id,
            "distance_to_centroid": self.distance_to_centroid,
            "similarity_score": self.similarity_score,
            "is_representative": self.is_representative,
            "delta_hash": self.delta_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterAssignment":
        """Create from dictionary."""
        return cls(
            weight_name=data["weight_name"],
            weight_hash=data["weight_hash"],
            cluster_id=data["cluster_id"],
            distance_to_centroid=data.get("distance_to_centroid", 0.0),
            similarity_score=data.get("similarity_score", 0.0),
            is_representative=data.get("is_representative", False),
            delta_hash=data.get("delta_hash"),
        )