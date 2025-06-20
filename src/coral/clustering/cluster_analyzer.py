"""
Repository-wide weight analysis and clustering algorithms.

This module provides comprehensive clustering analysis for neural network weights
across entire repositories, supporting multiple clustering algorithms and 
optimization strategies.
"""

import logging
import threading
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import warnings

import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import xxhash

from .cluster_config import ClusteringConfig, OptimizationConfig
from .cluster_types import (
    ClusteringStrategy, ClusterLevel, ClusterMetrics, ClusterInfo,
    ClusterAssignment, Centroid
)
from ..core.weight_tensor import WeightTensor
from ..version_control.repository import Repository

logger = logging.getLogger(__name__)


@dataclass
class RepositoryAnalysis:
    """Analysis results for a repository's weight distribution."""
    
    total_weights: int = 0
    unique_weights: int = 0
    total_commits: int = 0
    total_branches: int = 0
    weight_shapes: Dict[Tuple[int, ...], int] = field(default_factory=dict)
    weight_dtypes: Dict[str, int] = field(default_factory=dict)
    layer_types: Dict[str, int] = field(default_factory=dict)
    size_distribution: Dict[str, int] = field(default_factory=dict)
    
    @property
    def deduplication_ratio(self) -> float:
        """Calculate deduplication ratio (proportion of duplicate weights)."""
        if self.total_weights == 0:
            return 0.0
        return (self.total_weights - self.unique_weights) / self.total_weights
    
    @property
    def avg_weights_per_commit(self) -> float:
        """Calculate average weights per commit."""
        if self.total_commits == 0:
            return 0.0
        return self.total_weights / self.total_commits


@dataclass
class ClusteringResult:
    """Results of a clustering operation."""
    
    assignments: List[ClusterAssignment]
    centroids: List[Centroid]
    metrics: ClusterMetrics
    strategy: ClusteringStrategy
    execution_time: float = 0.0
    memory_usage: Optional[int] = None
    
    def is_valid(self) -> bool:
        """Validate clustering result consistency."""
        # Check that all assignments reference valid clusters
        cluster_ids = {centroid.cluster_id for centroid in self.centroids}
        for assignment in self.assignments:
            if assignment.cluster_id not in cluster_ids:
                return False
        
        # Check metrics consistency
        if not self.metrics.is_valid():
            return False
        
        # Check cluster count consistency
        if len(self.centroids) != self.metrics.num_clusters:
            return False
        
        return True
    
    def get_cluster_sizes(self) -> Dict[str, int]:
        """Get size of each cluster."""
        cluster_sizes = Counter(assignment.cluster_id for assignment in self.assignments)
        return dict(cluster_sizes)


class ClusterAnalyzer:
    """
    Comprehensive repository-wide weight analysis and clustering.
    
    Provides advanced clustering algorithms for neural network weights with
    support for repository scanning, feature extraction, and quality assessment.
    """
    
    def __init__(
        self,
        repository: Repository,
        config: Optional[ClusteringConfig] = None,
        optimization_config: Optional[OptimizationConfig] = None
    ):
        """
        Initialize ClusterAnalyzer.
        
        Args:
            repository: Repository to analyze
            config: Clustering configuration
            optimization_config: Optimization configuration
        """
        self.repository = repository
        self.config = config or ClusteringConfig()
        self.optimization_config = optimization_config or OptimizationConfig()
        
        # Validate configurations
        if not self.config.validate():
            raise ValueError("Invalid clustering configuration")
        if not self.optimization_config.validate():
            raise ValueError("Invalid optimization configuration")
        
        # Internal state
        self._feature_cache: Dict[str, np.ndarray] = {}
        self._similarity_cache: Dict[str, Dict[str, float]] = {}
        self._clustering_cache: Dict[str, ClusteringResult] = {}
        self._lock = threading.RLock()
        
        # Initialize sklearn components
        self._scaler = StandardScaler() if self.config.normalize_features else None
        self._pca = None
        
        logger.info(f"Initialized ClusterAnalyzer with {self.config.strategy.value} strategy")

    def analyze_repository(
        self,
        branches: Optional[List[str]] = None,
        commits: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> RepositoryAnalysis:
        """
        Analyze repository weight distribution and characteristics.
        
        Args:
            branches: Specific branches to analyze (None = all branches)
            commits: Specific commits to analyze (None = all commits)
            progress_callback: Optional progress callback
            
        Returns:
            Repository analysis results
        """
        logger.info("Starting repository analysis")
        
        analysis = RepositoryAnalysis()
        weight_hashes = set()
        processed_weights = 0
        
        try:
            # Get commits to analyze - use available Repository methods
            if commits:
                target_commits = commits
            elif branches:
                # For now, use log to get commits from specific branches
                target_commits = []
                for branch in branches:
                    branch_commits = self.repository.log(max_commits=100, branch=branch)
                    target_commits.extend([c.commit_hash for c in branch_commits])
                target_commits = list(set(target_commits))  # Remove duplicates
            else:
                # Get commits from main log
                all_commits = self.repository.log(max_commits=1000)
                target_commits = [c.commit_hash for c in all_commits]
            
            analysis.total_commits = len(target_commits)
            analysis.total_branches = len(branches) if branches else 1  # Default to 1 if not specified
            
            # Get all weights from repository (simplified approach)
            try:
                all_weights = self.repository.get_all_weights()
                
                if progress_callback:
                    progress_callback(0.5)
                
                for weight_name, weight in all_weights.items():
                    processed_weights += 1
                    
                    # Track unique weights
                    weight_hash = weight.compute_hash()
                    if weight_hash not in weight_hashes:
                        weight_hashes.add(weight_hash)
                        analysis.unique_weights += 1
                    
                    # Analyze weight characteristics
                    analysis.weight_shapes[weight.shape] = analysis.weight_shapes.get(weight.shape, 0) + 1
                    analysis.weight_dtypes[str(weight.dtype)] = analysis.weight_dtypes.get(str(weight.dtype), 0) + 1
                    
                    if weight.metadata and weight.metadata.layer_type:
                        layer_type = weight.metadata.layer_type
                        analysis.layer_types[layer_type] = analysis.layer_types.get(layer_type, 0) + 1
                    
                    # Size distribution
                    size_category = self._categorize_weight_size(weight.nbytes)
                    analysis.size_distribution[size_category] = analysis.size_distribution.get(size_category, 0) + 1
                        
            except Exception as e:
                logger.warning(f"Failed to process repository weights: {e}")
            
            analysis.total_weights = processed_weights
            
            if progress_callback:
                progress_callback(1.0)
            
            logger.info(f"Repository analysis complete: {analysis.total_weights} weights, "
                       f"{analysis.unique_weights} unique, {analysis.deduplication_ratio:.2%} deduplication")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Repository analysis failed: {e}")
            raise

    def extract_features(
        self,
        weights: List[WeightTensor],
        method: Optional[str] = None,
        cache_key: Optional[str] = None
    ) -> np.ndarray:
        """
        Extract features from weight tensors for clustering.
        
        Args:
            weights: List of weight tensors
            method: Feature extraction method ("raw", "pca", "statistical", "hash")
            cache_key: Optional cache key for results
            
        Returns:
            Feature matrix (n_weights x n_features)
        """
        if not weights:
            raise ValueError("Cannot extract features from empty weight list")
        
        method = method or self.config.feature_extraction
        
        # Check cache first
        if cache_key and cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        logger.debug(f"Extracting features using {method} method for {len(weights)} weights")
        
        try:
            if method == "raw":
                features = self._extract_raw_features(weights)
            elif method == "pca":
                features = self._extract_pca_features(weights)
            elif method == "statistical":
                features = self._extract_statistical_features(weights)
            elif method == "hash":
                features = self._extract_hash_features(weights)
            else:
                raise ValueError(f"Unknown feature extraction method: {method}")
            
            # Normalize features if configured
            if self.config.normalize_features and self._scaler is not None:
                features = self._scaler.fit_transform(features)
            
            # Cache results
            if cache_key:
                with self._lock:
                    self._feature_cache[cache_key] = features
            
            logger.debug(f"Extracted features shape: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise

    def compute_similarity_matrix(
        self,
        weights: List[WeightTensor],
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix for weights.
        
        Args:
            weights: List of weight tensors
            metric: Similarity metric ("cosine", "euclidean", "correlation")
            
        Returns:
            Similarity matrix (n_weights x n_weights)
        """
        if len(weights) < 2:
            raise ValueError("Need at least 2 weights for similarity computation")
        
        # Generate cache key
        weight_hashes = [w.compute_hash() for w in weights]
        cache_key = f"{metric}_{hash(tuple(weight_hashes))}"
        
        if cache_key in self._similarity_cache:
            cached_data = self._similarity_cache[cache_key]
            return np.array([[cached_data.get(f"{i}_{j}", 0.0) for j in range(len(weights))] 
                           for i in range(len(weights))])
        
        logger.debug(f"Computing similarity matrix for {len(weights)} weights using {metric}")
        
        try:
            # Extract features for similarity computation
            features = self.extract_features(weights, method="raw")
            
            n_weights = len(weights)
            similarity_matrix = np.zeros((n_weights, n_weights))
            
            # Compute pairwise similarities
            for i in range(n_weights):
                for j in range(i, n_weights):
                    if i == j:
                        similarity = 1.0
                    else:
                        if metric == "cosine":
                            similarity = 1.0 - cosine(features[i], features[j])
                        elif metric == "euclidean":
                            # Convert distance to similarity
                            distance = euclidean(features[i], features[j])
                            similarity = 1.0 / (1.0 + distance)
                        elif metric == "correlation":
                            similarity = np.corrcoef(features[i], features[j])[0, 1]
                            similarity = np.nan_to_num(similarity, nan=0.0)
                        else:
                            raise ValueError(f"Unknown similarity metric: {metric}")
                    
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
            
            # Cache results
            with self._lock:
                cache_data = {}
                for i in range(n_weights):
                    for j in range(n_weights):
                        cache_data[f"{i}_{j}"] = similarity_matrix[i, j]
                self._similarity_cache[cache_key] = cache_data
            
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"Similarity matrix computation failed: {e}")
            raise

    def detect_natural_clusters(
        self,
        weights: List[WeightTensor],
        method: str = "elbow",
        max_k: int = 10
    ) -> int:
        """
        Detect natural number of clusters in weight data.
        
        Args:
            weights: List of weight tensors
            method: Detection method ("elbow", "silhouette", "gap")
            max_k: Maximum number of clusters to test
            
        Returns:
            Optimal number of clusters
        """
        if len(weights) < 2:
            return 1
        
        max_k = min(max_k, len(weights))
        if max_k < 2:
            return 1
        
        logger.debug(f"Detecting natural clusters using {method} method, max_k={max_k}")
        
        try:
            features = self.extract_features(weights)
            
            if method == "elbow":
                return self._detect_clusters_elbow(features, max_k)
            elif method == "silhouette":
                return self._detect_clusters_silhouette(features, max_k)
            elif method == "gap":
                return self._detect_clusters_gap(features, max_k)
            else:
                raise ValueError(f"Unknown cluster detection method: {method}")
                
        except Exception as e:
            logger.error(f"Natural cluster detection failed: {e}")
            # Fallback to simple heuristic
            return min(max_k // 2, len(weights) // 3) or 2

    def cluster_kmeans(
        self,
        weights: List[WeightTensor],
        k: int,
        config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
        cancel_event: Optional[threading.Event] = None
    ) -> ClusteringResult:
        """
        Perform K-means clustering on weights.
        
        Args:
            weights: List of weight tensors to cluster
            k: Number of clusters
            config: Additional K-means parameters
            progress_callback: Optional progress callback
            cancel_event: Optional cancellation event
            
        Returns:
            Clustering results
        """
        if not weights:
            raise ValueError("Cannot cluster empty weight list")
        if len(weights) < 2:
            raise ValueError("Need at least 2 weights for clustering")
        if k <= 0:
            raise ValueError("k must be positive")
        if k > len(weights):
            raise ValueError("k cannot be larger than number of weights")
        
        logger.info(f"Starting K-means clustering with k={k} for {len(weights)} weights")
        start_time = time.time()
        
        try:
            # Check for cancellation
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("Operation cancelled")
            
            # Extract features
            if progress_callback:
                progress_callback(0.1)
            features = self.extract_features(weights)
            
            # Configure K-means
            kmeans_config = {
                "n_clusters": k,
                "random_state": self.config.random_seed,
                "max_iter": self.config.max_iterations,
                "tol": self.config.convergence_tolerance,
                "n_init": 10
            }
            if config:
                kmeans_config.update(config)
            
            # Fit K-means
            if progress_callback:
                progress_callback(0.3)
            
            kmeans = KMeans(**kmeans_config)
            
            # Check for cancellation during fitting
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("Operation cancelled")
            
            try:
                cluster_labels = kmeans.fit_predict(features)
            except Exception as e:
                raise RuntimeError(f"Clustering failed: {e}")
            
            if progress_callback:
                progress_callback(0.7)
            
            # Create clustering result
            result = self._create_clustering_result(
                weights, cluster_labels, kmeans.cluster_centers_,
                ClusteringStrategy.KMEANS, start_time
            )
            
            if progress_callback:
                progress_callback(1.0)
            
            logger.info(f"K-means clustering completed in {result.execution_time:.2f}s, "
                       f"silhouette score: {result.metrics.silhouette_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"K-means clustering failed: {e}")
            raise

    def cluster_hierarchical(
        self,
        weights: List[WeightTensor],
        config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> ClusteringResult:
        """
        Perform hierarchical clustering on weights.
        
        Args:
            weights: List of weight tensors to cluster
            config: Hierarchical clustering parameters
            progress_callback: Optional progress callback
            
        Returns:
            Clustering results
        """
        if not weights:
            raise ValueError("Cannot cluster empty weight list")
        if len(weights) < 2:
            raise ValueError("Need at least 2 weights for clustering")
        
        logger.info(f"Starting hierarchical clustering for {len(weights)} weights")
        start_time = time.time()
        
        try:
            # Extract features
            if progress_callback:
                progress_callback(0.1)
            features = self.extract_features(weights)
            
            # Configure hierarchical clustering
            hierarchical_config = {
                "linkage": "ward",
                "distance_threshold": 0.5,
                "n_clusters": None
            }
            if config:
                hierarchical_config.update(config)
            
            # Fit hierarchical clustering
            if progress_callback:
                progress_callback(0.3)
            
            clustering = AgglomerativeClustering(**hierarchical_config)
            cluster_labels = clustering.fit_predict(features)
            
            if progress_callback:
                progress_callback(0.7)
            
            # Compute cluster centers
            cluster_centers = self._compute_cluster_centers(features, cluster_labels)
            
            # Create clustering result
            result = self._create_clustering_result(
                weights, cluster_labels, cluster_centers,
                ClusteringStrategy.HIERARCHICAL, start_time
            )
            
            if progress_callback:
                progress_callback(1.0)
            
            logger.info(f"Hierarchical clustering completed in {result.execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Hierarchical clustering failed: {e}")
            raise

    def cluster_dbscan(
        self,
        weights: List[WeightTensor],
        config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> ClusteringResult:
        """
        Perform DBSCAN clustering on weights.
        
        Args:
            weights: List of weight tensors to cluster
            config: DBSCAN parameters
            progress_callback: Optional progress callback
            
        Returns:
            Clustering results
        """
        if not weights:
            raise ValueError("Cannot cluster empty weight list")
        if len(weights) < 2:
            raise ValueError("Need at least 2 weights for clustering")
        
        logger.info(f"Starting DBSCAN clustering for {len(weights)} weights")
        start_time = time.time()
        
        try:
            # Extract features
            if progress_callback:
                progress_callback(0.1)
            features = self.extract_features(weights)
            
            # Configure DBSCAN
            dbscan_config = {
                "eps": 0.5,
                "min_samples": max(2, self.config.min_cluster_size)
            }
            if config:
                dbscan_config.update(config)
            
            # Fit DBSCAN
            if progress_callback:
                progress_callback(0.3)
            
            dbscan = DBSCAN(**dbscan_config)
            cluster_labels = dbscan.fit_predict(features)
            
            if progress_callback:
                progress_callback(0.7)
            
            # Handle noise points (label -1)
            unique_labels = set(cluster_labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            
            if not unique_labels:
                # All points are noise, create single cluster
                cluster_labels = np.zeros(len(weights), dtype=int)
                unique_labels = {0}
            
            # Compute cluster centers
            cluster_centers = self._compute_cluster_centers(features, cluster_labels)
            
            # Create clustering result
            result = self._create_clustering_result(
                weights, cluster_labels, cluster_centers,
                ClusteringStrategy.DBSCAN, start_time
            )
            
            if progress_callback:
                progress_callback(1.0)
            
            logger.info(f"DBSCAN clustering completed in {result.execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"DBSCAN clustering failed: {e}")
            raise

    def cluster_adaptive(
        self,
        weights: List[WeightTensor],
        config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> ClusteringResult:
        """
        Perform adaptive clustering with automatic strategy selection.
        
        Args:
            weights: List of weight tensors to cluster
            config: Additional parameters
            progress_callback: Optional progress callback
            
        Returns:
            Clustering results with best strategy
        """
        if not weights:
            raise ValueError("Cannot cluster empty weight list")
        if len(weights) < 2:
            raise ValueError("Need at least 2 weights for clustering")
        
        logger.info(f"Starting adaptive clustering for {len(weights)} weights")
        
        try:
            # Analyze data characteristics
            features = self.extract_features(weights)
            n_weights = len(weights)
            
            # Select best strategy based on data characteristics
            if n_weights <= 10:
                # Small dataset: use hierarchical
                strategy = ClusteringStrategy.HIERARCHICAL
            elif n_weights <= 100:
                # Medium dataset: try multiple strategies
                strategies = [ClusteringStrategy.KMEANS, ClusteringStrategy.HIERARCHICAL]
            else:
                # Large dataset: use K-means or DBSCAN
                strategies = [ClusteringStrategy.KMEANS, ClusteringStrategy.DBSCAN]
            
            if n_weights <= 10:
                # Single strategy for small datasets
                if strategy == ClusteringStrategy.HIERARCHICAL:
                    return self.cluster_hierarchical(weights, config, progress_callback)
            
            # Try multiple strategies and select best
            best_result = None
            best_score = -1.0
            
            for i, strategy in enumerate(strategies):
                try:
                    if progress_callback:
                        progress_callback(i / len(strategies))
                    
                    if strategy == ClusteringStrategy.KMEANS:
                        # Detect optimal k
                        k = self.detect_natural_clusters(weights, max_k=min(10, n_weights // 2))
                        result = self.cluster_kmeans(weights, k, config)
                    elif strategy == ClusteringStrategy.HIERARCHICAL:
                        result = self.cluster_hierarchical(weights, config)
                    elif strategy == ClusteringStrategy.DBSCAN:
                        result = self.cluster_dbscan(weights, config)
                    else:
                        continue
                    
                    # Evaluate result
                    score = self._evaluate_clustering_quality(result)
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                        
                except Exception as e:
                    logger.warning(f"Strategy {strategy.value} failed: {e}")
                    continue
            
            if best_result is None:
                # Fallback to simple K-means
                k = min(3, n_weights // 2) or 2
                best_result = self.cluster_kmeans(weights, k, config)
            
            if progress_callback:
                progress_callback(1.0)
            
            logger.info(f"Adaptive clustering selected {best_result.strategy.value} "
                       f"with score {best_score:.3f}")
            
            return best_result
            
        except Exception as e:
            logger.error(f"Adaptive clustering failed: {e}")
            raise

    def evaluate_clustering(
        self,
        assignments: List[ClusterAssignment],
        weights: List[WeightTensor]
    ) -> ClusterMetrics:
        """
        Evaluate clustering quality using multiple metrics.
        
        Args:
            assignments: Cluster assignments
            weights: Original weight tensors
            
        Returns:
            Clustering quality metrics
        """
        if not assignments:
            return ClusterMetrics()
        
        try:
            # Extract features and labels
            features = self.extract_features(weights)
            labels = [assignment.cluster_id for assignment in assignments]
            
            # Convert cluster IDs to numeric labels
            unique_clusters = list(set(labels))
            label_map = {cluster_id: i for i, cluster_id in enumerate(unique_clusters)}
            numeric_labels = [label_map[cluster_id] for cluster_id in labels]
            
            # Compute metrics
            metrics = ClusterMetrics()
            
            if len(unique_clusters) > 1:
                # Silhouette score (requires at least 2 clusters)
                try:
                    metrics.silhouette_score = silhouette_score(features, numeric_labels)
                except Exception:
                    metrics.silhouette_score = 0.0
                
                # Calinski-Harabasz score
                try:
                    metrics.calinski_harabasz_score = calinski_harabasz_score(features, numeric_labels)
                except Exception:
                    metrics.calinski_harabasz_score = 0.0
                
                # Davies-Bouldin score
                try:
                    metrics.davies_bouldin_score = davies_bouldin_score(features, numeric_labels)
                except Exception:
                    metrics.davies_bouldin_score = 0.0
            
            # Cluster statistics
            metrics.num_clusters = len(unique_clusters)
            cluster_sizes = Counter(labels)
            metrics.avg_cluster_size = sum(cluster_sizes.values()) / len(cluster_sizes)
            
            # Compression ratio estimate
            total_size = sum(w.nbytes for w in weights)
            unique_size = sum(w.nbytes for w in weights[:len(unique_clusters)])  # Approximation
            metrics.compression_ratio = 1.0 - (unique_size / total_size) if total_size > 0 else 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Clustering evaluation failed: {e}")
            return ClusterMetrics()

    def optimize_cluster_count(
        self,
        weights: List[WeightTensor],
        max_k: int = 10,
        method: str = "silhouette"
    ) -> int:
        """
        Find optimal number of clusters using various methods.
        
        Args:
            weights: List of weight tensors
            max_k: Maximum number of clusters to test
            method: Optimization method ("silhouette", "elbow", "gap")
            
        Returns:
            Optimal number of clusters
        """
        return self.detect_natural_clusters(weights, method, max_k)

    def validate_clusters(
        self,
        assignments: List[ClusterAssignment],
        min_cluster_size: Optional[int] = None
    ) -> bool:
        """
        Validate cluster assignments for consistency and quality.
        
        Args:
            assignments: Cluster assignments to validate
            min_cluster_size: Minimum required cluster size
            
        Returns:
            True if clusters are valid
        """
        if not assignments:
            return False
        
        min_size = min_cluster_size or self.config.min_cluster_size
        
        # Check cluster sizes
        cluster_sizes = Counter(assignment.cluster_id for assignment in assignments)
        
        for cluster_id, size in cluster_sizes.items():
            if size < min_size:
                logger.warning(f"Cluster {cluster_id} has size {size} < {min_size}")
                return False
        
        # Check for valid cluster IDs
        for assignment in assignments:
            if not assignment.cluster_id or not isinstance(assignment.cluster_id, str):
                return False
        
        # Check similarity scores
        for assignment in assignments:
            if not (0.0 <= assignment.similarity_score <= 1.0):
                return False
        
        return True

    # Private helper methods
    
    def _extract_raw_features(self, weights: List[WeightTensor]) -> np.ndarray:
        """Extract raw flattened features."""
        # Group weights by shape to handle different sizes
        shape_groups = defaultdict(list)
        for i, weight in enumerate(weights):
            shape_groups[weight.shape].append((i, weight))
        
        if len(shape_groups) == 1:
            # All weights have same shape - simple flattening
            features = np.array([w.data.flatten() for w in weights])
        else:
            # Different shapes - fall back to statistical features for consistency
            logger.debug(f"Found {len(shape_groups)} different shapes, using statistical features")
            features = self._extract_statistical_features(weights)
        
        return features.astype(np.float32)

    def _extract_pca_features(self, weights: List[WeightTensor]) -> np.ndarray:
        """Extract PCA-reduced features."""
        # First get raw features
        raw_features = self._extract_raw_features(weights)
        
        # Apply PCA
        n_components = self.config.dimensionality_reduction or min(50, raw_features.shape[1])
        n_components = min(n_components, raw_features.shape[0] - 1, raw_features.shape[1])
        
        if n_components <= 0:
            return raw_features
        
        if self._pca is None or self._pca.n_components != n_components:
            self._pca = PCA(n_components=n_components, random_state=self.config.random_seed)
        
        return self._pca.fit_transform(raw_features)

    def _extract_statistical_features(self, weights: List[WeightTensor]) -> np.ndarray:
        """Extract statistical features from weights."""
        features = []
        
        for weight in weights:
            data = weight.data.flatten()
            
            # Basic statistics
            weight_features = [
                np.mean(data),
                np.std(data),
                np.min(data),
                np.max(data),
                skew(data),
                kurtosis(data)
            ]
            
            features.append(weight_features)
        
        return np.array(features, dtype=np.float32)

    def _extract_hash_features(self, weights: List[WeightTensor]) -> np.ndarray:
        """Extract hash-based features."""
        features = []
        
        for weight in weights:
            # Create multiple hash features
            data_bytes = weight.data.tobytes()
            
            # Different hash functions for different aspects
            hash1 = xxhash.xxh32(data_bytes, seed=0).intdigest()
            hash2 = xxhash.xxh32(data_bytes, seed=1).intdigest()
            hash3 = xxhash.xxh64(data_bytes, seed=0).intdigest()
            hash4 = xxhash.xxh64(data_bytes, seed=1).intdigest()
            
            # Normalize to [0, 1] range
            weight_features = [
                (hash1 % 1000000) / 1000000.0,
                (hash2 % 1000000) / 1000000.0,
                (hash3 % 1000000) / 1000000.0,
                (hash4 % 1000000) / 1000000.0,
            ]
            
            features.append(weight_features)
        
        return np.array(features, dtype=np.float32)

    def _detect_clusters_elbow(self, features: np.ndarray, max_k: int) -> int:
        """Detect optimal clusters using elbow method."""
        inertias = []
        k_range = range(1, min(max_k + 1, len(features)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.config.random_seed, n_init=10)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point
        if len(inertias) < 3:
            return min(2, len(features) - 1)
        
        # Simple elbow detection: find maximum change in slope
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        
        if len(second_diffs) > 0:
            elbow_idx = np.argmax(second_diffs) + 2  # +2 because of double diff
            return min(elbow_idx, max_k)
        
        return 2

    def _detect_clusters_silhouette(self, features: np.ndarray, max_k: int) -> int:
        """Detect optimal clusters using silhouette analysis."""
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(features)))  # Silhouette needs at least 2 clusters
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.config.random_seed, n_init=10)
            labels = kmeans.fit_predict(features)
            
            try:
                score = silhouette_score(features, labels)
                silhouette_scores.append(score)
            except Exception:
                silhouette_scores.append(0.0)
        
        if not silhouette_scores:
            return 2
        
        # Return k with highest silhouette score
        best_idx = np.argmax(silhouette_scores)
        return best_idx + 2  # +2 because range starts at 2

    def _detect_clusters_gap(self, features: np.ndarray, max_k: int) -> int:
        """Detect optimal clusters using gap statistic."""
        # Simplified gap statistic implementation
        # For production use, consider more sophisticated implementation
        
        inertias = []
        k_range = range(1, min(max_k + 1, len(features)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.config.random_seed, n_init=10)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
        
        # Generate reference data
        n_refs = 10
        ref_inertias = []
        
        for _ in range(n_refs):
            # Create random reference data with same bounds
            ref_data = np.random.uniform(
                low=features.min(axis=0),
                high=features.max(axis=0),
                size=features.shape
            )
            
            ref_inertia = []
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=self.config.random_seed, n_init=10)
                kmeans.fit(ref_data)
                ref_inertia.append(kmeans.inertia_)
            
            ref_inertias.append(ref_inertia)
        
        # Compute gap statistic
        ref_inertias = np.array(ref_inertias)
        ref_means = np.mean(ref_inertias, axis=0)
        
        gaps = np.log(ref_means) - np.log(inertias)
        
        # Find first k where gap(k) >= gap(k+1) - s(k+1)
        # Simplified: just return k with maximum gap
        if len(gaps) > 0:
            return np.argmax(gaps) + 1
        
        return 2

    def _compute_cluster_centers(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute cluster centers from features and labels."""
        unique_labels = np.unique(labels)
        centers = []
        
        for label in unique_labels:
            if label == -1:  # Skip noise points in DBSCAN
                continue
            
            mask = labels == label
            if np.any(mask):
                center = np.mean(features[mask], axis=0)
                centers.append(center)
        
        return np.array(centers) if centers else np.array([np.mean(features, axis=0)])

    def _create_clustering_result(
        self,
        weights: List[WeightTensor],
        labels: np.ndarray,
        centers: np.ndarray,
        strategy: ClusteringStrategy,
        start_time: float
    ) -> ClusteringResult:
        """Create ClusteringResult from clustering output."""
        execution_time = time.time() - start_time
        
        # Create assignments
        assignments = []
        for i, (weight, label) in enumerate(zip(weights, labels)):
            if label == -1:  # Noise point in DBSCAN
                cluster_id = "noise"
                distance = 0.0
                similarity = 0.0
            else:
                cluster_id = f"cluster_{label}"
                
                # Compute distance to centroid
                if label < len(centers):
                    try:
                        # Get the features used for clustering to ensure consistency
                        all_features = self.extract_features(weights)
                        if i < len(all_features):
                            distance = euclidean(all_features[i], centers[label])
                            similarity = 1.0 / (1.0 + distance)
                        else:
                            distance = 0.0
                            similarity = 1.0
                    except Exception:
                        # Fallback if distance computation fails
                        distance = 0.0
                        similarity = 1.0
                else:
                    distance = 0.0
                    similarity = 1.0
            
            assignment = ClusterAssignment(
                weight_name=weight.metadata.name if weight.metadata else f"weight_{i}",
                weight_hash=weight.compute_hash(),
                cluster_id=cluster_id,
                distance_to_centroid=distance,
                similarity_score=similarity
            )
            assignments.append(assignment)
        
        # Create centroids
        centroids = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            
            cluster_id = f"cluster_{label}"
            
            # Find weights in this cluster
            cluster_weights = [w for w, l in zip(weights, labels) if l == label]
            
            if cluster_weights and label < len(centers):
                # Create centroid - use actual weight centroid rather than feature centroid
                reference_weight = cluster_weights[0]
                
                # Compute actual weight centroid from cluster members
                if len(cluster_weights) > 1:
                    # Compute mean of actual weight data
                    weight_arrays = [w.data for w in cluster_weights]
                    if all(w.shape == reference_weight.shape for w in cluster_weights):
                        centroid_data = np.mean(weight_arrays, axis=0)
                    else:
                        # Mixed shapes - use first weight as representative
                        centroid_data = cluster_weights[0].data
                else:
                    centroid_data = cluster_weights[0].data
                
                centroid = Centroid(
                    data=centroid_data,
                    cluster_id=cluster_id,
                    shape=reference_weight.shape,
                    dtype=reference_weight.dtype
                )
                centroids.append(centroid)
        
        # Compute metrics
        metrics = self.evaluate_clustering(assignments, weights)
        
        return ClusteringResult(
            assignments=assignments,
            centroids=centroids,
            metrics=metrics,
            strategy=strategy,
            execution_time=execution_time
        )

    def _evaluate_clustering_quality(self, result: ClusteringResult) -> float:
        """Evaluate overall clustering quality for strategy selection."""
        # Weighted combination of metrics
        quality_score = 0.0
        
        # Silhouette score (higher is better)
        if result.metrics.silhouette_score > 0:
            quality_score += 0.4 * result.metrics.silhouette_score
        
        # Compression ratio (higher is better)
        quality_score += 0.3 * result.metrics.compression_ratio
        
        # Cluster balance (prefer balanced clusters)
        cluster_sizes = Counter(a.cluster_id for a in result.assignments)
        if cluster_sizes:
            sizes = list(cluster_sizes.values())
            balance_score = 1.0 - (np.std(sizes) / np.mean(sizes))
            quality_score += 0.2 * max(0, balance_score)
        
        # Execution time penalty (faster is better)
        if result.execution_time > 0:
            time_score = 1.0 / (1.0 + result.execution_time)
            quality_score += 0.1 * time_score
        
        return quality_score

    def _categorize_weight_size(self, nbytes: int) -> str:
        """Categorize weight by size."""
        if nbytes < 1024:  # < 1KB
            return "tiny"
        elif nbytes < 1024 * 1024:  # < 1MB
            return "small"
        elif nbytes < 10 * 1024 * 1024:  # < 10MB
            return "medium"
        elif nbytes < 100 * 1024 * 1024:  # < 100MB
            return "large"
        else:
            return "huge"