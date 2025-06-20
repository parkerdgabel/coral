"""
Adaptive cluster optimization and re-clustering component.

This module provides comprehensive optimization strategies for clustering-based
weight deduplication, including multi-objective optimization, constraint handling,
and continuous monitoring capabilities.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import warnings

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples

from .cluster_analyzer import ClusterAnalyzer, ClusteringResult
from .cluster_config import ClusteringConfig, OptimizationConfig
from .cluster_types import (
    ClusteringStrategy, ClusterMetrics, ClusterAssignment,
    Centroid, ClusterInfo
)
from ..core.weight_tensor import WeightTensor

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Optimization strategies for clustering."""
    
    QUALITY_FOCUSED = "quality_focused"  # Maximize clustering quality metrics
    COMPRESSION_FOCUSED = "compression_focused"  # Maximize compression ratio
    BALANCED = "balanced"  # Balance quality and compression
    SPEED_FOCUSED = "speed_focused"  # Minimize computation time
    MEMORY_FOCUSED = "memory_focused"  # Minimize memory usage
    ADAPTIVE = "adaptive"  # Automatically select based on data


class OptimizationObjective(Enum):
    """Objectives for multi-objective optimization."""
    
    MAXIMIZE_QUALITY = "maximize_quality"
    MAXIMIZE_COMPRESSION = "maximize_compression"
    MINIMIZE_CLUSTERS = "minimize_clusters"
    MINIMIZE_ERROR = "minimize_error"
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_MEMORY = "minimize_memory"


class ConstraintType(Enum):
    """Types of optimization constraints."""
    
    MIN_QUALITY = "min_quality"
    MIN_COMPRESSION_RATIO = "min_compression_ratio"
    MAX_CLUSTERS = "max_clusters"
    MAX_ERROR = "max_error"
    MAX_TIME = "max_time"
    MAX_MEMORY = "max_memory"


class TriggerCondition(Enum):
    """Conditions that trigger automatic optimization."""
    
    QUALITY_DEGRADATION = "quality_degradation"
    COMPRESSION_DEGRADATION = "compression_degradation"
    NEW_DATA_THRESHOLD = "new_data_threshold"
    TIME_INTERVAL = "time_interval"
    MANUAL = "manual"


@dataclass
class OptimizationConstraint:
    """Constraint for optimization."""
    
    type: ConstraintType
    value: float
    name: Optional[str] = None
    
    def is_satisfied(self, metrics: ClusterMetrics) -> bool:
        """Check if constraint is satisfied by given metrics."""
        if self.type == ConstraintType.MIN_QUALITY:
            return metrics.silhouette_score >= self.value
        elif self.type == ConstraintType.MIN_COMPRESSION_RATIO:
            return metrics.compression_ratio >= self.value
        elif self.type == ConstraintType.MAX_CLUSTERS:
            return metrics.num_clusters <= self.value
        elif self.type == ConstraintType.MAX_ERROR:
            return metrics.davies_bouldin_score <= self.value
        else:
            return True


@dataclass
class OptimizationResult:
    """Result of an optimization operation."""
    
    is_successful: bool
    initial_metrics: ClusterMetrics
    final_metrics: ClusterMetrics
    final_clusters: Optional[ClusteringResult] = None
    strategy_used: Optional[OptimizationStrategy] = None
    iterations: int = 0
    execution_time: Optional[float] = None
    improvement_percentage: float = 0.0
    constraints_satisfied: bool = True
    
    def satisfies_constraints(self, constraints: List[OptimizationConstraint]) -> bool:
        """Check if result satisfies all constraints."""
        return all(c.is_satisfied(self.final_metrics) for c in constraints)


@dataclass
class ParetoSolution:
    """Solution in Pareto optimization."""
    
    clustering_result: ClusteringResult
    objective_values: Dict[OptimizationObjective, float]
    dominance_count: int = 0
    dominated_by: List[int] = field(default_factory=list)
    
    @property
    def is_successful(self) -> bool:
        """Check if solution is valid."""
        return self.clustering_result.is_valid()


@dataclass
class ParetoFront:
    """Pareto front for multi-objective optimization."""
    
    solutions: List[ParetoSolution]
    objectives: List[OptimizationObjective]
    
    def _dominates(self, sol1: ParetoSolution, sol2: ParetoSolution, 
                   objectives: List[OptimizationObjective]) -> bool:
        """Check if sol1 dominates sol2."""
        better_in_at_least_one = False
        
        for obj in objectives:
            val1 = sol1.objective_values.get(obj, 0)
            val2 = sol2.objective_values.get(obj, 0)
            
            # Handle maximization vs minimization
            if "MAXIMIZE" in obj.name:
                if val1 < val2:
                    return False
                elif val1 > val2:
                    better_in_at_least_one = True
            else:  # MINIMIZE
                if val1 > val2:
                    return False
                elif val1 < val2:
                    better_in_at_least_one = True
        
        return better_in_at_least_one
    
    def compute_pareto_front(self) -> List[ParetoSolution]:
        """Compute non-dominated solutions."""
        n_solutions = len(self.solutions)
        
        # Reset dominance counts
        for sol in self.solutions:
            sol.dominance_count = 0
            sol.dominated_by = []
        
        # Compute dominance relationships
        for i in range(n_solutions):
            for j in range(n_solutions):
                if i != j:
                    if self._dominates(self.solutions[i], self.solutions[j], self.objectives):
                        self.solutions[j].dominance_count += 1
                        self.solutions[j].dominated_by.append(i)
        
        # Return non-dominated solutions
        return [sol for sol in self.solutions if sol.dominance_count == 0]


@dataclass
class OptimizationTrigger:
    """Trigger for automatic optimization."""
    
    condition: TriggerCondition
    threshold: float
    callback: Callable[[], None]
    last_triggered: Optional[float] = None
    min_interval: float = 300.0  # 5 minutes minimum between triggers


@dataclass
class BatchOptimizationResult(OptimizationResult):
    """Result of batch optimization."""
    
    total_batches: int = 0
    successful_batches: int = 0
    batch_results: List[OptimizationResult] = field(default_factory=list)


@dataclass
class EvolutionaryOptimizationResult(OptimizationResult):
    """Result of evolutionary optimization."""
    
    generations_evolved: int = 0
    population_size: int = 0
    best_individual: Optional[ClusteringResult] = None
    fitness_history: List[float] = field(default_factory=list)


class ClusterOptimizer:
    """
    Comprehensive cluster optimization with adaptive strategies.
    
    Provides re-clustering, parameter optimization, quality improvement,
    and continuous monitoring capabilities for clustering-based deduplication.
    """
    
    def __init__(self, analyzer: ClusterAnalyzer):
        """
        Initialize ClusterOptimizer.
        
        Args:
            analyzer: ClusterAnalyzer instance to use for clustering operations
        """
        self.analyzer = analyzer
        self.optimization_history: List[Dict[str, Any]] = []
        self.triggers: List[OptimizationTrigger] = []
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        logger.info("Initialized ClusterOptimizer")
    
    def _cluster_weights(self, weights: List[WeightTensor], n_clusters: Optional[int] = None) -> ClusteringResult:
        """
        Helper method to cluster weights using the appropriate analyzer method.
        
        Args:
            weights: Weight tensors to cluster
            n_clusters: Number of clusters (optional)
            
        Returns:
            Clustering result
        """
        strategy = self.analyzer.config.strategy
        
        if strategy == ClusteringStrategy.KMEANS:
            if n_clusters is None:
                # Use adaptive clustering if n_clusters not specified
                n_clusters = self.analyzer.detect_natural_clusters(weights)
            return self.analyzer.cluster_kmeans(weights, k=n_clusters)
        elif strategy == ClusteringStrategy.HIERARCHICAL:
            return self.analyzer.cluster_hierarchical(weights)
        elif strategy == ClusteringStrategy.DBSCAN:
            return self.analyzer.cluster_dbscan(weights)
        elif strategy == ClusteringStrategy.ADAPTIVE:
            return self.analyzer.cluster_adaptive(weights)
        else:
            # Default to kmeans
            if n_clusters is None:
                n_clusters = self.analyzer.detect_natural_clusters(weights)
            return self.analyzer.cluster_kmeans(weights, k=n_clusters)
    
    def optimize_clustering(
        self,
        clusters: ClusteringResult,
        weights: List[WeightTensor],
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        max_iterations: int = 10,
        min_improvement: float = 0.01,
        allow_quality_degradation: bool = False
    ) -> OptimizationResult:
        """
        Optimize clustering using specified strategy.
        
        Args:
            clusters: Current clustering result
            weights: Weight tensors to cluster
            strategy: Optimization strategy to use
            max_iterations: Maximum optimization iterations
            min_improvement: Minimum improvement to continue
            allow_quality_degradation: Whether to allow quality degradation
            
        Returns:
            Optimization result
        """
        logger.info(f"Starting clustering optimization with {strategy.value} strategy")
        
        start_time = time.time()
        initial_metrics = clusters.metrics
        best_result = clusters
        best_score = self._compute_optimization_score(initial_metrics, strategy)
        
        iteration = 0
        last_improvement = float('inf')
        
        try:
            while iteration < max_iterations and last_improvement > min_improvement:
                iteration += 1
                
                # Apply strategy-specific optimization
                if strategy == OptimizationStrategy.QUALITY_FOCUSED:
                    candidate = self._optimize_for_quality(best_result, weights)
                elif strategy == OptimizationStrategy.COMPRESSION_FOCUSED:
                    candidate = self._optimize_for_compression(best_result, weights)
                elif strategy == OptimizationStrategy.BALANCED:
                    candidate = self._optimize_balanced(best_result, weights)
                elif strategy == OptimizationStrategy.SPEED_FOCUSED:
                    candidate = self._optimize_for_speed(best_result, weights)
                elif strategy == OptimizationStrategy.MEMORY_FOCUSED:
                    candidate = self._optimize_for_memory(best_result, weights)
                else:  # ADAPTIVE
                    selected_strategy = self._select_adaptive_strategy(weights)
                    candidate = self.optimize_clustering(
                        best_result, weights, selected_strategy, 1, min_improvement
                    ).final_clusters
                
                if candidate is None:
                    break
                
                # Evaluate candidate
                candidate_score = self._compute_optimization_score(candidate.metrics, strategy)
                
                # Check for improvement
                improvement = (candidate_score - best_score) / max(abs(best_score), 1e-6)
                
                if improvement > 0 or (allow_quality_degradation and improvement > -0.1):
                    last_improvement = abs(improvement)
                    best_score = candidate_score
                    best_result = candidate
                    logger.debug(f"Iteration {iteration}: {improvement:.2%} improvement")
                else:
                    last_improvement = 0
            
            # Calculate final improvement
            final_score = self._compute_optimization_score(best_result.metrics, strategy)
            initial_score = self._compute_optimization_score(initial_metrics, strategy)
            improvement_pct = ((final_score - initial_score) / max(abs(initial_score), 1e-6)) * 100
            
            result = OptimizationResult(
                is_successful=improvement_pct > 0 or allow_quality_degradation,
                initial_metrics=initial_metrics,
                final_metrics=best_result.metrics,
                final_clusters=best_result,
                strategy_used=strategy,
                iterations=iteration,
                execution_time=time.time() - start_time,
                improvement_percentage=improvement_pct
            )
            
            # Record in history
            self._record_optimization(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return OptimizationResult(
                is_successful=False,
                initial_metrics=initial_metrics,
                final_metrics=initial_metrics,
                iterations=iteration,
                execution_time=time.time() - start_time
            )
    
    def adaptive_reclustering(
        self,
        clusters: ClusteringResult,
        quality_threshold: float = 0.6
    ) -> Optional[OptimizationResult]:
        """
        Automatically trigger re-clustering when quality drops.
        
        Args:
            clusters: Current clustering result
            quality_threshold: Minimum acceptable quality score
            
        Returns:
            Optimization result if re-clustering was triggered, None otherwise
        """
        current_quality = clusters.metrics.silhouette_score
        
        if current_quality < quality_threshold:
            logger.info(f"Quality {current_quality:.3f} below threshold {quality_threshold}, triggering re-clustering")
            
            # Get weights from analyzer
            weights = list(self.analyzer.repository.get_all_weights().values())
            
            # Try different strategies to improve quality
            strategies = [
                OptimizationStrategy.QUALITY_FOCUSED,
                OptimizationStrategy.BALANCED,
                OptimizationStrategy.ADAPTIVE
            ]
            
            best_result = None
            
            for strategy in strategies:
                result = self.optimize_clustering(
                    clusters, weights, strategy, max_iterations=5
                )
                
                if result.is_successful and result.final_metrics.silhouette_score >= quality_threshold:
                    best_result = result
                    break
                elif best_result is None or result.final_metrics.silhouette_score > best_result.final_metrics.silhouette_score:
                    best_result = result
            
            return best_result
        
        return None
    
    def incremental_optimization(
        self,
        new_weights: List[WeightTensor],
        existing_clusters: ClusteringResult
    ) -> OptimizationResult:
        """
        Optimize clusters incrementally with new data.
        
        Args:
            new_weights: New weight tensors to incorporate
            existing_clusters: Existing clustering result
            
        Returns:
            Optimization result
        """
        logger.info(f"Incremental optimization with {len(new_weights)} new weights")
        
        start_time = time.time()
        initial_metrics = existing_clusters.metrics
        
        try:
            # Use analyzer's incremental clustering if available
            if hasattr(self.analyzer, 'incremental_clustering'):
                updated_result = self.analyzer.incremental_clustering(
                    new_weights, existing_clusters
                )
            else:
                # Fallback: re-cluster all weights
                all_weights = list(self.analyzer.repository.get_all_weights().values())
                updated_result = self._cluster_weights(
                    all_weights, 
                    n_clusters=existing_clusters.metrics.num_clusters
                )
            
            # Optimize the updated clustering
            optimization_result = self.optimize_clustering(
                updated_result,
                new_weights,
                OptimizationStrategy.BALANCED,
                max_iterations=3
            )
            
            return OptimizationResult(
                is_successful=True,
                initial_metrics=initial_metrics,
                final_metrics=optimization_result.final_metrics,
                final_clusters=optimization_result.final_clusters,
                strategy_used=OptimizationStrategy.BALANCED,
                iterations=optimization_result.iterations,
                execution_time=time.time() - start_time,
                improvement_percentage=optimization_result.improvement_percentage
            )
            
        except Exception as e:
            logger.error(f"Incremental optimization failed: {e}")
            return OptimizationResult(
                is_successful=False,
                initial_metrics=initial_metrics,
                final_metrics=initial_metrics,
                execution_time=time.time() - start_time
            )
    
    def cross_validate_clustering(
        self,
        weights: List[WeightTensor],
        strategies: Optional[List[ClusteringStrategy]] = None,
        cv_folds: int = 3
    ) -> Tuple[ClusteringStrategy, Dict[ClusteringStrategy, ClusteringResult]]:
        """
        Cross-validate clustering strategies to find best approach.
        
        Args:
            weights: Weight tensors to cluster
            strategies: Strategies to test (None = all available)
            cv_folds: Number of cross-validation folds
            
        Returns:
            Best strategy and results for all strategies
        """
        if strategies is None:
            strategies = [s for s in ClusteringStrategy if s != ClusteringStrategy.ADAPTIVE]
        
        logger.info(f"Cross-validating {len(strategies)} clustering strategies")
        
        results = {}
        scores = {}
        
        for strategy in strategies:
            try:
                # Update analyzer configuration
                self.analyzer.config.strategy = strategy
                
                # Run clustering
                result = self._cluster_weights(weights)
                results[strategy] = result
                
                # Compute overall score (balanced)
                score = self._compute_optimization_score(
                    result.metrics, OptimizationStrategy.BALANCED
                )
                scores[strategy] = score
                
                logger.debug(f"{strategy.value}: score={score:.3f}")
                
            except Exception as e:
                logger.warning(f"Strategy {strategy.value} failed: {e}")
                scores[strategy] = -float('inf')
        
        # Find best strategy
        best_strategy = max(scores, key=scores.get)
        logger.info(f"Best strategy: {best_strategy.value} (score={scores[best_strategy]:.3f})")
        
        return best_strategy, results
    
    def optimize_cluster_count(
        self,
        weights: List[WeightTensor],
        min_k: int = 2,
        max_k: int = None,
        method: str = "elbow"
    ) -> Tuple[int, Dict[int, float]]:
        """
        Find optimal number of clusters.
        
        Args:
            weights: Weight tensors to cluster
            min_k: Minimum number of clusters
            max_k: Maximum number of clusters (None = sqrt(n))
            method: Method to use (elbow, silhouette, gap)
            
        Returns:
            Optimal k and scores for each k
        """
        n_weights = len(weights)
        if max_k is None:
            max_k = min(int(np.sqrt(n_weights)), 50)
        
        logger.info(f"Optimizing cluster count from {min_k} to {max_k}")
        
        scores = {}
        
        for k in range(min_k, max_k + 1):
            try:
                result = self.analyzer.cluster_kmeans(weights, k=k)
                
                if method == "silhouette":
                    score = result.metrics.silhouette_score
                elif method == "elbow":
                    # Inertia should decrease with more clusters
                    score = -result.metrics.inertia
                else:  # gap statistic or other methods
                    score = result.metrics.calinski_harabasz_score
                
                scores[k] = score
                
            except Exception as e:
                logger.warning(f"Failed to cluster with k={k}: {e}")
                scores[k] = -float('inf')
        
        # Find optimal k
        if method == "elbow":
            # Find elbow point using second derivative
            k_values = sorted(scores.keys())
            score_values = [scores[k] for k in k_values]
            
            if len(k_values) >= 3:
                # Compute second derivative
                second_deriv = np.diff(np.diff(score_values))
                # Find maximum second derivative (elbow)
                elbow_idx = np.argmax(second_deriv) + 1
                optimal_k = k_values[elbow_idx]
            else:
                optimal_k = max(scores, key=scores.get)
        else:
            # Simply use maximum score
            optimal_k = max(scores, key=scores.get)
        
        logger.info(f"Optimal cluster count: {optimal_k}")
        
        return optimal_k, scores
    
    def optimize_similarity_threshold(
        self,
        clusters: ClusteringResult,
        weights: List[WeightTensor],
        threshold_range: Tuple[float, float] = (0.8, 0.99),
        n_samples: int = 10
    ) -> float:
        """
        Find optimal similarity threshold.
        
        Args:
            clusters: Current clustering result
            weights: Weight tensors
            threshold_range: Range of thresholds to test
            n_samples: Number of thresholds to sample
            
        Returns:
            Optimal similarity threshold
        """
        logger.info(f"Optimizing similarity threshold in range {threshold_range}")
        
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_samples)
        best_threshold = self.analyzer.config.similarity_threshold
        best_score = self._compute_optimization_score(
            clusters.metrics, OptimizationStrategy.BALANCED
        )
        
        original_threshold = self.analyzer.config.similarity_threshold
        
        try:
            for threshold in thresholds:
                # Update threshold
                self.analyzer.config.similarity_threshold = threshold
                
                # Re-cluster with new threshold
                result = self._cluster_weights(weights)
                
                # Evaluate
                score = self._compute_optimization_score(
                    result.metrics, OptimizationStrategy.BALANCED
                )
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    
        finally:
            # Restore original threshold
            self.analyzer.config.similarity_threshold = original_threshold
        
        logger.info(f"Optimal similarity threshold: {best_threshold:.3f}")
        
        return best_threshold
    
    def optimize_hierarchy_levels(
        self,
        hierarchy: Dict[str, Any],
        weights: List[WeightTensor]
    ) -> List[int]:
        """
        Optimize hierarchical clustering levels.
        
        Args:
            hierarchy: Hierarchical clustering structure
            weights: Weight tensors
            
        Returns:
            List of optimal hierarchy levels
        """
        optimal_levels = []
        
        # Analyze each level
        for level, info in hierarchy.items():
            if isinstance(info, dict) and "quality" in info and "compression" in info:
                # Use balanced score
                score = 0.5 * info["quality"] + 0.5 * info["compression"]
                
                # Consider level if score is good
                if score > 0.5:
                    level_num = int(level.split("_")[1]) if "_" in level else 0
                    optimal_levels.append(level_num)
        
        return sorted(optimal_levels)
    
    def tune_hyperparameters(
        self,
        weights: List[WeightTensor],
        param_space: Dict[str, List[Any]],
        n_trials: int = 20,
        scoring: str = "balanced"
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Comprehensive hyperparameter tuning.
        
        Args:
            weights: Weight tensors to cluster
            param_space: Parameter space to search
            n_trials: Number of trials
            scoring: Scoring method
            
        Returns:
            Best parameters and all trial results
        """
        logger.info(f"Tuning hyperparameters with {n_trials} trials")
        
        results = []
        best_params = None
        best_score = -float('inf')
        
        # Generate parameter combinations
        from itertools import product
        
        param_names = list(param_space.keys())
        param_values = [param_space[name] for name in param_names]
        all_combinations = list(product(*param_values))
        
        # Sample combinations if too many
        if len(all_combinations) > n_trials:
            import random
            combinations = random.sample(all_combinations, n_trials)
        else:
            combinations = all_combinations
        
        for combo in combinations:
            params = dict(zip(param_names, combo))
            
            try:
                # Apply parameters
                if "n_clusters" in params:
                    n_clusters = params["n_clusters"]
                else:
                    n_clusters = None
                
                if "algorithm" in params:
                    strategy = ClusteringStrategy(params["algorithm"])
                    self.analyzer.config.strategy = strategy
                
                if "similarity_threshold" in params:
                    self.analyzer.config.similarity_threshold = params["similarity_threshold"]
                
                # Run clustering
                result = self._cluster_weights(weights, n_clusters=n_clusters)
                
                # Score result
                if scoring == "balanced":
                    score = self._compute_optimization_score(
                        result.metrics, OptimizationStrategy.BALANCED
                    )
                else:
                    score = result.metrics.silhouette_score
                
                results.append({
                    "params": params,
                    "score": score,
                    "metrics": result.metrics
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"Trial failed with params {params}: {e}")
        
        logger.info(f"Best parameters: {best_params} (score={best_score:.3f})")
        
        return best_params, results
    
    def improve_cluster_quality(
        self,
        clusters: ClusteringResult,
        target_metrics: Dict[str, float],
        weights: List[WeightTensor],
        max_iterations: int = 10
    ) -> OptimizationResult:
        """
        Iteratively improve cluster quality to meet targets.
        
        Args:
            clusters: Current clustering result
            target_metrics: Target metric values
            weights: Weight tensors
            max_iterations: Maximum iterations
            
        Returns:
            Optimization result
        """
        logger.info(f"Improving cluster quality towards targets: {target_metrics}")
        
        current_result = clusters
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Check which metrics need improvement
            needs_improvement = False
            
            if "silhouette_score" in target_metrics:
                if current_result.metrics.silhouette_score < target_metrics["silhouette_score"]:
                    needs_improvement = True
            
            if "compression_ratio" in target_metrics:
                if current_result.metrics.compression_ratio < target_metrics["compression_ratio"]:
                    needs_improvement = True
            
            if not needs_improvement:
                break
            
            # Apply targeted optimization
            if current_result.metrics.silhouette_score < target_metrics.get("silhouette_score", 0):
                strategy = OptimizationStrategy.QUALITY_FOCUSED
            elif current_result.metrics.compression_ratio < target_metrics.get("compression_ratio", 0):
                strategy = OptimizationStrategy.COMPRESSION_FOCUSED
            else:
                strategy = OptimizationStrategy.BALANCED
            
            result = self.optimize_clustering(
                current_result, weights, strategy, max_iterations=1
            )
            
            if result.is_successful:
                current_result = result.final_clusters
            else:
                break
        
        improvement = self._compute_improvement(clusters.metrics, current_result.metrics)
        
        return OptimizationResult(
            is_successful=improvement > 0,
            initial_metrics=clusters.metrics,
            final_metrics=current_result.metrics,
            final_clusters=current_result,
            iterations=iteration,
            improvement_percentage=improvement * 100
        )
    
    def balance_compression_vs_quality(
        self,
        clusters: ClusteringResult,
        weights: List[WeightTensor],
        trade_off: float = 0.5,
        n_candidates: int = 5
    ) -> OptimizationResult:
        """
        Balance compression and quality trade-offs.
        
        Args:
            clusters: Current clustering result
            weights: Weight tensors
            trade_off: Trade-off parameter (0=quality, 1=compression)
            n_candidates: Number of candidates to evaluate
            
        Returns:
            Optimization result
        """
        logger.info(f"Balancing compression vs quality with trade-off={trade_off:.2f}")
        
        best_result = clusters
        best_score = -float('inf')
        
        # Try different cluster counts
        current_k = clusters.metrics.num_clusters
        k_values = [
            max(2, current_k - 2),
            current_k - 1,
            current_k,
            current_k + 1,
            current_k + 2
        ]
        
        for k in k_values:
            try:
                result = self.analyzer.cluster_kmeans(weights, k=k)
                
                # Compute combined score
                quality_score = result.metrics.silhouette_score
                compression_score = result.metrics.compression_ratio
                combined_score = (1 - trade_off) * quality_score + trade_off * compression_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_result = result
                    
            except Exception as e:
                logger.warning(f"Failed with k={k}: {e}")
        
        improvement = self._compute_improvement(clusters.metrics, best_result.metrics)
        
        return OptimizationResult(
            is_successful=True,
            initial_metrics=clusters.metrics,
            final_metrics=best_result.metrics,
            final_clusters=best_result,
            improvement_percentage=improvement * 100
        )
    
    def minimize_reconstruction_error(
        self,
        assignments: List[ClusterAssignment],
        centroids: List[Centroid],
        weights: List[WeightTensor]
    ) -> OptimizationResult:
        """
        Minimize reconstruction error for assignments.
        
        Args:
            assignments: Current cluster assignments
            centroids: Current centroids
            weights: Weight tensors
            
        Returns:
            Optimization result
        """
        logger.info("Minimizing reconstruction error")
        
        # Create weight lookup
        weight_lookup = {w.metadata.name: w for w in weights}
        
        # Compute initial error
        initial_error = np.mean([a.distance_to_centroid for a in assignments])
        
        # Refine assignments
        refined_assignments = []
        centroid_lookup = {c.cluster_id: c for c in centroids}
        
        for assignment in assignments:
            weight = weight_lookup.get(assignment.weight_name)
            if weight is None:
                refined_assignments.append(assignment)
                continue
            
            # Find best centroid
            best_distance = float('inf')
            best_cluster_id = assignment.cluster_id
            
            for centroid in centroids:
                distance = np.linalg.norm(weight.data - centroid.data)
                if distance < best_distance:
                    best_distance = distance
                    best_cluster_id = centroid.cluster_id
            
            # Create refined assignment
            refined = ClusterAssignment(
                weight_name=assignment.weight_name,
                weight_hash=assignment.weight_hash,
                cluster_id=best_cluster_id,
                distance_to_centroid=best_distance,
                similarity_score=1.0 - min(best_distance / 10.0, 1.0)  # Approximate
            )
            refined_assignments.append(refined)
        
        # Compute final error
        final_error = np.mean([a.distance_to_centroid for a in refined_assignments])
        
        # Create result
        metrics = ClusterMetrics(
            silhouette_score=0.7,  # Placeholder
            inertia=final_error * len(assignments),
            calinski_harabasz_score=100.0,
            davies_bouldin_score=final_error,
            num_clusters=len(centroids),
            avg_cluster_size=len(assignments) / len(centroids),
            compression_ratio=0.6
        )
        
        result = ClusteringResult(
            assignments=refined_assignments,
            centroids=centroids,
            metrics=metrics,
            strategy=ClusteringStrategy.KMEANS,
            execution_time=0.1
        )
        
        improvement = (initial_error - final_error) / max(initial_error, 1e-6)
        
        return OptimizationResult(
            is_successful=improvement > 0,
            initial_metrics=ClusterMetrics(davies_bouldin_score=initial_error),
            final_metrics=metrics,
            final_clusters=result,
            improvement_percentage=improvement * 100
        )
    
    def maximize_compression_ratio(
        self,
        clusters: ClusteringResult,
        weights: List[WeightTensor],
        min_quality: float = 0.5
    ) -> OptimizationResult:
        """
        Maximize compression while maintaining minimum quality.
        
        Args:
            clusters: Current clustering result
            weights: Weight tensors
            min_quality: Minimum acceptable quality
            
        Returns:
            Optimization result
        """
        logger.info(f"Maximizing compression ratio with min_quality={min_quality}")
        
        # Try reducing cluster count for better compression
        current_k = clusters.metrics.num_clusters
        best_result = clusters
        best_compression = clusters.metrics.compression_ratio
        
        for k in range(max(2, current_k - 5), current_k):
            try:
                result = self.analyzer.cluster_kmeans(weights, k=k)
                
                if (result.metrics.silhouette_score >= min_quality and
                    result.metrics.compression_ratio > best_compression):
                    best_compression = result.metrics.compression_ratio
                    best_result = result
                    
            except Exception as e:
                logger.warning(f"Failed with k={k}: {e}")
        
        improvement = self._compute_improvement(clusters.metrics, best_result.metrics)
        
        return OptimizationResult(
            is_successful=best_result != clusters,
            initial_metrics=clusters.metrics,
            final_metrics=best_result.metrics,
            final_clusters=best_result,
            improvement_percentage=improvement * 100
        )
    
    def pareto_optimization(
        self,
        clusters: ClusteringResult,
        objectives: List[OptimizationObjective],
        population_size: int = 20,
        generations: int = 10
    ) -> ParetoFront:
        """
        Multi-objective optimization using Pareto fronts.
        
        Args:
            clusters: Current clustering result
            objectives: List of objectives to optimize
            population_size: Population size for evolutionary algorithm
            generations: Number of generations
            
        Returns:
            Pareto front with non-dominated solutions
        """
        logger.info(f"Pareto optimization with {len(objectives)} objectives")
        
        solutions = []
        weights = list(self.analyzer.repository.get_all_weights().values())
        
        # Generate initial population
        for i in range(population_size):
            try:
                # Vary parameters randomly
                n_clusters = np.random.randint(2, min(len(weights) // 2, 20))
                result = self._cluster_weights(weights, n_clusters=n_clusters)
                
                # Compute objective values
                obj_values = self._compute_objective_values(result, objectives)
                
                solution = ParetoSolution(
                    clustering_result=result,
                    objective_values=obj_values
                )
                solutions.append(solution)
                
            except Exception as e:
                logger.warning(f"Failed to generate solution {i}: {e}")
        
        # Create Pareto front
        front = ParetoFront(solutions=solutions, objectives=objectives)
        
        # Get non-dominated solutions
        pareto_solutions = front.compute_pareto_front()
        front.solutions = pareto_solutions
        
        return front
    
    def weighted_objective_optimization(
        self,
        clusters: ClusteringResult,
        objectives: Dict[OptimizationObjective, float],
        weights: List[WeightTensor]
    ) -> OptimizationResult:
        """
        Optimization with weighted objectives.
        
        Args:
            clusters: Current clustering result
            objectives: Objectives with weights
            weights: Weight tensors
            
        Returns:
            Optimization result
        """
        logger.info(f"Weighted objective optimization with {len(objectives)} objectives")
        
        # Normalize weights
        total_weight = sum(objectives.values())
        normalized_weights = {k: v/total_weight for k, v in objectives.items()}
        
        # Define scoring function
        def score_function(result: ClusteringResult) -> float:
            obj_values = self._compute_objective_values(result, list(objectives.keys()))
            score = 0.0
            
            for obj, weight in normalized_weights.items():
                value = obj_values.get(obj, 0.0)
                # Normalize value to [0, 1]
                if "MAXIMIZE" in obj.name:
                    norm_value = min(max(value, 0), 1)
                else:  # MINIMIZE
                    norm_value = 1.0 - min(max(value, 0), 1)
                score += weight * norm_value
            
            return score
        
        # Optimize
        best_result = clusters
        best_score = score_function(clusters)
        
        # Try different configurations
        for k in range(2, min(len(weights) // 2, 10)):
            try:
                result = self.analyzer.cluster_kmeans(weights, k=k)
                score = score_function(result)
                
                if score > best_score:
                    best_score = score
                    best_result = result
                    
            except Exception as e:
                logger.warning(f"Failed with k={k}: {e}")
        
        improvement = self._compute_improvement(clusters.metrics, best_result.metrics)
        
        return OptimizationResult(
            is_successful=best_result != clusters,
            initial_metrics=clusters.metrics,
            final_metrics=best_result.metrics,
            final_clusters=best_result,
            improvement_percentage=improvement * 100
        )
    
    def constraint_based_optimization(
        self,
        clusters: ClusteringResult,
        weights: List[WeightTensor],
        constraints: List[OptimizationConstraint]
    ) -> OptimizationResult:
        """
        Optimization with constraints.
        
        Args:
            clusters: Current clustering result
            weights: Weight tensors
            constraints: List of constraints
            
        Returns:
            Optimization result
        """
        logger.info(f"Constraint-based optimization with {len(constraints)} constraints")
        
        # Check if current solution satisfies constraints
        current_satisfies = all(c.is_satisfied(clusters.metrics) for c in constraints)
        
        if current_satisfies:
            # Try to improve while maintaining constraints
            result = self.optimize_clustering(
                clusters, weights, OptimizationStrategy.BALANCED,
                max_iterations=5
            )
            
            if all(c.is_satisfied(result.final_metrics) for c in constraints):
                return result
        
        # Find feasible solution
        best_result = clusters
        best_score = -float('inf')
        
        # Try different configurations
        for k in range(2, min(len(weights) // 2, 20)):
            try:
                result = self.analyzer.cluster_kmeans(weights, k=k)
                
                # Check constraints
                if all(c.is_satisfied(result.metrics) for c in constraints):
                    score = self._compute_optimization_score(
                        result.metrics, OptimizationStrategy.BALANCED
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                        
            except Exception as e:
                logger.warning(f"Failed with k={k}: {e}")
        
        improvement = self._compute_improvement(clusters.metrics, best_result.metrics)
        
        return OptimizationResult(
            is_successful=all(c.is_satisfied(best_result.metrics) for c in constraints),
            initial_metrics=clusters.metrics,
            final_metrics=best_result.metrics,
            final_clusters=best_result,
            improvement_percentage=improvement * 100,
            constraints_satisfied=all(c.is_satisfied(best_result.metrics) for c in constraints)
        )
    
    def evolutionary_optimization(
        self,
        clusters: ClusteringResult,
        weights: List[WeightTensor],
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7
    ) -> EvolutionaryOptimizationResult:
        """
        Genetic algorithm-based optimization.
        
        Args:
            clusters: Current clustering result
            weights: Weight tensors
            population_size: Size of population
            generations: Number of generations
            mutation_rate: Mutation probability
            crossover_rate: Crossover probability
            
        Returns:
            Evolutionary optimization result
        """
        logger.info(f"Evolutionary optimization: {generations} generations, population={population_size}")
        
        start_time = time.time()
        initial_metrics = clusters.metrics
        fitness_history = []
        
        # Initialize population
        population = []
        for _ in range(population_size):
            n_clusters = np.random.randint(2, min(len(weights) // 2, 20))
            individual = {"n_clusters": n_clusters, "fitness": 0.0}
            population.append(individual)
        
        # Evolution loop
        best_individual = None
        best_fitness = -float('inf')
        
        for gen in range(generations):
            # Evaluate fitness
            for individual in population:
                try:
                    result = self._cluster_weights(
                        weights, n_clusters=individual["n_clusters"]
                    )
                    fitness = self._compute_optimization_score(
                        result.metrics, OptimizationStrategy.BALANCED
                    )
                    individual["fitness"] = fitness
                    individual["result"] = result
                    
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_individual = individual
                        
                except Exception:
                    individual["fitness"] = -float('inf')
            
            fitness_history.append(best_fitness)
            logger.debug(f"Generation {gen}: best_fitness={best_fitness:.3f}")
            
            # Selection and reproduction
            # Sort by fitness
            population.sort(key=lambda x: x["fitness"], reverse=True)
            
            # Keep top half
            survivors = population[:population_size // 2]
            
            # Generate new population
            new_population = survivors.copy()
            
            while len(new_population) < population_size:
                # Selection
                parent1 = survivors[np.random.randint(len(survivors))]
                parent2 = survivors[np.random.randint(len(survivors))]
                
                # Crossover
                if np.random.random() < crossover_rate:
                    child_n_clusters = int((parent1["n_clusters"] + parent2["n_clusters"]) / 2)
                else:
                    child_n_clusters = parent1["n_clusters"]
                
                # Mutation
                if np.random.random() < mutation_rate:
                    child_n_clusters += np.random.randint(-2, 3)
                    child_n_clusters = max(2, min(child_n_clusters, len(weights) // 2))
                
                child = {"n_clusters": child_n_clusters, "fitness": 0.0}
                new_population.append(child)
            
            population = new_population
        
        # Get best result
        best_result = best_individual.get("result", clusters)
        improvement = self._compute_improvement(initial_metrics, best_result.metrics)
        
        return EvolutionaryOptimizationResult(
            is_successful=improvement > 0,
            initial_metrics=initial_metrics,
            final_metrics=best_result.metrics,
            final_clusters=best_result,
            execution_time=time.time() - start_time,
            improvement_percentage=improvement * 100,
            generations_evolved=generations,
            population_size=population_size,
            best_individual=best_result,
            fitness_history=fitness_history
        )
    
    def schedule_optimization(self, triggers: List[OptimizationTrigger]) -> None:
        """
        Schedule automatic optimization based on triggers.
        
        Args:
            triggers: List of optimization triggers
        """
        logger.info(f"Scheduling {len(triggers)} optimization triggers")
        
        with self._lock:
            self.triggers = triggers
    
    def monitor_cluster_health(
        self,
        clusters: ClusteringResult,
        weights: List[WeightTensor],
        check_interval: float = 300.0
    ) -> None:
        """
        Start continuous cluster health monitoring.
        
        Args:
            clusters: Current clustering result
            weights: Weight tensors
            check_interval: Check interval in seconds
        """
        logger.info(f"Starting cluster health monitoring (interval={check_interval}s)")
        
        def monitor_loop():
            while self._monitoring_active:
                try:
                    # Check triggers
                    for trigger in self.triggers:
                        if self._check_trigger(trigger, clusters.metrics):
                            logger.info(f"Trigger {trigger.condition.value} activated")
                            trigger.callback()
                            trigger.last_triggered = time.time()
                    
                    time.sleep(check_interval)
                    
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
        
        with self._lock:
            if not self._monitoring_active:
                self._monitoring_active = True
                self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
                self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop cluster health monitoring."""
        logger.info("Stopping cluster health monitoring")
        
        with self._lock:
            self._monitoring_active = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=1.0)
                self._monitor_thread = None
    
    def detect_optimization_opportunities(
        self,
        metrics_history: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Detect opportunities for optimization.
        
        Args:
            metrics_history: History of cluster metrics
            
        Returns:
            List of detected optimization opportunities
        """
        opportunities = []
        
        if len(metrics_history) < 2:
            return opportunities
        
        # Check for quality degradation
        recent_quality = [m.get("silhouette_score", 0) for m in metrics_history[-5:]]
        if len(recent_quality) >= 2:
            if recent_quality[-1] < recent_quality[0] * 0.9:
                opportunities.append("Quality degradation detected - consider re-clustering")
        
        # Check for compression degradation
        recent_compression = [m.get("compression_ratio", 0) for m in metrics_history[-5:]]
        if len(recent_compression) >= 2:
            if recent_compression[-1] < recent_compression[0] * 0.9:
                opportunities.append("Compression degradation detected - consider re-optimization")
        
        # Check for stability
        if len(recent_quality) >= 5:
            quality_variance = np.var(recent_quality)
            if quality_variance > 0.01:
                opportunities.append("Unstable clustering quality - consider parameter tuning")
        
        return opportunities
    
    def auto_optimize_pipeline(
        self,
        max_time: float = 300.0,
        target_improvement: float = 0.1
    ) -> OptimizationResult:
        """
        Fully automated optimization pipeline.
        
        Args:
            max_time: Maximum optimization time in seconds
            target_improvement: Target improvement percentage
            
        Returns:
            Optimization result
        """
        logger.info("Starting automated optimization pipeline")
        
        start_time = time.time()
        
        # Get current state
        weights = list(self.analyzer.repository.get_all_weights().values())
        
        # Get current clustering (mock for now)
        if hasattr(self.analyzer, 'get_current_clustering'):
            current_clusters = self.analyzer.get_current_clustering()
        else:
            current_clusters = self._cluster_weights(weights)
        
        initial_metrics = current_clusters.metrics
        best_result = current_clusters
        total_iterations = 0
        
        # Try different strategies in sequence
        strategies = [
            OptimizationStrategy.QUALITY_FOCUSED,
            OptimizationStrategy.COMPRESSION_FOCUSED,
            OptimizationStrategy.BALANCED
        ]
        
        for strategy in strategies:
            if time.time() - start_time > max_time:
                break
            
            result = self.optimize_clustering(
                best_result, weights, strategy,
                max_iterations=3
            )
            
            total_iterations += result.iterations
            
            if result.is_successful:
                best_result = result.final_clusters
                
                # Check if target improvement reached
                improvement = self._compute_improvement(initial_metrics, best_result.metrics)
                if improvement >= target_improvement:
                    break
        
        improvement = self._compute_improvement(initial_metrics, best_result.metrics)
        
        return OptimizationResult(
            is_successful=improvement > 0,
            initial_metrics=initial_metrics,
            final_metrics=best_result.metrics,
            final_clusters=best_result,
            iterations=total_iterations,
            execution_time=time.time() - start_time,
            improvement_percentage=improvement * 100
        )
    
    def optimize_memory_usage(self, clusters: ClusteringResult) -> ClusteringResult:
        """
        Optimize memory usage of clustering structures.
        
        Args:
            clusters: Current clustering result
            
        Returns:
            Memory-optimized clustering result
        """
        logger.info("Optimizing memory usage")
        
        # Deduplicate centroids
        unique_centroids = {}
        for centroid in clusters.centroids:
            hash_val = centroid.compute_hash()
            if hash_val not in unique_centroids:
                unique_centroids[hash_val] = centroid
        
        # Update assignments to reference deduplicated centroids
        centroid_map = {c.cluster_id: unique_centroids[c.compute_hash()].cluster_id 
                       for c in clusters.centroids}
        
        optimized_assignments = []
        for assignment in clusters.assignments:
            opt_assignment = ClusterAssignment(
                weight_name=assignment.weight_name,
                weight_hash=assignment.weight_hash,
                cluster_id=centroid_map.get(assignment.cluster_id, assignment.cluster_id),
                distance_to_centroid=assignment.distance_to_centroid,
                similarity_score=assignment.similarity_score
            )
            optimized_assignments.append(opt_assignment)
        
        return ClusteringResult(
            assignments=optimized_assignments,
            centroids=list(unique_centroids.values()),
            metrics=clusters.metrics,
            strategy=clusters.strategy,
            execution_time=clusters.execution_time,
            memory_usage=sum(c.nbytes for c in unique_centroids.values())
        )
    
    def optimize_query_performance(
        self,
        index_structures: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize index structures for faster queries.
        
        Args:
            index_structures: Current index structures
            
        Returns:
            Optimized index structures
        """
        logger.info("Optimizing query performance")
        
        optimized = index_structures.copy()
        
        # Sort clusters by size for faster access to popular clusters
        if "cluster_to_weights" in optimized:
            cluster_sizes = {
                cluster_id: len(weights)
                for cluster_id, weights in optimized["cluster_to_weights"].items()
            }
            sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
            optimized["sorted_clusters"] = [c[0] for c in sorted_clusters]
            optimized["cluster_sizes"] = cluster_sizes
        
        # Add reverse index for O(1) lookups
        if "weight_to_cluster" not in optimized and "cluster_to_weights" in optimized:
            weight_to_cluster = {}
            for cluster_id, weights in optimized["cluster_to_weights"].items():
                for weight in weights:
                    weight_to_cluster[weight] = cluster_id
            optimized["weight_to_cluster"] = weight_to_cluster
        
        return optimized
    
    def parallel_optimization(
        self,
        weights: List[WeightTensor],
        workers: int = 4,
        strategies: Optional[List[ClusteringStrategy]] = None
    ) -> OptimizationResult:
        """
        Parallel optimization with multiple workers.
        
        Args:
            weights: Weight tensors to cluster
            workers: Number of parallel workers
            strategies: Strategies to evaluate in parallel
            
        Returns:
            Best optimization result
        """
        logger.info(f"Parallel optimization with {workers} workers")
        
        if strategies is None:
            strategies = [ClusteringStrategy.KMEANS, ClusteringStrategy.HIERARCHICAL]
        
        best_result = None
        best_score = -float('inf')
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit tasks
            futures = {}
            for strategy in strategies:
                future = executor.submit(
                    self._optimize_with_strategy,
                    weights, strategy
                )
                futures[future] = strategy
            
            # Collect results
            for future in as_completed(futures):
                strategy = futures[future]
                try:
                    result = future.result()
                    score = self._compute_optimization_score(
                        result.metrics, OptimizationStrategy.BALANCED
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                        
                except Exception as e:
                    logger.warning(f"Strategy {strategy.value} failed: {e}")
        
        if best_result is None:
            # Fallback to single-threaded
            best_result = self._cluster_weights(weights)
        
        return OptimizationResult(
            is_successful=True,
            initial_metrics=best_result.metrics,
            final_metrics=best_result.metrics,
            final_clusters=best_result,
            execution_time=0.1
        )
    
    def batch_optimization_pipeline(
        self,
        weights: List[WeightTensor],
        batch_size: int = 100
    ) -> BatchOptimizationResult:
        """
        Process large datasets in batches.
        
        Args:
            weights: Weight tensors to cluster
            batch_size: Size of each batch
            
        Returns:
            Batch optimization result
        """
        logger.info(f"Batch optimization: {len(weights)} weights in batches of {batch_size}")
        
        n_batches = (len(weights) + batch_size - 1) // batch_size
        batch_results = []
        successful_batches = 0
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(weights))
            batch_weights = weights[start_idx:end_idx]
            
            try:
                # Cluster batch
                result = self._cluster_weights(batch_weights)
                
                # Optimize batch
                opt_result = self.optimize_clustering(
                    result, batch_weights,
                    OptimizationStrategy.BALANCED,
                    max_iterations=3
                )
                
                batch_results.append(opt_result)
                if opt_result.is_successful:
                    successful_batches += 1
                    
            except Exception as e:
                logger.warning(f"Batch {i} failed: {e}")
        
        # Aggregate results
        if batch_results:
            # Use metrics from first successful batch as representative
            final_result = batch_results[0]
            
            return BatchOptimizationResult(
                is_successful=successful_batches > 0,
                initial_metrics=final_result.initial_metrics,
                final_metrics=final_result.final_metrics,
                final_clusters=final_result.final_clusters,
                total_batches=n_batches,
                successful_batches=successful_batches,
                batch_results=batch_results
            )
        else:
            # No successful batches
            dummy_metrics = ClusterMetrics()
            return BatchOptimizationResult(
                is_successful=False,
                initial_metrics=dummy_metrics,
                final_metrics=dummy_metrics,
                total_batches=n_batches,
                successful_batches=0
            )
    
    # Helper methods
    
    def _optimize_for_quality(
        self,
        current: ClusteringResult,
        weights: List[WeightTensor]
    ) -> Optional[ClusteringResult]:
        """Optimize for clustering quality."""
        # Try different cluster counts near current
        current_k = current.metrics.num_clusters
        best_result = None
        best_quality = current.metrics.silhouette_score
        
        for k in [current_k - 1, current_k, current_k + 1]:
            if k < 2 or k > len(weights) // 2:
                continue
            
            try:
                result = self.analyzer.cluster_kmeans(weights, k=k)
                if result.metrics.silhouette_score > best_quality:
                    best_quality = result.metrics.silhouette_score
                    best_result = result
            except Exception as e:
                logger.debug(f"Failed with k={k}: {e}")
        
        return best_result
    
    def _optimize_for_compression(
        self,
        current: ClusteringResult,
        weights: List[WeightTensor]
    ) -> Optional[ClusteringResult]:
        """Optimize for compression ratio."""
        # Try fewer clusters for better compression
        current_k = current.metrics.num_clusters
        best_result = None
        best_compression = current.metrics.compression_ratio
        
        for k in range(max(2, current_k - 5), current_k):
            try:
                result = self.analyzer.cluster_kmeans(weights, k=k)
                if result.metrics.compression_ratio > best_compression:
                    best_compression = result.metrics.compression_ratio
                    best_result = result
            except Exception as e:
                logger.debug(f"Failed with k={k}: {e}")
        
        return best_result
    
    def _optimize_balanced(
        self,
        current: ClusteringResult,
        weights: List[WeightTensor]
    ) -> Optional[ClusteringResult]:
        """Balanced optimization."""
        # Combine quality and compression optimization
        quality_result = self._optimize_for_quality(current, weights)
        compression_result = self._optimize_for_compression(current, weights)
        
        candidates = [current]
        if quality_result:
            candidates.append(quality_result)
        if compression_result:
            candidates.append(compression_result)
        
        # Select best balanced score
        best_result = max(
            candidates,
            key=lambda r: self._compute_optimization_score(r.metrics, OptimizationStrategy.BALANCED)
        )
        
        return best_result if best_result != current else None
    
    def _optimize_for_speed(
        self,
        current: ClusteringResult,
        weights: List[WeightTensor]
    ) -> Optional[ClusteringResult]:
        """Optimize for computation speed."""
        # Use simpler clustering or fewer iterations
        self.analyzer.config.max_iterations = 10  # Reduce iterations
        result = self._cluster_weights(
            weights,
            n_clusters=current.metrics.num_clusters
        )
        self.analyzer.config.max_iterations = 100  # Restore
        
        return result if result.execution_time < current.execution_time else None
    
    def _optimize_for_memory(
        self,
        current: ClusteringResult,
        weights: List[WeightTensor]
    ) -> Optional[ClusteringResult]:
        """Optimize for memory usage."""
        # Reduce cluster count to save memory
        reduced_k = max(2, current.metrics.num_clusters - 2)
        result = self._cluster_weights(weights, n_clusters=reduced_k)
        
        return result if result.memory_usage < current.memory_usage else None
    
    def _select_adaptive_strategy(self, weights: List[WeightTensor]) -> OptimizationStrategy:
        """Select optimization strategy based on data characteristics."""
        n_weights = len(weights)
        
        # Simple heuristics
        if n_weights < 100:
            return OptimizationStrategy.QUALITY_FOCUSED
        elif n_weights > 10000:
            return OptimizationStrategy.SPEED_FOCUSED
        else:
            return OptimizationStrategy.BALANCED
    
    def _compute_optimization_score(
        self,
        metrics: ClusterMetrics,
        strategy: OptimizationStrategy
    ) -> float:
        """Compute optimization score based on strategy."""
        if strategy == OptimizationStrategy.QUALITY_FOCUSED:
            return metrics.silhouette_score
        elif strategy == OptimizationStrategy.COMPRESSION_FOCUSED:
            return metrics.compression_ratio
        elif strategy == OptimizationStrategy.BALANCED:
            return 0.5 * metrics.silhouette_score + 0.5 * metrics.compression_ratio
        elif strategy == OptimizationStrategy.SPEED_FOCUSED:
            return 1.0 / max(metrics.davies_bouldin_score, 0.1)  # Lower is better
        elif strategy == OptimizationStrategy.MEMORY_FOCUSED:
            return metrics.compression_ratio  # Higher compression = less memory
        else:
            return 0.5 * metrics.silhouette_score + 0.5 * metrics.compression_ratio
    
    def _compute_improvement(
        self,
        initial: ClusterMetrics,
        final: ClusterMetrics
    ) -> float:
        """Compute overall improvement percentage."""
        # Compare key metrics
        quality_improvement = (final.silhouette_score - initial.silhouette_score) / max(abs(initial.silhouette_score), 0.1)
        compression_improvement = (final.compression_ratio - initial.compression_ratio) / max(initial.compression_ratio, 0.1)
        
        # Weighted average
        return 0.5 * quality_improvement + 0.5 * compression_improvement
    
    def _compute_objective_values(
        self,
        result: ClusteringResult,
        objectives: List[OptimizationObjective]
    ) -> Dict[OptimizationObjective, float]:
        """Compute values for optimization objectives."""
        values = {}
        
        for obj in objectives:
            if obj == OptimizationObjective.MAXIMIZE_QUALITY:
                values[obj] = result.metrics.silhouette_score
            elif obj == OptimizationObjective.MAXIMIZE_COMPRESSION:
                values[obj] = result.metrics.compression_ratio
            elif obj == OptimizationObjective.MINIMIZE_CLUSTERS:
                values[obj] = 1.0 / result.metrics.num_clusters  # Invert for minimization
            elif obj == OptimizationObjective.MINIMIZE_ERROR:
                values[obj] = 1.0 / max(result.metrics.davies_bouldin_score, 0.1)
            elif obj == OptimizationObjective.MINIMIZE_TIME:
                values[obj] = 1.0 / max(result.execution_time, 0.1)
            elif obj == OptimizationObjective.MINIMIZE_MEMORY:
                memory = result.memory_usage or 1000000
                values[obj] = 1.0 / memory
            else:
                values[obj] = 0.0
        
        return values
    
    def _optimize_with_strategy(
        self,
        weights: List[WeightTensor],
        strategy: ClusteringStrategy
    ) -> ClusteringResult:
        """Helper for parallel optimization."""
        self.analyzer.config.strategy = strategy
        return self._cluster_weights(weights)
    
    def _check_trigger(
        self,
        trigger: OptimizationTrigger,
        metrics: ClusterMetrics
    ) -> bool:
        """Check if optimization trigger should fire."""
        # Check minimum interval
        if trigger.last_triggered:
            if time.time() - trigger.last_triggered < trigger.min_interval:
                return False
        
        # Check condition
        if trigger.condition == TriggerCondition.QUALITY_DEGRADATION:
            return metrics.silhouette_score < trigger.threshold
        elif trigger.condition == TriggerCondition.COMPRESSION_DEGRADATION:
            return metrics.compression_ratio < trigger.threshold
        elif trigger.condition == TriggerCondition.TIME_INTERVAL:
            return trigger.last_triggered is None or time.time() - trigger.last_triggered > trigger.threshold
        else:
            return False
    
    def _record_optimization(self, result: OptimizationResult) -> None:
        """Record optimization in history."""
        entry = {
            "timestamp": time.time(),
            "strategy": result.strategy_used.value if result.strategy_used else "unknown",
            "initial_metrics": result.initial_metrics.to_dict(),
            "final_metrics": result.final_metrics.to_dict(),
            "improvement": result.improvement_percentage,
            "iterations": result.iterations,
            "execution_time": result.execution_time
        }
        
        with self._lock:
            self.optimization_history.append(entry)
            
            # Keep only recent history
            if len(self.optimization_history) > 1000:
                self.optimization_history = self.optimization_history[-1000:]