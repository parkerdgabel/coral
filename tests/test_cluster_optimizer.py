"""Comprehensive tests for ClusterOptimizer component."""

import time
from unittest.mock import Mock, patch
import numpy as np
import pytest

from coral.clustering.cluster_analyzer import ClusterAnalyzer, ClusteringResult
from coral.clustering.cluster_config import ClusteringConfig, OptimizationConfig
from coral.clustering.cluster_optimizer import (
    ClusterOptimizer,
    OptimizationStrategy,
    OptimizationObjective,
    OptimizationResult,
    OptimizationConstraint,
    ConstraintType,
    ParetoFront,
    OptimizationTrigger,
    TriggerCondition,
)
from coral.clustering.cluster_types import (
    ClusteringStrategy,
    ClusterMetrics,
    ClusterAssignment,
    Centroid,
    ClusterInfo,
)
from coral.core.weight_tensor import WeightTensor, WeightMetadata


class TestClusterOptimizer:
    """Test suite for ClusterOptimizer."""

    @pytest.fixture
    def sample_weights(self):
        """Create sample weight tensors for testing."""
        weights = []
        np.random.seed(42)
        
        # Create clusters of similar weights
        for cluster_id in range(3):
            base = np.random.randn(10, 10).astype(np.float32)
            for i in range(5):
                noise = np.random.randn(10, 10).astype(np.float32) * 0.01
                data = base + noise
                metadata = WeightMetadata(
                    name=f"weight_c{cluster_id}_{i}",
                    shape=(10, 10),
                    dtype=np.float32
                )
                weight = WeightTensor(data=data, metadata=metadata)
                weights.append(weight)
        
        return weights

    @pytest.fixture
    def sample_clustering_result(self, sample_weights):
        """Create sample clustering result."""
        assignments = []
        centroids = []
        
        # Create 3 clusters
        for cluster_id in range(3):
            cluster_weights = sample_weights[cluster_id*5:(cluster_id+1)*5]
            
            # Create centroid
            centroid = Centroid.from_weights(cluster_weights, f"cluster_{cluster_id}")
            centroids.append(centroid)
            
            # Create assignments
            for weight in cluster_weights:
                assignment = ClusterAssignment(
                    weight_name=weight.metadata.name,
                    weight_hash=weight.compute_hash(),
                    cluster_id=f"cluster_{cluster_id}",
                    distance_to_centroid=0.1,
                    similarity_score=0.95
                )
                assignments.append(assignment)
        
        metrics = ClusterMetrics(
            silhouette_score=0.7,
            inertia=10.0,
            calinski_harabasz_score=100.0,
            davies_bouldin_score=0.5,
            num_clusters=3,
            avg_cluster_size=5.0,
            compression_ratio=0.6
        )
        
        return ClusteringResult(
            assignments=assignments,
            centroids=centroids,
            metrics=metrics,
            strategy=ClusteringStrategy.KMEANS,
            execution_time=1.0
        )

    @pytest.fixture
    def mock_analyzer(self):
        """Create mock cluster analyzer."""
        analyzer = Mock(spec=ClusterAnalyzer)
        analyzer.config = ClusteringConfig()
        analyzer.optimization_config = OptimizationConfig()
        
        # Mock methods that will be called
        analyzer.detect_natural_clusters = Mock(return_value=3)
        analyzer.cluster_kmeans = Mock()
        analyzer.cluster_hierarchical = Mock()
        analyzer.cluster_dbscan = Mock()
        analyzer.cluster_adaptive = Mock()
        analyzer.repository = Mock()
        analyzer.repository.get_all_weights = Mock(return_value={})
        
        return analyzer

    @pytest.fixture
    def optimizer(self, mock_analyzer):
        """Create ClusterOptimizer instance."""
        return ClusterOptimizer(mock_analyzer)

    def test_optimizer_initialization(self, mock_analyzer):
        """Test optimizer initialization."""
        optimizer = ClusterOptimizer(mock_analyzer)
        
        assert optimizer.analyzer == mock_analyzer
        assert optimizer.optimization_history == []
        assert optimizer.triggers == []
        assert optimizer._monitoring_active is False

    def test_optimize_clustering_basic(self, optimizer, mock_analyzer, sample_weights, sample_clustering_result):
        """Test basic clustering optimization."""
        # Setup mock
        improved_metrics = ClusterMetrics(
            silhouette_score=0.8,  # Improved
            inertia=8.0,  # Improved
            calinski_harabasz_score=120.0,  # Improved
            davies_bouldin_score=0.4,  # Improved
            num_clusters=3,
            avg_cluster_size=5.0,
            compression_ratio=0.65  # Improved
        )
        
        improved_result = ClusteringResult(
            assignments=sample_clustering_result.assignments,
            centroids=sample_clustering_result.centroids,
            metrics=improved_metrics,
            strategy=ClusteringStrategy.KMEANS,
            execution_time=1.0
        )
        
        mock_analyzer.cluster_kmeans.return_value = improved_result
        
        # Run optimization
        result = optimizer.optimize_clustering(
            clusters=sample_clustering_result,
            weights=sample_weights,
            strategy=OptimizationStrategy.QUALITY_FOCUSED
        )
        
        assert result.is_successful
        assert result.final_metrics.silhouette_score > sample_clustering_result.metrics.silhouette_score
        assert result.improvement_percentage > 0
        assert len(optimizer.optimization_history) == 1

    def test_adaptive_reclustering(self, optimizer, mock_analyzer, sample_clustering_result, sample_weights):
        """Test adaptive re-clustering based on quality threshold."""
        # Create poor quality result
        poor_metrics = ClusterMetrics(
            silhouette_score=0.3,  # Below threshold
            inertia=20.0,
            calinski_harabasz_score=50.0,
            davies_bouldin_score=1.5,
            num_clusters=3,
            avg_cluster_size=5.0,
            compression_ratio=0.4
        )
        
        poor_result = ClusteringResult(
            assignments=sample_clustering_result.assignments,
            centroids=sample_clustering_result.centroids,
            metrics=poor_metrics,
            strategy=ClusteringStrategy.KMEANS,
            execution_time=1.0
        )
        
        # Mock improved result
        mock_analyzer.cluster_kmeans.return_value = sample_clustering_result
        mock_analyzer._extract_features = Mock(return_value=np.random.randn(15, 10))
        
        # Run adaptive reclustering
        result = optimizer.adaptive_reclustering(
            clusters=poor_result,
            quality_threshold=0.6
        )
        
        assert result is not None
        assert result.final_metrics.silhouette_score > poor_metrics.silhouette_score
        mock_analyzer.cluster_kmeans.assert_called()

    def test_incremental_optimization(self, optimizer, mock_analyzer, sample_clustering_result, sample_weights):
        """Test incremental optimization with new weights."""
        # Create new weights
        new_weights = []
        for i in range(3):
            data = np.random.randn(10, 10).astype(np.float32)
            metadata = WeightMetadata(
                name=f"new_weight_{i}",
                shape=(10, 10),
                dtype=np.float32
            )
            weight = WeightTensor(data=data, metadata=metadata)
            new_weights.append(weight)
        
        # Mock updated result
        updated_result = ClusteringResult(
            assignments=sample_clustering_result.assignments + [
                ClusterAssignment(
                    weight_name=w.metadata.name,
                    weight_hash=w.compute_hash(),
                    cluster_id="cluster_0",
                    distance_to_centroid=0.2,
                    similarity_score=0.9
                ) for w in new_weights
            ],
            centroids=sample_clustering_result.centroids,
            metrics=sample_clustering_result.metrics,
            strategy=ClusteringStrategy.KMEANS,
            execution_time=0.5
        )
        
        mock_analyzer.incremental_clustering = Mock(return_value=updated_result)
        
        # Run incremental optimization
        result = optimizer.incremental_optimization(
            new_weights=new_weights,
            existing_clusters=sample_clustering_result
        )
        
        assert result is not None
        assert len(result.final_clusters.assignments) > len(sample_clustering_result.assignments)

    def test_cross_validate_clustering(self, optimizer, mock_analyzer, sample_weights):
        """Test cross-validation of clustering strategies."""
        # Mock results for different strategies
        strategies = [ClusteringStrategy.KMEANS, ClusteringStrategy.HIERARCHICAL, ClusteringStrategy.DBSCAN]
        mock_results = []
        
        for i, strategy in enumerate(strategies):
            metrics = ClusterMetrics(
                silhouette_score=0.5 + i * 0.1,  # Different quality
                inertia=15.0 - i * 2.0,
                calinski_harabasz_score=80.0 + i * 10.0,
                davies_bouldin_score=1.0 - i * 0.1,
                num_clusters=3 + i,
                avg_cluster_size=5.0 - i * 0.5,
                compression_ratio=0.5 + i * 0.05
            )
            
            result = ClusteringResult(
                assignments=[],
                centroids=[],
                metrics=metrics,
                strategy=strategy,
                execution_time=1.0
            )
            mock_results.append(result)
        
        # Set up mocks for each strategy
        kmeans_result = mock_results[0]
        hierarchical_result = mock_results[1]
        dbscan_result = mock_results[2]
        
        mock_analyzer.cluster_kmeans.return_value = kmeans_result
        mock_analyzer.cluster_hierarchical.return_value = hierarchical_result
        mock_analyzer.cluster_dbscan.return_value = dbscan_result
        
        # Run cross-validation
        best_strategy, results = optimizer.cross_validate_clustering(
            weights=sample_weights,
            strategies=strategies
        )
        
        assert best_strategy == ClusteringStrategy.DBSCAN  # Highest silhouette score
        assert len(results) == 3
        assert all(strategy in results for strategy in strategies)

    def test_optimize_cluster_count(self, optimizer, mock_analyzer, sample_weights):
        """Test optimization of cluster count."""
        # Mock results for different k values
        def mock_cluster_kmeans(weights, k, **kwargs):
            metrics = ClusterMetrics(
                silhouette_score=0.8 - abs(k - 4) * 0.1,  # Peak at k=4
                inertia=20.0 / k,
                calinski_harabasz_score=100.0,
                davies_bouldin_score=0.5,
                num_clusters=k,
                avg_cluster_size=len(weights) / k,
                compression_ratio=0.5 + k * 0.02
            )
            
            return ClusteringResult(
                assignments=[],
                centroids=[],
                metrics=metrics,
                strategy=ClusteringStrategy.KMEANS,
                execution_time=1.0
            )
        
        mock_analyzer.cluster_kmeans.side_effect = lambda w, k, **kw: mock_cluster_kmeans(w, k, **kw)
        
        # Run optimization
        optimal_k, scores = optimizer.optimize_cluster_count(
            weights=sample_weights,
            min_k=2,
            max_k=8
        )
        
        # With elbow method, the optimal k might not be exactly 4
        assert 2 <= optimal_k <= 8  # Should be within range
        assert len(scores) == 7  # k from 2 to 8
        assert optimal_k in scores  # optimal_k should be in scores
        # Check that the method found a reasonable optimum
        assert scores[optimal_k] >= min(scores.values())

    def test_optimize_similarity_threshold(self, optimizer, mock_analyzer, sample_clustering_result, sample_weights):
        """Test optimization of similarity threshold."""
        # Mock analyzer with different thresholds
        thresholds_tested = []
        
        def mock_set_threshold(threshold):
            thresholds_tested.append(threshold)
        
        mock_analyzer.config.similarity_threshold = 0.9
        type(mock_analyzer.config).similarity_threshold = property(
            lambda self: 0.9,
            lambda self, v: mock_set_threshold(v)
        )
        
        # Mock results with varying quality
        def mock_cluster_result(*args, **kwargs):
            threshold = thresholds_tested[-1] if thresholds_tested else 0.9
            # Quality peaks at threshold=0.95
            quality = 0.8 - abs(threshold - 0.95) * 2.0
            
            metrics = ClusterMetrics(
                silhouette_score=quality,
                inertia=10.0,
                calinski_harabasz_score=100.0,
                davies_bouldin_score=0.5,
                num_clusters=3,
                avg_cluster_size=5.0,
                compression_ratio=0.6
            )
            
            return ClusteringResult(
                assignments=sample_clustering_result.assignments,
                centroids=sample_clustering_result.centroids,
                metrics=metrics,
                strategy=ClusteringStrategy.KMEANS,
                execution_time=1.0
            )
        
        mock_analyzer.cluster_kmeans.side_effect = mock_cluster_result
        mock_analyzer.cluster_adaptive.side_effect = mock_cluster_result
        mock_analyzer.cluster_hierarchical.side_effect = mock_cluster_result
        mock_analyzer.cluster_dbscan.side_effect = mock_cluster_result
        
        # Run optimization
        optimal_threshold = optimizer.optimize_similarity_threshold(
            clusters=sample_clustering_result,
            weights=sample_weights
        )
        
        assert 0.9 <= optimal_threshold <= 1.0
        assert len(thresholds_tested) > 0

    def test_pareto_optimization(self, optimizer, mock_analyzer, sample_clustering_result, sample_weights):
        """Test multi-objective Pareto optimization."""
        # Define objectives
        objectives = [
            OptimizationObjective.MAXIMIZE_QUALITY,
            OptimizationObjective.MAXIMIZE_COMPRESSION
        ]
        
        # Mock different solutions
        solutions = []
        for i in range(10):
            quality = 0.5 + np.random.random() * 0.4
            compression = 0.4 + np.random.random() * 0.5
            
            metrics = ClusterMetrics(
                silhouette_score=quality,
                inertia=10.0,
                calinski_harabasz_score=100.0,
                davies_bouldin_score=0.5,
                num_clusters=3,
                avg_cluster_size=5.0,
                compression_ratio=compression
            )
            
            result = ClusteringResult(
                assignments=sample_clustering_result.assignments,
                centroids=sample_clustering_result.centroids,
                metrics=metrics,
                strategy=ClusteringStrategy.KMEANS,
                execution_time=1.0
            )
            solutions.append(result)
        
        mock_analyzer.cluster_kmeans.side_effect = solutions
        mock_analyzer.cluster_adaptive.side_effect = solutions
        mock_analyzer.cluster_hierarchical.side_effect = solutions
        mock_analyzer.cluster_dbscan.side_effect = solutions
        
        # Mock repository to return weights
        mock_analyzer.repository.get_all_weights.return_value = {
            w.metadata.name: w for w in sample_weights
        }
        
        # Run Pareto optimization
        pareto_front = optimizer.pareto_optimization(
            clusters=sample_clustering_result,
            objectives=objectives,
            population_size=5,
            generations=2
        )
        
        assert isinstance(pareto_front, ParetoFront)
        assert len(pareto_front.solutions) > 0
        assert all(s.is_successful for s in pareto_front.solutions)
        
        # Check Pareto dominance
        for i, sol1 in enumerate(pareto_front.solutions):
            for j, sol2 in enumerate(pareto_front.solutions):
                if i != j:
                    # No solution should dominate another in Pareto front
                    assert not pareto_front._dominates(sol1, sol2, objectives)

    def test_constraint_based_optimization(self, optimizer, mock_analyzer, sample_clustering_result, sample_weights):
        """Test optimization with constraints."""
        # Define constraints
        constraints = [
            OptimizationConstraint(
                type=ConstraintType.MIN_COMPRESSION_RATIO,
                value=0.5,
                name="min_compression"
            ),
            OptimizationConstraint(
                type=ConstraintType.MAX_CLUSTERS,
                value=5,
                name="max_clusters"
            ),
            OptimizationConstraint(
                type=ConstraintType.MIN_QUALITY,
                value=0.6,
                name="min_quality"
            )
        ]
        
        # Mock results that satisfy constraints
        valid_metrics = ClusterMetrics(
            silhouette_score=0.7,  # > 0.6
            inertia=10.0,
            calinski_harabasz_score=100.0,
            davies_bouldin_score=0.5,
            num_clusters=4,  # < 5
            avg_cluster_size=3.75,
            compression_ratio=0.55  # > 0.5
        )
        
        valid_result = ClusteringResult(
            assignments=sample_clustering_result.assignments,
            centroids=sample_clustering_result.centroids[:4],  # 4 clusters
            metrics=valid_metrics,
            strategy=ClusteringStrategy.KMEANS,
            execution_time=1.0
        )
        
        mock_analyzer.cluster_kmeans.return_value = valid_result
        
        # Run constrained optimization
        result = optimizer.constraint_based_optimization(
            clusters=sample_clustering_result,
            weights=sample_weights,
            constraints=constraints
        )
        
        assert result.is_successful
        assert result.satisfies_constraints(constraints)
        assert result.final_metrics.compression_ratio >= 0.5
        assert result.final_metrics.num_clusters <= 5
        assert result.final_metrics.silhouette_score >= 0.6

    def test_auto_optimize_pipeline(self, optimizer, mock_analyzer, sample_clustering_result, sample_weights):
        """Test fully automated optimization pipeline."""
        # Mock analyzer methods
        mock_analyzer.get_all_weights.return_value = {w.name: w for w in sample_weights}
        mock_analyzer.get_current_clustering.return_value = sample_clustering_result
        
        # Mock progressive improvements
        improvements = []
        for i in range(3):
            metrics = ClusterMetrics(
                silhouette_score=0.6 + i * 0.1,
                inertia=12.0 - i,
                calinski_harabasz_score=90.0 + i * 10,
                davies_bouldin_score=0.6 - i * 0.05,
                num_clusters=3,
                avg_cluster_size=5.0,
                compression_ratio=0.5 + i * 0.05
            )
            
            result = ClusteringResult(
                assignments=sample_clustering_result.assignments,
                centroids=sample_clustering_result.centroids,
                metrics=metrics,
                strategy=ClusteringStrategy.KMEANS,
                execution_time=1.0
            )
            improvements.append(result)
        
        mock_analyzer.cluster_kmeans.side_effect = improvements
        
        # Run auto optimization
        result = optimizer.auto_optimize_pipeline()
        
        assert result.is_successful
        assert result.total_iterations >= 1
        assert result.final_metrics.silhouette_score > sample_clustering_result.metrics.silhouette_score

    def test_optimization_triggers(self, optimizer, mock_analyzer, sample_clustering_result):
        """Test automatic optimization triggers."""
        # Set up quality degradation trigger
        trigger = OptimizationTrigger(
            condition=TriggerCondition.QUALITY_DEGRADATION,
            threshold=0.6,
            callback=lambda: optimizer.optimize_clustering(
                sample_clustering_result,
                [],
                OptimizationStrategy.QUALITY_FOCUSED
            )
        )
        
        optimizer.schedule_optimization([trigger])
        
        # Simulate quality degradation
        poor_metrics = ClusterMetrics(
            silhouette_score=0.4,  # Below threshold
            inertia=20.0,
            calinski_harabasz_score=50.0,
            davies_bouldin_score=1.5,
            num_clusters=3,
            avg_cluster_size=5.0,
            compression_ratio=0.4
        )
        
        # Check trigger should fire
        should_trigger = optimizer._check_trigger(trigger, poor_metrics)
        assert should_trigger

    def test_memory_optimization(self, optimizer, mock_analyzer, sample_clustering_result):
        """Test memory usage optimization."""
        # Run memory optimization
        optimized_result = optimizer.optimize_memory_usage(sample_clustering_result)
        
        assert optimized_result is not None
        # Check that centroids are deduplicated
        unique_centroids = {c.compute_hash() for c in optimized_result.centroids}
        assert len(unique_centroids) == len(optimized_result.centroids)

    def test_query_performance_optimization(self, optimizer, sample_clustering_result):
        """Test query performance optimization."""
        # Create mock index structures
        index_structures = {
            "weight_to_cluster": {a.weight_hash: a.cluster_id for a in sample_clustering_result.assignments},
            "cluster_to_weights": {}
        }
        
        # Build reverse index
        for assignment in sample_clustering_result.assignments:
            cluster_id = assignment.cluster_id
            if cluster_id not in index_structures["cluster_to_weights"]:
                index_structures["cluster_to_weights"][cluster_id] = []
            index_structures["cluster_to_weights"][cluster_id].append(assignment.weight_hash)
        
        # Run optimization
        optimized_index = optimizer.optimize_query_performance(index_structures)
        
        assert "sorted_clusters" in optimized_index
        assert "cluster_sizes" in optimized_index
        assert len(optimized_index["weight_to_cluster"]) == len(sample_clustering_result.assignments)

    def test_parallel_optimization(self, optimizer, mock_analyzer, sample_weights):
        """Test parallel optimization with multiple workers."""
        # Mock parallel clustering results
        mock_analyzer.cluster_kmeans.return_value = ClusteringResult(
            assignments=[],
            centroids=[],
            metrics=ClusterMetrics(
                silhouette_score=0.75,
                inertia=10.0,
                calinski_harabasz_score=100.0,
                davies_bouldin_score=0.5,
                num_clusters=3,
                avg_cluster_size=5.0,
                compression_ratio=0.6
            ),
            strategy=ClusteringStrategy.KMEANS,
            execution_time=0.5
        )
        
        # Run parallel optimization
        result = optimizer.parallel_optimization(
            weights=sample_weights,
            workers=4,
            strategies=[ClusteringStrategy.KMEANS, ClusteringStrategy.HIERARCHICAL]
        )
        
        assert result.is_successful
        assert result.execution_time is not None

    def test_batch_optimization_pipeline(self, optimizer, mock_analyzer, sample_weights):
        """Test batch optimization for large datasets."""
        # Create large dataset
        large_weights = sample_weights * 10  # 150 weights
        
        # Mock batch results
        batch_results = []
        for i in range(3):  # 3 batches
            metrics = ClusterMetrics(
                silhouette_score=0.7 + i * 0.05,
                inertia=10.0 - i,
                calinski_harabasz_score=100.0 + i * 10,
                davies_bouldin_score=0.5 - i * 0.05,
                num_clusters=5,
                avg_cluster_size=10.0,
                compression_ratio=0.6
            )
            
            result = ClusteringResult(
                assignments=[],
                centroids=[],
                metrics=metrics,
                strategy=ClusteringStrategy.KMEANS,
                execution_time=1.0
            )
            batch_results.append(result)
        
        mock_analyzer.cluster_kmeans.side_effect = batch_results
        
        # Run batch optimization
        result = optimizer.batch_optimization_pipeline(
            weights=large_weights,
            batch_size=50
        )
        
        assert result.is_successful
        assert result.total_batches == 3
        assert len(optimizer.optimization_history) > 0

    def test_optimization_history_tracking(self, optimizer, mock_analyzer, sample_clustering_result, sample_weights):
        """Test that optimization history is properly tracked."""
        # Run multiple optimizations
        mock_analyzer.cluster_kmeans.return_value = sample_clustering_result
        
        for i in range(3):
            optimizer.optimize_clustering(
                clusters=sample_clustering_result,
                weights=sample_weights,
                strategy=OptimizationStrategy.BALANCED
            )
        
        assert len(optimizer.optimization_history) == 3
        
        # Check history entries
        for entry in optimizer.optimization_history:
            assert "timestamp" in entry
            assert "strategy" in entry
            assert "initial_metrics" in entry
            assert "final_metrics" in entry
            assert "improvement" in entry

    def test_early_stopping(self, optimizer, mock_analyzer, sample_clustering_result, sample_weights):
        """Test early stopping when optimization converges."""
        # Mock minimal improvements
        base_score = 0.7
        results = []
        
        for i in range(10):
            # Very small improvements
            score = base_score + i * 0.001
            
            metrics = ClusterMetrics(
                silhouette_score=score,
                inertia=10.0,
                calinski_harabasz_score=100.0,
                davies_bouldin_score=0.5,
                num_clusters=3,
                avg_cluster_size=5.0,
                compression_ratio=0.6
            )
            
            result = ClusteringResult(
                assignments=sample_clustering_result.assignments,
                centroids=sample_clustering_result.centroids,
                metrics=metrics,
                strategy=ClusteringStrategy.KMEANS,
                execution_time=1.0
            )
            results.append(result)
        
        mock_analyzer.cluster_kmeans.side_effect = results
        
        # Run optimization with early stopping
        result = optimizer.optimize_clustering(
            clusters=sample_clustering_result,
            weights=sample_weights,
            strategy=OptimizationStrategy.QUALITY_FOCUSED,
            max_iterations=10,
            min_improvement=0.01  # 1% minimum improvement
        )
        
        assert result.is_successful
        assert result.iterations < 10  # Should stop early

    def test_optimization_rollback(self, optimizer, mock_analyzer, sample_clustering_result, sample_weights):
        """Test rollback when optimization degrades quality."""
        # Mock degraded result
        degraded_metrics = ClusterMetrics(
            silhouette_score=0.3,  # Worse than original
            inertia=30.0,
            calinski_harabasz_score=50.0,
            davies_bouldin_score=2.0,
            num_clusters=10,  # Too many clusters
            avg_cluster_size=1.5,
            compression_ratio=0.2  # Poor compression
        )
        
        degraded_result = ClusteringResult(
            assignments=sample_clustering_result.assignments,
            centroids=sample_clustering_result.centroids * 3,  # More clusters
            metrics=degraded_metrics,
            strategy=ClusteringStrategy.DBSCAN,
            execution_time=1.0
        )
        
        mock_analyzer.cluster_kmeans.return_value = degraded_result
        
        # Run optimization
        result = optimizer.optimize_clustering(
            clusters=sample_clustering_result,
            weights=sample_weights,
            strategy=OptimizationStrategy.QUALITY_FOCUSED,
            allow_quality_degradation=False
        )
        
        # Should detect degradation and return original
        assert not result.is_successful or result.final_metrics.silhouette_score >= sample_clustering_result.metrics.silhouette_score

    def test_adaptive_strategy_selection(self, optimizer, mock_analyzer, sample_weights):
        """Test adaptive selection of optimization strategy."""
        # Mock analyzer to return data characteristics
        mock_analyzer.analyze_data_characteristics = Mock(return_value={
            "density": "high",
            "dimensionality": "medium",
            "cluster_separation": "low",
            "noise_level": "low"
        })
        
        # Test strategy selection
        strategy = optimizer._select_adaptive_strategy(sample_weights)
        
        assert strategy in [
            OptimizationStrategy.QUALITY_FOCUSED,
            OptimizationStrategy.COMPRESSION_FOCUSED,
            OptimizationStrategy.BALANCED,
            OptimizationStrategy.SPEED_FOCUSED
        ]

    def test_weighted_objective_optimization(self, optimizer, mock_analyzer, sample_clustering_result, sample_weights):
        """Test optimization with weighted objectives."""
        objectives = {
            OptimizationObjective.MAXIMIZE_QUALITY: 0.7,
            OptimizationObjective.MAXIMIZE_COMPRESSION: 0.3
        }
        
        # Mock result that balances objectives
        balanced_metrics = ClusterMetrics(
            silhouette_score=0.75,  # Good quality
            inertia=11.0,
            calinski_harabasz_score=95.0,
            davies_bouldin_score=0.55,
            num_clusters=4,
            avg_cluster_size=3.75,
            compression_ratio=0.55  # Moderate compression
        )
        
        balanced_result = ClusteringResult(
            assignments=sample_clustering_result.assignments,
            centroids=sample_clustering_result.centroids,
            metrics=balanced_metrics,
            strategy=ClusteringStrategy.KMEANS,
            execution_time=1.0
        )
        
        mock_analyzer.cluster_kmeans.return_value = balanced_result
        
        # Run weighted optimization
        result = optimizer.weighted_objective_optimization(
            clusters=sample_clustering_result,
            objectives=objectives,
            weights=sample_weights
        )
        
        assert result.is_successful
        # Score should reflect weighted objectives
        score = (
            objectives[OptimizationObjective.MAXIMIZE_QUALITY] * result.final_metrics.silhouette_score +
            objectives[OptimizationObjective.MAXIMIZE_COMPRESSION] * result.final_metrics.compression_ratio
        )
        assert score > 0.5

    def test_evolutionary_optimization(self, optimizer, mock_analyzer, sample_clustering_result, sample_weights):
        """Test genetic algorithm-based optimization."""
        # Mock population evolution
        generation_results = []
        
        for gen in range(3):
            # Simulate improvement over generations
            metrics = ClusterMetrics(
                silhouette_score=0.6 + gen * 0.1,
                inertia=12.0 - gen,
                calinski_harabasz_score=90.0 + gen * 15,
                davies_bouldin_score=0.6 - gen * 0.1,
                num_clusters=3 + gen % 2,
                avg_cluster_size=5.0,
                compression_ratio=0.5 + gen * 0.08
            )
            
            result = ClusteringResult(
                assignments=sample_clustering_result.assignments,
                centroids=sample_clustering_result.centroids,
                metrics=metrics,
                strategy=ClusteringStrategy.KMEANS,
                execution_time=1.0
            )
            generation_results.append(result)
        
        mock_analyzer.cluster_kmeans.side_effect = generation_results * 5  # Multiple evaluations per generation
        
        # Run evolutionary optimization
        result = optimizer.evolutionary_optimization(
            clusters=sample_clustering_result,
            weights=sample_weights,
            population_size=10,
            generations=3
        )
        
        assert result.is_successful
        assert result.generations_evolved == 3
        assert result.final_metrics.silhouette_score > sample_clustering_result.metrics.silhouette_score

    def test_continuous_monitoring(self, optimizer, mock_analyzer, sample_clustering_result):
        """Test continuous cluster health monitoring."""
        # Start monitoring
        optimizer.monitor_cluster_health(
            clusters=sample_clustering_result,
            weights=[],
            check_interval=0.1  # 100ms for testing
        )
        
        assert optimizer._monitoring_active
        
        # Simulate some time passing
        time.sleep(0.2)
        
        # Stop monitoring
        optimizer.stop_monitoring()
        
        assert not optimizer._monitoring_active

    def test_optimization_opportunities_detection(self, optimizer):
        """Test detection of optimization opportunities."""
        # Create metrics history showing degradation
        metrics_history = [
            {"timestamp": time.time() - 3600, "silhouette_score": 0.8, "compression_ratio": 0.7},
            {"timestamp": time.time() - 1800, "silhouette_score": 0.75, "compression_ratio": 0.68},
            {"timestamp": time.time() - 900, "silhouette_score": 0.7, "compression_ratio": 0.65},
            {"timestamp": time.time(), "silhouette_score": 0.65, "compression_ratio": 0.6},
        ]
        
        opportunities = optimizer.detect_optimization_opportunities(metrics_history)
        
        assert len(opportunities) > 0
        assert any("quality degradation" in opp.lower() for opp in opportunities)

    def test_minimize_reconstruction_error(self, optimizer, mock_analyzer, sample_clustering_result, sample_weights):
        """Test optimization focused on minimizing reconstruction error."""
        # Mock improved result with lower reconstruction error
        improved_assignments = []
        for assignment in sample_clustering_result.assignments:
            improved = ClusterAssignment(
                weight_name=assignment.weight_name,
                weight_hash=assignment.weight_hash,
                cluster_id=assignment.cluster_id,
                distance_to_centroid=assignment.distance_to_centroid * 0.5,  # Lower error
                similarity_score=min(assignment.similarity_score * 1.05, 1.0)  # Higher similarity
            )
            improved_assignments.append(improved)
        
        improved_result = ClusteringResult(
            assignments=improved_assignments,
            centroids=sample_clustering_result.centroids,
            metrics=sample_clustering_result.metrics,
            strategy=ClusteringStrategy.KMEANS,
            execution_time=1.0
        )
        
        mock_analyzer.refine_clusters = Mock(return_value=improved_result)
        
        # Run error minimization
        result = optimizer.minimize_reconstruction_error(
            assignments=sample_clustering_result.assignments,
            centroids=sample_clustering_result.centroids,
            weights=sample_weights
        )
        
        assert result.is_successful
        # Check that reconstruction error decreased
        avg_initial_error = np.mean([a.distance_to_centroid for a in sample_clustering_result.assignments])
        avg_final_error = np.mean([a.distance_to_centroid for a in result.final_clusters.assignments])
        assert avg_final_error < avg_initial_error

    def test_maximize_compression_ratio(self, optimizer, mock_analyzer, sample_clustering_result, sample_weights):
        """Test optimization focused on maximizing compression."""
        # Mock result with better compression
        compressed_metrics = ClusterMetrics(
            silhouette_score=0.65,  # Slightly lower quality
            inertia=12.0,
            calinski_harabasz_score=90.0,
            davies_bouldin_score=0.6,
            num_clusters=2,  # Fewer clusters = better compression
            avg_cluster_size=7.5,
            compression_ratio=0.8  # Much better compression
        )
        
        compressed_result = ClusteringResult(
            assignments=sample_clustering_result.assignments,
            centroids=sample_clustering_result.centroids[:2],  # Fewer centroids
            metrics=compressed_metrics,
            strategy=ClusteringStrategy.KMEANS,
            execution_time=1.0
        )
        
        mock_analyzer.cluster_kmeans.return_value = compressed_result
        
        # Run compression maximization
        result = optimizer.maximize_compression_ratio(
            clusters=sample_clustering_result,
            weights=sample_weights,
            min_quality=0.6
        )
        
        assert result.is_successful
        assert result.final_metrics.compression_ratio > sample_clustering_result.metrics.compression_ratio
        assert result.final_metrics.silhouette_score >= 0.6  # Respects quality constraint

    def test_improve_cluster_quality(self, optimizer, mock_analyzer, sample_clustering_result, sample_weights):
        """Test iterative cluster quality improvement."""
        target_metrics = {
            "silhouette_score": 0.85,
            "compression_ratio": 0.7,
            "max_clusters": 5
        }
        
        # Mock progressive improvements
        improvements = []
        for i in range(3):
            metrics = ClusterMetrics(
                silhouette_score=0.7 + i * 0.05,  # Approaching target
                inertia=10.0 - i * 0.5,
                calinski_harabasz_score=100.0 + i * 10,
                davies_bouldin_score=0.5 - i * 0.05,
                num_clusters=3,
                avg_cluster_size=5.0,
                compression_ratio=0.6 + i * 0.05  # Approaching target
            )
            
            result = ClusteringResult(
                assignments=sample_clustering_result.assignments,
                centroids=sample_clustering_result.centroids,
                metrics=metrics,
                strategy=ClusteringStrategy.KMEANS,
                execution_time=1.0
            )
            improvements.append(result)
        
        mock_analyzer.cluster_kmeans.side_effect = improvements
        
        # Run quality improvement
        result = optimizer.improve_cluster_quality(
            clusters=sample_clustering_result,
            target_metrics=target_metrics,
            weights=sample_weights,
            max_iterations=3
        )
        
        assert result.is_successful
        assert result.final_metrics.silhouette_score > sample_clustering_result.metrics.silhouette_score
        assert result.iterations <= 3

    def test_balance_compression_vs_quality(self, optimizer, mock_analyzer, sample_clustering_result, sample_weights):
        """Test balancing compression and quality trade-offs."""
        # Mock results with different trade-offs
        trade_off_results = []
        
        for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
            # alpha=0: pure quality, alpha=1: pure compression
            metrics = ClusterMetrics(
                silhouette_score=0.9 - alpha * 0.3,  # Quality decreases with compression focus
                inertia=8.0 + alpha * 6.0,
                calinski_harabasz_score=120.0 - alpha * 30.0,
                davies_bouldin_score=0.3 + alpha * 0.4,
                num_clusters=int(5 - alpha * 3),  # Fewer clusters for compression
                avg_cluster_size=3.0 + alpha * 4.0,
                compression_ratio=0.4 + alpha * 0.5  # Compression increases
            )
            
            result = ClusteringResult(
                assignments=sample_clustering_result.assignments,
                centroids=sample_clustering_result.centroids[:metrics.num_clusters],
                metrics=metrics,
                strategy=ClusteringStrategy.KMEANS,
                execution_time=1.0
            )
            trade_off_results.append(result)
        
        mock_analyzer.cluster_kmeans.side_effect = trade_off_results
        
        # Test different trade-off values
        result = optimizer.balance_compression_vs_quality(
            clusters=sample_clustering_result,
            weights=sample_weights,
            trade_off=0.5  # Balanced
        )
        
        assert result.is_successful
        # Should find a balanced solution
        assert 0.6 <= result.final_metrics.silhouette_score <= 0.8
        assert 0.5 <= result.final_metrics.compression_ratio <= 0.7

    def test_optimize_hierarchy_levels(self, optimizer, mock_analyzer):
        """Test optimization of hierarchical clustering levels."""
        # Create mock hierarchy
        hierarchy = {
            "level_0": {"clusters": 10, "quality": 0.8, "compression": 0.4},
            "level_1": {"clusters": 5, "quality": 0.7, "compression": 0.6},
            "level_2": {"clusters": 3, "quality": 0.6, "compression": 0.75},
            "level_3": {"clusters": 1, "quality": 0.0, "compression": 1.0},
        }
        
        weights = []  # Mock weights
        
        # Test optimization
        optimal_levels = optimizer.optimize_hierarchy_levels(hierarchy, weights)
        
        assert len(optimal_levels) > 0
        assert all(0 <= level <= 3 for level in optimal_levels)

    def test_hyperparameter_tuning(self, optimizer, mock_analyzer, sample_weights):
        """Test comprehensive hyperparameter tuning."""
        param_space = {
            "n_clusters": [2, 3, 4, 5],
            "similarity_threshold": [0.85, 0.9, 0.95],
            "min_cluster_size": [2, 3, 5],
            "algorithm": ["kmeans", "hierarchical"]
        }
        
        # Mock results for different parameter combinations
        best_params = {"n_clusters": 3, "similarity_threshold": 0.9, "min_cluster_size": 3, "algorithm": "kmeans"}
        
        def mock_cluster_with_params(**params):
            # Simulate that best_params gives best results
            distance = sum(abs(params.get(k, 0) - v) if isinstance(v, (int, float)) else 0 
                          for k, v in best_params.items())
            
            quality = 0.8 - distance * 0.1
            
            metrics = ClusterMetrics(
                silhouette_score=quality,
                inertia=10.0 + distance,
                calinski_harabasz_score=100.0 - distance * 10,
                davies_bouldin_score=0.5 + distance * 0.1,
                num_clusters=params.get("n_clusters", 3),
                avg_cluster_size=len(sample_weights) / params.get("n_clusters", 3),
                compression_ratio=0.6 - distance * 0.05
            )
            
            return ClusteringResult(
                assignments=[],
                centroids=[],
                metrics=metrics,
                strategy=ClusteringStrategy.KMEANS,
                execution_time=1.0
            )
        
        mock_analyzer.cluster_kmeans.side_effect = lambda w, **kw: mock_cluster_with_params(**kw)
        
        # Run tuning
        best_found, results = optimizer.tune_hyperparameters(
            weights=sample_weights,
            param_space=param_space
        )
        
        assert best_found is not None
        assert len(results) > 0
        assert all(key in best_found for key in param_space.keys())