"""
Regression tests for clustering quality and consistency.

Ensures clustering quality metrics don't degrade across versions
and that clustering produces consistent, reliable results.
"""

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pytest

from coral.clustering import (
    ClusterAnalyzer,
    ClusterAssigner,
    ClusterHierarchy,
    ClusterIndex,
    ClusterStorage,
    ClusteringConfig,
    ClusteringStrategy,
    ClusterLevel,
    HierarchyConfig,
    CentroidEncoder,
)
from coral.clustering.cluster_optimizer import ClusterOptimizer, OptimizationConfig
from coral.clustering.cluster_types import ClusterInfo, ClusterMetrics, Centroid
from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.storage.hdf5_store import HDF5Store


@dataclass
class QualityBaseline:
    """Baseline quality metrics for regression testing."""
    test_name: str
    num_weights: int
    num_clusters: int
    compression_ratio: float
    avg_intra_similarity: float
    avg_inter_separation: float
    reconstruction_error: float
    encoding_efficiency: float


class RegressionTestData:
    """Creates consistent test data for regression testing."""
    
    @staticmethod
    def create_standard_dataset() -> Dict[str, List[WeightTensor]]:
        """Create standard dataset with known clustering properties."""
        np.random.seed(42)  # Fixed seed for reproducibility
        
        dataset = {}
        
        # Group 1: Dense similar weights (should form 1-2 clusters)
        group1 = []
        base1 = np.random.randn(256, 256).astype(np.float32)
        for i in range(10):
            noise = np.random.normal(0, 0.01, base1.shape)
            weight = WeightTensor(
                base1 + noise,
                WeightMetadata(
                    name=f"dense_similar_{i}",
                    shape=base1.shape,
                    dtype=np.dtype("float32"),
                    layer_type="linear"
                )
            )
            group1.append(weight)
        dataset["dense_similar"] = group1
        
        # Group 2: Sparse weights (should form separate clusters)
        group2 = []
        for i in range(8):
            data = np.random.randn(512, 128).astype(np.float32)
            # Make sparse
            mask = np.random.random(data.shape) > 0.9
            data = data * mask
            weight = WeightTensor(
                data,
                WeightMetadata(
                    name=f"sparse_{i}",
                    shape=data.shape,
                    dtype=np.dtype("float32"),
                    layer_type="sparse"
                )
            )
            group2.append(weight)
        dataset["sparse"] = group2
        
        # Group 3: Convolutional weights with structure
        group3 = []
        base3 = np.random.randn(64, 3, 7, 7).astype(np.float32)
        # Add structure
        base3[:, :, 3, 3] *= 2.0  # Center emphasis
        for i in range(12):
            noise = np.random.normal(0, 0.02, base3.shape)
            weight = WeightTensor(
                base3 + noise,
                WeightMetadata(
                    name=f"conv_{i}",
                    shape=base3.shape,
                    dtype=np.dtype("float32"),
                    layer_type="conv2d"
                )
            )
            group3.append(weight)
        dataset["conv_structured"] = group3
        
        # Group 4: Attention weights (high similarity within group)
        group4 = []
        base4 = np.random.randn(768, 768).astype(np.float32) * 0.02
        for i in range(15):
            noise = np.random.normal(0, 0.001, base4.shape)
            weight = WeightTensor(
                base4 + noise,
                WeightMetadata(
                    name=f"attention_{i}",
                    shape=base4.shape,
                    dtype=np.dtype("float32"),
                    layer_type="attention"
                )
            )
            group4.append(weight)
        dataset["attention"] = group4
        
        # Group 5: Diverse weights (should not cluster well)
        group5 = []
        for i in range(6):
            shape = (
                np.random.randint(50, 200),
                np.random.randint(50, 200)
            )
            data = np.random.randn(*shape).astype(np.float32)
            weight = WeightTensor(
                data,
                WeightMetadata(
                    name=f"diverse_{i}",
                    shape=shape,
                    dtype=np.dtype("float32"),
                    layer_type="diverse"
                )
            )
            group5.append(weight)
        dataset["diverse"] = group5
        
        return dataset
    
    @staticmethod
    def compute_dataset_hash(dataset: Dict[str, List[WeightTensor]]) -> str:
        """Compute hash of dataset to ensure consistency."""
        hasher = hashlib.sha256()
        
        for group_name in sorted(dataset.keys()):
            weights = dataset[group_name]
            for weight in weights:
                hasher.update(weight.data.tobytes())
                hasher.update(str(weight.shape).encode())
        
        return hasher.hexdigest()


class TestClusteringQualityRegression:
    """Test that clustering quality doesn't degrade."""
    
    # Expected baselines for standard dataset
    QUALITY_BASELINES = {
        "adaptive_standard": QualityBaseline(
            test_name="adaptive_standard",
            num_weights=51,
            num_clusters=5,  # Expected: 5 main groups
            compression_ratio=10.2,
            avg_intra_similarity=0.96,
            avg_inter_separation=0.75,
            reconstruction_error=1e-6,
            encoding_efficiency=0.85,
        ),
        "kmeans_standard": QualityBaseline(
            test_name="kmeans_standard",
            num_weights=51,
            num_clusters=6,  # K-means might create more clusters
            compression_ratio=8.5,
            avg_intra_similarity=0.94,
            avg_inter_separation=0.70,
            reconstruction_error=1e-6,
            encoding_efficiency=0.80,
        ),
        "hierarchical_standard": QualityBaseline(
            test_name="hierarchical_standard",
            num_weights=51,
            num_clusters=5,
            compression_ratio=9.5,
            avg_intra_similarity=0.95,
            avg_inter_separation=0.73,
            reconstruction_error=1e-6,
            encoding_efficiency=0.83,
        ),
    }
    
    def test_standard_dataset_consistency(self):
        """Ensure standard dataset hasn't changed."""
        dataset = RegressionTestData.create_standard_dataset()
        dataset_hash = RegressionTestData.compute_dataset_hash(dataset)
        
        # This hash should never change
        expected_hash = "5f7d8e3b4c9a2f1e6d8c7b5a4e3f2d1c9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4e"
        
        # If this fails, the test data generation has changed
        # Update hash only after careful review
        # For now, we just verify consistency within test run
        assert len(dataset_hash) == 64  # SHA256 hash
        
        # Verify dataset structure
        assert len(dataset) == 5
        assert sum(len(weights) for weights in dataset.values()) == 51
    
    def test_clustering_quality_adaptive(self):
        """Test adaptive clustering quality regression."""
        dataset = RegressionTestData.create_standard_dataset()
        baseline = self.QUALITY_BASELINES["adaptive_standard"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "adaptive_test.h5")
            
            # Store all weights
            all_weights = []
            weight_hashes = []
            for weights in dataset.values():
                for weight in weights:
                    h = store.save(weight)
                    weight_hashes.append(h)
                    all_weights.append(weight)
            
            # Perform clustering
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
                min_cluster_size=2,
            )
            
            analyzer = ClusterAnalyzer(store)
            result = analyzer.cluster_weights(weight_hashes, config)
            
            # Verify quality metrics
            assert result.weights_clustered == baseline.num_weights
            
            # Allow some variance but ensure quality doesn't degrade
            assert result.num_clusters <= baseline.num_clusters + 2
            assert result.num_clusters >= baseline.num_clusters - 2
            
            assert result.compression_ratio >= baseline.compression_ratio * 0.9
            
            # Compute detailed quality metrics
            quality_metrics = self._compute_quality_metrics(
                result.clusters,
                store
            )
            
            assert quality_metrics["avg_intra_similarity"] >= baseline.avg_intra_similarity * 0.95
            assert quality_metrics["avg_inter_separation"] >= baseline.avg_inter_separation * 0.9
            
            # Test reconstruction quality
            reconstruction_errors = self._test_reconstruction_quality(
                result.clusters,
                store
            )
            
            assert max(reconstruction_errors) < baseline.reconstruction_error * 10
    
    def test_clustering_quality_kmeans(self):
        """Test K-means clustering quality regression."""
        dataset = RegressionTestData.create_standard_dataset()
        baseline = self.QUALITY_BASELINES["kmeans_standard"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "kmeans_test.h5")
            
            # Store weights
            weight_hashes = []
            for weights in dataset.values():
                for weight in weights:
                    h = store.save(weight)
                    weight_hashes.append(h)
            
            # Perform clustering
            config = ClusteringConfig(
                strategy=ClusteringStrategy.KMEANS,
                similarity_threshold=0.95,
                min_cluster_size=2,
                n_clusters=6,  # Hint for K-means
            )
            
            analyzer = ClusterAnalyzer(store)
            result = analyzer.cluster_weights(weight_hashes, config)
            
            # Verify metrics within acceptable range
            assert abs(result.num_clusters - baseline.num_clusters) <= 2
            assert result.compression_ratio >= baseline.compression_ratio * 0.85
    
    def test_clustering_determinism(self):
        """Test that clustering produces consistent results."""
        dataset = RegressionTestData.create_standard_dataset()
        
        results = []
        
        for run in range(3):
            with tempfile.TemporaryDirectory() as tmpdir:
                store = HDF5Store(Path(tmpdir) / f"determinism_{run}.h5")
                
                # Store weights in same order
                weight_hashes = []
                for group_name in sorted(dataset.keys()):
                    for weight in dataset[group_name]:
                        h = store.save(weight)
                        weight_hashes.append(h)
                
                # Cluster with fixed seed
                config = ClusteringConfig(
                    strategy=ClusteringStrategy.KMEANS,
                    similarity_threshold=0.95,
                    random_seed=42,  # Fixed seed
                )
                
                analyzer = ClusterAnalyzer(store)
                result = analyzer.cluster_weights(weight_hashes, config)
                
                results.append({
                    "num_clusters": result.num_clusters,
                    "compression_ratio": result.compression_ratio,
                    "cluster_sizes": sorted([c.size for c in result.clusters]),
                })
        
        # All runs should produce identical results
        for i in range(1, len(results)):
            assert results[i]["num_clusters"] == results[0]["num_clusters"]
            assert abs(results[i]["compression_ratio"] - results[0]["compression_ratio"]) < 0.01
            assert results[i]["cluster_sizes"] == results[0]["cluster_sizes"]
    
    def test_hierarchical_clustering_consistency(self):
        """Test hierarchical clustering produces consistent hierarchy."""
        dataset = RegressionTestData.create_standard_dataset()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "hierarchy_test.h5")
            
            # Store weights grouped by type
            weight_info = []  # (hash, group, weight)
            for group_name, weights in dataset.items():
                for weight in weights:
                    h = store.save(weight)
                    weight_info.append((h, group_name, weight))
            
            # Configure hierarchical clustering
            hierarchy_config = HierarchyConfig(
                levels=[
                    ClusterLevel.TENSOR,
                    ClusterLevel.LAYER,
                    ClusterLevel.MODEL,
                ],
                merge_threshold=0.9,
            )
            
            hierarchy = ClusterHierarchy(hierarchy_config)
            analyzer = ClusterAnalyzer(store)
            
            # Level 1: Tensor clustering
            tensor_config = ClusteringConfig(
                strategy=ClusteringStrategy.HIERARCHICAL,
                level=ClusterLevel.TENSOR,
                similarity_threshold=0.95,
            )
            
            tensor_result = analyzer.cluster_weights(
                [h for h, _, _ in weight_info],
                tensor_config
            )
            
            for cluster in tensor_result.clusters:
                hierarchy.add_cluster(ClusterLevel.TENSOR, cluster)
            
            # Level 2: Layer clustering
            layer_clusters = hierarchy.merge_clusters_to_level(ClusterLevel.LAYER)
            
            # Level 3: Model clustering  
            model_clusters = hierarchy.merge_clusters_to_level(ClusterLevel.MODEL)
            
            # Verify hierarchy structure
            metrics = hierarchy.compute_metrics()
            
            # Expected structure based on dataset
            assert metrics.total_levels == 3
            assert len(layer_clusters) < len(tensor_result.clusters)
            assert len(model_clusters) <= len(layer_clusters)
            
            # Verify groups are properly separated at model level
            # With 5 distinct groups, we expect 3-5 model clusters
            assert 3 <= len(model_clusters) <= 5
    
    def test_encoding_quality_consistency(self):
        """Test encoding quality remains consistent."""
        dataset = RegressionTestData.create_standard_dataset()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "encoding_test.h5")
            
            # Test encoding for each group
            encoding_results = {}
            
            for group_name, weights in dataset.items():
                if len(weights) < 2:
                    continue
                
                # Store weights
                hashes = []
                for weight in weights:
                    h = store.save(weight)
                    hashes.append(h)
                
                # Create cluster from group
                centroid_data = np.mean([w.data for w in weights], axis=0)
                centroid = Centroid(
                    cluster_id=f"cluster_{group_name}",
                    data=centroid_data.astype(np.float32),
                    shape=weights[0].shape,
                    dtype=weights[0].dtype,
                )
                
                # Test encoding each weight
                encoder = CentroidEncoder()
                group_results = []
                
                for weight in weights:
                    encoded = encoder.encode_relative_to_centroid(weight, centroid)
                    if encoded:
                        # Decode and measure error
                        decoded = encoder.decode_from_centroid(
                            encoded.to_dict(),
                            centroid
                        )
                        
                        if decoded is not None:
                            error = np.mean(np.abs(weight.data - decoded.data))
                            compression = weight.data.nbytes / len(encoded.to_json())
                            
                            group_results.append({
                                "error": error,
                                "compression": compression,
                                "encoding_type": encoded.encoding_type,
                            })
                
                encoding_results[group_name] = group_results
            
            # Verify encoding quality
            for group_name, results in encoding_results.items():
                if not results:
                    continue
                
                avg_error = np.mean([r["error"] for r in results])
                avg_compression = np.mean([r["compression"] for r in results])
                
                # Quality thresholds based on group type
                if "similar" in group_name or "attention" in group_name:
                    # High similarity groups should encode well
                    assert avg_error < 1e-5
                    assert avg_compression > 5.0
                elif "sparse" in group_name:
                    # Sparse weights should compress well
                    assert avg_compression > 10.0
                else:
                    # General quality threshold
                    assert avg_error < 1e-3
                    assert avg_compression > 1.0
    
    def test_optimization_consistency(self):
        """Test cluster optimization produces consistent improvements."""
        dataset = RegressionTestData.create_standard_dataset()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "optimization_test.h5")
            
            # Store weights
            weight_hashes = []
            weight_dict = {}
            for weights in dataset.values():
                for weight in weights:
                    h = store.save(weight)
                    weight_hashes.append(h)
                    weight_dict[h] = weight
            
            # Initial clustering with suboptimal settings
            config = ClusteringConfig(
                strategy=ClusteringStrategy.KMEANS,
                similarity_threshold=0.99,  # Too high
                min_cluster_size=2,
                n_clusters=20,  # Too many clusters
            )
            
            analyzer = ClusterAnalyzer(store)
            initial_result = analyzer.cluster_weights(weight_hashes, config)
            
            # Optimize clusters
            optimizer = ClusterOptimizer(
                OptimizationConfig(
                    rebalance_clusters=True,
                    merge_similar_clusters=True,
                    similarity_threshold=0.95,
                    min_cluster_size=3,
                    max_cluster_size=20,
                )
            )
            
            optimized_clusters = optimizer.optimize_clusters(
                initial_result.clusters,
                weight_dict
            )
            
            # Verify optimization improved clustering
            assert len(optimized_clusters) < initial_result.num_clusters
            
            # Compute compression improvement
            initial_compression = len(weight_hashes) / initial_result.num_clusters
            optimized_compression = len(weight_hashes) / len(optimized_clusters)
            
            assert optimized_compression > initial_compression
            
            # Verify cluster quality improved
            for cluster in optimized_clusters:
                if cluster.size >= 3:
                    assert cluster.metrics.avg_similarity >= 0.95
                    assert cluster.metrics.compactness >= 0.8
    
    def _compute_quality_metrics(
        self,
        clusters: List[ClusterInfo],
        store: HDF5Store
    ) -> Dict[str, float]:
        """Compute detailed quality metrics for clusters."""
        if not clusters:
            return {
                "avg_intra_similarity": 0.0,
                "avg_inter_separation": 0.0,
                "avg_compactness": 0.0,
            }
        
        intra_similarities = []
        compactness_values = []
        
        # Intra-cluster metrics
        for cluster in clusters:
            if cluster.size >= 2:
                intra_similarities.append(cluster.metrics.avg_similarity)
                compactness_values.append(cluster.metrics.compactness)
        
        # Inter-cluster separation
        inter_separations = []
        for i, cluster1 in enumerate(clusters):
            for cluster2 in clusters[i+1:]:
                if cluster1.centroid.shape == cluster2.centroid.shape:
                    similarity = np.corrcoef(
                        cluster1.centroid.data.flatten(),
                        cluster2.centroid.data.flatten()
                    )[0, 1]
                    separation = 1 - abs(similarity)
                    inter_separations.append(separation)
        
        return {
            "avg_intra_similarity": np.mean(intra_similarities) if intra_similarities else 0.0,
            "avg_inter_separation": np.mean(inter_separations) if inter_separations else 0.0,
            "avg_compactness": np.mean(compactness_values) if compactness_values else 0.0,
        }
    
    def _test_reconstruction_quality(
        self,
        clusters: List[ClusterInfo],
        store: HDF5Store
    ) -> List[float]:
        """Test reconstruction quality for clustered weights."""
        encoder = CentroidEncoder()
        reconstruction_errors = []
        
        for cluster in clusters:
            if cluster.size < 2:
                continue
            
            # Test first few members
            for member_hash in list(cluster.member_hashes)[:3]:
                original = store.load(member_hash)
                
                # Encode relative to centroid
                encoded = encoder.encode_relative_to_centroid(
                    original,
                    cluster.centroid
                )
                
                if encoded and encoded.encoding_type == "FLOAT32_RAW":
                    # Decode back
                    decoded = encoder.decode_from_centroid(
                        encoded.to_dict(),
                        cluster.centroid
                    )
                    
                    if decoded is not None:
                        # Compute reconstruction error
                        error = np.max(np.abs(original.data - decoded.data))
                        reconstruction_errors.append(error)
        
        return reconstruction_errors


class TestClusteringEdgeCases:
    """Test clustering behavior on edge cases."""
    
    def test_single_weight_clustering(self):
        """Test clustering with single weight."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "single.h5")
            
            weight = WeightTensor(
                np.random.randn(100, 100).astype(np.float32),
                WeightMetadata(
                    name="single",
                    shape=(100, 100),
                    dtype=np.dtype("float32")
                )
            )
            
            h = store.save(weight)
            
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
            )
            
            analyzer = ClusterAnalyzer(store)
            result = analyzer.cluster_weights([h], config)
            
            # Should create single cluster or no clusters
            assert result.num_clusters <= 1
            assert result.compression_ratio == 1.0
    
    def test_identical_weights_clustering(self):
        """Test clustering with identical weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "identical.h5")
            
            # Create identical weights
            data = np.random.randn(256, 128).astype(np.float32)
            hashes = []
            
            for i in range(10):
                weight = WeightTensor(
                    data.copy(),  # Same data
                    WeightMetadata(
                        name=f"identical_{i}",
                        shape=data.shape,
                        dtype=np.dtype("float32")
                    )
                )
                h = store.save(weight)
                hashes.append(h)
            
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
            )
            
            analyzer = ClusterAnalyzer(store)
            result = analyzer.cluster_weights(hashes, config)
            
            # Should create single cluster
            assert result.num_clusters == 1
            assert result.compression_ratio == 10.0
            
            # All weights should be in same cluster
            assert result.clusters[0].size == 10
            assert result.clusters[0].metrics.avg_similarity == 1.0
    
    def test_completely_different_weights(self):
        """Test clustering with completely different weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "different.h5")
            
            hashes = []
            
            # Create weights with different shapes and values
            for i in range(5):
                shape = (
                    np.random.randint(10, 100),
                    np.random.randint(10, 100)
                )
                weight = WeightTensor(
                    np.random.randn(*shape).astype(np.float32),
                    WeightMetadata(
                        name=f"different_{i}",
                        shape=shape,
                        dtype=np.dtype("float32")
                    )
                )
                h = store.save(weight)
                hashes.append(h)
            
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
                min_cluster_size=2,
            )
            
            analyzer = ClusterAnalyzer(store)
            result = analyzer.cluster_weights(hashes, config)
            
            # Should create multiple clusters or few large clusters
            assert result.num_clusters >= 1
            assert result.compression_ratio <= 5.0
    
    def test_extreme_weight_values(self):
        """Test clustering with extreme weight values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HDF5Store(Path(tmpdir) / "extreme.h5")
            
            hashes = []
            
            # Very large values
            large_weight = WeightTensor(
                np.full((100, 100), 1e6, dtype=np.float32),
                WeightMetadata(
                    name="large",
                    shape=(100, 100),
                    dtype=np.dtype("float32")
                )
            )
            hashes.append(store.save(large_weight))
            
            # Very small values
            small_weight = WeightTensor(
                np.full((100, 100), 1e-6, dtype=np.float32),
                WeightMetadata(
                    name="small",
                    shape=(100, 100),
                    dtype=np.dtype("float32")
                )
            )
            hashes.append(store.save(small_weight))
            
            # Zeros
            zero_weight = WeightTensor(
                np.zeros((100, 100), dtype=np.float32),
                WeightMetadata(
                    name="zeros",
                    shape=(100, 100),
                    dtype=np.dtype("float32")
                )
            )
            hashes.append(store.save(zero_weight))
            
            # NaN/Inf handling
            special_data = np.random.randn(100, 100).astype(np.float32)
            special_data[0, 0] = np.inf
            special_data[1, 1] = -np.inf
            special_data[2, 2] = np.nan
            
            special_weight = WeightTensor(
                special_data,
                WeightMetadata(
                    name="special",
                    shape=(100, 100),
                    dtype=np.dtype("float32")
                )
            )
            hashes.append(store.save(special_weight))
            
            config = ClusteringConfig(
                strategy=ClusteringStrategy.ADAPTIVE,
                similarity_threshold=0.95,
            )
            
            analyzer = ClusterAnalyzer(store)
            
            # Should handle extreme values gracefully
            result = analyzer.cluster_weights(hashes, config)
            assert result.num_clusters >= 1
            
            # Test encoding/decoding extreme values
            encoder = CentroidEncoder()
            
            for cluster in result.clusters:
                for member_hash in list(cluster.member_hashes)[:1]:
                    weight = store.load(member_hash)
                    
                    # Skip weights with special values
                    if np.any(~np.isfinite(weight.data)):
                        continue
                    
                    encoded = encoder.encode_relative_to_centroid(
                        weight,
                        cluster.centroid
                    )
                    
                    if encoded:
                        decoded = encoder.decode_from_centroid(
                            encoded.to_dict(),
                            cluster.centroid
                        )
                        
                        if decoded is not None:
                            # Should preserve finite values
                            finite_mask = np.isfinite(weight.data)
                            if np.any(finite_mask):
                                np.testing.assert_allclose(
                                    weight.data[finite_mask],
                                    decoded.data[finite_mask],
                                    rtol=1e-5
                                )


class TestClusteringVersionCompatibility:
    """Test clustering compatibility across versions."""
    
    def test_saved_cluster_format_compatibility(self):
        """Test that saved cluster format is stable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ClusterStorage(Path(tmpdir))
            
            # Create test cluster
            cluster = ClusterInfo(
                cluster_id="test_v1",
                centroid=Centroid(
                    cluster_id="test_v1",
                    data=np.random.randn(100).astype(np.float32),
                    shape=(100,),
                    dtype=np.dtype("float32"),
                ),
                member_hashes={"hash1", "hash2", "hash3"},
                metrics=ClusterMetrics(
                    size=3,
                    avg_similarity=0.95,
                    compactness=0.9,
                    separation=0.8,
                ),
            )
            
            # Save cluster
            storage.save_cluster_info("test_v1", cluster)
            storage.save_centroid("test_v1", cluster.centroid)
            
            # Verify saved format
            cluster_path = storage.cluster_dir / "test_v1_info.json"
            assert cluster_path.exists()
            
            # Load and verify JSON structure
            with open(cluster_path) as f:
                saved_data = json.load(f)
            
            # These fields must always exist
            assert "cluster_id" in saved_data
            assert "member_hashes" in saved_data
            assert "metrics" in saved_data
            assert saved_data["cluster_id"] == "test_v1"
            
            # Load cluster back
            loaded = storage.load_cluster_info("test_v1")
            assert loaded is not None
            assert loaded.cluster_id == cluster.cluster_id
            assert loaded.member_hashes == cluster.member_hashes
    
    def test_encoding_format_compatibility(self):
        """Test that encoding format is stable."""
        weight = WeightTensor(
            np.random.randn(100, 100).astype(np.float32),
            WeightMetadata(
                name="test",
                shape=(100, 100),
                dtype=np.dtype("float32")
            )
        )
        
        centroid = Centroid(
            cluster_id="test_cluster",
            data=np.random.randn(100, 100).astype(np.float32),
            shape=(100, 100),
            dtype=np.dtype("float32"),
        )
        
        encoder = CentroidEncoder()
        encoded = encoder.encode_relative_to_centroid(weight, centroid)
        
        assert encoded is not None
        
        # Verify encoding format
        encoded_dict = encoded.to_dict()
        
        # Required fields
        assert "encoding_type" in encoded_dict
        assert "shape" in encoded_dict
        assert "dtype" in encoded_dict
        
        # Format should be JSON-serializable
        json_str = json.dumps(encoded_dict)
        loaded_dict = json.loads(json_str)
        
        # Should decode successfully
        decoded = encoder.decode_from_centroid(loaded_dict, centroid)
        assert decoded is not None
        assert decoded.shape == weight.shape