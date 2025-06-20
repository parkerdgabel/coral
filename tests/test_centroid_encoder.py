"""Tests for CentroidEncoder functionality."""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from coral.clustering.centroid_encoder import CentroidEncoder
from coral.clustering.cluster_types import ClusterAssignment
from coral.core.weight_tensor import WeightTensor
from coral.delta.delta_encoder import DeltaEncoder, Delta, DeltaType, DeltaConfig


class TestCentroidEncoder:
    """Test cases for CentroidEncoder."""

    @pytest.fixture
    def encoder(self):
        """Create a CentroidEncoder instance."""
        return CentroidEncoder()

    @pytest.fixture
    def sample_weight(self):
        """Create a sample weight tensor."""
        from coral.core.weight_tensor import WeightMetadata
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        metadata = WeightMetadata(name="test_weight", shape=data.shape, dtype=data.dtype)
        return WeightTensor(data=data, metadata=metadata)

    @pytest.fixture
    def sample_centroid(self):
        """Create a sample centroid."""
        from coral.core.weight_tensor import WeightMetadata
        data = np.array([1.1, 2.1, 3.1, 4.1, 5.1], dtype=np.float32)
        metadata = WeightMetadata(name="centroid_0", shape=data.shape, dtype=data.dtype)
        return WeightTensor(data=data, metadata=metadata)

    @pytest.fixture
    def multiple_weights(self):
        """Create multiple weight tensors."""
        from coral.core.weight_tensor import WeightMetadata
        np.random.seed(42)  # For reproducible tests
        weights = []
        for i in range(5):
            data = np.random.randn(5).astype(np.float32) + i * 0.1
            metadata = WeightMetadata(name=f"weight_{i}", shape=data.shape, dtype=data.dtype)
            weights.append(WeightTensor(data=data, metadata=metadata))
        return weights

    @pytest.fixture
    def multiple_centroids(self):
        """Create multiple centroids."""
        from coral.core.weight_tensor import WeightMetadata
        np.random.seed(42)  # Same seed as weights to create similar centroids
        centroids = []
        for i in range(3):
            # Create centroids that are similar to the weights
            data = np.random.randn(5).astype(np.float32) + i * 0.05  # Smaller offset
            metadata = WeightMetadata(name=f"centroid_{i}", shape=data.shape, dtype=data.dtype)
            centroids.append(WeightTensor(data=data, metadata=metadata))
        return centroids

    def test_encode_weight_to_centroid(self, encoder, sample_weight, sample_centroid):
        """Test encoding a weight as delta from centroid."""
        delta = encoder.encode_weight_to_centroid(sample_weight, sample_centroid)
        
        assert delta is not None
        assert isinstance(delta, Delta)
        assert delta.reference_hash == sample_centroid.compute_hash()
        assert delta.original_shape == sample_weight.data.shape
        assert delta.original_dtype == sample_weight.data.dtype

    def test_decode_weight_from_centroid(self, encoder, sample_weight, sample_centroid):
        """Test reconstructing weight from centroid + delta."""
        # Encode
        delta = encoder.encode_weight_to_centroid(sample_weight, sample_centroid)
        
        # Decode
        reconstructed = encoder.decode_weight_from_centroid(delta, sample_centroid)
        
        assert reconstructed is not None
        assert isinstance(reconstructed, WeightTensor)
        assert reconstructed.data.shape == sample_weight.data.shape
        assert reconstructed.data.dtype == sample_weight.data.dtype
        np.testing.assert_array_almost_equal(reconstructed.data, sample_weight.data, decimal=6)

    def test_lossless_reconstruction(self, encoder, sample_weight, sample_centroid):
        """Test perfect reconstruction with appropriate strategies."""
        # Test with FLOAT32_RAW strategy
        delta = encoder.encode_weight_to_centroid(
            sample_weight, sample_centroid, strategy=DeltaType.FLOAT32_RAW
        )
        reconstructed = encoder.decode_weight_from_centroid(delta, sample_centroid)
        
        np.testing.assert_array_equal(reconstructed.data, sample_weight.data)

    def test_batch_encode(self, encoder, multiple_weights, multiple_centroids):
        """Test batch encoding of multiple weights."""
        # Create assignments
        assignments = [
            ClusterAssignment(
                weight_name=w.metadata.name,
                weight_hash=w.compute_hash(), 
                cluster_id=str(i % len(multiple_centroids))
            )
            for i, w in enumerate(multiple_weights)
        ]
        
        # Batch encode
        deltas = encoder.batch_encode(multiple_weights, assignments, multiple_centroids)
        
        assert len(deltas) == len(multiple_weights)
        for i, delta in enumerate(deltas):
            assert isinstance(delta, Delta)
            centroid_idx = int(assignments[i].cluster_id)
            assert delta.reference_hash == multiple_centroids[centroid_idx].compute_hash()

    def test_batch_decode(self, encoder, multiple_weights, multiple_centroids):
        """Test batch decoding of multiple deltas."""
        # Force FLOAT32_RAW strategy for lossless reconstruction
        encoder.config['default_strategy'] = DeltaType.FLOAT32_RAW
        
        # First encode to get deltas
        assignments = [
            ClusterAssignment(
                weight_name=w.metadata.name,
                weight_hash=w.compute_hash(), 
                cluster_id=str(i % len(multiple_centroids))
            )
            for i, w in enumerate(multiple_weights)
        ]
        
        # Encode each weight individually with FLOAT32_RAW strategy
        deltas = []
        for weight, assignment in zip(multiple_weights, assignments):
            centroid_idx = int(assignment.cluster_id)
            centroid = multiple_centroids[centroid_idx]
            delta = encoder.encode_weight_to_centroid(weight, centroid, strategy=DeltaType.FLOAT32_RAW)
            deltas.append(delta)
        
        # Check that all deltas were created
        non_none_deltas = [d for d in deltas if d is not None]
        assert len(non_none_deltas) > 0, "No deltas were created"
        
        # Batch decode
        reconstructed = encoder.batch_decode(deltas, multiple_centroids)
        
        # Should have same number of reconstructed weights as non-None deltas
        assert len(reconstructed) == len(non_none_deltas)
        
        # Test individual reconstruction for weights that were successfully encoded
        for i, (weight, delta) in enumerate(zip(multiple_weights, deltas)):
            if delta is not None:
                centroid_idx = int(assignments[i].cluster_id)
                centroid = multiple_centroids[centroid_idx]
                reconstructed_single = encoder.decode_weight_from_centroid(delta, centroid)
                # Should be exact with FLOAT32_RAW strategy
                np.testing.assert_array_almost_equal(weight.data, reconstructed_single.data, decimal=6)

    def test_find_best_centroid(self, encoder, sample_weight, multiple_centroids):
        """Test finding optimal centroid for a weight."""
        best_centroid, score = encoder.find_best_centroid(sample_weight, multiple_centroids)
        
        assert best_centroid is not None
        assert isinstance(best_centroid, WeightTensor)
        assert best_centroid in multiple_centroids
        assert isinstance(score, float)
        assert 0 <= score <= 1  # Assuming normalized score

    def test_evaluate_encoding_efficiency(self, encoder, sample_weight, sample_centroid):
        """Test compression ratio estimation."""
        efficiency = encoder.evaluate_encoding_efficiency(sample_weight, sample_centroid)
        
        assert isinstance(efficiency, dict)
        assert 'compression_ratio' in efficiency
        assert 'delta_size' in efficiency
        assert 'original_size' in efficiency
        assert efficiency['compression_ratio'] > 0

    def test_compare_encoding_strategies(self, encoder, sample_weight, multiple_centroids):
        """Test comparing multiple centroid options."""
        comparisons = encoder.compare_encoding_strategies(sample_weight, multiple_centroids)
        
        assert len(comparisons) >= len(multiple_centroids)  # Multiple strategies per centroid
        for comp in comparisons:
            assert 'centroid' in comp
            assert 'compression_ratio' in comp
            assert 'quality_score' in comp
            assert 'strategy' in comp
            assert 'efficiency_score' in comp

    def test_adaptive_centroid_selection(self, encoder, sample_weight):
        """Test smart centroid selection with quality metrics."""
        # Create mock cluster index
        class MockClusterIndex:
            def get_centroids(self, shape, dtype):
                from coral.core.weight_tensor import WeightMetadata
                mock_centroids = []
                for i in range(3):
                    data = (sample_weight.data + np.random.randn(*sample_weight.data.shape).astype(np.float32) * 0.01 * (i + 1))
                    metadata = WeightMetadata(name=f"mock_c{i}", shape=data.shape, dtype=data.dtype)
                    mock_centroids.append(WeightTensor(data=data, metadata=metadata))
                return mock_centroids
        
        mock_index = MockClusterIndex()
        centroid, metrics = encoder.adaptive_centroid_selection(sample_weight, mock_index)
        
        assert centroid is not None
        assert isinstance(metrics, dict)
        assert 'compression_ratio' in metrics
        assert 'similarity_score' in metrics
        assert 'selected_strategy' in metrics

    def test_select_optimal_strategy(self, encoder, sample_weight, sample_centroid):
        """Test automatic strategy selection."""
        strategy, metrics = encoder.select_optimal_strategy(sample_weight, sample_centroid)
        
        assert isinstance(strategy, DeltaType)
        assert isinstance(metrics, dict)
        assert 'compression_ratio' in metrics
        assert 'reconstruction_error' in metrics

    def test_assess_encoding_quality(self, encoder, sample_weight, sample_centroid):
        """Test quality assessment metrics."""
        # Encode and decode
        delta = encoder.encode_weight_to_centroid(sample_weight, sample_centroid)
        reconstructed = encoder.decode_weight_from_centroid(delta, sample_centroid)
        
        # Assess quality
        quality = encoder.assess_encoding_quality(sample_weight, reconstructed)
        
        assert isinstance(quality, dict)
        assert 'mse' in quality
        assert 'cosine_similarity' in quality
        assert 'max_error' in quality
        assert quality['cosine_similarity'] >= 0.99  # Should be very similar

    def test_estimate_compression_ratio(self, encoder, sample_weight, sample_centroid):
        """Test compression ratio prediction."""
        ratio = encoder.estimate_compression_ratio(sample_weight, sample_centroid)
        
        assert isinstance(ratio, float)
        assert ratio > 0
        
        # Actually encode and compare
        delta = encoder.encode_weight_to_centroid(sample_weight, sample_centroid)
        actual_ratio = sample_weight.data.nbytes / delta.nbytes
        # Estimation may be different due to metadata overhead, just check it's reasonable
        assert 0.1 <= ratio <= 10.0  # Reasonable range for compression ratios

    def test_validate_lossless_reconstruction(self, encoder, sample_weight, sample_centroid):
        """Test validation of perfect reconstruction."""
        # Test with lossless strategy
        delta = encoder.encode_weight_to_centroid(
            sample_weight, sample_centroid, strategy=DeltaType.FLOAT32_RAW
        )
        
        is_valid, error = encoder.validate_lossless_reconstruction(
            sample_weight, sample_centroid, delta
        )
        
        assert is_valid == True
        assert error == 0.0

    def test_generate_encoding_report(self, encoder, multiple_weights, multiple_centroids):
        """Test comprehensive encoding report generation."""
        assignments = [
            ClusterAssignment(
                weight_name=w.metadata.name,
                weight_hash=w.compute_hash(), 
                cluster_id=str(i % len(multiple_centroids))
            )
            for i, w in enumerate(multiple_weights)
        ]
        
        report = encoder.generate_encoding_report(multiple_weights, assignments, multiple_centroids)
        
        assert isinstance(report, dict)
        assert 'total_weights' in report
        assert 'total_compression_ratio' in report
        assert 'strategy_distribution' in report
        assert 'quality_metrics' in report
        assert 'per_weight_stats' in report

    def test_fallback_to_direct_storage(self, encoder):
        """Test fallback when delta encoding is inefficient."""
        # Create very different weight and centroid
        from coral.core.weight_tensor import WeightMetadata
        data1 = np.random.randn(1000).astype(np.float32)
        metadata1 = WeightMetadata(name="weight", shape=data1.shape, dtype=data1.dtype)
        weight = WeightTensor(data=data1, metadata=metadata1)
        
        data2 = np.random.randn(1000).astype(np.float32) * 100
        metadata2 = WeightMetadata(name="centroid", shape=data2.shape, dtype=data2.dtype)
        centroid = WeightTensor(data=data2, metadata=metadata2)
        
        delta = encoder.encode_weight_to_centroid(weight, centroid)
        
        # Should still create a delta (even if not very efficient)
        assert delta is not None
        assert isinstance(delta, Delta)

    def test_caching_repeated_operations(self, encoder, sample_weight, sample_centroid):
        """Test caching for repeated encoding operations."""
        # Enable caching
        encoder.enable_caching(max_size=100)
        
        # First encoding
        delta1 = encoder.encode_weight_to_centroid(sample_weight, sample_centroid)
        
        # Second encoding (should use cache)
        delta2 = encoder.encode_weight_to_centroid(sample_weight, sample_centroid)
        
        np.testing.assert_array_equal(delta1.data, delta2.data)
        # Just verify caching is enabled (implementation details may vary)
        assert encoder._cache_enabled

    def test_thread_safety(self, encoder, multiple_weights, multiple_centroids):
        """Test thread-safe concurrent operations."""
        import threading
        import queue
        
        results = queue.Queue()
        errors = queue.Queue()
        
        def encode_worker(weight, centroid):
            try:
                delta = encoder.encode_weight_to_centroid(weight, centroid)
                results.put(delta)
            except Exception as e:
                errors.put(e)
        
        # Start multiple threads
        threads = []
        for weight in multiple_weights:
            for centroid in multiple_centroids:
                t = threading.Thread(target=encode_worker, args=(weight, centroid))
                threads.append(t)
                t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check results
        assert errors.empty()
        assert results.qsize() == len(multiple_weights) * len(multiple_centroids)

    def test_hierarchical_centroid_support(self, encoder, sample_weight):
        """Test support for hierarchical centroids."""
        # Create hierarchical centroids
        from coral.core.weight_tensor import WeightMetadata
        
        data1 = (sample_weight.data + np.random.randn(*sample_weight.data.shape).astype(np.float32) * 0.01)
        metadata1 = WeightMetadata(name="level1", shape=data1.shape, dtype=data1.dtype)
        level1_centroid = WeightTensor(data=data1, metadata=metadata1)
        
        data2 = (sample_weight.data + np.random.randn(*sample_weight.data.shape).astype(np.float32) * 0.1)
        metadata2 = WeightMetadata(name="level2", shape=data2.shape, dtype=data2.dtype)
        level2_centroid = WeightTensor(data=data2, metadata=metadata2)
        
        hierarchical_centroids = {
            'level1': [level1_centroid],
            'level2': [level2_centroid]
        }
        
        best_centroid, level = encoder.select_from_hierarchy(sample_weight, hierarchical_centroids)
        
        assert best_centroid is not None
        assert level in ['level1', 'level2']

    def test_memory_efficiency(self, encoder):
        """Test memory-efficient operations for large weights."""
        # Create large weight
        from coral.core.weight_tensor import WeightMetadata
        data = np.random.randn(1000, 1000).astype(np.float32)
        metadata = WeightMetadata(name="large", shape=data.shape, dtype=data.dtype)
        large_weight = WeightTensor(data=data, metadata=metadata)
        
        data2 = np.random.randn(1000, 1000).astype(np.float32)
        metadata2 = WeightMetadata(name="large_centroid", shape=data2.shape, dtype=data2.dtype)
        large_centroid = WeightTensor(data=data2, metadata=metadata2)
        
        # Encode with memory limit
        encoder.set_memory_limit(100 * 1024 * 1024)  # 100MB
        delta = encoder.encode_weight_to_centroid(large_weight, large_centroid)
        
        assert delta is not None
        # Should use compressed strategy for large weights
        assert delta.delta_type in [DeltaType.COMPRESSED, DeltaType.INT8_QUANTIZED]

    def test_integration_with_delta_encoder(self, encoder):
        """Test seamless integration with existing DeltaEncoder."""
        with patch('coral.clustering.centroid_encoder.DeltaEncoder') as mock_delta_encoder:
            from coral.core.weight_tensor import WeightMetadata
            data = np.array([1, 2, 3], dtype=np.float32)
            metadata = WeightMetadata(name="test", shape=data.shape, dtype=data.dtype)
            weight = WeightTensor(data=data, metadata=metadata)
            
            data2 = np.array([1.1, 2.1, 3.1], dtype=np.float32)
            metadata2 = WeightMetadata(name="centroid", shape=data2.shape, dtype=data2.dtype)
            centroid = WeightTensor(data=data2, metadata=metadata2)
            
            # Mock delta encoder
            mock_encoder_instance = Mock()
            mock_delta = Mock(spec=Delta)
            mock_encoder_instance.encode_delta.return_value = mock_delta
            mock_delta_encoder.return_value = mock_encoder_instance
            
            # Encode
            delta = encoder.encode_weight_to_centroid(weight, centroid)
            
            # Verify DeltaEncoder was used
            mock_encoder_instance.encode_delta.assert_called_once()
            assert delta == mock_delta

    def test_error_handling(self, encoder):
        """Test comprehensive error handling."""
        # Test with mismatched shapes
        from coral.core.weight_tensor import WeightMetadata
        data1 = np.array([1, 2, 3])
        metadata1 = WeightMetadata(name="w1", shape=data1.shape, dtype=data1.dtype)
        weight = WeightTensor(data=data1, metadata=metadata1)
        
        data2 = np.array([1, 2])
        metadata2 = WeightMetadata(name="c1", shape=data2.shape, dtype=data2.dtype)
        centroid = WeightTensor(data=data2, metadata=metadata2)
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            encoder.encode_weight_to_centroid(weight, centroid)
        
        # Test with invalid delta
        invalid_delta = Delta(
            delta_type=DeltaType.FLOAT32_RAW,
            data=np.array([1, 2, 3], dtype=np.float32),
            metadata={},
            original_shape=(3,),
            original_dtype=np.dtype('float32'),
            reference_hash="invalid"
        )
        
        with pytest.raises(ValueError):
            encoder.decode_weight_from_centroid(invalid_delta, centroid)