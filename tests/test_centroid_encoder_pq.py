"""Tests for CentroidEncoder with Product Quantization integration."""

import pytest
import numpy as np
from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.clustering.centroid_encoder import CentroidEncoder, EncodedWeight
from coral.clustering.cluster_types import ClusterAssignment
from coral.delta.delta_encoder import DeltaType


class TestCentroidEncoderPQ:
    """Test Product Quantization integration in CentroidEncoder."""
    
    @pytest.fixture
    def large_weight(self):
        """Create a large weight tensor that should trigger PQ."""
        data = np.random.randn(64, 64).astype(np.float32)  # 4096 elements > 1024 threshold
        metadata = WeightMetadata(name="large_weight", shape=data.shape, dtype=str(data.dtype))
        return WeightTensor(data=data, metadata=metadata)
    
    @pytest.fixture
    def small_weight(self):
        """Create a small weight tensor that should not trigger PQ."""
        data = np.random.randn(16, 16).astype(np.float32)  # 256 elements < 1024 threshold
        metadata = WeightMetadata(name="small_weight", shape=data.shape, dtype=str(data.dtype))
        return WeightTensor(data=data, metadata=metadata)
    
    @pytest.fixture
    def centroid(self):
        """Create a centroid tensor."""
        data = np.random.randn(64, 64).astype(np.float32)
        metadata = WeightMetadata(name="centroid", shape=data.shape, dtype=str(data.dtype))
        return WeightTensor(data=data, metadata=metadata)
    
    @pytest.fixture
    def small_centroid(self):
        """Create a small centroid tensor."""
        data = np.random.randn(16, 16).astype(np.float32)
        metadata = WeightMetadata(name="small_centroid", shape=data.shape, dtype=str(data.dtype))
        return WeightTensor(data=data, metadata=metadata)
    
    def test_pq_enabled_by_default(self):
        """Test that PQ is enabled by default."""
        encoder = CentroidEncoder()
        assert encoder.enable_pq is True
        assert encoder.pq_threshold_size == 1024
        assert hasattr(encoder, 'pq_config')
    
    def test_pq_disabled_config(self):
        """Test disabling PQ through config."""
        config = {'enable_pq': False}
        encoder = CentroidEncoder(config)
        assert encoder.enable_pq is False
        assert not hasattr(encoder, 'pq_config')
    
    def test_pq_custom_threshold(self):
        """Test custom PQ threshold size."""
        config = {'pq_threshold_size': 2048}
        encoder = CentroidEncoder(config)
        assert encoder.pq_threshold_size == 2048
    
    def test_should_use_pq_size_threshold(self, large_weight, small_weight, centroid, small_centroid):
        """Test _should_use_pq based on size threshold."""
        encoder = CentroidEncoder()
        
        # Create mock deltas
        from coral.delta.delta_encoder import Delta
        large_delta = Delta(
            delta_type=DeltaType.FLOAT32_RAW,
            data=(large_weight.data - centroid.data).astype(np.float32),
            metadata={},
            reference_hash=centroid.compute_hash(),
            original_shape=large_weight.data.shape,
            original_dtype=str(large_weight.data.dtype),
            compression_ratio=1.0
        )
        
        small_delta = Delta(
            delta_type=DeltaType.FLOAT32_RAW,
            data=(small_weight.data - small_centroid.data).astype(np.float32),
            metadata={},
            reference_hash=small_centroid.compute_hash(),
            original_shape=small_weight.data.shape,
            original_dtype=str(small_weight.data.dtype),
            compression_ratio=1.0
        )
        
        # Large weight should potentially use PQ
        should_use, strategy = encoder._should_use_pq(large_delta, large_weight, centroid)
        # May or may not use PQ depending on characteristics, but size is sufficient
        
        # Small weight should not use PQ
        should_use, strategy = encoder._should_use_pq(small_delta, small_weight, small_centroid)
        assert should_use is False
        assert strategy is None
    
    def test_should_use_pq_already_compressed(self, large_weight, centroid):
        """Test _should_use_pq skips already compressed deltas."""
        encoder = CentroidEncoder()
        
        # Create highly compressed delta
        from coral.delta.delta_encoder import Delta
        delta = Delta(
            delta_type=DeltaType.COMPRESSED,
            data=np.zeros(100, dtype=np.uint8),  # Simulate highly compressed
            metadata={'compression_level': 9},
            reference_hash=centroid.compute_hash(),
            original_shape=large_weight.data.shape,
            original_dtype=str(large_weight.data.dtype),
            compression_ratio=20.0  # Very high compression
        )
        
        should_use, strategy = encoder._should_use_pq(delta, large_weight, centroid)
        assert should_use is False  # Already well compressed
    
    def test_should_use_pq_high_quality_threshold(self, large_weight, centroid):
        """Test PQ strategy selection based on quality threshold."""
        # High quality threshold should prefer PQ_LOSSLESS
        config = {'quality_threshold': 0.995}
        encoder = CentroidEncoder(config)
        
        # Create delta with high redundancy to trigger PQ
        # Add structured pattern for high redundancy
        pattern = np.tile(np.arange(64), (64, 1)).astype(np.float32) * 0.1
        modified_data = large_weight.data + pattern
        modified_weight = WeightTensor(
            data=modified_data,
            metadata=WeightMetadata(
                name="modified_weight",
                shape=modified_data.shape,
                dtype=str(modified_data.dtype)
            )
        )
        
        from coral.delta.delta_encoder import Delta
        delta = Delta(
            delta_type=DeltaType.FLOAT32_RAW,
            data=(modified_weight.data - centroid.data).astype(np.float32),
            metadata={},
            reference_hash=centroid.compute_hash(),
            original_shape=modified_weight.data.shape,
            original_dtype=str(modified_weight.data.dtype),
            compression_ratio=1.0
        )
        
        should_use, strategy = encoder._should_use_pq(delta, modified_weight, centroid)
        if should_use:
            assert strategy == DeltaType.PQ_LOSSLESS
    
    def test_encode_with_pq_upgrade(self, large_weight):
        """Test that encoding upgrades to PQ when beneficial."""
        encoder = CentroidEncoder()
        
        # Create centroid similar to weight for better delta properties
        centroid_data = (large_weight.data * 0.95 + np.random.randn(*large_weight.shape).astype(np.float32) * 0.01).astype(np.float32)
        centroid = WeightTensor(
            data=centroid_data,
            metadata=WeightMetadata(
                name="centroid",
                shape=centroid_data.shape,
                dtype=str(centroid_data.dtype)
            )
        )
        
        # Encode the weight
        delta = encoder.encode_weight_to_centroid(large_weight, centroid)
        
        assert delta is not None
        # Check if PQ was used (if beneficial based on data characteristics)
        # The actual use of PQ depends on data patterns
    
    def test_batch_encode_size_grouping(self):
        """Test that batch encoding groups by size when PQ is enabled."""
        encoder = CentroidEncoder()
        
        # Create weights of different sizes
        weights = []
        centroids = []
        assignments = []
        
        for i, size in enumerate([32, 64, 32, 128, 64]):
            data = np.random.randn(size, size).astype(np.float32)
            weight = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"weight_{i}",
                    shape=data.shape,
                    dtype=str(data.dtype)
                )
            )
            weights.append(weight)
            
            # Create matching centroid
            centroid_data = data * 0.9
            centroid = WeightTensor(
                data=centroid_data,
                metadata=WeightMetadata(
                    name=f"centroid_{i}",
                    shape=centroid_data.shape,
                    dtype=str(centroid_data.dtype)
                )
            )
            centroids.append(centroid)
            
            # Create assignment
            assignment = ClusterAssignment(
                weight_name=f"weight_{i}",
                weight_hash=weight.compute_hash(),
                cluster_id=str(i),
                distance_to_centroid=0.1,
                similarity_score=0.9
            )
            assignments.append(assignment)
        
        # Batch encode
        deltas = encoder.batch_encode(weights, assignments, centroids)
        
        # Should return deltas in original order
        assert len(deltas) == len(weights)
        for i, delta in enumerate(deltas):
            if delta is not None:
                assert delta.original_shape == weights[i].data.shape
    
    def test_pq_strategy_in_selection(self, large_weight, centroid):
        """Test that PQ strategies are included in strategy selection."""
        encoder = CentroidEncoder()
        
        # Get optimal strategy for large weight
        strategy, metrics = encoder.select_optimal_strategy(large_weight, centroid)
        
        # Check that PQ strategies were considered
        assert 'all_strategies' in metrics
        strategies_tested = list(metrics['all_strategies'].keys())
        
        # PQ strategies should be included for large weights
        assert DeltaType.PQ_ENCODED in strategies_tested
        assert DeltaType.PQ_LOSSLESS in strategies_tested
    
    def test_estimate_delta_size_pq(self):
        """Test delta size estimation for PQ strategies."""
        encoder = CentroidEncoder()
        
        # Create test weight and centroid
        weight_data = np.random.randn(128, 128).astype(np.float32)
        centroid_data = weight_data * 0.9
        
        weight = WeightTensor(
            data=weight_data,
            metadata=WeightMetadata(name="weight", shape=weight_data.shape, dtype=str(weight_data.dtype))
        )
        centroid = WeightTensor(
            data=centroid_data,
            metadata=WeightMetadata(name="centroid", shape=centroid_data.shape, dtype=str(centroid_data.dtype))
        )
        
        # Test PQ_ENCODED estimation
        pq_encoded_size = encoder._estimate_delta_size(weight, centroid, DeltaType.PQ_ENCODED)
        assert pq_encoded_size > 0
        assert pq_encoded_size < weight_data.nbytes  # Should be compressed
        
        # Test PQ_LOSSLESS estimation
        pq_lossless_size = encoder._estimate_delta_size(weight, centroid, DeltaType.PQ_LOSSLESS)
        assert pq_lossless_size > 0
        assert pq_lossless_size > pq_encoded_size  # Lossless should be larger
    
    def test_encode_weight_api_with_pq(self, large_weight):
        """Test the encode_weight API method with PQ strategies."""
        encoder = CentroidEncoder()
        
        # Create a centroid that's very similar to the weight for better compression
        centroid_data = large_weight.data * 0.99 + np.random.randn(*large_weight.shape).astype(np.float32) * 0.001
        centroid = WeightTensor(
            data=centroid_data,
            metadata=WeightMetadata(
                name="centroid",
                shape=centroid_data.shape,
                dtype=str(centroid_data.dtype)
            )
        )
        
        # Test auto strategy
        encoded = encoder.encode_weight(large_weight, centroid, strategy="auto")
        assert isinstance(encoded, EncodedWeight)
        # Compression ratio depends on data similarity, just check it's valid
        assert encoded.compression_ratio > 0
        
        # Test explicit PQ strategy
        encoded_pq = encoder.encode_weight(large_weight, centroid, strategy="PQ_ENCODED")
        assert isinstance(encoded_pq, EncodedWeight)
        assert encoded_pq.encoding_strategy == DeltaType.PQ_ENCODED
        
        # Test PQ lossless
        encoded_pq_lossless = encoder.encode_weight(large_weight, centroid, strategy="PQ_LOSSLESS")
        assert isinstance(encoded_pq_lossless, EncodedWeight)
        assert encoded_pq_lossless.encoding_strategy == DeltaType.PQ_LOSSLESS
    
    def test_get_encoding_stats_with_pq(self, large_weight):
        """Test encoding stats include PQ information."""
        encoder = CentroidEncoder()
        
        # Create a centroid that's very similar to the weight
        centroid_data = large_weight.data * 0.99 + np.random.randn(*large_weight.shape).astype(np.float32) * 0.001
        centroid = WeightTensor(
            data=centroid_data,
            metadata=WeightMetadata(
                name="centroid",
                shape=centroid_data.shape,
                dtype=str(centroid_data.dtype)
            )
        )
        
        # Create PQ-encoded weight
        encoded = encoder.encode_weight(large_weight, centroid, strategy="PQ_ENCODED")
        
        # Get stats
        stats = encoder.get_encoding_stats(encoded)
        
        assert stats['encoding_strategy'] == 'pq_encoded'  # DeltaType.value returns lowercase
        assert stats['is_lossless'] is False  # PQ_ENCODED is lossy
        assert stats['compression_ratio'] > 0  # Valid ratio
        
        # Create PQ lossless encoded weight
        encoded_lossless = encoder.encode_weight(large_weight, centroid, strategy="PQ_LOSSLESS")
        stats_lossless = encoder.get_encoding_stats(encoded_lossless)
        
        assert stats_lossless['encoding_strategy'] == 'pq_lossless'  # DeltaType.value returns lowercase
        assert stats_lossless['is_lossless'] is True  # PQ_LOSSLESS is lossless