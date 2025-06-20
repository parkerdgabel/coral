"""CentroidEncoder for encoding weights as centroid + delta pairs."""

import logging
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from functools import lru_cache

from coral.core.weight_tensor import WeightTensor
from coral.delta.delta_encoder import DeltaEncoder, Delta, DeltaType, DeltaConfig
from coral.clustering.cluster_types import ClusterAssignment


logger = logging.getLogger(__name__)


class CentroidEncoder:
    """
    Encoder for representing weights as centroid + delta pairs.
    
    This class provides centroid-based delta encoding that can achieve
    better compression ratios by encoding weights as differences from
    similar centroids rather than standalone deltas.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CentroidEncoder.
        
        Args:
            config: Configuration dictionary with encoding parameters
        """
        self.config = config or {}
        delta_config = DeltaConfig()
        if 'delta_config' in self.config:
            delta_config = DeltaConfig.from_dict(self.config['delta_config'])
        self.delta_encoder = DeltaEncoder(delta_config)
        
        # Configuration parameters
        self.similarity_threshold = self.config.get('similarity_threshold', 0.95)
        self.min_compression_ratio = self.config.get('min_compression_ratio', 0.0)
        self.quality_threshold = self.config.get('quality_threshold', 0.99)
        self.memory_limit = self.config.get('memory_limit', None)
        
        # Caching
        self._cache_enabled = False
        self._cache_stats = {'hits': 0, 'misses': 0}
        self._encoding_cache = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"CentroidEncoder initialized with config: {self.config}")
    
    def encode_weight_to_centroid(
        self,
        weight: WeightTensor,
        centroid: WeightTensor,
        strategy: Optional[DeltaType] = None
    ) -> Optional[Delta]:
        """
        Encode weight as delta from centroid.
        
        Args:
            weight: Weight tensor to encode
            centroid: Reference centroid
            strategy: Specific encoding strategy to use
            
        Returns:
            Delta object or None if encoding is inefficient
        """
        try:
            # Validate inputs
            self._validate_weight_centroid_compatibility(weight, centroid)
            
            # Check cache
            if self._cache_enabled:
                cache_key = (weight.compute_hash(), centroid.compute_hash(), strategy)
                if cache_key in self._encoding_cache:
                    self._cache_stats['hits'] += 1
                    return self._encoding_cache[cache_key]
                self._cache_stats['misses'] += 1
            
            # Select optimal strategy if not specified
            if strategy is None:
                strategy, _ = self.select_optimal_strategy(weight, centroid)
            
            # Check if encoding would be beneficial
            estimated_ratio = self.estimate_compression_ratio(weight, centroid, strategy)
            if estimated_ratio < self.min_compression_ratio:
                logger.debug(f"Skipping encoding - ratio {estimated_ratio:.2f} below threshold")
                return None
            
            # Configure delta encoder for this strategy
            temp_config = DeltaConfig(delta_type=strategy)
            temp_encoder = DeltaEncoder(temp_config)
            
            # Encode using delta encoder
            delta = temp_encoder.encode_delta(weight, centroid)
            
            # Cache result
            if self._cache_enabled and delta is not None:
                cache_key = (weight.compute_hash(), centroid.compute_hash(), strategy)
                self._encoding_cache[cache_key] = delta
            
            return delta
            
        except Exception as e:
            logger.error(f"Error encoding weight {weight.metadata.name} with centroid {centroid.metadata.name}: {e}")
            raise
    
    def decode_weight_from_centroid(
        self,
        delta: Delta,
        centroid: WeightTensor
    ) -> WeightTensor:
        """
        Reconstruct weight from centroid + delta.
        
        Args:
            delta: Delta object containing encoded differences
            centroid: Reference centroid
            
        Returns:
            Reconstructed weight tensor
        """
        try:
            # Validate delta refers to this centroid
            if delta.reference_hash != centroid.compute_hash():
                raise ValueError(
                    f"Delta reference hash {delta.reference_hash} doesn't match "
                    f"centroid hash {centroid.compute_hash()}"
                )
            
            # Decode using delta encoder
            reconstructed = self.delta_encoder.decode_delta(delta, centroid)
            
            return reconstructed
            
        except Exception as e:
            logger.error(f"Error decoding delta with centroid {centroid.metadata.name}: {e}")
            raise
    
    def batch_encode(
        self,
        weights: List[WeightTensor],
        assignments: List[ClusterAssignment],
        centroids: List[WeightTensor]
    ) -> List[Optional[Delta]]:
        """
        Efficiently encode multiple weights in batch.
        
        Args:
            weights: List of weight tensors to encode
            assignments: Cluster assignments for each weight
            centroids: List of available centroids
            
        Returns:
            List of delta objects (None for inefficient encodings)
        """
        if len(weights) != len(assignments):
            raise ValueError("Number of weights must match number of assignments")
        
        # Create centroid lookup by cluster_id
        centroid_map = {}
        for i, centroid in enumerate(centroids):
            # Use index as cluster_id for backwards compatibility
            centroid_map[str(i)] = centroid
            if hasattr(centroid, 'cluster_id'):
                centroid_map[centroid.cluster_id] = centroid
        
        deltas = []
        
        # Use thread pool for parallel encoding
        max_workers = min(len(weights), self.config.get('max_workers', 4))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit encoding tasks
            future_to_index = {}
            for i, (weight, assignment) in enumerate(zip(weights, assignments)):
                centroid = centroid_map.get(str(assignment.cluster_id))
                if centroid is None:
                    # Try integer index as fallback
                    if isinstance(assignment.cluster_id, str) and assignment.cluster_id.isdigit():
                        idx = int(assignment.cluster_id)
                        if idx < len(centroids):
                            centroid = centroids[idx]
                
                if centroid is None:
                    raise ValueError(f"No centroid found for cluster_id {assignment.cluster_id}")
                
                future = executor.submit(self.encode_weight_to_centroid, weight, centroid)
                future_to_index[future] = i
            
            # Collect results in order
            results = [None] * len(weights)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Error encoding weight {index}: {e}")
                    results[index] = None
            
            deltas = results
        
        return deltas
    
    def batch_decode(
        self,
        deltas: List[Delta],
        centroids: List[WeightTensor]
    ) -> List[WeightTensor]:
        """
        Efficiently decode multiple deltas in batch.
        
        Args:
            deltas: List of delta objects to decode
            centroids: List of available centroids
            
        Returns:
            List of reconstructed weight tensors
        """
        # Create centroid lookup
        centroid_map = {c.compute_hash(): c for c in centroids}
        
        # Use thread pool for parallel decoding
        max_workers = min(len(deltas), self.config.get('max_workers', 4))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit decoding tasks
            future_to_index = {}
            for i, delta in enumerate(deltas):
                if delta is None:
                    continue
                
                centroid = centroid_map.get(delta.reference_hash)
                if centroid is None:
                    raise ValueError(f"Centroid not found for hash {delta.reference_hash}")
                
                future = executor.submit(self.decode_weight_from_centroid, delta, centroid)
                future_to_index[future] = i
            
            # Collect results in order
            results = [None] * len(deltas)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Error decoding delta {index}: {e}")
                    raise
            
            # Return results in order, skipping deltas that were None
            reconstructed = []
            for i, (delta, result) in enumerate(zip(deltas, results)):
                if delta is not None and result is not None:
                    reconstructed.append(result)
        
        return reconstructed
    
    def find_best_centroid(
        self,
        weight: WeightTensor,
        centroids: List[WeightTensor]
    ) -> Tuple[WeightTensor, float]:
        """
        Find centroid that minimizes delta size for given weight.
        
        Args:
            weight: Weight tensor to encode
            centroids: List of candidate centroids
            
        Returns:
            Tuple of (best_centroid, quality_score)
        """
        if not centroids:
            raise ValueError("No centroids provided")
        
        best_centroid = None
        best_score = float('-inf')
        
        for centroid in centroids:
            try:
                # Skip incompatible centroids
                if not self._are_compatible(weight, centroid):
                    continue
                
                # Evaluate encoding efficiency
                efficiency = self.evaluate_encoding_efficiency(weight, centroid)
                
                # Compute composite score (compression + similarity)
                compression_score = min(efficiency['compression_ratio'] / 5.0, 1.0)
                similarity_score = self._compute_similarity(weight, centroid)
                composite_score = 0.7 * compression_score + 0.3 * similarity_score
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_centroid = centroid
                    
            except Exception as e:
                logger.debug(f"Error evaluating centroid {centroid.metadata.name}: {e}")
                continue
        
        if best_centroid is None:
            raise ValueError("No compatible centroids found")
        
        return best_centroid, best_score
    
    def evaluate_encoding_efficiency(
        self,
        weight: WeightTensor,
        centroid: WeightTensor
    ) -> Dict[str, float]:
        """
        Estimate compression ratio and efficiency metrics.
        
        Args:
            weight: Weight tensor to encode
            centroid: Reference centroid
            
        Returns:
            Dictionary with efficiency metrics
        """
        original_size = weight.data.nbytes
        
        # Estimate delta size without actually encoding
        delta_estimate = self._estimate_delta_size(weight, centroid)
        
        compression_ratio = original_size / delta_estimate if delta_estimate > 0 else 1.0
        similarity = self._compute_similarity(weight, centroid)
        
        return {
            'compression_ratio': compression_ratio,
            'delta_size': delta_estimate,
            'original_size': original_size,
            'similarity_score': similarity,
            'efficiency_score': compression_ratio * similarity
        }
    
    def compare_encoding_strategies(
        self,
        weight: WeightTensor,
        candidates: List[WeightTensor]
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple centroid options for encoding.
        
        Args:
            weight: Weight tensor to encode
            candidates: List of candidate centroids
            
        Returns:
            List of comparison results sorted by efficiency
        """
        comparisons = []
        
        for centroid in candidates:
            if not self._are_compatible(weight, centroid):
                continue
                
            try:
                # Evaluate different strategies
                for strategy in [DeltaType.FLOAT32_RAW, DeltaType.COMPRESSED, 
                               DeltaType.INT8_QUANTIZED]:
                    efficiency = self.evaluate_encoding_efficiency(weight, centroid)
                    quality_score = self._compute_similarity(weight, centroid)
                    
                    comparisons.append({
                        'centroid': centroid,
                        'strategy': strategy,
                        'compression_ratio': efficiency['compression_ratio'],
                        'quality_score': quality_score,
                        'efficiency_score': efficiency['efficiency_score'],
                        'estimated_delta_size': efficiency['delta_size']
                    })
                    
            except Exception as e:
                logger.debug(f"Error comparing centroid {centroid.name}: {e}")
                continue
        
        # Sort by efficiency score
        comparisons.sort(key=lambda x: x['efficiency_score'], reverse=True)
        
        return comparisons
    
    def adaptive_centroid_selection(
        self,
        weight: WeightTensor,
        cluster_index: Any
    ) -> Tuple[WeightTensor, Dict[str, Any]]:
        """
        Smart centroid selection with quality metrics.
        
        Args:
            weight: Weight tensor to encode
            cluster_index: ClusterIndex instance for centroid lookup
            
        Returns:
            Tuple of (selected_centroid, selection_metrics)
        """
        # Get candidate centroids from index
        centroids = self._get_candidate_centroids(weight, cluster_index)
        
        if not centroids:
            raise ValueError("No candidate centroids found")
        
        # Find best centroid
        best_centroid, quality_score = self.find_best_centroid(weight, centroids)
        
        # Evaluate optimal strategy
        strategy, strategy_metrics = self.select_optimal_strategy(weight, best_centroid)
        
        # Compute comprehensive metrics
        efficiency = self.evaluate_encoding_efficiency(weight, best_centroid)
        
        metrics = {
            'compression_ratio': efficiency['compression_ratio'],
            'similarity_score': quality_score,
            'selected_strategy': strategy,
            'strategy_metrics': strategy_metrics,
            'candidate_count': len(centroids),
            'efficiency_score': efficiency['efficiency_score']
        }
        
        return best_centroid, metrics
    
    def select_optimal_strategy(
        self,
        weight: WeightTensor,
        centroid: WeightTensor
    ) -> Tuple[DeltaType, Dict[str, Any]]:
        """
        Choose best encoding strategy for weight-centroid pair.
        
        Args:
            weight: Weight tensor to encode
            centroid: Reference centroid
            
        Returns:
            Tuple of (optimal_strategy, metrics)
        """
        strategies_to_test = [
            DeltaType.FLOAT32_RAW,
            DeltaType.COMPRESSED,
            DeltaType.INT8_QUANTIZED,
            DeltaType.INT16_QUANTIZED
        ]
        
        best_strategy = DeltaType.FLOAT32_RAW
        best_score = 0.0
        strategy_results = {}
        
        for strategy in strategies_to_test:
            try:
                # Estimate compression and quality
                compression_ratio = self.estimate_compression_ratio(weight, centroid, strategy)
                
                # For quantized strategies, estimate reconstruction error
                if strategy in [DeltaType.INT8_QUANTIZED, DeltaType.INT16_QUANTIZED]:
                    error_estimate = self._estimate_quantization_error(weight, centroid, strategy)
                    quality_score = max(0, 1 - error_estimate)
                else:
                    quality_score = 1.0  # Lossless
                
                # Composite score favoring compression while maintaining quality
                composite_score = compression_ratio * (quality_score ** 2)
                
                strategy_results[strategy] = {
                    'compression_ratio': compression_ratio,
                    'quality_score': quality_score,
                    'composite_score': composite_score
                }
                
                if composite_score > best_score and quality_score >= self.quality_threshold:
                    best_score = composite_score
                    best_strategy = strategy
                    
            except Exception as e:
                logger.debug(f"Error evaluating strategy {strategy}: {e}")
                continue
        
        return best_strategy, {
            'compression_ratio': strategy_results[best_strategy]['compression_ratio'],
            'reconstruction_error': 1 - strategy_results[best_strategy]['quality_score'],
            'all_strategies': strategy_results
        }
    
    def assess_encoding_quality(
        self,
        original: WeightTensor,
        reconstructed: WeightTensor
    ) -> Dict[str, float]:
        """
        Assess quality of encoding/reconstruction.
        
        Args:
            original: Original weight tensor
            reconstructed: Reconstructed weight tensor
            
        Returns:
            Dictionary with quality metrics
        """
        if original.data.shape != reconstructed.data.shape:
            raise ValueError("Shape mismatch between original and reconstructed")
        
        # Mean squared error
        mse = np.mean((original.data - reconstructed.data) ** 2)
        
        # Cosine similarity
        orig_flat = original.data.flatten()
        recon_flat = reconstructed.data.flatten()
        cosine_sim = np.dot(orig_flat, recon_flat) / (
            np.linalg.norm(orig_flat) * np.linalg.norm(recon_flat) + 1e-8
        )
        
        # Maximum absolute error
        max_error = np.max(np.abs(original.data - reconstructed.data))
        
        # Relative error
        relative_error = np.mean(
            np.abs(original.data - reconstructed.data) / (np.abs(original.data) + 1e-8)
        )
        
        return {
            'mse': float(mse),
            'cosine_similarity': float(cosine_sim),
            'max_error': float(max_error),
            'relative_error': float(relative_error),
            'is_lossless': mse < 1e-10
        }
    
    def estimate_compression_ratio(
        self,
        weight: WeightTensor,
        centroid: WeightTensor,
        strategy: Optional[DeltaType] = None
    ) -> float:
        """
        Predict compression ratio without actual encoding.
        
        Args:
            weight: Weight tensor to encode
            centroid: Reference centroid
            strategy: Encoding strategy to evaluate
            
        Returns:
            Estimated compression ratio
        """
        original_size = weight.data.nbytes
        
        if strategy is None:
            strategy = DeltaType.FLOAT32_RAW
        
        # Estimate delta size based on strategy
        delta_size = self._estimate_delta_size(weight, centroid, strategy)
        
        return original_size / delta_size if delta_size > 0 else 1.0
    
    def validate_lossless_reconstruction(
        self,
        weight: WeightTensor,
        centroid: WeightTensor,
        delta: Delta
    ) -> Tuple[bool, float]:
        """
        Validate perfect reconstruction for lossless strategies.
        
        Args:
            weight: Original weight tensor
            centroid: Reference centroid
            delta: Encoded delta
            
        Returns:
            Tuple of (is_valid, reconstruction_error)
        """
        try:
            # Reconstruct
            reconstructed = self.decode_weight_from_centroid(delta, centroid)
            
            # Check for perfect match
            error = np.max(np.abs(weight.data - reconstructed.data))
            is_valid = error < 1e-10
            
            return is_valid, float(error)
            
        except Exception as e:
            logger.error(f"Error validating reconstruction: {e}")
            return False, float('inf')
    
    def generate_encoding_report(
        self,
        weights: List[WeightTensor],
        assignments: List[ClusterAssignment],
        centroids: List[WeightTensor]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive encoding analysis report.
        
        Args:
            weights: List of weight tensors
            assignments: Cluster assignments
            centroids: List of centroids
            
        Returns:
            Comprehensive report dictionary
        """
        # Encode all weights
        deltas = self.batch_encode(weights, assignments, centroids)
        
        # Collect statistics
        total_original_size = sum(w.data.nbytes for w in weights)
        total_encoded_size = sum(
            d.nbytes if d else w.data.nbytes 
            for d, w in zip(deltas, weights)
        )
        
        overall_compression = total_original_size / total_encoded_size
        
        # Strategy distribution
        strategy_counts = {}
        for delta in deltas:
            if delta:
                strategy = delta.delta_type
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Per-weight statistics
        per_weight_stats = []
        for i, (weight, delta, assignment) in enumerate(zip(weights, deltas, assignments)):
            # Find the centroid for this assignment
            centroid = None
            for c in centroids:
                if hasattr(c, 'cluster_id') and c.cluster_id == assignment.cluster_id:
                    centroid = c
                    break
            
            # Fallback to index-based lookup
            if centroid is None and assignment.cluster_id.isdigit():
                idx = int(assignment.cluster_id)
                if idx < len(centroids):
                    centroid = centroids[idx]
                    
            if centroid is None:
                raise ValueError(f"Cannot find centroid for cluster_id {assignment.cluster_id}")
            
            if delta:
                compression_ratio = weight.data.nbytes / delta.nbytes
                encoded_size = delta.nbytes
                strategy = delta.delta_type
            else:
                compression_ratio = 1.0
                encoded_size = weight.data.nbytes
                strategy = "direct"
            
            efficiency = self.evaluate_encoding_efficiency(weight, centroid)
            
            per_weight_stats.append({
                'weight_name': weight.metadata.name,
                'original_size': weight.data.nbytes,
                'encoded_size': encoded_size,
                'compression_ratio': compression_ratio,
                'strategy': strategy,
                'similarity_to_centroid': efficiency['similarity_score'],
                'cluster_id': assignment.cluster_id
            })
        
        # Quality metrics
        quality_metrics = {
            'total_weights_encoded': len([d for d in deltas if d]),
            'total_weights_direct': len([d for d in deltas if not d]),
            'average_similarity': np.mean([s['similarity_to_centroid'] for s in per_weight_stats]),
            'compression_efficiency': overall_compression
        }
        
        return {
            'total_weights': len(weights),
            'total_original_size': total_original_size,
            'total_encoded_size': total_encoded_size,
            'total_compression_ratio': overall_compression,
            'strategy_distribution': strategy_counts,
            'quality_metrics': quality_metrics,
            'per_weight_stats': per_weight_stats,
            'centroids_used': len(set(a.cluster_id for a in assignments))
        }
    
    def enable_caching(self, max_size: int = 1000):
        """Enable caching for repeated operations."""
        self._cache_enabled = True
        self._encoding_cache = {}
        
        # Wrap encode method with LRU cache
        self._original_encode = self.encode_weight_to_centroid
        self.encode_weight_to_centroid = lru_cache(maxsize=max_size)(
            self.encode_weight_to_centroid
        )
    
    def disable_caching(self):
        """Disable caching."""
        self._cache_enabled = False
        self._encoding_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get caching statistics."""
        return self._cache_stats.copy()
    
    def set_memory_limit(self, limit_bytes: int):
        """Set memory limit for operations."""
        self.memory_limit = limit_bytes
    
    def select_from_hierarchy(
        self,
        weight: WeightTensor,
        hierarchical_centroids: Dict[str, List[WeightTensor]]
    ) -> Tuple[WeightTensor, str]:
        """
        Select best centroid from hierarchical structure.
        
        Args:
            weight: Weight tensor to encode
            hierarchical_centroids: Dict mapping levels to centroid lists
            
        Returns:
            Tuple of (best_centroid, level)
        """
        best_centroid = None
        best_score = float('-inf')
        best_level = None
        
        for level, centroids in hierarchical_centroids.items():
            try:
                centroid, score = self.find_best_centroid(weight, centroids)
                if score > best_score:
                    best_score = score
                    best_centroid = centroid
                    best_level = level
            except ValueError:
                continue
        
        if best_centroid is None:
            raise ValueError("No compatible centroids found in hierarchy")
        
        return best_centroid, best_level
    
    # Private helper methods
    
    def _validate_weight_centroid_compatibility(
        self,
        weight: WeightTensor,
        centroid: WeightTensor
    ):
        """Validate that weight and centroid are compatible."""
        if weight.data.shape != centroid.data.shape:
            raise ValueError(
                f"Shape mismatch: weight {weight.data.shape} vs centroid {centroid.data.shape}"
            )
        
        if weight.data.dtype != centroid.data.dtype:
            raise ValueError(
                f"Dtype mismatch: weight {weight.data.dtype} vs centroid {centroid.data.dtype}"
            )
    
    def _are_compatible(self, weight: WeightTensor, centroid: WeightTensor) -> bool:
        """Check if weight and centroid are compatible."""
        return (weight.data.shape == centroid.data.shape and 
                weight.data.dtype == centroid.data.dtype)
    
    def _compute_similarity(self, weight: WeightTensor, centroid: WeightTensor) -> float:
        """Compute cosine similarity between weight and centroid."""
        w_flat = weight.data.flatten()
        c_flat = centroid.data.flatten()
        
        dot_product = np.dot(w_flat, c_flat)
        norm_product = np.linalg.norm(w_flat) * np.linalg.norm(c_flat)
        
        if norm_product == 0:
            return 1.0 if np.allclose(w_flat, c_flat) else 0.0
        
        return float(dot_product / norm_product)
    
    def _estimate_delta_size(
        self,
        weight: WeightTensor,
        centroid: WeightTensor,
        strategy: Optional[DeltaType] = None
    ) -> int:
        """Estimate encoded delta size without actual encoding."""
        if strategy is None:
            strategy = DeltaType.FLOAT32_RAW
        
        # Compute difference
        diff = weight.data - centroid.data
        
        if strategy == DeltaType.FLOAT32_RAW:
            return diff.nbytes
        elif strategy == DeltaType.COMPRESSED:
            # Estimate 30-50% compression for typical deltas
            return int(diff.nbytes * 0.4)
        elif strategy == DeltaType.INT8_QUANTIZED:
            return diff.size  # 1 byte per element
        elif strategy == DeltaType.INT16_QUANTIZED:
            return diff.size * 2  # 2 bytes per element
        elif strategy == DeltaType.SPARSE:
            # Estimate sparsity
            threshold = np.std(diff) * 0.1
            non_zero_count = np.sum(np.abs(diff) > threshold)
            return non_zero_count * (diff.itemsize + 4)  # value + index
        else:
            return diff.nbytes
    
    def _estimate_quantization_error(
        self,
        weight: WeightTensor,
        centroid: WeightTensor,
        strategy: DeltaType
    ) -> float:
        """Estimate reconstruction error for quantized strategies."""
        diff = weight.data - centroid.data
        
        if strategy == DeltaType.INT8_QUANTIZED:
            # Estimate 8-bit quantization error
            max_val = np.max(np.abs(diff))
            if max_val > 0:
                scale = max_val / 127
                quantized = np.round(diff / scale) * scale
                error = np.mean((diff - quantized) ** 2)
            else:
                error = 0.0  # Perfect reconstruction for zero diff
        elif strategy == DeltaType.INT16_QUANTIZED:
            # Estimate 16-bit quantization error  
            max_val = np.max(np.abs(diff))
            if max_val > 0:
                scale = max_val / 32767
                quantized = np.round(diff / scale) * scale
                error = np.mean((diff - quantized) ** 2)
            else:
                error = 0.0  # Perfect reconstruction for zero diff
        else:
            error = 0.0
        
        return float(error)
    
    def _get_candidate_centroids(
        self,
        weight: WeightTensor,
        cluster_index: Any
    ) -> List[WeightTensor]:
        """Get candidate centroids from cluster index."""
        # This would interface with the actual ClusterIndex
        # For now, return empty list as placeholder
        if hasattr(cluster_index, 'get_centroids'):
            return cluster_index.get_centroids(weight.data.shape, weight.data.dtype)
        return []
    
    def _get_cluster_index(self):
        """Get cluster index instance - placeholder for integration."""
        return None