"""Core deduplication engine for weight tensors"""

from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
from dataclasses import dataclass, field
import logging

from coral.core.weight_tensor import WeightTensor


logger = logging.getLogger(__name__)


@dataclass
class DeduplicationStats:
    """Statistics about deduplication results"""
    total_weights: int = 0
    unique_weights: int = 0
    duplicate_weights: int = 0
    similar_weights: int = 0
    bytes_saved: int = 0
    compression_ratio: float = 0.0
    
    def update(self, original_bytes: int, deduplicated_bytes: int):
        """Update compression statistics"""
        self.bytes_saved = original_bytes - deduplicated_bytes
        if original_bytes > 0:
            self.compression_ratio = self.bytes_saved / original_bytes


@dataclass 
class WeightGroup:
    """Group of weights that are identical or similar"""
    reference_hash: str
    reference_weight: WeightTensor
    duplicates: List[Tuple[str, WeightTensor]] = field(default_factory=list)
    similar: List[Tuple[str, WeightTensor, float]] = field(default_factory=list)  # (name, weight, similarity)
    
    @property
    def total_count(self) -> int:
        """Total number of weights in this group"""
        return 1 + len(self.duplicates) + len(self.similar)
    
    @property
    def bytes_saved(self) -> int:
        """Bytes saved by deduplication in this group"""
        ref_bytes = self.reference_weight.nbytes
        # Exact duplicates save full size
        duplicate_savings = ref_bytes * len(self.duplicates)
        # Similar weights might use delta encoding, estimate 50% savings
        similar_savings = int(ref_bytes * len(self.similar) * 0.5)
        return duplicate_savings + similar_savings


class Deduplicator:
    """
    Core deduplication engine for neural network weights.
    
    Supports:
    - Exact deduplication through content hashing
    - Similarity-based deduplication
    - Reference counting
    - Deduplication statistics
    """
    
    def __init__(self, similarity_threshold: float = 0.99):
        """
        Initialize the deduplicator.
        
        Args:
            similarity_threshold: Threshold for considering weights similar (0-1)
        """
        self.similarity_threshold = similarity_threshold
        self.weight_index: Dict[str, WeightTensor] = {}  # hash -> weight
        self.weight_groups: Dict[str, WeightGroup] = {}  # reference_hash -> group
        self.name_to_hash: Dict[str, str] = {}  # weight name -> hash
        self.stats = DeduplicationStats()
        
    def add_weight(self, weight: WeightTensor, name: Optional[str] = None) -> str:
        """
        Add a weight to the deduplicator and check for duplicates.
        
        Args:
            weight: WeightTensor to add
            name: Optional name for the weight
            
        Returns:
            Hash of the weight (or reference weight if duplicate/similar)
        """
        if name is None:
            name = weight.metadata.name
            
        # Compute hash
        weight_hash = weight.compute_hash()
        
        # Check for exact duplicate
        if weight_hash in self.weight_index:
            # Exact duplicate found
            self._add_duplicate(weight_hash, name, weight)
            return weight_hash
        
        # Check for similar weights
        similar_ref = self._find_similar_weight(weight)
        if similar_ref:
            # Similar weight found
            self._add_similar(similar_ref, name, weight)
            return similar_ref
        
        # New unique weight
        self._add_unique_weight(weight_hash, name, weight)
        return weight_hash
    
    def _add_duplicate(self, ref_hash: str, name: str, weight: WeightTensor):
        """Add an exact duplicate to existing group"""
        if ref_hash not in self.weight_groups:
            # Create group if it doesn't exist
            self.weight_groups[ref_hash] = WeightGroup(
                reference_hash=ref_hash,
                reference_weight=self.weight_index[ref_hash]
            )
        
        self.weight_groups[ref_hash].duplicates.append((name, weight))
        self.name_to_hash[name] = ref_hash
        self.stats.duplicate_weights += 1
        logger.debug(f"Found exact duplicate: {name} -> {ref_hash}")
    
    def _add_similar(self, ref_hash: str, name: str, weight: WeightTensor):
        """Add a similar weight to existing group"""
        ref_weight = self.weight_index[ref_hash]
        similarity = self._compute_similarity(weight, ref_weight)
        
        if ref_hash not in self.weight_groups:
            self.weight_groups[ref_hash] = WeightGroup(
                reference_hash=ref_hash,
                reference_weight=ref_weight
            )
        
        self.weight_groups[ref_hash].similar.append((name, weight, similarity))
        self.name_to_hash[name] = ref_hash
        self.stats.similar_weights += 1
        logger.debug(f"Found similar weight: {name} -> {ref_hash} (similarity: {similarity:.4f})")
    
    def _add_unique_weight(self, weight_hash: str, name: str, weight: WeightTensor):
        """Add a new unique weight"""
        self.weight_index[weight_hash] = weight
        self.name_to_hash[name] = weight_hash
        self.stats.unique_weights += 1
        logger.debug(f"Added unique weight: {name} ({weight_hash})")
    
    def _find_similar_weight(self, weight: WeightTensor) -> Optional[str]:
        """
        Find a similar weight in the index.
        
        Returns hash of similar weight or None.
        """
        # Only check weights with same shape and dtype
        candidates = [
            (hash_val, w) for hash_val, w in self.weight_index.items()
            if w.shape == weight.shape and w.dtype == weight.dtype
        ]
        
        # Find most similar weight above threshold
        best_similarity = self.similarity_threshold
        best_hash = None
        
        for hash_val, candidate in candidates:
            if weight.is_similar_to(candidate, self.similarity_threshold):
                similarity = self._compute_similarity(weight, candidate)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_hash = hash_val
        
        return best_hash
    
    def _compute_similarity(self, weight1: WeightTensor, weight2: WeightTensor) -> float:
        """Compute cosine similarity between two weights"""
        a = weight1.data.flatten()
        b = weight2.data.flatten()
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 1.0 if norm_a == norm_b else 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def get_weight_by_name(self, name: str) -> Optional[WeightTensor]:
        """Get weight by name"""
        if name not in self.name_to_hash:
            return None
        
        hash_val = self.name_to_hash[name]
        return self.weight_index.get(hash_val)
    
    def get_weight_group(self, name: str) -> Optional[WeightGroup]:
        """Get the weight group containing a named weight"""
        if name not in self.name_to_hash:
            return None
        
        hash_val = self.name_to_hash[name]
        
        # Check if it's a reference weight
        if hash_val in self.weight_groups:
            return self.weight_groups[hash_val]
        
        # Check all groups for this weight
        for group in self.weight_groups.values():
            for dup_name, _ in group.duplicates:
                if dup_name == name:
                    return group
            for sim_name, _, _ in group.similar:
                if sim_name == name:
                    return group
        
        return None
    
    def compute_stats(self) -> DeduplicationStats:
        """Compute and return deduplication statistics"""
        self.stats.total_weights = len(self.name_to_hash)
        
        # Calculate bytes
        original_bytes = 0
        deduplicated_bytes = 0
        
        for name in self.name_to_hash:
            weight = self.get_weight_by_name(name)
            if weight:
                original_bytes += weight.nbytes
        
        # Count unique weights and their bytes
        for weight in self.weight_index.values():
            deduplicated_bytes += weight.nbytes
        
        # Add estimated delta encoding for similar weights
        for group in self.weight_groups.values():
            # Estimate 50% size for delta-encoded similar weights
            for _, weight, _ in group.similar:
                deduplicated_bytes += weight.nbytes // 2
        
        self.stats.update(original_bytes, deduplicated_bytes)
        return self.stats
    
    def get_deduplication_report(self) -> Dict[str, Any]:
        """Get detailed deduplication report"""
        stats = self.compute_stats()
        
        # Find largest groups
        largest_groups = sorted(
            self.weight_groups.values(),
            key=lambda g: g.bytes_saved,
            reverse=True
        )[:10]
        
        return {
            'summary': {
                'total_weights': stats.total_weights,
                'unique_weights': stats.unique_weights,
                'duplicate_weights': stats.duplicate_weights,
                'similar_weights': stats.similar_weights,
                'bytes_saved': stats.bytes_saved,
                'compression_ratio': stats.compression_ratio,
            },
            'largest_groups': [
                {
                    'reference_name': group.reference_weight.metadata.name,
                    'total_weights': group.total_count,
                    'duplicates': len(group.duplicates),
                    'similar': len(group.similar),
                    'bytes_saved': group.bytes_saved,
                }
                for group in largest_groups
            ]
        }
    
    def clear(self):
        """Clear all stored weights and statistics"""
        self.weight_index.clear()
        self.weight_groups.clear()
        self.name_to_hash.clear()
        self.stats = DeduplicationStats()