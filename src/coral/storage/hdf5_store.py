"""HDF5-based storage backend for weight tensors"""

import os
from typing import Optional, Dict, List, Any
import h5py
import numpy as np
import json
import logging
from pathlib import Path

from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.storage.weight_store import WeightStore


logger = logging.getLogger(__name__)


class HDF5Store(WeightStore):
    """
    HDF5-based storage backend for weight tensors.
    
    Features:
    - Content-addressable storage using hash-based keys
    - Compression support (gzip, lzf)
    - Efficient batch operations
    - Metadata stored as HDF5 attributes
    """
    
    def __init__(
        self, 
        filepath: str, 
        compression: Optional[str] = 'gzip',
        compression_opts: Optional[int] = 4,
        mode: str = 'a'
    ):
        """
        Initialize HDF5 storage.
        
        Args:
            filepath: Path to HDF5 file
            compression: Compression algorithm ('gzip', 'lzf', None)
            compression_opts: Compression level (1-9 for gzip)
            mode: File mode ('r', 'r+', 'w', 'a')
        """
        self.filepath = Path(filepath)
        self.compression = compression
        self.compression_opts = compression_opts
        self.mode = mode
        
        # Ensure directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Open HDF5 file
        self.file = h5py.File(self.filepath, mode)
        
        # Create groups if they don't exist
        if mode in ['w', 'a']:
            if 'weights' not in self.file:
                self.file.create_group('weights')
            if 'metadata' not in self.file:
                self.file.create_group('metadata')
    
    def store(self, weight: WeightTensor, hash_key: Optional[str] = None) -> str:
        """Store a weight tensor"""
        if hash_key is None:
            hash_key = weight.compute_hash()
        
        # Check if already exists
        if self.exists(hash_key):
            logger.debug(f"Weight {hash_key} already exists in storage")
            return hash_key
        
        # Store weight data
        weights_group = self.file['weights']
        dataset = weights_group.create_dataset(
            hash_key,
            data=weight.data,
            compression=self.compression,
            compression_opts=self.compression_opts
        )
        
        # Store metadata as attributes
        metadata = weight.metadata
        dataset.attrs['name'] = metadata.name
        dataset.attrs['shape'] = metadata.shape
        dataset.attrs['dtype'] = str(metadata.dtype)
        dataset.attrs['layer_type'] = metadata.layer_type or ''
        dataset.attrs['model_name'] = metadata.model_name or ''
        dataset.attrs['compression_info'] = json.dumps(metadata.compression_info)
        dataset.attrs['hash'] = hash_key
        
        # Flush to ensure data is written
        self.file.flush()
        
        logger.debug(f"Stored weight {metadata.name} with hash {hash_key}")
        return hash_key
    
    def load(self, hash_key: str) -> Optional[WeightTensor]:
        """Load a weight tensor by hash"""
        if not self.exists(hash_key):
            return None
        
        dataset = self.file['weights'][hash_key]
        
        # Load data
        data = np.array(dataset)
        
        # Load metadata from attributes
        metadata = WeightMetadata(
            name=dataset.attrs['name'],
            shape=tuple(dataset.attrs['shape']),
            dtype=np.dtype(dataset.attrs['dtype']),
            layer_type=dataset.attrs.get('layer_type') or None,
            model_name=dataset.attrs.get('model_name') or None,
            compression_info=json.loads(dataset.attrs.get('compression_info', '{}')),
            hash=dataset.attrs.get('hash', hash_key)
        )
        
        return WeightTensor(data=data, metadata=metadata, store_ref=hash_key)
    
    def exists(self, hash_key: str) -> bool:
        """Check if a weight exists in storage"""
        return hash_key in self.file['weights']
    
    def delete(self, hash_key: str) -> bool:
        """Delete a weight from storage"""
        if not self.exists(hash_key):
            return False
        
        del self.file['weights'][hash_key]
        self.file.flush()
        return True
    
    def list_weights(self) -> List[str]:
        """List all weight hashes in storage"""
        return list(self.file['weights'].keys())
    
    def get_metadata(self, hash_key: str) -> Optional[WeightMetadata]:
        """Get metadata for a weight without loading data"""
        if not self.exists(hash_key):
            return None
        
        dataset = self.file['weights'][hash_key]
        
        return WeightMetadata(
            name=dataset.attrs['name'],
            shape=tuple(dataset.attrs['shape']),
            dtype=np.dtype(dataset.attrs['dtype']),
            layer_type=dataset.attrs.get('layer_type') or None,
            model_name=dataset.attrs.get('model_name') or None,
            compression_info=json.loads(dataset.attrs.get('compression_info', '{}')),
            hash=dataset.attrs.get('hash', hash_key)
        )
    
    def store_batch(self, weights: Dict[str, WeightTensor]) -> Dict[str, str]:
        """Store multiple weights efficiently"""
        result = {}
        
        for name, weight in weights.items():
            hash_key = self.store(weight)
            result[name] = hash_key
        
        return result
    
    def load_batch(self, hash_keys: List[str]) -> Dict[str, WeightTensor]:
        """Load multiple weights efficiently"""
        result = {}
        
        for hash_key in hash_keys:
            weight = self.load(hash_key)
            if weight is not None:
                result[hash_key] = weight
        
        return result
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about storage usage and statistics"""
        weights_group = self.file['weights']
        
        total_weights = len(weights_group)
        total_bytes = 0
        compressed_bytes = 0
        
        for key in weights_group:
            dataset = weights_group[key]
            total_bytes += dataset.nbytes
            compressed_bytes += dataset.id.get_storage_size()
        
        compression_ratio = 1.0 - (compressed_bytes / total_bytes) if total_bytes > 0 else 0.0
        
        return {
            'filepath': str(self.filepath),
            'file_size': os.path.getsize(self.filepath) if self.filepath.exists() else 0,
            'total_weights': total_weights,
            'total_bytes': total_bytes,
            'compressed_bytes': compressed_bytes,
            'compression_ratio': compression_ratio,
            'compression': self.compression,
        }
    
    def close(self):
        """Close the HDF5 file"""
        if hasattr(self, 'file') and self.file:
            self.file.close()
    
    def __repr__(self) -> str:
        info = self.get_storage_info()
        return (f"HDF5Store(filepath='{self.filepath}', "
                f"weights={info['total_weights']}, "
                f"compression={self.compression})")