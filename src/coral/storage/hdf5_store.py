"""HDF5-based storage backend for weight tensors"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

import h5py
import numpy as np

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.delta.delta_encoder import Delta
from coral.delta.product_quantization import PQCodebook
from coral.storage.weight_store import WeightStore
from coral.storage.graph_storage import GraphSerializer, GraphStorageFormat
from coral.utils.thread_safety import FileLock, RepositoryLockManager

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
        compression: Optional[str] = "gzip",
        compression_opts: Optional[int] = 4,
        mode: str = "a",
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
        
        # Thread safety
        self._lock = threading.RLock()
        self._file_lock = FileLock(self.filepath)
        self._is_open = False
        self.file = None

        # Ensure directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Open file with proper locking
        self._open()

    def _open(self):
        """Open HDF5 file with proper locking."""
        with self._lock:
            if self._is_open:
                return
                
            # Acquire file lock for opening
            self._file_lock.acquire()
            
            try:
                # Open HDF5 file
                self.file = h5py.File(self.filepath, self.mode)
                self._is_open = True
                
                # Create groups if they don't exist
                if self.mode in ["w", "a"]:
                    if "weights" not in self.file:
                        self.file.create_group("weights")
                    if "metadata" not in self.file:
                        self.file.create_group("metadata")
                    if "deltas" not in self.file:
                        self.file.create_group("deltas")
                    if "clustered_weights" not in self.file:
                        self.file.create_group("clustered_weights")
                    if "centroids" not in self.file:
                        self.file.create_group("centroids")
                    if "pq_codebooks" not in self.file:
                        self.file.create_group("pq_codebooks")
                    if "computation_graphs" not in self.file:
                        self.file.create_group("computation_graphs")
                    
                    # Store version info for migration support
                    if "version" not in self.file.attrs:
                        self.file.attrs["version"] = "3.0"  # Version with computation graph support
                        self.file.attrs["created_at"] = str(np.datetime64('now'))
                
                # Handle migration for older files
                elif self.mode in ["r+", "a"]:
                    # Check version and migrate if needed
                    file_version = self.file.attrs.get("version", "1.0")
                    
                    # Migrate to 2.0 if needed (PQ support)
                    if file_version < "2.0":
                        logger.info(f"Migrating HDF5 file from version {file_version} to 2.0")
                        if "pq_codebooks" not in self.file:
                            self.file.create_group("pq_codebooks")
                    
                    # Migrate to 3.0 if needed (computation graph support)
                    if file_version < "3.0":
                        logger.info(f"Migrating HDF5 file from version {file_version} to 3.0")
                        if "computation_graphs" not in self.file:
                            self.file.create_group("computation_graphs")
                        self.file.attrs["version"] = "3.0"
                        self.file.attrs["migrated_at"] = str(np.datetime64('now'))
                        
            except Exception:
                # Release lock on error
                self._file_lock.release()
                raise
                
    @contextmanager
    def _file_operation(self):
        """Context manager for thread-safe file operations."""
        with self._lock:
            if not self._is_open:
                self._open()
            try:
                yield self.file
            except Exception:
                # On error, close and reopen file to ensure clean state
                logger.error(f"Error during file operation on {self.filepath}")
                self._close_internal()
                self._open()
                raise

    def store(self, weight: WeightTensor, hash_key: Optional[str] = None) -> str:
        """Store a weight tensor"""
        if hash_key is None:
            hash_key = weight.compute_hash()

        with self._file_operation() as f:
            # Check if already exists
            if self.exists(hash_key):
                logger.debug(f"Weight {hash_key} already exists in storage")
                return hash_key

            # Store weight data
            weights_group = f["weights"]
            dataset = weights_group.create_dataset(
                hash_key,
                data=weight.data,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            # Store metadata as attributes
            metadata = weight.metadata
            dataset.attrs["name"] = metadata.name
            dataset.attrs["shape"] = metadata.shape
            dataset.attrs["dtype"] = np.dtype(metadata.dtype).name
            dataset.attrs["layer_type"] = metadata.layer_type or ""
            dataset.attrs["model_name"] = metadata.model_name or ""
            dataset.attrs["compression_info"] = json.dumps(metadata.compression_info)
            dataset.attrs["hash"] = hash_key

            # Flush to ensure data is written
            f.flush()

            logger.debug(f"Stored weight {metadata.name} with hash {hash_key}")
            return hash_key

    def load(self, hash_key: str) -> Optional[WeightTensor]:
        """Load a weight tensor by hash"""
        with self._file_operation() as f:
            # Check if this is a clustered weight
            if self.is_weight_clustered(hash_key):
                return self._load_clustered_weight(hash_key)
            
            # Otherwise load normally
            if not self.exists(hash_key):
                return None

            dataset = f["weights"][hash_key]

            # Load data
            data = np.array(dataset)

            # Load metadata from attributes
            # Ensure shape is converted to tuple of Python ints to maintain consistency
            shape_array = dataset.attrs["shape"]
            normalized_shape = tuple(int(dim) for dim in shape_array)

            metadata = WeightMetadata(
                name=dataset.attrs["name"],
                shape=normalized_shape,
                dtype=np.dtype(dataset.attrs["dtype"]),
                layer_type=dataset.attrs.get("layer_type") or None,
                model_name=dataset.attrs.get("model_name") or None,
                compression_info=json.loads(dataset.attrs.get("compression_info", "{}")),
                hash=dataset.attrs.get("hash", hash_key),
            )

            return WeightTensor(data=data, metadata=metadata, store_ref=hash_key)

    def exists(self, hash_key: str) -> bool:
        """Check if a weight exists in storage"""
        with self._file_operation() as f:
            # Check both regular weights and clustered weights
            return hash_key in f["weights"] or hash_key in f["clustered_weights"]

    def delete(self, hash_key: str) -> bool:
        """Delete a weight from storage"""
        with self._file_operation() as f:
            if not self.exists(hash_key):
                return False
            
            deleted = False
            
            # Delete from regular weights if present
            if hash_key in f["weights"]:
                del f["weights"][hash_key]
                deleted = True
            
            # Delete from clustered weights if present
            if hash_key in f["clustered_weights"]:
                # Note: This doesn't delete the delta or centroid,
                # as they may be shared by other weights
                del f["clustered_weights"][hash_key]
                deleted = True
            
            if deleted:
                f.flush()
            
            return deleted

    def list_weights(self) -> List[str]:
        """List all weight hashes in storage"""
        with self._file_operation() as f:
            # Include both regular weights and clustered weights
            regular_weights = list(f["weights"].keys())
            clustered_weights = list(f["clustered_weights"].keys())
            # Return unique list (in case of overlap during migration)
            return list(set(regular_weights + clustered_weights))

    def get_metadata(self, hash_key: str) -> Optional[WeightMetadata]:
        """Get metadata for a weight without loading data"""
        with self._file_operation() as f:
            if not self.exists(hash_key):
                return None
            
            # Check if this is a clustered weight
            if self.is_weight_clustered(hash_key):
                dataset = f["clustered_weights"][hash_key]
            else:
                dataset = f["weights"][hash_key]

            # Ensure shape is converted to tuple of Python ints to maintain consistency
            shape_array = dataset.attrs["shape"]
            normalized_shape = tuple(int(dim) for dim in shape_array)

            return WeightMetadata(
                name=dataset.attrs["name"],
                shape=normalized_shape,
                dtype=np.dtype(dataset.attrs["dtype"]),
                layer_type=dataset.attrs.get("layer_type") or None,
                model_name=dataset.attrs.get("model_name") or None,
                compression_info=json.loads(dataset.attrs.get("compression_info", "{}")),
                hash=dataset.attrs.get("hash", hash_key),
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
        with self._file_operation() as f:
            weights_group = f["weights"]
            clustered_group = f["clustered_weights"] if "clustered_weights" in f else None
            centroids_group = f["centroids"] if "centroids" in f else None
            pq_group = f["pq_codebooks"] if "pq_codebooks" in f else None
            graphs_group = f.get(GraphStorageFormat.GROUP_NAME)

            total_weights = len(weights_group)
            total_clustered = len(clustered_group) if clustered_group else 0
            total_centroids = len(centroids_group) if centroids_group else 0
            total_pq_codebooks = len(pq_group) if pq_group else 0
            total_graphs = len(graphs_group) if graphs_group else 0
            
            total_bytes = 0
            compressed_bytes = 0

            # Regular weights
            for key in weights_group:
                dataset = weights_group[key]
                total_bytes += dataset.nbytes
                compressed_bytes += dataset.id.get_storage_size()
            
            # Centroids
            if centroids_group:
                for key in centroids_group:
                    dataset = centroids_group[key]
                    total_bytes += dataset.nbytes
                    compressed_bytes += dataset.id.get_storage_size()
            
            # PQ Codebooks
            if pq_group:
                for codebook_id in pq_group:
                    codebook_group = pq_group[codebook_id]
                    if "codebooks" in codebook_group:
                        dataset = codebook_group["codebooks"]
                        total_bytes += dataset.nbytes
                        compressed_bytes += dataset.id.get_storage_size()
            
            # Computation graphs
            if graphs_group:
                graph_info = self.get_graph_storage_info()
                total_bytes += graph_info["total_graph_bytes"]
                # Graphs are already compressed via node data compression
                compressed_bytes += graph_info["total_graph_bytes"]

            compression_ratio = (
                1.0 - (compressed_bytes / total_bytes) if total_bytes > 0 else 0.0
            )

            return {
                "filepath": str(self.filepath),
                "file_size": os.path.getsize(self.filepath)
                if self.filepath.exists()
                else 0,
                "total_weights": total_weights,
                "total_clustered_weights": total_clustered,
                "total_centroids": total_centroids,
                "total_pq_codebooks": total_pq_codebooks,
                "total_graphs": total_graphs,
                "total_bytes": total_bytes,
                "compressed_bytes": compressed_bytes,
                "compression_ratio": compression_ratio,
                "compression": self.compression,
            }

    def _close_internal(self):
        """Internal close method without lock."""
        if hasattr(self, "file") and self.file:
            try:
                self.file.close()
            except Exception:
                pass
            self.file = None
            self._is_open = False

    def close(self):
        """Close the HDF5 file"""
        with self._lock:
            self._close_internal()
            if hasattr(self, "_file_lock") and self._file_lock:
                self._file_lock.release()

    def store_delta(self, delta: Delta, delta_hash: str) -> str:
        """Store a delta object."""
        # Validate delta_hash type
        if not isinstance(delta_hash, str):
            raise TypeError(f"delta_hash must be a string, got {type(delta_hash)}: {repr(delta_hash)}")
        
        with self._file_operation() as f:
            if self.delta_exists(delta_hash):
                logger.debug(f"Delta {delta_hash} already exists in storage")
                return delta_hash

            deltas_group = f["deltas"]

            # Handle PQ delta types specially
            from coral.delta.delta_encoder import DeltaType
            if delta.delta_type in [DeltaType.PQ_ENCODED, DeltaType.PQ_LOSSLESS]:
                # For PQ deltas, store indices as uint8/uint16 array
                indices = delta.data
                if indices.dtype not in [np.uint8, np.uint16]:
                    # Convert to appropriate dtype based on max value
                    max_val = np.max(indices)
                    if max_val <= 255:
                        indices = indices.astype(np.uint8)
                    else:
                        indices = indices.astype(np.uint16)
                
                # Create dataset for indices
                dataset = deltas_group.create_dataset(
                    delta_hash,
                    data=indices,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )
                
                # Store residual separately if present
                if "residual" in delta.metadata and delta.metadata["residual"] is not None:
                    residual_data = delta.metadata["residual"]
                    residual_dataset = deltas_group.create_dataset(
                        f"{delta_hash}_residual",
                        data=residual_data,
                        compression=self.compression,
                        compression_opts=self.compression_opts,
                    )
                    # Store residual info in metadata
                    delta.metadata["has_residual"] = True
                    delta.metadata["residual_shape"] = residual_data.shape
                    delta.metadata["residual_dtype"] = str(residual_data.dtype)
                    # Remove actual residual data from metadata
                    delta_metadata = delta.metadata.copy()
                    del delta_metadata["residual"]
                else:
                    delta_metadata = delta.metadata
                    
                # Store codebook_id reference in metadata
                if "codebook_id" not in delta_metadata:
                    logger.warning(f"PQ delta {delta_hash} missing codebook_id")
            else:
                # Store delta data normally for non-PQ types
                dataset = deltas_group.create_dataset(
                    delta_hash,
                    data=delta.data,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )
                delta_metadata = delta.metadata

            # Store delta metadata as attributes
            dataset.attrs["delta_type"] = delta.delta_type.value
            dataset.attrs["original_shape"] = delta.original_shape
            dataset.attrs["original_dtype"] = np.dtype(delta.original_dtype).name
            dataset.attrs["reference_hash"] = delta.reference_hash
            dataset.attrs["compression_ratio"] = delta.compression_ratio
            dataset.attrs["metadata"] = json.dumps(delta_metadata)

            f.flush()
            logger.debug(f"Stored delta {delta_hash}")
            return delta_hash

    def load_delta(self, delta_hash: str) -> Optional[Delta]:
        """Load a delta object by hash."""
        # Validate delta_hash type
        if not isinstance(delta_hash, str):
            raise TypeError(f"delta_hash must be a string, got {type(delta_hash)}: {repr(delta_hash)}")
        
        with self._file_operation() as f:
            if not self.delta_exists(delta_hash):
                return None

            dataset = f["deltas"][delta_hash]
            metadata = json.loads(dataset.attrs.get("metadata", "{}"))

            # Reconstruct delta object
            from coral.delta.delta_encoder import DeltaType
            delta_type = DeltaType(dataset.attrs["delta_type"])
            
            # Handle PQ delta reconstruction
            if delta_type in [DeltaType.PQ_ENCODED, DeltaType.PQ_LOSSLESS]:
                indices = np.array(dataset)
                
                # Load residual if present
                if metadata.get("has_residual", False):
                    residual_key = f"{delta_hash}_residual"
                    if residual_key in f["deltas"]:
                        residual_data = np.array(f["deltas"][residual_key])
                        metadata["residual"] = residual_data
                    else:
                        logger.warning(f"Residual data missing for PQ delta {delta_hash}")
                
                # Verify codebook exists (but don't load it)
                if "codebook_id" in metadata:
                    codebook_id = metadata["codebook_id"]
                    if not self.has_pq_codebook(codebook_id):
                        logger.warning(f"Codebook {codebook_id} not found for delta {delta_hash}")
                
                delta = Delta(
                    delta_type=delta_type,
                    data=indices,
                    metadata=metadata,
                    original_shape=tuple(dataset.attrs["original_shape"]),
                    original_dtype=np.dtype(dataset.attrs["original_dtype"]),
                    reference_hash=dataset.attrs["reference_hash"],
                    compression_ratio=dataset.attrs.get("compression_ratio", 0.0),
                )
            else:
                # Normal delta loading
                delta = Delta(
                    delta_type=delta_type,
                    data=np.array(dataset),
                    metadata=metadata,
                    original_shape=tuple(dataset.attrs["original_shape"]),
                    original_dtype=np.dtype(dataset.attrs["original_dtype"]),
                    reference_hash=dataset.attrs["reference_hash"],
                    compression_ratio=dataset.attrs.get("compression_ratio", 0.0),
                )

            return delta

    def _compute_delta_hash(self, delta: Delta) -> str:
        """Compute hash for a delta object.
        
        Args:
            delta: Delta object to hash
            
        Returns:
            Hexadecimal hash string
        """
        import xxhash
        
        hasher = xxhash.xxh3_64()
        hasher.update(delta.reference_hash.encode())
        hasher.update(delta.data.tobytes())
        hasher.update(str(delta.delta_type.value).encode())
        return hasher.hexdigest()
    
    def delta_exists(self, delta_hash: str) -> bool:
        """Check if a delta exists in storage."""
        with self._file_operation() as f:
            return delta_hash in f["deltas"]

    def delete_delta(self, delta_hash: str) -> bool:
        """Delete a delta from storage."""
        with self._file_operation() as f:
            if not self.delta_exists(delta_hash):
                return False

            del f["deltas"][delta_hash]
            f.flush()
            return True

    def list_deltas(self) -> List[str]:
        """List all delta hashes in storage."""
        with self._file_operation() as f:
            return list(f["deltas"].keys())

    def get_delta_storage_info(self) -> Dict[str, Any]:
        """Get information about delta storage."""
        with self._file_operation() as f:
            if "deltas" not in f:
                return {"total_deltas": 0, "total_delta_bytes": 0}

            deltas_group = f["deltas"]
            total_deltas = len(deltas_group)
            total_delta_bytes = 0

            for delta_hash in deltas_group:
                dataset = deltas_group[delta_hash]
                total_delta_bytes += dataset.nbytes

            return {"total_deltas": total_deltas, "total_delta_bytes": total_delta_bytes}
    
    def get_pq_storage_info(self) -> Dict[str, Any]:
        """Get information about PQ codebook storage."""
        with self._file_operation() as f:
            if "pq_codebooks" not in f:
                return {
                    "total_pq_codebooks": 0,
                    "total_pq_bytes": 0,
                    "codebook_stats": []
                }
            
            pq_group = f["pq_codebooks"]
            total_codebooks = len(pq_group)
            total_bytes = 0
            codebook_stats = []
            
            for codebook_id in pq_group:
                codebook_group = pq_group[codebook_id]
                if "codebooks" in codebook_group:
                    dataset = codebook_group["codebooks"]
                    bytes_size = dataset.nbytes
                    total_bytes += bytes_size
                    
                    codebook_stats.append({
                        "id": codebook_id,
                        "num_subvectors": int(codebook_group.attrs.get("num_subvectors", 0)),
                        "num_codewords": int(codebook_group.attrs.get("num_codewords", 0)),
                        "subvector_size": int(codebook_group.attrs.get("subvector_size", 0)),
                        "bytes": bytes_size,
                        "version": codebook_group.attrs.get("version", "unknown")
                    })
            
            return {
                "total_pq_codebooks": total_codebooks,
                "total_pq_bytes": total_bytes,
                "codebook_stats": codebook_stats
            }
    
    # PQ Codebook operations
    
    def store_pq_codebook(self, codebook_id: str, codebook: PQCodebook) -> None:
        """Store a Product Quantization codebook.
        
        Args:
            codebook_id: Unique identifier for the codebook
            codebook: PQCodebook object to store
        """
        with self._file_operation() as f:
            pq_group = f["pq_codebooks"]
            
            # Create group for this codebook
            if codebook_id in pq_group:
                logger.warning(f"Overwriting existing PQ codebook {codebook_id}")
                del pq_group[codebook_id]
            
            codebook_group = pq_group.create_group(codebook_id)
            
            # Store metadata
            codebook_group.attrs["version"] = codebook.version
            codebook_group.attrs["subvector_size"] = codebook.subvector_size
            codebook_group.attrs["num_subvectors"] = len(codebook.codebooks)
            codebook_group.attrs["num_codewords"] = codebook.codebooks[0].shape[0] if codebook.codebooks else 0
            codebook_group.attrs["training_stats"] = json.dumps(codebook.training_stats)
            
            # Store codebooks array
            # Stack all codebooks into a single 3D array for efficient storage
            codebooks_array = np.stack(codebook.codebooks, axis=0)
            codebook_group.create_dataset(
                "codebooks",
                data=codebooks_array,
                compression=self.compression,
                compression_opts=self.compression_opts,
                dtype=np.float32
            )
            
            f.flush()
            logger.debug(f"Stored PQ codebook {codebook_id}")
    
    def load_pq_codebook(self, codebook_id: str) -> Optional[PQCodebook]:
        """Load a Product Quantization codebook.
        
        Args:
            codebook_id: Unique identifier for the codebook
            
        Returns:
            PQCodebook object or None if not found
        """
        with self._file_operation() as f:
            if not self.has_pq_codebook(codebook_id):
                return None
            
            codebook_group = f["pq_codebooks"][codebook_id]
            
            # Load codebooks array
            codebooks_array = np.array(codebook_group["codebooks"])
            
            # Split back into list of codebooks
            codebooks = [codebooks_array[i] for i in range(codebooks_array.shape[0])]
            
            # Reconstruct PQCodebook
            return PQCodebook(
                subvector_size=int(codebook_group.attrs["subvector_size"]),
                codebooks=codebooks,
                version=codebook_group.attrs["version"],
                training_stats=json.loads(codebook_group.attrs.get("training_stats", "{}"))
            )
    
    def has_pq_codebook(self, codebook_id: str) -> bool:
        """Check if a PQ codebook exists.
        
        Args:
            codebook_id: Unique identifier for the codebook
            
        Returns:
            True if codebook exists, False otherwise
        """
        with self._file_operation() as f:
            return codebook_id in f["pq_codebooks"]
    
    def list_pq_codebooks(self) -> List[str]:
        """List all PQ codebook IDs.
        
        Returns:
            List of codebook IDs
        """
        with self._file_operation() as f:
            return list(f["pq_codebooks"].keys())
    
    def delete_pq_codebook(self, codebook_id: str) -> None:
        """Delete a PQ codebook.
        
        Args:
            codebook_id: Unique identifier for the codebook
        """
        with self._file_operation() as f:
            if codebook_id in f["pq_codebooks"]:
                del f["pq_codebooks"][codebook_id]
                f.flush()
                logger.debug(f"Deleted PQ codebook {codebook_id}")
            else:
                logger.warning(f"PQ codebook {codebook_id} not found for deletion")
    
    # Clustered weight operations
    
    def store_clustered_weight(
        self,
        weight_hash: str,
        cluster_id: str,
        delta_hash: str,
        centroid_hash: str,
        metadata: WeightMetadata
    ) -> str:
        """Store a weight that exists only as a delta from a centroid.
        
        Args:
            weight_hash: Hash of the original weight
            cluster_id: ID of the cluster this weight belongs to
            delta_hash: Hash of the delta object
            centroid_hash: Hash of the centroid
            metadata: Weight metadata
            
        Returns:
            Storage key for the clustered weight
        """
        with self._file_operation() as f:
            clustered_weights_group = f["clustered_weights"]
            
            # Store mapping information
            storage_key = weight_hash
            
            if storage_key in clustered_weights_group:
                # Update existing entry
                dataset = clustered_weights_group[storage_key]
            else:
                # Create new entry (small dataset just for metadata)
                dataset = clustered_weights_group.create_dataset(
                    storage_key,
                    shape=(1,),
                    dtype=np.int32
                )
            
            # Store all mapping information as attributes
            dataset.attrs["cluster_id"] = cluster_id
            dataset.attrs["delta_hash"] = delta_hash
            dataset.attrs["centroid_hash"] = centroid_hash
            dataset.attrs["is_clustered"] = True
            # Store timestamp as ISO string for HDF5 compatibility
            import datetime
            dataset.attrs["stored_at"] = datetime.datetime.now().isoformat()
            
            # Store metadata
            dataset.attrs["name"] = metadata.name
            dataset.attrs["shape"] = metadata.shape
            dataset.attrs["dtype"] = np.dtype(metadata.dtype).name
            dataset.attrs["layer_type"] = metadata.layer_type or ""
            dataset.attrs["model_name"] = metadata.model_name or ""
            dataset.attrs["compression_info"] = json.dumps(metadata.compression_info)
            dataset.attrs["hash"] = weight_hash
            
            f.flush()
            logger.debug(f"Stored clustered weight {weight_hash} -> cluster {cluster_id}")
            
            return storage_key
    
    def is_weight_clustered(self, weight_hash: str) -> bool:
        """Check if a weight is stored as clustered (delta-only).
        
        Args:
            weight_hash: Hash of the weight to check
            
        Returns:
            True if weight is clustered, False otherwise
        """
        with self._file_operation() as f:
            return weight_hash in f["clustered_weights"]
    
    def get_clustered_weight_info(self, weight_hash: str) -> Optional[Dict[str, Any]]:
        """Get clustering information for a weight.
        
        Args:
            weight_hash: Hash of the weight
            
        Returns:
            Dictionary with clustering info or None if not clustered
        """
        with self._file_operation() as f:
            if weight_hash not in f["clustered_weights"]:
                return None
            
            dataset = f["clustered_weights"][weight_hash]
            
            return {
                "cluster_id": dataset.attrs["cluster_id"],
                "delta_hash": dataset.attrs["delta_hash"],
                "centroid_hash": dataset.attrs["centroid_hash"],
                "is_clustered": bool(dataset.attrs.get("is_clustered", True)),
                "stored_at": dataset.attrs.get("stored_at", None)
            }
    
    def store_centroid(self, centroid: WeightTensor, centroid_hash: Optional[str] = None) -> str:
        """Store a centroid for cluster-based storage.
        
        Args:
            centroid: Centroid weight tensor
            centroid_hash: Optional hash (computed if not provided)
            
        Returns:
            Hash of the stored centroid
        """
        if centroid_hash is None:
            centroid_hash = centroid.compute_hash()
        
        with self._file_operation() as f:
            centroids_group = f["centroids"]
            
            # Check if already exists
            if centroid_hash in centroids_group:
                logger.debug(f"Centroid {centroid_hash} already exists")
                return centroid_hash
            
            # Store centroid data
            dataset = centroids_group.create_dataset(
                centroid_hash,
                data=centroid.data,
                compression=self.compression,
                compression_opts=self.compression_opts
            )
            
            # Store metadata
            metadata = centroid.metadata
            dataset.attrs["name"] = metadata.name
            dataset.attrs["shape"] = metadata.shape
            dataset.attrs["dtype"] = np.dtype(metadata.dtype).name
            dataset.attrs["layer_type"] = metadata.layer_type or ""
            dataset.attrs["model_name"] = metadata.model_name or ""
            dataset.attrs["compression_info"] = json.dumps(metadata.compression_info)
            dataset.attrs["hash"] = centroid_hash
            dataset.attrs["is_centroid"] = True
            
            f.flush()
            logger.debug(f"Stored centroid {centroid_hash}")
            
            return centroid_hash
    
    def load_centroid(self, centroid_hash: str) -> Optional[WeightTensor]:
        """Load a centroid by hash.
        
        Args:
            centroid_hash: Hash of the centroid
            
        Returns:
            Centroid weight tensor or None if not found
        """
        with self._file_operation() as f:
            if centroid_hash not in f["centroids"]:
                return None
            
            dataset = f["centroids"][centroid_hash]
            
            # Load data
            data = np.array(dataset)
            
            # Load metadata
            shape_array = dataset.attrs["shape"]
            normalized_shape = tuple(int(dim) for dim in shape_array)
            
            metadata = WeightMetadata(
                name=dataset.attrs["name"],
                shape=normalized_shape,
                dtype=np.dtype(dataset.attrs["dtype"]),
                layer_type=dataset.attrs.get("layer_type") or None,
                model_name=dataset.attrs.get("model_name") or None,
                compression_info=json.loads(dataset.attrs.get("compression_info", "{}")),
                hash=dataset.attrs.get("hash", centroid_hash)
            )
            
            return WeightTensor(data=data, metadata=metadata, store_ref=centroid_hash)
    
    def migrate_weight_to_clustered(
        self,
        weight_hash: str,
        cluster_id: str,
        delta_hash: str,
        centroid_hash: str,
        remove_original: bool = True
    ) -> bool:
        """Migrate an existing weight to clustered storage.
        
        Args:
            weight_hash: Hash of the weight to migrate
            cluster_id: Target cluster ID
            delta_hash: Hash of the delta object
            centroid_hash: Hash of the centroid
            remove_original: Whether to remove the original weight data
            
        Returns:
            True if migration successful
        """
        with self._file_operation() as f:
            # Check if weight exists
            if weight_hash not in f["weights"]:
                logger.warning(f"Weight {weight_hash} not found for migration")
                return False
            
            # Get weight metadata
            metadata = self.get_metadata(weight_hash)
            if metadata is None:
                logger.error(f"Could not load metadata for weight {weight_hash}")
                return False
            
            # Store as clustered weight
            self.store_clustered_weight(
                weight_hash=weight_hash,
                cluster_id=cluster_id,
                delta_hash=delta_hash,
                centroid_hash=centroid_hash,
                metadata=metadata
            )
            
            # Remove original weight data if requested
            if remove_original:
                del f["weights"][weight_hash]
                logger.debug(f"Removed original weight data for {weight_hash}")
            
            f.flush()
            logger.info(f"Migrated weight {weight_hash} to clustered storage")
            
            return True
    
    def list_clustered_weights(self) -> List[str]:
        """List all weights stored as clustered."""
        with self._file_operation() as f:
            return list(f["clustered_weights"].keys())
    
    def get_clustered_storage_info(self) -> Dict[str, Any]:
        """Get information about clustered weight storage."""
        with self._file_operation() as f:
            clustered_group = f["clustered_weights"]
            centroids_group = f["centroids"]
            
            total_clustered = len(clustered_group)
            total_centroids = len(centroids_group)
            
            # Calculate space savings
            original_size = 0
            clustered_size = 0
            
            for weight_hash in clustered_group:
                info = self.get_clustered_weight_info(weight_hash)
                if info:
                    # Estimate original size from metadata
                    dataset = clustered_group[weight_hash]
                    shape = dataset.attrs["shape"]
                    dtype = np.dtype(dataset.attrs["dtype"])
                    original_size += np.prod(shape) * dtype.itemsize
                    
                    # Get delta size
                    if info["delta_hash"] in f["deltas"]:
                        delta_dataset = f["deltas"][info["delta_hash"]]
                        clustered_size += delta_dataset.nbytes
            
            # Add centroid sizes (but divide by the number of weights sharing each centroid)
            # For now, we'll estimate that each centroid is shared by at least 2 weights
            # In practice, this would be tracked more accurately
            centroid_total_size = 0
            for centroid_hash in centroids_group:
                dataset = centroids_group[centroid_hash]
                centroid_total_size += dataset.nbytes
            
            # Assume each centroid is shared by at least 2 weights on average
            # This is a simplified calculation - in production you'd track actual sharing
            effective_centroid_size = centroid_total_size / max(1, min(2, total_clustered))
            clustered_size += effective_centroid_size
            
            compression_ratio = original_size / clustered_size if clustered_size > 0 else 1.0
            
            return {
                "total_clustered_weights": total_clustered,
                "total_centroids": total_centroids,
                "estimated_original_size": original_size,
                "actual_clustered_size": clustered_size,
                "compression_ratio": compression_ratio,
                "space_savings_percent": (1.0 - 1.0/compression_ratio) * 100 if compression_ratio > 1 else 0
            }
    
    def _load_clustered_weight(self, weight_hash: str) -> Optional[WeightTensor]:
        """Load a weight that is stored as clustered (centroid + delta).
        
        Args:
            weight_hash: Hash of the weight to load
            
        Returns:
            Reconstructed weight tensor or None if loading fails
        """
        with self._file_operation() as f:
            # Get clustering info
            cluster_info = self.get_clustered_weight_info(weight_hash)
            if not cluster_info:
                logger.error(f"No clustering info found for weight {weight_hash}")
                return None
            
            # Load centroid
            centroid = self.load_centroid(cluster_info["centroid_hash"])
            if centroid is None:
                logger.error(f"Centroid {cluster_info['centroid_hash']} not found")
                return None
            
            # Load delta
            delta = self.load_delta(cluster_info["delta_hash"])
            if delta is None:
                logger.error(f"Delta {cluster_info['delta_hash']} not found")
                return None
            
            # Reconstruct weight from centroid + delta
            try:
                # Import delta encoder for reconstruction
                from coral.delta.delta_encoder import DeltaEncoder
                
                encoder = DeltaEncoder()
                reconstructed = encoder.decode_delta(delta, centroid)
                
                # Load metadata from clustered_weights
                dataset = f["clustered_weights"][weight_hash]
                
                shape_array = dataset.attrs["shape"]
                normalized_shape = tuple(int(dim) for dim in shape_array)
                
                metadata = WeightMetadata(
                    name=dataset.attrs["name"],
                    shape=normalized_shape,
                    dtype=np.dtype(dataset.attrs["dtype"]),
                    layer_type=dataset.attrs.get("layer_type") or None,
                    model_name=dataset.attrs.get("model_name") or None,
                    compression_info=json.loads(dataset.attrs.get("compression_info", "{}")),
                    hash=weight_hash
                )
                
                # Create weight tensor with reconstructed data and original metadata
                weight = WeightTensor(
                    data=reconstructed.data,
                    metadata=metadata,
                    store_ref=weight_hash
                )
                
                logger.debug(f"Successfully reconstructed clustered weight {weight_hash}")
                return weight
                
            except Exception as e:
                logger.error(f"Error reconstructing weight {weight_hash}: {e}")
                return None
    
    def gc(self) -> Dict[str, int]:
        """Garbage collect unreferenced objects.
        
        Removes:
        - Orphaned deltas not referenced by any weight or cluster
        - Orphaned centroids not referenced by any cluster
        - Orphaned PQ codebooks not referenced by any delta
        
        Returns:
            Dictionary with counts of deleted objects
        """
        with self._file_operation() as f:
            deleted_counts = {
                "deltas": 0,
                "centroids": 0,
                "pq_codebooks": 0
            }
            
            # Collect all referenced deltas and centroids
            referenced_deltas = set()
            referenced_centroids = set()
            referenced_codebooks = set()
            
            # From clustered weights
            if "clustered_weights" in f:
                for weight_hash in f["clustered_weights"]:
                    info = self.get_clustered_weight_info(weight_hash)
                    if info:
                        referenced_deltas.add(info["delta_hash"])
                        referenced_centroids.add(info["centroid_hash"])
            
            # From deltas - collect referenced codebooks
            if "deltas" in f:
                for delta_hash in f["deltas"]:
                    dataset = f["deltas"][delta_hash]
                    metadata = json.loads(dataset.attrs.get("metadata", "{}"))
                    if "codebook_id" in metadata:
                        referenced_codebooks.add(metadata["codebook_id"])
            
            # Clean orphaned deltas
            if "deltas" in f:
                all_deltas = set(f["deltas"].keys())
                # Remove residual keys from the set
                all_deltas = {d for d in all_deltas if not d.endswith("_residual")}
                orphaned_deltas = all_deltas - referenced_deltas
                
                for delta_hash in orphaned_deltas:
                    del f["deltas"][delta_hash]
                    # Also delete residual if present
                    residual_key = f"{delta_hash}_residual"
                    if residual_key in f["deltas"]:
                        del f["deltas"][residual_key]
                    deleted_counts["deltas"] += 1
                    logger.debug(f"Deleted orphaned delta {delta_hash}")
            
            # Clean orphaned centroids
            if "centroids" in f:
                all_centroids = set(f["centroids"].keys())
                orphaned_centroids = all_centroids - referenced_centroids
                
                for centroid_hash in orphaned_centroids:
                    del f["centroids"][centroid_hash]
                    deleted_counts["centroids"] += 1
                    logger.debug(f"Deleted orphaned centroid {centroid_hash}")
            
            # Clean orphaned PQ codebooks
            if "pq_codebooks" in f:
                all_codebooks = set(f["pq_codebooks"].keys())
                orphaned_codebooks = all_codebooks - referenced_codebooks
                
                for codebook_id in orphaned_codebooks:
                    del f["pq_codebooks"][codebook_id]
                    deleted_counts["pq_codebooks"] += 1
                    logger.debug(f"Deleted orphaned PQ codebook {codebook_id}")
            
            if any(deleted_counts.values()):
                f.flush()
                logger.info(f"Garbage collection completed: {deleted_counts}")
            else:
                logger.debug("No orphaned objects found during garbage collection")
            
            return deleted_counts
    
    # Computation graph operations
    
    def store_computation_graph(self, graph_hash: str, graph) -> str:
        """Store a computation graph.
        
        Args:
            graph_hash: Hash identifier for the graph
            graph: ComputationGraph object to store
            
        Returns:
            Hash of the stored graph
        """
        # Import here to avoid circular imports
        from coral.core.weight_ops import ComputationGraph
        
        if not isinstance(graph, ComputationGraph):
            raise TypeError(f"Expected ComputationGraph, got {type(graph)}")
        
        with self._file_operation() as f:
            graphs_group = f[GraphStorageFormat.GROUP_NAME]
            
            # Check if already exists
            if graph_hash in graphs_group:
                logger.debug(f"Computation graph {graph_hash} already exists")
                return graph_hash
            
            # Serialize the graph
            serializer = GraphSerializer()
            serialized_data = serializer.serialize_graph(graph)
            
            # Create group for this graph
            graph_group = graphs_group.create_group(graph_hash)
            
            # Store metadata
            graph_group.attrs["version"] = GraphStorageFormat.VERSION
            graph_group.attrs["created_at"] = str(np.datetime64('now'))
            graph_group.attrs["root_id"] = serialized_data["root_id"]
            graph_group.attrs["num_nodes"] = serialized_data["metadata"]["num_nodes"]
            graph_group.attrs["num_edges"] = serialized_data["metadata"]["num_edges"]
            
            # Store nodes
            nodes_group = graph_group.create_group("nodes")
            for node_data in serialized_data["nodes"]:
                node_id = str(node_data["id"])
                node_group = nodes_group.create_group(node_id)
                
                # Store node attributes
                node_group.attrs["type"] = node_data["type"]
                node_group.attrs["metadata"] = json.dumps(node_data["data"])
                
                # If node has raw data (like IdentityOp), store it as dataset
                if "raw_data" in node_data["data"]:
                    raw_data = np.array(node_data["data"]["raw_data"])
                    node_group.create_dataset(
                        "data",
                        data=raw_data,
                        compression=self.compression,
                        compression_opts=self.compression_opts
                    )
            
            # Store edges as a dataset
            if serialized_data["edges"]:
                edges_array = np.array(serialized_data["edges"], dtype=np.int32)
                graph_group.create_dataset(
                    "edges",
                    data=edges_array,
                    compression=self.compression,
                    compression_opts=self.compression_opts
                )
            else:
                # Create empty edges dataset
                graph_group.create_dataset(
                    "edges",
                    shape=(0, 2),
                    dtype=np.int32
                )
            
            # Store additional metadata
            graph_group.attrs["metadata"] = json.dumps(serialized_data["metadata"])
            
            f.flush()
            logger.debug(f"Stored computation graph {graph_hash}")
            return graph_hash
    
    def load_computation_graph(self, graph_hash: str):
        """Load a computation graph by hash.
        
        Args:
            graph_hash: Hash identifier for the graph
            
        Returns:
            ComputationGraph object or None if not found
        """
        with self._file_operation() as f:
            graphs_group = f.get(GraphStorageFormat.GROUP_NAME)
            if graphs_group is None or graph_hash not in graphs_group:
                return None
            
            graph_group = graphs_group[graph_hash]
            
            # Check version compatibility
            version = graph_group.attrs.get("version", "1.0")
            if not GraphStorageFormat.validate_version(version):
                logger.warning(f"Incompatible graph format version: {version}")
                return None
            
            # Reconstruct serialized data
            nodes = []
            nodes_group = graph_group["nodes"]
            
            for node_id in sorted(nodes_group.keys(), key=int):
                node_group = nodes_group[node_id]
                node_data = {
                    "id": int(node_id),
                    "type": node_group.attrs["type"],
                    "data": json.loads(node_group.attrs["metadata"])
                }
                
                # Load raw data if present
                if "data" in node_group:
                    node_data["data"]["raw_data"] = np.array(node_group["data"])
                
                nodes.append(node_data)
            
            # Load edges
            if "edges" in graph_group:
                edges = np.array(graph_group["edges"]).tolist()
            else:
                edges = []
            
            # Reconstruct serialized format
            serialized_data = {
                "nodes": nodes,
                "edges": edges,
                "root_id": int(graph_group.attrs["root_id"]),
                "metadata": json.loads(graph_group.attrs.get("metadata", "{}"))
            }
            
            # Deserialize the graph
            serializer = GraphSerializer()
            graph = serializer.deserialize_graph(serialized_data)
            
            logger.debug(f"Loaded computation graph {graph_hash}")
            return graph
    
    def list_computation_graphs(self) -> List[str]:
        """List all computation graph hashes in storage.
        
        Returns:
            List of graph hash identifiers
        """
        with self._file_operation() as f:
            graphs_group = f.get(GraphStorageFormat.GROUP_NAME)
            if graphs_group is None:
                return []
            return list(graphs_group.keys())
    
    def delete_computation_graph(self, graph_hash: str) -> bool:
        """Delete a computation graph from storage.
        
        Args:
            graph_hash: Hash identifier for the graph
            
        Returns:
            True if deleted, False if not found
        """
        with self._file_operation() as f:
            graphs_group = f.get(GraphStorageFormat.GROUP_NAME)
            if graphs_group is None or graph_hash not in graphs_group:
                return False
            
            del graphs_group[graph_hash]
            f.flush()
            logger.debug(f"Deleted computation graph {graph_hash}")
            return True
    
    def get_computation_graph_info(self, graph_hash: str) -> Optional[Dict[str, Any]]:
        """Get information about a stored computation graph.
        
        Args:
            graph_hash: Hash identifier for the graph
            
        Returns:
            Dictionary with graph information or None if not found
        """
        with self._file_operation() as f:
            graphs_group = f.get(GraphStorageFormat.GROUP_NAME)
            if graphs_group is None or graph_hash not in graphs_group:
                return None
            
            graph_group = graphs_group[graph_hash]
            
            # Calculate storage size
            total_bytes = 0
            nodes_group = graph_group["nodes"]
            for node_id in nodes_group:
                node_group = nodes_group[node_id]
                if "data" in node_group:
                    total_bytes += node_group["data"].nbytes
            
            if "edges" in graph_group:
                total_bytes += graph_group["edges"].nbytes
            
            return {
                "hash": graph_hash,
                "version": graph_group.attrs.get("version", "unknown"),
                "created_at": graph_group.attrs.get("created_at", "unknown"),
                "root_id": int(graph_group.attrs.get("root_id", -1)),
                "num_nodes": int(graph_group.attrs.get("num_nodes", 0)),
                "num_edges": int(graph_group.attrs.get("num_edges", 0)),
                "storage_bytes": total_bytes
            }
    
    def get_graph_storage_info(self) -> Dict[str, Any]:
        """Get information about computation graph storage.
        
        Returns:
            Dictionary with storage statistics
        """
        with self._file_operation() as f:
            graphs_group = f.get(GraphStorageFormat.GROUP_NAME)
            if graphs_group is None:
                return {
                    "total_graphs": 0,
                    "total_graph_bytes": 0,
                    "graph_stats": []
                }
            
            total_graphs = len(graphs_group)
            total_bytes = 0
            graph_stats = []
            
            for graph_hash in graphs_group:
                info = self.get_computation_graph_info(graph_hash)
                if info:
                    total_bytes += info["storage_bytes"]
                    graph_stats.append({
                        "hash": graph_hash,
                        "num_nodes": info["num_nodes"],
                        "num_edges": info["num_edges"],
                        "bytes": info["storage_bytes"]
                    })
            
            return {
                "total_graphs": total_graphs,
                "total_graph_bytes": total_bytes,
                "graph_stats": graph_stats
            }

    def __repr__(self) -> str:
        info = self.get_storage_info()
        delta_info = self.get_delta_storage_info()
        clustered_info = self.get_clustered_storage_info()
        pq_info = self.get_pq_storage_info()
        graph_info = self.get_graph_storage_info()
        return (
            f"HDF5Store(filepath='{self.filepath}', "
            f"weights={info['total_weights']}, "
            f"clustered={clustered_info['total_clustered_weights']}, "
            f"centroids={clustered_info['total_centroids']}, "
            f"deltas={delta_info['total_deltas']}, "
            f"pq_codebooks={pq_info['total_pq_codebooks']}, "
            f"graphs={graph_info['total_graphs']}, "
            f"compression={self.compression})"
        )
    
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
