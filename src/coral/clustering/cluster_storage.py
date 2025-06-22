"""
ClusterStorage component for HDF5 cluster persistence and management.

This module provides comprehensive cluster storage capabilities built on top
of the existing HDF5Store infrastructure. It manages persistence of:
- Cluster hierarchies and metadata
- Centroids with efficient compression
- Weight-to-cluster assignments
- Delta encodings for centroid-based compression
- Hierarchy structure and relationships

Key features:
- Efficient HDF5-based storage with compression
- Thread-safe operations for concurrent access
- Backup and restoration capabilities
- Storage optimization and garbage collection
- Incremental updates and partial loading
- Transactional operations where possible
"""

import json
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np

from ..storage.hdf5_store import HDF5Store
from ..core.weight_tensor import WeightTensor, WeightMetadata
from ..delta.delta_encoder import Delta
from .cluster_types import (
    ClusterInfo, ClusterLevel, ClusteringStrategy, ClusterMetrics,
    Centroid, ClusterAssignment
)
from .cluster_hierarchy import ClusterHierarchy, HierarchyConfig

logger = logging.getLogger(__name__)


class ClusterStorage:
    """
    Comprehensive cluster storage system built on HDF5.
    
    Provides efficient storage and retrieval of:
    - Cluster metadata and hierarchies
    - Centroids with compression
    - Cluster assignments and relationships
    - Delta encodings for centroid-based storage
    
    Features:
    - Thread-safe operations
    - Efficient batch operations
    - Storage optimization
    - Backup and recovery
    - Transactional consistency
    """
    
    def __init__(
        self,
        storage_path: str,
        base_store: Optional[HDF5Store] = None,
        compression: str = "gzip",
        compression_opts: int = 6,
        mode: str = "a"
    ):
        """
        Initialize cluster storage system.
        
        Args:
            storage_path: Path to storage file
            base_store: Existing HDF5Store to extend (creates new if None)
            compression: Compression algorithm for centroids
            compression_opts: Compression level
            mode: File access mode
        """
        self.storage_path = Path(storage_path)
        self.compression = compression
        self.compression_opts = compression_opts
        self.mode = mode
        
        # Always create our own HDF5 file handle to avoid lifecycle issues
        self._owns_file = True
        self._owns_store = False
        
        # Ensure directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open HDF5 file directly
        self.file = h5py.File(str(self.storage_path), mode)
        
        # Initialize HDF5Store for delta operations
        self.store = HDF5Store(str(self.storage_path), mode=mode)
        self._owns_store = True
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize cluster-specific groups
        self._init_storage_groups()
        
        # Performance tracking
        self._operation_stats = {
            'clusters_stored': 0,
            'clusters_loaded': 0,
            'centroids_stored': 0,
            'centroids_loaded': 0,
            'assignments_stored': 0,
            'assignments_loaded': 0,
            'hierarchies_stored': 0,
            'hierarchies_loaded': 0
        }
        
        logger.info(f"ClusterStorage initialized at {storage_path}")
    
    def _init_storage_groups(self):
        """Initialize HDF5 groups for cluster storage."""
        with self._lock:
            # Create cluster-specific groups
            groups_to_create = [
                "clusters",           # Cluster metadata
                "centroids",         # Centroid data
                "assignments",       # Weight-to-cluster assignments
                "hierarchies",       # Hierarchy structures
                "centroid_deltas",   # Delta encodings for centroids
                "cluster_metadata",  # Additional cluster metadata
                "indexes"           # Index structures for fast lookup
            ]
            
            for group_name in groups_to_create:
                if group_name not in self.file:
                    self.file.create_group(group_name)
            
            # Initialize metadata group with version info
            if "cluster_storage_version" not in self.file.attrs:
                self.file.attrs["cluster_storage_version"] = "1.0"
                self.file.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Create indexes for efficient queries
            self._init_indexes()
    
    def _init_indexes(self):
        """Initialize index structures for fast queries."""
        indexes_group = self.file["indexes"]
        
        # Index structures
        index_datasets = [
            ("level_index", "S50"),      # level -> cluster_ids mapping
            ("strategy_index", "S50"),   # strategy -> cluster_ids mapping
            ("parent_index", "S50"),     # parent_id -> child_ids mapping
            ("centroid_index", "S64")    # centroid_hash -> cluster_ids mapping
        ]
        
        for index_name, dtype in index_datasets:
            if index_name not in indexes_group:
                # Create empty resizable dataset
                indexes_group.create_dataset(
                    index_name,
                    shape=(0,),
                    maxshape=(None,),
                    dtype=dtype,
                    chunks=True
                )
    
    # Core cluster operations
    
    def store_clusters(
        self,
        clusters: List[ClusterInfo],
        hierarchy: Optional[ClusterHierarchy] = None
    ) -> Dict[str, str]:
        """
        Store clusters and optional hierarchy to HDF5.
        
        Args:
            clusters: List of cluster information objects
            hierarchy: Optional hierarchy structure
            
        Returns:
            Dictionary mapping cluster_id to storage_key
        """
        with self._lock:
            logger.debug(f"Storing {len(clusters)} clusters")
            start_time = time.time()
            
            clusters_group = self.file["clusters"]
            result = {}
            
            for cluster in clusters:
                storage_key = f"cluster_{cluster.cluster_id}"
                
                # Convert cluster to serializable format
                cluster_data = cluster.to_dict()
                
                # Store cluster data as JSON in dataset attributes
                if storage_key in clusters_group:
                    del clusters_group[storage_key]
                
                # Create dataset for cluster (empty, just for metadata)
                dataset = clusters_group.create_dataset(
                    storage_key,
                    shape=(1,),
                    dtype=np.int32
                )
                
                # Store cluster metadata as attributes
                for key, value in cluster_data.items():
                    if isinstance(value, (list, dict)):
                        dataset.attrs[key] = json.dumps(value)
                    elif value is None:
                        dataset.attrs[key] = ""  # Store None as empty string
                    else:
                        dataset.attrs[key] = str(value)  # Convert to string for HDF5 compatibility
                
                # Store additional computed metadata
                dataset.attrs["storage_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                dataset.attrs["storage_key"] = storage_key
                
                result[cluster.cluster_id] = storage_key
                
                # Update indexes
                self._update_cluster_indexes(cluster)
            
            # Store hierarchy if provided
            if hierarchy:
                self.store_hierarchy(hierarchy)
            
            # Update stats
            self._operation_stats['clusters_stored'] += len(clusters)
            
            # Flush to disk
            self.file.flush()
            
            store_time = time.time() - start_time
            logger.debug(f"Stored {len(clusters)} clusters in {store_time:.3f}s")
            
            return result
    
    def load_clusters(
        self,
        cluster_ids: Optional[List[str]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[ClusterInfo]:
        """
        Load clusters from storage with optional filtering.
        
        Args:
            cluster_ids: Specific cluster IDs to load (None = all)
            filter_criteria: Filter criteria for selective loading
            
        Returns:
            List of loaded cluster information objects
        """
        with self._lock:
            logger.debug(f"Loading clusters with criteria: {filter_criteria}")
            start_time = time.time()
            
            clusters_group = self.file["clusters"]
            clusters = []
            
            # Determine which clusters to load
            if cluster_ids:
                storage_keys = [f"cluster_{cid}" for cid in cluster_ids]
            else:
                storage_keys = list(clusters_group.keys())
            
            for storage_key in storage_keys:
                if storage_key not in clusters_group:
                    continue
                
                try:
                    dataset = clusters_group[storage_key]
                    
                    # Reconstruct cluster data from attributes
                    cluster_data = {}
                    for attr_name in dataset.attrs:
                        value = dataset.attrs[attr_name]
                        
                        # Handle JSON-serialized attributes
                        if attr_name in ["child_cluster_ids"]:
                            try:
                                cluster_data[attr_name] = json.loads(value) if value else None
                            except (json.JSONDecodeError, TypeError):
                                cluster_data[attr_name] = value
                        elif attr_name in ["strategy", "level"]:
                            # Handle enum values stored as strings
                            cluster_data[attr_name] = value
                        elif attr_name == "member_count":
                            # Convert back to int
                            cluster_data[attr_name] = int(value) if value else 0
                        elif attr_name in ["storage_timestamp", "storage_key"]:
                            # Skip internal storage metadata
                            continue
                        else:
                            # Handle None values stored as empty strings
                            cluster_data[attr_name] = value if value != "" else None
                    
                    # Create cluster object
                    cluster = ClusterInfo.from_dict(cluster_data)
                    
                    # Apply filters if specified
                    if filter_criteria and not self._matches_filter(cluster, filter_criteria):
                        continue
                    
                    clusters.append(cluster)
                    
                except Exception as e:
                    logger.warning(f"Error loading cluster {storage_key}: {e}")
                    continue
            
            # Update stats
            self._operation_stats['clusters_loaded'] += len(clusters)
            
            load_time = time.time() - start_time
            logger.debug(f"Loaded {len(clusters)} clusters in {load_time:.3f}s")
            
            return clusters
    
    def store_centroids(self, centroids: List[Centroid]) -> Dict[str, str]:
        """
        Store centroids with efficient compression.
        
        Args:
            centroids: List of centroid objects
            
        Returns:
            Dictionary mapping centroid hash to storage key
        """
        with self._lock:
            logger.debug(f"Storing {len(centroids)} centroids")
            start_time = time.time()
            
            centroids_group = self.file["centroids"]
            result = {}
            
            for centroid in centroids:
                centroid_hash = centroid.compute_hash()
                storage_key = f"centroid_{centroid_hash}"
                
                # Skip if already exists
                if storage_key in centroids_group:
                    result[centroid_hash] = storage_key
                    continue
                
                # Store centroid data with compression
                dataset = centroids_group.create_dataset(
                    storage_key,
                    data=centroid.data,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                    chunks=True
                )
                
                # Store metadata as attributes
                dataset.attrs["cluster_id"] = centroid.cluster_id
                dataset.attrs["shape"] = centroid.shape
                dataset.attrs["dtype"] = np.dtype(centroid.dtype).name
                dataset.attrs["hash"] = centroid_hash
                dataset.attrs["nbytes"] = centroid.nbytes
                dataset.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                
                result[centroid_hash] = storage_key
                
                # Update centroid index
                self._update_centroid_index(centroid_hash, centroid.cluster_id)
            
            # Update stats
            self._operation_stats['centroids_stored'] += len(centroids)
            
            # Flush to disk
            self.file.flush()
            
            store_time = time.time() - start_time
            logger.debug(f"Stored {len(centroids)} centroids in {store_time:.3f}s")
            
            return result
    
    def load_centroids(
        self,
        centroid_hashes: Optional[List[str]] = None,
        cluster_ids: Optional[List[str]] = None,
        lazy_loading: bool = False
    ) -> List[Centroid]:
        """
        Load centroids with optional lazy loading.
        
        Args:
            centroid_hashes: Specific centroid hashes to load
            cluster_ids: Load centroids for specific clusters
            lazy_loading: If True, delay data loading until needed
            
        Returns:
            List of centroid objects
        """
        with self._lock:
            logger.debug(f"Loading centroids (lazy={lazy_loading})")
            start_time = time.time()
            
            centroids_group = self.file["centroids"]
            centroids = []
            
            # Determine which centroids to load
            storage_keys = []
            if centroid_hashes:
                storage_keys = [f"centroid_{h}" for h in centroid_hashes]
            elif cluster_ids:
                # Use index to find centroids for clusters
                storage_keys = self._get_centroids_for_clusters(cluster_ids)
            else:
                storage_keys = list(centroids_group.keys())
            
            for storage_key in storage_keys:
                if storage_key not in centroids_group:
                    continue
                
                try:
                    dataset = centroids_group[storage_key]
                    
                    # Load metadata
                    cluster_id = dataset.attrs["cluster_id"]
                    shape = tuple(dataset.attrs["shape"])
                    dtype = np.dtype(dataset.attrs["dtype"])
                    centroid_hash = dataset.attrs["hash"]
                    
                    if lazy_loading:
                        # For lazy loading, create a special placeholder
                        # In a full implementation, you'd create a custom LazyArray class
                        # For now, just load the data normally but mark it as lazy
                        data = np.array(dataset)
                        # In a real implementation, you'd defer loading until needed
                    else:
                        # Load data immediately
                        data = np.array(dataset)
                    
                    centroid = Centroid(
                        data=data,
                        cluster_id=cluster_id,
                        shape=shape,
                        dtype=dtype,
                        hash=centroid_hash
                    )
                    
                    centroids.append(centroid)
                    
                except Exception as e:
                    logger.warning(f"Error loading centroid {storage_key}: {e}")
                    continue
            
            # Update stats
            self._operation_stats['centroids_loaded'] += len(centroids)
            
            load_time = time.time() - start_time
            logger.debug(f"Loaded {len(centroids)} centroids in {load_time:.3f}s")
            
            return centroids
    
    def store_cluster_with_shape_metadata(
        self,
        cluster: ClusterInfo,
        shape_metadata: Dict[str, Any]
    ) -> str:
        """
        Store cluster with additional shape compatibility metadata.
        
        Args:
            cluster: Cluster information
            shape_metadata: Shape and dtype metadata for validation
            
        Returns:
            Storage key for the cluster
        """
        with self._lock:
            clusters_group = self.file["clusters"]
            storage_key = f"cluster_{cluster.cluster_id}"
            
            # Store cluster data
            cluster_data = cluster.to_dict()
            
            if storage_key in clusters_group:
                del clusters_group[storage_key]
            
            dataset = clusters_group.create_dataset(
                storage_key,
                shape=(1,),
                dtype=np.int32
            )
            
            # Store cluster attributes
            for key, value in cluster_data.items():
                if value is None:
                    dataset.attrs[key] = "None"
                elif isinstance(value, (list, tuple)):
                    dataset.attrs[key] = str(value)
                else:
                    dataset.attrs[key] = value
            
            # Store shape metadata for validation
            dataset.attrs["shape_metadata"] = str(shape_metadata)
            dataset.attrs["storage_key"] = storage_key
            dataset.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Update stats
            self._operation_stats['clusters_stored'] += 1
            
            # Flush to disk
            self.file.flush()
            
            logger.debug(f"Stored cluster {cluster.cluster_id} with shape metadata")
            return storage_key
    
    def validate_cluster_compatibility(
        self,
        cluster_id: str,
        weight_shape: Tuple[int, ...],
        weight_dtype: str
    ) -> bool:
        """
        Validate that a weight is compatible with a cluster's shape requirements.
        
        Args:
            cluster_id: ID of the cluster
            weight_shape: Shape of the weight to validate
            weight_dtype: Dtype of the weight to validate
            
        Returns:
            True if compatible, False otherwise
        """
        try:
            clusters = self.load_clusters(cluster_ids=[cluster_id])
            if not clusters:
                logger.warning(f"Cluster {cluster_id} not found for compatibility check")
                return False
            
            cluster = clusters[0]
            
            # Check if cluster has shape metadata stored
            clusters_group = self.file["clusters"]
            storage_key = f"cluster_{cluster_id}"
            
            if storage_key in clusters_group:
                dataset = clusters_group[storage_key]
                if "shape_metadata" in dataset.attrs:
                    import ast
                    try:
                        shape_metadata = ast.literal_eval(dataset.attrs["shape_metadata"])
                        expected_shape = tuple(shape_metadata.get("shape", []))
                        expected_dtype = shape_metadata.get("dtype", "")
                        
                        return (tuple(weight_shape) == expected_shape and 
                                str(weight_dtype) == expected_dtype)
                    except:
                        logger.warning(f"Failed to parse shape metadata for cluster {cluster_id}")
            
            # Fallback: assume compatible if no metadata
            return True
            
        except Exception as e:
            logger.error(f"Error validating cluster compatibility: {e}")
            return False
    
    # Hierarchy operations
    
    def store_hierarchy(self, hierarchy: ClusterHierarchy) -> str:
        """
        Store cluster hierarchy structure.
        
        Args:
            hierarchy: ClusterHierarchy object
            
        Returns:
            Storage key for the hierarchy
        """
        with self._lock:
            logger.debug("Storing cluster hierarchy")
            start_time = time.time()
            
            hierarchies_group = self.file["hierarchies"]
            
            # Generate unique hierarchy ID
            hierarchy_id = f"hierarchy_{uuid.uuid4().hex[:8]}"
            storage_key = f"hierarchy_{hierarchy_id}"
            
            # Serialize hierarchy to dictionary
            hierarchy_data = hierarchy.to_dict()
            
            # Store hierarchy as JSON
            if storage_key in hierarchies_group:
                del hierarchies_group[storage_key]
            
            # Create string dataset for JSON data
            json_data = json.dumps(hierarchy_data)
            dataset = hierarchies_group.create_dataset(
                storage_key,
                data=json_data,
                dtype=h5py.string_dtype(encoding='utf-8')
            )
            
            # Store metadata
            dataset.attrs["hierarchy_id"] = hierarchy_id
            dataset.attrs["total_clusters"] = len(hierarchy)
            dataset.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            dataset.attrs["levels"] = json.dumps([level.value for level in hierarchy.config.levels])
            
            # Update stats
            self._operation_stats['hierarchies_stored'] += 1
            
            # Flush to disk
            self.file.flush()
            
            store_time = time.time() - start_time
            logger.debug(f"Stored hierarchy in {store_time:.3f}s")
            
            return storage_key
    
    def load_hierarchy(self, hierarchy_id: Optional[str] = None) -> Optional[ClusterHierarchy]:
        """
        Load cluster hierarchy from storage.
        
        Args:
            hierarchy_id: Specific hierarchy ID (loads latest if None)
            
        Returns:
            ClusterHierarchy object or None if not found
        """
        with self._lock:
            logger.debug(f"Loading hierarchy: {hierarchy_id}")
            start_time = time.time()
            
            hierarchies_group = self.file["hierarchies"]
            
            # Determine which hierarchy to load
            if hierarchy_id:
                storage_key = f"hierarchy_{hierarchy_id}"
            else:
                # Load most recent hierarchy
                hierarchy_keys = list(hierarchies_group.keys())
                if not hierarchy_keys:
                    return None
                
                # Sort by creation time and get latest
                def get_creation_time(key):
                    try:
                        return hierarchies_group[key].attrs.get("created_at", "")
                    except:
                        return ""
                
                storage_key = max(hierarchy_keys, key=get_creation_time)
            
            if storage_key not in hierarchies_group:
                return None
            
            try:
                dataset = hierarchies_group[storage_key]
                
                # Load JSON data
                json_data = dataset[()]
                if isinstance(json_data, bytes):
                    json_data = json_data.decode('utf-8')
                
                hierarchy_data = json.loads(json_data)
                
                # Reconstruct hierarchy
                hierarchy = ClusterHierarchy.from_dict(hierarchy_data)
                
                # Update stats
                self._operation_stats['hierarchies_loaded'] += 1
                
                load_time = time.time() - start_time
                logger.debug(f"Loaded hierarchy in {load_time:.3f}s")
                
                return hierarchy
                
            except Exception as e:
                logger.error(f"Error loading hierarchy {storage_key}: {e}")
                return None
    
    def update_hierarchy_node(
        self,
        hierarchy_id: str,
        node_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update specific hierarchy node.
        
        Args:
            hierarchy_id: Hierarchy identifier
            node_id: Node/cluster identifier to update
            updates: Dictionary of updates to apply
            
        Returns:
            True if update successful
        """
        with self._lock:
            # Load hierarchy
            hierarchy = self.load_hierarchy(hierarchy_id)
            if not hierarchy:
                return False
            
            # Apply updates (this is a simplified implementation)
            # In practice, you'd need more sophisticated node updating logic
            
            # Re-store updated hierarchy
            self.store_hierarchy(hierarchy)
            
            return True
    
    def validate_hierarchy_integrity(self, hierarchy_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Check stored hierarchy consistency.
        
        Args:
            hierarchy_id: Specific hierarchy to validate
            
        Returns:
            Validation results dictionary
        """
        with self._lock:
            hierarchy = self.load_hierarchy(hierarchy_id)
            if not hierarchy:
                return {"valid": False, "error": "Hierarchy not found"}
            
            # Use hierarchy's built-in validation
            validation_result = hierarchy.validate_consistency()
            
            # Add storage-specific checks
            stored_clusters = self.load_clusters()
            hierarchy_clusters = hierarchy.get_all_clusters()
            
            # Check if all hierarchy clusters exist in storage
            stored_cluster_ids = {c.cluster_id for c in stored_clusters}
            hierarchy_cluster_ids = {c.cluster_id for c in hierarchy_clusters}
            
            missing_in_storage = hierarchy_cluster_ids - stored_cluster_ids
            missing_in_hierarchy = stored_cluster_ids - hierarchy_cluster_ids
            
            validation_result.update({
                "missing_in_storage": list(missing_in_storage),
                "missing_in_hierarchy": list(missing_in_hierarchy),
                "storage_cluster_count": len(stored_clusters),
                "hierarchy_cluster_count": len(hierarchy_clusters)
            })
            
            return validation_result
    
    # Assignment operations
    
    def store_assignments(self, assignments: List[ClusterAssignment]) -> Dict[str, str]:
        """
        Store weight-to-cluster assignments.
        
        Args:
            assignments: List of cluster assignment objects
            
        Returns:
            Dictionary mapping assignment key to storage key
        """
        with self._lock:
            logger.debug(f"Storing {len(assignments)} assignments")
            start_time = time.time()
            
            assignments_group = self.file["assignments"]
            result = {}
            
            for assignment in assignments:
                # Create unique key for assignment
                assignment_key = f"{assignment.weight_hash}_{assignment.cluster_id}"
                storage_key = f"assignment_{assignment_key}"
                
                # Convert to dictionary
                assignment_data = assignment.to_dict()
                
                # Store as dataset with metadata attributes
                if storage_key in assignments_group:
                    del assignments_group[storage_key]
                
                dataset = assignments_group.create_dataset(
                    storage_key,
                    shape=(1,),
                    dtype=np.int32
                )
                
                # Store assignment data as attributes
                for key, value in assignment_data.items():
                    # Handle None values and ensure HDF5 compatibility
                    if value is None:
                        dataset.attrs[key] = "None"
                    elif isinstance(value, (list, tuple)):
                        # Convert lists/tuples to strings for HDF5
                        dataset.attrs[key] = str(value)
                    else:
                        dataset.attrs[key] = value
                
                dataset.attrs["assignment_key"] = assignment_key
                dataset.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                
                result[assignment_key] = storage_key
            
            # Update stats
            self._operation_stats['assignments_stored'] += len(assignments)
            
            # Flush to disk
            self.file.flush()
            
            store_time = time.time() - start_time
            logger.debug(f"Stored {len(assignments)} assignments in {store_time:.3f}s")
            
            return result
    
    def load_assignments(
        self,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[ClusterAssignment]:
        """
        Load assignments with optional filtering.
        
        Args:
            filter_criteria: Filter criteria for selective loading
                - cluster_ids: List of cluster IDs
                - weight_hashes: List of weight hashes
                - min_similarity: Minimum similarity score
                
        Returns:
            List of cluster assignment objects
        """
        with self._lock:
            logger.debug("Loading cluster assignments")
            start_time = time.time()
            
            assignments_group = self.file["assignments"]
            assignments = []
            
            for storage_key in assignments_group.keys():
                try:
                    dataset = assignments_group[storage_key]
                    
                    # Reconstruct assignment from attributes
                    assignment_data = {}
                    for attr_name in dataset.attrs:
                        value = dataset.attrs[attr_name]
                        # Handle special values that were converted for HDF5
                        if value == "None":
                            assignment_data[attr_name] = None
                        else:
                            assignment_data[attr_name] = value
                    
                    assignment = ClusterAssignment.from_dict(assignment_data)
                    
                    # Apply filters
                    if filter_criteria:
                        if "cluster_ids" in filter_criteria:
                            if assignment.cluster_id not in filter_criteria["cluster_ids"]:
                                continue
                        
                        if "weight_hashes" in filter_criteria:
                            if assignment.weight_hash not in filter_criteria["weight_hashes"]:
                                continue
                        
                        if "min_similarity" in filter_criteria:
                            if assignment.similarity_score < filter_criteria["min_similarity"]:
                                continue
                    
                    assignments.append(assignment)
                    
                except Exception as e:
                    logger.warning(f"Error loading assignment {storage_key}: {e}")
                    continue
            
            # Update stats
            self._operation_stats['assignments_loaded'] += len(assignments)
            
            load_time = time.time() - start_time
            logger.debug(f"Loaded {len(assignments)} assignments in {load_time:.3f}s")
            
            return assignments
    
    # Delta operations
    
    def store_centroid_deltas(self, deltas: List[Delta]) -> Dict[str, str]:
        """
        Store deltas for centroid-based encoding.
        
        Args:
            deltas: List of delta objects
            
        Returns:
            Dictionary mapping delta hash to storage key
        """
        with self._lock:
            logger.debug(f"Storing {len(deltas)} centroid deltas")
            
            centroid_deltas_group = self.file["centroid_deltas"]
            result = {}
            
            for delta in deltas:
                # Use the existing store's delta storage functionality
                delta_hash = delta.compute_hash()
                stored_hash = self.store.store_delta(delta, delta_hash)
                
                # Create reference in centroid_deltas group
                storage_key = f"centroid_delta_{stored_hash}"
                
                if storage_key not in centroid_deltas_group:
                    # Create reference dataset
                    ref_dataset = centroid_deltas_group.create_dataset(
                        storage_key,
                        shape=(1,),
                        dtype=np.int32
                    )
                    ref_dataset.attrs["delta_hash"] = stored_hash
                    ref_dataset.attrs["reference_hash"] = delta.reference_hash
                    ref_dataset.attrs["delta_type"] = delta.delta_type.value
                    ref_dataset.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                
                result[stored_hash] = storage_key
            
            self.file.flush()
            return result
    
    def load_centroid_deltas(self, cluster_ids: List[str]) -> List[Delta]:
        """
        Load deltas for specific clusters.
        
        Args:
            cluster_ids: List of cluster identifiers
            
        Returns:
            List of delta objects
        """
        with self._lock:
            logger.debug(f"Loading deltas for {len(cluster_ids)} clusters")
            
            # Get centroid hashes for clusters
            centroid_hashes = self._get_centroid_hashes_for_clusters(cluster_ids)
            
            deltas = []
            centroid_deltas_group = self.file["centroid_deltas"]
            
            for storage_key in centroid_deltas_group.keys():
                try:
                    ref_dataset = centroid_deltas_group[storage_key]
                    delta_hash = ref_dataset.attrs["delta_hash"]
                    reference_hash = ref_dataset.attrs["reference_hash"]
                    
                    # Check if this delta is for one of our clusters
                    if reference_hash in centroid_hashes:
                        # Load the actual delta from the main deltas group
                        delta = self.store.load_delta(delta_hash)
                        if delta:
                            deltas.append(delta)
                            
                except Exception as e:
                    logger.warning(f"Error loading centroid delta {storage_key}: {e}")
                    continue
            
            return deltas
    
    # Query operations
    
    def get_clusters_by_level(self, level: ClusterLevel) -> List[ClusterInfo]:
        """
        Fast retrieval by hierarchy level.
        
        Args:
            level: Target hierarchy level
            
        Returns:
            List of clusters at the specified level
        """
        return self.load_clusters(filter_criteria={"level": level})
    
    def get_clusters_by_strategy(self, strategy: ClusteringStrategy) -> List[ClusterInfo]:
        """
        Filter by clustering strategy.
        
        Args:
            strategy: Clustering strategy
            
        Returns:
            List of clusters using the specified strategy
        """
        return self.load_clusters(filter_criteria={"strategy": strategy})
    
    def get_cluster_metadata(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata without loading full cluster.
        
        Args:
            cluster_id: Cluster identifier
            
        Returns:
            Cluster metadata dictionary or None
        """
        with self._lock:
            clusters_group = self.file["clusters"]
            storage_key = f"cluster_{cluster_id}"
            
            if storage_key not in clusters_group:
                return None
            
            try:
                dataset = clusters_group[storage_key]
                metadata = {}
                
                for attr_name in dataset.attrs:
                    value = dataset.attrs[attr_name]
                    if attr_name in ["child_cluster_ids"]:
                        try:
                            metadata[attr_name] = json.loads(value) if value else None
                        except (json.JSONDecodeError, TypeError):
                            metadata[attr_name] = value
                    else:
                        metadata[attr_name] = value
                
                return metadata
                
            except Exception as e:
                logger.warning(f"Error loading metadata for {cluster_id}: {e}")
                return None
    
    def delete_cluster(self, cluster_id: str) -> bool:
        """
        Delete a cluster from storage.
        
        Args:
            cluster_id: Cluster identifier to delete
            
        Returns:
            True if cluster was deleted, False if not found
        """
        with self._lock:
            clusters_group = self.file["clusters"]
            storage_key = f"cluster_{cluster_id}"
            
            if storage_key not in clusters_group:
                return False
            
            try:
                # Get metadata before deletion
                dataset = clusters_group[storage_key]
                centroid_hash = None
                for attr_name in dataset.attrs:
                    if attr_name == "centroid_hash":
                        centroid_hash = dataset.attrs[attr_name]
                        break
                
                # Delete the cluster
                del clusters_group[storage_key]
                
                # Delete associated assignments
                self.delete_assignments_for_cluster(cluster_id)
                
                # Also delete associated centroid if it exists
                if centroid_hash:
                    self.delete_centroid(centroid_hash)
                
                self.file.flush()
                logger.debug(f"Deleted cluster {cluster_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error deleting cluster {cluster_id}: {e}")
                return False
    
    def delete_centroid(self, centroid_hash: str) -> bool:
        """
        Delete a centroid from storage.
        
        Args:
            centroid_hash: Centroid hash to delete
            
        Returns:
            True if centroid was deleted, False if not found
        """
        with self._lock:
            centroids_group = self.file["centroids"]
            storage_key = f"centroid_{centroid_hash}"
            
            if storage_key not in centroids_group:
                return False
            
            try:
                del centroids_group[storage_key]
                self.file.flush()
                logger.debug(f"Deleted centroid {centroid_hash}")
                return True
                
            except Exception as e:
                logger.error(f"Error deleting centroid {centroid_hash}: {e}")
                return False
    
    def delete_assignment(self, weight_hash: str, cluster_id: str) -> bool:
        """
        Delete a specific assignment from storage.
        
        Args:
            weight_hash: Weight hash of the assignment
            cluster_id: Cluster ID of the assignment
            
        Returns:
            True if assignment was deleted, False if not found
        """
        with self._lock:
            assignments_group = self.file["assignments"]
            assignment_key = f"{weight_hash}_{cluster_id}"
            storage_key = f"assignment_{assignment_key}"
            
            if storage_key not in assignments_group:
                return False
            
            try:
                del assignments_group[storage_key]
                self.file.flush()
                logger.debug(f"Deleted assignment {assignment_key}")
                return True
                
            except Exception as e:
                logger.error(f"Error deleting assignment {assignment_key}: {e}")
                return False
    
    def delete_assignments_for_cluster(self, cluster_id: str) -> int:
        """
        Delete all assignments for a specific cluster.
        
        Args:
            cluster_id: Cluster ID whose assignments to delete
            
        Returns:
            Number of assignments deleted
        """
        with self._lock:
            assignments_group = self.file["assignments"]
            deleted_count = 0
            
            # Find all assignments for this cluster
            assignments_to_delete = []
            for storage_key in assignments_group.keys():
                try:
                    dataset = assignments_group[storage_key]
                    if dataset.attrs.get("cluster_id") == cluster_id:
                        assignments_to_delete.append(storage_key)
                except Exception:
                    continue
            
            # Delete them
            for storage_key in assignments_to_delete:
                try:
                    del assignments_group[storage_key]
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Error deleting assignment {storage_key}: {e}")
            
            if deleted_count > 0:
                self.file.flush()
                logger.debug(f"Deleted {deleted_count} assignments for cluster {cluster_id}")
            
            return deleted_count
    
    def get_cluster_members(self, cluster_id: str) -> List[str]:
        """
        Get all weight hashes that belong to a specific cluster.
        
        Args:
            cluster_id: Cluster identifier
            
        Returns:
            List of weight hashes assigned to this cluster
        """
        with self._lock:
            assignments = self.load_assignments(
                filter_criteria={'cluster_ids': [cluster_id]}
            )
            return [a.weight_hash for a in assignments]
    
    def batch_load_clusters(self, cluster_ids: List[str]) -> List[ClusterInfo]:
        """
        Efficient batch loading of clusters.
        
        Args:
            cluster_ids: List of cluster identifiers
            
        Returns:
            List of loaded clusters
        """
        # Use thread pool for parallel loading
        max_workers = min(len(cluster_ids), 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit loading tasks
            future_to_id = {}
            for cluster_id in cluster_ids:
                future = executor.submit(self.load_clusters, [cluster_id])
                future_to_id[future] = cluster_id
            
            # Collect results
            clusters = []
            for future in as_completed(future_to_id):
                try:
                    result = future.result()
                    if result:
                        clusters.extend(result)
                except Exception as e:
                    cluster_id = future_to_id[future]
                    logger.warning(f"Error loading cluster {cluster_id}: {e}")
        
        return clusters
    
    # Storage optimization
    
    def compress_storage(self) -> Dict[str, Any]:
        """
        Optimize HDF5 storage and remove unused data.
        
        Returns:
            Compression results dictionary
        """
        with self._lock:
            logger.info("Compressing cluster storage")
            start_time = time.time()
            
            # Get initial storage info
            initial_info = self.get_storage_info()
            
            # Repack the HDF5 file to remove unused space
            # Note: h5repack would be more efficient, but we'll do basic cleanup
            
            # Remove orphaned data
            orphaned_removed = self._remove_orphaned_data()
            
            # Compact storage by removing deleted datasets
            self.file.flush()
            
            # Get final storage info
            final_info = self.get_storage_info()
            
            compression_time = time.time() - start_time
            
            results = {
                "compression_time": compression_time,
                "initial_size": initial_info.get("file_size", 0),
                "final_size": final_info.get("file_size", 0),
                "space_saved": initial_info.get("file_size", 0) - final_info.get("file_size", 0),
                "orphaned_removed": orphaned_removed
            }
            
            logger.info(f"Storage compression completed in {compression_time:.3f}s")
            return results
    
    def garbage_collect(self) -> Dict[str, Any]:
        """
        Clean up orphaned clusters and deltas.
        
        Returns:
            Garbage collection results
        """
        with self._lock:
            logger.info("Running cluster storage garbage collection")
            start_time = time.time()
            
            # Find orphaned centroids (not referenced by any clusters)
            cluster_centroid_hashes = set()
            clusters = self.load_clusters()
            for cluster in clusters:
                if cluster.centroid_hash:
                    cluster_centroid_hashes.add(cluster.centroid_hash)
            
            # Remove orphaned centroids
            centroids_group = self.file["centroids"]
            orphaned_centroids = []
            
            for storage_key in list(centroids_group.keys()):
                dataset = centroids_group[storage_key]
                centroid_hash = dataset.attrs.get("hash", "")
                
                if centroid_hash not in cluster_centroid_hashes:
                    orphaned_centroids.append(storage_key)
                    del centroids_group[storage_key]
            
            # Find orphaned assignments
            valid_cluster_ids = {c.cluster_id for c in clusters}
            assignments_group = self.file["assignments"]
            orphaned_assignments = []
            
            for storage_key in list(assignments_group.keys()):
                dataset = assignments_group[storage_key]
                cluster_id = dataset.attrs.get("cluster_id", "")
                
                if cluster_id not in valid_cluster_ids:
                    orphaned_assignments.append(storage_key)
                    del assignments_group[storage_key]
            
            self.file.flush()
            
            gc_time = time.time() - start_time
            
            results = {
                "gc_time": gc_time,
                "orphaned_centroids_removed": len(orphaned_centroids),
                "orphaned_assignments_removed": len(orphaned_assignments),
                "total_clusters": len(clusters)
            }
            
            logger.info(f"Garbage collection completed in {gc_time:.3f}s")
            return results
    
    def estimate_storage_size(self) -> Dict[str, int]:
        """
        Calculate storage requirements.
        
        Returns:
            Storage size estimates in bytes
        """
        with self._lock:
            storage_info = self.get_storage_info()
            
            estimates = {
                "total_file_size": storage_info.get("file_size", 0),
                "clusters_size": self._estimate_group_size("clusters"),
                "centroids_size": self._estimate_group_size("centroids"),
                "assignments_size": self._estimate_group_size("assignments"),
                "hierarchies_size": self._estimate_group_size("hierarchies"),
                "centroid_deltas_size": self._estimate_group_size("centroid_deltas"),
                "indexes_size": self._estimate_group_size("indexes")
            }
            
            return estimates
    
    def optimize_layout(self) -> Dict[str, Any]:
        """
        Reorganize for better performance.
        
        Returns:
            Optimization results
        """
        with self._lock:
            logger.info("Optimizing storage layout")
            start_time = time.time()
            
            # Rebuild indexes for better query performance
            self._rebuild_indexes()
            
            # Compact frequently accessed data
            self.file.flush()
            
            optimization_time = time.time() - start_time
            
            results = {
                "optimization_time": optimization_time,
                "indexes_rebuilt": True
            }
            
            logger.info(f"Layout optimization completed in {optimization_time:.3f}s")
            return results
    
    # Backup and migration
    
    def export_clusters(self, filepath: str) -> Dict[str, Any]:
        """
        Export clusters to external file.
        
        Args:
            filepath: Target export file path
            
        Returns:
            Export results dictionary
        """
        logger.info(f"Exporting clusters to {filepath}")
        start_time = time.time()
        
        # Load all data
        clusters = self.load_clusters()
        centroids = self.load_centroids()
        assignments = self.load_assignments()
        hierarchy = self.load_hierarchy()
        
        # Prepare export data with proper serialization
        def serialize_centroid(centroid):
            """Serialize centroid data for JSON export."""
            centroid_dict = centroid.to_dict()
            # Convert bytes to base64 string
            import base64
            centroid_dict["data_bytes"] = base64.b64encode(centroid_dict["data_bytes"]).decode('utf-8')
            return centroid_dict
        
        export_data = {
            "version": "1.0",
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "clusters": [c.to_dict() for c in clusters],
            "centroids": [serialize_centroid(c) for c in centroids],
            "assignments": [a.to_dict() for a in assignments],
            "hierarchy": hierarchy.to_dict() if hierarchy else None,
            "storage_stats": self._operation_stats.copy()
        }
        
        # Write to file
        export_path = Path(filepath)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, cls=NumpyEncoder)
        
        export_time = time.time() - start_time
        
        results = {
            "export_time": export_time,
            "filepath": str(export_path),
            "clusters_exported": len(clusters),
            "centroids_exported": len(centroids),
            "assignments_exported": len(assignments),
            "file_size": export_path.stat().st_size
        }
        
        logger.info(f"Export completed in {export_time:.3f}s")
        return results
    
    def import_clusters(self, filepath: str) -> Dict[str, Any]:
        """
        Import clusters from external file.
        
        Args:
            filepath: Source import file path
            
        Returns:
            Import results dictionary
        """
        logger.info(f"Importing clusters from {filepath}")
        start_time = time.time()
        
        with open(filepath, 'r') as f:
            import_data = json.load(f)
        
        # Reconstruct objects
        clusters = [ClusterInfo.from_dict(d) for d in import_data.get("clusters", [])]
        
        centroids = []
        for centroid_data in import_data.get("centroids", []):
            # Reconstruct centroid data
            import base64
            data_bytes_encoded = centroid_data["data_bytes"]
            # Handle both base64 encoded strings and raw bytes
            if isinstance(data_bytes_encoded, str):
                data_bytes = base64.b64decode(data_bytes_encoded)
            else:
                data_bytes = data_bytes_encoded
                
            data_dtype = np.dtype(centroid_data["data_dtype"])
            data_shape = tuple(centroid_data["data_shape"])
            
            data_array = np.frombuffer(data_bytes, dtype=data_dtype).reshape(data_shape)
            
            centroid = Centroid(
                data=data_array,
                cluster_id=centroid_data["cluster_id"],
                shape=tuple(centroid_data["shape"]),
                dtype=np.dtype(centroid_data["dtype"]),
                hash=centroid_data.get("hash")
            )
            centroids.append(centroid)
        
        assignments = [ClusterAssignment.from_dict(d) for d in import_data.get("assignments", [])]
        
        # Import hierarchy if present
        hierarchy = None
        if import_data.get("hierarchy"):
            hierarchy = ClusterHierarchy.from_dict(import_data["hierarchy"])
        
        # Store imported data
        cluster_results = self.store_clusters(clusters, hierarchy)
        centroid_results = self.store_centroids(centroids)
        assignment_results = self.store_assignments(assignments)
        
        import_time = time.time() - start_time
        
        results = {
            "import_time": import_time,
            "clusters_imported": len(clusters),
            "centroids_imported": len(centroids),
            "assignments_imported": len(assignments),
            "hierarchy_imported": hierarchy is not None
        }
        
        logger.info(f"Import completed in {import_time:.3f}s")
        return results
    
    def create_backup(self, backup_dir: Optional[str] = None) -> str:
        """
        Create timestamped backup.
        
        Args:
            backup_dir: Backup directory (creates default if None)
            
        Returns:
            Path to backup file
        """
        if backup_dir is None:
            backup_dir = self.storage_path.parent / "backups"
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_file = backup_path / f"cluster_storage_backup_{timestamp}.h5"
        
        logger.info(f"Creating backup at {backup_file}")
        
        # Close current file temporarily
        if self._owns_store:
            self.file.close()
        
        try:
            # Copy the HDF5 file
            shutil.copy2(self.storage_path, backup_file)
            
            logger.info(f"Backup created successfully at {backup_file}")
            return str(backup_file)
            
        finally:
            # Reopen file
            if self._owns_store:
                self.file = h5py.File(self.storage_path, self.mode)
                # Reinitialize store with new file handle
                self.store = HDF5Store(str(self.storage_path), mode=self.mode)
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """
        Restore from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if restore successful
        """
        backup_file = Path(backup_path)
        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        logger.info(f"Restoring from backup: {backup_path}")
        
        # Close current file
        if self._owns_store:
            self.file.close()
        
        try:
            # Copy backup over current file
            shutil.copy2(backup_file, self.storage_path)
            
            # Reopen file
            if self._owns_store:
                self.file = h5py.File(self.storage_path, self.mode)
                # Reinitialize store with new file handle
                self.store = HDF5Store(str(self.storage_path), mode=self.mode)
                
                # Re-initialize storage groups
                self._init_storage_groups()
            
            logger.info("Restore completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during restore: {e}")
            return False
    
    # Utility methods
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get comprehensive storage information."""
        with self._lock:
            # Get basic file info
            file_size = self.storage_path.stat().st_size if self.storage_path.exists() else 0
            
            # Count items in each group
            group_counts = {}
            group_sizes = {}
            
            for group_name in ["clusters", "centroids", "assignments", "hierarchies", "centroid_deltas"]:
                if group_name in self.file:
                    group = self.file[group_name]
                    group_counts[group_name] = len(group)
                    group_sizes[group_name] = self._estimate_group_size(group_name)
                else:
                    group_counts[group_name] = 0
                    group_sizes[group_name] = 0
            
            return {
                "file_size": file_size,
                "storage_path": str(self.storage_path),
                "compression": self.compression,
                "group_counts": group_counts,
                "group_sizes": group_sizes,
                "operation_stats": self._operation_stats.copy(),
                "version": self.file.attrs.get("cluster_storage_version", "unknown"),
                "created_at": self.file.attrs.get("created_at", "unknown")
            }
    
    def close(self):
        """Close storage and clean up resources."""
        if self._owns_file and hasattr(self, 'file') and self.file:
            try:
                self.file.close()
            except Exception:
                pass  # Already closed
        if self._owns_store and hasattr(self, 'store') and self.store:
            self.store.close()
    
    # Private helper methods
    
    def _matches_filter(self, cluster: ClusterInfo, filter_criteria: Dict[str, Any]) -> bool:
        """Check if cluster matches filter criteria."""
        for key, value in filter_criteria.items():
            if key == "level":
                if cluster.level != value:
                    return False
            elif key == "strategy":
                if cluster.strategy != value:
                    return False
            elif key == "min_member_count":
                if cluster.member_count < value:
                    return False
            elif key == "max_member_count":
                if cluster.member_count > value:
                    return False
        
        return True
    
    def _update_cluster_indexes(self, cluster: ClusterInfo):
        """Update index structures for fast queries."""
        # This is a simplified implementation
        # In practice, you'd maintain more sophisticated indexes
        pass
    
    def _update_centroid_index(self, centroid_hash: str, cluster_id: str):
        """Update centroid index mapping."""
        # Simplified implementation
        pass
    
    def _get_centroids_for_clusters(self, cluster_ids: List[str]) -> List[str]:
        """Get storage keys for centroids belonging to specific clusters."""
        storage_keys = []
        
        # Load all centroids and check their cluster_id
        centroids_group = self.file["centroids"]
        for storage_key in centroids_group.keys():
            try:
                dataset = centroids_group[storage_key]
                centroid_cluster_id = dataset.attrs.get("cluster_id", "")
                if centroid_cluster_id in cluster_ids:
                    storage_keys.append(storage_key)
            except Exception:
                continue
        
        return storage_keys
    
    def _get_centroid_hashes_for_clusters(self, cluster_ids: List[str]) -> Set[str]:
        """Get centroid hashes for specific clusters."""
        centroid_hashes = set()
        
        for cluster_id in cluster_ids:
            metadata = self.get_cluster_metadata(cluster_id)
            if metadata and metadata.get("centroid_hash"):
                centroid_hashes.add(metadata["centroid_hash"])
        
        return centroid_hashes
    
    def _remove_orphaned_data(self) -> int:
        """Remove orphaned data and return count of items removed."""
        # This would implement comprehensive orphan detection and removal
        # For now, return 0 as placeholder
        return 0
    
    def _estimate_group_size(self, group_name: str) -> int:
        """Estimate size of a storage group in bytes."""
        if group_name not in self.file:
            return 0
        
        group = self.file[group_name]
        total_size = 0
        
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                nonlocal total_size
                total_size += obj.nbytes
        
        group.visititems(visitor)
        return total_size
    
    def _rebuild_indexes(self):
        """Rebuild index structures for better performance."""
        # Clear existing indexes
        indexes_group = self.file["indexes"]
        
        for index_name in list(indexes_group.keys()):
            del indexes_group[index_name]
        
        # Rebuild indexes
        self._init_indexes()
        
        # Populate indexes with current data
        clusters = self.load_clusters()
        for cluster in clusters:
            self._update_cluster_indexes(cluster)
    
    def __enter__(self):
        """Context manager entry."""
        # Reopen file if it was closed
        if hasattr(self, 'file') and (not self.file or not self.file.id.valid):
            self.file = h5py.File(str(self.storage_path), self.mode)
            # Ensure groups exist
            self._init_storage_groups()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __repr__(self) -> str:
        """String representation."""
        try:
            # Try to get info if file is valid
            if hasattr(self, 'file') and self.file and hasattr(self.file, 'id') and self.file.id.valid:
                info = self.get_storage_info()
                return (
                    f"ClusterStorage(path='{self.storage_path}', "
                    f"clusters={info['group_counts']['clusters']}, "
                    f"centroids={info['group_counts']['centroids']}, "
                    f"assignments={info['group_counts']['assignments']})"
                )
            else:
                return f"ClusterStorage(path='{self.storage_path}', file_closed=True)"
        except Exception:
            # Fallback if any error occurs
            return f"ClusterStorage(path='{self.storage_path}')"