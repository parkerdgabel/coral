"""Migration utilities for HDF5 storage format updates."""

import logging
from typing import Optional
import h5py
import numpy as np

logger = logging.getLogger(__name__)


class StorageMigrator:
    """Handles migration of HDF5 storage files between versions."""
    
    # Version history
    VERSION_1_0 = "1.0"  # Original format
    VERSION_2_0 = "2.0"  # Added PQ codebook support
    VERSION_3_0 = "3.0"  # Added computation graph support
    
    CURRENT_VERSION = VERSION_3_0
    
    @staticmethod
    def get_file_version(filepath: str) -> Optional[str]:
        """Get the version of an HDF5 file.
        
        Args:
            filepath: Path to HDF5 file
            
        Returns:
            Version string or None if file doesn't exist
        """
        try:
            with h5py.File(filepath, 'r') as f:
                return f.attrs.get("version", StorageMigrator.VERSION_1_0)
        except (OSError, IOError):
            return None
    
    @staticmethod
    def needs_migration(filepath: str) -> bool:
        """Check if a file needs migration to current version.
        
        Args:
            filepath: Path to HDF5 file
            
        Returns:
            True if migration is needed
        """
        version = StorageMigrator.get_file_version(filepath)
        if version is None:
            return False
        return version < StorageMigrator.CURRENT_VERSION
    
    @staticmethod
    def migrate_file(filepath: str, backup: bool = True) -> bool:
        """Migrate an HDF5 file to the current version.
        
        Args:
            filepath: Path to HDF5 file
            backup: Whether to create a backup before migration
            
        Returns:
            True if migration successful
        """
        current_version = StorageMigrator.get_file_version(filepath)
        if current_version is None:
            logger.error(f"Cannot read version from {filepath}")
            return False
        
        if current_version >= StorageMigrator.CURRENT_VERSION:
            logger.info(f"File {filepath} is already at current version {current_version}")
            return True
        
        if backup:
            import shutil
            backup_path = f"{filepath}.v{current_version}.backup"
            shutil.copy2(filepath, backup_path)
            logger.info(f"Created backup at {backup_path}")
        
        try:
            with h5py.File(filepath, 'r+') as f:
                # Migrate from 1.0 to 2.0
                if current_version < StorageMigrator.VERSION_2_0:
                    logger.info(f"Migrating from {current_version} to {StorageMigrator.VERSION_2_0}")
                    if "pq_codebooks" not in f:
                        f.create_group("pq_codebooks")
                    current_version = StorageMigrator.VERSION_2_0
                
                # Migrate from 2.0 to 3.0
                if current_version < StorageMigrator.VERSION_3_0:
                    logger.info(f"Migrating from {current_version} to {StorageMigrator.VERSION_3_0}")
                    if "computation_graphs" not in f:
                        f.create_group("computation_graphs")
                    current_version = StorageMigrator.VERSION_3_0
                
                # Update version and timestamp
                f.attrs["version"] = StorageMigrator.CURRENT_VERSION
                f.attrs["migrated_at"] = str(np.datetime64('now'))
                
                logger.info(f"Successfully migrated {filepath} to version {StorageMigrator.CURRENT_VERSION}")
                return True
                
        except Exception as e:
            logger.error(f"Error during migration of {filepath}: {e}")
            return False
    
    @staticmethod
    def validate_file_structure(filepath: str) -> dict:
        """Validate the structure of an HDF5 file.
        
        Args:
            filepath: Path to HDF5 file
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "version": None,
            "errors": [],
            "warnings": [],
            "structure": {}
        }
        
        try:
            with h5py.File(filepath, 'r') as f:
                results["version"] = f.attrs.get("version", "1.0")
                
                # Check required groups based on version
                if results["version"] >= "1.0":
                    required_groups = ["weights", "metadata", "deltas"]
                    for group in required_groups:
                        if group not in f:
                            results["errors"].append(f"Missing required group: {group}")
                            results["valid"] = False
                        else:
                            results["structure"][group] = len(f[group])
                
                if results["version"] >= "2.0":
                    required_groups = ["clustered_weights", "centroids", "pq_codebooks"]
                    for group in required_groups:
                        if group not in f:
                            results["warnings"].append(f"Missing v2.0 group: {group}")
                        else:
                            results["structure"][group] = len(f[group])
                
                if results["version"] >= "3.0":
                    if "computation_graphs" not in f:
                        results["warnings"].append("Missing v3.0 group: computation_graphs")
                    else:
                        results["structure"]["computation_graphs"] = len(f["computation_graphs"])
                
                # Check for unexpected groups
                expected_groups = {
                    "weights", "metadata", "deltas", "clustered_weights", 
                    "centroids", "pq_codebooks", "computation_graphs"
                }
                for group in f:
                    if group not in expected_groups:
                        results["warnings"].append(f"Unexpected group: {group}")
                
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Cannot read file: {e}")
        
        return results