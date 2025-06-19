"""Converter functions between Coral and Safetensors formats."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
from tqdm import tqdm

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.safetensors.metadata import (
    SafetensorsMetadata,
    create_coral_metadata,
    merge_metadata,
)
from coral.safetensors.reader import SafetensorsReader
from coral.safetensors.types import MetadataDict, TensorDict
from coral.safetensors.writer import SafetensorsWriter
from coral.storage.hdf5_store import HDF5Store
from coral.version_control.repository import Repository


def convert_coral_to_safetensors(
    source: Union[str, Path, Repository, HDF5Store],
    output_path: Union[str, Path],
    weight_names: Optional[List[str]] = None,
    include_metadata: bool = True,
    custom_metadata: Optional[MetadataDict] = None,
) -> None:
    """Convert Coral weights to Safetensors format.

    Args:
        source: Coral repository path, Repository instance, or HDF5Store
        output_path: Path for output Safetensors file
        weight_names: Specific weights to convert (None for all)
        include_metadata: Whether to include Coral metadata
        custom_metadata: Additional metadata to include

    Raises:
        ValueError: If source is invalid or weights not found
        SafetensorsError: If conversion fails
    """
    # Handle different source types
    if isinstance(source, (str, Path)):
        # Open repository from path
        repo = Repository(Path(source))
        # Repository already has a store initialized
    elif isinstance(source, Repository):
        repo = source
    elif isinstance(source, HDF5Store):
        repo = None
    else:
        raise ValueError(f"Invalid source type: {type(source)}")
    
    # Get weights to convert
    if weight_names is None:
        if repo is not None:
            # Get all weights from current commit
            weights_dict = repo.get_all_weights()
            weight_names = list(weights_dict.keys())
        else:
            # For HDF5Store, list all weight hashes
            weight_hashes = source.list_weights()
            # We'll need to process by hash instead of name
            weight_names = weight_hashes
    
    # Create metadata
    metadata: MetadataDict = {}
    
    if include_metadata and repo is not None:
        # Add Coral-specific metadata
        current_branch = repo.branch_manager.get_current_branch()
        current_commit_hash = repo.branch_manager.get_branch_commit(current_branch)
        
        coral_metadata = create_coral_metadata(
            branch=current_branch,
            commit_hash=current_commit_hash,
            similarity_threshold=repo.deduplicator.similarity_threshold,
            delta_encoding_enabled=True,  # Coral v1.0+ always uses delta encoding
        )
        metadata.update(coral_metadata)
    
    # Add custom metadata if provided
    if custom_metadata:
        metadata = merge_metadata(metadata, custom_metadata)
    
    # Create Safetensors writer
    writer = SafetensorsWriter(output_path, metadata=metadata)
    
    # Convert each weight
    if repo is not None:
        # Use repository to get weights (handles delta reconstruction)
        for name in weight_names:
            weight_tensor = repo.get_weight(name)
            if weight_tensor is None:
                raise ValueError(f"Weight '{name}' not found in repository")
            
            # Add to writer
            writer.add_tensor(name, weight_tensor.data)
    else:
        # Direct HDF5Store access - weights are stored by hash
        for hash_key in weight_names:
            weight_tensor = source.load(hash_key)
            if weight_tensor is None:
                raise ValueError(f"Weight with hash '{hash_key}' not found in store")
            
            # Use the weight's metadata name or hash as tensor name
            tensor_name = weight_tensor.metadata.name if weight_tensor.metadata else hash_key
            writer.add_tensor(tensor_name, weight_tensor.data)
    
    # Write the file
    writer.write()


def convert_safetensors_to_coral(
    source_path: Union[str, Path],
    target: Union[str, Path, Repository, HDF5Store],
    preserve_metadata: bool = True,
    weight_names: Optional[List[str]] = None,
    exclude_weights: Optional[Set[str]] = None,
) -> Dict[str, str]:
    """Convert Safetensors file to Coral format.

    Args:
        source_path: Path to Safetensors file
        target: Coral repository path, Repository instance, or HDF5Store
        preserve_metadata: Whether to preserve Safetensors metadata
        weight_names: Specific weights to convert (None for all)
        exclude_weights: Weights to exclude from conversion

    Returns:
        Dictionary mapping original names to Coral weight IDs

    Raises:
        ValueError: If target is invalid
        SafetensorsError: If conversion fails
    """
    # Handle different target types
    if isinstance(target, (str, Path)):
        # Create or open repository
        repo_path = Path(target)
        if repo_path.exists() and (repo_path / ".coral").exists():
            repo = Repository(repo_path)
        else:
            # Initialize new repository
            repo = Repository(repo_path, init=True)
        # Get the store from repository
        store_path = repo.coral_dir / "objects" / "weights.h5"
        store = HDF5Store(store_path)
    elif isinstance(target, Repository):
        repo = target
        # Get the store from repository
        store_path = repo.coral_dir / "objects" / "weights.h5"
        store = HDF5Store(store_path)
    elif isinstance(target, HDF5Store):
        repo = None
        store = target
    else:
        raise ValueError(f"Invalid target type: {type(target)}")
    
    # Open Safetensors file
    reader = SafetensorsReader(source_path)
    
    # Get weights to convert
    available_weights = reader.get_tensor_names()
    if weight_names is None:
        weight_names = available_weights
    else:
        # Validate requested weights exist
        missing = set(weight_names) - set(available_weights)
        if missing:
            raise ValueError(f"Weights not found in file: {missing}")
    
    # Apply exclusions
    if exclude_weights:
        weight_names = [n for n in weight_names if n not in exclude_weights]
    
    # Convert metadata if requested
    extra_metadata: Optional[Dict[str, Any]] = None
    if preserve_metadata and reader.metadata:
        # Extract relevant metadata
        extra_metadata = {}
        
        # Map common fields
        if "model_name" in reader.metadata:
            extra_metadata["model_name"] = reader.metadata["model_name"]
        if "model_version" in reader.metadata:
            extra_metadata["model_version"] = reader.metadata["model_version"]
        if "description" in reader.metadata:
            extra_metadata["description"] = reader.metadata["description"]
        
        # Preserve other metadata with prefix
        for key, value in reader.metadata.items():
            if key not in {"model_name", "model_version", "description"}:
                extra_metadata[f"safetensors.{key}"] = value
    
    # Convert each weight
    weight_mapping: Dict[str, str] = {}
    weights_to_stage: Dict[str, WeightTensor] = {}
    
    for name in weight_names:
        # Read tensor data
        tensor_data = reader.read_tensor(name)
        
        # Create WeightMetadata
        metadata = WeightMetadata(
            name=name,
            shape=tuple(tensor_data.shape),
            dtype=tensor_data.dtype,
            layer_type=name.split('.')[-1] if '.' in name else None,
            model_name=extra_metadata.get("model_name") if extra_metadata else None,
            compression_info={"source_format": "safetensors"},
        )
        
        # Create WeightTensor
        weight_tensor = WeightTensor(
            data=tensor_data,
            metadata=metadata,
        )
        
        if repo is not None:
            # Store for batch staging later
            weights_to_stage[name] = weight_tensor
        else:
            # Store directly in HDF5Store
            weight_hash = store.store(weight_tensor)
            weight_mapping[name] = weight_hash
    
    # If using a repository, stage weights with deduplication
    if repo is not None and weights_to_stage:
        # Stage all weights at once for efficient deduplication
        staged_hashes = repo.stage_weights(weights_to_stage)
        weight_mapping.update(staged_hashes)
        
        # Create informative commit message
        num_weights = len(weight_mapping)
        source_name = Path(source_path).name
        commit_message = f"Import {num_weights} weights from {source_name}"
        
        # Commit the changes
        repo.commit(commit_message)
    
    # Close store if we opened it
    if isinstance(target, (str, Path, Repository)):
        store.close()
    
    return weight_mapping


def batch_convert_safetensors(
    source_dir: Union[str, Path],
    target: Union[str, Path, Repository],
    pattern: str = "*.safetensors",
    recursive: bool = True,
    preserve_structure: bool = True,
) -> Dict[str, Dict[str, str]]:
    """Convert multiple Safetensors files to Coral format.

    Args:
        source_dir: Directory containing Safetensors files
        target: Target Coral repository
        pattern: Glob pattern for files to convert
        recursive: Whether to search recursively
        preserve_structure: Whether to preserve directory structure in naming

    Returns:
        Dictionary mapping file paths to weight mappings

    Raises:
        ValueError: If source directory doesn't exist
    """
    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise ValueError(f"Source directory not found: {source_dir}")
    
    # Find all matching files
    if recursive:
        files = list(source_dir.rglob(pattern))
    else:
        files = list(source_dir.glob(pattern))
    
    if not files:
        return {}
    
    # Prepare repository
    if isinstance(target, (str, Path)):
        repo_path = Path(target)
        if repo_path.exists() and (repo_path / ".coral").exists():
            repo = Repository(repo_path)
        else:
            repo = Repository(repo_path, init=True)
    else:
        repo = target
    
    # Open store once for all conversions
    store_path = repo.coral_dir / "objects" / "weights.h5"
    store = HDF5Store(store_path)
    
    # Convert each file with progress bar
    results: Dict[str, Dict[str, str]] = {}
    all_weights: Dict[str, WeightTensor] = {}
    
    # First pass: read all weights
    for file_path in tqdm(files, desc="Reading safetensors files"):
        # Determine weight name prefix
        if preserve_structure:
            relative_path = file_path.relative_to(source_dir)
            prefix = str(relative_path.parent / relative_path.stem)
            prefix = prefix.replace("/", ".")
            if prefix == ".":
                prefix = relative_path.stem
        else:
            prefix = file_path.stem
        
        # Read weights from file
        try:
            reader = SafetensorsReader(file_path)
            file_weights = {}
            
            for tensor_name in reader.get_tensor_names():
                # Read tensor data
                tensor_data = reader.read_tensor(tensor_name)
                
                # Create full name with prefix
                full_name = f"{prefix}.{tensor_name}" if prefix else tensor_name
                
                # Create metadata
                metadata = WeightMetadata(
                    name=full_name,
                    shape=tuple(tensor_data.shape),
                    dtype=tensor_data.dtype,
                    layer_type=tensor_name.split('.')[-1] if '.' in tensor_name else None,
                    model_name=prefix,
                    compression_info={
                        "source_format": "safetensors",
                        "source_file": str(file_path.name),
                    },
                )
                
                # Create WeightTensor
                weight_tensor = WeightTensor(
                    data=tensor_data,
                    metadata=metadata,
                )
                
                all_weights[full_name] = weight_tensor
                file_weights[tensor_name] = full_name
            
            results[str(file_path)] = file_weights
            
        except Exception as e:
            print(f"Error converting {file_path}: {e}")
            continue
    
    # Second pass: stage all weights at once for efficient deduplication
    if all_weights:
        print(f"\nStaging {len(all_weights)} weights for deduplication...")
        staged_hashes = repo.stage_weights(all_weights)
        
        # Update results with actual hashes
        for file_path, file_weights in results.items():
            hash_mapping = {}
            for orig_name, full_name in file_weights.items():
                if full_name in staged_hashes:
                    hash_mapping[orig_name] = staged_hashes[full_name]
            results[file_path] = hash_mapping
        
        # Commit all changes
        num_files = len(files)
        num_weights = len(all_weights)
        commit_message = f"Batch import: {num_weights} weights from {num_files} safetensors files"
        repo.commit(commit_message)
        
        print(f"✓ Imported {num_weights} weights from {num_files} files")
        
        # Show deduplication stats
        stats = repo.deduplicator.compute_stats()
        if stats.total_weights > stats.unique_weights:
            saved = stats.total_weights - stats.unique_weights
            print(f"✓ Deduplicated {saved} weights ({saved/stats.total_weights*100:.1f}% reduction)")
    
    # Close store
    store.close()
    
    return results