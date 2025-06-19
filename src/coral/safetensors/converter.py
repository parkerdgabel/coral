"""
Conversion functions between Coral and SafeTensors formats.

This module provides comprehensive conversion utilities for importing and exporting
between Coral's weight management system and SafeTensors format.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np

from ..core.weight_tensor import WeightMetadata, WeightTensor
from ..version_control.repository import Repository
from .metadata import SafetensorsMetadata, create_default_metadata
from .reader import SafetensorsReader
from .types import (
    MetadataDict,
    SafetensorsConversionError,
)
from .writer import SafetensorsWriter

logger = logging.getLogger(__name__)


def convert_coral_to_safetensors(
    repository: Repository = None,
    output_path: Union[str, Path] = None,
    branch: str = "main",
    weight_filter: Optional[Set[str]] = None,
    metadata: Optional[MetadataDict] = None,
    include_coral_metadata: bool = True,
    # New CLI-compatible parameters
    source: Repository = None,
    weight_names: Optional[List[str]] = None,
    include_metadata: bool = True,
    custom_metadata: Optional[MetadataDict] = None,
) -> Dict[str, Any]:
    """
    Convert Coral repository weights to SafeTensors format.

    Args:
        repository: Coral repository to export from (legacy)
        output_path: Path for output SafeTensors file
        branch: Branch to export (default: "main")
        weight_filter: Optional set of weight names to include (legacy)
        metadata: Additional metadata to include (legacy)
        include_coral_metadata: Whether to include Coral-specific metadata (legacy)

        # New CLI-compatible parameters:
        source: Coral repository to export from (new API)
        weight_names: List of weight names to include (new API)
        include_metadata: Whether to include Coral metadata (new API)
        custom_metadata: Custom metadata to include (new API)

    Returns:
        Dictionary with conversion statistics

    Raises:
        SafetensorsConversionError: If conversion fails
    """
    try:
        # Handle parameter compatibility
        repo = source or repository
        if repo is None:
            raise SafetensorsConversionError("Must provide source or repository")
        if output_path is None:
            raise SafetensorsConversionError("Must provide output_path")

        # Handle weight filtering
        if weight_names:
            weight_filter = set(weight_names)

        # Handle metadata settings
        if include_metadata is False:
            include_coral_metadata = False
        if custom_metadata:
            metadata = custom_metadata

        logger.info(f"Converting Coral repository to SafeTensors: {output_path}")

        # Switch to specified branch
        original_branch = repo.branch_manager.get_current_branch()
        if branch != original_branch:
            repo.checkout(branch)

        try:
            # Get all weights from repository
            all_weights = repo.get_all_weights()

            if not all_weights:
                raise SafetensorsConversionError("No weights found in repository")

            # Apply weight filter if specified
            if weight_filter:
                filtered_weights = {
                    name: weight
                    for name, weight in all_weights.items()
                    if name in weight_filter
                }
                if not filtered_weights:
                    raise SafetensorsConversionError(
                        "No weights match the specified filter"
                    )
                all_weights = filtered_weights

            # Create SafeTensors metadata
            st_metadata = SafetensorsMetadata(metadata or {})

            if include_coral_metadata:
                # Add Coral-specific metadata
                st_metadata.set_coral_info(
                    branch=branch, repository_path=str(repo.path)
                )

            # Convert weights and add tensor metadata
            tensor_data = {}
            conversion_stats = {
                "total_weights": len(all_weights),
                "total_parameters": 0,
                "total_size_bytes": 0,
                "dtype_distribution": {},
                "converted_weights": [],
                "skipped_weights": [],
            }

            for name, weight in all_weights.items():
                try:
                    # Convert Coral WeightTensor to numpy array
                    tensor_data[name] = weight.data

                    # Add tensor-specific metadata
                    st_metadata.set_tensor_info(
                        tensor_name=name,
                        dtype=weight.data.dtype,
                        shape=weight.data.shape,
                        layer_type=weight.metadata.layer_type,
                        model_name=weight.metadata.model_name,
                        compression_info=weight.metadata.compression_info,
                        description="Exported from Coral repository",
                    )

                    # Update statistics
                    conversion_stats["total_parameters"] += weight.data.size
                    conversion_stats["total_size_bytes"] += weight.data.nbytes

                    dtype_str = str(weight.data.dtype)
                    conversion_stats["dtype_distribution"][dtype_str] = (
                        conversion_stats["dtype_distribution"].get(dtype_str, 0) + 1
                    )

                    conversion_stats["converted_weights"].append(name)

                except Exception as e:
                    logger.warning(f"Failed to convert weight '{name}': {e}")
                    conversion_stats["skipped_weights"].append(name)

            if not tensor_data:
                raise SafetensorsConversionError("No weights could be converted")

            # Write SafeTensors file
            with SafetensorsWriter(
                output_path, metadata=st_metadata.to_dict()
            ) as writer:
                writer.add_tensors(tensor_data)
                writer.write()

            conversion_stats["output_file"] = str(output_path)
            conversion_stats["success"] = True

            logger.info(
                f"Successfully converted {len(tensor_data)} weights to SafeTensors"
            )
            return conversion_stats

        finally:
            # Restore original branch
            if branch != original_branch:
                repo.checkout(original_branch)

    except Exception as e:
        if isinstance(e, SafetensorsConversionError):
            raise
        raise SafetensorsConversionError(
            f"Failed to convert Coral to SafeTensors: {e}"
        ) from e


def convert_safetensors_to_coral(
    source_path: Union[str, Path] = None,
    target: Repository = None,
    preserve_metadata: bool = True,
    weight_names: Optional[List[str]] = None,
    exclude_weights: Optional[Set[str]] = None,
    commit_message: Optional[str] = None,
    overwrite_existing: bool = False,
    auto_commit: bool = True,
    # Legacy parameter names for backward compatibility
    input_path: Union[str, Path] = None,
    repository: Repository = None,
    weight_filter: Optional[Set[str]] = None,
    exclude_filter: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Convert SafeTensors file to Coral repository weights.

    Args:
        source_path: Path to SafeTensors file (new API)
        target: Coral repository to import into (new API)
        preserve_metadata: Whether to preserve metadata (new API)
        weight_names: List of weight names to include (new API)
        exclude_weights: Set of weight names to exclude (new API)
        commit_message: Optional commit message for imported weights
        overwrite_existing: Whether to overwrite existing weights
        auto_commit: Whether to automatically commit imported weights (default: True)

        # Legacy parameters for backward compatibility:
        input_path: Path to SafeTensors file (legacy)
        repository: Coral repository to import into (legacy)
        weight_filter: Set of weight names to include (legacy)
        exclude_filter: Set of weight names to exclude (legacy)

    Returns:
        Dictionary with conversion statistics including weight mapping

    Raises:
        SafetensorsConversionError: If conversion fails
    """
    try:
        # Handle parameter compatibility
        file_path = source_path or input_path
        repo = target or repository
        weight_include = set(weight_names) if weight_names else weight_filter
        weight_exclude = exclude_weights or exclude_filter

        if file_path is None:
            raise SafetensorsConversionError("Must provide source_path or input_path")
        if repo is None:
            raise SafetensorsConversionError("Must provide target or repository")

        logger.info(f"Converting SafeTensors to Coral repository: {file_path}")

        # Read SafeTensors file
        with SafetensorsReader(file_path) as reader:
            # Get available tensor names
            available_names = set(reader.get_tensor_names())

            # Apply filters
            target_names = available_names
            if weight_include:
                target_names = target_names.intersection(weight_include)
            if weight_exclude:
                target_names = target_names - weight_exclude

            if not target_names:
                raise SafetensorsConversionError(
                    "No tensors match the specified filters"
                )

            # Read tensors and metadata
            tensor_data = reader.read_tensors(list(target_names))
            global_metadata = reader.metadata

            # Convert to Coral WeightTensors
            coral_weights = {}
            conversion_stats = {
                "total_tensors": len(target_names),
                "total_parameters": 0,
                "total_size_bytes": 0,
                "dtype_distribution": {},
                "imported_weights": [],
                "skipped_weights": [],
                "overwritten_weights": [],
            }

            for name in target_names:
                try:
                    tensor = tensor_data[name]

                    # Get tensor-specific metadata if available
                    tensor_metadata = reader.get_tensor_metadata(name)

                    # Create Coral WeightMetadata
                    coral_metadata = WeightMetadata(
                        name=name,
                        shape=tensor.shape,
                        dtype=tensor.dtype,
                        layer_type=tensor_metadata.layer_type,
                        model_name=tensor_metadata.model_name,
                        compression_info=tensor_metadata.compression_info,
                    )

                    # Create WeightTensor
                    weight_tensor = WeightTensor(data=tensor, metadata=coral_metadata)

                    # Check if weight already exists
                    existing_weight = repo.get_weight(name)
                    if existing_weight is not None and not overwrite_existing:
                        logger.warning(f"Weight '{name}' already exists, skipping")
                        conversion_stats["skipped_weights"].append(name)
                        continue
                    elif existing_weight is not None:
                        conversion_stats["overwritten_weights"].append(name)

                    coral_weights[name] = weight_tensor

                    # Update statistics
                    conversion_stats["total_parameters"] += tensor.size
                    conversion_stats["total_size_bytes"] += tensor.nbytes

                    dtype_str = str(tensor.dtype)
                    conversion_stats["dtype_distribution"][dtype_str] = (
                        conversion_stats["dtype_distribution"].get(dtype_str, 0) + 1
                    )

                    conversion_stats["imported_weights"].append(name)

                except Exception as e:
                    logger.warning(f"Failed to convert tensor '{name}': {e}")
                    conversion_stats["skipped_weights"].append(name)

            if not coral_weights:
                raise SafetensorsConversionError("No tensors could be converted")

            # Stage weights in repository
            repo.stage_weights(coral_weights)

            # Commit if auto_commit is True or commit_message provided
            if auto_commit or commit_message:
                message = (
                    commit_message
                    or f"Import {len(coral_weights)} weights from SafeTensors"
                )
                commit_hash = repo.commit(message)
                conversion_stats["commit_hash"] = commit_hash

            conversion_stats["input_file"] = str(file_path)
            conversion_stats["success"] = True
            conversion_stats["safetensors_metadata"] = global_metadata

            # Create weight mapping for CLI compatibility
            weight_mapping = {name: name for name in coral_weights.keys()}
            conversion_stats["weight_mapping"] = weight_mapping

            logger.info(f"Successfully imported {len(coral_weights)} weights to Coral")

            # Return weight mapping for CLI compatibility, but keep stats for
            # detailed info
            return weight_mapping

    except Exception as e:
        if isinstance(e, SafetensorsConversionError):
            raise
        raise SafetensorsConversionError(
            f"Failed to convert SafeTensors to Coral: {e}"
        ) from e


def batch_convert_safetensors(
    input_paths: List[Union[str, Path]],
    output_directory: Union[str, Path],
    conversion_type: str = "to_coral",
    repository: Optional[Repository] = None,
    name_pattern: str = "{original_name}",
    **kwargs,
) -> Dict[str, Any]:
    """
    Convert multiple SafeTensors files in batch.

    Args:
        input_paths: List of input file paths
        output_directory: Directory for output files
        conversion_type: "to_coral" or "to_safetensors"
        repository: Coral repository (required for "to_coral")
        name_pattern: Naming pattern for output files
        **kwargs: Additional arguments for conversion functions

    Returns:
        Dictionary with batch conversion results

    Raises:
        SafetensorsConversionError: If batch conversion fails
    """
    try:
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_stats = {
            "total_files": len(input_paths),
            "successful_conversions": 0,
            "failed_conversions": 0,
            "conversion_results": {},
            "errors": {},
        }

        for input_path in input_paths:
            input_path = Path(input_path)

            try:
                if conversion_type == "to_coral":
                    if repository is None:
                        raise SafetensorsConversionError(
                            "Repository required for to_coral conversion"
                        )

                    result = convert_safetensors_to_coral(
                        input_path=input_path, repository=repository, **kwargs
                    )

                elif conversion_type == "to_safetensors":
                    if repository is None:
                        raise SafetensorsConversionError(
                            "Repository required for to_safetensors conversion"
                        )

                    # Generate output filename
                    output_name = name_pattern.format(
                        original_name=input_path.stem,
                        index=batch_stats["successful_conversions"],
                    )
                    output_path = output_dir / f"{output_name}.safetensors"

                    result = convert_coral_to_safetensors(
                        repository=repository, output_path=output_path, **kwargs
                    )

                else:
                    raise SafetensorsConversionError(
                        f"Unknown conversion type: {conversion_type}"
                    )

                batch_stats["conversion_results"][str(input_path)] = result
                batch_stats["successful_conversions"] += 1

            except Exception as e:
                batch_stats["failed_conversions"] += 1
                batch_stats["errors"][str(input_path)] = str(e)
                logger.error(f"Failed to convert {input_path}: {e}")

        batch_stats["success_rate"] = (
            batch_stats["successful_conversions"] / batch_stats["total_files"]
            if batch_stats["total_files"] > 0
            else 0
        )

        return batch_stats

    except Exception as e:
        raise SafetensorsConversionError(f"Batch conversion failed: {e}") from e


def convert_weights_dict_to_safetensors(
    weights: Dict[str, np.ndarray],
    output_path: Union[str, Path],
    metadata: Optional[MetadataDict] = None,
    model_type: str = "neural_network",
    framework: str = "unknown",
) -> Dict[str, Any]:
    """
    Convert a dictionary of weights directly to SafeTensors format.

    Args:
        weights: Dictionary mapping weight names to numpy arrays
        output_path: Path for output SafeTensors file
        metadata: Optional metadata dictionary
        model_type: Type of model for default metadata
        framework: Framework name for default metadata

    Returns:
        Dictionary with conversion statistics

    Raises:
        SafetensorsConversionError: If conversion fails
    """
    try:
        if not weights:
            raise SafetensorsConversionError("No weights provided")

        # Create metadata
        st_metadata = create_default_metadata(
            model_type=model_type, framework=framework
        )
        if metadata:
            st_metadata.load_from_dict(metadata)

        # Add tensor metadata
        for name, _tensor in weights.items():
            st_metadata.set_tensor_info(
                tensor_name=name, description="Converted numpy array"
            )

        # Write SafeTensors file
        with SafetensorsWriter(output_path, metadata=st_metadata.to_dict()) as writer:
            writer.add_tensors(weights)
            writer.write()

        # Calculate statistics
        stats = {
            "total_weights": len(weights),
            "total_parameters": sum(w.size for w in weights.values()),
            "total_size_bytes": sum(w.nbytes for w in weights.values()),
            "output_file": str(output_path),
            "success": True,
        }

        return stats

    except Exception as e:
        raise SafetensorsConversionError(f"Failed to convert weights dict: {e}") from e


def validate_conversion_compatibility(
    source_path: Union[str, Path], target_format: str = "safetensors"
) -> Dict[str, Any]:
    """
    Validate if a file/repository can be converted to target format.

    Args:
        source_path: Path to source file or repository
        target_format: Target format ("safetensors" or "coral")

    Returns:
        Dictionary with compatibility information
    """
    compatibility = {
        "compatible": True,
        "warnings": [],
        "errors": [],
        "unsupported_dtypes": [],
        "large_tensors": [],
        "total_size_mb": 0,
    }

    try:
        source_path = Path(source_path)

        if target_format == "safetensors":
            # Check if source is a Coral repository
            if source_path.is_dir() and (source_path / ".coral").exists():
                repo = Repository(source_path)
                weights = repo.get_all_weights()

                for name, weight in weights.items():
                    # Check dtype compatibility
                    try:
                        from .types import DType

                        DType.from_numpy(weight.data.dtype)
                    except ValueError:
                        compatibility["unsupported_dtypes"].append(
                            (name, str(weight.data.dtype))
                        )
                        compatibility["compatible"] = False

                    # Check for large tensors
                    size_mb = weight.data.nbytes / (1024 * 1024)
                    compatibility["total_size_mb"] += size_mb

                    if size_mb > 100:  # 100MB threshold
                        compatibility["large_tensors"].append((name, size_mb))
                        compatibility["warnings"].append(
                            f"Large tensor '{name}': {size_mb:.1f}MB"
                        )

        elif target_format == "coral":
            # Check if source is SafeTensors
            if source_path.suffix == ".safetensors":
                with SafetensorsReader(source_path) as reader:
                    file_info = reader.get_file_info()
                    compatibility["total_size_mb"] = file_info[
                        "total_tensor_size_bytes"
                    ] / (1024 * 1024)

                    # SafeTensors to Coral should always be compatible
                    if compatibility["total_size_mb"] > 1000:  # 1GB threshold
                        compatibility["warnings"].append(
                            f"Very large file: {compatibility['total_size_mb']:.1f}MB"
                        )

    except Exception as e:
        compatibility["compatible"] = False
        compatibility["errors"].append(str(e))

    return compatibility
