"""
SafeTensors metadata handling utilities.

This module provides utilities for handling, extracting, merging, and validating
metadata in SafeTensors files with Coral-specific extensions.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from .types import MetadataDict, SafetensorsMetadataError, TensorMetadata


class SafetensorsMetadata:
    """
    Comprehensive metadata handler for SafeTensors files.

    Features:
    - Global and tensor-specific metadata management
    - Metadata validation and schema enforcement
    - Coral-specific metadata extensions
    - Serialization/deserialization support
    - Metadata merging and conflict resolution

    Example:
        metadata = SafetensorsMetadata()
        metadata.set_global("model_type", "transformer")
        metadata.set_tensor_info("layer1.weight", layer_type="attention")
        metadata_dict = metadata.to_dict()
    """

    def __init__(self, metadata: Optional[MetadataDict] = None):
        """
        Initialize metadata handler.

        Args:
            metadata: Optional initial metadata dictionary
        """
        self._global_metadata: MetadataDict = {}
        self._tensor_metadata: Dict[str, TensorMetadata] = {}

        if metadata:
            self.load_from_dict(metadata)

    def set_global(self, key: str, value: Any) -> None:
        """
        Set a global metadata field.

        Args:
            key: Metadata key
            value: Metadata value

        Raises:
            SafetensorsMetadataError: If key or value is invalid
        """
        self._validate_metadata_key(key)
        self._validate_metadata_value(value)
        self._global_metadata[key] = value

    def get_global(self, key: str, default: Any = None) -> Any:
        """
        Get a global metadata field.

        Args:
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value or default
        """
        return self._global_metadata.get(key, default)

    def remove_global(self, key: str) -> None:
        """
        Remove a global metadata field.

        Args:
            key: Metadata key to remove
        """
        self._global_metadata.pop(key, None)

    def set_tensor_info(
        self,
        tensor_name: str,
        dtype: Optional[Any] = None,
        shape: Optional[Tuple[int, ...]] = None,
        layer_type: Optional[str] = None,
        model_name: Optional[str] = None,
        compression_info: Optional[Dict[str, Any]] = None,
        author: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Set tensor-specific metadata information.

        Args:
            tensor_name: Name of the tensor
            dtype: Data type of the tensor
            shape: Shape of the tensor
            layer_type: Type of neural network layer
            model_name: Name of the model
            compression_info: Compression/quantization information
            author: Author of the tensor/weights
            description: Human-readable description
            tags: List of tags for categorization
            **kwargs: Additional metadata fields
        """
        # Convert dtype if needed
        from .types import DType

        if dtype is not None and not isinstance(dtype, DType):
            try:
                import numpy as np

                if isinstance(dtype, np.dtype):
                    dtype = DType.from_numpy(dtype)
                elif isinstance(dtype, type):
                    dtype = DType.from_numpy(np.dtype(dtype))
                else:
                    # Assume it's already a string representation
                    dtype = DType(str(dtype))
            except (ValueError, TypeError, AttributeError):
                dtype = None

        # Create or update tensor metadata
        if tensor_name in self._tensor_metadata:
            tensor_meta = self._tensor_metadata[tensor_name]
        else:
            # Create tensor metadata with provided info
            tensor_meta = TensorMetadata(
                name=tensor_name,
                dtype=dtype or DType.FLOAT32,  # Default to float32 if not provided
                shape=shape or (),
            )

        # Update fields
        if dtype is not None:
            tensor_meta.dtype = dtype or DType.FLOAT32
        if shape is not None:
            tensor_meta.shape = shape
        if layer_type is not None:
            tensor_meta.layer_type = layer_type
        if model_name is not None:
            tensor_meta.model_name = model_name
        if compression_info is not None:
            tensor_meta.compression_info = compression_info
        if author is not None:
            tensor_meta.author = author
        if description is not None:
            tensor_meta.description = description
        if tags is not None:
            tensor_meta.tags = tags

        # Add any additional fields
        for key, value in kwargs.items():
            setattr(tensor_meta, key, value)

        self._tensor_metadata[tensor_name] = tensor_meta

        # Also store in global metadata with tensor prefix
        tensor_metadata_dict = tensor_meta.to_dict()
        self._global_metadata[f"tensor.{tensor_name}"] = tensor_metadata_dict

    def get_tensor_info(self, tensor_name: str) -> Optional[TensorMetadata]:
        """
        Get tensor-specific metadata.

        Args:
            tensor_name: Name of the tensor

        Returns:
            TensorMetadata object or None if not found
        """
        return self._tensor_metadata.get(tensor_name)

    def remove_tensor_info(self, tensor_name: str) -> None:
        """
        Remove tensor-specific metadata.

        Args:
            tensor_name: Name of the tensor
        """
        self._tensor_metadata.pop(tensor_name, None)
        self._global_metadata.pop(f"tensor.{tensor_name}", None)

    def set_coral_info(
        self,
        version: str = "1.0.0",
        branch: str = "main",
        commit_hash: Optional[str] = None,
        repository_path: Optional[str] = None,
        created_by: str = "coral-ml",
    ) -> None:
        """
        Set Coral-specific metadata information.

        Args:
            version: Coral version
            branch: Git branch name
            commit_hash: Git commit hash
            repository_path: Path to Coral repository
            created_by: Tool that created the file
        """
        coral_info = {
            "version": version,
            "branch": branch,
            "created_by": created_by,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }

        if commit_hash:
            coral_info["commit_hash"] = commit_hash
        if repository_path:
            coral_info["repository_path"] = repository_path

        # Store both as a nested object and as flattened keys for compatibility
        self.set_global("coral", coral_info)

        # Also store as flattened keys with coral. prefix for easier access
        for key, value in coral_info.items():
            self.set_global(f"coral.{key}", value)

    def set_model_info(
        self,
        model_type: str,
        architecture: Optional[str] = None,
        framework: Optional[str] = None,
        version: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set model-specific information.

        Args:
            model_type: Type of model (e.g., "transformer", "cnn")
            architecture: Specific architecture (e.g., "bert-base", "resnet50")
            framework: ML framework used (e.g., "pytorch", "tensorflow")
            version: Model version
            config: Model configuration dictionary
        """
        model_info: Dict[str, Any] = {"type": model_type}

        if architecture:
            model_info["architecture"] = architecture
        if framework:
            model_info["framework"] = framework
        if version:
            model_info["version"] = version
        if config:
            model_info["config"] = config

        self.set_global("model", model_info)

    def add_training_info(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        optimizer: Optional[str] = None,
        loss_function: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        dataset: Optional[str] = None,
    ) -> None:
        """
        Add training-related metadata.

        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate used
            optimizer: Optimizer name
            loss_function: Loss function used
            metrics: Final training metrics
            dataset: Dataset name or description
        """
        training_info: Dict[str, Any] = {}

        if epochs is not None:
            training_info["epochs"] = epochs
        if batch_size is not None:
            training_info["batch_size"] = batch_size
        if learning_rate is not None:
            training_info["learning_rate"] = learning_rate
        if optimizer is not None:
            training_info["optimizer"] = optimizer
        if loss_function is not None:
            training_info["loss_function"] = loss_function
        if metrics is not None:
            training_info["final_metrics"] = metrics
        if dataset is not None:
            training_info["dataset"] = dataset

        if training_info:
            self.set_global("training", training_info)

    def get_tensor_names(self) -> List[str]:
        """
        Get list of tensor names with metadata.

        Returns:
            List of tensor names
        """
        return list(self._tensor_metadata.keys())

    def get_global_keys(self) -> List[str]:
        """
        Get list of global metadata keys.

        Returns:
            List of global metadata keys
        """
        return list(self._global_metadata.keys())

    def to_dict(self) -> MetadataDict:
        """
        Convert metadata to dictionary for serialization.

        Returns:
            Dictionary containing all metadata
        """
        return self._global_metadata.copy()

    def load_from_dict(self, metadata: MetadataDict) -> None:
        """
        Load metadata from dictionary.

        Args:
            metadata: Metadata dictionary to load

        Raises:
            SafetensorsMetadataError: If metadata format is invalid
        """
        try:
            self._global_metadata = metadata.copy()

            # Extract tensor-specific metadata
            self._tensor_metadata.clear()
            tensor_keys = [k for k in metadata.keys() if k.startswith("tensor.")]

            for key in tensor_keys:
                tensor_name = key[7:]  # Remove "tensor." prefix
                tensor_data = metadata[key]

                if isinstance(tensor_data, dict):
                    try:
                        tensor_meta = TensorMetadata.from_dict(tensor_data)
                        self._tensor_metadata[tensor_name] = tensor_meta
                    except Exception:
                        # Skip invalid tensor metadata
                        pass
        except Exception as e:
            raise SafetensorsMetadataError(f"Failed to load metadata: {e}") from e

    def validate(self) -> Dict[str, Any]:
        """
        Validate metadata for consistency and completeness.

        Returns:
            Dictionary with validation results
        """
        validation_result: Dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "global_keys": len(self._global_metadata),
            "tensor_count": len(self._tensor_metadata),
        }

        # Validate JSON serializability
        try:
            json.dumps(self._global_metadata)
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Metadata not JSON serializable: {e}")

        # Check for required Coral fields
        if "coral" not in self._global_metadata:
            validation_result["warnings"].append("Missing Coral metadata information")

        # Validate tensor metadata consistency
        for tensor_name, tensor_meta in self._tensor_metadata.items():
            try:
                tensor_meta.to_dict()
            except Exception as e:
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"Invalid tensor metadata for '{tensor_name}': {e}"
                )

        return validation_result

    def merge(self, other: "SafetensorsMetadata", overwrite: bool = False) -> None:
        """
        Merge metadata from another SafetensorsMetadata instance.

        Args:
            other: Other metadata instance to merge
            overwrite: Whether to overwrite existing keys

        Raises:
            SafetensorsMetadataError: If merge fails
        """
        try:
            # Merge global metadata
            for key, value in other._global_metadata.items():
                if overwrite or key not in self._global_metadata:
                    self._global_metadata[key] = value

            # Merge tensor metadata
            for tensor_name, tensor_meta in other._tensor_metadata.items():
                if overwrite or tensor_name not in self._tensor_metadata:
                    self._tensor_metadata[tensor_name] = tensor_meta

        except Exception as e:
            raise SafetensorsMetadataError(f"Failed to merge metadata: {e}") from e

    def filter_tensors(self, tensor_names: Set[str]) -> "SafetensorsMetadata":
        """
        Create a new metadata instance with only specified tensors.

        Args:
            tensor_names: Set of tensor names to include

        Returns:
            New SafetensorsMetadata instance with filtered tensors
        """
        new_metadata = SafetensorsMetadata()

        # Copy global metadata (excluding tensor-specific)
        for key, value in self._global_metadata.items():
            if not key.startswith("tensor."):
                new_metadata._global_metadata[key] = value

        # Copy only specified tensor metadata
        for tensor_name in tensor_names:
            if tensor_name in self._tensor_metadata:
                tensor_meta = self._tensor_metadata[tensor_name]
                new_metadata._tensor_metadata[tensor_name] = tensor_meta
                new_metadata._global_metadata[f"tensor.{tensor_name}"] = (
                    tensor_meta.to_dict()
                )

        return new_metadata

    def _validate_metadata_key(self, key: str) -> None:
        """Validate metadata key format."""
        if not isinstance(key, str) or not key:
            raise SafetensorsMetadataError("Metadata key must be a non-empty string")

        if len(key) > 256:
            raise SafetensorsMetadataError(
                f"Metadata key too long: {len(key)} characters"
            )

    def _validate_metadata_value(self, value: Any) -> None:
        """Validate metadata value is JSON serializable."""
        try:
            json.dumps(value)
        except (TypeError, ValueError) as e:
            raise SafetensorsMetadataError(
                f"Metadata value not serializable: {e}"
            ) from e

    def __len__(self) -> int:
        """Get total number of metadata entries."""
        return len(self._global_metadata)

    def __contains__(self, key: str) -> bool:
        """Check if metadata key exists."""
        return key in self._global_metadata

    def __repr__(self) -> str:
        """String representation of metadata."""
        return (
            f"SafetensorsMetadata(global_keys={len(self._global_metadata)}, "
            f"tensors={len(self._tensor_metadata)})"
        )


def extract_metadata(metadata_dict: MetadataDict) -> SafetensorsMetadata:
    """
    Extract structured metadata from a raw metadata dictionary.

    Args:
        metadata_dict: Raw metadata dictionary

    Returns:
        SafetensorsMetadata instance
    """
    return SafetensorsMetadata(metadata_dict)


def merge_metadata(
    *metadata_instances: SafetensorsMetadata, overwrite: bool = False
) -> SafetensorsMetadata:
    """
    Merge multiple metadata instances into one.

    Args:
        *metadata_instances: SafetensorsMetadata instances to merge
        overwrite: Whether to overwrite existing keys

    Returns:
        New merged SafetensorsMetadata instance
    """
    if not metadata_instances:
        return SafetensorsMetadata()

    result = SafetensorsMetadata()

    for metadata in metadata_instances:
        result.merge(metadata, overwrite=overwrite)

    return result


def create_default_metadata(
    model_type: str = "neural_network",
    framework: str = "unknown",
    author: Optional[str] = None,
) -> SafetensorsMetadata:
    """
    Create a SafetensorsMetadata instance with sensible defaults.

    Args:
        model_type: Type of model
        framework: ML framework used
        author: Author name

    Returns:
        SafetensorsMetadata instance with defaults
    """
    metadata = SafetensorsMetadata()

    # Set Coral info
    metadata.set_coral_info()

    # Set basic model info
    metadata.set_model_info(model_type=model_type, framework=framework)

    # Set author if provided
    if author:
        metadata.set_global("author", author)

    return metadata
