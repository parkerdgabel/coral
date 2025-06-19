"""Metadata handling for Safetensors integration."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from coral.safetensors.types import MetadataDict


@dataclass
class SafetensorsMetadata:
    """Container for Safetensors file metadata.

    This class handles metadata that can be stored in Safetensors files,
    including model information, training configuration, and custom metadata.

    Attributes:
        format_version: Safetensors format version
        model_name: Name of the model
        model_version: Version of the model
        created_at: Timestamp when file was created
        created_by: Tool/library that created the file
        description: Human-readable description
        training_config: Training configuration if applicable
        custom_metadata: Additional custom metadata
    """

    format_version: str = "1.0"
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None
    description: Optional[str] = None
    training_config: Optional[Dict[str, Any]] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> MetadataDict:
        """Convert to dictionary format for Safetensors.

        Returns:
            Dictionary representation of metadata
        """
        result: MetadataDict = {
            "format_version": self.format_version,
        }
        
        if self.model_name is not None:
            result["model_name"] = self.model_name
        
        if self.model_version is not None:
            result["model_version"] = self.model_version
        
        if self.created_at is not None:
            result["created_at"] = self.created_at.isoformat()
        
        if self.created_by is not None:
            result["created_by"] = self.created_by
        
        if self.description is not None:
            result["description"] = self.description
        
        if self.training_config is not None:
            result["training_config"] = self.training_config
        
        # Add custom metadata
        result.update(self.custom_metadata)
        
        return result

    @classmethod
    def from_dict(cls, data: MetadataDict) -> "SafetensorsMetadata":
        """Create SafetensorsMetadata from dictionary.

        Args:
            data: Dictionary containing metadata

        Returns:
            SafetensorsMetadata instance
        """
        # Extract known fields
        format_version = data.get("format_version", "1.0")
        model_name = data.get("model_name")
        model_version = data.get("model_version")
        created_by = data.get("created_by")
        description = data.get("description")
        training_config = data.get("training_config")
        
        # Parse created_at if present
        created_at = None
        if "created_at" in data:
            try:
                created_at = datetime.fromisoformat(data["created_at"])
            except (ValueError, TypeError):
                # Skip if invalid format
                pass
        
        # Collect custom metadata
        known_fields = {
            "format_version",
            "model_name",
            "model_version",
            "created_at",
            "created_by",
            "description",
            "training_config",
        }
        custom_metadata = {k: v for k, v in data.items() if k not in known_fields}
        
        return cls(
            format_version=format_version,
            model_name=model_name,
            model_version=model_version,
            created_at=created_at,
            created_by=created_by,
            description=description,
            training_config=training_config,
            custom_metadata=custom_metadata,
        )


def extract_metadata(
    data: Dict[str, Any],
    prefix: Optional[str] = None,
) -> MetadataDict:
    """Extract metadata from a nested dictionary.

    This function extracts metadata fields from a potentially nested dictionary,
    optionally filtering by a prefix.

    Args:
        data: Dictionary to extract metadata from
        prefix: Optional prefix to filter metadata keys

    Returns:
        Extracted metadata dictionary
    """
    metadata: MetadataDict = {}
    
    for key, value in data.items():
        # Check if key matches prefix (if specified)
        if prefix and not key.startswith(prefix):
            continue
        
        # Remove prefix from key if present
        metadata_key = key
        if prefix and key.startswith(prefix):
            metadata_key = key[len(prefix):]
        
        # Handle nested dictionaries
        if isinstance(value, dict) and not _is_tensor_dict(value):
            nested = extract_metadata(value, prefix=None)
            for nested_key, nested_value in nested.items():
                full_key = f"{metadata_key}.{nested_key}"
                metadata[full_key] = nested_value
        else:
            metadata[metadata_key] = value
    
    return metadata


def merge_metadata(
    base: MetadataDict,
    update: MetadataDict,
    overwrite: bool = True,
) -> MetadataDict:
    """Merge two metadata dictionaries.

    Args:
        base: Base metadata dictionary
        update: Metadata to merge in
        overwrite: If True, overwrite existing keys; if False, skip them

    Returns:
        Merged metadata dictionary
    """
    result = base.copy()
    
    for key, value in update.items():
        if key in result and not overwrite:
            continue
        
        # Handle nested dictionaries
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
            and not _is_tensor_dict(value)
        ):
            # Recursively merge nested dictionaries
            result[key] = merge_metadata(result[key], value, overwrite=overwrite)
        else:
            result[key] = value
    
    return result


def _is_tensor_dict(data: Dict[str, Any]) -> bool:
    """Check if a dictionary represents tensor metadata.

    Args:
        data: Dictionary to check

    Returns:
        True if dictionary looks like tensor metadata
    """
    # Check for common tensor metadata keys
    tensor_keys = {"shape", "dtype", "data_offsets"}
    return tensor_keys.issubset(data.keys())


def validate_metadata(metadata: MetadataDict) -> List[str]:
    """Validate metadata dictionary.

    Args:
        metadata: Metadata to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check format version
    if "format_version" in metadata:
        version = metadata["format_version"]
        if not isinstance(version, str):
            errors.append(f"format_version must be string, got {type(version)}")
    
    # Check timestamps
    if "created_at" in metadata:
        created_at = metadata["created_at"]
        if isinstance(created_at, str):
            try:
                datetime.fromisoformat(created_at)
            except ValueError:
                errors.append(f"Invalid ISO format for created_at: {created_at}")
        elif not isinstance(created_at, datetime):
            errors.append(f"created_at must be string or datetime, got {type(created_at)}")
    
    # Check for reserved keys
    reserved_keys = {"__tensor_info__", "__header__"}
    for key in metadata:
        if key in reserved_keys:
            errors.append(f"Reserved metadata key used: {key}")
    
    return errors


def create_coral_metadata(
    branch: Optional[str] = None,
    commit_hash: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    delta_encoding_enabled: Optional[bool] = None,
    compression_type: Optional[str] = None,
    **kwargs: Any,
) -> MetadataDict:
    """Create metadata specific to Coral's needs.

    Args:
        branch: Git-like branch name
        commit_hash: Commit hash in Coral repository
        similarity_threshold: Threshold used for deduplication
        delta_encoding_enabled: Whether delta encoding was used
        compression_type: Type of compression applied
        **kwargs: Additional metadata fields

    Returns:
        Metadata dictionary
    """
    metadata: MetadataDict = {
        "created_by": "coral-ml",
        "created_at": datetime.utcnow().isoformat(),
    }
    
    if branch is not None:
        metadata["coral.branch"] = branch
    
    if commit_hash is not None:
        metadata["coral.commit"] = commit_hash
    
    if similarity_threshold is not None:
        metadata["coral.similarity_threshold"] = similarity_threshold
    
    if delta_encoding_enabled is not None:
        metadata["coral.delta_encoding"] = delta_encoding_enabled
    
    if compression_type is not None:
        metadata["coral.compression"] = compression_type
    
    # Add any additional metadata
    metadata.update(kwargs)
    
    return metadata