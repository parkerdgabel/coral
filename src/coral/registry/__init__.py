"""Model registry integration for Coral.

This module provides publishing capabilities to various model registries
including Hugging Face Hub, MLflow, and local registries.

Example:
    >>> from coral.registry import ModelPublisher
    >>>
    >>> publisher = ModelPublisher(repo)
    >>> publisher.publish_huggingface("my-org/my-model", commit_ref="abc123")
    >>> publisher.publish_mlflow("my-model", experiment_id="exp123")
"""

from .registry import (
    ModelPublisher,
    PublishResult,
    RegistryType,
)

__all__ = [
    "ModelPublisher",
    "PublishResult",
    "RegistryType",
]
