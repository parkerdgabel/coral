"""Configuration system for Coral.

This module provides a unified, hierarchical configuration system that
consolidates all settings into a single source of truth.

Configuration Sources (Priority Order):
1. Programmatic (highest) - Direct API calls
2. Environment Variables - CORAL_* prefixed variables
3. Repository Config - .coral/coral.toml (per-repository)
4. User Config - ~/.config/coral/config.toml (global)
5. Defaults (lowest) - Built-in defaults

Example Usage:
    from coral.config import load_config, CoralConfig

    # Load configuration from all sources
    config = load_config(repo_path=Path("./my_model"))

    # Access settings
    print(config.core.similarity_threshold)  # 0.98
    print(config.delta.delta_type)           # "compressed"

    # Create custom configuration
    from coral.config import CoreConfig
    custom = CoralConfig(core=CoreConfig(similarity_threshold=0.95))

Environment Variables:
    All settings can be overridden with CORAL_ prefixed variables:
    - CORAL_CORE_SIMILARITY_THRESHOLD=0.99
    - CORAL_USER_NAME="CI Bot"
    - CORAL_LOGGING_LEVEL=DEBUG
"""

from .loader import (
    ConfigLoader,
    get_default_config,
    load_config,
)
from .schema import (
    CheckpointDefaults,
    CompressionType,
    CoralConfig,
    CoreConfig,
    DeltaEncodingConfig,
    HuggingFaceConfig,
    LoggingConfig,
    LogLevel,
    LSHConfig,
    RemoteConfig,
    S3StorageConfig,
    SimHashConfig,
    StorageConfig,
    UserConfig,
)
from .validation import (
    ConfigValidationError,
    ValidationError,
    ValidationResult,
    validate_config,
    validate_value,
)

__all__ = [
    # Main config class
    "CoralConfig",
    # Section configs
    "UserConfig",
    "CoreConfig",
    "DeltaEncodingConfig",
    "StorageConfig",
    "S3StorageConfig",
    "LSHConfig",
    "SimHashConfig",
    "CheckpointDefaults",
    "LoggingConfig",
    "HuggingFaceConfig",
    "RemoteConfig",
    # Enums
    "CompressionType",
    "LogLevel",
    # Loader
    "ConfigLoader",
    "load_config",
    "get_default_config",
    # Validation
    "validate_config",
    "validate_value",
    "ValidationError",
    "ValidationResult",
    "ConfigValidationError",
]
