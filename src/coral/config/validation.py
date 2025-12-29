"""Configuration validation for Coral.

This module provides validation utilities for configuration values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .schema import CoralConfig


@dataclass
class ValidationError:
    """Represents a configuration validation error."""

    key: str
    message: str
    value: Any = None

    def __str__(self) -> str:
        if self.value is not None:
            return f"{self.key}: {self.message} (got: {self.value!r})"
        return f"{self.key}: {self.message}"


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    valid: bool
    errors: list[ValidationError]
    warnings: list[ValidationError]

    def __bool__(self) -> bool:
        return self.valid


class ConfigValidationError(Exception):
    """Raised when configuration validation fails during a save operation.

    This exception is raised when attempting to save an invalid configuration
    to disk. It provides detailed information about what validation failed.

    Attributes:
        errors: List of ValidationError objects describing what failed
        warnings: List of ValidationError objects for non-fatal issues
    """

    def __init__(
        self,
        message: str,
        errors: list[ValidationError],
        warnings: Optional[list[ValidationError]] = None,
    ) -> None:
        super().__init__(message)
        self.errors = errors
        self.warnings = warnings or []

    def __str__(self) -> str:
        lines = [super().__str__()]
        if self.errors:
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"  - {error}")
        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        return "\n".join(lines)


def validate_config(config: CoralConfig) -> ValidationResult:
    """Validate a configuration instance.

    Args:
        config: Configuration to validate

    Returns:
        ValidationResult with errors and warnings
    """
    errors: list[ValidationError] = []
    warnings: list[ValidationError] = []

    # Validate core settings
    _validate_core(config, errors, warnings)

    # Validate delta settings
    _validate_delta(config, errors, warnings)

    # Validate storage settings
    _validate_storage(config, errors, warnings)

    # Validate LSH settings
    _validate_lsh(config, errors, warnings)

    # Validate SimHash settings
    _validate_simhash(config, errors, warnings)

    # Validate logging settings
    _validate_logging(config, errors, warnings)

    # Validate remotes
    _validate_remotes(config, errors, warnings)

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def _validate_core(
    config: CoralConfig,
    errors: list[ValidationError],
    warnings: list[ValidationError],
) -> None:
    """Validate core configuration."""
    # Import at runtime to avoid circular imports
    from ..delta.delta_encoder import DeltaType

    core = config.core

    # Compression type
    valid_compressions = {"gzip", "lzf", "none"}
    if core.compression not in valid_compressions:
        errors.append(
            ValidationError(
                "core.compression",
                f"must be one of {valid_compressions}",
                core.compression,
            )
        )

    # Compression level
    if not 1 <= core.compression_level <= 9:
        errors.append(
            ValidationError(
                "core.compression_level",
                "must be between 1 and 9",
                core.compression_level,
            )
        )

    # Similarity threshold
    if not 0.0 <= core.similarity_threshold <= 1.0:
        errors.append(
            ValidationError(
                "core.similarity_threshold",
                "must be between 0.0 and 1.0",
                core.similarity_threshold,
            )
        )
    elif core.similarity_threshold < 0.9:
        warnings.append(
            ValidationError(
                "core.similarity_threshold",
                "low threshold may cause false positives in deduplication",
                core.similarity_threshold,
            )
        )

    # Magnitude tolerance
    if not 0.0 <= core.magnitude_tolerance <= 1.0:
        errors.append(
            ValidationError(
                "core.magnitude_tolerance",
                "must be between 0.0 and 1.0",
                core.magnitude_tolerance,
            )
        )

    # Delta type
    valid_delta_types = {e.value for e in DeltaType}
    delta_type_value = None

    # Handle both DeltaType enum and string values
    if isinstance(core.delta_type, DeltaType):
        delta_type_value = core.delta_type.value
    elif isinstance(core.delta_type, str):
        delta_type_value = core.delta_type
    else:
        delta_type_value = str(core.delta_type)

    if delta_type_value not in valid_delta_types:
        errors.append(
            ValidationError(
                "core.delta_type",
                f"must be one of {sorted(valid_delta_types)}",
                delta_type_value,
            )
        )

    # Lossy delta type warning
    lossy_types = {"int8_quantized", "int16_quantized", "sparse", "per_axis_scaled"}
    if delta_type_value in lossy_types and not core.strict_reconstruction:
        warnings.append(
            ValidationError(
                "core.delta_type",
                f"'{delta_type_value}' is lossy; enable strict_reconstruction",
                delta_type_value,
            )
        )


def _validate_delta(
    config: CoralConfig,
    errors: list[ValidationError],
    warnings: list[ValidationError],
) -> None:
    """Validate delta encoding configuration."""
    delta = config.delta

    # Sparse threshold
    if delta.sparse_threshold < 0:
        errors.append(
            ValidationError(
                "delta.sparse_threshold",
                "must be non-negative",
                delta.sparse_threshold,
            )
        )

    # Quantization bits
    if delta.quantization_bits not in (8, 16):
        errors.append(
            ValidationError(
                "delta.quantization_bits",
                "must be 8 or 16",
                delta.quantization_bits,
            )
        )

    # Min weight size
    if delta.min_weight_size < 0:
        errors.append(
            ValidationError(
                "delta.min_weight_size",
                "must be non-negative",
                delta.min_weight_size,
            )
        )

    # Max delta ratio
    if not 0.0 <= delta.max_delta_ratio <= 2.0:
        errors.append(
            ValidationError(
                "delta.max_delta_ratio",
                "must be between 0.0 and 2.0",
                delta.max_delta_ratio,
            )
        )


def _validate_storage(
    config: CoralConfig,
    errors: list[ValidationError],
    warnings: list[ValidationError],
) -> None:
    """Validate storage configuration."""
    storage = config.storage
    s3 = config.s3

    # Compression type
    valid_compressions = {"gzip", "lzf", "none"}
    if storage.compression not in valid_compressions:
        errors.append(
            ValidationError(
                "storage.compression",
                f"must be one of {valid_compressions}",
                storage.compression,
            )
        )

    # Compression level
    if not 1 <= storage.compression_level <= 9:
        errors.append(
            ValidationError(
                "storage.compression_level",
                "must be between 1 and 9",
                storage.compression_level,
            )
        )

    # S3 max concurrency
    if s3.max_concurrency < 1:
        errors.append(
            ValidationError(
                "s3.max_concurrency",
                "must be at least 1",
                s3.max_concurrency,
            )
        )
    elif s3.max_concurrency > 50:
        warnings.append(
            ValidationError(
                "s3.max_concurrency",
                "high concurrency may cause rate limiting",
                s3.max_concurrency,
            )
        )

    # S3 chunk size
    if s3.chunk_size < 5 * 1024 * 1024:  # 5MB minimum for S3 multipart
        warnings.append(
            ValidationError(
                "s3.chunk_size",
                "chunk size below 5MB may cause issues with S3 multipart uploads",
                s3.chunk_size,
            )
        )


def _validate_lsh(
    config: CoralConfig,
    errors: list[ValidationError],
    warnings: list[ValidationError],
) -> None:
    """Validate LSH configuration."""
    lsh = config.lsh

    if lsh.num_hyperplanes < 1:
        errors.append(
            ValidationError(
                "lsh.num_hyperplanes",
                "must be at least 1",
                lsh.num_hyperplanes,
            )
        )

    if lsh.num_tables < 1:
        errors.append(
            ValidationError(
                "lsh.num_tables",
                "must be at least 1",
                lsh.num_tables,
            )
        )

    if lsh.max_candidates < 1:
        errors.append(
            ValidationError(
                "lsh.max_candidates",
                "must be at least 1",
                lsh.max_candidates,
            )
        )


def _validate_simhash(
    config: CoralConfig,
    errors: list[ValidationError],
    warnings: list[ValidationError],
) -> None:
    """Validate SimHash configuration."""
    simhash = config.simhash

    if simhash.num_bits not in (64, 128):
        errors.append(
            ValidationError(
                "simhash.num_bits",
                "must be 64 or 128",
                simhash.num_bits,
            )
        )

    if not 0.0 <= simhash.similarity_threshold <= 1.0:
        errors.append(
            ValidationError(
                "simhash.similarity_threshold",
                "must be between 0.0 and 1.0",
                simhash.similarity_threshold,
            )
        )


def _validate_logging(
    config: CoralConfig,
    errors: list[ValidationError],
    warnings: list[ValidationError],
) -> None:
    """Validate logging configuration."""
    logging_cfg = config.logging

    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if logging_cfg.level.upper() not in valid_levels:
        errors.append(
            ValidationError(
                "logging.level",
                f"must be one of {valid_levels}",
                logging_cfg.level,
            )
        )


def _validate_remotes(
    config: CoralConfig,
    errors: list[ValidationError],
    warnings: list[ValidationError],
) -> None:
    """Validate remote configurations."""
    for name, remote in config.remotes.items():
        # URL is required
        if not remote.url:
            errors.append(
                ValidationError(
                    f"remotes.{name}.url",
                    "URL is required",
                )
            )

        # Backend type
        valid_backends = {"s3", "gcs", "azure", "file"}
        if remote.backend not in valid_backends:
            errors.append(
                ValidationError(
                    f"remotes.{name}.backend",
                    f"must be one of {valid_backends}",
                    remote.backend,
                )
            )

        # S3 URL format
        if remote.backend == "s3" and remote.url and not remote.url.startswith("s3://"):
            warnings.append(
                ValidationError(
                    f"remotes.{name}.url",
                    "S3 URLs should start with 's3://'",
                    remote.url,
                )
            )


def validate_value(key: str, value: Any) -> Optional[ValidationError]:
    """Validate a single configuration value.

    Args:
        key: Configuration key (dot notation)
        value: Value to validate

    Returns:
        ValidationError if invalid, None if valid
    """
    # Define validators for specific keys
    validators = {
        "core.similarity_threshold": lambda v: (
            None
            if 0.0 <= v <= 1.0
            else ValidationError(key, "must be between 0.0 and 1.0", v)
        ),
        "core.compression_level": lambda v: (
            None if 1 <= v <= 9 else ValidationError(key, "must be between 1 and 9", v)
        ),
        "core.compression": lambda v: (
            None
            if v in {"gzip", "lzf", "none"}
            else ValidationError(key, "must be gzip, lzf, or none", v)
        ),
        "delta.quantization_bits": lambda v: (
            None if v in (8, 16) else ValidationError(key, "must be 8 or 16", v)
        ),
        "simhash.num_bits": lambda v: (
            None if v in (64, 128) else ValidationError(key, "must be 64 or 128", v)
        ),
        "logging.level": lambda v: (
            None
            if v.upper() in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            else ValidationError(key, "must be a valid log level", v)
        ),
    }

    if key in validators:
        return validators[key](value)

    return None
