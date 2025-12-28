"""Configuration schema dataclasses for Coral.

This module defines all configuration options as typed dataclasses,
providing a single source of truth for default values and types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional


class CompressionType(str, Enum):
    """Compression algorithms for weight storage."""

    GZIP = "gzip"
    LZF = "lzf"
    NONE = "none"


class DeltaTypeEnum(str, Enum):
    """Delta encoding strategies."""

    # Lossless strategies
    FLOAT32_RAW = "float32_raw"
    COMPRESSED = "compressed"
    XOR_FLOAT32 = "xor_float32"
    XOR_BFLOAT16 = "xor_bfloat16"
    EXPONENT_MANTISSA = "exponent_mantissa"

    # Lossy strategies
    INT8_QUANTIZED = "int8_quantized"
    INT16_QUANTIZED = "int16_quantized"
    SPARSE = "sparse"
    PER_AXIS_SCALED = "per_axis_scaled"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class UserConfig:
    """User identity configuration."""

    name: str = "Anonymous"
    email: str = "anonymous@example.com"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"name": self.name, "email": self.email}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserConfig:
        """Create from dictionary."""
        return cls(
            name=data.get("name", cls.name),
            email=data.get("email", cls.email),
        )


@dataclass
class CoreConfig:
    """Core repository settings."""

    # Compression settings
    compression: str = "gzip"
    compression_level: int = 4

    # Similarity and deduplication
    similarity_threshold: float = 0.98
    magnitude_tolerance: float = 0.1
    enable_lsh: bool = False

    # Delta encoding
    delta_encoding: bool = True
    delta_type: str = "compressed"
    strict_reconstruction: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.magnitude_tolerance <= 1.0:
            raise ValueError("magnitude_tolerance must be between 0.0 and 1.0")
        if not 1 <= self.compression_level <= 9:
            raise ValueError("compression_level must be between 1 and 9")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "compression": self.compression,
            "compression_level": self.compression_level,
            "similarity_threshold": self.similarity_threshold,
            "magnitude_tolerance": self.magnitude_tolerance,
            "enable_lsh": self.enable_lsh,
            "delta_encoding": self.delta_encoding,
            "delta_type": self.delta_type,
            "strict_reconstruction": self.strict_reconstruction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoreConfig:
        """Create from dictionary."""
        return cls(
            compression=data.get("compression", "gzip"),
            compression_level=data.get("compression_level", 4),
            similarity_threshold=data.get("similarity_threshold", 0.98),
            magnitude_tolerance=data.get("magnitude_tolerance", 0.1),
            enable_lsh=data.get("enable_lsh", False),
            delta_encoding=data.get("delta_encoding", True),
            delta_type=data.get("delta_type", "compressed"),
            strict_reconstruction=data.get("strict_reconstruction", False),
        )


@dataclass
class DeltaEncodingConfig:
    """Delta encoding settings."""

    # Sparse encoding threshold
    sparse_threshold: float = 1e-6

    # Quantization settings
    quantization_bits: Literal[8, 16] = 8

    # Size thresholds
    min_weight_size: int = 512  # bytes
    max_delta_ratio: float = 1.0  # Max delta size relative to original
    min_compression_ratio: float = 0.0  # Minimum compression ratio (0% = always store)

    # Overhead estimates (bytes)
    object_overhead: int = 200
    metadata_overhead: int = 200

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.sparse_threshold < 0:
            raise ValueError("sparse_threshold must be non-negative")
        if self.quantization_bits not in (8, 16):
            raise ValueError("quantization_bits must be 8 or 16")
        if self.min_weight_size < 0:
            raise ValueError("min_weight_size must be non-negative")
        if not 0.0 <= self.max_delta_ratio <= 2.0:
            raise ValueError("max_delta_ratio must be between 0.0 and 2.0")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sparse_threshold": self.sparse_threshold,
            "quantization_bits": self.quantization_bits,
            "min_weight_size": self.min_weight_size,
            "max_delta_ratio": self.max_delta_ratio,
            "min_compression_ratio": self.min_compression_ratio,
            "object_overhead": self.object_overhead,
            "metadata_overhead": self.metadata_overhead,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeltaEncodingConfig:
        """Create from dictionary."""
        return cls(
            sparse_threshold=data.get("sparse_threshold", 1e-6),
            quantization_bits=data.get("quantization_bits", 8),
            min_weight_size=data.get("min_weight_size", 512),
            max_delta_ratio=data.get("max_delta_ratio", 1.0),
            min_compression_ratio=data.get("min_compression_ratio", 0.0),
            object_overhead=data.get("object_overhead", 200),
            metadata_overhead=data.get("metadata_overhead", 200),
        )


@dataclass
class StorageConfig:
    """Local storage settings."""

    compression: str = "gzip"
    compression_level: int = 4
    mode: str = "a"  # File mode for HDF5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "compression": self.compression,
            "compression_level": self.compression_level,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StorageConfig:
        """Create from dictionary."""
        return cls(
            compression=data.get("compression", "gzip"),
            compression_level=data.get("compression_level", 4),
            mode=data.get("mode", "a"),
        )


@dataclass
class S3StorageConfig:
    """S3/MinIO storage settings."""

    max_concurrency: int = 10
    chunk_size: int = 8 * 1024 * 1024  # 8MB
    default_region: str = "us-east-1"
    default_prefix: str = "coral/"

    # Credentials (prefer environment variables for secrets)
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    endpoint_url: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if self.chunk_size < 1024:
            raise ValueError("chunk_size must be at least 1024 bytes")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes secrets)."""
        return {
            "max_concurrency": self.max_concurrency,
            "chunk_size": self.chunk_size,
            "default_region": self.default_region,
            "default_prefix": self.default_prefix,
            "endpoint_url": self.endpoint_url,
            # Intentionally exclude access_key and secret_key
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> S3StorageConfig:
        """Create from dictionary."""
        return cls(
            max_concurrency=data.get("max_concurrency", 10),
            chunk_size=data.get("chunk_size", 8 * 1024 * 1024),
            default_region=data.get("default_region", "us-east-1"),
            default_prefix=data.get("default_prefix", "coral/"),
            access_key=data.get("access_key"),
            secret_key=data.get("secret_key"),
            endpoint_url=data.get("endpoint_url"),
        )


@dataclass
class LSHConfig:
    """Locality-Sensitive Hashing settings for O(1) similarity lookup."""

    num_hyperplanes: int = 8
    num_tables: int = 4
    max_candidates: int = 100
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.num_hyperplanes < 1:
            raise ValueError("num_hyperplanes must be at least 1")
        if self.num_tables < 1:
            raise ValueError("num_tables must be at least 1")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_hyperplanes": self.num_hyperplanes,
            "num_tables": self.num_tables,
            "max_candidates": self.max_candidates,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LSHConfig:
        """Create from dictionary."""
        return cls(
            num_hyperplanes=data.get("num_hyperplanes", 8),
            num_tables=data.get("num_tables", 4),
            max_candidates=data.get("max_candidates", 100),
            seed=data.get("seed", 42),
        )


@dataclass
class SimHashConfig:
    """SimHash fingerprinting settings."""

    num_bits: Literal[64, 128] = 64
    similarity_threshold: float = 0.1
    seed: int = 42
    num_hyperplanes: Optional[int] = None  # Defaults to num_bits

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.num_bits not in (64, 128):
            raise ValueError("num_bits must be 64 or 128")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_bits": self.num_bits,
            "similarity_threshold": self.similarity_threshold,
            "seed": self.seed,
            "num_hyperplanes": self.num_hyperplanes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimHashConfig:
        """Create from dictionary."""
        return cls(
            num_bits=data.get("num_bits", 64),
            similarity_threshold=data.get("similarity_threshold", 0.1),
            seed=data.get("seed", 42),
            num_hyperplanes=data.get("num_hyperplanes"),
        )


@dataclass
class CheckpointDefaults:
    """Default checkpoint configuration."""

    save_optimizer_state: bool = True
    save_scheduler_state: bool = True
    save_random_state: bool = True
    use_incremental_saves: bool = True
    auto_commit: bool = True
    tag_best_checkpoints: bool = True
    commit_message_template: str = "Checkpoint at epoch {epoch}, step {step}"
    minimize_metric: bool = True  # True for loss, False for accuracy

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "save_optimizer_state": self.save_optimizer_state,
            "save_scheduler_state": self.save_scheduler_state,
            "save_random_state": self.save_random_state,
            "use_incremental_saves": self.use_incremental_saves,
            "auto_commit": self.auto_commit,
            "tag_best_checkpoints": self.tag_best_checkpoints,
            "commit_message_template": self.commit_message_template,
            "minimize_metric": self.minimize_metric,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointDefaults:
        """Create from dictionary."""
        return cls(
            save_optimizer_state=data.get("save_optimizer_state", True),
            save_scheduler_state=data.get("save_scheduler_state", True),
            save_random_state=data.get("save_random_state", True),
            use_incremental_saves=data.get("use_incremental_saves", True),
            auto_commit=data.get("auto_commit", True),
            tag_best_checkpoints=data.get("tag_best_checkpoints", True),
            commit_message_template=data.get(
                "commit_message_template", "Checkpoint at epoch {epoch}, step {step}"
            ),
            minimize_metric=data.get("minimize_metric", True),
        )


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(levelname)s - %(name)s - %(message)s"
    file: Optional[str] = None
    console: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level,
            "format": self.format,
            "file": self.file,
            "console": self.console,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoggingConfig:
        """Create from dictionary."""
        return cls(
            level=data.get("level", "INFO"),
            format=data.get("format", "%(levelname)s - %(name)s - %(message)s"),
            file=data.get("file"),
            console=data.get("console", True),
        )


@dataclass
class HuggingFaceConfig:
    """HuggingFace Hub integration settings."""

    cache_dir: str = "~/.coral/hub"
    similarity_threshold: float = 0.95
    token: Optional[str] = None  # Prefer HF_TOKEN env var

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes token)."""
        return {
            "cache_dir": self.cache_dir,
            "similarity_threshold": self.similarity_threshold,
            # Intentionally exclude token
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HuggingFaceConfig:
        """Create from dictionary."""
        return cls(
            cache_dir=data.get("cache_dir", "~/.coral/hub"),
            similarity_threshold=data.get("similarity_threshold", 0.95),
            token=data.get("token"),
        )


@dataclass
class RemoteConfig:
    """Remote repository configuration."""

    name: str
    url: str
    backend: str = "s3"
    auto_push: bool = False
    auto_pull: bool = False
    endpoint_url: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    region: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes secrets)."""
        result = {
            "url": self.url,
            "backend": self.backend,
            "auto_push": self.auto_push,
            "auto_pull": self.auto_pull,
        }
        if self.endpoint_url:
            result["endpoint_url"] = self.endpoint_url
        if self.region:
            result["region"] = self.region
        return result

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> RemoteConfig:
        """Create from dictionary."""
        return cls(
            name=name,
            url=data.get("url", ""),
            backend=data.get("backend", "s3"),
            auto_push=data.get("auto_push", False),
            auto_pull=data.get("auto_pull", False),
            endpoint_url=data.get("endpoint_url"),
            access_key=data.get("access_key"),
            secret_key=data.get("secret_key"),
            region=data.get("region"),
        )


@dataclass
class CoralConfig:
    """Main configuration container for Coral.

    This is the top-level configuration class that contains all settings.
    Configuration is loaded from multiple sources with the following priority:
    1. Programmatic (highest) - Direct API calls
    2. Environment Variables - CORAL_* prefixed
    3. Repository Config - .coral/coral.toml
    4. User Config - ~/.config/coral/config.toml
    5. Defaults (lowest) - Built-in defaults
    """

    user: UserConfig = field(default_factory=UserConfig)
    core: CoreConfig = field(default_factory=CoreConfig)
    delta: DeltaEncodingConfig = field(default_factory=DeltaEncodingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    s3: S3StorageConfig = field(default_factory=S3StorageConfig)
    lsh: LSHConfig = field(default_factory=LSHConfig)
    simhash: SimHashConfig = field(default_factory=SimHashConfig)
    checkpoint: CheckpointDefaults = field(default_factory=CheckpointDefaults)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    remotes: dict[str, RemoteConfig] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {
            "user": self.user.to_dict(),
            "core": self.core.to_dict(),
            "delta": self.delta.to_dict(),
            "storage": self.storage.to_dict(),
            "s3": self.s3.to_dict(),
            "lsh": self.lsh.to_dict(),
            "simhash": self.simhash.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "logging": self.logging.to_dict(),
            "huggingface": self.huggingface.to_dict(),
        }
        if self.remotes:
            result["remotes"] = {
                name: remote.to_dict() for name, remote in self.remotes.items()
            }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoralConfig:
        """Create configuration from dictionary."""
        remotes = {}
        if "remotes" in data:
            for name, remote_data in data["remotes"].items():
                remotes[name] = RemoteConfig.from_dict(name, remote_data)

        return cls(
            user=UserConfig.from_dict(data.get("user", {})),
            core=CoreConfig.from_dict(data.get("core", {})),
            delta=DeltaEncodingConfig.from_dict(data.get("delta", {})),
            storage=StorageConfig.from_dict(data.get("storage", {})),
            s3=S3StorageConfig.from_dict(data.get("s3", {})),
            lsh=LSHConfig.from_dict(data.get("lsh", {})),
            simhash=SimHashConfig.from_dict(data.get("simhash", {})),
            checkpoint=CheckpointDefaults.from_dict(data.get("checkpoint", {})),
            logging=LoggingConfig.from_dict(data.get("logging", {})),
            huggingface=HuggingFaceConfig.from_dict(data.get("huggingface", {})),
            remotes=remotes,
        )

    def get_nested(self, key: str, default: Any = None) -> Any:
        """Get a nested configuration value using dot notation.

        Args:
            key: Dot-separated key path (e.g., "core.similarity_threshold")
            default: Default value if key not found

        Returns:
            The configuration value or default
        """
        parts = key.split(".")
        obj: Any = self
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return default
        return obj

    def set_nested(self, key: str, value: Any) -> None:
        """Set a nested configuration value using dot notation.

        Args:
            key: Dot-separated key path (e.g., "core.similarity_threshold")
            value: Value to set
        """
        parts = key.split(".")
        obj: Any = self
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                raise KeyError(f"Invalid configuration key: {key}")
        setattr(obj, parts[-1], value)
