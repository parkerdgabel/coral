"""Configuration loader for Coral.

This module handles loading configuration from multiple sources:
1. Default values (lowest priority)
2. User config file (~/.config/coral/config.toml)
3. Repository config file (.coral/coral.toml)
4. Environment variables (CORAL_* prefix)
5. Programmatic overrides (highest priority)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from .schema import (
    CoralConfig,
)

# Use tomllib (3.11+) or tomli for older Python versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Default paths
USER_CONFIG_DIR = Path.home() / ".config" / "coral"
USER_CONFIG_PATH = USER_CONFIG_DIR / "config.toml"
REPO_CONFIG_NAME = "coral.toml"
LEGACY_CONFIG_NAME = "config.json"
ENV_PREFIX = "CORAL_"


def _parse_env_value(value: str) -> Any:
    """Parse an environment variable value to appropriate Python type."""
    # Handle booleans
    if value.lower() in ("true", "yes", "1", "on"):
        return True
    if value.lower() in ("false", "no", "0", "off"):
        return False

    # Handle None
    if value.lower() in ("none", "null", ""):
        return None

    # Try numeric types
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    # Return as string
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class ConfigLoader:
    """Load configuration from multiple sources with priority handling."""

    def __init__(
        self,
        repo_path: Optional[Path] = None,
        user_config_path: Optional[Path] = None,
    ):
        """Initialize the configuration loader.

        Args:
            repo_path: Path to the repository root (contains .coral/)
            user_config_path: Optional override for user config path
        """
        self.repo_path = Path(repo_path) if repo_path else None
        self.user_config_path = user_config_path or USER_CONFIG_PATH

    def load(self) -> CoralConfig:
        """Load configuration from all sources with priority handling.

        Priority (highest to lowest):
        1. Environment variables (CORAL_*)
        2. Repository config (.coral/coral.toml)
        3. User config (~/.config/coral/config.toml)
        4. Default values

        Returns:
            Merged CoralConfig instance
        """
        # Start with defaults
        config_dict: dict[str, Any] = {}

        # Load user config
        if self.user_config_path.exists():
            user_data = self._load_toml(self.user_config_path)
            if user_data:
                config_dict = _deep_merge(config_dict, user_data)
                logger.debug(f"Loaded user config from {self.user_config_path}")

        # Load repository config
        if self.repo_path:
            # Try new TOML format first
            repo_config_path = self.repo_path / ".coral" / REPO_CONFIG_NAME
            if repo_config_path.exists():
                repo_data = self._load_toml(repo_config_path)
                if repo_data:
                    config_dict = _deep_merge(config_dict, repo_data)
                    logger.debug(f"Loaded repo config from {repo_config_path}")
            else:
                # Fall back to legacy JSON config
                legacy_path = self.repo_path / ".coral" / LEGACY_CONFIG_NAME
                if legacy_path.exists():
                    legacy_data = self._load_legacy_json(legacy_path)
                    if legacy_data:
                        config_dict = _deep_merge(config_dict, legacy_data)
                        logger.debug(f"Loaded legacy config from {legacy_path}")

        # Apply environment variable overrides
        env_overrides = self._load_env_vars()
        if env_overrides:
            config_dict = _deep_merge(config_dict, env_overrides)
            logger.debug("Applied environment variable overrides")

        # Create config instance
        return CoralConfig.from_dict(config_dict)

    def _load_toml(self, path: Path) -> Optional[dict[str, Any]]:
        """Load a TOML configuration file.

        Args:
            path: Path to the TOML file

        Returns:
            Parsed configuration dict or None if failed
        """
        if tomllib is None:
            logger.warning(
                "TOML support requires 'tomli' package for Python < 3.11. "
                "Install with: pip install tomli"
            )
            return None

        try:
            with open(path, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            logger.warning(f"Failed to load TOML config from {path}: {e}")
            return None

    def _load_legacy_json(self, path: Path) -> Optional[dict[str, Any]]:
        """Load a legacy JSON configuration file and convert to new format.

        Args:
            path: Path to the JSON file

        Returns:
            Converted configuration dict or None if failed
        """
        try:
            with open(path) as f:
                data = json.load(f)

            # Convert legacy format to new format
            return self._migrate_legacy_config(data)
        except Exception as e:
            logger.warning(f"Failed to load legacy config from {path}: {e}")
            return None

    def _migrate_legacy_config(self, legacy: dict[str, Any]) -> dict[str, Any]:
        """Migrate legacy config.json format to new format.

        Args:
            legacy: Legacy configuration dictionary

        Returns:
            Migrated configuration dictionary
        """
        result: dict[str, Any] = {}

        # User section stays the same
        if "user" in legacy:
            result["user"] = legacy["user"]

        # Core section - map old keys to new
        if "core" in legacy:
            core = legacy["core"]
            result["core"] = {
                "compression": core.get("compression", "gzip"),
                "similarity_threshold": core.get("similarity_threshold", 0.98),
                "delta_encoding": core.get("delta_encoding", True),
                "delta_type": core.get("delta_type", "compressed"),
            }

        return result

    def _load_env_vars(self) -> dict[str, Any]:
        """Load configuration from environment variables.

        Environment variables are prefixed with CORAL_ and use underscores
        to separate nested keys. For example:
        - CORAL_CORE_SIMILARITY_THRESHOLD -> core.similarity_threshold
        - CORAL_USER_NAME -> user.name

        Returns:
            Configuration dictionary from environment variables
        """
        result: dict[str, Any] = {}

        for key, value in os.environ.items():
            if not key.startswith(ENV_PREFIX):
                continue

            # Remove prefix and convert to lowercase
            config_key = key[len(ENV_PREFIX) :].lower()

            # Split into nested keys
            parts = config_key.split("_")

            # Handle special cases for multi-word keys
            # e.g., CORAL_CORE_COMPRESSION_LEVEL -> core.compression_level
            nested = self._build_nested_dict(parts, _parse_env_value(value))
            result = _deep_merge(result, nested)

        return result

    def _build_nested_dict(self, parts: list[str], value: Any) -> dict[str, Any]:
        """Build a nested dictionary from key parts.

        This handles the complexity of environment variable naming where
        underscores can be part of the key name or separators.

        Args:
            parts: List of key parts from splitting on underscores
            value: Value to set

        Returns:
            Nested dictionary
        """
        # Known section names (first level)
        sections = {
            "user",
            "core",
            "delta",
            "storage",
            "s3",
            "lsh",
            "simhash",
            "checkpoint",
            "logging",
            "huggingface",
            "remotes",
        }

        if not parts:
            return {}

        # Check if first part is a known section
        if parts[0] in sections:
            section = parts[0]
            remaining = parts[1:]

            if not remaining:
                return {section: value}

            # Join remaining parts with underscore as the key name
            key = "_".join(remaining)
            return {section: {key: value}}
        else:
            # Unknown section, join all parts
            key = "_".join(parts)
            return {key: value}

    def save_repo_config(self, config: CoralConfig) -> None:
        """Save configuration to the repository config file.

        Args:
            config: Configuration to save
        """
        if not self.repo_path:
            raise ValueError("No repository path set")

        config_path = self.repo_path / ".coral" / REPO_CONFIG_NAME
        config_path.parent.mkdir(parents=True, exist_ok=True)

        self._save_toml(config_path, config.to_dict())
        logger.info(f"Saved repository config to {config_path}")

    def save_user_config(self, config: CoralConfig) -> None:
        """Save configuration to the user config file.

        Args:
            config: Configuration to save
        """
        self.user_config_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_toml(self.user_config_path, config.to_dict())
        logger.info(f"Saved user config to {self.user_config_path}")

    def _save_toml(self, path: Path, data: dict[str, Any]) -> None:
        """Save configuration as TOML.

        Args:
            path: Path to save to
            data: Configuration data
        """
        # Filter out None values since TOML doesn't support them
        filtered_data = self._filter_none_values(data)

        try:
            # Use tomli_w if available, otherwise fallback to manual formatting
            try:
                import tomli_w

                with open(path, "wb") as f:
                    tomli_w.dump(filtered_data, f)
            except ImportError:
                # Manual TOML formatting fallback
                with open(path, "w") as f:
                    self._write_toml_manual(f, filtered_data)
        except Exception as e:
            logger.error(f"Failed to save config to {path}: {e}")
            raise

    def _filter_none_values(self, data: dict[str, Any]) -> dict[str, Any]:
        """Recursively filter out None values from a dictionary.

        TOML doesn't support None/null values, so we remove them.

        Args:
            data: Dictionary to filter

        Returns:
            Filtered dictionary without None values
        """
        result = {}
        for key, value in data.items():
            if value is None:
                continue
            elif isinstance(value, dict):
                filtered = self._filter_none_values(value)
                if filtered:  # Only add non-empty dicts
                    result[key] = filtered
            else:
                result[key] = value
        return result

    def _write_toml_manual(
        self, f: Any, data: dict[str, Any], prefix: str = ""
    ) -> None:
        """Manually write TOML format (fallback when tomli_w is not available).

        Args:
            f: File handle to write to
            data: Configuration data
            prefix: Current section prefix
        """
        # Write scalar values first
        for key, value in data.items():
            if not isinstance(value, dict):
                f.write(f"{key} = {self._toml_value(value)}\n")

        # Then write nested sections
        for key, value in data.items():
            if isinstance(value, dict):
                section = f"{prefix}{key}" if prefix else key
                f.write(f"\n[{section}]\n")
                self._write_toml_manual(f, value, f"{section}.")

    def _toml_value(self, value: Any) -> str:
        """Convert a Python value to TOML format."""
        if isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, str):
            # Escape and quote strings
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        elif isinstance(value, (int, float)):
            return str(value)
        elif value is None:
            return '""'  # TOML doesn't have null, use empty string
        elif isinstance(value, list):
            items = ", ".join(self._toml_value(v) for v in value)
            return f"[{items}]"
        else:
            return f'"{value}"'


def load_config(
    repo_path: Optional[Path] = None,
    user_config_path: Optional[Path] = None,
) -> CoralConfig:
    """Load Coral configuration from all sources.

    This is the main entry point for loading configuration.

    Args:
        repo_path: Optional path to repository root
        user_config_path: Optional override for user config path

    Returns:
        Merged CoralConfig instance
    """
    loader = ConfigLoader(repo_path, user_config_path)
    return loader.load()


def get_default_config() -> CoralConfig:
    """Get a CoralConfig with all default values.

    Returns:
        CoralConfig with defaults
    """
    return CoralConfig()
