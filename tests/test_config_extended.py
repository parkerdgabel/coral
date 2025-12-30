"""Extended tests for configuration module.

This module tests configuration loading, validation, and schema functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch


class TestConfigSchema:
    """Test configuration schema classes."""

    def test_core_config_defaults(self):
        """Test CoreConfig default values."""
        from coral.config.schema import CoreConfig

        config = CoreConfig()

        assert config.compression == "gzip"
        assert config.similarity_threshold == 0.98
        assert config.delta_encoding is True
        assert config.enable_lsh is False

    def test_core_config_custom(self):
        """Test CoreConfig custom values."""
        from coral.config.schema import CoreConfig
        from coral.delta.delta_encoder import DeltaType

        config = CoreConfig(
            compression="lzf",
            similarity_threshold=0.95,
            delta_encoding=False,
            delta_type=DeltaType.XOR_FLOAT32,
        )

        assert config.compression == "lzf"
        assert config.similarity_threshold == 0.95
        assert config.delta_encoding is False
        assert config.delta_type == DeltaType.XOR_FLOAT32

    def test_user_config_defaults(self):
        """Test UserConfig default values."""
        from coral.config.schema import UserConfig

        config = UserConfig()

        assert config.name == "Anonymous"
        assert config.email == "anonymous@example.com"

    def test_user_config_custom(self):
        """Test UserConfig custom values."""
        from coral.config.schema import UserConfig

        config = UserConfig(
            name="John Doe",
            email="john@example.com",
        )

        assert config.name == "John Doe"
        assert config.email == "john@example.com"

    def test_storage_config_defaults(self):
        """Test StorageConfig default values."""
        from coral.config.schema import StorageConfig

        config = StorageConfig()

        assert config.compression == "gzip"
        assert config.compression_level == 4
        assert config.mode == "a"

    def test_lsh_config_defaults(self):
        """Test LSHConfig default values."""
        from coral.config.schema import LSHConfig

        config = LSHConfig()

        assert config.num_tables == 4
        assert config.num_hyperplanes == 8
        assert config.max_candidates == 100

    def test_simhash_config_defaults(self):
        """Test SimHashConfig default values."""
        from coral.config.schema import SimHashConfig

        config = SimHashConfig()

        assert config.num_bits == 64
        assert config.similarity_threshold == 0.1
        assert config.seed == 42

    def test_checkpoint_defaults(self):
        """Test CheckpointDefaults default values."""
        from coral.config.schema import CheckpointDefaults

        config = CheckpointDefaults()

        assert config.save_optimizer_state is True
        assert config.save_scheduler_state is True
        assert config.auto_commit is True
        assert config.minimize_metric is True

    def test_logging_config_defaults(self):
        """Test LoggingConfig default values."""
        from coral.config.schema import LoggingConfig

        config = LoggingConfig()

        assert config.level in ("INFO", "WARNING", "DEBUG")  # Some valid level
        assert config.file is None

    def test_delta_encoding_config_defaults(self):
        """Test DeltaEncodingConfig default values."""
        from coral.config.schema import DeltaEncodingConfig

        config = DeltaEncodingConfig()

        assert config.sparse_threshold == 1e-6
        assert config.quantization_bits == 8
        assert config.max_delta_ratio == 1.0

    def test_coral_config_creation(self):
        """Test CoralConfig creation."""
        from coral.config.schema import (
            CheckpointDefaults,
            CoralConfig,
            CoreConfig,
            LoggingConfig,
            LSHConfig,
            SimHashConfig,
            StorageConfig,
            UserConfig,
        )

        config = CoralConfig(
            user=UserConfig(name="Test"),
            core=CoreConfig(compression="lzf"),
            storage=StorageConfig(),
            lsh=LSHConfig(num_tables=5),
            simhash=SimHashConfig(num_bits=64),
            checkpoint=CheckpointDefaults(),
            logging=LoggingConfig(level="DEBUG"),
        )

        assert config.user.name == "Test"
        assert config.core.compression == "lzf"
        assert config.lsh.num_tables == 5
        assert config.simhash.num_bits == 64
        assert config.logging.level == "DEBUG"

    def test_coral_config_to_dict(self):
        """Test CoralConfig to_dict method."""
        from coral.config.schema import CoralConfig, CoreConfig, UserConfig

        config = CoralConfig(
            user=UserConfig(name="Test"),
            core=CoreConfig(similarity_threshold=0.9),
        )

        config_dict = config.to_dict()

        assert config_dict["user"]["name"] == "Test"
        assert config_dict["core"]["similarity_threshold"] == 0.9

    def test_coral_config_from_dict(self):
        """Test CoralConfig from_dict method."""
        from coral.config.schema import CoralConfig

        config_dict = {
            "user": {"name": "Test", "email": "test@example.com"},
            "core": {"compression": "lzf", "similarity_threshold": 0.95},
        }

        config = CoralConfig.from_dict(config_dict)

        assert config.user.name == "Test"
        assert config.core.compression == "lzf"


class TestConfigLoader:
    """Test ConfigLoader functionality."""

    def test_load_empty_config(self):
        """Test loading with no config files."""
        from coral.config.loader import ConfigLoader

        loader = ConfigLoader()
        config = loader.load()

        # Should return default config
        assert config.user.name == "Anonymous"
        assert config.core.compression == "gzip"

    def test_load_from_repo_toml(self):
        """Test loading from repo coral.toml file."""
        from coral.config.loader import ConfigLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            coral_dir = repo_path / ".coral"
            coral_dir.mkdir()

            # Create coral.toml
            toml_content = """
[user]
name = "Repo User"
email = "repo@example.com"

[core]
compression = "lzf"
similarity_threshold = 0.9
"""
            (coral_dir / "coral.toml").write_text(toml_content)

            loader = ConfigLoader(repo_path=repo_path)
            config = loader.load()

            assert config.user.name == "Repo User"
            assert config.core.compression == "lzf"
            assert config.core.similarity_threshold == 0.9

    def test_environment_variable_override(self):
        """Test environment variable override."""
        from coral.config.loader import ConfigLoader

        with patch.dict(
            "os.environ",
            {
                "CORAL_USER_NAME": "Env User",
                "CORAL_CORE_SIMILARITY_THRESHOLD": "0.85",
            },
        ):
            loader = ConfigLoader()
            config = loader.load()

            assert config.user.name == "Env User"
            assert config.core.similarity_threshold == 0.85


class TestConfigValidation:
    """Test configuration validation."""

    def test_validate_valid_config(self):
        """Test validating a valid configuration."""
        from coral.config.schema import CoralConfig, CoreConfig, UserConfig
        from coral.config.validation import validate_config

        config = CoralConfig(
            user=UserConfig(name="Test", email="test@example.com"),
            core=CoreConfig(
                compression="gzip",
                similarity_threshold=0.98,
            ),
        )

        result = validate_config(config)

        assert result.valid is True
        assert len(result.errors) == 0


class TestGetDefaultConfig:
    """Test get_default_config function."""

    def test_get_default_config(self):
        """Test getting default configuration."""
        from coral.config import get_default_config

        config = get_default_config()

        assert config.user.name == "Anonymous"
        assert config.core.compression == "gzip"
        assert config.core.similarity_threshold == 0.98


class TestLoadConfig:
    """Test load_config function."""

    def test_load_config_default(self):
        """Test load_config returns valid config."""
        from coral.config import load_config

        config = load_config()

        assert config is not None
        assert hasattr(config, "user")
        assert hasattr(config, "core")

    def test_load_config_with_repo_path(self):
        """Test load_config with repo path."""
        from coral.config import load_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config(repo_path=Path(tmpdir))

            assert config is not None


class TestConfigMerge:
    """Test config merging functionality."""

    def test_merge_configs(self):
        """Test merging two configs."""
        from coral.config.schema import CoralConfig, CoreConfig, UserConfig

        base = CoralConfig(
            user=UserConfig(name="Base", email="base@example.com"),
            core=CoreConfig(compression="gzip"),
        )

        override = CoralConfig(
            user=UserConfig(name="Override"),
            core=CoreConfig(similarity_threshold=0.9),
        )

        # Manual merge
        merged = CoralConfig(
            user=UserConfig(
                name=override.user.name or base.user.name,
                email=override.user.email or base.user.email,
            ),
            core=CoreConfig(
                compression=base.core.compression,
                similarity_threshold=override.core.similarity_threshold,
            ),
        )

        assert merged.user.name == "Override"
        assert merged.user.email == "anonymous@example.com"  # Default from override
        assert merged.core.compression == "gzip"
        assert merged.core.similarity_threshold == 0.9
