"""Tests for the configuration system."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from coral.config import (
    CheckpointDefaults,
    ConfigLoader,
    CoralConfig,
    CoreConfig,
    DeltaEncodingConfig,
    LSHConfig,
    RemoteConfig,
    S3StorageConfig,
    SimHashConfig,
    StorageConfig,
    UserConfig,
    ValidationError,
    ValidationResult,
    get_default_config,
    load_config,
    validate_config,
    validate_value,
)
from coral.delta.delta_encoder import DeltaType


class TestUserConfig:
    """Tests for UserConfig dataclass."""

    def test_default_values(self):
        config = UserConfig()
        assert config.name == "Anonymous"
        assert config.email == "anonymous@example.com"

    def test_custom_values(self):
        config = UserConfig(name="Test User", email="test@example.com")
        assert config.name == "Test User"
        assert config.email == "test@example.com"

    def test_to_dict(self):
        config = UserConfig(name="Test", email="test@test.com")
        d = config.to_dict()
        assert d == {"name": "Test", "email": "test@test.com"}

    def test_from_dict(self):
        data = {"name": "From Dict", "email": "dict@test.com"}
        config = UserConfig.from_dict(data)
        assert config.name == "From Dict"
        assert config.email == "dict@test.com"

    def test_from_dict_with_defaults(self):
        config = UserConfig.from_dict({})
        assert config.name == "Anonymous"
        assert config.email == "anonymous@example.com"


class TestCoreConfig:
    """Tests for CoreConfig dataclass."""

    def test_default_values(self):
        config = CoreConfig()
        assert config.compression == "gzip"
        assert config.compression_level == 4
        assert config.similarity_threshold == 0.98
        assert config.magnitude_tolerance == 0.1
        assert config.enable_lsh is False
        assert config.delta_encoding is True
        assert config.delta_type == DeltaType.COMPRESSED
        assert config.strict_reconstruction is False

    def test_validation_similarity_threshold(self):
        with pytest.raises(ValueError, match="similarity_threshold"):
            CoreConfig(similarity_threshold=1.5)

        with pytest.raises(ValueError, match="similarity_threshold"):
            CoreConfig(similarity_threshold=-0.1)

    def test_validation_magnitude_tolerance(self):
        with pytest.raises(ValueError, match="magnitude_tolerance"):
            CoreConfig(magnitude_tolerance=2.0)

    def test_validation_compression_level(self):
        with pytest.raises(ValueError, match="compression_level"):
            CoreConfig(compression_level=0)

        with pytest.raises(ValueError, match="compression_level"):
            CoreConfig(compression_level=10)

    def test_to_dict(self):
        config = CoreConfig(similarity_threshold=0.95)
        d = config.to_dict()
        assert d["similarity_threshold"] == 0.95
        assert "compression" in d

    def test_from_dict(self):
        data = {"similarity_threshold": 0.90, "delta_type": "xor_float32"}
        config = CoreConfig.from_dict(data)
        assert config.similarity_threshold == 0.90
        assert config.delta_type == DeltaType.XOR_FLOAT32


class TestDeltaEncodingConfig:
    """Tests for DeltaEncodingConfig dataclass."""

    def test_default_values(self):
        config = DeltaEncodingConfig()
        assert config.sparse_threshold == 1e-6
        assert config.quantization_bits == 8
        assert config.min_weight_size == 512
        assert config.max_delta_ratio == 1.0

    def test_validation_sparse_threshold(self):
        with pytest.raises(ValueError, match="sparse_threshold"):
            DeltaEncodingConfig(sparse_threshold=-1)

    def test_validation_quantization_bits(self):
        with pytest.raises(ValueError, match="quantization_bits"):
            DeltaEncodingConfig(quantization_bits=4)

    def test_validation_min_weight_size(self):
        with pytest.raises(ValueError, match="min_weight_size"):
            DeltaEncodingConfig(min_weight_size=-100)

    def test_validation_max_delta_ratio(self):
        with pytest.raises(ValueError, match="max_delta_ratio"):
            DeltaEncodingConfig(max_delta_ratio=3.0)


class TestStorageConfig:
    """Tests for StorageConfig dataclass."""

    def test_default_values(self):
        config = StorageConfig()
        assert config.compression == "gzip"
        assert config.compression_level == 4
        assert config.mode == "a"

    def test_to_dict_from_dict(self):
        config = StorageConfig(compression="lzf", compression_level=6)
        d = config.to_dict()
        restored = StorageConfig.from_dict(d)
        assert restored.compression == "lzf"
        assert restored.compression_level == 6


class TestS3StorageConfig:
    """Tests for S3StorageConfig dataclass."""

    def test_default_values(self):
        config = S3StorageConfig()
        assert config.max_concurrency == 10
        assert config.chunk_size == 8 * 1024 * 1024
        assert config.default_region == "us-east-1"

    def test_validation_max_concurrency(self):
        with pytest.raises(ValueError, match="max_concurrency"):
            S3StorageConfig(max_concurrency=0)

    def test_validation_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size"):
            S3StorageConfig(chunk_size=100)

    def test_to_dict_excludes_secrets(self):
        config = S3StorageConfig(access_key="secret", secret_key="verysecret")
        d = config.to_dict()
        assert "access_key" not in d
        assert "secret_key" not in d


class TestLSHConfig:
    """Tests for LSHConfig dataclass."""

    def test_default_values(self):
        config = LSHConfig()
        assert config.num_hyperplanes == 8
        assert config.num_tables == 4
        assert config.max_candidates == 100
        assert config.seed == 42

    def test_validation(self):
        with pytest.raises(ValueError, match="num_hyperplanes"):
            LSHConfig(num_hyperplanes=0)

        with pytest.raises(ValueError, match="num_tables"):
            LSHConfig(num_tables=0)

        with pytest.raises(ValueError, match="max_candidates"):
            LSHConfig(max_candidates=0)


class TestSimHashConfig:
    """Tests for SimHashConfig dataclass."""

    def test_default_values(self):
        config = SimHashConfig()
        assert config.num_bits == 64
        assert config.similarity_threshold == 0.1
        assert config.seed == 42

    def test_validation_num_bits(self):
        with pytest.raises(ValueError, match="num_bits"):
            SimHashConfig(num_bits=32)

    def test_validation_similarity_threshold(self):
        with pytest.raises(ValueError, match="similarity_threshold"):
            SimHashConfig(similarity_threshold=1.5)


class TestCheckpointDefaults:
    """Tests for CheckpointDefaults dataclass."""

    def test_default_values(self):
        config = CheckpointDefaults()
        assert config.save_optimizer_state is True
        assert config.auto_commit is True
        assert "epoch" in config.commit_message_template


class TestRemoteConfig:
    """Tests for RemoteConfig dataclass."""

    def test_creation(self):
        config = RemoteConfig(name="origin", url="s3://bucket/path")
        assert config.name == "origin"
        assert config.url == "s3://bucket/path"
        assert config.backend == "s3"

    def test_to_dict_excludes_secrets(self):
        config = RemoteConfig(
            name="origin",
            url="s3://bucket",
            access_key="key",
            secret_key="secret",
        )
        d = config.to_dict()
        assert "access_key" not in d
        assert "secret_key" not in d


class TestCoralConfig:
    """Tests for the main CoralConfig class."""

    def test_default_config(self):
        config = CoralConfig()
        assert isinstance(config.user, UserConfig)
        assert isinstance(config.core, CoreConfig)
        assert isinstance(config.delta, DeltaEncodingConfig)
        assert isinstance(config.storage, StorageConfig)

    def test_to_dict(self):
        config = CoralConfig()
        d = config.to_dict()
        assert "user" in d
        assert "core" in d
        assert "delta" in d
        assert "storage" in d

    def test_from_dict(self):
        data = {
            "user": {"name": "Test"},
            "core": {"similarity_threshold": 0.95},
        }
        config = CoralConfig.from_dict(data)
        assert config.user.name == "Test"
        assert config.core.similarity_threshold == 0.95

    def test_get_nested(self):
        config = CoralConfig()
        assert config.get_nested("core.similarity_threshold") == 0.98
        assert config.get_nested("user.name") == "Anonymous"
        assert config.get_nested("invalid.key") is None
        assert config.get_nested("invalid.key", "default") == "default"

    def test_set_nested(self):
        config = CoralConfig()
        config.set_nested("core.similarity_threshold", 0.95)
        assert config.core.similarity_threshold == 0.95

        config.set_nested("user.name", "New Name")
        assert config.user.name == "New Name"

    def test_set_nested_invalid_key(self):
        config = CoralConfig()
        with pytest.raises(KeyError):
            config.set_nested("invalid.deep.key", "value")


class TestConfigLoader:
    """Tests for ConfigLoader."""

    def test_load_default_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ConfigLoader(repo_path=Path(tmpdir))
            config = loader.load()
            assert isinstance(config, CoralConfig)

    def test_load_legacy_json_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            coral_dir = repo_path / ".coral"
            coral_dir.mkdir()

            # Create legacy config
            legacy_config = {
                "user": {"name": "Legacy User", "email": "legacy@test.com"},
                "core": {
                    "compression": "lzf",
                    "similarity_threshold": 0.90,
                },
            }
            with open(coral_dir / "config.json", "w") as f:
                json.dump(legacy_config, f)

            loader = ConfigLoader(repo_path=repo_path)
            config = loader.load()

            assert config.user.name == "Legacy User"
            assert config.core.compression == "lzf"
            assert config.core.similarity_threshold == 0.90

    def test_env_var_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(
                os.environ,
                {
                    "CORAL_CORE_SIMILARITY_THRESHOLD": "0.85",
                    "CORAL_USER_NAME": "Env User",
                    "CORAL_CORE_DELTA_ENCODING": "false",
                },
            ):
                loader = ConfigLoader(repo_path=Path(tmpdir))
                config = loader.load()

                assert config.core.similarity_threshold == 0.85
                assert config.user.name == "Env User"
                assert config.core.delta_encoding is False

    def test_save_repo_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            coral_dir = repo_path / ".coral"
            coral_dir.mkdir()

            config = CoralConfig()
            config.user.name = "Saved User"

            loader = ConfigLoader(repo_path=repo_path)
            loader.save_repo_config(config)

            # Verify file was created
            config_file = coral_dir / "coral.toml"
            assert config_file.exists()


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_config_no_repo(self):
        config = load_config()
        assert isinstance(config, CoralConfig)

    def test_get_default_config(self):
        config = get_default_config()
        assert isinstance(config, CoralConfig)
        assert config.core.similarity_threshold == 0.98


class TestValidation:
    """Tests for configuration validation."""

    def test_validate_valid_config(self):
        config = CoralConfig()
        result = validate_config(config)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_invalid_compression(self):
        config = CoralConfig()
        config.core.compression = "invalid"
        result = validate_config(config)
        assert result.valid is False
        assert any("compression" in str(e) for e in result.errors)

    def test_validate_low_similarity_threshold_warning(self):
        config = CoralConfig()
        config.core.similarity_threshold = 0.5
        result = validate_config(config)
        # Should be valid but have warnings
        assert result.valid is True
        assert any("similarity_threshold" in str(w) for w in result.warnings)

    def test_validate_invalid_delta_type(self):
        config = CoralConfig()
        config.core.delta_type = "invalid_type"
        result = validate_config(config)
        assert result.valid is False
        assert any("delta_type" in str(e) for e in result.errors)

    def test_validate_lossy_delta_warning(self):
        config = CoralConfig()
        config.core.delta_type = "int8_quantized"
        result = validate_config(config)
        assert any("lossy" in str(w).lower() for w in result.warnings)

    def test_validate_invalid_log_level(self):
        config = CoralConfig()
        config.logging.level = "INVALID_LEVEL"
        result = validate_config(config)
        assert result.valid is False

    def test_validate_value_function(self):
        # Valid values
        assert validate_value("core.similarity_threshold", 0.95) is None
        assert validate_value("core.compression_level", 5) is None

        # Invalid values
        error = validate_value("core.similarity_threshold", 1.5)
        assert error is not None
        assert "0.0 and 1.0" in str(error)

        error = validate_value("core.compression_level", 15)
        assert error is not None


class TestValidationError:
    """Tests for ValidationError and ValidationResult."""

    def test_validation_error_str(self):
        error = ValidationError("test.key", "must be valid", 123)
        assert "test.key" in str(error)
        assert "must be valid" in str(error)
        assert "123" in str(error)

    def test_validation_error_no_value(self):
        error = ValidationError("test.key", "is required")
        assert "test.key" in str(error)
        assert "is required" in str(error)

    def test_validation_result_bool(self):
        result_valid = ValidationResult(valid=True, errors=[], warnings=[])
        assert bool(result_valid) is True

        result_invalid = ValidationResult(
            valid=False,
            errors=[ValidationError("key", "error")],
            warnings=[],
        )
        assert bool(result_invalid) is False


class TestEnvVarParsing:
    """Tests for environment variable parsing."""

    def test_parse_boolean_true(self):
        with mock.patch.dict(os.environ, {"CORAL_CORE_DELTA_ENCODING": "true"}):
            config = load_config()
            assert config.core.delta_encoding is True

        with mock.patch.dict(os.environ, {"CORAL_CORE_DELTA_ENCODING": "yes"}):
            config = load_config()
            assert config.core.delta_encoding is True

        with mock.patch.dict(os.environ, {"CORAL_CORE_DELTA_ENCODING": "1"}):
            config = load_config()
            assert config.core.delta_encoding is True

    def test_parse_boolean_false(self):
        with mock.patch.dict(os.environ, {"CORAL_CORE_DELTA_ENCODING": "false"}):
            config = load_config()
            assert config.core.delta_encoding is False

        with mock.patch.dict(os.environ, {"CORAL_CORE_DELTA_ENCODING": "no"}):
            config = load_config()
            assert config.core.delta_encoding is False

    def test_parse_float(self):
        with mock.patch.dict(os.environ, {"CORAL_CORE_SIMILARITY_THRESHOLD": "0.95"}):
            config = load_config()
            assert config.core.similarity_threshold == 0.95

    def test_parse_int(self):
        with mock.patch.dict(os.environ, {"CORAL_CORE_COMPRESSION_LEVEL": "7"}):
            config = load_config()
            assert config.core.compression_level == 7

    def test_parse_string(self):
        with mock.patch.dict(os.environ, {"CORAL_USER_NAME": "Test User"}):
            config = load_config()
            assert config.user.name == "Test User"


class TestConfigWithRepository:
    """Tests for configuration integration with Repository."""

    def test_repository_uses_config(self):
        from coral import Repository

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create repository with custom config
            custom_config = CoralConfig()
            custom_config.core.similarity_threshold = 0.85
            custom_config.user.name = "Custom User"

            repo = Repository(repo_path, init=True, config=custom_config)

            assert repo.coral_config.core.similarity_threshold == 0.85
            assert repo.coral_config.user.name == "Custom User"

    def test_repository_loads_config_from_file(self):
        from coral import Repository

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Initialize repository
            Repository(repo_path, init=True)

            # Verify coral.toml was created
            toml_path = repo_path / ".coral" / "coral.toml"
            assert toml_path.exists()

    def test_repository_save_config(self):
        from coral import Repository

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            repo = Repository(repo_path, init=True)

            # Modify and save config
            repo._coral_config.user.name = "Updated User"
            repo.save_config()

            # Create new repository instance and verify
            repo2 = Repository(repo_path)
            assert repo2.coral_config.user.name == "Updated User"
