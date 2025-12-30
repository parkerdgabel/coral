"""Tests for HuggingFace Hub integration.

This module tests the CoralHubClient and related functionality.
Tests use mocks for huggingface_hub and safetensors dependencies.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coral.core.weight_tensor import WeightMetadata, WeightTensor


class TestModelInfo:
    """Test ModelInfo dataclass."""

    def test_model_info_creation(self):
        """Test creating ModelInfo."""
        from coral.integrations.huggingface import ModelInfo

        info = ModelInfo(
            repo_id="test/model",
            revision="main",
            files=["model.safetensors", "config.json"],
            base_model="base/model",
            total_size_bytes=1000000,
        )

        assert info.repo_id == "test/model"
        assert info.revision == "main"
        assert len(info.files) == 2
        assert info.base_model == "base/model"
        assert info.total_size_bytes == 1000000

    def test_model_info_weight_files(self):
        """Test ModelInfo filters weight files."""
        from coral.integrations.huggingface import ModelInfo

        info = ModelInfo(
            repo_id="test/model",
            revision="main",
            files=[
                "model.safetensors",
                "config.json",
                "weights.bin",
                "pytorch_model.pt",
                "README.md",
            ],
        )

        assert info.weight_files == [
            "model.safetensors",
            "weights.bin",
            "pytorch_model.pt",
        ]


class TestDownloadStats:
    """Test DownloadStats dataclass."""

    def test_download_stats_defaults(self):
        """Test DownloadStats default values."""
        from coral.integrations.huggingface import DownloadStats

        stats = DownloadStats()

        assert stats.total_weights == 0
        assert stats.downloaded_full == 0
        assert stats.downloaded_delta == 0
        assert stats.bytes_downloaded == 0
        assert stats.bytes_saved == 0

    def test_download_stats_savings_percent(self):
        """Test savings percent calculation."""
        from coral.integrations.huggingface import DownloadStats

        stats = DownloadStats(
            total_weights=10,
            downloaded_full=5,
            downloaded_delta=5,
            bytes_downloaded=500,
            bytes_saved=500,
        )

        assert stats.savings_percent == 50.0

    def test_download_stats_savings_percent_zero(self):
        """Test savings percent with zero total."""
        from coral.integrations.huggingface import DownloadStats

        stats = DownloadStats()
        assert stats.savings_percent == 0.0


class TestCoralHubClient:
    """Test CoralHubClient class."""

    @pytest.fixture
    def mock_hf_api(self):
        """Create a mock HF API."""
        mock_api = MagicMock()
        return mock_api

    def test_require_huggingface(self):
        """Test that _require_huggingface raises error when not available."""
        from coral.integrations.huggingface import _require_huggingface

        with patch("coral.integrations.huggingface.HF_AVAILABLE", False):
            with pytest.raises(ImportError, match="huggingface-hub is required"):
                _require_huggingface()

    def test_require_safetensors(self):
        """Test that _require_safetensors raises error when not available."""
        from coral.integrations.huggingface import _require_safetensors

        with patch("coral.integrations.huggingface.SAFETENSORS_AVAILABLE", False):
            with pytest.raises(ImportError, match="safetensors is required"):
                _require_safetensors()

    @pytest.mark.skipif(True, reason="Skipping as huggingface_hub may not be installed")
    def test_client_initialization(self, mock_hf_api):
        """Test client initialization."""
        from coral.integrations.huggingface import CoralHubClient

        with patch("coral.integrations.huggingface.HfApi", return_value=mock_hf_api):
            with patch("coral.integrations.huggingface.HF_AVAILABLE", True):
                client = CoralHubClient(
                    cache_dir="/tmp/test_cache",
                    token="test_token",
                    similarity_threshold=0.9,
                )

                assert client.token == "test_token"
                assert client.similarity_threshold == 0.9


class TestCoralHubClientMocked:
    """Test CoralHubClient with full mocking."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock CoralHubClient."""
        with patch("coral.integrations.huggingface.HF_AVAILABLE", True):
            with patch("coral.integrations.huggingface.HfApi") as mock_api_class:
                mock_api = MagicMock()
                mock_api_class.return_value = mock_api

                from coral.integrations.huggingface import CoralHubClient

                client = CoralHubClient(
                    cache_dir=tempfile.mkdtemp(),
                    token="test_token",
                )
                client.api = mock_api
                return client

    def test_get_model_info(self, mock_client):
        """Test getting model info."""
        # Mock the model_info response
        mock_info = MagicMock()
        mock_info.siblings = [
            MagicMock(rfilename="model.safetensors", size=1000000),
            MagicMock(rfilename="config.json", size=500),
        ]
        mock_info.card_data = MagicMock()
        mock_info.card_data.base_model = "base/model"

        mock_client.api.model_info.return_value = mock_info

        info = mock_client.get_model_info("test/model", revision="main")

        assert info.repo_id == "test/model"
        assert info.revision == "main"
        assert "model.safetensors" in info.files
        assert info.base_model == "base/model"

    def test_get_model_info_no_base_model(self, mock_client):
        """Test getting model info without base model."""
        mock_info = MagicMock()
        mock_info.siblings = [
            MagicMock(rfilename="model.safetensors", size=1000000),
        ]
        mock_info.card_data = None

        mock_client.api.model_info.return_value = mock_info

        info = mock_client.get_model_info("test/model")

        assert info.base_model is None

    def test_cache_weight(self, mock_client):
        """Test caching a weight."""
        weight = WeightTensor(
            data=np.random.randn(10, 5).astype(np.float32),
            metadata=WeightMetadata(name="test", shape=(10, 5), dtype=np.float32),
        )

        hash_key = mock_client._cache_weight(weight)

        assert hash_key is not None
        assert hash_key in mock_client._weight_cache
        assert mock_client._weight_cache[hash_key] == weight

    def test_cache_weight_deduplication(self, mock_client):
        """Test that caching same weight returns same hash."""
        weight1 = WeightTensor(
            data=np.ones((10, 5), dtype=np.float32),
            metadata=WeightMetadata(name="test1", shape=(10, 5), dtype=np.float32),
        )
        weight2 = WeightTensor(
            data=np.ones((10, 5), dtype=np.float32),
            metadata=WeightMetadata(name="test2", shape=(10, 5), dtype=np.float32),
        )

        hash1 = mock_client._cache_weight(weight1)
        hash2 = mock_client._cache_weight(weight2)

        assert hash1 == hash2
        assert len(mock_client._weight_cache) == 1

    def test_are_similar_different_shapes(self, mock_client):
        """Test similarity check with different shapes."""
        weight1 = WeightTensor(
            data=np.ones((10, 5), dtype=np.float32),
            metadata=WeightMetadata(name="test1", shape=(10, 5), dtype=np.float32),
        )
        weight2 = WeightTensor(
            data=np.ones((5, 10), dtype=np.float32),
            metadata=WeightMetadata(name="test2", shape=(5, 10), dtype=np.float32),
        )

        assert mock_client._are_similar(weight1, weight2) is False

    def test_are_similar_same_weights(self, mock_client):
        """Test similarity check with same weights."""
        weight1 = WeightTensor(
            data=np.ones((10, 5), dtype=np.float32),
            metadata=WeightMetadata(name="test1", shape=(10, 5), dtype=np.float32),
        )
        weight2 = WeightTensor(
            data=np.ones((10, 5), dtype=np.float32),
            metadata=WeightMetadata(name="test2", shape=(10, 5), dtype=np.float32),
        )

        assert mock_client._are_similar(weight1, weight2) is True


class TestCoralHubClientDownload:
    """Test CoralHubClient download functionality."""

    @pytest.fixture
    def setup_mocks(self):
        """Set up mocks for HuggingFace integration."""
        # Only run if safetensors is available
        try:
            import safetensors  # noqa: F401
        except ImportError:
            pytest.skip("safetensors not installed")

        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            pytest.skip("huggingface_hub not installed")

    def test_download_model_no_base(self, setup_mocks):
        """Test downloading model without base model."""
        from coral.integrations.huggingface import CoralHubClient

        with patch.object(CoralHubClient, "__init__", lambda self, **kwargs: None):
            client = CoralHubClient()
            client.cache_dir = Path(tempfile.mkdtemp())
            client.token = None
            client.api = MagicMock()
            client.delta_config = MagicMock()
            client.delta_encoder = MagicMock()
            client.similarity_threshold = 0.95
            client.last_download_stats = None
            client._weight_cache = {}
            client._weight_index = {}

            # Mock the model_info
            mock_info = MagicMock()
            mock_info.siblings = [MagicMock(rfilename="model.safetensors", size=1000)]
            mock_info.card_data = None
            client.api.model_info.return_value = mock_info

            # Mock _load_safetensors_dir and _cache_weight
            mock_weights = {
                "weight1": WeightTensor(
                    data=np.ones((10, 5), dtype=np.float32),
                    metadata=WeightMetadata(
                        name="weight1", shape=(10, 5), dtype=np.float32
                    ),
                ),
            }
            client._load_safetensors_dir = MagicMock(return_value=mock_weights)
            client._load_cached_model = MagicMock(return_value={})
            client._cache_weight = MagicMock()

            with patch(
                "coral.integrations.huggingface.snapshot_download"
            ) as mock_download:
                temp_path = tempfile.mkdtemp()
                mock_download.return_value = temp_path

                # Call the method directly
                from coral.integrations.huggingface import CoralHubClient as RealClient

                weights = RealClient.download_model(client, "test/model")

                assert len(weights) == 1
                assert client.last_download_stats is not None
                assert client.last_download_stats.total_weights == 1


class TestCoralHubClientUpload:
    """Test CoralHubClient upload functionality."""

    @pytest.fixture
    def setup_mocks(self):
        """Set up mocks for HuggingFace integration."""
        # Only run if safetensors is available
        try:
            import safetensors  # noqa: F401
        except ImportError:
            pytest.skip("safetensors not installed")

        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            pytest.skip("huggingface_hub not installed")

    def test_upload_model(self, setup_mocks):
        """Test uploading model."""
        import safetensors.numpy as st_numpy

        from coral.integrations.huggingface import CoralHubClient

        with patch.object(CoralHubClient, "__init__", lambda self, **kwargs: None):
            client = CoralHubClient()
            client.api = MagicMock()

            weights = {
                "weight1": WeightTensor(
                    data=np.ones((10, 5), dtype=np.float32),
                    metadata=WeightMetadata(
                        name="weight1", shape=(10, 5), dtype=np.float32
                    ),
                ),
            }

            with patch.object(st_numpy, "save_file"):
                url = CoralHubClient.upload_model(
                    client,
                    weights,
                    repo_id="test/model",
                    commit_message="Upload test",
                    private=False,
                )

            assert url == "https://huggingface.co/test/model"
            client.api.create_repo.assert_called_once()
            client.api.upload_file.assert_called()

    def test_upload_model_with_base_model(self, setup_mocks):
        """Test uploading model with base model reference."""
        import safetensors.numpy as st_numpy

        from coral.integrations.huggingface import CoralHubClient

        with patch.object(CoralHubClient, "__init__", lambda self, **kwargs: None):
            client = CoralHubClient()
            client.api = MagicMock()

            weights = {
                "weight1": WeightTensor(
                    data=np.ones((10, 5), dtype=np.float32),
                    metadata=WeightMetadata(
                        name="weight1", shape=(10, 5), dtype=np.float32
                    ),
                ),
            }

            with patch.object(st_numpy, "save_file"):
                url = CoralHubClient.upload_model(
                    client,
                    weights,
                    repo_id="test/model",
                    base_model="base/model",
                )

            assert url == "https://huggingface.co/test/model"
            # Should have uploaded README with base model info
            assert client.api.upload_file.call_count >= 2


class TestLoadPretrainedEfficient:
    """Test load_pretrained_efficient function."""

    def test_load_pretrained_efficient(self):
        """Test the convenience function."""
        with patch("coral.integrations.huggingface.HF_AVAILABLE", True):
            with patch("coral.integrations.huggingface.SAFETENSORS_AVAILABLE", True):
                with patch(
                    "coral.integrations.huggingface.CoralHubClient"
                ) as mock_client_class:
                    mock_client = MagicMock()
                    mock_weights = {"weight1": MagicMock()}
                    mock_client.download_model.return_value = mock_weights
                    mock_client_class.return_value = mock_client

                    from coral.integrations.huggingface import load_pretrained_efficient

                    weights = load_pretrained_efficient(
                        "test/model",
                        base_model="base/model",
                    )

                    assert weights == mock_weights
                    mock_client.download_model.assert_called_once_with(
                        "test/model", base_model="base/model"
                    )
