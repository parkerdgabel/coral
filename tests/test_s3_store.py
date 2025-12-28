"""Tests for S3 storage backend.

This module tests the S3Store with mocked boto3 dependencies.
Tests are skipped if boto3 is not installed.
"""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coral.core.weight_tensor import WeightMetadata, WeightTensor

# Check if boto3 is available
try:
    import boto3

    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
    boto3 = None


class TestS3Config:
    """Tests for S3Config dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from coral.storage.s3_store import S3Config

        config = S3Config(bucket="test-bucket")

        assert config.bucket == "test-bucket"
        assert config.prefix == "coral/"
        assert config.region is None
        assert config.endpoint_url is None
        assert config.max_concurrency == 10
        assert config.compression == "gzip"
        assert config.chunk_size == 8 * 1024 * 1024

    def test_custom_values(self):
        """Test custom configuration values."""
        from coral.storage.s3_store import S3Config

        config = S3Config(
            bucket="my-bucket",
            prefix="models/",
            region="us-west-2",
            endpoint_url="http://localhost:9000",
            access_key="test-key",
            secret_key="test-secret",
            max_concurrency=20,
            compression="none",
        )

        assert config.bucket == "my-bucket"
        assert config.prefix == "models/"
        assert config.region == "us-west-2"
        assert config.endpoint_url == "http://localhost:9000"
        assert config.access_key == "test-key"
        assert config.secret_key == "test-secret"
        assert config.max_concurrency == 20
        assert config.compression == "none"


@pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
class TestS3Store:
    """Tests for S3Store class."""

    @pytest.fixture
    def mock_s3_client(self):
        """Create a mock S3 client."""
        mock_client = MagicMock()
        mock_client.head_bucket = MagicMock()
        mock_client.create_bucket = MagicMock()
        mock_client.put_object = MagicMock()
        mock_client.get_object = MagicMock()
        mock_client.head_object = MagicMock()
        mock_client.delete_objects = MagicMock()
        mock_client.get_paginator = MagicMock()
        return mock_client

    @pytest.fixture
    def s3_store(self, mock_s3_client):
        """Create an S3Store for testing."""
        from coral.storage.s3_store import S3Config, S3Store

        with patch.object(boto3, "client", return_value=mock_s3_client):
            config = S3Config(
                bucket="test-bucket",
                prefix="coral/",
                compression="none",
            )
            store = S3Store(config)
            store._client = mock_s3_client
            return store

    @pytest.fixture
    def sample_weight(self):
        """Create a sample weight for testing."""
        return WeightTensor(
            data=np.random.randn(10, 5).astype(np.float32),
            metadata=WeightMetadata(
                name="layer1.weight",
                shape=(10, 5),
                dtype=np.float32,
            ),
        )

    def test_weight_key(self, s3_store):
        """Test weight key generation."""
        key = s3_store._weight_key("abc123")
        assert key == "coral/weights/abc123.npz"

    def test_metadata_key(self, s3_store):
        """Test metadata key generation."""
        key = s3_store._metadata_key("abc123")
        assert key == "coral/metadata/abc123.json"

    def test_delta_key(self, s3_store):
        """Test delta key generation."""
        key = s3_store._delta_key("abc123")
        assert key == "coral/deltas/abc123.npz"

    def test_compress_data_none(self, s3_store):
        """Test compression with none."""
        s3_store.config.compression = "none"
        data = b"test data"
        compressed, compression_type = s3_store._compress_data(data)

        assert compressed == data
        assert compression_type == "none"

    def test_compress_data_gzip(self, s3_store):
        """Test compression with gzip."""
        s3_store.config.compression = "gzip"
        data = b"test data" * 100
        compressed, compression_type = s3_store._compress_data(data)

        assert len(compressed) < len(data)
        assert compression_type == "gzip"

    def test_decompress_data_none(self, s3_store):
        """Test decompression with none."""
        data = b"test data"
        decompressed = s3_store._decompress_data(data, "none")

        assert decompressed == data

    def test_decompress_data_gzip(self, s3_store):
        """Test decompression with gzip."""
        import gzip

        original = b"test data"
        compressed = gzip.compress(original)
        decompressed = s3_store._decompress_data(compressed, "gzip")

        assert decompressed == original

    def test_store_weight(self, s3_store, mock_s3_client, sample_weight):
        """Test storing a weight."""
        hash_key = s3_store.store(sample_weight)

        assert hash_key is not None
        assert mock_s3_client.put_object.call_count == 2  # weight + metadata

    def test_store_weight_with_hash(self, s3_store, mock_s3_client, sample_weight):
        """Test storing a weight with explicit hash."""
        hash_key = s3_store.store(sample_weight, hash_key="custom-hash")

        assert hash_key == "custom-hash"

    def test_load_weight(self, s3_store, mock_s3_client, sample_weight):
        """Test loading a weight."""
        # Mock the response
        buffer = io.BytesIO()
        np.savez_compressed(buffer, data=sample_weight.data)
        weight_bytes = buffer.getvalue()

        mock_response = {
            "Body": io.BytesIO(weight_bytes),
            "Metadata": {"compression": "none"},
        }
        mock_s3_client.get_object.return_value = mock_response

        loaded = s3_store.load("test-hash")

        assert loaded is not None
        mock_s3_client.get_object.assert_called()

    def test_load_nonexistent_weight(self, s3_store, mock_s3_client):
        """Test loading a weight that doesn't exist."""
        from botocore.exceptions import ClientError

        error_response = {"Error": {"Code": "NoSuchKey"}}
        mock_s3_client.get_object.side_effect = ClientError(
            error_response, "GetObject"
        )

        loaded = s3_store.load("nonexistent-hash")

        assert loaded is None

    def test_exists_true(self, s3_store, mock_s3_client):
        """Test exists returns True for existing weight."""
        mock_s3_client.head_object.return_value = {}

        assert s3_store.exists("test-hash") is True

    def test_exists_false(self, s3_store, mock_s3_client):
        """Test exists returns False for missing weight."""
        from botocore.exceptions import ClientError

        error_response = {"Error": {"Code": "404"}}
        mock_s3_client.head_object.side_effect = ClientError(
            error_response, "HeadObject"
        )

        assert s3_store.exists("test-hash") is False

    def test_delete(self, s3_store, mock_s3_client):
        """Test deleting a weight."""
        mock_s3_client.head_object.return_value = {}  # exists

        result = s3_store.delete("test-hash")

        assert result is True
        mock_s3_client.delete_objects.assert_called_once()

    def test_delete_nonexistent(self, s3_store, mock_s3_client):
        """Test deleting a nonexistent weight."""
        from botocore.exceptions import ClientError

        error_response = {"Error": {"Code": "404"}}
        mock_s3_client.head_object.side_effect = ClientError(
            error_response, "HeadObject"
        )

        result = s3_store.delete("test-hash")

        assert result is False

    def test_list_weights(self, s3_store, mock_s3_client):
        """Test listing weights."""
        # Mock paginator
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "coral/weights/hash1.npz"},
                    {"Key": "coral/weights/hash2.npz"},
                    {"Key": "coral/weights/hash3.npz"},
                ]
            }
        ]
        mock_s3_client.get_paginator.return_value = mock_paginator

        hashes = s3_store.list_weights()

        assert len(hashes) == 3
        assert "hash1" in hashes
        assert "hash2" in hashes
        assert "hash3" in hashes

    def test_get_metadata(self, s3_store, mock_s3_client):
        """Test getting metadata."""
        import json

        metadata_dict = {
            "name": "layer1.weight",
            "shape": [10, 5],
            "dtype": "float32",
        }
        mock_response = {"Body": io.BytesIO(json.dumps(metadata_dict).encode())}
        mock_s3_client.get_object.return_value = mock_response

        metadata = s3_store.get_metadata("test-hash")

        assert metadata is not None
        assert metadata.name == "layer1.weight"

    def test_store_batch(self, s3_store, mock_s3_client):
        """Test storing multiple weights."""
        weights = {
            "weight1": WeightTensor(
                data=np.random.randn(10, 5).astype(np.float32),
                metadata=WeightMetadata(
                    name="weight1", shape=(10, 5), dtype=np.float32
                ),
            ),
            "weight2": WeightTensor(
                data=np.random.randn(5, 3).astype(np.float32),
                metadata=WeightMetadata(
                    name="weight2", shape=(5, 3), dtype=np.float32
                ),
            ),
        }

        results = s3_store.store_batch(weights)

        assert len(results) == 2
        assert "weight1" in results
        assert "weight2" in results

    def test_load_batch(self, s3_store, mock_s3_client, sample_weight):
        """Test loading multiple weights."""
        # Mock the response
        buffer = io.BytesIO()
        np.savez_compressed(buffer, data=sample_weight.data)
        weight_bytes = buffer.getvalue()

        def mock_get_object(Bucket, Key):
            return {
                "Body": io.BytesIO(weight_bytes),
                "Metadata": {"compression": "none"},
            }

        mock_s3_client.get_object.side_effect = mock_get_object

        results = s3_store.load_batch(["hash1", "hash2"])

        assert len(results) == 2

    def test_get_storage_info(self, s3_store, mock_s3_client):
        """Test getting storage info."""
        # Mock paginator
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "coral/weights/hash1.npz", "Size": 1000},
                    {"Key": "coral/weights/hash2.npz", "Size": 2000},
                ]
            }
        ]
        mock_s3_client.get_paginator.return_value = mock_paginator

        info = s3_store.get_storage_info()

        assert info["backend"] == "s3"
        assert info["bucket"] == "test-bucket"
        assert info["weight_count"] >= 0

    def test_store_delta(self, s3_store, mock_s3_client):
        """Test storing a delta."""
        delta_data = b"delta content"
        metadata = {"base_hash": "abc123", "compression": "none"}

        s3_store.store_delta("delta-hash", delta_data, metadata)

        mock_s3_client.put_object.assert_called_once()

    def test_load_delta(self, s3_store, mock_s3_client):
        """Test loading a delta."""
        delta_data = b"delta content"
        mock_response = {
            "Body": io.BytesIO(delta_data),
            "Metadata": {"compression": "none", "base_hash": "abc123"},
        }
        mock_s3_client.get_object.return_value = mock_response

        result = s3_store.load_delta("delta-hash")

        assert result is not None
        data, metadata = result
        assert data == delta_data

    def test_load_delta_not_found(self, s3_store, mock_s3_client):
        """Test loading a nonexistent delta."""
        from botocore.exceptions import ClientError

        error_response = {"Error": {"Code": "NoSuchKey"}}
        mock_s3_client.get_object.side_effect = ClientError(
            error_response, "GetObject"
        )

        result = s3_store.load_delta("nonexistent")

        assert result is None

    def test_delta_exists_true(self, s3_store, mock_s3_client):
        """Test delta_exists returns True for existing delta."""
        mock_s3_client.head_object.return_value = {}

        assert s3_store.delta_exists("delta-hash") is True

    def test_delta_exists_false(self, s3_store, mock_s3_client):
        """Test delta_exists returns False for missing delta."""
        from botocore.exceptions import ClientError

        error_response = {"Error": {"Code": "404"}}
        mock_s3_client.head_object.side_effect = ClientError(
            error_response, "HeadObject"
        )

        assert s3_store.delta_exists("delta-hash") is False

    def test_close(self, s3_store):
        """Test closing the store."""
        # Should not raise
        s3_store.close()


@pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
class TestS3StoreSyncMethods:
    """Tests for S3Store sync methods."""

    @pytest.fixture
    def mock_s3_client(self):
        """Create a mock S3 client."""
        mock_client = MagicMock()
        mock_client.head_bucket = MagicMock()
        mock_client.put_object = MagicMock()
        mock_client.get_object = MagicMock()
        mock_client.head_object = MagicMock()
        mock_client.get_paginator = MagicMock()
        return mock_client

    @pytest.fixture
    def s3_store(self, mock_s3_client):
        """Create an S3Store for testing."""
        from coral.storage.s3_store import S3Config, S3Store

        with patch.object(boto3, "client", return_value=mock_s3_client):
            config = S3Config(
                bucket="test-bucket",
                prefix="coral/",
                compression="none",
            )
            store = S3Store(config)
            store._client = mock_s3_client
            return store

    def test_sync_from_local(self, s3_store, mock_s3_client):
        """Test syncing from local to S3."""
        # Mock local store
        mock_local_store = MagicMock()
        mock_local_store.list_weights.return_value = ["hash1", "hash2", "hash3"]
        mock_local_store.load.return_value = WeightTensor(
            data=np.random.randn(10, 5).astype(np.float32),
            metadata=WeightMetadata(
                name="weight", shape=(10, 5), dtype=np.float32
            ),
        )

        # Mock S3 store list_weights (empty at first)
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Contents": []}]
        mock_s3_client.get_paginator.return_value = mock_paginator

        stats = s3_store.sync_from_local(mock_local_store)

        assert stats["uploaded"] == 3
        assert stats["bytes_transferred"] > 0

    def test_sync_to_local(self, s3_store, mock_s3_client):
        """Test syncing from S3 to local."""
        # Mock local store (empty)
        mock_local_store = MagicMock()
        mock_local_store.list_weights.return_value = []

        # Mock S3 weights
        sample_data = np.random.randn(10, 5).astype(np.float32)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, data=sample_data)
        weight_bytes = buffer.getvalue()

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "coral/weights/hash1.npz"},
                    {"Key": "coral/weights/hash2.npz"},
                ]
            }
        ]
        mock_s3_client.get_paginator.return_value = mock_paginator

        mock_s3_client.get_object.return_value = {
            "Body": io.BytesIO(weight_bytes),
            "Metadata": {"compression": "none"},
        }

        stats = s3_store.sync_to_local(mock_local_store)

        assert stats["downloaded"] == 2
        assert stats["bytes_transferred"] > 0


@pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
class TestS3StoreCompression:
    """Tests for S3Store compression methods."""

    @pytest.fixture
    def mock_s3_client(self):
        """Create a mock S3 client."""
        mock_client = MagicMock()
        mock_client.head_bucket = MagicMock()
        return mock_client

    def test_gzip_compression_roundtrip(self, mock_s3_client):
        """Test gzip compression and decompression."""
        from coral.storage.s3_store import S3Config, S3Store

        with patch.object(boto3, "client", return_value=mock_s3_client):
            config = S3Config(bucket="test-bucket", compression="gzip")
            store = S3Store(config)
            store._client = mock_s3_client

            original_data = b"test data" * 1000
            compressed, compression = store._compress_data(original_data)
            decompressed = store._decompress_data(compressed, compression)

            assert decompressed == original_data
            assert len(compressed) < len(original_data)

    def test_no_compression_roundtrip(self, mock_s3_client):
        """Test no compression."""
        from coral.storage.s3_store import S3Config, S3Store

        with patch.object(boto3, "client", return_value=mock_s3_client):
            config = S3Config(bucket="test-bucket", compression="none")
            store = S3Store(config)
            store._client = mock_s3_client

            original_data = b"test data"
            compressed, compression = store._compress_data(original_data)
            decompressed = store._decompress_data(compressed, compression)

            assert decompressed == original_data
            assert compressed == original_data
            assert compression == "none"


class TestS3StoreWithoutBoto3:
    """Tests that work without boto3 installed."""

    def test_import_error_without_boto3(self):
        """Test that S3Store raises ImportError without boto3."""
        if HAS_BOTO3:
            pytest.skip("boto3 is installed")

        with patch("coral.storage.s3_store.BOTO3_AVAILABLE", False):
            from coral.storage.s3_store import S3Config, S3Store

            config = S3Config(bucket="test-bucket")
            with pytest.raises(ImportError, match="boto3 is required"):
                S3Store(config)
