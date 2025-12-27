"""S3-compatible storage backend for remote weight storage.

This module provides cloud storage support for Coral, enabling:
- Remote model storage on S3, MinIO, or any S3-compatible service
- Efficient delta uploads (only upload changed weights)
- Concurrent uploads/downloads for large models
- Automatic compression and decompression

Requires: boto3 (install with `pip install coral-ml[s3]`)
"""

import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.storage.weight_store import WeightStore

logger = logging.getLogger(__name__)

# Check for boto3 availability
try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    Config = None
    ClientError = Exception


@dataclass
class S3Config:
    """Configuration for S3 storage backend."""

    bucket: str
    prefix: str = "coral/"
    region: Optional[str] = None
    endpoint_url: Optional[str] = None  # For MinIO or other S3-compatible services
    access_key: Optional[str] = None  # Falls back to AWS credentials chain
    secret_key: Optional[str] = None
    max_concurrency: int = 10
    compression: str = "gzip"  # gzip, lz4, or none
    chunk_size: int = 8 * 1024 * 1024  # 8MB chunks for multipart upload


class S3Store(WeightStore):
    """S3-compatible storage backend for weights.

    Supports AWS S3, MinIO, DigitalOcean Spaces, and other S3-compatible services.

    Example:
        >>> config = S3Config(
        ...     bucket="my-models",
        ...     prefix="coral/",
        ...     endpoint_url="http://localhost:9000",  # For MinIO
        ... )
        >>> store = S3Store(config)
        >>> hash_key = store.store(weight)
        >>> loaded = store.load(hash_key)
    """

    def __init__(self, config: S3Config):
        """Initialize S3 storage backend.

        Args:
            config: S3 configuration

        Raises:
            ImportError: If boto3 is not installed
            ValueError: If bucket doesn't exist and can't be created
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for S3 storage. "
                "Install with: pip install coral-ml[s3]"
            )

        self.config = config
        self._client = self._create_client()
        self._ensure_bucket_exists()

    def _create_client(self):
        """Create S3 client with configuration."""
        client_config = Config(
            max_pool_connections=self.config.max_concurrency,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )

        kwargs = {"config": client_config}

        if self.config.region:
            kwargs["region_name"] = self.config.region

        if self.config.endpoint_url:
            kwargs["endpoint_url"] = self.config.endpoint_url

        if self.config.access_key and self.config.secret_key:
            kwargs["aws_access_key_id"] = self.config.access_key
            kwargs["aws_secret_access_key"] = self.config.secret_key

        return boto3.client("s3", **kwargs)

    def _ensure_bucket_exists(self):
        """Ensure the bucket exists, create if necessary."""
        try:
            self._client.head_bucket(Bucket=self.config.bucket)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                # Bucket doesn't exist, try to create it
                try:
                    if self.config.region and self.config.region != "us-east-1":
                        self._client.create_bucket(
                            Bucket=self.config.bucket,
                            CreateBucketConfiguration={
                                "LocationConstraint": self.config.region
                            },
                        )
                    else:
                        self._client.create_bucket(Bucket=self.config.bucket)
                    logger.info(f"Created bucket: {self.config.bucket}")
                except ClientError as create_error:
                    raise ValueError(
                        f"Cannot access or create bucket {self.config.bucket}: "
                        f"{create_error}"
                    ) from create_error
            else:
                raise ValueError(
                    f"Cannot access bucket {self.config.bucket}: {e}"
                ) from e

    def _weight_key(self, hash_key: str) -> str:
        """Get S3 key for a weight."""
        return f"{self.config.prefix}weights/{hash_key}.npz"

    def _metadata_key(self, hash_key: str) -> str:
        """Get S3 key for weight metadata."""
        return f"{self.config.prefix}metadata/{hash_key}.json"

    def _delta_key(self, hash_key: str) -> str:
        """Get S3 key for a delta."""
        return f"{self.config.prefix}deltas/{hash_key}.npz"

    def _compress_data(self, data: bytes) -> Tuple[bytes, str]:
        """Compress data based on configuration."""
        if self.config.compression == "none":
            return data, "none"

        if self.config.compression == "gzip":
            import gzip

            return gzip.compress(data, compresslevel=6), "gzip"

        if self.config.compression == "lz4":
            try:
                import lz4.frame

                return lz4.frame.compress(data), "lz4"
            except ImportError:
                logger.warning("lz4 not available, falling back to gzip")
                import gzip

                return gzip.compress(data, compresslevel=6), "gzip"

        return data, "none"

    def _decompress_data(self, data: bytes, compression: str) -> bytes:
        """Decompress data based on compression type."""
        if compression == "none" or not compression:
            return data

        if compression == "gzip":
            import gzip

            return gzip.decompress(data)

        if compression == "lz4":
            import lz4.frame

            return lz4.frame.decompress(data)

        return data

    def store(self, weight: WeightTensor, hash_key: Optional[str] = None) -> str:
        """Store a weight tensor to S3."""
        if hash_key is None:
            hash_key = weight.compute_hash()

        # Serialize weight data
        buffer = io.BytesIO()
        np.savez_compressed(buffer, data=weight.data)
        weight_bytes = buffer.getvalue()

        # Compress
        compressed_bytes, compression = self._compress_data(weight_bytes)

        # Upload weight data
        self._client.put_object(
            Bucket=self.config.bucket,
            Key=self._weight_key(hash_key),
            Body=compressed_bytes,
            Metadata={"compression": compression},
        )

        # Upload metadata
        metadata_dict = weight.metadata.to_dict() if weight.metadata else {}
        metadata_dict["compression"] = compression
        self._client.put_object(
            Bucket=self.config.bucket,
            Key=self._metadata_key(hash_key),
            Body=json.dumps(metadata_dict).encode(),
            ContentType="application/json",
        )

        logger.debug(f"Stored weight {hash_key} to S3 ({len(compressed_bytes)} bytes)")
        return hash_key

    def load(self, hash_key: str) -> Optional[WeightTensor]:
        """Load a weight tensor from S3."""
        try:
            # Load weight data
            response = self._client.get_object(
                Bucket=self.config.bucket,
                Key=self._weight_key(hash_key),
            )
            compressed_bytes = response["Body"].read()
            compression = response.get("Metadata", {}).get("compression", "none")

            # Decompress
            weight_bytes = self._decompress_data(compressed_bytes, compression)

            # Deserialize
            buffer = io.BytesIO(weight_bytes)
            npz = np.load(buffer)
            data = npz["data"]

            # Load metadata
            metadata = self.get_metadata(hash_key)

            return WeightTensor(data=data, metadata=metadata)

        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                return None
            raise

    def exists(self, hash_key: str) -> bool:
        """Check if a weight exists in S3."""
        try:
            self._client.head_object(
                Bucket=self.config.bucket,
                Key=self._weight_key(hash_key),
            )
            return True
        except ClientError:
            return False

    def delete(self, hash_key: str) -> bool:
        """Delete a weight from S3."""
        if not self.exists(hash_key):
            return False

        # Delete both weight and metadata
        self._client.delete_objects(
            Bucket=self.config.bucket,
            Delete={
                "Objects": [
                    {"Key": self._weight_key(hash_key)},
                    {"Key": self._metadata_key(hash_key)},
                ]
            },
        )
        return True

    def list_weights(self) -> List[str]:
        """List all weight hashes in S3."""
        hashes = []
        prefix = f"{self.config.prefix}weights/"

        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Extract hash from key
                if key.endswith(".npz"):
                    hash_key = key[len(prefix) : -4]  # Remove prefix and .npz
                    hashes.append(hash_key)

        return hashes

    def get_metadata(self, hash_key: str) -> Optional[WeightMetadata]:
        """Get metadata for a weight without loading data."""
        try:
            response = self._client.get_object(
                Bucket=self.config.bucket,
                Key=self._metadata_key(hash_key),
            )
            metadata_dict = json.loads(response["Body"].read().decode())
            # Remove compression field before creating metadata
            metadata_dict.pop("compression", None)
            return WeightMetadata(**metadata_dict) if metadata_dict else None
        except ClientError:
            return None

    def store_batch(self, weights: Dict[str, WeightTensor]) -> Dict[str, str]:
        """Store multiple weights efficiently using concurrent uploads."""
        results = {}

        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as executor:
            futures = {
                executor.submit(self.store, weight, None): name
                for name, weight in weights.items()
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    hash_key = future.result()
                    results[name] = hash_key
                except Exception as e:
                    logger.error(f"Failed to store {name}: {e}")
                    raise

        return results

    def load_batch(self, hash_keys: List[str]) -> Dict[str, WeightTensor]:
        """Load multiple weights efficiently using concurrent downloads."""
        results = {}

        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as executor:
            futures = {
                executor.submit(self.load, hash_key): hash_key
                for hash_key in hash_keys
            }

            for future in as_completed(futures):
                hash_key = futures[future]
                try:
                    weight = future.result()
                    if weight is not None:
                        results[hash_key] = weight
                except Exception as e:
                    logger.error(f"Failed to load {hash_key}: {e}")
                    raise

        return results

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about S3 storage usage."""
        total_size = 0
        weight_count = 0
        delta_count = 0

        # Count weights
        prefix = f"{self.config.prefix}weights/"
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                total_size += obj.get("Size", 0)
                weight_count += 1

        # Count deltas
        prefix = f"{self.config.prefix}deltas/"
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                total_size += obj.get("Size", 0)
                delta_count += 1

        return {
            "backend": "s3",
            "bucket": self.config.bucket,
            "prefix": self.config.prefix,
            "weight_count": weight_count,
            "delta_count": delta_count,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "compression": self.config.compression,
        }

    def close(self):
        """Close the S3 client."""
        # boto3 clients don't need explicit cleanup
        pass

    # === Delta Storage Methods ===

    def store_delta(self, delta_hash: str, delta_data: bytes, metadata: Dict) -> None:
        """Store a delta object to S3."""
        compressed, compression = self._compress_data(delta_data)
        metadata["compression"] = compression

        self._client.put_object(
            Bucket=self.config.bucket,
            Key=self._delta_key(delta_hash),
            Body=compressed,
            Metadata={k: str(v) for k, v in metadata.items()},
        )

    def load_delta(self, delta_hash: str) -> Optional[Tuple[bytes, Dict]]:
        """Load a delta object from S3."""
        try:
            response = self._client.get_object(
                Bucket=self.config.bucket,
                Key=self._delta_key(delta_hash),
            )
            compressed = response["Body"].read()
            metadata = response.get("Metadata", {})
            compression = metadata.get("compression", "none")

            data = self._decompress_data(compressed, compression)
            return data, metadata

        except ClientError:
            return None

    def delta_exists(self, delta_hash: str) -> bool:
        """Check if a delta exists in S3."""
        try:
            self._client.head_object(
                Bucket=self.config.bucket,
                Key=self._delta_key(delta_hash),
            )
            return True
        except ClientError:
            return False

    # === Sync Methods ===

    def sync_from_local(self, local_store: WeightStore) -> Dict[str, int]:
        """Sync weights from a local store to S3.

        Args:
            local_store: Local HDF5Store or other WeightStore

        Returns:
            Dict with sync statistics
        """
        stats = {"uploaded": 0, "skipped": 0, "bytes_transferred": 0}

        local_hashes = set(local_store.list_weights())
        remote_hashes = set(self.list_weights())

        # Find weights to upload
        to_upload = local_hashes - remote_hashes

        for hash_key in to_upload:
            weight = local_store.load(hash_key)
            if weight:
                self.store(weight, hash_key)
                stats["uploaded"] += 1
                stats["bytes_transferred"] += weight.nbytes

        stats["skipped"] = len(local_hashes) - stats["uploaded"]
        return stats

    def sync_to_local(self, local_store: WeightStore) -> Dict[str, int]:
        """Sync weights from S3 to a local store.

        Args:
            local_store: Local HDF5Store or other WeightStore

        Returns:
            Dict with sync statistics
        """
        stats = {"downloaded": 0, "skipped": 0, "bytes_transferred": 0}

        local_hashes = set(local_store.list_weights())
        remote_hashes = set(self.list_weights())

        # Find weights to download
        to_download = remote_hashes - local_hashes

        for hash_key in to_download:
            weight = self.load(hash_key)
            if weight:
                local_store.store(weight, hash_key)
                stats["downloaded"] += 1
                stats["bytes_transferred"] += weight.nbytes

        stats["skipped"] = len(remote_hashes) - stats["downloaded"]
        return stats
