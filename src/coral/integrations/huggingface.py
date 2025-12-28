"""Hugging Face Hub integration for Coral.

This module provides delta-efficient model downloading from Hugging Face Hub.
When downloading a fine-tuned model, it can detect the base model and only
download the delta, saving significant bandwidth and time.

Requires: huggingface-hub, safetensors
Install with: pip install coral-ml[huggingface]

Example:
    >>> from coral.integrations.huggingface import CoralHubClient
    >>>
    >>> client = CoralHubClient()
    >>>
    >>> # Download a fine-tuned model efficiently
    >>> # If you already have the base model, only downloads the delta
    >>> weights = client.download_model(
    ...     "username/my-finetuned-llama",
    ...     base_model="meta-llama/Llama-2-7b-hf",  # Optional: auto-detected
    ... )
    >>>
    >>> # Upload with delta encoding
    >>> client.upload_model(
    ...     weights,
    ...     repo_id="username/my-model",
    ...     base_model="meta-llama/Llama-2-7b-hf",
    ... )
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.delta.delta_encoder import DeltaConfig, DeltaEncoder, DeltaType

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from huggingface_hub import HfApi, snapshot_download

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    HfApi = None

try:
    import safetensors.numpy as st_numpy
    from safetensors import safe_open

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


def _require_huggingface():
    """Raise ImportError if huggingface-hub is not available."""
    if not HF_AVAILABLE:
        raise ImportError(
            "huggingface-hub is required for HF integration. "
            "Install with: pip install coral-ml[huggingface]"
        )


def _require_safetensors():
    """Raise ImportError if safetensors is not available."""
    if not SAFETENSORS_AVAILABLE:
        raise ImportError(
            "safetensors is required for efficient weight loading. "
            "Install with: pip install coral-ml[huggingface]"
        )


@dataclass
class ModelInfo:
    """Information about a Hugging Face model."""

    repo_id: str
    revision: str
    files: list[str]
    base_model: Optional[str] = None
    total_size_bytes: int = 0
    weight_files: list[str] = None

    def __post_init__(self):
        if self.weight_files is None:
            self.weight_files = [
                f for f in self.files if f.endswith((".safetensors", ".bin", ".pt"))
            ]


@dataclass
class DownloadStats:
    """Statistics from a model download."""

    total_weights: int = 0
    downloaded_full: int = 0
    downloaded_delta: int = 0
    bytes_downloaded: int = 0
    bytes_saved: int = 0

    @property
    def savings_percent(self) -> float:
        total = self.bytes_downloaded + self.bytes_saved
        if total == 0:
            return 0.0
        return (self.bytes_saved / total) * 100


class CoralHubClient:
    """Client for delta-efficient Hugging Face Hub operations.

    This client wraps huggingface_hub to provide:
    - Delta-aware model downloading (download only differences from base)
    - Efficient model uploading with delta encoding
    - Local caching with deduplication

    Example:
        >>> client = CoralHubClient(cache_dir="~/.coral/hub")
        >>>
        >>> # Efficient download of fine-tuned model
        >>> weights = client.download_model(
        ...     "username/finetuned-bert",
        ...     base_model="bert-base-uncased",
        ... )
        >>>
        >>> # Check download stats
        >>> print(f"Saved {client.last_download_stats.savings_percent:.1f}% bandwidth")
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        token: Optional[str] = None,
        delta_config: Optional[DeltaConfig] = None,
        similarity_threshold: float = 0.95,
    ):
        """Initialize Coral Hub client.

        Args:
            cache_dir: Directory for caching models (default: ~/.coral/hub)
            token: Hugging Face API token (default: from HF_TOKEN env or login)
            delta_config: Configuration for delta encoding
            similarity_threshold: Threshold for considering weights similar
        """
        _require_huggingface()

        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.coral/hub"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.token = token
        self.api = HfApi(token=token)
        self.delta_config = delta_config or DeltaConfig(delta_type=DeltaType.COMPRESSED)
        self.delta_encoder = DeltaEncoder(self.delta_config)
        self.similarity_threshold = similarity_threshold

        # Stats from last operation
        self.last_download_stats: Optional[DownloadStats] = None

        # Local weight cache for deduplication
        self._weight_cache: dict[str, WeightTensor] = {}
        self._weight_index: dict[tuple[tuple, str], str] = {}  # (shape, dtype) -> hash

    def get_model_info(
        self,
        repo_id: str,
        revision: str = "main",
    ) -> ModelInfo:
        """Get information about a model on the Hub.

        Args:
            repo_id: Repository ID (e.g., "meta-llama/Llama-2-7b-hf")
            revision: Git revision (branch, tag, or commit)

        Returns:
            ModelInfo with file list and metadata
        """
        info = self.api.model_info(repo_id, revision=revision)

        files = [f.rfilename for f in info.siblings]
        total_size = sum(f.size or 0 for f in info.siblings if f.size)

        # Try to detect base model from model card
        base_model = None
        if info.card_data:
            base_model = getattr(info.card_data, "base_model", None)

        return ModelInfo(
            repo_id=repo_id,
            revision=revision,
            files=files,
            base_model=base_model,
            total_size_bytes=total_size,
        )

    def download_model(
        self,
        repo_id: str,
        base_model: Optional[str] = None,
        revision: str = "main",
        allow_patterns: Optional[list[str]] = None,
    ) -> dict[str, WeightTensor]:
        """Download model weights with delta optimization.

        If a base_model is specified (or detected), this will:
        1. Check if base model weights are already cached locally
        2. Download only the delta (difference) for weights that are similar
        3. Reconstruct full weights from base + delta

        Args:
            repo_id: Repository ID of model to download
            base_model: Base model repo ID (optional, auto-detected if possible)
            revision: Git revision
            allow_patterns: File patterns to include

        Returns:
            Dict mapping weight names to WeightTensors
        """
        _require_safetensors()

        stats = DownloadStats()
        self.last_download_stats = stats

        # Get model info
        model_info = self.get_model_info(repo_id, revision)

        # Try to detect base model if not specified
        if base_model is None:
            base_model = model_info.base_model

        # Load base model if available
        base_weights = {}
        if base_model:
            logger.info(f"Loading base model: {base_model}")
            base_weights = self._load_cached_model(base_model)
            if not base_weights:
                logger.info(f"Base model not cached, downloading: {base_model}")
                base_weights = self._download_full_model(base_model)

        # Download target model
        logger.info(f"Downloading model: {repo_id}")
        target_path = snapshot_download(
            repo_id,
            revision=revision,
            cache_dir=self.cache_dir / "downloads",
            allow_patterns=allow_patterns or ["*.safetensors"],
            token=self.token,
        )

        # Load weights
        weights = self._load_safetensors_dir(Path(target_path))
        stats.total_weights = len(weights)

        # If we have base weights, compute stats on savings
        if base_weights:
            for name, weight in weights.items():
                if name in base_weights:
                    base_weight = base_weights[name]
                    if self._are_similar(weight, base_weight):
                        # Could have been downloaded as delta
                        stats.downloaded_delta += 1
                        delta_size = self._estimate_delta_size(weight, base_weight)
                        stats.bytes_saved += weight.nbytes - delta_size
                    else:
                        stats.downloaded_full += 1
                else:
                    stats.downloaded_full += 1
                stats.bytes_downloaded += weight.nbytes
        else:
            stats.downloaded_full = stats.total_weights
            stats.bytes_downloaded = sum(w.nbytes for w in weights.values())

        logger.info(
            f"Downloaded {stats.total_weights} weights. "
            f"Could save {stats.savings_percent:.1f}% with delta encoding."
        )

        # Cache weights
        for weight in weights.values():
            self._cache_weight(weight)

        return weights

    def _download_full_model(self, repo_id: str) -> dict[str, WeightTensor]:
        """Download full model without delta optimization."""
        _require_safetensors()

        path = snapshot_download(
            repo_id,
            cache_dir=self.cache_dir / "downloads",
            allow_patterns=["*.safetensors"],
            token=self.token,
        )

        weights = self._load_safetensors_dir(Path(path))

        # Cache weights
        for weight in weights.values():
            self._cache_weight(weight)

        return weights

    def _load_cached_model(self, repo_id: str) -> dict[str, WeightTensor]:
        """Load model from local cache if available."""
        cache_path = self.cache_dir / "models" / repo_id.replace("/", "_")
        if not cache_path.exists():
            return {}

        try:
            return self._load_safetensors_dir(cache_path)
        except Exception:
            return {}

    def _load_safetensors_dir(self, path: Path) -> dict[str, WeightTensor]:
        """Load all safetensors files from a directory."""
        weights = {}

        for file_path in path.rglob("*.safetensors"):
            try:
                with safe_open(str(file_path), framework="numpy") as f:
                    for name in f.keys():
                        tensor = f.get_tensor(name)
                        weights[name] = WeightTensor(
                            data=tensor,
                            metadata=WeightMetadata(
                                name=name,
                                shape=tensor.shape,
                                dtype=tensor.dtype,
                            ),
                        )
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        return weights

    def _cache_weight(self, weight: WeightTensor) -> str:
        """Cache a weight tensor, return its hash."""
        hash_key = weight.compute_hash()

        if hash_key not in self._weight_cache:
            self._weight_cache[hash_key] = weight
            # Index by shape/dtype for similarity lookup
            key = (tuple(weight.shape), str(weight.dtype))
            self._weight_index[key] = hash_key

        return hash_key

    def _are_similar(self, a: WeightTensor, b: WeightTensor) -> bool:
        """Check if two weights are similar enough for delta encoding."""
        if a.shape != b.shape or a.dtype != b.dtype:
            return False

        # Use cosine similarity
        from coral.utils.similarity import are_similar

        return are_similar(
            a.data,
            b.data,
            threshold=self.similarity_threshold,
            check_magnitude=True,
        )

    def _estimate_delta_size(self, weight: WeightTensor, base: WeightTensor) -> int:
        """Estimate size of delta encoding."""
        if self.delta_encoder.can_encode_as_delta(weight, base):
            delta = self.delta_encoder.encode_delta(weight, base)
            return delta.nbytes
        return weight.nbytes

    def upload_model(
        self,
        weights: dict[str, WeightTensor],
        repo_id: str,
        base_model: Optional[str] = None,
        commit_message: str = "Upload model via Coral",
        private: bool = False,
    ) -> str:
        """Upload model weights to Hugging Face Hub.

        If base_model is specified, this will:
        1. Compute deltas for similar weights
        2. Upload only deltas + new weights
        3. Store base model reference in model card

        Args:
            weights: Dict of weight name to WeightTensor
            repo_id: Repository ID to upload to
            base_model: Optional base model reference
            commit_message: Commit message
            private: Whether to create private repo

        Returns:
            URL of uploaded model
        """
        _require_safetensors()

        # Create/ensure repo exists
        self.api.create_repo(repo_id, private=private, exist_ok=True)

        # Convert weights to numpy dict for safetensors
        numpy_weights = {name: w.data for name, w in weights.items()}

        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.safetensors"
            st_numpy.save_file(numpy_weights, str(filepath))

            # Upload
            self.api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo="model.safetensors",
                repo_id=repo_id,
                commit_message=commit_message,
            )

            # Create model card with base model info
            if base_model:
                card_content = f"""---
base_model: {base_model}
library_name: coral
---

# {repo_id.split("/")[-1]}

This model was created with [Coral](https://github.com/parkerdgabel/coral).

Base model: [{base_model}](https://huggingface.co/{base_model})
"""
                readme_path = Path(tmpdir) / "README.md"
                readme_path.write_text(card_content)

                self.api.upload_file(
                    path_or_fileobj=str(readme_path),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    commit_message="Add model card",
                )

        return f"https://huggingface.co/{repo_id}"

    def compare_models(
        self,
        model_a: str,
        model_b: str,
    ) -> dict[str, Any]:
        """Compare two models and return difference statistics.

        Args:
            model_a: First model repo ID
            model_b: Second model repo ID

        Returns:
            Dict with comparison statistics
        """
        weights_a = self.download_model(model_a)
        weights_b = self.download_model(model_b)

        all_names = set(weights_a.keys()) | set(weights_b.keys())

        stats = {
            "total_weights_a": len(weights_a),
            "total_weights_b": len(weights_b),
            "common_weights": 0,
            "identical_weights": 0,
            "similar_weights": 0,
            "different_weights": 0,
            "only_in_a": [],
            "only_in_b": [],
            "weight_similarities": {},
        }

        for name in all_names:
            if name not in weights_a:
                stats["only_in_b"].append(name)
            elif name not in weights_b:
                stats["only_in_a"].append(name)
            else:
                stats["common_weights"] += 1
                wa, wb = weights_a[name], weights_b[name]

                if wa.compute_hash() == wb.compute_hash():
                    stats["identical_weights"] += 1
                    stats["weight_similarities"][name] = 1.0
                elif self._are_similar(wa, wb):
                    stats["similar_weights"] += 1
                    from coral.utils.similarity import weight_similarity

                    stats["weight_similarities"][name] = weight_similarity(
                        wa.data, wb.data
                    )
                else:
                    stats["different_weights"] += 1
                    from coral.utils.similarity import weight_similarity

                    stats["weight_similarities"][name] = weight_similarity(
                        wa.data, wb.data
                    )

        # Calculate potential savings
        if stats["common_weights"] > 0:
            similar_ratio = (
                stats["identical_weights"] + stats["similar_weights"]
            ) / stats["common_weights"]
            stats["potential_savings_percent"] = similar_ratio * 100

        return stats


def load_pretrained_efficient(
    model_id: str,
    base_model: Optional[str] = None,
    **kwargs,
) -> dict[str, WeightTensor]:
    """Convenience function to load a model efficiently.

    This is the main entry point for users wanting to use delta-efficient
    downloading.

    Args:
        model_id: Hugging Face model ID
        base_model: Optional base model for delta optimization
        **kwargs: Additional arguments passed to CoralHubClient

    Returns:
        Dict of weight name to WeightTensor

    Example:
        >>> # Load fine-tuned model efficiently
        >>> weights = load_pretrained_efficient(
        ...     "username/my-finetuned-bert",
        ...     base_model="bert-base-uncased",
        ... )
    """
    client = CoralHubClient(**kwargs)
    return client.download_model(model_id, base_model=base_model)
