"""Model registry publishing implementation.

Provides publishing capabilities to Hugging Face Hub, MLflow, and local registries.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from coral.version_control.repository import Repository

logger = logging.getLogger(__name__)


class RegistryType(Enum):
    """Supported model registries."""

    HUGGINGFACE = "huggingface"
    MLFLOW = "mlflow"
    LOCAL = "local"


@dataclass
class PublishResult:
    """Result of a publish operation."""

    success: bool
    registry: RegistryType
    model_name: str
    version: str | None = None
    url: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    published_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "registry": self.registry.value,
            "model_name": self.model_name,
            "version": self.version,
            "url": self.url,
            "error": self.error,
            "metadata": self.metadata,
            "published_at": self.published_at.isoformat(),
        }


class ModelPublisher:
    """Publishes models to various registries.

    Supports:
    - Hugging Face Hub
    - MLflow Model Registry
    - Local file registry

    Example:
        >>> from coral.version_control.repository import Repository
        >>> from coral.registry import ModelPublisher
        >>>
        >>> repo = Repository("./my-model")
        >>> publisher = ModelPublisher(repo)
        >>>
        >>> # Publish to Hugging Face Hub
        >>> result = publisher.publish_huggingface(
        ...     repo_id="my-org/my-model",
        ...     commit_ref="abc123",
        ...     private=False,
        ... )
        >>> print(f"Published: {result.url}")
        >>>
        >>> # Publish to MLflow
        >>> result = publisher.publish_mlflow(
        ...     model_name="my-model",
        ...     tracking_uri="http://mlflow:5000",
        ... )
    """

    def __init__(self, repo: Repository):
        """Initialize model publisher.

        Args:
            repo: Coral Repository instance
        """
        self.repo = repo
        self.registry_dir = repo.coral_dir / "registry"
        self.registry_dir.mkdir(exist_ok=True)
        self._history: list[PublishResult] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load publish history from disk."""
        history_file = self.registry_dir / "history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                for item in data:
                    item["registry"] = RegistryType(item["registry"])
                    item["published_at"] = datetime.fromisoformat(item["published_at"])
                    self._history.append(PublishResult(**item))
            except Exception as e:
                logger.warning(f"Failed to load publish history: {e}")

    def _save_history(self) -> None:
        """Save publish history to disk."""
        history_file = self.registry_dir / "history.json"
        with open(history_file, "w") as f:
            json.dump([r.to_dict() for r in self._history], f, indent=2)

    def _record_result(self, result: PublishResult) -> None:
        """Record a publish result."""
        self._history.append(result)
        self._save_history()

    def _get_weights_for_commit(self, commit_ref: str | None = None) -> dict[str, Any]:
        """Get weights from a commit as numpy arrays."""
        weights = self.repo.get_all_weights(commit_ref)
        return {name: w.data for name, w in weights.items()}

    def _generate_model_card(
        self,
        model_name: str,
        description: str | None = None,
        base_model: str | None = None,
        metrics: dict[str, float] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Generate a model card in markdown format."""
        lines = ["---"]

        # YAML front matter
        if tags:
            lines.append(f"tags: {json.dumps(tags)}")
        if base_model:
            lines.append(f"base_model: {base_model}")
        lines.append("library_name: coral")

        if metrics:
            lines.append("model-index:")
            lines.append(f"  - name: {model_name}")
            lines.append("    results:")
            for name, value in metrics.items():
                lines.append(f"      - task: {name}")
                lines.append(f"        value: {value}")

        lines.append("---")
        lines.append("")
        lines.append(f"# {model_name}")
        lines.append("")

        if description:
            lines.append(description)
            lines.append("")

        lines.append("## Model Details")
        lines.append("")
        lines.append(
            "This model was versioned and published using "
            "[Coral](https://github.com/parkerdgabel/coral)."
        )
        lines.append("")

        if base_model:
            lines.append(
                f"**Base Model**: [{base_model}](https://huggingface.co/{base_model})"
            )
            lines.append("")

        if metrics:
            lines.append("## Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for name, value in metrics.items():
                lines.append(f"| {name} | {value:.4f} |")
            lines.append("")

        return "\n".join(lines)

    def publish_huggingface(
        self,
        repo_id: str,
        commit_ref: str | None = None,
        private: bool = False,
        description: str | None = None,
        base_model: str | None = None,
        metrics: dict[str, float] | None = None,
        tags: list[str] | None = None,
        token: str | None = None,
    ) -> PublishResult:
        """Publish model to Hugging Face Hub.

        Args:
            repo_id: Hugging Face repository ID (org/name or user/name)
            commit_ref: Coral commit reference (default: HEAD)
            private: Create private repository
            description: Model description for model card
            base_model: Base model this was fine-tuned from
            metrics: Model metrics to include
            tags: Tags for the model
            token: Hugging Face API token (default: from env or login)

        Returns:
            PublishResult with status and URL
        """
        try:
            import safetensors.numpy as st_numpy
            from huggingface_hub import HfApi
        except ImportError:
            return PublishResult(
                success=False,
                registry=RegistryType.HUGGINGFACE,
                model_name=repo_id,
                error="huggingface-hub and safetensors required. "
                "Install with: pip install huggingface-hub safetensors",
            )

        try:
            api = HfApi(token=token)

            # Get weights
            weights = self._get_weights_for_commit(commit_ref)
            if not weights:
                return PublishResult(
                    success=False,
                    registry=RegistryType.HUGGINGFACE,
                    model_name=repo_id,
                    error="No weights found for the specified commit",
                )

            # Create repo
            api.create_repo(repo_id, private=private, exist_ok=True)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)

                # Save weights as safetensors
                weights_file = tmppath / "model.safetensors"
                st_numpy.save_file(weights, str(weights_file))

                # Generate model card
                model_card = self._generate_model_card(
                    model_name=repo_id.split("/")[-1],
                    description=description,
                    base_model=base_model,
                    metrics=metrics,
                    tags=tags,
                )
                readme_file = tmppath / "README.md"
                readme_file.write_text(model_card)

                # Save Coral metadata
                coral_meta = {
                    "coral_version": "1.0.0",
                    "commit_ref": commit_ref or "HEAD",
                    "published_at": datetime.now().isoformat(),
                    "weight_count": len(weights),
                }
                meta_file = tmppath / "coral_metadata.json"
                with open(meta_file, "w") as f:
                    json.dump(coral_meta, f, indent=2)

                # Upload files
                commit_msg = f"Publish from Coral (commit: {commit_ref or 'HEAD'})"
                api.upload_folder(
                    folder_path=str(tmppath),
                    repo_id=repo_id,
                    commit_message=commit_msg,
                )

            result = PublishResult(
                success=True,
                registry=RegistryType.HUGGINGFACE,
                model_name=repo_id,
                url=f"https://huggingface.co/{repo_id}",
                metadata={
                    "commit_ref": commit_ref,
                    "weight_count": len(weights),
                    "private": private,
                },
            )
            self._record_result(result)
            logger.info(f"Published to Hugging Face Hub: {result.url}")
            return result

        except Exception as e:
            result = PublishResult(
                success=False,
                registry=RegistryType.HUGGINGFACE,
                model_name=repo_id,
                error=str(e),
            )
            self._record_result(result)
            return result

    def publish_mlflow(
        self,
        model_name: str,
        commit_ref: str | None = None,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        description: str | None = None,
        tags: dict[str, str] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> PublishResult:
        """Publish model to MLflow Model Registry.

        Args:
            model_name: Name for the model in the registry
            commit_ref: Coral commit reference (default: HEAD)
            tracking_uri: MLflow tracking URI (default: from env)
            experiment_name: MLflow experiment name
            description: Model description
            tags: Tags for the model
            metrics: Metrics to log with the run

        Returns:
            PublishResult with status and version
        """
        try:
            import mlflow
            import mlflow.pyfunc
        except ImportError:
            return PublishResult(
                success=False,
                registry=RegistryType.MLFLOW,
                model_name=model_name,
                error="mlflow required. Install with: pip install mlflow",
            )

        try:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)

            if experiment_name:
                mlflow.set_experiment(experiment_name)

            # Get weights
            weights = self._get_weights_for_commit(commit_ref)
            if not weights:
                return PublishResult(
                    success=False,
                    registry=RegistryType.MLFLOW,
                    model_name=model_name,
                    error="No weights found for the specified commit",
                )

            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)

                # Save weights
                import numpy as np

                weights_file = tmppath / "weights.npz"
                np.savez(weights_file, **weights)

                # Create MLflow run
                with mlflow.start_run() as run:
                    # Log parameters
                    mlflow.log_param("coral_commit", commit_ref or "HEAD")
                    mlflow.log_param("weight_count", len(weights))

                    # Log metrics
                    if metrics:
                        for name, value in metrics.items():
                            mlflow.log_metric(name, value)

                    # Log tags
                    if tags:
                        for key, value in tags.items():
                            mlflow.set_tag(key, value)

                    mlflow.set_tag("coral.version", "1.0.0")

                    # Log model artifact
                    mlflow.log_artifact(str(weights_file))

                    # Register model
                    model_uri = f"runs:/{run.info.run_id}/weights.npz"
                    result_info = mlflow.register_model(model_uri, model_name)

                    if description:
                        from mlflow.tracking import MlflowClient

                        client = MlflowClient()
                        client.update_model_version(
                            name=model_name,
                            version=result_info.version,
                            description=description,
                        )

            result = PublishResult(
                success=True,
                registry=RegistryType.MLFLOW,
                model_name=model_name,
                version=result_info.version,
                url=f"{mlflow.get_tracking_uri()}/#/models/{model_name}",
                metadata={
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "commit_ref": commit_ref,
                    "weight_count": len(weights),
                },
            )
            self._record_result(result)
            logger.info(f"Published to MLflow: {model_name} v{result_info.version}")
            return result

        except Exception as e:
            result = PublishResult(
                success=False,
                registry=RegistryType.MLFLOW,
                model_name=model_name,
                error=str(e),
            )
            self._record_result(result)
            return result

    def publish_local(
        self,
        output_path: str | Path,
        commit_ref: str | None = None,
        format: str = "safetensors",
        include_metadata: bool = True,
    ) -> PublishResult:
        """Export model to local directory.

        Args:
            output_path: Output directory path
            commit_ref: Coral commit reference (default: HEAD)
            format: Output format ('safetensors', 'npz', 'pt')
            include_metadata: Include metadata files

        Returns:
            PublishResult with status
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Get weights
            weights = self._get_weights_for_commit(commit_ref)
            if not weights:
                return PublishResult(
                    success=False,
                    registry=RegistryType.LOCAL,
                    model_name=str(output_path),
                    error="No weights found for the specified commit",
                )

            # Save weights in requested format
            if format == "safetensors":
                try:
                    import safetensors.numpy as st_numpy

                    weights_file = output_path / "model.safetensors"
                    st_numpy.save_file(weights, str(weights_file))
                except ImportError:
                    return PublishResult(
                        success=False,
                        registry=RegistryType.LOCAL,
                        model_name=str(output_path),
                        error="safetensors required for this format",
                    )

            elif format == "npz":
                import numpy as np

                weights_file = output_path / "model.npz"
                np.savez(weights_file, **weights)

            elif format == "pt":
                try:
                    import torch

                    weights_file = output_path / "model.pt"
                    torch_weights = {k: torch.from_numpy(v) for k, v in weights.items()}
                    torch.save(torch_weights, weights_file)
                except ImportError:
                    return PublishResult(
                        success=False,
                        registry=RegistryType.LOCAL,
                        model_name=str(output_path),
                        error="torch required for this format",
                    )
            else:
                return PublishResult(
                    success=False,
                    registry=RegistryType.LOCAL,
                    model_name=str(output_path),
                    error=f"Unsupported format: {format}",
                )

            # Save metadata
            if include_metadata:
                metadata = {
                    "coral_version": "1.0.0",
                    "commit_ref": commit_ref or "HEAD",
                    "format": format,
                    "weight_count": len(weights),
                    "weight_names": list(weights.keys()),
                    "exported_at": datetime.now().isoformat(),
                }
                meta_file = output_path / "coral_metadata.json"
                with open(meta_file, "w") as f:
                    json.dump(metadata, f, indent=2)

            result = PublishResult(
                success=True,
                registry=RegistryType.LOCAL,
                model_name=str(output_path),
                url=f"file://{output_path.absolute()}",
                metadata={
                    "commit_ref": commit_ref,
                    "format": format,
                    "weight_count": len(weights),
                },
            )
            self._record_result(result)
            logger.info(f"Exported to: {output_path}")
            return result

        except Exception as e:
            result = PublishResult(
                success=False,
                registry=RegistryType.LOCAL,
                model_name=str(output_path),
                error=str(e),
            )
            self._record_result(result)
            return result

    def get_history(
        self,
        registry: RegistryType | None = None,
        success_only: bool = False,
        limit: int = 50,
    ) -> list[PublishResult]:
        """Get publish history.

        Args:
            registry: Filter by registry type
            success_only: Only return successful publishes
            limit: Maximum number to return

        Returns:
            List of PublishResults
        """
        results = self._history.copy()

        if registry:
            results = [r for r in results if r.registry == registry]

        if success_only:
            results = [r for r in results if r.success]

        # Sort by date, newest first
        results.sort(key=lambda r: r.published_at, reverse=True)

        return results[:limit]

    def get_latest(
        self,
        model_name: str,
        registry: RegistryType | None = None,
    ) -> PublishResult | None:
        """Get the latest successful publish for a model.

        Args:
            model_name: Model name to search for
            registry: Optional registry type filter

        Returns:
            Latest PublishResult or None
        """
        for result in self.get_history(registry=registry, success_only=True):
            if result.model_name == model_name:
                return result
        return None
