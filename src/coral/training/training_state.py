import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    try:
        import torch  # type: ignore
    except ImportError:
        torch = None


@dataclass
class TrainingState:
    """Represents the complete state of a training run."""

    epoch: int
    global_step: int
    learning_rate: float
    loss: float
    metrics: Dict[str, float] = field(default_factory=dict)
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    random_state: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)

    # Training configuration
    batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    max_epochs: Optional[int] = None
    max_steps: Optional[int] = None

    # Additional metadata
    model_name: Optional[str] = None
    dataset_name: Optional[str] = None
    experiment_name: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""

        def serialize_nested_dict(obj: Any) -> Any:
            """Recursively serialize PyTorch tensors in nested structures."""
            if obj is None:
                return None
            elif isinstance(obj, dict):
                return {k: serialize_nested_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_nested_dict(item) for item in obj]
            else:
                try:
                    import torch

                    if hasattr(torch, "Tensor") and isinstance(obj, torch.Tensor):
                        return {
                            "__tensor__": obj.cpu().numpy().tolist(),
                            "__dtype__": str(obj.dtype),
                        }
                    else:
                        return obj
                except ImportError:
                    return obj

        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
            "metrics": self.metrics,
            "optimizer_state": serialize_nested_dict(self.optimizer_state),
            "scheduler_state": serialize_nested_dict(self.scheduler_state),
            "random_state": serialize_nested_dict(self.random_state),
            "timestamp": self.timestamp.isoformat(),
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_epochs": self.max_epochs,
            "max_steps": self.max_steps,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "experiment_name": self.experiment_name,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingState":
        """Create from dictionary."""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        def deserialize_nested_dict(obj: Any) -> Any:
            """Recursively deserialize PyTorch tensors in nested structures."""
            if obj is None:
                return None
            elif isinstance(obj, dict):
                if "__tensor__" in obj:
                    # This is a serialized tensor
                    try:
                        import numpy as np
                        import torch

                        return torch.from_numpy(np.array(obj["__tensor__"]))
                    except ImportError:
                        return obj
                else:
                    return {k: deserialize_nested_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deserialize_nested_dict(item) for item in obj]
            else:
                return obj

        # Deserialize nested structures
        for key in ["optimizer_state", "scheduler_state", "random_state"]:
            if key in data:
                data[key] = deserialize_nested_dict(data[key])

        return cls(**data)

    def save(self, path: str) -> None:
        """Save state to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingState":
        """Load state from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def update_metrics(self, **metrics: float) -> None:
        """Update training metrics."""
        self.metrics.update(metrics)

    def format_summary(self) -> str:
        """Format a summary of the training state."""
        summary = [
            f"Epoch: {self.epoch}",
            f"Step: {self.global_step}",
            f"Learning Rate: {self.learning_rate:.2e}",
            f"Loss: {self.loss:.4f}",
        ]

        if self.metrics:
            for key, value in sorted(self.metrics.items()):
                summary.append(f"{key}: {value:.4f}")

        return " | ".join(summary)
