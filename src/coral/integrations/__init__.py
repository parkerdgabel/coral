try:
    from .pytorch import CoralTrainer, PyTorchIntegration

    __all__ = ["PyTorchIntegration", "CoralTrainer"]
except ImportError:
    # PyTorch not installed
    __all__ = []

import importlib.util

if importlib.util.find_spec("tensorflow") is not None:
    from .tensorflow import TensorFlowIntegration as TensorFlowIntegration  # noqa: F401

    __all__.append("TensorFlowIntegration")

# Hugging Face integration (requires huggingface-hub and safetensors)
try:
    from .huggingface import (
        CoralHubClient,
        DownloadStats,
        ModelInfo,
        load_pretrained_efficient,
    )

    __all__.extend([
        "CoralHubClient",
        "ModelInfo",
        "DownloadStats",
        "load_pretrained_efficient",
    ])
except ImportError:
    # huggingface-hub not installed
    pass
