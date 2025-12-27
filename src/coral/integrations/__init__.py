try:
    from .pytorch import (
        CoralTrainer,
        PyTorchIntegration,
        compare_model_weights,
        create_model_from_weights,
        load_from_remote,
        load_from_repo,
        load_into_model,
        load_model_from_coral,
        save_model,
        save_model_to_coral,
    )

    __all__ = [
        "PyTorchIntegration",
        "CoralTrainer",
        "load_into_model",
        "load_from_repo",
        "load_from_remote",
        "save_model",
        "compare_model_weights",
        "create_model_from_weights",
        "save_model_to_coral",
        "load_model_from_coral",
    ]
except ImportError:
    # PyTorch not installed
    __all__ = []

import importlib.util

if importlib.util.find_spec("tensorflow") is not None:
    from .tensorflow import TensorFlowIntegration as TensorFlowIntegration  # noqa: F401

    __all__.append("TensorFlowIntegration")

# Hugging Face integration (requires huggingface-hub and safetensors)
try:
    from .huggingface import (  # noqa: F401
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
