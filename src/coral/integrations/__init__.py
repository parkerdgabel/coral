# Experiment tracking bridges (always available, dependencies checked at runtime)
from .experiment_bridge import ExperimentBridge  # noqa: F401

__all__ = ["ExperimentBridge"]

# MLflow bridge (optional)
try:
    from .mlflow_bridge import MLflowBridge  # noqa: F401

    __all__.append("MLflowBridge")
except ImportError:
    pass

# W&B bridge (optional)
try:
    from .wandb_bridge import WandbBridge  # noqa: F401

    __all__.append("WandbBridge")
except ImportError:
    pass

# PyTorch integrations
try:
    from .pytorch import (  # noqa: F401
        CoralTrainer,
        PyTorchIntegration,
        StreamingWeightLoader,
        compare_model_weights,
        create_model_from_weights,
        load_from_remote,
        load_from_repo,
        load_into_model,
        load_model_from_coral,
        save_model,
        save_model_to_coral,
        stream_load_model,
    )

    __all__.extend([
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
        "StreamingWeightLoader",
        "stream_load_model",
    ])
except ImportError:
    # PyTorch not installed
    pass

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

    __all__.extend(
        [
            "CoralHubClient",
            "ModelInfo",
            "DownloadStats",
            "load_pretrained_efficient",
        ]
    )
except ImportError:
    # huggingface-hub not installed
    pass

# PyTorch Lightning integration
try:
    from .lightning import CoralCallback  # noqa: F401

    __all__.append("CoralCallback")
except ImportError:
    # pytorch-lightning not installed
    pass

# HuggingFace Transformers Trainer integration
try:
    from .hf_trainer import CoralTrainerCallback  # noqa: F401

    __all__.append("CoralTrainerCallback")
except ImportError:
    # transformers not installed
    pass
