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
