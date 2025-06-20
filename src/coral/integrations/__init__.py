import importlib.util
from typing import List

__all__: List[str] = []

try:
    from .pytorch import CoralTrainer, PyTorchIntegration  # noqa: F401

    __all__.extend(["PyTorchIntegration", "CoralTrainer"])
except ImportError:
    # PyTorch not installed
    pass

# Check for TensorFlow availability
if importlib.util.find_spec("tensorflow") is not None:
    try:
        from .tensorflow import TensorFlowIntegration  # type: ignore # noqa: F401

        __all__.append("TensorFlowIntegration")
    except ImportError:
        # TensorFlow integration not available
        pass
