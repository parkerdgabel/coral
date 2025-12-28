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
        Checkpointer,
        CoralTrainer,
        PyTorchIntegration,
        StreamingWeightLoader,
        compare_model_weights,
        create_model_from_weights,
        load,
        load_from_remote,
        load_from_repo,
        load_into_model,
        load_model_from_coral,
        save,
        save_model,
        save_model_to_coral,
        stream_load_model,
    )

    __all__.extend(
        [
            "PyTorchIntegration",
            "CoralTrainer",
            "Checkpointer",
            "load",
            "save",
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
        ]
    )
except ImportError:
    # PyTorch not installed
    pass

import importlib.util

# TensorFlow/Keras integration (optional)
if importlib.util.find_spec("tensorflow") is not None:
    try:
        from .tensorflow import TensorFlowIntegration  # noqa: F401
        from .tensorflow import (  # noqa: F401
            compare_model_weights as tf_compare_model_weights,
        )
        from .tensorflow import load as tf_load  # noqa: F401
        from .tensorflow import (  # noqa: F401
            load_into_model as tf_load_into_model,
        )
        from .tensorflow import (  # noqa: F401
            load_model_from_coral as tf_load_model_from_coral,
        )
        from .tensorflow import save as tf_save  # noqa: F401
        from .tensorflow import (  # noqa: F401
            save_model as tf_save_model,
        )

        __all__.extend(
            [
                "TensorFlowIntegration",
                "tf_save",
                "tf_load",
                "tf_save_model",
                "tf_load_model_from_coral",
                "tf_load_into_model",
                "tf_compare_model_weights",
            ]
        )
    except ImportError:
        # TensorFlow available but import failed
        pass

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
    from .lightning import CoralCallback, CoralLightningCallback  # noqa: F401

    __all__.extend(["CoralCallback", "CoralLightningCallback"])
except ImportError:
    # pytorch-lightning not installed
    pass

# HuggingFace Transformers Trainer integration
try:
    from .hf_trainer import CoralHFCallback, CoralTrainerCallback  # noqa: F401

    __all__.extend(["CoralTrainerCallback", "CoralHFCallback"])
except ImportError:
    # transformers not installed
    pass
