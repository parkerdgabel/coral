"""JSON utilities for handling numpy types."""

import json
from typing import Any

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types.

    This encoder converts numpy types to their Python equivalents:
    - numpy integers -> Python int
    - numpy floats -> Python float
    - numpy arrays -> Python lists
    - numpy dtypes -> string representation
    - numpy booleans -> Python bool
    """

    def default(self, obj: Any) -> Any:
        """Convert numpy types to JSON-serializable Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.dtype):
            return str(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.void, np.complexfloating)):
            return str(obj)
        return super().default(obj)


def dumps_numpy(obj: Any, **kwargs) -> str:
    """Serialize obj to a JSON string, handling numpy types.

    Args:
        obj: Object to serialize
        **kwargs: Additional arguments to pass to json.dumps

    Returns:
        JSON string
    """
    return json.dumps(obj, cls=NumpyJSONEncoder, **kwargs)


def dump_numpy(obj: Any, fp, **kwargs) -> None:
    """Serialize obj as JSON to a file, handling numpy types.

    Args:
        obj: Object to serialize
        fp: File-like object to write to
        **kwargs: Additional arguments to pass to json.dump
    """
    json.dump(obj, fp, cls=NumpyJSONEncoder, **kwargs)
