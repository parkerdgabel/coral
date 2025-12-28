# Coral API Design Review

This document identifies API design issues in the Coral codebase, following Pythonic principles and best practices.

> **Status:** All issues identified below have been addressed in commit `069d8bc`.

## Critical Issues

### 1. Duplicate Enum Definitions (DRY Violation) ✅ FIXED

**Location:** `delta/delta_encoder.py:72` and `config/schema.py:22`

Two separate enums define the same delta encoding strategies:
- `DeltaType` in delta_encoder.py (used in runtime code)
- `DeltaTypeEnum` in config/schema.py (used in configuration)

```python
# delta/delta_encoder.py
class DeltaType(Enum):
    COMPRESSED = "compressed"
    XOR_FLOAT32 = "xor_float32"
    # ...

# config/schema.py
class DeltaTypeEnum(str, Enum):
    COMPRESSED = "compressed"
    XOR_FLOAT32 = "xor_float32"
    # ...
```

**Pythonic Fix:** Use a single enum definition and import it where needed.

---

### 2. Inconsistent Type for delta_type Configuration ✅ FIXED

**Locations:**
- `config/schema.py:84` - `delta_type: str = "compressed"`
- `delta/delta_encoder.py:118` - `delta_type: DeltaType = DeltaType.FLOAT32_RAW`

The same configuration option uses different types in different contexts:
- `CoreConfig.delta_type` is a `str`
- `DeltaConfig.delta_type` is a `DeltaType` enum

**Impact:** Forces type conversion throughout the codebase:
```python
# repository.py:107
delta_config.delta_type = DeltaType(self._coral_config.core.delta_type)
```

**Pythonic Fix:** Use the enum type consistently. Configuration should validate and convert on load.

---

### 3. Deprecated Parameters Without Deprecation Warnings ✅ FIXED

**Location:** `core/deduplicator.py:92-103`

```python
def __init__(
    self,
    # ...
    enable_lsh: bool = False,  # Deprecated but no warning
    lsh_config: Optional[LSHConfig] = None,  # Deprecated but no warning
    # ...
    use_unified_index: bool = True,  # Preferred replacement
):
```

Parameters are marked deprecated in docstrings but not programmatically warned.

**Pythonic Fix:** Use `warnings.warn()` with `DeprecationWarning`:
```python
if enable_lsh:
    warnings.warn(
        "enable_lsh is deprecated, use use_unified_index=True instead",
        DeprecationWarning,
        stacklevel=2
    )
```

---

## API Consistency Issues

### 4. Inconsistent from_dict() Signatures ✅ FIXED

**Location:** `config/schema.py:453`

`RemoteConfig.from_dict()` has a unique signature:
```python
# All other classes:
@classmethod
def from_dict(cls, data: dict[str, Any]) -> SomeConfig:

# RemoteConfig is different:
@classmethod
def from_dict(cls, name: str, data: dict[str, Any]) -> RemoteConfig:
```

**Pythonic Fix:** Include `name` in the data dict for consistency, or use a factory function.

---

### 5. Multiple Redundant APIs for Same Operation ✅ FIXED

**Location:** `integrations/pytorch.py`

Four different ways to load a model:
- `load()` - unified function (lines 1288-1389)
- `load_from_repo()` - repository-specific (lines 975-1029)
- `load_from_remote()` - remote-specific (lines 1032-1088)
- `load_model_from_coral()` - legacy function (lines 859-875)

**Pythonic Principle:** "There should be one-- and preferably only one --obvious way to do it."

**Recommendation:** Deprecate the specific functions in favor of the unified `load()` API.

---

### 6. Inconsistent Return Types for Similar Operations ✅ FIXED

**Locations:**
- `CoralTrainer.step()` returns `None` (line 275)
- `Checkpointer.step()` returns `Optional[str]` (line 603)

Both perform the same conceptual operation but have different return signatures.

**Pythonic Fix:** Standardize return types across similar operations.

---

### 7. Boolean Return vs Exception Anti-Pattern ✅ FIXED

**Location:** `integrations/pytorch.py:859-875`

```python
def load_model_from_coral(...) -> bool:
    # ...
    if not weights:
        return False  # Silent failure
    # ...
    return True
```

**Pythonic Principle:** Use exceptions for exceptional conditions, not boolean returns.

**Better:**
```python
def load_model_from_coral(...) -> dict[str, WeightTensor]:
    weights = repository.get_all_weights(commit_ref)
    if not weights:
        raise ValueError("No weights found in repository")
    PyTorchIntegration.weights_to_model(weights, model)
    return weights
```

---

### 8. Missing Type Annotations ✅ FIXED

**Location:** `core/deduplicator.py:509`

```python
def set_store(self, store) -> None:  # Missing type hint for 'store'
```

Should be:
```python
def set_store(self, store: WeightStore) -> None:
```

---

## Naming Inconsistencies

### 9. Inconsistent Method Naming Patterns

**Locations:** Various

Mixed naming patterns for similar operations:
```python
# to/from pattern:
save_model_to_coral()
load_model_from_coral()

# Just verb pattern:
save_model()
load_from_repo()  # Inconsistent - uses 'from' without 'to' counterpart
```

**Pythonic Fix:** Choose one pattern and apply consistently.

---

### 10. Inconsistent Parameter Naming

**Locations:**
- `Deduplicator`: uses `similarity_threshold`, `magnitude_tolerance`
- `SimHashConfig`: uses `similarity_threshold` (same meaning but different context)
- `HuggingFaceConfig`: uses `similarity_threshold` (yet another context)

**Issue:** Same parameter name used for different purposes across configs.

---

## Constructor Design Issues

### 11. Inconsistent Path Parameter Types ✅ FIXED

**Locations:**
- `Repository.__init__(path: Path, ...)` - requires Path
- `HDF5Store.__init__(filepath: str, ...)` - requires str
- `load()` function accepts `Union[str, Path, Repository]`

**Pythonic Fix:** Accept both str and Path everywhere using `os.PathLike`:
```python
def __init__(self, path: Union[str, os.PathLike], ...):
    self.path = Path(path)
```

---

### 12. Optional Parameters Before Required in dataclass

**Location:** `config/schema.py:424-436`

```python
@dataclass
class RemoteConfig:
    name: str  # Required
    url: str   # Required
    backend: str = "s3"  # Optional with default
    # ...
```

This is correct, but compared to other config classes that have all defaults, it's inconsistent.

---

## Abstract Interface Issues

### 13. Inconsistent Error Handling in Abstract Interface

**Location:** `storage/weight_store.py`

```python
@abstractmethod
def load(self, hash_key: str) -> Optional[WeightTensor]:
    """Returns None if not found"""

@abstractmethod
def delete(self, hash_key: str) -> bool:
    """Returns False if not found"""
```

Mixed patterns: Optional return vs boolean return for similar "not found" conditions.

**Pythonic Fix:** Consistent approach - either raise KeyError for not found, or use Optional consistently.

---

### 14. Close Pattern Not Enforced

**Location:** `storage/weight_store.py:106-116`

`WeightStore` defines `close()` as abstract and provides context manager, but:
- `Deduplicator` manages stores but has no close()
- `Repository._get_remote_store()` has conditional: `if hasattr(remote_store, "close")`

**Pythonic Fix:** All resource-managing classes should be context managers.

---

## Type Safety Issues

### 15. Type Shadowing for Optional Dependencies ✅ FIXED

**Location:** `integrations/pytorch.py:21-29`

```python
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

    # These shadow the real types
    class nn:
        class Module:
            pass
```

**Issue:** IDE autocomplete and type checkers get confused.

**Better Pattern:**
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
```

---

### 16. Magic String Literals Instead of Constants ✅ FIXED

**Location:** `integrations/pytorch.py:249-258`

```python
def add_callback(self, event: str, callback: Callable) -> None:
    if event == "epoch_end":
        # ...
    elif event == "step_end":
        # ...
```

**Pythonic Fix:** Use Literal type or enum:
```python
from typing import Literal

CallbackEvent = Literal["epoch_end", "step_end", "checkpoint_save"]

def add_callback(self, event: CallbackEvent, callback: Callable) -> None:
```

---

## Documentation Issues

### 17. Inconsistent Docstring Presence

Some methods have comprehensive docstrings, others have none:
- `Deduplicator.add_weight()` - has docstring
- `Deduplicator._add_duplicate()` - has brief docstring
- `Deduplicator._compute_delta_hash()` - has brief docstring

**Best Practice:** All public methods should have docstrings. Private methods should at minimum describe non-obvious behavior.

---

## Recommendations Summary

### High Priority
1. Consolidate `DeltaType` and `DeltaTypeEnum` into single enum
2. Use consistent types for `delta_type` (enum everywhere)
3. Add deprecation warnings for deprecated parameters
4. Standardize `from_dict()` signatures

### Medium Priority
5. Deprecate redundant loading functions in favor of unified `load()`
6. Standardize return types for similar operations
7. Replace boolean returns with exceptions for error conditions
8. Add missing type annotations

### Low Priority
9. Standardize method naming patterns
10. Use consistent path parameter types
11. Use Literal types or enums for string constants
12. Improve docstring consistency

---

## Positive Design Patterns Found

1. **Dataclasses with serialization:** Consistent `to_dict()`/`from_dict()` pattern
2. **Context manager support:** `WeightStore` properly supports `with` statement
3. **Thread safety:** `Deduplicator` uses `threading.RLock`
4. **Lazy loading:** `WeightTensor.data` property with lazy loading
5. **Configuration hierarchy:** Environment > repo > user > defaults
6. **Optional dependency handling:** `TORCH_AVAILABLE` pattern (though could be improved)
