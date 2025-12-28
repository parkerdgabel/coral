"""Core abstractions for weight storage and deduplication"""

from coral.core.deduplicator import Deduplicator
from coral.core.lazy_weight import (
    LazyLoadConfig,
    LazyWeightCollection,
    StreamingWeightIterator,
    WeightProxy,
)
from coral.core.model_soup import (
    ModelSoup,
    SoupConfig,
    SoupResult,
    SoupStrategy,
    dare_ties_merge,
    greedy_soup,
    ties_merge,
    uniform_soup,
)
from coral.core.simhash import (
    MultiDimSimHashIndex,
    SimHash,
    SimHashConfig,
    SimHashIndex,
)
from coral.core.similarity_index import (
    Fingerprint,
    MultiDimSimilarityIndex,
    SimilarityIndex,
    SimilarityIndexConfig,
    SimilarityResult,
)
from coral.core.weight_tensor import WeightTensor

__all__ = [
    # Core
    "WeightTensor",
    "Deduplicator",
    # Lazy loading
    "WeightProxy",
    "LazyWeightCollection",
    "StreamingWeightIterator",
    "LazyLoadConfig",
    # Model Souping
    "ModelSoup",
    "SoupConfig",
    "SoupResult",
    "SoupStrategy",
    "uniform_soup",
    "greedy_soup",
    "ties_merge",
    "dare_ties_merge",
    # Unified similarity index (recommended)
    "SimilarityIndex",
    "MultiDimSimilarityIndex",
    "SimilarityIndexConfig",
    "SimilarityResult",
    "Fingerprint",
    # Legacy SimHash (deprecated, use SimilarityIndex)
    "SimHash",
    "SimHashConfig",
    "SimHashIndex",
    "MultiDimSimHashIndex",
]
