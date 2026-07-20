# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""AlayaLite embedded vector database SDK."""

import warnings

from ._capabilities import capabilities
from ._collection import Collection
from ._database import Database, connect
from .config import CollectionConfig, FlatIndexConfig, QGIndexConfig
from .exceptions import (
    CollectionCancelledError,
    CollectionClosedError,
    CollectionConflictError,
    CollectionCorruptionError,
    CollectionDeadlineExceededError,
    CollectionInternalError,
    CollectionInvalidArgumentError,
    CollectionIoError,
    CollectionNotFoundError,
    CollectionNotSupportedError,
    CollectionResourceExhaustedError,
    CollectionStatusError,
)
from .models import Capabilities, MutationResult, Record, SearchResult

# The extension module is compiled with -Ofast (fast-math), so loading it enables flush-to-zero /
# denormals-are-zero on the calling thread. numpy notices the changed FPU state and emits "smallest
# subnormal ... is zero" UserWarnings from any later finfo() call anywhere in the process. The FTZ
# side effect is expected and harmless for this workload; suppress exactly these two messages so
# they don't spam applications embedding the SDK.
warnings.filterwarnings(
    "ignore",
    message="The value of the smallest subnormal for <class 'numpy.float32'> type is zero.",
)
warnings.filterwarnings(
    "ignore", message="The value of the smallest subnormal for <class 'numpy.float64'> type is zero."
)


__all__ = [
    "connect",
    "capabilities",
    "Database",
    "Collection",
    "CollectionConfig",
    "FlatIndexConfig",
    "QGIndexConfig",
    "Capabilities",
    "Record",
    "SearchResult",
    "MutationResult",
    "CollectionStatusError",
    "CollectionInvalidArgumentError",
    "CollectionNotSupportedError",
    "CollectionConflictError",
    "CollectionNotFoundError",
    "CollectionResourceExhaustedError",
    "CollectionDeadlineExceededError",
    "CollectionCancelledError",
    "CollectionIoError",
    "CollectionCorruptionError",
    "CollectionClosedError",
    "CollectionInternalError",
]


try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("alayalite")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0.dev"
