# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
AlayaLite Python SDK.

This is the main entry point for the alayalite package, providing easy access
to all key components like the Client, Collection, and utility functions.
"""

import warnings

from ._alayalitepy import MetricType  # noqa: E402
from ._legacy import AlayaLiteLegacyApiWarning, raise_removed_legacy_api  # noqa: E402
from .client import Client  # noqa: E402
from .collection import (  # noqa: E402
    STATUS_VERSION as COLLECTION_STATUS_VERSION,
)
from .collection import (  # noqa: E402
    V_PUBLIC as COLLECTION_V_PUBLIC,
)
from .collection import (  # noqa: E402
    V_REMOVE as LEGACY_API_V_REMOVE,
)
from .collection import (
    Collection,
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
from .utils import calc_gt, calc_recall, load_fvecs, load_ivecs  # noqa: E402

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
    "Client",
    "Collection",
    "MetricType",
    "COLLECTION_V_PUBLIC",
    "LEGACY_API_V_REMOVE",
    "COLLECTION_STATUS_VERSION",
    "AlayaLiteLegacyApiWarning",
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
    # utils
    "load_fvecs",
    "load_ivecs",
    "calc_recall",
    "calc_gt",
]


def __getattr__(name: str):  # pylint: disable=invalid-name
    if name in {"Index", "DiskCollection"}:
        raise_removed_legacy_api(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("alayalite")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0.dev"
