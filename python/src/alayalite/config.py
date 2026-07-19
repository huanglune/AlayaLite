# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Immutable configuration models for the AlayaLite v2 Python core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias, final

Metric: TypeAlias = Literal["l2", "ip", "cosine"]
VectorDType: TypeAlias = Literal["float32", "int8", "uint8"]  # pylint: disable=invalid-name
IndexType: TypeAlias = Literal["flat", "qg"]

_METRICS = frozenset({"l2", "ip", "cosine"})
_VECTOR_DTYPES = frozenset({"float32", "int8", "uint8"})


def _positive_int(value: object, name: str) -> int:
    """Return a validated positive Python integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value


@final
@dataclass(frozen=True, slots=True)
class FlatIndexConfig:
    """Configure exact flat-vector search.

    Notes
    -----
    Flat search has no construction or query-effort controls. Passing an
    ``effort`` value to a flat collection is therefore rejected.
    """

    kind: Literal["flat"] = field(default="flat", init=False)


@final
@dataclass(frozen=True, slots=True)
class QGIndexConfig:
    """Configure the RaBitQ graph index family.

    Parameters
    ----------
    max_neighbors
        Maximum graph degree. The current sealed implementation accepts 32
        or 64; the combination is checked before collection creation.
    construction_effort
        Candidate pool used while building the graph.
    build_threads
        Number of build threads, or ``None`` to let the SDK choose.
    """

    kind: Literal["qg"] = field(default="qg", init=False)
    max_neighbors: int = 32
    construction_effort: int = 400
    build_threads: int | None = None

    def __post_init__(self) -> None:
        """Validate scalar construction controls."""
        _positive_int(self.max_neighbors, "max_neighbors")
        _positive_int(self.construction_effort, "construction_effort")
        if self.build_threads is not None:
            _positive_int(self.build_threads, "build_threads")


IndexConfig: TypeAlias = FlatIndexConfig | QGIndexConfig


@final
@dataclass(frozen=True, slots=True)
class CollectionConfig:
    """Describe an immutable collection schema.

    Parameters
    ----------
    dimension
        Number of scalar values in every stored vector.
    dtype
        Canonical vector dtype spelling.
    metric
        Canonical distance metric spelling.
    index
        Flat or QG discriminated index configuration. QG is the default.
    auto_seal_rows
        Positive row threshold for automatic sealing, or ``None`` to disable
        automatic sealing.

    Notes
    -----
    Platform-dependent QG constraints are validated by
    :meth:`Database.create_collection` before any collection directory is
    created.
    """

    dimension: int
    dtype: VectorDType = "float32"
    metric: Metric = "l2"
    index: IndexConfig = field(default_factory=QGIndexConfig)
    auto_seal_rows: int | None = None

    def __post_init__(self) -> None:
        """Validate the platform-independent schema contract."""
        _positive_int(self.dimension, "dimension")
        if not isinstance(self.dtype, str):
            raise TypeError("dtype must be a canonical string")
        if self.dtype not in _VECTOR_DTYPES:
            raise ValueError("dtype must be one of: float32, int8, uint8")
        if not isinstance(self.metric, str):
            raise TypeError("metric must be a canonical string")
        if self.metric not in _METRICS:
            raise ValueError("metric must be one of: l2, ip, cosine")
        if not isinstance(self.index, (FlatIndexConfig, QGIndexConfig)):
            raise TypeError("index must be FlatIndexConfig or QGIndexConfig")
        if self.auto_seal_rows is not None:
            _positive_int(self.auto_seal_rows, "auto_seal_rows")


__all__ = [
    "CollectionConfig",
    "FlatIndexConfig",
    "IndexConfig",
    "IndexType",
    "Metric",
    "QGIndexConfig",
    "VectorDType",
]
