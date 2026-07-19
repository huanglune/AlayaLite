# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Typed result, receipt, statistics, and value models for SDK v2."""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypeAlias, final

import numpy as np
import numpy.typing as npt

from .config import CollectionConfig, IndexType

MetadataScalar: TypeAlias = bool | int | float | str
Metadata: TypeAlias = Mapping[str, MetadataScalar]
FilterOperatorValue: TypeAlias = MetadataScalar | Sequence[MetadataScalar]
FilterFieldExpression: TypeAlias = Mapping[str, FilterOperatorValue]
Filter: TypeAlias = Mapping[
    str,
    MetadataScalar | FilterFieldExpression | Sequence["Filter"],
]
BatchMode: TypeAlias = Literal["atomic", "partial"]
WriteDurability: TypeAlias = Literal["fsync", "buffered"]
FilterPolicy: TypeAlias = Literal["auto", "strict", "allow_partial"]
VectorInput: TypeAlias = npt.ArrayLike


class RowStatus(str, Enum):
    """Stable outcome for one row in a mutation receipt."""

    INSERTED = "inserted"
    UPDATED = "updated"
    REPLACED = "replaced"
    DELETED = "deleted"
    ALREADY_EXISTS = "already_exists"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    INVALID_ARGUMENT = "invalid_argument"
    ABORTED = "aborted"


class DurabilityState(str, Enum):
    """Durability reached by a returned mutation receipt."""

    MEMORY_ONLY = "memory_only"
    BUFFERED = "buffered"
    FSYNC = "fsync"


class SearchStatus(str, Enum):
    """Stable per-query status translated from the native status protocol."""

    OK = "ok"
    INVALID_ARGUMENT = "invalid_argument"
    NOT_SUPPORTED = "not_supported"
    CONFLICT = "conflict"
    NOT_FOUND = "not_found"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    CANCELLED = "cancelled"
    IO_ERROR = "io_error"
    CORRUPTION = "corruption"
    CLOSED = "closed"
    INTERNAL = "internal"


class SearchCompleteness(str, Enum):
    """Explain why a query row contains its returned number of hits."""

    COMPLETE_K = "complete_k"
    ELIGIBLE_EXHAUSTED = "eligible_exhausted"
    STRATEGY_INCOMPLETE = "strategy_incomplete"
    CANCELLED_PARTIAL = "cancelled_partial"
    FAILED = "failed"


class CollectionLifecycle(str, Enum):
    """Lifecycle state reported by native collection statistics."""

    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


def _readonly_array(array: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    """Mark an owned or borrowed NumPy array as read-only."""
    array.setflags(write=False)
    return array


@final
@dataclass(frozen=True, slots=True)
class Capabilities:
    """Describe index families available in the installed native wheel.

    Parameters
    ----------
    index_types
        Supported public index family names.
    laser_enabled
        Whether the LASER implementation needed by QG is present.
    laser_simd
        Selected LASER SIMD implementation, or ``None`` without LASER.
    """

    index_types: frozenset[IndexType]
    laser_enabled: bool
    laser_simd: str | None


@final
@dataclass(frozen=True, slots=True)
class SearchBudget:
    """Bound resources consumed by one search call.

    Parameters
    ----------
    scratch_bytes
        Maximum query scratch bytes, or ``None`` for the native unlimited
        sentinel.
    io_requests
        Maximum I/O requests, or ``None`` for unlimited.
    io_bytes
        Maximum I/O bytes, or ``None`` for unlimited.
    """

    scratch_bytes: int | None = None
    io_requests: int | None = None
    io_bytes: int | None = None

    def __post_init__(self) -> None:
        """Reject negative and non-integer resource limits."""
        for name, value in (
            ("scratch_bytes", self.scratch_bytes),
            ("io_requests", self.io_requests),
            ("io_bytes", self.io_bytes),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int or None")
            if value < 0:
                raise ValueError(f"{name} must be greater than or equal to 0")


@final
@dataclass(frozen=True, slots=True)
class Record:
    """A position-preserving collection record projection.

    Parameters
    ----------
    id
        Logical string identifier.
    document
        Stored document text.
    metadata
        Read-only flat metadata mapping.
    version
        Monotonic record version.
    vector
        Owned, read-only vector when requested; otherwise ``None``.
    """

    id: str
    document: str
    metadata: Metadata
    version: int
    vector: npt.NDArray[np.generic] | None = None

    def __post_init__(self) -> None:
        """Freeze metadata and any projected vector buffer."""
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if self.vector is not None:
            vector = np.asarray(self.vector)
            if vector.ndim != 1:
                raise ValueError("record vector must be one-dimensional")
            object.__setattr__(self, "vector", _readonly_array(vector))


@final
@dataclass(frozen=True, slots=True)
class RowMutation:
    """Typed outcome for one input row.

    Parameters
    ----------
    id
        Input logical identifier.
    op_id, batch_op_id, row_op_id
        Stable operation identifiers assigned by the coordinator.
    visibility_watermark, durable_watermark
        Visibility and durability watermarks observed by this row.
    searchable
        Whether the successful row is searchable on return.
    durability
        Durability state reached on return.
    status
        Stable row outcome.
    idempotency_key
        Batch idempotency key, or ``None``.
    """

    id: str
    op_id: int
    batch_op_id: int
    row_op_id: int
    visibility_watermark: int
    durable_watermark: int
    searchable: bool
    durability: DurabilityState
    status: RowStatus
    idempotency_key: str | None


@final
@dataclass(frozen=True, slots=True)
class MutationResult:
    """Typed receipt for add, replace, upsert, or delete.

    Parameters
    ----------
    batch_op_id
        Stable batch operation identifier.
    visibility_watermark, durable_watermark
        Batch visibility and durable watermarks.
    searchable
        Whether successful rows are searchable on return.
    durable
        Whether the receipt carries the fsync durability guarantee.
    durability
        Native durability state translated to a public enum.
    idempotency_key
        Batch idempotency key, or ``None``.
    rows
        Input-order row receipts.
    """

    batch_op_id: int
    visibility_watermark: int
    durable_watermark: int
    searchable: bool
    durable: bool
    durability: DurabilityState
    idempotency_key: str | None
    rows: tuple[RowMutation, ...]


@final
@dataclass(frozen=True, slots=True)
class DeleteResult:
    """Summarize snapshot expansion and batched deletion.

    Parameters
    ----------
    matched
        Rows matched by the pinned scan snapshot.
    deleted
        Rows successfully deleted.
    not_found
        Rows lost to concurrent deletion before their batch committed.
    batches
        Number of atomic delete batches issued.
    """

    matched: int
    deleted: int
    not_found: int
    batches: int


@final
@dataclass(frozen=True, slots=True)
class SearchStats:
    """Search execution and resource-accounting statistics.

    ``effective_effort`` is ``None`` for Flat and the effective QG effort for
    QG, including the default ``max(100, limit)`` floor.
    """

    filter_active: bool
    filter_execution: str
    filter_examined: int
    filter_passed: int
    nan_discarded: int
    overfetch_rounds: int
    budget_consumed: int
    lease_acquired: int
    lease_released: int
    lease_peak_bytes: int
    io_requests_consumed: int
    io_bytes_consumed: int
    rerank_nanoseconds: int
    effective_effort: int | None


@final
@dataclass(frozen=True, slots=True)
class SearchRow:
    """Read-only zero-copy view of one CSR query row.

    Parameters
    ----------
    ids, distances
        Slices sharing storage with their parent :class:`SearchResult`.
    status
        Per-query status.
    completeness
        Per-query completeness classification.
    """

    ids: npt.NDArray[np.object_]
    distances: npt.NDArray[np.float32]
    status: SearchStatus
    completeness: SearchCompleteness

    def __post_init__(self) -> None:
        """Ensure the two row buffers cannot be mutated."""
        self.ids.setflags(write=False)
        self.distances.setflags(write=False)


@final
@dataclass(frozen=True, slots=True)
class SearchResult:
    """Read-only compressed-sparse-row search result.

    Parameters
    ----------
    ids, distances
        Flat hit columns with no sentinel padding.
    offsets
        CSR offsets of length ``query_count + 1``.
    status_codes, completeness_codes
        One native protocol code per query.
    visibility_watermark, metadata_epoch
        Snapshot identity used by the search.
    stats
        Typed execution statistics.

    Notes
    -----
    ``result[i]`` slices the flat columns and therefore shares memory with the
    parent result. All exposed arrays are read-only.
    """

    ids: npt.NDArray[np.object_]
    distances: npt.NDArray[np.float32]
    offsets: npt.NDArray[np.uint64]
    status_codes: npt.NDArray[np.uint8]
    completeness_codes: npt.NDArray[np.uint8]
    visibility_watermark: int
    metadata_epoch: int
    stats: SearchStats

    def __post_init__(self) -> None:
        """Canonicalize arrays and validate CSR invariants."""
        ids = np.asarray(self.ids, dtype=np.object_)
        distances = np.asarray(self.distances, dtype=np.float32)
        offsets = np.asarray(self.offsets, dtype=np.uint64)
        status_codes = np.asarray(self.status_codes, dtype=np.uint8)
        completeness_codes = np.asarray(self.completeness_codes, dtype=np.uint8)
        arrays = (ids, distances, offsets, status_codes, completeness_codes)
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("SearchResult arrays must be one-dimensional")
        if ids.size != distances.size:
            raise ValueError("SearchResult ids and distances must have equal lengths")
        if status_codes.size != completeness_codes.size:
            raise ValueError("SearchResult status columns must have equal lengths")
        if offsets.size != status_codes.size + 1:
            raise ValueError("SearchResult offsets must have query_count + 1 entries")
        if offsets.size == 0 or int(offsets[0]) != 0 or int(offsets[-1]) != ids.size:
            raise ValueError("SearchResult offsets do not bound the flat hit columns")
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("SearchResult offsets must be nondecreasing")
        for item in ids.tolist():
            if not isinstance(item, str):
                raise TypeError("SearchResult ids must contain strings")
        object.__setattr__(self, "ids", _readonly_array(ids))
        object.__setattr__(self, "distances", _readonly_array(distances))
        object.__setattr__(self, "offsets", _readonly_array(offsets))
        object.__setattr__(self, "status_codes", _readonly_array(status_codes))
        object.__setattr__(self, "completeness_codes", _readonly_array(completeness_codes))

    def __len__(self) -> int:
        """Return the number of query rows."""
        return int(self.status_codes.size)

    def __getitem__(self, index: int) -> SearchRow:
        """Return a shared-memory row view.

        Parameters
        ----------
        index
            Zero-based query row index. Negative indexing follows normal
            Python sequence rules.

        Returns
        -------
        SearchRow
            Read-only views into the parent flat columns.
        """
        position = operator.index(index)
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError("SearchResult row index out of range")
        begin = int(self.offsets[position])
        end = int(self.offsets[position + 1])
        return SearchRow(
            ids=self.ids[begin:end],
            distances=self.distances[begin:end],
            status=self.statuses[position],
            completeness=self.completeness[position],
        )

    @property
    def valid_counts(self) -> npt.NDArray[np.uint64]:
        """Return the number of valid hits in each query row."""
        counts = np.diff(self.offsets).astype(np.uint64, copy=False)
        counts.setflags(write=False)
        return counts

    @property
    def statuses(self) -> tuple[SearchStatus, ...]:
        """Return translated per-query status values."""
        return tuple(_search_status(code) for code in self.status_codes.tolist())

    @property
    def completeness(self) -> tuple[SearchCompleteness, ...]:
        """Return translated per-query completeness values."""
        return tuple(_search_completeness(code) for code in self.completeness_codes.tolist())


@final
@dataclass(frozen=True, slots=True)
class CollectionStats:
    """Typed collection storage, watermark, and lifecycle statistics."""

    size: int
    accepted_count: int
    pending_count: int
    searchable_bytes: int
    accepted_bytes: int
    searchable_vector_bytes: int
    accepted_vector_bytes: int
    pending_bytes: int
    allocated_count: int
    tombstone_count: int
    routing_generation: int
    visibility_watermark: int
    durable_watermark: int
    metadata_epoch: int
    sealed_segments_count: int
    gc_pending_count: int
    active_segment_algorithm: IndexType
    compacted_bytes: int
    lifecycle: CollectionLifecycle


@final
@dataclass(frozen=True, slots=True)
class CollectionInfo:
    """Stable collection identity and implementation diagnostics.

    Parameters
    ----------
    name, path
        Catalog name and resolved collection directory.
    read_only
        Whether this handle is narrowed to read-only operation.
    storage_format
        Canonical on-disk format identifier.
    config
        Immutable logical collection configuration.
    active_index
        Index family currently serving the mutable segment.
    legacy_quantization
        Ignored historical declaration, when one was discovered.
    """

    name: str
    path: Path
    read_only: bool
    storage_format: str
    config: CollectionConfig
    active_index: IndexType
    legacy_quantization: str | None = None


@final
@dataclass(frozen=True, slots=True)
class CheckpointReceipt:
    """Receipt for a durable full checkpoint."""

    durable_watermark: int
    wal_cut: int
    metadata_epoch: int
    checkpoint_name: str


@final
@dataclass(frozen=True, slots=True)
class SealReceipt:
    """Receipt for active-segment rotation and target sealing."""

    source_segment_id: int
    successor_segment_id: int
    sealed_segment_id: int
    wal_cut: int
    sealed_rows: int
    sealed_bytes: int
    manifest_generation: int


@final
@dataclass(frozen=True, slots=True)
class CompactionReceipt:
    """Receipt for sealed-generation compaction."""

    source_segment_ids: tuple[int, ...]
    compacted_segment_id: int
    compacted_rows: int
    input_bytes: int
    output_bytes: int
    manifest_generation: int


@final
@dataclass(frozen=True, slots=True)
class GarbageCollectionReceipt:
    """Receipt for explicit artifact garbage collection."""

    pending: int
    reclaimed: int
    deferred: int
    reclaimed_bytes: int
    manifest_generation: int


_SEARCH_STATUSES = tuple(SearchStatus)
_SEARCH_COMPLETENESS = tuple(SearchCompleteness)


def _search_status(code: int) -> SearchStatus:
    """Translate a native status byte to a stable enum."""
    try:
        return _SEARCH_STATUSES[code]
    except IndexError as error:
        raise ValueError(f"unknown native search status code: {code}") from error


def _search_completeness(code: int) -> SearchCompleteness:
    """Translate a native completeness byte to a stable enum."""
    try:
        return _SEARCH_COMPLETENESS[code]
    except IndexError as error:
        raise ValueError(f"unknown native search completeness code: {code}") from error


__all__ = [
    "BatchMode",
    "Capabilities",
    "CheckpointReceipt",
    "CollectionInfo",
    "CollectionLifecycle",
    "CollectionStats",
    "CompactionReceipt",
    "DeleteResult",
    "DurabilityState",
    "Filter",
    "FilterPolicy",
    "GarbageCollectionReceipt",
    "Metadata",
    "MetadataScalar",
    "MutationResult",
    "Record",
    "RowMutation",
    "RowStatus",
    "SearchBudget",
    "SearchCompleteness",
    "SearchResult",
    "SearchRow",
    "SearchStats",
    "SearchStatus",
    "VectorInput",
    "WriteDurability",
]
