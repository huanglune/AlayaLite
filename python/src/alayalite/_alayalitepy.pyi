# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""Reviewed private stub for the canonical pybind11 module."""

from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

MetadataScalar = bool | int | float | str

__version__: str

class MetricType(Enum):
    L2: MetricType
    IP: MetricType
    COS: MetricType

class CollectionStatusError(RuntimeError):
    status_code: int
    operation_stage: int
    status_detail: int
    retryability: int
    partial: bool
    status_version: str

class CollectionInvalidArgumentError(CollectionStatusError, ValueError): ...
class CollectionNotSupportedError(CollectionStatusError, NotImplementedError): ...
class CollectionConflictError(CollectionStatusError): ...
class CollectionNotFoundError(CollectionStatusError, KeyError): ...
class CollectionResourceExhaustedError(CollectionStatusError): ...
class CollectionDeadlineExceededError(CollectionStatusError): ...
class CollectionCancelledError(CollectionStatusError): ...
class CollectionIoError(CollectionStatusError): ...
class CollectionCorruptionError(CollectionStatusError): ...
class CollectionClosedError(CollectionStatusError): ...
class CollectionInternalError(CollectionStatusError): ...

class _RecordResponse:
    id: str | int
    upsert_sequence: int
    document: str
    metadata: dict[str, MetadataScalar]
    vector: npt.NDArray[np.generic] | None

class _MutationRowResponse:
    op_id: int
    batch_op_id: int
    row_op_id: int
    visibility_watermark: int
    durable_watermark: int
    searchable: bool
    durability: int
    row_status: int
    retry_token: str

class _MutationResponse:
    batch_op_id: int
    visibility_watermark: int
    durable_watermark: int
    searchable: bool
    durability: int
    retry_token: str
    rows: list[_MutationRowResponse]

class _SearchStatsResponse:
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

class _SearchResponse:
    ids: npt.NDArray[np.object_]
    distances: npt.NDArray[np.float32]
    offsets: npt.NDArray[np.uint64]
    valid_counts: npt.NDArray[np.uint64]
    status_codes: npt.NDArray[np.uint8]
    completeness_codes: npt.NDArray[np.uint8]
    visibility_watermark: int
    metadata_epoch: int
    search_stats: _SearchStatsResponse

class _CheckpointResponse:
    durable_watermark: int
    wal_cut: int
    metadata_epoch: int
    checkpoint_name: str

class _SealResponse:
    source_segment_id: int
    successor_segment_id: int
    sealed_segment_id: int
    wal_cut: int
    sealed_rows: int
    sealed_bytes: int
    manifest_generation: int

class _CompactResponse:
    source_segment_ids: list[int]
    compacted_segment_id: int
    compacted_rows: int
    input_bytes: int
    output_bytes: int
    manifest_generation: int

class _GcResponse:
    pending: int
    reclaimed: int
    deferred: int
    reclaimed_bytes: int
    manifest_generation: int

class _StatsResponse:
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
    active_segment_algorithm: str
    compacted_bytes: int
    lifecycle: int

class _OptionsResponse:
    root: str
    read_only: bool
    dim: int
    metric: str
    dtype: np.dtype[Any]
    index_type: str
    quantization_type: str
    build_threads: int
    max_neighbors: int
    ef_construction: int
    implementation_key: str
    engine_factory_key: str
    active_algorithm: str
    auto_seal_rows: int

class _CapabilitiesResponse:
    index_types: list[str]
    laser_enabled: bool
    laser_simd: str | None

def capabilities() -> _CapabilitiesResponse: ...

class _Collection:
    read_only: bool

    @staticmethod
    def create(
        root: str,
        dim: int,
        metric: str,
        dtype: np.dtype[Any],
        index_type: str,
        quantization_type: str,
        build_threads: int = ...,
        max_neighbors: int = ...,
        ef_construction: int = ...,
        auto_seal_rows: int = ...,
    ) -> _Collection: ...
    @staticmethod
    def open(root: str, read_only: bool = ...) -> _Collection: ...
    def mutate(
        self,
        ids: list[str],
        documents: list[str],
        vectors: npt.NDArray[np.generic],
        metadata: list[dict[str, MetadataScalar] | None],
        action: str,
        *,
        mode: str = ...,
        durability: str = ...,
        retry_token: str = ...,
    ) -> _MutationResponse: ...
    def remove(
        self,
        ids: list[str],
        *,
        mode: str = ...,
        durability: str = ...,
        retry_token: str = ...,
    ) -> _MutationResponse: ...
    def search(
        self,
        query: npt.NDArray[np.generic],
        top_k: int,
        *,
        effort: int = ...,
        metadata_filter: dict[str, Any] | None = ...,
        filter_policy: str = ...,
        filter_selectivity: float | None = ...,
        scratch_budget_bytes: int = ...,
        io_budget_requests: int = ...,
        io_budget_bytes: int = ...,
    ) -> _SearchResponse: ...
    def batch_search(
        self,
        queries: npt.NDArray[np.generic],
        top_k: int,
        *,
        effort: int = ...,
        metadata_filter: dict[str, Any] | None = ...,
        filter_policy: str = ...,
        filter_selectivity: float | None = ...,
        scratch_budget_bytes: int = ...,
        io_budget_requests: int = ...,
        io_budget_bytes: int = ...,
    ) -> _SearchResponse: ...
    def get_by_id(self, id: str, *, include_vector: bool = ...) -> _RecordResponse | None: ...
    def get_by_ids(self, ids: list[str], *, include_vector: bool = ...) -> list[_RecordResponse | None]: ...
    def records(self) -> list[_RecordResponse]: ...
    def scan(
        self,
        *,
        metadata_filter: dict[str, Any] | None = ...,
        limit: int = ...,
        include_vector: bool = ...,
    ) -> list[_RecordResponse]: ...
    def checkpoint(self) -> _CheckpointResponse: ...
    def seal(self) -> _SealResponse: ...
    def compact(self) -> _CompactResponse: ...
    def gc(self) -> _GcResponse: ...
    def stats(self) -> _StatsResponse: ...
    def options(self) -> _OptionsResponse: ...
    def close(self) -> None: ...
