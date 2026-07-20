# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Fully typed Collection implementation for the dormant SDK v2 core."""

from __future__ import annotations

import math
import os
import shutil
import sys
import threading
import uuid
import warnings
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast, final

import numpy as np
import numpy.typing as npt

from ._alayalitepy import (
    _CheckpointResponse,
    _CompactResponse,
    _GcResponse,
    _MutationResponse,
    _OptionsResponse,
    _RecordResponse,
    _SealResponse,
    _SearchResponse,
    _StatsResponse,
)
from ._alayalitepy import (
    _Collection as _NativeCollection,
)
from ._capabilities import capabilities
from ._schema import storage_format, write_collection_schema
from ._validation import (
    document_column,
    filter_expression,
    metadata_column,
    native_batch_mode,
    native_durability,
    positive_int,
    strict_ids,
)
from .config import (
    CollectionConfig,
    FlatIndexConfig,
    IndexConfig,
    IndexType,
    Metric,
    QGIndexConfig,
    VectorDType,
)
from .exceptions import (
    CollectionClosedError,
    CollectionInternalError,
    CollectionInvalidArgumentError,
    CollectionIoError,
    CollectionNotSupportedError,
    _status_error,
)
from .models import (
    BatchMode,
    CheckpointReceipt,
    CollectionInfo,
    CollectionLifecycle,
    CollectionStats,
    CompactionReceipt,
    DeleteResult,
    DurabilityState,
    Filter,
    FilterPolicy,
    GarbageCollectionReceipt,
    MetadataScalar,
    MutationResult,
    Record,
    RowMutation,
    RowStatus,
    SealReceipt,
    SearchBudget,
    SearchResult,
    SearchStats,
    VectorInput,
    WriteDurability,
)

if TYPE_CHECKING:
    from ._database import Database

_COLLECTION_TOKEN = object()
_UNLIMITED_RESOURCE = (1 << 64) - 1
_UINT32_MAX = (1 << 32) - 1
_ROW_STATUSES = tuple(RowStatus)
_DURABILITY_STATES = tuple(DurabilityState)
_LIFECYCLE_STATES = tuple(CollectionLifecycle)


def validate_creation_config(config: CollectionConfig) -> None:
    """Validate capability-dependent configuration before filesystem mutation."""
    if not isinstance(config, CollectionConfig):
        raise TypeError("config must be a CollectionConfig")
    if isinstance(config.index, FlatIndexConfig):
        return
    if config.dimension < 33 or config.dimension > 2048:
        raise _status_error(
            CollectionInvalidArgumentError,
            "QG dimension must be in the inclusive range [33, 2048]",
            status_code=1,
            operation_stage=2,
            status_detail=4,
        )
    if config.dtype != "float32":
        raise _status_error(
            CollectionInvalidArgumentError,
            "QG collections require dtype=float32",
            status_code=1,
            operation_stage=2,
            status_detail=5,
        )
    if config.index.max_neighbors not in {32, 64}:
        raise _status_error(
            CollectionInvalidArgumentError,
            "QG max_neighbors must be 32 or 64",
            status_code=1,
            operation_stage=2,
            status_detail=1,
        )
    if config.index.construction_effort < config.index.max_neighbors:
        raise _status_error(
            CollectionInvalidArgumentError,
            "QG construction_effort must be greater than or equal to max_neighbors",
            status_code=1,
            operation_stage=2,
            status_detail=1,
        )
    if config.auto_seal_rows is not None and config.auto_seal_rows <= 32:
        raise _status_error(
            CollectionInvalidArgumentError,
            "QG auto_seal_rows must be greater than 32",
            status_code=1,
            operation_stage=2,
            status_detail=1,
        )
    if "qg" not in capabilities().index_types:
        raise _status_error(
            CollectionNotSupportedError,
            "QG requires the LASER implementation on this platform; Flat fallback is disabled",
            status_code=2,
            operation_stage=2,
            status_detail=16,
        )


def create_native_collection(root: Path, config: CollectionConfig) -> _NativeCollection:
    """Create one native owner from a validated public configuration."""
    validate_creation_config(config)
    index = config.index
    build_threads = index.build_threads if isinstance(index, QGIndexConfig) and index.build_threads else 1
    max_neighbors = index.max_neighbors if isinstance(index, QGIndexConfig) else 32
    construction_effort = index.construction_effort if isinstance(index, QGIndexConfig) else 400
    return _NativeCollection.create(
        os.fspath(root),
        config.dimension,
        config.metric,
        np.dtype(config.dtype),
        index.kind,
        "rabitq" if isinstance(index, QGIndexConfig) else "none",
        build_threads,
        max_neighbors=max_neighbors,
        ef_construction=construction_effort,
        auto_seal_rows=config.auto_seal_rows or 0,
    )


def config_from_native(options: _OptionsResponse) -> CollectionConfig:
    """Project persisted native options into the immutable public schema."""
    if options.index_type == "qg":
        index: IndexConfig = QGIndexConfig(
            max_neighbors=options.max_neighbors,
            construction_effort=options.ef_construction,
            build_threads=options.build_threads,
        )
    elif options.index_type == "flat":
        index = FlatIndexConfig()
    else:
        raise _status_error(
            CollectionNotSupportedError,
            f"unsupported persisted collection index: {options.index_type}",
            status_code=2,
            operation_stage=3,
            status_detail=16,
        )
    dtype_name = np.dtype(options.dtype).name
    if dtype_name not in {"float32", "int8", "uint8"}:
        raise _status_error(
            CollectionNotSupportedError,
            f"unsupported persisted collection dtype: {dtype_name}",
            status_code=2,
            operation_stage=3,
            status_detail=5,
        )
    if options.metric not in {"l2", "ip", "cosine"}:
        raise _status_error(
            CollectionNotSupportedError,
            f"unsupported persisted collection metric: {options.metric}",
            status_code=2,
            operation_stage=3,
            status_detail=1,
        )
    return CollectionConfig(
        dimension=options.dim,
        dtype=cast(VectorDType, dtype_name),
        metric=cast(Metric, options.metric),
        index=index,
        auto_seal_rows=options.auto_seal_rows or None,
    )


def config_matches_native(config: CollectionConfig, options: _OptionsResponse) -> bool:
    """Check that a discovery schema agrees with native persisted options."""
    if (
        config.dimension != options.dim
        or config.dtype != np.dtype(options.dtype).name
        or config.metric != options.metric
        or config.index.kind != options.index_type
        or (config.auto_seal_rows or 0) != options.auto_seal_rows
    ):
        return False
    if isinstance(config.index, QGIndexConfig):
        return (
            config.index.max_neighbors == options.max_neighbors
            and config.index.construction_effort == options.ef_construction
        )
    return True


@final
class Collection:
    """A typed vector collection owned by a :class:`Database`.

    Notes
    -----
    Collection construction is intentionally private. Obtain handles from
    :meth:`Database.create_collection` or :meth:`Database.open_collection`.
    Closing is idempotent; every later operation raises
    :class:`CollectionClosedError`.
    """

    __slots__ = (
        "__weakref__",
        "_closed",
        "_config",
        "_database_ref",
        "_legacy_quantization",
        "_lock",
        "_name",
        "_native",
        "_path",
        "_read_only",
    )

    def __init__(
        self,
        token: object,
        *,
        database: Database,
        name: str,
        path: Path,
        config: CollectionConfig,
        native: _NativeCollection,
        legacy_quantization: str | None = None,
    ) -> None:
        if token is not _COLLECTION_TOKEN:
            raise TypeError("Collection handles must be created by Database")
        self._name = name
        self._path = path
        self._config = config
        self._native: _NativeCollection | None = native
        self._read_only = bool(native.read_only)
        self._closed = False
        self._database_ref = weakref.ref(database)
        self._legacy_quantization = legacy_quantization
        self._lock = threading.RLock()

    @classmethod
    def _create_handle(
        cls,
        *,
        database: Database,
        name: str,
        path: Path,
        config: CollectionConfig,
        native: _NativeCollection,
        legacy_quantization: str | None = None,
    ) -> Collection:
        """Construct a handle for the owning Database."""
        return cls(
            _COLLECTION_TOKEN,
            database=database,
            name=name,
            path=path,
            config=config,
            native=native,
            legacy_quantization=legacy_quantization,
        )

    @property
    def name(self) -> str:
        """Return the catalog name."""
        return self._name

    @property
    def path(self) -> Path:
        """Return the resolved collection directory."""
        return self._path

    @property
    def read_only(self) -> bool:
        """Return whether this handle rejects mutation and control methods."""
        return self._read_only

    @property
    def config(self) -> CollectionConfig:
        """Return the immutable logical configuration."""
        self._require_native()
        return self._config

    @property
    def info(self) -> CollectionInfo:
        """Return typed collection identity and implementation diagnostics."""
        native = self._require_native()
        options = native.options()
        active_index: IndexType = "qg" if options.active_algorithm in {"qg", "laser"} else "flat"
        return CollectionInfo(
            name=self._name,
            path=self._path,
            read_only=self._read_only,
            storage_format=storage_format(),
            config=self._config,
            active_index=active_index,
            legacy_quantization=self._legacy_quantization,
        )

    def add(
        self,
        *,
        ids: Sequence[str],
        vectors: VectorInput,
        documents: Sequence[str] | None = None,
        metadata: Sequence[Mapping[str, MetadataScalar] | None] | None = None,
        mode: BatchMode = "atomic",
        durability: WriteDurability = "fsync",
        idempotency_key: str | None = None,
    ) -> MutationResult:
        """Insert rows that do not already exist.

        Parameters
        ----------
        ids, vectors, documents, metadata
            Equal-length columnar batch. ``vectors`` must have shape
            ``(len(ids), collection.config.dimension)``.
        mode
            ``"atomic"`` or independent-row ``"partial"`` processing.
        durability
            ``"fsync"`` or explicitly non-durable ``"buffered"``.
        idempotency_key
            Optional batch retry identity.

        Returns
        -------
        MutationResult
            Typed batch and input-order row receipt.
        """
        return self._write(
            "add",
            ids=ids,
            vectors=vectors,
            documents=documents,
            metadata=metadata,
            mode=mode,
            durability=durability,
            idempotency_key=idempotency_key,
        )

    def replace(
        self,
        *,
        ids: Sequence[str],
        vectors: VectorInput,
        documents: Sequence[str] | None = None,
        metadata: Sequence[Mapping[str, MetadataScalar] | None] | None = None,
        mode: BatchMode = "atomic",
        durability: WriteDurability = "fsync",
        idempotency_key: str | None = None,
    ) -> MutationResult:
        """Replace complete rows that already exist.

        Parameters and return values match :meth:`add`; missing IDs receive a
        ``not_found`` row status.
        """
        return self._write(
            "replace",
            ids=ids,
            vectors=vectors,
            documents=documents,
            metadata=metadata,
            mode=mode,
            durability=durability,
            idempotency_key=idempotency_key,
        )

    def upsert(
        self,
        *,
        ids: Sequence[str],
        vectors: VectorInput,
        documents: Sequence[str] | None = None,
        metadata: Sequence[Mapping[str, MetadataScalar] | None] | None = None,
        mode: BatchMode = "atomic",
        durability: WriteDurability = "fsync",
        idempotency_key: str | None = None,
    ) -> MutationResult:
        """Insert missing rows and replace complete existing rows.

        Parameters and return values match :meth:`add`.
        """
        return self._write(
            "upsert",
            ids=ids,
            vectors=vectors,
            documents=documents,
            metadata=metadata,
            mode=mode,
            durability=durability,
            idempotency_key=idempotency_key,
        )

    def delete(
        self,
        ids: Sequence[str],
        *,
        mode: BatchMode = "atomic",
        durability: WriteDurability = "fsync",
        idempotency_key: str | None = None,
    ) -> MutationResult:
        """Delete logical IDs while preserving input-order row statuses."""
        native = self._require_writable()
        normalized_ids = strict_ids(ids, allow_empty=False)
        retry_token = _idempotency_key(idempotency_key)
        resolved_mode = native_batch_mode(mode)
        if mode == "atomic":
            # Native all_or_nothing currently treats a missing delete target as
            # a batch failure. The v2 contract treats that row as a stable
            # not_found no-op while deleting the other requested IDs.
            existing = native.get_by_ids(normalized_ids, include_vector=False)
            if any(record is None for record in existing):
                resolved_mode = "per_row_independent"
        response = native.remove(
            normalized_ids,
            mode=resolved_mode,
            durability=native_durability(durability),
            retry_token=retry_token,
        )
        return _mutation_result(normalized_ids, response)

    def delete_where(
        self,
        where: Filter,
        *,
        batch_size: int = 1000,
        durability: WriteDurability = "fsync",
    ) -> DeleteResult:
        """Expand a pinned filter snapshot and delete it in atomic batches.

        Parameters
        ----------
        where
            Non-empty fixed filter DSL expression. An empty expression is
            rejected to prevent accidental whole-collection deletion.
        batch_size
            Positive number of IDs in each independently atomic batch.
        durability
            Durability applied to every batch.

        Returns
        -------
        DeleteResult
            Matched, deleted, missing, and issued-batch counts.
        """
        native = self._require_writable()
        expression = filter_expression(where)
        if not expression:
            raise ValueError("delete_where requires a non-empty filter")
        size = positive_int(batch_size, "batch_size")
        records = native.scan(metadata_filter=expression, limit=sys.maxsize, include_vector=False)
        matched_ids = [str(record.id) for record in records]
        deleted = 0
        not_found = 0
        batches = 0
        native_durable = native_durability(durability)
        for offset in range(0, len(matched_ids), size):
            batch = matched_ids[offset : offset + size]
            response = native.remove(
                batch,
                mode="all_or_nothing",
                durability=native_durable,
                retry_token="",
            )
            statuses = [_row_status(row.row_status) for row in response.rows]
            deleted += statuses.count(RowStatus.DELETED)
            not_found += statuses.count(RowStatus.NOT_FOUND)
            batches += 1
        return DeleteResult(
            matched=len(matched_ids),
            deleted=deleted,
            not_found=not_found,
            batches=batches,
        )

    def search(
        self,
        queries: VectorInput,
        *,
        limit: int = 10,
        where: Filter | None = None,
        effort: int | None = None,
        filter_policy: FilterPolicy = "auto",
        selectivity_hint: float | None = None,
        budget: SearchBudget | None = None,
    ) -> SearchResult:
        """Search one or many vectors and return one read-only CSR result.

        Parameters
        ----------
        queries
            One vector of shape ``(dimension,)`` or a query matrix of shape
            ``(query_count, dimension)``.
        limit
            Positive maximum hits per query.
        where
            Optional fixed native metadata-filter expression.
        effort
            QG effort. ``None`` uses ``max(100, limit)``. Flat rejects an
            explicit value.
        filter_policy
            ``"auto"``, ``"strict"``, or ``"allow_partial"``.
        selectivity_hint
            Optional estimated matching fraction in ``[0, 1]``.
        budget
            Optional search resource limits.

        Returns
        -------
        SearchResult
            Read-only CSR columns and per-query status/completeness.
        """
        native = self._require_native()
        result_limit = positive_int(limit, "limit")
        effective_effort = self._effective_effort(result_limit, effort)
        array = _query_array(queries, self._config)
        expression = filter_expression(where)
        if filter_policy not in {"auto", "strict", "allow_partial"}:
            raise ValueError("filter_policy must be auto, strict, or allow_partial")
        selectivity = _selectivity(selectivity_hint)
        if budget is not None and not isinstance(budget, SearchBudget):
            raise TypeError("budget must be SearchBudget or None")
        resolved_budget = budget or SearchBudget()
        response = native.search(
            array,
            result_limit,
            effort=effective_effort,
            metadata_filter=expression,
            filter_policy=filter_policy,
            filter_selectivity=selectivity,
            scratch_budget_bytes=_budget_value(resolved_budget.scratch_bytes),
            io_budget_requests=_budget_value(resolved_budget.io_requests),
            io_budget_bytes=_budget_value(resolved_budget.io_bytes),
        )
        return _search_result(response)

    def scan(
        self,
        *,
        where: Filter | None = None,
        limit: int = 100,
        include_vector: bool = False,
    ) -> tuple[Record, ...]:
        """Scan a stable native projection in logical record order.

        Parameters
        ----------
        where
            Optional fixed filter DSL expression; ``None`` and ``{}`` scan
            all rows.
        limit
            Positive finite result limit.
        include_vector
            Include owned read-only vectors when true.
        """
        native = self._require_native()
        result_limit = positive_int(limit, "limit")
        if not isinstance(include_vector, bool):
            raise TypeError("include_vector must be a bool")
        expression = filter_expression(where)
        response = native.scan(
            metadata_filter=expression,
            limit=result_limit,
            include_vector=include_vector,
        )
        return tuple(_record(record) for record in response)

    def get(
        self,
        ids: Sequence[str],
        *,
        include_vector: bool = False,
    ) -> tuple[Record | None, ...]:
        """Return an input-position-aligned record projection.

        Missing IDs remain ``None`` and duplicate input IDs retain duplicate
        output positions.
        """
        native = self._require_native()
        normalized_ids = strict_ids(ids, allow_empty=True)
        if not isinstance(include_vector, bool):
            raise TypeError("include_vector must be a bool")
        response = native.get_by_ids(normalized_ids, include_vector=include_vector)
        return tuple(None if record is None else _record(record) for record in response)

    def checkpoint(self) -> CheckpointReceipt:
        """Create a durable recovery point and truncate the eligible WAL."""
        return _checkpoint_receipt(self._require_writable().checkpoint())

    def seal(self) -> SealReceipt:
        """Rotate the active segment and build the configured sealed target."""
        native = self._require_writable()
        if isinstance(self._config.index, QGIndexConfig) and self.count() <= 32:
            raise _status_error(
                CollectionInvalidArgumentError,
                "QG seal requires more than 32 live rows; Flat fallback is disabled",
                status_code=1,
                operation_stage=14,
                status_detail=1,
            )
        return _seal_receipt(native.seal())

    def compact(self) -> CompactionReceipt:
        """Compact eligible sealed generations."""
        return _compaction_receipt(self._require_writable().compact())

    def collect_garbage(self) -> GarbageCollectionReceipt:
        """Reclaim artifacts no longer pinned by a search epoch."""
        return _garbage_collection_receipt(self._require_writable().gc())

    def rebuild_index(self, *, index: IndexConfig | None = None) -> CheckpointReceipt:
        """Rebuild all live rows into an atomically swapped index owner.

        Parameters
        ----------
        index
            Replacement index configuration, or ``None`` to rebuild the
            current family and construction parameters.

        Returns
        -------
        CheckpointReceipt
            Durable checkpoint receipt from the replacement owner.

        Notes
        -----
        The current binding has no native rebuild primitive. This method uses
        a same-filesystem staging directory, closes the old native owner, and
        atomically swaps directories before reopening the handle.
        """
        with self._lock:
            current = self._require_writable()
            target = self._config.index if index is None else index
            if not isinstance(target, FlatIndexConfig | QGIndexConfig):
                raise TypeError("index must be FlatIndexConfig, QGIndexConfig, or None")
            replacement_config = replace(self._config, index=target)
            validate_creation_config(replacement_config)
            records = current.records()
            if isinstance(target, QGIndexConfig) and len(records) <= 32:
                raise _status_error(
                    CollectionInvalidArgumentError,
                    "QG rebuild requires more than 32 live rows; Flat fallback is disabled",
                    status_code=1,
                    operation_stage=4,
                    status_detail=1,
                )
            return self._rebuild(current, records, replacement_config)

    def stats(self) -> CollectionStats:
        """Return typed collection accounting and lifecycle statistics."""
        return _collection_stats(self._require_native().stats())

    def count(self) -> int:
        """Return the number of live logical records."""
        return self.stats().size

    def __len__(self) -> int:
        """Return :meth:`count` for Python sequence-style sizing."""
        return self.count()

    def close(self) -> None:
        """Idempotently drain and release this native handle."""
        with self._lock:
            if self._closed:
                return
            native = self._native
            if native is not None:
                native.close()
            self._native = None
            self._closed = True
        database = self._database_ref()
        if database is not None:
            database._collection_closed(self)  # pylint: disable=protected-access

    def __enter__(self) -> Collection:
        """Return this open collection handle."""
        self._require_native()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Close this handle without transaction or rollback semantics."""
        del exc_type, exc_value, traceback
        self.close()

    def __del__(self) -> None:
        """Warn and best-effort close an unclosed handle."""
        try:
            if not self._closed:
                warnings.warn(
                    f"unclosed AlayaLite Collection {self._name!r}",
                    ResourceWarning,
                    stacklevel=2,
                )
                self.close()
        except (AttributeError, RuntimeError):
            pass

    def _write(
        self,
        action: str,
        *,
        ids: Sequence[str],
        vectors: VectorInput,
        documents: Sequence[str] | None,
        metadata: Sequence[Mapping[str, MetadataScalar] | None] | None,
        mode: BatchMode,
        durability: WriteDurability,
        idempotency_key: str | None,
    ) -> MutationResult:
        """Validate common write columns and invoke the typed native path."""
        native = self._require_writable()
        normalized_ids = strict_ids(ids, allow_empty=False)
        vector_array = _write_array(vectors, len(normalized_ids), self._config)
        normalized_documents = document_column(documents, len(normalized_ids))
        normalized_metadata = metadata_column(metadata, len(normalized_ids))
        retry_token = _idempotency_key(idempotency_key)
        response = native.mutate(
            normalized_ids,
            normalized_documents,
            vector_array,
            normalized_metadata,
            action,
            mode=native_batch_mode(mode),
            durability=native_durability(durability),
            retry_token=retry_token,
        )
        return _mutation_result(normalized_ids, response)

    def _effective_effort(self, limit: int, effort: int | None) -> int:
        """Apply the public Flat/QG effort contract."""
        if isinstance(self._config.index, FlatIndexConfig):
            if effort is not None:
                raise ValueError("effort is not supported by Flat collections")
            return 100
        floor = max(100, limit)
        if floor > _UINT32_MAX:
            raise ValueError("limit exceeds the QG effort uint32 range")
        if effort is None:
            return floor
        if isinstance(effort, bool) or not isinstance(effort, int):
            raise TypeError("effort must be an int or None")
        if effort < floor:
            raise ValueError(f"effort must be greater than or equal to {floor}")
        if effort > _UINT32_MAX:
            raise ValueError("effort exceeds the native uint32 range")
        return effort

    def _require_native(self) -> _NativeCollection:
        """Return the native owner or raise a metadata-bearing closed error."""
        native = self._native
        if self._closed or native is None:
            raise _status_error(
                CollectionClosedError,
                "Collection handle is closed",
                status_code=10,
                operation_stage=1,
            )
        return native

    def _require_writable(self) -> _NativeCollection:
        """Return a writable native owner or raise the read-only protocol error."""
        native = self._require_native()
        if self._read_only:
            raise _status_error(
                CollectionNotSupportedError,
                "operation is unavailable on a read-only Collection handle",
                status_code=2,
                operation_stage=1,
                status_detail=15,
            )
        return native

    def _rebuild(
        self,
        current: _NativeCollection,
        records: list[_RecordResponse],
        config: CollectionConfig,
    ) -> CheckpointReceipt:
        """Stage, atomically swap, and reopen a replacement native owner."""
        suffix = uuid.uuid4().hex
        staging = self._path.parent / f".{self._path.name}.rebuild-{suffix}"
        backup = self._path.parent / f".{self._path.name}.backup-{suffix}"
        failed = self._path.parent / f".{self._path.name}.failed-{suffix}"
        replacement: _NativeCollection | None = None
        current_closed = False
        try:
            replacement = create_native_collection(staging, config)
            if records:
                ids, documents, vectors, metadata = _record_columns(records, config)
                response = replacement.mutate(
                    ids,
                    documents,
                    vectors,
                    metadata,
                    "add",
                    mode="all_or_nothing",
                    durability="wal_fsync",
                    retry_token="",
                )
                if any(_row_status(row.row_status) is not RowStatus.INSERTED for row in response.rows):
                    raise _status_error(
                        CollectionInternalError,
                        "replacement owner rejected exported live rows",
                        status_code=11,
                        operation_stage=4,
                        status_detail=1,
                    )
                replacement.seal()
            checkpoint = _checkpoint_receipt(replacement.checkpoint())
            write_collection_schema(staging, config)
            replacement.close()
            replacement = None

            current.close()
            current_closed = True
            self._native = None
            os.replace(self._path, backup)
            try:
                os.replace(staging, self._path)
            except OSError:
                os.replace(backup, self._path)
                self._native = _NativeCollection.open(os.fspath(self._path), False)
                current_closed = False
                raise
            try:
                reopened = _NativeCollection.open(os.fspath(self._path), False)
            except Exception:
                os.replace(self._path, failed)
                os.replace(backup, self._path)
                self._native = _NativeCollection.open(os.fspath(self._path), False)
                current_closed = False
                shutil.rmtree(failed, ignore_errors=True)
                raise
            self._native = reopened
            self._config = config
            self._legacy_quantization = None
            current_closed = False
            shutil.rmtree(backup)
            return checkpoint
        except OSError as error:
            if current_closed and self._native is None and self._path.exists():
                self._native = _NativeCollection.open(os.fspath(self._path), False)
            raise _status_error(
                CollectionIoError,
                f"index rebuild filesystem operation failed: {error}",
                status_code=8,
                operation_stage=4,
            ) from error
        finally:
            if replacement is not None:
                try:
                    replacement.close()
                except RuntimeError:
                    pass
            shutil.rmtree(staging, ignore_errors=True)


def _write_array(vectors: VectorInput, rows: int, config: CollectionConfig) -> npt.NDArray[np.generic]:
    """Normalize and validate a two-dimensional write matrix."""
    try:
        array = np.asarray(vectors, dtype=np.dtype(config.dtype))
    except (TypeError, ValueError) as error:
        raise ValueError("vectors must be a rectangular numeric matrix") from error
    if array.ndim != 2:
        raise ValueError("vectors must be a two-dimensional matrix")
    if array.shape != (rows, config.dimension):
        raise ValueError(f"vectors must have shape ({rows}, {config.dimension})")
    if not np.all(np.isfinite(array)):
        raise ValueError("vectors must contain only finite values")
    return np.ascontiguousarray(array)


def _query_array(queries: VectorInput, config: CollectionConfig) -> npt.NDArray[np.generic]:
    """Normalize one query vector or a non-empty query matrix."""
    try:
        array = np.asarray(queries, dtype=np.dtype(config.dtype))
    except (TypeError, ValueError) as error:
        raise ValueError("queries must be a rectangular numeric array") from error
    if array.ndim == 1:
        if array.shape[0] != config.dimension:
            raise ValueError(f"query dimension must be {config.dimension}")
    elif array.ndim == 2:
        if array.shape[0] == 0 or array.shape[1] != config.dimension:
            raise ValueError(f"queries must have shape (query_count, {config.dimension})")
    else:
        raise ValueError("queries must be one- or two-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError("queries must contain only finite values")
    return np.ascontiguousarray(array)


def _idempotency_key(value: str | None) -> str:
    """Validate and translate an optional batch idempotency key."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError("idempotency_key must be a str or None")
    if not value:
        raise ValueError("idempotency_key must not be empty")
    return value


def _selectivity(value: float | None) -> float | None:
    """Validate an optional selectivity fraction."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("selectivity_hint must be a float or None")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0 or resolved > 1.0:
        raise ValueError("selectivity_hint must be in the inclusive range [0, 1]")
    return resolved


def _budget_value(value: int | None) -> int:
    """Translate an optional budget component to the native unlimited sentinel."""
    return _UNLIMITED_RESOURCE if value is None else value


def _row_status(code: int) -> RowStatus:
    """Translate one native row-status byte."""
    try:
        return _ROW_STATUSES[code]
    except IndexError as error:
        raise _status_error(
            CollectionInternalError,
            f"unknown native row status code: {code}",
            status_code=11,
            operation_stage=18,
            status_detail=1,
        ) from error


def _durability_state(code: int) -> DurabilityState:
    """Translate one native durability byte."""
    try:
        return _DURABILITY_STATES[code]
    except IndexError as error:
        raise _status_error(
            CollectionInternalError,
            f"unknown native durability code: {code}",
            status_code=11,
            operation_stage=18,
            status_detail=1,
        ) from error


def _mutation_result(ids: list[str], response: _MutationResponse) -> MutationResult:
    """Project a typed native mutation response into frozen public models."""
    if len(ids) != len(response.rows):
        raise _status_error(
            CollectionInternalError,
            "native mutation receipt row count does not match the input batch",
            status_code=11,
            operation_stage=18,
            status_detail=1,
        )
    durability = _durability_state(response.durability)
    idempotency_key = response.retry_token or None
    rows = tuple(
        RowMutation(
            id=item_id,
            op_id=row.op_id,
            batch_op_id=row.batch_op_id,
            row_op_id=row.row_op_id,
            visibility_watermark=row.visibility_watermark,
            durable_watermark=row.durable_watermark,
            searchable=row.searchable,
            durability=_durability_state(row.durability),
            status=_row_status(row.row_status),
            idempotency_key=row.retry_token or None,
        )
        for item_id, row in zip(ids, response.rows, strict=True)
    )
    return MutationResult(
        batch_op_id=response.batch_op_id,
        visibility_watermark=response.visibility_watermark,
        durable_watermark=response.durable_watermark,
        searchable=response.searchable,
        durable=durability is DurabilityState.FSYNC,
        durability=durability,
        idempotency_key=idempotency_key,
        rows=rows,
    )


def _record(response: _RecordResponse) -> Record:
    """Project one native record response and freeze its optional vector."""
    vector: npt.NDArray[np.generic] | None = None
    if response.vector is not None:
        vector = np.asarray(response.vector)
        vector.setflags(write=False)
    return Record(
        id=str(response.id),
        document=response.document,
        metadata=response.metadata,
        version=response.upsert_sequence,
        vector=vector,
    )


def _search_result(response: _SearchResponse) -> SearchResult:
    """Project one typed native CSR response."""
    stats = response.search_stats
    return SearchResult(
        ids=response.ids,
        distances=response.distances,
        offsets=response.offsets,
        status_codes=response.status_codes,
        completeness_codes=response.completeness_codes,
        visibility_watermark=response.visibility_watermark,
        metadata_epoch=response.metadata_epoch,
        stats=SearchStats(
            filter_active=stats.filter_active,
            filter_execution=stats.filter_execution,
            filter_examined=stats.filter_examined,
            filter_passed=stats.filter_passed,
            nan_discarded=stats.nan_discarded,
            overfetch_rounds=stats.overfetch_rounds,
            budget_consumed=stats.budget_consumed,
            lease_acquired=stats.lease_acquired,
            lease_released=stats.lease_released,
            lease_peak_bytes=stats.lease_peak_bytes,
            io_requests_consumed=stats.io_requests_consumed,
            io_bytes_consumed=stats.io_bytes_consumed,
            rerank_nanoseconds=stats.rerank_nanoseconds,
            effective_effort=stats.effective_effort,
        ),
    )


def _checkpoint_receipt(response: _CheckpointResponse) -> CheckpointReceipt:
    """Project a native checkpoint receipt."""
    return CheckpointReceipt(
        durable_watermark=response.durable_watermark,
        wal_cut=response.wal_cut,
        metadata_epoch=response.metadata_epoch,
        checkpoint_name=response.checkpoint_name,
    )


def _seal_receipt(response: _SealResponse) -> SealReceipt:
    """Project a native seal receipt."""
    return SealReceipt(
        source_segment_id=response.source_segment_id,
        successor_segment_id=response.successor_segment_id,
        sealed_segment_id=response.sealed_segment_id,
        wal_cut=response.wal_cut,
        sealed_rows=response.sealed_rows,
        sealed_bytes=response.sealed_bytes,
        manifest_generation=response.manifest_generation,
    )


def _compaction_receipt(response: _CompactResponse) -> CompactionReceipt:
    """Project a native compaction receipt."""
    return CompactionReceipt(
        source_segment_ids=tuple(response.source_segment_ids),
        compacted_segment_id=response.compacted_segment_id,
        compacted_rows=response.compacted_rows,
        input_bytes=response.input_bytes,
        output_bytes=response.output_bytes,
        manifest_generation=response.manifest_generation,
    )


def _garbage_collection_receipt(response: _GcResponse) -> GarbageCollectionReceipt:
    """Project a native garbage-collection receipt."""
    return GarbageCollectionReceipt(
        pending=response.pending,
        reclaimed=response.reclaimed,
        deferred=response.deferred,
        reclaimed_bytes=response.reclaimed_bytes,
        manifest_generation=response.manifest_generation,
    )


def _collection_stats(response: _StatsResponse) -> CollectionStats:
    """Project typed native collection statistics."""
    try:
        lifecycle = _LIFECYCLE_STATES[response.lifecycle]
    except IndexError as error:
        raise _status_error(
            CollectionInternalError,
            f"unknown native collection lifecycle: {response.lifecycle}",
            status_code=11,
            operation_stage=15,
            status_detail=1,
        ) from error
    active_algorithm: IndexType = "qg" if response.active_segment_algorithm in {"qg", "laser"} else "flat"
    return CollectionStats(
        size=response.size,
        accepted_count=response.accepted_count,
        pending_count=response.pending_count,
        searchable_bytes=response.searchable_bytes,
        accepted_bytes=response.accepted_bytes,
        searchable_vector_bytes=response.searchable_vector_bytes,
        accepted_vector_bytes=response.accepted_vector_bytes,
        pending_bytes=response.pending_bytes,
        allocated_count=response.allocated_count,
        tombstone_count=response.tombstone_count,
        routing_generation=response.routing_generation,
        visibility_watermark=response.visibility_watermark,
        durable_watermark=response.durable_watermark,
        metadata_epoch=response.metadata_epoch,
        sealed_segments_count=response.sealed_segments_count,
        gc_pending_count=response.gc_pending_count,
        active_segment_algorithm=active_algorithm,
        compacted_bytes=response.compacted_bytes,
        lifecycle=lifecycle,
    )


def _record_columns(
    records: list[_RecordResponse],
    config: CollectionConfig,
) -> tuple[
    list[str],
    list[str],
    npt.NDArray[np.generic],
    list[dict[str, MetadataScalar] | None],
]:
    """Convert exported native records into replacement mutation columns."""
    ids: list[str] = []
    documents: list[str] = []
    metadata: list[dict[str, MetadataScalar] | None] = []
    vectors: list[npt.NDArray[np.generic]] = []
    for record in records:
        if record.vector is None:
            raise _status_error(
                CollectionInternalError,
                "native record export omitted a vector required for rebuild",
                status_code=11,
                operation_stage=12,
                status_detail=1,
            )
        ids.append(str(record.id))
        documents.append(record.document)
        metadata.append(dict(record.metadata))
        vectors.append(np.asarray(record.vector))
    array = np.ascontiguousarray(np.stack(vectors), dtype=np.dtype(config.dtype))
    return ids, documents, array, metadata


__all__ = ["Collection"]
