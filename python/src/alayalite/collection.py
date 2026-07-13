# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Canonical Collection facade backed by the native C++ coordinator."""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union

import numpy as np

from ._alayalitepy import (  # re-exported below as the public status hierarchy
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
from ._alayalitepy import _Collection as _NativeCollection
from ._alayalitepy import _CollectionReadView as _NativeCollectionReadView
from ._legacy import LEGACY_API_V_REMOVE, legacy_api
from .common import _assert, normalize_filter_execution_hint, valid_dtype
from .schema import IndexParams, load_schema, save_schema

V_PUBLIC = "1.1.0"
V_REMOVE = LEGACY_API_V_REMOVE
STATUS_VERSION = "1"


__all__ = [
    "Collection",
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
    "V_PUBLIC",
    "V_REMOVE",
    "STATUS_VERSION",
]


def _default_rocksdb_path(collection_name: str, base_dir: Optional[str] = None) -> str:
    """Retain the historical configuration spelling as a storage-root hint."""
    if base_dir is not None:
        return os.path.join(base_dir, collection_name, "rocksdb")
    rocksdb_base = os.environ.get("ALAYALITE_ROCKSDB_DIR", "./RocksDB")
    return os.path.join(rocksdb_base, collection_name)


def _canonical_root(rocksdb_path: str) -> str:
    """Translate the old config field without opening a RocksDB owner."""
    path = Path(rocksdb_path).expanduser()
    if path.name == "rocksdb":
        path = path.parent
    return os.path.abspath(os.fspath(path))


@dataclass(frozen=True)
class _CompiledMetadataFilter:
    expression: dict


def _validate_filter_expression(expression) -> None:
    if not isinstance(expression, dict):
        raise TypeError("metadata_filter must be a dict")
    for key, value in expression.items():
        if key in ("$and", "$or"):
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"{key} expects a list of filter expressions")
            for child in value:
                _validate_filter_expression(child)
            continue
        if isinstance(value, dict):
            for operator, operand in value.items():
                if operator not in ("$eq", "$gt", "$ge", "$lt", "$le", "$in"):
                    raise ValueError(f"Unsupported operator: {operator}")
                if operator == "$in" and not isinstance(operand, (list, tuple, set, frozenset)):
                    raise TypeError("$in expects a list-like operand")


def _matches_filter(metadata: dict, expression: dict) -> bool:
    for key, expected in expression.items():
        if key == "$and":
            if not all(_matches_filter(metadata, child) for child in expected):
                return False
            continue
        if key == "$or":
            if not any(_matches_filter(metadata, child) for child in expected):
                return False
            continue

        actual = metadata.get(key)
        if not isinstance(expected, dict):
            if actual != expected:
                return False
            continue
        for operator, operand in expected.items():
            if operator == "$eq" and actual != operand:
                return False
            if operator == "$gt" and not (actual is not None and actual > operand):
                return False
            if operator == "$ge" and not (actual is not None and actual >= operand):
                return False
            if operator == "$lt" and not (actual is not None and actual < operand):
                return False
            if operator == "$le" and not (actual is not None and actual <= operand):
                return False
            if operator == "$in" and actual not in operand:
                return False
    return True


def _row_status_name(value: int) -> str:
    return (
        "inserted",
        "updated",
        "replaced",
        "deleted",
        "already_exists",
        "not_found",
        "conflict",
        "invalid_argument",
        "aborted",
    )[int(value)]


class Collection:
    """The canonical Python facade over :class:`alaya::Collection`.

    Vectors, LogicalIds, documents, metadata, versions, tombstones, WAL and
    checkpoints all have one native owner.  The former ``rocksdb_path`` field
    is retained only as a source-compatible choice of collection directory.
    """

    def __init__(self, name: str, index_params: IndexParams = None):
        self.__index_params = index_params if index_params is not None else IndexParams()
        if not self.__index_params.rocksdb_path:
            self.__index_params.rocksdb_path = _default_rocksdb_path(name)
        self.__root = _canonical_root(self.__index_params.rocksdb_path)
        self.__native: Optional[_NativeCollection] = None
        self.__read_view: Optional[_NativeCollectionReadView] = None
        self.__dim: Optional[int] = None
        self.__dtype: Optional[np.dtype] = None

    def _bind_open_native(self, root: str) -> None:
        self.__root = os.path.abspath(root)
        self.__native = _NativeCollection.open(self.__root)
        native_options = self.__native.options()
        self.__dim = int(native_options["dim"])
        self.__dtype = np.dtype(native_options["dtype"])

    @staticmethod
    def _normalize_item_id(item_id) -> str:
        if isinstance(item_id, bytes):
            return item_id.decode("utf-8")
        return str(item_id)

    def _resolved_params(self) -> IndexParams:
        self.__index_params.fill_none_values()
        self.__index_params.data_type = valid_dtype(self.__index_params.data_type)
        self.__index_params.metric = str(self.__index_params.metric).lower()
        self.__index_params.quantization_type = str(self.__index_params.quantization_type).lower()
        self.__index_params.index_type = str(self.__index_params.index_type).lower()
        if self.__index_params.index_type not in ("hnsw", "nsg", "fusion", "qg", "flat"):
            raise ValueError("canonical Collection index_type must be flat, hnsw, nsg, fusion, or qg")
        if self.__index_params.quantization_type not in ("none", "sq4", "sq8", "rabitq"):
            raise ValueError("canonical Collection quantization_type is unsupported")
        return self.__index_params

    def _create_native(self, dim: int, dtype) -> _NativeCollection:
        if self.__index_params.data_type is None:
            self.__index_params.data_type = valid_dtype(dtype)
        params = self._resolved_params()
        dtype = valid_dtype(dtype)
        if params.data_type != dtype:
            raise ValueError(f"Data type mismatch: {params.data_type} vs {dtype}")
        build_threads = params.build_threads if params.build_threads is not None else 1
        _assert(build_threads > 0, "index_params.build_threads must be greater than 0")
        native = _NativeCollection.create(
            self.__root,
            int(dim),
            params.metric,
            dtype,
            params.index_type,
            params.quantization_type,
            int(build_threads),
            400,
        )
        self.__dim = int(dim)
        self.__dtype = dtype
        return native

    def _require_native(self) -> _NativeCollection:
        if self.__native is None:
            raise RuntimeError(
                "Collection index is not initialized yet. Call insert() with the first batch of data first."
            )
        return self.__native

    @legacy_api(
        "collection_get_cpp_index",
        "index",
        "Collection.get_cpp_index()",
        "canonical Collection methods",
        warning_category=DeprecationWarning,
        message=(
            f"Collection.get_cpp_index() is deprecated and will be removed in AlayaLite {V_REMOVE}; "
            "use canonical Collection methods instead. The returned native view is read-only."
        ),
    )
    def get_cpp_index(self) -> _NativeCollectionReadView:
        """Return the Gate 9-A read-only native view.

        The view deliberately has no mutation, RocksDB, or internal-row API.
        Gate 9-B deprecates this escape hatch while retaining the read-only view.
        """
        return self._get_cpp_index()

    def _get_cpp_index(self) -> _NativeCollectionReadView:
        native = self._require_native()
        if self.__read_view is None:
            self.__read_view = _NativeCollectionReadView(native)
        return self.__read_view

    def build_filter(self, filter_dict: Optional[Union[dict, _CompiledMetadataFilter]]):
        """Compile the compatibility metadata DSL for pinned-snapshot filtering."""
        if isinstance(filter_dict, _CompiledMetadataFilter):
            return filter_dict
        expression = {} if filter_dict is None else filter_dict
        _validate_filter_expression(expression)
        return _CompiledMetadataFilter(expression)

    @staticmethod
    def _filter_expression(metadata_filter) -> dict:
        if isinstance(metadata_filter, _CompiledMetadataFilter):
            return metadata_filter.expression
        expression = {} if metadata_filter is None else metadata_filter
        _validate_filter_expression(expression)
        return expression

    def _prepare_items(self, items: Iterable[tuple]):
        materialized = list(items)
        if not materialized:
            return materialized, [], [], np.empty((0, self.__dim or 0), dtype=self.__dtype or np.float32), []
        ids = []
        documents = []
        metadata = []
        vectors = []
        for entry in materialized:
            if len(entry) != 4:
                raise ValueError("Collection items must be (item_id, document, embedding, metadata) tuples")
            item_id, document, embedding, scalar = entry
            ids.append(self._normalize_item_id(item_id))
            documents.append(str(document))
            metadata.append({} if scalar is None else dict(scalar))
            vectors.append(np.asarray(embedding))
        array = np.asarray(vectors)
        _assert(array.ndim == 2, "vectors must be a 2D array")
        dtype = valid_dtype(array.dtype)
        array = np.ascontiguousarray(array, dtype=dtype)
        if self.__dim is not None:
            _assert(array.shape[1] == self.__dim, "Vector dimension must match the index dimension.")
        if self.__dtype is not None and dtype != self.__dtype:
            raise ValueError(f"Data type mismatch: {self.__dtype} vs {dtype}")
        return materialized, ids, documents, array, metadata

    @staticmethod
    def _first_duplicate(ids: List[str]) -> Optional[str]:
        seen = set()
        for item_id in ids:
            if item_id in seen:
                return item_id
            seen.add(item_id)
        return None

    def _persist_discovery_schema(self) -> None:
        if self.__native is None:
            return
        schema_path = os.path.join(self.__root, "schema.json")
        if self.__native.options()["imported_legacy_layout"] and os.path.exists(schema_path):
            return
        save_schema(
            schema_path,
            {
                "type": "collection",
                "format": "canonical_collection_v1",
                "public_version": V_PUBLIC,
                "index": self.__index_params.to_json_dict(),
            },
        )

    def insert(self, items: List[tuple]):
        """Insert rows in ``insert_only`` mode and return the native batch receipt."""
        materialized, ids, documents, vectors, metadata = self._prepare_items(items)
        if not materialized:
            return None
        duplicate = self._first_duplicate(ids)
        if duplicate is not None:
            raise RuntimeError(f"Duplicate item_id: {duplicate}")

        created_here = self.__native is None
        if not created_here:
            existing = self.__native.get_by_ids(ids)
            for item_id, record in zip(ids, existing):
                if record is not None:
                    raise RuntimeError(f"Duplicate item_id: {item_id}")

        root_existed = os.path.exists(self.__root)
        try:
            if self.__native is None:
                self.__native = self._create_native(vectors.shape[1], vectors.dtype)
            receipt = self.__native.mutate(ids, documents, vectors, metadata, "add")
            for item_id, row in zip(ids, receipt["rows"]):
                if _row_status_name(row["row_status"]) != "inserted":
                    raise RuntimeError(f"Duplicate item_id: {item_id}")
            self._persist_discovery_schema()
            return receipt
        except Exception:
            if created_here:
                try:
                    if self.__native is not None:
                        self.__native.close()
                finally:
                    self.__native = None
                    self.__read_view = None
                    self.__dim = None
                    self.__dtype = None
                    if not root_existed and os.path.exists(self.__root):
                        shutil.rmtree(self.__root)
            raise

    def mutate_batch(
        self,
        items,
        *,
        action: str = "upsert",
        mode: str = "per_row_independent",
        durability: str = "wal_fsync",
        retry_token: str = "",
    ):
        """Apply a canonical add/upsert batch with explicit §4.4 semantics."""
        if action not in ("add", "upsert", "replace"):
            raise ValueError("canonical Collection mutation action must be add, upsert, or replace")
        if mode not in ("per_row_independent", "all_or_nothing"):
            raise ValueError("canonical Collection batch mode must be per_row_independent or all_or_nothing")
        if durability not in ("wal_fsync", "searchable"):
            raise ValueError("canonical Collection durability must be wal_fsync or searchable")
        materialized, ids, documents, vectors, metadata = self._prepare_items(items)
        if not materialized:
            return None
        created_here = self.__native is None
        root_existed = os.path.exists(self.__root)
        try:
            if self.__native is None:
                self.__native = self._create_native(vectors.shape[1], vectors.dtype)
            receipt = self.__native.mutate(
                ids,
                documents,
                vectors,
                metadata,
                action,
                mode=mode,
                durability=durability,
                retry_token=str(retry_token),
            )
            self._persist_discovery_schema()
            return receipt
        except Exception:
            if created_here:
                try:
                    if self.__native is not None:
                        self.__native.close()
                finally:
                    self.__native = None
                    self.__read_view = None
                    self.__dim = None
                    self.__dtype = None
                    if not root_existed and os.path.exists(self.__root):
                        shutil.rmtree(self.__root)
            raise

    def add(
        self,
        items,
        *,
        mode: str = "per_row_independent",
        durability: str = "wal_fsync",
        retry_token: str = "",
    ):
        """Canonical insert-only batch returning stable per-row receipts."""
        return self.mutate_batch(
            items,
            action="add",
            mode=mode,
            durability=durability,
            retry_token=retry_token,
        )

    def upsert(
        self,
        items: List[tuple],
        *,
        mode: str = "per_row_independent",
        durability: str = "wal_fsync",
        retry_token: str = "",
    ):
        """Insert or update rows through the Collection version owner."""
        return self.mutate_batch(
            items,
            action="upsert",
            mode=mode,
            durability=durability,
            retry_token=retry_token,
        )

    def remove(
        self,
        ids,
        *,
        mode: str = "per_row_independent",
        durability: str = "wal_fsync",
        retry_token: str = "",
    ):
        """Canonical LogicalId remove batch with stable missing-row statuses."""
        if mode not in ("per_row_independent", "all_or_nothing"):
            raise ValueError("canonical Collection batch mode must be per_row_independent or all_or_nothing")
        if durability not in ("wal_fsync", "searchable"):
            raise ValueError("canonical Collection durability must be wal_fsync or searchable")
        if not ids or self.__native is None:
            return None
        normalized = [self._normalize_item_id(item_id) for item_id in ids]
        return self.__native.remove(
            normalized,
            mode=mode,
            durability=durability,
            retry_token=str(retry_token),
        )

    def delete_by_id(self, ids: List[str]):
        """Remove LogicalIds; missing rows retain stable ``not_found`` receipts."""
        return self.remove(ids)

    def get_by_id(self, ids: List[str]) -> dict:
        """Project item ID, document, and metadata for live LogicalIds."""
        results = {"id": [], "document": [], "metadata": []}
        if not ids or self.__native is None:
            return results
        normalized = [self._normalize_item_id(item_id) for item_id in ids]
        for record in self.__native.get_by_ids(normalized):
            if record is None:
                continue
            results["id"].append(str(record["id"]))
            results["document"].append(record["document"])
            results["metadata"].append(record["metadata"])
        return results

    def get_records(self, ids: List[str]) -> list:
        """Return canonical record projections, including vectors and versions."""
        if self.__native is None:
            return []
        normalized = [self._normalize_item_id(item_id) for item_id in ids]
        return [record for record in self.__native.get_by_ids(normalized) if record is not None]

    def filter_query(self, metadata_filter, limit: int = 100) -> dict:
        """Filter one pinned native record snapshot without Gate 10 pushdown."""
        self._require_native()
        _assert(limit > 0, "limit must be greater than 0")
        expression = self._filter_expression(metadata_filter)
        result = {"id": [], "document": [], "metadata": [], "internal_id": []}
        for record in self.__native.records():
            if not _matches_filter(record["metadata"], expression):
                continue
            result["id"].append(str(record["id"]))
            result["document"].append(record["document"])
            result["metadata"].append(record["metadata"])
            result["internal_id"].append(record["upsert_sequence"])
            if len(result["id"]) == limit:
                break
        return result

    def delete_by_filter(self, metadata_filter: dict, batch_size: int = 1000) -> int:
        """Expand a pinned filter deterministically, then remove through the coordinator."""
        self._require_native()
        _assert(batch_size > 0, "batch_size must be greater than 0")
        expression = self._filter_expression(metadata_filter)
        ids = [
            str(record["id"]) for record in self.__native.records() if _matches_filter(record["metadata"], expression)
        ]
        deleted = 0
        for offset in range(0, len(ids), batch_size):
            receipt = self.__native.remove(ids[offset : offset + batch_size])
            deleted += sum(_row_status_name(row["row_status"]) == "deleted" for row in receipt["rows"])
        return deleted

    def _canonical_queries(self, vectors, *, batch: bool, compatibility_cast: bool) -> np.ndarray:
        native = self._require_native()
        del native
        array = np.asarray(vectors, dtype=self.__dtype if compatibility_cast else None)
        if batch:
            _assert(array.ndim == 2, "queries must be a 2D array")
        else:
            _assert(array.ndim == 1, "query must be a 1D array")
        if array.dtype != self.__dtype:
            raise TypeError(f"query dtype must match collection dtype {self.__dtype}")
        expected_dim = array.shape[1] if batch else array.shape[0]
        _assert(expected_dim == self.__dim, "Vector dimension must match the index dimension.")
        return np.ascontiguousarray(array)

    def search(
        self,
        query,
        top_k: int = 10,
        *,
        metadata_filter=None,
        filter_policy: str = "auto",
        filter_selectivity: Optional[float] = None,
        scratch_budget_bytes: Optional[int] = None,
        io_budget_requests: Optional[int] = None,
        io_budget_bytes: Optional[int] = None,
    ) -> dict:
        """Canonical single-query response with native Gate-10 filtering/accounting."""
        _assert(top_k >= 0, "top_k must be greater than or equal to 0")
        array = self._canonical_queries(query, batch=False, compatibility_cast=False)
        expression = None if metadata_filter is None else self._filter_expression(metadata_filter)
        unlimited = (1 << 64) - 1
        return self.__native.search(
            array,
            int(top_k),
            metadata_filter=expression,
            filter_policy=filter_policy,
            filter_selectivity=filter_selectivity,
            scratch_budget_bytes=unlimited if scratch_budget_bytes is None else int(scratch_budget_bytes),
            io_budget_requests=unlimited if io_budget_requests is None else int(io_budget_requests),
            io_budget_bytes=unlimited if io_budget_bytes is None else int(io_budget_bytes),
        )

    def batch_search(
        self,
        queries,
        top_k: int = 10,
        *,
        metadata_filter=None,
        filter_policy: str = "auto",
        filter_selectivity: Optional[float] = None,
        scratch_budget_bytes: Optional[int] = None,
        io_budget_requests: Optional[int] = None,
        io_budget_bytes: Optional[int] = None,
    ) -> dict:
        """Canonical flat response with native Gate-10 filtering/accounting."""
        _assert(top_k >= 0, "top_k must be greater than or equal to 0")
        array = self._canonical_queries(queries, batch=True, compatibility_cast=False)
        expression = None if metadata_filter is None else self._filter_expression(metadata_filter)
        unlimited = (1 << 64) - 1
        return self.__native.batch_search(
            array,
            int(top_k),
            metadata_filter=expression,
            filter_policy=filter_policy,
            filter_selectivity=filter_selectivity,
            scratch_budget_bytes=unlimited if scratch_budget_bytes is None else int(scratch_budget_bytes),
            io_budget_requests=unlimited if io_budget_requests is None else int(io_budget_requests),
            io_budget_bytes=unlimited if io_budget_bytes is None else int(io_budget_bytes),
        )

    @staticmethod
    def _response_rows(response: dict):
        for begin, end in zip(response["offsets"][:-1], response["offsets"][1:]):
            yield int(begin), int(end)

    def batch_query(
        self,
        vectors: List[List[float]],
        limit: int,
        ef_search: int = 100,
        num_threads: int = 1,
    ) -> dict:
        """Compatibility projection over canonical ``batch_search``."""
        self._require_native()
        _assert(num_threads > 0, "num_threads must be greater than 0")
        _assert(ef_search >= limit, "ef_search must be greater than or equal to limit")
        _assert(limit >= 0, "limit must be greater than or equal to 0")
        queries = self._canonical_queries(vectors, batch=True, compatibility_cast=True)
        response = self.__native.batch_search(queries, int(limit))
        unique_ids = [str(value) for value in response["ids"].tolist()]
        records = {
            str(record["id"]): record
            for record in self.__native.get_by_ids(list(dict.fromkeys(unique_ids)))
            if record is not None
        }
        result = {"id": [], "document": [], "metadata": [], "distance": []}
        for begin, end in self._response_rows(response):
            row_ids = [str(value) for value in response["ids"][begin:end].tolist()]
            result["id"].append(row_ids)
            result["document"].append([records[item_id]["document"] for item_id in row_ids])
            result["metadata"].append([records[item_id]["metadata"] for item_id in row_ids])
            result["distance"].append([float(value) for value in response["distances"][begin:end]])
        return result

    def hybrid_query(
        self,
        vectors: List[List[float]],
        limit: int,
        *,
        metadata_filter=None,
        ef_search: int = 100,
        num_threads: int = 1,
        filter_execution_hint: Optional[str] = None,
    ) -> dict:
        """Compatibility projection over strict native Gate-10 filtering."""
        self._require_native()
        _assert(ef_search >= limit, "ef_search must be >= limit")
        _assert(num_threads > 0, "num_threads must be greater than 0")
        normalize_filter_execution_hint(filter_execution_hint)
        expression = self._filter_expression(metadata_filter)
        queries = self._canonical_queries(vectors, batch=True, compatibility_cast=True)
        response = self.__native.batch_search(
            queries,
            int(limit),
            metadata_filter=expression,
            filter_policy="strict",
        )
        rows = []
        for begin, end in self._response_rows(response):
            rows.append([str(value) for value in response["ids"][begin:end].tolist()])
        return {"id": rows}

    def stats(self) -> dict:
        """Return searchable, accepted, pending, and byte accounting."""
        return self._require_native().stats()

    def size(self) -> int:
        return int(self.stats()["size"])

    def checkpoint(self) -> dict:
        return self._require_native().checkpoint()

    def reindex(self, ef_construction: int = 400, num_threads: Optional[int] = None):
        """Export, recreate, re-add, atomically swap, and checkpoint the native owner."""
        _assert(int(ef_construction) > 0, "ef_construction must be greater than 0")
        native = self._require_native()
        native.checkpoint()
        records = native.records()
        options = native.options()
        threads = options["build_threads"] if num_threads is None else num_threads
        _assert(int(threads) > 0, "num_threads must be greater than 0")
        root = Path(self.__root)
        replacement_root = root.parent / f".{root.name}.reindex-{uuid.uuid4().hex}"
        backup_root = root.parent / f".{root.name}.backup-{uuid.uuid4().hex}"
        replacement = None
        try:
            replacement = _NativeCollection.create(
                os.fspath(replacement_root),
                int(options["dim"]),
                options["metric"],
                options["dtype"],
                options["index_type"],
                options["quantization_type"],
                int(threads),
                int(ef_construction),
            )
            if records:
                ids = [str(record["id"]) for record in records]
                documents = [record["document"] for record in records]
                metadata = [record["metadata"] for record in records]
                vectors = np.ascontiguousarray(np.stack([record["vector"] for record in records]))
                replacement.mutate(ids, documents, vectors, metadata, "add", mode="all_or_nothing")
            replacement.checkpoint()
            replacement.close()
            replacement = None
            native.close()
            self.__native = None
            self.__read_view = None
            try:
                os.replace(root, backup_root)
            except Exception:
                self.__native = _NativeCollection.open(os.fspath(root))
                raise
            try:
                os.replace(replacement_root, root)
            except Exception:
                os.replace(backup_root, root)
                self.__native = _NativeCollection.open(os.fspath(root))
                raise
            try:
                self.__native = _NativeCollection.open(os.fspath(root))
                self.__read_view = None
            except Exception:
                failed_root = root.parent / f".{root.name}.failed-{uuid.uuid4().hex}"
                os.replace(root, failed_root)
                os.replace(backup_root, root)
                try:
                    self.__native = _NativeCollection.open(os.fspath(root))
                finally:
                    shutil.rmtree(failed_root)
                raise
            shutil.rmtree(backup_root)
            self._persist_discovery_schema()
            return self.__native.checkpoint()
        finally:
            if replacement is not None:
                try:
                    replacement.close()
                except RuntimeError:
                    pass
            if replacement_root.exists():
                shutil.rmtree(replacement_root)

    def save(self, url):
        """Checkpoint and export the canonical directory without writing a legacy layout."""
        native = self._require_native()
        native.checkpoint()
        destination = Path(url).absolute()
        source = Path(self.__root).absolute()
        if destination != source:
            staging = destination.parent / f".{destination.name}.canonical-{uuid.uuid4().hex}"
            if staging.exists():
                shutil.rmtree(staging)
            shutil.copytree(source, staging)
            if destination.exists():
                if any(destination.iterdir()):
                    shutil.rmtree(staging)
                    raise RuntimeError(f"Collection save target is not empty: {destination}")
                destination.rmdir()
            os.replace(staging, destination)
        schema = {
            "type": "collection",
            "format": "canonical_collection_v1",
            "public_version": V_PUBLIC,
            "index": self.__index_params.to_json_dict(),
        }
        return schema

    @classmethod
    def load(cls, url, name):
        """Open canonical data or invoke the native non-destructive legacy importer."""
        collection_url = os.path.join(url, name)
        if not os.path.exists(collection_url):
            raise RuntimeError(f"Collection {name} does not exist")
        schema_url = os.path.join(collection_url, "schema.json")
        schema_map = load_schema(schema_url)
        if schema_map.get("type") != "collection":
            raise RuntimeError(f"{name} is not a collection")
        index_params = IndexParams.from_str_dict(schema_map["index"])
        if not index_params.rocksdb_path:
            index_params.rocksdb_path = _default_rocksdb_path(name, url)
        instance = cls(name, index_params)
        instance._bind_open_native(collection_url)
        return instance

    def set_metric(self, metric: str):
        if self.__native is not None:
            raise RuntimeError("Cannot change metric after index is created")
        self.__index_params.metric = metric

    def get_index_params(self):
        return self.__index_params

    def get_index(self):
        """Return the same read-only native view; no legacy Index owner exists."""
        if self.__native is None:
            return None
        return self._get_cpp_index()

    def close(self):
        if self.__native is not None:
            self.__native.close()
            self.__native = None
            self.__read_view = None

    def __del__(self):
        try:
            self.close()
        except (RuntimeError, AttributeError):
            pass
