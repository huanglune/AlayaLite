# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""HTTP request models for the SDK v2 example service."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

MetadataScalar = bool | int | float | str


class FlatIndexRequest(BaseModel):
    """Select exact Flat search."""

    kind: Literal["flat"] = "flat"


class QGIndexRequest(BaseModel):
    """Select the platform-gated QG index family."""

    kind: Literal["qg"] = "qg"
    max_neighbors: int = 32
    construction_effort: int = 400
    build_threads: int | None = None


IndexRequest = Annotated[FlatIndexRequest | QGIndexRequest, Field(discriminator="kind")]


class CreateCollectionRequest(BaseModel):
    """Create a collection with an explicit immutable schema."""

    collection_name: str
    dimension: int
    dtype: Literal["float32", "int8", "uint8"] = "float32"
    metric: Literal["l2", "ip", "cosine"] = "l2"
    index: IndexRequest = Field(default_factory=FlatIndexRequest)
    auto_seal_rows: int | None = None


class DropCollectionRequest(BaseModel):
    """Permanently remove a collection."""

    collection_name: str
    missing_ok: bool = False


class WriteCollectionRequest(BaseModel):
    """Columnar add/upsert payload."""

    collection_name: str
    ids: list[str]
    vectors: list[list[float]]
    documents: list[str] | None = None
    metadata: list[dict[str, MetadataScalar] | None] | None = None
    mode: Literal["atomic", "partial"] = "atomic"
    durability: Literal["fsync", "buffered"] = "fsync"
    idempotency_key: str | None = None


class SearchCollectionRequest(BaseModel):
    """CSR search request."""

    collection_name: str
    queries: list[list[float]]
    limit: int = 10
    where: dict[str, Any] | None = None
    effort: int | None = None
    filter_policy: Literal["auto", "strict", "allow_partial"] = "auto"
    selectivity_hint: float | None = None


class GetRecordsRequest(BaseModel):
    """Read records in request order, preserving missing positions."""

    collection_name: str
    ids: list[str]
    include_vector: bool = False


class DeleteRequest(BaseModel):
    """Delete logical IDs while preserving row receipts."""

    collection_name: str
    ids: list[str]
    mode: Literal["atomic", "partial"] = "atomic"
    durability: Literal["fsync", "buffered"] = "fsync"
    idempotency_key: str | None = None


class DeleteWhereRequest(BaseModel):
    """Delete records matched by a non-empty metadata filter."""

    collection_name: str
    where: dict[str, Any]
    batch_size: int = 1000
    durability: Literal["fsync", "buffered"] = "fsync"


class CheckpointCollectionRequest(BaseModel):
    """Create a durable recovery point for one collection."""

    collection_name: str


__all__ = [
    "CheckpointCollectionRequest",
    "CreateCollectionRequest",
    "DeleteRequest",
    "DeleteWhereRequest",
    "DropCollectionRequest",
    "FlatIndexRequest",
    "GetRecordsRequest",
    "QGIndexRequest",
    "SearchCollectionRequest",
    "WriteCollectionRequest",
]
