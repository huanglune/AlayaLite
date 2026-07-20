# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""FastAPI routes backed exclusively by the SDK v2 surface."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
from alayalite import (
    CollectionClosedError,
    CollectionConfig,
    CollectionConflictError,
    CollectionInvalidArgumentError,
    CollectionNotFoundError,
    CollectionNotSupportedError,
    CollectionStatusError,
    Database,
    FlatIndexConfig,
    MutationResult,
    QGIndexConfig,
    SearchResult,
)
from alayalite.models import CheckpointReceipt, DeleteResult, Record
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.models.collection import (
    CheckpointCollectionRequest,
    CreateCollectionRequest,
    DeleteRequest,
    DeleteWhereRequest,
    DropCollectionRequest,
    FlatIndexRequest,
    GetRecordsRequest,
    SearchCollectionRequest,
    WriteCollectionRequest,
)

router = APIRouter()


def _database(request: Request) -> Database:
    database = getattr(request.app.state, "database", None)
    if not isinstance(database, Database):
        raise RuntimeError("AlayaLite database lifespan has not started")
    return database


def _error_response(error: CollectionStatusError | TypeError | ValueError) -> JSONResponse:
    if isinstance(error, CollectionNotFoundError):
        code = status.HTTP_404_NOT_FOUND
    elif isinstance(error, CollectionConflictError):
        code = status.HTTP_409_CONFLICT
    elif isinstance(error, CollectionClosedError):
        code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(error, (CollectionInvalidArgumentError, CollectionNotSupportedError, TypeError, ValueError)):
        code = status.HTTP_400_BAD_REQUEST
    else:
        code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return JSONResponse(status_code=code, content={"error": str(error)})


def _mutation_json(result: MutationResult) -> dict[str, Any]:
    return {
        "batch_op_id": result.batch_op_id,
        "visibility_watermark": result.visibility_watermark,
        "durable_watermark": result.durable_watermark,
        "searchable": result.searchable,
        "durable": result.durable,
        "durability": result.durability.value,
        "idempotency_key": result.idempotency_key,
        "rows": [
            {
                "id": row.id,
                "op_id": row.op_id,
                "batch_op_id": row.batch_op_id,
                "row_op_id": row.row_op_id,
                "visibility_watermark": row.visibility_watermark,
                "durable_watermark": row.durable_watermark,
                "searchable": row.searchable,
                "durability": row.durability.value,
                "status": row.status.value,
                "idempotency_key": row.idempotency_key,
            }
            for row in result.rows
        ],
    }


def _search_json(result: SearchResult) -> dict[str, Any]:
    return {
        "ids": result.ids.tolist(),
        "distances": result.distances.tolist(),
        "offsets": result.offsets.tolist(),
        "valid_counts": result.valid_counts.tolist(),
        "statuses": [value.value for value in result.statuses],
        "completeness": [value.value for value in result.completeness],
        "visibility_watermark": result.visibility_watermark,
        "metadata_epoch": result.metadata_epoch,
        "stats": asdict(result.stats),
    }


def _record_json(record: Record | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "id": record.id,
        "document": record.document,
        "metadata": dict(record.metadata),
        "version": record.version,
        "vector": None if record.vector is None else record.vector.tolist(),
    }


@router.post("/collections/create", tags=["collections"])
async def create_collection(request: Request, payload: CreateCollectionRequest) -> Any:
    index = (
        FlatIndexConfig()
        if isinstance(payload.index, FlatIndexRequest)
        else QGIndexConfig(
            max_neighbors=payload.index.max_neighbors,
            construction_effort=payload.index.construction_effort,
            build_threads=payload.index.build_threads,
        )
    )
    try:
        config = CollectionConfig(
            dimension=payload.dimension,
            dtype=payload.dtype,
            metric=payload.metric,
            index=index,
            auto_seal_rows=payload.auto_seal_rows,
        )
        collection = _database(request).create_collection(payload.collection_name, config=config)
        collection.close()
        return {"name": payload.collection_name, "config": asdict(config)}
    except (CollectionStatusError, TypeError, ValueError) as error:
        return _error_response(error)


@router.get("/collections", tags=["collections"])
async def list_collections(request: Request) -> list[str]:
    return _database(request).list_collections()


@router.post("/collections/drop", tags=["collections"])
async def drop_collection(request: Request, payload: DropCollectionRequest) -> Any:
    try:
        _database(request).drop_collection(payload.collection_name, missing_ok=payload.missing_ok)
        return {"dropped": payload.collection_name}
    except (CollectionStatusError, TypeError, ValueError) as error:
        return _error_response(error)


@router.post("/collections/add", tags=["records"])
async def add_records(request: Request, payload: WriteCollectionRequest) -> Any:
    try:
        with _database(request).open_collection(payload.collection_name) as collection:
            result = collection.add(
                ids=payload.ids,
                vectors=np.asarray(payload.vectors),
                documents=payload.documents,
                metadata=payload.metadata,
                mode=payload.mode,
                durability=payload.durability,
                idempotency_key=payload.idempotency_key,
            )
        return _mutation_json(result)
    except (CollectionStatusError, TypeError, ValueError) as error:
        return _error_response(error)


@router.post("/collections/upsert", tags=["records"])
async def upsert_records(request: Request, payload: WriteCollectionRequest) -> Any:
    try:
        with _database(request).open_collection(payload.collection_name) as collection:
            result = collection.upsert(
                ids=payload.ids,
                vectors=np.asarray(payload.vectors),
                documents=payload.documents,
                metadata=payload.metadata,
                mode=payload.mode,
                durability=payload.durability,
                idempotency_key=payload.idempotency_key,
            )
        return _mutation_json(result)
    except (CollectionStatusError, TypeError, ValueError) as error:
        return _error_response(error)


@router.post("/collections/get", tags=["records"])
async def get_records(request: Request, payload: GetRecordsRequest) -> Any:
    try:
        with _database(request).open_collection(payload.collection_name) as collection:
            records = collection.get(payload.ids, include_vector=payload.include_vector)
        return [_record_json(record) for record in records]
    except (CollectionStatusError, TypeError, ValueError) as error:
        return _error_response(error)


@router.post("/collections/search", tags=["search"])
async def search_collection(request: Request, payload: SearchCollectionRequest) -> Any:
    try:
        with _database(request).open_collection(payload.collection_name) as collection:
            result = collection.search(
                np.asarray(payload.queries),
                limit=payload.limit,
                where=payload.where,
                effort=payload.effort,
                filter_policy=payload.filter_policy,
                selectivity_hint=payload.selectivity_hint,
            )
        return _search_json(result)
    except (CollectionStatusError, TypeError, ValueError) as error:
        return _error_response(error)


@router.post("/collections/delete", tags=["records"])
async def delete_records(request: Request, payload: DeleteRequest) -> Any:
    try:
        with _database(request).open_collection(payload.collection_name) as collection:
            result = collection.delete(
                payload.ids,
                mode=payload.mode,
                durability=payload.durability,
                idempotency_key=payload.idempotency_key,
            )
        return _mutation_json(result)
    except (CollectionStatusError, TypeError, ValueError) as error:
        return _error_response(error)


@router.post("/collections/delete-where", tags=["records"])
async def delete_where(request: Request, payload: DeleteWhereRequest) -> Any:
    try:
        with _database(request).open_collection(payload.collection_name) as collection:
            result: DeleteResult = collection.delete_where(
                payload.where,
                batch_size=payload.batch_size,
                durability=payload.durability,
            )
        return asdict(result)
    except (CollectionStatusError, TypeError, ValueError) as error:
        return _error_response(error)


@router.post("/collections/checkpoint", tags=["maintenance"])
async def checkpoint_collection(request: Request, payload: CheckpointCollectionRequest) -> Any:
    try:
        with _database(request).open_collection(payload.collection_name) as collection:
            receipt: CheckpointReceipt = collection.checkpoint()
        return asdict(receipt)
    except (CollectionStatusError, TypeError, ValueError) as error:
        return _error_response(error)
