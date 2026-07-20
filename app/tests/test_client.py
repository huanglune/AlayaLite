# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""End-to-end tests for the v2 FastAPI adapter."""

from __future__ import annotations

import importlib
import sys

import httpx
import pytest


def _create_payload(name: str, *, metric: str = "l2") -> dict[str, object]:
    return {
        "collection_name": name,
        "dimension": 3,
        "dtype": "float32",
        "metric": metric,
        "index": {"kind": "flat"},
    }


def _write_payload(name: str) -> dict[str, object]:
    return {
        "collection_name": name,
        "ids": ["one", "two", "three"],
        "vectors": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        "documents": ["first", "second", "third"],
        "metadata": [{"kind": "keep"}, {"kind": "drop"}, {"kind": "keep"}],
    }


@pytest.mark.asyncio
async def test_basic_crud_search_checkpoint_and_drop(fresh_client: httpx.AsyncClient) -> None:
    created = await fresh_client.post("/api/v2/collections/create", json=_create_payload("docs"))
    assert created.status_code == 200
    assert created.json()["config"]["index"] == {"kind": "flat"}

    duplicate = await fresh_client.post("/api/v2/collections/create", json=_create_payload("docs"))
    assert duplicate.status_code == 409

    added = await fresh_client.post("/api/v2/collections/add", json=_write_payload("docs"))
    assert added.status_code == 200
    assert [row["status"] for row in added.json()["rows"]] == ["inserted"] * 3

    fetched = await fresh_client.post(
        "/api/v2/collections/get",
        json={"collection_name": "docs", "ids": ["two", "missing", "one"], "include_vector": True},
    )
    assert fetched.status_code == 200
    assert [None if row is None else row["id"] for row in fetched.json()] == ["two", None, "one"]
    assert fetched.json()[0]["vector"] == [1.0, 0.0, 0.0]

    searched = await fresh_client.post(
        "/api/v2/collections/search",
        json={
            "collection_name": "docs",
            "queries": [[0.0, 0.0, 0.0]],
            "limit": 2,
            "where": {"kind": "keep"},
        },
    )
    assert searched.status_code == 200
    body = searched.json()
    assert body["ids"] == ["one", "three"]
    assert body["offsets"] == [0, 2]
    assert body["valid_counts"] == [2]
    assert body["statuses"] == ["ok"]

    upserted = await fresh_client.post(
        "/api/v2/collections/upsert",
        json={
            "collection_name": "docs",
            "ids": ["two"],
            "vectors": [[0.5, 0.0, 0.0]],
            "documents": ["second-v2"],
            "metadata": [{"kind": "keep"}],
        },
    )
    assert upserted.status_code == 200
    assert upserted.json()["rows"][0]["status"] == "updated"

    deleted = await fresh_client.post(
        "/api/v2/collections/delete",
        json={"collection_name": "docs", "ids": ["three", "missing"]},
    )
    assert deleted.status_code == 200
    assert [row["status"] for row in deleted.json()["rows"]] == ["deleted", "not_found"]

    deleted_where = await fresh_client.post(
        "/api/v2/collections/delete-where",
        json={"collection_name": "docs", "where": {"kind": "keep"}, "batch_size": 1},
    )
    assert deleted_where.status_code == 200
    assert deleted_where.json() == {"matched": 2, "deleted": 2, "not_found": 0, "batches": 2}

    checkpoint = await fresh_client.post("/api/v2/collections/checkpoint", json={"collection_name": "docs"})
    assert checkpoint.status_code == 200
    assert checkpoint.json()["checkpoint_name"]

    listed = await fresh_client.get("/api/v2/collections")
    assert listed.json() == ["docs"]
    dropped = await fresh_client.post("/api/v2/collections/drop", json={"collection_name": "docs"})
    assert dropped.status_code == 200
    assert (await fresh_client.get("/api/v2/collections")).json() == []


@pytest.mark.asyncio
async def test_typed_not_found_and_validation_errors(fresh_client: httpx.AsyncClient) -> None:
    missing = await fresh_client.post(
        "/api/v2/collections/search",
        json={"collection_name": "missing", "queries": [[0.0, 0.0, 0.0]]},
    )
    assert missing.status_code == 404
    assert "error" in missing.json()

    bad_config = await fresh_client.post(
        "/api/v2/collections/create",
        json={**_create_payload("bad"), "dimension": 0},
    )
    assert bad_config.status_code == 400

    assert (await fresh_client.get("/")).json()["message"].startswith("AlayaLite service")


@pytest.mark.asyncio
async def test_metric_is_fixed_at_create_time(fresh_client: httpx.AsyncClient) -> None:
    created = await fresh_client.post("/api/v2/collections/create", json=_create_payload("cos", metric="cosine"))
    assert created.status_code == 200
    added = await fresh_client.post(
        "/api/v2/collections/add",
        json={
            "collection_name": "cos",
            "ids": ["opposite", "same"],
            "vectors": [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        },
    )
    assert added.status_code == 200
    searched = await fresh_client.post(
        "/api/v2/collections/search",
        json={"collection_name": "cos", "queries": [[1.0, 0.0, 0.0]], "limit": 2},
    )
    assert searched.status_code == 200
    assert searched.json()["ids"] == ["same", "opposite"]


def _reload_application():
    for name in tuple(sys.modules):
        if name == "app" or name.startswith("app."):
            del sys.modules[name]
    return importlib.import_module("app.main").app


@pytest.mark.asyncio
async def test_lifespan_close_and_reopen_preserve_data(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ALAYALITE_DATA_DIR", str(tmp_path))
    first_app = _reload_application()
    async with first_app.router.lifespan_context(first_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=first_app), base_url="http://testserver"
        ) as first:
            assert (
                await first.post("/api/v2/collections/create", json=_create_payload("persisted"))
            ).status_code == 200
            assert (await first.post("/api/v2/collections/add", json=_write_payload("persisted"))).status_code == 200
            assert (
                await first.post("/api/v2/collections/checkpoint", json={"collection_name": "persisted"})
            ).status_code == 200

    second_app = _reload_application()
    async with second_app.router.lifespan_context(second_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=second_app), base_url="http://testserver"
        ) as second:
            assert (await second.get("/api/v2/collections")).json() == ["persisted"]
            records = await second.post(
                "/api/v2/collections/get", json={"collection_name": "persisted", "ids": ["one"]}
            )
            assert records.status_code == 200
            assert records.json()[0]["document"] == "first"
