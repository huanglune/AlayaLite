# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""FastAPI fixtures with real application lifespan handling."""

from __future__ import annotations

import importlib
import sys
from collections.abc import AsyncGenerator

import httpx
import pytest_asyncio


def reload_app():
    """Build a fresh application after environment changes."""
    for name in tuple(sys.modules):
        if name == "app" or name.startswith("app."):
            del sys.modules[name]
    return importlib.import_module("app.main").app


@pytest_asyncio.fixture()
async def fresh_client(tmp_path, monkeypatch) -> AsyncGenerator[httpx.AsyncClient, None]:
    monkeypatch.setenv("ALAYALITE_DATA_DIR", str(tmp_path))
    application = reload_app()
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client
