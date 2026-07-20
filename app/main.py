# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from alayalite import connect
from fastapi import FastAPI

from app.routers import client


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Own one Database for the complete application lifetime."""
    storage = Path(os.environ.get("ALAYALITE_DATA_DIR", "./data"))
    database = connect(storage)
    application.state.database = database
    try:
        yield
    finally:
        database.close()


app = FastAPI(
    title="AlayaLite embedded database service",
    description="Example HTTP adapter for the AlayaLite SDK v2 contract",
    version="2.0",
    lifespan=lifespan,
)

app.include_router(client.router, prefix="/api/v2")


@app.get("/")
async def root():
    return {"message": "AlayaLite service is ready; use /api/v2 for the example HTTP API."}
