# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""Shared fixtures for the SDK v2 executable contract."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture(name="sdk")
def sdk_fixture() -> Any:
    return importlib.import_module("alayalite")


@pytest.fixture(name="flat_config")
def flat_config_fixture(sdk: Any) -> Any:
    return sdk.CollectionConfig(
        dimension=3,
        dtype="float32",
        metric="l2",
        index=sdk.FlatIndexConfig(),
    )


@pytest.fixture
def flat_collection(sdk: Any, flat_config: Any, tmp_path: Any) -> Iterator[Any]:
    database = sdk.connect(tmp_path / "database")
    collection = database.create_collection("items", config=flat_config)
    try:
        yield collection
    finally:
        collection.close()
        database.close()
