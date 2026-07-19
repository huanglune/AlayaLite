# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""Shared activation guard and fixtures for the SDK v2 executable contract."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from typing import Any

import pytest

_V2_ROOT_SYMBOLS = frozenset(
    {
        "connect",
        "capabilities",
        "Database",
        "Collection",
        "CollectionConfig",
        "FlatIndexConfig",
        "QGIndexConfig",
        "Capabilities",
        "Record",
        "SearchResult",
        "MutationResult",
    }
)


def _v2_surface_is_ready() -> bool:
    try:
        sdk = importlib.import_module("alayalite")
    except ImportError:
        return False
    return _V2_ROOT_SYMBOLS.issubset(vars(sdk))


@pytest.fixture(autouse=True)
def _require_complete_v2_surface() -> None:
    """Keep every golden skipped behind one guard until wave C activates it."""
    if not _v2_surface_is_ready():
        pytest.skip("SDK v2 public core has not landed; contract goldens are dormant")


@pytest.fixture
def sdk() -> Any:
    return importlib.import_module("alayalite")


@pytest.fixture
def flat_config(sdk: Any) -> Any:
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
