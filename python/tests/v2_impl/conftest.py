# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Live fixtures for the dark-launched SDK v2 Python core."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from alayalite._capabilities import capabilities
from alayalite._collection import Collection
from alayalite._database import Database, connect
from alayalite.config import CollectionConfig, FlatIndexConfig, QGIndexConfig
from alayalite.exceptions import (
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
from alayalite.models import Capabilities, MutationResult, Record, SearchResult


@pytest.fixture(name="sdk")
def sdk_fixture() -> SimpleNamespace:
    """Expose the future root surface while importing only internal modules."""
    return SimpleNamespace(
        connect=connect,
        capabilities=capabilities,
        Database=Database,
        Collection=Collection,
        CollectionConfig=CollectionConfig,
        FlatIndexConfig=FlatIndexConfig,
        QGIndexConfig=QGIndexConfig,
        Capabilities=Capabilities,
        Record=Record,
        SearchResult=SearchResult,
        MutationResult=MutationResult,
        CollectionStatusError=CollectionStatusError,
        CollectionInvalidArgumentError=CollectionInvalidArgumentError,
        CollectionNotSupportedError=CollectionNotSupportedError,
        CollectionConflictError=CollectionConflictError,
        CollectionNotFoundError=CollectionNotFoundError,
        CollectionResourceExhaustedError=CollectionResourceExhaustedError,
        CollectionDeadlineExceededError=CollectionDeadlineExceededError,
        CollectionCancelledError=CollectionCancelledError,
        CollectionIoError=CollectionIoError,
        CollectionCorruptionError=CollectionCorruptionError,
        CollectionClosedError=CollectionClosedError,
        CollectionInternalError=CollectionInternalError,
    )


@pytest.fixture(name="flat_config")
def flat_config_fixture() -> CollectionConfig:
    """Return the small exact-search config used by most live tests."""
    return CollectionConfig(
        dimension=3,
        dtype="float32",
        metric="l2",
        index=FlatIndexConfig(),
    )


@pytest.fixture
def flat_collection(flat_config: CollectionConfig, tmp_path) -> Iterator[Collection]:
    """Yield one empty live Flat collection and close both owners."""
    database = connect(tmp_path / "database")
    collection = database.create_collection("items", config=flat_config)
    try:
        yield collection
    finally:
        collection.close()
        database.close()
