# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Live internal-core counterparts for the 15 lifecycle goldens."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from alayalite import __version__
from alayalite._collection import Collection
from alayalite._database import Database, connect
from alayalite.config import CollectionConfig, FlatIndexConfig, QGIndexConfig
from alayalite.models import Capabilities, MutationResult, Record, SearchResult

_EXCEPTIONS = {
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
}

_ROOT_API = [
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
    *_EXCEPTIONS,
]


def _parameter_names(callable_object):
    return tuple(inspect.signature(callable_object).parameters)


def test_root_all_is_the_exact_v2_public_surface(sdk):
    """Assemble the future root set without changing the current root."""
    assert set(vars(sdk)) == set(_ROOT_API)
    assert len(vars(sdk)) == len(_ROOT_API) == 23
    assert __version__ == "1.1.0"


def test_public_models_have_one_canonical_import_location(sdk):
    assert sdk.CollectionConfig is CollectionConfig
    assert sdk.FlatIndexConfig is FlatIndexConfig
    assert sdk.QGIndexConfig is QGIndexConfig
    assert sdk.Capabilities is Capabilities
    assert sdk.Record is Record
    assert sdk.SearchResult is SearchResult
    assert sdk.MutationResult is MutationResult
    assert CollectionConfig.__module__ == "alayalite.config"
    assert SearchResult.__module__ == "alayalite.models"


def test_removed_root_symbols_and_modules_are_really_absent(sdk):
    for name in (
        "Client",
        "MetricType",
        "IndexParams",
        "Index",
        "DiskCollection",
        "load_fvecs",
        "load_ivecs",
        "calc_recall",
        "calc_gt",
    ):
        assert not hasattr(sdk, name)


def test_connect_and_database_signatures_are_stable():
    assert _parameter_names(connect) == ("path", "read_only")
    signature = inspect.signature(connect)
    assert signature.parameters["path"].default is None
    assert signature.parameters["read_only"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["read_only"].default is False

    assert _parameter_names(Database.create_collection) == ("self", "name", "config")
    assert _parameter_names(Database.open_collection) == ("self", "name", "read_only")
    assert _parameter_names(Database.drop_collection) == ("self", "name", "missing_ok")


def test_collection_signature_vocabulary_has_no_legacy_spellings():
    assert _parameter_names(Collection.search) == (
        "self",
        "queries",
        "limit",
        "where",
        "effort",
        "filter_policy",
        "selectivity_hint",
        "budget",
    )
    assert _parameter_names(Collection.scan) == ("self", "where", "limit", "include_vector")
    assert _parameter_names(Collection.get) == ("self", "ids", "include_vector")
    for name in (
        "insert",
        "mutate_batch",
        "remove",
        "delete_by_id",
        "get_by_id",
        "get_records",
        "filter_query",
        "batch_search",
        "batch_query",
        "hybrid_query",
        "options",
        "size",
        "gc",
        "reindex",
        "save",
        "flush",
        "set_metric",
        "build_filter",
    ):
        assert not hasattr(Collection, name)


def test_collection_cannot_be_constructed_outside_database():
    with pytest.raises(TypeError):
        Collection()  # pylint: disable=no-value-for-parameter,missing-kwoa


def test_database_context_materializes_and_reopens_empty_collection(sdk, flat_config, tmp_path):
    root = tmp_path / "database"
    with connect(root) as database:
        assert database.path == root.resolve()
        assert database.read_only is False
        assert database.list_collections() == []

        created = database.create_collection("docs", config=flat_config)
        assert database.list_collections() == ["docs"]
        assert created.count() == 0
        assert created.config == flat_config
        created.close()

        with database.open_collection("docs") as reopened:
            assert reopened.count() == 0

    database.close()
    with pytest.raises(sdk.CollectionClosedError):
        database.list_collections()


def test_connect_is_lazy_in_the_presence_of_unknown_directories(tmp_path):
    root = tmp_path / "database"
    broken = root / "not-a-collection"
    broken.mkdir(parents=True)
    (broken / "garbage.bin").write_bytes(b"not a collection")

    with connect(root) as database:
        assert database.list_collections() == []


def test_collection_names_are_safe_single_path_components(flat_config, tmp_path):
    with connect(tmp_path / "database") as database:
        for name in ("", ".hidden", "..", "a/b", "a\\b", "nul\0byte", "x" * 129):
            with pytest.raises(ValueError):
                database.create_collection(name, config=flat_config)


def test_open_missing_raises_typed_not_found(sdk, tmp_path):
    with connect(tmp_path / "database") as database:
        with pytest.raises(sdk.CollectionNotFoundError):
            database.open_collection("missing")


def test_create_open_and_drop_have_unambiguous_conflict_rules(sdk, flat_config, tmp_path):
    with connect(tmp_path / "database") as database:
        collection = database.create_collection("docs", config=flat_config)
        with pytest.raises(sdk.CollectionConflictError):
            database.create_collection("docs", config=flat_config)
        with pytest.raises(sdk.CollectionConflictError):
            database.drop_collection("docs")

        collection.close()
        database.drop_collection("docs")
        assert database.list_collections() == []
        with pytest.raises(sdk.CollectionNotFoundError):
            database.drop_collection("docs")
        database.drop_collection("docs", missing_ok=True)


def test_collection_context_and_close_are_idempotent(sdk, flat_config, tmp_path):
    database = connect(tmp_path / "database")
    collection = database.create_collection("docs", config=flat_config)
    with collection as entered:
        assert entered is collection
    collection.close()
    with pytest.raises(sdk.CollectionClosedError):
        collection.count()
    database.close()


def test_memory_database_lifetime_is_scoped_to_the_handle(flat_config):
    database = connect(":memory:")
    root = database.path
    assert root.is_dir()
    database.create_collection("docs", config=flat_config).close()
    database.close()
    assert not root.exists()


def test_connect_rejects_remote_uri_spellings():
    for path in ("http://example.test/db", "https://example.test/db", "s3://bucket/db", "file:///tmp/db"):
        with pytest.raises(ValueError):
            connect(path)


def test_database_path_is_a_path_object(tmp_path):
    with connect(tmp_path / "db") as database:
        assert isinstance(database.path, Path)
