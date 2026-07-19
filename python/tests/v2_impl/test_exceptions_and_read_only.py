# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Live internal-core counterparts for the seven exception/read-only goldens."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from alayalite._database import connect


def _directory_bytes_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _create_closed_collection(root, flat_config):
    with connect(root) as database:
        collection = database.create_collection("docs", config=flat_config)
        collection.add(ids=["a"], vectors=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
        collection.close()


def test_exception_taxonomy_has_exact_python_base_relationships(sdk):
    assert issubclass(sdk.CollectionStatusError, RuntimeError)
    assert issubclass(sdk.CollectionInvalidArgumentError, (sdk.CollectionStatusError, ValueError))
    assert issubclass(sdk.CollectionNotSupportedError, (sdk.CollectionStatusError, NotImplementedError))
    assert issubclass(sdk.CollectionNotFoundError, (sdk.CollectionStatusError, KeyError))

    for name in (
        "CollectionConflictError",
        "CollectionResourceExhaustedError",
        "CollectionDeadlineExceededError",
        "CollectionCancelledError",
        "CollectionIoError",
        "CollectionCorruptionError",
        "CollectionClosedError",
        "CollectionInternalError",
    ):
        exception = getattr(sdk, name)
        assert issubclass(exception, sdk.CollectionStatusError)
        assert issubclass(exception, RuntimeError)


def test_native_status_metadata_survives_public_exception_mapping(sdk, tmp_path):
    with connect(tmp_path / "database") as database:
        with pytest.raises(sdk.CollectionNotFoundError) as captured:
            database.open_collection("missing")

    error = captured.value
    assert isinstance(error.status_code, int)
    assert isinstance(error.operation_stage, int)
    assert isinstance(error.status_detail, int)
    assert isinstance(error.retryability, int)
    assert error.partial is False
    assert error.status_version == "1"


def test_read_only_connect_does_not_change_any_collection_bytes(flat_config, tmp_path):
    root = tmp_path / "database"
    _create_closed_collection(root, flat_config)
    before = _directory_bytes_hash(root)

    with connect(root, read_only=True) as database:
        assert database.read_only is True
        with database.open_collection("docs") as collection:
            assert collection.info.read_only is True
            assert collection.count() == 1
            assert collection.get(["a"])[0].id == "a"
            assert collection.search(np.asarray([1.0, 2.0, 3.0], dtype=np.float32), limit=1)[0].ids.tolist() == ["a"]

    assert _directory_bytes_hash(root) == before


def test_read_only_database_rejects_catalog_mutation(sdk, flat_config, tmp_path):
    root = tmp_path / "database"
    _create_closed_collection(root, flat_config)

    with connect(root, read_only=True) as database:
        with pytest.raises(sdk.CollectionNotSupportedError):
            database.create_collection("new", config=flat_config)
        with pytest.raises(sdk.CollectionNotSupportedError):
            database.drop_collection("docs")


def test_read_only_collection_rejects_every_mutation_and_control_method(sdk, flat_config, tmp_path):
    root = tmp_path / "database"
    _create_closed_collection(root, flat_config)
    vector = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)

    with connect(root, read_only=True) as database:
        with database.open_collection("docs") as collection:
            operations = (
                lambda: collection.add(ids=["b"], vectors=vector),
                lambda: collection.replace(ids=["a"], vectors=vector),
                lambda: collection.upsert(ids=["a"], vectors=vector),
                lambda: collection.delete(["a"]),
                lambda: collection.delete_where({"kind": "anything"}),
                collection.checkpoint,
                collection.seal,
                collection.compact,
                collection.collect_garbage,
                collection.rebuild_index,
            )
            for operation in operations:
                with pytest.raises(sdk.CollectionNotSupportedError) as captured:
                    operation()
                assert captured.value.status_detail != 0


def test_read_only_database_cannot_upgrade_a_collection_handle(sdk, flat_config, tmp_path):
    root = tmp_path / "database"
    _create_closed_collection(root, flat_config)
    with connect(root, read_only=True) as database:
        with pytest.raises(sdk.CollectionNotSupportedError):
            database.open_collection("docs", read_only=False)


def test_read_write_database_can_narrow_one_collection_handle(sdk, flat_config, tmp_path):
    root = tmp_path / "database"
    _create_closed_collection(root, flat_config)
    with connect(root) as database:
        with database.open_collection("docs", read_only=True) as collection:
            assert collection.info.read_only is True
            with pytest.raises(sdk.CollectionNotSupportedError):
                collection.delete(["a"])
