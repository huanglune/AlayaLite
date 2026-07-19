# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Live internal-core counterparts for the 30 mutation goldens."""

from __future__ import annotations

import inspect
import uuid

import numpy as np
import pytest
from alayalite._collection import Collection


def _status_value(row):
    return getattr(row.status, "value", row.status)


def _write(collection, method="add", **overrides):
    arguments = {
        "ids": ["a", "b"],
        "vectors": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        "documents": ["A", "B"],
        "metadata": [{"kind": "first"}, {"kind": "second"}],
    }
    arguments.update(overrides)
    return getattr(collection, method)(**arguments)


def test_three_write_verbs_share_one_keyword_only_signature():
    expected = (
        "self",
        "ids",
        "vectors",
        "documents",
        "metadata",
        "mode",
        "durability",
        "idempotency_key",
    )
    for name in ("add", "replace", "upsert"):
        signature = inspect.signature(getattr(Collection, name))
        assert tuple(signature.parameters) == expected
        for parameter in tuple(signature.parameters.values())[1:]:
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters["mode"].default == "atomic"
        assert signature.parameters["durability"].default == "fsync"
        assert signature.parameters["idempotency_key"].default is None


def test_add_returns_typed_durable_receipt(flat_collection, sdk):
    result = _write(flat_collection)

    assert isinstance(result, sdk.MutationResult)
    assert result.durable is True
    assert result.durable_watermark >= result.visibility_watermark > 0
    assert result.idempotency_key is None
    assert [_status_value(row) for row in result.rows] == ["inserted", "inserted"]
    assert [row.id for row in result.rows] == ["a", "b"]
    assert all(row.searchable for row in result.rows)
    assert flat_collection.count() == 2


def test_atomic_is_the_default_and_prevents_partial_add(flat_collection):
    _write(flat_collection, ids=["a"], vectors=np.zeros((1, 3), dtype=np.float32), documents=None, metadata=None)
    result = _write(
        flat_collection,
        ids=["a", "new"],
        vectors=np.asarray([[1, 0, 0], [2, 0, 0]], dtype=np.float32),
        documents=None,
        metadata=None,
    )

    assert {_status_value(row) for row in result.rows} == {"already_exists", "aborted"}
    assert result.rows[0].searchable is False
    assert result.rows[1].searchable is False
    assert flat_collection.get(["new"]) == (None,)


def test_partial_mode_reports_independent_row_outcomes(flat_collection):
    _write(flat_collection, ids=["a"], vectors=np.zeros((1, 3), dtype=np.float32), documents=None, metadata=None)
    result = _write(
        flat_collection,
        ids=["a", "new"],
        vectors=np.asarray([[1, 0, 0], [2, 0, 0]], dtype=np.float32),
        documents=None,
        metadata=None,
        mode="partial",
    )

    assert [_status_value(row) for row in result.rows] == ["already_exists", "inserted"]
    assert flat_collection.get(["new"])[0] is not None


def test_replace_is_whole_row_and_requires_an_existing_id(flat_collection):
    _write(
        flat_collection,
        ids=["a"],
        vectors=np.zeros((1, 3), dtype=np.float32),
        documents=["A"],
        metadata=[{"kind": "first"}],
    )
    replaced = _write(
        flat_collection,
        method="replace",
        ids=["a"],
        vectors=np.asarray([[3, 2, 1]], dtype=np.float32),
        documents=["replacement"],
        metadata=[{"version": 2}],
    )
    missing = _write(
        flat_collection,
        method="replace",
        ids=["missing"],
        vectors=np.zeros((1, 3), dtype=np.float32),
        documents=None,
        metadata=None,
    )

    assert [_status_value(row) for row in replaced.rows] == ["replaced"]
    assert [_status_value(row) for row in missing.rows] == ["not_found"]
    record = flat_collection.get(["a"], include_vector=True)[0]
    assert record.document == "replacement"
    assert record.metadata == {"version": 2}
    np.testing.assert_array_equal(record.vector, [3, 2, 1])


def test_upsert_inserts_then_updates(flat_collection):
    inserted = _write(
        flat_collection,
        method="upsert",
        ids=["a"],
        vectors=np.zeros((1, 3), dtype=np.float32),
        documents=None,
        metadata=None,
    )
    updated = _write(
        flat_collection,
        method="upsert",
        ids=["a"],
        vectors=np.ones((1, 3), dtype=np.float32),
        documents=["updated"],
        metadata=[{"updated": True}],
    )

    assert [_status_value(row) for row in inserted.rows] == ["inserted"]
    assert [_status_value(row) for row in updated.rows] == ["updated"]
    assert flat_collection.get(["a"])[0].document == "updated"


def test_buffered_durability_is_explicitly_non_durable(flat_collection):
    result = _write(flat_collection, durability="buffered")
    assert result.durable is False
    assert result.visibility_watermark > result.durable_watermark
    assert all(row.searchable for row in result.rows)


def test_idempotency_key_returns_the_same_operation_receipt(flat_collection):
    arguments = {
        "ids": ["a"],
        "vectors": np.zeros((1, 3), dtype=np.float32),
        "documents": ["A"],
        "metadata": [{}],
        "idempotency_key": "request-42",
    }
    first = flat_collection.add(**arguments)
    second = flat_collection.add(**arguments)

    assert first.idempotency_key == second.idempotency_key == "request-42"
    assert first.batch_op_id == second.batch_op_id
    assert first.rows[0].op_id == second.rows[0].op_id
    assert first.rows[0].row_op_id == second.rows[0].row_op_id
    assert _status_value(first.rows[0]) == _status_value(second.rows[0]) == "inserted"


@pytest.mark.parametrize("bad_id", [1, np.int64(1), b"one", uuid.UUID(int=0)])
def test_ids_are_strict_strings_on_all_id_taking_methods(flat_collection, bad_id):
    vector = np.zeros((1, 3), dtype=np.float32)
    with pytest.raises(TypeError):
        flat_collection.add(ids=[bad_id], vectors=vector)
    with pytest.raises(TypeError):
        flat_collection.replace(ids=[bad_id], vectors=vector)
    with pytest.raises(TypeError):
        flat_collection.upsert(ids=[bad_id], vectors=vector)
    with pytest.raises(TypeError):
        flat_collection.delete([bad_id])
    with pytest.raises(TypeError):
        flat_collection.get([bad_id])


@pytest.mark.parametrize("method", ["add", "replace", "upsert"])
def test_empty_write_batches_are_rejected(flat_collection, method):
    with pytest.raises(ValueError):
        getattr(flat_collection, method)(ids=[], vectors=np.empty((0, 3), dtype=np.float32))


@pytest.mark.parametrize(
    "overrides",
    [
        {"ids": ["a"], "vectors": np.zeros((2, 3), dtype=np.float32)},
        {"documents": ["only one"]},
        {"metadata": [{}]},
        {"vectors": np.zeros((2, 2), dtype=np.float32)},
        {"vectors": [[1.0, 2.0], [3.0]]},
    ],
)
def test_write_columns_must_be_rectangular_and_equal_length(flat_collection, overrides):
    with pytest.raises((TypeError, ValueError)):
        _write(flat_collection, **overrides)


@pytest.mark.parametrize("mode", ["all_or_nothing", "per_row_independent", "unsafe", ""])
def test_only_v2_batch_mode_spellings_are_accepted(flat_collection, mode):
    with pytest.raises(ValueError):
        _write(flat_collection, mode=mode)


@pytest.mark.parametrize("durability", ["wal_fsync", "searchable", "async", ""])
def test_only_v2_durability_spellings_are_accepted(flat_collection, durability):
    with pytest.raises(ValueError):
        _write(flat_collection, durability=durability)


def test_delete_preserves_row_statuses_and_input_order(flat_collection):
    _write(flat_collection)
    result = flat_collection.delete(["b", "missing", "a"])

    assert [row.id for row in result.rows] == ["b", "missing", "a"]
    assert [_status_value(row) for row in result.rows] == ["deleted", "not_found", "deleted"]
    assert flat_collection.get(["a", "b"]) == (None, None)


def test_delete_where_rejects_empty_filter_and_documents_batch_atomicity(flat_collection):
    _write(flat_collection)
    with pytest.raises(ValueError):
        flat_collection.delete_where({})

    result = flat_collection.delete_where({"kind": "second"}, batch_size=1)
    assert result.matched == 1
    assert result.deleted == 1
    assert result.not_found == 0
    assert result.batches == 1
    assert flat_collection.get(["a", "b"])[0] is not None
    assert flat_collection.get(["a", "b"])[1] is None
