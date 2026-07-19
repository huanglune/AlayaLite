# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Live internal-core counterparts for the 22 read-path goldens."""

from __future__ import annotations

import gc

import numpy as np
import pytest


def _seed(collection):
    collection.add(
        ids=["a", "b", "c", "d"],
        vectors=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        documents=["A", "B", "C", "D"],
        metadata=[
            {"kind": "keep", "score": 1, "enabled": True},
            {"kind": "drop", "score": 2, "enabled": False},
            {"kind": "keep", "score": 3, "enabled": True},
            {"kind": "drop", "score": 4, "enabled": False},
        ],
    )


def test_one_and_many_queries_have_the_same_csr_type(flat_collection, sdk):
    _seed(flat_collection)
    one = flat_collection.search(np.asarray([0.0, 0.0, 0.0], dtype=np.float32), limit=3)
    many = flat_collection.search(
        np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32),
        limit=3,
    )

    assert isinstance(one, sdk.SearchResult)
    assert isinstance(many, sdk.SearchResult)
    assert len(one) == 1
    assert len(many) == 2
    assert one.offsets.tolist() == [0, 3]
    assert many.offsets.tolist() == [0, 3, 6]
    assert one.valid_counts.tolist() == [3]
    assert many.valid_counts.tolist() == [3, 3]


def test_search_result_is_read_only_csr_without_sentinels(flat_collection):
    _seed(flat_collection)
    result = flat_collection.search(
        np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32),
        limit=10,
    )

    assert result.offsets[0] == 0
    assert result.offsets[-1] == len(result.ids) == len(result.distances)
    np.testing.assert_array_equal(np.diff(result.offsets), result.valid_counts)
    assert result.valid_counts.tolist() == [4, 4]
    assert result.status_codes.shape == (2,)
    assert result.completeness_codes.shape == (2,)
    assert all(isinstance(item, str) and item for item in result.ids.tolist())
    for array in (
        result.ids,
        result.distances,
        result.offsets,
        result.status_codes,
        result.completeness_codes,
    ):
        assert array.flags.writeable is False


def test_search_row_is_a_shared_memory_read_only_view(flat_collection):
    _seed(flat_collection)
    result = flat_collection.search(
        np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32),
        limit=2,
    )
    row = result[1]

    assert np.shares_memory(row.ids, result.ids)
    assert np.shares_memory(row.distances, result.distances)
    assert row.ids.flags.writeable is False
    assert row.distances.flags.writeable is False
    assert row.status == result.statuses[1]
    assert row.completeness == result.completeness[1]
    assert row.ids.tolist() == ["d", "c"]


def test_search_result_buffers_outlive_collection(flat_collection):
    _seed(flat_collection)
    result = flat_collection.search(np.asarray([0.0, 0.0, 0.0], dtype=np.float32), limit=2)
    row = result[0]
    flat_collection.close()
    gc.collect()

    assert row.ids.tolist() == ["a", "b"]
    np.testing.assert_allclose(row.distances, [0.0, 1.0])


def test_search_where_uses_the_same_native_filter_contract(flat_collection):
    _seed(flat_collection)
    result = flat_collection.search(
        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        limit=10,
        where={"kind": "keep"},
        filter_policy="strict",
    )
    assert result[0].ids.tolist() == ["a", "c"]


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        ({"kind": {"$eq": "keep"}}, ["a", "c"]),
        ({"score": {"$gt": 2}}, ["c", "d"]),
        ({"score": {"$ge": 2, "$lt": 4}}, ["b", "c"]),
        ({"score": {"$le": 2}}, ["a", "b"]),
        ({"kind": {"$in": ["keep"]}}, ["a", "c"]),
        ({"$and": [{"kind": "keep"}, {"score": {"$gt": 1}}]}, ["c"]),
        ({"$or": [{"score": 1}, {"score": 4}]}, ["a", "d"]),
        ({"missing": "value"}, []),
    ],
)
def test_scan_supports_the_fixed_filter_dsl(flat_collection, where, expected):
    _seed(flat_collection)
    assert [record.id for record in flat_collection.scan(where=where, limit=10)] == expected


@pytest.mark.parametrize("where", [{"x": {"$ne": 1}}, {"x": {"$nin": [1]}}, {"$not": [{"x": 1}]}])
def test_filter_dsl_rejects_unimplemented_operators(flat_collection, where):
    with pytest.raises(ValueError):
        flat_collection.scan(where=where, limit=10)


def test_scan_is_limited_stable_and_projects_vectors_only_on_request(flat_collection):
    _seed(flat_collection)
    first = flat_collection.scan(limit=2)
    second = flat_collection.scan(where={}, limit=2)
    projected = flat_collection.scan(limit=1, include_vector=True)

    assert [record.id for record in first] == ["a", "b"]
    assert [record.id for record in second] == ["a", "b"]
    assert all(record.vector is None for record in first)
    assert projected[0].vector is not None
    assert projected[0].vector.flags.writeable is False


@pytest.mark.parametrize("limit", [0, -1])
def test_scan_requires_a_positive_finite_limit(flat_collection, limit):
    with pytest.raises(ValueError):
        flat_collection.scan(limit=limit)


def test_get_is_position_aligned_and_preserves_missing_rows(flat_collection, sdk):
    _seed(flat_collection)
    records = flat_collection.get(["c", "missing", "a", "c"])

    assert len(records) == 4
    assert isinstance(records[0], sdk.Record)
    assert records[0].id == "c"
    assert records[1] is None
    assert records[2].id == "a"
    assert records[3].id == "c"
    assert all(record is None or record.vector is None for record in records)


def test_get_vector_projection_is_owned_and_read_only(flat_collection):
    _seed(flat_collection)
    record = flat_collection.get(["b"], include_vector=True)[0]

    np.testing.assert_array_equal(record.vector, [1.0, 0.0, 0.0])
    assert record.vector.flags.writeable is False
    assert record.document == "B"
    assert record.metadata == {"kind": "drop", "score": 2, "enabled": False}
    assert record.version > 0


def test_search_validates_query_rank_dimension_and_limit(flat_collection):
    _seed(flat_collection)
    for query in (
        np.zeros((1, 1, 3), dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            flat_collection.search(query)
    with pytest.raises(ValueError):
        flat_collection.search(np.zeros(3, dtype=np.float32), limit=0)
