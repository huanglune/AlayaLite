# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""Canonical response, mutation mode, status, and C++/Python parity tests."""

import numpy as np
import pytest
from alayalite import (
    Collection,
    CollectionInvalidArgumentError,
    CollectionResourceExhaustedError,
)
from alayalite._alayalitepy import _Collection
from alayalite.schema import IndexParams


def _item(item_id, vector, document="", metadata=None):
    return (item_id, document, np.asarray(vector, dtype=np.float32), metadata or {})


def test_python_binding_reopens_cxx_owner_and_matches_direct_parity_sequence(tmp_path):
    root = tmp_path / "parity"
    collection = Collection("parity", IndexParams(storage_path=str(root / "storage")))
    collection.add([_item("a", [0, 0], "A", {"revision": 1}), _item("b", [2, 0], "B")])
    collection.close()

    native = _Collection.open(str(root))
    native.mutate(
        ["a"],
        ["A2"],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        [{"revision": 2}],
        "upsert",
    )
    native.remove(["b"])
    native.mutate(
        ["c"],
        ["C"],
        np.asarray([[0.0, 1.0]], dtype=np.float32),
        [{}],
        "add",
    )
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    direct = native.batch_search(queries, 9)
    native.checkpoint()
    native.close()

    python = Collection.load(tmp_path, "parity")
    projected = python.batch_search(queries, top_k=9)
    assert projected["ids"].tolist() == direct["ids"].tolist() == ["a", "c", "c", "a"]
    np.testing.assert_array_equal(projected["offsets"], direct["offsets"])
    np.testing.assert_array_equal(projected["valid_counts"], direct["valid_counts"])
    np.testing.assert_array_equal(projected["statuses"], direct["statuses"])
    np.testing.assert_array_equal(projected["completeness"], direct["completeness"])
    np.testing.assert_allclose(projected["distances"], [0.0, 2.0, 0.0, 2.0], rtol=0, atol=0)
    assert python.get_by_id(["a", "b", "c"]) == {
        "id": ["a", "c"],
        "document": ["A2", "C"],
        "metadata": [{"revision": 2}, {}],
    }


def test_canonical_batch_mutation_supports_independent_and_all_or_nothing(tmp_path):
    collection = Collection("batch", IndexParams(storage_path=str(tmp_path / "batch" / "storage")))
    collection.add([_item("existing", [0, 0])])

    independent = collection.add(
        [_item("existing", [1, 0]), _item("new", [1, 0])],
        mode="per_row_independent",
    )
    assert [row["row_status"] for row in independent["rows"]] == [4, 0]
    assert [row["searchable"] for row in independent["rows"]] == [False, True]
    assert collection.size() == 2

    atomic = collection.upsert(
        [_item("same", [0, 1]), _item("same", [1, 1])],
        mode="all_or_nothing",
    )
    assert [row["row_status"] for row in atomic["rows"]] == [8, 6]
    assert atomic["searchable"] is False
    assert collection.size() == 2
    assert not collection.get_by_id(["same"])["id"]


def test_native_status_maps_to_versioned_python_exception(tmp_path):
    collection = Collection(
        "bad-qg",
        IndexParams(
            index_type="qg",
            quantization_type="none",
            storage_path=str(tmp_path / "bad-qg" / "storage"),
        ),
    )

    with pytest.raises(CollectionInvalidArgumentError) as captured:
        collection.add([_item("row", [0, 0])])

    error = captured.value
    assert isinstance(error, ValueError)
    assert error.status_version == "1"
    assert error.status_code != 0
    assert error.partial is False


def test_gate10_filter_policies_overfetch_stats_and_budget_reuse(tmp_path):
    collection = Collection(
        "gate10-filter",
        IndexParams(storage_path=str(tmp_path / "gate10-filter" / "storage")),
    )
    collection.add(
        [
            _item(
                f"row-{row}",
                [row, 0],
                metadata={"selected": row >= 4, "all": True},
            )
            for row in range(8)
        ]
    )
    query = np.asarray([0.0, 0.0], dtype=np.float32)

    for expression, expected in (
        ({"missing": True}, []),
        ({"selected": True}, ["row-4", "row-5"]),
        ({"all": True}, ["row-0", "row-1"]),
    ):
        strict = collection.search(
            query,
            top_k=2,
            metadata_filter=expression,
            filter_policy="strict",
        )
        assert strict["ids"].tolist() == expected
        assert strict["search_stats"]["filter_execution"] == "prefilter"

    traversed = collection.search(
        query,
        top_k=2,
        metadata_filter={"selected": True},
        filter_policy="auto",
        filter_selectivity=0.5,
    )
    assert traversed["ids"].tolist() == ["row-4", "row-5"]
    assert traversed["search_stats"]["filter_execution"] == "traversal"
    assert traversed["search_stats"]["overfetch_rounds"] == 2
    assert traversed["search_stats"]["filter_examined"] > 0
    assert traversed["search_stats"]["filter_passed"] > 0
    assert traversed["search_stats"]["lease_acquired"] == 1
    assert traversed["search_stats"]["lease_released"] == 1

    postfiltered = collection.search(
        query,
        top_k=2,
        metadata_filter={"all": True},
        filter_policy="auto",
        filter_selectivity=1.0,
    )
    assert postfiltered["search_stats"]["filter_execution"] == "postfilter"

    with pytest.raises(CollectionResourceExhaustedError) as denied:
        collection.search(query, top_k=2, scratch_budget_bytes=1)
    assert denied.value.partial is False

    reused = collection.search(query, top_k=2)
    assert reused["ids"].tolist() == ["row-0", "row-1"]
    assert reused["search_stats"]["lease_acquired"] == 1
    assert reused["search_stats"]["lease_released"] == 1


def test_gate10_python_seal_compact_gc_and_collection_stats(tmp_path):
    collection = Collection(
        "gate10-lifecycle",
        IndexParams(index_type="flat", storage_path=str(tmp_path / "gate10-lifecycle" / "storage")),
    )
    collection.add([_item("a", [0, 0]), _item("b", [1, 0])])
    first = collection.seal()
    assert first["sealed_rows"] == 2
    assert first["source_segment_id"] == 2
    assert first["successor_segment_id"] == 3
    collection.gc()

    collection.add([_item("c", [2, 0]), _item("d", [3, 0])])
    second = collection.seal()
    assert second["sealed_rows"] == 2
    collection.gc()

    before = collection.search(np.asarray([0.0, 0.0], dtype=np.float32), top_k=10)
    compacted = collection.compact()
    assert compacted["source_segment_ids"] == [4, 6]
    assert compacted["compacted_rows"] == 4
    after = collection.search(np.asarray([0.0, 0.0], dtype=np.float32), top_k=10)
    assert after["ids"].tolist() == before["ids"].tolist()
    np.testing.assert_array_equal(after["distances"], before["distances"])

    reclaimed = collection.gc()
    assert reclaimed["reclaimed"] == 2
    stats = collection.stats()
    assert stats["sealed_segments_count"] == 1
    assert stats["gc_pending_count"] == 0
    assert stats["active_segment_algorithm"] == "flat"
    assert stats["compacted_bytes"] == compacted["input_bytes"]

    automatic = Collection(
        "gate10-auto-seal",
        IndexParams(storage_path=str(tmp_path / "gate10-auto-seal" / "storage")),
        auto_seal_rows=2,
    )
    automatic.add([_item("x", [0, 0]), _item("y", [1, 0])])
    assert automatic.stats()["sealed_segments_count"] == 1
