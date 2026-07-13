# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""Canonical response, mutation mode, status, and C++/Python parity tests."""

import numpy as np
import pytest
from alayalite import Collection, CollectionInvalidArgumentError
from alayalite._alayalitepy import _Collection
from alayalite.schema import IndexParams


def _item(item_id, vector, document="", metadata=None):
    return (item_id, document, np.asarray(vector, dtype=np.float32), metadata or {})


def test_python_binding_reopens_cxx_owner_and_matches_direct_parity_sequence(tmp_path):
    root = tmp_path / "parity"
    collection = Collection("parity", IndexParams(rocksdb_path=str(root / "rocksdb")))
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
    collection = Collection("batch", IndexParams(rocksdb_path=str(tmp_path / "batch" / "rocksdb")))
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
            index_type="hnsw",
            quantization_type="rabitq",
            rocksdb_path=str(tmp_path / "bad-qg" / "rocksdb"),
        ),
    )

    with pytest.raises(CollectionInvalidArgumentError) as captured:
        collection.add([_item("row", [0, 0])])

    error = captured.value
    assert isinstance(error, ValueError)
    assert error.status_version == "1"
    assert error.status_code != 0
    assert error.partial is False
