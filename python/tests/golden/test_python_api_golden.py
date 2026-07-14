# SPDX-License-Identifier: AGPL-3.0-only
"""Canonical Python Collection compatibility projection snapshot."""

import numpy as np
import pytest
from alayalite import Collection
from alayalite.schema import IndexParams

VECTORS = np.array([[0, 0], [1, 0], [0, 2]], dtype=np.float32)


def test_collection_structures_duplicates_and_empty_operations(tmp_path):
    params = IndexParams(storage_path=str(tmp_path / "rocks"))
    collection = Collection("golden", params)
    assert collection.insert([]) is None
    assert collection.get_by_id(["missing"]) == {"id": [], "document": [], "metadata": []}
    with pytest.raises(RuntimeError, match=r"Call insert\(\) with the first batch"):
        collection.options()

    items = [
        ("a", "A", VECTORS[0], {"kind": "x"}),
        ("b", "B", VECTORS[1], {"kind": "y"}),
    ]
    collection.insert(items)
    got = collection.batch_query([[0, 0]], limit=2, ef_search=2)
    assert list(got) == ["id", "document", "metadata", "distance"]
    assert got["id"] == [["a", "b"]]
    assert got["document"] == [["A", "B"]]
    assert got["metadata"] == [[{"kind": "x"}, {"kind": "y"}]]
    assert got["distance"] == [[0.0, 1.0]]

    with pytest.raises(RuntimeError, match="Duplicate item_id: a"):
        collection.insert([("a", "replacement", VECTORS[2], {})])
    with pytest.raises(ValueError, match="ef_search must be greater than or equal to limit"):
        collection.batch_query([[0, 0]], limit=2, ef_search=1)
    with pytest.raises(ValueError, match="Unsupported operator"):
        collection.build_filter({"x": {"$ne": 1}})
