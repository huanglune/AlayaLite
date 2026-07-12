# SPDX-License-Identifier: AGPL-3.0-only
"""Compatibility snapshots for the three public Python entry points."""

import numpy as np
import pytest

from alayalite import Collection, DiskCollection, Index, MetricType
from alayalite.schema import IndexParams


VECTORS = np.array([[0, 0], [1, 0], [0, 2]], dtype=np.float32)


def test_index_return_shapes_and_boundaries(tmp_path):
    index = Index("golden", IndexParams(capacity=8, rocksdb_path=str(tmp_path / "rocks")))
    with pytest.raises(ValueError, match="Index is not init yet"):
        index.search(np.zeros(2, dtype=np.float32), 1)
    index.fit(VECTORS, num_threads=1)

    result = index.search(np.array([0, 0], dtype=np.float32), 10)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint32
    assert result.shape == (10,)
    # Unlike DiskCollection, memory Index does not cap top-k. It pads with the
    # ID type's max value, which is observable compatibility behavior.
    assert result.tolist() == [0, 1, 2] + [np.iinfo(np.uint32).max] * 7
    with pytest.raises(ValueError, match="queries must not be empty"):
        index.batch_search(VECTORS[:0], 2)
    assert index.insert(np.array([3, 0], dtype=np.float32)) == 3

    with pytest.raises(ValueError, match="query must be a 1D array"):
        index.search(np.zeros((1, 2), dtype=np.float32), 1)
    with pytest.raises(ValueError, match="Vector dimension must match the index dimension"):
        index.search(np.zeros(3, dtype=np.float32), 1)
    with pytest.raises(RuntimeError, match="An index can be only fitted once"):
        index.fit(VECTORS)


def test_collection_structures_duplicates_and_empty_operations(tmp_path):
    params = IndexParams(rocksdb_path=str(tmp_path / "rocks"))
    collection = Collection("golden", params)
    assert collection.insert([]) is None
    assert collection.get_by_id(["missing"]) == {"id": [], "document": [], "metadata": []}
    with pytest.raises(RuntimeError, match=r"Call insert\(\) with the first batch"):
        collection.get_cpp_index()

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


def test_disk_collection_public_shape_visibility_and_errors(tmp_path):
    path = tmp_path / "disk"
    col = DiskCollection(path=str(path), dim=2, metric=MetricType.L2, index_type="disk_flat")
    query = np.zeros(2, dtype=np.float32)
    assert col.search(query, k=5) == []
    col.add(VECTORS, np.array([10, 11, 12], dtype=np.uint64))
    assert col.size() == 0
    assert col.search(query, k=5) == []
    col.flush()
    assert col.size() == 3
    assert col.search(query, k=99) == [(10, 0.0), (11, 1.0), (12, 4.0)]
    labels = col.batch_search(VECTORS[:2], k=2)
    assert isinstance(labels, np.ndarray) and labels.dtype == np.uint64
    assert labels.tolist() == [[10, 11], [11, 10]]
    labels_with_distance, distances = col.batch_search_with_distance(VECTORS[:2], k=2)
    assert labels_with_distance.tolist() == labels.tolist()
    assert isinstance(distances, np.ndarray) and distances.dtype == np.float32
    assert distances.tolist() == [[0.0, 1.0], [0.0, 1.0]]

    with pytest.raises(ValueError, match="DiskCollection.search: k must be > 0"):
        col.search(query, k=0)
    with pytest.raises(ValueError, match="ef must be > 0"):
        col.search(query, k=1, ef=0)
    with pytest.raises(ValueError, match="duplicate"):
        col.add(VECTORS[:2], np.array([20, 20], dtype=np.uint64))
        col.flush()
    with pytest.raises(RuntimeError):
        DiskCollection(path=str(path), dim=2, metric=MetricType.L2, index_type="disk_flat")
