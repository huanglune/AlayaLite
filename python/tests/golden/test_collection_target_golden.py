# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""Target-half goldens paired with the four §7.4 legacy quirks."""

from pathlib import Path

import numpy as np
from alayalite import Collection
from alayalite.schema import IndexParams

VECTORS = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
ITEMS = [
    ("a", "A", VECTORS[0], {"kind": "x"}),
    ("b", "B", VECTORS[1], {"kind": "y"}),
]


def _collection(tmp_path: Path, name: str) -> Collection:
    return Collection(name, IndexParams(storage_path=str(tmp_path / name / "storage")))


def test_target_search_truncates_without_id_max_or_any_sentinel(tmp_path):
    collection = _collection(tmp_path, "truncate")
    collection.insert(ITEMS)

    response = collection.search(np.asarray([0.0, 0.0], dtype=np.float32), top_k=9)

    assert response["ids"].tolist() == ["a", "b"]
    assert response["distances"].tolist() == [0.0, 1.0]
    assert response["offsets"].tolist() == [0, 2]
    assert response["valid_counts"].tolist() == [2]
    assert 0 <= len(response["ids"]) <= 9
    assert len(response["ids"]) == len(response["distances"])


def test_target_single_and_batch_share_one_short_row_response_schema(tmp_path):
    collection = _collection(tmp_path, "schema")
    collection.insert(ITEMS)
    query = np.asarray([0.0, 0.0], dtype=np.float32)

    single = collection.search(query, top_k=5)
    batch = collection.batch_search(np.stack([query, VECTORS[1]]), top_k=5)

    assert (
        set(single)
        == set(batch)
        == {
            "ids",
            "distances",
            "offsets",
            "valid_counts",
            "statuses",
            "completeness",
            "visibility_watermark",
            "metadata_epoch",
            "search_stats",
        }
    )
    assert single["search_stats"]["rerank_nanoseconds"] == 0
    assert batch["ids"].tolist() == ["a", "b", "b", "a"]
    assert batch["distances"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert batch["offsets"].tolist() == [0, 2, 4]
    assert batch["valid_counts"].tolist() == [2, 2]
    assert batch["ids"].dtype == np.dtype(object)
    assert batch["distances"].dtype == np.float32
    assert batch["offsets"].dtype == np.uint64
    assert batch["valid_counts"].dtype == np.uint64
    assert batch["statuses"].dtype == np.uint8


def test_target_cpp_collection_is_the_only_wal_and_scalar_owner(tmp_path):
    root = tmp_path / "owner"
    collection = Collection("owner", IndexParams(storage_path=str(root / "storage")))
    collection.insert(ITEMS)
    collection.checkpoint()

    assert (root / ".alaya_internal" / "collection_wal_v1" / "logical.wal").is_file()
    assert (root / ".alaya_internal" / "collection_wal_v1" / "CURRENT").is_file()
    assert not (root / "recovery" / "wal.bin").exists()
    assert not (root / "storage").exists()
    assert collection.get_by_id(["a"]) == {
        "id": ["a"],
        "document": ["A"],
        "metadata": [{"kind": "x"}],
    }


def test_target_mutable_add_is_searchable_before_return_and_size_is_live(tmp_path):
    collection = _collection(tmp_path, "visibility")

    receipt = collection.add(ITEMS, mode="all_or_nothing")
    stats = collection.stats()
    response = collection.search(np.asarray([0.0, 0.0], dtype=np.float32), top_k=10)

    assert receipt["searchable"] is True
    assert all(row["searchable"] is True for row in receipt["rows"])
    assert collection.size() == 2
    assert stats["size"] == stats["accepted_count"] == 2
    assert stats["pending_count"] == stats["pending_bytes"] == 0
    assert stats["searchable_bytes"] == stats["accepted_bytes"] == 16
    assert stats["searchable_vector_bytes"] == stats["accepted_vector_bytes"] == 16
    assert response["ids"].tolist() == ["a", "b"]
