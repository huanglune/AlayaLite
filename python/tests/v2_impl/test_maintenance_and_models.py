# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Live coverage for maintenance receipts and less-common public models."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from alayalite._database import connect
from alayalite.config import FlatIndexConfig
from alayalite.models import (
    CheckpointReceipt,
    CollectionStats,
    CompactionReceipt,
    GarbageCollectionReceipt,
    SealReceipt,
    SearchBudget,
    SearchStats,
)


def test_typed_maintenance_receipts_stats_count_and_len(flat_collection):
    flat_collection.add(
        ids=["a", "b"],
        vectors=np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
    )
    stats = flat_collection.stats()
    checkpoint = flat_collection.checkpoint()
    first_seal = flat_collection.seal()
    flat_collection.add(
        ids=["c", "d"],
        vectors=np.asarray([[2, 0, 0], [3, 0, 0]], dtype=np.float32),
    )
    second_seal = flat_collection.seal()
    compacted = flat_collection.compact()
    collected = flat_collection.collect_garbage()

    assert isinstance(stats, CollectionStats)
    assert stats.size == 2
    assert flat_collection.count() == len(flat_collection) == 4
    assert isinstance(checkpoint, CheckpointReceipt)
    assert isinstance(first_seal, SealReceipt)
    assert isinstance(second_seal, SealReceipt)
    assert isinstance(compacted, CompactionReceipt)
    assert compacted.compacted_rows == 4
    assert isinstance(collected, GarbageCollectionReceipt)
    assert collected.reclaimed >= 2


def test_rebuild_index_atomically_preserves_live_rows_and_reopens(flat_collection):
    flat_collection.add(
        ids=["a", "b"],
        vectors=np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        documents=["A", "B"],
        metadata=[{"rank": 1}, {"rank": 2}],
    )

    receipt = flat_collection.rebuild_index(index=FlatIndexConfig())

    assert isinstance(receipt, CheckpointReceipt)
    assert flat_collection.count() == 2
    assert [record.document for record in flat_collection.scan(limit=10)] == ["A", "B"]
    assert flat_collection.get(["a", "b"], include_vector=True)[1].metadata == {"rank": 2}


def test_search_budget_and_search_stats_are_frozen_slotted_models(flat_collection):
    budget = SearchBudget(scratch_bytes=1 << 20, io_requests=100, io_bytes=1 << 20)
    result = flat_collection.search(np.zeros(3, dtype=np.float32), budget=budget)

    assert isinstance(result.stats, SearchStats)
    assert result.stats.effective_effort is None
    assert not hasattr(budget, "__dict__")
    with pytest.raises(FrozenInstanceError):
        budget.io_requests = 1
    with pytest.raises(ValueError):
        SearchBudget(io_bytes=-1)


def test_database_close_closes_active_collection_handle(sdk, flat_config, tmp_path):
    database = connect(tmp_path / "database")
    collection = database.create_collection("docs", config=flat_config)

    database.close()

    with pytest.raises(sdk.CollectionClosedError):
        collection.count()
