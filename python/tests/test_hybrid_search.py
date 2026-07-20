# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""Extended recall coverage for vector search with a metadata filter."""

from __future__ import annotations

import time

import numpy as np
import pytest
from alayalite import CollectionConfig, FlatIndexConfig, QGIndexConfig, capabilities, connect

ROWS = 20_000
TARGET_ROWS = 2_000
DIMENSION = 64
LIMIT = 100


def _ground_truth(vectors: np.ndarray, targets: set[int], query: np.ndarray) -> set[int]:
    target_ids = np.asarray(sorted(targets))
    scores = vectors[target_ids] @ query
    return set(target_ids[np.argsort(-scores)[:LIMIT]].tolist())


@pytest.mark.extended
@pytest.mark.parametrize("index_kind", ["flat", "qg"])
def test_filtered_search_recall(index_kind: str, tmp_path) -> None:
    if index_kind == "qg" and "qg" not in capabilities().index_types:
        pytest.skip("QG is unavailable in this wheel")

    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(ROWS, DIMENSION)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    targets = set(rng.choice(ROWS, TARGET_ROWS, replace=False).tolist())
    query = rng.normal(size=DIMENSION).astype(np.float32)
    query /= np.linalg.norm(query)
    truth = _ground_truth(vectors, targets, query)
    index = FlatIndexConfig() if index_kind == "flat" else QGIndexConfig()

    with connect(tmp_path / index_kind) as database:
        collection = database.create_collection(
            "filtered",
            config=CollectionConfig(
                dimension=DIMENSION,
                metric="cosine",
                index=index,
            ),
        )
        collection.add(
            ids=[str(row) for row in range(ROWS)],
            vectors=vectors,
            documents=[f"Document {row}" for row in range(ROWS)],
            metadata=[{"label": "target" if row in targets else "other"} for row in range(ROWS)],
        )
        if index_kind == "qg":
            collection.seal()

        started = time.perf_counter()
        result = collection.search(
            query,
            limit=LIMIT,
            where={"label": "target"},
            effort=300 if index_kind == "qg" else None,
        )
        elapsed = time.perf_counter() - started
        found = {int(item_id) for item_id in result[0].ids.tolist()}
        assert found <= targets
        recall = len(found & truth) / len(truth)
        minimum = 1.0 if index_kind == "flat" else 0.90
        assert recall >= minimum
        print(f"{index_kind}: recall={recall:.4f}, qps={1.0 / elapsed:.2f}")
        collection.close()
