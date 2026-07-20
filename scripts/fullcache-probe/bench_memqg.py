# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""QG reference arm for the historical full-cache adjudication probe."""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from alayalite import CollectionConfig, QGIndexConfig, connect
from alayalite.models import SearchResult


def read_fbin(path: str) -> np.ndarray:
    """Read a float32 fbin matrix."""
    with open(path, "rb") as stream:
        rows, dimension = np.fromfile(stream, dtype=np.int32, count=2)
        data = np.fromfile(stream, dtype=np.float32, count=int(rows * dimension))
    return data.reshape(rows, dimension)


def read_ibin(path: str) -> np.ndarray:
    """Read an int32 ibin matrix."""
    with open(path, "rb") as stream:
        rows, dimension = np.fromfile(stream, dtype=np.int32, count=2)
        data = np.fromfile(stream, dtype=np.int32, count=int(rows * dimension))
    return data.reshape(rows, dimension)


def _search(collection, queries: np.ndarray, limit: int, effort: int, workers: int) -> list[list[str]]:
    def rows(result: SearchResult) -> list[list[str]]:
        return [result[index].ids.tolist() for index in range(len(result))]

    if workers == 1:
        return rows(collection.search(queries, limit=limit, effort=effort))
    chunks = [chunk for chunk in np.array_split(queries, workers) if len(chunk)]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(lambda chunk: collection.search(chunk, limit=limit, effort=effort), chunks)
        return [row for result in results for row in rows(result)]


def main() -> None:
    """Build QG and report recall/QPS for public effort settings."""
    base_path, query_path, truth_path = sys.argv[1], sys.argv[2], sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    build_threads = int(sys.argv[5]) if len(sys.argv) > 5 else 64
    base = read_fbin(base_path)
    queries = read_fbin(query_path)
    truth = read_ibin(truth_path)[:, :limit]
    print(f"base {base.shape} query {queries.shape} gt {truth.shape}", flush=True)

    config = CollectionConfig(
        dimension=int(base.shape[1]),
        metric="l2",
        index=QGIndexConfig(build_threads=build_threads),
    )
    with connect() as database:
        collection = database.create_collection("fullcache-probe-qg", config=config)
        started = time.perf_counter()
        collection.add(ids=[str(row) for row in range(len(base))], vectors=base)
        collection.seal()
        print(f"build_seconds,{time.perf_counter() - started:.2f}", flush=True)

        print("arm,effort,workers,recall,qps,mean_us", flush=True)
        for workers in (1, 16):
            for requested_effort in (100, 200, 400, 800):
                effort = max(requested_effort, limit, 100)
                _search(collection, queries[:1000], limit, effort, workers)
                best_qps = 0.0
                recall = 0.0
                for _ in range(3):
                    started = time.perf_counter()
                    result_rows = _search(collection, queries, limit, effort, workers)
                    elapsed = time.perf_counter() - started
                    best_qps = max(best_qps, len(queries) / elapsed)
                    hits = sum(
                        len(np.intersect1d(np.asarray(row, dtype=np.int64), truth[index]))
                        for index, row in enumerate(result_rows)
                    )
                    recall = hits / truth.size
                mean_us = 1e6 * workers / best_qps
                print(
                    f"qg,{effort},{workers},{recall:.4f},{best_qps:.1f},{mean_us:.1f}",
                    flush=True,
                )
        collection.close()


if __name__ == "__main__":
    main()
