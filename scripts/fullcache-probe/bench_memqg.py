# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Memory QG (RaBitQ) reference arm for the full-cache adjudication probe.

Builds the in-memory RaBitQ quantized graph on SIFT1M and measures the
search QPS/recall curve under the same query set + exact GT as the LASER
full-cache arm. Serial (num_threads=1) plus a 16-thread arm.
"""

import sys
import time

import numpy as np

from alayalite import Index
from alayalite.schema import IndexParams


def read_fbin(path):
    with open(path, "rb") as f:
        n, dim = np.fromfile(f, dtype=np.int32, count=2)
        data = np.fromfile(f, dtype=np.float32, count=n * dim)
    return data.reshape(n, dim)


def read_ibin(path):
    with open(path, "rb") as f:
        n, dim = np.fromfile(f, dtype=np.int32, count=2)
        data = np.fromfile(f, dtype=np.int32, count=n * dim)
    return data.reshape(n, dim)


def main():
    base_path, query_path, gt_path = sys.argv[1], sys.argv[2], sys.argv[3]
    topk = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    build_threads = int(sys.argv[5]) if len(sys.argv) > 5 else 64

    base = read_fbin(base_path)
    query = read_fbin(query_path)
    gt = read_ibin(gt_path)[:, :topk]
    print(f"base {base.shape} query {query.shape} gt {gt.shape}", flush=True)

    params = IndexParams(
        index_type="hnsw",
        quantization_type="rabitq",
        metric="l2",
        capacity=np.uint32(base.shape[0]),
    )
    idx = Index(name="fullcache-probe-memqg", params=params)
    t0 = time.perf_counter()
    idx.fit(base, num_threads=build_threads)
    t1 = time.perf_counter()
    print(f"build_seconds,{t1 - t0:.2f}", flush=True)

    print("arm,ef,threads,recall,qps,mean_us", flush=True)
    for threads in (1, 16):
        for ef in (40, 60, 100, 200):
            eff_ef = max(ef, topk)
            # warmup
            idx.batch_search(query[:1000], topk, eff_ef, threads)
            best_qps = 0.0
            recall = 0.0
            for _ in range(3):
                t0 = time.perf_counter()
                res = idx.batch_search(query, topk, eff_ef, threads)
                t1 = time.perf_counter()
                qps = query.shape[0] / (t1 - t0)
                if qps > best_qps:
                    best_qps = qps
                res = np.asarray(res)[:, :topk]
                hits = 0
                for i in range(gt.shape[0]):
                    hits += len(np.intersect1d(res[i], gt[i]))
                recall = hits / gt.size
            mean_us = 1e6 * threads / best_qps
            print(
                f"memqg,{ef},{threads},{recall:.4f},{best_qps:.1f},{mean_us:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
