# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Exact L2 top-100 ground truth for the full-cache adjudication probe.

Computes GT for (sift1m_pca_base.fbin, drift/ctrl-data/sift_query.fbin) so the
probe does not depend on the drift campaign's population-specific GT files.
"""

import sys

import numpy as np


def read_fbin(path):
    with open(path, "rb") as f:
        n, dim = np.fromfile(f, dtype=np.int32, count=2)
        data = np.fromfile(f, dtype=np.float32, count=n * dim)
    return data.reshape(n, dim)


def main():
    base_path, query_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    topk = 100
    base = read_fbin(base_path)
    query = read_fbin(query_path)
    print(f"base {base.shape} query {query.shape}", flush=True)

    bnorm = (base.astype(np.float32) ** 2).sum(axis=1)
    nq = query.shape[0]
    gt = np.empty((nq, topk), dtype=np.int32)
    chunk = 500
    for i in range(0, nq, chunk):
        q = query[i : i + chunk]
        # ||q-b||^2 = ||q||^2 - 2 q.b + ||b||^2 ; ||q||^2 is rank-constant, drop it
        d = bnorm[None, :] - 2.0 * (q @ base.T)
        part = np.argpartition(d, topk, axis=1)[:, :topk]
        row = np.take_along_axis(d, part, axis=1)
        order = np.argsort(row, axis=1, kind="stable")
        gt[i : i + chunk] = np.take_along_axis(part, order, axis=1)
        print(f"  {i + len(q)}/{nq}", flush=True)

    with open(out_path, "wb") as f:
        np.array([nq, topk], dtype=np.int32).tofile(f)
        gt.tofile(f)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
