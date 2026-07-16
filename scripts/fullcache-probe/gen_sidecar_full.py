# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Regenerate LASER static-cache sidecars covering 100% of nodes.

The builder writes _cache_ids/_cache_nodes for kCacheRatio (15%) of nodes.
For the full-cache adjudication probe we need every node resident. This tool
hardlinks the index under a new prefix and writes full-coverage sidecars next
to it (non-destructive: the original 15% sidecars stay untouched).

Layout (must match QGBuilder PHASE 3 / QuantizedGraph::load_cache):
  _cache_ids:   [u64 cache_num][u32 pid * cache_num]
  _cache_nodes: [u64 cache_num][u64 node_len][node bytes in cache_ids order]
"""

import os
import shutil
import sys

import numpy as np

SECTOR = 4096


def main():
    src_index = sys.argv[1]  # .../sift1m_R64_MD128.index
    dst_index = sys.argv[2]  # .../sift1m-full_R64_MD128.index
    n = int(sys.argv[3])
    node_len = int(sys.argv[4])
    page_size = int(sys.argv[5])
    npp = int(sys.argv[6])

    expected = SECTOR + ((n + npp - 1) // npp) * page_size
    actual = os.path.getsize(src_index)
    assert actual == expected, f"geometry mismatch: file={actual} expected={expected}"

    os.makedirs(os.path.dirname(dst_index), exist_ok=True)
    if not os.path.exists(dst_index):
        os.link(src_index, dst_index)
    for suffix in ("_rotator",):
        if os.path.exists(src_index + suffix) and not os.path.exists(dst_index + suffix):
            shutil.copy2(src_index + suffix, dst_index + suffix)

    pages = np.memmap(src_index, dtype=np.uint8, mode="r")[SECTOR:]
    pages = pages.reshape(-1, page_size)

    with open(dst_index + "_cache_ids", "wb") as f:
        np.array([n], dtype=np.uint64).tofile(f)
        np.arange(n, dtype=np.uint32).tofile(f)

    with open(dst_index + "_cache_nodes", "wb") as f:
        np.array([n, node_len], dtype=np.uint64).tofile(f)
        chunk = 65536
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            pids = np.arange(start, end)
            rows = pages[pids // npp]
            offs = (pids % npp) * node_len
            # gather node bytes: npp is small; handle per distinct offset
            out = np.empty((end - start, node_len), dtype=np.uint8)
            for off in np.unique(offs):
                mask = offs == off
                out[mask] = rows[mask, off : off + node_len]
            out.tofile(f)
            if start % (chunk * 4) == 0:
                print(f"  {end}/{n}", flush=True)

    ids_sz = os.path.getsize(dst_index + "_cache_ids")
    nodes_sz = os.path.getsize(dst_index + "_cache_nodes")
    assert ids_sz == 8 + 4 * n, ids_sz
    assert nodes_sz == 16 + node_len * n, nodes_sz
    print(f"ok: {dst_index} sidecars n={n} node_len={node_len}", flush=True)


if __name__ == "__main__":
    main()
