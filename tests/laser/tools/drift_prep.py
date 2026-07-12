#!/usr/bin/env python3
"""Prepare SIFT1M cluster-holdout and random-control drift traces.

The fbin/ibin layout matches bench_laser_update_sift.cpp: two little-endian
int32 values (n, dim), followed by a row-major float32/uint32 matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans


DEFAULT_DATA = Path("/home/huangliang/workspace/alaya-dev/data/sift-fbin")
DEFAULT_OUT = Path("/home/huangliang/workspace/alaya-dev/data/laser-update/drift")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def matrix_memmap(path: Path, dtype: np.dtype) -> tuple[np.memmap, int, int]:
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError(f"short header: {path}")
    n, dim = struct.unpack("<ii", raw)
    if n <= 0 or dim <= 0:
        raise ValueError(f"bad header in {path}: n={n}, dim={dim}")
    dt = np.dtype(dtype).newbyteorder("<")
    expected = 8 + n * dim * dt.itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"bad size for {path}: {path.stat().st_size}, expected {expected}")
    return np.memmap(path, dtype=dt, mode="r", offset=8, shape=(n, dim)), n, dim


def write_matrix(path: Path, source: np.ndarray, perm: np.ndarray | None,
                 dtype: np.dtype, rows_per_block: int = 32768) -> None:
    n, dim = source.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<ii", n, dim))
        for start in range(0, n, rows_per_block):
            stop = min(n, start + rows_per_block)
            block = source[start:stop] if perm is None else source[perm[start:stop]]
            np.asarray(block, dtype=np.dtype(dtype).newbyteorder("<"), order="C").tofile(f)


def write_ibin(path: Path, ids: np.ndarray) -> None:
    ids = np.asarray(ids, dtype="<u4", order="C")
    with path.open("wb") as f:
        f.write(struct.pack("<ii", *ids.shape))
        ids.tofile(f)


def md5(path: Path, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(block_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def assign_clusters(base: np.ndarray, model: MiniBatchKMeans,
                    block_rows: int) -> np.ndarray:
    labels = np.empty(base.shape[0], dtype=np.int16)
    for start in range(0, base.shape[0], block_rows):
        stop = min(base.shape[0], start + block_rows)
        labels[start:stop] = model.predict(base[start:stop])
    return labels


def select_heldout(order_desc: np.ndarray, sizes: np.ndarray,
                   lower: int, upper: int) -> tuple[np.ndarray, int, list[int]]:
    total = 0
    chosen: list[int] = []
    skipped: list[int] = []
    for cluster in order_desc.tolist():
        candidate = total + int(sizes[cluster])
        if candidate > upper:
            skipped.append(cluster)
            continue
        chosen.append(cluster)
        total = candidate
        if lower <= total <= upper:
            return np.asarray(chosen, dtype=np.int64), total, skipped
    raise RuntimeError(f"could not reach held-out lower bound {lower}")


def drift_permutation(labels: np.ndarray, heldout: np.ndarray,
                      center_dist: np.ndarray, seed: int) -> np.ndarray:
    held_mask = np.isin(labels, heldout)
    build_ids = np.flatnonzero(~held_mask).astype(np.uint32)
    rng = np.random.default_rng(seed)
    stream_parts: list[np.ndarray] = []
    for cluster in sorted(heldout.tolist(), key=lambda c: (center_dist[c], c)):
        ids = np.flatnonzero(labels == cluster).astype(np.uint32)
        rng.shuffle(ids)
        stream_parts.append(ids)
    return np.concatenate([build_ids, *stream_parts])


def exact_gt(base: np.ndarray, query: np.ndarray, topk: int,
             query_block: int, base_block: int, label: str) -> np.ndarray:
    """Exact squared-L2 top-k using block GEMM, returning row-sorted IDs."""
    nq, nb = query.shape[0], base.shape[0]
    result = np.empty((nq, topk), dtype=np.uint32)
    base_norm = np.empty(nb, dtype=np.float32)
    for bs in range(0, nb, base_block):
        be = min(nb, bs + base_block)
        x = np.asarray(base[bs:be], dtype=np.float32)
        base_norm[bs:be] = np.einsum("ij,ij->i", x, x, dtype=np.float32)
    for qs in range(0, nq, query_block):
        qe = min(nq, qs + query_block)
        q = np.asarray(query[qs:qe], dtype=np.float32)
        qnorm = np.einsum("ij,ij->i", q, q, dtype=np.float32)[:, None]
        best_d = np.full((qe - qs, topk), np.inf, dtype=np.float32)
        best_i = np.zeros((qe - qs, topk), dtype=np.uint32)
        for bs in range(0, nb, base_block):
            be = min(nb, bs + base_block)
            x = np.asarray(base[bs:be], dtype=np.float32)
            dist = qnorm + base_norm[None, bs:be] - np.float32(2.0) * (q @ x.T)
            local_k = min(topk, be - bs)
            cols = np.argpartition(dist, local_k - 1, axis=1)[:, :local_k]
            local_d = np.take_along_axis(dist, cols, axis=1)
            local_i = (cols + bs).astype(np.uint32)
            cand_d = np.concatenate((best_d, local_d), axis=1)
            cand_i = np.concatenate((best_i, local_i), axis=1)
            keep = np.argpartition(cand_d, topk - 1, axis=1)[:, :topk]
            best_d = np.take_along_axis(cand_d, keep, axis=1)
            best_i = np.take_along_axis(cand_i, keep, axis=1)
        # Deterministic distance then ID ordering, useful at tied SIFT boundaries.
        for row in range(qe - qs):
            order = np.lexsort((best_i[row], best_d[row]))
            result[qs + row] = best_i[row, order]
        log(f"{label} exact GT: {qe}/{nq} queries")
    return result


def write_groups(path: Path, gt: np.ndarray, n_build: int,
                 new_threshold: float) -> dict[str, int]:
    frac = np.mean(gt[:, :10] >= n_build, axis=1)
    groups = np.where(frac == 0.0, 0, np.where(frac >= new_threshold, 2, 1))
    counts = {"old": int(np.sum(groups == 0)), "mixed": int(np.sum(groups == 1)),
              "new": int(np.sum(groups == 2))}
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# n_build={n_build} groups: old={counts['old']} mixed={counts['mixed']} new={counts['new']}\n")
        for group, value in zip(groups, frac, strict=True):
            f.write(f"{int(group)} {value:.1f}\n")
    return counts


def overlap_sanity(gt_new: np.ndarray, perm: np.ndarray,
                   original_gt: np.ndarray) -> dict[str, object]:
    mapped = perm[gt_new].astype(np.uint32)
    overlaps = np.empty(mapped.shape[0], dtype=np.float64)
    for i in range(mapped.shape[0]):
        overlaps[i] = len(set(mapped[i].tolist()) & set(original_gt[i, :100].tolist())) / 100.0
    return {
        "mean": float(overlaps.mean()), "min": float(overlaps.min()),
        "p01": float(np.quantile(overlaps, 0.01)),
        "p05": float(np.quantile(overlaps, 0.05)),
        "p50": float(np.quantile(overlaps, 0.50)),
        "p95": float(np.quantile(overlaps, 0.95)),
        "p99": float(np.quantile(overlaps, 0.99)),
        "perfect_queries": int(np.sum(overlaps == 1.0)),
    }


def coordinate_sum_sanity(original: np.ndarray, reordered: np.ndarray,
                          block_rows: int) -> dict[str, object]:
    sums = []
    for matrix in (original, reordered):
        total = np.zeros(matrix.shape[1], dtype=np.float64)
        for start in range(0, matrix.shape[0], block_rows):
            total += np.sum(matrix[start:start + block_rows], axis=0, dtype=np.float64)
        sums.append(total)
    denom = np.maximum(np.abs(sums[0]), np.finfo(np.float64).tiny)
    rel = np.abs(sums[1] - sums[0]) / denom
    return {"max_coordinate_relative_error": float(rel.max()),
            "max_coordinate_absolute_error": float(np.max(np.abs(sums[1] - sums[0]))),
            "passed": bool(rel.max() <= 1e-3)}


def sampled_nn_mean(base: np.ndarray, query_ids: np.ndarray, ref_end: int,
                    query_block: int, base_block: int, exclude_self: bool) -> float:
    values: list[np.ndarray] = []
    ref_norm = np.empty(ref_end, dtype=np.float32)
    for bs in range(0, ref_end, base_block):
        be = min(ref_end, bs + base_block)
        x = np.asarray(base[bs:be], dtype=np.float32)
        ref_norm[bs:be] = np.einsum("ij,ij->i", x, x, dtype=np.float32)
    for qs in range(0, len(query_ids), query_block):
        ids = query_ids[qs:qs + query_block]
        q = np.asarray(base[ids], dtype=np.float32)
        qnorm = np.einsum("ij,ij->i", q, q, dtype=np.float32)[:, None]
        best = np.full(len(ids), np.inf, dtype=np.float32)
        for bs in range(0, ref_end, base_block):
            be = min(ref_end, bs + base_block)
            x = np.asarray(base[bs:be], dtype=np.float32)
            dist = qnorm + ref_norm[None, bs:be] - np.float32(2.0) * (q @ x.T)
            if exclude_self:
                rows = np.flatnonzero((ids >= bs) & (ids < be))
                dist[rows, ids[rows] - bs] = np.inf
            best = np.minimum(best, np.min(dist, axis=1))
        values.append(np.sqrt(np.maximum(best, 0.0), dtype=np.float32))
        log(f"NN sanity: {min(qs + query_block, len(query_ids))}/{len(query_ids)}")
    return float(np.mean(np.concatenate(values), dtype=np.float64))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=DEFAULT_DATA / "sift_base.fbin")
    p.add_argument("--query", type=Path, default=DEFAULT_DATA / "sift_query.fbin")
    p.add_argument("--original-gt", type=Path, default=DEFAULT_DATA / "sift_gt.ibin")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("-o", "--report", type=Path, default=None)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ctrl-seed", type=int, default=43)
    p.add_argument("--fit-sample", type=int, default=200_000)
    p.add_argument("--heldout-min", type=int, default=196_000)
    p.add_argument("--heldout-max", type=int, default=204_000)
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--query-block", type=int, default=256)
    p.add_argument("--base-block", type=int, default=32768)
    p.add_argument("--nn-sample", type=int, default=10_000)
    p.add_argument("--new-threshold", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = a.report or (a.out_dir / "report.md")
    base, n, dim = matrix_memmap(a.base, np.float32)
    query, nq, qdim = matrix_memmap(a.query, np.float32)
    original_gt, ngt, gtwidth = matrix_memmap(a.original_gt, np.uint32)
    if dim != qdim or nq != ngt or gtwidth < a.topk:
        raise ValueError("incompatible base/query/original-GT dimensions")
    log(f"loaded base={n}x{dim}, query={nq}x{qdim}, original GT width={gtwidth}")

    rng = np.random.default_rng(a.seed)
    sample_ids = np.sort(rng.choice(n, size=a.fit_sample, replace=False))
    # Only k and seed are prescribed. Keep sklearn's remaining parameters at
    # their version-recorded defaults instead of silently defining a variant.
    model = MiniBatchKMeans(n_clusters=a.k, random_state=a.seed).fit(
        np.asarray(base[sample_ids])
    )
    labels = assign_clusters(base, model, a.base_block)
    sizes = np.bincount(labels, minlength=a.k)
    global_mean = np.zeros(dim, dtype=np.float64)
    for start in range(0, n, a.base_block):
        global_mean += np.sum(base[start:start + a.base_block], axis=0, dtype=np.float64)
    global_mean /= n
    center_dist = np.linalg.norm(model.cluster_centers_.astype(np.float64) - global_mean, axis=1)
    order_desc = np.lexsort((np.arange(a.k), -center_dist))
    heldout, heldout_n, skipped_clusters = select_heldout(
        order_desc, sizes, a.heldout_min, a.heldout_max
    )
    perm = drift_permutation(labels, heldout, center_dist, a.seed)
    n_build = n - heldout_n
    if len(np.unique(perm)) != n:
        raise AssertionError("drift permutation is not bijective")
    log(f"selected {len(heldout)} held-out clusters, heldout={heldout_n}, n_build={n_build}")

    drift_base_path = a.out_dir / "sift_drift_base.fbin"
    perm_path = a.out_dir / "perm.u32"
    write_matrix(drift_base_path, base, perm, np.float32)
    np.asarray(perm, dtype="<u4").tofile(perm_path)
    drift_base, dn, ddim = matrix_memmap(drift_base_path, np.float32)
    drift_gt = exact_gt(drift_base, query, a.topk, a.query_block, a.base_block, "drift")
    drift_gt_path = a.out_dir / "sift_gt100_drift.ibin"
    write_ibin(drift_gt_path, drift_gt)
    drift_groups_path = a.out_dir / "query_groups.txt"
    drift_groups = write_groups(drift_groups_path, drift_gt, n_build, a.new_threshold)

    ctrl_rng = np.random.default_rng(a.ctrl_seed)
    ctrl_perm = ctrl_rng.permutation(n).astype(np.uint32)
    ctrl_base_path = a.out_dir / "sift_ctrl_base.fbin"
    write_matrix(ctrl_base_path, base, ctrl_perm, np.float32)
    ctrl_base, cn, cdim = matrix_memmap(ctrl_base_path, np.float32)
    ctrl_gt = exact_gt(ctrl_base, query, a.topk, a.query_block, a.base_block, "control")
    ctrl_gt_path = a.out_dir / "sift_gt100_ctrl.ibin"
    write_ibin(ctrl_gt_path, ctrl_gt)
    ctrl_groups_path = a.out_dir / "query_groups_ctrl.txt"
    ctrl_groups = write_groups(ctrl_groups_path, ctrl_gt, n_build, a.new_threshold)

    coord = coordinate_sum_sanity(base, drift_base, a.base_block)
    coord["header_matches"] = bool((dn, ddim) == (n, dim))
    overlap = overlap_sanity(drift_gt, perm, original_gt)
    nondegenerate = {"new_at_least_300": drift_groups["new"] >= 300,
                     "old_at_least_3000": drift_groups["old"] >= 3000,
                     "threshold": a.new_threshold}
    if not all(nondegenerate[k] for k in ("new_at_least_300", "old_at_least_3000")):
        raise RuntimeError(f"query grouping is degenerate at threshold {a.new_threshold}: {drift_groups}")

    nn_rng = np.random.default_rng(a.seed)
    build_sample = np.sort(nn_rng.choice(n_build, size=min(a.nn_sample, n_build), replace=False))
    held_sample = np.sort(nn_rng.choice(np.arange(n_build, n),
                                        size=min(a.nn_sample, heldout_n), replace=False))
    build_nn = sampled_nn_mean(drift_base, build_sample, n_build, a.query_block,
                               a.base_block, exclude_self=True)
    held_nn = sampled_nn_mean(drift_base, held_sample, n_build, a.query_block,
                              a.base_block, exclude_self=False)
    nn_sanity = {"sample_per_side": int(len(build_sample)),
                 "build_internal_mean_l2": build_nn,
                 "heldout_to_build_mean_l2": held_nn,
                 "ratio": held_nn / build_nn, "method": "exact NN for fixed sampled queries"}

    paths = [drift_base_path, drift_gt_path, drift_groups_path, perm_path,
             ctrl_base_path, ctrl_gt_path, ctrl_groups_path]
    artifacts = {p.name: {"bytes": p.stat().st_size, "md5": md5(p)} for p in paths}
    held_profiles = [{"cluster_id": int(c), "size": int(sizes[c]),
                      "center_distance_to_global_mean": float(center_dist[c])}
                     for c in sorted(heldout.tolist(), key=lambda x: (-center_dist[x], x))]
    manifest = {
        "format_version": 1, "k": a.k, "seed": a.seed, "control_seed": a.ctrl_seed,
        "minibatch_kmeans_params": model.get_params(),
        "fit_sample": a.fit_sample, "n": n, "dim": dim, "n_query": nq,
        "n_build": n_build, "n_heldout": heldout_n,
        "heldout_cluster_ids_outermost_first": [int(x) for x in heldout],
        "selection_skipped_clusters_due_to_upper_bound": [int(x) for x in skipped_clusters],
        "heldout_clusters": held_profiles, "cluster_sizes": [int(x) for x in sizes],
        "cluster_center_distances_to_global_mean": [float(x) for x in center_dist],
        "group_thresholds": {"old": "frac == 0", "new": f"frac >= {a.new_threshold}",
                             "mixed": "otherwise"},
        "groups": {"drift": drift_groups, "control": ctrl_groups},
        "sanity": {"coordinate_sum": coord, "gt_overlap": overlap,
                   "group_nondegenerate": nondegenerate, "nearest_neighbor": nn_sanity},
        "control": {"method": "numpy default_rng(seed=43) permutation; first n_build is build"},
        "artifacts": artifacts,
    }
    manifest_path = a.out_dir / "manifest.json"
    # The manifest cannot truthfully contain its own MD5; all other deliverables are included.
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    profile_lines = "\n".join(
        f"- cluster {p['cluster_id']}: {p['size']} vectors, center distance {p['center_distance_to_global_mean']:.6f}"
        for p in held_profiles
    )
    artifact_lines = "\n".join(
        f"- `{name}`: {info['bytes']} bytes, md5 `{info['md5']}`" for name, info in artifacts.items()
    )
    report = f"""# SIFT1M cluster-holdout OOD trace report

## Outputs

{artifact_lines}
- `manifest.json`: metadata and sanity results (self-MD5 intentionally omitted)

## Partition and query groups

- Build: {n_build}; held-out stream: {heldout_n}; total: {n}
- Drift groups: old={drift_groups['old']}, mixed={drift_groups['mixed']}, new={drift_groups['new']}
- Control groups: old={ctrl_groups['old']}, mixed={ctrl_groups['mixed']}, new={ctrl_groups['new']}
- Group thresholds: old iff frac=0, new iff frac>={a.new_threshold}, mixed otherwise.

## Sanity checks

1. Permutation/header: header match={coord['header_matches']}; maximum per-coordinate sum relative error={coord['max_coordinate_relative_error']:.3e} (limit 1e-3); pass={coord['passed']}.
2. GT overlap after mapping to original IDs: mean={overlap['mean']:.6f}, min={overlap['min']:.2f}, p01={overlap['p01']:.2f}, p05={overlap['p05']:.2f}, median={overlap['p50']:.2f}, p95={overlap['p95']:.2f}, p99={overlap['p99']:.2f}, perfect={overlap['perfect_queries']}/{nq}; pass={overlap['mean'] >= 0.99}.
3. Non-degenerate groups: old>=3000 is {nondegenerate['old_at_least_3000']}; new>=300 is {nondegenerate['new_at_least_300']}.
4. Peripheral separation (fixed seed sample, {len(build_sample)} queries per side, exact against the entire build set): held-out-to-build mean L2={held_nn:.6f}; build-internal mean NN L2={build_nn:.6f}; ratio={held_nn / build_nn:.6f}.

## Held-out cluster profile (outermost first)

{profile_lines}

## Decisions and deviations

- The nearest-neighbor separation statistic uses a reproducible {len(build_sample)}-vector sample from each side because an exhaustive all-vector cross-set NN calculation is outside the requested GT workload; every sampled NN is exact against all {n_build} build vectors.
- The random control uses seed 43 to permute all original IDs; its first {n_build} vectors form the build set and the remainder form the insertion stream.
- Strict outer-prefix accumulation could not satisfy [{a.heldout_min}, {a.heldout_max}]
  because the boundary cluster jumped past the upper limit. The selection therefore scans
  outermost-first but skips a whole cluster when adding it would exceed {a.heldout_max};
  no cluster is split. Skipped cluster IDs: {skipped_clusters}.
- No group threshold adjustment was made.
"""
    report_path.write_text(report, encoding="utf-8")
    log(f"wrote manifest {manifest_path} and report {report_path}")
    if not coord["passed"] or overlap["mean"] < 0.99:
        raise RuntimeError("one or more mandatory sanity checks failed; see report")


if __name__ == "__main__":
    main()
