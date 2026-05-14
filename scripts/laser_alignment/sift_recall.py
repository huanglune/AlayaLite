# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Build LASER on SIFT-1M (npp>1 configuration) and report recall@10.

This is the SIFT-1M acceptance harness for the `fix-laser-low-dim-page-layout`
change. Upstream LASER did not publish a SIFT-1M result, so the bar is an
absolute `recall@10 >= 0.95` smoke check plus full reproducibility metadata
(seed, CPU model, compiler flags, dataset shas) emitted into
`results/laser_sift1m_recall.{md,json}`. See `docs/LASER.md` § "SIFT-1M
Acceptance Configuration" for the rationale.

The harness exits 0 on PASS (recall@10 >= threshold) and 2 on soft-fail
(harness completed but recall is below the bar). Hard failures (build error,
missing dataset, etc.) exit 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import Final

import numpy as np

DEFAULT_DATASET_DIR: Final[Path] = Path("/md1/huangliang/alaya-dev/data/sift1m")
DEFAULT_OUTPUT_ROOT: Final[Path] = Path("/md1/huangliang/alaya-dev/build_graph/laser_sift1m_recall")
DEFAULT_RESULTS_DIR: Final[Path] = Path(__file__).resolve().parent.parent.parent / "results"

# Pinned dataset shas (see docs/LASER.md § SIFT-1M Acceptance Configuration).
DATASET_SHAS: Final[dict[str, str]] = {
    "sift_base.fbin": "8c7b3d999ba3133f865af72df078f77c2d248fdb80571d7ea1f1bb8e1750658e",  # pragma: allowlist secret
    "sift_query.fbin": "9b0082b67d0ac55b4c7d42216560344567ad87ce3e75a9d5214a0762f1c15d65",  # pragma: allowlist secret
    "sift_gt.ibin": "4c06dd3d1539b1de50f1b7e98a116833ed5c2f1571d0ef81f383a04541e797e7",  # pragma: allowlist secret
}

DEFAULT_RECALL_THRESHOLD: Final[float] = 0.95
DEFAULT_K: Final[int] = 10


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_fbin(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        n, d = struct.unpack("<ii", f.read(8))
        arr = np.fromfile(f, dtype=np.float32, count=n * d)
    if arr.size != n * d:
        raise ValueError(f"{path}: truncated fbin (n={n}, d={d}, read={arr.size})")
    return arr.reshape(n, d)


def read_ibin(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        n, d = struct.unpack("<ii", f.read(8))
        arr = np.fromfile(f, dtype=np.int32, count=n * d)
    if arr.size != n * d:
        raise ValueError(f"{path}: truncated ibin (n={n}, d={d}, read={arr.size})")
    return arr.reshape(n, d)


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def compiler_flags() -> str:
    cflags = os.environ.get("CXXFLAGS", "")
    march = "native" if "-march=native" in cflags else "(default)"
    return f"-Ofast -mavx2 -mfma -march={march} (cmake-default; see CMakeLists.txt)"


def recall_at_k(predicted: np.ndarray, truth: np.ndarray, k: int) -> float:
    if predicted.shape[0] != truth.shape[0]:
        raise ValueError(f"recall: row mismatch predicted={predicted.shape} truth={truth.shape}")
    n = predicted.shape[0]
    hits = 0
    for i in range(n):
        pred_set = {int(x) for x in predicted[i, :k]}
        truth_set = {int(x) for x in truth[i, :k]}
        hits += len(pred_set & truth_set)
    return hits / (n * k)


def verify_dataset(dataset_dir: Path) -> dict[str, str]:
    drift = []
    actual: dict[str, str] = {}
    for name, expected in DATASET_SHAS.items():
        path = dataset_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        sha = sha256_of(path)
        actual[name] = sha
        if sha != expected:
            drift.append(f"{name}: expected {expected}, got {sha}")
    if drift:
        details = "\n  ".join(drift)
        raise RuntimeError(f"SIFT-1M dataset sha mismatch:\n  {details}")
    return actual


def build_and_load(
    *,
    base_path: Path,
    output_dir: Path,
    main_dim: int,
    degree: int,
    ef_indexing: int,
    seed: int,
    num_threads: int,
    skip_existing: bool,
):
    # pylint: disable=import-outside-toplevel
    from alayalite import laser

    params = laser.BuildParams(
        metric="l2",
        main_dim=main_dim,
        R=degree,
        L=100,
        alpha=1.2,
        ef_indexing=ef_indexing,
    )
    return laser.Index.fit(
        str(base_path),
        output_dir=str(output_dir),
        name="sift1m",
        build_params=params,
        seed=seed,
        num_threads=num_threads,
        skip_existing=skip_existing,
        auto_load=True,
    )


def run_ef_sweep(
    index,
    queries: np.ndarray,
    truth: np.ndarray,
    ef_values: list[int],
    k: int,
    num_threads: int,
) -> list[dict]:
    results = []
    for ef in ef_values:
        index.set_params(ef_search=ef, num_threads=num_threads, beam_width=16)
        t0 = time.perf_counter()
        ids = index.batch_search(queries, k)
        elapsed = time.perf_counter() - t0
        ids = np.asarray(ids, dtype=np.int32)
        rec = recall_at_k(ids, truth, k)
        qps = float(queries.shape[0]) / elapsed if elapsed > 0 else float("inf")
        print(f"  ef_search={ef:>4d}  recall@{k}={rec:.4f}  qps={qps:.1f}")
        results.append(
            {
                "ef_search": int(ef),
                "recall_at_k": float(rec),
                "qps": float(qps),
                "elapsed_seconds": float(elapsed),
            }
        )
    return results


def write_report(
    results_dir: Path,
    payload: dict,
    threshold: float,
    best_recall: float,
) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "laser_sift1m_recall.json"
    md_path = results_dir / "laser_sift1m_recall.md"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# LASER SIFT-1M Recall Report",
        "",
        f"- Generated: {payload['generated_at_utc']}",
        f"- Git rev: `{payload['git_rev']}`",
        f"- CPU: {payload['cpu_model']}",
        f"- Compiler flags: `{payload['compiler_flags']}`",
        f"- Recall threshold: {threshold}",
        f"- Best recall@{payload['k']}: **{best_recall:.4f}** ({'PASS' if best_recall >= threshold else 'SOFT-FAIL'})",
        "",
        "## Build params",
        "",
    ]
    for key, val in payload["build_params"].items():
        lines.append(f"- `{key}`: `{val}`")
    lines.append("")
    lines.append("## Dataset shas")
    lines.append("")
    for name, sha in payload["dataset_shas"].items():
        lines.append(f"- `{name}`: `{sha}`")
    lines.append("")
    lines.append("## EF sweep")
    lines.append("")
    lines.append("| ef_search | recall@k | qps | elapsed_s |")
    lines.append("| --- | --- | --- | --- |")
    for row in payload["ef_sweep"]:
        lines.append(
            f"| {row['ef_search']} | {row['recall_at_k']:.4f} | {row['qps']:.1f} | {row['elapsed_seconds']:.2f} |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines))
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--main-dim", type=int, default=64)
    parser.add_argument("--degree", type=int, default=64)
    parser.add_argument("--ef-indexing", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="0 means use alayalite default (cpu_count).",
    )
    parser.add_argument(
        "--ef-sweep",
        type=int,
        nargs="+",
        default=[50, 100, 200, 400, 800],
        help="ef_search values for the recall sweep.",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--recall-threshold",
        type=float,
        default=DEFAULT_RECALL_THRESHOLD,
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(args.dataset_dir)

    base_path = args.dataset_dir / "sift_base.fbin"
    query_path = args.dataset_dir / "sift_query.fbin"
    gt_path = args.dataset_dir / "sift_gt.ibin"

    print("Verifying dataset shas...")
    dataset_shas = verify_dataset(args.dataset_dir)

    print(
        f"Building LASER (main_dim={args.main_dim}, R={args.degree}, "
        f"ef_indexing={args.ef_indexing}, seed={args.seed})..."
    )
    output_dir = args.output_root / f"main_dim_{args.main_dim}"
    if not args.skip_existing and output_dir.exists():
        shutil.rmtree(output_dir)

    t_build = time.perf_counter()
    index = build_and_load(
        base_path=base_path,
        output_dir=output_dir,
        main_dim=args.main_dim,
        degree=args.degree,
        ef_indexing=args.ef_indexing,
        seed=args.seed,
        num_threads=args.num_threads,
        skip_existing=args.skip_existing,
    )
    build_elapsed = time.perf_counter() - t_build
    print(f"Build elapsed: {build_elapsed:.1f}s")

    queries = read_fbin(query_path)
    truth = read_ibin(gt_path)

    print(f"Running ef sweep on {queries.shape[0]} queries (k={args.k})...")
    ef_sweep = run_ef_sweep(
        index,
        queries,
        truth,
        ef_values=list(args.ef_sweep),
        k=args.k,
        num_threads=args.num_threads,
    )

    best_recall = max(row["recall_at_k"] for row in ef_sweep)

    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_rev = "unknown"

    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_rev": git_rev,
        "cpu_model": cpu_model(),
        "compiler_flags": compiler_flags(),
        "k": args.k,
        "build_params": {
            "main_dim": args.main_dim,
            "R": args.degree,
            "L": 100,
            "alpha": 1.2,
            "ef_indexing": args.ef_indexing,
            "seed": args.seed,
        },
        "dataset_dir": str(args.dataset_dir),
        "dataset_shas": dataset_shas,
        "build_elapsed_seconds": float(build_elapsed),
        "ef_sweep": ef_sweep,
        "best_recall_at_k": float(best_recall),
        "recall_threshold": float(args.recall_threshold),
    }

    json_path, md_path = write_report(args.results_dir, payload, args.recall_threshold, best_recall)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    if best_recall >= args.recall_threshold:
        print(f"PASS: best recall@{args.k}={best_recall:.4f} >= {args.recall_threshold}")
        return 0
    print(f"SOFT-FAIL: best recall@{args.k}={best_recall:.4f} < {args.recall_threshold}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
