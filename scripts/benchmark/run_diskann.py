#!/usr/bin/env python3
"""Run the DiskANN searcher benchmark and save results.

Results are written to benchmark/results/diskann_searcher/runs/<run_id>/.

Usage:
    # Basic run
    uv run scripts/benchmark/run_diskann.py

    # Custom parameters
    uv run scripts/benchmark/run_diskann.py \
        --data-dir /md1/huangliang/data \
        --index-prefix /md1/huangliang/data/diskann_bench/bench_idx \
        --skip-build \
        --ef-values 16,32,64 \
        --cache-values 20,50,100 \
        --thread-values 1,4,16

    # Export JSON alongside CSV
    uv run scripts/benchmark/run_diskann.py --json
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "benchmark" / "results" / "diskann_searcher" / "runs"
DEFAULT_BUILD_DIR = REPO_ROOT / "build"
DEFAULT_BENCHMARK_BIN = DEFAULT_BUILD_DIR / "benchmark" / "index" / "diskann_searcher_benchmark"
DEFAULT_INDEX_PREFIX = "/tmp/diskann_searcher_benchmark/bench_idx"

BENCHMARK_RE = re.compile(
    r"^(?P<family>[^/]+)/(?P<topk>\d+)/(?P<ef_search>\d+)/(?P<cache_pct>\d+)/"
    r"(?P<num_threads>\d+)/iterations:(?P<iterations>\d+)/real_time$"
)

STRUCTURED_COLUMNS = [
    "run_id",
    "captured_at_utc",
    "topk",
    "ef_search",
    "cache_pct",
    "num_threads",
    "qps",
    "recall_at_k",
    "eval_reuse_hit_rate",
    "cache_footprint_mb",
    "bypass_page_cache",
    "ready_to_search_rss_mb",
    "thread_amplified_rss_mb",
    "timed_rss_delta_mb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DiskANN searcher benchmark.")
    parser.add_argument(
        "--run-id",
        default="",
        help="Run ID (default: UTC timestamp like 20260317-153138).",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Dataset root directory (ALAYA_DATA_DIR). Default: <repo>/data.",
    )
    parser.add_argument(
        "--index-prefix",
        default="",
        help=f"Index file path prefix. Default: {DEFAULT_INDEX_PREFIX}.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse existing index (ALAYA_BENCH_SKIP_BUILD=1).",
    )
    parser.add_argument(
        "--ef-values",
        default="",
        help="Comma-separated ef_search values (e.g. 16,32,64).",
    )
    parser.add_argument(
        "--cache-values",
        default="",
        help="Comma-separated cache percent values (e.g. 20,50,100).",
    )
    parser.add_argument(
        "--thread-values",
        default="",
        help="Comma-separated thread counts (e.g. 1,4,16).",
    )
    parser.add_argument(
        "--benchmark-filter",
        default="",
        help="Google Benchmark --benchmark_filter regex.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also save raw Google Benchmark JSON output.",
    )
    parser.add_argument(
        "--build-dir",
        default="",
        help=f"CMake build directory. Default: {DEFAULT_BUILD_DIR}.",
    )
    return parser.parse_args()


def resolve_benchmark_bin(build_dir: str) -> Path:
    if build_dir:
        base = Path(build_dir)
        if not base.is_absolute():
            base = REPO_ROOT / base
    else:
        base = DEFAULT_BUILD_DIR
    binary = base / "benchmark" / "index" / "diskann_searcher_benchmark"
    if not binary.exists():
        print(f"Error: benchmark binary not found: {binary}", file=sys.stderr)
        print("Run 'make build' first.", file=sys.stderr)
        sys.exit(1)
    return binary


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.data_dir:
        env["ALAYA_DATA_DIR"] = args.data_dir
    elif "ALAYA_DATA_DIR" not in env:
        env["ALAYA_DATA_DIR"] = str(REPO_ROOT / "data")
    if args.index_prefix:
        env["ALAYA_BENCH_INDEX_PREFIX"] = args.index_prefix
    env["ALAYA_BENCH_SKIP_BUILD"] = "1" if args.skip_build else "0"
    if args.ef_values:
        env["ALAYA_BENCH_EF_SEARCH_VALUES"] = args.ef_values
    if args.cache_values:
        env["ALAYA_BENCH_CACHE_PCT_VALUES"] = args.cache_values
    if args.thread_values:
        env["ALAYA_BENCH_THREAD_VALUES"] = args.thread_values
    return env


def run_benchmark(
    binary: Path,
    env: dict[str, str],
    run_dir: Path,
    benchmark_filter: str,
    save_json: bool,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = run_dir / "raw.txt"

    cmd = [
        str(binary),
        f"--benchmark_out={raw_csv}",
        "--benchmark_out_format=csv",
        "--benchmark_counters_tabular=true",
    ]
    if benchmark_filter:
        cmd.append(f"--benchmark_filter={benchmark_filter}")

    print(f"$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=env, text=True)
    if result.returncode != 0:
        print(f"Benchmark exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    if save_json:
        json_out = run_dir / "raw.json"
        cmd_json = [
            str(binary),
            f"--benchmark_out={json_out}",
            "--benchmark_out_format=json",
            "--benchmark_counters_tabular=true",
        ]
        if benchmark_filter:
            cmd_json.append(f"--benchmark_filter={benchmark_filter}")
        print(f"$ {' '.join(cmd_json)}", flush=True)
        subprocess.run(cmd_json, env=env, text=True, check=False)

    return raw_csv


def parse_raw_csv(raw_csv: Path) -> list[dict[str, str]]:
    lines = raw_csv.read_text(encoding="utf-8").splitlines()
    header_idx = next(
        (i for i, line in enumerate(lines) if line.startswith("name,")),
        None,
    )
    if header_idx is None:
        print(f"Error: no CSV header found in {raw_csv}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict[str, str]] = []
    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
        match = BENCHMARK_RE.match(row.get("name", ""))
        if match is None:
            continue
        parsed = dict(row)
        parsed.update(match.groupdict())
        rows.append(parsed)
    return rows


def write_structured_csv(
    path: Path,
    run_id: str,
    captured_at: str,
    raw_rows: list[dict[str, str]],
) -> None:
    structured: list[dict[str, str]] = []
    for row in raw_rows:
        structured.append(
            {
                "run_id": run_id,
                "captured_at_utc": captured_at,
                "topk": row["topk"],
                "ef_search": row["ef_search"],
                "cache_pct": row["cache_pct"],
                "num_threads": row["num_threads"],
                "qps": row.get("QPS", ""),
                "recall_at_k": row.get("Recall@K", ""),
                "eval_reuse_hit_rate": row.get("EvalReuseHitRate", ""),
                "cache_footprint_mb": row.get("CacheFootprintMB", ""),
                "bypass_page_cache": row.get("BypassPageCache", ""),
                "ready_to_search_rss_mb": row.get("ReadyToSearchRSSMB", ""),
                "thread_amplified_rss_mb": row.get("ThreadAmplifiedRSSMB", ""),
                "timed_rss_delta_mb": row.get("TimedRSSDeltaMB", ""),
            }
        )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STRUCTURED_COLUMNS)
        writer.writeheader()
        writer.writerows(structured)


def write_manifest(path: Path, entries: list[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(entries)


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    captured_at = datetime.now(timezone.utc).isoformat()

    binary = resolve_benchmark_bin(args.build_dir)
    env = build_env(args)
    run_dir = RUNS_ROOT / run_id

    print(f"Run ID: {run_id}")
    print(f"Results directory: {run_dir}")
    print()

    raw_csv = run_benchmark(binary, env, run_dir, args.benchmark_filter, args.json)
    raw_rows = parse_raw_csv(raw_csv)

    if not raw_rows:
        print("Error: no benchmark results parsed.", file=sys.stderr)
        sys.exit(1)

    structured_csv = run_dir / "structured.csv"
    write_structured_csv(structured_csv, run_id, captured_at, raw_rows)

    manifest_entries = [
        ("run_id", run_id),
        ("captured_at_utc", captured_at),
        ("data_dir", env.get("ALAYA_DATA_DIR", "")),
        ("index_prefix", env.get("ALAYA_BENCH_INDEX_PREFIX", DEFAULT_INDEX_PREFIX)),
        ("skip_build", str(args.skip_build).lower()),
        ("ef_search_values", args.ef_values or "(default)"),
        ("cache_pct_values", args.cache_values or "(default)"),
        ("thread_values", args.thread_values or "(default)"),
        ("benchmark_filter", args.benchmark_filter or "(all)"),
        ("case_count", str(len(raw_rows))),
    ]
    write_manifest(run_dir / "manifest.csv", manifest_entries)

    print()
    print(f"Parsed {len(raw_rows)} benchmark cases.")
    print(f"Structured CSV: {structured_csv}")
    print(f"Manifest:       {run_dir / 'manifest.csv'}")
    print(f"Raw CSV:        {raw_csv}")


if __name__ == "__main__":
    main()
