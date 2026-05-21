#!/usr/bin/env python3
"""CI helper for LASER cross-platform performance workflows."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_LABELS = [
    "linux-libaio-x86_64",
    "macos-threadpool-arm64",
    "macos-threadpool-x86_64",
]


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def _summary_path() -> Path:
    try:
        return Path(os.environ["GITHUB_STEP_SUMMARY"])
    except KeyError as exc:
        raise SystemExit("GITHUB_STEP_SUMMARY is not set") from exc


def _artifact_path(label: str) -> Path:
    return Path("artifacts") / f"laser-perf-{label}.json"


def _markdown_artifact_path(label: str) -> Path:
    return Path("artifacts") / f"laser-perf-{label}.md"


def _detect_backend(override: str) -> str:
    if override != "auto":
        return override
    system = platform.system()
    machine = platform.machine()
    if system == "Darwin":
        return "threadpool"
    if system == "Linux" and machine in {"x86_64", "AMD64"}:
        return "libaio"
    return "unsupported"


def _median(values: list[float | None]) -> float | None:
    concrete = [float(value) for value in values if value is not None]
    if not concrete:
        return None
    return float(statistics.median(concrete))


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _generate_vectors(n: int, dim: int, seed: int, n_clusters: int, cluster_std: float):
    import numpy as np  # pylint: disable=import-outside-toplevel

    rng = np.random.default_rng(seed)
    centers = rng.normal(loc=0.0, scale=1.0, size=(n_clusters, dim)).astype(np.float32)
    assignments = rng.integers(low=0, high=n_clusters, size=n)
    noise = rng.normal(loc=0.0, scale=cluster_std, size=(n, dim)).astype(np.float32)
    vectors = centers[assignments] + noise
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.maximum(norms, np.float32(1e-6))
    return vectors


def _build_laser_fixture(args: argparse.Namespace, fixture_dir: Path) -> Path:
    if args.main_dim != args.dim:
        raise SystemExit(
            "laser_cross_platform_perf: paired native-vs-disk_laser benchmark currently requires "
            f"--main-dim == --dim (got main_dim={args.main_dim}, dim={args.dim})"
        )

    from alayalite import laser as laser_module  # pylint: disable=import-outside-toplevel

    shutil.rmtree(fixture_dir, ignore_errors=True)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    vectors = _generate_vectors(args.n, args.dim, args.seed, args.n_clusters, args.cluster_std)
    qg_name = "dsqg_seg_00000001"
    laser_module.Index.fit(
        vectors,
        output_dir=fixture_dir,
        name=qg_name,
        build_params=laser_module.BuildParams(
            metric="l2",
            main_dim=args.main_dim,
            R=args.degree,
            L=args.build_l,
            alpha=args.alpha,
            ef_indexing=args.ef_indexing,
            ep_num=min(args.ep_num, args.n),
            disable_medoid=True,
        ),
        num_threads=args.build_threads if args.build_threads > 0 else (os.cpu_count() or 1),
        seed=args.seed,
        skip_existing=False,
        auto_load=False,
        dram_budget_gb=args.build_dram_budget_gb,
    )

    vectors_path = fixture_dir / f"{qg_name}_pca_base.fbin"
    if not vectors_path.is_file():
        raise RuntimeError(f"LASER fixture vector file was not written: {vectors_path}")
    return vectors_path


def _run_laser_compare(args: argparse.Namespace, fixture_dir: Path, vectors_path: Path, round_index: int) -> dict:
    run_id = f"{args.label}-round-{round_index}"
    out_root = args.results_dir / f"laser-perf-{args.label}"
    cmd = [
        sys.executable,
        "-m",
        "alayalite.bench.laser_compare",
        "--laser-src-dir",
        str(fixture_dir),
        "--vectors",
        str(vectors_path),
        "--n",
        str(args.n),
        "--dim",
        str(args.dim),
        "--queries",
        str(args.query_num),
        "--k",
        str(args.topk),
        "--ef",
        str(args.ef_search),
        "--beam-width",
        str(args.beam_width),
        "--num-threads",
        str(args.query_threads),
        "--warmup",
        str(args.warmup),
        "--seed",
        str(args.seed),
        "--dram-budget-gb",
        str(args.search_dram_budget_gb),
        "--out",
        str(out_root),
        "--run-id",
        run_id,
    ]
    completed = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "laser_compare failed with exit "
            f"{completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    summary_path = out_root / run_id / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(
            "laser_compare exited 0 but did not write summary.json; "
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _round_result(round_index: int, summary: dict) -> dict[str, float | int | None]:
    comparison = summary["comparison"]
    return {
        "round": round_index,
        "native_qps": comparison.get("qps_native"),
        "disk_laser_qps": comparison.get("qps_disk_laser"),
        "qps_ratio": comparison.get("qps_ratio"),
        "recall_delta": comparison.get("recall_delta"),
    }


def _build_result(args: argparse.Namespace, summaries: list[dict], backend: str) -> dict:
    round_results = [_round_result(index + 1, summary) for index, summary in enumerate(summaries)]
    recall_deltas = [item["recall_delta"] for item in round_results if item["recall_delta"] is not None]
    max_abs_recall_delta = max((abs(float(delta)) for delta in recall_deltas), default=None)
    return {
        "label": args.label,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "backend": backend,
        "dataset": {
            "distribution": "synthetic_gmm_l2norm",
            "n": args.n,
            "dim": args.dim,
            "main_dim": args.main_dim,
            "seed": args.seed,
            "n_clusters": args.n_clusters,
            "cluster_std": args.cluster_std,
        },
        "params": {
            "queries": args.query_num,
            "top_k": args.topk,
            "ef": args.ef_search,
            "beam_width": args.beam_width,
            "rounds": args.rounds,
            "warmup": args.warmup,
            "build_threads": args.build_threads,
            "query_threads": args.query_threads,
            "degree": args.degree,
            "build_l": args.build_l,
            "alpha": args.alpha,
            "ef_indexing": args.ef_indexing,
            "ep_num": args.ep_num,
            "build_dram_budget_gb": args.build_dram_budget_gb,
            "search_dram_budget_gb": args.search_dram_budget_gb,
        },
        "benchmark": {
            "median_native_qps": _median([item["native_qps"] for item in round_results]),
            "median_disk_laser_qps": _median([item["disk_laser_qps"] for item in round_results]),
            "median_qps_ratio": _median([item["qps_ratio"] for item in round_results]),
            "median_recall_delta": _median([item["recall_delta"] for item in round_results]),
            "max_abs_recall_delta": max_abs_recall_delta,
        },
        "round_results": round_results,
        "raw_summaries": summaries,
    }


def _single_summary_lines(result: dict) -> list[str]:
    platform_info = result["platform"]
    dataset = result["dataset"]
    params = result["params"]
    benchmark = result["benchmark"]
    lines = [
        f"## LASER Cross-platform Benchmark ({platform_info['system']} / {platform_info['machine']})",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Label | `{result['label']}` |",
        f"| Backend | `{result['backend']}` |",
        f"| Data size | `{dataset['n']}` |",
        f"| Dimension | `{dataset['dim']}` |",
        f"| Main dimension | `{dataset.get('main_dim', dataset['dim'])}` |",
        f"| Queries | `{params['queries']}` |",
        f"| Top-k | `{params['top_k']}` |",
        f"| ef_search | `{params['ef']}` |",
        f"| Beam width | `{params['beam_width']}` |",
        f"| Rounds | `{params['rounds']}` |",
        f"| Median native QPS | `{_fmt(benchmark['median_native_qps'])}` |",
        f"| Median disk_laser QPS | `{_fmt(benchmark['median_disk_laser_qps'])}` |",
        f"| Median QPS ratio | `{_fmt(benchmark['median_qps_ratio'])}` |",
        f"| Median recall delta | `{_fmt(benchmark['median_recall_delta'], 4)}` |",
        f"| Max abs recall delta | `{_fmt(benchmark['max_abs_recall_delta'], 4)}` |",
        "",
        "### Per-round results",
        "",
        "| Round | Native QPS | disk_laser QPS | QPS Ratio | Recall Delta |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in result["round_results"]:
        lines.append(
            f"| {item['round']} | {_fmt(item['native_qps'])} | {_fmt(item['disk_laser_qps'])} | "
            f"{_fmt(item['qps_ratio'])} | {_fmt(item['recall_delta'], 4)} |"
        )
    return lines


def _write_result_files(result: dict, output_path: Path, markdown_path: Path, results_dir: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_text = json.dumps(result, indent=2, sort_keys=True)
    markdown_text = "\n".join(_single_summary_lines(result)) + "\n"
    output_path.write_text(output_text + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_text, encoding="utf-8")

    label_slug = str(result["label"]).replace("-", "_")
    (results_dir / f"laser_{label_slug}_perf.json").write_text(output_text + "\n", encoding="utf-8")
    (results_dir / f"laser_{label_slug}_perf.md").write_text(markdown_text, encoding="utf-8")
    if str(result["label"]).startswith("macos-"):
        (results_dir / "laser_macos_smoke.json").write_text(output_text + "\n", encoding="utf-8")
        (results_dir / "laser_macos_smoke.md").write_text(markdown_text, encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> int:
    if args.query_threads != 1:
        raise SystemExit("laser_cross_platform_perf: --query-threads must be 1 for fair disk_laser comparison")
    if args.rounds <= 0:
        raise SystemExit("laser_cross_platform_perf: --rounds must be > 0")

    backend = _detect_backend(args.backend)
    fixture_dir = args.work_dir / args.label / "fixture"
    vectors_path = _build_laser_fixture(args, fixture_dir)
    summaries = [_run_laser_compare(args, fixture_dir, vectors_path, index + 1) for index in range(args.rounds)]
    result = _build_result(args, summaries, backend)
    output_path = args.output or _artifact_path(args.label)
    markdown_path = args.markdown_output or _markdown_artifact_path(args.label)
    _write_result_files(result, output_path, markdown_path, args.results_dir)
    return 0


def append_summary(args: argparse.Namespace) -> int:
    result_path = args.result or _artifact_path(args.label)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    with _summary_path().open("a", encoding="utf-8") as summary:
        summary.write("\n".join(_single_summary_lines(result)))
        summary.write("\n")
    return 0


def _aggregate_summary_lines(result_paths: list[Path], expected_labels: list[str], baseline_label: str) -> list[str]:
    if not result_paths:
        raise SystemExit("No LASER benchmark artifacts found to aggregate.")

    label_order = {label: index for index, label in enumerate(expected_labels)}
    results: list[dict[str, Any]] = []
    for result_path in result_paths:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        platform_info = result["platform"]
        benchmark = result["benchmark"]
        params = result["params"]
        dataset = result["dataset"]
        results.append(
            {
                "label": result["label"],
                "platform": f"{platform_info['system']} / {platform_info['machine']}",
                "backend": result["backend"],
                "n": dataset["n"],
                "dim": dataset["dim"],
                "queries": params["queries"],
                "top_k": params["top_k"],
                "ef": params["ef"],
                "beam_width": params["beam_width"],
                "rounds": params["rounds"],
                "native_qps": benchmark["median_native_qps"],
                "disk_laser_qps": benchmark["median_disk_laser_qps"],
                "qps_ratio": benchmark["median_qps_ratio"],
                "recall_delta": benchmark["median_recall_delta"],
                "max_abs_recall_delta": benchmark["max_abs_recall_delta"],
            }
        )

    results.sort(key=lambda item: label_order.get(item["label"], len(label_order)))
    present_labels = {item["label"] for item in results}
    missing_labels = [label for label in expected_labels if label not in present_labels]
    baseline = next((item for item in results if item["label"] == baseline_label), None)
    baseline_qps = baseline["disk_laser_qps"] if baseline is not None else None
    summary_baseline = baseline if baseline is not None else results[0]
    lines = [
        "## LASER Cross-platform Benchmark Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Data size | `{summary_baseline['n']}` |",
        f"| Dimension | `{summary_baseline['dim']}` |",
        f"| Queries | `{summary_baseline['queries']}` |",
        f"| Top-k | `{summary_baseline['top_k']}` |",
        f"| ef_search | `{summary_baseline['ef']}` |",
        f"| Beam width | `{summary_baseline['beam_width']}` |",
        f"| Rounds | `{summary_baseline['rounds']}` |",
        f"| libaio baseline | `{baseline_label}{'' if baseline is not None else ' (missing)'}` |",
        "",
        "| Platform | Label | Backend | Native QPS | disk_laser QPS | QPS Ratio | "
        "vs libaio baseline | Recall Delta | Max abs recall delta |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        if baseline_qps and item["disk_laser_qps"]:
            vs_baseline = float(item["disk_laser_qps"]) / float(baseline_qps)
        else:
            vs_baseline = None
        lines.append(
            f"| {item['platform']} | `{item['label']}` | `{item['backend']}` | "
            f"{_fmt(item['native_qps'])} | {_fmt(item['disk_laser_qps'])} | "
            f"{_fmt(item['qps_ratio'])} | {_fmt(vs_baseline)} | "
            f"{_fmt(item['recall_delta'], 4)} | {_fmt(item['max_abs_recall_delta'], 4)} |"
        )
    if missing_labels:
        lines.extend(["", "### Missing results", "", ", ".join(f"`{label}`" for label in missing_labels)])
    return lines


def _aggregate_result_paths(results_dir: Path) -> list[Path]:
    return sorted(results_dir.rglob("laser-perf-*.json"))


def aggregate_summary(args: argparse.Namespace) -> int:
    result_paths = _aggregate_result_paths(args.results_dir)
    expected_labels = args.expected_label or DEFAULT_LABELS
    with _summary_path().open("a", encoding="utf-8") as summary:
        summary.write("\n".join(_aggregate_summary_lines(result_paths, expected_labels, args.baseline_label)))
        summary.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run-benchmark")
    run.add_argument("--label", default=_env("BENCHMARK_LABEL", "local"))
    # Defaults aligned with examples/laser/configs/gist1m_run01.toml; dim==main_dim
    # is enforced by the v1 LaserSegmentImporter (alayalite.bench.laser_compare).
    # n defaults to 10k for PR/schedule (smoke validation of build+run flow);
    # paper-grade perf numbers (n=1M) are obtained via workflow_dispatch.
    run.add_argument("--n", type=int, default=int(_env("LASER_PERF_N", "10000")))
    run.add_argument("--dim", type=int, default=int(_env("LASER_PERF_DIM", "256")))
    run.add_argument("--main-dim", type=int, default=int(_env("LASER_PERF_MAIN_DIM", "256")))
    run.add_argument("--n-clusters", type=int, default=int(_env("LASER_PERF_N_CLUSTERS", "64")))
    run.add_argument("--cluster-std", type=float, default=float(_env("LASER_PERF_CLUSTER_STD", "0.35")))
    run.add_argument("--degree", type=int, default=int(_env("LASER_PERF_DEGREE", "64")))
    run.add_argument("--build-l", type=int, default=int(_env("LASER_PERF_BUILD_L", "200")))
    run.add_argument("--alpha", type=float, default=float(_env("LASER_PERF_ALPHA", "1.2")))
    run.add_argument("--ef-indexing", type=int, default=int(_env("LASER_PERF_EF_INDEXING", "200")))
    run.add_argument("--ep-num", type=int, default=int(_env("LASER_PERF_EP_NUM", "300")))
    run.add_argument("--query-num", type=int, default=int(_env("BENCHMARK_QUERY_NUM", "100")))
    run.add_argument("--topk", type=int, default=int(_env("BENCHMARK_TOPK", "10")))
    run.add_argument("--ef-search", type=int, default=int(_env("BENCHMARK_EF_SEARCH", "200")))
    run.add_argument("--beam-width", type=int, default=int(_env("BENCHMARK_BEAM_WIDTH", "16")))
    run.add_argument("--rounds", type=int, default=int(_env("BENCHMARK_ROUNDS", "3")))
    run.add_argument("--warmup", type=int, default=int(_env("BENCHMARK_WARMUP", "2")))
    run.add_argument("--build-threads", type=int, default=int(_env("BENCHMARK_BUILD_THREADS", "0")))
    run.add_argument("--query-threads", type=int, default=int(_env("BENCHMARK_QUERY_THREADS", "1")))
    run.add_argument("--seed", type=int, default=int(_env("LASER_PERF_SEED", "42")))
    run.add_argument(
        "--build-dram-budget-gb",
        type=float,
        default=float(_env("LASER_PERF_BUILD_DRAM_BUDGET_GB", "4.0")),
    )
    # v1 paired comparison: alayalite.bench.laser_compare hard-rejects values
    # other than 0.5 because the DiskCollection.import_laser_segment binding
    # does not expose search_dram_budget_gb (paper toml's 1.0 only applies to
    # single-side LASER benchmarks). See laser_compare.py finding 1.1.
    run.add_argument(
        "--search-dram-budget-gb",
        type=float,
        default=float(_env("LASER_PERF_SEARCH_DRAM_BUDGET_GB", "0.5")),
    )
    run.add_argument("--backend", default=_env("LASER_PERF_BACKEND", "auto"))
    run.add_argument("--work-dir", type=Path, default=Path(_env("LASER_PERF_WORK_DIR", ".github-tmp/laser-perf")))
    run.add_argument("--results-dir", type=Path, default=Path("results"))
    run.add_argument("--output", type=Path, default=None)
    run.add_argument("--markdown-output", type=Path, default=None)
    run.set_defaults(func=run_benchmark)

    single = subparsers.add_parser("append-summary")
    single.add_argument("--label", default=_env("BENCHMARK_LABEL", "local"))
    single.add_argument("--result", type=Path, default=None)
    single.set_defaults(func=append_summary)

    aggregate = subparsers.add_parser("aggregate-summary")
    aggregate.add_argument("--results-dir", type=Path, default=Path("artifacts/benchmarks"))
    aggregate.add_argument("--expected-label", action="append", default=None)
    aggregate.add_argument("--baseline-label", default="linux-libaio-x86_64")
    aggregate.set_defaults(func=aggregate_summary)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
