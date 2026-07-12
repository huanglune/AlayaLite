#!/usr/bin/env python3
"""Re-run and collect the locked hot-path performance baseline on a remote host."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shlex
import statistics
import subprocess
from pathlib import Path


SIMD = [
    "fht_benchmark", "ip_full_benchmark", "ip_sq4_benchmark", "ip_sq8_benchmark",
    "l2_sqr_full_benchmark", "l2_sqr_sq4_benchmark", "l2_sqr_sq8_benchmark",
]


def run(cmd: list[str], *, check: bool = True) -> str:
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if check and result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {shlex.join(cmd)}\n{result.stderr}")
    return result.stdout


def ssh(host: str, command: str, *, check: bool = True) -> str:
    return run(["ssh", host, command], check=check)


def remote_cat(host: str, path: str) -> str:
    return ssh(host, f"cat {shlex.quote(path)}")


def time_fields(text: str) -> dict:
    def field(name: str) -> str | None:
        match = re.search(rf"^\s*{re.escape(name)}:\s*(.+)$", text, re.MULTILINE)
        return match.group(1).strip() if match else None

    elapsed = field("Elapsed (wall clock) time (h:mm:ss or m:ss)")
    seconds = None
    if elapsed:
        parts = [float(x) for x in elapsed.split(":")]
        seconds = sum(value * 60 ** power for power, value in enumerate(reversed(parts)))
    rss = field("Maximum resident set size (kbytes)")
    return {"duration_s": seconds, "peak_rss_kb": int(rss) if rss else None}


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lo, hi = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def parse_simd(host: str, raw: str) -> list[dict]:
    results = []
    for binary in SIMD:
        by_key: dict[tuple[int, str], list[float]] = {}
        rss, durations = [], []
        for round_no in range(1, 6):
            output = remote_cat(host, f"{raw}/simd/{binary}.round{round_no}.out")
            timing = time_fields(remote_cat(host, f"{raw}/simd/{binary}.round{round_no}.time"))
            rss.append(timing["peak_rss_kb"])
            durations.append(timing["duration_s"])
            for line in output.splitlines():
                if not line.startswith("|") or "baseline" in line or "---" in line:
                    continue
                cells = [cell.strip().replace("**", "") for cell in line.strip("|").split("|")]
                dim_match = re.search(r"(?:\((\d+)\)|^(\d+)$)", cells[0])
                if not dim_match:
                    continue
                dim = int(dim_match.group(1) or dim_match.group(2))
                for variant, cell in zip(("generic", "avx2", "avx512", "auto"), cells[1:]):
                    ns = re.search(r"([0-9.]+) ns", cell)
                    if ns:
                        by_key.setdefault((dim, variant), []).append(float(ns.group(1)))
        for (dim, variant), values in sorted(by_key.items()):
            results.append({
                "name": f"{binary}:{variant}:d{dim}",
                "binary": f"build/Release/tests/simd/{binary}",
                "dataset": "deterministic_generated_seeded_vectors",
                "params": {"dimension": dim, "variant": variant, "iterations_per_round": 100000},
                "threads": 1, "rounds": 5, "unit": "ns_per_call",
                "median": statistics.median(values), "p50": percentile(values, .50),
                "p95": percentile(values, .95), "p99": percentile(values, .99),
                "round_values": values, "qps": None, "recall": None,
                "peak_rss_kb": max(x for x in rss if x is not None),
                "duration_s_median": statistics.median(x for x in durations if x is not None),
                "artifact_sha256": None,
            })
    return results


def parse_qg(host: str, raw: str, tmp: str) -> dict:
    curves: dict[int, list[dict]] = {}
    timings = []
    for round_no in range(1, 6):
        output = remote_cat(host, f"{raw}/qg/deep1m.round{round_no}.out")
        timings.append(time_fields(remote_cat(host, f"{raw}/qg/deep1m.round{round_no}.time")))
        for ef, qps, recall in re.findall(r"^(\d+)\t([0-9.]+)\t([0-9.]+)$", output, re.MULTILINE):
            curves.setdefault(int(ef), []).append({"qps": float(qps), "recall": float(recall)})
    sha_line = remote_cat(host, f"{raw}/qg/deep1m.sha256").split()[0]
    size = int(remote_cat(host, f"{raw}/qg/deep1m.size").strip())
    rows = []
    for ef, values in sorted(curves.items()):
        qps_values = [x["qps"] for x in values]
        recall_values = [x["recall"] for x in values]
        rows.append({"ef": ef, "qps_median": statistics.median(qps_values),
                     "recall_median": statistics.median(recall_values), "round_values": values})
    warm = [x["duration_s"] for x in timings[1:] if x["duration_s"] is not None]
    return {
        "name": "rabitq_qg_deep1m_l2", "binary": "build/Release/tests/index/rabitq_performance_test",
        "dataset": "deep1M (1,000,000 x 96, L2)",
        "params": {"gtest_filter": "RaBitQDeep1MTest.Deep1MQGTest", "topk": 10,
                   "efs": sorted(curves), "internal_rounds_per_process": 3},
        "threads": "builder default; NUMA node0 affinity (48 logical CPUs), search serial",
        "rounds": 5, "median": None, "p50": None, "p95": None, "p99": None,
        "qps_recall_curve": rows, "peak_rss_kb": max(x["peak_rss_kb"] for x in timings if x["peak_rss_kb"]),
        "cold_total_duration_s": timings[0]["duration_s"],
        "warm_total_duration_s_median": statistics.median(warm),
        "estimated_build_duration_s": timings[0]["duration_s"] - statistics.median(warm),
        "artifact": f"{tmp}/work/data/deep1M/deep1M_rabitq.qg", "artifact_size_bytes": size,
        "artifact_sha256": sha_line,
    }


def collect_environment(host: str, repo: Path, raw: str) -> dict:
    remote = ssh(host, "hostname; lscpu | grep -m1 'Model name'; nproc; uname -r; "
                       "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>&1; "
                       "numactl --hardware 2>&1; uptime")
    cache = (repo / "build/Release/CMakeCache.txt").read_text(encoding="utf-8")
    cmake = {key: match.group(1) for key in ("CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER",
             "CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE")
             if (match := re.search(rf"^{key}:[^=]*=(.*)$", cache, re.MULTILINE))}
    return {
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(), "remote_snapshot": remote,
        "load_before_qg": remote_cat(host, f"{raw}/qg/load-before.txt").strip(),
        "load_after_qg": remote_cat(host, f"{raw}/qg/load-after.txt").strip(),
        "compiler_version": run([cmake["CMAKE_CXX_COMPILER"], "--version"]).splitlines()[0],
        "cmake": cmake,
        "release_binary_build_time": dt.datetime.fromtimestamp(
            (repo / "build/Release/tests/simd/fht_benchmark").stat().st_mtime,
            dt.timezone.utc).isoformat(),
        "git_commit": run(["git", "rev-parse", "HEAD"]).strip(),
    }


def execute(args: argparse.Namespace, repo: Path) -> None:
    root, tmp, raw = str(repo), args.remote_tmp, f"{args.remote_tmp}/raw"
    setup = f"""set -euo pipefail
mkdir -p {tmp}/work/data/deep1M {tmp}/work/run {raw}/simd {raw}/qg {raw}/laser
for f in {args.data_root}/deep1M/deep1M_{{base.fvecs,query.fvecs,groundtruth.ivecs}}; do ln -sfn \"$f\" {tmp}/work/data/deep1M/; done
"""
    ssh(args.host, setup)
    simd = f"""set -euo pipefail; cd {tmp}/work/run; ROOT={root}; OUT={raw}/simd
for r in 1 2 3 4 5; do
  for b in ip_full_benchmark ip_sq4_benchmark ip_sq8_benchmark l2_sqr_full_benchmark l2_sqr_sq4_benchmark l2_sqr_sq8_benchmark; do
    /usr/bin/time -v -o \"$OUT/$b.round$r.time\" numactl --cpunodebind=0 --membind=0 \"$ROOT/build/Release/tests/simd/$b\" 64 96 128 256 512 960 1024 > \"$OUT/$b.round$r.out\" 2> \"$OUT/$b.round$r.err\"
  done
  /usr/bin/time -v -o \"$OUT/fht_benchmark.round$r.time\" numactl --cpunodebind=0 --membind=0 \"$ROOT/build/Release/tests/simd/fht_benchmark\" 6 7 8 9 10 > \"$OUT/fht_benchmark.round$r.out\" 2> \"$OUT/fht_benchmark.round$r.err\"
done"""
    ssh(args.host, simd)
    qg = f"""set -euo pipefail; cd {tmp}/work/run; ROOT={root}; OUT={raw}/qg; IDX={tmp}/work/data/deep1M/deep1M_rabitq.qg
rm -f \"$IDX\"; uptime > \"$OUT/load-before.txt\"
for r in 1 2 3 4 5; do
  /usr/bin/time -v -o \"$OUT/deep1m.round$r.time\" numactl --cpunodebind=0 --membind=0 \"$ROOT/build/Release/tests/index/rabitq_performance_test\" --gtest_filter=RaBitQDeep1MTest.Deep1MQGTest > \"$OUT/deep1m.round$r.out\" 2> \"$OUT/deep1m.round$r.err\"
done
uptime > \"$OUT/load-after.txt\"; sha256sum \"$IDX\" > \"$OUT/deep1m.sha256\"; stat -c %s \"$IDX\" > \"$OUT/deep1m.size\""""
    ssh(args.host, qg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="g03")
    parser.add_argument("--data-root", default="/md1/huangliang/data")
    parser.add_argument("--remote-tmp", default="/md1/huangliang/tmp/perf-baseline")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--collect-existing", action="store_true",
                        help="collect previously generated raw files without rerunning")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if not args.collect_existing:
        execute(args, repo)
    raw = f"{args.remote_tmp}/raw"
    result = {
        "schema_version": 1, "environment": collect_environment(args.host, repo, raw),
        "method": {"host": args.host, "numa_binding": "--cpunodebind=0 --membind=0",
                   "external_rounds": 5, "gate_regression_threshold_pct": 3.0},
        "simd": parse_simd(args.host, raw),
        "qg": [parse_qg(args.host, raw, args.remote_tmp)],
        "laser": {"status": "skipped", "cpp_inventory": ["test_laser_compile", "test_rotator_dump",
                  "test_laser_page_layout_round_trip", "test_threadpool_file_reader",
                  "test_iocp_file_reader", "laser_simd_dispatch_test", "rabitq_factor_equivalence_test"],
                  "reason": "No LASER build/search perf C++ target exists; Python perf requires alayalite bindings and this worktree has no .venv. BUILD_PYTHON intentionally not enabled."},
        "coverage_exceptions": [
            "gist1m QG skipped: rabitq_performance_test hard-codes deep1M L2 and disabled T2I-1M only.",
            "DiskANN update benchmark intentionally deferred until wave3 per task decision.",
        ],
    }
    commit8 = result["environment"]["git_commit"][:8]
    output = args.output or repo / f"scripts/perf_baseline/baseline-{commit8}-{args.host}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
