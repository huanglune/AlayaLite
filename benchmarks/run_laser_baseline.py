#!/usr/bin/env python3
"""Run or collect the locked LASER deep1M baseline on a remote host."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shlex
import statistics
import struct
import subprocess
from pathlib import Path

EFS = (100, 200, 300)


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {shlex.join(cmd)}\n{result.stderr}")
    return result.stdout


def ssh(host: str, command: str) -> str:
    return run(["ssh", host, command])


def remote_cat(host: str, path: str) -> str:
    return ssh(host, f"cat {shlex.quote(path)}")


def time_fields(text: str) -> dict:
    elapsed = re.search(
        r"^\s*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)$",
        text,
        re.MULTILINE,
    )
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    seconds = None
    if elapsed:
        parts = [float(value) for value in elapsed.group(1).split(":")]
        seconds = sum(value * 60**power for power, value in enumerate(reversed(parts)))
    return {"duration_s": seconds, "peak_rss_kb": int(rss.group(1)) if rss else None}


def prepare_worker(args: argparse.Namespace) -> None:
    import numpy as np

    source = Path(args.data_root) / "deep1M"
    base = Path(args.tmp_root)
    input_dir = base / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (base / "output").mkdir(parents=True, exist_ok=True)
    (base / "raw").mkdir(parents=True, exist_ok=True)

    def convert(source_path: Path, target: Path, *, vectors: bool) -> None:
        raw = np.fromfile(source_path, dtype=np.int32)
        dim = int(raw[0])
        rows = raw.reshape(-1, dim + 1)[:, 1:]
        data = np.ascontiguousarray(rows.view(np.float32) if vectors else rows.astype(np.uint32))
        with target.open("wb") as handle:
            handle.write(struct.pack("<II", *data.shape))
            data.tofile(handle)

    convert(source / "deep1M_base.fvecs", input_dir / "deep_base.fbin", vectors=True)
    convert(source / "deep1M_query.fvecs", input_dir / "deep_query.fbin", vectors=True)
    convert(source / "deep1M_groundtruth.ivecs", input_dir / "deep_gt.ibin", vectors=False)
    (base / "deep.toml").write_text(
        f'''seed = 42
[dataset]
name = "deep1m"
metric = "l2"
degree = 64
main_dimension = 64
[paths]
base = "{input_dir}/deep_base.fbin"
query = "{input_dir}/deep_query.fbin"
gt = "{input_dir}/deep_gt.ibin"
output = "{base}/output"
[build_vamana]
L = 200
alpha = 1.2
dram_budget_gb = 32.0
[build]
build_threads = 48
ef_indexing = 200
[search]
topk = 10
threads = 1
beam_width = 16
dram_budget = 1.0
ep_num = 300
warmup = 1
runs = 1
efs = [100, 200, 300]
''',
        encoding="utf-8",
    )


def execute(args: argparse.Namespace, repo: Path) -> None:
    root, tmp = shlex.quote(str(repo)), shlex.quote(args.tmp_root)
    python = shlex.quote(str(repo / ".venv/bin/python"))
    script = shlex.quote(str(repo / "benchmarks/run_laser_baseline.py"))
    ssh(
        args.host,
        f"rm -rf {tmp}; {python} {script} --worker-prepare --data-root {shlex.quote(args.data_root)} --tmp-root {tmp}; uptime > {tmp}/raw/load-before.txt",
    )
    build = (
        f"set -euo pipefail; cd {root}; /usr/bin/time -v -o {tmp}/raw/build.time "
        f"numactl --cpunodebind=0 --membind=0 {python} examples/laser/main.py -c {tmp}/deep.toml build "
        f"> {tmp}/raw/build.out 2> {tmp}/raw/build.err; uptime > {tmp}/raw/load-after-build.txt"
    )
    ssh(args.host, build)
    search = f"set -euo pipefail; cd {root}; "
    search += (
        f"for r in 1 2 3 4 5; do /usr/bin/time -v -o {tmp}/raw/search.round$r.time "
        f"numactl --cpunodebind=0 --membind=0 {python} examples/laser/main.py -c {tmp}/deep.toml search "
        f"> {tmp}/raw/search.round$r.out 2> {tmp}/raw/search.round$r.err; done; "
        f"uptime > {tmp}/raw/load-after-search.txt; D={tmp}/output/data/deep1m; "
        f'find "$D" -type f -print0 | sort -z | xargs -0 sha256sum > {tmp}/raw/artifacts.sha256; '
        f"find \"$D\" -type f -printf '%f %s\\n' | sort > {tmp}/raw/artifacts.size"
    )
    ssh(args.host, search)


def collect(args: argparse.Namespace, repo: Path) -> dict:
    raw = f"{args.tmp_root}/raw"
    build = time_fields(remote_cat(args.host, f"{raw}/build.time"))
    rounds: dict[int, list[dict]] = {ef: [] for ef in EFS}
    search_timings = []
    row_re = re.compile(r"^\s*(100|200|300)\s+([0-9.]+)\s+([0-9.]+)%", re.MULTILINE)
    for round_no in range(1, 6):
        output = remote_cat(args.host, f"{raw}/search.round{round_no}.out")
        search_timings.append(time_fields(remote_cat(args.host, f"{raw}/search.round{round_no}.time")))
        for ef, qps, recall in row_re.findall(output):
            rounds[int(ef)].append({"qps": float(qps), "recall_at_10": float(recall) / 100.0})
    hashes = {}
    for line in remote_cat(args.host, f"{raw}/artifacts.sha256").splitlines():
        digest, path = line.split(maxsplit=1)
        hashes[Path(path).name] = digest
    sizes = {}
    for line in remote_cat(args.host, f"{raw}/artifacts.size").splitlines():
        name, size = line.rsplit(maxsplit=1)
        sizes[name] = int(size)
    manifest_hash = hashlib.sha256(
        "".join(f"{name}\0{sizes[name]}\0{hashes[name]}\n" for name in sorted(hashes)).encode()
    ).hexdigest()
    curve = []
    for ef in EFS:
        values = rounds[ef]
        curve.append(
            {
                "ef": ef,
                "qps_median": statistics.median(row["qps"] for row in values),
                "recall_at_10_median": statistics.median(row["recall_at_10"] for row in values),
                "round_values": values,
            }
        )
    return {
        "schema_version": 1,
        "source_revision": run(["git", "rev-parse", "14ad5369"]).strip(),
        "recorded_from_commit": run(["git", "rev-parse", "HEAD"]).strip(),
        "host": args.host,
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": {
            "venv_python": ssh(args.host, f"{repo}/.venv/bin/python -V 2>&1").strip(),
            "venv_reused_over_nfs": True,
            "numa_policy": "numactl --cpunodebind=0 --membind=0",
            "load_before": remote_cat(args.host, f"{raw}/load-before.txt").strip(),
            "load_after_build": remote_cat(args.host, f"{raw}/load-after-build.txt").strip(),
            "load_after_search": remote_cat(args.host, f"{raw}/load-after-search.txt").strip(),
        },
        "benchmarks": [
            {
                "name": "laser_deep1m_l2",
                "binary": "examples/laser/main.py (alayalite.laser.Index)",
                "dataset": "deep1M (1,000,000 x 96, L2)",
                "params": {
                    "R": 64,
                    "main_dim": 64,
                    "L": 200,
                    "alpha": 1.2,
                    "ef_indexing": 200,
                    "topk": 10,
                    "beam_width": 16,
                    "efs": list(EFS),
                },
                "threads": "build=48; search=1; NUMA node0",
                "rounds": 5,
                "build_duration_s": build["duration_s"],
                "build_peak_rss_kb": build["peak_rss_kb"],
                "search_peak_rss_kb": max(row["peak_rss_kb"] for row in search_timings),
                "search_process_duration_s_median": statistics.median(row["duration_s"] for row in search_timings),
                "qps_recall_curve": curve,
                "artifact": f"{args.tmp_root}/output/data/deep1m",
                "artifact_size_bytes": sum(sizes.values()),
                "artifact_manifest_sha256": manifest_hash,
                "artifact_files": {
                    name: {"size_bytes": sizes[name], "sha256": hashes[name]} for name in sorted(hashes)
                },
            }
        ],
        "skipped": [
            "gist1m: omitted because deep1M satisfied the required dataset and the optional build was unnecessary for the two-hour budget."
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="g03")
    parser.add_argument("--data-root", default="/md1/huangliang/data")
    parser.add_argument("--tmp-root", default="/md1/huangliang/tmp/perf-baseline/14ad5369-laser")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/baselines/baseline-14ad5369-g03-laser.json"))
    parser.add_argument("--collect-existing", action="store_true")
    parser.add_argument("--worker-prepare", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_prepare:
        prepare_worker(args)
        return
    repo = Path(__file__).resolve().parents[2]
    if not args.collect_existing:
        execute(args, repo)
    result = collect(args, repo)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
