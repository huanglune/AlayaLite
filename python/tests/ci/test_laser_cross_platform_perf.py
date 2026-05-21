# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Static/unit checks for the LASER cross-platform perf workflow helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "laser-cross-platform-perf.yaml"
SCRIPT = ROOT / "scripts" / "laser_perf" / "laser_cross_platform_perf.py"
ALIGNED_FILE_READER = ROOT / "include" / "index" / "graph" / "laser" / "utils" / "aligned_file_reader.hpp"
LASER_IO_HEADER = ROOT / "include" / "index" / "graph" / "laser" / "utils" / "io.hpp"
LASER_MEMORY_HEADER = ROOT / "include" / "index" / "graph" / "laser" / "utils" / "memory.hpp"
LASER_ROTATOR_HEADER = ROOT / "include" / "index" / "graph" / "laser" / "utils" / "rotator.hpp"
PLATFORM_HEADER = ROOT / "include" / "utils" / "platform.hpp"
PLATFORM_FS_HEADER = ROOT / "include" / "utils" / "platform_fs.hpp"
TOOLS_HEADER = ROOT / "include" / "index" / "graph" / "laser" / "utils" / "tools.hpp"


def _load_helper():
    spec = importlib.util.spec_from_file_location("laser_cross_platform_perf", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_laser_perf_workflow_is_manual_macos_first_and_artifacted() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))

    assert "workflow_dispatch" in triggers
    assert "pull_request" in triggers
    assert "schedule" in triggers
    assert "push" not in triggers
    benchmark = workflow["jobs"]["benchmark"]
    matrix = benchmark["strategy"]["matrix"]["include"]
    enabled_labels = {entry["label"] for entry in matrix if entry.get("enabled", True)}
    assert {
        "linux-libaio-x86_64",
        "macos-threadpool-arm64",
        "macos-threadpool-x86_64",
    }.issubset(enabled_labels)
    backends_by_label = {entry["label"]: entry["backend"] for entry in matrix}
    assert backends_by_label["linux-libaio-x86_64"] == "libaio"
    assert backends_by_label["macos-threadpool-arm64"] == "threadpool"
    assert backends_by_label["macos-threadpool-x86_64"] == "threadpool"

    steps = benchmark["steps"]
    run_blocks = "\n".join(step.get("run", "") for step in steps)
    assert "brew install libomp" in run_blocks
    assert "libaio-dev" in run_blocks
    assert "laser_cross_platform_perf.py run-benchmark" in run_blocks

    upload = next(step for step in steps if step.get("uses") == "actions/upload-artifact@v4")
    assert upload["with"]["name"] == "laser-perf-${{ matrix.label }}"
    assert "artifacts/laser-perf-${{ matrix.label }}.json" in upload["with"]["path"]
    assert "results/laser_*" in upload["with"]["path"]

    aggregate_step = next(
        step for step in workflow["jobs"]["aggregate-summary"]["steps"] if "aggregate-summary" in step.get("run", "")
    )
    assert "--baseline-label linux-libaio-x86_64" in aggregate_step["run"]


def test_laser_portable_headers_avoid_macos_compile_traps() -> None:
    aligned_reader = ALIGNED_FILE_READER.read_text(encoding="utf-8")
    laser_io = LASER_IO_HEADER.read_text(encoding="utf-8")
    laser_memory = LASER_MEMORY_HEADER.read_text(encoding="utf-8")
    laser_rotator = LASER_ROTATOR_HEADER.read_text(encoding="utf-8")
    platform = PLATFORM_HEADER.read_text(encoding="utf-8")
    platform_fs = PLATFORM_FS_HEADER.read_text(encoding="utf-8")
    tools = TOOLS_HEADER.read_text(encoding="utf-8")

    assert "#include <malloc.h>" not in aligned_reader
    assert "file_size_or" in platform_fs
    assert '#include "utils/platform_fs.hpp"' in laser_io
    assert "stat64" not in laser_io
    assert "alaya_aligned_alloc_impl" in platform
    assert '#include "utils/platform.hpp"' in laser_memory
    assert "std::aligned_alloc" not in laser_memory
    assert "propagate_on_container_move_assignment" in laser_memory
    assert "is_always_equal" in laser_memory
    assert "operator==" in laser_memory
    assert '#include "utils/platform.hpp"' in laser_rotator
    assert "#if defined(ALAYA_ARCH_X86)" in laser_rotator
    assert "third_party/ffht/fht_avx.hpp" in laser_rotator
    assert "fht_float_portable" in laser_rotator
    assert "static_cast<std::mt19937::result_type>" in tools


def test_laser_perf_summary_and_aggregate_lines(tmp_path: Path) -> None:
    helper = _load_helper()

    def _build_result(label: str, platform_info: dict, backend: str, disk_qps: float) -> dict:
        return {
            "label": label,
            "platform": platform_info,
            "backend": backend,
            "dataset": {
                "distribution": "synthetic_gmm_l2norm",
                "n": 256,
                "dim": 128,
                "main_dim": 128,
                "seed": 42,
                "n_clusters": 8,
                "cluster_std": 0.35,
            },
            "params": {
                "queries": 8,
                "top_k": 10,
                "ef": 64,
                "beam_width": 4,
                "rounds": 2,
                "build_threads": 0,
                "query_threads": 1,
            },
            "benchmark": {
                "median_native_qps": 2 * disk_qps,
                "median_disk_laser_qps": disk_qps,
                "median_qps_ratio": 0.5,
                "median_recall_delta": 0.0,
                "max_abs_recall_delta": 0.01,
            },
            "round_results": [
                {
                    "round": 1,
                    "native_qps": 2 * disk_qps,
                    "disk_laser_qps": disk_qps,
                    "qps_ratio": 0.5,
                    "recall_delta": 0.0,
                },
            ],
        }

    macos_result = _build_result(
        "macos-threadpool-arm64",
        {"system": "Darwin", "machine": "arm64", "python": "3.11.0"},
        "threadpool",
        60.0,
    )
    linux_result = _build_result(
        "linux-libaio-x86_64",
        {"system": "Linux", "machine": "x86_64", "python": "3.11.0"},
        "libaio",
        120.0,
    )

    single_lines = helper._single_summary_lines(macos_result)  # pylint: disable=protected-access
    assert "## LASER Cross-platform Benchmark (Darwin / arm64)" in single_lines
    assert "| Backend | `threadpool` |" in single_lines
    assert any("Median disk_laser QPS" in line and "60.000" in line for line in single_lines)
    assert any("Max abs recall delta" in line and "0.0100" in line for line in single_lines)

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    macos_path = artifact_dir / "laser-perf-macos-threadpool-arm64.json"
    linux_path = artifact_dir / "laser-perf-linux-libaio-x86_64.json"
    macos_path.write_text(json.dumps(macos_result), encoding="utf-8")
    linux_path.write_text(json.dumps(linux_result), encoding="utf-8")
    result_paths = helper._aggregate_result_paths(tmp_path)  # pylint: disable=protected-access
    assert set(result_paths) == {macos_path, linux_path}
    aggregate_lines = helper._aggregate_summary_lines(  # pylint: disable=protected-access
        result_paths,
        ["linux-libaio-x86_64", "macos-threadpool-arm64", "macos-threadpool-x86_64"],
        "linux-libaio-x86_64",
    )
    assert "## LASER Cross-platform Benchmark Summary" in aggregate_lines
    assert any("`linux-libaio-x86_64`" in line and "`libaio`" in line for line in aggregate_lines)
    assert any("`macos-threadpool-arm64`" in line and "`threadpool`" in line for line in aggregate_lines)
    assert any("`macos-threadpool-x86_64`" in line for line in aggregate_lines)
    assert any("vs libaio baseline" in line for line in aggregate_lines)
    # macos disk_qps 60.0 vs linux baseline 120.0 → ratio 0.5
    assert any("0.500" in line and "`macos-threadpool-arm64`" in line for line in aggregate_lines)
