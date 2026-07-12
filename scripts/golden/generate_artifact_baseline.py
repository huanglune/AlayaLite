#!/usr/bin/env python3
"""Generate or verify deterministic pre-refactor persistence artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = ROOT / "tests/golden/artifact-baseline.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kv(path: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            result[key] = int(value) if value.isdigit() else value
    return result


def _header(path: Path) -> dict[str, object]:
    raw = path.read_bytes()[:64]
    parsed: dict[str, object] = {"first_32_hex": raw[:32].hex()}
    if len(raw) >= 16:
        parsed["u32le_0_3"] = list(struct.unpack("<4I", raw[:16]))
    return parsed


def _inventory(root: Path) -> dict[str, object]:
    files: dict[str, object] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != ".lock"):
        rel = path.relative_to(root).as_posix()
        entry: dict[str, object] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        if path.name.endswith("manifest.txt"):
            entry["fields"] = _kv(path)
        elif path.suffix in {".bin", ".index", ".graph", ".data", ".quant"}:
            entry["header"] = _header(path)
        files[rel] = entry
    return {"files": files}


def _vectors(rows: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(20260712)
    return rng.standard_normal((rows, dim)).astype(np.float32)


def _generate_python_artifacts(out: Path) -> None:
    from alayalite import DiskCollection, Index, MetricType
    from alayalite.schema import IndexParams

    vectors = _vectors(64, 8)
    for quantization in ("none", "sq8"):
        memory = out / f"memory_hnsw_{quantization}"
        index = Index("golden", IndexParams(capacity=80, max_nbrs=8,
                                             quantization_type=quantization))
        index.fit(vectors, ef_construction=32, num_threads=1)
        index.save(memory)
        index.close()

    # NSG's retained NN-Descent kernel requires more than 64 rows. Keep the
    # pre-existing HNSW corpus untouched and add deterministic per-engine
    # families with their own fixed shape.
    graph_vectors = _vectors(80, 8)
    for engine in ("nsg", "fusion"):
        for quantization in ("none", "sq8"):
            memory = out / f"memory_{engine}_{quantization}"
            index = Index(
                "golden",
                IndexParams(
                    index_type=engine,
                    capacity=96,
                    max_nbrs=8,
                    quantization_type=quantization,
                ),
            )
            index.fit(graph_vectors, ef_construction=32, num_threads=1)
            index.save(memory)
            index.close()

    ids = np.arange(1000, 1064, dtype=np.uint64)
    for engine in ("disk_flat", "disk_vamana"):
        target = out / engine
        kwargs = {}
        if engine == "disk_vamana":
            kwargs = {
                "vamana_R": 8,
                "vamana_L": 24,
                "vamana_alpha": 1.2,
                "vamana_seed": 424242,
                "vamana_num_threads": 1,
            }
        collection = DiskCollection(path=str(target), dim=8, metric=MetricType.L2,
                                    index_type=engine, **kwargs)
        collection.add(vectors, ids)
        collection.flush()
        del collection


def _copy_laser_fixture(out: Path, build_dir: Path) -> bool:
    source = build_dir / "tests/disk/fixtures/laser_segment"
    if not source.is_dir():
        return False
    target = out / "laser_fixture"
    target.mkdir()
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)
    return True


def generate(build_dir: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="alaya-artifact-golden-") as temp:
        out = Path(temp)
        _generate_python_artifacts(out)
        diskann_generator = build_dir / "tests/golden/artifact_diskann_generator"
        if not diskann_generator.is_file():
            raise RuntimeError(f"build the artifact_diskann_generator target first: {diskann_generator}")
        subprocess.run([str(diskann_generator), str(out / "diskann")], check=True)
        laser_present = _copy_laser_fixture(out, build_dir)
        artifacts = {
            name: _inventory(path)
            for name, path in sorted((p.name, p) for p in out.iterdir() if p.is_dir())
        }
        return {
            "schema_version": 1,
            "seed": 20260712,
            "generator": "scripts/golden/generate_artifact_baseline.py",
            "laser_fixture_present": laser_present,
            "artifacts": artifacts,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/Release")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--write", action="store_true", help="replace the checked-in baseline")
    args = parser.parse_args()
    actual = generate(args.build_dir.resolve())
    encoded = json.dumps(actual, indent=2, sort_keys=True) + "\n"
    if args.write:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(encoded, encoding="utf-8")
        print(f"wrote {args.baseline}")
        return 0
    expected = json.loads(args.baseline.read_text(encoding="utf-8"))
    if actual != expected:
        print("artifact baseline mismatch; rerun with --write and review the format diff", file=sys.stderr)
        return 1
    print(f"artifact baseline matches {args.baseline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
