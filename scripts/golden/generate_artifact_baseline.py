#!/usr/bin/env python3
"""Generate or verify deterministic pre-refactor persistence artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    # The provenance stamp gates fixture reuse, embeds an intentionally volatile .so hash,
    # and is metadata rather than artifact payload.
    excluded_names = {".lock", ".laser_fixture_provenance.json"}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in excluded_names):
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


def _generate_retained_v1_artifacts(out: Path, build_dir: Path) -> None:
    generator = build_dir / "tests/golden/artifact_legacy_v1_generator"
    if not generator.is_file():
        raise RuntimeError(f"build the artifact_legacy_v1_generator target first: {generator}")
    vectors_path = out / ".retained-v1-vectors.fbin"
    vectors = _vectors(80, 8)
    with vectors_path.open("wb") as stream:
        np.array(vectors.shape, dtype=np.int32).tofile(stream)
        vectors.tofile(stream)
    try:
        subprocess.run([str(generator), str(out), str(vectors_path)], check=True)
    finally:
        vectors_path.unlink(missing_ok=True)


def _build_tree_extension(build_dir: Path) -> Path | None:
    candidates = sorted((build_dir / "python").glob("_alayalitepy*.so"))
    if len(candidates) != 1:
        return None

    return candidates[0]


def _generate_laser_fixture(out: Path, build_dir: Path) -> bool:
    extension = _build_tree_extension(build_dir)
    if extension is None:
        print(
            f"LASER fixture omitted: expected exactly one build-tree extension under {build_dir / 'python'}",
            file=sys.stderr,
        )
        return False
    target = out / "laser_fixture"
    native_builder = build_dir / "tests/disk/laser_fixture_builder"
    if not native_builder.is_file():
        print(
            f"LASER fixture omitted: build the native fixture builder first: {native_builder}",
            file=sys.stderr,
        )
        return False
    command = [
        sys.executable,
        str(ROOT / "tests/disk/fixtures/build_laser_fixture.py"),
        "--output-dir",
        str(target),
        "--extension",
        str(extension),
        "--native-builder",
        str(native_builder),
        "--force",
        "--prefix",
        "dsqg_seg_00000001",
        "--count",
        "2048",
        "--dim",
        "128",
        "--R",
        "64",
        "--seed",
        "42",
        "--optional-sidecars",
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"LASER fixture omitted: generation unavailable: {exc}", file=sys.stderr)
        return False
    return True


def generate(build_dir: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="alaya-artifact-golden-") as temp:
        out = Path(temp)
        _generate_retained_v1_artifacts(out, build_dir)
        diskann_generator = build_dir / "tests/golden/artifact_diskann_generator"
        if not diskann_generator.is_file():
            raise RuntimeError(f"build the artifact_diskann_generator target first: {diskann_generator}")
        subprocess.run([str(diskann_generator), str(out / "diskann")], check=True)
        disk_flat_segment_generator = build_dir / "tests/golden/artifact_disk_flat_segment_generator"
        if not disk_flat_segment_generator.is_file():
            raise RuntimeError(
                "build the artifact_disk_flat_segment_generator target first: "
                f"{disk_flat_segment_generator}"
            )
        subprocess.run(
            [str(disk_flat_segment_generator), str(out / "disk_flat_segment")],
            check=True,
        )
        disk_vamana_segment_generator = (
            build_dir / "tests/golden/artifact_disk_vamana_segment_generator"
        )
        if not disk_vamana_segment_generator.is_file():
            raise RuntimeError(
                "build the artifact_disk_vamana_segment_generator target first: "
                f"{disk_vamana_segment_generator}"
            )
        subprocess.run(
            [
                str(disk_vamana_segment_generator),
                str(out / "disk_vamana_segment"),
            ],
            check=True,
        )
        memory_qg_generator = build_dir / "tests/golden/artifact_memory_qg_generator"
        if not memory_qg_generator.is_file():
            raise RuntimeError(
                f"build the artifact_memory_qg_generator target first: {memory_qg_generator}"
            )
        subprocess.run([str(memory_qg_generator), str(out / "memory_qg")], check=True)
        memory_vamana_generator = build_dir / "tests/golden/artifact_memory_vamana_generator"
        if not memory_vamana_generator.is_file():
            raise RuntimeError(
                "build the artifact_memory_vamana_generator target first: "
                f"{memory_vamana_generator}"
            )
        subprocess.run(
            [str(memory_vamana_generator), str(out / "memory_vamana")], check=True
        )
        laser_present = _generate_laser_fixture(out, build_dir)
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
