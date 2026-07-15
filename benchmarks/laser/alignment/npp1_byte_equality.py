# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Rebuild an npp==1 LASER QG from archived inputs and compare to baseline.

Baseline provenance:
- The original tier-A baseline (1f52723a...) was produced on a pre-merge LASER
  development branch on 2026-04-23, before the module landed on `main` in
  commit 2c83edb (2026-05-09). Review-time fixups between the dev branch and
  the merge commit (cpplint cleanup, VLA scratch buffer rewrite) shifted the
  factor-region bytes by a small float drift, so the pre-merge baseline is
  not byte-comparable to anything built from current `main`.
- The active baseline (f1554f3e...) is regenerated from current `main` at
  commit 589152c (fix-laser-low-dim-page-layout) with the same archived
  pre-QG inputs (PCA, medoids, Vamana). It pins both the source tree and the
  build toolchain at the regeneration moment; any future drift caught by
  this harness reflects a real change in current `main`.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import struct
from pathlib import Path

DEFAULT_NAME = "synth_20k_768d"
DEFAULT_PREFIX = f"dsqg_{DEFAULT_NAME}"
DEFAULT_BASELINE_DIR = Path("")
DEFAULT_BASELINE = DEFAULT_BASELINE_DIR / f"{DEFAULT_PREFIX}_R64_MD256.post_layout_fix.index"
DEFAULT_BASELINE_SHA = "f1554f3ebaa9d986f975878354749c287ba777d4f6aa82265d565e32c4c16f32"  # pragma: allowlist secret  # noqa: E501 commit 589152c
DEFAULT_VAMANA = Path("")
DEFAULT_VAMANA_SHA = "133298cbded6774b36ec32a5fc70a1d2036e71f3e693f3e618a817e84fd6be75"  # pragma: allowlist secret
DEFAULT_OUTPUT_ROOT = Path("")
K_SECTOR_LEN = 4096
PRE_QG_SUFFIXES = ("pca.bin", "pca_base.fbin", "medoids_indices", "medoids")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_index_header(path: Path) -> dict[str, int]:
    with path.open("rb") as f:
        raw = f.read(K_SECTOR_LEN)
    if len(raw) != K_SECTOR_LEN:
        raise ValueError(f"{path}: truncated metadata sector")
    metas = struct.unpack("<" + "Q" * (K_SECTOR_LEN // 8), raw)
    return {
        "num_points": int(metas[0]),
        "dimension": int(metas[1]),
        "entry_point": int(metas[2]),
        "node_len": int(metas[3]),
        "node_per_page": int(metas[4]),
        "file_size": int(metas[8]),
    }


def read_fbin_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError(f"{path}: truncated fbin header")
    n, dim = struct.unpack("<ii", raw)
    if n <= 0 or dim <= 0:
        raise ValueError(f"{path}: invalid fbin shape ({n}, {dim})")
    return int(n), int(dim)


def copy_pre_qg_artifacts(baseline_dir: Path, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in PRE_QG_SUFFIXES:
        source = baseline_dir / f"{prefix}_{suffix}"
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copyfile(source, output_dir / source.name)


def assert_pre_qg_equal(baseline_dir: Path, rebuilt_dir: Path, prefix: str) -> None:
    drift = []
    for suffix in PRE_QG_SUFFIXES:
        filename = f"{prefix}_{suffix}"
        baseline = baseline_dir / filename
        rebuilt = rebuilt_dir / filename
        baseline_sha = sha256_of(baseline)
        rebuilt_sha = sha256_of(rebuilt)
        if baseline_sha != rebuilt_sha:
            drift.append(f"{filename}: baseline={baseline_sha} rebuilt={rebuilt_sha}")
    if drift:
        details = "\n  ".join(drift)
        raise RuntimeError(f"pre-QG artifact copy drifted unexpectedly:\n  {details}")


def build_qg_from_archived_inputs(
    *,
    baseline_dir: Path,
    vamana: Path,
    output_root: Path,
    name: str,
    prefix: str,
    degree: int,
    main_dim: int,
    ef_indexing: int,
    num_threads: int,
    rotator_seed: int,
    skip_existing: bool,
) -> Path:
    from alayalite import laser  # pylint: disable=import-outside-toplevel

    output_dir = output_root / "data" / name
    index_path = output_dir / f"{prefix}_R{degree}_MD{main_dim}.index"
    if not skip_existing and output_dir.exists():
        shutil.rmtree(output_dir)
    if not (skip_existing and index_path.exists()):
        copy_pre_qg_artifacts(baseline_dir, output_dir, prefix)
        n, raw_dim = read_fbin_header(output_dir / f"{prefix}_pca_base.fbin")
        raw_index_cls = laser.RawIndex
        if raw_index_cls is None:
            raise RuntimeError("alayalite.laser.RawIndex is unavailable; build the native extension first")
        raw = raw_index_cls(
            index_type="QG",
            metric="l2",
            num_elements=n,
            main_dimension=main_dim,
            dimension=raw_dim,
            degree_bound=degree,
            rotator_seed=rotator_seed,
            rotator_dump_path=str(output_dir / f"{prefix}_rotator_signs.bin"),
        )
        raw.build_index(str(vamana), str(output_dir / prefix), EF=ef_indexing, num_thread=num_threads)

    assert_pre_qg_equal(baseline_dir, output_dir, prefix)
    return index_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--baseline-sha", default=DEFAULT_BASELINE_SHA)
    parser.add_argument("--vamana", type=Path, default=DEFAULT_VAMANA)
    parser.add_argument("--vamana-sha", default=DEFAULT_VAMANA_SHA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--degree", type=int, default=64)
    parser.add_argument("--main-dim", type=int, default=256)
    parser.add_argument("--ef-indexing", type=int, default=200)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--rotator-seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    for path in (args.baseline_dir, args.baseline, args.vamana):
        if not path.exists():
            raise FileNotFoundError(path)

    baseline_sha = sha256_of(args.baseline)
    if baseline_sha != args.baseline_sha:
        raise RuntimeError(f"baseline sha mismatch: expected {args.baseline_sha}, got {baseline_sha} ({args.baseline})")
    vamana_sha = sha256_of(args.vamana)
    if vamana_sha != args.vamana_sha:
        raise RuntimeError(f"vamana sha mismatch: expected {args.vamana_sha}, got {vamana_sha} ({args.vamana})")

    rebuilt = build_qg_from_archived_inputs(
        baseline_dir=args.baseline_dir,
        vamana=args.vamana,
        output_root=args.output_root,
        name=args.name,
        prefix=args.prefix,
        degree=args.degree,
        main_dim=args.main_dim,
        ef_indexing=args.ef_indexing,
        num_threads=args.num_threads,
        rotator_seed=args.rotator_seed,
        skip_existing=args.skip_existing,
    )
    rebuilt_sha = sha256_of(rebuilt)
    header = parse_index_header(rebuilt)
    actual_size = rebuilt.stat().st_size

    if header["node_per_page"] != 1:
        raise RuntimeError(f"expected node_per_page == 1, got {header['node_per_page']}")
    if header["file_size"] != actual_size:
        raise RuntimeError(f"metadata/file size mismatch: metas[8]={header['file_size']} actual={actual_size}")
    if rebuilt_sha != baseline_sha:
        raise RuntimeError(
            "npp==1 byte-equality failed after controlling archived pre-QG artifacts "
            "and the canonical Vamana graph:\n"
            f"  rebuilt:  {rebuilt} {rebuilt_sha}\n"
            f"  baseline: {args.baseline} {baseline_sha}"
        )

    print("npp==1 byte-equality PASS")
    print(f"  rebuilt:  {rebuilt}")
    print(f"  baseline: {args.baseline}")
    print(f"  sha256:   {rebuilt_sha}")
    print(f"  geometry: node_len={header['node_len']} node_per_page={header['node_per_page']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
