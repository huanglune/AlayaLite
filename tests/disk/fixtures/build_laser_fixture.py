#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Build the deterministic LASER segment fixture used by disk tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

# Pin numerical-library worker pools before importing numpy/faiss. The native
# builders have their own explicit thread-count arguments below.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

import numpy as np  # noqa: E402

DEFAULT_COUNT = 2048
DEFAULT_DIM = 128
DEFAULT_R = 64
DEFAULT_SEED = 42
DEFAULT_PREFIX = "dsqg_seg_00000001"
DEFAULT_MEDOIDS = 64
STAMP_NAME = ".laser_fixture_provenance.json"

_VAMANA_HEADER_BYTES = 24
_LASER_SECTOR_BYTES = 4096


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_build_tree_package(extension: Path, stage_root: Path) -> Path:
    stage_pkg = stage_root / "alayalite"
    src_pkg = _repo_root() / "python" / "src" / "alayalite"
    shutil.copytree(
        src_pkg,
        stage_pkg,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "_alayalitepy*.so"),
    )
    staged_extension = stage_pkg / extension.name
    staged_extension.symlink_to(extension)
    if str(stage_root) not in sys.path:
        sys.path.insert(0, str(stage_root))
    return staged_extension


def _load_build_tree_extension(extension: Path, stage_root: Path) -> None:
    extension = extension.resolve(strict=True)
    if extension.suffix != ".so":
        _die(f"--extension must name a build-tree .so: {extension}")
    staged_extension = _stage_build_tree_package(extension, stage_root)
    # Some Python launchers preload the installed package. Purge it after the
    # build-tree stage has been placed first so provenance cannot depend on
    # interpreter startup state.
    for module_name in tuple(sys.modules):
        if module_name == "alayalite" or module_name.startswith("alayalite."):
            del sys.modules[module_name]
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not type(finder).__module__.startswith("_editable_")
    ]
    try:
        from alayalite import _alayalitepy  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        _die(f"could not import the explicitly selected build-tree extension {extension}: {exc}")

    loaded = Path(_alayalitepy.__file__).resolve(strict=True)
    if loaded != extension or Path(staged_extension).resolve(strict=True) != extension:
        _die(f"extension provenance mismatch: requested={extension}, loaded={loaded}")
    print(f"[laser-fixture] loaded build-tree extension: {loaded}")


def _die(message: str) -> None:
    raise SystemExit(f"build_laser_fixture.py: {message}")


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


def _fbin_size(count: int, dim: int) -> int:
    return 8 + count * dim * 4


def _valid_fbin(path: Path, count: int, dim: int) -> bool:
    if _size(path) != _fbin_size(count, dim):
        return False
    with path.open("rb") as f:
        header = f.read(8)
    if len(header) != 8:
        return False
    got_count, got_dim = struct.unpack("<ii", header)
    return got_count == count and got_dim == dim


def _valid_pca_params(path: Path, dim: int) -> bool:
    if _size(path) != 8 + dim * 4 + dim * dim * 4:
        return False
    with path.open("rb") as f:
        header = f.read(8)
    return len(header) == 8 and struct.unpack("<Q", header)[0] == dim


def _valid_ibin(path: Path, rows: int, cols: int) -> bool:
    if _size(path) != 8 + rows * cols * 4:
        return False
    with path.open("rb") as f:
        header = f.read(8)
    if len(header) != 8:
        return False
    got_rows, got_cols = struct.unpack("<ii", header)
    return got_rows == rows and got_cols == cols


def _valid_vamana_index(path: Path, degree: int) -> bool:
    if _size(path) < _VAMANA_HEADER_BYTES:
        return False
    with path.open("rb") as f:
        header = f.read(_VAMANA_HEADER_BYTES)
    if len(header) != _VAMANA_HEADER_BYTES:
        return False
    expected_size, max_degree, _start, frozen_pts = struct.unpack("<QIIQ", header)
    return _size(path) == expected_size and max_degree == degree and frozen_pts == 0


def _valid_laser_index(path: Path, count: int, main_dim: int) -> bool:
    if _size(path) < _LASER_SECTOR_BYTES:
        return False
    with path.open("rb") as f:
        sector = f.read(_LASER_SECTOR_BYTES)
    if len(sector) != _LASER_SECTOR_BYTES:
        return False
    metas = struct.unpack("<" + "Q" * (_LASER_SECTOR_BYTES // 8), sector)
    return metas[0] == count and metas[1] == main_dim and metas[4] == 1 and metas[8] == _size(path)


def _nonempty(path: Path) -> bool:
    return path.is_file() and _size(path) > 0


def _write_fbin_atomic(path: Path, vectors: np.ndarray) -> None:
    count, dim = vectors.shape
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.array([count, dim], dtype=np.int32).tofile(f)
        vectors.astype(np.float32, copy=False).tofile(f)
    os.replace(tmp, path)


def _copy_file_atomic(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _generate_vectors(count: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(loc=0.0, scale=1.0, size=(count, dim)).astype(np.float32)
    # Keep the dynamic range bounded so brute-force truth in C++ is stable.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.maximum(norms, np.float32(1e-6))
    return vectors


def _ensure_input_vectors(path: Path, count: int, dim: int, seed: int) -> None:
    if _valid_fbin(path, count, dim):
        print(f"[laser-fixture] input vectors valid, skipping: {path}")
        return
    print(f"[laser-fixture] writing deterministic input vectors: {path}")
    _write_fbin_atomic(path, _generate_vectors(count, dim, seed))


def _ensure_optional_imports() -> tuple[object, object, object]:
    try:
        from alayalite.laser._io import read_fbin  # pylint: disable=import-outside-toplevel
        from alayalite.laser._medoid import (  # pylint: disable=import-outside-toplevel
            generate_and_save_medoids,
        )
        from alayalite.laser._pca import (  # pylint: disable=import-outside-toplevel
            fit_incremental_pca,
            pca_transform_and_save,
            save_pca_params,
        )
    except ImportError as exc:
        _die(
            "optional LASER sidecar generation requires numpy, sklearn, faiss, "
            "and the alayalite.laser Python helpers. Re-run with "
            f"`--no-optional-sidecars` for a required-artifacts-only fixture: {exc}"
        )
    try:
        import faiss  # pylint: disable=import-outside-toplevel

        faiss.omp_set_num_threads(1)
    except (ImportError, AttributeError) as exc:
        _die(f"could not pin FAISS to one thread: {exc}")
    return (
        read_fbin,
        generate_and_save_medoids,
        (fit_incremental_pca, pca_transform_and_save, save_pca_params),
    )


def _ensure_pca_artifacts(
    input_path: Path,
    pca_base_path: Path,
    pca_params_path: Path,
    count: int,
    dim: int,
    seed: int,
    optional_sidecars: bool,
) -> None:
    if _valid_fbin(pca_base_path, count, dim) and (not optional_sidecars or _valid_pca_params(pca_params_path, dim)):
        print(f"[laser-fixture] PCA base valid, skipping: {pca_base_path}")
        return

    if not optional_sidecars:
        print(f"[laser-fixture] copying raw input as QGBuilder base: {pca_base_path}")
        _copy_file_atomic(input_path, pca_base_path)
        return

    read_fbin, _generate_medoids, pca_funcs = _ensure_optional_imports()
    fit_incremental_pca, pca_transform_and_save, save_pca_params = pca_funcs

    print(f"[laser-fixture] fitting full-dimension PCA with seed={seed}")
    vectors = np.asarray(read_fbin(str(input_path), use_mmap=False), dtype=np.float32)
    pca = fit_incremental_pca(vectors, n_components=dim, batch_size=count)
    save_pca_params(pca, str(pca_params_path))
    pca_transform_and_save(vectors, pca, str(pca_base_path), chunk_size=count)

    if not _valid_fbin(pca_base_path, count, dim):
        _die(f"PCA base did not validate after generation: {pca_base_path}")
    if not _valid_pca_params(pca_params_path, dim):
        _die(f"PCA params did not validate after generation: {pca_params_path}")


def _ensure_medoids(
    pca_base_path: Path,
    medoid_indices_path: Path,
    medoid_vectors_path: Path,
    medoids: int,
    dim: int,
    seed: int,
    optional_sidecars: bool,
) -> None:
    if not optional_sidecars:
        return
    if _valid_ibin(medoid_indices_path, medoids, 1) and _valid_fbin(medoid_vectors_path, medoids, dim):
        print("[laser-fixture] medoid sidecars valid, skipping")
        return

    _read_fbin, generate_and_save_medoids, _pca_funcs = _ensure_optional_imports()
    print(f"[laser-fixture] generating {medoids} medoids with seed={seed}")
    generate_and_save_medoids(
        str(pca_base_path),
        str(medoid_indices_path),
        str(medoid_vectors_path),
        medoids,
        seed=seed,
    )

    if not _valid_ibin(medoid_indices_path, medoids, 1):
        _die(f"medoid indices did not validate after generation: {medoid_indices_path}")
    if not _valid_fbin(medoid_vectors_path, medoids, dim):
        _die(f"medoid vectors did not validate after generation: {medoid_vectors_path}")


def _ensure_vamana_graph(
    native_builder: Path,
    pca_base_path: Path,
    graph_path: Path,
    degree: int,
    seed: int,
    build_l: int,
    alpha: float,
    build_threads: int,
    dram_budget_gb: float,
) -> None:
    if _valid_vamana_index(graph_path, degree):
        print(f"[laser-fixture] Vamana graph valid, skipping: {graph_path}")
        return
    print(f"[laser-fixture] building Vamana graph: {graph_path}")
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(native_builder),
            "vamana",
            str(pca_base_path),
            str(graph_path),
            str(degree),
            str(build_l),
            str(alpha),
            str(seed),
            str(build_threads),
            str(dram_budget_gb),
        ],
        check=True,
    )
    if not _valid_vamana_index(graph_path, degree):
        _die(f"Vamana graph did not validate after generation: {graph_path}")


def _laser_required_paths(output_dir: Path, prefix: str, degree: int, main_dim: int) -> dict[str, Path]:
    index = output_dir / f"{prefix}_R{degree}_MD{main_dim}.index"
    return {
        "index": index,
        "rotator": Path(str(index) + "_rotator"),
        "cache_ids": Path(str(index) + "_cache_ids"),
        "cache_nodes": Path(str(index) + "_cache_nodes"),
    }


def _laser_required_valid(paths: dict[str, Path], count: int, main_dim: int) -> bool:
    return (
        _valid_laser_index(paths["index"], count, main_dim)
        and _nonempty(paths["rotator"])
        and _nonempty(paths["cache_ids"])
        and _nonempty(paths["cache_nodes"])
    )


def _ensure_laser_artifacts(
    native_builder: Path,
    vamana_graph_path: Path,
    output_dir: Path,
    prefix: str,
    count: int,
    dim: int,
    main_dim: int,
    degree: int,
    seed: int,
    ef_indexing: int,
    build_threads: int,
) -> None:
    required = _laser_required_paths(output_dir, prefix, degree, main_dim)
    if _laser_required_valid(required, count, main_dim):
        print("[laser-fixture] native LASER artifacts valid, skipping")
        return

    print(f"[laser-fixture] building native LASER artifacts with prefix={prefix}")
    subprocess.run(
        [
            str(native_builder),
            "laser",
            str(vamana_graph_path),
            str(output_dir / prefix),
            str(count),
            str(dim),
            str(main_dim),
            str(degree),
            str(seed),
            str(ef_indexing),
            str(build_threads),
        ],
        check=True,
    )
    if not _laser_required_valid(required, count, main_dim):
        missing = [name for name, path in required.items() if not _nonempty(path)]
        _die(
            f"native LASER artifacts did not validate after generation: invalid={missing}, prefix={output_dir / prefix}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--extension",
        type=Path,
        required=True,
        help="Exact build-tree _alayalitepy shared object to load.",
    )
    parser.add_argument(
        "--native-builder",
        type=Path,
        required=True,
        help="Exact build-tree laser_fixture_builder executable to use.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate even when provenance matches."
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--R", type=int, default=DEFAULT_R)
    parser.add_argument("--main-dim", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--medoids", type=int, default=DEFAULT_MEDOIDS)
    parser.add_argument("--build-L", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--build-threads", type=int, default=1)
    parser.add_argument("--ef-indexing", type=int, default=200)
    parser.add_argument("--dram-budget-gb", type=float, default=1.0)
    parser.add_argument(
        "--optional-sidecars",
        dest="optional_sidecars",
        action="store_true",
        default=True,
        help="Generate PCA and medoid sidecars (default).",
    )
    parser.add_argument(
        "--no-optional-sidecars",
        dest="optional_sidecars",
        action="store_false",
        help="Generate only the required native LASER artifacts.",
    )
    return parser.parse_args()


def _provenance(
    args: argparse.Namespace, extension: Path, native_builder: Path, main_dim: int
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "extension": {
            "filename": extension.name,
            "sha256": _sha256(extension),
        },
        "native_builder": {
            "filename": native_builder.name,
            "sha256": _sha256(native_builder),
        },
        "parameters": {
            "prefix": args.prefix,
            "count": args.count,
            "dim": args.dim,
            "R": args.R,
            "main_dim": main_dim,
            "seed": args.seed,
            "medoids": args.medoids,
            "build_L": args.build_L,
            "alpha": args.alpha,
            "build_threads": args.build_threads,
            "ef_indexing": args.ef_indexing,
            "dram_budget_gb": args.dram_budget_gb,
            "optional_sidecars": args.optional_sidecars,
            "faiss_blas_omp_threads": 1,
        },
    }


def _fixture_valid(output_dir: Path, args: argparse.Namespace, main_dim: int) -> bool:
    prefix = args.prefix
    required = _laser_required_paths(output_dir, prefix, args.R, main_dim)
    valid = (
        _valid_fbin(output_dir / f"{prefix}_input.fbin", args.count, args.dim)
        and _valid_fbin(output_dir / f"{prefix}_pca_base.fbin", args.count, args.dim)
        and _valid_vamana_index(output_dir / f"{prefix}_vamana_graph.index", args.R)
        and _laser_required_valid(required, args.count, main_dim)
    )
    if args.optional_sidecars:
        valid = (
            valid
            and _valid_pca_params(output_dir / f"{prefix}_pca.bin", args.dim)
            and _valid_ibin(output_dir / f"{prefix}_medoids_indices", args.medoids, 1)
            and _valid_fbin(output_dir / f"{prefix}_medoids", args.medoids, args.dim)
        )
    return valid


def _stamp_matches(output_dir: Path, expected: dict[str, object]) -> bool:
    try:
        actual = json.loads((output_dir / STAMP_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected


def _replace_fixture(output_dir: Path, generated_dir: Path) -> None:
    backup = output_dir.with_name(f".{output_dir.name}.old-{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    if output_dir.exists():
        os.replace(output_dir, backup)
    try:
        os.replace(generated_dir, output_dir)
    except BaseException:
        if backup.exists() and not output_dir.exists():
            os.replace(backup, output_dir)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def main() -> int:
    args = _parse_args()
    if args.count != DEFAULT_COUNT or args.dim != DEFAULT_DIM or args.R != DEFAULT_R:
        print(
            "[laser-fixture] non-default dimensions requested; the disk test fixture "
            "contract is count=2048, dim=128, R=64"
        )
    if args.dim < 128 or args.dim & (args.dim - 1):
        _die("dim must be a power of two and at least 128 for the current LASER port")
    if args.medoids <= 0 or args.medoids > int(args.count * 0.10):
        _die("--medoids must be > 0 and <= floor(count * 0.10) for faiss IVF training")

    output_dir = args.output_dir.resolve()
    extension = args.extension.resolve(strict=True)
    native_builder = args.native_builder.resolve(strict=True)
    if not native_builder.is_file():
        _die(f"--native-builder must name a regular file: {native_builder}")
    main_dim = args.main_dim or args.dim
    if main_dim != args.dim:
        _die("v1 fixture generation requires --main-dim to equal --dim")

    expected_stamp = _provenance(args, extension, native_builder, main_dim)
    python_stage = tempfile.TemporaryDirectory(prefix="alaya-laser-python-stage-")
    _load_build_tree_extension(extension, Path(python_stage.name))
    if (
        not args.force
        and _stamp_matches(output_dir, expected_stamp)
        and _fixture_valid(output_dir, args, main_dim)
    ):
        print(f"[laser-fixture] provenance matches, reusing: {output_dir}")
        return 0

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    generated_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))

    prefix = args.prefix
    input_path = generated_dir / f"{prefix}_input.fbin"
    pca_base_path = generated_dir / f"{prefix}_pca_base.fbin"
    pca_params_path = generated_dir / f"{prefix}_pca.bin"
    medoid_indices_path = generated_dir / f"{prefix}_medoids_indices"
    medoid_vectors_path = generated_dir / f"{prefix}_medoids"
    vamana_graph_path = generated_dir / f"{prefix}_vamana_graph.index"

    print(f"[laser-fixture] regenerating snapshot: {output_dir}")
    _ensure_input_vectors(input_path, args.count, args.dim, args.seed)
    _ensure_pca_artifacts(
        input_path,
        pca_base_path,
        pca_params_path,
        args.count,
        args.dim,
        args.seed,
        args.optional_sidecars,
    )
    _ensure_medoids(
        pca_base_path,
        medoid_indices_path,
        medoid_vectors_path,
        args.medoids,
        args.dim,
        args.seed,
        args.optional_sidecars,
    )
    _ensure_vamana_graph(
        native_builder,
        pca_base_path,
        vamana_graph_path,
        args.R,
        args.seed,
        args.build_L,
        args.alpha,
        args.build_threads,
        args.dram_budget_gb,
    )
    _ensure_laser_artifacts(
        native_builder,
        vamana_graph_path,
        generated_dir,
        prefix,
        args.count,
        args.dim,
        main_dim,
        args.R,
        args.seed,
        args.ef_indexing,
        args.build_threads,
    )
    (generated_dir / STAMP_NAME).write_text(
        json.dumps(expected_stamp, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _replace_fixture(output_dir, generated_dir)
    print(f"[laser-fixture] ready: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
