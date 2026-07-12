#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Generate the small, read-only legacy PyIndex recovery corpus.

The writer exercised here is the pre-Segment Python path: Graph/Space snapshots,
the custom framed WAL, RocksDB scalar checkpoints, snapshot manifests, and the
atomic CURRENT pointer.  A locally built ``alayalite._alayalitepy`` extension
from this checkout must be importable before running the script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = REPO_ROOT / "python" / "src"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from alayalite import Index  # noqa: E402
from alayalite.collection import Collection  # noqa: E402
from alayalite.schema import IndexParams, save_schema  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "python" / "tests" / "fixtures" / "legacy_recovery_corpus"
ROCKSDB_PLACEHOLDER = "__CORPUS_ROOT__/rocksdb"
DIM = 8
CAPACITY = 32
MAX_NBRS = 8
BASE_COUNT = 6
WAL_TRAILER_MAGIC = 0x5441494C
CORPUS_README = """# Legacy PyIndex recovery corpus

These fixtures were written by commit `ab2cb0f` before the Segment migration.
They pin the legacy Graph/Space snapshot, custom WAL, RocksDB checkpoint,
`manifest.txt`, and `CURRENT` formats. Tests always copy a case before opening
it because the legacy reader publishes a `post_recovery` snapshot after replay.

`schema.json` and scalar `data.snapshot`/`raw.data` files use
`__CORPUS_ROOT__/rocksdb` as a relocation placeholder. The reader test replaces
it only in its temporary copy. Each case's `sha256.json` lists every other file
in that case; the checksum manifest excludes itself.

After building the Python extension from this checkout, regenerate with:

```sh
PYTHONPATH=python/src .venv/bin/python scripts/generate_legacy_recovery_corpus.py
```

RocksDB embeds random DB/session identities in SST and MANIFEST files. Normal
in-place regeneration preserves the checked-in checkpoint bytes so a clean
checkout regenerates byte-for-byte. Pass `--refresh-rocksdb` only when the
scalar checkpoint contents are intentionally being replaced; then review the
new checksums and run the read-back test.
"""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _vectors(dtype, seed: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.floating):
        return rng.standard_normal((count, DIM)).astype(dtype)
    if np.issubdtype(dtype, np.signedinteger):
        return rng.integers(-40, 41, size=(count, DIM), dtype=dtype)
    return rng.integers(0, 81, size=(count, DIM), dtype=dtype)


def _params(case_dir: Path, dtype, id_type, *, has_scalar_data: bool) -> IndexParams:
    params = IndexParams(
        index_type="hnsw",
        data_type=dtype,
        id_type=id_type,
        quantization_type="none",
        metric="l2",
        capacity=CAPACITY,
        max_nbrs=MAX_NBRS,
        build_threads=1,
        rocksdb_path=str(case_dir / "rocksdb"),
        has_scalar_data=has_scalar_data,
    )
    params.fill_none_values()
    return params


def _schema(case_dir: Path, kind: str, params: IndexParams) -> None:
    save_schema(str(case_dir / "schema.json"), {"type": kind, "index": params.to_json_dict()})


def _wait_for_distinct_snapshot_id(case_dir: Path) -> None:
    """Avoid the legacy millisecond snapshot-name collision on consecutive saves."""
    current_path = case_dir / "recovery" / "CURRENT"
    if not current_path.exists():
        return
    current = current_path.read_text(encoding="utf-8").strip()
    try:
        previous_ms = int(current.removeprefix("snapshot-"))
    except ValueError:
        return
    while int(time.time() * 1000) <= previous_ms:
        time.sleep(0.001)


def _query(vector: np.ndarray, expected_ids: list[int]) -> dict[str, Any]:
    return {
        "vector": vector.tolist(),
        "topk": len(expected_ids),
        "expected_internal_ids": expected_ids,
    }


def _collection_query(vector: np.ndarray, expected_item_ids: list[str]) -> dict[str, Any]:
    return {
        "vector": vector.tolist(),
        "topk": len(expected_item_ids),
        "expected_item_ids": expected_item_ids,
    }


def _base_generation(
    name: str,
    *,
    kind: str,
    dtype,
    id_type,
    seed: int,
    scalar: bool,
    operations: list[str],
    terminal_shape: str,
) -> dict[str, Any]:
    return {
        "case": name,
        "writer": "legacy PyIndex Graph/Space + custom WAL + snapshot manifest/CURRENT",
        "kind": kind,
        "seed": seed,
        "dim": DIM,
        "base_count": BASE_COUNT if kind == "index" else None,
        "capacity": CAPACITY,
        "max_nbrs": MAX_NBRS,
        "data_type": np.dtype(dtype).name,
        "id_type": np.dtype(id_type).name,
        "quantization_type": "none",
        "index_type": "hnsw",
        "metric": "l2",
        "has_scalar_data": scalar,
        "operations": operations,
        "terminal_shape": terminal_shape,
    }


def _close_index(index: Index) -> None:
    index.close()
    gc.collect()


def _build_f32_u32_insert_clean(case_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    name = case_dir.name
    data = _vectors(np.float32, 101, BASE_COUNT + 2)
    params = _params(case_dir, np.float32, np.uint32, has_scalar_data=False)
    index = Index(name, params)
    try:
        index.fit(data[:BASE_COUNT], ef_construction=32, num_threads=1)
        assert index.insert(data[BASE_COUNT], ef=32) == BASE_COUNT
        assert index.insert(data[BASE_COUNT + 1], ef=32) == BASE_COUNT + 1
        _wait_for_distinct_snapshot_id(case_dir)
        index.save(case_dir)
        _schema(case_dir, "index", params)
    finally:
        _close_index(index)

    generation = _base_generation(
        name,
        kind="index",
        dtype=np.float32,
        id_type=np.uint32,
        seed=101,
        scalar=False,
        operations=["fit(6)", "insert(id=6)", "insert(id=7)", "manual_save/checkpoint"],
        terminal_shape="clean_snapshot",
    )
    expected = {
        "expected_count": 8,
        "queries": [_query(data[0], [0]), _query(data[7], [7])],
        "expected_applied_through_op_id": 2,
        "wal_after_writer": "absent_after_checkpoint",
        "reader_behavior": "load snapshot directly; no WAL replay",
    }
    return generation, expected


def _build_i8_u64_insert_remove_tail(case_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    name = case_dir.name
    data = _vectors(np.int8, 202, BASE_COUNT + 1)
    params = _params(case_dir, np.int8, np.uint64, has_scalar_data=False)
    index = Index(name, params)
    try:
        index.fit(data[:BASE_COUNT], ef_construction=32, num_threads=1)
        assert index.insert(data[BASE_COUNT], ef=32) == BASE_COUNT
        index.remove(2)
        _schema(case_dir, "index", params)
    finally:
        _close_index(index)

    generation = _base_generation(
        name,
        kind="index",
        dtype=np.int8,
        id_type=np.uint64,
        seed=202,
        scalar=False,
        operations=["fit(6)", "insert(id=6)", "remove(internal_id=2)"],
        terminal_shape="committed_wal_tail",
    )
    expected = {
        "expected_count": 7,
        "queries": [_query(data[0], [0]), _query(data[6], [6])],
        "expected_applied_through_op_id": 2,
        "wal_after_writer": "two committed mutations after snapshot",
        "reader_behavior": "replay insert and remove, publish post_recovery snapshot, truncate WAL",
    }
    return generation, expected


def _build_u8_u32_snapshot_insert_tail(case_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    name = case_dir.name
    data = _vectors(np.uint8, 303, BASE_COUNT + 2)
    params = _params(case_dir, np.uint8, np.uint32, has_scalar_data=False)
    index = Index(name, params)
    try:
        index.fit(data[:BASE_COUNT], ef_construction=32, num_threads=1)
        assert index.insert(data[BASE_COUNT], ef=32) == BASE_COUNT
        _wait_for_distinct_snapshot_id(case_dir)
        index.save(case_dir)
        assert index.insert(data[BASE_COUNT + 1], ef=32) == BASE_COUNT + 1
        _schema(case_dir, "index", params)
    finally:
        _close_index(index)

    generation = _base_generation(
        name,
        kind="index",
        dtype=np.uint8,
        id_type=np.uint32,
        seed=303,
        scalar=False,
        operations=["fit(6)", "insert(id=6)", "manual_save/checkpoint", "insert(id=7)"],
        terminal_shape="snapshot_plus_committed_wal_tail",
    )
    expected = {
        "expected_count": 8,
        "queries": [_query(data[0], [0]), _query(data[7], [7])],
        "expected_applied_through_op_id": 2,
        "wal_after_writer": "one committed insert after snapshot applied through op 1",
        "reader_behavior": "load snapshot at op 1, replay op 2, publish post_recovery snapshot",
    }
    return generation, expected


def _truncate_last_wal_trailer(case_dir: Path) -> dict[str, int]:
    wal_path = case_dir / "recovery" / "wal.bin"
    raw = wal_path.read_bytes()
    assert len(raw) > 4
    assert struct.unpack_from("<I", raw, len(raw) - 4)[0] == WAL_TRAILER_MAGIC
    wal_path.write_bytes(raw[:-4])
    return {"bytes_before": len(raw), "bytes_after": len(raw) - 4, "bytes_removed": 4}


def _build_f32_u64_torn_tail(case_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    name = case_dir.name
    data = _vectors(np.float32, 404, BASE_COUNT + 2)
    params = _params(case_dir, np.float32, np.uint64, has_scalar_data=False)
    index = Index(name, params)
    try:
        index.fit(data[:BASE_COUNT], ef_construction=32, num_threads=1)
        assert index.insert(data[BASE_COUNT], ef=32) == BASE_COUNT
        assert index.insert(data[BASE_COUNT + 1], ef=32) == BASE_COUNT + 1
        _schema(case_dir, "index", params)
    finally:
        _close_index(index)
    truncation = _truncate_last_wal_trailer(case_dir)

    generation = _base_generation(
        name,
        kind="index",
        dtype=np.float32,
        id_type=np.uint64,
        seed=404,
        scalar=False,
        operations=[
            "fit(6)",
            "insert(op=1,id=6)",
            "insert(op=2,id=7)",
            "truncate final COMMIT trailer",
        ],
        terminal_shape="torn_wal_tail",
    )
    generation["torn_tail_injection"] = truncation
    expected = {
        "expected_count": 7,
        "queries": [_query(data[0], [0]), _query(data[6], [6])],
        "expected_applied_through_op_id": 1,
        "wal_after_writer": "op 1 complete; op 2 PREPARE complete but COMMIT trailer torn",
        "reader_behavior": (
            "stop at the torn COMMIT, replay only op 1, publish post_recovery snapshot, "
            "and discard writer-live id 7"
        ),
    }
    return generation, expected


def _collection_items(data: np.ndarray) -> list[tuple[str, str, np.ndarray, dict[str, Any]]]:
    return [
        (f"item-{i}", f"document-{i}", data[i], {"group": "base", "version": 1, "ordinal": i})
        for i in range(len(data))
    ]


def _build_i8_u32_collection_upsert_tail(case_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    name = case_dir.name
    count = 4
    data = _vectors(np.int8, 505, count + 1)
    params = _params(case_dir, np.int8, np.uint32, has_scalar_data=True)
    collection = Collection(name, params)
    try:
        collection.insert(_collection_items(data[:count]))
        collection.upsert(
            [("item-1", "document-1-v2", data[count], {"group": "updated", "version": 2, "ordinal": 1})]
        )
        _schema(case_dir, "collection", params)
    finally:
        collection.close()
        gc.collect()

    generation = _base_generation(
        name,
        kind="collection",
        dtype=np.int8,
        id_type=np.uint32,
        seed=505,
        scalar=True,
        operations=["collection.insert(4 string IDs + documents + metadata)", "upsert(item-1)"],
        terminal_shape="committed_wal_tail",
    )
    generation["base_count"] = count
    expected = {
        "expected_count": 5,
        "expected_live_count": 4,
        "item_lookup_order": ["item-0", "item-1", "item-2", "item-3", "missing"],
        "expected_items": [
            {"id": "item-0", "document": "document-0", "metadata": {"group": "base", "version": 1, "ordinal": 0}},
            {
                "id": "item-1",
                "document": "document-1-v2",
                "metadata": {"group": "updated", "version": 2, "ordinal": 1},
            },
            {"id": "item-2", "document": "document-2", "metadata": {"group": "base", "version": 1, "ordinal": 2}},
            {"id": "item-3", "document": "document-3", "metadata": {"group": "base", "version": 1, "ordinal": 3}},
        ],
        "queries": [_collection_query(data[count], ["item-1"])],
        "expected_applied_through_op_id": 1,
        "wal_after_writer": "one committed upsert after the post-fit snapshot",
        "reader_behavior": "restore RocksDB checkpoint, replay upsert, and retain the rewritten scalar record",
    }
    return generation, expected


def _build_u8_u64_collection_delete_clean(case_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    name = case_dir.name
    count = 5
    data = _vectors(np.uint8, 606, count)
    params = _params(case_dir, np.uint8, np.uint64, has_scalar_data=True)
    collection = Collection(name, params)
    try:
        collection.insert(_collection_items(data))
        collection.delete_by_id(["item-3"])
        _wait_for_distinct_snapshot_id(case_dir)
        schema_map = collection.save(case_dir)
        save_schema(str(case_dir / "schema.json"), schema_map)
    finally:
        collection.close()
        gc.collect()

    generation = _base_generation(
        name,
        kind="collection",
        dtype=np.uint8,
        id_type=np.uint64,
        seed=606,
        scalar=True,
        operations=[
            "collection.insert(5 string IDs + documents + metadata)",
            "delete_by_id(item-3)",
            "manual_save/checkpoint",
        ],
        terminal_shape="clean_snapshot",
    )
    generation["base_count"] = count
    expected = {
        "expected_count": 5,
        "expected_live_count": 4,
        "item_lookup_order": ["item-0", "item-1", "item-2", "item-3", "item-4"],
        "expected_items": [
            {
                "id": f"item-{i}",
                "document": f"document-{i}",
                "metadata": {"group": "base", "version": 1, "ordinal": i},
            }
            for i in (0, 1, 2, 4)
        ],
        "queries": [_collection_query(data[0], ["item-0"])],
        "expected_applied_through_op_id": 1,
        "wal_after_writer": "absent_after_checkpoint",
        "reader_behavior": "load graph/data/RocksDB snapshot with item-3 already deleted",
    }
    return generation, expected


CASE_BUILDERS = {
    "f32_u32_insert_clean": _build_f32_u32_insert_clean,
    "i8_u64_insert_remove_wal": _build_i8_u64_insert_remove_tail,
    "u8_u32_snapshot_insert_wal": _build_u8_u32_snapshot_insert_tail,
    "f32_u64_torn_tail": _build_f32_u64_torn_tail,
    "i8_u32_collection_upsert_wal": _build_i8_u32_collection_upsert_tail,
    "u8_u64_collection_delete_clean": _build_u8_u64_collection_delete_clean,
}


def _normalize_schema(case_dir: Path) -> None:
    schema_path = case_dir / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["index"]["rocksdb_path"] = ROCKSDB_PLACEHOLDER
    _write_json(schema_path, schema)


def _replace_length_prefixed_path(path: Path, old_path: str, new_path: str) -> None:
    if not path.exists():
        return
    raw = path.read_bytes()
    old = old_path.encode()
    new = new_path.encode()
    assert raw.count(old) == 1, f"expected one persisted RocksDB path in {path}"
    path_offset = raw.index(old)
    word_width = struct.calcsize("P")
    length_offset = path_offset - word_width
    assert int.from_bytes(raw[length_offset:path_offset], "little") == len(old)
    replacement = len(new).to_bytes(word_width, "little") + new
    path.write_bytes(raw[:length_offset] + replacement + raw[path_offset + len(old) :])


def _normalize_scalar_paths(case_dir: Path) -> None:
    old_path = str(case_dir / "rocksdb")
    snapshot_data = case_dir / "recovery" / "snapshots" / "snapshot-legacy" / "data.snapshot"
    _replace_length_prefixed_path(snapshot_data, old_path, ROCKSDB_PLACEHOLDER)
    _replace_length_prefixed_path(case_dir / "raw.data", old_path, ROCKSDB_PLACEHOLDER)


def _normalize_snapshot(case_dir: Path) -> None:
    recovery_dir = case_dir / "recovery"
    current_path = recovery_dir / "CURRENT"
    current_id = current_path.read_text(encoding="utf-8").strip()
    old_snapshot = recovery_dir / "snapshots" / current_id
    normalized_id = "snapshot-legacy"
    normalized_snapshot = recovery_dir / "snapshots" / normalized_id
    if old_snapshot != normalized_snapshot:
        if normalized_snapshot.exists():
            shutil.rmtree(normalized_snapshot)
        old_snapshot.rename(normalized_snapshot)
    current_path.write_text(normalized_id + "\n", encoding="utf-8")

    manifest_path = normalized_snapshot / "manifest.txt"
    entries: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            entries[key] = value
    entries["snapshot_id"] = normalized_id
    entries["created_unix_ms"] = "0"
    order = [
        "format_version",
        "snapshot_id",
        "reason",
        "applied_through_op_id",
        "created_unix_ms",
        "graph_file",
        "data_file",
        "quant_file",
        "rocksdb_dir",
    ]
    manifest_path.write_text("".join(f"{key}={entries.get(key, '')}\n" for key in order), encoding="utf-8")


def _remove_volatile_rocksdb_files(case_dir: Path) -> None:
    """Drop optional logs/options while preserving the openable checkpoint."""
    snapshot_db = case_dir / "recovery" / "snapshots" / "snapshot-legacy" / "rocksdb"
    if not snapshot_db.exists():
        return
    for path in snapshot_db.iterdir():
        if path.name == "LOG" or path.name.startswith("LOG.old.") or path.name.startswith("OPTIONS-"):
            path.unlink()


def _case_file_manifest(case_dir: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for path in sorted(case_dir.rglob("*")):
        if not path.is_file() or path.name == "sha256.json":
            continue
        relative = path.relative_to(case_dir).as_posix()
        payload = path.read_bytes()
        entries[relative] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    return entries


def _capture_existing_rocksdb(output: Path) -> dict[str, dict[str, bytes]]:
    """Cache canonical engine files whose embedded random identities are non-reproducible."""
    captured: dict[str, dict[str, bytes]] = {}
    for name in CASE_BUILDERS:
        rocksdb_dir = output / name / "recovery" / "snapshots" / "snapshot-legacy" / "rocksdb"
        if not rocksdb_dir.is_dir():
            continue
        captured[name] = {
            path.relative_to(rocksdb_dir).as_posix(): path.read_bytes()
            for path in rocksdb_dir.rglob("*")
            if path.is_file()
        }
    return captured


def _restore_canonical_rocksdb(case_dir: Path, files: dict[str, bytes] | None) -> None:
    if not files:
        return
    rocksdb_dir = case_dir / "recovery" / "snapshots" / "snapshot-legacy" / "rocksdb"
    shutil.rmtree(rocksdb_dir, ignore_errors=True)
    for relative, payload in files.items():
        path = rocksdb_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _finalize_case(
    case_dir: Path,
    generation: dict[str, Any],
    expected: dict[str, Any],
    canonical_rocksdb: dict[str, bytes] | None,
) -> dict[str, Any]:
    _normalize_schema(case_dir)
    _normalize_snapshot(case_dir)
    if generation["has_scalar_data"]:
        _normalize_scalar_paths(case_dir)
    shutil.rmtree(case_dir / "rocksdb", ignore_errors=True)
    _remove_volatile_rocksdb_files(case_dir)
    _restore_canonical_rocksdb(case_dir, canonical_rocksdb)
    _write_json(case_dir / "generation.json", generation)
    _write_json(case_dir / "expected.json", expected)
    files = _case_file_manifest(case_dir)
    _write_json(case_dir / "sha256.json", {"format_version": 1, "files": files})
    total_bytes = sum(path.stat().st_size for path in case_dir.rglob("*") if path.is_file())
    return {
        "name": case_dir.name,
        "kind": generation["kind"],
        "data_type": generation["data_type"],
        "id_type": generation["id_type"],
        "has_scalar_data": generation["has_scalar_data"],
        "operations": generation["operations"],
        "terminal_shape": generation["terminal_shape"],
        "bytes": total_bytes,
    }


def generate(output: Path, *, refresh_rocksdb: bool = False) -> None:
    output = output.resolve()
    canonical_rocksdb = {} if refresh_rocksdb else _capture_existing_rocksdb(output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "README.md").write_text(CORPUS_README, encoding="utf-8")

    cases = []
    for name, builder in CASE_BUILDERS.items():
        case_dir = output / name
        case_dir.mkdir()
        generation, expected = builder(case_dir)
        cases.append(_finalize_case(case_dir, generation, expected, canonical_rocksdb.get(name)))

    total_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    manifest = {
        "format_version": 1,
        "writer_commit": "ab2cb0f16871c54c6a7f6a63ece88e25d9f66ebd",
        "generator": "scripts/generate_legacy_recovery_corpus.py",
        "rocksdb_path_placeholder": ROCKSDB_PLACEHOLDER,
        "rocksdb_checkpoint_mode": "fresh_writer" if refresh_rocksdb or not canonical_rocksdb else "preserved_existing",
        "cases": cases,
        "payload_bytes_before_corpus_manifest": total_bytes,
        "size_limit_bytes": 2 * 1024 * 1024,
    }
    _write_json(output / "corpus_manifest.json", manifest)
    actual_total = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if actual_total >= manifest["size_limit_bytes"]:
        raise RuntimeError(f"legacy recovery corpus is too large: {actual_total} bytes")
    print(f"generated {len(cases)} cases at {output} ({actual_total} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--refresh-rocksdb",
        action="store_true",
        help="replace canonical scalar checkpoints, including RocksDB's random DB/session identities",
    )
    args = parser.parse_args()
    generate(args.output, refresh_rocksdb=args.refresh_rocksdb)


if __name__ == "__main__":
    main()
