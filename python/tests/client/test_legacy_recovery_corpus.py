# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Read-back verification for the checked-in legacy recovery corpus."""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest
from alayalite import Client

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "legacy_recovery_corpus"
CORPUS_MANIFEST = json.loads((CORPUS_ROOT / "corpus_manifest.json").read_text(encoding="utf-8"))
ROCKSDB_PLACEHOLDER = CORPUS_MANIFEST["rocksdb_path_placeholder"]
WAL_TRAILER_MAGIC = 0x5441494C


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _case_id(case: dict) -> str:
    return case["name"]


def test_legacy_recovery_corpus_integrity_and_size():
    """Every payload byte is checksummed and the complete corpus stays below 2 MiB."""
    all_files = [path for path in CORPUS_ROOT.rglob("*") if path.is_file()]
    assert sum(path.stat().st_size for path in all_files) < CORPUS_MANIFEST["size_limit_bytes"]
    payload_files = [path for path in all_files if path.name != "corpus_manifest.json"]
    assert sum(path.stat().st_size for path in payload_files) == CORPUS_MANIFEST["payload_bytes_before_corpus_manifest"]

    case_names = {case["name"] for case in CORPUS_MANIFEST["cases"]}
    assert case_names == {path.name for path in CORPUS_ROOT.iterdir() if path.is_dir()}

    for case in CORPUS_MANIFEST["cases"]:
        case_dir = CORPUS_ROOT / case["name"]
        assert sum(path.stat().st_size for path in case_dir.rglob("*") if path.is_file()) == case["bytes"]
        checksum_manifest = _read_json(case_dir / "sha256.json")
        expected_files = checksum_manifest["files"]
        actual_files = {
            path.relative_to(case_dir).as_posix()
            for path in case_dir.rglob("*")
            if path.is_file() and path.name != "sha256.json"
        }
        assert set(expected_files) == actual_files
        for relative, expected_digest in expected_files.items():
            payload = (case_dir / relative).read_bytes()
            assert len(payload) == expected_digest["bytes"]
            assert hashlib.sha256(payload).hexdigest() == expected_digest["sha256"]

        terminal_shape = case["terminal_shape"]
        wal_path = case_dir / "recovery" / "wal.bin"
        if terminal_shape == "clean_snapshot":
            assert not wal_path.exists()
        elif terminal_shape == "torn_wal_tail":
            assert wal_path.is_file()
            assert struct.unpack_from("<I", wal_path.read_bytes(), wal_path.stat().st_size - 4)[0] != WAL_TRAILER_MAGIC
        else:
            assert wal_path.is_file()
            assert struct.unpack_from("<I", wal_path.read_bytes(), wal_path.stat().st_size - 4)[0] == WAL_TRAILER_MAGIC


def _relocate_schema(case_dir: Path) -> None:
    schema_path = case_dir / "schema.json"
    schema = _read_json(schema_path)
    assert schema["index"]["rocksdb_path"] == ROCKSDB_PLACEHOLDER
    schema["index"]["rocksdb_path"] = str(case_dir / "rocksdb")
    schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _replace_length_prefixed_path(path: Path, old_path: str, new_path: str) -> None:
    if not path.exists():
        return
    raw = path.read_bytes()
    old = old_path.encode()
    new = new_path.encode()
    assert raw.count(old) == 1
    path_offset = raw.index(old)
    word_width = struct.calcsize("P")
    length_offset = path_offset - word_width
    assert int.from_bytes(raw[length_offset:path_offset], "little") == len(old)
    replacement = len(new).to_bytes(word_width, "little") + new
    path.write_bytes(raw[:length_offset] + replacement + raw[path_offset + len(old) :])


def _relocate_scalar_paths(case_dir: Path, has_scalar_data: bool) -> None:
    if not has_scalar_data:
        return
    new_path = str(case_dir / "rocksdb")
    snapshot_data = case_dir / "recovery" / "snapshots" / "snapshot-legacy" / "data.snapshot"
    _replace_length_prefixed_path(snapshot_data, ROCKSDB_PLACEHOLDER, new_path)
    _replace_length_prefixed_path(case_dir / "raw.data", ROCKSDB_PLACEHOLDER, new_path)


def _verify_collection(collection, generation: dict, expected: dict, case: dict) -> None:
    stats = collection.stats()
    expected_live_count = expected.get("expected_live_count")
    if expected_live_count is None:
        removed_rows = sum("remove(" in operation for operation in case["operations"])
        expected_live_count = expected["expected_count"] - removed_rows
    assert int(stats["size"]) == expected_live_count
    assert int(stats["allocated_count"]) >= int(stats["size"])
    assert collection.options()["imported_legacy_layout"] is True

    if "item_lookup_order" in expected:
        found = collection.get_by_id(expected["item_lookup_order"])
        expected_items = expected["expected_items"]
        assert found["id"] == [item["id"] for item in expected_items]
        assert found["document"] == [item["document"] for item in expected_items]
        assert found["metadata"] == [item["metadata"] for item in expected_items]
        assert len(found["id"]) == expected_live_count

    dtype = np.dtype(generation["data_type"])
    for query in expected["queries"]:
        vector = np.asarray([query["vector"]], dtype=dtype)
        actual = collection.batch_search(vector, top_k=query["topk"])
        expected_ids = query.get("expected_item_ids")
        if expected_ids is None:
            expected_ids = [str(value) for value in query["expected_internal_ids"]]
        assert [str(value) for value in actual["ids"].tolist()] == expected_ids


@pytest.mark.parametrize("case", CORPUS_MANIFEST["cases"], ids=_case_id)
def test_legacy_reader_recovers_checked_in_corpus(case, tmp_path):
    """Import every legacy layout through canonical Collection without changing source bytes."""
    source = CORPUS_ROOT / case["name"]
    copied_case = tmp_path / case["name"]
    shutil.copytree(source, copied_case)
    _relocate_schema(copied_case)
    _relocate_scalar_paths(copied_case, case["has_scalar_data"])
    frozen_source = {
        path.relative_to(copied_case): (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in copied_case.rglob("*")
        if path.is_file()
    }

    generation = _read_json(copied_case / "generation.json")
    expected = _read_json(copied_case / "expected.json")
    client = Client(tmp_path)
    try:
        collection = client.get_collection(case["name"])
        assert collection is not None
        _verify_collection(collection, generation, expected, case)
        assert (copied_case / ".alaya_internal" / "legacy_import_v1" / "ACTIVE").is_file()
        for relative, (expected_bytes, expected_sha) in frozen_source.items():
            path = copied_case / relative
            assert path.stat().st_size == expected_bytes
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha
    finally:
        client.reset()
