# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""Focused tests for the wave-A private binding increments."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from alayalite import _alayalitepy as native_module
from alayalite._alayalitepy import _Collection


@pytest.fixture(name="native_collection")
def _native_collection_fixture(tmp_path):
    collection = _Collection.create(
        str(tmp_path / "collection"),
        3,
        "l2",
        np.dtype(np.float32),
        "flat",
        "none",
    )
    try:
        yield collection
    finally:
        collection.close()


def _columns():
    return (
        ["a", "b"],
        ["A", "B"],
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        [{"kind": "keep"}, {"kind": "drop"}],
    )


def test_typed_mutation_coexists_with_the_legacy_dict_response(native_collection):
    ids, documents, vectors, metadata = _columns()
    typed = native_collection.mutate_typed(ids, documents, vectors, metadata, "add")
    legacy = native_collection.mutate(
        ["c"],
        ["C"],
        np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32),
        [{}],
        "add",
    )

    assert type(typed).__name__ == "_MutationResponse"
    assert typed.batch_op_id > 0
    assert [type(row).__name__ for row in typed.rows] == ["_MutationRowResponse"] * 2
    assert [row.row_status for row in typed.rows] == [0, 0]
    assert isinstance(legacy, dict)
    assert legacy["rows"][0]["row_status"] == 0
    with pytest.raises(AttributeError):
        typed.batch_op_id = 0


def test_typed_search_has_named_nested_stats_and_old_search_stays_dict(native_collection):
    ids, documents, vectors, metadata = _columns()
    native_collection.mutate_typed(ids, documents, vectors, metadata, "add")
    query = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)

    typed = native_collection.search_typed(query, 2)
    legacy = native_collection.search(query, 2)

    assert type(typed).__name__ == "_SearchResponse"
    assert typed.ids.tolist() == ["a", "b"]
    assert typed.offsets.tolist() == [0, 2]
    assert typed.valid_counts.tolist() == [2]
    assert typed.status_codes.shape == (1,)
    assert typed.completeness_codes.shape == (1,)
    assert type(typed.search_stats).__name__ == "_SearchStatsResponse"
    assert typed.search_stats.effective_effort is None
    assert isinstance(legacy, dict)
    assert legacy["ids"].tolist() == typed.ids.tolist()


def test_native_scan_filters_and_projects_before_returning_records(native_collection):
    ids, documents, vectors, metadata = _columns()
    native_collection.mutate_typed(ids, documents, vectors, metadata, "add")

    records = native_collection.scan(metadata_filter={"kind": "keep"}, limit=1)
    vectors_included = native_collection.scan(
        metadata_filter={"kind": {"$in": ["keep", "drop"]}},
        limit=2,
        include_vector=True,
    )

    assert len(records) == 1
    assert type(records[0]).__name__ == "_RecordResponse"
    assert records[0].id == "a"
    assert records[0].document == "A"
    assert records[0].metadata == {"kind": "keep"}
    assert records[0].vector is None
    assert [record.id for record in vectors_included] == ["a", "b"]
    assert all(record.vector.shape == (3,) for record in vectors_included)


def test_typed_get_stats_options_checkpoint_and_maintenance(native_collection):
    ids, documents, vectors, metadata = _columns()
    native_collection.mutate_typed(ids, documents, vectors, metadata, "add")

    assert native_collection.get_by_id_typed("missing") is None
    record = native_collection.get_by_id_typed("a", include_vector=False)
    assert type(record).__name__ == "_RecordResponse"
    assert record.vector is None
    assert type(native_collection.stats_typed()).__name__ == "_StatsResponse"
    assert type(native_collection.options_typed()).__name__ == "_OptionsResponse"
    assert type(native_collection.checkpoint_typed()).__name__ == "_CheckpointResponse"

    assert type(native_collection.seal_typed()).__name__ == "_SealResponse"
    assert type(native_collection.gc_typed()).__name__ == "_GcResponse"
    native_collection.mutate_typed(
        ["c", "d"],
        ["C", "D"],
        np.asarray([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32),
        [{}, {}],
        "add",
    )
    native_collection.seal_typed()
    native_collection.gc_typed()
    assert type(native_collection.compact_typed()).__name__ == "_CompactResponse"


def test_capabilities_absorb_laser_selected_simd_without_removing_legacy_diagnostic():
    capabilities = native_module.capabilities()

    assert type(capabilities).__name__ == "_CapabilitiesResponse"
    assert capabilities.index_types[0] == "flat"
    assert capabilities.laser_enabled is ("qg" in capabilities.index_types)
    if capabilities.laser_enabled:
        assert capabilities.laser_simd == native_module.laser.selected_simd()
    else:
        assert capabilities.laser_simd is None


def test_private_response_fields_match_the_checked_in_stub():
    stub_path = Path(native_module.__file__).with_name("_alayalitepy.pyi")
    if not stub_path.is_file():
        stub_path = Path(__file__).parents[2] / "src" / "alayalite" / "_alayalitepy.pyi"
    tree = ast.parse(stub_path.read_text(encoding="utf-8"))
    stub_fields = {
        node.name: {statement.target.id for statement in node.body if isinstance(statement, ast.AnnAssign)}
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.endswith("Response")
    }

    for class_name, expected_fields in stub_fields.items():
        runtime_type = getattr(native_module, class_name)
        runtime_fields = {name for name in vars(runtime_type) if not name.startswith("_")}
        assert runtime_fields == expected_fields, class_name


def test_binding_read_only_open_is_native_and_byte_stable(tmp_path):
    root = tmp_path / "read-only"
    writer = _Collection.create(
        str(root),
        3,
        "l2",
        np.dtype(np.float32),
        "flat",
        "none",
    )
    writer.mutate_typed(
        ["a"],
        ["A"],
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        [{"kind": "keep"}],
        "add",
    )
    writer.close()
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    reader = _Collection.open(str(root), True)
    assert reader.read_only is True
    assert reader.options_typed().read_only is True
    assert reader.get_by_id_typed("a", include_vector=False).document == "A"
    with pytest.raises(native_module.CollectionNotSupportedError) as captured:
        reader.mutate_typed(
            ["b"],
            ["B"],
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            [{}],
            "add",
        )
    assert captured.value.status_detail == 15
    with pytest.raises(native_module.CollectionNotSupportedError):
        reader.checkpoint_typed()
    reader.close()

    after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert after == before
