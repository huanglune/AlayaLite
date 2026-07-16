# SPDX-License-Identifier: AGPL-3.0-only
"""Freeze the generated template-instantiation allowlist."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
EXPECTED_COUNTS = {
    ("float", "uint32_t"): 4,
    ("float", "uint64_t"): 3,
    ("int8_t", "uint32_t"): 1,
    ("int8_t", "uint64_t"): 1,
    ("uint8_t", "uint32_t"): 1,
    ("uint8_t", "uint64_t"): 1,
}


def test_dispatch_yaml_is_the_frozen_11_row_allowlist():
    text = (ROOT / "tools/codegen/dispatch.yaml").read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    required_keys = {
        "data",
        "id",
        "quant",
        "index",
        "implementation_key",
        "engine_factory_key",
    }
    combinations = config["combinations"]
    assert all(set(row) == required_keys for row in combinations)
    assert len(combinations) == 11
    by_name = [
        (
            row["data"],
            row["id"],
            row["quant"],
            row["index"],
            row["implementation_key"],
            row["engine_factory_key"],
        )
        for row in combinations
    ]
    counts = {key: sum(row[:2] == key for row in by_name) for key in EXPECTED_COUNTS}
    assert counts == EXPECTED_COUNTS
    assert {row[3] for row in by_name} == {"HNSW"}
    assert {row[2] for row in by_name if row[0] != "float"} == {"NONE"}
    assert {key: sum(row[4:] == key for row in by_name) for key in {row[4:] for row in by_name}} == {
        ("hnsw_segment", "hnsw"): 10,
        ("qg_segment", "qg"): 1,
    }

    assert set(config) == {"implementation_registry", "combinations"}
    assert config["implementation_registry"] == {
        "hnsw_segment": {"engine_factory_key": "hnsw", "artifact_identity": "hnsw"},
        "qg_segment": {"engine_factory_key": "qg", "artifact_identity": "qg"},
    }

    generated = (ROOT / "python/tests/client/_dispatch_matrix_params.py").read_text(encoding="utf-8")
    assert "Source: tools/codegen/dispatch.yaml (11 combinations)." in generated
    assert generated.count("False),") == 11
    assert generated.count('("hnsw_segment", "hnsw", "hnsw"),') == 10
    assert generated.count('("qg_segment", "qg", "qg"),') == 1
