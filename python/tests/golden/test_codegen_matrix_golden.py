# SPDX-License-Identifier: AGPL-3.0-only
"""Freeze the generated template-instantiation allowlist."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
EXPECTED_COUNTS = {
    ("float", "uint32_t"): 12,
    ("float", "uint64_t"): 9,
    ("int8_t", "uint32_t"): 3,
    ("int8_t", "uint64_t"): 3,
    ("uint8_t", "uint32_t"): 3,
    ("uint8_t", "uint64_t"): 3,
}


def test_dispatch_yaml_is_the_frozen_33_row_allowlist():
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
    assert len(combinations) == 33
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
    assert {row[3] for row in by_name} == {"HNSW", "NSG", "FUSION"}
    assert {row[2] for row in by_name if row[0] != "float"} == {"NONE"}
    assert {key: sum(row[4:] == key for row in by_name) for key in {row[4:] for row in by_name}} == {
        ("hnsw_segment", "hnsw"): 10,
        ("nsg_segment", "nsg"): 10,
        ("fusion_segment", "fusion"): 10,
        ("legacy_qg_model", "qg"): 3,
    }

    engine_factories = config["engine_factories"]
    assert engine_factories["hnsw"]["rollback"] == "source_revert"
    assert engine_factories["hnsw"]["feature_flag"] == "none"
    assert {engine_factories[key]["feature_flag"] for key in ("nsg", "fusion", "qg")} == {
        "nsg_segment",
        "fusion_segment",
        "qg_segment",
    }

    generated = (ROOT / "python/tests/client/_dispatch_matrix_params.py").read_text(encoding="utf-8")
    assert "Source: tools/codegen/dispatch.yaml (33 combinations)." in generated
    assert generated.count("False),") == 33
    assert generated.count('("hnsw_segment", "hnsw", "hnsw"),') == 10
    assert generated.count('("nsg_segment", "nsg", "nsg"),') == 10
    assert generated.count('("fusion_segment", "fusion", "fusion"),') == 10
    assert generated.count('("legacy_qg_model", "qg", "qg"),') == 3
