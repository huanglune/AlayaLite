# SPDX-License-Identifier: AGPL-3.0-only
"""Freeze the generated template-instantiation allowlist."""

import re
from pathlib import Path


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
    rows = re.findall(
        r"^- \{data: ([^,]+), id: ([^,]+), quant: ([^,]+), index: ([^}]+)\}$",
        text,
        flags=re.MULTILINE,
    )
    assert len(rows) == 33
    counts = {key: sum(row[:2] == key for row in rows) for key in EXPECTED_COUNTS}
    assert counts == EXPECTED_COUNTS
    assert {row[3] for row in rows} == {"HNSW", "NSG", "FUSION"}
    assert {row[2] for row in rows if row[0] != "float"} == {"NONE"}

    generated = (ROOT / "python/tests/client/_dispatch_matrix_params.py").read_text(encoding="utf-8")
    assert "Source: tools/codegen/dispatch.yaml (33 combinations)." in generated
    assert generated.count("False),") == 33
