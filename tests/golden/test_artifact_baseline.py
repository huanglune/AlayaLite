# SPDX-License-Identifier: AGPL-3.0-only
"""Structural checks for the checked-in persistence-format inventory."""

import json
from pathlib import Path


def test_artifact_baseline_covers_every_current_family():
    baseline = json.loads((Path(__file__).with_name("artifact-baseline.json")).read_text())
    assert baseline["schema_version"] == 1
    assert set(baseline["artifacts"]) == {
        "disk_flat", "disk_vamana", "diskann", "laser_fixture",
        "memory_fusion_none", "memory_fusion_sq8",
        "memory_hnsw_none", "memory_hnsw_sq8",
        "memory_nsg_none", "memory_nsg_sq8",
    }
    for artifact in baseline["artifacts"].values():
        assert artifact["files"]
        for record in artifact["files"].values():
            assert record["bytes"] >= 0
            assert len(record["sha256"]) == 64
