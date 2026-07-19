# SPDX-License-Identifier: AGPL-3.0-only
"""Structural checks for the checked-in persistence-format inventory."""

import json
from pathlib import Path


def test_artifact_baseline_covers_every_current_family():
    baseline = json.loads((Path(__file__).with_name("artifact-baseline.json")).read_text())
    assert baseline["schema_version"] == 1
    assert set(baseline["artifacts"]) == {
        "collection_qg_laser",
        "disk_flat_segment",
        "laser_fixture",
    }
    for artifact in baseline["artifacts"].values():
        assert artifact["files"]
        for record in artifact["files"].values():
            assert record["bytes"] >= 0
            assert len(record["sha256"]) == 64
