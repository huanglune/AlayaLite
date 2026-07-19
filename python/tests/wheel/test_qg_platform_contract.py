# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""Installed-wheel smoke for the qg same-id implementation contract."""

import platform

import alayalite
import numpy as np
import pytest
from alayalite import Collection, CollectionNotSupportedError
from alayalite.schema import IndexParams


def test_wheel_exports_the_1_1_public_surface():
    assert "Index" not in alayalite.__all__
    assert "DiskCollection" not in alayalite.__all__
    assert alayalite.__version__ == "1.1.0"


@pytest.mark.skipif(
    not (
        (platform.system() == "Linux" and platform.machine().lower() in {"aarch64", "arm64"})
        or platform.system() == "Windows"
    ),
    reason="this wheel platform includes LASER",
)
def test_laser_unavailable_wheel_rejects_qg_without_flat_fallback(tmp_path):
    collection = Collection(
        "qg-platform-gate",
        IndexParams(
            index_type="qg",
            quantization_type="rabitq",
            metric="euclidean",
            storage_path=str(tmp_path / "qg-platform-gate" / "storage"),
        ),
    )
    vectors = np.arange(40 * 64, dtype=np.float32).reshape(40, 64) / np.float32(257.0)
    collection.add([(str(row), "", vectors[row], {}) for row in range(len(vectors))])

    with pytest.raises(CollectionNotSupportedError) as captured:
        collection.seal()

    diagnostic = str(captured.value)
    assert "LASER" in diagnostic
    assert "not supported" in diagnostic
    assert "Flat fallback is disabled" in diagnostic
    assert collection.stats()["sealed_segments_count"] == 0
    collection.close()
