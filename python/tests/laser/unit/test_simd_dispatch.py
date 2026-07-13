# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit coverage for LASER runtime SIMD dispatch exposed through Python."""

from __future__ import annotations

import pytest
from alayalite import laser


def test_laser_selected_simd_allows_avx2_fallback() -> None:
    selected = laser.selected_simd()
    if selected == "unavailable":
        pytest.skip("LASER kernels are unavailable on this platform")
    print(f"laser_simd={selected}")
    assert selected in {"avx512", "avx2", "generic"}
