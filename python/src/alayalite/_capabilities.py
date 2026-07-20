# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Native capability projection for the v2 Python core."""

from __future__ import annotations

from typing import cast

from ._alayalitepy import capabilities as _native_capabilities
from .config import IndexType
from .exceptions import CollectionInternalError, _status_error
from .models import Capabilities


def capabilities() -> Capabilities:
    """Return capabilities of the installed native extension.

    Returns
    -------
    Capabilities
        Frozen supported-index and LASER SIMD diagnostics.

    Raises
    ------
    CollectionInternalError
        If the native capability response is self-inconsistent.
    """
    response = _native_capabilities()
    raw_types = frozenset(response.index_types)
    if not raw_types or not raw_types <= {"flat", "qg"} or "flat" not in raw_types:
        raise _status_error(
            CollectionInternalError,
            "native capabilities returned invalid index types",
            status_code=11,
            operation_stage=18,
            status_detail=1,
        )
    index_types = cast(frozenset[IndexType], raw_types)
    if response.laser_enabled != ("qg" in index_types):
        raise _status_error(
            CollectionInternalError,
            "native capabilities disagree about QG support",
            status_code=11,
            operation_stage=18,
            status_detail=1,
        )
    if response.laser_enabled != (response.laser_simd is not None):
        raise _status_error(
            CollectionInternalError,
            "native capabilities disagree about LASER SIMD selection",
            status_code=11,
            operation_stage=18,
            status_detail=1,
        )
    return Capabilities(
        index_types=index_types,
        laser_enabled=response.laser_enabled,
        laser_simd=response.laser_simd,
    )


__all__ = ["capabilities"]
