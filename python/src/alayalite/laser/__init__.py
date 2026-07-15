# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""LASER format helpers.

The public ``Index``/``RawIndex`` builder and search wrappers were removed in
AlayaLite 1.1.0. The immutable LASER format reader remains behind canonical
``Collection.open``.
"""

from alayalite._legacy import raise_removed_legacy_api

try:
    from alayalite._alayalitepy import laser as _laser_native  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - platform feature gate
    _laser_native = None

__all__ = ["selected_simd"]


def selected_simd() -> str:
    """Return the selected LASER kernel backend for diagnostics."""

    return "unavailable" if _laser_native is None else _laser_native.selected_simd()


def __getattr__(name: str):  # pylint: disable=invalid-name
    if name in {"Index", "RawIndex"}:
        raise_removed_legacy_api(f"laser.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
