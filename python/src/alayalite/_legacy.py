# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Stable tombstones for Python APIs removed at the Gate-11 boundary."""

LEGACY_API_V_REMOVE = "1.2.0"


class AlayaLiteLegacyApiWarning(DeprecationWarning):
    """Raised when code accesses a Python API removed in AlayaLite 1.2.0."""


_REMOVED_API_REPLACEMENTS = {
    "Index": "alayalite.Collection",
    "DiskCollection": "alayalite.Collection",
    "laser.Index": "alayalite.Collection",
    "laser.RawIndex": "alayalite.Collection",
    "vamana.build_index": "alayalite.Collection",
    "Client index methods": "Client collection methods",
    "Collection.get_cpp_index": "canonical Collection methods",
    "Collection.get_index": "canonical Collection methods",
}


def raise_removed_legacy_api(api_name: str) -> None:
    """Raise the stable 1.2.0 tombstone for a removed public entry."""

    replacement = _REMOVED_API_REPLACEMENTS[api_name]
    raise AlayaLiteLegacyApiWarning(
        f"alayalite.{api_name} was removed in AlayaLite {LEGACY_API_V_REMOVE}; use {replacement} instead."
    )
