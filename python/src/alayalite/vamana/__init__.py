# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Import tombstone for the Vamana builder removed in AlayaLite 1.1.0."""

from alayalite._legacy import raise_removed_legacy_api

__all__: list[str] = []


def __getattr__(name: str):  # pylint: disable=invalid-name
    if name == "build_index":
        raise_removed_legacy_api("vamana.build_index")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
