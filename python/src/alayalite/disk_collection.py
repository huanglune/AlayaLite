# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Import tombstone for ``DiskCollection`` removed in AlayaLite 1.2.0."""

from ._legacy import raise_removed_legacy_api

__all__: list[str] = []


def __getattr__(name: str):
    if name == "DiskCollection":
        raise_removed_legacy_api("DiskCollection")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
