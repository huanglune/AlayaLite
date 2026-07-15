# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Import tombstone for the ``Index`` API removed in AlayaLite 1.1.0."""

from ._legacy import raise_removed_legacy_api

__all__: list[str] = []


def __getattr__(name: str):  # pylint: disable=invalid-name
    if name == "Index":
        raise_removed_legacy_api("Index")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
