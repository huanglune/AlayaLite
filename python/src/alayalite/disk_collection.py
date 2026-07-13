# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Legacy outer wrapper for the native DiskCollection-v1 binding."""

from __future__ import annotations

from ._alayalitepy import DiskCollection as _NativeDiskCollection
from ._legacy import legacy_api


class DiskCollection:
    """Compatibility wrapper preserving the native DiskCollection-v1 contract."""

    @legacy_api(
        "disk_collection",
        "disk_collection",
        "alayalite.DiskCollection",
        "alayalite.Collection",
    )
    def __init__(self, *args, **kwargs):
        self.__native = _NativeDiskCollection(*args, **kwargs)

    @classmethod
    @legacy_api(
        "disk_collection",
        "disk_collection",
        "alayalite.DiskCollection",
        "alayalite.Collection",
    )
    def open(cls, *args, **kwargs):
        instance = cls.__new__(cls)
        instance.__native = _NativeDiskCollection.open(*args, **kwargs)
        return instance

    def add(self, *args, **kwargs):
        return self.__native.add(*args, **kwargs)

    def flush(self, *args, **kwargs):
        return self.__native.flush(*args, **kwargs)

    def import_laser_segment(self, *args, **kwargs):
        return self.__native.import_laser_segment(*args, **kwargs)

    def search(self, *args, **kwargs):
        return self.__native.search(*args, **kwargs)

    def batch_search(self, *args, **kwargs):
        return self.__native.batch_search(*args, **kwargs)

    def batch_search_with_distance(self, *args, **kwargs):
        return self.__native.batch_search_with_distance(*args, **kwargs)

    def size(self, *args, **kwargs):
        return self.__native.size(*args, **kwargs)

    def dim(self, *args, **kwargs):
        return self.__native.dim(*args, **kwargs)


# Preserve the binding's detailed distance, validation, and sentinel docs on
# the public compatibility methods.
for _method_name in (
    "add",
    "flush",
    "import_laser_segment",
    "search",
    "batch_search",
    "batch_search_with_distance",
    "size",
    "dim",
):
    getattr(DiskCollection, _method_name).__doc__ = getattr(_NativeDiskCollection, _method_name).__doc__


__all__ = ["DiskCollection"]
