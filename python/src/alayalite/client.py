# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Canonical Collection manager.

The ``Client`` index-method family was removed in AlayaLite 1.1.0. Legacy
PyIndex layouts discovered below a client URL are imported non-destructively
through ``Collection.open`` and appear in the collection map.
"""

import logging
import os
import shutil

from ._legacy import raise_removed_legacy_api
from .collection import Collection
from .schema import IndexParams, is_collection_url, is_index_url, save_schema

__all__ = ["Client"]

logger = logging.getLogger(__name__)

_REMOVED_INDEX_METHODS = frozenset(
    {
        "list_indices",
        "get_index",
        "create_index",
        "get_or_create_index",
        "delete_index",
        "save_index",
    }
)


class Client:
    """Manage canonical ``Collection`` instances under an optional root URL."""

    def __init__(self, url=None):
        self.__collection_map = {}
        self.__url = None
        if url is None:
            return

        self.__url = os.path.abspath(url)
        os.makedirs(self.__url, exist_ok=True)
        logger.info("Loading AlayaLite data from %s", self.__url)
        all_names = [entry for entry in os.listdir(self.__url) if os.path.isdir(os.path.join(self.__url, entry))]
        logger.debug("Discovered entries under client url: %s", all_names)
        for name in all_names:
            full_url = os.path.join(self.__url, name)
            if is_collection_url(full_url) or is_index_url(full_url):
                self.__collection_map[name] = Collection.load(self.__url, name)
                logger.info("Loaded collection %s", name)
            else:
                logger.warning("Ignoring unknown storage entry at %s", full_url)

    def __getattr__(self, name: str):
        if name in _REMOVED_INDEX_METHODS:
            raise_removed_legacy_api("Client index methods")
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def list_collections(self):
        return list(self.__collection_map.keys())

    def get_collection(self, name: str = "default") -> Collection:
        return self.__collection_map.get(name)

    @staticmethod
    def _close_collection(collection: Collection) -> None:
        try:
            collection.close()
        except (AttributeError, RuntimeError):
            pass

    def create_collection(self, name: str = "default", **kwargs) -> Collection:
        if name in self.__collection_map:
            raise RuntimeError(f"A collection with name '{name}' already exists")

        index_params = IndexParams.from_kwargs(**kwargs)
        if not index_params.storage_path and self.__url is not None:
            index_params.storage_path = os.path.join(self.__url, name, "storage")
        collection = Collection(name, index_params)
        self.__collection_map[name] = collection
        return collection

    def get_or_create_collection(self, name: str, **kwargs) -> Collection:
        collection = self.get_collection(name)
        if collection is None:
            collection = self.create_collection(name, **kwargs)
        return collection

    def delete_collection(self, collection_name: str, delete_on_disk: bool = False):
        if collection_name not in self.__collection_map:
            raise RuntimeError(f"Collection '{collection_name}' does not exist")
        if delete_on_disk and self.__url is None:
            raise RuntimeError("Client is not initialized with a url for disk operations")
        collection = self.__collection_map.pop(collection_name)
        self._close_collection(collection)
        if delete_on_disk:
            collection_url = os.path.join(self.__url, collection_name)
            if os.path.exists(collection_url):
                shutil.rmtree(collection_url)
                logger.info("Deleted collection '%s' from disk", collection_name)

    def reset(self, delete_on_disk: bool = False):
        if delete_on_disk and self.__url is None:
            raise RuntimeError("Client is not initialized with a url for disk operations")

        collection_items = list(self.__collection_map.items())
        for _, collection in collection_items:
            self._close_collection(collection)

        if delete_on_disk:
            for collection_name, _ in collection_items:
                collection_url = os.path.join(self.__url, collection_name)
                if os.path.exists(collection_url):
                    shutil.rmtree(collection_url)

        self.__collection_map = {}

    def save_collection(self, collection_name: str):
        if self.__url is None:
            raise RuntimeError("Client is not initialized with a url")
        if collection_name not in self.__collection_map:
            raise RuntimeError(f"Collection '{collection_name}' does not exist")

        collection_url = os.path.join(self.__url, collection_name)
        schema_map = self.__collection_map[collection_name].save(collection_url)
        collection_schema_url = os.path.join(collection_url, "schema.json")
        save_schema(collection_schema_url, schema_map)
        logger.info("Saved collection '%s'", collection_name)
