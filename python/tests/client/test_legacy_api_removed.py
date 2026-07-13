# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""AlayaLite 1.2.0 removal tombstones for the Gate-9 legacy API family."""

import importlib

import alayalite
import pytest
from alayalite import AlayaLiteLegacyApiWarning, Client, Collection


@pytest.mark.parametrize("name", ["Index", "DiskCollection"])
def test_root_package_no_longer_exports_legacy_types(name):
    assert name not in alayalite.__all__
    with pytest.raises(AlayaLiteLegacyApiWarning, match=r"was removed in AlayaLite 1\.2\.0"):
        getattr(alayalite, name)


@pytest.mark.parametrize(
    "module_name,attribute",
    [
        ("alayalite.index", "Index"),
        ("alayalite.disk_collection", "DiskCollection"),
        ("alayalite.laser", "Index"),
        ("alayalite.laser", "RawIndex"),
        ("alayalite.vamana", "build_index"),
    ],
)
def test_legacy_module_aliases_raise_stable_removal_warning(module_name, attribute):
    module = importlib.import_module(module_name)
    assert attribute not in module.__all__
    with pytest.raises(AlayaLiteLegacyApiWarning, match=r"was removed in AlayaLite 1\.2\.0"):
        getattr(module, attribute)


@pytest.mark.parametrize(
    "method",
    [
        "list_indices",
        "get_index",
        "create_index",
        "get_or_create_index",
        "delete_index",
        "save_index",
    ],
)
def test_client_index_method_family_is_removed(method):
    client = Client()
    assert method not in dir(client)
    with pytest.raises(AlayaLiteLegacyApiWarning, match=r"Client index methods was removed"):
        getattr(client, method)


@pytest.mark.parametrize("method", ["get_cpp_index", "get_index"])
def test_collection_escape_hatches_are_removed(method):
    collection = Collection("removed-api-check")
    assert method not in dir(collection)
    with pytest.raises(AlayaLiteLegacyApiWarning, match=rf"Collection\.{method} was removed"):
        getattr(collection, method)
