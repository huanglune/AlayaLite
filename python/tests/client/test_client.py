# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the canonical-only ``Client`` surface."""

import gc
import os
import shutil
import tempfile
import unittest

import numpy as np
from alayalite import Client, Collection


class _FakeClosable:
    def __init__(self, fail=False):
        self.fail = fail
        self.closed = False

    def close(self):
        self.closed = True
        if self.fail:
            raise RuntimeError("close failed")


class TestClient(unittest.TestCase):  # pylint: disable=missing-class-docstring
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._original_storage_dir = os.environ.get("ALAYALITE_STORAGE_DIR")
        os.environ["ALAYALITE_STORAGE_DIR"] = os.path.join(self.tmp_dir, "Storage")
        self.client = Client()

    def tearDown(self):
        del self.client
        gc.collect()
        if self._original_storage_dir is None:
            os.environ.pop("ALAYALITE_STORAGE_DIR", None)
        else:
            os.environ["ALAYALITE_STORAGE_DIR"] = self._original_storage_dir
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_create_get_and_duplicate_collection(self):
        collection = self.client.create_collection("test_collection")
        self.assertIsInstance(collection, Collection)
        self.assertIs(self.client.get_collection("test_collection"), collection)
        self.assertEqual(self.client.list_collections(), ["test_collection"])
        with self.assertRaises(RuntimeError):
            self.client.create_collection("test_collection")
        with self.assertRaises(RuntimeError):
            self.client.save_collection("test_collection")

    def test_get_or_create_collection(self):
        first = self.client.get_or_create_collection("test_collection")
        second = self.client.get_or_create_collection("test_collection")
        self.assertIs(first, second)
        self.assertIsNone(self.client.get_collection("missing"))

    def test_delete_collection_requires_url_for_disk_removal(self):
        self.client.create_collection("test_collection")
        with self.assertRaises(RuntimeError):
            self.client.delete_collection("test_collection", True)
        self.assertIn("test_collection", self.client.list_collections())
        with self.assertRaises(RuntimeError):
            self.client.delete_collection("missing")

    def test_reset_closes_managed_resources_and_deletes_disk_entries(self):
        url = os.path.join(self.tmp_dir, "client_url")
        os.makedirs(os.path.join(url, "collection_on_disk"))
        client = Client(url)
        collection = _FakeClosable()
        client._Client__collection_map = {"collection_on_disk": collection}  # pylint: disable=protected-access

        client.reset(delete_on_disk=True)

        self.assertTrue(collection.closed)
        self.assertFalse(os.path.exists(os.path.join(url, "collection_on_disk")))
        self.assertEqual(client.list_collections(), [])

    def test_close_helper_ignores_native_close_errors(self):
        collection = _FakeClosable(fail=True)
        Client._close_collection(collection)  # pylint: disable=protected-access
        self.assertTrue(collection.closed)

    def test_collection_params(self):
        items = [
            (1, "Document 1", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6], dtype=np.float32), {"category": "B"}),
        ]
        first = self.client.create_collection("col1")
        first.insert(items)
        # float32 with no explicit index_type/quantization_type now defaults
        # to qg+rabitq (HNSW retirement wave, see CHANGELOG).
        self.assertEqual(first.get_index_params().quantization_type, "rabitq")

        second = self.client.create_collection("col2", quantization_type="sq8")
        second.insert(items)
        self.assertEqual(second.get_index_params().quantization_type, "sq8")

    def test_collection_ann_build_kwargs_reach_native_and_validate(self):
        items = [
            (1, "Document 1", np.array([0.1, 0.2, 0.3], dtype=np.float32), {}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6], dtype=np.float32), {}),
        ]
        collection = self.client.create_collection(
            "ann_params",
            max_nbrs=19,
            ef_construction=157,
        )
        collection.insert(items)

        options = collection.options()
        self.assertEqual(options["max_neighbors"], 19)
        self.assertEqual(options["ef_construction"], 157)
        self.assertEqual(collection.get_index_params().max_nbrs, 19)
        self.assertEqual(collection.get_index_params().ef_construction, 157)

        with self.assertRaisesRegex(ValueError, "Max neighbors must be greater than 0"):
            self.client.create_collection("invalid_max_neighbors", max_nbrs=0)
        with self.assertRaisesRegex(ValueError, "ef_construction must be greater than 0"):
            self.client.create_collection("invalid_ef_construction", ef_construction=0)


if __name__ == "__main__":
    unittest.main()
