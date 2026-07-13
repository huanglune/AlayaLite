# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""URL-backed canonical ``Client`` tests."""

import gc
import os
import shutil
import tempfile
import unittest

import numpy as np
from alayalite import Client


class TestClientWithURL(unittest.TestCase):
    def setUp(self):
        self.items = [
            (1, "Document 1", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"category": "A"}),
            (2, "Document 2", np.array([0.4, 0.5, 0.6], dtype=np.float32), {"category": "B"}),
        ]
        self.temp_dir = tempfile.mkdtemp()
        self.client = Client(self.temp_dir)

    def tearDown(self):
        del self.client
        gc.collect()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_load_and_delete_collection(self):
        with self.assertRaises(RuntimeError):
            self.client.save_collection("missing")
        collection = self.client.create_collection("collection")
        collection.insert(self.items)
        self.client.save_collection("collection")

        root = os.path.join(self.temp_dir, "collection")
        self.assertTrue(os.path.isfile(os.path.join(root, "schema.json")))
        self.client.reset()

        reopened_client = Client(self.temp_dir)
        reopened = reopened_client.get_collection("collection")
        self.assertEqual(reopened.get_by_id([1, 2])["document"], ["Document 1", "Document 2"])
        reopened_client.delete_collection("collection", True)
        self.assertFalse(os.path.exists(root))

    def test_collection_recovers_without_explicit_save(self):
        collection = self.client.create_collection("recovering")
        collection.insert(
            [
                ("a", "Document A", np.array([1.0, 0.0, 0.0], dtype=np.float32), {"group": "keep"}),
                ("b", "Document B", np.array([0.0, 1.0, 0.0], dtype=np.float32), {"group": "drop"}),
            ]
        )
        collection.upsert(
            [
                ("a", "Document A v2", np.array([1.0, 0.1, 0.0], dtype=np.float32), {"group": "keep", "v": 2}),
                ("c", "Document C", np.array([0.0, 0.0, 1.0], dtype=np.float32), {"group": "keep"}),
            ]
        )
        collection.delete_by_id(["b"])

        self.client.reset()
        recovered = Client(self.temp_dir).get_collection("recovering")
        result = recovered.get_by_id(["a", "b", "c"])
        self.assertEqual(result["id"], ["a", "c"])
        self.assertEqual(result["document"][0], "Document A v2")
        self.assertEqual(result["metadata"][0]["v"], 2)

    def test_sq8_collection_recovery_uses_canonical_checkpoint(self):
        collection = self.client.create_collection("recovering_sq8", quantization_type="sq8", metric="ip")
        collection.insert(self.items)
        collection.upsert(
            [(2, "Document 2 v2", np.array([0.4, 0.5, 0.7], dtype=np.float32), {"category": "new"})]
        )
        self.client.reset()

        recovered = Client(self.temp_dir).get_collection("recovering_sq8")
        result = recovered.get_by_id([1, 2])
        self.assertEqual(result["document"][1], "Document 2 v2")
        self.assertEqual(result["metadata"][1]["category"], "new")


if __name__ == "__main__":
    unittest.main()
