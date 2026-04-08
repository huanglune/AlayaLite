# Copyright 2025 AlayaDB.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for LASER build and save/load through the Python SDK."""

import tempfile
import unittest

import numpy as np

from alayalite import Client

try:
    from alayalite import LaserBuildParams
except ImportError:  # pragma: no cover - depends on LASER build support
    LaserBuildParams = None


@unittest.skipIf(LaserBuildParams is None, "LASER support is not available")
class TestLaserBuild(unittest.TestCase):
    """Verify LASER can build, search, and survive a save/load round trip."""

    def test_fit_search_and_save_load_round_trip(self):
        rng = np.random.default_rng(42)
        vectors = rng.random((256, 128), dtype=np.float32)
        query = vectors[0]

        params = LaserBuildParams()
        params.max_degree = 64
        params.ef_construction = 32
        params.ef_build = 32
        params.num_medoids = 8
        params.max_memory_mb = 16
        params.num_threads = 1

        with tempfile.TemporaryDirectory() as temp_dir:
            client = Client(temp_dir)
            index = client.create_index("laser_index", index_type="laser")
            index.fit(vectors, ef_construction=32, num_threads=1, laser_build_params=params)

            result_before = index.search(query, 5, ef_search=32)
            self.assertEqual(result_before.shape[0], 5)
            self.assertTrue(np.all(result_before >= 0))
            self.assertTrue(np.all(result_before < vectors.shape[0]))

            client.save_index("laser_index")
            reloaded_client = Client(temp_dir)
            reloaded_index = reloaded_client.get_index("laser_index")

            result_after = reloaded_index.search(query, 5, ef_search=32)
            np.testing.assert_array_equal(result_before, result_after)


if __name__ == "__main__":
    unittest.main()
