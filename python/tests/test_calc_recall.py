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

"""Tests for recall calculation semantics."""

import importlib.util
import pathlib
import unittest

import numpy as np

UTILS_PATH = pathlib.Path(__file__).resolve().parents[1] / "src" / "alayalite" / "utils.py"
SPEC = importlib.util.spec_from_file_location("alayalite_utils_module", UTILS_PATH)
UTILS_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(UTILS_MODULE)
calc_recall = UTILS_MODULE.calc_recall


class TestCalcRecall(unittest.TestCase):
    """Test cases for recall calculation."""

    def test_uses_only_topk_ground_truth_when_gt_has_more_columns(self):
        result = np.array([[2, 3]], dtype=np.int32)
        gt_data = np.array([[0, 1, 2, 3, 4]], dtype=np.int32)
        recall = calc_recall(result, gt_data)
        self.assertEqual(recall, 0.0)


if __name__ == "__main__":
    unittest.main()
