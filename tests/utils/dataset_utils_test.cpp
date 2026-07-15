// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include "utils/dataset_utils.hpp"

namespace alaya {

TEST(DatasetTest, RandomDatasetHasCorrectDimensions) {
  auto ds = test::random_dataset({.num_vectors = 500, .dim = 64, .gt_topk = 5});

  EXPECT_EQ(ds.data_num, 500);
  EXPECT_EQ(ds.query_num, 50);
  EXPECT_EQ(ds.dim, 64);
  EXPECT_EQ(ds.gt_dim, 5);
  EXPECT_EQ(ds.data.size(), 500U * 64U);
  EXPECT_EQ(ds.queries.size(), 50U * 64U);
  EXPECT_EQ(ds.ground_truth.size(), 50U * 5U);
}

TEST(DatasetTest, RandomDatasetIsDeterministic) {
  auto ds1 = test::random_dataset({.num_vectors = 100, .dim = 32, .seed = 123});
  auto ds2 = test::random_dataset({.num_vectors = 100, .dim = 32, .seed = 123});

  EXPECT_EQ(ds1.data, ds2.data);
  EXPECT_EQ(ds1.queries, ds2.queries);
  EXPECT_EQ(ds1.ground_truth, ds2.ground_truth);
}

TEST(DatasetTest, TwoArgOverloadWorks) {
  auto ds = test::random_dataset(200, 16);

  EXPECT_EQ(ds.data_num, 200);
  EXPECT_EQ(ds.dim, 16);
}

}  // namespace alaya
