/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>

#include "utils/dataset_utils.hpp"

namespace alaya {

class DatasetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    data_dir_ = std::filesystem::current_path().parent_path() / "data";
  }

  std::filesystem::path data_dir_;
};

TEST_F(DatasetTest, LoadSiftSmall) {
  auto config = sift_small(data_dir_);

  auto ds = load_dataset(config);

  EXPECT_EQ(ds.name_, "siftsmall");
  EXPECT_GT(ds.data_num_, 0);
  EXPECT_GT(ds.query_num_, 0);
  EXPECT_GT(ds.dim_, 0);
  EXPECT_EQ(ds.data_.size(), ds.data_num_ * ds.dim_);
  EXPECT_EQ(ds.queries_.size(), ds.query_num_ * ds.dim_);

  EXPECT_TRUE(std::filesystem::exists(config.data_file_));
  EXPECT_TRUE(std::filesystem::exists(config.query_file_));
  EXPECT_TRUE(std::filesystem::exists(config.gt_file_));
}

TEST_F(DatasetTest, DISABLED_LoadDeep1M) {
  // Disabled: too slow due to large dataset download

  auto config = deep1m(data_dir_);

  auto ds = load_dataset(config);

  EXPECT_EQ(ds.name_, "deep1M");
  EXPECT_GT(ds.data_num_, 0);
  EXPECT_GT(ds.query_num_, 0);
  EXPECT_GT(ds.dim_, 0);
  EXPECT_EQ(ds.data_.size(), ds.data_num_ * ds.dim_);
  EXPECT_EQ(ds.queries_.size(), ds.query_num_ * ds.dim_);

  EXPECT_TRUE(std::filesystem::exists(config.data_file_));
  EXPECT_TRUE(std::filesystem::exists(config.query_file_));
  EXPECT_TRUE(std::filesystem::exists(config.gt_file_));
}

TEST_F(DatasetTest, RandomDataset) {
  constexpr uint32_t kDataNum = 500;
  constexpr uint32_t kQueryNum = 20;
  constexpr uint32_t kDim = 64;
  constexpr uint32_t kGtTopk = 10;

  auto ds = load_dataset(random_config(kDataNum, kQueryNum, kDim, kGtTopk));

  EXPECT_EQ(ds.name_, "random");
  EXPECT_EQ(ds.data_num_, kDataNum);
  EXPECT_EQ(ds.query_num_, kQueryNum);
  EXPECT_EQ(ds.dim_, kDim);
  EXPECT_EQ(ds.gt_dim_, kGtTopk);
  EXPECT_EQ(ds.data_.size(), static_cast<size_t>(kDataNum) * kDim);
  EXPECT_EQ(ds.queries_.size(), static_cast<size_t>(kQueryNum) * kDim);
  EXPECT_EQ(ds.ground_truth_.size(), static_cast<size_t>(kQueryNum) * kGtTopk);

  // Verify ground truth IDs are valid
  for (uint32_t i = 0; i < kQueryNum; ++i) {
    for (uint32_t j = 0; j < kGtTopk; ++j) {
      uint32_t gt_id = ds.ground_truth_[i * kGtTopk + j];
      EXPECT_LT(gt_id, kDataNum) << "GT ID " << gt_id << " exceeds data_num " << kDataNum;
    }
  }
}

TEST_F(DatasetTest, RandomDatasetReproducibility) {
  constexpr uint32_t kDataNum = 100;
  constexpr uint32_t kQueryNum = 10;
  constexpr uint32_t kDim = 32;
  constexpr uint32_t kSeed = 123;

  // Generate two datasets with the same seed
  auto ds1 = load_dataset(random_config(kDataNum, kQueryNum, kDim, 10, kSeed));
  auto ds2 = load_dataset(random_config(kDataNum, kQueryNum, kDim, 10, kSeed));

  // They should be identical
  EXPECT_EQ(ds1.data_, ds2.data_);
  EXPECT_EQ(ds1.queries_, ds2.queries_);
  EXPECT_EQ(ds1.ground_truth_, ds2.ground_truth_);
}

TEST_F(DatasetTest, RandomDatasetDifferentSeeds) {
  constexpr uint32_t kDataNum = 100;
  constexpr uint32_t kQueryNum = 10;
  constexpr uint32_t kDim = 32;

  // Generate two datasets with different seeds
  auto ds1 = load_dataset(random_config(kDataNum, kQueryNum, kDim, 10, 42));
  auto ds2 = load_dataset(random_config(kDataNum, kQueryNum, kDim, 10, 99));

  // They should be different
  EXPECT_NE(ds1.data_, ds2.data_);
  EXPECT_NE(ds1.queries_, ds2.queries_);
}

}  // namespace alaya
