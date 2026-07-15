// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <string>

#include "utils/dataset_utils.hpp"

namespace alaya {

TEST(DatasetTest, SiftMicroConfig) {
  auto config = sift_micro();

  EXPECT_EQ(config.name_, "siftmicro");
  EXPECT_EQ(config.max_data_num_, 1000);
  EXPECT_EQ(config.max_query_num_, 50);
  EXPECT_TRUE(config.data_file_.string().find("siftsmall") != std::string::npos);
}

TEST(DatasetTest, NoArgOverloadsUseTestDataDir) {
  auto ss = sift_small();
  auto sm = sift_micro();
  auto d1 = deep1m();

  EXPECT_EQ(ss.dir_.parent_path(), test::data_dir());
  EXPECT_EQ(sm.dir_.parent_path(), test::data_dir());
  EXPECT_EQ(d1.dir_.parent_path(), test::data_dir());
}

}  // namespace alaya
