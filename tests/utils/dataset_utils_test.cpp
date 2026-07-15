// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <string>

#include "utils/dataset_utils.hpp"

namespace alaya {

TEST(DatasetTest, SiftMicroConfig) {
  auto config = sift_micro();

  EXPECT_EQ(config.name, "siftmicro");
  EXPECT_EQ(config.max_data_num, 1000);
  EXPECT_EQ(config.max_query_num, 50);
  EXPECT_TRUE(config.data_file.string().find("siftsmall") != std::string::npos);
}

TEST(DatasetTest, NoArgOverloadsUseTestDataDir) {
  auto ss = sift_small();
  auto sm = sift_micro();
  auto d1 = deep1m();

  EXPECT_EQ(ss.dir.parent_path(), test::data_dir());
  EXPECT_EQ(sm.dir.parent_path(), test::data_dir());
  EXPECT_EQ(d1.dir.parent_path(), test::data_dir());
}

}  // namespace alaya
