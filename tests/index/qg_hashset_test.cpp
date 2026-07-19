// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include "index/graph/detail/build_support/hashset.hpp"

namespace alaya {
namespace {

TEST(QgBuilderHashSetTest, HandlesCollisionsAndClear) {
  HashBasedBooleanSet visited;
  visited.initialize(8);

  visited.set(1);
  visited.set(9);
  visited.set(17);

  EXPECT_TRUE(visited.get(1));
  EXPECT_TRUE(visited.get(9));
  EXPECT_TRUE(visited.get(17));
  EXPECT_FALSE(visited.get(2));

  visited.clear();

  EXPECT_FALSE(visited.get(1));
  EXPECT_FALSE(visited.get(9));
  EXPECT_FALSE(visited.get(17));
}

}  // namespace
}  // namespace alaya
