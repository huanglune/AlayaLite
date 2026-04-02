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

#include <cstdint>
#include <vector>

#include "index/diskann/inserted_edge_cache.hpp"

namespace alaya {

class InsertedEdgeCacheTest : public ::testing::Test {};

TEST_F(InsertedEdgeCacheTest, ConstructionWithIndexCapacity) {
  auto cache = InsertedEdgeCache<uint32_t>::with_index_capacity(1000000);

  EXPECT_EQ(cache.num_shards(), 1024U);
  EXPECT_EQ(cache.shard_capacity(), 977U);
  EXPECT_EQ(cache.capacity(), 1024U * 977U);
  EXPECT_EQ(cache.size(), 0U);
}

TEST_F(InsertedEdgeCacheTest, AddWithinCapacity) {
  InsertedEdgeCache<uint32_t> cache(2, 4);

  cache.add(0, 10, 100);
  cache.add(0, 10, 101);
  cache.add(1, 20, 200);

  EXPECT_EQ(cache.size(), 2U);
  EXPECT_EQ(cache.consume(0, 10), (std::vector<uint32_t>{100, 101}));
  EXPECT_EQ(cache.consume(1, 20), (std::vector<uint32_t>{200}));
  EXPECT_EQ(cache.size(), 0U);
}

TEST_F(InsertedEdgeCacheTest, PerKeyCapEvictsOldestSource) {
  InsertedEdgeCache<uint32_t> cache(1, 4);

  for (uint32_t i = 0; i < InsertedEdgeCache<uint32_t>::kMaxEntriesPerKey + 2; ++i) {
    cache.add(0, 7, 100 + i);
  }

  auto values = cache.consume(0, 7);
  ASSERT_EQ(values.size(), InsertedEdgeCache<uint32_t>::kMaxEntriesPerKey);
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(values[i], 102U + i);
  }
}

TEST_F(InsertedEdgeCacheTest, PerShardLRUEvictsLeastRecentlyUsedKey) {
  InsertedEdgeCache<uint32_t> cache(1, 2);

  cache.add(0, 1, 10);
  cache.add(0, 2, 20);
  cache.add(0, 1, 11);
  cache.add(0, 3, 30);

  EXPECT_TRUE(cache.consume(0, 2).empty());
  EXPECT_EQ(cache.consume(0, 1), (std::vector<uint32_t>{10, 11}));
  EXPECT_EQ(cache.consume(0, 3), (std::vector<uint32_t>{30}));
}

TEST_F(InsertedEdgeCacheTest, ConsumeErasesKeyAndMissingKeyReturnsEmpty) {
  InsertedEdgeCache<uint32_t> cache(1, 2);

  EXPECT_TRUE(cache.consume(0, 99).empty());

  cache.add(0, 1, 10);
  EXPECT_EQ(cache.size(), 1U);
  EXPECT_EQ(cache.consume(0, 1), (std::vector<uint32_t>{10}));
  EXPECT_TRUE(cache.consume(0, 1).empty());
  EXPECT_EQ(cache.size(), 0U);
}

TEST_F(InsertedEdgeCacheTest, EmptyShardReleasesBucketMemory) {
  InsertedEdgeCache<uint32_t> cache(1, 8);

  size_t empty_bucket_count = cache.bucket_count(0);
  for (uint32_t target = 0; target < 6; ++target) {
    cache.add(0, target, 100 + target);
  }

  EXPECT_GE(cache.bucket_count(0), empty_bucket_count);
  for (uint32_t target = 0; target < 6; ++target) {
    EXPECT_EQ(cache.consume(0, target), (std::vector<uint32_t>{100 + target}));
  }

  EXPECT_EQ(cache.size(), 0U);
  EXPECT_EQ(cache.bucket_count(0), empty_bucket_count);
}

}  // namespace alaya
