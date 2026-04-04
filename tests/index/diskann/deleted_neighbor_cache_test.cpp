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

#include "index/diskann/deleted_neighbor_cache.hpp"

namespace alaya {

class DeletedNeighborCacheTest : public ::testing::Test {
 protected:
  static constexpr size_t kDefaultCapacity = 4;
};

TEST_F(DeletedNeighborCacheTest, InsertAndLookup) {
  DeletedNeighborCache<uint32_t> cache(kDefaultCapacity);

  std::vector<uint32_t> neighbors = {1, 2, 3};
  cache.put(10, neighbors);

  auto result = cache.get(10);
  ASSERT_TRUE(result.has_value());
  auto &vec = result.value();  // NOLINT(bugprone-unchecked-optional-access)
  ASSERT_EQ(vec.size(), 3U);
  EXPECT_EQ(vec[0], 1U);
  EXPECT_EQ(vec[1], 2U);
  EXPECT_EQ(vec[2], 3U);
}

TEST_F(DeletedNeighborCacheTest, CacheMiss) {
  DeletedNeighborCache<uint32_t> cache(kDefaultCapacity);

  auto result = cache.get(999);
  EXPECT_FALSE(result.has_value());
}

TEST_F(DeletedNeighborCacheTest, LRUEvictionOrder) {
  DeletedNeighborCache<uint32_t> cache(3);

  // Insert 3 entries (fills capacity)
  cache.put(1, {10, 11});
  cache.put(2, {20, 21});
  cache.put(3, {30, 31});

  // Insert 4th entry - should evict node 1 (LRU)
  cache.put(4, {40, 41});

  EXPECT_FALSE(cache.get(1).has_value());  // Evicted
  EXPECT_TRUE(cache.get(2).has_value());
  EXPECT_TRUE(cache.get(3).has_value());
  EXPECT_TRUE(cache.get(4).has_value());
}

TEST_F(DeletedNeighborCacheTest, AccessRefreshesLRU) {
  DeletedNeighborCache<uint32_t> cache(3);

  cache.put(1, {10});
  cache.put(2, {20});
  cache.put(3, {30});

  // Access node 1 to refresh it (move to MRU)
  cache.get(1);

  // Insert 4th - should evict node 2 (now LRU, since 1 was refreshed)
  cache.put(4, {40});

  EXPECT_TRUE(cache.get(1).has_value());   // Refreshed, not evicted
  EXPECT_FALSE(cache.get(2).has_value());  // Evicted (was LRU)
  EXPECT_TRUE(cache.get(3).has_value());
  EXPECT_TRUE(cache.get(4).has_value());
}

TEST_F(DeletedNeighborCacheTest, CapacityBoundary) {
  DeletedNeighborCache<uint32_t> cache(2);

  cache.put(1, {10});
  cache.put(2, {20});
  EXPECT_EQ(cache.size(), 2U);

  // Insert at capacity boundary
  cache.put(3, {30});
  EXPECT_EQ(cache.size(), 2U);  // Should not exceed capacity

  EXPECT_FALSE(cache.get(1).has_value());
  EXPECT_TRUE(cache.get(2).has_value());
  EXPECT_TRUE(cache.get(3).has_value());
}

TEST_F(DeletedNeighborCacheTest, UpdateExistingEntry) {
  DeletedNeighborCache<uint32_t> cache(kDefaultCapacity);

  cache.put(1, {10, 11});
  cache.put(1, {20, 21, 22});  // Update

  auto result = cache.get(1);
  ASSERT_TRUE(result.has_value());
  auto &vec = result.value();  // NOLINT(bugprone-unchecked-optional-access)
  ASSERT_EQ(vec.size(), 3U);
  EXPECT_EQ(vec[0], 20U);
  EXPECT_EQ(cache.size(), 1U);
}

TEST_F(DeletedNeighborCacheTest, ContainsCheck) {
  DeletedNeighborCache<uint32_t> cache(kDefaultCapacity);

  EXPECT_FALSE(cache.contains(1));
  cache.put(1, {10});
  EXPECT_TRUE(cache.contains(1));
}

TEST_F(DeletedNeighborCacheTest, EmptyCache) {
  DeletedNeighborCache<uint32_t> cache(kDefaultCapacity);

  EXPECT_TRUE(cache.empty());
  EXPECT_EQ(cache.size(), 0U);
  EXPECT_EQ(cache.capacity(), kDefaultCapacity);
}

TEST_F(DeletedNeighborCacheTest, WithIndexCapacity) {
  // Default ratio is now 4%
  auto cache = DeletedNeighborCache<uint32_t>::with_index_capacity(100000);
  EXPECT_EQ(cache.capacity(), 4000U);

  auto cache_1m = DeletedNeighborCache<uint32_t>::with_index_capacity(1000000);
  EXPECT_EQ(cache_1m.capacity(), 40000U);

  // Small index should get minimum capacity of 64
  auto small_cache = DeletedNeighborCache<uint32_t>::with_index_capacity(100);
  EXPECT_EQ(small_cache.capacity(), 64U);
}

TEST_F(DeletedNeighborCacheTest, TwoHopCandidateCap) {
  // Verify the kMaxTwoHopCandidates constant is 5
  EXPECT_EQ(kMaxTwoHopCandidates, 5U);
}

TEST_F(DeletedNeighborCacheTest, EmptyNeighborList) {
  DeletedNeighborCache<uint32_t> cache(kDefaultCapacity);

  cache.put(1, {});
  auto result = cache.get(1);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value().size(), 0U);  // NOLINT(bugprone-unchecked-optional-access)
}

}  // namespace alaya
