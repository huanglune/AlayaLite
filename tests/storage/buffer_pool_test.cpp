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

#include "storage/buffer/buffer_pool.hpp"
#include "storage/buffer/replacer/lru.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

namespace alaya {

// =============================================================================
// LRUReplacer Unit Tests
// =============================================================================

class LRUReplacerTest : public ::testing::Test {
 protected:
  static constexpr size_t kCapacity = 10;
};

TEST_F(LRUReplacerTest, Construction) {
  LRUReplacer replacer(kCapacity);
  EXPECT_EQ(replacer.size(), 0U);
}

TEST_F(LRUReplacerTest, DefaultConstruction) {
  LRUReplacer replacer;
  EXPECT_EQ(replacer.size(), 0U);
}

TEST_F(LRUReplacerTest, UnpinAddsToEvictable) {
  LRUReplacer replacer(kCapacity);

  // Initially empty
  EXPECT_EQ(replacer.size(), 0U);

  // Unpin frame 0
  replacer.unpin(0);
  EXPECT_EQ(replacer.size(), 1U);

  // Unpin frame 1
  replacer.unpin(1);
  EXPECT_EQ(replacer.size(), 2U);

  // Unpin same frame again - should not duplicate
  replacer.unpin(0);
  EXPECT_EQ(replacer.size(), 2U);
}

TEST_F(LRUReplacerTest, PinRemovesFromEvictable) {
  LRUReplacer replacer(kCapacity);

  replacer.unpin(0);
  replacer.unpin(1);
  replacer.unpin(2);
  EXPECT_EQ(replacer.size(), 3U);

  // Pin frame 1 - should remove from evictable
  replacer.pin(1);
  EXPECT_EQ(replacer.size(), 2U);

  // Pin non-existent frame - should be no-op
  replacer.pin(999);
  EXPECT_EQ(replacer.size(), 2U);
}

TEST_F(LRUReplacerTest, EvictReturnsLRU) {
  LRUReplacer replacer(kCapacity);

  // Add frames in order: 0, 1, 2
  // LRU list: front=[2, 1, 0]=back
  replacer.unpin(0);
  replacer.unpin(1);
  replacer.unpin(2);

  // First evict should return 0 (least recently used)
  auto victim = replacer.evict();
  ASSERT_TRUE(victim.has_value());
  EXPECT_EQ(victim.value(), 0U);
  EXPECT_EQ(replacer.size(), 2U);

  // Second evict should return 1
  victim = replacer.evict();
  ASSERT_TRUE(victim.has_value());
  EXPECT_EQ(victim.value(), 1U);

  // Third evict should return 2
  victim = replacer.evict();
  ASSERT_TRUE(victim.has_value());
  EXPECT_EQ(victim.value(), 2U);

  // Fourth evict should return nullopt
  victim = replacer.evict();
  EXPECT_FALSE(victim.has_value());
}

TEST_F(LRUReplacerTest, UnpinUpdatesLRUOrder) {
  LRUReplacer replacer(kCapacity);

  // Add frames: 0, 1, 2
  replacer.unpin(0);
  replacer.unpin(1);
  replacer.unpin(2);

  // Access frame 0 again - should move to front
  replacer.unpin(0);

  // Evict order should now be: 1, 2, 0
  auto victim = replacer.evict();
  EXPECT_EQ(victim.value(), 1U);

  victim = replacer.evict();
  EXPECT_EQ(victim.value(), 2U);

  victim = replacer.evict();
  EXPECT_EQ(victim.value(), 0U);
}

TEST_F(LRUReplacerTest, Remove) {
  LRUReplacer replacer(kCapacity);

  replacer.unpin(0);
  replacer.unpin(1);
  replacer.unpin(2);

  // Remove frame 1
  replacer.remove(1);
  EXPECT_EQ(replacer.size(), 2U);

  // Remove non-existent frame - should be no-op
  replacer.remove(999);
  EXPECT_EQ(replacer.size(), 2U);

  // Evict should skip removed frame
  auto victim = replacer.evict();
  EXPECT_EQ(victim.value(), 0U);

  victim = replacer.evict();
  EXPECT_EQ(victim.value(), 2U);
}

TEST_F(LRUReplacerTest, Reset) {
  LRUReplacer replacer(kCapacity);

  replacer.unpin(0);
  replacer.unpin(1);
  replacer.unpin(2);
  EXPECT_EQ(replacer.size(), 3U);

  replacer.reset();
  EXPECT_EQ(replacer.size(), 0U);

  // Evict should return nullopt
  auto victim = replacer.evict();
  EXPECT_FALSE(victim.has_value());
}

TEST_F(LRUReplacerTest, UnpinBeyondCapacity) {
  LRUReplacer replacer(3);

  // Frames beyond capacity should be ignored
  replacer.unpin(0);
  replacer.unpin(1);
  replacer.unpin(2);
  replacer.unpin(3);  // Beyond capacity
  replacer.unpin(4);  // Beyond capacity

  EXPECT_EQ(replacer.size(), 3U);
}

TEST_F(LRUReplacerTest, MoveConstruction) {
  LRUReplacer replacer1(kCapacity);
  replacer1.unpin(0);
  replacer1.unpin(1);

  LRUReplacer replacer2(std::move(replacer1));
  EXPECT_EQ(replacer2.size(), 2U);

  auto victim = replacer2.evict();
  EXPECT_EQ(victim.value(), 0U);
}

TEST_F(LRUReplacerTest, MoveAssignment) {
  LRUReplacer replacer1(kCapacity);
  replacer1.unpin(0);
  replacer1.unpin(1);

  LRUReplacer replacer2;
  replacer2 = std::move(replacer1);
  EXPECT_EQ(replacer2.size(), 2U);
}

TEST_F(LRUReplacerTest, SetCapacity) {
  LRUReplacer replacer;
  EXPECT_EQ(replacer.size(), 0U);

  replacer.set_capacity(5);
  replacer.unpin(0);
  replacer.unpin(1);
  EXPECT_EQ(replacer.size(), 2U);
}

// =============================================================================
// BufferPool Unit Tests
// =============================================================================

class BufferPoolTest : public ::testing::Test {
 protected:
  static constexpr size_t kFrameSize = 4096;  // 4KB frames
  static constexpr size_t kCapacity = 10;
};

TEST_F(BufferPoolTest, Construction) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize);

  EXPECT_EQ(pool.capacity(), kCapacity);
  EXPECT_EQ(pool.frame_size(), kFrameSize);
  EXPECT_EQ(pool.size(), 0U);
  EXPECT_TRUE(pool.is_enabled());
}

TEST_F(BufferPoolTest, DefaultConstruction) {
  BufferPool<uint32_t> pool;
  EXPECT_EQ(pool.capacity(), 0U);
  EXPECT_FALSE(pool.is_enabled());
}

TEST_F(BufferPoolTest, ZeroCapacity) {
  BufferPool<uint32_t> pool(0, kFrameSize);
  EXPECT_FALSE(pool.is_enabled());
  EXPECT_EQ(pool.capacity(), 0U);
}

TEST_F(BufferPoolTest, ZeroFrameSize) {
  BufferPool<uint32_t> pool(kCapacity, 0);
  // Should be effectively disabled
  EXPECT_EQ(pool.frame_size(), 0U);
}

TEST_F(BufferPoolTest, PutAndGet) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize);

  // Create test data
  std::vector<uint8_t> test_data(kFrameSize);
  for (size_t i = 0; i < kFrameSize; ++i) {
    test_data[i] = static_cast<uint8_t>(i % 256);
  }

  // Put data into cache
  const uint8_t* cached = pool.put(42, test_data.data());
  EXPECT_NE(cached, nullptr);
  EXPECT_EQ(pool.size(), 1U);

  // Get data back
  const uint8_t* retrieved = pool.get(42);
  EXPECT_NE(retrieved, nullptr);

  // Verify data matches
  for (size_t i = 0; i < kFrameSize; ++i) {
    EXPECT_EQ(retrieved[i], test_data[i]);
  }
}

TEST_F(BufferPoolTest, CacheMiss) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize);

  // Try to get non-existent item
  const uint8_t* result = pool.get(999);
  EXPECT_EQ(result, nullptr);

  auto stats = pool.stats();
  EXPECT_EQ(stats.misses_.load(), 1U);
  EXPECT_EQ(stats.hits_.load(), 0U);
}

TEST_F(BufferPoolTest, CacheHit) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize);

  std::vector<uint8_t> data(kFrameSize, 0xAB);
  pool.put(1, data.data());

  // Should be a cache hit
  const uint8_t* result = pool.get(1);
  EXPECT_NE(result, nullptr);

  auto stats = pool.stats();
  EXPECT_EQ(stats.hits_.load(), 1U);
  EXPECT_EQ(stats.misses_.load(), 0U);
}

TEST_F(BufferPoolTest, LRUEviction) {
  BufferPool<uint32_t> pool(3, kFrameSize);  // Small capacity

  std::vector<uint8_t> data(kFrameSize, 0);

  // Fill the cache
  for (uint32_t i = 0; i < 3; ++i) {
    data[0] = static_cast<uint8_t>(i);
    pool.put(i, data.data());
  }
  EXPECT_EQ(pool.size(), 3U);

  // Access item 0 to make it recently used
  (void)pool.get(0);

  // Insert new item - should evict item 1 (LRU)
  data[0] = 100;
  pool.put(100, data.data());

  // Item 0 should still be there (was accessed recently)
  EXPECT_TRUE(pool.contains(0));
  // Item 1 should be evicted (LRU)
  EXPECT_FALSE(pool.contains(1));
  // Item 2 should still be there
  EXPECT_TRUE(pool.contains(2));
  // New item should be there
  EXPECT_TRUE(pool.contains(100));

  auto stats = pool.stats();
  EXPECT_GT(stats.evictions_.load(), 0U);
}

TEST_F(BufferPoolTest, PutDuplicate) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize);

  std::vector<uint8_t> data1(kFrameSize, 0xAA);
  std::vector<uint8_t> data2(kFrameSize, 0xBB);

  // Put first data
  pool.put(1, data1.data());
  EXPECT_EQ(pool.size(), 1U);

  // Put with same key - should update, not add
  const uint8_t* cached = pool.put(1, data2.data());
  EXPECT_NE(cached, nullptr);
  EXPECT_EQ(pool.size(), 1U);

  // Data should still be original (no overwrite on duplicate put)
  const uint8_t* retrieved = pool.get(1);
  EXPECT_EQ(retrieved[0], 0xAA);
}

TEST_F(BufferPoolTest, Contains) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize);

  std::vector<uint8_t> data(kFrameSize, 0);

  EXPECT_FALSE(pool.contains(1));

  pool.put(1, data.data());
  EXPECT_TRUE(pool.contains(1));
  EXPECT_FALSE(pool.contains(2));
}

TEST_F(BufferPoolTest, Clear) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize);

  std::vector<uint8_t> data(kFrameSize, 0);
  for (uint32_t i = 0; i < 5; ++i) {
    pool.put(i, data.data());
  }
  EXPECT_EQ(pool.size(), 5U);

  pool.clear();
  EXPECT_EQ(pool.size(), 0U);

  // Items should no longer be found
  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_FALSE(pool.contains(i));
  }
}

TEST_F(BufferPoolTest, Statistics) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize);

  std::vector<uint8_t> data(kFrameSize, 0);
  pool.put(1, data.data());

  // Miss
  (void)pool.get(999);
  // Hit
  (void)pool.get(1);
  // Another hit
  (void)pool.get(1);

  auto stats = pool.stats();
  EXPECT_EQ(stats.hits_.load(), 2U);
  EXPECT_EQ(stats.misses_.load(), 1U);
  EXPECT_DOUBLE_EQ(stats.hit_rate(), 2.0 / 3.0);

  // Reset stats
  pool.reset_stats();
  stats = pool.stats();
  EXPECT_EQ(stats.hits_.load(), 0U);
  EXPECT_EQ(stats.misses_.load(), 0U);
}

TEST_F(BufferPoolTest, MoveConstruction) {
  BufferPool<uint32_t> pool1(kCapacity, kFrameSize);

  std::vector<uint8_t> data(kFrameSize);
  for (size_t i = 0; i < kFrameSize; ++i) {
    data[i] = static_cast<uint8_t>(i % 256);
  }
  pool1.put(42, data.data());
  pool1.put(43, data.data());

  BufferPool<uint32_t> pool2(std::move(pool1));
  EXPECT_EQ(pool2.size(), 2U);
  EXPECT_TRUE(pool2.contains(42));
  EXPECT_TRUE(pool2.contains(43));

  // Verify data integrity after move
  const uint8_t* retrieved = pool2.get(42);
  EXPECT_NE(retrieved, nullptr);
  for (size_t i = 0; i < kFrameSize; ++i) {
    EXPECT_EQ(retrieved[i], data[i]);
  }
}

TEST_F(BufferPoolTest, MoveAssignment) {
  BufferPool<uint32_t> pool1(kCapacity, kFrameSize);

  std::vector<uint8_t> data(kFrameSize, 0xCD);
  pool1.put(1, data.data());

  BufferPool<uint32_t> pool2;
  pool2 = std::move(pool1);

  EXPECT_EQ(pool2.size(), 1U);
  EXPECT_TRUE(pool2.contains(1));
}

TEST_F(BufferPoolTest, PutNullData) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize);

  const uint8_t* result = pool.put(1, nullptr);
  EXPECT_EQ(result, nullptr);
  EXPECT_EQ(pool.size(), 0U);
}

TEST_F(BufferPoolTest, GetReplacer) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize);

  std::vector<uint8_t> data(kFrameSize, 0);
  pool.put(1, data.data());
  pool.put(2, data.data());

  const auto& replacer = pool.get_replacer();
  // Replacer should track evictable frames
  EXPECT_GE(replacer.size(), 0U);
}

// =============================================================================
// BufferPoolStats Unit Tests
// =============================================================================

class BufferPoolStatsTest : public ::testing::Test {};

TEST_F(BufferPoolStatsTest, DefaultConstruction) {
  BufferPoolStats stats;
  EXPECT_EQ(stats.hits_.load(), 0U);
  EXPECT_EQ(stats.misses_.load(), 0U);
  EXPECT_EQ(stats.evictions_.load(), 0U);
}

TEST_F(BufferPoolStatsTest, HitRate) {
  BufferPoolStats stats;

  // No accesses - hit rate should be 0
  EXPECT_DOUBLE_EQ(stats.hit_rate(), 0.0);

  stats.hits_.store(3);
  stats.misses_.store(1);
  EXPECT_DOUBLE_EQ(stats.hit_rate(), 0.75);
}

TEST_F(BufferPoolStatsTest, TotalAccesses) {
  BufferPoolStats stats;
  stats.hits_.store(10);
  stats.misses_.store(5);

  EXPECT_EQ(stats.total_accesses(), 15U);
}

TEST_F(BufferPoolStatsTest, Reset) {
  BufferPoolStats stats;
  stats.hits_.store(10);
  stats.misses_.store(5);
  stats.evictions_.store(2);

  stats.reset();

  EXPECT_EQ(stats.hits_.load(), 0U);
  EXPECT_EQ(stats.misses_.load(), 0U);
  EXPECT_EQ(stats.evictions_.load(), 0U);
}

TEST_F(BufferPoolStatsTest, CopyConstruction) {
  BufferPoolStats stats1;
  stats1.hits_.store(10);
  stats1.misses_.store(5);
  stats1.evictions_.store(2);

  BufferPoolStats stats2(stats1);

  EXPECT_EQ(stats2.hits_.load(), 10U);
  EXPECT_EQ(stats2.misses_.load(), 5U);
  EXPECT_EQ(stats2.evictions_.load(), 2U);
}

TEST_F(BufferPoolStatsTest, CopyAssignment) {
  BufferPoolStats stats1;
  stats1.hits_.store(10);
  stats1.misses_.store(5);

  BufferPoolStats stats2;
  stats2 = stats1;

  EXPECT_EQ(stats2.hits_.load(), 10U);
  EXPECT_EQ(stats2.misses_.load(), 5U);
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

class BufferPoolThreadTest : public ::testing::Test {
 protected:
  static constexpr size_t kFrameSize = 1024;
  static constexpr size_t kCapacity = 100;
};

TEST_F(BufferPoolThreadTest, ConcurrentPut) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize);

  constexpr int kNumThreads = 4;
  constexpr int kOpsPerThread = 50;

  std::vector<std::thread> threads;

  threads.reserve(kNumThreads);
for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&pool, t]() -> void {
      std::vector<uint8_t> data(kFrameSize);
      for (int i = 0; i < kOpsPerThread; ++i) {
        auto id = static_cast<uint32_t>(t * kOpsPerThread + i);
        data[0] = static_cast<uint8_t>(id % 256);
        pool.put(id, data.data());
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // Pool should have some items (may be less than total due to evictions)
  EXPECT_GT(pool.size(), 0U);
  EXPECT_LE(pool.size(), kCapacity);
}

TEST_F(BufferPoolThreadTest, ConcurrentGetPut) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize);

  // Pre-populate
  std::vector<uint8_t> init_data(kFrameSize, 0xFF);
  for (uint32_t i = 0; i < 50; ++i) {
    pool.put(i, init_data.data());
  }

  constexpr int kNumThreads = 4;
  constexpr int kOpsPerThread = 100;

  std::atomic<int> hit_count{0};
  std::atomic<int> miss_count{0};

  std::vector<std::thread> threads;

  threads.reserve(kNumThreads);
for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&pool, &hit_count, &miss_count, t]() -> void {
      std::vector<uint8_t> data(kFrameSize);
      for (int i = 0; i < kOpsPerThread; ++i) {
        auto id = static_cast<uint32_t>((t * kOpsPerThread + i) % 200);

        if (i % 2 == 0) {
          // Get operation
          const uint8_t* result = pool.get(id);
          if (result != nullptr) {
            hit_count.fetch_add(1);
          } else {
            miss_count.fetch_add(1);
          }
        } else {
          // Put operation
          data[0] = static_cast<uint8_t>(id % 256);
          pool.put(id, data.data());
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // Should have had some operations
  EXPECT_GT(hit_count.load() + miss_count.load(), 0);
}

TEST_F(BufferPoolThreadTest, ConcurrentClear) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize);

  std::atomic<bool> done{false};

  // Writer thread
  std::thread writer([&pool, &done]() -> void {
    std::vector<uint8_t> data(kFrameSize, 0xAB);
    uint32_t id = 0;
    while (!done.load()) {
      pool.put(id++, data.data());
      if (id >= 1000) {
        id = 0;
      }
    }
  });

  // Clear thread
  std::thread clearer([&pool, &done]() -> void {
    for (int i = 0; i < 10; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      pool.clear();
    }
    done.store(true);
  });

  writer.join();
  clearer.join();

  // Should complete without crash
  EXPECT_LE(pool.size(), kCapacity);
}

}  // namespace alaya
