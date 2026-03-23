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
#include "storage/buffer/replacer/clock.hpp"
#include "storage/buffer/replacer/clock_pro.hpp"
#include "storage/buffer/replacer/lru.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "utils/memory.hpp"

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

  EXPECT_EQ(replacer.size(), 0U);

  replacer.unpin(0);
  EXPECT_EQ(replacer.size(), 1U);

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
  EXPECT_EQ(*victim, 0U);  // NOLINT(bugprone-unchecked-optional-access)
  EXPECT_EQ(replacer.size(), 2U);

  // Second evict should return 1
  victim = replacer.evict();
  ASSERT_TRUE(victim.has_value());
  EXPECT_EQ(*victim, 1U);  // NOLINT(bugprone-unchecked-optional-access)

  // Third evict should return 2
  victim = replacer.evict();
  ASSERT_TRUE(victim.has_value());
  EXPECT_EQ(*victim, 2U);  // NOLINT(bugprone-unchecked-optional-access)

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
  ASSERT_TRUE(victim.has_value());
  EXPECT_EQ(*victim, 1U);  // NOLINT(bugprone-unchecked-optional-access)

  victim = replacer.evict();
  ASSERT_TRUE(victim.has_value());
  EXPECT_EQ(*victim, 2U);  // NOLINT(bugprone-unchecked-optional-access)

  victim = replacer.evict();
  ASSERT_TRUE(victim.has_value());
  EXPECT_EQ(*victim, 0U);  // NOLINT(bugprone-unchecked-optional-access)
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
  ASSERT_TRUE(victim.has_value());
  EXPECT_EQ(*victim, 0U);  // NOLINT(bugprone-unchecked-optional-access)

  victim = replacer.evict();
  ASSERT_TRUE(victim.has_value());
  EXPECT_EQ(*victim, 2U);  // NOLINT(bugprone-unchecked-optional-access)
}

TEST_F(LRUReplacerTest, Reset) {
  LRUReplacer replacer(kCapacity);

  replacer.unpin(0);
  replacer.unpin(1);
  replacer.unpin(2);
  EXPECT_EQ(replacer.size(), 3U);

  replacer.reset();
  EXPECT_EQ(replacer.size(), 0U);

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
  ASSERT_TRUE(victim.has_value());
  EXPECT_EQ(*victim, 0U);  // NOLINT(bugprone-unchecked-optional-access)
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
// ClockReplacer Unit Tests
// =============================================================================

class ClockReplacerTest : public ::testing::Test {
 protected:
  static constexpr size_t kCapacity = 10;
};

TEST_F(ClockReplacerTest, Construction) {
  ClockReplacer replacer(kCapacity);
  EXPECT_EQ(replacer.size(), 0U);
}

TEST_F(ClockReplacerTest, DefaultConstruction) {
  ClockReplacer replacer;
  EXPECT_EQ(replacer.size(), 0U);
}

TEST_F(ClockReplacerTest, UnpinAddsToEvictable) {
  ClockReplacer replacer(kCapacity);

  EXPECT_EQ(replacer.size(), 0U);

  replacer.unpin(0);
  EXPECT_EQ(replacer.size(), 1U);

  replacer.unpin(1);
  EXPECT_EQ(replacer.size(), 2U);

  // Unpin same frame again - should not duplicate
  replacer.unpin(0);
  EXPECT_EQ(replacer.size(), 2U);
}

TEST_F(ClockReplacerTest, PinRemovesFromEvictable) {
  ClockReplacer replacer(kCapacity);

  replacer.unpin(0);
  replacer.unpin(1);
  replacer.unpin(2);
  EXPECT_EQ(replacer.size(), 3U);

  replacer.pin(1);
  EXPECT_EQ(replacer.size(), 2U);

  // Pin non-existent frame - should be no-op
  replacer.pin(999);
  EXPECT_EQ(replacer.size(), 2U);
}

TEST_F(ClockReplacerTest, EvictWithSecondChance) {
  ClockReplacer replacer(kCapacity);

  // Add frames
  replacer.unpin(0);
  replacer.unpin(1);
  replacer.unpin(2);

  // All have ref_bit=true initially, first evict gives second chance
  auto victim = replacer.evict();
  ASSERT_TRUE(victim.has_value());
  EXPECT_EQ(replacer.size(), 2U);

  // Continue evicting
  victim = replacer.evict();
  ASSERT_TRUE(victim.has_value());

  victim = replacer.evict();
  ASSERT_TRUE(victim.has_value());

  // No more to evict
  victim = replacer.evict();
  EXPECT_FALSE(victim.has_value());
}

TEST_F(ClockReplacerTest, UnpinSetsRefBit) {
  ClockReplacer replacer(kCapacity);

  replacer.unpin(0);
  replacer.unpin(1);

  // Evict once to clear ref bits via second chance
  auto victim = replacer.evict();
  ASSERT_TRUE(victim.has_value());
  size_t vid = *victim;  // NOLINT(bugprone-unchecked-optional-access)

  // Access remaining frame - sets ref bit again
  size_t remaining = (vid == 0) ? 1 : 0;
  replacer.unpin(remaining);

  // Should still be evictable (but with ref bit set)
  EXPECT_EQ(replacer.size(), 1U);
}

TEST_F(ClockReplacerTest, Remove) {
  ClockReplacer replacer(kCapacity);

  replacer.unpin(0);
  replacer.unpin(1);
  replacer.unpin(2);

  replacer.remove(1);
  EXPECT_EQ(replacer.size(), 2U);

  // Remove non-existent - no-op
  replacer.remove(999);
  EXPECT_EQ(replacer.size(), 2U);
}

TEST_F(ClockReplacerTest, Reset) {
  ClockReplacer replacer(kCapacity);

  replacer.unpin(0);
  replacer.unpin(1);
  EXPECT_EQ(replacer.size(), 2U);

  replacer.reset();
  EXPECT_EQ(replacer.size(), 0U);

  auto victim = replacer.evict();
  EXPECT_FALSE(victim.has_value());
}

TEST_F(ClockReplacerTest, MoveConstruction) {
  ClockReplacer replacer1(kCapacity);
  replacer1.unpin(0);
  replacer1.unpin(1);

  ClockReplacer replacer2(std::move(replacer1));
  EXPECT_EQ(replacer2.size(), 2U);
}

TEST_F(ClockReplacerTest, MoveAssignment) {
  ClockReplacer replacer1(kCapacity);
  replacer1.unpin(0);
  replacer1.unpin(1);

  ClockReplacer replacer2;
  replacer2 = std::move(replacer1);
  EXPECT_EQ(replacer2.size(), 2U);
}

TEST_F(ClockReplacerTest, SetCapacity) {
  ClockReplacer replacer;
  replacer.set_capacity(5);
  replacer.unpin(0);
  replacer.unpin(1);
  EXPECT_EQ(replacer.size(), 2U);
}

// =============================================================================
// ClockProReplacer Unit Tests
// =============================================================================

class ClockProReplacerTest : public ::testing::Test {
 protected:
  static constexpr size_t kCapacity = 10;
};

TEST_F(ClockProReplacerTest, Construction) {
  ClockProReplacer replacer(kCapacity);
  EXPECT_EQ(replacer.size(), 0U);
  EXPECT_EQ(replacer.hot_size(), 0U);
  EXPECT_EQ(replacer.cold_size(), 0U);
}

TEST_F(ClockProReplacerTest, DefaultConstruction) {
  ClockProReplacer replacer;
  EXPECT_EQ(replacer.size(), 0U);
}

TEST_F(ClockProReplacerTest, UnpinAddsToCold) {
  ClockProReplacer replacer(kCapacity);

  replacer.unpin(0);
  EXPECT_EQ(replacer.size(), 1U);
  EXPECT_EQ(replacer.cold_size(), 1U);
  EXPECT_EQ(replacer.hot_size(), 0U);

  replacer.unpin(1);
  EXPECT_EQ(replacer.size(), 2U);
  EXPECT_EQ(replacer.cold_size(), 2U);
}

TEST_F(ClockProReplacerTest, PinRemoves) {
  ClockProReplacer replacer(kCapacity);

  replacer.unpin(0);
  replacer.unpin(1);
  EXPECT_EQ(replacer.size(), 2U);

  replacer.pin(0);
  EXPECT_EQ(replacer.size(), 1U);
}

TEST_F(ClockProReplacerTest, SingleEntryPinUnpinCycleRemainsValid) {
  ClockProReplacer replacer(1);

  for (int i = 0; i < 32; ++i) {
    replacer.unpin(0);
    EXPECT_EQ(replacer.size(), 1U);
    replacer.pin(0);
    EXPECT_EQ(replacer.size(), 0U);
  }

  replacer.unpin(0);
  auto victim = replacer.evict();
  ASSERT_TRUE(victim.has_value());
  EXPECT_EQ(*victim, 0U);  // NOLINT(bugprone-unchecked-optional-access)
  EXPECT_EQ(replacer.size(), 0U);
}

TEST_F(ClockProReplacerTest, EvictFromCold) {
  ClockProReplacer replacer(kCapacity);

  replacer.unpin(0);
  replacer.unpin(1);
  replacer.unpin(2);

  auto victim = replacer.evict();
  ASSERT_TRUE(victim.has_value());
  EXPECT_EQ(replacer.size(), 2U);

  // Should be able to continue evicting
  victim = replacer.evict();
  ASSERT_TRUE(victim.has_value());

  victim = replacer.evict();
  ASSERT_TRUE(victim.has_value());

  victim = replacer.evict();
  EXPECT_FALSE(victim.has_value());
}

TEST_F(ClockProReplacerTest, TestSetPromotion) {
  ClockProReplacer replacer(kCapacity);

  // Add and evict a page
  replacer.unpin(0);
  auto victim = replacer.evict();
  ASSERT_TRUE(victim.has_value());
  size_t evicted_id = *victim;  // NOLINT(bugprone-unchecked-optional-access)

  // The evicted page is now in test set
  EXPECT_EQ(replacer.test_size(), 1U);

  // Access the evicted page again - should promote to hot
  replacer.unpin(evicted_id);
  EXPECT_EQ(replacer.hot_size(), 1U);
  EXPECT_EQ(replacer.test_size(), 0U);
}

TEST_F(ClockProReplacerTest, Remove) {
  ClockProReplacer replacer(kCapacity);

  replacer.unpin(0);
  replacer.unpin(1);

  replacer.remove(0);
  EXPECT_EQ(replacer.size(), 1U);
}

TEST_F(ClockProReplacerTest, Reset) {
  ClockProReplacer replacer(kCapacity);

  replacer.unpin(0);
  replacer.unpin(1);

  replacer.reset();
  EXPECT_EQ(replacer.size(), 0U);
  EXPECT_EQ(replacer.cold_size(), 0U);
  EXPECT_EQ(replacer.hot_size(), 0U);
  EXPECT_EQ(replacer.test_size(), 0U);
}

TEST_F(ClockProReplacerTest, MoveConstruction) {
  ClockProReplacer replacer1(kCapacity);
  replacer1.unpin(0);
  replacer1.unpin(1);

  ClockProReplacer replacer2(std::move(replacer1));
  EXPECT_EQ(replacer2.size(), 2U);
}

TEST_F(ClockProReplacerTest, SetCapacity) {
  ClockProReplacer replacer;
  replacer.set_capacity(5);
  replacer.unpin(0);
  replacer.unpin(1);
  EXPECT_EQ(replacer.size(), 2U);
}

// =============================================================================
// BufferPoolStats Unit Tests
// =============================================================================

class BufferPoolStatsTest : public ::testing::Test {};

TEST_F(BufferPoolStatsTest, DefaultConstruction) {
  BufferPoolStats stats;
  EXPECT_EQ(stats.hits_.load(), 0U);
  EXPECT_EQ(stats.misses_.load(), 0U);
  EXPECT_EQ(stats.reuse_hits_.load(), 0U);
  EXPECT_EQ(stats.reuse_misses_.load(), 0U);
  EXPECT_EQ(stats.evictions_.load(), 0U);
  EXPECT_EQ(stats.pins_.load(), 0U);
}

TEST_F(BufferPoolStatsTest, Reset) {
  BufferPoolStats stats;
  stats.hits_.store(10);
  stats.misses_.store(5);
  stats.reuse_hits_.store(7);
  stats.reuse_misses_.store(4);
  stats.evictions_.store(2);
  stats.pins_.store(3);

  stats.reset();

  EXPECT_EQ(stats.hits_.load(), 0U);
  EXPECT_EQ(stats.misses_.load(), 0U);
  EXPECT_EQ(stats.reuse_hits_.load(), 0U);
  EXPECT_EQ(stats.reuse_misses_.load(), 0U);
  EXPECT_EQ(stats.evictions_.load(), 0U);
  EXPECT_EQ(stats.pins_.load(), 0U);
}

// =============================================================================
// BufferPool Unit Tests (with LRU replacer, default)
// =============================================================================

class BufferPoolTest : public ::testing::Test {
 protected:
  static constexpr size_t kFrameSize = 4096;
  static constexpr size_t kNumPages = 20;
  static constexpr size_t kCapacity = 10;

  std::string test_file_;

  void SetUp() override {
    test_file_ = "/tmp/buffer_pool_test_" +
                 std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + ".bin";
    create_test_file();
  }

  void TearDown() override {
    if (std::filesystem::exists(test_file_)) {
      std::filesystem::remove(test_file_);
    }
  }

  // Create a test file with kNumPages pages.
  // Each page is filled with its page index (page 0 -> all 0x00, page 3 -> all 0x03, etc.)
  void create_test_file() {
    size_t total_size = kFrameSize * kNumPages;
    AlignedBuffer buf(total_size);
    for (size_t i = 0; i < kNumPages; ++i) {
      std::memset(buf.data() + i * kFrameSize, static_cast<int>(i & 0xFF), kFrameSize);
    }
    std::ofstream ofs(test_file_, std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(buf.data()),
              static_cast<std::streamsize>(total_size));
    ofs.close();
  }
};

TEST_F(BufferPoolTest, Construction) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 2);

  EXPECT_EQ(pool.stats().hits_.load(), 0U);
  EXPECT_EQ(pool.stats().misses_.load(), 0U);
  EXPECT_EQ(pool.stats().evictions_.load(), 0U);
}

TEST_F(BufferPoolTest, ZeroCapacity) {
  EXPECT_THROW(
      {
        BufferPool<uint32_t> pool(0, kFrameSize);
      },
      std::invalid_argument);
}

TEST_F(BufferPoolTest, NumShardsGreaterThanCapacity) {
  BufferPool<uint32_t> pool(1, kFrameSize, 16);
  std::vector<uint8_t> data(kFrameSize, 0x5A);

  auto handle = pool.put(7, data.data());

  ASSERT_FALSE(handle.empty());
  EXPECT_EQ(handle.size(), kFrameSize);
  EXPECT_EQ(handle.data()[0], 0x5A);
}

TEST_F(BufferPoolTest, SingleShard) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 1);
  EXPECT_EQ(pool.stats().hits_.load(), 0U);
}

TEST_F(BufferPoolTest, GetOnEmptyPoolReturnsEmptyHandle) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 2);

  auto handle = pool.get(42);
  EXPECT_TRUE(handle.empty());
  EXPECT_EQ(handle.data(), nullptr);
  EXPECT_EQ(handle.size(), 0U);
}

TEST_F(BufferPoolTest, GetOrReadCacheMiss) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  // Read page 0 - should be a cache miss then load from disk
  auto handle = pool.get_or_read(0, io, 0, temp.data());
  EXPECT_FALSE(handle.empty());
  EXPECT_NE(handle.data(), nullptr);
  EXPECT_EQ(handle.size(), kFrameSize);

  // Page 0 is filled with 0x00
  for (size_t i = 0; i < kFrameSize; ++i) {
    EXPECT_EQ(handle.data()[i], 0x00);
  }

  EXPECT_EQ(pool.stats().misses_.load(), 1U);
  EXPECT_EQ(pool.stats().hits_.load(), 0U);
}

TEST_F(BufferPoolTest, GetOrReadCacheHit) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  // First read - cache miss
  {
    auto handle = pool.get_or_read(0, io, 0, temp.data());
    EXPECT_FALSE(handle.empty());
  }  // handle released, page unpinned

  // Second read - should be a cache hit (fast path in get_or_read)
  auto handle = pool.get_or_read(0, io, 0, temp.data());
  EXPECT_FALSE(handle.empty());
  EXPECT_EQ(handle.data()[0], 0x00);

  EXPECT_EQ(pool.stats().hits_.load(), 1U);
  EXPECT_EQ(pool.stats().misses_.load(), 1U);
}

TEST_F(BufferPoolTest, GetAfterGetOrRead) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  // Load page 5 via get_or_read
  {
    auto handle = pool.get_or_read(5, io, 5 * kFrameSize, temp.data());
    EXPECT_FALSE(handle.empty());
  }

  // get() should now find the cached page
  auto handle = pool.get(5);
  EXPECT_FALSE(handle.empty());
  EXPECT_EQ(handle.data()[0], 5);

  EXPECT_EQ(pool.stats().hits_.load(), 1U);
}

TEST_F(BufferPoolTest, MultiplePages) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  for (uint32_t i = 0; i < 5; ++i) {
    auto handle = pool.get_or_read(i, io, i * kFrameSize, temp.data());
    EXPECT_FALSE(handle.empty());
    EXPECT_EQ(handle.data()[0], static_cast<uint8_t>(i));
  }

  EXPECT_EQ(pool.stats().misses_.load(), 5U);
}

TEST_F(BufferPoolTest, PageHandleDataIntegrity) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  auto handle = pool.get_or_read(7, io, 7 * kFrameSize, temp.data());
  EXPECT_FALSE(handle.empty());

  // Check via view()
  auto span = handle.view();
  EXPECT_EQ(span.size(), kFrameSize);
  for (auto byte : span) {
    EXPECT_EQ(byte, 7);
  }
}

TEST_F(BufferPoolTest, PageHandleMutableData) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  auto handle = pool.get_or_read(1, io, kFrameSize, temp.data());
  EXPECT_FALSE(handle.empty());

  // Modify data through mutable_data
  handle.mutable_data()[0] = 0xFF;
  EXPECT_EQ(handle.data()[0], 0xFF);

  // Re-fetch from cache - should see the modification
  // First release current handle
  auto *modified_ptr = handle.data();
  {
    // get() should return the same cached frame
    auto handle2 = pool.get(1);
    EXPECT_FALSE(handle2.empty());
    EXPECT_EQ(handle2.data()[0], 0xFF);
    // Should point to same underlying frame data
    EXPECT_EQ(handle2.data(), modified_ptr);
  }
}

TEST_F(BufferPoolTest, PageHandleDefaultEmpty) {
  BufferPool<uint32_t>::PageHandle handle;
  EXPECT_TRUE(handle.empty());
  EXPECT_EQ(handle.data(), nullptr);
  EXPECT_EQ(handle.size(), 0U);
}

TEST_F(BufferPoolTest, PageHandleMoveConstruction) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  auto handle1 = pool.get_or_read(2, io, 2 * kFrameSize, temp.data());
  EXPECT_FALSE(handle1.empty());

  auto handle2 = std::move(handle1);
  EXPECT_TRUE(handle1.empty());   // NOLINT(bugprone-use-after-move)
  EXPECT_FALSE(handle2.empty());
  EXPECT_EQ(handle2.data()[0], 2);
}

TEST_F(BufferPoolTest, PageHandleMoveAssignment) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  auto handle1 = pool.get_or_read(3, io, 3 * kFrameSize, temp.data());
  decltype(handle1) handle2;
  EXPECT_TRUE(handle2.empty());

  handle2 = std::move(handle1);
  EXPECT_TRUE(handle1.empty());  // NOLINT(bugprone-use-after-move)
  EXPECT_FALSE(handle2.empty());
  EXPECT_EQ(handle2.data()[0], 3);
}

TEST_F(BufferPoolTest, PageHandleSelfMoveAssignment) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  auto handle = pool.get_or_read(4, io, 4 * kFrameSize, temp.data());
  EXPECT_FALSE(handle.empty());

  auto *self = &handle;
  handle = std::move(*self);  // NOLINT(clang-diagnostic-self-move)
  // Should survive self-move without crash
  EXPECT_FALSE(handle.empty());
}

TEST_F(BufferPoolTest, Eviction) {
  // Small pool: 3 frames, single shard for deterministic behavior
  BufferPool<uint32_t> pool(3, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  // Fill the cache with pages 0, 1, 2
  for (uint32_t i = 0; i < 3; ++i) {
    auto handle = pool.get_or_read(i, io, i * kFrameSize, temp.data());
    EXPECT_FALSE(handle.empty());
  }

  // Load page 3 - should trigger eviction of the LRU page
  {
    auto handle = pool.get_or_read(3, io, 3 * kFrameSize, temp.data());
    EXPECT_FALSE(handle.empty());
    EXPECT_EQ(handle.data()[0], 3);
  }

  EXPECT_GT(pool.stats().evictions_.load(), 0U);
}

TEST_F(BufferPoolTest, EvictionRespectsAccessOrder) {
  // 3 frames, single shard, LRU policy
  BufferPool<uint32_t> pool(3, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  // Load pages 0, 1, 2
  for (uint32_t i = 0; i < 3; ++i) {
    auto handle = pool.get_or_read(i, io, i * kFrameSize, temp.data());
    EXPECT_FALSE(handle.empty());
  }

  // Access page 0 again to make it recently used
  {
    auto handle = pool.get_or_read(0, io, 0, temp.data());
    EXPECT_FALSE(handle.empty());
  }

  // Insert page 3 - should evict page 1 (LRU among 0,1,2 after accessing 0)
  {
    auto handle = pool.get_or_read(3, io, 3 * kFrameSize, temp.data());
    EXPECT_FALSE(handle.empty());
  }

  // Page 0 should still be cached (was recently accessed)
  {
    auto handle = pool.get(0);
    EXPECT_FALSE(handle.empty());
    EXPECT_EQ(handle.data()[0], 0);
  }

  // Page 3 should be cached
  {
    auto handle = pool.get(3);
    EXPECT_FALSE(handle.empty());
    EXPECT_EQ(handle.data()[0], 3);
  }
}

TEST_F(BufferPoolTest, PinnedPageNotEvicted) {
  // Pool with 2 frames, 1 shard
  BufferPool<uint32_t> pool(2, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  // Load and keep page 0 pinned (hold the handle)
  auto pinned = pool.get_or_read(0, io, 0, temp.data());
  EXPECT_FALSE(pinned.empty());

  // Load page 1 (fills cache)
  {
    auto handle = pool.get_or_read(1, io, kFrameSize, temp.data());
    EXPECT_FALSE(handle.empty());
  }

  // Load page 2 - should evict page 1 (not page 0, which is pinned)
  {
    auto handle = pool.get_or_read(2, io, 2 * kFrameSize, temp.data());
    EXPECT_FALSE(handle.empty());
    EXPECT_EQ(handle.data()[0], 2);
  }

  // Page 0 should still be valid and accessible
  EXPECT_EQ(pinned.data()[0], 0);
}

TEST_F(BufferPoolTest, AllPagesPinnedEvictionFails) {
  // Pool with 2 frames, 1 shard
  BufferPool<uint32_t> pool(2, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  // Pin both frames
  auto h0 = pool.get_or_read(0, io, 0, temp.data());
  auto h1 = pool.get_or_read(1, io, kFrameSize, temp.data());
  EXPECT_FALSE(h0.empty());
  EXPECT_FALSE(h1.empty());

  // Try to load page 2 - should fail because no frames are evictable
  auto h2 = pool.get_or_read(2, io, 2 * kFrameSize, temp.data());
  EXPECT_TRUE(h2.empty());
}

TEST_F(BufferPoolTest, DuplicateInsertDeduplication) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  // Load page 5
  {
    auto handle = pool.get_or_read(5, io, 5 * kFrameSize, temp.data());
    EXPECT_FALSE(handle.empty());
  }

  // Load page 5 again - should be a cache hit, not duplicate insertion
  {
    auto handle = pool.get_or_read(5, io, 5 * kFrameSize, temp.data());
    EXPECT_FALSE(handle.empty());
    EXPECT_EQ(handle.data()[0], 5);
  }

  EXPECT_EQ(pool.stats().hits_.load(), 1U);
  EXPECT_EQ(pool.stats().misses_.load(), 1U);
}

TEST_F(BufferPoolTest, MultipleHandlesToSamePage) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  // Load page 1
  auto handle1 = pool.get_or_read(1, io, kFrameSize, temp.data());
  EXPECT_FALSE(handle1.empty());

  // Get another handle to the same page
  auto handle2 = pool.get(1);
  EXPECT_FALSE(handle2.empty());

  // Both should point to the same data
  EXPECT_EQ(handle1.data(), handle2.data());
}

TEST_F(BufferPoolTest, Clear) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  // Load several pages (release handles so they can be cleared)
  for (uint32_t i = 0; i < 5; ++i) {
    auto handle = pool.get_or_read(i, io, i * kFrameSize, temp.data());
    EXPECT_FALSE(handle.empty());
  }

  pool.clear();

  // Stats should be reset
  EXPECT_EQ(pool.stats().hits_.load(), 0U);
  EXPECT_EQ(pool.stats().misses_.load(), 0U);
  EXPECT_EQ(pool.stats().reuse_hits_.load(), 0U);
  EXPECT_EQ(pool.stats().reuse_misses_.load(), 0U);
  EXPECT_EQ(pool.stats().evictions_.load(), 0U);

  // Previously cached pages should no longer be found
  auto handle = pool.get(0);
  EXPECT_TRUE(handle.empty());

  handle = pool.get(4);
  EXPECT_TRUE(handle.empty());
}

TEST_F(BufferPoolTest, ClearWithActivePageHandlesThrows) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  auto handle = pool.get_or_read(0, io, 0, temp.data());
  ASSERT_FALSE(handle.empty());

  EXPECT_THROW(pool.clear(), std::logic_error);

  handle = BufferPool<uint32_t>::PageHandle{};
  EXPECT_NO_THROW(pool.clear());
}

TEST_F(BufferPoolTest, PrefetchUsesCachedPageOnSecondAccess) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);

  pool.prefetch(0, io, 0);
  pool.prefetch(0, io, 0);

  EXPECT_EQ(pool.stats().misses_.load(), 1U);
  EXPECT_EQ(pool.stats().hits_.load(), 1U);
  EXPECT_EQ(pool.stats().reuse_misses_.load(), 1U);
  EXPECT_EQ(pool.stats().reuse_hits_.load(), 1U);
}

TEST_F(BufferPoolTest, ResetStatsKeepsCachedPages) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  {
    auto handle = pool.get_or_read(0, io, 0, temp.data());
    EXPECT_FALSE(handle.empty());
  }
  {
    auto handle = pool.get(0);
    EXPECT_FALSE(handle.empty());
  }

  EXPECT_EQ(pool.stats().misses_.load(), 1U);
  EXPECT_EQ(pool.stats().hits_.load(), 1U);

  pool.reset_stats();

  EXPECT_EQ(pool.stats().hits_.load(), 0U);
  EXPECT_EQ(pool.stats().misses_.load(), 0U);
  EXPECT_EQ(pool.stats().reuse_hits_.load(), 0U);
  EXPECT_EQ(pool.stats().reuse_misses_.load(), 0U);

  auto handle = pool.get(0);
  EXPECT_FALSE(handle.empty());
  EXPECT_EQ(pool.stats().hits_.load(), 1U);
  EXPECT_EQ(pool.stats().misses_.load(), 0U);
}

TEST_F(BufferPoolTest, StatsTracking) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  // Miss: load page 0
  {
    auto handle = pool.get_or_read(0, io, 0, temp.data());
    EXPECT_FALSE(handle.empty());
  }

  // Hit: re-read page 0
  {
    auto handle = pool.get_or_read(0, io, 0, temp.data());
    EXPECT_FALSE(handle.empty());
  }

  // Hit: get page 0
  {
    auto handle = pool.get(0);
    EXPECT_FALSE(handle.empty());
  }

  // get() on non-existent page does NOT record a miss
  {
    auto handle = pool.get(999);
    EXPECT_TRUE(handle.empty());
  }

  EXPECT_EQ(pool.stats().misses_.load(), 1U);
  EXPECT_EQ(pool.stats().hits_.load(), 2U);
}

TEST_F(BufferPoolTest, ReadFailureReturnsEmptyHandle) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  // Read at an offset beyond the file size - should fail
  uint64_t bad_offset = kNumPages * kFrameSize + kFrameSize;
  auto handle = pool.get_or_read(99, io, bad_offset, temp.data());
  EXPECT_TRUE(handle.empty());
}

TEST_F(BufferPoolTest, MultipleShardsDistributePages) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 2);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  // Load several pages - they should be distributed across shards
  for (uint32_t i = 0; i < 8; ++i) {
    auto handle = pool.get_or_read(i, io, i * kFrameSize, temp.data());
    EXPECT_FALSE(handle.empty());
    EXPECT_EQ(handle.data()[0], static_cast<uint8_t>(i));
  }

  // All pages should be retrievable
  for (uint32_t i = 0; i < 8; ++i) {
    auto handle = pool.get(i);
    EXPECT_FALSE(handle.empty());
    EXPECT_EQ(handle.data()[0], static_cast<uint8_t>(i));
  }
}

// =============================================================================
// BufferPool with ClockReplacer Tests
// =============================================================================

class BufferPoolClockTest : public BufferPoolTest {};

TEST_F(BufferPoolClockTest, BasicOperations) {
  BufferPool<uint32_t, ClockReplacer> pool(kCapacity, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  // Cache miss then hit
  {
    auto handle = pool.get_or_read(1, io, kFrameSize, temp.data());
    EXPECT_FALSE(handle.empty());
    EXPECT_EQ(handle.data()[0], 1);
  }

  auto handle = pool.get(1);
  EXPECT_FALSE(handle.empty());
  EXPECT_EQ(handle.data()[0], 1);
}

TEST_F(BufferPoolClockTest, Eviction) {
  BufferPool<uint32_t, ClockReplacer> pool(3, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  // Fill the cache
  for (uint32_t i = 0; i < 3; ++i) {
    auto handle = pool.get_or_read(i, io, i * kFrameSize, temp.data());
    EXPECT_FALSE(handle.empty());
  }

  // Trigger eviction
  {
    auto handle = pool.get_or_read(10, io, 10 * kFrameSize, temp.data());
    EXPECT_FALSE(handle.empty());
    EXPECT_EQ(handle.data()[0], 10);
  }

  EXPECT_GT(pool.stats().evictions_.load(), 0U);
}

// =============================================================================
// BufferPool with ClockProReplacer Tests
// =============================================================================

class BufferPoolClockProTest : public BufferPoolTest {};

TEST_F(BufferPoolClockProTest, BasicOperations) {
  BufferPool<uint32_t, ClockProReplacer> pool(kCapacity, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  {
    auto handle = pool.get_or_read(2, io, 2 * kFrameSize, temp.data());
    EXPECT_FALSE(handle.empty());
    EXPECT_EQ(handle.data()[0], 2);
  }

  auto handle = pool.get(2);
  EXPECT_FALSE(handle.empty());
  EXPECT_EQ(handle.data()[0], 2);
}

TEST_F(BufferPoolClockProTest, Eviction) {
  BufferPool<uint32_t, ClockProReplacer> pool(3, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  for (uint32_t i = 0; i < 3; ++i) {
    auto handle = pool.get_or_read(i, io, i * kFrameSize, temp.data());
    EXPECT_FALSE(handle.empty());
  }

  {
    auto handle = pool.get_or_read(10, io, 10 * kFrameSize, temp.data());
    EXPECT_FALSE(handle.empty());
    EXPECT_EQ(handle.data()[0], 10);
  }

  EXPECT_GT(pool.stats().evictions_.load(), 0U);
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

class BufferPoolThreadTest : public BufferPoolTest {};

TEST_F(BufferPoolThreadTest, ConcurrentGetOrRead) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 2);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);

  constexpr int kNumThreads = 4;
  constexpr int kOpsPerThread = 50;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&pool, &io, t]() -> void {
      AlignedBuffer temp(kFrameSize);
      for (int i = 0; i < kOpsPerThread; ++i) {
        auto id = static_cast<uint32_t>((t * kOpsPerThread + i) % kNumPages);
        auto handle = pool.get_or_read(id, io, id * kFrameSize, temp.data());
        if (!handle.empty()) {
          EXPECT_EQ(handle.data()[0], static_cast<uint8_t>(id));
        }
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  // Should complete without crash and have recorded operations
  auto &stats = pool.stats();
  EXPECT_GT(stats.hits_.load() + stats.misses_.load(), 0U);
}

TEST_F(BufferPoolThreadTest, ConcurrentMixedOperations) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 2);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);

  constexpr int kNumThreads = 4;
  constexpr int kOpsPerThread = 100;

  std::atomic<int> hit_count{0};
  std::atomic<int> miss_count{0};

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&pool, &io, &hit_count, &miss_count, t]() -> void {
      AlignedBuffer temp(kFrameSize);
      for (int i = 0; i < kOpsPerThread; ++i) {
        auto id = static_cast<uint32_t>((t * kOpsPerThread + i) % kNumPages);

        if (i % 3 == 0) {
          // get-only (may miss, but no miss counter)
          auto handle = pool.get(id);
          if (!handle.empty()) {
            hit_count.fetch_add(1);
          } else {
            miss_count.fetch_add(1);
          }
        } else {
          // get_or_read (always succeeds for valid pages)
          auto handle = pool.get_or_read(id, io, id * kFrameSize, temp.data());
          if (!handle.empty()) {
            EXPECT_EQ(handle.data()[0], static_cast<uint8_t>(id));
          }
        }
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_GT(hit_count.load() + miss_count.load(), 0);
}

TEST_F(BufferPoolThreadTest, ConcurrentWithEviction) {
  // Small pool to force frequent evictions under contention
  BufferPool<uint32_t> pool(5, kFrameSize, 1);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);

  constexpr int kNumThreads = 4;
  constexpr int kOpsPerThread = 100;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&pool, &io, t]() -> void {
      AlignedBuffer temp(kFrameSize);
      for (int i = 0; i < kOpsPerThread; ++i) {
        // Access all 20 pages through a 5-frame pool
        auto id = static_cast<uint32_t>((t * 7 + i) % kNumPages);
        auto handle = pool.get_or_read(id, io, id * kFrameSize, temp.data());
        // May fail if all frames pinned, but should not crash
        if (!handle.empty()) {
          EXPECT_EQ(handle.data()[0], static_cast<uint8_t>(id));
        }
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  // Should have many evictions given the small pool
  EXPECT_GT(pool.stats().evictions_.load(), 0U);
}

TEST_F(BufferPoolThreadTest, ConcurrentClearAndAccess) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 2);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);

  std::atomic<bool> done{false};
  std::atomic<int> clear_failures{0};

  // Writer thread: continuously load pages
  std::thread writer([&pool, &io, &done]() -> void {
    AlignedBuffer temp(kFrameSize);
    uint32_t id = 0;
    while (!done.load(std::memory_order_relaxed)) {
      auto handle = pool.get_or_read(id, io, id * kFrameSize, temp.data());
      // Don't check result - clear may invalidate state
      (void)handle;
      id = (id + 1) % kNumPages;
    }
  });

  // Clear thread: periodically clear the pool
  std::thread clearer([&pool, &done, &clear_failures]() -> void {
    for (int i = 0; i < 10; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      try {
        pool.clear();
      } catch (const std::logic_error &) {
        clear_failures.fetch_add(1, std::memory_order_relaxed);
      }
    }
    done.store(true, std::memory_order_relaxed);
  });

  writer.join();
  clearer.join();

  // Should complete without crash
  EXPECT_GE(clear_failures.load(std::memory_order_relaxed), 0);
}

TEST_F(BufferPoolThreadTest, ConcurrentClockProReplacer) {
  BufferPool<uint32_t, ClockProReplacer> pool(kCapacity, kFrameSize, 2);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);

  constexpr int kNumThreads = 4;
  constexpr int kOpsPerThread = 50;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&pool, &io, t]() -> void {
      AlignedBuffer temp(kFrameSize);
      for (int i = 0; i < kOpsPerThread; ++i) {
        auto id = static_cast<uint32_t>((t * kOpsPerThread + i) % kNumPages);
        auto handle = pool.get_or_read(id, io, id * kFrameSize, temp.data());
        if (!handle.empty()) {
          EXPECT_EQ(handle.data()[0], static_cast<uint8_t>(id));
        }
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_GT(pool.stats().hits_.load() + pool.stats().misses_.load(), 0U);
}

TEST_F(BufferPoolTest, SingleFrameSingleShardWorks) {
  BufferPool<uint32_t> pool(1, kFrameSize, 1);
  std::vector<uint8_t> data(kFrameSize, 0x5A);

  auto inserted = pool.put(7, data.data());
  ASSERT_FALSE(inserted.empty());
  EXPECT_EQ(inserted.data()[0], 0x5A);

  auto cached = pool.get(7);
  ASSERT_FALSE(cached.empty());
  EXPECT_EQ(cached.data()[0], 0x5A);
}

TEST_F(BufferPoolTest, CapacityEqualsShardCountWorks) {
  BufferPool<uint32_t> pool(16, kFrameSize, 16);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  for (uint32_t i = 0; i < 16; ++i) {
    auto handle = pool.get_or_read(i, io, i * kFrameSize, temp.data());
    ASSERT_FALSE(handle.empty());
    EXPECT_EQ(handle.data()[0], static_cast<uint8_t>(i));
  }

  for (uint32_t i = 0; i < 16; ++i) {
    auto handle = pool.get(i);
    ASSERT_FALSE(handle.empty());
    EXPECT_EQ(handle.data()[0], static_cast<uint8_t>(i));
  }
}

TEST_F(BufferPoolTest, NonDivisibleCapacityFiveAcrossFourShardsWorks) {
  BufferPool<uint32_t> pool(5, kFrameSize, 4);
  const uint32_t kIds[] = {0, 1, 2, 3, 7};

  for (uint32_t id : kIds) {
    std::vector<uint8_t> data(kFrameSize, static_cast<uint8_t>(id));
    auto handle = pool.put(id, data.data());
    ASSERT_FALSE(handle.empty());
    EXPECT_EQ(handle.data()[0], static_cast<uint8_t>(id));
  }

  for (uint32_t id : kIds) {
    auto handle = pool.get(id);
    ASSERT_FALSE(handle.empty());
    EXPECT_EQ(handle.data()[0], static_cast<uint8_t>(id));
  }
}

TEST_F(BufferPoolTest, NonDivisibleCapacityTenAcrossSixShardsWorks) {
  BufferPool<uint32_t> pool(10, kFrameSize, 6);
  const uint32_t kIds[] = {0, 1, 7, 2, 8, 3, 4, 10, 5, 11};

  for (uint32_t id : kIds) {
    std::vector<uint8_t> data(kFrameSize, static_cast<uint8_t>(id));
    auto handle = pool.put(id, data.data());
    ASSERT_FALSE(handle.empty());
    EXPECT_EQ(handle.data()[0], static_cast<uint8_t>(id));
  }

  for (uint32_t id : kIds) {
    auto handle = pool.get(id);
    ASSERT_FALSE(handle.empty());
    EXPECT_EQ(handle.data()[0], static_cast<uint8_t>(id));
  }
}

TEST_F(BufferPoolTest, ClearWithActiveHandlesAcrossShardsThrows) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 4);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  auto h0 = pool.get_or_read(0, io, 0, temp.data());
  auto h1 = pool.get_or_read(1, io, kFrameSize, temp.data());
  ASSERT_FALSE(h0.empty());
  ASSERT_FALSE(h1.empty());

  EXPECT_THROW(pool.clear(), std::logic_error);
}

TEST_F(BufferPoolTest, ClearAfterReleasingHandlesAcrossShardsWorks) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize, 4);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);
  AlignedBuffer temp(kFrameSize);

  {
    auto h0 = pool.get_or_read(0, io, 0, temp.data());
    auto h1 = pool.get_or_read(1, io, kFrameSize, temp.data());
    ASSERT_FALSE(h0.empty());
    ASSERT_FALSE(h1.empty());
  }

  EXPECT_NO_THROW(pool.clear());
  EXPECT_TRUE(pool.get(0).empty());
  EXPECT_TRUE(pool.get(1).empty());
}

}  // namespace alaya
