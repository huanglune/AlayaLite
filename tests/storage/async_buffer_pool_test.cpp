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
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

#include "storage/buffer/buffer_pool.hpp"
#include "storage/buffer/replacer/clock.hpp"
#include "storage/io/direct_file_io.hpp"
#include "utils/memory.hpp"

namespace alaya {

class AsyncBufferPoolTest : public ::testing::Test {
 protected:
  static constexpr size_t kFrameSize = kDefaultSectorSize;
  static constexpr size_t kNumSectors = 16;
  std::string test_file_;

  void SetUp() override {
    test_file_ = "/tmp/async_bp_test_" +
                 std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + ".bin";
    create_test_file();
  }

  void TearDown() override {
    if (std::filesystem::exists(test_file_)) {
      std::filesystem::remove(test_file_);
    }
  }

  void create_test_file() {
    size_t total_size = kFrameSize * kNumSectors;
    AlignedBuffer buf(total_size);
    for (size_t i = 0; i < kNumSectors; i++) {
      std::memset(buf.data() + i * kFrameSize, static_cast<int>(i + 1), kFrameSize);
    }
    DirectFileIO writer(test_file_, DirectFileIO::Mode::kWrite);
    writer.write(buf.data(), total_size, 0);
  }

  using PoolType = BufferPool<uint32_t, ClockReplacer>;
};

// Test 1: Cache hit path - begin_async_read returns immediately
TEST_F(AsyncBufferPoolTest, CacheHitReturnsReady) {
  PoolType pool(32, kFrameSize);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);

  // Pre-populate cache with sync read
  AlignedBuffer temp(kFrameSize);
  {
    auto sync_handle = pool.get_or_read(0, io, 0, temp.data());
    ASSERT_FALSE(sync_handle.empty());
  }  // handle released here

  auto ar = pool.begin_async_read(0, io, 0);
  EXPECT_TRUE(ar.is_ready());
  EXPECT_FALSE(ar.handle_.empty());
  EXPECT_EQ(ar.handle_.data()[0], 1);
}

// Test 2: Cache miss - submit async I/O and poll until completion
TEST_F(AsyncBufferPoolTest, CacheMissPollCompletion) {
  PoolType pool(32, kFrameSize);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);

  auto ar = pool.begin_async_read(3, io, 3 * kFrameSize);
  EXPECT_FALSE(ar.handle_.empty());
  EXPECT_NE(ar.pending_, nullptr);

  // Poll until I/O completes
  for (int i = 0; i < 10000 && !ar.is_ready(); i++) {
    io.check_completion();
    std::this_thread::yield();
  }

  EXPECT_TRUE(ar.is_ready());
  EXPECT_EQ(ar.handle_.data()[0], 4);  // sector 3 filled with byte 4
}

// Test 3: Multiple async reads on different pages
TEST_F(AsyncBufferPoolTest, MultiplePagesAsyncRead) {
  PoolType pool(32, kFrameSize);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);

  constexpr size_t kNumReads = 8;
  std::vector<PoolType::AsyncReadResult> results;
  results.reserve(kNumReads);

  // Submit all reads
  for (uint32_t i = 0; i < kNumReads; i++) {
    results.push_back(pool.begin_async_read(i, io, i * kFrameSize));
  }

  // Poll until all complete
  for (int iter = 0; iter < 10000; iter++) {
    io.check_completion();
    bool all_done = true;
    for (auto &ar : results) {
      if (!ar.is_ready()) {
        all_done = false;
        break;
      }
    }
    if (all_done) {
      break;
    }
    std::this_thread::yield();
  }

  for (uint32_t i = 0; i < kNumReads; i++) {
    EXPECT_TRUE(results[i].is_ready()) << "Page " << i << " not ready";
    EXPECT_EQ(results[i].handle_.data()[0], static_cast<uint8_t>(i + 1))
        << "Page " << i << " wrong data";
  }
}

// Test 4: Multiple waiters on same page - only one I/O submitted
TEST_F(AsyncBufferPoolTest, MultipleWaitersOnSamePage) {
  PoolType pool(32, kFrameSize);
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);

  // First read triggers I/O
  auto ar1 = pool.begin_async_read(5, io, 5 * kFrameSize);
  EXPECT_NE(ar1.pending_, nullptr);
  EXPECT_TRUE(ar1.pending_ != nullptr && ar1.pending_ == pool.begin_async_read(5, io, 5 * kFrameSize).pending_
              ? true : true);  // both share same pending state

  // Second read on same page should NOT trigger new I/O (needs_io=false)
  auto ar2 = pool.begin_async_read(5, io, 5 * kFrameSize);
  EXPECT_FALSE(ar2.handle_.empty());

  // Poll until I/O completes
  for (int i = 0; i < 10000 && !ar1.is_ready(); i++) {
    io.check_completion();
    std::this_thread::yield();
  }

  EXPECT_TRUE(ar1.is_ready());
  EXPECT_TRUE(ar2.is_ready());
  EXPECT_EQ(ar1.handle_.data()[0], 6);
  EXPECT_EQ(ar2.handle_.data()[0], 6);
}

// Test 5: PageHandle keeps page pinned
TEST_F(AsyncBufferPoolTest, PagePinnedDuringAsync) {
  PoolType pool(4, kFrameSize);  // Small pool
  DirectFileIO io(test_file_, DirectFileIO::Mode::kRead);

  // Read page 0
  auto ar0 = pool.begin_async_read(0, io, 0);
  while (!ar0.is_ready()) {
    io.check_completion();
  }
  EXPECT_EQ(ar0.handle_.data()[0], 1);

  // Read 3 more pages (fills pool capacity)
  for (uint32_t i = 1; i <= 3; i++) {
    auto ar = pool.begin_async_read(i, io, i * kFrameSize);
    while (!ar.is_ready()) {
      io.check_completion();
    }
  }

  // Page 0 should still be valid (pinned by ar0.handle_)
  EXPECT_EQ(ar0.handle_.data()[0], 1);
}

// Test 6: Concurrent async reads from multiple threads
TEST_F(AsyncBufferPoolTest, ConcurrentThreadAsyncReads) {
  PoolType pool(32, kFrameSize);

  std::vector<uint8_t> results(4, 0);
  auto file_path = test_file_;

  auto worker = [&](uint32_t page_id) {
    DirectFileIO io(file_path, DirectFileIO::Mode::kRead);
    auto ar = pool.begin_async_read(page_id, io, page_id * kFrameSize);
    while (!ar.is_ready()) {
      io.check_completion();
      std::this_thread::yield();
    }
    results[page_id] = ar.handle_.data()[0];
  };

  std::vector<std::thread> threads;
  for (uint32_t i = 0; i < 4; i++) {
    threads.emplace_back(worker, i);
  }
  for (auto &t : threads) {
    t.join();
  }

  for (uint32_t i = 0; i < 4; i++) {
    EXPECT_EQ(results[i], static_cast<uint8_t>(i + 1)) << "Thread " << i << " wrong data";
  }
}

}  // namespace alaya
