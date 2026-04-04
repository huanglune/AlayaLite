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

#include "storage/io/direct_file_io.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

#include "utils/memory.hpp"

namespace alaya {

class IOUringAsyncTest : public ::testing::Test {
 protected:
  std::string test_file_;

  void SetUp() override {
    test_file_ = "/tmp/io_uring_async_test_" +
                 std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + ".bin";
  }

  void TearDown() override {
    if (std::filesystem::exists(test_file_)) {
      std::filesystem::remove(test_file_);
    }
  }

  void create_test_file(size_t num_sectors) {
    size_t total_size = kDefaultSectorSize * num_sectors;
    AlignedBuffer buf(total_size);
    // Fill each sector with its index byte
    for (size_t i = 0; i < num_sectors; i++) {
      std::memset(buf.data() + i * kDefaultSectorSize, static_cast<int>(i + 1), kDefaultSectorSize);
    }
    DirectFileIO writer(test_file_, DirectFileIO::Mode::kWrite);
    writer.write(buf.data(), total_size, 0);
  }
};

/// Callback that records result and sets a flag.
struct CallbackState {
  std::atomic<bool> completed_{false};
  int32_t result_{0};
};

static void test_callback(void *arg, int32_t result) {
  auto *state = static_cast<CallbackState *>(arg);
  state->result_ = result;
  state->completed_.store(true, std::memory_order_release);
}

TEST_F(IOUringAsyncTest, SubmitAndCheckCompletion) {
  create_test_file(4);

  DirectFileIO reader(test_file_, DirectFileIO::Mode::kRead);
  ASSERT_TRUE(reader.is_open());

  AlignedBuffer buf(kDefaultSectorSize);
  CallbackState state;

  // Submit async read for sector 0
  bool submitted =
      reader.submit_async_read(buf.data(), kDefaultSectorSize, 0, test_callback, &state);
  ASSERT_TRUE(submitted);

  // Poll until completion
  size_t total = 0;
  for (int i = 0; i < 1000 && !state.completed_.load(std::memory_order_acquire); i++) {
    total += reader.check_completion();
    std::this_thread::yield();
  }
  // Final check
  total += reader.check_completion();

  EXPECT_TRUE(state.completed_.load());
  EXPECT_EQ(state.result_, static_cast<int32_t>(kDefaultSectorSize));
  EXPECT_GE(total, 1U);

  // Verify data: sector 0 filled with byte value 1
  for (size_t i = 0; i < kDefaultSectorSize; i++) {
    EXPECT_EQ(buf[i], 1) << "Mismatch at byte " << i;
  }
}

TEST_F(IOUringAsyncTest, MultipleAsyncReads) {
  constexpr size_t kNumSectors = 8;
  create_test_file(kNumSectors);

  DirectFileIO reader(test_file_, DirectFileIO::Mode::kRead);
  ASSERT_TRUE(reader.is_open());

  std::vector<AlignedBuffer> buffers(kNumSectors);
  std::vector<CallbackState> states(kNumSectors);

  for (size_t i = 0; i < kNumSectors; i++) {
    buffers[i].resize(kDefaultSectorSize);
  }

  // Submit all reads
  for (size_t i = 0; i < kNumSectors; i++) {
    bool ok = reader.submit_async_read(buffers[i].data(), kDefaultSectorSize,
                                       i * kDefaultSectorSize, test_callback, &states[i]);
    ASSERT_TRUE(ok) << "Failed to submit read for sector " << i;
  }

  // Poll until all complete
  for (int iter = 0; iter < 10000; iter++) {
    reader.check_completion();
    bool all_done = true;
    for (size_t i = 0; i < kNumSectors; i++) {
      if (!states[i].completed_.load(std::memory_order_acquire)) {
        all_done = false;
        break;
      }
    }
    if (all_done) {
      break;
    }
    std::this_thread::yield();
  }

  // Verify all completed with correct data
  for (size_t i = 0; i < kNumSectors; i++) {
    EXPECT_TRUE(states[i].completed_.load()) << "Sector " << i << " not completed";
    EXPECT_EQ(states[i].result_, static_cast<int32_t>(kDefaultSectorSize))
        << "Sector " << i << " wrong result";
    for (size_t j = 0; j < kDefaultSectorSize; j++) {
      EXPECT_EQ(buffers[i][j], static_cast<uint8_t>(i + 1))
          << "Sector " << i << " mismatch at byte " << j;
    }
  }
}

TEST_F(IOUringAsyncTest, CheckCompletionNoPending) {
  // check_completion on a fresh engine with no pending I/O should return 0
  DirectFileIO reader;
  EXPECT_EQ(reader.check_completion(), 0U);
}

TEST_F(IOUringAsyncTest, DrainPending) {
  create_test_file(2);

  DirectFileIO reader(test_file_, DirectFileIO::Mode::kRead);
  ASSERT_TRUE(reader.is_open());

  AlignedBuffer buf(kDefaultSectorSize);
  CallbackState state;

  bool ok = reader.submit_async_read(buf.data(), kDefaultSectorSize, 0, test_callback, &state);
  ASSERT_TRUE(ok);

  // Drain should block until completion and invoke callback
  reader.drain_pending();

  EXPECT_TRUE(state.completed_.load());
  EXPECT_EQ(state.result_, static_cast<int32_t>(kDefaultSectorSize));
}

TEST_F(IOUringAsyncTest, ThreadLocalRingIsolation) {
  create_test_file(4);

  CallbackState state1;
  CallbackState state2;
  AlignedBuffer buf1(kDefaultSectorSize);
  AlignedBuffer buf2(kDefaultSectorSize);

  auto file_path = test_file_;

  // Two threads each submit and complete independently
  auto worker = [&file_path](AlignedBuffer &buf, CallbackState &state, uint64_t offset) {
    DirectFileIO reader(file_path, DirectFileIO::Mode::kRead);
    bool ok =
        reader.submit_async_read(buf.data(), kDefaultSectorSize, offset, test_callback, &state);
    EXPECT_TRUE(ok);

    for (int i = 0; i < 10000 && !state.completed_.load(std::memory_order_acquire); i++) {
      reader.check_completion();
      std::this_thread::yield();
    }
    reader.check_completion();
    EXPECT_TRUE(state.completed_.load());
  };

  std::thread t1(worker, std::ref(buf1), std::ref(state1), 0);
  std::thread t2(worker, std::ref(buf2), std::ref(state2), kDefaultSectorSize);
  t1.join();
  t2.join();

  EXPECT_TRUE(state1.completed_.load());
  EXPECT_TRUE(state2.completed_.load());

  // Verify data integrity
  for (size_t i = 0; i < kDefaultSectorSize; i++) {
    EXPECT_EQ(buf1[i], 1);
    EXPECT_EQ(buf2[i], 2);
  }
}

}  // namespace alaya
