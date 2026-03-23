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

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "utils/memory.hpp"

namespace alaya {

class DirectFileIOTest : public ::testing::Test {
 protected:
  std::string test_file_;

  void SetUp() override {
    test_file_ = "/tmp/async_reader_test_" +
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

    for (size_t i = 0; i < num_sectors; ++i) {
      auto* sector_start = reinterpret_cast<uint32_t*>(buf.data() + i * kDefaultSectorSize);
      for (size_t j = 0; j < kDefaultSectorSize / sizeof(uint32_t); ++j) {
        sector_start[j] = static_cast<uint32_t>(i);
      }
    }

    std::ofstream ofs(test_file_, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(total_size));
    ofs.close();
  }
};

// =============================================================================
// Basic File Operations
// =============================================================================

TEST_F(DirectFileIOTest, DefaultConstructor) {
  DirectFileIO reader;
  EXPECT_FALSE(reader.is_open());
}

TEST_F(DirectFileIOTest, OpenClose) {
  create_test_file(1);

  DirectFileIO reader;
  EXPECT_TRUE(reader.open(test_file_));
  EXPECT_TRUE(reader.is_open());
  EXPECT_EQ(reader.path(), test_file_);
  EXPECT_EQ(reader.file_size(), kDefaultSectorSize);

  reader.close();
  EXPECT_FALSE(reader.is_open());
}

TEST_F(DirectFileIOTest, ConstructorOpen) {
  create_test_file(1);

  DirectFileIO reader(test_file_);
  EXPECT_TRUE(reader.is_open());
  EXPECT_EQ(reader.file_size(), kDefaultSectorSize);
}

TEST_F(DirectFileIOTest, OpenNonExistent) {
  DirectFileIO reader;
  EXPECT_FALSE(reader.open("/nonexistent/path/file.bin"));
  EXPECT_FALSE(reader.is_open());
}

TEST_F(DirectFileIOTest, ConstructorThrows) {
  EXPECT_THROW(DirectFileIO reader("/nonexistent/path/file.bin"), std::runtime_error);
}

TEST_F(DirectFileIOTest, CreateFile) {
  DirectFileIO reader;
  EXPECT_TRUE(reader.open(test_file_, DirectFileIO::Mode::kWrite));
  EXPECT_TRUE(reader.is_open());
  reader.close();

  EXPECT_TRUE(std::filesystem::exists(test_file_));
}

// =============================================================================
// Synchronous I/O
// =============================================================================

TEST_F(DirectFileIOTest, SyncRead) {
  create_test_file(4);

  DirectFileIO reader(test_file_);
  AlignedBuffer buf(kDefaultSectorSize);

  ssize_t bytes = reader.read(buf.data(), kDefaultSectorSize, 0);
  EXPECT_EQ(bytes, static_cast<ssize_t>(kDefaultSectorSize));

  auto* data = reinterpret_cast<uint32_t*>(buf.data());
  EXPECT_EQ(data[0], 0);  // First sector

  // Read second sector
  bytes = reader.read(buf.data(), kDefaultSectorSize, kDefaultSectorSize);
  EXPECT_EQ(bytes, static_cast<ssize_t>(kDefaultSectorSize));
  data = reinterpret_cast<uint32_t*>(buf.data());
  EXPECT_EQ(data[0], 1);  // Second sector
}

TEST_F(DirectFileIOTest, SyncWrite) {
  DirectFileIO reader;
  EXPECT_TRUE(reader.open(test_file_, DirectFileIO::Mode::kReadWrite));

  AlignedBuffer write_buf(kDefaultSectorSize);
  auto* write_data = reinterpret_cast<uint32_t*>(write_buf.data());
  for (size_t i = 0; i < kDefaultSectorSize / sizeof(uint32_t); ++i) {
    write_data[i] = 0xDEADBEEF;
  }

  ssize_t bytes = reader.write(write_buf.data(), kDefaultSectorSize, 0);
  EXPECT_EQ(bytes, static_cast<ssize_t>(kDefaultSectorSize));

  // Read back and verify
  AlignedBuffer read_buf(kDefaultSectorSize);
  bytes = reader.read(read_buf.data(), kDefaultSectorSize, 0);
  EXPECT_EQ(bytes, static_cast<ssize_t>(kDefaultSectorSize));

  auto* read_data = reinterpret_cast<uint32_t*>(read_buf.data());
  EXPECT_EQ(read_data[0], 0xDEADBEEF);
}

TEST_F(DirectFileIOTest, SyncReadFileNotOpen) {
  DirectFileIO reader;

  AlignedBuffer buf(kDefaultSectorSize);
  ssize_t bytes = reader.read(buf.data(), kDefaultSectorSize, 0);
  EXPECT_EQ(bytes, -1);
}

TEST_F(DirectFileIOTest, ReadPastEOF) {
  create_test_file(1);

  DirectFileIO reader(test_file_);
  AlignedBuffer buf(kDefaultSectorSize);

  ssize_t bytes = reader.read(buf.data(), kDefaultSectorSize, kDefaultSectorSize);
  EXPECT_LE(bytes, 0);
}

// =============================================================================
// Batch Async I/O
// =============================================================================

TEST_F(DirectFileIOTest, BatchRead) {
  create_test_file(4);

  DirectFileIO reader(test_file_);

  AlignedBuffer buf0(kDefaultSectorSize);
  AlignedBuffer buf1(kDefaultSectorSize);

  std::vector<IORequest> requests = {
      IORequest(buf0.data(), kDefaultSectorSize, 0),
      IORequest(buf0.data(), kDefaultSectorSize, kDefaultSectorSize),
  };
  requests[1].buffer_ = buf1.data();

  size_t submitted = reader.submit_reads(requests);
  EXPECT_EQ(submitted, 2);

  // With sync engine, requests complete immediately
  for (const auto& req : requests) {
    EXPECT_TRUE(req.is_success());
  }

  auto* data0 = reinterpret_cast<uint32_t*>(buf0.data());
  auto* data1 = reinterpret_cast<uint32_t*>(buf1.data());
  EXPECT_EQ(data0[0], 0);
  EXPECT_EQ(data1[0], 1);
}

TEST_F(DirectFileIOTest, BatchWrite) {
  DirectFileIO reader;
  EXPECT_TRUE(reader.open(test_file_, DirectFileIO::Mode::kReadWrite));

  AlignedBuffer buf(kDefaultSectorSize);
  auto* data = reinterpret_cast<uint32_t*>(buf.data());
  data[0] = 0xCAFEBABE;

  std::vector<IORequest> requests = {
      IORequest(buf.data(), kDefaultSectorSize, 0),
  };

  size_t submitted = reader.submit_writes(requests);
  EXPECT_EQ(submitted, 1);

  // Verify
  AlignedBuffer read_buf(kDefaultSectorSize);
  ssize_t bytes = reader.read(read_buf.data(), kDefaultSectorSize, 0);
  EXPECT_EQ(bytes, static_cast<ssize_t>(kDefaultSectorSize));

  auto* read_data = reinterpret_cast<uint32_t*>(read_buf.data());
  EXPECT_EQ(read_data[0], 0xCAFEBABE);
}

TEST_F(DirectFileIOTest, BatchReadFileNotOpen) {
  DirectFileIO reader;

  AlignedBuffer buf(kDefaultSectorSize);
  std::vector<IORequest> requests = {
      IORequest(buf.data(), kDefaultSectorSize, 0),
  };

  size_t submitted = reader.submit_reads(requests);
  EXPECT_EQ(submitted, 0);
}

// =============================================================================
// Alignment Helpers
// =============================================================================

TEST_F(DirectFileIOTest, AlignmentHelpers) {
  create_test_file(1);

  DirectFileIO reader(test_file_);

  // Pointer alignment
  AlignedBuffer aligned_buf(kDefaultSectorSize);
  uint8_t unaligned_array[100];
  EXPECT_TRUE(reader.is_aligned(aligned_buf.data()));
  EXPECT_FALSE(reader.is_aligned(unaligned_array + 1));

  // Offset alignment
  EXPECT_TRUE(reader.is_aligned(static_cast<uint64_t>(0)));
  EXPECT_TRUE(reader.is_aligned(static_cast<uint64_t>(kDefaultSectorSize)));
  EXPECT_FALSE(reader.is_aligned(static_cast<uint64_t>(100)));

  // align_up
  EXPECT_EQ(reader.align_up(0), 0);
  EXPECT_EQ(reader.align_up(1), kDefaultSectorSize);
  EXPECT_EQ(reader.align_up(kDefaultSectorSize), kDefaultSectorSize);
  EXPECT_EQ(reader.align_up(kDefaultSectorSize + 1), 2 * kDefaultSectorSize);

  // align_down
  EXPECT_EQ(reader.align_down(0), 0);
  EXPECT_EQ(reader.align_down(kDefaultSectorSize - 1), 0);
  EXPECT_EQ(reader.align_down(kDefaultSectorSize), kDefaultSectorSize);
  EXPECT_EQ(reader.align_down(kDefaultSectorSize + 1), kDefaultSectorSize);
}

// =============================================================================
// Move Semantics
// =============================================================================

TEST_F(DirectFileIOTest, MoveConstructor) {
  create_test_file(1);

  DirectFileIO reader1(test_file_);
  EXPECT_TRUE(reader1.is_open());

  DirectFileIO reader2(std::move(reader1));
  EXPECT_TRUE(reader2.is_open());
  EXPECT_FALSE(reader1.is_open());  // NOLINT

  AlignedBuffer buf(kDefaultSectorSize);
  ssize_t bytes = reader2.read(buf.data(), kDefaultSectorSize, 0);
  EXPECT_EQ(bytes, static_cast<ssize_t>(kDefaultSectorSize));
}

TEST_F(DirectFileIOTest, MoveAssignment) {
  create_test_file(1);

  DirectFileIO reader1(test_file_);
  DirectFileIO reader2;

  reader2 = std::move(reader1);
  EXPECT_TRUE(reader2.is_open());
  EXPECT_FALSE(reader1.is_open());  // NOLINT

  AlignedBuffer buf(kDefaultSectorSize);
  ssize_t bytes = reader2.read(buf.data(), kDefaultSectorSize, 0);
  EXPECT_EQ(bytes, static_cast<ssize_t>(kDefaultSectorSize));
}

// =============================================================================
// Engine Info
// =============================================================================

TEST_F(DirectFileIOTest, EngineInfo) {
  create_test_file(1);

  DirectFileIO reader(test_file_);

#ifdef ALAYA_OS_LINUX
  // On Linux, should be io_uring or sync depending on kernel support
  EXPECT_TRUE(reader.engine_name() == "io_uring" || reader.engine_name() == "sync");
#else
  EXPECT_EQ(reader.engine_name(), "sync");
  EXPECT_FALSE(reader.supports_async());
#endif
}

// =============================================================================
// IORequest Tests
// =============================================================================

TEST_F(DirectFileIOTest, IORequestBasic) {
  AlignedBuffer buf(kDefaultSectorSize);
  IORequest req(buf.data(), kDefaultSectorSize, 4096, nullptr);

  EXPECT_EQ(req.buffer_, buf.data());
  EXPECT_EQ(req.size_, kDefaultSectorSize);
  EXPECT_EQ(req.offset_, 4096);
  EXPECT_EQ(req.result_, 0);

  // Test is_success
  EXPECT_FALSE(req.is_success());
  req.result_ = static_cast<int32_t>(kDefaultSectorSize);
  EXPECT_TRUE(req.is_success());
}

TEST_F(DirectFileIOTest, IORequestNotSubmittedIsNotSuccess) {
  AlignedBuffer buf(kDefaultSectorSize);
  IORequest req(buf.data(), kDefaultSectorSize, 4096, nullptr);

  req.result_ = std::numeric_limits<int32_t>::min();
  EXPECT_FALSE(req.is_success());
}

// =============================================================================
// Reopen File
// =============================================================================

TEST_F(DirectFileIOTest, ReopenFile) {
  create_test_file(1);

  DirectFileIO reader(test_file_);
  EXPECT_TRUE(reader.is_open());

  EXPECT_TRUE(reader.open(test_file_));
  EXPECT_TRUE(reader.is_open());
}

}  // namespace alaya
