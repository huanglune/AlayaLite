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

#include "storage/aligned_reader.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace alaya {

class AlignedFileReaderTest : public ::testing::Test {
 protected:
  std::string test_file_;

  void SetUp() override {
    // Create a temporary test file
    test_file_ = "/tmp/aligned_reader_test_" +
                 std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + ".bin";
  }

  void TearDown() override {
    // Clean up test file
    if (std::filesystem::exists(test_file_)) {
      std::filesystem::remove(test_file_);
    }
  }

  // Helper to create a test file with aligned data
  void create_test_file(size_t num_sectors) {
    size_t total_size = kDefaultSectorSize * num_sectors;
    AlignedBuffer buf(total_size);
    // Fill with pattern: each sector starts with its index
    for (size_t i = 0; i < num_sectors; ++i) {
      auto *sector_start =
          reinterpret_cast<uint32_t *>(buf.data() + i * kDefaultSectorSize);
      *sector_start = static_cast<uint32_t>(i);
      // Fill rest with sector index repeated
      for (size_t j = 1; j < kDefaultSectorSize / sizeof(uint32_t); ++j) {
        sector_start[j] = static_cast<uint32_t>(i);
      }
    }

    // Write using standard I/O first
    std::ofstream ofs(test_file_, std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(buf.data()),
              static_cast<std::streamsize>(buf.size()));
    ofs.close();
  }
};

// Test IOStatus to string conversion
TEST_F(AlignedFileReaderTest, IOStatusToStringTest) {
  EXPECT_STREQ(io_status_to_string(IOStatus::kSuccess), "Success");
  EXPECT_STREQ(io_status_to_string(IOStatus::kError), "Error");
  EXPECT_STREQ(io_status_to_string(IOStatus::kEOF), "EOF");
  EXPECT_STREQ(io_status_to_string(IOStatus::kNotAligned), "NotAligned");
}

// Test IORequest structure
TEST_F(AlignedFileReaderTest, IORequestTest) {
  AlignedBuffer buf(kDefaultSectorSize);
  IORequest req(buf.data(), kDefaultSectorSize, 0, false, nullptr);

  EXPECT_EQ(req.buffer_, buf.data());
  EXPECT_EQ(req.size_, kDefaultSectorSize);
  EXPECT_EQ(req.offset_, 0);
  EXPECT_FALSE(req.is_write_);
  EXPECT_EQ(req.status_.load(), IOStatus::kPending);
  EXPECT_FALSE(req.is_complete());
  EXPECT_FALSE(req.is_success());

  req.status_.store(IOStatus::kSuccess);
  EXPECT_TRUE(req.is_complete());
  EXPECT_TRUE(req.is_success());
}

// Test IOContext
TEST_F(AlignedFileReaderTest, IOContextTest) {
  IOContext ctx(16);

  EXPECT_EQ(ctx.max_depth(), 16);
  EXPECT_EQ(ctx.pending_count(), 0);
  EXPECT_TRUE(ctx.can_submit());

  AlignedBuffer buf(kDefaultSectorSize);
  auto *req = ctx.prepare_read(buf.data(), kDefaultSectorSize, 0);
  EXPECT_NE(req, nullptr);
  EXPECT_EQ(ctx.requests().size(), 1);

  ctx.clear();
  EXPECT_EQ(ctx.requests().size(), 0);
}

// Test file open/close
TEST_F(AlignedFileReaderTest, OpenCloseTest) {
  create_test_file(1);

  AlignedFileReader reader;
  EXPECT_FALSE(reader.is_open());

  auto status = reader.open(test_file_, AlignedFileReader::OpenMode::kReadOnly);
  EXPECT_EQ(status, IOStatus::kSuccess);
  EXPECT_TRUE(reader.is_open());
  EXPECT_EQ(reader.filepath(), test_file_);
  EXPECT_EQ(reader.file_size(), kDefaultSectorSize);

  reader.close();
  EXPECT_FALSE(reader.is_open());
}

// Test opening non-existent file
TEST_F(AlignedFileReaderTest, OpenNonExistentTest) {
  AlignedFileReader reader;
  auto status =
      reader.open("/nonexistent/path/file.bin", AlignedFileReader::OpenMode::kReadOnly);
  EXPECT_EQ(status, IOStatus::kError);
  EXPECT_FALSE(reader.is_open());
}

// Test creating new file
TEST_F(AlignedFileReaderTest, CreateFileTest) {
  AlignedFileReader reader;
  auto mode =
      AlignedFileReader::OpenMode::kReadWrite | AlignedFileReader::OpenMode::kCreate;
  auto status = reader.open(test_file_, mode);
  EXPECT_EQ(status, IOStatus::kSuccess);
  EXPECT_TRUE(reader.is_open());
  reader.close();

  // Verify file was created
  EXPECT_TRUE(std::filesystem::exists(test_file_));
}

// Test alignment helpers
TEST_F(AlignedFileReaderTest, AlignmentHelpersTest) {
  create_test_file(1);

  AlignedFileReader reader(test_file_);

  // Test is_aligned for pointers
  AlignedBuffer aligned_buf(kDefaultSectorSize);
  uint8_t unaligned_array[100];
  EXPECT_TRUE(reader.is_aligned(aligned_buf.data()));
  // Unaligned array might accidentally be aligned, so we force misalignment
  EXPECT_FALSE(reader.is_aligned(unaligned_array + 1));

  // Test is_aligned for offsets
  EXPECT_TRUE(reader.is_aligned(static_cast<uint64_t>(0)));
  EXPECT_TRUE(reader.is_aligned(static_cast<uint64_t>(kDefaultSectorSize)));
  EXPECT_FALSE(reader.is_aligned(static_cast<uint64_t>(100)));

  // Test align_up
  EXPECT_EQ(reader.align_up(0), 0);
  EXPECT_EQ(reader.align_up(1), kDefaultSectorSize);
  EXPECT_EQ(reader.align_up(kDefaultSectorSize), kDefaultSectorSize);
  EXPECT_EQ(reader.align_up(kDefaultSectorSize + 1), 2 * kDefaultSectorSize);

  // Test align_down
  EXPECT_EQ(reader.align_down(0), 0);
  EXPECT_EQ(reader.align_down(kDefaultSectorSize - 1), 0);
  EXPECT_EQ(reader.align_down(kDefaultSectorSize), kDefaultSectorSize);
  EXPECT_EQ(reader.align_down(kDefaultSectorSize + 1), kDefaultSectorSize);
}

// Test synchronous read without Direct IO
TEST_F(AlignedFileReaderTest, SyncReadBufferedTest) {
  create_test_file(4);

  // Open without Direct IO for this test
  AlignedFileReader reader;
  auto status = reader.open(test_file_, AlignedFileReader::OpenMode::kReadOnly);
  EXPECT_EQ(status, IOStatus::kSuccess);

  AlignedBuffer buf(kDefaultSectorSize);
  ssize_t bytes = reader.read(buf.data(), kDefaultSectorSize, 0);
  EXPECT_EQ(bytes, static_cast<ssize_t>(kDefaultSectorSize));

  // Verify first sector data
  auto *data = reinterpret_cast<uint32_t *>(buf.data());
  EXPECT_EQ(data[0], 0);  // First sector index is 0
}

#ifdef ALAYA_OS_LINUX
// Test synchronous read with Direct IO (Linux only)
TEST_F(AlignedFileReaderTest, SyncReadDirectIOTest) {
  create_test_file(4);

  AlignedFileReader reader;
  auto mode =
      AlignedFileReader::OpenMode::kReadOnly | AlignedFileReader::OpenMode::kDirectIO;
  auto status = reader.open(test_file_, mode);
  EXPECT_EQ(status, IOStatus::kSuccess);
  EXPECT_TRUE(reader.is_direct_io());

  AlignedBuffer buf(kDefaultSectorSize);
  ssize_t bytes = reader.read(buf.data(), kDefaultSectorSize, 0);
  EXPECT_EQ(bytes, static_cast<ssize_t>(kDefaultSectorSize));

  // Read second sector
  bytes = reader.read(buf.data(), kDefaultSectorSize, kDefaultSectorSize);
  EXPECT_EQ(bytes, static_cast<ssize_t>(kDefaultSectorSize));
  auto *data = reinterpret_cast<uint32_t *>(buf.data());
  EXPECT_EQ(data[0], 1);  // Second sector index is 1
}
#endif

// Test synchronous write
TEST_F(AlignedFileReaderTest, SyncWriteTest) {
  AlignedFileReader reader;
  auto mode =
      AlignedFileReader::OpenMode::kReadWrite | AlignedFileReader::OpenMode::kCreate;
  auto status = reader.open(test_file_, mode);
  EXPECT_EQ(status, IOStatus::kSuccess);

  // Write data
  AlignedBuffer write_buf(kDefaultSectorSize);
  auto *write_data = reinterpret_cast<uint32_t *>(write_buf.data());
  for (size_t i = 0; i < kDefaultSectorSize / sizeof(uint32_t); ++i) {
    write_data[i] = static_cast<uint32_t>(0xDEADBEEF);
  }

  ssize_t bytes = reader.write(write_buf.data(), kDefaultSectorSize, 0);
  EXPECT_EQ(bytes, static_cast<ssize_t>(kDefaultSectorSize));

  // Sync to disk
  EXPECT_EQ(reader.sync(), IOStatus::kSuccess);

  // Read back and verify
  AlignedBuffer read_buf(kDefaultSectorSize);
  bytes = reader.read(read_buf.data(), kDefaultSectorSize, 0);
  EXPECT_EQ(bytes, static_cast<ssize_t>(kDefaultSectorSize));

  auto *read_data = reinterpret_cast<uint32_t *>(read_buf.data());
  EXPECT_EQ(read_data[0], 0xDEADBEEF);
}

// Test async read interface (fallback to sync)
TEST_F(AlignedFileReaderTest, AsyncReadTest) {
  create_test_file(2);

  AlignedFileReader reader(test_file_);

  AlignedBuffer buf(kDefaultSectorSize);
  IORequest req(buf.data(), kDefaultSectorSize, 0, false);

  auto status = reader.read_async(&req);
  EXPECT_EQ(status, IOStatus::kSuccess);
  EXPECT_TRUE(req.is_complete());
  EXPECT_TRUE(req.is_success());
  EXPECT_EQ(req.bytes_transferred_, static_cast<ssize_t>(kDefaultSectorSize));

  // Verify data
  auto *data = reinterpret_cast<uint32_t *>(buf.data());
  EXPECT_EQ(data[0], 0);
}

// Test batch submit
TEST_F(AlignedFileReaderTest, BatchSubmitTest) {
  create_test_file(4);

  AlignedFileReader reader(test_file_);
  IOContext ctx(4);

  // Prepare multiple read requests
  AlignedBuffer buf0(kDefaultSectorSize);
  AlignedBuffer buf1(kDefaultSectorSize);

  ctx.prepare_read(buf0.data(), kDefaultSectorSize, 0);
  ctx.prepare_read(buf1.data(), kDefaultSectorSize, kDefaultSectorSize);

  size_t submitted = reader.submit_batch(&ctx);
  EXPECT_EQ(submitted, 2);

  // Verify all requests completed
  for (const auto &req : ctx.requests()) {
    EXPECT_TRUE(req.is_success());
  }

  // Verify data
  auto *data0 = reinterpret_cast<uint32_t *>(buf0.data());
  auto *data1 = reinterpret_cast<uint32_t *>(buf1.data());
  EXPECT_EQ(data0[0], 0);  // First sector
  EXPECT_EQ(data1[0], 1);  // Second sector
}

// Test move semantics
TEST_F(AlignedFileReaderTest, MoveTest) {
  create_test_file(1);

  AlignedFileReader reader1(test_file_);
  EXPECT_TRUE(reader1.is_open());

  AlignedFileReader reader2(std::move(reader1));
  EXPECT_TRUE(reader2.is_open());
  EXPECT_FALSE(reader1.is_open());  // NOLINT

  // Read should work on moved-to reader
  AlignedBuffer buf(kDefaultSectorSize);
  ssize_t bytes = reader2.read(buf.data(), kDefaultSectorSize, 0);
  EXPECT_EQ(bytes, static_cast<ssize_t>(kDefaultSectorSize));
}

// Test reading past EOF
TEST_F(AlignedFileReaderTest, ReadPastEOFTest) {
  create_test_file(1);

  AlignedFileReader reader(test_file_);
  AlignedBuffer buf(kDefaultSectorSize);

  // Try to read second sector (doesn't exist)
  ssize_t bytes = reader.read(buf.data(), kDefaultSectorSize, kDefaultSectorSize);
  // Should return 0 (EOF) or partial read
  EXPECT_LE(bytes, 0);
}

// Test OpenMode flag combination
TEST_F(AlignedFileReaderTest, OpenModeFlagsTest) {
  using OM = AlignedFileReader::OpenMode;

  auto combined = OM::kReadWrite | OM::kCreate | OM::kDirectIO;

  EXPECT_TRUE(AlignedFileReader::has_flag(combined, OM::kReadWrite));
  EXPECT_TRUE(AlignedFileReader::has_flag(combined, OM::kCreate));
  EXPECT_TRUE(AlignedFileReader::has_flag(combined, OM::kDirectIO));
  EXPECT_FALSE(AlignedFileReader::has_flag(combined, OM::kTruncate));
}

}  // namespace alaya
