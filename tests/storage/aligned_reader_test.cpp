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

// Test IORequest copy constructor
TEST_F(AlignedFileReaderTest, IORequestCopyConstructorTest) {
  AlignedBuffer buf(kDefaultSectorSize);
  IORequest req1(buf.data(), kDefaultSectorSize, 4096, true, nullptr);
  req1.status_.store(IOStatus::kSuccess);
  req1.bytes_transferred_ = 100;

  IORequest req2(req1);  // Copy constructor

  EXPECT_EQ(req2.buffer_, req1.buffer_);
  EXPECT_EQ(req2.size_, req1.size_);
  EXPECT_EQ(req2.offset_, req1.offset_);
  EXPECT_EQ(req2.is_write_, req1.is_write_);
  EXPECT_EQ(req2.user_data_, req1.user_data_);
  EXPECT_EQ(req2.status_.load(), IOStatus::kSuccess);
  EXPECT_EQ(req2.bytes_transferred_, 100);
}

// Test IORequest move constructor
TEST_F(AlignedFileReaderTest, IORequestMoveConstructorTest) {
  AlignedBuffer buf(kDefaultSectorSize);
  IORequest req1(buf.data(), kDefaultSectorSize, 8192, false, nullptr);
  req1.status_.store(IOStatus::kEOF);
  req1.bytes_transferred_ = 50;

  IORequest req2(std::move(req1));  // Move constructor

  EXPECT_EQ(req2.buffer_, buf.data());
  EXPECT_EQ(req2.size_, kDefaultSectorSize);
  EXPECT_EQ(req2.offset_, 8192);
  EXPECT_FALSE(req2.is_write_);
  EXPECT_EQ(req2.status_.load(), IOStatus::kEOF);
  EXPECT_EQ(req2.bytes_transferred_, 50);
}

// Test IORequest copy assignment operator
TEST_F(AlignedFileReaderTest, IORequestCopyAssignmentTest) {
  AlignedBuffer buf1(kDefaultSectorSize);
  AlignedBuffer buf2(kDefaultSectorSize);
  IORequest req1(buf1.data(), kDefaultSectorSize, 0, false, nullptr);
  req1.status_.store(IOStatus::kSuccess);
  req1.bytes_transferred_ = 200;

  IORequest req2(buf2.data(), 512, 100, true, nullptr);

  req2 = req1;  // Copy assignment

  EXPECT_EQ(req2.buffer_, buf1.data());
  EXPECT_EQ(req2.size_, kDefaultSectorSize);
  EXPECT_EQ(req2.offset_, 0);
  EXPECT_FALSE(req2.is_write_);
  EXPECT_EQ(req2.status_.load(), IOStatus::kSuccess);
  EXPECT_EQ(req2.bytes_transferred_, 200);
}

// Test IORequest move assignment operator
TEST_F(AlignedFileReaderTest, IORequestMoveAssignmentTest) {
  AlignedBuffer buf1(kDefaultSectorSize);
  AlignedBuffer buf2(kDefaultSectorSize);
  IORequest req1(buf1.data(), kDefaultSectorSize, 4096, true, nullptr);
  req1.status_.store(IOStatus::kError);
  req1.bytes_transferred_ = 0;

  IORequest req2(buf2.data(), 512, 100, false, nullptr);

  req2 = std::move(req1);  // Move assignment

  EXPECT_EQ(req2.buffer_, buf1.data());
  EXPECT_EQ(req2.size_, kDefaultSectorSize);
  EXPECT_EQ(req2.offset_, 4096);
  EXPECT_TRUE(req2.is_write_);
  EXPECT_EQ(req2.status_.load(), IOStatus::kError);
  EXPECT_EQ(req2.bytes_transferred_, 0);
}

// Test IORequest self-assignment (copy) - using pointer indirection to avoid compiler warning
TEST_F(AlignedFileReaderTest, IORequestSelfCopyAssignmentTest) {
  AlignedBuffer buf(kDefaultSectorSize);
  IORequest req(buf.data(), kDefaultSectorSize, 0, false, nullptr);
  req.status_.store(IOStatus::kSuccess);

  IORequest *ptr = &req;
  req = *ptr;  // Self-assignment via pointer

  EXPECT_EQ(req.buffer_, buf.data());
  EXPECT_EQ(req.status_.load(), IOStatus::kSuccess);
}

// Test IORequest self-assignment (move) - using pointer indirection to avoid compiler warning
TEST_F(AlignedFileReaderTest, IORequestSelfMoveAssignmentTest) {
  AlignedBuffer buf(kDefaultSectorSize);
  IORequest req(buf.data(), kDefaultSectorSize, 0, false, nullptr);
  req.status_.store(IOStatus::kSuccess);

  IORequest *ptr = &req;
  req = std::move(*ptr);  // Self-move-assignment via pointer

  EXPECT_EQ(req.buffer_, buf.data());
  EXPECT_EQ(req.status_.load(), IOStatus::kSuccess);
}

// Test IOContext prepare_write
TEST_F(AlignedFileReaderTest, IOContextPrepareWriteTest) {
  IOContext ctx(16);

  AlignedBuffer buf(kDefaultSectorSize);
  auto *req = ctx.prepare_write(buf.data(), kDefaultSectorSize, 0, nullptr);

  EXPECT_NE(req, nullptr);
  EXPECT_EQ(req->buffer_, buf.data());
  EXPECT_EQ(req->size_, kDefaultSectorSize);
  EXPECT_TRUE(req->is_write_);
  EXPECT_EQ(ctx.requests().size(), 1);
}

// Test IOContext queue full scenario
// Note: IOContext's pending_count_ tracking is a placeholder for future io_uring integration.
// Currently, can_submit() always returns true since pending_count_ is never incremented.
// This test verifies the current behavior.
TEST_F(AlignedFileReaderTest, IOContextQueueTest) {
  IOContext ctx(2);  // max_depth = 2

  AlignedBuffer buf1(kDefaultSectorSize);
  AlignedBuffer buf2(kDefaultSectorSize);
  AlignedBuffer buf3(kDefaultSectorSize);

  // All requests succeed because pending_count_ is never incremented in current impl
  auto *req1 = ctx.prepare_read(buf1.data(), kDefaultSectorSize, 0);
  auto *req2 = ctx.prepare_read(buf2.data(), kDefaultSectorSize, kDefaultSectorSize);
  auto *req3 = ctx.prepare_read(buf3.data(), kDefaultSectorSize, 2 * kDefaultSectorSize);

  EXPECT_NE(req1, nullptr);
  EXPECT_NE(req2, nullptr);
  EXPECT_NE(req3, nullptr);  // Succeeds because pending_count_ stays at 0
  EXPECT_EQ(ctx.requests().size(), 3);
  EXPECT_EQ(ctx.pending_count(), 0);  // Never incremented in current impl
  EXPECT_TRUE(ctx.can_submit());
}

// Test IOContext set_callback
TEST_F(AlignedFileReaderTest, IOContextSetCallbackTest) {
  IOContext ctx(16);
  bool callback_called = false;

  ctx.set_callback([&callback_called](IORequest *) -> void { callback_called = true; });

  // Callback is stored but not invoked directly by IOContext
  EXPECT_FALSE(callback_called);
}

// Test IOContext const requests accessor
TEST_F(AlignedFileReaderTest, IOContextConstRequestsTest) {
  IOContext ctx(16);
  AlignedBuffer buf(kDefaultSectorSize);
  ctx.prepare_read(buf.data(), kDefaultSectorSize, 0);

  const IOContext &const_ctx = ctx;
  const auto &requests = const_ctx.requests();

  EXPECT_EQ(requests.size(), 1);
}

// Test async write interface
TEST_F(AlignedFileReaderTest, AsyncWriteTest) {
  AlignedFileReader reader;
  auto mode =
      AlignedFileReader::OpenMode::kReadWrite | AlignedFileReader::OpenMode::kCreate;
  auto status = reader.open(test_file_, mode);
  EXPECT_EQ(status, IOStatus::kSuccess);

  AlignedBuffer buf(kDefaultSectorSize);
  auto *data = reinterpret_cast<uint32_t *>(buf.data());
  data[0] = 0xCAFEBABE;

  IORequest req(buf.data(), kDefaultSectorSize, 0, true);

  status = reader.write_async(&req);
  EXPECT_EQ(status, IOStatus::kSuccess);
  EXPECT_TRUE(req.is_complete());
  EXPECT_TRUE(req.is_success());
  EXPECT_EQ(req.bytes_transferred_, static_cast<ssize_t>(kDefaultSectorSize));

  // Verify write by reading back
  AlignedBuffer read_buf(kDefaultSectorSize);
  ssize_t bytes = reader.read(read_buf.data(), kDefaultSectorSize, 0);
  EXPECT_EQ(bytes, static_cast<ssize_t>(kDefaultSectorSize));
  auto *read_data = reinterpret_cast<uint32_t *>(read_buf.data());
  EXPECT_EQ(read_data[0], 0xCAFEBABE);
}

// Test async read/write when file not open
TEST_F(AlignedFileReaderTest, AsyncOperationsFileNotOpenTest) {
  AlignedFileReader reader;  // Not opened

  AlignedBuffer buf(kDefaultSectorSize);
  IORequest read_req(buf.data(), kDefaultSectorSize, 0, false);
  IORequest write_req(buf.data(), kDefaultSectorSize, 0, true);

  auto status = reader.read_async(&read_req);
  EXPECT_EQ(status, IOStatus::kFileNotOpen);
  EXPECT_EQ(read_req.status_.load(), IOStatus::kFileNotOpen);

  status = reader.write_async(&write_req);
  EXPECT_EQ(status, IOStatus::kFileNotOpen);
  EXPECT_EQ(write_req.status_.load(), IOStatus::kFileNotOpen);
}

// Test sync read/write when file not open
TEST_F(AlignedFileReaderTest, SyncOperationsFileNotOpenTest) {
  AlignedFileReader reader;  // Not opened

  AlignedBuffer buf(kDefaultSectorSize);
  ssize_t bytes = reader.read(buf.data(), kDefaultSectorSize, 0);
  EXPECT_EQ(bytes, -1);

  bytes = reader.write(buf.data(), kDefaultSectorSize, 0);
  EXPECT_EQ(bytes, -1);
}

// Test sync when file not open
TEST_F(AlignedFileReaderTest, SyncFileNotOpenTest) {
  AlignedFileReader reader;  // Not opened

  auto status = reader.sync();
  EXPECT_EQ(status, IOStatus::kFileNotOpen);
}

// Test update_file_size
TEST_F(AlignedFileReaderTest, UpdateFileSizeTest) {
  create_test_file(2);

  AlignedFileReader reader(test_file_);
  EXPECT_EQ(reader.file_size(), 2 * kDefaultSectorSize);

  // Update file size (should remain the same since file hasn't changed)
  auto status = reader.update_file_size();
  EXPECT_EQ(status, IOStatus::kSuccess);
  EXPECT_EQ(reader.file_size(), 2 * kDefaultSectorSize);
}

// Test update_file_size when file not open
TEST_F(AlignedFileReaderTest, UpdateFileSizeNotOpenTest) {
  AlignedFileReader reader;  // Not opened

  auto status = reader.update_file_size();
  EXPECT_EQ(status, IOStatus::kFileNotOpen);
}

// Test constructor that throws on failure
TEST_F(AlignedFileReaderTest, ConstructorThrowsTest) {
  EXPECT_THROW(AlignedFileReader reader("/nonexistent/path/file.bin"), std::runtime_error);
}

// Test move assignment operator
TEST_F(AlignedFileReaderTest, MoveAssignmentTest) {
  create_test_file(1);

  AlignedFileReader reader1(test_file_);
  EXPECT_TRUE(reader1.is_open());

  AlignedFileReader reader2;
  EXPECT_FALSE(reader2.is_open());

  reader2 = std::move(reader1);
  EXPECT_TRUE(reader2.is_open());
  EXPECT_FALSE(reader1.is_open());  // NOLINT

  // Read should work on moved-to reader
  AlignedBuffer buf(kDefaultSectorSize);
  ssize_t bytes = reader2.read(buf.data(), kDefaultSectorSize, 0);
  EXPECT_EQ(bytes, static_cast<ssize_t>(kDefaultSectorSize));
}

// Test move assignment to self - using pointer indirection to avoid compiler warning
TEST_F(AlignedFileReaderTest, MoveSelfAssignmentTest) {
  create_test_file(1);

  AlignedFileReader reader(test_file_);
  EXPECT_TRUE(reader.is_open());

  AlignedFileReader *ptr = &reader;
  reader = std::move(*ptr);  // Self-move-assignment via pointer
  EXPECT_TRUE(reader.is_open());

  // Read should still work
  AlignedBuffer buf(kDefaultSectorSize);
  ssize_t bytes = reader.read(buf.data(), kDefaultSectorSize, 0);
  EXPECT_EQ(bytes, static_cast<ssize_t>(kDefaultSectorSize));
}

// Test reap_completions
TEST_F(AlignedFileReaderTest, ReapCompletionsTest) {
  create_test_file(2);

  AlignedFileReader reader(test_file_);
  IOContext ctx(4);

  AlignedBuffer buf(kDefaultSectorSize);
  ctx.prepare_read(buf.data(), kDefaultSectorSize, 0);

  reader.submit_batch(&ctx);

  size_t completed = reader.reap_completions(&ctx);
  EXPECT_EQ(completed, 1);
}

// Test reap_completions with null context
TEST_F(AlignedFileReaderTest, ReapCompletionsNullContextTest) {
  create_test_file(1);

  AlignedFileReader reader(test_file_);
  size_t completed = reader.reap_completions(nullptr);
  EXPECT_EQ(completed, 0);
}

// Test submit_batch with null context
TEST_F(AlignedFileReaderTest, SubmitBatchNullContextTest) {
  create_test_file(1);

  AlignedFileReader reader(test_file_);
  size_t submitted = reader.submit_batch(nullptr);
  EXPECT_EQ(submitted, 0);
}

// Test submit_batch when file not open
TEST_F(AlignedFileReaderTest, SubmitBatchFileNotOpenTest) {
  AlignedFileReader reader;  // Not opened

  IOContext ctx(4);
  AlignedBuffer buf(kDefaultSectorSize);
  ctx.prepare_read(buf.data(), kDefaultSectorSize, 0);

  size_t submitted = reader.submit_batch(&ctx);
  EXPECT_EQ(submitted, 0);
}

// Test batch submit with write requests
TEST_F(AlignedFileReaderTest, BatchSubmitWriteTest) {
  AlignedFileReader reader;
  auto mode =
      AlignedFileReader::OpenMode::kReadWrite | AlignedFileReader::OpenMode::kCreate;
  auto status = reader.open(test_file_, mode);
  EXPECT_EQ(status, IOStatus::kSuccess);

  IOContext ctx(4);

  AlignedBuffer buf(kDefaultSectorSize);
  auto *data = reinterpret_cast<uint32_t *>(buf.data());
  data[0] = 0xDEADC0DE;

  ctx.prepare_write(buf.data(), kDefaultSectorSize, 0);

  size_t submitted = reader.submit_batch(&ctx);
  EXPECT_EQ(submitted, 1);

  // Verify write
  AlignedBuffer read_buf(kDefaultSectorSize);
  ssize_t bytes = reader.read(read_buf.data(), kDefaultSectorSize, 0);
  EXPECT_EQ(bytes, static_cast<ssize_t>(kDefaultSectorSize));
  auto *read_data = reinterpret_cast<uint32_t *>(read_buf.data());
  EXPECT_EQ(read_data[0], 0xDEADC0DE);
}

#ifdef ALAYA_OS_LINUX
// Test sync with data_only flag (Linux only - fdatasync)
TEST_F(AlignedFileReaderTest, SyncDataOnlyTest) {
  AlignedFileReader reader;
  auto mode =
      AlignedFileReader::OpenMode::kReadWrite | AlignedFileReader::OpenMode::kCreate;
  auto status = reader.open(test_file_, mode);
  EXPECT_EQ(status, IOStatus::kSuccess);

  // Write some data
  AlignedBuffer buf(kDefaultSectorSize);
  reader.write(buf.data(), kDefaultSectorSize, 0);

  // Sync data only
  status = reader.sync(true);
  EXPECT_EQ(status, IOStatus::kSuccess);

  // Also test full sync
  status = reader.sync(false);
  EXPECT_EQ(status, IOStatus::kSuccess);
}
#endif

// Test IOStatus to string for all values
TEST_F(AlignedFileReaderTest, IOStatusToStringAllValuesTest) {
  EXPECT_STREQ(io_status_to_string(IOStatus::kSuccess), "Success");
  EXPECT_STREQ(io_status_to_string(IOStatus::kError), "Error");
  EXPECT_STREQ(io_status_to_string(IOStatus::kEOF), "EOF");
  EXPECT_STREQ(io_status_to_string(IOStatus::kNotAligned), "NotAligned");
  EXPECT_STREQ(io_status_to_string(IOStatus::kFileNotOpen), "FileNotOpen");
  EXPECT_STREQ(io_status_to_string(IOStatus::kInvalidArg), "InvalidArg");
  EXPECT_STREQ(io_status_to_string(IOStatus::kPending), "Pending");
  EXPECT_STREQ(io_status_to_string(IOStatus::kCancelled), "Cancelled");
  EXPECT_STREQ(io_status_to_string(IOStatus::kNotSupported), "NotSupported");
  // Test unknown value
  EXPECT_STREQ(io_status_to_string(static_cast<IOStatus>(255)), "Unknown");
}

// Test opening file that's already open (should close first)
TEST_F(AlignedFileReaderTest, ReopenFileTest) {
  create_test_file(1);

  AlignedFileReader reader(test_file_);
  EXPECT_TRUE(reader.is_open());

  // Open again (should close first, then open)
  auto status = reader.open(test_file_, AlignedFileReader::OpenMode::kReadOnly);
  EXPECT_EQ(status, IOStatus::kSuccess);
  EXPECT_TRUE(reader.is_open());
}

// Test async read EOF
TEST_F(AlignedFileReaderTest, AsyncReadEOFTest) {
  create_test_file(1);  // Only 1 sector

  AlignedFileReader reader(test_file_);

  // Try to read at offset that partially overlaps with file
  AlignedBuffer buf(kDefaultSectorSize);
  IORequest req(buf.data(), kDefaultSectorSize, kDefaultSectorSize, false);

  [[maybe_unused]] auto status = reader.read_async(&req);
  // Should complete with EOF or Success depending on how much was read
  EXPECT_TRUE(req.is_complete());
  // bytes_transferred should be 0 or less than requested
  EXPECT_LE(req.bytes_transferred_, static_cast<ssize_t>(kDefaultSectorSize));
}

}  // namespace alaya
