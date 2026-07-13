// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "index/collection/logical_wal.hpp"

namespace alaya::internal::collection {
namespace {

class LogicalWalTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ =
        std::filesystem::temp_directory_path() /
        ("alaya-collection-logical-wal-" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    std::filesystem::remove_all(root_);
  }

  void TearDown() override { std::filesystem::remove_all(root_); }

  std::filesystem::path root_{};
};

TEST_F(LogicalWalTest, FramesCarryLengthChecksumTypeAndGlobalOrdering) {
  auto opened = CollectionLogicalWal::open(root_);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto wal = std::move(opened).value();
  const std::array<std::byte, 3> payload{std::byte{0x01}, std::byte{0x02}, std::byte{0x03}};
  ASSERT_TRUE(
      wal->append(LogicalWalRecordType::prepare, 7, 11, 10, payload, LogicalWalSync::flush).ok());
  ASSERT_TRUE(wal->append(LogicalWalRecordType::commit, 1, 11, 10, {}, LogicalWalSync::fsync).ok());
  ASSERT_TRUE(
      wal->append(LogicalWalRecordType::publish_marker, 1, 11, 10, {}, LogicalWalSync::flush).ok());

  auto scanned = CollectionLogicalWal::scan_file(wal->path());
  ASSERT_TRUE(scanned.ok());
  ASSERT_EQ(scanned.value().frames.size(), 3U);
  EXPECT_FALSE(scanned.value().stopped_at_corrupt_or_torn_tail);
  EXPECT_EQ(scanned.value().frames[0].type, LogicalWalRecordType::prepare);
  EXPECT_EQ(scanned.value().frames[0].flags, 7U);
  EXPECT_EQ(scanned.value().frames[0].op_id, 11U);
  EXPECT_EQ(scanned.value().frames[0].batch_id, 10U);
  EXPECT_EQ(scanned.value().frames[0].payload,
            std::vector<std::byte>(payload.begin(), payload.end()));
  EXPECT_EQ(scanned.value().frames[1].type, LogicalWalRecordType::commit);
  EXPECT_EQ(scanned.value().frames[2].type, LogicalWalRecordType::publish_marker);
}

TEST_F(LogicalWalTest, TornOrChecksumDamagedTailStopsAtLastCompleteFrameAndOpenHealsTail) {
  std::filesystem::path path;
  std::uint64_t first_frame_end{};
  {
    auto opened = CollectionLogicalWal::open(root_);
    ASSERT_TRUE(opened.ok());
    auto wal = std::move(opened).value();
    ASSERT_TRUE(
        wal->append(LogicalWalRecordType::prepare, 0, 1, 1, {}, LogicalWalSync::flush).ok());
    ASSERT_TRUE(wal->append(LogicalWalRecordType::commit, 1, 1, 1, {}, LogicalWalSync::fsync).ok());
    path = wal->path();
    auto intact = CollectionLogicalWal::scan_file(path);
    ASSERT_TRUE(intact.ok());
    ASSERT_EQ(intact.value().frames.size(), 2U);
    first_frame_end = intact.value().frames[0].size;
  }
  const auto original_size = std::filesystem::file_size(path);
  ASSERT_GT(original_size, first_frame_end + 1);
  std::filesystem::resize_file(path, original_size - 1);

  auto torn = CollectionLogicalWal::scan_file(path);
  ASSERT_TRUE(torn.ok());
  EXPECT_TRUE(torn.value().stopped_at_corrupt_or_torn_tail);
  ASSERT_EQ(torn.value().frames.size(), 1U);
  EXPECT_EQ(torn.value().valid_bytes, first_frame_end);

  auto reopened = CollectionLogicalWal::open(root_);
  ASSERT_TRUE(reopened.ok());
  EXPECT_TRUE(reopened.value()->recovery_scan().stopped_at_corrupt_or_torn_tail);
  EXPECT_EQ(std::filesystem::file_size(path), first_frame_end);
}

TEST_F(LogicalWalTest, CorruptFrameNeverResynchronizesToACommitAfterTheDamage) {
  std::filesystem::path path;
  std::uint64_t first_size{};
  {
    auto opened = CollectionLogicalWal::open(root_);
    ASSERT_TRUE(opened.ok());
    auto wal = std::move(opened).value();
    ASSERT_TRUE(
        wal->append(LogicalWalRecordType::prepare, 0, 1, 1, {}, LogicalWalSync::flush).ok());
    ASSERT_TRUE(wal->append(LogicalWalRecordType::commit, 1, 1, 1, {}, LogicalWalSync::fsync).ok());
    path = wal->path();
    auto scanned = CollectionLogicalWal::scan_file(path);
    ASSERT_TRUE(scanned.ok());
    first_size = scanned.value().frames[0].size;
  }
  {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(file);
    file.seekp(8);  // The covered frame-length field.
    const char damaged = static_cast<char>(0xff);
    file.write(&damaged, 1);
  }
  auto corrupted = CollectionLogicalWal::scan_file(path);
  ASSERT_TRUE(corrupted.ok());
  EXPECT_TRUE(corrupted.value().stopped_at_corrupt_or_torn_tail);
  EXPECT_TRUE(corrupted.value().frames.empty());
  EXPECT_EQ(corrupted.value().valid_bytes, 0U);
  EXPECT_GT(std::filesystem::file_size(path), first_size);
}

TEST_F(LogicalWalTest, CheckpointAtomicallyCutsTheWalToOneDurableMarker) {
  auto opened = CollectionLogicalWal::open(root_);
  ASSERT_TRUE(opened.ok());
  auto wal = std::move(opened).value();
  ASSERT_TRUE(wal->append(LogicalWalRecordType::prepare, 0, 8, 8, {}, LogicalWalSync::flush).ok());
  ASSERT_TRUE(wal->append(LogicalWalRecordType::commit, 1, 8, 8, {}, LogicalWalSync::fsync).ok());
  ASSERT_TRUE(wal->reset_to_checkpoint(8).ok());
  auto scanned = CollectionLogicalWal::scan_file(wal->path());
  ASSERT_TRUE(scanned.ok());
  ASSERT_EQ(scanned.value().frames.size(), 1U);
  EXPECT_EQ(scanned.value().frames[0].type, LogicalWalRecordType::checkpoint);
  EXPECT_EQ(scanned.value().frames[0].op_id, 8U);
}

}  // namespace
}  // namespace alaya::internal::collection
