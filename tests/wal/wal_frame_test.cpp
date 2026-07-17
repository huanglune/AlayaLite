// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// Framework-layer tests for the shared Physical WAL v1 envelope (wal/frame.hpp).
// Locks the byte format (golden), the structural scan, the one-envelope /
// two-family contract (unified-wal-vocabulary.md acceptance 3), and the
// collection layer's loud rejection of a foreign record type.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "index/collection/logical_wal.hpp"
#include "wal/frame.hpp"

namespace alaya::wal {
namespace {

std::vector<std::byte> bytes_of(std::initializer_list<unsigned> values) {
  std::vector<std::byte> out;
  out.reserve(values.size());
  for (unsigned v : values) {
    out.push_back(static_cast<std::byte>(v));
  }
  return out;
}

// Byte-for-byte lock. If make_frame ever changes the header/trailer layout or
// the CRC, this golden breaks — which is exactly the regression the migration
// off logical_wal_detail must never introduce.
TEST(WalFrameGolden, MakeFrameByteSequenceIsStable) {
  static constexpr std::array<unsigned char, 44> kGolden = {
      0x57, 0x41, 0x4c, 0x37, 0x01, 0x00, 0x02, 0x01, 0x2c, 0x00, 0x00, 0x00,
      0x04, 0x00, 0x00, 0x00, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,
      0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x6d, 0xd7, 0x12, 0xd0,
      0xde, 0xad, 0xbe, 0xef, 0x45, 0x4e, 0x44, 0x37,
  };
  const auto payload = bytes_of({0xDE, 0xAD, 0xBE, 0xEF});
  const auto frame = make_frame(/*type=*/2, /*flags=*/0x01, /*op_id=*/0x0102030405060708ULL,
                                /*batch_id=*/0x1112131415161718ULL, payload);
  ASSERT_EQ(frame.size(), kGolden.size());
  for (std::size_t i = 0; i < frame.size(); ++i) {
    EXPECT_EQ(std::to_integer<unsigned char>(frame[i]), kGolden[i]) << "byte " << i;
  }
}

TEST(WalFrameScan, RoundTripsEveryHeaderField) {
  const auto payload = bytes_of({1, 2, 3, 4, 5});
  const auto frame = make_frame(7, 0x80, 0xAABBCCDDULL, 0x99ULL, payload);
  const auto result = scan(frame);
  ASSERT_EQ(result.frames.size(), 1U);
  EXPECT_FALSE(result.stopped_at_corrupt_or_torn_tail);
  EXPECT_EQ(result.valid_bytes, frame.size());
  const auto &f = result.frames[0];
  EXPECT_EQ(f.type, 7);
  EXPECT_EQ(f.flags, 0x80);
  EXPECT_EQ(f.op_id, 0xAABBCCDDULL);
  EXPECT_EQ(f.batch_id, 0x99ULL);
  EXPECT_EQ(f.payload, payload);
  EXPECT_EQ(f.offset, 0U);
  EXPECT_EQ(f.size, frame.size());
}

TEST(WalFrameScan, ZeroTypeIsRejectedAsTorn) {
  auto frame = make_frame(1, 0, 0, 0, {});
  frame[6] = std::byte{0};  // clobber the record type; CRC now mismatches too
  const auto result = scan(frame);
  EXPECT_TRUE(result.stopped_at_corrupt_or_torn_tail);
  EXPECT_TRUE(result.frames.empty());
}

// Contract acceptance 3: one envelope carries both op families. A stream that
// interleaves a collection record type (2) and a SEGMENT_OP record type (5)
// scans back in order; the framing layer never rejects a foreign-but-valid
// type — that is the caller's job.
TEST(WalFileMixedFamily, InterleavedTypesScanInLogOrder) {
  const auto dir = std::filesystem::temp_directory_path() /
                   ("wal_mixed_" + std::to_string(::getpid()));
  std::filesystem::remove_all(dir);
  const auto path = dir / "mixed.wal";
  const std::array<std::uint8_t, 4> types{2, 5, 2, 5};
  {
    WalFile wal(path);
    std::uint64_t op = 1;
    for (auto type : types) {
      const auto payload = bytes_of({static_cast<unsigned>(type), static_cast<unsigned>(op)});
      wal.append(type, 0, op, op * 10, payload, WalFile::Sync::fsync);
      ++op;
    }
  }
  const auto result = WalFile::scan_path(path);
  ASSERT_EQ(result.frames.size(), types.size());
  EXPECT_FALSE(result.stopped_at_corrupt_or_torn_tail);
  for (std::size_t i = 0; i < types.size(); ++i) {
    EXPECT_EQ(result.frames[i].type, types[i]) << "frame " << i;
    EXPECT_EQ(result.frames[i].op_id, i + 1);
  }
  std::filesystem::remove_all(dir);
}

TEST(WalFile, ReopenRecoversDurableFramesAndTruncatesTornTail) {
  const auto dir = std::filesystem::temp_directory_path() /
                   ("wal_reopen_" + std::to_string(::getpid()));
  std::filesystem::remove_all(dir);
  const auto path = dir / "reopen.wal";
  {
    WalFile wal(path);
    wal.append(2, 0, 1, 0, bytes_of({1}), WalFile::Sync::fsync);
    wal.append(2, 0, 2, 0, bytes_of({2}), WalFile::Sync::fsync);
  }
  // Append 5 bytes of garbage: a torn tail.
  {
    std::ofstream out(path, std::ios::binary | std::ios::app);
    const char junk[5] = {'W', 'A', 'L', '7', 0};
    out.write(junk, sizeof(junk));
  }
  {
    WalFile wal(path);  // constructor scans + truncates the torn tail
    const auto &scanned = wal.recovery_scan();
    EXPECT_EQ(scanned.frames.size(), 2U);
    EXPECT_TRUE(scanned.stopped_at_corrupt_or_torn_tail);
    wal.append(2, 0, 3, 0, bytes_of({3}), WalFile::Sync::fsync);
  }
  const auto result = WalFile::scan_path(path);
  ASSERT_EQ(result.frames.size(), 3U);  // torn junk was healed, third append is clean
  EXPECT_FALSE(result.stopped_at_corrupt_or_torn_tail);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(result.frames[i].op_id, i + 1);
  }
  std::filesystem::remove_all(dir);
}

TEST(WalFile, ResetToSingleFrameLeavesExactlyOneDurableMarker) {
  const auto dir = std::filesystem::temp_directory_path() /
                   ("wal_reset_" + std::to_string(::getpid()));
  std::filesystem::remove_all(dir);
  const auto path = dir / "reset.wal";
  WalFile wal(path);
  for (std::uint64_t op = 1; op <= 5; ++op) {
    wal.append(2, 0, op, 0, bytes_of({static_cast<unsigned>(op)}), WalFile::Sync::buffered);
  }
  wal.reset_to_single_frame(6, 0, 42, 42, bytes_of({0xAB}));
  const auto result = WalFile::scan_path(path);
  ASSERT_EQ(result.frames.size(), 1U);
  EXPECT_EQ(result.frames[0].type, 6);
  EXPECT_EQ(result.frames[0].op_id, 42U);
  std::filesystem::remove_all(dir);
}

TEST(WalDecoder, ReadsPrimitivesAndDetectsTruncation) {
  std::vector<std::byte> buf;
  put_u16(buf, 0x0102);
  put_u32(buf, 0x03040506U);
  put_u64(buf, 0x0708090A0B0C0D0EULL);
  Decoder decoder(buf);
  EXPECT_EQ(decoder.u16(), 0x0102);
  EXPECT_EQ(decoder.u32(), 0x03040506U);
  EXPECT_EQ(decoder.u64(), 0x0708090A0B0C0D0EULL);
  EXPECT_TRUE(decoder.empty());
  EXPECT_THROW((void)decoder.u8(), std::invalid_argument);
}

// unified-wal-vocabulary.md item 2 / the manifest W1 rule: a structurally valid
// but foreign (type 5) frame inside a *collection* WAL is a cross-family
// contamination. The collection scan must fail loudly, never silently truncate
// and drop committed data.
TEST(CollectionScanCrossFamily, UnknownRecordTypeIsHardErrorNotSilentTruncation) {
  using alaya::internal::collection::CollectionLogicalWal;
  const auto dir = std::filesystem::temp_directory_path() /
                   ("wal_xfam_" + std::to_string(::getpid()));
  std::filesystem::remove_all(dir);
  const auto path = dir / "contaminated.wal";
  {
    WalFile wal(path);
    wal.append(/*type=*/2, 0, 1, 0, bytes_of({1}), WalFile::Sync::fsync);  // COMMIT (valid 1-4)
    wal.append(/*type=*/5, 0, 2, 0, bytes_of({2}), WalFile::Sync::fsync);  // SEGMENT_OP (foreign)
  }
  // The framework scan accepts both frames structurally.
  const auto raw = WalFile::scan_path(path);
  ASSERT_EQ(raw.frames.size(), 2U);
  EXPECT_FALSE(raw.stopped_at_corrupt_or_torn_tail);
  // The collection layer rejects the foreign type instead of truncating.
  const auto scanned = CollectionLogicalWal::scan_file(path);
  EXPECT_FALSE(scanned.ok());
  std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace alaya::wal
