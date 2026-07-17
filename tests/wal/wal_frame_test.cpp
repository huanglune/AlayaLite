// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// Framework-layer tests for the shared Physical WAL v1 envelope (wal/frame.hpp).
// Locks the byte format (golden), the structural scan, the one-envelope /
// two-family contract (unified-wal-vocabulary.md acceptance 3), and the
// collection layer's loud rejection of a foreign record type.

#include <gtest/gtest.h>
#include <unistd.h>

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

TEST(WalDecoder, ReadsPrimitivesTakesRawBytesAndDetectsTruncation) {
  std::vector<std::byte> buf;
  put_u16(buf, 0x0102);
  put_u32(buf, 0x03040506U);
  put_u64(buf, 0x0708090A0B0C0D0EULL);
  const auto tail = bytes_of({0xCA, 0xFE, 0xBA, 0xBE});
  buf.insert(buf.end(), tail.begin(), tail.end());
  Decoder decoder(buf);
  EXPECT_EQ(decoder.u16(), 0x0102);
  EXPECT_EQ(decoder.u32(), 0x03040506U);
  EXPECT_EQ(decoder.u64(), 0x0708090A0B0C0D0EULL);
  EXPECT_EQ(decoder.remaining(), tail.size());
  const auto taken = decoder.take(tail.size());
  EXPECT_TRUE(std::equal(taken.begin(), taken.end(), tail.begin()));
  EXPECT_TRUE(decoder.empty());
  EXPECT_THROW((void)decoder.take(1), std::invalid_argument);
}

std::vector<char> slurp(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// unified-wal-vocabulary.md clause J / the manifest W1 rule: a structurally
// valid but foreign (type 5) frame inside a *collection* WAL is a cross-family
// contamination. The collection layer must fail loudly with a typed error and
// must NOT truncate, reset, or otherwise mutate the file (no silent data loss).
TEST(CollectionScanCrossFamily, UnknownRecordTypeIsHardErrorAndLeavesFileByteIdentical) {
  using alaya::internal::collection::CollectionLogicalWal;
  const auto root = std::filesystem::temp_directory_path() /
                    ("wal_xfam_" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);
  const auto wal_path = root / ".alaya_internal" / "collection_wal_v1" / "logical.wal";
  {
    WalFile wal(wal_path);
    wal.append(/*type=*/2, 0, 1, 0, bytes_of({1}), WalFile::Sync::fsync);  // COMMIT (valid 1-4)
    wal.append(/*type=*/5, 0, 2, 0, bytes_of({2}), WalFile::Sync::fsync);  // SEGMENT_OP (foreign)
  }
  const auto before = slurp(wal_path);
  ASSERT_FALSE(before.empty());

  // The framework scan accepts both frames structurally...
  const auto raw = WalFile::scan_path(wal_path);
  ASSERT_EQ(raw.frames.size(), 2U);
  EXPECT_FALSE(raw.stopped_at_corrupt_or_torn_tail);

  // ...but the collection layer rejects the foreign type with a typed error.
  const auto scanned = CollectionLogicalWal::scan_file(wal_path);
  EXPECT_FALSE(scanned.ok());

  // And a full open (which truncates a genuine torn tail) must NOT touch the
  // file when the failure is a foreign record type.
  const auto opened = CollectionLogicalWal::open(root);
  EXPECT_FALSE(opened.ok());
  const auto after = slurp(wal_path);
  EXPECT_EQ(before, after) << "collection WAL must be byte-identical after a rejected foreign type";
  std::filesystem::remove_all(root);
}

// The structural-corruption (torn tail) path is unchanged: open heals it by
// truncating to the last verified boundary. This is deliberately distinct from
// the foreign-type case above.
TEST(CollectionScanCrossFamily, TornTailStillTruncatesToVerifiedBoundary) {
  using alaya::internal::collection::CollectionLogicalWal;
  const auto root = std::filesystem::temp_directory_path() /
                    ("wal_torn_" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);
  const auto wal_path = root / ".alaya_internal" / "collection_wal_v1" / "logical.wal";
  {
    WalFile wal(wal_path);
    wal.append(/*type=*/2, 0, 1, 0, bytes_of({1}), WalFile::Sync::fsync);
  }
  const auto clean_size = slurp(wal_path).size();
  {
    std::ofstream out(wal_path, std::ios::binary | std::ios::app);
    const char junk[6] = {'W', 'A', 'L', '7', 1, 0};
    out.write(junk, sizeof(junk));
  }
  const auto opened = CollectionLogicalWal::open(root);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  EXPECT_EQ(slurp(wal_path).size(), clean_size) << "torn tail should be healed to the boundary";
  std::filesystem::remove_all(root);
}

// B-2C-04: append() reports each frame's FrameLocation, visit_frames streams them
// back one at a time (bounded memory), and read_frame re-reads exactly one frame
// with full CRC re-validation. Wire is unchanged (the golden test above still
// pins make_frame).
TEST(WalFileStreamingApi, AppendLocationsVisitAndReadFrame) {
  const auto root = std::filesystem::temp_directory_path() /
                    ("wal_frame_stream_" + std::to_string(::getpid()));
  std::filesystem::create_directories(root);
  const auto path = root / "seg.opwal";
  std::vector<FrameLocation> locations;
  std::vector<std::vector<std::byte>> payloads;
  {
    WalFile wal(path);
    for (unsigned i = 0; i < 5; ++i) {
      auto payload = bytes_of({i, static_cast<unsigned>(i * 7 + 1), 0xAB, 0xCD});
      const auto loc = wal.append(/*type=*/5, /*flags=*/0, /*op_id=*/i, /*batch_id=*/i * 2,
                                  payload, WalFile::Sync::flush);
      locations.push_back(loc);
      payloads.push_back(std::move(payload));
    }
  }
  // Locations are contiguous and ascending.
  EXPECT_EQ(locations.front().offset, 0U);
  for (std::size_t i = 1; i < locations.size(); ++i) {
    EXPECT_EQ(locations[i].offset, locations[i - 1].offset + locations[i - 1].size);
  }
  // visit_frames streams every frame in order; stop early on demand.
  std::vector<ScannedFrame> visited;
  WalFile::visit_frames(path, [&](const ScannedFrame &f) {
    visited.push_back(f);
    return true;
  });
  ASSERT_EQ(visited.size(), payloads.size());
  for (std::size_t i = 0; i < visited.size(); ++i) {
    EXPECT_EQ(visited[i].op_id, i);
    EXPECT_EQ(visited[i].batch_id, i * 2);
    EXPECT_EQ(visited[i].offset, locations[i].offset);
    EXPECT_EQ(visited[i].size, locations[i].size);
    EXPECT_EQ(visited[i].payload, payloads[i]);
  }
  std::size_t seen = 0;
  WalFile::visit_frames(path, [&](const ScannedFrame &) {
    ++seen;
    return seen < 2;  // ask to stop after the second frame
  });
  EXPECT_EQ(seen, 2U);
  // read_frame re-reads exactly one frame at its location, CRC-validated.
  for (std::size_t i = 0; i < locations.size(); ++i) {
    const auto f = WalFile::read_frame(path, locations[i]);
    EXPECT_EQ(f.op_id, i);
    EXPECT_EQ(f.payload, payloads[i]);
  }
  // A mangled location fails closed (wrong size / off-boundary offset).
  EXPECT_THROW((void)WalFile::read_frame(path, FrameLocation{locations[1].offset + 1,
                                                             locations[1].size}),
               std::exception);
  EXPECT_THROW((void)WalFile::read_frame(path, FrameLocation{locations[1].offset, 3}),
               std::exception);
  // Re-open continues appending after the recovered boundary (append cursor is
  // restored from the last verified frame).
  {
    WalFile wal(path);
    auto payload = bytes_of({0xEE});
    const auto loc = wal.append(6, 0, 99, 0, payload, WalFile::Sync::fsync);
    EXPECT_EQ(loc.offset, locations.back().offset + locations.back().size);
    const auto f = WalFile::read_frame(path, loc);
    EXPECT_EQ(f.op_id, 99U);
  }
  std::filesystem::remove_all(root);
}

// wal-2c BLOCKER-3: scan_structure_streaming must agree with scan_path on the last
// verified boundary and the torn-tail flag, while holding at most one frame in memory
// (it never retains payloads). This is the recovery-open path the QG op-WAL now uses so a
// multi-GiB log opens in O(max frame) instead of O(file).
TEST(WalFileStreamingApi, StructureStreamingScanMatchesEagerScan) {
  const auto root = std::filesystem::temp_directory_path() /
                    ("wal_frame_streamscan_" + std::to_string(::getpid()));
  std::filesystem::create_directories(root);
  const auto path = root / "seg.opwal";
  {
    WalFile wal(path);
    for (unsigned i = 0; i < 7; ++i) {
      wal.append(/*type=*/5, 0, i, i, bytes_of({i, 0xAA, 0xBB, static_cast<unsigned>(i)}),
                 WalFile::Sync::fsync);
    }
  }
  // Clean log: streaming and eager agree on valid_bytes; streaming keeps no frames.
  const auto eager_clean = WalFile::scan_path(path);
  const auto stream_clean = WalFile::scan_structure_streaming(path);
  EXPECT_FALSE(eager_clean.stopped_at_corrupt_or_torn_tail);
  EXPECT_FALSE(stream_clean.stopped_at_corrupt_or_torn_tail);
  EXPECT_EQ(stream_clean.valid_bytes, eager_clean.valid_bytes);
  EXPECT_TRUE(stream_clean.frames.empty()) << "streaming scan must not retain payloads";
  EXPECT_EQ(eager_clean.frames.size(), 7U);
  // Append a partial (torn) trailing frame: both must stop at the same verified boundary.
  {
    std::ofstream out(path, std::ios::binary | std::ios::app);
    const char junk[9] = {'W', 'A', 'L', '7', 1, 5, 0, 0, 0};
    out.write(junk, sizeof(junk));
  }
  const auto eager_torn = WalFile::scan_path(path);
  const auto stream_torn = WalFile::scan_structure_streaming(path);
  EXPECT_TRUE(eager_torn.stopped_at_corrupt_or_torn_tail);
  EXPECT_TRUE(stream_torn.stopped_at_corrupt_or_torn_tail);
  EXPECT_EQ(stream_torn.valid_bytes, eager_torn.valid_bytes);
  EXPECT_EQ(stream_torn.valid_bytes, stream_clean.valid_bytes) << "torn tail heals to the boundary";
  EXPECT_TRUE(stream_torn.frames.empty());
  std::filesystem::remove_all(root);
}

}  // namespace
}  // namespace alaya::wal
