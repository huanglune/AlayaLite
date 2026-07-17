// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// SEGMENT_OP payload codec (segment_op_wal.hpp): round-trip every op kind,
// header/lineage fields, framing integration through the shared WAL7 envelope,
// and decode-time validation of version/kind/size/truncation/trailing bytes.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "index/graph/laser/qg/segment_op_wal.hpp"
#include "wal/frame.hpp"

namespace alaya::laser {
namespace {

std::vector<std::byte> make_bytes(std::size_t n, std::uint8_t seed) {
  std::vector<std::byte> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = static_cast<std::byte>((seed + i * 31U) & 0xffU);
  }
  return out;
}

constexpr std::uint64_t kSegId = 0xA1B2C3D4E5F60718ULL;
constexpr std::uint64_t kGen = 42;

TEST(SegmentOpCodec, RowPatchRoundTrip) {
  const auto payload = make_bytes(1024, 7);
  const auto encoded = encode_row_patch(kSegId, kGen, /*pid=*/13, /*offset=*/1024 + 4096, payload);
  const auto op = decode_segment_op(encoded);
  EXPECT_EQ(op.payload_version, kSegmentOpPayloadVersion);
  EXPECT_EQ(op.segment_id, kSegId);
  EXPECT_EQ(op.segment_generation, kGen);
  EXPECT_EQ(op.kind, SegmentOpKind::row_patch);
  EXPECT_EQ(op.pid, 13U);
  EXPECT_EQ(op.offset, 1024U + 4096U);
  EXPECT_EQ(op.bytes, payload);
}

TEST(SegmentOpCodec, TombstoneRoundTrip) {
  const auto encoded = encode_tombstone(kSegId, kGen, /*pid=*/99);
  const auto op = decode_segment_op(encoded);
  EXPECT_EQ(op.kind, SegmentOpKind::tombstone);
  EXPECT_EQ(op.pid, 99U);
}

TEST(SegmentOpCodec, ConsolidateMarkersRoundTrip) {
  for (auto kind : {SegmentOpKind::consolidate_begin, SegmentOpKind::consolidate_end}) {
    const auto encoded = encode_consolidate_marker(kSegId, kGen, kind, /*epoch=*/7);
    const auto op = decode_segment_op(encoded);
    EXPECT_EQ(op.kind, kind);
    EXPECT_EQ(op.epoch, 7U);
  }
  EXPECT_THROW((void)encode_consolidate_marker(kSegId, kGen, SegmentOpKind::publish, 1),
               std::invalid_argument);
}

TEST(SegmentOpCodec, PublishRoundTrip) {
  const auto encoded = encode_publish(kSegId, kGen, /*watermark=*/123456);
  const auto op = decode_segment_op(encoded);
  EXPECT_EQ(op.kind, SegmentOpKind::publish);
  EXPECT_EQ(op.watermark, 123456U);
}

TEST(SegmentOpCodec, SuperblockFlipRoundTrip) {
  const auto image = make_bytes(kSegmentSuperblockImageBytes, 0x5A);
  const auto encoded = encode_superblock_flip(kSegId, kGen, /*target_slot=*/1, image);
  const auto op = decode_segment_op(encoded);
  EXPECT_EQ(op.kind, SegmentOpKind::superblock_flip);
  EXPECT_EQ(op.target_slot, 1);
  ASSERT_EQ(op.bytes.size(), kSegmentSuperblockImageBytes);
  EXPECT_EQ(op.bytes, image);
  EXPECT_THROW((void)encode_superblock_flip(kSegId, kGen, 0, make_bytes(511, 1)),
               std::invalid_argument);
}

// The op payload rides inside the shared WAL7 envelope as record type 5 and is
// recovered by the framework scan, then decoded by this codec — end to end.
TEST(SegmentOpCodec, RidesInsideWal7EnvelopeAsTypeFive) {
  const auto payload = encode_publish(kSegId, kGen, 777);
  const auto frame = alaya::wal::make_frame(kSegmentOpRecordType, 0, 1, 1, payload);
  const auto scanned = alaya::wal::scan(frame);
  ASSERT_EQ(scanned.frames.size(), 1U);
  EXPECT_EQ(scanned.frames[0].type, kSegmentOpRecordType);
  const auto op = decode_segment_op(scanned.frames[0].payload);
  EXPECT_EQ(op.kind, SegmentOpKind::publish);
  EXPECT_EQ(op.watermark, 777U);
}

TEST(SegmentOpCodec, RejectsBadVersionKindAndTruncation) {
  // Unknown payload version.
  {
    auto encoded = encode_publish(kSegId, kGen, 1);
    encoded[0] = static_cast<std::byte>(2);  // bump payload_version low byte
    EXPECT_THROW((void)decode_segment_op(encoded), std::invalid_argument);
  }
  // Unknown kind byte (0 and 7 are out of range).
  {
    auto encoded = encode_publish(kSegId, kGen, 1);
    encoded[18] = static_cast<std::byte>(7);  // kind byte is right after version(2)+2*u64(16)
    EXPECT_THROW((void)decode_segment_op(encoded), std::invalid_argument);
    encoded[18] = static_cast<std::byte>(0);
    EXPECT_THROW((void)decode_segment_op(encoded), std::invalid_argument);
  }
  // Truncated payload.
  {
    auto encoded = encode_tombstone(kSegId, kGen, 5);
    encoded.pop_back();
    EXPECT_THROW((void)decode_segment_op(encoded), std::invalid_argument);
  }
  // Trailing bytes.
  {
    auto encoded = encode_tombstone(kSegId, kGen, 5);
    encoded.push_back(std::byte{0});
    EXPECT_THROW((void)decode_segment_op(encoded), std::invalid_argument);
  }
}

}  // namespace
}  // namespace alaya::laser
