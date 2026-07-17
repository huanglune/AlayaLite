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
  // Unknown kind byte (0 and 9 are out of range; 7/8 are now valid label ops).
  {
    auto encoded = encode_publish(kSegId, kGen, 1);
    encoded[18] = static_cast<std::byte>(9);  // kind byte is right after version(2)+2*u64(16)
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

// Frozen wire format for kinds 1-6: a byte-for-byte golden captured from the
// current codec. It guards the kind=5 emit path against the W3 publish_with_record
// refactor and any future codec drift. Fixed inputs (see the golden generator).
TEST(SegmentOpCodec, Kind1Through6GoldenBytesAreFrozen) {
  constexpr std::uint64_t sid = 0x0102030405060708ULL;
  constexpr std::uint64_t gen = 0x1112131415161718ULL;
  auto to_u8 = [](const std::vector<std::byte> &v) {
    std::vector<std::uint8_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = std::to_integer<std::uint8_t>(v[i]);
    return out;
  };
  const std::vector<std::byte> rp_bytes = {std::byte{0xAA}, std::byte{0xBB}, std::byte{0xCC},
                                           std::byte{0xDD}};
  EXPECT_EQ(
      to_u8(encode_row_patch(sid, gen, 0x2122232425262728ULL, 0x0000000000001800ULL, rp_bytes)),
      (std::vector<std::uint8_t>{0x01, 0x00, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x18,
                                 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x01, 0x28, 0x27, 0x26,
                                 0x25, 0x24, 0x23, 0x22, 0x21, 0x00, 0x18, 0x00, 0x00, 0x00, 0x00,
                                 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xaa, 0xbb, 0xcc, 0xdd}));
  EXPECT_EQ(to_u8(encode_tombstone(sid, gen, 0x3132333435363738ULL)),
            (std::vector<std::uint8_t>{0x01, 0x00, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02,
                                       0x01, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11,
                                       0x02, 0x38, 0x37, 0x36, 0x35, 0x34, 0x33, 0x32, 0x31}));
  EXPECT_EQ(to_u8(encode_consolidate_marker(sid, gen, SegmentOpKind::consolidate_begin,
                                            0x4142434445464748ULL)),
            (std::vector<std::uint8_t>{0x01, 0x00, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02,
                                       0x01, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11,
                                       0x03, 0x48, 0x47, 0x46, 0x45, 0x44, 0x43, 0x42, 0x41}));
  EXPECT_EQ(to_u8(encode_consolidate_marker(sid, gen, SegmentOpKind::consolidate_end,
                                            0x5152535455565758ULL)),
            (std::vector<std::uint8_t>{0x01, 0x00, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02,
                                       0x01, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11,
                                       0x04, 0x58, 0x57, 0x56, 0x55, 0x54, 0x53, 0x52, 0x51}));
  EXPECT_EQ(to_u8(encode_publish(sid, gen, 0x6162636465666768ULL)),
            (std::vector<std::uint8_t>{0x01, 0x00, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02,
                                       0x01, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11,
                                       0x05, 0x68, 0x67, 0x66, 0x65, 0x64, 0x63, 0x62, 0x61}));
  // superblock_flip: the 512-byte image is opaque passthrough (roundtrip covers
  // it); freeze the 24-byte header/slot/length prefix and the framing size.
  std::vector<std::byte> img(kSegmentSuperblockImageBytes);
  for (std::size_t i = 0; i < img.size(); ++i) img[i] = static_cast<std::byte>((i * 7 + 3) & 0xFF);
  const auto flip = to_u8(encode_superblock_flip(sid, gen, 1, img));
  ASSERT_EQ(flip.size(), 19U + 1U + 4U + kSegmentSuperblockImageBytes);
  const std::vector<std::uint8_t> flip_prefix(flip.begin(), flip.begin() + 24);
  EXPECT_EQ(flip_prefix,
            (std::vector<std::uint8_t>{0x01, 0x00, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,
                                       0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x06, 0x01,
                                       0x00, 0x02, 0x00, 0x00}));
}

// Frozen wire for the 2A label ops kind=7/8 (W0 acceptance: kind=1..8 golden are
// frozen). The pid_generation field is exercised with a NON-zero value to lock its
// byte position -- the codec wire is unchanged across 2C even after generation is
// activated in W2 (only the replay-time semantics change, never the layout).
TEST(SegmentOpCodec, Kind7And8GoldenBytesAreFrozen) {
  constexpr std::uint64_t sid = 0x0102030405060708ULL;
  constexpr std::uint64_t gen = 0x1112131415161718ULL;
  auto to_u8 = [](const std::vector<std::byte> &v) {
    std::vector<std::uint8_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = std::to_integer<std::uint8_t>(v[i]);
    return out;
  };
  EXPECT_EQ(to_u8(encode_label_bind(sid, gen, /*tx_id=*/0x4142434445464748ULL,
                                    /*row_op_id=*/0x5152535455565758ULL, /*pid=*/0x61626364U,
                                    /*pid_generation=*/0x71727374U, /*label=*/0x8182838485868788ULL)),
            (std::vector<std::uint8_t>{
                0x01, 0x00, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x18, 0x17, 0x16,
                0x15, 0x14, 0x13, 0x12, 0x11, 0x07, 0x48, 0x47, 0x46, 0x45, 0x44, 0x43, 0x42,
                0x41, 0x58, 0x57, 0x56, 0x55, 0x54, 0x53, 0x52, 0x51, 0x64, 0x63, 0x62, 0x61,
                0x74, 0x73, 0x72, 0x71, 0x88, 0x87, 0x86, 0x85, 0x84, 0x83, 0x82, 0x81}));
  EXPECT_EQ(to_u8(encode_tx_publish(sid, gen, /*tx_id=*/0x4142434445464748ULL,
                                    /*new_pid_watermark=*/0x5152535455565758ULL,
                                    /*binding_count=*/0x6162636465666768ULL,
                                    /*applied_collection_op_id=*/0x7172737475767778ULL)),
            (std::vector<std::uint8_t>{
                0x01, 0x00, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x18, 0x17, 0x16,
                0x15, 0x14, 0x13, 0x12, 0x11, 0x08, 0x48, 0x47, 0x46, 0x45, 0x44, 0x43, 0x42,
                0x41, 0x58, 0x57, 0x56, 0x55, 0x54, 0x53, 0x52, 0x51, 0x68, 0x67, 0x66, 0x65,
                0x64, 0x63, 0x62, 0x61, 0x78, 0x77, 0x76, 0x75, 0x74, 0x73, 0x72, 0x71}));
}

TEST(SegmentOpCodec, LabelBindRoundTrip) {
  const auto encoded = encode_label_bind(kSegId, kGen, /*tx_id=*/0xABCDEF01ULL, /*row_op_id=*/3,
                                         /*pid=*/0x1234U, /*pid_generation=*/0,
                                         /*label=*/0xFEEDFACEULL);
  const auto op = decode_segment_op(encoded);
  EXPECT_EQ(op.kind, SegmentOpKind::label_bind);
  EXPECT_EQ(op.tx_id, 0xABCDEF01ULL);
  EXPECT_EQ(op.row_op_id, 3U);
  EXPECT_EQ(op.pid, 0x1234U);
  EXPECT_EQ(op.pid_generation, 0U);
  EXPECT_EQ(op.label, 0xFEEDFACEULL);
}

TEST(SegmentOpCodec, TxPublishRoundTrip) {
  const auto encoded = encode_tx_publish(kSegId, kGen, /*tx_id=*/7, /*new_pid_watermark=*/210,
                                         /*binding_count=*/10, /*applied_collection_op_id=*/99);
  const auto op = decode_segment_op(encoded);
  EXPECT_EQ(op.kind, SegmentOpKind::tx_publish);
  EXPECT_EQ(op.tx_id, 7U);
  EXPECT_EQ(op.new_pid_watermark, 210U);
  EXPECT_EQ(op.binding_count, 10U);
  EXPECT_EQ(op.applied_collection_op_id, 99U);
}

TEST(SegmentOpCodec, LabelOpsRejectTruncationAndTrailing) {
  {
    auto e = encode_label_bind(kSegId, kGen, 1, 0, 5, 0, 7);
    e.pop_back();
    EXPECT_THROW((void)decode_segment_op(e), std::invalid_argument);
    auto t = encode_label_bind(kSegId, kGen, 1, 0, 5, 0, 7);
    t.push_back(std::byte{0});
    EXPECT_THROW((void)decode_segment_op(t), std::invalid_argument);
  }
  {
    auto e = encode_tx_publish(kSegId, kGen, 1, 2, 3, 4);
    e.pop_back();
    EXPECT_THROW((void)decode_segment_op(e), std::invalid_argument);
    auto t = encode_tx_publish(kSegId, kGen, 1, 2, 3, 4);
    t.push_back(std::byte{0});
    EXPECT_THROW((void)decode_segment_op(t), std::invalid_argument);
  }
}

}  // namespace
}  // namespace alaya::laser
