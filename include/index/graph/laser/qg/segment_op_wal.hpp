// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// SEGMENT_OP op-WAL payload codec (unified-wal-vocabulary.md sections 2-3).
//
// Segment-physical operations on a sealed LASER quantized graph (QGUpdater)
// ride inside the shared Physical WAL v1 envelope (wal/frame.hpp) as record
// type 5. This header owns only the *payload* format for that type; the frame
// header, CRC, scan, and file all belong to alaya::wal. Segment identity lives
// in the payload, never in the frame header (the header keeps its op-id/batch
// meanings). There is no second framing or primitive set.
//
// Layering: this file sits under index/graph/ and must stay a graph-kernel
// leaf — it includes only alaya::wal (bottom layer) plus std, never
// index/collection or index/disk.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "wal/frame.hpp"

namespace alaya::laser {

// SEGMENT_OP record type in the shared WAL7 envelope. Cross-reference:
// docs/design/unified-wal-vocabulary.md section 2, and the "5 is reserved for
// SEGMENT_OP" note on LogicalWalRecordType in
// include/index/collection/logical_wal.hpp.
inline constexpr std::uint8_t kSegmentOpRecordType = 5;

// Payload version, bumped independently of the frame format (exactly like the
// mutation payload codec's leading u16).
inline constexpr std::uint16_t kSegmentOpPayloadVersion = 1;

// A superblock_flip carries a full 512-byte QGSuperblockV2 image (its own
// checksum field is a superset of the contract's "slot + CRC", which lets
// replay rewrite the superblock directly rather than merely validate it).
inline constexpr std::size_t kSegmentSuperblockImageBytes = 512;

enum class SegmentOpKind : std::uint8_t {
  row_patch = 1,          // pid, absolute byte offset, length, bytes (idempotent rewrite)
  tombstone = 2,          // pid (set-only; a resurrect is a new insert, never an un-tombstone)
  consolidate_begin = 3,  // epoch (phase barrier)
  consolidate_end = 4,    // epoch (phase barrier)
  publish = 5,            // visibility watermark (monotone max on replay)
  superblock_flip = 6,    // target slot + 512-byte superblock image (checkpoint commit point)
};

// Crash-injection points for the G1 crash matrix, ordered along one op's
// lifecycle. Wired through UpdateParams::failpoint_hook; empty in production.
enum class SegmentOpFailPoint : std::uint8_t {
  none = 0,
  after_wal_append_before_apply,
  after_apply_before_publish_fsync,
  after_publish_fsync,
  after_flip_append_before_superblock_write,
  after_superblock_write_before_wal_reset,
};

// One decoded SEGMENT_OP. Only the body fields matching `kind` are meaningful.
struct SegmentOp {
  std::uint16_t payload_version{};
  std::uint64_t segment_id{};
  std::uint64_t segment_generation{};
  SegmentOpKind kind{};
  std::uint64_t pid{};             // row_patch (informational), tombstone
  std::uint64_t offset{};          // row_patch: absolute byte offset into the index file
  std::vector<std::byte> bytes{};  // row_patch bytes, or the 512-byte superblock image
  std::uint64_t epoch{};           // consolidate_begin / consolidate_end
  std::uint64_t watermark{};       // publish
  std::uint8_t target_slot{};      // superblock_flip
};

namespace segment_op_detail {

inline void put_header(std::vector<std::byte> &out,
                       std::uint64_t segment_id,
                       std::uint64_t segment_generation,
                       SegmentOpKind kind) {
  alaya::wal::put_u16(out, kSegmentOpPayloadVersion);
  alaya::wal::put_u64(out, segment_id);
  alaya::wal::put_u64(out, segment_generation);
  out.push_back(static_cast<std::byte>(kind));
}

inline void put_bytes(std::vector<std::byte> &out, std::span<const std::byte> value) {
  alaya::wal::put_u32(out, static_cast<std::uint32_t>(value.size()));
  out.insert(out.end(), value.begin(), value.end());
}

}  // namespace segment_op_detail

// row_patch body: pid, absolute byte offset, u32 length + bytes. `pid` is
// informational only — replay rewrites purely by absolute offset. A
// page-granularity patch that covers several rows records the page's first pid.
[[nodiscard]] inline auto encode_row_patch(std::uint64_t segment_id,
                                           std::uint64_t segment_generation,
                                           std::uint64_t pid,
                                           std::uint64_t absolute_offset,
                                           std::span<const std::byte> bytes)
    -> std::vector<std::byte> {
  std::vector<std::byte> out;
  out.reserve(29 + 12 + bytes.size());
  segment_op_detail::put_header(out, segment_id, segment_generation, SegmentOpKind::row_patch);
  alaya::wal::put_u64(out, pid);
  alaya::wal::put_u64(out, absolute_offset);
  segment_op_detail::put_bytes(out, bytes);
  return out;
}

[[nodiscard]] inline auto encode_tombstone(std::uint64_t segment_id,
                                           std::uint64_t segment_generation,
                                           std::uint64_t pid) -> std::vector<std::byte> {
  std::vector<std::byte> out;
  segment_op_detail::put_header(out, segment_id, segment_generation, SegmentOpKind::tombstone);
  alaya::wal::put_u64(out, pid);
  return out;
}

[[nodiscard]] inline auto encode_consolidate_marker(std::uint64_t segment_id,
                                                    std::uint64_t segment_generation,
                                                    SegmentOpKind kind,
                                                    std::uint64_t epoch)
    -> std::vector<std::byte> {
  if (kind != SegmentOpKind::consolidate_begin && kind != SegmentOpKind::consolidate_end) {
    throw std::invalid_argument("encode_consolidate_marker: kind must be a consolidate barrier");
  }
  std::vector<std::byte> out;
  segment_op_detail::put_header(out, segment_id, segment_generation, kind);
  alaya::wal::put_u64(out, epoch);
  return out;
}

[[nodiscard]] inline auto encode_publish(std::uint64_t segment_id,
                                         std::uint64_t segment_generation,
                                         std::uint64_t watermark) -> std::vector<std::byte> {
  std::vector<std::byte> out;
  segment_op_detail::put_header(out, segment_id, segment_generation, SegmentOpKind::publish);
  alaya::wal::put_u64(out, watermark);
  return out;
}

[[nodiscard]] inline auto encode_superblock_flip(std::uint64_t segment_id,
                                                 std::uint64_t segment_generation,
                                                 std::uint8_t target_slot,
                                                 std::span<const std::byte> superblock_image)
    -> std::vector<std::byte> {
  if (superblock_image.size() != kSegmentSuperblockImageBytes) {
    throw std::invalid_argument("encode_superblock_flip: image must be 512 bytes");
  }
  std::vector<std::byte> out;
  out.reserve(29 + 1 + 4 + kSegmentSuperblockImageBytes);
  segment_op_detail::put_header(
      out, segment_id, segment_generation, SegmentOpKind::superblock_flip);
  out.push_back(static_cast<std::byte>(target_slot));
  segment_op_detail::put_bytes(out, superblock_image);
  return out;
}

// Decode one SEGMENT_OP payload. Throws std::invalid_argument on an unknown
// payload version / kind or a truncated / trailing-byte payload.
[[nodiscard]] inline auto decode_segment_op(std::span<const std::byte> payload) -> SegmentOp {
  alaya::wal::Decoder decoder(payload);
  SegmentOp op;
  op.payload_version = decoder.u16();
  if (op.payload_version != kSegmentOpPayloadVersion) {
    throw std::invalid_argument("decode_segment_op: unsupported payload version");
  }
  op.segment_id = decoder.u64();
  op.segment_generation = decoder.u64();
  const auto kind_raw = decoder.u8();
  if (kind_raw < static_cast<std::uint8_t>(SegmentOpKind::row_patch) ||
      kind_raw > static_cast<std::uint8_t>(SegmentOpKind::superblock_flip)) {
    throw std::invalid_argument("decode_segment_op: unknown op kind");
  }
  op.kind = static_cast<SegmentOpKind>(kind_raw);
  switch (op.kind) {
    case SegmentOpKind::row_patch: {
      op.pid = decoder.u64();
      op.offset = decoder.u64();
      const auto length = decoder.u32();
      const auto bytes = decoder.take(length);
      op.bytes.assign(bytes.begin(), bytes.end());
      break;
    }
    case SegmentOpKind::tombstone:
      op.pid = decoder.u64();
      break;
    case SegmentOpKind::consolidate_begin:
    case SegmentOpKind::consolidate_end:
      op.epoch = decoder.u64();
      break;
    case SegmentOpKind::publish:
      op.watermark = decoder.u64();
      break;
    case SegmentOpKind::superblock_flip: {
      op.target_slot = decoder.u8();
      const auto image_length = decoder.u32();
      if (image_length != kSegmentSuperblockImageBytes) {
        throw std::invalid_argument("decode_segment_op: superblock image is not 512 bytes");
      }
      const auto image = decoder.take(image_length);
      op.bytes.assign(image.begin(), image.end());
      break;
    }
  }
  if (!decoder.empty()) {
    throw std::invalid_argument("decode_segment_op: payload has trailing bytes");
  }
  return op;
}

}  // namespace alaya::laser
