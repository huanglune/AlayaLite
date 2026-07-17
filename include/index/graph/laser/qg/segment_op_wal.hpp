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
#include <functional>
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

// Record-type compatibility matrix (2A decision 1, codec suggestion 2):
//   * old decoder -> new WAL: fails closed at the first kind=7/8 frame (its range
//     check rejects the unknown kind) — no silent misread.
//   * new decoder -> old WAL: accepts every kind=1..6 verbatim; labels fall back
//     to identity (no kind=7/8 present).
//   * new decoder -> mixed WAL: processes kind=5 (publish) and kind=8 (tx_publish)
//     side by side; kind=5 NEVER promotes any staged label binding.
// There is no safe downgrade. kind=1..6 wire bytes are frozen (golden-bytes test).
enum class SegmentOpKind : std::uint8_t {
  row_patch = 1,          // pid, absolute byte offset, length, bytes (idempotent rewrite)
  tombstone = 2,          // pid (set-only; a resurrect is a new insert, never an un-tombstone)
  consolidate_begin = 3,  // epoch (phase barrier)
  consolidate_end = 4,    // epoch (phase barrier)
  publish = 5,            // visibility watermark (monotone max on replay)
  superblock_flip = 6,    // target slot + 512-byte superblock image (checkpoint commit point)
  label_bind = 7,  // 2A: tx_id, row_op_id, pid, pid_generation, label (staged until tx_publish)
  tx_publish = 8,  // 2A: tx_id, new_pid_watermark, binding_count, applied_collection_op_id
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
  after_wal_reset,
  // 2A label-transaction cuts (appended so the existing six keep their values):
  after_label_bind_append,         // bundle kind=7 binds buffered, before tx_publish append
  before_tx_publish_append,        // snapshot pre-built, immediately before the kind=8 append
  after_tx_publish_fsync,          // kind=8 durable, before the snapshot swap + committed store
  label_slot_written_before_flip,  // inactive label slot durable, before the checkpoint flip frame
  // 2C consolidate-transaction cuts (design section 6.1; appended so every prior
  // value is preserved). C0-C11 map onto these + the whole-page install loop.
  after_consolidate_begin_append,          // kind=3 begin buffered, before its fsync
  after_consolidate_begin_fsync,           // begin durable, before any page mutation
  after_consolidate_spill_flush,           // an overlay page spilled (kind=1) + flushed
  before_consolidate_end_append,           // all pages staged, immediately before kind=4
  after_consolidate_end_fsync,             // kind=4 durable (the commit point), before install
  after_consolidate_install_page,          // one index page installed post-commit
  after_consolidate_install_before_publish,  // all pages installed, before free-list/epoch publish
};

// Test-only observer for the persistence-model (power-loss) crash layer. It is
// notified right after each durable fsync of the index fd and of the op-WAL, so
// a harness can snapshot the "forced" content of each file and later materialize
// the possible power-loss disk states (retain/drop the unforced tail of each
// stream independently). Null in production — zero overhead.
struct SegmentIoObserver {
  std::function<void()> on_index_fsync{};
  std::function<void()> on_wal_fsync{};
  // 2A: fired right after an inactive label-slot file is fsynced durable during a
  // checkpoint, so the power-loss harness can snapshot the forced slot contents.
  std::function<void()> on_label_slot_fsync{};
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
  // 2A label transaction. tx_id mirrors the frame batch_id (validated on replay).
  std::uint64_t tx_id{};                     // label_bind, tx_publish
  std::uint64_t row_op_id{};                 // label_bind: 0..binding_count-1 within the bundle
  std::uint32_t pid_generation{};            // label_bind: must be 0 (non-zero => poison)
  std::uint64_t label{};                     // label_bind: appended-row label (pid stored in `pid`)
  std::uint64_t new_pid_watermark{};         // tx_publish: old_hwm + binding_count
  std::uint64_t binding_count{};             // tx_publish: >= 1
  std::uint64_t applied_collection_op_id{};  // tx_publish: caller monotone op watermark
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
                                                    std::uint64_t epoch) -> std::vector<std::byte> {
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
  segment_op_detail::put_header(out,
                                segment_id,
                                segment_generation,
                                SegmentOpKind::superblock_flip);
  out.push_back(static_cast<std::byte>(target_slot));
  segment_op_detail::put_bytes(out, superblock_image);
  return out;
}

// label_bind body (2A): tx_id, row_op_id, pid (u32), pid_generation (u32), label.
// Staged by tx_id on replay; promoted only when the matching tx_publish is seen.
// The enclosing frame carries batch_id == tx_id (replay poisons on a mismatch).
[[nodiscard]] inline auto encode_label_bind(std::uint64_t segment_id,
                                            std::uint64_t segment_generation,
                                            std::uint64_t tx_id,
                                            std::uint64_t row_op_id,
                                            std::uint32_t pid,
                                            std::uint32_t pid_generation,
                                            std::uint64_t label) -> std::vector<std::byte> {
  std::vector<std::byte> out;
  segment_op_detail::put_header(out, segment_id, segment_generation, SegmentOpKind::label_bind);
  alaya::wal::put_u64(out, tx_id);
  alaya::wal::put_u64(out, row_op_id);
  alaya::wal::put_u32(out, pid);
  alaya::wal::put_u32(out, pid_generation);
  alaya::wal::put_u64(out, label);
  return out;
}

// tx_publish body (2A): tx_id, new_pid_watermark, binding_count, applied op-id.
// The single durable commit point of a physical bundle (fsync); its frame carries
// batch_id == tx_id. binding_count == 0 is illegal (no empty / pure-tombstone tx).
[[nodiscard]] inline auto encode_tx_publish(std::uint64_t segment_id,
                                            std::uint64_t segment_generation,
                                            std::uint64_t tx_id,
                                            std::uint64_t new_pid_watermark,
                                            std::uint64_t binding_count,
                                            std::uint64_t applied_collection_op_id)
    -> std::vector<std::byte> {
  std::vector<std::byte> out;
  segment_op_detail::put_header(out, segment_id, segment_generation, SegmentOpKind::tx_publish);
  alaya::wal::put_u64(out, tx_id);
  alaya::wal::put_u64(out, new_pid_watermark);
  alaya::wal::put_u64(out, binding_count);
  alaya::wal::put_u64(out, applied_collection_op_id);
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
      kind_raw > static_cast<std::uint8_t>(SegmentOpKind::tx_publish)) {
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
    case SegmentOpKind::label_bind: {
      op.tx_id = decoder.u64();
      op.row_op_id = decoder.u64();
      op.pid = decoder.u32();
      op.pid_generation = decoder.u32();
      op.label = decoder.u64();
      break;
    }
    case SegmentOpKind::tx_publish: {
      op.tx_id = decoder.u64();
      op.new_pid_watermark = decoder.u64();
      op.binding_count = decoder.u64();
      op.applied_collection_op_id = decoder.u64();
      break;
    }
  }
  if (!decoder.empty()) {
    throw std::invalid_argument("decode_segment_op: payload has trailing bytes");
  }
  return op;
}

}  // namespace alaya::laser
