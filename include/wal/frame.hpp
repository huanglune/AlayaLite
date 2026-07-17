// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// Physical WAL v1 framing ("WAL7"), shared by every durable log record.
//
// This is the single envelope described by docs/design/unified-wal-vocabulary.md:
// a 36-byte header (`WAL7` magic, format version, record type, flags, lengths,
// op ids, CRC-32 over the whole frame), an `END7` trailer, and prefix-safe,
// truncate-to-verified-boundary scanning. Two op families ride inside it:
//   * collection logical mutations (record types 1-4, index/collection/)
//   * segment-physical ops (record type 5, index/graph/laser/qg/segment_op_wal)
//
// Layering: `wal/` sits at the bottom of the stack. It depends only on the
// standard library plus `platform/` (the filesystem primitives WalFile needs
// for crash-safe fsync/rename). It must never include `core/`, `space/`,
// `index/`, or `storage/`. The frame primitives, `Decoder`, and `scan()` are
// pure `std`; only `WalFile` touches `platform/`.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "platform/fs.hpp"

namespace alaya::wal {

inline constexpr std::uint32_t kFrameMagic = 0x374C4157U;    // "WAL7" in little endian.
inline constexpr std::uint32_t kTrailerMagic = 0x37444E45U;  // "END7" in little endian.
inline constexpr std::uint16_t kFormatVersion = 1;
inline constexpr std::uint32_t kHeaderBytes = 36;
inline constexpr std::uint32_t kTrailerBytes = 4;
inline constexpr std::uint32_t kMaximumPayloadBytes = 64U << 20U;
inline constexpr std::size_t kChecksumOffset = 32;

// --- little-endian integer primitives -------------------------------------

inline void put_u16(std::vector<std::byte> &output, std::uint16_t value) {
  for (unsigned shift = 0; shift < 16; shift += 8) {
    output.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
  }
}

inline void put_u32(std::vector<std::byte> &output, std::uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8) {
    output.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
  }
}

inline void put_u64(std::vector<std::byte> &output, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8) {
    output.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
  }
}

[[nodiscard]] inline auto get_u16(std::span<const std::byte> input, std::size_t offset)
    -> std::uint16_t {
  return static_cast<std::uint16_t>(std::to_integer<unsigned>(input[offset])) |
         static_cast<std::uint16_t>(std::to_integer<unsigned>(input[offset + 1]) << 8U);
}

[[nodiscard]] inline auto get_u32(std::span<const std::byte> input, std::size_t offset)
    -> std::uint32_t {
  std::uint32_t value{};
  for (unsigned index = 0; index < 4; ++index) {
    value |= static_cast<std::uint32_t>(std::to_integer<unsigned>(input[offset + index]))
             << (index * 8U);
  }
  return value;
}

[[nodiscard]] inline auto get_u64(std::span<const std::byte> input, std::size_t offset)
    -> std::uint64_t {
  std::uint64_t value{};
  for (unsigned index = 0; index < 8; ++index) {
    value |= static_cast<std::uint64_t>(std::to_integer<unsigned>(input[offset + index]))
             << (index * 8U);
  }
  return value;
}

[[nodiscard]] inline auto crc32(std::span<const std::byte> bytes) noexcept -> std::uint32_t {
  std::uint32_t crc = 0xffffffffU;
  for (const auto byte : bytes) {
    crc ^= std::to_integer<std::uint8_t>(byte);
    for (unsigned bit = 0; bit < 8; ++bit) {
      const auto mask = static_cast<std::uint32_t>(-(static_cast<std::int32_t>(crc & 1U)));
      crc = (crc >> 1U) ^ (0xedb88320U & mask);
    }
  }
  return ~crc;
}

// Serialize one frame. `type` is an opaque non-zero record type; the framing
// layer never interprets it (the two op families own their own type spaces).
[[nodiscard]] inline auto make_frame(std::uint8_t type,
                                     std::uint8_t flags,
                                     std::uint64_t op_id,
                                     std::uint64_t batch_id,
                                     std::span<const std::byte> payload) -> std::vector<std::byte> {
  if (payload.size() > kMaximumPayloadBytes ||
      payload.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("WAL payload exceeds the format limit");
  }
  const auto frame_bytes =
      static_cast<std::uint32_t>(kHeaderBytes + payload.size() + kTrailerBytes);
  std::vector<std::byte> output;
  output.reserve(frame_bytes);
  put_u32(output, kFrameMagic);
  put_u16(output, kFormatVersion);
  output.push_back(static_cast<std::byte>(type));
  output.push_back(static_cast<std::byte>(flags));
  put_u32(output, frame_bytes);
  put_u32(output, static_cast<std::uint32_t>(payload.size()));
  put_u64(output, op_id);
  put_u64(output, batch_id);
  put_u32(output, 0);
  output.insert(output.end(), payload.begin(), payload.end());
  put_u32(output, kTrailerMagic);
  const auto checksum = crc32(output);
  for (unsigned index = 0; index < 4; ++index) {
    output[kChecksumOffset + index] = static_cast<std::byte>((checksum >> (index * 8U)) & 0xffU);
  }
  return output;
}

// --- structural scan -------------------------------------------------------

struct ScannedFrame {
  std::uint8_t type{};  // opaque, non-zero; the framing layer does not interpret it.
  std::uint8_t flags{};
  std::uint64_t op_id{};
  std::uint64_t batch_id{};
  std::vector<std::byte> payload{};
  std::uint64_t offset{};  // byte offset of this frame within the stream.
  std::uint64_t size{};    // complete frame length.
};

struct ScanResult {
  std::vector<ScannedFrame> frames{};
  std::uint64_t valid_bytes{};
  bool stopped_at_corrupt_or_torn_tail{};
};

// Byte location of one frame within a WAL file: start offset + complete length.
// Returned by WalFile::append and consumed by WalFile::read_frame, so a caller
// (the W1 maintenance overlay spill) can re-read exactly one frame later without
// loading the whole file (codex B-2C-04). Wire is unchanged: this is a pure
// bookkeeping value, never serialized.
struct FrameLocation {
  std::uint64_t offset{};
  std::uint64_t size{};
};

// Prefix-safe structural scan. A frame is accepted iff its magic, version,
// lengths, trailer, and CRC are consistent and its record type is non-zero.
// Semantic validation of the type (which family, which op) belongs to the
// caller: the framing layer only guarantees structural integrity and order.
// The first structurally-broken frame ends the scan (torn/corrupt tail); the
// scanner never resynchronizes past damage.
[[nodiscard]] inline auto scan(std::span<const std::byte> bytes) -> ScanResult {
  ScanResult result;
  std::size_t offset{};
  while (offset < bytes.size()) {
    const auto remaining = bytes.size() - offset;
    if (remaining < kHeaderBytes) {
      result.stopped_at_corrupt_or_torn_tail = true;
      break;
    }
    const auto input = bytes.subspan(offset);
    const auto magic = get_u32(input, 0);
    const auto version = get_u16(input, 4);
    const auto type = std::to_integer<std::uint8_t>(input[6]);
    const auto flags = std::to_integer<std::uint8_t>(input[7]);
    const auto frame_bytes = get_u32(input, 8);
    const auto payload_bytes = get_u32(input, 12);
    if (magic != kFrameMagic || version != kFormatVersion || type == 0 ||
        payload_bytes > kMaximumPayloadBytes ||
        frame_bytes != kHeaderBytes + payload_bytes + kTrailerBytes || frame_bytes > remaining) {
      result.stopped_at_corrupt_or_torn_tail = true;
      break;
    }
    const auto frame = input.first(frame_bytes);
    if (get_u32(frame, frame_bytes - 4) != kTrailerMagic) {
      result.stopped_at_corrupt_or_torn_tail = true;
      break;
    }
    auto checksum_input = std::vector<std::byte>(frame.begin(), frame.end());
    const auto expected = get_u32(frame, kChecksumOffset);
    std::fill_n(checksum_input.begin() + static_cast<std::ptrdiff_t>(kChecksumOffset),
                4,
                std::byte{});
    if (crc32(checksum_input) != expected) {
      result.stopped_at_corrupt_or_torn_tail = true;
      break;
    }
    ScannedFrame decoded;
    decoded.type = type;
    decoded.flags = flags;
    decoded.op_id = get_u64(frame, 16);
    decoded.batch_id = get_u64(frame, 24);
    decoded.payload.assign(frame.begin() + kHeaderBytes, frame.end() - kTrailerBytes);
    decoded.offset = offset;
    decoded.size = frame_bytes;
    result.frames.push_back(std::move(decoded));
    offset += frame_bytes;
    result.valid_bytes = offset;
  }
  return result;
}

// --- payload byte decoder --------------------------------------------------

// Generic, bounds-checked little-endian primitive reader over a frame payload.
// Hoisted out of the collection mutation codec (unified-wal-vocabulary.md
// sections 5 + clause J): both op families decode with this one reader. It
// exposes only scalar reads and raw take(n); length-prefixed byte/string fields,
// their size limits, and composite types (RowAddress, LogicalId, ...) stay in
// the owning family's codec, built on top of these primitives.
class Decoder {
 public:
  explicit Decoder(std::span<const std::byte> input) : input_(input) {}

  [[nodiscard]] auto u8() -> std::uint8_t {
    require(1);
    return std::to_integer<std::uint8_t>(input_[offset_++]);
  }

  [[nodiscard]] auto u16() -> std::uint16_t {
    require(2);
    const auto value = get_u16(input_, offset_);
    offset_ += 2;
    return value;
  }

  [[nodiscard]] auto u32() -> std::uint32_t {
    require(4);
    const auto value = get_u32(input_, offset_);
    offset_ += 4;
    return value;
  }

  [[nodiscard]] auto u64() -> std::uint64_t {
    require(8);
    const auto value = get_u64(input_, offset_);
    offset_ += 8;
    return value;
  }

  // Read exactly `count` raw bytes (bounds-checked). Any length framing on top
  // of this is the owning family's codec — the framework reader stays a pure
  // primitive set (unified-wal-vocabulary.md clause J; no collection string
  // limit, no RowAddress, here).
  [[nodiscard]] auto take(std::size_t count) -> std::span<const std::byte> {
    require(count);
    const auto value = input_.subspan(offset_, count);
    offset_ += count;
    return value;
  }

  [[nodiscard]] auto empty() const noexcept -> bool { return offset_ == input_.size(); }
  [[nodiscard]] auto remaining() const noexcept -> std::size_t { return input_.size() - offset_; }

 private:
  void require(std::size_t bytes) const {
    if (bytes > input_.size() - offset_) {
      throw std::invalid_argument("WAL payload is truncated");
    }
  }

  std::span<const std::byte> input_{};
  std::size_t offset_{};
};

// --- append-only frame file ------------------------------------------------

// Minimal append-only WAL file over the physical framing above. It generalizes
// the CollectionLogicalWal open/append/scan/reset logic so the segment op-WAL
// (index/graph/laser/qg/segment_op_wal.hpp) reuses it verbatim rather than
// growing a second file format. All errors are reported by throwing; callers
// that want a status object wrap it (as CollectionLogicalWal does).
class WalFile {
 public:
  enum class Sync : std::uint8_t {
    buffered = 0,  // stays in this process's userspace buffer (lost on process crash)
    flush = 1,     // pushed to the OS (survives a process crash, not power loss)
    fsync = 2,     // durable
  };

  // Opens `path`, creating it (and parent directories) if missing, then scans
  // for recovery and truncates any torn/corrupt tail to the last verified
  // frame boundary before the first append.
  //
  // stream_recovery (B-2C-04): when true, the constructor uses a STREAMING structural
  // scan (one frame in memory at a time, O(max frame) not O(file)) to find the last
  // verified boundary + torn-tail flag WITHOUT retaining any frame payloads;
  // recovery_scan() then reports valid_bytes + the torn-tail flag with an EMPTY frames
  // vector, and the caller replays via visit_frames(). Default false keeps the eager
  // full scan (small logs / unit tests / other op families) byte-for-byte unchanged.
  explicit WalFile(std::filesystem::path path, bool stream_recovery = false)
      : path_(std::move(path)) {
    std::filesystem::create_directories(path_.parent_path());
    if (!std::filesystem::exists(path_)) {
      std::ofstream create(path_, std::ios::binary | std::ios::trunc);
      if (!create) {
        throw std::runtime_error("WalFile: cannot create " + path_.string());
      }
      create.close();
      platform::sync_directory_or_throw(path_.parent_path());
    }
    recovery_scan_ = stream_recovery ? scan_structure_streaming(path_) : scan_path(path_);
    if (recovery_scan_.stopped_at_corrupt_or_torn_tail) {
      std::filesystem::resize_file(path_, recovery_scan_.valid_bytes);
      platform::sync_file_or_throw(path_);
    }
    // The append cursor starts at the last verified frame boundary (== the file
    // size after any torn-tail truncation), so append() can report accurate frame
    // locations (B-2C-04).
    append_offset_ = recovery_scan_.valid_bytes;
    open_stream();
  }

  WalFile(const WalFile &) = delete;
  auto operator=(const WalFile &) -> WalFile & = delete;
  WalFile(WalFile &&) = delete;
  auto operator=(WalFile &&) -> WalFile & = delete;

  ~WalFile() {
    if (stream_.is_open()) {
      stream_.close();
    }
  }

  [[nodiscard]] auto recovery_scan() const -> const ScanResult & { return recovery_scan_; }
  [[nodiscard]] auto path() const -> const std::filesystem::path & { return path_; }

  // Returns the FrameLocation (offset + size) of the appended frame (B-2C-04) so a
  // caller can re-read exactly this frame later via read_frame(). Existing callers
  // that ignore the return value are unaffected.
  FrameLocation append(std::uint8_t type,
                       std::uint8_t flags,
                       std::uint64_t op_id,
                       std::uint64_t batch_id,
                       std::span<const std::byte> payload,
                       Sync sync) {
    const auto frame = make_frame(type, flags, op_id, batch_id, payload);
    const FrameLocation location{append_offset_, static_cast<std::uint64_t>(frame.size())};
    stream_.write(reinterpret_cast<const char *>(frame.data()),
                  static_cast<std::streamsize>(frame.size()));
    if (!stream_) {
      throw std::runtime_error("WalFile: cannot append frame to " + path_.string());
    }
    if (sync != Sync::buffered) {
      stream_.flush();
      if (!stream_) {
        throw std::runtime_error("WalFile: cannot flush " + path_.string());
      }
    }
    if (sync == Sync::fsync) {
      platform::sync_file_or_throw(path_);
    }
    append_offset_ += frame.size();
    return location;
  }

  // Force the currently-buffered prefix durable without appending a frame. Used
  // as a group-commit / force-before-writeback barrier by callers that must make
  // earlier buffered frames durable before releasing dependent state to disk.
  void fsync() {
    stream_.flush();
    if (!stream_) {
      throw std::runtime_error("WalFile: cannot flush " + path_.string());
    }
    platform::sync_file_or_throw(path_);
  }

  // Atomically replace the file with exactly one durable frame. Mirrors the
  // collection checkpoint cut: a temp file is written, fsynced, atomically
  // renamed over the target, and the directory is fsynced, so every crash
  // point leaves either the old log or the one-frame log intact.
  void reset_to_single_frame(std::uint8_t type,
                             std::uint8_t flags,
                             std::uint64_t op_id,
                             std::uint64_t batch_id,
                             std::span<const std::byte> payload) {
    if (stream_.is_open()) {
      stream_.close();
    }
    const auto temporary = path_.parent_path() / (path_.filename().string() + ".reset.tmp");
    const auto frame = make_frame(type, flags, op_id, batch_id, payload);
    {
      std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
      output.write(reinterpret_cast<const char *>(frame.data()),
                   static_cast<std::streamsize>(frame.size()));
      output.flush();
      if (!output) {
        throw std::runtime_error("WalFile: cannot write reset marker for " + path_.string());
      }
    }
    platform::sync_file_or_throw(temporary);
    platform::atomic_replace(temporary, path_);
    platform::sync_directory_or_throw(path_.parent_path());
    recovery_scan_ = scan_path(path_);
    append_offset_ = recovery_scan_.valid_bytes;
    open_stream();
  }

  // Semantic truncation to a frame boundary (B-2C-04): drop everything at/after
  // `offset` (e.g. an unmatched maintenance BEGIN discovered during recovery),
  // fsync, and reopen the append stream so the next append starts clean at the new
  // boundary. `offset` must be a verified frame boundary chosen by the caller.
  void truncate_to(std::uint64_t offset) {
    if (stream_.is_open()) {
      stream_.close();
    }
    std::filesystem::resize_file(path_, offset);
    platform::sync_file_or_throw(path_);
    platform::sync_directory_or_throw(path_.parent_path());
    recovery_scan_ = scan_path(path_);
    append_offset_ = offset;
    open_stream();
  }

  // Streaming structural scan (B-2C-04): invoke visitor(const ScannedFrame&) for
  // each verified frame in file order, holding at most one frame in memory
  // (O(max frame), not O(file)). Stops at the first torn/corrupt frame -- never
  // resynchronizes past damage -- or when the visitor returns false. This lets QG
  // recovery replay a large op-WAL without loading it whole. visitor returns bool.
  template <typename Visitor>
  static void visit_frames(const std::filesystem::path &path, Visitor &&visitor) {
    if (!std::filesystem::exists(path)) {
      return;
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("WalFile: cannot read " + path.string());
    }
    std::uint64_t offset = 0;
    std::vector<std::byte> frame;
    for (;;) {
      std::array<std::byte, kHeaderBytes> header{};
      input.read(reinterpret_cast<char *>(header.data()), kHeaderBytes);
      if (input.gcount() != static_cast<std::streamsize>(kHeaderBytes)) {
        break;
      }
      const auto magic = get_u32(header, 0);
      const auto version = get_u16(header, 4);
      const auto type = std::to_integer<std::uint8_t>(header[6]);
      const auto flags = std::to_integer<std::uint8_t>(header[7]);
      const auto frame_bytes = get_u32(header, 8);
      const auto payload_bytes = get_u32(header, 12);
      if (magic != kFrameMagic || version != kFormatVersion || type == 0 ||
          payload_bytes > kMaximumPayloadBytes ||
          frame_bytes != kHeaderBytes + payload_bytes + kTrailerBytes) {
        break;
      }
      frame.resize(frame_bytes);
      std::copy(header.begin(), header.end(), frame.begin());
      input.read(reinterpret_cast<char *>(frame.data()) + kHeaderBytes,
                 static_cast<std::streamsize>(frame_bytes - kHeaderBytes));
      if (input.gcount() != static_cast<std::streamsize>(frame_bytes - kHeaderBytes)) {
        break;
      }
      if (get_u32(frame, frame_bytes - 4) != kTrailerMagic) {
        break;
      }
      const auto expected = get_u32(frame, kChecksumOffset);
      std::vector<std::byte> checksum_input(frame.begin(), frame.end());
      std::fill_n(checksum_input.begin() + static_cast<std::ptrdiff_t>(kChecksumOffset),
                  4,
                  std::byte{});
      if (crc32(checksum_input) != expected) {
        break;
      }
      ScannedFrame decoded;
      decoded.type = type;
      decoded.flags = flags;
      decoded.op_id = get_u64(frame, 16);
      decoded.batch_id = get_u64(frame, 24);
      decoded.payload.assign(frame.begin() + kHeaderBytes, frame.end() - kTrailerBytes);
      decoded.offset = offset;
      decoded.size = frame_bytes;
      if (!visitor(static_cast<const ScannedFrame &>(decoded))) {
        return;
      }
      offset += frame_bytes;
    }
  }

  // Read exactly the frame at `location` and re-validate header/length/trailer/CRC
  // (B-2C-04). Throws on any mismatch, so a stale/torn location fails closed.
  [[nodiscard]] static auto read_frame(const std::filesystem::path &path, FrameLocation location)
      -> ScannedFrame {
    if (location.size < kHeaderBytes + kTrailerBytes ||
        location.size > static_cast<std::uint64_t>(kHeaderBytes) + kMaximumPayloadBytes +
                            kTrailerBytes) {
      throw std::invalid_argument("WalFile::read_frame: implausible frame size");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("WalFile: cannot read " + path.string());
    }
    input.seekg(static_cast<std::streamoff>(location.offset));
    std::vector<std::byte> frame(static_cast<std::size_t>(location.size));
    input.read(reinterpret_cast<char *>(frame.data()),
               static_cast<std::streamsize>(frame.size()));
    if (static_cast<std::uint64_t>(input.gcount()) != location.size) {
      throw std::runtime_error("WalFile::read_frame: short read");
    }
    const auto magic = get_u32(frame, 0);
    const auto version = get_u16(frame, 4);
    const auto type = std::to_integer<std::uint8_t>(frame[6]);
    const auto flags = std::to_integer<std::uint8_t>(frame[7]);
    const auto frame_bytes = get_u32(frame, 8);
    const auto payload_bytes = get_u32(frame, 12);
    if (magic != kFrameMagic || version != kFormatVersion || type == 0 ||
        frame_bytes != location.size ||
        frame_bytes != kHeaderBytes + payload_bytes + kTrailerBytes) {
      throw std::runtime_error("WalFile::read_frame: header/length mismatch");
    }
    if (get_u32(frame, frame_bytes - 4) != kTrailerMagic) {
      throw std::runtime_error("WalFile::read_frame: bad trailer");
    }
    const auto expected = get_u32(frame, kChecksumOffset);
    std::vector<std::byte> checksum_input(frame.begin(), frame.end());
    std::fill_n(checksum_input.begin() + static_cast<std::ptrdiff_t>(kChecksumOffset),
                4,
                std::byte{});
    if (crc32(checksum_input) != expected) {
      throw std::runtime_error("WalFile::read_frame: CRC mismatch");
    }
    ScannedFrame decoded;
    decoded.type = type;
    decoded.flags = flags;
    decoded.op_id = get_u64(frame, 16);
    decoded.batch_id = get_u64(frame, 24);
    decoded.payload.assign(frame.begin() + kHeaderBytes, frame.end() - kTrailerBytes);
    decoded.offset = location.offset;
    decoded.size = frame_bytes;
    return decoded;
  }

  [[nodiscard]] static auto scan_path(const std::filesystem::path &path) -> ScanResult {
    if (!std::filesystem::exists(path)) {
      return {};
    }
    const auto file_size = std::filesystem::file_size(path);
    std::vector<std::byte> bytes(static_cast<std::size_t>(file_size));
    if (!bytes.empty()) {
      std::ifstream input(path, std::ios::binary);
      input.read(reinterpret_cast<char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
      if (!input) {
        throw std::runtime_error("WalFile: cannot read " + path.string());
      }
    }
    return scan(bytes);
  }

  // Streaming structural scan (B-2C-04): compute valid_bytes + the torn/corrupt-tail
  // flag WITHOUT retaining payloads (holds at most one frame in memory). Applies exactly
  // the same acceptance rules as scan()/visit_frames -- a frame is valid iff its magic,
  // version, non-zero type, lengths, trailer, and CRC are all consistent; the first
  // structurally-broken (or partially-present) trailing frame ends the scan and marks a
  // torn tail (never resynchronizes past damage). result.frames stays empty by design.
  [[nodiscard]] static auto scan_structure_streaming(const std::filesystem::path &path)
      -> ScanResult {
    ScanResult result;
    if (!std::filesystem::exists(path)) {
      return result;
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("WalFile: cannot read " + path.string());
    }
    std::vector<std::byte> frame;
    for (;;) {
      std::array<std::byte, kHeaderBytes> header{};
      input.read(reinterpret_cast<char *>(header.data()), kHeaderBytes);
      const auto got = input.gcount();
      if (got == 0) {
        break;  // clean EOF exactly on a frame boundary
      }
      if (got != static_cast<std::streamsize>(kHeaderBytes)) {
        result.stopped_at_corrupt_or_torn_tail = true;  // partial trailing header
        break;
      }
      const auto magic = get_u32(header, 0);
      const auto version = get_u16(header, 4);
      const auto type = std::to_integer<std::uint8_t>(header[6]);
      const auto frame_bytes = get_u32(header, 8);
      const auto payload_bytes = get_u32(header, 12);
      if (magic != kFrameMagic || version != kFormatVersion || type == 0 ||
          payload_bytes > kMaximumPayloadBytes ||
          frame_bytes != kHeaderBytes + payload_bytes + kTrailerBytes) {
        result.stopped_at_corrupt_or_torn_tail = true;
        break;
      }
      frame.resize(frame_bytes);
      std::copy(header.begin(), header.end(), frame.begin());
      input.read(reinterpret_cast<char *>(frame.data()) + kHeaderBytes,
                 static_cast<std::streamsize>(frame_bytes - kHeaderBytes));
      if (input.gcount() != static_cast<std::streamsize>(frame_bytes - kHeaderBytes)) {
        result.stopped_at_corrupt_or_torn_tail = true;  // partial trailing body
        break;
      }
      if (get_u32(frame, frame_bytes - 4) != kTrailerMagic) {
        result.stopped_at_corrupt_or_torn_tail = true;
        break;
      }
      const auto expected = get_u32(frame, kChecksumOffset);
      std::vector<std::byte> checksum_input(frame.begin(), frame.end());
      std::fill_n(checksum_input.begin() + static_cast<std::ptrdiff_t>(kChecksumOffset),
                  4,
                  std::byte{});
      if (crc32(checksum_input) != expected) {
        result.stopped_at_corrupt_or_torn_tail = true;
        break;
      }
      result.valid_bytes += frame_bytes;
    }
    return result;
  }

 private:
  void open_stream() {
    stream_.rdbuf()->pubsetbuf(write_buffer_.data(),
                               static_cast<std::streamsize>(write_buffer_.size()));
    stream_.open(path_, std::ios::binary | std::ios::app);
    if (!stream_) {
      throw std::runtime_error("WalFile: cannot open " + path_.string() + " for append");
    }
  }

  std::filesystem::path path_{};
  ScanResult recovery_scan_{};
  std::uint64_t append_offset_{};  // byte offset of the next append (B-2C-04 locations)
  std::array<char, 1U << 20U> write_buffer_{};
  std::ofstream stream_{};
};

}  // namespace alaya::wal
