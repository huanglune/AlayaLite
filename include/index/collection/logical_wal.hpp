// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/status.hpp"
#include "platform/fs.hpp"

namespace alaya::internal::collection {

inline constexpr std::string_view kCollectionWalNamespace{"collection_wal_v1"};
inline constexpr std::string_view kCollectionWalFilename{"logical.wal"};

enum class LogicalWalRecordType : std::uint8_t {
  prepare = 1,
  commit = 2,
  publish_marker = 3,
  checkpoint = 4,
};

enum class LogicalWalSync : std::uint8_t {
  buffered = 0,
  flush = 1,
  fsync = 2,
};

struct LogicalWalFrame {
  LogicalWalRecordType type{LogicalWalRecordType::prepare};
  std::uint8_t flags{};
  std::uint64_t op_id{};
  std::uint64_t batch_id{};
  std::vector<std::byte> payload{};
  std::uint64_t offset{};
  std::uint64_t size{};
};

struct LogicalWalScan {
  std::vector<LogicalWalFrame> frames{};
  std::uint64_t valid_bytes{};
  bool stopped_at_corrupt_or_torn_tail{};
};

namespace logical_wal_detail {

inline constexpr std::uint32_t kFrameMagic = 0x374C4157U;    // "WAL7" in little endian.
inline constexpr std::uint32_t kTrailerMagic = 0x37444E45U;  // "END7" in little endian.
inline constexpr std::uint16_t kFormatVersion = 1;
inline constexpr std::uint32_t kHeaderBytes = 36;
inline constexpr std::uint32_t kTrailerBytes = 4;
inline constexpr std::uint32_t kMaximumPayloadBytes = 64U << 20U;
inline constexpr std::size_t kChecksumOffset = 32;

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

[[nodiscard]] inline auto parse_record_type(std::uint8_t raw)
    -> std::optional<LogicalWalRecordType> {
  switch (raw) {
    case static_cast<std::uint8_t>(LogicalWalRecordType::prepare):
      return LogicalWalRecordType::prepare;
    case static_cast<std::uint8_t>(LogicalWalRecordType::commit):
      return LogicalWalRecordType::commit;
    case static_cast<std::uint8_t>(LogicalWalRecordType::publish_marker):
      return LogicalWalRecordType::publish_marker;
    case static_cast<std::uint8_t>(LogicalWalRecordType::checkpoint):
      return LogicalWalRecordType::checkpoint;
    default:
      return std::nullopt;
  }
}

[[nodiscard]] inline auto make_frame(LogicalWalRecordType type,
                                     std::uint8_t flags,
                                     std::uint64_t op_id,
                                     std::uint64_t batch_id,
                                     std::span<const std::byte> payload) -> std::vector<std::byte> {
  if (payload.size() > kMaximumPayloadBytes ||
      payload.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("collection WAL payload exceeds the format limit");
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

[[nodiscard]] inline auto io_error(core::OperationStage stage, std::string message)
    -> core::Status {
  return core::Status::error(core::StatusCode::io_error,
                             stage,
                             core::StatusDetail::none,
                             std::move(message));
}

}  // namespace logical_wal_detail

class CollectionLogicalWal {
 public:
  CollectionLogicalWal(const CollectionLogicalWal &) = delete;
  auto operator=(const CollectionLogicalWal &) -> CollectionLogicalWal & = delete;
  CollectionLogicalWal(CollectionLogicalWal &&) = delete;
  auto operator=(CollectionLogicalWal &&) -> CollectionLogicalWal & = delete;

  ~CollectionLogicalWal() {
    std::lock_guard lock(mutex_);
    if (stream_.is_open()) {
      stream_.close();
    }
  }

  [[nodiscard]] static auto open(const std::filesystem::path &root,
                                 std::string_view namespace_name = kCollectionWalNamespace)
      -> core::Result<std::unique_ptr<CollectionLogicalWal>> {
    if (root.empty() || namespace_name != kCollectionWalNamespace) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 "collection WAL requires its fixed new namespace");
    }
    try {
      auto wal = std::unique_ptr<CollectionLogicalWal>(new CollectionLogicalWal(
          root / ".alaya_internal" / std::string(kCollectionWalNamespace)));
      std::filesystem::create_directories(wal->directory_);
      if (!std::filesystem::exists(wal->path_)) {
        std::ofstream create(wal->path_, std::ios::binary | std::ios::trunc);
        if (!create) {
          return logical_wal_detail::io_error(core::OperationStage::open,
                                              "cannot create collection logical WAL");
        }
        create.close();
        platform::sync_directory_or_throw(wal->directory_);
      }
      auto scanned = scan_file(wal->path_);
      if (!scanned.ok()) {
        return scanned.status();
      }
      wal->recovery_scan_ = std::move(scanned).value();
      if (wal->recovery_scan_.stopped_at_corrupt_or_torn_tail) {
        std::filesystem::resize_file(wal->path_, wal->recovery_scan_.valid_bytes);
        platform::sync_file_or_throw(wal->path_);
      }
      wal->open_stream();
      return wal;
    } catch (const std::exception &error) {
      return logical_wal_detail::io_error(core::OperationStage::open, error.what());
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

  [[nodiscard]] auto append(LogicalWalRecordType type,
                            std::uint8_t flags,
                            std::uint64_t op_id,
                            std::uint64_t batch_id,
                            std::span<const std::byte> payload,
                            LogicalWalSync sync) -> core::Status {
    try {
      const auto frame = logical_wal_detail::make_frame(type, flags, op_id, batch_id, payload);
      std::lock_guard lock(mutex_);
      stream_.write(reinterpret_cast<const char *>(frame.data()),
                    static_cast<std::streamsize>(frame.size()));
      if (!stream_) {
        return logical_wal_detail::io_error(core::OperationStage::mutation_prepare,
                                            "cannot append collection logical WAL frame");
      }
      if (sync != LogicalWalSync::buffered) {
        stream_.flush();
        if (!stream_) {
          return logical_wal_detail::io_error(core::OperationStage::mutation_prepare,
                                              "cannot flush collection logical WAL frame");
        }
      }
      if (sync == LogicalWalSync::fsync) {
        platform::sync_file_or_throw(path_);
      }
      return core::Status::success();
    } catch (const std::exception &error) {
      return logical_wal_detail::io_error(core::OperationStage::mutation_prepare, error.what());
    } catch (...) {
      return core::status_from_exception(core::OperationStage::mutation_prepare);
    }
  }

  // A full checkpoint has already made every mutation through `wal_cut`
  // durable. Replacing the WAL with one fsynced CHECKPOINT frame is therefore
  // the physical cut; recovery starts after the checkpoint watermark.
  [[nodiscard]] auto reset_to_checkpoint(std::uint64_t wal_cut) -> core::Status {
    try {
      std::lock_guard lock(mutex_);
      if (stream_.is_open()) {
        stream_.close();  // Also promotes any searchable-only buffered prefix into the checkpoint.
      }
      const auto temporary = directory_ / "logical.wal.checkpoint.tmp";
      const auto frame =
          logical_wal_detail::make_frame(LogicalWalRecordType::checkpoint, 0, wal_cut, wal_cut, {});
      {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char *>(frame.data()),
                     static_cast<std::streamsize>(frame.size()));
        output.flush();
        if (!output) {
          return logical_wal_detail::io_error(core::OperationStage::checkpoint,
                                              "cannot write collection WAL checkpoint cut");
        }
      }
      platform::sync_file_or_throw(temporary);
      platform::atomic_replace(temporary, path_);
      platform::sync_directory_or_throw(directory_);
      recovery_scan_.frames.clear();
      recovery_scan_.frames.push_back(LogicalWalFrame{LogicalWalRecordType::checkpoint,
                                                      0,
                                                      wal_cut,
                                                      wal_cut,
                                                      {},
                                                      0,
                                                      frame.size()});
      recovery_scan_.valid_bytes = frame.size();
      recovery_scan_.stopped_at_corrupt_or_torn_tail = false;
      open_stream();
      return core::Status::success();
    } catch (const std::exception &error) {
      return logical_wal_detail::io_error(core::OperationStage::checkpoint, error.what());
    } catch (...) {
      return core::status_from_exception(core::OperationStage::checkpoint);
    }
  }

  [[nodiscard]] auto recovery_scan() const -> const LogicalWalScan & { return recovery_scan_; }
  [[nodiscard]] auto path() const -> const std::filesystem::path & { return path_; }
  [[nodiscard]] auto directory() const -> const std::filesystem::path & { return directory_; }

  [[nodiscard]] static auto scan_file(const std::filesystem::path &path)
      -> core::Result<LogicalWalScan> {
    try {
      LogicalWalScan result;
      if (!std::filesystem::exists(path)) {
        return result;
      }
      const auto file_size = std::filesystem::file_size(path);
      if (file_size > std::numeric_limits<std::size_t>::max()) {
        return core::Status::error(core::StatusCode::corruption,
                                   core::OperationStage::mutation_replay,
                                   core::StatusDetail::arithmetic_overflow,
                                   "collection WAL is too large for this process");
      }
      std::vector<std::byte> bytes(static_cast<std::size_t>(file_size));
      if (!bytes.empty()) {
        std::ifstream input(path, std::ios::binary);
        input.read(reinterpret_cast<char *>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        if (!input) {
          return logical_wal_detail::io_error(core::OperationStage::mutation_replay,
                                              "cannot read collection logical WAL");
        }
      }
      std::size_t offset{};
      while (offset < bytes.size()) {
        const auto remaining = bytes.size() - offset;
        if (remaining < logical_wal_detail::kHeaderBytes) {
          result.stopped_at_corrupt_or_torn_tail = true;
          break;
        }
        const auto input = std::span<const std::byte>(bytes).subspan(offset);
        const auto magic = logical_wal_detail::get_u32(input, 0);
        const auto version = logical_wal_detail::get_u16(input, 4);
        const auto type =
            logical_wal_detail::parse_record_type(std::to_integer<std::uint8_t>(input[6]));
        const auto flags = std::to_integer<std::uint8_t>(input[7]);
        const auto frame_bytes = logical_wal_detail::get_u32(input, 8);
        const auto payload_bytes = logical_wal_detail::get_u32(input, 12);
        if (magic != logical_wal_detail::kFrameMagic ||
            version != logical_wal_detail::kFormatVersion || !type.has_value() ||
            payload_bytes > logical_wal_detail::kMaximumPayloadBytes ||
            frame_bytes != logical_wal_detail::kHeaderBytes + payload_bytes +
                               logical_wal_detail::kTrailerBytes ||
            frame_bytes > remaining) {
          result.stopped_at_corrupt_or_torn_tail = true;
          break;
        }
        const auto frame = input.first(frame_bytes);
        if (logical_wal_detail::get_u32(frame, frame_bytes - 4) !=
            logical_wal_detail::kTrailerMagic) {
          result.stopped_at_corrupt_or_torn_tail = true;
          break;
        }
        auto checksum_input = std::vector<std::byte>(frame.begin(), frame.end());
        const auto expected =
            logical_wal_detail::get_u32(frame, logical_wal_detail::kChecksumOffset);
        std::fill_n(checksum_input.begin() + logical_wal_detail::kChecksumOffset, 4, std::byte{});
        if (logical_wal_detail::crc32(checksum_input) != expected) {
          result.stopped_at_corrupt_or_torn_tail = true;
          break;
        }
        LogicalWalFrame decoded;
        decoded.type = *type;
        decoded.flags = flags;
        decoded.op_id = logical_wal_detail::get_u64(frame, 16);
        decoded.batch_id = logical_wal_detail::get_u64(frame, 24);
        decoded.payload.assign(frame.begin() + logical_wal_detail::kHeaderBytes,
                               frame.end() - logical_wal_detail::kTrailerBytes);
        decoded.offset = offset;
        decoded.size = frame_bytes;
        result.frames.push_back(std::move(decoded));
        offset += frame_bytes;
        result.valid_bytes = offset;
      }
      return result;
    } catch (const std::exception &error) {
      return logical_wal_detail::io_error(core::OperationStage::mutation_replay, error.what());
    } catch (...) {
      return core::status_from_exception(core::OperationStage::mutation_replay);
    }
  }

 private:
  explicit CollectionLogicalWal(std::filesystem::path directory)
      : directory_(std::move(directory)), path_(directory_ / kCollectionWalFilename) {}

  void open_stream() {
    // Keeping weak searchable frames in this userspace buffer makes the
    // crash-loss contract observable under SIGKILL. A later durable commit or
    // checkpoint deliberately flushes/promotes the preceding prefix.
    stream_.rdbuf()->pubsetbuf(write_buffer_.data(),
                               static_cast<std::streamsize>(write_buffer_.size()));
    stream_.open(path_, std::ios::binary | std::ios::app);
    if (!stream_) {
      throw std::runtime_error("cannot open collection logical WAL for append");
    }
  }

  std::filesystem::path directory_{};
  std::filesystem::path path_{};
  LogicalWalScan recovery_scan_{};
  mutable std::mutex mutex_{};
  std::array<char, 1U << 20U> write_buffer_{};
  std::ofstream stream_{};
};

}  // namespace alaya::internal::collection
