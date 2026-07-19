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
#include "wal/frame.hpp"

namespace alaya::internal::collection {

inline constexpr std::string_view kCollectionWalNamespace{"collection_wal_v1"};
inline constexpr std::string_view kCollectionWalFilename{"logical.wal"};

enum class LogicalWalRecordType : std::uint8_t {
  prepare = 1,
  commit = 2,
  publish_marker = 3,
  checkpoint = 4,
  // 5 is reserved for SEGMENT_OP (segment-physical op family: row_patch,
  // tombstone, consolidate barriers, publish, superblock_flip). Consult
  // docs/design/unified-wal-vocabulary.md before assigning any new type.
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

// Physical WAL v1 framing now lives in the bottom-layer `wal/` module
// (unified-wal-vocabulary.md). These names delegate to it so the collection
// codec, the checkpoint image codec, and CollectionLogicalWal keep their
// existing spellings while the byte format has exactly one owner. The frame
// byte sequence is unchanged (locked by tests/wal/wal_frame_test.cpp).
inline constexpr std::uint32_t kFrameMagic = alaya::wal::kFrameMagic;
inline constexpr std::uint32_t kTrailerMagic = alaya::wal::kTrailerMagic;
inline constexpr std::uint16_t kFormatVersion = alaya::wal::kFormatVersion;
inline constexpr std::uint32_t kHeaderBytes = alaya::wal::kHeaderBytes;
inline constexpr std::uint32_t kTrailerBytes = alaya::wal::kTrailerBytes;
inline constexpr std::uint32_t kMaximumPayloadBytes = alaya::wal::kMaximumPayloadBytes;
inline constexpr std::size_t kChecksumOffset = alaya::wal::kChecksumOffset;

using alaya::wal::crc32;
using alaya::wal::get_u16;
using alaya::wal::get_u32;
using alaya::wal::get_u64;
using alaya::wal::put_u16;
using alaya::wal::put_u32;
using alaya::wal::put_u64;

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
  return alaya::wal::make_frame(static_cast<std::uint8_t>(type), flags, op_id, batch_id, payload);
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
                                 std::string_view namespace_name = kCollectionWalNamespace,
                                 bool read_only = false)
      -> core::Result<std::unique_ptr<CollectionLogicalWal>> {
    if (root.empty() || namespace_name != kCollectionWalNamespace) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 "collection WAL requires its fixed new namespace");
    }
    try {
      auto wal = std::unique_ptr<CollectionLogicalWal>(
          new CollectionLogicalWal(root / ".alaya_internal" / std::string(kCollectionWalNamespace),
                                   read_only));
      if (read_only && (!std::filesystem::is_directory(wal->directory_) ||
                        !std::filesystem::is_regular_file(wal->path_))) {
        return readonly_status(
            "read-only Collection open cannot create a missing logical WAL/checkpoint layout");
      }
      if (!read_only) {
        std::filesystem::create_directories(wal->directory_);
      }
      if (!read_only && !std::filesystem::exists(wal->path_)) {
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
        if (read_only) {
          return readonly_status(
              "read-only Collection open found a torn WAL tail; open in read-write mode to "
              "repair it first");
        }
        std::filesystem::resize_file(wal->path_, wal->recovery_scan_.valid_bytes);
        platform::sync_file_or_throw(wal->path_);
      }
      if (!read_only) {
        wal->open_stream();
      }
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
    if (read_only_) {
      return readonly_status("read-only Collection WAL cannot append records");
    }
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
    if (read_only_) {
      return readonly_status("read-only Collection WAL cannot publish a checkpoint");
    }
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

  // Structural scanning is delegated to the framework layer; the collection
  // layer adds the record-type semantics on top. A structurally valid frame
  // whose type is not 1-4 (for example a SEGMENT_OP type-5 frame) is a
  // cross-family contamination and fails loudly instead of being silently
  // truncated as a torn tail — no committed data is dropped without notice.
  [[nodiscard]] static auto scan_file(const std::filesystem::path &path)
      -> core::Result<LogicalWalScan> {
    try {
      auto raw = alaya::wal::WalFile::scan_path(path);
      LogicalWalScan result;
      result.valid_bytes = raw.valid_bytes;
      result.stopped_at_corrupt_or_torn_tail = raw.stopped_at_corrupt_or_torn_tail;
      result.frames.reserve(raw.frames.size());
      for (auto &frame : raw.frames) {
        const auto type = logical_wal_detail::parse_record_type(frame.type);
        if (!type.has_value()) {
          return core::Status::error(core::StatusCode::corruption,
                                     core::OperationStage::mutation_replay,
                                     core::StatusDetail::malformed_struct,
                                     "collection WAL contains an unknown record type");
        }
        LogicalWalFrame decoded;
        decoded.type = *type;
        decoded.flags = frame.flags;
        decoded.op_id = frame.op_id;
        decoded.batch_id = frame.batch_id;
        decoded.payload = std::move(frame.payload);
        decoded.offset = frame.offset;
        decoded.size = frame.size;
        result.frames.push_back(std::move(decoded));
      }
      return result;
    } catch (const std::exception &error) {
      return logical_wal_detail::io_error(core::OperationStage::mutation_replay, error.what());
    } catch (...) {
      return core::status_from_exception(core::OperationStage::mutation_replay);
    }
  }

 private:
  explicit CollectionLogicalWal(std::filesystem::path directory, bool read_only)
      : directory_(std::move(directory)),
        path_(directory_ / kCollectionWalFilename),
        read_only_(read_only) {}

  [[nodiscard]] static auto readonly_status(std::string diagnostic) -> core::Status {
    return core::Status::error(core::StatusCode::not_supported,
                               core::OperationStage::open,
                               core::StatusDetail::readonly_instance,
                               std::move(diagnostic));
  }

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
  bool read_only_{};
  LogicalWalScan recovery_scan_{};
  mutable std::mutex mutex_{};
  std::array<char, 1U << 20U> write_buffer_{};
  std::ofstream stream_{};
};

}  // namespace alaya::internal::collection
