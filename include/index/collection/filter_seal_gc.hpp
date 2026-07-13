// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "index/collection/artifact_manifest_v2.hpp"
#include "index/collection/types.hpp"
#include "utils/platform.hpp"
#include "utils/platform_fs.hpp"

namespace alaya::internal::collection {

inline constexpr std::string_view kFilterSealGcNamespace{"filter_seal_gc_v1"};
inline constexpr std::string_view kFilterSealGcStateFilename{"STATE"};
inline constexpr std::string_view kFilterSealGcMapFilename{"replacement.map"};

enum class CollectionControlOperation : std::uint8_t {
  idle = 0,
  seal = 1,
  compact = 2,
};

enum class CollectionControlPhase : std::uint8_t {
  idle = 0,
  cut_pending = 1,
  successor_active = 2,
  building = 3,
  manifest_published = 4,
};

struct CollectionControlState {
  std::uint32_t format_version{1};
  CollectionControlOperation operation{CollectionControlOperation::idle};
  CollectionControlPhase phase{CollectionControlPhase::idle};
  std::uint64_t active_segment_id{2};
  std::uint64_t active_generation{1};
  std::uint64_t next_segment_id{3};
  std::vector<RowAddress> sources{};
  std::uint64_t successor_segment_id{};
  std::uint64_t successor_generation{};
  std::uint64_t target_segment_id{};
  std::uint64_t target_generation{};
  std::uint64_t wal_cut{};
  std::uint64_t manifest_generation{};
  std::uint64_t last_sealed_segment_id{};
  std::uint64_t last_sealed_generation{};
  std::uint64_t compacted_bytes{};
  std::uint64_t pending_compacted_bytes{};
  std::uint64_t auto_seal_rows{};
  std::string mapping_file{};
};

namespace filter_seal_gc_detail {

[[nodiscard]] inline auto state_directory(const std::filesystem::path &root)
    -> std::filesystem::path {
  return root / ".alaya_internal" / kFilterSealGcNamespace;
}

[[nodiscard]] inline auto state_path(const std::filesystem::path &root) -> std::filesystem::path {
  return state_directory(root) / kFilterSealGcStateFilename;
}

[[nodiscard]] inline auto parse_u64(std::string_view value, std::string_view field)
    -> std::uint64_t {
  if (value.empty()) {
    throw std::invalid_argument("Gate-10 control integer is empty: " + std::string(field));
  }
  std::uint64_t parsed{};
  for (const auto digit : value) {
    if (digit < '0' || digit > '9' ||
        parsed >
            (std::numeric_limits<std::uint64_t>::max() - static_cast<std::uint64_t>(digit - '0')) /
                10U) {
      throw std::invalid_argument("Gate-10 control integer is invalid: " + std::string(field));
    }
    parsed = parsed * 10U + static_cast<std::uint64_t>(digit - '0');
  }
  return parsed;
}

[[nodiscard]] inline auto parse_fields(std::string_view body)
    -> std::map<std::string, std::string, std::less<>> {
  std::map<std::string, std::string, std::less<>> fields;
  std::istringstream lines{std::string(body)};
  for (std::string line; std::getline(lines, line);) {
    const auto equal = line.find('=');
    if (equal == std::string::npos || equal == 0 ||
        !fields.emplace(line.substr(0, equal), line.substr(equal + 1)).second) {
      throw std::invalid_argument("Gate-10 control file contains an invalid field");
    }
  }
  return fields;
}

[[nodiscard]] inline auto required(const std::map<std::string, std::string, std::less<>> &fields,
                                   std::string_view key) -> const std::string & {
  const auto found = fields.find(key);
  if (found == fields.end()) {
    throw std::invalid_argument("Gate-10 control file is missing field " + std::string(key));
  }
  return found->second;
}

[[nodiscard]] inline auto read_bounded(const std::filesystem::path &path,
                                       std::uint64_t maximum_bytes) -> std::string {
  if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) > maximum_bytes) {
    throw std::invalid_argument("Gate-10 control file is absent or exceeds its size limit");
  }
  std::ifstream input(path, std::ios::binary);
  std::string body{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  if (!input.eof() && !input) {
    throw std::runtime_error("cannot read Gate-10 control file");
  }
  return body;
}

inline void atomic_write(const std::filesystem::path &path, std::string_view body) {
  std::filesystem::create_directories(path.parent_path());
  const auto temporary = path.string() + ".tmp." + std::to_string(platform::get_pid());
  platform::write_all_fsync(temporary, body.data(), body.size());
  platform::atomic_replace(temporary, path);
  platform::sync_directory_or_throw(path.parent_path());
}

[[nodiscard]] inline auto encode_bytes(std::span<const std::byte> bytes) -> std::string {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string output(bytes.size() * 2, '0');
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    const auto value = std::to_integer<std::uint8_t>(bytes[index]);
    output[index * 2] = kHex[value >> 4U];
    output[index * 2 + 1] = kHex[value & 0x0fU];
  }
  return output;
}

[[nodiscard]] inline auto decode_bytes(std::string_view encoded) -> std::vector<std::byte> {
  if ((encoded.size() & 1U) != 0U) {
    throw std::invalid_argument("Gate-10 replacement map has odd hexadecimal length");
  }
  const auto digit = [](char value) -> std::uint8_t {
    if (value >= '0' && value <= '9') {
      return static_cast<std::uint8_t>(value - '0');
    }
    if (value >= 'a' && value <= 'f') {
      return static_cast<std::uint8_t>(value - 'a' + 10);
    }
    throw std::invalid_argument("Gate-10 replacement map contains non-hexadecimal data");
  };
  std::vector<std::byte> bytes(encoded.size() / 2);
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    bytes[index] =
        static_cast<std::byte>((digit(encoded[index * 2]) << 4U) | digit(encoded[index * 2 + 1]));
  }
  return bytes;
}

[[nodiscard]] inline auto decode_logical_id(core::LogicalIdKind kind, std::string_view encoded)
    -> core::LogicalId {
  const auto bytes = decode_bytes(encoded);
  if (kind == core::LogicalIdKind::utf8) {
    return core::LogicalId::from_utf8(
        std::string_view(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
  }
  if (kind != core::LogicalIdKind::legacy_uint64 || bytes.size() != sizeof(std::uint64_t)) {
    throw std::invalid_argument("Gate-10 replacement map LogicalId encoding is invalid");
  }
  std::uint64_t value{};
  for (const auto byte : bytes) {
    value = (value << 8U) | std::to_integer<std::uint8_t>(byte);
  }
  return core::LogicalId::from_legacy_uint64(value);
}

}  // namespace filter_seal_gc_detail

class CollectionControlStore {
 public:
  [[nodiscard]] static auto exists(const std::filesystem::path &root) -> bool {
    return std::filesystem::is_regular_file(filter_seal_gc_detail::state_path(root));
  }

  [[nodiscard]] static auto load(const std::filesystem::path &root)
      -> core::Result<CollectionControlState> {
    try {
      const auto body =
          filter_seal_gc_detail::read_bounded(filter_seal_gc_detail::state_path(root), 1U << 20U);
      const auto fields = filter_seal_gc_detail::parse_fields(body);
      const auto checksum = filter_seal_gc_detail::required(fields, "checksum");
      const auto checksum_offset = body.rfind("checksum=");
      if (checksum_offset == std::string::npos ||
          sha256(body.substr(0, checksum_offset)).hex() != checksum) {
        throw std::invalid_argument("Gate-10 control state checksum is invalid");
      }
      CollectionControlState state;
      state.format_version = static_cast<std::uint32_t>(
          filter_seal_gc_detail::parse_u64(filter_seal_gc_detail::required(fields, "format"),
                                           "format"));
      state.operation = static_cast<CollectionControlOperation>(
          filter_seal_gc_detail::parse_u64(filter_seal_gc_detail::required(fields, "operation"),
                                           "operation"));
      state.phase = static_cast<CollectionControlPhase>(
          filter_seal_gc_detail::parse_u64(filter_seal_gc_detail::required(fields, "phase"),
                                           "phase"));
      state.active_segment_id =
          filter_seal_gc_detail::parse_u64(filter_seal_gc_detail::required(fields,
                                                                           "active_segment_id"),
                                           "active_segment_id");
      state.active_generation =
          filter_seal_gc_detail::parse_u64(filter_seal_gc_detail::required(fields,
                                                                           "active_generation"),
                                           "active_generation");
      state.next_segment_id =
          filter_seal_gc_detail::parse_u64(filter_seal_gc_detail::required(fields,
                                                                           "next_segment_id"),
                                           "next_segment_id");
      state.successor_segment_id =
          filter_seal_gc_detail::parse_u64(filter_seal_gc_detail::required(fields,
                                                                           "successor_segment_id"),
                                           "successor_segment_id");
      state.successor_generation =
          filter_seal_gc_detail::parse_u64(filter_seal_gc_detail::required(fields,
                                                                           "successor_generation"),
                                           "successor_generation");
      state.target_segment_id =
          filter_seal_gc_detail::parse_u64(filter_seal_gc_detail::required(fields,
                                                                           "target_segment_id"),
                                           "target_segment_id");
      state.target_generation =
          filter_seal_gc_detail::parse_u64(filter_seal_gc_detail::required(fields,
                                                                           "target_generation"),
                                           "target_generation");
      state.wal_cut =
          filter_seal_gc_detail::parse_u64(filter_seal_gc_detail::required(fields, "wal_cut"),
                                           "wal_cut");
      state.manifest_generation =
          filter_seal_gc_detail::parse_u64(filter_seal_gc_detail::required(fields,
                                                                           "manifest_generation"),
                                           "manifest_generation");
      state.last_sealed_segment_id = filter_seal_gc_detail::
          parse_u64(filter_seal_gc_detail::required(fields, "last_sealed_segment_id"),
                    "last_sealed_segment_id");
      state.last_sealed_generation = filter_seal_gc_detail::
          parse_u64(filter_seal_gc_detail::required(fields, "last_sealed_generation"),
                    "last_sealed_generation");
      state.compacted_bytes =
          filter_seal_gc_detail::parse_u64(filter_seal_gc_detail::required(fields,
                                                                           "compacted_bytes"),
                                           "compacted_bytes");
      state.pending_compacted_bytes = filter_seal_gc_detail::
          parse_u64(filter_seal_gc_detail::required(fields, "pending_compacted_bytes"),
                    "pending_compacted_bytes");
      state.auto_seal_rows =
          filter_seal_gc_detail::parse_u64(filter_seal_gc_detail::required(fields,
                                                                           "auto_seal_rows"),
                                           "auto_seal_rows");
      state.mapping_file = artifact_manifest_v2_detail::decode_string(
          filter_seal_gc_detail::required(fields, "mapping_file"));
      const auto source_count =
          filter_seal_gc_detail::parse_u64(filter_seal_gc_detail::required(fields, "source_count"),
                                           "source_count");
      if (state.format_version != 1 || state.active_segment_id == 0 ||
          state.active_generation == 0 || state.next_segment_id == 0 || source_count > 4096 ||
          static_cast<std::uint8_t>(state.operation) >
              static_cast<std::uint8_t>(CollectionControlOperation::compact) ||
          static_cast<std::uint8_t>(state.phase) >
              static_cast<std::uint8_t>(CollectionControlPhase::manifest_published)) {
        throw std::invalid_argument("Gate-10 control state identity or range is invalid");
      }
      for (std::uint64_t index = 0; index < source_count; ++index) {
        const auto prefix = "source." + std::to_string(index) + ".";
        state.sources.push_back(
            {filter_seal_gc_detail::parse_u64(filter_seal_gc_detail::required(fields,
                                                                              prefix +
                                                                                  "segment_id"),
                                              prefix + "segment_id"),
             filter_seal_gc_detail::parse_u64(filter_seal_gc_detail::required(fields,
                                                                              prefix +
                                                                                  "generation"),
                                              prefix + "generation"),
             core::SegmentRowId{}});
      }
      return state;
    } catch (const std::invalid_argument &error) {
      return core::Status::error(core::StatusCode::corruption,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 error.what());
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

  [[nodiscard]] static auto save(const std::filesystem::path &root,
                                 const CollectionControlState &state) -> core::Status {
    try {
      std::string prefix;
      const auto append = [&](std::string_view key, auto value) {
        prefix += std::string(key) + "=" + std::to_string(value) + "\n";
      };
      append("format", state.format_version);
      append("operation", static_cast<std::uint8_t>(state.operation));
      append("phase", static_cast<std::uint8_t>(state.phase));
      append("active_segment_id", state.active_segment_id);
      append("active_generation", state.active_generation);
      append("next_segment_id", state.next_segment_id);
      append("successor_segment_id", state.successor_segment_id);
      append("successor_generation", state.successor_generation);
      append("target_segment_id", state.target_segment_id);
      append("target_generation", state.target_generation);
      append("wal_cut", state.wal_cut);
      append("manifest_generation", state.manifest_generation);
      append("last_sealed_segment_id", state.last_sealed_segment_id);
      append("last_sealed_generation", state.last_sealed_generation);
      append("compacted_bytes", state.compacted_bytes);
      append("pending_compacted_bytes", state.pending_compacted_bytes);
      append("auto_seal_rows", state.auto_seal_rows);
      prefix +=
          "mapping_file=" + artifact_manifest_v2_detail::encode_string(state.mapping_file) + "\n";
      append("source_count", state.sources.size());
      for (std::size_t index = 0; index < state.sources.size(); ++index) {
        append("source." + std::to_string(index) + ".segment_id", state.sources[index].segment_id);
        append("source." + std::to_string(index) + ".generation", state.sources[index].generation);
      }
      filter_seal_gc_detail::atomic_write(filter_seal_gc_detail::state_path(root),
                                          prefix + "checksum=" + sha256(prefix).hex() + "\n");
      return core::Status::success();
    } catch (...) {
      return core::status_from_exception(core::OperationStage::save);
    }
  }

  [[nodiscard]] static auto mapping_path(const std::filesystem::path &root,
                                         std::string_view logical_name) -> std::filesystem::path {
    return filter_seal_gc_detail::state_directory(root) / logical_name;
  }

  [[nodiscard]] static auto save_replacements(const std::filesystem::path &root,
                                              std::string_view logical_name,
                                              std::span<const SegmentReplacement> replacements)
      -> core::Status {
    try {
      if (logical_name.empty() || logical_name.find('/') != std::string_view::npos ||
          replacements.size() > (1U << 24U)) {
        throw std::invalid_argument("Gate-10 replacement map name/count is invalid");
      }
      std::string prefix = "format=1\ncount=" + std::to_string(replacements.size()) + "\n";
      for (std::size_t index = 0; index < replacements.size(); ++index) {
        const auto &replacement = replacements[index];
        const auto field = "row." + std::to_string(index) + ".";
        prefix += field + "logical_kind=" +
                  std::to_string(static_cast<std::uint8_t>(replacement.logical_id.kind())) + "\n";
        prefix += field + "logical_bytes=" +
                  filter_seal_gc_detail::encode_bytes(replacement.logical_id.canonical_bytes()) +
                  "\n";
        prefix +=
            field + "source_segment_id=" + std::to_string(replacement.source.segment_id) + "\n";
        prefix +=
            field + "source_generation=" + std::to_string(replacement.source.generation) + "\n";
        prefix += field + "source_row=" +
                  std::to_string(static_cast<std::uint64_t>(replacement.source.row_id)) + "\n";
        prefix +=
            field + "target_segment_id=" + std::to_string(replacement.target.segment_id) + "\n";
        prefix +=
            field + "target_generation=" + std::to_string(replacement.target.generation) + "\n";
        prefix += field + "target_row=" +
                  std::to_string(static_cast<std::uint64_t>(replacement.target.row_id)) + "\n";
        prefix += field + "upsert_sequence=" + std::to_string(replacement.upsert_sequence) + "\n";
      }
      filter_seal_gc_detail::atomic_write(mapping_path(root, logical_name),
                                          prefix + "checksum=" + sha256(prefix).hex() + "\n");
      return core::Status::success();
    } catch (const std::invalid_argument &error) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::save,
                                 core::StatusDetail::malformed_struct,
                                 error.what());
    } catch (...) {
      return core::status_from_exception(core::OperationStage::save);
    }
  }

  [[nodiscard]] static auto load_replacements(const std::filesystem::path &root,
                                              std::string_view logical_name)
      -> core::Result<std::vector<SegmentReplacement>> {
    try {
      const auto body =
          filter_seal_gc_detail::read_bounded(mapping_path(root, logical_name), 256U << 20U);
      const auto fields = filter_seal_gc_detail::parse_fields(body);
      const auto checksum_offset = body.rfind("checksum=");
      if (checksum_offset == std::string::npos ||
          sha256(body.substr(0, checksum_offset)).hex() !=
              filter_seal_gc_detail::required(fields, "checksum") ||
          filter_seal_gc_detail::required(fields, "format") != "1") {
        throw std::invalid_argument("Gate-10 replacement map checksum/version is invalid");
      }
      const auto count =
          filter_seal_gc_detail::parse_u64(filter_seal_gc_detail::required(fields, "count"),
                                           "count");
      if (count > (1U << 24U)) {
        throw std::invalid_argument("Gate-10 replacement map row count is excessive");
      }
      std::vector<SegmentReplacement> replacements;
      replacements.reserve(static_cast<std::size_t>(count));
      for (std::uint64_t index = 0; index < count; ++index) {
        const auto field = "row." + std::to_string(index) + ".";
        SegmentReplacement replacement;
        replacement.logical_id = filter_seal_gc_detail::
            decode_logical_id(static_cast<core::LogicalIdKind>(
                                  filter_seal_gc_detail::
                                      parse_u64(filter_seal_gc_detail::required(fields,
                                                                                field +
                                                                                    "logical_kind"),
                                                field + "logical_kind")),
                              filter_seal_gc_detail::required(fields, field + "logical_bytes"));
        replacement.source.segment_id = filter_seal_gc_detail::
            parse_u64(filter_seal_gc_detail::required(fields, field + "source_segment_id"),
                      field + "source_segment_id");
        replacement.source.generation = filter_seal_gc_detail::
            parse_u64(filter_seal_gc_detail::required(fields, field + "source_generation"),
                      field + "source_generation");
        replacement.source.row_id = core::SegmentRowId(
            filter_seal_gc_detail::parse_u64(filter_seal_gc_detail::required(fields,
                                                                             field + "source_row"),
                                             field + "source_row"));
        replacement.target.segment_id = filter_seal_gc_detail::
            parse_u64(filter_seal_gc_detail::required(fields, field + "target_segment_id"),
                      field + "target_segment_id");
        replacement.target.generation = filter_seal_gc_detail::
            parse_u64(filter_seal_gc_detail::required(fields, field + "target_generation"),
                      field + "target_generation");
        replacement.target.row_id = core::SegmentRowId(
            filter_seal_gc_detail::parse_u64(filter_seal_gc_detail::required(fields,
                                                                             field + "target_row"),
                                             field + "target_row"));
        replacement.upsert_sequence =
            filter_seal_gc_detail::parse_u64(filter_seal_gc_detail::required(fields,
                                                                             field +
                                                                                 "upsert_sequence"),
                                             field + "upsert_sequence");
        replacements.push_back(std::move(replacement));
      }
      return replacements;
    } catch (const std::invalid_argument &error) {
      return core::Status::error(core::StatusCode::corruption,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 error.what());
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

  static void remove_replacements(const std::filesystem::path &root,
                                  std::string_view logical_name) noexcept {
    std::error_code error;
    std::filesystem::remove(mapping_path(root, logical_name), error);
  }
};

[[nodiscard]] inline auto load_manifest_v2_if_present(const std::filesystem::path &root)
    -> core::Result<std::optional<ArtifactManifestV2>> {
  try {
    const auto path = root / kCollectionManifestFilename;
    if (!std::filesystem::is_regular_file(path)) {
      return std::optional<ArtifactManifestV2>{};
    }
    const auto prefix = platform::read_file_prefix(path, 9);
    if (prefix != "version=2") {
      return std::optional<ArtifactManifestV2>{};
    }
    return std::optional<ArtifactManifestV2>{ArtifactManifestV2::load(path)};
  } catch (const std::invalid_argument &error) {
    return core::Status::error(core::StatusCode::corruption,
                               core::OperationStage::open,
                               core::StatusDetail::malformed_struct,
                               error.what());
  } catch (...) {
    return core::status_from_exception(core::OperationStage::open);
  }
}

[[nodiscard]] inline auto publish_manifest_v2_atomic(const std::filesystem::path &root,
                                                     const ArtifactManifestV2 &manifest)
    -> core::Status {
  try {
    const auto body = manifest.serialize();
    const auto temporary =
        root / (".collection_manifest.v2.g10." + std::to_string(platform::get_pid()));
    platform::write_all_fsync(temporary, body.data(), body.size());
    platform::atomic_replace(temporary, root / kCollectionManifestFilename);
    platform::sync_directory_or_throw(root);
    return core::Status::success();
  } catch (const std::invalid_argument &error) {
    return core::Status::error(core::StatusCode::invalid_argument,
                               core::OperationStage::save,
                               core::StatusDetail::malformed_struct,
                               error.what());
  } catch (...) {
    return core::status_from_exception(core::OperationStage::save);
  }
}

}  // namespace alaya::internal::collection
