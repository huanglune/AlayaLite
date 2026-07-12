// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/algorithm_registry.hpp"
#include "core/value_types.hpp"
#include "index/collection/sha256.hpp"
#include "utils/platform_fs.hpp"

namespace alaya::internal::collection {

inline constexpr std::uint32_t kArtifactManifestV2SchemaVersion = 2;
inline constexpr std::string_view kCollectionManifestFilename = "collection_manifest.txt";

enum class LogicalIdEncodingV2 : std::uint8_t {
  canonical_kind_and_bytes = 0,
  legacy_u64_le = 1,
};

enum class ChecksumAlgorithmV2 : std::uint8_t {
  none = 0,
  sha256 = 1,
};

enum class SegmentRoleV2 : std::uint8_t {
  searchable = 0,
  build_source = 1,
  build_target = 2,
};

enum class SegmentLifecycleV2 : std::uint8_t {
  active = 0,
  successor = 1,
  sealed = 2,
  retired = 3,
  gc_pending = 4,
};

enum class GcPhaseV2 : std::uint8_t {
  idle = 0,
  pending = 1,
  reclaimable = 2,
};

struct ReaderCompatibilityV2 {
  std::uint32_t minimum_reader_version{kArtifactManifestV2SchemaVersion};
  std::uint32_t maximum_reader_version{kArtifactManifestV2SchemaVersion};
  std::vector<std::string> required_features{};

  auto operator<=>(const ReaderCompatibilityV2 &) const = default;
};

struct OwnedArtifactV2 {
  std::string logical_name{};
  std::string relative_path{};
  bool required{true};
  std::uint64_t size_bytes{};
  ChecksumAlgorithmV2 checksum_algorithm{ChecksumAlgorithmV2::sha256};
  Sha256Digest digest{};
  bool ready{};
  ReaderCompatibilityV2 reader_compatibility{};

  auto operator<=>(const OwnedArtifactV2 &) const = default;
};

struct CapabilitiesSnapshotV2 {
  std::uint64_t operations{};
  bool reentrant_search{true};
  bool search_with_stage{};
  bool search_with_publish{};
  bool serial_mutation{true};
  bool checkpoint_with_search{};
  bool freeze_with_search{};
  bool native_async{};
  bool cooperative_cancel{true};
  bool explicit_drain{true};

  [[nodiscard]] static auto from_runtime(const core::RuntimeCapabilities &value)
      -> CapabilitiesSnapshotV2 {
    CapabilitiesSnapshotV2 snapshot;
    snapshot.operations = value.operations;
    snapshot.reentrant_search = value.concurrency.reentrant_search;
    snapshot.search_with_stage = value.concurrency.search_with_stage;
    snapshot.search_with_publish = value.concurrency.search_with_publish;
    snapshot.serial_mutation = value.concurrency.serial_mutation;
    snapshot.checkpoint_with_search = value.concurrency.checkpoint_with_search;
    snapshot.freeze_with_search = value.concurrency.freeze_with_search;
    snapshot.native_async = value.concurrency.native_async;
    snapshot.cooperative_cancel = value.concurrency.cooperative_cancel;
    snapshot.explicit_drain = value.concurrency.explicit_drain;
    return snapshot;
  }

  auto operator<=>(const CapabilitiesSnapshotV2 &) const = default;
};

struct CollectionSchemaManifestV2 {
  std::string schema_name{"alaya.collection"};
  std::uint32_t schema_version{1};
  std::uint32_t dim{};
  core::Metric metric{core::Metric::l2};
  core::ScalarType scalar_type{core::ScalarType::float32};
  LogicalIdEncodingV2 logical_id_encoding{LogicalIdEncodingV2::canonical_kind_and_bytes};
  std::uint32_t logical_id_encoding_version{1};
  std::string metadata_checkpoint{};
  std::uint64_t metadata_epoch{};

  auto operator<=>(const CollectionSchemaManifestV2 &) const = default;
};

struct PublicationManifestV2 {
  std::uint64_t generation{1};
  std::string parent{};

  auto operator<=>(const PublicationManifestV2 &) const = default;
};

struct RowVersionRangeV2 {
  std::uint64_t minimum{};
  std::uint64_t maximum{};

  auto operator<=>(const RowVersionRangeV2 &) const = default;
};

struct SegmentEntryV2 {
  std::string segment_id{};
  std::uint64_t generation{1};
  SegmentRoleV2 role{SegmentRoleV2::searchable};
  core::AlgorithmId algorithm_id{};
  std::uint32_t format_version{};
  std::string factory_key{};
  CapabilitiesSnapshotV2 capabilities{};
  SegmentLifecycleV2 lifecycle{SegmentLifecycleV2::sealed};
  std::uint64_t wal_cut{};
  RowVersionRangeV2 row_versions{};
  std::string id_map_checkpoint{};
  std::vector<OwnedArtifactV2> artifacts{};
  bool ready{};
  std::string ready_marker{};
  Sha256Digest ready_digest{};
  ReaderCompatibilityV2 reader_compatibility{};
  std::vector<std::string> source_retention{};
  std::map<std::string, std::string> extensions{};

  auto operator<=>(const SegmentEntryV2 &) const = default;
};

struct GcStateManifestV2 {
  GcPhaseV2 phase{GcPhaseV2::idle};
  std::uint64_t generation{};
  std::vector<std::string> pending_segment_ids{};
  std::vector<std::string> retained_sources{};

  auto operator<=>(const GcStateManifestV2 &) const = default;
};

namespace artifact_manifest_v2_detail {

[[nodiscard]] inline auto encode_string(std::string_view value) -> std::string {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string output(value.size() * 2, '0');
  for (std::size_t i = 0; i < value.size(); ++i) {
    const auto byte = static_cast<unsigned char>(value[i]);
    output[i * 2] = kHex[byte >> 4U];
    output[i * 2 + 1] = kHex[byte & 0x0fU];
  }
  return output;
}

[[nodiscard]] inline auto decode_string(std::string_view value) -> std::string {
  if ((value.size() & 1U) != 0U) {
    throw std::invalid_argument("manifest string has an odd hexadecimal length");
  }
  auto nibble = [](char digit) -> unsigned {
    if (digit >= '0' && digit <= '9') {
      return static_cast<unsigned>(digit - '0');
    }
    if (digit >= 'a' && digit <= 'f') {
      return static_cast<unsigned>(digit - 'a' + 10);
    }
    if (digit >= 'A' && digit <= 'F') {
      return static_cast<unsigned>(digit - 'A' + 10);
    }
    throw std::invalid_argument("manifest string contains a non-hexadecimal digit");
  };
  std::string output(value.size() / 2, '\0');
  for (std::size_t i = 0; i < output.size(); ++i) {
    output[i] = static_cast<char>((nibble(value[i * 2]) << 4U) | nibble(value[i * 2 + 1]));
  }
  return output;
}

[[nodiscard]] inline auto parse_u64(std::string_view value, std::string_view field)
    -> std::uint64_t {
  if (value.empty()) {
    throw std::invalid_argument("manifest numeric field is empty: " + std::string(field));
  }
  std::uint64_t result{};
  for (char digit : value) {
    if (digit < '0' || digit > '9') {
      throw std::invalid_argument("manifest numeric field is not decimal: " + std::string(field));
    }
    const auto component = static_cast<std::uint64_t>(digit - '0');
    if (result > (std::numeric_limits<std::uint64_t>::max() - component) / 10U) {
      throw std::invalid_argument("manifest numeric field overflows uint64: " + std::string(field));
    }
    result = result * 10U + component;
  }
  return result;
}

[[nodiscard]] inline auto parse_u32(std::string_view value, std::string_view field)
    -> std::uint32_t {
  const auto parsed = parse_u64(value, field);
  if (parsed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("manifest numeric field overflows uint32: " + std::string(field));
  }
  return static_cast<std::uint32_t>(parsed);
}

[[nodiscard]] inline auto parse_bool(std::string_view value, std::string_view field) -> bool {
  if (value == "0") {
    return false;
  }
  if (value == "1") {
    return true;
  }
  throw std::invalid_argument("manifest boolean field must be 0 or 1: " + std::string(field));
}

[[nodiscard]] inline auto parse_map(std::string_view body) -> std::map<std::string, std::string> {
  std::map<std::string, std::string> result;
  std::size_t offset{};
  while (offset <= body.size()) {
    auto end = body.find('\n', offset);
    if (end == std::string_view::npos) {
      end = body.size();
    }
    auto line = body.substr(offset, end - offset);
    offset = end + 1;
    if (line.empty()) {
      continue;
    }
    const auto separator = line.find('=');
    if (separator == std::string_view::npos || separator == 0) {
      throw std::invalid_argument("manifest v2 line is missing a key/value separator");
    }
    auto key = std::string(line.substr(0, separator));
    auto value = std::string(line.substr(separator + 1));
    if (!result.emplace(std::move(key), std::move(value)).second) {
      throw std::invalid_argument("manifest v2 contains a duplicate key");
    }
  }
  return result;
}

[[nodiscard]] inline auto take(std::map<std::string, std::string> &values, const std::string &key)
    -> std::string {
  auto found = values.find(key);
  if (found == values.end()) {
    throw std::invalid_argument("manifest v2 missing required key '" + key + "'");
  }
  auto value = std::move(found->second);
  values.erase(found);
  return value;
}

inline void append(std::string &output, std::string_view key, std::string_view value) {
  output.append(key);
  output.push_back('=');
  output.append(value);
  output.push_back('\n');
}

template <std::integral Integer>
  requires(!std::same_as<std::remove_cv_t<Integer>, bool>)
inline void append(std::string &output, std::string_view key, Integer value) {
  append(output, key, std::to_string(value));
}

inline void append(std::string &output, std::string_view key, bool value) {
  append(output, key, value ? std::string_view{"1"} : std::string_view{"0"});
}

[[nodiscard]] inline auto safe_relative_path(std::string_view value) -> bool {
  if (value.empty()) {
    return false;
  }
  const std::filesystem::path path(value);
  if (path.is_absolute() || path.has_root_name() || path.has_root_directory()) {
    return false;
  }
  for (const auto &component : path) {
    if (component == ".." || component == "." || component.empty()) {
      return false;
    }
  }
  return value.find('\0') == std::string_view::npos;
}

template <class Enum>
[[nodiscard]] inline auto enum_value(Enum value) -> std::uint64_t {
  return static_cast<std::uint64_t>(value);
}

template <class Enum>
[[nodiscard]] inline auto parse_enum(std::string_view value,
                                     std::string_view field,
                                     std::uint64_t maximum) -> Enum {
  const auto parsed = parse_u64(value, field);
  if (parsed > maximum) {
    throw std::invalid_argument("manifest enum field is outside its defined range: " +
                                std::string(field));
  }
  return static_cast<Enum>(parsed);
}

}  // namespace artifact_manifest_v2_detail

struct ArtifactManifestV2 {
  std::uint32_t manifest_version{kArtifactManifestV2SchemaVersion};
  CollectionSchemaManifestV2 collection{};
  PublicationManifestV2 publication{};
  std::uint64_t wal_cut{};
  RowVersionRangeV2 row_versions{};
  std::string id_map_checkpoint{};
  std::uint64_t next_segment_id_hint{1};
  std::vector<SegmentEntryV2> segments{};
  GcStateManifestV2 gc{};
  std::map<std::string, std::string> extensions{};

  [[nodiscard]] auto validate() const -> core::Status {
    if (manifest_version != kArtifactManifestV2SchemaVersion || collection.schema_name.empty() ||
        collection.schema_version == 0 || collection.dim == 0 ||
        collection.logical_id_encoding_version == 0 || publication.generation == 0 ||
        row_versions.minimum > row_versions.maximum) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "manifest v2 collection/publication fields are invalid");
    }
    std::set<std::string> segment_ids;
    std::set<std::string> artifact_paths;
    for (const auto &segment : segments) {
      if (segment.segment_id.empty() || segment.segment_id.find('\0') != std::string::npos ||
          segment.generation == 0 || segment.algorithm_id == 0 || segment.format_version == 0 ||
          segment.factory_key.empty() ||
          segment.row_versions.minimum > segment.row_versions.maximum || !segment.ready ||
          !artifact_manifest_v2_detail::safe_relative_path(segment.ready_marker) ||
          segment.reader_compatibility.minimum_reader_version == 0 ||
          segment.reader_compatibility.minimum_reader_version >
              segment.reader_compatibility.maximum_reader_version ||
          !segment_ids.insert(segment.segment_id).second) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::malformed_struct,
                                   "manifest v2 segment entry is invalid or duplicated");
      }
      std::set<std::string> logical_names;
      if (segment.artifacts.empty()) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::malformed_struct,
                                   "manifest v2 segment contains no artifacts");
      }
      for (const auto &artifact : segment.artifacts) {
        if (artifact.logical_name.empty() || !logical_names.insert(artifact.logical_name).second ||
            !artifact_paths.insert(artifact.relative_path).second ||
            !artifact_manifest_v2_detail::safe_relative_path(artifact.relative_path) ||
            artifact.checksum_algorithm != ChecksumAlgorithmV2::sha256 || !artifact.ready ||
            artifact.reader_compatibility.minimum_reader_version == 0 ||
            artifact.reader_compatibility.minimum_reader_version >
                artifact.reader_compatibility.maximum_reader_version) {
          return core::Status::error(core::StatusCode::invalid_argument,
                                     core::OperationStage::validation,
                                     core::StatusDetail::malformed_struct,
                                     "manifest v2 artifact entry is invalid or duplicated");
        }
      }
    }
    return core::Status::success();
  }

  [[nodiscard]] auto serialize() const -> std::string {
    const auto status = validate();
    if (!status.ok()) {
      throw std::invalid_argument(status.diagnostic());
    }
    namespace detail = artifact_manifest_v2_detail;
    std::string output;
    output.reserve(2048 + segments.size() * 1024);
    detail::append(output, "version", manifest_version);
    detail::append(output, "collection.schema_name", detail::encode_string(collection.schema_name));
    detail::append(output, "collection.schema_version", collection.schema_version);
    detail::append(output, "collection.dim", collection.dim);
    detail::append(output, "collection.metric", detail::enum_value(collection.metric));
    detail::append(output, "collection.scalar_type", detail::enum_value(collection.scalar_type));
    detail::append(output,
                   "collection.logical_id_encoding",
                   detail::enum_value(collection.logical_id_encoding));
    detail::append(output,
                   "collection.logical_id_encoding_version",
                   collection.logical_id_encoding_version);
    detail::append(output,
                   "collection.metadata_checkpoint",
                   detail::encode_string(collection.metadata_checkpoint));
    detail::append(output, "collection.metadata_epoch", collection.metadata_epoch);
    detail::append(output, "publication.generation", publication.generation);
    detail::append(output, "publication.parent", detail::encode_string(publication.parent));
    detail::append(output, "wal_cut", wal_cut);
    detail::append(output, "row_versions.minimum", row_versions.minimum);
    detail::append(output, "row_versions.maximum", row_versions.maximum);
    detail::append(output, "id_map_checkpoint", detail::encode_string(id_map_checkpoint));
    detail::append(output, "next_segment_id_hint", next_segment_id_hint);
    detail::append(output, "extensions.count", extensions.size());
    {
      std::size_t extension_index{};
      for (const auto &[key, value] : extensions) {
        const auto prefix = "extension." + std::to_string(extension_index++) + ".";
        detail::append(output, prefix + "key", detail::encode_string(key));
        detail::append(output, prefix + "value", detail::encode_string(value));
      }
    }
    detail::append(output, "segments.count", segments.size());

    for (std::size_t index = 0; index < segments.size(); ++index) {
      const auto prefix = "segment." + std::to_string(index) + ".";
      const auto &segment = segments[index];
      detail::append(output, prefix + "id", detail::encode_string(segment.segment_id));
      detail::append(output, prefix + "generation", segment.generation);
      detail::append(output, prefix + "role", detail::enum_value(segment.role));
      detail::append(output, prefix + "algorithm_id", segment.algorithm_id);
      detail::append(output, prefix + "format_version", segment.format_version);
      detail::append(output, prefix + "factory_key", detail::encode_string(segment.factory_key));
      detail::append(output, prefix + "capabilities.operations", segment.capabilities.operations);
      detail::append(output,
                     prefix + "capabilities.reentrant_search",
                     segment.capabilities.reentrant_search);
      detail::append(output,
                     prefix + "capabilities.search_with_stage",
                     segment.capabilities.search_with_stage);
      detail::append(output,
                     prefix + "capabilities.search_with_publish",
                     segment.capabilities.search_with_publish);
      detail::append(output,
                     prefix + "capabilities.serial_mutation",
                     segment.capabilities.serial_mutation);
      detail::append(output,
                     prefix + "capabilities.checkpoint_with_search",
                     segment.capabilities.checkpoint_with_search);
      detail::append(output,
                     prefix + "capabilities.freeze_with_search",
                     segment.capabilities.freeze_with_search);
      detail::append(output,
                     prefix + "capabilities.native_async",
                     segment.capabilities.native_async);
      detail::append(output,
                     prefix + "capabilities.cooperative_cancel",
                     segment.capabilities.cooperative_cancel);
      detail::append(output,
                     prefix + "capabilities.explicit_drain",
                     segment.capabilities.explicit_drain);
      detail::append(output, prefix + "lifecycle", detail::enum_value(segment.lifecycle));
      detail::append(output, prefix + "wal_cut", segment.wal_cut);
      detail::append(output, prefix + "row_versions.minimum", segment.row_versions.minimum);
      detail::append(output, prefix + "row_versions.maximum", segment.row_versions.maximum);
      detail::append(output,
                     prefix + "id_map_checkpoint",
                     detail::encode_string(segment.id_map_checkpoint));
      detail::append(output, prefix + "ready", segment.ready);
      detail::append(output, prefix + "ready_marker", detail::encode_string(segment.ready_marker));
      detail::append(output, prefix + "ready_sha256", segment.ready_digest.hex());
      detail::append(output,
                     prefix + "reader.minimum",
                     segment.reader_compatibility.minimum_reader_version);
      detail::append(output,
                     prefix + "reader.maximum",
                     segment.reader_compatibility.maximum_reader_version);
      detail::append(output,
                     prefix + "reader.features.count",
                     segment.reader_compatibility.required_features.size());
      for (std::size_t feature = 0; feature < segment.reader_compatibility.required_features.size();
           ++feature) {
        detail::append(output,
                       prefix + "reader.feature." + std::to_string(feature),
                       detail::encode_string(
                           segment.reader_compatibility.required_features[feature]));
      }
      detail::append(output, prefix + "source_retention.count", segment.source_retention.size());
      for (std::size_t retained = 0; retained < segment.source_retention.size(); ++retained) {
        detail::append(output,
                       prefix + "source_retention." + std::to_string(retained),
                       detail::encode_string(segment.source_retention[retained]));
      }
      detail::append(output, prefix + "extensions.count", segment.extensions.size());
      {
        std::size_t extension_index{};
        for (const auto &[key, value] : segment.extensions) {
          const auto extension_prefix =
              prefix + "extension." + std::to_string(extension_index++) + ".";
          detail::append(output, extension_prefix + "key", detail::encode_string(key));
          detail::append(output, extension_prefix + "value", detail::encode_string(value));
        }
      }
      detail::append(output, prefix + "artifacts.count", segment.artifacts.size());
      for (std::size_t artifact_index = 0; artifact_index < segment.artifacts.size();
           ++artifact_index) {
        const auto artifact_prefix = prefix + "artifact." + std::to_string(artifact_index) + ".";
        const auto &artifact = segment.artifacts[artifact_index];
        detail::append(output,
                       artifact_prefix + "logical_name",
                       detail::encode_string(artifact.logical_name));
        detail::append(output,
                       artifact_prefix + "relative_path",
                       detail::encode_string(artifact.relative_path));
        detail::append(output, artifact_prefix + "required", artifact.required);
        detail::append(output, artifact_prefix + "size_bytes", artifact.size_bytes);
        detail::append(output,
                       artifact_prefix + "checksum_algorithm",
                       detail::enum_value(artifact.checksum_algorithm));
        detail::append(output, artifact_prefix + "sha256", artifact.digest.hex());
        detail::append(output, artifact_prefix + "ready", artifact.ready);
        detail::append(output,
                       artifact_prefix + "reader.minimum",
                       artifact.reader_compatibility.minimum_reader_version);
        detail::append(output,
                       artifact_prefix + "reader.maximum",
                       artifact.reader_compatibility.maximum_reader_version);
        detail::append(output,
                       artifact_prefix + "reader.features.count",
                       artifact.reader_compatibility.required_features.size());
        for (std::size_t feature = 0;
             feature < artifact.reader_compatibility.required_features.size();
             ++feature) {
          detail::append(output,
                         artifact_prefix + "reader.feature." + std::to_string(feature),
                         detail::encode_string(
                             artifact.reader_compatibility.required_features[feature]));
        }
      }
    }
    detail::append(output, "gc.phase", detail::enum_value(gc.phase));
    detail::append(output, "gc.generation", gc.generation);
    detail::append(output, "gc.pending.count", gc.pending_segment_ids.size());
    for (std::size_t index = 0; index < gc.pending_segment_ids.size(); ++index) {
      detail::append(output,
                     "gc.pending." + std::to_string(index),
                     detail::encode_string(gc.pending_segment_ids[index]));
    }
    detail::append(output, "gc.retained.count", gc.retained_sources.size());
    for (std::size_t index = 0; index < gc.retained_sources.size(); ++index) {
      detail::append(output,
                     "gc.retained." + std::to_string(index),
                     detail::encode_string(gc.retained_sources[index]));
    }
    return output;
  }

  [[nodiscard]] static auto deserialize(std::string_view body) -> ArtifactManifestV2 {
    namespace detail = artifact_manifest_v2_detail;
    auto values = detail::parse_map(body);
    ArtifactManifestV2 manifest;
    manifest.manifest_version = detail::parse_u32(detail::take(values, "version"), "version");
    if (manifest.manifest_version != kArtifactManifestV2SchemaVersion) {
      throw std::invalid_argument("collection manifest is not schema v2");
    }
    manifest.collection.schema_name =
        detail::decode_string(detail::take(values, "collection.schema_name"));
    manifest.collection.schema_version =
        detail::parse_u32(detail::take(values, "collection.schema_version"),
                          "collection.schema_version");
    manifest.collection.dim =
        detail::parse_u32(detail::take(values, "collection.dim"), "collection.dim");
    manifest.collection.metric =
        detail::parse_enum<core::Metric>(detail::take(values, "collection.metric"),
                                         "collection.metric",
                                         2);
    manifest.collection.scalar_type =
        detail::parse_enum<core::ScalarType>(detail::take(values, "collection.scalar_type"),
                                             "collection.scalar_type",
                                             3);
    manifest.collection.logical_id_encoding =
        detail::parse_enum<LogicalIdEncodingV2>(detail::take(values,
                                                             "collection.logical_id_encoding"),
                                                "collection.logical_id_encoding",
                                                1);
    manifest.collection.logical_id_encoding_version =
        detail::parse_u32(detail::take(values, "collection.logical_id_encoding_version"),
                          "collection.logical_id_encoding_version");
    manifest.collection.metadata_checkpoint =
        detail::decode_string(detail::take(values, "collection.metadata_checkpoint"));
    manifest.collection.metadata_epoch =
        detail::parse_u64(detail::take(values, "collection.metadata_epoch"),
                          "collection.metadata_epoch");
    manifest.publication.generation =
        detail::parse_u64(detail::take(values, "publication.generation"), "publication.generation");
    manifest.publication.parent = detail::decode_string(detail::take(values, "publication.parent"));
    manifest.wal_cut = detail::parse_u64(detail::take(values, "wal_cut"), "wal_cut");
    manifest.row_versions.minimum =
        detail::parse_u64(detail::take(values, "row_versions.minimum"), "row_versions.minimum");
    manifest.row_versions.maximum =
        detail::parse_u64(detail::take(values, "row_versions.maximum"), "row_versions.maximum");
    manifest.id_map_checkpoint = detail::decode_string(detail::take(values, "id_map_checkpoint"));
    manifest.next_segment_id_hint =
        detail::parse_u64(detail::take(values, "next_segment_id_hint"), "next_segment_id_hint");
    const auto extension_count =
        detail::parse_u64(detail::take(values, "extensions.count"), "extensions.count");
    if (extension_count > 1'000'000U) {
      throw std::invalid_argument("manifest v2 extension count exceeds safety limit");
    }
    for (std::size_t index = 0; index < extension_count; ++index) {
      const auto prefix = "extension." + std::to_string(index) + ".";
      auto key = detail::decode_string(detail::take(values, prefix + "key"));
      auto value = detail::decode_string(detail::take(values, prefix + "value"));
      if (!manifest.extensions.emplace(std::move(key), std::move(value)).second) {
        throw std::invalid_argument("manifest v2 contains duplicate extension key");
      }
    }
    const auto segment_count =
        detail::parse_u64(detail::take(values, "segments.count"), "segments.count");
    if (segment_count > 1'000'000U) {
      throw std::invalid_argument("manifest v2 segment count exceeds safety limit");
    }
    manifest.segments.resize(static_cast<std::size_t>(segment_count));
    for (std::size_t index = 0; index < manifest.segments.size(); ++index) {
      const auto prefix = "segment." + std::to_string(index) + ".";
      auto &segment = manifest.segments[index];
      segment.segment_id = detail::decode_string(detail::take(values, prefix + "id"));
      segment.generation =
          detail::parse_u64(detail::take(values, prefix + "generation"), prefix + "generation");
      segment.role = detail::parse_enum<SegmentRoleV2>(detail::take(values, prefix + "role"),
                                                       prefix + "role",
                                                       2);
      segment.algorithm_id =
          detail::parse_u64(detail::take(values, prefix + "algorithm_id"), prefix + "algorithm_id");
      segment.format_version = detail::parse_u32(detail::take(values, prefix + "format_version"),
                                                 prefix + "format_version");
      segment.factory_key = detail::decode_string(detail::take(values, prefix + "factory_key"));
      segment.capabilities.operations =
          detail::parse_u64(detail::take(values, prefix + "capabilities.operations"),
                            prefix + "capabilities.operations");
      segment.capabilities.reentrant_search =
          detail::parse_bool(detail::take(values, prefix + "capabilities.reentrant_search"),
                             prefix + "capabilities.reentrant_search");
      segment.capabilities.search_with_stage =
          detail::parse_bool(detail::take(values, prefix + "capabilities.search_with_stage"),
                             prefix + "capabilities.search_with_stage");
      segment.capabilities.search_with_publish =
          detail::parse_bool(detail::take(values, prefix + "capabilities.search_with_publish"),
                             prefix + "capabilities.search_with_publish");
      segment.capabilities.serial_mutation =
          detail::parse_bool(detail::take(values, prefix + "capabilities.serial_mutation"),
                             prefix + "capabilities.serial_mutation");
      segment.capabilities.checkpoint_with_search =
          detail::parse_bool(detail::take(values, prefix + "capabilities.checkpoint_with_search"),
                             prefix + "capabilities.checkpoint_with_search");
      segment.capabilities.freeze_with_search =
          detail::parse_bool(detail::take(values, prefix + "capabilities.freeze_with_search"),
                             prefix + "capabilities.freeze_with_search");
      segment.capabilities.native_async =
          detail::parse_bool(detail::take(values, prefix + "capabilities.native_async"),
                             prefix + "capabilities.native_async");
      segment.capabilities.cooperative_cancel =
          detail::parse_bool(detail::take(values, prefix + "capabilities.cooperative_cancel"),
                             prefix + "capabilities.cooperative_cancel");
      segment.capabilities.explicit_drain =
          detail::parse_bool(detail::take(values, prefix + "capabilities.explicit_drain"),
                             prefix + "capabilities.explicit_drain");
      segment.lifecycle =
          detail::parse_enum<SegmentLifecycleV2>(detail::take(values, prefix + "lifecycle"),
                                                 prefix + "lifecycle",
                                                 4);
      segment.wal_cut =
          detail::parse_u64(detail::take(values, prefix + "wal_cut"), prefix + "wal_cut");
      segment.row_versions.minimum =
          detail::parse_u64(detail::take(values, prefix + "row_versions.minimum"),
                            prefix + "row_versions.minimum");
      segment.row_versions.maximum =
          detail::parse_u64(detail::take(values, prefix + "row_versions.maximum"),
                            prefix + "row_versions.maximum");
      segment.id_map_checkpoint =
          detail::decode_string(detail::take(values, prefix + "id_map_checkpoint"));
      segment.ready = detail::parse_bool(detail::take(values, prefix + "ready"), prefix + "ready");
      segment.ready_marker = detail::decode_string(detail::take(values, prefix + "ready_marker"));
      segment.ready_digest = Sha256Digest::from_hex(detail::take(values, prefix + "ready_sha256"));
      segment.reader_compatibility.minimum_reader_version =
          detail::parse_u32(detail::take(values, prefix + "reader.minimum"),
                            prefix + "reader.minimum");
      segment.reader_compatibility.maximum_reader_version =
          detail::parse_u32(detail::take(values, prefix + "reader.maximum"),
                            prefix + "reader.maximum");
      const auto segment_feature_count =
          detail::parse_u64(detail::take(values, prefix + "reader.features.count"),
                            prefix + "reader.features.count");
      if (segment_feature_count > 1'000'000U) {
        throw std::invalid_argument("manifest v2 segment feature count exceeds safety limit");
      }
      for (std::size_t feature = 0; feature < segment_feature_count; ++feature) {
        segment.reader_compatibility.required_features.push_back(detail::decode_string(
            detail::take(values, prefix + "reader.feature." + std::to_string(feature))));
      }
      const auto retention_count =
          detail::parse_u64(detail::take(values, prefix + "source_retention.count"),
                            prefix + "source_retention.count");
      if (retention_count > 1'000'000U) {
        throw std::invalid_argument("manifest v2 source retention count exceeds safety limit");
      }
      for (std::size_t retained = 0; retained < retention_count; ++retained) {
        segment.source_retention.push_back(detail::decode_string(
            detail::take(values, prefix + "source_retention." + std::to_string(retained))));
      }
      const auto segment_extension_count =
          detail::parse_u64(detail::take(values, prefix + "extensions.count"),
                            prefix + "extensions.count");
      if (segment_extension_count > 1'000'000U) {
        throw std::invalid_argument("manifest v2 segment extension count exceeds safety limit");
      }
      for (std::size_t extension = 0; extension < segment_extension_count; ++extension) {
        const auto extension_prefix = prefix + "extension." + std::to_string(extension) + ".";
        auto key = detail::decode_string(detail::take(values, extension_prefix + "key"));
        auto value = detail::decode_string(detail::take(values, extension_prefix + "value"));
        if (!segment.extensions.emplace(std::move(key), std::move(value)).second) {
          throw std::invalid_argument("manifest v2 segment contains duplicate extension key");
        }
      }
      const auto artifact_count =
          detail::parse_u64(detail::take(values, prefix + "artifacts.count"),
                            prefix + "artifacts.count");
      if (artifact_count > 1'000'000U) {
        throw std::invalid_argument("manifest v2 artifact count exceeds safety limit");
      }
      segment.artifacts.resize(static_cast<std::size_t>(artifact_count));
      for (std::size_t artifact_index = 0; artifact_index < segment.artifacts.size();
           ++artifact_index) {
        const auto artifact_prefix = prefix + "artifact." + std::to_string(artifact_index) + ".";
        auto &artifact = segment.artifacts[artifact_index];
        artifact.logical_name =
            detail::decode_string(detail::take(values, artifact_prefix + "logical_name"));
        artifact.relative_path =
            detail::decode_string(detail::take(values, artifact_prefix + "relative_path"));
        artifact.required = detail::parse_bool(detail::take(values, artifact_prefix + "required"),
                                               artifact_prefix + "required");
        artifact.size_bytes =
            detail::parse_u64(detail::take(values, artifact_prefix + "size_bytes"),
                              artifact_prefix + "size_bytes");
        artifact.checksum_algorithm =
            detail::parse_enum<ChecksumAlgorithmV2>(detail::take(values,
                                                                 artifact_prefix +
                                                                     "checksum_algorithm"),
                                                    artifact_prefix + "checksum_algorithm",
                                                    1);
        artifact.digest = Sha256Digest::from_hex(detail::take(values, artifact_prefix + "sha256"));
        artifact.ready = detail::parse_bool(detail::take(values, artifact_prefix + "ready"),
                                            artifact_prefix + "ready");
        artifact.reader_compatibility.minimum_reader_version =
            detail::parse_u32(detail::take(values, artifact_prefix + "reader.minimum"),
                              artifact_prefix + "reader.minimum");
        artifact.reader_compatibility.maximum_reader_version =
            detail::parse_u32(detail::take(values, artifact_prefix + "reader.maximum"),
                              artifact_prefix + "reader.maximum");
        const auto feature_count =
            detail::parse_u64(detail::take(values, artifact_prefix + "reader.features.count"),
                              artifact_prefix + "reader.features.count");
        if (feature_count > 1'000'000U) {
          throw std::invalid_argument("manifest v2 artifact feature count exceeds safety limit");
        }
        for (std::size_t feature = 0; feature < feature_count; ++feature) {
          artifact.reader_compatibility.required_features.push_back(detail::decode_string(
              detail::take(values, artifact_prefix + "reader.feature." + std::to_string(feature))));
        }
      }
    }
    manifest.gc.phase =
        detail::parse_enum<GcPhaseV2>(detail::take(values, "gc.phase"), "gc.phase", 2);
    manifest.gc.generation =
        detail::parse_u64(detail::take(values, "gc.generation"), "gc.generation");
    const auto pending_count =
        detail::parse_u64(detail::take(values, "gc.pending.count"), "gc.pending.count");
    if (pending_count > 1'000'000U) {
      throw std::invalid_argument("manifest v2 GC pending count exceeds safety limit");
    }
    for (std::size_t index = 0; index < pending_count; ++index) {
      manifest.gc.pending_segment_ids.push_back(
          detail::decode_string(detail::take(values, "gc.pending." + std::to_string(index))));
    }
    const auto retained_count =
        detail::parse_u64(detail::take(values, "gc.retained.count"), "gc.retained.count");
    if (retained_count > 1'000'000U) {
      throw std::invalid_argument("manifest v2 GC retained count exceeds safety limit");
    }
    for (std::size_t index = 0; index < retained_count; ++index) {
      manifest.gc.retained_sources.push_back(
          detail::decode_string(detail::take(values, "gc.retained." + std::to_string(index))));
    }
    if (!values.empty()) {
      throw std::invalid_argument("manifest v2 contains unknown key '" + values.begin()->first +
                                  "'");
    }
    const auto status = manifest.validate();
    if (!status.ok()) {
      throw std::invalid_argument(status.diagnostic());
    }
    return manifest;
  }

  void save(const std::filesystem::path &path) const {
    const auto body = serialize();
    platform::write_all_fsync(path, body.data(), body.size());
  }

  [[nodiscard]] static auto load(const std::filesystem::path &path) -> ArtifactManifestV2 {
    constexpr std::size_t kMaximumManifestBytes = 8U << 20U;
    return deserialize(platform::read_regular_file_bounded(path, kMaximumManifestBytes));
  }

  auto operator<=>(const ArtifactManifestV2 &) const = default;
};

}  // namespace alaya::internal::collection
