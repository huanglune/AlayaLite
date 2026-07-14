// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/algorithm_registry.hpp"
#include "core/status.hpp"
#include "core/value_types.hpp"
#include "index/collection/artifact_manifest_v2.hpp"
#include "index/collection/artifact_transaction.hpp"
#include "index/disk/segment_manifest.hpp"
#include "platform/fs.hpp"

namespace alaya::internal::collection {

enum class ManifestSourceVersion : std::uint8_t {
  disk_collection_v1 = 1,
  artifact_manifest_v2 = 2,
};

struct ExplicitManifestDefault {
  std::string field_path{};
  std::string reason{};

  auto operator<=>(const ExplicitManifestDefault &) const = default;
};

struct UnifiedManifestView {
  ManifestSourceVersion source_version{ManifestSourceVersion::disk_collection_v1};
  ArtifactManifestV2 manifest{};
  std::vector<ExplicitManifestDefault> explicit_defaults{};

  [[nodiscard]] auto field_was_defaulted(std::string_view field_path) const noexcept -> bool {
    return std::ranges::any_of(explicit_defaults, [&](const ExplicitManifestDefault &entry) {
      return entry.field_path == field_path;
    });
  }
};

struct ManifestReaderOptions {
  std::uint32_t reader_version{kArtifactManifestV2SchemaVersion};
  bool verify_artifacts{true};
  std::set<std::string> available_features{"manifest_v2",
                                           "disk_flat_segment",
                                           "disk_vamana_segment",
                                           "disk_laser_segment"};
};

class CollectionManifestDualReader {
 public:
  [[nodiscard]] static auto open(const std::filesystem::path &collection_root,
                                 const ManifestReaderOptions &options = {}) noexcept
      -> core::Result<UnifiedManifestView> {
    try {
      const auto path = collection_root / kCollectionManifestFilename;
      constexpr std::size_t kMaximumManifestBytes = 8U << 20U;
      const auto body = platform::read_regular_file_bounded(path, kMaximumManifestBytes);
      const auto end = body.find('\n');
      const auto first_line = std::string_view(body).substr(0, end);
      if (first_line == "version=1") {
        return map_v1(collection_root, disk::CollectionManifest::load(path));
      }
      if (first_line != "version=2") {
        return core::Status::error(core::StatusCode::corruption,
                                   core::OperationStage::open,
                                   core::StatusDetail::malformed_struct,
                                   "collection manifest version is neither v1 nor v2");
      }
      UnifiedManifestView view;
      view.source_version = ManifestSourceVersion::artifact_manifest_v2;
      view.manifest = ArtifactManifestV2::deserialize(body);
      auto compatible = validate_reader_compatibility(view.manifest, options);
      if (!compatible.ok()) {
        return compatible;
      }
      if (options.verify_artifacts) {
        auto verified = verify_v2_artifacts(collection_root, view.manifest, options);
        if (!verified.ok()) {
          return verified;
        }
      }
      return view;
    } catch (const std::invalid_argument &error) {
      return core::Status::error(core::StatusCode::corruption,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 error.what());
    } catch (const std::exception &error) {
      return core::Status::error(core::StatusCode::io_error,
                                 core::OperationStage::open,
                                 core::StatusDetail::none,
                                 error.what());
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

 private:
  [[nodiscard]] static auto core_metric(core::Metric metric) noexcept -> core::Metric {
    return metric;
  }

  [[nodiscard]] static auto algorithm_id(disk::DiskIndexType type) noexcept -> core::AlgorithmId {
    switch (type) {
      case disk::DiskIndexType::Flat:
        return core::algorithm::flat;
      case disk::DiskIndexType::Vamana:
        return core::algorithm::vamana;
      case disk::DiskIndexType::Laser:
        return core::algorithm::laser;
    }
    return 0;
  }

  [[nodiscard]] static auto factory_key(disk::DiskIndexType type) -> std::string {
    switch (type) {
      case disk::DiskIndexType::Flat:
        return "flat";
      case disk::DiskIndexType::Vamana:
        return "vamana";
      case disk::DiskIndexType::Laser:
        return "laser";
    }
    return {};
  }

  static void mark_default(UnifiedManifestView &view,
                           std::string field,
                           std::string reason = "field is absent from DiskCollection manifest v1") {
    view.explicit_defaults.push_back({std::move(field), std::move(reason)});
  }

  static void add_v1_artifact(SegmentEntryV2 &entry,
                              std::string logical_name,
                              std::filesystem::path relative_path,
                              const std::filesystem::path &collection_root,
                              bool required = true) {
    OwnedArtifactV2 artifact;
    artifact.logical_name = std::move(logical_name);
    artifact.relative_path = relative_path.generic_string();
    artifact.required = required;
    std::error_code ec;
    const auto absolute = collection_root / relative_path;
    if (std::filesystem::is_regular_file(absolute, ec) && !ec) {
      artifact.size_bytes = static_cast<std::uint64_t>(std::filesystem::file_size(absolute));
    }
    artifact.checksum_algorithm = ChecksumAlgorithmV2::none;
    artifact.ready = false;
    artifact.reader_compatibility.minimum_reader_version = 1;
    artifact.reader_compatibility.maximum_reader_version = kArtifactManifestV2SchemaVersion;
    entry.artifacts.push_back(std::move(artifact));
  }

  [[nodiscard]] static auto map_v1(const std::filesystem::path &collection_root,
                                   const disk::CollectionManifest &legacy) -> UnifiedManifestView {
    UnifiedManifestView view;
    view.source_version = ManifestSourceVersion::disk_collection_v1;
    auto &mapped = view.manifest;
    if (legacy.dim > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("v1 collection dimension exceeds the v2 uint32 contract");
    }
    mapped.manifest_version = kArtifactManifestV2SchemaVersion;
    mapped.collection.schema_name = "disk_collection_v1";
    mapped.collection.schema_version = 1;
    mapped.collection.dim = static_cast<std::uint32_t>(legacy.dim);
    mapped.collection.metric = core_metric(legacy.metric);
    mapped.collection.scalar_type = core::ScalarType::float32;
    mapped.collection.logical_id_encoding = LogicalIdEncodingV2::legacy_u64_le;
    mapped.collection.logical_id_encoding_version = 1;
    mapped.next_segment_id_hint = legacy.next_segment_id;
    mapped.extensions = legacy.x_extras;
    mapped.publication.generation = 0;

    mark_default(view, "collection.metadata_checkpoint");
    mark_default(view, "collection.metadata_epoch");
    mark_default(view, "publication.generation");
    mark_default(view, "publication.parent");
    mark_default(view, "wal_cut");
    mark_default(view, "row_versions");
    mark_default(view, "id_map_checkpoint");
    mark_default(view, "gc");

    for (std::size_t index = 0; index < legacy.segment_ids.size(); ++index) {
      const auto &segment_id = legacy.segment_ids[index];
      const auto segment_relative = std::filesystem::path("segments") / segment_id;
      const auto legacy_segment =
          disk::SegmentManifest::load(collection_root / segment_relative / "manifest.txt");
      SegmentEntryV2 entry;
      entry.segment_id = legacy_segment.segment_id;
      entry.generation = 1;
      entry.role = SegmentRoleV2::searchable;
      entry.algorithm_id = algorithm_id(legacy_segment.index_type);
      entry.format_version = static_cast<std::uint32_t>(legacy_segment.version);
      entry.factory_key = factory_key(legacy_segment.index_type);
      entry.capabilities.operations =
          core::capability_bit(core::OperationCapability::search) |
          core::capability_bit(core::OperationCapability::batch_search) |
          core::capability_bit(core::OperationCapability::save) |
          core::capability_bit(core::OperationCapability::stats);
      entry.capabilities.cooperative_cancel = false;
      entry.capabilities.explicit_drain = false;
      entry.lifecycle = SegmentLifecycleV2::sealed;
      entry.reader_compatibility.minimum_reader_version = 1;
      entry.reader_compatibility.maximum_reader_version = kArtifactManifestV2SchemaVersion;
      entry.extensions = legacy_segment.x_extras;
      add_v1_artifact(entry, "manifest", segment_relative / "manifest.txt", collection_root);
      add_v1_artifact(entry, "ids", segment_relative / legacy_segment.ids_file, collection_root);
      if (!legacy_segment.vectors_file.empty()) {
        add_v1_artifact(entry,
                        "vectors",
                        segment_relative / legacy_segment.vectors_file,
                        collection_root);
      }
      std::set<std::string> recorded_paths;
      for (const auto &artifact : entry.artifacts) {
        recorded_paths.insert(artifact.relative_path);
      }
      for (const auto &[key, value] : legacy_segment.x_extras) {
        if (!key.ends_with("_file") || !disk::detail::is_valid_basename(value)) {
          continue;
        }
        const auto relative = segment_relative / value;
        if (recorded_paths.insert(relative.generic_string()).second) {
          add_v1_artifact(entry, key, relative, collection_root, false);
        }
      }
      const auto prefix = "segments[" + std::to_string(index) + "].";
      mark_default(view, prefix + "capabilities", "derived from the v1 engine contract");
      mark_default(view, prefix + "wal_cut");
      mark_default(view, prefix + "row_versions");
      mark_default(view, prefix + "id_map_checkpoint");
      mark_default(view, prefix + "ready");
      mark_default(view, prefix + "reader_compatibility");
      mark_default(view, prefix + "source_retention");
      for (std::size_t artifact = 0; artifact < entry.artifacts.size(); ++artifact) {
        mark_default(view, prefix + "artifacts[" + std::to_string(artifact) + "].sha256_ready");
      }
      mapped.segments.push_back(std::move(entry));
    }
    return view;
  }

  [[nodiscard]] static auto compatibility_supported(const ReaderCompatibilityV2 &compatibility,
                                                    const ManifestReaderOptions &options,
                                                    std::string_view subject) -> core::Status {
    if (options.reader_version < compatibility.minimum_reader_version ||
        options.reader_version > compatibility.maximum_reader_version) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::open,
                                 core::StatusDetail::unsupported_abi_version,
                                 std::string(subject) +
                                     " reader compatibility range excludes this reader");
    }
    for (const auto &feature : compatibility.required_features) {
      if (!options.available_features.contains(feature)) {
        return core::Status::error(core::StatusCode::not_supported,
                                   core::OperationStage::open,
                                   core::StatusDetail::operation_slot_absent,
                                   std::string(subject) + " requires unavailable reader feature '" +
                                       feature + "'");
      }
    }
    return core::Status::success();
  }

  [[nodiscard]] static auto validate_reader_compatibility(const ArtifactManifestV2 &manifest,
                                                          const ManifestReaderOptions &options)
      -> core::Status {
    for (const auto &segment : manifest.segments) {
      auto compatible = compatibility_supported(segment.reader_compatibility, options, "segment");
      if (!compatible.ok()) {
        return compatible;
      }
      for (const auto &artifact : segment.artifacts) {
        compatible = compatibility_supported(artifact.reader_compatibility, options, "artifact");
        if (!compatible.ok()) {
          return compatible;
        }
      }
    }
    return core::Status::success();
  }

  [[nodiscard]] static auto corrupt(std::string diagnostic) -> core::Status {
    return core::Status::error(core::StatusCode::corruption,
                               core::OperationStage::open,
                               core::StatusDetail::malformed_struct,
                               std::move(diagnostic));
  }

  [[nodiscard]] static auto verify_v2_artifacts(const std::filesystem::path &collection_root,
                                                const ArtifactManifestV2 &manifest,
                                                const ManifestReaderOptions &options)
      -> core::Status {
    (void)options;
    for (const auto &segment : manifest.segments) {
      const auto ready_path = collection_root / segment.ready_marker;
      if (!std::filesystem::is_regular_file(ready_path)) {
        return corrupt("manifest v2 segment READY marker is missing: " + segment.segment_id);
      }
      constexpr std::size_t kMaximumReadyBytes = 4096;
      const auto ready_body = platform::read_regular_file_bounded(ready_path, kMaximumReadyBytes);
      if (sha256(ready_body) != segment.ready_digest) {
        return corrupt("manifest v2 segment READY checksum mismatch: " + segment.segment_id);
      }
      const auto ready = artifact_manifest_v2_detail::parse_map(ready_body);
      const auto expected_segment = ready.find("segment");
      const auto expected_generation = ready.find("generation");
      const auto expected_owned_digest = ready.find("owned_manifest_sha256");
      if (expected_segment == ready.end() || expected_segment->second != segment.segment_id ||
          expected_generation == ready.end() ||
          artifact_manifest_v2_detail::parse_u64(expected_generation->second, "generation") !=
              segment.generation ||
          expected_owned_digest == ready.end()) {
        return corrupt("manifest v2 READY marker does not identify its segment/generation");
      }

      const OwnedArtifactV2 *owned_manifest{};
      for (const auto &artifact : segment.artifacts) {
        if (!artifact.ready || artifact.checksum_algorithm != ChecksumAlgorithmV2::sha256) {
          return corrupt("manifest v2 routed an artifact without READY/SHA-256");
        }
        const auto artifact_path = collection_root / artifact.relative_path;
        if (!std::filesystem::is_regular_file(artifact_path)) {
          return corrupt("manifest v2 artifact is missing: " + artifact.relative_path);
        }
        if (std::filesystem::file_size(artifact_path) != artifact.size_bytes) {
          return corrupt("manifest v2 artifact size mismatch: " + artifact.relative_path);
        }
        if (sha256_file(artifact_path) != artifact.digest) {
          return corrupt("manifest v2 artifact SHA-256 mismatch: " + artifact.relative_path);
        }
        if (artifact.logical_name == "artifact_manifest_v2") {
          owned_manifest = std::addressof(artifact);
        }
      }
      if (owned_manifest == nullptr ||
          expected_owned_digest->second != owned_manifest->digest.hex()) {
        return corrupt("manifest v2 READY marker is not bound to its owned artifact manifest");
      }
    }
    return core::Status::success();
  }
};

}  // namespace alaya::internal::collection
