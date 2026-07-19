// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "core/status.hpp"
#include "core/value_types.hpp"
#include "index/collection/artifact_manifest_v2.hpp"
#include "index/collection/artifact_transaction.hpp"
#include "platform/fs.hpp"

namespace alaya::internal::collection {

struct UnifiedManifestView {
  ArtifactManifestV2 manifest{};

  [[nodiscard]] auto field_was_defaulted(
      [[maybe_unused]] std::string_view field_path) const noexcept -> bool {
    return false;
  }
};

struct ManifestReaderOptions {
  std::uint32_t reader_version{kArtifactManifestV2SchemaVersion};
  bool verify_artifacts{true};
  // Keep qg_segment recognizable so dispatch can return the deliberate
  // legacy/re-seal diagnostic instead of an unknown-feature error.
  std::set<std::string> available_features{"manifest_v2",
                                           "disk_flat_segment",
                                           "qg_segment",
                                           "qg_laser_segment",
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
      if (first_line != "version=2") {
        return core::Status::error(core::StatusCode::corruption,
                                   core::OperationStage::open,
                                   core::StatusDetail::malformed_struct,
                                   "collection manifest version is not v2");
      }
      UnifiedManifestView view;
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
