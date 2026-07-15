// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "index/collection/artifact_transaction.hpp"
#include "index/collection/types.hpp"

namespace alaya::internal::collection::detail {

struct CollectionTargetPublication {
  std::filesystem::path collection_root{};
  std::string segment_id{};
  std::uint64_t segment_generation{1};
  std::uint64_t manifest_generation{1};
  std::string publication_parent{};
  std::uint64_t metadata_epoch{};
  std::string metadata_checkpoint{};
  std::uint64_t wal_cut{};
  RowVersionRangeV2 row_versions{};
  std::string id_map_checkpoint{};
  CollectionFeatureFlags collection_features{};
  ArtifactAbortPolicy abort_policy{ArtifactAbortPolicy::eager_cleanup};
  ArtifactTransactionFailPoint fail_point{ArtifactTransactionFailPoint::none};
  std::optional<ArtifactManifestV2> base_manifest{};
};

struct MemoryCollectionTargetDefinition {
  core::AlgorithmId algorithm_id{};
  std::string implementation_key{};
  std::string factory_key{};
  std::uint32_t format_version{1};
  core::MetricPreprocessing preprocessing{core::MetricPreprocessing::none};
  std::map<std::string, std::string> entry_extensions{};
  std::vector<LogicalArtifactSpec> artifacts{};
};

[[nodiscard]] inline auto memory_target_scalar_type_name(core::ScalarType scalar_type)
    -> std::string_view {
  switch (scalar_type) {
    case core::ScalarType::float32:
      return "float32";
    case core::ScalarType::int8:
      return "int8";
    case core::ScalarType::uint8:
      return "uint8";
  }
  return "unknown";
}

[[nodiscard]] inline auto memory_target_preprocessing_name(core::MetricPreprocessing preprocessing)
    -> std::string_view {
  switch (preprocessing) {
    case core::MetricPreprocessing::none:
      return "none";
    case core::MetricPreprocessing::l2_normalized:
      return "l2_normalized";
    case core::MetricPreprocessing::engine_quantized:
      return "engine_quantized";
  }
  return "unknown";
}

[[nodiscard]] inline auto memory_target_error(core::StatusCode code,
                                              core::StatusDetail detail,
                                              std::string diagnostic) -> core::Status {
  return core::Status::error(code, core::OperationStage::save, detail, std::move(diagnostic));
}

[[nodiscard]] inline auto collection_segment_number(std::string_view segment_id)
    -> core::Result<std::uint64_t> {
  if (!segment_id.starts_with("seg_") || segment_id.size() != 12) {
    return memory_target_error(core::StatusCode::invalid_argument,
                               core::StatusDetail::malformed_struct,
                               "memory target segment identity is malformed");
  }
  std::uint64_t value{};
  const auto suffix = segment_id.substr(4);
  const auto [end, error] = std::from_chars(suffix.data(), suffix.data() + suffix.size(), value);
  if (error != std::errc() || end != suffix.data() + suffix.size()) {
    return memory_target_error(core::StatusCode::invalid_argument,
                               core::StatusDetail::malformed_struct,
                               "memory target segment identity has a non-numeric suffix");
  }
  return value;
}

[[nodiscard]] inline auto make_memory_target_manifest(
    const CollectionSchema &schema,
    const CollectionTargetPublication &publication) -> core::Result<ArtifactManifestV2> {
  auto manifest = publication.base_manifest.value_or(ArtifactManifestV2{});
  if (publication.base_manifest.has_value() &&
      (manifest.collection.dim != schema.dim || manifest.collection.metric != schema.metric ||
       manifest.collection.scalar_type != schema.scalar_type)) {
    return memory_target_error(core::StatusCode::invalid_argument,
                               core::StatusDetail::malformed_struct,
                               "memory target publication disagrees with the base collection "
                               "schema");
  }
  auto numeric_id = collection_segment_number(publication.segment_id);
  if (!numeric_id.ok()) {
    return numeric_id.status();
  }

  manifest.collection.dim = schema.dim;
  manifest.collection.metric = schema.metric;
  manifest.collection.scalar_type = schema.scalar_type;
  manifest.collection.logical_id_encoding = LogicalIdEncodingV2::legacy_u64_le;
  manifest.collection.metadata_epoch = publication.metadata_epoch;
  manifest.collection.metadata_checkpoint = publication.metadata_checkpoint;
  manifest.publication.generation = publication.manifest_generation;
  manifest.publication.parent = publication.publication_parent;
  manifest.wal_cut = publication.wal_cut;
  manifest.row_versions = publication.row_versions;
  manifest.id_map_checkpoint = publication.id_map_checkpoint;
  manifest.next_segment_id_hint =
      std::max(manifest.next_segment_id_hint, numeric_id.value() + std::uint64_t{1});
  return manifest;
}

// Memory engines expose the native save slot but do not publish Collection
// manifest-v2 entries. This helper supplies that owner-layer transaction while
// keeping the engine-specific portion limited to identity and artifact specs.
[[nodiscard]] inline auto publish_memory_collection_target(
    const core::AnySegment &segment,
    const CollectionSchema &schema,
    const CollectionTargetPublication &publication,
    MemoryCollectionTargetDefinition definition,
    core::BuildContext &context) -> core::Result<std::uint64_t> {
  try {
    const auto descriptor = segment.descriptor();
    if (publication.collection_root.empty() || publication.segment_id.empty() ||
        publication.segment_generation == 0 || publication.manifest_generation == 0 ||
        definition.algorithm_id == 0 || definition.implementation_key.empty() ||
        definition.factory_key.empty() || definition.format_version == 0 ||
        definition.artifacts.empty()) {
      return memory_target_error(core::StatusCode::invalid_argument,
                                 core::StatusDetail::malformed_struct,
                                 "memory target publication options are incomplete");
    }
    if (descriptor.algorithm_id != definition.algorithm_id ||
        descriptor.format_version != definition.format_version || descriptor.dim != schema.dim ||
        descriptor.metric != schema.metric || descriptor.stored_scalar_type != schema.scalar_type ||
        descriptor.preprocessing != definition.preprocessing) {
      return memory_target_error(core::StatusCode::invalid_argument,
                                 core::StatusDetail::malformed_struct,
                                 "memory target descriptor disagrees with its Collection schema or "
                                 "identity");
    }

    for (auto &artifact : definition.artifacts) {
      auto &features = artifact.reader_compatibility.required_features;
      if (std::ranges::find(features, definition.implementation_key) == features.end()) {
        features.push_back(definition.implementation_key);
      }
    }

    ArtifactTransactionOptions transaction_options;
    transaction_options.collection_root = publication.collection_root;
    transaction_options.target_relative_directory =
        std::filesystem::path("segments") / publication.segment_id;
    transaction_options.transaction_id = "memory_" + definition.factory_key + "_" +
                                         publication.segment_id + "_g" +
                                         std::to_string(publication.segment_generation);
    transaction_options.manifest_v2_writer = true;
    transaction_options.abort_policy = publication.abort_policy;
    transaction_options.fail_point = publication.fail_point;
    auto begun = ArtifactControlPlaneTransaction::begin(std::move(transaction_options), context);
    if (!begun.ok()) {
      return begun.status();
    }
    auto transaction = std::move(begun).value();
    auto writer = transaction->writer(std::move(definition.artifacts));
    if (!writer.ok()) {
      return writer.status();
    }

    core::ArtifactManifest native_manifest;
    auto status = segment.save(writer.value(), core::SaveOptions{}, native_manifest);
    if (!status.ok()) {
      return status;
    }
    if (!core::is_current_struct(native_manifest) || native_manifest.schema_version != 1 ||
        native_manifest.algorithm_id != definition.algorithm_id ||
        native_manifest.format_version != definition.format_version ||
        native_manifest.artifacts.empty()) {
      return memory_target_error(core::StatusCode::corruption,
                                 core::StatusDetail::malformed_struct,
                                 "memory engine save returned an incompatible native manifest");
    }

    SegmentEntryV2 entry;
    entry.segment_id = publication.segment_id;
    entry.generation = publication.segment_generation;
    entry.role = SegmentRoleV2::searchable;
    entry.algorithm_id = definition.algorithm_id;
    entry.format_version = definition.format_version;
    entry.factory_key = definition.factory_key;
    entry.capabilities = CapabilitiesSnapshotV2::from_runtime(segment.capabilities());
    entry.lifecycle = SegmentLifecycleV2::sealed;
    entry.wal_cut = publication.wal_cut;
    entry.row_versions = publication.row_versions;
    entry.id_map_checkpoint = publication.id_map_checkpoint;
    entry.reader_compatibility.required_features = {definition.implementation_key};
    entry.extensions = std::move(definition.entry_extensions);
    entry.extensions.insert_or_assign("stored_scalar_type",
                                      memory_target_scalar_type_name(
                                          descriptor.stored_scalar_type));
    entry.extensions.insert_or_assign("preprocessing",
                                      memory_target_preprocessing_name(descriptor.preprocessing));
    auto prepared = transaction->prepare(std::move(entry));
    if (!prepared.ok()) {
      return prepared.status();
    }

    std::uint64_t artifact_bytes{};
    for (const auto &artifact : prepared.value().artifacts) {
      if (!core::checked_add(artifact_bytes, artifact.size_bytes, artifact_bytes)) {
        artifact_bytes = std::numeric_limits<std::uint64_t>::max();
        break;
      }
    }

    auto manifest = make_memory_target_manifest(schema, publication);
    if (!manifest.ok()) {
      return manifest.status();
    }
    status = transaction->publish(std::move(manifest).value());
    if (!status.ok()) {
      return status;
    }
    return artifact_bytes;
  } catch (...) {
    return core::status_from_exception(core::OperationStage::save);
  }
}

}  // namespace alaya::internal::collection::detail
