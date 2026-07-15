// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/algorithm_registry.hpp"
#include "core/any_segment.hpp"
#include "index/collection/artifact_transaction.hpp"
#include "index/collection/manifest_dual_reader.hpp"
#include "index/collection/types.hpp"
#include "index/disk/disk_engine_registry.hpp"
#include "index/disk/vamana_segment_builder.hpp"
#include "index/disk/vamana_segment_searcher.hpp"
#include "platform/fs.hpp"

namespace alaya::disk {

struct DiskVamanaBuildInput {
  core::VersionedStructHeader header{};
  core::TypedTensorView vectors{};
  std::span<const std::uint64_t> logical_ids{};
  std::uint64_t reserved[4]{};

  DiskVamanaBuildInput() : header(core::current_struct_header<DiskVamanaBuildInput>()) {}
  DiskVamanaBuildInput(core::TypedTensorView vector_view, std::span<const std::uint64_t> ids)
      : header(core::current_struct_header<DiskVamanaBuildInput>()),
        vectors(vector_view),
        logical_ids(ids) {}
};

struct DiskVamanaPublicationOptions {
  std::filesystem::path collection_root{};
  std::string segment_id{};
  std::uint64_t segment_generation{1};
  std::uint64_t manifest_generation{1};
  std::string publication_parent{};
  std::uint64_t metadata_epoch{};
  std::string metadata_checkpoint{};
  std::uint64_t wal_cut{};
  internal::collection::RowVersionRangeV2 row_versions{};
  std::string id_map_checkpoint{};
  internal::collection::CollectionFeatureFlags collection_features{};
  internal::collection::ArtifactAbortPolicy abort_policy{
      internal::collection::ArtifactAbortPolicy::eager_cleanup};
  internal::collection::ArtifactTransactionFailPoint fail_point{
      internal::collection::ArtifactTransactionFailPoint::none};
  std::optional<internal::collection::ArtifactManifestV2> base_manifest{};
};

struct DiskVamanaSearchExtension {
  core::VersionedStructHeader header{};
  std::uint32_t effort{100};
  std::uint32_t reserved_effort{};
  std::uint64_t reserved[3]{};

  DiskVamanaSearchExtension() : header(core::current_struct_header<DiskVamanaSearchExtension>()) {}
};

[[nodiscard]] inline auto make_disk_vamana_search_extension(
    const DiskVamanaSearchExtension &options) -> core::AlgorithmSearchExtension {
  core::AlgorithmSearchExtension extension;
  extension.algorithm_id = core::algorithm::vamana;
  extension.payload = std::addressof(options);
  extension.payload_size = sizeof(options);
  return extension;
}

// DiskVamanaSegment is the immutable AnySegment producer for the retained
// disk Vamana v1 codec. The builder, reader and greedy-search kernel remain
// owned by VamanaSegmentBuilder/VamanaSegmentSearcher; this layer only adapts
// the frozen core contract and the collection-owned publication transaction.
class DiskVamanaSegment {
 public:
  static constexpr std::uint32_t kFormatVersion = 1;
  static constexpr core::AlgorithmId kAlgorithmId = core::algorithm::vamana;
  static constexpr std::string_view kManifestArtifactName{"manifest"};
  static constexpr std::string_view kIdsArtifactName{"ids"};
  static constexpr std::string_view kVectorsArtifactName{"vectors"};
  static constexpr std::string_view kGraphArtifactName{"graph"};

  DiskVamanaSegment(const DiskVamanaSegment &) = delete;
  auto operator=(const DiskVamanaSegment &) -> DiskVamanaSegment & = delete;
  DiskVamanaSegment(DiskVamanaSegment &&) = delete;
  auto operator=(DiskVamanaSegment &&) -> DiskVamanaSegment & = delete;

  [[nodiscard]] static auto build(DiskVamanaBuildInput input,
                                  core::Metric metric,
                                  const VamanaSegmentBuildParams &build_params,
                                  const DiskVamanaPublicationOptions &options,
                                  core::BuildContext &context)
      -> core::Result<std::unique_ptr<DiskVamanaSegment>> {
    auto status = validate_build_input(input, metric, build_params, options, context);
    if (!status.ok()) {
      return status;
    }

    internal::collection::ArtifactTransactionOptions transaction_options;
    transaction_options.collection_root = options.collection_root;
    transaction_options.target_relative_directory =
        std::filesystem::path("segments") / options.segment_id;
    transaction_options.transaction_id =
        "disk_vamana_" + options.segment_id + "_g" + std::to_string(options.segment_generation);
    transaction_options.manifest_v2_writer = options.collection_features.manifest_v2_writer;
    transaction_options.abort_policy = options.abort_policy;
    transaction_options.fail_point = options.fail_point;
    auto begun =
        internal::collection::ArtifactControlPlaneTransaction::begin(std::move(transaction_options),
                                                                     context);
    if (!begun.ok()) {
      return begun.status();
    }
    auto transaction = std::move(begun).value();

    try {
      VamanaSegmentBuilder builder(input.vectors.dim, core::Metric::l2, build_params);
      if (input.vectors.row_stride ==
          static_cast<std::uint64_t>(input.vectors.dim) * sizeof(float)) {
        builder.add_batch(input.vectors.row<float>(0),
                          input.logical_ids.data(),
                          input.vectors.rows);
      } else {
        for (core::RowCount row = 0; row < input.vectors.rows; ++row) {
          builder.add_batch(input.vectors.row<float>(row), input.logical_ids.data() + row, 1);
        }
      }
      (void)builder.finish(transaction->staging_payload_directory());
    } catch (const std::invalid_argument &error) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::build,
                                 core::StatusDetail::engine_exception,
                                 error.what());
    } catch (const std::bad_alloc &error) {
      return core::Status::error(core::StatusCode::resource_exhausted,
                                 core::OperationStage::build,
                                 core::StatusDetail::allocation_failure,
                                 error.what(),
                                 core::Retryability::retryable_with_backoff);
    } catch (const std::exception &error) {
      return core::Status::error(core::StatusCode::io_error,
                                 core::OperationStage::build,
                                 core::StatusDetail::engine_exception,
                                 error.what());
    }

    status = transaction->adopt(native_artifact_specs());
    if (!status.ok()) {
      return status;
    }
    auto prepared = transaction->prepare(make_segment_entry(options));
    if (!prepared.ok()) {
      return prepared.status();
    }
    auto manifest = make_collection_manifest(input.vectors.dim, options);
    if (!manifest.ok()) {
      return manifest.status();
    }
    status = transaction->publish(std::move(manifest).value());
    if (!status.ok()) {
      return status;
    }

    try {
      const auto final_path = transaction->final_payload_directory();
      auto searcher = std::make_shared<VamanaSegmentSearcher>(final_path);
      return std::unique_ptr<DiskVamanaSegment>(
          new DiskVamanaSegment(std::move(searcher), final_path));
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

  [[nodiscard]] static auto build(DiskVamanaBuildInput input,
                                  core::Metric metric,
                                  const DiskVamanaPublicationOptions &options,
                                  core::BuildContext &context)
      -> core::Result<std::unique_ptr<DiskVamanaSegment>> {
    return build(std::move(input), metric, VamanaSegmentBuildParams{}, options, context);
  }

  [[nodiscard]] static auto open(core::ArtifactView artifact,
                                 const core::OpenOptions &,
                                 core::OpenContext &context)
      -> core::Result<std::unique_ptr<DiskVamanaSegment>> {
    auto control = core::validate_runtime_control(context.deadline,
                                                  context.cancellation,
                                                  core::OperationStage::open);
    if (!control.ok()) {
      return control;
    }

    const auto manifest_path = artifact.find(kManifestArtifactName);
    if (manifest_path.empty()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 "DiskVamana open requires a manifest artifact");
    }

    try {
      const auto directory = std::filesystem::path(manifest_path).parent_path();
      const auto native = SegmentManifest::load(std::filesystem::path(manifest_path));
      if (native.index_type != DiskIndexType::Vamana) {
        return core::Status::error(core::StatusCode::not_supported,
                                   core::OperationStage::open,
                                   core::StatusDetail::operation_slot_absent,
                                   "DiskVamana open received a non-Vamana native manifest");
      }
      if (native.metric != core::Metric::l2) {
        return l2_gate_status(core::OperationStage::open, native.metric);
      }

      const auto ids_path = artifact.find(kIdsArtifactName);
      const auto vectors_path = artifact.find(kVectorsArtifactName);
      const auto graph_path = artifact.find(kGraphArtifactName);
      if (ids_path.empty() || vectors_path.empty() || graph_path.empty()) {
        return core::Status::
            error(core::StatusCode::invalid_argument,
                  core::OperationStage::open,
                  core::StatusDetail::malformed_struct,
                  "DiskVamana open requires manifest, ids, vectors, and graph artifacts");
      }
      const auto graph_it = native.x_extras.find("x_graph_file");
      if (graph_it == native.x_extras.end()) {
        return core::Status::error(core::StatusCode::corruption,
                                   core::OperationStage::open,
                                   core::StatusDetail::malformed_struct,
                                   "DiskVamana native manifest has no x_graph_file");
      }
      if (std::filesystem::path(ids_path).lexically_normal() !=
              (directory / native.ids_file).lexically_normal() ||
          std::filesystem::path(vectors_path).lexically_normal() !=
              (directory / native.vectors_file).lexically_normal() ||
          std::filesystem::path(graph_path).lexically_normal() !=
              (directory / graph_it->second).lexically_normal()) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::open,
                                   core::StatusDetail::malformed_struct,
                                   "DiskVamana ArtifactView paths disagree with native manifest");
      }

      std::uint64_t bytes{};
      const std::array paths{std::filesystem::path(manifest_path),
                             std::filesystem::path(ids_path),
                             std::filesystem::path(vectors_path),
                             std::filesystem::path(graph_path)};
      for (const auto &path : paths) {
        const auto size = std::filesystem::file_size(path);
        if (size > std::numeric_limits<std::uint64_t>::max() - bytes) {
          return core::Status::error(core::StatusCode::invalid_argument,
                                     core::OperationStage::open,
                                     core::StatusDetail::arithmetic_overflow,
                                     "DiskVamana artifact bytes overflow uint64");
        }
        bytes += static_cast<std::uint64_t>(size);
      }
      auto budget = core::require_lease(context.resident_lease,
                                        bytes,
                                        core::OperationStage::open,
                                        "DiskVamana resident lease is too small for artifacts");
      if (!budget.ok()) {
        return budget;
      }
      budget = require_io_credits(context.io_credits,
                                  paths.size(),
                                  bytes,
                                  core::OperationStage::open,
                                  "DiskVamana open I/O credits are too small");
      if (!budget.ok()) {
        return budget;
      }

      auto searcher = std::make_shared<VamanaSegmentSearcher>(directory);
      return std::unique_ptr<DiskVamanaSegment>(
          new DiskVamanaSegment(std::move(searcher), directory));
    } catch (const std::invalid_argument &error) {
      return core::Status::error(core::StatusCode::corruption,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 error.what());
    } catch (const std::bad_alloc &error) {
      return core::Status::error(core::StatusCode::resource_exhausted,
                                 core::OperationStage::open,
                                 core::StatusDetail::allocation_failure,
                                 error.what());
    } catch (const std::exception &error) {
      return core::Status::error(core::StatusCode::io_error,
                                 core::OperationStage::open,
                                 core::StatusDetail::engine_exception,
                                 error.what());
    }
  }

  [[nodiscard]] static auto open_directory(const std::filesystem::path &segment_directory,
                                           const core::OpenOptions &options,
                                           core::OpenContext &context)
      -> core::Result<std::unique_ptr<DiskVamanaSegment>> {
    try {
      const auto native = SegmentManifest::load(segment_directory / "manifest.txt");
      if (native.metric != core::Metric::l2) {
        return l2_gate_status(core::OperationStage::open, native.metric);
      }
      const auto graph_it = native.x_extras.find("x_graph_file");
      if (graph_it == native.x_extras.end()) {
        return core::Status::error(core::StatusCode::corruption,
                                   core::OperationStage::open,
                                   core::StatusDetail::malformed_struct,
                                   "DiskVamana native manifest has no x_graph_file");
      }
      std::array<std::string, 4> paths{(segment_directory / "manifest.txt").string(),
                                       (segment_directory / native.ids_file).string(),
                                       (segment_directory / native.vectors_file).string(),
                                       (segment_directory / graph_it->second).string()};
      const std::array<core::ArtifactLocation, 4>
          locations{core::ArtifactLocation(kManifestArtifactName, paths[0]),
                    core::ArtifactLocation(kIdsArtifactName, paths[1]),
                    core::ArtifactLocation(kVectorsArtifactName, paths[2]),
                    core::ArtifactLocation(kGraphArtifactName, paths[3])};
      return open(core::ArtifactView(locations), options, context);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

  [[nodiscard]] static auto open_collection(
      const std::filesystem::path &collection_root,
      std::string_view segment_id,
      const core::OpenOptions &open_options,
      core::OpenContext &context,
      const internal::collection::ManifestReaderOptions &reader_options = {})
      -> core::Result<std::unique_ptr<DiskVamanaSegment>> {
    auto opened =
        internal::collection::CollectionManifestDualReader::open(collection_root, reader_options);
    if (!opened.ok()) {
      return opened.status();
    }
    if (opened.value().manifest.collection.metric != core::Metric::l2) {
      return l2_gate_status(core::OperationStage::open,
                            legacy_metric(opened.value().manifest.collection.metric));
    }
    const auto &segments = opened.value().manifest.segments;
    const auto found = std::find_if(segments.begin(), segments.end(), [&](const auto &entry) {
      return entry.segment_id == segment_id;
    });
    if (found == segments.end()) {
      return core::Status::error(core::StatusCode::not_found,
                                 core::OperationStage::open,
                                 core::StatusDetail::none,
                                 "DiskVamana segment is absent from the collection manifest");
    }
    if (found->algorithm_id != kAlgorithmId) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::open,
                                 core::StatusDetail::operation_slot_absent,
                                 "requested collection segment is not DiskVamana");
    }

    std::array<std::string, 4> paths{};
    for (const auto &artifact : found->artifacts) {
      if (artifact.logical_name == kManifestArtifactName) {
        paths[0] = (collection_root / artifact.relative_path).string();
      } else if (artifact.logical_name == kIdsArtifactName) {
        paths[1] = (collection_root / artifact.relative_path).string();
      } else if (artifact.logical_name == kVectorsArtifactName) {
        paths[2] = (collection_root / artifact.relative_path).string();
      } else if (artifact.logical_name == kGraphArtifactName ||
                 artifact.logical_name == "x_graph_file") {
        paths[3] = (collection_root / artifact.relative_path).string();
      }
    }
    const std::array<core::ArtifactLocation, 4>
        locations{core::ArtifactLocation(kManifestArtifactName, paths[0]),
                  core::ArtifactLocation(kIdsArtifactName, paths[1]),
                  core::ArtifactLocation(kVectorsArtifactName, paths[2]),
                  core::ArtifactLocation(kGraphArtifactName, paths[3])};
    return open(core::ArtifactView(locations), open_options, context);
  }

  [[nodiscard]] auto descriptor() const noexcept -> core::Descriptor {
    core::Descriptor descriptor;
    descriptor.algorithm_id = kAlgorithmId;
    descriptor.format_version = kFormatVersion;
    descriptor.factory_version = 1;
    descriptor.dim = searcher_->dim();
    descriptor.metric = core::Metric::l2;
    descriptor.stored_scalar_type = core::ScalarType::float32;
    descriptor.medium = core::Medium::disk;
    descriptor.preprocessing = core::MetricPreprocessing::none;
    descriptor.engine_factory_id = kAlgorithmId;
    return descriptor;
  }

  [[nodiscard]] static auto make_search_extension(const DiskVamanaSearchExtension &options)
      -> core::AlgorithmSearchExtension {
    return make_disk_vamana_search_extension(options);
  }

  [[nodiscard]] auto search(const core::SearchRequest &request) const -> core::Status {
    if (request.queries.rows != 1) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "DiskVamana single search requires exactly one query row");
    }
    return execute_search(request);
  }

  [[nodiscard]] auto batch_search(const core::SearchRequest &request) const -> core::Status {
    return execute_search(request);
  }

  [[nodiscard]] auto save(core::ArtifactWriter &writer,
                          const core::SaveOptions &,
                          core::ArtifactManifest &manifest) const -> core::Status {
    const auto manifest_target = writer.find(kManifestArtifactName);
    const auto ids_target = writer.find(kIdsArtifactName);
    const auto vectors_target = writer.find(kVectorsArtifactName);
    const auto graph_target = writer.find(kGraphArtifactName);
    if (manifest_target.empty() || ids_target.empty() || vectors_target.empty() ||
        graph_target.empty()) {
      return core::Status::
          error(core::StatusCode::invalid_argument,
                core::OperationStage::save,
                core::StatusDetail::malformed_struct,
                "DiskVamana save requires manifest, ids, vectors, and graph logical names");
    }

    try {
      const auto &native = searcher_->manifest();
      const auto graph_it = native.x_extras.find("x_graph_file");
      if (graph_it == native.x_extras.end()) {
        return core::Status::error(core::StatusCode::corruption,
                                   core::OperationStage::save,
                                   core::StatusDetail::malformed_struct,
                                   "DiskVamana native manifest has no x_graph_file");
      }
      const std::array sources{directory_ / "manifest.txt",
                               directory_ / native.ids_file,
                               directory_ / native.vectors_file,
                               directory_ / graph_it->second};
      const std::array targets{std::filesystem::path(manifest_target),
                               std::filesystem::path(ids_target),
                               std::filesystem::path(vectors_target),
                               std::filesystem::path(graph_target)};
      const std::array names{kManifestArtifactName,
                             kIdsArtifactName,
                             kVectorsArtifactName,
                             kGraphArtifactName};
      for (std::size_t index = 0; index < sources.size(); ++index) {
        if (std::filesystem::absolute(sources[index]).lexically_normal() ==
            std::filesystem::absolute(targets[index]).lexically_normal()) {
          return core::Status::error(core::StatusCode::conflict,
                                     core::OperationStage::save,
                                     core::StatusDetail::already_exists,
                                     "DiskVamana save destination aliases its source artifact");
        }
        std::filesystem::copy_file(sources[index],
                                   targets[index],
                                   std::filesystem::copy_options::overwrite_existing);
        platform::sync_file_or_throw(targets[index]);
        saved_artifacts_[index] =
            core::Artifact(names[index],
                           static_cast<std::uint64_t>(std::filesystem::file_size(targets[index])),
                           0);
      }
      manifest = core::ArtifactManifest{};
      manifest.schema_version = 1;
      manifest.format_version = kFormatVersion;
      manifest.algorithm_id = kAlgorithmId;
      manifest.artifacts = saved_artifacts_;
      return core::Status::success();
    } catch (const std::invalid_argument &error) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::save,
                                 core::StatusDetail::engine_exception,
                                 error.what());
    } catch (const std::exception &error) {
      return core::Status::error(core::StatusCode::io_error,
                                 core::OperationStage::save,
                                 core::StatusDetail::engine_exception,
                                 error.what());
    }
  }

  [[nodiscard]] auto save_transactional(const DiskVamanaPublicationOptions &options,
                                        core::BuildContext &context) const
      -> core::Result<std::filesystem::path> {
    if (options.segment_id != searcher_->manifest().segment_id) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::save,
                                 core::StatusDetail::malformed_struct,
                                 "byte-preserving DiskVamana save retains its native segment id");
    }

    internal::collection::ArtifactTransactionOptions transaction_options;
    transaction_options.collection_root = options.collection_root;
    transaction_options.target_relative_directory =
        std::filesystem::path("segments") / options.segment_id;
    transaction_options.transaction_id = "disk_vamana_save_" + options.segment_id + "_g" +
                                         std::to_string(options.segment_generation);
    transaction_options.manifest_v2_writer = options.collection_features.manifest_v2_writer;
    transaction_options.abort_policy = options.abort_policy;
    transaction_options.fail_point = options.fail_point;
    auto begun =
        internal::collection::ArtifactControlPlaneTransaction::begin(std::move(transaction_options),
                                                                     context);
    if (!begun.ok()) {
      return begun.status();
    }
    auto transaction = std::move(begun).value();
    auto writer = transaction->writer(native_artifact_specs());
    if (!writer.ok()) {
      return writer.status();
    }
    core::ArtifactManifest native_manifest;
    auto status = save(writer.value(), core::SaveOptions{}, native_manifest);
    if (!status.ok()) {
      return status;
    }
    auto prepared = transaction->prepare(make_segment_entry(options));
    if (!prepared.ok()) {
      return prepared.status();
    }
    auto collection_manifest = make_collection_manifest(searcher_->dim(), options);
    if (!collection_manifest.ok()) {
      return collection_manifest.status();
    }
    status = transaction->publish(std::move(collection_manifest).value());
    if (!status.ok()) {
      return status;
    }
    return transaction->final_payload_directory();
  }

  [[nodiscard]] auto stats(core::SegmentStats &stats) const noexcept -> core::Status {
    stats = core::SegmentStats{};
    stats.snapshot_version = 1;
    stats.live_rows = searcher_->size();
    stats.allocated_rows = searcher_->size();
    stats.resident_bytes = artifact_bytes_;
    stats.health = core::SegmentHealth::healthy;
    return core::Status::success();
  }

  [[nodiscard]] static auto into_any(std::unique_ptr<DiskVamanaSegment> segment)
      -> core::Result<core::AnySegment> {
    if (segment == nullptr) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::admission,
                                 core::StatusDetail::null_data,
                                 "cannot erase a null DiskVamana segment");
    }
    auto shared = std::shared_ptr<DiskVamanaSegment>(std::move(segment));
    core::SegmentInstanceConfig config;
    config.readonly = true;
    config.concurrency.reentrant_search = true;
    config.concurrency.search_with_stage = false;
    config.concurrency.search_with_publish = false;
    config.concurrency.serial_mutation = true;
    config.concurrency.native_async = false;
    config.concurrency.cooperative_cancel = true;
    config.concurrency.explicit_drain = false;
    return core::AnySegment::from_sync(std::move(shared), std::move(config));
  }

 private:
  DiskVamanaSegment(std::shared_ptr<VamanaSegmentSearcher> searcher,
                    std::filesystem::path directory)
      : searcher_(std::move(searcher)), directory_(std::move(directory)) {
    const auto &native = searcher_->manifest();
    const auto graph_it = native.x_extras.find("x_graph_file");
    if (graph_it == native.x_extras.end()) {
      throw std::runtime_error("DiskVamana native manifest has no x_graph_file");
    }
    for (const auto &path : {directory_ / "manifest.txt",
                             directory_ / native.ids_file,
                             directory_ / native.vectors_file,
                             directory_ / graph_it->second}) {
      const auto bytes = static_cast<std::uint64_t>(std::filesystem::file_size(path));
      artifact_bytes_ = bytes > std::numeric_limits<std::uint64_t>::max() - artifact_bytes_
                            ? std::numeric_limits<std::uint64_t>::max()
                            : artifact_bytes_ + bytes;
    }
  }

  [[nodiscard]] static auto native_artifact_specs()
      -> std::vector<internal::collection::LogicalArtifactSpec> {
    return {{std::string(kManifestArtifactName), "manifest.txt", true, {}},
            {std::string(kIdsArtifactName), "ids.u64.bin", true, {}},
            {std::string(kVectorsArtifactName), "vectors.f32.bin", true, {}},
            {std::string(kGraphArtifactName), "graph.index", true, {}}};
  }

  [[nodiscard]] static auto metric_name(core::Metric metric) -> std::string_view {
    return core::metric_to_string(metric);
  }

  [[nodiscard]] static auto l2_gate_status(core::OperationStage stage, core::Metric metric)
      -> core::Status {
    return core::Status::error(core::StatusCode::not_supported,
                               stage,
                               core::StatusDetail::operation_slot_absent,
                               "DiskVamanaSegment first version supports L2 only; non-L2 metric '" +
                                   std::string(metric_name(metric)) + "' is not supported");
  }

  [[nodiscard]] static auto legacy_metric(core::Metric metric) noexcept -> core::Metric {
    return metric;
  }

  [[nodiscard]] static auto require_io_credits(const core::IoCredits &credits,
                                               std::uint64_t requests,
                                               std::uint64_t bytes,
                                               core::OperationStage stage,
                                               const char *diagnostic) -> core::Status {
    if ((credits.available_requests != core::kUnlimitedResource &&
         requests > credits.available_requests) ||
        (credits.available_bytes != core::kUnlimitedResource && bytes > credits.available_bytes)) {
      return core::Status::error(core::StatusCode::resource_exhausted,
                                 stage,
                                 core::StatusDetail::budget_denied,
                                 diagnostic,
                                 core::Retryability::retryable_with_backoff);
    }
    return core::Status::success();
  }

  [[nodiscard]] static auto validate_build_input(const DiskVamanaBuildInput &input,
                                                 core::Metric metric,
                                                 const VamanaSegmentBuildParams &build_params,
                                                 const DiskVamanaPublicationOptions &options,
                                                 core::BuildContext &context) -> core::Status {
    if (metric != core::Metric::l2) {
      return l2_gate_status(core::OperationStage::build, legacy_metric(metric));
    }
    if (!core::is_current_struct(input) || input.vectors.dim == 0) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::build,
                                 core::StatusDetail::malformed_struct,
                                 "DiskVamana build input is incompatible or zero-dimensional");
    }
    auto status =
        core::validate_tensor(input.vectors, input.vectors.dim, core::OperationStage::build);
    if (!status.ok()) {
      return status;
    }
    if (input.vectors.scalar_type != core::ScalarType::float32) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::build,
                                 core::StatusDetail::unsupported_scalar_type,
                                 "DiskVamana stores float32 tensors without implicit conversion");
    }
    if (input.vectors.rows < 2 || input.vectors.rows > std::numeric_limits<std::uint32_t>::max() ||
        input.logical_ids.size() != input.vectors.rows || options.collection_root.empty() ||
        !detail::is_valid_segment_id(options.segment_id) || options.segment_generation == 0 ||
        options.manifest_generation == 0) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::build,
                                 core::StatusDetail::malformed_struct,
                                 "DiskVamana build rows/ids/publication options are invalid");
    }
    if (build_params.R == 0 || build_params.L < build_params.R ||
        !std::isfinite(build_params.alpha) || build_params.alpha < 1.0F) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::build,
                                 core::StatusDetail::malformed_struct,
                                 "DiskVamana build requires R>0, L>=R, and finite alpha>=1");
    }
    status = core::validate_runtime_control(context.deadline,
                                            context.cancellation,
                                            core::OperationStage::build);
    if (!status.ok()) {
      return status;
    }

    std::uint64_t components{};
    std::uint64_t vector_bytes{};
    std::uint64_t id_bytes{};
    std::uint64_t graph_slots{};
    std::uint64_t graph_bytes{};
    std::uint64_t bytes{};
    if (!core::checked_multiply(input.vectors.rows, input.vectors.dim, components) ||
        !core::checked_multiply(components, sizeof(float), vector_bytes) ||
        !core::checked_multiply(input.vectors.rows, sizeof(std::uint64_t), id_bytes) ||
        !core::checked_multiply(input.vectors.rows,
                                static_cast<std::uint64_t>(build_params.R) + 1,
                                graph_slots) ||
        !core::checked_multiply(graph_slots, sizeof(std::uint32_t), graph_bytes) ||
        !core::checked_add(vector_bytes, id_bytes, bytes) ||
        !core::checked_add(bytes, graph_bytes, bytes)) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::build,
                                 core::StatusDetail::arithmetic_overflow,
                                 "DiskVamana build artifact estimate overflows uint64");
    }
    return context.growing_reservation.ensure(bytes,
                                              core::OperationStage::build,
                                              "DiskVamana build reservation is too small");
  }

  [[nodiscard]] static auto make_segment_entry(const DiskVamanaPublicationOptions &options)
      -> internal::collection::SegmentEntryV2 {
    internal::collection::SegmentEntryV2 entry;
    entry.segment_id = options.segment_id;
    entry.generation = options.segment_generation;
    entry.role = internal::collection::SegmentRoleV2::searchable;
    entry.algorithm_id = kAlgorithmId;
    entry.format_version = kFormatVersion;
    entry.factory_key = "vamana";
    entry.capabilities.operations = core::capability_bit(core::OperationCapability::search) |
                                    core::capability_bit(core::OperationCapability::batch_search) |
                                    core::capability_bit(core::OperationCapability::save) |
                                    core::capability_bit(core::OperationCapability::stats);
    entry.capabilities.reentrant_search = true;
    entry.capabilities.cooperative_cancel = true;
    entry.capabilities.explicit_drain = false;
    entry.lifecycle = internal::collection::SegmentLifecycleV2::sealed;
    entry.wal_cut = options.wal_cut;
    entry.row_versions = options.row_versions;
    entry.id_map_checkpoint = options.id_map_checkpoint;
    entry.reader_compatibility.required_features = {"disk_vamana_segment"};
    entry.extensions.emplace("l2_only", "true");
    return entry;
  }

  [[nodiscard]] static auto make_collection_manifest(std::uint32_t dim,
                                                     const DiskVamanaPublicationOptions &options)
      -> core::Result<internal::collection::ArtifactManifestV2> {
    auto manifest = options.base_manifest.value_or(internal::collection::ArtifactManifestV2{});
    if (options.base_manifest.has_value() &&
        (manifest.collection.dim != dim || manifest.collection.metric != core::Metric::l2 ||
         manifest.collection.scalar_type != core::ScalarType::float32)) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::build,
                                 core::StatusDetail::malformed_struct,
                                 "DiskVamana publication disagrees with base collection schema");
    }
    manifest.collection.dim = dim;
    manifest.collection.metric = core::Metric::l2;
    manifest.collection.scalar_type = core::ScalarType::float32;
    manifest.collection.logical_id_encoding =
        internal::collection::LogicalIdEncodingV2::legacy_u64_le;
    manifest.collection.metadata_epoch = options.metadata_epoch;
    manifest.collection.metadata_checkpoint = options.metadata_checkpoint;
    manifest.publication.generation = options.manifest_generation;
    manifest.publication.parent = options.publication_parent;
    manifest.wal_cut = options.wal_cut;
    manifest.row_versions = options.row_versions;
    manifest.id_map_checkpoint = options.id_map_checkpoint;
    const auto numeric_id = static_cast<std::uint64_t>(std::stoull(options.segment_id.substr(4)));
    manifest.next_segment_id_hint =
        std::max(manifest.next_segment_id_hint, numeric_id + std::uint64_t{1});
    return manifest;
  }

  [[nodiscard]] auto resolve_effort(const core::SearchOptions &options) const
      -> core::Result<std::uint32_t> {
    std::uint32_t effort = 100;
    for (const auto &extension : options.extensions) {
      if (extension.algorithm_id != kAlgorithmId) {
        if (extension.unknown_policy == core::UnknownExtensionPolicy::reject) {
          return core::Status::
              error(core::StatusCode::invalid_argument,
                    core::OperationStage::validation,
                    core::StatusDetail::unknown_extension,
                    "DiskVamana received a search extension for another algorithm");
        }
        continue;
      }
      if (extension.payload == nullptr ||
          extension.payload_size < sizeof(DiskVamanaSearchExtension)) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::malformed_struct,
                                   "DiskVamana search extension payload is truncated");
      }
      const auto &typed = *static_cast<const DiskVamanaSearchExtension *>(extension.payload);
      if (!core::is_current_struct(typed)) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::malformed_struct,
                                   "DiskVamana search extension has an incompatible version");
      }
      effort = typed.effort;
    }
    if (effort == 0) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "DiskVamana search effort must be greater than zero");
    }
    return effort;
  }

  [[nodiscard]] auto validate_search_request(const core::SearchRequest &request) const
      -> core::Status {
    if (!core::is_current_struct(request) || !core::is_current_struct(request.options) ||
        request.context == nullptr || request.response == nullptr) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "DiskVamana search request is incomplete or incompatible");
    }
    auto status =
        core::validate_tensor(request.queries, searcher_->dim(), core::OperationStage::validation);
    if (!status.ok()) {
      return status;
    }
    if (request.queries.scalar_type != core::ScalarType::float32) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::validation,
                                 core::StatusDetail::unsupported_scalar_type,
                                 "DiskVamana search accepts float32 tensors only");
    }
    status = core::validate_response(*request.response,
                                     request.queries.rows,
                                     request.options.top_k,
                                     core::OperationStage::validation);
    if (!status.ok()) {
      return status;
    }
    if (request.filter.kind != core::SegmentFilterKind::none) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::validation,
                                 core::StatusDetail::operation_slot_absent,
                                 "DiskVamana has no engine-local metadata filter");
    }
    if (request.options.top_k > std::numeric_limits<std::uint32_t>::max()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "DiskVamana top_k exceeds uint32");
    }
    auto effort = resolve_effort(request.options);
    if (!effort.ok()) {
      return effort.status();
    }
    status = core::validate_runtime_control(request.context->deadline,
                                            request.context->cancellation,
                                            core::OperationStage::search);
    if (!status.ok()) {
      return status;
    }

    std::uint64_t visited_bytes{};
    std::uint64_t pool_bytes{};
    std::uint64_t scratch_bytes{};
    if (!core::checked_multiply(searcher_->size(),
                                sizeof(std::uint8_t) + sizeof(std::uint32_t),
                                visited_bytes) ||
        !core::checked_multiply(searcher_->size(), sizeof(vamana::GreedyHit), pool_bytes) ||
        !core::checked_add(visited_bytes, pool_bytes, scratch_bytes)) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "DiskVamana query scratch size overflows uint64");
    }
    status = core::require_lease(request.context->query_scratch_lease,
                                 scratch_bytes,
                                 core::OperationStage::search,
                                 "DiskVamana query scratch lease is too small");
    if (!status.ok()) {
      return status;
    }

    std::uint64_t vector_bytes{};
    std::uint64_t request_bytes{};
    if (!core::checked_multiply(searcher_->size(), searcher_->dim(), vector_bytes) ||
        !core::checked_multiply(vector_bytes, sizeof(float), vector_bytes) ||
        !core::checked_multiply(vector_bytes, request.queries.rows, request_bytes)) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "DiskVamana search I/O accounting overflows uint64");
    }
    return require_io_credits(request.context->io_credits,
                              request.queries.rows,
                              request_bytes,
                              core::OperationStage::search,
                              "DiskVamana search I/O credits are too small");
  }

  [[nodiscard]] auto execute_search(const core::SearchRequest &request) const -> core::Status {
    auto status = validate_search_request(request);
    if (!status.ok()) {
      return status;
    }
    auto &response = *request.response;
    response.score_kind = core::ScoreKind::distance;
    response.comparable_metric = core::Metric::l2;
    response.result_flags = core::ResultFlag::approximate;
    if (request.options.top_k == 0 || request.queries.rows == 0) {
      core::initialize_empty_response(response,
                                      request.queries.rows,
                                      request.options.top_k == 0
                                          ? core::SearchCompleteness::complete_k
                                          : core::SearchCompleteness::eligible_exhausted);
      return core::Status::success();
    }
    auto effort_result = resolve_effort(request.options);
    if (!effort_result.ok()) {
      return effort_result.status();
    }
    const auto requested_top_k = static_cast<std::uint32_t>(request.options.top_k);
    DiskSearchOptions options;
    options.top_k = requested_top_k;
    options.ef = std::move(effort_result).value();
    options.exact_rerank = false;

    response.query_count = request.queries.rows;
    response.offsets[0] = 0;
    core::RowCount cursor{};
    for (core::RowCount row = 0; row < request.queries.rows; ++row) {
      const auto control = core::validate_runtime_control(request.context->deadline,
                                                          request.context->cancellation,
                                                          core::OperationStage::search);
      if (!control.ok()) {
        for (core::RowCount remaining = row; remaining < request.queries.rows; ++remaining) {
          response.offsets[remaining + 1] = cursor;
          response.valid_counts[remaining] = 0;
          response.statuses[remaining] = control;
          response.completeness[remaining] = core::SearchCompleteness::failed;
        }
        return request.queries.rows == 1 ? control : core::Status::success();
      }
      try {
        const auto hits = searcher_->search(request.queries.row<float>(row), options);
        for (std::size_t index = 0; index < hits.size(); ++index) {
          if (std::isnan(hits[index].distance)) {
            throw std::runtime_error("DiskVamana search produced a NaN numeric score");
          }
          response.hits[static_cast<std::size_t>(cursor + index)] =
              core::SearchHit(core::SegmentRowId(hits[index].label),
                              hits[index].distance,
                              core::ScoreKind::distance,
                              core::Metric::l2,
                              core::ResultFlag::approximate);
        }
        const auto written = static_cast<core::RowCount>(hits.size());
        cursor += written;
        response.offsets[row + 1] = cursor;
        response.valid_counts[row] = written;
        response.statuses[row] = core::Status::success();
        if (written == requested_top_k) {
          response.completeness[row] = core::SearchCompleteness::complete_k;
        } else if (requested_top_k > searcher_->size() && written == searcher_->size()) {
          response.completeness[row] = core::SearchCompleteness::eligible_exhausted;
        } else {
          response.completeness[row] = core::SearchCompleteness::strategy_incomplete;
        }
      } catch (...) {
        const auto failure = core::status_from_exception(core::OperationStage::search);
        response.offsets[row + 1] = cursor;
        response.valid_counts[row] = 0;
        response.statuses[row] = failure;
        response.completeness[row] = core::SearchCompleteness::failed;
        if (request.queries.rows == 1) {
          return failure;
        }
      }
    }
    if (request.context->stats != nullptr) {
      const auto visited_per_query =
          std::min<std::uint64_t>(searcher_->size(),
                                  std::max<std::uint64_t>(options.ef, options.top_k));
      request.context->stats->visited += visited_per_query * request.queries.rows;
      request.context->stats->io_requests += request.queries.rows;
      request.context->stats->io_bytes +=
          searcher_->size() * searcher_->dim() * sizeof(float) * request.queries.rows;
    }
    return core::Status::success();
  }

  std::shared_ptr<VamanaSegmentSearcher> searcher_{};
  std::filesystem::path directory_{};
  std::uint64_t artifact_bytes_{};
  mutable std::array<core::Artifact, 4> saved_artifacts_{};
};

class DiskVamanaSegmentFactory {
 public:
  static constexpr auto registration = internal::disk::kDiskVamanaRegistration;

  [[nodiscard]] static auto build(
      DiskVamanaBuildInput input,
      core::Metric metric,
      const VamanaSegmentBuildParams &build_params,
      const DiskVamanaPublicationOptions &options,
      core::BuildContext &context,
      const internal::disk::DiskEngineFeatureFlags &features = {}) noexcept
      -> core::Result<std::unique_ptr<DiskVamanaSegment>> {
    if (!features.enabled(registration.feature)) {
      return disabled(core::OperationStage::build);
    }
    try {
      return DiskVamanaSegment::build(std::move(input), metric, build_params, options, context);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::build);
    }
  }

  [[nodiscard]] static auto build(
      DiskVamanaBuildInput input,
      core::Metric metric,
      const DiskVamanaPublicationOptions &options,
      core::BuildContext &context,
      const internal::disk::DiskEngineFeatureFlags &features = {}) noexcept
      -> core::Result<std::unique_ptr<DiskVamanaSegment>> {
    return build(std::move(input), metric, VamanaSegmentBuildParams{}, options, context, features);
  }

  [[nodiscard]] static auto open(
      core::ArtifactView artifacts,
      const core::OpenOptions &options,
      core::OpenContext &context,
      const internal::disk::DiskEngineFeatureFlags &features = {}) noexcept
      -> core::Result<std::unique_ptr<DiskVamanaSegment>> {
    if (!features.enabled(registration.feature)) {
      return disabled(core::OperationStage::open);
    }
    try {
      return DiskVamanaSegment::open(artifacts, options, context);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

 private:
  [[nodiscard]] static auto disabled(core::OperationStage stage) -> core::Status {
    return core::Status::
        error(core::StatusCode::not_supported,
              stage,
              core::StatusDetail::operation_slot_absent,
              "DiskVamanaSegment factory is disabled; DiskCollection v1 is unchanged");
  }
};
static_assert(core::Searchable<DiskVamanaSegment>);
static_assert(core::BatchSearchable<DiskVamanaSegment>);
static_assert(core::Saveable<DiskVamanaSegment>);
static_assert(core::StatsProvider<DiskVamanaSegment>);
static_assert(!core::Exportable<DiskVamanaSegment>);
static_assert(!core::Mutable<DiskVamanaSegment>);

}  // namespace alaya::disk
