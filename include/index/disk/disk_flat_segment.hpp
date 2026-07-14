// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
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
#include "index/disk/disk_flat_builder.hpp"
#include "index/disk/disk_flat_searcher.hpp"
#include "platform/fs.hpp"

namespace alaya::disk {

struct DiskFlatBuildInput {
  core::VersionedStructHeader header{};
  core::TypedTensorView vectors{};
  std::span<const std::uint64_t> logical_ids{};
  std::uint64_t reserved[4]{};

  DiskFlatBuildInput() : header(core::current_struct_header<DiskFlatBuildInput>()) {}
  DiskFlatBuildInput(core::TypedTensorView vector_view, std::span<const std::uint64_t> ids)
      : header(core::current_struct_header<DiskFlatBuildInput>()),
        vectors(vector_view),
        logical_ids(ids) {}
};

struct DiskFlatPublicationOptions {
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

struct DiskFlatExportBatch {
  std::uint64_t row_offset{};
  std::span<const std::uint64_t> logical_ids{};
  core::TypedTensorView vectors{};
  std::span<const std::string_view> metadata_references{};
  bool done{};
};

class DiskFlatExportState;

struct DiskFlatExportRequest {
  core::VersionedStructHeader header{};
  std::uint64_t batch_rows{1024};
  std::shared_ptr<DiskFlatExportState> *lifetime_owner{};
  std::uint64_t reserved[4]{};

  DiskFlatExportRequest() : header(core::current_struct_header<DiskFlatExportRequest>()) {}
};

class DiskFlatExportState {
 public:
  DiskFlatExportState(const DiskFlatExportState &) = delete;
  auto operator=(const DiskFlatExportState &) -> DiskFlatExportState & = delete;

  [[nodiscard]] auto next(DiskFlatExportBatch &batch) noexcept -> core::Status {
    std::lock_guard lock(mutex_);
    batch = DiskFlatExportBatch{};
    if (searcher_ == nullptr) {
      return core::Status::error(core::StatusCode::closed,
                                 core::OperationStage::export_rows,
                                 core::StatusDetail::none,
                                 "DiskFlat export cursor has no live source");
    }
    const auto count = searcher_->size();
    const auto begin = offset_;
    const auto rows = std::min<std::uint64_t>(batch_rows_, count - begin);
    batch.row_offset = begin;
    batch.logical_ids = std::span(searcher_->labels() + begin, static_cast<std::size_t>(rows));
    batch.vectors =
        core::TypedTensorView::contiguous(searcher_->vectors() + begin * searcher_->dim(),
                                          rows,
                                          searcher_->dim());
    metadata_references_.assign(static_cast<std::size_t>(rows), std::string_view{});
    batch.metadata_references = metadata_references_;
    offset_ += rows;
    batch.done = offset_ == count;
    return core::Status::success();
  }

 private:
  friend class DiskFlatSegment;

  DiskFlatExportState(std::shared_ptr<const DiskFlatSegmentSearcher> searcher,
                      std::uint64_t batch_rows)
      : searcher_(std::move(searcher)), batch_rows_(batch_rows) {}

  std::shared_ptr<const DiskFlatSegmentSearcher> searcher_{};
  std::uint64_t batch_rows_{};
  std::uint64_t offset_{};
  std::vector<std::string_view> metadata_references_{};
  std::mutex mutex_{};
};

class DiskFlatSegment {
 public:
  static constexpr std::uint32_t kFormatVersion = 1;
  static constexpr std::string_view kManifestArtifactName{"manifest"};
  static constexpr std::string_view kIdsArtifactName{"ids"};
  static constexpr std::string_view kVectorsArtifactName{"vectors"};

  DiskFlatSegment(const DiskFlatSegment &) = delete;
  auto operator=(const DiskFlatSegment &) -> DiskFlatSegment & = delete;
  DiskFlatSegment(DiskFlatSegment &&) = delete;
  auto operator=(DiskFlatSegment &&) -> DiskFlatSegment & = delete;

  [[nodiscard]] static auto build(DiskFlatBuildInput input,
                                  core::Metric metric,
                                  const DiskFlatPublicationOptions &options,
                                  core::BuildContext &context)
      -> core::Result<std::unique_ptr<DiskFlatSegment>> {
    auto status = validate_build_input(input, metric, options, context);
    if (!status.ok()) {
      return status;
    }
    internal::collection::ArtifactTransactionOptions transaction_options;
    transaction_options.collection_root = options.collection_root;
    transaction_options.target_relative_directory =
        std::filesystem::path("segments") / options.segment_id;
    transaction_options.transaction_id =
        "disk_flat_" + options.segment_id + "_g" + std::to_string(options.segment_generation);
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
      DiskFlatBuilder builder(input.vectors.dim, legacy_metric(metric));
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
    auto segment_entry = make_segment_entry(options);
    auto prepared = transaction->prepare(std::move(segment_entry));
    if (!prepared.ok()) {
      return prepared.status();
    }
    auto manifest = make_collection_manifest(input.vectors.dim, metric, options);
    if (!manifest.ok()) {
      return manifest.status();
    }
    status = transaction->publish(std::move(manifest).value());
    if (!status.ok()) {
      return status;
    }
    try {
      const auto final_path = transaction->final_payload_directory();
      auto searcher = std::make_shared<DiskFlatSegmentSearcher>(final_path);
      return std::unique_ptr<DiskFlatSegment>(new DiskFlatSegment(std::move(searcher), final_path));
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

  [[nodiscard]] static auto open(core::ArtifactView artifact,
                                 const core::OpenOptions &,
                                 core::OpenContext &context)
      -> core::Result<std::unique_ptr<DiskFlatSegment>> {
    auto control = core::validate_runtime_control(context.deadline,
                                                  context.cancellation,
                                                  core::OperationStage::open);
    if (!control.ok()) {
      return control;
    }
    const auto manifest_path = artifact.find(kManifestArtifactName);
    const auto ids_path = artifact.find(kIdsArtifactName);
    const auto vectors_path = artifact.find(kVectorsArtifactName);
    if (manifest_path.empty() || ids_path.empty() || vectors_path.empty()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 "DiskFlat open requires manifest, ids, and vectors artifacts");
    }
    try {
      const auto directory = std::filesystem::path(manifest_path).parent_path();
      const auto native = SegmentManifest::load(std::filesystem::path(manifest_path));
      if (std::filesystem::path(ids_path).lexically_normal() !=
              (directory / native.ids_file).lexically_normal() ||
          std::filesystem::path(vectors_path).lexically_normal() !=
              (directory / native.vectors_file).lexically_normal()) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::open,
                                   core::StatusDetail::malformed_struct,
                                   "DiskFlat ArtifactView paths disagree with native manifest");
      }
      std::uint64_t bytes{};
      for (const auto &path : {std::filesystem::path(manifest_path),
                               std::filesystem::path(ids_path),
                               std::filesystem::path(vectors_path)}) {
        const auto size = std::filesystem::file_size(path);
        if (size > std::numeric_limits<std::uint64_t>::max() - bytes) {
          return core::Status::error(core::StatusCode::invalid_argument,
                                     core::OperationStage::open,
                                     core::StatusDetail::arithmetic_overflow,
                                     "DiskFlat artifact bytes overflow uint64");
        }
        bytes += static_cast<std::uint64_t>(size);
      }
      auto budget =
          core::require_lease(context.resident_lease,
                              bytes,
                              core::OperationStage::open,
                              "DiskFlat resident lease is too small for mapped artifacts");
      if (!budget.ok()) {
        return budget;
      }
      budget = require_io_credits(context.io_credits,
                                  3,
                                  bytes,
                                  core::OperationStage::open,
                                  "DiskFlat open I/O credits are too small");
      if (!budget.ok()) {
        return budget;
      }
      auto searcher = std::make_shared<DiskFlatSegmentSearcher>(directory);
      return std::unique_ptr<DiskFlatSegment>(new DiskFlatSegment(std::move(searcher), directory));
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
      -> core::Result<std::unique_ptr<DiskFlatSegment>> {
    try {
      const auto native = SegmentManifest::load(segment_directory / "manifest.txt");
      std::array<std::string, 3> paths{(segment_directory / "manifest.txt").string(),
                                       (segment_directory / native.ids_file).string(),
                                       (segment_directory / native.vectors_file).string()};
      const std::array<core::ArtifactLocation, 3>
          locations{core::ArtifactLocation(kManifestArtifactName, paths[0]),
                    core::ArtifactLocation(kIdsArtifactName, paths[1]),
                    core::ArtifactLocation(kVectorsArtifactName, paths[2])};
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
      -> core::Result<std::unique_ptr<DiskFlatSegment>> {
    auto opened =
        internal::collection::CollectionManifestDualReader::open(collection_root, reader_options);
    if (!opened.ok()) {
      return opened.status();
    }
    const auto &segments = opened.value().manifest.segments;
    const auto found = std::find_if(segments.begin(), segments.end(), [&](const auto &entry) {
      return entry.segment_id == segment_id;
    });
    if (found == segments.end()) {
      return core::Status::error(core::StatusCode::not_found,
                                 core::OperationStage::open,
                                 core::StatusDetail::none,
                                 "DiskFlat segment is absent from the collection manifest");
    }
    if (found->algorithm_id != core::algorithm::flat) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::open,
                                 core::StatusDetail::operation_slot_absent,
                                 "requested collection segment is not DiskFlat");
    }
    std::array<std::string, 3> paths{};
    for (const auto &artifact : found->artifacts) {
      if (artifact.logical_name == kManifestArtifactName) {
        paths[0] = (collection_root / artifact.relative_path).string();
      } else if (artifact.logical_name == kIdsArtifactName) {
        paths[1] = (collection_root / artifact.relative_path).string();
      } else if (artifact.logical_name == kVectorsArtifactName) {
        paths[2] = (collection_root / artifact.relative_path).string();
      }
    }
    const std::array<core::ArtifactLocation, 3>
        locations{core::ArtifactLocation(kManifestArtifactName, paths[0]),
                  core::ArtifactLocation(kIdsArtifactName, paths[1]),
                  core::ArtifactLocation(kVectorsArtifactName, paths[2])};
    return open(core::ArtifactView(locations), open_options, context);
  }

  [[nodiscard]] auto descriptor() const noexcept -> core::Descriptor {
    core::Descriptor descriptor;
    descriptor.algorithm_id = core::algorithm::flat;
    descriptor.format_version = kFormatVersion;
    descriptor.factory_version = 1;
    descriptor.dim = searcher_->dim();
    descriptor.metric = core_metric(searcher_->manifest().metric);
    descriptor.stored_scalar_type = core::ScalarType::float32;
    descriptor.medium = core::Medium::disk;
    descriptor.preprocessing = searcher_->manifest().metric == core::Metric::cosine
                                   ? core::MetricPreprocessing::l2_normalized
                                   : core::MetricPreprocessing::none;
    descriptor.engine_factory_id = core::algorithm::flat;
    return descriptor;
  }

  [[nodiscard]] auto search(const core::SearchRequest &request) const -> core::Status {
    if (request.queries.rows != 1) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "DiskFlat single search requires exactly one query row");
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
    if (manifest_target.empty() || ids_target.empty() || vectors_target.empty()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::save,
                                 core::StatusDetail::malformed_struct,
                                 "DiskFlat save requires manifest, ids, and vectors logical names");
    }
    try {
      const auto &native = searcher_->manifest();
      const std::array sources{directory_ / "manifest.txt",
                               directory_ / native.ids_file,
                               directory_ / native.vectors_file};
      const std::array targets{std::filesystem::path(manifest_target),
                               std::filesystem::path(ids_target),
                               std::filesystem::path(vectors_target)};
      for (std::size_t index = 0; index < sources.size(); ++index) {
        if (std::filesystem::absolute(sources[index]).lexically_normal() ==
            std::filesystem::absolute(targets[index]).lexically_normal()) {
          return core::Status::error(core::StatusCode::conflict,
                                     core::OperationStage::save,
                                     core::StatusDetail::already_exists,
                                     "DiskFlat save destination aliases its source artifact");
        }
        std::filesystem::copy_file(sources[index],
                                   targets[index],
                                   std::filesystem::copy_options::overwrite_existing);
        platform::sync_file_or_throw(targets[index]);
        saved_artifacts_[index] =
            core::Artifact(index == 0   ? kManifestArtifactName
                           : index == 1 ? kIdsArtifactName
                                        : kVectorsArtifactName,
                           static_cast<std::uint64_t>(std::filesystem::file_size(targets[index])),
                           0);
      }
      manifest = core::ArtifactManifest{};
      manifest.schema_version = 1;
      manifest.format_version = kFormatVersion;
      manifest.algorithm_id = core::algorithm::flat;
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

  [[nodiscard]] auto save_transactional(const DiskFlatPublicationOptions &options,
                                        core::BuildContext &context) const
      -> core::Result<std::filesystem::path> {
    if (options.segment_id != searcher_->manifest().segment_id) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::save,
                                 core::StatusDetail::malformed_struct,
                                 "byte-preserving DiskFlat save retains its native segment id");
    }
    internal::collection::ArtifactTransactionOptions transaction_options;
    transaction_options.collection_root = options.collection_root;
    transaction_options.target_relative_directory =
        std::filesystem::path("segments") / options.segment_id;
    transaction_options.transaction_id =
        "disk_flat_save_" + options.segment_id + "_g" + std::to_string(options.segment_generation);
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
    auto collection_manifest =
        make_collection_manifest(searcher_->dim(), descriptor().metric, options);
    if (!collection_manifest.ok()) {
      return collection_manifest.status();
    }
    status = transaction->publish(std::move(collection_manifest).value());
    if (!status.ok()) {
      return status;
    }
    return transaction->final_payload_directory();
  }

  [[nodiscard]] auto export_rows(const core::OpaqueOperationRequest &request,
                                 core::ExportCursor &cursor) const -> core::Status {
    if (!core::is_current_struct(request) || request.payload == nullptr ||
        request.payload_size < sizeof(DiskFlatExportRequest)) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::export_rows,
                                 core::StatusDetail::malformed_struct,
                                 "DiskFlat export request payload is missing or truncated");
    }
    auto &typed = *static_cast<const DiskFlatExportRequest *>(request.payload);
    if (!core::is_current_struct(typed) || typed.batch_rows == 0 ||
        typed.lifetime_owner == nullptr) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::export_rows,
                                 core::StatusDetail::malformed_struct,
                                 "DiskFlat export request requires batch_rows and lifetime owner");
    }
    auto state =
        std::shared_ptr<DiskFlatExportState>(new DiskFlatExportState(searcher_, typed.batch_rows));
    cursor.state = state.get();
    *typed.lifetime_owner = std::move(state);
    return core::Status::success();
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

  [[nodiscard]] static auto into_any(std::unique_ptr<DiskFlatSegment> segment)
      -> core::Result<core::AnySegment> {
    if (segment == nullptr) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::admission,
                                 core::StatusDetail::null_data,
                                 "cannot erase a null DiskFlat segment");
    }
    auto shared = std::shared_ptr<DiskFlatSegment>(std::move(segment));
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
  friend class DiskFlatLegacyFactory;

  DiskFlatSegment(std::shared_ptr<DiskFlatSegmentSearcher> searcher,
                  std::filesystem::path directory)
      : searcher_(std::move(searcher)), directory_(std::move(directory)) {
    const auto &native = searcher_->manifest();
    for (const auto &path : {directory_ / "manifest.txt",
                             directory_ / native.ids_file,
                             directory_ / native.vectors_file}) {
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
            {std::string(kVectorsArtifactName), "vectors.f32.bin", true, {}}};
  }

  [[nodiscard]] static auto legacy_metric(core::Metric metric) -> core::Metric {
    switch (metric) {
      case core::Metric::l2:
        return core::Metric::l2;
      case core::Metric::inner_product:
        return core::Metric::inner_product;
      case core::Metric::cosine:
        return core::Metric::cosine;
    }
    throw std::invalid_argument("DiskFlat received an unknown core metric");
  }

  [[nodiscard]] static auto core_metric(core::Metric metric) noexcept -> core::Metric {
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

  [[nodiscard]] static auto validate_build_input(const DiskFlatBuildInput &input,
                                                 core::Metric metric,
                                                 const DiskFlatPublicationOptions &options,
                                                 core::BuildContext &context) -> core::Status {
    if (!core::is_current_struct(input) || input.vectors.dim == 0) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::build,
                                 core::StatusDetail::malformed_struct,
                                 "DiskFlat build input is incompatible or zero-dimensional");
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
                                 "DiskFlat stores float32 tensors without implicit conversion");
    }
    if (input.vectors.rows == 0 || input.logical_ids.size() != input.vectors.rows ||
        options.collection_root.empty() || !detail::is_valid_segment_id(options.segment_id) ||
        options.segment_generation == 0 || options.manifest_generation == 0) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::build,
                                 core::StatusDetail::malformed_struct,
                                 "DiskFlat build rows/ids/publication options are invalid");
    }
    try {
      (void)legacy_metric(metric);
    } catch (...) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::build,
                                 core::StatusDetail::malformed_struct,
                                 "DiskFlat build metric is invalid");
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
    std::uint64_t bytes{};
    if (!core::checked_multiply(input.vectors.rows, input.vectors.dim, components) ||
        !core::checked_multiply(components, sizeof(float), vector_bytes) ||
        !core::checked_multiply(input.vectors.rows, sizeof(std::uint64_t), id_bytes) ||
        !core::checked_add(vector_bytes, id_bytes, bytes)) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::build,
                                 core::StatusDetail::arithmetic_overflow,
                                 "DiskFlat build artifact size overflows uint64");
    }
    return context.growing_reservation.ensure(bytes,
                                              core::OperationStage::build,
                                              "DiskFlat build reservation is too small");
  }

  [[nodiscard]] static auto make_segment_entry(const DiskFlatPublicationOptions &options)
      -> internal::collection::SegmentEntryV2 {
    internal::collection::SegmentEntryV2 entry;
    entry.segment_id = options.segment_id;
    entry.generation = options.segment_generation;
    entry.role = internal::collection::SegmentRoleV2::searchable;
    entry.algorithm_id = core::algorithm::flat;
    entry.format_version = kFormatVersion;
    entry.factory_key = "flat";
    entry.capabilities.operations = core::capability_bit(core::OperationCapability::search) |
                                    core::capability_bit(core::OperationCapability::batch_search) |
                                    core::capability_bit(core::OperationCapability::save) |
                                    core::capability_bit(core::OperationCapability::export_rows) |
                                    core::capability_bit(core::OperationCapability::stats);
    entry.capabilities.reentrant_search = true;
    entry.capabilities.cooperative_cancel = true;
    entry.capabilities.explicit_drain = false;
    entry.lifecycle = internal::collection::SegmentLifecycleV2::sealed;
    entry.wal_cut = options.wal_cut;
    entry.row_versions = options.row_versions;
    entry.id_map_checkpoint = options.id_map_checkpoint;
    entry.reader_compatibility.required_features = {"disk_flat_segment"};
    entry.extensions.emplace("exact_oracle", "true");
    return entry;
  }

  [[nodiscard]] static auto make_collection_manifest(std::uint32_t dim,
                                                     core::Metric metric,
                                                     const DiskFlatPublicationOptions &options)
      -> core::Result<internal::collection::ArtifactManifestV2> {
    auto manifest = options.base_manifest.value_or(internal::collection::ArtifactManifestV2{});
    if (options.base_manifest.has_value() &&
        (manifest.collection.dim != dim || manifest.collection.metric != metric ||
         manifest.collection.scalar_type != core::ScalarType::float32)) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::build,
                                 core::StatusDetail::malformed_struct,
                                 "DiskFlat publication disagrees with base collection schema");
    }
    manifest.collection.dim = dim;
    manifest.collection.metric = metric;
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

  [[nodiscard]] auto validate_search_request(const core::SearchRequest &request) const
      -> core::Status {
    if (!core::is_current_struct(request) || !core::is_current_struct(request.options) ||
        request.context == nullptr || request.response == nullptr) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "DiskFlat search request is incomplete or incompatible");
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
                                 "DiskFlat search accepts float32 tensors only");
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
                                 "DiskFlat has no engine-local metadata filter");
    }
    if (request.options.top_k > std::numeric_limits<std::uint32_t>::max()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "DiskFlat top_k exceeds uint32");
    }
    for (const auto &extension : request.options.extensions) {
      if (extension.unknown_policy == core::UnknownExtensionPolicy::reject) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::unknown_extension,
                                   "DiskFlat defines no algorithm-specific search extension");
      }
    }
    status = core::validate_runtime_control(request.context->deadline,
                                            request.context->cancellation,
                                            core::OperationStage::search);
    if (!status.ok()) {
      return status;
    }
    const auto heap_rows = std::min<std::uint64_t>(request.options.top_k, searcher_->size());
    std::uint64_t scratch{};
    if (!core::checked_multiply(heap_rows, sizeof(DiskSearchHit), scratch)) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "DiskFlat scratch size overflows uint64");
    }
    if (searcher_->manifest().metric == core::Metric::cosine) {
      std::uint64_t normalized_bytes{};
      if (!core::checked_multiply(searcher_->dim(), sizeof(float), normalized_bytes) ||
          !core::checked_add(scratch, normalized_bytes, scratch)) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::arithmetic_overflow,
                                   "DiskFlat normalized query scratch overflows uint64");
      }
    }
    status = core::require_lease(request.context->query_scratch_lease,
                                 scratch,
                                 core::OperationStage::search,
                                 "DiskFlat query scratch lease is too small");
    if (!status.ok()) {
      return status;
    }
    std::uint64_t scan_bytes{};
    std::uint64_t request_bytes{};
    if (!core::checked_multiply(searcher_->size(), searcher_->dim(), scan_bytes) ||
        !core::checked_multiply(scan_bytes, sizeof(float), scan_bytes) ||
        !core::checked_multiply(scan_bytes, request.queries.rows, request_bytes)) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "DiskFlat scan byte accounting overflows uint64");
    }
    return require_io_credits(request.context->io_credits,
                              request.queries.rows,
                              request_bytes,
                              core::OperationStage::search,
                              "DiskFlat search I/O credits are too small");
  }

  [[nodiscard]] auto execute_search(const core::SearchRequest &request) const -> core::Status {
    auto status = validate_search_request(request);
    if (!status.ok()) {
      return status;
    }
    auto &response = *request.response;
    response.score_kind = core::ScoreKind::distance;
    response.comparable_metric = descriptor().metric;
    response.result_flags = core::ResultFlag::none;
    if (request.options.top_k == 0 || request.queries.rows == 0) {
      core::initialize_empty_response(response,
                                      request.queries.rows,
                                      request.options.top_k == 0
                                          ? core::SearchCompleteness::complete_k
                                          : core::SearchCompleteness::eligible_exhausted);
      return core::Status::success();
    }
    DiskSearchOptions options;
    options.top_k = static_cast<std::uint32_t>(request.options.top_k);
    options.exact_rerank = true;
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
          response.hits[static_cast<std::size_t>(cursor + index)] =
              core::SearchHit(core::SegmentRowId(hits[index].label),
                              hits[index].distance,
                              core::ScoreKind::distance,
                              response.comparable_metric,
                              core::ResultFlag::none);
        }
        const auto written = static_cast<core::RowCount>(hits.size());
        cursor += written;
        response.offsets[row + 1] = cursor;
        response.valid_counts[row] = written;
        response.statuses[row] = core::Status::success();
        response.completeness[row] = written == request.options.top_k
                                         ? core::SearchCompleteness::complete_k
                                         : core::SearchCompleteness::eligible_exhausted;
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
      request.context->stats->visited += searcher_->size() * request.queries.rows;
      request.context->stats->io_requests += request.queries.rows;
      request.context->stats->io_bytes +=
          searcher_->size() * searcher_->dim() * sizeof(float) * request.queries.rows;
    }
    return core::Status::success();
  }

  std::shared_ptr<DiskFlatSegmentSearcher> searcher_{};
  std::filesystem::path directory_{};
  std::uint64_t artifact_bytes_{};
  mutable std::array<core::Artifact, 3> saved_artifacts_{};
};

class DiskFlatSegmentFactory {
 public:
  static constexpr auto registration = internal::disk::kDiskFlatRegistration;

  [[nodiscard]] static auto build(
      DiskFlatBuildInput input,
      core::Metric metric,
      const DiskFlatPublicationOptions &options,
      core::BuildContext &context,
      const internal::disk::DiskEngineFeatureFlags &features = {}) noexcept
      -> core::Result<std::unique_ptr<DiskFlatSegment>> {
    if (!features.enabled(registration.feature)) {
      return disabled(core::OperationStage::build);
    }
    try {
      return DiskFlatSegment::build(std::move(input), metric, options, context);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::build);
    }
  }

  [[nodiscard]] static auto open(
      core::ArtifactView artifacts,
      const core::OpenOptions &options,
      core::OpenContext &context,
      const internal::disk::DiskEngineFeatureFlags &features = {}) noexcept
      -> core::Result<std::unique_ptr<DiskFlatSegment>> {
    if (!features.enabled(registration.feature)) {
      return disabled(core::OperationStage::open);
    }
    try {
      return DiskFlatSegment::open(artifacts, options, context);
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
              "DiskFlatSegment factory is disabled; DiskCollection v1 is unchanged");
  }
};

// Explicit legacy registration target used by differential tests and rollback
// tooling. It is intentionally not selected by DiskFlatSegmentFactory's flag.
class DiskFlatLegacyFactory {
 public:
  static constexpr auto registration = internal::disk::kDiskFlatRegistration;

  [[nodiscard]] static auto build(const DiskFlatBuildInput &input,
                                  core::Metric metric,
                                  const std::filesystem::path &segment_directory) noexcept
      -> core::Result<std::unique_ptr<DiskFlatSegmentSearcher>> {
    try {
      if (input.vectors.scalar_type != core::ScalarType::float32 || input.vectors.rows == 0 ||
          input.logical_ids.size() != input.vectors.rows) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::build,
                                   core::StatusDetail::malformed_struct,
                                   "legacy DiskFlat factory input is invalid");
      }
      DiskFlatBuilder builder(input.vectors.dim, DiskFlatSegment::legacy_metric(metric));
      for (core::RowCount row = 0; row < input.vectors.rows; ++row) {
        builder.add_batch(input.vectors.row<float>(row), input.logical_ids.data() + row, 1);
      }
      (void)builder.finish(segment_directory);
      return std::make_unique<DiskFlatSegmentSearcher>(segment_directory);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::build);
    }
  }

  [[nodiscard]] static auto open(const std::filesystem::path &segment_directory) noexcept
      -> core::Result<std::unique_ptr<DiskFlatSegmentSearcher>> {
    try {
      return std::make_unique<DiskFlatSegmentSearcher>(segment_directory);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }
};

static_assert(core::Searchable<DiskFlatSegment>);
static_assert(core::BatchSearchable<DiskFlatSegment>);
static_assert(core::Saveable<DiskFlatSegment>);
static_assert(core::Exportable<DiskFlatSegment>);
static_assert(core::StatsProvider<DiskFlatSegment>);
static_assert(!core::Mutable<DiskFlatSegment>);

}  // namespace alaya::disk
