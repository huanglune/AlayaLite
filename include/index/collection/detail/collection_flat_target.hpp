// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "index/collection/filter_seal_gc.hpp"
#include "index/disk/disk_flat_segment.hpp"

namespace alaya::internal::collection::detail {

class CollectionTypedFlatSegment {
 public:
  CollectionTypedFlatSegment(std::shared_ptr<::alaya::disk::DiskFlatSegment> flat,
                             core::ScalarType scalar_type)
      : flat_(std::move(flat)), scalar_type_(scalar_type) {}

  [[nodiscard]] auto descriptor() const noexcept -> core::Descriptor {
    auto descriptor = flat_->descriptor();
    descriptor.stored_scalar_type = scalar_type_;
    return descriptor;
  }

  [[nodiscard]] auto search(const core::SearchRequest &request) const -> core::Status {
    if (request.queries.rows != 1) {
      return failure(core::OperationStage::validation,
                     core::StatusCode::invalid_argument,
                     "Collection Flat target single search requires one query");
    }
    return execute(request, false);
  }

  [[nodiscard]] auto batch_search(const core::SearchRequest &request) const -> core::Status {
    return execute(request, true);
  }

  [[nodiscard]] auto export_rows(const core::OpaqueOperationRequest &request,
                                 core::ExportCursor &cursor) const -> core::Status {
    return flat_->export_rows(request, cursor);
  }

  [[nodiscard]] auto stats(core::SegmentStats &stats) const noexcept -> core::Status {
    return flat_->stats(stats);
  }

 private:
  [[nodiscard]] static auto failure(
      core::OperationStage stage,
      core::StatusCode code,
      std::string diagnostic,
      core::StatusDetail detail = core::StatusDetail::malformed_struct) -> core::Status {
    return core::Status::error(code, stage, detail, std::move(diagnostic));
  }

  [[nodiscard]] auto execute(const core::SearchRequest &request, bool batch) const -> core::Status {
    auto status =
        core::validate_tensor(request.queries, descriptor().dim, core::OperationStage::validation);
    if (!status.ok()) {
      return status;
    }
    if (request.queries.scalar_type != scalar_type_) {
      return failure(core::OperationStage::validation,
                     core::StatusCode::not_supported,
                     "Collection Flat target query dtype disagrees with the Collection schema",
                     core::StatusDetail::unsupported_scalar_type);
    }
    if (scalar_type_ == core::ScalarType::float32) {
      return batch ? flat_->batch_search(request) : flat_->search(request);
    }
    std::vector<float> converted(static_cast<std::size_t>(request.queries.rows) *
                                 request.queries.dim);
    for (core::RowCount row = 0; row < request.queries.rows; ++row) {
      for (std::uint32_t column = 0; column < request.queries.dim; ++column) {
        const auto output = static_cast<std::size_t>(row * request.queries.dim + column);
        if (scalar_type_ == core::ScalarType::int8) {
          converted[output] = static_cast<float>(request.queries.row<std::int8_t>(row)[column]);
        } else {
          converted[output] = static_cast<float>(request.queries.row<std::uint8_t>(row)[column]);
        }
      }
    }
    auto delegated = request;
    delegated.queries = core::TypedTensorView::contiguous(converted.data(),
                                                          request.queries.rows,
                                                          request.queries.dim);
    return batch ? flat_->batch_search(delegated) : flat_->search(delegated);
  }

  std::shared_ptr<::alaya::disk::DiskFlatSegment> flat_{};
  core::ScalarType scalar_type_{core::ScalarType::float32};
};

[[nodiscard]] inline auto erase_collection_flat(
    std::unique_ptr<::alaya::disk::DiskFlatSegment> flat,
    core::ScalarType scalar_type) -> core::Result<core::AnySegment> {
  if (flat == nullptr) {
    return core::Status::error(core::StatusCode::invalid_argument,
                               core::OperationStage::open,
                               core::StatusDetail::null_data,
                               "cannot erase a null Collection Flat target");
  }
  auto adapter =
      std::make_shared<CollectionTypedFlatSegment>(std::shared_ptr<::alaya::disk::DiskFlatSegment>(
                                                       std::move(flat)),
                                                   scalar_type);
  core::SegmentInstanceConfig config;
  config.readonly = true;
  config.concurrency.reentrant_search = true;
  config.concurrency.search_with_stage = false;
  config.concurrency.search_with_publish = false;
  config.concurrency.serial_mutation = true;
  config.concurrency.native_async = false;
  config.concurrency.cooperative_cancel = true;
  config.concurrency.explicit_drain = false;
  return core::AnySegment::from_sync(std::move(adapter), std::move(config));
}

[[nodiscard]] inline auto open_collection_flat_entry(const std::filesystem::path &root,
                                                     const SegmentEntryV2 &entry,
                                                     core::ScalarType scalar_type,
                                                     core::OpenContext &context)
    -> core::Result<core::AnySegment> {
  if (entry.algorithm_id != core::algorithm::flat ||
      entry.lifecycle == SegmentLifecycleV2::retired) {
    return core::Status::error(core::StatusCode::not_supported,
                               core::OperationStage::open,
                               core::StatusDetail::operation_slot_absent,
                               "Gate-10 control plane can open only Flat replacement entries");
  }
  std::array<std::string, 3> paths{};
  for (const auto &artifact : entry.artifacts) {
    if (artifact.logical_name == ::alaya::disk::DiskFlatSegment::kManifestArtifactName) {
      paths[0] = (root / artifact.relative_path).string();
    } else if (artifact.logical_name == ::alaya::disk::DiskFlatSegment::kIdsArtifactName) {
      paths[1] = (root / artifact.relative_path).string();
    } else if (artifact.logical_name == ::alaya::disk::DiskFlatSegment::kVectorsArtifactName) {
      paths[2] = (root / artifact.relative_path).string();
    }
  }
  if (std::ranges::any_of(paths, [](const auto &path) {
        return path.empty();
      })) {
    return core::Status::error(core::StatusCode::corruption,
                               core::OperationStage::open,
                               core::StatusDetail::malformed_struct,
                               "Flat replacement manifest is missing a native artifact");
  }
  const std::array<core::ArtifactLocation, 3>
      locations{core::ArtifactLocation(::alaya::disk::DiskFlatSegment::kManifestArtifactName,
                                       paths[0]),
                core::ArtifactLocation(::alaya::disk::DiskFlatSegment::kIdsArtifactName, paths[1]),
                core::ArtifactLocation(::alaya::disk::DiskFlatSegment::kVectorsArtifactName,
                                       paths[2])};
  auto opened = ::alaya::disk::DiskFlatSegment::open(core::ArtifactView(locations), {}, context);
  if (!opened.ok()) {
    return opened.status();
  }
  return erase_collection_flat(std::move(opened).value(), scalar_type);
}

[[nodiscard]] inline auto vector_as_float(const OwnedVector &vector, std::vector<float> &output)
    -> core::Status {
  const auto view = vector.view();
  output.reserve(output.size() + view.dim);
  for (std::uint32_t column = 0; column < view.dim; ++column) {
    switch (view.scalar_type) {
      case core::ScalarType::float32:
        output.push_back(view.row<float>(0)[column]);
        break;
      case core::ScalarType::int8:
        output.push_back(static_cast<float>(view.row<std::int8_t>(0)[column]));
        break;
      case core::ScalarType::uint8:
        output.push_back(static_cast<float>(view.row<std::uint8_t>(0)[column]));
        break;
    }
  }
  return core::Status::success();
}

struct FlatTargetBuildResult {
  core::AnySegment segment{};
  std::uint64_t artifact_bytes{};
};

[[nodiscard]] inline auto build_collection_flat_target(
    const CollectionSchema &schema,
    std::span<const RegisteredRow> rows,
    const ::alaya::disk::DiskFlatPublicationOptions &publication,
    core::BuildContext &context) -> core::Result<FlatTargetBuildResult> {
  std::vector<float> vectors;
  std::vector<std::uint64_t> row_ids;
  for (const auto &row : rows) {
    if (row.state != VersionState::live) {
      continue;
    }
    if (!row.payload.vector.has_value()) {
      return core::Status::error(core::StatusCode::resource_exhausted,
                                 core::OperationStage::build,
                                 core::StatusDetail::budget_denied,
                                 "Flat seal target requires every live source vector");
    }
    auto status = vector_as_float(*row.payload.vector, vectors);
    if (!status.ok()) {
      return status;
    }
    row_ids.push_back(static_cast<std::uint64_t>(row.row_id));
  }
  if (row_ids.empty()) {
    return core::Status::error(core::StatusCode::not_found,
                               core::OperationStage::build,
                               core::StatusDetail::none,
                               "cannot build an empty Flat seal target");
  }
  const auto input =
      ::alaya::disk::DiskFlatBuildInput(core::TypedTensorView::contiguous(vectors.data(),
                                                                          row_ids.size(),
                                                                          schema.dim),
                                        row_ids);
  auto built = ::alaya::disk::DiskFlatSegment::build(input, schema.metric, publication, context);
  if (!built.ok()) {
    return built.status();
  }
  auto erased = erase_collection_flat(std::move(built).value(), schema.scalar_type);
  if (!erased.ok()) {
    return erased.status();
  }
  std::uint64_t bytes{};
  const auto manifest = load_manifest_v2_if_present(publication.collection_root);
  if (!manifest.ok()) {
    return manifest.status();
  }
  if (manifest.value().has_value()) {
    const auto found = std::ranges::find_if(manifest.value()->segments, [&](const auto &entry) {
      return entry.segment_id == publication.segment_id &&
             entry.generation == publication.segment_generation;
    });
    if (found != manifest.value()->segments.end()) {
      for (const auto &artifact : found->artifacts) {
        if (!core::checked_add(bytes, artifact.size_bytes, bytes)) {
          bytes = std::numeric_limits<std::uint64_t>::max();
          break;
        }
      }
    }
  }
  return FlatTargetBuildResult{std::move(erased).value(), bytes};
}

[[nodiscard]] inline auto flat_segment_name(std::uint64_t segment_id) -> std::string {
  auto digits = std::to_string(segment_id);
  if (digits.size() > 8) {
    throw std::invalid_argument("Flat target segment id exceeds the native eight-digit namespace");
  }
  return "seg_" + std::string(8 - digits.size(), '0') + digits;
}

}  // namespace alaya::internal::collection::detail
