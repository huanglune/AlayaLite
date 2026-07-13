// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

#include "index/collection/types.hpp"
#include "index/disk/detail/disk_collection_v1.hpp"

namespace alaya::internal::collection {

// Search-only slot for the legacy DiskCollection v1 model. In particular, this
// type does not forward add_batch, flush, mark_deleted, or any other mutation
// entry. Durable mutable disk remains behind Gates 7 and 8.
class LegacyDiskCollectionAdapter {
 public:
  LegacyDiskCollectionAdapter(std::shared_ptr<disk::DiskCollection> collection,
                              core::Descriptor descriptor,
                              bool rank_only)
      : collection_(std::move(collection)),
        descriptor_(std::move(descriptor)),
        rank_only_(rank_only) {}

  [[nodiscard]] auto descriptor() const noexcept -> core::Descriptor { return descriptor_; }

  [[nodiscard]] auto search(const core::SearchRequest &request) const -> core::Status {
    if (request.queries.rows != 1) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "legacy disk single search requires one query row");
    }
    return execute(request);
  }

  [[nodiscard]] auto batch_search(const core::SearchRequest &request) const -> core::Status {
    return execute(request);
  }

  [[nodiscard]] auto stats(core::SegmentStats &stats) const noexcept -> core::Status {
    stats = core::SegmentStats{};
    stats.snapshot_version = 1;
    stats.live_rows = collection_->size();
    stats.allocated_rows = collection_->size();
    stats.health = core::SegmentHealth::healthy;
    return core::Status::success();
  }

 private:
  [[nodiscard]] auto execute(const core::SearchRequest &request) const -> core::Status {
    if (collection_ == nullptr || request.context == nullptr || request.response == nullptr ||
        request.queries.scalar_type != core::ScalarType::float32) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::unsupported_scalar_type,
                                 "legacy disk adapter only accepts float32 queries");
    }
    if (request.filter.kind != core::SegmentFilterKind::none) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::validation,
                                 core::StatusDetail::operation_slot_absent,
                                 "legacy disk adapter has no v3 filter pushdown");
    }
    if (request.options.top_k > std::numeric_limits<std::uint32_t>::max()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "legacy disk top_k exceeds uint32");
    }
    disk::DiskSearchOptions options;
    options.top_k = static_cast<std::uint32_t>(request.options.top_k);
    options.exact_rerank = request.options.rerank_policy != core::RerankPolicy::disabled;
    auto &response = *request.response;
    response.query_count = request.queries.rows;
    response.score_kind = rank_only_ ? core::ScoreKind::rank_only : core::ScoreKind::distance;
    response.comparable_metric = descriptor_.metric;
    response.result_flags = core::ResultFlag::approximate;
    response.offsets[0] = 0;
    core::RowCount cursor{};
    for (core::RowCount row = 0; row < request.queries.rows; ++row) {
      const auto control = core::validate_runtime_control(request.context->deadline,
                                                          request.context->cancellation,
                                                          core::OperationStage::search);
      if (!control.ok()) {
        return control;
      }
      std::vector<disk::DiskSearchHit> hits;
      try {
        hits = collection_->search(request.queries.row<float>(row), options);
      } catch (...) {
        return core::status_from_exception(core::OperationStage::search);
      }
      core::RowCount written{};
      for (const auto &hit : hits) {
        const auto score = rank_only_ ? static_cast<float>(written) : hit.distance;
        if (!rank_only_ && std::isnan(score)) {
          return core::Status::error(core::StatusCode::internal,
                                     core::OperationStage::search,
                                     core::StatusDetail::invalid_score,
                                     "legacy disk numeric search produced a NaN score");
        }
        response.hits[static_cast<std::size_t>(cursor + written)] =
            core::SearchHit(core::SegmentRowId(hit.label),
                            score,
                            rank_only_ ? core::ScoreKind::rank_only : core::ScoreKind::distance,
                            descriptor_.metric,
                            core::ResultFlag::approximate);
        ++written;
      }
      cursor += written;
      response.offsets[static_cast<std::size_t>(row + 1)] = cursor;
      response.valid_counts[static_cast<std::size_t>(row)] = written;
      response.statuses[static_cast<std::size_t>(row)] = core::Status::success();
      response.completeness[static_cast<std::size_t>(row)] =
          written == request.options.top_k ? core::SearchCompleteness::complete_k
                                           : core::SearchCompleteness::eligible_exhausted;
    }
    return core::Status::success();
  }

  std::shared_ptr<disk::DiskCollection> collection_{};
  core::Descriptor descriptor_{};
  bool rank_only_{};
};

[[nodiscard]] inline auto make_legacy_disk_segment(std::shared_ptr<disk::DiskCollection> collection,
                                                   core::Descriptor descriptor,
                                                   const CollectionFeatureFlags &features,
                                                   bool rank_only = false)
    -> core::Result<core::AnySegment> {
  if (!features.legacy_disk_adapter) {
    return core::Status::error(core::StatusCode::not_supported,
                               core::OperationStage::open,
                               core::StatusDetail::operation_slot_absent,
                               "legacy disk adapter feature is disabled");
  }
  if (collection == nullptr || descriptor.dim != collection->dim() ||
      descriptor.stored_scalar_type != core::ScalarType::float32 ||
      descriptor.medium != core::Medium::disk) {
    return core::Status::error(core::StatusCode::invalid_argument,
                               core::OperationStage::open,
                               core::StatusDetail::malformed_struct,
                               "legacy disk adapter configuration is invalid");
  }
  auto adapter =
      std::make_shared<LegacyDiskCollectionAdapter>(std::move(collection), descriptor, rank_only);
  core::SegmentInstanceConfig config;
  config.readonly = true;
  config.concurrency.reentrant_search = true;
  config.concurrency.search_with_stage = false;
  config.concurrency.search_with_publish = false;
  config.concurrency.serial_mutation = true;
  config.concurrency.native_async = false;
  config.concurrency.cooperative_cancel = false;
  config.concurrency.explicit_drain = false;
  return core::AnySegment::from_sync(std::move(adapter), std::move(config));
}

}  // namespace alaya::internal::collection
