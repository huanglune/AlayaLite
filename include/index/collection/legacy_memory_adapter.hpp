// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "index/collection/types.hpp"

namespace alaya::internal::collection {

// Adapts the raw GraphSearchJob::search_solo shape used by the current PyIndex
// implementation. It is readonly even though PyIndex exposes legacy mutation:
// no engine-native v3 mutation bundle exists for that path.
template <class SearchJob, class DataType, class IdType, class DistanceType = float>
class LegacyMemorySegmentAdapter {
 public:
  LegacyMemorySegmentAdapter(std::shared_ptr<SearchJob> search_job,
                             core::Descriptor descriptor,
                             core::RowCount rows,
                             std::uint32_t default_effort)
      : search_job_(std::move(search_job)),
        descriptor_(std::move(descriptor)),
        rows_(rows),
        default_effort_(default_effort) {}

  [[nodiscard]] auto descriptor() const noexcept -> core::Descriptor { return descriptor_; }

  [[nodiscard]] auto search(const core::SearchRequest &request) const -> core::Status {
    if (request.queries.rows != 1) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "legacy memory single search requires one query row");
    }
    return execute(request);
  }

  [[nodiscard]] auto batch_search(const core::SearchRequest &request) const -> core::Status {
    return execute(request);
  }

  [[nodiscard]] auto stats(core::SegmentStats &stats) const noexcept -> core::Status {
    stats = core::SegmentStats{};
    stats.snapshot_version = 1;
    stats.live_rows = rows_;
    stats.allocated_rows = rows_;
    stats.health = core::SegmentHealth::healthy;
    return core::Status::success();
  }

 private:
  [[nodiscard]] auto execute(const core::SearchRequest &request) const -> core::Status {
    if (search_job_ == nullptr || request.response == nullptr || request.context == nullptr ||
        request.queries.scalar_type != core::scalar_type_for<DataType>) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::unsupported_scalar_type,
                                 "legacy memory adapter request is invalid");
    }
    if (request.filter.kind != core::SegmentFilterKind::none) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::validation,
                                 core::StatusDetail::operation_slot_absent,
                                 "legacy memory adapter has no v3 filter pushdown");
    }
    if (request.options.top_k > std::numeric_limits<std::uint32_t>::max()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "legacy memory top_k exceeds uint32");
    }
    auto &response = *request.response;
    response.query_count = request.queries.rows;
    response.score_kind = core::ScoreKind::distance;
    response.comparable_metric = descriptor_.metric;
    response.result_flags = core::ResultFlag::approximate;
    response.offsets[0] = 0;
    const auto top_k = static_cast<std::uint32_t>(request.options.top_k);
    const auto effort = std::max(default_effort_, top_k);
    std::vector<IdType> ids(top_k);
    std::vector<DistanceType> distances(top_k);
    core::RowCount cursor{};
    for (core::RowCount row = 0; row < request.queries.rows; ++row) {
      const auto control = core::validate_runtime_control(request.context->deadline,
                                                          request.context->cancellation,
                                                          core::OperationStage::search);
      if (!control.ok()) {
        return control;
      }
      try {
        search_job_->search_solo(const_cast<DataType *>(request.queries.row<DataType>(row)),
                                 ids.data(),
                                 distances.data(),
                                 top_k,
                                 effort);
      } catch (...) {
        return core::status_from_exception(core::OperationStage::search);
      }
      core::RowCount written{};
      while (written < top_k &&
             ids[static_cast<std::size_t>(written)] != std::numeric_limits<IdType>::max()) {
        const auto score = static_cast<float>(distances[static_cast<std::size_t>(written)]);
        if (std::isnan(score)) {
          return core::Status::error(core::StatusCode::internal,
                                     core::OperationStage::search,
                                     core::StatusDetail::invalid_score,
                                     "legacy memory search produced a NaN score");
        }
        response.hits[static_cast<std::size_t>(cursor + written)] =
            core::SearchHit(core::SegmentRowId(
                                static_cast<std::uint64_t>(ids[static_cast<std::size_t>(written)])),
                            score,
                            core::ScoreKind::distance,
                            descriptor_.metric,
                            core::ResultFlag::approximate);
        ++written;
      }
      cursor += written;
      response.offsets[static_cast<std::size_t>(row + 1)] = cursor;
      response.valid_counts[static_cast<std::size_t>(row)] = written;
      response.statuses[static_cast<std::size_t>(row)] = core::Status::success();
      response.completeness[static_cast<std::size_t>(row)] =
          written == top_k   ? core::SearchCompleteness::complete_k
          : written == rows_ ? core::SearchCompleteness::eligible_exhausted
                             : core::SearchCompleteness::strategy_incomplete;
    }
    return core::Status::success();
  }

  std::shared_ptr<SearchJob> search_job_{};
  core::Descriptor descriptor_{};
  core::RowCount rows_{};
  std::uint32_t default_effort_{100};
};

template <class DataType, class IdType, class DistanceType = float, class SearchJob>
[[nodiscard]] auto make_legacy_memory_segment(std::shared_ptr<SearchJob> search_job,
                                              core::Descriptor descriptor,
                                              core::RowCount rows,
                                              const CollectionFeatureFlags &features,
                                              std::uint32_t default_effort = 100)
    -> core::Result<core::AnySegment> {
  if (!features.legacy_memory_adapter) {
    return core::Status::error(core::StatusCode::not_supported,
                               core::OperationStage::open,
                               core::StatusDetail::operation_slot_absent,
                               "legacy memory adapter feature is disabled");
  }
  if (search_job == nullptr || descriptor.stored_scalar_type != core::scalar_type_for<DataType>) {
    return core::Status::error(core::StatusCode::invalid_argument,
                               core::OperationStage::open,
                               core::StatusDetail::malformed_struct,
                               "legacy memory adapter configuration is invalid");
  }
  using Adapter = LegacyMemorySegmentAdapter<SearchJob, DataType, IdType, DistanceType>;
  auto adapter = std::make_shared<Adapter>(std::move(search_job), descriptor, rows, default_effort);
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
