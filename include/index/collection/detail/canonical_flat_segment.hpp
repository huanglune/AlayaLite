// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

#include "index/collection/types.hpp"

namespace alaya::internal::collection::detail {

// CanonicalFlatSegment is the facade-private active mutation target.  It is
// deliberately not a public DiskFlatSegment mutation API: the Collection owns
// logical IDs, metadata, WAL ordering and visibility, while this operation
// table owns only exact-searchable physical rows for the active generation.
class CanonicalFlatSegment {
 public:
  CanonicalFlatSegment(CollectionSchema schema, std::uint64_t segment_id, std::uint64_t generation)
      : schema_(schema), segment_id_(segment_id), generation_(generation) {}

  [[nodiscard]] auto descriptor() const noexcept -> core::Descriptor {
    core::Descriptor descriptor;
    descriptor.algorithm_id = core::algorithm::flat;
    descriptor.format_version = 1;
    descriptor.factory_version = 1;
    descriptor.dim = schema_.dim;
    descriptor.metric = schema_.metric;
    descriptor.stored_scalar_type = schema_.scalar_type;
    descriptor.medium = core::Medium::memory;
    descriptor.preprocessing = core::MetricPreprocessing::none;
    descriptor.engine_factory_id = core::algorithm::flat;
    return descriptor;
  }

  [[nodiscard]] auto search(const core::SearchRequest &request) const -> core::Status {
    if (request.queries.rows != 1) {
      return failure(core::OperationStage::validation,
                     core::StatusCode::invalid_argument,
                     core::StatusDetail::malformed_struct,
                     "canonical Flat single search requires exactly one query row");
    }
    return execute_search(request);
  }

  [[nodiscard]] auto batch_search(const core::SearchRequest &request) const -> core::Status {
    return execute_search(request);
  }

  [[nodiscard]] auto prepare_mutation(const core::OpaqueOperationRequest &request,
                                      core::MutationContext &,
                                      core::MutationToken &token) -> core::Status {
    if (request.payload == nullptr) {
      return malformed_mutation("canonical Flat mutation payload is missing");
    }
    Transaction transaction;
    std::uint64_t transaction_id{};
    if (request.payload_size == sizeof(SegmentMutationBundlePayload)) {
      const auto &bundle = *static_cast<const SegmentMutationBundlePayload *>(request.payload);
      if (!core::is_current_struct(bundle) || bundle.rows.empty()) {
        return malformed_mutation("canonical Flat mutation bundle is invalid");
      }
      transaction_id = bundle.batch_op_id;
      transaction.rows.reserve(bundle.rows.size());
      for (const auto &payload : bundle.rows) {
        auto row = copy_row(payload);
        if (!row.ok()) {
          return row.status();
        }
        transaction.rows.push_back(std::move(row).value());
      }
    } else if (request.payload_size == sizeof(SegmentMutationPayload)) {
      const auto &payload = *static_cast<const SegmentMutationPayload *>(request.payload);
      transaction_id = payload.op_id;
      auto row = copy_row(payload);
      if (!row.ok()) {
        return row.status();
      }
      transaction.rows.push_back(std::move(row).value());
    } else {
      return malformed_mutation("canonical Flat mutation payload size is invalid");
    }
    if (transaction_id == 0) {
      return malformed_mutation("canonical Flat mutation transaction ID is zero");
    }
    {
      std::lock_guard lock(mutex_);
      transactions_.insert_or_assign(transaction_id, std::move(transaction));
    }
    token.value = transaction_id;
    return core::Status::success();
  }

  [[nodiscard]] auto stage_mutation(core::MutationToken &token, core::MutationContext &)
      -> core::Status {
    std::lock_guard lock(mutex_);
    const auto found = transactions_.find(token.value);
    if (found == transactions_.end()) {
      return failure(core::OperationStage::mutation_stage,
                     core::StatusCode::internal,
                     core::StatusDetail::malformed_struct,
                     "canonical Flat mutation token is unknown");
    }
    found->second.staged = true;
    return core::Status::success();
  }

  [[nodiscard]] auto publish_mutation(core::MutationToken &token, core::MutationContext &)
      -> core::Status {
    std::lock_guard lock(mutex_);
    const auto found = transactions_.find(token.value);
    if (found == transactions_.end() || !found->second.staged) {
      return failure(core::OperationStage::mutation_publish,
                     core::StatusCode::internal,
                     core::StatusDetail::malformed_struct,
                     "canonical Flat mutation was not staged");
    }
    apply_rows_locked(found->second.rows);
    transactions_.erase(found);
    return core::Status::success();
  }

  [[nodiscard]] auto abort_mutation(core::MutationToken &token, core::MutationContext &)
      -> core::Status {
    std::lock_guard lock(mutex_);
    transactions_.erase(token.value);
    return core::Status::success();
  }

  [[nodiscard]] auto replay_mutation(const core::OpaqueOperationRequest &request,
                                     core::MutationContext &) -> core::Status {
    if (request.payload == nullptr) {
      return malformed_mutation("canonical Flat replay payload is missing");
    }
    std::vector<OwnedRow> rows;
    std::uint64_t transaction_id{};
    if (request.payload_size == sizeof(SegmentMutationBundlePayload)) {
      const auto &bundle = *static_cast<const SegmentMutationBundlePayload *>(request.payload);
      if (!core::is_current_struct(bundle) || bundle.rows.empty()) {
        return malformed_mutation("canonical Flat replay bundle is invalid");
      }
      transaction_id = bundle.batch_op_id;
      rows.reserve(bundle.rows.size());
      for (const auto &payload : bundle.rows) {
        auto row = copy_row(payload);
        if (!row.ok()) {
          return row.status();
        }
        rows.push_back(std::move(row).value());
      }
    } else if (request.payload_size == sizeof(SegmentMutationPayload)) {
      const auto &payload = *static_cast<const SegmentMutationPayload *>(request.payload);
      transaction_id = payload.op_id;
      auto row = copy_row(payload);
      if (!row.ok()) {
        return row.status();
      }
      rows.push_back(std::move(row).value());
    } else {
      return malformed_mutation("canonical Flat replay payload size is invalid");
    }
    std::lock_guard lock(mutex_);
    apply_rows_locked(rows);
    transactions_.erase(transaction_id);
    return core::Status::success();
  }

  [[nodiscard]] auto checkpoint(core::CheckpointContext &, core::CheckpointToken &token)
      -> core::Status {
    std::lock_guard lock(mutex_);
    if (!transactions_.empty()) {
      return failure(core::OperationStage::checkpoint,
                     core::StatusCode::conflict,
                     core::StatusDetail::none,
                     "canonical Flat checkpoint observed a staged transaction");
    }
    token.value = applied_watermark_;
    return core::Status::success();
  }

  [[nodiscard]] auto stats(core::SegmentStats &stats) const noexcept -> core::Status {
    std::lock_guard lock(mutex_);
    stats = core::SegmentStats{};
    stats.snapshot_version = applied_watermark_;
    stats.live_rows = rows_.size();
    stats.allocated_rows = rows_.size();
    stats.pending_rows = transactions_.size();
    stats.health = core::SegmentHealth::healthy;
    return core::Status::success();
  }

 private:
  struct PublishedRow {
    OwnedVector vector{};
    std::uint64_t sequence{};
  };

  struct OwnedRow {
    SegmentMutationAction action{SegmentMutationAction::write};
    std::uint64_t op_id{};
    std::uint64_t upsert_sequence{};
    RowAddress target{};
    std::optional<RowAddress> previous{};
    std::optional<OwnedVector> vector{};
  };

  struct Transaction {
    std::vector<OwnedRow> rows{};
    bool staged{};
  };

  struct ScoredRow {
    std::uint64_t row_id{};
    std::uint64_t sequence{};
    float score{};
  };

  [[nodiscard]] static auto failure(core::OperationStage stage,
                                    core::StatusCode code,
                                    core::StatusDetail detail,
                                    std::string diagnostic) -> core::Status {
    return core::Status::error(code, stage, detail, std::move(diagnostic));
  }

  [[nodiscard]] static auto malformed_mutation(std::string diagnostic) -> core::Status {
    return failure(core::OperationStage::mutation_prepare,
                   core::StatusCode::invalid_argument,
                   core::StatusDetail::malformed_struct,
                   std::move(diagnostic));
  }

  [[nodiscard]] auto copy_row(const SegmentMutationPayload &payload) const
      -> core::Result<OwnedRow> {
    if (!core::is_current_struct(payload) || payload.op_id == 0 ||
        payload.target.segment_id != segment_id_ || payload.target.generation != generation_) {
      return malformed_mutation("canonical Flat mutation row identity is invalid");
    }
    OwnedRow result;
    result.action = payload.action;
    result.op_id = payload.op_id;
    result.upsert_sequence = payload.upsert_sequence;
    result.target = payload.target;
    result.previous = payload.previous;
    if (payload.action == SegmentMutationAction::write) {
      auto status = core::validate_tensor(payload.vector,
                                          schema_.dim,
                                          core::OperationStage::mutation_prepare);
      if (!status.ok()) {
        return status;
      }
      if (payload.vector.rows != 1 || payload.vector.scalar_type != schema_.scalar_type) {
        return malformed_mutation("canonical Flat mutation vector does not match the schema");
      }
      auto copied = OwnedVector::copy_row(payload.vector, 0);
      if (!copied.ok()) {
        return copied.status();
      }
      result.vector = std::move(copied).value();
    }
    return result;
  }

  void apply_rows_locked(const std::vector<OwnedRow> &mutation_rows) {
    for (const auto &row : mutation_rows) {
      if (applied_ops_.contains(row.op_id)) {
        continue;
      }
      if (row.previous.has_value() && row.previous->segment_id == segment_id_ &&
          row.previous->generation == generation_) {
        rows_.erase(static_cast<std::uint64_t>(row.previous->row_id));
      }
      if (row.action == SegmentMutationAction::write && row.vector.has_value()) {
        rows_.insert_or_assign(static_cast<std::uint64_t>(row.target.row_id),
                               PublishedRow{*row.vector, row.upsert_sequence});
      }
      applied_ops_.insert(row.op_id);
      applied_watermark_ = std::max(applied_watermark_, row.op_id);
    }
  }

  template <class T>
  [[nodiscard]] static auto distance_typed(const core::TypedTensorView &queries,
                                           core::RowCount query_index,
                                           const OwnedVector &stored,
                                           core::Metric metric) -> float {
    const auto *left = queries.row<T>(query_index);
    const auto *right = stored.view().row<T>(0);
    double dot{};
    double left_norm{};
    double right_norm{};
    double l2{};
    for (std::uint32_t index = 0; index < queries.dim; ++index) {
      const auto lhs = static_cast<double>(left[index]);
      const auto rhs = static_cast<double>(right[index]);
      const auto difference = lhs - rhs;
      l2 += difference * difference;
      dot += lhs * rhs;
      left_norm += lhs * lhs;
      right_norm += rhs * rhs;
    }
    if (metric == core::Metric::l2) {
      return static_cast<float>(l2);
    }
    if (metric == core::Metric::inner_product) {
      return static_cast<float>(-dot);
    }
    if (left_norm == 0 || right_norm == 0) {
      return 0.0F;
    }
    return static_cast<float>(-dot / std::sqrt(left_norm * right_norm));
  }

  [[nodiscard]] auto distance(const core::TypedTensorView &queries,
                              core::RowCount query_index,
                              const OwnedVector &stored) const -> float {
    switch (schema_.scalar_type) {
      case core::ScalarType::float32:
        return distance_typed<float>(queries, query_index, stored, schema_.metric);
      case core::ScalarType::int8:
        return distance_typed<std::int8_t>(queries, query_index, stored, schema_.metric);
      case core::ScalarType::uint8:
        return distance_typed<std::uint8_t>(queries, query_index, stored, schema_.metric);
    }
    return std::numeric_limits<float>::quiet_NaN();
  }

  [[nodiscard]] auto execute_search(const core::SearchRequest &request) const -> core::Status {
    if (request.context == nullptr || request.response == nullptr) {
      return failure(core::OperationStage::validation,
                     core::StatusCode::invalid_argument,
                     core::StatusDetail::malformed_struct,
                     "canonical Flat search request is incomplete");
    }
    auto status =
        core::validate_tensor(request.queries, schema_.dim, core::OperationStage::validation);
    if (!status.ok()) {
      return status;
    }
    if (request.queries.scalar_type != schema_.scalar_type) {
      return failure(core::OperationStage::validation,
                     core::StatusCode::not_supported,
                     core::StatusDetail::unsupported_scalar_type,
                     "canonical Flat does not implicitly convert query scalar types");
    }
    status = core::validate_response(*request.response,
                                     request.queries.rows,
                                     request.options.top_k,
                                     core::OperationStage::validation);
    if (!status.ok()) {
      return status;
    }
    status = core::validate_runtime_control(request.context->deadline,
                                            request.context->cancellation,
                                            core::OperationStage::search);
    if (!status.ok()) {
      return status;
    }

    std::map<std::uint64_t, PublishedRow> rows;
    {
      std::lock_guard lock(mutex_);
      rows = rows_;
    }
    auto &response = *request.response;
    response.query_count = request.queries.rows;
    response.score_kind = core::ScoreKind::distance;
    response.comparable_metric = schema_.metric;
    response.result_flags = core::ResultFlag::exact_reranked;
    response.offsets[0] = 0;
    core::RowCount cursor{};
    for (core::RowCount query_index = 0; query_index < request.queries.rows; ++query_index) {
      std::vector<ScoredRow> scored;
      scored.reserve(rows.size());
      for (const auto &[row_id, row] : rows) {
        scored.push_back(
            {row_id, row.sequence, distance(request.queries, query_index, row.vector)});
      }
      std::sort(scored.begin(), scored.end(), [](const ScoredRow &left, const ScoredRow &right) {
        return left.score != right.score ? left.score < right.score : left.row_id < right.row_id;
      });
      const auto count = std::min<std::uint64_t>(scored.size(), request.options.top_k);
      for (std::uint64_t index = 0; index < count; ++index) {
        auto hit = core::SearchHit(core::SegmentRowId(scored[index].row_id),
                                   scored[index].score,
                                   core::ScoreKind::distance,
                                   schema_.metric,
                                   core::ResultFlag::exact_reranked);
        hit.row_version = scored[index].sequence;
        response.hits[static_cast<std::size_t>(cursor++)] = hit;
      }
      const auto response_index = static_cast<std::size_t>(query_index);
      response.offsets[response_index + 1] = cursor;
      response.valid_counts[response_index] = count;
      response.statuses[response_index] = core::Status::success();
      response.completeness[response_index] = count == request.options.top_k
                                                  ? core::SearchCompleteness::complete_k
                                                  : core::SearchCompleteness::eligible_exhausted;
    }
    return core::Status::success();
  }

  CollectionSchema schema_{};
  std::uint64_t segment_id_{};
  std::uint64_t generation_{};
  mutable std::mutex mutex_{};
  std::map<std::uint64_t, PublishedRow> rows_{};
  std::map<std::uint64_t, Transaction> transactions_{};
  std::set<std::uint64_t> applied_ops_{};
  std::uint64_t applied_watermark_{};
};

[[nodiscard]] inline auto make_canonical_flat_registration(CollectionSchema schema,
                                                           std::uint64_t segment_id,
                                                           std::uint64_t generation)
    -> core::Result<SegmentRegistration> {
  auto segment = std::make_shared<CanonicalFlatSegment>(schema, segment_id, generation);
  core::SegmentInstanceConfig config;
  config.readonly = false;
  config.concurrency.reentrant_search = true;
  config.concurrency.search_with_stage = true;
  config.concurrency.search_with_publish = true;
  config.concurrency.serial_mutation = true;
  config.concurrency.checkpoint_with_search = false;
  config.concurrency.native_async = false;
  config.concurrency.cooperative_cancel = false;
  config.concurrency.explicit_drain = false;
  auto erased = core::AnySegment::from_sync(std::move(segment), std::move(config));
  if (!erased.ok()) {
    return erased.status();
  }
  SegmentRegistration registration;
  registration.segment_id = segment_id;
  registration.generation = generation;
  registration.role = SegmentRole::active_mutable;
  registration.segment = std::move(erased).value();
  registration.atomic_mutation_bundle = true;
  return registration;
}

}  // namespace alaya::internal::collection::detail
