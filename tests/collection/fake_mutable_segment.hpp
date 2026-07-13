// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "index/collection/segmented_collection.hpp"

namespace alaya::internal::collection::test {

class Barrier {
 public:
  void enable() {
    std::lock_guard lock(mutex_);
    enabled_ = true;
    entered_ = false;
    released_ = false;
  }

  void arrive_and_wait() {
    std::unique_lock lock(mutex_);
    if (!enabled_) {
      return;
    }
    entered_ = true;
    changed_.notify_all();
    changed_.wait(lock, [&] {
      return released_;
    });
    enabled_ = false;
  }

  [[nodiscard]] auto wait_until_entered() -> bool {
    std::unique_lock lock(mutex_);
    return changed_.wait_for(lock, std::chrono::seconds(5), [&] {
      return entered_;
    });
  }

  void release() {
    std::lock_guard lock(mutex_);
    released_ = true;
    changed_.notify_all();
  }

 private:
  std::mutex mutex_{};
  std::condition_variable changed_{};
  bool enabled_{};
  bool entered_{};
  bool released_{};
};

class FakeMutableSegment {
 public:
  struct PublishedRow {
    std::array<float, 2> vector{};
    std::uint64_t sequence{};
  };

  struct Transaction {
    struct Row {
      SegmentMutationPayload payload{};
      std::array<float, 2> vector{};
    };

    std::vector<Row> rows{};
    bool staged{};
  };

  [[nodiscard]] auto descriptor() const noexcept -> core::Descriptor {
    core::Descriptor descriptor;
    descriptor.algorithm_id = 9001;
    descriptor.format_version = 1;
    descriptor.factory_version = 1;
    descriptor.dim = 2;
    descriptor.metric = core::Metric::l2;
    descriptor.stored_scalar_type = core::ScalarType::float32;
    descriptor.medium = core::Medium::memory;
    descriptor.engine_factory_id = 9001;
    return descriptor;
  }

  [[nodiscard]] auto search(const core::SearchRequest &request) const -> core::Status {
    if (request.queries.rows != 1) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "fake mutable single search requires one row");
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
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::mutation_prepare,
                                 core::StatusDetail::malformed_struct,
                                 "fake mutation payload is missing");
    }
    Transaction transaction;
    std::uint64_t transaction_id{};
    if (request.payload_size == sizeof(SegmentMutationBundlePayload)) {
      const auto &bundle = *static_cast<const SegmentMutationBundlePayload *>(request.payload);
      if (!core::is_current_struct(bundle) || bundle.rows.empty()) {
        return malformed_prepare("fake mutation bundle is invalid");
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
      auto row = copy_row(payload);
      if (!row.ok()) {
        return row.status();
      }
      transaction_id = payload.op_id;
      transaction.rows.push_back(std::move(row).value());
    } else {
      return malformed_prepare("fake mutation payload size is invalid");
    }
    {
      std::lock_guard lock(mutex_);
      transactions_.insert_or_assign(transaction_id, transaction);
      for (const auto &row : transaction.rows) {
        prepared_op_ids_.push_back(row.payload.op_id);
      }
    }
    token.value = transaction_id;
    return core::Status::success();
  }

  [[nodiscard]] auto stage_mutation(core::MutationToken &token, core::MutationContext &)
      -> core::Status {
    const auto active = active_mutations_.fetch_add(1, std::memory_order_acq_rel) + 1;
    auto maximum = maximum_active_mutations_.load(std::memory_order_acquire);
    while (active > maximum &&
           !maximum_active_mutations_.compare_exchange_weak(maximum,
                                                            active,
                                                            std::memory_order_acq_rel)) {
    }
    stage_barrier_.arrive_and_wait();
    if (fail_next_stage_.exchange(false, std::memory_order_acq_rel)) {
      active_mutations_.fetch_sub(1, std::memory_order_acq_rel);
      return core::Status::error(core::StatusCode::internal,
                                 core::OperationStage::mutation_stage,
                                 core::StatusDetail::engine_exception,
                                 "injected fake stage failure");
    }
    {
      std::lock_guard lock(mutex_);
      const auto found = transactions_.find(token.value);
      if (found == transactions_.end()) {
        active_mutations_.fetch_sub(1, std::memory_order_acq_rel);
        return core::Status::error(core::StatusCode::internal,
                                   core::OperationStage::mutation_stage,
                                   core::StatusDetail::malformed_struct,
                                   "fake mutation token is unknown");
      }
      found->second.staged = true;
    }
    return core::Status::success();
  }

  [[nodiscard]] auto publish_mutation(core::MutationToken &token, core::MutationContext &)
      -> core::Status {
    std::lock_guard lock(mutex_);
    const auto found = transactions_.find(token.value);
    if (found == transactions_.end() || !found->second.staged) {
      active_mutations_.fetch_sub(1, std::memory_order_acq_rel);
      return core::Status::error(core::StatusCode::internal,
                                 core::OperationStage::mutation_publish,
                                 core::StatusDetail::malformed_struct,
                                 "fake mutation was not staged");
    }
    if (fail_next_publish_.exchange(false, std::memory_order_acq_rel)) {
      return core::Status::error(core::StatusCode::internal,
                                 core::OperationStage::mutation_publish,
                                 core::StatusDetail::engine_exception,
                                 "injected fake publish failure");
    }
    apply_rows_locked(found->second.rows);
    transactions_.erase(found);
    active_mutations_.fetch_sub(1, std::memory_order_acq_rel);
    return core::Status::success();
  }

  [[nodiscard]] auto abort_mutation(core::MutationToken &token, core::MutationContext &)
      -> core::Status {
    std::lock_guard lock(mutex_);
    const auto found = transactions_.find(token.value);
    if (found != transactions_.end() && found->second.staged) {
      active_mutations_.fetch_sub(1, std::memory_order_acq_rel);
    }
    transactions_.erase(token.value);
    abort_count_.fetch_add(1, std::memory_order_acq_rel);
    return core::Status::success();
  }

  [[nodiscard]] auto replay_mutation(const core::OpaqueOperationRequest &request,
                                     core::MutationContext &) -> core::Status {
    if (request.payload == nullptr) {
      return malformed_prepare("fake replay payload is missing");
    }
    Transaction transaction;
    std::uint64_t transaction_id{};
    if (request.payload_size == sizeof(SegmentMutationBundlePayload)) {
      const auto &bundle = *static_cast<const SegmentMutationBundlePayload *>(request.payload);
      transaction_id = bundle.batch_op_id;
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
      return malformed_prepare("fake replay payload size is invalid");
    }
    std::lock_guard lock(mutex_);
    apply_rows_locked(transaction.rows);
    const auto found = transactions_.find(transaction_id);
    if (found != transactions_.end()) {
      if (found->second.staged) {
        active_mutations_.fetch_sub(1, std::memory_order_acq_rel);
      }
      transactions_.erase(found);
    }
    return core::Status::success();
  }

  [[nodiscard]] auto checkpoint(core::CheckpointContext &, core::CheckpointToken &token)
      -> core::Status {
    std::lock_guard lock(mutex_);
    token.value = published_op_ids_.empty() ? 0 : published_op_ids_.back();
    return core::Status::success();
  }

  [[nodiscard]] auto stats(core::SegmentStats &stats) const noexcept -> core::Status {
    std::lock_guard lock(mutex_);
    stats = core::SegmentStats{};
    stats.snapshot_version = published_op_ids_.empty() ? 0 : published_op_ids_.back();
    stats.live_rows = rows_.size();
    stats.allocated_rows = rows_.size();
    stats.pending_rows = transactions_.size();
    stats.health = core::SegmentHealth::healthy;
    return core::Status::success();
  }

  void fail_next_stage() { fail_next_stage_.store(true, std::memory_order_release); }
  void fail_next_publish() { fail_next_publish_.store(true, std::memory_order_release); }
  void gate_next_stage() { stage_barrier_.enable(); }
  void gate_next_search() { search_barrier_.enable(); }
  [[nodiscard]] auto wait_for_stage() -> bool { return stage_barrier_.wait_until_entered(); }
  [[nodiscard]] auto wait_for_search() -> bool { return search_barrier_.wait_until_entered(); }
  void release_stage() { stage_barrier_.release(); }
  void release_search() { search_barrier_.release(); }
  [[nodiscard]] auto abort_count() const -> std::uint64_t {
    return abort_count_.load(std::memory_order_acquire);
  }
  [[nodiscard]] auto maximum_active_mutations() const -> std::uint64_t {
    return maximum_active_mutations_.load(std::memory_order_acquire);
  }
  [[nodiscard]] auto prepared_op_ids() const -> std::vector<std::uint64_t> {
    std::lock_guard lock(mutex_);
    return prepared_op_ids_;
  }
  [[nodiscard]] auto published_op_ids() const -> std::vector<std::uint64_t> {
    std::lock_guard lock(mutex_);
    return published_op_ids_;
  }

  static constexpr std::uint64_t kSegmentId = 2;

 private:
  [[nodiscard]] static auto malformed_prepare(std::string diagnostic) -> core::Status {
    return core::Status::error(core::StatusCode::invalid_argument,
                               core::OperationStage::mutation_prepare,
                               core::StatusDetail::malformed_struct,
                               std::move(diagnostic));
  }

  [[nodiscard]] static auto copy_row(const SegmentMutationPayload &payload)
      -> core::Result<Transaction::Row> {
    if (!core::is_current_struct(payload)) {
      return malformed_prepare("fake mutation row payload is incompatible");
    }
    Transaction::Row row;
    row.payload = payload;
    if (payload.action == SegmentMutationAction::write) {
      if (payload.vector.data == nullptr || payload.vector.rows != 1 || payload.vector.dim != 2 ||
          payload.vector.scalar_type != core::ScalarType::float32) {
        return malformed_prepare("fake write payload tensor is invalid");
      }
      const auto *values = payload.vector.row<float>(0);
      row.vector = {values[0], values[1]};
    }
    return row;
  }

  void apply_rows_locked(const std::vector<Transaction::Row> &transaction_rows) {
    for (const auto &row : transaction_rows) {
      const auto &payload = row.payload;
      if (published_op_id_set_.contains(payload.op_id)) {
        continue;
      }
      if (payload.previous.has_value() && payload.previous->segment_id == kSegmentId) {
        rows_.erase(static_cast<std::uint64_t>(payload.previous->row_id));
      }
      if (payload.action == SegmentMutationAction::write) {
        rows_.insert_or_assign(static_cast<std::uint64_t>(payload.target.row_id),
                               PublishedRow{row.vector, payload.upsert_sequence});
      }
      published_op_ids_.push_back(payload.op_id);
      published_op_id_set_.insert(payload.op_id);
    }
  }

  [[nodiscard]] auto execute_search(const core::SearchRequest &request) const -> core::Status {
    search_barrier_.arrive_and_wait();
    std::map<std::uint64_t, PublishedRow> rows;
    {
      std::lock_guard lock(mutex_);
      rows = rows_;
    }
    auto &response = *request.response;
    response.query_count = request.queries.rows;
    response.score_kind = core::ScoreKind::distance;
    response.comparable_metric = core::Metric::l2;
    response.result_flags = core::ResultFlag::approximate;
    response.offsets[0] = 0;
    core::RowCount cursor{};
    for (core::RowCount query_index = 0; query_index < request.queries.rows; ++query_index) {
      const auto *query = request.queries.row<float>(query_index);
      struct Scored {
        std::uint64_t row{};
        float distance{};
      };
      std::vector<Scored> scored;
      scored.reserve(rows.size());
      for (const auto &[row, value] : rows) {
        const auto first = query[0] - value.vector[0];
        const auto second = query[1] - value.vector[1];
        scored.push_back({row, first * first + second * second});
      }
      std::sort(scored.begin(), scored.end(), [](const Scored &lhs, const Scored &rhs) {
        return lhs.distance != rhs.distance ? lhs.distance < rhs.distance : lhs.row < rhs.row;
      });
      const auto count = std::min<std::size_t>(scored.size(), request.options.top_k);
      for (std::size_t index = 0; index < count; ++index) {
        response.hits[static_cast<std::size_t>(cursor)] =
            core::SearchHit(core::SegmentRowId(scored[index].row),
                            scored[index].distance,
                            core::ScoreKind::distance,
                            core::Metric::l2,
                            core::ResultFlag::approximate);
        ++cursor;
      }
      response.offsets[static_cast<std::size_t>(query_index + 1)] = cursor;
      response.valid_counts[static_cast<std::size_t>(query_index)] = count;
      response.statuses[static_cast<std::size_t>(query_index)] = core::Status::success();
      response.completeness[static_cast<std::size_t>(query_index)] =
          count == request.options.top_k ? core::SearchCompleteness::complete_k
                                         : core::SearchCompleteness::eligible_exhausted;
    }
    return core::Status::success();
  }

  mutable std::mutex mutex_{};
  mutable Barrier search_barrier_{};
  Barrier stage_barrier_{};
  std::map<std::uint64_t, PublishedRow> rows_{};
  std::map<std::uint64_t, Transaction> transactions_{};
  std::vector<std::uint64_t> prepared_op_ids_{};
  std::vector<std::uint64_t> published_op_ids_{};
  std::set<std::uint64_t> published_op_id_set_{};
  std::atomic_bool fail_next_stage_{};
  std::atomic_bool fail_next_publish_{};
  std::atomic_uint64_t abort_count_{};
  std::atomic_uint64_t active_mutations_{};
  std::atomic_uint64_t maximum_active_mutations_{};
};

[[nodiscard]] inline auto make_fake_mutable_any(const std::shared_ptr<FakeMutableSegment> &producer)
    -> core::Result<core::AnySegment> {
  core::SegmentInstanceConfig config;
  config.readonly = false;
  config.concurrency.reentrant_search = true;
  config.concurrency.search_with_stage = true;
  config.concurrency.search_with_publish = true;
  config.concurrency.serial_mutation = true;
  config.concurrency.native_async = false;
  config.concurrency.cooperative_cancel = false;
  config.concurrency.explicit_drain = false;
  return core::AnySegment::from_sync(producer, std::move(config));
}

}  // namespace alaya::internal::collection::test
