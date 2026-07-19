// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// MutableLaserCollectionAdapter: the facade-private active-mutation target that
// wires a durable on-disk mutable LASER segment (disk::MutableLaserSegment, the
// G1 op-WAL handle) into the Collection as its writable active generation. It is
// the LASER counterpart of CanonicalFlatSegment, but the physical rows live on
// disk in a RaBitQ graph rather than an in-memory exact-search table.
//
// Correctness spine (2B rulings / codex BLOCKERs):
//  - Idempotency (B-01/B-02): the physical transaction id and the transaction's
//    max row op_id arrive EXPLICITLY through MutationContext (set by Collection at
//    its three dispatch sites); never re-derived from the token or the payload
//    batch_op_id. A write is skipped iff max_row_op_id <= applied_collection_op_id;
//    otherwise it requires txid > last_committed_txid, else it is WAL corruption.
//  - tombstone-plan-includes-previous (B-05): every row's same-segment previous is
//    tombstoned (deduped) after the write bundle, plus explicit erase targets, so a
//    superseded version cannot stay live in the graph.
//  - failure latch (B-04): any failure once physical state may have advanced
//    latches the adapter and gates search / batch_search / prepare / stage /
//    publish / checkpoint / stats until reopen; only abort / close / drain pass.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "core/algorithm_registry.hpp"
#include "index/collection/mutation_wal_codec.hpp"
#include "index/collection/types.hpp"
#include "index/disk/laser_segment.hpp"
#include "index/disk/mutable_laser_segment.hpp"

namespace alaya::internal::collection::detail {

class MutableLaserCollectionAdapter {
 public:
  MutableLaserCollectionAdapter(std::shared_ptr<::alaya::disk::MutableLaserSegment> segment,
                                CollectionSchema schema,
                                std::uint64_t segment_id,
                                std::uint64_t generation)
      : segment_(std::move(segment)),
        schema_(schema),
        segment_id_(segment_id),
        generation_(generation) {}

  // ---- DescriptorProvider -------------------------------------------------
  [[nodiscard]] auto descriptor() const noexcept -> core::Descriptor {
    core::Descriptor descriptor;
    descriptor.algorithm_id = core::algorithm::laser;
    descriptor.format_version = 2;  // v2 op-WAL superblock
    descriptor.factory_version = 1;
    descriptor.dim = schema_.dim;
    descriptor.metric = schema_.metric;
    descriptor.stored_scalar_type = schema_.scalar_type;
    descriptor.medium = core::Medium::disk;
    descriptor.preprocessing = core::MetricPreprocessing::engine_quantized;  // RaBitQ
    descriptor.engine_factory_id = core::algorithm::laser;
    return descriptor;
  }

  // ---- Searchable / BatchSearchable --------------------------------------
  [[nodiscard]] auto search(const core::SearchRequest &request) const -> core::Status {
    if (auto gate = ensure_live(core::OperationStage::search); !gate.ok()) {
      return gate;
    }
    if (request.queries.rows != 1) {
      return failure(core::OperationStage::validation,
                     core::StatusCode::invalid_argument,
                     core::StatusDetail::malformed_struct,
                     "active LASER single search requires exactly one query row");
    }
    return execute_search(request);
  }

  [[nodiscard]] auto batch_search(const core::SearchRequest &request) const -> core::Status {
    if (auto gate = ensure_live(core::OperationStage::search); !gate.ok()) {
      return gate;
    }
    return execute_search(request);
  }

  // ---- Mutable ------------------------------------------------------------
  [[nodiscard]] auto prepare_mutation(const core::OpaqueOperationRequest &request,
                                      core::MutationContext &context,
                                      core::MutationToken &token) -> core::Status {
    if (auto gate = ensure_live(core::OperationStage::mutation_prepare); !gate.ok()) {
      return gate;
    }
    // Ruling 11 / codex non-blocking 3: enforce the wal_fsync-only durability tier
    // BEFORE creating any pending state, so a rejected durability leaves nothing for
    // Collection to abort (a leaked pending would otherwise ride the next fsync).
    if (auto gate = require_durable(context); !gate.ok()) {
      return gate;
    }
    try {
      Pending pending;
      auto status = decode_transaction(request, pending.rows);
      if (!status.ok()) {
        return status;
      }
      // B-01/B-02: the physical txid and idempotency watermark are TYPED context
      // fields set by Collection; the adapter never guesses them.
      pending.txid = context.transaction_id;
      pending.max_row_op_id = context.max_row_op_id;
      if (pending.txid == 0 || pending.max_row_op_id == 0) {
        return failure(core::OperationStage::mutation_prepare,
                       core::StatusCode::invalid_argument,
                       core::StatusDetail::malformed_struct,
                       "active LASER mutation context carries no physical transaction id");
      }
      const std::lock_guard<std::mutex> lock(mutex_);
      if (maintenance_active_) {
        return failure(core::OperationStage::mutation_prepare,
                       core::StatusCode::conflict,
                       core::StatusDetail::none,
                       "active LASER mutation conflicts with consolidate maintenance");
      }
      transactions_.insert_or_assign(pending.txid, std::move(pending));
      token.value = context.transaction_id;
      return core::Status::success();
    } catch (...) {
      return latch(core::status_from_exception(core::OperationStage::mutation_prepare));
    }
  }

  [[nodiscard]] auto stage_mutation(core::MutationToken &token, core::MutationContext &)
      -> core::Status {
    if (auto gate = ensure_live(core::OperationStage::mutation_stage); !gate.ok()) {
      return gate;
    }
    const std::lock_guard<std::mutex> lock(mutex_);
    if (maintenance_active_) {
      return latch(failure(core::OperationStage::mutation_stage,
                           core::StatusCode::internal,
                           core::StatusDetail::readonly_instance,
                           "active LASER staged mutation interleaved with maintenance"));
    }
    const auto found = transactions_.find(token.value);
    if (found == transactions_.end()) {
      return failure(core::OperationStage::mutation_stage,
                     core::StatusCode::internal,
                     core::StatusDetail::malformed_struct,
                     "active LASER mutation token is unknown at stage");
    }
    found->second.staged = true;
    return core::Status::success();
  }

  [[nodiscard]] auto publish_mutation(core::MutationToken &token, core::MutationContext &)
      -> core::Status {
    if (auto gate = ensure_live(core::OperationStage::mutation_publish); !gate.ok()) {
      return gate;
    }
    Pending pending;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (maintenance_active_) {
        return latch(failure(core::OperationStage::mutation_publish,
                             core::StatusCode::internal,
                             core::StatusDetail::readonly_instance,
                             "active LASER publish interleaved with maintenance"));
      }
      const auto found = transactions_.find(token.value);
      if (found == transactions_.end() || !found->second.staged) {
        return failure(core::OperationStage::mutation_publish,
                       core::StatusCode::internal,
                       core::StatusDetail::malformed_struct,
                       "active LASER mutation was not staged");
      }
      pending = std::move(found->second);
      transactions_.erase(found);
    }
    publish_gate_.wait();
    // Post-COMMIT: any failure from here latches (B-04). The pending entry is
    // already removed, so recovery replays this transaction through the WAL.
    return apply_transaction(pending, /*is_replay=*/false);
  }

  [[nodiscard]] auto abort_mutation(core::MutationToken &token, core::MutationContext &)
      -> core::Status {
    // abort always proceeds (even latched): it only drops an un-published pending.
    const std::lock_guard<std::mutex> lock(mutex_);
    transactions_.erase(token.value);
    return core::Status::success();
  }

  [[nodiscard]] auto replay_mutation(const core::OpaqueOperationRequest &request,
                                     core::MutationContext &context) -> core::Status {
    try {
      Pending pending;
      auto status = decode_transaction(request, pending.rows);
      if (!status.ok()) {
        return status;
      }
      pending.txid = context.transaction_id;
      pending.max_row_op_id = context.max_row_op_id;
      if (pending.txid == 0 || pending.max_row_op_id == 0) {
        return failure(core::OperationStage::mutation_replay,
                       core::StatusCode::corruption,
                       core::StatusDetail::malformed_struct,
                       "active LASER replay context carries no physical transaction id");
      }
      return apply_transaction(pending, /*is_replay=*/true);
    } catch (...) {
      return latch(core::status_from_exception(core::OperationStage::mutation_replay));
    }
  }

  // ---- Checkpointable -----------------------------------------------------
  [[nodiscard]] auto checkpoint(core::CheckpointContext &, core::CheckpointToken &token)
      -> core::Status {
    if (auto gate = ensure_live(core::OperationStage::checkpoint); !gate.ok()) {
      return gate;
    }
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (maintenance_active_ || !transactions_.empty()) {
        return failure(core::OperationStage::checkpoint,
                       core::StatusCode::conflict,
                       core::StatusDetail::none,
                       "active LASER checkpoint conflicts with maintenance or a staged "
                       "transaction");
      }
    }
    try {
      segment_->checkpoint();
    } catch (...) {
      return latch(core::status_from_exception(core::OperationStage::checkpoint));
    }
    token.value = segment_->applied_collection_op_id();
    return core::Status::success();
  }

  [[nodiscard]] auto consolidate(std::size_t num_threads,
                                 std::size_t r_target,
                                 bool reclaim_slots,
                                 bool bloom_consolidate) -> core::Status {
    if (auto gate = ensure_live(core::OperationStage::checkpoint); !gate.ok()) {
      return gate;
    }
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (failed_.load(std::memory_order_acquire)) {
        return ensure_live(core::OperationStage::checkpoint);
      }
      if (maintenance_active_ || !transactions_.empty()) {
        return failure(core::OperationStage::checkpoint,
                       core::StatusCode::conflict,
                       core::StatusDetail::none,
                       "active LASER consolidate conflicts with maintenance or a staged "
                       "transaction");
      }
      maintenance_active_ = true;
    }

    const auto clear_active = [this] {
      const std::lock_guard<std::mutex> lock(mutex_);
      maintenance_active_ = false;
    };
    try {
      segment_->consolidate(num_threads, r_target, reclaim_slots, bloom_consolidate);
      clear_active();
      return core::Status::success();
    } catch (...) {
      const bool recovery = segment_->recovery_required();
      clear_active();
      auto status = core::status_from_exception(core::OperationStage::checkpoint);
      return recovery ? latch(std::move(status)) : status;
    }
  }

  // ---- StatsProvider ------------------------------------------------------
  [[nodiscard]] auto stats(core::SegmentStats &stats) const noexcept -> core::Status {
    if (failed_.load(std::memory_order_acquire)) {
      return failure(core::OperationStage::validation,
                     core::StatusCode::internal,
                     core::StatusDetail::readonly_instance,
                     "active LASER segment latched (recovery-required)");
    }
    try {
      const auto committed = static_cast<core::RowCount>(segment_->size());
      const auto live = static_cast<core::RowCount>(segment_->live_count());
      stats = core::SegmentStats{};
      stats.snapshot_version = segment_->applied_collection_op_id();
      stats.live_rows = live;
      stats.allocated_rows = static_cast<core::RowCount>(segment_->allocated_count());
      stats.tombstone_rows = committed >= live ? committed - live : 0;
      {
        const std::lock_guard<std::mutex> lock(mutex_);
        stats.pending_rows = static_cast<core::RowCount>(transactions_.size());
      }
      stats.health = core::SegmentHealth::healthy;
      return core::Status::success();
    } catch (...) {
      return failure(core::OperationStage::validation,
                     core::StatusCode::internal,
                     core::StatusDetail::engine_exception,
                     "active LASER stats failed");
    }
  }

  // ---- Closable -----------------------------------------------------------
  [[nodiscard]] auto close() -> core::Status { return core::Status::success(); }
  [[nodiscard]] auto drain(const core::Deadline &) -> core::Status {
    return core::Status::success();
  }

  // ---- Fault injection (real seams inside the adapter, W1/W3) -------------
  // Latch the next publish right after the physical write bundle commits, before
  // tombstones -- exercises B-04's "post-commit failure gates search/checkpoint".
  void fail_next_publish() { fail_next_publish_.store(true, std::memory_order_release); }
  // Latch on the k-th (0-based) tombstone AFTER the write bundle committed --
  // exercises B-04's partial-tombstone-failure divergence.
  void fail_tombstone_at(int k) { fail_tombstone_at_.store(k, std::memory_order_release); }
  // Block the next publish inside apply until release_publish(); lets a test drive
  // a concurrent search/checkpoint against an in-flight publish.
  void gate_next_publish() { publish_gate_.arm(); }
  void release_publish() { publish_gate_.release(); }
  [[nodiscard]] auto is_latched() const -> bool { return failed_.load(std::memory_order_acquire); }
  [[nodiscard]] auto recovery_required() const noexcept -> bool {
    return failed_.load(std::memory_order_acquire) || segment_->recovery_required();
  }

 private:
  struct Row {
    SegmentMutationAction action{SegmentMutationAction::write};
    std::uint64_t op_id{};
    std::uint64_t target_label{};
    std::optional<std::uint64_t> previous_label{};
    std::vector<float> vector{};
  };

  struct Pending {
    std::vector<Row> rows{};
    std::uint64_t txid{};
    std::uint64_t max_row_op_id{};
    bool staged{};
  };

  // A one-shot barrier for concurrency fault injection (off by default).
  class Gate {
   public:
    void arm() {
      const std::lock_guard<std::mutex> lock(mutex_);
      armed_ = true;
      released_ = false;
    }
    void wait() {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!armed_) {
        return;
      }
      changed_.wait(lock, [&] {
        return released_;
      });
      armed_ = false;
    }
    void release() {
      const std::lock_guard<std::mutex> lock(mutex_);
      released_ = true;
      changed_.notify_all();
    }

   private:
    std::mutex mutex_{};
    std::condition_variable changed_{};
    bool armed_{};
    bool released_{};
  };

  [[nodiscard]] static auto failure(core::OperationStage stage,
                                    core::StatusCode code,
                                    core::StatusDetail detail,
                                    std::string diagnostic) -> core::Status {
    return core::Status::error(code, stage, detail, std::move(diagnostic));
  }

  // The physical txid + max op arrive TYPED via MutationContext (B-01); the raw
  // token is retained ONLY as the opaque handle back to Collection's
  // WalMutationTransaction, so the adapter can read its durability tier here. This
  // is the sole dereference of the token (never to guess an id). A null token means
  // a caller that did not route through Collection's mutation path.
  [[nodiscard]] static auto require_durable(const core::MutationContext &context) -> core::Status {
    static_assert(WriteDurability::wal_fsync == static_cast<WriteDurability>(1),
                  "durability gate assumes the wal_fsync tier ordinal is stable");
    const auto *transaction =
        static_cast<const WalMutationTransaction *>(context.transaction_token);
    if (transaction == nullptr || transaction->durability != WriteDurability::wal_fsync) {
      return failure(core::OperationStage::mutation_prepare,
                     core::StatusCode::not_supported,
                     core::StatusDetail::readonly_instance,
                     "active LASER segment supports only the wal_fsync durability tier");
    }
    return core::Status::success();
  }

  // Latch the adapter and return the failing status; every post-commit failure path
  // funnels through here so search/checkpoint see a fail-closed segment (B-04).
  [[nodiscard]] auto latch(core::Status status) -> core::Status {
    failed_.store(true, std::memory_order_release);
    {
      const std::lock_guard<std::mutex> lock(diag_mutex_);
      if (diagnostic_.empty()) {
        diagnostic_ = status.diagnostic();
      }
    }
    return status;
  }

  [[nodiscard]] auto ensure_live(core::OperationStage stage) const -> core::Status {
    if (!failed_.load(std::memory_order_acquire)) {
      return core::Status::success();
    }
    std::string diagnostic;
    {
      const std::lock_guard<std::mutex> lock(diag_mutex_);
      diagnostic = diagnostic_;
    }
    return failure(stage,
                   core::StatusCode::internal,
                   core::StatusDetail::readonly_instance,
                   "active LASER segment latched (recovery-required): " + diagnostic);
  }

  [[nodiscard]] auto decode_row(const SegmentMutationPayload &payload, Row &row) const
      -> core::Status {
    if (!core::is_current_struct(payload) || payload.op_id == 0 ||
        payload.target.segment_id != segment_id_ || payload.target.generation != generation_) {
      return failure(core::OperationStage::mutation_prepare,
                     core::StatusCode::invalid_argument,
                     core::StatusDetail::malformed_struct,
                     "active LASER mutation row identity is invalid");
    }
    row.action = payload.action;
    row.op_id = payload.op_id;
    row.target_label = static_cast<std::uint64_t>(payload.target.row_id);
    if (payload.previous.has_value() && payload.previous->segment_id == segment_id_ &&
        payload.previous->generation == generation_) {
      row.previous_label = static_cast<std::uint64_t>(payload.previous->row_id);
    }
    if (payload.action == SegmentMutationAction::write) {
      auto status = core::validate_tensor(payload.vector,
                                          schema_.dim,
                                          core::OperationStage::mutation_prepare);
      if (!status.ok()) {
        return status;
      }
      if (payload.vector.rows != 1 || payload.vector.scalar_type != core::ScalarType::float32) {
        return failure(core::OperationStage::mutation_prepare,
                       core::StatusCode::not_supported,
                       core::StatusDetail::unsupported_scalar_type,
                       "active LASER requires a single float32 vector per write row");
      }
      const auto *values = payload.vector.row<float>(0);
      row.vector.assign(values, values + schema_.dim);
    }
    return core::Status::success();
  }

  [[nodiscard]] auto decode_transaction(const core::OpaqueOperationRequest &request,
                                        std::vector<Row> &rows) const -> core::Status {
    if (request.payload == nullptr) {
      return failure(core::OperationStage::mutation_prepare,
                     core::StatusCode::invalid_argument,
                     core::StatusDetail::null_data,
                     "active LASER mutation payload is missing");
    }
    if (request.payload_size == sizeof(SegmentMutationBundlePayload)) {
      const auto &bundle = *static_cast<const SegmentMutationBundlePayload *>(request.payload);
      if (!core::is_current_struct(bundle) || bundle.rows.empty()) {
        return failure(core::OperationStage::mutation_prepare,
                       core::StatusCode::invalid_argument,
                       core::StatusDetail::malformed_struct,
                       "active LASER mutation bundle is invalid");
      }
      rows.reserve(bundle.rows.size());
      for (const auto &payload : bundle.rows) {
        Row row;
        auto status = decode_row(payload, row);
        if (!status.ok()) {
          return status;
        }
        rows.push_back(std::move(row));
      }
      return core::Status::success();
    }
    if (request.payload_size == sizeof(SegmentMutationPayload)) {
      const auto &payload = *static_cast<const SegmentMutationPayload *>(request.payload);
      Row row;
      auto status = decode_row(payload, row);
      if (!status.ok()) {
        return status;
      }
      rows.push_back(std::move(row));
      return core::Status::success();
    }
    return failure(core::OperationStage::mutation_prepare,
                   core::StatusCode::invalid_argument,
                   core::StatusDetail::malformed_struct,
                   "active LASER mutation payload size is invalid");
  }

  // Ruling 4 publish plan: (1) one commit_physical_bundle over ALL write rows
  // (labels = target row_id), gated by the idempotency decision; (2) every row's
  // same-segment previous, deduped, then explicit erase targets, tombstoned. Pure
  // erase / pure previous transactions never call the bundle.
  [[nodiscard]] auto apply_transaction(const Pending &pending, bool is_replay) -> core::Status {
    std::vector<float> vecs;
    std::vector<std::uint64_t> labels;
    for (const auto &row : pending.rows) {
      if (row.action == SegmentMutationAction::write) {
        vecs.insert(vecs.end(), row.vector.begin(), row.vector.end());
        labels.push_back(row.target_label);
      }
    }
    std::vector<std::uint64_t> tombstones;
    std::set<std::uint64_t> seen;
    const auto add_tombstone = [&](std::uint64_t label) {
      if (seen.insert(label).second) {
        tombstones.push_back(label);
      }
    };
    for (const auto &row : pending.rows) {
      if (row.previous_label.has_value()) {
        add_tombstone(*row.previous_label);
      }
    }
    for (const auto &row : pending.rows) {
      if (row.action == SegmentMutationAction::erase) {
        add_tombstone(row.target_label);
      }
    }

    try {
      // B-2C-07 / codex B.7: capture every tombstone target's TOKEN (pid, generation)
      // BEFORE the write bundle rebinds any label, so a same-txid rebind + PID reuse can
      // never make us tombstone a freshly-written incarnation. The captured token drives
      // the segment's ABA check (stale token => idempotent no-op; future => corruption).
      std::vector<std::optional<::alaya::laser::PidToken>> captured;
      captured.reserve(tombstones.size());
      for (const std::uint64_t label : tombstones) {
        captured.push_back(segment_->token_for_label(label));
      }
      if (!labels.empty()) {
        const std::uint64_t applied = segment_->applied_collection_op_id();
        const std::uint64_t last = segment_->last_committed_txid();
        if (pending.max_row_op_id <= applied) {
          // Whole write set already applied (idempotent retry / replay) -- skip.
        } else if (pending.txid > last) {
          (void)segment_->commit_physical_bundle(pending.txid,
                                                 pending.max_row_op_id,
                                                 vecs.data(),
                                                 labels.data(),
                                                 labels.size());
        } else {
          // Unapplied (max_row_op > applied) yet the txid is not new: WAL corruption.
          return latch(failure(is_replay ? core::OperationStage::mutation_replay
                                         : core::OperationStage::mutation_publish,
                               core::StatusCode::corruption,
                               core::StatusDetail::malformed_struct,
                               "active LASER write is unapplied yet txid <= last_committed "
                               "(WAL corruption / lost idempotency basis)"));
        }
      }
      if (!is_replay && fail_next_publish_.exchange(false, std::memory_order_acq_rel)) {
        return latch(failure(core::OperationStage::mutation_publish,
                             core::StatusCode::internal,
                             core::StatusDetail::engine_exception,
                             "injected active LASER publish failure (post-commit)"));
      }
      int tombstone_index = 0;
      for (std::size_t i = 0; i < tombstones.size(); ++i) {
        if (!is_replay && fail_tombstone_at_.load(std::memory_order_acquire) == tombstone_index) {
          fail_tombstone_at_.store(-1, std::memory_order_release);
          return latch(failure(core::OperationStage::mutation_publish,
                               core::StatusCode::internal,
                               core::StatusDetail::engine_exception,
                               "injected active LASER tombstone failure (post-commit)"));
        }
        ++tombstone_index;
        if (captured[i].has_value()) {
          // ABA-safe: the segment no-ops a stale token (the PID was reused by a newer
          // incarnation) and only erases the reverse map on full-token equality.
          segment_->tombstone(*captured[i]);
        } else if (!is_replay) {
          // Ruling 10: a runtime previous/erase miss is a high-severity diagnostic,
          // but the end state is already correct (the label has no live token), so it
          // is idempotent success -- do not latch. Replay misses stay silent.
          const std::lock_guard<std::mutex> lock(diag_mutex_);
          last_runtime_miss_ = tombstones[i];
        }
      }
    } catch (...) {
      return latch(core::status_from_exception(is_replay ? core::OperationStage::mutation_replay
                                                         : core::OperationStage::mutation_publish));
    }
    return core::Status::success();
  }

  [[nodiscard]] auto execute_search(const core::SearchRequest &request) const -> core::Status {
    if (request.context == nullptr || request.response == nullptr) {
      return failure(core::OperationStage::validation,
                     core::StatusCode::invalid_argument,
                     core::StatusDetail::malformed_struct,
                     "active LASER search request is incomplete");
    }
    auto status =
        core::validate_tensor(request.queries, schema_.dim, core::OperationStage::validation);
    if (!status.ok()) {
      return status;
    }
    if (request.queries.scalar_type != core::ScalarType::float32) {
      return failure(core::OperationStage::validation,
                     core::StatusCode::not_supported,
                     core::StatusDetail::unsupported_scalar_type,
                     "active LASER search requires float32 queries");
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

    ::alaya::disk::DiskSearchOptions options;
    options.ef = std::max<std::uint32_t>(static_cast<std::uint32_t>(request.options.top_k), 128U);
    auto resolved = ::alaya::disk::resolve_laser_search_extensions(request.options, options);
    if (!resolved.ok()) {
      return resolved.status();
    }
    options = std::move(resolved).value();

    auto &response = *request.response;
    response.query_count = request.queries.rows;
    const auto score_kind =
        options.return_distances ? core::ScoreKind::distance : core::ScoreKind::rank_only;
    response.score_kind = score_kind;
    response.comparable_metric = schema_.metric;
    response.result_flags = core::ResultFlag::approximate;
    response.offsets[0] = 0;
    core::RowCount cursor{};
    for (core::RowCount query_index = 0; query_index < request.queries.rows; ++query_index) {
      std::vector<::alaya::disk::DiskSearchHit> hits;
      try {
        hits = segment_->search(request.queries.row<float>(query_index), options);
      } catch (...) {
        return core::status_from_exception(core::OperationStage::search);
      }
      const auto count = std::min<std::uint64_t>(hits.size(), request.options.top_k);
      for (std::uint64_t index = 0; index < count; ++index) {
        core::SearchHit hit(core::SegmentRowId(hits[index].label),
                            options.return_distances ? hits[index].distance : 0.0F,
                            score_kind,
                            schema_.metric,
                            core::ResultFlag::approximate);
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

  std::shared_ptr<::alaya::disk::MutableLaserSegment> segment_{};
  CollectionSchema schema_{};
  std::uint64_t segment_id_{};
  std::uint64_t generation_{};

  mutable std::mutex mutex_{};                       // guards transactions_
  std::map<std::uint64_t, Pending> transactions_{};  // keyed by physical txid
  bool maintenance_active_{};

  std::atomic<bool> failed_{false};  // B-04 latch (lock-free)
  mutable std::mutex diag_mutex_{};  // guards diagnostic strings
  std::string diagnostic_{};
  std::uint64_t last_runtime_miss_{};

  std::atomic<bool> fail_next_publish_{false};
  std::atomic<int> fail_tombstone_at_{-1};
  mutable Gate publish_gate_{};
};

[[nodiscard]] inline auto make_active_laser_registration(
    std::shared_ptr<::alaya::disk::MutableLaserSegment> segment,
    CollectionSchema schema,
    std::uint64_t segment_id,
    std::uint64_t generation) -> core::Result<SegmentRegistration> {
  auto adapter = std::make_shared<MutableLaserCollectionAdapter>(std::move(segment),
                                                                 schema,
                                                                 segment_id,
                                                                 generation);
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
  auto erased = core::AnySegment::from_sync(adapter, std::move(config));
  if (!erased.ok()) {
    return erased.status();
  }
  SegmentRegistration registration;
  registration.segment_id = segment_id;
  registration.generation = generation;
  registration.role = SegmentRole::active_mutable;
  registration.segment = std::move(erased).value();
  registration.atomic_mutation_bundle = true;
  registration.maintenance.consolidate = [adapter](std::size_t num_threads,
                                                   std::size_t r_target,
                                                   bool reclaim_slots,
                                                   bool bloom_consolidate) {
    return adapter->consolidate(num_threads, r_target, reclaim_slots, bloom_consolidate);
  };
  registration.maintenance.recovery_required = [adapter] {
    return adapter->recovery_required();
  };
  return registration;
}

}  // namespace alaya::internal::collection::detail
