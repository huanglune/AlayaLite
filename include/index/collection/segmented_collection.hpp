// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "index/collection/collection_checkpoint.hpp"
#include "index/collection/experimental_snapshot_writer.hpp"

namespace alaya::internal::collection {

class SegmentedCollection {
 public:
  SegmentedCollection(const SegmentedCollection &) = delete;
  auto operator=(const SegmentedCollection &) -> SegmentedCollection & = delete;
  SegmentedCollection(SegmentedCollection &&) = delete;
  auto operator=(SegmentedCollection &&) -> SegmentedCollection & = delete;

  ~SegmentedCollection() {
#ifndef NDEBUG
    assert(outstanding_search_leases_.load(std::memory_order_acquire) == 0 &&
           "SegmentedCollection destroyed with an unreleased search lease");
    assert(leased_search_bytes_.load(std::memory_order_acquire) == 0 &&
           "SegmentedCollection destroyed with accounted scratch bytes still leased");
#endif
  }

  [[nodiscard]] static auto open(CollectionSchema schema,
                                 std::vector<SegmentRegistration> registrations,
                                 CollectionConfig config = {})
      -> core::Result<std::shared_ptr<SegmentedCollection>>;

  [[nodiscard]] auto concurrency_profile() const noexcept -> core::ConcurrencyProfile {
    core::ConcurrencyProfile profile;
    profile.reentrant_search = true;
    profile.search_with_stage = true;
    profile.search_with_publish = true;
    profile.serial_mutation = true;
    profile.checkpoint_with_search = false;
    profile.freeze_with_search = false;
    profile.native_async = false;
    profile.cooperative_cancel = false;
    profile.explicit_drain = true;
    return profile;
  }

  [[nodiscard]] auto search(const CollectionSearchRequest &request)
      -> core::Result<CollectionSearchResult> {
    auto admission = admit();
    if (!admission.has_value()) {
      return closed_status(core::OperationStage::admission);
    }
    return search_at_snapshot(load_snapshot(), request, false);
  }

  [[nodiscard]] auto hybrid_query(const CollectionSearchRequest &request)
      -> core::Result<CollectionSearchResult> {
    auto admission = admit();
    if (!admission.has_value()) {
      return closed_status(core::OperationStage::admission);
    }
    if (!request.filter.active()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "hybrid query requires a logical filter");
    }
    return search_at_snapshot(load_snapshot(), request, true);
  }

  [[nodiscard]] auto get_by_id(const core::LogicalId &logical_id,
                               Projection projection = Projection::all)
      -> core::Result<CollectionRecord> {
    auto admission = admit();
    if (!admission.has_value()) {
      return closed_status(core::OperationStage::admission);
    }
    const auto snapshot = load_snapshot();
    const auto found = snapshot->versions.find(logical_id);
    if (found == snapshot->versions.end() || found->second.state != VersionState::live ||
        found->second.upsert_sequence > snapshot->visibility_watermark) {
      return not_found("logical ID is not live at the admitted watermark");
    }
    return materialize_record(found->first, found->second, projection);
  }

  [[nodiscard]] auto scalar_query(const LogicalFilter &filter,
                                  std::size_t limit,
                                  Projection projection = Projection::all)
      -> core::Result<std::vector<CollectionRecord>>;

  [[nodiscard]] auto write(const WriteRequest &request, core::MutationContext &context)
      -> core::Result<MutationReceipt>;

  [[nodiscard]] auto erase(const core::LogicalId &logical_id,
                           core::MutationContext &context,
                           WriteOptions options = {}) -> core::Result<MutationReceipt>;

  [[nodiscard]] auto delete_by_filter(const LogicalFilter &filter, core::MutationContext &context)
      -> core::Result<std::vector<MutationReceipt>>;

  [[nodiscard]] auto mutate_batch(const BatchMutationRequest &request,
                                  core::MutationContext &context)
      -> core::Result<BatchMutationReceipt>;

  [[nodiscard]] auto retire_segment(std::uint64_t segment_id, std::uint64_t generation)
      -> core::Status;

  [[nodiscard]] auto stats() const -> CollectionStats;

  [[nodiscard]] auto close() -> core::Status {
    std::lock_guard lock(lifecycle_mutex_);
    if (lifecycle_ == LifecycleState::open) {
      lifecycle_ = LifecycleState::closing;
    }
    return core::Status::success();
  }

  [[nodiscard]] auto drain(const core::Deadline &deadline = {}) -> core::Status;

  [[nodiscard]] auto persist_experimental_snapshot() const -> core::Status {
    if (const auto writable = ensure_writable(core::OperationStage::save); !writable.ok()) {
      return writable;
    }
    if (!config_.features.experimental_persistence_writer) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::save,
                                 core::StatusDetail::operation_slot_absent,
                                 "experimental collection writer is disabled");
    }
    return ExperimentalSnapshotWriter::write(config_.persistence, *load_snapshot());
  }

  [[nodiscard]] auto checkpoint(core::CheckpointContext &context)
      -> core::Result<CheckpointReceipt>;

  [[nodiscard]] auto consolidate(std::size_t num_threads,
                                 std::size_t r_target,
                                 bool reclaim_slots,
                                 bool bloom_consolidate) -> core::Result<SegmentMaintenanceReceipt>;

  static void apply_checkpoint_to_manifest(const CheckpointReceipt &checkpoint,
                                           ArtifactManifestV2 &manifest) {
    CollectionCheckpointStore::apply_to_manifest(checkpoint, manifest);
  }

  using RotationDurableCallback = std::function<core::Status(const ActiveRotationReceipt &)>;

  [[nodiscard]] auto rotate_to_successor(SegmentRegistration successor,
                                         core::CheckpointContext &context,
                                         RotationDurableCallback durable_switch)
      -> core::Result<ActiveRotationReceipt>;

  [[nodiscard]] auto install_segment_replacement(std::span<const RowAddress> sources,
                                                 SegmentRegistration target,
                                                 std::span<const SegmentReplacement> replacements)
      -> core::Status {
    if (const auto writable = ensure_writable(core::OperationStage::save); !writable.ok()) {
      return writable;
    }
    auto admission = admit();
    if (!admission.has_value()) {
      return closed_status(core::OperationStage::save);
    }
    std::lock_guard mutation_lock(mutation_mutex_);
    return install_segment_replacement_locked(sources, std::move(target), replacements, false);
  }

  [[nodiscard]] auto resume_segment_replacement(std::span<const RowAddress> sources,
                                                std::uint64_t target_segment_id,
                                                std::uint64_t target_generation,
                                                std::span<const SegmentReplacement> replacements)
      -> core::Status;

  // Internal tests and control-plane code can pin this immutable epoch. The
  // shared reference is the reclamation barrier for every SegmentEntry it owns.
  [[nodiscard]] auto pin_routing_snapshot() const -> RoutingSnapshotPtr { return load_snapshot(); }

  [[nodiscard]] auto outstanding_search_leases() const noexcept -> std::uint64_t {
    return outstanding_search_leases_.load(std::memory_order_acquire);
  }

  [[nodiscard]] auto recovery_gate(core::OperationStage stage) const -> core::Status {
    return ensure_not_recovery_required(stage);
  }

 private:
  class OperationGuard {
   public:
    explicit OperationGuard(SegmentedCollection *owner) : owner_(owner) {}
    OperationGuard(const OperationGuard &) = delete;
    auto operator=(const OperationGuard &) -> OperationGuard & = delete;
    OperationGuard(OperationGuard &&other) noexcept : owner_(other.owner_) {
      other.owner_ = nullptr;
    }
    auto operator=(OperationGuard &&) -> OperationGuard & = delete;
    ~OperationGuard() {
      if (owner_ != nullptr) {
        owner_->release_admission();
      }
    }

   private:
    SegmentedCollection *owner_{};
  };

  class ControlPlaneGate {
   public:
    explicit ControlPlaneGate(SegmentedCollection *owner) : owner_(owner) {
      std::lock_guard lock(owner_->lifecycle_mutex_);
      owner_->control_plane_gate_ = true;
    }
    ControlPlaneGate(const ControlPlaneGate &) = delete;
    auto operator=(const ControlPlaneGate &) -> ControlPlaneGate & = delete;
    ~ControlPlaneGate() {
      std::lock_guard lock(owner_->lifecycle_mutex_);
      owner_->control_plane_gate_ = false;
      owner_->lifecycle_changed_.notify_all();
    }

    [[nodiscard]] auto drain(const core::Deadline &deadline,
                             const core::CancellationToken &cancellation) -> core::Status {
      std::unique_lock lock(owner_->lifecycle_mutex_);
      while (owner_->inflight_operations_ != 1) {  // The checkpoint owns the remaining admission.
        auto control = core::validate_runtime_control(deadline,
                                                      cancellation,
                                                      core::OperationStage::checkpoint);
        if (!control.ok()) {
          return control;
        }
        if (deadline.enabled) {
          owner_->lifecycle_changed_.wait_for(lock, std::chrono::milliseconds(1));
        } else {
          owner_->lifecycle_changed_.wait(lock);
        }
      }
      return core::Status::success();
    }

   private:
    SegmentedCollection *owner_{};
  };

  class PendingGuard {
   public:
    PendingGuard(SegmentedCollection *owner, std::uint64_t bytes, std::uint64_t rows = 1)
        : owner_(owner), bytes_(bytes), rows_(rows) {
      owner_->pending_count_.fetch_add(rows_, std::memory_order_acq_rel);
      owner_->pending_bytes_.fetch_add(bytes_, std::memory_order_acq_rel);
    }
    PendingGuard(const PendingGuard &) = delete;
    auto operator=(const PendingGuard &) -> PendingGuard & = delete;
    ~PendingGuard() {
      owner_->pending_count_.fetch_sub(rows_, std::memory_order_acq_rel);
      owner_->pending_bytes_.fetch_sub(bytes_, std::memory_order_acq_rel);
    }

   private:
    SegmentedCollection *owner_{};
    std::uint64_t bytes_{};
    std::uint64_t rows_{};
  };

  class AcceptedGuard {
   public:
    AcceptedGuard(SegmentedCollection *owner, RowMutationStatus status)
        : owner_(owner), status_(status) {
      if (status_ == RowMutationStatus::inserted) {
        owner_->accepted_count_.fetch_add(1, std::memory_order_acq_rel);
      }
    }
    AcceptedGuard(const AcceptedGuard &) = delete;
    auto operator=(const AcceptedGuard &) -> AcceptedGuard & = delete;
    ~AcceptedGuard() {
      if (!committed_ && status_ == RowMutationStatus::inserted) {
        owner_->accepted_count_.fetch_sub(1, std::memory_order_acq_rel);
      }
    }

    void commit() {
      if (status_ == RowMutationStatus::deleted) {
        owner_->accepted_count_.fetch_sub(1, std::memory_order_acq_rel);
      }
      committed_ = true;
    }

   private:
    SegmentedCollection *owner_{};
    RowMutationStatus status_{RowMutationStatus::aborted};
    bool committed_{};
  };

  // B-03: RAII latch over the post-COMMIT window (L:COMMIT durable .. publish_snapshot
  // done). If destroyed still armed (any non-termination early return in that window),
  // it latches a Collection-level recovery-required state; disarm() marks the clean
  // success path. Engine-agnostic: harmless for an in-memory flat active segment.
  class RecoveryGuard {
   public:
    explicit RecoveryGuard(SegmentedCollection *owner) : owner_(owner) {}
    RecoveryGuard(const RecoveryGuard &) = delete;
    auto operator=(const RecoveryGuard &) -> RecoveryGuard & = delete;
    ~RecoveryGuard() {
      if (owner_ == nullptr) {
        return;
      }
      owner_->latch_recovery_required(
          "a committed transaction failed to publish (post-COMMIT window)");
    }
    void disarm() { owner_ = nullptr; }

   private:
    SegmentedCollection *owner_{};
  };

  class SearchLeaseGuard {
   public:
    SearchLeaseGuard(SegmentedCollection *owner, std::uint64_t bytes, CollectionSearchStats *stats)
        : owner_(owner), bytes_(bytes), stats_(stats) {
      owner_->outstanding_search_leases_.fetch_add(1, std::memory_order_acq_rel);
      const auto leased =
          owner_->leased_search_bytes_.fetch_add(bytes_, std::memory_order_acq_rel) + bytes_;
      if (stats_ != nullptr) {
        ++stats_->lease_acquired;
        stats_->lease_peak_bytes = std::max(stats_->lease_peak_bytes, leased);
      }
    }

    SearchLeaseGuard(const SearchLeaseGuard &) = delete;
    auto operator=(const SearchLeaseGuard &) -> SearchLeaseGuard & = delete;

    ~SearchLeaseGuard() { release(); }

    void release() noexcept {
      if (owner_ == nullptr) {
        return;
      }
      [[maybe_unused]] const auto previous_bytes =
          owner_->leased_search_bytes_.fetch_sub(bytes_, std::memory_order_acq_rel);
      [[maybe_unused]] const auto previous_leases =
          owner_->outstanding_search_leases_.fetch_sub(1, std::memory_order_acq_rel);
#ifndef NDEBUG
      assert(previous_bytes >= bytes_ && "search byte lease accounting underflow");
      assert(previous_leases != 0 && "search lease accounting underflow");
#endif
      if (stats_ != nullptr) {
        ++stats_->lease_released;
      }
      owner_ = nullptr;
    }

   private:
    SegmentedCollection *owner_{};
    std::uint64_t bytes_{};
    CollectionSearchStats *stats_{};
  };

  struct SegmentSearchStorage {
    SegmentSearchStorage(core::RowCount rows, core::RowCount top_k)
        : hits(static_cast<std::size_t>(rows * top_k)),
          offsets(static_cast<std::size_t>(rows + 1)),
          counts(static_cast<std::size_t>(rows)),
          statuses(static_cast<std::size_t>(rows)),
          completeness(static_cast<std::size_t>(rows)) {
      response.hits = hits;
      response.offsets = offsets;
      response.valid_counts = counts;
      response.statuses = statuses;
      response.completeness = completeness;
    }

    std::vector<core::SearchHit> hits{};
    std::vector<core::RowCount> offsets{};
    std::vector<core::RowCount> counts{};
    std::vector<core::Status> statuses{};
    std::vector<core::SearchCompleteness> completeness{};
    core::SearchResponse response{};
  };

  struct Candidate {
    CollectionHit hit{};
    std::shared_ptr<SegmentEntry> segment{};
    const RecordPayload *payload{};
  };

  explicit SegmentedCollection(CollectionSchema schema, CollectionConfig config)
      : schema_(std::move(schema)), config_(std::move(config)) {}

  [[nodiscard]] auto install_segment_replacement_locked(
      std::span<const RowAddress> sources,
      SegmentRegistration target,
      std::span<const SegmentReplacement> replacements,
      bool target_is_already_routed) -> core::Status;

  [[nodiscard]] auto initialize(std::vector<SegmentRegistration> registrations) -> core::Status;

  [[nodiscard]] auto admit() -> std::optional<OperationGuard> {
    std::lock_guard lock(lifecycle_mutex_);
    if (lifecycle_ != LifecycleState::open || control_plane_gate_) {
      return std::nullopt;
    }
    ++inflight_operations_;
    return OperationGuard(this);
  }

  void release_admission() {
    std::lock_guard lock(lifecycle_mutex_);
    --inflight_operations_;
    lifecycle_changed_.notify_all();
  }

  struct SearchBudgetPlan {
    std::uint64_t scratch_bytes{};
    std::uint64_t io_requests{};
    std::uint64_t io_bytes{};
  };

  [[nodiscard]] static auto search_budget_denied(std::string diagnostic) -> core::Status {
    return core::Status::error(core::StatusCode::resource_exhausted,
                               core::OperationStage::admission,
                               core::StatusDetail::budget_denied,
                               std::move(diagnostic),
                               core::Retryability::retryable_with_backoff);
  }

  [[nodiscard]] auto preflight_search_budget(const RoutingSnapshot &snapshot,
                                             const CollectionSearchRequest &request)
      -> core::Result<SearchBudgetPlan>;

  [[nodiscard]] static auto estimate_filter_selectivity(const RoutingSnapshot &snapshot,
                                                        const LogicalFilter &filter,
                                                        CollectionSearchStats *stats) -> double;

  [[nodiscard]] static auto select_filter_execution(const RoutingSnapshot &snapshot,
                                                    const CollectionSearchRequest &request,
                                                    bool prefer_exact)
      -> core::Result<core::FilterExecution>;

  [[nodiscard]] auto search_at_snapshot(const RoutingSnapshotPtr &snapshot,
                                        const CollectionSearchRequest &request,
                                        bool prefer_exact) -> core::Result<CollectionSearchResult>;

  [[nodiscard]] auto exact_search(const RoutingSnapshotPtr &snapshot,
                                  const CollectionSearchRequest &request)
      -> core::Result<CollectionSearchResult>;

  [[nodiscard]] auto fanout_search(const RoutingSnapshotPtr &snapshot,
                                   const CollectionSearchRequest &request,
                                   core::FilterExecution execution)
      -> core::Result<CollectionSearchResult>;

  [[nodiscard]] auto normalize_scores(const std::vector<Candidate> &candidates,
                                      const core::TypedTensorView &query,
                                      core::SearchContext *context,
                                      CollectionSearchStats *stats)
      -> core::Result<std::vector<CollectionHit>>;

  [[nodiscard]] auto mutate_locked(RoutingSnapshotPtr current,
                                   const core::LogicalId &logical_id,
                                   SegmentMutationAction action,
                                   RecordPayload payload,
                                   RowMutationStatus row_status,
                                   core::MutationContext &context,
                                   const WriteOptions &options,
                                   std::uint64_t batch_op_id = 0) -> core::Result<MutationReceipt>;

  struct ValidatedBatchRow {
    bool valid{};
    SegmentMutationAction action{SegmentMutationAction::write};
    RowMutationStatus status{RowMutationStatus::invalid_argument};
    RecordPayload payload{};
    std::optional<RowAddress> previous{};
  };

  [[nodiscard]] auto preflight_batch_resources(const BatchMutationRequest &request,
                                               core::MutationContext &context) const
      -> core::Status;

  [[nodiscard]] auto validate_batch_row(const RoutingSnapshotPtr &current,
                                        const BatchRowMutation &row) const -> ValidatedBatchRow;

  [[nodiscard]] auto make_non_searchable_receipt(const RoutingSnapshotPtr &current,
                                                 std::uint64_t batch_op_id,
                                                 RowMutationStatus status,
                                                 std::string retry_token) -> MutationReceipt;

  [[nodiscard]] auto mutate_batch_row_locked(RoutingSnapshotPtr current,
                                             const BatchRowMutation &row,
                                             core::MutationContext &context,
                                             const WriteOptions &options,
                                             std::uint64_t batch_op_id)
      -> core::Result<MutationReceipt>;

  [[nodiscard]] auto mutate_atomic_batch_locked(RoutingSnapshotPtr current,
                                                const BatchMutationRequest &request,
                                                core::MutationContext &context,
                                                std::uint64_t batch_op_id)
      -> core::Result<BatchMutationReceipt>;

  [[nodiscard]] static auto durability_state(WriteDurability durability) -> DurabilityState {
    return durability == WriteDurability::wal_fsync ? DurabilityState::wal_fsync
                                                    : DurabilityState::searchable_not_durable;
  }

  [[nodiscard]] auto persist_batch_receipt(const BatchMutationReceipt &receipt,
                                           WriteDurability durability) -> core::Status;

  [[nodiscard]] auto failpoint(MutationFailPoint point) -> bool {
    if (config_.failpoint_hook) {
      config_.failpoint_hook(point);
    }
    return config_.fail_point == point;
  }

  [[nodiscard]] static auto injected_failure(MutationFailPoint point) -> core::Status {
    const auto stage = point == MutationFailPoint::after_commit ||
                               point == MutationFailPoint::after_publish ||
                               point == MutationFailPoint::after_engine_publish_before_snapshot
                           ? core::OperationStage::mutation_publish
                       : point == MutationFailPoint::after_stage ||
                               point == MutationFailPoint::metadata_stage_failure
                           ? core::OperationStage::mutation_stage
                           : core::OperationStage::mutation_prepare;
    return core::Status::error(core::StatusCode::internal,
                               stage,
                               core::StatusDetail::engine_exception,
                               "injected collection mutation failpoint");
  }

  // B-01: the physical transaction id an engine sees is the REAL logical-WAL txid --
  // an atomic batch uses its shared frame id (batch_op_id); a single/per-row
  // transaction uses the row's own op_id (never the per-row batch's shared id).
  [[nodiscard]] static auto physical_txid(const WalMutationTransaction &transaction)
      -> std::uint64_t {
    if (transaction.batch_mode == BatchMutationMode::all_or_nothing &&
        transaction.rows.size() > 1) {
      return transaction.batch_op_id;
    }
    return transaction.rows.empty() ? transaction.batch_op_id : transaction.rows.front().op_id;
  }
  // The idempotency basis: the maximum row op_id of the transaction, compared by the
  // engine against its persisted applied_collection_op_id.
  [[nodiscard]] static auto transaction_max_row_op(const WalMutationTransaction &transaction)
      -> std::uint64_t {
    std::uint64_t maximum = 0;
    for (const auto &row : transaction.rows) {
      maximum = std::max(maximum, row.op_id);
    }
    return maximum;
  }

  // B-03: reject write/erase/checkpoint/seal/rotate once the Collection has latched a
  // recovery-required state (a post-COMMIT window exited abnormally). Only a reopen
  // (WAL/checkpoint replay) clears it.
  void latch_recovery_required(std::string_view diagnostic) noexcept {
    recovery_required_.store(true, std::memory_order_release);
    try {
      const std::lock_guard<std::mutex> lock(recovery_diag_mutex_);
      if (recovery_diagnostic_.empty()) {
        recovery_diagnostic_.assign(diagnostic);
      }
    } catch (...) {
      // The atomic latch is authoritative; a diagnostic allocation failure must
      // never make the recovery gate reopen.
    }
  }

  [[nodiscard]] auto ensure_not_recovery_required(core::OperationStage stage) const
      -> core::Status {
    if (!recovery_required_.load(std::memory_order_acquire)) {
      return core::Status::success();
    }
    std::string diagnostic;
    {
      const std::lock_guard<std::mutex> lock(recovery_diag_mutex_);
      diagnostic = recovery_diagnostic_;
    }
    return core::Status::error(core::StatusCode::internal,
                               stage,
                               core::StatusDetail::readonly_instance,
                               "collection requires recovery (reopen to replay): " + diagnostic);
  }

  [[nodiscard]] auto make_engine_payloads(const WalMutationTransaction &transaction)
      -> std::vector<SegmentMutationPayload>;

  [[nodiscard]] auto build_dark_snapshot(const RoutingSnapshotPtr &current,
                                         const WalMutationTransaction &transaction,
                                         bool durable)
      -> core::Result<std::shared_ptr<RoutingSnapshot>>;

  [[nodiscard]] auto execute_transaction_locked(RoutingSnapshotPtr current,
                                                const std::shared_ptr<SegmentEntry> &target,
                                                const WalMutationTransaction &transaction,
                                                core::MutationContext &context,
                                                std::uint64_t transaction_id)
      -> core::Result<std::vector<MutationReceipt>>;

  [[nodiscard]] auto replay_engine_transaction(const WalMutationTransaction &transaction)
      -> core::Status;

  void install_recovered_receipts(const WalMutationTransaction &transaction,
                                  const RoutingSnapshot &snapshot,
                                  DurabilityState durability);

  [[nodiscard]] auto apply_checkpoint_image(std::shared_ptr<RoutingSnapshot> &snapshot,
                                            CollectionCheckpointImage image) -> core::Status;

  [[nodiscard]] auto recover_durable_state(std::shared_ptr<RoutingSnapshot> &snapshot)
      -> core::Status;

  [[nodiscard]] auto validate_logical_id(const core::LogicalId &logical_id,
                                         core::OperationStage stage) const -> core::Status {
    if (!core::is_current_struct(logical_id) ||
        logical_id.canonical_bytes().size() > schema_.max_logical_id_bytes) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 stage,
                                 core::StatusDetail::malformed_struct,
                                 "logical ID is incompatible or exceeds the schema limit");
    }
    return core::Status::success();
  }

  [[nodiscard]] static auto materialize_record(const core::LogicalId &logical_id,
                                               const VersionEntry &version,
                                               Projection projection)
      -> core::Result<CollectionRecord>;

  [[nodiscard]] static auto validate_segment_response(const core::SearchResponse &response,
                                                      core::RowCount query_count,
                                                      core::RowCount top_k) -> core::Status;

  [[nodiscard]] static auto malformed_engine_response() -> core::Status {
    return core::Status::error(core::StatusCode::internal,
                               core::OperationStage::search,
                               core::StatusDetail::malformed_struct,
                               "segment returned an invalid flattened response");
  }

  [[nodiscard]] static auto query_row(const core::TypedTensorView &queries, core::RowCount row)
      -> core::TypedTensorView {
    const auto *bytes = static_cast<const std::byte *>(queries.data) + row * queries.row_stride;
    return {bytes, queries.scalar_type, 1, queries.dim, queries.row_stride};
  }

  template <class T>
  [[nodiscard]] static auto exact_distance_typed(const core::TypedTensorView &query,
                                                 const OwnedVector &vector,
                                                 core::Metric metric) -> float {
    const auto *lhs = query.row<T>(0);
    const auto stored = vector.view();
    const auto *rhs = stored.row<T>(0);
    double dot{};
    double lhs_norm{};
    double rhs_norm{};
    double l2{};
    for (std::uint32_t index = 0; index < query.dim; ++index) {
      const auto left = static_cast<double>(lhs[index]);
      const auto right = static_cast<double>(rhs[index]);
      const auto difference = left - right;
      l2 += difference * difference;
      dot += left * right;
      lhs_norm += left * left;
      rhs_norm += right * right;
    }
    switch (metric) {
      case core::Metric::l2:
        return static_cast<float>(l2);
      case core::Metric::inner_product:
        return static_cast<float>(-dot);
      case core::Metric::cosine:
        if (lhs_norm == 0 || rhs_norm == 0) {
          return 0.0F;
        }
        return static_cast<float>(-dot / std::sqrt(lhs_norm * rhs_norm));
    }
    return 0.0F;
  }

  [[nodiscard]] static auto exact_distance(const core::TypedTensorView &query,
                                           const OwnedVector &vector,
                                           core::Metric metric) -> core::Result<float>;

  static void sort_hits(std::vector<CollectionHit> &hits) {
    std::sort(hits.begin(), hits.end(), [](const CollectionHit &lhs, const CollectionHit &rhs) {
      if (lhs.score != rhs.score) {
        return lhs.score_kind == core::ScoreKind::similarity ? lhs.score > rhs.score
                                                             : lhs.score < rhs.score;
      }
      const auto logical_order = lhs.logical_id.compare(rhs.logical_id);
      if (logical_order != 0) {
        return logical_order < 0;
      }
      return lhs.source < rhs.source;
    });
  }

  [[nodiscard]] static auto is_nan_score(float score) noexcept -> bool {
    const auto bits = std::bit_cast<std::uint32_t>(score);
    return (bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0;
  }

  static void recalculate_counts(RoutingSnapshot &snapshot) {
    snapshot.searchable_live_count = 0;
    snapshot.tombstone_count = 0;
    for (const auto &[unused, version] : snapshot.versions) {
      (void)unused;
      if (version.state == VersionState::live) {
        ++snapshot.searchable_live_count;
      } else {
        ++snapshot.tombstone_count;
      }
    }
    snapshot.rebuild_known_row_counts();
  }

  [[nodiscard]] auto closed_status(core::OperationStage stage) const -> core::Status {
    std::lock_guard lock(lifecycle_mutex_);
    // Admission can observe the checkpoint gate and then reach this mapper
    // after the gate has already reopened.  The lifecycle is the stable fact:
    // an open collection rejected by admit() was temporarily gated and must
    // remain retryable rather than being misreported as permanently closed.
    if (lifecycle_ == LifecycleState::open) {
      return core::Status::error(core::StatusCode::conflict,
                                 stage,
                                 core::StatusDetail::none,
                                 "collection admission is temporarily gated by checkpoint");
    }
    return core::Status::error(core::StatusCode::closed,
                               stage,
                               core::StatusDetail::operation_slot_absent,
                               "collection admission is closed");
  }

  [[nodiscard]] auto ensure_writable(core::OperationStage stage) const -> core::Status {
    if (!config_.read_only) {
      return core::Status::success();
    }
    return core::Status::error(core::StatusCode::not_supported,
                               stage,
                               core::StatusDetail::readonly_instance,
                               "operation is unavailable on a read-only Collection handle");
  }

  [[nodiscard]] static auto not_found(std::string diagnostic) -> core::Status {
    return core::Status::error(core::StatusCode::not_found,
                               core::OperationStage::admission,
                               core::StatusDetail::none,
                               std::move(diagnostic));
  }

  [[nodiscard]] auto load_snapshot() const -> RoutingSnapshotPtr {
    return std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
  }

  void publish_snapshot(std::shared_ptr<RoutingSnapshot> snapshot) {
    std::atomic_store_explicit(&snapshot_,
                               RoutingSnapshotPtr(std::move(snapshot)),
                               std::memory_order_release);
  }

  CollectionSchema schema_{};
  CollectionConfig config_{};
  std::unique_ptr<CollectionLogicalWal> wal_{};
  std::map<std::string, MutationReceipt, std::less<>> retry_receipts_{};
  std::map<std::string, BatchMutationReceipt, std::less<>> batch_retry_receipts_{};
  std::shared_ptr<RoutingSnapshot> load_or_initializing_snapshot_{};
  std::uint64_t maximum_recovered_op_id_{};
  RoutingSnapshotPtr snapshot_{};
  mutable std::mutex lifecycle_mutex_{};
  std::condition_variable lifecycle_changed_{};
  LifecycleState lifecycle_{LifecycleState::open};
  bool control_plane_gate_{};
  std::uint64_t inflight_operations_{};
  // B-03 Collection-level recovery-required latch (set by RecoveryGuard).
  std::atomic<bool> recovery_required_{false};
  mutable std::mutex recovery_diag_mutex_{};
  std::string recovery_diagnostic_{};
  std::mutex drain_mutex_{};
  std::mutex checkpoint_mutex_{};
  std::mutex mutation_mutex_{};
  std::atomic_uint64_t next_op_id_{1};
  std::atomic_uint64_t accepted_count_{};
  std::atomic_uint64_t pending_count_{};
  std::atomic_uint64_t pending_bytes_{};
  std::atomic_uint64_t outstanding_search_leases_{};
  std::atomic_uint64_t leased_search_bytes_{};
};

}  // namespace alaya::internal::collection
