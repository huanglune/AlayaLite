// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
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

  [[nodiscard]] static auto open(CollectionSchema schema,
                                 std::vector<SegmentRegistration> registrations,
                                 CollectionConfig config = {})
      -> core::Result<std::shared_ptr<SegmentedCollection>> {
    if (!config.features.collection_shell) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::open,
                                 core::StatusDetail::operation_slot_absent,
                                 "internal segmented collection feature is disabled");
    }
    if (schema.dim == 0 || core::scalar_type_size(schema.scalar_type) == 0 ||
        schema.max_logical_id_bytes == 0) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 "collection schema is invalid");
    }
    try {
      auto collection = std::shared_ptr<SegmentedCollection>(
          new SegmentedCollection(std::move(schema), std::move(config)));
      const auto status = collection->initialize(std::move(registrations));
      if (!status.ok()) {
        return status;
      }
      return collection;
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

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
      -> core::Result<std::vector<CollectionRecord>> {
    auto admission = admit();
    if (!admission.has_value()) {
      return closed_status(core::OperationStage::admission);
    }
    const auto snapshot = load_snapshot();
    std::vector<CollectionRecord> records;
    records.reserve(std::min<std::size_t>(limit, snapshot->searchable_live_count));
    for (const auto &[logical_id, version] : snapshot->versions) {
      if (records.size() == limit) {
        break;
      }
      if (version.state != VersionState::live ||
          version.upsert_sequence > snapshot->visibility_watermark ||
          !filter.matches(logical_id, version.payload.metadata, version.payload.document)) {
        continue;
      }
      auto record = materialize_record(logical_id, version, projection);
      if (!record.ok()) {
        return record.status();
      }
      records.push_back(std::move(record).value());
    }
    return records;
  }

  [[nodiscard]] auto write(const WriteRequest &request, core::MutationContext &context)
      -> core::Result<MutationReceipt> {
    auto admission = admit();
    if (!admission.has_value()) {
      return closed_status(core::OperationStage::admission);
    }
    std::lock_guard mutation_lock(mutation_mutex_);
    auto current = load_snapshot();
    if (!request.options.retry_token.empty()) {
      const auto retried = retry_receipts_.find(request.options.retry_token);
      if (retried != retry_receipts_.end()) {
        return retried->second;
      }
    }
    auto status = validate_logical_id(request.logical_id, core::OperationStage::validation);
    if (!status.ok()) {
      return status;
    }
    status = core::validate_tensor(request.vector, schema_.dim, core::OperationStage::validation);
    if (!status.ok()) {
      return status;
    }
    if (request.vector.rows != 1 || request.vector.scalar_type != schema_.scalar_type) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 request.vector.scalar_type == schema_.scalar_type
                                     ? core::StatusDetail::malformed_struct
                                     : core::StatusDetail::unsupported_scalar_type,
                                 "collection write requires one row with the schema scalar type");
    }

    const auto existing = current->versions.find(request.logical_id);
    const auto live =
        existing != current->versions.end() && existing->second.state == VersionState::live;
    if (request.mode == WriteMode::insert_only && live) {
      return core::Status::error(core::StatusCode::conflict,
                                 core::OperationStage::admission,
                                 core::StatusDetail::already_exists,
                                 "insert_only logical ID already exists");
    }
    if (request.mode == WriteMode::replace && !live) {
      return not_found("replace logical ID is not live");
    }
    auto owned = OwnedVector::copy_row(request.vector, 0);
    if (!owned.ok()) {
      return owned.status();
    }
    RecordPayload payload;
    payload.vector = std::move(owned).value();
    payload.metadata = request.metadata;
    payload.document = request.document;
    const auto row_status = !live                                ? RowMutationStatus::inserted
                            : request.mode == WriteMode::replace ? RowMutationStatus::replaced
                                                                 : RowMutationStatus::updated;
    return mutate_locked(std::move(current),
                         request.logical_id,
                         SegmentMutationAction::write,
                         std::move(payload),
                         row_status,
                         context,
                         request.options);
  }

  [[nodiscard]] auto erase(const core::LogicalId &logical_id,
                           core::MutationContext &context,
                           WriteOptions options = {}) -> core::Result<MutationReceipt> {
    auto admission = admit();
    if (!admission.has_value()) {
      return closed_status(core::OperationStage::admission);
    }
    std::lock_guard mutation_lock(mutation_mutex_);
    auto current = load_snapshot();
    if (!options.retry_token.empty()) {
      const auto retried = retry_receipts_.find(options.retry_token);
      if (retried != retry_receipts_.end()) {
        return retried->second;
      }
    }
    const auto status = validate_logical_id(logical_id, core::OperationStage::validation);
    if (!status.ok()) {
      return status;
    }
    const auto found = current->versions.find(logical_id);
    if (found == current->versions.end() || found->second.state != VersionState::live) {
      return not_found("delete logical ID is not live");
    }
    return mutate_locked(std::move(current),
                         logical_id,
                         SegmentMutationAction::erase,
                         found->second.payload,
                         RowMutationStatus::deleted,
                         context,
                         options);
  }

  [[nodiscard]] auto delete_by_filter(const LogicalFilter &filter, core::MutationContext &context)
      -> core::Result<std::vector<MutationReceipt>> {
    auto admission = admit();
    if (!admission.has_value()) {
      return closed_status(core::OperationStage::admission);
    }
    std::lock_guard mutation_lock(mutation_mutex_);
    const auto admitted = load_snapshot();
    std::vector<core::LogicalId> expanded;
    for (const auto &[logical_id, version] : admitted->versions) {
      if (version.state == VersionState::live &&
          version.upsert_sequence <= admitted->visibility_watermark &&
          filter.matches(logical_id, version.payload.metadata, version.payload.document)) {
        expanded.push_back(logical_id);
      }
    }

    // The expansion is deterministic because VersionMap is ordered by canonical
    // LogicalId bytes. Each expanded delete is persisted as its own logical
    // transaction; recovery never re-evaluates the predicate.
    std::vector<MutationReceipt> receipts;
    receipts.reserve(expanded.size());
    for (const auto &logical_id : expanded) {
      auto current = load_snapshot();
      const auto found = current->versions.find(logical_id);
      if (found == current->versions.end() || found->second.state != VersionState::live) {
        continue;
      }
      auto receipt = mutate_locked(std::move(current),
                                   logical_id,
                                   SegmentMutationAction::erase,
                                   found->second.payload,
                                   RowMutationStatus::deleted,
                                   context,
                                   {});
      if (!receipt.ok()) {
        return receipt.status();
      }
      receipts.push_back(std::move(receipt).value());
    }
    return receipts;
  }

  [[nodiscard]] auto mutate_batch(const BatchMutationRequest &request,
                                  core::MutationContext &context)
      -> core::Result<BatchMutationReceipt> {
    auto admission = admit();
    if (!admission.has_value()) {
      return closed_status(core::OperationStage::admission);
    }
    std::lock_guard mutation_lock(mutation_mutex_);
    if (!request.options.retry_token.empty()) {
      const auto retried = batch_retry_receipts_.find(request.options.retry_token);
      if (retried != batch_retry_receipts_.end()) {
        return retried->second;
      }
    }
    auto current = load_snapshot();
    if (request.rows.empty()) {
      BatchMutationReceipt empty;
      empty.visibility_watermark = current->visibility_watermark;
      empty.durable_watermark = current->durable_watermark;
      empty.searchable = true;
      empty.durability = config_.features.wal_coordinator
                             ? durability_state(request.options.durability)
                             : DurabilityState::memory_only;
      empty.retry_token = request.options.retry_token;
      auto marker = persist_batch_receipt(empty, request.options.durability);
      if (!marker.ok()) {
        return marker;
      }
      if (!empty.retry_token.empty()) {
        batch_retry_receipts_.insert_or_assign(empty.retry_token, empty);
      }
      return empty;
    }
    if (request.rows.size() > std::numeric_limits<std::uint32_t>::max()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "batch mutation row count exceeds uint32");
    }
    const auto batch_op_id = next_op_id_.fetch_add(1, std::memory_order_acq_rel);
    if (request.mode == BatchMutationMode::all_or_nothing) {
      return mutate_atomic_batch_locked(std::move(current), request, context, batch_op_id);
    }

    BatchMutationReceipt batch;
    batch.batch_op_id = batch_op_id;
    batch.retry_token = request.options.retry_token;
    batch.rows.reserve(request.rows.size());
    for (std::size_t index = 0; index < request.rows.size(); ++index) {
      auto row_options = request.options;
      row_options.retry_token = request.rows[index].retry_token;
      if (row_options.retry_token.empty() && !request.options.retry_token.empty()) {
        row_options.retry_token = request.options.retry_token + "#" + std::to_string(index);
      }
      const auto retried = row_options.retry_token.empty()
                               ? retry_receipts_.end()
                               : retry_receipts_.find(row_options.retry_token);
      if (retried != retry_receipts_.end()) {
        auto receipt = retried->second;
        receipt.batch_op_id = batch_op_id;
        batch.rows.push_back(std::move(receipt));
        continue;
      }
      auto receipt = mutate_batch_row_locked(load_snapshot(),
                                             request.rows[index],
                                             context,
                                             row_options,
                                             batch_op_id);
      if (!receipt.ok()) {
        return receipt.status();
      }
      batch.rows.push_back(std::move(receipt).value());
    }
    const auto final = load_snapshot();
    batch.visibility_watermark = final->visibility_watermark;
    batch.durable_watermark = final->durable_watermark;
    batch.searchable = std::ranges::any_of(batch.rows, [](const MutationReceipt &row) {
      return row.searchable;
    });
    batch.durability = config_.features.wal_coordinator
                           ? durability_state(request.options.durability)
                           : DurabilityState::memory_only;
    auto marker = persist_batch_receipt(batch, request.options.durability);
    if (!marker.ok()) {
      return marker;
    }
    if (!batch.retry_token.empty()) {
      batch_retry_receipts_.insert_or_assign(batch.retry_token, batch);
    }
    return batch;
  }

  [[nodiscard]] auto retire_segment(std::uint64_t segment_id, std::uint64_t generation)
      -> core::Status {
    auto admission = admit();
    if (!admission.has_value()) {
      return closed_status(core::OperationStage::admission);
    }
    std::lock_guard mutation_lock(mutation_mutex_);
    const auto current = load_snapshot();
    if (current->find_segment(segment_id, generation) == nullptr) {
      return not_found("segment is not present in the current routing snapshot");
    }
    for (const auto &[unused, version] : current->versions) {
      (void)unused;
      if (version.state == VersionState::live && version.address.segment_id == segment_id &&
          version.address.generation == generation) {
        return core::Status::error(core::StatusCode::conflict,
                                   core::OperationStage::admission,
                                   core::StatusDetail::none,
                                   "cannot retire a segment that owns a live logical version");
      }
    }
    auto next = std::make_shared<RoutingSnapshot>(*current);
    std::erase_if(next->segments, [&](const auto &entry) {
      return entry->segment_id == segment_id && entry->generation == generation;
    });
    next->generation = current->generation + 1;
    publish_snapshot(std::move(next));
    return core::Status::success();
  }

  [[nodiscard]] auto stats() const -> CollectionStats {
    const auto snapshot = load_snapshot();
    CollectionStats result;
    result.size = snapshot == nullptr ? 0 : snapshot->searchable_live_count;
    result.accepted_count = accepted_count_.load(std::memory_order_acquire);
    result.pending_count = pending_count_.load(std::memory_order_acquire);
    result.pending_bytes = pending_bytes_.load(std::memory_order_acquire);
    result.tombstone_count = snapshot == nullptr ? 0 : snapshot->tombstone_count;
    if (snapshot != nullptr) {
      result.routing_generation = snapshot->generation;
      result.visibility_watermark = snapshot->visibility_watermark;
      result.durable_watermark = snapshot->durable_watermark;
      result.metadata_epoch = snapshot->metadata_epoch;
      for (const auto &entry : snapshot->segments) {
        core::SegmentStats segment_stats;
        if (entry->segment.stats(segment_stats).ok()) {
          result.allocated_count += segment_stats.allocated_rows;
        } else {
          result.allocated_count += snapshot->known_rows_for(*entry);
        }
      }
    }
    {
      std::lock_guard lock(lifecycle_mutex_);
      result.lifecycle = lifecycle_;
    }
    return result;
  }

  [[nodiscard]] auto close() -> core::Status {
    std::lock_guard lock(lifecycle_mutex_);
    if (lifecycle_ == LifecycleState::open) {
      lifecycle_ = LifecycleState::closing;
    }
    return core::Status::success();
  }

  [[nodiscard]] auto drain(const core::Deadline &deadline = {}) -> core::Status {
    std::lock_guard drain_lock(drain_mutex_);
    {
      std::unique_lock lock(lifecycle_mutex_);
      if (lifecycle_ == LifecycleState::open) {
        return core::Status::error(core::StatusCode::conflict,
                                   core::OperationStage::drain,
                                   core::StatusDetail::none,
                                   "collection must be closed before drain");
      }
      while (inflight_operations_ != 0) {
        if (deadline.expired()) {
          return core::Status::error(core::StatusCode::deadline_exceeded,
                                     core::OperationStage::drain,
                                     core::StatusDetail::deadline_reached,
                                     "collection drain deadline was reached");
        }
        if (deadline.enabled) {
          lifecycle_changed_.wait_for(lock, std::chrono::milliseconds(1));
        } else {
          lifecycle_changed_.wait(lock);
        }
      }
      if (lifecycle_ == LifecycleState::closed) {
        return core::Status::success();
      }
    }

    const auto snapshot = load_snapshot();
    for (const auto &entry : snapshot->segments) {
      const auto capabilities = entry->segment.capabilities();
      if (capabilities.supports(core::OperationCapability::close)) {
        const auto status = entry->segment.close();
        if (!status.ok()) {
          return status;
        }
      }
      if (capabilities.supports(core::OperationCapability::drain)) {
        const auto status = entry->segment.drain(deadline);
        if (!status.ok()) {
          return status;
        }
      }
    }
    {
      std::lock_guard lock(lifecycle_mutex_);
      lifecycle_ = LifecycleState::closed;
    }
    return core::Status::success();
  }

  [[nodiscard]] auto persist_experimental_snapshot() const -> core::Status {
    if (!config_.features.experimental_persistence_writer) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::save,
                                 core::StatusDetail::operation_slot_absent,
                                 "experimental collection writer is disabled");
    }
    return ExperimentalSnapshotWriter::write(config_.persistence, *load_snapshot());
  }

  [[nodiscard]] auto checkpoint(core::CheckpointContext &context)
      -> core::Result<CheckpointReceipt> {
    std::lock_guard checkpoint_lock(checkpoint_mutex_);
    auto admission = admit();
    if (!admission.has_value()) {
      return closed_status(core::OperationStage::checkpoint);
    }
    if (!config_.features.wal_coordinator || wal_ == nullptr) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::checkpoint,
                                 core::StatusDetail::operation_slot_absent,
                                 "collection WAL/checkpoint coordinator is disabled");
    }
    auto control = core::validate_runtime_control(context.deadline,
                                                  context.cancellation,
                                                  core::OperationStage::checkpoint);
    if (!control.ok()) {
      return control;
    }
    ControlPlaneGate gate(this);
    control = gate.drain(context.deadline, context.cancellation);
    if (!control.ok()) {
      return control;
    }
    std::lock_guard mutation_lock(mutation_mutex_);
    const auto snapshot = load_snapshot();
    for (const auto &entry : snapshot->segments) {
      if (entry->role != SegmentRole::active_mutable) {
        continue;
      }
      if (!entry->segment.capabilities().supports(core::OperationCapability::checkpoint)) {
        return core::Status::error(core::StatusCode::not_supported,
                                   core::OperationStage::checkpoint,
                                   core::StatusDetail::operation_slot_absent,
                                   "active mutable segment lacks full checkpoint capability");
      }
      core::CheckpointToken token;
      auto status = entry->segment.checkpoint(context, token);
      if (!status.ok()) {
        return status;
      }
    }
    auto stored = CollectionCheckpointStore::write(wal_->directory(),
                                                   *snapshot,
                                                   retry_receipts_,
                                                   batch_retry_receipts_);
    if (!stored.ok()) {
      return stored.status();
    }
    auto receipt = std::move(stored).value();
    const auto status = wal_->reset_to_checkpoint(receipt.wal_cut);
    if (!status.ok()) {
      return status;
    }
    auto durable = std::make_shared<RoutingSnapshot>(*snapshot);
    durable->durable_watermark = receipt.durable_watermark;
    publish_snapshot(std::move(durable));
    return receipt;
  }

  static void apply_checkpoint_to_manifest(const CheckpointReceipt &checkpoint,
                                           ArtifactManifestV2 &manifest) {
    CollectionCheckpointStore::apply_to_manifest(checkpoint, manifest);
  }

  // Internal tests and control-plane code can pin this immutable epoch. The
  // shared reference is the reclamation barrier for every SegmentEntry it owns.
  [[nodiscard]] auto pin_routing_snapshot() const -> RoutingSnapshotPtr { return load_snapshot(); }

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

  [[nodiscard]] auto initialize(std::vector<SegmentRegistration> registrations) -> core::Status {
    auto snapshot = std::make_shared<RoutingSnapshot>();
    std::uint64_t maximum_sequence{};
    for (auto &registration : registrations) {
      const auto descriptor = registration.segment.descriptor();
      if (!registration.segment.capabilities().supports(core::OperationCapability::search) ||
          descriptor.dim != schema_.dim || descriptor.metric != schema_.metric ||
          descriptor.stored_scalar_type != schema_.scalar_type || registration.segment_id == 0 ||
          registration.generation == 0) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::open,
                                   core::StatusDetail::malformed_struct,
                                   "segment registration does not match the collection schema");
      }
      if (snapshot->find_segment(registration.segment_id, registration.generation) != nullptr) {
        return core::Status::error(core::StatusCode::conflict,
                                   core::OperationStage::open,
                                   core::StatusDetail::already_exists,
                                   "duplicate segment identity in routing registration");
      }
      if (registration.role == SegmentRole::active_mutable &&
          !registration.segment.capabilities().supports(core::OperationCapability::mutation)) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::open,
                                   core::StatusDetail::readonly_instance,
                                   "active mutable registration lacks the mutation bundle");
      }
      if (registration.role == SegmentRole::active_mutable &&
          snapshot->find_active_mutable() != nullptr) {
        return core::Status::error(core::StatusCode::conflict,
                                   core::OperationStage::open,
                                   core::StatusDetail::already_exists,
                                   "only one active mutable segment is supported in Gate 4");
      }

      std::uint64_t first_unused = registration.next_row_id;
      for (const auto &row : registration.rows) {
        const auto id_status = validate_logical_id(row.logical_id, core::OperationStage::open);
        if (!id_status.ok()) {
          return id_status;
        }
        if (row.payload.vector.has_value() &&
            (row.payload.vector->dim() != schema_.dim ||
             row.payload.vector->scalar_type() != schema_.scalar_type)) {
          return core::Status::error(core::StatusCode::invalid_argument,
                                     core::OperationStage::open,
                                     core::StatusDetail::dimension_mismatch,
                                     "registered row vector does not match the collection schema");
        }
        const auto row_value = static_cast<std::uint64_t>(row.row_id);
        if (row_value == std::numeric_limits<std::uint64_t>::max()) {
          return core::Status::error(core::StatusCode::invalid_argument,
                                     core::OperationStage::open,
                                     core::StatusDetail::arithmetic_overflow,
                                     "registered row ID leaves no successor value");
        }
        first_unused = std::max(first_unused, row_value + 1);
        const RowAddress address{registration.segment_id, registration.generation, row.row_id};
        if (!snapshot->reverse.emplace(address, ReverseEntry{row.logical_id, row.upsert_sequence})
                 .second) {
          return core::Status::error(core::StatusCode::conflict,
                                     core::OperationStage::open,
                                     core::StatusDetail::already_exists,
                                     "duplicate segment row address in routing registration");
        }
        auto found = snapshot->versions.find(row.logical_id);
        if (found != snapshot->versions.end() &&
            found->second.upsert_sequence == row.upsert_sequence) {
          return core::Status::error(core::StatusCode::conflict,
                                     core::OperationStage::open,
                                     core::StatusDetail::already_exists,
                                     "logical ID has two versions with the same sequence");
        }
        if (found == snapshot->versions.end() ||
            found->second.upsert_sequence < row.upsert_sequence) {
          snapshot->versions
              .insert_or_assign(row.logical_id,
                                VersionEntry{address, row.upsert_sequence, row.state, row.payload});
        }
        maximum_sequence = std::max(maximum_sequence, row.upsert_sequence);
      }
      snapshot->segments.push_back(
          std::make_shared<SegmentEntry>(registration.segment_id,
                                         registration.generation,
                                         registration.role,
                                         std::move(registration.segment),
                                         std::move(registration.exact_rerank),
                                         first_unused,
                                         registration.atomic_mutation_bundle));
    }
    recalculate_counts(*snapshot);
    snapshot->visibility_watermark =
        std::max(maximum_sequence, config_.recovery.minimum_visibility_watermark);
    snapshot->durable_watermark = 0;
    if (config_.features.wal_coordinator) {
      auto opened = CollectionLogicalWal::open(config_.wal.root, config_.wal.namespace_name);
      if (!opened.ok()) {
        return opened.status();
      }
      wal_ = std::move(opened).value();
      const auto recovered = recover_durable_state(snapshot);
      if (!recovered.ok()) {
        return recovered;
      }
    }
    const auto minimum_next = std::max({maximum_sequence + 1,
                                        snapshot->visibility_watermark + 1,
                                        maximum_recovered_op_id_ + 1,
                                        config_.recovery.minimum_next_op_id});
    next_op_id_.store(minimum_next, std::memory_order_release);
    accepted_count_.store(snapshot->searchable_live_count, std::memory_order_release);
    publish_snapshot(std::move(snapshot));
    return core::Status::success();
  }

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

  [[nodiscard]] auto search_at_snapshot(const RoutingSnapshotPtr &snapshot,
                                        const CollectionSearchRequest &request,
                                        bool prefer_exact) -> core::Result<CollectionSearchResult> {
    if (snapshot == nullptr || request.context == nullptr ||
        !core::is_current_struct(*request.context) || !core::is_current_struct(request.options)) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "collection search request is incomplete or incompatible");
    }
    auto status =
        core::validate_tensor(request.queries, schema_.dim, core::OperationStage::validation);
    if (!status.ok()) {
      return status;
    }
    if (request.queries.scalar_type != schema_.scalar_type) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::validation,
                                 core::StatusDetail::unsupported_scalar_type,
                                 "collection does not implicitly convert query scalar types");
    }
    status = core::validate_runtime_control(request.context->deadline,
                                            request.context->cancellation,
                                            core::OperationStage::search);
    if (!status.ok()) {
      return status;
    }

    CollectionSearchResult result;
    result.visibility_watermark = snapshot->visibility_watermark;
    result.metadata_epoch = snapshot->metadata_epoch;
    result.queries.resize(static_cast<std::size_t>(request.queries.rows));
    if (request.options.top_k == 0 || request.queries.rows == 0) {
      for (auto &query : result.queries) {
        query.status = core::Status::success();
        query.completeness = core::SearchCompleteness::complete_k;
      }
      return result;
    }

    if (prefer_exact || request.filter.active()) {
      const auto all_vectors_owned =
          std::all_of(snapshot->versions.begin(), snapshot->versions.end(), [](const auto &item) {
            return item.second.state != VersionState::live ||
                   item.second.payload.vector.has_value();
          });
      if (all_vectors_owned) {
        return exact_search(snapshot, request);
      }
      if (request.options.filter_policy == core::FilterPolicy::strict) {
        return core::Status::error(core::StatusCode::not_supported,
                                   core::OperationStage::search,
                                   core::StatusDetail::operation_slot_absent,
                                   "strict filtered search requires vectors for exact fallback");
      }
    }
    return fanout_search(snapshot, request);
  }

  [[nodiscard]] auto exact_search(const RoutingSnapshotPtr &snapshot,
                                  const CollectionSearchRequest &request)
      -> core::Result<CollectionSearchResult> {
    CollectionSearchResult result;
    result.visibility_watermark = snapshot->visibility_watermark;
    result.metadata_epoch = snapshot->metadata_epoch;
    result.queries.resize(static_cast<std::size_t>(request.queries.rows));
    for (core::RowCount query_index = 0; query_index < request.queries.rows; ++query_index) {
      auto control = core::validate_runtime_control(request.context->deadline,
                                                    request.context->cancellation,
                                                    core::OperationStage::search);
      if (!control.ok()) {
        return control;
      }
      auto &query_result = result.queries[static_cast<std::size_t>(query_index)];
      for (const auto &[logical_id, version] : snapshot->versions) {
        if (version.state != VersionState::live ||
            version.upsert_sequence > snapshot->visibility_watermark ||
            !request.filter.matches(logical_id,
                                    version.payload.metadata,
                                    version.payload.document)) {
          continue;
        }
        if (!version.payload.vector.has_value()) {
          return core::Status::error(core::StatusCode::not_supported,
                                     core::OperationStage::search,
                                     core::StatusDetail::operation_slot_absent,
                                     "exact fallback cannot read a live row vector");
        }
        auto score = exact_distance(query_row(request.queries, query_index),
                                    *version.payload.vector,
                                    schema_.metric);
        if (!score.ok()) {
          return score.status();
        }
        query_result.hits.push_back(CollectionHit{logical_id,
                                                  std::move(score).value(),
                                                  core::ScoreKind::distance,
                                                  schema_.metric,
                                                  core::ResultFlag::exact_reranked |
                                                      core::ResultFlag::filtered |
                                                      core::ResultFlag::version_checked,
                                                  version.upsert_sequence,
                                                  version.address});
        if (request.context->stats != nullptr) {
          ++request.context->stats->filter_candidates;
          ++request.context->stats->rerank_count;
        }
      }
      sort_hits(query_result.hits);
      if (query_result.hits.size() > request.options.top_k) {
        query_result.hits.resize(static_cast<std::size_t>(request.options.top_k));
      }
      query_result.status = core::Status::success();
      query_result.completeness = query_result.hits.size() == request.options.top_k
                                      ? core::SearchCompleteness::complete_k
                                      : core::SearchCompleteness::eligible_exhausted;
    }
    return result;
  }

  [[nodiscard]] auto fanout_search(const RoutingSnapshotPtr &snapshot,
                                   const CollectionSearchRequest &request)
      -> core::Result<CollectionSearchResult> {
    std::vector<std::vector<Candidate>> candidates(static_cast<std::size_t>(request.queries.rows));
    std::vector<bool> exhaustive(static_cast<std::size_t>(request.queries.rows), true);
    std::vector<core::Status> per_query_status(static_cast<std::size_t>(request.queries.rows),
                                               core::Status::success());

    for (const auto &entry : snapshot->segments) {
      const auto known_rows = snapshot->known_rows_for(*entry);
      if (known_rows == 0) {
        continue;
      }
      const auto candidate_limit = std::max<core::RowCount>(request.options.top_k, known_rows);
      std::uint64_t sink_count{};
      if (!core::checked_multiply(request.queries.rows, candidate_limit, sink_count) ||
          sink_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::search,
                                   core::StatusDetail::arithmetic_overflow,
                                   "collection fanout sink size is not representable");
      }
      SegmentSearchStorage storage(request.queries.rows, candidate_limit);
      core::SearchRequest segment_request;
      segment_request.queries = request.queries;
      segment_request.options = request.options;
      segment_request.options.top_k = candidate_limit;
      // Gate 4 owns the exact post-filter seam. Engine capability negotiation
      // and physical pushdown remain Gate 10, so the frozen view stays `none`.
      segment_request.filter = core::SegmentFilterView{};
      segment_request.context = request.context;
      segment_request.response = &storage.response;
      segment_request.lifetime_pin = std::const_pointer_cast<RoutingSnapshot>(snapshot);

      const auto capabilities = entry->segment.capabilities();
      core::Status segment_status;
      if (capabilities.concurrency.reentrant_search) {
        std::shared_lock operation_lock(entry->operation_mutex);
        segment_status = entry->segment.search(std::move(segment_request));
      } else {
        std::unique_lock operation_lock(entry->operation_mutex);
        segment_status = entry->segment.search(std::move(segment_request));
      }
      if (!segment_status.ok()) {
        return segment_status;
      }
      auto response_status =
          validate_segment_response(storage.response, request.queries.rows, candidate_limit);
      if (!response_status.ok()) {
        return response_status;
      }

      for (core::RowCount query_index = 0; query_index < request.queries.rows; ++query_index) {
        const auto index = static_cast<std::size_t>(query_index);
        if (!storage.statuses[index].ok()) {
          per_query_status[index] = storage.statuses[index];
          exhaustive[index] = false;
          continue;
        }
        exhaustive[index] =
            exhaustive[index] &&
            (storage.completeness[index] == core::SearchCompleteness::eligible_exhausted ||
             storage.counts[index] >= known_rows);
        for (core::RowCount hit_index = storage.offsets[index];
             hit_index < storage.offsets[index + 1];
             ++hit_index) {
          const auto &hit = storage.hits[static_cast<std::size_t>(hit_index)];
          if (hit.score_kind != core::ScoreKind::rank_only && is_nan_score(hit.score)) {
            return core::Status::error(core::StatusCode::internal,
                                       core::OperationStage::search,
                                       core::StatusDetail::invalid_score,
                                       "numeric segment returned a NaN score");
          }
          const RowAddress address{entry->segment_id, entry->generation, hit.row_id};
          const auto reverse = snapshot->reverse.find(address);
          if (reverse == snapshot->reverse.end() ||
              reverse->second.upsert_sequence > snapshot->visibility_watermark) {
            continue;  // dark or newer than the admitted watermark
          }
          const auto version = snapshot->versions.find(reverse->second.logical_id);
          if (version == snapshot->versions.end() || version->second.state != VersionState::live ||
              version->second.upsert_sequence != reverse->second.upsert_sequence ||
              version->second.address != address ||
              version->second.upsert_sequence > snapshot->visibility_watermark ||
              !request.filter.matches(version->first,
                                      version->second.payload.metadata,
                                      version->second.payload.document)) {
            continue;
          }
          auto flags = hit.result_flags | core::ResultFlag::version_checked;
          if (request.filter.active()) {
            flags = flags | core::ResultFlag::filtered;
          }
          candidates[index].push_back(Candidate{CollectionHit{version->first,
                                                              hit.score,
                                                              hit.score_kind,
                                                              hit.comparable_metric,
                                                              flags,
                                                              version->second.upsert_sequence,
                                                              address},
                                                entry,
                                                &version->second.payload});
        }
      }
    }

    CollectionSearchResult result;
    result.visibility_watermark = snapshot->visibility_watermark;
    result.metadata_epoch = snapshot->metadata_epoch;
    result.queries.resize(static_cast<std::size_t>(request.queries.rows));
    for (core::RowCount query_index = 0; query_index < request.queries.rows; ++query_index) {
      const auto index = static_cast<std::size_t>(query_index);
      auto &query_result = result.queries[index];
      if (!per_query_status[index].ok()) {
        query_result.status = per_query_status[index];
        query_result.completeness = core::SearchCompleteness::failed;
        continue;
      }
      auto normalized = normalize_scores(candidates[index],
                                         query_row(request.queries, query_index),
                                         request.context);
      if (!normalized.ok()) {
        return normalized.status();
      }
      query_result.hits = std::move(normalized).value();
      sort_hits(query_result.hits);
      query_result.hits.erase(std::unique(query_result.hits.begin(),
                                          query_result.hits.end(),
                                          [](const CollectionHit &lhs, const CollectionHit &rhs) {
                                            return lhs.logical_id == rhs.logical_id;
                                          }),
                              query_result.hits.end());
      if (query_result.hits.size() > request.options.top_k) {
        query_result.hits.resize(static_cast<std::size_t>(request.options.top_k));
      }
      query_result.status = core::Status::success();
      query_result.completeness =
          query_result.hits.size() == request.options.top_k ? core::SearchCompleteness::complete_k
          : exhaustive[index] ? core::SearchCompleteness::eligible_exhausted
                              : core::SearchCompleteness::strategy_incomplete;
    }
    return result;
  }

  [[nodiscard]] auto normalize_scores(const std::vector<Candidate> &candidates,
                                      const core::TypedTensorView &query,
                                      core::SearchContext *context)
      -> core::Result<std::vector<CollectionHit>> {
    std::vector<CollectionHit> result;
    result.reserve(candidates.size());
    std::optional<std::pair<core::ScoreKind, core::Metric>> domain;
    for (const auto &candidate : candidates) {
      auto hit = candidate.hit;
      if (hit.score_kind == core::ScoreKind::rank_only) {
        core::Result<float> reranked =
            core::Status::error(core::StatusCode::not_supported,
                                core::OperationStage::search,
                                core::StatusDetail::operation_slot_absent,
                                "rank-only segment has no exact rerank source");
        if (candidate.segment->exact_rerank) {
          reranked = candidate.segment->exact_rerank(query, hit.source.row_id);
        } else if (candidate.payload != nullptr && candidate.payload->vector.has_value()) {
          reranked = exact_distance(query, *candidate.payload->vector, schema_.metric);
        }
        if (!reranked.ok()) {
          return reranked.status();
        }
        hit.score = std::move(reranked).value();
        hit.score_kind = core::ScoreKind::distance;
        hit.comparable_metric = schema_.metric;
        hit.result_flags = hit.result_flags | core::ResultFlag::exact_reranked;
        if (context != nullptr && context->stats != nullptr) {
          ++context->stats->rerank_count;
        }
      }
      const auto current_domain = std::pair{hit.score_kind, hit.comparable_metric};
      if (domain.has_value() && *domain != current_domain) {
        return core::Status::error(core::StatusCode::not_supported,
                                   core::OperationStage::search,
                                   core::StatusDetail::invalid_score,
                                   "segment score domains are not comparable");
      }
      domain = current_domain;
      result.push_back(std::move(hit));
    }
    return result;
  }

  [[nodiscard]] auto mutate_locked(RoutingSnapshotPtr current,
                                   const core::LogicalId &logical_id,
                                   SegmentMutationAction action,
                                   RecordPayload payload,
                                   RowMutationStatus row_status,
                                   core::MutationContext &context,
                                   const WriteOptions &options,
                                   std::uint64_t batch_op_id = 0) -> core::Result<MutationReceipt> {
    auto control = core::validate_runtime_control(context.deadline,
                                                  context.cancellation,
                                                  core::OperationStage::admission);
    if (!control.ok()) {
      return control;
    }
    const auto target = current->find_active_mutable();
    if (target == nullptr) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::admission,
                                 core::StatusDetail::readonly_instance,
                                 "collection has no active mutable segment");
    }
    const auto op_id = next_op_id_.fetch_add(1, std::memory_order_acq_rel);
    if (batch_op_id == 0) {
      batch_op_id = op_id;
    }
    const auto row_value = target->next_row_id.fetch_add(1, std::memory_order_acq_rel);
    if (row_value == std::numeric_limits<std::uint64_t>::max()) {
      return core::Status::error(core::StatusCode::resource_exhausted,
                                 core::OperationStage::admission,
                                 core::StatusDetail::arithmetic_overflow,
                                 "active segment exhausted its row ID space");
    }
    const RowAddress address{target->segment_id, target->generation, core::SegmentRowId(row_value)};
    const auto previous = current->versions.find(logical_id);

    WalMutationTransaction transaction;
    transaction.batch_op_id = batch_op_id;
    transaction.batch_mode = BatchMutationMode::per_row_independent;
    transaction.durability = options.durability;
    WalMutationRow row;
    row.op_id = op_id;
    row.action = action;
    row.status = row_status;
    row.logical_id = logical_id;
    row.target = address;
    if (previous != current->versions.end()) {
      row.previous = previous->second.address;
    }
    row.payload = std::move(payload);
    row.retry_token = options.retry_token;
    transaction.rows.push_back(std::move(row));
    auto executed =
        execute_transaction_locked(std::move(current), target, transaction, context, op_id);
    if (!executed.ok()) {
      return executed.status();
    }
    return std::move(executed).value().front();
  }

  struct ValidatedBatchRow {
    bool valid{};
    SegmentMutationAction action{SegmentMutationAction::write};
    RowMutationStatus status{RowMutationStatus::invalid_argument};
    RecordPayload payload{};
    std::optional<RowAddress> previous{};
  };

  [[nodiscard]] auto validate_batch_row(const RoutingSnapshotPtr &current,
                                        const BatchRowMutation &row) const -> ValidatedBatchRow {
    ValidatedBatchRow result;
    if (!validate_logical_id(row.logical_id, core::OperationStage::validation).ok()) {
      return result;
    }
    const auto found = current->versions.find(row.logical_id);
    const auto live = found != current->versions.end() && found->second.state == VersionState::live;
    if (found != current->versions.end()) {
      result.previous = found->second.address;
    }
    if (row.action == RowMutationAction::erase) {
      result.action = SegmentMutationAction::erase;
      if (!live) {
        result.status = RowMutationStatus::not_found;
        return result;
      }
      result.payload = found->second.payload;
      result.status = RowMutationStatus::deleted;
      result.valid = true;
      return result;
    }
    auto tensor = core::validate_tensor(row.vector, schema_.dim, core::OperationStage::validation);
    if (!tensor.ok() || row.vector.rows != 1 || row.vector.scalar_type != schema_.scalar_type) {
      return result;
    }
    if (row.write_mode == WriteMode::insert_only && live) {
      result.status = RowMutationStatus::already_exists;
      return result;
    }
    if (row.write_mode == WriteMode::replace && !live) {
      result.status = RowMutationStatus::not_found;
      return result;
    }
    auto owned = OwnedVector::copy_row(row.vector, 0);
    if (!owned.ok()) {
      return result;
    }
    result.payload.vector = std::move(owned).value();
    result.payload.metadata = row.metadata;
    result.payload.document = row.document;
    result.status = !live                                  ? RowMutationStatus::inserted
                    : row.write_mode == WriteMode::replace ? RowMutationStatus::replaced
                                                           : RowMutationStatus::updated;
    result.valid = true;
    return result;
  }

  [[nodiscard]] auto make_non_searchable_receipt(const RoutingSnapshotPtr &current,
                                                 std::uint64_t batch_op_id,
                                                 RowMutationStatus status,
                                                 std::string retry_token) -> MutationReceipt {
    const auto op_id = next_op_id_.fetch_add(1, std::memory_order_acq_rel);
    MutationReceipt receipt;
    receipt.op_id = op_id;
    receipt.batch_op_id = batch_op_id;
    receipt.row_op_id = op_id;
    receipt.visibility_watermark = current->visibility_watermark;
    receipt.durable_watermark = current->durable_watermark;
    receipt.searchable = false;
    receipt.durability = DurabilityState::memory_only;
    receipt.row_status = status;
    receipt.retry_token = std::move(retry_token);
    if (!receipt.retry_token.empty()) {
      retry_receipts_.insert_or_assign(receipt.retry_token, receipt);
    }
    return receipt;
  }

  [[nodiscard]] auto mutate_batch_row_locked(RoutingSnapshotPtr current,
                                             const BatchRowMutation &row,
                                             core::MutationContext &context,
                                             const WriteOptions &options,
                                             std::uint64_t batch_op_id)
      -> core::Result<MutationReceipt> {
    auto validated = validate_batch_row(current, row);
    if (!validated.valid) {
      return make_non_searchable_receipt(current,
                                         batch_op_id,
                                         validated.status,
                                         options.retry_token);
    }
    return mutate_locked(std::move(current),
                         row.logical_id,
                         validated.action,
                         std::move(validated.payload),
                         validated.status,
                         context,
                         options,
                         batch_op_id);
  }

  [[nodiscard]] auto mutate_atomic_batch_locked(RoutingSnapshotPtr current,
                                                const BatchMutationRequest &request,
                                                core::MutationContext &context,
                                                std::uint64_t batch_op_id)
      -> core::Result<BatchMutationReceipt> {
    const auto target = current->find_active_mutable();
    if (target == nullptr || !target->atomic_mutation_bundle) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::admission,
                                 core::StatusDetail::operation_slot_absent,
                                 "active engine does not support an atomic mutation bundle");
    }
    std::vector<ValidatedBatchRow> validated;
    validated.reserve(request.rows.size());
    std::map<core::LogicalId, std::size_t, LogicalIdLess> first_occurrence;
    std::optional<std::size_t> failed;
    RowMutationStatus failed_status{RowMutationStatus::invalid_argument};
    for (std::size_t index = 0; index < request.rows.size(); ++index) {
      auto [unused, inserted] = first_occurrence.emplace(request.rows[index].logical_id, index);
      (void)unused;
      auto row = validate_batch_row(current, request.rows[index]);
      if (!inserted) {
        row.valid = false;
        row.status = RowMutationStatus::conflict;
      }
      if (!row.valid && !failed.has_value()) {
        failed = index;
        failed_status = row.status;
      }
      validated.push_back(std::move(row));
    }
    if (failed.has_value()) {
      BatchMutationReceipt receipt;
      receipt.batch_op_id = batch_op_id;
      receipt.visibility_watermark = current->visibility_watermark;
      receipt.durable_watermark = current->durable_watermark;
      receipt.retry_token = request.options.retry_token;
      receipt.rows.reserve(request.rows.size());
      for (std::size_t index = 0; index < request.rows.size(); ++index) {
        auto token = request.rows[index].retry_token;
        if (token.empty() && !request.options.retry_token.empty()) {
          token = request.options.retry_token + "#" + std::to_string(index);
        }
        receipt.rows.push_back(make_non_searchable_receipt(current,
                                                           batch_op_id,
                                                           index == *failed
                                                               ? failed_status
                                                               : RowMutationStatus::aborted,
                                                           std::move(token)));
      }
      auto marker = persist_batch_receipt(receipt, request.options.durability);
      if (!marker.ok()) {
        return marker;
      }
      if (!receipt.retry_token.empty()) {
        batch_retry_receipts_.insert_or_assign(receipt.retry_token, receipt);
      }
      return receipt;
    }

    WalMutationTransaction transaction;
    transaction.batch_op_id = batch_op_id;
    transaction.batch_mode = BatchMutationMode::all_or_nothing;
    transaction.durability = request.options.durability;
    transaction.retry_token = request.options.retry_token;
    transaction.rows.reserve(request.rows.size());
    for (std::size_t index = 0; index < request.rows.size(); ++index) {
      const auto row_value = target->next_row_id.fetch_add(1, std::memory_order_acq_rel);
      if (row_value == std::numeric_limits<std::uint64_t>::max()) {
        return core::Status::error(core::StatusCode::resource_exhausted,
                                   core::OperationStage::admission,
                                   core::StatusDetail::arithmetic_overflow,
                                   "active segment exhausted its row ID space");
      }
      WalMutationRow row;
      row.op_id = next_op_id_.fetch_add(1, std::memory_order_acq_rel);
      row.action = validated[index].action;
      row.status = validated[index].status;
      row.logical_id = request.rows[index].logical_id;
      row.target = {target->segment_id, target->generation, core::SegmentRowId(row_value)};
      row.previous = validated[index].previous;
      row.payload = std::move(validated[index].payload);
      row.retry_token = request.rows[index].retry_token;
      if (row.retry_token.empty() && !request.options.retry_token.empty()) {
        row.retry_token = request.options.retry_token + "#" + std::to_string(index);
      }
      transaction.rows.push_back(std::move(row));
    }
    auto executed =
        execute_transaction_locked(std::move(current), target, transaction, context, batch_op_id);
    if (!executed.ok()) {
      if (executed.status().stage() == core::OperationStage::mutation_publish) {
        return executed.status();
      }
      const auto snapshot = load_snapshot();
      BatchMutationReceipt aborted;
      aborted.batch_op_id = batch_op_id;
      aborted.visibility_watermark = snapshot->visibility_watermark;
      aborted.durable_watermark = snapshot->durable_watermark;
      aborted.retry_token = request.options.retry_token;
      for (const auto &row : transaction.rows) {
        MutationReceipt receipt;
        receipt.op_id = row.op_id;
        receipt.batch_op_id = batch_op_id;
        receipt.row_op_id = row.op_id;
        receipt.visibility_watermark = snapshot->visibility_watermark;
        receipt.durable_watermark = snapshot->durable_watermark;
        receipt.row_status = RowMutationStatus::aborted;
        receipt.retry_token = row.retry_token;
        if (!receipt.retry_token.empty()) {
          retry_receipts_.insert_or_assign(receipt.retry_token, receipt);
        }
        aborted.rows.push_back(std::move(receipt));
      }
      if (!aborted.retry_token.empty()) {
        auto marker = persist_batch_receipt(aborted, request.options.durability);
        if (!marker.ok()) {
          return marker;
        }
        batch_retry_receipts_.insert_or_assign(aborted.retry_token, aborted);
      }
      return aborted;
    }
    const auto snapshot = load_snapshot();
    BatchMutationReceipt receipt;
    receipt.batch_op_id = batch_op_id;
    receipt.visibility_watermark = snapshot->visibility_watermark;
    receipt.durable_watermark = snapshot->durable_watermark;
    receipt.searchable = true;
    receipt.durability = config_.features.wal_coordinator
                             ? durability_state(request.options.durability)
                             : DurabilityState::memory_only;
    receipt.retry_token = request.options.retry_token;
    receipt.rows = std::move(executed).value();
    auto marker = persist_batch_receipt(receipt, request.options.durability);
    if (!marker.ok()) {
      return marker;
    }
    if (!receipt.retry_token.empty()) {
      batch_retry_receipts_.insert_or_assign(receipt.retry_token, receipt);
    }
    return receipt;
  }

  [[nodiscard]] static auto durability_state(WriteDurability durability) -> DurabilityState {
    return durability == WriteDurability::wal_fsync ? DurabilityState::wal_fsync
                                                    : DurabilityState::searchable_not_durable;
  }

  [[nodiscard]] auto persist_batch_receipt(const BatchMutationReceipt &receipt,
                                           WriteDurability durability) -> core::Status {
    if (wal_ == nullptr || receipt.retry_token.empty()) {
      return core::Status::success();
    }
    try {
      const auto payload = encode_batch_receipt_marker(receipt);
      const auto durable = durability == WriteDurability::wal_fsync;
      return wal_->append(LogicalWalRecordType::publish_marker,
                          static_cast<std::uint8_t>(0x80U | (durable ? 1U : 0U)),
                          receipt.batch_op_id,
                          receipt.batch_op_id,
                          payload,
                          durable ? LogicalWalSync::fsync : LogicalWalSync::buffered);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::completion);
    }
  }

  [[nodiscard]] auto failpoint(MutationFailPoint point) -> bool {
    if (config_.failpoint_hook) {
      config_.failpoint_hook(point);
    }
    return config_.fail_point == point;
  }

  [[nodiscard]] static auto injected_failure(MutationFailPoint point) -> core::Status {
    const auto stage =
        point == MutationFailPoint::after_commit || point == MutationFailPoint::after_publish
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

  [[nodiscard]] auto make_engine_payloads(const WalMutationTransaction &transaction)
      -> std::vector<SegmentMutationPayload> {
    std::vector<SegmentMutationPayload> payloads;
    payloads.reserve(transaction.rows.size());
    for (const auto &row : transaction.rows) {
      SegmentMutationPayload payload;
      payload.action = row.action;
      payload.op_id = row.op_id;
      payload.upsert_sequence = row.op_id;
      payload.target = row.target;
      payload.previous = row.previous;
      if (row.action == SegmentMutationAction::write && row.payload.vector.has_value()) {
        payload.vector = row.payload.vector->view();
      }
      payloads.push_back(std::move(payload));
    }
    return payloads;
  }

  [[nodiscard]] auto build_dark_snapshot(const RoutingSnapshotPtr &current,
                                         const WalMutationTransaction &transaction,
                                         bool durable)
      -> core::Result<std::shared_ptr<RoutingSnapshot>> {
    try {
      auto next = std::make_shared<RoutingSnapshot>(*current);
      next->generation = current->generation + 1;
      next->metadata_epoch = current->metadata_epoch + 1;
      for (const auto &row : transaction.rows) {
        next->visibility_watermark = std::max(next->visibility_watermark, row.op_id);
        next->reverse.insert_or_assign(row.target, ReverseEntry{row.logical_id, row.op_id});
        next->versions.insert_or_assign(row.logical_id,
                                        VersionEntry{row.target,
                                                     row.op_id,
                                                     row.action == SegmentMutationAction::write
                                                         ? VersionState::live
                                                         : VersionState::tombstone,
                                                     row.payload});
      }
      if (durable) {
        next->durable_watermark = next->visibility_watermark;
      }
      recalculate_counts(*next);
      return next;
    } catch (...) {
      return core::status_from_exception(core::OperationStage::mutation_stage);
    }
  }

  [[nodiscard]] auto execute_transaction_locked(RoutingSnapshotPtr current,
                                                const std::shared_ptr<SegmentEntry> &target,
                                                const WalMutationTransaction &transaction,
                                                core::MutationContext &context,
                                                std::uint64_t transaction_id)
      -> core::Result<std::vector<MutationReceipt>> {
    if (failpoint(MutationFailPoint::before_prepare)) {
      return injected_failure(MutationFailPoint::before_prepare);
    }
    std::vector<std::byte> wal_payload;
    try {
      wal_payload = encode_wal_transaction(transaction);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::mutation_prepare);
    }
    const auto durable =
        config_.features.wal_coordinator && transaction.durability == WriteDurability::wal_fsync;
    if (wal_ != nullptr) {
      auto status = wal_->append(LogicalWalRecordType::prepare,
                                 durable ? 1U : 0U,
                                 transaction_id,
                                 transaction.batch_op_id,
                                 wal_payload,
                                 durable ? LogicalWalSync::flush : LogicalWalSync::buffered);
      if (!status.ok()) {
        return status;
      }
    }
    if (failpoint(MutationFailPoint::after_prepare)) {
      return injected_failure(MutationFailPoint::after_prepare);
    }

    auto dark = build_dark_snapshot(current, transaction, durable);
    if (!dark.ok()) {
      return dark.status();
    }
    auto engine_payloads = make_engine_payloads(transaction);
    SegmentMutationBundlePayload bundle;
    bundle.batch_op_id = transaction.batch_op_id;
    bundle.rows = engine_payloads;
    core::OpaqueOperationRequest opaque;
    const auto bundled =
        transaction.batch_mode == BatchMutationMode::all_or_nothing && transaction.rows.size() > 1;
    opaque.payload = bundled ? static_cast<const void *>(&bundle)
                             : static_cast<const void *>(&engine_payloads.front());
    opaque.payload_size = bundled ? sizeof(bundle) : sizeof(SegmentMutationPayload);
    core::MutationContext engine_context = context;
    engine_context.transaction_token = &transaction;
    std::uint64_t pending_bytes{};
    for (const auto &row : transaction.rows) {
      if (row.payload.vector.has_value()) {
        pending_bytes += row.payload.vector->bytes().size();
      }
    }
    PendingGuard pending(this, pending_bytes, transaction.rows.size());
    std::vector<std::unique_ptr<AcceptedGuard>> accepted;
    accepted.reserve(transaction.rows.size());
    for (const auto &row : transaction.rows) {
      accepted.push_back(std::make_unique<AcceptedGuard>(this, row.status));
    }
    core::MutationToken token;
    const auto capabilities = target->segment.capabilities();
    std::unique_lock<std::shared_mutex> exclusion;
    if (!capabilities.concurrency.search_with_stage ||
        !capabilities.concurrency.search_with_publish) {
      exclusion = std::unique_lock<std::shared_mutex>(target->operation_mutex);
    }
    auto status = target->segment.prepare_mutation(opaque, engine_context, token);
    if (!status.ok()) {
      return status;
    }
    status = target->segment.stage_mutation(token, engine_context);
    if (!status.ok()) {
      (void)target->segment.abort_mutation(token, engine_context);
      return status;
    }
    if (failpoint(MutationFailPoint::after_stage)) {
      (void)target->segment.abort_mutation(token, engine_context);
      return injected_failure(MutationFailPoint::after_stage);
    }
    if (failpoint(MutationFailPoint::metadata_stage_failure)) {
      (void)target->segment.abort_mutation(token, engine_context);
      return injected_failure(MutationFailPoint::metadata_stage_failure);
    }
    if (wal_ != nullptr) {
      status = wal_->append(LogicalWalRecordType::commit,
                            durable ? 1U : 0U,
                            transaction_id,
                            transaction.batch_op_id,
                            {},
                            durable ? LogicalWalSync::fsync : LogicalWalSync::buffered);
      if (!status.ok()) {
        (void)target->segment.abort_mutation(token, engine_context);
        return status;
      }
    }
    if (failpoint(MutationFailPoint::after_commit)) {
      return injected_failure(MutationFailPoint::after_commit);
    }
    status = target->segment.publish_mutation(token, engine_context);
    if (!status.ok()) {
      // COMMIT is already authoritative. Recovery must retry publish; aborting
      // here would incorrectly discard a durable transaction.
      return status;
    }
    publish_snapshot(std::move(dark).value());
    for (auto &guard : accepted) {
      guard->commit();
    }
    const auto published = load_snapshot();
    std::vector<MutationReceipt> receipts;
    receipts.reserve(transaction.rows.size());
    for (const auto &row : transaction.rows) {
      MutationReceipt receipt;
      receipt.op_id = row.op_id;
      receipt.batch_op_id = transaction.batch_op_id;
      receipt.row_op_id = row.op_id;
      receipt.visibility_watermark = published->visibility_watermark;
      receipt.durable_watermark = published->durable_watermark;
      receipt.searchable = true;
      receipt.durability =
          wal_ == nullptr ? DurabilityState::memory_only : durability_state(transaction.durability);
      receipt.row_status = row.status;
      receipt.retry_token = row.retry_token;
      if (!receipt.retry_token.empty()) {
        retry_receipts_.insert_or_assign(receipt.retry_token, receipt);
      }
      receipts.push_back(std::move(receipt));
    }
    if (wal_ != nullptr) {
      status = wal_->append(LogicalWalRecordType::publish_marker,
                            durable ? 1U : 0U,
                            transaction_id,
                            transaction.batch_op_id,
                            {},
                            durable ? LogicalWalSync::flush : LogicalWalSync::buffered);
      if (!status.ok()) {
        return status;
      }
    }
    if (failpoint(MutationFailPoint::after_publish)) {
      return injected_failure(MutationFailPoint::after_publish);
    }
    return receipts;
  }

  [[nodiscard]] auto replay_engine_transaction(const WalMutationTransaction &transaction)
      -> core::Status {
    if (transaction.rows.empty()) {
      return core::Status::success();
    }
    const auto target =
        load_or_initializing_snapshot_->find_segment(transaction.rows.front().target.segment_id,
                                                     transaction.rows.front().target.generation);
    if (target == nullptr ||
        !target->segment.capabilities().supports(core::OperationCapability::mutation)) {
      return core::Status::error(core::StatusCode::corruption,
                                 core::OperationStage::mutation_replay,
                                 core::StatusDetail::readonly_instance,
                                 "committed WAL targets an unavailable mutable segment");
    }
    for (const auto &row : transaction.rows) {
      if (row.target.segment_id != target->segment_id ||
          row.target.generation != target->generation) {
        return core::Status::error(core::StatusCode::corruption,
                                   core::OperationStage::mutation_replay,
                                   core::StatusDetail::malformed_struct,
                                   "one WAL transaction targets multiple segment instances");
      }
    }
    auto payloads = make_engine_payloads(transaction);
    SegmentMutationBundlePayload bundle;
    bundle.batch_op_id = transaction.batch_op_id;
    bundle.rows = payloads;
    const auto bundled =
        transaction.batch_mode == BatchMutationMode::all_or_nothing && transaction.rows.size() > 1;
    core::OpaqueOperationRequest opaque;
    opaque.payload =
        bundled ? static_cast<const void *>(&bundle) : static_cast<const void *>(&payloads.front());
    opaque.payload_size = bundled ? sizeof(bundle) : sizeof(SegmentMutationPayload);
    core::MutationContext context;
    context.transaction_token = &transaction;
    return target->segment.replay_mutation(opaque, context);
  }

  void install_recovered_receipts(const WalMutationTransaction &transaction,
                                  const RoutingSnapshot &snapshot,
                                  DurabilityState durability) {
    BatchMutationReceipt batch;
    batch.batch_op_id = transaction.batch_op_id;
    batch.visibility_watermark = snapshot.visibility_watermark;
    batch.durable_watermark = snapshot.durable_watermark;
    batch.searchable = true;
    batch.durability = durability;
    batch.retry_token = transaction.retry_token;
    for (const auto &row : transaction.rows) {
      MutationReceipt receipt;
      receipt.op_id = row.op_id;
      receipt.batch_op_id = transaction.batch_op_id;
      receipt.row_op_id = row.op_id;
      receipt.visibility_watermark = snapshot.visibility_watermark;
      receipt.durable_watermark = snapshot.durable_watermark;
      receipt.searchable = true;
      receipt.durability = durability;
      receipt.row_status = row.status;
      receipt.retry_token = row.retry_token;
      if (!receipt.retry_token.empty()) {
        retry_receipts_.insert_or_assign(receipt.retry_token, receipt);
      }
      batch.rows.push_back(std::move(receipt));
    }
    if (!batch.retry_token.empty()) {
      batch_retry_receipts_.insert_or_assign(batch.retry_token, std::move(batch));
    }
  }

  [[nodiscard]] auto apply_checkpoint_image(std::shared_ptr<RoutingSnapshot> &snapshot,
                                            CollectionCheckpointImage image) -> core::Status {
    snapshot->versions.clear();
    snapshot->reverse.clear();
    snapshot->generation = image.generation;
    snapshot->visibility_watermark = image.visibility_watermark;
    snapshot->durable_watermark = image.durable_watermark;
    snapshot->metadata_epoch = image.metadata_epoch;
    for (const auto &row : image.state.rows) {
      const auto target = snapshot->find_segment(row.target.segment_id, row.target.generation);
      if (target == nullptr) {
        return core::Status::error(core::StatusCode::corruption,
                                   core::OperationStage::open,
                                   core::StatusDetail::malformed_struct,
                                   "checkpoint targets an unregistered segment instance");
      }
      snapshot->reverse.insert_or_assign(row.target, ReverseEntry{row.logical_id, row.op_id});
      snapshot->versions.insert_or_assign(row.logical_id,
                                          VersionEntry{row.target,
                                                       row.op_id,
                                                       row.action == SegmentMutationAction::write
                                                           ? VersionState::live
                                                           : VersionState::tombstone,
                                                       row.payload});
      const auto row_id = static_cast<std::uint64_t>(row.target.row_id);
      if (row_id != std::numeric_limits<std::uint64_t>::max()) {
        auto current = target->next_row_id.load(std::memory_order_acquire);
        while (current <= row_id &&
               !target->next_row_id.compare_exchange_weak(current,
                                                          row_id + 1,
                                                          std::memory_order_acq_rel)) {
        }
      }
      maximum_recovered_op_id_ = std::max(maximum_recovered_op_id_, row.op_id);
    }
    recalculate_counts(*snapshot);
    retry_receipts_ = std::move(image.retry_receipts);
    batch_retry_receipts_ = std::move(image.batch_retry_receipts);
    // A fresh fake/engine instance rebuilds its current mutable view through
    // the idempotent replay seam; a persistent engine treats this as a no-op.
    load_or_initializing_snapshot_ = snapshot;
    for (const auto &row : image.state.rows) {
      const auto target = snapshot->find_segment(row.target.segment_id, row.target.generation);
      if (target != nullptr && target->role == SegmentRole::active_mutable) {
        WalMutationTransaction single;
        single.batch_op_id = row.op_id;
        single.batch_mode = BatchMutationMode::per_row_independent;
        single.durability = WriteDurability::wal_fsync;
        single.rows.push_back(row);
        const auto status = replay_engine_transaction(single);
        if (!status.ok()) {
          return status;
        }
      }
    }
    return core::Status::success();
  }

  [[nodiscard]] auto recover_durable_state(std::shared_ptr<RoutingSnapshot> &snapshot)
      -> core::Status {
    load_or_initializing_snapshot_ = snapshot;
    auto checkpoint = CollectionCheckpointStore::load(wal_->directory());
    if (!checkpoint.ok()) {
      return checkpoint.status();
    }
    std::uint64_t wal_cut{};
    if (checkpoint.value().has_value()) {
      wal_cut = checkpoint.value()->wal_cut;
      auto status = apply_checkpoint_image(snapshot, std::move(*checkpoint.value()));
      if (!status.ok()) {
        return status;
      }
    }
    struct Pending {
      WalMutationTransaction transaction{};
      std::uint64_t transaction_id{};
    };
    struct Committed {
      WalMutationTransaction transaction{};
      std::uint64_t transaction_id{};
      bool durable{};
      bool publish_marker{};
    };
    std::map<std::uint64_t, Pending> pending;
    std::vector<Committed> committed;
    std::map<std::uint64_t, std::size_t> committed_index;
    try {
      for (const auto &frame : wal_->recovery_scan().frames) {
        maximum_recovered_op_id_ = std::max(maximum_recovered_op_id_, frame.op_id);
        if (frame.type == LogicalWalRecordType::checkpoint) {
          wal_cut = std::max(wal_cut, frame.op_id);
          continue;
        }
        if (frame.type == LogicalWalRecordType::prepare) {
          auto transaction = decode_wal_transaction(frame.payload);
          if (transaction.rows.empty()) {
            return core::Status::error(core::StatusCode::corruption,
                                       core::OperationStage::mutation_replay,
                                       core::StatusDetail::malformed_struct,
                                       "WAL PREPARE contains no mutation rows");
          }
          for (const auto &row : transaction.rows) {
            maximum_recovered_op_id_ = std::max(maximum_recovered_op_id_, row.op_id);
          }
          pending.insert_or_assign(frame.op_id, Pending{std::move(transaction), frame.op_id});
          continue;
        }
        if (frame.type == LogicalWalRecordType::commit) {
          const auto found = pending.find(frame.op_id);
          if (found == pending.end() || committed_index.contains(frame.op_id)) {
            continue;
          }
          committed_index.emplace(frame.op_id, committed.size());
          committed.push_back(Committed{std::move(found->second.transaction),
                                        frame.op_id,
                                        (frame.flags & 1U) != 0,
                                        false});
          pending.erase(found);
          continue;
        }
        if ((frame.flags & 0x80U) != 0U) {
          auto receipt = decode_batch_receipt_marker(frame.payload);
          if (receipt.batch_op_id != frame.batch_id || receipt.retry_token.empty()) {
            return core::Status::error(core::StatusCode::corruption,
                                       core::OperationStage::mutation_replay,
                                       core::StatusDetail::malformed_struct,
                                       "batch receipt WAL marker identity is invalid");
          }
          batch_retry_receipts_.insert_or_assign(receipt.retry_token, std::move(receipt));
          continue;
        }
        const auto found = committed_index.find(frame.op_id);
        if (found != committed_index.end()) {
          committed[found->second].publish_marker = true;
        }
      }
    } catch (const std::invalid_argument &error) {
      return core::Status::error(core::StatusCode::corruption,
                                 core::OperationStage::mutation_replay,
                                 core::StatusDetail::malformed_struct,
                                 error.what());
    } catch (...) {
      return core::status_from_exception(core::OperationStage::mutation_replay);
    }

    const auto active = snapshot->find_active_mutable();
    if (active != nullptr) {
      core::MutationContext context;
      for (const auto &[unused, transaction] : pending) {
        (void)unused;
        core::MutationToken token;
        token.value = transaction.transaction_id;
        (void)active->segment.abort_mutation(token, context);
      }
    }

    for (auto &entry : committed) {
      const auto maximum_row =
          std::ranges::max(entry.transaction.rows, {}, &WalMutationRow::op_id).op_id;
      if (maximum_row > wal_cut && maximum_row > snapshot->visibility_watermark) {
        load_or_initializing_snapshot_ = snapshot;
        auto status = replay_engine_transaction(entry.transaction);
        if (!status.ok()) {
          return status;
        }
        auto next = build_dark_snapshot(snapshot, entry.transaction, entry.durable);
        if (!next.ok()) {
          return next.status();
        }
        snapshot = std::move(next).value();
        load_or_initializing_snapshot_ = snapshot;
      }
      install_recovered_receipts(entry.transaction,
                                 *snapshot,
                                 entry.durable ? DurabilityState::wal_fsync
                                               : DurabilityState::searchable_not_durable);
      if (!entry.publish_marker) {
        const auto status = wal_->append(LogicalWalRecordType::publish_marker,
                                         entry.durable ? 1U : 0U,
                                         entry.transaction_id,
                                         entry.transaction.batch_op_id,
                                         {},
                                         LogicalWalSync::flush);
        if (!status.ok()) {
          return status;
        }
      }
    }
    load_or_initializing_snapshot_.reset();
    return core::Status::success();
  }

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
      -> core::Result<CollectionRecord> {
    CollectionRecord result;
    result.logical_id = logical_id;
    result.upsert_sequence = version.upsert_sequence;
    if (projection_contains(projection, Projection::vector)) {
      if (!version.payload.vector.has_value()) {
        return core::Status::error(core::StatusCode::not_supported,
                                   core::OperationStage::search,
                                   core::StatusDetail::operation_slot_absent,
                                   "requested vector projection is unavailable");
      }
      result.vector = version.payload.vector;
    }
    if (projection_contains(projection, Projection::metadata)) {
      result.metadata = version.payload.metadata;
    }
    if (projection_contains(projection, Projection::document)) {
      result.document = version.payload.document;
    }
    return result;
  }

  [[nodiscard]] static auto validate_segment_response(const core::SearchResponse &response,
                                                      core::RowCount query_count,
                                                      core::RowCount top_k) -> core::Status {
    if (response.query_count != query_count || response.offsets.empty() ||
        response.offsets[0] != 0) {
      return malformed_engine_response();
    }
    for (core::RowCount query = 0; query < query_count; ++query) {
      const auto index = static_cast<std::size_t>(query);
      if (response.offsets[index + 1] < response.offsets[index] ||
          response.offsets[index + 1] - response.offsets[index] != response.valid_counts[index] ||
          response.valid_counts[index] > top_k ||
          response.offsets[index + 1] > response.hits.size()) {
        return malformed_engine_response();
      }
    }
    return core::Status::success();
  }

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
                                           core::Metric metric) -> core::Result<float> {
    if (query.rows != 1 || query.dim != vector.dim() || query.scalar_type != vector.scalar_type()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::search,
                                 core::StatusDetail::dimension_mismatch,
                                 "exact rerank inputs do not share a tensor schema");
    }
    float score{};
    switch (query.scalar_type) {
      case core::ScalarType::float32:
        score = exact_distance_typed<float>(query, vector, metric);
        break;
      case core::ScalarType::int8:
        score = exact_distance_typed<std::int8_t>(query, vector, metric);
        break;
      case core::ScalarType::uint8:
        score = exact_distance_typed<std::uint8_t>(query, vector, metric);
        break;
    }
    if (is_nan_score(score)) {
      return core::Status::error(core::StatusCode::internal,
                                 core::OperationStage::search,
                                 core::StatusDetail::invalid_score,
                                 "exact rerank produced a NaN score");
    }
    return score;
  }

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
  }

  [[nodiscard]] auto closed_status(core::OperationStage stage) const -> core::Status {
    std::lock_guard lock(lifecycle_mutex_);
    if (lifecycle_ == LifecycleState::open && control_plane_gate_) {
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
  std::mutex drain_mutex_{};
  std::mutex checkpoint_mutex_{};
  std::mutex mutation_mutex_{};
  std::atomic_uint64_t next_op_id_{1};
  std::atomic_uint64_t accepted_count_{};
  std::atomic_uint64_t pending_count_{};
  std::atomic_uint64_t pending_bytes_{};
};

}  // namespace alaya::internal::collection
