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
                         context);
  }

  [[nodiscard]] auto erase(const core::LogicalId &logical_id, core::MutationContext &context)
      -> core::Result<MutationReceipt> {
    auto admission = admit();
    if (!admission.has_value()) {
      return closed_status(core::OperationStage::admission);
    }
    std::lock_guard mutation_lock(mutation_mutex_);
    auto current = load_snapshot();
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
                         context);
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
    // LogicalId bytes. Gate 7 will persist this list in WAL; Gate 4 never
    // re-evaluates the predicate while applying it.
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
                                   context);
      if (!receipt.ok()) {
        return receipt.status();
      }
      receipts.push_back(std::move(receipt).value());
    }
    return receipts;
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

  class PendingGuard {
   public:
    PendingGuard(SegmentedCollection *owner, std::uint64_t bytes) : owner_(owner), bytes_(bytes) {
      owner_->pending_count_.fetch_add(1, std::memory_order_acq_rel);
      owner_->pending_bytes_.fetch_add(bytes_, std::memory_order_acq_rel);
    }
    PendingGuard(const PendingGuard &) = delete;
    auto operator=(const PendingGuard &) -> PendingGuard & = delete;
    ~PendingGuard() {
      owner_->pending_count_.fetch_sub(1, std::memory_order_acq_rel);
      owner_->pending_bytes_.fetch_sub(bytes_, std::memory_order_acq_rel);
    }

   private:
    SegmentedCollection *owner_{};
    std::uint64_t bytes_{};
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
                                         first_unused));
    }
    recalculate_counts(*snapshot);
    snapshot->visibility_watermark = maximum_sequence;
    next_op_id_.store(maximum_sequence + 1, std::memory_order_release);
    accepted_count_.store(snapshot->searchable_live_count, std::memory_order_release);
    publish_snapshot(std::move(snapshot));
    return core::Status::success();
  }

  [[nodiscard]] auto admit() -> std::optional<OperationGuard> {
    std::lock_guard lock(lifecycle_mutex_);
    if (lifecycle_ != LifecycleState::open) {
      return std::nullopt;
    }
    ++inflight_operations_;
    return OperationGuard(this);
  }

  void release_admission() {
    std::lock_guard lock(lifecycle_mutex_);
    --inflight_operations_;
    if (inflight_operations_ == 0) {
      lifecycle_changed_.notify_all();
    }
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
          if (is_nan_score(hit.score)) {
            return core::Status::error(core::StatusCode::internal,
                                       core::OperationStage::search,
                                       core::StatusDetail::invalid_score,
                                       "segment returned a NaN score");
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
                                   core::MutationContext &context)
      -> core::Result<MutationReceipt> {
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
    AcceptedGuard accepted(this, row_status);
    const auto row_value = target->next_row_id.fetch_add(1, std::memory_order_acq_rel);
    if (row_value == std::numeric_limits<std::uint64_t>::max()) {
      return core::Status::error(core::StatusCode::resource_exhausted,
                                 core::OperationStage::admission,
                                 core::StatusDetail::arithmetic_overflow,
                                 "active segment exhausted its row ID space");
    }
    const RowAddress address{target->segment_id, target->generation, core::SegmentRowId(row_value)};
    const auto previous = current->versions.find(logical_id);

    SegmentMutationPayload mutation;
    mutation.action = action;
    mutation.op_id = op_id;
    mutation.upsert_sequence = op_id;
    mutation.target = address;
    if (previous != current->versions.end()) {
      mutation.previous = previous->second.address;
    }
    if (action == SegmentMutationAction::write && payload.vector.has_value()) {
      mutation.vector = payload.vector->view();
    }
    core::OpaqueOperationRequest opaque;
    opaque.payload = &mutation;
    opaque.payload_size = sizeof(mutation);
    core::MutationContext engine_context = context;
    engine_context.transaction_token = &mutation;
    const auto pending_bytes =
        payload.vector.has_value() ? static_cast<std::uint64_t>(payload.vector->bytes().size()) : 0;
    PendingGuard pending(this, pending_bytes);
    core::MutationToken token;
    bool prepared{};

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
    prepared = true;
    status = target->segment.stage_mutation(token, engine_context);
    if (!status.ok()) {
      (void)target->segment.abort_mutation(token, engine_context);
      return status;
    }

    std::shared_ptr<RoutingSnapshot> next;
    try {
      next = std::make_shared<RoutingSnapshot>(*current);
      next->generation = current->generation + 1;
      next->visibility_watermark = op_id;
      next->metadata_epoch = current->metadata_epoch + 1;
      next->reverse.insert_or_assign(address, ReverseEntry{logical_id, op_id});
      next->versions.insert_or_assign(logical_id,
                                      VersionEntry{address,
                                                   op_id,
                                                   action == SegmentMutationAction::write
                                                       ? VersionState::live
                                                       : VersionState::tombstone,
                                                   std::move(payload)});
      recalculate_counts(*next);
    } catch (...) {
      if (prepared) {
        (void)target->segment.abort_mutation(token, engine_context);
      }
      return core::status_from_exception(core::OperationStage::mutation_stage);
    }

    status = target->segment.publish_mutation(token, engine_context);
    if (!status.ok()) {
      (void)target->segment.abort_mutation(token, engine_context);
      return status;
    }

    // This atomic pointer swap is the Gate 4 visibility linearization point.
    // There is deliberately no WAL or durability acknowledgement here; Gate 7
    // inserts COMMIT between dark stage and this publication.
    publish_snapshot(std::move(next));
    accepted.commit();
    return MutationReceipt{op_id, op_id, true, DurabilityState::memory_only, row_status};
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

  [[nodiscard]] static auto closed_status(core::OperationStage stage) -> core::Status {
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
  RoutingSnapshotPtr snapshot_{};
  mutable std::mutex lifecycle_mutex_{};
  std::condition_variable lifecycle_changed_{};
  LifecycleState lifecycle_{LifecycleState::open};
  std::uint64_t inflight_operations_{};
  std::mutex drain_mutex_{};
  std::mutex mutation_mutex_{};
  std::atomic_uint64_t next_op_id_{1};
  std::atomic_uint64_t accepted_count_{};
  std::atomic_uint64_t pending_count_{};
  std::atomic_uint64_t pending_bytes_{};
};

}  // namespace alaya::internal::collection
