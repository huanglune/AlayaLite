// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/collection/segmented_collection.hpp"

namespace alaya::internal::collection {

[[nodiscard]] auto SegmentedCollection::open(CollectionSchema schema,
                                             std::vector<SegmentRegistration> registrations,
                                             CollectionConfig config)
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

[[nodiscard]] auto SegmentedCollection::scalar_query(const LogicalFilter &filter,
                                                     std::size_t limit,
                                                     Projection projection)
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

[[nodiscard]] auto SegmentedCollection::write(const WriteRequest &request,
                                              core::MutationContext &context)
    -> core::Result<MutationReceipt> {
  if (const auto writable = ensure_writable(core::OperationStage::admission); !writable.ok()) {
    return writable;
  }
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

[[nodiscard]] auto SegmentedCollection::erase(const core::LogicalId &logical_id,
                                              core::MutationContext &context,
                                              WriteOptions options)
    -> core::Result<MutationReceipt> {
  if (const auto writable = ensure_writable(core::OperationStage::admission); !writable.ok()) {
    return writable;
  }
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

[[nodiscard]] auto SegmentedCollection::delete_by_filter(const LogicalFilter &filter,
                                                         core::MutationContext &context)
    -> core::Result<std::vector<MutationReceipt>> {
  if (const auto writable = ensure_writable(core::OperationStage::admission); !writable.ok()) {
    return writable;
  }
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

[[nodiscard]] auto SegmentedCollection::mutate_batch(const BatchMutationRequest &request,
                                                     core::MutationContext &context)
    -> core::Result<BatchMutationReceipt> {
  if (const auto writable = ensure_writable(core::OperationStage::admission); !writable.ok()) {
    return writable;
  }
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
  auto batch_resources = preflight_batch_resources(request, context);
  if (!batch_resources.ok()) {
    return batch_resources;
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
  batch.durability = config_.features.wal_coordinator ? durability_state(request.options.durability)
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

[[nodiscard]] auto SegmentedCollection::retire_segment(std::uint64_t segment_id,
                                                       std::uint64_t generation) -> core::Status {
  if (const auto writable = ensure_writable(core::OperationStage::admission); !writable.ok()) {
    return writable;
  }
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

[[nodiscard]] auto SegmentedCollection::stats() const -> CollectionStats {
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

[[nodiscard]] auto SegmentedCollection::drain(const core::Deadline &deadline) -> core::Status {
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

[[nodiscard]] auto SegmentedCollection::checkpoint(core::CheckpointContext &context)
    -> core::Result<CheckpointReceipt> {
  if (const auto writable = ensure_writable(core::OperationStage::checkpoint); !writable.ok()) {
    return writable;
  }
  std::lock_guard checkpoint_lock(checkpoint_mutex_);
  auto admission = admit();
  if (!admission.has_value()) {
    return closed_status(core::OperationStage::checkpoint);
  }
  if (auto guard = ensure_not_recovery_required(core::OperationStage::checkpoint); !guard.ok()) {
    return guard;
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

[[nodiscard]] auto SegmentedCollection::consolidate(std::size_t num_threads,
                                                    std::size_t r_target,
                                                    bool reclaim_slots,
                                                    bool bloom_consolidate)
    -> core::Result<SegmentMaintenanceReceipt> {
  if (const auto writable = ensure_writable(core::OperationStage::checkpoint); !writable.ok()) {
    return writable;
  }
  std::lock_guard checkpoint_lock(checkpoint_mutex_);
  auto admission = admit();
  if (!admission.has_value()) {
    return closed_status(core::OperationStage::checkpoint);
  }
  if (auto guard = ensure_not_recovery_required(core::OperationStage::checkpoint); !guard.ok()) {
    return guard;
  }
  if (num_threads == 0) {
    return core::Status::error(core::StatusCode::invalid_argument,
                               core::OperationStage::checkpoint,
                               core::StatusDetail::malformed_struct,
                               "Collection consolidate requires num_threads > 0");
  }

  std::lock_guard mutation_lock(mutation_mutex_);
  // A logical writer can latch recovery-required while this call waits for the
  // post-COMMIT mutation window. Recheck after taking the mutation barrier and
  // before resolving or invoking the active hook.
  if (auto guard = ensure_not_recovery_required(core::OperationStage::checkpoint); !guard.ok()) {
    return guard;
  }
  const auto snapshot = load_snapshot();
  const auto active = snapshot->find_active_mutable();
  if (active == nullptr || !active->maintenance.consolidate) {
    return core::Status::error(core::StatusCode::not_supported,
                               core::OperationStage::checkpoint,
                               core::StatusDetail::operation_slot_absent,
                               "active mutable segment has no consolidate maintenance hook");
  }

  core::Status status;
  try {
    status =
        active->maintenance.consolidate(num_threads, r_target, reclaim_slots, bloom_consolidate);
  } catch (...) {
    status = core::status_from_exception(core::OperationStage::checkpoint);
  }
  if (!status.ok()) {
    bool recovery_required = false;
    try {
      recovery_required =
          active->maintenance.recovery_required && active->maintenance.recovery_required();
    } catch (...) {
      recovery_required = true;
    }
    if (recovery_required) {
      latch_recovery_required(status.diagnostic());
    }
    return status;
  }
  return SegmentMaintenanceReceipt{active->segment_id, active->generation};
}

[[nodiscard]] auto SegmentedCollection::rotate_to_successor(SegmentRegistration successor,
                                                            core::CheckpointContext &context,
                                                            RotationDurableCallback durable_switch)
    -> core::Result<ActiveRotationReceipt> {
  if (const auto writable = ensure_writable(core::OperationStage::freeze); !writable.ok()) {
    return writable;
  }
  std::lock_guard checkpoint_lock(checkpoint_mutex_);
  auto admission = admit();
  if (!admission.has_value()) {
    return closed_status(core::OperationStage::freeze);
  }
  if (auto guard = ensure_not_recovery_required(core::OperationStage::freeze); !guard.ok()) {
    return guard;
  }
  if (!config_.features.wal_coordinator || wal_ == nullptr) {
    return core::Status::error(core::StatusCode::not_supported,
                               core::OperationStage::freeze,
                               core::StatusDetail::operation_slot_absent,
                               "successor rotation requires the Collection WAL coordinator");
  }
  auto control = core::validate_runtime_control(context.deadline,
                                                context.cancellation,
                                                core::OperationStage::freeze);
  if (!control.ok()) {
    return control;
  }
  if (successor.role != SegmentRole::active_mutable || !successor.rows.empty()) {
    return core::Status::error(core::StatusCode::invalid_argument,
                               core::OperationStage::freeze,
                               core::StatusDetail::malformed_struct,
                               "successor registration must be an empty active mutable segment");
  }
  const auto successor_descriptor = successor.segment.descriptor();
  if (!successor.segment.capabilities().supports(core::OperationCapability::mutation) ||
      !successor.segment.capabilities().supports(core::OperationCapability::checkpoint) ||
      successor_descriptor.dim != schema_.dim || successor_descriptor.metric != schema_.metric ||
      successor_descriptor.stored_scalar_type != schema_.scalar_type || successor.segment_id == 0 ||
      successor.generation == 0) {
    return core::Status::error(core::StatusCode::invalid_argument,
                               core::OperationStage::freeze,
                               core::StatusDetail::readonly_instance,
                               "successor does not satisfy the active mutable schema/capabilities");
  }

  ControlPlaneGate gate(this);
  control = gate.drain(context.deadline, context.cancellation);
  if (!control.ok()) {
    return control;
  }
  std::lock_guard mutation_lock(mutation_mutex_);
  const auto current = load_snapshot();
  const auto source = current->find_active_mutable();
  if (source == nullptr) {
    return core::Status::error(core::StatusCode::conflict,
                               core::OperationStage::freeze,
                               core::StatusDetail::readonly_instance,
                               "successor rotation found no active mutable source");
  }
  if (current->find_segment(successor.segment_id, successor.generation) != nullptr) {
    return core::Status::error(core::StatusCode::conflict,
                               core::OperationStage::freeze,
                               core::StatusDetail::already_exists,
                               "successor segment identity is already routed");
  }

  core::CheckpointToken source_token;
  auto status = source->segment.checkpoint(context, source_token);
  if (!status.ok()) {
    return status;
  }
  core::CheckpointToken successor_token;
  status = successor.segment.checkpoint(context, successor_token);
  if (!status.ok()) {
    return status;
  }

  auto next = std::make_shared<RoutingSnapshot>(*current);
  for (auto &entry : next->segments) {
    if (entry->segment_id == source->segment_id && entry->generation == source->generation) {
      entry = std::make_shared<SegmentEntry>(source->segment_id,
                                             source->generation,
                                             SegmentRole::sealed,
                                             source->segment,
                                             source->exact_rerank,
                                             source->next_row_id.load(std::memory_order_acquire),
                                             source->atomic_mutation_bundle,
                                             SegmentMaintenanceHook{});
      break;
    }
  }
  next->segments.push_back(std::make_shared<SegmentEntry>(successor.segment_id,
                                                          successor.generation,
                                                          successor.role,
                                                          std::move(successor.segment),
                                                          std::move(successor.exact_rerank),
                                                          successor.next_row_id,
                                                          successor.atomic_mutation_bundle,
                                                          std::move(successor.maintenance)));
  next->generation = current->generation + 1;

  auto stored = CollectionCheckpointStore::write(wal_->directory(),
                                                 *next,
                                                 retry_receipts_,
                                                 batch_retry_receipts_);
  if (!stored.ok()) {
    return stored.status();
  }
  auto receipt = std::move(stored).value();
  status = wal_->reset_to_checkpoint(receipt.wal_cut);
  if (!status.ok()) {
    return status;
  }
  next->durable_watermark = receipt.durable_watermark;
  ActiveRotationReceipt rotation{source->segment_id,
                                 source->generation,
                                 successor.segment_id,
                                 successor.generation,
                                 receipt};
  if (durable_switch) {
    status = durable_switch(rotation);
    if (!status.ok()) {
      return status;
    }
  }
  publish_snapshot(std::move(next));
  return rotation;
}

[[nodiscard]] auto SegmentedCollection::resume_segment_replacement(
    std::span<const RowAddress> sources,
    std::uint64_t target_segment_id,
    std::uint64_t target_generation,
    std::span<const SegmentReplacement> replacements) -> core::Status {
  if (const auto writable = ensure_writable(core::OperationStage::open); !writable.ok()) {
    return writable;
  }
  auto admission = admit();
  if (!admission.has_value()) {
    return closed_status(core::OperationStage::open);
  }
  std::lock_guard mutation_lock(mutation_mutex_);
  SegmentRegistration target;
  target.segment_id = target_segment_id;
  target.generation = target_generation;
  return install_segment_replacement_locked(sources, std::move(target), replacements, true);
}

[[nodiscard]] auto SegmentedCollection::install_segment_replacement_locked(
    std::span<const RowAddress> sources,
    SegmentRegistration target,
    std::span<const SegmentReplacement> replacements,
    bool target_is_already_routed) -> core::Status {
  if (sources.empty() || target.segment_id == 0 || target.generation == 0) {
    return core::Status::error(core::StatusCode::invalid_argument,
                               core::OperationStage::save,
                               core::StatusDetail::malformed_struct,
                               "segment replacement identities are incomplete");
  }
  const auto is_source = [&](const RowAddress &address) {
    return std::ranges::any_of(sources, [&](const RowAddress &source) {
      return source.segment_id == address.segment_id && source.generation == address.generation;
    });
  };
  const auto current = load_snapshot();
  for (const auto &source : sources) {
    if (current->find_segment(source.segment_id, source.generation) == nullptr) {
      return core::Status::error(core::StatusCode::not_found,
                                 core::OperationStage::save,
                                 core::StatusDetail::none,
                                 "segment replacement source is no longer routed");
    }
  }

  std::shared_ptr<SegmentEntry> target_entry;
  if (target_is_already_routed) {
    target_entry = current->find_segment(target.segment_id, target.generation);
    if (target_entry == nullptr) {
      return core::Status::error(core::StatusCode::corruption,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 "published replacement target is not registered");
    }
  } else {
    if (current->find_segment(target.segment_id, target.generation) != nullptr) {
      return core::Status::error(core::StatusCode::conflict,
                                 core::OperationStage::save,
                                 core::StatusDetail::already_exists,
                                 "segment replacement target is already routed");
    }
    const auto descriptor = target.segment.descriptor();
    if (!target.segment.capabilities().supports(core::OperationCapability::search) ||
        descriptor.dim != schema_.dim || descriptor.metric != schema_.metric ||
        descriptor.stored_scalar_type != schema_.scalar_type) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::save,
                                 core::StatusDetail::malformed_struct,
                                 "segment replacement target disagrees with the Collection schema");
    }
    target_entry = std::make_shared<SegmentEntry>(target.segment_id,
                                                  target.generation,
                                                  SegmentRole::sealed,
                                                  std::move(target.segment),
                                                  std::move(target.exact_rerank),
                                                  target.next_row_id,
                                                  false,
                                                  SegmentMaintenanceHook{});
  }

  std::set<RowAddress> mapped_sources;
  std::set<RowAddress> mapped_targets;
  for (const auto &replacement : replacements) {
    if (!is_source(replacement.source) || replacement.target.segment_id != target.segment_id ||
        replacement.target.generation != target.generation ||
        !mapped_sources.insert(replacement.source).second ||
        !mapped_targets.insert(replacement.target).second) {
      return core::Status::
          error(core::StatusCode::invalid_argument,
                core::OperationStage::save,
                core::StatusDetail::malformed_struct,
                "segment replacement map contains an invalid or duplicate address");
    }
  }
  for (const auto &[unused, version] : current->versions) {
    (void)unused;
    if (is_source(version.address) && !mapped_sources.contains(version.address)) {
      return core::Status::
          error(core::StatusCode::conflict,
                core::OperationStage::save,
                core::StatusDetail::none,
                "segment replacement map does not cover every current source version");
    }
  }

  auto next = std::make_shared<RoutingSnapshot>(*current);
  if (!target_is_already_routed) {
    next->segments.push_back(target_entry);
  }
  for (const auto &replacement : replacements) {
    next->reverse.insert_or_assign(replacement.target,
                                   ReverseEntry{replacement.logical_id,
                                                replacement.upsert_sequence});
    const auto found = next->versions.find(replacement.logical_id);
    if (found != next->versions.end() && found->second.address == replacement.source &&
        found->second.upsert_sequence == replacement.upsert_sequence) {
      found->second.address = replacement.target;
    }
    const auto row = static_cast<std::uint64_t>(replacement.target.row_id);
    if (row != std::numeric_limits<std::uint64_t>::max()) {
      auto first_unused = target_entry->next_row_id.load(std::memory_order_acquire);
      while (first_unused <= row &&
             !target_entry->next_row_id.compare_exchange_weak(first_unused,
                                                              row + 1,
                                                              std::memory_order_acq_rel)) {
      }
    }
  }
  std::erase_if(next->reverse, [&](const auto &item) {
    return is_source(item.first);
  });
  std::erase_if(next->segments, [&](const auto &entry) {
    return std::ranges::any_of(sources, [&](const RowAddress &source) {
      return source.segment_id == entry->segment_id && source.generation == entry->generation;
    });
  });
  next->generation = current->generation + 1;
  recalculate_counts(*next);
  publish_snapshot(std::move(next));
  return core::Status::success();
}

[[nodiscard]] auto SegmentedCollection::initialize(std::vector<SegmentRegistration> registrations)
    -> core::Status {
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
                                       registration.atomic_mutation_bundle,
                                       std::move(registration.maintenance)));
  }
  recalculate_counts(*snapshot);
  snapshot->visibility_watermark =
      std::max(maximum_sequence, config_.recovery.minimum_visibility_watermark);
  snapshot->durable_watermark = 0;
  if (config_.features.wal_coordinator) {
    auto opened =
        CollectionLogicalWal::open(config_.wal.root, config_.wal.namespace_name, config_.read_only);
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

[[nodiscard]] auto SegmentedCollection::preflight_search_budget(
    const RoutingSnapshot &snapshot,
    const CollectionSearchRequest &request) -> core::Result<SearchBudgetPlan> {
  SearchBudgetPlan plan;
  const auto candidate_rows =
      std::max<core::RowCount>(snapshot.searchable_live_count, request.options.top_k);
  std::uint64_t candidates{};
  std::uint64_t candidate_bytes{};
  std::uint64_t query_bytes{};
  if (!core::checked_multiply(request.queries.rows, candidate_rows, candidates) ||
      !core::checked_multiply(candidates, sizeof(CollectionHit), candidate_bytes) ||
      !core::checked_multiply(request.queries.rows, request.queries.row_stride, query_bytes) ||
      !core::checked_add(candidate_bytes, query_bytes, plan.scratch_bytes)) {
    return search_budget_denied("collection search scratch accounting overflowed");
  }

  const auto rounds = static_cast<std::uint64_t>(request.maximum_overfetch_rounds) + 1;
  for (const auto &entry : snapshot.segments) {
    if (entry->segment.descriptor().medium != core::Medium::disk) {
      continue;
    }
    const auto known_rows = snapshot.known_rows_for(*entry);
    std::uint64_t requests{};
    std::uint64_t bytes{};
    std::uint64_t row_bytes{};
    if (!core::checked_multiply(request.queries.rows, rounds, requests) ||
        !core::checked_multiply(entry->segment.descriptor().dim, sizeof(float), row_bytes) ||
        !core::checked_multiply(known_rows, row_bytes, bytes) ||
        !core::checked_multiply(bytes, request.queries.rows, bytes) ||
        !core::checked_multiply(bytes, rounds, bytes) ||
        !core::checked_add(plan.io_requests, requests, plan.io_requests) ||
        !core::checked_add(plan.io_bytes, bytes, plan.io_bytes)) {
      return search_budget_denied("collection search I/O accounting overflowed");
    }
  }

  if (!request.context->query_scratch_lease.permits(plan.scratch_bytes)) {
    return search_budget_denied("collection search scratch lease is too small");
  }
  const auto &credits = request.context->io_credits;
  if ((credits.available_requests != core::kUnlimitedResource &&
       credits.available_requests < plan.io_requests) ||
      (credits.available_bytes != core::kUnlimitedResource &&
       credits.available_bytes < plan.io_bytes)) {
    return search_budget_denied("collection search I/O credits are too small");
  }
  return plan;
}

[[nodiscard]] auto SegmentedCollection::estimate_filter_selectivity(const RoutingSnapshot &snapshot,
                                                                    const LogicalFilter &filter,
                                                                    CollectionSearchStats *stats)
    -> double {
  if (const auto provided = filter.selectivity_estimate(); provided.has_value()) {
    return *provided;
  }
  constexpr std::uint64_t kSampleRows = 256;
  std::uint64_t examined{};
  std::uint64_t passed{};
  for (const auto &[logical_id, version] : snapshot.versions) {
    if (examined == kSampleRows) {
      break;
    }
    if (version.state != VersionState::live ||
        version.upsert_sequence > snapshot.visibility_watermark) {
      continue;
    }
    ++examined;
    const auto matches =
        filter.matches(logical_id, version.payload.metadata, version.payload.document);
    passed += matches ? 1U : 0U;
  }
  if (stats != nullptr) {
    stats->filter_examined += examined;
    stats->filter_passed += passed;
  }
  return examined == 0 ? 0.0 : static_cast<double>(passed) / static_cast<double>(examined);
}

[[nodiscard]] auto SegmentedCollection::select_filter_execution(
    const RoutingSnapshot &snapshot,
    const CollectionSearchRequest &request,
    bool prefer_exact) -> core::Result<core::FilterExecution> {
  if (!request.filter.active()) {
    return core::FilterExecution::postfilter;
  }
  // The active mutable LASER adapter intentionally does not thread a segment
  // filter into MutableLaserSegment::search. Keep the aggregate capability
  // report honest: an active LASER route therefore executes the logical
  // predicate only after candidate retrieval. Sealed LASER remains eligible
  // for its existing bitmap traversal path.
  if (std::ranges::any_of(snapshot.segments, [](const auto &entry) {
        return entry->role == SegmentRole::active_mutable &&
               entry->segment.descriptor().algorithm_id == core::algorithm::laser;
      })) {
    return core::FilterExecution::postfilter;
  }
  if (prefer_exact || request.options.filter_policy == core::FilterPolicy::strict) {
    return core::FilterExecution::prefilter;
  }
  const auto selectivity = estimate_filter_selectivity(snapshot, request.filter, request.stats);
  if (selectivity < 0.0 || selectivity > 1.0) {
    return core::Status::error(core::StatusCode::invalid_argument,
                               core::OperationStage::validation,
                               core::StatusDetail::malformed_struct,
                               "filter selectivity estimate must be in [0, 1]");
  }
  if (selectivity <= 0.15) {
    return core::FilterExecution::prefilter;
  }
  if (selectivity <= 0.60) {
    return core::FilterExecution::traversal;
  }
  return core::FilterExecution::postfilter;
}
}  // namespace alaya::internal::collection
