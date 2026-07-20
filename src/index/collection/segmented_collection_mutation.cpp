// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/collection/segmented_collection.hpp"

namespace alaya::internal::collection {

[[nodiscard]] auto SegmentedCollection::mutate_locked(RoutingSnapshotPtr current,
                                                      const core::LogicalId &logical_id,
                                                      SegmentMutationAction action,
                                                      RecordPayload payload,
                                                      RowMutationStatus row_status,
                                                      core::MutationContext &context,
                                                      const WriteOptions &options,
                                                      std::uint64_t batch_op_id)
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

[[nodiscard]] auto SegmentedCollection::preflight_batch_resources(
    const BatchMutationRequest &request,
    core::MutationContext &context) const -> core::Status {
  if (!core::is_current_struct(context)) {
    return core::Status::error(core::StatusCode::invalid_argument,
                               core::OperationStage::admission,
                               core::StatusDetail::malformed_struct,
                               "batch mutation context is incompatible");
  }
  auto control = core::validate_runtime_control(context.deadline,
                                                context.cancellation,
                                                core::OperationStage::admission);
  if (!control.ok()) {
    return control;
  }
  std::uint64_t bytes{};
  if (!core::checked_multiply(request.rows.size(), sizeof(SegmentMutationPayload), bytes)) {
    return core::Status::error(core::StatusCode::resource_exhausted,
                               core::OperationStage::admission,
                               core::StatusDetail::arithmetic_overflow,
                               "batch mutation reservation overflows uint64");
  }
  for (const auto &row : request.rows) {
    std::uint64_t row_bytes = row.logical_id.canonical_bytes().size() + row.document.size();
    if (row.action == RowMutationAction::write && row.vector.rows == 1 &&
        row.vector.dim == schema_.dim && row.vector.scalar_type == schema_.scalar_type) {
      std::uint64_t vector_bytes{};
      if (!core::checked_multiply(row.vector.dim,
                                  core::scalar_type_size(row.vector.scalar_type),
                                  vector_bytes) ||
          !core::checked_add(row_bytes, vector_bytes, row_bytes)) {
        return core::Status::error(core::StatusCode::resource_exhausted,
                                   core::OperationStage::admission,
                                   core::StatusDetail::arithmetic_overflow,
                                   "batch mutation vector bytes overflow uint64");
      }
    }
    for (const auto &[key, value] : row.metadata) {
      std::uint64_t metadata_bytes = key.size() + sizeof(value);
      if (const auto *text = std::get_if<std::string>(&value); text != nullptr) {
        metadata_bytes += text->size();
      }
      if (!core::checked_add(row_bytes, metadata_bytes, row_bytes)) {
        return core::Status::error(core::StatusCode::resource_exhausted,
                                   core::OperationStage::admission,
                                   core::StatusDetail::arithmetic_overflow,
                                   "batch mutation metadata bytes overflow uint64");
      }
    }
    if (!core::checked_add(bytes, row_bytes, bytes)) {
      return core::Status::error(core::StatusCode::resource_exhausted,
                                 core::OperationStage::admission,
                                 core::StatusDetail::arithmetic_overflow,
                                 "batch mutation bytes overflow uint64");
    }
  }
  auto resource =
      context.pending_reservation.ensure(bytes,
                                         core::OperationStage::admission,
                                         "batch pending-mutation reservation is too small");
  if (!resource.ok()) {
    return resource;
  }
  resource = context.stage_reservation.ensure(bytes,
                                              core::OperationStage::admission,
                                              "batch mutation-stage reservation is too small");
  if (!resource.ok()) {
    return resource;
  }
  if (wal_ != nullptr) {
    std::uint64_t wal_records{};
    std::uint64_t framing{};
    std::uint64_t wal_bytes{};
    if (!core::checked_multiply(request.rows.size(), 3, wal_records) ||
        !core::checked_add(wal_records, 1, wal_records) ||
        !core::checked_multiply(wal_records,
                                logical_wal_detail::kHeaderBytes +
                                    logical_wal_detail::kTrailerBytes,
                                framing) ||
        !core::checked_add(bytes, framing, wal_bytes)) {
      return core::Status::error(core::StatusCode::resource_exhausted,
                                 core::OperationStage::admission,
                                 core::StatusDetail::arithmetic_overflow,
                                 "batch mutation WAL accounting overflows uint64");
    }
    if ((context.wal_io_credits.available_requests != core::kUnlimitedResource &&
         context.wal_io_credits.available_requests < wal_records) ||
        (context.wal_io_credits.available_bytes != core::kUnlimitedResource &&
         context.wal_io_credits.available_bytes < wal_bytes)) {
      return core::Status::error(core::StatusCode::resource_exhausted,
                                 core::OperationStage::admission,
                                 core::StatusDetail::budget_denied,
                                 "batch mutation WAL I/O credits are too small",
                                 core::Retryability::retryable_with_backoff);
    }
  }
  return core::Status::success();
}

[[nodiscard]] auto SegmentedCollection::validate_batch_row(const RoutingSnapshotPtr &current,
                                                           const BatchRowMutation &row) const
    -> ValidatedBatchRow {
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

[[nodiscard]] auto SegmentedCollection::make_non_searchable_receipt(
    const RoutingSnapshotPtr &current,
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

[[nodiscard]] auto SegmentedCollection::mutate_batch_row_locked(RoutingSnapshotPtr current,
                                                                const BatchRowMutation &row,
                                                                core::MutationContext &context,
                                                                const WriteOptions &options,
                                                                std::uint64_t batch_op_id)
    -> core::Result<MutationReceipt> {
  auto validated = validate_batch_row(current, row);
  if (!validated.valid) {
    return make_non_searchable_receipt(current, batch_op_id, validated.status, options.retry_token);
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

[[nodiscard]] auto SegmentedCollection::mutate_atomic_batch_locked(
    RoutingSnapshotPtr current,
    const BatchMutationRequest &request,
    core::MutationContext &context,
    std::uint64_t batch_op_id) -> core::Result<BatchMutationReceipt> {
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
      receipt.rows.push_back(
          make_non_searchable_receipt(current,
                                      batch_op_id,
                                      index == *failed ? failed_status : RowMutationStatus::aborted,
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

[[nodiscard]] auto SegmentedCollection::persist_batch_receipt(const BatchMutationReceipt &receipt,
                                                              WriteDurability durability)
    -> core::Status {
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

[[nodiscard]] auto SegmentedCollection::make_engine_payloads(
    const WalMutationTransaction &transaction) -> std::vector<SegmentMutationPayload> {
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

[[nodiscard]] auto SegmentedCollection::build_dark_snapshot(
    const RoutingSnapshotPtr &current,
    const WalMutationTransaction &transaction,
    bool durable) -> core::Result<std::shared_ptr<RoutingSnapshot>> {
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

[[nodiscard]] auto SegmentedCollection::execute_transaction_locked(
    RoutingSnapshotPtr current,
    const std::shared_ptr<SegmentEntry> &target,
    const WalMutationTransaction &transaction,
    core::MutationContext &context,
    std::uint64_t transaction_id) -> core::Result<std::vector<MutationReceipt>> {
  if (auto guard = ensure_not_recovery_required(core::OperationStage::mutation_prepare);
      !guard.ok()) {
    return guard;
  }
  if (failpoint(MutationFailPoint::before_prepare)) {
    return injected_failure(MutationFailPoint::before_prepare);
  }
  if (!core::is_current_struct(context)) {
    return core::Status::error(core::StatusCode::invalid_argument,
                               core::OperationStage::mutation_prepare,
                               core::StatusDetail::malformed_struct,
                               "collection mutation context is incompatible");
  }
  std::uint64_t pending_bytes{};
  for (const auto &row : transaction.rows) {
    if (row.payload.vector.has_value()) {
      if (!core::checked_add(pending_bytes, row.payload.vector->bytes().size(), pending_bytes)) {
        return core::Status::error(core::StatusCode::resource_exhausted,
                                   core::OperationStage::mutation_prepare,
                                   core::StatusDetail::arithmetic_overflow,
                                   "collection mutation pending bytes overflow uint64");
      }
    }
  }
  std::uint64_t row_overhead{};
  std::uint64_t reservation_bytes{};
  if (!core::checked_multiply(transaction.rows.size(),
                              sizeof(SegmentMutationPayload),
                              row_overhead) ||
      !core::checked_add(pending_bytes, row_overhead, reservation_bytes)) {
    return core::Status::error(core::StatusCode::resource_exhausted,
                               core::OperationStage::mutation_prepare,
                               core::StatusDetail::arithmetic_overflow,
                               "collection mutation stage bytes overflow uint64");
  }
  auto resource =
      context.pending_reservation.ensure(reservation_bytes,
                                         core::OperationStage::mutation_prepare,
                                         "collection pending-mutation reservation is too small");
  if (!resource.ok()) {
    return resource;
  }
  resource = context.stage_reservation.ensure(reservation_bytes,
                                              core::OperationStage::mutation_prepare,
                                              "collection mutation-stage reservation is too small");
  if (!resource.ok()) {
    return resource;
  }
  std::vector<std::byte> wal_payload;
  try {
    wal_payload = encode_wal_transaction(transaction);
  } catch (...) {
    return core::status_from_exception(core::OperationStage::mutation_prepare);
  }
  if (wal_ != nullptr) {
    constexpr std::uint64_t kWalRecords = 3;
    constexpr std::uint64_t kWalFramingBytes =
        kWalRecords * (logical_wal_detail::kHeaderBytes + logical_wal_detail::kTrailerBytes);
    std::uint64_t wal_bytes{};
    if (!core::checked_add(wal_payload.size(), kWalFramingBytes, wal_bytes)) {
      return core::Status::error(core::StatusCode::resource_exhausted,
                                 core::OperationStage::mutation_prepare,
                                 core::StatusDetail::arithmetic_overflow,
                                 "collection mutation WAL bytes overflow uint64");
    }
    if ((context.wal_io_credits.available_requests != core::kUnlimitedResource &&
         context.wal_io_credits.available_requests < kWalRecords) ||
        (context.wal_io_credits.available_bytes != core::kUnlimitedResource &&
         context.wal_io_credits.available_bytes < wal_bytes)) {
      return core::Status::error(core::StatusCode::resource_exhausted,
                                 core::OperationStage::mutation_prepare,
                                 core::StatusDetail::budget_denied,
                                 "collection mutation WAL I/O credits are too small",
                                 core::Retryability::retryable_with_backoff);
    }
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
  // B-01 set-point #1 (live write): the physical txid is the caller-supplied
  // transaction_id (op_id for single/per-row, batch_op_id for an atomic batch),
  // never the shared batch_op_id; max_row_op_id is the idempotency basis.
  engine_context.transaction_id = transaction_id;
  engine_context.max_row_op_id = transaction_max_row_op(transaction);
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
  // B-03: from here (L:COMMIT is durable) until publish_snapshot completes, any
  // non-termination exit must latch a Collection-level recovery-required state, or
  // a committed-but-unpublished transaction could be silently dropped by a later
  // checkpoint that cuts the logical WAL. The RAII guard latches on every early
  // return in this window; disarm() below marks the clean success path.
  RecoveryGuard recovery_guard(this);
  if (failpoint(MutationFailPoint::after_commit)) {
    return injected_failure(MutationFailPoint::after_commit);
  }
  status = target->segment.publish_mutation(token, engine_context);
  if (!status.ok()) {
    // COMMIT is already authoritative. Recovery must retry publish; aborting
    // here would incorrectly discard a durable transaction.
    return status;
  }
  if (failpoint(MutationFailPoint::after_engine_publish_before_snapshot)) {
    return injected_failure(MutationFailPoint::after_engine_publish_before_snapshot);
  }
  publish_snapshot(std::move(dark).value());
  recovery_guard.disarm();
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

[[nodiscard]] auto SegmentedCollection::replay_engine_transaction(
    const WalMutationTransaction &transaction) -> core::Status {
  if (transaction.rows.empty()) {
    return core::Status::success();
  }
  const auto target =
      load_or_initializing_snapshot_->find_segment(transaction.rows.front().target.segment_id,
                                                   transaction.rows.front().target.generation);
  if (target == nullptr) {
    return core::Status::error(core::StatusCode::corruption,
                               core::OperationStage::mutation_replay,
                               core::StatusDetail::readonly_instance,
                               "committed WAL targets an unavailable segment");
  }
  for (const auto &row : transaction.rows) {
    if (row.target.segment_id != target->segment_id ||
        row.target.generation != target->generation) {
      return core::Status::error(core::StatusCode::corruption,
                                 core::OperationStage::mutation_replay,
                                 core::StatusDetail::malformed_struct,
                                 "one WAL transaction targets multiple segment instances");
    }
    const auto row_id = static_cast<std::uint64_t>(row.target.row_id);
    if (row_id != std::numeric_limits<std::uint64_t>::max()) {
      auto first_unused = target->next_row_id.load(std::memory_order_acquire);
      while (first_unused <= row_id &&
             !target->next_row_id.compare_exchange_weak(first_unused,
                                                        row_id + 1,
                                                        std::memory_order_acq_rel)) {
      }
    }
  }
  if (!target->segment.capabilities().supports(core::OperationCapability::mutation)) {
    core::SegmentStats stats;
    const auto maximum_row = std::ranges::max(transaction.rows, {}, &WalMutationRow::op_id).op_id;
    if (target->segment.stats(stats).ok() && stats.snapshot_version >= maximum_row) {
      // A roll-forward reader may have consumed the physical WAL tail into
      // its private working generation before erasing mutation slots. The
      // Collection still rebuilds its logical snapshot from this WAL, but
      // does not need a writer slot to repeat already-applied physical work.
      return core::Status::success();
    }
    return core::Status::error(core::StatusCode::corruption,
                               core::OperationStage::mutation_replay,
                               core::StatusDetail::readonly_instance,
                               "committed WAL targets a reader below its applied watermark");
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
  // B-01 set-point #2 (replay): derive the physical txid from the transaction shape
  // (atomic batch -> batch_op_id, else the single row's op_id), matching the live
  // path so a replayed transaction hits the same idempotency decision.
  context.transaction_id = physical_txid(transaction);
  context.max_row_op_id = transaction_max_row_op(transaction);
  return target->segment.replay_mutation(opaque, context);
}

void SegmentedCollection::install_recovered_receipts(const WalMutationTransaction &transaction,
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

[[nodiscard]] auto SegmentedCollection::apply_checkpoint_image(
    std::shared_ptr<RoutingSnapshot> &snapshot,
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
  // B-01/B-02 set-point #3: an active engine that rebuilds an EMPTY physical
  // segment from the checkpoint image must replay the synthesized single-row
  // transactions in op_id-ascending order. Out of order, a fresh physical
  // watermark would commit a high txid first and then wrongly skip a lower one.
  std::vector<std::size_t> active_rows;
  for (std::size_t i = 0; i < image.state.rows.size(); ++i) {
    const auto &row = image.state.rows[i];
    const auto target = snapshot->find_segment(row.target.segment_id, row.target.generation);
    if (target != nullptr && target->role == SegmentRole::active_mutable) {
      active_rows.push_back(i);
    }
  }
  std::sort(active_rows.begin(), active_rows.end(), [&](std::size_t lhs, std::size_t rhs) {
    return image.state.rows[lhs].op_id < image.state.rows[rhs].op_id;
  });
  for (const auto index : active_rows) {
    WalMutationTransaction single;
    single.batch_op_id = image.state.rows[index].op_id;
    single.batch_mode = BatchMutationMode::per_row_independent;
    single.durability = WriteDurability::wal_fsync;
    single.rows.push_back(image.state.rows[index]);
    const auto status = replay_engine_transaction(single);
    if (!status.ok()) {
      return status;
    }
  }
  return core::Status::success();
}

[[nodiscard]] auto SegmentedCollection::recover_durable_state(
    std::shared_ptr<RoutingSnapshot> &snapshot) -> core::Status {
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
    if (!entry.publish_marker && !config_.read_only) {
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
}  // namespace alaya::internal::collection
