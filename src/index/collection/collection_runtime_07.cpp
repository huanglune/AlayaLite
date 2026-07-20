// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// collection_runtime_07: successor rotation and flat-export verification.
// One compile-cost-balanced Collection runtime unit; see CMakeLists.txt.

#include "index/collection/collection.hpp"

namespace alaya {

[[nodiscard]] auto Collection::rotate_to_successor_locked(const CollectionRotationHandle &handle,
                                                          core::SealContext &context)
    -> core::Result<CollectionSealReceipt> {
  if (auto gate = implementation_->recovery_gate(core::OperationStage::save); !gate.ok()) {
    return gate;
  }
  if (!pending_rotation_.has_value() ||
      control_state_.phase != internal::collection::CollectionControlPhase::manifest_published ||
      handle.successor_segment_id != control_state_.target_segment_id ||
      handle.successor_generation != control_state_.target_generation ||
      handle.predecessor_segment_ids.size() != control_state_.sources.size() ||
      !std::equal(handle.predecessor_segment_ids.begin(),
                  handle.predecessor_segment_ids.end(),
                  control_state_.sources.begin(),
                  [](std::uint64_t segment_id, const auto &source) {
                    return segment_id == source.segment_id;
                  })) {
    return error(core::StatusCode::not_found,
                 core::OperationStage::save,
                 core::StatusDetail::none,
                 "rotate_to_successor handle does not match a prepared successor");
  }

  auto &prepared = *pending_rotation_;
  internal::collection::SegmentRegistration target;
  target.segment_id = control_state_.target_segment_id;
  target.generation = control_state_.target_generation;
  target.role = internal::collection::SegmentRole::sealed;
  target.segment = std::move(prepared.built_target.segment);
  target.rows = prepared.build_data.rows;
  auto status = implementation_->install_segment_replacement(control_state_.sources,
                                                             std::move(target),
                                                             prepared.build_data.replacements);
  if (!status.ok()) {
    return status;
  }
  core::CheckpointContext checkpoint_context;
  checkpoint_context.deadline = context.deadline;
  checkpoint_context.cancellation = context.cancellation;
  checkpoint_context.lane = context.lane;
  checkpoint_context.dirty_page_io_credits = context.io_credits;
  checkpoint_context.wal_io_credits = context.io_credits;
  checkpoint_context.durability_target = core::DurabilityTarget::full_checkpoint;
  auto checkpoint = checkpoint_locked(checkpoint_context);
  if (!checkpoint.ok()) {
    return checkpoint.status();
  }

  CollectionSealReceipt receipt;
  receipt.source_segment_id = control_state_.sources.front().segment_id;
  receipt.successor_segment_id = control_state_.active_segment_id;
  receipt.sealed_segment_id = control_state_.target_segment_id;
  receipt.wal_cut = control_state_.wal_cut;
  receipt.sealed_rows = prepared.build_data.live_rows;
  receipt.sealed_bytes = prepared.built_target.artifact_bytes;
  receipt.manifest_generation = control_state_.manifest_generation;
  receipt.built_algorithm = prepared.built_target.built_algorithm;
  receipt.effective_ef_construction = prepared.built_target.effective_ef_construction;
  receipt.flat_fallback = prepared.built_target.flat_fallback;
  receipt.fallback_reason = prepared.built_target.fallback_reason;
  const auto mapping_file = control_state_.mapping_file;
  control_state_.operation = internal::collection::CollectionControlOperation::idle;
  control_state_.phase = internal::collection::CollectionControlPhase::idle;
  control_state_.last_sealed_segment_id = control_state_.target_segment_id;
  control_state_.last_sealed_generation = control_state_.target_generation;
  control_state_.sources.clear();
  control_state_.successor_segment_id = 0;
  control_state_.successor_generation = 0;
  control_state_.target_segment_id = 0;
  control_state_.target_generation = 0;
  control_state_.wal_cut = 0;
  control_state_.mapping_file.clear();
  status = internal::collection::CollectionControlStore::save(options_.root, control_state_);
  if (!status.ok()) {
    return status;
  }
  internal::collection::CollectionControlStore::remove_replacements(options_.root, mapping_file);
  pending_rotation_.reset();
  return receipt;
}

[[nodiscard]] auto Collection::verify_flat_exports(
    const internal::collection::RoutingSnapshot &snapshot,
    std::span<const internal::collection::RowAddress> sources) const -> core::Status {
  for (const auto &source : sources) {
    const auto entry = snapshot.find_segment(source.segment_id, source.generation);
    if (entry == nullptr ||
        !entry->segment.capabilities().supports(core::OperationCapability::export_rows)) {
      return error(core::StatusCode::not_supported,
                   core::OperationStage::export_rows,
                   core::StatusDetail::operation_slot_absent,
                   "Flat compact source does not expose export_rows");
    }
    std::shared_ptr<::alaya::disk::DiskFlatExportState> owner;
    ::alaya::disk::DiskFlatExportRequest typed;
    typed.batch_rows = 256;
    typed.lifetime_owner = &owner;
    core::OpaqueOperationRequest request;
    request.payload = &typed;
    request.payload_size = sizeof(typed);
    core::ExportCursor cursor;
    auto status = entry->segment.export_rows(request, cursor);
    if (!status.ok()) {
      return status;
    }
    if (owner == nullptr || cursor.state != owner.get()) {
      return error(core::StatusCode::internal,
                   core::OperationStage::export_rows,
                   core::StatusDetail::malformed_struct,
                   "Flat export cursor did not retain its source epoch");
    }
    bool done{};
    while (!done) {
      ::alaya::disk::DiskFlatExportBatch batch;
      status = owner->next(batch);
      if (!status.ok()) {
        return status;
      }
      if (batch.logical_ids.size() != batch.vectors.rows ||
          batch.vectors.scalar_type != core::ScalarType::float32 ||
          batch.vectors.dim != options_.dim) {
        return error(core::StatusCode::corruption,
                     core::OperationStage::export_rows,
                     core::StatusDetail::malformed_struct,
                     "Flat export batch has an inconsistent row schema");
      }
      for (std::size_t index = 0; index < batch.logical_ids.size(); ++index) {
        const internal::collection::RowAddress address{source.segment_id,
                                                       source.generation,
                                                       core::SegmentRowId(
                                                           batch.logical_ids[index])};
        const auto reverse = snapshot.reverse.find(address);
        if (reverse == snapshot.reverse.end()) {
          continue;
        }
        const auto version = snapshot.versions.find(reverse->second.logical_id);
        if (version == snapshot.versions.end() || version->second.address != address ||
            version->second.state != internal::collection::VersionState::live ||
            !version->second.payload.vector.has_value() ||
            options_.metric == core::Metric::cosine) {
          continue;
        }
        std::vector<float> expected;
        status = internal::collection::detail::vector_as_float(*version->second.payload.vector,
                                                               expected);
        if (!status.ok()) {
          return status;
        }
        if (expected.size() != options_.dim || std::memcmp(expected.data(),
                                                           batch.vectors.row<float>(index),
                                                           expected.size() * sizeof(float)) != 0) {
          return error(core::StatusCode::corruption,
                       core::OperationStage::export_rows,
                       core::StatusDetail::malformed_struct,
                       "Flat compact export row differs from the Collection-owned vector");
        }
      }
      done = batch.done;
    }
  }
  return core::Status::success();
}
}  // namespace alaya
