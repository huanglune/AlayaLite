// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// collection_runtime_05: control-state recovery and seal internals.
// One compile-cost-balanced Collection runtime unit; see CMakeLists.txt.

#include "index/collection/collection.hpp"

namespace alaya {

[[nodiscard]] auto Collection::normalize_control_state_before_open(
    const std::filesystem::path &root,
    internal::collection::CollectionControlState &state) -> core::Status {
  if (state.phase == internal::collection::CollectionControlPhase::cut_pending) {
    internal::collection::CollectionControlStore::remove_replacements(root, state.mapping_file);
    state.operation = internal::collection::CollectionControlOperation::idle;
    state.phase = internal::collection::CollectionControlPhase::idle;
    state.sources.clear();
    state.successor_segment_id = 0;
    state.successor_generation = 0;
    state.target_segment_id = 0;
    state.target_generation = 0;
    state.wal_cut = 0;
    state.mapping_file.clear();
    return internal::collection::CollectionControlStore::save(root, state);
  }

  auto manifest = internal::collection::load_manifest_v2_if_present(root);
  if (!manifest.ok()) {
    return manifest.status();
  }
  const auto target_name =
      state.target_segment_id == 0
          ? std::string{}
          : internal::collection::detail::collection_segment_name(state.target_segment_id);
  const auto target_published =
      manifest.value().has_value() &&
      std::ranges::any_of(manifest.value()->segments, [&](const auto &entry) {
        return entry.segment_id == target_name && entry.generation == state.target_generation;
      });
  if (state.phase == internal::collection::CollectionControlPhase::building && target_published) {
    state.phase = internal::collection::CollectionControlPhase::manifest_published;
    return internal::collection::CollectionControlStore::save(root, state);
  }
  if (state.phase == internal::collection::CollectionControlPhase::manifest_published &&
      !target_published) {
    return error(core::StatusCode::corruption,
                 core::OperationStage::open,
                 core::StatusDetail::malformed_struct,
                 "Collection state says manifest-published but the target is absent");
  }
  if (state.phase == internal::collection::CollectionControlPhase::building && !target_published) {
    auto status = internal::collection::ArtifactControlPlaneTransaction::cleanup_orphans(root);
    if (!status.ok()) {
      return status;
    }
    internal::collection::CollectionControlStore::remove_replacements(root, state.mapping_file);
    state.mapping_file.clear();
    state.phase = state.operation == internal::collection::CollectionControlOperation::seal
                      ? internal::collection::CollectionControlPhase::successor_active
                      : internal::collection::CollectionControlPhase::idle;
    if (state.phase == internal::collection::CollectionControlPhase::idle) {
      state.operation = internal::collection::CollectionControlOperation::idle;
      state.pending_compacted_bytes = 0;
      state.sources.clear();
      state.target_segment_id = 0;
      state.target_generation = 0;
    }
    return internal::collection::CollectionControlStore::save(root, state);
  }
  return core::Status::success();
}

[[nodiscard]] auto Collection::recover_control_state() -> core::Status {
  if (control_state_.phase != internal::collection::CollectionControlPhase::manifest_published) {
    return core::Status::success();
  }
  auto status = patch_published_target_manifest();
  if (!status.ok()) {
    return status;
  }
  auto replacements =
      internal::collection::CollectionControlStore::load_replacements(options_.root,
                                                                      control_state_.mapping_file);
  if (!replacements.ok()) {
    return replacements.status();
  }
  auto pinned = implementation_->pin_routing_snapshot();
  for (const auto &source : control_state_.sources) {
    if (auto entry = pinned->find_segment(source.segment_id, source.generation)) {
      const auto source_name =
          internal::collection::detail::collection_segment_name(source.segment_id);
      pending_gc_.push_back(
          {source_name,
           control_state_.operation == internal::collection::CollectionControlOperation::compact
               ? options_.root / "segments" / source_name
               : std::filesystem::path{},
           entry});
    }
  }
  status = implementation_->resume_segment_replacement(control_state_.sources,
                                                       control_state_.target_segment_id,
                                                       control_state_.target_generation,
                                                       replacements.value());
  if (!status.ok()) {
    return status;
  }
  pinned.reset();
  core::CheckpointContext checkpoint_context;
  checkpoint_context.durability_target = core::DurabilityTarget::full_checkpoint;
  auto checkpoint = checkpoint_locked(checkpoint_context);
  if (!checkpoint.ok()) {
    return checkpoint.status();
  }
  const auto mapping_file = control_state_.mapping_file;
  if (control_state_.operation == internal::collection::CollectionControlOperation::compact) {
    if (!core::checked_add(control_state_.compacted_bytes,
                           control_state_.pending_compacted_bytes,
                           control_state_.compacted_bytes)) {
      control_state_.compacted_bytes = std::numeric_limits<std::uint64_t>::max();
    }
  }
  control_state_.pending_compacted_bytes = 0;
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
  return core::Status::success();
}

[[nodiscard]] auto Collection::seal_locked(core::SealContext &context,
                                           const CollectionSealOptions &options)
    -> core::Result<CollectionSealReceipt> {
  if (auto gate = implementation_->recovery_gate(core::OperationStage::freeze); !gate.ok()) {
    return gate;
  }
  auto handle = prepare_successor_locked(context, options);
  if (!handle.ok()) {
    return handle.status();
  }
  return rotate_to_successor_locked(handle.value(), context);
}
}  // namespace alaya
