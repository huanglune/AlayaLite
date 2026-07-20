// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/collection/collection.hpp"

namespace alaya {

[[nodiscard]] auto Collection::prepare_successor_locked(core::SealContext &context,
                                                        const CollectionSealOptions &options)
    -> core::Result<CollectionRotationHandle> {
  if (auto gate = implementation_->recovery_gate(core::OperationStage::freeze); !gate.ok()) {
    return gate;
  }
  if (pending_rotation_.has_value()) {
    return error(core::StatusCode::conflict,
                 core::OperationStage::freeze,
                 core::StatusDetail::none,
                 "a successor is already prepared; call rotate_to_successor() first");
  }
  auto status = core::validate_runtime_control(context.deadline,
                                               context.cancellation,
                                               core::OperationStage::freeze);
  if (!status.ok()) {
    return status;
  }
  status = normalize_control_state_before_open(options_.root, control_state_);
  if (!status.ok()) {
    return status;
  }
  if (control_state_.phase == internal::collection::CollectionControlPhase::manifest_published) {
    status = recover_control_state();
    if (!status.ok()) {
      return status;
    }
  }
  if (control_state_.phase != internal::collection::CollectionControlPhase::idle &&
      control_state_.phase != internal::collection::CollectionControlPhase::successor_active) {
    return error(core::StatusCode::conflict,
                 core::OperationStage::freeze,
                 core::StatusDetail::none,
                 "another Collection control-plane operation is in progress");
  }

  if (control_state_.phase == internal::collection::CollectionControlPhase::idle) {
    const auto snapshot = implementation_->pin_routing_snapshot();
    const auto source = snapshot->find_active_mutable();
    if (source == nullptr || snapshot->known_rows_for(*source) == 0) {
      return error(core::StatusCode::not_found,
                   core::OperationStage::freeze,
                   core::StatusDetail::none,
                   "cannot seal an empty active segment");
    }
    if (control_state_.next_segment_id > 99'999'998) {
      return error(core::StatusCode::resource_exhausted,
                   core::OperationStage::freeze,
                   core::StatusDetail::arithmetic_overflow,
                   "Collection segment namespace is exhausted");
    }
    control_state_.operation = internal::collection::CollectionControlOperation::seal;
    control_state_.phase = internal::collection::CollectionControlPhase::cut_pending;
    control_state_.sources = {internal::collection::RowAddress{source->segment_id,
                                                               source->generation,
                                                               core::SegmentRowId{}}};
    control_state_.successor_segment_id = control_state_.next_segment_id++;
    control_state_.successor_generation = 1;
    control_state_.target_segment_id = control_state_.next_segment_id++;
    control_state_.target_generation = 1;
    control_state_.wal_cut = snapshot->visibility_watermark;
    control_state_.mapping_file.clear();
    status = internal::collection::CollectionControlStore::save(options_.root, control_state_);
    if (!status.ok()) {
      return status;
    }
    fire_seal_failpoint(options, CollectionSealFailPoint::after_cut_before_successor);

    if (options_.active_engine == core::algorithm::laser) {
      auto created = create_active_laser_segment(options_,
                                                 control_state_.successor_segment_id,
                                                 control_state_.successor_generation);
      if (!created.ok()) {
        return created;
      }
    }
    auto successor = make_active_registration(options_,
                                              control_state_.successor_segment_id,
                                              control_state_.successor_generation);
    if (!successor.ok()) {
      return successor.status();
    }
    core::CheckpointContext checkpoint_context;
    checkpoint_context.deadline = context.deadline;
    checkpoint_context.cancellation = context.cancellation;
    checkpoint_context.lane = context.lane;
    checkpoint_context.dirty_page_io_credits = context.io_credits;
    checkpoint_context.wal_io_credits = context.io_credits;
    checkpoint_context.durability_target = core::DurabilityTarget::full_checkpoint;
    auto rotated = implementation_->rotate_to_successor(
        std::move(successor).value(),
        checkpoint_context,
        [&](const internal::collection::ActiveRotationReceipt &receipt) {
          control_state_.active_segment_id = receipt.successor_segment_id;
          control_state_.active_generation = receipt.successor_generation;
          control_state_.wal_cut = receipt.checkpoint.wal_cut;
          control_state_.phase = internal::collection::CollectionControlPhase::successor_active;
          auto saved =
              internal::collection::CollectionControlStore::save(options_.root, control_state_);
          if (!saved.ok()) {
            return saved;
          }
          fire_seal_failpoint(options,
                              CollectionSealFailPoint::
                                  after_active_control_publish_before_routing_install);
          return core::Status::success();
        });
    if (!rotated.ok()) {
      return rotated.status();
    }
    fire_seal_failpoint(options, CollectionSealFailPoint::after_successor_switch);
  }

  auto pinned = implementation_->pin_routing_snapshot();
  auto build_data = collect_replacement_rows(*pinned,
                                             control_state_.sources,
                                             control_state_.target_segment_id,
                                             control_state_.target_generation);
  if (!build_data.ok()) {
    return build_data.status();
  }
  if (build_data.value().live_rows == 0) {
    return error(core::StatusCode::not_found,
                 core::OperationStage::build,
                 core::StatusDetail::none,
                 "active seal snapshot contains no live rows");
  }
  status = context.snapshot_reservation.ensure(build_data.value().snapshot_bytes,
                                               core::OperationStage::freeze,
                                               "seal snapshot reservation is too small");
  if (!status.ok()) {
    return status;
  }
  control_state_.mapping_file = "seal_" + std::to_string(control_state_.target_segment_id) + ".map";
  status =
      internal::collection::CollectionControlStore::save_replacements(options_.root,
                                                                      control_state_.mapping_file,
                                                                      build_data.value()
                                                                          .replacements);
  if (!status.ok()) {
    return status;
  }
  control_state_.phase = internal::collection::CollectionControlPhase::building;
  status = internal::collection::CollectionControlStore::save(options_.root, control_state_);
  if (!status.ok()) {
    return status;
  }
  auto base_manifest = internal::collection::load_manifest_v2_if_present(options_.root);
  if (!base_manifest.ok()) {
    return base_manifest.status();
  }
  internal::collection::detail::CollectionTargetPublication publication;
  publication.collection_root = options_.root;
  publication.segment_id =
      internal::collection::detail::collection_segment_name(control_state_.target_segment_id);
  publication.segment_generation = control_state_.target_generation;
  publication.manifest_generation =
      std::max(control_state_.manifest_generation + 1,
               base_manifest.value().has_value() ? base_manifest.value()->publication.generation + 1
                                                 : std::uint64_t{1});
  publication.publication_parent = std::string(internal::collection::kCollectionManifestFilename);
  publication.metadata_epoch = pinned->metadata_epoch;
  publication.metadata_checkpoint = "checkpoint_" + std::to_string(control_state_.wal_cut) + ".bin";
  publication.wal_cut = control_state_.wal_cut;
  publication.row_versions = {control_state_.wal_cut == 0 ? std::uint64_t{0} : std::uint64_t{1},
                              control_state_.wal_cut};
  publication.id_map_checkpoint = publication.metadata_checkpoint;
  publication.collection_features.manifest_v2_writer = true;
  publication.abort_policy = internal::collection::ArtifactAbortPolicy::retain_for_restart_cleanup;
  if (options.fail_point == CollectionSealFailPoint::during_export_build &&
      options.failpoint_hook) {
    publication.fail_point =
        internal::collection::ArtifactTransactionFailPoint::after_staging_write;
  }
  publication.base_manifest = std::move(base_manifest).value();
  core::BuildContext build_context;
  build_context.growing_reservation = context.build_reservation;
  build_context.io_credits = context.io_credits;
  build_context.deadline = context.deadline;
  build_context.cancellation = context.cancellation;
  build_context.lane = context.lane;
  internal::collection::CollectionSchema schema{options_.dim,
                                                options_.metric,
                                                options_.scalar_type,
                                                options_.max_logical_id_bytes};
  internal::collection::detail::CollectionTargetBuildParams build_params;
  build_params.quantization = options_.quantization;
  build_params.max_neighbors = options_.max_neighbors;
  build_params.ef_construction = options_.ef_construction;
  build_params.thread_count = options_.build_threads;
  const auto resolution = resolve_build_algorithm(options_.target_algorithm,
                                                  schema,
                                                  build_data.value().live_rows,
                                                  build_params);
  auto built = internal::collection::detail::build_collection_target(resolution.algorithm,
                                                                     schema,
                                                                     build_data.value().rows,
                                                                     build_params,
                                                                     publication,
                                                                     build_context);
  if (options.fail_point == CollectionSealFailPoint::during_export_build &&
      options.failpoint_hook) {
    fire_seal_failpoint(options, CollectionSealFailPoint::during_export_build);
  }
  if (!built.ok()) {
    return built.status();
  }
  auto built_target = std::move(built).value();
  built_target.requested_algorithm = options_.target_algorithm;
  built_target.flat_fallback = resolution.flat_fallback;
  built_target.fallback_reason = resolution.fallback_reason;
  if (resolution.flat_fallback) {
    built_target.built_algorithm = core::algorithm::flat;
  }
  status = patch_published_target_manifest();
  if (!status.ok()) {
    return status;
  }
  control_state_.phase = internal::collection::CollectionControlPhase::manifest_published;
  status = internal::collection::CollectionControlStore::save(options_.root, control_state_);
  if (!status.ok()) {
    return status;
  }
  fire_seal_failpoint(options, CollectionSealFailPoint::after_manifest_publish);

  // Register the predecessor(s) for deferred GC now, in the same
  // control_mutex_ hold that just made them gc_pending in the durable
  // manifest. gc() takes control_mutex_ too, so without this an
  // interleaved gc() call between prepare_successor() and
  // rotate_to_successor() would see a segment the manifest calls
  // gc_pending but that pending_gc_ (in memory) knows nothing about yet,
  // and could reclaim it on disk while it is still the live, routed
  // predecessor. Registering it here means gc()'s weak_ptr check always
  // finds it non-expired (still routed) until rotate_to_successor_locked()
  // actually removes it from the routing table, so an interleaved gc()
  // still correctly defers instead of reclaiming a live segment.
  for (const auto &source : control_state_.sources) {
    if (auto entry = pinned->find_segment(source.segment_id, source.generation)) {
      pending_gc_.push_back(
          {internal::collection::detail::collection_segment_name(source.segment_id), {}, entry});
    }
  }

  CollectionRotationHandle handle;
  handle.successor_segment_id = control_state_.target_segment_id;
  handle.successor_generation = control_state_.target_generation;
  handle.predecessor_segment_ids.reserve(control_state_.sources.size());
  for (const auto &source : control_state_.sources) {
    handle.predecessor_segment_ids.push_back(source.segment_id);
  }
  pending_rotation_ = PendingRotation{std::move(build_data).value(), std::move(built_target)};
  return handle;
}
}  // namespace alaya
