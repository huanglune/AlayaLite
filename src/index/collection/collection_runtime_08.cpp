// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/collection/collection.hpp"

namespace alaya {

[[nodiscard]] auto Collection::compact_locked(core::SealContext &context)
    -> core::Result<CollectionCompactReceipt> {
  if (auto gate = implementation_->recovery_gate(core::OperationStage::build); !gate.ok()) {
    return gate;
  }
  auto status = core::validate_runtime_control(context.deadline,
                                               context.cancellation,
                                               core::OperationStage::build);
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
  if (control_state_.phase != internal::collection::CollectionControlPhase::idle) {
    return error(core::StatusCode::conflict,
                 core::OperationStage::build,
                 core::StatusDetail::none,
                 "another Collection control-plane operation is in progress");
  }
  auto loaded = internal::collection::load_manifest_v2_if_present(options_.root);
  if (!loaded.ok()) {
    return loaded.status();
  }
  if (!loaded.value().has_value()) {
    return error(core::StatusCode::not_found,
                 core::OperationStage::build,
                 core::StatusDetail::none,
                 "Flat compact requires at least two sealed manifest entries");
  }
  auto base_manifest = std::move(*loaded.value());
  std::vector<internal::collection::RowAddress> sources;
  std::uint64_t input_bytes{};
  for (const auto &entry : base_manifest.segments) {
    if (entry.lifecycle != internal::collection::SegmentLifecycleV2::sealed ||
        entry.algorithm_id != core::algorithm::flat) {
      continue;
    }
    sources.push_back(
        {numeric_segment_id(entry.segment_id), entry.generation, core::SegmentRowId{}});
    for (const auto &artifact : entry.artifacts) {
      if (!core::checked_add(input_bytes, artifact.size_bytes, input_bytes)) {
        input_bytes = std::numeric_limits<std::uint64_t>::max();
        break;
      }
    }
  }
  if (sources.size() < 2) {
    return error(core::StatusCode::not_found,
                 core::OperationStage::build,
                 core::StatusDetail::none,
                 "Flat compact requires at least two sealed Flat segments");
  }
  if (control_state_.next_segment_id > 99'999'999) {
    return error(core::StatusCode::resource_exhausted,
                 core::OperationStage::build,
                 core::StatusDetail::arithmetic_overflow,
                 "Flat segment namespace is exhausted");
  }
  auto pinned = implementation_->pin_routing_snapshot();
  status = verify_flat_exports(*pinned, sources);
  if (!status.ok()) {
    return status;
  }
  const auto target_segment_id = control_state_.next_segment_id++;
  constexpr std::uint64_t kTargetGeneration = 1;
  auto build_data =
      collect_replacement_rows(*pinned, sources, target_segment_id, kTargetGeneration);
  if (!build_data.ok()) {
    return build_data.status();
  }
  if (build_data.value().live_rows == 0) {
    return error(core::StatusCode::not_found,
                 core::OperationStage::build,
                 core::StatusDetail::none,
                 "Flat compact sources contain no live rows");
  }
  status = context.snapshot_reservation.ensure(build_data.value().snapshot_bytes,
                                               core::OperationStage::build,
                                               "compact snapshot reservation is too small");
  if (!status.ok()) {
    return status;
  }
  control_state_.operation = internal::collection::CollectionControlOperation::compact;
  control_state_.phase = internal::collection::CollectionControlPhase::building;
  control_state_.sources = sources;
  control_state_.target_segment_id = target_segment_id;
  control_state_.target_generation = kTargetGeneration;
  control_state_.wal_cut = pinned->visibility_watermark;
  control_state_.pending_compacted_bytes = input_bytes;
  control_state_.mapping_file = "compact_" + std::to_string(target_segment_id) + ".map";
  status =
      internal::collection::CollectionControlStore::save_replacements(options_.root,
                                                                      control_state_.mapping_file,
                                                                      build_data.value()
                                                                          .replacements);
  if (!status.ok()) {
    return status;
  }
  status = internal::collection::CollectionControlStore::save(options_.root, control_state_);
  if (!status.ok()) {
    return status;
  }

  internal::collection::detail::CollectionTargetPublication publication;
  publication.collection_root = options_.root;
  publication.segment_id = internal::collection::detail::collection_segment_name(target_segment_id);
  publication.segment_generation = kTargetGeneration;
  publication.manifest_generation =
      std::max(base_manifest.publication.generation + 1, control_state_.manifest_generation + 1);
  publication.publication_parent = std::string(internal::collection::kCollectionManifestFilename);
  publication.metadata_epoch = pinned->metadata_epoch;
  publication.metadata_checkpoint = "checkpoint_" + std::to_string(control_state_.wal_cut) + ".bin";
  publication.wal_cut = control_state_.wal_cut;
  publication.row_versions = {control_state_.wal_cut == 0 ? std::uint64_t{0} : std::uint64_t{1},
                              control_state_.wal_cut};
  publication.id_map_checkpoint = publication.metadata_checkpoint;
  publication.collection_features.manifest_v2_writer = true;
  publication.abort_policy = internal::collection::ArtifactAbortPolicy::retain_for_restart_cleanup;
  publication.base_manifest = std::move(base_manifest);
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

  for (const auto &source : sources) {
    const auto source_name =
        internal::collection::detail::collection_segment_name(source.segment_id);
    if (auto entry = pinned->find_segment(source.segment_id, source.generation)) {
      pending_gc_.push_back({source_name, options_.root / "segments" / source_name, entry});
    }
  }
  internal::collection::SegmentRegistration target;
  target.segment_id = target_segment_id;
  target.generation = kTargetGeneration;
  target.role = internal::collection::SegmentRole::sealed;
  target.segment = std::move(built_target.segment);
  target.rows = build_data.value().rows;
  status = implementation_->install_segment_replacement(sources,
                                                        std::move(target),
                                                        build_data.value().replacements);
  if (!status.ok()) {
    return status;
  }
  pinned.reset();
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

  CollectionCompactReceipt receipt;
  for (const auto &source : sources) {
    receipt.source_segment_ids.push_back(source.segment_id);
  }
  receipt.compacted_segment_id = target_segment_id;
  receipt.compacted_rows = build_data.value().live_rows;
  receipt.input_bytes = input_bytes;
  receipt.output_bytes = built_target.artifact_bytes;
  receipt.manifest_generation = control_state_.manifest_generation;
  receipt.built_algorithm = built_target.built_algorithm;
  receipt.effective_ef_construction = built_target.effective_ef_construction;
  receipt.flat_fallback = built_target.flat_fallback;
  receipt.fallback_reason = built_target.fallback_reason;
  const auto mapping_file = control_state_.mapping_file;
  if (!core::checked_add(control_state_.compacted_bytes,
                         input_bytes,
                         control_state_.compacted_bytes)) {
    control_state_.compacted_bytes = std::numeric_limits<std::uint64_t>::max();
  }
  control_state_.pending_compacted_bytes = 0;
  control_state_.operation = internal::collection::CollectionControlOperation::idle;
  control_state_.phase = internal::collection::CollectionControlPhase::idle;
  control_state_.last_sealed_segment_id = target_segment_id;
  control_state_.last_sealed_generation = kTargetGeneration;
  control_state_.sources.clear();
  control_state_.target_segment_id = 0;
  control_state_.target_generation = 0;
  control_state_.wal_cut = 0;
  control_state_.mapping_file.clear();
  status = internal::collection::CollectionControlStore::save(options_.root, control_state_);
  if (!status.ok()) {
    return status;
  }
  internal::collection::CollectionControlStore::remove_replacements(options_.root, mapping_file);
  return receipt;
}
}  // namespace alaya
