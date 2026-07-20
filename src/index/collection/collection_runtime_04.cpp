// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// collection_runtime_04: segment open, replacement rows, and checkpoint internals.
// One compile-cost-balanced Collection runtime unit; see CMakeLists.txt.

#include "index/collection/collection.hpp"

namespace alaya {

[[nodiscard]] auto Collection::numeric_segment_id(std::string_view segment_id) -> std::uint64_t {
  if (!segment_id.starts_with("seg_") || segment_id.size() != 12) {
    throw std::invalid_argument("Collection segment identity is malformed");
  }
  return parse_u64(segment_id.substr(4));
}

[[nodiscard]] auto Collection::open_segmented(
    const CollectionOptions &options,
    const internal::collection::CollectionControlState &control_state,
    bool read_only) -> core::Result<std::shared_ptr<internal::collection::SegmentedCollection>> {
  internal::collection::CollectionSchema schema{options.dim,
                                                options.metric,
                                                options.scalar_type,
                                                options.max_logical_id_bytes};
  std::vector<internal::collection::SegmentRegistration> registrations;
  auto manifest = internal::collection::load_manifest_v2_if_present(options.root);
  if (!manifest.ok()) {
    return manifest.status();
  }
  if (manifest.value().has_value()) {
    for (const auto &entry : manifest.value()->segments) {
      if (entry.lifecycle == internal::collection::SegmentLifecycleV2::retired ||
          entry.lifecycle == internal::collection::SegmentLifecycleV2::gc_pending) {
        continue;
      }
      core::OpenContext context;
      auto erased = internal::collection::detail::CollectionSegmentFactory::open_entry(options.root,
                                                                                       entry,
                                                                                       schema,
                                                                                       context);
      if (!erased.ok()) {
        return erased.status();
      }
      internal::collection::SegmentRegistration registration;
      registration.segment_id = numeric_segment_id(entry.segment_id);
      registration.generation = entry.generation;
      registration.role = internal::collection::SegmentRole::sealed;
      registration.segment = std::move(erased).value();
      registrations.push_back(std::move(registration));
    }
  }

  const auto already_registered = [&](std::uint64_t segment_id, std::uint64_t generation) {
    return std::ranges::any_of(registrations, [&](const auto &registration) {
      return registration.segment_id == segment_id && registration.generation == generation;
    });
  };
  if (control_state.phase != internal::collection::CollectionControlPhase::idle &&
      control_state.phase != internal::collection::CollectionControlPhase::cut_pending) {
    for (const auto &source : control_state.sources) {
      if (already_registered(source.segment_id, source.generation)) {
        continue;
      }
      auto source_registration =
          make_active_registration(options, source.segment_id, source.generation);
      if (!source_registration.ok()) {
        return source_registration.status();
      }
      source_registration.value().role = internal::collection::SegmentRole::sealed;
      registrations.push_back(std::move(source_registration).value());
    }
  }

  if (!read_only && options.active_engine == core::algorithm::laser) {
    sweep_orphan_active_laser_dirs(options.root, control_state);
  }
  auto active = make_active_registration(options,
                                         control_state.active_segment_id,
                                         control_state.active_generation);
  if (!active.ok()) {
    return active.status();
  }
  internal::collection::CollectionConfig config;
  config.features.wal_coordinator = true;
  config.features.manifest_v2_writer = true;
  config.wal.root = options.root;
  config.read_only = read_only;
  registrations.push_back(std::move(active).value());
  return internal::collection::SegmentedCollection::open(schema,
                                                         std::move(registrations),
                                                         std::move(config));
}

void Collection::fire_seal_failpoint(const CollectionSealOptions &options,
                                     CollectionSealFailPoint point) {
  if (options.fail_point == point && options.failpoint_hook) {
    options.failpoint_hook(point);
  }
}

[[nodiscard]] auto Collection::address_is_source(
    const internal::collection::RowAddress &address,
    std::span<const internal::collection::RowAddress> sources) -> bool {
  return std::ranges::any_of(sources, [&](const auto &source) {
    return source.segment_id == address.segment_id && source.generation == address.generation;
  });
}

[[nodiscard]] auto Collection::collect_replacement_rows(
    const internal::collection::RoutingSnapshot &snapshot,
    std::span<const internal::collection::RowAddress> sources,
    std::uint64_t target_segment_id,
    std::uint64_t target_generation) -> core::Result<ReplacementBuildData> {
  ReplacementBuildData result;
  std::uint64_t next_row{};
  for (const auto state :
       {internal::collection::VersionState::live, internal::collection::VersionState::tombstone}) {
    for (const auto &[logical_id, version] : snapshot.versions) {
      if (version.state != state || !address_is_source(version.address, sources)) {
        continue;
      }
      internal::collection::RowAddress target{target_segment_id,
                                              target_generation,
                                              core::SegmentRowId(next_row++)};
      result.replacements.push_back({logical_id, version.address, target, version.upsert_sequence});
      result.rows.push_back(
          {logical_id, target.row_id, version.upsert_sequence, version.state, version.payload});
      if (version.state == internal::collection::VersionState::live) {
        ++result.live_rows;
      }
      std::uint64_t row_bytes = version.payload.document.size();
      if (version.payload.vector.has_value() &&
          !core::checked_add(row_bytes, version.payload.vector->bytes().size(), row_bytes)) {
        return error(core::StatusCode::resource_exhausted,
                     core::OperationStage::freeze,
                     core::StatusDetail::arithmetic_overflow,
                     "seal snapshot accounting overflowed");
      }
      for (const auto &[key, value] : version.payload.metadata) {
        std::uint64_t scalar_bytes = key.size();
        std::visit(
            [&](const auto &item) {
              using T = std::decay_t<decltype(item)>;
              if constexpr (std::is_same_v<T, std::string>) {
                scalar_bytes += item.size();
              } else {
                scalar_bytes += sizeof(item);
              }
            },
            value);
        if (!core::checked_add(row_bytes, scalar_bytes, row_bytes)) {
          return error(core::StatusCode::resource_exhausted,
                       core::OperationStage::freeze,
                       core::StatusDetail::arithmetic_overflow,
                       "seal metadata accounting overflowed");
        }
      }
      if (!core::checked_add(result.snapshot_bytes, row_bytes, result.snapshot_bytes)) {
        return error(core::StatusCode::resource_exhausted,
                     core::OperationStage::freeze,
                     core::StatusDetail::arithmetic_overflow,
                     "seal snapshot total accounting overflowed");
      }
    }
  }
  return result;
}

[[nodiscard]] auto Collection::checkpoint_locked(core::CheckpointContext &context)
    -> core::Result<CollectionCheckpointReceipt> {
  auto receipt = implementation_->checkpoint(context);
  if (!receipt.ok()) {
    return receipt.status();
  }
  auto manifest = internal::collection::load_manifest_v2_if_present(options_.root);
  if (!manifest.ok()) {
    return manifest.status();
  }
  if (manifest.value().has_value()) {
    auto updated = std::move(*manifest.value());
    internal::collection::SegmentedCollection::apply_checkpoint_to_manifest(receipt.value(),
                                                                            updated);
    updated.publication.generation =
        std::max(updated.publication.generation + 1, control_state_.manifest_generation + 1);
    updated.publication.parent = std::string(internal::collection::kCollectionManifestFilename);
    auto status = internal::collection::publish_manifest_v2_atomic(options_.root, updated);
    if (!status.ok()) {
      return status;
    }
    control_state_.manifest_generation = updated.publication.generation;
    status = internal::collection::CollectionControlStore::save(options_.root, control_state_);
    if (!status.ok()) {
      return status;
    }
  }
  return receipt;
}

[[nodiscard]] auto Collection::patch_published_target_manifest() -> core::Status {
  auto loaded = internal::collection::load_manifest_v2_if_present(options_.root);
  if (!loaded.ok()) {
    return loaded.status();
  }
  if (!loaded.value().has_value()) {
    return error(core::StatusCode::corruption,
                 core::OperationStage::save,
                 core::StatusDetail::malformed_struct,
                 "Collection target build did not publish manifest v2");
  }
  auto manifest = std::move(*loaded.value());
  const auto target_name =
      internal::collection::detail::collection_segment_name(control_state_.target_segment_id);
  const auto target = std::ranges::find_if(manifest.segments, [&](const auto &entry) {
    return entry.segment_id == target_name && entry.generation == control_state_.target_generation;
  });
  if (target == manifest.segments.end()) {
    return error(core::StatusCode::corruption,
                 core::OperationStage::save,
                 core::StatusDetail::malformed_struct,
                 "published manifest omits the Collection replacement target");
  }
  target->lifecycle = internal::collection::SegmentLifecycleV2::sealed;
  target->source_retention.clear();
  for (const auto &source : control_state_.sources) {
    const auto source_name =
        internal::collection::detail::collection_segment_name(source.segment_id);
    target->source_retention.push_back(source_name);
    const auto source_entry = std::ranges::find_if(manifest.segments, [&](const auto &entry) {
      return entry.segment_id == source_name && entry.generation == source.generation;
    });
    if (source_entry != manifest.segments.end()) {
      source_entry->lifecycle = internal::collection::SegmentLifecycleV2::gc_pending;
    }
    if (std::ranges::find(manifest.gc.pending_segment_ids, source_name) ==
        manifest.gc.pending_segment_ids.end()) {
      manifest.gc.pending_segment_ids.push_back(source_name);
    }
  }
  manifest.gc.phase = internal::collection::GcPhaseV2::pending;
  ++manifest.gc.generation;
  manifest.gc.retained_sources = {target_name};
  manifest.publication.generation =
      std::max(manifest.publication.generation + 1, control_state_.manifest_generation + 1);
  manifest.publication.parent = std::string(internal::collection::kCollectionManifestFilename);
  auto status = internal::collection::publish_manifest_v2_atomic(options_.root, manifest);
  if (!status.ok()) {
    return status;
  }
  control_state_.manifest_generation = manifest.publication.generation;
  return core::Status::success();
}
}  // namespace alaya
