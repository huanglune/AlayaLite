// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// collection_runtime_09: GC, write/search dispatch, and facade schema write.
// One compile-cost-balanced Collection runtime unit; see CMakeLists.txt.

#include "index/collection/collection.hpp"

namespace alaya {

[[nodiscard]] auto Collection::gc_locked() -> core::Result<CollectionGcReceipt> {
  if (auto gate = implementation_->recovery_gate(core::OperationStage::save); !gate.ok()) {
    return gate;
  }
  auto loaded = internal::collection::load_manifest_v2_if_present(options_.root);
  if (!loaded.ok()) {
    return loaded.status();
  }
  CollectionGcReceipt receipt;
  if (!loaded.value().has_value()) {
    return receipt;
  }
  auto manifest = std::move(*loaded.value());
  receipt.pending = manifest.gc.pending_segment_ids.size();
  std::vector<std::string> reclaim;
  const auto retained = control_state_.last_sealed_segment_id == 0
                            ? std::string{}
                            : internal::collection::detail::collection_segment_name(
                                  control_state_.last_sealed_segment_id);
  for (const auto &segment_id : manifest.gc.pending_segment_ids) {
    if (segment_id == retained) {
      ++receipt.deferred;
      continue;
    }
    const auto candidate = std::ranges::find_if(pending_gc_, [&](const auto &item) {
      return item.manifest_segment_id == segment_id;
    });
    if (candidate != pending_gc_.end() && !candidate->epoch_reference.expired()) {
      ++receipt.deferred;
      continue;
    }
    reclaim.push_back(segment_id);
  }
  if (reclaim.empty()) {
    receipt.manifest_generation = manifest.publication.generation;
    return receipt;
  }

  manifest.gc.phase = internal::collection::GcPhaseV2::reclaimable;
  ++manifest.gc.generation;
  manifest.publication.generation =
      std::max(manifest.publication.generation + 1, control_state_.manifest_generation + 1);
  manifest.publication.parent = std::string(internal::collection::kCollectionManifestFilename);
  auto status = internal::collection::publish_manifest_v2_atomic(options_.root, manifest);
  if (!status.ok()) {
    return status;
  }

  try {
    for (const auto &segment_id : reclaim) {
      const auto entry = std::ranges::find_if(manifest.segments, [&](const auto &item) {
        return item.segment_id == segment_id;
      });
      if (entry != manifest.segments.end()) {
        if (entry->lifecycle != internal::collection::SegmentLifecycleV2::gc_pending) {
          return error(core::StatusCode::conflict,
                       core::OperationStage::save,
                       core::StatusDetail::none,
                       "GC refused a segment that is not gc_pending");
        }
        for (const auto &artifact : entry->artifacts) {
          if (!core::checked_add(receipt.reclaimed_bytes,
                                 artifact.size_bytes,
                                 receipt.reclaimed_bytes)) {
            receipt.reclaimed_bytes = std::numeric_limits<std::uint64_t>::max();
            break;
          }
        }
        std::filesystem::remove_all(options_.root / "segments" / segment_id);
      }
      ++receipt.reclaimed;
    }
    const auto segments_root = options_.root / "segments";
    if (std::filesystem::is_directory(segments_root)) {
      platform::sync_directory_or_throw(segments_root);
    }
  } catch (...) {
    return core::status_from_exception(core::OperationStage::save);
  }

  std::erase_if(manifest.segments, [&](const auto &entry) {
    return std::ranges::find(reclaim, entry.segment_id) != reclaim.end();
  });
  std::erase_if(manifest.gc.pending_segment_ids, [&](const auto &segment_id) {
    return std::ranges::find(reclaim, segment_id) != reclaim.end();
  });
  manifest.gc.phase = manifest.gc.pending_segment_ids.empty()
                          ? internal::collection::GcPhaseV2::idle
                          : internal::collection::GcPhaseV2::pending;
  ++manifest.gc.generation;
  manifest.gc.retained_sources =
      retained.empty() ? std::vector<std::string>{} : std::vector<std::string>{retained};
  manifest.publication.generation =
      std::max(manifest.publication.generation + 1, control_state_.manifest_generation + 1);
  manifest.publication.parent = std::string(internal::collection::kCollectionManifestFilename);
  status = internal::collection::publish_manifest_v2_atomic(options_.root, manifest);
  if (!status.ok()) {
    return status;
  }
  control_state_.manifest_generation = manifest.publication.generation;
  status = internal::collection::CollectionControlStore::save(options_.root, control_state_);
  if (!status.ok()) {
    return status;
  }
  std::erase_if(pending_gc_, [&](const auto &candidate) {
    return std::ranges::find(reclaim, candidate.manifest_segment_id) != reclaim.end();
  });
  receipt.manifest_generation = manifest.publication.generation;
  return receipt;
}

[[nodiscard]] auto Collection::write(const CollectionItem &item,
                                     internal::collection::WriteMode mode,
                                     CollectionWriteOptions options)
    -> core::Result<CollectionMutationReceipt> {
  if (const auto writable = ensure_writable(core::OperationStage::admission); !writable.ok()) {
    return writable;
  }
  internal::collection::WriteRequest request;
  request.logical_id = item.logical_id;
  request.vector = item.vector;
  request.metadata = item.metadata;
  request.document = item.document;
  request.mode = mode;
  request.options = std::move(options);
  core::MutationContext context;
  auto receipt = implementation_->write(request, context);
  if (receipt.ok()) {
    maybe_auto_seal();
  }
  return receipt;
}

void Collection::maybe_auto_seal() noexcept {
  if (options_.auto_seal_rows == 0) {
    return;
  }
  try {
    const auto snapshot = implementation_->pin_routing_snapshot();
    const auto active = snapshot->find_active_mutable();
    if (active == nullptr || snapshot->known_rows_for(*active) < options_.auto_seal_rows) {
      return;
    }
    (void)seal();
  } catch (...) {
    // The committed mutation remains authoritative. Auto-seal is a
    // best-effort control-plane policy and can be retried explicitly.
  }
}

[[nodiscard]] auto Collection::execute_search(const core::TypedTensorView &queries,
                                              const core::SearchOptions &options,
                                              core::SearchContext &context,
                                              const CollectionFilter &filter)
    -> core::Result<CollectionSearchResponse> {
  CollectionSearchStatistics search_stats;
  internal::collection::CollectionSearchRequest request;
  request.queries = queries;
  request.options = options;
  request.filter = filter;
  request.context = std::addressof(context);
  request.stats = std::addressof(search_stats);
  auto result = implementation_->search(request);
  if (!result.ok()) {
    return result.status();
  }
  CollectionSearchResponse response;
  response.search_stats = search_stats;
  response.visibility_watermark = result.value().visibility_watermark;
  response.metadata_epoch = result.value().metadata_epoch;
  response.offsets.reserve(result.value().queries.size() + 1);
  response.valid_counts.reserve(result.value().queries.size());
  response.statuses.reserve(result.value().queries.size());
  response.completeness.reserve(result.value().queries.size());
  response.offsets.push_back(0);
  for (const auto &query : result.value().queries) {
    response.valid_counts.push_back(query.hits.size());
    response.statuses.push_back(query.status);
    response.completeness.push_back(query.completeness);
    for (const auto &hit : query.hits) {
      response.ids.push_back(hit.logical_id);
      response.distances.push_back(hit.score);
    }
    response.offsets.push_back(response.ids.size());
  }
  return response;
}

[[nodiscard]] auto Collection::facade_schema_path(const std::filesystem::path &root)
    -> std::filesystem::path {
  return root / ".alaya_internal" / kFacadeNamespace / kFacadeSchemaFilename;
}

[[nodiscard]] auto Collection::schema_prefix(const CollectionOptions &options) -> std::string {
  std::string prefix =
      "format=1\npublic_version=" + std::string(kCollectionPublicVersion) +
      "\ndim=" + std::to_string(options.dim) +
      "\nmetric=" + std::to_string(static_cast<unsigned>(options.metric)) +
      "\nscalar_type=" + std::to_string(static_cast<unsigned>(options.scalar_type)) +
      "\ntarget_algorithm=" + std::to_string(options.target_algorithm) +
      "\nquantization=" + std::to_string(static_cast<unsigned>(options.quantization)) +
      "\nbuild_threads=" + std::to_string(options.build_threads) +
      "\nmax_neighbors=" + std::to_string(options.max_neighbors) +
      "\nef_construction=" + std::to_string(options.ef_construction) +
      "\nmax_logical_id_bytes=" + std::to_string(options.max_logical_id_bytes) +
      "\nactive_segment_id=" + std::to_string(kActiveSegmentId) +
      "\nactive_generation=" + std::to_string(kActiveSegmentGeneration) + "\n";
  // B-08: only a non-default active engine widens the schema to 15 fields, so a
  // flat collection stays byte-compatible with pre-2B readers while a laser
  // collection makes an old binary fail-closed on the strict field count.
  if (options.active_engine != core::algorithm::flat) {
    prefix += "active_engine=" + std::to_string(options.active_engine) + "\n";
  }
  return prefix;
}

[[nodiscard]] auto Collection::write_facade_schema(const CollectionOptions &options)
    -> core::Status {
  try {
    const auto path = facade_schema_path(options.root);
    std::filesystem::create_directories(path.parent_path());
    const auto prefix = schema_prefix(options);
    const auto body = prefix + "checksum=" + internal::collection::sha256(prefix).hex() + "\n";
    const auto temporary = path.string() + ".tmp";
    platform::write_all_fsync(temporary, body.data(), body.size());
    platform::atomic_replace(temporary, path);
    platform::sync_directory_or_throw(path.parent_path());
    return core::Status::success();
  } catch (...) {
    return core::status_from_exception(core::OperationStage::save);
  }
}
}  // namespace alaya
