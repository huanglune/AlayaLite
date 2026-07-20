// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/collection/collection.hpp"

namespace alaya {

[[nodiscard]] auto Collection::stats() const -> CollectionStatistics {
  const auto native = implementation_->stats();
  CollectionStatistics result;
  result.size = native.size;
  result.accepted_count = native.accepted_count;
  result.pending_count = native.pending_count;
  result.pending_bytes = native.pending_bytes;
  result.allocated_count = native.allocated_count;
  result.tombstone_count = native.tombstone_count;
  result.routing_generation = native.routing_generation;
  result.visibility_watermark = native.visibility_watermark;
  result.durable_watermark = native.durable_watermark;
  result.metadata_epoch = native.metadata_epoch;
  result.lifecycle = native.lifecycle;
  std::uint64_t row_bytes{};
  if (core::checked_multiply(options_.dim,
                             core::scalar_type_size(options_.scalar_type),
                             row_bytes) &&
      core::checked_multiply(result.size, row_bytes, result.searchable_vector_bytes)) {
    if (!core::checked_add(result.searchable_vector_bytes,
                           result.pending_bytes,
                           result.accepted_vector_bytes)) {
      result.accepted_vector_bytes = std::numeric_limits<std::uint64_t>::max();
    }
  } else {
    result.searchable_vector_bytes = std::numeric_limits<std::uint64_t>::max();
    result.accepted_vector_bytes = std::numeric_limits<std::uint64_t>::max();
  }
  result.searchable_bytes = result.searchable_vector_bytes;
  result.accepted_bytes = result.accepted_vector_bytes;
  result.active_segment_algorithm = core::algorithm::flat;
  {
    std::lock_guard lock(control_mutex_);
    result.compacted_bytes = control_state_.compacted_bytes;
    const auto manifest = internal::collection::load_manifest_v2_if_present(options_.root);
    if (manifest.ok() && manifest.value().has_value()) {
      for (const auto &entry : manifest.value()->segments) {
        if (entry.lifecycle == internal::collection::SegmentLifecycleV2::sealed) {
          ++result.sealed_segments_count;
        }
      }
      result.gc_pending_count = manifest.value()->gc.pending_segment_ids.size();
    }
  }
  return result;
}

[[nodiscard]] auto Collection::target_implementation_key() const -> std::string_view {
  const auto *registration =
      internal::collection::detail::find_collection_target_registration(options_.target_algorithm);
  return registration == nullptr ? std::string_view{"unknown"} : registration->implementation_key;
}

[[nodiscard]] auto Collection::target_engine_factory_key() const -> std::string_view {
  const auto *registration =
      internal::collection::detail::find_collection_target_registration(options_.target_algorithm);
  return registration == nullptr ? std::string_view{"unknown"} : registration->factory_key;
}

[[nodiscard]] auto Collection::close() -> core::Status {
  std::lock_guard lock(control_mutex_);
  auto status = implementation_->close();
  if (!status.ok()) {
    return status;
  }
  status = implementation_->drain();
  if (status.ok()) {
    closed_.store(true, std::memory_order_release);
    process_lock_.reset();
  }
  return status;
}

[[nodiscard]] auto Collection::error(core::StatusCode code,
                                     core::OperationStage stage,
                                     core::StatusDetail detail,
                                     std::string diagnostic) -> core::Status {
  return core::Status::error(code, stage, detail, std::move(diagnostic));
}

[[nodiscard]] auto Collection::readonly_open_requires_recovery(std::string diagnostic)
    -> core::Status {
  return error(core::StatusCode::not_supported,
               core::OperationStage::open,
               core::StatusDetail::readonly_instance,
               std::move(diagnostic));
}

[[nodiscard]] auto Collection::ensure_writable(core::OperationStage stage) const -> core::Status {
  if (closed_.load(std::memory_order_acquire)) {
    return error(core::StatusCode::closed,
                 stage,
                 core::StatusDetail::operation_slot_absent,
                 "Collection handle is closed");
  }
  if (!read_only_) {
    return core::Status::success();
  }
  return error(core::StatusCode::not_supported,
               stage,
               core::StatusDetail::readonly_instance,
               "operation is unavailable on a read-only Collection handle");
}

[[nodiscard]] auto Collection::resolve_build_algorithm(
    core::AlgorithmId requested_algorithm,
    const internal::collection::CollectionSchema &schema,
    core::RowCount live_row_count,
    const internal::collection::detail::CollectionTargetBuildParams &params)
    -> BuildAlgorithmResolution {
  const auto *registration =
      internal::collection::detail::find_collection_target_registration(requested_algorithm);
  if (registration != nullptr && registration->supports(schema, live_row_count, params) ==
                                     internal::collection::detail::TargetSupport::supported) {
    return {requested_algorithm, false, {}};
  }

  BuildAlgorithmResolution resolution;
  resolution.algorithm = core::algorithm::flat;
  resolution.flat_fallback = true;
  if (registration == nullptr) {
    resolution.fallback_reason = "requested Collection target algorithm " +
                                 std::to_string(requested_algorithm) +
                                 " has no registered sealed builder; built Flat instead";
  } else if (requested_algorithm == core::algorithm::qg && live_row_count <= 32) {
    resolution.fallback_reason = "qg requires >32 live rows; built Flat instead";
  } else if (requested_algorithm == core::algorithm::qg &&
             schema.scalar_type != core::ScalarType::float32) {
    resolution.fallback_reason = "qg requires float32 vectors; built Flat instead";
  } else if (requested_algorithm == core::algorithm::laser && live_row_count <= 32) {
    resolution.fallback_reason = "laser requires >32 live rows; built Flat instead";
  } else if (requested_algorithm == core::algorithm::laser &&
             !::alaya::disk::laser_importer_detail::dimension_supported_v1(schema.dim)) {
    resolution.fallback_reason =
        "laser requires dim in [33, 2048]; non-power-of-two dims use FHT padding; built Flat "
        "instead";
  } else {
    resolution.fallback_reason = "requested Collection target '" +
                                 std::string(registration->factory_key) +
                                 "' is unsupported for this schema or live row count; built "
                                 "Flat instead";
  }
  return resolution;
}
}  // namespace alaya
