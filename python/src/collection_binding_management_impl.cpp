// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "collection_binding.hpp"

namespace alaya::python::collection_binding {

[[nodiscard]] auto PyCollection::checkpoint() -> PyCheckpointResponse {
  const auto receipt = [&] {
    py::gil_scoped_release release;
    return unwrap(collection_->checkpoint());
  }();
  return {receipt.durable_watermark,
          receipt.wal_cut,
          receipt.metadata_epoch,
          receipt.checkpoint_name};
}

[[nodiscard]] auto PyCollection::seal() -> PySealResponse {
  const auto receipt = [&] {
    py::gil_scoped_release release;
    return unwrap(collection_->seal());
  }();
  return {receipt.source_segment_id,
          receipt.successor_segment_id,
          receipt.sealed_segment_id,
          receipt.wal_cut,
          receipt.sealed_rows,
          receipt.sealed_bytes,
          receipt.manifest_generation};
}

[[nodiscard]] auto PyCollection::compact() -> PyCompactResponse {
  const auto receipt = [&] {
    py::gil_scoped_release release;
    return unwrap(collection_->compact());
  }();
  return {receipt.source_segment_ids,
          receipt.compacted_segment_id,
          receipt.compacted_rows,
          receipt.input_bytes,
          receipt.output_bytes,
          receipt.manifest_generation};
}

[[nodiscard]] auto PyCollection::gc() -> PyGcResponse {
  const auto receipt = [&] {
    py::gil_scoped_release release;
    return unwrap(collection_->gc());
  }();
  return {receipt.pending,
          receipt.reclaimed,
          receipt.deferred,
          receipt.reclaimed_bytes,
          receipt.manifest_generation};
}

[[nodiscard]] auto PyCollection::stats() const -> PyStatsResponse {
  const auto stats = collection_->stats();
  return {stats.size,
          stats.accepted_count,
          stats.pending_count,
          stats.searchable_bytes,
          stats.accepted_bytes,
          stats.searchable_vector_bytes,
          stats.accepted_vector_bytes,
          stats.pending_bytes,
          stats.allocated_count,
          stats.tombstone_count,
          stats.routing_generation,
          stats.visibility_watermark,
          stats.durable_watermark,
          stats.metadata_epoch,
          stats.sealed_segments_count,
          stats.gc_pending_count,
          algorithm_name(stats.active_segment_algorithm),
          stats.compacted_bytes,
          static_cast<std::uint8_t>(stats.lifecycle)};
}

[[nodiscard]] auto PyCollection::options() const -> PyOptionsResponse {
  const auto &options = collection_->options();
  return {options.root.string(),
          collection_->read_only(),
          options.dim,
          metric_name(options.metric),
          scalar_dtype(options.scalar_type),
          algorithm_name(options.target_algorithm),
          quantization_name(options.quantization),
          options.build_threads,
          options.max_neighbors,
          options.ef_construction,
          std::string(collection_->target_implementation_key()),
          std::string(collection_->target_engine_factory_key()),
          algorithm_name(collection_->active_algorithm()),
          options.auto_seal_rows};
}

void PyCollection::close() {
  py::gil_scoped_release release;
  throw_status(collection_->close());
}

[[nodiscard]] auto PyCollection::read_only() const noexcept -> bool {
  return collection_->read_only();
}

[[nodiscard]] auto PyCollection::collection() const -> const std::shared_ptr<Collection> & {
  return collection_;
}

}  // namespace alaya::python::collection_binding
