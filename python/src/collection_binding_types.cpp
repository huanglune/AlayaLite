// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "collection_binding.hpp"

namespace alaya::python::collection_binding {

void register_response_types(py::module_ &module) {
  py::class_<PyRecordResponse>(module, "_RecordResponse")
      .def_readonly("id", &PyRecordResponse::id)
      .def_readonly("upsert_sequence", &PyRecordResponse::upsert_sequence)
      .def_readonly("document", &PyRecordResponse::document)
      .def_readonly("metadata", &PyRecordResponse::metadata)
      .def_readonly("vector", &PyRecordResponse::vector);

  py::class_<PyMutationRowResponse>(module, "_MutationRowResponse")
      .def_readonly("op_id", &PyMutationRowResponse::op_id)
      .def_readonly("batch_op_id", &PyMutationRowResponse::batch_op_id)
      .def_readonly("row_op_id", &PyMutationRowResponse::row_op_id)
      .def_readonly("visibility_watermark", &PyMutationRowResponse::visibility_watermark)
      .def_readonly("durable_watermark", &PyMutationRowResponse::durable_watermark)
      .def_readonly("searchable", &PyMutationRowResponse::searchable)
      .def_readonly("durability", &PyMutationRowResponse::durability)
      .def_readonly("row_status", &PyMutationRowResponse::row_status)
      .def_readonly("retry_token", &PyMutationRowResponse::retry_token);

  py::class_<PyMutationResponse>(module, "_MutationResponse")
      .def_readonly("batch_op_id", &PyMutationResponse::batch_op_id)
      .def_readonly("visibility_watermark", &PyMutationResponse::visibility_watermark)
      .def_readonly("durable_watermark", &PyMutationResponse::durable_watermark)
      .def_readonly("searchable", &PyMutationResponse::searchable)
      .def_readonly("durability", &PyMutationResponse::durability)
      .def_readonly("retry_token", &PyMutationResponse::retry_token)
      .def_readonly("rows", &PyMutationResponse::rows);

  py::class_<PySearchStatsResponse>(module, "_SearchStatsResponse")
      .def_readonly("filter_active", &PySearchStatsResponse::filter_active)
      .def_readonly("filter_execution", &PySearchStatsResponse::filter_execution)
      .def_readonly("filter_examined", &PySearchStatsResponse::filter_examined)
      .def_readonly("filter_passed", &PySearchStatsResponse::filter_passed)
      .def_readonly("nan_discarded", &PySearchStatsResponse::nan_discarded)
      .def_readonly("overfetch_rounds", &PySearchStatsResponse::overfetch_rounds)
      .def_readonly("budget_consumed", &PySearchStatsResponse::budget_consumed)
      .def_readonly("lease_acquired", &PySearchStatsResponse::lease_acquired)
      .def_readonly("lease_released", &PySearchStatsResponse::lease_released)
      .def_readonly("lease_peak_bytes", &PySearchStatsResponse::lease_peak_bytes)
      .def_readonly("io_requests_consumed", &PySearchStatsResponse::io_requests_consumed)
      .def_readonly("io_bytes_consumed", &PySearchStatsResponse::io_bytes_consumed)
      .def_readonly("rerank_nanoseconds", &PySearchStatsResponse::rerank_nanoseconds)
      .def_readonly("effective_effort", &PySearchStatsResponse::effective_effort);

  py::class_<PySearchResponse>(module, "_SearchResponse")
      .def_readonly("ids", &PySearchResponse::ids)
      .def_readonly("distances", &PySearchResponse::distances)
      .def_readonly("offsets", &PySearchResponse::offsets)
      .def_readonly("valid_counts", &PySearchResponse::valid_counts)
      .def_readonly("status_codes", &PySearchResponse::status_codes)
      .def_readonly("completeness_codes", &PySearchResponse::completeness_codes)
      .def_readonly("visibility_watermark", &PySearchResponse::visibility_watermark)
      .def_readonly("metadata_epoch", &PySearchResponse::metadata_epoch)
      .def_readonly("search_stats", &PySearchResponse::search_stats);

  py::class_<PyCheckpointResponse>(module, "_CheckpointResponse")
      .def_readonly("durable_watermark", &PyCheckpointResponse::durable_watermark)
      .def_readonly("wal_cut", &PyCheckpointResponse::wal_cut)
      .def_readonly("metadata_epoch", &PyCheckpointResponse::metadata_epoch)
      .def_readonly("checkpoint_name", &PyCheckpointResponse::checkpoint_name);

  py::class_<PySealResponse>(module, "_SealResponse")
      .def_readonly("source_segment_id", &PySealResponse::source_segment_id)
      .def_readonly("successor_segment_id", &PySealResponse::successor_segment_id)
      .def_readonly("sealed_segment_id", &PySealResponse::sealed_segment_id)
      .def_readonly("wal_cut", &PySealResponse::wal_cut)
      .def_readonly("sealed_rows", &PySealResponse::sealed_rows)
      .def_readonly("sealed_bytes", &PySealResponse::sealed_bytes)
      .def_readonly("manifest_generation", &PySealResponse::manifest_generation);

  py::class_<PyCompactResponse>(module, "_CompactResponse")
      .def_readonly("source_segment_ids", &PyCompactResponse::source_segment_ids)
      .def_readonly("compacted_segment_id", &PyCompactResponse::compacted_segment_id)
      .def_readonly("compacted_rows", &PyCompactResponse::compacted_rows)
      .def_readonly("input_bytes", &PyCompactResponse::input_bytes)
      .def_readonly("output_bytes", &PyCompactResponse::output_bytes)
      .def_readonly("manifest_generation", &PyCompactResponse::manifest_generation);

  py::class_<PyGcResponse>(module, "_GcResponse")
      .def_readonly("pending", &PyGcResponse::pending)
      .def_readonly("reclaimed", &PyGcResponse::reclaimed)
      .def_readonly("deferred", &PyGcResponse::deferred)
      .def_readonly("reclaimed_bytes", &PyGcResponse::reclaimed_bytes)
      .def_readonly("manifest_generation", &PyGcResponse::manifest_generation);

  py::class_<PyStatsResponse>(module, "_StatsResponse")
      .def_readonly("size", &PyStatsResponse::size)
      .def_readonly("accepted_count", &PyStatsResponse::accepted_count)
      .def_readonly("pending_count", &PyStatsResponse::pending_count)
      .def_readonly("searchable_bytes", &PyStatsResponse::searchable_bytes)
      .def_readonly("accepted_bytes", &PyStatsResponse::accepted_bytes)
      .def_readonly("searchable_vector_bytes", &PyStatsResponse::searchable_vector_bytes)
      .def_readonly("accepted_vector_bytes", &PyStatsResponse::accepted_vector_bytes)
      .def_readonly("pending_bytes", &PyStatsResponse::pending_bytes)
      .def_readonly("allocated_count", &PyStatsResponse::allocated_count)
      .def_readonly("tombstone_count", &PyStatsResponse::tombstone_count)
      .def_readonly("routing_generation", &PyStatsResponse::routing_generation)
      .def_readonly("visibility_watermark", &PyStatsResponse::visibility_watermark)
      .def_readonly("durable_watermark", &PyStatsResponse::durable_watermark)
      .def_readonly("metadata_epoch", &PyStatsResponse::metadata_epoch)
      .def_readonly("sealed_segments_count", &PyStatsResponse::sealed_segments_count)
      .def_readonly("gc_pending_count", &PyStatsResponse::gc_pending_count)
      .def_readonly("active_segment_algorithm", &PyStatsResponse::active_segment_algorithm)
      .def_readonly("compacted_bytes", &PyStatsResponse::compacted_bytes)
      .def_readonly("lifecycle", &PyStatsResponse::lifecycle);

  py::class_<PyOptionsResponse>(module, "_OptionsResponse")
      .def_readonly("root", &PyOptionsResponse::root)
      .def_readonly("read_only", &PyOptionsResponse::read_only)
      .def_readonly("dim", &PyOptionsResponse::dim)
      .def_readonly("metric", &PyOptionsResponse::metric)
      .def_readonly("dtype", &PyOptionsResponse::dtype)
      .def_readonly("index_type", &PyOptionsResponse::index_type)
      .def_readonly("quantization_type", &PyOptionsResponse::quantization_type)
      .def_readonly("build_threads", &PyOptionsResponse::build_threads)
      .def_readonly("max_neighbors", &PyOptionsResponse::max_neighbors)
      .def_readonly("ef_construction", &PyOptionsResponse::ef_construction)
      .def_readonly("implementation_key", &PyOptionsResponse::implementation_key)
      .def_readonly("engine_factory_key", &PyOptionsResponse::engine_factory_key)
      .def_readonly("active_algorithm", &PyOptionsResponse::active_algorithm)
      .def_readonly("auto_seal_rows", &PyOptionsResponse::auto_seal_rows);

  py::class_<PyCapabilitiesResponse>(module, "_CapabilitiesResponse")
      .def_readonly("index_types", &PyCapabilitiesResponse::index_types)
      .def_readonly("laser_enabled", &PyCapabilitiesResponse::laser_enabled)
      .def_readonly("laser_simd", &PyCapabilitiesResponse::laser_simd);
}

void register_capabilities(py::module_ &module,
                           bool laser_enabled,
                           std::optional<std::string> laser_simd) {
  module.def("capabilities", [laser_enabled, laser_simd = std::move(laser_simd)] {
    std::vector<std::string> index_types{"flat"};
    if (laser_enabled) {
      index_types.emplace_back("qg");
    }
    return PyCapabilitiesResponse{std::move(index_types), laser_enabled, laser_simd};
  });
}

}  // namespace alaya::python::collection_binding
