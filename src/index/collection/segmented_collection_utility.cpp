// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/collection/segmented_collection.hpp"

namespace alaya::internal::collection {

[[nodiscard]] auto SegmentedCollection::materialize_record(const core::LogicalId &logical_id,
                                                           const VersionEntry &version,
                                                           Projection projection)
    -> core::Result<CollectionRecord> {
  CollectionRecord result;
  result.logical_id = logical_id;
  result.upsert_sequence = version.upsert_sequence;
  if (projection_contains(projection, Projection::vector)) {
    if (!version.payload.vector.has_value()) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::search,
                                 core::StatusDetail::operation_slot_absent,
                                 "requested vector projection is unavailable");
    }
    result.vector = version.payload.vector;
  }
  if (projection_contains(projection, Projection::metadata)) {
    result.metadata = version.payload.metadata;
  }
  if (projection_contains(projection, Projection::document)) {
    result.document = version.payload.document;
  }
  return result;
}

[[nodiscard]] auto SegmentedCollection::validate_segment_response(
    const core::SearchResponse &response,
    core::RowCount query_count,
    core::RowCount top_k) -> core::Status {
  if (response.query_count != query_count || response.offsets.empty() || response.offsets[0] != 0) {
    return malformed_engine_response();
  }
  for (core::RowCount query = 0; query < query_count; ++query) {
    const auto index = static_cast<std::size_t>(query);
    if (response.offsets[index + 1] < response.offsets[index] ||
        response.offsets[index + 1] - response.offsets[index] != response.valid_counts[index] ||
        response.valid_counts[index] > top_k ||
        response.offsets[index + 1] > response.hits.size()) {
      return malformed_engine_response();
    }
  }
  return core::Status::success();
}

[[nodiscard]] auto SegmentedCollection::exact_distance(const core::TypedTensorView &query,
                                                       const OwnedVector &vector,
                                                       core::Metric metric) -> core::Result<float> {
  if (query.rows != 1 || query.dim != vector.dim() || query.scalar_type != vector.scalar_type()) {
    return core::Status::error(core::StatusCode::invalid_argument,
                               core::OperationStage::search,
                               core::StatusDetail::dimension_mismatch,
                               "exact rerank inputs do not share a tensor schema");
  }
  float score{};
  switch (query.scalar_type) {
    case core::ScalarType::float32:
      score = exact_distance_typed<float>(query, vector, metric);
      break;
    case core::ScalarType::int8:
      score = exact_distance_typed<std::int8_t>(query, vector, metric);
      break;
    case core::ScalarType::uint8:
      score = exact_distance_typed<std::uint8_t>(query, vector, metric);
      break;
  }
  return score;
}
}  // namespace alaya::internal::collection
