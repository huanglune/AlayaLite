// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "core/any_segment.hpp"

namespace alaya::internal::collection::detail {

// Collection graph targets use cosine's exact score domain by storing
// normalized float32 vectors and forwarding normalized queries to an inner
// negative-dot segment. These helpers live above the engine boundary so the
// same adapter can be reused by HNSW, NSG, and Fusion.
[[nodiscard]] inline auto l2_normalize_float_rows(std::span<float> values,
                                                  std::uint32_t dim,
                                                  core::OperationStage stage) -> core::Status {
  if (dim == 0 || values.size() % dim != 0) {
    return core::Status::error(core::StatusCode::invalid_argument,
                               stage,
                               core::StatusDetail::dimension_mismatch,
                               "L2 normalization input does not contain whole non-empty rows");
  }
  for (std::size_t offset = 0; offset < values.size(); offset += dim) {
    double squared_norm{};
    for (std::uint32_t column = 0; column < dim; ++column) {
      const auto value = static_cast<double>(values[offset + column]);
      squared_norm += value * value;
    }
    if (squared_norm == 0.0) {
      std::fill_n(values.begin() + static_cast<std::ptrdiff_t>(offset), dim, 0.0F);
      continue;
    }
    const auto inverse_norm = 1.0 / std::sqrt(squared_norm);
    for (std::uint32_t column = 0; column < dim; ++column) {
      values[offset + column] =
          static_cast<float>(static_cast<double>(values[offset + column]) * inverse_norm);
    }
  }
  return core::Status::success();
}

[[nodiscard]] inline auto l2_normalized_float_copy(const core::TypedTensorView &source,
                                                   std::uint32_t expected_dim,
                                                   core::OperationStage stage)
    -> core::Result<std::vector<float>> {
  auto status = core::validate_tensor(source, expected_dim, stage);
  if (!status.ok()) {
    return status;
  }
  if (source.scalar_type != core::ScalarType::float32) {
    return core::Status::error(core::StatusCode::not_supported,
                               stage,
                               core::StatusDetail::unsupported_scalar_type,
                               "L2 query normalization requires float32 tensors");
  }
  std::uint64_t element_count{};
  if (!core::checked_multiply(source.rows, source.dim, element_count) ||
      element_count > std::numeric_limits<std::size_t>::max()) {
    return core::Status::error(core::StatusCode::invalid_argument,
                               stage,
                               core::StatusDetail::arithmetic_overflow,
                               "L2 normalization element count is not representable");
  }
  try {
    std::vector<float> normalized(static_cast<std::size_t>(element_count));
    for (core::RowCount row = 0; row < source.rows; ++row) {
      const auto *input = source.row<float>(row);
      auto *output = normalized.data() + static_cast<std::ptrdiff_t>(row * source.dim);
      std::copy_n(input, source.dim, output);
    }
    status = l2_normalize_float_rows(normalized, source.dim, stage);
    if (!status.ok()) {
      return status;
    }
    return normalized;
  } catch (...) {
    return core::status_from_exception(stage);
  }
}

class L2NormalizedQuerySegment {
 public:
  explicit L2NormalizedQuerySegment(core::AnySegment inner) : inner_(std::move(inner)) {
    descriptor_ = inner_.descriptor();
    descriptor_.preprocessing = core::MetricPreprocessing::l2_normalized;
  }

  [[nodiscard]] auto descriptor() const noexcept -> core::Descriptor { return descriptor_; }

  [[nodiscard]] static auto operations() noexcept -> const core::AnySegmentOperationTable & {
    static const core::AnySegmentOperationTable value = [] {
      core::AnySegmentOperationTable table;
      table.table_size = sizeof(core::AnySegmentOperationTable);
      table.table_version = core::kOperationTableVersion;
      table.start_search = &start_search<false>;
      table.start_batch_search = &start_search<true>;
      table.save = &save;
      table.stats = &stats;
      return table;
    }();
    return value;
  }

 private:
  struct SearchLifetime {
    std::vector<float> normalized_queries{};
    core::SearchContext context{};
    std::shared_ptr<void> upstream_pin{};
  };

  template <bool Batch>
  static auto start_search(const std::shared_ptr<void> &instance,
                           core::SearchRequest request,
                           core::SearchCompletion completion) noexcept
      -> core::Result<core::OperationHandle> {
    try {
      if (request.context == nullptr) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::null_data,
                                   "normalized query adapter requires a search context");
      }
      std::uint64_t element_count{};
      std::uint64_t normalized_bytes{};
      if (!core::checked_multiply(request.queries.rows, request.queries.dim, element_count) ||
          !core::checked_multiply(element_count, sizeof(float), normalized_bytes)) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::arithmetic_overflow,
                                   "normalized query scratch size overflows uint64");
      }
      auto status = core::require_lease(request.context->query_scratch_lease,
                                        normalized_bytes,
                                        core::OperationStage::search,
                                        "normalized query scratch lease is too small");
      if (!status.ok()) {
        return status;
      }
      auto normalized = l2_normalized_float_copy(request.queries,
                                                 request.queries.dim,
                                                 core::OperationStage::validation);
      if (!normalized.ok()) {
        return normalized.status();
      }
      auto lifetime = std::make_shared<SearchLifetime>();
      lifetime->normalized_queries = std::move(normalized).value();
      lifetime->context = *request.context;
      if (lifetime->context.query_scratch_lease.available_bytes != core::kUnlimitedResource) {
        lifetime->context.query_scratch_lease.available_bytes -= normalized_bytes;
      }
      lifetime->upstream_pin = std::move(request.lifetime_pin);
      request.queries = core::TypedTensorView::contiguous(lifetime->normalized_queries.data(),
                                                          request.queries.rows,
                                                          request.queries.dim);
      request.context = std::addressof(lifetime->context);
      request.lifetime_pin = lifetime;

      const auto segment = std::static_pointer_cast<L2NormalizedQuerySegment>(instance);
      if constexpr (Batch) {
        return segment->inner_.start_batch_search(std::move(request), std::move(completion));
      }
      return segment->inner_.start_search(std::move(request), std::move(completion));
    } catch (...) {
      return core::status_from_exception(core::OperationStage::admission);
    }
  }

  static auto save(const std::shared_ptr<void> &instance,
                   core::ArtifactWriter &writer,
                   const core::SaveOptions &options,
                   core::ArtifactManifest &manifest) noexcept -> core::Status {
    try {
      return std::static_pointer_cast<L2NormalizedQuerySegment>(instance)->inner_.save(writer,
                                                                                       options,
                                                                                       manifest);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::save);
    }
  }

  static auto stats(const std::shared_ptr<void> &instance, core::SegmentStats &stats) noexcept
      -> core::Status {
    try {
      return std::static_pointer_cast<L2NormalizedQuerySegment>(instance)->inner_.stats(stats);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::stats);
    }
  }

  core::AnySegment inner_{};
  core::Descriptor descriptor_{};
};

[[nodiscard]] inline auto make_l2_normalized_query_segment(core::AnySegment inner)
    -> core::Result<core::AnySegment> {
  const auto descriptor = inner.descriptor();
  if (descriptor.metric != core::Metric::cosine ||
      descriptor.stored_scalar_type != core::ScalarType::float32 ||
      descriptor.preprocessing != core::MetricPreprocessing::none) {
    return core::Status::error(core::StatusCode::invalid_argument,
                               core::OperationStage::admission,
                               core::StatusDetail::malformed_struct,
                               "L2-normalized query adapter requires a raw float32 cosine segment");
  }
  const auto inner_capabilities = inner.capabilities();
  if (!inner_capabilities.supports(core::OperationCapability::search) ||
      !inner_capabilities.supports(core::OperationCapability::batch_search) ||
      !inner_capabilities.supports(core::OperationCapability::save) ||
      !inner_capabilities.supports(core::OperationCapability::stats)) {
    return core::Status::error(core::StatusCode::not_supported,
                               core::OperationStage::admission,
                               core::StatusDetail::operation_slot_absent,
                               "L2-normalized query adapter inner segment lacks required slots");
  }

  auto adapter = std::make_shared<L2NormalizedQuerySegment>(std::move(inner));
  const auto adapted_descriptor = adapter->descriptor();
  core::SegmentInstanceConfig config;
  config.readonly = true;
  config.concurrency = inner_capabilities.concurrency;
  config.concurrency.explicit_drain = false;
  return core::AnySegment::from_raw(std::move(adapter),
                                    std::addressof(L2NormalizedQuerySegment::operations()),
                                    adapted_descriptor,
                                    std::move(config));
}

}  // namespace alaya::internal::collection::detail
