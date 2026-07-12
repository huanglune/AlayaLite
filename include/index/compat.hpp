// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <array>
#include <stdexcept>

#include "core/value_types.hpp"
#include "index/disk/types.hpp"
#include "index/index_type.hpp"
#include "utils/metric_type.hpp"

namespace alaya::index_compat {

struct LegacyDiskSearchExtension {
  core::VersionedStructHeader header{};
  std::uint32_t effort{};
  std::uint32_t beam_width{};
  std::uint64_t reserved[3]{};

  LegacyDiskSearchExtension() : header(core::current_struct_header<LegacyDiskSearchExtension>()) {}
};

class DiskSearchOptionsAdapter {
 public:
  DiskSearchOptionsAdapter(const disk::DiskSearchOptions &legacy, core::AlgorithmId algorithm_id)
      : payload_(), extension_(), options_(legacy.top_k) {
    payload_.effort = legacy.ef;
    payload_.beam_width = legacy.beam_width;
    extension_.algorithm_id = algorithm_id;
    extension_.payload = std::addressof(payload_);
    extension_.payload_size = sizeof(payload_);
    options_.rerank_policy =
        legacy.exact_rerank ? core::RerankPolicy::exact_required : core::RerankPolicy::disabled;
    options_.extensions = std::span<const core::AlgorithmSearchExtension>(&extension_, 1);
  }

  DiskSearchOptionsAdapter(const DiskSearchOptionsAdapter &) = delete;
  auto operator=(const DiskSearchOptionsAdapter &) -> DiskSearchOptionsAdapter & = delete;
  DiskSearchOptionsAdapter(DiskSearchOptionsAdapter &&) = delete;
  auto operator=(DiskSearchOptionsAdapter &&) -> DiskSearchOptionsAdapter & = delete;

  [[nodiscard]] auto options() const noexcept -> const core::SearchOptions & { return options_; }

 private:
  LegacyDiskSearchExtension payload_{};
  core::AlgorithmSearchExtension extension_{};
  core::SearchOptions options_{};
};

[[deprecated("migration-only adapter; use core::SearchOptions")]] inline auto
from_disk_search_options(const disk::DiskSearchOptions &options, core::AlgorithmId algorithm_id)
    -> DiskSearchOptionsAdapter {
  return DiskSearchOptionsAdapter(options, algorithm_id);
}

[[deprecated("migration-only adapter; use disk::DiskSearchOptions")]] inline auto
to_disk_search_options(const core::SearchOptions &options, core::AlgorithmId algorithm_id)
    -> disk::DiskSearchOptions {
  disk::DiskSearchOptions legacy{};
  legacy.top_k = static_cast<std::uint32_t>(options.top_k);
  legacy.exact_rerank = options.rerank_policy != core::RerankPolicy::disabled;
  for (const auto &extension : options.extensions) {
    if (extension.algorithm_id != algorithm_id || extension.payload == nullptr ||
        extension.payload_size < sizeof(LegacyDiskSearchExtension)) {
      continue;
    }
    const auto &payload = *static_cast<const LegacyDiskSearchExtension *>(extension.payload);
    legacy.ef = payload.effort;
    legacy.beam_width = payload.beam_width;
    break;
  }
  return legacy;
}

[[deprecated("migration-only adapter; use core::SearchHit")]] inline auto from_disk_search_hit(
    const disk::DiskSearchHit &hit,
    core::Metric metric) -> core::SearchHit {
  return {core::SegmentRowId(hit.label),
          hit.distance,
          core::ScoreKind::distance,
          metric,
          core::ResultFlag::approximate};
}

[[deprecated("migration-only adapter; use disk::DiskSearchHit")]] inline auto to_disk_search_hit(
    const core::SearchHit &hit) -> disk::DiskSearchHit {
  return {static_cast<std::uint64_t>(hit.row_id), hit.score};
}

[[deprecated("migration-only adapter; use Descriptor::algorithm_id")]] constexpr auto
from_index_type(IndexType type) noexcept -> core::AlgorithmId {
  switch (type) {
    case IndexType::FLAT:
      return core::algorithm::flat;
    case IndexType::HNSW:
      return core::algorithm::hnsw;
    case IndexType::NSG:
      return core::algorithm::nsg;
    case IndexType::FUSION:
      return core::algorithm::fusion;
    case IndexType::QG:
      return core::algorithm::qg;
  }
  return 0;
}

[[deprecated("migration-only adapter; use Descriptor::algorithm_id")]] constexpr auto
from_disk_index_type(disk::DiskIndexType type) noexcept -> core::AlgorithmId {
  switch (type) {
    case disk::DiskIndexType::Flat:
      return core::algorithm::flat;
    case disk::DiskIndexType::Vamana:
      return core::algorithm::vamana;
    case disk::DiskIndexType::Laser:
      return core::algorithm::laser;
  }
  return 0;
}

[[deprecated("migration-only adapter; use core::Metric")]] constexpr auto from_metric_type(
    MetricType metric) -> core::Metric {
  switch (metric) {
    case MetricType::L2:
      return core::Metric::l2;
    case MetricType::IP:
      return core::Metric::inner_product;
    case MetricType::COS:
      return core::Metric::cosine;
    case MetricType::NONE:
      break;
  }
  throw std::invalid_argument("MetricType::NONE has no core Metric equivalent");
}

[[deprecated("migration-only adapter; use MetricType")]] constexpr auto to_metric_type(
    core::Metric metric) noexcept -> MetricType {
  switch (metric) {
    case core::Metric::l2:
      return MetricType::L2;
    case core::Metric::inner_product:
      return MetricType::IP;
    case core::Metric::cosine:
      return MetricType::COS;
  }
  return MetricType::NONE;
}

}  // namespace alaya::index_compat
