// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <stdexcept>

#include "core/value_types.hpp"
#include "index/disk/types.hpp"
#include "index/index_type.hpp"
#include "utils/metric_type.hpp"

namespace alaya::core::compat {

inline constexpr std::uint64_t kAlgorithmFlat = 1;
inline constexpr std::uint64_t kAlgorithmHnsw = 2;
inline constexpr std::uint64_t kAlgorithmNsg = 3;
inline constexpr std::uint64_t kAlgorithmFusion = 4;
inline constexpr std::uint64_t kAlgorithmQg = 5;
inline constexpr std::uint64_t kAlgorithmVamana = 6;
inline constexpr std::uint64_t kAlgorithmLaser = 7;

// Migration-only adapters. Remove after all step 3-5 consumers use core types.
[[deprecated("migration-only adapter; use alaya::core::SearchOptions")]] constexpr auto
from_disk_search_options(const disk::DiskSearchOptions &options) noexcept -> SearchOptions {
  return SearchOptions{options.top_k, options.ef, options.beam_width, options.exact_rerank,
                       nullptr, nullptr};
}

[[deprecated("migration-only adapter; use alaya::core::SearchHit")]] constexpr auto
from_disk_search_hit(const disk::DiskSearchHit &hit) noexcept -> SearchHit {
  return SearchHit{hit.label, hit.distance};
}

[[deprecated("migration-only adapter; use alaya::disk::DiskSearchOptions")]] constexpr auto
to_disk_search_options(const SearchOptions &options) noexcept -> disk::DiskSearchOptions {
  return disk::DiskSearchOptions{options.top_k, options.effort, options.beam_width,
                                 options.exact_rerank};
}

[[deprecated("migration-only adapter; use alaya::disk::DiskSearchHit")]] constexpr auto
to_disk_search_hit(const SearchHit &hit) noexcept -> disk::DiskSearchHit {
  return disk::DiskSearchHit{hit.id, hit.distance};
}

[[deprecated("migration-only adapter; use Descriptor::algorithm_id")]] constexpr auto
from_index_type(IndexType type) noexcept -> std::uint64_t {
  switch (type) {
    case IndexType::FLAT:
      return kAlgorithmFlat;
    case IndexType::HNSW:
      return kAlgorithmHnsw;
    case IndexType::NSG:
      return kAlgorithmNsg;
    case IndexType::FUSION:
      return kAlgorithmFusion;
    case IndexType::QG:
      return kAlgorithmQg;
  }
  return 0;
}

[[deprecated("migration-only adapter; use Descriptor::algorithm_id")]] constexpr auto
from_disk_index_type(disk::DiskIndexType type) noexcept -> std::uint64_t {
  switch (type) {
    case disk::DiskIndexType::Flat:
      return kAlgorithmFlat;
    case disk::DiskIndexType::Vamana:
      return kAlgorithmVamana;
    case disk::DiskIndexType::Laser:
      return kAlgorithmLaser;
  }
  return 0;
}

[[deprecated("migration-only adapter; use alaya::core::Metric")]] constexpr auto from_metric_type(
    MetricType metric) -> Metric {
  switch (metric) {
    case MetricType::L2:
      return Metric::l2;
    case MetricType::IP:
      return Metric::inner_product;
    case MetricType::COS:
      return Metric::cosine;
    case MetricType::NONE:
      break;
  }
  throw std::invalid_argument("MetricType::NONE has no core Metric equivalent");
}

[[deprecated("migration-only adapter; use alaya::MetricType")]] constexpr auto to_metric_type(
    Metric metric) noexcept -> MetricType {
  switch (metric) {
    case Metric::l2:
      return MetricType::L2;
    case Metric::inner_product:
      return MetricType::IP;
    case Metric::cosine:
      return MetricType::COS;
  }
  return MetricType::NONE;
}

}  // namespace alaya::core::compat
