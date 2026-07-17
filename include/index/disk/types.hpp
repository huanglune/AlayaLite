// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "core/value_types.hpp"

namespace alaya::disk {

enum class DiskIndexType {
  Flat,
  Laser,
};

struct DiskSearchOptions {
  uint32_t top_k = 10;
  uint32_t ef = 100;
  uint32_t beam_width = 4;
  bool exact_rerank = true;
  // Segment admission contract (docs/design/segment-admission-contract.md):
  // a value-copied view, not an owner. `filter.payload` (when kind !=
  // none) must stay valid for the duration of the search/batch_search
  // call this DiskSearchOptions is passed to -- it does not outlive one
  // call. Default kind=none keeps every existing caller byte-identical.
  core::SegmentFilterView filter{};
};

// Distance contract by metric (smaller-is-better in all three):
//   L2  squared L2 distance between query and stored vector
//   IP  negative inner product, -Sum(q_i * v_i)
//   COS stored vectors and query are L2-normalized internally before
//       scoring; distance is the negative inner product of the
//       normalized pair, -Sum(qn_i * vn_i)
// label is the external label supplied at build time, never the
// internal row index.
struct DiskSearchHit {
  uint64_t label;
  float distance;
};

class SegmentSearcher {
 public:
  virtual ~SegmentSearcher() = default;

  virtual auto search(const float *query, const DiskSearchOptions &opts) const
      -> std::vector<DiskSearchHit> = 0;

  // Batch entry point. The default fans out to search() one query at a
  // time; engines with a native batch kernel (e.g. the resident-arena
  // Laser path) override it. `queries` is row-major, num_queries * dim().
  virtual auto batch_search(const float *queries,
                            uint32_t num_queries,
                            const DiskSearchOptions &opts) const
      -> std::vector<std::vector<DiskSearchHit>> {
    std::vector<std::vector<DiskSearchHit>> out;
    out.reserve(num_queries);
    for (uint32_t q = 0; q < num_queries; ++q) {
      out.push_back(search(queries + static_cast<size_t>(q) * dim(), opts));
    }
    return out;
  }

  virtual auto size() const -> uint64_t = 0;
  virtual auto dim() const -> uint32_t = 0;
  virtual auto type() const -> DiskIndexType = 0;
};

constexpr auto index_type_to_string(DiskIndexType t) noexcept -> std::string_view {
  switch (t) {
    case DiskIndexType::Flat:
      return "disk_flat";
    case DiskIndexType::Laser:
      return "disk_laser";
  }
  return {};
}

inline auto index_type_from_string(std::string_view s) -> DiskIndexType {
  if (s == "disk_flat") {
    return DiskIndexType::Flat;
  }
  if (s == "disk_laser") {
    return DiskIndexType::Laser;
  }
  throw std::invalid_argument(std::string("unknown disk index_type string: ") + std::string(s));
}

}  // namespace alaya::disk
