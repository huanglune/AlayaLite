// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "index/graph/hnsw/hnsw_segment.hpp"

namespace alaya::detail {

// Internal bridge for generic GraphSearchJob/GraphUpdateJob consumers that are
// migrated in later abstraction steps.  It deliberately lives outside the
// public HnswSegment header: no user-facing builder/Graph pairing remains.
template <typename SearchSpaceType, typename BuildSpaceType>
struct HnswSegmentBridge {
  using Segment = HnswSegment<SearchSpaceType, BuildSpaceType>;

  static auto graph(const Segment &segment) { return segment.graph_; }
  static auto search_space(const Segment &segment) { return segment.search_space_; }
  static auto build_space(const Segment &segment) { return segment.build_space_; }
};

}  // namespace alaya::detail
