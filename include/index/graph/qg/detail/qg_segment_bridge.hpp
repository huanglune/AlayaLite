// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "index/graph/qg/qg_segment.hpp"

namespace alaya::detail {

// Temporary access for legacy Python hybrid/materialized-view plumbing. The
// Space accessor preserves those consumers; Segment-native code uses graph().
template <typename SpaceType>
struct QgSegmentBridge {
  using Segment = QgSegment<SpaceType>;

  static auto space(const Segment &segment) { return segment.space_; }
  static auto graph(const Segment &segment) { return segment.graph_; }
};

}  // namespace alaya::detail
