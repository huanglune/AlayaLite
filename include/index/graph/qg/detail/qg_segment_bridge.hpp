// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "index/graph/qg/qg_segment.hpp"

namespace alaya::detail {

// Temporary access for legacy Python hybrid/materialized-view plumbing.  QG's
// graph is embedded in RaBitQSpace, so the bridge exposes only that owned space.
template <typename SpaceType>
struct QgSegmentBridge {
  using Segment = QgSegment<SpaceType>;

  static auto space(const Segment &segment) { return segment.space_; }
};

}  // namespace alaya::detail
