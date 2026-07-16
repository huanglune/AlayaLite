// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cmath>
#include <cstdint>

namespace alaya {

/**
 * @brief The unified structure for a neighbor.
 *
 * @tparam IDType The data type for storing IDs is determined by the number of
 vectors that need to be indexed, with the default type being uint64_t.
 */
template <typename IDType = uint64_t, typename DistanceType = float>
struct Neighbor {
  IDType id_;              ///< The id of the current point.
  DistanceType distance_;  ///< The distance between the query point and the current point.
  bool flag_;              ///< The flag identify the current point is visited or not.

  Neighbor() = default;
  Neighbor(IDType id, DistanceType distance, bool f = false)
      : id_(id), distance_(distance), flag_(f) {}

  friend auto operator<(const Neighbor &lhs, const Neighbor &rhs) -> bool {
    return lhs.distance_ < rhs.distance_ || (lhs.distance_ == rhs.distance_ && lhs.id_ < rhs.id_);
  }

  friend auto operator>(const Neighbor &lhs, const Neighbor &rhs) -> bool { return !(lhs < rhs); }
};

template <typename IDType = uint64_t, typename DistanceType = float>
struct Node {
  IDType id_;
  DistanceType distance_;

  Node() = default;
  Node(IDType id, DistanceType distance) : id_(id), distance_(distance) {}

  auto operator<(const Node &other) const -> bool { return distance_ < other.distance_; }
};

}  // namespace alaya
