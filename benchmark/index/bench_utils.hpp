/**
 * @file bench_utils.hpp
 * @brief Shared utilities for benchmark programs: percentile.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace alaya::bench {

/**
 * @brief Compute the p-th percentile from a vector of values.
 *
 * Sorts the input in-place and returns a linearly interpolated value.
 */
inline auto percentile(std::vector<double> &v, double p) -> double {
  std::sort(v.begin(), v.end());
  double idx = p / 100.0 * static_cast<double>(v.size() - 1);
  auto lo = static_cast<size_t>(idx);
  auto hi = lo + 1;
  if (hi >= v.size()) {
    return v.back();
  }
  double frac = idx - static_cast<double>(lo);
  return v[lo] * (1.0 - frac) + v[hi] * frac;
}

}  // namespace alaya::bench
