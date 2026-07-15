// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file tools.hpp
 * @brief General utility functions for memory alignment and random number generation.
 */

#pragma once

#include <cstddef>
#include <thread>

namespace alaya::laser {

constexpr size_t round_up_to_multiple(size_t val, size_t multiple_of) {
  return ((val + multiple_of - 1) / multiple_of) * multiple_of;
}

inline size_t ceil_log2(size_t val) {
  size_t res = 0;
  for (size_t i = 0; i < 31; ++i) {
    if ((1U << i) >= val) {
      res = i;
      break;
    }
  }
  return res;
}

/** @brief Returns hardware thread count (or 1 if unavailable). */
inline size_t total_threads() {
  const auto threads = std::thread::hardware_concurrency();
  return threads == 0 ? 1 : threads;
}
}  // namespace alaya::laser
