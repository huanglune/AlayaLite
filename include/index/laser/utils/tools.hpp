/**
 * @file tools.hpp
 * @brief General utility functions for memory alignment and random number generation.
 */

#pragma once

#include <ctime>
#include <random>
#include <thread>

namespace symqg {

/** @brief Thread-safe random integer generator in range [min, max]. */
template <typename T>
inline auto rand_integer(T min, T max) -> T {
  static thread_local std::mt19937 generator(
      std::random_device{}() +  // NOLINT(whitespace/braces)
      std::hash<std::thread::id>()(std::this_thread::get_id()));
  std::uniform_int_distribution<T> distribution(min, max);
  return distribution(generator);
}

constexpr auto div_round_up(size_t val, size_t div) -> size_t {
  return (val / div) + static_cast<size_t>((val % div) != 0);
}

constexpr auto round_up_to_multiple(size_t val, size_t multiple_of) -> size_t {
  return multiple_of * (div_round_up(val, multiple_of));
}

inline auto ceil_log2(size_t val) -> size_t {
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
inline auto total_threads() -> size_t {
  const auto kThreads = std::thread::hardware_concurrency();
  return kThreads == 0 ? 1 : kThreads;
}
}  // namespace symqg
