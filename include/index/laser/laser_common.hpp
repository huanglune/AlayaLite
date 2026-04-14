/**
 * @file laser_common.hpp
 * @brief Common type definitions and utilities for the library.
 *
 * Defines fundamental types used throughout the codebase:
 * - PID: Point/node identifier type (uint32_t)
 * - RowMatrix/ColMatrix: Eigen matrix wrappers for vector data
 * - DistFunc: Distance function type alias
 */

#pragma once

#include <cstdint>
#include <functional>

#include <Eigen/Dense>

namespace symqg {
#define RANDOM_QUERY_QUANTIZATION
#define QG_BQUERY 6
#define FORCE_INLINE inline __attribute__((always_inline))
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

using PID = uint32_t;                     // Point ID type for graph nodes
constexpr uint32_t kPidMax = 0xFFFFFFFF;  // Maximum valid PID value

constexpr float kCacheRatio = 0.15;  // Maximum ratio of nodes to cache in memory (15%)

template <typename T>
using RowMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using ColMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template <typename T>
using DistFunc = std::function<T(const T *, const T *, size_t)>;

}  // namespace symqg
