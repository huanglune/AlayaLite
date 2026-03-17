/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <tuple>

using byte = std::byte;
using WorkerID = uint32_t;
using CpuID = uint32_t;

namespace alaya {

// ============================================================================
// MetricType - Distance metric type for vector search
// ============================================================================

// NOLINTBEGIN
enum class MetricType {
  L2,
  IP,
  COS,
  NONE,
};
// NOLINTEND

struct MetricMap {
  static constexpr std::array<std::tuple<std::string_view, MetricType>, 3> kStaticMap = {
      std::make_tuple("L2", MetricType::L2),
      std::make_tuple("IP", MetricType::IP),
      std::make_tuple("COS", MetricType::COS),
  };

  constexpr auto operator[](const std::string_view str) const -> MetricType {
    for (const auto &[key, val] : kStaticMap) {
      if (key == str) {
        return val;
      }
    }
    return MetricType::NONE;
  }

  constexpr auto operator[](const MetricType type) const -> std::string_view {
    for (const auto &[key, val] : kStaticMap) {
      if (val == type) {
        return key;
      }
    }
    return "NONE";
  }
};

inline constexpr MetricMap kMetricMap{};
static_assert(kMetricMap["L2"] == MetricType::L2);
static_assert(kMetricMap["IP"] == MetricType::IP);
static_assert(kMetricMap["COS"] == MetricType::COS);
static_assert(kMetricMap[MetricType::L2] == "L2");
static_assert(kMetricMap[MetricType::IP] == "IP");
static_assert(kMetricMap[MetricType::COS] == "COS");

// ============================================================================
// DiskDataType - Data type identifier for vector elements stored on disk
// ============================================================================

/**
 * @brief Data type identifier for vector elements stored on disk.
 *
 * Uses uint32_t as base type for disk alignment compatibility.
 */
// NOLINTNEXTLINE(performance-enum-size)
enum class DiskDataType : uint32_t {  // NOLINT
  kFloat32 = 0,                       ///< 32-bit floating point (float)
  kInt8 = 1,                          ///< 8-bit signed integer (int8_t)
  kUInt8 = 2,                         ///< 8-bit unsigned integer (uint8_t)
  kFloat16 = 3,                       ///< 16-bit floating point (half precision)
  kBFloat16 = 4                       ///< 16-bit brain floating point
};

}  // namespace alaya
