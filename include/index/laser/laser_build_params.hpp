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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include <thread>

namespace alaya {

struct LaserBuildParams {
  uint32_t main_dim_{0};
  uint32_t max_degree_{64};
  uint32_t ef_construction_{200};
  uint32_t ef_build_{128};
  float alpha_{1.2F};
  uint32_t num_medoids_{300};
  float pca_sample_ratio_{0.25F};
  uint32_t pca_sample_cap_{5000000};
  float medoid_sample_ratio_{0.1F};
  uint32_t medoid_sample_cap_{500000};
  uint32_t num_threads_{0};
  size_t max_memory_mb_{4096};
  bool keep_intermediates_{false};
  size_t vamana_max_memory_mb_{0};  ///< 0 = inherit from max_memory_mb_
  std::string external_vamana_;     ///< If set, use this Vamana file instead of building
  std::string external_rotator_;    ///< If set, load this rotator file instead of random init

  [[nodiscard]] auto resolved_vamana_memory_mb() const -> size_t {
    return vamana_max_memory_mb_ > 0 ? vamana_max_memory_mb_ : max_memory_mb_;
  }

  [[nodiscard]] static auto auto_main_dim(uint32_t full_dim) -> uint32_t {
    if (full_dim == 0) {
      throw std::invalid_argument("LaserBuildParams::auto_main_dim requires full_dim > 0");
    }

    uint32_t half_dim = std::max<uint32_t>(1, full_dim / 2);
    uint32_t candidate = 1;
    while ((candidate << 1U) > candidate && (candidate << 1U) <= half_dim) {
      candidate <<= 1U;
    }

    candidate = std::clamp<uint32_t>(candidate, 64U, 512U);

    uint32_t full_dim_floor_pow2 = 1;
    while ((full_dim_floor_pow2 << 1U) > full_dim_floor_pow2 &&
           (full_dim_floor_pow2 << 1U) <= full_dim) {
      full_dim_floor_pow2 <<= 1U;
    }
    return std::min(candidate, full_dim_floor_pow2);
  }

  [[nodiscard]] auto resolved_main_dim(uint32_t full_dim) const -> uint32_t {
    auto dim = main_dim_ == 0 ? auto_main_dim(full_dim) : main_dim_;
    if (dim == 0 || (dim & (dim - 1U)) != 0U) {
      throw std::invalid_argument("LASER main_dim must be a non-zero power of 2");
    }
    if (dim > full_dim) {
      throw std::invalid_argument("LASER main_dim cannot exceed the input dimension");
    }
    return dim;
  }

  [[nodiscard]] auto resolved_num_threads() const -> uint32_t {
    if (num_threads_ != 0) {
      return num_threads_;
    }
    auto n = std::thread::hardware_concurrency();
    return n == 0 ? 1U : n;
  }

  [[nodiscard]] auto resolved_pca_sample_count(uint32_t num_points) const -> uint32_t {
    if (num_points == 0) {
      return 0;
    }
    auto requested = static_cast<uint32_t>(static_cast<double>(num_points) *
                                           static_cast<double>(pca_sample_ratio_));
    auto capped = std::min(num_points, pca_sample_cap_);
    return std::clamp<uint32_t>(requested == 0 ? 1U : requested,
                                1U,
                                std::max<uint32_t>(1U, capped));
  }

  [[nodiscard]] auto resolved_medoid_sample_count(uint32_t num_points) const -> uint32_t {
    if (num_points == 0) {
      return 0;
    }
    auto requested = static_cast<uint32_t>(static_cast<double>(num_points) *
                                           static_cast<double>(medoid_sample_ratio_));
    auto capped = std::min(num_points, medoid_sample_cap_);
    return std::clamp<uint32_t>(requested == 0 ? 1U : requested,
                                1U,
                                std::max<uint32_t>(1U, capped));
  }

  [[nodiscard]] auto params_hash() const -> std::string {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(8) << main_dim_ << '|' << max_degree_ << '|'
           << ef_construction_ << '|' << ef_build_ << '|' << alpha_ << '|' << num_medoids_ << '|'
           << pca_sample_ratio_ << '|' << pca_sample_cap_ << '|' << medoid_sample_ratio_ << '|'
           << medoid_sample_cap_ << '|' << num_threads_ << '|' << max_memory_mb_ << '|'
           << keep_intermediates_ << '|' << vamana_max_memory_mb_;

    constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
    constexpr uint64_t kFnvPrime = 1099511628211ULL;
    uint64_t hash = kFnvOffset;
    for (unsigned char ch : stream.str()) {
      hash ^= static_cast<uint64_t>(ch);
      hash *= kFnvPrime;
    }

    std::ostringstream hex;
    hex << std::hex << std::setw(16) << std::setfill('0') << hash;
    return hex.str();
  }
};

}  // namespace alaya
