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

#include <cstddef>
#include <cstdint>

namespace alaya {

/**
 * @brief Parameters for DiskANN index construction.
 *
 * These parameters control the behavior of the Vamana graph construction
 * algorithm.
 *
 * Key parameters:
 * - R (max_degree_): Maximum out-degree of each node. Higher values improve
 *   recall but increase memory and search time. Typical: 32-128.
 * - α (alpha_): Distance threshold multiplier for pruning in pass 2+. Controls
 *   the trade-off between graph connectivity and search efficiency. Typical: 1.2.
 * - α_first (alpha_first_pass_): Alpha for first pass. Strict pruning (1.0)
 *   builds k-NN like graph for basic connectivity.
 * - L (ef_construction_): Search list size during construction. Higher values
 *   improve graph quality but increase build time. Typical: 100-200.
 */
struct DiskANNBuildParams {
  uint32_t max_degree_{64};        ///< R: Maximum out-degree of each node
  float alpha_{1.2F};              ///< Alpha for pass 2+ (relaxed pruning, adds long-range edges)
  float alpha_first_pass_{1.0F};   ///< Alpha for pass 1 (strict pruning, builds k-NN connectivity)
  uint32_t ef_construction_{128};  ///< L: Search list size during construction
  uint32_t num_threads_{0};        ///< Number of threads (0 = hardware concurrency)
  uint32_t num_iterations_{2};     ///< Number of Vamana iterations (typically 2)
  size_t max_memory_mb_{4096};     ///< Memory budget for out-of-core partitioned build
  float sample_rate_{0.02F};       ///< KMeans sample rate for partitioning (fraction in (0,1])
  uint32_t overlap_factor_{2};     ///< Number of shard assignments per vector during partitioning

  DiskANNBuildParams() = default;

  /**
   * @brief Construct with essential parameters.
   *
   * @param r Maximum out-degree (R)
   * @param a Alpha parameter for pruning
   * @param l Search list size during construction (L)
   */
  DiskANNBuildParams(uint32_t r, float a, uint32_t l)
      : max_degree_(r), alpha_(a), ef_construction_(l) {}

  /**
   * @brief Builder pattern for fluent configuration.
   */
  auto set_max_degree(uint32_t r) -> DiskANNBuildParams & {
    max_degree_ = r;
    return *this;
  }

  auto set_alpha(float a) -> DiskANNBuildParams & {
    alpha_ = a;
    return *this;
  }

  auto set_alpha_first_pass(float a) -> DiskANNBuildParams & {
    alpha_first_pass_ = a;
    return *this;
  }

  auto set_ef_construction(uint32_t l) -> DiskANNBuildParams & {
    ef_construction_ = l;
    return *this;
  }

  auto set_num_threads(uint32_t n) -> DiskANNBuildParams & {
    num_threads_ = n;
    return *this;
  }

  auto set_num_iterations(uint32_t n) -> DiskANNBuildParams & {
    num_iterations_ = n;
    return *this;
  }

  auto set_max_memory_mb(size_t memory_mb) -> DiskANNBuildParams & {
    max_memory_mb_ = memory_mb;
    return *this;
  }

  auto set_sample_rate(float rate) -> DiskANNBuildParams & {
    sample_rate_ = rate;
    return *this;
  }

  auto set_overlap_factor(uint32_t factor) -> DiskANNBuildParams & {
    overlap_factor_ = factor;
    return *this;
  }
};

/**
 * @brief Parameters for DiskANN search operations.
 */
struct DiskANNSearchParams {
  uint32_t ef_search_{64};         ///< Search list size (L)
  uint32_t num_threads_{1};        ///< Number of threads for batch search
  uint32_t cache_capacity_{4096};  ///< Buffer pool capacity (number of 4KB pages)
  uint32_t beam_width_{4};         ///< Beam width for batched candidate expansion
  uint32_t pipeline_width_{
      64};  ///< Number of queries processed concurrently in pipelined batch search

  DiskANNSearchParams() = default;

  explicit DiskANNSearchParams(uint32_t ef) : ef_search_(ef) {}

  auto set_ef_search(uint32_t ef) -> DiskANNSearchParams & {
    ef_search_ = ef;
    return *this;
  }

  auto set_num_threads(uint32_t n) -> DiskANNSearchParams & {
    num_threads_ = n;
    return *this;
  }

  auto set_cache_capacity(uint32_t cap) -> DiskANNSearchParams & {
    cache_capacity_ = cap;
    return *this;
  }

  auto set_beam_width(uint32_t bw) -> DiskANNSearchParams & {
    beam_width_ = bw;
    return *this;
  }

  auto set_pipeline_width(uint32_t pw) -> DiskANNSearchParams & {
    pipeline_width_ = pw;
    return *this;
  }
};

/**
 * @brief Parameters for DiskANN insert operations.
 */
struct DiskANNInsertParams {
  uint32_t ef_construction_{128};  ///< Search budget for finding neighbors
  float alpha_{1.2F};              ///< Pruning alpha for RobustPrune
  uint32_t beam_width_{4};         ///< Beam width for disk-based search

  DiskANNInsertParams() = default;

  auto set_ef_construction(uint32_t ef) -> DiskANNInsertParams & {
    ef_construction_ = ef;
    return *this;
  }

  auto set_alpha(float a) -> DiskANNInsertParams & {
    alpha_ = a;
    return *this;
  }

  auto set_beam_width(uint32_t bw) -> DiskANNInsertParams & {
    beam_width_ = bw;
    return *this;
  }
};

}  // namespace alaya
