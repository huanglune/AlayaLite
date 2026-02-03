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
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include "simd/distance_l2.hpp"

namespace alaya {

/**
 * @brief K-means clustering algorithm implementation.
 *
 * Supports K-means++ initialization and Lloyd's algorithm for iterative optimization.
 *
 * @tparam DataType The data type of input values (e.g., float).
 */
template <typename DataType = float>
class KMeans {
 public:
  /**
   * @brief Configuration parameters for K-means.
   */
  struct Config {
    uint32_t num_clusters_{256};  ///< Number of clusters (K)
    uint32_t max_iter_{20};       ///< Maximum iterations
    uint32_t num_trials_{3};      ///< Number of trials (best selected)
  };

  /**
   * @brief Result of K-means clustering.
   */
  struct Result {
    std::vector<DataType> centroids_;    ///< Cluster centroids (K × dim)
    std::vector<uint32_t> assignments_;  ///< Cluster assignment for each point
    float cost_{0.0F};                   ///< Final quantization cost
  };

 private:
  Config config_;

 public:
  KMeans() = default;
  ~KMeans() = default;

  KMeans(const KMeans &other) = default;
  auto operator=(const KMeans &other) -> KMeans & = default;
  KMeans(KMeans &&other) noexcept = default;
  auto operator=(KMeans &&other) noexcept -> KMeans & = default;

  /**
   * @brief Construct KMeans with specified configuration.
   *
   * @param config K-means configuration parameters
   */
  explicit KMeans(const Config &config) : config_(config) {}

  /**
   * @brief Construct KMeans with specified parameters.
   *
   * @param num_clusters Number of clusters (K)
   * @param max_iter Maximum iterations (default: 20)
   * @param num_trials Number of trials (default: 3)
   */
  explicit KMeans(uint32_t num_clusters, uint32_t max_iter = 20, uint32_t num_trials = 3)
      : config_{num_clusters, max_iter, num_trials} {}

  /**
   * @brief Get the configuration.
   */
  [[nodiscard]] auto config() const -> const Config & { return config_; }

  /**
   * @brief Set the configuration.
   */
  void set_config(const Config &config) { config_ = config; }

  /**
   * @brief Run K-means clustering on the input data.
   *
   * @param data Pointer to input data (num_points × dim)
   * @param num_points Number of data points
   * @param dim Dimension of each point
   * @return Clustering result containing centroids, assignments, and cost
   */
  auto fit(const DataType *data, size_t num_points, uint32_t dim) -> Result {
    Result best_result;
    best_result.centroids_.resize(static_cast<size_t>(config_.num_clusters_) * dim);
    best_result.cost_ = std::numeric_limits<float>::max();

    std::vector<DataType> trial_centroids(static_cast<size_t>(config_.num_clusters_) * dim);

    for (uint32_t trial = 0; trial < config_.num_trials_; ++trial) {
      // Initialize centroids using K-means++
      init_centroids(data, num_points, dim, trial_centroids.data(), trial);

      // Run K-means iterations
      std::vector<uint32_t> assignments(num_points);
      float cost = iterate(data, num_points, dim, trial_centroids.data(), assignments);

      if (cost < best_result.cost_) {
        best_result.cost_ = cost;
        std::memcpy(best_result.centroids_.data(),
                    trial_centroids.data(),
                    trial_centroids.size() * sizeof(DataType));
        best_result.assignments_ = std::move(assignments);
      }
    }

    return best_result;
  }

  /**
   * @brief Run K-means clustering with externally provided centroid storage.
   *
   * This method allows reusing existing centroid storage, useful for PQ subspaces.
   *
   * @param data Pointer to input data (num_points × dim)
   * @param num_points Number of data points
   * @param dim Dimension of each point
   * @param centroids Output centroid storage (num_clusters × dim)
   * @return Final quantization cost
   */
  auto fit(const DataType *data, size_t num_points, uint32_t dim, DataType *centroids) -> float {
    float best_cost = std::numeric_limits<float>::max();
    std::vector<DataType> best_centroids(static_cast<size_t>(config_.num_clusters_) * dim);

    for (uint32_t trial = 0; trial < config_.num_trials_; ++trial) {
      // Initialize centroids using K-means++
      init_centroids(data, num_points, dim, centroids, trial);

      // Run K-means iterations
      std::vector<uint32_t> assignments(num_points);
      float cost = iterate(data, num_points, dim, centroids, assignments);

      if (cost < best_cost) {
        best_cost = cost;
        std::memcpy(best_centroids.data(),
                    centroids,
                    static_cast<size_t>(config_.num_clusters_) * dim * sizeof(DataType));
      }
    }

    // Copy best centroids back
    std::memcpy(centroids,
                best_centroids.data(),
                static_cast<size_t>(config_.num_clusters_) * dim * sizeof(DataType));
    return best_cost;
  }

  /**
   * @brief Find the nearest centroid for a single point.
   *
   * @param point Pointer to the point (dim dimensions)
   * @param centroids Pointer to centroids (num_clusters × dim)
   * @param dim Dimension of each point
   * @return Index of the nearest centroid
   */
  [[nodiscard]] auto find_nearest(const DataType *point,
                                  const DataType *centroids,
                                  uint32_t dim) const -> uint32_t {
    float min_dist = std::numeric_limits<float>::max();
    uint32_t best_idx = 0;

    for (uint32_t k = 0; k < config_.num_clusters_; ++k) {
      float dist = compute_l2_sqr(point, centroids + static_cast<size_t>(k) * dim, dim);
      if (dist < min_dist) {
        min_dist = dist;
        best_idx = k;
      }
    }
    return best_idx;
  }

  /**
   * @brief Compute squared L2 distance between two vectors using SIMD.
   */
  static auto compute_l2_sqr(const DataType *a, const DataType *b, uint32_t len) -> float {
    return simd::l2_sqr(a, b, len);
  }

 private:
  /**
   * @brief Initialize centroids using K-means++ algorithm.
   */
  void init_centroids(const DataType *data,
                      size_t num_points,
                      uint32_t dim,
                      DataType *centroids,
                      uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, num_points - 1);

    // Choose first centroid randomly
    size_t first_idx = dist(rng);
    std::memcpy(centroids, data + first_idx * dim, dim * sizeof(DataType));

    // Distance from each point to nearest centroid
    std::vector<float> min_dists(num_points, std::numeric_limits<float>::max());

    // Choose remaining centroids with probability proportional to distance²
    for (uint32_t k = 1; k < config_.num_clusters_; ++k) {
      // Update distances to include new centroid
      const DataType *last_centroid = centroids + static_cast<size_t>(k - 1) * dim;
      float total_dist = 0.0F;

      for (size_t i = 0; i < num_points; ++i) {
        float d = compute_l2_sqr(data + i * dim, last_centroid, dim);
        min_dists[i] = std::min(min_dists[i], d);
        total_dist += min_dists[i];
      }

      // Sample next centroid
      std::uniform_real_distribution<float> uniform(0.0F, total_dist);
      float threshold = uniform(rng);
      float cumsum = 0.0F;
      size_t chosen_idx = 0;

      for (size_t i = 0; i < num_points; ++i) {
        cumsum += min_dists[i];
        if (cumsum >= threshold) {
          chosen_idx = i;
          break;
        }
      }

      std::memcpy(centroids + static_cast<size_t>(k) * dim,
                  data + chosen_idx * dim,
                  dim * sizeof(DataType));
    }
  }

  /**
   * @brief Run K-means Lloyd iterations.
   *
   * @return Final quantization cost (sum of squared distances)
   */
  auto iterate(const DataType *data,
               size_t num_points,
               uint32_t dim,
               DataType *centroids,
               std::vector<uint32_t> &assignments) -> float {
    std::vector<uint32_t> counts(config_.num_clusters_);
    std::vector<double> new_centroids(static_cast<size_t>(config_.num_clusters_) * dim);

    float cost = 0.0F;

    for (uint32_t iter = 0; iter < config_.max_iter_; ++iter) {
      // Assignment step
      cost = 0.0F;
      for (size_t i = 0; i < num_points; ++i) {
        const DataType *vec = data + i * dim;
        float min_dist = std::numeric_limits<float>::max();
        uint32_t best_k = 0;

        for (uint32_t k = 0; k < config_.num_clusters_; ++k) {
          float d = compute_l2_sqr(vec, centroids + static_cast<size_t>(k) * dim, dim);
          if (d < min_dist) {
            min_dist = d;
            best_k = k;
          }
        }
        assignments[i] = best_k;
        cost += min_dist;
      }

      // Update step
      std::ranges::fill(counts, 0);
      std::ranges::fill(new_centroids, 0.0);

      for (size_t i = 0; i < num_points; ++i) {
        uint32_t k = assignments[i];
        counts[k]++;
        for (uint32_t d = 0; d < dim; ++d) {
          new_centroids[static_cast<size_t>(k) * dim + d] += static_cast<double>(data[i * dim + d]);
        }
      }

      // Compute new centroids
      for (uint32_t k = 0; k < config_.num_clusters_; ++k) {
        if (counts[k] > 0) {
          for (uint32_t d = 0; d < dim; ++d) {
            centroids[static_cast<size_t>(k) * dim + d] =
                static_cast<DataType>(new_centroids[static_cast<size_t>(k) * dim + d] / counts[k]);
          }
        }
      }
    }

    return cost;
  }
};

}  // namespace alaya
