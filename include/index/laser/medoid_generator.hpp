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

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "utils/kmeans.hpp"
#include "utils/timer.hpp"
#include "utils/vector_file_reader.hpp"

namespace alaya {

class MedoidGenerator {
 public:
  struct Config {
    uint32_t num_medoids_{300};
    float sample_ratio_{0.1F};
    uint32_t sample_cap_{500000};
    uint32_t num_threads_{0};
    uint32_t random_seed_{42};
    size_t max_memory_mb_{4096};
  };

  struct Result {
    uint32_t num_points_{0};
    uint32_t dimension_{0};
    std::vector<uint32_t> medoid_ids_;
    std::vector<float> medoid_vectors_;
  };

  MedoidGenerator() = default;
  explicit MedoidGenerator(Config config) : config_(config) {}

  [[nodiscard]] auto generate(const std::filesystem::path &fbin_path,
                              const std::filesystem::path &output_prefix) const -> Result {
    FbinFileReader reader;
    reader.open(fbin_path.string());

    auto num_points = reader.num_vectors();
    auto dim = reader.dim();
    if (num_points == 0 || dim == 0) {
      throw std::invalid_argument("MedoidGenerator requires a non-empty PCA dataset");
    }

    auto sample_size = resolved_sample_size(num_points);
    auto medoid_count = std::min<uint32_t>(config_.num_medoids_, num_points);

    alaya::Timer timer;

    std::vector<uint32_t> sample_ids(num_points);
    std::iota(sample_ids.begin(), sample_ids.end(), 0U);
    std::mt19937 rng(config_.random_seed_);
    std::shuffle(sample_ids.begin(), sample_ids.end(), rng);
    sample_ids.resize(sample_size);
    std::sort(sample_ids.begin(), sample_ids.end());

    std::vector<float> sample(static_cast<size_t>(sample_size) * dim);
    reader.read_by_ids(sample_ids.data(), sample_size, sample.data());
    sample_ids.clear();
    sample_ids.shrink_to_fit();

    std::cout << "  [Medoid] sampling " << sample_size << " vectors: " << std::fixed
              << std::setprecision(1) << timer.elapsed_s() << " s" << '\n';
    timer.reset();

    KMeans<float> kmeans({.num_clusters_ = medoid_count,
                          .max_iter_ = 20,
                          .num_trials_ = 3,
                          .num_threads_ = config_.num_threads_});
    auto clustering = kmeans.fit(sample.data(), sample_size, dim);

    std::cout << "  [Medoid] KMeans fit (k=" << medoid_count << ", samples=" << sample_size
              << "): " << timer.elapsed_s() << " s" << '\n';
    timer.reset();

    sample.clear();
    sample.shrink_to_fit();

    auto result = find_nearest_points_blocked(reader,
                                              num_points,
                                              dim,
                                              clustering.centroids_.data(),
                                              medoid_count);

    std::cout << "  [Medoid] nearest neighbor search: " << timer.elapsed_s() << " s" << '\n';

    write_outputs(output_prefix, result);
    return result;
  }

 private:
  Config config_;

  /**
   * Scan data ONCE (point-major order) so each vector is loaded into cache only once.
   * Centroids (K * dim floats ~ 1 MB for K=300, dim=960) stay hot in L2 cache.
   * Thread-local best arrays are merged at the end.
   */
  [[nodiscard]] auto find_nearest_points_blocked(FbinFileReader &reader,
                                                 uint32_t num_points,
                                                 uint32_t dim,
                                                 const float *centroids,
                                                 uint32_t medoid_count) const -> Result {
    auto vec_bytes = static_cast<size_t>(dim) * sizeof(float);
    auto budget = config_.max_memory_mb_ * 1024UL * 1024UL * 9UL / 10UL;
    auto scan_block =
        std::max(1000U, std::min(num_points, static_cast<uint32_t>(budget / vec_bytes)));

    Result result;
    result.num_points_ = num_points;
    result.dimension_ = dim;
    result.medoid_ids_.resize(medoid_count);
    result.medoid_vectors_.resize(static_cast<size_t>(medoid_count) * dim);

    std::vector<float> best_dists(medoid_count, std::numeric_limits<float>::max());
    std::vector<uint32_t> best_ids(medoid_count, 0);

    std::vector<float> block(static_cast<size_t>(scan_block) * dim);
    auto thread_count =
        static_cast<int>(config_.num_threads_ == 0 ? omp_get_max_threads() : config_.num_threads_);

    for (uint32_t start = 0; start < num_points; start += scan_block) {
      auto count = std::min(scan_block, num_points - start);
      reader.read_sequential(start, count, block.data());

#pragma omp parallel num_threads(thread_count)
      {
        std::vector<float> local_best_dists(medoid_count, std::numeric_limits<float>::max());
        std::vector<uint32_t> local_best_ids(medoid_count, 0);

#pragma omp for schedule(static)
        for (uint32_t offset = 0; offset < count; ++offset) {
          auto point_id = start + offset;
          const auto *point = block.data() + static_cast<size_t>(offset) * dim;
          for (uint32_t c = 0; c < medoid_count; ++c) {
            auto d =
                KMeans<float>::compute_l2_sqr(centroids + static_cast<size_t>(c) * dim, point, dim);
            if (d < local_best_dists[c]) {
              local_best_dists[c] = d;
              local_best_ids[c] = point_id;
            }
          }
        }

#pragma omp critical
        {
          for (uint32_t c = 0; c < medoid_count; ++c) {
            if (local_best_dists[c] < best_dists[c]) {
              best_dists[c] = local_best_dists[c];
              best_ids[c] = local_best_ids[c];
            }
          }
        }
      }
    }

    std::vector<uint32_t> sorted_order(medoid_count);
    std::iota(sorted_order.begin(), sorted_order.end(), 0U);
    std::sort(sorted_order.begin(), sorted_order.end(), [&best_ids](uint32_t lhs, uint32_t rhs) {
      return best_ids[lhs] < best_ids[rhs];
    });

    std::vector<uint32_t> sorted_ids(medoid_count);
    for (uint32_t i = 0; i < medoid_count; ++i) {
      sorted_ids[i] = best_ids[sorted_order[i]];
    }

    std::vector<float> sorted_vectors(static_cast<size_t>(medoid_count) * dim);
    reader.read_by_ids(sorted_ids.data(), medoid_count, sorted_vectors.data());

    for (uint32_t i = 0; i < medoid_count; ++i) {
      auto out_idx = sorted_order[i];
      result.medoid_ids_[out_idx] = sorted_ids[i];
      std::copy_n(sorted_vectors.data() + static_cast<size_t>(i) * dim,
                  dim,
                  result.medoid_vectors_.data() + static_cast<size_t>(out_idx) * dim);
    }

    return result;
  }

  [[nodiscard]] auto resolved_sample_size(uint32_t num_points) const -> uint32_t {
    auto requested = static_cast<uint32_t>(static_cast<double>(num_points) *
                                           static_cast<double>(config_.sample_ratio_));
    auto capped = std::min(num_points, config_.sample_cap_);
    auto minimum = std::min<uint32_t>(std::max<uint32_t>(1U, config_.num_medoids_), num_points);
    return std::clamp<uint32_t>(requested == 0 ? minimum : requested,
                                minimum,
                                std::max<uint32_t>(minimum, capped));
  }

  static void write_outputs(const std::filesystem::path &output_prefix, const Result &result) {
    auto dir = output_prefix.parent_path();
    if (!dir.empty()) {
      std::filesystem::create_directories(dir);
    }

    auto ids_path = output_prefix.string() + "_medoids_indices";
    auto vectors_path = output_prefix.string() + "_medoids";

    std::ofstream ids_out(ids_path, std::ios::binary | std::ios::trunc);
    if (!ids_out) {
      throw std::runtime_error("Failed to create medoid index file: " + ids_path);
    }
    auto num_medoids = static_cast<int32_t>(result.medoid_ids_.size());
    int32_t columns = 1;
    ids_out.write(reinterpret_cast<const char *>(&num_medoids), sizeof(num_medoids));
    ids_out.write(reinterpret_cast<const char *>(&columns), sizeof(columns));
    ids_out.write(reinterpret_cast<const char *>(result.medoid_ids_.data()),
                  static_cast<std::streamsize>(result.medoid_ids_.size() * sizeof(int32_t)));

    std::ofstream vectors_out(vectors_path, std::ios::binary | std::ios::trunc);
    if (!vectors_out) {
      throw std::runtime_error("Failed to create medoid vector file: " + vectors_path);
    }
    auto dim = static_cast<int32_t>(result.dimension_);
    vectors_out.write(reinterpret_cast<const char *>(&num_medoids), sizeof(num_medoids));
    vectors_out.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
    vectors_out.write(reinterpret_cast<const char *>(result.medoid_vectors_.data()),
                      static_cast<std::streamsize>(result.medoid_vectors_.size() * sizeof(float)));
  }
};

}  // namespace alaya
