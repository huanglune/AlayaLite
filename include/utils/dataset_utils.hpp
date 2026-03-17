/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <random>
#include <string>
#include <vector>
#include "utils/evaluate.hpp"
#include "utils/io_utils.hpp"
#include "utils/locks.hpp"
#include "utils/types.hpp"

namespace alaya {

/**
 * @brief Loaded dataset containing vectors, queries and ground truth.
 *
 * Usage:
 *   auto ds = load_dataset(sift_small("/path/to/data"));
 *   space->fit(ds.data_.data(), ds.data_num_);
 */
struct Dataset {
  std::string name_;
  std::vector<float> data_;
  std::vector<float> queries_;
  std::vector<uint32_t> ground_truth_;
  uint32_t data_num_ = 0;
  uint32_t query_num_ = 0;
  uint32_t dim_ = 0;
  uint32_t gt_dim_ = 0;
};

/**
 * @brief Configuration for loading or generating a dataset.
 */
struct DatasetConfig {
  std::string name_;

  // File-based dataset parameters
  std::filesystem::path dir_;
  std::filesystem::path data_file_;
  std::filesystem::path query_file_;
  std::filesystem::path gt_file_;
  std::string download_url_;
  std::string archive_name_ = "data.tar.gz";
  int strip_components_ = 1;

  // Random generation parameters (used when is_random_ is true)
  bool is_random_ = false;
  uint32_t random_data_num_ = 0;
  uint32_t random_query_num_ = 0;
  uint32_t random_dim_ = 0;
  uint32_t random_gt_topk_ = 100;
  uint32_t random_seed_ = 42;
  MetricType metric_ = MetricType::L2;
};

/**
 * @brief Create config for a random dataset (no external files needed).
 *
 * @param data_num Number of data vectors to generate
 * @param query_num Number of query vectors to generate
 * @param dim Vector dimension
 * @param gt_topk Number of ground truth neighbors per query (default: 100)
 * @param seed Random seed for reproducibility (default: 42)
 * @return DatasetConfig for random generation
 */
inline auto random_config(uint32_t data_num,
                          uint32_t query_num,
                          uint32_t dim,
                          uint32_t gt_topk = 100,
                          uint32_t seed = 42,
                          MetricType metric = MetricType::L2) -> DatasetConfig {
  return DatasetConfig{
      .name_ = "random",
      .is_random_ = true,
      .random_data_num_ = data_num,
      .random_query_num_ = query_num,
      .random_dim_ = dim,
      .random_gt_topk_ = gt_topk,
      .random_seed_ = seed,
      .metric_ = metric,
  };
}

/**
 * @brief Create config for SIFT small dataset (10K vectors, 128 dim).
 */
inline auto sift_small(const std::filesystem::path &data_dir) -> DatasetConfig {
  auto dir = data_dir / "siftsmall";
  return DatasetConfig{
      .name_ = "siftsmall",
      .dir_ = dir,
      .data_file_ = dir / "siftsmall_base.fvecs",
      .query_file_ = dir / "siftsmall_query.fvecs",
      .gt_file_ = dir / "siftsmall_groundtruth.ivecs",
      .download_url_ = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
  };
}

/**
 * @brief Create config for SIFT1M dataset (1M vectors, 128 dim).
 */
inline auto sift1m(const std::filesystem::path &data_dir) -> DatasetConfig {
  auto dir = data_dir / "sift";
  return DatasetConfig{
      .name_ = "sift1M",
      .dir_ = dir,
      .data_file_ = dir / "sift_base.fvecs",
      .query_file_ = dir / "sift_query.fvecs",
      .gt_file_ = dir / "sift_groundtruth.ivecs",
      .download_url_ = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
      .archive_name_ = "sift.tar.gz",
  };
}

/**
 * @brief Create config for DEEP1M dataset (1M vectors, 96 dim).
 */
inline auto deep1m(const std::filesystem::path &data_dir) -> DatasetConfig {
  auto dir = data_dir / "deep1M";
  return DatasetConfig{
      .name_ = "deep1M",
      .dir_ = dir,
      .data_file_ = dir / "deep1M_base.fvecs",
      .query_file_ = dir / "deep1M_query.fvecs",
      .gt_file_ = dir / "deep1M_groundtruth.ivecs",
      .download_url_ = "http://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/deep1M.tar.gz",
      .archive_name_ = "deep1M.tar.gz",
  };
}

/**
 * @brief Create config for DEEP10M dataset (10M vectors, 96 dim, angular/cosine).
 *
 * Converted from http://ann-benchmarks.com/deep-image-96-angular.hdf5
 * via: uv run scripts/hdf5_to_fvecs.py --input <url> --output data/deep10M
 */
inline auto deep10m(const std::filesystem::path &data_dir) -> DatasetConfig {
  auto dir = data_dir / "deep10M";
  return DatasetConfig{
      .name_ = "deep10M",
      .dir_ = dir,
      .data_file_ = dir / "deep10M_base.fvecs",
      .query_file_ = dir / "deep10M_query.fvecs",
      .gt_file_ = dir / "deep10M_groundtruth.ivecs",
  };
}

namespace detail {
inline auto generate_random_dataset(const DatasetConfig &config) -> Dataset {
  Dataset ds;
  ds.name_ = config.name_;
  ds.data_num_ = config.random_data_num_;
  ds.query_num_ = config.random_query_num_;
  ds.dim_ = config.random_dim_;
  ds.gt_dim_ = config.random_gt_topk_;

  std::mt19937 rng(config.random_seed_);
  std::uniform_real_distribution<float> dist(0.0F, 1.0F);

  ds.data_.resize(static_cast<size_t>(ds.data_num_) * ds.dim_);
  for (auto &val : ds.data_) {
    val = dist(rng);
  }

  ds.queries_.resize(static_cast<size_t>(ds.query_num_) * ds.dim_);
  for (auto &val : ds.queries_) {
    val = dist(rng);
  }

  ds.ground_truth_ = find_exact_gt<float, float, uint32_t>(ds.queries_,
                                                           ds.data_,
                                                           ds.dim_,
                                                           ds.gt_dim_,
                                                           nullptr,
                                                           config.metric_);

  return ds;
}

inline auto load_dataset(const DatasetConfig &config) -> Dataset {
  Dataset ds;
  ds.name_ = config.name_;

  // File-based dataset loading
  // Ensure lock directory exists before creating lock file
  auto lock_dir = config.dir_.parent_path();
  if (!std::filesystem::exists(lock_dir)) {
    std::filesystem::create_directories(lock_dir);
  }

  // Use file lock to prevent concurrent downloads
  // Lock based on directory name (not dataset name) to handle configs sharing the same dir
  auto lock_file = lock_dir / (config.dir_.filename().string() + ".lock");
  FileLock lock(lock_file);

  // Download if files don't exist (check again after acquiring lock)
  bool files_exist = std::filesystem::exists(config.data_file_) &&
                     std::filesystem::exists(config.query_file_) &&
                     std::filesystem::exists(config.gt_file_);
  if (!files_exist) {
    if (config.download_url_.empty()) {
      LOG_CRITICAL(
          "Dataset '{}' files not found and no download URL configured. "
          "Expected files:\n  {}\n  {}\n  {}",
          config.name_,
          config.data_file_.string(),
          config.query_file_.string(),
          config.gt_file_.string());
      exit(-1);
    }
    if (!std::filesystem::exists(config.dir_)) {
      std::filesystem::create_directories(config.dir_);
    }
    auto archive_path = config.dir_ / config.archive_name_;
    auto download_cmd = "wget " + config.download_url_ + " -O " + archive_path.string();
    auto extract_cmd = "tar -zxvf " + archive_path.string() +
                       " --strip-components=" + std::to_string(config.strip_components_) + " -C " +
                       config.dir_.string();
    [[maybe_unused]] int ret1 = std::system(download_cmd.c_str());
    [[maybe_unused]] int ret2 = std::system(extract_cmd.c_str());
  }

  uint32_t data_dim = 0;
  uint32_t query_dim = 0;
  load_fvecs(config.data_file_, ds.data_, ds.data_num_, data_dim);
  load_fvecs(config.query_file_, ds.queries_, ds.query_num_, query_dim);
  load_ivecs(config.gt_file_, ds.ground_truth_, ds.query_num_, ds.gt_dim_);

  if (data_dim != query_dim) {
    LOG_CRITICAL("Dimension mismatch: data_dim={}, query_dim={}", data_dim, query_dim);
    exit(-1);
  }
  ds.dim_ = data_dim;

  return ds;
}
}  // namespace detail

/**
 * @brief Load dataset from config. Downloads if needed, or generates random data.
 *
 * Uses file locking to prevent concurrent downloads when multiple tests run in parallel.
 *
 * Usage:
 *   // Load from file
 *   auto ds = load_dataset(sift_small("/data"));
 *
 *   // Generate random data
 *   auto ds = load_dataset(random_config(1000, 50, 128));
 */
inline auto load_dataset(const DatasetConfig &config) -> Dataset {
  // Handle random dataset generation
  if (config.is_random_) {
    return detail::generate_random_dataset(config);
  }
  return detail::load_dataset(config);
}

}  // namespace alaya
