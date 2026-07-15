// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils/evaluate.hpp"
#include "utils/io_utils.hpp"
#include "utils/locks.hpp"
#include "utils/test_paths.hpp"

namespace alaya {

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

struct DatasetConfig {
  std::string name_;
  std::filesystem::path dir_;
  std::filesystem::path data_file_;
  std::filesystem::path query_file_;
  std::filesystem::path gt_file_;
  std::string download_url_;
  std::string archive_name_ = "data.tar.gz";
  int strip_components_ = 1;
  uint32_t max_data_num_ = 0;
  uint32_t max_query_num_ = 0;
};

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

inline auto sift_small() -> DatasetConfig { return sift_small(test::data_dir()); }

inline auto sift_micro(const std::filesystem::path &data_dir) -> DatasetConfig {
  auto dir = data_dir / "siftsmall";
  return DatasetConfig{
      .name_ = "siftmicro",
      .dir_ = dir,
      .data_file_ = dir / "siftsmall_base.fvecs",
      .query_file_ = dir / "siftsmall_query.fvecs",
      .gt_file_ = dir / "siftsmall_groundtruth.ivecs",
      .download_url_ = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
      .max_data_num_ = 1000,
      .max_query_num_ = 50,
  };
}

inline auto sift_micro() -> DatasetConfig { return sift_micro(test::data_dir()); }

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

inline auto deep1m() -> DatasetConfig { return deep1m(test::data_dir()); }

inline auto t2i1m(const std::filesystem::path &data_dir) -> DatasetConfig {
  auto dir = data_dir / "t2i-1m";
  return DatasetConfig{
      .name_ = "t2i-1m",
      .dir_ = dir,
      .data_file_ = dir / "base.fvecs",
      .query_file_ = dir / "query.fvecs",
      .gt_file_ = dir / "groundtruth.ivecs",
  };
}

inline auto t2i1m() -> DatasetConfig { return t2i1m(test::data_dir()); }

inline auto load_dataset(const DatasetConfig &config) -> Dataset {
  auto lock_dir = config.dir_.parent_path();
  if (!std::filesystem::exists(lock_dir)) {
    std::filesystem::create_directories(lock_dir);
  }

  auto lock_file = lock_dir / (config.dir_.filename().string() + ".lock");
  FileLock lock(lock_file);

  bool files_exist = std::filesystem::exists(config.data_file_) &&
                     std::filesystem::exists(config.query_file_) &&
                     std::filesystem::exists(config.gt_file_);
  if (!files_exist) {
    if (config.download_url_.empty()) {
      throw std::runtime_error("Dataset '" + config.name_ + "' not found at " +
                               config.dir_.string() + " and no download URL provided.");
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

  Dataset ds;
  ds.name_ = config.name_;

  uint32_t data_dim = 0;
  uint32_t query_dim = 0;
  load_fvecs(config.data_file_, ds.data_, ds.data_num_, data_dim);
  load_fvecs(config.query_file_, ds.queries_, ds.query_num_, query_dim);
  load_ivecs(config.gt_file_, ds.ground_truth_, ds.query_num_, ds.gt_dim_);

  if (data_dim != query_dim) {
    throw std::runtime_error("Dimension mismatch: data_dim=" + std::to_string(data_dim) +
                             ", query_dim=" + std::to_string(query_dim));
  }
  ds.dim_ = data_dim;

  bool data_truncated = config.max_data_num_ > 0 && ds.data_num_ > config.max_data_num_;
  bool query_truncated = config.max_query_num_ > 0 && ds.query_num_ > config.max_query_num_;

  if (data_truncated) {
    ds.data_num_ = config.max_data_num_;
    ds.data_.resize(ds.data_num_ * ds.dim_);
  }

  if (query_truncated) {
    ds.query_num_ = config.max_query_num_;
    ds.queries_.resize(ds.query_num_ * ds.dim_);
  }

  if (data_truncated) {
    ds.ground_truth_ = find_exact_gt(ds.queries_, ds.data_, ds.dim_, ds.gt_dim_);
  } else if (query_truncated) {
    ds.ground_truth_.resize(ds.query_num_ * ds.gt_dim_);
  }

  return ds;
}

}  // namespace alaya
