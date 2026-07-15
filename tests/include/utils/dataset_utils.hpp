// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils/evaluate.hpp"
#include "utils/io_utils.hpp"
#include "utils/locks.hpp"
#include "utils/test_paths.hpp"

namespace alaya {

// ---------------------------------------------------------------------------
// Dataset types
// ---------------------------------------------------------------------------

struct Dataset {
  std::string name;
  std::vector<float> data;
  std::vector<float> queries;
  std::vector<uint32_t> ground_truth;
  uint32_t data_num = 0;
  uint32_t query_num = 0;
  uint32_t dim = 0;
  uint32_t gt_dim = 0;
};

struct DatasetConfig {
  std::string name;
  std::filesystem::path dir;
  std::filesystem::path data_file;
  std::filesystem::path query_file;
  std::filesystem::path gt_file;
  std::string download_url;
  std::string archive_name = "data.tar.gz";
  int strip_components = 1;
  uint32_t max_data_num = 0;
  uint32_t max_query_num = 0;
};

// ---------------------------------------------------------------------------
// Dataset catalog — each returns a ready-to-use config
// ---------------------------------------------------------------------------

inline auto sift_small() -> DatasetConfig {
  auto dir = test::data_dir() / "siftsmall";
  return {
      .name = "siftsmall",
      .dir = dir,
      .data_file = dir / "siftsmall_base.fvecs",
      .query_file = dir / "siftsmall_query.fvecs",
      .gt_file = dir / "siftsmall_groundtruth.ivecs",
      .download_url = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
  };
}

inline auto sift_micro() -> DatasetConfig {
  auto cfg = sift_small();
  cfg.name = "siftmicro";
  cfg.max_data_num = 1000;
  cfg.max_query_num = 50;
  return cfg;
}

inline auto deep1m() -> DatasetConfig {
  auto dir = test::data_dir() / "deep1M";
  return {
      .name = "deep1M",
      .dir = dir,
      .data_file = dir / "deep1M_base.fvecs",
      .query_file = dir / "deep1M_query.fvecs",
      .gt_file = dir / "deep1M_groundtruth.ivecs",
      .download_url = "http://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/deep1M.tar.gz",
      .archive_name = "deep1M.tar.gz",
  };
}

inline auto t2i1m() -> DatasetConfig {
  auto dir = test::data_dir() / "t2i-1m";
  return {
      .name = "t2i-1m",
      .dir = dir,
      .data_file = dir / "base.fvecs",
      .query_file = dir / "query.fvecs",
      .gt_file = dir / "groundtruth.ivecs",
  };
}

// ---------------------------------------------------------------------------
// Dataset loader
// ---------------------------------------------------------------------------

namespace detail {

inline void ensure_downloaded(const DatasetConfig &cfg) {
  bool exists = std::filesystem::exists(cfg.data_file) &&
                std::filesystem::exists(cfg.query_file) &&
                std::filesystem::exists(cfg.gt_file);
  if (exists) return;

  if (cfg.download_url.empty()) {
    throw std::runtime_error("Dataset '" + cfg.name + "' not found at " +
                             cfg.dir.string() + " and no download URL provided.");
  }
  std::filesystem::create_directories(cfg.dir);
  auto archive = cfg.dir / cfg.archive_name;

  auto wget = "wget -q " + cfg.download_url + " -O " + archive.string();
  if (std::system(wget.c_str()) != 0) {
    throw std::runtime_error("Download failed: " + cfg.download_url);
  }

  auto tar = "tar -zxf " + archive.string() +
             " --strip-components=" + std::to_string(cfg.strip_components) +
             " -C " + cfg.dir.string();
  if (std::system(tar.c_str()) != 0) {
    throw std::runtime_error("Extract failed: " + archive.string());
  }
}

}  // namespace detail

inline auto load_dataset(const DatasetConfig &cfg) -> Dataset {
  std::filesystem::create_directories(cfg.dir.parent_path());

  auto lock_file = cfg.dir.parent_path() / (cfg.dir.filename().string() + ".lock");
  FileLock lock(lock_file);

  detail::ensure_downloaded(cfg);

  Dataset ds;
  ds.name = cfg.name;

  uint32_t data_dim = 0;
  uint32_t query_dim = 0;
  load_fvecs(cfg.data_file, ds.data, ds.data_num, data_dim);
  load_fvecs(cfg.query_file, ds.queries, ds.query_num, query_dim);
  load_ivecs(cfg.gt_file, ds.ground_truth, ds.query_num, ds.gt_dim);

  if (data_dim != query_dim) {
    throw std::runtime_error("Dimension mismatch: data_dim=" + std::to_string(data_dim) +
                             ", query_dim=" + std::to_string(query_dim));
  }
  ds.dim = data_dim;

  bool data_truncated = cfg.max_data_num > 0 && ds.data_num > cfg.max_data_num;
  bool query_truncated = cfg.max_query_num > 0 && ds.query_num > cfg.max_query_num;

  if (data_truncated) {
    ds.data_num = cfg.max_data_num;
    ds.data.resize(ds.data_num * ds.dim);
  }
  if (query_truncated) {
    ds.query_num = cfg.max_query_num;
    ds.queries.resize(ds.query_num * ds.dim);
  }

  if (data_truncated) {
    ds.ground_truth = find_exact_gt(ds.queries, ds.data, ds.dim, ds.gt_dim);
  } else if (query_truncated) {
    ds.ground_truth.resize(ds.query_num * ds.gt_dim);
  }

  return ds;
}

}  // namespace alaya
