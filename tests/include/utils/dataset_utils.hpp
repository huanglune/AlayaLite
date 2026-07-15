// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "utils/evaluate.hpp"

namespace alaya::test {

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

struct DatasetOptions {
  uint32_t num_vectors = 1000;
  uint32_t num_queries = 50;
  uint32_t dim = 128;
  uint32_t gt_topk = 10;
  uint64_t seed = 42;
  std::string name = "random";
};

inline auto random_dataset(DatasetOptions opts = {}) -> Dataset {
  std::mt19937 rng(opts.seed);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

  Dataset ds;
  ds.name = opts.name;
  ds.data_num = opts.num_vectors;
  ds.query_num = opts.num_queries;
  ds.dim = opts.dim;
  ds.gt_dim = opts.gt_topk;

  ds.data.resize(static_cast<std::size_t>(ds.data_num) * ds.dim);
  for (auto &v : ds.data) v = dist(rng);

  ds.queries.resize(static_cast<std::size_t>(ds.query_num) * ds.dim);
  for (auto &v : ds.queries) v = dist(rng);

  ds.ground_truth = find_exact_gt(ds.queries, ds.data, ds.dim, ds.gt_dim);

  return ds;
}

inline auto random_dataset(uint32_t num_vectors, uint32_t dim) -> Dataset {
  return random_dataset({.num_vectors = num_vectors, .dim = dim});
}

}  // namespace alaya::test
