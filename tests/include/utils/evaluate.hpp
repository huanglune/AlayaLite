// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <utility>
#include <vector>

#include "simd/distance_l2.hpp"

namespace alaya {

template <typename DataType = float, typename DistanceType = float, typename IDType = uint32_t>
auto find_exact_gt(const std::vector<DataType> &queries,
                   const std::vector<DataType> &data_view,
                   uint32_t dim,
                   uint32_t topk,
                   std::unordered_set<IDType> *deleted = nullptr) -> std::vector<IDType> {
  if (queries.empty() || data_view.empty() || queries.size() % dim != 0 ||
      data_view.size() % dim != 0) {
    return {};
  }
  auto query_num = queries.size() / dim;
  std::vector<IDType> res(topk * query_num, 0);

  for (IDType i = 0; i < query_num; i++) {
    std::vector<std::pair<IDType, DistanceType>> dists;
    for (uint32_t j = 0; j < data_view.size() / dim; j++) {
      if (deleted && deleted->find(j) != deleted->end()) {
        continue;
      }
      float dist = simd::l2_sqr<DataType, DistanceType>(queries.data() + (i * dim),
                                                        data_view.data() + (j * dim),
                                                        dim);
      dists.emplace_back(j, dist);
    }
    std::sort(dists.begin(), dists.end(), [](const auto &a, const auto &b) {
      return a.second < b.second;
    });
    for (uint32_t j = 0; j < topk; j++) {
      res[(i * topk) + j] = dists[j].first;
    }
  }
  return res;
}

template <typename IDType>
auto calc_recall(const IDType *res,
                 const IDType *gt,
                 uint32_t query_num,
                 uint32_t gt_dim,
                 uint32_t topk) -> float {
  uint32_t cnt = 0;
  for (uint32_t i = 0; i < query_num; i++) {
    for (uint32_t j = 0; j < topk; j++) {
      for (uint32_t k = 0; k < gt_dim; k++) {
        if (res[i * topk + j] == gt[i * gt_dim + k]) {
          cnt++;
          break;
        }
      }
    }
  }
  return static_cast<float>(cnt) / static_cast<float>(query_num * topk);
}

template <typename IDType>
auto calc_recall(const std::vector<std::vector<IDType>> &res,
                 const IDType *gt,
                 uint32_t query_num,
                 uint32_t gt_dim,
                 uint32_t topk) -> float {
  uint32_t cnt = 0;
  for (uint32_t i = 0; i < query_num; i++) {
    for (uint32_t j = 0; j < topk; j++) {
      for (uint32_t k = 0; k < gt_dim; k++) {
        if (res[i][j] == gt[i * gt_dim + k]) {
          cnt++;
          break;
        }
      }
    }
  }
  return static_cast<float>(cnt) / static_cast<float>(query_num * topk);
}

template <typename T>
auto horizontal_avg(const std::vector<std::vector<T>> &data) -> std::vector<T> {
  auto rows = data.size();
  auto cols = data[0].size();
  std::vector<T> avg(cols, 0);
  for (const auto &row : data) {
    for (std::size_t j = 0; j < cols; ++j) {
      avg[j] += row[j];
    }
  }
  for (std::size_t j = 0; j < cols; ++j) {
    avg[j] /= static_cast<T>(rows);
  }
  return avg;
}

}  // namespace alaya
