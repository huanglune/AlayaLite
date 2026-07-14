/*
 * Copyright 2025 VectorDB.NTU
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

// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file rabitq.hpp
 * @brief RaBitQ (Randomized Binary Quantization) encoding for vectors.
 *
 * RaBitQ encodes vectors as binary codes with correction factors, enabling
 * fast approximate distance computation using SIMD operations. Each neighbor
 * is encoded with:
 * - Binary code: Sign of each dimension after rotation (1 bit per dimension)
 * - Correction factors: triple_x, factor_dq, factor_vq for accurate estimation
 */

#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include "index/graph/laser/common.hpp"
#include "index/graph/laser/quantization/fastscan_impl.hpp"
#include "index/graph/laser/space/bitwise.hpp"
#include "space/quant/rabitq_core.hpp"

namespace alaya::laser {

static inline void rabitq_factors(const kernels::linalg::RowMajorMatrix<float> &rotated_data,
                                  const kernels::linalg::RowMajorMatrix<float> &rotated_centroid,
                                  const kernels::linalg::RowMajorMatrix<int> &bin_x,
                                  float *triple_x,
                                  float *factor_dq,
                                  float *factor_vq);

/**
 * @brief Computes RaBitQ binary codes and correction factors for neighbor vectors.
 *
 * @param rotated_data    Neighbor vectors after FHT rotation (modified: becomes residuals)
 * @param rotated_centroid Center vector (the node whose neighbors we're encoding)
 * @param packed_code     Output: packed binary codes for SIMD scanning
 * @param triple_x        Output: ||x||^2 + 2*x*x1/x0 factor
 * @param factor_dq       Output: factor for delta * ||q_r|| term
 * @param factor_vq       Output: factor for v_l * ||q_r|| term
 */
inline void rabitq_codes(kernels::linalg::RowMajorMatrix<float> &rotated_data,
                         const kernels::linalg::RowMajorMatrix<float> &rotated_centroid,
                         uint8_t *packed_code,
                         float *triple_x,
                         float *factor_dq,
                         float *factor_vq) {
  int64_t num_points = rotated_data.rows();
  int64_t dim = rotated_data.cols();

  // current dim should be a multiple of 64
  assert(dim % 64 == 0);

  // get residual
  for (int64_t i = 0; i < num_points; ++i) {
    rotated_data.row(i) = rotated_data.row(i) - rotated_centroid;
  }

  // binary representation
  kernels::linalg::RowMajorMatrix<int> bin_x(num_points, dim);
  for (int64_t i = 0; i < num_points; ++i) {
    for (int64_t j = 0; j < dim; ++j) {
      bin_x(i, j) = static_cast<int>(rotated_data(i, j) > 0);
    }
  }

  // change bin_x to uint64
  std::vector<uint64_t> binary(num_points * (dim / 64));
  space::pack_binary(bin_x.data(), binary.data(), num_points * dim);

  // compute codes of RaBitQ, store at packed_code
  pack_codes(dim, binary.data(), num_points, packed_code);

  // compute factors for RaBitQ
  rabitq_factors(rotated_data, rotated_centroid, bin_x, triple_x, factor_dq, factor_vq);
}

static inline void rabitq_factors(
    const kernels::linalg::RowMajorMatrix<float> &rotated_data_residual,
    const kernels::linalg::RowMajorMatrix<float> &rotated_centroid,
    const kernels::linalg::RowMajorMatrix<int> &bin_x,
    float *triple_x,
    float *factor_dq,
    float *factor_vq) {
  int64_t num_points = rotated_data_residual.rows();
  int64_t dim = rotated_data_residual.cols();

  float fac_norm = 1.F / std::sqrt(static_cast<float>(dim));

  for (int64_t j = 0; j < num_points; ++j) {
    const auto factors = RaBitQCore::laser_l2_factors(rotated_data_residual.row(j).data(),
                                                      rotated_centroid.data(),
                                                      bin_x.row(j).data(),
                                                      dim,
                                                      fac_norm);
    triple_x[j] = factors.base;
    factor_dq[j] = factors.signed_query_scale;
    factor_vq[j] = static_cast<float>(factors.signed_query_scale *
                                      (bin_x.row(j).sum() * 2 - static_cast<int>(dim)));
  }
}
}  // namespace alaya::laser
