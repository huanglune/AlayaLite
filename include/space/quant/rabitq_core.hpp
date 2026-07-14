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

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "core/value_types.hpp"
#include "utils/rabitq_utils/defines.hpp"

namespace alaya {

template <typename DataType>
struct RaBitQCoreFactors {
  DataType base;
  DataType signed_query_scale;
};

struct RaBitQCore {
  template <typename DataType>
  static inline auto memory_factors(const DataType *data,
                                    const DataType *centroid,
                                    size_t dim,
                                    int *sign_bits,
                                    const core::Metric metric) -> RaBitQCoreFactors<DataType> {
    kernels::linalg::ConstRowMajorArrayMap<DataType> data_arr(data, 1, dim);
    kernels::linalg::ConstRowMajorArrayMap<DataType> cent_arr(centroid, 1, dim);
    kernels::linalg::RowMajorArray<DataType> residual_arr = data_arr - cent_arr;
    DataType residual_l2_sqr = ::alaya::l2_sqr<DataType>(residual_arr.data(), dim);

    kernels::linalg::RowMajorArrayMap<int> bits(sign_bits, 1, static_cast<long>(dim));  // NOLINT
    bits = (residual_arr > 0).template cast<int>();

    DataType binary_offset = -((1 << 1) - 1) / 2.F;
    kernels::linalg::RowMajorArray<DataType> half_signed =
        bits.template cast<DataType>() + binary_offset;
    DataType centroid_dot_half_signed = dot_product<DataType>(centroid, half_signed.data(), dim);
    DataType residual_dot_half_signed =
        dot_product<DataType>(residual_arr.data(), half_signed.data(), dim);
    if (residual_dot_half_signed == 0) {
      // TODO(rabitq-format): unify this finite exact-zero policy with LASER in a format upgrade.
      residual_dot_half_signed = std::numeric_limits<DataType>::infinity();
    }

    if (metric == core::Metric::l2) {
      return {
          residual_l2_sqr +
              (2 * residual_l2_sqr * centroid_dot_half_signed / residual_dot_half_signed),
          -residual_l2_sqr / residual_dot_half_signed,
      };
    }

    auto residual_dot_centroid = dot_product<DataType>(residual_arr.data(), centroid, dim);
    return {
        1 - residual_dot_centroid +
            residual_l2_sqr * centroid_dot_half_signed / residual_dot_half_signed,
        -residual_l2_sqr / residual_dot_half_signed,
    };
  }

  static inline auto laser_l2_factors(const float *residual,
                                      const float *centroid,
                                      const int *sign_bits,
                                      int64_t dim,
                                      float fac_norm) -> RaBitQCoreFactors<float> {
    kernels::linalg::ConstRowMajorArrayMap<float> residual_arr(residual, 1, dim);
    kernels::linalg::ConstRowMajorArrayMap<float> centroid_arr(centroid, 1, dim);
    kernels::linalg::ConstRowMajorArrayMap<int> bits(sign_bits, 1, dim);
    kernels::linalg::RowMajorArray<float> signed_x = 2 * bits.cast<float>() - 1.F;

    float fac_x0 = (residual_arr * signed_x * fac_norm).sum();
    float x_rotated_norm = residual_arr.matrix().norm();
    float cur_x0 = fac_x0;
    fac_x0 = cur_x0 / x_rotated_norm;
    float fac_x1 = static_cast<float>((centroid_arr * signed_x).sum() * fac_norm);

    double cur_x = x_rotated_norm;
    double normalized_x0 = fac_x0;
    double normalized_x1 = fac_x1;
    long double x_x0 = static_cast<long double>(cur_x) / normalized_x0;

    // TODO(rabitq-format): unify the exact-zero residual policy in an explicit format upgrade.
    // This path deliberately retains the historical 0/0 operation (and its NaN bit behavior).
    return {
        static_cast<float>((cur_x * cur_x) + (2 * x_x0 * normalized_x1)),
        static_cast<float>(-2 * x_x0 * fac_norm),
    };
  }
};

}  // namespace alaya
