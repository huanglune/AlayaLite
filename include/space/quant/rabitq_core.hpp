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
#include "space/quant/rabitq/defines.hpp"

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

    // Inner-product / cosine branch (metric != l2; shared by both since
    // Collection today never actually routes cosine into QG -- it rejects and
    // falls back to flat, see tests/collection/collection_qg_seal_test.cpp:
    // 509-523 -- but this space-level formula has to stay correct for any
    // future or direct caller that feeds pre-normalized cosine vectors too).
    //
    // Derivation: g_add = ip_sqr(query, centroid) = -<q,c> is computed exactly
    // by RaBitQSpace::QueryComputer (rabitq_space.hpp), and the fastscan LUT
    // estimates <half_signed, q_rot> (q rotated, *not* q-c -- unlike the L2
    // branch above, IP does not need the q-c expansion). Substituting the same
    // one-factor estimator <r,q> ~= K * <half_signed,q> that the L2 branch
    // above relies on (K = residual_l2_sqr / residual_dot_half_signed) into
    // est = base + rescale * <half_signed,q_rot> + g_add, with base and
    // rescale as returned below, reduces algebraically to the *exact* identity
    //   est(q, o) = 1 - <q, o>
    // The leading "1" is a constant that does not depend on the candidate o
    // (or on q) at all: it shifts every candidate's estimate by the same
    // amount under a fixed centroid c. Nearest-neighbor ranking only ever
    // compares estimates against each other for one fixed query, so this
    // constant is invisible to the ranking -- minimizing 1-<q,o> is exactly
    // equivalent to maximizing <q,o>, for *any* ||o||, not just unit vectors.
    // This is NOT an implicit "||o||=1" assumption, despite resembling half
    // the squared L2 distance between unit vectors (1-<q,o> = ||q-o||^2 / 2
    // when ||q||=||o||=1) -- that resemblance is exactly why cosine (unit
    // vectors) can reuse this same branch.
    //
    // Do not "fix" this by replacing the literal 1 with a data-driven norm
    // term such as dot_product(data,data,dim) (||o||^2): that changes the
    // target to ||o||^2 - <q,o>, which is NOT order-preserving across
    // candidates with different norms and would corrupt inner_product ranking
    // for non-unit data. (This was seriously considered and rejected during
    // the U4-preflight IP audit -- tests/space/rabitq_space_test.cpp's
    // RaBitQCoreTest.InnerProductBranchLocksToOneMinusDot pins this exact
    // formula shape via a q=c construction where the K-estimator's own
    // approximation error is algebraically zero, so a "||o||=1 bug" and a
    // "candidate-independent constant" are exactly distinguishable.)
    // Calibrating the *value* to the true -<q,o> would need K=0 here instead
    // of K=1, but nothing downstream needs that: search only needs order, and
    // QG re-ranks returned candidates with an exact score before it reaches
    // the caller (index/graph/laser/qg/qg_segment.hpp's response path).
    //
    // K=1 (this literal) is the upstream RaBitQ-Library convention, not
    // something introduced by this port: the reference implementation's
    // one_bit_code_with_factor(), METRIC_IP branch (baselines/RaBitQ-Library/
    // include/rabitqlib/quantization/rabitq_impl.hpp, cloned read-only
    // alongside this repo) computes the byte-for-byte identical
    // `f_add = 1 - dot_product(residual, centroid) + ...`.
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
