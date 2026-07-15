// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file l2.hpp
 * @brief SIMD-optimized L2 (Euclidean) squared distance computation.
 *
 * Provides AVX512/AVX2 vectorized implementations for computing squared
 * Euclidean distance between vectors. Used for exact distance calculations
 * between query and candidate vectors during k-NN search.
 */

#pragma once

#include <cstddef>

#include "platform/detect.hpp"
#include "simd/distance_l2.hpp"

namespace alaya::laser::space {

/**
 * @brief Computes squared L2 distance between two vectors.
 * @param vec0 First vector
 * @param vec1 Second vector
 * @param dim Vector dimension
 * @return ||vec0 - vec1||^2
 */
inline float l2_sqr(const float *ALAYA_RESTRICT vec0,
                    const float *ALAYA_RESTRICT vec1,
                    size_t dim) {
  return ::alaya::simd::l2_sqr<float, float>(vec0, vec1, dim);
}

}  // namespace alaya::laser::space
