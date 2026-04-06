/**
 * @file l2.hpp
 * @brief SIMD-optimized L2 (Euclidean) squared distance computation.
 *
 * Provides AVX512/AVX2 vectorized implementations for computing squared
 * Euclidean distance between vectors. Used for exact distance calculations
 * between query and candidate vectors during k-NN search.
 */

#pragma once

#include <immintrin.h>

#include <cstddef>

namespace symqg::space {

/** @brief Horizontal sum reduction for AVX2 __m256 vector. */
inline auto reduce_add_m256(__m256 x) -> float {
// NOLINTBEGIN(portability-simd-intrinsics)
    auto sumh = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    auto tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
    auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
    return _mm_cvtss_f32(tmp2);
// NOLINTEND(portability-simd-intrinsics)
}

/**
 * @brief Computes squared L2 distance between two vectors.
 * @param vec0 First vector
 * @param vec1 Second vector
 * @param dim Vector dimension
 * @return ||vec0 - vec1||^2
 */
inline auto l2_sqr(
    const float* __restrict__ vec0, const float* __restrict__ vec1, size_t dim
) -> float {
    float result = 0;
// NOLINTBEGIN(portability-simd-intrinsics)
#if defined(__AVX512F__)
    size_t mul16 = dim - (dim & 0b1111);
    auto sum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i < mul16; i += 16) {
        auto xxx = _mm512_loadu_ps(&vec0[i]);
        auto yyy = _mm512_loadu_ps(&vec1[i]);
        auto ttt = _mm512_sub_ps(xxx, yyy);
        sum = _mm512_fmadd_ps(ttt, ttt, sum);
    }
    result = _mm512_reduce_add_ps(sum);
    for (; i < dim; ++i) {
        float tmp = vec0[i] - vec1[i];
        result += tmp * tmp;
    }

#elif defined(__AVX2__)
    size_t mul8 = dim - (dim & 0b111);
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i < mul8; i += 8) {
        __m256 xx = _mm256_loadu_ps(&vec0[i]);
        __m256 yy = _mm256_loadu_ps(&vec1[i]);
        __m256 t = _mm256_sub_ps(xx, yy);
        sum = _mm256_fmadd_ps(t, t, sum);
    }
    result = reduce_add_m256(sum);
    for (; i < dim; ++i) {
        float tmp = vec0[i] - vec1[i];
        result += tmp * tmp;
    }

// NOLINTEND(portability-simd-intrinsics)

#else
    for (size_t i = 0; i < dim; ++i) {
        float tmp = vec0[i] - vec1[i];
        result += tmp * tmp;
    }
#endif
    return result;
}

/** @brief Computes squared L2 norm of a single vector: ||vec0||^2 */
inline auto l2_sqr_single(const float* __restrict__ vec0, size_t dim) -> float {
    float result = 0;
// NOLINTBEGIN(portability-simd-intrinsics)
#if defined(__AVX512F__)
    size_t mul16 = dim - (dim & 0b1111);
    auto sum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i < mul16; i += 16) {
        auto xxx = _mm512_loadu_ps(&vec0[i]);
        sum = _mm512_fmadd_ps(xxx, xxx, sum);
    }
    result = _mm512_reduce_add_ps(sum);
    for (; i < dim; ++i) {
        float tmp = vec0[i];
        result += tmp * tmp;
    }

// NOLINTEND(portability-simd-intrinsics)
#else
    for (size_t i = 0; i < dim; ++i) {
        float tmp = vec0[i];
        result += tmp * tmp;
    }
#endif
    return result;
}

}  // namespace symqg::space
