/**
 * @file qg_scanner.hpp
 * @brief SIMD-accelerated neighbor scanning using RaBitQ quantization.
 *
 * Zero-alloc refactored: scan_neighbors() now accepts LaserSearchContext&
 * and uses pre-allocated scan_result/scan_float buffers instead of local vectors.
 */

#pragma once

#include <immintrin.h>

#include <cstdint>

#include "index/laser/laser_search_context.hpp"
#include "index/laser/quantization/fastscan_impl.hpp"

namespace symqg {

/**
 * @brief Computes approximate distances using precomputed factors.
 */
static inline void appro_dist_impl(
    size_t num_points,
    float sqr_y,
    float width,
    float vl,
    float sqr_qr,
    const float* __restrict__ result,
    const float* __restrict__ triple_x,
    const float* __restrict__ fac_dq,
    const float* __restrict__ fac_vq,
    float* __restrict__ appro_dist
) {
// NOLINTBEGIN(portability-simd-intrinsics)
#if defined(__AVX512F__)
    const __m512 kSqrYSimd = _mm512_set1_ps(sqr_y);
    const __m512 kWidthSimd = _mm512_set1_ps(width);
    const __m512 kVlSimd = _mm512_set1_ps(vl);
    const __m512 kSqrQrSimd = _mm512_set1_ps(sqr_qr);

    for (size_t i = 0; i < num_points; i += 16) {
        __m512 result_simd = _mm512_loadu_ps(&result[i]);
        __m512 triple_x_simd = _mm512_loadu_ps(&triple_x[i]);
        __m512 fac_dq_simd = _mm512_loadu_ps(&fac_dq[i]);
        __m512 fac_vq_simd = _mm512_loadu_ps(&fac_vq[i]);

        triple_x_simd = _mm512_add_ps(triple_x_simd, kSqrYSimd);
        fac_dq_simd = _mm512_mul_ps(fac_dq_simd, kWidthSimd);
        fac_dq_simd = _mm512_mul_ps(fac_dq_simd, result_simd);
        fac_vq_simd = _mm512_fmadd_ps(fac_vq_simd, kVlSimd, triple_x_simd);

        triple_x_simd = _mm512_add_ps(fac_dq_simd, fac_vq_simd);
        triple_x_simd = _mm512_add_ps(triple_x_simd, kSqrQrSimd);
        _mm512_storeu_ps(&appro_dist[i], triple_x_simd);
    }
#elif defined(__AVX2__)
    const __m256 kSqrYSimd = _mm256_set1_ps(sqr_y);
    const __m256 kWidthSimd = _mm256_set1_ps(width);
    const __m256 kVlSimd = _mm256_set1_ps(vl);
    const __m256 kSqrQrSimd = _mm256_set1_ps(sqr_qr);

    for (size_t i = 0; i < num_points; i += 8) {
        __m256 result_simd = _mm256_loadu_ps(&result[i]);
        __m256 triple_x_simd = _mm256_loadu_ps(&triple_x[i]);
        __m256 fac_dq_simd = _mm256_loadu_ps(&fac_dq[i]);
        __m256 fac_vq_simd = _mm256_loadu_ps(&fac_vq[i]);

        triple_x_simd = _mm256_add_ps(triple_x_simd, kSqrYSimd);
        fac_dq_simd = _mm256_mul_ps(fac_dq_simd, kWidthSimd);
        fac_dq_simd = _mm256_mul_ps(fac_dq_simd, result_simd);
        fac_vq_simd = _mm256_mul_ps(fac_vq_simd, kVlSimd);

        triple_x_simd =
            _mm256_add_ps(_mm256_add_ps(triple_x_simd, fac_dq_simd), fac_vq_simd);
        triple_x_simd = _mm256_add_ps(triple_x_simd, kSqrQrSimd);
    }
// NOLINTEND(portability-simd-intrinsics)
#else
    std::cerr << "SIMD (AVX512 or AVX2) REQUIRED!\n";
    abort();
#endif
}

/**
 * @brief Scanner for computing approximate distances to all neighbors of a node.
 *
 * Refactored: scan_neighbors() uses pre-allocated buffers from LaserSearchContext
 * instead of allocating std::vector per call.
 */
class QGScanner {
   private:
    size_t padded_dim_ = 0;
    size_t degree_bound_ = 0;

   public:
    QGScanner() = default;

    explicit QGScanner(size_t padded_dim, size_t degree_bound)
        : padded_dim_(padded_dim), degree_bound_(degree_bound) {}

    void pack_lut(const uint8_t* __restrict__ byte_query, uint8_t* __restrict__ LUT) const {
        pack_lut_impl(padded_dim_, byte_query, LUT);
    }

    /**
     * @brief Scan neighbors using pre-allocated buffers from SearchContext.
     * Zero heap allocations on this path.
     */
    void scan_neighbors(
        float* __restrict__ appro_dist,
        const uint8_t* __restrict__ LUT,
        float sqr_y,
        float vl,
        float width,
        float sqr_qr,
        int32_t sumq,
        const uint8_t* packed_code,
        const float* factor,
        LaserSearchContext& ctx
    ) const {
        uint16_t* result = ctx.scan_result();
        float* result_float = ctx.scan_float();

        // Compute block by block
        for (size_t i = 0; i < degree_bound_; i += kBatchSize) {
            accumulate_impl(padded_dim_, packed_code, LUT, &result[i]);
            packed_code = &packed_code[padded_dim_ << 2];
        }

        // Cast to float and apply: 2 * result - sumq
// NOLINTBEGIN(portability-simd-intrinsics)
#if defined(__AVX512F__)
        const __m512i kQq = _mm512_set1_epi32(sumq);
        for (size_t i = 0; i < degree_bound_; i += 32) {
            __m256i i16a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&result[i]));
            __m256i i16b =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&result[i + 16]));
            __m512i i32a = _mm512_cvtepi16_epi32(i16a);
            __m512i i32b = _mm512_cvtepi16_epi32(i16b);

            i32a = _mm512_sub_epi32(_mm512_slli_epi32(i32a, 1), kQq);
            i32b = _mm512_sub_epi32(_mm512_slli_epi32(i32b, 1), kQq);
            __m512 f32a = _mm512_cvtepi32_ps(i32a);
            __m512 f32b = _mm512_cvtepi32_ps(i32b);

            _mm512_storeu_ps(&result_float[i], f32a);
            _mm512_storeu_ps(&result_float[i + 16], f32b);
        }
// NOLINTEND(portability-simd-intrinsics)
#else
        for (size_t i = 0; i < degree_bound_; ++i) {
            result_float[i] = static_cast<float>((static_cast<int>(result[i]) << 1) - sumq);
        }
#endif
        const float* triple_x = factor;
        const float* fac_dq = &triple_x[degree_bound_];
        const float* fac_vq = &fac_dq[degree_bound_];
        appro_dist_impl(
            degree_bound_, sqr_y, width, vl, sqr_qr,
            result_float, triple_x, fac_dq, fac_vq, appro_dist
        );
    }
};
}  // namespace symqg
