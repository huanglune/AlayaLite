// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file qg_scanner.hpp
 * @brief SIMD-accelerated neighbor scanning using RaBitQ quantization.
 *
 * Computes approximate distances to all neighbors of a node in a single
 * vectorized pass. Uses lookup table accumulation for inner product computation
 * between quantized query and binary-coded neighbor vectors.
 *
 * Distance formula: dist ≈ ||y||² + triple_x + factor_dq * width * result + factor_vq * vl + sqr_qr
 * where result comes from the fast scan accumulation of quantized codes.
 * Note: triple_x already includes ||x_r||^2 (residual dim norm) pre-added during build.
 */

#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>

#include "index/graph/laser/quantization/fastscan_impl.hpp"
#include "simd/laser_dispatch.hpp"

namespace alaya::laser {

namespace qg_scanner_detail {

using FusedScanBatchFn = void (*)(size_t,
                                  const uint8_t *,
                                  const uint8_t *,
                                  float,
                                  float,
                                  float,
                                  float,
                                  int32_t,
                                  const float *,
                                  const float *,
                                  const float *,
                                  float *);

inline void fused_scan_batch_generic(size_t padded_dim,
                                     const uint8_t *ALAYA_RESTRICT packed_code,
                                     const uint8_t *ALAYA_RESTRICT lut,
                                     float sqr_y,
                                     float vl,
                                     float width,
                                     float sqr_qr,
                                     int32_t sumq,
                                     const float *ALAYA_RESTRICT triple_x,
                                     const float *ALAYA_RESTRICT fac_dq,
                                     const float *ALAYA_RESTRICT fac_vq,
                                     float *ALAYA_RESTRICT appro_dist) {
  alignas(64) std::array<uint16_t, kBatchSize> accumulated{};
  simd::detail::accumulate_impl_generic(padded_dim, packed_code, lut, accumulated.data());
  for (size_t i = 0; i < kBatchSize; ++i) {
    const float result = static_cast<float>((static_cast<int32_t>(accumulated[i]) * 2) - sumq);
    appro_dist[i] = sqr_y + triple_x[i] + (fac_dq[i] * width * result) + (fac_vq[i] * vl) + sqr_qr;
  }
}

#ifdef ALAYA_ARCH_X86

template <bool kMatchAvx512Arithmetic>
ALAYA_TARGET_AVX2 inline void estimate_eight_avx2(__m128i accumulated_u16,
                                                  __m256i sumq,
                                                  __m256 sqr_y,
                                                  __m256 vl,
                                                  __m256 width,
                                                  __m256 sqr_qr,
                                                  const float *ALAYA_RESTRICT triple_x,
                                                  const float *ALAYA_RESTRICT fac_dq,
                                                  const float *ALAYA_RESTRICT fac_vq,
                                                  float *ALAYA_RESTRICT appro_dist) {
  __m256i accumulated_i32 = _mm256_cvtepu16_epi32(accumulated_u16);
  accumulated_i32 = _mm256_sub_epi32(_mm256_slli_epi32(accumulated_i32, 1), sumq);
  const __m256 accumulated = _mm256_cvtepi32_ps(accumulated_i32);

  #ifdef __FAST_MATH__
  const __m256 dq = _mm256_mul_ps(_mm256_loadu_ps(fac_dq), accumulated);
  if constexpr (kMatchAvx512Arithmetic) {
    // Under the production -Ofast flags, the legacy AVX-512 estimator emits
    // these two contractions. Keep its exact per-lane instruction order while
    // using 256-bit vectors: dq * width + (vq * vl + triple + norms).
    __m256 base = _mm256_add_ps(_mm256_loadu_ps(triple_x), sqr_y);
    base = _mm256_fmadd_ps(_mm256_loadu_ps(fac_vq), vl, base);
    base = _mm256_add_ps(base, sqr_qr);
    _mm256_storeu_ps(appro_dist, _mm256_fmadd_ps(dq, width, base));
  } else {
    // The legacy AVX2 estimator groups the two norms, then contracts the dq
    // width multiplication with the already-rounded vq term.
    __m256 base = _mm256_add_ps(sqr_y, sqr_qr);
    base = _mm256_add_ps(_mm256_loadu_ps(triple_x), base);
    const __m256 vq = _mm256_mul_ps(_mm256_loadu_ps(fac_vq), vl);
    const __m256 terms = _mm256_fmadd_ps(dq, width, vq);
    _mm256_storeu_ps(appro_dist, _mm256_add_ps(base, terms));
  }
  #else
  // Without -ffast-math the legacy kernels retain their source-level order.
  const __m256 dq = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(fac_dq), width), accumulated);
  __m256 base = _mm256_add_ps(_mm256_loadu_ps(triple_x), sqr_y);
  if constexpr (kMatchAvx512Arithmetic) {
    base = _mm256_fmadd_ps(_mm256_loadu_ps(fac_vq), vl, base);
    base = _mm256_add_ps(dq, base);
  } else {
    const __m256 vq = _mm256_mul_ps(_mm256_loadu_ps(fac_vq), vl);
    base = _mm256_add_ps(_mm256_add_ps(base, dq), vq);
  }
  _mm256_storeu_ps(appro_dist, _mm256_add_ps(base, sqr_qr));
  #endif
}

template <bool kMatchAvx512Arithmetic>
ALAYA_TARGET_AVX2 inline void fused_scan_batch_avx2(size_t padded_dim,
                                                    const uint8_t *ALAYA_RESTRICT packed_code,
                                                    const uint8_t *ALAYA_RESTRICT lut_table,
                                                    float sqr_y,
                                                    float vl,
                                                    float width,
                                                    float sqr_qr,
                                                    int32_t sumq,
                                                    const float *ALAYA_RESTRICT triple_x,
                                                    const float *ALAYA_RESTRICT fac_dq,
                                                    const float *ALAYA_RESTRICT fac_vq,
                                                    float *ALAYA_RESTRICT appro_dist) {
  const size_t code_length = padded_dim << 2;
  const __m256i low_mask = _mm256_set1_epi8(0x0f);
  __m256i accu0 = _mm256_setzero_si256();
  __m256i accu1 = _mm256_setzero_si256();
  __m256i accu2 = _mm256_setzero_si256();
  __m256i accu3 = _mm256_setzero_si256();

  for (size_t i = 0; i < code_length; i += 64) {
    __m256i codes = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&packed_code[i]));
    __m256i lut = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&lut_table[i]));
    __m256i lo = _mm256_and_si256(codes, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(codes, 4), low_mask);
    __m256i res_lo = _mm256_shuffle_epi8(lut, lo);
    __m256i res_hi = _mm256_shuffle_epi8(lut, hi);

    accu0 = _mm256_add_epi16(accu0, res_lo);
    accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
    accu2 = _mm256_add_epi16(accu2, res_hi);
    accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));

    codes = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&packed_code[i + 32]));
    lut = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&lut_table[i + 32]));
    lo = _mm256_and_si256(codes, low_mask);
    hi = _mm256_and_si256(_mm256_srli_epi16(codes, 4), low_mask);
    res_lo = _mm256_shuffle_epi8(lut, lo);
    res_hi = _mm256_shuffle_epi8(lut, hi);

    accu0 = _mm256_add_epi16(accu0, res_lo);
    accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
    accu2 = _mm256_add_epi16(accu2, res_hi);
    accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));
  }

  accu0 = _mm256_sub_epi16(accu0, _mm256_slli_epi16(accu1, 8));
  const __m256i accumulated_lo = _mm256_add_epi16(_mm256_permute2f128_si256(accu0, accu1, 0x21),
                                                  _mm256_blend_epi32(accu0, accu1, 0xF0));
  accu2 = _mm256_sub_epi16(accu2, _mm256_slli_epi16(accu3, 8));
  const __m256i accumulated_hi = _mm256_add_epi16(_mm256_permute2f128_si256(accu2, accu3, 0x21),
                                                  _mm256_blend_epi32(accu2, accu3, 0xF0));

  const __m256i sumq_simd = _mm256_set1_epi32(sumq);
  const __m256 sqr_y_simd = _mm256_set1_ps(sqr_y);
  const __m256 vl_simd = _mm256_set1_ps(vl);
  const __m256 width_simd = _mm256_set1_ps(width);
  const __m256 sqr_qr_simd = _mm256_set1_ps(sqr_qr);

  estimate_eight_avx2<kMatchAvx512Arithmetic>(_mm256_castsi256_si128(accumulated_lo),
                                              sumq_simd,
                                              sqr_y_simd,
                                              vl_simd,
                                              width_simd,
                                              sqr_qr_simd,
                                              triple_x,
                                              fac_dq,
                                              fac_vq,
                                              appro_dist);
  estimate_eight_avx2<kMatchAvx512Arithmetic>(_mm256_extracti128_si256(accumulated_lo, 1),
                                              sumq_simd,
                                              sqr_y_simd,
                                              vl_simd,
                                              width_simd,
                                              sqr_qr_simd,
                                              triple_x + 8,
                                              fac_dq + 8,
                                              fac_vq + 8,
                                              appro_dist + 8);
  estimate_eight_avx2<kMatchAvx512Arithmetic>(_mm256_castsi256_si128(accumulated_hi),
                                              sumq_simd,
                                              sqr_y_simd,
                                              vl_simd,
                                              width_simd,
                                              sqr_qr_simd,
                                              triple_x + 16,
                                              fac_dq + 16,
                                              fac_vq + 16,
                                              appro_dist + 16);
  estimate_eight_avx2<kMatchAvx512Arithmetic>(_mm256_extracti128_si256(accumulated_hi, 1),
                                              sumq_simd,
                                              sqr_y_simd,
                                              vl_simd,
                                              width_simd,
                                              sqr_qr_simd,
                                              triple_x + 24,
                                              fac_dq + 24,
                                              fac_vq + 24,
                                              appro_dist + 24);
}

#endif  // ALAYA_ARCH_X86

inline auto get_fused_scan_batch_func() -> FusedScanBatchFn {
#ifdef ALAYA_ARCH_X86
  switch (simd::get_laser_simd_level()) {
    case simd::LaserSimdLevel::kAvx512:
      // Zen4 executes AVX-512 as two 256-bit pumps. Use the AVX2 accumulator,
      // but preserve the legacy AVX-512 estimator's contracted arithmetic.
      return fused_scan_batch_avx2<true>;
    case simd::LaserSimdLevel::kAvx2:
      return fused_scan_batch_avx2<false>;
    case simd::LaserSimdLevel::kGeneric:
      return fused_scan_batch_generic;
  }
  throw std::runtime_error("unknown LASER SIMD level");
#else
  return fused_scan_batch_generic;
#endif
}

}  // namespace qg_scanner_detail

/**
 * @brief Scanner for computing approximate distances to all neighbors of a node.
 *
 * Processes quantized neighbor codes and computes distance estimates using
 * SIMD-accelerated lookup table accumulation followed by factor correction.
 */
class QGScanner {
 private:
  // Reference-path scratch upper bound: rows are packed at degree_bound edges
  // (R64 is the largest shipped config). The production fused path below has
  // no accumulator or converted-distance scratch.
  static constexpr size_t kMaxDegreeBound = 64;

  size_t padded_dim_ = 0;
  size_t degree_bound_ = 0;
  simd::AccumulateFn accumulate_ = nullptr;
  simd::ConvertAccumFn convert_ = nullptr;
  simd::AppDistFn compute_appro_dist_ = nullptr;
  qg_scanner_detail::FusedScanBatchFn fused_scan_batch_ = nullptr;

  void scan_neighbors_three_pass(float *ALAYA_RESTRICT appro_dist,
                                 const uint8_t *ALAYA_RESTRICT lut,
                                 float sqr_y,
                                 float vl,
                                 float width,
                                 float sqr_qr,
                                 int32_t sumq,
                                 const uint8_t *packed_code,
                                 const float *factor) const {
    alignas(64) std::array<uint16_t, kMaxDegreeBound> result;

    for (size_t i = 0; i < degree_bound_; i += kBatchSize) {
      accumulate_(padded_dim_, packed_code, lut, &result[i]);
      packed_code = &packed_code[padded_dim_ << 2];
    }

    alignas(64) std::array<float, kMaxDegreeBound> result_float;
    convert_(degree_bound_, result.data(), sumq, result_float.data());
    const float *triple_x = factor;
    const float *fac_dq = &triple_x[degree_bound_];
    const float *fac_vq = &fac_dq[degree_bound_];
    compute_appro_dist_(degree_bound_,
                        sqr_y,
                        width,
                        vl,
                        sqr_qr,
                        result_float.data(),
                        triple_x,
                        fac_dq,
                        fac_vq,
                        appro_dist);
  }

 public:
  QGScanner() = default;

  explicit QGScanner(size_t padded_dim, size_t degree_bound)
      : padded_dim_(padded_dim),
        degree_bound_(degree_bound),
        accumulate_(simd::get_accumulate_func()),
        convert_(simd::get_convert_func()),
        compute_appro_dist_(simd::get_appro_dist_func()),
        fused_scan_batch_(qg_scanner_detail::get_fused_scan_batch_func()) {
    if (degree_bound_ > kMaxDegreeBound) {
      throw std::invalid_argument("QGScanner: degree_bound exceeds kMaxDegreeBound scratch");
    }
  }

  void pack_lut(const uint8_t *ALAYA_RESTRICT byte_query, uint8_t *ALAYA_RESTRICT LUT) const {
    pack_lut_impl(padded_dim_, byte_query, LUT);
  }

  void scan_neighbors(float *ALAYA_RESTRICT appro_dist,
                      const uint8_t *ALAYA_RESTRICT lut,
                      float sqr_y,
                      float vl,
                      float width,
                      float sqr_qr,
                      int32_t sumq,
                      const uint8_t *packed_code,
                      const float *factor) const {
    const float *triple_x = factor;
    const float *fac_dq = &triple_x[degree_bound_];
    const float *fac_vq = &fac_dq[degree_bound_];
    for (size_t i = 0; i < degree_bound_; i += kBatchSize) {
      fused_scan_batch_(padded_dim_,
                        packed_code,
                        lut,
                        sqr_y,
                        vl,
                        width,
                        sqr_qr,
                        sumq,
                        triple_x + i,
                        fac_dq + i,
                        fac_vq + i,
                        appro_dist + i);
      packed_code = &packed_code[padded_dim_ << 2];
    }
  }

#ifdef ALAYA_LASER_TESTING
  // Test oracle only: the production hot path has no switch back to the old
  // accumulate -> convert -> estimate implementation.
  void scan_neighbors_three_pass_reference(float *ALAYA_RESTRICT appro_dist,
                                           const uint8_t *ALAYA_RESTRICT lut,
                                           float sqr_y,
                                           float vl,
                                           float width,
                                           float sqr_qr,
                                           int32_t sumq,
                                           const uint8_t *packed_code,
                                           const float *factor) const {
    scan_neighbors_three_pass(appro_dist, lut, sqr_y, vl, width, sqr_qr, sumq, packed_code, factor);
  }
#endif
};
}  // namespace alaya::laser
