// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// Runtime SIMD dispatch for the memory RaBitQ fastscan/rotator/lut hot path.
//
// This mirrors simd/laser_dispatch.hpp's function-multi-versioning pattern:
// function-level ALAYA_TARGET_* attributes (platform/detect.hpp) generate every
// ISA variant into the same translation unit regardless of the TU's baseline
// compile flags, simd::get_cpu_features() detects the host once per process,
// and a `static const` function pointer picks the winning variant the first
// time each kernel's getter runs. This keeps a single wheel build (baseline
// -mavx2) emitting and running AVX-512 code on hosts that support it, instead
// of gating on the compiler's baseline via #if defined(__AVX512F__).
//
// Deliberately independent from simd/laser_dispatch.hpp (not included, not
// referenced): memory RaBitQ (this file) and LASER's on-disk QG path are
// separate module boundaries, and this dispatcher must keep working when
// LASER is not built into the target at all.
//
// Cross-referenced against the official RaBitQ-Library's own dispatch split
// (baselines/RaBitQ-Library include/rabitqlib/simd/{fastscan,rotator}_dispatch.hpp)
// for the rotator/lut function-boundary choices below; the dispatch mechanism
// itself (single-TU target-attribute multi-versioning vs. their per-ISA
// translation units) follows this repo's own LASER pattern per the execution
// manifest, not the reference library's mechanism.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "core/log.hpp"
#include "kernels/linalg/types.hpp"
#include "platform/detect.hpp"
#include "simd/cpu_features.hpp"
#include "simd/fastscan.hpp"

namespace alaya::rabitq_simd {

using ::alaya::simd::fastscan::kBatchSize;

enum class RabitqSimdLevel : std::uint8_t { kGeneric, kAvx2, kAvx512 };

inline auto get_rabitq_simd_name(RabitqSimdLevel level) -> const char * {
  switch (level) {
    case RabitqSimdLevel::kGeneric:
      return "generic";
    case RabitqSimdLevel::kAvx2:
      return "avx2";
    case RabitqSimdLevel::kAvx512:
      return "avx512";
  }
  ALAYA_UNREACHABLE;
}

inline auto detect_rabitq_simd_level() -> RabitqSimdLevel {
#ifdef ALAYA_ARCH_X86
  const auto &features = ::alaya::simd::get_cpu_features();
  if (features.avx512f_ && features.avx512bw_) {
    return RabitqSimdLevel::kAvx512;
  }
  if (features.avx2_ && features.fma_) {
    return RabitqSimdLevel::kAvx2;
  }
  // Unlike LASER (which requires -mavx2 -mfma as its x86 build baseline and
  // therefore treats a missing AVX2 as a misconfiguration), the memory RaBitQ
  // path has always shipped a real, portable scalar fallback for non-AVX2 x86
  // hosts and non-x86 architectures alike. Fall through to kGeneric instead of
  // throwing.
#endif
  return RabitqSimdLevel::kGeneric;
}

inline auto get_rabitq_simd_level() -> RabitqSimdLevel {
  static const RabitqSimdLevel kLevel = [] {
    const RabitqSimdLevel level = detect_rabitq_simd_level();
    LOG_INFO("rabitq_simd={}", get_rabitq_simd_name(level));
    return level;
  }();
  return kLevel;
}

inline auto get_rabitq_simd_name() -> const char * {
  return get_rabitq_simd_name(get_rabitq_simd_level());
}

// Three-tier selector: used only by accumulate(), the single kernel that
// already had a real AVX2 tier before this refactor.
template <typename Fn>
inline auto select_rabitq_simd(Fn generic_fn, Fn avx2_fn, Fn avx512_fn) -> Fn {
  switch (get_rabitq_simd_level()) {
    case RabitqSimdLevel::kGeneric:
      return generic_fn;
    case RabitqSimdLevel::kAvx2:
      return avx2_fn;
    case RabitqSimdLevel::kAvx512:
      return avx512_fn;
  }
  ALAYA_UNREACHABLE;
}

// Two-tier selector: the other five kernels only ever had an AVX-512 branch
// and a portable fallback (no AVX2 branch existed pre-refactor). Scope is
// mechanical translation of the existing #if ladder, not adding new SIMD
// tiers, so AVX2-only hosts intentionally fall back to the generic path here,
// exactly as they did before.
template <typename Fn>
inline auto select_rabitq_simd_avx512_or_generic(Fn generic_fn, Fn avx512_fn) -> Fn {
  return get_rabitq_simd_level() == RabitqSimdLevel::kAvx512 ? avx512_fn : generic_fn;
}

// ============================================================================
// 1. accumulate — reuses simd::fastscan's own three already-dispatched-and-
//    tested kernels directly (include/simd/fastscan.hpp) rather than
//    reimplementing them here. Differential fuzz coverage for these three
//    already exists in tests/laser/simd_dispatch_test.cpp.
// ============================================================================

using AccumulateFn = void (*)(size_t, const uint8_t *, const uint8_t *, uint16_t *);

inline auto get_accumulate_func() -> AccumulateFn {
#ifdef ALAYA_ARCH_X86
  static const AccumulateFn kFunc =
      select_rabitq_simd<AccumulateFn>(::alaya::simd::fastscan::accumulate_generic,
                                       ::alaya::simd::fastscan::accumulate_avx2,
                                       ::alaya::simd::fastscan::accumulate_avx512);
#else
  static const AccumulateFn kFunc = ::alaya::simd::fastscan::accumulate_generic;
#endif
  return kFunc;
}

// ============================================================================
// 2. estimate_distances — float-only (static_assert'd at the call site in
//    fastscan.hpp); the AVX-512 branch below is the existing #if defined
//    (__AVX512F__) body lifted verbatim into a target-attributed function.
// ============================================================================

using EstimateDistancesFn =
    void (*)(const uint16_t *, const float *, const float *, float, float, float, float *);

namespace detail {

inline void estimate_distances_generic(const uint16_t *ALAYA_RESTRICT nth_segments,
                                       const float *ALAYA_RESTRICT f_add,
                                       const float *ALAYA_RESTRICT f_rescale,
                                       float g_add,
                                       float lut_delta,
                                       float lut_bias,
                                       float *ALAYA_RESTRICT result) {
  for (size_t off = 0; off < kBatchSize; ++off) {
    const auto inner = static_cast<float>(nth_segments[off]) * lut_delta + lut_bias;
    result[off] = f_rescale[off] * inner + f_add[off] + g_add;
  }
}

#ifdef ALAYA_ARCH_X86
ALAYA_TARGET_AVX512_BW
inline void estimate_distances_avx512(const uint16_t *ALAYA_RESTRICT nth_segments,
                                      const float *ALAYA_RESTRICT f_add,
                                      const float *ALAYA_RESTRICT f_rescale,
                                      float g_add,
                                      float lut_delta,
                                      float lut_bias,
                                      float *ALAYA_RESTRICT result) {
  const __m512 v_delta = _mm512_set1_ps(lut_delta);
  const __m512 v_bias = _mm512_set1_ps(lut_bias);
  const __m512 v_gadd = _mm512_set1_ps(g_add);

  for (size_t off = 0; off < kBatchSize; off += 16) {
    const __m256i nth_u16 =
        _mm256_load_si256(reinterpret_cast<const __m256i *>(nth_segments + off));
    const __m512 nth_f = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(nth_u16));

    const __m512 f_add_v = _mm512_loadu_ps(f_add + off);
    const __m512 f_rescale_v = _mm512_loadu_ps(f_rescale + off);

    const __m512 inner = _mm512_fmadd_ps(v_delta, nth_f, v_bias);
    const __m512 est = _mm512_fmadd_ps(f_rescale_v, inner, _mm512_add_ps(f_add_v, v_gadd));

    _mm512_storeu_ps(result + off, est);
  }
}
#endif

}  // namespace detail

inline auto get_estimate_distances_func() -> EstimateDistancesFn {
#ifdef ALAYA_ARCH_X86
  static const EstimateDistancesFn kFunc =
      select_rabitq_simd_avx512_or_generic<EstimateDistancesFn>(detail::estimate_distances_generic,
                                                                detail::estimate_distances_avx512);
#else
  static const EstimateDistancesFn kFunc = detail::estimate_distances_generic;
#endif
  return kFunc;
}

// ============================================================================
// 3. accumulate_and_estimate_distances — the fused hot kernel memqg actually
//    calls per candidate node (rabitq_space.hpp QueryComputer::batch_est_dist).
//    Float-only. The pre-refactor generic branch composed the (separately
//    dispatched) accumulate() and estimate_distances() rather than having its
//    own scalar body; we preserve that exactly, including its side effect of
//    picking up accumulate()'s own AVX2 tier on AVX2-only hosts even though
//    this fused kernel itself has no AVX2 tier.
// ============================================================================

using AccumulateAndEstimateDistancesFn = void (*)(const uint8_t *,
                                                  const uint8_t *,
                                                  const float *,
                                                  const float *,
                                                  float,
                                                  float,
                                                  float,
                                                  float *,
                                                  size_t);

namespace detail {

inline void accumulate_and_estimate_distances_generic(const uint8_t *ALAYA_RESTRICT codes,
                                                      const uint8_t *ALAYA_RESTRICT lp_table,
                                                      const float *ALAYA_RESTRICT f_add,
                                                      const float *ALAYA_RESTRICT f_rescale,
                                                      float g_add,
                                                      float lut_delta,
                                                      float lut_bias,
                                                      float *ALAYA_RESTRICT result,
                                                      size_t dim) {
  alignas(64) std::array<uint16_t, kBatchSize> nth_segments{};
  get_accumulate_func()(dim, codes, lp_table, nth_segments.data());
  get_estimate_distances_func()(nth_segments.data(),
                                f_add,
                                f_rescale,
                                g_add,
                                lut_delta,
                                lut_bias,
                                result);
}

#ifdef ALAYA_ARCH_X86
ALAYA_TARGET_AVX512_BW
inline void accumulate_and_estimate_distances_avx512(const uint8_t *ALAYA_RESTRICT codes,
                                                     const uint8_t *ALAYA_RESTRICT lp_table,
                                                     const float *ALAYA_RESTRICT f_add,
                                                     const float *ALAYA_RESTRICT f_rescale,
                                                     float g_add,
                                                     float lut_delta,
                                                     float lut_bias,
                                                     float *ALAYA_RESTRICT result,
                                                     size_t dim) {
  size_t code_length = dim << 2;
  __m512i c;
  __m512i lo;
  __m512i hi;
  __m512i lut;
  __m512i res_lo;
  __m512i res_hi;

  const __m512i lo_mask = _mm512_set1_epi8(0x0f);
  __m512i accu0 = _mm512_setzero_si512();
  __m512i accu1 = _mm512_setzero_si512();
  __m512i accu2 = _mm512_setzero_si512();
  __m512i accu3 = _mm512_setzero_si512();

  for (size_t i = 0; i < code_length; i += 64) {
    c = _mm512_loadu_si512(&codes[i]);
    lut = _mm512_loadu_si512(&lp_table[i]);
    lo = _mm512_and_si512(c, lo_mask);
    hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);

    res_lo = _mm512_shuffle_epi8(lut, lo);
    res_hi = _mm512_shuffle_epi8(lut, hi);

    accu0 = _mm512_add_epi16(accu0, res_lo);
    accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
    accu2 = _mm512_add_epi16(accu2, res_hi);
    accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));
  }

  accu0 = _mm512_sub_epi16(accu0, _mm512_slli_epi16(accu1, 8));
  accu2 = _mm512_sub_epi16(accu2, _mm512_slli_epi16(accu3, 8));

  const __m512i ret1 = _mm512_add_epi16(_mm512_mask_blend_epi64(0b11110000, accu0, accu1),
                                        _mm512_shuffle_i64x2(accu0, accu1, 0b01001110));
  const __m512i ret2 = _mm512_add_epi16(_mm512_mask_blend_epi64(0b11110000, accu2, accu3),
                                        _mm512_shuffle_i64x2(accu2, accu3, 0b01001110));
  __m512i ret = _mm512_setzero_si512();

  ret = _mm512_add_epi16(ret, _mm512_shuffle_i64x2(ret1, ret2, 0b10001000));
  ret = _mm512_add_epi16(ret, _mm512_shuffle_i64x2(ret1, ret2, 0b11011101));

  const __m512 v_delta = _mm512_set1_ps(lut_delta);
  const __m512 v_bias = _mm512_set1_ps(lut_bias);
  const __m512 v_gadd = _mm512_set1_ps(g_add);

  const __m256i nth_u16_lo = _mm512_castsi512_si256(ret);
  const __m256i nth_u16_hi = _mm512_extracti64x4_epi64(ret, 1);

  const __m512 nth_f_lo = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(nth_u16_lo));
  const __m512 f_add_lo = _mm512_loadu_ps(f_add);
  const __m512 f_rescale_lo = _mm512_loadu_ps(f_rescale);
  const __m512 inner_lo = _mm512_fmadd_ps(v_delta, nth_f_lo, v_bias);
  const __m512 est_lo = _mm512_fmadd_ps(f_rescale_lo, inner_lo, _mm512_add_ps(f_add_lo, v_gadd));
  _mm512_storeu_ps(result, est_lo);

  const __m512 nth_f_hi = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(nth_u16_hi));
  const __m512 f_add_hi = _mm512_loadu_ps(f_add + 16);
  const __m512 f_rescale_hi = _mm512_loadu_ps(f_rescale + 16);
  const __m512 inner_hi = _mm512_fmadd_ps(v_delta, nth_f_hi, v_bias);
  const __m512 est_hi = _mm512_fmadd_ps(f_rescale_hi, inner_hi, _mm512_add_ps(f_add_hi, v_gadd));
  _mm512_storeu_ps(result + 16, est_hi);
}
#endif

}  // namespace detail

inline auto get_accumulate_and_estimate_distances_func() -> AccumulateAndEstimateDistancesFn {
#ifdef ALAYA_ARCH_X86
  static const AccumulateAndEstimateDistancesFn kFunc = select_rabitq_simd_avx512_or_generic<
      AccumulateAndEstimateDistancesFn>(detail::accumulate_and_estimate_distances_generic,
                                        detail::accumulate_and_estimate_distances_avx512);
#else
  static const AccumulateAndEstimateDistancesFn kFunc =
      detail::accumulate_and_estimate_distances_generic;
#endif
  return kFunc;
}

// ============================================================================
// 4. flip_sign (rotator.hpp) — needs AVX512DQ for the masked packed-float xor
//    (_mm512_mask_xor_ps), hence ALAYA_TARGET_AVX512 (F+BW+DQ) rather than the
//    _BW-only macro; matches distance_ip.ipp's own ip_sqr_avx512 precedent of
//    using the DQ-inclusive macro while gating dispatch on avx512f_+avx512bw_.
// ============================================================================

using FlipSignFn = void (*)(const uint8_t *, float *, size_t);

namespace detail {

inline void flip_sign_generic(const uint8_t *flip, float *data, size_t dim) {
  for (size_t i = 0; i < dim; ++i) {
    const auto byte = flip[i / 8];
    const auto bit = static_cast<uint8_t>((byte >> (i % 8)) & 0x1U);
    if (bit != 0U) {
      data[i] = -data[i];
    }
  }
}

#ifdef ALAYA_ARCH_X86
ALAYA_TARGET_AVX512
inline void flip_sign_avx512(const uint8_t *flip, float *data, size_t dim) {
  constexpr size_t kFloatsPerChunk = 64;  // Process 64 floats per iteration
  static_assert(kFloatsPerChunk % 16 == 0,
                "floats_per_chunk must be divisible by AVX512 register width");
  for (size_t i = 0; i < dim; i += kFloatsPerChunk) {
    // Load 64 bits (8 bytes) from the bit sequence
    uint64_t mask_bits;
    std::memcpy(&mask_bits, &flip[i / 8], sizeof(mask_bits));

    // Split into four 16-bit mask segments
    const __mmask16 mask0 = _cvtu32_mask16(static_cast<uint32_t>(mask_bits & 0xFFFF));
    const __mmask16 mask1 = _cvtu32_mask16(static_cast<uint32_t>((mask_bits >> 16) & 0xFFFF));
    const __mmask16 mask2 = _cvtu32_mask16(static_cast<uint32_t>((mask_bits >> 32) & 0xFFFF));
    const __mmask16 mask3 = _cvtu32_mask16(static_cast<uint32_t>((mask_bits >> 48) & 0xFFFF));

    // Prepare sign-flip constant
    const __m512 sign_flip = _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000));

    // Process 16 floats at a time with each mask segment
    __m512 vec0 = _mm512_loadu_ps(&data[i]);
    vec0 = _mm512_mask_xor_ps(vec0, mask0, vec0, sign_flip);
    _mm512_storeu_ps(&data[i], vec0);

    __m512 vec1 = _mm512_loadu_ps(&data[i + 16]);
    vec1 = _mm512_mask_xor_ps(vec1, mask1, vec1, sign_flip);
    _mm512_storeu_ps(&data[i + 16], vec1);

    __m512 vec2 = _mm512_loadu_ps(&data[i + 32]);
    vec2 = _mm512_mask_xor_ps(vec2, mask2, vec2, sign_flip);
    _mm512_storeu_ps(&data[i + 32], vec2);

    __m512 vec3 = _mm512_loadu_ps(&data[i + 48]);
    vec3 = _mm512_mask_xor_ps(vec3, mask3, vec3, sign_flip);
    _mm512_storeu_ps(&data[i + 48], vec3);
  }
}
#endif

}  // namespace detail

inline auto get_flip_sign_func() -> FlipSignFn {
#ifdef ALAYA_ARCH_X86
  static const FlipSignFn kFunc =
      select_rabitq_simd_avx512_or_generic<FlipSignFn>(detail::flip_sign_generic,
                                                       detail::flip_sign_avx512);
#else
  static const FlipSignFn kFunc = detail::flip_sign_generic;
#endif
  return kFunc;
}

// ============================================================================
// 5. kacs_walk (rotator.hpp, FhtKacRotator) — pure add/sub, AVX512F suffices;
//    ALAYA_TARGET_AVX512_BW used for consistency with the other kernels here.
// ============================================================================

using KacsWalkFn = void (*)(float *, size_t);

namespace detail {

inline void kacs_walk_generic(float *data, size_t len) {
  for (size_t i = 0; i < len / 2; ++i) {
    const float x = data[i];
    const float y = data[i + (len / 2)];
    data[i] = x + y;
    data[i + (len / 2)] = x - y;
  }
}

#ifdef ALAYA_ARCH_X86
ALAYA_TARGET_AVX512_BW
inline void kacs_walk_avx512(float *data, size_t len) {
  // ! len % 32 == 0;
  for (size_t i = 0; i < len / 2; i += 16) {
    __m512 x = _mm512_loadu_ps(&data[i]);
    __m512 y = _mm512_loadu_ps(&data[i + (len / 2)]);

    __m512 new_x = _mm512_add_ps(x, y);
    __m512 new_y = _mm512_sub_ps(x, y);

    _mm512_storeu_ps(&data[i], new_x);
    _mm512_storeu_ps(&data[i + (len / 2)], new_y);
  }
}
#endif

}  // namespace detail

inline auto get_kacs_walk_func() -> KacsWalkFn {
#ifdef ALAYA_ARCH_X86
  static const KacsWalkFn kFunc =
      select_rabitq_simd_avx512_or_generic<KacsWalkFn>(detail::kacs_walk_generic,
                                                       detail::kacs_walk_avx512);
#else
  static const KacsWalkFn kFunc = detail::kacs_walk_generic;
#endif
  return kFunc;
}

// ============================================================================
// 6. scalar_quantize_optimized (lut.hpp) — float-only dispatch; the generic
//    tier reproduces lut.hpp's scalar_quantize_normal<float> body verbatim
//    (Eigen round-then-cast) rather than calling it, since lut.hpp includes
//    this file (not the other way around) and calling back into lut.hpp would
//    be circular. The two bodies must stay numerically identical.
// ============================================================================

using ScalarQuantizeOptimizedFn = void (*)(uint8_t *, const float *, size_t, float, float);

namespace detail {

inline void scalar_quantize_optimized_generic(uint8_t *ALAYA_RESTRICT result,
                                              const float *ALAYA_RESTRICT vec0,
                                              size_t dim,
                                              float lo,
                                              float delta) {
  const float one_over_delta = 1.0F / delta;
  kernels::linalg::ConstRowMajorArrayMap<float> v0(vec0, 1, static_cast<long>(dim));  // NOLINT
  kernels::linalg::RowMajorArrayMap<uint8_t> res(result, 1, dim);
  res = ((v0 - lo) * one_over_delta).round().template cast<uint8_t>();
}

#ifdef ALAYA_ARCH_X86
// _mm_storeu_epi8 (128-bit byte store) needs AVX512VL+AVX512BW, which neither
// ALAYA_TARGET_AVX512 nor ALAYA_TARGET_AVX512_BW declare (confirmed by
// compiler error: "inlining failed ... target specific option mismatch" with
// just F+BW+DQ) — add avx512vl explicitly for this one kernel.
ALAYA_TARGET_AVX512_VL inline void scalar_quantize_optimized_avx512(uint8_t *ALAYA_RESTRICT result,
                                                                    const float *ALAYA_RESTRICT
                                                                        vec0,
                                                                    size_t dim,
                                                                    float lo,
                                                                    float delta) {
  size_t mul16 = dim - (dim & 0b1111);
  size_t i = 0;
  float one_over_delta = 1 / delta;
  auto lo512 = _mm512_set1_ps(lo);
  auto od512 = _mm512_set1_ps(one_over_delta);
  for (; i < mul16; i += 16) {
    auto cur = _mm512_loadu_ps(&vec0[i]);
    cur = _mm512_mul_ps(_mm512_sub_ps(cur, lo512), od512);  // NOLINT
    auto i8 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(cur));
    _mm_storeu_epi8(&result[i], i8);
  }
  for (; i < dim; ++i) {
    result[i] = static_cast<uint8_t>(std::round((vec0[i] - lo) * one_over_delta));
  }
}
#endif

}  // namespace detail

inline auto get_scalar_quantize_optimized_func() -> ScalarQuantizeOptimizedFn {
#ifdef ALAYA_ARCH_X86
  static const ScalarQuantizeOptimizedFn kFunc = select_rabitq_simd_avx512_or_generic<
      ScalarQuantizeOptimizedFn>(detail::scalar_quantize_optimized_generic,
                                 detail::scalar_quantize_optimized_avx512);
#else
  static const ScalarQuantizeOptimizedFn kFunc = detail::scalar_quantize_optimized_generic;
#endif
  return kFunc;
}

}  // namespace alaya::rabitq_simd
