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

#pragma once

#include "platform/detect.hpp"

#include <array>
#include <cstdint>
#include <type_traits>

#include "core/log.hpp"
#include "simd/fastscan.hpp"
#include "space/quant/rabitq/defines.hpp"
#include "space/quant/rabitq/dispatch.hpp"

namespace alaya::fastscan {
using ::alaya::simd::fastscan::kBatchSize;

inline auto log_scalar_fastscan_fallback() -> void {
  LOG_INFO_ONCE("rabitq fallback: AVX-512 fastscan is unavailable, using portable fallback path");
}

namespace detail {

inline void accumulate_scalar(const uint8_t *ALAYA_RESTRICT codes,
                              const uint8_t *ALAYA_RESTRICT lp_table,
                              uint16_t *ALAYA_RESTRICT result,
                              size_t dim) {
  ::alaya::simd::fastscan::accumulate_generic(dim, codes, lp_table, result);
}

}  // namespace detail

/**
 * @brief Pack quantization codes, store in blocks, the data organization is illustrated in
 * the link and kPerm0. Since we pack codes as 32-sized groups, if the num is not a multiple
 * of 32, we have to use some space for these absent data
 *
 * @param padded_dim dimension of quantized data (i.e., quantization code)
 * @param quantization_code quantizaiton code, stored as uint8
 * @param num   number of quantization code
 * @param blocks packed quantization code
 */
inline void pack_codes(size_t padded_dim,
                       const uint8_t *quantization_code,
                       size_t num,
                       uint8_t *blocks) {
  ::alaya::simd::fastscan::pack_codes_bytes<false>(padded_dim, quantization_code, num, blocks);
}

//  use fast scan to accumulate one block, dim % 16 == 0.
//  Runtime-dispatched across scalar/AVX2/AVX-512 SIMD tiers; see
//  space/quant/rabitq/dispatch.hpp (reuses simd::fastscan's own three kernels
//  directly instead of re-implementing them here).
inline void accumulate(const uint8_t *ALAYA_RESTRICT codes,
                       const uint8_t *ALAYA_RESTRICT lp_table,
                       uint16_t *ALAYA_RESTRICT result,
                       size_t dim) {
  if ((dim & 0x0FU) != 0U) {
    log_scalar_fastscan_fallback();
    detail::accumulate_scalar(codes, lp_table, result, dim);
    return;
  }

  rabitq_simd::get_accumulate_func()(dim, codes, lp_table, result);
}

// Runtime-dispatched (generic/AVX-512); see space/quant/rabitq/dispatch.hpp.
template <typename T>
inline void estimate_distances(const uint16_t *ALAYA_RESTRICT nth_segments,
                               const T *ALAYA_RESTRICT f_add,
                               const T *ALAYA_RESTRICT f_rescale,
                               T g_add,
                               T lut_delta,
                               T lut_bias,
                               T *ALAYA_RESTRICT result) {
  static_assert(std::is_same_v<T, float>, "fastscan::estimate_distances only supports float.");
  rabitq_simd::get_estimate_distances_func()(nth_segments,
                                             f_add,
                                             f_rescale,
                                             g_add,
                                             lut_delta,
                                             lut_bias,
                                             result);
}

template <typename T>
inline void accumulate_and_estimate_distances(const uint8_t *ALAYA_RESTRICT codes,
                                              const uint8_t *ALAYA_RESTRICT lp_table,
                                              const T *ALAYA_RESTRICT f_add,
                                              const T *ALAYA_RESTRICT f_rescale,
                                              T g_add,
                                              T lut_delta,
                                              T lut_bias,
                                              T *ALAYA_RESTRICT result,
                                              size_t dim) {
  // Keep this AVX-512 fused memory-QG builder path separate from LASER's dispatched
  // accumulate -> convert -> distance pipeline.  Sharing the standalone integer
  // accumulate kernel does not justify adding an intermediate store/load here;
  // the two consumers intentionally retain their individually optimized pipelines.
  // Runtime-dispatched (generic/AVX-512); see space/quant/rabitq/dispatch.hpp.
  static_assert(std::is_same_v<T, float>,
                "fastscan::accumulate_and_estimate_distances only supports float.");
  if ((dim & 0x0FU) != 0U) {
    log_scalar_fastscan_fallback();
    alignas(64) std::array<uint16_t, kBatchSize> nth_segments{};
    detail::accumulate_scalar(codes, lp_table, nth_segments.data(), dim);
    estimate_distances(nth_segments.data(), f_add, f_rescale, g_add, lut_delta, lut_bias, result);
    return;
  }

  rabitq_simd::get_accumulate_and_estimate_distances_func()(codes,
                                                            lp_table,
                                                            f_add,
                                                            f_rescale,
                                                            g_add,
                                                            lut_delta,
                                                            lut_bias,
                                                            result,
                                                            dim);
}

// pack lookup table for fastscan, for each 4 dim, we have 16 (2^4) different results
// ! dim % 4 == 0
template <typename T>
inline void pack_lut(size_t dim, const T *ALAYA_RESTRICT query, T *ALAYA_RESTRICT lut) {
  ::alaya::simd::fastscan::build_lut(dim, query, lut);
}

}  // namespace alaya::fastscan
