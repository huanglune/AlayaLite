// SPDX-FileCopyrightText: 2025 VectorDB.NTU
// SPDX-FileCopyrightText: 2025 AlayaDB.AI
// SPDX-License-Identifier: MIT AND Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "utils/platform.hpp"

namespace alaya::simd::fastscan {

constexpr std::array<int, 16> kPackedLaneOrder =
    {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
inline void accumulate_generic(size_t dim,
                               const uint8_t *ALAYA_RESTRICT codes,
                               const uint8_t *ALAYA_RESTRICT lut,
                               uint16_t *ALAYA_RESTRICT result) {
  std::fill(result, result + 32, static_cast<uint16_t>(0));
  const size_t num_codebook = dim >> 2;
  const uint8_t *packed = codes;
  for (size_t codebook = 0; codebook < num_codebook; codebook += 2) {
    const uint8_t *lut0 = lut + codebook * 16;
    const uint8_t *lut1 = lut0 + 16;
    for (size_t lane = 0; lane < kPackedLaneOrder.size(); ++lane) {
      const int low_id = kPackedLaneOrder[lane];
      const int high_id = low_id + 16;
      const uint8_t packed0 = packed[lane];
      result[low_id] = static_cast<uint16_t>(result[low_id] + lut0[packed0 & 0x0FU]);
      result[high_id] = static_cast<uint16_t>(result[high_id] + lut0[packed0 >> 4U]);
      const uint8_t packed1 = packed[lane + 16];
      result[low_id] = static_cast<uint16_t>(result[low_id] + lut1[packed1 & 0x0FU]);
      result[high_id] = static_cast<uint16_t>(result[high_id] + lut1[packed1 >> 4U]);
    }
    packed += 32;
  }
}

#ifdef ALAYA_ARCH_X86
ALAYA_TARGET_AVX512_BW
inline void accumulate_avx512(size_t dim,
                              const uint8_t *ALAYA_RESTRICT codes,
                              const uint8_t *ALAYA_RESTRICT lut_table,
                              uint16_t *ALAYA_RESTRICT result) {
  const size_t code_length = dim << 2;
  const __m512i lo_mask = _mm512_set1_epi8(0x0f);
  __m512i accu0 = _mm512_setzero_si512();
  __m512i accu1 = _mm512_setzero_si512();
  __m512i accu2 = _mm512_setzero_si512();
  __m512i accu3 = _mm512_setzero_si512();
  for (size_t i = 0; i < code_length; i += 64) {
    const __m512i c = _mm512_loadu_si512(&codes[i]);
    const __m512i lut = _mm512_loadu_si512(&lut_table[i]);
    const __m512i lo = _mm512_and_si512(c, lo_mask);
    const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);
    const __m512i res_lo = _mm512_shuffle_epi8(lut, lo);
    const __m512i res_hi = _mm512_shuffle_epi8(lut, hi);
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
  __m512i ret = _mm512_add_epi16(_mm512_shuffle_i64x2(ret1, ret2, 0b10001000),
                                 _mm512_shuffle_i64x2(ret1, ret2, 0b11011101));
  _mm512_storeu_si512(result, ret);
}

ALAYA_TARGET_AVX2
inline void accumulate_avx2(size_t dim,
                            const uint8_t *ALAYA_RESTRICT codes,
                            const uint8_t *ALAYA_RESTRICT lut_table,
                            uint16_t *ALAYA_RESTRICT result) {
  const size_t code_length = dim << 2;
  const __m256i low_mask = _mm256_set1_epi8(0xf);
  __m256i accu0 = _mm256_setzero_si256();
  __m256i accu1 = _mm256_setzero_si256();
  __m256i accu2 = _mm256_setzero_si256();
  __m256i accu3 = _mm256_setzero_si256();
  for (size_t i = 0; i < code_length; i += 64) {
    for (size_t half = 0; half < 64; half += 32) {
      const __m256i c =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&codes[i + half]));
      const __m256i lut =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&lut_table[i + half]));
      const __m256i lo = _mm256_and_si256(c, low_mask);
      const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);
      const __m256i res_lo = _mm256_shuffle_epi8(lut, lo);
      const __m256i res_hi = _mm256_shuffle_epi8(lut, hi);
      accu0 = _mm256_add_epi16(accu0, res_lo);
      accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
      accu2 = _mm256_add_epi16(accu2, res_hi);
      accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));
    }
  }
  accu0 = _mm256_sub_epi16(accu0, _mm256_slli_epi16(accu1, 8));
  const __m256i dis0 = _mm256_add_epi16(_mm256_permute2f128_si256(accu0, accu1, 0x21),
                                        _mm256_blend_epi32(accu0, accu1, 0xF0));
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(result), dis0);
  accu2 = _mm256_sub_epi16(accu2, _mm256_slli_epi16(accu3, 8));
  const __m256i dis1 = _mm256_add_epi16(_mm256_permute2f128_si256(accu2, accu3, 0x21),
                                        _mm256_blend_epi32(accu2, accu3, 0xF0));
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(&result[16]), dis1);
}
#endif

}  // namespace alaya::simd::fastscan
