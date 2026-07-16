// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "index/graph/laser/qg/qg_scanner.hpp"

namespace alaya::laser {
namespace {

TEST(QGScannerFusedTest, RandomRowsAndQueriesMatchThreePassReferenceBitwise) {
  constexpr std::array<size_t, 2> kDegreeBounds = {32, 64};
  // The production 768d configuration pads to 1024; 2048 also stresses
  // zero-extension when uint16 accumulators cross INT16_MAX.
  constexpr std::array<size_t, 4> kPaddedDims = {64, 128, 1024, 2048};
  constexpr size_t kTrials = 24;

  std::mt19937_64 rng(0x5CA11F05EDULL);
  std::uniform_int_distribution<uint64_t> code_dist;
  std::uniform_int_distribution<int> query_dist(0, 255);
  std::uniform_real_distribution<float> factor_dist(-4.0F, 4.0F);
  std::uniform_real_distribution<float> norm_dist(0.0F, 128.0F);
  std::uniform_real_distribution<float> scale_dist(0.0001F, 2.0F);
  std::uniform_real_distribution<float> lower_dist(-8.0F, 8.0F);
#ifdef ALAYA_ARCH_X86
  simd::AccumulateFn volatile avx2_accumulate = simd::detail::accumulate_impl_avx2;
  simd::ConvertAccumFn volatile avx2_convert = simd::detail::convert_accum_to_float_avx2;
  simd::AppDistFn volatile avx2_estimate = simd::detail::appro_dist_impl_avx2;
  qg_scanner_detail::FusedScanBatchFn volatile avx2_fused_scan =
      qg_scanner_detail::fused_scan_batch_avx2<false>;
  bool saw_accumulator_above_int16_max = false;
#endif

  for (const size_t degree_bound : kDegreeBounds) {
    for (const size_t padded_dim : kPaddedDims) {
      QGScanner scanner(padded_dim, degree_bound);
      std::vector<uint64_t> binary_codes(degree_bound * padded_dim / 64);
      std::vector<uint8_t> packed_codes(degree_bound * padded_dim / 8);
      std::vector<uint8_t> byte_query(padded_dim);
      std::vector<uint8_t> lut(padded_dim << 2);
      std::vector<float> factors(3 * degree_bound);
      std::vector<float> fused(degree_bound);
      std::vector<float> reference(degree_bound);
#ifdef ALAYA_ARCH_X86
      alignas(64) std::array<uint16_t, kBatchSize> avx2_accumulated{};
      alignas(64) std::array<float, kBatchSize> avx2_converted{};
      alignas(64) std::array<float, kBatchSize> avx2_fused{};
      alignas(64) std::array<float, kBatchSize> avx2_reference{};
#endif

      for (size_t trial = 0; trial < kTrials; ++trial) {
        for (uint64_t &code : binary_codes) {
          code = code_dist(rng);
        }
        pack_codes(padded_dim, binary_codes.data(), degree_bound, packed_codes.data());

        for (uint8_t &value : byte_query) {
          value = static_cast<uint8_t>(query_dist(rng));
        }
        scanner.pack_lut(byte_query.data(), lut.data());

        for (float &factor : factors) {
          factor = factor_dist(rng);
        }

        const float sqr_y = norm_dist(rng);
        const float vl = lower_dist(rng);
        const float width = scale_dist(rng);
        const float sqr_qr = norm_dist(rng);
        const int32_t sumq = std::accumulate(byte_query.begin(), byte_query.end(), int32_t{0});

        scanner.scan_neighbors_three_pass_reference(reference.data(),
                                                    lut.data(),
                                                    sqr_y,
                                                    vl,
                                                    width,
                                                    sqr_qr,
                                                    sumq,
                                                    packed_codes.data(),
                                                    factors.data());
        scanner.scan_neighbors(fused.data(),
                               lut.data(),
                               sqr_y,
                               vl,
                               width,
                               sqr_qr,
                               sumq,
                               packed_codes.data(),
                               factors.data());

        for (size_t neighbor = 0; neighbor < degree_bound; ++neighbor) {
          ASSERT_EQ(std::bit_cast<uint32_t>(fused[neighbor]),
                    std::bit_cast<uint32_t>(reference[neighbor]))
              << "degree_bound=" << degree_bound << " padded_dim=" << padded_dim
              << " trial=" << trial << " neighbor=" << neighbor << " fused=" << fused[neighbor]
              << " reference=" << reference[neighbor];
        }

#ifdef ALAYA_ARCH_X86
        // The deployment CPU selects the legacy AVX-512 arithmetic oracle,
        // while the fused accumulator itself is intentionally AVX2. Exercise
        // the AVX2-only dispatch mode explicitly as well.
        const float *triple_x = factors.data();
        const float *fac_dq = triple_x + degree_bound;
        const float *fac_vq = fac_dq + degree_bound;
        for (size_t batch = 0; batch < degree_bound; batch += kBatchSize) {
          const uint8_t *batch_codes = packed_codes.data() + batch * padded_dim / 8;
          avx2_accumulate(padded_dim, batch_codes, lut.data(), avx2_accumulated.data());
          for (const uint16_t accumulated : avx2_accumulated) {
            saw_accumulator_above_int16_max |= accumulated > 0x7FFFU;
          }
          avx2_convert(kBatchSize, avx2_accumulated.data(), sumq, avx2_converted.data());
          avx2_estimate(kBatchSize,
                        sqr_y,
                        width,
                        vl,
                        sqr_qr,
                        avx2_converted.data(),
                        triple_x + batch,
                        fac_dq + batch,
                        fac_vq + batch,
                        avx2_reference.data());
          avx2_fused_scan(padded_dim,
                          batch_codes,
                          lut.data(),
                          sqr_y,
                          vl,
                          width,
                          sqr_qr,
                          sumq,
                          triple_x + batch,
                          fac_dq + batch,
                          fac_vq + batch,
                          avx2_fused.data());
          for (size_t lane = 0; lane < kBatchSize; ++lane) {
            ASSERT_EQ(std::bit_cast<uint32_t>(avx2_fused[lane]),
                      std::bit_cast<uint32_t>(avx2_reference[lane]))
                << "AVX2 degree_bound=" << degree_bound << " padded_dim=" << padded_dim
                << " trial=" << trial << " batch=" << batch << " lane=" << lane;
          }
        }
#endif
      }
    }
  }
#ifdef ALAYA_ARCH_X86
  EXPECT_TRUE(saw_accumulator_above_int16_max);
#endif
}

}  // namespace
}  // namespace alaya::laser
