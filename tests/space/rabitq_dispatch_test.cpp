// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "platform/detect.hpp"
#include "simd/cpu_features.hpp"
#include "simd/fastscan.hpp"
#include "space/quant/rabitq/dispatch.hpp"

// Differential coverage for space/quant/rabitq/dispatch.hpp: every dispatched
// kernel is checked against its own generic tier (unconditionally — both tiers
// live in the same translation unit now, see dispatch.hpp's header comment),
// plus a factory-selection assertion mirroring
// tests/laser/simd_dispatch_test.cpp's LaserSimdDispatchTest.
// FactoriesSelectHighestSupportedIsa.
namespace alaya {
namespace {

auto has_avx512_vl_target() -> bool {
  const auto &features = simd::get_cpu_features();
  return features.avx512f_ && features.avx512bw_ && features.avx512dq_ && features.avx512vl_ &&
         features.avx512_os_state_;
}

auto has_avx2_fma() -> bool {
  const auto &features = simd::get_cpu_features();
  return features.avx2_ && features.fma_;
}

TEST(RabitqDispatchTest, FactoriesSelectHighestSupportedIsa) {
#ifdef ALAYA_ARCH_X86
  if (has_avx512_vl_target()) {
    EXPECT_STREQ(rabitq_simd::get_rabitq_simd_name(), "avx512");
    EXPECT_EQ(rabitq_simd::get_accumulate_func(), ::alaya::simd::fastscan::accumulate_avx512);
    EXPECT_EQ(rabitq_simd::get_estimate_distances_func(),
              rabitq_simd::detail::estimate_distances_avx512);
    EXPECT_EQ(rabitq_simd::get_accumulate_and_estimate_distances_func(),
              rabitq_simd::detail::accumulate_and_estimate_distances_avx512);
    EXPECT_EQ(rabitq_simd::get_flip_sign_func(), rabitq_simd::detail::flip_sign_avx512);
    EXPECT_EQ(rabitq_simd::get_kacs_walk_func(), rabitq_simd::detail::kacs_walk_avx512);
    EXPECT_EQ(rabitq_simd::get_scalar_quantize_optimized_func(),
              rabitq_simd::detail::scalar_quantize_optimized_avx512);
    return;
  }
  if (has_avx2_fma()) {
    EXPECT_STREQ(rabitq_simd::get_rabitq_simd_name(), "avx2");
    EXPECT_EQ(rabitq_simd::get_accumulate_func(), ::alaya::simd::fastscan::accumulate_avx2);
    // These five kernels never had an AVX2 tier pre-refactor (scope-controlled:
    // convert existing #if ladders to dispatch, do not add new SIMD tiers), so
    // an AVX2-only host falls back to generic for them, exactly as before.
    EXPECT_EQ(rabitq_simd::get_estimate_distances_func(),
              rabitq_simd::detail::estimate_distances_generic);
    EXPECT_EQ(rabitq_simd::get_accumulate_and_estimate_distances_func(),
              rabitq_simd::detail::accumulate_and_estimate_distances_generic);
    EXPECT_EQ(rabitq_simd::get_flip_sign_func(), rabitq_simd::detail::flip_sign_generic);
    EXPECT_EQ(rabitq_simd::get_kacs_walk_func(), rabitq_simd::detail::kacs_walk_generic);
    EXPECT_EQ(rabitq_simd::get_scalar_quantize_optimized_func(),
              rabitq_simd::detail::scalar_quantize_optimized_generic);
    return;
  }
#endif
  EXPECT_STREQ(rabitq_simd::get_rabitq_simd_name(), "generic");
  EXPECT_EQ(rabitq_simd::get_accumulate_func(), ::alaya::simd::fastscan::accumulate_generic);
  EXPECT_EQ(rabitq_simd::get_estimate_distances_func(),
            rabitq_simd::detail::estimate_distances_generic);
  EXPECT_EQ(rabitq_simd::get_accumulate_and_estimate_distances_func(),
            rabitq_simd::detail::accumulate_and_estimate_distances_generic);
  EXPECT_EQ(rabitq_simd::get_flip_sign_func(), rabitq_simd::detail::flip_sign_generic);
  EXPECT_EQ(rabitq_simd::get_kacs_walk_func(), rabitq_simd::detail::kacs_walk_generic);
  EXPECT_EQ(rabitq_simd::get_scalar_quantize_optimized_func(),
            rabitq_simd::detail::scalar_quantize_optimized_generic);
}

TEST(RabitqDispatchTest, Avx512RequiresFullVlTargetFeatureSetAndOsState) {
  simd::CpuFeatures features;
  features.avx512f_ = true;
  features.avx512bw_ = true;
  features.avx2_ = true;
  features.fma_ = true;

  EXPECT_EQ(rabitq_simd::select_rabitq_simd_level(features), rabitq_simd::RabitqSimdLevel::kAvx2);

  features.avx512dq_ = true;
  features.avx512vl_ = true;
  EXPECT_EQ(rabitq_simd::select_rabitq_simd_level(features), rabitq_simd::RabitqSimdLevel::kAvx2);

  features.avx512_os_state_ = true;
  EXPECT_EQ(rabitq_simd::select_rabitq_simd_level(features), rabitq_simd::RabitqSimdLevel::kAvx512);
}

TEST(RabitqDispatchTest, AccumulateDifferentialFuzz) {
  std::mt19937 rng(0x5A11A5A1U);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  for (size_t dim : {16U, 32U, 64U, 128U, 256U}) {
    for (int trial = 0; trial < 50; ++trial) {
      std::vector<uint8_t> codes(dim << 2);
      std::vector<uint8_t> lut(dim << 2);
      std::generate(codes.begin(), codes.end(), [&] {
        return byte_dist(rng);
      });
      std::generate(lut.begin(), lut.end(), [&] {
        return byte_dist(rng);
      });

      std::array<uint16_t, 32> generic{};
      std::array<uint16_t, 32> dispatched{};
      ::alaya::simd::fastscan::accumulate_generic(dim, codes.data(), lut.data(), generic.data());
      rabitq_simd::get_accumulate_func()(dim, codes.data(), lut.data(), dispatched.data());

      EXPECT_EQ(generic, dispatched) << "dim=" << dim << " trial=" << trial;
    }
  }
}

TEST(RabitqDispatchTest, EstimateDistancesDifferentialFuzz) {
  std::mt19937 rng(0xE5714A7EU);
  std::uniform_int_distribution<int> seg_dist(0, 4000);
  std::uniform_real_distribution<float> coef_dist(-5.0F, 5.0F);

  for (int trial = 0; trial < 200; ++trial) {
    alignas(64) std::array<uint16_t, rabitq_simd::kBatchSize> nth_segments{};
    alignas(64) std::array<float, rabitq_simd::kBatchSize> f_add{};
    alignas(64) std::array<float, rabitq_simd::kBatchSize> f_rescale{};
    for (size_t i = 0; i < rabitq_simd::kBatchSize; ++i) {
      nth_segments[i] = static_cast<uint16_t>(seg_dist(rng));
      f_add[i] = coef_dist(rng);
      f_rescale[i] = coef_dist(rng);
    }
    const float g_add = coef_dist(rng);
    const float lut_delta = coef_dist(rng);
    const float lut_bias = coef_dist(rng);

    alignas(64) std::array<float, rabitq_simd::kBatchSize> generic_result{};
    alignas(64) std::array<float, rabitq_simd::kBatchSize> dispatched_result{};
    rabitq_simd::detail::estimate_distances_generic(nth_segments.data(),
                                                    f_add.data(),
                                                    f_rescale.data(),
                                                    g_add,
                                                    lut_delta,
                                                    lut_bias,
                                                    generic_result.data());
    rabitq_simd::get_estimate_distances_func()(nth_segments.data(),
                                               f_add.data(),
                                               f_rescale.data(),
                                               g_add,
                                               lut_delta,
                                               lut_bias,
                                               dispatched_result.data());

    for (size_t i = 0; i < rabitq_simd::kBatchSize; ++i) {
      const float scale = std::max(1.0F, std::abs(generic_result[i]));
      EXPECT_NEAR(generic_result[i], dispatched_result[i], 1.0e-3F * scale)
          << "trial=" << trial << " i=" << i;
    }
  }
}

TEST(RabitqDispatchTest, AccumulateAndEstimateDistancesDifferentialFuzz) {
  std::mt19937 rng(0xACC5E57AU);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  std::uniform_real_distribution<float> coef_dist(-3.0F, 3.0F);

  for (size_t dim : {16U, 32U, 64U, 128U}) {
    for (int trial = 0; trial < 50; ++trial) {
      std::vector<uint8_t> codes(dim << 2);
      std::vector<uint8_t> lut(dim << 2);
      std::generate(codes.begin(), codes.end(), [&] {
        return byte_dist(rng);
      });
      std::generate(lut.begin(), lut.end(), [&] {
        return byte_dist(rng);
      });

      alignas(64) std::array<float, rabitq_simd::kBatchSize> f_add{};
      alignas(64) std::array<float, rabitq_simd::kBatchSize> f_rescale{};
      for (size_t i = 0; i < rabitq_simd::kBatchSize; ++i) {
        f_add[i] = coef_dist(rng);
        f_rescale[i] = coef_dist(rng);
      }
      const float g_add = coef_dist(rng);
      const float lut_delta = coef_dist(rng);
      const float lut_bias = coef_dist(rng);

      // Scalar oracle: compose the two known-generic primitives directly,
      // bypassing dispatch entirely (rather than calling
      // accumulate_and_estimate_distances_generic, whose own body calls the
      // live-dispatched get_accumulate_func()/get_estimate_distances_func()
      // and would therefore silently run AVX-512 sub-kernels on this host).
      std::array<uint16_t, rabitq_simd::kBatchSize> accumulated{};
      ::alaya::simd::fastscan::accumulate_generic(dim,
                                                  codes.data(),
                                                  lut.data(),
                                                  accumulated.data());
      alignas(64) std::array<float, rabitq_simd::kBatchSize> oracle{};
      rabitq_simd::detail::estimate_distances_generic(accumulated.data(),
                                                      f_add.data(),
                                                      f_rescale.data(),
                                                      g_add,
                                                      lut_delta,
                                                      lut_bias,
                                                      oracle.data());

      alignas(64) std::array<float, rabitq_simd::kBatchSize> dispatched{};
      rabitq_simd::get_accumulate_and_estimate_distances_func()(codes.data(),
                                                                lut.data(),
                                                                f_add.data(),
                                                                f_rescale.data(),
                                                                g_add,
                                                                lut_delta,
                                                                lut_bias,
                                                                dispatched.data(),
                                                                dim);

      for (size_t i = 0; i < rabitq_simd::kBatchSize; ++i) {
        const float scale = std::max(1.0F, std::abs(oracle[i]));
        EXPECT_NEAR(oracle[i], dispatched[i], 1.0e-3F * scale)
            << "dim=" << dim << " trial=" << trial << " i=" << i;
      }
    }
  }
}

TEST(RabitqDispatchTest, FlipSignDifferentialFuzz) {
  std::mt19937 rng(0xF119515CU);
  std::uniform_real_distribution<float> val_dist(-10.0F, 10.0F);
  std::uniform_int_distribution<int> byte_dist(0, 255);

  for (size_t dim : {64U, 128U, 256U, 512U}) {
    for (int trial = 0; trial < 50; ++trial) {
      std::vector<uint8_t> flip(dim / 8);
      std::generate(flip.begin(), flip.end(), [&] {
        return byte_dist(rng);
      });

      std::vector<float> base(dim);
      std::generate(base.begin(), base.end(), [&] {
        return val_dist(rng);
      });

      std::vector<float> generic_data = base;
      std::vector<float> dispatched_data = base;
      rabitq_simd::detail::flip_sign_generic(flip.data(), generic_data.data(), dim);
      rabitq_simd::get_flip_sign_func()(flip.data(), dispatched_data.data(), dim);

      EXPECT_EQ(generic_data, dispatched_data) << "dim=" << dim << " trial=" << trial;
    }
  }
}

TEST(RabitqDispatchTest, KacsWalkDifferentialFuzz) {
  std::mt19937 rng(0x4AC5EA1CU);
  std::uniform_real_distribution<float> val_dist(-10.0F, 10.0F);

  for (size_t len : {32U, 64U, 128U, 256U}) {
    for (int trial = 0; trial < 50; ++trial) {
      std::vector<float> base(len);
      std::generate(base.begin(), base.end(), [&] {
        return val_dist(rng);
      });

      std::vector<float> generic_data = base;
      std::vector<float> dispatched_data = base;
      rabitq_simd::detail::kacs_walk_generic(generic_data.data(), len);
      rabitq_simd::get_kacs_walk_func()(dispatched_data.data(), len);

      for (size_t i = 0; i < len; ++i) {
        EXPECT_FLOAT_EQ(generic_data[i], dispatched_data[i])
            << "len=" << len << " trial=" << trial << " i=" << i;
      }
    }
  }
}

TEST(RabitqDispatchTest, ScalarQuantizeOptimizedDifferentialFuzz) {
  std::mt19937 rng(0x5CA1A45CU);
  std::uniform_real_distribution<float> val_dist(-7.0F, 7.0F);

  for (size_t dim : {15U, 16U, 17U, 64U, 100U, 257U}) {
    for (int trial = 0; trial < 50; ++trial) {
      std::vector<float> vec0(dim);
      std::generate(vec0.begin(), vec0.end(), [&] {
        return val_dist(rng);
      });
      float lo = *std::min_element(vec0.begin(), vec0.end());
      float hi = *std::max_element(vec0.begin(), vec0.end());
      float delta = (hi - lo) / 255.0F;
      if (delta <= 0.0F) {
        delta = 1.0F;
      }

      std::vector<uint8_t> generic_result(dim);
      std::vector<uint8_t> dispatched_result(dim);
      rabitq_simd::detail::scalar_quantize_optimized_generic(generic_result.data(),
                                                             vec0.data(),
                                                             dim,
                                                             lo,
                                                             delta);
      rabitq_simd::get_scalar_quantize_optimized_func()(dispatched_result.data(),
                                                        vec0.data(),
                                                        dim,
                                                        lo,
                                                        delta);

      EXPECT_EQ(generic_result, dispatched_result) << "dim=" << dim << " trial=" << trial;
    }
  }
}

}  // namespace
}  // namespace alaya
