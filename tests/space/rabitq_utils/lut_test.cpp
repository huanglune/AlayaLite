// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "index/graph/detail/search_runtime/buffer.hpp"
#include "platform/detect.hpp"
#include "simd/cpu_features.hpp"
#include "space/quant/rabitq/dispatch.hpp"
#include "space/quant/rabitq/lut.hpp"

namespace alaya {

TEST(LutTest, SimpleExample) {
  size_t padded_dim = 64;
  std::vector<float> rotated_query(padded_dim, 1.0F);
  Lut<float> lookup_table;
  lookup_table = Lut<float>(rotated_query.data(), padded_dim);
  float delta = 4.0F / 255.0F;  // vr = 4, 4/(2^8-1) = 4/255

  EXPECT_EQ(lookup_table.delta(), delta);
  EXPECT_EQ(lookup_table.sum_vl(), 0);  // vl = 0

  EXPECT_EQ(*(lookup_table.lut() + 0), std::round(0 / delta));
  EXPECT_EQ(*(lookup_table.lut() + 1), std::round(1.0F / delta));
  EXPECT_EQ(*(lookup_table.lut() + 3), std::round(2.0F / delta));
  EXPECT_EQ(*(lookup_table.lut() + 7), std::round(3.0F / delta));
  EXPECT_EQ(*(lookup_table.lut() + 15), std::round(4.0F / delta));
}

TEST(LutTest, InvalidDataType) {
  size_t padded_dim = 64;
  std::vector<int> rotated_query(padded_dim, 1);
  EXPECT_THROW(auto lookup_table = Lut(rotated_query.data(), padded_dim), std::invalid_argument);
}

TEST(LutTest, ScalarFastScanAccumulatesUnsignedLookupValues) {
  std::array<uint8_t, fastscan::kBatchSize> codes{};
  std::array<uint8_t, fastscan::kBatchSize> lookup_table{};
  std::array<uint16_t, fastscan::kBatchSize> result{};

  lookup_table[0] = 200;
  lookup_table[16] = 250;

  fastscan::detail::accumulate_scalar(codes.data(), lookup_table.data(), result.data(), 8);

  for (const auto value : result) {
    EXPECT_EQ(value, 450);
  }
}

// These two tests used to be gated behind `#if !defined(__AVX512F__)`, because
// pre-dispatch, fastscan::estimate_distances/accumulate_and_estimate_distances
// were compile-time gated to the AVX-512 branch on an AVX-512 build, with no
// way to force the portable path in the same binary to check it against these
// hand-computed expected values. Now that both tiers are runtime-dispatched
// (space/quant/rabitq/dispatch.hpp) and live in the same translation unit
// unconditionally, we call the generic tier directly (so the expected values
// hold on every host) and add a differential check against the AVX-512 tier
// when the host supports it.
TEST(LutTest, FastScanFallbackEstimatesDistances) {
  alignas(64) std::array<uint16_t, fastscan::kBatchSize> nth_segments{};
  alignas(64) std::array<float, fastscan::kBatchSize> f_add{};
  alignas(64) std::array<float, fastscan::kBatchSize> f_rescale{};
  alignas(64) std::array<float, fastscan::kBatchSize> result{};

  for (size_t i = 0; i < fastscan::kBatchSize; ++i) {
    nth_segments[i] = 2;
    f_add[i] = static_cast<float>(i);
    f_rescale[i] = 2.0F;
  }

  rabitq_simd::detail::estimate_distances_generic(nth_segments.data(),
                                                  f_add.data(),
                                                  f_rescale.data(),
                                                  3.0F,
                                                  0.5F,
                                                  1.0F,
                                                  result.data());

  for (size_t i = 0; i < fastscan::kBatchSize; ++i) {
    EXPECT_FLOAT_EQ(result[i], static_cast<float>(i) + 7.0F);
  }

#ifdef ALAYA_ARCH_X86
  const auto &features = simd::get_cpu_features();
  if (features.avx512f_ && features.avx512bw_) {
    alignas(64) std::array<float, fastscan::kBatchSize> avx512_result{};
    rabitq_simd::detail::estimate_distances_avx512(nth_segments.data(),
                                                   f_add.data(),
                                                   f_rescale.data(),
                                                   3.0F,
                                                   0.5F,
                                                   1.0F,
                                                   avx512_result.data());
    for (size_t i = 0; i < fastscan::kBatchSize; ++i) {
      EXPECT_NEAR(avx512_result[i], result[i], 1.0e-4F) << "i=" << i;
    }
  }
#endif
}

TEST(LutTest, FastScanFallbackAccumulatesAndEstimatesDistances) {
  // dim=8 is not a multiple of 16, so this exercises
  // fastscan::accumulate_and_estimate_distances' dim-misalignment fallback
  // (unrelated to CPU-tier dispatch: that guard runs before any SIMD tier is
  // selected, both pre- and post-refactor) rather than the fused AVX-512
  // kernel, which requires dim % 16 == 0 as a precondition. Differential
  // coverage between the generic and AVX-512 tiers of the fused kernel itself
  // (valid, aligned dims) lives in rabitq_dispatch_test.cpp's
  // AccumulateAndEstimateDistancesDifferentialFuzz.
  alignas(64) std::array<uint8_t, fastscan::kBatchSize> codes{};
  alignas(64) std::array<uint8_t, fastscan::kBatchSize> lookup_table{};
  alignas(64) std::array<float, fastscan::kBatchSize> f_add{};
  alignas(64) std::array<float, fastscan::kBatchSize> f_rescale{};
  alignas(64) std::array<float, fastscan::kBatchSize> result{};

  lookup_table[0] = 4;
  lookup_table[16] = 6;
  for (size_t i = 0; i < fastscan::kBatchSize; ++i) {
    f_rescale[i] = 0.5F;
  }

  // Composed by hand exactly like the dim-misalignment fallback branch of
  // fastscan::accumulate_and_estimate_distances does (accumulate_scalar, which
  // always calls accumulate_generic, followed by the dispatched
  // estimate_distances), so this is independent of which tier the dispatcher
  // itself would select on this host for the estimate step.
  std::array<uint16_t, fastscan::kBatchSize> accumulated{};
  ::alaya::simd::fastscan::accumulate_generic(8,
                                              codes.data(),
                                              lookup_table.data(),
                                              accumulated.data());
  rabitq_simd::detail::estimate_distances_generic(accumulated.data(),
                                                  f_add.data(),
                                                  f_rescale.data(),
                                                  1.0F,
                                                  2.0F,
                                                  3.0F,
                                                  result.data());

  for (const auto value : result) {
    EXPECT_FLOAT_EQ(value, 12.5F);
  }

  // Differential: the public (dispatched) API must agree with the hand-composed
  // generic oracle above when it takes the same dim-misalignment path.
  alignas(64) std::array<float, fastscan::kBatchSize> dispatched_result{};
  fastscan::accumulate_and_estimate_distances(codes.data(),
                                              lookup_table.data(),
                                              f_add.data(),
                                              f_rescale.data(),
                                              1.0F,
                                              2.0F,
                                              3.0F,
                                              dispatched_result.data(),
                                              8);
  for (size_t i = 0; i < fastscan::kBatchSize; ++i) {
    EXPECT_NEAR(dispatched_result[i], result[i], 1.0e-4F) << "i=" << i;
  }
}

TEST(SearchBufferTest, HandlesEmptyResizeAndSortedInsertion) {
  SearchBuffer<float> buffer;

  EXPECT_FALSE(buffer.insert(1, 1.0F));
  EXPECT_EQ(buffer.top_dist(), std::numeric_limits<float>::max());
  EXPECT_FALSE(buffer.has_next());

  buffer.resize(2);
  EXPECT_TRUE(buffer.insert(2, 2.0F));
  EXPECT_TRUE(buffer.insert(1, 1.0F));
  EXPECT_FALSE(buffer.insert(3, 3.0F));
  EXPECT_TRUE(buffer.is_full());
  EXPECT_FLOAT_EQ(buffer.top_dist(), 2.0F);

  std::array<uint32_t, 2> results{};
  buffer.copy_results_to(results.data(), results.size());
  EXPECT_EQ(results[0], 1U);
  EXPECT_EQ(results[1], 2U);

  EXPECT_TRUE(buffer.has_next());
  EXPECT_EQ(buffer.next_id(), 1U);
  EXPECT_EQ(buffer.pop(), 1U);
  EXPECT_TRUE(buffer.has_next());
  EXPECT_EQ(buffer.pop(), 2U);
  EXPECT_FALSE(buffer.has_next());

  buffer.clear();
  EXPECT_EQ(buffer.size(), 0U);
  EXPECT_EQ(buffer.top_dist(), std::numeric_limits<float>::max());
}

}  // namespace alaya
