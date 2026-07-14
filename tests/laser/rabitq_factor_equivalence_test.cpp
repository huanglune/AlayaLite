// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "index/graph/laser/quantization/rabitq.hpp"
#include "space/quant/rabitq.hpp"
#include "core/metric_type.hpp"

namespace alaya {
namespace {

constexpr size_t kNumPoints = 32;

void expect_near_scaled(float actual, float expected, float relative_tolerance = 4e-5F) {
  const float scale = std::max({1.0F, std::abs(actual), std::abs(expected)});
  EXPECT_LE(std::abs(actual - expected), relative_tolerance * scale)
      << "actual=" << actual << " expected=" << expected;
}

std::vector<float> make_centroid(size_t dim) {
  std::mt19937 rng(0xC37A01D);
  std::normal_distribution<float> normal(0.0F, 0.7F);
  std::vector<float> centroid(dim);
  std::generate(centroid.begin(), centroid.end(), [&] { return normal(rng); });
  return centroid;
}

std::vector<float> make_residuals(size_t dim) {
  std::mt19937 rng(static_cast<uint32_t>(0xA11CE + dim));
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::uniform_real_distribution<float> uniform(-2.0F, 2.0F);
  std::vector<float> residuals(kNumPoints * dim);

  for (size_t row = 0; row < kNumPoints; ++row) {
    for (size_t col = 0; col < dim; ++col) {
      float value = 0.0F;
      switch (row % 6) {
        case 0:
          value = normal(rng);
          break;
        case 1:
          value = uniform(rng);
          break;
        case 2:
          value = std::exp(uniform(rng)) * (col % 2 == 0 ? 1.0F : -1.0F);
          break;
        case 3:
          value = col % 17 == 0 ? normal(rng) : 0.0F;
          break;
        case 4:
          value = std::abs(normal(rng)) + 1e-4F;  // all signs equal
          break;
        default:
          value = normal(rng) * 1e-5F;  // denominator tends to zero but survives float addition
          break;
      }
      residuals[row * dim + col] = value;
    }
  }
  return residuals;
}

TEST(RaBitQFactorEquivalence, L2FactorsCodesAndEstimatorsAgree) {
  for (const size_t dim : {64U, 128U, 256U}) {
    const std::vector<float> centroid = make_centroid(dim);
    const std::vector<float> residuals = make_residuals(dim);
    std::vector<float> data(residuals.size());
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = residuals[i] + centroid[i % dim];
    }

    std::vector<uint8_t> memory_codes(kNumPoints * dim / 8);
    std::vector<float> f_add(kNumPoints);
    std::vector<float> f_rescale(kNumPoints);
    RaBitQQuantizer<float> memory_quantizer(static_cast<uint32_t>(dim),
                                            static_cast<uint32_t>(dim));
    memory_quantizer.batch_quantize(data.data(),
                                    centroid.data(),
                                    kNumPoints,
                                    memory_codes.data(),
                                    f_add.data(),
                                    f_rescale.data(),
                                    MetricType::L2);

    laser::RowMatrix<float> laser_data(kNumPoints, static_cast<int64_t>(dim));
    laser::RowMatrix<float> laser_centroid(1, static_cast<int64_t>(dim));
    std::copy(data.begin(), data.end(), laser_data.data());
    std::copy(centroid.begin(), centroid.end(), laser_centroid.data());
    std::vector<uint8_t> laser_codes(kNumPoints * dim / 8);
    std::vector<float> triple_x(kNumPoints);
    std::vector<float> factor_dq(kNumPoints);
    std::vector<float> factor_vq(kNumPoints);
    laser::rabitq_codes(laser_data,
                        laser_centroid,
                        laser_codes.data(),
                        triple_x.data(),
                        factor_dq.data(),
                        factor_vq.data());

    EXPECT_EQ(memory_codes, laser_codes) << "dim=" << dim;

    std::mt19937 query_rng(static_cast<uint32_t>(0x0BADC0DE + dim));
    std::normal_distribution<float> query_dist(0.0F, 1.0F);
    std::vector<float> query(dim);
    std::generate(query.begin(), query.end(), [&] { return query_dist(query_rng); });
    float centroid_distance = 0.0F;
    for (size_t col = 0; col < dim; ++col) {
      const float delta = query[col] - centroid[col];
      centroid_distance += delta * delta;
    }

    for (size_t row = 0; row < kNumPoints; ++row) {
      SCOPED_TRACE(::testing::Message() << "dim=" << dim << " row=" << row);
      int popcount = 0;
      float memory_inner = 0.0F;
      float laser_inner = 0.0F;
      for (size_t col = 0; col < dim; ++col) {
        // rabitq_codes mutates laser_data to the actual float residual. Use that value because
        // data = centroid + residual can round a tiny component back onto the centroid.
        const bool bit = laser_data(static_cast<int64_t>(row), static_cast<int64_t>(col)) > 0.0F;
        popcount += static_cast<int>(bit);
        const float sign = bit ? 1.0F : -1.0F;
        memory_inner += 0.5F * sign * query[col];
        laser_inner += sign * query[col];
      }

      // The centroid correction is ill-conditioned as the residual tends to zero. The two
      // Eigen reduction orders can then differ by O(1e-5) in absolute terms even though the
      // complete estimate remains stable and agrees below.
      {
        SCOPED_TRACE("triple_x");
        expect_near_scaled(triple_x[row], f_add[row], row % 6 == 5 ? 1e-4F : 4e-5F);
      }
      {
        SCOPED_TRACE("factor_dq");
        expect_near_scaled(factor_dq[row], 0.5F * f_rescale[row]);
      }
      {
        SCOPED_TRACE("factor_vq");
        expect_near_scaled(
            factor_vq[row],
            factor_dq[row] * static_cast<float>(2 * popcount - static_cast<int>(dim)));
      }

      const float memory_estimate =
          centroid_distance + f_add[row] + f_rescale[row] * memory_inner;
      const float laser_estimate =
          centroid_distance + triple_x[row] + factor_dq[row] * laser_inner;
      {
        SCOPED_TRACE("complete estimator");
        expect_near_scaled(laser_estimate, memory_estimate, 8e-5F);
      }
    }
  }
}

TEST(RaBitQFactorEquivalence, ZeroResidualHasDifferentNonFinitePolicy) {
  constexpr size_t dim = 64;
  const std::vector<float> centroid = make_centroid(dim);
  std::vector<float> data(kNumPoints * dim);
  for (size_t row = 0; row < kNumPoints; ++row) {
    std::copy(centroid.begin(), centroid.end(), data.begin() + row * dim);
  }

  std::vector<uint8_t> memory_codes(kNumPoints * dim / 8);
  std::vector<float> f_add(kNumPoints);
  std::vector<float> f_rescale(kNumPoints);
  RaBitQQuantizer<float> memory_quantizer(dim, dim);
  memory_quantizer.batch_quantize(data.data(),
                                  centroid.data(),
                                  kNumPoints,
                                  memory_codes.data(),
                                  f_add.data(),
                                  f_rescale.data(),
                                  MetricType::L2);

  laser::RowMatrix<float> laser_data(kNumPoints, dim);
  laser::RowMatrix<float> laser_centroid(1, dim);
  std::copy(data.begin(), data.end(), laser_data.data());
  std::copy(centroid.begin(), centroid.end(), laser_centroid.data());
  std::vector<uint8_t> laser_codes(kNumPoints * dim / 8);
  std::vector<float> triple_x(kNumPoints);
  std::vector<float> factor_dq(kNumPoints);
  std::vector<float> factor_vq(kNumPoints);
  laser::rabitq_codes(laser_data,
                      laser_centroid,
                      laser_codes.data(),
                      triple_x.data(),
                      factor_dq.data(),
                      factor_vq.data());

  EXPECT_EQ(memory_codes, laser_codes);
  for (size_t row = 0; row < kNumPoints; ++row) {
    EXPECT_EQ(f_add[row], 0.0F);
    EXPECT_EQ(f_rescale[row], 0.0F);
    constexpr uint32_t kExponentMask = 0x7F800000U;
    constexpr uint32_t kMantissaMask = 0x007FFFFFU;
    const auto has_nan_bits = [](float value) {
      const uint32_t bits = std::bit_cast<uint32_t>(value);
      return (bits & kExponentMask) == kExponentMask && (bits & kMantissaMask) != 0;
    };
    EXPECT_TRUE(has_nan_bits(triple_x[row]));
    EXPECT_TRUE(has_nan_bits(factor_dq[row]));
    EXPECT_TRUE(has_nan_bits(factor_vq[row]));
  }
}

}  // namespace
}  // namespace alaya
