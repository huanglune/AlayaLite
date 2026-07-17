// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file ip_kernel_test.cpp
 * @brief W1 unit test for the LASER inner-product (IP) distance kernel
 *        (include/index/graph/laser/space/ip.hpp).
 *
 * ip.hpp is DORMANT (included by no production TU); this is its sole consumer. It
 * proves, against the W0 oracle base (ip_oracle_test.cpp + the immutable official
 * fixture), that:
 *   - the full-sign LASER IP factors assembled into the single-level estimate are
 *     point-equivalent to the memqg path and the official METRIC_IP estimate;
 *   - laser factor_dq == official f_rescale / 2 (the half-sign vs full-sign factor
 *     of 2), including the codex 1-D counterexample (c=2,o=3,q=4 -> -11, not -15);
 *   - factor_vq = factor_dq * sum(s) correctly captures the query-quantization vl
 *     offset;
 *   - the zero-residual policy is the NaN-free {base=1, dq=0, vq=0};
 *   - the exact ip() wrapper agrees across generic / AVX2 / AVX-512 within
 *     tolerance and equals -<a,b>.
 *
 * All value comparisons use tolerance (no float bitwise equality; B-LIP-03).
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "core/value_types.hpp"
#include "index/graph/laser/space/ip.hpp"  // the dormant kernel under test
#include "ip_fixture_common.hpp"
#include "simd/cpu_features.hpp"
#include "simd/distance_ip.hpp"
#include "space/quant/rabitq_core.hpp"

namespace alaya {
namespace {

using namespace ip_fixture;  // NOLINT(build/namespaces)

auto random_vec(std::mt19937 &rng, size_t dim, float lo, float hi) -> std::vector<float> {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(dim);
  for (auto &x : v) {
    x = dist(rng);
  }
  return v;
}

const std::vector<size_t> kDims = {64, 128, 192, 256, 1024};

// Assembles residual + sign bits for a (data, centroid) pair.
void residual_and_bits(const std::vector<float> &data, const std::vector<float> &centroid,
                       std::vector<float> *residual, std::vector<int> *bits) {
  const size_t dim = data.size();
  residual->resize(dim);
  bits->resize(dim);
  for (size_t i = 0; i < dim; ++i) {
    (*residual)[i] = data[i] - centroid[i];
    (*bits)[i] = (*residual)[i] > 0.0F ? 1 : 0;
  }
}

}  // namespace

// ===========================================================================
// W1-a -- estimate equivalence: LASER IP factors, assembled into the single-level
// estimate g_add + base + factor_dq*<s,q>, equal the memqg estimate
// g_add + f_add + f_rescale*<h,q> AND the algebra estimate, with identical codes.
// g_add = -<q,c> is taken from the exact ip() wrapper (dogfooding it).
// ===========================================================================
TEST(LaserIpKernel, EstimateMatchesMemqgAndAlgebra) {
  std::mt19937 rng(0x1A2B3C4DU);
  for (size_t dim : kDims) {
    const float fac_norm = 1.0F / std::sqrt(static_cast<float>(dim));
    for (int trial = 0; trial < 24; ++trial) {
      const auto data = random_vec(rng, dim, -3.0F, 3.0F);
      const auto centroid = random_vec(rng, dim, -3.0F, 3.0F);
      const auto query = random_vec(rng, dim, -3.0F, 3.0F);

      std::vector<float> residual;
      std::vector<int> bits;
      residual_and_bits(data, centroid, &residual, &bits);

      std::vector<int> mem_bits(dim);
      const auto mem = RaBitQCore::memory_factors<float>(
          data.data(), centroid.data(), dim, mem_bits.data(), core::Metric::inner_product);
      std::vector<int> alg_bits;
      const auto alg = algebra_ip_factors(data.data(), centroid.data(), dim, &alg_bits);
      const auto lz = laser::space::laser_ip_factors(residual.data(), centroid.data(), bits.data(),
                                                     static_cast<int64_t>(dim), fac_norm);

      EXPECT_EQ(bits, mem_bits) << "dim=" << dim << " trial=" << trial;
      EXPECT_EQ(bits, alg_bits) << "dim=" << dim << " trial=" << trial;

      const float g_add = laser::space::ip(query.data(), centroid.data(), dim);  // -<q,c>
      float memory_inner = 0.0F;  // <half_signed, q>
      float laser_inner = 0.0F;   // <signed_x, q> == 2 * memory_inner
      for (size_t i = 0; i < dim; ++i) {
        const float sign = bits[i] != 0 ? 1.0F : -1.0F;
        memory_inner += 0.5F * sign * query[i];
        laser_inner += sign * query[i];
      }
      const float mem_est = g_add + mem.base + (mem.signed_query_scale * memory_inner);
      const float alg_est = g_add + alg.f_add + (alg.f_rescale * memory_inner);
      const float laser_est = g_add + lz.base + (lz.signed_query_scale * laser_inner);

      const float tol = 5.0e-4F * std::max(1.0F, std::abs(mem_est));
      EXPECT_NEAR(laser_est, mem_est, tol) << "dim=" << dim << " trial=" << trial;
      EXPECT_NEAR(laser_est, alg_est, tol) << "dim=" << dim << " trial=" << trial;
    }
  }
}

// ===========================================================================
// W1-b -- shape lock: with q == c the LASER IP estimate collapses to exactly the
// score-domain target 1 - <c,o> (zero estimator slack).
// ===========================================================================
TEST(LaserIpKernel, EstimateLocksToOneMinusDotAtQEqualsC) {
  std::mt19937 rng(0x5EED01U);
  for (size_t dim : kDims) {
    const float fac_norm = 1.0F / std::sqrt(static_cast<float>(dim));
    for (int trial = 0; trial < 12; ++trial) {
      const auto data = random_vec(rng, dim, -3.0F, 3.0F);
      const auto centroid = random_vec(rng, dim, -3.0F, 3.0F);
      std::vector<float> residual;
      std::vector<int> bits;
      residual_and_bits(data, centroid, &residual, &bits);
      const auto lz = laser::space::laser_ip_factors(residual.data(), centroid.data(), bits.data(),
                                                     static_cast<int64_t>(dim), fac_norm);
      const float g_add = laser::space::ip(centroid.data(), centroid.data(), dim);  // -<c,c>
      float laser_inner = 0.0F;
      for (size_t i = 0; i < dim; ++i) {
        laser_inner += (bits[i] != 0 ? 1.0F : -1.0F) * centroid[i];
      }
      const float est = g_add + lz.base + (lz.signed_query_scale * laser_inner);
      const float expected = 1.0F - static_cast<float>(exact_ip(centroid.data(), data.data(), dim));
      EXPECT_NEAR(est, expected, 2.0e-3F * std::max(1.0F, std::abs(expected)))
          << "dim=" << dim << " trial=" << trial;
    }
  }
}

// ===========================================================================
// W1-c -- FACTOR-OF-2 LOCK against the immutable official fixture (B-LIP-02). For
// every fixture row: reproduce inputs, compute LASER IP factors, and assert
//   laser.base              ~= official f_add        (unchanged; both = 1-<r,c>+...)
//   laser.signed_query_scale ~= official f_rescale / 2 (HALVED: full sign vs half)
// Ties W1 directly to the real RaBitQ-Library output, not just to in-repo math.
// ===========================================================================
TEST(LaserIpKernel, FactorDqIsHalfOfficialRescale) {
  std::string err;
  const std::vector<FixtureRow> rows = load_ip_fixture(&err);
  ASSERT_TRUE(err.empty()) << err;
  ASSERT_FALSE(rows.empty());

  for (const auto &row : rows) {
    SCOPED_TRACE(::testing::Message() << "dim=" << row.dim << " trial=" << row.trial);
    const RowInputs in = make_row_inputs(row.seed, row.dim);
    ASSERT_EQ(input_fingerprint(in.data, in.centroid), row.input_fnv);

    std::vector<float> residual;
    std::vector<int> bits;
    residual_and_bits(in.data, in.centroid, &residual, &bits);
    const float fac_norm = 1.0F / std::sqrt(static_cast<float>(row.dim));
    const auto lz = laser::space::laser_ip_factors(residual.data(), in.centroid.data(), bits.data(),
                                                   static_cast<int64_t>(row.dim), fac_norm);

    EXPECT_NEAR(lz.base, row.f_add, 1.0e-3F * std::max(1.0F, std::abs(row.f_add)));
    EXPECT_NEAR(lz.signed_query_scale, row.f_rescale / 2.0F,
                1.0e-3F * std::max(1.0F, std::abs(row.f_rescale / 2.0F)));
  }
}

// ===========================================================================
// W1-d -- the codex 1-D counterexample, made executable. c=2, o=3, q=4: the
// correct full-sign estimate is -11; verbatim-copying the official half-sign
// rescale (-2A) without halving gives -15. Both are asserted, and shown distinct.
// ===========================================================================
TEST(LaserIpKernel, OneDimensionalCounterexample) {
  const float c = 2.0F;
  const float o = 3.0F;
  const float q = 4.0F;
  const float residual = o - c;  // 1
  const int bit = residual > 0.0F ? 1 : 0;
  const auto lz = laser::space::laser_ip_factors(&residual, &c, &bit, 1, 1.0F);

  EXPECT_NEAR(lz.base, 1.0F, 1.0e-5F);                 // 1 - <r,c> + A<c,s> = 1-2+2
  EXPECT_NEAR(lz.signed_query_scale, -1.0F, 1.0e-5F);  // factor_dq = -A = -1

  const float g_add = -(q * c);                       // -8
  const float sign_dot_q = static_cast<float>((2 * bit) - 1) * q;  // <s,q> = +4
  const float correct = g_add + lz.base + (lz.signed_query_scale * sign_dot_q);
  EXPECT_NEAR(correct, -11.0F, 1.0e-4F);
  EXPECT_NEAR(correct, 1.0F - (q * o), 1.0e-4F);  // == 1 - <q,o>

  // Verbatim-copy bug: reuse the official half-sign rescale (-2A) with the full
  // sign. -2A = 2*factor_dq.
  const float buggy_scale = 2.0F * lz.signed_query_scale;  // -2
  const float buggy = g_add + lz.base + (buggy_scale * sign_dot_q);
  EXPECT_NEAR(buggy, -15.0F, 1.0e-4F);
  EXPECT_GT(std::abs(buggy - correct), 1.0F);  // the bug is a real, detectable divergence
}

// ===========================================================================
// W1-e -- factor_vq mapping. Under fastscan query quantization q_i = vl + delta*qcode_i,
// the scanner reconstructs factor_dq*<s,q> as
//     factor_dq * delta * sum(s_i*qcode_i)  +  factor_vq * vl,
// with factor_vq = factor_dq * sum(s) (rabitq.hpp:119, laser_dispatch.hpp:118).
// This proves factor_vq captures the vl offset exactly.
// ===========================================================================
TEST(LaserIpKernel, FactorVqCapturesQuantizationOffset) {
  std::mt19937 rng(0xF00DBEEFU);
  for (size_t dim : {64UL, 128UL, 256UL}) {
    const float fac_norm = 1.0F / std::sqrt(static_cast<float>(dim));
    const auto data = random_vec(rng, dim, -2.0F, 2.0F);
    const auto centroid = random_vec(rng, dim, -2.0F, 2.0F);
    std::vector<float> residual;
    std::vector<int> bits;
    residual_and_bits(data, centroid, &residual, &bits);
    const auto lz = laser::space::laser_ip_factors(residual.data(), centroid.data(), bits.data(),
                                                   static_cast<int64_t>(dim), fac_norm);

    int popcount = 0;
    for (int b : bits) {
      popcount += b;
    }
    const int sum_s = (2 * popcount) - static_cast<int>(dim);        // sum(s_i)
    const float factor_vq = lz.signed_query_scale * static_cast<float>(sum_s);

    // Deterministic fastscan-style query quantization.
    const float vl = -0.7F;
    const float delta = 0.05F;
    std::uniform_int_distribution<int> qcode_dist(0, 15);
    float exact = 0.0F;         // factor_dq * <s,q>
    float reconstructed = 0.0F;  // factor_dq*delta*sum(s*qcode) + factor_vq*vl
    float sum_s_qcode = 0.0F;
    std::vector<float> q(dim);
    for (size_t i = 0; i < dim; ++i) {
      const int qcode = qcode_dist(rng);
      q[i] = vl + (delta * static_cast<float>(qcode));
      const float s = static_cast<float>((2 * bits[i]) - 1);
      sum_s_qcode += s * static_cast<float>(qcode);
      exact += s * q[i];
    }
    exact *= lz.signed_query_scale;
    reconstructed = (lz.signed_query_scale * delta * sum_s_qcode) + (factor_vq * vl);

    EXPECT_NEAR(reconstructed, exact, 1.0e-3F * std::max(1.0F, std::abs(exact))) << "dim=" << dim;
  }
}

// ===========================================================================
// W1-f -- zero-residual policy (o == c). The IP kernel returns the strict, NaN-free
// {base=1, factor_dq=0}, so factor_vq=0 too. This DIVERGES from laser_l2_factors,
// which deliberately retains NaN (B-LIP-09). Also confirms the estimate is exactly
// the target 1 - <q,c>.
// ===========================================================================
TEST(LaserIpKernel, ZeroResidualPolicyIsNanFree) {
  std::mt19937 rng(0xABCDEF01U);
  for (size_t dim : {64UL, 128UL, 256UL}) {
    const float fac_norm = 1.0F / std::sqrt(static_cast<float>(dim));
    const auto centroid = random_vec(rng, dim, -2.0F, 2.0F);
    const std::vector<float> residual(dim, 0.0F);  // o == c
    std::vector<int> bits(dim, 0);                  // [0 > 0] == 0
    const auto lz = laser::space::laser_ip_factors(residual.data(), centroid.data(), bits.data(),
                                                   static_cast<int64_t>(dim), fac_norm);
    EXPECT_EQ(lz.base, 1.0F) << "dim=" << dim;
    EXPECT_EQ(lz.signed_query_scale, 0.0F) << "dim=" << dim;
    EXPECT_FALSE(std::isnan(lz.base));
    EXPECT_FALSE(std::isnan(lz.signed_query_scale));

    int popcount = 0;  // 0
    const int sum_s = (2 * popcount) - static_cast<int>(dim);
    const float factor_vq = lz.signed_query_scale * static_cast<float>(sum_s);
    EXPECT_EQ(factor_vq, 0.0F);

    // Estimate at an arbitrary query collapses to the exact target 1 - <q,c>.
    const auto query = random_vec(rng, dim, -2.0F, 2.0F);
    const float g_add = laser::space::ip(query.data(), centroid.data(), dim);  // -<q,c>
    const float est = g_add + lz.base + (lz.signed_query_scale * 0.0F);
    const float expected = 1.0F - static_cast<float>(exact_ip(query.data(), centroid.data(), dim));
    EXPECT_NEAR(est, expected, 1.0e-3F * std::max(1.0F, std::abs(expected)));
  }
}

// ===========================================================================
// W1-g -- ISA coverage for the exact ip() wrapper: generic / AVX2 / AVX-512 all
// agree with -<a,b> within tolerance (no cross-ISA bitwise equality required).
// AVX paths are guarded by runtime CPU feature detection.
// ===========================================================================
TEST(LaserIpKernel, ExactWrapperAcrossIsaLevels) {
  std::mt19937 rng(0x15A15A15U);
  for (size_t dim : {65UL, 128UL, 192UL, 257UL}) {  // incl. non-multiple-of-8 tails
    const auto a = random_vec(rng, dim, -3.0F, 3.0F);
    const auto b = random_vec(rng, dim, -3.0F, 3.0F);
    const float want = -static_cast<float>(exact_ip(a.data(), b.data(), dim));
    const float tol = 1.0e-3F * std::max(1.0F, std::abs(want));

    // Dispatched wrapper + explicit generic.
    EXPECT_NEAR(laser::space::ip(a.data(), b.data(), dim), want, tol) << "wrapper dim=" << dim;
    EXPECT_NEAR(simd::ip_sqr_generic(a.data(), b.data(), dim), want, tol) << "generic dim=" << dim;

#ifdef ALAYA_ARCH_X86
    const auto &features = simd::get_cpu_features();
    if (features.avx2_ && features.fma_) {
      EXPECT_NEAR(simd::ip_sqr_avx2(a.data(), b.data(), dim), want, tol) << "avx2 dim=" << dim;
    }
    if (features.avx512f_) {
      EXPECT_NEAR(simd::ip_sqr_avx512(a.data(), b.data(), dim), want, tol) << "avx512 dim=" << dim;
    }
#endif
  }
}

}  // namespace alaya
