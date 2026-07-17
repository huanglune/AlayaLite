// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file ip_kernel_test.cpp
 * @brief W1 skeleton for the LASER inner-product (IP) distance kernel.
 *
 * STATUS: skeleton. The production kernel
 * include/index/graph/laser/space/ip.hpp is gated on the codex review
 * (manifest W1 review gate). Until it lands, the LASER-convention IP factor
 * formula lives here as reference_laser_ip_factors() -- the derivation recorded
 * in REPORT-laser-ip.md section 3. When ip.hpp lands:
 *   - replace reference_laser_ip_factors() with a call to
 *     alaya::laser::space::laser_ip_factors(...),
 *   - replace reference_ip_exact() with alaya::laser::space::ip(...),
 * and this file becomes the real W1 unit test with zero structural change.
 *
 * What it proves (against the W0 oracle base in ip_oracle_test.cpp): the LASER
 * IP factor convention, assembled into the single-level RaBitQ estimate, is
 * point-equivalent to the memqg path (RaBitQCore::memory_factors + the
 * batch_quantize IP rescale convention) and the official METRIC_IP estimate,
 * and produces the identical binary code. This is the estimate-level
 * equivalence range that side-steps the factor-2 convention gap between the
 * LASER (signed_x = +-1) and memory (half_signed = +-0.5) inner products --
 * see REPORT section 2 (J5) and tests/laser/rabitq_factor_equivalence_test.cpp
 * for the L2 analog this mirrors.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "core/value_types.hpp"
#include "space/quant/rabitq_core.hpp"

namespace alaya {
namespace {

// kConstEpsilon, RaBitQ-Library rabitq_impl.hpp:17.
constexpr double kOfficialEpsilon = 1.9;

auto exact_ip(const float *a, const float *b, size_t dim) -> double {
  double s = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  return s;
}

// Official METRIC_IP factor reference (rabitq_impl.hpp:80-136), float precision.
struct OfficialIpFactors {
  float f_add;
  float f_rescale;
  float f_error;
};

auto official_ip_factors(const float *data, const float *centroid, size_t dim,
                         std::vector<int> *bits_out) -> OfficialIpFactors {
  std::vector<float> residual(dim);
  for (size_t i = 0; i < dim; ++i) {
    residual[i] = data[i] - centroid[i];
  }
  std::vector<int> bits(dim);
  for (size_t i = 0; i < dim; ++i) {
    bits[i] = residual[i] > 0.0F ? 1 : 0;
  }
  const float cb = -0.5F;
  std::vector<float> xu_cb(dim);
  for (size_t i = 0; i < dim; ++i) {
    xu_cb[i] = static_cast<float>(bits[i]) + cb;
  }
  float l2_sqr = 0, ip_resi_xucb = 0, ip_cent_xucb = 0, resi_cent = 0, xucb_sqr = 0;
  for (size_t i = 0; i < dim; ++i) {
    l2_sqr += residual[i] * residual[i];
    ip_resi_xucb += residual[i] * xu_cb[i];
    ip_cent_xucb += centroid[i] * xu_cb[i];
    resi_cent += residual[i] * centroid[i];
    xucb_sqr += xu_cb[i] * xu_cb[i];
  }
  const float l2_norm = std::sqrt(l2_sqr);
  if (ip_resi_xucb == 0.0F) {
    ip_resi_xucb = std::numeric_limits<float>::infinity();
  }
  const float tmp_error =
      l2_norm * static_cast<float>(kOfficialEpsilon) *
      std::sqrt((((l2_sqr * xucb_sqr) / (ip_resi_xucb * ip_resi_xucb)) - 1.0F) /
                (static_cast<float>(dim) - 1.0F));
  OfficialIpFactors f{};
  f.f_add = 1.0F - resi_cent + (l2_sqr * ip_cent_xucb / ip_resi_xucb);
  f.f_rescale = -l2_sqr / ip_resi_xucb;
  f.f_error = 1.0F * tmp_error;
  if (bits_out != nullptr) {
    *bits_out = bits;
  }
  return f;
}

// ===========================================================================
// REFERENCE LASER-convention IP factors (REPORT section 3). Mirrors
// RaBitQCore::laser_l2_factors (rabitq_core.hpp:128-155) exactly, with the two
// L2->IP substitutions:
//   base:  cur_x^2 + 2*x_x0*nx1   ->  (1 - <residual,centroid>) + 1*x_x0*nx1
//   scale: -2*x_x0*fac_norm       ->  -1*x_x0*fac_norm            (i.e. halved)
// Same signature as laser_l2_factors so ip.hpp can drop in verbatim.
// PENDING codex review confirmation of the numeric form (J5).
// ===========================================================================
auto reference_laser_ip_factors(const float *residual, const float *centroid, const int *bits,
                                int64_t dim, float fac_norm) -> RaBitQCoreFactors<float> {
  double fac_x0_num = 0.0;   // sum(residual * signed_x) * fac_norm
  double cent_sx = 0.0;      // sum(centroid * signed_x)
  double r_norm_sqr = 0.0;   // ||residual||^2
  double resi_cent = 0.0;    // <residual, centroid>
  for (int64_t i = 0; i < dim; ++i) {
    const double sx = 2.0 * static_cast<double>(bits[i]) - 1.0;  // signed_x in {-1,+1}
    fac_x0_num += static_cast<double>(residual[i]) * sx;
    cent_sx += static_cast<double>(centroid[i]) * sx;
    r_norm_sqr += static_cast<double>(residual[i]) * static_cast<double>(residual[i]);
    resi_cent += static_cast<double>(residual[i]) * static_cast<double>(centroid[i]);
  }
  const double x_rotated_norm = std::sqrt(r_norm_sqr);
  const double normalized_x0 = (fac_x0_num * fac_norm) / x_rotated_norm;
  const double normalized_x1 = cent_sx * fac_norm;
  const double x_x0 = x_rotated_norm / normalized_x0;
  return {
      static_cast<float>((1.0 - resi_cent) + (x_x0 * normalized_x1)),  // base (IP): single cross term
      static_cast<float>(-1.0 * x_x0 * fac_norm),                      // scale (IP): half of L2
  };
}

auto random_vec(std::mt19937 &rng, size_t dim, float lo, float hi) -> std::vector<float> {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(dim);
  for (auto &x : v) {
    x = dist(rng);
  }
  return v;
}

const std::vector<size_t> kDims = {64, 128, 192, 256, 1024};

}  // namespace

// ===========================================================================
// W1-a -- estimate equivalence: LASER IP factors, assembled into the
// single-level estimate g_add + base + scale*<signed_x,q>, equal the memqg IP
// estimate g_add + f_add + f_rescale*<half_signed,q> AND the official estimate,
// and the binary codes are identical. This is the core W1 correctness claim.
// ===========================================================================
TEST(LaserIpKernelSkeleton, EstimateMatchesMemqgAndOfficial) {
  std::mt19937 rng(0x1A2B3C4DU);
  for (size_t dim : kDims) {
    const float fac_norm = 1.0F / std::sqrt(static_cast<float>(dim));
    for (int trial = 0; trial < 24; ++trial) {
      const auto data = random_vec(rng, dim, -3.0F, 3.0F);
      const auto centroid = random_vec(rng, dim, -3.0F, 3.0F);
      const auto query = random_vec(rng, dim, -3.0F, 3.0F);

      std::vector<float> residual(dim);
      std::vector<int> bits(dim);
      for (size_t i = 0; i < dim; ++i) {
        residual[i] = data[i] - centroid[i];
        bits[i] = residual[i] > 0.0F ? 1 : 0;
      }

      // Memqg IP factors (raw memory_factors; batch_quantize applies 1x for IP,
      // i.e. no rescale doubling -- rabitq.hpp:82).
      std::vector<int> mem_bits(dim);
      const auto mem = RaBitQCore::memory_factors<float>(
          data.data(), centroid.data(), dim, mem_bits.data(), core::Metric::inner_product);

      // Official IP factors.
      std::vector<int> off_bits;
      const auto off = official_ip_factors(data.data(), centroid.data(), dim, &off_bits);

      // Reference LASER IP factors.
      const auto lz =
          reference_laser_ip_factors(residual.data(), centroid.data(), bits.data(),
                                     static_cast<int64_t>(dim), fac_norm);

      // Codes: all three agree on the sign bits.
      EXPECT_EQ(bits, mem_bits) << "dim=" << dim << " trial=" << trial;
      EXPECT_EQ(bits, off_bits) << "dim=" << dim << " trial=" << trial;

      // Assemble the single-level estimate each way. g_add = -<q,c> (IP).
      const float g_add = -static_cast<float>(exact_ip(query.data(), centroid.data(), dim));
      float memory_inner = 0.0F;  // <half_signed, q>
      float laser_inner = 0.0F;   // <signed_x, q> == 2 * memory_inner
      for (size_t i = 0; i < dim; ++i) {
        const float sign = bits[i] != 0 ? 1.0F : -1.0F;
        memory_inner += 0.5F * sign * query[i];
        laser_inner += sign * query[i];
      }
      const float mem_est = g_add + mem.base + (mem.signed_query_scale * memory_inner);
      const float off_est = g_add + off.f_add + (off.f_rescale * memory_inner);
      const float laser_est = g_add + lz.base + (lz.signed_query_scale * laser_inner);

      const float tol = 5.0e-4F * std::max(1.0F, std::abs(mem_est));
      EXPECT_NEAR(laser_est, mem_est, tol) << "dim=" << dim << " trial=" << trial;
      EXPECT_NEAR(laser_est, off_est, tol) << "dim=" << dim << " trial=" << trial;
    }
  }
}

// ===========================================================================
// W1-b -- shape lock: with q == c the LASER IP estimate collapses to exactly
// 1 - <c,o> (zero estimator slack), pinning the affine target form.
// ===========================================================================
TEST(LaserIpKernelSkeleton, EstimateLocksToOneMinusDotAtQEqualsC) {
  std::mt19937 rng(0x5EED01U);
  for (size_t dim : kDims) {
    const float fac_norm = 1.0F / std::sqrt(static_cast<float>(dim));
    for (int trial = 0; trial < 12; ++trial) {
      const auto data = random_vec(rng, dim, -3.0F, 3.0F);
      const auto centroid = random_vec(rng, dim, -3.0F, 3.0F);
      std::vector<float> residual(dim);
      std::vector<int> bits(dim);
      for (size_t i = 0; i < dim; ++i) {
        residual[i] = data[i] - centroid[i];
        bits[i] = residual[i] > 0.0F ? 1 : 0;
      }
      const auto lz =
          reference_laser_ip_factors(residual.data(), centroid.data(), bits.data(),
                                     static_cast<int64_t>(dim), fac_norm);
      // q = centroid.
      const float g_add = -static_cast<float>(exact_ip(centroid.data(), centroid.data(), dim));
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

}  // namespace alaya
