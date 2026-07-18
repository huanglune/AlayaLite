// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file ip.hpp
 * @brief LASER inner-product (IP) distance kernel: exact wrapper + full-sign
 *        RaBitQ IP factor helper.
 *
 * DORMANT (W1). This header is intentionally included by NO production
 * translation unit -- only by tests/laser/space/ip_kernel_test.cpp. The IP metric
 * is wired into the production estimator chain (rabitq.hpp / qg_scanner.hpp /
 * laser_dispatch.hpp / qg.hpp / searcher / segment / importer / target build+open)
 * only in the post-WAL-2C atomic phase (see REPORT-laser-ip.md section 6). Landing
 * this kernel alone cannot produce a physically-IP LASER segment, so it stays off
 * the production include graph until that phase.
 *
 * Two pieces:
 *   1. ip(): the exact distance wrapper. Returns the NEGATIVE inner product
 *      -<vec0, vec1> (AlayaLite distance convention, "smaller == more similar";
 *      distance_ip.ipp:39). This is NOT a squared quantity, despite forwarding to
 *      simd::ip_sqr -- the "sqr" in that name is a legacy family label, not IP
 *      semantics.
 *   2. laser_ip_factors(): the LASER full-sign RaBitQ correction factors for
 *      METRIC_IP. This is the IP sibling of RaBitQCore::laser_l2_factors
 *      (space/quant/rabitq_core.hpp:128), with the identical signature so the
 *      atomic phase can dispatch to it as a drop-in from
 *      laser::rabitq_factors (rabitq.hpp:99).
 *
 * ---------------------------------------------------------------------------
 * FULL-SIGN IP FACTOR DERIVATION (codex B-LIP-02).
 *
 * Let r = o - c (residual), s_i = 2*bit_i - 1 in {-1,+1} (LASER full sign;
 * bit_i = [r_i > 0]), and A = ||r||^2 / <r,s>. The single-level estimate the
 * LASER scanner assembles is
 *
 *     est = g_add + base + factor_dq * <s, q>,     g_add = -<q,c>,
 *
 * and it must target the score-domain value d_est = 1 - <q,o>. Solving gives
 *
 *     base       (triple_x) = 1 - <r,c> + A * <c,s>
 *     factor_dq            = -A
 *     factor_vq            = factor_dq * sum(s)   [derived downstream in
 *                            rabitq_factors (rabitq.hpp:119); metric-independent]
 *
 * CRITICAL half-sign vs full-sign factor of 2: the official RaBitQ-Library and
 * the in-repo memqg path use the HALF sign h_i = s_i/2 (in {-0.5,+0.5}) and pair
 * it with f_rescale = -2A (official METRIC_IP f_rescale = -l2_sqr/<r,h> = -2A).
 * The LASER scanner instead pairs factor_dq with the FULL sign <s,q> = 2<h,q>, so
 * factor_dq MUST be half of the official rescale:
 *
 *     laser factor_dq (= -A)  ==  official f_rescale (= -2A) / 2
 *
 * 1-D counterexample (c=2, o=3, q=4): r=1, s=+1, A=1, base=1, factor_dq=-1,
 * g_add=-8, so est = -8 + 1 + (-1)*(1*4) = -11 = 1 - <q,o>. Copying the official
 * -2A verbatim (forgetting to halve) would give -8 + 1 + (-2)*4 = -15. The
 * kernel test locks both the mapping and this counterexample.
 *
 * Zero-residual policy (o == c, so <r,s> = sum|r_i| = 0): the official code lets
 * the denominator go to +inf (f_add=1, f_rescale=-0) but its f_error is 0*sqrt(-x)
 * = NaN; laser_l2_factors likewise deliberately retains NaN. For IP the production
 * policy is instead the strict, NaN-free base=1, factor_dq=0 (=> factor_vq=0,
 * error=0), which keeps a strict weak ordering in the search buffer and is exactly
 * d_est (est = g_add + 1 + 0 = 1 - <q,c> = 1 - <q,o> when o=c). See B-LIP-09.
 * ---------------------------------------------------------------------------
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "platform/detect.hpp"
#include "simd/distance_ip.hpp"
#include "space/quant/rabitq_core.hpp"  // RaBitQCoreFactors

namespace alaya::laser::space {

/**
 * @brief Exact NEGATIVE inner product between two vectors (distance convention).
 * @param vec0 First vector
 * @param vec1 Second vector
 * @param dim  Vector dimension
 * @return -<vec0, vec1>  (smaller == more similar). NOT a squared quantity.
 */
inline float ip(const float *ALAYA_RESTRICT vec0, const float *ALAYA_RESTRICT vec1, size_t dim) {
  return ::alaya::simd::ip_sqr<float, float>(vec0, vec1, dim);
}

/**
 * @brief LASER full-sign RaBitQ correction factors for METRIC_IP.
 *
 * Drop-in sibling of RaBitQCore::laser_l2_factors (same signature) that returns
 * {base = triple_x, signed_query_scale = factor_dq} for the inner-product metric.
 * factor_vq is derived downstream (factor_dq * sum(s)) exactly as for L2, so it is
 * not returned here. See the file header for the derivation and the half-sign
 * factor-of-2 caveat.
 *
 * @param residual  o - c (residual vector), length dim
 * @param centroid  c, length dim
 * @param sign_bits bit_i = [residual_i > 0], length dim
 * @param dim       dimension
 * @param fac_norm  1/sqrt(dim); cancels in the A-form below and is accepted only
 *                  to match laser_l2_factors' signature for drop-in dispatch.
 * @return {base, signed_query_scale} = {1 - <r,c> + A<c,s>, -A}
 */
inline auto laser_ip_factors(const float *ALAYA_RESTRICT residual,
                             const float *ALAYA_RESTRICT centroid,
                             const int *ALAYA_RESTRICT sign_bits,
                             int64_t dim,
                             [[maybe_unused]] float fac_norm) -> RaBitQCoreFactors<float> {
  double residual_dot_signed = 0.0;    // <r, s>  = sum|r_i| >= 0
  double centroid_dot_signed = 0.0;    // <c, s>
  double residual_norm_sqr = 0.0;      // ||r||^2
  double residual_dot_centroid = 0.0;  // <r, c>
  for (int64_t i = 0; i < dim; ++i) {
    const double s = (2 * sign_bits[i]) - 1;  // full sign in {-1, +1}
    const double r = residual[i];
    residual_dot_signed += r * s;
    centroid_dot_signed += static_cast<double>(centroid[i]) * s;
    residual_norm_sqr += r * r;
    residual_dot_centroid += r * static_cast<double>(centroid[i]);
  }

  // Zero-residual (o == c): <r,s> == 0 iff r == 0. NaN-free IP policy (B-LIP-09).
  if (residual_dot_signed == 0.0) {
    return {1.0F, 0.0F};
  }

  const double a = residual_norm_sqr / residual_dot_signed;  // A = ||r||^2 / <r,s>
  return {
      static_cast<float>((1.0 - residual_dot_centroid) + (a * centroid_dot_signed)),  // triple_x
      static_cast<float>(-a),                                                         // factor_dq
  };
}

}  // namespace alaya::laser::space
