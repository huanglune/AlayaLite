// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file ip_oracle_test.cpp
 * @brief W0 oracle base for the LASER inner-product (IP) distance kernel.
 *
 * memqg-retirement route (1): before a production LASER IP kernel
 * (include/index/graph/laser/space/ip.hpp, W1) can be trusted, this file pins the
 * ground truth it must reproduce. The authoritative references are:
 *
 *   1. The IMMUTABLE official fixture ip_official_fixture.tsv, produced OFFLINE by
 *      tests/laser/space/tools/gen_ip_official_fixture.cpp actually linking RaBitQ-Library at commit
 *      b1f613d7412a041000d1e71aaa323d3e7554e733 and calling
 *      one_bit_code_with_factor(..., METRIC_IP). CI here only READS the fixture; a
 *      missing / truncated / commit-mismatched / unparsable fixture is a HARD
 *      FAILURE. There is no regeneration path in the test (B-LIP-08).
 *   2. The in-repo memqg path: RaBitQCore::memory_factors(..., inner_product) and
 *      the RaBitQSpace fastscan estimator (space/rabitq_space.hpp).
 *   3. Brute-force exact <q,o> in double precision.
 *
 * The demoted ALGEBRA re-derivation (ip_fixture::algebra_ip_factors) is a FOURTH
 * cross-check only -- not an official oracle. See ip_fixture_common.hpp for the
 * score-domain spec, the deterministic input mapping and the fixture reader.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "core/value_types.hpp"
#include "index/neighbor.hpp"
#include "ip_fixture_common.hpp"
#include "simd/distance_ip.hpp"
#include "space/quant/rabitq/defines.hpp"
#include "space/quant/rabitq_core.hpp"
#include "space/rabitq_space.hpp"

namespace alaya {
namespace {

using namespace ip_fixture;  // NOLINT(build/namespaces) -- test-local convenience

auto random_vec(std::mt19937 &rng, size_t dim, float lo, float hi) -> std::vector<float> {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(dim);
  for (auto &x : v) {
    x = dist(rng);
  }
  return v;
}

// Bit-pattern non-finite predicate. Release builds LASER with -Ofast (=> -ffast-math,
// finite-math-only), under which std::isnan/std::isfinite fold to constants and FP
// NaN neither propagates nor is detectable. The ONLY robust rejection predicate is
// an integer inspection of the IEEE-754 exponent field (immune to -ffast-math),
// exactly as tests/laser/rabitq_factor_equivalence_test.cpp does for NaN bits.
auto is_non_finite_bits(float v) -> bool {
  const uint32_t b = std::bit_cast<uint32_t>(v);
  return (b & 0x7F800000U) == 0x7F800000U;  // exponent all ones => +-Inf or NaN
}

// In-repo cross-check dims (self-consistency only: memqg vs algebra vs brute, all
// computed in this binary -- stdlib RNG drift picks different inputs but cannot
// break an invariant checked on both sides of the same inputs). The OFFICIAL lock
// uses the drift-free fixture instead.
const std::vector<size_t> kOracleDims = {64, 128, 192, 256, 1024};

}  // namespace

// ===========================================================================
// T1 -- OFFICIAL LOCK. For every immutable fixture row: reproduce the exact
// inputs from the stored seed (verified against the stored FNV fingerprint),
// then assert the in-repo memqg IP factors reproduce the official f_add/f_rescale
// and sign-bit popcount. The demoted algebra re-derivation is checked too (incl.
// f_error, which memqg does not expose). A missing/bad fixture is a hard failure.
// ===========================================================================
TEST(LaserIpOracle, OfficialFixtureMatchesMemqgAndAlgebra) {
  std::string err;
  const std::vector<FixtureRow> rows = load_ip_fixture(&err);
  ASSERT_TRUE(err.empty()) << err;
  ASSERT_FALSE(rows.empty()) << "fixture unexpectedly empty";

  for (const auto &row : rows) {
    SCOPED_TRACE(::testing::Message() << "dim=" << row.dim << " trial=" << row.trial);
    const RowInputs in = make_row_inputs(row.seed, row.dim);

    // Input reproduction lock: the regenerated inputs must byte-match what the
    // official generator hashed. Guards against any mapping drift.
    ASSERT_EQ(input_fingerprint(in.data, in.centroid), row.input_fnv)
        << "reproduced input does not match the fixture fingerprint";

    // memqg IP factors (half-sign convention, same as official).
    std::vector<int> mem_bits(row.dim);
    const auto mem = RaBitQCore::memory_factors<float>(in.data.data(), in.centroid.data(), row.dim,
                                                       mem_bits.data(),
                                                       core::Metric::inner_product);
    int mem_pc = 0;
    for (int b : mem_bits) {
      mem_pc += b;
    }
    EXPECT_EQ(mem_pc, row.popcount) << "memqg popcount != official";

    const float tol_add = 1.0e-3F * std::max(1.0F, std::abs(row.f_add));
    const float tol_rescale = 1.0e-3F * std::max(1.0F, std::abs(row.f_rescale));
    EXPECT_NEAR(row.f_add, mem.base, tol_add);
    EXPECT_NEAR(row.f_rescale, mem.signed_query_scale, tol_rescale);

    // Fourth (algebra) cross-check, including f_error which is fixture-only.
    std::vector<int> alg_bits;
    const auto alg = algebra_ip_factors(in.data.data(), in.centroid.data(), row.dim, &alg_bits);
    EXPECT_EQ(alg_bits, mem_bits);
    EXPECT_NEAR(row.f_add, alg.f_add, tol_add);
    EXPECT_NEAR(row.f_rescale, alg.f_rescale, tol_rescale);
    EXPECT_NEAR(row.f_error, alg.f_error, 1.0e-3F * std::max(1.0F, std::abs(row.f_error)));
  }
}

// ===========================================================================
// T2 -- shape lock: with q == c the estimator's own approximation is
// algebraically exact, so the assembled estimate collapses to *exactly* the
// score-domain target d_est = 1 - <c,o>. Pins the affine target form.
// ===========================================================================
TEST(LaserIpOracle, EstimatorLocksToOneMinusDotAtQEqualsC) {
  std::mt19937 rng(0xB0B02U);
  for (size_t dim : kOracleDims) {
    for (int trial = 0; trial < 12; ++trial) {
      const auto data = random_vec(rng, dim, -4.0F, 4.0F);
      const auto centroid = random_vec(rng, dim, -4.0F, 4.0F);
      std::vector<int> bits;
      const auto alg = algebra_ip_factors(data.data(), centroid.data(), dim, &bits);
      // q = centroid.
      const float est =
          unquantized_ip_estimate(alg.f_add, alg.f_rescale, bits, centroid.data(),
                                  centroid.data(), dim);
      const float expected = 1.0F - static_cast<float>(exact_ip(centroid.data(), data.data(), dim));
      EXPECT_NEAR(est, expected, 2.0e-3F * std::max(1.0F, std::abs(expected)))
          << "dim=" << dim << " trial=" << trial;
    }
  }
}

// ===========================================================================
// T3 -- one-bit theoretical error bound, LUT BYPASSED (B-LIP-09). The un-quantized
// estimate tracks d_est within the official high-probability bound f_error*||q-c||
// (IP's f_error coefficient is half of L2's). Multiple rotator-style seeds are
// aggregated and the out-of-bound *rate* is asserted small -- this is explicitly a
// high-probability guarantee, NOT a per-sample absolute bound. Query quantization
// error is out of scope here (see T5).
// ===========================================================================
TEST(LaserIpOracle, OneBitBoundIsHighProbability) {
  size_t n = 0;
  size_t within = 0;
  double sum_signed = 0.0;
  double sum_bound = 0.0;
  double max_ratio = 0.0;

  for (uint32_t seed : {0xC0FFEEU, 0x1357BDU, 0x2468ACU, 0x0F1E2DU}) {
    std::mt19937 rng(seed);
    for (size_t dim : kOracleDims) {
      for (int trial = 0; trial < 120; ++trial) {
        const auto centroid = random_vec(rng, dim, -2.0F, 2.0F);
        const auto data = random_vec(rng, dim, -2.0F, 2.0F);
        // Query near the data (deterministic uniform perturbation -- no
        // std::normal_distribution, which drifts across stdlibs).
        auto query = data;
        const auto jitter = random_vec(rng, dim, -0.3F, 0.3F);
        for (size_t i = 0; i < dim; ++i) {
          query[i] += jitter[i];
        }

        std::vector<int> bits;
        const auto alg = algebra_ip_factors(data.data(), centroid.data(), dim, &bits);
        const float est =
            unquantized_ip_estimate(alg.f_add, alg.f_rescale, bits, query.data(), centroid.data(),
                                    dim);
        const float target = 1.0F - static_cast<float>(exact_ip(query.data(), data.data(), dim));
        const float err = est - target;
        const float bound = alg.f_error * g_error_ip(query.data(), centroid.data(), dim);

        ++n;
        sum_signed += err;
        sum_bound += bound;
        if (std::abs(err) <= bound + 1.0e-4F) {
          ++within;
        }
        if (bound > 1.0e-6F) {
          max_ratio = std::max(max_ratio, static_cast<double>(std::abs(err) / bound));
        }
      }
    }
  }

  const double frac_within = static_cast<double>(within) / static_cast<double>(n);
  const double mean_signed = sum_signed / static_cast<double>(n);
  const double mean_bound = sum_bound / static_cast<double>(n);
  std::cout << "ip_oracle_bound_samples=" << n << '\n';
  std::cout << "ip_oracle_bound_frac_within=" << frac_within << '\n';
  std::cout << "ip_oracle_bound_max_err_over_bound=" << max_ratio << '\n';
  std::cout << "ip_oracle_mean_signed_err=" << mean_signed << '\n';
  std::cout << "ip_oracle_mean_bound=" << mean_bound << '\n';

  EXPECT_GE(frac_within, 0.98) << "high-probability bound violated too often";
  EXPECT_LE(std::abs(mean_signed), 0.30 * mean_bound) << "estimator looks biased";
}

// ===========================================================================
// T4 -- ordering contract. (a) EXACT affine-order identity: sorting candidates by
// d_exact=-<q,o> and by d_est_target=1-<q,o> yields the IDENTICAL permutation,
// because the +1 offset is candidate-independent. Computed in DOUBLE with a unified
// candidate-id tie-break and checked on integer indices only (no float bitwise
// equality). (b) statistical: the un-quantized estimate recovers exact top-k well
// above random (10/32~=0.31); soft floor mirrors the sibling RaBitQSpaceIpNormTest.
// ===========================================================================
TEST(LaserIpOracle, AffineOrderIdentityAndRecall) {
  std::mt19937 rng(0xD00DU);
  constexpr size_t kCand = 32;
  constexpr size_t kTopK = 10;
  double recall_sum = 0.0;
  int rounds = 0;

  for (size_t dim : {128UL, 256UL}) {
    for (int q = 0; q < 40; ++q) {
      const auto centroid = random_vec(rng, dim, -1.5F, 1.5F);
      std::vector<std::vector<float>> cand(kCand);
      for (auto &c : cand) {
        c = random_vec(rng, dim, -1.5F, 1.5F);
      }
      const auto query = random_vec(rng, dim, -1.5F, 1.5F);

      std::vector<std::pair<double, size_t>> by_neg_ip;
      std::vector<std::pair<double, size_t>> by_target;
      std::vector<std::pair<float, size_t>> by_est;
      by_neg_ip.reserve(kCand);
      by_target.reserve(kCand);
      by_est.reserve(kCand);
      for (size_t i = 0; i < kCand; ++i) {
        std::vector<int> bits;
        const auto alg = algebra_ip_factors(cand[i].data(), centroid.data(), dim, &bits);
        const float est =
            unquantized_ip_estimate(alg.f_add, alg.f_rescale, bits, query.data(), centroid.data(),
                                    dim);
        const double ip = exact_ip(query.data(), cand[i].data(), dim);
        by_neg_ip.emplace_back(-ip, i);       // d_exact
        by_target.emplace_back(1.0 - ip, i);  // d_est target (affine +1)
        by_est.emplace_back(est, i);
      }
      // std::pair sorts by value then by id -> unified candidate-id tie-break.
      std::sort(by_neg_ip.begin(), by_neg_ip.end());
      std::sort(by_target.begin(), by_target.end());
      std::sort(by_est.begin(), by_est.end());
      for (size_t i = 0; i < kCand; ++i) {
        ASSERT_EQ(by_neg_ip[i].second, by_target[i].second)
            << "affine +1 offset changed the order at rank " << i << " dim=" << dim;
      }

      std::set<size_t> top_exact;
      for (size_t i = 0; i < kTopK; ++i) {
        top_exact.insert(by_neg_ip[i].second);
      }
      size_t hit = 0;
      for (size_t i = 0; i < kTopK; ++i) {
        if (top_exact.count(by_est[i].second) > 0) {
          ++hit;
        }
      }
      recall_sum += static_cast<double>(hit) / static_cast<double>(kTopK);
      ++rounds;
    }
  }
  const double recall = recall_sum / rounds;
  std::cout << "ip_oracle_unquantized_rank_recall_at_10=" << recall << '\n';
  EXPECT_GE(recall, 0.45);
}

// ===========================================================================
// T5 -- memqg fastscan end-to-end (the real production estimator, WITH LUT byte
// quantization). Tracks exact -<q,o>. Reports recall (query quantization error is
// included here and is why this is a recall floor, not the T3 theoretical bound).
// ===========================================================================
TEST(LaserIpOracle, MemqgFastscanPathTracksExactIp) {
  using SpaceType = RaBitQSpace<float, float, uint32_t>;
  constexpr uint32_t kDegree = static_cast<uint32_t>(SpaceType::kDegreeBound);  // 32
  constexpr uint32_t kNum = kDegree + 1;
  constexpr uint32_t kCentroid = 0;
  constexpr size_t kTopK = 10;

  for (size_t dim : {128UL, 768UL}) {  // pow2 and 768->1024 pad
    std::mt19937 rng(0xE5750000U + static_cast<uint32_t>(dim));
    std::uniform_real_distribution<float> uni(-1.0F, 1.0F);
    std::vector<float> data(static_cast<size_t>(kNum) * dim);
    for (auto &x : data) {
      x = uni(rng);
    }

    auto space = std::make_shared<SpaceType>(kNum, dim, core::Metric::inner_product);
    space->fit(data.data(), kNum);
    std::vector<Neighbor<uint32_t, float>> nbrs;
    nbrs.reserve(kDegree);
    for (uint32_t i = 1; i < kNum; ++i) {
      nbrs.emplace_back(i, 0.0F);
    }
    space->update_nei(kCentroid, nbrs);

    constexpr int kQueries = 30;
    double recall_sum = 0.0;
    for (int q = 0; q < kQueries; ++q) {
      const uint32_t base = 1 + (static_cast<uint32_t>(q) % kDegree);
      std::vector<float> query(dim);
      std::uniform_real_distribution<float> jit(-0.1F, 0.1F);
      for (size_t d = 0; d < dim; ++d) {
        query[d] = data[(base * dim) + d] + jit(rng);
      }
      auto qc = space->get_query_computer(query.data());
      qc.load_centroid(kCentroid);
      const float *est = qc.est_data();

      std::vector<std::pair<float, uint32_t>> exact;
      std::vector<std::pair<float, uint32_t>> estimated;
      exact.reserve(kDegree);
      estimated.reserve(kDegree);
      for (uint32_t i = 0; i < kDegree; ++i) {
        const float neg = -static_cast<float>(exact_ip(query.data(), &data[nbrs[i].id_ * dim], dim));
        exact.emplace_back(neg, i);
        estimated.emplace_back(est[i], i);
      }
      std::sort(exact.begin(), exact.end());
      std::sort(estimated.begin(), estimated.end());
      std::set<uint32_t> top;
      for (size_t i = 0; i < kTopK; ++i) {
        top.insert(exact[i].second);
      }
      size_t hit = 0;
      for (size_t i = 0; i < kTopK; ++i) {
        if (top.count(estimated[i].second) > 0) {
          ++hit;
        }
      }
      recall_sum += static_cast<double>(hit) / static_cast<double>(kTopK);
    }
    const double recall = recall_sum / kQueries;
    std::cout << "ip_oracle_memqg_path_recall_at_10_dim" << dim << "=" << recall << '\n';
    EXPECT_GE(recall, 0.45) << "memqg fastscan IP path recall collapsed at dim=" << dim;
  }
}

// ===========================================================================
// AlayaLite IP-as-distance convention lock: simd::ip_sqr returns -<a,b>, so
// "smaller == more similar" holds on the min-heap search path. Tolerance, not
// bitwise (SIMD reduction order differs) -- see tests/simd/ip_test.cpp:53.
// ===========================================================================
TEST(LaserIpOracle, SimdIpConventionIsNegated) {
  std::mt19937 rng(0xF00DU);
  auto ip_func = simd::get_ip_sqr_func();
  for (size_t dim : {128UL, 256UL}) {
    const auto a = random_vec(rng, dim, -3.0F, 3.0F);
    const auto b = random_vec(rng, dim, -3.0F, 3.0F);
    const float got = ip_func(a.data(), b.data(), dim);
    const float want = -static_cast<float>(exact_ip(a.data(), b.data(), dim));
    EXPECT_NEAR(got, want, 1.0e-3F * std::max(1.0F, std::abs(want))) << "dim=" << dim;
  }
}

// ===========================================================================
// Degenerate-input contract (B-LIP-09). Each is a named corner the production IP
// kernel must survive; all use the in-repo memqg + brute oracles.
// ===========================================================================

// Negative IP: <q,o> < 0 must yield a positive distance d_exact=-<q,o> and a
// finite estimate near d_est=1-<q,o>; nothing NaNs.
TEST(LaserIpOracleDegenerate, NegativeInnerProduct) {
  constexpr size_t dim = 128;
  std::vector<float> q(dim, 0.5F);
  std::vector<float> o(dim, -0.7F);  // <q,o> = 0.5*-0.7*dim < 0
  std::vector<float> c(dim, 0.1F);
  const double ip = exact_ip(q.data(), o.data(), dim);
  ASSERT_LT(ip, 0.0);
  EXPECT_GT(-ip, 0.0);  // distance is positive

  std::vector<int> bits;
  const auto alg = algebra_ip_factors(o.data(), c.data(), dim, &bits);
  const float est = unquantized_ip_estimate(alg.f_add, alg.f_rescale, bits, q.data(), c.data(), dim);
  EXPECT_TRUE(std::isfinite(est));
  const float target = 1.0F - static_cast<float>(ip);
  const float bound = alg.f_error * g_error_ip(q.data(), c.data(), dim);
  EXPECT_LE(std::abs(est - target), bound + 1.0e-3F);
}

// Zero query: q=0 -> all d_exact=0, d_est=1. The estimate must be finite and the
// one-bit bound must still hold (q=0 is a legal query, ||q-c||=||c||).
TEST(LaserIpOracleDegenerate, ZeroQuery) {
  constexpr size_t dim = 128;
  std::mt19937 rng(0x0F00DU);
  const auto data = random_vec(rng, dim, -2.0F, 2.0F);
  const auto centroid = random_vec(rng, dim, -2.0F, 2.0F);
  const std::vector<float> query(dim, 0.0F);

  std::vector<int> bits;
  const auto alg = algebra_ip_factors(data.data(), centroid.data(), dim, &bits);
  const float est =
      unquantized_ip_estimate(alg.f_add, alg.f_rescale, bits, query.data(), centroid.data(), dim);
  EXPECT_TRUE(std::isfinite(est));
  const float target = 1.0F;  // 1 - <0,o>
  const float bound = alg.f_error * g_error_ip(query.data(), centroid.data(), dim);
  EXPECT_LE(std::abs(est - target), bound + 1.0e-3F);
}

// Extreme non-unit norm: refutes any hidden ||o||=1 assumption. q=(1,0,...),
// a=(100,100,0,...) (large norm, IP 100), b=(1,0,...) (small norm, IP 1). IP and
// the estimator must BOTH prefer a (larger inner product => smaller distance),
// even though a has the far larger norm.
TEST(LaserIpOracleDegenerate, ExtremeNonUnitNormOrdersByIpNotNorm) {
  constexpr size_t dim = 128;
  std::vector<float> q(dim, 0.0F);
  q[0] = 1.0F;
  std::vector<float> a(dim, 0.0F);
  a[0] = 100.0F;
  a[1] = 100.0F;
  std::vector<float> b(dim, 0.0F);
  b[0] = 1.0F;
  const std::vector<float> centroid(dim, 0.0F);

  const double ip_a = exact_ip(q.data(), a.data(), dim);  // 100
  const double ip_b = exact_ip(q.data(), b.data(), dim);  // 1
  ASSERT_GT(ip_a, ip_b);
  EXPECT_LT(-ip_a, -ip_b);  // a is nearer under IP distance

  std::vector<int> bits_a;
  std::vector<int> bits_b;
  const auto fa = algebra_ip_factors(a.data(), centroid.data(), dim, &bits_a);
  const auto fb = algebra_ip_factors(b.data(), centroid.data(), dim, &bits_b);
  const float est_a =
      unquantized_ip_estimate(fa.f_add, fa.f_rescale, bits_a, q.data(), centroid.data(), dim);
  const float est_b =
      unquantized_ip_estimate(fb.f_add, fb.f_rescale, bits_b, q.data(), centroid.data(), dim);
  EXPECT_LT(est_a, est_b) << "estimator ordered by norm instead of inner product";
}

// Tie / all-equal-|value|: candidates with identical <q,o> must resolve to a
// deterministic order under the candidate-id tie-break, and all-positive residual
// (all sign bits 1) must still yield finite factors.
TEST(LaserIpOracleDegenerate, TieBreakAndAllEqualSign) {
  constexpr size_t dim = 64;
  std::vector<float> q(dim, 0.3F);
  const std::vector<float> centroid(dim, 0.0F);
  // Three identical candidates -> exact IP ties.
  std::vector<std::vector<float>> cand(3, std::vector<float>(dim, 0.9F));
  std::vector<std::pair<double, size_t>> order;
  for (size_t i = 0; i < cand.size(); ++i) {
    order.emplace_back(-exact_ip(q.data(), cand[i].data(), dim), i);
  }
  std::sort(order.begin(), order.end());
  EXPECT_EQ(order[0].second, 0U);  // tie broken by id, deterministically
  EXPECT_EQ(order[1].second, 1U);
  EXPECT_EQ(order[2].second, 2U);

  // All-positive residual (data > centroid everywhere) -> every sign bit is 1.
  std::vector<float> data(dim);
  for (size_t i = 0; i < dim; ++i) {
    data[i] = 1.0F + 0.01F * static_cast<float>(i);
  }
  std::vector<int> bits;
  const auto alg = algebra_ip_factors(data.data(), centroid.data(), dim, &bits);
  int pc = 0;
  for (int bt : bits) {
    pc += bt;
  }
  EXPECT_EQ(pc, static_cast<int>(dim));  // all signs equal (all 1)
  EXPECT_TRUE(std::isfinite(alg.f_add));
  EXPECT_TRUE(std::isfinite(alg.f_rescale));
}

// Non-finite REJECTION contract (B-LIP-09). Non-finite inputs are UB for the
// -ffast-math kernel, so a future importer/descriptor/target-admission MUST reject
// them fail-closed BEFORE quantization. That rejection cannot rely on kernel FP
// math (folded away); it uses the bit-pattern predicate above. This locks the
// predicate the atomic-phase gate will apply to raw input rows.
TEST(LaserIpOracleDegenerate, NonFiniteRejectionPredicate) {
  EXPECT_TRUE(is_non_finite_bits(std::numeric_limits<float>::quiet_NaN()));
  EXPECT_TRUE(is_non_finite_bits(std::numeric_limits<float>::infinity()));
  EXPECT_TRUE(is_non_finite_bits(-std::numeric_limits<float>::infinity()));
  EXPECT_FALSE(is_non_finite_bits(0.0F));
  EXPECT_FALSE(is_non_finite_bits(-0.0F));
  EXPECT_FALSE(is_non_finite_bits(3.14159F));
  EXPECT_FALSE(is_non_finite_bits(-1.0e30F));
  EXPECT_FALSE(is_non_finite_bits(std::numeric_limits<float>::denorm_min()));

  // A row-level scan (what the gate applies to a raw input vector before it ever
  // reaches the quantizer) flags a poisoned coordinate.
  constexpr size_t dim = 64;
  std::vector<float> row(dim, 1.0F);
  bool any = false;
  for (float x : row) {
    any = any || is_non_finite_bits(x);
  }
  EXPECT_FALSE(any) << "clean row wrongly rejected";
  row[dim - 1] = std::numeric_limits<float>::infinity();
  any = false;
  for (float x : row) {
    any = any || is_non_finite_bits(x);
  }
  EXPECT_TRUE(any) << "poisoned row not rejected";
}

}  // namespace alaya
