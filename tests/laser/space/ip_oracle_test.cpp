// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file ip_oracle_test.cpp
 * @brief W0 oracle base for the LASER inner-product (IP) distance kernel.
 *
 * memqg-retirement route (1): before a production LASER IP kernel
 * (include/index/graph/laser/space/ip.hpp, W1) can be trusted, this file pins
 * the ground truth it must reproduce, from THREE independent references:
 *
 *   1. Official RaBitQ-Library METRIC_IP reference -- a from-scratch
 *      re-derivation of one_bit_code_with_factor(..., METRIC_IP)
 *      (baselines/RaBitQ-Library/include/rabitqlib/quantization/rabitq_impl.hpp
 *      :80-136) and its estimator full_est_dist (rabitq.hpp:384). See
 *      official_ip_factors() below, with per-line citations.
 *   2. The in-repo memqg path: RaBitQCore::memory_factors(..., inner_product)
 *      (space/quant/rabitq_core.hpp) and the RaBitQSpace fastscan estimator
 *      (space/rabitq_space.hpp).
 *   3. Brute-force exact <q,o> in double precision.
 *
 * These three are convention-independent truths (the LASER factor *convention*
 * -- laser_l2_factors' signed_x/fac_norm form -- is a fourth thing that W1's
 * ip.hpp produces and validates against this base; it is deliberately NOT
 * baked in here, so the W1 review gate stays open).
 *
 * The AlayaLite IP-as-distance convention is -<q,o> (simd::ip_sqr,
 * include/simd/distance_ip.ipp:39 "Negative for distance metric"). The RaBitQ
 * estimator natively targets 1 - <q,o>; the "+1" is a candidate-independent
 * constant (rabitq_core.hpp:87-112), so ordering by either is identical.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "core/value_types.hpp"
#include "index/neighbor.hpp"
#include "simd/distance_ip.hpp"
#include "space/quant/rabitq/defines.hpp"
#include "space/quant/rabitq_core.hpp"
#include "space/rabitq_space.hpp"

namespace alaya {
namespace {

// kConstEpsilon, baselines/RaBitQ-Library/include/rabitqlib/quantization/rabitq_impl.hpp:17
constexpr double kOfficialEpsilon = 1.9;

// ---------------------------------------------------------------------------
// Oracle 3: brute-force exact inner product (double ground truth).
// ---------------------------------------------------------------------------
auto exact_ip(const float *a, const float *b, size_t dim) -> double {
  double s = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  return s;
}

// ---------------------------------------------------------------------------
// Oracle 1: official RaBitQ-Library METRIC_IP factor reference.
//
// Re-derivation of one_bit_code_with_factor(..., METRIC_IP), rabitq_impl.hpp
// :80-136. Operates in already-rotated space (data/centroid are post-rotation
// vectors, exactly as the production quantizer feeds them). Accumulated in
// float to mirror the library's T=float production precision, so the memqg
// cross-check (Oracle 2) is tight; the exact/bound comparisons use the double
// oracle above as ground truth.
//
// The literal 1 in f_add (line 128) and the literal 1 coefficient in f_error
// (line 131, vs 2 for L2 on line 126) are the upstream convention, NOT a port
// artifact: for METRIC_IP the estimator target is 1 - <q,o> and the residual
// error term carries coefficient 1. This is the exact provenance the manifest
// asks W1 to cite for the "factor literal 1".
// ---------------------------------------------------------------------------
struct OfficialIpFactors {
  float f_add;      // rabitq_impl.hpp:128
  float f_rescale;  // rabitq_impl.hpp:130
  float f_error;    // rabitq_impl.hpp:131
};

auto official_ip_factors(const float *data, const float *centroid, size_t dim,
                         std::vector<int> *bits_out) -> OfficialIpFactors {
  std::vector<float> residual(dim);
  for (size_t i = 0; i < dim; ++i) {
    residual[i] = data[i] - centroid[i];
  }
  std::vector<int> bits(dim);
  for (size_t i = 0; i < dim; ++i) {
    bits[i] = residual[i] > 0.0F ? 1 : 0;  // sign(residual), rabitq_impl.hpp one_bit_code
  }

  const float cb = -((1 << 1) - 1) / 2.0F;  // -0.5F, rabitq_impl.hpp:89
  std::vector<float> xu_cb(dim);
  for (size_t i = 0; i < dim; ++i) {
    xu_cb[i] = static_cast<float>(bits[i]) + cb;  // half_signed
  }

  float l2_sqr = 0.0F;
  float ip_resi_xucb = 0.0F;
  float ip_cent_xucb = 0.0F;
  float resi_cent = 0.0F;
  float xucb_sqr = 0.0F;
  for (size_t i = 0; i < dim; ++i) {
    l2_sqr += residual[i] * residual[i];
    ip_resi_xucb += residual[i] * xu_cb[i];
    ip_cent_xucb += centroid[i] * xu_cb[i];
    resi_cent += residual[i] * centroid[i];
    xucb_sqr += xu_cb[i] * xu_cb[i];
  }
  const float l2_norm = std::sqrt(l2_sqr);
  if (ip_resi_xucb == 0.0F) {
    ip_resi_xucb = std::numeric_limits<float>::infinity();  // rabitq_impl.hpp:103-105
  }

  // tmp_error, rabitq_impl.hpp:110-116.
  const float tmp_error =
      l2_norm * static_cast<float>(kOfficialEpsilon) *
      std::sqrt((((l2_sqr * xucb_sqr) / (ip_resi_xucb * ip_resi_xucb)) - 1.0F) /
                (static_cast<float>(dim) - 1.0F));

  OfficialIpFactors f{};
  f.f_add = 1.0F - resi_cent + (l2_sqr * ip_cent_xucb / ip_resi_xucb);  // :128
  f.f_rescale = -l2_sqr / ip_resi_xucb;                                 // :130
  f.f_error = 1.0F * tmp_error;                                         // :131
  if (bits_out != nullptr) {
    *bits_out = std::move(bits);
  }
  return f;
}

// Un-quantized RaBitQ estimate, assembled exactly like
// RaBitQSpace::QueryComputer (rabitq_space.hpp:212 in the sibling test) and the
// official full_est_dist (RaBitQ-Library rabitq.hpp:384), with 1-bit codes:
//   est = f_add + g_add + f_rescale * <half_signed, q_rot>,  g_add = -<q,c>.
// No LUT byte quantization -> isolates the factor algebra (binary-code error
// only), which is what the bound in T3 is stated against.
auto unquantized_ip_estimate(const OfficialIpFactors &f, const std::vector<int> &bits,
                             const float *query, const float *centroid, size_t dim) -> float {
  float half_signed_dot_q = 0.0F;
  for (size_t i = 0; i < dim; ++i) {
    half_signed_dot_q += (static_cast<float>(bits[i]) - 0.5F) * query[i];
  }
  const float g_add = -static_cast<float>(exact_ip(query, centroid, dim));  // query.hpp:104
  return f.f_add + g_add + (f.f_rescale * half_signed_dot_q);
}

// g_error = ||q - c|| (query residual norm), query.hpp:105.
auto g_error_ip(const float *query, const float *centroid, size_t dim) -> float {
  float s = 0.0F;
  for (size_t i = 0; i < dim; ++i) {
    const float d = query[i] - centroid[i];
    s += d * d;
  }
  return std::sqrt(s);
}

auto random_vec(std::mt19937 &rng, size_t dim, float lo, float hi) -> std::vector<float> {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(dim);
  for (auto &x : v) {
    x = dist(rng);
  }
  return v;
}

// Dims exercised by the factor/estimate oracles: pow2 128/256 (importer-valid),
// 192 (multiple-of-64 non-pow2), 1024 (the padded dim a 768-d dataset rotates
// into -- the GIST/Cohere pad scenario). 64 anchors the sibling test's dim.
const std::vector<size_t> kOracleDims = {64, 128, 192, 256, 1024};

}  // namespace

// ===========================================================================
// T1 -- Oracle 1 == Oracle 2: the from-scratch official IP factor re-derivation
// is byte-close to the in-repo memqg IP branch (RaBitQCore::memory_factors).
// Both are the same formula (rabitq_core.hpp:114-119 documents the identity),
// so this triangulates that neither drifted. sign bits must match exactly.
// ===========================================================================
TEST(LaserIpOracle, OfficialFactorsMatchMemqg) {
  std::mt19937 rng(0xA11CE01U);
  for (size_t dim : kOracleDims) {
    for (int trial = 0; trial < 24; ++trial) {
      const auto data = random_vec(rng, dim, -4.0F, 4.0F);
      const auto centroid = random_vec(rng, dim, -4.0F, 4.0F);

      std::vector<int> off_bits;
      const auto off = official_ip_factors(data.data(), centroid.data(), dim, &off_bits);

      std::vector<int> mem_bits(dim);
      const auto mem = RaBitQCore::memory_factors<float>(
          data.data(), centroid.data(), dim, mem_bits.data(), core::Metric::inner_product);

      EXPECT_EQ(off_bits, mem_bits) << "dim=" << dim << " trial=" << trial;

      const float tol_add = 1.0e-3F * std::max(1.0F, std::abs(off.f_add));
      const float tol_rescale = 1.0e-3F * std::max(1.0F, std::abs(off.f_rescale));
      EXPECT_NEAR(off.f_add, mem.base, tol_add) << "dim=" << dim << " trial=" << trial;
      EXPECT_NEAR(off.f_rescale, mem.signed_query_scale, tol_rescale)
          << "dim=" << dim << " trial=" << trial;
    }
  }
}

// ===========================================================================
// T2 -- shape lock: with q == c, the K-estimator's own approximation is
// algebraically exact, so est collapses to *exactly* 1 - <c,o>. This pins the
// estimator target's affine form (the sibling
// RaBitQCoreTest.InnerProductBranchLocksToOneMinusDot pins the same via
// memory_factors; here we also confirm the official reimpl's assembled estimate
// agrees, closing the loop between Oracle 1 and Oracle 2).
// ===========================================================================
TEST(LaserIpOracle, EstimatorLocksToOneMinusDotAtQEqualsC) {
  std::mt19937 rng(0xB0B02U);
  for (size_t dim : kOracleDims) {
    for (int trial = 0; trial < 12; ++trial) {
      const auto data = random_vec(rng, dim, -4.0F, 4.0F);
      const auto centroid = random_vec(rng, dim, -4.0F, 4.0F);

      std::vector<int> bits;
      const auto off = official_ip_factors(data.data(), centroid.data(), dim, &bits);
      // q = centroid.
      const float est = unquantized_ip_estimate(off, bits, centroid.data(), centroid.data(), dim);
      const float expected = 1.0F - static_cast<float>(exact_ip(centroid.data(), data.data(), dim));
      EXPECT_NEAR(est, expected, 2.0e-3F * std::max(1.0F, std::abs(expected)))
          << "dim=" << dim << " trial=" << trial;
    }
  }
}

// ===========================================================================
// T3 -- error-bound scaffolding: the un-quantized estimate tracks the target
// 1 - <q,o> within the official bound f_error * ||q-c||, and is (near-)unbiased.
// Reports the error distribution so W1 can reuse this harness for ip.hpp.
// ===========================================================================
TEST(LaserIpOracle, UnbiasedAndBounded) {
  std::mt19937 rng(0xC0FFEEU);
  size_t n = 0;
  size_t within = 0;
  double sum_signed = 0.0;
  double sum_abs = 0.0;
  double sum_bound = 0.0;
  double max_ratio = 0.0;

  for (size_t dim : kOracleDims) {
    for (int trial = 0; trial < 400; ++trial) {
      const auto centroid = random_vec(rng, dim, -2.0F, 2.0F);
      const auto data = random_vec(rng, dim, -2.0F, 2.0F);
      // Query near a random direction of the data, lightly perturbed.
      auto query = data;
      std::normal_distribution<float> noise(0.0F, 0.3F);
      for (auto &x : query) {
        x += noise(rng);
      }

      std::vector<int> bits;
      const auto off = official_ip_factors(data.data(), centroid.data(), dim, &bits);
      const float est = unquantized_ip_estimate(off, bits, query.data(), centroid.data(), dim);
      const float target = 1.0F - static_cast<float>(exact_ip(query.data(), data.data(), dim));
      const float err = est - target;
      const float bound = off.f_error * g_error_ip(query.data(), centroid.data(), dim);

      ++n;
      sum_signed += err;
      sum_abs += std::abs(err);
      sum_bound += bound;
      if (std::abs(err) <= bound + 1.0e-4F) {
        ++within;
      }
      if (bound > 1.0e-6F) {
        max_ratio = std::max(max_ratio, static_cast<double>(std::abs(err) / bound));
      }
    }
  }

  const double frac_within = static_cast<double>(within) / static_cast<double>(n);
  const double mean_signed = sum_signed / static_cast<double>(n);
  const double mean_abs = sum_abs / static_cast<double>(n);
  const double mean_bound = sum_bound / static_cast<double>(n);
  std::cout << "ip_oracle_bound_samples=" << n << '\n';
  std::cout << "ip_oracle_bound_frac_within=" << frac_within << '\n';
  std::cout << "ip_oracle_bound_max_err_over_bound=" << max_ratio << '\n';
  std::cout << "ip_oracle_mean_signed_err=" << mean_signed << '\n';
  std::cout << "ip_oracle_mean_abs_err=" << mean_abs << '\n';
  std::cout << "ip_oracle_mean_bound=" << mean_bound << '\n';

  // The eps=1.9 bound is a high-probability guarantee; the un-quantized
  // estimate (no LUT error) should sit inside it essentially always.
  EXPECT_GE(frac_within, 0.98) << "error bound violated too often";
  // (Near-)unbiased: mean signed error is small relative to the mean bound.
  EXPECT_LE(std::abs(mean_signed), 0.30 * mean_bound) << "estimator looks biased";
}

// ===========================================================================
// T4 -- the ordering contract. Two claims, one exact and one statistical:
//
//  (a) EXACT (the "IP order == neg-distance order, fully identical" contract
//      W1's search path relies on): the RaBitQ estimator target 1 - <q,o> and
//      the AlayaLite IP distance -<q,o> differ only by the candidate-independent
//      constant +1, so ascending sorts of the two produce the *identical*
//      permutation. This is what lets the min-heap search path treat the IP
//      estimate as a distance with zero metric-aware reordering. Asserted to be
//      byte-for-byte identical here (no threshold).
//
//  (b) statistical (reported, soft floor): a single un-quantized RaBitQ batch
//      estimate over uniform-random candidates recovers the exact top-10 well
//      above random choice (10/32 ~= 0.31). Not a high-recall claim -- one
//      fastscan batch with no graph re-rank is inherently coarse, exactly as
//      the sibling RaBitQSpaceIpNormTest documents -- so the hard floor tracks
//      that test's 0.45, and the measured value is printed for the record.
// ===========================================================================
TEST(LaserIpOracle, RankingMatchesExactNegIp) {
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

      // (a) Exact affine-ordering contract: sort by target (1-<q,o>) and by
      // neg_ip (-<q,o>); the permutations must be identical.
      std::vector<std::pair<float, size_t>> by_target;
      std::vector<std::pair<float, size_t>> by_neg_ip;
      std::vector<std::pair<float, size_t>> by_est;
      by_target.reserve(kCand);
      by_neg_ip.reserve(kCand);
      by_est.reserve(kCand);
      for (size_t i = 0; i < kCand; ++i) {
        std::vector<int> bits;
        const auto off = official_ip_factors(cand[i].data(), centroid.data(), dim, &bits);
        const float est =
            unquantized_ip_estimate(off, bits, query.data(), centroid.data(), dim);
        const float ip = static_cast<float>(exact_ip(query.data(), cand[i].data(), dim));
        by_target.emplace_back(1.0F - ip, i);
        by_neg_ip.emplace_back(-ip, i);
        by_est.emplace_back(est, i);
      }
      std::sort(by_target.begin(), by_target.end());
      std::sort(by_neg_ip.begin(), by_neg_ip.end());
      std::sort(by_est.begin(), by_est.end());
      for (size_t i = 0; i < kCand; ++i) {
        ASSERT_EQ(by_target[i].second, by_neg_ip[i].second)
            << "affine-ordering contract broken at rank " << i << " dim=" << dim;
      }

      // (b) statistical recall of the un-quantized estimate vs exact top-k.
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
// T5 -- Oracle 2 end-to-end: the real memqg fastscan path (RaBitQSpace,
// inner_product) tracks exact -<q,o>. Exercises the actual production estimator
// (LUT byte quantization included), the named "rabitq_space.hpp IP path".
// Mirrors the sibling recall harness but also reports the quantized error
// distribution for the scaffolding record.
// ===========================================================================
TEST(LaserIpOracle, MemqgFastscanPathTracksExactIp) {
  using SpaceType = RaBitQSpace<float, float, uint32_t>;
  constexpr uint32_t kDegree = static_cast<uint32_t>(SpaceType::kDegreeBound);  // 32
  constexpr uint32_t kNum = kDegree + 1;
  constexpr uint32_t kCentroid = 0;
  constexpr size_t kTopK = 10;

  for (size_t dim : {128UL, 768UL}) {  // pow2 and 768->1024 pad
    std::mt19937 rng(0xE5750000U + static_cast<uint32_t>(dim));
    std::normal_distribution<float> gauss(0.0F, 1.0F);
    std::vector<float> data(static_cast<size_t>(kNum) * dim);
    for (auto &x : data) {
      x = gauss(rng);
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
      std::normal_distribution<float> noise(0.0F, 0.1F);
      for (size_t d = 0; d < dim; ++d) {
        query[d] = data[(base * dim) + d] + noise(rng);
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
    // Single fastscan batch-of-32, no graph re-rank: floor well above random
    // (10/32 ~= 0.31), matching the sibling RaBitQSpaceIpNormTest methodology.
    EXPECT_GE(recall, 0.45) << "memqg fastscan IP path recall collapsed at dim=" << dim;
  }
}

// ===========================================================================
// AlayaLite IP-as-distance convention lock: simd::ip_sqr returns -<a,b>.
// W1's ip.hpp exact path must preserve this sign so "smaller == more similar"
// holds on the min-heap search path.
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
// Golden: pin the Oracle-1 factor values + exact IP for a fixed seeded set so a
// silent formula change that stays self-consistent is still caught. Regenerate
// with ALAYA_IP_ORACLE_REGEN=1 (or when the file is absent). Located via the
// ALAYA_IP_ORACLE_GOLDEN_DIR compile-def (source tree), so it is committable.
// ===========================================================================
namespace {

struct GoldenRow {
  size_t dim;
  int trial;
  float f_add;
  float f_rescale;
  float f_error;
  double exact_ip_qo;
};

auto compute_golden_rows() -> std::vector<GoldenRow> {
  std::vector<GoldenRow> rows;
  std::mt19937 rng(0x601DEA7U);
  for (size_t dim : {128UL, 256UL}) {
    for (int trial = 0; trial < 8; ++trial) {
      const auto data = random_vec(rng, dim, -3.0F, 3.0F);
      const auto centroid = random_vec(rng, dim, -3.0F, 3.0F);
      const auto query = random_vec(rng, dim, -3.0F, 3.0F);
      std::vector<int> bits;
      const auto off = official_ip_factors(data.data(), centroid.data(), dim, &bits);
      rows.push_back(GoldenRow{dim, trial, off.f_add, off.f_rescale, off.f_error,
                               exact_ip(query.data(), data.data(), dim)});
    }
  }
  return rows;
}

auto golden_path() -> std::string {
#ifdef ALAYA_IP_ORACLE_GOLDEN_DIR
  return std::string(ALAYA_IP_ORACLE_GOLDEN_DIR) + "/ip_oracle_golden.tsv";
#else
  return "ip_oracle_golden.tsv";
#endif
}

void write_golden(const std::vector<GoldenRow> &rows) {
  std::ofstream out(golden_path());
  ASSERT_TRUE(out.is_open()) << "cannot write golden at " << golden_path();
  out.precision(9);
  out << "# dim\ttrial\tf_add\tf_rescale\tf_error\texact_ip_qo\n";
  for (const auto &r : rows) {
    out << r.dim << '\t' << r.trial << '\t' << r.f_add << '\t' << r.f_rescale << '\t' << r.f_error
        << '\t' << r.exact_ip_qo << '\n';
  }
}

}  // namespace

TEST(LaserIpOracle, GoldenFactorsStable) {
  const auto rows = compute_golden_rows();
  const bool regen = std::getenv("ALAYA_IP_ORACLE_REGEN") != nullptr;
  std::ifstream in(golden_path());
  if (regen || !in.good()) {
    write_golden(rows);
    std::cout << "ip_oracle_golden_regenerated=" << golden_path() << '\n';
    SUCCEED() << "golden (re)generated at " << golden_path();
    return;
  }

  std::string line;
  std::getline(in, line);  // header
  size_t idx = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream iss(line);
    GoldenRow g{};
    iss >> g.dim >> g.trial >> g.f_add >> g.f_rescale >> g.f_error >> g.exact_ip_qo;
    ASSERT_LT(idx, rows.size()) << "golden has more rows than expected";
    const auto &r = rows[idx];
    EXPECT_EQ(r.dim, g.dim) << "row " << idx;
    EXPECT_NEAR(r.f_add, g.f_add, 1.0e-3F * std::max(1.0F, std::abs(g.f_add))) << "row " << idx;
    EXPECT_NEAR(r.f_rescale, g.f_rescale, 1.0e-3F * std::max(1.0F, std::abs(g.f_rescale)))
        << "row " << idx;
    EXPECT_NEAR(r.f_error, g.f_error, 1.0e-3F * std::max(1.0F, std::abs(g.f_error))) << "row " << idx;
    EXPECT_NEAR(r.exact_ip_qo, g.exact_ip_qo, 1.0e-3 * std::max(1.0, std::abs(g.exact_ip_qo)))
        << "row " << idx;
    ++idx;
  }
  EXPECT_EQ(idx, rows.size()) << "golden has fewer rows than expected";
}

}  // namespace alaya
