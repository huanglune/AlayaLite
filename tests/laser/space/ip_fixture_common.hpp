// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file ip_fixture_common.hpp
 * @brief Shared support for the LASER inner-product (IP) oracle + kernel tests.
 *
 * This header is consumed by BOTH ip_oracle_test.cpp (W0) and ip_kernel_test.cpp
 * (W1). It provides:
 *
 *   1. A deterministic, cross-platform-stable input generator (make_row_inputs)
 *      that maps a per-row integer seed to float vectors using only integer LCG
 *      arithmetic and an exactly-representable float mapping (multiples of 1/16).
 *      This is used byte-identically by BOTH the offline fixture generator
 *      (tests/laser/space/tools/gen_ip_official_fixture.cpp, which links the official
 *      RaBitQ-Library) and the CI tests, so the tests reproduce the exact inputs
 *      the official factors were computed from -- with no std::normal_distribution
 *      cross-stdlib drift (codex review item 4 / B-LIP-08).
 *
 *   2. A read-only fixture reader (load_ip_fixture). CI tests ONLY read the
 *      immutable fixture; a missing / truncated / unparsable fixture is a HARD
 *      FAILURE, never a skip and never a regeneration (B-LIP-08). Regeneration
 *      lives solely in the offline generator tool, off the CI build graph.
 *
 *   3. The ALGEBRA oracle (algebra_ip_factors) -- a from-scratch re-derivation of
 *      the METRIC_IP one_bit_code_with_factor formula. This is explicitly the
 *      *fourth* cross-check (see the score-domain note), NOT the authoritative
 *      official oracle. The authoritative official values are the fixture rows,
 *      produced by actually linking RaBitQ-Library at the pinned commit.
 *
 * ---------------------------------------------------------------------------
 * SCORE DOMAIN SPEC (codex B-LIP-03 -- written here, mirrored in REPORT sec.3a):
 *
 *   d_exact = -<q,o>                (AlayaLite IP-as-distance; simd::ip_sqr,
 *                                    distance_ip.ipp:39 "smaller == more similar")
 *   d_est   = 1 - <q,o>             (RaBitQ estimator native target; the official
 *                                    METRIC_IP branch AND the in-repo memqg IP
 *                                    branch both target this)
 *   error   : compare (d_est - 1) against d_exact   (both equal -<q,o>)
 *   SymQG full path (NOT used here) : 2 - <q,o>      (subtract the documented
 *                                    constant 2 before comparing)
 *
 *   Ordering: the +1 (or +2) offset is candidate-INDEPENDENT, so ascending sorts
 *   of d_exact and d_est induce the identical permutation. Ordering comparisons
 *   are done in double with a unified candidate-id tie-break; the "+1 is order-
 *   invisible" claim is checked exactly on integer candidate indices, never on
 *   float bitwise equality. All value comparisons use tolerance (SIMD/FMA/Eigen
 *   reduction orders differ) -- see tests/simd/ip_test.cpp:53 for the sibling
 *   convention. There is deliberately NO bitwise-equality assertion on floats.
 * ---------------------------------------------------------------------------
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace alaya::ip_fixture {

// kConstEpsilon, RaBitQ-Library rabitq_impl.hpp:17.
inline constexpr double kOfficialEpsilon = 1.9;

// The RaBitQ-Library commit the official fixture is pinned to. The fixture header
// records the same string; load_ip_fixture cross-checks it so a fixture generated
// against a different commit is a hard failure.
inline constexpr const char *kPinnedRabitqCommit =
    "b1f613d7412a041000d1e71aaa323d3e7554e733";  // pragma: allowlist secret

// ---------------------------------------------------------------------------
// Deterministic input generator (integer LCG -> exactly-representable floats).
// Both the offline generator and the CI tests call this verbatim, so a stored
// per-row seed reproduces the exact float inputs on any IEEE-754 platform.
// ---------------------------------------------------------------------------
inline auto lcg_to_float(uint32_t r) -> float {
  // Top byte -> integer in [-128, 127] -> multiple of 1/16 in [-8, 7.9375].
  // 1/16 is exact in float and the integer is < 2^24, so the product is exact.
  const int byte = static_cast<int>((r >> 24) & 0xFFU);
  return static_cast<float>(byte - 128) * 0.0625F;
}

struct RowInputs {
  std::vector<float> data;
  std::vector<float> centroid;
  std::vector<float> query;
};

// data[dim], then centroid[dim], then query[dim], drawn sequentially from a
// Numerical-Recipes LCG seeded by `seed`. Unsigned overflow is well defined.
inline auto make_row_inputs(uint32_t seed, size_t dim) -> RowInputs {
  uint32_t s = seed;
  const auto next = [&s]() -> float {
    s = (s * 1664525U) + 1013904223U;
    return lcg_to_float(s);
  };
  RowInputs in;
  in.data.resize(dim);
  in.centroid.resize(dim);
  in.query.resize(dim);
  for (size_t i = 0; i < dim; ++i) {
    in.data[i] = next();
  }
  for (size_t i = 0; i < dim; ++i) {
    in.centroid[i] = next();
  }
  for (size_t i = 0; i < dim; ++i) {
    in.query[i] = next();
  }
  return in;
}

// FNV-1a 64 over raw float bytes. Locks that the tests reproduced the exact input
// (data ++ centroid) the fixture's official factors were computed from.
inline auto fnv1a64_bytes(uint64_t h, const void *data, size_t n) -> uint64_t {
  const auto *p = static_cast<const unsigned char *>(data);
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

inline auto input_fingerprint(const std::vector<float> &data, const std::vector<float> &centroid)
    -> uint64_t {
  uint64_t h = 1469598103934665603ULL;  // FNV offset basis
  h = fnv1a64_bytes(h, data.data(), data.size() * sizeof(float));
  h = fnv1a64_bytes(h, centroid.data(), centroid.size() * sizeof(float));
  return h;
}

// ---------------------------------------------------------------------------
// Immutable official fixture (generated offline by linking RaBitQ-Library).
// ---------------------------------------------------------------------------
struct FixtureRow {
  size_t dim = 0;
  int trial = 0;
  uint32_t seed = 0;
  uint64_t input_fnv = 0;
  int popcount = 0;
  float f_add = 0.0F;      // official METRIC_IP f_add     (rabitq_impl.hpp:128)
  float f_rescale = 0.0F;  // official METRIC_IP f_rescale  (rabitq_impl.hpp:130)
  float f_error = 0.0F;    // official METRIC_IP f_error    (rabitq_impl.hpp:131)
};

inline auto fixture_path() -> std::string {
#ifdef ALAYA_IP_FIXTURE_DIR
  return std::string(ALAYA_IP_FIXTURE_DIR) + "/ip_official_fixture.tsv";
#else
  return "ip_official_fixture.tsv";
#endif
}

// Reads the fixture. On ANY problem (missing file, wrong/absent commit pin,
// truncated header, unparsable row, zero rows) returns {} and sets *err. The
// caller MUST treat a non-empty *err as a hard failure. There is no regeneration
// path here by design.
inline auto load_ip_fixture(std::string *err) -> std::vector<FixtureRow> {
  std::vector<FixtureRow> rows;
  const std::string path = fixture_path();
  std::ifstream in(path);
  if (!in.good()) {
    *err = "official IP fixture missing or unreadable at '" + path +
           "' -- regenerate with tests/laser/space/tools/gen_ip_official_fixture.cpp (NOT in CI build)";
    return {};
  }
  bool commit_seen = false;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    if (line[0] == '#') {
      const std::string needle = "commit:";
      const auto pos = line.find(needle);
      if (pos != std::string::npos) {
        std::string commit = line.substr(pos + needle.size());
        // trim surrounding whitespace
        const auto b = commit.find_first_not_of(" \t\r");
        const auto e = commit.find_last_not_of(" \t\r");
        if (b != std::string::npos) {
          commit = commit.substr(b, e - b + 1);
        }
        commit_seen = true;
        if (commit != kPinnedRabitqCommit) {
          *err = "fixture commit pin '" + commit + "' != expected '" + kPinnedRabitqCommit + "'";
          return {};
        }
      }
      continue;
    }
    std::istringstream iss(line);
    FixtureRow r;
    if (!(iss >> r.dim >> r.trial >> r.seed >> r.input_fnv >> r.popcount >> r.f_add >>
          r.f_rescale >> r.f_error)) {
      *err = "malformed fixture row: '" + line + "'";
      return {};
    }
    rows.push_back(r);
  }
  if (!commit_seen) {
    *err = "fixture is missing its '# commit:' pin line";
    return {};
  }
  if (rows.empty()) {
    *err = "fixture parsed but contains zero data rows";
    return {};
  }
  return rows;
}

// ---------------------------------------------------------------------------
// Brute-force exact inner product (double ground truth).
// ---------------------------------------------------------------------------
inline auto exact_ip(const float *a, const float *b, size_t dim) -> double {
  double s = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  return s;
}

// ---------------------------------------------------------------------------
// ALGEBRA oracle (the demoted fourth cross-check -- NOT the official library).
//
// A from-scratch re-derivation of one_bit_code_with_factor(..., METRIC_IP),
// rabitq_impl.hpp:80-136, in float precision (mirrors the library's T=float).
// It exists only to triangulate the immutable official fixture and the in-repo
// memqg path; it is deliberately NOT called an official oracle (B-LIP-08).
// ---------------------------------------------------------------------------
struct AlgebraIpFactors {
  float f_add = 0.0F;
  float f_rescale = 0.0F;
  float f_error = 0.0F;
};

inline auto algebra_ip_factors(const float *data, const float *centroid, size_t dim,
                               std::vector<int> *bits_out) -> AlgebraIpFactors {
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
  // Cauchy-Schwarz guarantees l2_sqr*xucb_sqr >= <residual,xu_cb>^2, so the
  // sqrt argument is mathematically >= 0; at the equality case (residual
  // exactly parallel to xu_cb, e.g. constant vectors) float rounding under a
  // scalar/portable reduction order can land just below 1 and hand sqrt a
  // negative. Clamp to 0: that IS the exact value there, and the official
  // formula (rabitq_impl.hpp:131) is otherwise preserved verbatim.
  const float cs_ratio_minus_one =
      std::max(0.0F, ((l2_sqr * xucb_sqr) / (ip_resi_xucb * ip_resi_xucb)) - 1.0F);
  const float tmp_error = l2_norm * static_cast<float>(kOfficialEpsilon) *
                          std::sqrt(cs_ratio_minus_one / (static_cast<float>(dim) - 1.0F));
  AlgebraIpFactors f{};
  f.f_add = 1.0F - resi_cent + (l2_sqr * ip_cent_xucb / ip_resi_xucb);  // :128
  f.f_rescale = -l2_sqr / ip_resi_xucb;                                 // :130
  f.f_error = 1.0F * tmp_error;                                         // :131 (coeff 1, vs 2 for L2)
  if (bits_out != nullptr) {
    *bits_out = std::move(bits);
  }
  return f;
}

// Un-quantized RaBitQ estimate assembled like RaBitQSpace::QueryComputer and the
// official full_est_dist, with 1-bit codes and NO LUT byte quantization (isolates
// the binary-code factor algebra):
//   est = f_add + g_add + f_rescale * <half_signed, q>,  g_add = -<q,c>.
// f_add/f_rescale are the HALF-SIGN (official/algebra/memqg) factors, paired with
// the half_signed (+-0.5) inner product.
inline auto unquantized_ip_estimate(float f_add, float f_rescale, const std::vector<int> &bits,
                                    const float *query, const float *centroid, size_t dim)
    -> float {
  float half_signed_dot_q = 0.0F;
  for (size_t i = 0; i < dim; ++i) {
    half_signed_dot_q += (static_cast<float>(bits[i]) - 0.5F) * query[i];
  }
  const float g_add = -static_cast<float>(exact_ip(query, centroid, dim));  // query.hpp:104
  return f_add + g_add + (f_rescale * half_signed_dot_q);
}

// g_error = ||q - c|| (query residual norm), query.hpp:105.
inline auto g_error_ip(const float *query, const float *centroid, size_t dim) -> float {
  float s = 0.0F;
  for (size_t i = 0; i < dim; ++i) {
    const float d = query[i] - centroid[i];
    s += d * d;
  }
  return std::sqrt(s);
}

// Dims the offline generator emits and the fixture-backed tests iterate. All are
// multiples of 64 (192 is the non-pow2 case; 64/128/256 anchor the sibling
// tests). one_bit_code_with_factor imposes no dim constraint of its own.
inline auto fixture_dims() -> std::vector<size_t> { return {64, 128, 192, 256}; }
inline constexpr int kTrialsPerDim = 6;

}  // namespace alaya::ip_fixture
