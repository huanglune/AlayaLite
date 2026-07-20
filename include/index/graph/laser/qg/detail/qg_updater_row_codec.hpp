// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "index/graph/laser/qg/qg.hpp"
#include "index/graph/laser/quantization/rabitq.hpp"
#include "simd/fastscan.hpp"

namespace alaya::laser {

/**
 * @brief Inverse of pack_codes for exactly one 32-slot FastScan block.
 *
 * @param padded_dim  vector dimension (multiple of 64)
 * @param block       packed block, `padded_dim * 4` bytes (32 codes)
 * @param binary_out  32 * (padded_dim/64) uint64 words, slot-major
 */
inline void unpack_codes_block(size_t padded_dim, const uint8_t *block, uint64_t *binary_out) {
  const size_t num_codebook = padded_dim / 4;
  const size_t bytes_per_code = padded_dim / 8;
  // Reused scratch: this runs millions of times per drain phase, and per-call
  // heap traffic (with 64 allocating threads) dominated the profile.
  thread_local std::vector<uint8_t> tmp;
  tmp.resize(kBatchSize * bytes_per_code);

  // Invert pack_codes_helper: nibble columns -> per-code byte rows.
  const uint8_t *codes2 = block;
  for (size_t i = 0; i < num_codebook; i += 2) {
    std::array<uint8_t, kBatchSize> col_lo{};
    std::array<uint8_t, kBatchSize> col_hi{};
    for (size_t j = 0; j < 16; ++j) {
      const uint8_t val0 = codes2[j];
      const uint8_t val1 = codes2[j + 16];
      col_lo[::alaya::simd::fastscan::kPackedLaneOrder[j]] = val0 & 15;
      col_lo[::alaya::simd::fastscan::kPackedLaneOrder[j] + 16] = val0 >> 4;
      col_hi[::alaya::simd::fastscan::kPackedLaneOrder[j]] = val1 & 15;
      col_hi[::alaya::simd::fastscan::kPackedLaneOrder[j] + 16] = val1 >> 4;
    }
    for (size_t j = 0; j < kBatchSize; ++j) {
      tmp[j * bytes_per_code + i / 2] = static_cast<uint8_t>(col_lo[j] | (col_hi[j] << 4));
    }
    codes2 += 32;
  }

  // Invert the per-byte nibble swap.
  for (auto &b : tmp) {
    b = static_cast<uint8_t>((b << 4) | (b >> 4));
  }
  // Invert the byte reversal inside each 8-byte (64-bit) group.
  for (size_t i = 0; i < kBatchSize; ++i) {
    for (size_t j = 0; j < padded_dim / 64; ++j) {
      for (size_t k = 0; k < 4; ++k) {
        std::swap(tmp[(i * bytes_per_code) + (8 * j) + k],
                  tmp[(i * bytes_per_code) + (8 * j) + 8 - k - 1]);
      }
    }
  }
  std::memcpy(binary_out, tmp.data(), tmp.size());
}

/** @brief Per-edge RaBitQ payload for one neighbor slot. */
struct EdgePayload {
  std::vector<uint64_t> bin;  // padded_dim/64 sign words (pre-pack layout)
  float triple_x = 0;
  float factor_dq = 0;
  float factor_vq = 0;
  bool degenerate = false;
};

/**
 * @brief Lock-free preparation result for one reverse-edge patch.
 *
 * `codes` contains one slot's sign code in the unpacked FastScan layout
 * (`padded_dim / 8` bytes).  Applying the intent only has to unpack the
 * destination 32-slot block, replace these bytes, and repack it.  The three
 * factors map directly to the row's triple_x/factor_dq/factor_vq arrays.
 */
struct PatchIntent {
  PID target_row = 0;
  PID candidate_pid = 0;
  std::vector<uint8_t> codes;
  std::array<float, 3> factors{};
  float estimated_distance = 0;
  float exact_distance = 0;
  uint64_t row_generation = 0;
};

/** Compact read-only membership filter for one consolidation's dead PIDs. */
class DeadPIDBloom {
 public:
  explicit DeadPIDBloom(size_t expected_count)
      : num_bits_(rounded_bit_count(expected_count)), bits_((num_bits_ + 63) / 64, 0) {}

  void insert(PID pid) {
    for (size_t seed = 0; seed < kNumHashes; ++seed) {
      const size_t bit = hash(pid, seed);
      bits_[bit / 64] |= uint64_t{1} << (bit % 64);
    }
  }

  [[nodiscard]] bool maybe_contains(PID pid) const {
    for (size_t seed = 0; seed < kNumHashes; ++seed) {
      const size_t bit = hash(pid, seed);
      if ((bits_[bit / 64] & (uint64_t{1} << (bit % 64))) == 0) return false;
    }
    return true;
  }

 private:
  static constexpr size_t kBitsPerElement = 10;
  static constexpr size_t kNumHashes = 7;

  [[nodiscard]] static size_t rounded_bit_count(size_t expected_count) {
    const size_t requested = std::max<size_t>(64, expected_count * kBitsPerElement);
    size_t bits = 64;
    while (bits < requested) bits <<= 1U;
    return bits;
  }

  [[nodiscard]] size_t hash(PID pid, size_t seed) const {
    uint64_t h = static_cast<uint64_t>(pid) * 0x9e3779b97f4a7c15ULL;
    h ^= static_cast<uint64_t>(seed) * 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return static_cast<size_t>(h) & (num_bits_ - 1);
  }

  size_t num_bits_;
  std::vector<uint64_t> bits_;
};

/** Per-row format-v2 metadata stored in the page trailer. */
struct QGRowTrailer {
  uint16_t valid_degree = 0;
  uint16_t flags = 0;
};
static_assert(sizeof(QGRowTrailer) == kQGRowTrailerSize);

constexpr uint16_t kQGRowTombstone = 1U << 0U;
constexpr uint16_t kQGRowFree = 1U << 1U;

inline size_t qg_page_trailer_offset(size_t page_size, size_t nodes_per_page, size_t row_slot) {
  if (nodes_per_page == 0 || row_slot >= nodes_per_page ||
      nodes_per_page * sizeof(QGRowTrailer) > page_size) {
    throw std::out_of_range("qg_page_trailer_offset: invalid page geometry/slot");
  }
  return page_size - nodes_per_page * sizeof(QGRowTrailer) + row_slot * sizeof(QGRowTrailer);
}

inline QGRowTrailer qg_read_page_trailer(const char *page,
                                         size_t page_size,
                                         size_t nodes_per_page,
                                         size_t row_slot) {
  QGRowTrailer trailer;
  std::memcpy(&trailer,
              page + qg_page_trailer_offset(page_size, nodes_per_page, row_slot),
              sizeof(trailer));
  return trailer;
}

inline void qg_write_page_trailer(char *page,
                                  size_t page_size,
                                  size_t nodes_per_page,
                                  size_t row_slot,
                                  QGRowTrailer trailer) {
  std::memcpy(page + qg_page_trailer_offset(page_size, nodes_per_page, row_slot),
              &trailer,
              sizeof(trailer));
}

/**
 * @brief Compute the RaBitQ payload of a single edge u->v.
 *
 * Mirrors rabitq_codes()/rabitq_factors() for one row so a patched slot is
 * byte-identical to what the static builder would produce.
 *
 * @param c_rot        rot(u) — rotated main-dim vector of the row owner
 * @param x_rot        rot(v) — rotated main-dim vector of the new neighbor
 * @param padded_dim   main (== padded) dimension
 * @param x_res_sqr    ||v_residual||^2, pre-added to triple_x like the builder
 */
/// Returns a reference to a thread_local payload (valid until this thread's
/// next call) — per-edge heap traffic in the drain phase serialized on glibc.
inline const EdgePayload &make_edge_payload(const float *c_rot,
                                            const float *x_rot,
                                            size_t padded_dim,
                                            float x_res_sqr) {
  thread_local EdgePayload out;
  out.degenerate = false;
  out.bin.assign(padded_dim / 64, 0);

  // Degeneracy pre-check: identical main-dim vectors cannot be sign-encoded.
  double norm_sqr = 0;
  for (size_t j = 0; j < padded_dim; ++j) {
    const double r = static_cast<double>(x_rot[j]) - c_rot[j];
    norm_sqr += r * r;
  }
  if (!(norm_sqr > 0)) {
    out.degenerate = true;
    return out;
  }

  // Delegate to the builder's own kernel with a 1-row matrix so the patched
  // slot is bit-identical to a builder-written slot (same Eigen accumulation
  // order and float rounding). Scratch is thread_local: Eigen's aligned
  // alloc/free per edge was the top drain-phase cost at 64 threads.
  thread_local RowMatrix<float> x;
  thread_local RowMatrix<float> c;
  if (x.cols() != static_cast<int64_t>(padded_dim)) {
    x.resize(1, static_cast<int64_t>(padded_dim));
    c.resize(1, static_cast<int64_t>(padded_dim));
  }
  for (size_t j = 0; j < padded_dim; ++j) {
    x(0, static_cast<int64_t>(j)) = x_rot[j];
    c(0, static_cast<int64_t>(j)) = c_rot[j];
  }
  thread_local std::vector<uint8_t> block;
  block.assign(padded_dim * 4, 0);  // one 32-slot FastScan block
  rabitq_codes(x, c, block.data(), &out.triple_x, &out.factor_dq, &out.factor_vq);
  out.triple_x += x_res_sqr;

  std::vector<uint64_t> bins(kBatchSize * padded_dim / 64);
  unpack_codes_block(padded_dim, block.data(), bins.data());
  std::copy(bins.begin(), bins.begin() + static_cast<int64_t>(padded_dim / 64), out.bin.begin());

  if (!std::isfinite(out.triple_x) || !std::isfinite(out.factor_dq) ||
      !std::isfinite(out.factor_vq)) {
    out.degenerate = true;
  }
  return out;
}

}  // namespace alaya::laser
