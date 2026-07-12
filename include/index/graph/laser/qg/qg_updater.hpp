// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file qg_updater.hpp
 * @brief RESEARCH PROTOTYPE: streaming updates (insert / tombstone-delete /
 * consolidation) for the LASER on-disk quantized graph.
 *
 * Feasibility basis: LASER's per-edge RaBitQ payload (1-bit sign code +
 * triple_x/factor_dq/factor_vq) for an edge u->v depends only on the fixed FHT
 * rotator and the raw (PCA-domain) vectors of u and v — there is no global
 * codebook to retrain. The FastScan 32-slot interleave is a lossless
 * permutation, so a single logical slot can be replaced by unpacking one
 * 32-slot block, swapping the code, and repacking. File addressing is pure id
 * arithmetic; inserts either reuse a consolidated free PID in place or append
 * at EOF.
 *
 * Insert = FreshDiskANN-style: beam search (capturing expanded rows) ->
 * alpha-RobustPrune over captured exact distances -> assemble + append the new
 * row -> patch one reverse edge per chosen neighbor via page RMW.
 *
 * Reverse-edge arms (UpdateParams::backlink_mode):
 *   kNone       — no reverse edges (control arm)
 *   kEvict      — fill a ghost slot, else evict the farthest edge (FastScan
 *                 estimate with the row owner as query) if the new edge is
 *                 shorter ("nearest-only replacement")
 *   kAlphaEvict — kEvict + free alpha-occlusion test against the neighbors of
 *                 v that already sit in this insert's captured search pool
 *   kFullPrune  — read all live neighbors' raw vectors, run full RobustPrune,
 *                 rewrite the whole row (quality reference, R extra reads)
 *
 * Concurrency model (multi-writer inserts, batch three-phase publish):
 *   - The caller runs a parallel append batch with insert_with_id(), or lets
 *     allocate_and_insert() pop reclaimed PIDs. publish() advances the append
 *     watermark and simultaneously removes reused PIDs from the tombstone
 *     result filter after their rows/backlinks are complete.
 *   - Every row write (append, backlink patch, consolidation rewrite) runs
 *     under a striped page-lock table, and bumps a per-page seqlock version
 *     (odd = write in progress). Lock-free search reads validate the version
 *     before/after the pread and retry on a torn page.
 *   - tombstone() / consolidate() must not run concurrently with inserts
 *     (phase separation is the caller's responsibility).
 *
 * Deletes have persistent trailer flags plus a RAM result filter (routing can
 * still traverse tombstoned rows until they become free). consolidate() purges dead out-edges:
 * each dead slot is
 * spliced to the nearest live neighbor of the dead node (ranked with zero
 * extra I/O by FastScan-scanning the dead node's own row with the live row's
 * vector as query), or zeroed back into a free ghost slot (regenerating
 * update headroom) when no candidate exists.
 *
 * Format v2 stores authoritative per-row valid_degree/flags trailers and A/B
 * CRC-protected superblocks. It is not production ready: there is no WAL or
 * torn-page recovery yet; the legacy ghost heuristic exists only in the one-
 * time v1 migration scan.
 */

#pragma once

#include <fcntl.h>
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "index/graph/laser/common.hpp"
#include "index/graph/laser/qg/qg.hpp"
#include "index/graph/laser/qg/qg_query.hpp"
#include "index/graph/laser/quantization/fastscan_impl.hpp"
#include "index/graph/laser/quantization/rabitq.hpp"
#include "index/graph/laser/space/l2.hpp"

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
      col_lo[kPerm0[j]] = val0 & 15;
      col_lo[kPerm0[j] + 16] = val0 >> 4;
      col_hi[kPerm0[j]] = val1 & 15;
      col_hi[kPerm0[j] + 16] = val1 >> 4;
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
  return page_size - nodes_per_page * sizeof(QGRowTrailer) +
         row_slot * sizeof(QGRowTrailer);
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

struct UpdateStats {
  uint64_t inserts = 0;
  uint64_t search_page_reads = 0;
  uint64_t query_page_reads = 0;
  uint64_t seqlock_read_calls = 0;
  uint64_t seqlock_read_retries = 0;
  uint64_t query_seqlock_read_calls = 0;
  uint64_t query_seqlock_read_retries = 0;
  uint64_t patch_page_reads = 0;  // reverse-edge RMW reads (incl. full-prune vector reads)
  uint64_t page_writes = 0;
  uint64_t physical_writes = 0;  // explicit P0.2 name; same counter as page_writes
  uint64_t logical_row_writes = 0;
  uint64_t flush_unique_pages = 0;
  uint64_t drain_us = 0;  // staged-backlink drain wall time (summed over flushes)
  uint64_t flush_us = 0;  // dirty-page writeback wall time (summed over flushes)
  uint64_t free_slot_fills = 0;
  uint64_t evictions = 0;
  uint64_t est_skips = 0;    // new edge longer than current farthest -> skipped
  uint64_t alpha_skips = 0;  // occluded by an already-captured neighbor -> skipped
  uint64_t degenerate_skips = 0;
  uint64_t full_recomputes = 0;
  uint64_t forced_links = 0;  // inserts whose only backlink came from the force policy
  uint64_t evict_tel_samples = 0;
  uint64_t evict_tel_agree = 0;
  std::array<uint64_t, 4> evict_tel_regret{};  // exact rank: 0, 1, 2, 3+
  double evict_tel_relerr_sum = 0;
  uint64_t consolidated_rows = 0;
  uint64_t spliced_slots = 0;
  uint64_t ghosted_slots = 0;
  uint64_t freed_slots = 0;
  uint64_t reused_slots = 0;
  uint64_t gardened_rows = 0;
  uint64_t garden_pump_links = 0;
  uint64_t garden_us = 0;
};

inline std::array<double, 3> grouped_recall(const std::vector<uint64_t> &query_hits,
                                            const std::vector<uint64_t> &query_totals,
                                            const std::vector<int> &groups) {
  if (query_hits.size() != query_totals.size() || query_hits.size() != groups.size()) {
    throw std::invalid_argument("grouped_recall: size mismatch");
  }
  std::array<uint64_t, 3> hits{};
  std::array<uint64_t, 3> totals{};
  for (size_t i = 0; i < groups.size(); ++i) {
    if (groups[i] < 0 || groups[i] >= 3) {
      throw std::invalid_argument("grouped_recall: group outside [0,2]");
    }
    hits[groups[i]] += query_hits[i];
    totals[groups[i]] += query_totals[i];
  }
  std::array<double, 3> out{};
  for (size_t g = 0; g < out.size(); ++g) {
    out[g] = totals[g] == 0 ? -1 : static_cast<double>(hits[g]) / totals[g];
  }
  return out;
}

struct GardenParams {
  double frac = 0.05;
  size_t ef_maintenance = 200;
  size_t pump_budget = 4;
  size_t r_target = 0;
  enum class Policy { kLowIndegree, kRandom } policy = Policy::kLowIndegree;
};

struct UpdateParams {
  size_t ef_insert = 100;
  float alpha = 1.2F;
  size_t prune_pool_cap = 300;  // candidates fed to RobustPrune
  enum class Backlink { kNone, kEvict, kExactEvict, kAlphaEvict, kFullPrune };
  Backlink backlink_mode = Backlink::kAlphaEvict;
  size_t alpha_check_max = 16;  // captured neighbors tested per reverse edge
  double evict_telemetry = 0;   // kEvict decision sampling probability
  size_t max_points = 0;        // page-version table capacity; 0 -> 2*N + 4096
  size_t splice_rerank = 4;     // consolidation: FastScan-recalled candidates reranked exactly
  bool maintain_indegree = false;
  bool direct_io = false;       // route writes through a dedicated O_DIRECT fd.
                                // P0.1 verdict: synchronous per-patch DIO writes LOSE
                                // to buffered pwrite (5.9k vs 7.4k inserts/s @64T) —
                                // the kernel page cache is already a write-back cache.
                                // Keep buffered until the user-space dirty-page cache
                                // (P0.2) moves writes off the hot path; the DIO fd is
                                // meant for its batched flush.
  bool write_cache = true;      // absorb page RMWs in a resident page pool (see cache_cap_pages)
  bool stage_backlinks = false;  // false: patch reverse edges inline at insert time (they land
                                 // in the pool, so same-page coalescing happens regardless, the
                                 // insert threads overlap patch CPU with search IO, and the
                                 // kAlphaEvict capture-pool check works); true: stage per target
                                 // and drain at the batch barrier (phase-separated variant)
  size_t cache_cap_pages = 1U << 20;  // pool high watermark (pages). Hub rows are patched by
                                      // many batches; keeping pages resident coalesces those
                                      // writes across batches (physical writes happen only on
                                      // watermark eviction, consolidate() and finalize())
};

class QGUpdater {
 public:
  QGUpdater(QuantizedGraph &qg, UpdateParams params)
      : qg_(qg),
        params_(params),
        committed_(qg.num_points_),
        allocated_points_(qg.num_points_),
        next_append_id_(static_cast<PID>(qg.num_points_)),
        live_count_(qg.num_points_),
        dim_(qg.dimension_),
        res_dim_(qg.residual_dimension_),
        full_dim_(qg.dimension_ + qg.residual_dimension_),
        pd_(qg.padded_dim_),
        deg_(qg.degree_bound_),
        node_len_(qg.node_len_),
        page_size_(qg.page_size_),
        npp_(qg.node_per_page_),
        write_cache_(page_size_),
        page_locks_(kLockStripes),
        indegree_(std::max(params.max_points == 0 ? 2 * qg.num_points_ + 4096 : params.max_points,
                           qg.num_points_)),
        page_versions_(
            (std::max(params.max_points == 0 ? 2 * qg.num_points_ + 4096 : params.max_points,
                      qg.num_points_) +
             qg.node_per_page_ - 1) /
            qg.node_per_page_),
        hidden_words_((std::max(params.max_points == 0 ? 2 * qg.num_points_ + 4096
                                                       : params.max_points,
                                qg.num_points_) +
                       63) /
                      64) {
    if (qg_.index_file_name_.empty()) {
      throw std::logic_error("QGUpdater: call load_disk_index() first");
    }
    if (qg_.num_points_ > kPidMax || deg_ > std::numeric_limits<uint16_t>::max()) {
      throw std::invalid_argument("QGUpdater: PID/valid-degree field overflow");
    }
    if (npp_ * node_len_ + npp_ * sizeof(QGRowTrailer) > page_size_) {
      throw std::invalid_argument(
          "QGUpdater: page has insufficient slack for format-v2 row trailers");
    }
    for (auto &word : hidden_words_) word.store(0, std::memory_order_relaxed);
    // Reads stay buffered: the OS page cache serves the ~ef_insert row reads
    // per insert from RAM. Writes get their own O_DIRECT fd — buffered pwrite
    // serializes on the ext4 exclusive inode lock (~287k pwrites/s measured),
    // while aligned O_DIRECT overwrites of allocated blocks take it shared.
    // The per-page seqlock orders the buffered read against the completed DIO
    // write (the version flips to even only after pwrite returned, and the DIO
    // write invalidates the stale page-cache range), so mixed-mode IO on the
    // two fds stays coherent inside the process.
    fd_ = ::open(qg_.index_file_name_.c_str(), O_RDWR);
    if (fd_ < 0) {
      throw std::runtime_error("QGUpdater: cannot open for write: " + qg_.index_file_name_ +
                               " errno=" + std::to_string(errno));
    }
    if (params_.direct_io) {
      wfd_ = ::open(qg_.index_file_name_.c_str(), O_RDWR | O_DIRECT);
      direct_io_ = wfd_ >= 0;  // unsupported (e.g. tmpfs) -> buffered writes on fd_
    }
    try {
      load_or_migrate_format();
    } catch (...) {
      if (wfd_ >= 0) ::close(wfd_);
      ::close(fd_);
      wfd_ = -1;
      fd_ = -1;
      throw;
    }
    refresh_routing_snapshot();
    qg_.set_result_filter(&deleted_);
  }

  QGUpdater(const QGUpdater &) = delete;
  auto operator=(const QGUpdater &) -> QGUpdater & = delete;

  ~QGUpdater() {
    if (qg_.result_filter_ == &deleted_) {
      qg_.set_result_filter(nullptr);
    }
    if (fd_ >= 0) {
      ::close(fd_);
    }
    if (wfd_ >= 0) {
      ::close(wfd_);
    }
  }

  [[nodiscard]] size_t num_points() const { return committed_.load(std::memory_order_acquire); }

  [[nodiscard]] size_t allocated_points() const {
    return allocated_points_.load(std::memory_order_acquire);
  }

  [[nodiscard]] uint64_t live_count() const { return live_count_.load(std::memory_order_acquire); }

  [[nodiscard]] uint64_t free_count() const { return free_count_.load(std::memory_order_acquire); }

  [[nodiscard]] PID free_list_head() const {
    return free_list_head_.load(std::memory_order_acquire);
  }

  [[nodiscard]] uint64_t generation() const { return superblock_.generation; }

  [[nodiscard]] int active_superblock_slot() const { return active_superblock_slot_; }

  /** Page count represented by the currently selected/checkpointed superblock. */
  [[nodiscard]] size_t file_pages() const {
    const size_t n = static_cast<size_t>(superblock_.num_points);
    return (n + npp_ - 1) / npp_;
  }

  /** @brief True when writes go through the O_DIRECT fd (false = buffered fallback). */
  [[nodiscard]] bool direct_io() const { return direct_io_; }

  /** Current resident update-pool size and its configured high watermark. */
  [[nodiscard]] size_t pool_pages() const { return write_cache_.total_pages(); }
  [[nodiscard]] size_t cache_cap_pages() const { return params_.cache_cap_pages; }

  /** @brief Snapshot of the (atomic) counters. */
  [[nodiscard]] UpdateStats stats() const {
    UpdateStats s;
    s.inserts = stats_.inserts.load();
    s.search_page_reads = stats_.search_page_reads.load();
    s.query_page_reads = stats_.query_page_reads.load();
    s.seqlock_read_calls = stats_.seqlock_read_calls.load();
    s.seqlock_read_retries = stats_.seqlock_read_retries.load();
    s.query_seqlock_read_calls = stats_.query_seqlock_read_calls.load();
    s.query_seqlock_read_retries = stats_.query_seqlock_read_retries.load();
    s.patch_page_reads = stats_.patch_page_reads.load();
    s.page_writes = stats_.page_writes.load();
    s.physical_writes = s.page_writes;
    s.logical_row_writes = stats_.logical_row_writes.load();
    s.flush_unique_pages = stats_.flush_unique_pages.load();
    s.drain_us = stats_.drain_us.load();
    s.flush_us = stats_.flush_us.load();
    s.free_slot_fills = stats_.free_slot_fills.load();
    s.evictions = stats_.evictions.load();
    s.est_skips = stats_.est_skips.load();
    s.alpha_skips = stats_.alpha_skips.load();
    s.degenerate_skips = stats_.degenerate_skips.load();
    s.full_recomputes = stats_.full_recomputes.load();
    s.forced_links = stats_.forced_links.load();
    s.evict_tel_samples = stats_.evict_tel_samples.load();
    s.evict_tel_agree = stats_.evict_tel_agree.load();
    for (size_t i = 0; i < s.evict_tel_regret.size(); ++i) {
      s.evict_tel_regret[i] = stats_.evict_tel_regret[i].load();
    }
    s.evict_tel_relerr_sum =
        static_cast<double>(stats_.evict_tel_relerr_nano.load()) / 1e9;
    s.consolidated_rows = stats_.consolidated_rows.load();
    s.spliced_slots = stats_.spliced_slots.load();
    s.ghosted_slots = stats_.ghosted_slots.load();
    s.freed_slots = stats_.freed_slots.load();
    s.reused_slots = stats_.reused_slots.load();
    s.gardened_rows = stats_.gardened_rows.load();
    s.garden_pump_links = stats_.garden_pump_links.load();
    s.garden_us = stats_.garden_us.load();
    return s;
  }

  /**
   * Legacy phase-boundary view used only by QuantizedGraph's static eval path.
   * Mixed/concurrent callers must use QGUpdater::search(), whose visibility
   * checks use the atomic hidden bitmap instead of this container.
   */
  [[nodiscard]] const std::unordered_set<PID> &deleted() const { return deleted_; }

  /**
   * @brief Pool-coherent top-k search with exact reranking of expanded rows.
   *
   * The traversal matches search_for_insert(): FastScan supplies frontier
   * distances and every expanded row is reranked by exact full-dimensional
   * L2. The query takes one committed watermark snapshot. Appends published
   * after that snapshot are intentionally invisible; reused PIDs below the
   * watermark may become visible during the query at their hidden-bit clear.
   * Tombstoned/free/dark rows remain traversable but are never results.
   */
  [[nodiscard]] std::vector<PID> search(const float *query_vec, size_t k, size_t ef) {
    if (query_vec == nullptr) throw std::invalid_argument("QGUpdater::search null query");
    if (k == 0) return {};
    const size_t snapshot = committed_.load(std::memory_order_acquire);
    if (snapshot == 0) return {};

    const float *tvec = query_vec;
    std::vector<float> pca_buf;
    if (qg_.pca_transform_.is_loaded()) {
      pca_buf.resize(full_dim_);
      qg_.pca_transform_.transform(query_vec, pca_buf.data());
      tvec = pca_buf.data();
    }

    QGQuery q_obj(tvec, pd_);
    q_obj.query_prepare(qg_.rotator_, qg_.scanner_);
    const float *res_query = tvec + dim_;
    float sqr_qr = 0;
    for (size_t i = 0; i < res_dim_; ++i) sqr_qr += res_query[i] * res_query[i];
    q_obj.set_sqr_qr(sqr_qr);

    const size_t search_ef = std::max(k, ef);
    buffer::SearchBuffer sp(search_ef);
    std::unordered_set<PID> visited;
    visited.reserve(search_ef * 8);
    bool seeded = false;
    const RoutingSnapshot *routing = routing_snapshot_.load(std::memory_order_acquire);
    if (routing != nullptr && !routing->medoids.empty() &&
        routing->medoid_vectors.size() == routing->medoids.size() * full_dim_) {
      PID best_medoid = kPidMax;
      float best = FLT_MAX;
      for (size_t m = 0; m < routing->medoids.size(); ++m) {
        if (routing->medoids[m] >= snapshot) continue;
        const float d =
            space::l2_sqr(tvec, routing->medoid_vectors.data() + full_dim_ * m, dim_);
        if (d < best) {
          best = d;
          best_medoid = routing->medoids[m];
        }
      }
      if (best_medoid != kPidMax) {
        sp.insert(best_medoid, FLT_MAX);
        seeded = true;
      }
    }
    if (routing != nullptr && routing->entry_point < snapshot) {
      sp.insert(routing->entry_point, FLT_MAX);
      seeded = true;
    }
    if (!seeded) return {};

    struct QueryCandidate {
      PID id;
      float distance;
    };
    std::vector<QueryCandidate> candidates;
    candidates.reserve(search_ef);
    std::vector<float> appro(deg_);
    AlignedBuf page(page_size_);
    size_t beam_width = 1;
    while (sp.has_next()) {
      // Match QuantizedGraph's current eval traversal cadence: dequeue a
      // growing beam before any row in that beam contributes neighbors. The
      // reads themselves stay synchronous through the updater pool, but this
      // preserves the diversity effect of the native async beam (2,4,8,16).
      beam_width = std::min<size_t>(16, beam_width * 2);
      std::vector<PID> frontier;
      frontier.reserve(beam_width);
      while (sp.has_next() && frontier.size() < beam_width) {
        const PID cur = sp.pop();
        if (!visited.insert(cur).second || cur >= snapshot) continue;
        frontier.push_back(cur);
      }
      for (PID cur : frontier) {
        read_node_page(cur, page.data(), /*query_read=*/true);
        stats_.query_page_reads.fetch_add(1, std::memory_order_relaxed);
        const char *row = page.data() + node_offset_in_page(cur);
        const auto *row_f = reinterpret_cast<const float *>(row);
        const QGRowTrailer trailer = row_trailer(page.data(), cur);

        float sqr_y = space::l2_sqr(tvec, row_f, dim_);
        qg_.scanner_.scan_neighbors(appro.data(),
                                    q_obj.lut().data(),
                                    sqr_y,
                                    q_obj.lower_val(),
                                    q_obj.width(),
                                    q_obj.sqr_qr(),
                                    q_obj.sumq(),
                                    reinterpret_cast<const uint8_t *>(row + code_off_bytes()),
                                    reinterpret_cast<const float *>(row + factor_off_bytes()));
        const auto *nbs = reinterpret_cast<const PID *>(row + neighbor_off_bytes());
        for (size_t j = 0; j < trailer.valid_degree; ++j) {
          const PID nb = nbs[j];
          if (nb >= snapshot || visited.count(nb) != 0 || sp.is_full(appro[j])) continue;
          sp.insert(nb, appro[j]);
        }

        if (res_dim_ > 0) sqr_y += space::l2_sqr(row_f + dim_, res_query, res_dim_);
        // A hidden bit is checked after the row copy, so a concurrent tombstone
        // cannot leak an old live row. Trailer flags independently reject the
        // safe-side publish intermediate (cleared hidden is always last).
        if ((trailer.flags & (kQGRowTombstone | kQGRowFree)) == 0 && !is_hidden(cur)) {
          candidates.push_back({cur, sqr_y});
        }
      }
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) {
      return a.distance != b.distance ? a.distance < b.distance : a.id < b.id;
    });
    std::vector<PID> out;
    out.reserve(std::min(k, candidates.size()));
    for (const auto &candidate : candidates) {
      // Close the row-read -> return window for tombstones. A tombstone that
      // begins after this acquire may linearize after the query return.
      if (candidate.id < snapshot && !is_hidden(candidate.id)) out.push_back(candidate.id);
      if (out.size() == k) break;
    }
    return out;
  }

  [[nodiscard]] int32_t indegree(PID id) const {
    if (!params_.maintain_indegree || id >= indegree_.size()) return 0;
    return indegree_[id].load(std::memory_order_relaxed);
  }

  /** Rebuild the optional RAM indegree index from the current committed graph. */
  void init_indegree(size_t num_threads) {
    if (!params_.maintain_indegree) return;
    for (auto &v : indegree_) v.store(0, std::memory_order_relaxed);
    const size_t n = committed_.load(std::memory_order_acquire);
    const int nt = static_cast<int>(std::max<size_t>(1, num_threads));
#pragma omp parallel for num_threads(nt) schedule(dynamic, 256)
    for (int64_t ui = 0; ui < static_cast<int64_t>(n); ++ui) {
      if (is_hidden(static_cast<PID>(ui))) continue;
      AlignedBuf page(page_size_);
      read_node_page(static_cast<PID>(ui), page.data());
      const char *row = page.data() + node_offset_in_page(static_cast<PID>(ui));
      const auto *ids = reinterpret_cast<const PID *>(row + neighbor_off_bytes());
      const size_t degree = row_trailer(page.data(), static_cast<PID>(ui)).valid_degree;
      for (size_t j = 0; j < degree; ++j) {
        if (ids[j] < indegree_.size()) {
          indegree_[ids[j]].fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
  }

  /**
   * @brief Tombstone a node: routing still passes through, results filter it.
   * Safe concurrently with QGUpdater::search(). Inserts/consolidate remain
   * phase-separated update operations.
   */
  void tombstone(PID id) {
    if (id >= allocated_points()) {
      throw std::out_of_range("QGUpdater::tombstone id outside allocated range");
    }
    // This is the deletion visibility point. A racing reader that still has
    // the old row may route through it, but can no longer return the PID.
    if (!mark_hidden(id)) return;
    mirror_deleted_insert(id);
    repair_routing_roots(id);
    const std::lock_guard<std::mutex> guard(page_lock(id));
    modify_node_page(id, [&](char *page) {
      QGRowTrailer trailer = row_trailer(page, id);
      add_row_indegree(page + node_offset_in_page(id), trailer.valid_degree, -1);
      trailer.flags |= kQGRowTombstone;
      set_row_trailer(page, id, trailer);
      return true;
    });
    live_count_.fetch_sub(1, std::memory_order_acq_rel);
  }

  /** @brief Serial allocate/reuse, insert, and publish. */
  PID insert(const float *vec) {
    const PID id = allocate_and_insert(vec);
    flush(1);
    publish(allocated_points());
    return id;
  }

  /** Allocate a reclaimed PID when available, otherwise append a new PID. */
  PID allocate_and_insert(const float *vec) {
    PID id = pop_free_slot();
    const bool reused = id != kPidMax;
    if (!reused) {
      id = next_append_id_.fetch_add(1, std::memory_order_acq_rel);
      if (id == kPidMax) {
        throw std::overflow_error("QGUpdater: PID space exhausted");
      }
      note_allocated(static_cast<size_t>(id) + 1);
    }
    insert_with_id_impl(vec, id, reused);
    if (reused) {
      const std::lock_guard<std::mutex> guard(pending_reused_mutex_);
      pending_reused_.push_back(id);
      stats_.reused_slots++;
    }
    return id;
  }

  /**
   * @brief Publish the committed watermark after a batch barrier. All rows
   * and backlinks for ids below `new_committed` must already be written.
   */
  void publish(size_t new_committed) {
    const size_t old_committed = committed_.load(std::memory_order_acquire);
    if (new_committed < old_committed || new_committed > allocated_points()) {
      throw std::invalid_argument("QGUpdater::publish invalid committed watermark");
    }

    std::vector<PID> reused;
    {
      const std::lock_guard<std::mutex> guard(pending_reused_mutex_);
      reused.swap(pending_reused_);
    }
    // A reused id is below the existing watermark.  Its trailer is made live
    // first, then the result-filter tombstone is erased; that erase is the
    // reused-row visibility point.  The ordinary committed watermark only
    // gates newly appended ids.
    for (PID id : reused) {
      const std::lock_guard<std::mutex> guard(page_lock(id));
      modify_node_page(id, [&](char *page) {
        QGRowTrailer trailer = row_trailer(page, id);
        trailer.flags &= static_cast<uint16_t>(~(kQGRowTombstone | kQGRowFree));
        set_row_trailer(page, id, trailer);
        return true;
      });
      // Preserve the P2 ordering under concurrent readers: trailer first,
      // then both legacy and lock-free filters. Observing any intermediate
      // state therefore keeps the row dark (safe-side false negative).
      mirror_deleted_erase(id);
      clear_hidden(id);
    }
    live_count_.fetch_add(reused.size() + (new_committed - old_committed),
                          std::memory_order_acq_rel);
    committed_.store(new_committed, std::memory_order_release);
    qg_.num_points_ = new_committed;
  }

  /**
   * @brief Batch barrier maintenance: drain staged backlinks, then enforce the
   * pool watermark. Pages stay resident across batches (searches read through
   * the pool), so no writeback happens here until the pool exceeds
   * cache_cap_pages — cross-batch coalescing is the point of the pool.
   */
  void flush(size_t num_threads) {
    const auto t0 = std::chrono::steady_clock::now();
    drain_staged_edges(num_threads);
    const auto t1 = std::chrono::steady_clock::now();
    if (write_cache_.total_pages() > params_.cache_cap_pages) {
      flush_dirty(num_threads);
      evict_clean(params_.cache_cap_pages / 2);
    }
    const auto t2 = std::chrono::steady_clock::now();
    stats_.drain_us.fetch_add(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    stats_.flush_us.fetch_add(
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
  }

  /** Force dirty update-cache pages to the index file without advancing metadata. */
  void writeback(size_t num_threads) {
    drain_staged_edges(num_threads);
    flush_dirty(num_threads);
  }

  /**
   * @brief Multi-writer insert with a caller-assigned dense id (batch phase).
   *
   * Batch contract (three-phase publish, no per-id tickets): the caller runs a
   * parallel batch of insert_with_id() over a dense id range, then calls
   * publish(base + batch_size) once after the batch barrier. Searches inside
   * the batch see the pre-batch snapshot only (mini-batch semantics: batch
   * members cannot link to each other); every new row and all its backlinks
   * are fully materialized in the update cache/file before the watermark
   * moves.
   */
  void insert_with_id(const float *vec, PID id) {
    note_allocated(static_cast<size_t>(id) + 1);
    insert_with_id_impl(vec, id, false);
  }

 private:
  void insert_with_id_impl(const float *vec, PID id, bool reused) {
    if (page_index(id) >= page_versions_.size()) {
      throw std::runtime_error("QGUpdater: id exceeds max_points capacity");
    }
    // --- PCA transform (same path queries take) ---
    const float *tvec = vec;
    std::vector<float> pca_buf;
    if (qg_.pca_transform_.is_loaded()) {
      pca_buf.resize(full_dim_);
      qg_.pca_transform_.transform(vec, pca_buf.data());
      tvec = pca_buf.data();
    }

    // --- beam search with row capture (only committed ids) ---
    std::vector<CapturedNode> pool;
    search_for_insert(tvec, pool);

    // --- alpha-RobustPrune over exact distances ---
    std::vector<size_t> sel;
    robust_prune(pool, sel);

    // --- assemble + append the new row ---
    std::vector<const float *> nb_vecs;
    std::vector<PID> nb_ids;
    nb_vecs.reserve(sel.size());
    nb_ids.reserve(sel.size());
    for (size_t idx : sel) {
      nb_vecs.push_back(pool[idx].raw());
      nb_ids.push_back(pool[idx].id);
    }
    std::vector<char> row(node_len_);
    assemble_row(row.data(), tvec, nb_vecs, nb_ids);
    append_node(id, row.data(), nb_ids.size(), reused);

    // --- reverse edges ---
    if (params_.backlink_mode != UpdateParams::Backlink::kNone && !sel.empty()) {
      if (params_.stage_backlinks && params_.backlink_mode != UpdateParams::Backlink::kFullPrune) {
        for (size_t rank = 0; rank < sel.size(); ++rank) {
          const auto &v = pool[sel[rank]];
          stage_edge({v.id, id, v.dist, rank == 0});
        }
        stats_.inserts++;
        return;
      }
      std::unordered_map<PID, const CapturedNode *> captured;
      if (params_.backlink_mode == UpdateParams::Backlink::kAlphaEvict) {
        captured.reserve(pool.size());
        for (const auto &c : pool) {
          captured.emplace(c.id, &c);
        }
      }
      size_t links = 0;
      for (size_t idx : sel) {
        if (params_.backlink_mode == UpdateParams::Backlink::kFullPrune) {
          links += static_cast<size_t>(full_reverse_recompute(pool[idx], id, tvec));
        } else {
          links += static_cast<size_t>(
              patch_reverse_edge(pool[idx], id, tvec, captured, /*force=*/false));
        }
      }
      // Reachability guarantee: a node with zero in-edges is unsearchable, so
      // force one link onto the closest selected neighbor (evicting its
      // farthest edge unconditionally).
      if (links == 0) {
        if (patch_reverse_edge(pool[sel[0]], id, tvec, captured, /*force=*/true)) {
          stats_.forced_links++;
        }
      }
    }

    stats_.inserts++;
  }

 public:

  /**
   * @brief Purge dead out-edges from all live rows (FreshDiskANN-style
   * consolidation instantiated on the LASER layout).
   *
   * For each live row u with tombstoned neighbors: rank the dead neighbor d's
   * own adjacency for u with a single FastScan pass over d's row (codes are
   * centered at d, but the estimator targets ||u - n_i|| directly, no extra
   * I/O), splice the nearest live candidate into the slot, or remove the slot
   * from the packed valid prefix when no candidate exists. When
   * `reclaim_slots` is true, the purged tombstoned rows then enter the LIFO
   * free-list. Must not run concurrently with inserts.
   */
  void consolidate(size_t num_threads, size_t r_target = 0, bool reclaim_slots = true) {
    const size_t n = committed_.load(std::memory_order_acquire);
    const size_t target = r_target == 0 ? deg_ : std::min(r_target, deg_);
    const int nt = static_cast<int>(std::max<size_t>(1, num_threads));
#pragma omp parallel for num_threads(nt) schedule(dynamic, 256)
    for (int64_t ui = 0; ui < static_cast<int64_t>(n); ++ui) {
      const PID u = static_cast<PID>(ui);
      if (is_hidden(u)) {
        continue;
      }
      consolidate_row(u, n, target);
    }
    drain_staged_edges(num_threads);
    flush_dirty(num_threads);
    if (reclaim_slots) {
      // Only after every live row has purged dead out-edges and that purge has
      // reached the file may a tombstoned row become a reusable free slot.
      std::vector<PID> eligible = deleted_snapshot();
      eligible.erase(std::remove_if(eligible.begin(), eligible.end(), [n](PID id) {
                       return id >= n;
                     }),
                     eligible.end());
      std::sort(eligible.begin(), eligible.end());
      for (PID id : eligible) push_free_slot(id);
      flush_dirty(num_threads);
    }
  }

  /** Refresh a deterministic budget of live rows; phase-separated from updates. */
  void garden(size_t num_threads, const GardenParams &gp) {
    if (!params_.maintain_indegree) {
      throw std::logic_error("QGUpdater::garden requires maintain_indegree");
    }
    const auto t0 = std::chrono::steady_clock::now();
    const size_t n = committed_.load(std::memory_order_acquire);
    std::vector<PID> live;
    live.reserve(std::min<uint64_t>(n, live_count_.load(std::memory_order_acquire)));
    for (size_t i = 0; i < n; ++i) {
      if (!is_hidden(static_cast<PID>(i))) live.push_back(static_cast<PID>(i));
    }
    const double frac = std::clamp(gp.frac, 0.0, 1.0);
    size_t k = static_cast<size_t>(std::ceil(frac * static_cast<double>(live.size())));
    k = std::min(k, live.size());
    if (gp.policy == GardenParams::Policy::kLowIndegree) {
      std::sort(live.begin(), live.end(), [&](PID a, PID b) {
        const int32_t ia = indegree(a), ib = indegree(b);
        return ia != ib ? ia < ib : a < b;
      });
    } else {
      std::mt19937_64 rng(0x4c41534552ULL);
      std::shuffle(live.begin(), live.end(), rng);
    }
    live.resize(k);
    const size_t target = gp.r_target == 0 ? deg_ : std::min(gp.r_target, deg_);
    const int nt = static_cast<int>(std::max<size_t>(1, num_threads));
#pragma omp parallel for num_threads(nt) schedule(dynamic, 1)
    for (int64_t i = 0; i < static_cast<int64_t>(live.size()); ++i) {
      garden_row(live[static_cast<size_t>(i)], gp, target);
    }
    flush_dirty(num_threads);
    stats_.garden_us.fetch_add(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0)
            .count());
  }

  /** Persist dirty pages and atomically advance the alternate A/B superblock. */
  void checkpoint() {
    const std::lock_guard<std::mutex> checkpoint_guard(checkpoint_mutex_);
    drain_staged_edges(1);
    flush_dirty(1);
    const size_t n = allocated_points_.load(std::memory_order_acquire);
    const size_t page_num = (n + npp_ - 1) / npp_;
    const uint64_t file_size = kSectorLen + page_num * page_size_;
    if (::ftruncate(fd_, static_cast<off_t>(file_size)) != 0) {
      throw std::runtime_error("QGUpdater: ftruncate failed");
    }
    QGSuperblockV2 next = superblock_;
    next.magic = kQGSuperblockMagic;
    next.format_version = kQGFormatVersion;
    next.generation = superblock_.generation + 1;
    next.num_points = n;
    next.live_count = live_count_.load(std::memory_order_acquire);
    next.free_list_head = free_list_head_.load(std::memory_order_acquire);
    next.free_count = free_count_.load(std::memory_order_acquire);
    next.entry_point = qg_.entry_point_;
    next.file_size = file_size;
    next.checksum = qg_superblock_checksum(next);
    const int next_slot = active_superblock_slot_ == 0 ? 1 : 0;
    write_superblock(next_slot, next);
    superblock_ = next;
    active_superblock_slot_ = next_slot;
    qg_.num_points_ = committed_.load(std::memory_order_acquire);
  }

  void finalize() { checkpoint(); }

  /**
   * Migration-only v1 ghost-slot heuristic.  Format-v2 update paths never use
   * this predicate; they read valid_degree from the authoritative trailer.
   */
  bool is_v1_ghost_slot(const char *row, size_t slot) const {
    const auto *ids = reinterpret_cast<const PID *>(row + neighbor_off_bytes());
    if (ids[slot] != 0) {
      return false;
    }
    const auto *fac = reinterpret_cast<const float *>(row + factor_off_bytes());
    if (fac[slot] != 0 || fac[deg_ + slot] != 0 || fac[2 * deg_ + slot] != 0) {
      return false;
    }
    // all code bits of this slot zero?
    const uint8_t *block =
        reinterpret_cast<const uint8_t *>(row + code_off_bytes()) + (slot / kBatchSize) * pd_ * 4;
    std::vector<uint64_t> bins(kBatchSize * pd_ / 64);
    unpack_codes_block(pd_, block, bins.data());
    const uint64_t *b = &bins[(slot % kBatchSize) * (pd_ / 64)];
    for (size_t w = 0; w < pd_ / 64; ++w) {
      if (b[w] != 0) {
        return false;
      }
    }
    return true;
  }

  // exposed for unit tests
  [[nodiscard]] size_t code_off_bytes() const { return qg_.code_offset_ * sizeof(float); }
  [[nodiscard]] size_t factor_off_bytes() const { return qg_.factor_offset_ * sizeof(float); }
  [[nodiscard]] size_t neighbor_off_bytes() const { return qg_.neighbor_offset_ * sizeof(float); }

  [[nodiscard]] QGRowTrailer trailer(PID id) {
    if (id >= allocated_points()) throw std::out_of_range("QGUpdater::trailer id");
    AlignedBuf page(page_size_);
    read_node_page(id, page.data());
    return row_trailer(page.data(), id);
  }

  [[nodiscard]] size_t trailer_offset_in_page(PID id) const {
    return qg_page_trailer_offset(page_size_, npp_, id % npp_);
  }

  /** @brief Assemble a full node row exactly like the static builder. */
  void assemble_row(char *row,
                    const float *vec,
                    const std::vector<const float *> &nb_vecs,
                    const std::vector<PID> &nb_ids) {
    assert(nb_vecs.size() == nb_ids.size());
    assert(nb_ids.size() <= deg_);
    std::memset(row, 0, node_len_);
    std::memcpy(row, vec, full_dim_ * sizeof(float));

    const size_t cur_degree = nb_ids.size();
    auto *ids_out = reinterpret_cast<PID *>(row + neighbor_off_bytes());
    for (size_t i = 0; i < cur_degree; ++i) {
      ids_out[i] = nb_ids[i];
    }
    if (cur_degree == 0) {
      return;
    }

    RowMatrix<float> x_pad(cur_degree, pd_);
    RowMatrix<float> c_pad(1, pd_);
    x_pad.setZero();
    c_pad.setZero();
    for (size_t i = 0; i < cur_degree; ++i) {
      std::copy(nb_vecs[i], nb_vecs[i] + dim_, &x_pad(static_cast<int64_t>(i), 0));
    }
    std::copy(vec, vec + dim_, &c_pad(0, 0));

    RowMatrix<float> x_rot(cur_degree, pd_);
    RowMatrix<float> c_rot(1, pd_);
    for (int64_t i = 0; i < static_cast<int64_t>(cur_degree); ++i) {
      qg_.rotator_.rotate(&x_pad(i, 0), &x_rot(i, 0));
    }
    qg_.rotator_.rotate(&c_pad(0, 0), &c_rot(0, 0));

    auto *fac_ptr = reinterpret_cast<float *>(row + factor_off_bytes());
    auto *code_ptr = reinterpret_cast<uint8_t *>(row + code_off_bytes());
    float *triple_x = fac_ptr;
    float *factor_dq = triple_x + deg_;
    float *factor_vq = factor_dq + deg_;
    rabitq_codes(x_rot, c_rot, code_ptr, triple_x, factor_dq, factor_vq);

    for (size_t i = 0; i < cur_degree; ++i) {
      const float *residual = nb_vecs[i] + dim_;
      float sqr_xr = 0;
      for (size_t j = 0; j < res_dim_; ++j) {
        sqr_xr += residual[j] * residual[j];
      }
      triple_x[i] += sqr_xr;
    }
  }

 private:
  static constexpr size_t kLockStripes = 4096;

  struct AtomicStats {
    std::atomic<uint64_t> inserts{0};
    std::atomic<uint64_t> search_page_reads{0};
    std::atomic<uint64_t> query_page_reads{0};
    std::atomic<uint64_t> seqlock_read_calls{0};
    std::atomic<uint64_t> seqlock_read_retries{0};
    std::atomic<uint64_t> query_seqlock_read_calls{0};
    std::atomic<uint64_t> query_seqlock_read_retries{0};
    std::atomic<uint64_t> patch_page_reads{0};
    std::atomic<uint64_t> page_writes{0};
    std::atomic<uint64_t> logical_row_writes{0};
    std::atomic<uint64_t> flush_unique_pages{0};
    std::atomic<uint64_t> drain_us{0};
    std::atomic<uint64_t> flush_us{0};
    std::atomic<uint64_t> free_slot_fills{0};
    std::atomic<uint64_t> evictions{0};
    std::atomic<uint64_t> est_skips{0};
    std::atomic<uint64_t> alpha_skips{0};
    std::atomic<uint64_t> degenerate_skips{0};
    std::atomic<uint64_t> full_recomputes{0};
    std::atomic<uint64_t> forced_links{0};
    std::atomic<uint64_t> evict_tel_samples{0};
    std::atomic<uint64_t> evict_tel_agree{0};
    std::array<std::atomic<uint64_t>, 4> evict_tel_regret{};
    std::atomic<uint64_t> evict_tel_relerr_nano{0};
    std::atomic<uint64_t> consolidated_rows{0};
    std::atomic<uint64_t> spliced_slots{0};
    std::atomic<uint64_t> ghosted_slots{0};
    std::atomic<uint64_t> freed_slots{0};
    std::atomic<uint64_t> reused_slots{0};
    std::atomic<uint64_t> gardened_rows{0};
    std::atomic<uint64_t> garden_pump_links{0};
    std::atomic<uint64_t> garden_us{0};
  };

  struct CapturedNode {
    PID id = 0;
    float dist = 0;          // exact full-dim squared L2 to the inserted vector
    std::vector<float> vec;  // raw (PCA-domain) vector only — the rest of the
                             // row is re-read fresh under the page lock
    [[nodiscard]] const float *raw() const { return vec.data(); }
  };

  struct RoutingSnapshot {
    PID entry_point = kPidMax;
    std::vector<PID> medoids;
    std::vector<float> medoid_vectors;
  };

  /// O_DIRECT alignment unit (buffer address, file offset, and length).
  static constexpr size_t kDioAlign = 4096;

  /// Page-aligned IO buffer; O_DIRECT rejects unaligned user memory.
  struct AlignedBuf {
    explicit AlignedBuf(size_t len) {
      if (::posix_memalign(reinterpret_cast<void **>(&p_), kDioAlign, len) != 0) {
        throw std::bad_alloc();
      }
    }
    AlignedBuf(const AlignedBuf &) = delete;
    auto operator=(const AlignedBuf &) -> AlignedBuf & = delete;
    ~AlignedBuf() { ::free(p_); }
    [[nodiscard]] char *data() const { return p_; }

   private:
    char *p_ = nullptr;
  };

  [[nodiscard]] QGRowTrailer row_trailer(const char *page, PID id) const {
    return qg_read_page_trailer(page, page_size_, npp_, id % npp_);
  }

  void set_row_trailer(char *page, PID id, QGRowTrailer trailer) const {
    if (trailer.valid_degree > deg_) {
      throw std::runtime_error("QGUpdater: corrupt valid_degree");
    }
    qg_write_page_trailer(page, page_size_, npp_, id % npp_, trailer);
  }

  [[nodiscard]] bool is_hidden(PID id) const {
    const size_t wi = static_cast<size_t>(id) >> 6U;
    if (wi >= hidden_words_.size()) return true;
    const uint64_t mask = uint64_t{1} << (id & 63U);
    return (hidden_words_[wi].load(std::memory_order_acquire) & mask) != 0;
  }

  /** Mark tombstone/free/dark before touching any mutable row state. */
  bool mark_hidden(PID id) {
    const size_t wi = static_cast<size_t>(id) >> 6U;
    if (wi >= hidden_words_.size()) {
      throw std::runtime_error("QGUpdater: visibility id exceeds max_points capacity");
    }
    const uint64_t mask = uint64_t{1} << (id & 63U);
    const uint64_t old = hidden_words_[wi].fetch_or(mask, std::memory_order_acq_rel);
    return (old & mask) == 0;
  }

  /**
   * Publish a reused row. The release clear is its visibility point: all row
   * bytes and the cleared trailer happen-before a query that observes live.
   */
  void clear_hidden(PID id) {
    const size_t wi = static_cast<size_t>(id) >> 6U;
    if (wi >= hidden_words_.size()) {
      throw std::runtime_error("QGUpdater: visibility id exceeds max_points capacity");
    }
    const uint64_t mask = uint64_t{1} << (id & 63U);
    hidden_words_[wi].fetch_and(~mask, std::memory_order_release);
  }

  void reset_hidden() {
    for (auto &word : hidden_words_) word.store(0, std::memory_order_relaxed);
  }

  void mirror_deleted_insert(PID id) {
    const std::lock_guard<std::mutex> guard(deleted_mutex_);
    deleted_.insert(id);
  }

  void mirror_deleted_erase(PID id) {
    const std::lock_guard<std::mutex> guard(deleted_mutex_);
    deleted_.erase(id);
  }

  [[nodiscard]] std::vector<PID> deleted_snapshot() const {
    const std::lock_guard<std::mutex> guard(deleted_mutex_);
    return {deleted_.begin(), deleted_.end()};
  }

  /**
   * Publish immutable routing seeds without a query-side lock. Old snapshots
   * stay owned until updater destruction, so an in-flight query can safely
   * finish even when tombstone() relocates a seed concurrently.
   */
  void refresh_routing_snapshot() {
    auto next = std::make_unique<RoutingSnapshot>();
    next->entry_point = qg_.entry_point_;
    next->medoids = qg_.medoids_;
    next->medoid_vectors = qg_.medoids_vector_;
    const RoutingSnapshot *published = next.get();
    const std::lock_guard<std::mutex> guard(routing_snapshot_mutex_);
    routing_snapshots_.push_back(std::move(next));
    routing_snapshot_.store(published, std::memory_order_release);
  }

  void note_allocated(size_t end) {
    if (end > static_cast<size_t>(kPidMax)) {
      throw std::overflow_error("QGUpdater: PID space exhausted");
    }
    size_t current = allocated_points_.load(std::memory_order_acquire);
    while (current < end &&
           !allocated_points_.compare_exchange_weak(
               current, end, std::memory_order_acq_rel, std::memory_order_acquire)) {
    }
    PID next = next_append_id_.load(std::memory_order_acquire);
    const PID desired = static_cast<PID>(end);
    while (next < desired &&
           !next_append_id_.compare_exchange_weak(
               next, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
    }
  }

  void write_superblock(int slot, const QGSuperblockV2 &sb) {
    if (slot < 0 || slot >= static_cast<int>(kQGSuperblockCopies) ||
        !qg_superblock_valid(sb)) {
      throw std::invalid_argument("QGUpdater::write_superblock invalid slot/block");
    }
    const off_t off = static_cast<off_t>(slot * kQGSuperblockSize);
    ssize_t written;
    do {
      written = ::pwrite(fd_, &sb, sizeof(sb), off);
    } while (written < 0 && errno == EINTR);
    if (written != static_cast<ssize_t>(sizeof(sb))) {
      throw std::runtime_error("QGUpdater: superblock pwrite failed/short errno=" +
                               std::to_string(errno));
    }
    if (::fsync(fd_) != 0) {
      throw std::runtime_error("QGUpdater: superblock fsync failed");
    }
  }

  size_t compact_v1_row(char *row) {
    if (deg_ % kBatchSize != 0) {
      throw std::runtime_error("QGUpdater: degree must be a multiple of FastScan batch size");
    }
    std::vector<size_t> live_slots;
    live_slots.reserve(deg_);
    for (size_t slot = 0; slot < deg_; ++slot) {
      if (!is_v1_ghost_slot(row, slot)) live_slots.push_back(slot);
    }

    const size_t words = pd_ / 64;
    std::vector<uint64_t> old_bins(deg_ * words);
    std::vector<uint64_t> block_bins(kBatchSize * words);
    const auto *code = reinterpret_cast<const uint8_t *>(row + code_off_bytes());
    for (size_t block = 0; block < deg_ / kBatchSize; ++block) {
      unpack_codes_block(pd_, code + block * pd_ * 4, block_bins.data());
      std::copy(block_bins.begin(),
                block_bins.end(),
                old_bins.begin() + block * kBatchSize * words);
    }
    std::vector<uint64_t> compact_bins(deg_ * words, 0);
    const auto *old_fac = reinterpret_cast<const float *>(row + factor_off_bytes());
    const auto *old_ids = reinterpret_cast<const PID *>(row + neighbor_off_bytes());
    std::vector<float> compact_fac(3 * deg_, 0.0F);
    std::vector<PID> compact_ids(deg_, 0);
    for (size_t dst = 0; dst < live_slots.size(); ++dst) {
      const size_t src = live_slots[dst];
      std::copy(old_bins.begin() + src * words,
                old_bins.begin() + (src + 1) * words,
                compact_bins.begin() + dst * words);
      compact_fac[dst] = old_fac[src];
      compact_fac[deg_ + dst] = old_fac[deg_ + src];
      compact_fac[2 * deg_ + dst] = old_fac[2 * deg_ + src];
      compact_ids[dst] = old_ids[src];
    }
    auto *new_code = reinterpret_cast<uint8_t *>(row + code_off_bytes());
    for (size_t block = 0; block < deg_ / kBatchSize; ++block) {
      pack_codes(pd_,
                 compact_bins.data() + block * kBatchSize * words,
                 kBatchSize,
                 new_code + block * pd_ * 4);
    }
    std::memcpy(row + factor_off_bytes(), compact_fac.data(), compact_fac.size() * sizeof(float));
    std::memcpy(row + neighbor_off_bytes(), compact_ids.data(), compact_ids.size() * sizeof(PID));
    return live_slots.size();
  }

  void migrate_v1(const char *header) {
    std::array<uint64_t, kSectorLen / sizeof(uint64_t)> metas{};
    std::memcpy(metas.data(), header, kSectorLen);
    const size_t n = qg_.num_points_;
    const size_t pages = (n + npp_ - 1) / npp_;
    std::clog << "[QGUpdater] LASER v1->v2 migration: " << n << " rows, " << pages
              << " pages\n";
    AlignedBuf page(page_size_);
    const size_t progress_step = std::max<size_t>(1, pages / 10);
    for (size_t pi = 0; pi < pages; ++pi) {
      read_at(kSectorLen + pi * page_size_, page.data(), page_size_);
      for (size_t slot = 0; slot < npp_; ++slot) {
        const size_t id = pi * npp_ + slot;
        QGRowTrailer trailer{};
        if (id < n) {
          char *row = page.data() + slot * node_len_;
          trailer.valid_degree = static_cast<uint16_t>(compact_v1_row(row));
        }
        qg_write_page_trailer(page.data(), page_size_, npp_, slot, trailer);
      }
      write_at(kSectorLen + pi * page_size_, page.data(), page_size_);
      if ((pi + 1) % progress_step == 0 || pi + 1 == pages) {
        std::clog << "[QGUpdater] migration " << (pi + 1) << "/" << pages << " pages\n";
      }
    }

    QGSuperblockV2 sb;
    sb.magic = kQGSuperblockMagic;
    sb.format_version = kQGFormatVersion;
    sb.generation = 1;
    sb.num_points = n;
    sb.live_count = n;
    sb.free_list_head = kPidMax;
    sb.free_count = 0;
    sb.entry_point = metas[2];
    sb.dimension = dim_;
    sb.node_len = node_len_;
    sb.node_per_page = npp_;
    sb.page_size = page_size_;
    sb.file_size = kSectorLen + pages * page_size_;
    sb.checksum = qg_superblock_checksum(sb);
    write_superblock(0, sb);

    superblock_ = sb;
    active_superblock_slot_ = 0;
    committed_.store(n, std::memory_order_release);
    allocated_points_.store(n, std::memory_order_release);
    next_append_id_.store(static_cast<PID>(n), std::memory_order_release);
    live_count_.store(n, std::memory_order_release);
    free_list_head_.store(kPidMax, std::memory_order_release);
    free_count_.store(0, std::memory_order_release);
    reset_hidden();
    std::clog << "[QGUpdater] migration complete: generation=1\n";
  }

  void rebuild_free_chain(const std::vector<PID> &free_ids) {
    PID head = kPidMax;
    for (PID id : free_ids) {
      const std::lock_guard<std::mutex> guard(page_lock(id));
      const PID next = head;
      modify_node_page(id, [&](char *page) {
        char *row = page + node_offset_in_page(id);
        const uint64_t next64 = next;
        std::memcpy(row, &next64, sizeof(next64));
        QGRowTrailer trailer = row_trailer(page, id);
        trailer.valid_degree = 0;
        trailer.flags |= kQGRowTombstone | kQGRowFree;
        set_row_trailer(page, id, trailer);
        return true;
      });
      head = id;
    }
    free_list_head_.store(head, std::memory_order_release);
    free_count_.store(free_ids.size(), std::memory_order_release);
  }

  void load_v2_state(const QGSuperblockV2 &sb) {
    const size_t n = static_cast<size_t>(sb.num_points);
    if (n > kPidMax || sb.node_len != node_len_ || sb.node_per_page != npp_ ||
        sb.page_size != page_size_ || sb.dimension != dim_) {
      throw std::runtime_error("QGUpdater: invalid v2 geometry/count");
    }
    committed_.store(n, std::memory_order_release);
    allocated_points_.store(n, std::memory_order_release);
    next_append_id_.store(static_cast<PID>(n), std::memory_order_release);
    qg_.num_points_ = n;

    {
      const std::lock_guard<std::mutex> guard(deleted_mutex_);
      deleted_.clear();
    }
    reset_hidden();
    std::vector<PID> free_ids;
    const size_t pages = (n + npp_ - 1) / npp_;
    AlignedBuf page(page_size_);
    for (size_t pi = 0; pi < pages; ++pi) {
      read_at(kSectorLen + pi * page_size_, page.data(), page_size_);
      for (size_t slot = 0; slot < npp_; ++slot) {
        const size_t raw_id = pi * npp_ + slot;
        if (raw_id >= n) break;
        const PID id = static_cast<PID>(raw_id);
        const QGRowTrailer trailer = qg_read_page_trailer(page.data(), page_size_, npp_, slot);
        if (trailer.valid_degree > deg_) {
          throw std::runtime_error("QGUpdater: v2 trailer valid_degree exceeds degree bound");
        }
        if ((trailer.flags & kQGRowFree) != 0) free_ids.push_back(id);
        if ((trailer.flags & (kQGRowTombstone | kQGRowFree)) != 0) {
          mark_hidden(id);
          mirror_deleted_insert(id);
        }
      }
    }
    live_count_.store(n - deleted_snapshot().size(), std::memory_order_release);

    bool chain_ok = sb.free_count == free_ids.size();
    std::vector<uint8_t> seen(n, 0);
    PID cur = sb.free_list_head;
    size_t chain_len = 0;
    while (chain_ok && cur != kPidMax) {
      if (cur >= n || seen[cur] != 0 || chain_len >= free_ids.size()) {
        chain_ok = false;
        break;
      }
      seen[cur] = 1;
      read_at(page_offset(cur), page.data(), page_size_);
      const QGRowTrailer trailer = row_trailer(page.data(), cur);
      if ((trailer.flags & kQGRowFree) == 0) {
        chain_ok = false;
        break;
      }
      uint64_t next64 = kPidMax;
      std::memcpy(&next64, page.data() + node_offset_in_page(cur), sizeof(next64));
      if (next64 != kPidMax && next64 >= n) {
        chain_ok = false;
        break;
      }
      cur = static_cast<PID>(next64);
      ++chain_len;
    }
    chain_ok = chain_ok && chain_len == free_ids.size() && cur == kPidMax;
    if (chain_ok) {
      for (PID id : free_ids) chain_ok = chain_ok && seen[id] != 0;
    }
    if (chain_ok) {
      free_list_head_.store(sb.free_list_head, std::memory_order_release);
      free_count_.store(sb.free_count, std::memory_order_release);
    } else {
      std::clog << "[QGUpdater] free-list chain invalid; rebuilding from trailer flags\n";
      rebuild_free_chain(free_ids);
      flush_dirty(1);
    }
    if (sb.live_count != live_count_.load(std::memory_order_acquire)) {
      std::clog << "[QGUpdater] superblock live_count mismatch; trailer scan is authoritative\n";
    }
  }

  void load_or_migrate_format() {
    std::array<char, kSectorLen> header{};
    read_at(0, header.data(), header.size());
    QGSuperblockV2 sb;
    const int slot = select_qg_superblock(header.data(), sb);
    if (slot >= 0) {
      superblock_ = sb;
      active_superblock_slot_ = slot;
      load_v2_state(sb);
      repair_routing_roots(kPidMax);
      return;
    }
    if (qg_header_has_v2_magic(header.data())) {
      throw std::runtime_error("QGUpdater: both v2 superblocks have invalid checksums");
    }
    migrate_v1(header.data());
  }

  void repair_routing_roots(PID newly_deleted) {
    // A reused routing seed is especially dangerous: while dark it cannot be
    // selected as a live insertion neighbor, so its freshly assembled row can
    // be isolated.  Remove deleted medoid seeds and relocate the global entry
    // point before the row becomes free/reusable.
    bool changed = false;
    for (size_t i = qg_.medoids_.size(); i > 0; --i) {
      const size_t idx = i - 1;
      const PID medoid = qg_.medoids_[idx];
      if (medoid != newly_deleted && !is_hidden(medoid)) continue;
      qg_.medoids_.erase(qg_.medoids_.begin() + static_cast<int64_t>(idx));
      const auto first = qg_.medoids_vector_.begin() +
                         static_cast<int64_t>(idx * full_dim_);
      qg_.medoids_vector_.erase(first, first + static_cast<int64_t>(full_dim_));
      changed = true;
    }
    if (qg_.entry_point_ == newly_deleted || is_hidden(qg_.entry_point_)) {
      const size_t n = committed_.load(std::memory_order_acquire);
      if (n != 0) {
        const size_t start = (static_cast<size_t>(qg_.entry_point_) + 1) % n;
        for (size_t offset = 0; offset < n; ++offset) {
          const PID candidate = static_cast<PID>((start + offset) % n);
          if (!is_hidden(candidate)) {
            qg_.entry_point_ = candidate;
            changed = true;
            break;
          }
        }
      }
    }
    if (changed) refresh_routing_snapshot();
  }

  void push_free_slot(PID id) {
    bool pushed = false;
    const std::lock_guard<std::mutex> guard(page_lock(id));
    modify_node_page(id, [&](char *page) {
      QGRowTrailer trailer = row_trailer(page, id);
      if ((trailer.flags & kQGRowTombstone) == 0 || (trailer.flags & kQGRowFree) != 0) {
        return false;
      }
      char *row = page + node_offset_in_page(id);
      PID expected = free_list_head_.load(std::memory_order_acquire);
      for (;;) {
        const uint64_t next64 = expected;
        std::memcpy(row, &next64, sizeof(next64));
        if (free_list_head_.compare_exchange_weak(
                expected, id, std::memory_order_acq_rel, std::memory_order_acquire)) {
          break;
        }
      }
      trailer.valid_degree = 0;
      trailer.flags |= kQGRowFree;
      set_row_trailer(page, id, trailer);
      pushed = true;
      return true;
    });
    if (pushed) {
      free_count_.fetch_add(1, std::memory_order_acq_rel);
      stats_.freed_slots++;
    }
  }

  PID pop_free_slot() {
    for (;;) {
      PID head = free_list_head_.load(std::memory_order_acquire);
      if (head == kPidMax) return kPidMax;
      const std::lock_guard<std::mutex> guard(page_lock(head));
      if (head != free_list_head_.load(std::memory_order_acquire)) continue;
      AlignedBuf page(page_size_);
      read_rmw_page(head, page.data());
      const QGRowTrailer trailer = row_trailer(page.data(), head);
      if ((trailer.flags & (kQGRowTombstone | kQGRowFree)) !=
          (kQGRowTombstone | kQGRowFree)) {
        throw std::runtime_error("QGUpdater: free-list head lacks free/tombstone flags");
      }
      uint64_t next64 = kPidMax;
      std::memcpy(&next64, page.data() + node_offset_in_page(head), sizeof(next64));
      if (next64 != kPidMax && next64 >= allocated_points()) {
        throw std::runtime_error("QGUpdater: corrupt next_free PID");
      }
      PID expected = head;
      const PID next = static_cast<PID>(next64);
      if (!free_list_head_.compare_exchange_strong(
              expected, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
        continue;
      }
      free_count_.fetch_sub(1, std::memory_order_acq_rel);
      return head;
    }
  }

  /**
   * @brief Sharded resident dirty-page cache used by update searches and RMWs.
   */
  class PageWriteCache {
   public:
    static constexpr size_t kShards = 256;
    struct CachedPage {
      explicit CachedPage(size_t len) : bytes(len) {}
      AlignedBuf bytes;
      // Update-side dependency reads may inspect this page while another
      // worker owns its striped writer lock. A separate, try-only byte latch
      // makes those copies C++/TSan race-free without introducing cross-page
      // lock cycles inside consolidate/full-prune callbacks.
      std::mutex bytes_mutex;
      bool dirty = false;
    };
    struct Shard {
      std::mutex mutex;
      std::unordered_map<size_t, std::unique_ptr<CachedPage>> pages;
    };

    explicit PageWriteCache(size_t page_size) : page_size_(page_size) {}
    [[nodiscard]] Shard &shard(size_t page_index) { return shards_[page_index % kShards]; }
    [[nodiscard]] size_t page_size() const { return page_size_; }
    [[nodiscard]] size_t total_pages() const { return total_pages_.load(std::memory_order_relaxed); }
    void note_insert() { total_pages_.fetch_add(1, std::memory_order_relaxed); }
    void note_erase(size_t n) { total_pages_.fetch_sub(n, std::memory_order_relaxed); }

   private:
    size_t page_size_;
    std::array<Shard, kShards> shards_;
    std::atomic<size_t> total_pages_{0};
  };

  struct StagedEdge {
    PID v;
    PID x;
    float dist_vx;
    bool primary;
  };
  struct StagedStripe {
    std::mutex mutex;
    std::unordered_map<PID, std::vector<StagedEdge>> by_target;
  };

  /// Thread-local aligned bounce buffer for unaligned caller memory (public
  /// read_node_page/assemble_row users, tests) under O_DIRECT.
  [[nodiscard]] static char *tls_bounce(size_t len) {
    thread_local std::unique_ptr<char, decltype(&::free)> buf(nullptr, &::free);
    thread_local size_t cap = 0;
    if (cap < len) {
      char *p = nullptr;
      if (::posix_memalign(reinterpret_cast<void **>(&p), kDioAlign, len) != 0) {
        throw std::bad_alloc();
      }
      buf.reset(p);
      cap = len;
    }
    return buf.get();
  }

  [[nodiscard]] uint64_t page_offset(PID id) const {
    return kSectorLen + page_size_ * (static_cast<uint64_t>(id) / npp_);
  }
  [[nodiscard]] size_t node_offset_in_page(PID id) const { return (id % npp_) * node_len_; }
  [[nodiscard]] std::mutex &page_lock(PID id) {
    return page_locks_[(static_cast<size_t>(id) / npp_) % kLockStripes];
  }

  [[nodiscard]] size_t page_index(PID id) const { return static_cast<size_t>(id) / npp_; }

  template <typename Fn>
  bool modify_node_page(PID id, Fn &&fn) {
    const size_t pi = page_index(id);
    if (!params_.write_cache) {
      AlignedBuf page(page_size_);
      read_at(page_offset(id), page.data(), page_size_);
      stats_.patch_page_reads++;
      if (!fn(page.data())) {
        return false;
      }
      stats_.logical_row_writes++;
      write_node_page(id, page.data());
      return true;
    }
    auto &shard = write_cache_.shard(pi);
    PageWriteCache::CachedPage *cached = nullptr;
    {
      const std::lock_guard<std::mutex> cache_guard(shard.mutex);
      auto [it, inserted] = shard.pages.try_emplace(pi);
      if (inserted) {
        it->second = std::make_unique<PageWriteCache::CachedPage>(page_size_);
        read_at(kSectorLen + pi * page_size_, it->second->bytes.data(), page_size_);
        stats_.patch_page_reads++;
        write_cache_.note_insert();
      }
      cached = it->second.get();
    }
    // page_lock(id) serializes this buffer's RMW. Do not retain the shard lock
    // across fn: consolidation/full-prune may consult another cached page,
    // and holding two shard locks would introduce cross-page lock cycles.
    // The seqlock covers the in-cache mutation: overlay readers (searches read
    // through the pool) validate the page version exactly like disk readers.
    if (pi >= page_versions_.size()) {
      throw std::runtime_error("QGUpdater: id exceeds max_points capacity");
    }
    bool changed = false;
    {
      const std::lock_guard<std::mutex> bytes_guard(cached->bytes_mutex);
      page_versions_[pi].fetch_add(1, std::memory_order_acq_rel);
      changed = fn(cached->bytes.data());
      page_versions_[pi].fetch_add(1, std::memory_order_release);
    }
    if (!changed) {
      return false;
    }
    {
      const std::lock_guard<std::mutex> cache_guard(shard.mutex);
      cached->dirty = true;
    }
    stats_.logical_row_writes++;
    return true;
  }

  [[nodiscard]] const float *cached_raw(PID id) {
    const size_t pi = page_index(id);
    if (params_.write_cache) {
      auto &shard = write_cache_.shard(pi);
      const std::lock_guard<std::mutex> guard(shard.mutex);
      auto it = shard.pages.find(pi);
      if (it != shard.pages.end()) {
        return reinterpret_cast<const float *>(it->second->bytes.data() + node_offset_in_page(id));
      }
    }
    return nullptr;
  }

  /** @brief Page read for update-side dependencies; unlike search, sees dirty RMW state. */
  void read_rmw_page(PID id, char *out) {
    const size_t pi = page_index(id);
    if (params_.write_cache) {
      auto &shard = write_cache_.shard(pi);
      const std::lock_guard<std::mutex> guard(shard.mutex);
      auto it = shard.pages.find(pi);
      if (it != shard.pages.end()) {
        std::unique_lock<std::mutex> bytes_guard(it->second->bytes_mutex, std::try_to_lock);
        if (bytes_guard.owns_lock()) {
          std::memcpy(out, it->second->bytes.data(), page_size_);
          return;
        }
      }
    }
    // A busy cached page means the caller may already hold another page lock;
    // waiting here could form A->B/B->A cycles. The on-disk page is an older
    // but consistent dependency snapshot and is preferable to a torn copy.
    read_at(page_offset(id), out, page_size_);
  }

  /** @brief Versioned page write; caller must hold the page lock. */
  void write_node_page(PID id, const char *page) {
    const size_t pi = page_index(id);
    if (pi >= page_versions_.size()) {
      throw std::runtime_error("QGUpdater: id exceeds max_points capacity");
    }
    page_versions_[pi].fetch_add(1, std::memory_order_acq_rel);  // -> odd
    write_at(page_offset(id), page, page_size_);
    page_versions_[pi].fetch_add(1, std::memory_order_release);  // -> even
  }

  /**
   * @brief Torn-read-safe page read (seqlock retry), reading THROUGH the
   * resident pool: a cached page is fresher than disk (writeback may not have
   * happened yet), so overlay hits copy from the pool and only misses touch
   * the file. Safe for the insert beam search because its snapshot guards
   * (`cur/nb >= snapshot`) already ignore unpublished ids regardless of how
   * fresh the row bytes are. External readers (a separate process on its own
   * fd) must still wait for finalize().
   */
  void read_node_page(PID id, char *buf, bool query_read = false) {
    const size_t pi = page_index(id);
    if (pi >= page_versions_.size()) {
      throw std::runtime_error("QGUpdater: id exceeds max_points capacity");
    }
    stats_.seqlock_read_calls.fetch_add(1, std::memory_order_relaxed);
    if (query_read) {
      stats_.query_seqlock_read_calls.fetch_add(1, std::memory_order_relaxed);
    }
    const auto note_retry = [&] {
      stats_.seqlock_read_retries.fetch_add(1, std::memory_order_relaxed);
      if (query_read) {
        stats_.query_seqlock_read_retries.fetch_add(1, std::memory_order_relaxed);
      }
    };
    for (;;) {
      const uint32_t v1 = page_versions_[pi].load(std::memory_order_acquire);
      if ((v1 & 1U) != 0) {
        note_retry();
        std::this_thread::yield();
        continue;
      }
      bool from_pool = false;
      {
        // A classic seqlock permits a C-level racy copy, but that is undefined
        // behavior in C++ and is reported by TSan. Writers already hold this
        // striped page lock; taking it only for the resident-page memcpy keeps
        // bytes race-free without introducing a global query lock. The outer
        // version checks still detect a writer/writeback that won the race.
        const std::lock_guard<std::mutex> page_guard(page_lock(id));
        const uint32_t locked_version =
            page_versions_[pi].load(std::memory_order_acquire);
        if (locked_version != v1 || (locked_version & 1U) != 0) {
          note_retry();
          continue;
        }
        if (params_.write_cache) {
          auto &shard = write_cache_.shard(pi);
          const std::lock_guard<std::mutex> guard(shard.mutex);
          auto it = shard.pages.find(pi);
          if (it != shard.pages.end()) {
            std::memcpy(buf, it->second->bytes.data(), page_size_);
            from_pool = true;
          }
        }
      }
      if (!from_pool) {
        read_at(page_offset(id), buf, page_size_);
      }
      const uint32_t v2 = page_versions_[pi].load(std::memory_order_acquire);
      if (v1 == v2) {
        return;
      }
      note_retry();
    }
  }

  void read_at(uint64_t off, char *buf, size_t len) const {
    size_t done = 0;
    while (done < len) {
      const ssize_t r = ::pread(fd_, buf + done, len - done, static_cast<off_t>(off + done));
      if (r < 0) {
        throw std::runtime_error("QGUpdater: pread failed errno=" + std::to_string(errno));
      }
      if (r == 0) {  // reading a page that does not exist yet -> zero-fill
        std::memset(buf + done, 0, len - done);
        return;
      }
      done += static_cast<size_t>(r);
    }
  }

  void write_at(uint64_t off, const char *buf, size_t len) {
    const int fd = direct_io_ ? wfd_ : fd_;
    if (direct_io_ && (reinterpret_cast<uintptr_t>(buf) & (kDioAlign - 1)) != 0) {
      char *b = tls_bounce(len);
      std::memcpy(b, buf, len);
      buf = b;
    }
    size_t done = 0;
    while (done < len) {
      const ssize_t w = ::pwrite(fd, buf + done, len - done, static_cast<off_t>(off + done));
      if (w < 0) {
        throw std::runtime_error("QGUpdater: pwrite failed errno=" + std::to_string(errno));
      }
      done += static_cast<size_t>(w);
    }
    stats_.page_writes++;
  }

  void stage_edge(StagedEdge edge) {
    auto &stripe = staged_[static_cast<size_t>(edge.v) % staged_.size()];
    const std::lock_guard<std::mutex> guard(stripe.mutex);
    stripe.by_target[edge.v].push_back(edge);
  }

  /**
   * @brief Apply staged kEvict/kAlphaEvict backlinks once per target row.
   *
   * AlphaEvict intentionally degrades to Evict here: the insert-local
   * capture pool is gone at the batch barrier, and E1 found their quality
   * effectively overlapping. Entries are ordered by exact d(v,x).
   */
  void drain_staged_edges(size_t num_threads) {
    struct Group {
      PID v;
      std::vector<StagedEdge> edges;
    };
    std::vector<Group> groups;
    PID x_min = std::numeric_limits<PID>::max();
    PID x_max = 0;
    for (auto &stripe : staged_) {
      const std::lock_guard<std::mutex> guard(stripe.mutex);
      for (auto &[v, edges] : stripe.by_target) {
        for (const auto &edge : edges) {
          x_min = std::min(x_min, edge.x);
          x_max = std::max(x_max, edge.x);
        }
        groups.push_back({v, std::move(edges)});
      }
      stripe.by_target.clear();
    }
    if (groups.empty()) {
      return;
    }

    // Staged x ids are one dense batch range, so per-x bookkeeping is flat
    // arrays with atomic counters — a shared map mutex here serializes the
    // whole drain (~35 successes per insert).
    static constexpr PID kNoPrimary = std::numeric_limits<PID>::max();
    const size_t x_range = static_cast<size_t>(x_max - x_min) + 1;
    std::vector<std::atomic<uint32_t>> successes(x_range);
    std::vector<PID> primary(x_range, kNoPrimary);
    for (const auto &group : groups) {
      for (const auto &edge : group.edges) {
        if (edge.primary) {
          primary[edge.x - x_min] = group.v;
        }
      }
    }

    const int nt = static_cast<int>(std::max<size_t>(1, num_threads));
#pragma omp parallel for num_threads(nt) schedule(dynamic, 64)
    for (int64_t gi = 0; gi < static_cast<int64_t>(groups.size()); ++gi) {
      auto &group = groups[static_cast<size_t>(gi)];
      auto &edges = group.edges;
      std::sort(edges.begin(), edges.end(), [](const StagedEdge &a, const StagedEdge &b) {
        if (a.dist_vx != b.dist_vx) return a.dist_vx < b.dist_vx;
        return a.x < b.x;
      });
      CapturedNode v;
      v.id = group.v;
      const std::unordered_map<PID, const CapturedNode *> no_capture;
      for (const auto &edge : edges) {
        std::vector<float> x_vec;
        // x was appended earlier in this batch. This hits its dirty cache
        // page (or disk in write_cache=false control mode); no vector is
        // retained in StagedEdge.
        const float *x_raw = cached_raw(edge.x);
        if (x_raw == nullptr) {
          AlignedBuf x_page(page_size_);
          read_at(page_offset(edge.x), x_page.data(), page_size_);
          stats_.patch_page_reads++;
          const auto *raw =
              reinterpret_cast<const float *>(x_page.data() + node_offset_in_page(edge.x));
          x_vec.assign(raw, raw + full_dim_);
          x_raw = x_vec.data();
        }
        if (patch_reverse_edge(v, edge.x, x_raw, no_capture, false)) {
          successes[edge.x - x_min].fetch_add(1, std::memory_order_relaxed);
        }
      }
    }

    // Batch reachability fallback: force the primary backlink only for nodes
    // for which every ordinary staged patch was rejected.
    for (size_t xi = 0; xi < x_range; ++xi) {
      if (successes[xi].load(std::memory_order_relaxed) != 0 || primary[xi] == kNoPrimary) {
        continue;
      }
      const PID x = x_min + static_cast<PID>(xi);
      CapturedNode v;
      v.id = primary[xi];
      std::vector<float> x_vec;
      const float *x_raw = cached_raw(x);
      if (x_raw == nullptr) {
        AlignedBuf x_page(page_size_);
        read_at(page_offset(x), x_page.data(), page_size_);
        stats_.patch_page_reads++;
        const auto *raw = reinterpret_cast<const float *>(x_page.data() + node_offset_in_page(x));
        x_vec.assign(raw, raw + full_dim_);
        x_raw = x_vec.data();
      }
      const std::unordered_map<PID, const CapturedNode *> no_capture;
      if (patch_reverse_edge(v, x, x_raw, no_capture, true)) {
        stats_.forced_links++;
      }
    }
  }

  void flush_dirty(size_t num_threads) {
    if (!params_.write_cache) return;
    struct DirtyPage {
      size_t index;
      char *data;
    };
    std::vector<DirtyPage> dirty;
    for (size_t si = 0; si < PageWriteCache::kShards; ++si) {
      auto &shard = write_cache_.shard(si);
      const std::lock_guard<std::mutex> guard(shard.mutex);
      for (auto &[pi, page] : shard.pages) {
        if (page->dirty) dirty.push_back({pi, page->bytes.data()});
      }
    }
    stats_.flush_unique_pages.fetch_add(dirty.size());
#if defined(__SANITIZE_THREAD__)
    // GCC TSan does not model libgomp's publication of the stack-owned dirty
    // vector into this second, nested parallel region and reports the OpenMP
    // frame itself as a race. Maintenance row workers remain concurrent in
    // sanitizer tests; serialize only the disjoint-page writeback loop.
    (void)num_threads;
    const int nt = 1;
#else
    const int nt = static_cast<int>(std::max<size_t>(1, num_threads));
#endif
#pragma omp parallel for num_threads(nt) schedule(dynamic, 1)
    for (int64_t i = 0; i < static_cast<int64_t>(dirty.size()); ++i) {
      const auto &page = dirty[static_cast<size_t>(i)];
      if (page.index >= page_versions_.size()) {
        throw std::runtime_error("QGUpdater: dirty page exceeds max_points capacity");
      }
      page_versions_[page.index].fetch_add(1, std::memory_order_acq_rel);
      write_at(kSectorLen + page.index * page_size_, page.data, page_size_);
      page_versions_[page.index].fetch_add(1, std::memory_order_release);
    }
    // Pages stay resident (the pool's cross-batch coalescing is the point);
    // only the dirty flags drop. No mutator runs concurrently with a flush —
    // phase separation is the caller's contract, same as consolidate().
    for (size_t si = 0; si < PageWriteCache::kShards; ++si) {
      auto &shard = write_cache_.shard(si);
      const std::lock_guard<std::mutex> guard(shard.mutex);
      for (auto &[pi, page] : shard.pages) {
        page->dirty = false;
      }
    }
    // Kick asynchronous writeback now (WB_SYNC_NONE) so the dirty backlog
    // drains overlapped with the next batch instead of piling up until
    // balance_dirty_pages throttles some future pwrite. Durability is still
    // finalize()+fsync only.
    if (!direct_io_ && !dirty.empty()) {
      ::sync_file_range(fd_, 0, 0, SYNC_FILE_RANGE_WRITE);
    }
  }

  /** @brief Drop clean pages (arbitrary order) until the pool holds at most
   * @p target pages. Runs at the batch barrier only — cached_raw() pointers
   * from the drain phase are dead by then. */
  void evict_clean(size_t target) {
    for (size_t si = 0; si < PageWriteCache::kShards && write_cache_.total_pages() > target;
         ++si) {
      auto &shard = write_cache_.shard(si);
      const std::lock_guard<std::mutex> guard(shard.mutex);
      size_t erased = 0;
      for (auto it = shard.pages.begin();
           it != shard.pages.end() && write_cache_.total_pages() - erased > target;) {
        if (!it->second->dirty) {
          it = shard.pages.erase(it);
          ++erased;
        } else {
          ++it;
        }
      }
      write_cache_.note_erase(erased);
    }
  }

  /** @brief Greedy beam search over the on-disk graph, capturing expanded rows. */
  void search_for_insert(const float *tvec,
                         std::vector<CapturedNode> &pool,
                         size_t ef_override = 0,
                         size_t cap_override = 0) {
    const size_t snapshot = committed_.load(std::memory_order_acquire);
    QGQuery q_obj(tvec, pd_);
    q_obj.query_prepare(qg_.rotator_, qg_.scanner_);
    const float *res_query = tvec + dim_;
    float sqr_qr = 0;
    for (size_t i = 0; i < res_dim_; ++i) {
      sqr_qr += res_query[i] * res_query[i];
    }
    q_obj.set_sqr_qr(sqr_qr);

    const size_t ef = ef_override == 0 ? params_.ef_insert : ef_override;
    buffer::SearchBuffer sp(ef);
    std::unordered_set<PID> visited;
    visited.reserve(ef * 8);

    const RoutingSnapshot *routing = routing_snapshot_.load(std::memory_order_acquire);
    if (routing != nullptr && !routing->medoids.empty() &&
        routing->medoid_vectors.size() == routing->medoids.size() * full_dim_) {
      PID best_medoid = kPidMax;
      float best = FLT_MAX;
      for (size_t m = 0; m < routing->medoids.size(); ++m) {
        if (routing->medoids[m] >= snapshot) continue;
        const float d =
            space::l2_sqr(tvec, routing->medoid_vectors.data() + full_dim_ * m, dim_);
        if (d < best) {
          best = d;
          best_medoid = routing->medoids[m];
        }
      }
      if (best_medoid != kPidMax) sp.insert(best_medoid, FLT_MAX);
    }
    if (routing != nullptr && routing->entry_point < snapshot) {
      sp.insert(routing->entry_point, FLT_MAX);
    }

    std::vector<float> appro(deg_);
    AlignedBuf page(page_size_);
    while (sp.has_next()) {
      const PID cur = sp.pop();
      if (visited.count(cur) != 0) {
        continue;
      }
      visited.insert(cur);
      if (cur >= snapshot) {
        continue;
      }
      read_node_page(cur, page.data());
      stats_.search_page_reads++;
      const char *row = page.data() + node_offset_in_page(cur);
      const auto *row_f = reinterpret_cast<const float *>(row);

      float sqr_y = space::l2_sqr(tvec, row_f, dim_);
      // FastScan over this node's neighbors
      qg_.scanner_.scan_neighbors(appro.data(),
                                  q_obj.lut().data(),
                                  sqr_y,
                                  q_obj.lower_val(),
                                  q_obj.width(),
                                  q_obj.sqr_qr(),
                                  q_obj.sumq(),
                                  reinterpret_cast<const uint8_t *>(row + code_off_bytes()),
                                  reinterpret_cast<const float *>(row + factor_off_bytes()));
      const auto *nbs = reinterpret_cast<const PID *>(row + neighbor_off_bytes());
      const size_t degree = row_trailer(page.data(), cur).valid_degree;
      for (size_t j = 0; j < degree; ++j) {
        const PID nb = nbs[j];
        if (nb >= snapshot || visited.count(nb) != 0 || sp.is_full(appro[j])) {
          continue;
        }
        sp.insert(nb, appro[j]);
      }

      if (res_dim_ > 0) {
        sqr_y += space::l2_sqr(row_f + dim_, res_query, res_dim_);
      }
      CapturedNode cap;
      cap.id = cur;
      cap.dist = sqr_y;
      cap.vec.assign(row_f, row_f + full_dim_);
      pool.push_back(std::move(cap));
    }

    std::sort(pool.begin(), pool.end(), [](const CapturedNode &a, const CapturedNode &b) {
      return a.dist < b.dist;
    });
    const size_t cap = cap_override == 0 ? params_.prune_pool_cap : cap_override;
    if (pool.size() > cap) {
      pool.resize(cap);
    }
  }

  /** @brief DiskANN-style alpha occlusion prune over the captured pool. */
  void robust_prune(const std::vector<CapturedNode> &pool, std::vector<size_t> &sel) {
    sel.clear();
    for (size_t i = 0; i < pool.size() && sel.size() < deg_; ++i) {
      if (is_hidden(pool[i].id)) {
        continue;
      }
      bool occluded = false;
      for (size_t s : sel) {
        const float d_sc =
            space::l2_sqr(pool[s].raw(), pool[i].raw(), dim_) +
            (res_dim_ > 0 ? space::l2_sqr(pool[s].raw() + dim_, pool[i].raw() + dim_, res_dim_)
                          : 0.0F);
        // squared-distance domain: alpha^2 * d2(s,c) <= d2(x,c) prunes c
        if (params_.alpha * params_.alpha * d_sc <= pool[i].dist) {
          occluded = true;
          break;
        }
      }
      if (!occluded) {
        sel.push_back(i);
      }
    }
  }

  void append_node(PID id, const char *row, size_t degree, bool reused) {
    const std::lock_guard<std::mutex> guard(page_lock(id));
    modify_node_page(id, [&](char *page) {
      // A newly-created one-row page has deterministic zero tail padding;
      // read_at already zero-fills an EOF cache miss.
      QGRowTrailer trailer = row_trailer(page, id);
      if (reused &&
          (trailer.flags & (kQGRowTombstone | kQGRowFree)) !=
              (kQGRowTombstone | kQGRowFree)) {
        throw std::runtime_error("QGUpdater: reused row lost tombstone/free state");
      }
      std::memcpy(page + node_offset_in_page(id), row, node_len_);
      trailer.valid_degree = static_cast<uint16_t>(degree);
      trailer.flags = reused ? static_cast<uint16_t>(trailer.flags |
                                                      kQGRowTombstone | kQGRowFree)
                             : 0;
      set_row_trailer(page, id, trailer);
      add_row_indegree(row, degree, 1);
      return true;
    });
  }

  /**
   * @brief Cheap reverse-edge patch: fill a ghost slot, else evict the
   * farthest current edge (FastScan estimate with v as its own query) when the
   * new edge is shorter. kAlphaEvict additionally rejects the new edge when an
   * already-captured current neighbor alpha-occludes it (zero extra I/O).
   */
  bool patch_reverse_edge(const CapturedNode &v,
                          PID x_id,
                          const float *x_vec,
                          const std::unordered_map<PID, const CapturedNode *> &captured,
                          bool force) {
    if (is_hidden(v.id)) {
      return false;
    }
    const std::lock_guard<std::mutex> guard(page_lock(v.id));
    return modify_node_page(v.id, [&](char *page) {
      char *row = page + node_offset_in_page(v.id);
      const auto *row_f = reinterpret_cast<const float *>(row);
      auto *ids = reinterpret_cast<PID *>(row + neighbor_off_bytes());
      auto *fac = reinterpret_cast<float *>(row + factor_off_bytes());
      QGRowTrailer trailer = row_trailer(page, v.id);
      size_t degree = trailer.valid_degree;
      for (size_t j = 0; j < degree; ++j) {
        if (ids[j] == x_id) return true;
      }

      // residual-consistent comparison value for the new edge (v,x)
      float x_res_sqr = 0;
      float v_res_sqr = 0;
      for (size_t j = 0; j < res_dim_; ++j) {
        x_res_sqr += x_vec[dim_ + j] * x_vec[dim_ + j];
        v_res_sqr += row_f[dim_ + j] * row_f[dim_ + j];
      }
      const float est_new = space::l2_sqr(row_f, x_vec, dim_) + x_res_sqr + v_res_sqr;

      // alpha-occlusion test using neighbors already captured by this search
      if (!force && params_.backlink_mode == UpdateParams::Backlink::kAlphaEvict &&
          !captured.empty()) {
        size_t checked = 0;
        const float d_vx =
            space::l2_sqr(row_f, x_vec, dim_) +
            (res_dim_ > 0 ? space::l2_sqr(row_f + dim_, x_vec + dim_, res_dim_) : 0.0F);
        for (size_t j = 0; j < degree && checked < params_.alpha_check_max; ++j) {
          if (ids[j] == x_id) {
            continue;
          }
          auto it = captured.find(ids[j]);
          if (it == captured.end()) {
            continue;
          }
          ++checked;
          const float d_sx =
              space::l2_sqr(it->second->raw(), x_vec, dim_) +
              (res_dim_ > 0 ? space::l2_sqr(it->second->raw() + dim_, x_vec + dim_, res_dim_)
                            : 0.0F);
          if (params_.alpha * params_.alpha * d_sx <= d_vx) {
            stats_.alpha_skips++;
            return false;
          }
        }
      }

      // valid_degree owns a packed prefix: append at its tail, or evict when full.
      size_t slot = degree;
      bool evicted = false;
      bool tel_pending = false;
      bool tel_agree = false;
      size_t tel_rank = 0;
      double tel_relerr = 0;
      if (degree == deg_) {
        // Reused per-thread query object: constructing a QGQuery here means an
        // aligned lut_ allocation per evict decision (millions per drain).
        thread_local std::unique_ptr<QGQuery> vq_holder;
        if (!vq_holder || vq_holder->padded_dim() != pd_) {
          vq_holder = std::make_unique<QGQuery>(row_f, pd_);
        } else {
          vq_holder->rebind(row_f);
        }
        QGQuery &vq = *vq_holder;
        vq.query_prepare(qg_.rotator_, qg_.scanner_);
        vq.set_sqr_qr(v_res_sqr);
        thread_local std::vector<float> appro;
        appro.resize(deg_);
        qg_.scanner_.scan_neighbors(appro.data(),
                                    vq.lut().data(),
                                    0.0F,
                                    vq.lower_val(),
                                    vq.width(),
                                    vq.sqr_qr(),
                                    vq.sumq(),
                                    reinterpret_cast<const uint8_t *>(row + code_off_bytes()),
                                    fac);
        float worst = -1;
        size_t estimated_slot = deg_;
        for (size_t j = 0; j < degree; ++j) {
          if (appro[j] > worst) {
            worst = appro[j];
            estimated_slot = j;
          }
        }
        slot = estimated_slot;

        const bool exact_mode = params_.backlink_mode == UpdateParams::Backlink::kExactEvict;
        bool sample = false;
        if (params_.backlink_mode == UpdateParams::Backlink::kEvict &&
            params_.evict_telemetry > 0) {
          thread_local std::mt19937_64 tel_rng(
              static_cast<uint64_t>(std::random_device{}()) ^
              static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id())));
          sample = params_.evict_telemetry >= 1 ||
                   std::generate_canonical<double, 53>(tel_rng) < params_.evict_telemetry;
        }
        std::vector<float> exact;
        size_t exact_slot = deg_;
        float exact_worst = -1;
        if (exact_mode || sample) {
          exact.resize(deg_);
          AlignedBuf nb_page(page_size_);
          for (size_t j = 0; j < degree; ++j) {
            read_rmw_page(ids[j], nb_page.data());
            stats_.patch_page_reads++;
            const auto *nb_f = reinterpret_cast<const float *>(
                nb_page.data() + node_offset_in_page(ids[j]));
            exact[j] = space::l2_sqr(row_f, nb_f, dim_) +
                       (res_dim_ > 0
                            ? space::l2_sqr(row_f + dim_, nb_f + dim_, res_dim_)
                            : 0.0F);
            if (exact[j] > exact_worst) {
              exact_worst = exact[j];
              exact_slot = j;
            }
          }
          if (exact_mode) {
            slot = exact_slot;
            worst = exact_worst;
          }
        }
        const float new_distance =
            exact_mode
                ? space::l2_sqr(row_f, x_vec, dim_) +
                      (res_dim_ > 0
                           ? space::l2_sqr(row_f + dim_, x_vec + dim_, res_dim_)
                           : 0.0F)
                : est_new;
        if (!force && new_distance >= worst) {
          stats_.est_skips++;
          return false;
        }
        evicted = true;

        // Telemetry is committed only for an accepted eviction decision, so
        // p=1 sample count has the same denominator as evictions.
        if (sample) {
          tel_pending = true;
          tel_agree = estimated_slot == exact_slot;
          size_t rank = 0;
          for (size_t j = 0; j < degree; ++j) {
            if (exact[j] > exact[estimated_slot]) ++rank;
          }
          tel_rank = std::min<size_t>(rank, 3);
          const double denom = static_cast<double>(exact[estimated_slot]);
          tel_relerr = denom > 0
                           ? std::abs(static_cast<double>(appro[estimated_slot]) - denom) / denom
                           : (appro[estimated_slot] == 0 ? 0.0 : 1.0);
        }
      }

      if (!patch_slot(row, slot, x_id, x_vec, x_res_sqr, evicted)) {
        stats_.degenerate_skips++;
        return false;
      }
      if (evicted) {
        stats_.evictions++;
        if (tel_pending) {
          stats_.evict_tel_samples++;
          if (tel_agree) stats_.evict_tel_agree++;
          stats_.evict_tel_regret[tel_rank]++;
          stats_.evict_tel_relerr_nano.fetch_add(
              static_cast<uint64_t>(std::min(tel_relerr, 1e9) * 1e9));
        }
      } else {
        trailer.valid_degree = static_cast<uint16_t>(degree + 1);
        set_row_trailer(page, v.id, trailer);
        stats_.free_slot_fills++;
      }
      return true;
    });
  }

  /** @brief Quality-reference arm: full RobustPrune with all neighbor vectors read back. */
  bool full_reverse_recompute(const CapturedNode &v, PID x_id, const float *x_vec) {
    if (is_hidden(v.id)) {
      return false;
    }
    const std::lock_guard<std::mutex> guard(page_lock(v.id));
    bool survived = false;
    modify_node_page(v.id, [&](char *page) {
      char *row = page + node_offset_in_page(v.id);
      const auto *row_f = reinterpret_cast<const float *>(row);
      auto *ids = reinterpret_cast<PID *>(row + neighbor_off_bytes());
      QGRowTrailer trailer = row_trailer(page, v.id);
      const size_t old_degree = trailer.valid_degree;

      const size_t snapshot = committed_.load(std::memory_order_acquire);

      // gather live neighbors + the new node as prune candidates
      struct Cand {
        PID id;
        float dist;
        std::vector<float> vec;
      };
      std::vector<Cand> cands;
      cands.reserve(deg_ + 1);
      AlignedBuf nb_page(page_size_);
      for (size_t j = 0; j < old_degree; ++j) {
        const PID nb = ids[j];
        if ((nb >= snapshot && nb != x_id) || nb == v.id || is_hidden(nb)) {
          continue;
        }
        bool dup = false;
        for (const auto &c : cands) {
          if (c.id == nb) {
            dup = true;
            break;
          }
        }
        if (dup) {
          continue;
        }
        read_rmw_page(nb, nb_page.data());
        stats_.patch_page_reads++;
        const auto *nb_f =
            reinterpret_cast<const float *>(nb_page.data() + node_offset_in_page(nb));
        Cand c;
        c.id = nb;
        c.vec.assign(nb_f, nb_f + full_dim_);
        c.dist = space::l2_sqr(row_f, c.vec.data(), dim_) +
                 (res_dim_ > 0 ? space::l2_sqr(row_f + dim_, c.vec.data() + dim_, res_dim_) : 0.0F);
        cands.push_back(std::move(c));
      }
      {
        Cand c;
        c.id = x_id;
        c.vec.assign(x_vec, x_vec + full_dim_);
        c.dist = space::l2_sqr(row_f, x_vec, dim_) +
                 (res_dim_ > 0 ? space::l2_sqr(row_f + dim_, x_vec + dim_, res_dim_) : 0.0F);
        cands.push_back(std::move(c));
      }
      std::sort(cands.begin(), cands.end(), [](const Cand &a, const Cand &b) {
        return a.dist < b.dist;
      });

      std::vector<size_t> sel;
      for (size_t i = 0; i < cands.size() && sel.size() < deg_; ++i) {
        bool occluded = false;
        for (size_t s : sel) {
          const float d_sc =
              space::l2_sqr(cands[s].vec.data(), cands[i].vec.data(), dim_) +
              (res_dim_ > 0
                   ? space::l2_sqr(cands[s].vec.data() + dim_, cands[i].vec.data() + dim_, res_dim_)
                   : 0.0F);
          if (params_.alpha * params_.alpha * d_sc <= cands[i].dist) {
            occluded = true;
            break;
          }
        }
        if (!occluded) {
          sel.push_back(i);
        }
      }

      std::vector<float> v_vec(row_f, row_f + full_dim_);  // row buffer is rewritten below
      std::vector<const float *> nb_vecs;
      std::vector<PID> nb_ids;
      bool x_survived = false;
      for (size_t s : sel) {
        nb_vecs.push_back(cands[s].vec.data());
        nb_ids.push_back(cands[s].id);
        x_survived = x_survived || (cands[s].id == x_id);
      }
      add_row_indegree(row, old_degree, -1);
      assemble_row(row, v_vec.data(), nb_vecs, nb_ids);
      trailer.valid_degree = static_cast<uint16_t>(nb_ids.size());
      set_row_trailer(page, v.id, trailer);
      add_row_indegree(row, nb_ids.size(), 1);
      survived = x_survived;
      return true;
    });
    stats_.full_recomputes++;
    return survived;
  }

  void garden_row(PID u, const GardenParams &gp, size_t r_target) {
    AlignedBuf old_page(page_size_);
    read_rmw_page(u, old_page.data());
    const char *old_row = old_page.data() + node_offset_in_page(u);
    const auto *u_raw = reinterpret_cast<const float *>(old_row);
    std::vector<float> u_vec(u_raw, u_raw + full_dim_);
    const auto *old_ids = reinterpret_cast<const PID *>(old_row + neighbor_off_bytes());
    const size_t old_degree = row_trailer(old_page.data(), u).valid_degree;
    std::unordered_set<PID> old_set;
    for (size_t j = 0; j < old_degree; ++j) {
      if (old_ids[j] != u && !is_hidden(old_ids[j])) {
        old_set.insert(old_ids[j]);
      }
    }

    std::vector<CapturedNode> pool;
    search_for_insert(u_vec.data(), pool, gp.ef_maintenance,
                      std::max(params_.prune_pool_cap, gp.ef_maintenance));
    std::unordered_set<PID> present;
    for (const auto &c : pool) present.insert(c.id);
    AlignedBuf nb_page(page_size_);
    for (PID nb : old_set) {
      if (present.count(nb) != 0) continue;
      read_rmw_page(nb, nb_page.data());
      const auto *raw = reinterpret_cast<const float *>(nb_page.data() + node_offset_in_page(nb));
      CapturedNode c;
      c.id = nb;
      c.vec.assign(raw, raw + full_dim_);
      c.dist = space::l2_sqr(u_vec.data(), raw, dim_) +
               (res_dim_ > 0 ? space::l2_sqr(u_vec.data() + dim_, raw + dim_, res_dim_) : 0.0F);
      pool.push_back(std::move(c));
    }
    pool.erase(std::remove_if(pool.begin(), pool.end(), [&](const CapturedNode &c) {
                 return c.id == u || is_hidden(c.id);
               }),
               pool.end());
    std::sort(pool.begin(), pool.end(), [](const CapturedNode &a, const CapturedNode &b) {
      return a.dist != b.dist ? a.dist < b.dist : a.id < b.id;
    });
    pool.erase(std::unique(pool.begin(), pool.end(), [](const CapturedNode &a, const CapturedNode &b) {
                 return a.id == b.id;
               }),
               pool.end());

    std::vector<size_t> sel;
    for (size_t i = 0; i < pool.size() && sel.size() < r_target; ++i) {
      bool occluded = false;
      for (size_t s : sel) {
        const float d = space::l2_sqr(pool[s].raw(), pool[i].raw(), dim_) +
                        (res_dim_ > 0 ? space::l2_sqr(pool[s].raw() + dim_,
                                                     pool[i].raw() + dim_, res_dim_)
                                      : 0.0F);
        if (params_.alpha * params_.alpha * d <= pool[i].dist) {
          occluded = true;
          break;
        }
      }
      if (!occluded) sel.push_back(i);
    }
    std::vector<const float *> nb_vecs;
    std::vector<PID> nb_ids;
    for (size_t s : sel) {
      nb_vecs.push_back(pool[s].raw());
      nb_ids.push_back(pool[s].id);
    }
    {
      const std::lock_guard<std::mutex> guard(page_lock(u));
      modify_node_page(u, [&](char *page) {
        char *row = page + node_offset_in_page(u);
        QGRowTrailer trailer = row_trailer(page, u);
        add_row_indegree(row, trailer.valid_degree, -1);
        assemble_row(row, u_vec.data(), nb_vecs, nb_ids);
        trailer.valid_degree = static_cast<uint16_t>(nb_ids.size());
        set_row_trailer(page, u, trailer);
        add_row_indegree(row, nb_ids.size(), 1);
        return true;
      });
    }
    stats_.gardened_rows++;

    size_t attempted = 0;
    const std::unordered_map<PID, const CapturedNode *> no_capture;
    for (size_t s : sel) {
      if (attempted >= gp.pump_budget) break;
      const auto &v = pool[s];
      if (old_set.count(v.id) != 0) continue;
      ++attempted;
      if (patch_reverse_edge(v, u, u_vec.data(), no_capture, false)) {
        stats_.garden_pump_links++;
      }
    }
  }

  /** @brief Purge/splice one live row; see consolidate(). */
  void consolidate_row(PID u, size_t snapshot, size_t r_target) {
    const std::lock_guard<std::mutex> guard(page_lock(u));
    modify_node_page(u, [&](char *page) {
      char *row = page + node_offset_in_page(u);
      const auto *row_f = reinterpret_cast<const float *>(row);
      auto *ids = reinterpret_cast<PID *>(row + neighbor_off_bytes());
      QGRowTrailer trailer = row_trailer(page, u);
      size_t degree = trailer.valid_degree;

      // Collect the current live adjacency.  The v2 row invariant is a packed
      // prefix [0, valid_degree), so removal below swaps the last valid slot
      // into the hole and keeps processing the same index.
      std::unordered_set<PID> chosen;
      bool has_dead = false;
      for (size_t j = 0; j < degree; ++j) {
        if (is_hidden(ids[j]))
          has_dead = true;
        else
          chosen.insert(ids[j]);
      }
      if (!has_dead) return false;
      size_t live_degree = chosen.size();

      float u_res_sqr = 0;
      for (size_t j = 0; j < res_dim_; ++j) {
        u_res_sqr += row_f[dim_ + j] * row_f[dim_ + j];
      }
      QGQuery uq(row_f, pd_);
      uq.query_prepare(qg_.rotator_, qg_.scanner_);
      uq.set_sqr_qr(u_res_sqr);

      AlignedBuf d_page(page_size_);
      AlignedBuf cand_page(page_size_);
      std::vector<float> appro(deg_);
      bool dirty = false;
      size_t j = 0;
      while (j < degree) {
        if (!is_hidden(ids[j])) {
          ++j;
          continue;
        }
        const PID d = ids[j];
        // Headroom preservation: only splice back up to r_target live edges;
        // surplus dead slots are removed from the packed prefix.
        if (live_degree >= r_target) {
          erase_slot(row, j, degree);
          --degree;
          stats_.ghosted_slots++;
          dirty = true;
          continue;
        }
        // FastScan over the dead node's own row (codes centered at d, but the
        // estimator still targets ||u - n_i||) recalls candidates cheaply...
        read_rmw_page(d, d_page.data());
        stats_.patch_page_reads++;
        const char *d_row = d_page.data() + node_offset_in_page(d);
        const size_t d_degree = row_trailer(d_page.data(), d).valid_degree;
        qg_.scanner_.scan_neighbors(appro.data(),
                                    uq.lut().data(),
                                    space::l2_sqr(row_f,
                                                  reinterpret_cast<const float *>(d_row),
                                                  dim_),
                                    uq.lower_val(),
                                    uq.width(),
                                    uq.sqr_qr(),
                                    uq.sumq(),
                                    reinterpret_cast<const uint8_t *>(d_row + code_off_bytes()),
                                    reinterpret_cast<const float *>(d_row + factor_off_bytes()));
        const auto *d_ids = reinterpret_cast<const PID *>(d_row + neighbor_off_bytes());
        std::vector<std::pair<float, PID>> recalled;
        for (size_t k = 0; k < d_degree; ++k) {
          const PID cand = d_ids[k];
          if (cand == u || cand >= snapshot || !std::isfinite(appro[k]) ||
              is_hidden(cand) || chosen.count(cand) != 0) {
            continue;
          }
          recalled.emplace_back(appro[k], cand);
        }
        std::sort(recalled.begin(), recalled.end());
        if (recalled.size() > params_.splice_rerank) {
          recalled.resize(params_.splice_rerank);
        }
        // ...then rerank the recalled few exactly on raw vectors before patching.
        bool patched = false;
        PID best = kPidMax;
        float best_dist = FLT_MAX;
        std::vector<float> best_vec;
        for (const auto &[est, cand] : recalled) {
          read_rmw_page(cand, cand_page.data());
          stats_.patch_page_reads++;
          const auto *cand_f =
              reinterpret_cast<const float *>(cand_page.data() + node_offset_in_page(cand));
          const float d_exact =
              space::l2_sqr(row_f, cand_f, dim_) +
              (res_dim_ > 0 ? space::l2_sqr(row_f + dim_, cand_f + dim_, res_dim_) : 0.0F);
          if (d_exact < best_dist) {
            best_dist = d_exact;
            best = cand;
            best_vec.assign(cand_f, cand_f + full_dim_);
          }
        }
        if (best != kPidMax) {
          float cand_res_sqr = 0;
          for (size_t r = 0; r < res_dim_; ++r) {
            cand_res_sqr += best_vec[dim_ + r] * best_vec[dim_ + r];
          }
          if (patch_slot(row, j, best, best_vec.data(), cand_res_sqr, true)) {
            chosen.insert(best);
            ++live_degree;
            stats_.spliced_slots++;
            dirty = true;
            patched = true;
            ++j;
          }
        }
        if (!patched) {
          erase_slot(row, j, degree);
          --degree;
          stats_.ghosted_slots++;
          dirty = true;
        }
      }
      if (dirty) {
        trailer.valid_degree = static_cast<uint16_t>(degree);
        set_row_trailer(page, u, trailer);
        stats_.consolidated_rows++;
      }
      return dirty;
    });
  }

  /** @brief Replace one slot's code + factors + id inside a row buffer. */
  bool patch_slot(char *row,
                  size_t slot,
                  PID x_id,
                  const float *x_vec,
                  float x_res_sqr,
                  bool had_old) {
    const PID old_id = reinterpret_cast<const PID *>(row + neighbor_off_bytes())[slot];
    const auto *row_f = reinterpret_cast<const float *>(row);
    // thread_local scratch: one patch_slot per staged backlink — per-call heap
    // traffic here was a top drain cost (see make_edge_payload note).
    thread_local std::vector<float> c_pad;
    thread_local std::vector<float> x_pad;
    thread_local std::vector<float> c_rot;
    thread_local std::vector<float> x_rot;
    c_pad.assign(pd_, 0.0F);
    x_pad.assign(pd_, 0.0F);
    c_rot.resize(pd_);
    x_rot.resize(pd_);
    std::copy(row_f, row_f + dim_, c_pad.begin());
    std::copy(x_vec, x_vec + dim_, x_pad.begin());
    qg_.rotator_.rotate(c_pad.data(), c_rot.data());
    qg_.rotator_.rotate(x_pad.data(), x_rot.data());

    const EdgePayload &payload = make_edge_payload(c_rot.data(), x_rot.data(), pd_, x_res_sqr);
    if (payload.degenerate) {
      return false;
    }

    const size_t block_idx = slot / kBatchSize;
    const size_t in_block = slot % kBatchSize;
    uint8_t *block = reinterpret_cast<uint8_t *>(row + code_off_bytes()) + block_idx * pd_ * 4;
    thread_local std::vector<uint64_t> bins;
    bins.resize(kBatchSize * pd_ / 64);
    unpack_codes_block(pd_, block, bins.data());
    std::copy(payload.bin.begin(), payload.bin.end(), bins.begin() + in_block * (pd_ / 64));
    pack_codes(pd_, bins.data(), kBatchSize, block);

    auto *fac = reinterpret_cast<float *>(row + factor_off_bytes());
    fac[slot] = payload.triple_x;
    fac[deg_ + slot] = payload.factor_dq;
    fac[2 * deg_ + slot] = payload.factor_vq;
    auto *ids = reinterpret_cast<PID *>(row + neighbor_off_bytes());
    ids[slot] = x_id;
    if (params_.maintain_indegree) {
      if (had_old && old_id < indegree_.size()) {
        indegree_[old_id].fetch_sub(1, std::memory_order_relaxed);
      }
      if (x_id < indegree_.size()) indegree_[x_id].fetch_add(1, std::memory_order_relaxed);
    }
    return true;
  }

  /** Remove one valid slot while preserving the packed-prefix invariant. */
  void erase_slot(char *row, size_t slot, size_t degree) {
    assert(degree > 0 && slot < degree);
    const PID old_id = reinterpret_cast<const PID *>(row + neighbor_off_bytes())[slot];
    const size_t last = degree - 1;
    const size_t words = pd_ / 64;
    thread_local std::vector<uint64_t> bins;
    bins.resize(deg_ * words);
    auto *codes = reinterpret_cast<uint8_t *>(row + code_off_bytes());
    for (size_t block = 0; block < deg_ / kBatchSize; ++block) {
      unpack_codes_block(pd_,
                         codes + block * pd_ * 4,
                         bins.data() + block * kBatchSize * words);
    }
    if (slot != last) {
      std::copy(bins.begin() + last * words,
                bins.begin() + (last + 1) * words,
                bins.begin() + slot * words);
    }
    std::fill(bins.begin() + last * words, bins.begin() + (last + 1) * words, uint64_t{0});
    for (size_t block = 0; block < deg_ / kBatchSize; ++block) {
      pack_codes(pd_,
                 bins.data() + block * kBatchSize * words,
                 kBatchSize,
                 codes + block * pd_ * 4);
    }
    auto *fac = reinterpret_cast<float *>(row + factor_off_bytes());
    if (slot != last) {
      fac[slot] = fac[last];
      fac[deg_ + slot] = fac[deg_ + last];
      fac[2 * deg_ + slot] = fac[2 * deg_ + last];
    }
    fac[last] = 0;
    fac[deg_ + last] = 0;
    fac[2 * deg_ + last] = 0;
    auto *ids = reinterpret_cast<PID *>(row + neighbor_off_bytes());
    if (slot != last) ids[slot] = ids[last];
    ids[last] = 0;
    if (params_.maintain_indegree && old_id < indegree_.size()) {
      indegree_[old_id].fetch_sub(1, std::memory_order_relaxed);
    }
  }

  void add_row_indegree(const char *row, size_t degree, int delta) {
    if (!params_.maintain_indegree) return;
    const auto *ids = reinterpret_cast<const PID *>(row + neighbor_off_bytes());
    for (size_t j = 0; j < degree; ++j) {
      if (ids[j] >= indegree_.size()) continue;
      indegree_[ids[j]].fetch_add(delta, std::memory_order_relaxed);
    }
  }

  QuantizedGraph &qg_;
  UpdateParams params_;
  int fd_ = -1;             // buffered fd: all reads (page-cache served) + fallback writes
  int wfd_ = -1;            // O_DIRECT write fd (parallel inode-shared overwrites)
  bool direct_io_ = false;  // wfd_ opened successfully and writes routed to it
  std::atomic<size_t> committed_;
  std::atomic<size_t> allocated_points_;
  std::atomic<PID> next_append_id_;
  std::atomic<uint64_t> live_count_;
  std::atomic<PID> free_list_head_{kPidMax};
  std::atomic<uint64_t> free_count_{0};
  size_t dim_, res_dim_, full_dim_, pd_, deg_, node_len_, page_size_, npp_;
  std::unordered_set<PID> deleted_;
  mutable std::mutex deleted_mutex_;
  QGSuperblockV2 superblock_{};
  int active_superblock_slot_ = -1;
  std::mutex checkpoint_mutex_;
  std::mutex pending_reused_mutex_;
  std::vector<PID> pending_reused_;
  AtomicStats stats_;
  PageWriteCache write_cache_;
  std::array<StagedStripe, 64> staged_;
  std::vector<std::mutex> page_locks_;
  std::vector<std::atomic<int32_t>> indegree_;
  // Per-page seqlock: odd while a locked writer rewrites the page. Readers
  // outside the page lock validate before/after the pread and retry.
  std::vector<std::atomic<uint32_t>> page_versions_;
  // One fixed, atomic bit per PID. Set means tombstoned, free, or a reused row
  // that is still dark. Fixed sizing makes hot-path reads pointer-stable.
  std::vector<std::atomic<uint64_t>> hidden_words_;
  std::mutex routing_snapshot_mutex_;
  std::vector<std::unique_ptr<RoutingSnapshot>> routing_snapshots_;
  std::atomic<const RoutingSnapshot *> routing_snapshot_{nullptr};
};

}  // namespace alaya::laser
