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
 * CRC-protected superblocks.
 *
 * Durability (UpdateParams::enable_wal, off by default): when enabled the
 * updater keeps a per-segment after-image op-WAL (`<index>.opwal`, the SEGMENT_OP
 * family of the shared WAL7 envelope, see segment_op_wal.hpp /
 * docs/design/unified-wal-vocabulary.md). A no-steal page cache appends a
 * whole-page after-image before the page is installed, publish() group-commits
 * the batch (fsync) before advancing the watermark, and checkpoint() commits an
 * A/B superblock flip. Reopen runs a dedicated recovery path (read-only base ->
 * WAL redo under a replaying_ guard -> one authoritative trailer scan that
 * rebuilds committed/allocated/next/live/hidden/deleted and the physical length
 * -> routing repair). A durable segment lineage uid in the superblock reserved
 * area rejects a stale/foreign .opwal, and any WAL/critical-index error poisons
 * the writer (fail closed). The G1 minimal safe scope forbids PID reuse/reclaim,
 * consolidate/garden, and bloom under enable_wal — those paths throw and their
 * WAL transaction formats are a later wave. Remaining caller contracts: phase
 * separation (tombstone/consolidate vs inserts) still applies, the labels
 * sidecar is not covered by the op-WAL, and a single writer per segment must be
 * enforced above (W3 handle: exclusive flock + checkpoint/mutation lane).
 *
 * With enable_wal off the paths below are byte-for-byte unchanged; the legacy
 * ghost heuristic exists only in the one-time v1 migration scan.
 */

#pragma once

#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
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
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <set>
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
#include "index/graph/laser/qg/segment_op_wal.hpp"
#include "index/graph/laser/quantization/fastscan_impl.hpp"
#include "index/graph/laser/quantization/rabitq.hpp"
#include "index/graph/laser/space/l2.hpp"
#include "platform/fs.hpp"
#include "simd/fastscan.hpp"
#include "wal/frame.hpp"

#include "index/graph/laser/qg/detail/qg_updater_api.hpp"
#include "index/graph/laser/qg/detail/qg_updater_row_codec.hpp"

namespace alaya::laser {
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
        turnover_(std::max(params.max_points == 0 ? 2 * qg.num_points_ + 4096 : params.max_points,
                           qg.num_points_)),
        row_generations_(
            std::max(params.max_points == 0 ? 2 * qg.num_points_ + 4096 : params.max_points,
                     qg.num_points_)),
        page_versions_(
            (std::max(params.max_points == 0 ? 2 * qg.num_points_ + 4096 : params.max_points,
                      qg.num_points_) +
             qg.node_per_page_ - 1) /
            qg.node_per_page_),
        hidden_words_(
            (std::max(params.max_points == 0 ? 2 * qg.num_points_ + 4096 : params.max_points,
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
    enable_wal_ = params_.enable_wal;
    // PID reuse only makes sense with a durable free-list (codex B.1): a reuse-enabled
    // bundle pops FREE PIDs that recovery must reconstruct from the on-disk trailers,
    // which only the WAL guarantees. Fail fast on the illegal combination.
    if (params_.enable_pid_reuse && !enable_wal_) {
      throw std::invalid_argument("QGUpdater: enable_pid_reuse requires enable_wal");
    }
    enable_pid_reuse_ = params_.enable_pid_reuse && enable_wal_;
    io_observer_ = params_.io_observer;
    if (enable_wal_ && params_.direct_io) {
      throw std::invalid_argument(
          "QGUpdater: enable_wal is incompatible with direct_io (O_DIRECT bypasses the "
          "write-ahead ordering between the WAL and the index)");
    }
    // No-steal (clause B): with write_cache off, modify_node_page writes each page
    // through immediately (force_wal + pwrite), making an uncommitted after-image
    // durable BEFORE the commit fsync. A torn bundle then leaves a durable backlink
    // patch whose tentative PID is re-appended by the next transaction => a stale
    // edge. enable_wal therefore requires the no-steal page cache.
    if (enable_wal_ && !params_.write_cache) {
      throw std::invalid_argument(
          "QGUpdater: enable_wal requires write_cache (the no-steal page cache; a write-through "
          "cache would make uncommitted after-images durable before the commit fsync)");
    }
    for (auto &word : hidden_words_) word.store(0, std::memory_order_relaxed);
    for (auto &generation : row_generations_) generation.store(0, std::memory_order_relaxed);
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
    const size_t committed = committed_.load(std::memory_order_acquire);
    for (size_t id = 0; id < row_generations_.size(); ++id) {
      row_generations_[id].store(id < committed ? 1 : 0, std::memory_order_relaxed);
    }
    refresh_routing_snapshot();
    // Resident-arena seam: pre-size the arena for every PID this updater can
    // allocate so appends land inside a stable allocation; write_at() then
    // mirrors committed page writes into it. Reservation happens at ctor time,
    // never concurrently with readers.
    if (qg_.arena_resident()) {
      qg_.arena_reserve_rows(row_generations_.size());
    }
    qg_.set_result_filter(&deleted_);
    if (enable_wal_) {
      try {
        open_op_wal_and_recover();
      } catch (...) {
        qg_.set_result_filter(nullptr);
        op_wal_.reset();
        if (wfd_ >= 0) ::close(wfd_);
        if (fd_ >= 0) ::close(fd_);
        wfd_ = -1;
        fd_ = -1;
        throw;
      }
    }
  }

  QGUpdater(const QGUpdater &) = delete;
  auto operator=(const QGUpdater &) -> QGUpdater & = delete;

  ~QGUpdater() {
    if (qg_.result_filter_ == &deleted_) {
      qg_.set_result_filter(nullptr);
    }
    pid_scan_mapping_.reset();
    if (fd_ >= 0) {
      ::close(fd_);
    }
    if (wfd_ >= 0) {
      ::close(wfd_);
    }
  }

  [[nodiscard]] size_t num_points() const { return committed_.load(std::memory_order_acquire); }

  // Read-path poison gate (B-02): search/label translation never take the handle
  // write mutex, so they check this lock-free atomic latch at entry and exit. A
  // poisoned handle requires recovery — reads must fail closed rather than serve a
  // half-published or torn state.
  void ensure_readable() const {
    if (poisoned_.load(std::memory_order_acquire)) {
      throw std::runtime_error("QGUpdater op-WAL handle is poisoned (recovery required)");
    }
  }

  [[nodiscard]] bool is_poisoned() const noexcept {
    return poisoned_.load(std::memory_order_acquire);
  }

  // Current immutable appended-label snapshot (B-02). Acquire-load AFTER the
  // committed watermark used by search: the snapshot is published before committed
  // on the write side, so it always covers every binding for a committed PID.
  [[nodiscard]] std::shared_ptr<const LabelBindings> label_snapshot() const {
    const std::lock_guard<std::mutex> guard(label_snapshot_mutex_);
    return label_snapshot_;
  }
  // True iff pid is a committed, non-hidden (live) row. Used by the segment-layer
  // global bijection check (B-06) over the effective-label domain.
  [[nodiscard]] bool row_is_live(PID pid) const {
    return static_cast<size_t>(pid) < committed_.load(std::memory_order_acquire) && !is_hidden(pid);
  }
  // The current durable incarnation generation of `pid` (design 3.3 / codex B.7): the
  // generation of its live label binding, or 0 for a never-reused PID (base identity /
  // append-only). The 2B adapter's tombstone ABA check compares a captured token's
  // generation against this: a smaller current generation means the token is from the
  // future (corruption); a larger one means the PID was reused (the token is stale).
  [[nodiscard]] uint32_t durable_generation(PID pid) const {
    const auto snap = label_snapshot();
    const auto *b = snap ? snap->find_binding(pid) : nullptr;
    return b != nullptr ? b->pid_generation : 0;
  }
  [[nodiscard]] uint64_t last_committed_txid() const { return last_committed_txid_; }
  [[nodiscard]] uint64_t applied_collection_op_id() const { return applied_collection_op_id_; }

  [[nodiscard]] size_t allocated_points() const {
    return allocated_points_.load(std::memory_order_acquire);
  }

  [[nodiscard]] uint64_t live_count() const { return live_count_.load(std::memory_order_acquire); }

  [[nodiscard]] uint64_t free_count() const { return free_count_.load(std::memory_order_acquire); }

  [[nodiscard]] PID free_list_head() const {
    return free_list_head_.load(std::memory_order_acquire);
  }

  // wal-2c BLOCKER-2: the highest maintenance epoch whose consolidate transaction is
  // durably committed (0 == none). Part of the crash-matrix S_old/S_new fingerprint.
  [[nodiscard]] uint64_t last_completed_consolidate_epoch() const {
    return last_completed_consolidate_epoch_;
  }

  [[nodiscard]] uint64_t generation() const { return superblock_.generation; }
  // Test/diagnostic: the on-disk base format version (v2 legacy vs v3 2C-activated).
  [[nodiscard]] uint32_t superblock_format_version() const { return superblock_.format_version; }
  // Test/diagnostic: true once the v3 pid-reuse feature is active in the base.
  [[nodiscard]] bool pid_generation_activated() const { return pid_generation_activated_; }
  [[nodiscard]] uint64_t segment_uid() const { return segment_uid_; }
  // Debug/test read-only accessors for the crash-matrix fingerprint (routing + medoid +
  // hidden state), so a recovered state can be compared to S_old/S_new bit-for-bit.
  [[nodiscard]] PID entry_point() const { return qg_.entry_point_; }
  [[nodiscard]] const std::vector<PID> &medoids() const { return qg_.medoids_; }
  [[nodiscard]] bool row_hidden(PID id) const { return is_hidden(id); }
  [[nodiscard]] uint32_t maintenance_activation_gen() const {
    return static_cast<uint32_t>(maintenance_activation_gen_);
  }
  [[nodiscard]] uint32_t pid_reuse_activation_gen() const {
    return static_cast<uint32_t>(pid_reuse_activation_gen_);
  }

  [[nodiscard]] int active_superblock_slot() const { return active_superblock_slot_; }

  // An activated mutable source may become readable through a sealed role after
  // Collection rotation. Before that handoff both A/B copies must be supported v3
  // images carrying every feature required by the selected base; a stale v2 or
  // lower-feature fallback is not safe to expose.
  void require_dual_v3_if_activated() {
    if (superblock_.format_version == kQGFormatVersion) {
      return;
    }
    try {
      if (superblock_.format_version != kQGFormatVersionV3) {
        poison("activated QG selected an unsupported outer superblock version");
      }
      std::array<char, kSectorLen> header{};
      read_at(0, header.data(), header.size());
      QGSuperblockV2 copies[kQGSuperblockCopies];
      std::memcpy(&copies[0], header.data(), sizeof(QGSuperblockV2));
      std::memcpy(&copies[1], header.data() + kQGSuperblockSize, sizeof(QGSuperblockV2));
      const uint32_t required = qg_read_required_feature_flags(superblock_);
      for (const auto &copy : copies) {
        if (!qg_superblock_valid(copy) || copy.format_version != kQGFormatVersionV3 ||
            !qg_superblock_supported(copy, kQgSupportedRequiredFeatures) ||
            (qg_read_required_feature_flags(copy) & required) != required) {
          poison("activated QG requires two supported v3 superblock copies before role handoff");
        }
      }
    } catch (...) {
      poison_current_exception("failed to verify dual v3 superblock copies");
    }
  }

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
  [[nodiscard]] size_t maintenance_evict_stride() const { return params_.maintenance_evict_stride; }

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
    s.patch_intents_prepared = stats_.patch_intents_prepared.load();
    s.patch_intents_applied = stats_.patch_intents_applied.load();
    s.patch_intent_stale_fallbacks = stats_.patch_intent_stale_fallbacks.load();
    s.full_recomputes = stats_.full_recomputes.load();
    s.forced_links = stats_.forced_links.load();
    s.evict_tel_samples = stats_.evict_tel_samples.load();
    s.evict_tel_agree = stats_.evict_tel_agree.load();
    for (size_t i = 0; i < s.evict_tel_regret.size(); ++i) {
      s.evict_tel_regret[i] = stats_.evict_tel_regret[i].load();
    }
    s.evict_tel_relerr_sum = static_cast<double>(stats_.evict_tel_relerr_nano.load()) / 1e9;
    s.consolidated_rows = stats_.consolidated_rows.load();
    s.spliced_slots = stats_.spliced_slots.load();
    s.ghosted_slots = stats_.ghosted_slots.load();
    s.bloom_scan_rows = stats_.bloom_scan_rows.load();
    s.bloom_candidate_rows = stats_.bloom_candidate_rows.load();
    s.bloom_scan_us = stats_.bloom_scan_us.load();
    s.bloom_row_us = stats_.bloom_row_us.load();
    s.bloom_finalize_us = stats_.bloom_finalize_us.load();
    s.freed_slots = stats_.freed_slots.load();
    s.reused_slots = stats_.reused_slots.load();
    s.gardened_rows = stats_.gardened_rows.load();
    s.garden_pump_links = stats_.garden_pump_links.load();
    s.garden_us = stats_.garden_us.load();
    s.garden_selected_turnover_sum = stats_.garden_selected_turnover_sum.load();
    s.garden_selected_turnover_rows = stats_.garden_selected_turnover_rows.load();
    s.garden_all_turnover_sum = stats_.garden_all_turnover_sum.load();
    s.garden_all_turnover_rows = stats_.garden_all_turnover_rows.load();
    s.maintenance_peak_pool_pages = stats_.maintenance_peak_pool_pages.load();
    s.maintenance_peak_overlay_pages = stats_.maintenance_peak_overlay_pages.load();
    s.maintenance_page_frames = stats_.maintenance_page_frames.load();
    s.maintenance_page_frame_bytes = stats_.maintenance_page_frame_bytes.load();
    s.maintenance_last_preflight_page_frames = stats_.maintenance_last_preflight_page_frames.load();
    s.maintenance_last_preflight_wal_bytes = stats_.maintenance_last_preflight_wal_bytes.load();
    s.garden_skipped = stats_.garden_skipped.load();
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
  [[nodiscard]] std::vector<PID> search(const float *query_vec,
                                        size_t k,
                                        size_t ef,
                                        size_t max_beam_width = 16,
                                        float *distances = nullptr) {
    if (query_vec == nullptr) throw std::invalid_argument("QGUpdater::search null query");
    if (k == 0) return {};
    if (max_beam_width == 0) {
      throw std::invalid_argument("QGUpdater::search max_beam_width must be > 0");
    }
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
        const float d = space::l2_sqr(tvec, routing->medoid_vectors.data() + full_dim_ * m, dim_);
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
      beam_width = std::min(max_beam_width, beam_width * 2);
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
      if (candidate.id < snapshot && !is_hidden(candidate.id)) {
        if (distances != nullptr) {
          distances[out.size()] = candidate.distance;
        }
        out.push_back(candidate.id);
      }
      if (out.size() == k) break;
    }
    return out;
  }

  [[nodiscard]] int32_t indegree(PID id) const {
    if (!params_.maintain_indegree || id >= indegree_.size()) return 0;
    return indegree_[id].load(std::memory_order_relaxed);
  }

  [[nodiscard]] uint16_t turnover(PID id) const {
    if (!params_.maintain_turnover || id >= turnover_.size()) return 0;
    return turnover_[id].load(std::memory_order_relaxed);
  }

  /** Reinitialize the optional RAM-only row-turnover sidecar. */
  void init_turnover() {
    if (!params_.maintain_turnover) return;
    for (auto &value : turnover_) value.store(0, std::memory_order_relaxed);
  }

  /** Distribution over current live rows; counters are never persisted. */
  [[nodiscard]] TurnoverSummary turnover_summary() const {
    TurnoverSummary out;
    if (!params_.maintain_turnover) return out;
    const size_t n = committed_.load(std::memory_order_acquire);
    std::vector<uint16_t> values;
    values.reserve(std::min<uint64_t>(n, live_count_.load(std::memory_order_acquire)));
    for (size_t i = 0; i < n; ++i) {
      const PID id = static_cast<PID>(i);
      if (is_hidden(id)) continue;
      const uint16_t value = turnover(id);
      values.push_back(value);
      out.sum += value;
    }
    out.rows = values.size();
    if (!values.empty()) {
      std::sort(values.begin(), values.end());
      out.p50 = values[static_cast<size_t>(0.50 * static_cast<double>(values.size() - 1))];
      out.p99 = values[static_cast<size_t>(0.99 * static_cast<double>(values.size() - 1))];
    }
    return out;
  }

  /** Rebuild the optional RAM indegree index from the current committed graph. */
  void init_indegree(size_t num_threads) {
    if (!params_.maintain_indegree) return;
    for (auto &v : indegree_) v.store(0, std::memory_order_relaxed);
    const size_t n = committed_.load(std::memory_order_acquire);
    const int nt = static_cast<int>(std::max<size_t>(1, num_threads));
    parallel_for_catch(0, static_cast<int64_t>(n), nt, 256, [&](int64_t ui) {
      if (is_hidden(static_cast<PID>(ui))) return;
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
    });
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
    if (enable_wal_) ensure_writable();
    if (!mark_hidden(id)) return;
    churn_since_garden_.fetch_add(1, std::memory_order_relaxed);
    mirror_deleted_insert(id);
    if (enable_wal_ && !replaying_) {
      // Durability of a tombstone is the next publish/checkpoint/flush fsync; the
      // trailer flag itself is captured by the modify_node_page after-image below.
      wal_append(encode_tombstone(segment_uid_, superblock_.generation, id),
                 alaya::wal::WalFile::Sync::buffered);
      wal_failpoint(SegmentOpFailPoint::after_wal_append_before_apply);
    }
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
    if (enable_wal_) ensure_writable();
    // PID reuse/reclaim is out of the G1 scope (clause A): under enable_wal every
    // allocation is a pure append.
    PID id = enable_wal_ ? kPidMax : pop_free_slot();
    const bool reused = id != kPidMax;
    if (!reused) {
      id = next_append_id_.fetch_add(1, std::memory_order_acq_rel);
      if (id == kPidMax) {
        throw std::overflow_error("QGUpdater: PID space exhausted");
      }
      note_allocated(static_cast<size_t>(id) + 1);
    }
    insert_with_id_impl(vec, id, reused);
    churn_since_garden_.fetch_add(1, std::memory_order_relaxed);
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
    // The kind=5 emit is byte-identical to the pre-refactor path (golden bytes +
    // the 12-cell crash matrix protect it); commit_physical_bundle reuses the
    // shared core below with a snapshot-swap emit instead of a publish frame.
    publish_common(new_committed, [&] {
      if (enable_wal_ && !replaying_) {
        if (has_staged_edges()) {
          poison("staged backlinks must be drained before publish under enable_wal");
        }
        wal_failpoint(SegmentOpFailPoint::after_apply_before_publish_fsync);
        wal_append(encode_publish(segment_uid_, superblock_.generation, new_committed),
                   alaya::wal::WalFile::Sync::fsync);
        wal_failpoint(SegmentOpFailPoint::after_publish_fsync);
      }
    });
  }

  // Shared publish core (clause B): validate the watermark, drain reused ids,
  // then run `emit_record` (the kind=5 publish frame for publish(); the label
  // snapshot swap for a bundle) at exactly the pre-refactor point — after the
  // reused-row visibility fixups and before the committed release-store. Group
  // commit still forces every buffered after-image durable inside `emit_record`.
  template <typename EmitFn>
  void publish_common(size_t new_committed, EmitFn &&emit_record) {
    if (enable_wal_) ensure_writable();
    const size_t old_committed = committed_.load(std::memory_order_acquire);
    if (new_committed < old_committed || new_committed > allocated_points()) {
      throw std::invalid_argument("QGUpdater::publish invalid committed watermark");
    }

    std::vector<PID> reused;
    {
      const std::lock_guard<std::mutex> guard(pending_reused_mutex_);
      reused.swap(pending_reused_);
    }
    if (enable_wal_ && !reused.empty()) {
      poison("PID reuse is disabled under enable_wal (clause A)");
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
    emit_record();
    live_count_.fetch_add(reused.size() + (new_committed - old_committed),
                          std::memory_order_acq_rel);
    committed_.store(new_committed, std::memory_order_release);
    qg_.num_points_ = new_committed;
  }

  // Commit a physical label bundle and return per-row (pid, generation) tokens
  // (design 3.3 / codex B.7). The canonical path returns real reuse tokens; the
  // legacy 2A path returns dense gen=0 tokens over [old_hwm, new_hwm). The 2B adapter
  // binds label->token from `rows` instead of guessing a dense range (mixed / all-reuse
  // bundles are NOT dense). Preconditions throw (caller error, never poison); any I/O
  // or internal failure poisons.
  // Shared bundle preconditions (caller errors -- validated BEFORE any mutating work, so
  // they never poison). Kept in one place so the token + pair entries agree exactly.
  void validate_bundle_preconditions(uint64_t txid,
                                     uint64_t applied_op_id,
                                     const float *vecs,
                                     const uint64_t *labels,
                                     size_t n) {
    if (!enable_wal_) {
      throw std::logic_error("QGUpdater::commit_physical_bundle requires enable_wal");
    }
    ensure_writable();
    if (n == 0 || vecs == nullptr || labels == nullptr) {
      throw std::invalid_argument("commit_physical_bundle: empty/null bundle");
    }
    if (txid <= last_committed_txid_) {
      throw std::invalid_argument("commit_physical_bundle: txid must exceed last_committed_txid");
    }
    if (applied_op_id < applied_collection_op_id_) {
      throw std::invalid_argument(
          "commit_physical_bundle: applied_collection_op_id must not regress");
    }
  }

  PhysicalBundleResult commit_physical_bundle_tokens(uint64_t txid,
                                                     uint64_t applied_op_id,
                                                     const float *vecs,
                                                     const uint64_t *labels,
                                                     size_t n) {
    validate_bundle_preconditions(txid, applied_op_id, vecs, labels, n);
    // Canonical prebind branch (design B.4): once PID reuse is activated (v3 pid base)
    // OR opt-in via enable_pid_reuse, every bundle -- append-only, mixed, or all-reuse --
    // takes the append-only canonical writer. A never-activated 2A segment keeps the
    // legacy interleaved path byte-for-byte.
    if (pid_generation_activated_ || enable_pid_reuse_) {
      return commit_physical_bundle_canonical(txid, applied_op_id, vecs, labels, n);
    }
    // BLOCKER-6: pre-allocate the token result BEFORE the mutating/commit region. Otherwise a
    // post-commit std::bad_alloc (out.rows growth) reports failure to the caller while the
    // legacy transaction is already durably committed AND the handle un-poisoned -- a torn
    // "committed but reported failed" state the pair callers never had. The legacy body
    // returns a dense gen-0 range, so reserving n up front means the fill never allocates.
    PhysicalBundleResult out;
    out.rows.reserve(n);
    const auto range = commit_physical_bundle_legacy_2a(txid, applied_op_id, vecs, labels, n);
    out.old_hwm = range.first;
    out.new_hwm = range.second;
    for (size_t i = 0; i < n; ++i) {
      out.rows.push_back(PidToken{static_cast<PID>(range.first + i), 0});
    }
    return out;
  }

  // Back-compat pair entry (existing callers unchanged): the appended PID range
  // [old_hwm, new_hwm). BLOCKER-6: the legacy path goes STRAIGHT to the legacy body -- there
  // is no token vector to allocate, so there is no post-commit allocation window at all.
  std::pair<PID, PID> commit_physical_bundle(uint64_t txid,
                                             uint64_t applied_op_id,
                                             const float *vecs,
                                             const uint64_t *labels,
                                             size_t n) {
    validate_bundle_preconditions(txid, applied_op_id, vecs, labels, n);
    if (pid_generation_activated_ || enable_pid_reuse_) {
      const auto r = commit_physical_bundle_canonical(txid, applied_op_id, vecs, labels, n);
      return {static_cast<PID>(r.old_hwm), static_cast<PID>(r.new_hwm)};
    }
    return commit_physical_bundle_legacy_2a(txid, applied_op_id, vecs, labels, n);
  }

  // Legacy 2A interleaved bundle (append kind=7 per row + one kind=8 fsync as the
  // single durable commit point, snapshot swap, publish_common). Byte-for-byte the
  // pre-reuse path; the shared preconditions are validated by the caller.
  std::pair<PID, PID> commit_physical_bundle_legacy_2a(uint64_t txid,
                                                       uint64_t applied_op_id,
                                                       const float *vecs,
                                                       const uint64_t *labels,
                                                       size_t n) {
    const size_t old_hwm = committed_.load(std::memory_order_acquire);
    if (old_hwm != allocated_points()) {
      throw std::logic_error("commit_physical_bundle requires allocated == committed");
    }
    const size_t new_hwm = old_hwm + n;
    if (new_hwm > static_cast<size_t>(kPidMax) || new_hwm > row_generations_.size()) {
      throw std::invalid_argument("commit_physical_bundle: PID capacity exceeded");
    }
    // B-07: capture the pre-bundle live population so we can switch the entry point
    // to the first new PID on an empty->non-empty transition (fresh segment or all
    // rows previously tombstoned) after the bundle commits.
    const bool was_empty_of_live = live_count_.load(std::memory_order_acquire) == 0;
    // Mutating section (B-04 hardening): past this point allocated may exceed
    // committed, so ANY failure -- including an exception that does not itself
    // poison (e.g. std::bad_alloc while pre-building the snapshot) -- must poison
    // the handle. Otherwise a later add() would kind=5-publish over the allocation
    // gap and commit the failed bundle's rows WITHOUT bindings (identity labels).
    try {
      // (0) B-07 empty->non-empty entry-point switch: when no rows were live before
      // this bundle (fresh segment, or every prior row tombstoned), the persisted/
      // default entry point is dead and sits in a disconnected hidden component, so
      // search_for_insert would seed that stale component and leave every new row
      // edgeless. Point the entry at the first row of THIS bundle and publish the
      // routing snapshot BEFORE inserting, so rows 1..n-1 seed from row 0 (via the
      // bundle-internal visibility floor) and form a connected component. Concurrent
      // search never seeds it early (it gates entry_point < committed_, still old
      // here); a torn bundle re-derives it on reopen via rebuild_state_after_replay
      // -> repair_routing_roots.
      if (was_empty_of_live) {
        qg_.entry_point_ = static_cast<PID>(old_hwm);
        refresh_routing_snapshot();
      }
      // (1) allocate + insert each row and append its kind=7 bind (buffered). Row
      // after-images buffer too; the kind=8 fsync forces the whole group durable.
      for (size_t i = 0; i < n; ++i) {
        // B-07 bundle-internal visibility: expose rows 0..i-1 of THIS bundle to
        // row i's search_for_insert (old_hwm base rows + i already-appended rows).
        insert_visible_override_ = old_hwm + i;
        const PID pid = allocate_and_insert(vecs + i * dim_);
        if (static_cast<size_t>(pid) != old_hwm + i) {
          poison("commit_physical_bundle: append produced a non-dense PID");
        }
        wal_append(encode_label_bind(segment_uid_,
                                     superblock_.generation,
                                     txid,
                                     static_cast<uint64_t>(i),
                                     pid,
                                     /*pid_generation=*/0,
                                     labels[i]),
                   alaya::wal::WalFile::Sync::buffered,
                   txid);
      }
      insert_visible_override_ = 0;  // B-07: bundle window closed
      wal_failpoint(SegmentOpFailPoint::after_label_bind_append);
      // Reverse edges must be materialized inline before the commit (B-03/clause B):
      // a staged (deferred) backlink would let kind=8 commit a row whose only routing
      // edges live in RAM and vanish on a crash, leaving it permanently unreachable.
      if (has_staged_edges()) {
        poison(
            "commit_physical_bundle: staged backlinks must be drained (inline patching "
            "is required under enable_wal)");
      }
      // (2) pre-build the new immutable snapshot (zero allocation / no failure after
      // this point on the publish path).
      auto next = std::make_shared<LabelBindings>(*load_label_snapshot());
      for (size_t i = 0; i < n; ++i) {
        next->bindings.emplace(static_cast<PID>(old_hwm + i), PidBinding{0, labels[i]});
      }
      std::shared_ptr<const LabelBindings> published = std::move(next);
      // (3) append kind=8 + fsync: the single durable commit point of the bundle.
      wal_failpoint(SegmentOpFailPoint::before_tx_publish_append);
      wal_append(encode_tx_publish(segment_uid_,
                                   superblock_.generation,
                                   txid,
                                   static_cast<uint64_t>(new_hwm),
                                   static_cast<uint64_t>(n),
                                   applied_op_id),
                 alaya::wal::WalFile::Sync::fsync,
                 txid);
      wal_failpoint(SegmentOpFailPoint::after_tx_publish_fsync);
      // (4) publish snapshot (release) THEN committed (release), via the shared core.
      publish_common(new_hwm, [&] {
        store_label_snapshot(published);
      });
      ++label_content_revision_;  // published bindings changed (design 3.1 slot-dirty tracking)
      last_committed_txid_ = txid;
      applied_collection_op_id_ = applied_op_id;
      // (5) B-07: republish the routing snapshot after the committed watermark
      // advanced (ruling 6: refresh after publish) so search now seeds the switched
      // entry point that is finally < committed_.
      if (was_empty_of_live) {
        refresh_routing_snapshot();
      }
      return {static_cast<PID>(old_hwm), static_cast<PID>(new_hwm)};
    } catch (const std::exception &error) {
      insert_visible_override_ = 0;
      // poison() re-throws (preserving the first reason if already poisoned).
      poison(std::string("commit_physical_bundle failed mid-transaction: ") + error.what());
    }
  }

  // Canonical prebind physical bundle (design B.4 / codex B.4): the append-only,
  // reuse-safe replacement for the legacy interleaved bundle. Reserves all PIDs first
  // (canonical free-chain pops for the reuse prefix, dense append for the rest), builds
  // every row + reverse edge + bundle spine into a writer-private page overlay, appends
  // ALL kind=7 then the reused-page FREE preimages then the final kind=1 pages, forces
  // ONE kind=8 as the durable commit point, then installs + publishes in a fixed order.
  // Any failure past the reservation poisons (allocated may exceed committed; reopen
  // rebuilds the free chain from the on-disk trailers). Returns per-row (pid, generation)
  // tokens so the 2B adapter binds label->token instead of guessing a dense range.
  PhysicalBundleResult commit_physical_bundle_canonical(uint64_t txid,
                                                        uint64_t applied_op_id,
                                                        const float *vecs,
                                                        const uint64_t *labels,
                                                        size_t n) {
    // Activation must precede the checkpoint lane lock: ensure_pid_reuse_activated may
    // drive checkpoint(), which takes checkpoint_mutex_ (exactly like consolidate calls
    // ensure_maintenance_activated before its guard).
    ensure_pid_reuse_activated();
    const std::lock_guard<std::mutex> checkpoint_guard(checkpoint_mutex_);
    // B-07: capture the pre-bundle live population so the entry point can switch to the
    // first new row on an empty->non-empty transition (fresh / all-tombstoned segment).
    const bool was_empty_of_live = live_count_.load(std::memory_order_acquire) == 0;
    // Reserve every PID BEFORE the mutating section. Preconditions (allocated==committed,
    // free-chain rebuilt, no staged edges, capacity) throw as caller errors here; a
    // corrupt canonical free chain poisons inside reserve. State is published (kBuilding)
    // so a concurrent checkpoint is refused for the whole bundle (incl. all-reuse).
    PhysicalBundleResult result = reserve_bundle_pids(n, /*allow_reuse=*/enable_pid_reuse_);
    const uint64_t old_hwm = result.old_hwm;
    const uint64_t new_hwm = result.new_hwm;
    // BLOCKER-3: past the reservation, bundle_state_ is kBuilding, so EVERY subsequent
    // statement (including the context allocation) runs inside the poison-on-exception
    // boundary -- a bad_alloc here must fail closed, never latch a non-idle state silently.
    try {
      BundleInsertContext ctx;
      ctx.old_hwm = old_hwm;
      ctx.reserved.reserve(n * 2);
      for (const auto &tok : result.rows) {
        ctx.reserved.insert(tok.pid);
      }
      // R0: tokens reserved (free PIDs popped in RAM), before any durable canonical frame.
      // A crash here drops the whole bundle -- the popped PIDs are still FREE on disk, so
      // recovery rebuilds the old free set (S_old).
      wal_failpoint(SegmentOpFailPoint::after_reuse_reserve_before_binds);
      // (1) pre-build the next immutable snapshot: insert_or_assign so a reused PID's new
      // incarnation REPLACES its old {generation,label} (a dense append inserts). Built
      // before the commit point so the post-kind=8 publish allocates nothing.
      auto next = std::make_shared<LabelBindings>(*load_label_snapshot());
      for (size_t i = 0; i < n; ++i) {
        next->bindings.insert_or_assign(result.rows[i].pid,
                                        PidBinding{result.rows[i].pid_generation, labels[i]});
      }
      std::shared_ptr<const LabelBindings> published = std::move(next);
      // (2) append ALL kind=7 label binds (row_op 0..n-1, buffered; frame batch_id == txid).
      // Every kind=7 precedes every kind=1 (the canonical replay lane poisons a kind=7 after
      // a kind=1), so the whole bind set is logged up front.
      for (size_t i = 0; i < n; ++i) {
        wal_append(encode_label_bind(segment_uid_,
                                     superblock_.generation,
                                     txid,
                                     static_cast<uint64_t>(i),
                                     result.rows[i].pid,
                                     result.rows[i].pid_generation,
                                     labels[i]),
                   alaya::wal::WalFile::Sync::buffered,
                   txid);
        if (i == 0) {
          wal_failpoint(SegmentOpFailPoint::after_reuse_first_bind_append);  // R1a: partial bind
        }
      }
      wal_failpoint(SegmentOpFailPoint::after_label_bind_append);
      // Route the writer's page RMW/dependency reads into the private overlay from here.
      bundle_ctx_ = &ctx;
      // (3) FREE preimage: for each unique page holding a RESERVED reused PID, log its
      // committed FREE image once, BEFORE any modification (design B.4). Recovery keeps
      // the FIRST kind=1 per page as the pre-transaction FREE evidence; the on-disk page
      // may already be sector-torn by the time recovery runs, so the preimage is the only
      // durable proof the reused row was FREE at transaction start.
      std::set<size_t> reuse_pages;
      for (const auto &tok : result.rows) {
        if (tok.pid_generation != 0) {
          reuse_pages.insert(page_index(tok.pid));
        }
      }
      for (size_t pi : reuse_pages) {
        const char *page = bundle_overlay_page(pi);  // committed FREE image (reused row FREE)
        bundle_log_page(pi, page, alaya::wal::WalFile::Sync::buffered);
      }
      wal_failpoint(SegmentOpFailPoint::after_reuse_free_preimage_before_build);  // R2
      // (4) build every row + reverse edges into the private overlay. A per-row seed
      // spine (last_built -> current) keeps each row reachable DURING construction; the
      // authoritative bidirectional spine is installed as a FINAL pass below (a later
      // row's robust_prune / full-recompute may otherwise rewrite an earlier spine edge).
      size_t appends_built = 0;
      std::vector<char> spine_page(page_size_);
      for (size_t i = 0; i < n; ++i) {
        const PID pid = result.rows[i].pid;
        const bool reused = result.rows[i].pid_generation != 0;
        // Appended rows use the visibility floor (reveal earlier appends of this bundle);
        // reused rows built earlier are revealed to the writer via ctx.revealed instead.
        insert_visible_override_ = reused ? old_hwm : (old_hwm + appends_built);
        if (ctx.private_entry == kPidMax) {
          ctx.private_entry = pid;
        }
        insert_with_id_impl(vecs + i * dim_, pid, reused);
        insert_visible_override_ = 0;
        ctx.revealed.insert(pid);
        if (!reused) {
          ++appends_built;
        }
        if (ctx.last_built != kPidMax && ctx.last_built != pid) {
          bundle_force_edge(ctx.last_built, pid, spine_page);  // build-time seed spine
        }
        ctx.last_built = pid;
      }
      insert_visible_override_ = 0;
      if (has_staged_edges()) {
        poison("canonical bundle: staged backlinks must be drained (inline patching required)");
      }
      // (4b) FINAL directed Hamiltonian cycle (MAJOR-1 fix): after every ordinary backlink /
      // full-recompute has run, force rows[i] -> rows[(i+1) % n] so every bundle row has an
      // outgoing spine edge and the cycle makes ALL bundle rows reachable from ANY one of
      // them -- exactly the row recovery / repair_routing_roots may select as the entry after
      // a delete-all. Each row is the `from` of EXACTLY ONE forced edge, so a forced eviction
      // (when a row is degree-saturated) can never remove a spine edge installed for another
      // row. The earlier bidirectional pass modified each middle row twice, so the second
      // forced eviction could drop the just-installed neighbor -- it was not a stable
      // invariant. Nothing rewrites these edges before finalize.
      for (size_t i = 0; n >= 2 && i < n; ++i) {
        bundle_force_edge(result.rows[i].pid, result.rows[(i + 1) % n].pid, spine_page);
      }
      // (5) finalize: append each dirty overlay page's final kind=1 (buffered).
      bundle_finalize_pages();
      // (6) B-2C-02 self-check: every bound PID's final overlay page trailer is live. Read
      // each page into a single reused scratch buffer (MAJOR-3: never re-resident spilled
      // pages here, or a large bundle under a tiny cap would blow the overlay memory bound).
      std::vector<char> check_page(page_size_);
      for (const auto &tok : result.rows) {
        bundle_read_page_scratch(page_index(tok.pid), check_page.data());
        const QGRowTrailer tr = qg_read_page_trailer(check_page.data(),
                                                     page_size_,
                                                     npp_,
                                                     static_cast<size_t>(tok.pid) % npp_);
        if ((tr.flags & (kQGRowTombstone | kQGRowFree)) != 0) {
          poison("canonical bundle: bound PID final trailer is not live (B-2C-02)");
        }
      }
      // (7) kind=8 buffered -> force: the single durable commit point of the bundle.
      // Split append from force (like the consolidate END) so the crash matrix can cut a
      // torn / unforced END: a SIGKILL after the buffered append leaves a complete kind=8
      // in the page cache -> recovery rolls forward (S_new); a power-loss that drops the
      // unforced tail loses it -> the whole bundle is discarded (S_old).
      wal_failpoint(SegmentOpFailPoint::before_tx_publish_append);
      wal_append(encode_tx_publish(segment_uid_,
                                   superblock_.generation,
                                   txid,
                                   static_cast<uint64_t>(new_hwm),
                                   static_cast<uint64_t>(n),
                                   applied_op_id),
                 alaya::wal::WalFile::Sync::buffered,
                 txid);
      wal_failpoint(SegmentOpFailPoint::after_reuse_tx_publish_append_before_fsync);
      force_wal();  // the single durable commit point of the bundle
      wal_failpoint(SegmentOpFailPoint::after_tx_publish_fsync);
      // (8) commit point passed -- publish in the fixed order (design B.4). Any failure
      // past the kind=8 fsync poisons (never rolls back; reopen rolls forward).
      bundle_install_to_cache();  // final pages -> shared write cache (seqlock, mark dirty)
      bundle_ctx_ = nullptr;      // overlay consumed; RMWs take the normal path again
      wal_failpoint(SegmentOpFailPoint::after_reuse_install_before_snapshot);  // R6b
      // Fixed publish order (design B.4 / BLOCKER-1): snapshot -> routing -> hidden ->
      // committed. Nothing is published to the query face until routing is set, and the
      // committed watermark (which makes new rows visible to a concurrent query) is LAST.
      store_label_snapshot(published);  // (a) snapshot
      // (b) routing: relocate over the NEW watermark, seeded from the LAST-CHECKPOINT entry
      // (superblock_.entry_point) so this reproduces recovery's single relocate-from-base --
      // recovery seeds from the same base entry and repairs once at the end, so both land on
      // the identical entry (BLOCKER-1: the old repair scanned committed_==old_hwm and left
      // a dead entry a delete-all->all-append bundle, diverging from replay). The bundle's
      // rows are force-live even though the reused ones are still hidden here (their trailers
      // are already live in the cache); a still->committed_ query cannot seed a new row yet
      // (routing entry gated by < committed_) and skips a hidden reused entry in the beam.
      std::unordered_set<PID> bundle_live;
      bundle_live.reserve(result.rows.size() * 2);
      for (const auto &tok : result.rows) {
        bundle_live.insert(tok.pid);
      }
      repair_routing_roots_seeded(superblock_.entry_point, new_hwm, &bundle_live);
      wal_failpoint(SegmentOpFailPoint::after_reuse_routing_before_hidden_clear);  // R7
      // (c) hidden: reveal reused rows to the query face -- trailers are already live in the
      // cache, so clear the result-filter tombstone then the hidden bit (its release is the
      // visibility point). The routing entry published above is now backed by a live row.
      bool cleared_one = false;
      for (const auto &tok : result.rows) {
        if (tok.pid_generation != 0) {
          mirror_deleted_erase(tok.pid);
          clear_hidden(tok.pid);
          if (!cleared_one) {
            cleared_one = true;
            wal_failpoint(
                SegmentOpFailPoint::after_reuse_hidden_clear_partial_before_commit);  // R8
          }
        }
      }
      // (d) committed: appended rows become visible to a concurrent query only now.
      committed_.store(new_hwm, std::memory_order_release);
      qg_.num_points_ = new_hwm;
      live_count_.fetch_add(n, std::memory_order_acq_rel);
      ++label_content_revision_;
      last_committed_txid_ = txid;
      applied_collection_op_id_ = applied_op_id;
      bundle_state_ = BundleState::kIdle;
      reservation_count_ = 0;
      return result;
    } catch (...) {
      // BLOCKER-2: reservation state is kBuilding here, so ANY exception -- including a
      // non-std::exception failpoint throw or a bad_alloc while composing a diagnostic --
      // must fail the handle closed. Latch first (noexcept), then rethrow the original.
      insert_visible_override_ = 0;
      bundle_ctx_ = nullptr;  // stop routing into a half-built overlay
      poison_current_exception("canonical bundle failed mid-transaction");
    }
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
      // QPatch phase 1: encode every selected reverse edge before acquiring
      // any target-row lock.  CapturedNode owns the raw snapshot and its row
      // generation, so apply can validate the payload without another read.
      std::vector<PatchIntent> patch_intents;
      if (params_.preencode_patch_intents &&
          params_.backlink_mode != UpdateParams::Backlink::kFullPrune) {
        patch_intents.reserve(sel.size());
        float candidate_res_sqr = 0;
        for (size_t j = 0; j < res_dim_; ++j) {
          candidate_res_sqr += tvec[dim_ + j] * tvec[dim_ + j];
        }
        // rot(candidate) is shared by every u->candidate intent belonging to
        // this insertion; computing it once avoids O(selected_degree)
        // duplicate FHTs in the preencode arm.
        thread_local std::vector<float> intent_candidate_pad;
        thread_local std::vector<float> intent_candidate_rot;
        intent_candidate_pad.assign(pd_, 0.0F);
        intent_candidate_rot.resize(pd_);
        std::copy(tvec, tvec + dim_, intent_candidate_pad.begin());
        qg_.rotator_.rotate(intent_candidate_pad.data(), intent_candidate_rot.data());
        for (size_t idx : sel) {
          patch_intents.push_back(
              prepare_patch_intent_from_rotated_candidate(pool[idx].id,
                                                          pool[idx].raw(),
                                                          tvec,
                                                          intent_candidate_rot.data(),
                                                          candidate_res_sqr,
                                                          id,
                                                          pool[idx].row_generation));
        }
      }
      if (params_.stage_backlinks && params_.backlink_mode != UpdateParams::Backlink::kFullPrune) {
        for (size_t rank = 0; rank < sel.size(); ++rank) {
          const auto &v = pool[sel[rank]];
          StagedEdge edge{v.id, id, v.dist, rank == 0, nullptr};
          if (!patch_intents.empty()) {
            edge.intent = std::make_unique<PatchIntent>(std::move(patch_intents[rank]));
          }
          stage_edge(std::move(edge));
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
      for (size_t rank = 0; rank < sel.size(); ++rank) {
        const size_t idx = sel[rank];
        if (params_.backlink_mode == UpdateParams::Backlink::kFullPrune) {
          links += static_cast<size_t>(full_reverse_recompute(pool[idx], id, tvec));
        } else if (!patch_intents.empty()) {
          const PatchApplyResult result =
              apply_patch_intent(patch_intents[rank], tvec, captured, /*force=*/false);
          if (result == PatchApplyResult::kStale) {
            stats_.patch_intent_stale_fallbacks.fetch_add(1, std::memory_order_relaxed);
            links += static_cast<size_t>(
                patch_reverse_edge(pool[idx], id, tvec, captured, /*force=*/false));
          } else {
            links += static_cast<size_t>(result == PatchApplyResult::kApplied);
          }
        } else {
          links += static_cast<size_t>(
              patch_reverse_edge(pool[idx], id, tvec, captured, /*force=*/false));
        }
      }
      // Reachability guarantee: a node with zero in-edges is unsearchable, so
      // force one link onto the closest selected neighbor (evicting its
      // farthest edge unconditionally).
      if (links == 0) {
        bool forced = false;
        if (!patch_intents.empty()) {
          const PatchApplyResult result =
              apply_patch_intent(patch_intents[0], tvec, captured, /*force=*/true);
          if (result == PatchApplyResult::kStale) {
            stats_.patch_intent_stale_fallbacks.fetch_add(1, std::memory_order_relaxed);
            forced = patch_reverse_edge(pool[sel[0]], id, tvec, captured, /*force=*/true);
          } else {
            forced = result == PatchApplyResult::kApplied;
          }
        } else {
          forced = patch_reverse_edge(pool[sel[0]], id, tvec, captured, /*force=*/true);
        }
        if (forced) {
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
   * free-list. `bloom_consolidate` first scans only row trailers/PID prefixes
   * through a shared file mapping and runs the full RMW only for exact-confirmed
   * Bloom hits. Must not run concurrently with inserts.
   */
  void consolidate(size_t num_threads,
                   size_t r_target = 0,
                   bool reclaim_slots = true,
                   bool bloom_consolidate = false) {
    if (enable_wal_) {
      // 2C: consolidate is now a real maintenance transaction (design section 1).
      consolidate_wal_transaction(num_threads, r_target, reclaim_slots, bloom_consolidate);
      return;
    }
    const auto consolidate_begin = std::chrono::steady_clock::now();
    const size_t n = committed_.load(std::memory_order_acquire);
    const size_t target = r_target == 0 ? deg_ : std::min(r_target, deg_);
#if defined(__SANITIZE_THREAD__)
    const int nt = bloom_consolidate ? 1 : static_cast<int>(std::max<size_t>(1, num_threads));
#else
    const int nt = static_cast<int>(std::max<size_t>(1, num_threads));
#endif
    std::unique_ptr<DeadPIDBloom> dead_bloom;
    size_t dead_count = 0;
    const auto bloom_build_begin = std::chrono::steady_clock::now();
    if (bloom_consolidate) {
      std::vector<PID> dead;
      dead.reserve(n - std::min<uint64_t>(n, live_count_.load(std::memory_order_acquire)));
      for (size_t p = 0; p < n; ++p) {
        const PID pid = static_cast<PID>(p);
        if (is_hidden(pid)) dead.push_back(pid);
      }
      dead_count = dead.size();
      dead_bloom = std::make_unique<DeadPIDBloom>(dead.size());
      for (PID pid : dead) dead_bloom->insert(pid);
    }
    const auto bloom_build_end = std::chrono::steady_clock::now();
    const size_t stride = params_.maintenance_evict_stride;
    const bool in_pass_evict =
        stride != 0 && params_.write_cache && params_.cache_cap_pages < file_pages();
    const size_t rows_per_batch =
        !in_pass_evict || stride > std::numeric_limits<size_t>::max() / npp_
            ? n
            : std::max<size_t>(1, stride * npp_);
    auto row_phase_begin = std::chrono::steady_clock::now();
    auto bloom_scan_begin = row_phase_begin;
    auto bloom_scan_end = row_phase_begin;
    size_t bloom_passed_rows = 0;
    std::chrono::steady_clock::duration bloom_repair_duration{};
    if (dead_bloom == nullptr) {
      if (!in_pass_evict) {
        parallel_for_catch(0, static_cast<int64_t>(n), nt, 256, [&](int64_t ui) {
          const PID u = static_cast<PID>(ui);
          if (is_hidden(u)) {
            return;
          }
          consolidate_row(u, n, target);
        });
      } else {
        size_t batch_begin = 0;
        while (batch_begin < n) {
          const size_t batch_end = std::min(n, batch_begin + rows_per_batch);
          parallel_for_catch(static_cast<int64_t>(batch_begin),
                             static_cast<int64_t>(batch_end),
                             nt,
                             256,
                             [&](int64_t ui) {
                               const PID u = static_cast<PID>(ui);
                               if (!is_hidden(u)) consolidate_row(u, n, target);
                             });
          batch_begin = batch_end;
          const bool need_evict = note_maintenance_pool_and_test_high();
          if (need_evict) enforce_maintenance_watermark(num_threads);
        }
      }
    } else {
      std::vector<PID> rows;
      const auto scan_begin = std::chrono::steady_clock::now();
      if (dead_count != 0) rows = bloom_consolidation_rows(n, *dead_bloom, nt);
      const auto scan_end = std::chrono::steady_clock::now();
      bloom_scan_begin = scan_begin;
      bloom_scan_end = scan_end;
      bloom_passed_rows = rows.size();
      row_phase_begin = scan_end;
      stats_.bloom_scan_rows.fetch_add(dead_count == 0 ? 0 : n, std::memory_order_relaxed);
      stats_.bloom_candidate_rows.fetch_add(rows.size(), std::memory_order_relaxed);
      stats_.bloom_scan_us.fetch_add(static_cast<uint64_t>(
                                         std::chrono::duration_cast<std::chrono::microseconds>(
                                             scan_end - scan_begin)
                                             .count()),
                                     std::memory_order_relaxed);
      if (!in_pass_evict) {
        const auto repair_begin = std::chrono::steady_clock::now();
        parallel_for_catch(0, static_cast<int64_t>(rows.size()), nt, 256, [&](int64_t i) {
          consolidate_row(rows[static_cast<size_t>(i)], n, target, true);
        });
        bloom_repair_duration = std::chrono::steady_clock::now() - repair_begin;
      } else {
        std::sort(rows.begin(), rows.end());
        size_t candidate_begin = 0;
        for (size_t row_begin = 0; row_begin < n; row_begin += rows_per_batch) {
          const size_t row_end = std::min(n, row_begin + rows_per_batch);
          const size_t candidate_end = static_cast<size_t>(
              std::lower_bound(rows.begin() + static_cast<int64_t>(candidate_begin),
                               rows.end(),
                               static_cast<PID>(row_end)) -
              rows.begin());
          const auto repair_begin = std::chrono::steady_clock::now();
          parallel_for_catch(static_cast<int64_t>(candidate_begin),
                             static_cast<int64_t>(candidate_end),
                             nt,
                             256,
                             [&](int64_t i) {
                               consolidate_row(rows[static_cast<size_t>(i)], n, target, true);
                             });
          bloom_repair_duration += std::chrono::steady_clock::now() - repair_begin;
          enforce_bloom_maintenance_watermark(num_threads);
          candidate_begin = candidate_end;
        }
      }
    }
    const auto row_phase_end = std::chrono::steady_clock::now();
    if (dead_bloom != nullptr) {
      stats_.bloom_row_us.fetch_add(static_cast<uint64_t>(
                                        std::chrono::duration_cast<std::chrono::microseconds>(
                                            row_phase_end - row_phase_begin)
                                            .count()),
                                    std::memory_order_relaxed);
    }
    const auto finalize_begin = std::chrono::steady_clock::now();
    if (stride != 0) note_maintenance_pool_and_test_high();
    drain_staged_edges(num_threads);
    if (dead_bloom == nullptr)
      flush_dirty(num_threads);
    else
      merge_dirty_into_mapping(num_threads);
    if (dead_bloom != nullptr) {
      stats_.bloom_finalize_us.fetch_add(static_cast<uint64_t>(
                                             std::chrono::duration_cast<std::chrono::microseconds>(
                                                 std::chrono::steady_clock::now() - finalize_begin)
                                                 .count()),
                                         std::memory_order_relaxed);
    }
    if (reclaim_slots) {
      // Only after every live row has purged dead out-edges and that purge has
      // reached the file may a tombstoned row become a reusable free slot.
      std::vector<PID> eligible = deleted_snapshot();
      eligible.erase(std::remove_if(eligible.begin(),
                                    eligible.end(),
                                    [n](PID id) {
                                      return id >= n;
                                    }),
                     eligible.end());
      std::sort(eligible.begin(), eligible.end());
      const size_t reclaim_batch =
          !in_pass_evict || stride > std::numeric_limits<size_t>::max() / npp_
              ? std::max<size_t>(1, eligible.size())
              : std::max<size_t>(1, stride * npp_);
      for (size_t begin = 0; begin < eligible.size(); begin += reclaim_batch) {
        const size_t end = std::min(eligible.size(), begin + reclaim_batch);
        for (size_t i = begin; i < end; ++i) push_free_slot(eligible[i]);
        if (in_pass_evict) {
          if (dead_bloom == nullptr)
            enforce_maintenance_watermark(num_threads);
          else
            enforce_bloom_maintenance_watermark(num_threads);
        }
      }
      if (dead_bloom == nullptr)
        flush_dirty(num_threads);
      else
        merge_dirty_into_mapping(num_threads);
    }
    if (dead_bloom != nullptr) {
      const auto consolidate_end = std::chrono::steady_clock::now();
      const auto milliseconds = [](auto duration) {
        return std::chrono::duration<double, std::milli>(duration).count();
      };
      std::cout << "[consolidate] bloom: build="
                << milliseconds(bloom_build_end - bloom_build_begin)
                << "ms scan=" << milliseconds(bloom_scan_end - bloom_scan_begin)
                << "ms (passed=" << bloom_passed_rows << "/" << n
                << ") repair=" << milliseconds(bloom_repair_duration)
                << "ms total=" << milliseconds(consolidate_end - consolidate_begin) << "ms\n";
    }
  }

  /** Refresh a deterministic budget of live rows; phase-separated from updates. */
  void garden(size_t num_threads, const GardenParams &gp);

  /** Persist dirty pages and atomically advance the alternate A/B superblock. */
  void checkpoint();
  void checkpoint_locked();
  void finalize();

/**
 * Migration-only v1 ghost-slot heuristic.  Format-v2 update paths never use
 * this predicate; they read valid_degree from the authoritative trailer.
 */
#include "index/graph/laser/qg/detail/qg_updater_debug.hpp"

 private:
  static constexpr size_t kLockStripes = 4096;

  // C++ exceptions may not leave an OpenMP structured block. Capture the first
  // worker failure, let the implicit barrier retire every worker, then rethrow on
  // the caller thread. The three-state latch publishes first_error only after its
  // exception_ptr is fully assigned.
  template <typename Fn>
  static void parallel_for_catch(int64_t begin,
                                 int64_t end,
                                 int num_threads,
                                 int chunk_size,
                                 Fn &&fn) {
    std::atomic<uint8_t> error_state{0};  // 0 = none, 1 = claimed, 2 = ready
    std::exception_ptr first_error;
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int64_t i = begin; i < end; ++i) {
      if (error_state.load(std::memory_order_acquire) != 0) {
        continue;
      }
      try {
        fn(i);
      } catch (...) {
        uint8_t expected = 0;
        if (error_state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
          first_error = std::current_exception();
          error_state.store(2, std::memory_order_release);
        }
      }
    }
    if (error_state.load(std::memory_order_acquire) != 0) {
      std::rethrow_exception(first_error);
    }
  }

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
    std::atomic<uint64_t> patch_intents_prepared{0};
    std::atomic<uint64_t> patch_intents_applied{0};
    std::atomic<uint64_t> patch_intent_stale_fallbacks{0};
    std::atomic<uint64_t> full_recomputes{0};
    std::atomic<uint64_t> forced_links{0};
    std::atomic<uint64_t> evict_tel_samples{0};
    std::atomic<uint64_t> evict_tel_agree{0};
    std::array<std::atomic<uint64_t>, 4> evict_tel_regret{};
    std::atomic<uint64_t> evict_tel_relerr_nano{0};
    std::atomic<uint64_t> consolidated_rows{0};
    std::atomic<uint64_t> spliced_slots{0};
    std::atomic<uint64_t> ghosted_slots{0};
    std::atomic<uint64_t> bloom_scan_rows{0};
    std::atomic<uint64_t> bloom_candidate_rows{0};
    std::atomic<uint64_t> bloom_scan_us{0};
    std::atomic<uint64_t> bloom_row_us{0};
    std::atomic<uint64_t> bloom_finalize_us{0};
    std::atomic<uint64_t> freed_slots{0};
    std::atomic<uint64_t> reused_slots{0};
    std::atomic<uint64_t> gardened_rows{0};
    std::atomic<uint64_t> garden_pump_links{0};
    std::atomic<uint64_t> garden_us{0};
    std::atomic<uint64_t> garden_selected_turnover_sum{0};
    std::atomic<uint64_t> garden_selected_turnover_rows{0};
    std::atomic<uint64_t> garden_all_turnover_sum{0};
    std::atomic<uint64_t> garden_all_turnover_rows{0};
    std::atomic<uint64_t> maintenance_peak_pool_pages{0};
    std::atomic<uint64_t> maintenance_peak_overlay_pages{0};
    std::atomic<uint64_t> maintenance_page_frames{0};
    std::atomic<uint64_t> maintenance_page_frame_bytes{0};
    std::atomic<uint64_t> maintenance_last_preflight_page_frames{0};
    std::atomic<uint64_t> maintenance_last_preflight_wal_bytes{0};
    std::atomic<uint64_t> garden_skipped{0};
  };

  std::atomic<uint64_t> churn_since_garden_{0};

  struct CapturedNode {
    PID id = 0;
    float dist = 0;  // exact full-dim squared L2 to the inserted vector
    uint64_t row_generation = 0;
    std::vector<float> vec;  // raw (PCA-domain) vector only — the rest of the
                             // row is re-read fresh under the page lock
    [[nodiscard]] const float *raw() const { return vec.data(); }
  };

  PatchIntent prepare_patch_intent_from_rotated_candidate(PID target_row,
                                                          const float *target_row_snapshot,
                                                          const float *candidate_raw,
                                                          const float *candidate_rot,
                                                          float candidate_res_sqr,
                                                          PID candidate_pid,
                                                          uint64_t row_generation) {
    PatchIntent intent;
    intent.target_row = target_row;
    intent.candidate_pid = candidate_pid;
    intent.row_generation = row_generation;

    const float main_distance = space::l2_sqr(target_row_snapshot, candidate_raw, dim_);
    float target_res_sqr = 0;
    for (size_t j = 0; j < res_dim_; ++j) {
      target_res_sqr += target_row_snapshot[dim_ + j] * target_row_snapshot[dim_ + j];
    }
    intent.estimated_distance = main_distance + candidate_res_sqr + target_res_sqr;
    intent.exact_distance =
        main_distance +
        (res_dim_ > 0 ? space::l2_sqr(target_row_snapshot + dim_, candidate_raw + dim_, res_dim_)
                      : 0.0F);

    thread_local std::vector<float> target_pad;
    thread_local std::vector<float> target_rot;
    target_pad.assign(pd_, 0.0F);
    target_rot.resize(pd_);
    std::copy(target_row_snapshot, target_row_snapshot + dim_, target_pad.begin());
    qg_.rotator_.rotate(target_pad.data(), target_rot.data());

    const EdgePayload &payload =
        make_edge_payload(target_rot.data(), candidate_rot, pd_, candidate_res_sqr);
    if (!payload.degenerate) {
      intent.codes.resize(pd_ / 8);
      std::memcpy(intent.codes.data(), payload.bin.data(), intent.codes.size());
      intent.factors = {payload.triple_x, payload.factor_dq, payload.factor_vq};
    }
    stats_.patch_intents_prepared.fetch_add(1, std::memory_order_relaxed);
    return intent;
  }

  struct RoutingSnapshot {
    PID entry_point = kPidMax;
    std::vector<PID> medoids;
    std::vector<float> medoid_vectors;
  };

  /// O_DIRECT alignment unit (buffer address, file offset, and length).
  static constexpr size_t kDioAlign = 4096;

  /// Page-aligned IO buffer; O_DIRECT rejects unaligned user memory.
  struct AlignedBuf {
    AlignedBuf() = default;
    explicit AlignedBuf(size_t len) { resize(len); }
    void resize(size_t len) {
      if (len <= capacity_) return;
      char *next = nullptr;
      if (::posix_memalign(reinterpret_cast<void **>(&next), kDioAlign, len) != 0) {
        throw std::bad_alloc();
      }
      ::free(p_);
      p_ = next;
      capacity_ = len;
    }
    AlignedBuf(const AlignedBuf &) = delete;
    auto operator=(const AlignedBuf &) -> AlignedBuf & = delete;
    ~AlignedBuf() { ::free(p_); }
    [[nodiscard]] char *data() const { return p_; }

   private:
    char *p_ = nullptr;
    size_t capacity_ = 0;
  };

  /** Read-only shared file view used by the Bloom PID scan (design section 4.1:
   * the MAP_SHARED writable path is removed -- the kernel could write back an
   * uncommitted page ahead of the maintenance commit, and it forked paged vs
   * resident-arena state. All Bloom RMWs now go through modify_node_page /
   * write_at like every other write). */
  struct SharedFileMapping {
    SharedFileMapping(int fd, uint64_t offset, size_t len) : len_(len) {
      // MAP_SHARED + PROT_READ (not PROT_WRITE): a read-only shared mapping still
      // reflects the file page cache (so it observes our write_at pwrites), but we
      // never dirty it, so the kernel can never write an uncommitted page back
      // through it ahead of the maintenance commit (design section 4).
      p_ = ::mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, static_cast<off_t>(offset));
      if (p_ == MAP_FAILED) {
        p_ = nullptr;
        throw std::runtime_error("QGUpdater: mmap failed errno=" + std::to_string(errno));
      }
    }
    SharedFileMapping(const SharedFileMapping &) = delete;
    auto operator=(const SharedFileMapping &) -> SharedFileMapping & = delete;
    ~SharedFileMapping() {
      if (p_ != nullptr) ::munmap(p_, len_);
    }
    [[nodiscard]] const char *data() const { return static_cast<const char *>(p_); }

   private:
    void *p_ = nullptr;
    size_t len_ = 0;
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

  void increment_turnover(PID id) {
    if (!params_.maintain_turnover || id >= turnover_.size()) return;
    auto &counter = turnover_[id];
    uint16_t old = counter.load(std::memory_order_relaxed);
    constexpr uint16_t saturated = std::numeric_limits<uint16_t>::max();
    while (old != saturated && !counter.compare_exchange_weak(old,
                                                              static_cast<uint16_t>(old + 1),
                                                              std::memory_order_relaxed,
                                                              std::memory_order_relaxed)) {
    }
  }

  void clear_turnover(PID id) {
    if (params_.maintain_turnover && id < turnover_.size()) {
      turnover_[id].store(0, std::memory_order_relaxed);
    }
  }

  [[nodiscard]] bool is_hidden(PID id) const {
    const size_t wi = static_cast<size_t>(id) >> 6U;
    if (wi >= hidden_words_.size()) return true;
    const uint64_t mask = uint64_t{1} << (id & 63U);
    return (hidden_words_[wi].load(std::memory_order_acquire) & mask) != 0;
  }

  // Writer-private visibility (design B.3): a reserved reused row is globally hidden
  // (still dead to the concurrent query face) but VISIBLE to the single writer once it
  // has been built in THIS bundle, so robust_prune / backlink patching / search seeding
  // may treat it as a live neighbor. Outside a canonical bundle (bundle_ctx_ == nullptr)
  // this is byte-for-byte is_hidden, so every 2A/2B/maintenance path is unchanged.
  [[nodiscard]] bool writer_hidden(PID id) const {
    if (bundle_ctx_ == nullptr) {
      return is_hidden(id);
    }
    return is_hidden(id) && bundle_ctx_->revealed.count(id) == 0;
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
    while (current < end && !allocated_points_.compare_exchange_weak(current,
                                                                     end,
                                                                     std::memory_order_acq_rel,
                                                                     std::memory_order_acquire)) {
    }
    PID next = next_append_id_.load(std::memory_order_acquire);
    const PID desired = static_cast<PID>(end);
    while (next < desired && !next_append_id_.compare_exchange_weak(next,
                                                                    desired,
                                                                    std::memory_order_acq_rel,
                                                                    std::memory_order_acquire)) {
    }
  }

  void write_superblock(int slot, const QGSuperblockV2 &sb) {
    assert_no_maintenance_steal("superblock pwrite");  // wal-2c MAJOR-8
    if (slot < 0 || slot >= static_cast<int>(kQGSuperblockCopies) || !qg_superblock_valid(sb)) {
      throw std::invalid_argument("QGUpdater::write_superblock invalid slot/block");
    }
    // BLOCKER-3 (leg-7): defensive full-image gate -- never persist a superblock this build
    // cannot reopen (unsupported / self-inconsistent feature set, out-of-range activation
    // generations, or a stray non-zero 2C reserved byte). Every legitimate caller (checkpoint,
    // migrate_v1, replay_flip after its pre-write validation) builds a supported image; a
    // violation here is an internal invariant break and must fail closed rather than write a
    // base whose own next open would fail closed.
    if (!qg_superblock_supported(sb, kQgSupportedRequiredFeatures)) {
      throw std::runtime_error(
          "QGUpdater::write_superblock refusing an unsupported/full-image-invalid superblock");
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
    notify_index_fsync();
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
    std::clog << "[QGUpdater] LASER v1->v2 migration: " << n << " rows, " << pages << " pages\n";
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
    const size_t stride = params_.maintenance_evict_stride;
    const bool in_pass_evict =
        stride != 0 && params_.write_cache && params_.cache_cap_pages < file_pages();
    const size_t rows_per_batch =
        !in_pass_evict || stride > std::numeric_limits<size_t>::max() / npp_
            ? std::max<size_t>(1, free_ids.size())
            : std::max<size_t>(1, stride * npp_);
    for (size_t i = 0; i < free_ids.size(); ++i) {
      const PID id = free_ids[i];
      const PID next = head;
      {
        const std::lock_guard<std::mutex> guard(page_lock(id));
        modify_node_page(id, [&](char *page) {
          char *row = page + node_offset_in_page(id);
          const uint64_t next64 = next;
          std::memcpy(row, &next64, sizeof(next64));
          row_generations_[id].fetch_add(1, std::memory_order_release);
          QGRowTrailer trailer = row_trailer(page, id);
          trailer.valid_degree = 0;
          trailer.flags |= kQGRowTombstone | kQGRowFree;
          set_row_trailer(page, id, trailer);
          return true;
        });
      }
      head = id;
      if (in_pass_evict && ((i + 1) % rows_per_batch == 0 || i + 1 == free_ids.size())) {
        enforce_maintenance_watermark(1);
      }
    }
    if (stride != 0) note_maintenance_pool_and_test_high();
    free_list_head_.store(head, std::memory_order_release);
    free_count_.store(free_ids.size(), std::memory_order_release);
  }

  void load_v2_state(const QGSuperblockV2 &sb, bool wal_mode = false) {
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

    if (wal_mode) {
      // Recovery base load (clause C): read-only, no free-chain rebuild/flush.
      // PID reuse is disabled under enable_wal, so there is no free list; the
      // authoritative state (including hidden/deleted) is rebuilt after redo.
      free_list_head_.store(kPidMax, std::memory_order_release);
      free_count_.store(0, std::memory_order_release);
      return;
    }
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
    const int slot = select_qg_superblock_checked(header.data(), sb, kQgSupportedRequiredFeatures);
    if (slot == -2) {
      // Fail closed (codex B-2C-06): the highest-generation valid superblock is a
      // newer format this build cannot support; never downgrade to an older slot.
      throw std::runtime_error(
          "QGUpdater: superblock is a newer format this build does not support (fail closed)");
    }
    if (slot >= 0) {
      superblock_ = sb;
      active_superblock_slot_ = slot;
      load_v2_state(sb, enable_wal_);
      // WAL mode defers ALL derived-state repair (routing/free/hidden) to the single
      // convergence point rebuild_state_after_replay, run under the replaying_ guard
      // after redo (design section 2.1). Non-WAL keeps the eager repair.
      if (!enable_wal_) {
        repair_routing_roots(kPidMax);
      }
      return;
    }
    if (qg_header_has_v2_magic(header.data())) {
      throw std::runtime_error("QGUpdater: both v2 superblocks have invalid checksums");
    }
    if (enable_wal_) {
      // Migration mutates every page; a WAL lineage must start from a clean
      // checkpoint (clause C). Migrate without the WAL and checkpoint first.
      throw std::logic_error(
          "QGUpdater: enable_wal requires an already-checkpointed v2 index; run the v1->v2 "
          "migration without the WAL first");
    }
    migrate_v1(header.data());
  }

  // The current durable generation of a PID's binding. During recovery the authoritative
  // source is label_working_ (adopt + promotions); at runtime it is the published snapshot.
  // Both converge to the same {pid -> generation}, so routing repair reads the right one.
  [[nodiscard]] uint32_t current_pid_generation(PID id) const {
    if (replaying_) {
      const auto it = label_working_.find(id);
      return it != label_working_.end() ? it->second.pid_generation : 0;
    }
    const auto snap = load_label_snapshot();
    if (snap == nullptr) {
      return 0;
    }
    const auto *b = snap->find_binding(id);
    return b != nullptr ? b->pid_generation : 0;
  }

  // Runtime/replay routing repair: relocate a dead entry + drop dead/reused medoids over
  // the CURRENT committed range, seeded from the current entry. Used by tombstone() and
  // the post-replay rebuild -- byte-for-byte the historical behavior for 2A/2B (no reuse).
  void repair_routing_roots(PID newly_deleted) {
    repair_routing_roots_seeded(qg_.entry_point_,
                                committed_.load(std::memory_order_acquire),
                                nullptr,
                                newly_deleted);
  }

  // Deterministic routing repair (BLOCKER-1 / MAJOR-2). `seed_entry` is the entry the
  // relocation starts from; a canonical clean publish passes the LAST-CHECKPOINT entry
  // (superblock_.entry_point) so it reproduces recovery's single relocate-from-base
  // (recovery seeds from the same base entry and repairs once at the end). `scan_n` is the
  // range to relocate over (the NEW committed watermark for a bundle -- NOT the stale old
  // one, which is the B1 divergence). `force_live` (may be null) marks reserved reused rows
  // as live even while their global hidden bit is still set (their trailers are already
  // live in the cache). A medoid is kept ONLY if it is live AND never reused (generation 0):
  // a reused medoid's immutable sidecar vector is stale, so recovery (which reloads it) and
  // the clean path (which dropped it at tombstone) both drop it -> they converge (MAJOR-2).
  void repair_routing_roots_seeded(PID seed_entry,
                                   uint64_t scan_n,
                                   const std::unordered_set<PID> *force_live,
                                   PID newly_deleted = kPidMax) {
    const auto live = [&](PID id) -> bool {
      if (id == newly_deleted) return false;
      if (force_live != nullptr && force_live->count(id) != 0) return true;
      return !is_hidden(id);
    };
    bool changed = false;
    for (size_t i = qg_.medoids_.size(); i > 0; --i) {
      const size_t idx = i - 1;
      const PID medoid = qg_.medoids_[idx];
      if (live(medoid) && current_pid_generation(medoid) == 0) continue;
      qg_.medoids_.erase(qg_.medoids_.begin() + static_cast<int64_t>(idx));
      const auto first = qg_.medoids_vector_.begin() + static_cast<int64_t>(idx * full_dim_);
      qg_.medoids_vector_.erase(first, first + static_cast<int64_t>(full_dim_));
      changed = true;
    }
    PID new_entry = seed_entry;
    if (!live(seed_entry) && scan_n != 0) {
      const size_t start = (static_cast<size_t>(seed_entry) + 1) % scan_n;
      for (size_t offset = 0; offset < scan_n; ++offset) {
        const PID candidate = static_cast<PID>((start + offset) % scan_n);
        if (live(candidate)) {
          new_entry = candidate;
          break;
        }
      }
    }
    if (qg_.entry_point_ != new_entry) {
      qg_.entry_point_ = new_entry;
      changed = true;
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
        if (free_list_head_.compare_exchange_weak(expected,
                                                  id,
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_acquire)) {
          break;
        }
      }
      row_generations_[id].fetch_add(1, std::memory_order_release);
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
      if ((trailer.flags & (kQGRowTombstone | kQGRowFree)) != (kQGRowTombstone | kQGRowFree)) {
        throw std::runtime_error("QGUpdater: free-list head lacks free/tombstone flags");
      }
      uint64_t next64 = kPidMax;
      std::memcpy(&next64, page.data() + node_offset_in_page(head), sizeof(next64));
      if (next64 != kPidMax && next64 >= allocated_points()) {
        throw std::runtime_error("QGUpdater: corrupt next_free PID");
      }
      PID expected = head;
      const PID next = static_cast<PID>(next64);
      if (!free_list_head_.compare_exchange_strong(expected,
                                                   next,
                                                   std::memory_order_acq_rel,
                                                   std::memory_order_acquire)) {
        continue;
      }
      free_count_.fetch_sub(1, std::memory_order_acq_rel);
      return head;
    }
  }

  // Reserve n PIDs for a canonical reuse bundle (design 3.6 / codex B.2): pop the
  // canonical (ascending) free chain head for the reuse prefix, dense-append the rest.
  // Publishes bundle_state_/reservation_count_ BEFORE the first pop so a concurrent
  // checkpoint is refused even for an all-reuse reservation whose HWM does not move
  // (B-2C-06/B.6). Any failure after the state is published keeps the state and poisons
  // -- reopen rebuilds the free chain from the on-disk FREE trailers (the popped PIDs
  // are still FREE on disk, so R0 recovers the old free set). Dormant until the canonical
  // writer (step 5) calls it under checkpoint_mutex_.
  PhysicalBundleResult reserve_bundle_pids(size_t n, bool allow_reuse) {
    if (bundle_state_ != BundleState::kIdle) {
      throw std::logic_error("reserve_bundle_pids: a bundle reservation is already active");
    }
    if (!free_chain_rebuild_complete_) {
      throw std::logic_error("reserve_bundle_pids: the free chain is not rebuilt yet");
    }
    const uint64_t old = committed_.load(std::memory_order_acquire);
    if (old != allocated_points()) {
      throw std::logic_error("reserve_bundle_pids requires allocated == committed");
    }
    if (has_staged_edges()) {
      throw std::logic_error("reserve_bundle_pids requires no staged backlinks");
    }
    const uint64_t free_now = free_count_.load(std::memory_order_acquire);
    const size_t reuse_n = allow_reuse ? static_cast<size_t>((std::min<uint64_t>)(n, free_now)) : 0;
    const size_t append_n = n - reuse_n;
    // Capacity check BEFORE the first pop (design B.2): a dense-append overflow is a
    // caller error, never a mid-reservation poison.
    const uint64_t new_hwm = old + append_n;
    if (new_hwm > static_cast<uint64_t>(kPidMax) || new_hwm > row_generations_.size()) {
      throw std::invalid_argument("reserve_bundle_pids: PID capacity exceeded");
    }
    // Publish the reservation state before the first pop (checkpoint admission B.6).
    // BLOCKER-3: once the state is published, ANY exception (even a bad_alloc that does not
    // itself poison) must poison the handle -- otherwise bundle_state_ latches non-idle and
    // checkpoint / the next bundle are blocked forever with no fail-closed signal. Reopen
    // rebuilds the free chain from the on-disk FREE trailers (the popped PIDs are still FREE
    // on disk), so this is a clean roll-back-to-S_old on the next open.
    bundle_state_ = BundleState::kReserving;
    reservation_count_ = n;
    try {
      PhysicalBundleResult out;
      out.old_hwm = old;
      out.rows.reserve(n);
      const auto snap = load_label_snapshot();
      std::unordered_set<uint64_t> popped;
      popped.reserve(reuse_n * 2);
      for (size_t i = 0; i < reuse_n; ++i) {
        const PID pid = pop_free_slot();
        if (pid == kPidMax || static_cast<uint64_t>(pid) >= old ||
            !popped.insert(static_cast<uint64_t>(pid)).second) {
          poison("reserve_bundle_pids: corrupt canonical free-list reservation");
        }
        const auto *b = snap->find_binding(pid);
        const uint32_t old_gen = b != nullptr ? b->pid_generation : 0;
        if (old_gen == (std::numeric_limits<uint32_t>::max)()) {
          poison("reserve_bundle_pids: PID generation would wrap (UINT32_MAX)");
        }
        out.rows.push_back(PidToken{pid, old_gen + 1});
      }
      for (size_t i = 0; i < append_n; ++i) {
        out.rows.push_back(PidToken{static_cast<PID>(old + i), 0});
      }
      out.new_hwm = new_hwm;
      note_allocated(new_hwm);
      bundle_state_ = BundleState::kBuilding;
      return out;
    } catch (...) {
      // BLOCKER-2/3: reservation state is published (kReserving); a non-std::exception or
      // an allocation failure past this point must still fail the handle closed.
      poison_current_exception("reserve_bundle_pids failed after publishing reservation state");
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
    [[nodiscard]] size_t total_pages() const {
      return total_pages_.load(std::memory_order_relaxed);
    }
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
    std::unique_ptr<PatchIntent> intent;
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
    if (pi >= page_versions_.size()) {
      throw std::runtime_error("QGUpdater: id exceeds max_points capacity");
    }
    // 2C maintenance transaction (design section 1.2): route the RMW into the
    // PRIVATE overlay. No inline WAL log, no index write, no seqlock bump -- a
    // concurrent search keeps reading the committed state (disk) until the
    // END-durable install. The kind=1 after-image is appended at spill / finalize
    // (serial maintenance lane). unordered_map reference stability keeps the page
    // pointer valid across nested overlay materializations inside fn.
    if (enable_wal_ && !replaying_ && maintenance_active_) {
      char *page = maint_overlay_page(pi);
      const bool changed = fn(page);
      if (changed) {
        maint_dirty_.insert(pi);
        maint_resident_dirty_.insert(pi);
        stats_.logical_row_writes++;
        maint_spill_over_cap(pi);  // wal-2c BLOCKER-1: pin the page just modified
      }
      return changed;
    }
    // 2C canonical reuse bundle (design B.4): route the RMW into the writer-private
    // overlay. No kind=1 log here (finalize appends the final page after-image) and no
    // shared-cache install (that happens after the kind=8 commit point) -- a concurrent
    // search keeps reading committed state until then.
    if (enable_wal_ && !replaying_ && bundle_ctx_ != nullptr) {
      char *page = bundle_overlay_page(pi);
      const bool changed = fn(page);
      if (changed) {
        bundle_ctx_->dirty.insert(pi);
        stats_.logical_row_writes++;
        bundle_spill_over_cap(pi);  // pin the page just modified
      }
      return changed;
    }
    // No-steal RMW (clause B): with the WAL live, the callback runs on a private
    // scratch page, its full-page after-image is appended, and only then is the
    // page installed (cache) / written (no cache). On WAL append failure the
    // writer is poisoned and nothing is installed (fail closed).
    if (enable_wal_ && !replaying_) {
      ensure_writable();
      AlignedBuf scratch(page_size_);
      if (!params_.write_cache) {
        read_at(page_offset(id), scratch.data(), page_size_);
        stats_.patch_page_reads++;
        if (!fn(scratch.data())) {
          return false;
        }
        log_page_after_image(id, scratch.data());
        stats_.logical_row_writes++;
        write_node_page(id, scratch.data());  // force_wal() then pwrite
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
      {
        const std::lock_guard<std::mutex> bytes_guard(cached->bytes_mutex);
        std::memcpy(scratch.data(), cached->bytes.data(), page_size_);
      }
      if (!fn(scratch.data())) {
        return false;
      }
      log_page_after_image(id, scratch.data());  // appended before the install
      {
        const std::lock_guard<std::mutex> bytes_guard(cached->bytes_mutex);
        // No exception-capable work may be added between these bumps: std::memcpy
        // is the sole operation and cannot throw a C++ exception.
        page_versions_[pi].fetch_add(1, std::memory_order_acq_rel);
        std::memcpy(cached->bytes.data(), scratch.data(), page_size_);
        page_versions_[pi].fetch_add(1, std::memory_order_release);
      }
      {
        const std::lock_guard<std::mutex> cache_guard(shard.mutex);
        cached->dirty = true;
      }
      stats_.logical_row_writes++;
      return true;
    }
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
    bool changed = false;
    {
      const std::lock_guard<std::mutex> bytes_guard(cached->bytes_mutex);
      write_page_versioned(pi, [&] {
        changed = fn(cached->bytes.data());
      });
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

  /**
   * Bloom path RMW (design section 4.1): the writable MAP_SHARED direct-write is
   * removed -- every Bloom candidate RMW now goes through the ordinary
   * modify_node_page / write_at path (admitting the page to the write pool),
   * which keeps paged and resident-arena state identical, respects the no-steal
   * WAL ordering, and honors the page seqlock. The caller holds page_lock(id).
   */
  template <typename Fn>
  bool modify_bloom_node_page(PID id, Fn &&fn) {
    return modify_node_page(id, std::forward<Fn>(fn));
  }

  [[nodiscard]] const float *cached_raw(PID id) {
    if (enable_wal_ && !replaying_ && maintenance_active_) {
      return reinterpret_cast<const float *>(maint_overlay_page(page_index(id)) +
                                             node_offset_in_page(id));
    }
    if (enable_wal_ && !replaying_ && bundle_ctx_ != nullptr) {
      return reinterpret_cast<const float *>(bundle_overlay_page(page_index(id)) +
                                             node_offset_in_page(id));
    }
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
    if (enable_wal_ && !replaying_ && maintenance_active_) {
      std::memcpy(out, maint_overlay_page(page_index(id)), page_size_);
      return;
    }
    if (enable_wal_ && !replaying_ && bundle_ctx_ != nullptr) {
      std::memcpy(out, bundle_overlay_page(page_index(id)), page_size_);
      return;
    }
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
    if (enable_wal_ && !replaying_) {
      force_wal();  // the page after-image must be durable before the index pwrite
    }
    write_page_versioned(pi, [&] {
      write_at(page_offset(id), page, page_size_);
    });
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
  void read_node_page(PID id,
                      char *buf,
                      bool query_read = false,
                      uint64_t *row_generation = nullptr) {
    const size_t pi = page_index(id);
    if (pi >= page_versions_.size()) {
      throw std::runtime_error("QGUpdater: id exceeds max_points capacity");
    }
    if (row_generation != nullptr && id >= row_generations_.size()) {
      throw std::runtime_error("QGUpdater: row generation exceeds max_points capacity");
    }
    // 2C canonical reuse bundle (design B.3): the single writer's own dependency reads
    // (search_for_insert, query_read=false) see its private overlay for pages it has
    // already TOUCHED in this bundle, so a revealed reused row reads its NEW incarnation
    // bytes. Untouched pages fall through to the committed read below, and a concurrent
    // public search (query_read=true, another thread) never takes this branch -- it only
    // ever observes committed state. No seqlock: the writer owns the overlay exclusively.
    if (!query_read && enable_wal_ && !replaying_ && bundle_ctx_ != nullptr &&
        (bundle_ctx_->pages.count(pi) != 0 || bundle_ctx_->spilled.count(pi) != 0)) {
      std::memcpy(buf, bundle_overlay_page(pi), page_size_);
      if (row_generation != nullptr) {
        *row_generation = row_generations_[id].load(std::memory_order_acquire);
      }
      return;
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
      const uint64_t row_generation_before =
          row_generation == nullptr ? 0 : row_generations_[id].load(std::memory_order_acquire);
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
        const uint32_t locked_version = page_versions_[pi].load(std::memory_order_acquire);
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
        if (row_generation == nullptr) {
          return;
        }
        const uint64_t row_generation_after = row_generations_[id].load(std::memory_order_acquire);
        if (row_generation_before == row_generation_after) {
          *row_generation = row_generation_after;
          return;
        }
      }
      note_retry();
    }
  }

  /** Inspect only one row's authoritative PID prefix in an already-read page. */
  [[nodiscard]] bool row_has_dead_neighbor_bloom(PID id,
                                                 const DeadPIDBloom &bloom,
                                                 const char *page) const {
    const char *row = page + node_offset_in_page(id);
    const auto *ids = reinterpret_cast<const PID *>(row + neighbor_off_bytes());
    const size_t degree = row_trailer(page, id).valid_degree;
    for (size_t slot = 0; slot < degree; ++slot) {
      // Confirm Bloom hits against the exact hidden bitmap. At a ~1% per-element
      // false-positive rate, an unconfirmed 32-neighbor row test would otherwise
      // send roughly a quarter of clean rows through the expensive write path.
      if (bloom.maybe_contains(ids[slot]) && is_hidden(ids[slot])) return true;
    }
    return false;
  }

  /**
   * PID-only scan over physical pages. The file mapping supplies clean pages;
   * a pointer snapshot of resident write-cache pages overlays newer dirty
   * bytes without copying or admitting scan misses into that cache. Phase
   * separation guarantees the pointer snapshot stays stable until this
   * read-only phase completes; concurrent searches only read these pages.
   */
  [[nodiscard]] std::vector<PID> bloom_consolidation_rows(size_t n,
                                                          const DeadPIDBloom &bloom,
                                                          int num_threads) {
    if (n == 0) return {};
    const size_t page_count = (n + npp_ - 1) / npp_;
    const uint64_t required_file_size = kSectorLen + file_pages() * page_size_;
    struct stat file_stat{};
    if (::fstat(fd_, &file_stat) != 0) {
      throw std::runtime_error("QGUpdater: fstat failed errno=" + std::to_string(errno));
    }
    if (static_cast<uint64_t>(file_stat.st_size) < required_file_size) {
      assert_no_maintenance_steal("index ftruncate (bloom-scan grow)");  // wal-2c MAJOR-8
      if (::ftruncate(fd_, static_cast<off_t>(required_file_size)) != 0) {
        throw std::runtime_error("QGUpdater: ftruncate before mmap failed errno=" +
                                 std::to_string(errno));
      }
    }
    if (pid_scan_mapping_ == nullptr) {
      pid_scan_mapping_ =
          std::make_unique<SharedFileMapping>(fd_, kSectorLen, page_versions_.size() * page_size_);
    }

    std::vector<const char *> cached_pages(page_count, nullptr);
    if (params_.write_cache) {
      for (size_t si = 0; si < PageWriteCache::kShards; ++si) {
        auto &shard = write_cache_.shard(si);
        const std::lock_guard<std::mutex> guard(shard.mutex);
        for (const auto &[pi, page] : shard.pages) {
          if (pi < page_count) cached_pages[pi] = page->bytes.data();
        }
      }
    }

#if defined(__SANITIZE_THREAD__)
    (void)num_threads;
    const int nt = 1;
#else
    const int nt = std::max(1, num_threads);
#endif
    std::vector<std::vector<PID>> thread_rows(static_cast<size_t>(nt));
    for (auto &local : thread_rows) {
      local.reserve(std::max<size_t>(16, n / static_cast<size_t>(nt) / 16));
    }
    parallel_for_catch(0, static_cast<int64_t>(page_count), nt, 1, [&](int64_t raw_pi) {
      auto &local = thread_rows[static_cast<size_t>(omp_get_thread_num())];
      const size_t pi = static_cast<size_t>(raw_pi);
      const char *page = cached_pages[pi];
      if (page == nullptr) page = pid_scan_mapping_->data() + pi * page_size_;
      const size_t row_begin = pi * npp_;
      const size_t row_end = std::min(n, row_begin + npp_);
      for (size_t raw_id = row_begin; raw_id < row_end; ++raw_id) {
        const PID id = static_cast<PID>(raw_id);
        if (!is_hidden(id) && row_has_dead_neighbor_bloom(id, bloom, page)) {
          local.push_back(id);
        }
      }
    });

    size_t total = 0;
    for (const auto &local : thread_rows) total += local.size();
    std::vector<PID> rows;
    rows.reserve(total);
    for (auto &local : thread_rows) {
      rows.insert(rows.end(), local.begin(), local.end());
    }
    return rows;
  }

  /**
   * Return a stable dependency-page view during one Bloom maintenance batch.
   * Cache insertion may run concurrently, so lookup takes the shard latch;
   * eviction happens only at the batch barrier. Consolidation never mutates a
   * hidden row, and raw-vector bytes in live rows are immutable.
   */
  [[nodiscard]] const char *bloom_dependency_page(PID id) {
    if (enable_wal_ && !replaying_ && maintenance_active_) {
      return maint_overlay_page(page_index(id));
    }
    const size_t pi = page_index(id);
    if (params_.write_cache) {
      auto &shard = write_cache_.shard(pi);
      const std::lock_guard<std::mutex> guard(shard.mutex);
      const auto it = shard.pages.find(pi);
      if (it != shard.pages.end()) return it->second->bytes.data();
    }
    return pid_scan_mapping_->data() + pi * page_size_;
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

  // wal-2c MAJOR-8: unified no-steal audit for EVERY durable index-mutating primitive
  // (pwrite via write_at, ftruncate, superblock pwrite). Between a durable maintenance BEGIN
  // and its durable END the only durable store is the private overlay -> op-WAL; any index /
  // arena / file-length mutation in that window is a steal-before-commit -> poison (fail-
  // closed). Recovery (replaying_) and the pre-BEGIN activation checkpoint are outside the
  // window, so the guard is a no-op on every legitimate path.
  void assert_no_maintenance_steal(const char *what) {
    if (maint_in_build_phase_ && !replaying_) {
      poison(std::string("maintenance ") + what +
             " before the END commit point (no-steal violation)");
    }
  }

  void write_at(uint64_t off, const char *buf, size_t len) {
    assert_no_maintenance_steal("index/arena pwrite");
    if (params_.before_index_write_hook) {
      params_.before_index_write_hook(off, len);
    }
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
    qg_.arena_mirror_write(off, buf, len);
    stats_.page_writes++;
  }

  // Close every page-version pair even when a cached RMW callback, pwrite, or
  // post-write failpoint throws. Poison is published first, so a reader released
  // by the even store observes recovery-required at its exit gate instead of
  // serving partial state.
  template <typename Fn>
  void write_page_versioned(size_t pi, Fn &&write) {
    page_versions_[pi].fetch_add(1, std::memory_order_acq_rel);  // -> odd
    try {
      write();
    } catch (...) {
      poison_latch();
      page_versions_[pi].fetch_add(1, std::memory_order_release);  // -> even
      throw;
    }
    page_versions_[pi].fetch_add(1, std::memory_order_release);  // -> even
  }

  // ===================== 2C maintenance overlay (design section 1.2) ===========
  // Materialize page pi in the private maintenance overlay: reuse a resident copy,
  // reload a spilled one from its kind=1 frame, or copy the committed image from
  // disk on first touch. Single-threaded within an epoch. unordered_map guarantees
  // reference stability, so a char* returned here stays valid across later inserts
  // (the cap/release helpers erase only after the caller finishes using it).
  void note_maintenance_overlay_resident() noexcept {
    const uint64_t pages = maint_pages_.size();
    uint64_t peak = stats_.maintenance_peak_overlay_pages.load(std::memory_order_relaxed);
    while (
        peak < pages &&
        !stats_.maintenance_peak_overlay_pages.compare_exchange_weak(peak,
                                                                     pages,
                                                                     std::memory_order_relaxed,
                                                                     std::memory_order_relaxed)) {
    }
  }

  char *maint_overlay_page(size_t pi) {
    auto it = maint_pages_.find(pi);
    if (it != maint_pages_.end()) {
      return it->second.data();
    }
    std::vector<char> bytes(page_size_);
    auto sp = maint_spilled_.find(pi);
    if (sp != maint_spilled_.end()) {
      const auto frame = alaya::wal::WalFile::read_frame(op_wal_->path(), sp->second);
      SegmentOp op;
      try {
        op = decode_segment_op(frame.payload);
      } catch (const std::exception &error) {
        poison(std::string("maintenance overlay reload decode failed: ") + error.what());
      }
      if (op.kind != SegmentOpKind::row_patch || op.bytes.size() != page_size_) {
        poison("maintenance overlay reload got a non-page frame");
      }
      std::memcpy(bytes.data(), op.bytes.data(), page_size_);
    } else {
      read_at(kSectorLen + pi * page_size_, bytes.data(), page_size_);
    }
    auto [ins, unused] = maint_pages_.emplace(pi, std::move(bytes));
    (void)unused;
    note_maintenance_overlay_resident();
    return ins->second.data();
  }

  // Append one kind=1 whole-page after-image for the overlay page (maintenance
  // lane; serial). Returns its FrameLocation so a spilled page can be reloaded.
  alaya::wal::FrameLocation maint_log_page(size_t pi,
                                           const char *bytes,
                                           alaya::wal::WalFile::Sync sync) {
    const uint64_t offset = kSectorLen + pi * page_size_;
    const auto first_pid = static_cast<uint64_t>(pi * npp_);
    auto payload =
        encode_row_patch(segment_uid_,
                         superblock_.generation,
                         first_pid,
                         offset,
                         std::span<const std::byte>(reinterpret_cast<const std::byte *>(bytes),
                                                    page_size_));
    try {
      const auto location =
          op_wal_->append(kSegmentOpRecordType, 0, ++wal_op_id_, 0, payload, sync);
      stats_.maintenance_page_frames.fetch_add(1, std::memory_order_relaxed);
      stats_.maintenance_page_frame_bytes.fetch_add(location.size, std::memory_order_relaxed);
      return location;
    } catch (const std::exception &error) {
      poison(std::string("maintenance WAL append failed: ") + error.what());
    }
  }

  // Spill resident overlay pages (logging each latest kind=1, flushed so read_frame
  // can reload it) until the overlay is back under cap. Called only after a modify
  // fn returns, so no live char* into maint_pages_ can dangle.
  // Spill resident overlay pages back under cap (see below). `pin_pi` is the page whose RMW
  // triggered this call: it is NEVER a spill victim (wal-2c BLOCKER-1). Otherwise the eviction
  // scan (unordered_map order) could pick the page currently being modified, forcing the very
  // next row on it to reload from the WAL and re-log -- an unbounded reload/respill amplification
  // that voids the BEGIN-time statvfs bound (repair_pages + reclaim_pages) and can ENOSPC-poison
  // a committed-past-BEGIN transaction. Pinning the active page removes that pathology; only one
  // page is ever pinned, so cap (>= 1) is always reachable.
  void maint_spill_over_cap(size_t pin_pi) {
    const size_t cap = std::max<size_t>(1, params_.cache_cap_pages);
    // A page clean since its latest spill can be dropped for free. This includes
    // dependency-only disk pages and dirty-history pages reloaded from latest_spill.
    for (auto it = maint_pages_.begin(); it != maint_pages_.end() && maint_pages_.size() > cap;) {
      if (it->first != pin_pi && maint_resident_dirty_.count(it->first) == 0) {
        it = maint_pages_.erase(it);
      } else {
        ++it;
      }
    }
    // Then spill dirty pages: log the latest kind=1 (flushed so read_frame can
    // reload it), record its location, and release the page memory.
    for (auto it = maint_pages_.begin(); it != maint_pages_.end() && maint_pages_.size() > cap;) {
      const size_t pi = it->first;
      if (pi == pin_pi) {  // never spill the page whose RMW is in progress
        ++it;
        continue;
      }
      wal_failpoint(SegmentOpFailPoint::after_consolidate_overlay_modify_before_spill);  // C4
      const auto loc = maint_log_page(pi, it->second.data(), alaya::wal::WalFile::Sync::flush);
      maint_spilled_[pi] = loc;
      maint_resident_dirty_.erase(pi);
      wal_failpoint(SegmentOpFailPoint::after_consolidate_spill_flush);  // C5
      it = maint_pages_.erase(it);
    }
  }

  // A dependency-only page has no reason to occupy the overlay after its last
  // read. Drop a clean page immediately; if a prior maintenance phase dirtied it,
  // enforce the ordinary cap without pinning it so its latest image can spill.
  void maint_release_dependency_page(size_t pi) {
    auto it = maint_pages_.find(pi);
    if (it != maint_pages_.end() && maint_resident_dirty_.count(pi) == 0) {
      maint_pages_.erase(it);
    }
    maint_spill_over_cap((std::numeric_limits<size_t>::max)());
  }

  // END-durable install (design section 1.2 step 6): write every touched page's
  // final image to the index under the page seqlock (odd -> write_at/arena mirror
  // -> even) so a concurrent search never copies a half-installed page. Honors the
  // after_consolidate_install_page failpoint after each page.
  void maint_install_all() {
    std::vector<size_t> pages(maint_dirty_.begin(), maint_dirty_.end());
    std::sort(pages.begin(), pages.end());
    for (size_t pi : pages) {
      std::vector<char> reload;
      const char *bytes = nullptr;
      auto it = maint_pages_.find(pi);
      if (it != maint_pages_.end()) {
        bytes = it->second.data();
      } else {
        const auto spilled = maint_spilled_.find(pi);
        if (spilled == maint_spilled_.end()) {
          poison("maintenance dirty page has neither resident nor spilled bytes");
        }
        const auto frame = alaya::wal::WalFile::read_frame(op_wal_->path(), spilled->second);
        const auto op = decode_segment_op(frame.payload);
        if (op.kind != SegmentOpKind::row_patch || op.bytes.size() != page_size_) {
          poison("maintenance install reload got a non-page frame");
        }
        reload.assign(reinterpret_cast<const char *>(op.bytes.data()),
                      reinterpret_cast<const char *>(op.bytes.data()) + page_size_);
        bytes = reload.data();
      }
      const PID first_pid = static_cast<PID>(pi * npp_);
      const std::lock_guard<std::mutex> guard(page_lock(first_pid));
      write_page_versioned(pi, [&] {
        wal_failpoint(SegmentOpFailPoint::after_consolidate_install_version_odd);
        write_at(kSectorLen + pi * page_size_, bytes, page_size_);
        wal_failpoint(SegmentOpFailPoint::after_consolidate_install_write_before_even);
      });
      wal_failpoint(SegmentOpFailPoint::after_consolidate_install_page);
    }
  }

  void maint_reset_overlay() {
    maint_pages_.clear();
    maint_spilled_.clear();
    maint_dirty_.clear();
    maint_resident_dirty_.clear();
    maint_local_free_head_ = kPidMax;
    maint_local_free_count_ = 0;
  }

  // ===================== 2C canonical reuse bundle overlay (design B.3/B.4) ======
  // The bundle overlay mirrors the maintenance overlay (private resident pages + WAL
  // spill), but installs into the shared write cache (not the index file) after the
  // kind=8 commit point, exactly like an ordinary insert batch. Single-writer only.
  // Materialize page pi in the bundle overlay: reuse a resident copy, reload a spilled
  // one from its kind=1 frame, or copy the committed image (disk / shared write cache)
  // on first touch. unordered_map reference stability keeps a returned char* valid
  // across later inserts (only bundle_spill_over_cap, after the caller's fn, erases).
  char *bundle_overlay_page(size_t pi) {
    auto it = bundle_ctx_->pages.find(pi);
    if (it != bundle_ctx_->pages.end()) {
      return it->second.data();
    }
    std::vector<char> bytes(page_size_);
    auto sp = bundle_ctx_->spilled.find(pi);
    if (sp != bundle_ctx_->spilled.end()) {
      const auto frame = alaya::wal::WalFile::read_frame(op_wal_->path(), sp->second);
      SegmentOp op;
      try {
        op = decode_segment_op(frame.payload);
      } catch (const std::exception &error) {
        poison(std::string("bundle overlay reload decode failed: ") + error.what());
      }
      if (op.kind != SegmentOpKind::row_patch || op.bytes.size() != page_size_) {
        poison("bundle overlay reload got a non-page frame");
      }
      std::memcpy(bytes.data(), op.bytes.data(), page_size_);
      bundle_ctx_->spilled.erase(sp);
    } else {
      read_committed_page(pi, bytes.data());  // committed disk OR resident shared-cache image
    }
    auto [ins, unused] = bundle_ctx_->pages.emplace(pi, std::move(bytes));
    (void)unused;
    return ins->second.data();
  }

  // Copy page pi's currently-committed image (the shared write-cache copy if resident,
  // else the on-disk page). Used to seed the bundle overlay and to log a reused page's
  // pre-transaction FREE preimage.
  void read_committed_page(size_t pi, char *out) {
    if (params_.write_cache) {
      auto &shard = write_cache_.shard(pi);
      const std::lock_guard<std::mutex> guard(shard.mutex);
      const auto it = shard.pages.find(pi);
      if (it != shard.pages.end()) {
        const std::lock_guard<std::mutex> bytes_guard(it->second->bytes_mutex);
        std::memcpy(out, it->second->bytes.data(), page_size_);
        return;
      }
    }
    read_at(kSectorLen + pi * page_size_, out, page_size_);
  }

  // Append one kind=1 whole-page after-image for a bundle overlay page. Returns its
  // FrameLocation so a spilled page can be reloaded. Buffered by default; the kind=8
  // force makes the whole group (kind=7 + preimages + finals) durable at once.
  alaya::wal::FrameLocation bundle_log_page(size_t pi,
                                            const char *bytes,
                                            alaya::wal::WalFile::Sync sync) {
    const uint64_t offset = kSectorLen + pi * page_size_;
    const auto first_pid = static_cast<uint64_t>(pi * npp_);
    auto payload =
        encode_row_patch(segment_uid_,
                         superblock_.generation,
                         first_pid,
                         offset,
                         std::span<const std::byte>(reinterpret_cast<const std::byte *>(bytes),
                                                    page_size_));
    try {
      return op_wal_->append(kSegmentOpRecordType, 0, ++wal_op_id_, 0, payload, sync);
    } catch (const std::exception &error) {
      poison(std::string("bundle WAL append failed: ") + error.what());
    }
  }

  // Spill resident overlay pages (logging each latest dirty kind=1, flushed so
  // read_frame can reload it) until back under cap. `pin_pi` (the page whose RMW is in
  // progress) is never a victim (BLOCKER-1 pin). Clean dependency pages drop for free.
  void bundle_spill_over_cap(size_t pin_pi) {
    const size_t cap = std::max<size_t>(1, params_.cache_cap_pages);
    for (auto it = bundle_ctx_->pages.begin();
         it != bundle_ctx_->pages.end() && bundle_ctx_->pages.size() > cap;) {
      if (it->first != pin_pi && bundle_ctx_->dirty.count(it->first) == 0) {
        it = bundle_ctx_->pages.erase(it);
      } else {
        ++it;
      }
    }
    for (auto it = bundle_ctx_->pages.begin();
         it != bundle_ctx_->pages.end() && bundle_ctx_->pages.size() > cap;) {
      if (it->first == pin_pi) {
        ++it;
        continue;
      }
      // Before the FIRST flush-spill, force the lane-opening kind=7 binds + FREE preimages
      // durable. A spill is flushed (Sync::flush -> OS page cache, survives a process crash),
      // but the kind=7 are only buffered (userspace, lost on SIGKILL). Without this force a
      // surviving spilled kind=1 could ORPHAN a lost kind=7, and recovery would apply that
      // canonical page as a legacy row_patch (line ~replay_row_patch), wrongly installing a torn
      // bundle. Forcing here makes the lane durable so a surviving spill is staged inside it
      // and discarded at EOF (S_old) -- the same guarantee consolidate gets by forcing BEGIN
      // before its spills. (The net WAL bytes are unchanged; only durability timing moves.)
      if (!bundle_ctx_->binds_durable) {
        force_wal();
        bundle_ctx_->binds_durable = true;
      }
      const auto loc =
          bundle_log_page(it->first, it->second.data(), alaya::wal::WalFile::Sync::flush);
      bundle_ctx_->spilled[it->first] = loc;
      it = bundle_ctx_->pages.erase(it);
    }
  }

  // Read bundle overlay page `pi` into a caller scratch buffer WITHOUT residenting it
  // (MAJOR-3): a spilled page is re-read from its kind=1 frame but NOT re-added to the
  // resident set, so a self-check / validation pass over N pages stays O(page_size), not
  // O(N x page_size) -- otherwise a large bundle under a tiny cache cap would bad_alloc.
  void bundle_read_page_scratch(size_t pi, char *out) {
    const auto it = bundle_ctx_->pages.find(pi);
    if (it != bundle_ctx_->pages.end()) {
      std::memcpy(out, it->second.data(), page_size_);
      return;
    }
    const auto sp = bundle_ctx_->spilled.find(pi);
    if (sp != bundle_ctx_->spilled.end()) {
      const auto frame = alaya::wal::WalFile::read_frame(op_wal_->path(), sp->second);
      SegmentOp op;
      try {
        op = decode_segment_op(frame.payload);
      } catch (const std::exception &error) {
        poison(std::string("bundle scratch reload decode failed: ") + error.what());
      }
      if (op.kind != SegmentOpKind::row_patch || op.bytes.size() != page_size_) {
        poison("bundle scratch reload got a non-page frame");
      }
      std::memcpy(out, op.bytes.data(), page_size_);
      return;
    }
    read_committed_page(pi, out);
  }

  // Finalize (design B.4): log every remaining resident DIRTY page as its final kind=1
  // (buffered; the kind=8 force makes the group -- including earlier spills -- durable).
  // Spilled pages already carry a final kind=1 from their spill, so only resident dirty
  // pages need a fresh frame here.
  void bundle_finalize_pages() {
    bool first_final = true;
    for (auto &[pi, bytes] : bundle_ctx_->pages) {
      if (bundle_ctx_->dirty.count(pi) != 0) {
        bundle_log_page(pi, bytes.data(), alaya::wal::WalFile::Sync::buffered);
        if (first_final) {
          first_final = false;
          wal_failpoint(SegmentOpFailPoint::after_reuse_partial_final_page);  // R3
        }
      }
    }
  }

  // Post-kind=8 install (design B.4): copy every dirty overlay page's final image into
  // the shared write cache under the page seqlock (odd -> memcpy -> even) + mark dirty,
  // so a concurrent search reads a whole page. Same install shape the enable_wal RMW
  // path uses; physical writeback happens later on eviction / checkpoint.
  void bundle_install_to_cache() {
    std::vector<size_t> pages(bundle_ctx_->dirty.begin(), bundle_ctx_->dirty.end());
    std::sort(pages.begin(), pages.end());
    for (size_t pi : pages) {
      std::vector<char> reload;
      const char *bytes = nullptr;
      auto it = bundle_ctx_->pages.find(pi);
      if (it != bundle_ctx_->pages.end()) {
        bytes = it->second.data();
      } else {
        const auto frame =
            alaya::wal::WalFile::read_frame(op_wal_->path(), bundle_ctx_->spilled[pi]);
        const auto op = decode_segment_op(frame.payload);
        if (op.kind != SegmentOpKind::row_patch || op.bytes.size() != page_size_) {
          poison("bundle install reload got a non-page frame");
        }
        reload.assign(reinterpret_cast<const char *>(op.bytes.data()),
                      reinterpret_cast<const char *>(op.bytes.data()) + page_size_);
        bytes = reload.data();
      }
      const PID first_pid = static_cast<PID>(pi * npp_);
      const std::lock_guard<std::mutex> guard(page_lock(first_pid));
      auto &shard = write_cache_.shard(pi);
      PageWriteCache::CachedPage *cached = nullptr;
      bool created = false;
      {
        const std::lock_guard<std::mutex> cache_guard(shard.mutex);
        auto it = shard.pages.find(pi);
        if (it != shard.pages.end()) {
          cached = it->second.get();
        } else {
          // BLOCKER-4: fully construct the CachedPage BEFORE it goes into the shared map,
          // so a bad_alloc never leaves a {pi, nullptr} entry a concurrent post-commit
          // reader (read_node_page -> shard.pages.find(pi)) would dereference and crash.
          auto page = std::make_unique<PageWriteCache::CachedPage>(page_size_);
          cached = page.get();
          shard.pages.emplace(pi, std::move(page));
          created = true;
        }
      }
      {
        const std::lock_guard<std::mutex> bytes_guard(cached->bytes_mutex);
        // No exception-capable work may be added between these bumps: std::memcpy
        // is the sole operation and cannot throw a C++ exception.
        page_versions_[pi].fetch_add(1, std::memory_order_acq_rel);  // -> odd
        std::memcpy(cached->bytes.data(), bytes, page_size_);
        page_versions_[pi].fetch_add(1, std::memory_order_release);  // -> even
      }
      {
        const std::lock_guard<std::mutex> cache_guard(shard.mutex);
        cached->dirty = true;
      }
      if (created) write_cache_.note_insert();
    }
  }

  void bundle_reset_overlay() {
    if (bundle_ctx_ != nullptr) {
      bundle_ctx_->pages.clear();
      bundle_ctx_->spilled.clear();
      bundle_ctx_->dirty.clear();
    }
  }

  // Force a directed from->to edge into the private overlay (design B.3 bundle spine).
  // `to`'s vector is read from the overlay into `scratch`; a forced patch evicts the
  // farthest edge if `from` is full. A failure to install the spine edge means a bundle
  // row could be unreachable, so it poisons before the commit point.
  void bundle_force_edge(PID from, PID to, std::vector<char> &scratch) {
    read_rmw_page(to, scratch.data());  // overlay copy of `to`
    const auto *to_vec = reinterpret_cast<const float *>(scratch.data() + node_offset_in_page(to));
    std::vector<float> vec(to_vec, to_vec + full_dim_);
    const std::unordered_map<PID, const CapturedNode *> empty_captured;
    if (patch_reverse_edge_impl(from, to, vec.data(), empty_captured, /*force=*/true, nullptr) !=
        PatchApplyResult::kApplied) {
      poison("canonical bundle spine edge install failed (unreachable bundle row)");
    }
  }

  // Ensure the base is a v3 maintenance-activated superblock before starting a
  // consolidate epoch (design section 7.2): the first WAL consolidate does an
  // activation checkpoint that flips a v3 base carrying the maintenance feature
  // bits. Maintenance never starts on a v2 base.
  void ensure_maintenance_activated() {
    if (maintenance_activated_) {
      return;
    }
    const auto st = read_superblock_wal2c_state(superblock_);
    if (superblock_.format_version == kQGFormatVersionV3 &&
        (st.required_feature_flags & kQgFeatMaintenanceTxV1) != 0) {
      maintenance_activated_ = true;
      maintenance_activation_gen_ = st.maintenance_activation_sb_generation;
      return;
    }
    maintenance_activating_ = true;
    try {
      checkpoint();  // emits a v3 base (checkpoint sees maintenance_activating_)
    } catch (...) {
      maintenance_activating_ = false;
      throw;
    }
    maintenance_activating_ = false;
    maintenance_activated_ = true;
  }

  // Ensure the base is a v3 pid-reuse-activated superblock before the first canonical
  // reuse bundle (design 7.2 / codex B.1). Mirrors ensure_maintenance_activated: if the
  // base already carries the pid_generation bit adopt it; otherwise flip a v3 base that
  // ORs the maintenance + pid-reuse feature bits and stamps the activation generation.
  // Called BEFORE the writer takes checkpoint_mutex_ (checkpoint() takes it), exactly
  // like consolidate calls ensure_maintenance_activated. Dormant until a bundle runs the
  // canonical branch (enable_pid_reuse_ || pid_generation_activated_).
  void ensure_pid_reuse_activated() {
    if (pid_generation_activated_) {
      return;
    }
    const auto st = read_superblock_wal2c_state(superblock_);
    if (superblock_.format_version == kQGFormatVersionV3 &&
        (st.required_feature_flags & kQgFeatPidGenerationV1) != 0) {
      pid_generation_activated_ = true;
      maintenance_activated_ = true;  // pid_generation implies maintenance (feature deps)
      pid_reuse_activation_gen_ = st.pid_reuse_activation_sb_generation;
      maintenance_activation_gen_ = st.maintenance_activation_sb_generation;
      return;
    }
    pid_reuse_activating_ = true;
    maintenance_activating_ = true;  // the pid bits require the maintenance/postredo bits too
    try {
      checkpoint();  // emits a v3 base (checkpoint sees pid_reuse_activating_)
    } catch (...) {
      pid_reuse_activating_ = false;
      maintenance_activating_ = false;
      throw;
    }
    pid_reuse_activating_ = false;
    maintenance_activating_ = false;
    pid_generation_activated_ = true;
    maintenance_activated_ = true;
  }

  [[nodiscard]] size_t cached_dirty_page_count() {
    size_t dirty = 0;
    for (size_t si = 0; si < PageWriteCache::kShards; ++si) {
      auto &shard = write_cache_.shard(si);
      const std::lock_guard<std::mutex> guard(shard.mutex);
      for (const auto &[pi, page] : shard.pages) {
        (void)pi;
        dirty += page->dirty ? 1 : 0;
      }
    }
    return dirty;
  }

  // BEGIN-time headroom preflight. A shortfall is an ordinary pre-transaction
  // error (the epoch has not started, so nothing needs rollback).
  void maint_statvfs_preflight(bool reclaim_slots, uint64_t baseline_dirty_pages) {
    struct statvfs vfs{};
    wal_failpoint(SegmentOpFailPoint::before_consolidate_statvfs);
    if (::statvfs(op_wal_->path().c_str(), &vfs) != 0) {
      throw std::runtime_error("QGUpdater::consolidate: statvfs failed errno=" +
                               std::to_string(errno));
    }
    const uint64_t available = static_cast<uint64_t>(vfs.f_bavail) * vfs.f_frsize;
    // Proven frame bound: the row phase visits target pages in physical order and
    // pins the current target; dependency reloads are clean relative to latest_spill
    // and evict without a frame. Reclaim first reads eligibility, then writes FREE
    // trailers and next pointers together in one physical-page pass. Therefore each
    // page contributes at most one repair frame plus one reclaim frame.
    const uint64_t committed = committed_.load(std::memory_order_acquire);
    const uint64_t committed_pages = committed == 0 ? 0 : (committed + npp_ - 1) / npp_;
    const uint64_t repair_pages = committed_pages;  // <= all live rows' pages
    const uint64_t reclaim_pages = reclaim_slots ? committed_pages : 0;
    constexpr uint64_t kPerPageFrameOverhead =
        alaya::wal::kHeaderBytes + alaya::wal::kTrailerBytes + 19 + 20;  // == 79
    constexpr uint64_t kMarkerFrameBytes = alaya::wal::kHeaderBytes + alaya::wal::kTrailerBytes +
                                           19 + 8;  // == 67 (begin/end epoch marker)
    const uint64_t per_page = page_size_ + kPerPageFrameOverhead;
    // Saturating arithmetic: a page count large enough to overflow cannot fit on any
    // real device, so saturate to UINT64_MAX (guaranteed reject) instead of wrapping.
    auto sat_mul = [](uint64_t a, uint64_t b) -> uint64_t {
      return (b != 0 && a > (std::numeric_limits<uint64_t>::max)() / b)
                 ? (std::numeric_limits<uint64_t>::max)()
                 : a * b;
    };
    auto sat_add = [](uint64_t a, uint64_t b) -> uint64_t {
      return a > (std::numeric_limits<uint64_t>::max)() - b ? (std::numeric_limits<uint64_t>::max)()
                                                            : a + b;
    };
    const uint64_t page_frame_upper = sat_add(repair_pages, reclaim_pages);
    const uint64_t wal_growth = sat_add(sat_mul(page_frame_upper, per_page), 2 * kMarkerFrameBytes);
    // Baseline writeback happens after this admission and before BEGIN; the
    // post-END install can rewrite every committed page. Count both CoW events.
    const uint64_t index_write_pages = sat_add(baseline_dirty_pages, committed_pages);
    const uint64_t index_cow_growth = sat_mul(index_write_pages, page_size_);
    const uint64_t base = sat_add(wal_growth, index_cow_growth);
    const uint64_t needed = sat_add(base, (std::max<uint64_t>)(uint64_t{16} << 20U, base / 20));
    stats_.maintenance_last_preflight_page_frames.store(page_frame_upper,
                                                        std::memory_order_relaxed);
    stats_.maintenance_last_preflight_wal_bytes.store(wal_growth, std::memory_order_relaxed);
    if (available < needed) {
      throw std::runtime_error(
          "QGUpdater::consolidate: insufficient free space for the maintenance WAL");
    }
  }

  // The single-threaded maintenance row phase over the private overlay.
  void consolidate_row_phase(size_t r_target, bool bloom_consolidate) {
    const size_t n = committed_.load(std::memory_order_acquire);
    const size_t target = r_target == 0 ? deg_ : std::min(r_target, deg_);
    if (bloom_consolidate) {
      std::vector<PID> dead;
      for (size_t p = 0; p < n; ++p) {
        if (is_hidden(static_cast<PID>(p))) {
          dead.push_back(static_cast<PID>(p));
        }
      }
      if (!dead.empty()) {
        DeadPIDBloom bloom(dead.size());
        for (PID pid : dead) {
          bloom.insert(pid);
        }
        // The scan reads the committed disk image (the overlay is private and the
        // shared cache was emptied at admission), so it finds candidates against
        // the pre-epoch state -- exactly what consolidate must purge.
        const auto rows = bloom_consolidation_rows(n, bloom, 1);
        for (PID u : rows) {
          consolidate_row(u, n, target, /*bloom_prefiltered=*/true);
        }
      }
    } else {
      for (size_t u = 0; u < n; ++u) {
        if (!is_hidden(static_cast<PID>(u))) {
          consolidate_row(static_cast<PID>(u), n, target);
        }
      }
    }
  }

  // Reclaim (MAJOR-3: runtime free chain must be GLOBALLY canonical, not just the new
  // set): collect the FULL final free set = the pre-existing chain UNION the newly-freed
  // rows, then rewrite EVERY free row's next pointer in ascending PID order (head = the
  // smallest free PID). Every overlay access below is grouped by physical page and
  // enforces the maintenance cap before moving on. Published at END.
  void maint_reclaim_phase() {
    const size_t n = committed_.load(std::memory_order_acquire);
    // 1. Walk the EXISTING free chain (canonical ascending or empty) to collect its PIDs.
    std::vector<PID> all_free;
    {
      std::unordered_set<uint64_t> seen;
      PID cur = maint_local_free_head_;
      while (cur != kPidMax) {
        if (static_cast<uint64_t>(cur) >= n || !seen.insert(static_cast<uint64_t>(cur)).second) {
          poison("consolidate reclaim: corrupt pre-existing free chain");
        }
        all_free.push_back(cur);
        const char *page = maint_overlay_page(page_index(cur));
        uint64_t next64 = kPidMax;
        std::memcpy(&next64, page + node_offset_in_page(cur), sizeof(next64));
        const PID next = next64 == kPidMax ? kPidMax : static_cast<PID>(next64);
        const size_t pi = page_index(cur);
        if (next == kPidMax || page_index(next) != pi) {
          maint_release_dependency_page(pi);
        }
        cur = next;
      }
    }
    if (all_free.size() != maint_local_free_count_) {
      poison("consolidate reclaim: free chain length disagrees with free_count");
    }
    // 2. Add newly-eligible reclaimed rows. A PID whose durable generation is already
    // saturated (UINT32_MAX) must never re-enter the free list -- reuse would wrap its
    // incarnation (design 2.2 / 3.1); it stays a permanent tombstone.
    const auto reclaim_snap = load_label_snapshot();
    std::unordered_set<uint64_t> existing(all_free.size() * 2);
    for (PID id : all_free) {
      existing.insert(static_cast<uint64_t>(id));
    }
    std::vector<PID> eligible = deleted_snapshot();
    std::sort(eligible.begin(), eligible.end());
    eligible.erase(std::remove_if(eligible.begin(),
                                  eligible.end(),
                                  [&](PID id) {
                                    if (id >= n || existing.count(static_cast<uint64_t>(id)) != 0) {
                                      return true;
                                    }
                                    const auto *b = reclaim_snap->find_binding(id);
                                    return b != nullptr &&
                                           b->pid_generation ==
                                               (std::numeric_limits<uint32_t>::max)();
                                  }),
                   eligible.end());
    // Inspect eligibility read-only first. The final free set is then known, so
    // FREE trailers and canonical next pointers can be written together in one
    // physical-page pass (one reclaim frame per page).
    std::vector<PID> newly_free;
    newly_free.reserve(eligible.size());
    for (size_t begin = 0; begin < eligible.size();) {
      const size_t pi = page_index(eligible[begin]);
      size_t end = begin + 1;
      while (end < eligible.size() && page_index(eligible[end]) == pi) {
        ++end;
      }
      const char *page = maint_overlay_page(pi);
      for (size_t i = begin; i < end; ++i) {
        const PID id = eligible[i];
        const QGRowTrailer trailer = qg_read_page_trailer(page, page_size_, npp_, id % npp_);
        if ((trailer.flags & kQGRowTombstone) == 0 || (trailer.flags & kQGRowFree) != 0) {
          continue;  // only tombstoned, not-yet-free rows are eligible
        }
        newly_free.push_back(id);
        all_free.push_back(id);
      }
      maint_release_dependency_page(pi);
      begin = end;
    }

    // 3. Canonicalize the WHOLE free set ascending -- byte-identical to the recovery
    // rebuild (rebuild_state_after_replay), so a reopen never re-orders the chain.
    std::sort(all_free.begin(), all_free.end());
    all_free.erase(std::unique(all_free.begin(), all_free.end()), all_free.end());
    for (size_t begin = 0; begin < all_free.size();) {
      const size_t pi = page_index(all_free[begin]);
      size_t end = begin + 1;
      while (end < all_free.size() && page_index(all_free[end]) == pi) {
        ++end;
      }
      char *page = maint_overlay_page(pi);
      for (size_t i = begin; i < end; ++i) {
        const PID id = all_free[i];
        if (std::binary_search(newly_free.begin(), newly_free.end(), id)) {
          QGRowTrailer trailer = qg_read_page_trailer(page, page_size_, npp_, id % npp_);
          trailer.valid_degree = 0;
          trailer.flags |= kQGRowFree;
          qg_write_page_trailer(page, page_size_, npp_, id % npp_, trailer);
          stats_.freed_slots++;
        }
        const uint64_t next64 = i + 1 < all_free.size() ? all_free[i + 1] : kPidMax;
        std::memcpy(page + node_offset_in_page(id), &next64, sizeof(next64));
      }
      maint_dirty_.insert(pi);  // rewritten next pointers must be logged + installed
      maint_resident_dirty_.insert(pi);
      maint_spill_over_cap(pi);
      begin = end;
    }
    maint_local_free_head_ = all_free.empty() ? kPidMax : all_free.front();
    maint_local_free_count_ = all_free.size();
  }

  // consolidate() under enable_wal: one maintenance transaction (design section 1).
  void consolidate_wal_transaction(size_t num_threads,
                                   size_t r_target,
                                   bool reclaim_slots,
                                   bool bloom_consolidate) {
    (void)num_threads;  // maintenance runs single-threaded (serial WAL lane; B-2C-05
                        // parallel page workers are documented follow-on hardening).
    ensure_writable();
    if (allocated_points_.load(std::memory_order_acquire) !=
        committed_.load(std::memory_order_acquire)) {
      throw std::logic_error("QGUpdater::consolidate requires allocated == committed");
    }
    if (has_staged_edges()) {
      throw std::logic_error("QGUpdater::consolidate requires no staged backlinks");
    }
    if (maintenance_active_) {
      throw std::logic_error("QGUpdater::consolidate is already in progress");
    }
    ensure_maintenance_activated();  // may checkpoint (takes checkpoint_mutex_) -> BEFORE the guard
    const std::lock_guard<std::mutex> checkpoint_guard(checkpoint_mutex_);
    // Admission must precede every baseline pwrite that may allocate filesystem
    // blocks. A statvfs rejection is still a clean, retryable pre-transaction
    // error because no page write and no BEGIN has happened.
    maint_statvfs_preflight(reclaim_slots, cached_dirty_page_count());
    // Baseline normalization (design section 1.2 step 2): flush committed dirty
    // pages, then empty the shared cache so a concurrent search reads only committed
    // disk state during the epoch and the private overlay is the sole mutation store.
    // (has_staged_edges() above already guaranteed no staged edges to drain.)
    try {
      flush_dirty(1);
    } catch (...) {
      // A failed pwrite can be partial even before BEGIN. Fail this handle closed;
      // flush_dirty has already published poison and closed every odd version.
      poison_current_exception("consolidate baseline flush failed before BEGIN");
    }
    evict_clean(0);
    pid_scan_mapping_.reset();  // a stale bloom mapping from a prior epoch is unsafe
    if (write_cache_.total_pages() != 0) {
      poison("consolidate baseline cache did not drain to zero");
    }
    const uint64_t epoch = last_completed_consolidate_epoch_ + 1;
    maint_epoch_ = epoch;
    maint_reset_overlay();
    maint_local_free_head_ = free_list_head_.load(std::memory_order_acquire);
    maint_local_free_count_ = free_count_.load(std::memory_order_acquire);
    try {
      wal_failpoint(SegmentOpFailPoint::before_consolidate_begin_append);  // C0
      wal_append(encode_consolidate_marker(segment_uid_,
                                           superblock_.generation,
                                           SegmentOpKind::consolidate_begin,
                                           epoch),
                 alaya::wal::WalFile::Sync::buffered);
      wal_failpoint(SegmentOpFailPoint::after_consolidate_begin_append);  // C1
      wal_failpoint(SegmentOpFailPoint::before_consolidate_begin_fsync);  // C2
      force_wal();  // BEGIN durable before any page mutation
      wal_failpoint(SegmentOpFailPoint::after_consolidate_begin_fsync);  // C3
      maintenance_active_ = true;
      maint_in_build_phase_ = true;  // no index/arena write allowed until END durable
      consolidate_row_phase(r_target, bloom_consolidate);
      wal_failpoint(SegmentOpFailPoint::after_consolidate_live_repair_before_free_image);  // C6
      if (reclaim_slots) {
        maint_reclaim_phase();
      }
      // Finalize only resident bytes changed since their latest spill. A dirty-history
      // page reloaded solely for dependency reads already has a current WAL image and
      // must not be appended again.
      wal_failpoint(SegmentOpFailPoint::before_consolidate_end_append);
      for (auto &[pi, bytes] : maint_pages_) {
        if (maint_resident_dirty_.count(pi) != 0) {
          maint_spilled_[pi] =
              maint_log_page(pi, bytes.data(), alaya::wal::WalFile::Sync::buffered);
        }
      }
      maint_resident_dirty_.clear();
      // END: buffered append -> torn-END window (C7) -> the single durable commit
      // point (C8). Splitting append from force lets the harness cut a torn END; the
      // net WAL bytes + observer notify are identical to one Sync::fsync append.
      wal_append(encode_consolidate_marker(segment_uid_,
                                           superblock_.generation,
                                           SegmentOpKind::consolidate_end,
                                           epoch),
                 alaya::wal::WalFile::Sync::buffered);
      wal_failpoint(SegmentOpFailPoint::after_consolidate_end_append_before_fsync);  // C7
      force_wal();  // END durable: the single maintenance commit point
      wal_failpoint(SegmentOpFailPoint::after_consolidate_end_fsync);  // C8
      // Commit point passed: leave the BUILD window so the install writes are allowed.
      maint_in_build_phase_ = false;
      maint_install_all();  // fires after_consolidate_install_page (C9) per page
      wal_failpoint(SegmentOpFailPoint::after_consolidate_install_before_publish);  // C10
      if (reclaim_slots) {
        free_list_head_.store(maint_local_free_head_, std::memory_order_release);
        free_count_.store(maint_local_free_count_, std::memory_order_release);
      }
      last_completed_consolidate_epoch_ = epoch;
      maintenance_active_ = false;
      maint_reset_overlay();
      wal_failpoint(SegmentOpFailPoint::after_consolidate_publish);  // C11
    } catch (...) {
      // Design/checkpoint-admission ruling: a failure past BEGIN keeps the epoch
      // state and overlay intact and poisons the handle -- recovery rebuilds from
      // the WAL (an unmatched BEGIN is semantically truncated; a post-END failure
      // rolls forward). Never clean up and pretend to continue on this handle.
      // BLOCKER-2: catch-all so a non-std::exception / bad_alloc past BEGIN still latches.
      poison_current_exception("consolidate failed mid-transaction");
    }
  }

  void stage_edge(StagedEdge edge) {
    auto &stripe = staged_[static_cast<size_t>(edge.v) % staged_.size()];
    const std::lock_guard<std::mutex> guard(stripe.mutex);
    stripe.by_target[edge.v].push_back(std::move(edge));
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
    std::vector<PatchIntent *> primary_intents(x_range, nullptr);
    for (const auto &group : groups) {
      for (const auto &edge : group.edges) {
        if (edge.primary) {
          primary[edge.x - x_min] = group.v;
          primary_intents[edge.x - x_min] = edge.intent.get();
        }
      }
    }

    const int nt = static_cast<int>(std::max<size_t>(1, num_threads));
    parallel_for_catch(0, static_cast<int64_t>(groups.size()), nt, 64, [&](int64_t gi) {
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
        const float *x_raw = nullptr;
        const auto ensure_candidate_raw = [&] {
          if (x_raw != nullptr) return;
          // x was appended earlier in this batch. This normally hits its
          // dirty cache page; the copy keeps a disk-read fallback alive.
          x_raw = cached_raw(edge.x);
          if (x_raw == nullptr) {
            AlignedBuf x_page(page_size_);
            read_at(page_offset(edge.x), x_page.data(), page_size_);
            stats_.patch_page_reads++;
            const auto *raw =
                reinterpret_cast<const float *>(x_page.data() + node_offset_in_page(edge.x));
            x_vec.assign(raw, raw + full_dim_);
            x_raw = x_vec.data();
          }
        };
        bool applied = false;
        if (edge.intent != nullptr) {
          const PatchApplyResult result =
              apply_patch_intent(*edge.intent, nullptr, no_capture, false);
          if (result == PatchApplyResult::kStale) {
            stats_.patch_intent_stale_fallbacks.fetch_add(1, std::memory_order_relaxed);
            ensure_candidate_raw();
            applied = patch_reverse_edge(v, edge.x, x_raw, no_capture, false);
          } else {
            applied = result == PatchApplyResult::kApplied;
          }
        } else {
          ensure_candidate_raw();
          applied = patch_reverse_edge(v, edge.x, x_raw, no_capture, false);
        }
        if (applied) {
          successes[edge.x - x_min].fetch_add(1, std::memory_order_relaxed);
        }
      }
    });

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
      const float *x_raw = nullptr;
      const auto ensure_candidate_raw = [&] {
        if (x_raw != nullptr) return;
        x_raw = cached_raw(x);
        if (x_raw == nullptr) {
          AlignedBuf x_page(page_size_);
          read_at(page_offset(x), x_page.data(), page_size_);
          stats_.patch_page_reads++;
          const auto *raw = reinterpret_cast<const float *>(x_page.data() + node_offset_in_page(x));
          x_vec.assign(raw, raw + full_dim_);
          x_raw = x_vec.data();
        }
      };
      const std::unordered_map<PID, const CapturedNode *> no_capture;
      bool applied = false;
      if (primary_intents[xi] != nullptr) {
        const PatchApplyResult result =
            apply_patch_intent(*primary_intents[xi], nullptr, no_capture, true);
        if (result == PatchApplyResult::kStale) {
          stats_.patch_intent_stale_fallbacks.fetch_add(1, std::memory_order_relaxed);
          ensure_candidate_raw();
          applied = patch_reverse_edge(v, x, x_raw, no_capture, true);
        } else {
          applied = result == PatchApplyResult::kApplied;
        }
      } else {
        ensure_candidate_raw();
        applied = patch_reverse_edge(v, x, x_raw, no_capture, true);
      }
      if (applied) {
        stats_.forced_links++;
      }
    }
  }

  void flush_dirty(size_t num_threads) {
    if (!params_.write_cache) return;
    if (enable_wal_ && !replaying_) {
      force_wal();  // group-commit the WAL prefix before any index writeback
    }
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
    parallel_for_catch(0, static_cast<int64_t>(dirty.size()), nt, 1, [&](int64_t i) {
      const auto &page = dirty[static_cast<size_t>(i)];
      if (page.index >= page_versions_.size()) {
        throw std::runtime_error("QGUpdater: dirty page exceeds max_points capacity");
      }
      write_page_versioned(page.index, [&] {
        write_at(kSectorLen + page.index * page_size_, page.data, page_size_);
      });
    });
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

  /**
   * Design section 4.1: the MAP_SHARED merge path is retired. Dirty overlays are
   * now flushed to the index via the ordinary write_at path (which mirrors the
   * resident arena), so there is no second, forking writeback route. Kept as a
   * thin alias so the existing Bloom call sites need no rewire.
   */
  void merge_dirty_into_mapping(size_t num_threads) { flush_dirty(num_threads); }

  /** Record a maintenance-batch peak and report whether high was reached. */
  bool note_maintenance_pool_and_test_high() {
    const size_t pages = write_cache_.total_pages();
    uint64_t peak = stats_.maintenance_peak_pool_pages.load(std::memory_order_relaxed);
    while (peak < pages &&
           !stats_.maintenance_peak_pool_pages.compare_exchange_weak(peak,
                                                                     pages,
                                                                     std::memory_order_relaxed,
                                                                     std::memory_order_relaxed)) {
    }
    return params_.write_cache && pages >= params_.cache_cap_pages;
  }

  /** Enforce high->low after the maintenance worker team and all latches exit. */
  void enforce_maintenance_watermark(size_t num_threads) {
    if (note_maintenance_pool_and_test_high()) {
      flush_dirty(num_threads);
      evict_clean(params_.cache_cap_pages / 2);
    }
  }

  void enforce_bloom_maintenance_watermark(size_t num_threads) {
    if (note_maintenance_pool_and_test_high()) {
      merge_dirty_into_mapping(num_threads);
      evict_clean(params_.cache_cap_pages / 2);
    }
  }

  /** @brief Drop clean pages (arbitrary order) until the pool holds at most
   * @p target pages. Runs at the batch barrier only — cached_raw() pointers
   * from the drain phase are dead by then. */
  void evict_clean(size_t target) {
    for (size_t si = 0; si < PageWriteCache::kShards && write_cache_.total_pages() > target; ++si) {
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
    const size_t snapshot =
        std::max(committed_.load(std::memory_order_acquire), insert_visible_override_);
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
        const float d = space::l2_sqr(tvec, routing->medoid_vectors.data() + full_dim_ * m, dim_);
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
    // Writer-private seed (design B.3): during a canonical reuse bundle the published
    // routing entry may be a hidden dead row (delete-all -> all-reuse) that reaches no
    // built bundle row, so seed the beam with the first row of THIS bundle. Only the
    // single writer reaches this (search_for_insert is not on the concurrent query face),
    // so the private entry never leaks to a public search.
    if (bundle_ctx_ != nullptr && bundle_ctx_->private_entry != kPidMax &&
        bundle_ctx_->private_entry < snapshot) {
      sp.insert(bundle_ctx_->private_entry, FLT_MAX);
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
      uint64_t row_generation = 0;
      read_node_page(cur, page.data(), false, &row_generation);
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
      cap.row_generation = row_generation;
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
      if (writer_hidden(pool[i].id)) {  // wal-2c B.3: a revealed reused bundle row is selectable
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
    const bool canonical = bundle_ctx_ != nullptr;  // wal-2c B.4 canonical reuse bundle
    modify_node_page(id, [&](char *page) {
      // A newly-created one-row page has deterministic zero tail padding;
      // read_at already zero-fills an EOF cache miss.
      QGRowTrailer trailer = row_trailer(page, id);
      if (reused &&
          (trailer.flags & (kQGRowTombstone | kQGRowFree)) != (kQGRowTombstone | kQGRowFree)) {
        // Legacy reuse (non-WAL LIFO): a caller precondition. Canonical reuse: the
        // overlay preimage came from the committed FREE page, so a non-FREE preimage
        // is corruption -> poison (fail closed) rather than a caller error.
        if (canonical) {
          poison("QGUpdater: canonical reused row overlay preimage is not TOMBSTONE|FREE");
        }
        throw std::runtime_error("QGUpdater: reused row lost tombstone/free state");
      }
      std::memcpy(page + node_offset_in_page(id), row, node_len_);
      row_generations_[id].fetch_add(1, std::memory_order_release);
      trailer.valid_degree = static_cast<uint16_t>(degree);
      // Legacy reuse keeps the row FREE (hidden) until publish_common clears it; a
      // canonical bundle makes the FINAL page trailer live now (design B.4) while the
      // RAM hidden bit stays set until the post-kind=8 clear_hidden.
      trailer.flags = (reused && !canonical)
                          ? static_cast<uint16_t>(trailer.flags | kQGRowTombstone | kQGRowFree)
                          : 0;
      set_row_trailer(page, id, trailer);
      add_row_indegree(row, degree, 1);
      clear_turnover(id);
      return true;
    });
  }

  /**
   * @brief Cheap reverse-edge patch: fill a ghost slot, else evict the
   * farthest current edge (FastScan estimate with v as its own query) when the
   * new edge is shorter. kAlphaEvict additionally rejects the new edge when an
   * already-captured current neighbor alpha-occludes it (zero extra I/O).
   */
  enum class PatchApplyResult { kRejected, kApplied, kStale };

  bool patch_reverse_edge(const CapturedNode &v,
                          PID x_id,
                          const float *x_vec,
                          const std::unordered_map<PID, const CapturedNode *> &captured,
                          bool force) {
    return patch_reverse_edge_impl(v.id, x_id, x_vec, captured, force, nullptr) ==
           PatchApplyResult::kApplied;
  }

  /** Validate and apply a payload prepared from an earlier raw-row snapshot. */
  PatchApplyResult apply_patch_intent(const PatchIntent &intent,
                                      const float *candidate_raw,
                                      const std::unordered_map<PID, const CapturedNode *> &captured,
                                      bool force) {
    return patch_reverse_edge_impl(intent.target_row,
                                   intent.candidate_pid,
                                   candidate_raw,
                                   captured,
                                   force,
                                   &intent);
  }

  PatchApplyResult patch_reverse_edge_impl(
      PID target_id,
      PID x_id,
      const float *x_vec,
      const std::unordered_map<PID, const CapturedNode *> &captured,
      bool force,
      const PatchIntent *intent) {
    if (intent != nullptr && (intent->target_row != target_id || intent->candidate_pid != x_id ||
                              target_id >= row_generations_.size())) {
      return PatchApplyResult::kStale;
    }
    if (intent == nullptr && x_vec == nullptr) {
      throw std::invalid_argument("QGUpdater::patch_reverse_edge null candidate vector");
    }
    if (!force && params_.backlink_mode == UpdateParams::Backlink::kAlphaEvict &&
        !captured.empty() && x_vec == nullptr) {
      throw std::invalid_argument("QGUpdater: alpha intent requires candidate raw vector");
    }
    if (writer_hidden(target_id)) {  // wal-2c B.3: a revealed reused bundle row accepts backlinks
      return PatchApplyResult::kRejected;
    }
    const std::lock_guard<std::mutex> guard(page_lock(target_id));
    if (intent != nullptr &&
        row_generations_[target_id].load(std::memory_order_acquire) != intent->row_generation) {
      return PatchApplyResult::kStale;
    }
    bool installed_preencoded_payload = false;
    const bool changed = modify_node_page(target_id, [&](char *page) {
      char *row = page + node_offset_in_page(target_id);
      const auto *row_f = reinterpret_cast<const float *>(row);
      auto *ids = reinterpret_cast<PID *>(row + neighbor_off_bytes());
      auto *fac = reinterpret_cast<float *>(row + factor_off_bytes());
      QGRowTrailer trailer = row_trailer(page, target_id);
      size_t degree = trailer.valid_degree;
      for (size_t j = 0; j < degree; ++j) {
        if (ids[j] == x_id) return true;
      }

      // residual-consistent comparison value for the new edge (v,x)
      float x_res_sqr = 0;
      float v_res_sqr = 0;
      for (size_t j = 0; j < res_dim_; ++j) {
        if (intent == nullptr) {
          x_res_sqr += x_vec[dim_ + j] * x_vec[dim_ + j];
        }
        v_res_sqr += row_f[dim_ + j] * row_f[dim_ + j];
      }
      const float est_new = intent == nullptr
                                ? space::l2_sqr(row_f, x_vec, dim_) + x_res_sqr + v_res_sqr
                                : intent->estimated_distance;

      // alpha-occlusion test using neighbors already captured by this search
      if (!force && params_.backlink_mode == UpdateParams::Backlink::kAlphaEvict &&
          !captured.empty()) {
        size_t checked = 0;
        const float d_vx =
            intent == nullptr
                ? space::l2_sqr(row_f, x_vec, dim_) +
                      (res_dim_ > 0 ? space::l2_sqr(row_f + dim_, x_vec + dim_, res_dim_) : 0.0F)
                : intent->exact_distance;
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
              static_cast<uint64_t>(std::random_device()()) ^
              static_cast<uint64_t>(std::hash<std::thread::id>()(std::this_thread::get_id())));
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
            const auto *nb_f =
                reinterpret_cast<const float *>(nb_page.data() + node_offset_in_page(ids[j]));
            exact[j] = space::l2_sqr(row_f, nb_f, dim_) +
                       (res_dim_ > 0 ? space::l2_sqr(row_f + dim_, nb_f + dim_, res_dim_) : 0.0F);
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
                ? (intent == nullptr
                       ? space::l2_sqr(row_f, x_vec, dim_) +
                             (res_dim_ > 0 ? space::l2_sqr(row_f + dim_, x_vec + dim_, res_dim_)
                                           : 0.0F)
                       : intent->exact_distance)
                : est_new;
        const double improvement = static_cast<double>(worst) - static_cast<double>(new_distance);
        const double required_margin = params_.evict_margin * std::abs(static_cast<double>(worst));
        if ((!force && improvement <= 0) ||
            (params_.evict_margin > 0 && improvement <= required_margin)) {
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

      const bool patched = intent == nullptr
                               ? patch_slot(row, slot, x_id, x_vec, x_res_sqr, evicted)
                               : patch_slot(row, slot, *intent, evicted);
      if (!patched) {
        stats_.degenerate_skips++;
        return false;
      }
      installed_preencoded_payload = intent != nullptr;
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
        set_row_trailer(page, target_id, trailer);
        stats_.free_slot_fills++;
      }
      increment_turnover(target_id);
      return true;
    });
    if (installed_preencoded_payload) {
      stats_.patch_intents_applied.fetch_add(1, std::memory_order_relaxed);
    }
    return changed ? PatchApplyResult::kApplied : PatchApplyResult::kRejected;
  }

  /** @brief Quality-reference arm: full RobustPrune with all neighbor vectors read back. */
  bool full_reverse_recompute(const CapturedNode &v, PID x_id, const float *x_vec) {
    if (writer_hidden(v.id)) {  // wal-2c B.3: a revealed reused bundle row is recomputable
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

      // Use the writer-visible floor ONLY inside a canonical bundle (bundle_ctx_ != null):
      // otherwise committed_ (== old_hwm mid-bundle) would filter this bundle's own appended
      // rows out of the recompute, dropping bundle-internal edges. BLOCKER-6: the LEGACY 2A
      // path ALSO raises insert_visible_override_ (for B-07 seeding), so applying the floor
      // unconditionally changed which candidates a legacy FullPrune recompute considered --
      // altering the prune result + kind=1 payload vs the pre-2C behavior. Gate on the bundle
      // context so legacy stays exactly committed_ (byte-for-byte the pre-reuse FullPrune).
      const size_t snapshot =
          bundle_ctx_ != nullptr
              ? std::max(committed_.load(std::memory_order_acquire), insert_visible_override_)
              : committed_.load(std::memory_order_acquire);

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
        if ((nb >= snapshot && nb != x_id) || nb == v.id || writer_hidden(nb)) {
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
      clear_turnover(v.id);
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
    search_for_insert(u_vec.data(),
                      pool,
                      gp.ef_maintenance,
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
    pool.erase(std::remove_if(pool.begin(),
                              pool.end(),
                              [&](const CapturedNode &c) {
                                return c.id == u || is_hidden(c.id);
                              }),
               pool.end());
    std::sort(pool.begin(), pool.end(), [](const CapturedNode &a, const CapturedNode &b) {
      return a.dist != b.dist ? a.dist < b.dist : a.id < b.id;
    });
    pool.erase(std::unique(pool.begin(),
                           pool.end(),
                           [](const CapturedNode &a, const CapturedNode &b) {
                             return a.id == b.id;
                           }),
               pool.end());

    std::vector<size_t> sel;
    for (size_t i = 0; i < pool.size() && sel.size() < r_target; ++i) {
      bool occluded = false;
      for (size_t s : sel) {
        const float d =
            space::l2_sqr(pool[s].raw(), pool[i].raw(), dim_) +
            (res_dim_ > 0 ? space::l2_sqr(pool[s].raw() + dim_, pool[i].raw() + dim_, res_dim_)
                          : 0.0F);
        if (params_.alpha * params_.alpha * d <= pool[i].dist) {
          occluded = true;
          break;
        }
      }
      if (!occluded) sel.push_back(i);
    }
    if (!gp.pump_only) {
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
          clear_turnover(u);
          return true;
        });
      }
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
  void consolidate_row(PID u, size_t snapshot, size_t r_target, bool bloom_prefiltered = false) {
    const std::lock_guard<std::mutex> guard(page_lock(u));
    const auto mutate = [&](char *page) {
      char *row = page + node_offset_in_page(u);
      const auto *row_f = reinterpret_cast<const float *>(row);
      auto *ids = reinterpret_cast<PID *>(row + neighbor_off_bytes());
      QGRowTrailer trailer = row_trailer(page, u);
      size_t degree = trailer.valid_degree;

      // Collect the current live adjacency.  The v2 row invariant is a packed
      // prefix [0, valid_degree), so removal below swaps the last valid slot
      // into the hole and keeps processing the same index.
      std::unordered_set<PID> chosen;
      thread_local std::vector<PID> bloom_chosen;
      if (bloom_prefiltered) {
        bloom_chosen.clear();
        bloom_chosen.reserve(deg_);
      }
      const auto chosen_contains = [&](PID id) {
        return bloom_prefiltered
                   ? std::find(bloom_chosen.begin(), bloom_chosen.end(), id) != bloom_chosen.end()
                   : chosen.count(id) != 0;
      };
      const auto chosen_insert = [&](PID id) {
        if (bloom_prefiltered) {
          if (!chosen_contains(id)) bloom_chosen.push_back(id);
        } else {
          chosen.insert(id);
        }
      };
      bool has_dead = bloom_prefiltered;
      for (size_t j = 0; j < degree; ++j) {
        if (is_hidden(ids[j]))
          has_dead = true;
        else
          chosen_insert(ids[j]);
      }
      if (!has_dead) return false;
      size_t live_degree = bloom_prefiltered ? bloom_chosen.size() : chosen.size();

      float u_res_sqr = 0;
      for (size_t j = 0; j < res_dim_; ++j) {
        u_res_sqr += row_f[dim_ + j] * row_f[dim_ + j];
      }
      QGQuery uq(row_f, pd_);
      uq.query_prepare(qg_.rotator_, qg_.scanner_);
      uq.set_sqr_qr(u_res_sqr);

      AlignedBuf d_page;
      AlignedBuf cand_page;
      if (!bloom_prefiltered) {
        d_page.resize(page_size_);
        cand_page.resize(page_size_);
      }
      std::vector<float> appro_owned;
      thread_local std::vector<float> bloom_appro;
      auto &appro = bloom_prefiltered ? bloom_appro : appro_owned;
      appro.resize(deg_);
      bool dirty = false;
      size_t j = 0;
      while (j < degree) {
        if (!is_hidden(ids[j])) {
          ++j;
          continue;
        }
        const PID d = ids[j];
        // A splice is two neighborhood-turnover events: the dead edge is
        // purged, then a newly sourced replacement edge is installed.
        increment_turnover(u);
        // Headroom preservation: only splice back up to r_target live edges;
        // surplus dead slots are removed from the packed prefix.
        if (!params_.splice_enabled || live_degree >= r_target) {
          erase_slot(row, j, degree);
          --degree;
          stats_.ghosted_slots++;
          dirty = true;
          continue;
        }
        // FastScan over the dead node's own row (codes centered at d, but the
        // estimator still targets ||u - n_i||) recalls candidates cheaply...
        const char *d_page_data = nullptr;
        if (bloom_prefiltered) {
          d_page_data = bloom_dependency_page(d);
        } else {
          read_rmw_page(d, d_page.data());
          d_page_data = d_page.data();
        }
        stats_.patch_page_reads++;
        const char *d_row = d_page_data + node_offset_in_page(d);
        const size_t d_degree = row_trailer(d_page_data, d).valid_degree;
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
        std::vector<std::pair<float, PID>> recalled_owned;
        thread_local std::vector<std::pair<float, PID>> bloom_recalled;
        auto &recalled = bloom_prefiltered ? bloom_recalled : recalled_owned;
        recalled.clear();
        if (bloom_prefiltered) recalled.reserve(deg_);
        for (size_t k = 0; k < d_degree; ++k) {
          const PID cand = d_ids[k];
          if (cand == u || cand >= snapshot || !std::isfinite(appro[k]) || is_hidden(cand) ||
              chosen_contains(cand)) {
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
        std::vector<float> best_vec_owned;
        thread_local std::vector<float> bloom_best_vec;
        auto &best_vec = bloom_prefiltered ? bloom_best_vec : best_vec_owned;
        best_vec.clear();
        for (const auto &[est, cand] : recalled) {
          const char *cand_page_data = nullptr;
          if (bloom_prefiltered) {
            cand_page_data = bloom_dependency_page(cand);
          } else {
            read_rmw_page(cand, cand_page.data());
            cand_page_data = cand_page.data();
          }
          stats_.patch_page_reads++;
          const auto *cand_f =
              reinterpret_cast<const float *>(cand_page_data + node_offset_in_page(cand));
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
            chosen_insert(best);
            ++live_degree;
            stats_.spliced_slots++;
            increment_turnover(u);
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
    };
    if (bloom_prefiltered) {
      modify_bloom_node_page(u, mutate);
    } else {
      modify_node_page(u, mutate);
    }
  }

  /** @brief Replace one slot's code + factors + id inside a row buffer. */
  bool patch_slot(char *row,
                  size_t slot,
                  PID x_id,
                  const float *x_vec,
                  float x_res_sqr,
                  bool had_old) {
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
    const std::array<float, 3> factors = {payload.triple_x, payload.factor_dq, payload.factor_vq};
    return install_patch_payload(row,
                                 slot,
                                 x_id,
                                 reinterpret_cast<const uint8_t *>(payload.bin.data()),
                                 factors,
                                 had_old);
  }

  /** @brief Install a lock-free prepared payload into one logical slot. */
  bool patch_slot(char *row, size_t slot, const PatchIntent &intent, bool had_old) {
    if (intent.codes.size() != pd_ / 8 || !std::isfinite(intent.factors[0]) ||
        !std::isfinite(intent.factors[1]) || !std::isfinite(intent.factors[2])) {
      return false;
    }
    return install_patch_payload(row,
                                 slot,
                                 intent.candidate_pid,
                                 intent.codes.data(),
                                 intent.factors,
                                 had_old);
  }

  bool install_patch_payload(char *row,
                             size_t slot,
                             PID x_id,
                             const uint8_t *slot_code,
                             const std::array<float, 3> &factors,
                             bool had_old) {
    const PID old_id = reinterpret_cast<const PID *>(row + neighbor_off_bytes())[slot];

    const size_t block_idx = slot / kBatchSize;
    const size_t in_block = slot % kBatchSize;
    uint8_t *block = reinterpret_cast<uint8_t *>(row + code_off_bytes()) + block_idx * pd_ * 4;
    thread_local std::vector<uint64_t> bins;
    bins.resize(kBatchSize * pd_ / 64);
    unpack_codes_block(pd_, block, bins.data());
    std::memcpy(reinterpret_cast<uint8_t *>(bins.data() + in_block * (pd_ / 64)),
                slot_code,
                pd_ / 8);
    pack_codes(pd_, bins.data(), kBatchSize, block);

    auto *fac = reinterpret_cast<float *>(row + factor_off_bytes());
    fac[slot] = factors[0];
    fac[deg_ + slot] = factors[1];
    fac[2 * deg_ + slot] = factors[2];
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
      unpack_codes_block(pd_, codes + block * pd_ * 4, bins.data() + block * kBatchSize * words);
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

  // ===================== op-WAL: durable in-place updates (G1) ================
  // Design: unified-wal-vocabulary.md + its amendment. no-steal page cache with
  // force-before-writeback (clause B), a durable segment lineage uid in the
  // superblock reserved area (clause F), a dedicated recovery open + full state
  // rebuild (clause C), a flip-frame generation state machine (clause E), and a
  // fail-closed writer that poisons on any WAL/critical-index error (clause I).

  // QGSuperblockV2::reserved[408] sub-layout (host-endian, mirroring the uid
  // convention). 512B superblock size is unchanged; see qg.hpp for the map.
  //   [0..8)   segment_uid
  //   [8..40)  label state: slot | generation | count | checksum (4x u64)
  //   [40..56) tx state: last_committed_txid | applied_collection_op_id (2x u64)
  //   [56..408) reserved for 2C.
  static constexpr size_t kUidReservedOffset = 0;
  static constexpr size_t kLabelStateReservedOffset = 8;
  static constexpr size_t kTxStateReservedOffset = 40;
  static_assert(kTxStateReservedOffset + 16 <= 408,
                "superblock reserved sub-layout overflows QGSuperblockV2::reserved[]");

#include "index/graph/laser/qg/detail/qg_updater_wal.hpp"
  QuantizedGraph &qg_;
  UpdateParams params_;
  int fd_ = -1;             // buffered fd: all reads (page-cache served) + fallback writes
  int wfd_ = -1;            // O_DIRECT write fd (parallel inode-shared overwrites)
  bool direct_io_ = false;  // wfd_ opened successfully and writes routed to it
  std::atomic<size_t> committed_;
  std::atomic<size_t> allocated_points_;
  // Bundle-internal visibility floor (B-07): during commit_physical_bundle's
  // serial insert loop, search_for_insert must see rows already appended earlier
  // in the SAME bundle (not only committed_), so an empty-graph first bundle of
  // N>1 forms a connected seed graph instead of N-1 unreachable singletons.
  // Writer-only (single-writer insert path), 0 outside a bundle, never read by
  // the concurrent search() path (which reads committed_).
  size_t insert_visible_override_ = 0;
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
  std::unique_ptr<SharedFileMapping> pid_scan_mapping_;
  std::array<StagedStripe, 64> staged_;
  std::vector<std::mutex> page_locks_;
  std::vector<std::atomic<int32_t>> indegree_;
  std::vector<std::atomic<uint16_t>> turnover_;
  // Changes only when a PID's raw row identity changes (append/reuse/free-list
  // overlay), not for adjacency patches.  This avoids false invalidation from
  // unrelated writers sharing the same physical page.
  std::vector<std::atomic<uint64_t>> row_generations_;
  // Per-page seqlock: odd while a locked writer rewrites the page. Readers
  // outside the page lock validate before/after the pread and retry.
  std::vector<std::atomic<uint32_t>> page_versions_;
  // One fixed, atomic bit per PID. Set means tombstoned, free, or a reused row
  // that is still dark. Fixed sizing makes hot-path reads pointer-stable.
  std::vector<std::atomic<uint64_t>> hidden_words_;
  std::mutex routing_snapshot_mutex_;
  std::vector<std::unique_ptr<RoutingSnapshot>> routing_snapshots_;
  std::atomic<const RoutingSnapshot *> routing_snapshot_{nullptr};

  // --- op-WAL (durable in-place updates, G1) ---
  bool enable_wal_ = false;                      // mirrors params_.enable_wal; the WAL is live
  std::unique_ptr<alaya::wal::WalFile> op_wal_;  // <index>.opwal, present iff enable_wal_
  bool replaying_ = false;                       // set during recovery redo: suppress log + force
  std::string poison_reason_;                    // non-empty => writer permanently poisoned
  std::atomic<bool> poisoned_{false};         // lock-free latch mirroring poison_reason_ for reads
  uint64_t segment_uid_ = 0;                  // durable lineage id (superblock reserved[0..8))
  uint64_t wal_op_id_ = 0;                    // monotone frame op-id (informational/diagnostic)
  SegmentIoObserver *io_observer_ = nullptr;  // persistence-model harness hook (test only)

  // --- 2A appended-label transaction state ---
  // Immutable published snapshot (B-02). The single writer swaps the pointer under
  // label_snapshot_mutex_ (NOT the handle mutex); search copies it under the same
  // tiny mutex. Ordering "snapshot before committed" holds via the mutex
  // release/acquire plus search's committed acquire. Never null post-recovery.
  mutable std::mutex label_snapshot_mutex_;
  std::shared_ptr<const LabelBindings> label_snapshot_;
  std::map<PID, PidBinding> label_working_;  // recovery-only scratch (slot load + promotions)
  std::string label_slot_path_[2];           // <index>.labels.slot0 / .slot1
  int active_label_slot_ = 0;                // slot holding the persisted bindings
  uint64_t label_generation_ = 0;  // persisted-slot generation (bumped on content checkpoint)
  uint64_t label_count_ = 0;       // persisted-slot binding count (== superblock label_count)
  uint64_t label_checksum_ = 0;    // persisted-slot checksum (low 32 = crc32, high 32 = 0)
  // Label-slot content revision (design 3.1 / execution pitfall 2): PID reuse rebinds
  // {generation,label} at an UNCHANGED key count, so "count grew" no longer implies
  // "slot dirty". Bump on every published binding-set mutation (bundle commit, replay
  // promotion); checkpoint writes the inactive slot whenever it differs from the
  // persisted revision. In append-only mode a bump coincides with a count increase, so
  // this stays byte-identical to the old count gate (golden / 2A / 2B unchanged).
  uint64_t label_content_revision_ = 0;
  uint64_t persisted_label_content_revision_ = 0;
  uint64_t last_committed_txid_ = 0;       // running: strictly increases per committed bundle
  uint64_t applied_collection_op_id_ = 0;  // running: caller op watermark (2B idempotency basis)
  uint64_t base_committed_txid_ = 0;       // adopted base's persisted txid (case (a)/(b) split)
  uint64_t base_applied_op_id_ = 0;        // adopted base's persisted applied op id
  uint64_t base_num_points_ = 0;  // adopted base committed watermark (case (b) bound check)
  // Recovery staging: label_bind frames accumulated per tx_id until tx_publish.
  struct LabelBindStage {
    uint64_t row_op_id = 0;
    PID pid = 0;
    uint64_t label = 0;
  };
  std::unordered_map<uint64_t, std::vector<LabelBindStage>> staged_binds_;

  // --- 2C maintenance transaction (consolidate under enable_wal) ---
  // A consolidate epoch runs single-threaded (design/manual: parallel page workers
  // + the B-2C-05 concurrency protocol are documented follow-on hardening; the
  // serial path is the minimal-viable-correct W1). maintenance_active_ routes every
  // maintenance page RMW/read into a PRIVATE overlay so a concurrent search keeps
  // reading the committed state until the END-driven, seqlock-guarded install.
  // PID-reuse bundle reservation state (design 3.6 / codex B.2, checkpoint admission
  // B.6). kIdle outside a bundle; kReserving once the first FREE PID is popped; kBuilding
  // once the token set is fixed. checkpoint() admits only kIdle -- an all-reuse bundle
  // leaves the HWM unchanged, so allocated==committed no longer implies quiescent.
  // Dormant until the writer (step 5) calls reserve_bundle_pids.
  enum class BundleState { kIdle, kReserving, kBuilding };
  BundleState bundle_state_ = BundleState::kIdle;
  uint64_t reservation_count_ = 0;
  bool enable_pid_reuse_ = false;  // mirrors params_.enable_pid_reuse && enable_wal_ (opt-in reuse)
  // Writer-private bundle overlay (design B.3/B.4): the canonical reuse writer builds
  // every row / reverse edge / bundle spine into this PRIVATE page overlay so a
  // concurrent search reads only committed state until the kind=8-durable install into
  // the shared write cache. Set only by the single writer for the duration of one
  // commit_physical_bundle_canonical; null (and therefore byte-for-byte inert) on every
  // 2A/2B/maintenance/query path. `revealed` reveals reserved rows already built in THIS
  // bundle to the writer's own robust_prune/backlink/search seeding (writer_hidden), never
  // to the concurrent query face (which only reads the global hidden bitmap + committed).
  struct BundleInsertContext {
    uint64_t old_hwm = 0;
    std::unordered_set<PID> reserved;  // every reserved PID (reused gen>0 + dense append)
    std::unordered_set<PID> revealed;  // reserved PIDs already built in THIS bundle
    PID private_entry = kPidMax;       // writer-only search seed (first built row)
    PID last_built = kPidMax;          // bundle-spine tail (forced last_built -> current edge)
    std::unordered_map<size_t, std::vector<char>> pages;  // page_index -> resident overlay bytes
    std::unordered_set<size_t> dirty;                     // pages modified (final kind=1 + install)
    std::unordered_map<size_t, alaya::wal::FrameLocation> spilled;  // evicted page -> latest kind=1
    bool binds_durable = false;  // the kind=7 lane has been forced durable (before the 1st spill)
  };
  BundleInsertContext *bundle_ctx_ = nullptr;
  bool free_chain_rebuild_complete_ = true;  // false only mid-recovery; gates reserve_bundle_pids
  bool maintenance_active_ = false;
  bool maintenance_activating_ = false;  // an activation checkpoint is in flight (emit v3)
  bool pid_reuse_activating_ = false;    // a pid-reuse activation checkpoint is in flight
  uint64_t last_completed_consolidate_epoch_ = 0;  // adopted from base; advanced at END
  bool maintenance_activated_ = false;             // v3 maintenance features activated in the base
  bool pid_generation_activated_ = false;          // v3 pid-reuse feature activated (W2c)
  uint64_t maintenance_activation_gen_ = 0;  // superblock generation of the activation checkpoint
  uint64_t pid_reuse_activation_gen_ = 0;    // superblock gen when PID reuse activated (0 = never)
  // Private overlay: page_index -> resident bytes; and evicted pages -> latest kind=1
  // frame location (reload via WalFile::read_frame). Union = every touched page.
  std::unordered_map<size_t, std::vector<char>> maint_pages_;
  std::unordered_set<size_t> maint_dirty_;  // overlay pages actually modified (need install)
  // Resident bytes newer than their latest spill (or than disk if never spilled).
  // Reloading a latest spill for dependency reads does not enter this set.
  std::unordered_set<size_t> maint_resident_dirty_;
  std::unordered_map<size_t, alaya::wal::FrameLocation> maint_spilled_;
  uint64_t maint_epoch_ = 0;             // the in-flight epoch id
  PID maint_local_free_head_ = kPidMax;  // transaction-local free head (published at END)
  uint64_t maint_local_free_count_ = 0;
  // True only inside the maintenance BUILD window [BEGIN durable, END durable):
  // the last-line steal guard in write_at() poisons on ANY index/arena write in
  // this window (design section 9.1 / W1 acceptance: zero index/arena maintenance writes
  // before END).
  bool maint_in_build_phase_ = false;
};

}  // namespace alaya::laser
