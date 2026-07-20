// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include "index/graph/laser/common.hpp"
#include "index/graph/laser/qg/segment_op_wal.hpp"

namespace alaya::laser {

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
  uint64_t patch_intents_prepared = 0;
  uint64_t patch_intents_applied = 0;
  uint64_t patch_intent_stale_fallbacks = 0;
  uint64_t full_recomputes = 0;
  uint64_t forced_links = 0;  // inserts whose only backlink came from the force policy
  uint64_t evict_tel_samples = 0;
  uint64_t evict_tel_agree = 0;
  std::array<uint64_t, 4> evict_tel_regret{};  // exact rank: 0, 1, 2, 3+
  double evict_tel_relerr_sum = 0;
  uint64_t consolidated_rows = 0;
  uint64_t spliced_slots = 0;
  uint64_t ghosted_slots = 0;
  uint64_t bloom_scan_rows = 0;
  uint64_t bloom_candidate_rows = 0;
  uint64_t bloom_scan_us = 0;
  uint64_t bloom_row_us = 0;
  uint64_t bloom_finalize_us = 0;
  uint64_t freed_slots = 0;
  uint64_t reused_slots = 0;
  uint64_t gardened_rows = 0;
  uint64_t garden_pump_links = 0;
  uint64_t garden_us = 0;
  uint64_t garden_selected_turnover_sum = 0;
  uint64_t garden_selected_turnover_rows = 0;
  uint64_t garden_all_turnover_sum = 0;
  uint64_t garden_all_turnover_rows = 0;
  uint64_t maintenance_peak_pool_pages = 0;
  uint64_t maintenance_peak_overlay_pages = 0;
  uint64_t maintenance_page_frames = 0;
  uint64_t maintenance_page_frame_bytes = 0;
  uint64_t maintenance_last_preflight_page_frames = 0;
  uint64_t maintenance_last_preflight_wal_bytes = 0;
  uint64_t garden_skipped = 0;
};

struct TurnoverSummary {
  uint64_t sum = 0;
  uint64_t rows = 0;
  uint16_t p50 = 0;
  uint16_t p99 = 0;
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
  enum class Policy { kLowIndegree, kRandom, kTurnover } policy = Policy::kLowIndegree;
  bool pump_only = false;  // true: skip RobustPrune rewrite, only do the search + pump phase
                           // (adds incoming edges to low-indegree nodes without touching their
                           // out-edges; avoids the degree-reduction effect of pruning)
};

// Immutable appended-label snapshot (2A, B-02). Maps appended PIDs (pid >=
// base_count) to explicit labels; PIDs absent from the map fall back to identity
// at the segment layer. Published via std::atomic<shared_ptr> and never mutated
// after publish, so lock-free search translates labels off a consistent snapshot.
// The map is ordered so slot serialization is deterministic (ascending pid).
// A PID's durable incarnation: pid_generation (0 for a never-reused / append-only
// PID, else strictly old+1 per reuse) plus its logical label. The 16-byte label
// slot wire already carries {pid u32, pid_generation u32, label u64} (2A wrote the
// generation as 0 and rejected non-zero); W2 activates it (design section 3.1).
struct PidBinding {
  uint32_t pid_generation = 0;
  uint64_t label = 0;
  bool operator==(const PidBinding &) const = default;
};

// The reuse-safe identity of a committed row: (pid, pid_generation). commit returns
// these so the caller (2B adapter) binds label->token instead of guessing a dense
// range, and tombstone(token) can reject a stale incarnation via an ABA check.
struct PidToken {
  PID pid = kPidMax;  // W2c: kPidMax sentinel -- 0 is a valid PID, so a default token
                      // must not alias PID 0 (design 3.3 / codex B.1 / JC-16).
  uint32_t pid_generation = 0;
  bool operator==(const PidToken &) const = default;
};

// The result of commit_physical_bundle once PID reuse is active (design 3.3): the
// per-row (pid, pid_generation) tokens in row_op order plus the append high-water
// marks. The 2B adapter binds label->token from `rows` instead of guessing a dense
// [old_hwm, new_hwm) range (mixed / all-reuse bundles are not dense). Dormant until
// the reuse writer lands (steps 4-7); the append path still returns a pair today.
struct PhysicalBundleResult {
  std::vector<PidToken> rows;  // one per input row, in row_op_id order
  uint64_t old_hwm = 0;
  uint64_t new_hwm = 0;
};

struct LabelBindings {
  std::map<PID, PidBinding> bindings;  // pid -> {generation, label}

  // Label-only lookup (the historical API): a pointer to the binding's label, or
  // null. Consumers that need the incarnation use find_binding().
  [[nodiscard]] const uint64_t *find(PID pid) const {
    const auto it = bindings.find(pid);
    return it == bindings.end() ? nullptr : &it->second.label;
  }
  [[nodiscard]] const PidBinding *find_binding(PID pid) const {
    const auto it = bindings.find(pid);
    return it == bindings.end() ? nullptr : &it->second;
  }
};

struct UpdateParams {
  size_t ef_insert = 100;
  float alpha = 1.2F;
  size_t prune_pool_cap = 300;  // candidates fed to RobustPrune
  enum class Backlink { kNone, kEvict, kExactEvict, kAlphaEvict, kFullPrune };
  Backlink backlink_mode = Backlink::kAlphaEvict;
  size_t alpha_check_max = 16;  // captured neighbors tested per reverse edge
  double evict_telemetry = 0;   // kEvict decision sampling probability
  double evict_margin = 0;      // require (worst-new) > margin*abs(worst); 0 preserves behavior
  size_t max_points = 0;        // page-version table capacity; 0 -> 2*N + 4096
  size_t splice_rerank = 4;     // consolidation: FastScan-recalled candidates reranked exactly
  bool splice_enabled = true;   // false: purge dead edges without splice reconnection
  bool maintain_indegree = false;
  bool maintain_turnover = false;  // optional saturated per-row neighborhood replacement count
  bool direct_io = false;          // route writes through a dedicated O_DIRECT fd.
                                   // P0.1 verdict: synchronous per-patch DIO writes LOSE
                                   // to buffered pwrite (5.9k vs 7.4k inserts/s @64T) —
                                   // the kernel page cache is already a write-back cache.
                                   // Keep buffered until the user-space dirty-page cache
                                   // (P0.2) moves writes off the hot path; the DIO fd is
                                   // meant for its batched flush.
  bool write_cache = true;         // absorb page RMWs in a resident page pool (see cache_cap_pages)
  bool stage_backlinks = false;    // false: patch reverse edges inline at insert time (they land
                                   // in the pool, so same-page coalescing happens regardless, the
                                   // insert threads overlap patch CPU with search IO, and the
                                   // kAlphaEvict capture-pool check works); true: stage per target
                                   // and drain at the batch barrier (phase-separated variant)
  bool preencode_patch_intents = false;    // QPatch phase 1: compute per-edge RaBitQ codes/factors
                                           // from captured raw rows before taking the target lock
  size_t cache_cap_pages = 1U << 20;       // pool high watermark (pages). Hub rows are patched by
                                           // many batches; keeping pages resident coalesces those
                                           // writes across batches (physical writes happen only on
                                           // watermark eviction, consolidate() and finalize())
  size_t maintenance_evict_stride = 4096;  // maintenance batch size in pages/rows before a
                                           // high->low pool check; 0 preserves phase-boundary-only
                                           // eviction
  size_t consolidate_every = 1;            // consolidate every N rounds; 0 = never
  double garden_churn_threshold = 0.0;     // fraction of committed; garden() is a no-op until
                                           // accumulated churn (tombstones + inserts) since the
                                           // last garden reaches threshold * committed.
                                           // 0 = garden every call (legacy behavior).

  // --- durable in-place updates: segment op-WAL (G1) ---
  // Off by default so every existing test/bench is byte-for-byte unchanged.
  // When true, the updater logs an after-image WAL and recovers on reopen; the
  // G1 minimal safe scope forbids PID reuse/reclaim/consolidate/garden/bloom
  // (unified-wal-vocabulary.md clause A) — those paths throw under enable_wal.
  bool enable_wal = false;
  // 2C PID reuse (design section 3 / codex B.1): opt-in canonical prebind bundles
  // that reuse freed PIDs. Requires enable_wal (the free-list is only durable under
  // the WAL). Off by default so every 2A/2B path is byte-for-byte unchanged; the
  // first reuse-enabled bundle activates a v3 pid-generation base (design 7.2).
  bool enable_pid_reuse = false;
  // Crash-matrix injection: invoked at labelled lifecycle points; empty in prod.
  std::function<void(SegmentOpFailPoint)> failpoint_hook{};
  // Low-level pwrite fault injection. Invoked immediately before an index write;
  // empty in production. Kept separate from SegmentOpFailPoint so test coverage
  // does not add or renumber any WAL lifecycle value.
  std::function<void(uint64_t, size_t)> before_index_write_hook{};
  // Persistence-model (power-loss) harness hook; null in prod (zero overhead).
  SegmentIoObserver *io_observer = nullptr;
};

class QGUpdater;

}  // namespace alaya::laser
