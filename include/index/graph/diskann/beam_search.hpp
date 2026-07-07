// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file beam_search.hpp
 * @brief Disk search over the DiskANN layout — PQ beam search and No-PQ greedy.
 *
 * Two modes (design D2/D5), dispatched by cached_beam_search():
 *
 *   - PQ mode (use_pq): a beam that pops the closest unexpanded candidates from
 *     the frontier, serves cache hits from memory, and submits cache misses as
 *     async O_DIRECT page reads. Neighbor distances come from the per-query PQ
 *     table, so the frontier is ordered best-first cheaply. Optional rerank reads
 *     the top candidates' coordinates and re-scores them with exact L2.
 *
 *     Two scheduling variants (SearchParams::deterministic): the default
 *     pipelines I/O with compute (up to beam_width reads in flight, processed in
 *     completion order — fastest, but results vary with I/O timing); the opt-in
 *     deterministic variant uses a per-beam barrier (process in popped order,
 *     ~10-15% slower) so concurrent batch_search() matches sequential search()
 *     byte for byte.
 *
 *   - No-PQ mode: a faithful disk port of Vamana greedy search. Expanding a node
 *     reads each unvisited neighbor's coordinates and computes its *exact* L2
 *     distance, so the frontier is ordered by true distance and descends to the
 *     query. This costs more I/O than PQ (every discovered node is read) but
 *     needs no quantization and yields full-precision results. (A lazy
 *     FLT_MAX / parent-distance placeholder was measured to converge far too
 *     slowly — recall < 0.5 even reading 20% of the index — so the neighbor
 *     distances are computed eagerly, matching the in-memory VamanaGreedySearch
 *     model that design D5 names.)
 *
 *     Like PQ, No-PQ has two scheduling variants (SearchParams::deterministic).
 *     Because every distance is exact and the frontier's insert is order-
 *     independent in its final contents, both yield the same recall; they differ
 *     only in how neighbor reads are issued. deterministic = true (batched
 *     barrier) reads each expansion's unvisited neighbors as one concurrent batch
 *     and is byte-for-byte identical to a one-at-a-time synchronous read, so
 *     batch_search() == search(). The default pipelines reads across expansions
 *     (up to 2*beam_width in flight), maximising I/O/compute overlap at the cost
 *     of strict-greedy ordering (only tie-ordering of equally-distant nodes may
 *     differ, not recall).
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "index/graph/diskann/disk_layout.hpp"
#include "index/graph/diskann/disk_page_io.hpp"
#include "index/graph/diskann/node_cache.hpp"
#include "index/graph/diskann/pq_table.hpp"
#include "index/graph/diskann/search_scratch.hpp"
#include "index/graph/diskann/tombstone_bitmap.hpp"
#include "index/graph/laser/utils/aligned_file_reader.hpp"
#include "index/graph/vamana/robust_prune.hpp"
#include "simd/distance_l2.hpp"

namespace alaya::diskann {

/// Tunable runtime search parameters.
struct SearchParams {
  uint32_t search_list_size = 100;  ///< L: frontier capacity (>= top_k)
  uint32_t beam_width = 4;          ///< PQ mode: max async reads per beam step
  bool use_pq = true;               ///< use PQ approx distances (needs a PQTable)
  bool rerank = true;               ///< PQ only: re-score top candidates with exact L2
  uint32_t rerank_count = 0;        ///< PQ rerank pool size; 0 => top_k*3 (spec default)
  bool deterministic = false;       ///< Reproducible batch==sequential via a per-expansion
                                    ///< barrier (PQ: per-beam). Default off = async-pipelined
                                    ///< I/O (faster). Applies to both PQ and No-PQ modes.
};

/// Immutable per-index context the search borrows (no ownership).
struct SearchContext {
  AlignedFileReader *reader = nullptr;
  const DiskLayoutGeometry *geom = nullptr;
  const NodeCache *cache = nullptr;
  const PQTable *pq = nullptr;  ///< nullptr => index has no PQ
  uint32_t medoid = 0;
  uint64_t num_points = 0;

  // Tombstone-aware search (null for static indices).
  const TombstoneSnapshot *tombstone = nullptr;

  // Updatable indices: the shard page cache doubles as a unified buffer pool
  // (Yi semantics) — searches peek it before device reads and offer their
  // read pages back through the versioned fill protocol. nullptr keeps the
  // static-index behavior: BFS cache + device reads only.
  DiskPageIO *page_io = nullptr;
};

/// Optional instrumentation for tests / profiling.
struct SearchStats {
  uint64_t n_ios = 0;                ///< page reads issued to disk
  uint64_t n_cache_hits = 0;         ///< nodes served from the BFS cache
  uint64_t n_page_cache_hits = 0;    ///< nodes served from the update page cache
  uint64_t n_nodes_processed = 0;    ///< nodes whose exact distance was computed
  uint64_t n_rerank_reads = 0;       ///< extra synchronous reads for PQ rerank
  uint64_t n_waits = 0;              ///< get_events calls in the async pipe
  uint64_t inflight_sum = 0;         ///< inflight reads before each wait (depth = sum/waits)
  uint64_t wait_us = 0;              ///< time blocked in get_events
  uint64_t proc_us = 0;              ///< time in process_node (CPU)
  uint64_t setup_us = 0;             ///< per-query snapshot + scratch setup
  uint64_t peek_us = 0;              ///< time in search_peek_page
  uint64_t fillpg_us = 0;            ///< time in search_fill_page
  std::vector<uint32_t> read_order;  ///< ids in the order first processed
};

/**
 * @brief Scan a node's neighbors and insert the unvisited ones into the frontier
 *        with their PQ approximate distances (PQ mode).
 *
 * The frontier's bounded insert prunes neighbors that cannot beat the current
 * worst entry. Already-visited neighbors and out-of-range ids are skipped.
 * Extracted as a free function so the contract is unit-testable in isolation.
 */
inline void scan_and_insert_neighbors(alaya::vamana::NeighborPriorityQueue &retset,
                                      VisitedBitset &visited,
                                      const uint32_t *nbrs,
                                      uint32_t n_nbrs,
                                      const PQTable &pq,
                                      const float *pq_table,
                                      uint64_t num_points,
                                      const TombstoneSnapshot *tombstone = nullptr) {
  // Filter first, then score all survivors with one batched PQ pass
  // (pq_distance_batch): per-neighbor pq_distance() stalls on a dependent
  // random 32 B code fetch per call — the eval hot spot at 100M scale. The
  // filter order, the accumulation order per point, and the insertion order
  // all match the former per-neighbor loop, so results are bit-identical.
  constexpr uint32_t kScanTile = 128;  // >= max node degree in practice; loop-safe beyond
  uint32_t ids[kScanTile];
  float dists[kScanTile];
  uint32_t cnt = 0;
  const auto flush = [&]() {
    pq.pq_distance_batch(ids, cnt, pq_table, dists);
    for (uint32_t i = 0; i < cnt; ++i) {
      retset.insert(alaya::vamana::Neighbor(ids[i], dists[i]));
    }
    cnt = 0;
  };
  for (uint32_t k = 0; k < n_nbrs; ++k) {
    const uint32_t m = nbrs[k];
    if (m >= num_points) {
      continue;  // defensive: ignore corrupt neighbor ids
    }
    if (tombstone != nullptr && tombstone->is_deleted(m)) {
      continue;
    }
    if (!visited.test_and_set(m)) {
      continue;
    }
    ids[cnt] = m;
    if (++cnt == kScanTile) {
      flush();
    }
  }
  if (cnt != 0) {
    flush();
  }
}

/**
 * @brief No-PQ disk greedy search: a disk port of Vamana greedy search.
 *
 * Expanding a node reads each unvisited neighbor's coordinates and computes its
 * exact L2 distance, so the frontier (NeighborPriorityQueue of capacity L) is
 * ordered by true distance and the search descends to the query. Each node is
 * read at most once; its neighbor list is cached for when it is expanded.
 *
 * Two I/O scheduling variants share this code (SearchParams::deterministic):
 * a batched per-expansion barrier (byte-exact, batch == sequential) and the
 * default async pipeline (reads overlap across expansions). Because all
 * distances are exact and the frontier insert is order-independent, both give
 * identical recall — see the file header for the contract.
 */
inline std::vector<std::pair<uint32_t, float>> disk_greedy_search(const SearchContext &ctx,
                                                                  const float *query,
                                                                  uint32_t top_k,
                                                                  const SearchParams &params,
                                                                  ThreadData &td,
                                                                  SearchStats *stats) {
  AlignedFileReader &reader = *ctx.reader;
  const DiskLayoutGeometry &geom = *ctx.geom;
  const NodeCache &cache = *ctx.cache;
  const uint64_t dim = geom.dim;
  const uint64_t page_size = geom.page_size;
  const auto l2 = alaya::simd::get_l2_sqr_func();
  // Page slots available in the sector scratch (alloc_scratch sizes it for
  // 2 * beam_width pages); both schedulers cap their in-flight reads at this.
  const uint64_t n_slots = std::max<uint64_t>(1, td.sector_scratch_bytes / page_size);

  const size_t list_size = std::max<size_t>(params.search_list_size, top_k);
  td.reset_query(list_size);
  auto &frontier = td.retset;
  auto &visited = td.visited_bits;

  // Absorb a freshly-read node: cache its neighbor list and insert it into the
  // frontier with its exact L2 distance (coords are co-located in the record).
  auto absorb = [&](uint32_t id, const char *rec) {
    NodeRecordView view{rec, dim};
    const float distance = l2(query, view.coords(), dim);
    const auto insert_result = frontier.insert_with_result(alaya::vamana::Neighbor(id, distance));
    if (insert_result.evicted) {
      td.release_cached_neighbors(insert_result.evicted_id);
    }
    if (insert_result.inserted) {
      td.cache_neighbors(id, view.nbrs(), view.n_nbrs());
    }
    if (stats != nullptr) {
      stats->read_order.push_back(id);
      stats->n_nodes_processed++;
    }
  };

  NodeCache::Lookup seed_hit;

  // Unified-pool view (updatable indices): peek the update page cache before
  // any device read and fold read pages back in afterwards. Fill must run
  // BEFORE the record is parsed — a conflicting writer refreshes the buffer.
  DiskPageIO *page_io = ctx.page_io;

  // Synchronously read one node into scratch slot 0 (used only to seed the
  // medoid); returns a pointer to its record (cache or scratch).
  auto read_seed = [&](uint32_t id) -> const char * {
    seed_hit = cache.lookup_record(id);
    if (seed_hit) {
      if (stats != nullptr) {
        stats->n_cache_hits++;
      }
      return seed_hit.get();
    }
    const uint64_t off = geom.get_page_offset(id);
    uint64_t version = 0;
    if (page_io != nullptr && page_io->search_peek_page(off, td.sector_scratch, &version)) {
      if (stats != nullptr) {
        stats->n_page_cache_hits++;
      }
      return td.sector_scratch + geom.offset_to_node(id);
    }
    std::vector<AlignedRead> r1;
    r1.emplace_back(off, page_size, id, td.sector_scratch);
    reader.read(r1, td.ctx_);
    if (page_io != nullptr) {
      page_io->search_fill_page(off, td.sector_scratch, version);
    }
    if (stats != nullptr) {
      stats->n_ios++;
    }
    return td.sector_scratch + geom.offset_to_node(id);
  };

  // IP-DiskANN: tombstoned nodes are skipped (graph repaired at delete time).
  const TombstoneSnapshot *tomb = ctx.tombstone;
  auto consider = [&](uint32_t m, auto &&emit) {
    if (m >= ctx.num_points || !visited.test_and_set(m)) {
      return;
    }
    if (tomb != nullptr && tomb->is_deleted(m)) {
      return;
    }
    emit(m);
  };

  visited.set(ctx.medoid);
  absorb(ctx.medoid, read_seed(ctx.medoid));

  std::vector<AlignedRead> reqs;

  if (params.deterministic) {
    // Batched barrier: expand strictly closest-first. For each expansion, read
    // ALL unvisited neighbors concurrently (chunked to n_slots), wait for the
    // chunk, then absorb in neighbor-list order. Identical insert sequence to a
    // one-at-a-time synchronous read, so concurrent batch_search() reproduces
    // sequential search() byte for byte — it only overlaps the up-to-R reads of
    // each expansion instead of serialising them.
    std::vector<uint32_t> todo;
    std::vector<uint32_t> chunk_ids;
    std::vector<const char *> chunk_recs;
    std::vector<NodeCache::Lookup> chunk_hits;
    while (frontier.has_unexpanded_node()) {
      const uint32_t x = frontier.closest_unexpanded().id;
      const NeighborScratchView nbrs = td.cached_neighbors(x);
      todo.clear();
      for (const uint32_t m : nbrs) {
        consider(m, [&](uint32_t id) {
          todo.push_back(id);
        });
      }
      for (size_t off = 0; off < todo.size(); off += n_slots) {
        const size_t end = std::min<size_t>(off + n_slots, todo.size());
        reqs.clear();
        td.io_req_versions.clear();
        chunk_ids.clear();
        chunk_recs.clear();
        chunk_hits.clear();
        chunk_hits.reserve(end - off);
        uint64_t slot = 0;
        for (size_t i = off; i < end; ++i) {
          const uint32_t m = todo[i];
          NodeCache::Lookup cached = cache.lookup_record(m);
          if (cached) {
            if (stats != nullptr) {
              stats->n_cache_hits++;
            }
            chunk_ids.push_back(m);
            chunk_recs.push_back(cached.get());
            chunk_hits.push_back(std::move(cached));
            continue;
          }
          char *buf = td.sector_scratch + (slot++) * page_size;
          const uint64_t page_off = geom.get_page_offset(m);
          uint64_t version = 0;
          if (page_io != nullptr && page_io->search_peek_page(page_off, buf, &version)) {
            if (stats != nullptr) {
              stats->n_page_cache_hits++;
            }
            chunk_ids.push_back(m);
            chunk_recs.push_back(buf + geom.offset_to_node(m));
            continue;  // slot consumed, no device read
          }
          chunk_ids.push_back(m);
          chunk_recs.push_back(buf + geom.offset_to_node(m));
          reqs.emplace_back(page_off, page_size, m, buf);
          td.io_req_versions.push_back(version);
        }
        if (!reqs.empty()) {
          reader.read(reqs, td.ctx_);  // blocking; kernel runs the batch concurrently
          if (page_io != nullptr) {
            for (size_t r = 0; r < reqs.size(); ++r) {
              page_io->search_fill_page(reqs[r].offset,
                                        static_cast<char *>(reqs[r].buf),
                                        td.io_req_versions[r]);
            }
          }
          if (stats != nullptr) {
            stats->n_ios += reqs.size();
          }
        }
        for (size_t j = 0; j < chunk_ids.size(); ++j) {
          absorb(chunk_ids[j], chunk_recs[j]);
        }
      }
    }
  } else {
    // Async pipeline (default): keep up to n_slots reads in flight across
    // expansions. `pending` holds discovered (visited-marked) ids awaiting a slot;
    // ThreadData's inflight array tracks scratch slots and record pointers. Processing
    // follows I/O completion order, so the strict-greedy expansion order is
    // relaxed (results may differ only in tie-ordering of equally-distant nodes).
    std::vector<uint64_t> free_slots;
    free_slots.reserve(n_slots);
    for (uint64_t s = 0; s < n_slots; ++s) {
      free_slots.push_back(s);
    }
    std::deque<uint32_t> pending;
    std::vector<AlignedReadEvent> evts;

    // Expand closest-unexpanded nodes into `pending` until there is enough queued
    // to fill the free slots (relaxed greedy: may expand before a closer in-flight
    // node lands). closest_unexpanded() advances the cursor monotonically, so this
    // always makes progress and cannot loop forever.
    auto refill_pending = [&]() {
      while (pending.size() < free_slots.size() && frontier.has_unexpanded_node()) {
        const uint32_t x = frontier.closest_unexpanded().id;
        const NeighborScratchView nbrs = td.cached_neighbors(x);
        for (const uint32_t m : nbrs) {
          consider(m, [&](uint32_t id) {
            pending.push_back(id);
          });
        }
      }
    };

    // Drain `pending` into free slots: cache hits absorbed inline (no slot), misses
    // submitted as O_DIRECT reads.
    if (page_io != nullptr) {
      td.ensure_peek_scratch(page_size);
    }
    auto submit_pending = [&]() {
      reqs.clear();
      while (!free_slots.empty() && !pending.empty()) {
        const uint32_t m = pending.front();
        pending.pop_front();
        NodeCache::Lookup cached = cache.lookup_record(m);
        if (cached) {
          if (stats != nullptr) {
            stats->n_cache_hits++;
          }
          absorb(m, cached.get());
          continue;
        }
        const uint64_t page_off = geom.get_page_offset(m);
        uint64_t version = 0;
        if (page_io != nullptr && page_io->search_peek_page(page_off, td.peek_scratch, &version)) {
          if (stats != nullptr) {
            stats->n_page_cache_hits++;
          }
          absorb(m, td.peek_scratch + geom.offset_to_node(m));
          continue;
        }
        const uint64_t slot = free_slots.back();
        free_slots.pop_back();
        char *buf = td.sector_scratch + slot * page_size;
        td.set_inflight(slot, m, buf + geom.offset_to_node(m));
        td.slot_versions[slot] = version;
        reqs.emplace_back(page_off, page_size, m, buf);
      }
      if (!reqs.empty()) {
        reader.submit_reqs(reqs, td.ctx_);
        if (stats != nullptr) {
          stats->n_ios += reqs.size();
        }
      }
    };

    refill_pending();
    submit_pending();
    // Continue while any work remains: a read in flight, a queued read, or an
    // unexpanded frontier node (the latter matters when cache hits add nodes
    // without ever entering `inflight`).
    while (td.has_inflight() || !pending.empty() || frontier.has_unexpanded_node()) {
      if (td.has_inflight()) {
        reader.get_events(td.ctx_, 1, evts);  // get_events clears + fills `evts`
        for (const auto &e : evts) {
          if (e.result != static_cast<int64_t>(page_size)) {
            throw std::runtime_error("disk_greedy_search: short/failed read, result=" +
                                     std::to_string(e.result));
          }
          InFlightSlot completed;
          if (!td.remove_inflight(static_cast<uint32_t>(e.id), completed)) {
            continue;  // defensive: completion for an id we are not tracking
          }
          if (page_io != nullptr) {
            page_io->search_fill_page(geom.get_page_offset(static_cast<uint32_t>(e.id)),
                                      td.sector_scratch + completed.page_slot * page_size,
                                      td.slot_versions[completed.page_slot]);
          }
          absorb(static_cast<uint32_t>(e.id), completed.record);
          free_slots.push_back(completed.page_slot);
        }
      }
      refill_pending();
      submit_pending();
    }
  }

  std::vector<std::pair<uint32_t, float>> out;
  out.reserve(std::min<size_t>(frontier.size(), top_k));
  for (size_t i = 0; i < frontier.size() && out.size() < top_k; ++i) {
    const uint32_t id = frontier[i].id;
    if (tomb != nullptr && tomb->is_deleted(id)) {
      continue;
    }
    out.emplace_back(id, frontier[i].distance);
  }
  return out;
}

/**
 * @brief Run the disk search for one query (PQ beam or No-PQ greedy).
 *
 * @return Up to @p top_k (internal_id, distance) pairs sorted by ascending
 *         distance. Distances are exact L2-squared except for PQ no-rerank
 *         results on nodes that were never read (those carry their PQ
 *         approximation).
 */
inline std::vector<std::pair<uint32_t, float>> cached_beam_search(const SearchContext &ctx,
                                                                  const float *query,
                                                                  uint32_t top_k,
                                                                  const SearchParams &params,
                                                                  ThreadData &td,
                                                                  SearchStats *stats = nullptr) {
  const bool use_pq = params.use_pq && (ctx.pq != nullptr);
  if (!use_pq) {
    return disk_greedy_search(ctx, query, top_k, params, td, stats);
  }

  AlignedFileReader &reader = *ctx.reader;
  const DiskLayoutGeometry &geom = *ctx.geom;
  const NodeCache &cache = *ctx.cache;
  const PQTable &pq = *ctx.pq;
  const uint64_t dim = geom.dim;
  const uint64_t page_size = geom.page_size;
  const auto l2 = alaya::simd::get_l2_sqr_func();

  const size_t list_size = std::max<size_t>(params.search_list_size, top_k);
  td.reset_query(list_size);
  auto &retset = td.retset;
  auto &visited = td.visited_bits;

  pq.preprocess_query(query, td.pq_table.data(), td.pq_qres.data());
  const float *pq_table = td.pq_table.data();

  // Seed from the medoid (first node expanded — spec scenario).
  visited.set(ctx.medoid);
  retset.insert(alaya::vamana::Neighbor(ctx.medoid, pq.pq_distance(ctx.medoid, pq_table)));

  const uint64_t beam = std::max<uint64_t>(1, params.beam_width);

  // Process a freshly-read node: compute its exact L2 (the coords are co-located
  // on the sector we just read) and insert its unvisited neighbors into the
  // frontier with PQ approximate distances. Shared by both search paths.
  auto process_node = [&](uint32_t id, const char *rec) {
    const auto tp0 = std::chrono::steady_clock::now();
    NodeRecordView view{rec, dim};
    td.set_exact_dist(id, l2(query, view.coords(), dim));
    if (stats != nullptr) {
      stats->read_order.push_back(id);
      stats->n_nodes_processed++;
    }
    scan_and_insert_neighbors(retset,
                              visited,
                              view.nbrs(),
                              view.n_nbrs(),
                              pq,
                              pq_table,
                              ctx.num_points,
                              ctx.tombstone);
    if (stats != nullptr) {
      stats->proc_us += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                  std::chrono::steady_clock::now() - tp0)
                                                  .count());
    }
  };

  auto &reqs = td.io_reqs;
  auto &evts = td.io_events;
  reqs.clear();
  evts.clear();

  // Unified-pool view (updatable indices) — see disk_greedy_search.
  DiskPageIO *page_io = ctx.page_io;
  // Page slots available in the sector scratch; page-cache hits borrow slots
  // in the barrier mode below, so the batch is bounded by them.
  const uint64_t n_slots = std::max<uint64_t>(1, td.sector_scratch_bytes / page_size);

  if (params.deterministic) {
    // Deterministic per-beam barrier: pop the closest unexpanded candidates, read
    // the cache misses together, wait for ALL of them, then process the batch in
    // popped order. Processing order is independent of I/O completion timing, so
    // concurrent batch_search() matches sequential search() byte for byte. Opt-in
    // (DiskANNSearchParams::deterministic) — it forgoes the cross-beam I/O/compute
    // overlap of the default path and so runs ~10-15% slower.
    std::vector<uint32_t> batch;
    std::vector<const char *> batch_recs;
    std::vector<NodeCache::Lookup> batch_hits;
    while (retset.has_unexpanded_node()) {
      batch.clear();
      batch_recs.clear();
      batch_hits.clear();
      batch_hits.reserve(beam);
      reqs.clear();
      td.io_req_versions.clear();
      uint64_t slot = 0;
      while (retset.has_unexpanded_node() && reqs.size() < beam && slot < n_slots) {
        const uint32_t id = retset.closest_unexpanded().id;
        batch.push_back(id);
        NodeCache::Lookup cached = cache.lookup_record(id);
        if (cached) {
          batch_recs.push_back(cached.get());
          batch_hits.push_back(std::move(cached));
          if (stats != nullptr) {
            stats->n_cache_hits++;
          }
          continue;
        }
        char *buf = td.sector_scratch + (slot++) * page_size;
        const uint64_t page_off = geom.get_page_offset(id);
        uint64_t version = 0;
        if (page_io != nullptr && page_io->search_peek_page(page_off, buf, &version)) {
          if (stats != nullptr) {
            stats->n_page_cache_hits++;
          }
          batch_recs.push_back(buf + geom.offset_to_node(id));
          continue;  // slot consumed, no device read
        }
        batch_recs.push_back(buf + geom.offset_to_node(id));
        reqs.emplace_back(page_off, page_size, id, buf);
        td.io_req_versions.push_back(version);
      }

      if (!reqs.empty()) {
        reader.submit_reqs(reqs, td.ctx_);
        reader.get_events(td.ctx_, static_cast<int>(reqs.size()), evts);
        if (stats != nullptr) {
          stats->n_ios += reqs.size();
        }
        for (const auto &e : evts) {
          if (e.result != static_cast<int64_t>(page_size)) {
            throw std::runtime_error("cached_beam_search: short/failed read, result=" +
                                     std::to_string(e.result));
          }
        }
        if (page_io != nullptr) {
          for (size_t r = 0; r < reqs.size(); ++r) {
            page_io->search_fill_page(reqs[r].offset,
                                      static_cast<char *>(reqs[r].buf),
                                      td.io_req_versions[r]);
          }
        }
      }

      for (size_t i = 0; i < batch.size(); ++i) {
        process_node(batch[i], batch_recs[i]);
      }
    }
  } else {
    // Async pipelined beam (default): keep up to `beam` reads in flight and
    // process each as it completes, refilling the pipe immediately, so disk I/O
    // overlaps with distance computation. Processing follows I/O completion order,
    // so results are NOT bitwise-reproducible across runs/threads (set
    // deterministic for that). Thread-safety, correctness and recall are
    // unaffected — only the tie-ordering of equally-good candidates differs.
    auto &free_slots = td.free_page_slots;
    free_slots.clear();
    for (uint64_t s = 0; s < beam; ++s) {
      free_slots.push_back(s);
    }
    // Pop closest-unexpanded candidates, serving cache hits inline and submitting
    // cache misses into free scratch slots. Stops when the slots are full or the
    // frontier has no unexpanded node left.
    if (page_io != nullptr) {
      td.ensure_peek_scratch(page_size);
    }
    auto fill_pipe = [&]() {
      reqs.clear();
      while (!free_slots.empty() && retset.has_unexpanded_node()) {
        const uint32_t id = retset.closest_unexpanded().id;
        NodeCache::Lookup cached = cache.lookup_record(id);
        if (cached) {
          if (stats != nullptr) {
            stats->n_cache_hits++;
          }
          process_node(id, cached.get());
          continue;
        }
        const uint64_t page_off = geom.get_page_offset(id);
        uint64_t version = 0;
        bool peeked = false;
        if (page_io != nullptr) {
          const auto tk0 = std::chrono::steady_clock::now();
          peeked = page_io->search_peek_page(page_off, td.peek_scratch, &version);
          if (stats != nullptr) {
            stats->peek_us +=
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                          std::chrono::steady_clock::now() - tk0)
                                          .count());
          }
        }
        if (peeked) {
          if (stats != nullptr) {
            stats->n_page_cache_hits++;
          }
          process_node(id, td.peek_scratch + geom.offset_to_node(id));
          continue;
        }
        const uint64_t slot = free_slots.back();
        free_slots.pop_back();
        char *buf = td.sector_scratch + slot * page_size;
        td.set_inflight(slot, id, buf + geom.offset_to_node(id));
        td.slot_versions[slot] = version;
        reqs.emplace_back(page_off, page_size, id, buf);
      }
      if (!reqs.empty()) {
        reader.submit_reqs(reqs, td.ctx_);
        if (stats != nullptr) {
          stats->n_ios += reqs.size();
        }
      }
    };

    fill_pipe();
    while (td.has_inflight()) {
      // Wait for exactly one completion (get_events clears and fills `evts`). The
      // other in-flight reads keep the disk busy while we process this one — that
      // is the I/O/compute overlap. NOTE: poll_events/get_events both clear their
      // out-vector, so they must not share one; we use get_events alone here.
      const auto tw0 = std::chrono::steady_clock::now();
      if (stats != nullptr) {
        stats->n_waits++;
        stats->inflight_sum += beam - free_slots.size();
      }
      reader.get_events(td.ctx_, 1, evts);
      if (stats != nullptr) {
        stats->wait_us +=
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now() - tw0)
                                      .count());
      }
      for (const auto &e : evts) {
        if (e.result != static_cast<int64_t>(page_size)) {
          throw std::runtime_error("cached_beam_search: short/failed read, result=" +
                                   std::to_string(e.result));
        }
        InFlightSlot completed;
        if (!td.remove_inflight(static_cast<uint32_t>(e.id), completed)) {
          continue;  // defensive: completion for an id we are not tracking
        }
        if (page_io != nullptr) {
          const auto tf0 = std::chrono::steady_clock::now();
          page_io->search_fill_page(geom.get_page_offset(static_cast<uint32_t>(e.id)),
                                    td.sector_scratch + completed.page_slot * page_size,
                                    td.slot_versions[completed.page_slot]);
          if (stats != nullptr) {
            stats->fillpg_us +=
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                          std::chrono::steady_clock::now() - tf0)
                                          .count());
          }
        }
        process_node(static_cast<uint32_t>(e.id), completed.record);
        free_slots.push_back(completed.page_slot);
      }
      fill_pipe();
    }
  }

  // ---------------- Result extraction (PQ) ----------------
  std::vector<std::pair<uint32_t, float>> out;
  auto &rerank_req = td.rerank_reqs;
  rerank_req.resize(1);
  auto is_live = [&](uint32_t id) {
    return ctx.tombstone == nullptr || !ctx.tombstone->is_deleted(id);
  };

  auto read_exact_sync = [&](uint32_t id) -> float {
    const float known = td.exact_dist(id);
    if (!ThreadData::is_missing_exact(known)) {
      return known;
    }
    char *buf = td.sector_scratch;
    const uint64_t page_off = geom.get_page_offset(id);
    uint64_t version = 0;
    const bool peeked = page_io != nullptr && page_io->search_peek_page(page_off, buf, &version);
    if (!peeked) {
      rerank_req[0] = {page_off, page_size, id, buf};
      reader.read(rerank_req, td.ctx_);
      if (page_io != nullptr) {
        page_io->search_fill_page(page_off, buf, version);
      }
      if (stats != nullptr) {
        stats->n_rerank_reads++;
      }
    } else if (stats != nullptr) {
      stats->n_page_cache_hits++;
    }
    NodeRecordView view{buf + geom.offset_to_node(id), dim};
    const float ex = l2(query, view.coords(), dim);
    td.set_exact_dist(id, ex);
    return ex;
  };

  if (params.rerank) {
    // Re-score the top PQ candidates with exact L2.
    const size_t want =
        params.rerank_count > 0 ? params.rerank_count : static_cast<size_t>(top_k) * 3;
    out.reserve(std::min<size_t>(retset.size(), want));
    for (size_t i = 0; i < retset.size() && out.size() < want; ++i) {
      const uint32_t id = retset[i].id;
      if (!is_live(id)) {
        continue;
      }
      out.emplace_back(id, read_exact_sync(id));
    }
  } else {
    // Top candidates ranked by PQ distance (exact where known during traversal).
    out.reserve(std::min<size_t>(retset.size(), static_cast<size_t>(top_k)));
    for (size_t i = 0; i < retset.size() && out.size() < top_k; ++i) {
      const uint32_t id = retset[i].id;
      if (!is_live(id)) {
        continue;
      }
      const float exact = td.exact_dist(id);
      out.emplace_back(id, !ThreadData::is_missing_exact(exact) ? exact : retset[i].distance);
    }
  }

  std::sort(out.begin(), out.end(), [](const auto &a, const auto &b) {
    return a.second < b.second || (a.second == b.second && a.first < b.first);
  });
  if (out.size() > top_k) {
    out.resize(top_k);
  }
  return out;
}

}  // namespace alaya::diskann
