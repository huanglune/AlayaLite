// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file beam_search_async.hpp
 * @brief Coroutine variant of the No-PQ disk greedy search for the update path.
 *
 * disk_greedy_search_async() is disk_greedy_search() with the blocking libaio
 * reads replaced by io_uring reactor waves: each expansion gathers the misses
 * of one neighbor list into a single BatchRead and co_awaits it, so the pool
 * thread runs other update coroutines while the wave is in flight (the sync
 * search parks the thread in io_getevents instead — with pool == cores that
 * blocked thread is exactly the CPU the reactor's reaper and the other update
 * coroutines needed).
 *
 * Scheduling contract: cache hits are absorbed inline before the wave is
 * submitted (NodeCache::Lookup guards must never be held across a suspension)
 * and misses are absorbed in neighbor-list order once the wave lands. All
 * distances are exact, so like the two sync schedulers this changes only
 * tie-ordering, not recall; with no cache hits the absorb sequence — and
 * therefore the result — is byte-identical to the sync deterministic variant.
 *
 * Reads go through the index-neutral PageReader awaitable. DiskPageIO's own
 * O_DIRECT fd remains private to the unchanged update/write path.
 * When ctx.page_io is set, both searches peek the update shard cache before
 * each device read and offer the pages they read back through the versioned
 * fill protocol — the shard cache then behaves like Yi's unified buffer
 * pool instead of a write-only view.
 *
 * The wave buffer is ThreadData::wave_scratch (one page per neighbor of an
 * expansion), separate from sector_scratch whose slot count doubles as the
 * sync No-PQ pipeline depth.
 */

#pragma once

#if defined(__linux__)

  #include <algorithm>
  #include <stdexcept>
  #include <string>
  #include <utility>
  #include <vector>

  #include "coro/task.hpp"
  #include "coro/thread_pool.hpp"
  #include "index/graph/diskann/beam_search.hpp"
  #include "index/graph/diskann/disk_page_io.hpp"
  #include "storage/io/page_awaitable.hpp"
  #include "storage/io/uring_reactor.hpp"

namespace alaya::diskann {

// Optional, non-owning cancellation hook for the coroutine beam. The owner
// must keep state alive until the coroutine returns. A null probe preserves
// the historical path and avoids an indirect call at wave boundaries.
struct BeamSearchCancelProbe {
  using IsCancelled = bool (*)(const void *) noexcept;

  const void *state{};
  IsCancelled is_cancelled{};

  [[nodiscard]] auto requested() const noexcept -> bool {
    return is_cancelled != nullptr && is_cancelled(state);
  }
};

namespace beam_async_detail {
inline void check_wave(const std::vector<storage::io::ReadResult> &results,
                       uint64_t page_size) {
  for (const auto &result : results) {
    if (result.status != storage::io::ReadStatus::ok || result.bytes != page_size) {
      throw std::runtime_error("beam_search_async: short/failed PageReader read");
    }
  }
}

class PoolExecutor final : public storage::io::ResumeExecutor {
 public:
  explicit PoolExecutor(coro::thread_pool &pool) : pool_(pool) {}
  auto execute(std::coroutine_handle<> handle) noexcept -> bool override {
    return pool_.resume(handle);
  }

 private:
  coro::thread_pool &pool_;
};
}  // namespace beam_async_detail

/// Coroutine No-PQ greedy search: one reactor wave per expansion.
/// Preconditions: ctx describes a No-PQ index and @p td is exclusively owned by this coroutine until it
/// completes (acquire it through a suspending gate, never a thread-blocking
/// pool — a blocked pool thread cannot run the resume that frees a td).
inline auto disk_greedy_search_async(const SearchContext &ctx,
                                     const float *query,
                                     uint32_t top_k,
                                     const SearchParams &params,
                                     ThreadData &td,
                                     SearchStats *stats,
                                     alaya::UringReactor *reactor,
                                     coro::thread_pool &pool,
                                     int fd,
                                     const BeamSearchCancelProbe *cancel_probe = nullptr)
    -> coro::task<std::vector<std::pair<uint32_t, float>>> {
  (void)reactor;
  (void)fd;
  beam_async_detail::PoolExecutor executor(pool);
  auto &reader = *ctx.reader;
  const DiskLayoutGeometry &geom = *ctx.geom;
  const NodeCache &cache = *ctx.cache;
  DiskPageIO *page_io = ctx.page_io;  ///< unified-pool peek + fill (may be null)
  const uint64_t dim = geom.dim;
  const uint64_t page_size = geom.page_size;
  const auto l2 = alaya::simd::get_l2_sqr_func();
  // One page per potential neighbor of one expansion; the seed reuses slot 0
  // before the first expansion runs.
  const uint64_t wave_slots = std::max<uint64_t>(1, geom.max_degree);
  td.ensure_wave_scratch(wave_slots * page_size);

  const size_t list_size = std::max<size_t>(params.search_list_size, top_k);
  td.reset_query(list_size);
  auto &frontier = td.retset;
  auto &visited = td.visited_bits;

  // Same absorb as the sync search: exact distance into the frontier, neighbor
  // list into the per-query scratch cache.
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
  {
    bool seeded = false;
    {
      NodeCache::Lookup hit = cache.lookup_record(ctx.medoid);
      if (hit) {
        if (stats != nullptr) {
          stats->n_cache_hits++;
        }
        absorb(ctx.medoid, hit.get());
        seeded = true;
      }
    }  // Lookup released before any suspension.
    if (!seeded) {
      const uint64_t seed_off = geom.get_page_offset(ctx.medoid);
      uint64_t seed_version = 0;
      if (page_io != nullptr &&
          page_io->search_peek_page(seed_off, td.wave_scratch, &seed_version)) {
        if (stats != nullptr) {
          stats->n_page_cache_hits++;
        }
        absorb(ctx.medoid, td.wave_scratch + geom.offset_to_node(ctx.medoid));
        seeded = true;
      }
      if (!seeded) {
        std::vector<storage::io::ReadRequest> seed_req;
        seed_req.push_back(beam_io_detail::request(seed_off, page_size, ctx.medoid, td.wave_scratch));
        auto seed_results = co_await storage::io::read_pages(reader, executor, seed_req);
        beam_async_detail::check_wave(seed_results, page_size);
        if (page_io != nullptr) {
          page_io->search_fill_page(seed_off, td.wave_scratch, seed_version);
        }
        if (stats != nullptr) {
          stats->n_ios++;
        }
        absorb(ctx.medoid, td.wave_scratch + geom.offset_to_node(ctx.medoid));
      }
    }
  }
  if (cancel_probe != nullptr && cancel_probe->requested()) {
    co_return std::vector<std::pair<uint32_t, float>>{};
  }

  std::vector<uint32_t> todo;
  std::vector<uint32_t> miss_ids;
  std::vector<uint64_t> miss_versions;
  std::vector<storage::io::ReadRequest> reqs;
  miss_ids.reserve(wave_slots);
  miss_versions.reserve(wave_slots);
  reqs.reserve(wave_slots);
  while (frontier.has_unexpanded_node()) {
    const uint32_t x = frontier.closest_unexpanded().id;
    const NeighborScratchView nbrs = td.cached_neighbors(x);
    todo.clear();
    for (const uint32_t m : nbrs) {
      consider(m, [&](uint32_t id) {
        todo.push_back(id);
      });
    }
    reqs.clear();
    miss_ids.clear();
    miss_versions.clear();
    uint64_t slot = 0;
    for (const uint32_t m : todo) {
      NodeCache::Lookup cached = cache.lookup_record(m);
      if (cached) {
        if (stats != nullptr) {
          stats->n_cache_hits++;
        }
        absorb(m, cached.get());
        continue;
      }
      char *buf = td.wave_scratch + slot * page_size;
      const uint64_t page_off = geom.get_page_offset(m);
      uint64_t version = 0;
      if (page_io != nullptr && page_io->search_peek_page(page_off, buf, &version)) {
        // Unified-pool hit: serve without a device read (slot buffer is
        // consumed by absorb, reused).
        if (stats != nullptr) {
          stats->n_page_cache_hits++;
        }
        absorb(m, buf + geom.offset_to_node(m));
        continue;
      }
      ++slot;
      miss_ids.push_back(m);
      miss_versions.push_back(version);
      reqs.push_back(beam_io_detail::request(page_off, page_size, m, buf));
    }
    if (!reqs.empty()) {
      auto results = co_await storage::io::read_pages(reader, executor, reqs);
      beam_async_detail::check_wave(results, page_size);
      if (stats != nullptr) {
        stats->n_ios += reqs.size();
      }
      for (size_t j = 0; j < miss_ids.size(); ++j) {
        char *buf = reinterpret_cast<char *>(reqs[j].buffer.data());
        if (page_io != nullptr) {
          // Offer the read page to the pool BEFORE parsing: a conflicting
          // writer refreshes buf instead of the pool taking stale bytes.
          page_io->search_fill_page(reqs[j].offset, buf, miss_versions[j]);
        }
        absorb(miss_ids[j], buf + geom.offset_to_node(miss_ids[j]));
      }
    }
    if (cancel_probe != nullptr && cancel_probe->requested()) {
      break;
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
  co_return out;
}

// Retain the original reactor-reference overload for every existing caller.
// PageReader owns the actual reads; the pointer overload lets readonly callers
// avoid manufacturing an unused reactor.
inline auto disk_greedy_search_async(const SearchContext &ctx,
                                     const float *query,
                                     uint32_t top_k,
                                     const SearchParams &params,
                                     ThreadData &td,
                                     SearchStats *stats,
                                     alaya::UringReactor &reactor,
                                     coro::thread_pool &pool,
                                     int fd,
                                     const BeamSearchCancelProbe *cancel_probe = nullptr)
    -> coro::task<std::vector<std::pair<uint32_t, float>>> {
  return disk_greedy_search_async(ctx,
                                  query,
                                  top_k,
                                  params,
                                  td,
                                  stats,
                                  &reactor,
                                  pool,
                                  fd,
                                  cancel_probe);
}

/// Coroutine PQ beam search. Existing null-probe callers retain the historical
/// no-rerank-only contract. A probed readonly operation may exact-rerank the
/// retained entries already expanded by traversal, without submitting a new
/// blocking read through the td's AIO context. Each beam step reads
/// its cache misses as one reactor wave; cache hits are processed inline while
/// popping, exactly like the sync pipelined variant's fill_pipe, so this sits
/// in the same recall-equivalence class as both sync schedulers (and is
/// byte-identical to the deterministic one when nothing is cached).
///
/// PQ-lock note: callers do NOT hold pq_mutex_ across this coroutine (a
/// shared_mutex may not be released on a different thread than took it).
/// That is safe by protocol: concurrent encode_pq_slot() only writes codes of
/// slots that are still dark (allocated, not yet published), and the search's
/// tombstone snapshot — taken before the search — masks every such slot, so
/// pq_distance is never called on an entry being written.
inline auto pq_beam_search_async(const SearchContext &ctx,
                                 const float *query,
                                 uint32_t top_k,
                                 const SearchParams &params,
                                 ThreadData &td,
                                 SearchStats *stats,
                                 alaya::UringReactor *reactor,
                                 coro::thread_pool &pool,
                                 int fd,
                                 const BeamSearchCancelProbe *cancel_probe = nullptr)
    -> coro::task<std::vector<std::pair<uint32_t, float>>> {
  if (params.rerank && cancel_probe == nullptr) {
    throw std::invalid_argument("pq_beam_search_async: rerank is not supported");
  }
  (void)reactor;
  (void)fd;
  beam_async_detail::PoolExecutor executor(pool);
  auto &reader = *ctx.reader;
  const DiskLayoutGeometry &geom = *ctx.geom;
  const NodeCache &cache = *ctx.cache;
  DiskPageIO *page_io = ctx.page_io;  ///< unified-pool peek + fill (may be null)
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

  visited.set(ctx.medoid);
  retset.insert(alaya::vamana::Neighbor(ctx.medoid, pq.pq_distance(ctx.medoid, pq_table)));

  const uint64_t beam = std::max<uint64_t>(1, params.beam_width);
  td.ensure_wave_scratch(beam * page_size);

  auto process_node = [&](uint32_t id, const char *rec) {
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
  };

  std::vector<uint32_t> miss_ids;
  std::vector<uint64_t> miss_versions;
  std::vector<storage::io::ReadRequest> reqs;
  miss_ids.reserve(beam);
  miss_versions.reserve(beam);
  reqs.reserve(beam);
  while (retset.has_unexpanded_node()) {
    reqs.clear();
    miss_ids.clear();
    miss_versions.clear();
    uint64_t slot = 0;
    while (retset.has_unexpanded_node() && reqs.size() < beam) {
      const uint32_t id = retset.closest_unexpanded().id;
      NodeCache::Lookup cached = cache.lookup_record(id);
      if (cached) {
        if (stats != nullptr) {
          stats->n_cache_hits++;
        }
        process_node(id, cached.get());
        continue;  // Lookup released here — never held across the wave below
      }
      char *buf = td.wave_scratch + slot * page_size;
      const uint64_t page_off = geom.get_page_offset(id);
      uint64_t version = 0;
      if (page_io != nullptr && page_io->search_peek_page(page_off, buf, &version)) {
        // Unified-pool hit: no device read; the slot buffer is consumed by
        // process_node and reused.
        if (stats != nullptr) {
          stats->n_page_cache_hits++;
        }
        process_node(id, buf + geom.offset_to_node(id));
        continue;
      }
      ++slot;
      miss_ids.push_back(id);
      miss_versions.push_back(version);
      reqs.push_back(beam_io_detail::request(page_off, page_size, id, buf));
    }
    if (!reqs.empty()) {
      auto results = co_await storage::io::read_pages(reader, executor, reqs);
      beam_async_detail::check_wave(results, page_size);
      if (stats != nullptr) {
        stats->n_ios += reqs.size();
      }
      for (size_t j = 0; j < miss_ids.size(); ++j) {
        char *buf = reinterpret_cast<char *>(reqs[j].buffer.data());
        if (page_io != nullptr) {
          // Offer the read page to the pool BEFORE parsing: a conflicting
          // writer refreshes buf instead of the pool taking stale bytes.
          page_io->search_fill_page(reqs[j].offset, buf, miss_versions[j]);
        }
        process_node(miss_ids[j], buf + geom.offset_to_node(miss_ids[j]));
      }
    }
    if (cancel_probe != nullptr && cancel_probe->requested()) {
      break;
    }
  }

  // Result extraction: a completed traversal has exact distances for every
  // retained node. A cancelled traversal retains only the already-expanded
  // exact candidates when rerank was requested; no new page wave is submitted.
  std::vector<std::pair<uint32_t, float>> out;
  auto is_live = [&](uint32_t id) {
    return ctx.tombstone == nullptr || !ctx.tombstone->is_deleted(id);
  };
  if (params.rerank) {
    const size_t want =
        params.rerank_count > 0 ? params.rerank_count : static_cast<size_t>(top_k) * 3;
    out.reserve(std::min<size_t>(retset.size(), want));
    for (size_t i = 0; i < retset.size() && out.size() < want; ++i) {
      const uint32_t id = retset[i].id;
      if (!is_live(id)) {
        continue;
      }
      const float exact = td.exact_dist(id);
      if (!ThreadData::is_missing_exact(exact)) {
        out.emplace_back(id, exact);
      }
    }
  } else {
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
  co_return out;
}

inline auto pq_beam_search_async(const SearchContext &ctx,
                                 const float *query,
                                 uint32_t top_k,
                                 const SearchParams &params,
                                 ThreadData &td,
                                 SearchStats *stats,
                                 alaya::UringReactor &reactor,
                                 coro::thread_pool &pool,
                                 int fd,
                                 const BeamSearchCancelProbe *cancel_probe = nullptr)
    -> coro::task<std::vector<std::pair<uint32_t, float>>> {
  return pq_beam_search_async(ctx,
                              query,
                              top_k,
                              params,
                              td,
                              stats,
                              &reactor,
                              pool,
                              fd,
                              cancel_probe);
}

}  // namespace alaya::diskann

#endif  // __linux__
