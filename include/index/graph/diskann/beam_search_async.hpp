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
 * Reads go through the reader's O_DIRECT fd (AlignedFileReader::get_fd), not
 * DiskPageIO's buffered fd: search I/O must keep bypassing the OS page cache
 * or a disk-bound configuration silently turns into a RAM-cached one.
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
  #include "storage/io/uring_reactor.hpp"

namespace alaya::diskann {

namespace beam_async_detail {
[[noreturn]] inline void throw_wave_failure(const std::vector<alaya::IORequest> &reqs) {
  for (const auto &req : reqs) {
    if (!req.is_success()) {
      throw std::runtime_error("beam_search_async: short/failed read, result=" +
                               std::to_string(req.result_));
    }
  }
  throw std::runtime_error("beam_search_async: wave reported failures");
}
}  // namespace beam_async_detail

/// Coroutine No-PQ greedy search: one reactor wave per expansion.
/// Preconditions: ctx describes a No-PQ index, @p fd is the reader's O_DIRECT
/// descriptor, and @p td is exclusively owned by this coroutine until it
/// completes (acquire it through a suspending gate, never a thread-blocking
/// pool — a blocked pool thread cannot run the resume that frees a td).
inline auto disk_greedy_search_async(const SearchContext &ctx,
                                     const float *query,
                                     uint32_t top_k,
                                     const SearchParams &params,
                                     ThreadData &td,
                                     SearchStats *stats,
                                     alaya::UringReactor &reactor,
                                     coro::thread_pool &pool,
                                     int fd,
                                     DiskPageIO *page_peek = nullptr)
    -> coro::task<std::vector<std::pair<uint32_t, float>>> {
  const DiskLayoutGeometry &geom = *ctx.geom;
  const NodeCache &cache = *ctx.cache;
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
      std::vector<alaya::IORequest> seed_req;
      seed_req.emplace_back(td.wave_scratch, page_size, geom.get_page_offset(ctx.medoid));
      if (co_await reactor.read_batch(pool, fd, seed_req.data(), 1) != 0) {
        beam_async_detail::throw_wave_failure(seed_req);
      }
      if (stats != nullptr) {
        stats->n_ios++;
      }
      absorb(ctx.medoid, td.wave_scratch + geom.offset_to_node(ctx.medoid));
    }
  }

  std::vector<uint32_t> todo;
  std::vector<uint32_t> miss_ids;
  std::vector<alaya::IORequest> reqs;
  miss_ids.reserve(wave_slots);
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
      if (page_peek != nullptr && page_peek->try_read_cached_page(geom.get_page_offset(m), buf)) {
        // Recently written page still in the update shard cache: serve it
        // without a device read (slot buffer is consumed by absorb, reused).
        if (stats != nullptr) {
          stats->n_cache_hits++;
        }
        absorb(m, buf + geom.offset_to_node(m));
        continue;
      }
      ++slot;
      miss_ids.push_back(m);
      reqs.emplace_back(buf, page_size, geom.get_page_offset(m));
    }
    if (!reqs.empty()) {
      if (co_await reactor.read_batch(pool, fd, reqs.data(), static_cast<uint32_t>(reqs.size())) !=
          0) {
        beam_async_detail::throw_wave_failure(reqs);
      }
      if (stats != nullptr) {
        stats->n_ios += reqs.size();
      }
      for (size_t j = 0; j < miss_ids.size(); ++j) {
        absorb(miss_ids[j],
               static_cast<const char *>(reqs[j].buffer_) + geom.offset_to_node(miss_ids[j]));
      }
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

/// Coroutine PQ beam search (no-rerank only — the update search never
/// reranks; rerank would need blocking reads through the td's AIO context,
/// which gate ThreadData objects deliberately do not register). Each beam step reads
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
                                 alaya::UringReactor &reactor,
                                 coro::thread_pool &pool,
                                 int fd,
                                 DiskPageIO *page_peek = nullptr)
    -> coro::task<std::vector<std::pair<uint32_t, float>>> {
  if (params.rerank) {
    throw std::invalid_argument("pq_beam_search_async: rerank is not supported");
  }
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
  std::vector<alaya::IORequest> reqs;
  miss_ids.reserve(beam);
  reqs.reserve(beam);
  while (retset.has_unexpanded_node()) {
    reqs.clear();
    miss_ids.clear();
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
      if (page_peek != nullptr && page_peek->try_read_cached_page(geom.get_page_offset(id), buf)) {
        // Update shard cache hit (recently written page): no device read; the
        // slot buffer is consumed by process_node and reused.
        if (stats != nullptr) {
          stats->n_cache_hits++;
        }
        process_node(id, buf + geom.offset_to_node(id));
        continue;
      }
      ++slot;
      miss_ids.push_back(id);
      reqs.emplace_back(buf, page_size, geom.get_page_offset(id));
    }
    if (!reqs.empty()) {
      if (co_await reactor.read_batch(pool, fd, reqs.data(), static_cast<uint32_t>(reqs.size())) !=
          0) {
        beam_async_detail::throw_wave_failure(reqs);
      }
      if (stats != nullptr) {
        stats->n_ios += reqs.size();
      }
      for (size_t j = 0; j < miss_ids.size(); ++j) {
        process_node(miss_ids[j],
                     static_cast<const char *>(reqs[j].buffer_) + geom.offset_to_node(miss_ids[j]));
      }
    }
  }

  // Result extraction: PQ no-rerank (exact where known during traversal).
  std::vector<std::pair<uint32_t, float>> out;
  auto is_live = [&](uint32_t id) {
    return ctx.tombstone == nullptr || !ctx.tombstone->is_deleted(id);
  };
  out.reserve(std::min<size_t>(retset.size(), static_cast<size_t>(top_k)));
  for (size_t i = 0; i < retset.size() && out.size() < top_k; ++i) {
    const uint32_t id = retset[i].id;
    if (!is_live(id)) {
      continue;
    }
    const float exact = td.exact_dist(id);
    out.emplace_back(id, !ThreadData::is_missing_exact(exact) ? exact : retset[i].distance);
  }
  std::sort(out.begin(), out.end(), [](const auto &a, const auto &b) {
    return a.second < b.second || (a.second == b.second && a.first < b.first);
  });
  if (out.size() > top_k) {
    out.resize(top_k);
  }
  co_return out;
}

}  // namespace alaya::diskann

#endif  // __linux__
