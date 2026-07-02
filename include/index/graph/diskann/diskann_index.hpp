// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file diskann_index.hpp
 * @brief Self-contained DiskANN disk index: build / load / search.
 *
 * `DiskANNIndex` is an autonomous index class (peer of LASER's QuantizedGraph,
 * design D1): it builds a Vamana graph in memory, packs it into a sector-aligned
 * disk layout, optionally trains PQ, and serves cached beam search over the
 * resulting on-disk index. It does not participate in the segment / disk-
 * collection subsystem.
 *
 * On-disk directory:
 *   meta.bin           index metadata (this file's MetaHeader)
 *   diskann.index      sector-aligned graph + vectors (disk_layout.hpp)
 *   ids.bin            internal-id -> external uint64 label map
 *   cache_ids.bin      BFS cache node ids        (node_cache.hpp)
 *   cache_nodes.bin    BFS cache node records
 *   pq_pivots.bin      PQ global centroid + codebook  (PQ builds only)
 *   pq_compressed.bin  PQ codes                       (PQ builds only)
 */

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "coro/sync_wait.hpp"
#include "coro/task.hpp"
#include "coro/thread_pool.hpp"
#include "coro/when_all.hpp"
#include "index/graph/diskann/beam_search.hpp"
#include "index/graph/diskann/disk_layout.hpp"
#include "index/graph/diskann/disk_page_io.hpp"
#include "index/graph/diskann/disk_update_context.hpp"
#include "index/graph/diskann/node_cache.hpp"
#include "index/graph/diskann/pq_table.hpp"
#include "index/graph/diskann/search_scratch.hpp"
#include "index/graph/diskann/slot_allocator.hpp"
#include "index/graph/diskann/tombstone_bitmap.hpp"
#include "index/graph/laser/utils/aligned_file_reader_factory.hpp"
#include "index/graph/laser/utils/concurrent_queue.hpp"
#include "index/graph/vamana/robust_prune.hpp"
#include "index/graph/vamana/vamana_builder.hpp"
#include "simd/distance_l2.hpp"

namespace alaya::diskann {

inline constexpr uint32_t kDefaultDiskANNScratchSearchListSize = 150;
inline constexpr uint32_t kDefaultDiskANNUpdateReconnectThreads = 4;
inline constexpr uint32_t kDefaultDiskANNUpdateInsertThreads = 32;

/// Build-time configuration.
struct DiskANNBuildParams {
  uint32_t R = 64;            ///< graph degree bound
  uint32_t L = 100;           ///< Vamana build beam width
  float alpha = 1.2f;         ///< Vamana α-RNG pruning
  uint32_t pq_n_chunks = 0;   ///< 0 => no PQ
  double cache_ratio = 0.05;  ///< BFS cache fraction
  uint32_t num_threads = 0;   ///< 0 => all cores (Vamana build + PQ train/encode)
  uint32_t pq_train_iters = 15;
  uint64_t seed = 1234;
  bool verbose = false;  ///< print per-phase build wall-times to stderr
};

/// Load-time configuration (sizes the thread-scratch pool).
struct DiskANNLoadParams {
  uint32_t num_threads = 4;    ///< max concurrent searches (ThreadData pool size)
  uint32_t beam_width = 4;     ///< PQ I/O beam width (PQ caps in-flight reads at this)
  uint32_t nopq_io_depth = 0;  ///< No-PQ async pipeline depth (max reads in flight); 0 => 32,
                               ///< the benchmark-tuned default (SIFT1M/NVMe: ~2x single-query
                               ///< latency cut and +50% 8-thread throughput vs 2*beam_width,
                               ///< recall unchanged). No-PQ issues thousands of reads/query, so a
                               ///< deeper pipeline overlaps more I/O. Explicit values are floored
                               ///< at 2*beam_width and capped at the libaio context size (1024).
                               ///< Sizes the sector scratch to that many pages per thread.
  uint32_t scratch_search_list_size = kDefaultDiskANNScratchSearchListSize;
  ///< No-PQ neighbor scratch capacity, in search-list entries. Set this >= the
  ///< largest DiskANNSearchParams::search_list_size used after load.

  // --- In-place update mode ---
  bool updatable = false;             ///< open O_RDWR + enable insert/remove/update_node/flush
  uint32_t update_search_l = 100;     ///< L for the insert NN-search (candidate pool before prune)
  float update_alpha = 1.2f;          ///< alpha-RNG pruning for insert/reconnect (Vamana default)
  double safety_net_ratio = 0.05;     ///< tombstone ratio that arms the safety-net reconnect
  uint64_t safety_net_ops = 16;       ///< deletes without an insert before the safety net may fire
  size_t page_cache_capacity = 4096;  ///< update-path page LRU cache capacity; 0 disables it
  uint32_t update_insert_threads = kDefaultDiskANNUpdateInsertThreads;
  ///< Coroutine worker count for batch_insert's outer insert concurrency.
  uint32_t update_reconnect_threads = kDefaultDiskANNUpdateReconnectThreads;
  ///< Coroutine worker count for reconnecting a single insert's pruned neighbors.
};

/// Per-query search configuration.
struct DiskANNSearchParams {
  uint32_t search_list_size = 100;  ///< L (retset capacity)
  bool use_pq = true;               ///< use PQ approx distances (ignored if no PQ)
  bool rerank = true;               ///< PQ only: re-score top candidates with exact L2
  uint32_t rerank_count = 0;        ///< PQ rerank pool size; 0 => top_k*3 (spec default)
  bool deterministic = false;       ///< Reproducible batch==sequential via a per-expansion
                                    ///< barrier (PQ: per-beam; ~10-15% slower). Default off =
                                    ///< async-pipelined I/O. Applies to both PQ and No-PQ.
};

class DiskANNIndex {
 public:
  static constexpr uint64_t kMetaMagic = 0x414C594144534B4EULL;  // "ALYADSKN"
  /// v2 adds max_slot_id + live_count for in-place updates; v1 is read back-compat.
  static constexpr uint32_t kMetaVersion = 2;
  /// Default No-PQ async pipeline depth when DiskANNLoadParams::nopq_io_depth == 0.
  /// Benchmark-tuned on SIFT1M/NVMe (knee at ~32; deeper gives no gain).
  static constexpr uint32_t kDefaultNoPQIoDepth = 32;

  DiskANNIndex() = default;
  ~DiskANNIndex() { teardown(); }

  DiskANNIndex(const DiskANNIndex &) = delete;
  DiskANNIndex &operator=(const DiskANNIndex &) = delete;
  DiskANNIndex(DiskANNIndex &&) = delete;
  DiskANNIndex &operator=(DiskANNIndex &&) = delete;

  // ------------------------------------------------------------------ build
  /**
   * @brief Build a complete index directory from vectors + external labels.
   * @throws std::invalid_argument for dim==0 / n==0 / null inputs.
   * @throws std::runtime_error if @p index_dir already exists.
   */
  static void build(const std::string &index_dir,
                    const float *vectors,
                    const uint64_t *labels,
                    uint64_t n,
                    uint64_t dim,
                    const DiskANNBuildParams &params) {
    if (dim == 0) {
      throw std::invalid_argument("DiskANNIndex::build: dim must be > 0");
    }
    if (n == 0) {
      throw std::invalid_argument("DiskANNIndex::build: n must be > 0");
    }
    if (vectors == nullptr || labels == nullptr) {
      throw std::invalid_argument("DiskANNIndex::build: null vectors/labels");
    }
    if (params.pq_n_chunks > 0 && dim % params.pq_n_chunks != 0) {
      throw std::invalid_argument("DiskANNIndex::build: dim not divisible by pq_n_chunks");
    }
    namespace fs = std::filesystem;
    if (fs::exists(index_dir)) {
      throw std::runtime_error("DiskANNIndex::build: index_dir already exists: " + index_dir);
    }
    if (!fs::create_directories(index_dir)) {
      throw std::runtime_error("DiskANNIndex::build: cannot create " + index_dir);
    }
    // Remove the partial directory if any build phase throws, so the failed build
    // does not block a retry with "index_dir already exists".
    struct DirGuard {
      const std::string &dir;
      bool committed = false;
      ~DirGuard() {
        if (!committed) {
          std::error_code ec;
          std::filesystem::remove_all(dir, ec);
        }
      }
    } dir_guard{index_dir};

    // Per-phase wall-clock timing (opt-in; mirrors official build_disk_index logging).
    using clk = std::chrono::steady_clock;
    auto stamp = [&params](const char *name, clk::time_point a, clk::time_point b) {
      if (params.verbose) {
        std::cerr << "[build] " << name << ": " << std::chrono::duration<double>(b - a).count()
                  << " s\n";
      }
    };

    // 1. Vamana graph.
    auto t_vamana0 = clk::now();
    alaya::vamana::VamanaBuildParams vparams;
    vparams.R = params.R;
    vparams.L = params.L;
    vparams.alpha = params.alpha;
    vparams.num_threads = params.num_threads;
    vparams.seed = params.seed;
    alaya::vamana::VamanaBuilder builder(vectors, n, static_cast<uint32_t>(dim), vparams);
    builder.build();
    const auto &graph = builder.graph();
    const uint32_t medoid = builder.medoid();
    auto t_vamana1 = clk::now();
    stamp("vamana", t_vamana0, t_vamana1);

    const DiskLayoutGeometry geom = DiskLayoutGeometry::compute(dim, params.R);

    // 2. Sector-aligned disk layout.  3. External labels.
    write_disk_layout(path(index_dir, "diskann.index"), vectors, graph, {n, dim, params.R, medoid});
    write_ids(path(index_dir, "ids.bin"), labels, n);
    auto t_layout = clk::now();
    stamp("layout+ids", t_vamana1, t_layout);

    // 4. Optional PQ.
    const bool has_pq = params.pq_n_chunks > 0;
    if (has_pq) {
      PQTable pq;
      pq.train(vectors,
               n,
               dim,
               params.pq_n_chunks,
               params.pq_train_iters,
               params.seed,
               params.num_threads);
      pq.encode(vectors, n, params.num_threads);
      pq.save(path(index_dir, "pq_pivots.bin"), path(index_dir, "pq_compressed.bin"));
    }
    auto t_pq = clk::now();
    stamp("pq(train+encode+save)", t_layout, t_pq);

    // 5. BFS cache.
    NodeCache cache;
    cache.generate(graph, vectors, medoid, n, dim, params.R, params.cache_ratio);
    cache.save(path(index_dir, "cache_ids.bin"), path(index_dir, "cache_nodes.bin"));
    stamp("cache", t_pq, clk::now());

    // 6. Metadata.
    MetaHeader meta;
    meta.num_points = n;
    meta.dim = dim;
    meta.max_degree = params.R;
    meta.medoid = medoid;
    meta.has_pq = has_pq ? 1 : 0;
    meta.pq_n_chunks = params.pq_n_chunks;
    meta.node_len = geom.node_len;
    meta.nodes_per_sector = geom.nodes_per_sector;
    meta.max_slot_id = n;  // fresh build: file capacity == num_points
    meta.live_count = n;   // fresh build: every slot is live
    write_meta(path(index_dir, "meta.bin"), meta);

    dir_guard.committed = true;  // build complete — keep the directory
  }

  // ------------------------------------------------------------------- load
  void load(const std::string &index_dir, const DiskANNLoadParams &params = {}) {
    teardown();
    namespace fs = std::filesystem;
    if (!fs::exists(index_dir) || !fs::is_directory(index_dir)) {
      throw std::runtime_error("DiskANNIndex::load: not a directory: " + index_dir);
    }

    const MetaHeader meta = read_meta(path(index_dir, "meta.bin"));
    const uint64_t num_points = meta.num_points;
    dim_ = meta.dim;
    max_degree_ = meta.max_degree;
    medoid_ = meta.medoid;
    has_pq_ = meta.has_pq != 0;
    pq_n_chunks_ = meta.pq_n_chunks;
    max_slot_id_ = meta.max_slot_id;  // file capacity (slots); == num_points for static/v1
    live_count_ = meta.live_count;    // live nodes; == num_points for static/v1
    geom_ = DiskLayoutGeometry::compute(dim_, max_degree_);
    if (geom_.node_len != meta.node_len || geom_.nodes_per_sector != meta.nodes_per_sector) {
      throw std::runtime_error("DiskANNIndex::load: meta geometry inconsistent");
    }

    read_ids(path(index_dir, "ids.bin"), max_slot_id_);
    cache_.load(path(index_dir, "cache_ids.bin"), path(index_dir, "cache_nodes.bin"));
    if (has_pq_) {
      pq_.load(path(index_dir, "pq_pivots.bin"),
               path(index_dir, "pq_compressed.bin"),
               num_points,
               dim_,
               pq_n_chunks_);
    }

    // Open the disk index and pre-register one libaio I/O context + scratch
    // buffer per pool slot. Each context is created on a short-lived
    // registration thread, then borrowed by whichever search thread pops the
    // slot. A libaio context is a process-wide handle, and the slot pool
    // guarantees only one thread uses a given context at a time, so this is safe
    // (verified by the concurrency stress test). The deterministic beam loop
    // (beam_search.hpp) makes results independent of I/O completion timing.
    reader_ = make_aligned_file_reader();
    reader_->open(path(index_dir, "diskann.index"));
    beam_width_ = std::max<uint32_t>(1, params.beam_width);
    uint32_t pool = std::max<uint32_t>(1, params.num_threads);
    if (params.updatable) {
      if (params.update_insert_threads == 0) {
        throw std::invalid_argument("DiskANNIndex::load: update_insert_threads must be > 0");
      }
      if (params.update_reconnect_threads == 0) {
        throw std::invalid_argument("DiskANNIndex::load: update_reconnect_threads must be > 0");
      }
      pool = std::max({pool, params.update_insert_threads, params.update_reconnect_threads});
    }
    const uint32_t pq_table_entries = has_pq_ ? pq_n_chunks_ * kPQNumCentroids : 0;
    // One page slot per concurrent read. nopq_io_depth = 0 resolves to the
    // benchmark-tuned default (kDefaultNoPQIoDepth = 32); No-PQ issues far more
    // reads than PQ, so this deeper pipeline overlaps more I/O. Floored at
    // 2*beam_width (PQ's needs) and capped at MAX_EVENTS (libaio context size).
    const uint64_t nopq_depth =
        params.nopq_io_depth == 0 ? kDefaultNoPQIoDepth : params.nopq_io_depth;
    const uint64_t scratch_slots =
        std::min<uint64_t>(1024, std::max<uint64_t>(2ull * beam_width_, nopq_depth));
    const uint32_t scratch_list_size = std::max({DiskANNSearchParams{}.search_list_size,
                                                 params.update_search_l,
                                                 params.scratch_search_list_size});
    ThreadDataScratchConfig scratch_config;
    scratch_config.n_page_slots = scratch_slots;
    scratch_config.page_size = geom_.page_size;
    scratch_config.pq_table_entries = pq_table_entries;
    scratch_config.max_slot_id = max_slot_id_;
    scratch_config.max_degree = max_degree_;
    scratch_config.search_list_size = scratch_list_size;
    scratch_config.query_dim = dim_;
    thread_data_storage_.resize(pool);
    {
      std::vector<std::thread> regs;
      regs.reserve(pool);
      for (uint32_t t = 0; t < pool; ++t) {
        regs.emplace_back([this, t, scratch_config]() {
          reader_->register_thread();
          auto td = std::make_unique<ThreadData>();
          td->ctx_ = reader_->get_ctx();
          td->alloc_scratch(scratch_config);
          thread_data_storage_[t] = std::move(td);
        });
      }
      for (auto &th : regs) {
        th.join();
      }
    }
    for (auto &td : thread_data_storage_) {
      ThreadData *p = td.get();
      thread_data_pool_.push(p);
    }
    num_pool_ = pool;

    if (params.updatable) {
      init_updatable(index_dir, params);
    }
    loaded_ = true;
  }

  // ----------------------------------------------------------------- search
  /**
   * @brief Single-query search. Writes up to @p top_k external labels +
   *        distances (ascending L2) into the caller's buffers.
   * @return number of results written (<= top_k).
   */
  uint32_t search(const float *query,
                  uint32_t top_k,
                  uint64_t *out_labels,
                  float *out_distances,
                  const DiskANNSearchParams &params = {},
                  SearchStats *stats = nullptr) const {
    if (!loaded_) {
      throw std::runtime_error("DiskANNIndex::search: index not loaded");
    }
    if (query == nullptr) {
      throw std::invalid_argument("DiskANNIndex::search: null query");
    }
    if (top_k == 0) {
      throw std::invalid_argument("DiskANNIndex::search: top_k must be > 0");
    }
    if (out_labels == nullptr || out_distances == nullptr) {
      throw std::invalid_argument("DiskANNIndex::search: null output buffers");
    }

    std::shared_lock<std::shared_mutex> search_lock;
    if (updatable_) {
      search_lock = std::shared_lock<std::shared_mutex>(update_mutex_);
    }

    ThreadData *td = acquire();
    uint32_t count = 0;
    try {
      SearchContext ctx;
      ctx.reader = reader_.get();
      ctx.geom = &geom_;
      ctx.cache = &cache_;
      ctx.pq = has_pq_ ? &pq_ : nullptr;
      ctx.medoid = medoid_;
      ctx.num_points = max_slot_id_;
      if (updatable_) {
        ctx.tombstone = &slot_alloc_.tombstone();
      }

      SearchParams sp;
      sp.search_list_size = params.search_list_size;
      sp.beam_width = beam_width_;
      sp.use_pq = params.use_pq && has_pq_;
      sp.rerank = params.rerank;
      sp.rerank_count = params.rerank_count;
      sp.deterministic = params.deterministic;

      const auto results = cached_beam_search(ctx, query, top_k, sp, *td, stats);
      count = static_cast<uint32_t>(results.size());
      for (uint32_t i = 0; i < count; ++i) {
        out_labels[i] = labels_[results[i].first];
        out_distances[i] = results[i].second;
      }
    } catch (...) {
      release(td);
      throw;
    }
    release(td);

    // Pad unused slots with sentinels so callers can detect short result rows.
    for (uint32_t i = count; i < top_k; ++i) {
      out_labels[i] = kNoLabel;
      out_distances[i] = std::numeric_limits<float>::max();
    }
    return count;
  }

  // ------------------------------------------------------------ batch search
  /**
   * @brief Run @p n_queries searches across @p num_threads workers; results are
   *        written row-major into @p out_labels / @p out_distances
   *        (n_queries * top_k).
   */
  void batch_search(const float *queries,
                    uint32_t n_queries,
                    uint32_t top_k,
                    uint64_t *out_labels,
                    float *out_distances,
                    uint32_t num_threads,
                    const DiskANNSearchParams &params = {}) const {
    if (!loaded_) {
      throw std::runtime_error("DiskANNIndex::batch_search: index not loaded");
    }
    if (queries == nullptr) {
      throw std::invalid_argument("DiskANNIndex::batch_search: null queries");
    }
    if (top_k == 0) {
      throw std::invalid_argument("DiskANNIndex::batch_search: top_k must be > 0");
    }

    auto run_one = [&](uint32_t qi) {
      search(queries + static_cast<uint64_t>(qi) * dim_,
             top_k,
             out_labels + static_cast<uint64_t>(qi) * top_k,
             out_distances + static_cast<uint64_t>(qi) * top_k,
             params);
    };

    const uint32_t workers = std::min(std::max<uint32_t>(1, num_threads), num_pool_);
    if (workers <= 1) {
      for (uint32_t qi = 0; qi < n_queries; ++qi) {
        run_one(qi);
      }
      return;
    }

    std::atomic<uint32_t> next{0};
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (uint32_t w = 0; w < workers; ++w) {
      pool.emplace_back([&]() {
        for (;;) {
          const uint32_t qi = next.fetch_add(1);
          if (qi >= n_queries) {
            break;
          }
          run_one(qi);
        }
      });
    }
    for (auto &th : pool) {
      th.join();
    }
  }

  // --------------------------------------------------------------- accessors
  [[nodiscard]] uint64_t size() const { return live_count_; }  // live (non-tombstoned) vectors
  [[nodiscard]] uint64_t dim() const { return dim_; }
  [[nodiscard]] bool has_pq() const { return has_pq_; }
  [[nodiscard]] uint32_t medoid() const { return medoid_; }
  [[nodiscard]] bool updatable() const { return updatable_; }
  [[nodiscard]] uint64_t live_count() const { return live_count_; }
  [[nodiscard]] uint64_t max_slot_id() const { return max_slot_id_; }
  [[nodiscard]] uint64_t tombstone_count() const { return slot_alloc_.tombstone_count(); }
  [[nodiscard]] uint64_t free_slot_count() const { return slot_alloc_.free_count(); }
  [[nodiscard]] uint64_t safety_net_fire_count() const { return safety_net_fires_; }
  [[nodiscard]] bool is_deleted(uint32_t id) const { return slot_alloc_.is_deleted(id); }

  /// Sentinel label for padded (missing) result slots.
  static constexpr uint64_t kNoLabel = std::numeric_limits<uint64_t>::max();

  // ------------------------------------------------------- in-place updates
  /// Insert a vector: NN search -> alpha-RNG prune -> alloc slot -> write ->
  /// reconnect pruned neighbors. Returns the allocated internal slot id.
  uint32_t insert(const float *query, uint64_t label) {
    if (!updatable_) {
      throw std::runtime_error("DiskANNIndex::insert: index not loaded in updatable mode");
    }
    if (query == nullptr) {
      throw std::invalid_argument("DiskANNIndex::insert: null query");
    }
    std::unique_lock<std::shared_mutex> lock(update_mutex_);
    page_io_->clear_cache();

    const std::vector<uint32_t> pruned = select_insert_neighbors(query);
    uint32_t slot = 0;
    {
      std::lock_guard<std::mutex> slot_lock(slot_mutex_);
      slot = allocate_update_slot_unlocked(label);
      resize_thread_data_slot_capacity();
    }
    encode_pq_slot(query, slot);
    write_inserted_node(slot, query, pruned);
    reconnect_inserted_neighbors(pruned, slot);
    page_io_->flush_dirty_pages();
    return slot;
  }

  /// Insert a batch of row-major vectors. The synchronous API is preserved, but
  /// the work is scheduled as coroutine tasks in chunks of @p batch_size.
  std::vector<uint32_t> batch_insert(const float *vectors,
                                     const uint64_t *labels,
                                     uint32_t count,
                                     uint32_t batch_size = 32) {
    if (!updatable_) {
      throw std::runtime_error("DiskANNIndex::batch_insert: index not loaded in updatable mode");
    }
    if (count == 0) {
      return {};
    }
    if (vectors == nullptr || labels == nullptr) {
      throw std::invalid_argument("DiskANNIndex::batch_insert: null vectors/labels");
    }
    if (batch_size == 0) {
      throw std::invalid_argument("DiskANNIndex::batch_insert: batch_size must be > 0");
    }

    std::unique_lock<std::shared_mutex> update_lock(update_mutex_);
    page_io_->clear_cache();

    std::vector<uint32_t> ids(count, 0);
    std::vector<std::vector<uint32_t>> pruned(count);
    const uint32_t workers = std::min({batch_size, count, update_insert_threads_});
    coro::thread_pool pool{{.thread_count = workers,
                            .on_thread_start_functor = nullptr,
                            .on_thread_stop_functor = nullptr}};

    auto plan_one = [this, &pool, vectors, &pruned](uint32_t i) -> coro::task<> {
      co_await pool.schedule();
      pruned[i] = select_insert_neighbors(vectors + static_cast<uint64_t>(i) * dim_);
    };
    auto write_one = [this, &pool, vectors, &ids, &pruned](uint32_t i) -> coro::task<> {
      co_await pool.schedule();
      const float *vec = vectors + static_cast<uint64_t>(i) * dim_;
      encode_pq_slot(vec, ids[i]);
      write_inserted_node(ids[i], vec, pruned[i]);
    };

    try {
      for (uint32_t off = 0; off < count; off += batch_size) {
        const uint32_t end = std::min<uint32_t>(count, off + batch_size);
        run_batch_tasks(off, end, plan_one);
        reserve_batch_insert_slots(labels, ids, off, end);
        run_batch_tasks(off, end, write_one);
        run_batch_reconnect_tasks(off, end, ids, pruned, pool);
      }
      page_io_->flush_dirty_pages();
    } catch (...) {
      pool.shutdown();
      throw;
    }
    pool.shutdown();
    return ids;
  }

  /// Lazy-delete: cache old neighbors for two-hop, tombstone + free the slot.
  /// Reconnect is deferred to the next insert or the safety net.
  void remove(uint32_t internal_id) {
    if (!updatable_) {
      throw std::runtime_error("DiskANNIndex::remove: index not loaded in updatable mode");
    }
    std::unique_lock<std::shared_mutex> lock(update_mutex_);
    validate_removable_slot(internal_id);
    page_io_->clear_cache();
    remove_unlocked(internal_id);
    if (maybe_safety_net_reconnect()) {
      page_io_->flush_dirty_pages();
    }
  }

  /// Lazy-delete a batch of internal ids using the same semantics as remove().
  /// The batch form amortizes update locking and cache setup over Yi-style
  /// delete rounds.
  void batch_remove(const uint32_t *internal_ids, uint32_t count) {
    if (!updatable_) {
      throw std::runtime_error("DiskANNIndex::batch_remove: index not loaded in updatable mode");
    }
    if (count == 0) {
      return;
    }
    if (internal_ids == nullptr) {
      throw std::invalid_argument("DiskANNIndex::batch_remove: null ids");
    }
    std::unique_lock<std::shared_mutex> lock(update_mutex_);
    validate_remove_batch(internal_ids, count);
    page_io_->clear_cache();
    std::vector<std::vector<uint32_t>> old_neighbors =
        page_io_->read_neighbors_batch_parallel(internal_ids, count, update_insert_threads_);
    for (uint32_t i = 0; i < count; ++i) {
      remove_unlocked_with_neighbors(internal_ids[i], std::move(old_neighbors[i]));
    }
    if (maybe_safety_net_reconnect()) {
      page_io_->flush_dirty_pages();
    }
  }

  /**
   * @brief Reconnect @p node_id's neighbor list in place (Yi's connect-task).
   * @throws std::invalid_argument if @p node_id is out of range.
   */
  void update_node(uint32_t node_id) {
    if (!updatable_) {
      throw std::runtime_error("DiskANNIndex::update_node: index not loaded in updatable mode");
    }
    std::unique_lock<std::shared_mutex> lock(update_mutex_);
    if (node_id >= max_slot_id_) {
      throw std::invalid_argument("DiskANNIndex::update_node: node_id out of range");
    }
    if (slot_alloc_.is_deleted(node_id)) {
      throw std::invalid_argument("DiskANNIndex::update_node: node_id is deleted");
    }
    page_io_->clear_cache();
    update_node_impl(node_id);
    page_io_->flush_dirty_pages();
  }

  /// Persist meta + ids + slot allocator state to disk.
  void flush() {
    if (!updatable_) {
      throw std::runtime_error("DiskANNIndex::flush: index not loaded in updatable mode");
    }
    std::unique_lock<std::shared_mutex> lock(update_mutex_);
    page_io_->flush_dirty_pages();
    MetaHeader m;
    m.num_points = max_slot_id_;
    m.dim = dim_;
    m.max_degree = max_degree_;
    m.medoid = medoid_;
    m.has_pq = has_pq_ ? 1 : 0;
    m.pq_n_chunks = pq_n_chunks_;
    m.node_len = geom_.node_len;
    m.nodes_per_sector = geom_.nodes_per_sector;
    m.max_slot_id = max_slot_id_;
    m.live_count = live_count_;
    if (has_pq_) {
      if (pq_.num_points() != max_slot_id_) {
        throw std::runtime_error("DiskANNIndex::flush: PQ code count does not match max_slot_id");
      }
      pq_.save(path(index_dir_, "pq_pivots.bin"), path(index_dir_, "pq_compressed.bin"));
    }
    write_meta(path(index_dir_, "meta.bin"), m);
    write_ids(path(index_dir_, "ids.bin"), labels_.data(), labels_.size());
    slot_alloc_.save(path(index_dir_, "slots.bin"));
  }

 private:
  static constexpr uint32_t kNoSelf = std::numeric_limits<uint32_t>::max();
  static constexpr size_t kUpdateNodeLockStripes = 4096;

  struct MetaHeader {
    uint64_t num_points = 0;
    uint64_t dim = 0;
    uint32_t max_degree = 0;
    uint32_t medoid = 0;
    uint8_t has_pq = 0;
    uint32_t pq_n_chunks = 0;
    uint64_t node_len = 0;
    uint64_t nodes_per_sector = 0;
    uint64_t max_slot_id = 0;  // v2: file capacity in slots (only grows)
    uint64_t live_count = 0;   // v2: num_points minus tombstones
  };

  static std::string path(const std::string &dir, const char *name) {
    return (std::filesystem::path(dir) / name).string();
  }

  // ---- in-place update internals (all called under update_mutex_) ----

  /// Wire up the update subsystem (page IO, slot allocator, context, config).
  void init_updatable(const std::string &index_dir, const DiskANNLoadParams &params) {
    index_dir_ = index_dir;
    update_alpha_ = params.update_alpha;
    update_search_l_ = std::max<uint32_t>(1, params.update_search_l);
    safety_net_ratio_ = params.safety_net_ratio;
    safety_net_ops_ = params.safety_net_ops;
    if (params.update_reconnect_threads == 0) {
      throw std::invalid_argument("DiskANNIndex::load: update_reconnect_threads must be > 0");
    }
    if (params.update_insert_threads == 0) {
      throw std::invalid_argument("DiskANNIndex::load: update_insert_threads must be > 0");
    }
    update_insert_threads_ = params.update_insert_threads;
    update_reconnect_threads_ = params.update_reconnect_threads;
    safety_net_fires_ = 0;
    ops_since_last_insert_ = 0;
    update_ctx_.clear();

    page_io_ = std::make_unique<DiskPageIO>(path(index_dir, "diskann.index"),
                                            geom_,
                                            params.page_cache_capacity);

    const std::string slots_path = path(index_dir, "slots.bin");
    if (std::filesystem::exists(slots_path)) {
      slot_alloc_.load(slots_path);  // restore free list + next id + tombstones
      max_slot_id_ = std::max<uint64_t>(max_slot_id_, slot_alloc_.next_fresh_id());
      resize_thread_data_slot_capacity();
    } else {
      slot_alloc_.reset(static_cast<uint32_t>(max_slot_id_));
    }
    updatable_ = true;
  }

  /// Tombstone-aware update search returning up to @p l (id, dist) candidates.
  /// PQ indexes use PQ beam distances; No-PQ indexes use exact-L2 disk greedy.
  std::vector<std::pair<uint32_t, float>> run_update_search(const float *query, uint32_t l) {
    ThreadData *td = acquire();
    std::vector<std::pair<uint32_t, float>> results;
    try {
      SearchContext ctx;
      ctx.reader = reader_.get();
      ctx.geom = &geom_;
      ctx.cache = &cache_;
      ctx.pq = has_pq_ ? &pq_ : nullptr;
      ctx.medoid = medoid_;
      ctx.num_points = max_slot_id_;
      ctx.tombstone = &slot_alloc_.tombstone();
      SearchParams sp;
      sp.search_list_size = l;
      sp.beam_width = beam_width_;
      sp.use_pq = has_pq_;
      sp.rerank = false;
      sp.deterministic = false;
      results = cached_beam_search(ctx, query, l, sp, *td, nullptr);
    } catch (...) {
      release(td);
      throw;
    }
    release(td);
    return results;
  }

  /// Cached L2 distance between two nodes via the page IO coords cache.
  float cached_l2(uint32_t a, uint32_t b) {
    const auto l2 = alaya::simd::get_l2_sqr_func();
    const std::vector<float> ca = page_io_->read_coords_cached(a);
    const std::vector<float> cb = page_io_->read_coords_cached(b);
    return l2(ca.data(), cb.data(), dim_);
  }

  template <typename DistFn>
  std::vector<uint32_t> prune_insert_pool(std::vector<alaya::vamana::Neighbor> &pool,
                                          DistFn dist_fn) {
    std::vector<uint32_t> pruned;
    std::vector<float> occlude_scratch;
    const uint32_t maxc = std::max<uint32_t>(max_degree_, static_cast<uint32_t>(pool.size()));
    alaya::vamana::prune_neighbors(kNoSelf,
                                   pool,
                                   update_alpha_,
                                   max_degree_,
                                   maxc,
                                   pruned,
                                   occlude_scratch,
                                   dist_fn);
    return pruned;
  }

  std::vector<uint32_t> select_insert_neighbors_exact(const float *query) {
    const auto cand = run_update_search(query, update_search_l_);
    std::vector<alaya::vamana::Neighbor> pool;
    pool.reserve(cand.size());
    for (const auto &c : cand) {
      pool.emplace_back(c.first, c.second);
    }
    auto dist_fn = [this](uint32_t a, uint32_t b) -> float {
      return cached_l2(a, b);
    };
    return prune_insert_pool(pool, dist_fn);
  }

  std::vector<uint32_t> select_insert_neighbors_pq(const float *query) {
    const auto cand = run_update_search(query, update_search_l_);
    std::vector<uint8_t> query_code(pq_n_chunks_);
    pq_.encode_to_code(query, query_code.data());
    std::vector<alaya::vamana::Neighbor> pool;
    pool.reserve(cand.size());
    for (const auto &c : cand) {
      pool.emplace_back(c.first, pq_.pq_symmetric_distance(query_code.data(), c.first));
    }
    auto dist_fn = [this](uint32_t a, uint32_t b) -> float {
      return pq_.pq_symmetric_distance(a, b);
    };
    return prune_insert_pool(pool, dist_fn);
  }

  std::vector<uint32_t> select_insert_neighbors(const float *query) {
    if (has_pq_) {
      return select_insert_neighbors_pq(query);
    }
    return select_insert_neighbors_exact(query);
  }

  uint32_t allocate_update_slot_unlocked(uint64_t label) {
    const uint32_t slot = slot_alloc_.alloc();
    update_ctx_.forget_slot(slot);
    max_slot_id_ = std::max<uint64_t>(max_slot_id_, slot_alloc_.next_fresh_id());
    set_label(slot, label);
    ++live_count_;
    ops_since_last_insert_ = 0;
    return slot;
  }

  void reserve_batch_insert_slots(const uint64_t *labels,
                                  std::vector<uint32_t> &ids,
                                  uint32_t begin,
                                  uint32_t end) {
    std::lock_guard<std::mutex> slot_lock(slot_mutex_);
    for (uint32_t i = begin; i < end; ++i) {
      ids[i] = allocate_update_slot_unlocked(labels[i]);
    }
    resize_thread_data_slot_capacity();
  }

  void encode_pq_slot(const float *query, uint32_t slot) {
    if (!has_pq_) {
      return;
    }
    std::lock_guard<std::mutex> pq_lock(pq_mutex_);
    pq_.encode_one(query, slot);
  }

  void validate_removable_slot(uint32_t internal_id) const {
    if (internal_id >= max_slot_id_) {
      throw std::invalid_argument("DiskANNIndex::remove: internal_id out of range");
    }
    if (slot_alloc_.is_deleted(internal_id)) {
      throw std::invalid_argument("DiskANNIndex::remove: node already deleted");
    }
  }

  void validate_remove_batch(const uint32_t *internal_ids, uint32_t count) const {
    std::unordered_set<uint32_t> seen;
    seen.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      validate_removable_slot(internal_ids[i]);
      if (!seen.insert(internal_ids[i]).second) {
        throw std::invalid_argument("DiskANNIndex::batch_remove: duplicate internal_id");
      }
    }
  }

  void remove_unlocked(uint32_t internal_id) {
    const DiskPageIO::NodeData nd = page_io_->read_node(internal_id);
    remove_unlocked_with_neighbors(internal_id, nd.nbrs);
  }

  void remove_unlocked_with_neighbors(uint32_t internal_id, std::vector<uint32_t> old_neighbors) {
    update_ctx_.removed_node_nbrs_[internal_id] = std::move(old_neighbors);
    slot_alloc_.free(internal_id);
    --live_count_;
    ++ops_since_last_insert_;
  }

  void write_inserted_node(uint32_t slot,
                           const float *query,
                           const std::vector<uint32_t> &neighbors) {
    page_io_->write_node(slot, query, static_cast<uint32_t>(neighbors.size()), neighbors.data());
  }

  template <typename MakeTask>
  void run_batch_tasks(uint32_t begin, uint32_t end, MakeTask make_task) {
    auto run = [&]() -> coro::task<> {
      std::vector<coro::task<>> tasks;
      tasks.reserve(end - begin);
      for (uint32_t i = begin; i < end; ++i) {
        tasks.emplace_back(make_task(i));
      }
      co_await coro::when_all(std::move(tasks));
    };
    coro::sync_wait(run());
  }

  struct ReconnectWork {
    uint32_t node_id = 0;
    std::vector<uint32_t> extra_edges;
  };

  void run_batch_reconnect_tasks(uint32_t begin,
                                 uint32_t end,
                                 const std::vector<uint32_t> &ids,
                                 const std::vector<std::vector<uint32_t>> &pruned,
                                 coro::thread_pool &pool) {
    std::vector<ReconnectWork> work = collect_batch_reconnect_work(begin, end, ids, pruned);
    if (work.empty()) {
      return;
    }
    auto reconnect_one = [this, &pool, &work](uint32_t i) -> coro::task<> {
      const ReconnectWork &item = work[i];
      const DiskPageIO::NodeData node = co_await page_io_->read_node_async(item.node_id, pool);
      std::lock_guard<std::mutex> node_lock(update_node_mutex(item.node_id));
      update_node_impl_from_snapshot(item.node_id, node, item.extra_edges);
    };
    auto run = [&]() -> coro::task<> {
      std::vector<coro::task<>> tasks;
      tasks.reserve(work.size());
      for (uint32_t i = 0; i < work.size(); ++i) {
        tasks.emplace_back(reconnect_one(i));
      }
      co_await coro::when_all(std::move(tasks));
    };
    coro::sync_wait(run());
  }

  std::vector<ReconnectWork> collect_batch_reconnect_work(
      uint32_t begin,
      uint32_t end,
      const std::vector<uint32_t> &ids,
      const std::vector<std::vector<uint32_t>> &pruned) {
    std::vector<ReconnectWork> work;
    std::unordered_map<uint32_t, size_t> pos_by_node;
    for (uint32_t i = begin; i < end; ++i) {
      for (const uint32_t node_id : pruned[i]) {
        auto [pos, inserted] = pos_by_node.emplace(node_id, work.size());
        if (inserted) {
          ReconnectWork item;
          item.node_id = node_id;
          item.extra_edges.push_back(ids[i]);
          work.push_back(std::move(item));
          continue;
        }
        work[pos->second].extra_edges.push_back(ids[i]);
      }
    }
    return work;
  }

  void add_extra_edges(uint32_t node_id,
                       const std::vector<uint32_t> &extra_edges,
                       std::unordered_set<uint32_t> &cand) const {
    for (const uint32_t v : extra_edges) {
      if (v != node_id && !slot_alloc_.is_deleted(v)) {
        cand.insert(v);
      }
    }
  }

  std::vector<alaya::vamana::Neighbor> score_candidates(uint32_t node_id,
                                                        const std::unordered_set<uint32_t> &cand) {
    if (has_pq_) {
      return score_candidates_pq(node_id, cand);
    }
    const std::vector<float> self_coords = page_io_->read_coords_cached(node_id);
    const auto l2 = alaya::simd::get_l2_sqr_func();
    std::vector<alaya::vamana::Neighbor> pool;
    pool.reserve(cand.size());
    for (const uint32_t c : cand) {
      const std::vector<float> cc = page_io_->read_coords_cached(c);
      pool.emplace_back(c, l2(self_coords.data(), cc.data(), dim_));
    }
    return pool;
  }

  std::vector<alaya::vamana::Neighbor> score_candidates_pq(
      uint32_t node_id,
      const std::unordered_set<uint32_t> &cand) {
    std::vector<alaya::vamana::Neighbor> pool;
    pool.reserve(cand.size());
    for (const uint32_t c : cand) {
      pool.emplace_back(c, pq_.pq_symmetric_distance(node_id, c));
    }
    return pool;
  }

  template <typename DistFn>
  std::vector<uint32_t> prune_candidate_pool_with_dist(uint32_t node_id,
                                                       std::vector<alaya::vamana::Neighbor> &pool,
                                                       DistFn dist_fn) {
    std::vector<uint32_t> new_nbrs;
    std::vector<float> occlude_scratch;
    alaya::vamana::prune_neighbors(node_id,
                                   pool,
                                   update_alpha_,
                                   max_degree_,
                                   static_cast<uint32_t>(pool.size()),
                                   new_nbrs,
                                   occlude_scratch,
                                   dist_fn);
    return new_nbrs;
  }

  std::vector<uint32_t> prune_candidate_pool(uint32_t node_id,
                                             std::vector<alaya::vamana::Neighbor> &pool) {
    std::vector<uint32_t> new_nbrs;
    if (pool.size() <= max_degree_) {
      std::sort(pool.begin(), pool.end());
      for (const auto &nb : pool) {
        new_nbrs.push_back(nb.id);
      }
      return new_nbrs;
    }
    if (has_pq_) {
      auto dist_fn = [this](uint32_t a, uint32_t b) -> float {
        return pq_.pq_symmetric_distance(a, b);
      };
      return prune_candidate_pool_with_dist(node_id, pool, dist_fn);
    }
    auto dist_fn = [this](uint32_t a, uint32_t b) -> float {
      return cached_l2(a, b);
    };
    return prune_candidate_pool_with_dist(node_id, pool, dist_fn);
  }

  /// Shared reconnect backbone: score candidates, prune, write if changed.
  void prune_and_write(uint32_t node_id,
                       const std::vector<uint32_t> &old_nbrs,
                       std::unordered_set<uint32_t> &cand,
                       const std::vector<uint32_t> &extra_edges) {
    add_extra_edges(node_id, extra_edges, cand);
    if (cand.empty()) {
      return;
    }
    auto pool = score_candidates(node_id, cand);
    const std::vector<uint32_t> new_nbrs = prune_candidate_pool(node_id, pool);
    if (!same_neighbor_set(old_nbrs, new_nbrs)) {
      page_io_->write_node_neighbors(node_id,
                                     static_cast<uint32_t>(new_nbrs.size()),
                                     new_nbrs.data());
    }
  }

  void update_node_impl(uint32_t node_id) {
    const std::vector<uint32_t> extra_edges;
    update_node_impl_locked(node_id, extra_edges);
  }

  /// Reconnect node_id's neighbor list: candidates = live old + two-hop through
  /// deleted nodes + explicit extra edges, exact-L2 ranked, alpha-RNG pruned.
  void update_node_impl(uint32_t node_id, const std::vector<uint32_t> &extra_edges) {
    const DiskPageIO::NodeData nd = page_io_->read_node(node_id);
    update_node_impl_from_snapshot(node_id, nd, extra_edges);
  }

  void update_node_impl_from_snapshot(uint32_t node_id,
                                      const DiskPageIO::NodeData &nd,
                                      const std::vector<uint32_t> &extra_edges) {
    std::unordered_set<uint32_t> cand;
    const auto &removed_node_nbrs = update_ctx_.removed_node_nbrs_;
    for (const uint32_t nbr : nd.nbrs) {
      if (nbr == node_id) {
        continue;
      }
      if (slot_alloc_.is_deleted(nbr)) {
        const auto it = removed_node_nbrs.find(nbr);
        if (it != removed_node_nbrs.end()) {
          for (const uint32_t o : it->second) {
            if (o != node_id && !slot_alloc_.is_deleted(o)) {
              cand.insert(o);
            }
          }
        }
      } else {
        cand.insert(nbr);
      }
    }
    prune_and_write(node_id, nd.nbrs, cand, extra_edges);
  }

  void update_node_impl_locked(uint32_t node_id, const std::vector<uint32_t> &extra_edges) {
    std::lock_guard<std::mutex> node_lock(update_node_mutex(node_id));
    update_node_impl(node_id, extra_edges);
  }

  void update_node_impl_for_insert(uint32_t node_id, uint32_t inserted_slot) {
    const std::vector<uint32_t> extra_edges{inserted_slot};
    update_node_impl_locked(node_id, extra_edges);
  }

  void reconnect_inserted_neighbors_serial(const std::vector<uint32_t> &neighbors, uint32_t slot) {
    for (const uint32_t n : neighbors) {
      update_node_impl_for_insert(n, slot);
    }
  }

  void reconnect_inserted_neighbors_parallel(const std::vector<uint32_t> &neighbors,
                                             uint32_t slot) {
    const uint32_t workers =
        std::min<uint32_t>(update_reconnect_threads_, static_cast<uint32_t>(neighbors.size()));
    coro::thread_pool pool{{.thread_count = workers,
                            .on_thread_start_functor = nullptr,
                            .on_thread_stop_functor = nullptr}};
    auto reconnect_one = [this, &pool, slot](uint32_t node_id) -> coro::task<> {
      co_await pool.schedule();
      update_node_impl_for_insert(node_id, slot);
    };
    auto run = [&]() -> coro::task<> {
      std::vector<coro::task<>> tasks;
      tasks.reserve(neighbors.size());
      for (const uint32_t node_id : neighbors) {
        tasks.emplace_back(reconnect_one(node_id));
      }
      co_await coro::when_all(std::move(tasks));
    };
    try {
      coro::sync_wait(run());
    } catch (...) {
      pool.shutdown();
      throw;
    }
    pool.shutdown();
  }

  void reconnect_inserted_neighbors(const std::vector<uint32_t> &neighbors, uint32_t slot) {
    if (neighbors.empty()) {
      return;
    }
    if (update_reconnect_threads_ == 1 || neighbors.size() == 1) {
      reconnect_inserted_neighbors_serial(neighbors, slot);
      return;
    }
    reconnect_inserted_neighbors_parallel(neighbors, slot);
  }

  /// Lightweight consolidation: just strip dangling edges to tombstoned nodes.
  bool maybe_safety_net_reconnect() {
    if (!update_ctx_.needs_safety_net_reconnect(safety_net_ratio_,
                                                slot_alloc_.tombstone_count(),
                                                max_slot_id_,
                                                ops_since_last_insert_,
                                                safety_net_ops_)) {
      return false;
    }
    std::unordered_set<uint32_t> affected;
    for (const auto &entry : update_ctx_.removed_node_nbrs_) {
      for (const uint32_t nb : entry.second) {
        if (!slot_alloc_.is_deleted(nb)) {
          affected.insert(nb);
        }
      }
    }
    page_io_->clear_cache();
    for (const uint32_t nid : affected) {
      const DiskPageIO::NodeData nd = page_io_->read_node(nid);
      std::vector<uint32_t> live;
      for (const uint32_t nbr : nd.nbrs) {
        if (!slot_alloc_.is_deleted(nbr)) {
          live.push_back(nbr);
        }
      }
      if (live.size() != nd.nbrs.size()) {
        page_io_->write_node_neighbors(nid, static_cast<uint32_t>(live.size()), live.data());
      }
    }
    ops_since_last_insert_ = 0;
    ++safety_net_fires_;
    return true;
  }

  /// Grow labels_ on append; assign in place on slot reuse.
  void set_label(uint32_t slot, uint64_t label) {
    if (slot >= labels_.size()) {
      labels_.resize(static_cast<size_t>(slot) + 1, kNoLabel);
    }
    labels_[slot] = label;
  }

  void resize_thread_data_slot_capacity() {
    for (auto &td : thread_data_storage_) {
      td->resize_slot_capacity(max_slot_id_);
    }
  }

  std::mutex &update_node_mutex(uint32_t node_id) {
    return update_node_locks_[node_id % update_node_locks_.size()];
  }

  /// Order-independent equality of two neighbor id lists.
  static bool same_neighbor_set(std::vector<uint32_t> a, std::vector<uint32_t> b) {
    if (a.size() != b.size()) {
      return false;
    }
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    return a == b;
  }

  ThreadData *acquire() const {
    ThreadData *td = thread_data_pool_.pop();
    while (td == nullptr) {
      thread_data_pool_.wait_for_push_notify();
      td = thread_data_pool_.pop();
    }
    return td;
  }
  void release(ThreadData *td) const {
    thread_data_pool_.push(td);
    thread_data_pool_.push_notify_all();
  }

  void teardown() {
    if (reader_) {
      reader_->close();
      reader_->deregister_all_threads();
    }
    for (auto &td : thread_data_storage_) {
      if (td) {
        td->free_scratch();
      }
    }
    thread_data_storage_.clear();
    while (thread_data_pool_.pop() != nullptr) {
    }
    reader_.reset();
    page_io_.reset();
    update_ctx_.clear();
    updatable_ = false;
    loaded_ = false;
  }

  // --- meta.bin / ids.bin serialization ---
  static void write_meta(const std::string &p, const MetaHeader &m) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    if (!out) {
      throw std::runtime_error("DiskANNIndex: cannot write " + p);
    }
    const uint64_t magic = kMetaMagic;
    const uint32_t version = kMetaVersion;
    auto w = [&](const void *d, size_t n) {
      out.write(reinterpret_cast<const char *>(d), n);
    };
    w(&magic, sizeof(magic));
    w(&version, sizeof(version));
    w(&m.num_points, sizeof(m.num_points));
    w(&m.dim, sizeof(m.dim));
    w(&m.max_degree, sizeof(m.max_degree));
    w(&m.medoid, sizeof(m.medoid));
    w(&m.has_pq, sizeof(m.has_pq));
    w(&m.pq_n_chunks, sizeof(m.pq_n_chunks));
    w(&m.node_len, sizeof(m.node_len));
    w(&m.nodes_per_sector, sizeof(m.nodes_per_sector));
    w(&m.max_slot_id, sizeof(m.max_slot_id));  // v2
    w(&m.live_count, sizeof(m.live_count));    // v2
    if (!out) {
      throw std::runtime_error("DiskANNIndex: meta write failed " + p);
    }
  }

  static MetaHeader read_meta(const std::string &p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
      throw std::runtime_error("DiskANNIndex::load: cannot open meta " + p);
    }
    uint64_t magic = 0;
    uint32_t version = 0;
    MetaHeader m;
    auto r = [&](void *d, size_t n) {
      in.read(reinterpret_cast<char *>(d), n);
    };
    r(&magic, sizeof(magic));
    r(&version, sizeof(version));
    r(&m.num_points, sizeof(m.num_points));
    r(&m.dim, sizeof(m.dim));
    r(&m.max_degree, sizeof(m.max_degree));
    r(&m.medoid, sizeof(m.medoid));
    r(&m.has_pq, sizeof(m.has_pq));
    r(&m.pq_n_chunks, sizeof(m.pq_n_chunks));
    r(&m.node_len, sizeof(m.node_len));
    r(&m.nodes_per_sector, sizeof(m.nodes_per_sector));
    if (!in) {
      throw std::runtime_error("DiskANNIndex::load: meta.bin truncated/corrupt " + p);
    }
    if (magic != kMetaMagic) {
      throw std::runtime_error("DiskANNIndex::load: bad meta magic " + p);
    }
    if (version == 1) {
      // v1 predates in-place updates: every slot is live, capacity == num_points.
      m.max_slot_id = m.num_points;
      m.live_count = m.num_points;
    } else if (version == 2) {
      r(&m.max_slot_id, sizeof(m.max_slot_id));
      r(&m.live_count, sizeof(m.live_count));
      if (!in) {
        throw std::runtime_error("DiskANNIndex::load: meta.bin truncated/corrupt " + p);
      }
    } else {
      throw std::runtime_error("DiskANNIndex::load: unsupported meta version " + p);
    }
    if (m.num_points == 0 || m.dim == 0) {
      throw std::runtime_error("DiskANNIndex::load: zero num_points/dim in meta " + p);
    }
    return m;
  }

  static void write_ids(const std::string &p, const uint64_t *labels, uint64_t n) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    if (!out) {
      throw std::runtime_error("DiskANNIndex: cannot write " + p);
    }
    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
    out.write(reinterpret_cast<const char *>(labels),
              static_cast<std::streamsize>(n * sizeof(uint64_t)));
    if (!out) {
      throw std::runtime_error("DiskANNIndex: ids write failed " + p);
    }
  }

  void read_ids(const std::string &p, uint64_t expected_n) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
      throw std::runtime_error("DiskANNIndex::load: cannot open ids " + p);
    }
    uint64_t count = 0;
    in.read(reinterpret_cast<char *>(&count), sizeof(count));
    if (!in || count != expected_n) {
      throw std::runtime_error("DiskANNIndex::load: ids.bin count mismatch " + p);
    }
    labels_.assign(count, 0);
    in.read(reinterpret_cast<char *>(labels_.data()),
            static_cast<std::streamsize>(count * sizeof(uint64_t)));
    if (!in) {
      throw std::runtime_error("DiskANNIndex::load: ids.bin truncated " + p);
    }
  }

  // metadata
  uint64_t dim_ = 0;
  uint32_t max_degree_ = 0;
  uint32_t medoid_ = 0;
  bool has_pq_ = false;
  uint32_t pq_n_chunks_ = 0;
  uint32_t beam_width_ = 4;
  uint32_t num_pool_ = 0;
  bool loaded_ = false;
  DiskLayoutGeometry geom_;

  // in-memory artifacts
  std::vector<uint64_t> labels_;
  NodeCache cache_;
  PQTable pq_;
  std::unique_ptr<AlignedFileReader> reader_;

  // thread-scratch pool
  std::vector<std::unique_ptr<ThreadData>> thread_data_storage_;
  mutable ::ConcurrentQueue<ThreadData *> thread_data_pool_{nullptr};

  // in-place update state (active only when updatable_)
  bool updatable_ = false;
  uint64_t max_slot_id_ = 0;  ///< file capacity in slots (valid-id bound; only grows)
  uint64_t live_count_ = 0;   ///< live (non-tombstoned) vector count
  std::string index_dir_;     ///< saved at load for flush() output paths
  std::unique_ptr<DiskPageIO> page_io_;
  SlotAllocator slot_alloc_;
  DiskUpdateContext update_ctx_;
  mutable std::shared_mutex update_mutex_;  ///< shared search, exclusive mutation
  std::mutex slot_mutex_;
  std::mutex pq_mutex_;
  std::array<std::mutex, kUpdateNodeLockStripes> update_node_locks_;
  uint64_t ops_since_last_insert_ = 0;
  uint64_t safety_net_fires_ = 0;
  float update_alpha_ = 1.2f;
  uint32_t update_search_l_ = 100;
  double safety_net_ratio_ = 0.05;
  uint64_t safety_net_ops_ = 16;
  uint32_t update_insert_threads_ = kDefaultDiskANNUpdateInsertThreads;
  uint32_t update_reconnect_threads_ = kDefaultDiskANNUpdateReconnectThreads;
};

}  // namespace alaya::diskann
