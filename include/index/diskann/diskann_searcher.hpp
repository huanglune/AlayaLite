/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "deleted_neighbor_cache.hpp"
#include "diskann_params.hpp"
#include "index/neighbor.hpp"
#include "inserted_edge_cache.hpp"
#include "storage/buffer/disk_read_awaitable.hpp"

#if defined(__linux__)
  #include "coro/task.hpp"
#endif
#include "simd/distance_ip.hpp"
#include "simd/distance_l2.hpp"
#include "storage/buffer/replacer/clock.hpp"
#include "storage/diskann/diskann_storage.hpp"
#include "utils/candidate_list.hpp"
#include "utils/locks.hpp"
#include "utils/log.hpp"
#include "utils/macros.hpp"
#include "utils/math.hpp"
#include "utils/types.hpp"
#include "visited_list.hpp"

namespace alaya {

/**
 * @brief High-performance DiskANN searcher with disk-based beam search.
 *
 * Uses in-memory graph topology to avoid redundant block reads for neighbor
 * lists, with BufferPool caching for disk I/O.
 *
 * @tparam DataType Vector element type (default: float)
 * @tparam IDType Node ID type (default: uint32_t)
 */
template <typename DataType = float,
          typename IDType = uint32_t,
          ReplacerStrategy ReplacerType = ClockReplacer>
class DiskANNSearcher {
 public:
  using DistanceType = float;
  using NeighborType = Neighbor<IDType, DistanceType>;
  using BufferPoolType = BufferPool<IDType, ReplacerType>;
  using StorageType = DiskANNStorage<DataType, IDType, ReplacerType>;
  using NodeRef = typename StorageType::NodeRef;
  using Params = DiskANNSearchParams;
  using Result = SearchResult<IDType, DistanceType>;

  /// Distance function signature: (query, vector, dim) -> distance
  using DistanceFn = auto (*)(const DataType *, const DataType *, size_t) -> float;
  static constexpr IDType kInvalidID = static_cast<IDType>(-1);
  struct SearchContext {
    CandidateList<DistanceType, IDType> candidates_;
    diskann::VisitedList visited_;
    std::vector<IDType> beam_queue_;
    std::vector<IDType> next_beam_queue_;
    std::vector<uint32_t> next_beam_io_;  // For IO deduplication

    explicit SearchContext(uint32_t capacity, uint32_t max_degree, uint32_t beam_width)
        : candidates_(capacity, 200) {
      beam_queue_.reserve(beam_width * 2);
      next_beam_queue_.reserve(beam_width * 2);
      next_beam_io_.reserve(beam_width * max_degree);
      visited_.reset(capacity);
    }
    void clear() {
      candidates_.clear();
      beam_queue_.clear();
      next_beam_queue_.clear();
      next_beam_io_.clear();
    }
  };

 private:
  std::unique_ptr<BufferPoolType> buffer_pool_;
  std::unique_ptr<StorageType> storage_;

  uint32_t dimension_{0};
  uint32_t max_degree_{0};
  std::atomic<uint32_t> num_points_{0};
  std::atomic<uint32_t> capacity_{0};
  IDType medoid_id_{0};

  DistanceFn dist_fn_{simd::l2_sqr};  ///< Distance function (L2/IP, set at open time)

 public:
  DiskANNSearcher() = default;
  ALAYA_NON_COPYABLE_BUT_MOVABLE(DiskANNSearcher);
  ~DiskANNSearcher() = default;

  /**
   * @brief Open a disk index for searching.
   *
   * @param base_path Base path for index files (without extension)
   * @param cache_capacity Number of 4KB pages in the buffer pool
   */
  /**
   * @brief Compute BufferPool shard count based on worker count.
   * Formula: max(16, num_workers * 4)
   */
  static auto compute_num_shards(size_t num_workers) -> size_t {
    return std::max(size_t{16}, num_workers * 4);
  }

  void open(std::string_view base_path,
            size_t cache_capacity = 4096,
            bool writable = false,
            size_t num_shards = 16) {
    LOG_INFO("DiskANNSearcher: Opening index at {}", base_path);

    buffer_pool_ = std::make_unique<BufferPoolType>(cache_capacity, kDataBlockSize, num_shards);
    storage_ = std::make_unique<StorageType>(buffer_pool_.get());
    storage_->open(base_path, writable);

    dimension_ = storage_->dimension();
    max_degree_ = storage_->max_degree();

    num_points_.store(storage_->num_points());
    capacity_.store(storage_->capacity());
    medoid_id_ = static_cast<IDType>(storage_->entry_point());

    // Select distance function based on metric type stored in the index
    auto metric = static_cast<MetricType>(storage_->metric_type());
    switch (metric) {
      case MetricType::IP:
      case MetricType::COS:
        dist_fn_ = simd::ip_sqr;
        break;
      default:
        dist_fn_ = simd::l2_sqr;
        break;
    }

    // Initialize deleted neighbor cache for graph repair (4% of capacity)
    deleted_neighbors_.emplace(DeletedNeighborCache<IDType>::with_index_capacity(capacity_.load()));
    inserted_edge_cache_ =
        InsertedEdgeCache<IDType>::with_index_capacity(capacity_.load(), kNumNodeLocks);

    size_t preload_blocks =
        std::min(size_t{100}, static_cast<size_t>(storage_->data().num_blocks()));
    for (size_t i = 0; i < preload_blocks; ++i) {
      storage_->data().preload_block(i);
    }

    LOG_INFO("DiskANNSearcher: Opened index, capacity={}, num_point={}, dim={}, R={}, medoid={}",
             capacity_,
             num_points_,
             dimension_,
             max_degree_,
             medoid_id_);
  }

  /**
   * @brief Search for k nearest neighbors using disk-based beam search.
   */
  auto search(const DataType *query, uint32_t topk, const Params &params = Params{}) -> Result {
    if (!is_open()) {
      throw std::runtime_error("DiskANNSearcher: Index not loaded");
    }
    auto &ctx = get_search_context(params);
    ctx.clear();

    uint32_t ef = std::max(params.ef_search_, topk);
    ctx.candidates_.resize(ef);
    ctx.visited_.reset(capacity_);

    auto result = search_disk(ctx, query, topk, params);

    // Convert internal slot IDs to external IDs
    map_result_ids(result);
    return result;
  }

#if defined(__linux__)
  /**
   * @brief Coroutine search using async I/O with beam search helpers.
   *
   * Must be scheduled as a top-level coroutine via Scheduler (not nested).
   * Uses provided SearchContext and writes result to the output parameter.
   *
   * @param ctx     Pre-allocated SearchContext (one per concurrent search)
   * @param query   Query vector
   * @param topk    Number of results
   * @param params  Search parameters
   * @param result  Output: search result written here on completion
   */
  auto co_search(SearchContext &ctx,
                 const DataType *query,
                 uint32_t topk,
                 const Params &params,
                 Result &result) -> coro::task<> {
    ctx.clear();
    uint32_t ef = std::max(params.ef_search_, topk);
    ctx.candidates_.resize(ef);
    ctx.visited_.reset(capacity_);

    init_beam_search(ctx, query);

    uint32_t beam_width = std::max(1U, params.beam_width_);
    while (!ctx.beam_queue_.empty()) {
      expand_beam(ctx);

      if (ctx.next_beam_queue_.empty()) {
        if (!refill_beam(ctx, beam_width)) {
          break;
        }
        continue;
      }

      dedup_io_blocks(ctx);

      // Async prefetch + yield
      using DataFileT = typename StorageType::DataFileType;
      auto pending = storage_->data().begin_async_prefetch(ctx.next_beam_io_);
      while (!pending.empty() && !DataFileT::all_async_ready(pending)) {
        co_await YieldAwaitable{};
      }
      bool io_error = DataFileT::any_async_error(pending);
      pending.clear();

      if (!io_error) {
        sort_by_block_locality(ctx);
        compute_beam_distances(ctx, query);
      }

      refill_beam(ctx, beam_width);
    }

    result = ctx.candidates_.to_search_result(topk);
    map_result_ids(result);
  }

  /**
   * @brief Coroutine insert: same logic as insert() but uses async I/O
   *        for the greedy search phase.
   *
   * Must be scheduled as a top-level coroutine via Scheduler.
   */
  auto co_insert(const DataType *vector,
                 IDType external_id,
                 const DiskANNInsertParams &params = DiskANNInsertParams{}) -> coro::task<> {
    insert(vector, external_id, params);
    co_return;
  }

  /**
   * @brief Coroutine delete: same logic as delete_vector() but uses
   *        async I/O for reading the neighbor list before invalidation.
   */
  auto co_delete_vector(IDType external_id) -> coro::task<> {
    delete_vector(external_id);
    co_return;
  }
#endif

  /**
   * @brief Batch search for multiple queries.
   *
   * Runs single-query search sequentially for each query vector.
   *
   * @param queries Query vectors (num_queries * dimension)
   * @param num_queries Number of queries
   * @param topk Number of neighbors per query
   * @param params Search parameters
   * @return Vector of SearchResult, one per query
   */
  auto batch_search(const DataType *queries,
                    uint32_t num_queries,
                    uint32_t topk,
                    const Params &params = Params{}) -> std::vector<Result> {
    std::vector<Result> results(num_queries);

#pragma omp parallel for schedule(dynamic, 1) num_threads(params.num_threads_)
    for (uint32_t i = 0; i < num_queries; ++i) {
      results[i] = std::move(search(queries + static_cast<size_t>(i) * dimension_, topk, params));
    }
    return results;
  }

  // =========================================================================
  // Accessors
  // =========================================================================

  [[nodiscard]] auto is_open() const -> bool { return storage_ != nullptr && storage_->is_open(); }
  [[nodiscard]] auto num_points() const -> uint32_t { return num_points_.load(); }
  [[nodiscard]] auto dimension() const -> uint32_t { return dimension_; }
  [[nodiscard]] auto capacity() const -> uint32_t { return capacity_.load(); }
  [[nodiscard]] auto buffer_pool() -> BufferPoolType & { return *buffer_pool_; }
  [[nodiscard]] auto bypasses_page_cache() const -> bool {
    return storage_ != nullptr && storage_->data().bypasses_page_cache();
  }

  // =========================================================================
  // Mutation (for incremental insert support)
  // =========================================================================

  void reserve_capacity(uint32_t new_capacity) {
    require_writable("reserve_capacity");
    if (new_capacity <= capacity_) {
      return;
    }

    storage_->meta().grow(new_capacity);
    storage_->data().grow(storage_->meta().capacity());
    capacity_.store(storage_->capacity());
  }

  /**
   * @brief Delete a vector by external ID (lazy delete with graph repair cache).
   *
   * Marks the internal slot as invalid, removes the ID mapping, and caches
   * the deleted node's neighbor list for two-hop repair during future connect tasks.
   *
   * @param external_id External ID of the vector to delete
   */
  void delete_vector(IDType external_id) {
    std::lock_guard<std::mutex> update_guard(update_mutex_);
    require_writable("delete_vector");

    auto &meta = storage_->meta();

    // Resolve external -> internal (handle builder-built identity mapping)
    uint32_t internal_id = resolve_external(meta, external_id);
    if (internal_id == kInvalidMapping) {
      throw std::invalid_argument("DiskANNSearcher::delete_vector: external ID " +
                                  std::to_string(external_id) + " not found");
    }

    // Check already deleted
    if (!meta.is_valid(internal_id)) {
      throw std::invalid_argument("DiskANNSearcher::delete_vector: external ID " +
                                  std::to_string(external_id) + " already deleted");
    }

    // Protect entry point
    if (static_cast<IDType>(internal_id) == medoid_id_) {
      throw std::logic_error("DiskANNSearcher::delete_vector: cannot delete entry point");
    }

    // Cache the neighbor list before invalidation (for two-hop repair)
    std::vector<IDType> cached_neighbors;
    {
      auto node = storage_->data().get_node(internal_id);
      auto nbrs = node.neighbors();
      cached_neighbors.reserve(nbrs.size());
      for (auto nbr : nbrs) {
        if (nbr != kInvalidID) {
          cached_neighbors.push_back(nbr);
        }
      }
      if (deleted_neighbors_.has_value()) {
        deleted_neighbors_->put(static_cast<IDType>(internal_id),
                                std::vector<IDType>(cached_neighbors));
      }
    }

    // Mark invalid and remove mapping (if explicit mapping exists)
    meta.set_invalid(internal_id);
    if (meta.has_mapping(static_cast<uint32_t>(external_id))) {
      meta.remove_mapping(static_cast<uint32_t>(external_id));
    }
    --num_points_;

    // Proactive neighbor repair: re-prune each valid former neighbor
    // Batch-prefetch neighbor blocks before the repair loop
    {
      std::vector<uint32_t> repair_block_ids;
      repair_block_ids.reserve(cached_neighbors.size());
      for (auto nbr_id : cached_neighbors) {
        if (meta.is_valid(nbr_id)) {
          repair_block_ids.push_back(storage_->data().block_id_of(nbr_id));
        }
      }
      std::sort(repair_block_ids.begin(), repair_block_ids.end());
      repair_block_ids.erase(std::unique(repair_block_ids.begin(), repair_block_ids.end()),
                             repair_block_ids.end());
      storage_->data().prefetch_blocks(repair_block_ids);

      for (auto nbr_id : cached_neighbors) {
        if (meta.is_valid(nbr_id)) {
          connect_task(nbr_id, kInvalidID, repair_alpha_);
        }
      }
    }
  }

  /**
   * @brief Insert a new vector with graph connectivity maintenance.
   *
   * Allocates a slot, writes the vector, finds neighbors via greedy search,
   * prunes with RobustPrune, and executes connect tasks for bidirectional edges.
   *
   * @param vector Vector data (dimension elements)
   * @param external_id User-visible external ID
   * @param params Insert parameters (ef_construction, alpha, beam_width)
   */
  void insert(const DataType *vector,
              IDType external_id,
              const DiskANNInsertParams &params = DiskANNInsertParams{}) {
    std::lock_guard<std::mutex> update_guard(update_mutex_);
    require_writable("insert");

    auto &meta = storage_->meta();

    // Check for duplicate external ID
    if (resolve_external(meta, external_id) != kInvalidMapping) {
      throw std::invalid_argument("DiskANNSearcher::insert: external ID " +
                                  std::to_string(external_id) + " already exists");
    }

    // Allocate slot (auto-grows if needed)
    int32_t slot = storage_->allocate_node_id();
    auto internal_id = static_cast<uint32_t>(slot);
    capacity_.store(storage_->capacity());

    // Write vector and empty neighbor list
    {
      auto node = storage_->data().get_node(internal_id);
      node.set_vector(std::span<const DataType>(vector, dimension_));
      node.set_neighbors(std::span<const IDType>());
    }

    // Find candidate neighbors via greedy search
    auto candidates = greedy_search_for_insert(vector, params);

    // Prune to select initial neighbors
    std::vector<IDType> pruned;
    robust_prune_disk(vector, candidates, params.alpha_, pruned);

    // Write pruned neighbor list
    {
      auto node = storage_->data().get_node(internal_id);
      node.set_neighbors(std::span<const IDType>(pruned));
    }

    // Publish the new node before graph repair so other concurrent inserts can
    // incorporate it during connect_task candidate generation.
    auto ext = static_cast<uint32_t>(external_id);
    if (ext >= meta.capacity()) {
      uint32_t new_cap = math::round_up_pow2(ext + 1, 64);
      meta.grow(new_cap);
      storage_->data().grow(meta.capacity());
      capacity_.store(storage_->capacity());
    }
    meta.set_valid(internal_id);
    meta.insert_mapping(ext, internal_id);
    ++num_points_;

    // Execute connect tasks for each selected neighbor (bidirectional edges).
    for (auto nbr_id : pruned) {
      if (nbr_id == kInvalidID) {
        break;
      }
      connect_task(nbr_id, static_cast<IDType>(internal_id), params.alpha_);
    }

    // Record reverse edges: each pruned neighbor gains new_node as incoming neighbor.
    for (auto nbr_id : pruned) {
      if (nbr_id == kInvalidID) {
        break;
      }
      size_t shard = nbr_id % kNumNodeLocks;
      SpinLockGuard guard(node_locks_[shard]);
      inserted_edge_cache_.add(shard, nbr_id, static_cast<IDType>(internal_id));
    }
  }

  void flush() {
    if (storage_ && storage_->is_open()) {
      storage_->data().flush();
      storage_->save_meta();
    }
  }

  /// Telemetry: number of entries in the InsertedEdgeCache.
  [[nodiscard]] auto inserted_edge_cache_size() const -> size_t {
    return inserted_edge_cache_.size();
  }

  /// Telemetry: number of entries in the DeletedNeighborCache.
  [[nodiscard]] auto deleted_neighbor_cache_size() const -> size_t {
    return deleted_neighbors_.has_value() ? deleted_neighbors_->size() : 0;
  }

  /// Telemetry: number of free slots in the meta freelist.
  [[nodiscard]] auto freelist_depth() const -> size_t {
    return storage_ ? storage_->meta().freelist_depth() : 0;
  }

 private:
  /**
   * @brief Map search result IDs from internal slots to external IDs.
   *
   * For builder-built indices (no explicit mapping), internal == external (identity).
   * For inserted vectors, uses reverse_map to translate.
   */
  void map_result_ids(Result &result) const {
    auto &meta = storage_->meta();
    for (auto &id : result.ids_) {
      if (id == kInvalidID) {
        continue;
      }
      uint32_t ext = meta.resolve_reverse(static_cast<uint32_t>(id));
      if (ext != kInvalidMapping) {
        id = static_cast<IDType>(ext);
      }
      // If no reverse mapping, keep internal ID (identity mapping for builder-built)
    }
  }

  /**
   * @brief Resolve external ID to internal slot via mapping table.
   *
   * Builder writes identity mapping at build time, so all indices have
   * explicit mappings. No fallback logic needed.
   */
  static auto resolve_external(const MetaFile &meta, IDType external_id) -> uint32_t {
    return meta.resolve(static_cast<uint32_t>(external_id));
  }

  void require_writable(std::string_view operation) const {
    if (!is_open()) {
      throw std::runtime_error("DiskANNSearcher: Index not loaded");
    }
    if (!storage_->data().is_writable()) {
      throw std::runtime_error("DiskANNSearcher: " + std::string(operation) +
                               " requires writable=true");
    }
  }

  auto get_search_context(const Params &params) -> SearchContext & {
    static thread_local std::unique_ptr<SearchContext> ctx;
    static thread_local uint32_t ctx_capacity = 0;
    static thread_local uint32_t ctx_max_degree = 0;
    static thread_local uint32_t ctx_beam_width = 0;

    uint32_t beam_width = params.beam_width_ > 0 ? params.beam_width_ : 128;
    uint32_t cur_capacity = capacity_.load();
    if (!ctx || ctx_capacity != cur_capacity || ctx_max_degree != max_degree_ ||
        ctx_beam_width != beam_width) {
      ctx = std::make_unique<SearchContext>(cur_capacity, max_degree_, beam_width);
      ctx_capacity = cur_capacity;
      ctx_max_degree = max_degree_;
      ctx_beam_width = beam_width;
    }
    return *ctx;
  }

  // =========================================================================
  // Helper: Compute Exact Distance using NodeRef
  // =========================================================================
  auto compute_exact_distance(const DataType *query, IDType node_id) -> float {
    auto node = storage_->data().get_node(node_id);
    auto vec_span = node.vector();
    return dist_fn_(query, vec_span.data(), dimension_);
  }

  // =========================================================================
  // Shared helpers: pure computation, reusable by sync and async paths
  // =========================================================================

  /**
   * @brief Process a pre-fetched node's neighbors, adding unvisited valid ones
   *        to the beam queues. Shared by sync and async beam search paths.
   *
   * @param ctx  SearchContext with visited set and beam queues
   * @param node Already-fetched NodeRef (no I/O inside this method)
   */
  void collect_unvisited_from_node(SearchContext &ctx, const NodeRef &node) {
    auto neighbors_span = node.neighbors();

    for (size_t i = 0; i < neighbors_span.size(); ++i) {
      IDType nbr = neighbors_span[i];
      if (nbr == kInvalidID) {
        break;
      }

      if (i + 1 < neighbors_span.size() && neighbors_span[i + 1] != kInvalidID) {
        ctx.visited_.prefetch(neighbors_span[i + 1]);
      }

      if (ctx.visited_.is_visited(nbr)) {
        continue;
      }

      if (!storage_->meta().is_valid(nbr)) {
        ctx.visited_.mark(nbr);
        continue;
      }

      ctx.visited_.mark(nbr);
      ctx.next_beam_io_.push_back(storage_->data().block_id_of(nbr));
      ctx.next_beam_queue_.push_back(nbr);
    }
  }

  /// Sync wrapper: fetches the node then delegates to shared helper.
  void collect_unvisited_neighbors(SearchContext &ctx, IDType node_id) {
    auto node = storage_->data().get_node(node_id);
    collect_unvisited_from_node(ctx, node);
  }

  // =========================================================================
  // Beam search step helpers
  // =========================================================================

  /// Seed beam search from medoid: compute distance, insert into candidates,
  /// mark visited, push to beam queue.
  void init_beam_search(SearchContext &ctx, const DataType *query) {
    float start_dist = compute_exact_distance(query, medoid_id_);
    ctx.candidates_.insert(medoid_id_, start_dist);
    ctx.visited_.mark(medoid_id_);
    ctx.beam_queue_.push_back(ctx.candidates_.pop());
  }

  /// Expand beam: collect unvisited neighbors for all nodes in beam_queue_.
  void expand_beam(SearchContext &ctx) {
    ctx.next_beam_queue_.clear();
    ctx.next_beam_io_.clear();
    for (IDType node_id : ctx.beam_queue_) {
      collect_unvisited_neighbors(ctx, node_id);
    }
  }

  /// Sort + unique on next_beam_io_ for I/O deduplication.
  void dedup_io_blocks(SearchContext &ctx) {
    std::sort(ctx.next_beam_io_.begin(), ctx.next_beam_io_.end());
    ctx.next_beam_io_.erase(std::unique(ctx.next_beam_io_.begin(), ctx.next_beam_io_.end()),
                            ctx.next_beam_io_.end());
  }

  /// Sort next_beam_queue_ by block ID (then node ID) for sequential I/O.
  void sort_by_block_locality(SearchContext &ctx) {
    std::sort(ctx.next_beam_queue_.begin(),
              ctx.next_beam_queue_.end(),
              [this](IDType lhs, IDType rhs) -> bool {
                auto &data = storage_->data();
                uint32_t lhs_block = data.block_id_of(lhs);
                uint32_t rhs_block = data.block_id_of(rhs);
                return lhs_block == rhs_block ? lhs < rhs : lhs_block < rhs_block;
              });
  }

  /// Compute exact distances for all nodes in next_beam_queue_, insert into candidates.
  void compute_beam_distances(SearchContext &ctx, const DataType *query) {
    for (auto nbr : ctx.next_beam_queue_) {
      float dist = compute_exact_distance(query, nbr);
      ctx.candidates_.insert(nbr, dist);
    }
  }

  /// Clear beam_queue_, pop up to beam_width candidates. Returns false if empty.
  auto refill_beam(SearchContext &ctx, uint32_t beam_width) -> bool {
    ctx.beam_queue_.clear();
    for (uint32_t i = 0; i < beam_width && ctx.candidates_.has_next(); ++i) {
      ctx.beam_queue_.push_back(ctx.candidates_.pop());
    }
    return !ctx.beam_queue_.empty();
  }

  // =========================================================================
  // Search orchestrators
  // =========================================================================

  auto search_disk(SearchContext &ctx, const DataType *query, uint32_t topk, const Params &params)
      -> Result {
    init_beam_search(ctx, query);

    uint32_t beam_width = std::max(1U, params.beam_width_);
    while (!ctx.beam_queue_.empty()) {
      expand_beam(ctx);

      if (ctx.next_beam_queue_.empty()) {
        if (!refill_beam(ctx, beam_width)) {
          break;
        }
        continue;
      }

      dedup_io_blocks(ctx);
      storage_->data().prefetch_blocks(ctx.next_beam_io_);
      sort_by_block_locality(ctx);
      compute_beam_distances(ctx, query);
      refill_beam(ctx, beam_width);
    }

    return ctx.candidates_.to_search_result(topk);
  }

  // =========================================================================
  // Insert/Delete helpers
  // =========================================================================

  /**
   * @brief Greedy beam search returning candidates with distances (for insert).
   *
   * Reuses search_disk logic but returns vector<Neighbor> instead of SearchResult,
   * using ef_construction as the search budget.
   */
  auto greedy_search_for_insert(const DataType *vector, const DiskANNInsertParams &params)
      -> std::vector<NeighborType> {
    Params search_params;
    search_params.ef_search_ = params.ef_construction_;
    search_params.beam_width_ = params.beam_width_;

    auto &ctx = get_search_context(search_params);
    ctx.clear();

    uint32_t ef = params.ef_construction_;
    ctx.candidates_.resize(ef);
    ctx.visited_.reset(capacity_);

    // Run disk search
    auto result = search_disk(ctx, vector, ef, search_params);

    // Convert SearchResult to vector<Neighbor> with distances
    std::vector<NeighborType> candidates;
    candidates.reserve(result.ids_.size());
    for (size_t i = 0; i < result.ids_.size(); ++i) {
      if (result.ids_[i] != kInvalidID) {
        candidates.emplace_back(result.ids_[i], result.distances_[i]);
      }
    }
    return candidates;
  }

  // =========================================================================
  // === Graph Repair ===
  //
  // Pruning, connect_task, and associated state for insert/delete graph
  // maintenance. All repair-specific constants, members, and methods are
  // grouped here.
  // =========================================================================

  /// Striped locks for connect_task thread safety (1024 locks)
  static constexpr size_t kNumNodeLocks = 1024;
  std::vector<SpinLock> node_locks_{kNumNodeLocks};

  /// LRU cache for deleted nodes' neighbor lists (for two-hop graph repair)
  std::optional<DeletedNeighborCache<IDType>> deleted_neighbors_;

  /// Reverse-edge tracking keyed by target node; shard i is protected by node_locks_[i].
  InsertedEdgeCache<IDType> inserted_edge_cache_;

  /// Alpha parameter for repair connect_tasks triggered by deletion.
  float repair_alpha_{1.2F};

  /// Serialize graph updates so concurrent coroutine inserts/deletes preserve
  /// graph integrity even if the underlying update path is not lock-free.
  std::mutex update_mutex_;

  /**
   * @brief Alpha-dominance pruning shared helper.
   *
   * Takes a callable `get_vector(IDType) -> const DataType*` to retrieve
   * vector data, enabling both sync (get_node) and async (pre-fetched) paths.
   *
   * @param source_vec Source node's vector
   * @param candidates Candidate neighbors with pre-computed distances to source
   * @param alpha Pruning alpha
   * @param get_vector Callable returning vector pointer for a given node ID
   * @return Pruned neighbor IDs, padded with kInvalidID to max_degree
   */
  template <typename GetVectorFn>
  void robust_prune_impl(const DataType *source_vec,
                         std::vector<NeighborType> &candidates,
                         float alpha,
                         GetVectorFn get_vector,
                         std::vector<IDType> &output) {
    (void)source_vec;
    std::sort(candidates.begin(), candidates.end());

    auto inflate = [alpha](float dist) -> float {
      return dist >= 0 ? dist * alpha : dist / alpha;
    };

    output.clear();
    output.reserve(max_degree_);

    for (const auto &cand : candidates) {
      if (output.size() >= max_degree_) {
        break;
      }

      bool dominated = false;
      for (auto selected_id : output) {
        const DataType *cand_vec = get_vector(cand.id_);
        const DataType *selected_vec = get_vector(selected_id);
        float dist_cand_selected = dist_fn_(cand_vec, selected_vec, dimension_);

        if (inflate(dist_cand_selected) < cand.distance_) {
          dominated = true;
          break;
        }
      }

      if (!dominated) {
        output.push_back(cand.id_);
      }
    }

    output.resize(max_degree_, kInvalidID);
  }

  void ensure_neighbor_present(std::vector<IDType> &neighbors, IDType required_id) const {
    if (required_id == kInvalidID || required_id == medoid_id_) {
      return;
    }

    for (auto id : neighbors) {
      if (id == required_id) {
        return;
      }
    }

    for (auto &id : neighbors) {
      if (id == kInvalidID) {
        id = required_id;
        return;
      }
    }

    if (!neighbors.empty()) {
      neighbors.back() = required_id;
    }
  }

  /// Sync wrapper: uses get_node() for vector access.
  void robust_prune_disk(const DataType *source_vec,
                         std::vector<NeighborType> &candidates,
                         float alpha,
                         std::vector<IDType> &output) {
    robust_prune_impl(
        source_vec,
        candidates,
        alpha,
        [this](IDType id) -> const DataType * {
          return storage_->data().get_node(id).vector().data();
        },
        output);
  }

  /**
   * @brief Yi's connect task: unified graph repair for insert + delete.
   *
   * Reads vertex_v's current neighbors, expands candidates (adds new_node,
   * removes deleted neighbors, adds two-hop from cache), prunes via
   * robust_prune_disk, and writes back the updated neighbor list.
   *
   * @param vertex_v Node whose neighbor list is being updated
   * @param new_node_id The newly inserted node to add as candidate
   * @param alpha Pruning alpha parameter
   */
  /**
   * @brief Build connect-task candidates from pre-fetched node data.
   *
   * Shared helper: collects candidates from current neighbors, handles
   * deleted-neighbor two-hop repair, adds new node, and deduplicates.
   * Does NOT call get_node() — all vectors are accessed via get_vector.
   *
   * @param vertex_vec    Pre-fetched vertex_v vector
   * @param vertex_v      Vertex whose neighbor list is being updated
   * @param new_node_id   The newly inserted node
   * @param current_nbrs  Pre-fetched current neighbor list of vertex_v
   * @param get_vector    Callable returning vector pointer for a given node ID
   * @return Deduplicated candidate list ready for pruning
   */
  template <typename GetVectorFn>
  void build_connect_candidates(const DataType *vertex_vec,
                                IDType vertex_v,
                                IDType new_node_id,
                                std::span<const IDType> current_nbrs,
                                GetVectorFn get_vector,
                                std::vector<NeighborType> &candidates) {
    auto &meta = storage_->meta();

    candidates.clear();
    candidates.reserve(max_degree_ * 2);

    // Add the new node (skip for repair-only mode when new_node_id == kInvalidID)
    if (new_node_id != kInvalidID) {
      const DataType *new_vec = get_vector(new_node_id);
      float dist = dist_fn_(vertex_vec, new_vec, dimension_);
      candidates.emplace_back(new_node_id, dist);
    }

    // Process current neighbors
    for (auto nbr : current_nbrs) {
      if (nbr == kInvalidID) {
        break;
      }
      if (new_node_id != kInvalidID && nbr == new_node_id) {
        continue;
      }

      if (!meta.is_valid(nbr)) {
        // Deleted neighbor: add two-hop candidates from cache (capped)
        auto cached = deleted_neighbors_.has_value() ? deleted_neighbors_->get(nbr) : std::nullopt;
        if (cached.has_value()) {
          size_t two_hop_count = 0;
          for (auto two_hop : cached.value()) {
            if (two_hop_count >= kMaxTwoHopCandidates) {
              break;
            }
            if (two_hop == kInvalidID || two_hop == vertex_v || !meta.is_valid(two_hop)) {
              continue;
            }
            const DataType *hop_vec = get_vector(two_hop);
            float dist = dist_fn_(vertex_vec, hop_vec, dimension_);
            candidates.emplace_back(two_hop, dist);
            ++two_hop_count;
          }
        }
        continue;
      }

      // Valid existing neighbor
      const DataType *nbr_vec = get_vector(nbr);
      float dist = dist_fn_(vertex_vec, nbr_vec, dimension_);
      candidates.emplace_back(nbr, dist);
    }

    // Consume reverse-edge hints for vertex_v: add recently inserted nodes as candidates.
    // Note: caller (connect_task) already holds node_locks_[vertex_v % kNumNodeLocks].
    {
      auto inserted_ids = inserted_edge_cache_.consume(vertex_v % kNumNodeLocks, vertex_v);
      for (auto inserted_id : inserted_ids) {
        if (inserted_id == kInvalidID || inserted_id == vertex_v || !meta.is_valid(inserted_id)) {
          continue;
        }
        const DataType *ins_vec = get_vector(inserted_id);
        float dist = dist_fn_(vertex_vec, ins_vec, dimension_);
        candidates.emplace_back(inserted_id, dist);
      }
    }

    // Deduplicate by ID: sort by id so duplicates are adjacent, keep closest
    std::sort(candidates.begin(),
              candidates.end(),
              [](const NeighborType &a, const NeighborType &b) {
                return a.id_ < b.id_ || (a.id_ == b.id_ && a.distance_ < b.distance_);
              });
    candidates.erase(std::unique(candidates.begin(),
                                 candidates.end(),
                                 [](const NeighborType &a, const NeighborType &b) -> bool {
                                   return a.id_ == b.id_;
                                 }),
                     candidates.end());
    // Re-sort by distance for pruning
    std::sort(candidates.begin(), candidates.end());
  }

  /// Sync wrapper: fetches nodes via get_node(), delegates to shared helpers.
  /// Uses striped lock for thread-safe read-modify-write of vertex_v's neighbor list.
  void connect_task(IDType vertex_v, IDType new_node_id, float alpha) {
    SpinLockGuard guard(node_locks_[vertex_v % kNumNodeLocks]);

    // Thread-local buffers: reused across calls to avoid per-call heap churn
    static thread_local std::vector<DataType> vertex_vec;
    static thread_local std::vector<IDType> current_nbrs_copy;
    static thread_local std::vector<NeighborType> candidates;
    static thread_local std::vector<IDType> pruned;
    vertex_vec.resize(dimension_);
    current_nbrs_copy.clear();
    {
      auto node = storage_->data().get_node(vertex_v);
      auto vec = node.vector();
      std::copy(vec.begin(), vec.end(), vertex_vec.begin());
      // Copy neighbor IDs into local vector before NodeRef destructs (avoid dangling span)
      auto nbrs = node.neighbors();
      current_nbrs_copy.assign(nbrs.begin(), nbrs.end());
    }

    auto get_vector = [this](IDType id) -> const DataType * {
      return storage_->data().get_node(id).vector().data();
    };

    // Build candidates using shared helper (uses copied neighbor list)
    build_connect_candidates(vertex_vec.data(),
                             vertex_v,
                             new_node_id,
                             std::span<const IDType>(current_nbrs_copy),
                             get_vector,
                             candidates);

    // Prune using shared helper
    robust_prune_impl(vertex_vec.data(), candidates, alpha, get_vector, pruned);
    if (new_node_id != kInvalidID) {
      ensure_neighbor_present(pruned, new_node_id);
    }

    // Write back updated neighbor list
    {
      auto node = storage_->data().get_node(vertex_v);
      node.set_neighbors(std::span<const IDType>(pruned));
    }
  }
};

}  // namespace alaya
