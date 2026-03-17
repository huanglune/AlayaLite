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
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "diskann_params.hpp"
#include "index/neighbor.hpp"
#include "simd/distance_ip.hpp"
#include "simd/distance_l2.hpp"
#include "storage/buffer/replacer/clock.hpp"
#include "storage/diskann/diskann_storage.hpp"
#include "utils/candidate_list.hpp"
#include "utils/log.hpp"
#include "utils/macros.hpp"
#include "utils/types.hpp"
#include "visited_list.hpp"

namespace alaya {

/**
 * @brief High-performance DiskANN searcher with two-phase PQ search.
 *
 * When PQ is enabled, uses a two-phase search strategy:
 * 1. PQ Navigation (in-memory): Traverse graph using PQ approximate distances.
 *    Graph topology and PQ codes are held in memory (via mmap).
 * 2. Exact Reranking (disk I/O): Read full vectors from disk only for top
 *    candidates and compute exact L2 distances.
 *
 * This reduces disk I/O from O(ef * R) reads to O(rerank_count) reads per query.
 *
 * When PQ is disabled, falls back to disk-only search using in-memory graph
 * topology to avoid redundant block reads for neighbor lists.
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
    // Buffers for PQ Search
    std::vector<float> adc_table_;  // M * K
    std::vector<NeighborType> rerank_candidates_;

    // Helper buffer to hold neighbors read from disk temporarily
    std::vector<IDType> neighbor_buffer_;

    explicit SearchContext(uint32_t capacity,
                           uint32_t max_degree,
                           uint32_t beam_width,
                           uint32_t num_subspaces = 0,
                           uint32_t num_centroids = 0)
        : candidates_(capacity, 200) {
      beam_queue_.reserve(beam_width * 2);
      next_beam_queue_.reserve(beam_width * 2);
      next_beam_io_.reserve(beam_width * max_degree);
      neighbor_buffer_.reserve(max_degree);
      if (num_subspaces > 0) {
        adc_table_.resize(static_cast<size_t>(num_subspaces) * num_centroids);
      }
      rerank_candidates_.reserve(200);
      visited_.reset(capacity);
    }
    void clear() {
      candidates_.clear();
      beam_queue_.clear();
      next_beam_queue_.clear();
      next_beam_io_.clear();
      rerank_candidates_.clear();
      neighbor_buffer_.clear();
    }
  };

 private:
  std::unique_ptr<BufferPoolType> buffer_pool_;
  std::unique_ptr<StorageType> storage_;

  uint32_t dimension_{0};
  uint32_t max_degree_{0};
  uint32_t num_points_{0};
  uint32_t capacity_{0};
  IDType medoid_id_{0};

  bool pq_enabled_{false};
  DistanceFn dist_fn_{simd::l2_sqr};  ///< Distance function (L2/IP, set at open time)

 public:
  DiskANNSearcher() = default;
  ALAYA_NON_COPYABLE_BUT_MOVABLE(DiskANNSearcher);
  ~DiskANNSearcher() = default;

  /**
   * @brief Open a disk index for searching.
   *
   * Loads graph topology into memory and, if PQ was built, enables
   * two-phase PQ search. Buffer pool caches disk pages for reranking.
   *
   * @param base_path Base path for index files (without extension)
   * @param cache_capacity Number of 4KB pages in the buffer pool
   */
  void open(std::string_view base_path, size_t cache_capacity = 4096, bool writable = false) {
    LOG_INFO("DiskANNSearcher: Opening index at {}", base_path);

    buffer_pool_ = std::make_unique<BufferPoolType>(cache_capacity, kDataBlockSize);
    storage_ = std::make_unique<StorageType>(buffer_pool_.get());
    storage_->open(base_path, writable);

    dimension_ = storage_->dimension();
    max_degree_ = storage_->max_degree();

    num_points_ = storage_->num_points();
    capacity_ = storage_->capacity();
    medoid_id_ = static_cast<IDType>(storage_->entry_point());
    pq_enabled_ = storage_->is_pq_enabled();

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

    size_t preload_blocks =
        std::min(size_t{100}, static_cast<size_t>(storage_->data().num_blocks()));
    for (size_t i = 0; i < preload_blocks; ++i) {
      storage_->data().preload_block(i);
    }

    LOG_INFO(
        "DiskANNSearcher: Opened index, capacity={}, num_point={}, dim={}, R={}, medoid={}, pq={}",
        capacity_,
        num_points_,
        dimension_,
        max_degree_,
        medoid_id_,
        pq_enabled_);
  }

  /**
   * @brief Search for k nearest neighbors.
   *
   * Dispatches to PQ two-phase search or disk-only fallback.
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

    if (pq_enabled_ && params.use_pq_rerank_) {
      return search_pq(ctx, query, topk, params);
    }
    return search_disk(ctx, query, topk, params);
  }

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
  [[nodiscard]] auto num_points() const -> uint32_t { return num_points_; }
  [[nodiscard]] auto dimension() const -> uint32_t { return dimension_; }
  [[nodiscard]] auto capacity() const -> uint32_t { return capacity_; }
  [[nodiscard]] auto buffer_pool() -> BufferPoolType & { return *buffer_pool_; }

  // =========================================================================
  // Mutation (for incremental insert support)
  // =========================================================================

  void reserve_capacity(uint32_t /*new_capacity*/) {
    // TODO(diskann): implement capacity growth for incremental inserts
  }

  void insert(const DataType * /*vector*/,
              IDType /*external_id*/,
              const DiskANNInsertParams & /*params*/ = DiskANNInsertParams{}) {
    // TODO(diskann): implement single-vector insertion
  }

  void flush() {
    if (storage_ && storage_->is_open()) {
      storage_->save_meta();
    }
  }

 private:
  auto get_search_context(const Params &params) -> SearchContext & {
    static thread_local SearchContext ctx(num_points_,
                                          max_degree_,
                                          params.beam_width_ > 0 ? params.beam_width_ : 128,
                                          pq_enabled_ ? storage_->pq().num_subspaces() : 0,
                                          pq_enabled_ ? storage_->pq().num_centroids() : 0);
    return ctx;
  }

  // =========================================================================
  // Helper: Read Neighbors using NodeRef (Zero-Copy)
  // =========================================================================
  auto get_neighbors_from_disk(IDType node_id, std::vector<IDType> &out_buffer) -> uint32_t {
    auto node = storage_->data().get_node(node_id);
    auto neighbors_span = node.neighbors();

    // 3. Copy to output buffer (Required because span invalidates when node dies)
    out_buffer.clear();
    for (auto n : neighbors_span) {
      if (n == kInvalidID) {
        break;
      }
      out_buffer.push_back(n);
    }

    // 4. NodeRef destructor is called here -> Page is unpinned
    return static_cast<uint32_t>(out_buffer.size());
  }

  // =========================================================================
  // Helper: Compute Exact Distance using NodeRef
  // =========================================================================
  auto compute_exact_distance(const DataType *query, IDType node_id) -> float {
    auto node = storage_->data().get_node(node_id);
    auto vec_span = node.vector();
    return dist_fn_(query, vec_span.data(), dimension_);
  }

  auto search_disk(SearchContext &ctx, const DataType *query, uint32_t topk, const Params &params)
      -> Result {
    auto &candidates = ctx.candidates_;
    auto &visited = ctx.visited_;
    auto &beam_queue = ctx.beam_queue_;
    auto &next_beam_io = ctx.next_beam_io_;
    auto &next_beam_queue = ctx.next_beam_queue_;

    // Init Medoid
    // No explicit prefetch needed for single point, get_node handles it
    float start_dist = compute_exact_distance(query, medoid_id_);
    candidates.insert(medoid_id_, start_dist);
    visited.mark(medoid_id_);
    beam_queue.push_back(medoid_id_);

    uint32_t beam_width = std::max(1U, params.beam_width_);
    while (!beam_queue.empty()) {
      next_beam_queue.clear();
      next_beam_io.clear();

      // A. Expand Beam (Read Neighbors)
      // Since beam nodes were just computed, their pages are HOT in BufferPool.
      // get_node() will likely return immediately without IO.
      for (IDType node_id : beam_queue) {
        get_neighbors_from_disk(node_id, ctx.neighbor_buffer_);

        for (IDType nbr : ctx.neighbor_buffer_) {
          if (visited.is_visited(nbr)) {
            continue;
          }
          visited.mark(nbr);

          // Collect block ID for batch prefetch
          next_beam_io.push_back(storage_->data().block_id_of(nbr));
          next_beam_queue.push_back(nbr);
        }
      }

      if (next_beam_queue.empty()) {
        break;
      }

      // B. IO Deduplication & Prefetching
      std::sort(next_beam_io.begin(), next_beam_io.end());
      next_beam_io.erase(std::unique(next_beam_io.begin(), next_beam_io.end()), next_beam_io.end());

      // Batch submit IO requests to OS/SSD
      storage_->data().prefetch_blocks(next_beam_io);

      // C. Compute Distances
      // Iterate over nodes. Their blocks have been prefetched.
      // get_node() will find them in BufferPool.
      for (auto nbr : next_beam_queue) {
        float dist = compute_exact_distance(query, nbr);
        candidates.insert(nbr, dist);
      }

      // D. Form next beam
      beam_queue.clear();
      for (uint32_t i = 0; i < beam_width && candidates.has_next(); ++i) {
        beam_queue.push_back(candidates.pop());
      }
    }

    return candidates.to_search_result(topk);
  }

  auto search_pq(SearchContext & /*ctx*/,
                 const DataType * /*query*/,
                 uint32_t /*topk*/,
                 const Params & /*params*/) -> Result {
    // TODO(diskann): implement PQ two-phase search
    return Result(0);
  }
};

}  // namespace alaya
