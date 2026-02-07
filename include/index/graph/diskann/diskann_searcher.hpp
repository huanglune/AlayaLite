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
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
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
  using Viewer = typename StorageType::Viewer;
  using Params = DiskANNSearchParams;
  using Result = SearchResult<IDType, DistanceType>;

  /// Distance function signature: (query, vector, dim) -> distance
  using DistanceFn = auto (*)(const DataType *, const DataType *, size_t) -> float;

  static constexpr IDType kInvalidID = static_cast<IDType>(-1);

 private:
  std::unique_ptr<BufferPoolType> buffer_pool_;
  std::unique_ptr<StorageType> storage_;

  uint32_t dimension_{0};
  uint32_t max_degree_{0};
  uint32_t capacity_{0};
  IDType medoid_id_{0};

  bool pq_enabled_{false};
  DistanceFn dist_fn_{simd::l2_sqr};     ///< Distance function (L2/IP, set at open time)
  std::vector<IDType> graph_;            ///< Flat array: capacity_ * max_degree_ neighbor IDs
  std::vector<uint32_t> graph_degrees_;  ///< Actual degree per node

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
  void open(std::string_view base_path, size_t cache_capacity = 4096) {
    buffer_pool_ = std::make_unique<BufferPoolType>(cache_capacity, kDataBlockSize);

    storage_ = std::make_unique<StorageType>(buffer_pool_.get());
    storage_->open(base_path, false);

    dimension_ = storage_->dimension();
    max_degree_ = storage_->max_degree();
    capacity_ = storage_->capacity();
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

    // Load graph topology into memory for fast neighbor lookup
    load_graph_topology();

    // Preload data blocks into buffer pool via sequential I/O.
    // This converts random search-time I/O into sequential startup I/O,
    // which is ~100x faster on HDD and ~10x on SSD.
    preload_data_blocks();

    // Enable PQ if the index was built with PQ
    pq_enabled_ = storage_->is_pq_enabled();

    LOG_INFO("DiskANNSearcher: Opened index, capacity={}, dim={}, R={}, medoid={}, pq={}",
             capacity_,
             dimension_,
             max_degree_,
             medoid_id_,
             pq_enabled_);
  }

  [[nodiscard]] auto is_open() const -> bool { return storage_ != nullptr && storage_->is_open(); }
  [[nodiscard]] auto is_pq_enabled() const -> bool { return pq_enabled_; }
  [[nodiscard]] auto num_points() const -> uint64_t {
    return storage_ ? storage_->num_active() : 0;
  }
  [[nodiscard]] auto dimension() const -> uint32_t { return dimension_; }
  [[nodiscard]] auto max_degree() const -> uint32_t { return max_degree_; }
  [[nodiscard]] auto storage() const -> const StorageType * { return storage_.get(); }
  [[nodiscard]] auto buffer_pool() const -> const BufferPoolType * { return buffer_pool_.get(); }

  /**
   * @brief Search for k nearest neighbors.
   *
   * Dispatches to PQ two-phase search or disk-only fallback.
   */
  auto search(const DataType *query, uint32_t topk, const Params &params = Params{}) -> Result {
    if (!is_open()) {
      throw std::runtime_error("DiskANNSearcher: Index not loaded");
    }
    if (pq_enabled_ && params.use_pq_rerank_) {
      return search_pq(query, topk, params);
    }
    return search_disk(query, topk, params);
  }

  /**
   * @brief Batch search for multiple queries.
   */
  auto batch_search(const DataType *queries,
                    uint32_t num_queries,
                    uint32_t topk,
                    const Params &params = Params{}) -> std::vector<Result> {
    std::vector<Result> batch_results(num_queries);

    auto search_thread = params.num_threads_;
    if (search_thread == 0) {
      search_thread = 1;
    }

#pragma omp parallel for schedule(dynamic) num_threads(search_thread)
    for (uint32_t i = 0; i < num_queries; ++i) {
      batch_results[i] = search(queries + i * dimension_, topk, params);
    }

    return batch_results;
  }

  // -------------------------------------------------------------------------
  // Graph mutation methods (for future insert/delete support)
  // -------------------------------------------------------------------------

  /**
   * @brief Update the in-memory graph topology for a node.
   *
   * Call this after inserting a node and writing its neighbors to disk.
   */
  void insert_to_graph(IDType node_id, std::span<const IDType> neighbors) {
    if (node_id >= capacity_) {
      return;
    }
    auto *dst = graph_.data() + static_cast<size_t>(node_id) * max_degree_;
    auto count = std::min(static_cast<uint32_t>(neighbors.size()), max_degree_);
    std::copy_n(neighbors.begin(), count, dst);
    std::fill(dst + count, dst + max_degree_, kInvalidID);
    graph_degrees_[node_id] = count;
  }

  /**
   * @brief Mark a node as removed from the in-memory graph.
   *
   * Sets degree to 0 so the node is never expanded during search.
   */
  void remove_from_graph(IDType node_id) {
    if (node_id >= capacity_) {
      return;
    }
    graph_degrees_[node_id] = 0;
  }

 private:
  // =========================================================================
  // Graph topology loading
  // =========================================================================

  /**
   * @brief Load all neighbor IDs from disk into the in-memory graph.
   *
   * After this, graph traversal requires zero disk I/O.
   * Memory cost: capacity * (max_degree * sizeof(IDType) + sizeof(uint32_t)).
   */
  void load_graph_topology() {
    graph_.resize(static_cast<size_t>(capacity_) * max_degree_, kInvalidID);
    graph_degrees_.resize(capacity_, 0);

    for (uint32_t node_id = 0; node_id < capacity_; ++node_id) {
      if (!storage_->is_valid(node_id)) {
        continue;
      }
      storage_->inspect_node(node_id, [&](const Viewer &viewer) -> void {
        const auto &nbrs = viewer.neighbors_view();
        auto degree = static_cast<uint32_t>(nbrs.size());
        graph_degrees_[node_id] = degree;
        auto *dst = graph_.data() + static_cast<size_t>(node_id) * max_degree_;
        std::copy_n(nbrs.begin(), degree, dst);
      });
    }
    LOG_INFO("DiskANNSearcher: Loaded graph topology for {} nodes", capacity_);
  }

  /**
   * @brief Sequentially preload all data blocks into the buffer pool.
   *
   * Sequential I/O at startup is much faster than random I/O during queries.
   * Only loads blocks that fit in the buffer pool capacity.
   */
  void preload_data_blocks() {
    auto &data_file = storage_->data();
    uint32_t total_blocks = data_file.num_blocks();
    uint32_t loaded = 0;

    for (uint32_t block_id = 0; block_id < total_blocks; ++block_id) {
      data_file.preload_block(block_id);
      ++loaded;
    }
    LOG_INFO("DiskANNSearcher: Preloaded {} data blocks into buffer pool", loaded);
  }

  // =========================================================================
  // Two-phase PQ search
  // =========================================================================

  /**
   * @brief Phase 1 + Phase 2: PQ navigation followed by exact reranking.
   *
   * Phase 1 (all in-memory):
   *   - Pre-compute ADC distance table for the query.
   *   - Traverse graph using in-memory neighbor IDs and PQ distances.
   *
   * Phase 2 (minimal disk I/O):
   *   - Read full vectors from disk for top rerank_count candidates.
   *   - Compute exact distances and return top-k.
   */
  auto search_pq(const DataType *query, uint32_t topk, const Params &params) -> Result {
    uint32_t ef = std::max(params.ef_search_, topk);
    ef = std::min(ef, capacity_);
    uint32_t rerank_count = std::min(ef, topk * params.pq_rerank_factor_);
    rerank_count = std::max(rerank_count, topk);

    // Pre-compute ADC distance table: M * K floats
    auto num_subspaces = storage_->pq().num_subspaces();
    auto num_centroids = storage_->pq().num_centroids();
    std::vector<float> adc_table(static_cast<size_t>(num_subspaces) * num_centroids);
    storage_->pq().compute_adc_table(query, adc_table.data());

    // Phase 1: PQ navigation (in-memory)
    CandidateList<DistanceType, IDType> candidates(capacity_, static_cast<int>(ef));

    // Seed with medoid
    auto medoid_dist = storage_->pq().compute_distance(adc_table.data(), medoid_id_);
    candidates.insert(medoid_id_, medoid_dist);
    candidates.vis_.set(medoid_id_);

    while (candidates.has_next()) {
      auto cur_id = candidates.pop();

      // Get neighbors from IN-MEMORY graph (zero disk I/O)
      auto [nbrs, degree] = get_neighbors(cur_id);

      for (uint32_t i = 0; i < degree; ++i) {
        auto neighbor = nbrs[i];
        if (neighbor == kInvalidID) {
          break;
        }
        if (candidates.vis_.get(neighbor)) {
          continue;
        }
        candidates.vis_.set(neighbor);

        // PQ distance: O(M) table lookups (in-memory, ~100x faster than exact L2)
        auto pq_dist = storage_->pq().compute_distance(adc_table.data(), neighbor);
        candidates.insert(neighbor, pq_dist);
      }
    }

    // Phase 2: Exact reranking (disk I/O only for top candidates)
    std::vector<NeighborType> rerank_pool;
    rerank_pool.reserve(rerank_count);
    for (uint32_t i = 0; i < std::min(static_cast<uint32_t>(candidates.size()), rerank_count);
         ++i) {
      rerank_pool.emplace_back(candidates.id(i), candidates.dist(i));
    }

    // Batch prefetch rerank candidates
    prefetch_node_blocks(rerank_pool);

    for (auto &candidate : rerank_pool) {
      auto exact_dist = compute_node_distance(query, candidate.id_);
      candidate.distance_ = exact_dist;
    }

    std::sort(rerank_pool.begin(), rerank_pool.end());

    // Build result
    auto actual_k = std::min(static_cast<size_t>(topk), rerank_pool.size());
    Result result(actual_k);
    for (size_t i = 0; i < actual_k; ++i) {
      result.ids_.emplace_back(rerank_pool[i].id_);
      result.distances_.emplace_back(rerank_pool[i].distance_);
    }
    return result;
  }

  // =========================================================================
  // Disk-only search with beam expansion and block-sorted I/O
  // =========================================================================

  /**
   * @brief Search using beam expansion + block-sorted distance computation.
   *
   * Instead of expanding one candidate at a time (random I/O per neighbor),
   * expands beam_width candidates together. All their unvisited neighbors are
   * collected, sorted by block_id, and processed sequentially. This yields:
   * - Block deduplication: nodes sharing a 4KB block require only one pread
   * - Sequential access: sorted block_ids enable OS readahead / SSD prefetch
   * - Reduced LRU pressure: same block processed consecutively stays pinned
   */
  auto search_disk(const DataType *query, uint32_t topk, const Params &params) -> Result {
    uint32_t ef = std::max(params.ef_search_, topk);
    ef = std::min(ef, capacity_);
    uint32_t beam_width = std::max(params.beam_width_, 1U);

    CandidateList<DistanceType, IDType> candidates(capacity_, static_cast<int>(ef));

    auto dist = compute_node_distance(query, medoid_id_);
    candidates.insert(medoid_id_, dist);
    candidates.vis_.set(medoid_id_);

    // Reusable buffers to avoid per-iteration heap allocation
    std::vector<IDType> beam;
    beam.reserve(beam_width);
    std::vector<std::pair<uint32_t, IDType>> neighbor_blocks;
    neighbor_blocks.reserve(static_cast<size_t>(beam_width) * max_degree_);

    while (candidates.has_next()) {
      // --- Pop beam_width candidates at once ---
      beam.clear();
      for (uint32_t b = 0; b < beam_width && candidates.has_next(); ++b) {
        beam.push_back(candidates.pop());
      }

      // --- Collect all unvisited neighbors from the beam ---
      neighbor_blocks.clear();
      for (auto cur_id : beam) {
        auto [nbrs, degree] = get_neighbors(cur_id);
        for (uint32_t i = 0; i < degree; ++i) {
          auto nbr = nbrs[i];
          if (nbr == kInvalidID) {
            break;
          }
          if (candidates.vis_.get(nbr)) {
            continue;
          }
          candidates.vis_.set(nbr);
          uint32_t block_id = storage_->data().block_id_of(static_cast<uint32_t>(nbr));
          neighbor_blocks.emplace_back(block_id, nbr);
        }
      }

      if (neighbor_blocks.empty()) {
        continue;
      }

      // --- Sort by block_id for sequential disk access ---
      std::sort(neighbor_blocks.begin(), neighbor_blocks.end());

      // --- Batch prefetch via io_uring ---
      prefetch_neighbor_blocks(neighbor_blocks);

      // --- Block-local pinning: one pin/unpin per unique block ---
      uint32_t prev_blk = UINT32_MAX;
      typename BufferPoolType::PageHandle block_handle;
      bool handle_valid = false;

      for (auto [blk, nbr] : neighbor_blocks) {
        if (blk != prev_blk) {
          block_handle = buffer_pool_->get(static_cast<IDType>(blk));
          prev_blk = blk;
          handle_valid = !block_handle.empty();
        }

        if (handle_valid) {
          auto vec = storage_->data().get_vector_in_block(block_handle, nbr);
          auto nbr_dist = dist_fn_(query, vec.data(), dimension_);
          candidates.insert(nbr, nbr_dist);
        } else {
          auto nbr_dist = compute_node_distance(query, nbr);
          candidates.insert(nbr, nbr_dist);
        }
      }
    }

    return candidates.to_search_result(topk);
  }

  // =========================================================================
  // Helper methods
  // =========================================================================

  /**
   * @brief Get neighbor IDs from the in-memory graph.
   */
  [[nodiscard]] auto get_neighbors(IDType node_id) const -> std::pair<const IDType *, uint32_t> {
    const IDType *neighbors = graph_.data() + static_cast<size_t>(node_id) * max_degree_;
    uint32_t degree = graph_degrees_[node_id];
    return {neighbors, degree};
  }

  /**
   * @brief Read a node's full vector from disk and compute exact distance.
   */
  auto compute_node_distance(const DataType *query, IDType node_id) -> DistanceType {
    DistanceType dist = 0;
    storage_->inspect_node(node_id, [&](const Viewer &viewer) -> void {
      auto vec = viewer.vector_view();
      dist = dist_fn_(query, vec.data(), dimension_);
    });
    return dist;
  }

  /**
   * @brief Prefetch blocks for rerank candidates (search_pq Phase 2).
   *
   * Computes block_id for each candidate, sorts, and submits batch I/O.
   */
  void prefetch_node_blocks(std::span<const NeighborType> candidates) {
    std::vector<uint32_t> block_ids;
    block_ids.reserve(candidates.size());
    for (const auto &c : candidates) {
      block_ids.push_back(storage_->data().block_id_of(static_cast<uint32_t>(c.id_)));
    }
    std::ranges::sort(block_ids);
    storage_->data().prefetch_blocks(block_ids);
  }

  /**
   * @brief Prefetch blocks for neighbor expansion (search_disk).
   *
   * neighbor_blocks is already sorted by block_id (first element of pair).
   */
  void prefetch_neighbor_blocks(std::span<const std::pair<uint32_t, IDType>> neighbor_blocks) {
    std::vector<uint32_t> block_ids;
    block_ids.reserve(neighbor_blocks.size());
    for (auto [blk, nbr] : neighbor_blocks) {
      block_ids.push_back(blk);
    }
    storage_->data().prefetch_blocks(block_ids);
  }
};

}  // namespace alaya
