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
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "disk_layout.hpp"
#include "diskann_params.hpp"
#include "index/neighbor.hpp"
#include "simd/distance_l2.hpp"
#include "space/quant/pq.hpp"
#include "storage/buffer/buffer_pool.hpp"
#include "storage/io/direct_file_io.hpp"
#include "utils/bitset.hpp"
#include "utils/log.hpp"
#include "utils/memory.hpp"

namespace alaya {

/**
 * @brief DiskANN searcher for disk-based approximate nearest neighbor search.
 *
 * This class provides efficient disk-based search using a two-phase approach:
 * 1. **PQ Navigation Phase**: Use in-memory PQ codes for fast approximate distance
 *    computation during graph traversal (beam search)
 * 2. **Disk Reranking Phase**: Read full vectors from disk for accurate distance
 *    computation on the top candidates
 *
 * When PQ is enabled:
 * - PQ codebook and PQ codes are loaded into memory at startup
 * - Graph topology (neighbor IDs) is loaded into memory
 * - Only reranking candidates require disk I/O
 *
 * When PQ is disabled (fallback mode):
 * - Every distance computation requires reading the full vector from disk
 *
 * @tparam DataType The data type of vector elements (default: float)
 * @tparam IDType The data type for node IDs (default: uint32_t)
 */
template <typename DataType = float, typename IDType = uint32_t>
class DiskANNSearcher {
 public:
  using DistanceType = float;
  using NeighborType = Neighbor<IDType, DistanceType>;
  using NodeType = DiskNode<DataType, IDType>;
  using NodeAccessor = typename NodeType::Accessor;

 private:
  std::unique_ptr<DirectFileIO> reader_;          ///< File reader for disk access
  DiskIndexHeader header_;                        ///< Index header metadata
  DiskNodeBuffer<DataType, IDType> node_buffer_;  ///< Buffer for reading nodes
  Bitset visited_;                                ///< Visited node tracking

  // Cached parameters
  uint32_t dimension_{0};
  uint32_t max_degree_{0};
  uint64_t num_points_{0};
  IDType medoid_id_{0};
  size_t node_size_{0};

  // PQ-related members for in-memory navigation
  bool pq_enabled_{false};
  PQQuantizer<DataType> pq_quantizer_;   ///< PQ codebook
  std::vector<uint8_t> pq_codes_;        ///< In-memory PQ codes: N × M bytes
  std::vector<IDType> graph_;            ///< In-memory graph: N × R neighbor IDs
  std::vector<uint32_t> graph_degrees_;  ///< Actual degree of each node

  // Buffer pool for caching disk nodes (optional)
  std::unique_ptr<BufferPool<IDType>> buffer_pool_;  ///< LRU cache for disk nodes
  bool caching_enabled_{false};                      ///< Whether caching is active

 public:
  DiskANNSearcher() = default;

  DiskANNSearcher(const DiskANNSearcher &) = delete;
  auto operator=(const DiskANNSearcher &) -> DiskANNSearcher & = delete;
  DiskANNSearcher(DiskANNSearcher &&) = default;
  auto operator=(DiskANNSearcher &&) -> DiskANNSearcher & = default;
  ~DiskANNSearcher() = default;

  /**
   * @brief Open a disk index file for searching.
   *
   * This loads the index header and, if PQ is enabled, loads:
   * - PQ codebook from disk
   * - PQ codes from disk (for approximate distance computation)
   * - Graph topology from disk (neighbor IDs for navigation)
   *
   * @param index_path Path to the index file
   */
  auto open(std::string_view index_path) -> void {
    // Open file with Direct IO if possible (DirectFileIO handles fallback internally)
    reader_ = std::make_unique<DirectFileIO>();
    if (!reader_->open(index_path, DirectFileIO::Mode::kRead)) {
      throw std::runtime_error("Failed to open index file: " + std::string(index_path));
    }

    // Read header
    DiskNodeBuffer<DataType, IDType> header_buffer;
    header_buffer.allocate(1, 1, 1);  // Minimal allocation for header read

    auto bytes = reader_->read(header_buffer.data(), kDiskSectorSize, 0);
    if (bytes != static_cast<ssize_t>(kDiskSectorSize)) {
      throw std::runtime_error("Failed to read index header");
    }

    std::memcpy(&header_, header_buffer.data(), sizeof(DiskIndexHeader));
    if (!header_.is_valid()) {
      throw std::runtime_error("Invalid DiskANN index header");
    }

    // Cache parameters
    dimension_ = header_.meta_.dimension_;
    max_degree_ = header_.meta_.max_degree_;
    num_points_ = header_.meta_.num_points_;
    medoid_id_ = static_cast<IDType>(header_.meta_.medoid_id_);
    node_size_ = header_.meta_.node_sector_size_;

    // Allocate node buffer for disk reads
    node_buffer_.allocate(dimension_, max_degree_, 1);

    // Initialize visited bitset
    visited_.resize(num_points_);

    // Check if PQ is enabled and load PQ data
    pq_enabled_ = header_.is_pq_enabled();
    if (pq_enabled_) {
      load_pq_data();
      load_graph_topology();
      LOG_INFO("DiskANNSearcher: Opened PQ-enabled index with {} points, dim={}, R={}, M={}",
               num_points_,
               dimension_,
               max_degree_,
               header_.meta_.pq_num_subspaces_);
    } else {
      LOG_INFO("DiskANNSearcher: Opened index with {} points, dim={}, R={}, medoid={} (no PQ)",
               num_points_,
               dimension_,
               max_degree_,
               medoid_id_);
    }
  }

  /**
   * @brief Check if the searcher is ready.
   */
  [[nodiscard]] auto is_open() const -> bool { return reader_ != nullptr && reader_->is_open(); }

  /**
   * @brief Check if PQ is enabled for this index.
   */
  [[nodiscard]] auto is_pq_enabled() const -> bool { return pq_enabled_; }

  /**
   * @brief Get the number of points in the index.
   */
  [[nodiscard]] auto num_points() const -> uint64_t { return num_points_; }

  /**
   * @brief Get the vector dimension.
   */
  [[nodiscard]] auto dimension() const -> uint32_t { return dimension_; }

  /**
   * @brief Get the maximum out-degree.
   */
  [[nodiscard]] auto max_degree() const -> uint32_t { return max_degree_; }

  /**
   * @brief Get the index header.
   */
  [[nodiscard]] auto header() const -> const DiskIndexHeader & { return header_; }

  /**
   * @brief Enable node caching with specified capacity.
   *
   * Creates a buffer pool to cache recently accessed disk nodes,
   * reducing disk I/O for repeated accesses.
   *
   * @param cache_capacity Number of nodes to cache (0 = disable)
   */
  void enable_caching(size_t cache_capacity) {
    if (cache_capacity > 0 && node_size_ > 0) {
      buffer_pool_ = std::make_unique<BufferPool<IDType>>(cache_capacity, node_size_);
      caching_enabled_ = true;
      LOG_INFO("DiskANNSearcher: Enabled caching with {} frames ({} MB)",
               cache_capacity,
               (cache_capacity * node_size_) / (1024 * 1024));
    } else {
      disable_caching();
    }
  }

  /**
   * @brief Disable caching and release buffer pool memory.
   */
  void disable_caching() {
    buffer_pool_.reset();
    caching_enabled_ = false;
  }

  /**
   * @brief Check if caching is enabled.
   */
  [[nodiscard]] auto is_caching_enabled() const -> bool { return caching_enabled_; }

  /**
   * @brief Get cache statistics.
   *
   * Returns statistics about cache hits, misses, and evictions.
   * Returns empty stats if caching is disabled.
   */
  [[nodiscard]] auto cache_stats() const -> BufferPoolStats {
    if (buffer_pool_) {
      return buffer_pool_->stats();
    }
    return BufferPoolStats{};
  }

  /**
   * @brief Reset cache statistics counters.
   */
  void reset_cache_stats() {
    if (buffer_pool_) {
      buffer_pool_->reset_stats();
    }
  }

  /**
   * @brief Clear all cached data.
   */
  void clear_cache() {
    if (buffer_pool_) {
      buffer_pool_->clear();
    }
  }

  /**
   * @brief Search for k nearest neighbors of a query vector.
   *
   * When PQ is enabled, uses two-phase search:
   * 1. PQ navigation to find candidates
   * 2. Disk reranking to get accurate distances
   *
   * @param query Pointer to the query vector
   * @param topk Number of nearest neighbors to return
   * @param results Output array for result IDs
   * @param ef Search list size (0 = use default)
   */
  auto search(const DataType *query, uint32_t topk, IDType *results, uint32_t ef = 0) -> void {
    if (!is_open()) {
      throw std::runtime_error("DiskANNSearcher: Index not loaded");
    }

    if (ef == 0) {
      ef = std::max(topk, 50U);
    }

    std::vector<NeighborType> neighbors;
    if (pq_enabled_) {
      neighbors = beam_search_pq(query, ef, topk);
    } else {
      neighbors = beam_search_disk(query, ef);
    }

    // Extract top-k results
    size_t result_count = std::min(static_cast<size_t>(topk), neighbors.size());
    for (size_t i = 0; i < result_count; ++i) {
      results[i] = neighbors[i].id_;
    }

    // Fill remaining with invalid IDs
    for (size_t i = result_count; i < topk; ++i) {
      results[i] = static_cast<IDType>(-1);
    }
  }

  /**
   * @brief Search for k nearest neighbors with search params.
   */
  auto search(const DataType *query,
              uint32_t topk,
              IDType *results,
              const DiskANNSearchParams &params) -> void {
    search(query, topk, results, params.ef_search_);
  }

  /**
   * @brief Search for k nearest neighbors with distances.
   */
  auto search_with_distance(const DataType *query,
                            uint32_t topk,
                            IDType *results,
                            DistanceType *distances,
                            uint32_t ef = 0) -> void {
    if (!is_open()) {
      throw std::runtime_error("DiskANNSearcher: Index not loaded");
    }

    if (ef == 0) {
      ef = std::max(topk, 50U);
    }

    std::vector<NeighborType> neighbors;
    if (pq_enabled_) {
      neighbors = beam_search_pq(query, ef, topk);
    } else {
      neighbors = beam_search_disk(query, ef);
    }

    size_t result_count = std::min(static_cast<size_t>(topk), neighbors.size());
    for (size_t i = 0; i < result_count; ++i) {
      results[i] = neighbors[i].id_;
      distances[i] = neighbors[i].distance_;
    }

    for (size_t i = result_count; i < topk; ++i) {
      results[i] = static_cast<IDType>(-1);
      distances[i] = std::numeric_limits<DistanceType>::max();
    }
  }

  /**
   * @brief Search for k nearest neighbors with distances using search params.
   */
  auto search_with_distance(const DataType *query,
                            uint32_t topk,
                            IDType *results,
                            DistanceType *distances,
                            const DiskANNSearchParams &params) -> void {
    search_with_distance(query, topk, results, distances, params.ef_search_);
  }

  /**
   * @brief Batch search for multiple queries.
   */
  auto batch_search(const DataType *queries,
                    uint32_t num_queries,
                    uint32_t topk,
                    IDType *results,
                    uint32_t ef = 0) -> void {
    for (uint32_t i = 0; i < num_queries; ++i) {
      search(queries + i * dimension_, topk, results + i * topk, ef);
    }
  }

  /**
   * @brief Batch search for multiple queries with search params.
   */
  auto batch_search(const DataType *queries,
                    uint32_t num_queries,
                    uint32_t topk,
                    IDType *results,
                    const DiskANNSearchParams &params) -> void {
    batch_search(queries, num_queries, topk, results, params.ef_search_);
  }

 private:
  /**
   * @brief Load PQ codebook and codes from disk into memory.
   */
  void load_pq_data() {
    uint32_t num_subspaces = header_.meta_.pq_num_subspaces_;
    uint32_t subspace_dim = dimension_ / num_subspaces;

    // Initialize quantizer with correct dimensions
    pq_quantizer_ = PQQuantizer<DataType>(dimension_, num_subspaces);

    // Load codebook from disk using aligned buffer for Direct IO
    size_t codebook_size = header_.meta_.pq_codebook_size_;
    std::vector<uint8_t, AlignedAlloc<uint8_t, kAlign4K>> codebook_buffer(codebook_size);

    auto bytes =
        reader_->read(codebook_buffer.data(), codebook_size, header_.meta_.pq_codebook_offset_);
    if (bytes != static_cast<ssize_t>(codebook_size)) {
      throw std::runtime_error("Failed to read PQ codebook");
    }

    // Copy codebook data to quantizer
    // Note: We need to set the codebook directly. For now, we'll use a stream-based load.
    // The codebook in the file should be in the same format as PQQuantizer expects.
    std::string temp_path =
        "/tmp/pq_codebook_load_" + std::to_string(reinterpret_cast<uintptr_t>(this)) + ".tmp";
    {
      std::ofstream temp_writer(temp_path, std::ios::binary);
      // Write dimension info that PQQuantizer::load expects
      temp_writer.write(reinterpret_cast<const char *>(&dimension_), sizeof(dimension_));
      temp_writer.write(reinterpret_cast<const char *>(&num_subspaces), sizeof(num_subspaces));
      temp_writer.write(reinterpret_cast<const char *>(&subspace_dim), sizeof(subspace_dim));
      // Write codebook data
      size_t raw_codebook_size =
          static_cast<size_t>(num_subspaces) * 256 * subspace_dim * sizeof(DataType);
      temp_writer.write(reinterpret_cast<const char *>(codebook_buffer.data()), raw_codebook_size);
    }
    {
      std::ifstream temp_reader(temp_path, std::ios::binary);
      pq_quantizer_.load(temp_reader);
    }
    std::remove(temp_path.c_str());

    // Load PQ codes from disk using aligned buffer
    size_t codes_size = header_.meta_.pq_codes_size_;
    size_t raw_codes_size = num_points_ * num_subspaces;
    pq_codes_.resize(raw_codes_size);

    // Read aligned PQ codes section
    std::vector<uint8_t, AlignedAlloc<uint8_t, kAlign4K>> codes_buffer(codes_size);
    bytes = reader_->read(codes_buffer.data(), codes_size, header_.meta_.pq_codes_offset_);
    if (bytes != static_cast<ssize_t>(codes_size)) {
      throw std::runtime_error("Failed to read PQ codes");
    }
    std::memcpy(pq_codes_.data(), codes_buffer.data(), raw_codes_size);

    LOG_INFO("DiskANNSearcher: Loaded PQ codebook ({} bytes) and codes ({} bytes)",
             codebook_size,
             raw_codes_size);
  }

  /**
   * @brief Load graph topology (neighbor IDs) from disk into memory.
   */
  void load_graph_topology() {
    // Allocate memory for graph
    graph_.resize(num_points_ * max_degree_);
    graph_degrees_.resize(num_points_);

    // Read each node's neighbor list from disk
    for (uint64_t i = 0; i < num_points_; ++i) {
      auto accessor = read_node(static_cast<IDType>(i));
      graph_degrees_[i] = accessor.num_neighbors();

      const IDType *neighbors = accessor.neighbor_ids();
      IDType *graph_row = graph_.data() + i * max_degree_;
      std::memcpy(graph_row, neighbors, max_degree_ * sizeof(IDType));
    }

    LOG_INFO("DiskANNSearcher: Loaded graph topology ({} nodes × {} max degree)",
             num_points_,
             max_degree_);
  }

  /**
   * @brief Get in-memory neighbor IDs for a node.
   */
  [[nodiscard]] auto get_neighbors(IDType node_id) const -> std::pair<const IDType *, uint32_t> {
    const IDType *neighbors = graph_.data() + static_cast<size_t>(node_id) * max_degree_;
    uint32_t degree = graph_degrees_[node_id];
    return {neighbors, degree};
  }

  /**
   * @brief Get PQ codes for a node.
   */
  [[nodiscard]] auto get_pq_code(IDType node_id) const -> const uint8_t * {
    uint32_t num_subspaces = header_.meta_.pq_num_subspaces_;
    return pq_codes_.data() + static_cast<size_t>(node_id) * num_subspaces;
  }

  /**
   * @brief Read a node from disk (with optional caching).
   *
   * If caching is enabled, checks the buffer pool first. On cache miss,
   * reads from disk and caches the result for future accesses.
   */
  auto read_node(IDType node_id) -> NodeAccessor {
    uint64_t offset = header_.get_node_offset(node_id);

    if (caching_enabled_ && buffer_pool_) {
      // Use buffer pool with caching
      const uint8_t *cached_data =
          buffer_pool_->get_or_read(node_id, *reader_, offset, node_buffer_.data());

      if (cached_data != nullptr) {
        // Return accessor pointing to cached data (or node_buffer_ if just read)
        return NodeAccessor(const_cast<uint8_t *>(cached_data), dimension_, max_degree_);
      }
      // Fall through to direct read on error
    }

    // Direct read without caching
    auto bytes = reader_->read(node_buffer_.data(), node_size_, offset);

    if (bytes != static_cast<ssize_t>(node_size_)) {
      LOG_ERROR("Failed to read node {}: got {} bytes, expected {}", node_id, bytes, node_size_);
      throw std::runtime_error("Failed to read node from disk");
    }

    return node_buffer_.get_node(0);
  }

  /**
   * @brief Compute exact L2 distance between query and a vector.
   */
  auto compute_distance(const DataType *query, const DataType *vec) -> DistanceType {
    return simd::l2_sqr(query, vec, dimension_);
  }

  /**
   * @brief Two-phase beam search: PQ navigation + disk reranking.
   *
   * Phase 1: Use in-memory PQ codes and graph for fast navigation
   * Phase 2: Read full vectors from disk for top-k reranking
   *
   * @param query Query vector
   * @param ef Search list size for navigation
   * @param topk Number of candidates to rerank
   * @return Vector of nearest neighbors with accurate distances
   */
  auto beam_search_pq(const DataType *query, uint32_t ef, uint32_t topk)
      -> std::vector<NeighborType> {
    // Reset visited bitset
    visited_.reset();

    // Precompute ADC table for query
    uint32_t num_subspaces = header_.meta_.pq_num_subspaces_;
    std::vector<float> adc_table(num_subspaces * 256);
    pq_quantizer_.compute_adc_table(query, adc_table.data());

    // Result pool for PQ navigation
    std::vector<NeighborType> pool;
    pool.reserve(ef + max_degree_);

    // Priority queue for candidates (min-heap by PQ distance)
    auto cmp = [](const NeighborType &a, const NeighborType &b) -> bool {
      return a.distance_ > b.distance_;  // Min-heap
    };
    std::priority_queue<NeighborType, std::vector<NeighborType>, decltype(cmp)> candidates(cmp);

    // Start from medoid
    float medoid_pq_dist =
        pq_quantizer_.compute_distance_with_table(adc_table.data(), get_pq_code(medoid_id_));
    candidates.emplace(medoid_id_, medoid_pq_dist, false);
    pool.emplace_back(medoid_id_, medoid_pq_dist, false);
    visited_.set(medoid_id_);

    // Track threshold for pruning
    DistanceType threshold = std::numeric_limits<DistanceType>::max();

    // Phase 1: PQ-based navigation (no disk I/O for distances)
    while (!candidates.empty()) {
      auto cur = candidates.top();
      candidates.pop();

      // Early termination
      if (pool.size() >= ef && cur.distance_ > threshold) {
        break;
      }

      // Get neighbors from in-memory graph
      auto [neighbor_ids, num_neighbors] = get_neighbors(cur.id_);

      for (uint32_t i = 0; i < num_neighbors; ++i) {
        auto neighbor = neighbor_ids[i];
        if (neighbor == static_cast<IDType>(-1)) {
          break;
        }
        if (visited_.get(neighbor)) {
          continue;
        }
        visited_.set(neighbor);

        // Compute PQ distance (fast, in-memory)
        float pq_dist =
            pq_quantizer_.compute_distance_with_table(adc_table.data(), get_pq_code(neighbor));

        // Add to pool if promising
        if (pool.size() < ef || pq_dist < threshold) {
          candidates.emplace(neighbor, pq_dist, false);
          pool.emplace_back(neighbor, pq_dist, false);

          // Update threshold
          if (pool.size() > ef) {
            std::sort(pool.begin(), pool.end());
            pool.resize(ef);
            threshold = pool.back().distance_;
          } else if (pool.size() == ef) {
            std::sort(pool.begin(), pool.end());
            threshold = pool.back().distance_;
          }
        }
      }
    }

    // Sort pool by PQ distance
    std::sort(pool.begin(), pool.end());

    // Phase 2: Disk reranking for top candidates
    uint32_t rerank_count = std::min(static_cast<uint32_t>(pool.size()),
                                     std::max(topk * 2, ef));  // Rerank more than topk
    std::vector<NeighborType> reranked;
    reranked.reserve(rerank_count);

    for (uint32_t i = 0; i < rerank_count; ++i) {
      IDType node_id = pool[i].id_;

      // Read full vector from disk
      auto accessor = read_node(node_id);
      float exact_dist = compute_distance(query, accessor.vector_data());

      reranked.emplace_back(node_id, exact_dist, true);
    }

    // Sort by exact distance
    std::sort(reranked.begin(), reranked.end());

    return reranked;
  }

  /**
   * @brief Fallback beam search using disk I/O for all distance computations.
   *
   * Used when PQ is not enabled.
   */
  auto beam_search_disk(const DataType *query, uint32_t ef) -> std::vector<NeighborType> {
    // Reset visited bitset
    visited_.reset();

    std::vector<NeighborType> pool;
    pool.reserve(ef + max_degree_);

    auto cmp = [](const NeighborType &a, const NeighborType &b) -> bool {
      return a.distance_ > b.distance_;  // Min-heap
    };
    std::priority_queue<NeighborType, std::vector<NeighborType>, decltype(cmp)> candidates(cmp);

    // Start from medoid
    auto accessor = read_node(medoid_id_);
    auto dist = compute_distance(query, accessor.vector_data());
    candidates.emplace(medoid_id_, dist, false);
    pool.emplace_back(medoid_id_, dist, false);
    visited_.set(medoid_id_);

    DistanceType threshold = std::numeric_limits<DistanceType>::max();

    while (!candidates.empty()) {
      auto cur = candidates.top();
      candidates.pop();

      if (pool.size() >= ef && cur.distance_ > threshold) {
        break;
      }

      cur.flag_ = true;
      accessor = read_node(cur.id_);

      uint32_t num_neighbors = accessor.num_neighbors();
      const auto *neighbor_ids = accessor.neighbor_ids();

      for (uint32_t i = 0; i < num_neighbors; ++i) {
        auto neighbor = neighbor_ids[i];
        if (neighbor == static_cast<IDType>(-1)) {
          break;
        }
        if (visited_.get(neighbor)) {
          continue;
        }
        visited_.set(neighbor);

        auto neighbor_accessor = read_node(neighbor);
        auto neighbor_dist = compute_distance(query, neighbor_accessor.vector_data());

        if (pool.size() < ef || neighbor_dist < threshold) {
          candidates.emplace(neighbor, neighbor_dist, false);
          pool.emplace_back(neighbor, neighbor_dist, false);

          if (pool.size() > ef) {
            std::sort(pool.begin(), pool.end());
            pool.resize(ef);
            threshold = pool.back().distance_;
          } else if (pool.size() == ef) {
            std::sort(pool.begin(), pool.end());
            threshold = pool.back().distance_;
          }
        }
      }
    }

    std::sort(pool.begin(), pool.end());
    return pool;
  }
};

}  // namespace alaya
