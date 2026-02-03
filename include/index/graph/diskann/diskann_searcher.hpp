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
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "disk_layout.hpp"
#include "diskann_params.hpp"
#include "index/neighbor.hpp"
#include "simd/distance_l2.hpp"
#include "storage/aligned_reader.hpp"
#include "utils/bitset.hpp"
#include "utils/log.hpp"

namespace alaya {

/**
 * @brief DiskANN searcher for disk-based approximate nearest neighbor search.
 *
 * This class provides efficient disk-based search using beam search algorithm.
 * It reads nodes from disk on-demand and maintains a search pool.
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
  std::unique_ptr<AlignedFileReader> reader_;     ///< File reader for disk access
  DiskIndexHeader header_;                        ///< Index header metadata
  DiskNodeBuffer<DataType, IDType> node_buffer_;  ///< Buffer for reading nodes
  Bitset visited_;                                ///< Visited node tracking

  // Cached parameters
  uint32_t dimension_{0};
  uint32_t max_degree_{0};
  uint64_t num_points_{0};
  IDType medoid_id_{0};
  size_t node_size_{0};

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
   * @param index_path Path to the index file
   */
  auto open(std::string_view index_path) -> void {
    // Open file with Direct IO if possible
    reader_ = std::make_unique<AlignedFileReader>();
    auto status = reader_->open(index_path,
                                AlignedFileReader::OpenMode::kReadOnly |
                                    AlignedFileReader::OpenMode::kDirectIO);

    if (status != IOStatus::kSuccess) {
      // Try without Direct IO
      status = reader_->open(index_path, AlignedFileReader::OpenMode::kReadOnly);
      if (status != IOStatus::kSuccess) {
        throw std::runtime_error("Failed to open index file: " + std::string(index_path));
      }
      LOG_WARN("DiskANNSearcher: Opened {} without Direct IO", index_path);
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

    // Allocate node buffer
    node_buffer_.allocate(dimension_, max_degree_, 1);

    // Initialize visited bitset
    visited_.resize(num_points_);

    LOG_INFO("DiskANNSearcher: Opened index with {} points, dim={}, R={}, medoid={}",
             num_points_,
             dimension_,
             max_degree_,
             medoid_id_);
  }

  /**
   * @brief Check if the searcher is ready.
   *
   * @return true if a valid index is loaded
   */
  [[nodiscard]] auto is_open() const -> bool { return reader_ != nullptr && reader_->is_open(); }

  /**
   * @brief Get the number of points in the index.
   *
   * @return Number of indexed vectors
   */
  [[nodiscard]] auto num_points() const -> uint64_t { return num_points_; }

  /**
   * @brief Get the vector dimension.
   *
   * @return Vector dimension
   */
  [[nodiscard]] auto dimension() const -> uint32_t { return dimension_; }

  /**
   * @brief Get the maximum out-degree.
   *
   * @return Maximum number of neighbors per node
   */
  [[nodiscard]] auto max_degree() const -> uint32_t { return max_degree_; }

  /**
   * @brief Get the index header.
   *
   * @return Reference to the header structure
   */
  [[nodiscard]] auto header() const -> const DiskIndexHeader & { return header_; }

  /**
   * @brief Search for k nearest neighbors of a query vector.
   *
   * @param query Pointer to the query vector
   * @param topk Number of nearest neighbors to return
   * @param results Output array for result IDs (must have space for topk elements)
   * @param ef Search list size (0 = use default = max(topk, 50))
   */
  auto search(const DataType *query, uint32_t topk, IDType *results, uint32_t ef = 0) -> void {
    if (!is_open()) {
      throw std::runtime_error("DiskANNSearcher: Index not loaded");
    }

    // Set default ef if not specified
    if (ef == 0) {
      ef = std::max(topk, 50U);
    }

    // Perform beam search
    auto neighbors = beam_search(query, ef);

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
   *
   * @param query Pointer to the query vector
   * @param topk Number of nearest neighbors to return
   * @param results Output array for result IDs
   * @param params Search parameters
   */
  auto search(const DataType *query,
              uint32_t topk,
              IDType *results,
              const DiskANNSearchParams &params) -> void {
    search(query, topk, results, params.ef_search_);
  }

  /**
   * @brief Search for k nearest neighbors with distances.
   *
   * @param query Pointer to the query vector
   * @param topk Number of nearest neighbors to return
   * @param results Output array for result IDs
   * @param distances Output array for distances
   * @param ef Search list size
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

    auto neighbors = beam_search(query, ef);

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
   *
   * @param query Pointer to the query vector
   * @param topk Number of nearest neighbors to return
   * @param results Output array for result IDs
   * @param distances Output array for distances
   * @param params Search parameters
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
   *
   * @param queries Pointer to query vectors (row-major: num_queries * dimension)
   * @param num_queries Number of queries
   * @param topk Number of nearest neighbors per query
   * @param results Output array (num_queries * topk)
   * @param ef Search list size
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
   *
   * @param queries Pointer to query vectors (row-major: num_queries * dimension)
   * @param num_queries Number of queries
   * @param topk Number of nearest neighbors per query
   * @param results Output array (num_queries * topk)
   * @param params Search parameters
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
   * @brief Read a node from disk.
   *
   * @param node_id ID of the node to read
   * @return Node accessor for the read node
   */
  auto read_node(IDType node_id) -> NodeAccessor {
    uint64_t offset = kDiskSectorSize + static_cast<uint64_t>(node_id) * node_size_;
    auto bytes = reader_->read(node_buffer_.data(), node_size_, offset);

    if (bytes != static_cast<ssize_t>(node_size_)) {
      LOG_ERROR("Failed to read node {}: got {} bytes, expected {}", node_id, bytes, node_size_);
      throw std::runtime_error("Failed to read node from disk");
    }

    return node_buffer_.get_node(0);
  }

  /**
   * @brief Compute L2 distance between query and a vector.
   *
   * @param query Query vector
   * @param vec Target vector
   * @return L2 distance
   */
  auto compute_distance(const DataType *query, const DataType *vec) -> DistanceType {
    return simd::l2_sqr(query, vec, dimension_);
  }

  /**
   * @brief Beam search algorithm for disk-based ANN.
   *
   * @param query Query vector
   * @param ef Search list size
   * @return Vector of nearest neighbors sorted by distance
   */
  auto beam_search(const DataType *query, uint32_t ef) -> std::vector<NeighborType> {
    // Reset visited bitset
    visited_.reset();

    // Result pool (max-heap to maintain top-ef candidates)
    std::vector<NeighborType> pool;
    pool.reserve(ef + max_degree_);

    // Priority queue for candidates (min-heap by distance)
    auto cmp = [](const NeighborType &a, const NeighborType &b) -> auto {
      return a.distance_ > b.distance_;  // Min-heap
    };
    std::priority_queue<NeighborType, std::vector<NeighborType>, decltype(cmp)> candidates(cmp);

    // Start from medoid
    auto accessor = read_node(medoid_id_);
    auto dist = compute_distance(query, accessor.vector_data());
    candidates.emplace(medoid_id_, dist, false);
    pool.emplace_back(medoid_id_, dist, false);
    visited_.set(medoid_id_);

    // Track the threshold distance (worst distance in top-ef)
    DistanceType threshold = std::numeric_limits<DistanceType>::max();

    while (!candidates.empty()) {
      // Get closest unvisited candidate
      auto cur = candidates.top();
      candidates.pop();

      // Early termination: if current distance exceeds threshold and pool is full
      if (pool.size() >= ef && cur.distance_ > threshold) {
        break;
      }

      // Mark as expanded
      cur.flag_ = true;

      // Read node from disk
      accessor = read_node(cur.id_);

      // Expand neighbors
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

        // Read neighbor node to get vector for distance computation
        auto neighbor_accessor = read_node(neighbor);
        auto neighbor_dist = compute_distance(query, neighbor_accessor.vector_data());

        // Add to pool if better than threshold or pool not full
        if (pool.size() < ef || neighbor_dist < threshold) {
          candidates.emplace(neighbor, neighbor_dist, false);
          pool.emplace_back(neighbor, neighbor_dist, false);

          // Update threshold
          if (pool.size() > ef) {
            // Keep pool bounded
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

    // Sort final results
    std::sort(pool.begin(), pool.end());

    return pool;
  }
};

}  // namespace alaya
