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
#include <string>
#include <string_view>

#include "diskann_builder.hpp"
#include "diskann_params.hpp"
#include "diskann_searcher.hpp"
#include "space/space_concepts.hpp"
#include "utils/log.hpp"
#include "utils/macros.hpp"

namespace alaya {

/**
 * @brief Unified interface for DiskANN index operations.
 *
 * This class provides a high-level API for building, loading, and searching
 * DiskANN indices. It combines the builder and searcher functionalities into
 * a single interface.
 *
 * Usage:
 * ```cpp
 * // Build a new index
 * auto space = std::make_shared<RawSpace<float>>(capacity, dim, MetricType::L2);
 * space->fit(data, num_points);
 * DiskANNIndex<float, uint32_t>::build(space, "index.diskann");
 *
 * // Load and search
 * DiskANNIndex<float, uint32_t> index;
 * index.load("index.diskann");
 * std::vector<uint32_t> results(topk);
 * index.search(query, topk, results.data());
 * ```
 *
 * @tparam DataType The data type of vector elements (default: float)
 * @tparam IDType The data type for node IDs (default: uint32_t)
 * @tparam ReplacerType Buffer pool replacement policy (default: ClockReplacer)
 */
template <typename DataType = float,
          typename IDType = uint32_t,
          ReplacerStrategy ReplacerType = ClockReplacer>
class DiskANNIndex {
 public:
  using DistanceType = float;
  using SearcherType = DiskANNSearcher<DataType, IDType, ReplacerType>;

 private:
  std::unique_ptr<SearcherType> searcher_;  ///< Searcher for query operations
  std::string index_path_;                  ///< Path to the loaded index

 public:
  DiskANNIndex() = default;
  ALAYA_NON_COPYABLE_BUT_MOVABLE(DiskANNIndex);
  ~DiskANNIndex() = default;

  // ==========================================================================
  // Static Build Methods
  // ==========================================================================

  /**
   * @brief Build a DiskANN index from a distance space.
   *
   * @tparam DistanceSpaceType Type satisfying the Space concept
   * @param space Shared pointer to the distance space containing data
   * @param output_path Path to write the index file
   * @param params Build parameters (default values if not specified)
   */
  template <typename DistanceSpaceType>
    requires Space<DistanceSpaceType>
  static auto build(std::shared_ptr<DistanceSpaceType> space,
                    std::string_view output_path,
                    const DiskANNBuildParams &params = DiskANNBuildParams{}) -> void {
    DiskANNBuilder<DistanceSpaceType> builder(space, params);
    builder.build_disk_index(output_path, params.num_threads_);
  }

  /**
   * @brief Build a DiskANN index and return the in-memory graph.
   *
   * This is useful when you want to both have an in-memory graph and save to disk.
   *
   * @tparam DistanceSpaceType Type satisfying the Space concept
   * @param space Shared pointer to the distance space containing data
   * @param output_path Optional path to write the index file (empty to skip)
   * @param params Build parameters (default values if not specified)
   * @return Unique pointer to the constructed graph
   */
  template <typename DistanceSpaceType>
    requires Space<DistanceSpaceType>
  static auto build_graph(std::shared_ptr<DistanceSpaceType> space,
                          std::string_view output_path = "",
                          const DiskANNBuildParams &params = DiskANNBuildParams{})
      -> std::unique_ptr<Graph<typename DistanceSpaceType::DataTypeAlias,
                               typename DistanceSpaceType::IDTypeAlias>> {
    DiskANNBuilder<DistanceSpaceType> builder(space, params);

    // Build the graph
    auto graph = builder.build_graph(params.num_threads_);

    // Optionally write to disk
    if (!output_path.empty()) {
      builder.build_disk_index(output_path, params.num_threads_);
    }

    return graph;
  }

  // ==========================================================================
  // Load and Search Methods
  // ==========================================================================

  /**
   * @brief Load a DiskANN index from file.
   *
   * @param index_path Path to the index file
   */
  auto load(std::string_view index_path, size_t cache_capacity = 4096) -> void {
    searcher_ = std::make_unique<SearcherType>();
    searcher_->open(index_path, cache_capacity);
    index_path_ = std::string(index_path);
    LOG_INFO("DiskANNIndex: Loaded index from {}", index_path_);
  }

  /**
   * @brief Check if an index is loaded.
   *
   * @return true if ready for search
   */
  [[nodiscard]] auto is_loaded() const -> bool {
    return searcher_ != nullptr && searcher_->is_open();
  }

  /**
   * @brief Get the number of vectors in the index.
   *
   * @return Number of indexed vectors
   */
  [[nodiscard]] auto size() const -> uint64_t { return searcher_ ? searcher_->num_points() : 0; }

  /**
   * @brief Get the vector dimension.
   *
   * @return Vector dimension
   */
  [[nodiscard]] auto dimension() const -> uint32_t {
    return searcher_ ? searcher_->dimension() : 0;
  }

  /**
   * @brief Get the index file path.
   *
   * @return Path to the loaded index
   */
  [[nodiscard]] auto path() const -> const std::string & { return index_path_; }

  /**
   * @brief Search for k nearest neighbors.
   *
   * @param query Query vector
   * @param topk Number of neighbors to return
   * @param results Output array for result IDs
   * @param params Search parameters (default values if not specified)
   */
  auto search(const DataType *query,
              uint32_t topk,
              IDType *results,
              const DiskANNSearchParams &params = DiskANNSearchParams{}) -> void {
    if (!is_loaded()) {
      throw std::runtime_error("DiskANNIndex: No index loaded");
    }
    auto result = searcher_->search(query, topk, params);
    std::copy_n(result.ids_.begin(),
                std::min(static_cast<size_t>(topk), result.ids_.size()),
                results);
  }

  /**
   * @brief Search with distances.
   *
   * @param query Query vector
   * @param topk Number of neighbors to return
   * @param results Output array for result IDs
   * @param distances Output array for distances
   * @param params Search parameters (default values if not specified)
   */
  auto search_with_distance(const DataType *query,
                            uint32_t topk,
                            IDType *results,
                            DistanceType *distances,
                            const DiskANNSearchParams &params = DiskANNSearchParams{}) -> void {
    if (!is_loaded()) {
      throw std::runtime_error("DiskANNIndex: No index loaded");
    }
    auto result = searcher_->search(query, topk, params);
    auto count = std::min(static_cast<size_t>(topk), result.ids_.size());
    std::copy_n(result.ids_.begin(), count, results);
    std::copy_n(result.distances_.begin(), std::min(count, result.distances_.size()), distances);
  }

  /**
   * @brief Batch search for multiple queries.
   *
   * @param queries Query vectors (num_queries * dimension)
   * @param num_queries Number of queries
   * @param topk Number of neighbors per query
   * @param results Output array (num_queries * topk)
   * @param params Search parameters (default values if not specified)
   */
  auto batch_search(const DataType *queries,
                    uint32_t num_queries,
                    uint32_t topk,
                    IDType *results,
                    const DiskANNSearchParams &params = DiskANNSearchParams{}) -> void {
    if (!is_loaded()) {
      throw std::runtime_error("DiskANNIndex: No index loaded");
    }
    auto batch_results = searcher_->batch_search(queries, num_queries, topk, params);
    for (uint32_t i = 0; i < num_queries; ++i) {
      auto &result = batch_results[i];
      auto count = std::min(static_cast<size_t>(topk), result.ids_.size());
      std::copy_n(result.ids_.begin(), count, results + static_cast<size_t>(i) * topk);
    }
  }

  /**
   * @brief Close the index and release resources.
   */
  void close() {
    searcher_.reset();
    index_path_.clear();
  }

  /**
   * @brief Get the underlying searcher.
   *
   * @return Pointer to the searcher (nullptr if not loaded)
   */
  [[nodiscard]] auto searcher() -> SearcherType * { return searcher_.get(); }
  [[nodiscard]] auto searcher() const -> const SearcherType * { return searcher_.get(); }
};

}  // namespace alaya
