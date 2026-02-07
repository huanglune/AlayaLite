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
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "diskann_params.hpp"
#include "index/graph/graph.hpp"
#include "index/neighbor.hpp"
#include "space/quant/pq.hpp"
#include "space/space_concepts.hpp"
#include "storage/diskann/diskann_storage.hpp"
#include "utils/log.hpp"
#include "utils/macros.hpp"
#include "utils/thread_pool.hpp"
#include "utils/timer.hpp"

namespace alaya {

/**
 * @brief DiskANN graph builder using the Vamana algorithm.
 *
 * This builder constructs a graph optimized for disk-based approximate nearest
 * neighbor search. It implements the Vamana algorithm from the DiskANN paper.
 *
 * @tparam DistanceSpaceType The distance space type satisfying the Space concept.
 */
template <typename DistanceSpaceType>
  requires Space<DistanceSpaceType>
struct DiskANNBuilder {
  using DataType = typename DistanceSpaceType::DataTypeAlias;
  using DistanceType = typename DistanceSpaceType::DistanceTypeAlias;
  using IDType = typename DistanceSpaceType::IDTypeAlias;
  using DistanceSpaceTypeAlias = DistanceSpaceType;
  using NeighborType = Neighbor<IDType, DistanceType>;

  static constexpr IDType kInvalidID = static_cast<IDType>(-1);

  DiskANNBuildParams params_;  ///< Build parameters
  uint16_t dim_;               ///< Vector dimension

  std::shared_ptr<DistanceSpaceType> space_;  ///< Data space
  IDType medoid_id_{0};                       ///< Entry point for search

  // PQ-related members
  PQQuantizer<DataType> pq_quantizer_;  ///< PQ codebook (trained if PQ is enabled)
  std::vector<uint8_t> pq_codes_;       ///< PQ codes for all vectors

  /**
   * @brief Construct a new DiskANN Builder.
   *
   * @param space Shared pointer to the distance space
   * @param params Build parameters (default values if not specified)
   */
  explicit DiskANNBuilder(std::shared_ptr<DistanceSpaceType> space,
                          const DiskANNBuildParams &params = DiskANNBuildParams{})
      : params_(params), dim_(space->get_dim()), space_(std::move(space)) {}

  ALAYA_NON_COPYABLE_NON_MOVABLE(DiskANNBuilder);
  ~DiskANNBuilder() = default;

  /**
   * @brief Build the graph using Vamana algorithm.
   *
   * @param thread_num Number of threads for parallel construction
   * @return Unique pointer to the constructed graph
   */
  auto build_graph(uint32_t thread_num = 1) -> std::unique_ptr<Graph<DataType, IDType>> {
    auto vec_num = space_->get_data_num();
    LOG_INFO("DiskANN: Building graph with {} vectors, R={}, L={}, alpha={}",
             vec_num,
             params_.max_degree_,
             params_.ef_construction_,
             params_.alpha_);

    // Initialize graph
    auto graph =
        std::make_unique<Graph<DataType, IDType>>(space_->get_capacity(), params_.max_degree_);

    // Initialize with random edges
    initialize_random_graph(graph.get(), thread_num);

    // Compute medoid
    medoid_id_ = compute_medoid();
    LOG_INFO("DiskANN: Medoid ID = {}", medoid_id_);

    // Build graph with multiple passes (2-pass strategy from DiskANN paper)
    // Pass 1: alpha = alpha_first_pass_ (strict pruning, builds k-NN like graph for connectivity)
    // Pass 2+: alpha = alpha_ (relaxed pruning, adds long-range edges as "highways")
    Timer timer;
    for (uint32_t pass = 0; pass < params_.num_iterations_; ++pass) {
      float current_alpha = (pass == 0) ? params_.alpha_first_pass_ : params_.alpha_;
      LOG_INFO("DiskANN: Building pass {}/{}, alpha={:.2f}",
               pass + 1,
               params_.num_iterations_,
               current_alpha);
      build_pass(graph.get(), thread_num, current_alpha);
    }
    LOG_INFO("DiskANN: Graph building cost: {:.2f}s", timer.elapsed() / 1e6);

    // Set entry point
    graph->eps_.push_back(medoid_id_);

    // Compute and log average degree
    auto avg_r = compute_avg_degree(graph.get());
    LOG_INFO("DiskANN: Graph built, avg_r={:.2f}", avg_r);

    return graph;
  }

  /**
   * @brief Build and save index directly to disk.
   *
   * @param output_path Path to the output index file
   * @param thread_num Number of threads for construction
   */
  auto build_disk_index(std::string_view output_path, uint32_t thread_num = 1) -> void {
    // Build graph in memory
    auto graph = build_graph(thread_num);

    // Train PQ if enabled
    if (params_.is_pq_enabled()) {
      train_pq();
    }

    // Write to disk
    write_disk_index(output_path, *graph);
  }

  /**
   * @brief Train PQ quantizer and encode all vectors.
   *
   * Call this after graph construction but before writing to disk.
   */
  void train_pq() {
    auto vec_num = space_->get_data_num();
    uint32_t num_subspaces = params_.num_pq_chunks_;

    if (num_subspaces == 0) {
      // Default: use dimension/8 subspaces (16 dims per subspace)
      num_subspaces = std::max(1U, static_cast<uint32_t>(dim_) / 16);
    }

    // Ensure dimension is divisible by num_subspaces
    if (dim_ % num_subspaces != 0) {
      // Adjust to nearest valid value
      while (num_subspaces > 1 && dim_ % num_subspaces != 0) {
        --num_subspaces;
      }
      LOG_WARN("DiskANN: Adjusted PQ subspaces to {} (dim={} must be divisible)",
               num_subspaces,
               dim_);
    }

    LOG_INFO("DiskANN: Training PQ with {} subspaces on {} vectors", num_subspaces, vec_num);

    // Initialize quantizer
    pq_quantizer_ = PQQuantizer<DataType>(dim_, num_subspaces);

    // Collect training data (optionally sample for large datasets)
    size_t sample_size = vec_num;
    if (params_.pq_sample_rate_ < 100 && vec_num > 10000) {
      sample_size = vec_num * params_.pq_sample_rate_ / 100;
      sample_size = std::max(sample_size, size_t{10000});  // Min 10k samples
    }

    std::vector<DataType> training_data;
    if (sample_size < vec_num) {
      // Sample training data
      LOG_INFO("DiskANN: Sampling {} vectors for PQ training ({}%)",
               sample_size,
               params_.pq_sample_rate_);
      training_data.resize(sample_size * dim_);

      std::mt19937 rng(42);
      std::vector<IDType> indices(vec_num);
      std::iota(indices.begin(), indices.end(), 0);
      std::shuffle(indices.begin(), indices.end(), rng);

      for (size_t i = 0; i < sample_size; ++i) {
        const auto *vec = space_->get_data_by_id(indices[i]);
        std::memcpy(training_data.data() + i * dim_, vec, dim_ * sizeof(DataType));
      }

      pq_quantizer_.fit(training_data.data(), sample_size);
    } else {
      // Use all vectors for training
      training_data.resize(static_cast<size_t>(vec_num) * dim_);
      for (IDType i = 0; i < vec_num; ++i) {
        const auto *vec = space_->get_data_by_id(i);
        std::memcpy(training_data.data() + static_cast<size_t>(i) * dim_,
                    vec,
                    dim_ * sizeof(DataType));
      }
      pq_quantizer_.fit(training_data.data(), vec_num);
    }

    // Encode all vectors
    LOG_INFO("DiskANN: Encoding {} vectors with PQ", vec_num);
    pq_codes_.resize(static_cast<size_t>(vec_num) * num_subspaces);

    for (IDType i = 0; i < vec_num; ++i) {
      const auto *vec = space_->get_data_by_id(i);
      pq_quantizer_.encode(vec, pq_codes_.data() + static_cast<size_t>(i) * num_subspaces);
    }

    LOG_INFO("DiskANN: PQ training completed, code size = {} bytes/vector", num_subspaces);
  }

 private:
  /**
   * @brief Compute the average out-degree of the graph.
   *
   * @param graph Pointer to the graph
   * @return Average out-degree
   */
  auto compute_avg_degree(const Graph<DataType, IDType> *graph) -> float {
    auto vec_num = space_->get_data_num();
    uint64_t total_edges = 0;

    for (IDType i = 0; i < vec_num; ++i) {
      const auto *edges = graph->edges(i);
      for (uint32_t j = 0; j < params_.max_degree_; ++j) {
        if (edges[j] == kInvalidID) {
          break;
        }
        ++total_edges;
      }
    }

    return static_cast<float>(total_edges) / static_cast<float>(vec_num);
  }

  /**
   * @brief Initialize graph with random neighbors.
   *
   * @param graph Pointer to the graph
   * @param thread_num Number of threads
   */
  void initialize_random_graph(Graph<DataType, IDType> *graph, uint32_t thread_num) {
    auto vec_num = space_->get_data_num();
    std::vector<IDType> init_edges(params_.max_degree_, kInvalidID);

    // Initialize random neighbors for each node
    ThreadPool pool(thread_num);
    std::atomic<uint32_t> progress{0};

    for (IDType i = 0; i < vec_num; ++i) {
      pool.enqueue([this, i, vec_num, &graph, &progress]() -> auto {
        std::vector<IDType> edges(params_.max_degree_, kInvalidID);

        // Generate random neighbors
        std::mt19937 rng(i);
        std::uniform_int_distribution<IDType> dist(0, vec_num - 1);

        for (uint32_t j = 0; j < std::min(params_.max_degree_, vec_num - 1); ++j) {
          IDType neighbor;
          do {
            neighbor = dist(rng);
          } while (neighbor == i);
          edges[j] = neighbor;
        }

        graph->insert(edges.data());

        uint32_t cur = progress.fetch_add(1) + 1;
        if (cur % 100000 == 0) {
          LOG_INFO("DiskANN: Initialization progress: [{}/{}]", cur, vec_num);
        }
      });
    }
    pool.wait_until_all_tasks_completed(vec_num);
    LOG_DEBUG("DiskANN: Random graph initialization done");
  }

  /**
   * @brief Compute the medoid (centroid approximation).
   *
   * The medoid is the point closest to the geometric center of all points.
   *
   * @return ID of the medoid
   */
  auto compute_medoid() -> IDType {
    auto vec_num = space_->get_data_num();

    // Compute centroid
    std::vector<double> centroid(dim_, 0.0);
    for (IDType i = 0; i < vec_num; ++i) {
      const auto *vec = space_->get_data_by_id(i);
      for (uint32_t d = 0; d < dim_; ++d) {
        centroid[d] += static_cast<double>(vec[d]);
      }
    }
    for (uint32_t d = 0; d < dim_; ++d) {
      centroid[d] /= static_cast<double>(vec_num);
    }

    // Find point closest to centroid
    IDType medoid = 0;
    double min_dist = std::numeric_limits<double>::max();

    for (IDType i = 0; i < vec_num; ++i) {
      const auto *vec = space_->get_data_by_id(i);
      double dist = 0.0;
      for (uint32_t d = 0; d < dim_; ++d) {
        double diff = static_cast<double>(vec[d]) - centroid[d];
        dist += diff * diff;
      }
      if (dist < min_dist) {
        min_dist = dist;
        medoid = i;
      }
    }

    return medoid;
  }

  /**
   * @brief Perform one pass of the Vamana build algorithm.
   *
   * @param graph Pointer to the graph
   * @param thread_num Number of threads
   * @param alpha Alpha parameter for RobustPrune (1.0 for pass 1, higher for pass 2+)
   */
  void build_pass(Graph<DataType, IDType> *graph, uint32_t thread_num, float alpha) {
    auto vec_num = space_->get_data_num();

    // Create random permutation
    std::vector<IDType> perm(vec_num);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), std::mt19937(std::random_device()()));

    // Node-level locks for concurrent updates
    std::vector<std::mutex> node_locks(vec_num);

    ThreadPool pool(thread_num);
    std::atomic<uint32_t> progress{0};

    for (auto node_id : perm) {
      pool.enqueue([this, node_id, &graph, &node_locks, &progress, vec_num, alpha]() -> auto {
        // Greedy search from medoid to find candidates
        auto candidates = greedy_search(graph, node_id, params_.ef_construction_);

        // RobustPrune to select final neighbors
        auto pruned = robust_prune(node_id, candidates, alpha);

        // Update edges with locking
        {
          std::lock_guard<std::mutex> lock(node_locks[node_id]);
          std::vector<IDType> new_edges(params_.max_degree_, kInvalidID);
          for (size_t i = 0; i < pruned.size() && i < params_.max_degree_; ++i) {
            new_edges[i] = pruned[i];
          }
          graph->update(node_id, new_edges.data());
        }

        // Add reverse edges
        for (auto neighbor : pruned) {
          std::lock_guard<std::mutex> lock(node_locks[neighbor]);
          add_reverse_edge(graph, neighbor, node_id, alpha);
        }

        uint32_t cur = progress.fetch_add(1) + 1;
        if (cur % 100000 == 0) {
          LOG_INFO("DiskANN: Build progress: [{}/{}]", cur, vec_num);
        }
      });
    }
    pool.wait_until_all_tasks_completed(vec_num);
  }

  /**
   * @brief Greedy search to find candidates for a query point.
   *
   * @param graph Pointer to the graph
   * @param query_id ID of the query point
   * @param ef Search list size
   * @return Vector of neighbor candidates sorted by distance
   */
  auto greedy_search(const Graph<DataType, IDType> *graph, IDType query_id, uint32_t ef)
      -> std::vector<NeighborType> {
    auto vec_num = space_->get_data_num();

    // Visited set
    std::vector<bool> visited(vec_num, false);

    // Result pool (sorted by distance)
    std::vector<NeighborType> pool;
    pool.reserve(ef + params_.max_degree_);

    // Start from medoid
    auto dist = space_->get_distance(query_id, medoid_id_);
    pool.emplace_back(medoid_id_, dist, false);
    visited[medoid_id_] = true;

    size_t cursor = 0;
    while (cursor < pool.size()) {
      // Find the first unvisited node
      while (cursor < pool.size() && pool[cursor].flag_) {
        ++cursor;
      }
      if (cursor >= pool.size()) {
        break;
      }

      // Mark as visited
      pool[cursor].flag_ = true;
      auto cur_id = pool[cursor].id_;

      // Expand neighbors
      const auto *edges = graph->edges(cur_id);
      for (uint32_t i = 0; i < params_.max_degree_; ++i) {
        auto neighbor = edges[i];
        if (neighbor == kInvalidID) {
          break;
        }
        if (visited[neighbor]) {
          continue;
        }
        visited[neighbor] = true;

        auto neighbor_dist = space_->get_distance(query_id, neighbor);
        pool.emplace_back(neighbor, neighbor_dist, false);
      }

      // Keep pool sorted and bounded
      std::sort(pool.begin(), pool.end());
      if (pool.size() > ef) {
        pool.resize(ef);
      }

      // Reset cursor if order changed
      cursor = 0;
    }

    return pool;
  }

  /**
   * @brief RobustPrune algorithm from DiskANN paper.
   *
   * Selects neighbors that are not α-dominated by already selected neighbors.
   * When α = 1.0, strict pruning keeps only the closest neighbors (k-NN like).
   * When α > 1.0, relaxed pruning allows diverse neighbors (long-range edges).
   *
   * @param node_id ID of the node being pruned
   * @param candidates Candidate neighbors sorted by distance
   * @param alpha Distance threshold multiplier for pruning
   * @return Vector of selected neighbor IDs
   */
  auto robust_prune(IDType node_id, std::vector<NeighborType> &candidates, float alpha)
      -> std::vector<IDType> {
    std::vector<IDType> result;
    result.reserve(params_.max_degree_);

    // Sort by distance (should already be sorted)
    std::sort(candidates.begin(), candidates.end());

    // All distances from the same space share the same sign convention:
    //   L2 → positive, IP/COS → negative (−dot product).
    // For α-domination we need d * α to inflate "away from zero":
    //   positive d: d * α  → larger  (farther from zero)   ⇒ eff_alpha = α
    //   negative d: d / α  → less negative (farther from zero) ⇒ eff_alpha = 1/α
    float eff_alpha =
        (!candidates.empty() && candidates.front().distance_ < 0) ? (1.0F / alpha) : alpha;

    for (const auto &cand : candidates) {
      if (result.size() >= params_.max_degree_) {
        break;
      }
      if (cand.id_ == node_id) {
        continue;
      }

      // Check if cand is α-dominated by any selected neighbor
      bool dominated = false;
      for (auto selected_id : result) {
        auto dist_cand_selected = space_->get_distance(cand.id_, selected_id);
        if (dist_cand_selected * eff_alpha < cand.distance_) {
          dominated = true;
          break;
        }
      }

      if (!dominated) {
        result.push_back(cand.id_);
      }
    }

    return result;
  }

  /**
   * @brief Add a reverse edge from src to dst.
   *
   * @param graph Pointer to the graph
   * @param src Source node
   * @param dst Destination node
   * @param alpha Alpha parameter for RobustPrune if edge list is full
   */
  void add_reverse_edge(Graph<DataType, IDType> *graph, IDType src, IDType dst, float alpha) {
    auto *edges = graph->edges(src);

    // Check if edge already exists
    for (uint32_t i = 0; i < params_.max_degree_; ++i) {
      if (edges[i] == dst) {
        return;  // Already exists
      }
      if (edges[i] == kInvalidID) {
        // Found empty slot
        edges[i] = dst;
        return;
      }
    }

    // No empty slot, need to prune
    std::vector<NeighborType> candidates;
    candidates.reserve(params_.max_degree_ + 1);

    // Collect existing neighbors
    for (uint32_t i = 0; i < params_.max_degree_; ++i) {
      if (edges[i] == kInvalidID) {
        break;
      }
      auto dist = space_->get_distance(src, edges[i]);
      candidates.emplace_back(edges[i], dist);
    }

    // Add new edge
    auto new_dist = space_->get_distance(src, dst);
    candidates.emplace_back(dst, new_dist);

    // Prune
    auto pruned = robust_prune(src, candidates, alpha);

    // Update edges
    std::fill_n(edges, params_.max_degree_, kInvalidID);
    for (size_t i = 0; i < pruned.size(); ++i) {
      edges[i] = pruned[i];
    }
  }

  /**
   * @brief Write the graph to disk using the three-file storage architecture.
   *
   * Creates .meta, .data, and optionally .pq files at the given base path.
   *
   * @param path Base path for index files (without extension)
   * @param graph The graph to write
   */
  void write_disk_index(std::string_view path, const Graph<DataType, IDType> &graph) {
    auto vec_num = space_->get_data_num();
    auto avg_r = compute_avg_degree(&graph);

    LOG_INFO("DiskANN: Writing disk index to {}", path);
    LOG_INFO("DiskANN: {} vectors, avg_r={:.2f}", vec_num, avg_r);

    // Determine PQ subspaces count (0 = disabled)
    uint32_t num_pq_subspaces = 0;
    if (params_.is_pq_enabled() && !pq_codes_.empty()) {
      num_pq_subspaces = pq_quantizer_.num_subspaces();
    }

    // Create buffer pool for disk I/O caching during build
    BufferPool<IDType> buffer_pool(256, kDataBlockSize);

    // Create storage (three files: .meta, .data, .pq)
    DiskANNStorage<DataType, IDType> storage(&buffer_pool);
    storage.create(path,
                   vec_num,
                   dim_,
                   params_.max_degree_,
                   num_pq_subspaces,
                   static_cast<uint32_t>(space_->metric_));

    // Set metadata
    storage.set_entry_point(static_cast<uint32_t>(medoid_id_));
    storage.set_alpha(params_.alpha_);
    storage.set_build_timestamp(
        static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count()));

    // Write PQ data if enabled
    if (num_pq_subspaces > 0) {
      storage.write_pq_codebook(pq_quantizer_.codebook_data());
      storage.write_pq_codes(pq_codes_.data(), vec_num);
      LOG_INFO("DiskANN: PQ enabled, {} subspaces", num_pq_subspaces);
    }

    // Write nodes block by block (single disk write per block)
    storage.data().batch_modify(0, vec_num, [&](uint32_t i, auto &editor) -> void {
      const auto *vec = space_->get_data_by_id(i);
      const auto *edges = graph.edges(i);

      // Count actual neighbors
      uint32_t num_neighbors = 0;
      for (uint32_t j = 0; j < params_.max_degree_; ++j) {
        if (edges[j] == kInvalidID) {
          break;
        }
        ++num_neighbors;
      }

      // Mark node as valid in metadata
      storage.meta().set_valid(i);

      // Write node data (vector + neighbors)
      editor.set_vector(std::span<const DataType>(vec, dim_));
      editor.set_neighbors(std::span<const IDType>(edges, num_neighbors));
    });

    // Save metadata to disk
    storage.save_meta();
    storage.close();

    LOG_INFO("DiskANN: Disk index written successfully");
  }
};

}  // namespace alaya
