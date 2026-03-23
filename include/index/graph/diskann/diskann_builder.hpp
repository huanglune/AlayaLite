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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <span>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "diskann_params.hpp"
#include "index/graph/diskann/visited_list.hpp"
#include "index/graph/graph.hpp"
#include "index/neighbor.hpp"
#include "space/quant/pq.hpp"
#include "space/space_concepts.hpp"
#include "storage/diskann/data_file.hpp"
#include "storage/diskann/diskann_storage.hpp"
#include "utils/candidate_list.hpp"
#include "utils/log.hpp"
#include "utils/macros.hpp"
#include "utils/prefetch.hpp"
#include "utils/progress_bar.hpp"
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

  struct PruneScratch {
    std::vector<IDType> result_buf_;
    std::vector<IDType> reserve_buf_;
    std::vector<IDType> neighbor_snapshot_;
    explicit PruneScratch(uint32_t max_degree) {
      result_buf_.reserve(max_degree + 1);
      reserve_buf_.reserve(max_degree + 1);
      neighbor_snapshot_.reserve(max_degree);
    }
    void clear() { result_buf_.clear(); }
  };

  constexpr static IDType kInvalidID = static_cast<IDType>(-1);
  constexpr static const float kUncomputedDist = std::numeric_limits<float>::max();
  constexpr static const size_t kChunkSize = 256;

  DiskANNBuildParams params_;                 ///< Build parameters
  uint16_t dim_;                              ///< Vector dimension
  std::shared_ptr<DistanceSpaceType> space_;  ///< Data space
  IDType medoid_id_{0};                       ///< Entry point for search
  float average_degree_{0.0F};                ///< Average degree of the graph (for logging)
  std::vector<std::unique_ptr<std::mutex>> locks_;
  uint32_t num_locks_ = 65536;

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
      : params_(params), dim_(space->get_dim()), space_(std::move(space)) {
    locks_.resize(num_locks_);
    for (auto &ptr : locks_) {
      ptr = std::make_unique<std::mutex>();
    }
  }

  ALAYA_NON_COPYABLE_NON_MOVABLE(DiskANNBuilder);
  ~DiskANNBuilder() = default;

  auto get_prune_scratch() -> PruneScratch & {
    static thread_local PruneScratch scratch(params_.max_degree_);
    return scratch;
  }

  auto snapshot_neighbors(const Graph<DataType, IDType> *graph, IDType node_id)
      -> std::span<const IDType> {
    auto &snapshot = get_prune_scratch().neighbor_snapshot_;
    snapshot.clear();

    std::lock_guard<std::mutex> lock(*locks_[node_id % num_locks_]);
    const auto *edges = graph->edges(node_id);
    for (uint32_t i = 0; i < params_.max_degree_; ++i) {
      if (edges[i] == kInvalidID) {
        break;
      }
      snapshot.push_back(edges[i]);
    }
    return snapshot;
  }

  void merge_existing_neighbors(IDType node_id, std::vector<NeighborType> &candidates) {
    std::sort(candidates.begin(), candidates.end(), [](const auto &lhs, const auto &rhs) -> bool {
      if (lhs.id_ != rhs.id_) {
        return lhs.id_ < rhs.id_;
      }
      return lhs.distance_ < rhs.distance_;
    });
    auto last = std::unique(candidates.begin(),
                            candidates.end(),
                            [](const auto &lhs, const auto &rhs) -> bool {
                              return lhs.id_ == rhs.id_;
                            });
    candidates.erase(last, candidates.end());

    for (auto &candidate : candidates) {
      if (candidate.distance_ != kUncomputedDist) {
        continue;
      }
      candidate.distance_ = space_->get_distance(node_id, candidate.id_);
    }
    std::sort(candidates.begin(), candidates.end());
  }

  /**
   * @brief Build the graph using Vamana algorithm.
   *
   * @param thread_num Number of threads for parallel construction
   * @return Unique pointer to the constructed graph
   */
  auto build_graph(uint32_t thread_num = 1) -> std::unique_ptr<Graph<DataType, IDType>> {
    if (thread_num == 0) {
      thread_num = std::max(1U, std::thread::hardware_concurrency());
    }

    auto vec_num = space_->get_data_num();
    LOG_INFO("DiskANN: Building graph with {} vectors, R={}, L={}, alpha={}, threads={}, metric={}",
             vec_num,
             params_.max_degree_,
             params_.ef_construction_,
             params_.alpha_,
             thread_num,
             space_->get_metric_name());
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
      build_pass(graph.get(),
                 thread_num,
                 current_alpha,
                 fmt::format("Pass {}/{}", pass + 1, params_.num_iterations_));

      calculate_average_degree(graph.get());
      LOG_INFO("DiskANN: Pass {}/{} completed, cost: {:.2f}s, avg_r={:.2f}(R={})",
               pass + 1,
               params_.num_iterations_,
               timer.elapsed_s(),
               average_degree_,
               params_.max_degree_);
    }
    LOG_INFO("DiskANN: Graph building cost: {:.2f}s", timer.elapsed_s());

    // Set entry point
    graph->eps_.push_back(medoid_id_);

    return graph;
  }

  /**
   * @brief Save an already-built graph to disk (with optional PQ training).
   *
   * Use this when you already have a constructed graph and want to persist it
   * without rebuilding.
   *
   * @param output_path Path to the output index file
   * @param graph The graph to write
   */
  auto save_disk_index(std::string_view output_path, const Graph<DataType, IDType> &graph) -> void {
    if (params_.is_pq_enabled()) {
      train_pq();
    }
    write_disk_index(output_path, graph);
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
  auto calculate_average_degree(const Graph<DataType, IDType> *graph) -> void {
    auto vec_num = space_->get_data_num();
    uint64_t total_edges = 0;

    for (IDType i = 0; i < vec_num; ++i) {
      const auto *edges = graph->edges(i);
      total_edges += std::find(edges, edges + params_.max_degree_, kInvalidID) - edges;
    }

    average_degree_ = static_cast<float>(total_edges) / static_cast<float>(vec_num);
  }

  /**
   * @brief Initialize graph with random neighbors.
   *
   * @param graph Pointer to the graph
   * @param thread_num Number of threads
   */
  void initialize_random_graph(Graph<DataType, IDType> *graph, uint32_t thread_num) {
    auto vec_num = space_->get_data_num();

    // Initialize random neighbors for each node
    ThreadPool pool(thread_num);
    ProgressBar progress_bar("Init Random Graph", static_cast<uint64_t>(vec_num));

    size_t num_tasks = 0;
    for (size_t chunk_begin = 0; chunk_begin < vec_num; chunk_begin += kChunkSize) {
      size_t chunk_end = std::min(chunk_begin + kChunkSize, static_cast<size_t>(vec_num));
      pool.enqueue([this, chunk_begin, chunk_end, &graph, &progress_bar, vec_num]() -> auto {
        for (size_t node_id = chunk_begin; node_id < chunk_end; ++node_id) {
          auto *cur_edges = graph->edges(node_id);

          // Generate random neighbors
          std::mt19937 rng(node_id);
          std::uniform_int_distribution<IDType> dist_generate(0, vec_num - 1);

          std::vector<IDType> candidates;
          candidates.reserve(params_.max_degree_);

          // Simple rejection sampling (efficient when R << N)
          uint32_t attempts = 0;
          while (candidates.size() < params_.max_degree_ && attempts < params_.max_degree_ * 2) {
            IDType dist = dist_generate(rng);
            if (dist == node_id) {
              continue;
            }

            if (std::find(candidates.begin(), candidates.end(), dist) == candidates.end()) {
              candidates.push_back(dist);
            }
            ++attempts;  // NOLINT
          }
          if (candidates.size() < params_.max_degree_) {
            // Fill remaining with kInvalidID if not enough unique neighbors found
            candidates.resize(params_.max_degree_, kInvalidID);
          }

          memcpy(cur_edges, candidates.data(), candidates.size() * sizeof(IDType));
          progress_bar.tick();
        }
      });
      ++num_tasks;
    }
    pool.wait_until_all_tasks_completed(num_tasks);
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

    // Step 1: Compute centroid (arithmetic mean — metric-agnostic)
    std::vector<double> centroid_acc(dim_, 0.0);
    for (IDType i = 0; i < vec_num; ++i) {
      const auto *vec = space_->get_data_by_id(i);
      for (uint32_t d = 0; d < dim_; ++d) {
        centroid_acc[d] += static_cast<double>(vec[d]);
      }
    }

    // Step 2: Convert centroid to DataType for SIMD distance computation
    std::vector<DataType> centroid(dim_);
    for (uint32_t d = 0; d < dim_; ++d) {
      centroid[d] = static_cast<DataType>(centroid_acc[d] / static_cast<double>(vec_num));
    }

    // Step 3: Find closest point to centroid using the space's distance function
    auto dist_fn = space_->get_dist_func();
    IDType medoid = 0;
    auto min_dist = dist_fn(centroid.data(), space_->get_data_by_id(0), dim_);

    for (IDType i = 1; i < vec_num; ++i) {
      auto dist = dist_fn(centroid.data(), space_->get_data_by_id(i), dim_);
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
  void build_pass(Graph<DataType, IDType> *graph,
                  uint32_t thread_num,
                  float alpha,
                  const std::string &progress_prefix) {
    auto vec_num = space_->get_data_num();

    // Create random permutation
    std::vector<IDType> perm(vec_num);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), std::mt19937(std::random_device()()));

    ThreadPool pool(thread_num);
    ProgressBar progress_bar(progress_prefix, static_cast<uint64_t>(vec_num));

    size_t num_tasks = 0;
    for (IDType chunk_begin = 0; chunk_begin < vec_num;
         chunk_begin += static_cast<IDType>(kChunkSize)) {
      IDType chunk_end = std::min(chunk_begin + static_cast<IDType>(kChunkSize), vec_num);

      pool.enqueue([this, chunk_begin, chunk_end, &graph, &perm, &progress_bar, alpha]() -> auto {
        for (IDType chunk_id = chunk_begin; chunk_id < chunk_end; ++chunk_id) {
          IDType node_id = perm[chunk_id];
          auto candidates = greedy_search(graph, node_id, params_.ef_construction_);

          auto &scratch = get_prune_scratch();  // get thread-local scratch for pruning
          auto &res_buf = scratch.result_buf_;
          {
            std::lock_guard<std::mutex> lock(*locks_[node_id % num_locks_]);
            const auto *edges = graph->edges(node_id);
            for (uint32_t i = 0; i < params_.max_degree_; ++i) {
              if (edges[i] == kInvalidID) {
                break;
              }
              candidates.emplace_back(edges[i], kUncomputedDist, true);
            }
          }

          merge_existing_neighbors(node_id, candidates);
          robust_prune(node_id, candidates, alpha, res_buf);

          {
            std::lock_guard<std::mutex> lock(*locks_[node_id % num_locks_]);
            graph->update(node_id, res_buf.data());
          }

          // Add reverse edges
          for (auto neighbor : res_buf) {
            if (neighbor == kInvalidID) {
              break;
            }
            std::lock_guard<std::mutex> lock(*locks_[neighbor % num_locks_]);
            add_reverse_edge(graph, neighbor, node_id, alpha);
          }

          progress_bar.tick();
        }
      });
      ++num_tasks;
    }
    pool.wait_until_all_tasks_completed(num_tasks);
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
    auto &visited = diskann::GlobalVisitedList::get(vec_num);
    CandidateList<float, IDType> candidates(vec_num, ef + params_.max_degree_);

    // Start from medoid
    visited.mark(medoid_id_);  // mark first, void recompute.
    auto dist = space_->get_distance(query_id, medoid_id_);
    candidates.insert(medoid_id_, dist);

    while (candidates.has_next()) {
      // If the pool is full and the next node to be expanded (cur_) is further away than the worst
      // node in the pool (size_-1) It means it's impossible to find a better result, just prune
      // directly.
      if (candidates.is_full()) {
        if (candidates.dist(candidates.cur_) > candidates.dist(candidates.size() - 1)) {
          break;
        }
      }

      auto cur_id = candidates.pop();
      auto neighbors = snapshot_neighbors(graph, cur_id);
      if (!neighbors.empty()) {
        prefetch_l1(neighbors.data());
      }

      for (auto neighbor : neighbors) {
        if (visited.is_visited(neighbor)) {
          continue;
        }
        visited.mark(neighbor);

        auto neighbor_dist = space_->get_distance(query_id, neighbor);
        candidates.insert(neighbor, neighbor_dist);
      }
    }

    size_t final_size = std::min<size_t>(candidates.size(), ef);
    std::vector<NeighborType> results;
    results.reserve(final_size);
    for (size_t i = 0; i < final_size; ++i) {
      results.emplace_back(candidates.id(i), candidates.dist(i), true);
    }

    return results;
  }

  /**
   * @brief Zero-Malloc Robust Prune
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
  auto robust_prune(IDType node_id,
                    std::vector<NeighborType> &candidates,
                    float alpha,
                    std::vector<IDType> &output_edges) -> void {
    output_edges.clear();

    auto inflate = [alpha](float distance) -> float {
      return distance >= 0 ? distance * alpha : distance / alpha;
    };

    for (const auto &cand : candidates) {
      if (output_edges.size() >= params_.max_degree_) {
        break;
      }
      if (cand.id_ == node_id) {
        continue;
      }

      // Check if cand is α-dominated by any selected neighbor
      bool dominated = false;
      for (const auto &selected_id : output_edges) {
        auto dist_cand_selected = space_->get_distance(cand.id_, selected_id);
        if (inflate(dist_cand_selected) < cand.distance_) {
          dominated = true;
          break;
        }
      }

      if (!dominated) {
        output_edges.push_back(cand.id_);
      }
    }

    // padding with kInvalidID if needed
    if (output_edges.size() < params_.max_degree_) {
      output_edges.resize(params_.max_degree_, kInvalidID);
    }
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
    uint32_t current_degree = 0;

    // Check if edge already exists
    for (; current_degree < params_.max_degree_; ++current_degree) {
      if (edges[current_degree] == kInvalidID) {
        break;
      }
      if (edges[current_degree] == dst) {
        return;
      }
    }

    if (current_degree < params_.max_degree_) {
      edges[current_degree] = dst;
      return;
    }

    // No empty slot, need to prune
    std::vector<NeighborType> candidates;
    candidates.reserve(params_.max_degree_ + 1);

    // Collect existing neighbors
    for (uint32_t i = 0; i < params_.max_degree_; ++i) {
      auto dist = space_->get_distance(src, edges[i]);
      candidates.emplace_back(edges[i], dist);
    }

    // Add new edge
    auto new_dist = space_->get_distance(src, dst);
    candidates.emplace_back(dst, new_dist);
    std::sort(candidates.begin(), candidates.end());

    // Prune
    auto &scratch = get_prune_scratch();
    robust_prune(src, candidates, alpha, scratch.reserve_buf_);

    // directly update
    assert(scratch.reserve_buf_.size() == params_.max_degree_);
    memcpy(edges, scratch.reserve_buf_.data(), params_.max_degree_ * sizeof(IDType));
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
    calculate_average_degree(&graph);
    auto avg_r = average_degree_;

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

    // Write nodes one by one
    for (IDType i = 0; i < vec_num; ++i) {
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
      auto ref = storage.data().get_node(i);
      ref.set_vector(std::span<const DataType>(vec, dim_));
      ref.set_neighbors(std::span<const IDType>(edges, num_neighbors));
    }

    // Flush remaining dirty pages to disk and save metadata
    storage.data().flush();
    storage.save_meta();
    storage.close();

    LOG_INFO("DiskANN: Disk index written successfully");
  }
};

}  // namespace alaya
