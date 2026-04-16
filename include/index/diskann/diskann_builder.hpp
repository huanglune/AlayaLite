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
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "cross_shard_merger.hpp"
#include "diskann_params.hpp"
#include "kmeans_partitioner.hpp"
#include "shard_vamana_builder.hpp"
#include "space/space_concepts.hpp"
#include "storage/buffer/buffer_pool.hpp"
#include "storage/diskann/data_file.hpp"
#include "storage/diskann/diskann_storage.hpp"
#include "utils/console.hpp"
#include "utils/log.hpp"
#include "utils/macros.hpp"
#include "utils/progress_bar.hpp"
#include "utils/timer.hpp"
#include "vamana_build_stages.hpp"

namespace alaya {

/**
 * @brief DiskANN out-of-core graph builder.
 *
 * Builds a disk-based ANN index via:
 *   Phase 1: KMeans partitioning with overlapping assignment
 *   Phase 2: Per-shard in-memory Vamana construction
 *   Phase 3: Cross-shard merge with distance-ordered selection
 *   Phase 4: Finalization (entry point, MetaFile, cleanup)
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

  DiskANNBuildParams params_;                 ///< Build parameters
  uint16_t dim_;                              ///< Vector dimension
  std::shared_ptr<DistanceSpaceType> space_;  ///< Data space

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
   * @brief Build a DiskANN index and write it to disk.
   *
   * Orchestrates the full out-of-core pipeline:
   * partitioning -> per-shard Vamana -> cross-shard merge -> finalization.
   *
   * @param output_path Base path for index files (without extension)
   */
  void build(std::string_view output_path) {
    auto vec_num = space_->get_data_num();
    LOG_INFO(
        "DiskANN: Building out-of-core index with {} vectors, R={}, dim={}, "
        "max_memory={}MB, overlap={}",
        vec_num,
        params_.max_degree_,
        dim_,
        params_.max_memory_mb_,
        params_.overlap_factor_);

    Timer total_timer;

    // Create intermediate file prefix
    auto output_dir = std::filesystem::path(output_path).parent_path();
    if (!output_dir.empty()) {
      std::filesystem::create_directories(output_dir);
    }
    auto intermediate_prefix = std::filesystem::path(output_path).string() + ".build_tmp";

    // Track intermediate files for cleanup (on success or failure)
    std::vector<std::filesystem::path> cleanup_paths;

    try {
      // ---- Phase 1: KMeans Partitioning ----
      Timer phase_timer;
      console::phase_start("Phase 1", "KMeans partitioning");

      typename KMeansPartitioner<DataType, IDType>::Config partition_config;
      partition_config.max_memory_mb_ = params_.max_memory_mb_;
      partition_config.sample_rate_ = params_.sample_rate_;
      partition_config.overlap_factor_ = params_.overlap_factor_;

      auto partition =
          run_partition_stage(*space_, params_.max_degree_, intermediate_prefix, partition_config);
      cleanup_paths.insert(cleanup_paths.end(),
                           partition.cleanup_paths_.begin(),
                           partition.cleanup_paths_.end());
      auto shard_capacity =
          KMeansPartitioner<DataType, IDType>::compute_shard_capacity(dim_,
                                                                      params_.max_degree_,
                                                                      params_.max_memory_mb_);

      console::phase_done("KMeans partitioning", phase_timer.elapsed_s());
      LOG_INFO("DiskANN: Phase 1 - {} shards, capacity {}", partition.num_shards_, shard_capacity);

      // ---- Phase 2: Per-Shard Vamana Build ----
      phase_timer.reset();
      console::phase_start("Phase 2", "Per-shard Vamana build");

      auto dist_fn = space_->get_dist_func();

      typename ShardVamanaBuilder<DataType, IDType>::Config shard_config;
      // Reduce per-shard degree so that after merge (each node appears in
      // ~overlap_factor shards) the combined degree is ~R.
      // Formula: shard_degree = 2*R / (overlap_factor + 1).
      // overlap=2 → 2R/3, overlap=1 → R (no reduction), overlap=3 → R/2.
      shard_config.max_degree_ =
          std::max(1U, 2 * params_.max_degree_ / (params_.overlap_factor_ + 1));
      shard_config.ef_construction_ = params_.ef_construction_;
      shard_config.num_iterations_ = params_.num_iterations_;
      shard_config.alpha_ = params_.alpha_;
      shard_config.alpha_first_pass_ = params_.alpha_first_pass_;
      shard_config.max_memory_mb_ = params_.max_memory_mb_;
      shard_config.num_threads_ = params_.num_threads_;

      auto shard_graph_paths =
          run_shard_build_stage(partition,
                                dim_,
                                dist_fn,
                                shard_config,
                                [&](uint32_t shard_id,
                                    const typename ShardVamanaBuilder<DataType, IDType>::
                                        ShardExportSummary &summary) {
                                  cleanup_paths.push_back(summary.graph_path_);
                                  LOG_INFO(
                                      "DiskANN: Shard {}/{} complete, {} nodes exported, "
                                      "peak ~{}MB",
                                      shard_id + 1,
                                      partition.num_shards_,
                                      summary.num_nodes_,
                                      summary.estimated_peak_memory_bytes_ / (1024 * 1024));
                                });

      console::phase_done("Per-shard Vamana build", phase_timer.elapsed_s());

      // ---- Phase 3: Cross-Shard Merge ----
      phase_timer.reset();
      console::phase_start("Phase 3", "Cross-shard merge");

      CrossShardMerger::Config merge_config;
      merge_config.max_degree_ = params_.max_degree_;
      merge_config.alpha_ = params_.alpha_;

      BufferPool<IDType> buffer_pool(256, kDataBlockSize);
      DiskANNStorage<DataType, IDType> storage(&buffer_pool);
      storage.create(output_path,
                     vec_num,
                     dim_,
                     params_.max_degree_,
                     0,
                     static_cast<uint32_t>(space_->metric_));

      uint32_t nodes_written = 0;
      ProgressBar merge_bar("Cross-shard merge", vec_num);
      run_merge_stage(shard_graph_paths,
                      merge_config,
                      [&](const CrossShardMerger::MergedNode &node) {
                        auto node_id = static_cast<IDType>(node.global_id_);
                        const auto *vec = space_->get_data_by_id(node_id);

                        storage.meta().set_valid(node_id);
                        storage.meta().insert_mapping(static_cast<uint32_t>(node_id),
                                                      static_cast<uint32_t>(node_id));

                        auto ref = storage.data().get_node(node_id);
                        ref.set_vector(std::span<const DataType>(vec, dim_));

                        std::vector<IDType> neighbor_ids(node.neighbor_ids_.begin(),
                                                         node.neighbor_ids_.end());
                        ref.set_neighbors(
                            std::span<const IDType>(neighbor_ids.data(), neighbor_ids.size()));

                        ++nodes_written;
                        merge_bar.tick();
                      });
      merge_bar.finish();

      console::phase_done("Cross-shard merge", phase_timer.elapsed_s());
      LOG_INFO("DiskANN: Phase 3 - {} nodes merged", nodes_written);

      // ---- Phase 4: Finalization ----
      phase_timer.reset();
      console::phase_start("Phase 4", "Finalization");

      auto entry_point = select_entry_point(partition);
      LOG_INFO("DiskANN: Entry point = {}", entry_point);

      storage.set_entry_point(static_cast<uint32_t>(entry_point));
      storage.set_alpha(params_.alpha_);
      storage.set_build_timestamp(
          static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count()));

      storage.data().flush();
      storage.save_meta();

#ifndef NDEBUG
      validate_connectivity(storage, entry_point, vec_num);
#endif

      storage.close();

      // Cleanup intermediate files (success path)
      cleanup_intermediates(cleanup_paths);

      console::phase_done("Finalization", phase_timer.elapsed_s());
      auto total_elapsed = total_timer.elapsed_s();
      std::string detail = std::to_string(vec_num) + " vectors";
      console::summary("Index built", detail, total_elapsed);
      LOG_INFO("DiskANN: Index built successfully in {:.2f}s total", total_elapsed);
    } catch (...) {
      LOG_WARN("DiskANN: Build failed, cleaning up intermediate files");
      cleanup_intermediates(cleanup_paths);
      throw;
    }
  }

 private:
  /**
   * @brief Select global entry point: nearest vector to weighted mean of shard vectors.
   *
   * Each shard contributes proportionally to its member count.
   */
  auto select_entry_point(const PartitionResult<DataType, IDType> &partition) -> IDType {
    std::vector<double> weighted_centroid(dim_, 0.0);
    uint64_t total_weight = 0;

    for (uint32_t shard_id = 0; shard_id < partition.num_shards_; ++shard_id) {
      const auto &members = partition.shard_members_[shard_id];
      total_weight += members.size();
      for (auto node_id : members) {
        const auto *vec = space_->get_data_by_id(node_id);
        for (uint32_t d = 0; d < dim_; ++d) {
          weighted_centroid[d] += static_cast<double>(vec[d]);
        }
      }
    }

    if (total_weight > 0) {
      for (uint32_t d = 0; d < dim_; ++d) {
        weighted_centroid[d] /= static_cast<double>(total_weight);
      }
    }

    std::vector<DataType> centroid_vec(dim_);
    for (uint32_t d = 0; d < dim_; ++d) {
      centroid_vec[d] = static_cast<DataType>(weighted_centroid[d]);
    }

    auto dist_fn = space_->get_dist_func();
    auto vec_num = space_->get_data_num();
    IDType best_id = 0;
    auto best_dist = dist_fn(centroid_vec.data(), space_->get_data_by_id(0), dim_);

    for (IDType i = 1; i < vec_num; ++i) {
      auto d = dist_fn(centroid_vec.data(), space_->get_data_by_id(i), dim_);
      if (d < best_dist) {
        best_dist = d;
        best_id = i;
      }
    }

    return best_id;
  }

  /**
   * @brief BFS connectivity validation from the entry point.
   *
   * Logs a warning if the graph is not fully connected.
   */
  void validate_connectivity(DiskANNStorage<DataType, IDType> &storage,
                             IDType entry_point,
                             IDType num_nodes) {
    std::vector<bool> visited(num_nodes, false);
    std::vector<IDType> queue;
    queue.reserve(num_nodes);
    queue.push_back(entry_point);
    visited[entry_point] = true;

    size_t head = 0;
    while (head < queue.size()) {
      auto node_id = queue[head++];
      auto ref = storage.data().get_node(node_id);
      auto nbrs = ref.neighbors();
      for (auto neighbor : nbrs) {
        if (neighbor < num_nodes && !visited[neighbor]) {
          visited[neighbor] = true;
          queue.push_back(neighbor);
        }
      }
    }

    auto visited_count = static_cast<size_t>(std::count(visited.begin(), visited.end(), true));
    LOG_INFO("DiskANN: BFS connectivity: {}/{} nodes reachable from entry point {}",
             visited_count,
             num_nodes,
             entry_point);
    if (visited_count < num_nodes) {
      LOG_WARN("DiskANN: Graph is NOT fully connected! {}/{} nodes unreachable",
               num_nodes - visited_count,
               num_nodes);
    }
  }

  /**
   * @brief Delete intermediate build files.
   */
  static void cleanup_intermediates(const std::vector<std::filesystem::path> &paths) {
    for (const auto &path : paths) {
      std::error_code ec;
      if (std::filesystem::exists(path, ec)) {
        std::filesystem::remove(path, ec);
        if (ec) {
          LOG_WARN("DiskANN: Failed to clean up {}: {}", path.string(), ec.message());
        }
      }
    }
    LOG_INFO("DiskANN: Cleaned up {} intermediate files", paths.size());
  }
};

}  // namespace alaya
