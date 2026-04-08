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

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "utils/timer.hpp"

#include "index/diskann/cross_shard_merger.hpp"
#include "index/diskann/kmeans_partitioner.hpp"
#include "index/diskann/shard_vamana_builder.hpp"
#include "index/laser/laser_build_params.hpp"
#include "index/laser/laser_build_state.hpp"
#include "index/laser/medoid_generator.hpp"
#include "index/laser/qg_builder.hpp"
#include "index/laser/transform/pca_transform.hpp"
#include "simd/distance_l2.hpp"
#include "space/raw_space.hpp"
#include "space/space_concepts.hpp"
#include "utils/progress_bar.hpp"

namespace alaya {

struct FbinData {
  uint32_t num_points_{0};
  uint32_t dimension_{0};
  std::vector<float> data_;
};

inline auto load_fbin(const std::filesystem::path &path) -> FbinData {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Failed to open .fbin file: " + path.string());
  }

  int32_t num_points = 0;
  int32_t dim = 0;
  input.read(reinterpret_cast<char *>(&num_points), sizeof(num_points));
  input.read(reinterpret_cast<char *>(&dim), sizeof(dim));
  if (!input || num_points <= 0 || dim <= 0) {
    throw std::runtime_error("Invalid .fbin header: " + path.string());
  }

  FbinData result;
  result.num_points_ = static_cast<uint32_t>(num_points);
  result.dimension_ = static_cast<uint32_t>(dim);
  result.data_.resize(static_cast<size_t>(result.num_points_) * result.dimension_);
  input.read(reinterpret_cast<char *>(result.data_.data()),
             static_cast<std::streamsize>(result.data_.size() * sizeof(float)));
  if (!input) {
    throw std::runtime_error("Failed to read .fbin payload: " + path.string());
  }
  return result;
}

class VamanaFormatWriter {
 public:
  VamanaFormatWriter(std::filesystem::path path, uint32_t degree_bound, uint32_t num_nodes)
      : path_(std::move(path)), degree_bound_(degree_bound), in_degree_(num_nodes, 0) {}

  void open() {
    auto dir = path_.parent_path();
    if (!dir.empty()) {
      std::filesystem::create_directories(dir);
    }
    output_.open(path_, std::ios::binary | std::ios::trunc);
    if (!output_) {
      throw std::runtime_error("Failed to create Vamana graph: " + path_.string());
    }

    size_t placeholder_size = 0;
    uint32_t max_degree = degree_bound_;
    uint32_t entry_point = 0;
    size_t frozen_points = 0;
    output_.write(reinterpret_cast<const char *>(&placeholder_size), sizeof(placeholder_size));
    output_.write(reinterpret_cast<const char *>(&max_degree), sizeof(max_degree));
    output_.write(reinterpret_cast<const char *>(&entry_point), sizeof(entry_point));
    output_.write(reinterpret_cast<const char *>(&frozen_points), sizeof(frozen_points));
  }

  void write_node(const CrossShardMerger::MergedNode &node) {
    if (!output_.is_open()) {
      open();
    }

    max_observed_degree_ =
        std::max<uint32_t>(max_observed_degree_, static_cast<uint32_t>(node.neighbor_ids_.size()));

    auto degree = static_cast<uint32_t>(node.neighbor_ids_.size());
    output_.write(reinterpret_cast<const char *>(&degree), sizeof(degree));
    output_.write(reinterpret_cast<const char *>(node.neighbor_ids_.data()),
                  static_cast<std::streamsize>(node.neighbor_ids_.size() * sizeof(uint32_t)));

    for (auto neighbor_id : node.neighbor_ids_) {
      if (neighbor_id < in_degree_.size()) {
        ++in_degree_[neighbor_id];
      }
    }
  }

  void finalize() {
    if (!output_.is_open()) {
      open();
    }

    uint32_t entry_point = 0;
    for (uint32_t node_id = 1; node_id < in_degree_.size(); ++node_id) {
      if (in_degree_[node_id] > in_degree_[entry_point]) {
        entry_point = node_id;
      }
    }

    output_.flush();
    auto file_size = static_cast<size_t>(std::filesystem::file_size(path_));
    output_.seekp(0);
    size_t frozen_points = 0;
    auto header_degree = std::max<uint32_t>(degree_bound_, max_observed_degree_);
    output_.write(reinterpret_cast<const char *>(&file_size), sizeof(file_size));
    output_.write(reinterpret_cast<const char *>(&header_degree), sizeof(header_degree));
    output_.write(reinterpret_cast<const char *>(&entry_point), sizeof(entry_point));
    output_.write(reinterpret_cast<const char *>(&frozen_points), sizeof(frozen_points));
    output_.close();
  }

  [[nodiscard]] auto path() const -> const std::filesystem::path & { return path_; }

 private:
  std::filesystem::path path_;
  uint32_t degree_bound_{0};
  uint32_t max_observed_degree_{0};
  std::vector<uint32_t> in_degree_;
  std::ofstream output_;
};

template <typename DistanceSpaceType>
  requires Space<DistanceSpaceType> &&
           std::is_same_v<typename DistanceSpaceType::DataTypeAlias, float> &&
           requires(DistanceSpaceType &space, typename DistanceSpaceType::IDTypeAlias id) {
             space.get_data_by_id(id);
             space.get_data_num();
             space.get_dim();
           }
class LaserBuilder {
 public:
  using DataType = typename DistanceSpaceType::DataTypeAlias;
  using IDType = typename DistanceSpaceType::IDTypeAlias;

  explicit LaserBuilder(std::shared_ptr<DistanceSpaceType> space,
                        LaserBuildParams params = LaserBuildParams{})
      : space_(std::move(space)), params_(std::move(params)) {
    if (space_ == nullptr) {
      throw std::invalid_argument("LaserBuilder requires a non-null distance space");
    }
  }

  void build(const std::filesystem::path &output_path) {
    initialize_build_context(output_path);

    if (state_.all_completed()) {
      if (!params_.keep_intermediates_) {
        cleanup_intermediates();
      }
      return;
    }

    run_phase(LaserBuildPhase::kPca, [this]() { run_pca(); });
    if (!params_.external_vamana_.empty()) {
      // Use externally-built Vamana graph (e.g. from DiskANN)
      run_phase(LaserBuildPhase::kPartition, []() {});
      run_phase(LaserBuildPhase::kShardBuilds, []() {});
      run_phase(LaserBuildPhase::kMerge, [this]() {
        std::filesystem::copy_file(params_.external_vamana_, vamana_path(),
                                   std::filesystem::copy_options::overwrite_existing);
      });
    } else if (params_.single_shard_) {
      run_phase(LaserBuildPhase::kPartition, []() {});
      run_phase(LaserBuildPhase::kShardBuilds, []() {});
      run_phase(LaserBuildPhase::kMerge, [this]() { run_single_shard_vamana(); });
    } else {
      run_phase(LaserBuildPhase::kPartition, [this]() { run_partition(); });
      run_phase(LaserBuildPhase::kShardBuilds, [this]() { run_shard_builds(); });
      run_phase(LaserBuildPhase::kMerge, [this]() { run_merge(); });
    }
    run_phase(LaserBuildPhase::kMedoids, [this]() { run_medoid(); });
    run_phase(LaserBuildPhase::kQGBuild, [this]() { run_qg_build(); });

    if (!params_.keep_intermediates_) {
      cleanup_intermediates();
    }
  }

  void set_phase_hook_for_test(std::function<void(LaserBuildPhase)> hook) {
    phase_hook_for_test_ = std::move(hook);
  }

 private:
  struct PartitionArtifacts {
    uint32_t num_shards_{0};
    std::vector<std::vector<uint32_t>> shard_members_;
    std::vector<uint64_t> shuffle_offsets_;
    std::vector<uint64_t> shuffle_counts_;
    std::filesystem::path shuffle_path_;
    std::filesystem::path node_to_shards_path_;
    std::filesystem::path shard_members_path_;
  };

  std::shared_ptr<DistanceSpaceType> space_;
  LaserBuildParams params_;
  BuildState state_{std::filesystem::path{}};
  std::filesystem::path output_prefix_;
  std::filesystem::path build_dir_;
  uint32_t num_points_{0};
  uint32_t full_dim_{0};
  uint32_t main_dim_{0};
  uint32_t num_threads_{0};
  std::function<void(LaserBuildPhase)> phase_hook_for_test_;

  void initialize_build_context(const std::filesystem::path &output_path) {
    num_points_ = static_cast<uint32_t>(space_->get_data_num());
    full_dim_ = static_cast<uint32_t>(space_->get_dim());
    if (num_points_ == 0 || full_dim_ == 0) {
      throw std::invalid_argument("LaserBuilder requires a non-empty float dataset");
    }
    if (params_.max_degree_ == 0 || params_.max_degree_ % 32U != 0U) {
      throw std::invalid_argument("LASER max_degree must be a positive multiple of 32");
    }

    output_prefix_ = output_path;
    build_dir_ = output_prefix_.parent_path().empty() ? std::filesystem::current_path()
                                                      : output_prefix_.parent_path();
    std::filesystem::create_directories(build_dir_);

    main_dim_ = params_.resolved_main_dim(full_dim_);
    num_threads_ = params_.resolved_num_threads();

    LaserBuildParams effective_params = params_;
    effective_params.main_dim_ = main_dim_;
    effective_params.num_threads_ = num_threads_;

    state_ = BuildState(build_dir_ / "build_state.json");
    std::string checkpoint_hash = effective_params.params_hash() + "|" + std::to_string(num_points_) +
                                  "|" + std::to_string(full_dim_);
    state_.load_or_reset(checkpoint_hash);
  }

  template <typename Fn>
  void run_phase(LaserBuildPhase phase, Fn &&fn) {
    if (state_.is_completed(phase)) {
      return;
    }

    std::cout << "[LaserBuilder] Phase " << BuildState::phase_name(phase)
              << " started..." << std::endl;
    Timer phase_timer;

    state_.mark_in_progress(phase);
    if (phase_hook_for_test_) {
      phase_hook_for_test_(phase);
    }
    fn();
    state_.mark_completed(phase);

    std::cout << "[LaserBuilder] Phase " << BuildState::phase_name(phase)
              << " completed in " << std::fixed << std::setprecision(1)
              << phase_timer.elapsed_s() << " s" << std::endl;
  }

  void run_pca() {
    // Scope Eigen thread count to match requested num_threads_
    int prev_eigen_threads = Eigen::nbThreads();
    Eigen::setNbThreads(static_cast<int>(num_threads_));

    auto sample_size = params_.resolved_pca_sample_count(num_points_);
    std::vector<uint32_t> ids(num_points_);
    std::iota(ids.begin(), ids.end(), 0U);
    std::mt19937 rng(42);
    std::shuffle(ids.begin(), ids.end(), rng);

    std::vector<float> sample(static_cast<size_t>(sample_size) * full_dim_);
    for (uint32_t sample_idx = 0; sample_idx < sample_size; ++sample_idx) {
      std::copy_n(space_->get_data_by_id(static_cast<IDType>(ids[sample_idx])),
                  full_dim_,
                  sample.data() + static_cast<size_t>(sample_idx) * full_dim_);
    }

    symqg::PCATransform pca(full_dim_, full_dim_);
    pca.train(sample.data(), sample_size);
    if (!pca.save(pca_path().string())) {
      throw std::runtime_error("Failed to save PCA parameters");
    }

    std::ofstream output(pca_base_path(), std::ios::binary | std::ios::trunc);
    if (!output) {
      throw std::runtime_error("Failed to create PCA base file: " + pca_base_path().string());
    }

    auto num_points = static_cast<int32_t>(num_points_);
    auto dim = static_cast<int32_t>(full_dim_);
    output.write(reinterpret_cast<const char *>(&num_points), sizeof(num_points));
    output.write(reinterpret_cast<const char *>(&dim), sizeof(dim));

    constexpr uint32_t kBatchSize = 10000;
    std::vector<float> batch_in(static_cast<size_t>(kBatchSize) * full_dim_);
    std::vector<float> batch_out(static_cast<size_t>(kBatchSize) * full_dim_);

    for (uint32_t start = 0; start < num_points_; start += kBatchSize) {
      uint32_t count = std::min(kBatchSize, num_points_ - start);

      for (uint32_t i = 0; i < count; ++i) {
        std::copy_n(space_->get_data_by_id(static_cast<IDType>(start + i)),
                    full_dim_,
                    batch_in.data() + static_cast<size_t>(i) * full_dim_);
      }

      pca.transform_batch(batch_in.data(), batch_out.data(), count);
      output.write(reinterpret_cast<const char *>(batch_out.data()),
                   static_cast<std::streamsize>(static_cast<size_t>(count) * full_dim_ * sizeof(float)));
    }

    Eigen::setNbThreads(prev_eigen_threads);
  }

  void run_partition() {
    auto pca_vectors = load_fbin(pca_base_path());
    RawSpace<float, float, uint32_t> pca_space(pca_vectors.num_points_, pca_vectors.dimension_, MetricType::L2);
    pca_space.fit(pca_vectors.data_.data(), pca_vectors.num_points_);

    typename KMeansPartitioner<float, uint32_t>::Config config;
    config.max_memory_mb_ = params_.max_memory_mb_;
    KMeansPartitioner<float, uint32_t> partitioner(config);
    (void)partitioner.partition(pca_space, params_.max_degree_, output_prefix_);
  }

  void run_shard_builds() {
    auto partition = load_partition_artifacts();
    state_.ensure_shard_count(partition.num_shards_);

    typename ShardVamanaBuilder<float, uint32_t>::Config config;
    config.max_degree_ = params_.max_degree_;
    config.ef_construction_ = params_.ef_construction_;
    config.alpha_ = params_.alpha_;
    config.max_memory_mb_ = params_.max_memory_mb_;
    config.num_threads_ = num_threads_;

    for (uint32_t shard_id = 0; shard_id < partition.num_shards_; ++shard_id) {
      if (partition.shard_members_[shard_id].empty() || state_.is_shard_completed(shard_id)) {
        continue;
      }

      auto vectors =
          ShardVamanaBuilder<float, uint32_t>::load_vectors_from_shuffle(partition.shuffle_path_,
                                                                         partition.shuffle_offsets_[shard_id],
                                                                         partition.shuffle_counts_[shard_id],
                                                                         full_dim_);
      ShardVamanaBuilder<float, uint32_t> builder(std::move(vectors),
                                                  full_dim_,
                                                  partition.shard_members_[shard_id],
                                                  simd::l2_sqr<float, float>,
                                                  config);
      uint64_t total_work =
          static_cast<uint64_t>(partition.shard_members_[shard_id].size()) * config.num_iterations_;
      std::string label = "Shard " + std::to_string(shard_id + 1) + "/" +
                           std::to_string(partition.num_shards_);
      auto bar = std::make_shared<ProgressBar>(label, total_work);
      builder.build([bar]() { bar->tick(); });
      auto summary = builder.export_graph(shard_id, shard_graph_path(shard_id));
      (void)summary;
      state_.mark_shard_completed(shard_id);
    }
  }

  void run_merge() {
    auto partition = load_partition_artifacts();
    std::vector<std::filesystem::path> shard_paths;
    shard_paths.reserve(partition.num_shards_);
    for (uint32_t shard_id = 0; shard_id < partition.num_shards_; ++shard_id) {
      if (!partition.shard_members_[shard_id].empty()) {
        shard_paths.push_back(shard_graph_path(shard_id));
      }
    }

    CrossShardMerger merger({params_.max_degree_, params_.alpha_});
    merger.open(shard_paths);

    VamanaFormatWriter writer(vamana_path(), params_.max_degree_, num_points_);
    writer.open();
    merger.merge_all([&writer](const CrossShardMerger::MergedNode &node) { writer.write_node(node); });
    writer.finalize();
  }

  void run_single_shard_vamana() {
    auto pca_vectors = load_fbin(pca_base_path());

    std::vector<uint32_t> all_ids(num_points_);
    std::iota(all_ids.begin(), all_ids.end(), 0U);

    typename ShardVamanaBuilder<float, uint32_t>::Config config;
    config.max_degree_ = params_.max_degree_;
    config.ef_construction_ = params_.ef_construction_;
    config.alpha_ = params_.alpha_;
    config.max_memory_mb_ = params_.max_memory_mb_;
    config.num_threads_ = num_threads_;

    ShardVamanaBuilder<float, uint32_t> builder(
        std::move(pca_vectors.data_), full_dim_, std::move(all_ids),
        simd::l2_sqr<float, float>, config);

    uint64_t total_work = static_cast<uint64_t>(num_points_) * config.num_iterations_;
    auto bar = std::make_shared<ProgressBar>("Building Vamana graph", total_work);
    builder.build([bar]() { bar->tick(); });

    // Export directly to Vamana format
    VamanaFormatWriter writer(vamana_path(), params_.max_degree_, num_points_);
    writer.open();
    auto exported = builder.export_nodes();
    for (const auto &node : exported) {
      CrossShardMerger::MergedNode merged;
      merged.global_id_ = node.global_id_;
      merged.neighbor_ids_.reserve(node.neighbors_.size());
      for (const auto &nbr : node.neighbors_) {
        merged.neighbor_ids_.push_back(nbr.id_);
      }
      writer.write_node(merged);
    }
    writer.finalize();
  }

  void run_medoid() {
    Timer load_timer;
    auto pca_vectors = load_fbin(pca_base_path());
    std::cout << "  [Medoid] load PCA vectors: " << std::fixed << std::setprecision(1)
              << load_timer.elapsed_s()
              << " s (" << pca_vectors.num_points_ << " x " << pca_vectors.dimension_ << ")"
              << std::endl;

    MedoidGenerator generator({.num_medoids_ = params_.num_medoids_,
                               .sample_ratio_ = params_.medoid_sample_ratio_,
                               .sample_cap_ = params_.medoid_sample_cap_,
                               .num_threads_ = num_threads_});
    (void)generator.generate(pca_vectors.data_, pca_vectors.num_points_, pca_vectors.dimension_, output_prefix_);
  }

  void run_qg_build() {
    symqg::QuantizedGraph graph(num_points_, params_.max_degree_, main_dim_, full_dim_);
    symqg::QGBuilder builder(graph, params_.ef_build_, num_threads_);
    auto vamana = vamana_path().string();
    auto prefix = output_prefix_.string();
    builder.build(vamana.c_str(), prefix.c_str());
  }

  void cleanup_intermediates() const {
    cleanup_path(pca_base_path());
    cleanup_path(shuffle_path());
    cleanup_path(node_to_shards_path());
    cleanup_path(shard_members_path());
    cleanup_path(vamana_path());
    cleanup_path(tmp_fbin_path());

    auto partition = try_load_partition_artifacts();
    if (!partition.has_value()) {
      for (uint32_t shard_id = 0; shard_id < state_.shard_count(); ++shard_id) {
        cleanup_path(shard_graph_path(shard_id));
      }
      return;
    }

    for (uint32_t shard_id = 0; shard_id < partition->num_shards_; ++shard_id) {
      cleanup_path(shard_graph_path(shard_id));
    }
  }

  [[nodiscard]] auto try_load_partition_artifacts() const -> std::optional<PartitionArtifacts> {
    if (!std::filesystem::exists(shard_members_path()) || !std::filesystem::exists(shuffle_path())) {
      return std::nullopt;
    }
    return load_partition_artifacts();
  }

  [[nodiscard]] auto load_partition_artifacts() const -> PartitionArtifacts {
    auto persisted = KMeansPartitioner<float, uint32_t>::load_shard_members(shard_members_path());

    PartitionArtifacts artifacts;
    artifacts.num_shards_ = persisted.num_shards_;
    artifacts.shard_members_ = std::move(persisted.members_);
    artifacts.shuffle_offsets_.assign(artifacts.num_shards_, 0);
    artifacts.shuffle_counts_.assign(artifacts.num_shards_, 0);
    artifacts.shuffle_path_ = shuffle_path();
    artifacts.node_to_shards_path_ = node_to_shards_path();
    artifacts.shard_members_path_ = shard_members_path();

    uint64_t offset = 0;
    auto row_bytes = static_cast<uint64_t>(full_dim_) * sizeof(float);
    for (uint32_t shard_id = 0; shard_id < artifacts.num_shards_; ++shard_id) {
      artifacts.shuffle_offsets_[shard_id] = offset;
      artifacts.shuffle_counts_[shard_id] = artifacts.shard_members_[shard_id].size();
      offset += static_cast<uint64_t>(artifacts.shuffle_counts_[shard_id]) * row_bytes;
    }
    return artifacts;
  }

  [[nodiscard]] auto pca_path() const -> std::filesystem::path {
    return output_prefix_.string() + "_pca.bin";
  }

  [[nodiscard]] auto pca_base_path() const -> std::filesystem::path {
    return output_prefix_.string() + "_pca_base.fbin";
  }

  [[nodiscard]] auto node_to_shards_path() const -> std::filesystem::path {
    return output_prefix_.string() + ".node_to_shards.bin";
  }

  [[nodiscard]] auto shard_members_path() const -> std::filesystem::path {
    return output_prefix_.string() + ".shard_members.bin";
  }

  [[nodiscard]] auto shuffle_path() const -> std::filesystem::path {
    return output_prefix_.string() + ".shuffle.bin";
  }

  [[nodiscard]] auto shard_graph_path(uint32_t shard_id) const -> std::filesystem::path {
    return output_prefix_.string() + ".shard_" + std::to_string(shard_id) + ".graph";
  }

  [[nodiscard]] auto vamana_path() const -> std::filesystem::path {
    return output_prefix_.string() + ".vamana.index";
  }

  [[nodiscard]] auto tmp_fbin_path() const -> std::filesystem::path {
    return output_prefix_.string() + "_tmp.fbin";
  }

  static void cleanup_path(const std::filesystem::path &path) {
    std::error_code error;
    std::filesystem::remove(path, error);
  }
};

}  // namespace alaya
