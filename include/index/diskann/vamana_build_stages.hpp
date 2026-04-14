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

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "cross_shard_merger.hpp"
#include "kmeans_partitioner.hpp"
#include "shard_vamana_builder.hpp"
#include "space/space_concepts.hpp"
#include "utils/progress_bar.hpp"

namespace alaya {

template <typename DataType = float, typename IDType = uint32_t>
struct PartitionResult {
  uint32_t num_shards_{0};
  std::vector<std::vector<IDType>> shard_members_;
  std::vector<uint64_t> shuffle_offsets_;
  std::vector<uint64_t> shuffle_counts_;
  std::filesystem::path shuffle_path_;
  std::vector<std::filesystem::path> cleanup_paths_;
};

namespace detail {

template <typename DataType, typename IDType>
auto to_partition_result(const PartitionedShardLayout<DataType, IDType> &layout)
    -> PartitionResult<DataType, IDType> {
  PartitionResult<DataType, IDType> result;
  result.num_shards_ = layout.num_shards_;
  result.shard_members_.resize(layout.shard_members_.size());
  for (size_t shard_id = 0; shard_id < layout.shard_members_.size(); ++shard_id) {
    const auto &members = layout.shard_members_[shard_id];
    auto &out = result.shard_members_[shard_id];
    out.reserve(members.size());
    for (auto member : members) {
      out.push_back(static_cast<IDType>(member));
    }
  }
  result.shuffle_offsets_ = layout.shuffle_offsets_;
  result.shuffle_counts_ = layout.shuffle_counts_;
  result.shuffle_path_ = layout.shuffle_path_;
  if (!layout.node_to_shards_path_.empty()) {
    result.cleanup_paths_.push_back(layout.node_to_shards_path_);
  }
  if (!layout.shard_members_path_.empty()) {
    result.cleanup_paths_.push_back(layout.shard_members_path_);
  }
  if (!layout.shuffle_path_.empty()) {
    result.cleanup_paths_.push_back(layout.shuffle_path_);
  }
  return result;
}

template <typename DataType, typename IDType>
auto shard_graph_prefix(const PartitionResult<DataType, IDType> &partition) -> std::string {
  auto shuffle_path = partition.shuffle_path_.string();
  constexpr std::string_view kShuffleSuffix = ".shuffle.bin";
  if (shuffle_path.ends_with(kShuffleSuffix)) {
    return shuffle_path.substr(0, shuffle_path.size() - kShuffleSuffix.size());
  }
  return shuffle_path;
}

}  // namespace detail

template <typename DataType = float, typename IDType = uint32_t>
auto run_partition_stage(
    const std::filesystem::path &data_path,
    uint32_t max_degree,
    const std::filesystem::path &output_prefix,
    const typename KMeansPartitioner<DataType, IDType>::Config &partition_config)
    -> PartitionResult<DataType, IDType> {
  KMeansPartitioner<DataType, IDType> partitioner(partition_config);
  auto layout = partitioner.partition(data_path, max_degree, output_prefix);
  return detail::to_partition_result(layout);
}

template <typename DistanceSpaceType>
  requires Space<DistanceSpaceType> &&
      requires(DistanceSpaceType &space, typename DistanceSpaceType::IDTypeAlias id) {
    space.get_data_by_id(id);
  }
auto run_partition_stage(
    DistanceSpaceType &space,
    uint32_t max_degree,
    const std::filesystem::path &output_prefix,
    const typename KMeansPartitioner<typename DistanceSpaceType::DataTypeAlias,
                                     typename DistanceSpaceType::IDTypeAlias>::Config
        &partition_config)
    -> PartitionResult<typename DistanceSpaceType::DataTypeAlias,
                       typename DistanceSpaceType::IDTypeAlias> {
  using DataType = typename DistanceSpaceType::DataTypeAlias;
  using IDType = typename DistanceSpaceType::IDTypeAlias;
  KMeansPartitioner<DataType, IDType> partitioner(partition_config);
  auto layout = partitioner.partition(space, max_degree, output_prefix);
  return detail::to_partition_result(layout);
}

template <typename DataType, typename IDType, typename DistFunc, typename ShardCallback>
auto run_shard_build_stage(
    const PartitionResult<DataType, IDType> &partition,
    uint32_t dim,
    DistFunc dist_fn,
    const typename ShardVamanaBuilder<DataType, IDType>::Config &shard_config,
    ShardCallback on_shard_complete) -> std::vector<std::filesystem::path> {
  using ShardBuilder = ShardVamanaBuilder<DataType, IDType>;
  using Summary = typename ShardBuilder::ShardExportSummary;

  if (partition.num_shards_ == 0) {
    return {};
  }
  if (partition.shard_members_.size() < partition.num_shards_ ||
      partition.shuffle_offsets_.size() < partition.num_shards_ ||
      partition.shuffle_counts_.size() < partition.num_shards_) {
    throw std::invalid_argument("PartitionResult has inconsistent shard metadata");
  }
  if (partition.shuffle_path_.empty()) {
    throw std::invalid_argument("PartitionResult shuffle_path must not be empty");
  }

  auto graph_prefix = detail::shard_graph_prefix(partition);
  std::vector<std::filesystem::path> shard_graph_paths;
  shard_graph_paths.reserve(partition.num_shards_);

  for (uint32_t shard_id = 0; shard_id < partition.num_shards_; ++shard_id) {
    const auto &members = partition.shard_members_[shard_id];
    if (members.empty()) {
      continue;
    }

    uint64_t shard_total = static_cast<uint64_t>(members.size()) * shard_config.num_iterations_;
    std::string shard_prefix =
        "Shard " + std::to_string(shard_id + 1) + "/" + std::to_string(partition.num_shards_);
    ProgressBar shard_bar(shard_prefix, shard_total);

    auto shard_vectors = ShardBuilder::load_vectors_from_shuffle(partition.shuffle_path_,
                                                                 partition.shuffle_offsets_[shard_id],
                                                                 partition.shuffle_counts_[shard_id],
                                                                 dim);

    ShardBuilder shard_builder(std::move(shard_vectors), dim, members, dist_fn, shard_config);
    shard_builder.build([&shard_bar]() {
      shard_bar.tick();
    });
    shard_bar.finish();

    auto graph_path = graph_prefix + ".shard_" + std::to_string(shard_id) + ".graph";
    Summary summary = shard_builder.export_graph(shard_id, graph_path);
    shard_graph_paths.push_back(summary.graph_path_);
    on_shard_complete(shard_id, summary);
  }

  return shard_graph_paths;
}

template <typename NodeCallback>
auto run_merge_stage(const std::vector<std::filesystem::path> &shard_paths,
                     const CrossShardMerger::Config &merge_config,
                     NodeCallback on_node) -> void {
  if (shard_paths.empty()) {
    return;
  }

  CrossShardMerger merger(merge_config);
  merger.open(shard_paths);
  merger.merge_all([&on_node](const CrossShardMerger::MergedNode &node) {
    on_node(node);
  });
}

}  // namespace alaya
