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

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include "index/diskann/kmeans_partitioner.hpp"
#include "index/diskann/shard_vamana_builder.hpp"
#include "space/raw_space.hpp"

namespace alaya {
namespace {

constexpr uint32_t kDim = 8;
constexpr uint32_t kNumPoints = 12000;
constexpr uint32_t kMaxDegree = 16;

auto make_vectors() -> std::vector<float> {
  std::vector<float> data(static_cast<size_t>(kNumPoints) * kDim);
  for (uint32_t id = 0; id < kNumPoints; ++id) {
    float cluster_bias = static_cast<float>(id % 3) * 15.0F;
    data[static_cast<size_t>(id) * kDim] = cluster_bias + static_cast<float>(id % 101) * 0.01F;
    data[static_cast<size_t>(id) * kDim + 1] =
        static_cast<float>((id / 3) % 101) * 0.01F + cluster_bias * 0.5F;
    for (uint32_t d = 2; d < kDim; ++d) {
      data[static_cast<size_t>(id) * kDim + d] = static_cast<float>(id * 10 + d);
    }
  }
  return data;
}

auto make_space(const std::vector<float> &data) -> std::shared_ptr<RawSpace<>> {
  auto space = std::make_shared<RawSpace<>>(kNumPoints, kDim, MetricType::L2);
  space->fit(data.data(), kNumPoints);
  return space;
}

auto make_temp_prefix(const char *label) -> std::filesystem::path {
  auto suffix = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  auto dir = std::filesystem::temp_directory_path() / ("shard_vamana_builder_test_" + suffix);
  std::filesystem::create_directories(dir);
  return dir / label;
}

struct ExportedNeighborRecord {
  uint32_t id_;
  float distance_;
};

struct ExportedNodeRecord {
  uint32_t global_id_;
  std::vector<ExportedNeighborRecord> neighbors_;
};

auto read_exported_nodes(const std::filesystem::path &path) -> std::vector<ExportedNodeRecord> {
  std::ifstream input(path, std::ios::binary);
  EXPECT_TRUE(input.is_open());

  std::vector<ExportedNodeRecord> nodes;
  while (true) {
    ExportedNodeRecord node;
    uint16_t count = 0;
    input.read(reinterpret_cast<char *>(&node.global_id_), sizeof(node.global_id_));
    if (!input) {
      break;
    }
    input.read(reinterpret_cast<char *>(&count), sizeof(count));
    node.neighbors_.resize(count);
    for (uint16_t i = 0; i < count; ++i) {
      input.read(reinterpret_cast<char *>(&node.neighbors_[i].id_), sizeof(node.neighbors_[i].id_));
      input.read(reinterpret_cast<char *>(&node.neighbors_[i].distance_),
                 sizeof(node.neighbors_[i].distance_));
    }
    nodes.push_back(std::move(node));
  }
  return nodes;
}

class ShardVamanaBuilderTest : public ::testing::Test {};

TEST_F(ShardVamanaBuilderTest, ExportContainsValidGlobalIdsAndCorrectDistances) {
  auto data = make_vectors();
  auto space = make_space(data);
  auto prefix = make_temp_prefix("shard_export");

  KMeansPartitioner<float> partitioner({.max_memory_mb_ = 1, .sample_rate_ = 0.02F, .overlap_factor_ = 2});
  auto layout = partitioner.partition(*space, kMaxDegree, prefix);

  auto shard_id = 0U;
  auto vectors = ShardVamanaBuilder<float>::load_vectors_from_shuffle(
      layout.shuffle_path_, layout.shuffle_offsets_[shard_id], layout.shuffle_counts_[shard_id], kDim);

  ShardVamanaBuilder<float> builder(
      std::move(vectors),
      kDim,
      layout.shard_members_[shard_id],
      space->get_dist_func(),
      {.max_degree_ = kMaxDegree, .ef_construction_ = 32, .num_iterations_ = 2, .max_memory_mb_ = 2});
  builder.build();

  auto export_path = prefix.parent_path() / "shard_0.graph";
  auto summary = builder.export_graph(shard_id, export_path);
  auto exported = read_exported_nodes(summary.graph_path_);

  ASSERT_EQ(exported.size(), layout.shard_members_[shard_id].size());
  for (const auto &node : exported) {
    EXPECT_TRUE(std::find(layout.shard_members_[shard_id].begin(),
                          layout.shard_members_[shard_id].end(),
                          node.global_id_) != layout.shard_members_[shard_id].end());
    for (const auto &neighbor : node.neighbors_) {
      EXPECT_TRUE(std::find(layout.shard_members_[shard_id].begin(),
                            layout.shard_members_[shard_id].end(),
                            neighbor.id_) != layout.shard_members_[shard_id].end());
      auto expected = space->get_dist_func()(space->get_data_by_id(node.global_id_),
                                             space->get_data_by_id(neighbor.id_),
                                             kDim);
      EXPECT_FLOAT_EQ(neighbor.distance_, expected);
    }
  }
}

TEST_F(ShardVamanaBuilderTest, ExportIsSortedByGlobalIdAndNeighborDistance) {
  auto data = make_vectors();
  auto space = make_space(data);
  auto prefix = make_temp_prefix("shard_sorted");

  KMeansPartitioner<float> partitioner({.max_memory_mb_ = 1, .sample_rate_ = 0.02F, .overlap_factor_ = 2});
  auto layout = partitioner.partition(*space, kMaxDegree, prefix);

  auto shard_id = 1U;
  auto vectors = ShardVamanaBuilder<float>::load_vectors_from_shuffle(
      layout.shuffle_path_, layout.shuffle_offsets_[shard_id], layout.shuffle_counts_[shard_id], kDim);

  ShardVamanaBuilder<float> builder(
      std::move(vectors),
      kDim,
      layout.shard_members_[shard_id],
      space->get_dist_func(),
      {.max_degree_ = kMaxDegree, .ef_construction_ = 32, .num_iterations_ = 2, .max_memory_mb_ = 2});
  builder.build();

  auto export_path = prefix.parent_path() / "shard_1.graph";
  auto exported = read_exported_nodes(builder.export_graph(shard_id, export_path).graph_path_);

  for (size_t i = 1; i < exported.size(); ++i) {
    EXPECT_LT(exported[i - 1].global_id_, exported[i].global_id_);
  }
  for (const auto &node : exported) {
    for (size_t i = 1; i < node.neighbors_.size(); ++i) {
      EXPECT_LE(node.neighbors_[i - 1].distance_, node.neighbors_[i].distance_);
    }
  }
}

TEST_F(ShardVamanaBuilderTest, EstimatedPeakMemoryStaysWithinBudgetAndVectorsReleaseAfterExport) {
  auto data = make_vectors();
  auto space = make_space(data);
  auto prefix = make_temp_prefix("shard_memory");

  KMeansPartitioner<float> partitioner({.max_memory_mb_ = 1, .sample_rate_ = 0.02F, .overlap_factor_ = 2});
  auto layout = partitioner.partition(*space, kMaxDegree, prefix);

  auto shard_id = 2U;
  auto vectors = ShardVamanaBuilder<float>::load_vectors_from_shuffle(
      layout.shuffle_path_, layout.shuffle_offsets_[shard_id], layout.shuffle_counts_[shard_id], kDim);

  ShardVamanaBuilder<float> builder(
      std::move(vectors),
      kDim,
      layout.shard_members_[shard_id],
      space->get_dist_func(),
      {.max_degree_ = kMaxDegree, .ef_construction_ = 32, .num_iterations_ = 2, .max_memory_mb_ = 2});

  auto budget_bytes = static_cast<size_t>(0.9 * 2.0 * 1024.0 * 1024.0);
  EXPECT_LE(builder.estimated_peak_memory_bytes(), budget_bytes);

  builder.build();
  auto summary = builder.export_graph(shard_id, prefix.parent_path() / "shard_2.graph");
  EXPECT_LE(summary.estimated_peak_memory_bytes_, budget_bytes);
  EXPECT_FALSE(builder.vectors_loaded());
}

}  // namespace
}  // namespace alaya
