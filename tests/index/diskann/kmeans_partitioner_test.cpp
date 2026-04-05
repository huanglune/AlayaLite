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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "index/diskann/kmeans_partitioner.hpp"
#include "space/raw_space.hpp"

namespace alaya {
namespace {

auto make_partition_vectors(uint32_t large_cluster_points,
                            uint32_t side_cluster_points,
                            uint32_t dim) -> std::vector<float> {
  std::vector<float> data(static_cast<size_t>(large_cluster_points + 2 * side_cluster_points) * dim);
  std::mt19937 rng(7);
  std::normal_distribution<float> noise(0.0F, 0.08F);

  auto write_cluster = [&](uint32_t start_idx, uint32_t count, float cx, float cy) {
    for (uint32_t i = 0; i < count; ++i) {
      auto base = static_cast<size_t>(start_idx + i) * dim;
      data[base] = cx + noise(rng);
      data[base + 1] = cy + noise(rng);
      for (uint32_t d = 2; d < dim; ++d) {
        data[base + d] = static_cast<float>(d) * 0.01F + noise(rng) * 0.1F;
      }
    }
  };

  write_cluster(0, large_cluster_points, 0.0F, 0.0F);
  write_cluster(large_cluster_points, side_cluster_points, 10.0F, 0.0F);
  write_cluster(large_cluster_points + side_cluster_points, side_cluster_points, 0.0F, 10.0F);
  return data;
}

auto make_shuffle_vectors(uint32_t num_points, uint32_t dim) -> std::vector<float> {
  std::vector<float> data(static_cast<size_t>(num_points) * dim);
  for (uint32_t id = 0; id < num_points; ++id) {
    for (uint32_t d = 0; d < dim; ++d) {
      data[static_cast<size_t>(id) * dim + d] = static_cast<float>(id * 100 + d);
    }
  }
  return data;
}

auto make_temp_prefix(const char *label) -> std::filesystem::path {
  auto suffix = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  auto dir = std::filesystem::temp_directory_path() / ("kmeans_partitioner_test_" + suffix);
  std::filesystem::create_directories(dir);
  return dir / label;
}

class KMeansPartitionerTest : public ::testing::Test {
 protected:
  static constexpr uint32_t kDim = 8;
  static constexpr uint32_t kMaxDegree = 16;

  static auto make_space(const std::vector<float> &data) -> std::shared_ptr<RawSpace<>> {
    auto count = static_cast<uint32_t>(data.size() / kDim);
    auto space = std::make_shared<RawSpace<>>(count, kDim, MetricType::L2);
    space->fit(data.data(), count);
    return space;
  }
};

TEST_F(KMeansPartitionerTest, OverlappingAssignmentPreservesPrimaryAndPersistsMappings) {
  auto data = make_partition_vectors(8000, 2000, kDim);
  auto space = make_space(data);
  auto output_prefix = make_temp_prefix("assignment");

  KMeansPartitioner<float> partitioner({
      .max_memory_mb_ = 1,
      .sample_rate_ = 0.02F,
      .overlap_factor_ = 2,
      .shard_overflow_factor_ = 1.2F,
  });

  auto layout = partitioner.partition(*space, kMaxDegree, output_prefix);

  ASSERT_EQ(layout.num_shards_, 5U);
  uint32_t exact_overlap = 0;
  uint32_t reduced_overlap = 0;
  for (uint32_t node_id = 0; node_id < layout.num_nodes_; ++node_id) {
    auto assignment_count = layout.assignment_count(node_id);
    EXPECT_GE(assignment_count, 1U);
    EXPECT_LE(assignment_count, layout.max_assignments_);
    EXPECT_TRUE(layout.has_assignment(node_id, layout.primary_shards_[node_id]));
    if (assignment_count == layout.max_assignments_) {
      ++exact_overlap;
    } else {
      ++reduced_overlap;
    }
  }

  EXPECT_GT(exact_overlap, layout.num_nodes_ / 2);
  EXPECT_GT(reduced_overlap, 0U);

  auto persisted_assignments = KMeansPartitioner<float>::load_node_to_shards(layout.node_to_shards_path_);
  auto persisted_members = KMeansPartitioner<float>::load_shard_members(layout.shard_members_path_);

  EXPECT_EQ(persisted_assignments.num_nodes_, layout.num_nodes_);
  EXPECT_EQ(persisted_assignments.num_shards_, layout.num_shards_);
  EXPECT_EQ(persisted_assignments.max_assignments_, layout.max_assignments_);
  EXPECT_EQ(persisted_assignments.assignments_, layout.node_to_shards_);
  EXPECT_EQ(persisted_members.num_shards_, layout.num_shards_);
  EXPECT_EQ(persisted_members.members_, layout.shard_members_);
}

TEST_F(KMeansPartitionerTest, ShardBalancingHonorsCap) {
  auto data = make_partition_vectors(8000, 2000, kDim);
  auto space = make_space(data);
  auto output_prefix = make_temp_prefix("balancing");

  KMeansPartitioner<float> partitioner({
      .max_memory_mb_ = 1,
      .sample_rate_ = 0.02F,
      .overlap_factor_ = 2,
      .shard_overflow_factor_ = 1.2F,
  });

  auto layout = partitioner.partition(*space, kMaxDegree, output_prefix);

  for (const auto &members : layout.shard_members_) {
    EXPECT_LE(members.size(), layout.shard_size_cap_);
  }

  uint32_t reduced_overlap = 0;
  for (uint32_t node_id = 0; node_id < layout.num_nodes_; ++node_id) {
    if (layout.assignment_count(node_id) < layout.max_assignments_) {
      ++reduced_overlap;
    }
  }
  EXPECT_GT(reduced_overlap, 0U);
}

TEST_F(KMeansPartitionerTest, ShuffleProducesContiguousShardDataAndPreservesVectors) {
  constexpr uint32_t kNumPoints = 12000;
  auto data = make_shuffle_vectors(kNumPoints, kDim);
  auto space = make_space(data);
  auto output_prefix = make_temp_prefix("shuffle");

  KMeansPartitioner<float> partitioner({
      .max_memory_mb_ = 1,
      .sample_rate_ = 0.02F,
      .overlap_factor_ = 2,
      .shard_overflow_factor_ = 1.2F,
  });

  auto layout = partitioner.partition(*space, kMaxDegree, output_prefix);
  auto row_bytes = static_cast<uint64_t>(kDim) * sizeof(float);

  std::ifstream input(layout.shuffle_path_, std::ios::binary);
  ASSERT_TRUE(input.is_open());

  uint64_t expected_offset = 0;
  for (uint32_t shard = 0; shard < layout.num_shards_; ++shard) {
    EXPECT_EQ(layout.shuffle_offsets_[shard], expected_offset);
    EXPECT_EQ(layout.shuffle_counts_[shard], layout.shard_members_[shard].size());

    for (size_t idx = 0; idx < layout.shard_members_[shard].size(); ++idx) {
      auto node_id = layout.shard_members_[shard][idx];
      std::vector<float> actual(kDim);
      input.seekg(static_cast<std::streamoff>(layout.shuffle_offsets_[shard] + idx * row_bytes));
      input.read(reinterpret_cast<char *>(actual.data()), static_cast<std::streamsize>(row_bytes));

      const auto *expected = space->get_data_by_id(node_id);
      for (uint32_t d = 0; d < kDim; ++d) {
        EXPECT_FLOAT_EQ(actual[d], expected[d]);
      }
    }

    expected_offset += layout.shuffle_counts_[shard] * row_bytes;
  }
}

}  // namespace
}  // namespace alaya
