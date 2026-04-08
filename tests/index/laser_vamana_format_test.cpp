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
#include <vector>

#include "index/diskann/cross_shard_merger.hpp"
#include "index/laser/laser_builder.hpp"
#include "index/laser/qg_builder.hpp"

namespace alaya {
namespace {

auto make_temp_path(const char *label) -> std::filesystem::path {
  auto suffix = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  auto dir = std::filesystem::temp_directory_path() / ("laser_vamana_format_test_" + suffix);
  std::filesystem::create_directories(dir);
  return dir / label;
}

TEST(VamanaFormatWriterTest, WritesGraphQGBuilderCanParse) {
  constexpr uint32_t kNumPoints = 4;
  constexpr uint32_t kDegree = 3;
  auto vamana_path = make_temp_path("graph.vamana");

  std::vector<CrossShardMerger::MergedNode> nodes = {
      {0, {1, 2}},
      {1, {0, 2, 3}},
      {2, {1, 3}},
      {3, {1}},
  };

  VamanaFormatWriter writer(vamana_path, kDegree, kNumPoints);
  writer.open();
  for (const auto &node : nodes) {
    writer.write_node(node);
  }
  writer.finalize();

  std::ifstream input(vamana_path, std::ios::binary);
  ASSERT_TRUE(input.is_open());
  size_t file_size = 0;
  uint32_t degree_bound = 0;
  uint32_t entry_point = 0;
  size_t frozen_points = 0;
  input.read(reinterpret_cast<char *>(&file_size), sizeof(file_size));
  input.read(reinterpret_cast<char *>(&degree_bound), sizeof(degree_bound));
  input.read(reinterpret_cast<char *>(&entry_point), sizeof(entry_point));
  input.read(reinterpret_cast<char *>(&frozen_points), sizeof(frozen_points));

  EXPECT_EQ(file_size, std::filesystem::file_size(vamana_path));
  EXPECT_EQ(degree_bound, kDegree);
  EXPECT_EQ(entry_point, 1U);
  EXPECT_EQ(frozen_points, 0U);

  uint32_t nodes_read = 0;
  uint32_t max_observed_degree = 0;
  while (input) {
    uint32_t node_degree = 0;
    input.read(reinterpret_cast<char *>(&node_degree), sizeof(node_degree));
    if (!input) {
      break;
    }
    std::vector<uint32_t> neighbors(node_degree);
    input.read(reinterpret_cast<char *>(neighbors.data()),
               static_cast<std::streamsize>(neighbors.size() * sizeof(uint32_t)));
    max_observed_degree = std::max(max_observed_degree, node_degree);
    ++nodes_read;
  }

  EXPECT_EQ(nodes_read, kNumPoints);
  EXPECT_LE(max_observed_degree, kDegree);

  symqg::QuantizedGraph graph(kNumPoints, kDegree, 64, 128);
  symqg::QGBuilder builder(graph, 32, 1);
  builder.init_from_vamana(vamana_path.string());
  EXPECT_EQ(graph.entry_point(), 1U);
}

}  // namespace
}  // namespace alaya
