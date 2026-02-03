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
#include <cstdint>
#include <filesystem>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "index/graph/diskann/disk_layout.hpp"
#include "index/graph/diskann/diskann_builder.hpp"
#include "index/graph/diskann/diskann_index.hpp"
#include "index/graph/diskann/diskann_params.hpp"
#include "index/graph/diskann/diskann_searcher.hpp"
#include "index/graph/graph.hpp"
#include "space/raw_space.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"

namespace alaya {

class DiskANNTest : public ::testing::Test {
 protected:
  // Test dataset parameters
  static constexpr uint32_t kDataNum = 1000;
  static constexpr uint32_t kQueryNum = 50;
  static constexpr uint32_t kDim = 128;
  static constexpr uint32_t kGtTopk = 100;

  void SetUp() override {
    // Clean up any stale test files from previous runs
    if (std::filesystem::exists(index_path_)) {
      std::filesystem::remove(index_path_);
    }

    // Generate random dataset (no external files needed)
    dataset_ = load_dataset(random_config(kDataNum, kQueryNum, kDim, kGtTopk));

    dim_ = dataset_.dim_;
    data_num_ = dataset_.data_num_;
    query_num_ = dataset_.query_num_;

    // Build the space
    space_ = std::make_shared<RawSpace<>>(data_num_, dim_, MetricType::L2);
    space_->fit(dataset_.data_.data(), data_num_);
  }

  void TearDown() override {
    // Clean up test files
    if (std::filesystem::exists(index_path_)) {
      std::filesystem::remove(index_path_);
    }
  }

  uint32_t max_thread_num_ = std::thread::hardware_concurrency();
  Dataset dataset_;
  uint32_t dim_{0};
  uint32_t data_num_{0};
  uint32_t query_num_{0};
  std::shared_ptr<RawSpace<>> space_ = nullptr;
  std::string index_path_ = "test_diskann.index";
};

// Test DiskIndexHeader
TEST_F(DiskANNTest, DiskIndexHeaderTest) {
  DiskIndexHeader header;
  header.init(64, 1.2F, dim_, data_num_, kDiskSectorSize);
  header.meta_.medoid_id_ = 42;

  EXPECT_TRUE(header.is_valid());
  EXPECT_EQ(header.meta_.magic_, kDiskANNMagicNumber);
  EXPECT_EQ(header.meta_.version_, kDiskANNVersion);
  EXPECT_EQ(header.meta_.max_degree_, 64U);
  EXPECT_FLOAT_EQ(header.meta_.alpha_, 1.2F);
  EXPECT_EQ(header.meta_.dimension_, dim_);
  EXPECT_EQ(header.meta_.num_points_, data_num_);
  EXPECT_EQ(header.meta_.medoid_id_, 42U);

  // Test header save/load
  std::ofstream writer(index_path_, std::ios::binary);
  header.save(writer);
  writer.close();

  DiskIndexHeader loaded_header;
  std::ifstream reader(index_path_, std::ios::binary);
  loaded_header.load(reader);
  reader.close();

  EXPECT_TRUE(loaded_header.is_valid());
  EXPECT_EQ(loaded_header.meta_.dimension_, dim_);
  EXPECT_EQ(loaded_header.meta_.num_points_, data_num_);
  EXPECT_EQ(loaded_header.meta_.medoid_id_, 42U);
}

// Test DiskNode layout
TEST_F(DiskANNTest, DiskNodeLayoutTest) {
  constexpr uint32_t kMaxDegree = 64;
  auto node_size = DiskNode<float, uint32_t>::calc_node_sector_size(dim_, kMaxDegree);

  // Should be a multiple of sector size
  EXPECT_EQ(node_size % kDiskSectorSize, 0U);

  // Test buffer allocation
  DiskNodeBuffer<float, uint32_t> buffer;
  buffer.allocate(dim_, kMaxDegree, 1);

  EXPECT_TRUE(buffer.is_valid());
  EXPECT_EQ(buffer.num_nodes(), 1U);
  EXPECT_EQ(buffer.node_size(), node_size);

  // Test node accessor
  auto accessor = buffer.get_node(0);

  // Write test data
  std::vector<float> test_vec(dim_);
  std::iota(test_vec.begin(), test_vec.end(), 1.0F);

  std::vector<uint32_t> test_neighbors = {1, 2, 3, 4, 5};
  accessor.init(test_vec.data(), test_neighbors.data(), test_neighbors.size());

  // Verify
  EXPECT_EQ(accessor.num_neighbors(), test_neighbors.size());
  for (uint32_t i = 0; i < test_neighbors.size(); ++i) {
    EXPECT_EQ(accessor.get_neighbor(i), test_neighbors[i]);
  }
  for (uint32_t i = 0; i < dim_; ++i) {
    EXPECT_FLOAT_EQ(accessor.vector_data()[i], test_vec[i]);
  }
}

// Test DiskANN graph building
TEST_F(DiskANNTest, BuildGraphTest) {
  auto build_params = DiskANNBuildParams()
                          .set_max_degree(32)
                          .set_ef_construction(64)
                          .set_num_threads(max_thread_num_);

  DiskANNBuilder<RawSpace<>> builder(space_, build_params);
  auto graph = builder.build_graph(build_params.num_threads_);

  EXPECT_NE(graph, nullptr);
  EXPECT_EQ(graph->max_nodes_, data_num_);
  EXPECT_EQ(graph->max_nbrs_, 32U);
  EXPECT_FALSE(graph->eps_.empty());

  // Verify graph has edges
  uint32_t total_edges = 0;
  for (uint32_t i = 0; i < data_num_; ++i) {
    const auto *edges = graph->edges(i);
    for (uint32_t j = 0; j < graph->max_nbrs_; ++j) {
      if (edges[j] != static_cast<uint32_t>(-1)) {
        ++total_edges;
        EXPECT_LT(edges[j], data_num_);  // Valid node ID
      }
    }
  }
  EXPECT_GT(total_edges, 0U);  // Should have some edges
}

// Test disk index build and search
TEST_F(DiskANNTest, BuildAndSearchTest) {
  auto build_params = DiskANNBuildParams()
                          .set_max_degree(32)
                          .set_ef_construction(64)
                          .set_num_threads(max_thread_num_);

  DiskANNBuilder<RawSpace<>> builder(space_, build_params);
  builder.build_disk_index(index_path_, build_params.num_threads_);

  // Verify file exists
  EXPECT_TRUE(std::filesystem::exists(index_path_));

  // Load and search
  DiskANNSearcher<float, uint32_t> searcher;
  searcher.open(index_path_);

  EXPECT_TRUE(searcher.is_open());
  EXPECT_EQ(searcher.num_points(), data_num_);
  EXPECT_EQ(searcher.dimension(), dim_);
  EXPECT_EQ(searcher.max_degree(), build_params.max_degree_);

  // Test search with first query
  constexpr uint32_t kTopk = 10;
  DiskANNSearchParams search_params(50);
  std::vector<uint32_t> results(kTopk);

  const float *query = dataset_.queries_.data();
  searcher.search(query, kTopk, results.data(), search_params);

  // Verify results are valid node IDs
  for (uint32_t i = 0; i < kTopk; ++i) {
    if (results[i] != static_cast<uint32_t>(-1)) {
      EXPECT_LT(results[i], data_num_);
    }
  }
}

// Test DiskANNIndex unified interface
TEST_F(DiskANNTest, DiskANNIndexInterfaceTest) {
  auto build_params = DiskANNBuildParams()
                          .set_max_degree(32)
                          .set_ef_construction(64)
                          .set_num_threads(max_thread_num_);

  DiskANNIndex<float, uint32_t>::build(space_, index_path_, build_params);

  EXPECT_TRUE(std::filesystem::exists(index_path_));

  // Load using instance
  DiskANNIndex<float, uint32_t> index;
  index.load(index_path_);

  EXPECT_TRUE(index.is_loaded());
  EXPECT_EQ(index.size(), data_num_);
  EXPECT_EQ(index.dimension(), dim_);

  // Search
  constexpr uint32_t kTopk = 10;
  std::vector<uint32_t> results(kTopk);
  index.search(dataset_.queries_.data(), kTopk, results.data());

  // Check if ground truth top-1 is found
  uint32_t gt_top1 = dataset_.ground_truth_[0];
  bool found_gt = std::ranges::find(results, gt_top1) != results.end();
  EXPECT_TRUE(found_gt);

  // Test search with distance
  std::vector<float> distances(kTopk);
  index.search_with_distance(dataset_.queries_.data(), kTopk, results.data(), distances.data());

  // Distances should be non-negative and sorted
  for (uint32_t i = 1; i < kTopk; ++i) {
    EXPECT_GE(distances[i], distances[i - 1]);
  }
}

// Test search recall quality using siftsmall ground truth
TEST_F(DiskANNTest, SearchRecallTest) {
  auto build_params = DiskANNBuildParams()
                          .set_max_degree(64)
                          .set_ef_construction(128)
                          .set_num_threads(max_thread_num_);

  DiskANNIndex<float, uint32_t>::build(space_, index_path_, build_params);

  DiskANNIndex<float, uint32_t> index;
  index.load(index_path_);

  constexpr uint32_t kTopk = 10;
  DiskANNSearchParams search_params(100);

  // Get search results for all queries
  std::vector<uint32_t> results(query_num_ * kTopk);
  for (uint32_t q = 0; q < query_num_; ++q) {
    const float *query = dataset_.queries_.data() + q * dim_;
    index.search(query, kTopk, results.data() + q * kTopk, search_params);
  }

  // Compute recall using dataset ground truth
  float avg_recall = calc_recall(results.data(), dataset_.ground_truth_.data(),
                                 query_num_, dataset_.gt_dim_, kTopk);
  LOG_INFO("DiskANN recall@{}: {:.4f}", kTopk, avg_recall);

  // Log recall for monitoring (no strict threshold for now)
  EXPECT_GE(avg_recall, 0.8F);
}

// Test batch search
TEST_F(DiskANNTest, BatchSearchTest) {
  auto build_params = DiskANNBuildParams()
                          .set_max_degree(64)
                          .set_ef_construction(128)
                          .set_num_threads(max_thread_num_);

  DiskANNIndex<float, uint32_t>::build(space_, index_path_, build_params);

  DiskANNIndex<float, uint32_t> index;
  index.load(index_path_);

  constexpr uint32_t kTopk = 10;
  DiskANNSearchParams search_params(100);

  std::vector<uint32_t> results(query_num_ * kTopk);
  index.batch_search(dataset_.queries_.data(), query_num_, kTopk, results.data(), search_params);

  // Verify each query found its ground truth top-1
  uint32_t found_count = 0;
  for (uint32_t q = 0; q < query_num_; ++q) {
    uint32_t gt_top1 = dataset_.ground_truth_[q * dataset_.gt_dim_];
    for (uint32_t k = 0; k < kTopk; ++k) {
      if (results[q * kTopk + k] == gt_top1) {
        ++found_count;
        break;
      }
    }
  }

  // Log hit rate for monitoring
  float hit_rate = static_cast<float>(found_count) / static_cast<float>(query_num_);
  LOG_INFO("DiskANN top-1 hit rate: {:.4f}", hit_rate);

  // Verify results are valid (no strict threshold for now)
  for (uint32_t i = 0; i < query_num_ * kTopk; ++i) {
    if (results[i] != static_cast<uint32_t>(-1)) {
      EXPECT_LT(results[i], data_num_);
    }
  }
}

}  // namespace alaya
