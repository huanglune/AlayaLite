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
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <iostream>
#include <memory>
#include <set>
#include <thread>
#include <vector>

#include "index/graph/diskann/diskann_builder.hpp"
#include "index/graph/diskann/diskann_params.hpp"
#include "index/graph/graph.hpp"
#include "space/raw_space.hpp"
#include "storage/buffer/buffer_pool.hpp"
#include "storage/diskann/diskann_storage.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/log.hpp"

namespace alaya {

// =============================================================================
// Shared test resources
// =============================================================================

struct BuilderTestResources {
  Dataset ds_;
  uint32_t dim_{0};
  uint32_t data_num_{0};
  uint32_t query_num_{0};
  std::shared_ptr<RawSpace<>> space_;
  uint32_t max_threads_{std::min(std::thread::hardware_concurrency(), 60U)};

  void load() {
    ds_ = load_dataset(random_config(kDataNum, kQueryNum, kDim, kGtTopk));
    dim_ = ds_.dim_;
    data_num_ = ds_.data_num_;
    query_num_ = ds_.query_num_;
    space_ = std::make_shared<RawSpace<>>(data_num_, dim_, MetricType::L2);
    space_->fit(ds_.data_.data(), data_num_);
    LOG_INFO("BuilderTestResources: Loaded {} vectors, dim={}", data_num_, dim_);
  }

 private:
  static constexpr uint32_t kDataNum = 500;
  static constexpr uint32_t kQueryNum = 20;
  static constexpr uint32_t kDim = 32;
  static constexpr uint32_t kGtTopk = 50;
};

struct BuilderDeep1MTestResources {
  Dataset ds_;
  uint32_t dim_{0};
  uint32_t data_num_{0};
  uint32_t query_num_{0};
  std::shared_ptr<RawSpace<>> space_;
  uint32_t max_threads_{std::min(std::thread::hardware_concurrency(), 60U)};

  void load() {
    ds_ = load_dataset(deep1m(data_dir_ / "deep1M"));
    dim_ = ds_.dim_;
    data_num_ = ds_.data_num_;
    query_num_ = ds_.query_num_;
    space_ = std::make_shared<RawSpace<>>(data_num_, dim_, MetricType::L2);
    space_->fit(ds_.data_.data(), data_num_);
    LOG_INFO("BuilderTestResources: Loaded {} vectors, dim={}", data_num_, dim_);
  }

 private:
  std::filesystem::path data_dir_ = std::filesystem::current_path().parent_path() / "data";
};

static BuilderTestResources g_res;
static BuilderDeep1MTestResources g_deep1m_res;

// =============================================================================
// Helper: count valid edges for a node
// =============================================================================

static auto count_edges(const Graph<float, uint32_t> *graph,
                        uint32_t node_id,
                        uint32_t max_degree) -> uint32_t {
  const auto *edges = graph->edges(node_id);
  return static_cast<uint32_t>(
      std::find(edges, edges + max_degree, static_cast<uint32_t>(-1)) - edges);
}

// =============================================================================
// Test Fixture: DiskANN Builder Unit Tests
// =============================================================================

class DiskANNBuilderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (g_res.dim_ == 0) {
      g_res.load();
    }
  }

  auto make_params(uint32_t max_degree = kDefaultR,
                   uint32_t ef_construction = kDefaultEf) const -> DiskANNBuildParams {
    return DiskANNBuildParams()
        .set_max_degree(max_degree)
        .set_ef_construction(ef_construction)
        .set_num_threads(max_threads());
  }

  void check_no_self_loops(const Graph<float, uint32_t> *graph, uint32_t max_degree) const {
    for (uint32_t i = 0; i < data_num(); ++i) {
      const auto *edges = graph->edges(i);
      for (uint32_t j = 0; j < max_degree; ++j) {
        if (edges[j] == static_cast<uint32_t>(-1)) {
          break;
        }
        EXPECT_NE(edges[j], i) << "Self-loop detected at node " << i;
      }
    }
  }

  void check_no_duplicate_edges(const Graph<float, uint32_t> *graph,
                                uint32_t max_degree) const {
    for (uint32_t i = 0; i < data_num(); ++i) {
      const auto *edges = graph->edges(i);
      std::set<uint32_t> seen;
      for (uint32_t j = 0; j < max_degree; ++j) {
        if (edges[j] == static_cast<uint32_t>(-1)) {
          break;
        }
        EXPECT_TRUE(seen.insert(edges[j]).second)
            << "Duplicate edge at node " << i << " neighbor " << edges[j];
      }
    }
  }

  void check_edges_in_range(const Graph<float, uint32_t> *graph, uint32_t max_degree) const {
    for (uint32_t i = 0; i < data_num(); ++i) {
      const auto *edges = graph->edges(i);
      for (uint32_t j = 0; j < max_degree; ++j) {
        if (edges[j] == static_cast<uint32_t>(-1)) {
          break;
        }
        EXPECT_LT(edges[j], data_num())
            << "Out-of-range edge at node " << i << ": " << edges[j];
      }
    }
  }

  void check_edges_padded_with_invalid_id(const Graph<float, uint32_t> *graph,
                                          uint32_t max_degree) const {
    for (uint32_t i = 0; i < data_num(); ++i) {
      const auto *edges = graph->edges(i);
      bool seen_invalid = false;
      for (uint32_t j = 0; j < max_degree; ++j) {
        if (edges[j] == static_cast<uint32_t>(-1)) {
          seen_invalid = true;
        } else if (seen_invalid) {
          FAIL() << "Valid edge after kInvalidID at node " << i << " position " << j;
        }
      }
    }
  }

  void check_no_isolated_nodes(const Graph<float, uint32_t> *graph,
                               uint32_t max_degree) const {
    uint32_t isolated_count = 0;
    for (uint32_t i = 0; i < data_num(); ++i) {
      if (count_edges(graph, i, max_degree) == 0) {
        ++isolated_count;
      }
    }
    EXPECT_EQ(isolated_count, 0U) << "Found isolated nodes with no edges";
  }

  auto space() { return g_res.space_; }
  auto dim() const -> uint32_t { return g_res.dim_; }
  auto data_num() const -> uint32_t { return g_res.data_num_; }
  auto query_num() const -> uint32_t { return g_res.query_num_; }
  auto dataset() const -> const Dataset & { return g_res.ds_; }
  auto max_threads() const -> uint32_t { return g_res.max_threads_; }

  static constexpr uint32_t kDefaultR = 16;
  static constexpr uint32_t kDefaultEf = 32;
};

// -----------------------------------------------------------------------------
// 1. Builder construction
// -----------------------------------------------------------------------------

TEST_F(DiskANNBuilderTest, ConstructWithDefaultParams) {
  DiskANNBuilder<RawSpace<>> builder(space());
  EXPECT_EQ(builder.dim_, dim());
  EXPECT_EQ(builder.params_.max_degree_, 64U);
  EXPECT_FLOAT_EQ(builder.params_.alpha_, 1.2F);
}

TEST_F(DiskANNBuilderTest, ConstructWithCustomParams) {
  auto params = DiskANNBuildParams()
                    .set_max_degree(32)
                    .set_alpha(1.5F)
                    .set_ef_construction(100)
                    .set_num_iterations(3);

  DiskANNBuilder<RawSpace<>> builder(space(), params);
  EXPECT_EQ(builder.params_.max_degree_, 32U);
  EXPECT_FLOAT_EQ(builder.params_.alpha_, 1.5F);
  EXPECT_EQ(builder.params_.ef_construction_, 100U);
  EXPECT_EQ(builder.params_.num_iterations_, 3U);
}

// -----------------------------------------------------------------------------
// 2. Graph build - structural invariants
// -----------------------------------------------------------------------------

TEST_F(DiskANNBuilderTest, BuildGraphBasicStructure) {
  auto params = make_params();

  DiskANNBuilder<RawSpace<>> builder(space(), params);
  auto graph = builder.build_graph(params.num_threads_);

  ASSERT_NE(graph, nullptr);
  EXPECT_EQ(graph->max_nodes_, data_num());
  EXPECT_EQ(graph->max_nbrs_, kDefaultR);
  EXPECT_FALSE(graph->eps_.empty());
  EXPECT_EQ(graph->eps_[0], builder.medoid_id_);

  EXPECT_LT(builder.medoid_id_, data_num());
  check_no_self_loops(graph.get(), kDefaultR);
  check_no_duplicate_edges(graph.get(), kDefaultR);
  check_edges_in_range(graph.get(), kDefaultR);
  check_edges_padded_with_invalid_id(graph.get(), kDefaultR);
  check_no_isolated_nodes(graph.get(), kDefaultR);
}

TEST_F(DiskANNBuilderTest, ParallelBuildStressKeepsGraphInvariants) {
  if (max_threads() < 2) {
    GTEST_SKIP() << "Parallel build test requires at least 2 threads";
  }

  constexpr int kStressRuns = 3;
  auto params = make_params().set_num_threads(std::min(max_threads(), 8U));
  for (int run = 0; run < kStressRuns; ++run) {
    DiskANNBuilder<RawSpace<>> builder(space(), params);
    auto graph = builder.build_graph(params.num_threads_);

    ASSERT_NE(graph, nullptr);
    check_no_self_loops(graph.get(), kDefaultR);
    check_no_duplicate_edges(graph.get(), kDefaultR);
    check_edges_in_range(graph.get(), kDefaultR);
    check_edges_padded_with_invalid_id(graph.get(), kDefaultR);
    check_no_isolated_nodes(graph.get(), kDefaultR);
  }
}

// -----------------------------------------------------------------------------
// 3. Different max_degree values
// -----------------------------------------------------------------------------

TEST_F(DiskANNBuilderTest, MaxDegreeVariants) {
  {
    auto params = make_params(4, 16).set_num_iterations(1);
    DiskANNBuilder<RawSpace<>> builder(space(), params);
    auto graph = builder.build_graph(params.num_threads_);

    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(graph->max_nbrs_, 4U);
    check_no_self_loops(graph.get(), 4);
    check_edges_in_range(graph.get(), 4);
  }

  {
    auto params = make_params(128, 256).set_num_iterations(1);
    DiskANNBuilder<RawSpace<>> builder(space(), params);
    auto graph = builder.build_graph(params.num_threads_);

    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(graph->max_nbrs_, 128U);

    uint64_t total = 0;
    for (uint32_t i = 0; i < data_num(); ++i) {
      total += count_edges(graph.get(), i, 128);
    }
    float avg = static_cast<float>(total) / static_cast<float>(data_num());
    EXPECT_GT(avg, 4.0F) << "Large R graph should have decent average degree";
    LOG_INFO("Large R (128) avg degree: {:.2f}", avg);
  }
}

// -----------------------------------------------------------------------------
// 11. PQ training
// -----------------------------------------------------------------------------

TEST_F(DiskANNBuilderTest, TrainPQ) {
  auto params = make_params().set_pq_params(4);  // 4 subspaces

  DiskANNBuilder<RawSpace<>> builder(space(), params);

  EXPECT_TRUE(params.is_pq_enabled());

  builder.train_pq();

  // PQ codes should be populated
  EXPECT_EQ(builder.pq_codes_.size(), static_cast<size_t>(data_num()) * 4);
}

// -----------------------------------------------------------------------------
// 12. Build graph end-to-end recall check
// -----------------------------------------------------------------------------

TEST_F(DiskANNBuilderTest, EndToEndRecall) {
  auto params = make_params(32, 64).set_num_iterations(2);

  DiskANNBuilder<RawSpace<>> builder(space(), params);
  auto graph = builder.build_graph(params.num_threads_);

  // Evaluate graph quality: for each data point, check if its nearest neighbor
  // (by brute force) appears in its edge list
  auto dist_fn = space()->get_dist_func();
  uint32_t nn_in_edges_count = 0;

  for (uint32_t i = 0; i < data_num(); ++i) {
    // Find true nearest neighbor
    float min_dist = std::numeric_limits<float>::max();
    uint32_t true_nn = i;
    for (uint32_t j = 0; j < data_num(); ++j) {
      if (i == j) {
        continue;
      }
      float d = dist_fn(space()->get_data_by_id(i), space()->get_data_by_id(j), dim());
      if (d < min_dist) {
        min_dist = d;
        true_nn = j;
      }
    }

    // Check if true NN is in edge list
    const auto *edges = graph->edges(i);
    for (uint32_t k = 0; k < 32; ++k) {
      if (edges[k] == static_cast<uint32_t>(-1)) {
        break;
      }
      if (edges[k] == true_nn) {
        ++nn_in_edges_count;
        break;
      }
    }
  }

  float nn_recall = static_cast<float>(nn_in_edges_count) / static_cast<float>(data_num());
  LOG_INFO("1-NN in edge list: {:.2f}% ({}/{})", nn_recall * 100, nn_in_edges_count, data_num());

  // A well-built Vamana graph should have most 1-NNs in their edge lists
  EXPECT_GE(nn_recall, 0.7F)
      << "At least 70% of nodes should have their 1-NN in the edge list";
}

// -----------------------------------------------------------------------------
// 13. Save to disk and reload: verify graph consistency
// -----------------------------------------------------------------------------

TEST_F(DiskANNBuilderTest, SaveAndReloadConsistency) {
  constexpr uint32_t kR = 32;
  constexpr uint32_t kEf = 64;

  auto params = make_params(kR, kEf).set_num_iterations(2);

  DiskANNBuilder<RawSpace<>> builder(space(), params);
  auto graph = builder.build_graph(params.num_threads_);
  std::vector<std::vector<uint32_t>> origin_nbrs(graph->max_nodes_);
  for (uint32_t i = 0; i < graph->max_nodes_; ++i) {
    const auto *edges = graph->edges(i);
    origin_nbrs[i] = std::vector<uint32_t>(edges, edges + kR);
  }
  std::string nbrs_str;
  for (uint32_t j = 0; j < std::min<uint32_t>(3, kR); ++j) {
    nbrs_str += std::to_string(origin_nbrs[0][j]) + " ";
  }
  LOG_INFO("[DEBUG]: origin_nbrs 0: {}", nbrs_str);

  ASSERT_NE(graph, nullptr);
  // Save to disk
  auto tmp_dir = std::filesystem::temp_directory_path() / "diskann_save_reload_test";
  std::filesystem::create_directories(tmp_dir);
  auto index_path = (tmp_dir / "test_idx").string();

  builder.save_disk_index(index_path, *graph);

  // Reload via DiskANNStorage
  BufferPool<uint32_t> bp(256, kDataBlockSize);
  DiskANNStorage<float, uint32_t> storage(&bp);
  storage.open(index_path);

  // Verify metadata
  EXPECT_EQ(storage.dimension(), dim());
  EXPECT_EQ(storage.max_degree(), kR);
  EXPECT_EQ(storage.entry_point(), static_cast<uint32_t>(builder.medoid_id_));
  EXPECT_EQ(storage.num_points(), data_num());

  LOG_INFO("SaveAndReload: capacity={}, num_points={}, dim={}, R={}, entry={}",
           storage.capacity(),
           storage.num_points(),
           storage.dimension(),
           storage.max_degree(),
           storage.entry_point());

  // neighbor consistency check
  size_t cnt = 0;
  for (uint32_t i = 0; i < storage.num_points(); ++i) {
    auto ref = storage.data().get_node(i);
    auto nbrs = ref.neighbors();
    for (size_t j = 0; j < kR; ++j) {
      EXPECT_EQ(nbrs[j], origin_nbrs[i][j]) << "Mismatch at node " << i << " neighbor " << j;
      if (nbrs[j] != origin_nbrs[i][j]) {
        LOG_INFO("total correct {}", cnt);
        return;  // Early exit on mismatch to avoid spamming logs
      }
      cnt += 1;
    }
  }

  storage.close();
  std::filesystem::remove_all(tmp_dir);
}

class DiskANNBuilderDeep1MTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (g_deep1m_res.dim_ == 0) {
      g_deep1m_res.load();
    }
  }

  auto space() { return g_deep1m_res.space_; }
  auto dim() const -> uint32_t { return g_deep1m_res.dim_; }
  auto data_num() const -> uint32_t { return g_deep1m_res.data_num_; }
  auto query_num() const -> uint32_t { return g_deep1m_res.query_num_; }
  auto dataset() const -> const Dataset & { return g_deep1m_res.ds_; }
  auto max_threads() const -> uint32_t { return g_deep1m_res.max_threads_; }

  static constexpr uint32_t kDefaultR = 64;
  static constexpr uint32_t kDefaultEf = 128;
};


/**
 * @brief Test building a graph on the Deep1M dataset.
 *
 * RUN it use ./xxxx_test --gtest_also_run_disabled_tests --gtest_filter=DiskANNBuilderDeep1MTest.DISABLED_BuildGraph
 */
TEST_F(DiskANNBuilderDeep1MTest, DISABLED_BuildGraph) {
  auto params = DiskANNBuildParams()
                    .set_max_degree(kDefaultR)
                    .set_ef_construction(kDefaultEf)
                    .set_num_iterations(2)
                    .set_num_threads(max_threads());

  DiskANNBuilder<RawSpace<>> builder(space(), params);
  auto graph = builder.build_graph(params.num_threads_);

  ASSERT_NE(graph, nullptr);
  EXPECT_EQ(graph->max_nodes_, data_num());
  EXPECT_EQ(graph->max_nbrs_, kDefaultR);
  EXPECT_FALSE(graph->eps_.empty());
  EXPECT_EQ(graph->eps_[0], builder.medoid_id_);

  std::vector<bool> visited(data_num(), false);
    std::vector<uint32_t> q;
    q.push_back(graph->eps_[0]);
    visited[graph->eps_[0]] = true;
    size_t head = 0;
    size_t visited_count = 0;

    while(head < q.size() && visited_count < data_num()) {
        uint32_t u = q[head++];
        visited_count++;
        const auto* edges = graph->edges(u);
        for(uint32_t i=0; i<params.max_degree_; ++i) {
            if (edges[i] == DiskANNBuilder<RawSpace<>>::kInvalidID) { break;
}
            if (!visited[edges[i]]) {
                visited[edges[i]] = true;
                q.push_back(edges[i]);
            }
        }
    }
    std::cout << "[CHECK] BFS Connectivity from Medoid: " << visited_count << "/" << data_num() << '\n';
    EXPECT_GT(visited_count, data_num() * 0.9);
}

TEST_F(DiskANNBuilderTest, IPMetricRobustPruneBuildGraphRecallPositive) {
  std::vector<float> normalized_data(dataset().data_.begin(), dataset().data_.end());
  for (uint32_t i = 0; i < data_num(); ++i) {
    math::normalize(normalized_data.data() + i * dim(), dim());
  }

  auto ip_space = std::make_shared<RawSpace<float>>(data_num(), dim(), MetricType::IP);
  ip_space->fit(normalized_data.data(), data_num());

  auto params = DiskANNBuildParams()
                    .set_max_degree(kDefaultR)
                    .set_ef_construction(kDefaultEf)
                    .set_num_iterations(2)
                    .set_num_threads(max_threads());

  DiskANNBuilder<RawSpace<float>> builder(ip_space, params);
  // build_graph() exercises robust_prune during neighbor selection
  // and reverse-edge updates.
  auto graph = builder.build_graph(params.num_threads_);
  ASSERT_NE(graph, nullptr);

  auto dist_fn = ip_space->get_dist_func();
  uint32_t hits = 0;
  for (uint32_t i = 0; i < data_num(); ++i) {
    float best_dist = std::numeric_limits<float>::max();
    uint32_t true_nn = i;
    for (uint32_t j = 0; j < data_num(); ++j) {
      if (i == j) {
        continue;
      }
      float dist = dist_fn(ip_space->get_data_by_id(i), ip_space->get_data_by_id(j), dim());
      if (dist < best_dist) {
        best_dist = dist;
        true_nn = j;
      }
    }

    const auto *edges = graph->edges(i);
    for (uint32_t k = 0; k < kDefaultR; ++k) {
      if (edges[k] == DiskANNBuilder<RawSpace<float>>::kInvalidID) {
        break;
      }
      if (edges[k] == true_nn) {
        ++hits;
        break;
      }
    }
  }

  float recall = static_cast<float>(hits) / static_cast<float>(data_num());
  EXPECT_GT(recall, 0.0F);
}

}  // namespace alaya
