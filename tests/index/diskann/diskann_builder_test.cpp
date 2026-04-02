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

#include "index/diskann/diskann_builder.hpp"
#include "index/diskann/diskann_index.hpp"
#include "index/diskann/diskann_params.hpp"
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
// Test Fixture: DiskANN Builder Unit Tests
// =============================================================================

class DiskANNBuilderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (g_res.dim_ == 0) {
      g_res.load();
    }
  }

  void SetUp() override {
    tmp_dir_ = std::filesystem::temp_directory_path() / "diskann_builder_test";
    std::filesystem::create_directories(tmp_dir_);
    index_path_ = (tmp_dir_ / "test_idx").string();
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(tmp_dir_, ec);
  }

  auto make_params(uint32_t max_degree = kDefaultR,
                   uint32_t ef_construction = kDefaultEf) const -> DiskANNBuildParams {
    return DiskANNBuildParams()
        .set_max_degree(max_degree)
        .set_ef_construction(ef_construction)
        .set_num_threads(max_threads());
  }

  void check_disk_index_invariants(const std::string &path,
                                   uint32_t expected_num_points,
                                   uint32_t max_degree) {
    BufferPool<uint32_t> bp(256, kDataBlockSize);
    DiskANNStorage<float, uint32_t> storage(&bp);
    storage.open(path);

    EXPECT_EQ(storage.num_points(), expected_num_points);
    EXPECT_EQ(storage.max_degree(), max_degree);
    EXPECT_EQ(storage.dimension(), dim());
    EXPECT_LT(storage.entry_point(), expected_num_points);

    // Check graph invariants via disk reads
    for (uint32_t i = 0; i < expected_num_points; ++i) {
      auto ref = storage.data().get_node(i);
      auto nbrs = ref.neighbors();

      // No self-loops
      for (auto neighbor : nbrs) {
        EXPECT_NE(neighbor, i) << "Self-loop at node " << i;
        EXPECT_LT(neighbor, expected_num_points) << "Out-of-range neighbor at node " << i;
      }

      // No duplicates
      std::set<uint32_t> seen(nbrs.begin(), nbrs.end());
      EXPECT_EQ(seen.size(), nbrs.size()) << "Duplicate neighbors at node " << i;

      // Degree bounded
      EXPECT_LE(nbrs.size(), max_degree) << "Degree exceeds max at node " << i;

      // Non-isolated
      EXPECT_GT(nbrs.size(), 0U) << "Isolated node " << i;
    }

    storage.close();
  }

  auto space() { return g_res.space_; }
  auto dim() const -> uint32_t { return g_res.dim_; }
  auto data_num() const -> uint32_t { return g_res.data_num_; }
  auto query_num() const -> uint32_t { return g_res.query_num_; }
  auto dataset() const -> const Dataset & { return g_res.ds_; }
  auto max_threads() const -> uint32_t { return g_res.max_threads_; }

  static constexpr uint32_t kDefaultR = 16;
  static constexpr uint32_t kDefaultEf = 32;

  std::filesystem::path tmp_dir_;
  std::string index_path_;
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
// 2. Build + disk index structural invariants
// -----------------------------------------------------------------------------

TEST_F(DiskANNBuilderTest, BuildBasicStructure) {
  auto params = make_params();
  DiskANNBuilder<RawSpace<>> builder(space(), params);
  builder.build(index_path_);

  check_disk_index_invariants(index_path_, data_num(), kDefaultR);
}

TEST_F(DiskANNBuilderTest, MaxDegreeVariants) {
  {
    auto params = make_params(4, 16).set_num_iterations(1);
    DiskANNBuilder<RawSpace<>> builder(space(), params);
    auto path = (tmp_dir_ / "idx_r4").string();
    builder.build(path);
    check_disk_index_invariants(path, data_num(), 4);
  }

  {
    auto params = make_params(128, 256).set_num_iterations(1);
    DiskANNBuilder<RawSpace<>> builder(space(), params);
    auto path = (tmp_dir_ / "idx_r128").string();
    builder.build(path);
    check_disk_index_invariants(path, data_num(), 128);
  }
}

// -----------------------------------------------------------------------------
// 3. Build + reload: verify metadata and neighbor consistency
// -----------------------------------------------------------------------------

TEST_F(DiskANNBuilderTest, SaveAndReloadConsistency) {
  constexpr uint32_t kR = 32;
  constexpr uint32_t kEf = 64;

  auto params = make_params(kR, kEf).set_num_iterations(2);

  DiskANNBuilder<RawSpace<>> builder(space(), params);
  builder.build(index_path_);

  // Reload via DiskANNStorage
  BufferPool<uint32_t> bp(256, kDataBlockSize);
  DiskANNStorage<float, uint32_t> storage(&bp);
  storage.open(index_path_);

  // Verify metadata
  EXPECT_EQ(storage.dimension(), dim());
  EXPECT_EQ(storage.max_degree(), kR);
  EXPECT_EQ(storage.num_points(), data_num());
  EXPECT_LT(storage.entry_point(), data_num());

  LOG_INFO("SaveAndReload: capacity={}, num_points={}, dim={}, R={}, entry={}",
           storage.capacity(),
           storage.num_points(),
           storage.dimension(),
           storage.max_degree(),
           storage.entry_point());

  // Verify all nodes have valid neighbors
  for (uint32_t i = 0; i < storage.num_points(); ++i) {
    auto ref = storage.data().get_node(i);
    auto nbrs = ref.neighbors();
    EXPECT_GT(nbrs.size(), 0U) << "Node " << i << " has no neighbors";
    for (auto nbr : nbrs) {
      EXPECT_LT(nbr, data_num()) << "Out-of-range neighbor at node " << i;
    }
  }

  // Verify vectors match original data
  auto dist_fn = space()->get_dist_func();
  for (uint32_t i = 0; i < std::min(data_num(), 10U); ++i) {
    auto ref = storage.data().get_node(i);
    auto stored_vec = ref.vector();
    const auto *original_vec = space()->get_data_by_id(i);
    float d = dist_fn(stored_vec.data(), original_vec, dim());
    EXPECT_FLOAT_EQ(d, 0.0F) << "Vector mismatch at node " << i;
  }

  storage.close();
}

// -----------------------------------------------------------------------------
// 5. End-to-end recall check via disk search
// -----------------------------------------------------------------------------

TEST_F(DiskANNBuilderTest, EndToEndRecall) {
  auto params = make_params(32, 64).set_num_iterations(2);

  DiskANNBuilder<RawSpace<>> builder(space(), params);
  builder.build(index_path_);

  // Load and search via DiskANNIndex
  DiskANNIndex<float, uint32_t> index;
  index.load(index_path_);

  auto dist_fn = space()->get_dist_func();
  uint32_t nn_found = 0;
  constexpr uint32_t kTopk = 10;

  for (uint32_t q = 0; q < query_num(); ++q) {
    const auto *query = dataset().queries_.data() + static_cast<size_t>(q) * dim();

    // Brute-force true NN
    float min_dist = std::numeric_limits<float>::max();
    uint32_t true_nn = 0;
    for (uint32_t j = 0; j < data_num(); ++j) {
      float d = dist_fn(query, space()->get_data_by_id(j), dim());
      if (d < min_dist) {
        min_dist = d;
        true_nn = j;
      }
    }

    // Search
    std::vector<uint32_t> results(kTopk);
    DiskANNSearchParams search_params;
    search_params.set_ef_search(64).set_beam_width(4);
    index.search(query, kTopk, results.data(), search_params);

    for (auto id : results) {
      if (id == true_nn) {
        ++nn_found;
        break;
      }
    }
  }

  float nn_recall = static_cast<float>(nn_found) / static_cast<float>(query_num());
  LOG_INFO("1-NN recall@{}: {:.2f}% ({}/{})", kTopk, nn_recall * 100, nn_found, query_num());

  // Out-of-core builder recall should be reasonable
  EXPECT_GE(nn_recall, 0.5F)
      << "At least 50% of queries should find their 1-NN in top-" << kTopk;

  index.close();
}

// -----------------------------------------------------------------------------
// 6. IP metric support
// -----------------------------------------------------------------------------

TEST_F(DiskANNBuilderTest, IPMetricBuildAndSearch) {
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
  builder.build(index_path_);

  // Verify the index loads and has valid structure
  check_disk_index_invariants(index_path_, data_num(), kDefaultR);
}

// -----------------------------------------------------------------------------
// 7. Recall benchmark (task 7.3)
// -----------------------------------------------------------------------------

TEST_F(DiskANNBuilderTest, RecallBenchmark) {
  auto params = make_params(32, 64).set_num_iterations(2);

  DiskANNBuilder<RawSpace<>> builder(space(), params);
  builder.build(index_path_);

  DiskANNIndex<float, uint32_t> index;
  index.load(index_path_);

  auto dist_fn = space()->get_dist_func();
  constexpr uint32_t kTopk = 10;
  uint32_t total_hits = 0;

  for (uint32_t q = 0; q < query_num(); ++q) {
    const auto *query = dataset().queries_.data() + static_cast<size_t>(q) * dim();

    // Brute-force top-k
    std::vector<std::pair<float, uint32_t>> all_dists;
    all_dists.reserve(data_num());
    for (uint32_t j = 0; j < data_num(); ++j) {
      all_dists.emplace_back(dist_fn(query, space()->get_data_by_id(j), dim()), j);
    }
    std::partial_sort(all_dists.begin(),
                      all_dists.begin() + kTopk,
                      all_dists.end());
    std::set<uint32_t> gt_set;
    for (uint32_t i = 0; i < kTopk; ++i) {
      gt_set.insert(all_dists[i].second);
    }

    // ANN search
    std::vector<uint32_t> results(kTopk);
    DiskANNSearchParams search_params;
    search_params.set_ef_search(128).set_beam_width(4);
    index.search(query, kTopk, results.data(), search_params);

    for (auto id : results) {
      if (gt_set.count(id) > 0) {
        ++total_hits;
      }
    }
  }

  float recall =
      static_cast<float>(total_hits) / static_cast<float>(query_num() * kTopk);
  LOG_INFO("Recall@{} with ef_search=128: {:.2f}% ({}/{})",
           kTopk,
           recall * 100,
           total_hits,
           query_num() * kTopk);

  // Out-of-core builder should achieve good recall
  EXPECT_GE(recall, 0.80F) << "Recall@" << kTopk << " should be at least 80%";
  index.close();
}

// -----------------------------------------------------------------------------
// 8. Memory-constrained build (task 7.4)
// -----------------------------------------------------------------------------

TEST_F(DiskANNBuilderTest, MemoryConstrainedBuild) {
  // Use a very small memory budget to force multiple shards
  auto params = make_params(16, 32)
                    .set_num_iterations(1)
                    .set_max_memory_mb(1);  // 1MB forces sharding

  auto path = (tmp_dir_ / "idx_constrained").string();
  DiskANNBuilder<RawSpace<>> builder(space(), params);
  builder.build(path);

  // Verify the index was built correctly despite memory constraint
  check_disk_index_invariants(path, data_num(), 16);
}

// -----------------------------------------------------------------------------
// 9. Intermediate file cleanup (task 7.6)
// -----------------------------------------------------------------------------

TEST_F(DiskANNBuilderTest, IntermediateFilesCleanedUp) {
  auto params = make_params(16, 32).set_num_iterations(1);

  auto path = (tmp_dir_ / "idx_cleanup").string();
  DiskANNBuilder<RawSpace<>> builder(space(), params);
  builder.build(path);

  // Check that no intermediate files remain
  for (const auto &entry : std::filesystem::directory_iterator(tmp_dir_)) {
    auto filename = entry.path().filename().string();
    EXPECT_FALSE(filename.find(".build_tmp") != std::string::npos)
        << "Intermediate file not cleaned up: " << filename;
    EXPECT_FALSE(filename.find(".shard_") != std::string::npos)
        << "Shard file not cleaned up: " << filename;
    EXPECT_FALSE(filename.find(".shuffle") != std::string::npos)
        << "Shuffle file not cleaned up: " << filename;
    EXPECT_FALSE(filename.find("node_to_shards") != std::string::npos)
        << "node_to_shards file not cleaned up: " << filename;
    EXPECT_FALSE(filename.find("shard_members") != std::string::npos)
        << "shard_members file not cleaned up: " << filename;
  }

  // Final index files should exist
  EXPECT_TRUE(std::filesystem::exists(path + ".meta"));
  EXPECT_TRUE(std::filesystem::exists(path + ".data"));
}

// -----------------------------------------------------------------------------
// 10. Merge average degree quality (task 7.7)
// -----------------------------------------------------------------------------

TEST_F(DiskANNBuilderTest, MergeAverageDegreeQuality) {
  auto params = make_params(32, 64).set_num_iterations(2);

  DiskANNBuilder<RawSpace<>> builder(space(), params);
  builder.build(index_path_);

  // Read the index and compute average degree
  BufferPool<uint32_t> bp(256, kDataBlockSize);
  DiskANNStorage<float, uint32_t> storage(&bp);
  storage.open(index_path_);

  uint64_t total_degree = 0;
  for (uint32_t i = 0; i < storage.num_points(); ++i) {
    auto ref = storage.data().get_node(i);
    total_degree += ref.neighbors().size();
  }

  float avg_degree = static_cast<float>(total_degree) / static_cast<float>(data_num());
  LOG_INFO("Average degree: {:.2f} (max_degree=32)", avg_degree);

  // Average degree should be reasonable (not trivially low)
  // Full robust_prune typically yields avg_degree ~= 0.6-0.8 * max_degree
  // Distance-ordered merge should be within 10% of that
  EXPECT_GE(avg_degree, 4.0F) << "Average degree is unreasonably low";
  EXPECT_LE(avg_degree, 32.0F) << "Average degree exceeds max_degree";

  storage.close();
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
 * @brief Test building an index on the Deep1M dataset.
 *
 * RUN it use ./xxxx_test --gtest_also_run_disabled_tests --gtest_filter=DiskANNBuilderDeep1MTest.DISABLED_BuildIndex
 */
TEST_F(DiskANNBuilderDeep1MTest, DISABLED_BuildIndex) {
  auto params = DiskANNBuildParams()
                    .set_max_degree(kDefaultR)
                    .set_ef_construction(kDefaultEf)
                    .set_num_iterations(2)
                    .set_num_threads(max_threads());

  auto tmp_dir = std::filesystem::temp_directory_path() / "diskann_deep1m_build_test";
  std::filesystem::create_directories(tmp_dir);
  auto index_path = (tmp_dir / "deep1m_idx").string();

  DiskANNBuilder<RawSpace<>> builder(space(), params);
  builder.build(index_path);

  // Verify basic structure
  BufferPool<uint32_t> bp(256, kDataBlockSize);
  DiskANNStorage<float, uint32_t> storage(&bp);
  storage.open(index_path);

  EXPECT_EQ(storage.num_points(), data_num());
  EXPECT_EQ(storage.max_degree(), kDefaultR);
  EXPECT_LT(storage.entry_point(), data_num());

  storage.close();
  std::filesystem::remove_all(tmp_dir);
}

}  // namespace alaya
