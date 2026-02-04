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
#include "storage/buffer/buffer_pool.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"
#include "utils/log.hpp"

namespace alaya {

// =============================================================================
// Shared test resources (loaded once for all tests)
// =============================================================================

struct TestResources {
  Dataset ds_;
  uint32_t dim_{0};
  uint32_t data_num_{0};
  uint32_t query_num_{0};

  std::shared_ptr<RawSpace<>> space_;
  uint32_t max_threads_{std::thread::hardware_concurrency()};

  void load() {
    ds_ = load_dataset(random_config(kDataNum, kQueryNum, kDim, kGtTopk));
    dim_ = ds_.dim_;
    data_num_ = ds_.data_num_;
    query_num_ = ds_.query_num_;
    space_ = std::make_shared<RawSpace<>>(data_num_, dim_, MetricType::L2);
    space_->fit(ds_.data_.data(), data_num_);
    LOG_INFO("TestResources: Loaded {} vectors, dim={}, {} queries", data_num_, dim_, query_num_);
  }

private:
  static constexpr uint32_t kDataNum = 1000;
  static constexpr uint32_t kQueryNum = 50;
  static constexpr uint32_t kDim = 128;
  static constexpr uint32_t kGtTopk = 100;
};

static TestResources g_resources;

// =============================================================================
// Test Fixture 1: Layout tests (no index building, fast)
// =============================================================================

class DiskLayoutTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (g_resources.dim_ == 0) {
      g_resources.load();
    }
  }

  auto dim() const -> uint32_t { return g_resources.dim_; }
  auto data_num() const -> uint32_t { return g_resources.data_num_; }
};

TEST_F(DiskLayoutTest, HeaderInitAndValidate) {
  DiskIndexHeader header;
  header.init(64, 1.2F, dim(), data_num(), kDiskSectorSize);
  header.meta_.medoid_id_ = 42;

  EXPECT_TRUE(header.is_valid());
  EXPECT_EQ(header.meta_.magic_, kDiskANNMagicNumber);
  EXPECT_EQ(header.meta_.version_, kDiskANNVersion);
  EXPECT_EQ(header.meta_.max_degree_, 64U);
  EXPECT_FLOAT_EQ(header.meta_.alpha_, 1.2F);
  EXPECT_EQ(header.meta_.dimension_, dim());
  EXPECT_EQ(header.meta_.num_points_, data_num());
  EXPECT_EQ(header.meta_.medoid_id_, 42U);
}

TEST_F(DiskLayoutTest, HeaderSaveLoad) {
  const std::string kTestFile = "/tmp/diskann_header_test.bin";

  DiskIndexHeader header;
  header.init(64, 1.2F, dim(), data_num(), kDiskSectorSize);
  header.meta_.medoid_id_ = 42;

  std::ofstream writer(kTestFile, std::ios::binary);
  header.save(writer);
  writer.close();

  DiskIndexHeader loaded;
  std::ifstream reader(kTestFile, std::ios::binary);
  loaded.load(reader);
  reader.close();

  EXPECT_TRUE(loaded.is_valid());
  EXPECT_EQ(loaded.meta_.dimension_, dim());
  EXPECT_EQ(loaded.meta_.num_points_, data_num());
  EXPECT_EQ(loaded.meta_.medoid_id_, 42U);

  std::filesystem::remove(kTestFile);
}

TEST_F(DiskLayoutTest, HeaderPQInit) {
  DiskIndexHeader header;
  header.init(64, 1.2F, dim(), data_num(), kDiskSectorSize);
  header.init_pq(8);  // 8 subspaces

  EXPECT_TRUE(header.is_pq_enabled());
  EXPECT_EQ(header.meta_.pq_num_subspaces_, 8U);
  EXPECT_GT(header.meta_.pq_codebook_offset_, 0U);
  EXPECT_GT(header.meta_.pq_codes_offset_, header.meta_.pq_codebook_offset_);
  EXPECT_GT(header.meta_.nodes_offset_, header.meta_.pq_codes_offset_);
}

TEST_F(DiskLayoutTest, NodeSectorSize) {
  constexpr uint32_t kMaxDegree = 64;
  auto node_size = DiskNode<float, uint32_t>::calc_node_sector_size(dim(), kMaxDegree);

  // Should be a multiple of sector size
  EXPECT_EQ(node_size % kDiskSectorSize, 0U);
  EXPECT_GT(node_size, 0U);
}

TEST_F(DiskLayoutTest, NodeBufferAllocation) {
  constexpr uint32_t kMaxDegree = 64;
  auto node_size = DiskNode<float, uint32_t>::calc_node_sector_size(dim(), kMaxDegree);

  DiskNodeBuffer<float, uint32_t> buffer;
  buffer.allocate(dim(), kMaxDegree, 1);

  EXPECT_TRUE(buffer.is_valid());
  EXPECT_EQ(buffer.num_nodes(), 1U);
  EXPECT_EQ(buffer.node_size(), node_size);
}

TEST_F(DiskLayoutTest, NodeAccessor) {
  constexpr uint32_t kMaxDegree = 64;

  DiskNodeBuffer<float, uint32_t> buffer;
  buffer.allocate(dim(), kMaxDegree, 1);
  auto accessor = buffer.get_node(0);

  // Write test data
  std::vector<float> test_vec(dim());
  std::iota(test_vec.begin(), test_vec.end(), 1.0F);
  std::vector<uint32_t> test_neighbors = {1, 2, 3, 4, 5};
  accessor.init(test_vec.data(), test_neighbors.data(), test_neighbors.size());

  // Verify
  EXPECT_EQ(accessor.num_neighbors(), test_neighbors.size());
  for (uint32_t i = 0; i < test_neighbors.size(); ++i) {
    EXPECT_EQ(accessor.get_neighbor(i), test_neighbors[i]);
  }
  for (uint32_t i = 0; i < dim(); ++i) {
    EXPECT_FLOAT_EQ(accessor.vector_data()[i], test_vec[i]);
  }
}

// =============================================================================
// Test Fixture 2: Build tests (test graph construction)
// =============================================================================

class DiskANNBuildTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (g_resources.dim_ == 0) {
      g_resources.load();
    }
  }

  auto space() { return g_resources.space_; }
  auto dim() const -> uint32_t { return g_resources.dim_; }
  auto data_num() const -> uint32_t { return g_resources.data_num_; }
  auto max_threads() const -> uint32_t { return g_resources.max_threads_; }
};

TEST_F(DiskANNBuildTest, BuildGraph) {
  auto params = DiskANNBuildParams().set_max_degree(32).set_ef_construction(64).set_num_threads(
      max_threads());

  DiskANNBuilder<RawSpace<>> builder(space(), params);
  auto graph = builder.build_graph(params.num_threads_);

  EXPECT_NE(graph, nullptr);
  EXPECT_EQ(graph->max_nodes_, data_num());
  EXPECT_EQ(graph->max_nbrs_, 32U);
  EXPECT_FALSE(graph->eps_.empty());

  // Verify graph has edges
  uint32_t total_edges = 0;
  for (uint32_t i = 0; i < data_num(); ++i) {
    const auto *edges = graph->edges(i);
    for (uint32_t j = 0; j < graph->max_nbrs_; ++j) {
      if (edges[j] != static_cast<uint32_t>(-1)) {
        ++total_edges;
        EXPECT_LT(edges[j], data_num());
      }
    }
  }
  EXPECT_GT(total_edges, 0U);
}

// =============================================================================
// Test Fixture 3: Search tests with shared index (build once)
// =============================================================================

class DiskANNSearchTest : public ::testing::Test {
 protected:
  static inline std::string index_path = "/tmp/diskann_search_test.index";
  static inline bool index_built = false;

  static void SetUpTestSuite() {
    if (g_resources.dim_ == 0) {
      g_resources.load();
    }

    if (!index_built) {
      auto params = DiskANNBuildParams()
                        .set_max_degree(64)
                        .set_ef_construction(128)
                        .set_num_threads(g_resources.max_threads_);

      DiskANNIndex<float, uint32_t>::build(g_resources.space_, index_path, params);
      index_built = true;
      LOG_INFO("DiskANNSearchTest: Built shared index at {}", index_path);
    }
  }

  static void TearDownTestSuite() {
    if (std::filesystem::exists(index_path)) {
      std::filesystem::remove(index_path);
    }
  }

  auto dataset() const -> const Dataset & { return g_resources.ds_; }
  auto dim() const -> uint32_t { return g_resources.dim_; }
  auto data_num() const -> uint32_t { return g_resources.data_num_; }
  auto query_num() const -> uint32_t { return g_resources.query_num_; }
  auto get_index_path() const -> const std::string & { return index_path; }
};

TEST_F(DiskANNSearchTest, LoadIndex) {
  DiskANNIndex<float, uint32_t> index;
  index.load(get_index_path());

  EXPECT_TRUE(index.is_loaded());
  EXPECT_EQ(index.size(), data_num());
  EXPECT_EQ(index.dimension(), dim());
}

TEST_F(DiskANNSearchTest, SearchSingle) {
  DiskANNIndex<float, uint32_t> index;
  index.load(get_index_path());

  constexpr uint32_t kTopk = 10;
  DiskANNSearchParams params(100);
  std::vector<uint32_t> results(kTopk);

  index.search(dataset().queries_.data(), kTopk, results.data(), params);

  // Verify results are valid node IDs
  for (uint32_t i = 0; i < kTopk; ++i) {
    if (results[i] != static_cast<uint32_t>(-1)) {
      EXPECT_LT(results[i], data_num());
    }
  }
}

TEST_F(DiskANNSearchTest, SearchWithDistance) {
  DiskANNIndex<float, uint32_t> index;
  index.load(get_index_path());

  constexpr uint32_t kTopk = 10;
  DiskANNSearchParams params(100);
  std::vector<uint32_t> results(kTopk);
  std::vector<float> distances(kTopk);

  index.search_with_distance(dataset().queries_.data(), kTopk, results.data(), distances.data(),
                             params);

  // Distances should be non-negative and sorted
  for (uint32_t i = 1; i < kTopk; ++i) {
    EXPECT_GE(distances[i], distances[i - 1]);
  }
}

TEST_F(DiskANNSearchTest, BatchSearch) {
  DiskANNIndex<float, uint32_t> index;
  index.load(get_index_path());

  constexpr uint32_t kTopk = 10;
  DiskANNSearchParams params(100);
  std::vector<uint32_t> results(query_num() * kTopk);

  index.batch_search(dataset().queries_.data(), query_num(), kTopk, results.data(), params);

  // Verify results are valid
  for (uint32_t i = 0; i < query_num() * kTopk; ++i) {
    if (results[i] != static_cast<uint32_t>(-1)) {
      EXPECT_LT(results[i], data_num());
    }
  }
}

TEST_F(DiskANNSearchTest, RecallQuality) {
  DiskANNIndex<float, uint32_t> index;
  index.load(get_index_path());

  constexpr uint32_t kTopk = 10;
  DiskANNSearchParams params(200);  // Higher ef for better recall

  std::vector<uint32_t> results(query_num() * kTopk);
  for (uint32_t q = 0; q < query_num(); ++q) {
    const float *query = dataset().queries_.data() + q * dim();
    index.search(query, kTopk, results.data() + q * kTopk, params);
  }

  float recall =
      calc_recall(results.data(), dataset().ground_truth_.data(), query_num(), dataset().gt_dim_, kTopk);
  LOG_INFO("DiskANN recall@{}: {:.4f}", kTopk, recall);

  EXPECT_GE(recall, 0.7F);
}

TEST_F(DiskANNSearchTest, HitRate) {
  DiskANNIndex<float, uint32_t> index;
  index.load(get_index_path());

  constexpr uint32_t kTopk = 10;
  DiskANNSearchParams params(100);

  uint32_t hit_count = 0;
  for (uint32_t q = 0; q < query_num(); ++q) {
    std::vector<uint32_t> results(kTopk);
    index.search(dataset().queries_.data() + q * dim(), kTopk, results.data(), params);

    uint32_t gt_top1 = dataset().ground_truth_[q * dataset().gt_dim_];
    if (std::ranges::find(results, gt_top1) != results.end()) {
      ++hit_count;
    }
  }

  float hit_rate = static_cast<float>(hit_count) / static_cast<float>(query_num());
  LOG_INFO("DiskANN top-1 hit rate: {:.4f}", hit_rate);

  EXPECT_GE(hit_rate, 0.5F);
}

// =============================================================================
// Test Fixture 4: PQ-enabled tests (separate index with PQ)
// =============================================================================

class DiskANNPQTest : public ::testing::Test {
 protected:
  static inline std::string index_path = "/tmp/diskann_pq_test.index";
  static inline bool index_built = false;

  static void SetUpTestSuite() {
    if (g_resources.dim_ == 0) {
      g_resources.load();
    }

    if (!index_built) {
      auto params = DiskANNBuildParams()
                        .set_max_degree(64)
                        .set_ef_construction(128)
                        .set_num_threads(g_resources.max_threads_)
                        .set_pq_params(8);  // 8 subspaces

      DiskANNIndex<float, uint32_t>::build(g_resources.space_, index_path, params);
      index_built = true;
      LOG_INFO("DiskANNPQTest: Built PQ-enabled index at {}", index_path);
    }
  }

  static void TearDownTestSuite() {
    if (std::filesystem::exists(index_path)) {
      std::filesystem::remove(index_path);
    }
  }

  auto dataset() const -> const Dataset & { return g_resources.ds_; }
  auto dim() const -> uint32_t { return g_resources.dim_; }
  auto data_num() const -> uint32_t { return g_resources.data_num_; }
  auto query_num() const -> uint32_t { return g_resources.query_num_; }
  auto get_index_path() const -> const std::string & { return index_path; }
};

TEST_F(DiskANNPQTest, LoadPQIndex) {
  DiskANNIndex<float, uint32_t> index;
  index.load(get_index_path());

  EXPECT_TRUE(index.is_loaded());
  EXPECT_EQ(index.size(), data_num());
  EXPECT_EQ(index.dimension(), dim());
}

TEST_F(DiskANNPQTest, SearchWithPQ) {
  DiskANNIndex<float, uint32_t> index;
  index.load(get_index_path());

  constexpr uint32_t kTopk = 10;
  DiskANNSearchParams params(100);

  uint32_t hit_count = 0;
  for (uint32_t q = 0; q < query_num(); ++q) {
    std::vector<uint32_t> results(kTopk);
    index.search(dataset().queries_.data() + q * dim(), kTopk, results.data(), params);

    // Check if any ground truth is in results
    for (uint32_t k = 0; k < 100 && k < data_num(); ++k) {
      uint32_t gt_id = dataset().ground_truth_[q * 100 + k];
      if (std::ranges::find(results, gt_id) != results.end()) {
        ++hit_count;
        break;
      }
    }
  }

  float hit_rate = static_cast<float>(hit_count) / static_cast<float>(query_num());
  LOG_INFO("DiskANN PQ top-1 hit rate: {:.4f}", hit_rate);

  EXPECT_GE(hit_rate, 0.7F);
}

TEST_F(DiskANNPQTest, PQRecallQuality) {
  DiskANNIndex<float, uint32_t> index;
  index.load(get_index_path());

  constexpr uint32_t kTopk = 10;
  DiskANNSearchParams params(200);

  std::vector<uint32_t> results(query_num() * kTopk);
  for (uint32_t q = 0; q < query_num(); ++q) {
    const float *query = dataset().queries_.data() + q * dim();
    index.search(query, kTopk, results.data() + q * kTopk, params);
  }

  float recall =
      calc_recall(results.data(), dataset().ground_truth_.data(), query_num(), dataset().gt_dim_, kTopk);
  LOG_INFO("DiskANN PQ recall@{}: {:.4f}", kTopk, recall);

  // PQ may have slightly lower recall due to quantization
  EXPECT_GE(recall, 0.6F);
}

// =============================================================================
// Test Fixture 5: Searcher low-level tests
// =============================================================================

class DiskANNSearcherTest : public ::testing::Test {
 protected:
  static inline std::string index_path = "/tmp/diskann_searcher_test.index";

  static void SetUpTestSuite() {
    if (g_resources.dim_ == 0) {
      g_resources.load();
    }
  }

  void SetUp() override {
    // Build a fresh index for each test
    auto params = DiskANNBuildParams()
                      .set_max_degree(32)
                      .set_ef_construction(64)
                      .set_num_threads(g_resources.max_threads_);

    DiskANNBuilder<RawSpace<>> builder(g_resources.space_, params);
    builder.build_disk_index(index_path, params.num_threads_);
  }

  void TearDown() override {
    if (std::filesystem::exists(index_path)) {
      std::filesystem::remove(index_path);
    }
  }

  auto dataset() const -> const Dataset & { return g_resources.ds_; }
  auto dim() const -> uint32_t { return g_resources.dim_; }
  auto data_num() const -> uint32_t { return g_resources.data_num_; }
};

TEST_F(DiskANNSearcherTest, OpenAndVerify) {
  DiskANNSearcher<float, uint32_t> searcher;
  searcher.open(index_path);

  EXPECT_TRUE(searcher.is_open());
  EXPECT_EQ(searcher.num_points(), data_num());
  EXPECT_EQ(searcher.dimension(), dim());
  EXPECT_EQ(searcher.max_degree(), 32U);
}

TEST_F(DiskANNSearcherTest, SearchDirect) {
  DiskANNSearcher<float, uint32_t> searcher;
  searcher.open(index_path);

  constexpr uint32_t kTopk = 10;
  DiskANNSearchParams params(50);
  std::vector<uint32_t> results(kTopk);

  searcher.search(dataset().queries_.data(), kTopk, results.data(), params);

  for (uint32_t i = 0; i < kTopk; ++i) {
    if (results[i] != static_cast<uint32_t>(-1)) {
      EXPECT_LT(results[i], data_num());
    }
  }
}

TEST_F(DiskANNSearcherTest, CachingEnabled) {
  DiskANNSearcher<float, uint32_t> searcher;
  searcher.open(index_path);

  // Enable caching with 100 frames
  searcher.enable_caching(100);
  EXPECT_TRUE(searcher.is_caching_enabled());

  // Perform searches
  constexpr uint32_t kTopk = 10;
  DiskANNSearchParams params(50);
  std::vector<uint32_t> results(kTopk);

  // First search - should be cache misses
  searcher.search(dataset().queries_.data(), kTopk, results.data(), params);

  auto stats = searcher.cache_stats();
  EXPECT_GT(stats.total_accesses(), 0U);

  // Disable caching
  searcher.disable_caching();
  EXPECT_FALSE(searcher.is_caching_enabled());
}

TEST_F(DiskANNSearcherTest, CacheHitRate) {
  DiskANNSearcher<float, uint32_t> searcher;
  searcher.open(index_path);

  // Enable caching
  searcher.enable_caching(50);

  constexpr uint32_t kTopk = 10;
  DiskANNSearchParams params(50);
  std::vector<uint32_t> results(kTopk);

  // Run same query multiple times - should get cache hits
  for (int i = 0; i < 5; ++i) {
    searcher.search(dataset().queries_.data(), kTopk, results.data(), params);
  }

  auto stats = searcher.cache_stats();
  LOG_INFO("Cache stats: hits={}, misses={}, hit_rate={:.4f}",
           stats.hits_.load(), stats.misses_.load(), stats.hit_rate());

  // With repeated queries, we should have some cache hits
  EXPECT_GT(stats.hits_.load(), 0U);
}

// =============================================================================
// Test Fixture 6: BufferPool unit tests
// =============================================================================

class BufferPoolTest : public ::testing::Test {
 protected:
  static constexpr size_t kFrameSize = 4096;  // 4KB frames
  static constexpr size_t kCapacity = 10;
};

TEST_F(BufferPoolTest, Construction) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize);

  EXPECT_EQ(pool.capacity(), kCapacity);
  EXPECT_EQ(pool.frame_size(), kFrameSize);
  EXPECT_EQ(pool.size(), 0U);
  EXPECT_TRUE(pool.is_enabled());
}

TEST_F(BufferPoolTest, EmptyPoolDisabled) {
  BufferPool<uint32_t> pool(0, kFrameSize);
  EXPECT_FALSE(pool.is_enabled());
  EXPECT_EQ(pool.capacity(), 0U);
}

TEST_F(BufferPoolTest, PutAndGet) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize);

  // Create test data
  std::vector<uint8_t> test_data(kFrameSize);
  for (size_t i = 0; i < kFrameSize; ++i) {
    test_data[i] = static_cast<uint8_t>(i % 256);
  }

  // Put data into cache
  const uint8_t* cached = pool.put(42, test_data.data());
  EXPECT_NE(cached, nullptr);
  EXPECT_EQ(pool.size(), 1U);

  // Get data back
  const uint8_t* retrieved = pool.get(42);
  EXPECT_NE(retrieved, nullptr);

  // Verify data matches
  for (size_t i = 0; i < kFrameSize; ++i) {
    EXPECT_EQ(retrieved[i], test_data[i]);
  }
}

TEST_F(BufferPoolTest, CacheMiss) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize);

  // Try to get non-existent item
  const uint8_t* result = pool.get(999);
  EXPECT_EQ(result, nullptr);

  auto stats = pool.stats();
  EXPECT_EQ(stats.misses_.load(), 1U);
  EXPECT_EQ(stats.hits_.load(), 0U);
}

TEST_F(BufferPoolTest, LRUEviction) {
  BufferPool<uint32_t> pool(3, kFrameSize);  // Small capacity for testing eviction

  std::vector<uint8_t> data(kFrameSize, 0);

  // Fill the cache
  for (uint32_t i = 0; i < 3; ++i) {
    data[0] = static_cast<uint8_t>(i);
    pool.put(i, data.data());
  }
  EXPECT_EQ(pool.size(), 3U);

  // Access item 0 to make it recently used
  (void)pool.get(0);

  // Insert new item - should evict item 1 (LRU)
  data[0] = 100;
  pool.put(100, data.data());

  // Item 0 should still be there (was accessed recently)
  EXPECT_TRUE(pool.contains(0));
  // Item 1 should be evicted (LRU)
  EXPECT_FALSE(pool.contains(1));
  // Item 2 should still be there
  EXPECT_TRUE(pool.contains(2));
  // New item should be there
  EXPECT_TRUE(pool.contains(100));

  auto stats = pool.stats();
  EXPECT_GT(stats.evictions_.load(), 0U);
}

TEST_F(BufferPoolTest, Clear) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize);

  std::vector<uint8_t> data(kFrameSize, 0);
  for (uint32_t i = 0; i < 5; ++i) {
    pool.put(i, data.data());
  }
  EXPECT_EQ(pool.size(), 5U);

  pool.clear();
  EXPECT_EQ(pool.size(), 0U);

  // Items should no longer be found
  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_FALSE(pool.contains(i));
  }
}

TEST_F(BufferPoolTest, Statistics) {
  BufferPool<uint32_t> pool(kCapacity, kFrameSize);

  std::vector<uint8_t> data(kFrameSize, 0);
  pool.put(1, data.data());

  // Miss
  (void)pool.get(999);
  // Hit
  (void)pool.get(1);
  // Another hit
  (void)pool.get(1);

  auto stats = pool.stats();
  EXPECT_EQ(stats.hits_.load(), 2U);
  EXPECT_EQ(stats.misses_.load(), 1U);
  EXPECT_DOUBLE_EQ(stats.hit_rate(), 2.0 / 3.0);

  // Reset stats
  pool.reset_stats();
  stats = pool.stats();
  EXPECT_EQ(stats.hits_.load(), 0U);
  EXPECT_EQ(stats.misses_.load(), 0U);
}

}  // namespace alaya
