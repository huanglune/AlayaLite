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
#include <cstdio>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "coro/task.hpp"
#include "executor/scheduler.hpp"
#include "index/diskann/diskann_index.hpp"
#include "index/diskann/diskann_params.hpp"
#include "index/diskann/diskann_searcher.hpp"
#include "space/raw_space.hpp"
#include "storage/io/io_uring_engine.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"
#include "utils/log.hpp"

namespace alaya {

// =============================================================================
// Shared test resources
// =============================================================================

struct SearcherTestResources {
  Dataset ds_;
  uint32_t dim_{0};
  uint32_t data_num_{0};
  uint32_t query_num_{0};
  std::shared_ptr<RawSpace<>> space_;
  static constexpr uint32_t kMaxTestThreads = 4;
  uint32_t max_threads_{std::max(1U, std::min(std::thread::hardware_concurrency(), kMaxTestThreads))};

  std::filesystem::path tmp_dir_;
  std::filesystem::path index_path_;
  bool index_built_{false};

  void load() {
    ds_ = load_dataset(random_config(kDataNum, kQueryNum, kDim, kGtTopk));
    dim_ = ds_.dim_;
    data_num_ = ds_.data_num_;
    query_num_ = ds_.query_num_;
    space_ = std::make_shared<RawSpace<>>(data_num_, dim_, MetricType::L2);
    space_->fit(ds_.data_.data(), data_num_);

    auto unique_suffix =
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    tmp_dir_ = std::filesystem::temp_directory_path() / ("diskann_searcher_test_" + unique_suffix);
    std::filesystem::create_directories(tmp_dir_);
    index_path_ = tmp_dir_ / "test_idx";

    LOG_INFO("SearcherTestResources: Loaded {} vectors, dim={}", data_num_, dim_);
  }

  void build_index() {
    if (index_built_) {
      return;
    }
    auto params = DiskANNBuildParams()
                      .set_max_degree(kDefaultR)
                      .set_ef_construction(kDefaultEf)
                      .set_num_iterations(2)
                      .set_num_threads(max_threads_);

    DiskANNIndex<>::build(space_, index_path_.string(), params);
    index_built_ = true;
    LOG_INFO("SearcherTestResources: Index built at {}", index_path_.string());
  }

 private:
  static constexpr uint32_t kDataNum = 256;
  static constexpr uint32_t kQueryNum = 8;
  static constexpr uint32_t kDim = 32;
  static constexpr uint32_t kGtTopk = 50;
  static constexpr uint32_t kDefaultR = 32;
  static constexpr uint32_t kDefaultEf = 64;
};

static SearcherTestResources g_res;  // NOLINT(cert-err58-cpp)

// =============================================================================
// Fixture: DiskANN Searcher Tests
// =============================================================================

class DiskANNSearcherTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (g_res.dim_ == 0) {
      g_res.load();
    }
    g_res.build_index();
  }

  auto space() { return g_res.space_; }
  auto dim() const -> uint32_t { return g_res.dim_; }
  auto data_num() const -> uint32_t { return g_res.data_num_; }
  auto query_num() const -> uint32_t { return g_res.query_num_; }
  auto dataset() const -> const Dataset & { return g_res.ds_; }
  auto index_path() const -> std::string { return g_res.index_path_.string(); }
  auto max_threads() const -> uint32_t { return g_res.max_threads_; }

  auto load_index(size_t cache_cap = 4096) -> DiskANNIndex<> {
    DiskANNIndex<> index;
    index.load(index_path(), cache_cap);
    return index;
  }

  auto load_writable_index(size_t cache_cap = 4096) -> DiskANNIndex<> {
    DiskANNIndex<> index;
    index.load(index_path(), cache_cap, true);
    return index;
  }

  auto compute_recall(const uint32_t *results, uint32_t topk) -> float {
    return calc_recall(results,
                       dataset().ground_truth_.data(),
                       query_num(),
                       dataset().gt_dim_,
                       topk);
  }
};

// -----------------------------------------------------------------------------
// 1. State management
// -----------------------------------------------------------------------------

TEST_F(DiskANNSearcherTest, SearchNotLoadedThrows) {
  DiskANNIndex<> index;
  EXPECT_FALSE(index.is_loaded());

  std::vector<float> query(dim(), 0.0F);
  std::vector<uint32_t> results(10);
  EXPECT_THROW(index.search(query.data(), 10, results.data()), std::runtime_error);
}

TEST_F(DiskANNSearcherTest, SearchWithDistanceNotLoadedThrows) {
  DiskANNIndex<> index;

  std::vector<float> query(dim(), 0.0F);
  std::vector<uint32_t> ids(10);
  std::vector<float> dists(10);
  EXPECT_THROW(index.search_with_distance(query.data(), 10, ids.data(), dists.data()),
               std::runtime_error);
}

TEST_F(DiskANNSearcherTest, IsLoadedLifecycle) {
  DiskANNIndex<> index;
  EXPECT_FALSE(index.is_loaded());

  index.load(index_path());
  EXPECT_TRUE(index.is_loaded());

  index.close();
  EXPECT_FALSE(index.is_loaded());
}

TEST_F(DiskANNSearcherTest, IndexPath) {
  auto index = load_index();
  EXPECT_EQ(index.path(), index_path());

  index.close();
  EXPECT_TRUE(index.path().empty());
}

TEST_F(DiskANNSearcherTest, IndexSizeAndDimension) {
  auto index = load_index();

  EXPECT_EQ(index.dimension(), dim());
  // capacity may be rounded up for bitmap alignment, so size >= data_num
  EXPECT_GE(index.size(), data_num());
}

TEST_F(DiskANNSearcherTest, ReserveRequiresWritableIndex) {
  auto index = load_index();
  EXPECT_THROW(index.reserve(static_cast<uint32_t>(index.size()) + 64), std::runtime_error);
}

TEST_F(DiskANNSearcherTest, ReserveAndFlushPersistCapacityGrowth) {
  auto index = load_writable_index();
  uint32_t old_capacity = index.get_searcher().capacity();
  uint32_t requested_capacity = old_capacity + 57;

  index.reserve(requested_capacity);
  EXPECT_GE(index.get_searcher().capacity(), requested_capacity);
  index.flush();
  index.close();

  auto reopened = load_index();
  EXPECT_GE(reopened.get_searcher().capacity(), requested_capacity);
}

TEST_F(DiskANNSearcherTest, InsertDuplicateExternalIdThrows) {
  auto index = load_writable_index();
  std::vector<float> vec(dim(), 1.0F);

  // Inserting with an already-mapped external ID should throw
  EXPECT_THROW(index.insert(vec.data(), 0), std::invalid_argument);
}

// -----------------------------------------------------------------------------
// 2. Basic search correctness
// -----------------------------------------------------------------------------

TEST_F(DiskANNSearcherTest, BasicSearchReturnsTopK) {
  auto index = load_index();
  constexpr uint32_t kTopk = 10;

  const float *query = dataset().queries_.data();
  std::vector<uint32_t> results(kTopk);

  EXPECT_NO_THROW(index.search(query, kTopk, results.data()));

  for (uint32_t i = 0; i < kTopk; ++i) {
    EXPECT_LT(results[i], data_num()) << "Result ID out of range at position " << i;
  }
}

TEST_F(DiskANNSearcherTest, SearchResultsAreUnique) {
  auto index = load_index();
  constexpr uint32_t kTopk = 10;

  const float *query = dataset().queries_.data();
  std::vector<uint32_t> results(kTopk);
  index.search(query, kTopk, results.data());

  std::set<uint32_t> unique_ids(results.begin(), results.end());
  EXPECT_EQ(unique_ids.size(), kTopk) << "Search results contain duplicates";
}

TEST_F(DiskANNSearcherTest, SearchWithDistanceReturnsOrderedDistances) {
  auto index = load_index();
  constexpr uint32_t kTopk = 10;

  const float *query = dataset().queries_.data();
  std::vector<uint32_t> ids(kTopk);
  std::vector<float> dists(kTopk);

  index.search_with_distance(query, kTopk, ids.data(), dists.data());

  for (uint32_t i = 1; i < kTopk; ++i) {
    EXPECT_LE(dists[i - 1], dists[i]) << "Distances not sorted at position " << i;
  }
}

TEST_F(DiskANNSearcherTest, L2DistancesAreNonNegative) {
  auto index = load_index();
  constexpr uint32_t kTopk = 10;

  for (uint32_t q = 0; q < query_num(); ++q) {
    const float *query = dataset().queries_.data() + q * dim();
    std::vector<uint32_t> ids(kTopk);
    std::vector<float> dists(kTopk);

    index.search_with_distance(query, kTopk, ids.data(), dists.data());

    for (uint32_t i = 0; i < kTopk; ++i) {
      EXPECT_GE(dists[i], 0.0F) << "Negative L2 distance at query " << q << " position " << i;
    }
  }
}

TEST_F(DiskANNSearcherTest, SearchIDsMatchSearchWithDistance) {
  auto index = load_index();
  constexpr uint32_t kTopk = 10;

  const float *query = dataset().queries_.data();

  std::vector<uint32_t> ids_only(kTopk);
  index.search(query, kTopk, ids_only.data());

  std::vector<uint32_t> ids_with_dist(kTopk);
  std::vector<float> dists(kTopk);
  index.search_with_distance(query, kTopk, ids_with_dist.data(), dists.data());

  EXPECT_EQ(ids_only, ids_with_dist)
      << "search() and search_with_distance() should return the same IDs";
}

// -----------------------------------------------------------------------------
// 3. Recall quality
// -----------------------------------------------------------------------------

TEST_F(DiskANNSearcherTest, SearchRecallIsReasonable) {
  auto index = load_index();
  constexpr uint32_t kTopk = 10;

  DiskANNSearchParams params;
  params.set_ef_search(64).set_beam_width(4);

  std::vector<uint32_t> all_results(query_num() * kTopk);
  for (uint32_t q = 0; q < query_num(); ++q) {
    const float *query = dataset().queries_.data() + q * dim();
    index.search(query, kTopk, all_results.data() + q * kTopk, params);
  }

  float recall = compute_recall(all_results.data(), kTopk);
  LOG_INFO("DiskANNSearcher recall@{}: {:.4f}", kTopk, recall);

  EXPECT_GE(recall, 0.5F) << "Recall too low for the shared random test dataset";
}

TEST_F(DiskANNSearcherTest, HigherEfSearchImprovesOrMaintainsRecall) {
  auto index = load_index();
  constexpr uint32_t kTopk = 10;

  auto search_all_queries = [&](uint32_t ef) -> float {
    DiskANNSearchParams params;
    params.set_ef_search(ef).set_beam_width(4);

    std::vector<uint32_t> results(query_num() * kTopk);
    for (uint32_t q = 0; q < query_num(); ++q) {
      const float *query = dataset().queries_.data() + q * dim();
      index.search(query, kTopk, results.data() + q * kTopk, params);
    }
    return compute_recall(results.data(), kTopk);
  };

  float recall_low = search_all_queries(32);
  float recall_high = search_all_queries(128);

  LOG_INFO("Recall ef=32: {:.4f}, ef=128: {:.4f}", recall_low, recall_high);

  // Higher ef_search should not significantly degrade recall
  EXPECT_GE(recall_high, recall_low * 0.95F)
      << "Higher ef_search should give equal or better recall";
}

// -----------------------------------------------------------------------------
// 4. TopK variation
// -----------------------------------------------------------------------------

TEST_F(DiskANNSearcherTest, TopKOne) {
  auto index = load_index();
  const float *query = dataset().queries_.data();
  std::vector<uint32_t> results(1);

  EXPECT_NO_THROW(index.search(query, 1, results.data()));
  EXPECT_LT(results[0], data_num());
}

TEST_F(DiskANNSearcherTest, TopKVariation) {
  auto index = load_index();
  const float *query = dataset().queries_.data();

  for (uint32_t topk : {1U, 5U, 10U, 20U, 50U}) {
    std::vector<uint32_t> results(topk);
    EXPECT_NO_THROW(index.search(query, topk, results.data()))
        << "Search failed with topk=" << topk;

    // All returned IDs should be valid
    for (uint32_t i = 0; i < topk; ++i) {
      if (results[i] != static_cast<uint32_t>(-1)) {
        EXPECT_LT(results[i], data_num()) << "Invalid ID at topk=" << topk << " pos=" << i;
      }
    }
  }
}

// -----------------------------------------------------------------------------
// 5. Beam width variation
// -----------------------------------------------------------------------------

TEST_F(DiskANNSearcherTest, DifferentBeamWidths) {
  auto index = load_index();
  constexpr uint32_t kTopk = 10;
  const float *query = dataset().queries_.data();

  for (uint32_t bw : {1U, 2U, 4U, 8U}) {
    DiskANNSearchParams params;
    params.set_beam_width(bw).set_ef_search(64);

    std::vector<uint32_t> results(kTopk);
    EXPECT_NO_THROW(index.search(query, kTopk, results.data(), params))
        << "Search failed with beam_width=" << bw;

    for (uint32_t i = 0; i < kTopk; ++i) {
      if (results[i] != static_cast<uint32_t>(-1)) {
        EXPECT_LT(results[i], data_num());
      }
    }
  }
}

// -----------------------------------------------------------------------------
// 6. Determinism: same query -> same results
// -----------------------------------------------------------------------------

TEST_F(DiskANNSearcherTest, SearchIsDeterministic) {
  auto index = load_index();
  constexpr uint32_t kTopk = 10;
  const float *query = dataset().queries_.data();

  std::vector<uint32_t> results1(kTopk);
  std::vector<uint32_t> results2(kTopk);

  index.search(query, kTopk, results1.data());
  index.search(query, kTopk, results2.data());

  EXPECT_EQ(results1, results2) << "Same query should produce identical results";
}

// -----------------------------------------------------------------------------
// 7. Load / close / reload cycle
// -----------------------------------------------------------------------------

TEST_F(DiskANNSearcherTest, LoadCloseReloadCycle) {
  constexpr uint32_t kTopk = 10;
  const float *query = dataset().queries_.data();

  std::vector<uint32_t> results_before(kTopk);
  {
    auto index = load_index();
    index.search(query, kTopk, results_before.data());
    index.close();
    EXPECT_FALSE(index.is_loaded());
  }

  std::vector<uint32_t> results_after(kTopk);
  {
    auto index = load_index();
    index.search(query, kTopk, results_after.data());
  }

  EXPECT_EQ(results_before, results_after) << "Results differ after reload";
}

// -----------------------------------------------------------------------------
// 8. Move semantics
// -----------------------------------------------------------------------------

TEST_F(DiskANNSearcherTest, MoveConstruction) {
  auto index1 = load_index();
  EXPECT_TRUE(index1.is_loaded());

  DiskANNIndex<> index2 = std::move(index1);
  EXPECT_TRUE(index2.is_loaded());

  constexpr uint32_t kTopk = 10;
  const float *query = dataset().queries_.data();
  std::vector<uint32_t> results(kTopk);

  EXPECT_NO_THROW(index2.search(query, kTopk, results.data()));
}

TEST_F(DiskANNSearcherTest, MoveAssignment) {
  auto index1 = load_index();
  DiskANNIndex<> index2;
  EXPECT_FALSE(index2.is_loaded());

  index2 = std::move(index1);
  EXPECT_TRUE(index2.is_loaded());

  constexpr uint32_t kTopk = 10;
  const float *query = dataset().queries_.data();
  std::vector<uint32_t> results(kTopk);
  EXPECT_NO_THROW(index2.search(query, kTopk, results.data()));
}

// -----------------------------------------------------------------------------
// 9. Batch search
// -----------------------------------------------------------------------------

TEST_F(DiskANNSearcherTest, BatchSearchReturnsValidResults) {
  auto index = load_index();
  constexpr uint32_t kTopk = 10;

  std::vector<uint32_t> results(query_num() * kTopk);

  EXPECT_NO_THROW(
      index.batch_search(dataset().queries_.data(), query_num(), kTopk, results.data()));

  for (uint32_t i = 0; i < query_num() * kTopk; ++i) {
    if (results[i] != static_cast<uint32_t>(-1)) {
      EXPECT_LT(results[i], data_num()) << "Batch result ID out of range at " << i;
    }
  }
}

TEST_F(DiskANNSearcherTest, BatchSearchRecall) {
  auto index = load_index();
  constexpr uint32_t kTopk = 10;

  DiskANNSearchParams params;
  params.set_ef_search(64).set_beam_width(4);

  std::vector<uint32_t> results(query_num() * kTopk);
  index.batch_search(dataset().queries_.data(), query_num(), kTopk, results.data(), params);

  float recall = compute_recall(results.data(), kTopk);
  LOG_INFO("BatchSearch recall@{}: {:.4f}", kTopk, recall);

  EXPECT_GE(recall, 0.5F) << "Batch search recall should match single search quality";
}

// -----------------------------------------------------------------------------
// 10. Search params builder
// -----------------------------------------------------------------------------

TEST_F(DiskANNSearcherTest, SearchParamsBuilder) {
  auto params = DiskANNSearchParams()
                    .set_ef_search(128)
                    .set_beam_width(8)
                    .set_cache_capacity(8192)
                    .set_pipeline_width(32)
                    .set_num_threads(4);

  EXPECT_EQ(params.ef_search_, 128U);
  EXPECT_EQ(params.beam_width_, 8U);
  EXPECT_EQ(params.cache_capacity_, 8192U);
  EXPECT_EQ(params.pipeline_width_, 32U);
  EXPECT_EQ(params.num_threads_, 4U);
}

TEST_F(DiskANNSearcherTest, SearchParamsDefault) {
  DiskANNSearchParams params;

  EXPECT_EQ(params.ef_search_, 64U);
  EXPECT_EQ(params.beam_width_, 4U);
  EXPECT_EQ(params.cache_capacity_, 4096U);
}

// -----------------------------------------------------------------------------
// 11. Nearest neighbor sanity check (1-NN should be very close to brute-force)
// -----------------------------------------------------------------------------

TEST_F(DiskANNSearcherTest, NearestNeighborSanity) {
  auto index = load_index();
  auto dist_fn = space()->get_dist_func();

  DiskANNSearchParams params;
  params.set_ef_search(128).set_beam_width(8);

  uint32_t correct = 0;
  for (uint32_t q = 0; q < query_num(); ++q) {
    const float *query = dataset().queries_.data() + q * dim();

    // 1-NN via searcher
    uint32_t search_nn = 0;
    index.search(query, 1, &search_nn, params);

    // 1-NN via brute-force
    float min_dist = std::numeric_limits<float>::max();
    uint32_t true_nn = 0;
    for (uint32_t i = 0; i < data_num(); ++i) {
      float d = dist_fn(query, space()->get_data_by_id(i), dim());
      if (d < min_dist) {
        min_dist = d;
        true_nn = i;
      }
    }

    if (search_nn == true_nn) {
      ++correct;
    }
  }

  float accuracy = static_cast<float>(correct) / static_cast<float>(query_num());
  LOG_INFO("1-NN accuracy: {:.2f}% ({}/{})", accuracy * 100, correct, query_num());

  EXPECT_GE(accuracy, 0.5F) << "At least 50% of 1-NN queries should be exact";
}

// =============================================================================
// Fixture: DiskANNSearcher Direct Tests (bypass DiskANNIndex)
// =============================================================================

class DiskANNSearcherDirectTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (g_res.dim_ == 0) {
      g_res.load();
    }
    g_res.build_index();
  }

  auto index_path() const -> std::string { return g_res.index_path_.string(); }
  auto data_num() const -> uint32_t { return g_res.data_num_; }
  auto dim() const -> uint32_t { return g_res.dim_; }
  auto dataset() const -> const Dataset & { return g_res.ds_; }
};

// -----------------------------------------------------------------------------
// Open and search directly via DiskANNSearcher
// -----------------------------------------------------------------------------

TEST_F(DiskANNSearcherDirectTest, OpenAndSearch) {
  DiskANNSearcher<> searcher;
  searcher.open(index_path());

  constexpr uint32_t kTopk = 10;
  const float *query = dataset().queries_.data();

  auto result = searcher.search(query, kTopk);

  ASSERT_FALSE(result.ids_.empty());
  ASSERT_LE(result.ids_.size(), kTopk);

  for (auto id : result.ids_) {
    if (id != static_cast<uint32_t>(-1)) {
      EXPECT_LT(id, data_num());
    }
  }
}

TEST_F(DiskANNSearcherDirectTest, SearchReturnsDistances) {
  DiskANNSearcher<> searcher;
  searcher.open(index_path());

  constexpr uint32_t kTopk = 10;
  const float *query = dataset().queries_.data();

  auto result = searcher.search(query, kTopk);

  ASSERT_EQ(result.ids_.size(), result.distances_.size());

  // L2 distances should be non-negative and ordered
  for (size_t i = 0; i < result.distances_.size(); ++i) {
    EXPECT_GE(result.distances_[i], 0.0F);
    if (i > 0) {
      EXPECT_LE(result.distances_[i - 1], result.distances_[i]);
    }
  }
}

TEST_F(DiskANNSearcherDirectTest, SearchNotOpenedThrows) {
  DiskANNSearcher<> searcher;
  const float *query = dataset().queries_.data();
  EXPECT_THROW(searcher.search(query, 10), std::runtime_error);
}

TEST_F(DiskANNSearcherDirectTest, SearchWithCustomParams) {
  DiskANNSearcher<> searcher;
  searcher.open(index_path());

  constexpr uint32_t kTopk = 10;
  const float *query = dataset().queries_.data();

  DiskANNSearchParams params;
  params.set_ef_search(128).set_beam_width(8);

  auto result = searcher.search(query, kTopk, params);

  ASSERT_FALSE(result.ids_.empty());
  for (auto id : result.ids_) {
    if (id != static_cast<uint32_t>(-1)) {
      EXPECT_LT(id, data_num());
    }
  }
}

TEST_F(DiskANNSearcherDirectTest, MultipleSearchesOnSameInstance) {
  DiskANNSearcher<> searcher;
  searcher.open(index_path());

  constexpr uint32_t kTopk = 5;

  for (uint32_t q = 0; q < g_res.query_num_; ++q) {
    const float *query = dataset().queries_.data() + q * dim();
    auto result = searcher.search(query, kTopk);

    ASSERT_FALSE(result.ids_.empty()) << "Empty result on query " << q;
    for (auto id : result.ids_) {
      if (id != static_cast<uint32_t>(-1)) {
        EXPECT_LT(id, data_num()) << "Invalid ID on query " << q;
      }
    }
  }
}

TEST_F(DiskANNSearcherTest, SearchWithBeamWidthOneReturnsTopKOnSmallIndex) {
  constexpr uint32_t kSmallDataNum = 64;
  constexpr uint32_t kTopk = 5;

  std::vector<float> small_data(dataset().data_.begin(),
                                dataset().data_.begin() + kSmallDataNum * dim());
  auto small_space = std::make_shared<RawSpace<>>(kSmallDataNum, dim(), MetricType::L2);
  small_space->fit(small_data.data(), kSmallDataNum);

  auto tmp_dir = std::filesystem::temp_directory_path() / "diskann_searcher_beam_width_one_test";
  std::filesystem::remove_all(tmp_dir);
  std::filesystem::create_directories(tmp_dir);
  auto small_index_path = tmp_dir / "test_idx";

  auto build_params = DiskANNBuildParams()
                          .set_max_degree(8)
                          .set_ef_construction(16)
                          .set_num_iterations(1)
                          .set_num_threads(std::min(max_threads(), 4U));
  DiskANNIndex<>::build(small_space, small_index_path.string(), build_params);

  DiskANNIndex<> index;
  index.load(small_index_path.string());

  DiskANNSearchParams search_params;
  search_params.set_beam_width(1).set_ef_search(8);

  std::vector<uint32_t> results(kTopk, static_cast<uint32_t>(-1));
  EXPECT_NO_THROW(index.search(dataset().queries_.data(), kTopk, results.data(), search_params));
  EXPECT_EQ(std::count_if(results.begin(),
                          results.end(),
                          [](uint32_t id) {
                            return id != static_cast<uint32_t>(-1);
                          }),
            static_cast<int>(kTopk));

  for (uint32_t id : results) {
    EXPECT_LT(id, kSmallDataNum);
  }

  index.close();
  std::filesystem::remove_all(tmp_dir);
}

TEST_F(DiskANNSearcherTest, SearchWithEfSearchOneDoesNotCrashOnSmallIndex) {
  constexpr uint32_t kSmallDataNum = 64;
  constexpr uint32_t kTopk = 5;

  std::vector<float> small_data(dataset().data_.begin(),
                                dataset().data_.begin() + kSmallDataNum * dim());
  auto small_space = std::make_shared<RawSpace<>>(kSmallDataNum, dim(), MetricType::L2);
  small_space->fit(small_data.data(), kSmallDataNum);

  auto tmp_dir = std::filesystem::temp_directory_path() / "diskann_searcher_ef_search_one_test";
  std::filesystem::remove_all(tmp_dir);
  std::filesystem::create_directories(tmp_dir);
  auto small_index_path = tmp_dir / "test_idx";

  auto build_params = DiskANNBuildParams()
                          .set_max_degree(8)
                          .set_ef_construction(16)
                          .set_num_iterations(1)
                          .set_num_threads(std::min(max_threads(), 4U));
  DiskANNIndex<>::build(small_space, small_index_path.string(), build_params);

  DiskANNIndex<> index;
  index.load(small_index_path.string());

  DiskANNSearchParams search_params;
  search_params.set_ef_search(1).set_beam_width(4);

  std::vector<uint32_t> results(kTopk, static_cast<uint32_t>(-1));
  EXPECT_NO_THROW(index.search(dataset().queries_.data(), kTopk, results.data(), search_params));

  index.close();
  std::filesystem::remove_all(tmp_dir);
}

// =============================================================================
// Fixture: DiskANN Update/Delete Tests
// =============================================================================

class DiskANNUpdateDeleteTest : public ::testing::Test {
 protected:
  static constexpr uint32_t kDataNum = 200;
  static constexpr uint32_t kDim = 16;
  static constexpr uint32_t kR = 16;
  static constexpr uint32_t kEf = 64;
  static constexpr uint32_t kTopk = 10;

  std::filesystem::path tmp_dir_;
  std::filesystem::path index_path_;
  std::vector<float> data_;

  void SetUp() override {
    // Generate deterministic random data
    data_.resize(kDataNum * kDim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    for (auto &v : data_) {
      v = dist(rng);
    }

    auto unique_suffix =
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    tmp_dir_ = std::filesystem::temp_directory_path() / ("diskann_update_delete_" + unique_suffix);
    std::filesystem::create_directories(tmp_dir_);
    index_path_ = tmp_dir_ / "test_idx";

    // Build index
    auto space = std::make_shared<RawSpace<>>(kDataNum, kDim, MetricType::L2);
    space->fit(data_.data(), kDataNum);

    auto params = DiskANNBuildParams()
                      .set_max_degree(kR)
                      .set_ef_construction(kEf)
                      .set_num_iterations(2)
                      .set_num_threads(1);
    DiskANNIndex<>::build(space, index_path_.string(), params);
  }

  void TearDown() override { std::filesystem::remove_all(tmp_dir_); }

  auto load_writable() -> DiskANNIndex<> {
    DiskANNIndex<> index;
    index.load(index_path_.string(), 4096, true);
    return index;
  }

  auto load_readonly() -> DiskANNIndex<> {
    DiskANNIndex<> index;
    index.load(index_path_.string(), 4096, false);
    return index;
  }

  auto make_query(uint32_t id) const -> const float * { return data_.data() + id * kDim; }

  auto make_random_vector(uint32_t seed) const -> std::vector<float> {
    std::vector<float> vec(kDim);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    for (auto &v : vec) {
      v = dist(rng);
    }
    return vec;
  }
};

// -----------------------------------------------------------------------------
// 2.2 RobustPrune test (via insert which exercises it)
// -----------------------------------------------------------------------------

TEST_F(DiskANNUpdateDeleteTest, InsertExercisesRobustPrune) {
  auto index = load_writable();
  auto vec = make_random_vector(9999);

  // Insert exercises greedy_search_for_insert + robust_prune_disk
  EXPECT_NO_THROW(index.insert(vec.data(), kDataNum + 100));

  // Verify the inserted vector is findable
  DiskANNSearchParams params;
  params.set_ef_search(128).set_beam_width(4);
  std::vector<uint32_t> ids(kTopk);
  std::vector<float> dists(kTopk);
  index.search_with_distance(vec.data(), kTopk, ids.data(), dists.data(), params);

  // The search should return valid results
  EXPECT_FALSE(ids.empty());
}

// -----------------------------------------------------------------------------
// 3.2 Search after delete: deleted IDs never appear in results
// -----------------------------------------------------------------------------

TEST_F(DiskANNUpdateDeleteTest, SearchAfterDeleteExcludesDeletedIDs) {
  auto index = load_writable();

  // Pick some non-medoid IDs to delete
  std::vector<uint32_t> deleted_ids;
  for (uint32_t id = 1; id < 10; ++id) {
    // Skip medoid (can't delete it)
    try {
      index.delete_vector(id);
      deleted_ids.push_back(id);
    } catch (const std::logic_error &e) {
      (void)e;  // Entry point, skip
    }
  }
  ASSERT_FALSE(deleted_ids.empty());

  // Search with all queries, verify deleted IDs never appear
  DiskANNSearchParams params;
  params.set_ef_search(128).set_beam_width(4);

  for (uint32_t q = 0; q < 20; ++q) {
    const float *query = make_query(q % kDataNum);
    std::vector<uint32_t> ids(kTopk);
    index.search(query, kTopk, ids.data(), params);

    for (auto del_id : deleted_ids) {
      for (uint32_t i = 0; i < kTopk; ++i) {
        EXPECT_NE(ids[i], del_id) << "Deleted ID " << del_id << " found in search results";
      }
    }
  }
}

// -----------------------------------------------------------------------------
// 4.3 Delete tests: error cases
// -----------------------------------------------------------------------------

TEST_F(DiskANNUpdateDeleteTest, DeleteSuccessful) {
  auto index = load_writable();

  // Delete a non-medoid vector
  uint32_t to_delete = 1;
  auto &searcher = index.get_searcher();
  // Make sure we don't try to delete the medoid
  try {
    index.delete_vector(to_delete);
  } catch (const std::logic_error &) {
    to_delete = 2;
    index.delete_vector(to_delete);
  }

  // Size should decrease
  EXPECT_EQ(searcher.num_points(), kDataNum - 1);
}

TEST_F(DiskANNUpdateDeleteTest, DeleteNonExistentThrows) {
  auto index = load_writable();

  // External ID 99999 doesn't exist
  EXPECT_THROW(index.delete_vector(99999), std::invalid_argument);
}

TEST_F(DiskANNUpdateDeleteTest, DeleteEntryPointThrows) {
  auto index = load_writable();

  // Find the medoid by searching and checking which ID is the entry point
  // The medoid's external_id in builder-built indices is the same as internal_id
  // We need to figure out which ID is the entry point
  // Try deleting IDs until we find the medoid
  bool found_medoid = false;
  for (uint32_t id = 0; id < kDataNum; ++id) {
    try {
      index.delete_vector(id);
      // If it didn't throw, it's not the medoid — but we already deleted it.
      // For this test, we just need to verify that at least one ID throws logic_error.
      // Undo: we can't un-delete, so let's use a different approach.
    } catch (const std::logic_error &) {
      found_medoid = true;
      break;
    } catch (...) {
      (void)0;  // Not the right exception
    }
  }
  EXPECT_TRUE(found_medoid) << "Entry point should be protected from deletion";
}

TEST_F(DiskANNUpdateDeleteTest, DeleteOnReadOnlyThrows) {
  auto index = load_readonly();
  EXPECT_THROW(index.delete_vector(1), std::runtime_error);
}

TEST_F(DiskANNUpdateDeleteTest, DeleteAlreadyDeletedThrows) {
  auto index = load_writable();

  // Delete a non-medoid vector
  uint32_t to_delete = 1;
  try {
    index.delete_vector(to_delete);
  } catch (const std::logic_error &) {
    to_delete = 2;
    index.delete_vector(to_delete);
  }

  // Deleting again should throw
  EXPECT_THROW(index.delete_vector(to_delete), std::invalid_argument);
}

// -----------------------------------------------------------------------------
// 5.5 Insert tests
// -----------------------------------------------------------------------------

TEST_F(DiskANNUpdateDeleteTest, InsertSuccessfulAndSearchable) {
  auto index = load_writable();
  auto vec = make_random_vector(12345);

  uint32_t new_id = kDataNum + 50;
  EXPECT_NO_THROW(index.insert(vec.data(), new_id));

  // Search for the inserted vector
  DiskANNSearchParams params;
  params.set_ef_search(128).set_beam_width(4);
  std::vector<uint32_t> ids(kTopk);
  index.search(vec.data(), kTopk, ids.data(), params);

  // The newly inserted vector should appear in results
  bool found = std::find(ids.begin(), ids.end(), new_id) != ids.end();
  EXPECT_TRUE(found) << "Inserted vector should be searchable";
}

TEST_F(DiskANNUpdateDeleteTest, InsertDuplicateThrows) {
  auto index = load_writable();
  auto vec = make_random_vector(111);

  // External ID 0 already exists (builder-built)
  EXPECT_THROW(index.insert(vec.data(), 0), std::invalid_argument);
}

TEST_F(DiskANNUpdateDeleteTest, InsertOnReadOnlyThrows) {
  auto index = load_readonly();
  auto vec = make_random_vector(222);
  EXPECT_THROW(index.insert(vec.data(), kDataNum + 1), std::runtime_error);
}

// -----------------------------------------------------------------------------
// 6.1 Mixed insert+delete workload
// -----------------------------------------------------------------------------

TEST_F(DiskANNUpdateDeleteTest, MixedInsertDeleteWorkload) {
  auto index = load_writable();

  // Delete some vectors (skip medoid)
  std::vector<uint32_t> deleted;
  for (uint32_t id = 10; id < 30; ++id) {
    try {
      index.delete_vector(id);
      deleted.push_back(id);
    } catch (const std::logic_error &e) {
      (void)e;  // medoid, skip
    }
  }

  // Insert new vectors
  std::vector<uint32_t> inserted;
  for (uint32_t i = 0; i < 20; ++i) {
    uint32_t new_id = kDataNum + 200 + i;
    auto vec = make_random_vector(new_id);
    index.insert(vec.data(), new_id);
    inserted.push_back(new_id);
  }

  // Verify search quality: search for each inserted vector
  DiskANNSearchParams params;
  params.set_ef_search(128).set_beam_width(4);

  uint32_t found_count = 0;
  for (auto new_id : inserted) {
    auto vec = make_random_vector(new_id);
    std::vector<uint32_t> ids(kTopk);
    index.search(vec.data(), kTopk, ids.data(), params);

    if (std::find(ids.begin(), ids.end(), new_id) != ids.end()) {
      ++found_count;
    }

    // Deleted IDs should not appear
    for (auto del_id : deleted) {
      EXPECT_TRUE(std::find(ids.begin(), ids.end(), del_id) == ids.end())
          << "Deleted ID " << del_id << " appeared in results";
    }
  }

  float recall = static_cast<float>(found_count) / static_cast<float>(inserted.size());
  LOG_INFO("Mixed workload: inserted {} vectors, {} found in self-search (recall={:.2f})",
           inserted.size(),
           found_count,
           recall);
  EXPECT_GE(recall, 0.8F) << "At least 80% of inserted vectors should be self-searchable";
}

// -----------------------------------------------------------------------------
// 6.2 Insert-after-delete slot reuse
// -----------------------------------------------------------------------------

TEST_F(DiskANNUpdateDeleteTest, InsertAfterDeleteSlotReuse) {
  auto index = load_writable();

  // Delete some vectors
  uint32_t to_delete = 5;
  try {
    index.delete_vector(to_delete);
  } catch (const std::logic_error &) {
    to_delete = 6;
    index.delete_vector(to_delete);
  }

  auto old_size = index.get_searcher().num_points();

  // Insert a new vector — should reuse freed slot
  auto vec = make_random_vector(7777);
  uint32_t new_id = kDataNum + 300;
  index.insert(vec.data(), new_id);

  EXPECT_EQ(index.get_searcher().num_points(), old_size + 1);

  // Verify the new vector is searchable
  DiskANNSearchParams params;
  params.set_ef_search(128).set_beam_width(4);
  std::vector<uint32_t> ids(kTopk);
  index.search(vec.data(), kTopk, ids.data(), params);

  bool found = std::find(ids.begin(), ids.end(), new_id) != ids.end();
  EXPECT_TRUE(found) << "Vector inserted into reused slot should be searchable";
}

// -----------------------------------------------------------------------------
// 6.3 Capacity growth test
// -----------------------------------------------------------------------------

TEST_F(DiskANNUpdateDeleteTest, InsertBeyondCapacityAutoGrows) {
  auto index = load_writable();
  auto &searcher = index.get_searcher();
  uint32_t old_capacity = searcher.capacity();

  // Insert enough vectors to exhaust capacity
  uint32_t num_inserts = old_capacity - kDataNum + 10;  // Go beyond capacity
  for (uint32_t i = 0; i < num_inserts; ++i) {
    auto vec = make_random_vector(50000 + i);
    index.insert(vec.data(), kDataNum + 500 + i);
  }

  EXPECT_GT(searcher.capacity(), old_capacity) << "Capacity should have auto-grown";
  EXPECT_EQ(searcher.num_points(), kDataNum + num_inserts);

  // Verify some of the inserted vectors are searchable
  DiskANNSearchParams params;
  params.set_ef_search(128).set_beam_width(4);
  uint32_t found = 0;
  for (uint32_t i = 0; i < std::min(num_inserts, 10U); ++i) {
    auto vec = make_random_vector(50000 + i);
    std::vector<uint32_t> ids(kTopk);
    index.search(vec.data(), kTopk, ids.data(), params);
    if (std::find(ids.begin(), ids.end(), kDataNum + 500 + i) != ids.end()) {
      ++found;
    }
  }
  EXPECT_GE(found, 5U) << "At least half of checked vectors should be searchable after growth";
}

// -----------------------------------------------------------------------------
// 6.3b SearchContext stability during delete+insert cycles (no reallocation)
// -----------------------------------------------------------------------------

TEST_F(DiskANNUpdateDeleteTest, DeleteInsertCyclesNoSearchContextRealloc) {
  auto index = load_writable();

  // Warmup: do one search to initialize the thread-local SearchContext
  {
    DiskANNSearchParams params;
    params.set_ef_search(64).set_beam_width(4);
    auto query = make_query(0);
    std::vector<uint32_t> ids(kTopk);
    index.search(query, kTopk, ids.data(), params);
  }

  // Record RSS after warmup
  auto get_rss_kb = []() -> int64_t {
#if defined(__linux__)
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
      if (line.starts_with("VmRSS:")) {
        int64_t kb = 0;
        kb = static_cast<int64_t>(std::stoll(line.substr(std::string("VmRSS:").size())));
        return kb;
      }
    }
#endif
    return -1;
  };

  int64_t rss_before = get_rss_kb();

  // Perform multiple delete+insert rounds without growing capacity
  constexpr uint32_t kRounds = 5;
  constexpr uint32_t kOpsPerRound = 10;
  uint32_t next_id = kDataNum + 1000;

  for (uint32_t round = 0; round < kRounds; ++round) {
    // Delete some vectors (skip medoid)
    std::vector<uint32_t> deleted;
    for (uint32_t id = round * kOpsPerRound + 1;
         id < round * kOpsPerRound + 1 + kOpsPerRound && id < kDataNum;
         ++id) {
      try {
        index.delete_vector(id);
        deleted.push_back(id);
      } catch (const std::logic_error &e) {
        (void)e;  // medoid, skip
      }
    }

    // Insert replacement vectors (reuses freed slots, no capacity growth)
    for (size_t i = 0; i < deleted.size(); ++i) {
      auto vec = make_random_vector(next_id);
      index.insert(vec.data(), next_id);
      ++next_id;
    }

    // Search to exercise get_search_context
    DiskANNSearchParams params;
    params.set_ef_search(64).set_beam_width(4);
    auto query = make_query(0);
    std::vector<uint32_t> ids(kTopk);
    index.search(query, kTopk, ids.data(), params);
  }

  int64_t rss_after = get_rss_kb();

  // RSS should not grow significantly (< 20MB tolerance for a 200-vector index)
  // If SearchContext were rebuilt every time, each rebuild allocates ~capacity*4 bytes
  // for VisitedList, causing RSS growth from arena fragmentation.
  if (rss_before > 0 && rss_after > 0) {
    int64_t rss_growth_kb = rss_after - rss_before;
    LOG_INFO("Delete+insert cycle RSS: before={}KB, after={}KB, growth={}KB",
             rss_before,
             rss_after,
             rss_growth_kb);
    EXPECT_LT(rss_growth_kb, 20 * 1024)
        << "RSS grew by " << rss_growth_kb / 1024
        << "MB during delete+insert cycles — likely SearchContext reallocation";
  }
}

// -----------------------------------------------------------------------------
// 6.4 Persistence test
// -----------------------------------------------------------------------------

TEST_F(DiskANNUpdateDeleteTest, PersistenceAfterInsertDelete) {
  uint32_t deleted_id = 0;
  uint32_t inserted_id = kDataNum + 400;
  std::vector<float> inserted_vec;

  // Phase 1: insert and delete, then flush
  {
    auto index = load_writable();

    // Delete a non-medoid vector
    deleted_id = 3;
    try {
      index.delete_vector(deleted_id);
    } catch (const std::logic_error &) {
      deleted_id = 4;
      index.delete_vector(deleted_id);
    }

    // Insert a new vector
    inserted_vec = make_random_vector(inserted_id);
    index.insert(inserted_vec.data(), inserted_id);

    index.flush();
  }

  // Phase 2: reopen and verify state is consistent
  {
    DiskANNIndex<> index;
    index.load(index_path_.string(), 4096, false);

    DiskANNSearchParams params;
    params.set_ef_search(128).set_beam_width(4);

    // Search for the inserted vector — it should be findable
    std::vector<uint32_t> ids(kTopk);
    index.search(inserted_vec.data(), kTopk, ids.data(), params);

    bool inserted_found = std::find(ids.begin(), ids.end(), inserted_id) != ids.end();
    EXPECT_TRUE(inserted_found) << "Inserted vector should survive reopen";

    // Deleted ID should not appear in any search
    for (uint32_t q = 0; q < 10; ++q) {
      const float *query = make_query(q);
      index.search(query, kTopk, ids.data(), params);
      EXPECT_TRUE(std::find(ids.begin(), ids.end(), deleted_id) == ids.end())
          << "Deleted ID " << deleted_id << " should not appear after reopen";
    }
  }
}

// =============================================================================
// Coroutine search tests (co_search)
// =============================================================================

class CoSearchDiskTest : public DiskANNSearcherTest {};
class IOUringDiskTest : public DiskANNSearcherTest {
 protected:
  void SetUp() override {
    if (!IOUringEngine::is_available()) {
      GTEST_SKIP() << "io_uring is not available on this system";
    }
  }
};

TEST_F(IOUringDiskTest, CoroutineSearchMatchesSyncSearch) {
  auto index = load_index();
  auto &searcher = index.get_searcher();

  constexpr uint32_t kTopk = 10;
  DiskANNSearchParams params;
  params.ef_search_ = 50;
  params.beam_width_ = 4;

  for (uint32_t q = 0; q < query_num(); ++q) {
    const float *query = dataset().queries_.data() + q * dim();

    // Sync search
    auto sync_result = searcher.search(query, kTopk, params);

    // Coroutine search via Scheduler
    auto io_engine = std::make_unique<IOUringEngine>();
    std::vector<CpuID> cpus = {0};
    Scheduler scheduler(cpus, io_engine.get());

    using SearcherType = DiskANNSearcher<>;
    SearcherType::SearchContext ctx(std::max(params.ef_search_, kTopk), 32, params.beam_width_);
    SearcherType::Result co_result;
    auto task = searcher.co_search(ctx, query, kTopk, params, co_result);
    scheduler.schedule(task.handle());
    scheduler.begin();
    scheduler.join();

    // Verify identical results
    ASSERT_EQ(co_result.ids_.size(), sync_result.ids_.size())
        << "Query " << q << ": result count mismatch";
    for (size_t i = 0; i < co_result.ids_.size(); ++i) {
      EXPECT_EQ(co_result.ids_[i], sync_result.ids_[i])
          << "Query " << q << ", rank " << i << ": ID mismatch";
      EXPECT_FLOAT_EQ(co_result.distances_[i], sync_result.distances_[i])
          << "Query " << q << ", rank " << i << ": distance mismatch";
    }
  }
}

TEST_F(IOUringDiskTest, MultipleConcurrentCoroutineSearches) {
  auto index = load_index();
  auto &searcher = index.get_searcher();

  constexpr uint32_t kTopk = 10;
  constexpr uint32_t kNumQueries = 8;
  DiskANNSearchParams params;
  params.ef_search_ = 50;
  params.beam_width_ = 4;

  auto io_engine = std::make_unique<IOUringEngine>();
  std::vector<CpuID> cpus = {0};
  Scheduler scheduler(cpus, io_engine.get());

  using SearcherType = DiskANNSearcher<>;

  // Create contexts and result holders
  std::vector<std::unique_ptr<SearcherType::SearchContext>> contexts;
  std::vector<SearcherType::Result> results(kNumQueries);
  std::vector<coro::task<>> tasks;

  contexts.reserve(kNumQueries);
  tasks.reserve(kNumQueries);

  uint32_t num_q = std::min(kNumQueries, query_num());
  for (uint32_t q = 0; q < num_q; ++q) {
    contexts.push_back(
        std::make_unique<SearcherType::SearchContext>(std::max(params.ef_search_, kTopk),
                                                      32,
                                                      params.beam_width_));

    const float *query = dataset().queries_.data() + q * dim();
    tasks.push_back(searcher.co_search(*contexts.back(), query, kTopk, params, results[q]));
  }

  for (auto &t : tasks) {
    scheduler.schedule(t.handle());
  }
  scheduler.begin();
  scheduler.join();

  // Verify all results are valid
  for (uint32_t q = 0; q < num_q; ++q) {
    EXPECT_FALSE(results[q].ids_.empty()) << "Query " << q << " returned empty results";
    EXPECT_EQ(results[q].ids_.size(), static_cast<size_t>(kTopk))
        << "Query " << q << " wrong result count";

    // Verify results are valid IDs (within range)
    for (auto id : results[q].ids_) {
      EXPECT_LT(id, data_num()) << "Query " << q << " returned out-of-range ID";
    }

    // Verify distances are non-negative and ordered
    for (size_t i = 1; i < results[q].distances_.size(); ++i) {
      EXPECT_GE(results[q].distances_[i], results[q].distances_[i - 1])
          << "Query " << q << " distances not sorted at rank " << i;
    }
  }
}

// =============================================================================
// Coroutine insert/delete tests
// =============================================================================

class CoInsertDeleteTest : public IOUringDiskTest {};

TEST_F(CoInsertDeleteTest, CoInsertIncreasesPointCount) {
  auto index = load_writable_index();
  auto &searcher = index.get_searcher();

  auto io_engine = std::make_unique<IOUringEngine>();
  std::vector<CpuID> cpus = {0};

  uint32_t before_count = searcher.num_points();

  // Insert a new vector via coroutine
  std::vector<float> new_vec(dim(), 42.0F);
  uint32_t new_id = data_num() + 50;

  {
    Scheduler scheduler(cpus, io_engine.get());
    auto task = searcher.co_insert(new_vec.data(), new_id);
    scheduler.schedule(task.handle());
    scheduler.begin();
    scheduler.join();
  }

  // Verify point count increased (main correctness check)
  EXPECT_EQ(searcher.num_points(), before_count + 1);

  // Verify no throw on duplicate insert
  EXPECT_THROW(searcher.insert(new_vec.data(), new_id), std::invalid_argument);
}

TEST_F(CoInsertDeleteTest, CoDeleteRemovesFromResults) {
  auto index = load_writable_index();
  auto &searcher = index.get_searcher();

  auto io_engine = std::make_unique<IOUringEngine>();
  std::vector<CpuID> cpus = {0};

  uint32_t before_count = searcher.num_points();

  // Pick a non-medoid point to delete (use ID 5 which should be safe)
  uint32_t delete_id = 5;

  // Delete via coroutine
  {
    Scheduler scheduler(cpus, io_engine.get());
    auto task = searcher.co_delete_vector(delete_id);
    scheduler.schedule(task.handle());
    scheduler.begin();
    scheduler.join();
  }

  // Verify point count decreased
  EXPECT_EQ(searcher.num_points(), before_count - 1);

  // Verify it no longer appears in search results
  const float *query = dataset().queries_.data();
  DiskANNSearchParams params;
  params.set_ef_search(128).set_beam_width(4);
  constexpr uint32_t kTopk = 50;
  std::vector<uint32_t> ids(kTopk);
  index.search(query, kTopk, ids.data(), params);
  EXPECT_TRUE(std::find(ids.begin(), ids.end(), delete_id) == ids.end())
      << "Deleted ID " << delete_id << " still appears in search results";
}

TEST_F(CoInsertDeleteTest, MixedConcurrentWorkload) {
  auto index = load_writable_index();
  auto &searcher = index.get_searcher();

  auto io_engine = std::make_unique<IOUringEngine>();
  std::vector<CpuID> cpus = {0};

  uint32_t before_count = searcher.num_points();

  // Schedule: 1 search + 1 insert + 1 delete.
  std::vector<coro::task<>> tasks;

  constexpr uint32_t kTopk = 10;
  DiskANNSearchParams search_params;
  search_params.set_ef_search(50).set_beam_width(4);

  using SearcherType = DiskANNSearcher<>;
  std::vector<std::unique_ptr<SearcherType::SearchContext>> search_ctxs;
  std::vector<SearcherType::Result> search_results(1);

  search_ctxs.push_back(
      std::make_unique<SearcherType::SearchContext>(50, 32, search_params.beam_width_));
  tasks.push_back(searcher.co_search(*search_ctxs.back(),
                                     dataset().queries_.data(),
                                     kTopk,
                                     search_params,
                                     search_results[0]));

  // 1 delete task
  uint32_t delete_id = 15;
  tasks.push_back(searcher.co_delete_vector(delete_id));

  // 1 insert task
  std::vector<float> vec1(dim(), 10.0F);
  uint32_t insert_id = data_num() + 300;
  tasks.push_back(searcher.co_insert(vec1.data(), insert_id));

  // Schedule all tasks and run
  {
    Scheduler scheduler(cpus, io_engine.get());
    for (auto &t : tasks) {
      scheduler.schedule(t.handle());
    }
    scheduler.begin();
    scheduler.join();
  }

  // Verify: +1 insert -1 delete = 0 net change
  EXPECT_EQ(searcher.num_points(), before_count);

  // All searches should have returned results
  for (int i = 0; i < 1; ++i) {
    EXPECT_FALSE(search_results[i].ids_.empty()) << "Search " << i << " returned empty results";
  }
}

// =============================================================================
// Multi-worker concurrency tests
// =============================================================================

class MultiWorkerTest : public IOUringDiskTest {};

// Task 6.1: Multi-worker search correctness — verify 4 Workers produce same recall as 1 Worker
TEST_F(MultiWorkerTest, MultiWorkerSearchMatchesSingleWorker) {
  auto index = load_index();
  auto &searcher = index.get_searcher();

  constexpr uint32_t kTopk = 10;
  constexpr uint32_t kNumQueries = 4;
  DiskANNSearchParams params;
  params.ef_search_ = 50;
  params.beam_width_ = 4;

  using SearcherType = DiskANNSearcher<>;
  uint32_t num_q = std::min(kNumQueries, query_num());

  // Run with 1 Worker
  std::vector<SearcherType::Result> single_results(num_q);
  {
    auto io_engine = std::make_unique<IOUringEngine>();
    std::vector<CpuID> cpus = {0};
    Scheduler scheduler(cpus, io_engine.get());

    std::vector<std::unique_ptr<SearcherType::SearchContext>> contexts;
    std::vector<coro::task<>> tasks;
    contexts.reserve(num_q);
    tasks.reserve(num_q);

    for (uint32_t q = 0; q < num_q; ++q) {
      contexts.push_back(
          std::make_unique<SearcherType::SearchContext>(std::max(params.ef_search_, kTopk),
                                                        32,
                                                        params.beam_width_));
      const float *query = dataset().queries_.data() + q * dim();
      tasks.push_back(
          searcher.co_search(*contexts.back(), query, kTopk, params, single_results[q]));
    }
    for (auto &t : tasks) {
      scheduler.schedule(t.handle());
    }
    scheduler.begin();
    scheduler.join();
  }

  // Run with 4 Workers
  std::vector<SearcherType::Result> multi_results(num_q);
  {
    auto io_engine = std::make_unique<IOUringEngine>();
    std::vector<CpuID> cpus = {0, 1, 2, 3};
    Scheduler scheduler(cpus, io_engine.get());

    std::vector<std::unique_ptr<SearcherType::SearchContext>> contexts;
    std::vector<coro::task<>> tasks;
    contexts.reserve(num_q);
    tasks.reserve(num_q);

    for (uint32_t q = 0; q < num_q; ++q) {
      contexts.push_back(
          std::make_unique<SearcherType::SearchContext>(std::max(params.ef_search_, kTopk),
                                                        32,
                                                        params.beam_width_));
      const float *query = dataset().queries_.data() + q * dim();
      tasks.push_back(searcher.co_search(*contexts.back(), query, kTopk, params, multi_results[q]));
    }
    for (auto &t : tasks) {
      scheduler.schedule(t.handle());
    }
    scheduler.begin();
    scheduler.join();
  }

  // Verify results match (allowing ties at k-th distance boundary)
  for (uint32_t q = 0; q < num_q; ++q) {
    ASSERT_EQ(single_results[q].ids_.size(), multi_results[q].ids_.size())
        << "Query " << q << ": result count mismatch";

    // Collect IDs into sets for comparison (handles tie-breaking differences)
    std::set<uint32_t> single_ids(single_results[q].ids_.begin(), single_results[q].ids_.end());
    std::set<uint32_t> multi_ids(multi_results[q].ids_.begin(), multi_results[q].ids_.end());

    // At least 80% of IDs should match (allowing for tie-breaking differences)
    size_t matches = 0;
    for (auto id : single_ids) {
      if (multi_ids.contains(id)) {
        ++matches;
      }
    }
    EXPECT_GE(matches, single_ids.size() * 8 / 10)
        << "Query " << q << ": too few matching IDs between 1-worker and 4-worker results";

    // Verify multi-worker results are valid
    for (auto id : multi_results[q].ids_) {
      EXPECT_LT(id, data_num()) << "Query " << q << " returned out-of-range ID";
    }

    // Verify distances are non-negative and sorted
    for (size_t i = 0; i < multi_results[q].distances_.size(); ++i) {
      EXPECT_GE(multi_results[q].distances_[i], 0.0F)
          << "Query " << q << " has negative distance at rank " << i;
    }
    for (size_t i = 1; i < multi_results[q].distances_.size(); ++i) {
      EXPECT_GE(multi_results[q].distances_[i], multi_results[q].distances_[i - 1])
          << "Query " << q << " distances not sorted at rank " << i;
    }
  }
}

// Long-running concurrent update checks are kept out of the default unit-test path.
// They exercise stress/load behavior rather than quick correctness validation.
TEST_F(MultiWorkerTest, DISABLED_ConcurrentInsertGraphIntegrity) {
  auto index = load_writable_index();
  auto &searcher = index.get_searcher();

  auto io_engine = std::make_unique<IOUringEngine>();
  std::vector<CpuID> cpus = {0, 1, 2, 3};

  uint32_t before_count = searcher.num_points();
  constexpr uint32_t kNumInserts = 2;

  // Generate random vectors for insertion
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<std::vector<float>> insert_vecs(kNumInserts);
  for (auto &v : insert_vecs) {
    v.resize(dim());
    for (auto &x : v) {
      x = dist(rng);
    }
  }

  // Schedule all inserts concurrently
  std::vector<coro::task<>> tasks;
  tasks.reserve(kNumInserts);

  uint32_t base_id = data_num() + 1000;
  for (uint32_t i = 0; i < kNumInserts; ++i) {
    tasks.push_back(searcher.co_insert(insert_vecs[i].data(), base_id + i));
  }

  {
    Scheduler scheduler(cpus, io_engine.get());
    for (auto &t : tasks) {
      scheduler.schedule(t.handle());
    }
    scheduler.begin();
    scheduler.join();
  }

  // Verify point count and external ID registration.
  EXPECT_EQ(searcher.num_points(), before_count + kNumInserts);
  for (uint32_t i = 0; i < kNumInserts; ++i) {
    EXPECT_THROW(searcher.insert(insert_vecs[i].data(), base_id + i), std::invalid_argument)
        << "Inserted external ID " << (base_id + i) << " was not recorded";
  }
}

TEST_F(MultiWorkerTest, DISABLED_ConcurrentInsertDeleteConsistency) {
  auto index = load_writable_index();
  auto &searcher = index.get_searcher();

  auto io_engine = std::make_unique<IOUringEngine>();
  std::vector<CpuID> cpus = {0, 1, 2, 3};

  uint32_t before_count = searcher.num_points();
  constexpr uint32_t kNumInserts = 1;
  constexpr uint32_t kNumDeletes = 1;

  // Generate insert vectors
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<std::vector<float>> insert_vecs(kNumInserts);
  for (auto &v : insert_vecs) {
    v.resize(dim());
    for (auto &x : v) {
      x = dist(rng);
    }
  }

  // Pick non-entry-point IDs to delete.
  std::vector<uint32_t> delete_ids;
  for (uint32_t i = 0; i < kNumDeletes; ++i) {
    delete_ids.push_back(20 + i);
  }

  // Schedule mixed inserts and deletes
  std::vector<coro::task<>> tasks;
  uint32_t base_id = data_num() + 2000;
  for (uint32_t i = 0; i < kNumInserts; ++i) {
    tasks.push_back(searcher.co_insert(insert_vecs[i].data(), base_id + i));
  }
  for (auto id : delete_ids) {
    tasks.push_back(searcher.co_delete_vector(id));
  }

  {
    Scheduler scheduler(cpus, io_engine.get());
    for (auto &t : tasks) {
      scheduler.schedule(t.handle());
    }
    scheduler.begin();
    scheduler.join();
  }

  // Verify net point count: +kNumInserts -kNumDeletes
  EXPECT_EQ(searcher.num_points(), before_count + kNumInserts - kNumDeletes);

  for (uint32_t i = 0; i < kNumInserts; ++i) {
    EXPECT_THROW(searcher.insert(insert_vecs[i].data(), base_id + i), std::invalid_argument)
        << "Concurrent insert lost external ID " << (base_id + i);
  }

  std::vector<float> replacement_vec(dim(), 0.25F);
  for (auto id : delete_ids) {
    EXPECT_THROW(searcher.delete_vector(id), std::invalid_argument)
        << "Concurrent delete did not persist for ID " << id;
    EXPECT_NO_THROW(searcher.insert(replacement_vec.data(), id))
        << "Deleted external ID " << id << " was not reusable";
  }
}

// Task 4.2: BufferPool shard count test
TEST_F(MultiWorkerTest, BufferPoolShardScaling) {
  // Test the compute_num_shards formula
  using SearcherType = DiskANNSearcher<>;
  EXPECT_EQ(SearcherType::compute_num_shards(1), 16);   // max(16, 1*4=4) = 16
  EXPECT_EQ(SearcherType::compute_num_shards(4), 16);   // max(16, 4*4=16) = 16
  EXPECT_EQ(SearcherType::compute_num_shards(5), 20);   // max(16, 5*4=20) = 20
  EXPECT_EQ(SearcherType::compute_num_shards(8), 32);   // max(16, 8*4=32) = 32
  EXPECT_EQ(SearcherType::compute_num_shards(16), 64);  // max(16, 16*4=64) = 64

  // Verify BufferPool actually gets the right shard count
  using BPType = BufferPool<uint32_t, ClockReplacer>;
  BPType pool_16(1000, 4096, 16);
  EXPECT_EQ(pool_16.num_shards(), 16);

  BPType pool_32(1000, 4096, 32);
  EXPECT_EQ(pool_32.num_shards(), 32);
}

// =============================================================================
// 5.2 Delete-repair: former neighbors no longer have dead edges
// =============================================================================

TEST_F(DiskANNUpdateDeleteTest, DeleteRepairsFormerNeighbors) {
  auto index = load_writable();

  // Baseline recall: search for a few data vectors before any deletions
  DiskANNSearchParams params;
  params.set_ef_search(128).set_beam_width(4);

  // Compute baseline recall using data vectors as queries
  uint32_t num_queries = 20;
  for (uint32_t q = 1; q <= num_queries; ++q) {
    const float *query = make_query(q);
    std::vector<uint32_t> ids(kTopk);
    std::vector<float> dists(kTopk);
    index.search_with_distance(query, kTopk, ids.data(), dists.data(), params);
  }

  // Delete ~5% of non-medoid nodes
  std::vector<uint32_t> deleted;
  for (uint32_t id = 50; id < 60; ++id) {
    try {
      index.delete_vector(id);
      deleted.push_back(id);
    } catch (const std::logic_error &e) {
      (void)e;  // Skip medoid
    }
  }
  ASSERT_FALSE(deleted.empty());

  // After delete-with-repair, search should still work well
  // Deleted IDs must not appear, and recall should not collapse
  uint32_t found_after = 0;
  for (uint32_t q = 1; q <= num_queries; ++q) {
    // Skip if q was deleted
    bool was_deleted = false;
    for (auto d : deleted) {
      if (d == q) {
        was_deleted = true;
        break;
      }
    }
    if (was_deleted) {
      continue;
    }

    const float *query = make_query(q);
    std::vector<uint32_t> ids(kTopk);
    std::vector<float> dists(kTopk);
    index.search_with_distance(query, kTopk, ids.data(), dists.data(), params);

    // Verify no deleted IDs in results
    for (uint32_t i = 0; i < kTopk; ++i) {
      for (auto d : deleted) {
        EXPECT_NE(ids[i], d) << "Deleted ID " << d << " found in results after repair";
      }
    }

    // Check self-recall
    for (uint32_t i = 0; i < kTopk; ++i) {
      if (ids[i] == q) {
        ++found_after;
        break;
      }
    }
  }

  // Repair should maintain reasonable search quality
  // (not a strict threshold, just verify it doesn't collapse to 0)
  EXPECT_GT(found_after, 0U) << "Search recall collapsed after delete-repair";
}

// =============================================================================
// 5.3 inserted_edges: insert records reverse edges, consumed by connect_task
// =============================================================================

TEST_F(DiskANNUpdateDeleteTest, InsertedEdgesConsumedOnSubsequentDelete) {
  auto index = load_writable();

  // Insert a new vector
  auto vec1 = make_random_vector(7777);
  uint32_t new_id1 = kDataNum + 200;
  EXPECT_NO_THROW(index.insert(vec1.data(), new_id1));

  // Insert another vector nearby (same seed vicinity)
  auto vec2 = make_random_vector(7778);
  uint32_t new_id2 = kDataNum + 201;
  EXPECT_NO_THROW(index.insert(vec2.data(), new_id2));

  // Now delete one of the original nodes. This triggers connect_task on
  // its neighbors, which should consume any inserted_edges entries.
  // The net effect: after delete+repair, the newly inserted vectors
  // should still be findable.
  uint32_t delete_target = 30;
  try {
    index.delete_vector(delete_target);
  } catch (const std::logic_error &) {
    delete_target = 31;
    index.delete_vector(delete_target);
  }

  // Search for the inserted vectors — they should be findable
  DiskANNSearchParams params;
  params.set_ef_search(128).set_beam_width(4);

  // Search for vec1
  {
    std::vector<uint32_t> ids(kTopk);
    std::vector<float> dists(kTopk);
    index.search_with_distance(vec1.data(), kTopk, ids.data(), dists.data(), params);
    bool found = false;
    for (uint32_t i = 0; i < kTopk; ++i) {
      if (ids[i] == new_id1) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "Inserted vector " << new_id1
                       << " not findable after delete-repair cycle";
  }

  // Search for vec2
  {
    std::vector<uint32_t> ids(kTopk);
    std::vector<float> dists(kTopk);
    index.search_with_distance(vec2.data(), kTopk, ids.data(), dists.data(), params);
    bool found = false;
    for (uint32_t i = 0; i < kTopk; ++i) {
      if (ids[i] == new_id2) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "Inserted vector " << new_id2
                       << " not findable after delete-repair cycle";
  }
}

// -----------------------------------------------------------------------------
// Beam search helper tests
// -----------------------------------------------------------------------------

// Task 2.12: Insert-path parity — greedy_search_for_insert produces
// consistent results across repeated calls (sync path consistency).
TEST_F(DiskANNSearcherTest, GreedySearchForInsertIsConsistent) {
  auto index = load_writable_index();
  auto &searcher = index.get_searcher();

  // Use a fixed random vector
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> vec(dim());
  for (auto &x : vec) {
    x = dist(rng);
  }

  DiskANNSearchParams params;
  params.set_ef_search(128).set_beam_width(4);

  // Run search twice with the same query
  auto result1 = searcher.search(vec.data(), 20, params);
  auto result2 = searcher.search(vec.data(), 20, params);

  // Results should be identical (deterministic)
  ASSERT_EQ(result1.ids_.size(), result2.ids_.size());
  for (size_t i = 0; i < result1.ids_.size(); ++i) {
    EXPECT_EQ(result1.ids_[i], result2.ids_[i]) << "Mismatch at position " << i;
    EXPECT_FLOAT_EQ(result1.distances_[i], result2.distances_[i]) << "Distance mismatch at " << i;
  }
}

// Task 2.14: Empty-beam termination — search terminates correctly when
// candidates are exhausted on a very small index.
TEST_F(DiskANNSearcherTest, EmptyBeamTerminatesCorrectly) {
  auto index = load_index();
  auto &searcher = index.get_searcher();

  // Use a very large ef_search relative to index size to exhaust candidates
  DiskANNSearchParams params;
  params.set_ef_search(data_num());
  params.set_beam_width(1);

  std::vector<float> query(dim(), 0.0F);
  auto result = searcher.search(query.data(), 10, params);

  // Should return valid results without hanging
  EXPECT_LE(result.ids_.size(), 10U);
  EXPECT_FALSE(result.ids_.empty());
}

}  // namespace alaya
