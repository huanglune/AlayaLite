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
#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "index/graph/diskann/diskann_index.hpp"
#include "index/graph/diskann/diskann_params.hpp"
#include "index/graph/diskann/diskann_searcher.hpp"
#include "space/raw_space.hpp"
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
  uint32_t max_threads_{std::max(1U, std::min(std::thread::hardware_concurrency(), 60U))};

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
    tmp_dir_ =
        std::filesystem::temp_directory_path() / ("diskann_searcher_test_" + unique_suffix);
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

    DiskANNIndex<>::build_graph(space_, index_path_.string(), params);
    index_built_ = true;
    LOG_INFO("SearcherTestResources: Index built at {}", index_path_.string());
  }

 private:
  static constexpr uint32_t kDataNum = 500;
  static constexpr uint32_t kQueryNum = 20;
  static constexpr uint32_t kDim = 32;
  static constexpr uint32_t kGtTopk = 50;
  static constexpr uint32_t kDefaultR = 32;
  static constexpr uint32_t kDefaultEf = 64;
};

static SearcherTestResources g_res;  // NOLINT(cert-err58-cpp)

// =============================================================================
// Fixture: DiskANN Searcher Tests (disk-only, no PQ)
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
    return calc_recall(results, dataset().ground_truth_.data(), query_num(), dataset().gt_dim_, topk);
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

TEST_F(DiskANNSearcherTest, InsertFailsFastInsteadOfSilentlySucceeding) {
  auto index = load_writable_index();
  std::vector<float> vec(dim(), 1.0F);

  EXPECT_THROW(index.insert(vec.data(), data_num() + 1), std::logic_error);
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

  EXPECT_GE(recall, 0.5F) << "Recall too low for R=32, ef=64 on 500 random vectors";
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
                    .set_pq_rerank(true, 6)
                    .set_cache_capacity(8192)
                    .set_pipeline_width(32)
                    .set_num_threads(4);

  EXPECT_EQ(params.ef_search_, 128U);
  EXPECT_EQ(params.beam_width_, 8U);
  EXPECT_TRUE(params.use_pq_rerank_);
  EXPECT_EQ(params.pq_rerank_factor_, 6U);
  EXPECT_EQ(params.cache_capacity_, 8192U);
  EXPECT_EQ(params.pipeline_width_, 32U);
  EXPECT_EQ(params.num_threads_, 4U);
}

TEST_F(DiskANNSearcherTest, SearchParamsDefault) {
  DiskANNSearchParams params;

  EXPECT_EQ(params.ef_search_, 64U);
  EXPECT_EQ(params.beam_width_, 4U);
  EXPECT_FALSE(params.use_pq_rerank_);
  EXPECT_EQ(params.pq_rerank_factor_, 4U);
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

  auto tmp_dir =
      std::filesystem::temp_directory_path() / "diskann_searcher_beam_width_one_test";
  std::filesystem::remove_all(tmp_dir);
  std::filesystem::create_directories(tmp_dir);
  auto small_index_path = tmp_dir / "test_idx";

  auto build_params = DiskANNBuildParams()
                          .set_max_degree(8)
                          .set_ef_construction(16)
                          .set_num_iterations(1)
                          .set_num_threads(std::min(max_threads(), 4U));
  DiskANNIndex<>::build_graph(small_space, small_index_path.string(), build_params);

  DiskANNIndex<> index;
  index.load(small_index_path.string());

  DiskANNSearchParams search_params;
  search_params.set_beam_width(1).set_ef_search(8);

  std::vector<uint32_t> results(kTopk, static_cast<uint32_t>(-1));
  EXPECT_NO_THROW(
      index.search(dataset().queries_.data(), kTopk, results.data(), search_params));
  EXPECT_EQ(std::count_if(results.begin(),
                          results.end(),
                          [](uint32_t id) { return id != static_cast<uint32_t>(-1); }),
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

  auto tmp_dir =
      std::filesystem::temp_directory_path() / "diskann_searcher_ef_search_one_test";
  std::filesystem::remove_all(tmp_dir);
  std::filesystem::create_directories(tmp_dir);
  auto small_index_path = tmp_dir / "test_idx";

  auto build_params = DiskANNBuildParams()
                          .set_max_degree(8)
                          .set_ef_construction(16)
                          .set_num_iterations(1)
                          .set_num_threads(std::min(max_threads(), 4U));
  DiskANNIndex<>::build_graph(small_space, small_index_path.string(), build_params);

  DiskANNIndex<> index;
  index.load(small_index_path.string());

  DiskANNSearchParams search_params;
  search_params.set_ef_search(1).set_beam_width(4);

  std::vector<uint32_t> results(kTopk, static_cast<uint32_t>(-1));
  EXPECT_NO_THROW(
      index.search(dataset().queries_.data(), kTopk, results.data(), search_params));

  index.close();
  std::filesystem::remove_all(tmp_dir);
}

}  // namespace alaya
