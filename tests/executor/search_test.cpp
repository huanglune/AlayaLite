// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include "index/graph/detail/search_runtime/graph_search_job.hpp"
#include "index/graph/graph.hpp"
#include "index/graph/hnsw/detail/hnsw_segment_bridge.hpp"
#include "index/graph/hnsw/hnsw_segment.hpp"
#include "index/graph/qg/detail/qg_builder_kernel.hpp"
#include "space/rabitq_space.hpp"
#include "space/raw_space.hpp"
#include "space/sq8_space.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"
#include "utils/locks.hpp"
#include "utils/log.hpp"
#include "utils/metadata_filter.hpp"
#include "utils/scalar_data.hpp"
#include "utils/thread_config.hpp"
#include "utils/timer.hpp"

namespace alaya {
namespace {

constexpr size_t kHnswM = 64;
constexpr uint32_t kDefaultTopk = 10;
constexpr uint32_t kDefaultEf = 100;
constexpr uint32_t kSparseTailCount = 50;

using RawSpaceType = RawSpace<>;
using GraphType = Graph<>;
using SQ8SpaceType = SQ8Space<>;
using RawSpaceWithScalarType =
    RawSpace<float, float, uint32_t, SequentialStorage<float, uint32_t>, ScalarData>;
using SQ8SpaceWithScalar =
    SQ8Space<float, float, uint32_t, SequentialStorage<uint8_t, uint32_t>, ScalarData>;

auto test_data_dir() -> std::filesystem::path {
  return std::filesystem::current_path().parent_path() / "data";
}

auto test_cache_dir() -> const std::filesystem::path & {
  static const auto dir = [] {
    auto cache_dir = std::filesystem::current_path() / "search_test_cache";
    std::filesystem::create_directories(cache_dir);
    return cache_dir;
  }();
  return dir;
}

void remove_path_if_exists(const std::filesystem::path &path) {
  if (std::filesystem::exists(path)) {
    std::filesystem::remove_all(path);
  }
}

struct ScopedTempDbDir {
  explicit ScopedTempDbDir(std::string name) : path_(test_cache_dir() / std::move(name)) {
    remove_path_if_exists(path_);
  }

  ~ScopedTempDbDir() { remove_path_if_exists(path_); }

  std::filesystem::path path_;
};

auto max_thread_num() -> uint32_t { return configured_thread_limit(); }

auto make_test_metadata(uint32_t item_cnt) -> std::vector<ScalarData> {
  std::vector<ScalarData> metadata(item_cnt);
  for (uint32_t i = 0; i < item_cnt; ++i) {
    MetadataMap meta;
    meta["id"] = static_cast<int64_t>(i);
    meta["category"] = static_cast<int64_t>(i % 5);
    meta["score"] = static_cast<double>(i) * 10.0;
    meta["name"] = std::string("item_") + std::to_string(i);
    metadata[i] = ScalarData("id_" + std::to_string(i), "doc_" + std::to_string(i), meta);
  }
  return metadata;
}

auto make_raw_space(const Dataset &ds) -> std::shared_ptr<RawSpaceType> {
  auto space = std::make_shared<RawSpaceType>(ds.data_num_, ds.dim_, MetricType::L2);
  space->fit(ds.data_.data(), ds.data_num_);
  return space;
}

auto make_sq8_space(const Dataset &ds) -> std::shared_ptr<SQ8SpaceType> {
  auto space = std::make_shared<SQ8SpaceType>(ds.data_num_, ds.dim_, MetricType::L2);
  space->fit(ds.data_.data(), ds.data_num_);
  return space;
}

auto make_one_dim_raw_space(const std::vector<float> &values) -> std::shared_ptr<RawSpaceType> {
  auto space =
      std::make_shared<RawSpaceType>(static_cast<uint32_t>(values.size()), 1, MetricType::L2);
  space->fit(values.data(), static_cast<uint32_t>(values.size()));
  return space;
}

auto make_one_dim_sq8_space(const std::vector<float> &values) -> std::shared_ptr<SQ8SpaceType> {
  auto space =
      std::make_shared<SQ8SpaceType>(static_cast<uint32_t>(values.size()), 1, MetricType::L2);
  space->fit(values.data(), static_cast<uint32_t>(values.size()));
  return space;
}

auto make_one_dim_scalar_space(const std::vector<float> &values,
                               const std::filesystem::path &db_path,
                               const std::vector<std::string> &indexed_fields)
    -> std::shared_ptr<RawSpaceWithScalarType> {
  RocksDBConfig config;
  config.db_path_ = db_path.string();
  config.indexed_fields_ = indexed_fields;

  auto space = std::make_shared<RawSpaceWithScalarType>(static_cast<uint32_t>(values.size()),
                                                        1,
                                                        MetricType::L2,
                                                        config);

  std::vector<ScalarData> metadata;
  metadata.reserve(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    MetadataMap meta;
    meta["id"] = static_cast<int64_t>(i);
    meta["group"] = static_cast<int64_t>(i >= 3 ? 1 : 0);
    metadata.emplace_back("id_" + std::to_string(i), "doc_" + std::to_string(i), std::move(meta));
  }

  space->fit(values.data(), static_cast<uint32_t>(values.size()), metadata.data());
  return space;
}

auto make_graph_from_edges(const std::vector<std::vector<uint32_t>> &adjacency)
    -> std::shared_ptr<GraphType> {
  uint32_t max_nbrs = 1;
  for (const auto &neighbors : adjacency) {
    max_nbrs = std::max<uint32_t>(max_nbrs, static_cast<uint32_t>(neighbors.size()));
  }

  auto graph = std::make_shared<GraphType>(static_cast<uint32_t>(adjacency.size()), max_nbrs);
  graph->eps_.push_back(0);
  for (size_t node = 0; node < adjacency.size(); ++node) {
    for (size_t idx = 0; idx < adjacency[node].size(); ++idx) {
      graph->at(static_cast<uint32_t>(node), static_cast<uint32_t>(idx)) = adjacency[node][idx];
    }
  }
  return graph;
}

auto selective_score_threshold(const Dataset &ds) -> double {
  auto tail_count = std::min<uint32_t>(kSparseTailCount, ds.data_num_);
  auto cutoff = ds.data_num_ > tail_count ? ds.data_num_ - tail_count : ds.data_num_ / 2;
  return static_cast<double>(cutoff) * 10.0;
}

auto sparse_id_threshold(const Dataset &ds) -> int64_t {
  auto tail_count = std::min<uint32_t>(kSparseTailCount, ds.data_num_);
  if (tail_count >= ds.data_num_) {
    return 0;
  }
  return static_cast<int64_t>(ds.data_num_ - tail_count);
}

auto load_or_build_hnsw_graph(const Dataset &ds, const std::shared_ptr<RawSpaceType> &space)
    -> std::shared_ptr<GraphType> {
  auto graph_path = test_cache_dir() / fmt::format("{}_M{}.HNSW", ds.name_, kHnswM);
  auto lock_path = graph_path;
  lock_path += ".lock";
  auto graph_path_str = graph_path.string();

  {
    FileLock lock(lock_path);
    if (!std::filesystem::exists(graph_path)) {
      auto build_start = std::chrono::steady_clock::now();

      core::BuildContext build_context;
      auto segment =
          HnswSegment<RawSpaceType>::build({core::TypedTensorView::contiguous(ds.data_.data(),
                                                                              ds.data_num_,
                                                                              ds.dim_),
                                            space,
                                            space},
                                           {.thread_count = max_thread_num()},
                                           build_context);
      auto hnsw_graph = detail::HnswSegmentBridge<RawSpaceType, RawSpaceType>::graph(*segment);

      auto build_end = std::chrono::steady_clock::now();
      auto build_time = static_cast<std::chrono::duration<double>>(build_end - build_start).count();
      LOG_INFO("Building cached HNSW graph took {}s, saving to {}", build_time, graph_path_str);

      auto tmp_graph_path = graph_path;
      tmp_graph_path += ".tmp";
      remove_path_if_exists(tmp_graph_path);
      auto tmp_graph_path_str = tmp_graph_path.string();
      hnsw_graph->save(tmp_graph_path_str);
      std::filesystem::rename(tmp_graph_path, graph_path);
    }
  }

  auto graph = std::make_shared<GraphType>(ds.data_num_, kHnswM);
  graph->load(graph_path_str);
  return graph;
}

template <typename SearchJobPtr>
auto run_parallel_search(SearchJobPtr &search_job, Dataset &ds, uint32_t topk, uint32_t ef)
    -> std::vector<std::vector<uint32_t>> {
  Timer timer{};
  std::vector<std::vector<uint32_t>> res_pool(ds.query_num_, std::vector<uint32_t>(topk));
  const size_t search_thread_num =
      std::min<size_t>(cap_thread_count(16),
                       std::max<size_t>(1, static_cast<size_t>(ds.query_num_)));
  std::vector<std::thread> tasks(search_thread_num);

  auto search_knn = [&](uint32_t i) {
    for (; i < ds.query_num_; i += search_thread_num) {
      std::vector<uint32_t> ids(topk);
      auto cur_query = ds.queries_.data() + i * ds.dim_;
      search_job->search_solo(cur_query, ids.data(), topk, ef);

      auto id_set = std::set(ids.begin(), ids.end());
      if (id_set.size() < topk) {
        fmt::println("query {} has duplicated ids, unique size {}", i, id_set.size());
      }
      res_pool[i] = std::move(ids);
    }
  };

  for (size_t i = 0; i < search_thread_num; ++i) {
    tasks[i] = std::thread(search_knn, static_cast<uint32_t>(i));
  }

  for (auto &task : tasks) {
    if (task.joinable()) {
      task.join();
    }
  }

  LOG_INFO("total time: {} s.", timer.elapsed() / 1000000.0);
  return res_pool;
}

struct BaseSearchResources {
  Dataset ds_;
  std::shared_ptr<RawSpaceType> raw_space_;
  std::shared_ptr<GraphType> hnsw_graph_;

  BaseSearchResources()
      : ds_(load_dataset(sift_micro(test_data_dir()))),
        raw_space_(make_raw_space(ds_)),
        hnsw_graph_(load_or_build_hnsw_graph(ds_, raw_space_)) {}
};

auto base_search_resources() -> BaseSearchResources & {
  static BaseSearchResources resources;
  return resources;
}


class SearchTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { sq_space_ = make_sq8_space(base_search_resources().ds_); }

  static void TearDownTestSuite() { sq_space_.reset(); }

  static auto ds() -> Dataset & { return base_search_resources().ds_; }

  static auto raw_space() -> const std::shared_ptr<RawSpaceType> & {
    return base_search_resources().raw_space_;
  }

  static auto graph() -> const std::shared_ptr<GraphType> & {
    return base_search_resources().hnsw_graph_;
  }

  static auto sq_space() -> const std::shared_ptr<SQ8SpaceType> & { return sq_space_; }

  static inline std::shared_ptr<SQ8SpaceType> sq_space_ = nullptr;
};

TEST_F(SearchTest, FullGraphTest) {
  auto &dataset = ds();
  const auto &load_graph = *graph();

  std::vector<uint32_t> inpoint_num(dataset.data_num_);
  std::vector<uint32_t> outpoint_num(dataset.data_num_);

  for (uint32_t i = 0; i < dataset.data_num_; ++i) {
    for (uint32_t j = 0; j < load_graph.max_nbrs_; ++j) {
      auto id = load_graph.at(i, j);
      if (id == Graph<uint32_t>::kEmptyId) {
        break;
      }
      outpoint_num[i]++;
      inpoint_num[id]++;
    }
  }

  uint64_t zero_outpoint_cnt = 0;
  uint64_t zero_inpoint_cnt = 0;
  for (uint32_t i = 0; i < dataset.data_num_; ++i) {
    if (outpoint_num[i] != 0) {
      zero_outpoint_cnt++;
    }
    if (inpoint_num[i] != 0) {
      zero_inpoint_cnt++;
    }
  }

  auto inpoint_ratio = static_cast<double>(zero_inpoint_cnt) / dataset.data_num_;
  LOG_INFO("no_zero_inpoint = {} , no_zero_oupoint = {}", zero_inpoint_cnt, zero_outpoint_cnt);
  EXPECT_GE(inpoint_ratio, 0.9);
  EXPECT_EQ(zero_outpoint_cnt, dataset.data_num_);
}

TEST_F(SearchTest, SearchHNSWTest) {
  auto &dataset = ds();
  auto search_job = std::make_unique<GraphSearchJob<RawSpaceType>>(raw_space(), graph());

  auto res_pool = run_parallel_search(search_job, dataset, kDefaultTopk, kDefaultEf);
  auto recall = calc_recall(res_pool,
                            dataset.ground_truth_.data(),
                            dataset.query_num_,
                            dataset.gt_dim_,
                            kDefaultTopk);
  LOG_INFO("recall is {}.", recall);
  EXPECT_GE(recall, 0.5);
}

TEST_F(SearchTest, SearchHNSWTestSQSpace) {
  auto &dataset = ds();
  auto search_job = std::make_unique<GraphSearchJob<SQ8SpaceType, RawSpaceType>>(sq_space(),
                                                                                 graph(),
                                                                                 nullptr,
                                                                                 raw_space());

  auto res_pool = run_parallel_search(search_job, dataset, kDefaultTopk, kDefaultEf);
  auto recall = calc_recall(res_pool,
                            dataset.ground_truth_.data(),
                            dataset.query_num_,
                            dataset.gt_dim_,
                            kDefaultTopk);
  LOG_INFO("sq8 recall is {}.", recall);
  EXPECT_GE(recall, 0.5);
}

TEST(GraphSearchJobUnitTest, SearchInfoOverloadWithoutBlockedMaskMatchesPlainSearch) {
  auto space = make_one_dim_raw_space({10.0F, 20.0F, 30.0F, 0.0F, 1.0F});
  auto graph = make_graph_from_edges({{3, 1}, {2}, {}, {4}, {}});
  GraphSearchJob<RawSpaceType> search_job(space, graph);

  std::vector<float> query = {0.1F};
  std::vector<uint32_t> plain_ids(2);
  std::vector<uint32_t> info_ids(2);

  search_job.search_solo(query.data(), plain_ids.data(), 2, 4);
  search_job.search_solo(query.data(), info_ids.data(), SearchInfo{2, 4});

  EXPECT_EQ(info_ids, plain_ids);
}

TEST(GraphSearchJobUnitTest, BlockedMaskSearchSkipsFilteredCandidates) {
  auto space = make_one_dim_raw_space({10.0F, 20.0F, 30.0F, 0.0F, 1.0F});
  auto graph = make_graph_from_edges({{3, 1}, {2}, {}, {4}, {}});
  GraphSearchJob<RawSpaceType> search_job(space, graph);

  DynamicBitset blocked_mask(space->get_data_num());
  blocked_mask.set(3);

  std::vector<float> query = {0.1F};
  std::vector<uint32_t> ids(2, std::numeric_limits<uint32_t>::max());
  search_job.search_solo(query.data(), ids.data(), SearchInfo{2, 4}, &blocked_mask);

  EXPECT_EQ(ids[0], 4U);
  EXPECT_EQ(std::find(ids.begin(), ids.end(), 3U), ids.end());
}

TEST(GraphSearchJobUnitTest, UpdatedSearchUsesSecondHopNeighborsFromJobContext) {
  auto space = make_one_dim_raw_space({10.0F, 20.0F, 0.0F});
  auto graph = make_graph_from_edges({{}, {}, {}});
  auto job_context = std::make_shared<JobContext<uint32_t>>();
  job_context->removed_node_nbrs_[0] = {2};

  GraphSearchJob<RawSpaceType> search_job(space, graph, job_context);

  std::vector<float> query = {0.0F};
  std::vector<uint32_t> ids(1, std::numeric_limits<uint32_t>::max());
  search_job.search_solo_updated(query.data(), ids.data(), 3, 1);

  EXPECT_EQ(ids[0], 2U);
}

TEST(GraphSearchJobUnitTest, UnderfilledSearchMarksMissingIdsInvalid) {
  auto space = make_one_dim_raw_space({0.0F, 1.0F, 10.0F});
  auto graph = make_graph_from_edges({{1}, {}, {}});
  GraphSearchJob<RawSpaceType> search_job(space, graph);

  std::vector<float> query = {0.0F};
  std::vector<uint32_t> ids(3, 0U);
  search_job.search_solo(query.data(), ids.data(), 3, 3);

  EXPECT_EQ(ids[0], 0U);
  EXPECT_EQ(ids[1], 1U);
  EXPECT_EQ(ids[2], std::numeric_limits<uint32_t>::max());
}

TEST(GraphSearchJobUnitTest, UnderfilledRerankMarksMissingIdsInvalid) {
  auto build_space = make_one_dim_raw_space({0.0F, 1.0F, 10.0F});
  auto search_space = make_one_dim_sq8_space({0.0F, 1.0F, 10.0F});
  auto graph = make_graph_from_edges({{1}, {}, {}});
  GraphSearchJob<SQ8SpaceType, RawSpaceType> search_job(search_space, graph, nullptr, build_space);

  std::vector<float> query = {0.0F};
  std::vector<uint32_t> ids(3, 0U);
  search_job.search_solo(query.data(), ids.data(), 3, 3);

  EXPECT_EQ(ids[0], 0U);
  EXPECT_EQ(ids[1], 1U);
  EXPECT_EQ(ids[2], std::numeric_limits<uint32_t>::max());
}
TEST(RaBitQGraphSearchJobTest, AllowsNullBuildSpaceForScalarMetadataVariant) {
  using SearchSpaceType = RaBitQSpace<float, float, uint32_t, ScalarData>;
  using BuildSpaceType = RaBitQSpace<float, float, uint32_t, EmptyScalarData>;
  using SearchJobType = GraphSearchJob<SearchSpaceType, BuildSpaceType>;

  auto db_path = test_cache_dir() / "rabitq_graph_search_job_rocksdb";
  remove_path_if_exists(db_path);
  {
    RocksDBConfig config;
    config.db_path_ = db_path.string();

    constexpr uint32_t kDim = 64;
    constexpr uint32_t kCount = 4;
    auto search_space = std::make_shared<SearchSpaceType>(kCount, kDim, MetricType::L2, config);

    std::vector<float> vectors(kCount * kDim, 0.0f);
    for (uint32_t i = 0; i < kCount; ++i) {
      vectors[i * kDim + i] = 1.0f;
    }

    std::vector<ScalarData> metadata;
    metadata.reserve(kCount);
    for (uint32_t i = 0; i < kCount; ++i) {
      MetadataMap meta;
      meta["category"] = static_cast<int64_t>(i % 2);
      metadata.emplace_back("id_" + std::to_string(i), "doc_" + std::to_string(i), std::move(meta));
    }

    search_space->fit(vectors.data(), kCount, metadata.data());

    EXPECT_NO_THROW((void)std::make_unique<SearchJobType>(search_space, nullptr));

    auto search_job = std::make_unique<SearchJobType>(search_space, nullptr);
    EXPECT_NE(search_job, nullptr);
    search_space->close_db();
  }
  remove_path_if_exists(db_path);
}

}  // namespace
}  // namespace alaya
