// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/graph/knng/detail/nndescent_kernel.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <memory>
#include <string_view>

#include "index/graph/detail/search_runtime/graph_search_job.hpp"
#include "index/graph/graph.hpp"
#include "index/memory_engine_registry.hpp"
#include "space/raw_space.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"
#include "index/graph/detail/thread_config.hpp"
#include "index/graph/detail/timer.hpp"

namespace alaya {
using test::Dataset;

static_assert(internal::memory::kKnngKernelRegistration.role ==
              internal::memory::EngineRole::build_kernel);
static_assert(!internal::memory::kKnngKernelRegistration.feature_switches_behavior);
static_assert(!internal::memory::kKnngKernelRegistration.has_legacy_factory);

class NnDescentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ds_ = test::random_dataset();

    space_ = std::make_shared<RawSpace<>>(ds_.data_num, ds_.dim, core::Metric::l2);
    space_->fit(ds_.data.data(), ds_.data_num);
    nn_descent_ = std::make_unique<NndescentImpl<RawSpace<>>>(space_, topk_);
  }

  void TearDown() override {
    if (std::filesystem::exists(filename_)) {
      remove(filename_.data());
    }
  }

  int topk_ = 10;
  Dataset ds_;
  std::unique_ptr<NndescentImpl<RawSpace<>>> nn_descent_ = nullptr;
  std::shared_ptr<RawSpace<>> space_ = nullptr;
  std::string_view filename_ = "nndescent_test.graph";
};

TEST_F(NnDescentTest, BuildGraphTest) {
  auto graph = nn_descent_->build_graph();
  // graph->print_graph();
}

class NnDescentSearchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ds_ = test::random_dataset();
  }

  void TearDown() override {}

  Dataset ds_;
};

TEST_F(NnDescentSearchTest, SimpleSearchTest) {
  const size_t kM = 32;
  size_t topk = 10;
  size_t ef = 100;
  std::string index_type = "NnDescent";

  std::filesystem::path index_file = fmt::format("{}_M{}.{}", ds_.name, kM, index_type);
  std::shared_ptr<alaya::RawSpace<>> space =
      std::make_shared<alaya::RawSpace<>>(ds_.data_num, ds_.dim, core::Metric::l2);
  space->fit(ds_.data.data(), ds_.data_num);

  auto load_graph = std::make_shared<alaya::Graph<>>(ds_.data_num, kM);

  if (!std::filesystem::exists(index_file)) {
    auto t1 = Timer();

    alaya::NndescentImpl<alaya::RawSpace<>> nndescent =
        alaya::NndescentImpl<alaya::RawSpace<>>(space, kM);
    auto graph = nndescent.build_graph();

    LOG_INFO("The time of building hnsw is {}s.", t1.elapsed() / 1000000.0);

    std::string_view path = index_file.native();
    graph->save(path);
  }
  std::string_view path = index_file.native();
  load_graph->load(path);

  // load_graph->print_graph();

  auto task_generator =
      std::make_unique<alaya::GraphSearchJob<alaya::RawSpace<>>>(space, load_graph);

  LOG_INFO("creator task generator success");

  using IDType = uint32_t;

  Timer timer{};
  std::vector<std::vector<IDType>> res_pool(ds_.query_num, std::vector<IDType>(topk));
  const size_t kSearchThreadNum =
      std::min<size_t>(cap_thread_count(16), static_cast<size_t>(ds_.query_num));
  std::vector<std::thread> tasks(kSearchThreadNum);

  auto search_knn = [&](uint32_t i) {
    for (; i < ds_.query_num; i += kSearchThreadNum) {
      std::vector<uint32_t> ids(topk);  // Now returns topk directly
      auto cur_query = ds_.queries.data() + i * ds_.dim;
      // New interface: search_solo(query, ids, topk, ef) returns topk results
      task_generator->search_solo(cur_query, ids.data(), topk, ef);

      // search_solo now returns topk results directly
      auto id_set = std::set(ids.begin(), ids.end());

      if (id_set.size() < topk) {
        fmt::println("i id: {}", i);
        fmt::println("ids size: {}", id_set.size());
      }
      res_pool[i] = ids;
    }
  };

  for (size_t i = 0; i < kSearchThreadNum; i++) {
    tasks[i] = std::thread(search_knn, i);
  }

  for (size_t i = 0; i < kSearchThreadNum; i++) {
    if (tasks[i].joinable()) {
      tasks[i].join();
    }
  }

  LOG_INFO("total time: {} s.", timer.elapsed() / 1000000.0);

  // Computing recall;
  auto recall = calc_recall(res_pool,  ds_.ground_truth.data(), ds_.query_num, ds_.gt_dim, topk);
  LOG_INFO("recall is {}.", recall);
  EXPECT_GE(recall, 0.5);
}

}  // namespace alaya
