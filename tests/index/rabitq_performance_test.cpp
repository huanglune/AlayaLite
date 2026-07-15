// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include "index/graph/detail/search_runtime/graph_search_job.hpp"
#include "index/graph/qg/detail/qg_builder_kernel.hpp"
#include "space/rabitq_space.hpp"
#include "utils/evaluate.hpp"
#include "utils/io_utils.hpp"
#include "utils/test_paths.hpp"
#include "core/log.hpp"
#include "index/graph/detail/timer.hpp"

namespace alaya {

class RaBitQDeep1MTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto dir = test::data_dir() / "deep1M";
    auto base = dir / "deep1M_base.fvecs";
    if (!std::filesystem::exists(base)) {
      GTEST_SKIP() << "deep1M dataset not found at " << dir;
    }
    load_fvecs(dir / "deep1M_base.fvecs", data_, data_num_, dim_);
    load_fvecs(dir / "deep1M_query.fvecs", queries_, query_num_, query_dim_);
    load_ivecs(dir / "deep1M_groundtruth.ivecs", gt_, gt_num_, gt_dim_);
    index_dir_ = dir;
  }

  std::vector<float> data_;
  std::vector<float> queries_;
  std::vector<uint32_t> gt_;
  uint32_t data_num_ = 0;
  uint32_t query_num_ = 0;
  uint32_t dim_ = 0;
  uint32_t query_dim_ = 0;
  uint32_t gt_num_ = 0;
  uint32_t gt_dim_ = 0;
  std::filesystem::path index_dir_;
};

using IDType = uint32_t;
TEST_F(RaBitQDeep1MTest, Deep1MQGTest) {
  LOG_INFO("Building QG on Deep1M...");
  auto index_file = index_dir_ / "deep1M_rabitq.qg";
  std::string_view path = index_file.native();

  if (!std::filesystem::exists(index_file)) {
    auto space = std::make_shared<RaBitQSpace<>>(data_num_, dim_, core::Metric::l2);
    space->fit(data_.data(), data_num_);
    auto qg = detail::QgBuilderKernel<RaBitQSpace<>>(space);
    qg.build_graph();
    space->save(path);
  }

  auto load_space = std::make_shared<RaBitQSpace<>>();
  load_space->load(path);
  auto search_job = std::make_unique<GraphSearchJob<RaBitQSpace<>>>(load_space, nullptr);

  std::vector<size_t> efs = {10, 20, 40, 50, 55, 60, 80, 100, 150, 170,
                             190, 200, 250, 300, 400, 500, 600, 800, 1500};
  constexpr size_t test_round = 3;
  constexpr size_t topk = 10;
  Timer timer;
  std::vector<std::vector<float>> all_qps(test_round, std::vector<float>(efs.size()));
  std::vector<std::vector<float>> all_recall(test_round, std::vector<float>(efs.size()));

  for (size_t r = 0; r < test_round; ++r) {
    for (size_t i = 0; i < efs.size(); ++i) {
      size_t ef = efs[i];
      size_t total_correct = 0;
      float total_time = 0;
      std::vector<IDType> results(topk);
      for (uint32_t n = 0; n < query_num_; ++n) {
        timer.reset();
        search_job->rabitq_search_solo(queries_.data() + (n * dim_), topk, results.data(), ef);
        total_time += timer.elapsed_us();
        for (size_t k = 0; k < topk; ++k) {
          for (uint32_t j = 0; j < gt_dim_; ++j) {
            if (results[k] == gt_[(n * gt_dim_) + j]) {
              total_correct++;
              break;
            }
          }
        }
      }
      all_qps[r][i] = static_cast<float>(query_num_) / (total_time / 1e6F);
      all_recall[r][i] = static_cast<float>(total_correct) / static_cast<float>(query_num_ * topk);
    }
  }

  auto avg_qps = horizontal_avg(all_qps);
  auto avg_recall = horizontal_avg(all_recall);

  std::cout << "\n===== RaBitQ Deep1M Performance Results =====\n";
  std::cout << "ef\tQPS\tRecall\n";
  for (size_t i = 0; i < avg_qps.size(); ++i) {
    std::cout << efs[i] << '\t' << avg_qps[i] << '\t' << avg_recall[i] << '\n';
  }
  std::cout << "=============================================\n";
}

}  // namespace alaya
