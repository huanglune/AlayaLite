// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "index/graph/detail/search_runtime/graph_search_job.hpp"
#include "index/graph/qg/detail/qg_builder_kernel.hpp"
#include "space/rabitq_space.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"
#include "core/log.hpp"
#include "index/graph/detail/timer.hpp"

namespace alaya {

using IDType = uint32_t;

TEST(RaBitQTest, BuildAndSearchQG) {
  auto ds = test::random_dataset({.num_vectors = 2000, .dim = 128});

  auto space = std::make_shared<RaBitQSpace<>>(ds.data_num, ds.dim, core::Metric::l2);
  space->fit(ds.data.data(), ds.data_num);

  auto qg = detail::QgBuilderKernel<RaBitQSpace<>>(space);
  qg.build_graph();

  auto search_job = std::make_unique<GraphSearchJob<RaBitQSpace<>>>(space, nullptr);

  constexpr uint32_t topk = 10;
  constexpr size_t ef = 100;
  std::vector<IDType> results(topk);

  size_t total_correct = 0;
  for (uint32_t n = 0; n < ds.query_num; ++n) {
    search_job->rabitq_search_solo(ds.queries.data() + (n * ds.dim), topk, results.data(), ef);
    for (size_t k = 0; k < topk; ++k) {
      for (uint32_t j = 0; j < ds.gt_dim; ++j) {
        if (results[k] == ds.ground_truth[(n * ds.gt_dim) + j]) {
          total_correct++;
          break;
        }
      }
    }
  }

  float recall = static_cast<float>(total_correct) / static_cast<float>(ds.query_num * topk);
  LOG_INFO("RaBitQ recall@{} = {:.4f}", topk, recall);
  EXPECT_GT(recall, 0.3F);
}

}  // namespace alaya
