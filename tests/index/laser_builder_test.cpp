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

#include <chrono>
#include <filesystem>
#include <vector>

#include "index/laser/laser_builder.hpp"
#include "index/laser/laser_index.hpp"
#include "space/raw_space.hpp"

namespace alaya {
namespace {

auto make_temp_prefix(const char *label) -> std::filesystem::path {
  auto suffix = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  auto dir = std::filesystem::temp_directory_path() / ("laser_builder_test_" + suffix);
  std::filesystem::create_directories(dir);
  return dir / label;
}

auto make_vectors(uint32_t num_points, uint32_t dim) -> std::vector<float> {
  std::vector<float> vectors(static_cast<size_t>(num_points) * dim);
  for (uint32_t point_id = 0; point_id < num_points; ++point_id) {
    float cluster_bias = static_cast<float>(point_id % 8) * 10.0F;
    for (uint32_t axis = 0; axis < dim; ++axis) {
      vectors[static_cast<size_t>(point_id) * dim + axis] =
          cluster_bias + static_cast<float>(point_id * 3 + axis) * 0.001F;
    }
  }
  return vectors;
}

TEST(LaserBuilderTest, BuildsSearchableIndexAndWritesExpectedFiles) {
  constexpr uint32_t kNumPoints = 256;
  constexpr uint32_t kDim = 128;

  auto prefix = make_temp_prefix("laser");
  auto vectors = make_vectors(kNumPoints, kDim);
  auto space = std::make_shared<RawSpace<>>(kNumPoints, kDim, MetricType::L2);
  space->fit(vectors.data(), kNumPoints);

  LaserBuildParams params;
  params.max_degree_ = 64;
  params.num_medoids_ = 8;
  params.ef_construction_ = 32;
  params.ef_build_ = 32;
  params.max_memory_mb_ = 16;

  LaserBuilder<RawSpace<>> builder(space, params);
  builder.build(prefix);

  EXPECT_TRUE(std::filesystem::exists(prefix.string() + "_pca.bin"));
  EXPECT_TRUE(std::filesystem::exists(prefix.string() + "_medoids_indices"));
  EXPECT_TRUE(std::filesystem::exists(prefix.string() + "_medoids"));
  EXPECT_TRUE(std::filesystem::exists(prefix.parent_path() / "build_state.json"));
  EXPECT_TRUE(std::filesystem::exists(prefix.string() + "_R64_MD64.index"));
  EXPECT_TRUE(std::filesystem::exists(prefix.string() + "_R64_MD64.index_rotator"));
  EXPECT_TRUE(std::filesystem::exists(prefix.string() + "_R64_MD64.index_cache_ids"));
  EXPECT_TRUE(std::filesystem::exists(prefix.string() + "_R64_MD64.index_cache_nodes"));

  LaserIndex index;
  symqg::LaserSearchParams search_params;
  search_params.ef_search = 32;
  search_params.num_threads = 1;
  search_params.beam_width = 8;
  search_params.search_dram_budget_gb = 1.0F;
  index.load(prefix.string(), kNumPoints, 64, 64, kDim, search_params);

  std::vector<uint32_t> results(5, 0);
  index.search(vectors.data(), 5, results.data());
  EXPECT_LT(results[0], kNumPoints);
}

TEST(LaserBuilderTest, ResumesCompletedPcaAndRestartsWhenParamsChange) {
  constexpr uint32_t kNumPoints = 192;
  constexpr uint32_t kDim = 128;

  auto prefix = make_temp_prefix("resume");
  auto vectors = make_vectors(kNumPoints, kDim);
  auto space = std::make_shared<RawSpace<>>(kNumPoints, kDim, MetricType::L2);
  space->fit(vectors.data(), kNumPoints);

  LaserBuildParams params;
  params.max_degree_ = 32;
  params.num_medoids_ = 4;
  params.ef_construction_ = 24;
  params.ef_build_ = 24;
  params.max_memory_mb_ = 16;
  params.keep_intermediates_ = true;

  LaserBuilder<RawSpace<>> interrupted_builder(space, params);
  interrupted_builder.set_phase_hook_for_test([](LaserBuildPhase phase) {
    if (phase == LaserBuildPhase::kPartition) {
      throw std::runtime_error("interrupt after pca");
    }
  });
  EXPECT_THROW(interrupted_builder.build(prefix), std::runtime_error);

  auto pca_time_before = std::filesystem::last_write_time(prefix.string() + "_pca_base.fbin");

  LaserBuilder<RawSpace<>> resumed_builder(space, params);
  resumed_builder.build(prefix);
  auto pca_time_after_resume = std::filesystem::last_write_time(prefix.string() + "_pca_base.fbin");
  EXPECT_EQ(pca_time_after_resume, pca_time_before);

  auto final_index_time_before = std::filesystem::last_write_time(prefix.string() + "_R32_MD64.index");

  auto changed_params = params;
  changed_params.max_degree_ = 64;
  LaserBuilder<RawSpace<>> restart_builder(space, changed_params);
  restart_builder.build(prefix);

  auto pca_time_after_restart = std::filesystem::last_write_time(prefix.string() + "_pca_base.fbin");
  auto final_index_time_after = std::filesystem::last_write_time(prefix.string() + "_R64_MD64.index");
  EXPECT_GT(pca_time_after_restart, pca_time_after_resume);
  EXPECT_GT(final_index_time_after, final_index_time_before);
}

}  // namespace
}  // namespace alaya
