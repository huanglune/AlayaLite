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
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <vector>

#include "index/laser/medoid_generator.hpp"
#include "index/laser/quantized_graph.hpp"
#include "utils/vector_file_reader.hpp"

namespace alaya {
namespace {

auto make_temp_prefix(const char *label) -> std::filesystem::path {
  auto suffix = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  auto dir = std::filesystem::temp_directory_path() / ("laser_medoid_generator_test_" + suffix);
  std::filesystem::create_directories(dir);
  return dir / label;
}

void write_fbin(const std::filesystem::path &path,
                const std::vector<float> &data,
                uint32_t num_vectors,
                uint32_t dim) {
  std::ofstream out(path, std::ios::binary);
  int32_t hdr[2] = {static_cast<int32_t>(num_vectors), static_cast<int32_t>(dim)};
  out.write(reinterpret_cast<const char *>(hdr), sizeof(hdr));
  out.write(reinterpret_cast<const char *>(data.data()),
            static_cast<std::streamsize>(static_cast<size_t>(num_vectors) * dim * sizeof(float)));
}

TEST(MedoidGeneratorTest, WritesLoadableMedoidFiles) {
  constexpr uint32_t kNumPoints = 64;
  constexpr uint32_t kDim = 128;
  constexpr uint32_t kNumMedoids = 8;

  std::vector<float> vectors(static_cast<size_t>(kNumPoints) * kDim);
  for (uint32_t point_id = 0; point_id < kNumPoints; ++point_id) {
    for (uint32_t dim = 0; dim < kDim; ++dim) {
      vectors[static_cast<size_t>(point_id) * kDim + dim] =
          static_cast<float>(point_id * 10 + dim) * 0.01F;
    }
  }

  auto prefix = make_temp_prefix("medoids");
  auto fbin_path = prefix.parent_path() / "vectors.fbin";
  write_fbin(fbin_path, vectors, kNumPoints, kDim);

  MedoidGenerator generator({.num_medoids_ = kNumMedoids,
                             .sample_ratio_ = 0.5F,
                             .sample_cap_ = kNumPoints,
                             .num_threads_ = 2,
                             .random_seed_ = 42});
  auto result = generator.generate(fbin_path, prefix);

  ASSERT_EQ(result.medoid_ids_.size(), kNumMedoids);
  ASSERT_EQ(result.medoid_vectors_.size(), static_cast<size_t>(kNumMedoids) * kDim);

  // Verify medoid vectors match original data
  for (size_t medoid_idx = 0; medoid_idx < result.medoid_ids_.size(); ++medoid_idx) {
    auto medoid_id = result.medoid_ids_[medoid_idx];
    EXPECT_LT(medoid_id, kNumPoints);

    const auto *expected = vectors.data() + static_cast<size_t>(medoid_id) * kDim;
    const auto *actual = result.medoid_vectors_.data() + medoid_idx * kDim;
    for (uint32_t dim = 0; dim < kDim; ++dim) {
      EXPECT_FLOAT_EQ(actual[dim], expected[dim]);
    }
  }

  symqg::QuantizedGraph graph(kNumPoints, 16, 64, kDim);
  graph.load_medoids(prefix.string().c_str());

  std::filesystem::remove_all(prefix.parent_path());
}

}  // namespace
}  // namespace alaya
