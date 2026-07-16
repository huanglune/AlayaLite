// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "index/graph/diskann/diskann_index.hpp"
#include "index/graph/frozen_graph_snapshot.hpp"
#include "index/graph/vamana/vamana_builder.hpp"

namespace {

class DiskANNBuildFromGraphTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static std::atomic<std::uint64_t> counter{0};
    root_ = std::filesystem::temp_directory_path() /
            ("alaya-diskann-seal-" + std::to_string(counter.fetch_add(1)));
    std::filesystem::remove_all(root_);
    std::filesystem::create_directories(root_);
  }

  void TearDown() override { std::filesystem::remove_all(root_); }

  static auto read_bytes(const std::filesystem::path &path) -> std::vector<char> {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("cannot read test artifact: " + path.string());
    }
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  }

  static auto files_in(const std::filesystem::path &directory) -> std::vector<std::string> {
    std::vector<std::string> files;
    for (const auto &entry : std::filesystem::directory_iterator(directory)) {
      if (entry.is_regular_file()) {
        files.push_back(entry.path().filename().string());
      }
    }
    std::sort(files.begin(), files.end());
    return files;
  }

  std::filesystem::path root_;
};

TEST_F(DiskANNBuildFromGraphTest, MatchesDefaultBuildDirectoryByteForByte) {
  constexpr std::uint64_t kRows = 10000;
  constexpr std::uint64_t kDim = 64;
  constexpr std::uint32_t kDegree = 32;
  constexpr std::uint32_t kBuildList = 64;
  constexpr std::uint64_t kSeed = 20260716;

  std::mt19937 rng(static_cast<std::mt19937::result_type>(kSeed));
  std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
  std::vector<float> vectors(kRows * kDim);
  for (auto &value : vectors) {
    value = distribution(rng);
  }
  std::vector<std::uint64_t> labels(kRows);
  for (std::uint64_t row = 0; row < kRows; ++row) {
    labels[row] = 1000000 + row * 3;
  }

  alaya::diskann::DiskANNBuildParams build_params;
  build_params.R = kDegree;
  build_params.L = kBuildList;
  build_params.alpha = 1.2F;
  build_params.pq_n_chunks = 8;
  build_params.cache_ratio = 0.05;
  build_params.num_threads = 1;
  build_params.pq_train_iters = 3;
  build_params.seed = kSeed;

  const auto default_directory = root_ / "default";
  alaya::diskann::DiskANNIndex::build(default_directory.string(),
                                      vectors.data(),
                                      labels.data(),
                                      kRows,
                                      kDim,
                                      build_params);

  alaya::vamana::VamanaBuildParams vamana_params;
  vamana_params.R = build_params.R;
  vamana_params.L = build_params.L;
  vamana_params.alpha = build_params.alpha;
  vamana_params.num_threads = build_params.num_threads;
  vamana_params.seed = build_params.seed;
  alaya::vamana::VamanaBuilder builder(vectors.data(),
                                       kRows,
                                       static_cast<std::uint32_t>(kDim),
                                       vamana_params);
  builder.build();
  auto snapshot = alaya::FrozenGraphSnapshot::from_vamana(std::move(builder), kDegree);

  alaya::diskann::DiskANNMaterializeParams materialize_params;
  materialize_params.pq_n_chunks = build_params.pq_n_chunks;
  materialize_params.cache_ratio = build_params.cache_ratio;
  materialize_params.num_threads = build_params.num_threads;
  materialize_params.pq_train_iters = build_params.pq_train_iters;
  materialize_params.seed = build_params.seed;

  const auto sealed_directory = root_ / "sealed";
  alaya::diskann::DiskANNIndex::build_from_graph(sealed_directory.string(),
                                                 snapshot,
                                                 vectors.data(),
                                                 labels.data(),
                                                 kDim,
                                                 materialize_params);

  const auto expected_files = files_in(default_directory);
  ASSERT_EQ(files_in(sealed_directory), expected_files);
  ASSERT_EQ(expected_files.size(), 7U);
  for (const auto &filename : expected_files) {
    EXPECT_EQ(read_bytes(default_directory / filename), read_bytes(sealed_directory / filename))
        << "artifact differs: " << filename;
  }
}

}  // namespace
