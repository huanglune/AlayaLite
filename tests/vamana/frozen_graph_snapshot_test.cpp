// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "index/graph/frozen_graph_snapshot.hpp"

namespace {

class FrozenGraphSnapshotTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = std::filesystem::temp_directory_path() /
            ("alaya-frozen-graph-" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    std::filesystem::remove_all(root_);
    std::filesystem::create_directories(root_);
  }

  void TearDown() override { std::filesystem::remove_all(root_); }

  static auto read_bytes(const std::filesystem::path &path) -> std::vector<char> {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  }

  std::filesystem::path root_;
};

TEST_F(FrozenGraphSnapshotTest, SaveLoadRoundTripIsByteIdentical) {
  alaya::FrozenGraphSnapshot snapshot({{1, 2}, {0, 3}, {3}, {}},
                                      /*entry_point=*/2,
                                      /*max_degree=*/3,
                                      /*frozen_pts=*/7);
  ASSERT_NO_THROW(snapshot.validate());

  const auto first = root_ / "first.index";
  const auto second = root_ / "second.index";
  snapshot.save(first);
  auto loaded = alaya::FrozenGraphSnapshot::load(first);
  loaded.save(second);

  EXPECT_EQ(read_bytes(first), read_bytes(second));
  EXPECT_EQ(loaded.adjacency(), snapshot.adjacency());
  EXPECT_EQ(loaded.entry_point(), 2U);
  EXPECT_EQ(loaded.medoid(), 2U);
  EXPECT_EQ(loaded.num_points(), 4U);
  EXPECT_EQ(loaded.max_degree(), 3U);
  EXPECT_EQ(loaded.frozen_pts(), 7U);
}

TEST_F(FrozenGraphSnapshotTest, ValidationAcceptsAndRejectsExpectedTopologies) {
  EXPECT_NO_THROW((alaya::FrozenGraphSnapshot({{1}, {0}}, 0, 1).validate()));
  EXPECT_THROW((alaya::FrozenGraphSnapshot({}, 0, 0).validate()), std::invalid_argument);
  EXPECT_THROW((alaya::FrozenGraphSnapshot({{1}, {0}}, 2, 1).validate()), std::invalid_argument);
  EXPECT_THROW((alaya::FrozenGraphSnapshot({{1, 2}, {0}, {0}}, 0, 1).validate()),
               std::invalid_argument);
  EXPECT_THROW((alaya::FrozenGraphSnapshot({{0}, {0}}, 0, 1).validate()), std::invalid_argument);
  EXPECT_THROW((alaya::FrozenGraphSnapshot({{2}, {0}}, 0, 1).validate()), std::invalid_argument);
}

TEST_F(FrozenGraphSnapshotTest, VamanaConstructionTransfersAdjacencyOwnership) {
  constexpr std::uint32_t kRows = 96;
  constexpr std::uint32_t kDim = 8;
  constexpr std::uint32_t kDegree = 12;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
  std::vector<float> vectors(static_cast<std::size_t>(kRows) * kDim);
  for (auto &value : vectors) {
    value = distribution(rng);
  }

  alaya::vamana::VamanaBuildParams params;
  params.R = kDegree;
  params.L = 32;
  params.num_threads = 1;
  alaya::vamana::VamanaBuilder builder(vectors.data(), kRows, kDim, params);
  builder.build();
  const auto medoid = builder.medoid();

  auto snapshot = alaya::FrozenGraphSnapshot::from_vamana(std::move(builder));
  EXPECT_TRUE(builder.graph().empty());
  EXPECT_EQ(snapshot.entry_point(), medoid);
  EXPECT_EQ(snapshot.num_points(), kRows);
  EXPECT_EQ(snapshot.max_degree(), kDegree);
  EXPECT_NO_THROW(snapshot.validate());

  static_assert(!std::is_copy_constructible_v<alaya::FrozenGraphSnapshot>);
  static_assert(std::is_nothrow_move_constructible_v<alaya::FrozenGraphSnapshot>);
}

}  // namespace
