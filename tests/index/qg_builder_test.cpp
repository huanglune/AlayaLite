// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "core/capabilities.hpp"
#include "index/graph/qg/qg_builder.hpp"
#include "space/rabitq_space.hpp"
#include "utils/openmp.hpp"

namespace alaya {
namespace {

using Space = RaBitQSpace<>;
using Builder = memory_qg::Builder<Space>;

static_assert(!core::Searchable<Builder>);
static_assert(!core::BatchSearchable<Builder>);
static_assert(!core::Saveable<Builder>);
static_assert(!core::StatsProvider<Builder>);
static_assert(!core::Mutable<Builder>);

constexpr std::uint32_t kRows = 128;
constexpr std::uint32_t kCapacity = 144;
constexpr std::uint32_t kDim = 64;

[[nodiscard]] auto make_vectors() -> std::vector<float> {
  std::vector<float> data(kRows * kDim);
  for (std::uint32_t row = 0; row < kRows; ++row) {
    for (std::uint32_t column = 0; column < kDim; ++column) {
      data[row * kDim + column] =
          std::sin(static_cast<float>((row + 1) * (column + 3)) * 0.03125F) +
          static_cast<float>(row % 11) * 0.01F;
    }
  }
  return data;
}

[[nodiscard]] auto build_snapshot(const std::vector<float> &data,
                                  core::Metric metric = core::Metric::l2)
    -> FrozenGraphSnapshot {
  auto space = std::make_shared<Space>(kCapacity, kDim, metric);
  space->fit(data.data(), kRows);
  core::BuildContext context;
  memory_qg::BuildOptions options;
  options.ef_build = 64;
  options.thread_count = 1;
  return Builder::build(
      {core::TypedTensorView::contiguous(data.data(), kRows, kDim), std::move(space)},
      options,
      context);
}

TEST(QgBuilderTest, ExportsValidatedFixedDegreeTopology) {
  platform::set_openmp_thread_count(1);
  const auto data = make_vectors();
  const auto snapshot = build_snapshot(data);

  EXPECT_EQ(snapshot.num_points(), kRows);
  EXPECT_EQ(snapshot.max_degree(), Space::kDegreeBound);
  EXPECT_LT(snapshot.entry_point(), kRows);
  EXPECT_NO_THROW(snapshot.validate());

  std::size_t edge_count{};
  for (std::size_t node = 0; node < snapshot.adjacency().size(); ++node) {
    const auto &neighbors = snapshot.adjacency()[node];
    EXPECT_EQ(neighbors.size(), Space::kDegreeBound);
    EXPECT_EQ(std::find(neighbors.begin(), neighbors.end(), node), neighbors.end());
    edge_count += neighbors.size();
  }
  EXPECT_EQ(edge_count, static_cast<std::size_t>(kRows) * Space::kDegreeBound);
}

TEST(QgBuilderTest, PreservesMetricAwareInnerProductBuildPath) {
  platform::set_openmp_thread_count(1);
  const auto data = make_vectors();
  const auto snapshot = build_snapshot(data, core::Metric::inner_product);
  EXPECT_EQ(snapshot.num_points(), kRows);
  EXPECT_EQ(snapshot.max_degree(), Space::kDegreeBound);
  EXPECT_NO_THROW(snapshot.validate());
}

TEST(QgBuilderTest, EnforcesInputOptionsAndBuildBudget) {
  platform::set_openmp_thread_count(1);
  const auto data = make_vectors();

  auto make_space = [&]() {
    auto space = std::make_shared<Space>(kCapacity, kDim, core::Metric::l2);
    space->fit(data.data(), kRows);
    return space;
  };
  const auto vectors = core::TypedTensorView::contiguous(data.data(), kRows, kDim);

  core::BuildContext denied;
  denied.growing_reservation = core::MemoryReservation(0);
  EXPECT_THROW(Builder::build({vectors, make_space()}, {.ef_build = 64, .thread_count = 1}, denied),
               std::runtime_error);

  core::BuildContext context;
  EXPECT_THROW(Builder::build({vectors, make_space()}, {.ef_build = 0, .thread_count = 1}, context),
               std::invalid_argument);
  EXPECT_THROW(Builder::build({vectors, make_space()}, {.ef_build = 64, .thread_count = 0}, context),
               std::invalid_argument);

  const auto truncated = core::TypedTensorView::contiguous(data.data(), kRows - 1, kDim);
  EXPECT_THROW(Builder::build({truncated, make_space()},
                              {.ef_build = 64, .thread_count = 1},
                              context),
               std::invalid_argument);
}

}  // namespace
}  // namespace alaya
