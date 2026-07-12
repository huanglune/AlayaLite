// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "index/graph/qg_naming.hpp"
#include "space/rabitq_space.hpp"

namespace alaya {
namespace {

using MemorySpace = RaBitQSpace<>;
using MemoryBuilder = memory_qg::Builder<MemorySpace>;

static_assert(std::is_same_v<MemoryBuilder, QGBuilder<MemorySpace>>);
static_assert(std::is_same_v<disk_laser_qg::Builder, laser::QGBuilder>);
static_assert(std::is_same_v<disk_laser_qg::Graph, laser::QuantizedGraph>);

static_assert(std::is_constructible_v<MemoryBuilder, std::shared_ptr<MemorySpace> &, size_t>);
static_assert(!std::is_constructible_v<MemoryBuilder, disk_laser_qg::Graph &, uint32_t, size_t>);
static_assert(
    std::is_constructible_v<disk_laser_qg::Builder, disk_laser_qg::Graph &, uint32_t, size_t>);
static_assert(
    !std::is_constructible_v<disk_laser_qg::Builder, std::shared_ptr<MemorySpace> &, size_t>);

// Golden anchors for the LASER v1 factor field order. The disk payload stores
// three structure-of-arrays blocks in this same order, not Factor objects.
static_assert(sizeof(float) == 4);
static_assert(std::is_standard_layout_v<disk_laser_qg::Factor>);
static_assert(sizeof(disk_laser_qg::Factor) == 3 * sizeof(float));
static_assert(offsetof(disk_laser_qg::Factor, triple_x) == 0);
static_assert(offsetof(disk_laser_qg::Factor, factor_dq) == sizeof(float));
static_assert(offsetof(disk_laser_qg::Factor, factor_vq) == 2 * sizeof(float));

TEST(QGNamingContract, MemoryV1FactorArraysKeepSerializedOrder) {
  constexpr size_t kDim = 64;
  MemorySpace space(/*capacity=*/1, kDim, MetricType::L2);
  std::vector<float> data(kDim, 0.0F);
  space.fit(data.data(), /*item_cnt=*/1);

  const auto address = [](const void *ptr) {
    return reinterpret_cast<uintptr_t>(ptr);
  };
  const uintptr_t codes = address(space.get_nei_qc_ptr(0));
  const uintptr_t f_add = address(space.get_f_add_ptr(0));
  const uintptr_t f_rescale = address(space.get_f_rescale_ptr(0));
  const uintptr_t neighbors = address(space.get_edges(0));

  EXPECT_EQ(f_add - codes, space.get_padded_dim() * MemorySpace::kDegreeBound / 8);
  EXPECT_EQ(f_rescale - f_add, MemorySpace::kDegreeBound * sizeof(float));
  EXPECT_EQ(neighbors - f_rescale, MemorySpace::kDegreeBound * sizeof(float));
}

}  // namespace
}  // namespace alaya
