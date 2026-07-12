// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <queue>
#include <span>
#include <vector>

#include "core/capabilities.hpp"
#include "executor/jobs/graph_search_job.hpp"
#include "executor/jobs/graph_update_job.hpp"
#include "index/graph/hnsw/hnsw_segment.hpp"
#include "space/raw_space.hpp"

namespace alaya {
namespace {

using Space = RawSpace<>;
using Segment = HnswSegment<Space>;

static_assert(core::Searchable<Segment>);
static_assert(core::BatchSearchable<Segment>);
static_assert(core::Persistable<Segment>);
static_assert(!core::Mutable<Segment>);

class HnswSegmentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    data_.resize(kRows * kDim);
    for (std::uint32_t row = 0; row < kRows; ++row) {
      for (std::uint32_t col = 0; col < kDim; ++col) {
        data_[row * kDim + col] = static_cast<float>((row * 17 + col * 5) % 101) / 101.0F;
      }
    }
    space_ = std::make_shared<Space>(kCapacity, kDim, MetricType::L2);
    space_->fit(data_.data(), kRows);
    core::BuildContext context;
    segment_ = Segment::build({space_, space_},
                              {.max_neighbors = 8, .ef_construction = 32, .thread_count = 1},
                              context);
  }

  void TearDown() override {
    std::filesystem::remove(graph_path_);
    std::filesystem::remove(data_path_);
  }

  static constexpr std::uint32_t kRows = 64;
  static constexpr std::uint32_t kCapacity = 80;
  static constexpr std::uint32_t kDim = 8;
  std::vector<float> data_;
  std::shared_ptr<Space> space_;
  std::unique_ptr<Segment> segment_;
  std::filesystem::path graph_path_ = "hnsw-segment.graph";
  std::filesystem::path data_path_ = "hnsw-segment.data";
};

TEST_F(HnswSegmentTest, BuildReturnsFullySearchableSegment) {
  const auto descriptor = segment_->descriptor();
  EXPECT_EQ(descriptor.algorithm_id, core::compat::kAlgorithmHnsw);
  EXPECT_EQ(descriptor.format_version, 1U);
  EXPECT_EQ(descriptor.rows, kRows);
  EXPECT_EQ(descriptor.dim, kDim);
  EXPECT_EQ(descriptor.state, core::SegmentState::sealed);

  std::vector<core::SearchHit> hits(4);
  auto result = segment_->search({data_.data(), kDim}, {.top_k = 4, .effort = 24}, hits);
  ASSERT_EQ(result.count, 4U);
  EXPECT_EQ(hits.front().id, 0U);
  EXPECT_FLOAT_EQ(hits.front().distance, 0.0F);
}

TEST_F(HnswSegmentTest, BatchSearchWritesCallerOwnedSink) {
  std::vector<core::SearchHit> hits(6);
  auto result = segment_->batch_search({data_.data(), 2, kDim}, {.top_k = 3, .effort = 24}, hits);
  EXPECT_EQ(result.query_count, 2U);
  EXPECT_EQ(result.hit_count, 6U);
  EXPECT_EQ(hits[0].id, 0U);
  EXPECT_EQ(hits[3].id, 1U);
}

TEST_F(HnswSegmentTest, SaveManifestAndOpenLegacyArtifactsWithoutChangingGraphBytes) {
  const auto graph = graph_path_.string();
  const auto data = data_path_.string();
  core::ArtifactWriter writer{graph, data, {}};
  const auto manifest = segment_->save(writer, {});
  ASSERT_EQ(manifest.schema_version, 1U);
  ASSERT_EQ(manifest.format_version, 1U);
  ASSERT_EQ(manifest.algorithm_id, core::compat::kAlgorithmHnsw);
  ASSERT_EQ(manifest.artifacts.size(), 2U);

  core::OpenContext context;
  auto reopened = Segment::open({graph, data, {}}, {}, context);
  std::vector<core::SearchHit> hits(4);
  const auto result = reopened->search({data_.data(), kDim}, {.top_k = 4, .effort = 24}, hits);
  EXPECT_EQ(result.count, 4U);
  EXPECT_EQ(hits.front().id, 0U);

  Graph<> legacy;
  legacy.load(graph);
  EXPECT_EQ(legacy.max_nodes_, kCapacity);
  EXPECT_NE(legacy.overlay_graph_, nullptr);
}

// Decision-point-6 characterization: the old generic update job only edits the
// level-0 Graph/Space.  It does not assign an HNSW level or update overlay
// adjacency for an inserted node, so HNSW layer invariants are not preserved.
// Keeping this executable characterization prevents declaring Mutable merely
// because GraphUpdateJob happens to be callable.
TEST(HnswMutationInvariantTest, GenericInsertFailsHnswLayerInvariantSoCapabilityStaysAbsent) {
  constexpr std::uint32_t rows = 32;
  constexpr std::uint32_t capacity = 40;
  constexpr std::uint32_t dim = 4;
  std::vector<float> data(capacity * dim);
  for (std::uint32_t i = 0; i < rows * dim; ++i) data[i] = static_cast<float>(i % 37);
  auto space = std::make_shared<Space>(capacity, dim, MetricType::L2);
  space->fit(data.data(), rows);
  detail::HnswBuilderKernel<Space> kernel(space, 8, 24);
  auto graph = std::shared_ptr<Graph<>>(kernel.build_graph(1).release());
  auto search = std::make_shared<GraphSearchJob<Space>>(space, graph);
  GraphUpdateJob<Space> update(search);

  float inserted[dim] = {0.25F, 0.5F, 0.75F, 1.0F};
  const auto id = update.insert_and_update(inserted, 16);
  ASSERT_EQ(id, rows);

  // The level-0 degree bound remains mechanically respected and the inserted
  // point is reachable by search before deletion.
  for (std::uint32_t node = 0; node <= id; ++node) {
    std::uint32_t degree = 0;
    for (; degree < graph->max_nbrs_; ++degree) {
      if (graph->at(node, degree) == Graph<>::kEmptyId) break;
    }
    EXPECT_LE(degree, graph->max_nbrs_);
  }
  std::vector<std::uint32_t> result_ids(8);
  search->search_solo(inserted, result_ids.data(), 8, 16);
  EXPECT_NE(std::find(result_ids.begin(), result_ids.end(), id), result_ids.end());

  // Level storage was sized for capacity, but generic insertion leaves the new
  // node at the default level and never performs HNSW overlay neighbor updates.
  ASSERT_NE(graph->overlay_graph_, nullptr);
  EXPECT_EQ(graph->overlay_graph_->levels_[id], 0);
  bool appears_in_overlay = false;
  for (std::uint32_t node = 0; node < rows; ++node) {
    const auto level = graph->overlay_graph_->levels_[node];
    for (int layer = 1; layer <= level; ++layer) {
      for (std::uint32_t edge = 0; edge < graph->max_nbrs_; ++edge) {
        appears_in_overlay |= graph->overlay_graph_->at(layer, node, edge) == id;
      }
    }
  }
  EXPECT_FALSE(appears_in_overlay);

  update.remove(id);
  bool deleted_id_still_referenced = false;
  for (std::uint32_t node = 0; node < rows; ++node) {
    for (std::uint32_t edge = 0; edge < graph->max_nbrs_; ++edge) {
      deleted_id_still_referenced |= graph->at(node, edge) == id;
    }
  }
  // Generic erase is lazy and leaves inbound references until unrelated nodes
  // are explicitly repaired; this violates the HNSW deletion invariant.
  EXPECT_TRUE(deleted_id_still_referenced);
  static_assert(!core::Mutable<Segment>);
}

}  // namespace
}  // namespace alaya
