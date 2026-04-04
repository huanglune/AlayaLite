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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "index/diskann/cross_shard_merger.hpp"
#include "index/diskann/kmeans_partitioner.hpp"
#include "index/diskann/shard_vamana_builder.hpp"
#include "space/raw_space.hpp"
#include "storage/buffer/buffer_pool.hpp"
#include "storage/diskann/data_file.hpp"
#include "storage/diskann/meta_file.hpp"

namespace alaya {
namespace {

constexpr uint32_t kDim = 8;
constexpr uint32_t kMaxDegree = 16;

auto make_temp_dir(const char *label) -> std::filesystem::path {
  auto suffix = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  auto dir = std::filesystem::temp_directory_path() / ("cross_shard_merger_test_" + suffix) / label;
  std::filesystem::create_directories(dir);
  return dir;
}

/**
 * @brief Write a shard graph file in the ShardVamanaBuilder export format.
 *
 * Format per node: [global_id(4B)][count(2B)][{nbr_id(4B), dist(4B)} x count]
 * Nodes must be sorted by global_id ascending.
 */
void write_shard_graph(const std::filesystem::path &path,
                       const std::vector<ShardGraphReader::NodeEntry> &nodes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open()) << "Failed to create shard graph: " << path;
  for (const auto &node : nodes) {
    out.write(reinterpret_cast<const char *>(&node.global_id_), sizeof(node.global_id_));
    auto count = static_cast<uint16_t>(node.neighbors_.size());
    out.write(reinterpret_cast<const char *>(&count), sizeof(count));
    for (const auto &nbr : node.neighbors_) {
      out.write(reinterpret_cast<const char *>(&nbr.id_), sizeof(nbr.id_));
      out.write(reinterpret_cast<const char *>(&nbr.distance_), sizeof(nbr.distance_));
    }
  }
}

// ============================================================================
// Test 5.6: Merge correctly unions neighbors from 2+ shards for overlapping nodes
// ============================================================================

TEST(CrossShardMergerTest, UnionFromTwoShards) {
  auto dir = make_temp_dir("union_two_shards");

  // Shard 0: node 0 has neighbors {1, 2}, node 1 has neighbor {0}
  std::vector<ShardGraphReader::NodeEntry> shard0_nodes = {
      {0, {{1, 1.0F, 0}, {2, 2.0F, 0}}},
      {1, {{0, 1.0F, 0}}},
  };

  // Shard 1: node 0 has neighbors {3, 4}, node 1 has neighbors {0, 5}
  // node 0 overlaps — should get union of {1,2,3,4}
  // node 1 overlaps — neighbor 0 is duplicated, should dedup and get {0,5}
  std::vector<ShardGraphReader::NodeEntry> shard1_nodes = {
      {0, {{3, 3.0F, 1}, {4, 4.0F, 1}}},
      {1, {{0, 0.5F, 1}, {5, 5.0F, 1}}},
  };

  auto shard0_path = dir / "shard_0.graph";
  auto shard1_path = dir / "shard_1.graph";
  write_shard_graph(shard0_path, shard0_nodes);
  write_shard_graph(shard1_path, shard1_nodes);

  CrossShardMerger merger({kMaxDegree, 1.2F});
  merger.open({shard0_path, shard1_path});

  std::vector<CrossShardMerger::MergedNode> results;
  merger.merge_all([&](const CrossShardMerger::MergedNode &node) { results.push_back(node); });

  ASSERT_EQ(results.size(), 2U);

  // Node 0: union of {1,2} from shard0 and {3,4} from shard1 = {1,2,3,4}
  EXPECT_EQ(results[0].global_id_, 0U);
  std::set<uint32_t> node0_nbrs(results[0].neighbor_ids_.begin(), results[0].neighbor_ids_.end());
  EXPECT_EQ(node0_nbrs.count(1), 1U);
  EXPECT_EQ(node0_nbrs.count(2), 1U);
  EXPECT_EQ(node0_nbrs.count(3), 1U);
  EXPECT_EQ(node0_nbrs.count(4), 1U);
  EXPECT_EQ(node0_nbrs.size(), 4U);

  // Node 1: deduped union of {0} from shard0 and {0,5} from shard1 = {0,5}
  EXPECT_EQ(results[1].global_id_, 1U);
  std::set<uint32_t> node1_nbrs(results[1].neighbor_ids_.begin(), results[1].neighbor_ids_.end());
  EXPECT_EQ(node1_nbrs.count(0), 1U);
  EXPECT_EQ(node1_nbrs.count(5), 1U);
  EXPECT_EQ(node1_nbrs.size(), 2U);

  std::filesystem::remove_all(dir);
}

TEST(CrossShardMergerTest, DedupKeepsMinDistance) {
  auto dir = make_temp_dir("dedup_min_dist");

  // Same neighbor in two shards with different distances
  std::vector<ShardGraphReader::NodeEntry> shard0 = {
      {0, {{1, 5.0F, 0}, {2, 3.0F, 0}}},
  };
  std::vector<ShardGraphReader::NodeEntry> shard1 = {
      {0, {{1, 2.0F, 1}, {3, 4.0F, 1}}},
  };

  auto s0 = dir / "shard_0.graph";
  auto s1 = dir / "shard_1.graph";
  write_shard_graph(s0, shard0);
  write_shard_graph(s1, shard1);

  CrossShardMerger merger({kMaxDegree, 1.2F});
  merger.open({s0, s1});

  std::vector<CrossShardMerger::MergedNode> results;
  merger.merge_all([&](const CrossShardMerger::MergedNode &node) { results.push_back(node); });

  ASSERT_EQ(results.size(), 1U);
  // Neighbor 1 should appear once, neighbors should be sorted by distance
  // Expected order: neighbor 1 (dist 2.0), neighbor 2 (dist 3.0), neighbor 3 (dist 4.0)
  EXPECT_EQ(results[0].neighbor_ids_.size(), 3U);
  EXPECT_EQ(results[0].neighbor_ids_[0], 1U);  // min distance = 2.0
  EXPECT_EQ(results[0].neighbor_ids_[1], 2U);  // distance = 3.0
  EXPECT_EQ(results[0].neighbor_ids_[2], 3U);  // distance = 4.0

  std::filesystem::remove_all(dir);
}

// ============================================================================
// Test 5.7: Distance-ordered selection produces <= R neighbors per node
// ============================================================================

TEST(CrossShardMergerTest, HeuristicPruneRespectsMaxDegree) {
  auto dir = make_temp_dir("prune_max_degree");

  constexpr uint32_t kSmallR = 4;

  // Build two shards that give node 0 many neighbors (more than kSmallR)
  std::vector<ShardGraphReader::NodeEntry> shard0;
  std::vector<ShardGraphReader::NodeEntry> shard1;
  ShardGraphReader::NodeEntry node0_s0{0, {}};
  ShardGraphReader::NodeEntry node0_s1{0, {}};
  for (uint32_t i = 1; i <= 5; ++i) {
    node0_s0.neighbors_.push_back({i, static_cast<float>(i), 0});
  }
  for (uint32_t i = 6; i <= 10; ++i) {
    node0_s1.neighbors_.push_back({i, static_cast<float>(i), 1});
  }
  shard0.push_back(node0_s0);
  shard1.push_back(node0_s1);

  auto s0 = dir / "shard_0.graph";
  auto s1 = dir / "shard_1.graph";
  write_shard_graph(s0, shard0);
  write_shard_graph(s1, shard1);

  CrossShardMerger merger({kSmallR, 1.2F});
  merger.open({s0, s1});

  std::vector<CrossShardMerger::MergedNode> results;
  merger.merge_all([&](const CrossShardMerger::MergedNode &node) { results.push_back(node); });

  ASSERT_EQ(results.size(), 1U);
  EXPECT_LE(results[0].neighbor_ids_.size(), kSmallR);
  EXPECT_GE(results[0].neighbor_ids_.size(), 1U);

  std::filesystem::remove_all(dir);
}

// ============================================================================
// Regression: cross-shard neighbors at similar distances must not be dropped
// ============================================================================

TEST(CrossShardMergerTest, CrossShardNeighborsNotFalselyPruned) {
  // This test catches a bug where the heuristic prune used
  //   |dist(node,sel) - dist(node,cand)| * alpha < dist(node,cand)
  // as a domination condition. That condition is almost always true for
  // candidates at similar distances (the difference is small), causing the
  // merger to drop nearly all cross-shard candidates after the first.
  //
  // Setup: max_degree=4, but node 0 has 6 unique candidates (3 per shard),
  // forcing the code into the heuristic_prune path (deduped.size() > max_degree).
  // The closest 4 candidates include neighbors from BOTH shards.
  // The buggy code would keep the first selected cross-shard neighbor and
  // prune the remaining cross-shard ones, leaving fewer than 4 neighbors or
  // only same-shard neighbors.

  auto dir = make_temp_dir("cross_shard_not_pruned");

  constexpr uint32_t kTestR = 4;

  // Shard 0: neighbors 1, 2, 3 at distances 10.0, 11.0, 15.0
  // Shard 1: neighbors 4, 5, 6 at distances 10.5, 11.5, 16.0
  // Sorted by distance: 1(10.0), 4(10.5), 2(11.0), 5(11.5), 3(15.0), 6(16.0)
  // Correct top-4: {1, 4, 2, 5} — mixed from both shards.
  // Buggy code: selects 1(shard 0), then checks 4(shard 1) —
  //   |10.0 - 10.5| * 1.2 = 0.6 < 10.5 → falsely "dominated" → pruned!
  //   Then checks 2(shard 0) — same shard as 1 → continue → kept.
  //   Then checks 5(shard 1) — |10.0 - 11.5| * 1.2 = 1.8 < 11.5 → pruned!
  //   Then checks 3(shard 0) — same shard as 1,2 → continue → kept.
  //   Result: {1, 2, 3} — only 3 neighbors, all from shard 0.
  //   Missing: neighbors 4 and 5 from shard 1 are wrongly dropped.

  std::vector<ShardGraphReader::NodeEntry> shard0 = {
      {0, {{1, 10.0F, 0}, {2, 11.0F, 0}, {3, 15.0F, 0}}},
  };
  std::vector<ShardGraphReader::NodeEntry> shard1 = {
      {0, {{4, 10.5F, 1}, {5, 11.5F, 1}, {6, 16.0F, 1}}},
  };

  write_shard_graph(dir / "shard_0.graph", shard0);
  write_shard_graph(dir / "shard_1.graph", shard1);

  CrossShardMerger merger({kTestR, 1.2F});
  merger.open({dir / "shard_0.graph", dir / "shard_1.graph"});

  std::vector<CrossShardMerger::MergedNode> results;
  merger.merge_all([&](const CrossShardMerger::MergedNode &node) { results.push_back(node); });

  ASSERT_EQ(results.size(), 1U);
  ASSERT_EQ(results[0].neighbor_ids_.size(), kTestR)
      << "Should have exactly max_degree neighbors (6 candidates, top-4 selected)";

  // The top-4 by distance must be {1, 4, 2, 5} in that order.
  EXPECT_EQ(results[0].neighbor_ids_[0], 1U);  // dist 10.0 (shard 0)
  EXPECT_EQ(results[0].neighbor_ids_[1], 4U);  // dist 10.5 (shard 1)
  EXPECT_EQ(results[0].neighbor_ids_[2], 2U);  // dist 11.0 (shard 0)
  EXPECT_EQ(results[0].neighbor_ids_[3], 5U);  // dist 11.5 (shard 1)

  std::filesystem::remove_all(dir);
}

// ============================================================================
// Test 5.8: Merged graph has valid structure (bounded degree, no self-loops,
//           no duplicates, non-trivial connectivity)
// ============================================================================

TEST(CrossShardMergerTest, MergedGraphStructureValid) {
  // End-to-end validation: build 2 real shards with ShardVamanaBuilder,
  // merge them, and verify structural invariants on every merged node.
  // This catches both over-pruning (dropped valid neighbors) and
  // under-pruning (degree overflow, duplicates, self-loops).

  constexpr uint32_t kN = 200;
  constexpr uint32_t kD = 4;
  constexpr uint32_t kR = 8;
  constexpr float kAlpha = 1.2F;

  auto dir = make_temp_dir("triangle_zero_fp");

  // Create vectors with 2 clusters
  std::vector<float> data(static_cast<size_t>(kN) * kD);
  for (uint32_t i = 0; i < kN; ++i) {
    float bias = (i < kN / 2) ? 0.0F : 50.0F;
    for (uint32_t d = 0; d < kD; ++d) {
      data[static_cast<size_t>(i) * kD + d] = bias + static_cast<float>(i * 7 + d) * 0.1F;
    }
  }

  auto space = std::make_shared<RawSpace<>>(kN, kD, MetricType::L2);
  space->fit(data.data(), kN);
  auto dist_fn = space->get_dist_func();

  // Manual 2-shard split: first half and second half, with overlap in the middle
  std::vector<uint32_t> shard0_members;
  std::vector<uint32_t> shard1_members;
  for (uint32_t i = 0; i < kN; ++i) {
    if (i < kN * 3 / 5) {
      shard0_members.push_back(i);
    }
    if (i >= kN * 2 / 5) {
      shard1_members.push_back(i);
    }
  }

  // Build shard 0
  std::vector<float> shard0_vecs;
  shard0_vecs.reserve(shard0_members.size() * kD);
  for (auto id : shard0_members) {
    shard0_vecs.insert(shard0_vecs.end(),
                       data.begin() + static_cast<ptrdiff_t>(id) * kD,
                       data.begin() + static_cast<ptrdiff_t>(id + 1) * kD);
  }
  ShardVamanaBuilder<float> builder0(
      std::move(shard0_vecs), kD, shard0_members, dist_fn, {kR, kR * 2, 2, kAlpha, 1.0F, 4096});
  builder0.build();
  auto summary0 = builder0.export_graph(0, dir / "shard_0.graph");

  // Build shard 1
  std::vector<float> shard1_vecs;
  shard1_vecs.reserve(shard1_members.size() * kD);
  for (auto id : shard1_members) {
    shard1_vecs.insert(shard1_vecs.end(),
                       data.begin() + static_cast<ptrdiff_t>(id) * kD,
                       data.begin() + static_cast<ptrdiff_t>(id + 1) * kD);
  }
  ShardVamanaBuilder<float> builder1(
      std::move(shard1_vecs), kD, shard1_members, dist_fn, {kR, kR * 2, 2, kAlpha, 1.0F, 4096});
  builder1.build();
  auto summary1 = builder1.export_graph(1, dir / "shard_1.graph");

  // Merge
  CrossShardMerger merger({kR, kAlpha});
  merger.open({dir / "shard_0.graph", dir / "shard_1.graph"});

  std::unordered_map<uint32_t, std::vector<uint32_t>> merged_results;
  merger.merge_all([&](const CrossShardMerger::MergedNode &node) {
    merged_results[node.global_id_] = node.neighbor_ids_;
  });

  // Verify structural invariants: bounded degree, valid IDs, no self-loops,
  // no duplicates.
  for (const auto &[node_id, nbrs] : merged_results) {
    EXPECT_LE(nbrs.size(), kR) << "Node " << node_id << " exceeds max_degree";

    // Verify all neighbor IDs are valid and not self
    std::unordered_set<uint32_t> seen;
    for (auto nbr_id : nbrs) {
      EXPECT_NE(nbr_id, node_id) << "Self-loop found in node " << node_id;
      EXPECT_LT(nbr_id, kN) << "Invalid neighbor ID " << nbr_id << " in node " << node_id;
      EXPECT_TRUE(seen.insert(nbr_id).second) << "Duplicate neighbor " << nbr_id << " in node "
                                               << node_id;
    }
  }

  // Verify non-trivial connectivity: each node should have at least 1 neighbor.
  for (const auto &[node_id, nbrs] : merged_results) {
    EXPECT_GE(nbrs.size(), 1U) << "Node " << node_id << " has no neighbors after merge";
  }

  std::filesystem::remove_all(dir);
}

// ============================================================================
// Test 5.9: Output DataFile is sequentially ordered, readable by DiskANNSearcher
// ============================================================================

TEST(CrossShardMergerTest, OutputIsSequentiallyOrdered) {
  auto dir = make_temp_dir("sequential_output");

  // 3 shards, some nodes overlap
  std::vector<ShardGraphReader::NodeEntry> shard0 = {
      {0, {{1, 1.0F, 0}, {2, 2.0F, 0}}},
      {1, {{0, 1.0F, 0}, {2, 3.0F, 0}}},
      {2, {{0, 2.0F, 0}, {1, 3.0F, 0}}},
  };
  std::vector<ShardGraphReader::NodeEntry> shard1 = {
      {2, {{3, 1.5F, 1}, {4, 2.5F, 1}}},
      {3, {{2, 1.5F, 1}, {4, 1.0F, 1}}},
      {4, {{2, 2.5F, 1}, {3, 1.0F, 1}}},
  };

  write_shard_graph(dir / "shard_0.graph", shard0);
  write_shard_graph(dir / "shard_1.graph", shard1);

  CrossShardMerger merger({kMaxDegree, 1.2F});
  merger.open({dir / "shard_0.graph", dir / "shard_1.graph"});

  std::vector<CrossShardMerger::MergedNode> results;
  merger.merge_all([&](const CrossShardMerger::MergedNode &node) { results.push_back(node); });

  // Verify sequential ordering
  ASSERT_GE(results.size(), 5U);
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_GT(results[i].global_id_, results[i - 1].global_id_)
        << "Output not sequentially ordered at index " << i;
  }

  // Verify node 2 (overlapping) got union from both shards
  auto it = std::find_if(results.begin(), results.end(),
                         [](const auto &n) { return n.global_id_ == 2; });
  ASSERT_NE(it, results.end());
  std::set<uint32_t> node2_nbrs(it->neighbor_ids_.begin(), it->neighbor_ids_.end());
  // Should have neighbors from both shards: {0, 1} from shard0 and {3, 4} from shard1
  EXPECT_GE(node2_nbrs.size(), 3U);  // At least some from each shard

  // Write merged results to a DataFile and verify readability
  constexpr uint32_t kCapacity = 8;  // Slightly more than 5 nodes
  auto data_path = dir / "test.data";
  auto meta_path = dir / "test.meta";

  // Create MetaFile
  MetaFile meta;
  meta.create(meta_path.string(), kCapacity, kDim, kMaxDegree);

  // Create DataFile
  BufferPool<uint32_t, LRUReplacer> bp(64, kDataBlockSize);
  DataFile<float, uint32_t, LRUReplacer> df(&bp);
  df.create(data_path.string(), kCapacity, kDim, kMaxDegree);

  // Write merged nodes and dummy vectors
  for (const auto &node : results) {
    auto ref = df.get_node(node.global_id_);
    ref.set_neighbors(
        std::span<const uint32_t>(node.neighbor_ids_.data(), node.neighbor_ids_.size()));
    std::vector<float> dummy_vec(kDim, static_cast<float>(node.global_id_));
    ref.set_vector(std::span<const float>(dummy_vec.data(), kDim));
    meta.set_valid(node.global_id_);
  }
  meta.set_entry_point(results[0].global_id_);
  df.flush();
  meta.save();

  // Reopen and verify
  df.close();
  meta.close();

  MetaFile meta2;
  meta2.open(meta_path.string());
  EXPECT_EQ(meta2.entry_point(), results[0].global_id_);
  EXPECT_EQ(meta2.num_active_points(), results.size());

  BufferPool<uint32_t, LRUReplacer> bp2(64, kDataBlockSize);
  DataFile<float, uint32_t, LRUReplacer> df2(&bp2);
  df2.open(data_path.string(), kCapacity, kDim, kMaxDegree, false);

  for (const auto &node : results) {
    ASSERT_TRUE(meta2.is_valid(node.global_id_));
    auto ref = df2.get_node(node.global_id_);
    auto nbrs = ref.neighbors();
    EXPECT_EQ(nbrs.size(), node.neighbor_ids_.size())
        << "Neighbor count mismatch for node " << node.global_id_;
    auto vec = ref.vector();
    EXPECT_EQ(vec.size(), kDim);
    EXPECT_FLOAT_EQ(vec[0], static_cast<float>(node.global_id_));
  }

  std::filesystem::remove_all(dir);
}

// ============================================================================
// Test: Non-overlapping shards produce valid merge
// ============================================================================

TEST(CrossShardMergerTest, NonOverlappingShards) {
  auto dir = make_temp_dir("non_overlapping");

  std::vector<ShardGraphReader::NodeEntry> shard0 = {
      {0, {{1, 1.0F, 0}}},
      {1, {{0, 1.0F, 0}}},
  };
  std::vector<ShardGraphReader::NodeEntry> shard1 = {
      {2, {{3, 1.0F, 1}}},
      {3, {{2, 1.0F, 1}}},
  };

  write_shard_graph(dir / "shard_0.graph", shard0);
  write_shard_graph(dir / "shard_1.graph", shard1);

  CrossShardMerger merger({kMaxDegree, 1.2F});
  merger.open({dir / "shard_0.graph", dir / "shard_1.graph"});

  std::vector<CrossShardMerger::MergedNode> results;
  merger.merge_all([&](const CrossShardMerger::MergedNode &node) { results.push_back(node); });

  ASSERT_EQ(results.size(), 4U);
  EXPECT_EQ(results[0].global_id_, 0U);
  EXPECT_EQ(results[1].global_id_, 1U);
  EXPECT_EQ(results[2].global_id_, 2U);
  EXPECT_EQ(results[3].global_id_, 3U);

  std::filesystem::remove_all(dir);
}

// ============================================================================
// Test: Empty shard is handled gracefully
// ============================================================================

TEST(CrossShardMergerTest, EmptyShardFile) {
  auto dir = make_temp_dir("empty_shard");

  // Create an empty shard file
  auto empty_shard_path = dir / "shard_0.graph";
  std::ofstream empty_shard_file(empty_shard_path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(empty_shard_file.good());
  empty_shard_file.close();

  std::vector<ShardGraphReader::NodeEntry> shard1 = {
      {0, {{1, 1.0F, 1}}},
      {1, {{0, 1.0F, 1}}},
  };
  write_shard_graph(dir / "shard_1.graph", shard1);

  CrossShardMerger merger({kMaxDegree, 1.2F});
  merger.open({empty_shard_path, dir / "shard_1.graph"});

  std::vector<CrossShardMerger::MergedNode> results;
  merger.merge_all([&](const CrossShardMerger::MergedNode &node) { results.push_back(node); });

  ASSERT_EQ(results.size(), 2U);
  EXPECT_EQ(results[0].global_id_, 0U);
  EXPECT_EQ(results[1].global_id_, 1U);

  std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace alaya
