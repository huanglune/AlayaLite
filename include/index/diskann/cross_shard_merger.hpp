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

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace alaya {

/**
 * @brief A single neighbor entry with shard origin for cross-shard merge.
 */
struct MergeNeighbor {
  uint32_t id_{0};
  float distance_{0.0F};
  uint32_t origin_shard_{0};

  auto operator<(const MergeNeighbor &other) const -> bool { return distance_ < other.distance_; }
};

/**
 * @brief Reader for a single shard graph file produced by ShardVamanaBuilder.
 *
 * Reads the binary format: [global_id (4B)][count (2B)][{nbr_id (4B), dist (4B)} x count]...
 * Nodes are sorted by global_id in the file.
 */
class ShardGraphReader {
 public:
  struct NodeEntry {
    uint32_t global_id_{0};
    std::vector<MergeNeighbor> neighbors_;
  };

  ShardGraphReader(const std::filesystem::path &path, uint32_t shard_id)
      : shard_id_(shard_id), stream_(path, std::ios::binary) {
    if (!stream_) {
      throw std::runtime_error("Failed to open shard graph file: " + path.string());
    }
    advance();
  }

  [[nodiscard]] auto done() const -> bool { return done_; }
  [[nodiscard]] auto current() const -> const NodeEntry & { return current_; }
  [[nodiscard]] auto shard_id() const -> uint32_t { return shard_id_; }

  void advance() {
    uint32_t global_id = 0;
    if (!stream_.read(reinterpret_cast<char *>(&global_id), sizeof(global_id))) {
      done_ = true;
      return;
    }

    uint16_t count = 0;
    stream_.read(reinterpret_cast<char *>(&count), sizeof(count));

    current_.global_id_ = global_id;
    current_.neighbors_.resize(count);
    for (uint16_t i = 0; i < count; ++i) {
      uint32_t nbr_id = 0;
      float dist = 0.0F;
      stream_.read(reinterpret_cast<char *>(&nbr_id), sizeof(nbr_id));
      stream_.read(reinterpret_cast<char *>(&dist), sizeof(dist));
      current_.neighbors_[i] = {nbr_id, dist, shard_id_};
    }
  }

 private:
  uint32_t shard_id_{0};
  std::ifstream stream_;
  NodeEntry current_;
  bool done_{false};
};

/**
 * @brief Cross-shard merge with distance-ordered neighbor selection.
 *
 * Implements K-way merge of shard graph files, union+dedup of neighbor lists,
 * and distance-ordered selection (top-R) that avoids all vector I/O.
 */
class CrossShardMerger {
 public:
  struct Config {
    uint32_t max_degree_ = 64;
    float alpha_ = 1.2F;
  };

  /**
   * @brief Result of merging a single node's neighbor lists from multiple shards.
   */
  struct MergedNode {
    uint32_t global_id_{0};
    std::vector<uint32_t> neighbor_ids_;
  };

  CrossShardMerger() = default;
  explicit CrossShardMerger(Config config) : config_(config) {}

  /**
   * @brief Open K shard graph files for K-way merge.
   *
   * @param shard_paths Paths to shard_k.graph files, indexed by shard_id.
   */
  void open(const std::vector<std::filesystem::path> &shard_paths) {
    readers_.clear();
    readers_.reserve(shard_paths.size());
    for (uint32_t shard_id = 0; shard_id < shard_paths.size(); ++shard_id) {
      readers_.emplace_back(shard_paths[shard_id], shard_id);
    }
  }

  /**
   * @brief Run the full K-way merge, calling the visitor for each merged node in global_id order.
   *
   * @param visitor Called with (global_id, neighbor_ids) for each node.
   */
  void merge_all(const std::function<void(const MergedNode &)> &visitor) {
    while (true) {
      // Find the minimum global_id across all active readers
      uint32_t min_id = std::numeric_limits<uint32_t>::max();
      for (const auto &reader : readers_) {
        if (!reader.done() && reader.current().global_id_ < min_id) {
          min_id = reader.current().global_id_;
        }
      }
      if (min_id == std::numeric_limits<uint32_t>::max()) {
        break;
      }

      // Collect all neighbor entries for this global_id from all shards
      std::vector<MergeNeighbor> candidates;
      for (auto &reader : readers_) {
        if (!reader.done() && reader.current().global_id_ == min_id) {
          const auto &neighbors = reader.current().neighbors_;
          candidates.insert(candidates.end(), neighbors.begin(), neighbors.end());
          reader.advance();
        }
      }

      // Union + dedup + prune
      auto merged = merge_node(min_id, std::move(candidates));
      visitor(merged);
    }
  }

  /**
   * @brief Merge a single node's neighbor list from multiple shards.
   *
   * Steps:
   * 1. Dedup by neighbor_id (keep min distance, union origin shards)
   * 2. Sort by distance
   * 3. Select top max_degree by distance
   */
  [[nodiscard]] auto merge_node(uint32_t node_id, std::vector<MergeNeighbor> candidates)
      -> MergedNode {
    // Remove self-loops
    candidates.erase(std::remove_if(candidates.begin(),
                                    candidates.end(),
                                    [node_id](const auto &c) {
                                      return c.id_ == node_id;
                                    }),
                     candidates.end());

    // Dedup: sort by id, keep min distance per id
    std::sort(candidates.begin(), candidates.end(), [](const auto &lhs, const auto &rhs) {
      return lhs.id_ < rhs.id_ || (lhs.id_ == rhs.id_ && lhs.distance_ < rhs.distance_);
    });

    std::vector<MergeNeighbor> deduped;
    deduped.reserve(candidates.size());
    for (const auto &c : candidates) {
      if (!deduped.empty() && deduped.back().id_ == c.id_) {
        // Keep min distance; mark as multi-origin by setting origin_shard to max
        if (c.distance_ < deduped.back().distance_) {
          deduped.back().distance_ = c.distance_;
        }
        // If from different shards, mark as cross-shard (mixed origin)
        if (c.origin_shard_ != deduped.back().origin_shard_) {
          deduped.back().origin_shard_ = kMixedOrigin;
        }
      } else {
        deduped.push_back(c);
      }
    }

    // Sort by distance ascending
    std::sort(deduped.begin(), deduped.end());

    // Distance-ordered selection
    MergedNode result;
    result.global_id_ = node_id;

    if (deduped.size() <= config_.max_degree_) {
      result.neighbor_ids_.reserve(deduped.size());
      for (const auto &n : deduped) {
        result.neighbor_ids_.push_back(n.id_);
      }
      return result;
    }

    result.neighbor_ids_ = heuristic_prune(deduped);
    return result;
  }

 private:
  static constexpr uint32_t kMixedOrigin = std::numeric_limits<uint32_t>::max();

  Config config_;
  std::vector<ShardGraphReader> readers_;

  /**
   * @brief Distance-ordered selection (top-R by distance after dedup).
   *
   * Cross-shard domination-based pruning is not attempted because proving
   * domination requires dist(sel, cand) which needs vector I/O. Without
   * vectors, the triangle inequality lower bound can only prove
   * NON-domination, never domination.
   *
   * Trade-offs vs full robust_prune:
   * - May drop far-but-diverse "highway" edges that robust_prune would keep
   *   for graph navigability. This can reduce recall slightly.
   * - Will never incorrectly prune a close neighbor (no heuristic to misfire).
   * - Intra-shard diversity is preserved by each shard's robust_prune.
   * - Pure ALU: no vector I/O.
   */
  [[nodiscard]] auto heuristic_prune(const std::vector<MergeNeighbor> &sorted_candidates)
      -> std::vector<uint32_t> {
    std::vector<uint32_t> ids;
    ids.reserve(config_.max_degree_);

    for (const auto &cand : sorted_candidates) {
      if (ids.size() >= config_.max_degree_) {
        break;
      }
      ids.push_back(cand.id_);
    }
    return ids;
  }
};

}  // namespace alaya
