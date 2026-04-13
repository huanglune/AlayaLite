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
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "index/diskann/visited_list.hpp"
#include "index/neighbor.hpp"
#include "space/space_concepts.hpp"
#include "utils/candidate_list.hpp"

namespace alaya {

template <typename IDType = uint32_t>
class CompactNeighborTable {
 public:
  static constexpr IDType kInvalidID = static_cast<IDType>(-1);

  CompactNeighborTable() = default;

  CompactNeighborTable(uint32_t num_nodes, uint32_t max_degree, float slack_factor = 1.0F)
      : max_degree_(static_cast<uint32_t>(static_cast<float>(max_degree) * slack_factor)),
        degrees_(num_nodes, 0),
        neighbors_(static_cast<size_t>(num_nodes) * max_degree_, kInvalidID) {}

  [[nodiscard]] auto num_nodes() const -> uint32_t {
    return static_cast<uint32_t>(degrees_.size());
  }
  [[nodiscard]] auto max_degree() const -> uint32_t { return max_degree_; }
  [[nodiscard]] auto bytes_used() const -> size_t { return neighbors_.size() * sizeof(IDType); }

  [[nodiscard]] auto data(uint32_t node_id) -> IDType * {
    return neighbors_.data() + static_cast<size_t>(node_id) * max_degree_;
  }
  [[nodiscard]] auto data(uint32_t node_id) const -> const IDType * {
    return neighbors_.data() + static_cast<size_t>(node_id) * max_degree_;
  }

  void update(uint32_t node_id, const std::vector<Neighbor<IDType, float>> &neighbors) {
    auto *row = data(node_id);
    uint32_t count = 0;
    for (; count < neighbors.size() && count < max_degree_; ++count) {
      row[count] = neighbors[count].id_;
    }
    degrees_[node_id] = count;
  }

  void update_ids(uint32_t node_id, const std::vector<IDType> &neighbors) {
    auto *row = data(node_id);
    uint32_t count = 0;
    for (; count < neighbors.size() && count < max_degree_; ++count) {
      row[count] = neighbors[count];
    }
    degrees_[node_id] = count;
  }

  [[nodiscard]] auto degree(uint32_t node_id) const -> uint32_t {
    return degrees_[node_id];
  }

  [[nodiscard]] auto neighbors_view(uint32_t node_id) const -> std::span<const IDType> {
    return {data(node_id), degrees_[node_id]};
  }

  auto append(uint32_t node_id, IDType neighbor_id) -> bool {
    auto deg = degrees_[node_id];
    if (deg >= max_degree_) {
      return false;
    }
    data(node_id)[deg] = neighbor_id;
    degrees_[node_id] = deg + 1;
    return true;
  }

  [[nodiscard]] auto neighbors(uint32_t node_id) const -> std::vector<IDType> {
    auto view = neighbors_view(node_id);
    return {view.begin(), view.end()};
  }

 private:
  uint32_t max_degree_{0};
  std::vector<uint32_t> degrees_;
  std::vector<IDType> neighbors_;
};

template <typename GlobalIDType = uint32_t>
struct ShardIdMap {
  std::vector<GlobalIDType> local_to_global_;
  std::unordered_map<GlobalIDType, uint32_t> global_to_local_;

  [[nodiscard]] static auto build(const std::vector<GlobalIDType> &shard_members) -> ShardIdMap {
    ShardIdMap map;
    map.local_to_global_ = shard_members;
    map.global_to_local_.reserve(shard_members.size());
    for (uint32_t local_id = 0; local_id < shard_members.size(); ++local_id) {
      map.global_to_local_.emplace(shard_members[local_id], local_id);
    }
    return map;
  }
};

template <typename DataType = float, typename GlobalIDType = uint32_t>
class ShardVamanaBuilder {
 public:
  using DistanceType = float;
  using LocalIDType = uint32_t;
  using NeighborType = Neighbor<LocalIDType, DistanceType>;
  using GlobalNeighborType = Neighbor<GlobalIDType, DistanceType>;
  using DistanceFn = DistFunc<DataType, DistanceType>;
  using ProgressCallback = std::function<void()>;

  struct Config {
    uint32_t max_degree_{64};
    uint32_t ef_construction_{128};
    uint32_t num_iterations_{1};        ///< Matching official DiskANN (1 pass)
    float alpha_{1.2F};
    float alpha_first_pass_{1.0F};
    uint32_t max_occlusion_size_{750};  ///< Max candidates for pruning (DiskANN maxc)
    size_t max_memory_mb_{4096};
    uint32_t num_threads_{0};           ///< 0 = hardware concurrency, 1 = single-threaded
    bool saturate_graph_{false};        ///< Backfill under-connected nodes to max_degree
  };

  static constexpr float kGraphSlackFactor = 1.3F;

  struct ExportedNode {
    GlobalIDType global_id_{0};
    std::vector<GlobalNeighborType> neighbors_;
  };

  struct ShardExportSummary {
    uint32_t shard_id_{0};
    size_t num_nodes_{0};
    std::filesystem::path graph_path_;
    size_t estimated_peak_memory_bytes_{0};
  };

  static constexpr LocalIDType kInvalidLocalID = CompactNeighborTable<LocalIDType>::kInvalidID;

  ShardVamanaBuilder(std::vector<DataType> vectors,
                     uint32_t dim,
                     std::vector<GlobalIDType> shard_members,
                     DistanceFn dist_fn,
                     Config config = {})
      : vectors_(std::move(vectors)),
        dim_(dim),
        id_map_(ShardIdMap<GlobalIDType>::build(shard_members)),
        dist_fn_(dist_fn),
        config_(config),
        neighbor_table_(static_cast<uint32_t>(id_map_.local_to_global_.size()),
                        config.max_degree_,
                        kGraphSlackFactor),
        node_mutexes_(id_map_.local_to_global_.size()) {
    if (dim_ == 0 || dist_fn_ == nullptr) {
      throw std::invalid_argument("ShardVamanaBuilder requires a dimension and distance function");
    }
    if (vectors_.size() != static_cast<size_t>(num_nodes()) * dim_) {
      throw std::invalid_argument(
          "ShardVamanaBuilder vector buffer size does not match shard layout");
    }
    if (num_nodes() == 0) {
      throw std::invalid_argument("ShardVamanaBuilder requires at least one node");
    }
  }

  [[nodiscard]] static auto estimate_peak_memory_bytes(uint32_t shard_size,
                                                       uint32_t dim,
                                                       uint32_t max_degree) -> size_t {
    auto slack_degree = static_cast<uint32_t>(static_cast<float>(max_degree) * kGraphSlackFactor);
    auto vectors_bytes = static_cast<size_t>(shard_size) * dim * sizeof(DataType);
    auto neighbor_table_bytes =
        static_cast<size_t>(shard_size) * slack_degree * sizeof(LocalIDType);
    auto scratch_bytes = static_cast<size_t>(shard_size) * slack_degree * sizeof(DistanceType);
    return vectors_bytes + neighbor_table_bytes + scratch_bytes;
  }

  [[nodiscard]] auto estimated_peak_memory_bytes() const -> size_t {
    return estimate_peak_memory_bytes(num_nodes(), dim_, config_.max_degree_);
  }

  [[nodiscard]] auto num_nodes() const -> uint32_t {
    return static_cast<uint32_t>(id_map_.local_to_global_.size());
  }

  [[nodiscard]] auto neighbor_table() const -> const CompactNeighborTable<LocalIDType> & {
    return neighbor_table_;
  }

  [[nodiscard]] auto local_to_global() const -> const std::vector<GlobalIDType> & {
    return id_map_.local_to_global_;
  }

  [[nodiscard]] auto global_to_local() const -> const std::unordered_map<GlobalIDType, uint32_t> & {
    return id_map_.global_to_local_;
  }

  [[nodiscard]] auto vectors_loaded() const -> bool { return !vectors_.empty(); }

  [[nodiscard]] auto medoid_local_id() const -> LocalIDType { return medoid_local_id_; }

  void build(ProgressCallback on_progress = nullptr) {
    enforce_memory_budget();
    medoid_local_id_ = compute_medoid();
    bootstrap_medoid();
    for (uint32_t pass = 0; pass < config_.num_iterations_; ++pass) {
      // Single-pass: use alpha_ directly (matching official DiskANN).
      // Multi-pass: pass 0 uses alpha_first_pass_ (strict), subsequent passes use alpha_.
      auto alpha =
          (config_.num_iterations_ == 1) ? config_.alpha_ : (pass == 0 ? config_.alpha_first_pass_ : config_.alpha_);
      build_pass(alpha, on_progress);
    }
    // Final cleanup: prune any over-provisioned nodes back to max_degree
    cleanup_overprovisioned();
  }

  [[nodiscard]] auto export_nodes() const -> std::vector<ExportedNode> {
    std::vector<ExportedNode> exported;
    exported.reserve(num_nodes());

    for (LocalIDType local_id = 0; local_id < num_nodes(); ++local_id) {
      ExportedNode node;
      node.global_id_ = id_map_.local_to_global_[local_id];
      auto local_neighbors = neighbor_table_.neighbors_view(local_id);
      node.neighbors_.reserve(local_neighbors.size());
      for (auto neighbor_local : local_neighbors) {
        auto neighbor_distance = this->distance(local_id, neighbor_local);
        node.neighbors_.emplace_back(id_map_.local_to_global_[neighbor_local], neighbor_distance);
      }
      std::sort(node.neighbors_.begin(), node.neighbors_.end());
      exported.push_back(std::move(node));
    }

    std::sort(exported.begin(), exported.end(), [](const auto &lhs, const auto &rhs) {
      return lhs.global_id_ < rhs.global_id_;
    });
    return exported;
  }

  [[nodiscard]] auto export_graph(uint32_t shard_id, const std::filesystem::path &path)
      -> ShardExportSummary {
    auto dir = path.parent_path();
    if (!dir.empty()) {
      std::filesystem::create_directories(dir);
    }

    auto exported = export_nodes();
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
      throw std::runtime_error("Failed to create shard graph export: " + path.string());
    }

    for (const auto &node : exported) {
      auto count = static_cast<uint16_t>(node.neighbors_.size());
      output.write(reinterpret_cast<const char *>(&node.global_id_), sizeof(node.global_id_));
      output.write(reinterpret_cast<const char *>(&count), sizeof(count));
      for (const auto &neighbor : node.neighbors_) {
        output.write(reinterpret_cast<const char *>(&neighbor.id_), sizeof(neighbor.id_));
        output.write(reinterpret_cast<const char *>(&neighbor.distance_),
                     sizeof(neighbor.distance_));
      }
    }

    ShardExportSummary summary;
    summary.shard_id_ = shard_id;
    summary.num_nodes_ = exported.size();
    summary.graph_path_ = path;
    summary.estimated_peak_memory_bytes_ = estimated_peak_memory_bytes();
    release_memory();
    return summary;
  }

  static auto load_vectors_from_shuffle(const std::filesystem::path &shuffle_path,
                                        uint64_t offset_bytes,
                                        uint64_t num_vectors,
                                        uint32_t dim) -> std::vector<DataType> {
    std::ifstream input(shuffle_path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("Failed to open shuffle file: " + shuffle_path.string());
    }

    std::vector<DataType> vectors(static_cast<size_t>(num_vectors) * dim);
    input.seekg(static_cast<std::streamoff>(offset_bytes));
    input.read(reinterpret_cast<char *>(vectors.data()),
               static_cast<std::streamsize>(vectors.size() * sizeof(DataType)));
    if (!input) {
      throw std::runtime_error("Failed to read shard vectors from shuffle file");
    }
    return vectors;
  }

 private:
  struct BuildScratch {
    CandidateList<float, LocalIDType> candidate_list_;
    std::vector<NeighborType> expanded_nodes_;
    std::vector<float> occlude_factor_;
    std::vector<LocalIDType> neighbor_snapshot_;
    std::vector<LocalIDType> unvisited_ids_;  // Batch buffer for greedy_search

    BuildScratch(uint32_t ef, uint32_t max_occlusion_size) : candidate_list_(static_cast<int>(ef)) {
      expanded_nodes_.reserve(
          std::max(static_cast<size_t>(ef) * 2, static_cast<size_t>(max_occlusion_size)));
      occlude_factor_.reserve(max_occlusion_size);
    }

    void reset(uint32_t ef, uint32_t max_occlusion_size) {
      candidate_list_.resize(ef);
      expanded_nodes_.clear();
      if (expanded_nodes_.capacity() <
          std::max(static_cast<size_t>(ef) * 2, static_cast<size_t>(max_occlusion_size))) {
        expanded_nodes_.reserve(
            std::max(static_cast<size_t>(ef) * 2, static_cast<size_t>(max_occlusion_size)));
      }
      occlude_factor_.clear();
      if (occlude_factor_.capacity() < max_occlusion_size) {
        occlude_factor_.reserve(max_occlusion_size);
      }
    }
  };

  std::vector<DataType> vectors_;
  uint32_t dim_{0};
  ShardIdMap<GlobalIDType> id_map_;
  DistanceFn dist_fn_{nullptr};
  Config config_;
  CompactNeighborTable<LocalIDType> neighbor_table_;
  std::vector<std::mutex> node_mutexes_;
  LocalIDType medoid_local_id_{0};

  void cleanup_overprovisioned() {
    auto num_threads = std::min(effective_threads(), num_nodes());
    std::atomic<uint32_t> next_idx{0};
    constexpr uint32_t kBatchSize = 256;

    auto worker = [this, &next_idx] {
      BuildScratch scratch(config_.ef_construction_, config_.max_occlusion_size_);
      std::vector<NeighborType> selected;
      while (true) {
        auto begin = next_idx.fetch_add(kBatchSize);
        if (begin >= num_nodes()) {
          break;
        }
        auto end = std::min(begin + kBatchSize, num_nodes());
        for (auto node_id = begin; node_id < end; ++node_id) {
          auto existing = neighbor_table_.neighbors_view(node_id);
          if (existing.size() <= config_.max_degree_) {
            continue;
          }
          std::vector<NeighborType> candidates;
          candidates.reserve(existing.size());
          for (auto neighbor_id : existing) {
            candidates.emplace_back(neighbor_id, distance(node_id, neighbor_id));
          }
          std::sort(candidates.begin(), candidates.end());
          robust_prune(node_id, candidates, config_.alpha_, scratch, selected);
          neighbor_table_.update(node_id, selected);
        }
      }
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (uint32_t t = 0; t < num_threads; ++t) {
      threads.emplace_back(worker);
    }
    for (auto &t : threads) {
      t.join();
    }
  }

  void enforce_memory_budget() const {
    constexpr double kBudgetFraction = 0.9;
    auto budget_bytes = static_cast<size_t>(static_cast<double>(config_.max_memory_mb_) * 1024.0 *
                                            1024.0 * kBudgetFraction);
    if (estimated_peak_memory_bytes() > budget_bytes) {
      throw std::runtime_error(
          "ShardVamanaBuilder estimated peak memory exceeds configured budget");
    }
  }

  [[nodiscard]] auto vector_ptr(LocalIDType id) const -> const DataType * {
    return vectors_.data() + static_cast<size_t>(id) * dim_;
  }

  [[nodiscard]] auto distance(LocalIDType lhs, LocalIDType rhs) const -> DistanceType {
    return dist_fn_(vector_ptr(lhs), vector_ptr(rhs), dim_);
  }

  void initialize_random_graph() {
    auto degree_limit = std::min(config_.max_degree_, num_nodes() - 1);
    std::vector<LocalIDType> candidates;
    candidates.reserve(degree_limit);
    std::vector<LocalIDType> permutation(num_nodes());
    std::iota(permutation.begin(), permutation.end(), 0U);
    std::vector<std::pair<uint32_t, uint32_t>> swaps;
    swaps.reserve(degree_limit + 1);
    for (LocalIDType node_id = 0; node_id < num_nodes(); ++node_id) {
      std::minstd_rand rng(node_id + 1U);
      candidates.clear();
      swaps.clear();
      for (uint32_t i = 0; i < num_nodes() && candidates.size() < degree_limit; ++i) {
        std::uniform_int_distribution<uint32_t> dist(i, num_nodes() - 1);
        auto picked = dist(rng);
        if (picked != i) {
          swaps.emplace_back(i, picked);
        }
        std::swap(permutation[i], permutation[picked]);
        auto candidate = permutation[i];
        if (candidate != node_id) {
          candidates.push_back(candidate);
        }
      }
      for (const auto &swap : std::views::reverse(swaps)) {
        std::swap(permutation[swap.first], permutation[swap.second]);
      }
      neighbor_table_.update_ids(node_id, candidates);
    }
  }

  [[nodiscard]] auto compute_medoid() const -> LocalIDType {
    std::vector<double> centroid(dim_, 0.0);
    for (LocalIDType i = 0; i < num_nodes(); ++i) {
      const auto *vec = vector_ptr(i);
      for (uint32_t d = 0; d < dim_; ++d) {
        centroid[d] += static_cast<double>(vec[d]);
      }
    }

    std::vector<DataType> centroid_vec(dim_);
    for (uint32_t d = 0; d < dim_; ++d) {
      centroid_vec[d] = static_cast<DataType>(centroid[d] / static_cast<double>(num_nodes()));
    }

    LocalIDType medoid = 0;
    auto best_dist = dist_fn_(centroid_vec.data(), vector_ptr(0), dim_);
    for (LocalIDType i = 1; i < num_nodes(); ++i) {
      auto cur_dist = dist_fn_(centroid_vec.data(), vector_ptr(i), dim_);
      if (cur_dist < best_dist) {
        best_dist = cur_dist;
        medoid = i;
      }
    }
    return medoid;
  }

  /// Bootstrap the medoid with its nearest neighbors so greedy_search on
  /// the empty graph can immediately find reachable nodes. Equivalent to
  /// official DiskANN's "frozen point" mechanism.
  void bootstrap_medoid() {
    auto degree_limit = std::min(config_.max_degree_, num_nodes() - 1);
    std::vector<NeighborType> candidates;
    candidates.reserve(num_nodes());
    for (LocalIDType i = 0; i < num_nodes(); ++i) {
      if (i == medoid_local_id_) {
        continue;
      }
      candidates.emplace_back(i, distance(medoid_local_id_, i));
    }
    std::sort(candidates.begin(), candidates.end());
    if (candidates.size() > degree_limit) {
      candidates.resize(degree_limit);
    }
    neighbor_table_.update(medoid_local_id_, candidates);
  }

  [[nodiscard]] auto effective_threads() const -> uint32_t {
    auto t = config_.num_threads_;
    if (t == 0) {
      t = std::min(std::thread::hardware_concurrency(), 60U);
    }
    return std::max(1U, t);
  }

  void build_pass(float alpha, const ProgressCallback &on_progress) {
    constexpr uint32_t kBatchSize = 2048;

    auto num_threads = std::min(effective_threads(), num_nodes());
    std::atomic<uint32_t> next_idx{0};

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (uint32_t t = 0; t < num_threads; ++t) {
      threads.emplace_back([this, &on_progress, alpha, &next_idx] {
        BuildScratch scratch(config_.ef_construction_, config_.max_occlusion_size_);
        std::vector<NeighborType> selected;
        selected.reserve(config_.max_degree_);

        while (true) {
          auto begin = next_idx.fetch_add(kBatchSize);
          if (begin >= num_nodes()) {
            break;
          }
          auto end = std::min(begin + kBatchSize, num_nodes());
          for (auto node_id = begin; node_id < end; ++node_id) {
            greedy_search(node_id, config_.ef_construction_, scratch);
            auto &candidates = scratch.expanded_nodes_;
            robust_prune(node_id, candidates, alpha, scratch, selected);
            {
              std::lock_guard<std::mutex> lock(node_mutexes_[node_id]);
              neighbor_table_.update(node_id, selected);
            }
            for (const auto &neighbor : selected) {
              add_reverse_edge(neighbor.id_, node_id, alpha, scratch);
            }
            if (on_progress) {
              on_progress();
            }
          }
        }
      });
    }
    for (auto &t : threads) {
      t.join();
    }
  }

  void merge_existing_neighbors(LocalIDType node_id, std::vector<NeighborType> &candidates) const {
    auto existing = neighbor_table_.neighbors_view(node_id);
    for (auto neighbor_id : existing) {
      if (neighbor_id >= num_nodes()) {
        continue;
      }  // skip stale IDs from concurrent writes
      candidates.emplace_back(neighbor_id, distance(node_id, neighbor_id), true);
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto &lhs, const auto &rhs) {
      if (lhs.id_ != rhs.id_) {
        return lhs.id_ < rhs.id_;
      }
      return lhs.distance_ < rhs.distance_;
    });
    auto last =
        std::unique(candidates.begin(), candidates.end(), [](const auto &lhs, const auto &rhs) {
          return lhs.id_ == rhs.id_;
        });
    candidates.erase(last, candidates.end());
    std::sort(candidates.begin(), candidates.end());
  }

  void greedy_search(LocalIDType query_id, uint32_t ef, BuildScratch &scratch) const {
    auto &visited = diskann::GlobalVisitedList::get(num_nodes());
    scratch.reset(ef, config_.max_occlusion_size_);
    auto &candidates = scratch.candidate_list_;
    auto &expanded_nodes = scratch.expanded_nodes_;
    auto &unvisited = scratch.unvisited_ids_;

    const auto *query_vec = vector_ptr(query_id);

    visited.mark(medoid_local_id_);
    candidates.insert(medoid_local_id_, dist_fn_(query_vec, vector_ptr(medoid_local_id_), dim_));

    while (candidates.has_next()) {
      auto cur_dist = candidates.dist(candidates.cur_);
      auto cur_id = candidates.pop();
      expanded_nodes.emplace_back(cur_id, cur_dist);

      // Snapshot + collect unvisited neighbors in batch
      auto nbr_view = neighbor_table_.neighbors_view(cur_id);
      scratch.neighbor_snapshot_.assign(nbr_view.begin(), nbr_view.end());
      unvisited.clear();
      for (auto neighbor_id : scratch.neighbor_snapshot_) {
        if (neighbor_id >= num_nodes()) {
          continue;
        }
        if (visited.is_visited(neighbor_id)) {
          continue;
        }
        visited.mark(neighbor_id);
        unvisited.push_back(neighbor_id);
      }

      // Batch distance computation with prefetch
      for (size_t idx = 0; idx < unvisited.size(); ++idx) {
        if (idx + 1 < unvisited.size()) {
          __builtin_prefetch(vector_ptr(unvisited[idx + 1]), 0, 0);
        }
        candidates.insert(unvisited[idx], dist_fn_(query_vec, vector_ptr(unvisited[idx]), dim_));
      }
    }

    // Include unexpanded candidates from the search frontier in the pruning pool.
    // Official DiskANN prunes from the full pool (expanded + unexpanded).
    for (size_t idx = 0; idx < candidates.size(); ++idx) {
      auto nid = candidates.id(static_cast<LocalIDType>(idx));
      if (nid != query_id) {
        expanded_nodes.emplace_back(nid, candidates.dist(static_cast<LocalIDType>(idx)));
      }
    }

    // Dedup (expanded nodes may overlap with candidate list entries)
    std::sort(expanded_nodes.begin(), expanded_nodes.end(), [](const auto &a, const auto &b) {
      return a.id_ < b.id_ || (a.id_ == b.id_ && a.distance_ < b.distance_);
    });
    auto last = std::unique(expanded_nodes.begin(), expanded_nodes.end(),
                            [](const auto &a, const auto &b) { return a.id_ == b.id_; });
    expanded_nodes.erase(last, expanded_nodes.end());
    std::sort(expanded_nodes.begin(), expanded_nodes.end());
  }

  [[nodiscard]] auto greedy_search(LocalIDType query_id, uint32_t ef) const
      -> std::vector<NeighborType> {
    BuildScratch scratch(ef, config_.max_occlusion_size_);
    greedy_search(query_id, ef, scratch);
    return scratch.expanded_nodes_;
  }

  void robust_prune(LocalIDType node_id,
                    std::vector<NeighborType> &candidates,
                    float alpha,
                    BuildScratch &scratch,
                    std::vector<NeighborType> &selected) const {
    // Truncate candidate pool at maxc (matching DiskANN behavior)
    if (candidates.size() > config_.max_occlusion_size_) {
      candidates.resize(config_.max_occlusion_size_);
    }

    selected.clear();
    if (selected.capacity() < config_.max_degree_) {
      selected.reserve(config_.max_degree_);
    }

    // Per-candidate occlusion factor: max ratio of dist(query,cand)/dist(selected,cand)
    scratch.occlude_factor_.assign(candidates.size(), 0.0F);
    auto &occlude_factor = scratch.occlude_factor_;

    // Multi-round alpha escalation: 1.0 -> 1.2 -> 1.44 -> ...
    constexpr float kAlphaStep = 1.2F;
    float cur_alpha = 1.0F;
    while (cur_alpha <= alpha && selected.size() < config_.max_degree_) {
      for (size_t i = 0; i < candidates.size(); ++i) {
        if (selected.size() >= config_.max_degree_) {
          break;
        }
        if (occlude_factor[i] > cur_alpha) {
          continue;
        }

        // Select this candidate
        occlude_factor[i] = std::numeric_limits<float>::max();
        if (candidates[i].id_ != node_id) {
          selected.push_back(candidates[i]);
        }

        // Update occlusion factors for remaining candidates
        for (size_t j = i + 1; j < candidates.size(); ++j) {
          if (occlude_factor[j] > alpha) {
            continue;
          }
          float djk = distance(candidates[j].id_, candidates[i].id_);
          if (djk == 0.0F) {
            occlude_factor[j] = std::numeric_limits<float>::max();
          } else {
            occlude_factor[j] = std::max(occlude_factor[j], candidates[j].distance_ / djk);
          }
        }
      }
      cur_alpha *= kAlphaStep;
    }

    // Saturation: if enabled and alpha > 1, backfill under-connected nodes.
    if (config_.saturate_graph_ && alpha > 1.0F && selected.size() < config_.max_degree_) {
      for (size_t i = 0; i < candidates.size(); ++i) {
        if (selected.size() >= config_.max_degree_) {
          break;
        }
        if (occlude_factor[i] >= std::numeric_limits<float>::max()) {
          continue;  // already selected
        }
        if (candidates[i].id_ == node_id) {
          continue;
        }
        selected.push_back(candidates[i]);
      }
    }
  }

  [[nodiscard]] auto robust_prune(LocalIDType node_id,
                                  std::vector<NeighborType> &candidates,
                                  float alpha) const -> std::vector<NeighborType> {
    BuildScratch scratch(config_.ef_construction_, config_.max_occlusion_size_);
    std::vector<NeighborType> selected;
    robust_prune(node_id, candidates, alpha, scratch, selected);
    return selected;
  }

  void add_reverse_edge(LocalIDType src, LocalIDType dst, float alpha, BuildScratch &scratch) {
    std::vector<NeighborType> candidates;
    bool needs_prune = false;

    {
      std::lock_guard<std::mutex> lock(node_mutexes_[src]);
      auto existing = neighbor_table_.neighbors_view(src);
      if (std::find(existing.begin(), existing.end(), dst) != existing.end()) {
        return;
      }

      auto slack_limit =
          static_cast<size_t>(static_cast<float>(config_.max_degree_) * kGraphSlackFactor);

      if (existing.size() < slack_limit) {
        if (!neighbor_table_.append(src, dst)) {
          throw std::logic_error("append failed below slack limit");
        }
        return;
      }

      // At slack limit: snapshot neighbors and release lock before pruning
      candidates.reserve(existing.size() + 1);
      for (auto neighbor_id : existing) {
        candidates.emplace_back(neighbor_id, distance(src, neighbor_id));
      }
      candidates.emplace_back(dst, distance(src, dst));
      needs_prune = true;
    }

    if (needs_prune) {
      std::sort(candidates.begin(), candidates.end());
      std::vector<NeighborType> selected;
      robust_prune(src, candidates, alpha, scratch, selected);
      std::lock_guard<std::mutex> lock(node_mutexes_[src]);
      neighbor_table_.update(src, selected);
    }
  }

  void add_reverse_edge(LocalIDType src, LocalIDType dst, float alpha) {
    BuildScratch scratch(config_.ef_construction_, config_.max_occlusion_size_);
    add_reverse_edge(src, dst, alpha, scratch);
  }

  friend struct ShardVamanaBuilderTestAccess;

  void release_memory() {
    std::vector<DataType>().swap(vectors_);
    neighbor_table_ = CompactNeighborTable<LocalIDType>();
    node_mutexes_ = std::vector<std::mutex>();
    id_map_.local_to_global_.clear();
    id_map_.global_to_local_.clear();
  }
};

}  // namespace alaya
