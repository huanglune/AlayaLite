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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
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

  CompactNeighborTable(uint32_t num_nodes, uint32_t max_degree)
      : max_degree_(max_degree),
        neighbors_(static_cast<size_t>(num_nodes) * max_degree, kInvalidID) {}

  [[nodiscard]] auto num_nodes() const -> uint32_t {
    return max_degree_ == 0 ? 0 : static_cast<uint32_t>(neighbors_.size() / max_degree_);
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
    for (; count < max_degree_; ++count) {
      row[count] = kInvalidID;
    }
  }

  void update_ids(uint32_t node_id, const std::vector<IDType> &neighbors) {
    auto *row = data(node_id);
    uint32_t count = 0;
    for (; count < neighbors.size() && count < max_degree_; ++count) {
      row[count] = neighbors[count];
    }
    for (; count < max_degree_; ++count) {
      row[count] = kInvalidID;
    }
  }

  [[nodiscard]] auto neighbors(uint32_t node_id) const -> std::vector<IDType> {
    std::vector<IDType> snapshot;
    auto *row = data(node_id);
    for (uint32_t i = 0; i < max_degree_; ++i) {
      if (row[i] == kInvalidID) {
        break;
      }
      snapshot.push_back(row[i]);
    }
    return snapshot;
  }

 private:
  uint32_t max_degree_{0};
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

  struct Config {
    uint32_t max_degree_{64};
    uint32_t ef_construction_{128};
    uint32_t num_iterations_{2};
    float alpha_{1.2F};
    float alpha_first_pass_{1.0F};
    size_t max_memory_mb_{4096};
    uint32_t num_threads_{0};  ///< 0 = hardware concurrency, 1 = single-threaded
  };

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
                        config.max_degree_),
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
    auto vectors_bytes = static_cast<size_t>(shard_size) * dim * sizeof(DataType);
    auto neighbor_table_bytes = static_cast<size_t>(shard_size) * max_degree * sizeof(LocalIDType);
    auto scratch_bytes = static_cast<size_t>(shard_size) * max_degree * sizeof(DistanceType);
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

  void build() {
    enforce_memory_budget();
    initialize_random_graph();
    medoid_local_id_ = compute_medoid();
    for (uint32_t pass = 0; pass < config_.num_iterations_; ++pass) {
      auto alpha = pass == 0 ? config_.alpha_first_pass_ : config_.alpha_;
      build_pass(alpha);
    }
  }

  [[nodiscard]] auto export_nodes() const -> std::vector<ExportedNode> {
    std::vector<ExportedNode> exported;
    exported.reserve(num_nodes());

    for (LocalIDType local_id = 0; local_id < num_nodes(); ++local_id) {
      ExportedNode node;
      node.global_id_ = id_map_.local_to_global_[local_id];
      auto local_neighbors = neighbor_table_.neighbors(local_id);
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
  std::vector<DataType> vectors_;
  uint32_t dim_{0};
  ShardIdMap<GlobalIDType> id_map_;
  DistanceFn dist_fn_{nullptr};
  Config config_;
  CompactNeighborTable<LocalIDType> neighbor_table_;
  std::vector<std::mutex> node_mutexes_;
  LocalIDType medoid_local_id_{0};

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
    std::vector<LocalIDType> candidates;
    candidates.reserve(config_.max_degree_);
    for (LocalIDType node_id = 0; node_id < num_nodes(); ++node_id) {
      std::mt19937 rng(node_id);
      std::uniform_int_distribution<LocalIDType> dist(0, num_nodes() - 1);
      candidates.clear();
      while (candidates.size() < config_.max_degree_ &&
             candidates.size() < static_cast<size_t>(num_nodes() - 1)) {
        auto candidate = dist(rng);
        if (candidate == node_id) {
          continue;
        }
        if (std::find(candidates.begin(), candidates.end(), candidate) == candidates.end()) {
          candidates.push_back(candidate);
        }
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

  [[nodiscard]] auto effective_threads() const -> uint32_t {
    auto t = config_.num_threads_;
    if (t == 0) {
      t = std::min(std::thread::hardware_concurrency(), 60U);
    }
    return std::max(1U, t);
  }

  void build_pass(float alpha) {
    std::vector<LocalIDType> perm(num_nodes());
    std::iota(perm.begin(), perm.end(), 0U);
    std::shuffle(perm.begin(), perm.end(), std::mt19937(42));

    auto num_threads = std::min(effective_threads(), num_nodes());
    if (num_threads <= 1) {
      build_pass_single(perm, alpha);
      return;
    }

    // Partition work into chunks for each thread
    auto n = perm.size();
    auto chunk = (n + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (uint32_t t = 0; t < num_threads; ++t) {
      auto begin = t * chunk;
      auto end = std::min(begin + chunk, n);
      if (begin >= n) break;
      threads.emplace_back([this, &perm, alpha, begin, end] {
        std::vector<NeighborType> candidates;
        std::vector<NeighborType> selected;
        for (auto i = begin; i < end; ++i) {
          auto node_id = perm[i];
          candidates = greedy_search(node_id, config_.ef_construction_);
          merge_existing_neighbors(node_id, candidates);
          selected = robust_prune(node_id, candidates, alpha);
          {
            std::lock_guard<std::mutex> lock(node_mutexes_[node_id]);
            neighbor_table_.update(node_id, selected);
          }
          for (const auto &neighbor : selected) {
            add_reverse_edge(neighbor.id_, node_id, alpha);
          }
        }
      });
    }
    for (auto &t : threads) {
      t.join();
    }
  }

  void build_pass_single(const std::vector<LocalIDType> &perm, float alpha) {
    std::vector<NeighborType> candidates;
    std::vector<NeighborType> selected;
    for (auto node_id : perm) {
      candidates = greedy_search(node_id, config_.ef_construction_);
      merge_existing_neighbors(node_id, candidates);
      selected = robust_prune(node_id, candidates, alpha);
      neighbor_table_.update(node_id, selected);
      for (const auto &neighbor : selected) {
        add_reverse_edge(neighbor.id_, node_id, alpha);
      }
    }
  }

  void merge_existing_neighbors(LocalIDType node_id, std::vector<NeighborType> &candidates) const {
    auto existing = neighbor_table_.neighbors(node_id);
    for (auto neighbor_id : existing) {
      if (neighbor_id >= num_nodes()) continue;  // skip stale IDs from concurrent writes
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

  [[nodiscard]] auto greedy_search(LocalIDType query_id, uint32_t ef) const
      -> std::vector<NeighborType> {
    auto &visited = diskann::GlobalVisitedList::get(num_nodes());
    CandidateList<float, LocalIDType> candidates(num_nodes(), ef + config_.max_degree_);

    visited.mark(medoid_local_id_);
    candidates.insert(medoid_local_id_, distance(query_id, medoid_local_id_));

    while (candidates.has_next()) {
      if (candidates.is_full() &&
          candidates.dist(candidates.cur_) > candidates.dist(candidates.size() - 1)) {
        break;
      }

      auto cur_id = candidates.pop();
      for (auto neighbor_id : neighbor_table_.neighbors(cur_id)) {
        if (neighbor_id >= num_nodes()) {
          continue;  // skip stale IDs from concurrent writes
        }
        if (visited.is_visited(neighbor_id)) {
          continue;
        }
        visited.mark(neighbor_id);
        candidates.insert(neighbor_id, distance(query_id, neighbor_id));
      }
    }

    size_t final_size = std::min<size_t>(candidates.size(), ef);
    std::vector<NeighborType> results;
    results.reserve(final_size);
    for (size_t i = 0; i < final_size; ++i) {
      results.emplace_back(candidates.id(i), candidates.dist(i), true);
    }
    return results;
  }

  [[nodiscard]] auto robust_prune(LocalIDType node_id,
                                  const std::vector<NeighborType> &candidates,
                                  float alpha) const -> std::vector<NeighborType> {
    std::vector<NeighborType> selected;
    selected.reserve(config_.max_degree_);

    auto inflate = [alpha](float dist) -> float {
      return dist >= 0.0F ? dist * alpha : dist / alpha;
    };

    for (const auto &candidate : candidates) {
      if (selected.size() >= config_.max_degree_) {
        break;
      }
      if (candidate.id_ == node_id) {
        continue;
      }

      bool dominated = false;
      for (const auto &chosen : selected) {
        if (inflate(distance(candidate.id_, chosen.id_)) < candidate.distance_) {
          dominated = true;
          break;
        }
      }

      if (!dominated) {
        selected.push_back(candidate);
      }
    }

    return selected;
  }

  void add_reverse_edge(LocalIDType src, LocalIDType dst, float alpha) {
    std::lock_guard<std::mutex> lock(node_mutexes_[src]);
    auto existing = neighbor_table_.neighbors(src);
    if (std::find(existing.begin(), existing.end(), dst) != existing.end()) {
      return;
    }

    if (existing.size() < config_.max_degree_) {
      existing.push_back(dst);
      std::vector<NeighborType> expanded;
      expanded.reserve(existing.size());
      for (auto neighbor_id : existing) {
        expanded.emplace_back(neighbor_id, distance(src, neighbor_id));
      }
      std::sort(expanded.begin(), expanded.end());
      neighbor_table_.update(src, expanded);
      return;
    }

    std::vector<NeighborType> candidates;
    candidates.reserve(existing.size() + 1);
    for (auto neighbor_id : existing) {
      candidates.emplace_back(neighbor_id, distance(src, neighbor_id));
    }
    candidates.emplace_back(dst, distance(src, dst));
    std::sort(candidates.begin(), candidates.end());
    neighbor_table_.update(src, robust_prune(src, candidates, alpha));
  }

  void release_memory() {
    std::vector<DataType>().swap(vectors_);
    neighbor_table_ = CompactNeighborTable<LocalIDType>();
    node_mutexes_ = std::vector<std::mutex>();
    id_map_.local_to_global_.clear();
    id_map_.global_to_local_.clear();
  }
};

}  // namespace alaya
