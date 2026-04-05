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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "space/space_concepts.hpp"
#include "utils/kmeans.hpp"

namespace alaya {

template <typename DataType = float, typename IDType = uint32_t>
struct PartitionedShardLayout {
  static constexpr uint32_t kInvalidShard = std::numeric_limits<uint32_t>::max();

  struct PersistedNodeToShards {
    uint32_t num_nodes_{0};
    uint32_t num_shards_{0};
    uint32_t max_assignments_{0};
    std::vector<uint32_t> assignments_;
  };

  struct PersistedShardMembers {
    uint32_t num_shards_{0};
    std::vector<std::vector<uint32_t>> members_;
  };

  uint32_t num_nodes_{0};
  uint32_t num_shards_{0};
  uint32_t dim_{0};
  uint32_t max_assignments_{0};
  uint32_t shard_capacity_{0};
  uint32_t shard_size_cap_{0};

  std::vector<DataType> centroids_;
  std::vector<uint32_t> primary_shards_;
  std::vector<uint32_t> node_to_shards_;
  std::vector<float> assignment_distances_;
  std::vector<std::vector<uint32_t>> shard_members_;
  std::vector<uint64_t> shuffle_offsets_;
  std::vector<uint64_t> shuffle_counts_;

  std::filesystem::path node_to_shards_path_;
  std::filesystem::path shard_members_path_;
  std::filesystem::path shuffle_path_;

  [[nodiscard]] auto assignment_count(uint32_t node_id) const -> uint32_t {
    uint32_t count = 0;
    for (uint32_t slot = 0; slot < max_assignments_; ++slot) {
      if (node_to_shards_[static_cast<size_t>(node_id) * max_assignments_ + slot] !=
          kInvalidShard) {
        ++count;
      }
    }
    return count;
  }

  [[nodiscard]] auto assignment_at(uint32_t node_id, uint32_t slot) const -> uint32_t {
    return node_to_shards_[static_cast<size_t>(node_id) * max_assignments_ + slot];
  }

  [[nodiscard]] auto distance_at(uint32_t node_id, uint32_t slot) const -> float {
    return assignment_distances_[static_cast<size_t>(node_id) * max_assignments_ + slot];
  }

  [[nodiscard]] auto has_assignment(uint32_t node_id, uint32_t shard_id) const -> bool {
    for (uint32_t slot = 0; slot < max_assignments_; ++slot) {
      if (assignment_at(node_id, slot) == shard_id) {
        return true;
      }
    }
    return false;
  }
};

template <typename DataType = float, typename IDType = uint32_t>
class KMeansPartitioner {
 public:
  using Layout = PartitionedShardLayout<DataType, IDType>;

  struct Config {
    size_t max_memory_mb_{4096};
    float sample_rate_{0.02F};
    uint32_t overlap_factor_{2};
    float shard_overflow_factor_{1.2F};
    uint32_t kmeans_max_iter_{20};
    uint32_t kmeans_num_trials_{3};
    uint32_t sample_seed_{42};
  };

  explicit KMeansPartitioner(Config config = {}) : config_(std::move(config)) {}

  [[nodiscard]] auto config() const -> const Config & { return config_; }

  /// Must match ShardVamanaBuilder::estimate_peak_memory_bytes per-node cost
  /// (vectors + neighbor_table + scratch).
  [[nodiscard]] static auto estimate_per_node_memory(uint32_t dim, uint32_t max_degree) -> size_t {
    auto vectors_bytes = static_cast<size_t>(dim) * sizeof(DataType);
    auto neighbor_table_bytes = static_cast<size_t>(max_degree) * sizeof(IDType);
    auto scratch_bytes = static_cast<size_t>(max_degree) * sizeof(float);
    return vectors_bytes + neighbor_table_bytes + scratch_bytes;
  }

  [[nodiscard]] static auto compute_shard_capacity(uint32_t dim,
                                                   uint32_t max_degree,
                                                   size_t max_memory_mb) -> uint32_t {
    constexpr double kUsableFraction = 0.9;
    auto usable_bytes =
        static_cast<size_t>(static_cast<double>(max_memory_mb) * 1024.0 * 1024.0 * kUsableFraction);
    auto per_node = std::max<size_t>(1, estimate_per_node_memory(dim, max_degree));
    return static_cast<uint32_t>(std::max<size_t>(1, usable_bytes / per_node));
  }

  [[nodiscard]] static auto compute_num_shards(uint32_t num_nodes, uint32_t shard_capacity)
      -> uint32_t {
    if (num_nodes == 0) {
      return 0;
    }
    auto clamped_capacity = std::max<uint32_t>(1, shard_capacity);
    return std::max<uint32_t>(1, (num_nodes + clamped_capacity - 1) / clamped_capacity);
  }

  [[nodiscard]] static auto compute_shard_size_cap(uint32_t num_nodes,
                                                   uint32_t num_shards,
                                                   uint32_t overlap_factor,
                                                   float overflow_factor = 1.2F) -> uint32_t {
    if (num_nodes == 0 || num_shards == 0) {
      return 0;
    }
    auto target = static_cast<double>(num_nodes) *
                  static_cast<double>(std::max(1U, overlap_factor)) /
                  static_cast<double>(num_shards);
    return static_cast<uint32_t>(std::ceil(target * std::max(1.0F, overflow_factor)));
  }

  template <typename DistanceSpaceType>
    requires Space<DistanceSpaceType> &&
             requires(DistanceSpaceType &space, typename DistanceSpaceType::IDTypeAlias id) {
               space.get_data_by_id(id);
             }
  auto partition(DistanceSpaceType &space,
                 uint32_t max_degree,
                 const std::filesystem::path &output_prefix) const
      -> PartitionedShardLayout<typename DistanceSpaceType::DataTypeAlias,
                                typename DistanceSpaceType::IDTypeAlias> {
    using SpaceDataType = typename DistanceSpaceType::DataTypeAlias;
    using SpaceIDType = typename DistanceSpaceType::IDTypeAlias;
    using SpaceLayout = PartitionedShardLayout<SpaceDataType, SpaceIDType>;

    auto num_nodes = static_cast<uint32_t>(space.get_data_num());
    auto dim = static_cast<uint32_t>(space.get_dim());
    if (num_nodes == 0) {
      throw std::invalid_argument("KMeansPartitioner requires at least one vector");
    }
    if (config_.sample_rate_ <= 0.0F || config_.sample_rate_ > 1.0F) {
      throw std::invalid_argument("sample_rate must be in (0, 1]");
    }
    if (config_.overlap_factor_ == 0) {
      throw std::invalid_argument("overlap_factor must be >= 1");
    }

    SpaceLayout layout;
    layout.num_nodes_ = num_nodes;
    layout.dim_ = dim;
    layout.max_assignments_ = config_.overlap_factor_;
    layout.shard_capacity_ = compute_shard_capacity(dim, max_degree, config_.max_memory_mb_);
    // Each node appears in ~overlap_factor shards, so effective node count per shard
    // is num_nodes * overlap_factor / num_shards. Account for this when sizing.
    auto effective_nodes = static_cast<uint64_t>(num_nodes) * config_.overlap_factor_;
    layout.num_shards_ =
        compute_num_shards(static_cast<uint32_t>(std::min<uint64_t>(effective_nodes, UINT32_MAX)),
                           layout.shard_capacity_);
    layout.shard_size_cap_ =
        std::min(compute_shard_size_cap(num_nodes,
                                        layout.num_shards_,
                                        config_.overlap_factor_,
                                        config_.shard_overflow_factor_),
                 layout.shard_capacity_);

    auto sample = build_sample(space, layout.num_shards_);
    KMeans<SpaceDataType> kmeans(
        typename KMeans<SpaceDataType>::Config{layout.num_shards_,
                                               config_.kmeans_max_iter_,
                                               config_.kmeans_num_trials_});
    auto kmeans_result = kmeans.fit(sample.data(), sample.size() / dim, dim);
    layout.centroids_ = std::move(kmeans_result.centroids_);

    assign_top_l(space, layout);
    rebuild_shard_members(layout);
    apply_shard_size_cap(layout);
    rebuild_shard_members(layout);

    persist(layout, output_prefix);
    write_shuffle_file(space, layout, output_prefix);

    return layout;
  }

  [[nodiscard]] static auto load_node_to_shards(const std::filesystem::path &path) ->
      typename Layout::PersistedNodeToShards {
    struct Header {
      uint32_t num_nodes_;
      uint32_t num_shards_;
      uint32_t max_assignments_;
      uint32_t reserved_;
    };

    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("Failed to open node_to_shards file: " + path.string());
    }

    Header header{};
    input.read(reinterpret_cast<char *>(&header), sizeof(header));
    typename Layout::PersistedNodeToShards loaded;
    loaded.num_nodes_ = header.num_nodes_;
    loaded.num_shards_ = header.num_shards_;
    loaded.max_assignments_ = header.max_assignments_;
    loaded.assignments_.resize(static_cast<size_t>(header.num_nodes_) * header.max_assignments_);
    input.read(reinterpret_cast<char *>(loaded.assignments_.data()),
               static_cast<std::streamsize>(loaded.assignments_.size() * sizeof(uint32_t)));
    return loaded;
  }

  [[nodiscard]] static auto load_shard_members(const std::filesystem::path &path) ->
      typename Layout::PersistedShardMembers {
    struct Header {
      uint32_t num_shards_;
      uint32_t reserved_;
    };

    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("Failed to open shard_members file: " + path.string());
    }

    Header header{};
    input.read(reinterpret_cast<char *>(&header), sizeof(header));

    typename Layout::PersistedShardMembers loaded;
    loaded.num_shards_ = header.num_shards_;
    loaded.members_.resize(header.num_shards_);
    for (uint32_t shard = 0; shard < header.num_shards_; ++shard) {
      uint32_t count = 0;
      input.read(reinterpret_cast<char *>(&count), sizeof(count));
      loaded.members_[shard].resize(count);
      input.read(reinterpret_cast<char *>(loaded.members_[shard].data()),
                 static_cast<std::streamsize>(count * sizeof(uint32_t)));
    }
    return loaded;
  }

 private:
  Config config_;

  template <typename DistanceSpaceType>
  auto build_sample(DistanceSpaceType &space, uint32_t num_shards) const -> std::vector<DataType> {
    auto num_nodes = static_cast<uint32_t>(space.get_data_num());
    auto dim = static_cast<uint32_t>(space.get_dim());
    auto requested =
        static_cast<uint32_t>(std::ceil(static_cast<double>(num_nodes) * config_.sample_rate_));
    auto sample_size = std::clamp<uint32_t>(requested, std::max(1U, num_shards), num_nodes);

    std::vector<uint32_t> ids(num_nodes);
    std::iota(ids.begin(), ids.end(), 0U);
    std::mt19937 rng(config_.sample_seed_);
    std::shuffle(ids.begin(), ids.end(), rng);

    std::vector<DataType> sample(static_cast<size_t>(sample_size) * dim);
    for (uint32_t i = 0; i < sample_size; ++i) {
      const auto *vec = space.get_data_by_id(ids[i]);
      std::copy_n(vec, dim, sample.data() + static_cast<size_t>(i) * dim);
    }
    return sample;
  }

  template <typename DistanceSpaceType, typename SpaceLayout>
  void assign_top_l(DistanceSpaceType &space, SpaceLayout &layout) const {
    auto num_nodes = layout.num_nodes_;
    auto dim = layout.dim_;
    auto num_shards = layout.num_shards_;
    auto max_assignments = layout.max_assignments_;

    layout.primary_shards_.assign(num_nodes, Layout::kInvalidShard);
    layout.node_to_shards_.assign(static_cast<size_t>(num_nodes) * max_assignments,
                                  Layout::kInvalidShard);
    layout.assignment_distances_.assign(static_cast<size_t>(num_nodes) * max_assignments,
                                        std::numeric_limits<float>::max());

    std::vector<std::pair<float, uint32_t>> distances(num_shards);
    for (uint32_t node_id = 0; node_id < num_nodes; ++node_id) {
      const auto *vec = space.get_data_by_id(node_id);
      for (uint32_t shard = 0; shard < num_shards; ++shard) {
        distances[shard] = {KMeans<DataType>::compute_l2_sqr(vec,
                                                             layout.centroids_.data() +
                                                                 static_cast<size_t>(shard) * dim,
                                                             dim),
                            shard};
      }

      auto take = std::min<uint32_t>(max_assignments, num_shards);
      std::partial_sort(distances.begin(),
                        distances.begin() + static_cast<std::ptrdiff_t>(take),
                        distances.end(),
                        [](const auto &lhs, const auto &rhs) {
                          return lhs.first < rhs.first;
                        });

      layout.primary_shards_[node_id] = distances[0].second;
      for (uint32_t slot = 0; slot < take; ++slot) {
        layout.node_to_shards_[static_cast<size_t>(node_id) * max_assignments + slot] =
            distances[slot].second;
        layout.assignment_distances_[static_cast<size_t>(node_id) * max_assignments + slot] =
            distances[slot].first;
      }
    }
  }

  template <typename SpaceLayout>
  static void rebuild_shard_members(SpaceLayout &layout) {
    layout.shard_members_.assign(layout.num_shards_, {});
    for (uint32_t node_id = 0; node_id < layout.num_nodes_; ++node_id) {
      for (uint32_t slot = 0; slot < layout.max_assignments_; ++slot) {
        auto shard = layout.assignment_at(node_id, slot);
        if (shard != Layout::kInvalidShard) {
          layout.shard_members_[shard].push_back(node_id);
        }
      }
    }
  }

  template <typename SpaceLayout>
  static void apply_shard_size_cap(SpaceLayout &layout) {
    for (uint32_t shard = 0; shard < layout.num_shards_; ++shard) {
      auto &members = layout.shard_members_[shard];
      if (members.size() <= layout.shard_size_cap_) {
        continue;
      }

      std::vector<std::pair<float, uint32_t>> removable;
      removable.reserve(members.size());
      for (uint32_t node_id : members) {
        if (layout.primary_shards_[node_id] == shard) {
          continue;
        }
        for (uint32_t slot = 0; slot < layout.max_assignments_; ++slot) {
          if (layout.assignment_at(node_id, slot) == shard) {
            removable.emplace_back(layout.distance_at(node_id, slot), node_id);
            break;
          }
        }
      }

      std::sort(removable.begin(), removable.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.first > rhs.first;
      });

      size_t remove_idx = 0;
      while (members.size() > layout.shard_size_cap_ && remove_idx < removable.size()) {
        auto node_id = removable[remove_idx].second;
        for (uint32_t slot = 0; slot < layout.max_assignments_; ++slot) {
          auto idx = static_cast<size_t>(node_id) * layout.max_assignments_ + slot;
          if (layout.node_to_shards_[idx] == shard) {
            layout.node_to_shards_[idx] = Layout::kInvalidShard;
            layout.assignment_distances_[idx] = std::numeric_limits<float>::max();
            members.pop_back();
            break;
          }
        }
        ++remove_idx;
      }

      if (members.size() > layout.shard_size_cap_) {
        throw std::runtime_error(
            "Shard size cap cannot be satisfied without dropping primary assignments");
      }
    }
  }

  template <typename SpaceLayout>
  static void persist(const SpaceLayout &layout, const std::filesystem::path &output_prefix) {
    auto dir = output_prefix.parent_path();
    if (!dir.empty()) {
      std::filesystem::create_directories(dir);
    }

    auto node_to_shards_path = output_prefix.string() + ".node_to_shards.bin";
    auto shard_members_path = output_prefix.string() + ".shard_members.bin";

    struct NodeToShardsHeader {
      uint32_t num_nodes_;
      uint32_t num_shards_;
      uint32_t max_assignments_;
      uint32_t reserved_;
    } node_header{layout.num_nodes_, layout.num_shards_, layout.max_assignments_, 0};

    std::ofstream node_out(node_to_shards_path, std::ios::binary | std::ios::trunc);
    if (!node_out) {
      throw std::runtime_error("Failed to create node_to_shards file: " + node_to_shards_path);
    }
    node_out.write(reinterpret_cast<const char *>(&node_header), sizeof(node_header));
    node_out.write(reinterpret_cast<const char *>(layout.node_to_shards_.data()),
                   static_cast<std::streamsize>(layout.node_to_shards_.size() * sizeof(uint32_t)));

    struct ShardMembersHeader {
      uint32_t num_shards_;
      uint32_t reserved_;
    } shard_header{layout.num_shards_, 0};

    std::ofstream shard_out(shard_members_path, std::ios::binary | std::ios::trunc);
    if (!shard_out) {
      throw std::runtime_error("Failed to create shard_members file: " + shard_members_path);
    }
    shard_out.write(reinterpret_cast<const char *>(&shard_header), sizeof(shard_header));
    for (const auto &members : layout.shard_members_) {
      auto count = static_cast<uint32_t>(members.size());
      shard_out.write(reinterpret_cast<const char *>(&count), sizeof(count));
      shard_out.write(reinterpret_cast<const char *>(members.data()),
                      static_cast<std::streamsize>(members.size() * sizeof(uint32_t)));
    }

    const_cast<SpaceLayout &>(layout).node_to_shards_path_ = node_to_shards_path;
    const_cast<SpaceLayout &>(layout).shard_members_path_ = shard_members_path;
  }

  template <typename DistanceSpaceType, typename SpaceLayout>
  static void write_shuffle_file(DistanceSpaceType &space,
                                 SpaceLayout &layout,
                                 const std::filesystem::path &output_prefix) {
    auto shuffle_path = output_prefix.string() + ".shuffle.bin";
    std::ofstream shuffle_out(shuffle_path, std::ios::binary | std::ios::trunc);
    if (!shuffle_out) {
      throw std::runtime_error("Failed to create shuffle file: " + shuffle_path);
    }

    layout.shuffle_offsets_.assign(layout.num_shards_, 0);
    layout.shuffle_counts_.assign(layout.num_shards_, 0);

    uint64_t offset = 0;
    auto row_bytes = static_cast<uint64_t>(layout.dim_) * sizeof(DataType);
    for (uint32_t shard = 0; shard < layout.num_shards_; ++shard) {
      layout.shuffle_offsets_[shard] = offset;
      layout.shuffle_counts_[shard] = layout.shard_members_[shard].size();
      for (uint32_t node_id : layout.shard_members_[shard]) {
        const auto *vec = space.get_data_by_id(node_id);
        shuffle_out.write(reinterpret_cast<const char *>(vec),
                          static_cast<std::streamsize>(row_bytes));
        offset += row_bytes;
      }
    }

    layout.shuffle_path_ = shuffle_path;
  }
};

}  // namespace alaya
