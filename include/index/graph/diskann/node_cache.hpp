// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file node_cache.hpp
 * @brief BFS node cache for the DiskANN disk index.
 *
 * At build time a BFS from the medoid collects the hottest
 * @c ceil(cache_ratio * num_points) nodes (those near the entry point, which
 * every search traverses). Their full records (@c [coords | n_nbrs | nbr_ids],
 * byte-identical to the on-disk layout) are serialized to @c cache_ids.bin and
 * @c cache_nodes.bin. At load time the records are read into memory and indexed
 * for O(1) lookup, so a cache hit during beam search costs zero disk I/O.
 *
 * Build-time records stay immutable after generate()/load(). In-place updates
 * publish per-node override records so concurrent search never observes a
 * partially-written cached node.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "index/graph/diskann/disk_layout.hpp"

namespace alaya::diskann {

/**
 * @brief In-memory cache of hot node records selected by BFS from the medoid.
 *
 * Internally the id->record index stores byte offsets into @c node_data_ (not
 * raw pointers), so the structure stays valid across vector growth and moves;
 * lookup() resolves an offset to a borrowed const pointer.
 */
class NodeCache {
 public:
  NodeCache() = default;

  struct Lookup {
    const char *record = nullptr;
    std::shared_ptr<const std::vector<char>> owned;

    [[nodiscard]] const char *get() const { return record; }
    [[nodiscard]] explicit operator bool() const { return record != nullptr; }
  };

  /**
   * @brief Select and pack the BFS cache from an in-memory graph + vectors.
   *
   * @param graph        Adjacency lists (graph[i] = neighbor ids of node i).
   * @param vectors      Row-major num_points*dim float32 coordinates.
   * @param medoid       BFS root (entry point); always included first.
   * @param num_points   Number of graph nodes.
   * @param dim          Vector dimension.
   * @param max_degree   Graph degree bound (defines node_len).
   * @param cache_ratio  Fraction of nodes to cache; count = ceil(ratio * N),
   *                     capped at N. ratio <= 0 produces an empty cache.
   */
  void generate(const std::vector<std::vector<uint32_t>> &graph,
                const float *vectors,
                uint32_t medoid,
                uint64_t num_points,
                uint64_t dim,
                uint32_t max_degree,
                double cache_ratio) {
    if (graph.size() != num_points) {
      throw std::invalid_argument("NodeCache::generate: graph size != num_points");
    }
    if (num_points == 0 || dim == 0) {
      throw std::invalid_argument("NodeCache::generate: num_points/dim must be > 0");
    }
    if (medoid >= num_points) {
      throw std::invalid_argument("NodeCache::generate: medoid out of range");
    }

    const DiskLayoutGeometry geom = DiskLayoutGeometry::compute(dim, max_degree);
    dim_ = dim;
    max_degree_ = max_degree;
    node_len_ = geom.node_len;

    uint64_t target = 0;
    if (cache_ratio > 0.0) {
      target = static_cast<uint64_t>(std::ceil(cache_ratio * static_cast<double>(num_points)));
      if (target > num_points) {
        target = num_points;
      }
    }

    ids_.clear();
    map_.clear();
    node_data_.clear();
    clear_overrides_unsafe();
    if (target == 0) {
      return;
    }

    // BFS from medoid, collecting nodes in dequeue (BFS) order.
    std::vector<uint8_t> visited(num_points, 0);
    std::queue<uint32_t> bfs;
    bfs.push(medoid);
    visited[medoid] = 1;
    ids_.reserve(target);
    while (!bfs.empty() && ids_.size() < target) {
      const uint32_t u = bfs.front();
      bfs.pop();
      ids_.push_back(u);
      for (const uint32_t v : graph[u]) {
        if (v < num_points && visited[v] == 0) {
          visited[v] = 1;
          bfs.push(v);
        }
      }
    }

    // Pack records (zero-initialized so unused neighbor slots are zero-filled).
    node_data_.assign(ids_.size() * node_len_, 0);
    map_.reserve(ids_.size() * 2);
    for (size_t i = 0; i < ids_.size(); ++i) {
      const uint32_t id = ids_[i];
      char *rec = node_data_.data() + i * node_len_;
      const auto &nbrs = graph[id];
      const uint32_t n_nbrs =
          static_cast<uint32_t>(nbrs.size() > max_degree ? max_degree : nbrs.size());
      pack_node_record(rec, vectors + static_cast<uint64_t>(id) * dim, nbrs.data(), n_nbrs, dim);
      map_[id] = i * node_len_;
    }
  }

  // --- Persistence ---------------------------------------------------------

  /// Write cache_ids.bin and cache_nodes.bin.
  void save(const std::string &ids_path, const std::string &nodes_path) const {
    std::vector<uint32_t> ids = ids_;
    std::vector<uint32_t> extra_ids;
    for (const auto &shard : override_shards_) {
      std::shared_lock<std::shared_mutex> lock(shard.mutex);
      for (const auto &entry : shard.map) {
        if (map_.find(entry.first) == map_.end()) {
          extra_ids.push_back(entry.first);
        }
      }
    }
    std::sort(extra_ids.begin(), extra_ids.end());
    ids.insert(ids.end(), extra_ids.begin(), extra_ids.end());

    const uint64_t count = ids.size();
    {
      std::ofstream out(ids_path, std::ios::binary | std::ios::trunc);
      if (!out) {
        throw std::runtime_error("NodeCache::save: cannot open " + ids_path);
      }
      out.write(reinterpret_cast<const char *>(&count), sizeof(count));
      if (count > 0) {
        out.write(reinterpret_cast<const char *>(ids.data()),
                  static_cast<std::streamsize>(count * sizeof(uint32_t)));
      }
      if (!out) {
        throw std::runtime_error("NodeCache::save: write failed for " + ids_path);
      }
    }
    {
      std::ofstream out(nodes_path, std::ios::binary | std::ios::trunc);
      if (!out) {
        throw std::runtime_error("NodeCache::save: cannot open " + nodes_path);
      }
      out.write(reinterpret_cast<const char *>(&count), sizeof(count));
      out.write(reinterpret_cast<const char *>(&node_len_), sizeof(node_len_));
      for (const uint32_t id : ids) {
        const auto override = find_override(id);  // hold ownership across the write
        const char *record = override != nullptr ? override->data() : find_base_record(id);
        if (record == nullptr) {
          throw std::runtime_error("NodeCache::save: missing cached record for id " +
                                   std::to_string(id));
        }
        out.write(record, static_cast<std::streamsize>(node_len_));
      }
      if (!out) {
        throw std::runtime_error("NodeCache::save: write failed for " + nodes_path);
      }
    }
  }

  /// Read cache_ids.bin and cache_nodes.bin and rebuild the lookup index.
  void load(const std::string &ids_path, const std::string &nodes_path) {
    uint64_t id_count = 0;
    {
      std::ifstream in(ids_path, std::ios::binary);
      if (!in) {
        throw std::runtime_error("NodeCache::load: cannot open " + ids_path);
      }
      in.read(reinterpret_cast<char *>(&id_count), sizeof(id_count));
      if (!in) {
        throw std::runtime_error("NodeCache::load: short read (ids count) " + ids_path);
      }
      ids_.assign(id_count, 0);
      if (id_count > 0) {
        in.read(reinterpret_cast<char *>(ids_.data()),
                static_cast<std::streamsize>(id_count * sizeof(uint32_t)));
        if (!in) {
          throw std::runtime_error("NodeCache::load: short read (ids) " + ids_path);
        }
      }
    }

    uint64_t node_count = 0;
    {
      std::ifstream in(nodes_path, std::ios::binary);
      if (!in) {
        throw std::runtime_error("NodeCache::load: cannot open " + nodes_path);
      }
      in.read(reinterpret_cast<char *>(&node_count), sizeof(node_count));
      in.read(reinterpret_cast<char *>(&node_len_), sizeof(node_len_));
      if (!in) {
        throw std::runtime_error("NodeCache::load: short read (nodes header) " + nodes_path);
      }
      if (node_count != id_count) {
        throw std::runtime_error("NodeCache::load: ids/nodes count mismatch (" +
                                 std::to_string(id_count) + " vs " + std::to_string(node_count) +
                                 ")");
      }
      node_data_.assign(node_count * node_len_, 0);
      if (!node_data_.empty()) {
        in.read(node_data_.data(), static_cast<std::streamsize>(node_data_.size()));
        if (!in) {
          throw std::runtime_error("NodeCache::load: short read (node data) " + nodes_path);
        }
      }
    }

    map_.clear();
    map_.reserve(ids_.size() * 2);
    for (size_t i = 0; i < ids_.size(); ++i) {
      map_[ids_[i]] = i * node_len_;
    }
    clear_overrides_unsafe();
  }

  void configure_geometry(uint64_t dim, uint32_t max_degree) {
    // Load-time only (before any concurrent runtime access).
    const DiskLayoutGeometry geom = DiskLayoutGeometry::compute(dim, max_degree);
    if (node_len_ != 0 && node_len_ != geom.node_len) {
      throw std::invalid_argument("NodeCache::configure_geometry: node_len mismatch");
    }
    dim_ = dim;
    max_degree_ = max_degree;
    node_len_ = geom.node_len;
  }

  // --- Runtime lookup ------------------------------------------------------

  /**
   * @brief O(1) cache lookup.
   * @return Pointer to the node's @c node_len -byte record, or nullptr on miss.
   *         The returned pointer is read-only and owned by this cache.
   */
  [[nodiscard]] Lookup lookup_record(uint32_t node_id) const {
    auto override = find_override(node_id);
    if (override != nullptr) {
      Lookup hit;
      hit.owned = std::move(override);
      hit.record = hit.owned->data();
      return hit;
    }
    // Base records are immutable after generate()/load(): lock-free read.
    return Lookup{find_base_record(node_id), {}};
  }

  [[nodiscard]] const char *lookup(uint32_t node_id) const {
    const Lookup hit = lookup_record(node_id);
    return hit.get();
  }

  void upsert_node(uint32_t node_id,
                   const float *coords,
                   uint32_t n_nbrs,
                   const uint32_t *nbr_ids) {
    validate_record_input(coords, n_nbrs, nbr_ids, "NodeCache::upsert_node");
    auto record = make_record(coords, n_nbrs, nbr_ids);
    OverrideShard &shard = override_shard(node_id);
    std::unique_lock<std::shared_mutex> lock(shard.mutex);
    shard.map[node_id] = std::move(record);
  }

  void update_neighbors(uint32_t node_id, uint32_t n_nbrs, const uint32_t *nbr_ids) {
    validate_neighbors_input(n_nbrs, nbr_ids, "NodeCache::update_neighbors");
    std::vector<float> coords;
    {
      const auto override = find_override(node_id);
      const char *record = override != nullptr ? override->data() : find_base_record(node_id);
      if (record == nullptr) {
        return;
      }
      coords.assign(reinterpret_cast<const float *>(record),
                    reinterpret_cast<const float *>(record) + dim_);
    }
    upsert_node(node_id, coords.data(), n_nbrs, nbr_ids);
  }

  /// Drop override records for nodes that are NOT pinned in the BFS hot cache.
  /// Only safe when the on-disk pages are current (i.e. right after a flush):
  /// a dropped node falls back to the disk read path, while a hot-cached
  /// node's base record predates its updates, so its override must stay.
  /// Bounds override memory the way Yi's write-back bounds its dirty set.
  void drop_disk_backed_overrides() {
    for (auto &shard : override_shards_) {
      std::unique_lock<std::shared_mutex> lock(shard.mutex);
      for (auto it = shard.map.begin(); it != shard.map.end();) {
        if (map_.find(it->first) == map_.end()) {
          it = shard.map.erase(it);
        } else {
          ++it;
        }
      }
    }
  }

  // --- Accessors -----------------------------------------------------------

  [[nodiscard]] uint64_t size() const {
    uint64_t count = ids_.size();
    for (const auto &shard : override_shards_) {
      std::shared_lock<std::shared_mutex> lock(shard.mutex);
      for (const auto &entry : shard.map) {
        if (map_.find(entry.first) == map_.end()) {
          ++count;
        }
      }
    }
    return count;
  }
  [[nodiscard]] uint64_t node_len() const { return node_len_; }
  [[nodiscard]] const std::vector<uint32_t> &ids() const { return ids_; }

 private:
  void validate_neighbors_input(uint32_t n_nbrs, const uint32_t *nbr_ids, const char *name) const {
    if (dim_ == 0 || node_len_ == 0) {
      throw std::runtime_error(std::string(name) + ": cache geometry is not configured");
    }
    if (n_nbrs > max_degree_) {
      throw std::invalid_argument(std::string(name) + ": n_nbrs exceeds max_degree");
    }
    if (n_nbrs > 0 && nbr_ids == nullptr) {
      throw std::invalid_argument(std::string(name) + ": null neighbor ids");
    }
  }

  void validate_record_input(const float *coords,
                             uint32_t n_nbrs,
                             const uint32_t *nbr_ids,
                             const char *name) const {
    if (coords == nullptr) {
      throw std::invalid_argument(std::string(name) + ": null coords");
    }
    validate_neighbors_input(n_nbrs, nbr_ids, name);
  }

  std::shared_ptr<const std::vector<char>> make_record(const float *coords,
                                                       uint32_t n_nbrs,
                                                       const uint32_t *nbr_ids) const {
    auto record = std::make_shared<std::vector<char>>(node_len_, 0);
    pack_node_record(record->data(), coords, nbr_ids, n_nbrs, dim_);
    return record;
  }

  /// Base records are immutable after generate()/load(): lock-free.
  [[nodiscard]] const char *find_base_record(uint32_t node_id) const {
    const auto it = map_.find(node_id);
    if (it == map_.end()) {
      return nullptr;
    }
    return node_data_.data() + it->second;
  }

  /// Overrides are sharded by node id: reconnect-heavy update workloads take
  /// a unique lock per published node, and one global mutex was measured to
  /// invert thread scaling (more workers -> lower throughput). The base cache
  /// (ids_/map_/node_data_) is immutable after generate()/load() and is read
  /// lock-free.
  static constexpr uint32_t kOverrideShards = 64;

  struct OverrideShard {
    mutable std::shared_mutex mutex;
    std::unordered_map<uint32_t, std::shared_ptr<const std::vector<char>>> map;
  };

  OverrideShard &override_shard(uint32_t node_id) const {
    return override_shards_[node_id % kOverrideShards];
  }

  [[nodiscard]] std::shared_ptr<const std::vector<char>> find_override(uint32_t node_id) const {
    const OverrideShard &shard = override_shard(node_id);
    std::shared_lock<std::shared_mutex> lock(shard.mutex);
    const auto it = shard.map.find(node_id);
    if (it == shard.map.end()) {
      return nullptr;
    }
    return it->second;
  }

  void clear_overrides_unsafe() {
    for (auto &shard : override_shards_) {
      shard.map.clear();
    }
  }

  uint64_t dim_ = 0;
  uint32_t max_degree_ = 0;
  uint64_t node_len_ = 0;
  std::vector<uint32_t> ids_;                   // cached node ids, BFS order
  std::vector<char> node_data_;                 // size() * node_len_ bytes
  std::unordered_map<uint32_t, uint64_t> map_;  // node_id -> byte offset in node_data_
  mutable std::array<OverrideShard, kOverrideShards> override_shards_;
};

}  // namespace alaya::diskann
