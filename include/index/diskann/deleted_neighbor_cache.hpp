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
#include <list>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "utils/locks.hpp"

namespace alaya {

/// Maximum number of two-hop candidates per deleted neighbor during graph repair.
static constexpr size_t kMaxTwoHopCandidates = 5;

/**
 * @brief LRU cache for deleted nodes' neighbor lists.
 *
 * Stores neighbor lists of deleted nodes keyed by internal node ID.
 * Used during connect tasks for two-hop graph repair (Yi, SIGMOD 2025).
 * When the cache is full, the least recently used entry is evicted.
 *
 * @tparam IDType Node ID type (default: uint32_t)
 */
template <typename IDType = uint32_t>
class DeletedNeighborCache {
 public:
  /**
   * @brief Construct with explicit capacity.
   *
   * @param capacity Maximum number of entries in the cache
   */
  explicit DeletedNeighborCache(size_t capacity) : capacity_(std::max(capacity, size_t{1})) {}

  /**
   * @brief Construct with default capacity based on index capacity (4%).
   *
   * @param index_capacity Total index capacity
   * @param ratio Cache size as fraction of index capacity (default 0.04)
   */
  static auto with_index_capacity(uint32_t index_capacity, double ratio = 0.04)
      -> DeletedNeighborCache {
    auto cap = static_cast<size_t>(static_cast<double>(index_capacity) * ratio);
    return DeletedNeighborCache(std::max(cap, size_t{64}));
  }

  /**
   * @brief Store a deleted node's neighbor list in the cache.
   *
   * If the node already exists, its entry is updated and moved to MRU.
   * If the cache is full, the LRU entry is evicted.
   *
   * @param node_id Internal node ID of the deleted node
   * @param neighbors The node's neighbor list at time of deletion
   */
  void put(IDType node_id, std::vector<IDType> neighbors) {
    SpinLockGuard guard(*lock_);
    auto it = map_.find(node_id);
    if (it != map_.end()) {
      // Update existing entry, move to front (MRU)
      it->second->neighbors_ = std::move(neighbors);
      lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
      return;
    }

    // Evict LRU if at capacity
    if (map_.size() >= capacity_) {
      auto &back = lru_list_.back();
      map_.erase(back.node_id_);
      lru_list_.pop_back();
    }

    // Insert new entry at front (MRU)
    lru_list_.emplace_front(Entry{node_id, std::move(neighbors)});
    map_[node_id] = lru_list_.begin();
  }

  /**
   * @brief Look up a deleted node's cached neighbor list.
   *
   * On cache hit, the entry is promoted to MRU position.
   *
   * @param node_id Internal node ID to look up
   * @return Copy of neighbor IDs, or nullopt on cache miss
   */
  auto get(IDType node_id) -> std::optional<std::vector<IDType>> {
    SpinLockGuard guard(*lock_);
    auto it = map_.find(node_id);
    if (it == map_.end()) {
      return std::nullopt;
    }

    // Move to front (MRU)
    lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
    return it->second->neighbors_;  // return a copy so caller is safe after lock release
  }

  /**
   * @brief Check if the cache contains a given node ID.
   */
  [[nodiscard]] auto contains(IDType node_id) const -> bool {
    SpinLockGuard guard(*lock_);
    return map_.count(node_id) > 0;
  }

  [[nodiscard]] auto size() const -> size_t {
    SpinLockGuard guard(*lock_);
    return map_.size();
  }
  [[nodiscard]] auto capacity() const -> size_t { return capacity_; }
  [[nodiscard]] auto empty() const -> bool {
    SpinLockGuard guard(*lock_);
    return map_.empty();
  }

 private:
  struct Entry {
    IDType node_id_;
    std::vector<IDType> neighbors_;
  };

  size_t capacity_;
  mutable std::unique_ptr<SpinLock> lock_{std::make_unique<SpinLock>()};
  std::list<Entry> lru_list_;  // Front = MRU, Back = LRU
  std::unordered_map<IDType, typename std::list<Entry>::iterator> map_;
};

}  // namespace alaya
