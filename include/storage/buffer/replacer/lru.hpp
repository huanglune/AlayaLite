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

#include <cstddef>
#include <list>
#include <optional>
#include <unordered_map>

#include "concept.hpp"
#include "utils/macros.hpp"

namespace alaya {

/**
 * @brief LRU (Least Recently Used) page replacement algorithm.
 *
 * Maintains frames in order of access time. When eviction is needed,
 * the least recently used frame is selected as the victim.
 *
 * Implementation uses a doubly-linked list for O(1) operations:
 * - pin(): O(1) - remove from list
 * - unpin(): O(1) - add to front of list
 * - evict(): O(1) - remove from back of list
 *
 * Satisfies the Replacer concept.
 */
class LRUReplacer {
 public:
  /**
   * @brief Construct an LRU replacer with given capacity.
   * @param capacity Maximum number of frames that can be tracked
   */
  explicit LRUReplacer(size_t capacity) : capacity_(capacity) {}

  LRUReplacer() : capacity_(0) {}

  ~LRUReplacer() = default;
  ALAYA_NON_COPYABLE_BUT_MOVABLE(LRUReplacer);

  /**
   * @brief Record that a frame has been accessed (pinned).
   *
   * Removes the frame from the evictable list if present.
   * A pinned frame cannot be evicted.
   */
  void pin(size_t frame_id) {
    auto it = frame_map_.find(frame_id);
    if (it != frame_map_.end()) {
      // Remove from LRU list (no longer evictable)
      lru_list_.erase(it->second);
      frame_map_.erase(it);
    }
  }

  /**
   * @brief Mark a frame as evictable (unpinned).
   *
   * Adds the frame to the front of the LRU list (most recently used).
   * If the frame is already in the list, moves it to front.
   */
  void unpin(size_t frame_id) {
    if (frame_id >= capacity_) {
      return;
    }

    auto it = frame_map_.find(frame_id);
    if (it != frame_map_.end()) {
      // Already in list - move to front (most recently used)
      lru_list_.erase(it->second);
      lru_list_.push_front(frame_id);
      it->second = lru_list_.begin();
    } else {
      // Not in list - add to front
      lru_list_.push_front(frame_id);
      frame_map_[frame_id] = lru_list_.begin();
    }
  }

  /**
   * @brief Select and remove the least recently used frame.
   *
   * Returns the frame at the back of the list (least recently used).
   */
  auto evict() -> std::optional<size_t> {
    if (lru_list_.empty()) {
      return std::nullopt;
    }

    // Evict from back (least recently used)
    size_t victim = lru_list_.back();
    lru_list_.pop_back();
    frame_map_.erase(victim);
    return victim;
  }

  /**
   * @brief Remove a frame from the replacer.
   */
  void remove(size_t frame_id) {
    auto it = frame_map_.find(frame_id);
    if (it != frame_map_.end()) {
      lru_list_.erase(it->second);
      frame_map_.erase(it);
    }
  }

  /**
   * @brief Get the number of evictable frames.
   */
  [[nodiscard]] auto size() const -> size_t { return lru_list_.size(); }

  /**
   * @brief Reset the replacer to initial state.
   */
  void reset() {
    lru_list_.clear();
    frame_map_.clear();
  }

  /**
   * @brief Set the capacity (for deferred initialization).
   */
  void set_capacity(size_t capacity) { capacity_ = capacity; }

 private:
  size_t capacity_;

  // Doubly-linked list: front = most recently used, back = least recently used
  std::list<size_t> lru_list_;

  // Map from frame_id to iterator in lru_list_ for O(1) access
  std::unordered_map<size_t, std::list<size_t>::iterator> frame_map_;
};

// Static assertion to verify LRUReplacer satisfies the Replacer concept
static_assert(ReplacerStrategy<LRUReplacer>, "LRUReplacer must satisfy the Replacer concept");

}  // namespace alaya
