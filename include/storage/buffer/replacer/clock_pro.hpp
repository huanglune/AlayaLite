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

#include "concept.hpp"
#include "utils/hash_map.hpp"

namespace alaya {

/**
 * @brief CLOCK-Pro page replacement algorithm.
 *
 * CLOCK-Pro is an enhancement of CLOCK that distinguishes between
 * "hot" (frequently accessed) and "cold" (infrequently accessed) pages.
 * It provides better performance than LRU for workloads with varying
 * access patterns.
 *
 * Key concepts:
 * - COLD pages: Newly inserted or infrequently accessed pages (evicted first)
 * - HOT pages: Frequently accessed pages (protected from eviction)
 * - TEST pages: Metadata of recently evicted cold pages (for promotion detection)
 *
 * When a cold page is accessed again, it's promoted to hot.
 * When a test page is accessed, we know we're evicting useful pages too quickly.
 *
 * Satisfies the Replacer concept.
 */
class ClockProReplacer {
 public:
  /**
   * @brief Construct a CLOCK-Pro replacer with given capacity.
   * @param capacity Maximum number of frames that can be tracked
   */
  explicit ClockProReplacer(size_t capacity)
      : capacity_(capacity),
        max_hot_size_(capacity > 1 ? capacity / 2 : 1),
        max_test_size_(capacity),
        cold_hand_(cold_list_.end()),
        hot_hand_(hot_list_.end()) {}

  ClockProReplacer() : capacity_(0), max_hot_size_(0), max_test_size_(0) {}

  ~ClockProReplacer() = default;

  // Non-copyable
  ClockProReplacer(const ClockProReplacer &) = delete;
  auto operator=(const ClockProReplacer &) -> ClockProReplacer & = delete;

  // Movable
  ClockProReplacer(ClockProReplacer &&other) noexcept
      : capacity_(other.capacity_),
        max_hot_size_(other.max_hot_size_),
        max_test_size_(other.max_test_size_),
        cold_list_(std::move(other.cold_list_)),
        hot_list_(std::move(other.hot_list_)),
        test_set_(std::move(other.test_set_)),
        frame_map_(std::move(other.frame_map_)),
        cold_hand_(cold_list_.end()),
        hot_hand_(hot_list_.end()) {
    other.capacity_ = 0;
  }

  auto operator=(ClockProReplacer &&other) noexcept -> ClockProReplacer & {
    if (this != &other) {
      capacity_ = other.capacity_;
      max_hot_size_ = other.max_hot_size_;
      max_test_size_ = other.max_test_size_;
      cold_list_ = std::move(other.cold_list_);
      hot_list_ = std::move(other.hot_list_);
      test_set_ = std::move(other.test_set_);
      frame_map_ = std::move(other.frame_map_);
      cold_hand_ = cold_list_.end();
      hot_hand_ = hot_list_.end();
      other.capacity_ = 0;
    }
    return *this;
  }

  /**
   * @brief Pin a frame (mark as not evictable).
   *
   * Removes the frame from cold or hot list.
   */
  void pin(size_t frame_id) {
    auto it = frame_map_.find(frame_id);
    if (it == frame_map_.end()) {
      return;
    }

    const auto &entry = it->second;
    if (entry.is_hot_) {
      erase_hot_entry(entry.iter_);
    } else {
      erase_cold_entry(entry.iter_);
    }
    frame_map_.erase(it);
  }

  /**
   * @brief Unpin a frame (mark as evictable).
   *
   * - If frame is in test set: promote to hot (it was useful)
   * - If frame is already tracked: set reference bit
   * - Otherwise: add as cold page
   */
  void unpin(size_t frame_id) {
    if (frame_id >= capacity_) {
      return;
    }

    // Check if this was a recently evicted page (in test set)
    if (test_set_.contains(frame_id)) {
      // Promote to hot - this page was evicted but accessed again
      test_set_.erase(frame_id);
      add_hot(frame_id);
      return;
    }

    auto it = frame_map_.find(frame_id);
    if (it != frame_map_.end()) {
      // Already in replacer - set reference bit
      it->second.ref_bit_ = true;
    } else {
      // New page - add as cold
      add_cold(frame_id);
    }
  }

  /**
   * @brief Select and remove a victim frame.
   *
   * Priority: cold pages with ref_bit=0
   * If all cold pages have ref_bit=1, demote some hot pages first.
   */
  auto evict() -> std::optional<size_t> {
    // Try to evict from cold list first
    auto victim = evict_cold();
    if (victim.has_value()) {
      return victim;
    }

    // If cold list is empty but hot list has pages,
    // demote some hot pages to cold
    if (!hot_list_.empty()) {
      demote_hot_pages();
      return evict_cold();
    }

    return std::nullopt;
  }

  /**
   * @brief Remove a specific frame from the replacer.
   */
  void remove(size_t frame_id) {
    pin(frame_id);  // pin() already handles removal
    test_set_.erase(frame_id);
  }

  /**
   * @brief Get the number of evictable frames.
   */
  [[nodiscard]] auto size() const -> size_t { return frame_map_.size(); }

  /**
   * @brief Reset the replacer to initial state.
   */
  void reset() {
    cold_list_.clear();
    hot_list_.clear();
    test_set_.clear();
    frame_map_.clear();
    cold_hand_ = cold_list_.end();
    hot_hand_ = hot_list_.end();
  }

  /**
   * @brief Set the capacity (for deferred initialization).
   */
  void set_capacity(size_t capacity) {
    capacity_ = capacity;
    max_hot_size_ = capacity > 1 ? capacity / 2 : 1;
    max_test_size_ = capacity;
  }

  /**
   * @brief Get count of hot pages.
   */
  [[nodiscard]] auto hot_size() const -> size_t { return hot_list_.size(); }

  /**
   * @brief Get count of cold pages.
   */
  [[nodiscard]] auto cold_size() const -> size_t { return cold_list_.size(); }

  /**
   * @brief Get count of test entries.
   */
  [[nodiscard]] auto test_size() const -> size_t { return test_set_.size(); }

 private:
  struct PageEntry {
    size_t frame_id_;
    bool ref_bit_{true};  // Reference bit (recently accessed)
  };

  using PageList = std::list<PageEntry>;
  using PageIter = PageList::iterator;

  struct FrameInfo {
    PageIter iter_;
    bool is_hot_;   // true if in hot_list_, false if in cold_list_
    bool ref_bit_;  // Cached reference bit for quick access
  };

  static auto wrap_hand(PageList &list, PageIter iter) -> PageIter {
    if (iter == list.end() && !list.empty()) {
      return list.begin();
    }
    return iter;
  }

  static auto erase_entry(PageList &list, PageIter hand, PageIter entry) -> PageIter {
    if (hand == entry) {
      return wrap_hand(list, list.erase(entry));
    }

    list.erase(entry);
    return list.empty() ? list.end() : hand;
  }

  void erase_cold_entry(PageIter entry) { cold_hand_ = erase_entry(cold_list_, cold_hand_, entry); }

  void erase_hot_entry(PageIter entry) { hot_hand_ = erase_entry(hot_list_, hot_hand_, entry); }

  /**
   * @brief Add a frame to cold list.
   */
  void add_cold(size_t frame_id) {
    // Ensure we don't exceed capacity
    while (frame_map_.size() >= capacity_) {
      auto victim = evict();
      if (!victim.has_value()) {
        break;
      }
    }

    PageEntry entry{.frame_id_ = frame_id, .ref_bit_ = true};
    cold_list_.push_back(entry);
    auto iter = std::prev(cold_list_.end());

    frame_map_[frame_id] = FrameInfo{.iter_ = iter, .is_hot_ = false, .ref_bit_ = true};

    if (cold_hand_ == cold_list_.end()) {
      cold_hand_ = cold_list_.begin();
    }
  }

  /**
   * @brief Add a frame to hot list.
   */
  void add_hot(size_t frame_id) {
    // If hot list is full, demote oldest hot page
    while (hot_list_.size() >= max_hot_size_) {
      demote_one_hot();
    }

    PageEntry entry{.frame_id_ = frame_id, .ref_bit_ = true};
    hot_list_.push_back(entry);
    auto iter = std::prev(hot_list_.end());

    frame_map_[frame_id] = FrameInfo{.iter_ = iter, .is_hot_ = true, .ref_bit_ = true};

    if (hot_hand_ == hot_list_.end()) {
      hot_hand_ = hot_list_.begin();
    }
  }

  /**
   * @brief Demote one hot page to cold.
   */
  void demote_one_hot() {
    if (hot_list_.empty()) {
      return;
    }

    // Scan hot list for page with ref_bit=0
    size_t scanned = 0;
    while (scanned < hot_list_.size()) {
      if (hot_hand_ == hot_list_.end()) {
        hot_hand_ = hot_list_.begin();
      }

      auto &entry = *hot_hand_;
      auto it = frame_map_.find(entry.frame_id_);

      if (it != frame_map_.end() && !it->second.ref_bit_) {
        // Found victim - demote to cold
        size_t frame_id = entry.frame_id_;
        auto old_iter = hot_hand_;
        erase_hot_entry(old_iter);
        frame_map_.erase(it);

        // Add to cold list
        add_cold(frame_id);
        return;
      }

      // Give second chance
      if (it != frame_map_.end()) {
        it->second.ref_bit_ = false;
        entry.ref_bit_ = false;
      }
      advance_hot_hand();
      ++scanned;
    }

    // All hot pages have ref_bit=1, force demote the current one
    if (hot_hand_ == hot_list_.end()) {
      hot_hand_ = hot_list_.begin();
    }
    if (hot_hand_ != hot_list_.end()) {
      size_t frame_id = hot_hand_->frame_id_;
      auto old_iter = hot_hand_;
      erase_hot_entry(old_iter);
      frame_map_.erase(frame_id);
      add_cold(frame_id);
    }
  }

  /**
   * @brief Demote multiple hot pages when cold list is exhausted.
   */
  void demote_hot_pages() {
    // Demote at least one, up to half of hot pages
    size_t to_demote = std::max(size_t{1}, hot_list_.size() / 2);
    for (size_t i = 0; i < to_demote && !hot_list_.empty(); ++i) {
      demote_one_hot();
    }
  }

  /**
   * @brief Evict from cold list using CLOCK algorithm.
   */
  auto evict_cold() -> std::optional<size_t> {
    if (cold_list_.empty()) {
      return std::nullopt;
    }

    size_t scanned = 0;
    while (scanned < 2 * cold_list_.size()) {
      if (cold_hand_ == cold_list_.end()) {
        cold_hand_ = cold_list_.begin();
      }
      if (cold_hand_ == cold_list_.end()) {
        break;
      }

      auto &entry = *cold_hand_;
      auto it = frame_map_.find(entry.frame_id_);

      if (it == frame_map_.end()) {
        // Stale entry, remove and continue
        auto old_iter = cold_hand_;
        erase_cold_entry(old_iter);
        continue;
      }

      if (!it->second.ref_bit_) {
        // Found victim
        size_t victim = entry.frame_id_;
        auto old_iter = cold_hand_;
        erase_cold_entry(old_iter);
        frame_map_.erase(it);

        // Add to test set (track recently evicted)
        add_to_test(victim);
        return victim;
      }

      // Second chance: clear ref bit and potentially promote to hot
      it->second.ref_bit_ = false;
      entry.ref_bit_ = false;

      // If accessed twice (ref_bit was set after being cleared), promote to hot
      // For now, just give second chance
      advance_cold_hand();
      ++scanned;
    }

    // Force evict if we've scanned too many
    if (!cold_list_.empty()) {
      if (cold_hand_ == cold_list_.end()) {
        cold_hand_ = cold_list_.begin();
      }
      size_t victim = cold_hand_->frame_id_;
      auto old_iter = cold_hand_;
      erase_cold_entry(old_iter);
      frame_map_.erase(victim);
      add_to_test(victim);
      return victim;
    }

    return std::nullopt;
  }

  /**
   * @brief Add frame to test set (track recently evicted).
   */
  void add_to_test(size_t frame_id) {
    // Limit test set size
    while (test_set_.size() >= max_test_size_) {
      // Remove oldest (we use unordered_set so just remove any)
      test_set_.erase(test_set_.begin());
    }
    test_set_.insert(frame_id);
  }

  void advance_cold_hand() {
    if (cold_hand_ != cold_list_.end()) {
      ++cold_hand_;
    }
    if (cold_hand_ == cold_list_.end() && !cold_list_.empty()) {
      cold_hand_ = cold_list_.begin();
    }
  }

  void advance_hot_hand() {
    if (hot_hand_ != hot_list_.end()) {
      ++hot_hand_;
    }
    if (hot_hand_ == hot_list_.end() && !hot_list_.empty()) {
      hot_hand_ = hot_list_.begin();
    }
  }

  size_t capacity_;
  size_t max_hot_size_;   ///< Maximum hot pages (typically capacity/2)
  size_t max_test_size_;  ///< Maximum test entries

  PageList cold_list_;          ///< Cold pages (candidates for eviction)
  PageList hot_list_;           ///< Hot pages (protected)
  fast::set<size_t> test_set_;  ///< Recently evicted page IDs

  fast::map<size_t, FrameInfo> frame_map_;  ///< frame_id -> info

  PageIter cold_hand_;  ///< Clock hand for cold list
  PageIter hot_hand_;   ///< Clock hand for hot list
};

// Static assertion to verify ClockProReplacer satisfies the Replacer concept
static_assert(ReplacerStrategy<ClockProReplacer>,
              "ClockProReplacer must satisfy the Replacer concept");

}  // namespace alaya
