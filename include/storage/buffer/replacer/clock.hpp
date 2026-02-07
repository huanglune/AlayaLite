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

#include <atomic>
#include <cstddef>
#include <optional>
#include <vector>

#include "concept.hpp"

namespace alaya {

/**
 * @brief CLOCK page replacement algorithm.
 *
 * CLOCK is an approximation of LRU with O(1) access overhead.
 * Instead of maintaining strict access order, it uses a reference bit
 * that is set on access and cleared during eviction scanning.
 *
 * Key advantage: unpin() only sets an atomic flag, no lock needed for reads.
 *
 * Implementation:
 * - Circular buffer of frames with reference bits
 * - Clock hand points to next eviction candidate
 * - On evict: scan for frame with ref_bit=false, clearing bits as we go
 *
 * Satisfies the Replacer concept.
 */
class ClockReplacer {
 public:
  /**
   * @brief Construct a CLOCK replacer with given capacity.
   * @param capacity Maximum number of frames that can be tracked
   */
  explicit ClockReplacer(size_t capacity) : capacity_(capacity), hand_(0), size_(0) {
    frames_.resize(capacity);
    for (size_t i = 0; i < capacity; ++i) {
      frames_[i].ref_bit_.store(false, std::memory_order_relaxed);
      frames_[i].in_replacer_.store(false, std::memory_order_relaxed);
    }
  }

  ClockReplacer() : capacity_(0), hand_(0), size_(0) {}

  ~ClockReplacer() = default;

  // Non-copyable
  ClockReplacer(const ClockReplacer &) = delete;
  auto operator=(const ClockReplacer &) -> ClockReplacer & = delete;

  // Movable
  ClockReplacer(ClockReplacer &&other) noexcept
      : capacity_(other.capacity_),
        hand_(other.hand_.load(std::memory_order_relaxed)),
        size_(other.size_.load(std::memory_order_relaxed)),
        frames_(std::move(other.frames_)) {
    other.capacity_ = 0;
    other.hand_.store(0, std::memory_order_relaxed);
    other.size_.store(0, std::memory_order_relaxed);
  }

  auto operator=(ClockReplacer &&other) noexcept -> ClockReplacer & {
    if (this != &other) {
      capacity_ = other.capacity_;
      hand_.store(other.hand_.load(std::memory_order_relaxed), std::memory_order_relaxed);
      size_.store(other.size_.load(std::memory_order_relaxed), std::memory_order_relaxed);
      frames_ = std::move(other.frames_);
      other.capacity_ = 0;
      other.hand_.store(0, std::memory_order_relaxed);
      other.size_.store(0, std::memory_order_relaxed);
    }
    return *this;
  }

  /**
   * @brief Pin a frame (mark as not evictable).
   *
   * Removes the frame from the clock if present.
   */
  void pin(size_t frame_id) {
    if (frame_id >= capacity_) {
      return;
    }
    if (frames_[frame_id].in_replacer_.exchange(false, std::memory_order_acq_rel)) {
      size_.fetch_sub(1, std::memory_order_relaxed);
    }
  }

  /**
   * @brief Unpin a frame (mark as evictable).
   *
   * Sets the reference bit to indicate recent access.
   * This is O(1) and lock-free for the common case.
   */
  void unpin(size_t frame_id) {
    if (frame_id >= capacity_) {
      return;
    }
    // Set reference bit (recently accessed)
    frames_[frame_id].ref_bit_.store(true, std::memory_order_relaxed);

    // Add to replacer if not already present
    if (!frames_[frame_id].in_replacer_.exchange(true, std::memory_order_acq_rel)) {
      size_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  /**
   * @brief Select and remove a victim frame using CLOCK algorithm.
   *
   * Scans from current hand position:
   * - If ref_bit=0 and in_replacer=true: evict this frame
   * - If ref_bit=1: clear it (second chance) and continue
   */
  auto evict() -> std::optional<size_t> {
    if (size_.load(std::memory_order_relaxed) == 0) {
      return std::nullopt;
    }

    // Scan up to 2 * capacity to handle all second chances
    for (size_t scan = 0; scan < 2 * capacity_; ++scan) {
      size_t idx = hand_.load(std::memory_order_relaxed);

      // Advance hand (circular)
      size_t next = (idx + 1) % capacity_;
      hand_.store(next, std::memory_order_relaxed);

      if (!frames_[idx].in_replacer_.load(std::memory_order_acquire)) {
        continue;
      }

      // Check reference bit
      if (frames_[idx].ref_bit_.load(std::memory_order_relaxed)) {
        // Second chance: clear ref bit and continue
        frames_[idx].ref_bit_.store(false, std::memory_order_relaxed);
      } else {
        // Found victim: ref_bit=0, in_replacer=true
        if (frames_[idx].in_replacer_.exchange(false, std::memory_order_acq_rel)) {
          size_.fetch_sub(1, std::memory_order_relaxed);
          return idx;
        }
      }
    }

    return std::nullopt;
  }

  /**
   * @brief Remove a specific frame from the replacer.
   */
  void remove(size_t frame_id) {
    if (frame_id >= capacity_) {
      return;
    }
    if (frames_[frame_id].in_replacer_.exchange(false, std::memory_order_acq_rel)) {
      size_.fetch_sub(1, std::memory_order_relaxed);
    }
    frames_[frame_id].ref_bit_.store(false, std::memory_order_relaxed);
  }

  /**
   * @brief Get the number of evictable frames.
   */
  [[nodiscard]] auto size() const -> size_t { return size_.load(std::memory_order_relaxed); }

  /**
   * @brief Reset the replacer to initial state.
   */
  void reset() {
    for (size_t i = 0; i < capacity_; ++i) {
      frames_[i].ref_bit_.store(false, std::memory_order_relaxed);
      frames_[i].in_replacer_.store(false, std::memory_order_relaxed);
    }
    hand_.store(0, std::memory_order_relaxed);
    size_.store(0, std::memory_order_relaxed);
  }

  /**
   * @brief Set the capacity (for deferred initialization).
   */
  void set_capacity(size_t capacity) {
    capacity_ = capacity;
    frames_.resize(capacity);
    for (size_t i = 0; i < capacity; ++i) {
      frames_[i].ref_bit_.store(false, std::memory_order_relaxed);
      frames_[i].in_replacer_.store(false, std::memory_order_relaxed);
    }
  }

 private:
  struct FrameState {
    std::atomic<bool> ref_bit_{false};      ///< Reference bit (recently accessed)
    std::atomic<bool> in_replacer_{false};  ///< Whether frame is in the replacer

    FrameState() = default;

    // Copy constructor for vector resize
    FrameState(const FrameState &other)
        : ref_bit_(other.ref_bit_.load(std::memory_order_relaxed)),
          in_replacer_(other.in_replacer_.load(std::memory_order_relaxed)) {}

    // Move constructor
    FrameState(FrameState &&other) noexcept
        : ref_bit_(other.ref_bit_.load(std::memory_order_relaxed)),
          in_replacer_(other.in_replacer_.load(std::memory_order_relaxed)) {}

    // Copy assignment
    auto operator=(const FrameState &other) -> FrameState & {
      if (this != &other) {
        ref_bit_.store(other.ref_bit_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        in_replacer_.store(other.in_replacer_.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
      }
      return *this;
    }

    // Move assignment
    auto operator=(FrameState &&other) noexcept -> FrameState & {
      if (this != &other) {
        ref_bit_.store(other.ref_bit_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        in_replacer_.store(other.in_replacer_.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
      }
      return *this;
    }
  };

  size_t capacity_;
  std::atomic<size_t> hand_;  ///< Clock hand position
  std::atomic<size_t> size_;  ///< Number of evictable frames
  std::vector<FrameState> frames_;
};

// Static assertion to verify ClockReplacer satisfies the Replacer concept
static_assert(ReplacerStrategy<ClockReplacer>, "ClockReplacer must satisfy the Replacer concept");

}  // namespace alaya
