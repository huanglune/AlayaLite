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
#include <cstdint>
#include <cstring>
#include <queue>
#include <unordered_map>
#include <vector>

#include "replacer/lru.hpp"
#include "storage/io/direct_file_io.hpp"
#include "utils/locks.hpp"
#include "utils/memory.hpp"

namespace alaya {

/**
 * @brief Statistics for buffer pool monitoring.
 */
struct BufferPoolStats {
  std::atomic<uint64_t> hits_{0};       ///< Cache hits
  std::atomic<uint64_t> misses_{0};     ///< Cache misses
  std::atomic<uint64_t> evictions_{0};  ///< Pages evicted

  BufferPoolStats() = default;

  // Copy constructor - manually copy atomic values
  BufferPoolStats(const BufferPoolStats &other)
      : hits_(other.hits_.load(std::memory_order_relaxed)),
        misses_(other.misses_.load(std::memory_order_relaxed)),
        evictions_(other.evictions_.load(std::memory_order_relaxed)) {}

  // Copy assignment - manually copy atomic values
  auto operator=(const BufferPoolStats &other) -> BufferPoolStats & {
    if (this != &other) {
      hits_.store(other.hits_.load(std::memory_order_relaxed), std::memory_order_relaxed);
      misses_.store(other.misses_.load(std::memory_order_relaxed), std::memory_order_relaxed);
      evictions_.store(other.evictions_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    return *this;
  }

  /**
   * @brief Calculate cache hit rate.
   * @return Hit rate as a value between 0.0 and 1.0
   */
  [[nodiscard]] auto hit_rate() const -> double {
    uint64_t total =
        hits_.load(std::memory_order_relaxed) + misses_.load(std::memory_order_relaxed);
    return total > 0 ? static_cast<double>(hits_.load(std::memory_order_relaxed)) /
                           static_cast<double>(total)
                     : 0.0;
  }

  /**
   * @brief Reset all statistics counters.
   */
  void reset() {
    hits_.store(0, std::memory_order_relaxed);
    misses_.store(0, std::memory_order_relaxed);
    evictions_.store(0, std::memory_order_relaxed);
  }

  /**
   * @brief Get total number of accesses.
   */
  [[nodiscard]] auto total_accesses() const -> uint64_t {
    return hits_.load(std::memory_order_relaxed) + misses_.load(std::memory_order_relaxed);
  }
};

/**
 * @brief A buffer pool for caching disk nodes with pluggable replacement policy.
 *
 * This buffer pool caches aligned disk pages identified by node ID.
 * It uses a pluggable Replacer (via C++20 concepts) for eviction policy.
 *
 * Key Features:
 * - Aligned buffers for Direct I/O compatibility
 * - Pluggable replacement policy (LRU, CLOCK, etc.) via concepts
 * - Thread-safe with shared/exclusive locking
 * - Configurable capacity
 * - Hit/miss statistics for tuning
 *
 * @tparam IDType The node ID type (default: uint32_t)
 * @tparam ReplacerType The replacement policy type (must satisfy Replacer concept, default:
 * LRUReplacer)
 */
template <typename IDType = uint32_t, Replacer ReplacerType = LRUReplacer>
class BufferPool {
 public:
  /**
   * @brief Internal frame structure holding a cached page.
   */
  struct Frame {
    IDType node_id_{static_cast<IDType>(-1)};  ///< Node ID (or -1 if empty)
    bool is_valid_{false};                     ///< Whether frame contains valid data
    uint8_t *data_{nullptr};                   ///< Pointer to aligned buffer data
  };

 private:
  // Memory pool: pre-allocated aligned buffer for all frames
  AlignedBuffer buffer_pool_;

  // Frame metadata array
  std::vector<Frame> frames_;

  // Hash map: node_id -> frame_index
  std::unordered_map<IDType, size_t> page_table_;

  // Free list: frames that have not been used yet
  std::queue<size_t> free_list_;

  // Configuration
  size_t capacity_{0};      ///< Maximum number of frames
  size_t frame_size_{0};    ///< Size of each frame in bytes
  size_t current_size_{0};  ///< Current number of valid frames

  // Replacement policy (compile-time polymorphism via concepts)
  ReplacerType replacer_;

  // Thread safety using existing SharedLock
  mutable SharedLock lock_;

  // Statistics (atomic for lock-free access)
  BufferPoolStats stats_;

 public:
  /**
   * @brief Construct a buffer pool.
   *
   * @param capacity Maximum number of nodes to cache
   * @param frame_size Size of each frame in bytes (should be aligned)
   */
  BufferPool(size_t capacity, size_t frame_size)
      : capacity_(capacity), frame_size_(frame_size), current_size_(0), replacer_(capacity) {
    if (capacity_ == 0 || frame_size_ == 0) {
      return;
    }

    // Pre-allocate aligned buffer for all frames
    buffer_pool_.resize(capacity_ * frame_size_);

    // Initialize frames array
    frames_.resize(capacity_);
    for (size_t i = 0; i < capacity_; ++i) {
      frames_[i].data_ = buffer_pool_.data() + i * frame_size_;
      frames_[i].is_valid_ = false;
      frames_[i].node_id_ = static_cast<IDType>(-1);
      // Add to free list
      free_list_.push(i);
    }

    page_table_.reserve(capacity_);
  }

  /// Default constructor creates an empty, unusable pool
  BufferPool() = default;

  /// Destructor
  ~BufferPool() = default;

  // Non-copyable
  BufferPool(const BufferPool &) = delete;
  auto operator=(const BufferPool &) -> BufferPool & = delete;

  // Movable
  BufferPool(BufferPool &&other) noexcept
      : buffer_pool_(std::move(other.buffer_pool_)),
        frames_(std::move(other.frames_)),
        page_table_(std::move(other.page_table_)),
        free_list_(std::move(other.free_list_)),
        capacity_(other.capacity_),
        frame_size_(other.frame_size_),
        current_size_(other.current_size_),
        replacer_(std::move(other.replacer_)) {
    // Fix data pointers after move
    if (capacity_ > 0) {
      for (size_t i = 0; i < capacity_; ++i) {
        frames_[i].data_ = buffer_pool_.data() + i * frame_size_;
      }
    }
    other.capacity_ = 0;
    other.current_size_ = 0;
  }

  auto operator=(BufferPool &&other) noexcept -> BufferPool & {
    if (this != &other) {
      buffer_pool_ = std::move(other.buffer_pool_);
      frames_ = std::move(other.frames_);
      page_table_ = std::move(other.page_table_);
      free_list_ = std::move(other.free_list_);
      replacer_ = std::move(other.replacer_);
      capacity_ = other.capacity_;
      frame_size_ = other.frame_size_;
      current_size_ = other.current_size_;

      // Fix data pointers after move
      if (capacity_ > 0) {
        for (size_t i = 0; i < capacity_; ++i) {
          frames_[i].data_ = buffer_pool_.data() + i * frame_size_;
        }
      }
      other.capacity_ = 0;
      other.current_size_ = 0;
    }
    return *this;
  }

  /**
   * @brief Try to get a cached node.
   *
   * If the node is in the cache, returns pointer to the data and updates
   * the replacement policy (e.g., moves to front for LRU).
   *
   * @param node_id The node ID to look up
   * @return Pointer to cached data, or nullptr if not cached
   */
  [[nodiscard]] auto get(IDType node_id) -> const uint8_t * {
    lock_.lock_shared();

    auto it = page_table_.find(node_id);
    if (it == page_table_.end()) {
      lock_.unlock_shared();
      stats_.misses_.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }

    size_t frame_idx = it->second;
    Frame *frame = &frames_[frame_idx];
    if (!frame->is_valid_) {
      lock_.unlock_shared();
      stats_.misses_.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }

    // Found in cache - need to update replacement policy
    // Upgrade to exclusive lock
    lock_.unlock_shared();
    lock_.lock();

    // Re-verify after lock upgrade (another thread might have evicted)
    it = page_table_.find(node_id);
    if (it == page_table_.end() || !frames_[it->second].is_valid_) {
      lock_.unlock();
      stats_.misses_.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }

    frame_idx = it->second;
    frame = &frames_[frame_idx];

    // Update replacement policy: unpin to mark as accessed (moves to front for LRU)
    replacer_.unpin(frame_idx);

    const uint8_t *data = frame->data_;
    lock_.unlock();
    stats_.hits_.fetch_add(1, std::memory_order_relaxed);
    return data;
  }

  /**
   * @brief Put a node into the cache.
   *
   * If the cache is full, evicts a frame using the replacement policy.
   * The caller must provide a pre-read buffer; this method copies
   * the data into the cache.
   *
   * @param node_id The node ID
   * @param data Pointer to the node data to cache (must be frame_size bytes)
   * @return Pointer to the cached copy
   */
  auto put(IDType node_id, const uint8_t *data) -> const uint8_t * {
    if (capacity_ == 0 || data == nullptr) {
      return nullptr;
    }

    lock_.lock();

    // Check if already cached
    auto it = page_table_.find(node_id);
    if (it != page_table_.end() && frames_[it->second].is_valid_) {
      size_t frame_idx = it->second;
      // Update replacement policy
      replacer_.unpin(frame_idx);
      const uint8_t *cached_data = frames_[frame_idx].data_;
      lock_.unlock();
      return cached_data;
    }

    // Get a frame: prefer free list, then evict
    size_t frame_idx;
    if (!free_list_.empty()) {
      frame_idx = free_list_.front();
      free_list_.pop();
    } else {
      // Evict using replacement policy
      auto victim_opt = replacer_.evict();
      if (!victim_opt.has_value()) {
        // No evictable frame available
        lock_.unlock();
        return nullptr;
      }
      frame_idx = victim_opt.value();

      // Remove victim from page table
      Frame *victim = &frames_[frame_idx];
      if (victim->is_valid_) {
        page_table_.erase(victim->node_id_);
        stats_.evictions_.fetch_add(1, std::memory_order_relaxed);
        current_size_--;
      }
    }

    // Update frame with new data
    Frame *frame = &frames_[frame_idx];
    frame->node_id_ = node_id;
    frame->is_valid_ = true;
    std::memcpy(frame->data_, data, frame_size_);

    // Update page table
    page_table_[node_id] = frame_idx;
    current_size_++;

    // Mark as evictable in replacement policy
    replacer_.unpin(frame_idx);

    const uint8_t *cached_data = frame->data_;
    lock_.unlock();
    return cached_data;
  }

  /**
   * @brief Get a node from cache, or read from disk if not cached.
   *
   * This is the primary method for integration with DiskANNSearcher.
   * It combines get() and put() in one operation, minimizing lock overhead.
   *
   * @param node_id The node ID
   * @param reader The file reader for disk I/O
   * @param node_offset The byte offset of the node in the file
   * @param temp_buffer A temporary buffer for disk reads (must be aligned and >= frame_size)
   * @return Pointer to the node data (either cached or freshly read)
   */
  auto get_or_read(IDType node_id, DirectFileIO &reader, uint64_t node_offset, uint8_t *temp_buffer)
      -> const uint8_t * {
    if (capacity_ == 0) {
      // No caching - read directly into temp buffer
      auto bytes = reader.read(temp_buffer, frame_size_, node_offset);
      if (bytes != static_cast<ssize_t>(frame_size_)) {
        return nullptr;
      }
      return temp_buffer;
    }

    // Fast path: check cache with shared lock
    lock_.lock_shared();
    auto it = page_table_.find(node_id);
    if (it != page_table_.end() && frames_[it->second].is_valid_) {
      size_t frame_idx = it->second;
      lock_.unlock_shared();

      // Update replacement policy with exclusive lock
      lock_.lock();
      // Re-verify
      it = page_table_.find(node_id);
      if (it != page_table_.end() && frames_[it->second].is_valid_) {
        frame_idx = it->second;
        replacer_.unpin(frame_idx);
        const uint8_t *data = frames_[frame_idx].data_;
        lock_.unlock();
        stats_.hits_.fetch_add(1, std::memory_order_relaxed);
        return data;
      }
      lock_.unlock();
      // Fall through to cache miss path
    } else {
      lock_.unlock_shared();
    }

    // Cache miss - read from disk (outside lock to avoid blocking)
    stats_.misses_.fetch_add(1, std::memory_order_relaxed);
    auto bytes = reader.read(temp_buffer, frame_size_, node_offset);
    if (bytes != static_cast<ssize_t>(frame_size_)) {
      return nullptr;
    }

    // Insert into cache
    return put(node_id, temp_buffer);
  }

  /**
   * @brief Clear all cached data.
   */
  void clear() {
    lock_.lock();

    page_table_.clear();
    current_size_ = 0;

    // Reset replacer
    replacer_.reset();

    // Clear and rebuild free list
    std::queue<size_t> empty_queue;
    free_list_.swap(empty_queue);

    // Invalidate all frames and add back to free list
    for (size_t i = 0; i < capacity_; ++i) {
      frames_[i].is_valid_ = false;
      frames_[i].node_id_ = static_cast<IDType>(-1);
      free_list_.push(i);
    }

    lock_.unlock();
  }

  /**
   * @brief Get current number of cached frames.
   */
  [[nodiscard]] auto size() const -> size_t {
    lock_.lock_shared();
    size_t s = current_size_;
    lock_.unlock_shared();
    return s;
  }

  /**
   * @brief Get maximum capacity.
   */
  [[nodiscard]] auto capacity() const -> size_t { return capacity_; }

  /**
   * @brief Get frame size in bytes.
   */
  [[nodiscard]] auto frame_size() const -> size_t { return frame_size_; }

  /**
   * @brief Get cache statistics.
   */
  [[nodiscard]] auto stats() const -> const BufferPoolStats & { return stats_; }

  /**
   * @brief Reset statistics counters.
   */
  void reset_stats() { stats_.reset(); }

  /**
   * @brief Check if a node is cached.
   */
  [[nodiscard]] auto contains(IDType node_id) const -> bool {
    lock_.lock_shared();
    auto it = page_table_.find(node_id);
    bool found = (it != page_table_.end() && frames_[it->second].is_valid_);
    lock_.unlock_shared();
    return found;
  }

  /**
   * @brief Check if the buffer pool is enabled (has capacity > 0).
   */
  [[nodiscard]] auto is_enabled() const -> bool { return capacity_ > 0; }

  /**
   * @brief Get the replacer for testing/introspection.
   */
  [[nodiscard]] auto get_replacer() const -> const ReplacerType & { return replacer_; }
};

}  // namespace alaya
