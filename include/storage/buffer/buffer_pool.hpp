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

/**
 * @file buffer_pool.hpp
 * @brief Buffer pool for caching disk pages with pluggable replacement policies.
 *
 * @section Usage
 *
 * Basic usage:
 * @code
 *   #include "storage/buffer/buffer_pool.hpp"
 *
 *   // Create a buffer pool with 1000 frames, each 4KB
 *   // Default uses LRU replacement policy
 *   BufferPool<uint32_t> pool(1000, 4096);
 *
 *   // Or use CLOCK for better read performance (lock-free unpin)
 *   BufferPool<uint32_t, ClockReplacer> clock_pool(1000, 4096);
 *
 *   // Or use CLOCK-Pro for adaptive hot/cold page management
 *   BufferPool<uint32_t, ClockProReplacer> clockpro_pool(1000, 4096);
 *
 *   // Method 1: Manual get/put
 *   const uint8_t* data = pool.get(node_id);  // Returns nullptr on miss
 *   if (data == nullptr) {
 *       // Read from disk manually, then cache it
 *       pool.put(node_id, disk_data);
 *   }
 *
 *   // Method 2: Automatic get or read from disk (recommended)
 *   DirectFileIO reader;
 *   reader.open("data.bin", DirectFileIO::Mode::kRead);
 *   AlignedBuffer temp(4096);
 *   const uint8_t* result = pool.get_or_read(node_id, reader, offset, temp.data());
 *
 *   // Check statistics
 *   auto& stats = pool.stats();
 *   printf("Hit rate: %.2f%%\n", stats.hit_rate() * 100);
 *   printf("Hits: %lu, Misses: %lu, Evictions: %lu\n",
 *          stats.hits_.load(), stats.misses_.load(), stats.evictions_.load());
 * @endcode
 *
 * @section ReplacementPolicies Replacement Policies
 *
 * | Policy       | Read Perf | Scan Resist | Memory | Use Case                    |
 * |--------------|-----------|-------------|--------|-----------------------------|
 * | LRUReplacer  | Medium    | Poor        | O(n)   | General purpose             |
 * | ClockReplacer| High      | Medium      | O(n)   | Read-heavy, low contention  |
 * | ClockProReplacer | High  | Good        | O(n)   | Mixed workloads, adaptive   |
 *
 * Integration with DiskANNSearcher:
 * @code
 *   DiskANNSearcher<float> searcher;
 *   searcher.open("index.bin");
 *   searcher.enable_caching(1000);  // Enable buffer pool with 1000 node capacity
 *   // Searches now benefit from node caching
 * @endcode
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <queue>
#include <vector>

#include "replacer/clock.hpp"
#include "replacer/clock_pro.hpp"
#include "replacer/lru.hpp"
#include "storage/io/direct_file_io.hpp"
#include "utils/hash_map.hpp"
#include "utils/locks.hpp"
#include "utils/macros.hpp"
#include "utils/memory.hpp"

namespace alaya {
struct BufferPoolStats {
  std::atomic<uint64_t> hits_{0};
  std::atomic<uint64_t> misses_{0};
  std::atomic<uint64_t> evictions_{0};
  std::atomic<uint64_t> pins_{0};

  void reset() {
    hits_ = 0;
    misses_ = 0;
    evictions_ = 0;
    pins_ = 0;
  }
};

template <typename IDType = uint32_t, ReplacerStrategy ReplacerType = LRUReplacer>
class BufferPool {
 private:
  class Shard;

 public:
  /**
   * @brief PageHandle (RAII Smart Pointer)
   * * User-held handles.
   * - Constructed: represents a pinned page (ref count +1), ensuring it won't be evicted.
   * - Destructed: automatically unpins (ref count -1), allowing eviction.
   * - Move semantics: supports move, copy is disabled.
   */
  class PageHandle {
    friend class Shard;

   public:
    PageHandle() = default;

    PageHandle(PageHandle &&other) noexcept
        : shard_(other.shard_),
          frame_idx_(other.frame_idx_),
          data_ptr_(other.data_ptr_),
          size_(other.size_) {
      other.shard_ = nullptr;
      other.data_ptr_ = nullptr;
    }

    auto operator=(PageHandle &&other) noexcept -> PageHandle & {
      if (this != &other) {
        reset();
        shard_ = other.shard_;
        frame_idx_ = other.frame_idx_;
        data_ptr_ = other.data_ptr_;
        size_ = other.size_;

        other.shard_ = nullptr;
        other.data_ptr_ = nullptr;
      }
      return *this;
    }

    ALAYA_NON_COPYABLE(PageHandle);
    ~PageHandle() { reset(); }

    // Accessors
    [[nodiscard]] auto data() const -> const uint8_t * { return data_ptr_; }
    [[nodiscard]] auto mutable_data() -> uint8_t * { return data_ptr_; }
    [[nodiscard]] auto size() const -> size_t { return size_; }
    [[nodiscard]] auto empty() const -> bool { return data_ptr_ == nullptr; }

    //  View
    [[nodiscard]] auto view() const -> std::span<const uint8_t> { return {data_ptr_, size_}; }

   private:
    // only Shard can construct PageHandles
    PageHandle(Shard *shard, size_t frame_idx, uint8_t *data, size_t size)
        : shard_(shard), frame_idx_(frame_idx), data_ptr_(data), size_(size) {}

    void reset();  // defined after Shard

    Shard *shard_{nullptr};
    size_t frame_idx_{0};
    uint8_t *data_ptr_{nullptr};
    size_t size_{0};
  };

  // -----------------------------------------------------------------------

  BufferPool(size_t capacity, size_t frame_size, size_t num_shards = 16)
      : total_capacity_(capacity), frame_size_(frame_size), num_shards_(num_shards) {
    if (num_shards_ == 0) {
      num_shards_ = 1;
    }
    if (total_capacity_ == 0) {
      return;
    }

    // 1. Allocate a large contiguous memory block (Huge Page Friendly)
    size_t total_bytes = total_capacity_ * frame_size_;
    memory_blob_.resize(total_bytes);

    // 2. Initialize shards
    size_t cap_per_shard = (total_capacity_ + num_shards_ - 1) / num_shards_;
    shards_.reserve(num_shards_);

    for (size_t i = 0; i < num_shards_; ++i) {
      size_t actual_cap = cap_per_shard;
      // Adjust the size of the last shard
      if (i == num_shards_ - 1) {
        actual_cap = total_capacity_ - (cap_per_shard * (num_shards_ - 1));
      }

      uint8_t *shard_base = memory_blob_.data() + (i * cap_per_shard * frame_size_);
      shards_.emplace_back(std::make_unique<Shard>(i, actual_cap, frame_size_, shard_base, stats_));
    }
  }

  // Once created, the address must be fixed.
  ALAYA_NON_COPYABLE(BufferPool);

  /**
   * @brief Get a page. Returns an empty Handle if not cached.
   */
  auto get(IDType node_id) -> PageHandle { return get_shard(node_id).get_page(node_id); }

  /**
   * @brief Insert a pre-read page into the cache (no disk I/O).
   *
   * Used by batch prefetch: data is already read via io_uring batch submission,
   * so we only need to copy it into a cache frame.
   *
   * @param node_id Cache key (e.g., block_id)
   * @param data Source buffer (frame_size_ bytes, already read from disk)
   * @return PageHandle pinning the inserted page
   */
  auto put(IDType node_id, const uint8_t *data) -> PageHandle {
    return get_shard(node_id).insert_page(node_id, data);
  }

  /**
   * @brief Core interface: get or read a page.
   * 1. Try to get from cache.
   * 2. If Miss, read from disk into temp_buffer.
   * 3. Insert temp_buffer into cache and return Handle.
   */
  auto get_or_read(IDType node_id, DirectFileIO &io, uint64_t offset, uint8_t *temp_buffer)
      -> PageHandle {
    Shard &shard = get_shard(node_id);

    // 1. Fast Path: Cache Hit
    {
      auto handle = shard.get_page(node_id);
      if (!handle.empty()) {
        return handle;
      }
    }

    // 2. Cache Miss: Read IO (lock-free IO)
    // Note: temp_buffer must be an externally provided thread-local buffer to avoid allocation
    stats_.misses_.fetch_add(1, std::memory_order_relaxed);
    auto bytes = io.read(reinterpret_cast<char *>(temp_buffer), frame_size_, offset);
    if (bytes != static_cast<ssize_t>(frame_size_)) {
      return {};  // Read failed
    }

    // 3. Insert into Cache
    // Race condition may occur: two threads read, whoever inserts later wins, or shard does
    // deduplication internally
    return shard.insert_page(node_id, temp_buffer);
  }

  void clear() {
    for (auto &s : shards_) {
      s->clear();
    }
    stats_.reset();
  }

  auto stats() const -> const BufferPoolStats & { return stats_; }

 private:
  // -----------------------------------------------------------------------
  // Shard Implementation
  // -----------------------------------------------------------------------
  class Shard {
   public:
    struct Frame {
      IDType node_id_{static_cast<IDType>(-1)};
      uint8_t *data_{nullptr};
      bool is_valid_{false};
      uint32_t pin_count_{0};
    };

    Shard(size_t id, size_t cap, size_t fsize, const uint8_t *base, BufferPoolStats &stats)
        : shard_id_(id), capacity_(cap), frame_size_(fsize), stats_ref_(stats) {
      frames_.resize(capacity_);
      for (size_t i = 0; i < capacity_; ++i) {
        frames_[i].data_ = const_cast<uint8_t *>(base + (i * frame_size_));
        free_list_.push(i);
      }
      page_table_.reserve(capacity_);
    }

    // try to get page and pin it
    auto get_page(IDType node_id) -> PageHandle {
      std::lock_guard<SpinLock> guard(lock_);

      auto it = page_table_.find(node_id);
      if (it == page_table_.end()) {
        return {};
      }

      size_t frame_idx = it->second;
      Frame &frame = frames_[frame_idx];

      if (frame.pin_count_ == 0) {
        replacer_.pin(frame_idx);
      }
      frame.pin_count_++;

      stats_ref_.hits_.fetch_add(1, std::memory_order_relaxed);
      return PageHandle(this, frame_idx, frame.data_, frame_size_);
    }

    auto insert_page(IDType node_id, const uint8_t *source_data) -> PageHandle {
      std::lock_guard<SpinLock> guard(lock_);

      // check again if another thread inserted it during IO
      if (auto it = page_table_.find(node_id); it != page_table_.end()) {
        size_t idx = it->second;
        Frame &frame = frames_[idx];

        if (frame.pin_count_ == 0) {
          replacer_.pin(idx);
        }
        frame.pin_count_++;

        // Optional: overwrite data? Usually not needed, assuming disk data is immutable or version
        // control is handled by upper layers
        return PageHandle(this, idx, frame.data_, frame_size_);
      }

      // mallocate new frame or evict existing one
      size_t frame_idx;
      if (!free_list_.empty()) {
        frame_idx = free_list_.front();
        free_list_.pop();
      } else {
        auto victim = replacer_.evict();
        if (!victim) {
          // Cache is full and all pages are pinned! This is an edge case of system overload.
          return {};
        }
        frame_idx = *victim;

        // Clean up victim
        Frame &vic_frame = frames_[frame_idx];
        if (vic_frame.is_valid_) {
          page_table_.erase(vic_frame.node_id_);
          stats_ref_.evictions_.fetch_add(1, std::memory_order_relaxed);
        }
      }

      // 3. fill frame
      Frame &frame = frames_[frame_idx];
      frame.node_id_ = node_id;
      frame.is_valid_ = true;
      frame.pin_count_ = 1;
      std::memcpy(frame.data_, source_data, frame_size_);

      page_table_[node_id] = frame_idx;
      replacer_.pin(frame_idx);

      return PageHandle(this, frame_idx, frame.data_, frame_size_);
    }

    // only called by PageHandle destructor
    void unpin_page(size_t frame_idx) {
      std::lock_guard<SpinLock> guard(lock_);
      Frame &frame = frames_[frame_idx];

      if (frame.pin_count_ > 0) {
        frame.pin_count_--;
        if (frame.pin_count_ == 0) {
          replacer_.unpin(frame_idx);
        }
      }
    }

    void clear() {
      std::lock_guard<SpinLock> guard(lock_);
      page_table_.clear();
      replacer_.reset();
      while (!free_list_.empty()) {
        free_list_.pop();
      }
      for (size_t i = 0; i < capacity_; ++i) {
        frames_[i].is_valid_ = false;
        frames_[i].pin_count_ = 0;
        free_list_.push(i);
      }
    }

   private:
    const size_t shard_id_;    // NOLINT
    const size_t capacity_;    // NOLINT
    const size_t frame_size_;  // NOLINT

    // aliasing to avoid False Sharing
    alignas(64) SpinLock lock_;

    std::vector<Frame> frames_;
    fast::map<IDType, size_t> page_table_;
    std::queue<size_t> free_list_;
    ReplacerType replacer_{capacity_};

    BufferPoolStats &stats_ref_;
  };

  auto get_shard(IDType node_id) -> Shard & {
    size_t h = std::hash<IDType>{}(node_id);
    return *shards_[h % num_shards_];
  }

  // -----------------------------------------------------------------------
  // Members
  // -----------------------------------------------------------------------
  size_t total_capacity_;
  size_t frame_size_;
  size_t num_shards_;

  AlignedBuffer memory_blob_;  // unified memory management
  std::vector<std::unique_ptr<Shard>> shards_;
  mutable BufferPoolStats stats_;
};

// -------------------------------------------------------------------------
// PageHandle implementation
// -------------------------------------------------------------------------
template <typename IDType, ReplacerStrategy R>
void BufferPool<IDType, R>::PageHandle::reset() {
  if (shard_ && data_ptr_) {
    shard_->unpin_page(frame_idx_);
  }
  shard_ = nullptr;
  data_ptr_ = nullptr;
  size_ = 0;
}

}  // namespace alaya
