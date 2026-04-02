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
 *
 *   // Write-back example (flush dirty page on eviction)
 *   DirectFileIO writer;
 *   writer.open("data.bin", DirectFileIO::Mode::kWrite);
 *   auto flush_cb = [&](uint32_t node_id, const uint8_t* data) {
 *     // map node_id -> file offset as needed
 *     uint64_t offset = static_cast<uint64_t>(node_id) * 4096;
 *     writer.write(reinterpret_cast<const char*>(data), 4096, offset);
 *   };
 *
 *   BufferPool<uint32_t> wb_pool(1000, 4096, 16, flush_cb);
 *   auto handle = wb_pool.get_or_read(node_id, reader, offset, temp.data());
 *   if (!handle.empty()) {
 *     auto* mutable_ptr = handle.mutable_data();
 *     // ... modify page for insert/delete ...
 *     handle.mark_dirty();
 *   }
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

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <vector>

#include "disk_read_awaitable.hpp"
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
  std::atomic<uint64_t> reuse_hits_{0};
  std::atomic<uint64_t> reuse_misses_{0};
  std::atomic<uint64_t> evictions_{0};
  std::atomic<uint64_t> pins_{0};

  [[nodiscard]] auto hit_rate() const -> double {
    auto hits = hits_.load(std::memory_order_relaxed);
    auto misses = misses_.load(std::memory_order_relaxed);
    auto total = hits + misses;
    if (total == 0) {
      return 0.0;
    }
    return static_cast<double>(hits) / static_cast<double>(total);
  }

  [[nodiscard]] auto reuse_hit_rate() const -> double {
    auto hits = reuse_hits_.load(std::memory_order_relaxed);
    auto misses = reuse_misses_.load(std::memory_order_relaxed);
    auto total = hits + misses;
    if (total == 0) {
      return 0.0;
    }
    return static_cast<double>(hits) / static_cast<double>(total);
  }

  void reset() {
    hits_ = 0;
    misses_ = 0;
    reuse_hits_ = 0;
    reuse_misses_ = 0;
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
    friend class BufferPool;

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

    void mark_dirty() {
      if (shard_ && frame_idx_ < shard_->capacity()) {
        shard_->mark_dirty(frame_idx_);
      }
    }

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

  BufferPool(size_t capacity,
             size_t frame_size,
             size_t num_shards = 16,
             std::function<void(IDType, const uint8_t *)> flush_cb = nullptr)
      : total_capacity_(capacity),
        frame_size_(frame_size),
        num_shards_(num_shards),
        flush_callback_(std::move(flush_cb)) {
    if (total_capacity_ == 0) {
      throw std::invalid_argument("BufferPool capacity must be greater than 0");
    }
    num_shards_ = std::max(size_t{1}, std::min(num_shards_, total_capacity_));

    // 1. Allocate a large contiguous memory block (Huge Page Friendly)
    size_t total_bytes = total_capacity_ * frame_size_;
    memory_blob_.resize(total_bytes);

    // 2. Initialize shards
    shards_.reserve(num_shards_);

    for (size_t i = 0; i < num_shards_; ++i) {
      size_t start = i * total_capacity_ / num_shards_;
      size_t end = (i + 1) * total_capacity_ / num_shards_;
      size_t actual_cap = end - start;

      uint8_t *shard_base = memory_blob_.data() + (start * frame_size_);
      shards_.emplace_back(
          std::make_unique<Shard>(i, actual_cap, frame_size_, shard_base, stats_, flush_callback_));
    }
  }

  // Once created, the address must be fixed.
  ALAYA_NON_COPYABLE(BufferPool);

  /**
   * @brief Get a page. Returns an empty Handle if not cached.
   */
  auto get(IDType node_id) -> PageHandle { return get_shard(node_id).get_page(node_id); }

  /**
   * @brief Flush all dirty pages via a caller-supplied callback.
   *
   * @param fn Callback invoked as fn(node_id, data_ptr) for each dirty frame.
   */
  template <typename Fn>
  void flush_all(Fn &&fn) {
    for (auto &shard : shards_) {
      shard->flush_dirty(fn);
    }
  }

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

  /// Result of begin_async_read: a pinned PageHandle plus optional pending I/O state.
  struct AsyncReadResult {
    PageHandle handle;        ///< Pinned page handle (data valid only when is_ready() == true)
    AsyncReadState *pending;  ///< Non-null if async I/O is in-flight

    [[nodiscard]] auto is_ready() const -> bool {
      return pending == nullptr || pending->finish_read.load(std::memory_order_acquire);
    }

    /// True if the async I/O completed with an error (short read or negative errno).
    /// Only meaningful after is_ready() returns true.
    [[nodiscard]] auto has_error() const -> bool {
      return pending != nullptr && pending->has_error.load(std::memory_order_acquire);
    }
  };

  /**
   * @brief Start an async page read. Returns immediately with a pinned handle.
   *
   * On cache hit, data is immediately available (pending == nullptr).
   * On cache miss, submits async I/O via io_uring and returns pending state.
   *
   * Usage from a top-level coroutine (in Worker's local_tasks_):
   * @code
   *   auto ar = pool.begin_async_read(node_id, io, offset);
   *   while (!ar.is_ready()) {
   *     co_await YieldAwaitable{};  // yield to Worker, which polls io_uring
   *   }
   *   // ar.handle.data() is now valid
   * @endcode
   */
  auto begin_async_read(IDType node_id, DirectFileIO &io, uint64_t offset) -> AsyncReadResult {
    Shard &shard = get_shard(node_id);

    auto result = shard.allocate_for_async_read(node_id);
    if (result.frame_data == nullptr) {
      return {PageHandle{}, nullptr};
    }

    if (result.is_hit) {
      return {PageHandle(&shard, result.frame_idx, result.frame_data, frame_size_), nullptr};
    }

    stats_.misses_.fetch_add(1, std::memory_order_relaxed);

    if (result.needs_io) {
      auto *notifier = new AsyncReadNotifier{result.async_state};
      bool submitted = io.submit_async_read(result.frame_data,
                                            frame_size_,
                                            offset,
                                            async_read_callback,
                                            notifier);
      if (!submitted) {
        // Async submission failed — fall back to synchronous read.
        delete notifier;
        auto bytes = io.read(reinterpret_cast<char *>(result.frame_data), frame_size_, offset);
        if (bytes != static_cast<ssize_t>(frame_size_)) {
          result.async_state->has_error.store(true, std::memory_order_relaxed);
        }
        result.async_state->finish_read.store(true, std::memory_order_release);
      }
    }

    return {PageHandle(&shard, result.frame_idx, result.frame_data, frame_size_),
            result.async_state};
  }

  auto touch_if_cached(IDType node_id) -> bool {
    auto handle = get(node_id);
    return !handle.empty();
  }

  /**
   * @brief Prefetch a page from disk into the buffer pool.
   * * This is a "fire-and-forget" operation used for batch I/O optimization.
   * It ensures the page is in memory (LRU MRU) but does NOT keep it pinned.
   * * Implementation details:
   * 1. Checks if page exists (fast path).
   * 2. If miss, reads into a thread-local buffer (no malloc).
   * 3. Inserts into shard (handles eviction).
   * 4. Immediately releases handle (pin_count -> 0).
   */
  void prefetch(IDType node_id, DirectFileIO &io, uint64_t offset) {
    if (touch_if_cached(node_id)) {
      record_reuse_hit();
      return;
    }

    // 2. Cache Miss: Read from disk
    record_reuse_miss();
    record_miss();
    // Use thread_local buffer to avoid allocation overhead during heavy batch prefetching
    static thread_local AlignedBuffer tl_buffer;
    if (tl_buffer.size() != frame_size_) {
      tl_buffer.resize(frame_size_);
    }

    auto bytes = io.read(reinterpret_cast<char *>(tl_buffer.data()), frame_size_, offset);
    if (bytes != static_cast<ssize_t>(frame_size_)) {
      return;  // Read failed, ignore
    }

    // 3. Insert into Cache
    // insert_page pins the page (count=1).
    // The returned handle destructor immediately unpins it (count=0).
    // The page remains in the pool as MRU (Most Recently Used).
    Shard &shard = get_shard(node_id);
    shard.insert_page(node_id, tl_buffer.data());
  }

  void record_miss() { stats_.misses_.fetch_add(1, std::memory_order_relaxed); }
  void record_reuse_hit() { stats_.reuse_hits_.fetch_add(1, std::memory_order_relaxed); }
  void record_reuse_miss() { stats_.reuse_misses_.fetch_add(1, std::memory_order_relaxed); }

  void clear() {
    for (const auto &s : shards_) {
      if (s->has_active_pins()) {
        throw std::logic_error("Cannot clear BufferPool: active page handles exist");
      }
    }
    for (auto &s : shards_) {
      s->clear();
    }
    stats_.reset();
  }

  void reset_stats() { stats_.reset(); }

  /**
   * @brief Set or replace the write-back callback invoked on dirty page eviction.
   *
   * Use this when the flush target (e.g., DataFile) is not yet available at
   * BufferPool construction time.
   */
  void set_flush_callback(std::function<void(IDType, const uint8_t *)> cb) {
    flush_callback_ = cb;
    for (auto &shard : shards_) {
      shard->set_flush_callback(cb);
    }
  }

  auto stats() const -> const BufferPoolStats & { return stats_; }
  [[nodiscard]] auto num_shards() const -> size_t { return num_shards_; }

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
      bool is_dirty_{false};
      uint32_t pin_count_{0};
      AsyncReadState async_state_;  ///< Async I/O state for coroutine-based reads
    };

    Shard(size_t id,
          size_t cap,
          size_t fsize,
          const uint8_t *base,
          BufferPoolStats &stats,
          std::function<void(IDType, const uint8_t *)> flush_cb)
        : shard_id_(id),
          capacity_(cap),
          frame_size_(fsize),
          stats_ref_(stats),
          flush_callback_(std::move(flush_cb)) {
      frames_.resize(capacity_);
      for (size_t i = 0; i < capacity_; ++i) {
        frames_[i].data_ = const_cast<uint8_t *>(base + (i * frame_size_));
        free_list_.push(i);
      }
      page_table_.reserve(capacity_);
    }

    [[nodiscard]] auto capacity() const -> size_t { return capacity_; }

    // try to get page and pin it (sync path — skips frames still being async-loaded)
    auto get_page(IDType node_id) -> PageHandle {
      std::lock_guard<SpinLock> guard(lock_);

      auto it = page_table_.find(node_id);
      if (it == page_table_.end()) {
        return {};
      }

      size_t frame_idx = it->second;
      Frame &frame = frames_[frame_idx];

      // If an async read is still in-flight, treat as miss. The sync caller
      // (get_or_read / prefetch) will do its own disk read and insert_page()
      // will handle the race with the in-flight async I/O.
      if (!frame.async_state_.finish_read.load(std::memory_order_acquire)) {
        return {};
      }

      if (frame.pin_count_ == 0) {
        replacer_.pin(frame_idx);
      }
      frame.pin_count_++;

      stats_ref_.hits_.fetch_add(1, std::memory_order_relaxed);
      return PageHandle(this, frame_idx, frame.data_, frame_size_);
    }

    auto insert_page(IDType node_id, const uint8_t *source_data) -> PageHandle {
      // Variables for deferred flush (performed outside the lock).
      IDType flush_node_id{};
      uint8_t *flush_data_ptr{nullptr};
      bool need_flush{false};
      bool overwrite_async{false};

      size_t frame_idx{0};

      {
        std::lock_guard<SpinLock> guard(lock_);

        // check again if another thread inserted it during IO
        if (auto it = page_table_.find(node_id); it != page_table_.end()) {
          size_t idx = it->second;
          Frame &frame = frames_[idx];

          // If an async read is still in-flight, reuse the frame and overwrite
          // with our sync-read data. The async callback only sets finish_read
          // (targets the same frame.data_ buffer), so the overwrite is safe.
          if (!frame.async_state_.finish_read.load(std::memory_order_acquire)) {
            if (frame.pin_count_ == 0) {
              replacer_.pin(idx);
            }
            frame.pin_count_++;
            frame_idx = idx;
            overwrite_async = true;
          } else {
            if (frame.pin_count_ == 0) {
              replacer_.pin(idx);
            }
            frame.pin_count_++;
            return PageHandle(this, idx, frame.data_, frame_size_);
          }
        }

        if (!overwrite_async) {
          // Allocate a new frame or evict an existing one.
          if (!free_list_.empty()) {
            frame_idx = free_list_.front();
            free_list_.pop();
          } else {
            auto victim = replacer_.evict();
            if (!victim) {
              // Cache is full and all pages are pinned — system overload edge case.
              return {};
            }
            frame_idx = *victim;

            // Collect dirty-page info to flush after releasing the lock.
            Frame &vic_frame = frames_[frame_idx];
            if (vic_frame.is_valid_) {
              if (vic_frame.is_dirty_ && flush_callback_) {
                flush_node_id = vic_frame.node_id_;
                flush_data_ptr = vic_frame.data_;
                need_flush = true;
              }
              page_table_.erase(vic_frame.node_id_);
              stats_ref_.evictions_.fetch_add(1, std::memory_order_relaxed);
            }
          }

          // Mark the frame as pinned/in-use before we release the lock so no
          // other thread can evict it while we are copying data into it.
          Frame &frame = frames_[frame_idx];
          frame.node_id_ = node_id;
          frame.is_valid_ = true;
          frame.is_dirty_ = false;
          frame.pin_count_ = 1;
          page_table_[node_id] = frame_idx;
          replacer_.pin(frame_idx);
        }
      }

      // Perform the potentially slow disk write outside the lock.
      if (need_flush) {
        flush_callback_(flush_node_id, flush_data_ptr);
      }

      // Fill the frame with the new data (lock not held — safe because
      // pin_count >= 1 prevents eviction).
      Frame &frame = frames_[frame_idx];
      std::memcpy(frame.data_, source_data, frame_size_);

      // Ensure finish_read is true so sync get_page() returns this frame.
      frame.async_state_.has_error.store(false, std::memory_order_relaxed);
      frame.async_state_.finish_read.store(true, std::memory_order_release);

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

    void mark_dirty(size_t frame_idx) {
      std::lock_guard<SpinLock> guard(lock_);
      frames_[frame_idx].is_dirty_ = true;
    }

    /**
     * @brief Flush all dirty frames via a caller-supplied callback.
     */
    template <typename Fn>
    void flush_dirty(Fn &&fn) {
      std::lock_guard<SpinLock> guard(lock_);
      for (auto &frame : frames_) {
        if (frame.is_valid_ && frame.is_dirty_) {
          fn(frame.node_id_, frame.data_);
          frame.is_dirty_ = false;
        }
      }
    }

    [[nodiscard]] auto has_active_pins() const -> bool {
      std::lock_guard<SpinLock> guard(lock_);
      return std::any_of(frames_.begin(), frames_.end(), [](const Frame &frame) {
        return frame.pin_count_ > 0;
      });
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
        frames_[i].is_dirty_ = false;
        frames_[i].pin_count_ = 0;
        free_list_.push(i);
      }
    }

    void set_flush_callback(std::function<void(IDType, const uint8_t *)> cb) {
      std::lock_guard<SpinLock> guard(lock_);
      flush_callback_ = std::move(cb);
    }

    /// Result of allocate_for_async_read.
    struct AsyncAllocResult {
      size_t frame_idx{0};
      uint8_t *frame_data{nullptr};
      AsyncReadState *async_state{nullptr};
      bool is_hit{false};    ///< true = data already valid (cache hit or sync-loaded)
      bool needs_io{false};  ///< true = caller must submit async I/O
    };

    /**
     * @brief Allocate or find a frame for an async read.
     *
     * Under the shard lock:
     * - If page is cached and fully loaded: returns is_hit=true
     * - If page is cached but loading (another coroutine started I/O): returns is_hit=false,
     *   needs_io=false (caller should co_await)
     * - If page is not cached: allocates frame, returns needs_io=true (caller submits I/O)
     *
     * In all non-hit cases, the frame is pinned (pin_count incremented).
     */
    auto allocate_for_async_read(IDType node_id) -> AsyncAllocResult {
      IDType flush_node_id{};
      uint8_t *flush_data_ptr{nullptr};
      bool need_flush{false};
      size_t frame_idx{0};
      bool needs_io = false;

      {
        std::lock_guard<SpinLock> guard(lock_);

        auto it = page_table_.find(node_id);
        if (it != page_table_.end()) {
          // Page exists in page_table
          size_t idx = it->second;
          Frame &frame = frames_[idx];
          if (frame.pin_count_ == 0) {
            replacer_.pin(idx);
          }
          frame.pin_count_++;

          bool data_ready = frame.async_state_.finish_read.load(std::memory_order_acquire);
          if (data_ready) {
            stats_ref_.hits_.fetch_add(1, std::memory_order_relaxed);
          }
          return {idx, frame.data_, &frame.async_state_, data_ready, false};
        }

        // Not in page_table - allocate new frame
        if (!free_list_.empty()) {
          frame_idx = free_list_.front();
          free_list_.pop();
        } else {
          auto victim = replacer_.evict();
          if (!victim) {
            return {};  // pool full, all pinned
          }
          frame_idx = *victim;

          Frame &vic_frame = frames_[frame_idx];
          if (vic_frame.is_valid_) {
            if (vic_frame.is_dirty_ && flush_callback_) {
              flush_node_id = vic_frame.node_id_;
              flush_data_ptr = vic_frame.data_;
              need_flush = true;
            }
            page_table_.erase(vic_frame.node_id_);
            stats_ref_.evictions_.fetch_add(1, std::memory_order_relaxed);
          }
        }

        Frame &frame = frames_[frame_idx];
        frame.node_id_ = node_id;
        frame.is_valid_ = true;
        frame.is_dirty_ = false;
        frame.pin_count_ = 1;
        frame.async_state_.reset_for_async();  // finish_read=false, awaiting_list=nullptr
        page_table_[node_id] = frame_idx;
        replacer_.pin(frame_idx);
        needs_io = true;
      }

      if (need_flush) {
        flush_callback_(flush_node_id, flush_data_ptr);
      }

      Frame &frame = frames_[frame_idx];
      return {frame_idx, frame.data_, &frame.async_state_, false, needs_io};
    }

   private:
    // aliasing to avoid False Sharing
    alignas(64) mutable SpinLock lock_;

    const size_t shard_id_;    // NOLINT
    const size_t capacity_;    // NOLINT
    const size_t frame_size_;  // NOLINT
    BufferPoolStats &stats_ref_;

    std::vector<Frame> frames_;
    std::function<void(IDType, const uint8_t *)> flush_callback_;
    ReplacerType replacer_{capacity_};
    fast::map<IDType, size_t> page_table_;
    std::queue<size_t> free_list_;
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
  std::function<void(IDType, const uint8_t *)> flush_callback_;
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
