/**
 * @file laser_types.hpp
 * @brief Core type definitions for the Laser index module.
 *
 * Defines zero-allocation data structures for the search hot path:
 * - OngoingSlot/OngoingTable: generation-tagged open-addressing hash (replaces unordered_map)
 * - TaggedVisitedSet: generation-tagged visited set (replaces HashBasedBooleanSet)
 * - FixedRingBuffer: fixed-size FIFO queue (replaces std::deque)
 * - FixedStack: fixed-size LIFO stack (replaces std::deque)
 * - LaserSearchParams: search configuration parameters
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "index/laser/laser_common.hpp"
#include "utils/memory.hpp"

namespace symqg {

// ============================================================================
// OngoingSlot: 16-byte cache-line-friendly slot for in-flight I/O tracking
// ============================================================================

struct OngoingSlot {
  char *buffer;  // 8B: pointer to sector scratch buffer  // NOLINT(readability-identifier-naming)
  PID node_id;   // 4B: node being read  // NOLINT(readability-identifier-naming)
  uint32_t
      gen;  // 4B: generation tag for O(1) invalidation  // NOLINT(readability-identifier-naming)
};
static_assert(sizeof(OngoingSlot) == 16, "OngoingSlot must be exactly 16 bytes");

// ============================================================================
// OngoingTable: fixed-size open-addressing hash table with generation tags
// Replaces std::unordered_map<PID, char*> on the hot path.
// ============================================================================

class OngoingTable {
 public:
  static constexpr size_t kDefaultCapacity = 64;
  // Tombstone: gen matches current but buffer is nullptr
  // This keeps the probe chain intact after erase.

  OngoingTable() = default;

  explicit OngoingTable(size_t capacity)
      : capacity_(next_pow2(capacity)), mask_(capacity_ - 1), slots_(capacity_) {
    std::memset(slots_.data(), 0, capacity_ * sizeof(OngoingSlot));
  }

  auto reset() -> void {
    ++gen_;
    if (UNLIKELY(gen_ == 0)) {
      std::memset(slots_.data(), 0, capacity_ * sizeof(OngoingSlot));
      gen_ = 1;
    }
  }

  auto insert(PID node_id, char *buffer) -> void {
    size_t idx = node_id & mask_;
    for (size_t i = 0; i < capacity_; ++i) {
      size_t probe = (idx + i) & mask_;
      auto &slot = slots_[probe];
      if (slot.gen != gen_ || slot.buffer == nullptr) {
        slot = {buffer, node_id, gen_};
        return;
      }
    }
  }

  [[nodiscard]] auto find(PID node_id) const -> char * {
    size_t idx = node_id & mask_;
    for (size_t i = 0; i < capacity_; ++i) {
      size_t probe = (idx + i) & mask_;
      auto &slot = slots_[probe];
      if (slot.gen != gen_) {
        return nullptr;  // empty slot — not found
      }
      // Tombstone (buffer==nullptr): keep probing
      if (slot.buffer != nullptr && slot.node_id == node_id) {
        return slot.buffer;
      }
    }
    return nullptr;
  }

  auto erase(PID node_id) -> void {
    size_t idx = node_id & mask_;
    for (size_t i = 0; i < capacity_; ++i) {
      size_t probe = (idx + i) & mask_;
      auto &slot = slots_[probe];
      if (slot.gen != gen_) {
        return;
      }
      if (slot.buffer != nullptr && slot.node_id == node_id) {
        slot.buffer = nullptr;  // tombstone: keeps probe chain intact
        return;
      }
    }
  }

 private:
  /// Round up to the next power of two (open-addressing requires power-of-two capacity).
  static constexpr auto next_pow2(size_t v) -> size_t {
    if (v == 0) {
      return 1;
    }
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
  }

  size_t capacity_ = kDefaultCapacity;
  size_t mask_ = kDefaultCapacity - 1;
  uint32_t gen_ = 1;
  std::vector<OngoingSlot> slots_;
};

// ============================================================================
// TaggedVisitedSet: generation-tagged visited set with O(1) reset
// Replaces HashBasedBooleanSet on the hot path.
// ============================================================================

class TaggedVisitedSet {
 public:
  struct Entry {
    PID id;        // NOLINT(readability-identifier-naming)
    uint32_t gen;  // NOLINT(readability-identifier-naming)
  };
  static_assert(sizeof(Entry) == 8, "Entry must be 8 bytes");

  TaggedVisitedSet() = default;

  explicit TaggedVisitedSet(size_t capacity) {
    // Round up to power of 2
    size_t actual = 1;
    while (actual < capacity) {
      actual <<= 1;
    }
    capacity_ = actual;
    mask_ = actual - 1;
    table_.resize(actual);
    std::memset(table_.data(), 0, actual * sizeof(Entry));
  }

  auto reset() -> void {
    ++gen_;
    if (UNLIKELY(gen_ == 0)) {
      std::memset(table_.data(), 0, capacity_ * sizeof(Entry));
      gen_ = 1;
    }
  }

  [[nodiscard]] auto get(PID data_id) const -> bool {
    size_t idx = data_id & mask_;
    // Linear probe
    for (size_t i = 0; i < capacity_; ++i) {
      size_t probe = (idx + i) & mask_;
      auto &entry = table_[probe];
      if (entry.gen != gen_) {
        return false;
      }
      if (entry.id == data_id) {
        return true;
      }
    }
    return false;
  }

  auto set(PID data_id) -> void {
    size_t idx = data_id & mask_;
    for (size_t i = 0; i < capacity_; ++i) {
      size_t probe = (idx + i) & mask_;
      auto &entry = table_[probe];
      if (entry.gen != gen_) {
        entry = {data_id, gen_};
        return;
      }
      if (entry.id == data_id) {
        return;  // already present
      }
    }
  }

 private:
  size_t capacity_ = 0;
  size_t mask_ = 0;
  uint32_t gen_ = 1;
  std::vector<Entry> table_;
};

// ============================================================================
// FixedRingBuffer: fixed-size FIFO for prepared_nodes (replaces std::deque)
// ============================================================================

template <typename T>
class FixedRingBuffer {
 public:
  FixedRingBuffer() = default;

  explicit FixedRingBuffer(size_t capacity) : data_(capacity), capacity_(capacity) {}

  auto push_back(const T &val) -> void {
    data_[tail_ % capacity_] = val;
    ++tail_;
  }

  auto pop_front() -> T {
    T val = data_[head_ % capacity_];
    ++head_;
    return val;
  }

  [[nodiscard]] auto front() const -> const T & { return data_[head_ % capacity_]; }

  [[nodiscard]] auto empty() const -> bool { return head_ == tail_; }

  [[nodiscard]] auto size() const -> size_t { return tail_ - head_; }

  auto reset() -> void { head_ = tail_ = 0; }

 private:
  std::vector<T> data_;
  size_t capacity_ = 0;
  size_t head_ = 0;
  size_t tail_ = 0;
};

// ============================================================================
// FixedStack: fixed-size LIFO for free_slots (replaces std::deque)
// ============================================================================

template <typename T>
class FixedStack {
 public:
  FixedStack() = default;

  explicit FixedStack(size_t capacity) : data_(capacity), capacity_(capacity) {}

  auto push(const T &val) -> void {
    data_[top_] = val;
    ++top_;
  }

  auto pop() -> T {
    --top_;
    return data_[top_];
  }

  [[nodiscard]] auto empty() const -> bool { return top_ == 0; }

  [[nodiscard]] auto size() const -> size_t { return top_; }

  auto reset_full() -> void { top_ = capacity_; }

  auto reset_empty() -> void { top_ = 0; }

 private:
  std::vector<T> data_;
  size_t capacity_ = 0;
  size_t top_ = 0;
};

// ============================================================================
// LaserSearchParams: search configuration
// ============================================================================

struct LaserSearchParams {
  size_t ef_search = 200;              // NOLINT(readability-identifier-naming)
  size_t num_threads = 1;              // NOLINT(readability-identifier-naming)
  size_t beam_width = 16;              // NOLINT(readability-identifier-naming)
  float search_dram_budget_gb = 1.0F;  // NOLINT(readability-identifier-naming)
  size_t aio_events_per_thread_ = 0;   // 0 = auto (2 * beam_width)

  [[nodiscard]] auto effective_aio_events() const -> size_t {
    return aio_events_per_thread_ > 0 ? aio_events_per_thread_ : 2 * beam_width;
  }
};

// ============================================================================
// ResultBuffer: sorted buffer for final k-NN results (separate id/distance arrays)
// ============================================================================

class ResultBuffer {
 public:
  explicit ResultBuffer(size_t capacity)
      : ids_(capacity + 1), distances_(capacity + 1), capacity_(capacity) {}

  void insert(PID data_id, float dist) {
    if (size_ == capacity_ && dist > distances_[size_ - 1]) {
      return;
    }
    size_t lo = binary_search(dist);
    std::memmove(&ids_[lo + 1], &ids_[lo], (size_ - lo) * sizeof(PID));
    ids_[lo] = data_id;
    std::memmove(&distances_[lo + 1], &distances_[lo], (size_ - lo) * sizeof(float));
    distances_[lo] = dist;
    size_ += static_cast<size_t>(size_ < capacity_);
  }

  [[nodiscard]] auto is_full() const -> bool { return size_ == capacity_; }

  auto ids() -> const std::vector<PID, alaya::AlignedAlloc<PID>> & { return ids_; }

  void copy_results(PID *knn) const { std::copy(ids_.begin(), ids_.begin() + size_, knn); }

  void clear() { size_ = 0; }

  void reset(size_t new_capacity) {
    if (new_capacity > capacity_) {
      ids_.resize(new_capacity + 1);
      distances_.resize(new_capacity + 1);
    }
    capacity_ = new_capacity;
    size_ = 0;
  }

 private:
  std::vector<PID, alaya::AlignedAlloc<PID>> ids_;
  std::vector<float, alaya::AlignedAlloc<float>> distances_;
  size_t size_ = 0, capacity_;

  [[nodiscard]] auto binary_search(float dist) const -> size_t {
    size_t lo = 0;
    size_t len = size_;
    size_t half;
    while (len > 1) {
      half = len >> 1;
      len -= half;
      lo += static_cast<size_t>(distances_[lo + half - 1] < dist) * half;
    }
    return (lo < size_ && distances_[lo] < dist) ? lo + 1 : lo;
  }
};

}  // namespace symqg
