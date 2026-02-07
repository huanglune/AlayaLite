/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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
#include <cstring>
#include <unordered_set>
#include <vector>
#include "math.hpp"
#include "memory.hpp"

namespace alaya {

/**
 * @brief A dynamic bitset implementation using a vector of uint64_t.
 *
 * Pros:
 * - Dynamic size, can be determined at runtime
 * - Efficient memory usage for dense bit sets
 * - Fast set, get, and reset operations
 *
 * Cons:
 * - Slightly slower than std::bitset for small, fixed-size bit sets
 * - Not optimized for very sparse data
 */
class DynamicBitset {
 private:
  std::vector<uint64_t, AlignedAlloc<uint64_t>> data_;
  size_t size_{0};

 public:
  /**
   * @brief Default constructor creates an empty bitset
   */
  DynamicBitset() = default;

  /**
   * @brief Construct a new Dynamic Bitset object
   *
   * @param num_bits The number of bits in the bitset
   */
  explicit DynamicBitset(size_t num_bits) : size_(num_bits) {
    data_.resize(math::round_up_pow2(num_bits, 64) / 64, 0);
  }

  /**
   * @brief Resize the bitset to a new size
   *
   * @param num_bits The new number of bits
   */
  void resize(size_t num_bits) {
    size_ = num_bits;
    data_.resize(math::round_up_pow2(num_bits, 64) / 64, 0);
  }

  /**
   * @brief Set the bit at the specified position
   *
   * @param pos The position of the bit to set
   */
  void set(size_t pos) noexcept { data_[pos >> 6] |= (1ULL << (pos & 63)); }

  /**
   * @brief Get the value of the bit at the specified position
   *
   * @param pos The position of the bit to get
   * @return true if the bit is set, false otherwise
   */
  [[nodiscard]] auto get(size_t pos) const -> bool {
    return (data_[pos >> 6] >> (pos & 63)) & 1ULL;  // NOLINT
  }

  /**
   * @brief Get the pos address object
   *
   * @param pos The position of the bit to get
   * @return void* The address of the bit
   */
  auto get_address(size_t pos) -> void * { return data_.data() + pos / 64; }

  /**
   * @brief Reset the bit at the specified position
   *
   * @param pos The position of the bit to reset
   */
  void reset(size_t pos) { data_[pos / 64] &= ~(1ULL << (pos % 64)); }

  /**
   * @brief Reset all bits to zero
   */
  void reset() { std::ranges::fill(data_, 0ULL); }

  /**
   * @brief Get pointer to internal data for direct I/O operations.
   * @return Pointer to internal uint64_t array
   */
  [[nodiscard]] auto data() noexcept -> uint64_t * { return data_.data(); }
  [[nodiscard]] auto data() const noexcept -> const uint64_t * { return data_.data(); }

  /**
   * @brief Get size of internal storage in bytes.
   * @return Size in bytes
   */
  [[nodiscard]] auto size_in_bytes() const noexcept -> size_t {
    return data_.size() * sizeof(uint64_t);
  }

  /**
   * @brief Find the position of the first zero (unset) bit
   *
   * Uses CPU instructions (TZCNT/BSF) for O(1) lookup within each 64-bit block.
   * Overall complexity is O(n/64) where n is the number of bits.
   *
   * @return int The position of the first zero bit, or -1 if all bits are set
   */
  [[nodiscard]] auto find_first_zero() const -> int {
    // TODO(hl): Optimize with SIMD if needed
    for (size_t i = 0; i < data_.size(); ++i) {
      if (data_[i] != ~0ULL) {  // Block has at least one zero bit
        size_t pos = (i * 64) + math::count_trailing_zeros(~data_[i]);
        return (pos < size_) ? static_cast<int>(pos) : -1;
      }
    }
    return -1;
  }
};

/**
 * @brief A sparse bitset implementation using an unordered set.
 *
 * Pros:
 * - Extremely memory efficient for very sparse bit sets
 * - Dynamic size
 *
 * Cons:
 * - Slower than dense bitset implementations for most operations
 * - High memory usage for dense bit sets
 * - No cache locality
 */
class SparseBitset {
 private:
  std::unordered_set<size_t> set_bits_;

 public:
  /**
   * @brief Set the bit at the specified position
   *
   * @param pos The position of the bit to set
   */
  void set(size_t pos) { set_bits_.insert(pos); }

  /**
   * @brief Get the value of the bit at the specified position
   *
   * @param pos The position of the bit to get
   * @return true if the bit is set, false otherwise
   */
  auto get(size_t pos) const -> bool { return set_bits_.contains(pos); }

  /**
   * @brief Reset the bit at the specified position
   *
   * @param pos The position of the bit to reset
   */
  void reset(size_t pos) { set_bits_.erase(pos); }
};

/**
 * @brief A hierarchical bitset implementation for efficient "find first set" operations.
 *
 * Pros:
 * - Very fast "find first set" operation
 * - Good performance for large bitsets
 *
 * Cons:
 * - More complex implementation
 * - Slightly higher memory usage than simple bitset
 * - Set and get operations are slightly slower than simple bitset
 */
class HierarchicalBitset {
 private:
  std::vector<uint64_t> data_;
  std::vector<uint64_t> summary_;
  size_t size_;  // NOLINT
  static const size_t kBitsPerBlock = 512;
  static const size_t kSummaryBlockSize = 64;

 public:
  /**
   * @brief Construct a new Hierarchical Bitset object
   *
   * @param num_bits The number of bits in the bitset
   */
  explicit HierarchicalBitset(size_t num_bits) : size_(num_bits) {
    data_.resize((num_bits + 63) / 64, 0);
    summary_.resize((data_.size() + 63) / 64, 0);
  }

  /**
   * @brief Set the bit at the specified position
   *
   * @param pos The position of the bit to set
   */
  void set(size_t pos) {
    size_t block = pos / kBitsPerBlock;
    size_t offset = pos % kBitsPerBlock;
    data_[block * 8 + offset / 64] |= (1ULL << (offset % 64));
    summary_[block / kSummaryBlockSize] |= (1ULL << (block % kSummaryBlockSize));
  }

  /**
   * @brief Get the value of the bit at the specified position
   *
   * @param pos The position of the bit to get
   * @return true if the bit is set, false otherwise
   */
  auto get(size_t pos) const -> bool {
    size_t block = pos / kBitsPerBlock;
    size_t offset = pos % kBitsPerBlock;
    return (data_[(block * 8) + (offset / 64)] & (1ULL << (offset % 64))) != 0;
  }

  /**
   * @brief Find the position of the first set bit
   *
   * @return int The position of the first set bit, or -1 if no bit is set
   */
  auto find_first_set() const -> int {
    for (size_t i = 0; i < summary_.size(); ++i) {
      if (summary_[i] == 0) {
        continue;
      }
      size_t block = (i * kSummaryBlockSize) + math::count_trailing_zeros(summary_[i]);
      for (size_t j = 0; j < 8; ++j) {
        if (data_[(block * 8) + j] == 0) {
          continue;
        }
        return static_cast<int>((block * kBitsPerBlock) + (j * 64) +
                                math::count_trailing_zeros(data_[block * 8 + j]));
      }
    }
    return -1;
  }
};

/**
 * @brief A fixed-size bitset for very small sizes (8, 16, 32, 64 bits).
 *
 * Uses a single integer for O(1) all operations with single CPU instructions.
 * Ideal for managing small resource pools (e.g., buffer slots, connection pools).
 *
 * @tparam N Number of bits (must be 8, 16, 32, or 64)
 */
template <size_t N>
class FixedBitset {
  static_assert(N == 8 || N == 16 || N == 32 || N == 64 || N == 128, "Unsupported size");

 private:
  // clang-format off
  using StorageType = std::conditional_t<N <= 8, uint8_t,      // NOLINT(whitespace/operators)
                      std::conditional_t<N <= 16, uint16_t,   // NOLINT(whitespace/operators)
                      std::conditional_t<N <= 32, uint32_t, uint64_t>>>;
  // clang-format on
  StorageType data_{0};
  static constexpr StorageType kAllOnes = static_cast<StorageType>(~StorageType{0});

 public:
  FixedBitset() = default;

  void set(size_t pos) noexcept { data_ |= (StorageType{1} << pos); }
  void reset(size_t pos) noexcept { data_ &= ~(StorageType{1} << pos); }
  void reset() noexcept { data_ = 0; }

  [[nodiscard]] auto get(size_t pos) const noexcept -> bool {
    return (data_ >> pos) & StorageType{1};
  }

  /**
   * @brief Find first zero bit in O(1) using TZCNT/BSF instruction
   * @return Position of first zero bit, or -1 if all bits are set
   */
  [[nodiscard]] auto find_first_zero() const noexcept -> int { return std::countr_one(data_); }

  /**
   * @brief Find first set bit in O(1) using TZCNT/BSF instruction
   * @return Position of first set bit, or -1 if all bits are zero
   */
  [[nodiscard]] auto find_first_set() const noexcept -> int { return std::countr_zero(data_); }

  [[nodiscard]] auto count() const noexcept -> int { return std::popcount(data_); }

  [[nodiscard]] auto empty() const noexcept -> bool { return data_ == 0; }
  [[nodiscard]] auto full() const noexcept -> bool { return data_ == kAllOnes; }
  [[nodiscard]] static constexpr auto capacity() noexcept -> size_t { return N; }
  [[nodiscard]] auto data() const noexcept -> StorageType { return data_; }
};
/**
 * @brief Specialization of FixedBitset for 128 bits using two uint64_t.
 */
template <>
class alignas(16) FixedBitset<128> {
 private:
  uint64_t low_{0};   // bits 0-63
  uint64_t high_{0};  // bits 64-127

 public:
  FixedBitset() = default;

  void set(size_t pos) noexcept {
    if (pos < 64) {
      low_ |= (1ULL << pos);
    } else {
      high_ |= (1ULL << (pos - 64));
    }
  }

  void reset(size_t pos) noexcept {
    if (pos < 64) {
      low_ &= ~(1ULL << pos);
    } else {
      high_ &= ~(1ULL << (pos - 64));
    }
  }

  void reset() noexcept {
    low_ = 0;
    high_ = 0;
  }

  [[nodiscard]] auto get(size_t pos) const noexcept -> bool {
    if (pos < 64) {
      return ((low_ >> pos) & 1ULL) != 0U;
    }
    return ((high_ >> (pos - 64)) & 1ULL) != 0U;
  }

  /**
   * @brief Find the first zero bit (first free slot).
   * @return 0-127 for position, 128 if full.
   */
  [[nodiscard]] auto find_first_zero() const noexcept -> int {
    if (low_ != ~0ULL) {
      return std::countr_one(low_);
    }
    return 64 + std::countr_one(high_);
  }

  /**
   * @brief Find the first set bit (first occupied slot).
   * @return 0-127 for position, 128 if empty.
   */
  [[nodiscard]] auto find_first_set() const noexcept -> int {
    if (low_ != 0) {
      return std::countr_zero(low_);
    }
    return 64 + std::countr_zero(high_);
  }

  [[nodiscard]] auto count() const noexcept -> int {
    return std::popcount(low_) + std::popcount(high_);
  }

  [[nodiscard]] auto empty() const noexcept -> bool { return (low_ | high_) == 0; }

  [[nodiscard]] auto full() const noexcept -> bool { return (low_ == ~0ULL) && (high_ == ~0ULL); }

  [[nodiscard]] static constexpr auto capacity() noexcept -> size_t { return 128; }

  [[nodiscard]] auto raw_data() const noexcept -> std::pair<uint64_t, uint64_t> {
    return {low_, high_};
  }
};

// Convenient type aliases
using Bitset = DynamicBitset;  // Change to SparseBitset or HierarchicalBitset as needed
using Bitset8 = FixedBitset<8>;
using Bitset16 = FixedBitset<16>;
using Bitset32 = FixedBitset<32>;
using Bitset64 = FixedBitset<64>;
using Bitset128 = FixedBitset<128>;

static_assert(sizeof(Bitset8) == 1, "Bitset8 size mismatch");
static_assert(sizeof(Bitset16) == 2, "Bitset16 size mismatch");
static_assert(sizeof(Bitset32) == 4, "Bitset32 size mismatch");
static_assert(sizeof(Bitset64) == 8, "Bitset64 size mismatch");
static_assert(sizeof(Bitset128) == 16, "Bitset128 size mismatch");

}  // namespace alaya
