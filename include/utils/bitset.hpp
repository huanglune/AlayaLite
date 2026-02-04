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
#include <ranges>
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
    data_.resize(math::round_up_pow2(num_bits, 64), 0);
  }

  /**
   * @brief Resize the bitset to a new size
   *
   * @param num_bits The new number of bits
   */
  void resize(size_t num_bits) {
    size_ = num_bits;
    data_.resize(math::round_up_pow2(num_bits, 64), 0);
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

using Bitset = DynamicBitset;  // Change to SparseBitset or HierarchicalBitset as needed

}  // namespace alaya
