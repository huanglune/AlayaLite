// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace alaya::diskann {

class VisitedBitset {
 public:
  void resize(uint64_t max_slot_id) {
    n_bits_ = max_slot_id;
    words_.assign(word_count_for(max_slot_id), 0);
    dirty_words_.clear();
  }

  void clear() {
    for (const uint32_t w : dirty_words_) {
      words_[w] = 0;
    }
    dirty_words_.clear();
  }

  [[nodiscard]] bool test(uint32_t id) const {
    ensure_in_range(id);
    const uint64_t word = words_[id >> kWordShift];
    return (word & bit_mask(id)) != 0;
  }

  void set(uint32_t id) {
    ensure_in_range(id);
    uint64_t &word = words_[id >> kWordShift];
    if (word == 0) {
      dirty_words_.push_back(id >> kWordShift);
    }
    word |= bit_mask(id);
  }

  [[nodiscard]] bool test_and_set(uint32_t id) {
    ensure_in_range(id);
    uint64_t &word = words_[id >> kWordShift];
    if (word == 0) {
      dirty_words_.push_back(id >> kWordShift);
    }
    const uint64_t mask = bit_mask(id);
    const bool was_set = (word & mask) != 0;
    word |= mask;
    return !was_set;
  }

  [[nodiscard]] uint64_t size_bits() const { return n_bits_; }
  [[nodiscard]] size_t word_count() const { return words_.size(); }

 private:
  static constexpr uint32_t kWordShift = 6;
  static constexpr uint32_t kWordMask = 63;

  static size_t word_count_for(uint64_t n_bits) {
    return static_cast<size_t>((n_bits + kWordMask) >> kWordShift);
  }

  static uint64_t bit_mask(uint32_t id) { return uint64_t{1} << (id & kWordMask); }

  void ensure_in_range(uint32_t id) const {
    if (id >= n_bits_) {
      throw std::out_of_range("VisitedBitset: id out of range");
    }
  }

  uint64_t n_bits_ = 0;
  std::vector<uint64_t> words_;
  /// Indices of every nonzero word (recorded on the 0 -> nonzero transition,
  /// so no duplicates). Lets clear() cost O(bits set) instead of a full-array
  /// memset, which at large slot counts is megabytes per query.
  std::vector<uint32_t> dirty_words_;
};

}  // namespace alaya::diskann
