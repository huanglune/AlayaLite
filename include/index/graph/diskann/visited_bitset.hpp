// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace alaya::diskann {

class VisitedBitset {
 public:
  void resize(uint64_t max_slot_id) {
    n_bits_ = max_slot_id;
    words_.assign(word_count_for(max_slot_id), 0);
  }

  void clear() {
    if (!words_.empty()) {
      std::memset(words_.data(), 0, words_.size() * sizeof(uint64_t));
    }
  }

  [[nodiscard]] bool test(uint32_t id) const {
    ensure_in_range(id);
    const uint64_t word = words_[id >> kWordShift];
    return (word & bit_mask(id)) != 0;
  }

  void set(uint32_t id) {
    ensure_in_range(id);
    words_[id >> kWordShift] |= bit_mask(id);
  }

  [[nodiscard]] bool test_and_set(uint32_t id) {
    ensure_in_range(id);
    uint64_t &word = words_[id >> kWordShift];
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
};

}  // namespace alaya::diskann
