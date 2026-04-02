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
#include <cstdint>
#include <vector>

namespace alaya {

/**
 * @brief Stack-based free slot tracker.
 *
 * Provides O(1) allocation and deallocation by maintaining a stack of free slot IDs.
 * When a slot is freed, its ID is pushed onto the stack.
 * When allocating, we pop from the stack if available.
 *
 * File format:
 * - uint32_t count: number of free slots
 * - uint32_t slots[count]: array of free slot IDs
 */
class FreeList {
 public:
  FreeList() = default;

  void push(uint32_t slot_id) { free_slots_.push_back(slot_id); }
  [[nodiscard]] auto pop() -> int32_t {
    if (free_slots_.empty()) {
      return -1;
    }
    auto slot = static_cast<int32_t>(free_slots_.back());
    free_slots_.pop_back();
    return slot;
  }
  [[nodiscard]] auto empty() const noexcept -> bool { return free_slots_.empty(); }
  [[nodiscard]] auto size() const noexcept -> size_t { return free_slots_.size(); }
  void clear() { free_slots_.clear(); }
  [[nodiscard]] auto data() noexcept -> uint32_t * { return free_slots_.data(); }
  [[nodiscard]] auto data() const noexcept -> const uint32_t * { return free_slots_.data(); }
  void resize(size_t count) { free_slots_.resize(count); }
  [[nodiscard]] auto size_in_bytes() const noexcept -> size_t {
    return sizeof(uint32_t) + free_slots_.size() * sizeof(uint32_t);
  }

 private:
  std::vector<uint32_t> free_slots_;
};

}  // namespace alaya
