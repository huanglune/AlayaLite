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
#include <limits>
#include <type_traits>
#include <vector>
#include "../index/neighbor.hpp"
#include "memory.hpp"

namespace alaya {

template <typename DistanceType, typename IDType>
struct CandidateList {
  static_assert(std::is_unsigned_v<IDType>, "CandidateList requires an unsigned IDType");
  using SearchResultType = ::alaya::SearchResult<IDType, DistanceType>;
  explicit CandidateList(int capacity) : capacity_(capacity), data_(capacity_ + 1) {}

  auto find_bsearch(DistanceType dist) -> int {
    int l = 0;
    int r = size_;
    while (l < r) {
      int mid = (l + r) / 2;
      if (data_[mid].distance_ > dist) {
        r = mid;
      } else {
        l = mid + 1;
      }
    }
    return l;
  }

  auto insert(IDType u, DistanceType dist) -> bool {
    if (size_ == capacity_ && dist >= data_[size_ - 1].distance_) {
      return false;
    }
    int lo = find_bsearch(dist);
    std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Neighbor<IDType, DistanceType>));
    data_[lo] = {u, dist};
    if (size_ < capacity_) {
      size_++;
    }
    if (static_cast<size_t>(lo) < cur_) {
      cur_ = lo;
    }
    return true;
  }

  void emplace_insert(IDType u, DistanceType dist) {
    if (size_ == 0 || dist >= data_[size_ - 1].distance_) {
      return;
    }
    int lo = find_bsearch(dist);
    std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Neighbor<IDType, DistanceType>));
    data_[lo] = {u, dist};
  }

  auto top() -> IDType { return data_[cur_].id_; }
  auto pop() -> IDType {
    set_checked(data_[cur_].id_);
    int pre = cur_;
    while (cur_ < size_ && is_checked(data_[cur_].id_)) {
      cur_++;
    }

    return get_id(data_[pre].id_);
  }

  auto has_next() const -> bool { return cur_ < size_; }
  // Short-circuit check: caller can skip insert() + its internal bookkeeping entirely.
  // Matches Laser symqglib SearchBuffer::is_full(float) — a measurable win in scan_neighbors
  // where 64 candidates per node get rejected when the pool is saturated.
  auto is_full(DistanceType dist) const -> bool {
    return size_ == capacity_ && dist >= data_[size_ - 1].distance_;
  }
  auto next_id() const -> IDType { return get_id(data_[cur_].id_); }
  auto id(IDType i) const -> IDType { return get_id(data_[i].id_); }
  auto dist(IDType i) const -> DistanceType { return data_[i].distance_; }
  auto size() const -> size_t { return size_; }
  auto capacity() const -> size_t { return capacity_; }

  constexpr static IDType kMask = static_cast<IDType>(0x7fffffffU);
  auto get_id(IDType id) const -> IDType { return id & kMask; }
  void set_checked(IDType &id) { id |= static_cast<IDType>(1U << 31); }
  auto is_checked(IDType id) -> bool { return (id >> 31 & 1) != 0; }
  auto is_full() -> bool { return size_ == capacity_; }

  void clear() {
    size_ = 0;
    cur_ = 0;
  }

  void resize(size_t new_capacity) {
    capacity_ = new_capacity;
    if (data_.size() < capacity_ + 1) {
      data_.resize(capacity_ + 1);
    }
    size_ = 0;
    cur_ = 0;
  }

  auto to_search_result(size_t topk = static_cast<size_t>(-1)) -> SearchResultType {
    size_t effective_topk = (topk == static_cast<size_t>(-1)) ? size_ : topk;
    SearchResultType result(effective_topk);

    size_t actual = std::min(size_, effective_topk);
    for (size_t i = 0; i < actual; ++i) {
      result.ids_.emplace_back(get_id(data_[i].id_));
      result.distances_.emplace_back(data_[i].distance_);
    }

    if (actual < effective_topk) {
      result.ids_.insert(result.ids_.end(), effective_topk - actual, static_cast<IDType>(-1));
      result.distances_.insert(result.distances_.end(),
                               effective_topk - actual,
                               std::numeric_limits<DistanceType>::max());
    }
    return result;
  }

  size_t size_ = 0, cur_ = 0, capacity_;
  std::vector<Neighbor<IDType, DistanceType>, AlignedAlloc<Neighbor<IDType, DistanceType>>> data_;
};

}  // namespace alaya
