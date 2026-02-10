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

#include <cstddef>
#include <cstring>
#include <vector>
#include "../index/neighbor.hpp"
#include "bitset.hpp"
#include "memory.hpp"

namespace alaya {

template <typename DistanceType, typename IDType>
struct CandidateList {
  using SearchResultType = ::alaya::SearchResult<IDType, DistanceType>;
  CandidateList(IDType n, int capacity)
      : nb_(n), capacity_(capacity), data_(capacity_ + 1), vis_(n) {}

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
    // for (size_t i = 0; i < size_; i++) {
    //   LOG_INFO("i {} ,dist is {}", data_[i].id_, data_[i].distance_);
    // }
    // LOG_INFO("cur is {} , size {}", cur_, size_);
    return true;
  }

  void emplace_insert(IDType u, DistanceType dist) {
    if (dist >= data_[size_ - 1].distance_) {
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

    // LOG_INFO("pop idx is {} , {}",data_[pre].id_, get_id(data_[pre].id_));
    return get_id(data_[pre].id_);
  }

  auto has_next() const -> bool { return cur_ < size_; }
  auto next_id() const -> IDType { return get_id(data_[cur_].id_); }
  auto id(IDType i) const -> IDType { return get_id(data_[i].id_); }
  auto dist(IDType i) const -> DistanceType { return data_[i].distance_; }
  auto size() const -> size_t { return size_; }
  auto capacity() const -> size_t { return capacity_; }

  constexpr static int kMask = 2147483647;
  auto get_id(IDType id) const -> IDType { return id & kMask; }
  // Need to assert IDType is uint32_t instead of uint64_t
  void set_checked(IDType &id) { id |= 1 << 31; }
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

  size_t nb_, size_ = 0, cur_ = 0, capacity_;
  std::vector<Neighbor<IDType, DistanceType>, AlignedAlloc<Neighbor<IDType, DistanceType>>> data_;
  Bitset vis_;
};

}  // namespace alaya
