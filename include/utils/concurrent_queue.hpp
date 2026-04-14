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
 * @file concurrent_queue.hpp
 * @brief Thread-safe queue for managing shared resources.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace alaya {

template <typename T>
class ConcurrentQueue {
  using chrono_us_t = std::chrono::microseconds;
  using mutex_locker = std::unique_lock<std::mutex>;

  std::queue<T> queue_;
  std::mutex mutex_;
  std::mutex push_mut_;
  std::mutex pop_mut_;
  std::condition_variable push_cv_;
  std::condition_variable pop_cv_;

 public:
  ConcurrentQueue() = default;

  ~ConcurrentQueue() {
    this->push_cv_.notify_all();
    this->pop_cv_.notify_all();
  }

  auto size() -> uint64_t {
    mutex_locker lk(this->mutex_);
    uint64_t ret = queue_.size();
    lk.unlock();
    return ret;
  }

  auto empty() -> bool { return (this->size() == 0); }

  void push(const T &new_val) {
    mutex_locker lk(this->mutex_);
    this->queue_.push(new_val);
    lk.unlock();
  }

  void push(T &&new_val) {
    mutex_locker lk(this->mutex_);
    this->queue_.push(std::move(new_val));
    lk.unlock();
  }

  template <class Iterator>
  void insert(Iterator iter_begin, Iterator iter_end) {
    mutex_locker lk(this->mutex_);
    for (Iterator it = iter_begin; it != iter_end; ++it) {
      this->queue_.push(*it);
    }
    lk.unlock();
  }

  auto pop() -> T {
    mutex_locker lk(this->mutex_);
    if (this->queue_.empty()) {
      lk.unlock();
      return T{};
    }
    T ret = std::move(this->queue_.front());
    this->queue_.pop();
    lk.unlock();
    return ret;
  }

  void wait_for_push_notify(chrono_us_t wait_time = chrono_us_t{10}) {
    mutex_locker lk(this->push_mut_);
    this->push_cv_.wait_for(lk, wait_time);
    lk.unlock();
  }

  void wait_for_pop_notify(chrono_us_t wait_time = chrono_us_t{10}) {
    mutex_locker lk(this->pop_mut_);
    this->pop_cv_.wait_for(lk, wait_time);
    lk.unlock();
  }

  void push_notify_one() { this->push_cv_.notify_one(); }
  void push_notify_all() { this->push_cv_.notify_all(); }
  void pop_notify_one() { this->pop_cv_.notify_one(); }
  void pop_notify_all() { this->pop_cv_.notify_all(); }
};

}  // namespace alaya
