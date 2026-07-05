// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file coro_gate.hpp
 * @brief AsyncGate<T>: a coroutine-suspending object pool.
 *
 * Hands out pooled objects to coroutines without ever blocking the executor
 * thread. A thread-blocking pool (pop + condition-variable wait) deadlocks
 * under coroutine executors: once every pooled object is held by a SUSPENDED
 * coroutine and every executor thread is blocked waiting for one, the resume
 * that would release an object has no thread left to run on. AsyncGate
 * suspends the acquiring coroutine instead, and release() hands the object
 * directly to the oldest waiter and reschedules it through its thread pool —
 * never inline, so a release cannot run an unbounded chain of waiter bodies
 * on the releasing thread's stack.
 *
 * Ownership: the gate stores raw pointers; object lifetime belongs to the
 * caller. clear() empties the free list and requires that no waiter is parked
 * (i.e. quiesce acquirers first — same discipline as draining I/O before
 * destroying a reactor).
 */

#pragma once

#include <coroutine>
#include <deque>
#include <mutex>
#include <vector>

#include "coro/thread_pool.hpp"
#include "utils/log.hpp"

namespace alaya {

template <typename T>
class AsyncGate {
 public:
  AsyncGate() = default;
  AsyncGate(const AsyncGate &) = delete;
  AsyncGate &operator=(const AsyncGate &) = delete;

  /// Add an object to the free list (setup phase, or returning one that no
  /// longer participates in acquire/release cycles).
  void add(T *item) {
    std::lock_guard<std::mutex> guard(mutex_);
    free_.push_back(item);
  }

  class Awaiter {
   public:
    Awaiter(AsyncGate &gate, coro::thread_pool &pool) : gate_(gate), pool_(pool) {}

    auto await_ready() const noexcept -> bool { return false; }

    auto await_suspend(std::coroutine_handle<> handle) -> bool {
      std::lock_guard<std::mutex> guard(gate_.mutex_);
      if (!gate_.free_.empty()) {
        item_ = gate_.free_.back();
        gate_.free_.pop_back();
        return false;  // resume immediately, object in hand
      }
      handle_ = handle;
      gate_.waiters_.push_back(this);
      // From here release() may hand us an object and resume the coroutine on
      // another thread at any time; touch no member after the guard drops.
      return true;
    }

    auto await_resume() const noexcept -> T * { return item_; }

   private:
    friend class AsyncGate;
    AsyncGate &gate_;
    coro::thread_pool &pool_;
    T *item_ = nullptr;
    std::coroutine_handle<> handle_;
  };

  /// co_await gate.acquire(pool) -> T*. Suspends when the free list is empty;
  /// the eventual release() resumes the coroutine on @p pool.
  auto acquire(coro::thread_pool &pool) -> Awaiter { return Awaiter{*this, pool}; }

  /// Return an object. If a coroutine is parked the object is handed straight
  /// to the oldest waiter (it never touches the free list — no lost wakeups)
  /// and the waiter is rescheduled through its pool.
  void release(T *item) {
    Awaiter *waiter = nullptr;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (waiters_.empty()) {
        free_.push_back(item);
        return;
      }
      waiter = waiters_.front();
      waiters_.pop_front();
      waiter->item_ = item;
    }
    // The waiter frame lives in the coroutine we are about to resume; copy
    // what we need first and touch nothing of it after resume().
    coro::thread_pool &pool = waiter->pool_;
    const std::coroutine_handle<> handle = waiter->handle_;
    if (!pool.resume(handle)) {
      LOG_ERROR("AsyncGate: thread pool rejected resume (shutdown with waiters parked?)");
    }
  }

  /// Drop all free objects (does not delete them). Callers must have
  /// quiesced acquirers: clearing with parked waiters is a bug.
  void clear() {
    std::lock_guard<std::mutex> guard(mutex_);
    free_.clear();
  }

  [[nodiscard]] auto free_count() -> size_t {
    std::lock_guard<std::mutex> guard(mutex_);
    return free_.size();
  }

 private:
  std::mutex mutex_;
  std::vector<T *> free_;
  std::deque<Awaiter *> waiters_;
};

}  // namespace alaya
