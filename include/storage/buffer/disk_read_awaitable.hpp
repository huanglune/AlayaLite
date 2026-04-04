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

#include <atomic>
#include <coroutine>
#include <cstdint>

#include "storage/io/io_engine.hpp"

namespace alaya {

/**
 * @brief A simple yield awaitable that suspends the coroutine without re-queuing.
 *
 * The Worker's round-robin loop will resume it from local_tasks_ on the next pass.
 * This avoids the double-resume issue caused by co_await scheduler.schedule()
 * which pushes the handle back to the TaskQueue.
 */
struct YieldAwaitable {
  auto await_ready() noexcept -> bool { return false; }
  void await_suspend(std::coroutine_handle<> /*h*/) noexcept {}
  void await_resume() noexcept {}
};

/// Per-page async I/O state, embedded in the BufferPool Frame.
/// Move-constructible to allow storage in std::vector (move happens only during
/// single-threaded initialization, so non-atomic load/store is safe).
struct AsyncReadState {
  std::atomic<bool> finish_read_{true};  ///< true = data valid (default for sync pages)
  std::atomic<bool> has_error_{false};   ///< true = I/O completed with error

  AsyncReadState() = default;

  AsyncReadState(AsyncReadState &&other) noexcept
      : finish_read_(other.finish_read_.load(std::memory_order_relaxed)),
        has_error_(other.has_error_.load(std::memory_order_relaxed)) {}

  auto operator=(AsyncReadState &&other) noexcept -> AsyncReadState & {
    finish_read_.store(other.finish_read_.load(std::memory_order_relaxed),
                       std::memory_order_relaxed);
    has_error_.store(other.has_error_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    return *this;
  }

  AsyncReadState(const AsyncReadState &) = delete;
  auto operator=(const AsyncReadState &) -> AsyncReadState & = delete;

  void reset_for_async() {
    has_error_.store(false, std::memory_order_relaxed);
    finish_read_.store(false, std::memory_order_release);
  }
};

/// Context passed as callback_arg to submit_async_read. Heap-allocated, freed in callback.
struct AsyncReadNotifier {
  AsyncReadState *state_;
};

/**
 * @brief I/O completion callback for async buffer pool reads.
 *
 * On success (result == expected bytes), sets finish_read_ = true.
 * On error (negative errno or short read), sets has_error_ = true then finish_read_ = true
 * so polling coroutines wake up and can detect the failure.
 * Frees the AsyncReadNotifier.
 */
inline void async_read_callback(void *arg, int32_t result) {
  auto *notifier = static_cast<AsyncReadNotifier *>(arg);
  if (result <= 0) {
    notifier->state_->has_error_.store(true, std::memory_order_relaxed);
  }
  notifier->state_->finish_read_.store(true, std::memory_order_release);
  delete notifier;
}

}  // namespace alaya
