// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "storage/io/page_reader.hpp"

namespace alaya::storage::io {

class ReadBatchAwaitable {
 public:
  ReadBatchAwaitable(PageReader &reader,
                     ResumeExecutor &executor,
                     std::span<const ReadRequest> requests)
      : state_(std::make_shared<State>(reader, executor, requests.size())) {
    state_->requests.assign(requests.begin(), requests.end());
  }
  ReadBatchAwaitable(ReadBatchAwaitable &&) noexcept = default;
  ReadBatchAwaitable(const ReadBatchAwaitable &) = delete;
  auto operator=(const ReadBatchAwaitable &) -> ReadBatchAwaitable & = delete;
  ~ReadBatchAwaitable() {
    if (state_ != nullptr && state_->handle) (void)state_->handle.cancel();
  }

  [[nodiscard]] auto await_ready() const noexcept -> bool {
    return state_->remaining.load(std::memory_order_acquire) == 0;
  }

  auto await_suspend(std::coroutine_handle<> coroutine) -> bool {
    auto &state = *state_;
    state.coroutine = coroutine;
    state.keep_alive = state_;
    if (state.remaining.load(std::memory_order_acquire) == 0) {
      state.keep_alive.reset();
      return false;
    }
    try {
      state.handle = state.reader.submit(state.requests, {&State::complete, &state});
    } catch (...) {
      state.keep_alive.reset();
      throw;
    }
    state.armed.store(true, std::memory_order_release);
    if (state.remaining.load(std::memory_order_acquire) == 0) state.schedule();
    return true;
  }

  auto await_resume() -> std::vector<ReadResult> {
    std::lock_guard lock(state_->mutex);
    return std::move(state_->results);
  }

  auto cancel() noexcept -> CancelResult {
    return state_->handle ? state_->handle.cancel() : CancelResult::already_complete;
  }

 private:
  struct State {
    State(PageReader &reader_in, ResumeExecutor &executor_in, std::size_t count)
        : reader(reader_in), executor(executor_in), remaining(count) {
      requests.reserve(count);
      results.reserve(count);
    }

    static void complete(void *context, ReadResult result) noexcept {
      auto &state = *static_cast<State *>(context);
      {
        std::lock_guard lock(state.mutex);
        state.results.push_back(std::move(result));
      }
      if (state.remaining.fetch_sub(1, std::memory_order_acq_rel) == 1 &&
          state.armed.load(std::memory_order_acquire)) {
        state.schedule();
      }
    }

    void schedule() noexcept {
      if (!scheduled.exchange(true, std::memory_order_acq_rel)) {
        auto lifetime = std::move(keep_alive);
        if (!executor.execute(coroutine)) std::terminate();
      }
    }

    PageReader &reader;
    ResumeExecutor &executor;
    std::vector<ReadRequest> requests;
    std::vector<ReadResult> results;
    std::mutex mutex;
    std::atomic_size_t remaining;
    std::atomic_bool armed{false};
    std::atomic_bool scheduled{false};
    std::coroutine_handle<> coroutine{};
    BatchHandle handle;
    std::shared_ptr<State> keep_alive;
  };

  std::shared_ptr<State> state_;
};

[[nodiscard]] inline auto read_pages(PageReader &reader,
                                     ResumeExecutor &executor,
                                     std::span<const ReadRequest> requests) -> ReadBatchAwaitable {
  return ReadBatchAwaitable(reader, executor, requests);
}

[[nodiscard]] inline auto read_pages_blocking(PageReader &reader,
                                              std::span<const ReadRequest> requests)
    -> std::vector<ReadResult> {
  if (requests.empty()) return {};
  struct State {
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<ReadResult> results;
    std::size_t remaining;
  } state{{}, {}, {}, requests.size()};
  state.results.reserve(requests.size());
  const auto complete = [](void *context, ReadResult result) noexcept {
    auto &state = *static_cast<State *>(context);
    std::lock_guard lock(state.mutex);
    state.results.push_back(std::move(result));
    if (--state.remaining == 0) state.cv.notify_one();
  };
  auto handle = reader.submit(requests, {complete, &state});
  std::unique_lock lock(state.mutex);
  state.cv.wait(lock, [&] {
    return state.remaining == 0;
  });
  return std::move(state.results);
}

}  // namespace alaya::storage::io
