// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <cerrno>

#include "storage/io/alignment.hpp"

namespace alaya::storage::io {

class ThreadpoolPageReader final : public PageReader {
 private:
  struct BatchState {
    std::atomic_size_t remaining{0};
    std::atomic_bool cancel_requested{false};
  };

  struct Work {
    ReadRequest request;
    Completion completion;
    std::shared_ptr<BatchState> batch;
  };

 public:
  explicit ThreadpoolPageReader(const std::filesystem::path &path, ReaderOptions options = {})
      : constraints_(conservative_constraints(OpenMode::buffered,
                                              std::max<std::uint32_t>(1, options.queue_depth))),
        capacity_(constraints_.max_batch) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::system_error(errno, std::generic_category(), "open");
    const auto hardware = std::max(1U, std::thread::hardware_concurrency());
    const auto count = std::min<std::uint32_t>(capacity_, std::min(32U, hardware));
    try {
      workers_.reserve(count);
      for (std::uint32_t i = 0; i < count; ++i)
        workers_.emplace_back([this] {
          worker_loop();
        });
    } catch (...) {
      shutdown();
      throw;
    }
  }

  ~ThreadpoolPageReader() override { shutdown(); }

  [[nodiscard]] auto constraints() const noexcept -> ReadConstraints override {
    return constraints_;
  }

  [[nodiscard]] auto submit(std::span<const ReadRequest> requests, Completion completion)
      -> BatchHandle override {
    if ((completion.fn == nullptr && !requests.empty()) || requests.size() > capacity_)
      throw std::invalid_argument("invalid PageReader batch");
    for (const auto &request : requests) {
      if (!validate_read_request(request, constraints_))
        throw std::invalid_argument("read request violates alignment constraints");
    }

    auto batch = std::make_shared<BatchState>();
    batch->remaining.store(requests.size(), std::memory_order_relaxed);
    auto handle = make_batch_handle([weak = std::weak_ptr(batch)]() noexcept {
      const auto state = weak.lock();
      if (!state || state->remaining.load(std::memory_order_acquire) == 0)
        return CancelResult::already_complete;
      state->cancel_requested.store(true, std::memory_order_release);
      return CancelResult::requested;
    });

    std::unique_lock lock(mutex_);
    space_cv_.wait(lock, [&] {
      return stopping_ || queue_.size() + requests.size() <= capacity_;
    });
    if (stopping_) throw std::runtime_error("PageReader is shut down");
    for (const auto &request : requests) queue_.push_back({request, completion, batch});
    lock.unlock();
    work_cv_.notify_all();
    return handle;
  }

  void shutdown() noexcept override {
    {
      std::lock_guard lock(mutex_);
      if (stopping_) return;
      stopping_ = true;
    }
    work_cv_.notify_all();
    space_cv_.notify_all();
    for (auto &worker : workers_) {
      if (worker.joinable()) worker.join();
    }
    workers_.clear();
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

 private:
  void worker_loop() noexcept {
    for (;;) {
      Work work;
      {
        std::unique_lock lock(mutex_);
        work_cv_.wait(lock, [&] {
          return stopping_ || !queue_.empty();
        });
        if (queue_.empty()) {
          if (stopping_) return;
          continue;
        }
        work = std::move(queue_.front());
        queue_.pop_front();
      }
      space_cv_.notify_all();

      ReadResult result{.id = work.request.id};
      if (work.batch->cancel_requested.load(std::memory_order_acquire)) {
        result.status = ReadStatus::cancelled;
      } else {
        errno = 0;
        const auto bytes = ::pread(fd_,
                                   work.request.buffer.data(),
                                   work.request.buffer.size(),
                                   static_cast<off_t>(work.request.offset));
        if (bytes < 0) {
          result.status = ReadStatus::io_error;
          result.error = {errno, std::generic_category()};
        } else {
          result.bytes = static_cast<std::size_t>(bytes);
          if (work.batch->cancel_requested.load(std::memory_order_acquire))
            result.status = ReadStatus::cancelled;
          else if (Clock::now() > work.request.deadline)
            result.status = ReadStatus::timed_out;
          else
            result.status = result.bytes == work.request.buffer.size() ? ReadStatus::ok
                                                                       : ReadStatus::short_read;
        }
      }
      work.batch->remaining.fetch_sub(1, std::memory_order_acq_rel);
      work.completion.fn(work.completion.context, result);
    }
  }

  int fd_ = -1;
  ReadConstraints constraints_{};
  std::size_t capacity_ = 1;
  std::mutex mutex_;
  std::condition_variable work_cv_;
  std::condition_variable space_cv_;
  std::deque<Work> queue_;
  std::vector<std::thread> workers_;
  bool stopping_ = false;
};

}  // namespace alaya::storage::io
